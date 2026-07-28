// ui/history.ts — 씬 편집 Undo/Redo 스냅샷 스택 (UX_DESIGN §7 "되돌리기/다시하기", Phase 7)
//
// 전략: SceneEditor.serialize() 스냅샷 스택 + **전체 씬 재로드** 복원
// (core/scene-edit-types.ts 헤더의 설계 결정 — 정확성 우선). 부분 diff 적용 대신
// 검증된 SceneSpec 전체를 loadScene 경로(검증 → teardown → 클린 빌드)로 다시 빌드한다.
// 로봇 URDF도 다시 로드되지만 브라우저 HTTP 캐시로 충분히 빠르다 — EXPERIMENTS 기록.
//
// - noteChange(capture): 편집 통지(SceneEditor.onChange)마다 호출 — capture는 호출
//   "즉시" 실행되어 그 시점 상태를 고정하고, 트레일링 디바운스가 "연속 편집 burst"
//   (스크럽 드래그·기즈모 커밋 연타)를 최종 스냅샷 1장으로 합쳐 커밋한다.
//   (지연 캡처였다면 burst 도중 구조 변경이 끼어들 때 이전 상태가 유실된다 — 통합자는
//   add/remove/rename 같은 구조 변경 전후로 flushPending()을 호출해 경계를 세운다.)
// - undo()/redo(): pending을 먼저 flush해 마지막 burst도 되돌릴 수 있게 한 뒤 스택을
//   이동하고 deps.restore(spec)를 기다린다. 복원 실패(전환 busy/빌드 실패) 시 스택을
//   원위치로 되돌린다 — 스택과 실제 씬이 어긋나지 않는다.
// - cap(기본 50): 초과 시 가장 오래된 스냅샷을 버린다(그 이전으로는 되돌 수 없음).
// - undoStack의 마지막 원소가 항상 "현재 씬 상태"다(reset의 기준 스냅샷 포함) —
//   undo 가능 조건은 length ≥ 2.
//
// 계층 규칙 (CLAUDE.md §3): 이 모듈은 core/render를 모른다 — SceneSpec 타입만 안다.
// 씬 재로드·엔진 정지·토스트는 통합자(main.ts)가 deps.restore 안에서 수행한다.

import type { ControlSequence, SceneSpec } from '../schema/types';

/**
 * 히스토리 스냅샷 — **씬과 시퀀스가 한 덩어리다** (UX_AUDIT C-4).
 *
 * 구 구현은 `SceneSpec`만 저장해서 플로우 그래프 편집(노드 삭제·재정렬·복제·파라미터
 * 변경, `교체` 모드 재생성)이 전부 되돌릴 수 없었다. `Ctrl+Z`가 "때때로만 동작하는"
 * 것은 아예 없는 것보다 나쁘다 — 잘못된 안전감을 준다.
 *
 * 크리에이션 툴에서 "되돌릴 수 있다"는 확신은 기능이 아니라 **탐색의 전제조건**이다.
 * 되돌릴 수 없으면 사용자는 실험을 멈추고, 실험을 멈추면 시뮬레이터는 쓸모가 없다.
 */
export interface HistorySnapshot {
  readonly scene: SceneSpec;
  readonly sequence: ControlSequence | null;
}

// ── 상수 (매직넘버 금지 — CLAUDE.md §4) ─────────────────────────────

/** 스냅샷 스택 상한 (초과 시 가장 오래된 것부터 폐기) */
export const HISTORY_CAP_DEFAULT = 50;
/** 편집 burst 합침 디바운스 (트레일링) */
export const HISTORY_DEBOUNCE_MS_DEFAULT = 300;

// ── 공개 타입 ───────────────────────────────────────────────────────

export interface SceneHistoryDeps {
  /**
   * 스냅샷 복원 — 통합자가 loadScene(전체 클린 재빌드, 시퀀스 재검증 포함)으로
   * 구현한다. `restoreSceneFromHistory`가 이미 {scene, sequence} 봉투를 받으므로
   * 배선 비용은 작다. false 반환(전환 busy/빌드 실패) 시 이번 스택 이동은 취소된다.
   */
  restore(snapshot: HistorySnapshot): Promise<boolean>;
  /** 스택 상태 변화 통지 (↶/↷ 버튼 활성화 동기화용) */
  onStateChange?(canUndo: boolean, canRedo: boolean): void;
}

export interface SceneHistoryOptions {
  cap?: number;
  debounceMs?: number;
}

// ── SceneHistory ────────────────────────────────────────────────────

export class SceneHistory {
  private readonly deps: SceneHistoryDeps;
  private readonly cap: number;
  private readonly debounceMs: number;

  /** 마지막 원소 = 현재 씬 상태 스냅샷 (reset 기준 스냅샷 포함) */
  private undoStack: HistorySnapshot[] = [];
  private redoStack: HistorySnapshot[] = [];
  private pendingTimer: ReturnType<typeof setTimeout> | null = null;
  /** noteChange 시점에 "즉시" 캡처된 스냅샷 — 디바운스 발화/flush 시 커밋된다 */
  private pendingSnapshot: HistorySnapshot | null = null;
  private restoring = false;

  constructor(deps: SceneHistoryDeps, opts: SceneHistoryOptions = {}) {
    this.deps = deps;
    this.cap = opts.cap ?? HISTORY_CAP_DEFAULT;
    this.debounceMs = opts.debounceMs ?? HISTORY_DEBOUNCE_MS_DEFAULT;
  }

  get canUndo(): boolean {
    return this.undoStack.length >= 2 && !this.restoring;
  }

  get canRedo(): boolean {
    return this.redoStack.length > 0 && !this.restoring;
  }

  /** 복원(재로드) 진행 중 여부 — 통합자가 이중 트리거를 거를 때 참조 가능 */
  get isRestoring(): boolean {
    return this.restoring;
  }

  /**
   * 새 "논리 씬" 시작(부트/프리셋 전환/업로드) — 스택을 비우고 기준 스냅샷을 쌓는다.
   * 히스토리 복원(restore)이 유발한 재빌드에서는 호출하지 않는다 — 스택이 이어진다.
   */
  reset(initial: HistorySnapshot): void {
    this.cancelPending();
    this.undoStack = [structuredClone(initial)];
    this.redoStack = [];
    this.emitState();
  }

  /**
   * 편집 통지 — capture()를 **즉시** 실행해 이 시점 상태를 고정하고, 디바운스 발화
   * 시 최종 pending 스냅샷 1장을 커밋한다(redo 스택은 커밋 시 무효화). burst 안의
   * 이전 pending은 최신 것으로 대체된다(합침). 구조 변경(add/remove/rename)의 경계는
   * 통합자가 flushPending()으로 세운다 — 그 앞의 연속 burst가 별도 스냅샷으로 남는다.
   */
  noteChange(capture: () => HistorySnapshot): void {
    this.pendingSnapshot = structuredClone(capture());
    if (this.pendingTimer !== null) clearTimeout(this.pendingTimer);
    this.pendingTimer = setTimeout(() => {
      this.pendingTimer = null;
      this.flushPending();
    }, this.debounceMs);
  }

  /** pending 스냅샷을 즉시 커밋 (undo/redo 진입·구조 변경 경계 — burst 확정) */
  flushPending(): void {
    if (this.pendingTimer !== null) {
      clearTimeout(this.pendingTimer);
      this.pendingTimer = null;
    }
    const snapshot = this.pendingSnapshot;
    this.pendingSnapshot = null;
    if (snapshot === null) return;
    this.undoStack.push(snapshot);
    this.redoStack = []; // 새 편집은 redo 가지를 무효화한다 (선형 히스토리)
    while (this.undoStack.length > this.cap) this.undoStack.shift();
    this.emitState();
  }

  /**
   * pending 폐기 — 씬 teardown 시 통합자가 호출한다. 디바운스 타이머가 이미 해제된
   * 씬의 낡은 스냅샷을 뒤늦게 쌓는 것을 막는다.
   */
  cancelPending(): void {
    if (this.pendingTimer !== null) {
      clearTimeout(this.pendingTimer);
      this.pendingTimer = null;
    }
    this.pendingSnapshot = null;
  }

  /** 한 단계 되돌리기 — 복원 성공 여부 반환 (스택 이동은 성공 시에만 확정) */
  async undo(): Promise<boolean> {
    this.flushPending();
    if (this.restoring || this.undoStack.length < 2) return false;
    const popped = this.undoStack.pop();
    const target = this.undoStack[this.undoStack.length - 1];
    if (popped === undefined || target === undefined) return false; // 방어 (위 길이 가드로 도달 불가)
    this.restoring = true;
    this.emitState();
    let ok = false;
    try {
      ok = await this.deps.restore(structuredClone(target));
    } finally {
      if (ok) this.redoStack.push(popped);
      else this.undoStack.push(popped); // 복원 실패 — 스택 원위치 (씬과 어긋나지 않게)
      this.restoring = false;
      this.emitState();
    }
    return ok;
  }

  /** 한 단계 다시하기 — 복원 성공 여부 반환 */
  async redo(): Promise<boolean> {
    // 주의: pending이 있으면 flush가 redo 스택을 무효화한다 — "편집 후 redo"는
    // 정의상 불가능한 선형 히스토리 의미론이므로 올바른 동작이다.
    this.flushPending();
    if (this.restoring || this.redoStack.length === 0) return false;
    const target = this.redoStack[this.redoStack.length - 1];
    if (target === undefined) return false;
    this.restoring = true;
    this.emitState();
    let ok = false;
    try {
      ok = await this.deps.restore(structuredClone(target));
    } finally {
      if (ok) {
        this.redoStack.pop();
        this.undoStack.push(target);
        while (this.undoStack.length > this.cap) this.undoStack.shift();
      }
      this.restoring = false;
      this.emitState();
    }
    return ok;
  }

  private emitState(): void {
    this.deps.onStateChange?.(this.canUndo, this.canRedo);
  }
}
