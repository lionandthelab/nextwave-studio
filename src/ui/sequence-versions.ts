// ui/sequence-versions.ts — 시퀀스 버전 스택 (JSON 직접 편집의 안전망)
//
// ── 왜 Undo와 별도인가 ──────────────────────────────────────────────
//
// `ui/history.ts`의 Undo는 **씬+시퀀스 한 덩어리**를 되돌린다(전체 재로드). 강력하지만
// 세 가지를 못 한다:
//   1. **보이지 않는다.** Ctrl+Z를 몇 번 눌러야 그때로 가는지 알 수 없다.
//   2. **건너뛸 수 없다.** 30번 전 상태로 가려면 그 사이 씬 편집까지 전부 되돌아간다.
//   3. **시퀀스만 되돌릴 수 없다.** 씬은 그대로 두고 시퀀스만 이전으로 돌리는 조작이
//      JSON 직접 편집에서는 가장 자주 필요하다(텍스트를 망쳤을 때).
//
// 이 스택은 **시퀀스 축만** 이름표와 시각을 붙여 쌓고, 임의 버전으로 되돌린다.
// 되돌리기도 새 버전으로 append된다 — "되돌리기를 되돌릴" 수 있어야 실수의 값이 0이 된다.
// (파괴적 동작에는 되돌릴 경로가 있어야 한다 — CLAUDE.md §2.11.)
//
// ── 수명 ────────────────────────────────────────────────────────────
// **세션 한정(메모리)** 이다. 새로고침 후 복구는 IndexedDB 자동저장 초안이 담당하고
// (ui/document.ts), 영구 보존은 파일 저장(.workcell.json)·서버 저장이 담당한다.
// 축이 셋으로 갈리는 것을 피하려고 이 스택은 영속하지 않으며, UI가 그 한계를 명시한다.
//
// 계층 규칙 (CLAUDE.md §3): schema 타입(POJO)만 안다. DOM·core·render 비의존 —
// 전부 node 환경에서 단위 테스트된다.

import type { ControlSequence } from '../schema/types';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4) ─────────────────────────────

/** 버전 스택 상한 — 초과 시 가장 오래된 버전부터 폐기한다 */
export const SEQUENCE_VERSION_CAP_DEFAULT = 50;

// ── 공개 타입 ───────────────────────────────────────────────────────

export interface SequenceVersion {
  /** 1부터 단조 증가. 폐기된 뒤에도 재사용하지 않는다(목록의 안정 식별자) */
  readonly version: number;
  readonly atIso: string;
  /** 무엇이 바뀌었나 (한국어) — 자동 도출 또는 호출자 지정 */
  readonly labelKo: string;
  readonly stepCount: number;
  /** 그 시점 시퀀스의 깊은 복사본 (외부 변형으로 이력이 오염되지 않는다) */
  readonly sequence: ControlSequence;
}

// ── 변경 라벨 자동 도출 (순수) ──────────────────────────────────────

/** 두 시퀀스가 내용상 같은가 (키 순서까지 포함한 구조 비교 — 기록 중복 억제용) */
export function sequencesEqual(
  a: ControlSequence | null,
  b: ControlSequence | null,
): boolean {
  if (a === null || b === null) return a === b;
  return JSON.stringify(a) === JSON.stringify(b);
}

/**
 * 변경 라벨을 자동으로 붙인다. 편집 호출부(캔버스 드래그·인스펙터 폼·파사드)가
 * 저마다 라벨을 넘기게 하면 **한 곳만 빠뜨려도 이력에 '알 수 없음'이 남는다.**
 * 대신 이전/이후 시퀀스를 비교해 도출한다 — 호출부를 하나도 건드리지 않고,
 * 새 편집 경로가 생겨도 자동으로 이름이 붙는다.
 *
 * 우선순위: 노드 수 → 구성(kind) → 순서 → 활성 → 파라미터.
 */
export function describeSequenceChange(
  prev: ControlSequence | null,
  next: ControlSequence,
): string {
  if (prev === null) return '시퀀스 생성';

  const prevSteps = prev.steps;
  const nextSteps = next.steps;
  const delta = nextSteps.length - prevSteps.length;
  if (delta > 0) return `노드 ${delta}개 추가`;
  if (delta < 0) return `노드 ${-delta}개 삭제`;

  const prevKinds = prevSteps.map((s) => s.kind);
  const nextKinds = nextSteps.map((s) => s.kind);
  const sameOrder = prevKinds.every((k, i) => k === nextKinds[i]);
  if (!sameOrder) {
    // 같은 kind 묶음이 순서만 다르면 재정렬, 구성 자체가 다르면 교체
    const sorted = (arr: readonly string[]): string => [...arr].sort().join('|');
    return sorted(prevKinds) === sorted(nextKinds) ? '노드 재정렬' : '노드 교체';
  }

  const enabledChanged = prevSteps.some((s, i) => (s.enabled !== false) !== (nextSteps[i]?.enabled !== false));
  if (enabledChanged) return '노드 활성 상태 변경';

  if (prev.robot !== next.robot) return '대상 로봇 변경';
  if ((prev.loop === true) !== (next.loop === true)) return '반복 설정 변경';

  const noteChanged = prevSteps.some((s, i) => (s.note ?? '') !== (nextSteps[i]?.note ?? ''));
  if (noteChanged) return '노트 변경';

  return '파라미터 변경';
}

// ── 스택 ────────────────────────────────────────────────────────────

export interface SequenceVersionsOptions {
  /** 상한 (기본 SEQUENCE_VERSION_CAP_DEFAULT) */
  readonly cap?: number;
  /** 시각 주입 (테스트) — 기본 new Date().toISOString() */
  nowIso?(): string;
}

export interface RecordOptions {
  /** 라벨 직접 지정 (예: 'JSON 직접 편집'). 없으면 describeSequenceChange가 도출 */
  readonly labelKo?: string;
  readonly nowIso?: string;
}

/**
 * 시퀀스 버전 스택. 커밋마다 `record`가 불리고, 직전 버전과 **내용이 같으면 기록하지
 * 않는다**(no-op 커밋이 이력을 채워 진짜 변경을 밀어내지 않게).
 */
export class SequenceVersions {
  private readonly cap: number;
  private readonly clock: () => string;
  private readonly stack: SequenceVersion[] = [];
  private nextVersion = 1;

  constructor(opts: SequenceVersionsOptions = {}) {
    this.cap = opts.cap !== undefined && opts.cap > 0 ? Math.floor(opts.cap) : SEQUENCE_VERSION_CAP_DEFAULT;
    this.clock = opts.nowIso ?? ((): string => new Date().toISOString());
  }

  /**
   * 새 버전을 기록한다. 직전 버전과 내용이 같으면 **기록하지 않고 null**을 돌려준다.
   * 되돌리기로 과거와 같은 내용이 되어도 직전 버전과는 다르므로 새 버전이 쌓인다 —
   * append-only가 "되돌리기를 되돌릴 수 있다"는 성질을 만든다.
   */
  record(sequence: ControlSequence, opts: RecordOptions = {}): SequenceVersion | null {
    const previous = this.stack.length > 0 ? (this.stack[this.stack.length - 1] ?? null) : null;
    if (previous !== null && sequencesEqual(previous.sequence, sequence)) return null;

    const entry: SequenceVersion = {
      version: this.nextVersion,
      atIso: opts.nowIso ?? this.clock(),
      labelKo: opts.labelKo ?? describeSequenceChange(previous?.sequence ?? null, sequence),
      stepCount: sequence.steps.length,
      sequence: structuredClone(sequence),
    };
    this.nextVersion += 1;
    this.stack.push(entry);
    // 상한 초과 — 가장 오래된 것부터 버린다(그 이전으로는 되돌릴 수 없다)
    while (this.stack.length > this.cap) this.stack.shift();
    return entry;
  }

  /** 최신 우선 목록 (UI가 그대로 그린다) */
  list(): readonly SequenceVersion[] {
    return [...this.stack].reverse();
  }

  /** 특정 버전 (폐기됐거나 없으면 null) */
  get(version: number): SequenceVersion | null {
    return this.stack.find((v) => v.version === version) ?? null;
  }

  /** 현재(가장 최근) 버전 번호 — 비어 있으면 null */
  currentVersion(): number | null {
    return this.stack.length > 0 ? (this.stack[this.stack.length - 1]?.version ?? null) : null;
  }

  size(): number {
    return this.stack.length;
  }

  /** 씬 전환 등으로 이력이 무의미해질 때 (버전 번호도 1부터 다시) */
  clear(): void {
    this.stack.length = 0;
    this.nextVersion = 1;
  }
}
