// ui/orchestrator.ts — 실행 오케스트레이터 (Phase 10, UX_DESIGN §5, ROADMAP Phase 10)
//
// "JSON대로 시뮬레이터에 동작을 하나씩 요청"하는 실행 계층. core의 Engine/ControlPlayer/
// CollisionMonitor를 감싸 **노드 단위 상태와 컨트롤**을 UI에 노출한다 — 활성 노드(그래프)
// ↔ 로봇 동작(뷰포트) ↔ Timeline 커서가 항상 일치하도록 하나의 진실을 방출한다.
//
// ── 계층 규칙 (CLAUDE.md §3) ────────────────────────────────────────
// ui → {core, schema}. 이 모듈은 core의 공개 API(Engine/Player/Monitor의 좁은 인터페이스)
// 와 schema 타입만 참조한다 — Rapier/three 심볼은 절대 넘어오지 않는다(주입된 deps 인터
// 페이스로만 대화한다). Phase 8이 배선한 player.onStepChange → canvas.setStatuses를 이곳이
// 일급 오케스트레이션(노드 경계 제어·트라이페인 동기·충돌 연동·결정론적 재실행)으로 심화한다.
//
// ── 불변식 / 진실의 소유 ────────────────────────────────────────────
// - 시뮬 진실(물리 pose·시퀀스 커서)은 core가 소유한다. 오케스트레이터는 커서 통지를 관찰해
//   "표현 상태"(노드 상태 맵·활성 노드)만 파생·방출한다 — 커서를 임의로 되돌려 물리와
//   어긋나게 만들지 않는다(재실행은 resetScene를 통한 결정론적 되감기로만).
// - 오케스트레이터는 player를 직접 tick하지 않는다. player는 Engine preStep 훅이 구동하고
//   (ARCHITECTURE §5 ①), 이 모듈은 engine.play/pause/stop/setSpeed로 재생만 제어한다.
//
// ── 순수 로직 분리 (단위 테스트 대상) ───────────────────────────────
// 상태 맵 계산(computeStatusMap)·활성 노드 도출(activeNodeIdOf)·노드 경계 정지 판정
// (shouldPauseAtBoundary)은 순수 함수로 추출한다. Orchestrator는 이들을 주입된 deps 위에
// 얇게 감싼다 — 동일 이벤트 스크립트 → 동일 상태 맵 시퀀스(결정론, SIMULATION §6).

import type { NodeRunStatus } from './flow-graph/node-render';
import type { CollisionEvent } from '../schema/types';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4) ─────────────────────────────

/** 재실행 fast-forward 속도 (ENGINE_SPEED_OPTIONS 최대값 4× — engine.setSpeed 허용값). */
export const RUN_FROM_NODE_SPEED_MULT = 4;

/** 기본 재생 속도 배율 (setSpeed 전, runFromNode 복원 기준값). */
const DEFAULT_SPEED_MULT = 1;

// ── 순수 로직: 노드 상태 맵 ─────────────────────────────────────────

/** 체인 상의 한 노드 (step 인덱스 ↔ FlowGraph 노드 id ↔ 활성 여부). */
export interface NodeEntry {
  /** ControlSequence.steps[] 인덱스 (0-based, 체인 순서). */
  index: number;
  /** FlowGraph 노드 id ('n1','n2',… — fromSequence 안정 id). */
  nodeId: string;
  /** enabled:false면 실행에서 제외(순서 유지 — DATA_MODEL §6, Phase 8). */
  enabled: boolean;
}

/**
 * 노드 상태 맵 계산 (순수 — Phase 8 paintFlowStatuses 시맨틱을 일급화):
 * - activeIndex === null → 전부 'pending' (⏹ Stop/리셋 — 오류 오버레이 없음).
 * - activeIndex 유효값 → index < activeIndex: 'done' · index === activeIndex && enabled:
 *   'active' · 그 외: 'pending'. activeIndex ≥ 노드 수(시퀀스 끝)면 전부 'done'.
 * - erroredNodeIds는 activeIndex !== null일 때만 'error'로 덮어쓴다(오류 노드 보존 —
 *   "engine idle/player done: all done, or leave error nodes").
 *
 * 비활성 노드가 active 인덱스에 놓이면 'active'가 아니라 'pending'이다(스킵은 같은 tick에
 * 통과 — "없는 것처럼", player 계약). index < activeIndex인 비활성 노드는 'done'이다.
 */
export function computeStatusMap(
  nodes: readonly NodeEntry[],
  activeIndex: number | null,
  erroredNodeIds: ReadonlySet<string>,
): Record<string, NodeRunStatus> {
  const map: Record<string, NodeRunStatus> = {};
  for (const node of nodes) {
    let status: NodeRunStatus;
    if (activeIndex === null) {
      status = 'pending';
    } else if (node.index < activeIndex) {
      status = 'done';
    } else if (node.index === activeIndex && node.enabled) {
      status = 'active';
    } else {
      status = 'pending';
    }
    if (activeIndex !== null && erroredNodeIds.has(node.nodeId)) {
      status = 'error';
    }
    map[node.nodeId] = status;
  }
  return map;
}

/**
 * 활성 노드 id 도출 (순수): activeIndex의 enabled 노드. 비활성 노드/범위 밖/리셋(null)
 * 이면 null. 오류로 색칠돼도(오버레이) 활성 노드는 실행 중인 그 노드를 계속 가리킨다 —
 * 상태 색이 아니라 커서 인덱스로 판정한다(뷰포트 배지/그래프 선택이 흔들리지 않게).
 */
export function activeNodeIdOf(nodes: readonly NodeEntry[], activeIndex: number | null): string | null {
  if (activeIndex === null) return null;
  for (const node of nodes) {
    if (node.index === activeIndex && node.enabled) return node.nodeId;
  }
  return null;
}

// ── 순수 로직: 노드 경계 정지 판정 ──────────────────────────────────

/**
 * 정지 정책 (노드 경계 러너 — 다음 onStepChange에서 무엇을 할지):
 * - 'none': 연속 재생(정지하지 않음).
 * - 'atNext': 다음 노드 경계(어떤 onStepChange든)에서 정지 (⏸ pauseAtNextNode).
 * - 'stepOne': 시작 인덱스와 달라진 첫 경계에서 정지 (⏭ Step = 노드 1개).
 * - 'runTo': index ≥ target인 첫 경계에서 정지 + 속도 복원 (재실행 fast-forward).
 */
export type PausePolicy =
  | { kind: 'none' }
  | { kind: 'atNext' }
  | { kind: 'stepOne'; fromIndex: number }
  | { kind: 'runTo'; index: number; restoreSpeed: number };

/**
 * 경계 정지 판정 (순수): onStepChange(incomingIndex)가 왔을 때 재생을 멈춰야 하는가.
 * stepOne은 "인덱스가 바뀌었는가"(index !== from) — 노드가 <1프레임에 완료돼도 다음
 * 경계에서 멈춘다. runTo는 target에 도달/추월(index ≥ target)한 첫 경계.
 */
export function shouldPauseAtBoundary(policy: PausePolicy, incomingIndex: number): boolean {
  switch (policy.kind) {
    case 'none':
      return false;
    case 'atNext':
      return true;
    case 'stepOne':
      return incomingIndex !== policy.fromIndex;
    case 'runTo':
      return incomingIndex >= policy.index;
    default: {
      const _exhaustive: never = policy;
      return _exhaustive;
    }
  }
}

// ── 주입 계약 (core 공개 표면 — Rapier/three 비노출) ─────────────────

export interface OrchestratorEngine {
  play(): void;
  pause(): void;
  stop(): void;
  // 참고: 노드-경계 러너는 engine.play()+PausePolicy로 구동한다 — 물리-tick 프레임 스텝
  // (engine.stepOnce)은 Orchestrator가 쓰지 않으므로 주입 계약에서 제외한다(표면 최소화).
  setSpeed(mult: number): void;
  readonly state: 'idle' | 'playing' | 'paused';
  readonly simTimeSec: number;
  onTick(fn: (info: { state: string; simTimeSec: number }) => void): () => void;
}

export interface OrchestratorPlayer {
  load(seq: unknown): void;
  reset(): void;
  readonly status: 'idle' | 'running' | 'done';
  readonly currentStepIndex: number;
  readonly stepCount: number;
  onStepChange(fn: (index: number, step: unknown) => void): () => void;
}

export interface OrchestratorMonitor {
  subscribe(fn: (e: CollisionEvent) => void): () => void;
}

export interface OrchestratorDeps {
  engine: OrchestratorEngine;
  player: OrchestratorPlayer;
  monitor: OrchestratorMonitor;
  /** 노드 상태 맵 방출 → canvas.setStatuses + Timeline (트라이페인 동기 ①). */
  onNodeStatus(map: Record<string, NodeRunStatus>): void;
  /** 활성 노드 방출 → 그래프 선택 + 뷰포트 배지 (트라이페인 동기 ②). null = 없음. */
  onActiveNode(nodeId: string | null): void;
  /** 결정론적 되감기: sceneHandle.reset + player.reset + monitor.clear (통합자 제공). */
  resetScene(): void;
  /** FlowGraph 노드 id ↔ steps[] 인덱스 (현재 그래프 기준, 통합자 제공). */
  nodeIdByStepIndex(index: number): string | null;
  stepIndexByNodeId(id: string): number | null;
  /** 활성(enabled) step 인덱스 목록 (비활성 노드 제외 — Phase 8 enabled). */
  enabledStepIndices(): number[];
  /**
   * "예기치 않은" 충돌 판정(옵션): 통합자가 결정한다(예: 현재 waitForCollision 쌍이 아닌
   * robot×env). 미제공/false면 오류로 승격하지 않는다(정상 접촉은 소음).
   */
  unexpectedCollision?(e: CollisionEvent): boolean;
}

// ── 오류 이벤트 (UI 표시용) ─────────────────────────────────────────

export interface OrchestratorErrorEvent {
  /** 오류로 표시된 노드 id. */
  nodeId: string;
  /** 'collision': 예기치 않은 충돌 · 'external': 통합자 라우팅(waitForCollision 타임아웃 등). */
  reason: 'collision' | 'external';
  /** collision 원인일 때의 충돌 이벤트(가능할 때). */
  collision?: CollisionEvent;
}

// ── Orchestrator ────────────────────────────────────────────────────

/**
 * 노드 단위 실행 오케스트레이터. player 커서 통지를 상태 맵/활성 노드로 파생하고, 재생을
 * 노드 경계 단위로 제어한다(연속/노드경계정지/1노드/재실행 fast-forward). 충돌 인지 정지·
 * 오류 마킹을 관장한다.
 *
 * 재실행(runFromNode)의 결정론: per-node 씬 스냅샷이 없으므로 **항상 처음으로 되감아**
 * (resetScene) 재생한다 — 같은 시퀀스는 같은 setJoints 로그를 낸다(ControlPlayer.load/reset
 * 계약, SIMULATION §6). 뒤쪽 노드가 목표면 앞 노드를 최고 속도로 재생 후 목표 경계에서
 * 정지한다. 한계(정직한 MVP): 목표 노드의 첫 물리 tick이 같은 프레임 안에서 실행될 수
 * 있어(엔진 프레임 루프는 pause를 tick 사이에 재검사하지 않음) 정확히 "실행 직전"이 아니라
 * "경계 ±1 tick"에서 멈춘다. 정밀 재개점은 per-node 스냅샷 도입 시 개선한다(백로그).
 * ⏭ Step의 <1프레임 노드 오버스텝도 같은 계열의 한계다 — 항상 "다음 경계"에서 멈춘다.
 */
export class Orchestrator {
  private readonly deps: OrchestratorDeps;
  private readonly offStepChange: () => void;
  private readonly offCollision: () => void;
  private readonly errorListeners = new Set<(e: OrchestratorErrorEvent) => void>();

  /** 이번 재생 런에서 오류로 마킹된 노드(스티키 — stop/runFromNode에서 리셋). */
  private readonly erroredNodeIds = new Set<string>();

  private pausePolicy: PausePolicy = { kind: 'none' };
  /** 마지막 커서 인덱스(표현 상태). null = 리셋/idle(전부 pending). */
  private activeIndex: number | null = null;
  private activeNode: string | null = null;
  private statusSnapshot: Record<string, NodeRunStatus> = {};
  private autoPause = false;
  /** setSpeed로 마지막 지정한 배율 — runFromNode fast-forward 후 복원 기준. */
  private speedMult = DEFAULT_SPEED_MULT;
  /** resetScene가 유발하는 player.reset 커서 통지를 무시하는 가드(명시 상태로 덮어씀). */
  private resetting = false;

  constructor(deps: OrchestratorDeps) {
    this.deps = deps;
    this.offStepChange = deps.player.onStepChange((index) => {
      this.handleStepChange(index);
    });
    this.offCollision = deps.monitor.subscribe((e) => {
      this.handleCollision(e);
    });
  }

  // ── 읽기 표면 (표현 상태 스냅샷) ──────────────────────────────────

  /** 현재 노드 상태 맵 스냅샷(사본 — 외부 변형 불가). */
  get statuses(): Readonly<Record<string, NodeRunStatus>> {
    return { ...this.statusSnapshot };
  }

  /** 현재 활성 노드 id(실행 중), 없으면 null. */
  get activeNodeId(): string | null {
    return this.activeNode;
  }

  get autoPauseOnCollision(): boolean {
    return this.autoPause;
  }

  get engineState(): 'idle' | 'playing' | 'paused' {
    return this.deps.engine.state;
  }

  get simTimeSec(): number {
    return this.deps.engine.simTimeSec;
  }

  /** rAF tick 통지 패스스루(통합자가 Timeline/뷰포트 시계에 잇는다). */
  onTick(fn: (info: { state: string; simTimeSec: number }) => void): () => void {
    return this.deps.engine.onTick(fn);
  }

  /** 오류 이벤트 구독 (UI가 토스트/강조로 표시). 반환값은 해제 함수. */
  onError(fn: (e: OrchestratorErrorEvent) => void): () => void {
    this.errorListeners.add(fn);
    return () => {
      this.errorListeners.delete(fn);
    };
  }

  // ── 재생 컨트롤 (§5) ──────────────────────────────────────────────

  /** ▶ 연속 재생(정지 정책 해제 후 엔진 재생). 오류 마킹은 보존(재개). */
  play(): void {
    this.pausePolicy = { kind: 'none' };
    this.deps.engine.play();
  }

  /** ⏸ 즉시 일시정지. */
  pause(): void {
    this.pausePolicy = { kind: 'none' };
    this.deps.engine.pause();
  }

  /**
   * ⏹ 정지 = 결정론적 재생 준비: 엔진 정지 → resetScene(씬 pose·player 커서·충돌 이력)
   * → 오류/상태 전부 리셋(전부 pending, 활성 null). 다음 ▶ Play는 처음부터 재생한다.
   */
  stop(): void {
    this.pausePolicy = { kind: 'none' };
    this.erroredNodeIds.clear();
    this.deps.engine.stop();
    this.withReset(() => {
      this.deps.resetScene();
    });
    this.activeIndex = null;
    this.recompute();
  }

  /**
   * ⏭ Step = 노드 정확히 1개 전진(물리 1 tick이 아니라). 현재 인덱스를 기록하고 재생을
   * 시작해, 인덱스가 달라진 첫 경계에서 정지한다(노드가 <1프레임에 끝나도 다음 경계에서
   * 멈춘다). 시퀀스가 없거나 이미 done이면 no-op.
   */
  step(): void {
    if (this.deps.player.stepCount === 0) return;
    if (this.deps.player.status === 'done') return;
    this.pausePolicy = { kind: 'stepOne', fromIndex: this.deps.player.currentStepIndex };
    this.deps.engine.play();
  }

  /**
   * 다음 노드 경계에서 일시정지하도록 예약(⏸ 노드 경계 정지 — §5). 즉시 멈추지 않고
   * 다음 onStepChange에서 engine.pause()한다. 연속 재생 중 호출을 전제한다(정지/idle에서
   * 호출하면 다음 재생·경계까지 대기).
   */
  pauseAtNextNode(): void {
    this.pausePolicy = { kind: 'atNext' };
  }

  /** 재생 속도 배율 패스스루(engine이 허용값 검증). runFromNode 복원 기준으로도 기록. */
  setSpeed(mult: number): void {
    this.speedMult = mult;
    this.deps.engine.setSpeed(mult);
  }

  /** 충돌 인지 자동 일시정지 토글(§5 "예기치 않은 충돌 시 자동 ⏸"). */
  setAutoPauseOnCollision(enabled: boolean): void {
    this.autoPause = enabled;
  }

  /**
   * 노드/타임라인 마커 클릭 → 그 노드부터 재실행(§5). 결정론: 처음으로 되감아 재생한다
   * (클래스 주석의 재실행 결정론 참조). 목표 앞에 실행할 활성 노드가 없으면 되감은 채
   * 정지(사용자 ▶ Play가 목표부터 재생), 있으면 최고 속도로 앞 노드를 재생 후 목표 경계
   * 에서 정지한다. 알 수 없는 노드 id는 no-op(통합자가 유효 id만 전달).
   */
  runFromNode(nodeId: string): void {
    const targetIndex = this.deps.stepIndexByNodeId(nodeId);
    if (targetIndex === null) return;

    // 1) 결정론적 되감기 (per-node 스냅샷 부재 — 항상 처음부터).
    this.pausePolicy = { kind: 'none' };
    this.erroredNodeIds.clear();
    this.deps.engine.stop();
    this.withReset(() => {
      this.deps.resetScene(); // sceneHandle.reset + player.reset(커서 0) + monitor.clear
    });
    this.activeIndex = 0;

    // 2) 목표 앞에 실행할(enabled) 노드가 없으면 되감은 상태로 둔다(엔진 idle).
    const hasEnabledPrior = this.deps.enabledStepIndices().some((i) => i < targetIndex);
    if (!hasEnabledPrior) {
      this.recompute();
      return;
    }

    // 3) 앞 노드를 최고 속도로 재생 후 목표 경계에서 정지 + 속도 복원.
    this.recompute();
    this.pausePolicy = { kind: 'runTo', index: targetIndex, restoreSpeed: this.speedMult };
    this.deps.engine.setSpeed(RUN_FROM_NODE_SPEED_MULT);
    this.deps.engine.play();
  }

  /**
   * 노드를 오류로 마킹(통합자 라우팅 — waitForCollision 타임아웃 경고 등). 상태 맵을 다시
   * 그리고 오류 이벤트를 방출한다. 오류는 이번 런 동안 스티키(stop/runFromNode에서 해제).
   */
  markError(nodeId: string): void {
    this.registerError(nodeId, 'external');
  }

  /** 현재 player 상태로 상태 맵을 강제 재동기화(통합자 초기 페인트/재정렬 후). */
  refresh(): void {
    this.recompute();
  }

  /**
   * 그래프 편집으로 재생이 unarm될 때(시퀀스 편집 정책 — 재생 정지 후 처음부터)
   * 런 표현만 초기화한다: player 커서를 되감고(resetting 가드로 통지 무시) 상태 맵을
   * 전부 pending으로 되돌린다. **엔진(씬 물리)·씬 pose는 건드리지 않는다** — stop()과
   * 달리 결정론적 되감기가 아니라 "편집 후 다음 ▶ Play가 처음부터"를 위한 표현 리셋이다.
   */
  resetForEdit(): void {
    this.pausePolicy = { kind: 'none' };
    this.erroredNodeIds.clear();
    this.withReset(() => {
      this.deps.player.reset();
    });
    this.activeIndex = null;
    this.recompute();
  }

  /** 구독 해제 (씬 teardown). */
  dispose(): void {
    this.offStepChange();
    this.offCollision();
    this.errorListeners.clear();
  }

  // ── 내부 ──────────────────────────────────────────────────────────

  /** player 커서 통지 → 상태 재동기 + 경계 정지 정책 적용. reset 유발 통지는 무시한다. */
  private handleStepChange(index: number): void {
    if (this.resetting) return;
    this.activeIndex = index;
    this.recompute(); // 상태 맵은 새 인덱스를 즉시 반영(정지 여부와 무관)
    this.applyPausePolicy(index);
  }

  /** 정책이 경계 정지를 지시하면 엔진 일시정지(runTo는 속도 복원 후). */
  private applyPausePolicy(index: number): void {
    const policy = this.pausePolicy;
    if (!shouldPauseAtBoundary(policy, index)) return;
    if (policy.kind === 'runTo') this.deps.engine.setSpeed(policy.restoreSpeed);
    this.pausePolicy = { kind: 'none' };
    this.deps.engine.pause();
  }

  /** 충돌 → 예기치 않으면 활성 노드 오류 + (옵션) 자동 일시정지. 재생 중에만 반응. */
  private handleCollision(e: CollisionEvent): void {
    if (this.deps.engine.state !== 'playing') return;
    if (!(this.deps.unexpectedCollision?.(e) ?? false)) return;
    const nodeId = this.activeNode;
    if (nodeId === null) return;
    this.registerError(nodeId, 'collision', e);
    if (this.autoPause) {
      this.pausePolicy = { kind: 'none' };
      this.deps.engine.pause();
    }
  }

  private registerError(
    nodeId: string,
    reason: 'collision' | 'external',
    collision?: CollisionEvent,
  ): void {
    this.erroredNodeIds.add(nodeId);
    this.recompute();
    this.emitError({ nodeId, reason, ...(collision ? { collision } : {}) });
  }

  private emitError(e: OrchestratorErrorEvent): void {
    for (const listener of [...this.errorListeners]) listener(e);
  }

  /** deps에서 현재 노드 엔트리 목록을 구성(모든 step 인덱스 + enabled 플래그). */
  private buildNodeEntries(): NodeEntry[] {
    const count = this.deps.player.stepCount;
    const enabled = new Set(this.deps.enabledStepIndices());
    const entries: NodeEntry[] = [];
    for (let i = 0; i < count; i += 1) {
      const nodeId = this.deps.nodeIdByStepIndex(i);
      if (nodeId === null) continue; // 1:1 매핑 전제 — 방어적으로 건너뜀
      entries.push({ index: i, nodeId, enabled: enabled.has(i) });
    }
    return entries;
  }

  /** 상태 맵 + 활성 노드 재계산 후 방출(활성 노드는 변경 시에만 통지). */
  private recompute(): void {
    const entries = this.buildNodeEntries();
    const map = computeStatusMap(entries, this.activeIndex, this.erroredNodeIds);
    this.statusSnapshot = map;
    this.deps.onNodeStatus({ ...map });
    const nextActive = activeNodeIdOf(entries, this.activeIndex);
    if (nextActive !== this.activeNode) {
      this.activeNode = nextActive;
      this.deps.onActiveNode(nextActive);
    }
  }

  private withReset(fn: () => void): void {
    this.resetting = true;
    try {
      fn();
    } finally {
      this.resetting = false;
    }
  }
}
