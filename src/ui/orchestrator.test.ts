// ui/orchestrator.test.ts — 실행 오케스트레이터 순수 로직 단위 테스트 (Phase 10)
//
// 물리/렌더/DOM 없이 완결적으로 검증한다 (CLAUDE.md §4 "부수효과 격리"):
// - 목 engine/player/monitor는 호출을 기록하는 순수 객체다.
// - player.reset()은 실제 ControlPlayer처럼 커서 0으로 되감으며 onStepChange(0)을 방출한다
//   (resetting 가드 검증). monitor는 _emit로 충돌을 주입한다.
// 규범: docs/UX_DESIGN.md §5 (실행 오케스트레이션), ROADMAP Phase 10.

import { describe, expect, it, vi } from 'vitest';
import {
  Orchestrator,
  RUN_FROM_NODE_SPEED_MULT,
  activeNodeIdOf,
  computeStatusMap,
  shouldPauseAtBoundary,
} from './orchestrator';
import type {
  NodeEntry,
  OrchestratorDeps,
  OrchestratorEngine,
  OrchestratorErrorEvent,
  OrchestratorMonitor,
  OrchestratorPlayer,
  PausePolicy,
} from './orchestrator';
import type { NodeRunStatus } from './flow-graph/node-render';
import type { CollisionEvent } from '../schema/types';

// ── 목 engine/player/monitor ────────────────────────────────────────

interface MockEngine extends OrchestratorEngine {
  calls: string[];
  speeds: number[];
  emitTick(info: { state: string; simTimeSec: number }): void;
  setStateForTest(s: 'idle' | 'playing' | 'paused'): void;
}

function makeMockEngine(): MockEngine {
  let state: 'idle' | 'playing' | 'paused' = 'idle';
  const calls: string[] = [];
  const speeds: number[] = [];
  const tickListeners = new Set<(info: { state: string; simTimeSec: number }) => void>();
  return {
    calls,
    speeds,
    get state() {
      return state;
    },
    get simTimeSec() {
      return 0;
    },
    play() {
      calls.push('play');
      state = 'playing';
    },
    pause() {
      calls.push('pause');
      state = 'paused';
    },
    stop() {
      calls.push('stop');
      state = 'idle';
    },
    setSpeed(m: number) {
      calls.push(`setSpeed:${m}`);
      speeds.push(m);
    },
    onTick(fn) {
      tickListeners.add(fn);
      return () => {
        tickListeners.delete(fn);
      };
    },
    emitTick(info) {
      for (const l of [...tickListeners]) l(info);
    },
    setStateForTest(s) {
      state = s;
    },
  };
}

interface MockPlayer extends OrchestratorPlayer {
  emitStep(index: number): void;
  setCursorForTest(index: number): void;
  setStatusForTest(s: 'idle' | 'running' | 'done'): void;
}

function makeMockPlayer(stepCount: number): MockPlayer {
  let cursor = 0;
  let status: 'idle' | 'running' | 'done' = 'idle';
  const listeners = new Set<(index: number, step: unknown) => void>();
  const emit = (index: number): void => {
    cursor = index;
    if (index >= stepCount) status = 'done';
    for (const l of [...listeners]) l(index, null);
  };
  return {
    get status() {
      return status;
    },
    get currentStepIndex() {
      return cursor;
    },
    get stepCount() {
      return stepCount;
    },
    load() {
      status = 'running';
      emit(0);
    },
    reset() {
      status = 'running';
      emit(0); // 실제 ControlPlayer.reset처럼 커서 0으로 되감으며 통지
    },
    onStepChange(fn) {
      listeners.add(fn);
      return () => {
        listeners.delete(fn);
      };
    },
    emitStep(index) {
      emit(index);
    },
    setCursorForTest(index) {
      cursor = index;
    },
    setStatusForTest(s) {
      status = s;
    },
  };
}

interface MockMonitor extends OrchestratorMonitor {
  emit(e: CollisionEvent): void;
}

function makeMockMonitor(): MockMonitor {
  const listeners = new Set<(e: CollisionEvent) => void>();
  return {
    subscribe(fn) {
      listeners.add(fn);
      return () => {
        listeners.delete(fn);
      };
    },
    emit(e) {
      for (const l of [...listeners]) l(e);
    },
  };
}

// ── 하네스 ──────────────────────────────────────────────────────────

interface HarnessOptions {
  stepCount?: number;
  /** enabled step 인덱스(기본: 전부). */
  enabled?: number[];
  unexpectedCollision?: (e: CollisionEvent) => boolean;
}

function makeHarness(opts: HarnessOptions = {}) {
  const stepCount = opts.stepCount ?? 5;
  const nodeIds = Array.from({ length: stepCount }, (_, i) => `n${i + 1}`);
  const enabled = opts.enabled ?? nodeIds.map((_, i) => i);

  const engine = makeMockEngine();
  const player = makeMockPlayer(stepCount);
  const monitor = makeMockMonitor();

  const statusMaps: Record<string, NodeRunStatus>[] = [];
  const activeNodes: (string | null)[] = [];
  const resetScene = vi.fn(() => {
    // 실제 resetScene 계약: sceneHandle.reset + player.reset(+ monitor.clear).
    player.reset();
  });

  const deps: OrchestratorDeps = {
    engine,
    player,
    monitor,
    onNodeStatus: (m) => statusMaps.push(m),
    onActiveNode: (id) => activeNodes.push(id),
    resetScene,
    nodeIdByStepIndex: (i) => nodeIds[i] ?? null,
    stepIndexByNodeId: (id) => {
      const i = nodeIds.indexOf(id);
      return i >= 0 ? i : null;
    },
    enabledStepIndices: () => [...enabled],
    ...(opts.unexpectedCollision ? { unexpectedCollision: opts.unexpectedCollision } : {}),
  };

  const orch = new Orchestrator(deps);
  const errors: OrchestratorErrorEvent[] = [];
  orch.onError((e) => errors.push(e));

  return { orch, engine, player, monitor, statusMaps, activeNodes, resetScene, errors, nodeIds };
}

const collisionEvent = (a: string, b: string): CollisionEvent => ({
  timeSec: 1,
  a,
  b,
  phase: 'start',
  kind: 'contact',
});

const lastOf = <T,>(arr: readonly T[]): T => {
  const v = arr[arr.length - 1];
  if (v === undefined) throw new Error('빈 배열');
  return v;
};

// ── 순수 함수 ───────────────────────────────────────────────────────

describe('computeStatusMap (pure)', () => {
  const nodes: NodeEntry[] = [
    { index: 0, nodeId: 'n1', enabled: true },
    { index: 1, nodeId: 'n2', enabled: true },
    { index: 2, nodeId: 'n3', enabled: true },
    { index: 3, nodeId: 'n4', enabled: true },
  ];
  const noErrors = new Set<string>();

  it('activeIndex 앞은 done · 현재는 active · 뒤는 pending', () => {
    expect(computeStatusMap(nodes, 2, noErrors)).toEqual({
      n1: 'done',
      n2: 'done',
      n3: 'active',
      n4: 'pending',
    });
  });

  it('activeIndex === null → 전부 pending (오류 오버레이 없음)', () => {
    expect(computeStatusMap(nodes, null, new Set(['n2']))).toEqual({
      n1: 'pending',
      n2: 'pending',
      n3: 'pending',
      n4: 'pending',
    });
  });

  it('시퀀스 끝(activeIndex ≥ 노드 수) → 전부 done', () => {
    expect(computeStatusMap(nodes, 4, noErrors)).toEqual({
      n1: 'done',
      n2: 'done',
      n3: 'done',
      n4: 'done',
    });
  });

  it('비활성 노드가 active 인덱스면 active가 아니라 pending', () => {
    const withDisabled: NodeEntry[] = [
      { index: 0, nodeId: 'n1', enabled: true },
      { index: 1, nodeId: 'n2', enabled: false },
      { index: 2, nodeId: 'n3', enabled: true },
    ];
    expect(computeStatusMap(withDisabled, 1, noErrors)).toEqual({
      n1: 'done',
      n2: 'pending',
      n3: 'pending',
    });
  });

  it('erroredNodeIds는 activeIndex 유효 시 error로 덮어쓴다(오류 보존)', () => {
    expect(computeStatusMap(nodes, 4, new Set(['n2']))).toEqual({
      n1: 'done',
      n2: 'error',
      n3: 'done',
      n4: 'done',
    });
  });
});

describe('activeNodeIdOf (pure)', () => {
  const nodes: NodeEntry[] = [
    { index: 0, nodeId: 'n1', enabled: true },
    { index: 1, nodeId: 'n2', enabled: false },
    { index: 2, nodeId: 'n3', enabled: true },
  ];
  it('활성 인덱스의 enabled 노드 id', () => {
    expect(activeNodeIdOf(nodes, 2)).toBe('n3');
  });
  it('비활성 노드 인덱스 → null', () => {
    expect(activeNodeIdOf(nodes, 1)).toBeNull();
  });
  it('null/범위 밖 → null', () => {
    expect(activeNodeIdOf(nodes, null)).toBeNull();
    expect(activeNodeIdOf(nodes, 9)).toBeNull();
  });
});

describe('shouldPauseAtBoundary (pure)', () => {
  it('none은 절대 멈추지 않는다', () => {
    expect(shouldPauseAtBoundary({ kind: 'none' }, 3)).toBe(false);
  });
  it('atNext는 항상 멈춘다', () => {
    expect(shouldPauseAtBoundary({ kind: 'atNext' }, 3)).toBe(true);
  });
  it('stepOne은 인덱스가 달라졌을 때만', () => {
    const p: PausePolicy = { kind: 'stepOne', fromIndex: 2 };
    expect(shouldPauseAtBoundary(p, 2)).toBe(false);
    expect(shouldPauseAtBoundary(p, 3)).toBe(true);
    expect(shouldPauseAtBoundary(p, 1)).toBe(true);
  });
  it('runTo는 target 도달/추월 시', () => {
    const p: PausePolicy = { kind: 'runTo', index: 3, restoreSpeed: 1 };
    expect(shouldPauseAtBoundary(p, 2)).toBe(false);
    expect(shouldPauseAtBoundary(p, 3)).toBe(true);
    expect(shouldPauseAtBoundary(p, 4)).toBe(true);
  });
});

// ── onStepChange → 상태 맵 ──────────────────────────────────────────

describe('Orchestrator — onStepChange 상태 동기', () => {
  it('커서 통지가 done/active/pending 맵을 방출하고 활성 노드를 통지한다', () => {
    const h = makeHarness({ stepCount: 5 });
    h.player.emitStep(2);
    expect(lastOf(h.statusMaps)).toEqual({
      n1: 'done',
      n2: 'done',
      n3: 'active',
      n4: 'pending',
      n5: 'pending',
    });
    expect(lastOf(h.activeNodes)).toBe('n3');
  });

  it('비활성 노드는 active로 표시하지 않는다(활성 노드 null)', () => {
    // index 2(n3) 비활성
    const h = makeHarness({ stepCount: 5, enabled: [0, 1, 3, 4] });
    h.player.emitStep(2);
    const map = lastOf(h.statusMaps);
    expect(map['n3']).toBe('pending');
    // 활성 노드 없음 — onActiveNode는 변경 시에만 통지하므로 getter로 확인(초기 null 유지)
    expect(h.orch.activeNodeId).toBeNull();
  });

  it('비활성 노드도 커서가 지나가면 done이 된다', () => {
    const h = makeHarness({ stepCount: 5, enabled: [0, 1, 3, 4] });
    h.player.emitStep(3);
    const map = lastOf(h.statusMaps);
    expect(map['n3']).toBe('done'); // index 2 < 3
    expect(map['n4']).toBe('active');
    expect(lastOf(h.activeNodes)).toBe('n4');
  });

  it('시퀀스 끝 통지(index = stepCount) → 전부 done + 활성 null', () => {
    const h = makeHarness({ stepCount: 3 });
    h.player.emitStep(1);
    expect(lastOf(h.activeNodes)).toBe('n2');
    h.player.emitStep(3); // handleSequenceEnd 통지
    expect(lastOf(h.statusMaps)).toEqual({ n1: 'done', n2: 'done', n3: 'done' });
    expect(lastOf(h.activeNodes)).toBeNull();
  });
});

// ── stop ────────────────────────────────────────────────────────────

describe('Orchestrator — stop', () => {
  it('엔진 정지 + resetScene + 전부 pending + 활성 null', () => {
    const h = makeHarness({ stepCount: 4 });
    h.player.emitStep(2);
    expect(lastOf(h.activeNodes)).toBe('n3');

    h.orch.stop();

    expect(h.engine.calls).toContain('stop');
    expect(h.resetScene).toHaveBeenCalledTimes(1);
    expect(lastOf(h.statusMaps)).toEqual({
      n1: 'pending',
      n2: 'pending',
      n3: 'pending',
      n4: 'pending',
    });
    expect(lastOf(h.activeNodes)).toBeNull();
  });

  it('resetScene가 유발한 player.reset 커서 통지는 무시된다(중간 active 방출 없음)', () => {
    const h = makeHarness({ stepCount: 4 });
    h.player.emitStep(2);
    const before = h.statusMaps.length;
    h.orch.stop();
    // stop 이후 방출된 맵은 정확히 1개(명시적 recompute) — reset 통지가 별도 맵을 만들지 않음
    expect(h.statusMaps.length).toBe(before + 1);
  });

  it('stop은 오류 마킹을 지운다', () => {
    const h = makeHarness({ stepCount: 4 });
    h.player.emitStep(1);
    h.orch.markError('n2');
    expect(lastOf(h.statusMaps)['n2']).toBe('error');
    h.orch.stop();
    // 새 런에서 커서가 다시 그 노드에 와도 error가 아니다
    h.player.emitStep(2);
    expect(lastOf(h.statusMaps)['n2']).toBe('done');
  });
});

// ── pauseAtNextNode / step ──────────────────────────────────────────

describe('Orchestrator — 노드 경계 정지', () => {
  it('pauseAtNextNode는 즉시가 아니라 다음 경계에서 pause한다', () => {
    const h = makeHarness({ stepCount: 5 });
    h.orch.play();
    h.player.emitStep(1);
    const pauseCountBefore = h.engine.calls.filter((c) => c === 'pause').length;

    h.orch.pauseAtNextNode();
    expect(h.engine.calls.filter((c) => c === 'pause').length).toBe(pauseCountBefore); // 즉시 아님

    h.player.emitStep(2); // 다음 경계
    expect(h.engine.calls.filter((c) => c === 'pause').length).toBe(pauseCountBefore + 1);
  });

  it('step은 엔진을 재생하고 노드 1개 뒤 경계에서 정지한다', () => {
    const h = makeHarness({ stepCount: 5 });
    h.player.setCursorForTest(1);
    h.player.setStatusForTest('running');

    h.orch.step();
    expect(h.engine.calls).toContain('play');
    expect(h.engine.calls).not.toContain('pause'); // 아직

    h.player.emitStep(2); // 노드 경계 크로스
    expect(h.engine.calls).toContain('pause');
  });

  it('step 후 같은 프레임에 추가 경계가 와도 재-pause하지 않는다(정책 소진)', () => {
    const h = makeHarness({ stepCount: 5 });
    h.player.setCursorForTest(1);
    h.player.setStatusForTest('running');
    h.orch.step();

    h.player.emitStep(2); // 첫 경계 → pause
    const pauses = h.engine.calls.filter((c) => c === 'pause').length;
    h.player.emitStep(3); // 같은 프레임 오버스텝 시뮬 → 정책 none, 재-pause 없음
    expect(h.engine.calls.filter((c) => c === 'pause').length).toBe(pauses);
  });

  it('step은 시퀀스가 done이면 no-op', () => {
    const h = makeHarness({ stepCount: 3 });
    h.player.setStatusForTest('done');
    h.orch.step();
    expect(h.engine.calls).not.toContain('play');
  });

  it('play는 대기 중인 정지 정책을 해제한다', () => {
    const h = makeHarness({ stepCount: 5 });
    h.orch.pauseAtNextNode();
    h.orch.play(); // 정책 해제
    h.player.emitStep(1);
    expect(h.engine.calls).not.toContain('pause');
  });
});

// ── 충돌 → 오류 + 자동정지 ──────────────────────────────────────────

describe('Orchestrator — 충돌 인지 오류/정지', () => {
  it('예기치 않은 충돌 → 활성 노드 error + 자동정지(활성화 시) + 오류 이벤트', () => {
    const h = makeHarness({
      stepCount: 5,
      unexpectedCollision: () => true,
    });
    h.orch.setAutoPauseOnCollision(true);
    h.orch.play();
    h.player.emitStep(2); // 활성 n3, 엔진 playing

    h.monitor.emit(collisionEvent('arm', 'wall'));

    expect(lastOf(h.statusMaps)['n3']).toBe('error');
    expect(h.engine.calls).toContain('pause');
    expect(h.errors).toHaveLength(1);
    expect(h.errors[0]).toMatchObject({ nodeId: 'n3', reason: 'collision' });
  });

  it('자동정지 비활성 시 error만 마킹, pause 없음', () => {
    const h = makeHarness({ stepCount: 5, unexpectedCollision: () => true });
    h.orch.setAutoPauseOnCollision(false);
    h.orch.play();
    h.player.emitStep(2);

    h.monitor.emit(collisionEvent('arm', 'wall'));
    expect(lastOf(h.statusMaps)['n3']).toBe('error');
    expect(h.engine.calls).not.toContain('pause');
  });

  it('예기치 않은 충돌이 아니면 무시', () => {
    const h = makeHarness({ stepCount: 5, unexpectedCollision: () => false });
    h.orch.setAutoPauseOnCollision(true);
    h.orch.play();
    h.player.emitStep(2);

    h.monitor.emit(collisionEvent('arm', 'box'));
    expect(lastOf(h.statusMaps)['n3']).toBe('active');
    expect(h.errors).toHaveLength(0);
  });

  it('재생 중이 아니면 충돌을 오류로 승격하지 않는다', () => {
    const h = makeHarness({ stepCount: 5, unexpectedCollision: () => true });
    h.orch.setAutoPauseOnCollision(true);
    h.player.emitStep(2); // 엔진 idle (play 안 함)
    h.monitor.emit(collisionEvent('arm', 'wall'));
    expect(h.errors).toHaveLength(0);
    expect(lastOf(h.statusMaps)['n3']).toBe('active');
  });
});

// ── markError ───────────────────────────────────────────────────────

describe('Orchestrator — markError', () => {
  it('지정 노드를 error로 마킹하고 이벤트를 방출한다', () => {
    const h = makeHarness({ stepCount: 5 });
    h.player.emitStep(3);
    h.orch.markError('n4');
    expect(lastOf(h.statusMaps)['n4']).toBe('error');
    expect(lastOf(h.errors)).toMatchObject({ nodeId: 'n4', reason: 'external' });
  });
});

// ── runFromNode ─────────────────────────────────────────────────────

describe('Orchestrator — runFromNode (결정론적 재실행)', () => {
  it('첫 노드가 목표면 처음으로 되감고 재생하지 않는다(엔진 idle)', () => {
    const h = makeHarness({ stepCount: 5 });
    h.player.emitStep(3);
    h.engine.calls.length = 0;

    h.orch.runFromNode('n1'); // index 0
    expect(h.engine.calls).toContain('stop');
    expect(h.resetScene).toHaveBeenCalled();
    expect(h.engine.calls).not.toContain('play'); // 앞에 실행할 노드 없음
  });

  it('뒤쪽 노드가 목표면 되감고 최고 속도로 재생 후 목표 경계에서 정지+속도복원', () => {
    const h = makeHarness({ stepCount: 5 });
    h.orch.setSpeed(2); // 복원 기준 속도
    h.engine.calls.length = 0;
    h.engine.speeds.length = 0;

    h.orch.runFromNode('n4'); // index 3
    expect(h.engine.calls).toContain('stop');
    expect(h.resetScene).toHaveBeenCalled();
    expect(h.engine.speeds).toContain(RUN_FROM_NODE_SPEED_MULT); // fast-forward
    expect(h.engine.calls).toContain('play');
    expect(h.engine.calls).not.toContain('pause'); // 아직 목표 미도달

    // fast-forward 진행
    h.player.emitStep(1);
    h.player.emitStep(2);
    expect(h.engine.calls).not.toContain('pause');

    h.player.emitStep(3); // 목표 도달
    expect(h.engine.calls).toContain('pause');
    expect(lastOf(h.engine.speeds)).toBe(2); // 복원 속도(setSpeed(2))
    // 목표 상태: 앞 done, 목표 active
    expect(lastOf(h.statusMaps)).toEqual({
      n1: 'done',
      n2: 'done',
      n3: 'done',
      n4: 'active',
      n5: 'pending',
    });
  });

  it('알 수 없는 노드 id는 no-op', () => {
    const h = makeHarness({ stepCount: 5 });
    h.engine.calls.length = 0;
    h.orch.runFromNode('does-not-exist');
    expect(h.engine.calls).toHaveLength(0);
    expect(h.resetScene).not.toHaveBeenCalled();
  });

  it('runFromNode는 이전 런의 오류를 지운다', () => {
    const h = makeHarness({ stepCount: 5 });
    h.player.emitStep(2);
    h.orch.markError('n3');
    h.orch.runFromNode('n2'); // index 1
    // 되감기 후 상태에 error가 없다
    const map = lastOf(h.statusMaps);
    expect(Object.values(map)).not.toContain('error');
  });
});

// ── setSpeed passthrough ────────────────────────────────────────────

describe('Orchestrator — setSpeed passthrough', () => {
  it('엔진 setSpeed로 그대로 전달한다', () => {
    const h = makeHarness({ stepCount: 3 });
    h.orch.setSpeed(4);
    expect(lastOf(h.engine.speeds)).toBe(4);
  });
});

// ── 결정론 ──────────────────────────────────────────────────────────

describe('Orchestrator — 결정론', () => {
  it('동일 이벤트 스크립트 → 동일 상태 맵 시퀀스', () => {
    const script = (h: ReturnType<typeof makeHarness>): void => {
      h.orch.play();
      h.player.emitStep(1);
      h.player.emitStep(2);
      h.orch.pauseAtNextNode();
      h.player.emitStep(3);
      h.orch.stop();
    };

    const a = makeHarness({ stepCount: 5 });
    script(a);
    const b = makeHarness({ stepCount: 5 });
    script(b);

    expect(a.statusMaps).toEqual(b.statusMaps);
    expect(a.activeNodes).toEqual(b.activeNodes);
    expect(a.engine.calls).toEqual(b.engine.calls);
  });
});

// ── dispose ─────────────────────────────────────────────────────────

describe('Orchestrator — dispose', () => {
  it('구독 해제 후 통지를 받지 않는다', () => {
    const h = makeHarness({ stepCount: 5 });
    h.orch.dispose();
    const before = h.statusMaps.length;
    h.player.emitStep(2);
    expect(h.statusMaps.length).toBe(before);
  });
});
