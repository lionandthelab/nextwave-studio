// schema/flow-graph.test.ts — Flow Graph ↔ ControlSequence 뷰 모델 단위 테스트
//
// 이 파일은 불변식 §2.8("그래프 편집으로 직렬화 불가능한 상태를 만들 수 없다")의
// 지킴이다: 모든 편집 연산의 ok 결과가 serializeGraph를 통과함을 성질 테스트로 강제한다.
// 규범: docs/UX_DESIGN.md §6, docs/ROADMAP.md Phase 8 게이트.

import { describe, expect, it } from 'vitest';
import type {
  ControlSequence,
  ControlStep,
  ControlStepKind,
  EntitySpec,
  RobotSpec,
  SceneSpec,
} from './types';
import { validateSequence } from './validate';
import type { FlowGraph, FlowNode } from './flow-graph';
import {
  CONTROL_STEP_KINDS,
  DEFAULT_COLLISION_TIMEOUT_SEC,
  DEFAULT_GRIPPER_DURATION_SEC,
  DEFAULT_MOVE_DURATION_SEC,
  DEFAULT_WAIT_DURATION_SEC,
  DUPLICATE_UI_OFFSET,
  FLOW_GRAPH_SEQUENCE_ID,
  LAYOUT_ROW_LENGTH,
  LAYOUT_X_GAP,
  LAYOUT_Y_GAP,
  defaultNodeFor,
  deriveEdges,
  duplicateNode,
  fromSequence,
  insertNode,
  layoutPosition,
  moveNode,
  nextNodeId,
  remapEntityId,
  removeNode,
  serializeGraph,
  setNodeEnabled,
  toSequence,
  updateNodeParams,
} from './flow-graph';
import armTouchBoxJson from '../assets/sequences/arm-touch-box.sequence.json';
import obstacleAvoidanceJson from '../assets/sequences/obstacle-avoidance.sequence.json';
import pickAndPlaceJson from '../assets/sequences/pick-and-place.sequence.json';

// ── 헬퍼 ────────────────────────────────────────────────────────────

function mustSequence(input: unknown, name: string): ControlSequence {
  const result = validateSequence(input);
  if (!result.ok) throw new Error(`${name} 검증 실패: ${result.errors.join(' | ')}`);
  return result.value;
}

/** 편집 결과가 ok임을 단언하고 그래프를 꺼낸다 */
function expectEditOk(
  result: ReturnType<typeof moveNode>,
): FlowGraph {
  if (!result.ok) throw new Error(`편집 성공을 기대했지만 실패: ${result.errors.join(' | ')}`);
  return result.graph;
}

function expectEditFail(result: ReturnType<typeof moveNode>): string[] {
  if (result.ok) throw new Error('편집 거부를 기대했지만 통과했습니다');
  return result.errors;
}

/** §2.8 성질: ok인 편집 결과는 반드시 구조 직렬화를 통과한다 */
function expectSerializable(graph: FlowGraph): ControlSequence {
  const result = serializeGraph(graph);
  if (!result.ok) {
    throw new Error(`§2.8 위반 — 편집 결과가 직렬화 불가능: ${result.errors.join(' | ')}`);
  }
  return result.sequence;
}

/** 그래프의 실행 의미 투영 (id/ui/status/origin 제외) — 왕복 비교용 */
function semantics(graph: FlowGraph): unknown[] {
  return graph.nodes.map((node) => ({
    kind: node.kind,
    params: node.params,
    enabled: node.enabled,
    note: node.note,
  }));
}

/** 깊은 스냅샷 — 순수성(입력 불변) 검증용 */
function snapshot(graph: FlowGraph): FlowGraph {
  return structuredClone(graph);
}

// ── 픽스처 ──────────────────────────────────────────────────────────

const SAMPLE_SEQUENCES: { name: string; seq: ControlSequence }[] = [
  { name: 'arm-touch-box', seq: mustSequence(armTouchBoxJson, 'arm-touch-box') },
  { name: 'obstacle-avoidance', seq: mustSequence(obstacleAvoidanceJson, 'obstacle-avoidance') },
  { name: 'pick-and-place', seq: mustSequence(pickAndPlaceJson, 'pick-and-place') },
];

/** 모든 step 종류 + enabled:false + note를 포함하는 합성 시퀀스 (왕복 커버리지) */
const ALL_KINDS_SEQ: ControlSequence = {
  id: 'all-kinds',
  robot: 'arm',
  loop: false,
  steps: [
    { kind: 'label', name: 'start', note: '루프 시작' },
    {
      kind: 'moveJoints',
      targets: { j1: 0.5, j2: -0.2 },
      durationSec: 1.5,
      easing: 'easeInOut',
    },
    { kind: 'setJoints', targets: { j1: 0 }, enabled: false, note: '임시 비활성' },
    { kind: 'gripper', state: 0.5, durationSec: 0.4 },
    { kind: 'wait', durationSec: 0.3, enabled: false },
    { kind: 'waitForCollision', between: ['arm', 'box_a'], timeoutSec: 5, note: '접촉 대기' },
    {
      kind: 'moveToPose',
      target: { position: [0.1, 0.2, 0.3], rotation: [0, 0, 0, 1] },
      durationSec: 2,
    },
    { kind: 'goto', label: 'start', times: 2, enabled: false, note: '반복 꺼둠' },
  ],
};

/** 씬 참조 무결성 테스트용 최소 씬 (robot 'arm' + object 'box_a') */
const armRobot: RobotSpec = {
  id: 'arm',
  type: 'robot',
  transform: { position: [0, 0, 0] },
  visual: { kind: 'urdf', ref: 'assets/arm/arm.urdf' },
  urdf: 'assets/arm/arm.urdf',
  home: { j1: 0, j2: 0 },
  controller: 'sequence',
};

const boxA: EntitySpec = {
  id: 'box_a',
  type: 'object',
  transform: { position: [0.4, 0.05, 0] },
  visual: { kind: 'primitive', primitive: { kind: 'box', halfExtents: [0.05, 0.05, 0.05] } },
  physics: {
    bodyType: 'dynamic',
    colliders: [
      {
        shape: { kind: 'box', halfExtents: [0.05, 0.05, 0.05] },
        group: 'OBJECT',
        collidesWith: ['ENV', 'ROBOT'],
        emitEvents: true,
      },
    ],
  },
};

const testScene: SceneSpec = {
  name: 'flow-graph-test',
  version: 1,
  gravity: [0, -9.81, 0],
  timestepHz: 240,
  entities: [armRobot, boxA],
};

/** 편집 테스트 기본 그래프: label + wait + goto (loop 엣지 포함) */
function makeLoopGraph(): FlowGraph {
  return fromSequence({
    id: 'edit-base',
    robot: 'arm',
    steps: [
      { kind: 'label', name: 'L' },
      { kind: 'wait', durationSec: 0.5 },
      { kind: 'goto', label: 'L', times: 1 },
    ],
  });
}

// ── fromSequence ────────────────────────────────────────────────────

describe('fromSequence', () => {
  it("노드 id는 'n1','n2',... 순서 안정 부여, kind는 step과 1:1", () => {
    const graph = fromSequence(ALL_KINDS_SEQ);
    expect(graph.nodes.map((n) => n.id)).toEqual(
      ALL_KINDS_SEQ.steps.map((_, i) => `n${i + 1}`),
    );
    expect(graph.nodes.map((n) => n.kind)).toEqual(ALL_KINDS_SEQ.steps.map((s) => s.kind));
    expect(graph.robot).toBe('arm');
  });

  it('params는 step 필드에서 kind/enabled/note를 뺀 값이다', () => {
    const graph = fromSequence(ALL_KINDS_SEQ);
    const moveNode1 = graph.nodes[1];
    expect(moveNode1?.params).toEqual({
      targets: { j1: 0.5, j2: -0.2 },
      durationSec: 1.5,
      easing: 'easeInOut',
    });
    const disabledSet = graph.nodes[2];
    expect(disabledSet?.params).toEqual({ targets: { j1: 0 } });
    expect(disabledSet?.params).not.toHaveProperty('enabled');
    expect(disabledSet?.params).not.toHaveProperty('note');
    expect(disabledSet?.params).not.toHaveProperty('kind');
  });

  it('enabled 기본 true, enabled:false와 note는 노드 필드로 옮겨진다', () => {
    const graph = fromSequence(ALL_KINDS_SEQ);
    expect(graph.nodes[0]?.enabled).toBe(true);
    expect(graph.nodes[0]?.note).toBe('루프 시작');
    expect(graph.nodes[2]?.enabled).toBe(false);
    expect(graph.nodes[2]?.note).toBe('임시 비활성');
    expect(graph.nodes[4]?.enabled).toBe(false);
    expect(graph.nodes[4]?.note).toBeUndefined();
  });

  it("origin 기본 'manual', 옵션으로 'generated' 지정 가능 (플래너 로드)", () => {
    const manual = fromSequence(ALL_KINDS_SEQ);
    expect(manual.nodes.every((n) => n.origin === 'manual')).toBe(true);
    const generated = fromSequence(ALL_KINDS_SEQ, { origin: 'generated' });
    expect(generated.nodes.every((n) => n.origin === 'generated')).toBe(true);
  });

  it('ui는 layoutPosition 자동 배치 — 가로 체인 + LAYOUT_ROW_LENGTH개마다 wrap', () => {
    const manySteps: ControlStep[] = Array.from({ length: LAYOUT_ROW_LENGTH + 2 }, () => ({
      kind: 'wait',
      durationSec: 1,
    }));
    const graph = fromSequence({ id: 'many', robot: 'arm', steps: manySteps });

    expect(graph.nodes[0]?.ui).toEqual({ x: 0, y: 0 });
    expect(graph.nodes[1]?.ui).toEqual({ x: LAYOUT_X_GAP, y: 0 });
    expect(graph.nodes[LAYOUT_ROW_LENGTH - 1]?.ui).toEqual({
      x: (LAYOUT_ROW_LENGTH - 1) * LAYOUT_X_GAP,
      y: 0,
    });
    // wrap: 다음 줄 첫 칸
    expect(graph.nodes[LAYOUT_ROW_LENGTH]?.ui).toEqual({ x: 0, y: LAYOUT_Y_GAP });
    expect(graph.nodes[LAYOUT_ROW_LENGTH + 1]?.ui).toEqual({ x: LAYOUT_X_GAP, y: LAYOUT_Y_GAP });
    // layoutPosition과 일치 (결정론)
    graph.nodes.forEach((node, i) => expect(node.ui).toEqual(layoutPosition(i)));
  });

  it('호출마다 동일한 그래프 (순수 결정론)', () => {
    expect(fromSequence(ALL_KINDS_SEQ)).toEqual(fromSequence(ALL_KINDS_SEQ));
  });

  it('params는 깊은 복사 — 그래프를 변형해도 원본 시퀀스가 오염되지 않는다', () => {
    const seq = structuredClone(ALL_KINDS_SEQ);
    const graph = fromSequence(seq);
    const targets = graph.nodes[1]?.params['targets'] as Record<string, number>;
    targets['j1'] = 999;
    const step1 = seq.steps[1];
    expect(step1?.kind === 'moveJoints' && step1.targets['j1']).toBe(0.5);
  });
});

// ── deriveEdges ─────────────────────────────────────────────────────

describe('deriveEdges', () => {
  it('체인을 따라 seq 엣지 n-1개 + goto → 대상 label loop 백엣지', () => {
    const graph = makeLoopGraph();
    expect(graph.edges).toEqual([
      { from: 'n1', to: 'n2', kind: 'seq' },
      { from: 'n2', to: 'n3', kind: 'seq' },
      { from: 'n3', to: 'n1', kind: 'loop' }, // goto → label 백엣지
    ]);
  });

  it('중복 label 이름은 첫 번째 label 노드가 이긴다 (player 규칙과 동일)', () => {
    const graph = fromSequence({
      id: 'dup-labels',
      robot: 'arm',
      steps: [
        { kind: 'label', name: 'L' },
        { kind: 'label', name: 'L' },
        { kind: 'goto', label: 'L', times: 1 },
      ],
    });
    const loops = graph.edges.filter((e) => e.kind === 'loop');
    expect(loops).toEqual([{ from: 'n3', to: 'n1', kind: 'loop' }]);
  });

  it('대상 label이 없는 goto는 loop 엣지를 만들지 않는다 (deriveEdges는 total)', () => {
    const nodes: FlowNode[] = [
      {
        id: 'n1',
        kind: 'goto',
        params: { label: 'ghost' },
        enabled: true,
        origin: 'manual',
        ui: { x: 0, y: 0 },
      },
    ];
    expect(deriveEdges(nodes)).toEqual([]);
  });

  it('비활성 goto도 loop 엣지를 유지한다 — 엣지는 구조 뷰, 실행 여부는 enabled', () => {
    const graph = fromSequence(ALL_KINDS_SEQ); // 마지막 goto가 enabled:false
    const loops = graph.edges.filter((e) => e.kind === 'loop');
    expect(loops).toEqual([{ from: 'n8', to: 'n1', kind: 'loop' }]);
  });

  it('노드 1개면 엣지가 없다', () => {
    const graph = fromSequence({
      id: 'one',
      robot: 'arm',
      steps: [{ kind: 'wait', durationSec: 1 }],
    });
    expect(graph.edges).toEqual([]);
  });
});

// ── toSequence / serializeGraph ─────────────────────────────────────

describe('toSequence', () => {
  it('노드 순서 → steps, robot은 그래프에서, enabled true는 생략(정규형)', () => {
    const graph = fromSequence(ALL_KINDS_SEQ);
    const seq = toSequence(graph, { id: ALL_KINDS_SEQ.id, loop: ALL_KINDS_SEQ.loop });
    expect(seq.steps).toEqual(ALL_KINDS_SEQ.steps);
    expect(seq.robot).toBe('arm');
    expect(seq.id).toBe('all-kinds');
    expect(seq.loop).toBe(false);
    // enabled:true인 step에는 enabled 키 자체가 없다 (정규형)
    expect(seq.steps[0]).not.toHaveProperty('enabled');
    // enabled:false는 보존된다
    expect(seq.steps[2]).toHaveProperty('enabled', false);
  });

  it('옵션 미지정 시 기본 id, loop 키 없음', () => {
    const seq = toSequence(makeLoopGraph());
    expect(seq.id).toBe(FLOW_GRAPH_SEQUENCE_ID);
    expect(seq).not.toHaveProperty('loop');
  });

  it("source의 enabled:true 명시는 생략으로 정규화된다 (실행 의미 동일)", () => {
    const seq: ControlSequence = {
      id: 'explicit-true',
      robot: 'arm',
      steps: [{ kind: 'wait', durationSec: 1, enabled: true }],
    };
    const out = toSequence(fromSequence(seq));
    expect(out.steps[0]).toEqual({ kind: 'wait', durationSec: 1 });
  });

  it('params 안의 kind/enabled/note 키는 무시된다 (노드 필드가 이긴다 — 방어)', () => {
    const graph = makeLoopGraph();
    const polluted: FlowGraph = {
      ...graph,
      nodes: graph.nodes.map((n) =>
        n.id === 'n2'
          ? { ...n, params: { durationSec: 0.5, kind: 'goto', enabled: false, note: 'x' } }
          : n,
      ),
    };
    const seq = toSequence(polluted);
    expect(seq.steps[1]).toEqual({ kind: 'wait', durationSec: 0.5 });
  });

  it('알 수 없는 kind에서만 throw한다 (한국어 메시지)', () => {
    const graph = makeLoopGraph();
    const broken: FlowGraph = {
      ...graph,
      nodes: graph.nodes.map((n) =>
        n.id === 'n2' ? { ...n, kind: 'teleport' as ControlStepKind } : n,
      ),
    };
    expect(() => toSequence(broken)).toThrow(/teleport.*알 수 없는 step 종류/);
    // 반면 파라미터 값 오류는 throw하지 않는다 — validateSequence 소관
    const withBadDuration: FlowGraph = {
      ...graph,
      nodes: graph.nodes.map((n) => (n.id === 'n2' ? { ...n, params: { durationSec: -1 } } : n)),
    };
    expect(() => toSequence(withBadDuration)).not.toThrow();
  });
});

describe('serializeGraph — UI 유일 직렬화 경로 (§2.8)', () => {
  it('유효 그래프: ok + validateSequence 통과 시퀀스', () => {
    const result = serializeGraph(makeLoopGraph());
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(validateSequence(result.sequence).ok).toBe(true);
    }
  });

  it('무효 파라미터(음수 duration)는 ok:false + 한국어 오류', () => {
    const graph = makeLoopGraph();
    const bad: FlowGraph = {
      ...graph,
      nodes: graph.nodes.map((n) => (n.id === 'n2' ? { ...n, params: { durationSec: -1 } } : n)),
    };
    const result = serializeGraph(bad);
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.errors.join('\n')).toContain('durationSec');
    }
  });

  it('알 수 없는 kind(구조 불가능)도 throw 대신 ok:false로 흡수한다', () => {
    const graph = makeLoopGraph();
    const broken: FlowGraph = {
      ...graph,
      nodes: graph.nodes.map((n) =>
        n.id === 'n1' ? { ...n, kind: 'teleport' as ControlStepKind } : n,
      ),
    };
    const result = serializeGraph(broken);
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.errors.join('\n')).toContain('알 수 없는 step 종류');
  });

  it('scene 제공 시 참조 무결성까지 검사한다 (ghost robot 거부)', () => {
    const graph: FlowGraph = { ...makeLoopGraph(), robot: 'ghost' };
    const withoutScene = serializeGraph(graph);
    expect(withoutScene.ok).toBe(true); // 구조만으로는 유효
    const withScene = serializeGraph(graph, testScene);
    expect(withScene.ok).toBe(false);
    if (!withScene.ok) expect(withScene.errors.join('\n')).toContain("'ghost'");
  });

  it('opts.id/loop가 시퀀스에 반영된다', () => {
    const result = serializeGraph(makeLoopGraph(), undefined, { id: 'restored', loop: true });
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.sequence.id).toBe('restored');
      expect(result.sequence.loop).toBe(true);
    }
  });
});

// ── 무손실 왕복 (§2.8 핵심 성질) ────────────────────────────────────

describe('왕복: toSequence(fromSequence(seq)).steps ≡ seq.steps', () => {
  for (const { name, seq } of SAMPLE_SEQUENCES) {
    it(`샘플 ${name} (${seq.steps.length} steps)`, () => {
      const graph = fromSequence(seq);
      const out = toSequence(graph, { id: seq.id, loop: seq.loop });
      expect(out.steps).toEqual(seq.steps);
      expect(out.robot).toBe(seq.robot);
      expect(validateSequence(out).ok).toBe(true);
    });
  }

  it('합성 all-kinds 시퀀스 (enabled:false + note 포함, 모든 kind)', () => {
    const graph = fromSequence(ALL_KINDS_SEQ);
    const out = toSequence(graph, { id: ALL_KINDS_SEQ.id, loop: ALL_KINDS_SEQ.loop });
    expect(out.steps).toEqual(ALL_KINDS_SEQ.steps);
    expect(validateSequence(out).ok).toBe(true);
    // 모든 kind가 실제로 커버되는지 (테스트 자체 무결성)
    const kinds = new Set(ALL_KINDS_SEQ.steps.map((s) => s.kind));
    expect([...kinds].sort()).toEqual([...CONTROL_STEP_KINDS].sort());
  });

  it('fromSequence(toSequence(g))는 실행 의미(kind/params/enabled/note)를 보존한다', () => {
    for (const { seq } of SAMPLE_SEQUENCES) {
      const g = fromSequence(seq);
      const g2 = fromSequence(toSequence(g));
      expect(semantics(g2)).toEqual(semantics(g));
    }
    const g = fromSequence(ALL_KINDS_SEQ, { origin: 'generated' });
    const g2 = fromSequence(toSequence(g));
    expect(semantics(g2)).toEqual(semantics(g));
  });
});

// ── 편집 연산 ───────────────────────────────────────────────────────

describe('moveNode', () => {
  it('노드를 지정 인덱스로 이동하고 엣지를 재유도한다 (입력 그래프 불변)', () => {
    const graph = makeLoopGraph();
    const before = snapshot(graph);
    const moved = expectEditOk(moveNode(graph, 'n2', 0)); // wait를 맨 앞으로
    expect(moved.nodes.map((n) => n.id)).toEqual(['n2', 'n1', 'n3']);
    expect(moved.edges).toEqual([
      { from: 'n2', to: 'n1', kind: 'seq' },
      { from: 'n1', to: 'n3', kind: 'seq' },
      { from: 'n3', to: 'n1', kind: 'loop' },
    ]);
    expect(graph).toEqual(before); // 순수성
    expectSerializable(moved);
  });

  it('goto를 label 앞으로 옮겨도 유효하다 (전방 goto 허용)', () => {
    const graph = makeLoopGraph();
    const moved = expectEditOk(moveNode(graph, 'n3', 0));
    expect(moved.nodes.map((n) => n.id)).toEqual(['n3', 'n1', 'n2']);
    expectSerializable(moved);
  });

  it('없는 노드 / 범위 밖 인덱스는 거부한다 (한국어)', () => {
    const graph = makeLoopGraph();
    expect(expectEditFail(moveNode(graph, 'ghost', 0)).join('')).toContain("'ghost'");
    expect(expectEditFail(moveNode(graph, 'n1', 3)).join('')).toContain('범위');
    expect(expectEditFail(moveNode(graph, 'n1', -1)).join('')).toContain('범위');
    expect(expectEditFail(moveNode(graph, 'n1', 1.5)).join('')).toContain('범위');
  });
});

describe('insertNode', () => {
  it('지정 위치에 삽입한다 (경계 0과 length 포함)', () => {
    const graph = makeLoopGraph();
    const node = defaultNodeFor('wait', { robot: 'arm', entityIds: [], labels: ['L'] });
    const atFront = expectEditOk(insertNode(graph, node, 0));
    expect(atFront.nodes[0]?.kind).toBe('wait');
    const atEnd = expectEditOk(insertNode(graph, node, graph.nodes.length));
    expect(atEnd.nodes.at(-1)?.kind).toBe('wait');
    expectSerializable(atFront);
    expectSerializable(atEnd);
  });

  it("id ''(자리표시자)는 nextNodeId로 보정된다", () => {
    const graph = makeLoopGraph(); // n1..n3
    const node = defaultNodeFor('wait', { robot: 'arm', entityIds: [], labels: [] });
    expect(node.id).toBe('');
    const result = expectEditOk(insertNode(graph, node, 1));
    expect(result.nodes[1]?.id).toBe('n4');
  });

  it('중복 id는 거부한다', () => {
    const graph = makeLoopGraph();
    const node = { ...defaultNodeFor('wait', { robot: 'arm', entityIds: [], labels: [] }), id: 'n1' };
    expect(expectEditFail(insertNode(graph, node, 0)).join('')).toContain("'n1'");
  });

  it('범위 밖 인덱스는 거부한다', () => {
    const graph = makeLoopGraph();
    const node = defaultNodeFor('wait', { robot: 'arm', entityIds: [], labels: [] });
    expect(expectEditFail(insertNode(graph, node, 4)).join('')).toContain('범위');
    expect(expectEditFail(insertNode(graph, node, -1)).join('')).toContain('범위');
  });

  it('직렬화를 깨는 노드(없는 label로의 goto)는 삽입이 거부된다 (§2.8)', () => {
    const graph = makeLoopGraph();
    const badGoto: FlowNode = {
      id: '',
      kind: 'goto',
      params: { label: 'ghost' },
      enabled: true,
      origin: 'manual',
      ui: { x: 0, y: 0 },
    };
    const errors = expectEditFail(insertNode(graph, badGoto, 3));
    expect(errors.join('\n')).toContain("'ghost'");
  });

  it('입력 노드는 깊은 복사된다 — 삽입 후 원본 변형이 그래프에 새지 않는다', () => {
    const graph = makeLoopGraph();
    const node = defaultNodeFor('moveJoints', { robot: 'arm', entityIds: [], labels: [] });
    const result = expectEditOk(insertNode(graph, node, 0));
    (node.params['targets'] as Record<string, number>)['j1'] = 123;
    expect(result.nodes[0]?.params['targets']).toEqual({});
  });
});

describe('removeNode', () => {
  it('중간 노드를 삭제하면 앞뒤 엣지가 자동 재연결된다', () => {
    const graph = fromSequence({
      id: 'chain',
      robot: 'arm',
      steps: [
        { kind: 'wait', durationSec: 1 },
        { kind: 'wait', durationSec: 2 },
        { kind: 'wait', durationSec: 3 },
      ],
    });
    const removed = expectEditOk(removeNode(graph, 'n2'));
    expect(removed.nodes.map((n) => n.id)).toEqual(['n1', 'n3']);
    expect(removed.edges).toEqual([{ from: 'n1', to: 'n3', kind: 'seq' }]);
    expectSerializable(removed);
  });

  it('goto가 참조하는 label 삭제는 한국어 오류로 거부한다 (§2.8)', () => {
    const graph = makeLoopGraph();
    const errors = expectEditFail(removeNode(graph, 'n1'));
    expect(errors.join('\n')).toContain("label 'L'");
    expect(errors.join('\n')).toContain('goto');
  });

  it('같은 이름 label이 남아 있으면 label 삭제를 허용한다', () => {
    const graph = fromSequence({
      id: 'dup-labels',
      robot: 'arm',
      steps: [
        { kind: 'label', name: 'L' },
        { kind: 'label', name: 'L' },
        { kind: 'goto', label: 'L', times: 1 },
      ],
    });
    const removed = expectEditOk(removeNode(graph, 'n1'));
    expect(removed.nodes.map((n) => n.id)).toEqual(['n2', 'n3']);
    expectSerializable(removed);
  });

  it('goto 삭제는 허용되고 loop 엣지가 사라진다', () => {
    const graph = makeLoopGraph();
    const removed = expectEditOk(removeNode(graph, 'n3'));
    expect(removed.edges.filter((e) => e.kind === 'loop')).toEqual([]);
    // goto가 사라졌으니 이제 label도 삭제 가능
    expectEditOk(removeNode(removed, 'n1'));
  });

  it('마지막 남은 노드는 삭제할 수 없다 (최소 1 step)', () => {
    const graph = fromSequence({
      id: 'single',
      robot: 'arm',
      steps: [{ kind: 'wait', durationSec: 1 }],
    });
    expect(expectEditFail(removeNode(graph, 'n1')).join('')).toContain('최소 1개');
  });

  it('없는 노드는 거부한다', () => {
    expect(expectEditFail(removeNode(makeLoopGraph(), 'ghost')).join('')).toContain("'ghost'");
  });
});

describe('duplicateNode', () => {
  it('원본 바로 뒤에 새 id로 삽입되고 params가 독립 복사된다', () => {
    const graph = fromSequence(ALL_KINDS_SEQ);
    const result = expectEditOk(duplicateNode(graph, 'n2')); // moveJoints
    expect(result.nodes.map((n) => n.id).slice(0, 3)).toEqual(['n1', 'n2', 'n9']);
    const source = result.nodes[1];
    const copy = result.nodes[2];
    expect(copy?.kind).toBe('moveJoints');
    expect(copy?.params).toEqual(source?.params);
    // 독립성: 복사본 변형이 원본에 새지 않는다
    (copy?.params['targets'] as Record<string, number>)['j1'] = 42;
    expect((source?.params['targets'] as Record<string, number>)['j1']).toBe(0.5);
    expectSerializable(result);
  });

  it("복제본은 origin 'manual', status 없음, ui는 오프셋 배치", () => {
    const graph = fromSequence(ALL_KINDS_SEQ, { origin: 'generated' });
    const withStatus: FlowGraph = {
      ...graph,
      nodes: graph.nodes.map((n) => (n.id === 'n2' ? { ...n, status: 'done' as const } : n)),
    };
    const result = expectEditOk(duplicateNode(withStatus, 'n2'));
    const copy = result.nodes[2];
    expect(copy?.origin).toBe('manual');
    expect(copy?.status).toBeUndefined();
    const source = result.nodes[1];
    expect(copy?.ui).toEqual({
      x: (source?.ui.x ?? 0) + DUPLICATE_UI_OFFSET,
      y: (source?.ui.y ?? 0) + DUPLICATE_UI_OFFSET,
    });
  });

  it('label 복제(중복 이름)도 유효하다 — 첫 번째 label이 goto 대상', () => {
    const graph = makeLoopGraph();
    const result = expectEditOk(duplicateNode(graph, 'n1'));
    const loops = result.edges.filter((e) => e.kind === 'loop');
    expect(loops).toEqual([{ from: 'n3', to: 'n1', kind: 'loop' }]); // 여전히 첫 label
    expectSerializable(result);
  });

  it('없는 노드는 거부한다', () => {
    expect(expectEditFail(duplicateNode(makeLoopGraph(), 'ghost')).join('')).toContain("'ghost'");
  });
});

describe('setNodeEnabled', () => {
  it('비활성화는 직렬화에 enabled:false로 반영되고 순서는 유지된다', () => {
    const graph = makeLoopGraph();
    const result = expectEditOk(setNodeEnabled(graph, 'n2', false));
    const seq = expectSerializable(result);
    expect(seq.steps[1]).toHaveProperty('enabled', false);
    expect(seq.steps).toHaveLength(3);
  });

  it('재활성화하면 enabled 키가 사라진다 (정규형)', () => {
    const graph = expectEditOk(setNodeEnabled(makeLoopGraph(), 'n2', false));
    const result = expectEditOk(setNodeEnabled(graph, 'n2', true));
    const seq = expectSerializable(result);
    expect(seq.steps[1]).not.toHaveProperty('enabled');
  });

  it('goto가 참조하는 label을 비활성화해도 유효하다 (label은 위치 마커)', () => {
    const graph = makeLoopGraph();
    const result = expectEditOk(setNodeEnabled(graph, 'n1', false));
    const seq = expectSerializable(result);
    expect(seq.steps[0]).toEqual({ kind: 'label', name: 'L', enabled: false });
  });

  it("generated 노드를 토글하면 origin이 'modified'가 된다 (수정됨 배지)", () => {
    const graph = fromSequence(ALL_KINDS_SEQ, { origin: 'generated' });
    const result = expectEditOk(setNodeEnabled(graph, 'n2', false));
    expect(result.nodes[1]?.origin).toBe('modified');
    expect(result.nodes[0]?.origin).toBe('generated'); // 다른 노드는 그대로
  });

  it('없는 노드는 거부한다', () => {
    expect(
      expectEditFail(setNodeEnabled(makeLoopGraph(), 'ghost', false)).join(''),
    ).toContain("'ghost'");
  });
});

describe('updateNodeParams', () => {
  it('유효한 파라미터 교체는 직렬화에 반영된다', () => {
    const graph = makeLoopGraph();
    const result = expectEditOk(updateNodeParams(graph, 'n2', { durationSec: 2.5 }));
    const seq = expectSerializable(result);
    expect(seq.steps[1]).toEqual({ kind: 'wait', durationSec: 2.5 });
  });

  it('무효한 파라미터(음수 duration)는 검증 오류와 함께 거부된다 (§2.8)', () => {
    const graph = makeLoopGraph();
    const before = snapshot(graph);
    const errors = expectEditFail(updateNodeParams(graph, 'n2', { durationSec: -1 }));
    expect(errors.join('\n')).toContain('durationSec');
    expect(graph).toEqual(before); // 거부 시에도 입력 불변
  });

  it('goto의 label을 없는 이름으로 바꾸면 거부된다 (§2.8)', () => {
    const graph = makeLoopGraph();
    const errors = expectEditFail(updateNodeParams(graph, 'n3', { label: 'ghost', times: 1 }));
    expect(errors.join('\n')).toContain("'ghost'");
  });

  it('params 안의 kind/enabled/note 키는 버려진다 (노드 필드가 단일 진실)', () => {
    const graph = makeLoopGraph();
    const result = expectEditOk(
      updateNodeParams(graph, 'n2', { durationSec: 1, kind: 'goto', enabled: false, note: 'x' }),
    );
    expect(result.nodes[1]?.params).toEqual({ durationSec: 1 });
    expect(result.nodes[1]?.kind).toBe('wait');
    expect(result.nodes[1]?.enabled).toBe(true);
  });

  it("generated 노드를 편집하면 origin이 'modified'가 된다", () => {
    const graph = fromSequence(ALL_KINDS_SEQ, { origin: 'generated' });
    const result = expectEditOk(updateNodeParams(graph, 'n5', { durationSec: 9 }));
    expect(result.nodes[4]?.origin).toBe('modified');
  });

  it('없는 노드는 거부한다', () => {
    expect(
      expectEditFail(updateNodeParams(makeLoopGraph(), 'ghost', {})).join(''),
    ).toContain("'ghost'");
  });
});

// ── remapEntityId — 씬 엔티티 개명 동기화 ───────────────────────────

describe('remapEntityId', () => {
  /** 기본 로봇 + step 로봇 + between 참조를 모두 포함하는 그래프 */
  const remapSeq: ControlSequence = {
    id: 'remap-test',
    robot: 'arm',
    steps: [
      { kind: 'gripper', robot: 'arm', state: 'open', durationSec: 0.4 },
      { kind: 'waitForCollision', between: ['arm', 'box_a'], timeoutSec: 5, note: '접촉' },
      { kind: 'wait', durationSec: 1 },
    ],
  };

  it('robot 기본값 · step params.robot · between 참조를 모두 치환한다', () => {
    const graph = fromSequence(remapSeq);
    const result = remapEntityId(graph, 'arm', 'arm2');
    expect(result.robot).toBe('arm2');
    expect(result.nodes[0]?.params['robot']).toBe('arm2');
    expect(result.nodes[1]?.params['between']).toEqual(['arm2', 'box_a']);
    expect(result.nodes[2]?.params).toEqual({ durationSec: 1 }); // 참조 없는 노드는 그대로
  });

  it('개명된 씬에 대해 직렬화(참조 무결성)를 통과한다 — 편집 잠김 방지', () => {
    const renamedScene = structuredClone(testScene);
    const robotEntity = renamedScene.entities.find((e) => e.id === 'arm');
    if (robotEntity) robotEntity.id = 'arm2';

    const graph = fromSequence(remapSeq);
    // 리매핑 전: 옛 id 참조로 씬 검증 실패 (잠김 상태 재현)
    const stale = serializeGraph(graph, renamedScene);
    expect(stale.ok).toBe(false);
    // 리매핑 후: 통과
    const remapped = remapEntityId(graph, 'arm', 'arm2');
    const result = serializeGraph(remapped, renamedScene);
    expect(result.ok).toBe(true);
    if (result.ok) expect(result.sequence.robot).toBe('arm2');
  });

  it('between 한쪽(비로봇 엔티티)만 참조해도 치환된다', () => {
    const graph = fromSequence(remapSeq);
    const result = remapEntityId(graph, 'box_a', 'box_b');
    expect(result.robot).toBe('arm'); // 기본 로봇 불변
    expect(result.nodes[1]?.params['between']).toEqual(['arm', 'box_b']);
  });

  it('참조가 없으면 입력 그래프를 동일 참조로 반환한다 (호출자의 변경 판별 계약)', () => {
    const graph = fromSequence(remapSeq);
    expect(remapEntityId(graph, 'ghost', 'ghost2')).toBe(graph);
    expect(remapEntityId(graph, 'arm', 'arm')).toBe(graph); // oldId === newId
  });

  it('순수성: 입력 그래프를 변형하지 않는다', () => {
    const graph = fromSequence(remapSeq);
    const before = snapshot(graph);
    remapEntityId(graph, 'arm', 'arm2');
    expect(graph).toEqual(before);
  });

  it('시스템 동기화 — origin/enabled/note/ui는 바뀌지 않는다', () => {
    const graph = fromSequence(remapSeq, { origin: 'generated' });
    const result = remapEntityId(graph, 'arm', 'arm2');
    result.nodes.forEach((node, i) => {
      expect(node.origin).toBe('generated'); // '수정됨' 승격 없음
      expect(node.enabled).toBe(graph.nodes[i]?.enabled);
      expect(node.note).toBe(graph.nodes[i]?.note);
      expect(node.ui).toEqual(graph.nodes[i]?.ui);
      expect(node.id).toBe(graph.nodes[i]?.id);
    });
  });

  it('빈 그래프의 robot 기본값도 치환된다 (노드 없는 씬의 기본 로봇 추종)', () => {
    const empty: FlowGraph = { nodes: [], edges: [], robot: 'arm' };
    expect(remapEntityId(empty, 'arm', 'arm2').robot).toBe('arm2');
  });
});

// ── §2.8 성질 테스트: 모든 편집 연산의 ok 결과는 직렬화 가능하다 ─────

describe('§2.8 성질: 편집 연산 체인의 모든 ok 결과가 serializeGraph를 통과한다', () => {
  it('all-kinds 그래프에 연산을 연쇄 적용해도 직렬화 가능성이 유지된다', () => {
    let graph = fromSequence(ALL_KINDS_SEQ);
    expectSerializable(graph);

    const ops: ((g: FlowGraph) => ReturnType<typeof moveNode>)[] = [
      (g) => moveNode(g, 'n4', 1),
      (g) => setNodeEnabled(g, 'n2', false),
      (g) => duplicateNode(g, 'n6'),
      (g) =>
        insertNode(g, defaultNodeFor('wait', { robot: 'arm', entityIds: [], labels: [] }), 0),
      (g) => updateNodeParams(g, 'n5', { durationSec: 0.9 }),
      (g) => removeNode(g, 'n7'),
      (g) => setNodeEnabled(g, 'n1', false),
      (g) => duplicateNode(g, 'n1'),
      (g) => moveNode(g, 'n8', 0),
    ];

    for (const op of ops) {
      const result = op(graph);
      if (result.ok) {
        expectSerializable(result.graph); // §2.8 핵심 단언
        graph = result.graph;
      }
      // ok가 아니어도 원 그래프는 여전히 직렬화 가능해야 한다
      expectSerializable(graph);
    }
  });

  it('거부된 편집은 그래프를 바꾸지 않는다 — 직렬화 출력이 동일하다', () => {
    const graph = makeLoopGraph();
    const beforeSeq = expectSerializable(graph);
    expectEditFail(removeNode(graph, 'n1')); // label 삭제 거부
    expectEditFail(updateNodeParams(graph, 'n2', { durationSec: -5 }));
    expect(expectSerializable(graph)).toEqual(beforeSeq);
  });
});

// ── nextNodeId ──────────────────────────────────────────────────────

describe('nextNodeId', () => {
  it("'n{최대+1}'를 발급한다 (구멍이 있어도 최대 기준)", () => {
    const graph = fromSequence(ALL_KINDS_SEQ); // n1..n8
    expect(nextNodeId(graph.nodes)).toBe('n9');
    const removed = expectEditOk(removeNode(graph, 'n3'));
    expect(nextNodeId(removed.nodes)).toBe('n9'); // n3이 비어도 재사용하지 않는다
  });

  it('규약 밖 id는 무시하되 충돌은 회피한다', () => {
    const nodes: FlowNode[] = [
      { id: 'custom', kind: 'wait', params: { durationSec: 1 }, enabled: true, origin: 'manual', ui: { x: 0, y: 0 } },
      { id: 'n1', kind: 'wait', params: { durationSec: 1 }, enabled: true, origin: 'manual', ui: { x: 0, y: 0 } },
    ];
    expect(nextNodeId(nodes)).toBe('n2');
    expect(nextNodeId([])).toBe('n1');
  });
});

// ── defaultNodeFor ──────────────────────────────────────────────────

describe('defaultNodeFor', () => {
  const ctx = { robot: 'arm', entityIds: ['arm', 'box_a'], labels: ['L'] };

  it('모든 kind의 기본 노드가 그래프에 삽입 시 직렬화를 통과한다', () => {
    const base = makeLoopGraph(); // label 'L' 존재 — goto 기본값 유효
    for (const kind of CONTROL_STEP_KINDS) {
      const node = defaultNodeFor(kind, ctx);
      expect(node.id).toBe('');
      expect(node.enabled).toBe(true);
      expect(node.origin).toBe('manual');
      const result = insertNode(base, node, base.nodes.length);
      const graph = expectEditOk(result);
      expectSerializable(graph);
    }
  });

  it('kind별 기본값이 명세와 일치한다', () => {
    expect(defaultNodeFor('moveJoints', ctx).params).toEqual({
      targets: {},
      durationSec: DEFAULT_MOVE_DURATION_SEC,
      easing: 'easeInOut',
    });
    expect(defaultNodeFor('setJoints', ctx).params).toEqual({ targets: {} });
    expect(defaultNodeFor('gripper', ctx).params).toEqual({
      state: 'close',
      durationSec: DEFAULT_GRIPPER_DURATION_SEC,
    });
    expect(defaultNodeFor('wait', ctx).params).toEqual({
      durationSec: DEFAULT_WAIT_DURATION_SEC,
    });
    expect(defaultNodeFor('waitForCollision', ctx).params).toEqual({
      between: ['arm', 'box_a'],
      timeoutSec: DEFAULT_COLLISION_TIMEOUT_SEC,
    });
    expect(defaultNodeFor('goto', ctx).params).toEqual({ label: 'L', times: 1 });
    expect(defaultNodeFor('moveToPose', ctx).params).toEqual({
      target: { position: [0, 0, 0] },
      durationSec: DEFAULT_MOVE_DURATION_SEC,
    });
  });

  it('label 기본 이름은 기존 label과 충돌하지 않는다', () => {
    expect(defaultNodeFor('label', { ...ctx, labels: [] }).params).toEqual({ name: 'L1' });
    expect(defaultNodeFor('label', { ...ctx, labels: ['L1', 'L2'] }).params).toEqual({
      name: 'L3',
    });
  });

  it('label이 없으면 goto는 한국어 오류로 throw한다', () => {
    expect(() => defaultNodeFor('goto', { ...ctx, labels: [] })).toThrow(/label 노드가 필요/);
  });

  it('엔티티가 2개 미만이면 waitForCollision은 한국어 오류로 throw한다', () => {
    expect(() => defaultNodeFor('waitForCollision', { ...ctx, entityIds: ['arm'] })).toThrow(
      /엔티티가 2개 이상/,
    );
  });
});
