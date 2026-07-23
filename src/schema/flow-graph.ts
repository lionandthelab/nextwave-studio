// schema/flow-graph.ts — Flow Graph ↔ ControlSequence 뷰 모델 (규범: docs/UX_DESIGN.md §6)
//
// Flow Graph는 선형 `ControlSequence.steps[]`에 대한 **편집 가능한 뷰**다 (불변식 §2.8):
// - 노드 순서(체인)는 steps 배열 순서와 1:1. edges는 노드 배열(순서 + goto params)에서
//   **파생되는 상태**다 — 단일 진실은 노드 배열이며, 모든 편집 연산이 deriveEdges로
//   엣지를 재유도한다.
// - ui.x/y와 status는 순수 표현 상태 — 직렬화(toSequence) 결과에 영향하지 않는다.
// - 모든 편집 연산은 순수 함수(그래프 in → 그래프 out, 입력 불변)이고, ok 결과는
//   반드시 serializeGraph(구조 검증)를 통과한 그래프만 반환한다. 통과하지 못하는 편집은
//   { ok:false, errors(한국어) }로 거부된다 — 그래프 편집으로 직렬화 불가능한 상태를
//   만들 수 없다 (CLAUDE.md §2.8, ROADMAP Phase 8 게이트).
// - 씬 참조 무결성(robot/관절/엔티티 존재)은 씬이 있어야 검사 가능하므로 편집 연산이
//   아니라 serializeGraph(graph, scene) 호출 지점(UI 저장/실행 경로)의 소관이다.
// - 이 모듈은 순수하다: schema 내부(types/validate)에만 의존한다 (ARCHITECTURE §3).

import type {
  ControlSequence,
  ControlStep,
  ControlStepKind,
  SceneSpec,
} from './types';
import { validateSequence } from './validate';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4) ─────────────────────────────

/** 자동 배치: 체인 방향(가로) 노드 간격 px */
export const LAYOUT_X_GAP = 180;
/** 자동 배치: 줄바꿈 시 세로 간격 px */
export const LAYOUT_Y_GAP = 140;
/** 자동 배치: 한 줄에 배치하는 최대 노드 수 (초과 시 다음 줄로 wrap) */
export const LAYOUT_ROW_LENGTH = 8;
/** duplicateNode가 복제본 ui 좌표에 더하는 오프셋 px (원본과 겹치지 않게) */
export const DUPLICATE_UI_OFFSET = 32;
/** toSequence 기본 시퀀스 id (옵션 미지정 시) */
export const FLOW_GRAPH_SEQUENCE_ID = 'flow-graph';

/** defaultNodeFor 기본값 (UX_DESIGN §3.5(B) 노드 폼 초기값) */
export const DEFAULT_MOVE_DURATION_SEC = 1;
export const DEFAULT_WAIT_DURATION_SEC = 1;
export const DEFAULT_GRIPPER_DURATION_SEC = 0.5;
export const DEFAULT_COLLISION_TIMEOUT_SEC = 5;
export const DEFAULT_GOTO_TIMES = 1;

// ── 타입 (UX_DESIGN §6과 1:1) ───────────────────────────────────────

export type FlowNodeOrigin = 'generated' | 'manual' | 'modified';
export type FlowNodeStatus = 'pending' | 'active' | 'done' | 'error';
export type FlowEdgeKind = 'seq' | 'loop';

export interface FlowNode {
  id: string;
  kind: ControlStepKind;
  /** 해당 step의 필드 값 (kind/enabled/note 제외 — 이 세 필드는 노드 자체가 든다) */
  params: Record<string, unknown>;
  /** false면 실행 제외(순서 유지) — 직렬화 시 enabled:false로 나간다 */
  enabled: boolean;
  /** 출처 배지: 플래너 생성(generated) / 사용자 작성(manual) / 생성 후 편집(modified) */
  origin: FlowNodeOrigin;
  note?: string;
  /** 캔버스 좌표 — 레이아웃 전용, 실행/직렬화 무관 */
  ui: { x: number; y: number };
  /** 런타임 실행 상태 — 순수 표현, 직렬화 무관 */
  status?: FlowNodeStatus;
}

export interface FlowEdge {
  from: string;
  to: string;
  kind: FlowEdgeKind; // 순차 | goto 백엣지
}

export interface FlowGraph {
  nodes: FlowNode[];
  edges: FlowEdge[];
  robot: string; // 기본 대상 로봇
}

// ── step kind 표 (types.ts와 컴파일 타임 동기 — kind 추가 시 여기가 깨진다) ──

const STEP_KIND_FLAGS: Record<ControlStepKind, true> = {
  moveJoints: true,
  setJoints: true,
  gripper: true,
  wait: true,
  waitForCollision: true,
  label: true,
  goto: true,
  moveToPose: true,
};

/** 알려진 모든 ControlStep kind (팔레트/검사용) */
export const CONTROL_STEP_KINDS = Object.keys(STEP_KIND_FLAGS) as readonly ControlStepKind[];

const STEP_KIND_SET: ReadonlySet<string> = new Set(CONTROL_STEP_KINDS);

// ── 내부 유틸 (순수 — 입력을 절대 변형하지 않는다) ───────────────────

/** params 깊은 복사. kind/enabled/note 키는 노드 필드와 이중 진실이 되므로 제거한다. */
function cloneParams(params: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(params)) {
    if (key === 'kind' || key === 'enabled' || key === 'note') continue;
    out[key] = structuredClone(value);
  }
  return out;
}

/** 노드 깊은 복사 (params 포함). undefined 옵션 키는 만들지 않는다. */
function cloneNode(node: FlowNode): FlowNode {
  const copy: FlowNode = {
    id: node.id,
    kind: node.kind,
    params: cloneParams(node.params),
    enabled: node.enabled,
    origin: node.origin,
    ui: { x: node.ui.x, y: node.ui.y },
  };
  if (node.note !== undefined) copy.note = node.note;
  if (node.status !== undefined) copy.status = node.status;
  return copy;
}

/** 사용자 편집이 닿은 노드의 출처 배지: generated → modified (UX §3.4 '수정됨') */
function markEdited(origin: FlowNodeOrigin): FlowNodeOrigin {
  return origin === 'generated' ? 'modified' : origin;
}

/** label 노드의 이름 (params.name) — 문자열이 아니면 undefined */
function labelNameOf(node: FlowNode): string | undefined {
  if (node.kind !== 'label') return undefined;
  const name = node.params['name'];
  return typeof name === 'string' ? name : undefined;
}

/** goto 노드의 대상 label 이름 (params.label) — 문자열이 아니면 undefined */
function gotoTargetOf(node: FlowNode): string | undefined {
  if (node.kind !== 'goto') return undefined;
  const label = node.params['label'];
  return typeof label === 'string' ? label : undefined;
}

// ── 자동 배치 (순수 결정론 — UX §6 "ui.x/y는 표현 전용") ─────────────

/**
 * 체인 인덱스 → 캔버스 좌표. 가로 체인, LAYOUT_ROW_LENGTH개마다 다음 줄로 wrap.
 * 순수 결정론 — 같은 인덱스는 항상 같은 좌표다.
 */
export function layoutPosition(index: number): { x: number; y: number } {
  return {
    x: (index % LAYOUT_ROW_LENGTH) * LAYOUT_X_GAP,
    y: Math.floor(index / LAYOUT_ROW_LENGTH) * LAYOUT_Y_GAP,
  };
}

// ── 엣지 파생 (단일 진실 = 노드 순서 + goto params) ──────────────────

/**
 * 노드 배열에서 엣지를 재유도한다.
 * - seq: 체인 순서를 따라 인접 노드 쌍마다 1개 (비활성 노드도 순서를 유지하므로 포함).
 * - loop: 각 goto 노드 → 같은 이름의 **첫 번째** label 노드 (player의 "중복 label은
 *   첫 번째가 이긴다" 규칙과 동일). 대상 label이 없으면 엣지를 만들지 않는다
 *   (직렬화 가능한 그래프에서는 발생하지 않는다 — validateSequence가 거부).
 * - 비활성 goto도 loop 엣지를 만든다 — 엣지는 구조 뷰이며 실행 여부는 enabled가 든다.
 */
export function deriveEdges(nodes: readonly FlowNode[]): FlowEdge[] {
  const edges: FlowEdge[] = [];
  for (let i = 1; i < nodes.length; i += 1) {
    const prev = nodes[i - 1];
    const curr = nodes[i];
    if (prev && curr) edges.push({ from: prev.id, to: curr.id, kind: 'seq' });
  }

  const firstLabelIdByName = new Map<string, string>();
  for (const node of nodes) {
    const name = labelNameOf(node);
    if (name !== undefined && !firstLabelIdByName.has(name)) {
      firstLabelIdByName.set(name, node.id);
    }
  }
  for (const node of nodes) {
    const target = gotoTargetOf(node);
    if (target === undefined) continue;
    const labelId = firstLabelIdByName.get(target);
    if (labelId !== undefined) edges.push({ from: node.id, to: labelId, kind: 'loop' });
  }
  return edges;
}

// ── fromSequence — 플래너 출력 로드/편집용 ──────────────────────────

export interface FromSequenceOptions {
  /** 노드 출처 배지. 기본 'manual' — 플래너 출력을 로드할 때 'generated'로 지정 */
  origin?: FlowNodeOrigin;
}

/**
 * ControlSequence → FlowGraph. 노드는 steps 순서 그대로이며 id는 'n1','n2',...로
 * 호출마다 동일하게(안정적으로) 부여된다. params는 step 필드에서 kind/enabled/note를
 * 뺀 깊은 복사본이고, enabled 기본값은 true다. ui는 layoutPosition 자동 배치.
 *
 * 시퀀스의 id/loop는 그래프에 실리지 않는다 — 그래프는 steps의 뷰이며, 시퀀스
 * 메타데이터는 UI가 보관했다가 toSequence 옵션으로 되돌린다.
 */
export function fromSequence(seq: ControlSequence, opts?: FromSequenceOptions): FlowGraph {
  const origin = opts?.origin ?? 'manual';
  const nodes: FlowNode[] = seq.steps.map((step, index) => {
    const params: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(step)) {
      if (key === 'kind' || key === 'enabled' || key === 'note') continue;
      params[key] = structuredClone(value);
    }
    const node: FlowNode = {
      id: `n${index + 1}`,
      kind: step.kind,
      params,
      enabled: step.enabled !== false,
      origin,
      ui: layoutPosition(index),
    };
    if (step.note !== undefined) node.note = step.note;
    return node;
  });
  return { nodes, edges: deriveEdges(nodes), robot: seq.robot };
}

// ── toSequence — 실행/저장용 ────────────────────────────────────────

export interface ToSequenceOptions {
  /** 시퀀스 id (기본 FLOW_GRAPH_SEQUENCE_ID). fromSequence로 잃은 원본 id 복원용 */
  id?: string;
  /** 전체 반복 플래그 — 지정 시에만 출력에 포함된다 */
  loop?: boolean;
}

/**
 * FlowGraph → ControlSequence. 노드 순서 = steps 순서. 각 step은
 * params + kind + enabled(true면 생략 — 정규형) + note(있을 때만)로 조립된다.
 *
 * **알 수 없는 kind(구조적으로 불가능한 그래프)에서만 throw한다.** 파라미터 값 오류
 * (음수 duration, 없는 label 참조 등)는 여기서 잡지 않는다 — validateSequence의 소관
 * 이며, UI는 반드시 serializeGraph(= toSequence + validateSequence)만 사용해야 한다
 * (불변식 §2.8). 이 함수의 반환값은 검증 전까지 신뢰하지 않는다.
 */
export function toSequence(graph: FlowGraph, opts?: ToSequenceOptions): ControlSequence {
  const steps = graph.nodes.map((node, index) => {
    if (!STEP_KIND_SET.has(node.kind)) {
      throw new Error(
        `flow-graph: nodes[${index}] (id '${node.id}')의 kind '${String(node.kind)}'은(는) ` +
          '알 수 없는 step 종류입니다 — 구조적으로 직렬화할 수 없는 그래프입니다',
      );
    }
    const raw: Record<string, unknown> = cloneParams(node.params);
    raw['kind'] = node.kind;
    if (!node.enabled) raw['enabled'] = false;
    if (node.note !== undefined) raw['note'] = node.note;
    // params는 Record<string, unknown>이므로 여기서는 형태만 조립한다. 값의 정합성은
    // validateSequence(serializeGraph 경유)가 보증하는 신뢰 경계다 — 위 doc 참조.
    return raw as unknown as ControlStep;
  });
  const sequence: ControlSequence = {
    id: opts?.id ?? FLOW_GRAPH_SEQUENCE_ID,
    robot: graph.robot,
    steps,
  };
  if (opts?.loop !== undefined) sequence.loop = opts.loop;
  return sequence;
}

// ── serializeGraph — UI가 사용해야 하는 유일한 직렬화 경로 (§2.8) ────

export type SerializeGraphResult =
  | { ok: true; sequence: ControlSequence }
  | { ok: false; errors: string[] };

/**
 * toSequence + validateSequence. scene을 주면 참조 무결성(robot/entity/joint)까지 검사.
 * throw하지 않는다 — 구조 불가능 그래프(toSequence throw)도 { ok:false }로 흡수한다.
 * 그래프를 실행/저장/JSON 표시로 내보내는 **모든** 경로는 이 함수를 거쳐야 한다.
 */
export function serializeGraph(
  graph: FlowGraph,
  scene?: SceneSpec,
  opts?: ToSequenceOptions,
): SerializeGraphResult {
  let sequence: ControlSequence;
  try {
    sequence = toSequence(graph, opts);
  } catch (error) {
    return { ok: false, errors: [error instanceof Error ? error.message : String(error)] };
  }
  const result = validateSequence(sequence, scene);
  if (!result.ok) return { ok: false, errors: result.errors };
  return { ok: true, sequence: result.value };
}

// ── 편집 연산 (전부 순수 — ok 결과는 직렬화 가능성이 보장된다) ───────

export type FlowEditResult =
  | { ok: true; graph: FlowGraph }
  | { ok: false; errors: string[] };

/**
 * 편집 결과 마감: 엣지 재유도 + 구조 직렬화 검증(§2.8 게이트).
 * 검증 실패면 편집을 거부한다 — 호출 전 그래프가 유효했다면 ok 결과도 항상 유효하다.
 */
function finishEdit(robot: string, nodes: FlowNode[]): FlowEditResult {
  const graph: FlowGraph = { nodes, edges: deriveEdges(nodes), robot };
  const serialized = serializeGraph(graph);
  if (!serialized.ok) return { ok: false, errors: serialized.errors };
  return { ok: true, graph };
}

function indexOfNode(graph: FlowGraph, nodeId: string): number {
  return graph.nodes.findIndex((node) => node.id === nodeId);
}

function nodeNotFoundError(nodeId: string): FlowEditResult {
  return { ok: false, errors: [`그래프에 id '${nodeId}' 노드가 없습니다`] };
}

/**
 * 다음 노드 id 발급: 'n{k}' 중 최대 숫자 + 1 (충돌 시 증가). 결정론적.
 * fromSequence의 'n1','n2',... 규약과 이어진다.
 */
export function nextNodeId(nodes: readonly FlowNode[]): string {
  const used = new Set(nodes.map((node) => node.id));
  let max = 0;
  for (const id of used) {
    const match = /^n(\d+)$/.exec(id);
    const digits = match?.[1];
    if (digits !== undefined) max = Math.max(max, Number(digits));
  }
  let candidate = max + 1;
  while (used.has(`n${candidate}`)) candidate += 1;
  return `n${candidate}`;
}

/**
 * 노드를 체인 내 다른 위치로 이동한다(드래그 재정렬 — UX §3.4).
 * toIndex는 결과 배열에서의 최종 인덱스(0..length-1)다.
 */
export function moveNode(graph: FlowGraph, nodeId: string, toIndex: number): FlowEditResult {
  const from = indexOfNode(graph, nodeId);
  if (from < 0) return nodeNotFoundError(nodeId);
  if (!Number.isInteger(toIndex) || toIndex < 0 || toIndex >= graph.nodes.length) {
    return {
      ok: false,
      errors: [
        `이동 위치 ${toIndex}이(가) 범위를 벗어났습니다 (0..${graph.nodes.length - 1})`,
      ],
    };
  }
  const nodes = [...graph.nodes];
  const moved = nodes.splice(from, 1);
  const target = moved[0];
  if (!target) return nodeNotFoundError(nodeId); // noUncheckedIndexedAccess 방어 — 도달 불가
  nodes.splice(toIndex, 0, target);
  return finishEdit(graph.robot, nodes);
}

/**
 * 노드 삽입 (팔레트/엣지 ＋ — UX §3.4). atIndex는 0..length (끝 삽입 허용).
 * node.id가 빈 문자열이면 nextNodeId로 보정(defaultNodeFor의 자리표시자 규약),
 * 이미 존재하는 id면 거부한다. 입력 node는 깊은 복사되어 들어간다(순수성).
 */
export function insertNode(graph: FlowGraph, node: FlowNode, atIndex: number): FlowEditResult {
  if (!Number.isInteger(atIndex) || atIndex < 0 || atIndex > graph.nodes.length) {
    return {
      ok: false,
      errors: [`삽입 위치 ${atIndex}이(가) 범위를 벗어났습니다 (0..${graph.nodes.length})`],
    };
  }
  const copy = cloneNode(node);
  if (copy.id === '') copy.id = nextNodeId(graph.nodes);
  else if (indexOfNode(graph, copy.id) >= 0) {
    return { ok: false, errors: [`노드 id '${copy.id}'이(가) 이미 그래프에 존재합니다`] };
  }
  const nodes = [...graph.nodes];
  nodes.splice(atIndex, 0, copy);
  return finishEdit(graph.robot, nodes);
}

/**
 * 노드 삭제. 앞뒤 엣지는 deriveEdges가 자동 재연결한다 (UX §3.4 "삭제 시 자동 재연결").
 * - goto가 참조하는 label은 같은 이름의 다른 label이 남지 않는 한 삭제를 거부한다
 *   (§2.8 — 삭제하면 goto 대상이 사라져 직렬화 불가).
 * - 마지막 남은 노드는 삭제할 수 없다 (시퀀스는 최소 1 step — DATA_MODEL §8).
 */
export function removeNode(graph: FlowGraph, nodeId: string): FlowEditResult {
  const index = indexOfNode(graph, nodeId);
  if (index < 0) return nodeNotFoundError(nodeId);
  if (graph.nodes.length <= 1) {
    return {
      ok: false,
      errors: ['마지막 남은 노드는 삭제할 수 없습니다 — 시퀀스에는 최소 1개의 step이 필요합니다'],
    };
  }
  const target = graph.nodes[index];
  if (!target) return nodeNotFoundError(nodeId); // noUncheckedIndexedAccess 방어 — 도달 불가

  const name = labelNameOf(target);
  if (name !== undefined) {
    const remaining = graph.nodes.filter((node) => node.id !== nodeId);
    const stillHasLabel = remaining.some((node) => labelNameOf(node) === name);
    const referenced = remaining.some((node) => gotoTargetOf(node) === name);
    if (referenced && !stillHasLabel) {
      return {
        ok: false,
        errors: [
          `label '${name}'을(를) 참조하는 goto 노드가 있어 삭제할 수 없습니다 — ` +
            '먼저 goto 노드를 삭제하거나 대상 label을 바꾸세요',
        ],
      };
    }
  }
  const nodes = graph.nodes.filter((node) => node.id !== nodeId);
  return finishEdit(graph.robot, nodes);
}

/**
 * 노드 복제 (Ctrl/Cmd+D — UX §3.4). 원본 바로 뒤에 새 id로 삽입된다.
 * 복제본은 사용자가 만든 것이므로 origin 'manual', 런타임 status는 버린다.
 * ui는 원본에서 DUPLICATE_UI_OFFSET만큼 비껴 놓는다. label 복제는 같은 이름의 중복
 * label을 만들 수 있으나 유효하다 — goto/player는 첫 번째 label을 따른다.
 */
export function duplicateNode(graph: FlowGraph, nodeId: string): FlowEditResult {
  const index = indexOfNode(graph, nodeId);
  if (index < 0) return nodeNotFoundError(nodeId);
  const source = graph.nodes[index];
  if (!source) return nodeNotFoundError(nodeId); // noUncheckedIndexedAccess 방어 — 도달 불가

  const copy = cloneNode(source);
  copy.id = nextNodeId(graph.nodes);
  copy.origin = 'manual';
  delete copy.status;
  copy.ui = { x: source.ui.x + DUPLICATE_UI_OFFSET, y: source.ui.y + DUPLICATE_UI_OFFSET };

  const nodes = [...graph.nodes];
  nodes.splice(index + 1, 0, copy);
  return finishEdit(graph.robot, nodes);
}

/**
 * 활성/비활성 토글 (UX §3.4). 비활성 노드는 직렬화에 enabled:false로 남고 실행에서
 * 제외되나 순서는 유지된다. 비활성 label도 goto 대상으로 유효하다(위치 마커) —
 * 직렬화 검증은 enabled와 무관하게 label 존재만 본다.
 */
export function setNodeEnabled(graph: FlowGraph, nodeId: string, enabled: boolean): FlowEditResult {
  const index = indexOfNode(graph, nodeId);
  if (index < 0) return nodeNotFoundError(nodeId);
  const nodes = graph.nodes.map((node) =>
    node.id === nodeId ? { ...cloneNode(node), enabled, origin: markEdited(node.origin) } : node,
  );
  return finishEdit(graph.robot, nodes);
}

/**
 * 노드 파라미터 전체 교체 (인스펙터 폼 제출 — UX §3.5(B)). params 안의
 * kind/enabled/note 키는 노드 필드와의 이중 진실을 막기 위해 무시된다.
 * 무효한 파라미터(음수 duration, 없는 label 등)는 검증 오류(한국어)와 함께 거부된다.
 */
export function updateNodeParams(
  graph: FlowGraph,
  nodeId: string,
  params: Record<string, unknown>,
): FlowEditResult {
  const index = indexOfNode(graph, nodeId);
  if (index < 0) return nodeNotFoundError(nodeId);
  const nodes = graph.nodes.map((node) =>
    node.id === nodeId
      ? { ...cloneNode(node), params: cloneParams(params), origin: markEdited(node.origin) }
      : node,
  );
  return finishEdit(graph.robot, nodes);
}

// ── remapEntityId — 씬 엔티티 개명 동기화 (편집 연산이 아닌 시스템 동기화) ──

/**
 * 씬 엔티티 개명(oldId→newId)을 그래프에 반영한 사본을 돌려준다. UI 통합자가
 * SceneEditor rename 통지에서 호출한다 — 개명 후에도 그래프가 씬 참조 무결성을
 * 유지해 편집/재생이 잠기지 않게 한다 (§2.8 잠김 방지).
 * - 치환 대상: graph.robot(기본 로봇) · step params의 robot · waitForCollision의 between.
 * - 시스템 동기화이므로 사용자 편집 배지(origin)는 바꾸지 않는다.
 * - 치환할 참조가 하나도 없으면 **입력 그래프를 그대로**(동일 참조) 반환한다 —
 *   호출자가 변경 여부를 동일성 비교로 판별할 수 있다.
 * - 순수 치환만 한다: 직렬화 가능성(씬 참조 무결성)은 호출자가 serializeGraph(scene)로
 *   재확인해야 한다 (씬 검사는 씬을 아는 지점의 소관 — 모듈 헤더 원칙).
 */
export function remapEntityId(graph: FlowGraph, oldId: string, newId: string): FlowGraph {
  if (oldId === newId) return graph;
  let anyNodeChanged = false;
  const nodes = graph.nodes.map((node) => {
    let nodeChanged = false;
    const params = cloneParams(node.params);
    if (params['robot'] === oldId) {
      params['robot'] = newId;
      nodeChanged = true;
    }
    const between: unknown = params['between'];
    if (Array.isArray(between) && between.some((v: unknown) => v === oldId)) {
      params['between'] = between.map((v: unknown) => (v === oldId ? newId : v));
      nodeChanged = true;
    }
    if (!nodeChanged) return node;
    anyNodeChanged = true;
    return { ...cloneNode(node), params };
  });
  const robotChanged = graph.robot === oldId;
  if (!anyNodeChanged && !robotChanged) return graph;
  const nextNodes = anyNodeChanged ? nodes : graph.nodes;
  return {
    nodes: nextNodes,
    edges: deriveEdges(nextNodes),
    robot: robotChanged ? newId : graph.robot,
  };
}

// ── defaultNodeFor — 노드 팔레트 기본값 (UX §3.4 ＋ 삽입) ────────────

export interface DefaultNodeContext {
  /** 기본 대상 로봇 (FlowGraph.robot) — step.robot은 생략해 시퀀스 기본을 따른다 */
  robot: string;
  /** 씬의 엔티티 id 목록 (waitForCollision 기본 쌍) */
  entityIds: string[];
  /** 그래프에 존재하는 label 이름 목록 (goto 기본 대상 / label 이름 충돌 회피) */
  labels: string[];
}

/** ctx.labels와 충돌하지 않는 첫 'L{n}' 이름 */
function freshLabelName(labels: readonly string[]): string {
  const used = new Set(labels);
  let n = 1;
  while (used.has(`L${n}`)) n += 1;
  return `L${n}`;
}

/**
 * step 종류별 합리적 기본값 노드를 만든다 (팔레트 삽입용).
 * 반환 노드의 id는 빈 문자열(자리표시자) — insertNode가 nextNodeId로 보정한다.
 * ui는 (0,0) — 배치는 삽입 위치를 아는 호출자(UI) 소관이다.
 *
 * throw (한국어): goto인데 label이 하나도 없을 때, waitForCollision인데 엔티티가
 * 2개 미만일 때 — 팔레트 UI는 이 경우 해당 항목을 비활성화해야 한다.
 */
export function defaultNodeFor(kind: ControlStepKind, ctx: DefaultNodeContext): FlowNode {
  let params: Record<string, unknown>;
  switch (kind) {
    case 'moveJoints':
      params = { targets: {}, durationSec: DEFAULT_MOVE_DURATION_SEC, easing: 'easeInOut' };
      break;
    case 'setJoints':
      params = { targets: {} };
      break;
    case 'gripper':
      params = { state: 'close', durationSec: DEFAULT_GRIPPER_DURATION_SEC };
      break;
    case 'wait':
      params = { durationSec: DEFAULT_WAIT_DURATION_SEC };
      break;
    case 'waitForCollision': {
      const [first, second] = ctx.entityIds;
      if (first === undefined || second === undefined) {
        throw new Error(
          'waitForCollision 노드를 추가하려면 씬에 엔티티가 2개 이상 필요합니다',
        );
      }
      params = { between: [first, second], timeoutSec: DEFAULT_COLLISION_TIMEOUT_SEC };
      break;
    }
    case 'label':
      params = { name: freshLabelName(ctx.labels) };
      break;
    case 'goto': {
      const firstLabel = ctx.labels[0];
      if (firstLabel === undefined) {
        throw new Error('goto 노드를 추가하려면 먼저 label 노드가 필요합니다');
      }
      params = { label: firstLabel, times: DEFAULT_GOTO_TIMES };
      break;
    }
    case 'moveToPose':
      params = { target: { position: [0, 0, 0] }, durationSec: DEFAULT_MOVE_DURATION_SEC };
      break;
  }
  return {
    id: '',
    kind,
    params,
    enabled: true,
    origin: 'manual',
    ui: { x: 0, y: 0 },
  };
}
