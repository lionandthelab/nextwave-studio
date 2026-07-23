// ui/flow-graph/canvas.ts — n8n형 노드 캔버스 (Phase 8, UX_DESIGN §3.4 · §6)
//
// ControlSequence의 FlowGraph 뷰를 SVG 체인으로 그리고 편집한다.
//
// 계약 (불변식 §2.8 — 그래프 편집은 직렬화 불가능한 상태를 만들 수 없다):
// - 이 모듈은 그래프를 **절대 직접 변형하지 않는다.** 모든 편집(재정렬/삽입/삭제/복제)은
//   deps.applyOp(op)로 흐르고, op는 schema/flow-graph의 순수 편집 연산이다 — ok가 아니면
//   통합자(main.ts)가 한국어 오류 토스트를 띄우고 그래프는 그대로 남는다.
// - 렌더링은 (graph, statuses, selection, viewport)의 순수 함수다 — 변경 시 전체
//   재렌더(MVP 규모 ≤ 64 노드). 리스너는 마운트 시 1세트만 svg 루트에 위임 부착하고
//   render()는 DOM만 재구축한다(리스너 누수 없음).
// - 노드 좌표는 체인 순서에서 결정론적으로 계산한다(가로 1행 체인). FlowNode.ui.x/y는
//   순수 표현 상태(UX §6)로 실행/직렬화에 영향이 없으므로 MVP 캔버스는 자체 레이아웃을
//   쓴다 — 드래그 재정렬의 "drop x → 삽입 인덱스" 계산(§3.4)이 1행 체인을 전제한다.
//
// 상호작용 (UX §3.4): 클릭 선택(5px 임계로 드래그와 구분) · 드래그 재정렬(고스트 +
// 삽입선 프리뷰) · 엣지/체인 끝 ＋ → 팔레트 팝오버 삽입 · Del 삭제 · Ctrl/Cmd+D 복제 ·
// 팬(스페이스/휠버튼/배경 드래그) · 휠 줌(0.4–2.0, 커서 중심) · fit · 미니맵.
//
// 계층 규칙 (CLAUDE.md §3): ui → schema. core/render를 import하지 않는다.

import {
  defaultNodeFor,
  duplicateNode,
  insertNode,
  moveNode,
  removeNode,
} from '../../schema/flow-graph';
import type { FlowGraph, FlowNode } from '../../schema/flow-graph';
import {
  COLOR,
  FONT,
  RADIUS,
  SHADOW,
  SPACE,
  ensureThemeStyles,
  makeButton,
  styled,
} from '../theme';
import {
  PALETTE_GROUPS,
  kindMeta,
  nodeSummary,
  originBadge,
  statusColor,
  statusLabelKo,
  truncateText,
} from './node-render';
import type { NodeRunStatus } from './node-render';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4) ─────────────────────────────

/** 휠 줌 하한/상한 (Phase 8 요구: 0.4–2.0) */
export const ZOOM_MIN = 0.4;
export const ZOOM_MAX = 2.0;
/** 휠 1노치당 줌 배율 */
const ZOOM_WHEEL_FACTOR = 1.1;
/** 툴바 ＋/− 버튼당 줌 배율 */
const ZOOM_BUTTON_FACTOR = 1.2;
/** 클릭 vs 드래그 구분 임계 (px — Phase 8 요구: 5px) */
export const DRAG_THRESHOLD_PX = 5;

/** 노드 크기/체인 배치 (월드 좌표 px) */
export const NODE_W = 168;
export const NODE_H = 54;
export const NODE_GAP_X = 56;
export const CHAIN_MARGIN_X = 40;
export const CHAIN_Y = 64;
/** 체인 bounds 여유 — 위/아래는 goto 루프 아크가 지나간다 */
const CHAIN_BOUNDS_PAD_X = 24;
const CHAIN_BOUNDS_PAD_Y = 70;
/** goto 루프 아크 기본 높이 + 점프 거리당 추가 높이 */
const LOOP_ARC_BASE = 46;
const LOOP_ARC_PER_SPAN = 8;
const LOOP_ARC_MAX_SPAN = 6;

/** fit-to-view 화면 여백 */
const FIT_PADDING_PX = 40;
/** 배경 그리드 점 간격 */
const GRID_SIZE_PX = 24;

/** 미니맵 (우하단 개요 — 저렴하게) */
const MINIMAP_W = 150;
const MINIMAP_H = 56;
const MINIMAP_PAD = 4;
const MINIMAP_MAX_SCALE = 0.25;

/** 노드 텍스트 절단 예산 (SVG text — node-render.truncateText) */
const LABEL_MAX_CHARS = 15;
const SUMMARY_MAX_CHARS = 19;

/** ＋ 버튼 반지름 / 삽입선 상하 돌출 */
const PLUS_R = 9;
const INSERT_LINE_OVERHANG = 12;

/** 팔레트 팝오버 크기 */
const PALETTE_W = 190;
const PALETTE_MAX_H = 250;

// ── 순수 헬퍼 (DOM 비의존 — canvas.test.ts 대상) ────────────────────

export interface CanvasViewport {
  /** 월드 원점의 화면 오프셋 px (screen = world × zoom + offset) */
  x: number;
  y: number;
  zoom: number;
}

export interface WorldBounds {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

/** 줌 클램프 (0.4–2.0). NaN은 하한으로 방어. */
export function clampZoom(zoom: number): number {
  if (Number.isNaN(zoom)) return ZOOM_MIN;
  return Math.min(Math.max(zoom, ZOOM_MIN), ZOOM_MAX);
}

/** 커서 중심 줌: (cursorX, cursorY) 화면점 아래의 월드점이 줌 후에도 그 자리에 남는다 */
export function zoomAt(
  vp: CanvasViewport,
  cursorX: number,
  cursorY: number,
  factor: number,
): CanvasViewport {
  const zoom = clampZoom(vp.zoom * factor);
  const worldX = (cursorX - vp.x) / vp.zoom;
  const worldY = (cursorY - vp.y) / vp.zoom;
  return { x: cursorX - worldX * zoom, y: cursorY - worldY * zoom, zoom };
}

/** 팬: 화면 px 오프셋 가산 (줌 불변) */
export function panBy(vp: CanvasViewport, dxPx: number, dyPx: number): CanvasViewport {
  return { x: vp.x + dxPx, y: vp.y + dyPx, zoom: vp.zoom };
}

/**
 * fit-to-view: bounds가 화면(viewW×viewH, 여백 paddingPx)에 들어오는 최대 줌으로
 * 중앙 정렬한다. 줌은 [ZOOM_MIN, ZOOM_MAX] 클램프. bounds가 없거나 화면 크기가 0이면
 * 항등 뷰포트를 돌려준다.
 */
export function fitViewport(
  bounds: WorldBounds | null,
  viewW: number,
  viewH: number,
  paddingPx: number,
): CanvasViewport {
  if (bounds === null || viewW <= 0 || viewH <= 0) return { x: 0, y: 0, zoom: 1 };
  const boundsW = Math.max(bounds.maxX - bounds.minX, 1);
  const boundsH = Math.max(bounds.maxY - bounds.minY, 1);
  const zoom = clampZoom(
    Math.min((viewW - paddingPx * 2) / boundsW, (viewH - paddingPx * 2) / boundsH),
  );
  const centerX = (bounds.minX + bounds.maxX) / 2;
  const centerY = (bounds.minY + bounds.maxY) / 2;
  return { x: viewW / 2 - centerX * zoom, y: viewH / 2 - centerY * zoom, zoom };
}

/** 체인 i번째 노드의 좌상단 x (월드) */
export function chainNodeX(index: number): number {
  return CHAIN_MARGIN_X + index * (NODE_W + NODE_GAP_X);
}

/** 체인 i번째 노드의 중심 x (월드) */
export function chainCenterX(index: number): number {
  return chainNodeX(index) + NODE_W / 2;
}

/** 체인 전체 bounds (루프 아크 여유 포함). 노드가 없으면 null. */
export function chainBounds(count: number): WorldBounds | null {
  if (count <= 0) return null;
  return {
    minX: CHAIN_MARGIN_X - CHAIN_BOUNDS_PAD_X,
    minY: CHAIN_Y - CHAIN_BOUNDS_PAD_Y,
    maxX: chainNodeX(count - 1) + NODE_W + CHAIN_BOUNDS_PAD_X,
    maxY: CHAIN_Y + NODE_H + CHAIN_BOUNDS_PAD_Y,
  };
}

/**
 * drop x → 삽입 인덱스 (0..n): 중심 x가 dropX보다 왼쪽인 노드 수.
 * 노드의 왼쪽 절반에 놓으면 그 앞, 오른쪽 절반이면 그 뒤 — "가장 가까운 두 노드 사이"
 * (UX §3.4 드래그 재정렬). centersX는 체인 순서(오름차순)를 전제한다.
 */
export function insertionIndexFromPoint(centersX: readonly number[], dropX: number): number {
  let index = 0;
  for (const centerX of centersX) {
    if (centerX < dropX) index += 1;
  }
  return index;
}

/**
 * 드래그 재정렬의 최종 인덱스: fromIndex 노드를 뺀 나머지 체인에 대한 삽입 인덱스
 * (= moveNode의 toIndex — 결과 배열에서의 최종 위치, 0..n-1).
 * 결과가 fromIndex와 같으면 순서 변화가 없는 no-op 드롭이다.
 */
export function reorderTargetIndex(
  centersX: readonly number[],
  fromIndex: number,
  dropX: number,
): number {
  const rest = centersX.filter((_, i) => i !== fromIndex);
  return insertionIndexFromPoint(rest, dropX);
}

/**
 * 삽입선 프리뷰 x (월드): fromIndex를 뺀 체인에서 finalIndex 위치의 틈.
 * count는 전체 노드 수(드래그 중인 노드 포함). 노드가 1개뿐이면 첫 틈을 돌려준다.
 */
export function insertionLineX(fromIndex: number, finalIndex: number, count: number): number {
  const remainingCount = count - 1;
  if (remainingCount <= 0) return chainNodeX(0) - NODE_GAP_X / 2;
  if (finalIndex >= remainingCount) {
    // 체인 끝 — 마지막 남은 노드의 오른쪽
    const lastOriginal = fromIndex === count - 1 ? count - 2 : count - 1;
    return chainNodeX(lastOriginal) + NODE_W + NODE_GAP_X / 2;
  }
  // remaining[finalIndex]의 원래 인덱스 앞 틈
  const originalIndex = finalIndex < fromIndex ? finalIndex : finalIndex + 1;
  return chainNodeX(originalIndex) - NODE_GAP_X / 2;
}

// ── 공개 타입 (통합자 계약) ─────────────────────────────────────────

/** schema/flow-graph 편집 연산의 결과 형태 (FlowEditResult와 구조 호환) */
export interface FlowCanvasOpResult {
  ok: boolean;
  graph?: FlowGraph;
  errors?: string[];
}

export interface FlowCanvasDeps {
  /** 현재 그래프 진실 (통합자 소유 — 캔버스는 읽기만 한다) */
  getGraph(): FlowGraph;
  /**
   * 편집 커밋 경로 (§2.8): op는 schema/flow-graph의 순수 연산. 통합자가
   * ok → 커밋+재렌더 / 실패 → 한국어 토스트로 감싼다. 반환값은 커밋 여부.
   */
  applyOp(op: (g: FlowGraph) => FlowCanvasOpResult): boolean;
  /** 선택 변경 통지 (인스펙터 연동) — 캔버스 내 클릭/삭제로 인한 변경만 통지한다 */
  onSelectNode(id: string | null): void;
  /** defaultNodeFor 컨텍스트 (팔레트 삽입용 — goto/waitForCollision 활성 판정 포함) */
  paletteContext(): { robot: string; entityIds: string[]; labels: string[] };
}

export interface FlowCanvasHandle {
  /** (graph, statuses, selection, viewport)로 전체 재렌더 — 그래프 변경 시 호출 */
  render(): void;
  /** 실행 상태 갱신 (노드 재구축 없이 상태 점/aria만 갱신) */
  setStatuses(map: Record<string, NodeRunStatus>): void;
  /** 외부 주도 선택 (onSelectNode 에코 없음) */
  selectNode(id: string | null): void;
  /** fit-to-view (이후 그래프 변경 시 자동 맞춤 재개) */
  fit(): void;
  dispose(): void;
}

// ── SVG 헬퍼 ────────────────────────────────────────────────────────

const SVG_NS = 'http://www.w3.org/2000/svg';

function svgEl<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attrs?: Record<string, string>,
): SVGElementTagNameMap[K] {
  const el = document.createElementNS(SVG_NS, tag);
  if (attrs) {
    for (const [name, value] of Object.entries(attrs)) el.setAttribute(name, value);
  }
  return el;
}

function svgText(content: string, attrs: Record<string, string>): SVGTextElement {
  const el = svgEl('text', attrs);
  el.textContent = content;
  return el;
}

/** 배지 폭 추정 (8px 폰트 — ASCII 5px/한글 9px + 좌우 여백) */
function badgeWidthPx(text: string): number {
  let width = 8;
  for (const ch of text) width += ch.charCodeAt(0) < 256 ? 5 : 9;
  return width;
}

/** input/textarea/select/contentEditable 위 키 입력은 캔버스 단축키에서 제외 (UX §9) */
function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  const tag = target.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT';
}

// ── 캔버스 전용 스타일 (hover/펄스 — 토큰만 소비, 1회 주입) ──────────

const CANVAS_STYLE_ID = 'rsw-flow-canvas-styles';

function ensureCanvasStyles(): void {
  if (document.getElementById(CANVAS_STYLE_ID) !== null) return;
  const style = document.createElement('style');
  style.id = CANVAS_STYLE_ID;
  style.textContent = `
.rsw-fg-node { cursor: grab; }
.rsw-fg-node-body {
  fill: ${COLOR.bgRaised};
  stroke: ${COLOR.borderStrong};
  stroke-width: 1;
  transition: stroke 0.12s ease;
}
.rsw-fg-node:hover .rsw-fg-node-body { stroke: ${COLOR.borderHover}; }
.rsw-fg-node--selected .rsw-fg-node-body,
.rsw-fg-node--selected:hover .rsw-fg-node-body {
  stroke: var(--rsw-accent);
  stroke-width: 1.6;
}
@keyframes rsw-fg-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
.rsw-fg-dot--active { animation: rsw-fg-pulse 1.2s ease-in-out infinite; }
.rsw-fg-plus { cursor: pointer; }
.rsw-fg-plus circle { transition: stroke 0.12s ease, fill 0.12s ease; }
.rsw-fg-plus:hover circle { stroke: var(--rsw-accent); fill: var(--rsw-accent-soft); }
.rsw-fg-plus:hover text { fill: var(--rsw-accent-text); }
`;
  document.head.appendChild(style);
}

// ── 마운트 ──────────────────────────────────────────────────────────

/** 그리드 패턴/마커 id 충돌 방지용 마운트 시퀀스 */
let mountSeq = 0;

/** 드래그 상태 (클릭 vs 드래그는 moved 5px 임계로 구분) */
type DragState =
  | {
      mode: 'pan';
      pointerId: number;
      startX: number;
      startY: number;
      moved: boolean;
      startViewport: CanvasViewport;
    }
  | {
      mode: 'node';
      pointerId: number;
      startX: number;
      startY: number;
      moved: boolean;
      nodeId: string;
      nodeIndex: number;
    };

export function mountFlowCanvas(host: HTMLElement, deps: FlowCanvasDeps): FlowCanvasHandle {
  ensureThemeStyles();
  ensureCanvasStyles();
  mountSeq += 1;
  const gridId = `rsw-fg-grid-${mountSeq}`;
  const seqMarkerId = `rsw-fg-arrow-seq-${mountSeq}`;
  const loopMarkerId = `rsw-fg-arrow-loop-${mountSeq}`;

  // ── 뷰 상태 (ui 소유 — 순수 표현, UX §6) ──────────────────────────
  let viewport: CanvasViewport = { x: 0, y: 0, zoom: 1 };
  /** 사용자가 팬/줌을 만졌는가 — 아니면 그래프 변경 시 자동 fit을 유지한다 */
  let userAdjusted = false;
  let selection: string | null = null;
  let statuses: Record<string, NodeRunStatus> = {};
  let spaceHeld = false;
  let drag: DragState | null = null;
  /** 드래그 종료 직후 따라오는 합성 click 1회 무시 (드롭 지점의 ＋ 오발동 방지) */
  let suppressNextClick = false;
  /** 팔레트가 삽입할 체인 인덱스 (null = 닫힘) */
  let pendingInsertIndex: number | null = null;

  // 렌더 산출물 맵 (상태/선택 경량 페인트용 — render()마다 재구축)
  const nodeGroupById = new Map<string, SVGGElement>();
  const dotById = new Map<string, SVGCircleElement>();
  const ariaBaseById = new Map<string, string>();
  let ghostEl: SVGGElement | null = null;
  let insertLineEl: SVGLineElement | null = null;
  /** 미니맵 월드→미니 변환 (rebuildMinimap이 갱신) */
  let miniTransform: { scale: number; offX: number; offY: number } | null = null;

  // ── DOM 골격 ──────────────────────────────────────────────────────
  const root = styled(document.createElement('div'), {
    position: 'absolute',
    inset: '0',
    overflow: 'hidden',
    fontFamily: FONT.ui,
    fontSize: '12px',
    color: COLOR.text,
  });
  root.dataset.testid = 'flow-canvas';

  const svg = svgEl('svg', { width: '100%', height: '100%' });
  svg.style.position = 'absolute';
  svg.style.inset = '0';
  svg.style.display = 'block';

  const defs = svgEl('defs');
  const gridPattern = svgEl('pattern', {
    id: gridId,
    width: String(GRID_SIZE_PX),
    height: String(GRID_SIZE_PX),
    patternUnits: 'userSpaceOnUse',
  });
  gridPattern.appendChild(svgEl('circle', { cx: '1', cy: '1', r: '1', fill: COLOR.gridDot }));
  defs.appendChild(gridPattern);

  const seqMarker = svgEl('marker', {
    id: seqMarkerId,
    viewBox: '0 0 8 8',
    refX: '7',
    refY: '4',
    markerWidth: '7',
    markerHeight: '7',
    orient: 'auto-start-reverse',
  });
  seqMarker.appendChild(svgEl('path', { d: 'M0,0 L8,4 L0,8 Z', fill: COLOR.borderStrong }));
  defs.appendChild(seqMarker);

  const loopMarker = svgEl('marker', {
    id: loopMarkerId,
    viewBox: '0 0 8 8',
    refX: '7',
    refY: '4',
    markerWidth: '7',
    markerHeight: '7',
    orient: 'auto-start-reverse',
  });
  loopMarker.appendChild(svgEl('path', { d: 'M0,0 L8,4 L0,8 Z', fill: COLOR.accent }));
  defs.appendChild(loopMarker);
  svg.appendChild(defs);

  const gridRect = svgEl('rect', { width: '100%', height: '100%', fill: `url(#${gridId})` });
  svg.appendChild(gridRect);

  /** 월드 그룹 — viewport 변환이 걸리는 유일한 지점 */
  const worldG = svgEl('g');
  svg.appendChild(worldG);
  root.appendChild(svg);

  // ── 툴바 (fit / 줌 — UX §2 목업 [fit][−][+]) ──────────────────────
  const toolbar = styled(document.createElement('div'), {
    position: 'absolute',
    top: SPACE.md,
    right: SPACE.md,
    display: 'flex',
    alignItems: 'center',
    gap: SPACE.sm,
    padding: `${SPACE.xs} ${SPACE.sm}`,
    background: COLOR.bgBar,
    border: `1px solid ${COLOR.border}`,
    borderRadius: RADIUS.sm,
  });
  toolbar.dataset.testid = 'flow-toolbar';
  const zoomOutBtn = makeButton('−', '축소', 'flow-zoom-out');
  const zoomReadout = styled(document.createElement('span'), {
    color: COLOR.label,
    fontFamily: FONT.mono,
    fontSize: '11px',
    minWidth: '38px',
    textAlign: 'center',
  });
  zoomReadout.dataset.testid = 'flow-zoom-readout';
  const zoomInBtn = makeButton('＋', '확대', 'flow-zoom-in');
  const fitBtn = makeButton('맞춤', '전체 보기 (fit)', 'flow-fit');
  toolbar.appendChild(zoomOutBtn);
  toolbar.appendChild(zoomReadout);
  toolbar.appendChild(zoomInBtn);
  toolbar.appendChild(fitBtn);
  root.appendChild(toolbar);

  // ── 미니맵 (우하단 개요 — 클릭 점프) ──────────────────────────────
  const minimapBox = styled(document.createElement('div'), {
    position: 'absolute',
    right: SPACE.md,
    bottom: SPACE.md,
    width: `${MINIMAP_W}px`,
    height: `${MINIMAP_H}px`,
    background: COLOR.bgBar,
    border: `1px solid ${COLOR.border}`,
    borderRadius: RADIUS.sm,
    overflow: 'hidden',
    cursor: 'pointer',
    display: 'none',
  });
  minimapBox.dataset.testid = 'flow-minimap';
  minimapBox.title = '미니맵 — 클릭한 위치로 이동';
  const miniSvg = svgEl('svg', { width: '100%', height: '100%' });
  const miniNodesG = svgEl('g');
  const miniViewRect = svgEl('rect', {
    fill: COLOR.accentSoft,
    stroke: COLOR.accent,
    'stroke-width': '1',
  });
  miniSvg.appendChild(miniNodesG);
  miniSvg.appendChild(miniViewRect);
  minimapBox.appendChild(miniSvg);
  root.appendChild(minimapBox);

  // ── 빈 그래프 상태 (UX §7 "빈 플로우") ────────────────────────────
  const emptyOverlay = styled(document.createElement('div'), {
    position: 'absolute',
    inset: '0',
    display: 'none',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    gap: SPACE.md,
    color: COLOR.muted,
    textAlign: 'center',
    pointerEvents: 'none',
  });
  emptyOverlay.dataset.testid = 'flow-empty';
  const emptyAddBtn = makeButton('＋ 노드 추가', '첫 노드 삽입', 'flow-empty-add', 'accent');
  styled(emptyAddBtn, { pointerEvents: 'auto' });
  const emptyHint = document.createElement('div');
  emptyHint.textContent = '또는 자연어로 플로우 생성 — Phase 9에서 제공됩니다';
  emptyOverlay.appendChild(emptyAddBtn);
  emptyOverlay.appendChild(emptyHint);
  root.appendChild(emptyOverlay);

  // ── 노드 팔레트 팝오버 (＋ 삽입 — step 종류별 분류) ────────────────
  const paletteEl = styled(document.createElement('div'), {
    position: 'absolute',
    display: 'none',
    width: `${PALETTE_W}px`,
    maxHeight: `${PALETTE_MAX_H}px`,
    overflowY: 'auto',
    background: COLOR.bgPanel,
    border: `1px solid ${COLOR.border}`,
    borderRadius: RADIUS.md,
    boxShadow: SHADOW.panel,
    padding: SPACE.sm,
    boxSizing: 'border-box',
    zIndex: '10',
  });
  paletteEl.classList.add('ui-scroll');
  paletteEl.dataset.testid = 'flow-palette';
  paletteEl.setAttribute('role', 'menu');
  paletteEl.setAttribute('aria-label', '노드 삽입 팔레트');

  const paletteButtons: { kind: FlowNode['kind']; btn: HTMLButtonElement }[] = [];
  for (const group of PALETTE_GROUPS) {
    const header = styled(document.createElement('div'), {
      color: COLOR.label,
      fontSize: '10px',
      padding: `${SPACE.xs} ${SPACE.xs} 2px`,
      letterSpacing: '0.06em',
    });
    header.textContent = group.labelKo;
    paletteEl.appendChild(header);
    for (const kind of group.kinds) {
      const meta = kindMeta(kind);
      const btn = makeButton(
        `${meta.icon} ${meta.label}`,
        meta.descriptionKo,
        `flow-palette-${kind}`,
        'ghost',
      );
      styled(btn, { display: 'block', width: '100%', textAlign: 'left' });
      btn.addEventListener('click', () => {
        const index = pendingInsertIndex;
        closePalette();
        if (index === null) return;
        let node: FlowNode;
        try {
          node = defaultNodeFor(kind, deps.paletteContext());
        } catch {
          return; // 전제조건 미충족 kind는 버튼이 비활성 — 방어적 무시
        }
        deps.applyOp((g) => insertNode(g, node, index));
        render();
      });
      paletteEl.appendChild(btn);
      paletteButtons.push({ kind, btn });
    }
  }
  root.appendChild(paletteEl);
  host.appendChild(root);

  // ── 좌표 변환 ─────────────────────────────────────────────────────
  const rootW = (): number => root.clientWidth;
  const rootH = (): number => root.clientHeight;

  const clientToWorld = (clientX: number, clientY: number): { x: number; y: number } => {
    const rect = svg.getBoundingClientRect();
    return {
      x: (clientX - rect.left - viewport.x) / viewport.zoom,
      y: (clientY - rect.top - viewport.y) / viewport.zoom,
    };
  };

  // ── 팔레트 열기/닫기 ──────────────────────────────────────────────
  const closePalette = (): void => {
    pendingInsertIndex = null;
    paletteEl.style.display = 'none';
  };

  const openPaletteAt = (index: number, clientX: number, clientY: number): void => {
    pendingInsertIndex = index;
    const ctx = deps.paletteContext();
    for (const { kind, btn } of paletteButtons) {
      let disabledReason: string | null = null;
      if (kind === 'goto' && ctx.labels.length === 0) {
        disabledReason = 'goto를 추가하려면 먼저 label 노드가 필요합니다';
      } else if (kind === 'waitForCollision' && ctx.entityIds.length < 2) {
        disabledReason = 'waitForCollision을 추가하려면 씬에 엔티티가 2개 이상 필요합니다';
      }
      btn.disabled = disabledReason !== null;
      btn.title = disabledReason ?? kindMeta(kind).descriptionKo;
    }
    const rect = root.getBoundingClientRect();
    const left = Math.max(8, Math.min(clientX - rect.left, rect.width - PALETTE_W - 8));
    const top = Math.max(8, Math.min(clientY - rect.top + 8, Math.max(8, rect.height - PALETTE_MAX_H - 8)));
    paletteEl.style.left = `${left}px`;
    paletteEl.style.top = `${top}px`;
    paletteEl.style.display = 'block';
  };

  emptyAddBtn.addEventListener('click', () => {
    const rect = emptyAddBtn.getBoundingClientRect();
    openPaletteAt(0, rect.left, rect.bottom);
  });

  // ── 뷰포트 반영 (팬/줌 — 노드 재구축 없음) ────────────────────────
  const applyViewport = (): void => {
    const transform = `translate(${viewport.x} ${viewport.y}) scale(${viewport.zoom})`;
    worldG.setAttribute('transform', transform);
    gridPattern.setAttribute('patternTransform', transform);
    zoomReadout.textContent = `${Math.round(viewport.zoom * 100)}%`;
    updateMinimapViewRect();
  };

  const updateMinimapViewRect = (): void => {
    if (miniTransform === null) return;
    const { scale, offX, offY } = miniTransform;
    const worldX0 = (0 - viewport.x) / viewport.zoom;
    const worldY0 = (0 - viewport.y) / viewport.zoom;
    const worldX1 = (rootW() - viewport.x) / viewport.zoom;
    const worldY1 = (rootH() - viewport.y) / viewport.zoom;
    miniViewRect.setAttribute('x', String(worldX0 * scale + offX));
    miniViewRect.setAttribute('y', String(worldY0 * scale + offY));
    miniViewRect.setAttribute('width', String(Math.max(0, (worldX1 - worldX0) * scale)));
    miniViewRect.setAttribute('height', String(Math.max(0, (worldY1 - worldY0) * scale)));
  };

  const rebuildMinimap = (count: number): void => {
    const bounds = chainBounds(count);
    if (bounds === null) {
      miniTransform = null;
      minimapBox.style.display = 'none';
      return;
    }
    minimapBox.style.display = 'block';
    const boundsW = bounds.maxX - bounds.minX;
    const boundsH = bounds.maxY - bounds.minY;
    const scale = Math.min(
      (MINIMAP_W - MINIMAP_PAD * 2) / boundsW,
      (MINIMAP_H - MINIMAP_PAD * 2) / boundsH,
      MINIMAP_MAX_SCALE,
    );
    const offX = (MINIMAP_W - boundsW * scale) / 2 - bounds.minX * scale;
    const offY = (MINIMAP_H - boundsH * scale) / 2 - bounds.minY * scale;
    miniTransform = { scale, offX, offY };
    miniNodesG.replaceChildren();
    for (let i = 0; i < count; i += 1) {
      miniNodesG.appendChild(
        svgEl('rect', {
          x: String(chainNodeX(i) * scale + offX),
          y: String(CHAIN_Y * scale + offY),
          width: String(Math.max(2, NODE_W * scale)),
          height: String(Math.max(1.5, NODE_H * scale)),
          rx: '1',
          fill: COLOR.borderStrong,
        }),
      );
    }
  };

  minimapBox.addEventListener('pointerdown', (e) => {
    if (miniTransform === null) return;
    const { scale, offX, offY } = miniTransform;
    const rect = minimapBox.getBoundingClientRect();
    const worldX = (e.clientX - rect.left - offX) / scale;
    const worldY = (e.clientY - rect.top - offY) / scale;
    userAdjusted = true;
    viewport = {
      x: rootW() / 2 - worldX * viewport.zoom,
      y: rootH() / 2 - worldY * viewport.zoom,
      zoom: viewport.zoom,
    };
    applyViewport();
    e.preventDefault();
  });

  // ── 노드/엣지 그리기 ──────────────────────────────────────────────

  const drawBadge = (
    parent: SVGGElement,
    rightX: number,
    topY: number,
    text: string,
    bg: string,
    fg: string,
  ): number => {
    const width = badgeWidthPx(text);
    const x = rightX - width;
    parent.appendChild(
      svgEl('rect', {
        x: String(x),
        y: String(topY),
        width: String(width),
        height: '13',
        rx: '3',
        fill: bg,
      }),
    );
    parent.appendChild(
      svgText(text, {
        x: String(x + width / 2),
        y: String(topY + 9.5),
        'text-anchor': 'middle',
        'font-family': FONT.ui,
        'font-size': '8px',
        fill: fg,
      }),
    );
    return x;
  };

  const drawNode = (node: FlowNode, index: number, count: number): void => {
    const x = chainNodeX(index);
    const y = CHAIN_Y;
    const meta = kindMeta(node.kind);
    const status: NodeRunStatus = statuses[node.id] ?? node.status ?? 'pending';
    const summary = nodeSummary(node.kind, node.params);

    const g = svgEl('g', { class: 'rsw-fg-node', role: 'button' });
    g.dataset.fgNode = node.id;
    g.dataset.testid = 'flow-node';
    // 출처를 DOM에 노출한다 (배지 텍스트와 병행 — 게이트/자동화의 'AI'(generated) 검증용).
    g.dataset.origin = node.origin;
    if (!node.enabled) g.setAttribute('opacity', '0.5');

    // 본체 (stroke는 CSS 클래스 소유 — hover/선택 상태와 조합)
    g.appendChild(
      svgEl('rect', {
        class: 'rsw-fg-node-body',
        x: String(x),
        y: String(y),
        width: String(NODE_W),
        height: String(NODE_H),
        rx: '8',
      }),
    );
    // 종류 컬러 스트립 (좌측)
    g.appendChild(
      svgEl('rect', {
        x: String(x + 1.5),
        y: String(y + 1.5),
        width: '3.5',
        height: String(NODE_H - 3),
        rx: '2',
        fill: meta.color,
      }),
    );
    // 아이콘
    g.appendChild(
      svgText(meta.icon, {
        x: String(x + 21),
        y: String(y + NODE_H / 2 + 5),
        'text-anchor': 'middle',
        'font-size': '14px',
      }),
    );
    // 타입명 + 요약
    g.appendChild(
      svgText(truncateText(meta.label, LABEL_MAX_CHARS), {
        x: String(x + 38),
        y: String(y + 21),
        'font-family': FONT.ui,
        'font-size': '11px',
        'font-weight': '600',
        fill: COLOR.textStrong,
      }),
    );
    g.appendChild(
      svgText(truncateText(summary, SUMMARY_MAX_CHARS), {
        x: String(x + 38),
        y: String(y + 38),
        'font-family': FONT.mono,
        'font-size': '9.5px',
        fill: COLOR.muted,
      }),
    );
    // 상태 점 (우상단 — 색 + aria 텍스트 병행)
    const dot = svgEl('circle', {
      cx: String(x + NODE_W - 12),
      cy: String(y + 12),
      r: '4',
      fill: statusColor(status),
      class: status === 'active' ? 'rsw-fg-dot rsw-fg-dot--active' : 'rsw-fg-dot',
    });
    g.appendChild(dot);

    // 배지 (노드 위 오른쪽 정렬): [수정됨|AI] [비활성]
    let badgeRightX = x + NODE_W;
    if (!node.enabled) {
      badgeRightX =
        drawBadge(g, badgeRightX, y - 17, '비활성', COLOR.mutedSoft, COLOR.muted) - 4;
    }
    const origin = originBadge(node.origin);
    if (origin !== null) {
      drawBadge(
        g,
        badgeRightX,
        y - 17,
        origin,
        origin === 'AI' ? COLOR.accentSoft : COLOR.infoSoft,
        origin === 'AI' ? COLOR.accentText : COLOR.info,
      );
    }

    const ariaBase =
      `${meta.label} 노드 ${index + 1}/${count}` +
      (summary !== '' ? ` — ${summary}` : '') +
      (node.enabled ? '' : ' · 비활성');
    ariaBaseById.set(node.id, ariaBase);
    g.setAttribute('aria-label', `${ariaBase} · 상태 ${statusLabelKo(status)}`);
    g.setAttribute('aria-pressed', String(node.id === selection));
    if (node.id === selection) g.classList.add('rsw-fg-node--selected');

    nodeGroupById.set(node.id, g);
    dotById.set(node.id, dot);
    worldG.appendChild(g);
  };

  const drawPlus = (centerX: number, centerY: number, insertIndex: number): void => {
    const g = svgEl('g', { class: 'rsw-fg-plus', role: 'button' });
    g.dataset.fgPlus = String(insertIndex);
    g.dataset.testid = 'flow-plus';
    g.setAttribute('aria-label', `위치 ${insertIndex}에 노드 삽입`);
    g.appendChild(
      svgEl('circle', {
        cx: String(centerX),
        cy: String(centerY),
        r: String(PLUS_R),
        fill: COLOR.bgRaised,
        stroke: COLOR.borderStrong,
        'stroke-width': '1',
      }),
    );
    g.appendChild(
      svgText('＋', {
        x: String(centerX),
        y: String(centerY + 3.5),
        'text-anchor': 'middle',
        'font-size': '10px',
        fill: COLOR.muted,
      }),
    );
    worldG.appendChild(g);
  };

  const drawSeqEdges = (count: number): void => {
    const midY = CHAIN_Y + NODE_H / 2;
    for (let i = 1; i < count; i += 1) {
      const fromX = chainNodeX(i - 1) + NODE_W;
      const toX = chainNodeX(i);
      worldG.appendChild(
        svgEl('path', {
          d:
            `M ${fromX} ${midY} C ${fromX + NODE_GAP_X / 2} ${midY}, ` +
            `${toX - NODE_GAP_X / 2} ${midY}, ${toX} ${midY}`,
          fill: 'none',
          stroke: COLOR.borderStrong,
          'stroke-width': '1.5',
          'marker-end': `url(#${seqMarkerId})`,
        }),
      );
    }
    // 엣지 중점 ＋ (사이 삽입) + 체인 끝 ＋
    for (let i = 1; i < count; i += 1) {
      drawPlus(chainNodeX(i) - NODE_GAP_X / 2, midY, i);
    }
    drawPlus(chainNodeX(count - 1) + NODE_W + NODE_GAP_X / 2, midY, count);
  };

  const drawLoopEdges = (graph: FlowGraph): void => {
    const indexById = new Map<string, number>();
    graph.nodes.forEach((node, i) => indexById.set(node.id, i));
    for (const edge of graph.edges) {
      if (edge.kind !== 'loop') continue;
      const fromIndex = indexById.get(edge.from);
      const toIndex = indexById.get(edge.to);
      if (fromIndex === undefined || toIndex === undefined) continue;
      const span = Math.min(Math.abs(fromIndex - toIndex), LOOP_ARC_MAX_SPAN);
      const arc = LOOP_ARC_BASE + span * LOOP_ARC_PER_SPAN;
      // 뒤로 점프(루프)는 위, 앞으로 점프는 아래 곡선 (UX §3.4 "곡선 백엣지")
      const above = toIndex <= fromIndex;
      const edgeY = above ? CHAIN_Y : CHAIN_Y + NODE_H;
      const ctrlY = above ? edgeY - arc : edgeY + arc;
      const fromX = chainCenterX(fromIndex);
      const toX = chainCenterX(toIndex);
      worldG.appendChild(
        svgEl('path', {
          d: `M ${fromX} ${edgeY} C ${fromX} ${ctrlY}, ${toX} ${ctrlY}, ${toX} ${edgeY}`,
          fill: 'none',
          stroke: COLOR.accent,
          'stroke-width': '1.5',
          'stroke-dasharray': '4 4',
          'marker-end': `url(#${loopMarkerId})`,
          opacity: '0.85',
        }),
      );
    }
  };

  // ── 경량 페인트 (재구축 없이 상태/선택만) ─────────────────────────
  const paintStatuses = (): void => {
    const graph = deps.getGraph();
    for (const node of graph.nodes) {
      const dot = dotById.get(node.id);
      const g = nodeGroupById.get(node.id);
      if (!dot || !g) continue;
      const status: NodeRunStatus = statuses[node.id] ?? node.status ?? 'pending';
      dot.setAttribute('fill', statusColor(status));
      dot.setAttribute(
        'class',
        status === 'active' ? 'rsw-fg-dot rsw-fg-dot--active' : 'rsw-fg-dot',
      );
      const ariaBase = ariaBaseById.get(node.id);
      if (ariaBase !== undefined) {
        g.setAttribute('aria-label', `${ariaBase} · 상태 ${statusLabelKo(status)}`);
      }
    }
  };

  const paintSelection = (): void => {
    for (const [id, g] of nodeGroupById) {
      g.classList.toggle('rsw-fg-node--selected', id === selection);
      g.setAttribute('aria-pressed', String(id === selection));
    }
  };

  // ── 드래그 프리뷰 (고스트 + 삽입선) ───────────────────────────────
  const clearDragArtifacts = (): void => {
    ghostEl?.remove();
    ghostEl = null;
    insertLineEl?.remove();
    insertLineEl = null;
  };

  const updateNodeDragPreview = (
    d: Extract<DragState, { mode: 'node' }>,
    e: PointerEvent,
  ): void => {
    const start = clientToWorld(d.startX, d.startY);
    const now = clientToWorld(e.clientX, e.clientY);
    if (ghostEl === null) {
      const source = nodeGroupById.get(d.nodeId);
      if (source) {
        source.setAttribute('opacity', '0.3');
        const clone = source.cloneNode(true);
        if (clone instanceof SVGGElement) {
          clone.removeAttribute('data-fg-node');
          clone.removeAttribute('data-testid');
          clone.setAttribute('pointer-events', 'none');
          clone.setAttribute('opacity', '0.65');
          worldG.appendChild(clone);
          ghostEl = clone;
        }
      }
    }
    ghostEl?.setAttribute('transform', `translate(${now.x - start.x} ${now.y - start.y})`);

    const count = deps.getGraph().nodes.length;
    const centers = Array.from({ length: count }, (_, i) => chainCenterX(i));
    const finalIndex = reorderTargetIndex(centers, d.nodeIndex, now.x);
    if (count >= 2 && finalIndex !== d.nodeIndex) {
      if (insertLineEl === null) {
        insertLineEl = svgEl('line', {
          stroke: COLOR.accent,
          'stroke-width': '2',
          'stroke-linecap': 'round',
        });
        insertLineEl.dataset.testid = 'flow-insert-line';
        worldG.appendChild(insertLineEl);
      }
      const lineX = insertionLineX(d.nodeIndex, finalIndex, count);
      insertLineEl.setAttribute('x1', String(lineX));
      insertLineEl.setAttribute('x2', String(lineX));
      insertLineEl.setAttribute('y1', String(CHAIN_Y - INSERT_LINE_OVERHANG));
      insertLineEl.setAttribute('y2', String(CHAIN_Y + NODE_H + INSERT_LINE_OVERHANG));
      insertLineEl.setAttribute('visibility', 'visible');
    } else {
      insertLineEl?.setAttribute('visibility', 'hidden');
    }
  };

  // ── 전체 재렌더 — (graph, statuses, selection, viewport)의 순수 함수 ─
  const render = (): void => {
    closePalette();
    clearDragArtifacts();
    const graph = deps.getGraph();
    const nodes = graph.nodes;

    // 사라진 노드가 선택돼 있으면 해제하고 통지 (삭제 후 인스펙터 잔상 방지)
    if (selection !== null && !nodes.some((n) => n.id === selection)) {
      selection = null;
      deps.onSelectNode(null);
    }

    nodeGroupById.clear();
    dotById.clear();
    ariaBaseById.clear();
    worldG.replaceChildren();

    emptyOverlay.style.display = nodes.length === 0 ? 'flex' : 'none';
    if (nodes.length > 0) {
      drawSeqEdges(nodes.length);
      drawLoopEdges(graph);
      nodes.forEach((node, i) => drawNode(node, i, nodes.length));
    }
    rebuildMinimap(nodes.length);

    // 사용자가 팬/줌을 만지기 전에는 그래프 변화를 따라 자동 fit
    if (!userAdjusted) {
      viewport = fitViewport(chainBounds(nodes.length), rootW(), rootH(), FIT_PADDING_PX);
    }
    applyViewport();
  };

  // ── 위임 포인터/휠/키 리스너 (마운트 시 1회 — render()는 재부착하지 않는다) ─
  const onPointerDown = (e: PointerEvent): void => {
    suppressNextClick = false; // 새 제스처 시작 — 이전 드래그의 클릭 억제 잔상 해제
    if (pendingInsertIndex !== null) closePalette();
    if (e.button !== 0 && e.button !== 1) return;
    const target = e.target instanceof Element ? e.target : null;
    if (target?.closest('[data-fg-plus]')) return; // ＋는 click 핸들러 몫
    const nodeG = target?.closest('[data-fg-node]');
    const forcePan = e.button === 1 || spaceHeld;

    if (nodeG instanceof SVGElement && !forcePan) {
      const nodeId = nodeG.dataset.fgNode;
      if (nodeId === undefined) return;
      const nodeIndex = deps.getGraph().nodes.findIndex((n) => n.id === nodeId);
      if (nodeIndex < 0) return;
      drag = {
        mode: 'node',
        pointerId: e.pointerId,
        startX: e.clientX,
        startY: e.clientY,
        moved: false,
        nodeId,
        nodeIndex,
      };
    } else {
      drag = {
        mode: 'pan',
        pointerId: e.pointerId,
        startX: e.clientX,
        startY: e.clientY,
        moved: false,
        startViewport: viewport,
      };
    }
    svg.setPointerCapture(e.pointerId);
    e.preventDefault();
  };

  const onPointerMove = (e: PointerEvent): void => {
    if (drag === null || e.pointerId !== drag.pointerId) return;
    const dx = e.clientX - drag.startX;
    const dy = e.clientY - drag.startY;
    if (!drag.moved && Math.hypot(dx, dy) < DRAG_THRESHOLD_PX) return;
    drag.moved = true;
    if (drag.mode === 'pan') {
      userAdjusted = true;
      viewport = panBy(drag.startViewport, dx, dy);
      root.style.cursor = 'grabbing';
      applyViewport();
    } else {
      updateNodeDragPreview(drag, e);
    }
  };

  const finishPointer = (e: PointerEvent, cancelled: boolean): void => {
    if (drag === null || e.pointerId !== drag.pointerId) return;
    const d = drag;
    drag = null;
    root.style.cursor = '';
    if (d.moved) suppressNextClick = true;
    if (svg.hasPointerCapture(e.pointerId)) svg.releasePointerCapture(e.pointerId);

    if (d.mode === 'pan') {
      if (!d.moved && !cancelled && selection !== null) {
        // 배경 클릭 = 선택 해제 (UX §3.3 규약과 일관)
        selection = null;
        paintSelection();
        deps.onSelectNode(null);
      }
      return;
    }

    if (cancelled) {
      if (d.moved) render(); // 고스트/투명도 원복
      return;
    }
    if (!d.moved) {
      // 클릭 = 선택 (5px 임계 미만)
      if (selection !== d.nodeId) {
        selection = d.nodeId;
        paintSelection();
        deps.onSelectNode(d.nodeId);
      }
      return;
    }
    // 드롭 = 재정렬 (moveNode op — §2.8: 거부되면 통합자가 한국어 토스트)
    const dropWorld = clientToWorld(e.clientX, e.clientY);
    const count = deps.getGraph().nodes.length;
    const centers = Array.from({ length: count }, (_, i) => chainCenterX(i));
    const finalIndex = reorderTargetIndex(centers, d.nodeIndex, dropWorld.x);
    if (finalIndex === d.nodeIndex) {
      render(); // no-op 드롭 — 원위치 복원만
      return;
    }
    const nodeId = d.nodeId;
    deps.applyOp((g) => moveNode(g, nodeId, finalIndex));
    render();
  };

  const onPointerUp = (e: PointerEvent): void => {
    finishPointer(e, false);
  };
  const onPointerCancel = (e: PointerEvent): void => {
    finishPointer(e, true);
  };

  const onClick = (e: MouseEvent): void => {
    if (suppressNextClick) {
      suppressNextClick = false;
      return;
    }
    const target = e.target instanceof Element ? e.target : null;
    const plusG = target?.closest('[data-fg-plus]');
    if (plusG instanceof SVGElement) {
      const index = Number(plusG.dataset.fgPlus);
      if (Number.isInteger(index) && index >= 0) openPaletteAt(index, e.clientX, e.clientY);
    }
  };

  const onWheel = (e: WheelEvent): void => {
    e.preventDefault();
    const rect = svg.getBoundingClientRect();
    const factor = e.deltaY < 0 ? ZOOM_WHEEL_FACTOR : 1 / ZOOM_WHEEL_FACTOR;
    userAdjusted = true;
    viewport = zoomAt(viewport, e.clientX - rect.left, e.clientY - rect.top, factor);
    applyViewport();
  };

  /** 키 단축은 캔버스가 보일 때만 (숨은 페인의 Del 오동작 방지) */
  const canvasActive = (): boolean => root.isConnected && rootW() > 0 && rootH() > 0;

  const onKeyDown = (e: KeyboardEvent): void => {
    if (isTypingTarget(e.target)) return;
    if (e.code === 'Space') {
      if (canvasActive()) spaceHeld = true;
      return;
    }
    if (!canvasActive()) return;
    if (e.key === 'Escape') {
      closePalette();
      if (drag !== null && drag.mode === 'node' && drag.moved) {
        drag = null;
        render();
      }
      return;
    }
    if (e.key === 'Delete' && selection !== null) {
      const nodeId = selection;
      deps.applyOp((g) => removeNode(g, nodeId));
      render();
      return;
    }
    if ((e.ctrlKey || e.metaKey) && (e.key === 'd' || e.key === 'D') && selection !== null) {
      e.preventDefault(); // 브라우저 북마크 단축키 억제
      const nodeId = selection;
      deps.applyOp((g) => duplicateNode(g, nodeId));
      render();
    }
  };

  const onKeyUp = (e: KeyboardEvent): void => {
    if (e.code === 'Space') spaceHeld = false;
  };

  /** 페인 리사이즈(워크스페이스 스플리터의 notifyResize 합성 이벤트) 추종 */
  const onResize = (): void => {
    if (!userAdjusted) {
      viewport = fitViewport(
        chainBounds(deps.getGraph().nodes.length),
        rootW(),
        rootH(),
        FIT_PADDING_PX,
      );
    }
    applyViewport();
  };

  svg.addEventListener('pointerdown', onPointerDown);
  svg.addEventListener('pointermove', onPointerMove);
  svg.addEventListener('pointerup', onPointerUp);
  svg.addEventListener('pointercancel', onPointerCancel);
  svg.addEventListener('click', onClick);
  svg.addEventListener('wheel', onWheel, { passive: false });
  window.addEventListener('keydown', onKeyDown);
  window.addEventListener('keyup', onKeyUp);
  window.addEventListener('resize', onResize);

  // 툴바 배선
  zoomInBtn.addEventListener('click', () => {
    userAdjusted = true;
    viewport = zoomAt(viewport, rootW() / 2, rootH() / 2, ZOOM_BUTTON_FACTOR);
    applyViewport();
  });
  zoomOutBtn.addEventListener('click', () => {
    userAdjusted = true;
    viewport = zoomAt(viewport, rootW() / 2, rootH() / 2, 1 / ZOOM_BUTTON_FACTOR);
    applyViewport();
  });

  const fit = (): void => {
    userAdjusted = false; // fit 이후엔 그래프 변경 자동 맞춤 재개
    viewport = fitViewport(
      chainBounds(deps.getGraph().nodes.length),
      rootW(),
      rootH(),
      FIT_PADDING_PX,
    );
    applyViewport();
  };
  fitBtn.addEventListener('click', fit);

  render(); // 초기 페인트

  return {
    render,
    setStatuses: (map): void => {
      statuses = { ...map };
      paintStatuses();
    },
    selectNode: (id): void => {
      selection = id;
      paintSelection();
    },
    fit,
    dispose: (): void => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
      window.removeEventListener('resize', onResize);
      clearDragArtifacts();
      nodeGroupById.clear();
      dotById.clear();
      ariaBaseById.clear();
      root.remove(); // svg 리스너는 루트와 함께 소멸
    },
  };
}
