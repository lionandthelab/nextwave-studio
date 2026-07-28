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
// - 노드 좌표는 체인 순서에서 결정론적으로 계산한다(snake — perRow개마다 줄바꿈).
//   FlowNode.ui.x/y는 순수 표현 상태(UX §6)로 실행/직렬화에 영향이 없으므로 MVP 캔버스는
//   자체 레이아웃을 쓴다 — 드래그 재정렬의 "drop 지점 → 삽입 인덱스" 계산(§3.4)은
//   (행, x) 읽기 순서를 전제한다.
//
// ── 규모 대응 (UX_AUDIT C-10) ───────────────────────────────────────
// 구 레이아웃은 1행 고정 + ZOOM_MIN 0.4라 fit이 실패하는 지점이 1366폭에서 9노드,
// 1920폭에서도 15노드였다. 실제 워크셀 시퀀스는 30~80스텝이 정상이다. 세 축으로 연다:
//   (a) ZOOM_MIN 0.4 → 0.12
//   (b) LOD — 줌 0.5 미만에서 노드를 아이콘 + 범주 색 칩으로 축약 (node-render.nodeLod)
//   (c) snake 레이아웃 — 페인 폭에서 계산한 perRow개마다 다음 줄로 접는다
// (c)는 페인의 남는 세로 공간을 쓰므로 fit 줌이 노드 수에 선형으로 붕괴하지 않는다.
//
// 상호작용 (UX §3.4): 클릭 선택(5px 임계로 드래그와 구분) · 드래그 재정렬(고스트 +
// 삽입선 프리뷰) · 엣지/체인 끝 ＋ → 팔레트 팝오버 삽입 · Del 삭제 · Ctrl/Cmd+D 복제 ·
// 팬(스페이스/휠버튼/배경 드래그) · 휠 줌(커서 중심) · fit · 미니맵 ·
// **키보드 전 기능 도달**(roving tabindex + Enter/Space — C-5).
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
import { isTextEntryTarget, rovingTabindex, trapFocus } from '../a11y';
import type { FocusTrapHandle, RovingTabindexHandle } from '../a11y';
import { icon, makeIconButton } from '../icons';
import {
  BORDER,
  BORDER_WIDTH,
  COLOR,
  FONT,
  ICON,
  MOTION,
  RADIUS,
  SELECT,
  SHADOW,
  SPACE,
  SURFACE,
  TYPE,
  applyType,
  ensureThemeStyles,
  styled,
  tr,
} from '../theme';
import {
  PALETTE_GROUPS,
  kindMeta,
  nodeLod,
  nodeSummary,
  originBadge,
  statusColor,
  statusLabelKo,
  truncateText,
} from './node-render';
import type { NodeLod, NodeRunStatus } from './node-render';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4) ─────────────────────────────

/**
 * 휠 줌 하한/상한.
 *
 * 하한 0.12는 C-10의 처방이다 — 구 0.4에서는 1366폭 기준 9노드부터 "맞춤"이 전체를
 * 담지 못했다. 이 배율에서 텍스트는 판독 불가지만, LOD가 그 구간을 아이콘 칩으로
 * 대체하므로 **개요 보기**로서 의미가 있다(오히려 텍스트를 지우는 것이 목적이다).
 */
export const ZOOM_MIN = 0.12;
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
export const NODE_GAP_Y = 44;
export const CHAIN_MARGIN_X = 40;
export const CHAIN_Y = 64;
/** 체인 가로 피치 (노드 폭 + 가로 간격) */
export const NODE_PITCH_X = NODE_W + NODE_GAP_X;
/** snake 줄 피치 (노드 높이 + 세로 간격) */
export const ROW_PITCH_Y = NODE_H + NODE_GAP_Y;

/**
 * `perRow` 자리에 넘기면 줄바꿈 없는 1행 체인이 된다.
 * (`index % Infinity === index`, `floor(index / Infinity) === 0`)
 */
export const SINGLE_ROW = Number.POSITIVE_INFINITY;
/** 페인 폭을 아직 모를 때(마운트 직후·측정 불가) 쓰는 줄당 노드 수 */
export const DEFAULT_PER_ROW = 6;
/** 아무리 좁아도 이보다 적게 접지 않는다 (1열이 되면 세로 스크롤 지옥) */
export const MIN_PER_ROW = 2;

/** 체인 bounds 여유 — 좌우는 체인 끝 ＋, 위/아래는 goto 루프 아크가 지나간다 */
const CHAIN_BOUNDS_PAD_X = 44;
const CHAIN_BOUNDS_PAD_Y = 70;
/** goto 루프 아크 기본 높이 + 점프 거리당 추가 높이 */
const LOOP_ARC_BASE = 46;
const LOOP_ARC_PER_SPAN = 8;
const LOOP_ARC_MAX_SPAN = 6;
/** 줄바꿈 연결선의 제어점 가로 돌출 (줄 끝 → 다음 줄 시작 S자) */
const WRAP_CTRL_DX = 88;

/** fit-to-view 화면 여백 */
const FIT_PADDING_PX = 40;
/** 배경 그리드 점 간격 */
const GRID_SIZE_PX = 24;

/** 미니맵 (우하단 개요 — 저렴하게). snake는 세로로 자라므로 높이를 준다. */
const MINIMAP_W = 150;
const MINIMAP_H = 96;
const MINIMAP_PAD = 4;
const MINIMAP_MAX_SCALE = 0.25;

/** 노드 텍스트 절단 예산 (SVG text — node-render.truncateText) */
const LABEL_MAX_CHARS = 15;
const SUMMARY_MAX_CHARS = 19;

/** 노드 내부 배치 */
const NODE_STRIP_W = 3.5;
const NODE_ICON_X = 13;
const NODE_TEXT_X = 38;
const NODE_LABEL_BASELINE_Y = 22;
const NODE_SUMMARY_BASELINE_Y = 39;
const NODE_DOT_R = 4;
/** 배지 (노드 위 오른쪽 정렬) */
const BADGE_H = 15;
const BADGE_GAP = 4;
const BADGE_OFFSET_Y = 19;

/** ＋ 버튼 반지름 / 히트 반지름(WCAG 2.5.8 — 시각 크기는 유지) / 삽입선 상하 돌출 */
const PLUS_R = 9;
const PLUS_HIT_R = 13;
const PLUS_GLYPH_R = 4.5;
const INSERT_LINE_OVERHANG = 12;

/** 실행 커서 링이 노드 밖으로 나가는 여유 */
const CURSOR_PAD = 3;

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

// ── snake 레이아웃 (C-10) ───────────────────────────────────────────
//
// perRow개마다 다음 줄로 접는다. perRow는 **페인 폭에서 파생되는 뷰 상태**일 뿐이며,
// FlowNode.ui.x/y도 실행/직렬화에 영향이 없다(불변식 §2.8, UX §6) — 즉 리사이즈로
// 레이아웃이 접혀도 toSequence 결과는 한 글자도 바뀌지 않는다(canvas.test.ts가 증명).

/**
 * 페인 폭 → 줄당 노드 수 (**폭 전용 근사** — 높이를 모를 때의 폴백).
 *
 * "한 줄이 fit 여백 안에 정확히 들어차는 최대 개수"를 고른다. 측정 불가(0/NaN)면
 * DEFAULT_PER_ROW.
 *
 * ⚠ 이것만으로 레이아웃을 정하면 **줄을 줄여서 손해를 보는 구간**이 생긴다 — fit 줌은
 * 폭과 높이 **둘 다**의 함수인데 이 함수는 폭만 본다. 실측 회귀: 836×240 페인에서
 * 이 값은 2가 되고, 7노드가 4줄로 접혀 세로가 구속 조건이 되면서 fit 줌이 33%로
 * 떨어졌다(단일 행 47%보다 나쁘다). 캔버스는 `bestPerRow`를 쓴다.
 */
export function nodesPerRow(paneWidthPx: number): number {
  if (!Number.isFinite(paneWidthPx) || paneWidthPx <= 0) return DEFAULT_PER_ROW;
  // k개 줄의 월드 폭 = k·PITCH − GAP + 2·MARGIN + 2·PAD ≤ paneW − 2·FIT_PADDING
  const usable =
    paneWidthPx -
    FIT_PADDING_PX * 2 -
    CHAIN_MARGIN_X * 2 -
    CHAIN_BOUNDS_PAD_X * 2 +
    NODE_GAP_X;
  return Math.max(MIN_PER_ROW, Math.floor(usable / NODE_PITCH_X));
}

/** perRow를 유효한 양수로 정규화 (SINGLE_ROW = Infinity 허용) */
function normalizePerRow(perRow: number): number {
  if (Number.isNaN(perRow) || perRow < 1) return SINGLE_ROW;
  return perRow;
}

/** 체인 i번째 노드가 속한 줄 (0부터) */
export function chainRow(index: number, perRow: number = SINGLE_ROW): number {
  const per = normalizePerRow(perRow);
  return Number.isFinite(per) ? Math.floor(index / per) : 0;
}

/** 체인 i번째 노드의 줄 안 열 (0부터) */
export function chainCol(index: number, perRow: number = SINGLE_ROW): number {
  const per = normalizePerRow(perRow);
  return Number.isFinite(per) ? index % per : index;
}

/** count개 노드가 차지하는 줄 수 */
export function chainRowCount(count: number, perRow: number = SINGLE_ROW): number {
  if (count <= 0) return 0;
  return chainRow(count - 1, perRow) + 1;
}

/** 체인 i번째 노드의 좌상단 x (월드) */
export function chainNodeX(index: number, perRow: number = SINGLE_ROW): number {
  return CHAIN_MARGIN_X + chainCol(index, perRow) * NODE_PITCH_X;
}

/** 체인 i번째 노드의 좌상단 y (월드) */
export function chainNodeY(index: number, perRow: number = SINGLE_ROW): number {
  return CHAIN_Y + chainRow(index, perRow) * ROW_PITCH_Y;
}

/** 체인 i번째 노드의 중심 x (월드) */
export function chainCenterX(index: number, perRow: number = SINGLE_ROW): number {
  return chainNodeX(index, perRow) + NODE_W / 2;
}

/** 체인 i번째 노드의 중심 y (월드) */
export function chainCenterY(index: number, perRow: number = SINGLE_ROW): number {
  return chainNodeY(index, perRow) + NODE_H / 2;
}

/** 월드 y → 줄 인덱스 (줄 사이 중점이 경계). rowCount 범위로 클램프. */
export function chainRowAt(worldY: number, rowCount: number): number {
  if (rowCount <= 1) return 0;
  const raw = Math.round((worldY - (CHAIN_Y + NODE_H / 2)) / ROW_PITCH_Y);
  return Math.min(Math.max(raw, 0), rowCount - 1);
}

/** 체인 전체 bounds (루프 아크 여유 포함). 노드가 없으면 null. */
export function chainBounds(count: number, perRow: number = SINGLE_ROW): WorldBounds | null {
  if (count <= 0) return null;
  const rows = chainRowCount(count, perRow);
  const widestRow = rows > 1 ? normalizePerRow(perRow) : count;
  return {
    minX: CHAIN_MARGIN_X - CHAIN_BOUNDS_PAD_X,
    minY: CHAIN_Y - CHAIN_BOUNDS_PAD_Y,
    maxX: CHAIN_MARGIN_X + (widestRow - 1) * NODE_PITCH_X + NODE_W + CHAIN_BOUNDS_PAD_X,
    maxY: CHAIN_Y + (rows - 1) * ROW_PITCH_Y + NODE_H + CHAIN_BOUNDS_PAD_Y,
  };
}

/**
 * 여백을 뺀 화면에 bounds를 담는 데 필요한 배율 — **클램프 없는** 원값.
 *
 * `fitViewport`는 [ZOOM_MIN, ZOOM_MAX]로 클램프하므로 후보 비교에 쓰면 상·하한에서
 * 대량 동률이 생겨 argmax가 무의미해진다. 레이아웃 선택은 원값으로 비교한다.
 */
function rawFitScale(
  bounds: WorldBounds,
  viewW: number,
  viewH: number,
  paddingPx: number,
): number {
  const boundsW = Math.max(bounds.maxX - bounds.minX, 1);
  const boundsH = Math.max(bounds.maxY - bounds.minY, 1);
  return Math.min((viewW - paddingPx * 2) / boundsW, (viewH - paddingPx * 2) / boundsH);
}

/** 동률 판정 여유 (부동소수 잡음) */
const BEST_PER_ROW_EPS = 1e-9;

/**
 * fit 줌을 최대화하는 perRow를 고른다.
 *
 * k = 1..n 각각에 대해 `chainBounds(n, k)` 기준 fit 배율을 계산하고 argmax를 취한다.
 * **단일 행(k = n)도 후보에 포함**되므로 결과는 정의상 단일 행보다 나쁠 수 없다 —
 * 폭만 보던 `nodesPerRow`가 836×240 페인에서 7노드를 4줄로 접어 단일 행보다 나쁜
 * 33% 줌을 만들던 회귀가 구조적으로 불가능해진다.
 *
 * 동률이면 **더 큰 k(= 적은 행 수)** 를 고른다 — 가로 스캔이 세로 스캔보다 읽기 쉽다.
 *
 * LOD 임계와의 관계: 반환값은 달성 가능한 최대 배율의 k이므로, 어떤 k로든 fit 줌이
 * `LOD_ZOOM_THRESHOLD` 이상이 될 수 있다면 **이 k가 반드시 그중 하나다**(최댓값이
 * 임계 위이면 argmax도 임계 위다). 즉 "라벨이 읽히는 배치"가 별도 분기 없이 선호된다.
 *
 * 페인 높이를 모르면(레이아웃 전 0/NaN) 폭 전용 근사 `nodesPerRow`로 폴백한다.
 * 비용은 노드 수에 선형이고 산술 연산뿐이다(MVP 규모 ≤ 수백 — 레이아웃당 1회).
 */
export function bestPerRow(
  nodeCount: number,
  paneWidthPx: number,
  paneHeightPx: number,
): number {
  if (nodeCount <= 0) return DEFAULT_PER_ROW; // 그릴 노드가 없다 — 레이아웃 미사용
  if (nodeCount === 1) return 1;

  const measurable =
    Number.isFinite(paneWidthPx) &&
    paneWidthPx > 0 &&
    Number.isFinite(paneHeightPx) &&
    paneHeightPx > 0;
  if (!measurable) {
    return Math.min(Math.max(nodesPerRow(paneWidthPx), 1), nodeCount);
  }

  let bestK = 1;
  let bestScale = Number.NEGATIVE_INFINITY;
  for (let k = 1; k <= nodeCount; k += 1) {
    const bounds = chainBounds(nodeCount, k);
    if (bounds === null) continue;
    const scale = rawFitScale(bounds, paneWidthPx, paneHeightPx, FIT_PADDING_PX);
    // k 오름차순이므로 `>=`(동률 포함)면 동률 중 가장 큰 k가 남는다
    if (scale >= bestScale - BEST_PER_ROW_EPS) {
      bestScale = Math.max(bestScale, scale);
      bestK = k;
    }
  }
  return bestK;
}

/**
 * drop x → 삽입 인덱스 (0..n): 중심 x가 dropX보다 왼쪽인 노드 수.
 * 노드의 왼쪽 절반에 놓으면 그 앞, 오른쪽 절반이면 그 뒤 — "가장 가까운 두 노드 사이"
 * (UX §3.4 드래그 재정렬). centersX는 체인 순서(오름차순)를 전제한다 — 즉 1행 체인용이다.
 * snake에서는 insertionIndexAt을 쓴다.
 */
export function insertionIndexFromPoint(centersX: readonly number[], dropX: number): number {
  let index = 0;
  for (const centerX of centersX) {
    if (centerX < dropX) index += 1;
  }
  return index;
}

/**
 * drop 지점(월드) → 삽입 인덱스 (0..count) — snake 대응.
 *
 * 판정은 **읽기 순서**다: 드롭 지점보다 윗줄이면 무조건 앞, 같은 줄이면 중심 x 비교.
 * perRow = SINGLE_ROW면 insertionIndexFromPoint(체인 중심 x, dropX)와 정확히 같다.
 */
export function insertionIndexAt(
  count: number,
  perRow: number,
  dropX: number,
  dropY: number,
): number {
  if (count <= 0) return 0;
  const dropRow = chainRowAt(dropY, chainRowCount(count, perRow));
  let index = 0;
  for (let i = 0; i < count; i += 1) {
    const row = chainRow(i, perRow);
    if (row < dropRow || (row === dropRow && chainCenterX(i, perRow) < dropX)) index += 1;
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
 * reorderTargetIndex의 snake판.
 *
 * 전체 체인 기준 삽입 인덱스에서 "드래그 노드가 그 앞에 있었으면 1을 뺀다"로 유도한다
 * — 체인 중심이 오름차순이면 `raw > fromIndex` ⟺ `centers[fromIndex] < drop`이므로
 * 1행에서는 reorderTargetIndex와 결과가 동일하다.
 */
export function reorderTargetIndexAt(
  count: number,
  perRow: number,
  fromIndex: number,
  dropX: number,
  dropY: number,
): number {
  const raw = insertionIndexAt(count, perRow, dropX, dropY);
  return raw > fromIndex ? raw - 1 : raw;
}

/** 삽입선 프리뷰 위치 (월드) — 세로선의 x와 그 줄의 상단 y */
export interface InsertionLine {
  x: number;
  /** 선이 걸치는 줄의 노드 상단 y */
  rowTopY: number;
}

/**
 * 삽입선 프리뷰: fromIndex를 뺀 체인에서 finalIndex 위치의 틈.
 * count는 전체 노드 수(드래그 중인 노드 포함). 노드가 1개뿐이면 첫 틈을 돌려준다.
 */
export function insertionLine(
  fromIndex: number,
  finalIndex: number,
  count: number,
  perRow: number = SINGLE_ROW,
): InsertionLine {
  const remainingCount = count - 1;
  if (remainingCount <= 0) {
    return { x: chainNodeX(0, perRow) - NODE_GAP_X / 2, rowTopY: chainNodeY(0, perRow) };
  }
  if (finalIndex >= remainingCount) {
    // 체인 끝 — 마지막 남은 노드의 오른쪽
    const lastOriginal = fromIndex === count - 1 ? count - 2 : count - 1;
    return {
      x: chainNodeX(lastOriginal, perRow) + NODE_W + NODE_GAP_X / 2,
      rowTopY: chainNodeY(lastOriginal, perRow),
    };
  }
  // remaining[finalIndex]의 원래 인덱스 앞 틈
  const originalIndex = finalIndex < fromIndex ? finalIndex : finalIndex + 1;
  return {
    x: chainNodeX(originalIndex, perRow) - NODE_GAP_X / 2,
    rowTopY: chainNodeY(originalIndex, perRow),
  };
}

/** insertionLine의 x만 (1행 체인 호환 경로) */
export function insertionLineX(fromIndex: number, finalIndex: number, count: number): number {
  return insertionLine(fromIndex, finalIndex, count).x;
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
  /**
   * 빈 플로우의 "자연어로 만들기" 버튼 (선택).
   *
   * 통합자가 커맨드바의 자연어 입력에 `focus()`를 배선한다. **주지 않으면 버튼을
   * 만들지 않고** 안내 문구만 남긴다 — 눌러도 아무 일이 없는 유령 버튼을 만드느니
   * 없는 편이 낫다(구 코드는 "Phase 9에서 제공됩니다"라는 거짓 안내였다, C-12).
   */
  onRequestNlFocus?: () => void;
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

/** 배지 폭 추정 (TYPE.micro 10px — ASCII 6px/한글 11px + 좌우 여백) */
function badgeWidthPx(text: string): number {
  let width = 10;
  for (const ch of text) width += ch.charCodeAt(0) < 256 ? 6 : 11;
  return width;
}

/**
 * SVG 요소를 a11y 프리미티브(rovingTabindex)에 넘기기 위한 구조적 캐스트.
 *
 * `rovingTabindex`가 항목에서 쓰는 것은 `tabIndex` · `focus()` · 배열 identity 비교뿐이고,
 * 이 셋은 SVGElement에도 (HTML 명세의 `HTMLOrSVGElement` 믹스인으로) 그대로 존재한다.
 * `a11y.ts`는 다른 담당자의 파일이라 시그니처를 넓힐 수 없으므로 호출부에서 좁힌다.
 * `any`가 아니라 지역화된 구조적 캐스트다 (CLAUDE.md §4).
 */
function asFocusItem(el: SVGElement): HTMLElement {
  return el as unknown as HTMLElement;
}

// ── 캔버스 전용 스타일 (hover/펄스/커서 — 토큰만 소비, 1회 주입) ─────

const CANVAS_STYLE_ID = 'rsw-flow-canvas-styles';

function ensureCanvasStyles(): void {
  if (document.getElementById(CANVAS_STYLE_ID) !== null) return;
  const style = document.createElement('style');
  style.id = CANVAS_STYLE_ID;
  style.textContent = `
/* color는 노드 안 SVG 아이콘의 currentColor 소스다 — 아이콘이 hover/선택에서
   텍스트와 **함께** 변한다(이모지로는 불가능했던 것 — C-13). */
.rsw-fg-node { cursor: grab; color: ${COLOR.text}; }
.rsw-fg-node:hover, .rsw-fg-node--selected { color: ${COLOR.textStrong}; }
.rsw-fg-node-body {
  fill: ${SURFACE.raised};
  stroke: ${BORDER.strong};
  stroke-width: ${BORDER_WIDTH.hair};
  transition: ${tr('stroke', MOTION.instant)};
}
.rsw-fg-node:hover .rsw-fg-node-body { stroke: ${BORDER.hover}; }
.rsw-fg-node--selected .rsw-fg-node-body,
.rsw-fg-node--selected:hover .rsw-fg-node-body {
  stroke: var(--rsw-select);
  stroke-width: ${BORDER_WIDTH.thick};
}
@keyframes rsw-fg-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
.rsw-fg-dot--active { animation: rsw-fg-pulse ${MOTION.loop.ms}ms ease-in-out infinite; }
.rsw-fg-plus { cursor: pointer; color: ${COLOR.muted}; }
.rsw-fg-plus circle { transition: ${tr('stroke', MOTION.instant)}, ${tr('fill', MOTION.instant)}; }
.rsw-fg-plus:hover { color: var(--rsw-accent-text); }
.rsw-fg-plus:hover .rsw-fg-plus-ring { stroke: var(--rsw-accent); fill: var(--rsw-accent-soft); }

/* 실행 커서 — 활성 노드가 바뀌면 **이동한다**(순간 점프 금지, C-9).
   prefers-reduced-motion은 theme의 전역 미디어 쿼리가 이미 무력화한다. */
.rsw-fg-cursor {
  pointer-events: none;
  transition: ${tr('transform', MOTION.emphasis)};
}
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
  /** 팔레트 포커스 트랩 (열려 있을 때만) */
  let paletteTrap: FocusTrapHandle | null = null;
  /** snake 줄당 노드 수 — 페인 폭에서 파생(순수 표현, §2.8 무관) */
  let perRow = DEFAULT_PER_ROW;
  /** 현재 그려져 있는 상세도 — 줌이 임계를 넘으면 노드만 다시 그린다 */
  let renderedLod: NodeLod = 'full';
  /** 실행 커서가 이미 보이는가 (첫 등장은 보간하지 않는다 — 0,0에서 날아오지 않게) */
  let cursorShown = false;

  // 렌더 산출물 맵 (상태/선택 경량 페인트용 — drawWorld()마다 재구축)
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

  /**
   * 실행 커서 (C-9) — 노드마다 하나씩 그리지 않고 **1개를 이동**시킨다.
   * drawWorld()가 worldG를 비워도 이 요소는 살아남아 다시 붙으므로 transform 보간이
   * 재렌더를 가로질러 이어진다.
   */
  const cursorG = svgEl('g', { class: 'rsw-fg-cursor', visibility: 'hidden' });
  cursorG.appendChild(
    svgEl('rect', {
      x: String(-CURSOR_PAD),
      y: String(-CURSOR_PAD),
      width: String(NODE_W + CURSOR_PAD * 2),
      height: String(NODE_H + CURSOR_PAD * 2),
      rx: '11',
      fill: 'none',
      stroke: COLOR.accent,
      'stroke-width': BORDER_WIDTH.thick,
    }),
  );

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
  const zoomOutBtn = makeIconButton('minus', '', '축소', 'flow-zoom-out');
  const zoomReadout = applyType(document.createElement('span'), TYPE.monoBody);
  styled(zoomReadout, { color: COLOR.label, minWidth: '38px', textAlign: 'center' });
  zoomReadout.dataset.testid = 'flow-zoom-readout';
  const zoomInBtn = makeIconButton('plus', '', '확대', 'flow-zoom-in');
  const fitBtn = makeIconButton('fit', '맞춤', '전체 보기', 'flow-fit');
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
  //
  // 구 문구는 "또는 자연어로 플로우 생성 — Phase 9에서 제공됩니다"였다. 자연어 플래너는
  // 이미 출시돼 있으므로 **거짓 안내**이자 내부 로드맵 어휘 노출이었고, 첫 사용자가 가장
  // 도움이 필요한 순간에 "이 기능은 없다"고 말해 이탈시켰다(C-12 · Nielsen #1/#2).
  // 문구를 지우는 대신 **행동 유도 버튼**으로 바꾼다.
  const emptyOverlay = styled(document.createElement('div'), {
    position: 'absolute',
    inset: '0',
    display: 'none',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    gap: SPACE.xl,
    color: COLOR.muted,
    textAlign: 'center',
    pointerEvents: 'none',
  });
  emptyOverlay.dataset.testid = 'flow-empty';
  const emptyTitle = applyType(document.createElement('div'), TYPE.subhead);
  styled(emptyTitle, { color: COLOR.text });
  emptyTitle.textContent = '플로우가 비어 있습니다';
  const emptyActions = styled(document.createElement('div'), {
    display: 'flex',
    alignItems: 'center',
    gap: SPACE.md,
    pointerEvents: 'auto',
  });
  const nlFocus = deps.onRequestNlFocus;
  // 주요 액션은 1화면 1개다 — 자연어 생성이 있으면 그쪽이 주요 액션이고 수동 추가는 보조다
  const emptyAddBtn = makeIconButton(
    'plus',
    '노드 추가',
    '첫 노드 삽입',
    'flow-empty-add',
    nlFocus === undefined ? 'primary' : 'default',
  );
  emptyActions.appendChild(emptyAddBtn);
  emptyOverlay.appendChild(emptyTitle);
  emptyOverlay.appendChild(emptyActions);

  if (nlFocus !== undefined) {
    const emptyNlBtn = makeIconButton(
      'wand',
      '자연어로 만들기',
      '자연어로 플로우 만들기',
      'flow-empty-nl',
      'primary',
    );
    emptyNlBtn.addEventListener('click', () => {
      nlFocus();
    });
    emptyActions.appendChild(emptyNlBtn);
  } else {
    const emptyHint = applyType(document.createElement('div'), TYPE.caption);
    emptyHint.textContent = '자연어로 만들려면 상단 입력창에 지시를 적으세요';
    emptyOverlay.appendChild(emptyHint);
  }
  root.appendChild(emptyOverlay);

  // ── 노드 팔레트 팝오버 (＋ 삽입 — step 종류별 분류) ────────────────
  //
  // role은 menu가 아니라 **dialog**다: `role="menu"`의 자식은 menuitem/group/separator만
  // 허용되는데 이 팔레트에는 4개의 시각적 그룹 제목(동작/시간/충돌/흐름)이 있다. menu로
  // 두려면 제목마다 `role="group"` + `aria-label`로 감싸 텍스트를 ARIA에 중복 선언해야
  // 하고, 그래도 얻는 것이 없다. dialog는 현재 구조를 그대로 받고, 계약대로 `trapFocus`를
  // 걸면 Tab이 배경으로 새지 않는다(구 코드는 role="menu"에 순수 <button> 자식 —
  // ARIA상 무효 구조였고 트랩도 없었다, C-5).
  const paletteEl = styled(document.createElement('div'), {
    position: 'absolute',
    display: 'none',
    width: `${PALETTE_W}px`,
    maxHeight: `${PALETTE_MAX_H}px`,
    overflowY: 'auto',
    background: SURFACE.overlay,
    border: `${BORDER_WIDTH.hair} solid ${BORDER.default}`,
    borderRadius: RADIUS.md,
    boxShadow: SHADOW.overlay,
    padding: SPACE.sm,
    boxSizing: 'border-box',
    zIndex: '10',
  });
  paletteEl.classList.add('ui-scroll');
  paletteEl.dataset.testid = 'flow-palette';
  paletteEl.setAttribute('role', 'dialog');
  paletteEl.setAttribute('aria-modal', 'true');
  paletteEl.setAttribute('aria-label', '노드 삽입');

  const paletteButtons: { kind: FlowNode['kind']; btn: HTMLButtonElement }[] = [];
  for (const group of PALETTE_GROUPS) {
    const header = applyType(document.createElement('div'), TYPE.micro);
    styled(header, {
      color: COLOR.label,
      padding: `${SPACE.sm} ${SPACE.xs} ${SPACE.xxs}`,
      letterSpacing: '0.06em',
    });
    header.textContent = group.labelKo;
    paletteEl.appendChild(header);
    for (const kind of group.kinds) {
      const meta = kindMeta(kind);
      const btn = makeIconButton(
        meta.icon,
        meta.label,
        meta.descriptionKo,
        `flow-palette-${kind}`,
        'ghost',
      );
      // 도메인 식별자는 영문 유지 — 한국어 TTS가 CamelCase를 철자로 읽지 않게 한다
      btn.querySelector('.ui-btn__label')?.setAttribute('lang', 'en');
      styled(btn, { width: '100%', justifyContent: 'flex-start' });
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
        // 키보드로 삽입했다면 포커스가 갈 곳이 사라졌다 — 새 노드로 옮겨 준다
        focusNodeAtIndex(index);
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

  // ── 키보드 도달 (C-5) ─────────────────────────────────────────────
  //
  // 구 코드는 노드/삽입점에 `role="button"` + `aria-label`만 있고 `tabindex`가 없었다 —
  // 탭 순서에 등장하지 않으니 aria만 있고 조작할 수 없는 유령 버튼 14개였고, Delete /
  // Ctrl+D 단축키의 전제인 "선택"이 마우스 전용이었다. roving tabindex로 탭 1번에
  // 그래프에 들어오고 방향키로 훑는다(snake 레이아웃이라 ←→↑↓ 전부 의미가 있다).
  const focusItems: SVGGElement[] = [];
  let roving: RovingTabindexHandle | null = null;

  const activateItem = (el: SVGGElement): void => {
    const nodeId = el.dataset.fgNode;
    if (nodeId !== undefined) {
      if (selection !== nodeId) {
        selection = nodeId;
        paintSelection();
        deps.onSelectNode(nodeId);
      }
      return;
    }
    const plusIndex = Number(el.dataset.fgPlus);
    if (!Number.isInteger(plusIndex) || plusIndex < 0) return;
    const rect = el.getBoundingClientRect();
    openPaletteAt(plusIndex, rect.left, rect.bottom);
  };

  /** 삽입 후 새 노드로 포커스를 옮긴다 (키보드 흐름이 끊기지 않게) */
  const focusNodeAtIndex = (index: number): void => {
    const node = deps.getGraph().nodes[index];
    if (node === undefined) return;
    const g = nodeGroupById.get(node.id);
    if (g === undefined) return;
    const itemIndex = focusItems.indexOf(g);
    if (itemIndex >= 0) roving?.setActive(itemIndex, true);
  };

  // ── 팔레트 열기/닫기 ──────────────────────────────────────────────
  const closePalette = (): void => {
    pendingInsertIndex = null;
    paletteEl.style.display = 'none';
    // release()가 진입 전 포커스(= 그 삽입점)를 복원한다
    paletteTrap?.release();
    paletteTrap = null;
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
    // aria-modal="true"를 선언했으면 트랩은 선택이 아니라 의무다 (계약 §4)
    const firstEnabled = paletteButtons.find(({ btn }) => !btn.disabled)?.btn ?? null;
    paletteTrap = trapFocus(paletteEl, {
      initialFocus: firstEnabled,
      onEscape: closePalette,
    });
  };

  emptyAddBtn.addEventListener('click', () => {
    const rect = emptyAddBtn.getBoundingClientRect();
    openPaletteAt(0, rect.left, rect.bottom);
  });

  // ── 뷰포트 반영 (팬/줌 — 노드 재구축 없음) ────────────────────────
  //
  // 예외: 줌이 LOD 임계를 가로지르면 노드 표현이 달라지므로 그때만 다시 그린다
  // (drawWorld는 applyViewport를 호출하지 않는다 — 재귀 없음).
  const applyViewport = (): void => {
    const transform = `translate(${viewport.x} ${viewport.y}) scale(${viewport.zoom})`;
    worldG.setAttribute('transform', transform);
    gridPattern.setAttribute('patternTransform', transform);
    zoomReadout.textContent = `${Math.round(viewport.zoom * 100)}%`;
    updateMinimapViewRect();
    if (nodeLod(viewport.zoom) !== renderedLod) drawWorld();
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
    const bounds = chainBounds(count, perRow);
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
          x: String(chainNodeX(i, perRow) * scale + offX),
          y: String(chainNodeY(i, perRow) * scale + offY),
          width: String(Math.max(2, NODE_W * scale)),
          height: String(Math.max(1.5, NODE_H * scale)),
          rx: '1',
          fill: BORDER.strong,
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
        height: String(BADGE_H),
        rx: '3',
        fill: bg,
      }),
    );
    const label = svgText(text, {
      x: String(x + width / 2),
      y: String(topY + BADGE_H - 4.5),
      'text-anchor': 'middle',
      fill: fg,
    });
    applyType(label, TYPE.micro);
    parent.appendChild(label);
    return x;
  };

  /** 노드 안 SVG 아이콘 — 색은 currentColor(=.rsw-fg-node의 color)를 상속한다 */
  const nodeIcon = (meta: ReturnType<typeof kindMeta>, x: number, y: number, size: number) => {
    const el = icon(meta.icon, size);
    el.setAttribute('x', String(x));
    el.setAttribute('y', String(y));
    return el;
  };

  const drawNode = (node: FlowNode, index: number, count: number, lod: NodeLod): void => {
    const x = chainNodeX(index, perRow);
    const y = chainNodeY(index, perRow);
    const meta = kindMeta(node.kind);
    const status: NodeRunStatus = statuses[node.id] ?? node.status ?? 'pending';
    const summary = nodeSummary(node.kind, node.params);

    const g = svgEl('g', { class: 'rsw-fg-node', role: 'button', tabindex: '-1' });
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
    // 종류 컬러 스트립 (좌측) — 범주 색은 **여기만** 든다 (아이콘엔 칠하지 않는다)
    g.appendChild(
      svgEl('rect', {
        x: String(x + 1.5),
        y: String(y + 1.5),
        width: String(NODE_STRIP_W),
        height: String(NODE_H - 3),
        rx: '2',
        fill: meta.color,
      }),
    );

    if (lod === 'compact') {
      // 축약: 범주 색 칩 + 중앙 아이콘. 텍스트는 이 배율에서 어차피 판독 불가다.
      g.appendChild(
        svgEl('rect', {
          x: String(x + 1.5),
          y: String(y + 1.5),
          width: String(NODE_W - 3),
          height: String(NODE_H - 3),
          rx: '7',
          fill: meta.color,
          'fill-opacity': '0.18',
        }),
      );
      g.appendChild(
        nodeIcon(meta, x + (NODE_W - ICON.xl) / 2, y + (NODE_H - ICON.xl) / 2, ICON.xl),
      );
    } else {
      g.appendChild(nodeIcon(meta, x + NODE_ICON_X, y + (NODE_H - ICON.md) / 2, ICON.md));
      // 타입명 — 영문 도메인 식별자이므로 lang="en" (한국어 TTS의 철자 나열 방지, WCAG 3.1.2)
      const label = svgText(truncateText(meta.label, LABEL_MAX_CHARS), {
        x: String(x + NODE_TEXT_X),
        y: String(y + NODE_LABEL_BASELINE_Y),
        lang: 'en',
        fill: COLOR.textStrong,
      });
      applyType(label, TYPE.bodyStrong);
      g.appendChild(label);
      const summaryEl = svgText(truncateText(summary, SUMMARY_MAX_CHARS), {
        x: String(x + NODE_TEXT_X),
        y: String(y + NODE_SUMMARY_BASELINE_Y),
        fill: COLOR.muted,
      });
      applyType(summaryEl, TYPE.monoMicro);
      g.appendChild(summaryEl);
    }

    // 상태 점 (우상단 — 색 + aria 텍스트 병행)
    const dot = svgEl('circle', {
      cx: String(x + NODE_W - 12),
      cy: String(y + 12),
      r: String(NODE_DOT_R),
      fill: statusColor(status),
      class: status === 'active' ? 'rsw-fg-dot rsw-fg-dot--active' : 'rsw-fg-dot',
    });
    g.appendChild(dot);

    // 배지 (노드 위 오른쪽 정렬): [수정됨|AI] [비활성] — 축약 시엔 생략(판독 불가)
    if (lod === 'full') {
      let badgeRightX = x + NODE_W;
      if (!node.enabled) {
        badgeRightX =
          drawBadge(g, badgeRightX, y - BADGE_OFFSET_Y, '비활성', COLOR.mutedSoft, COLOR.muted) -
          BADGE_GAP;
      }
      const origin = originBadge(node.origin);
      if (origin !== null) {
        drawBadge(
          g,
          badgeRightX,
          y - BADGE_OFFSET_Y,
          origin,
          origin === 'AI' ? COLOR.accentSoft : COLOR.infoSoft,
          origin === 'AI' ? COLOR.accentText : COLOR.infoText,
        );
      }
    }

    // aria는 LOD와 무관하게 **전체 요약**을 유지한다 — 축약은 시각 채널만 줄인다
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
    const g = svgEl('g', { class: 'rsw-fg-plus', role: 'button', tabindex: '-1' });
    g.dataset.fgPlus = String(insertIndex);
    g.dataset.testid = 'flow-plus';
    g.setAttribute('aria-label', `위치 ${insertIndex}에 노드 삽입`);
    // 히트 영역만 24px 이상으로 넓힌다 — 시각 크기는 유지 (WCAG 2.2 SC 2.5.8)
    g.appendChild(
      svgEl('circle', {
        cx: String(centerX),
        cy: String(centerY),
        r: String(PLUS_HIT_R),
        fill: 'transparent',
      }),
    );
    g.appendChild(
      svgEl('circle', {
        class: 'rsw-fg-plus-ring',
        cx: String(centerX),
        cy: String(centerY),
        r: String(PLUS_R),
        fill: SURFACE.raised,
        stroke: BORDER.strong,
        'stroke-width': BORDER_WIDTH.hair,
      }),
    );
    g.appendChild(
      svgEl('path', {
        d:
          `M ${centerX - PLUS_GLYPH_R} ${centerY} H ${centerX + PLUS_GLYPH_R} ` +
          `M ${centerX} ${centerY - PLUS_GLYPH_R} V ${centerY + PLUS_GLYPH_R}`,
        fill: 'none',
        stroke: 'currentColor',
        'stroke-width': '1.5',
        'stroke-linecap': 'round',
      }),
    );
    worldG.appendChild(g);
  };

  const drawSeqEdges = (count: number): void => {
    for (let i = 1; i < count; i += 1) {
      const sameRow = chainRow(i - 1, perRow) === chainRow(i, perRow);
      const fromX = chainNodeX(i - 1, perRow) + NODE_W;
      const fromY = chainCenterY(i - 1, perRow);
      const toX = chainNodeX(i, perRow);
      const toY = chainCenterY(i, perRow);
      // 같은 줄이면 짧은 S, 줄바꿈이면 크게 돌아 다음 줄 머리로 (연결이 끊기지 않게)
      const ctrl = sameRow ? NODE_GAP_X / 2 : WRAP_CTRL_DX;
      worldG.appendChild(
        svgEl('path', {
          class: 'rsw-fg-edge',
          d: `M ${fromX} ${fromY} C ${fromX + ctrl} ${fromY}, ${toX - ctrl} ${toY}, ${toX} ${toY}`,
          fill: 'none',
          stroke: BORDER.strong,
          'stroke-width': '1.5',
          'marker-end': `url(#${seqMarkerId})`,
        }),
      );
    }
    // 엣지 중점 ＋ (사이 삽입) — 줄바꿈 구간에서는 앞 노드의 오른쪽에 둔다
    for (let i = 1; i < count; i += 1) {
      const sameRow = chainRow(i - 1, perRow) === chainRow(i, perRow);
      const plusX = sameRow
        ? chainNodeX(i, perRow) - NODE_GAP_X / 2
        : chainNodeX(i - 1, perRow) + NODE_W + NODE_GAP_X / 2;
      drawPlus(plusX, chainCenterY(sameRow ? i : i - 1, perRow), i);
    }
    // 체인 끝 ＋
    drawPlus(
      chainNodeX(count - 1, perRow) + NODE_W + NODE_GAP_X / 2,
      chainCenterY(count - 1, perRow),
      count,
    );
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
      const fromEdgeY = chainNodeY(fromIndex, perRow) + (above ? 0 : NODE_H);
      const toEdgeY = chainNodeY(toIndex, perRow) + (above ? 0 : NODE_H);
      const ctrlFromY = above ? fromEdgeY - arc : fromEdgeY + arc;
      const ctrlToY = above ? toEdgeY - arc : toEdgeY + arc;
      const fromX = chainCenterX(fromIndex, perRow);
      const toX = chainCenterX(toIndex, perRow);
      worldG.appendChild(
        svgEl('path', {
          class: 'rsw-fg-edge',
          d: `M ${fromX} ${fromEdgeY} C ${fromX} ${ctrlFromY}, ${toX} ${ctrlToY}, ${toX} ${toEdgeY}`,
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

  // ── 실행 커서 (C-9) ───────────────────────────────────────────────
  //
  // 활성 노드가 바뀔 때 강조가 순간 점프하면 시선이 따라가지 못한다 — 노드가 많을수록
  // 심해진다. 링 하나를 480ms(MOTION.emphasis)로 **이전 노드에서 다음 노드로 이동**시켜
  // 눈이 경로를 따라오게 한다.
  const activeNodeIndex = (): number =>
    deps
      .getGraph()
      .nodes.findIndex((node) => (statuses[node.id] ?? node.status ?? 'pending') === 'active');

  const updateCursor = (): void => {
    const index = activeNodeIndex();
    if (index < 0) {
      cursorG.setAttribute('visibility', 'hidden');
      cursorShown = false;
      return;
    }
    const x = chainNodeX(index, perRow);
    const y = chainNodeY(index, perRow);
    // 첫 등장은 보간하지 않는다 (원점에서 날아오지 않게)
    cursorG.style.transition = cursorShown ? tr('transform', MOTION.emphasis) : 'none';
    cursorG.style.transform = `translate(${x}px, ${y}px)`;
    cursorG.setAttribute('visibility', 'visible');
    cursorShown = true;
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
    updateCursor();
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
    const finalIndex = reorderTargetIndexAt(count, perRow, d.nodeIndex, now.x, now.y);
    if (count >= 2 && finalIndex !== d.nodeIndex) {
      if (insertLineEl === null) {
        insertLineEl = svgEl('line', {
          stroke: SELECT.base,
          'stroke-width': '2',
          'stroke-linecap': 'round',
        });
        insertLineEl.dataset.testid = 'flow-insert-line';
        worldG.appendChild(insertLineEl);
      }
      const line = insertionLine(d.nodeIndex, finalIndex, count, perRow);
      insertLineEl.setAttribute('x1', String(line.x));
      insertLineEl.setAttribute('x2', String(line.x));
      insertLineEl.setAttribute('y1', String(line.rowTopY - INSERT_LINE_OVERHANG));
      insertLineEl.setAttribute('y2', String(line.rowTopY + NODE_H + INSERT_LINE_OVERHANG));
      insertLineEl.setAttribute('visibility', 'visible');
    } else {
      insertLineEl?.setAttribute('visibility', 'hidden');
    }
  };

  // ── 월드 재구축 (현재 perRow/LOD로 노드·엣지·＋를 다시 그린다) ─────
  //
  // 뷰포트는 건드리지 않는다 — applyViewport()가 LOD 임계 통과 시 이 함수만 부른다.
  const drawWorld = (): void => {
    const graph = deps.getGraph();
    const nodes = graph.nodes;
    const lod = nodeLod(viewport.zoom);
    renderedLod = lod;

    nodeGroupById.clear();
    dotById.clear();
    ariaBaseById.clear();
    focusItems.length = 0;
    worldG.replaceChildren();

    emptyOverlay.style.display = nodes.length === 0 ? 'flex' : 'none';
    if (nodes.length > 0) {
      drawSeqEdges(nodes.length);
      drawLoopEdges(graph);
      nodes.forEach((node, i) => drawNode(node, i, nodes.length, lod));
    }
    // 실행 커서는 worldG를 비워도 살아남는 지속 요소다 (transform 보간 연속성)
    worldG.appendChild(cursorG);
    rebuildMinimap(nodes.length);

    // 탭 순서 = 읽기 순서 [노드0, ＋1, 노드1, ＋2, …, ＋n]
    const plusById = new Map<number, SVGGElement>();
    for (const el of worldG.querySelectorAll<SVGGElement>('[data-fg-plus]')) {
      plusById.set(Number(el.dataset.fgPlus), el);
    }
    nodes.forEach((node, i) => {
      const nodeG = nodeGroupById.get(node.id);
      if (nodeG !== undefined) focusItems.push(nodeG);
      const plusG = plusById.get(i + 1);
      if (plusG !== undefined) focusItems.push(plusG);
    });
    roving?.setItems(focusItems.map(asFocusItem));
    updateCursor();
  };

  // ── 전체 재렌더 — (graph, statuses, selection, viewport)의 순수 함수 ─
  const render = (): void => {
    closePalette();
    clearDragArtifacts();
    const nodes = deps.getGraph().nodes;

    // 사라진 노드가 선택돼 있으면 해제하고 통지 (삭제 후 인스펙터 잔상 방지)
    if (selection !== null && !nodes.some((n) => n.id === selection)) {
      selection = null;
      deps.onSelectNode(null);
    }

    // snake 줄당 노드 수는 (노드 수, 페인 폭, 페인 높이)에서 파생된다 —
    // 레이아웃보다 먼저 확정해야 한다. 노드 수가 바뀌면 최적 k도 바뀐다.
    perRow = bestPerRow(nodes.length, rootW(), rootH());

    // 사용자가 팬/줌을 만지기 전에는 그래프 변화를 따라 자동 fit
    if (!userAdjusted) {
      viewport = fitViewport(chainBounds(nodes.length, perRow), rootW(), rootH(), FIT_PADDING_PX);
    }
    drawWorld();
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
    const finalIndex = reorderTargetIndexAt(count, perRow, d.nodeIndex, dropWorld.x, dropWorld.y);
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
    if (isTextEntryTarget(e.target)) return;
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
    // 페인 크기가 바뀌면 최적 줄당 노드 수가 달라진다 → 좌표가 전부 바뀌므로 재렌더
    if (bestPerRow(deps.getGraph().nodes.length, rootW(), rootH()) !== perRow) {
      render();
      return;
    }
    if (!userAdjusted) {
      viewport = fitViewport(
        chainBounds(deps.getGraph().nodes.length, perRow),
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
    const count = deps.getGraph().nodes.length;
    const nextPerRow = bestPerRow(count, rootW(), rootH());
    if (nextPerRow !== perRow) {
      // 레이아웃이 달라진다 — 좌표부터 다시 그려야 fit 결과가 맞는다
      perRow = nextPerRow;
      drawWorld();
    }
    viewport = fitViewport(chainBounds(count, perRow), rootW(), rootH(), FIT_PADDING_PX);
    applyViewport();
  };
  fitBtn.addEventListener('click', fit);

  // roving tabindex — 컨테이너는 svg 루트다(툴바/팔레트는 svg 밖이라 방향키를 뺏기지 않는다)
  roving = rovingTabindex(asFocusItem(svg), [], {
    orientation: 'both',
    wrap: false,
    onActivate: (_el, index) => {
      const target = focusItems[index];
      if (target !== undefined) activateItem(target);
    },
  });

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
      paletteTrap?.release();
      paletteTrap = null;
      roving?.dispose();
      roving = null;
      clearDragArtifacts();
      nodeGroupById.clear();
      dotById.clear();
      ariaBaseById.clear();
      focusItems.length = 0;
      root.remove(); // svg 리스너는 루트와 함께 소멸
    },
  };
}
