// ui/flow-graph/canvas.test.ts — 캔버스 순수 헬퍼 단위 테스트 (DOM 비의존, node 환경)
//
// mountFlowCanvas의 DOM 조립/포인터 배선은 브라우저 게이트(gate-browser.mjs) 몫이다.
// 여기서는 Phase 8 요구 + Phase 11 규모 대응(UX_AUDIT C-10)의 순수 계산만 검증한다:
// - 팬/줌 수학 (clampZoom · zoomAt 커서 중심 불변 · panBy · fitViewport)
// - snake 레이아웃 (nodesPerRow · chainNodeX/Y · chainRow · chainBounds) 과
//   그것이 **직렬화에 영향이 없다**는 증명 (toSequence 왕복 — 불변식 §2.8)
// - 드래그 재정렬의 drop 지점 → 삽입 인덱스 (1행 경로 + snake 경로의 등가성)
// - LOD 정책 (node-render.nodeLod)
// - 노드 요약 텍스트 (node-render.nodeSummary — UX §3.4 예시 형식 고정)
// - 종류 메타/팔레트 그룹/상태·출처 표현 헬퍼 (아이콘 SVG화 C-13, 범주 색 C-14)

import { describe, expect, it } from 'vitest';
import {
  CHAIN_MARGIN_X,
  CHAIN_Y,
  DEFAULT_PER_ROW,
  DRAG_THRESHOLD_PX,
  MIN_PER_ROW,
  NODE_GAP_X,
  NODE_H,
  NODE_PITCH_X,
  NODE_W,
  ROW_PITCH_Y,
  SINGLE_ROW,
  ZOOM_MAX,
  ZOOM_MIN,
  chainBounds,
  chainCenterX,
  chainCenterY,
  chainCol,
  chainNodeX,
  chainNodeY,
  chainRow,
  chainRowAt,
  chainRowCount,
  clampZoom,
  bestPerRow,
  fitViewport,
  insertionIndexAt,
  insertionIndexFromPoint,
  insertionLine,
  insertionLineX,
  nodesPerRow,
  panBy,
  reorderTargetIndex,
  reorderTargetIndexAt,
  zoomAt,
} from './canvas';
import {
  LOD_ZOOM_THRESHOLD,
  PALETTE_GROUPS,
  formatDurationSec,
  formatNum,
  kindMeta,
  nodeLod,
  nodeSummary,
  originBadge,
  statusColor,
  statusLabelKo,
  truncateText,
} from './node-render';
import { fromSequence, moveNode, toSequence } from '../../schema/flow-graph';
import type { ControlSequence } from '../../schema/types';
import { CATEGORY, COLLISION, COLOR } from '../theme';

// ── 줌 클램프 ───────────────────────────────────────────────────────

describe('clampZoom', () => {
  it('하한 0.12 — 규모 천장 해제 (C-10)', () => {
    // 구 하한 0.4는 1366폭에서 9노드부터 "맞춤"이 전체를 담지 못하게 만들었다.
    expect(ZOOM_MIN).toBe(0.12);
    expect(ZOOM_MAX).toBe(2.0);
    expect(clampZoom(1)).toBe(1);
    expect(clampZoom(0.01)).toBe(ZOOM_MIN);
    expect(clampZoom(0.3)).toBe(0.3); // 구 하한 아래도 이제 유효 배율이다
    expect(clampZoom(9)).toBe(ZOOM_MAX);
    expect(clampZoom(ZOOM_MIN)).toBe(ZOOM_MIN);
    expect(clampZoom(ZOOM_MAX)).toBe(ZOOM_MAX);
  });

  it('NaN은 하한으로 방어한다', () => {
    expect(clampZoom(Number.NaN)).toBe(ZOOM_MIN);
  });
});

// ── 커서 중심 줌 ────────────────────────────────────────────────────

describe('zoomAt', () => {
  const worldUnderCursor = (
    vp: { x: number; y: number; zoom: number },
    cx: number,
    cy: number,
  ): { x: number; y: number } => ({ x: (cx - vp.x) / vp.zoom, y: (cy - vp.y) / vp.zoom });

  it('커서 아래의 월드점이 줌 후에도 그 자리에 남는다 (커서 중심)', () => {
    const vp = { x: 10, y: 20, zoom: 1 };
    const before = worldUnderCursor(vp, 100, 50);
    const zoomed = zoomAt(vp, 100, 50, 1.5);
    expect(zoomed.zoom).toBeCloseTo(1.5, 10);
    const after = worldUnderCursor(zoomed, 100, 50);
    expect(after.x).toBeCloseTo(before.x, 10);
    expect(after.y).toBeCloseTo(before.y, 10);
  });

  it('상한/하한에서 클램프되고, 이미 상한이면 뷰포트가 변하지 않는다', () => {
    const vp = { x: -30, y: 5, zoom: 1 };
    expect(zoomAt(vp, 0, 0, 100).zoom).toBe(ZOOM_MAX);
    expect(zoomAt(vp, 0, 0, 0.001).zoom).toBe(ZOOM_MIN);
    const atMax = { x: 7, y: -3, zoom: ZOOM_MAX };
    const still = zoomAt(atMax, 120, 40, 2);
    expect(still).toEqual(atMax);
  });
});

// ── 팬 ──────────────────────────────────────────────────────────────

describe('panBy', () => {
  it('화면 오프셋만 가산하고 줌은 유지한다', () => {
    expect(panBy({ x: 5, y: -2, zoom: 1.3 }, 10, -20)).toEqual({ x: 15, y: -22, zoom: 1.3 });
  });
});

// ── fit-to-view ─────────────────────────────────────────────────────

describe('fitViewport', () => {
  it('bounds가 없거나(빈 그래프) 화면 크기가 0이면 항등 뷰포트', () => {
    expect(fitViewport(null, 800, 400, 40)).toEqual({ x: 0, y: 0, zoom: 1 });
    expect(fitViewport({ minX: 0, minY: 0, maxX: 100, maxY: 50 }, 0, 0, 40)).toEqual({
      x: 0,
      y: 0,
      zoom: 1,
    });
  });

  it('여백을 뺀 화면에 들어오는 최대 줌 + 콘텐츠 중앙 정렬', () => {
    const vp = fitViewport({ minX: 0, minY: 0, maxX: 400, maxY: 100 }, 800, 400, 40);
    // zoom = min(720/400, 320/100) = 1.8, 중심 (200,50) → 화면 중앙 (400,200)
    expect(vp.zoom).toBeCloseTo(1.8, 10);
    expect(vp.x).toBeCloseTo(400 - 200 * 1.8, 10);
    expect(vp.y).toBeCloseTo(200 - 50 * 1.8, 10);
    // 검산: 콘텐츠 중심의 화면 좌표 = 화면 중앙
    expect(200 * vp.zoom + vp.x).toBeCloseTo(400, 10);
    expect(50 * vp.zoom + vp.y).toBeCloseTo(200, 10);
  });

  it('줌은 하한/상한으로 클램프된다 (작은 콘텐츠 확대 상한 / 큰 콘텐츠 축소 하한)', () => {
    const tiny = fitViewport({ minX: 0, minY: 0, maxX: 10, maxY: 10 }, 800, 400, 40);
    expect(tiny.zoom).toBe(ZOOM_MAX);
    const huge = fitViewport({ minX: 0, minY: 0, maxX: 1000000, maxY: 100 }, 800, 400, 40);
    expect(huge.zoom).toBe(ZOOM_MIN);
  });
});

// ── 체인 레이아웃 (1행 — 기본 SINGLE_ROW) ───────────────────────────

describe('chain layout', () => {
  it('노드 x는 결정론적 등간격 (마진 + 인덱스 × (노드폭+간격))', () => {
    expect(chainNodeX(0)).toBe(CHAIN_MARGIN_X);
    expect(chainNodeX(1) - chainNodeX(0)).toBe(NODE_W + NODE_GAP_X);
    expect(NODE_PITCH_X).toBe(NODE_W + NODE_GAP_X);
    expect(chainCenterX(2)).toBe(chainNodeX(2) + NODE_W / 2);
  });

  it('perRow 기본값은 줄바꿈 없음 — 전부 0행, y는 CHAIN_Y 고정', () => {
    for (const i of [0, 1, 7, 40]) {
      expect(chainRow(i)).toBe(0);
      expect(chainCol(i)).toBe(i);
      expect(chainNodeY(i)).toBe(CHAIN_Y);
    }
    expect(chainRow(40, SINGLE_ROW)).toBe(0);
  });

  it('chainBounds: 빈 체인은 null, 노드가 있으면 전체 노드를 포함한다', () => {
    expect(chainBounds(0)).toBeNull();
    const bounds = chainBounds(3);
    expect(bounds).not.toBeNull();
    if (bounds === null) return;
    expect(bounds.minX).toBeLessThan(chainNodeX(0));
    expect(bounds.maxX).toBeGreaterThan(chainNodeX(2) + NODE_W);
    expect(bounds.minY).toBeLessThan(CHAIN_Y);
    expect(bounds.maxY).toBeGreaterThan(CHAIN_Y);
  });

  it('클릭/드래그 구분 임계는 5px (Phase 8 요구)', () => {
    expect(DRAG_THRESHOLD_PX).toBe(5);
  });
});

// ── snake 레이아웃 (C-10 — 확장성 천장 해제) ────────────────────────

describe('nodesPerRow (폭 전용 근사 — 높이를 모를 때의 폴백)', () => {
  it('측정 불가(0/음수/NaN)면 기본값', () => {
    expect(nodesPerRow(0)).toBe(DEFAULT_PER_ROW);
    expect(nodesPerRow(-100)).toBe(DEFAULT_PER_ROW);
    expect(nodesPerRow(Number.NaN)).toBe(DEFAULT_PER_ROW);
  });

  it('아무리 좁아도 MIN_PER_ROW 아래로 접지 않는다 (1열 세로 스크롤 방지)', () => {
    expect(nodesPerRow(200)).toBe(MIN_PER_ROW);
    expect(nodesPerRow(1)).toBe(MIN_PER_ROW);
  });

  it('폭에 대해 단조 증가하고, 한 줄이 페인 폭 안에 실제로 들어간다', () => {
    const widths = [900, 1366, 1600, 1920, 2560];
    const perRows = widths.map((w) => nodesPerRow(w));
    for (let i = 1; i < perRows.length; i += 1) {
      expect(perRows[i]! >= perRows[i - 1]!, `${widths[i]}px`).toBe(true);
    }
    // 계산된 줄이 실제로 fit 여백 안에 들어차야 한다 (= fit 줌이 1 밑으로 붕괴하지 않는다)
    for (const w of widths) {
      const bounds = chainBounds(nodesPerRow(w), nodesPerRow(w));
      expect(bounds).not.toBeNull();
      if (bounds === null) continue;
      expect(bounds.maxX - bounds.minX, `${w}px`).toBeLessThanOrEqual(w - 40 * 2);
    }
  });

  it('리사이즈로 값이 바뀐다 (캔버스는 이 변화에 재렌더로 반응한다)', () => {
    expect(nodesPerRow(1920)).toBeGreaterThan(nodesPerRow(900));
  });

  it('폭만 보면 손해를 보는 구간이 실제로 존재한다 — bestPerRow가 필요한 이유', () => {
    // 실측 회귀 조건: 1366×768 창의 flowGraph 페인 = 836×240
    const per = nodesPerRow(836);
    expect(per).toBe(2); // 7노드가 4줄로 접힌다
    const wrapped = fitViewport(chainBounds(7, per), 836, 240, 40).zoom;
    const flat = fitViewport(chainBounds(7, SINGLE_ROW), 836, 240, 40).zoom;
    expect(wrapped).toBeLessThan(flat); // ← 폭 전용 근사의 결함(문서화된 폴백 한계)
  });
});

// ── bestPerRow — fit 줌 argmax (폭·높이 동시 고려) ──────────────────

describe('bestPerRow', () => {
  /** 실측 회귀 조건: 1366×768 창의 flowGraph 페인 */
  const PANE_W = 836;
  const PANE_H = 240;
  const zoomWith = (count: number, per: number, w = PANE_W, h = PANE_H): number =>
    fitViewport(chainBounds(count, per), w, h, 40).zoom;

  it('항상 1 ≤ k ≤ n', () => {
    for (const n of [1, 2, 3, 7, 12, 42, 80]) {
      const k = bestPerRow(n, PANE_W, PANE_H);
      expect(k, `n=${n}`).toBeGreaterThanOrEqual(1);
      expect(k, `n=${n}`).toBeLessThanOrEqual(n);
      expect(Number.isInteger(k), `n=${n}`).toBe(true);
    }
  });

  it('회귀 고정: 836×240 / 7노드에서 단일 행보다 나쁘지 않다', () => {
    const k = bestPerRow(7, PANE_W, PANE_H);
    expect(zoomWith(7, k)).toBeGreaterThanOrEqual(zoomWith(7, SINGLE_ROW));
  });

  it('회귀 고정: 836×240 / 7노드는 LOD full을 유지한다 (라벨 판독 가능)', () => {
    const k = bestPerRow(7, PANE_W, PANE_H);
    const zoom = zoomWith(7, k);
    expect(zoom).toBeGreaterThanOrEqual(LOD_ZOOM_THRESHOLD);
    expect(nodeLod(zoom)).toBe('full');
    // 구 구현(폭 전용, perRow=2)은 33%로 떨어져 칩으로 렌더됐다
    expect(zoom).toBeGreaterThan(zoomWith(7, nodesPerRow(PANE_W)));
  });

  it('어떤 노드 수에서도 단일 행보다 나쁠 수 없다 (단일 행이 후보에 포함된다)', () => {
    for (const n of [1, 2, 3, 5, 7, 9, 12, 20, 30, 50, 80]) {
      const k = bestPerRow(n, PANE_W, PANE_H);
      expect(zoomWith(n, k), `n=${n}`).toBeGreaterThanOrEqual(zoomWith(n, SINGLE_ROW));
    }
  });

  it('큰 체인에서는 여전히 wrap이 이긴다 (n=50 · n=80)', () => {
    for (const n of [50, 80]) {
      const k = bestPerRow(n, PANE_W, PANE_H);
      expect(k, `n=${n}`).toBeLessThan(n);
      // 1행은 줌 하한에 눌러앉는다 = "맞춤"이 전체를 담지 못하고 잘려 나간다
      expect(zoomWith(n, SINGLE_ROW), `n=${n}`).toBe(ZOOM_MIN);
      // wrap은 하한 위 = 전체가 담긴다
      expect(zoomWith(n, k), `n=${n}`).toBeGreaterThan(ZOOM_MIN);
      expect(zoomWith(n, k) / zoomWith(n, SINGLE_ROW), `n=${n}`).toBeGreaterThan(1.9);
    }
  });

  it('동률이면 더 큰 k(= 적은 행 수)를 고른다 — 가로 스캔 선호', () => {
    // 836×240 / 7노드: k=4,5,6이 전부 세로 구속(2줄)으로 같은 배율이다 → 6
    expect(bestPerRow(7, PANE_W, PANE_H)).toBe(6);
  });

  it('작은 체인이 넓은 페인에 들어가면 단일 행을 고른다', () => {
    expect(bestPerRow(3, 1600, 400)).toBe(3);
    expect(bestPerRow(1, PANE_W, PANE_H)).toBe(1);
  });

  it('페인 높이를 모르면(0/NaN) 폭 전용 근사로 폴백한다', () => {
    expect(bestPerRow(20, 836, 0)).toBe(nodesPerRow(836));
    expect(bestPerRow(20, 836, Number.NaN)).toBe(nodesPerRow(836));
    expect(bestPerRow(20, 0, 0)).toBe(DEFAULT_PER_ROW);
    // 폴백도 1 ≤ k ≤ n을 지킨다
    expect(bestPerRow(1, 836, 0)).toBe(1);
  });

  it('노드가 없으면 레이아웃이 쓰이지 않으므로 기본값', () => {
    expect(bestPerRow(0, PANE_W, PANE_H)).toBe(DEFAULT_PER_ROW);
  });

  it('페인이 커지면 줄당 노드 수가 줄지 않는다 (단조성)', () => {
    const n = 40;
    const widths = [700, 836, 1100, 1400, 1800];
    const ks = widths.map((w) => bestPerRow(n, w, PANE_H));
    for (let i = 1; i < ks.length; i += 1) {
      expect(ks[i]! >= ks[i - 1]!, `${widths[i]}px`).toBe(true);
    }
  });
});

describe('snake 좌표', () => {
  const PER_ROW = 4;

  it('perRow개마다 다음 줄로 접는다 — (index % perRow, floor(index / perRow))', () => {
    expect(chainCol(0, PER_ROW)).toBe(0);
    expect(chainCol(3, PER_ROW)).toBe(3);
    expect(chainCol(4, PER_ROW)).toBe(0);
    expect(chainRow(3, PER_ROW)).toBe(0);
    expect(chainRow(4, PER_ROW)).toBe(1);
    expect(chainRow(9, PER_ROW)).toBe(2);

    expect(chainNodeX(4, PER_ROW)).toBe(chainNodeX(0, PER_ROW));
    expect(chainNodeY(4, PER_ROW)).toBe(CHAIN_Y + ROW_PITCH_Y);
    expect(chainNodeY(9, PER_ROW)).toBe(CHAIN_Y + ROW_PITCH_Y * 2);
    expect(chainCenterY(4, PER_ROW)).toBe(chainNodeY(4, PER_ROW) + NODE_H / 2);
  });

  it('chainRowCount: 마지막 노드가 속한 줄 + 1', () => {
    expect(chainRowCount(0, PER_ROW)).toBe(0);
    expect(chainRowCount(1, PER_ROW)).toBe(1);
    expect(chainRowCount(4, PER_ROW)).toBe(1);
    expect(chainRowCount(5, PER_ROW)).toBe(2);
    expect(chainRowCount(42, PER_ROW)).toBe(11);
    expect(chainRowCount(42)).toBe(1); // 1행 체인
  });

  it('chainBounds: 가장 넓은 줄의 폭 + 줄 수만큼의 높이', () => {
    const one = chainBounds(3, PER_ROW);
    const many = chainBounds(9, PER_ROW);
    expect(one).not.toBeNull();
    expect(many).not.toBeNull();
    if (one === null || many === null) return;
    // 3노드는 아직 1줄 — 폭은 3노드분
    expect(one.maxX - one.minX).toBeLessThan(many.maxX - many.minX);
    // 9노드는 3줄 — 마지막 줄 하단까지 포함
    expect(many.maxY).toBeGreaterThan(CHAIN_Y + ROW_PITCH_Y * 2 + NODE_H);
    // 가장 넓은 줄은 perRow개다 — 9개여도 가로 폭은 "꽉 찬 한 줄"에서 더 늘지 않는다
    const fullRow = chainBounds(PER_ROW, PER_ROW);
    expect(fullRow).not.toBeNull();
    if (fullRow === null) return;
    expect(many.maxX).toBe(fullRow.maxX);
    expect(chainBounds(42, PER_ROW)?.maxX).toBe(fullRow.maxX);
  });

  // C-10의 실측 조건: 1366×768에서 플로우 페인 높이 ≈ 300px
  const PANE_W = 1366;
  const PANE_H = 300;
  const fitZoom = (count: number, per: number): number =>
    fitViewport(chainBounds(count, per), PANE_W, PANE_H, 40).zoom;
  /** 캔버스가 실제로 고르는 레이아웃 */
  const chosen = (count: number): number => bestPerRow(count, PANE_W, PANE_H);

  /**
   * "맞춤"이 전체를 담는 최대 노드 수 = 필요한 배율이 줌 하한 위인 최대 n.
   * 하한에 걸리는 순간 콘텐츠가 화면 밖으로 잘려 나간다("전체를 볼 수 없는 편집기").
   */
  const maxFittingCount = (perOf: (n: number) => number, zoomMin: number): number => {
    let max = 0;
    for (let n = 1; n <= 200; n += 1) {
      if (fitZoom(n, perOf(n)) > zoomMin + 1e-9) max = n;
      else break;
    }
    return max;
  };

  it('구 조합(1행 + 하한 0.4)의 규모 천장을 새 조합(snake + 0.12)이 걷어낸다', () => {
    const oldMax = maxFittingCount(() => SINGLE_ROW, 0.4);
    const newMax = maxFittingCount(chosen, ZOOM_MIN);
    // 구 천장은 실제 워크셀 시퀀스(30~80스텝)에 한참 못 미쳤다
    expect(oldMax).toBeLessThan(16);
    // 새 조합은 80스텝을 담는다
    expect(newMax).toBeGreaterThanOrEqual(80);
  });

  it('snake는 9노드 이상에서 1행보다 큰 fit 줌을 준다 (천장이 시작되는 지점부터)', () => {
    for (const count of [10, 15, 30, 42]) {
      expect(fitZoom(count, chosen(count)), `${count}노드`).toBeGreaterThan(
        fitZoom(count, SINGLE_ROW),
      );
    }
    // 42노드 실측: 1행에서는 40% 줌에서도 라벨이 판독 불가였다(C-10). snake는 그 1.5배 이상.
    expect(fitZoom(42, chosen(42)) / fitZoom(42, SINGLE_ROW)).toBeGreaterThan(1.5);
  });

  it('작은 체인에서는 절대 1행보다 나빠지지 않는다 (회귀 방향 고정)', () => {
    for (const count of [2, 3, 5, 7, 9]) {
      expect(fitZoom(count, chosen(count)), `${count}노드`).toBeGreaterThanOrEqual(
        fitZoom(count, SINGLE_ROW),
      );
    }
  });

  it('현재 데모 규모(7노드)는 라벨이 살아 있는 배율이다', () => {
    expect(fitZoom(7, chosen(7))).toBeGreaterThan(LOD_ZOOM_THRESHOLD);
  });

  it('chainRowAt: 줄 사이 중점이 경계이고 범위를 벗어나면 클램프', () => {
    expect(chainRowAt(CHAIN_Y + NODE_H / 2, 3)).toBe(0);
    expect(chainRowAt(CHAIN_Y + NODE_H / 2 + ROW_PITCH_Y, 3)).toBe(1);
    expect(chainRowAt(CHAIN_Y + NODE_H / 2 + ROW_PITCH_Y * 0.51, 3)).toBe(1);
    expect(chainRowAt(CHAIN_Y + NODE_H / 2 + ROW_PITCH_Y * 0.49, 3)).toBe(0);
    expect(chainRowAt(-9999, 3)).toBe(0);
    expect(chainRowAt(9999, 3)).toBe(2);
    expect(chainRowAt(9999, 1)).toBe(0);
  });
});

// ── snake 레이아웃은 직렬화에 영향이 없다 (불변식 §2.8) ─────────────

describe('레이아웃 ↔ 직렬화 독립 (§2.8 무손실 왕복)', () => {
  const sequence: ControlSequence = {
    id: 'seq-snake',
    robot: 'arm',
    steps: [
      { kind: 'label', name: 'start' },
      { kind: 'moveJoints', targets: { joint1: 0.4, joint2: -0.2 }, durationSec: 2 },
      { kind: 'gripper', state: 'close', durationSec: 0.5 },
      { kind: 'wait', durationSec: 1 },
      { kind: 'waitForCollision', between: ['arm', 'box_a'], timeoutSec: 5 },
      { kind: 'setJoints', targets: { joint1: 0 } },
      { kind: 'goto', label: 'start', times: 3 },
    ],
  };

  it('ui.x/y를 snake 좌표로 덮어써도 toSequence 결과가 한 글자도 변하지 않는다', () => {
    const graph = fromSequence(sequence);
    const before = toSequence(graph, { id: sequence.id });
    // 캔버스가 쓰는 레이아웃을 그래프에 직접 반영한다 (최악의 경우 가정)
    const per = nodesPerRow(1366);
    graph.nodes.forEach((node, i) => {
      node.ui = { x: chainNodeX(i, per), y: chainNodeY(i, per) };
    });
    const after = toSequence(graph, { id: sequence.id });
    expect(after).toEqual(before);
    expect(after.steps).toEqual(sequence.steps);
  });

  it('perRow가 달라져도(리사이즈) 같은 시퀀스를 낸다', () => {
    const graph = fromSequence(sequence);
    const narrow = toSequence(
      { ...graph, nodes: graph.nodes.map((n, i) => ({ ...n, ui: { x: chainNodeX(i, 2), y: chainNodeY(i, 2) } })) },
      { id: sequence.id },
    );
    const wide = toSequence(
      { ...graph, nodes: graph.nodes.map((n, i) => ({ ...n, ui: { x: chainNodeX(i, 9), y: chainNodeY(i, 9) } })) },
      { id: sequence.id },
    );
    expect(narrow).toEqual(wide);
    expect(wide.steps).toEqual(sequence.steps);
  });

  it('snake 드롭으로 계산한 인덱스를 moveNode에 넣으면 순서가 그대로 반영된다', () => {
    const graph = fromSequence(sequence);
    const per = 3;
    const count = graph.nodes.length;
    // 마지막 노드(goto)를 두 번째 줄 첫 칸 왼쪽에 드롭 → 최종 인덱스 3
    const dropX = chainCenterX(3, per) - 1;
    const dropY = chainCenterY(3, per);
    const target = reorderTargetIndexAt(count, per, count - 1, dropX, dropY);
    expect(target).toBe(3);
    const moved = moveNode(graph, graph.nodes[count - 1]!.id, target);
    expect(moved.ok).toBe(true);
    if (!moved.ok) return;
    expect(moved.graph.nodes.map((n) => n.kind)).toEqual([
      'label',
      'moveJoints',
      'gripper',
      'goto',
      'wait',
      'waitForCollision',
      'setJoints',
    ]);
    // 편집 후에도 직렬화가 가능하다 (finishEdit이 이미 검증했지만 왕복을 명시한다)
    expect(toSequence(moved.graph).steps).toHaveLength(count);
  });
});

// ── drop x → 삽입 인덱스 ────────────────────────────────────────────

describe('insertionIndexFromPoint', () => {
  const centers = [100, 200, 300];

  it('빈 체인은 항상 0', () => {
    expect(insertionIndexFromPoint([], 123)).toBe(0);
  });

  it('노드 중심 기준: 왼쪽 절반은 그 앞, 오른쪽 절반은 그 뒤', () => {
    expect(insertionIndexFromPoint(centers, 50)).toBe(0); // 첫 노드 앞
    expect(insertionIndexFromPoint(centers, 150)).toBe(1); // 0과 1 사이
    expect(insertionIndexFromPoint(centers, 250)).toBe(2); // 1과 2 사이
    expect(insertionIndexFromPoint(centers, 350)).toBe(3); // 끝
  });

  it('경계값: 중심과 정확히 같으면 그 노드 앞 (strict <)', () => {
    expect(insertionIndexFromPoint(centers, 100)).toBe(0);
    expect(insertionIndexFromPoint(centers, 200)).toBe(1);
  });
});

describe('reorderTargetIndex', () => {
  const centers = [100, 200, 300];

  it('드래그 노드를 뺀 체인 기준의 최종 인덱스 (moveNode toIndex 규약)', () => {
    // 노드0을 노드1 오른쪽에 드롭 → 최종 1
    expect(reorderTargetIndex(centers, 0, 250)).toBe(1);
    // 노드2를 맨 앞에 드롭 → 최종 0
    expect(reorderTargetIndex(centers, 2, 50)).toBe(0);
    // 노드0을 맨 끝에 드롭 → 최종 2
    expect(reorderTargetIndex(centers, 0, 999)).toBe(2);
  });

  it('원래 자리 근처 드롭은 fromIndex와 같은 값 = no-op 판정', () => {
    expect(reorderTargetIndex(centers, 1, 150)).toBe(1); // 노드0과 노드2 사이 = 원위치
    expect(reorderTargetIndex(centers, 0, 50)).toBe(0);
    expect(reorderTargetIndex(centers, 2, 350)).toBe(2);
  });

  it('실제 체인 레이아웃 좌표와 합성해도 동일 규약', () => {
    const layout = [0, 1, 2, 3].map((i) => chainCenterX(i));
    // 노드3을 노드0 왼쪽 절반에 드롭 → 최종 0
    expect(reorderTargetIndex(layout, 3, chainCenterX(0) - 1)).toBe(0);
    // 노드1을 노드2 오른쪽 절반에 드롭 → 최종 2
    expect(reorderTargetIndex(layout, 1, chainCenterX(2) + 1)).toBe(2);
  });
});

describe('insertionLineX', () => {
  it('중간 틈: 대상 노드 앞 (간격 절반 지점)', () => {
    // 3개 체인에서 노드2를 최종 0으로 → 원래 노드0 앞 틈
    expect(insertionLineX(2, 0, 3)).toBe(chainNodeX(0) - NODE_GAP_X / 2);
    // 노드0을 최종 1로 → 남은 체인 [1,2]에서 인덱스1 = 원래 노드2 앞 틈
    expect(insertionLineX(0, 1, 3)).toBe(chainNodeX(2) - NODE_GAP_X / 2);
  });

  it('체인 끝 틈: 마지막 남은 노드의 오른쪽', () => {
    expect(insertionLineX(0, 2, 3)).toBe(chainNodeX(2) + NODE_W + NODE_GAP_X / 2);
    // 마지막 노드를 드래그해 끝에 두면 남은 체인의 끝 = 원래 노드1 오른쪽
    expect(insertionLineX(2, 2, 3)).toBe(chainNodeX(1) + NODE_W + NODE_GAP_X / 2);
  });

  it('노드 1개뿐이면 첫 틈으로 방어한다', () => {
    expect(insertionLineX(0, 0, 1)).toBe(chainNodeX(0) - NODE_GAP_X / 2);
  });
});

describe('insertionLine (snake — 줄까지 돌려준다)', () => {
  const PER_ROW = 3;

  it('삽입선은 대상 노드가 있는 줄에 그려진다', () => {
    // 7개 체인(3/3/1)에서 노드6을 최종 4로 → 남은 체인 [0..5]의 인덱스4 = 원래 노드4 앞
    const line = insertionLine(6, 4, 7, PER_ROW);
    expect(line.x).toBe(chainNodeX(4, PER_ROW) - NODE_GAP_X / 2);
    expect(line.rowTopY).toBe(chainNodeY(4, PER_ROW));
    expect(line.rowTopY).toBe(CHAIN_Y + ROW_PITCH_Y); // 두 번째 줄
  });

  it('1행 경로(insertionLineX)와 x가 일치한다', () => {
    for (const final of [0, 1, 2]) {
      expect(insertionLine(2, final, 3).x).toBe(insertionLineX(2, final, 3));
    }
  });
});

describe('insertionIndexAt / reorderTargetIndexAt (snake)', () => {
  it('1행(SINGLE_ROW)에서는 기존 1D 경로와 결과가 완전히 같다', () => {
    const count = 5;
    const centers = Array.from({ length: count }, (_, i) => chainCenterX(i));
    const probes = [-500, chainCenterX(0) - 1, chainCenterX(0), chainCenterX(2) + 1, 99999];
    for (const dropX of probes) {
      expect(insertionIndexAt(count, SINGLE_ROW, dropX, CHAIN_Y)).toBe(
        insertionIndexFromPoint(centers, dropX),
      );
      for (const from of [0, 2, 4]) {
        expect(reorderTargetIndexAt(count, SINGLE_ROW, from, dropX, CHAIN_Y)).toBe(
          reorderTargetIndex(centers, from, dropX),
        );
      }
    }
  });

  it('읽기 순서 판정: 윗줄이면 무조건 앞, 같은 줄이면 중심 x 비교', () => {
    const per = 3;
    const count = 7; // 3 / 3 / 1
    // 두 번째 줄 첫 칸 왼쪽 → 앞줄 3개가 전부 앞
    expect(insertionIndexAt(count, per, chainCenterX(3, per) - 1, chainCenterY(3, per))).toBe(3);
    // 두 번째 줄 두 번째 칸 오른쪽 → 0,1,2,3,4
    expect(insertionIndexAt(count, per, chainCenterX(4, per) + 1, chainCenterY(4, per))).toBe(5);
    // 첫 줄 맨 왼쪽 → 0
    expect(insertionIndexAt(count, per, -999, chainCenterY(0, per))).toBe(0);
    // 마지막 줄 오른쪽 끝 → count
    expect(insertionIndexAt(count, per, 99999, chainCenterY(6, per))).toBe(count);
  });

  it('같은 x라도 줄이 다르면 다른 인덱스가 나온다 (1행 판정으로는 불가능했던 것)', () => {
    const per = 3;
    const count = 9;
    const x = chainCenterX(1, per) + 1; // 각 줄의 두 번째 칸 오른쪽
    expect(insertionIndexAt(count, per, x, chainCenterY(1, per))).toBe(2);
    expect(insertionIndexAt(count, per, x, chainCenterY(4, per))).toBe(5);
    expect(insertionIndexAt(count, per, x, chainCenterY(7, per))).toBe(8);
  });

  it('원위치 드롭은 fromIndex와 같은 값 = no-op 판정 (줄바꿈 경계 포함)', () => {
    const per = 3;
    const count = 7;
    for (const from of [0, 2, 3, 5, 6]) {
      const x = chainCenterX(from, per);
      const y = chainCenterY(from, per);
      expect(reorderTargetIndexAt(count, per, from, x, y), `from=${from}`).toBe(from);
    }
  });
});

// ── LOD (C-10) ──────────────────────────────────────────────────────

describe('nodeLod', () => {
  it('임계 0.5 미만은 축약, 이상은 전체', () => {
    expect(LOD_ZOOM_THRESHOLD).toBe(0.5);
    expect(nodeLod(ZOOM_MIN)).toBe('compact');
    expect(nodeLod(0.49)).toBe('compact');
    expect(nodeLod(0.5)).toBe('full');
    expect(nodeLod(1)).toBe('full');
    expect(nodeLod(ZOOM_MAX)).toBe('full');
  });

  it('NaN은 안전하게 full로 본다', () => {
    expect(nodeLod(Number.NaN)).toBe('full');
  });
});

// ── 노드 요약 텍스트 (UX §3.4 형식 고정) ────────────────────────────

describe('nodeSummary', () => {
  it('moveJoints: 첫 관절 + 나머지 개수 + duration', () => {
    expect(
      nodeSummary('moveJoints', { targets: { joint2: 0.2, joint3: 0.6 }, durationSec: 2 }),
    ).toBe('joint2→0.2 외 1 · 2.0s');
    expect(nodeSummary('moveJoints', { targets: { joint1: -0.5 }, durationSec: 1.5 })).toBe(
      'joint1→-0.5 · 1.5s',
    );
  });

  it('moveJoints: 부동소수 잡음은 반올림해 표기한다', () => {
    expect(
      nodeSummary('moveJoints', { targets: { j: 0.30000000000000004 }, durationSec: 1 }),
    ).toBe('j→0.3 · 1.0s');
  });

  it('setJoints: 관절 텍스트만 (즉시 적용 — duration 없음)', () => {
    expect(nodeSummary('setJoints', { targets: { joint1: 0, joint2: 1.2 } })).toBe(
      'joint1→0 외 1',
    );
  });

  it('gripper: 한국어 상태 + duration / 숫자 상태 그대로', () => {
    expect(nodeSummary('gripper', { state: 'close', durationSec: 0.5 })).toBe('닫기 · 0.5s');
    expect(nodeSummary('gripper', { state: 'open' })).toBe('열기');
    expect(nodeSummary('gripper', { state: 0.5 })).toBe('0.5');
  });

  it('wait: duration만', () => {
    expect(nodeSummary('wait', { durationSec: 1 })).toBe('1.0s');
    expect(nodeSummary('wait', { durationSec: 0.25 })).toBe('0.25s');
  });

  it('waitForCollision: 엔티티 쌍 × 타임아웃', () => {
    expect(
      nodeSummary('waitForCollision', { between: ['arm', 'box_a'], timeoutSec: 5 }),
    ).toBe('arm × box_a · 5s');
    expect(nodeSummary('waitForCollision', { between: ['arm', 'box_a'] })).toBe('arm × box_a');
  });

  it('label/goto: 라벨명 · 반복 횟수 (미지정 = 무한)', () => {
    expect(nodeSummary('label', { name: 'pick' })).toBe('pick');
    expect(nodeSummary('goto', { label: 'pick', times: 3 })).toBe('→ pick ×3');
    expect(nodeSummary('goto', { label: 'pick' })).toBe('→ pick ∞');
  });

  it('moveToPose: 목표 위치 + duration', () => {
    expect(
      nodeSummary('moveToPose', { target: { position: [0.1, 0.2, 0.3] }, durationSec: 2 }),
    ).toBe('[0.1, 0.2, 0.3] · 2.0s');
  });

  it('형태가 어긋난 params에도 throw하지 않고 그릴 수 있는 부분만 그린다', () => {
    expect(nodeSummary('moveJoints', { targets: 'oops', durationSec: 2 })).toBe('2.0s');
    expect(nodeSummary('waitForCollision', { between: [1, 2] })).toBe('');
    expect(nodeSummary('wait', {})).toBe('');
    expect(nodeSummary('unknown-kind', { anything: true })).toBe('');
  });
});

// ── 종류 메타 / 팔레트 그룹 ─────────────────────────────────────────

describe('kindMeta / PALETTE_GROUPS', () => {
  it('팔레트는 동작/시간/충돌/흐름 4그룹이고 모든 항목에 메타가 있다', () => {
    expect(PALETTE_GROUPS.map((g) => g.labelKo)).toEqual(['동작', '시간', '충돌', '흐름']);
    for (const group of PALETTE_GROUPS) {
      for (const kind of group.kinds) {
        const meta = kindMeta(kind);
        expect(meta.icon.length, kind).toBeGreaterThan(0);
        expect(meta.label.length, kind).toBeGreaterThan(0);
        expect(meta.groupKo, kind).toBe(group.labelKo);
        expect(meta.color.length, kind).toBeGreaterThan(0);
      }
    }
  });

  it('moveToPose는 팔레트에 없다 (IK 로드맵 — 실행 시 건너뜀) — 표시는 지원한다', () => {
    const paletteKinds = PALETTE_GROUPS.flatMap((g) => [...g.kinds]);
    expect(paletteKinds).not.toContain('moveToPose');
    expect(kindMeta('moveToPose').label).toBe('MoveToPose');
  });

  it('알 수 없는 kind도 안전하게 표시한다 (label = kind 그대로)', () => {
    const meta = kindMeta('someFutureKind');
    expect(meta.label).toBe('someFutureKind');
    expect(meta.icon.length).toBeGreaterThan(0);
  });

  it('아이콘은 SVG 아이콘 이름이다 — 이모지/딩벳 금지 (C-13)', () => {
    const kinds = [...PALETTE_GROUPS.flatMap((g) => [...g.kinds]), 'moveToPose', 'someFutureKind'];
    for (const kind of kinds) {
      const name: string = kindMeta(kind).icon;
      // IconName은 ASCII 식별자다 — 이모지가 들어오면 여기서 걸린다
      expect(name, kind).toMatch(/^[a-zA-Z][a-zA-Z0-9]*$/);
    }
  });

  it('범주 색은 CATEGORY 토큰에서 온다 — 시맨틱 재사용 금지 (C-14)', () => {
    expect(kindMeta('moveJoints').color).toBe(CATEGORY.motion);
    expect(kindMeta('setJoints').color).toBe(CATEGORY.motion);
    expect(kindMeta('gripper').color).toBe(CATEGORY.motion);
    expect(kindMeta('moveToPose').color).toBe(CATEGORY.motion);
    expect(kindMeta('wait').color).toBe(CATEGORY.time);
    expect(kindMeta('waitForCollision').color).toBe(CATEGORY.collision);
    expect(kindMeta('label').color).toBe(CATEGORY.flow);
    expect(kindMeta('goto').color).toBe(CATEGORY.flow);

    const categoryValues = new Set<string>(Object.values(CATEGORY));
    const paletteKinds = PALETTE_GROUPS.flatMap((g) => [...g.kinds]);
    for (const kind of paletteKinds) {
      expect(categoryValues.has(kindMeta(kind).color), kind).toBe(true);
      // 흐름 제어가 완료-초록과 같은 값이면 "초록 노드"의 의미가 무너진다 (C-14의 핵심)
      expect(kindMeta(kind).color, kind).not.toBe(COLOR.success);
    }
  });
});

// ── 상태/출처 표현 ──────────────────────────────────────────────────

describe('status / origin helpers', () => {
  it('상태 점 색: pending muted / active 액센트 / done 성공 / error 충돌 램프', () => {
    expect(statusColor('pending')).toBe(COLOR.muted);
    expect(statusColor('active')).toBe(COLOR.accent);
    expect(statusColor('done')).toBe(COLOR.success);
    // 오류는 충돌 램프에서 나온다 — 3D 펄스/접촉점 마커와 같은 값 (C-7)
    expect(statusColor('error')).toBe(COLLISION.base);
  });

  it('4개 상태가 서로 다른 색이다 (색 하나에 두 의미를 겸직시키지 않는다)', () => {
    const colors = (['pending', 'active', 'done', 'error'] as const).map(statusColor);
    expect(new Set(colors).size).toBe(4);
  });

  it('상태 텍스트 채널 (색만으로 전달 금지 — UX §9)', () => {
    expect(statusLabelKo('pending')).toBe('대기');
    expect(statusLabelKo('active')).toBe('실행중');
    expect(statusLabelKo('done')).toBe('완료');
    expect(statusLabelKo('error')).toBe('오류');
  });

  it('출처 배지: generated → AI, modified → 수정됨, manual → 없음', () => {
    expect(originBadge('generated')).toBe('AI');
    expect(originBadge('modified')).toBe('수정됨');
    expect(originBadge('manual')).toBeNull();
  });
});

// ── 텍스트/숫자 포맷 ────────────────────────────────────────────────

describe('format helpers', () => {
  it('truncateText: 예산 안은 그대로, 초과는 말줄임', () => {
    expect(truncateText('short', 10)).toBe('short');
    expect(truncateText('exactly-10', 10)).toBe('exactly-10');
    expect(truncateText('this-is-way-too-long', 10)).toBe('this-is-w…');
    expect(truncateText('this-is-way-too-long', 10).length).toBe(10);
  });

  it('formatDurationSec: 0.1s 해상도는 소수 1자리 고정, 세밀한 값은 그대로', () => {
    expect(formatDurationSec(2)).toBe('2.0s');
    expect(formatDurationSec(1)).toBe('1.0s');
    expect(formatDurationSec(0.5)).toBe('0.5s');
    expect(formatDurationSec(0.25)).toBe('0.25s');
  });

  it('formatNum: 소수 3자리 반올림 + 뒤 0 제거', () => {
    expect(formatNum(0.30000000000000004)).toBe('0.3');
    expect(formatNum(5)).toBe('5');
    expect(formatNum(-0.5)).toBe('-0.5');
    expect(formatNum(1.23456)).toBe('1.235');
  });
});
