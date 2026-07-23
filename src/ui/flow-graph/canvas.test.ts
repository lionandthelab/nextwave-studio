// ui/flow-graph/canvas.test.ts — 캔버스 순수 헬퍼 단위 테스트 (DOM 비의존, node 환경)
//
// mountFlowCanvas의 DOM 조립/포인터 배선은 브라우저 게이트(gate-browser.mjs) 몫이다.
// 여기서는 Phase 8 요구의 순수 계산만 검증한다:
// - 팬/줌 수학 (clampZoom · zoomAt 커서 중심 불변 · panBy · fitViewport)
// - 드래그 재정렬의 drop x → 삽입 인덱스 (insertionIndexFromPoint · reorderTargetIndex ·
//   insertionLineX) 와 체인 레이아웃 (chainNodeX/chainCenterX/chainBounds)
// - 노드 요약 텍스트 (node-render.nodeSummary — UX §3.4 예시 형식 고정)
// - 종류 메타/팔레트 그룹/상태·출처 표현 헬퍼

import { describe, expect, it } from 'vitest';
import {
  CHAIN_MARGIN_X,
  CHAIN_Y,
  DRAG_THRESHOLD_PX,
  NODE_GAP_X,
  NODE_W,
  ZOOM_MAX,
  ZOOM_MIN,
  chainBounds,
  chainCenterX,
  chainNodeX,
  clampZoom,
  fitViewport,
  insertionIndexFromPoint,
  insertionLineX,
  panBy,
  reorderTargetIndex,
  zoomAt,
} from './canvas';
import {
  PALETTE_GROUPS,
  formatDurationSec,
  formatNum,
  kindMeta,
  nodeSummary,
  originBadge,
  statusColor,
  statusLabelKo,
  truncateText,
} from './node-render';
import { COLOR } from '../theme';

// ── 줌 클램프 ───────────────────────────────────────────────────────

describe('clampZoom', () => {
  it('0.4–2.0 요구 범위로 클램프한다', () => {
    expect(ZOOM_MIN).toBe(0.4);
    expect(ZOOM_MAX).toBe(2.0);
    expect(clampZoom(1)).toBe(1);
    expect(clampZoom(0.1)).toBe(ZOOM_MIN);
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

  it('줌은 0.4–2.0으로 클램프된다 (작은 콘텐츠 확대 상한 / 큰 콘텐츠 축소 하한)', () => {
    const tiny = fitViewport({ minX: 0, minY: 0, maxX: 10, maxY: 10 }, 800, 400, 40);
    expect(tiny.zoom).toBe(ZOOM_MAX);
    const huge = fitViewport({ minX: 0, minY: 0, maxX: 100000, maxY: 100 }, 800, 400, 40);
    expect(huge.zoom).toBe(ZOOM_MIN);
  });
});

// ── 체인 레이아웃 ───────────────────────────────────────────────────

describe('chain layout', () => {
  it('노드 x는 결정론적 등간격 (마진 + 인덱스 × (노드폭+간격))', () => {
    expect(chainNodeX(0)).toBe(CHAIN_MARGIN_X);
    expect(chainNodeX(1) - chainNodeX(0)).toBe(NODE_W + NODE_GAP_X);
    expect(chainCenterX(2)).toBe(chainNodeX(2) + NODE_W / 2);
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
});

// ── 상태/출처 표현 ──────────────────────────────────────────────────

describe('status / origin helpers', () => {
  it('상태 점 색: pending 회색 / active 액센트 / done 성공 / error 오류 (theme 토큰)', () => {
    expect(statusColor('pending')).toBe(COLOR.muted);
    expect(statusColor('active')).toBe(COLOR.accent);
    expect(statusColor('done')).toBe(COLOR.success);
    expect(statusColor('error')).toBe(COLOR.error);
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
