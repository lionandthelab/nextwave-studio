// ui/workspace.test.ts — 워크스페이스 셸 순수 로직 단위 테스트 (DOM 비의존, node)
//
// mountWorkspace의 DOM 조립(grid/슬롯/스플리터 배선)은 얇은 글루라 여기서 다루지
// 않는다(브라우저 게이트 몫). 여기서는 스플리터 크기 계산(clamp/드래그 방향 부호),
// 크기 상수 계약, flowMode 높이 매핑, 반응형 자동 접힘 판정을 검증한다.

import { describe, expect, it } from 'vitest';
import {
  DOCK_COLLAPSED_PX,
  FLOW_STRIP_PX,
  LEFT_RAIL_WIDTH_PX,
  RIGHT_RAIL_WIDTH_PX,
  VIEWPORT_MIN_HEIGHT_PX,
  WORKSPACE_SIZES,
  clampPanelSize,
  decideAutoLayout,
  dragPanelSize,
  flowModeHeightCss,
} from './workspace';
import { BREAKPOINT, LAYOUT } from './theme';

// ── 크기 상수 계약 ──────────────────────────────────────────────────

describe('WORKSPACE_SIZES', () => {
  it('모든 패널: min ≤ default ≤ max', () => {
    for (const [name, limits] of Object.entries(WORKSPACE_SIZES)) {
      expect(limits.minPx, name).toBeLessThanOrEqual(limits.defaultPx);
      expect(limits.defaultPx, name).toBeLessThanOrEqual(limits.maxPx);
    }
  });

  it('flowGraph 하한은 flowHeightCss의 clamp 하한(148px)과 일치한다', () => {
    // 구 하한 200px은 "그래프가 항상 보여야 한다"는 Phase 7 요구를 지키려던 값이다.
    // 그 요구는 이제 strip 모드(56px)가 더 싸게 만족시키므로, full 모드 하한은
    // clamp() 하한에 맞춰 내려간다 — 그만큼 뷰포트가 픽셀을 돌려받는다 (UX_AUDIT C-1).
    expect(WORKSPACE_SIZES.flowGraph.minPx).toBe(148);
    expect(LAYOUT.flowHeightCss).toContain('148px');
  });

  it('좌 패널 기본 ~240px · 접힘 레일 36px (UX §2)', () => {
    expect(WORKSPACE_SIZES.left.defaultPx).toBe(240);
    expect(LEFT_RAIL_WIDTH_PX).toBe(36);
    // 레일은 최소 폭보다 좁아야 "접힘"이 의미가 있다
    expect(LEFT_RAIL_WIDTH_PX).toBeLessThan(WORKSPACE_SIZES.left.minPx);
  });

  it('우 패널도 접을 수 있다 — 레일 폭이 최소 폭보다 좁다 (UX §2 "좌/우 패널")', () => {
    expect(RIGHT_RAIL_WIDTH_PX).toBeLessThan(WORKSPACE_SIZES.right.minPx);
  });

  it('독 접힘 높이는 펼침 하한보다 작다 — 접기가 실제로 픽셀을 돌려준다', () => {
    // 구 구현은 접어도 그리드 슬롯 211px이 그대로 남아 뷰포트가 1px도 커지지 않았다.
    expect(DOCK_COLLAPSED_PX).toBeLessThan(WORKSPACE_SIZES.dock.minPx);
  });

  it('세로 크롬 최소 합이 뷰포트 하한을 침범하지 않는다 (720p 기준)', () => {
    // 최악의 경우: 커맨드바 + 독(접힘) + 그래프(strip) + 스플리터 2개
    const chrome = LAYOUT.barHeightPx + DOCK_COLLAPSED_PX + FLOW_STRIP_PX + 5 * 2;
    expect(720 - chrome).toBeGreaterThanOrEqual(VIEWPORT_MIN_HEIGHT_PX);
  });
});

// ── clampPanelSize ──────────────────────────────────────────────────

describe('clampPanelSize', () => {
  const limits = { defaultPx: 240, minPx: 180, maxPx: 420 };

  it('범위 안 값은 그대로', () => {
    expect(clampPanelSize(240, limits)).toBe(240);
    expect(clampPanelSize(180, limits)).toBe(180);
    expect(clampPanelSize(420, limits)).toBe(420);
  });

  it('범위 밖 값은 경계로 클램프', () => {
    expect(clampPanelSize(0, limits)).toBe(180);
    expect(clampPanelSize(-50, limits)).toBe(180);
    expect(clampPanelSize(10000, limits)).toBe(420);
  });

  it('NaN은 min으로 방어한다', () => {
    expect(clampPanelSize(Number.NaN, limits)).toBe(180);
  });
});

// ── dragPanelSize (드래그 방향 부호 규약) ───────────────────────────

describe('dragPanelSize', () => {
  const limits = { defaultPx: 240, minPx: 180, maxPx: 420 };

  it('sign=+1: 포인터 +방향 이동이 패널을 키운다 (좌 패널)', () => {
    expect(dragPanelSize(240, 60, 1, limits)).toBe(300);
    expect(dragPanelSize(240, -60, 1, limits)).toBe(180);
  });

  it('sign=-1: 포인터 -방향 이동이 패널을 키운다 (우 패널/독/flowGraph)', () => {
    expect(dragPanelSize(240, -60, -1, limits)).toBe(300);
    expect(dragPanelSize(240, 60, -1, limits)).toBe(180);
  });

  it('드래그 결과도 min/max로 클램프된다', () => {
    expect(dragPanelSize(240, 10000, 1, limits)).toBe(420);
    expect(dragPanelSize(240, -10000, 1, limits)).toBe(180);
  });

  it('클램프 후 반대 방향 드래그는 시작 크기 기준으로 계산된다 (누적 오차 없음)', () => {
    // 스플리터 구현은 dragStartSize를 고정해 매 move마다 재계산한다 — 그 계약의 순수 검증
    const overshoot = dragPanelSize(240, 500, 1, limits); // 420으로 클램프
    expect(overshoot).toBe(420);
    // 시작 크기(240) 기준 -10px → 230 (클램프된 420 기준이 아니다)
    expect(dragPanelSize(240, -10, 1, limits)).toBe(230);
  });
});

// ── flowMode 높이 매핑 (UX_AUDIT C-9) ───────────────────────────────

describe('flowModeHeightCss', () => {
  it('off는 0, strip은 고정 56px, full은 조절된 높이', () => {
    expect(flowModeHeightCss('off', 300)).toBe('0px');
    expect(flowModeHeightCss('strip', 300)).toBe(`${FLOW_STRIP_PX}px`);
    expect(flowModeHeightCss('full', 300)).toBe('300px');
  });

  it('strip은 full의 어떤 값보다도 작다 — 실행 중 뷰포트가 픽셀을 돌려받는다', () => {
    expect(FLOW_STRIP_PX).toBeLessThan(WORKSPACE_SIZES.flowGraph.minPx);
  });
});

// ── 반응형 자동 접힘 판정 (UX_DESIGN §8 — 구현이 0개이던 축) ────────

describe('decideAutoLayout', () => {
  it('와이드 화면에서는 아무것도 접지 않는다', () => {
    const d = decideAutoLayout(1920, 1080);
    expect(d).toEqual({
      collapseLeft: false,
      collapseRight: false,
      collapseDock: false,
      compactBar: false,
    });
  });

  it('1024×768 태블릿: 좌 패널이 자동으로 레일이 된다', () => {
    const d = decideAutoLayout(1024, 768);
    expect(d.collapseLeft).toBe(true);
    expect(d.collapseRight).toBe(false); // 900px 미만이 아니다
  });

  it('768px 세로 화면: 좌·우 둘 다 접힌다 (폭의 68.5%를 패널이 먹던 상태 해소)', () => {
    const d = decideAutoLayout(768, 1024);
    expect(d.collapseLeft).toBe(true);
    expect(d.collapseRight).toBe(true);
    expect(d.compactBar).toBe(true);
  });

  it('낮은 창 높이에서는 독이 자동으로 접힌다', () => {
    expect(decideAutoLayout(1920, 600).collapseDock).toBe(true);
    expect(decideAutoLayout(1920, 700).collapseDock).toBe(false);
  });

  it('브레이크포인트 경계는 미만(<) 규약이다', () => {
    expect(decideAutoLayout(BREAKPOINT.autoCollapseLeftPx, 1080).collapseLeft).toBe(false);
    expect(decideAutoLayout(BREAKPOINT.autoCollapseLeftPx - 1, 1080).collapseLeft).toBe(true);
    expect(decideAutoLayout(BREAKPOINT.autoCollapseRightPx - 1, 1080).collapseRight).toBe(true);
    expect(decideAutoLayout(BREAKPOINT.compactBarPx - 1, 1080).compactBar).toBe(true);
  });
});

// ── 공간 예산 회귀 (UX_AUDIT C-1의 핵심 주장) ───────────────────────

describe('공간 예산', () => {
  /**
   * 구 구현의 뷰포트 면적은 `(W−530)×(H−506)`이라는 상수 차감 함수였다.
   * 이제 세로 크롬은 clamp()라 창 높이에 비례하고, 뷰포트에는 하한이 있다.
   * 여기서는 "최소 크롬 합"이 구 구현보다 실제로 작아졌는지를 고정한다.
   */
  it('최소 세로 크롬 합이 구 구현(374px)보다 작다', () => {
    const now = LAYOUT.barHeightPx + DOCK_COLLAPSED_PX + FLOW_STRIP_PX + 5 * 2;
    expect(now).toBeLessThan(374);
  });

  it('최소 가로 크롬 합이 구 구현(261px)보다 작다', () => {
    const now = LEFT_RAIL_WIDTH_PX + RIGHT_RAIL_WIDTH_PX + 5 * 2;
    expect(now).toBeLessThan(261);
  });
});
