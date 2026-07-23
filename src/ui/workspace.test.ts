// ui/workspace.test.ts — 워크스페이스 셸 순수 로직 단위 테스트 (DOM 비의존, node)
//
// mountWorkspace의 DOM 조립(grid/슬롯/스플리터 배선)은 얇은 글루라 여기서 다루지
// 않는다(브라우저 게이트 몫). 여기서는 스플리터 크기 계산(clamp/드래그 방향 부호)과
// 크기 상수 계약(UX_DESIGN §2 · Phase 7 요구: flowGraph 최소 200px, 레일 36px)을
// 검증한다.

import { describe, expect, it } from 'vitest';
import {
  LEFT_RAIL_WIDTH_PX,
  WORKSPACE_SIZES,
  clampPanelSize,
  dragPanelSize,
} from './workspace';

// ── 크기 상수 계약 ──────────────────────────────────────────────────

describe('WORKSPACE_SIZES', () => {
  it('모든 패널: min ≤ default ≤ max', () => {
    for (const [name, limits] of Object.entries(WORKSPACE_SIZES)) {
      expect(limits.minPx, name).toBeLessThanOrEqual(limits.defaultPx);
      expect(limits.defaultPx, name).toBeLessThanOrEqual(limits.maxPx);
    }
  });

  it('flowGraph 최소 높이는 200px (Phase 7 요구)', () => {
    expect(WORKSPACE_SIZES.flowGraph.minPx).toBe(200);
  });

  it('좌 패널 기본 ~240px · 접힘 레일 36px (UX §2)', () => {
    expect(WORKSPACE_SIZES.left.defaultPx).toBe(240);
    expect(LEFT_RAIL_WIDTH_PX).toBe(36);
    // 레일은 최소 폭보다 좁아야 "접힘"이 의미가 있다
    expect(LEFT_RAIL_WIDTH_PX).toBeLessThan(WORKSPACE_SIZES.left.minPx);
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
