// render/interaction.test.ts — 뷰포트 상호작용 순수 헬퍼 단위 테스트
//
// ViewportInteraction 클래스 자체(three 씬·DOM·TransformControls)는 node 단위 테스트
// 대상이 아니다(vitest.config.ts 원칙 — 렌더 의존은 브라우저 게이트에서 검증).
// 여기서는 스냅 반올림·NDC 변환·바닥 광선 교차·클릭 판정 등 순수 수학/판정 헬퍼만
// 검증한다.

import { describe, expect, it } from 'vitest';
import type { Vec3 } from '../schema/types';
import {
  clientToNdc,
  COMMIT_MIN_DELTA,
  isTypingTarget,
  keyToGizmoMode,
  keyToNudgeAxis,
  NUDGE_FINE_STEP_M,
  NUDGE_STEP_M,
  rayGroundPoint,
  rayGroundT,
  ROTATION_SNAP_DEG,
  ROTATION_SNAP_RAD,
  snapToStep,
  transformsAlmostEqual,
  TRANSLATION_SNAP_M,
  withinClickThreshold,
} from './interaction';

const DIGITS = 10;

describe('clientToNdc', () => {
  const rect = { left: 0, top: 0, width: 200, height: 100 };

  it('maps the rect center to (0, 0)', () => {
    const ndc = clientToNdc(100, 50, rect);
    expect(ndc).not.toBeNull();
    expect(ndc![0]).toBeCloseTo(0, DIGITS);
    expect(ndc![1]).toBeCloseTo(0, DIGITS);
  });

  it('maps the top-left corner to (-1, +1) and bottom-right to (+1, -1)', () => {
    // NDC의 y는 화면 위가 +1 — 클라이언트 y축(아래로 증가)을 반전해야 한다
    expect(clientToNdc(0, 0, rect)).toEqual([-1, 1]);
    expect(clientToNdc(200, 100, rect)).toEqual([1, -1]);
  });

  it('respects a rect offset from the page origin (canvas not at 0,0)', () => {
    const offset = { left: 50, top: 20, width: 200, height: 100 };
    const ndc = clientToNdc(150, 70, offset); // offset 기준 중앙
    expect(ndc![0]).toBeCloseTo(0, DIGITS);
    expect(ndc![1]).toBeCloseTo(0, DIGITS);
  });

  it('returns values outside [-1, 1] for points outside the rect (no clamping)', () => {
    const ndc = clientToNdc(300, 50, rect);
    expect(ndc![0]).toBeCloseTo(2, DIGITS);
  });

  it('returns null for degenerate rects (zero width or height)', () => {
    expect(clientToNdc(10, 10, { left: 0, top: 0, width: 0, height: 100 })).toBeNull();
    expect(clientToNdc(10, 10, { left: 0, top: 0, width: 200, height: 0 })).toBeNull();
  });
});

describe('snapToStep', () => {
  it('rounds to the nearest multiple of the step', () => {
    expect(snapToStep(0.07, 0.05)).toBeCloseTo(0.05, DIGITS);
    expect(snapToStep(0.08, 0.05)).toBeCloseTo(0.1, DIGITS);
    expect(snapToStep(0.13, 0.05)).toBeCloseTo(0.15, DIGITS);
  });

  it('keeps exact multiples unchanged (within float tolerance)', () => {
    expect(snapToStep(0.15, 0.05)).toBeCloseTo(0.15, DIGITS);
    expect(snapToStep(0, 0.05)).toBe(0);
  });

  it('snaps negative values toward the nearest multiple', () => {
    expect(snapToStep(-0.07, 0.05)).toBeCloseTo(-0.05, DIGITS);
    expect(snapToStep(-0.13, 0.05)).toBeCloseTo(-0.15, DIGITS);
  });

  it('passes the value through when the step is zero or negative (snap disabled)', () => {
    expect(snapToStep(0.123, 0)).toBe(0.123);
    expect(snapToStep(0.123, -0.05)).toBe(0.123);
  });

  it('passes non-finite values through unchanged', () => {
    expect(snapToStep(Number.POSITIVE_INFINITY, 0.05)).toBe(Number.POSITIVE_INFINITY);
    expect(Number.isNaN(snapToStep(Number.NaN, 0.05))).toBe(true);
  });

  it('works with the project snap constants', () => {
    expect(snapToStep(0.51, TRANSLATION_SNAP_M)).toBeCloseTo(0.5, DIGITS);
    expect(snapToStep(0.2, ROTATION_SNAP_RAD)).toBeCloseTo(ROTATION_SNAP_RAD, DIGITS); // 0.2rad ≈ 11.5° → 15°
  });
});

describe('snap constants', () => {
  it('rotation snap is exactly 15 degrees in radians', () => {
    expect(ROTATION_SNAP_DEG).toBe(15);
    expect(ROTATION_SNAP_RAD).toBeCloseTo((15 * Math.PI) / 180, DIGITS);
  });

  it('translation snap is 0.05 m', () => {
    expect(TRANSLATION_SNAP_M).toBe(0.05);
  });
});

describe('rayGroundT', () => {
  it('returns the distance for a downward ray above the plane', () => {
    expect(rayGroundT(2, -1)).toBeCloseTo(2, DIGITS);
    expect(rayGroundT(1, -0.5)).toBeCloseTo(2, DIGITS);
  });

  it('returns the distance for an upward ray below the plane', () => {
    expect(rayGroundT(-1, 1)).toBeCloseTo(1, DIGITS);
  });

  it('returns 0 when the origin already lies on the plane', () => {
    expect(rayGroundT(0, -1)).toBe(0);
  });

  it('returns null when the ray points away from the plane', () => {
    expect(rayGroundT(2, 1)).toBeNull(); // 위에서 위로
    expect(rayGroundT(-2, -1)).toBeNull(); // 아래에서 아래로
  });

  it('returns null when the ray is (near) parallel to the plane', () => {
    expect(rayGroundT(2, 0)).toBeNull();
    expect(rayGroundT(2, 1e-12)).toBeNull();
  });
});

describe('rayGroundPoint', () => {
  it('intersects a vertical ray directly below the origin', () => {
    const origin: Vec3 = [1, 2, 3];
    const dir: Vec3 = [0, -1, 0];
    expect(rayGroundPoint(origin, dir)).toEqual([1, 0, 3]);
  });

  it('intersects a diagonal ray at the expected x/z offset', () => {
    const inv = 1 / Math.sqrt(2);
    const point = rayGroundPoint([0, 1, 0], [inv, -inv, 0]);
    expect(point).not.toBeNull();
    expect(point![0]).toBeCloseTo(1, DIGITS);
    expect(point![1]).toBe(0); // y는 정확히 0으로 고정
    expect(point![2]).toBeCloseTo(0, DIGITS);
  });

  it('returns null for rays that never reach the plane', () => {
    expect(rayGroundPoint([0, 1, 0], [1, 0, 0])).toBeNull(); // 평행
    expect(rayGroundPoint([0, 1, 0], [0, 1, 0])).toBeNull(); // 반대 방향
  });
});

describe('keyToGizmoMode', () => {
  it('maps W/E/R to translate/rotate/scale (case-insensitive)', () => {
    expect(keyToGizmoMode('w')).toBe('translate');
    expect(keyToGizmoMode('W')).toBe('translate');
    expect(keyToGizmoMode('e')).toBe('rotate');
    expect(keyToGizmoMode('E')).toBe('rotate');
    expect(keyToGizmoMode('r')).toBe('scale');
    expect(keyToGizmoMode('R')).toBe('scale');
  });

  it('returns null for any other key', () => {
    expect(keyToGizmoMode('q')).toBeNull();
    expect(keyToGizmoMode('Escape')).toBeNull();
    expect(keyToGizmoMode(' ')).toBeNull();
    expect(keyToGizmoMode('')).toBeNull();
  });
});

describe('isTypingTarget', () => {
  it('treats form fields as typing targets (shortcut suppression)', () => {
    expect(isTypingTarget({ tagName: 'INPUT' })).toBe(true);
    expect(isTypingTarget({ tagName: 'input' })).toBe(true); // 대소문자 무관
    expect(isTypingTarget({ tagName: 'TEXTAREA' })).toBe(true);
    expect(isTypingTarget({ tagName: 'SELECT' })).toBe(true);
  });

  it('treats contentEditable elements as typing targets', () => {
    expect(isTypingTarget({ tagName: 'DIV', isContentEditable: true })).toBe(true);
  });

  it('does not suppress for canvas/body/null targets', () => {
    expect(isTypingTarget({ tagName: 'CANVAS' })).toBe(false);
    expect(isTypingTarget({ tagName: 'BODY', isContentEditable: false })).toBe(false);
    expect(isTypingTarget({})).toBe(false);
    expect(isTypingTarget(null)).toBe(false);
  });
});

describe('withinClickThreshold', () => {
  it('accepts movements at or below the threshold (Euclidean distance)', () => {
    expect(withinClickThreshold(10, 10, 10, 10, 5)).toBe(true); // 제자리
    expect(withinClickThreshold(10, 10, 13, 14, 5)).toBe(true); // 거리 5 (3-4-5)
  });

  it('rejects movements beyond the threshold (orbit drag)', () => {
    expect(withinClickThreshold(10, 10, 16, 10, 5)).toBe(false);
    expect(withinClickThreshold(10, 10, 14, 14, 5)).toBe(false); // 거리 ≈ 5.66
  });

  it('is symmetric in direction', () => {
    expect(withinClickThreshold(100, 100, 97, 96, 5)).toBe(true);
    expect(withinClickThreshold(100, 100, 94, 100, 5)).toBe(false);
  });
});

describe('transformsAlmostEqual', () => {
  const identity = (): {
    position: [number, number, number];
    rotation: [number, number, number, number];
    scale: [number, number, number];
  } => ({ position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] });

  it('동일 트랜스폼과 오차 내 미세 변화는 같다고 판정한다 (no-op 드래그 스킵)', () => {
    expect(transformsAlmostEqual(identity(), identity())).toBe(true);
    const jitter = identity();
    jitter.position = [COMMIT_MIN_DELTA / 2, 0, -COMMIT_MIN_DELTA / 2];
    expect(transformsAlmostEqual(identity(), jitter)).toBe(true);
  });

  it('오차를 넘는 이동/회전/스케일 변화는 다르다고 판정한다', () => {
    const moved = identity();
    moved.position = [0.01, 0, 0];
    expect(transformsAlmostEqual(identity(), moved)).toBe(false);

    const rotated = identity();
    rotated.rotation = [0.1, 0, 0, Math.sqrt(1 - 0.01)];
    expect(transformsAlmostEqual(identity(), rotated)).toBe(false);

    const scaled = identity();
    scaled.scale = [1.1, 1, 1];
    expect(transformsAlmostEqual(identity(), scaled)).toBe(false);
  });

  it('q와 -q는 같은 회전으로 본다 (쿼터니언 이중 피복)', () => {
    const a = identity();
    a.rotation = [0.5, 0.5, 0.5, 0.5];
    const b = identity();
    b.rotation = [-0.5, -0.5, -0.5, -0.5];
    expect(transformsAlmostEqual(a, b)).toBe(true);
  });
});

describe('keyToNudgeAxis — 방향키 오브젝트 이동 (UX §3.3)', () => {
  it('방향키는 카메라 기준 수평 축으로 매핑된다', () => {
    expect(keyToNudgeAxis('ArrowRight')).toEqual({ kind: 'right', sign: 1 });
    expect(keyToNudgeAxis('ArrowLeft')).toEqual({ kind: 'right', sign: -1 });
    expect(keyToNudgeAxis('ArrowUp')).toEqual({ kind: 'forward', sign: 1 });
    expect(keyToNudgeAxis('ArrowDown')).toEqual({ kind: 'forward', sign: -1 });
  });

  it('PageUp/PageDown은 월드 수직 축이다', () => {
    expect(keyToNudgeAxis('PageUp')).toEqual({ kind: 'vertical', sign: 1 });
    expect(keyToNudgeAxis('PageDown')).toEqual({ kind: 'vertical', sign: -1 });
  });

  it('이동 키가 아니면 null (기즈모 단축키·타이핑과 충돌하지 않는다)', () => {
    for (const key of ['w', 'e', 'r', 'a', 'Enter', ' ', 'Escape', 'Delete']) {
      expect(keyToNudgeAxis(key)).toBeNull();
    }
  });

  it('미세 이동 폭은 기본 이동보다 작고, 기본 이동은 스냅 격자와 같다', () => {
    expect(NUDGE_FINE_STEP_M).toBeLessThan(NUDGE_STEP_M);
    expect(NUDGE_STEP_M).toBe(TRANSLATION_SNAP_M);
  });
});
