// core/math.test.ts — 순수 수학 헬퍼 단위 테스트 (물리/렌더 비의존)

import { describe, expect, it } from 'vitest';
import { clamp01, ease, lerp } from './math';

describe('clamp01', () => {
  it('clamps below 0 to 0', () => {
    expect(clamp01(-0.5)).toBe(0);
    expect(clamp01(-Infinity)).toBe(0);
  });

  it('clamps above 1 to 1', () => {
    expect(clamp01(1.5)).toBe(1);
    expect(clamp01(Infinity)).toBe(1);
  });

  it('passes through values in [0, 1]', () => {
    expect(clamp01(0)).toBe(0);
    expect(clamp01(0.25)).toBe(0.25);
    expect(clamp01(1)).toBe(1);
  });
});

describe('lerp', () => {
  it('returns endpoints at t=0 and t=1', () => {
    expect(lerp(-2, 6, 0)).toBe(-2);
    expect(lerp(-2, 6, 1)).toBe(6);
  });

  it('interpolates linearly', () => {
    expect(lerp(0, 10, 0.5)).toBe(5);
    expect(lerp(1, 2, 0.25)).toBeCloseTo(1.25, 12);
  });

  it('extrapolates when t is outside [0,1] (unclamped by contract)', () => {
    expect(lerp(0, 10, 2)).toBe(20);
    expect(lerp(0, 10, -1)).toBe(-10);
  });
});

describe('ease — boundary values', () => {
  it('linear: e(0)=0, e(1)=1', () => {
    expect(ease('linear', 0)).toBe(0);
    expect(ease('linear', 1)).toBe(1);
  });

  it('easeInOut: e(0)=0, e(1)=1', () => {
    expect(ease('easeInOut', 0)).toBeCloseTo(0, 12);
    expect(ease('easeInOut', 1)).toBeCloseTo(1, 12);
  });

  it('step: e(0)=0, e(1)=1 (t>=1에서만 1로 점프)', () => {
    expect(ease('step', 0)).toBe(0);
    expect(ease('step', 0.999999)).toBe(0);
    expect(ease('step', 1)).toBe(1);
  });

  it('clamps t outside [0,1]', () => {
    expect(ease('linear', -0.5)).toBe(0);
    expect(ease('linear', 1.5)).toBe(1);
    expect(ease('easeInOut', -1)).toBeCloseTo(0, 12);
    expect(ease('easeInOut', 2)).toBeCloseTo(1, 12);
    expect(ease('step', 1.5)).toBe(1);
    expect(ease('step', -1)).toBe(0);
  });
});

describe('ease — easeInOut shape', () => {
  const SAMPLES = 1000;

  it('is monotonically non-decreasing on [0,1]', () => {
    let prev = ease('easeInOut', 0);
    for (let i = 1; i <= SAMPLES; i += 1) {
      const cur = ease('easeInOut', i / SAMPLES);
      expect(cur).toBeGreaterThanOrEqual(prev);
      prev = cur;
    }
  });

  it('passes midpoint at 0.5 and is symmetric: e(t)+e(1-t)=1', () => {
    expect(ease('easeInOut', 0.5)).toBeCloseTo(0.5, 12);
    for (let i = 0; i <= 100; i += 1) {
      const t = i / 100;
      expect(ease('easeInOut', t) + ease('easeInOut', 1 - t)).toBeCloseTo(1, 12);
    }
  });

  it('stays within [0,1]', () => {
    for (let i = 0; i <= SAMPLES; i += 1) {
      const v = ease('easeInOut', i / SAMPLES);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });
});
