// core/stepper.test.ts — FixedStepper 단위 테스트 (Phase 1 게이트: 프레임레이트 독립성)
//
// 순수 로직만 검증한다 — 물리/렌더/rAF 비의존 (CLAUDE.md §4 부수효과 격리).

import { describe, expect, it } from 'vitest';
import { FixedStepper, MAX_FRAME_DT_SEC } from './engine';

/** 프레임 dt 배열을 stepper에 흘려 누적 스텝 수를 반환 */
function runPattern(stepper: FixedStepper, frameDtsSec: number[]): number {
  let total = 0;
  for (const dtSec of frameDtsSec) {
    total += stepper.advance(dtSec).steps;
  }
  return total;
}

describe('FixedStepper — constructor', () => {
  it('rejects non-positive or non-finite fixedDtSec', () => {
    expect(() => new FixedStepper(0)).toThrow(RangeError);
    expect(() => new FixedStepper(-1)).toThrow(RangeError);
    expect(() => new FixedStepper(NaN)).toThrow(RangeError);
    expect(() => new FixedStepper(Infinity)).toThrow(RangeError);
  });
});

describe('FixedStepper — (a) constant 60fps frames at 240Hz', () => {
  it('yields exactly 4 steps per frame with alpha 0', () => {
    const FIXED_DT_SEC = 1 / 240;
    const FRAME_DT_SEC = 1 / 60; // 정확히 4 × fixedDt (이진 부동소수에서도 정확)
    const FRAMES = 600;

    const stepper = new FixedStepper(FIXED_DT_SEC);
    for (let i = 0; i < FRAMES; i += 1) {
      const { steps, alpha } = stepper.advance(FRAME_DT_SEC);
      expect(steps).toBe(4);
      expect(alpha).toBe(0);
    }
  });
});

describe('FixedStepper — (b) frame-rate independence (Phase 1 gate)', () => {
  // 이진 부동소수에서 정확히 표현되는 값만 사용해 결정론적으로 검증한다:
  // fixedDt = 1/256s, 모든 프레임 dt는 1/1024s(QUANTUM)의 배수 → 모든 산술이 정확.
  const FIXED_DT_SEC = 1 / 256;
  const QUANTUM_SEC = 1 / 1024;
  const QUANTA_PER_STEP = 4; // fixedDt / quantum
  const TOTAL_QUANTA = 3200; // 총 3.125s → 정확히 800 스텝
  const EXPECTED_TOTAL_STEPS = TOTAL_QUANTA / QUANTA_PER_STEP;

  /** 결정론적 LCG로 불규칙 프레임 dt 목록 생성 (합계는 정확히 TOTAL_QUANTA×quantum) */
  function irregularFrames(seed: number): number[] {
    const MAX_FRAME_QUANTA = 90; // 90/1024 ≈ 0.088s < MAX_FRAME_DT_SEC → 클램프 비간섭
    let s = seed >>> 0;
    const frames: number[] = [];
    let leftQuanta = TOTAL_QUANTA;
    while (leftQuanta > 0) {
      s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
      const take = Math.min(1 + (s % MAX_FRAME_QUANTA), leftQuanta);
      frames.push(take * QUANTUM_SEC);
      leftQuanta -= take;
    }
    return frames;
  }

  it('irregular vs regular frame patterns over the same total time give identical cumulative steps', () => {
    const REGULAR_FRAME_QUANTA = 32; // 1/32s per frame
    const regular = Array.from(
      { length: TOTAL_QUANTA / REGULAR_FRAME_QUANTA },
      () => REGULAR_FRAME_QUANTA * QUANTUM_SEC,
    );
    const irregularA = irregularFrames(1);
    const irregularB = irregularFrames(0xdeadbeef);

    const stepsRegular = runPattern(new FixedStepper(FIXED_DT_SEC), regular);
    const stepsIrregularA = runPattern(new FixedStepper(FIXED_DT_SEC), irregularA);
    const stepsIrregularB = runPattern(new FixedStepper(FIXED_DT_SEC), irregularB);

    expect(stepsRegular).toBe(EXPECTED_TOTAL_STEPS);
    expect(stepsIrregularA).toBe(EXPECTED_TOTAL_STEPS);
    expect(stepsIrregularB).toBe(EXPECTED_TOTAL_STEPS);
  });
});

describe('FixedStepper — (c) spiral-of-death clamp', () => {
  it('clamps a 0.5s frame to at most MAX_FRAME_DT_SEC worth of steps', () => {
    const FIXED_DT_SEC = 1 / 240;
    const MAX_STEPS_PER_FRAME = Math.floor(MAX_FRAME_DT_SEC / FIXED_DT_SEC); // 24

    const stepper = new FixedStepper(FIXED_DT_SEC);
    const { steps, alpha } = stepper.advance(0.5);
    expect(steps).toBe(MAX_STEPS_PER_FRAME);
    expect(alpha).toBeGreaterThanOrEqual(0);
    expect(alpha).toBeLessThan(1);
  });

  it('clamps absurdly large frame dt (background tab return) identically', () => {
    const FIXED_DT_SEC = 1 / 240;
    const a = new FixedStepper(FIXED_DT_SEC).advance(1000);
    const b = new FixedStepper(FIXED_DT_SEC).advance(MAX_FRAME_DT_SEC);
    expect(a.steps).toBe(b.steps);
  });

  it('treats negative / NaN / Infinity frame dt as 0', () => {
    const stepper = new FixedStepper(1 / 240);
    for (const bad of [-1, NaN, Infinity, -Infinity]) {
      const { steps, alpha } = stepper.advance(bad);
      expect(steps).toBe(0);
      expect(alpha).toBe(0);
    }
  });
});

describe('FixedStepper — (d) alpha invariant', () => {
  it('alpha is always in [0, 1) across irregular non-dyadic frame dts', () => {
    const FIXED_DT_SEC = 1 / 240;
    const stepper = new FixedStepper(FIXED_DT_SEC);

    // 결정론적 의사난수(비이진 유리수 포함)로 5000프레임 흘리기
    let s = 42 >>> 0;
    for (let i = 0; i < 5000; i += 1) {
      s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
      const dtSec = (s / 0xffffffff) * 0.13; // 0..0.13s — 클램프 경계도 넘나든다
      const { steps, alpha } = stepper.advance(dtSec);
      expect(Number.isInteger(steps)).toBe(true);
      expect(steps).toBeGreaterThanOrEqual(0);
      expect(alpha).toBeGreaterThanOrEqual(0);
      expect(alpha).toBeLessThan(1);
    }
  });
});

describe('FixedStepper — reset', () => {
  it('clears the accumulator', () => {
    const FIXED_DT_SEC = 1 / 240;
    const stepper = new FixedStepper(FIXED_DT_SEC);

    // fixedDt의 절반만 누적 → 스텝 0, alpha ≈ 0.5
    const before = stepper.advance(FIXED_DT_SEC / 2);
    expect(before.steps).toBe(0);
    expect(before.alpha).toBeCloseTo(0.5, 12);

    stepper.reset();
    const after = stepper.advance(0);
    expect(after.steps).toBe(0);
    expect(after.alpha).toBe(0);
  });
});
