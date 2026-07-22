// render/sync-math.test.ts — pose 보간 순수 수학 단위 테스트 (three.js/Rapier 비의존)

import { describe, expect, it } from 'vitest';
import type { Quat, Vec3 } from '../schema/types';
import {
  clamp01,
  copyQuatInto,
  copyVec3Into,
  lerpVec3Into,
  slerpQuatInto,
} from './sync-math';

const DIGITS = 10;

function quatAboutY(angleRad: number): Quat {
  return [0, Math.sin(angleRad / 2), 0, Math.cos(angleRad / 2)];
}

function quatLength(q: Readonly<Quat>): number {
  return Math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
}

function expectQuatClose(actual: Readonly<Quat>, expected: Readonly<Quat>): void {
  expect(actual[0]).toBeCloseTo(expected[0], DIGITS);
  expect(actual[1]).toBeCloseTo(expected[1], DIGITS);
  expect(actual[2]).toBeCloseTo(expected[2], DIGITS);
  expect(actual[3]).toBeCloseTo(expected[3], DIGITS);
}

describe('clamp01', () => {
  it('passes through values inside [0, 1]', () => {
    expect(clamp01(0)).toBe(0);
    expect(clamp01(0.25)).toBe(0.25);
    expect(clamp01(1)).toBe(1);
  });

  it('clamps values outside [0, 1]', () => {
    expect(clamp01(-0.5)).toBe(0);
    expect(clamp01(1.5)).toBe(1);
  });
});

describe('copyVec3Into / copyQuatInto', () => {
  it('copies components without replacing the out reference', () => {
    const outV: Vec3 = [0, 0, 0];
    copyVec3Into(outV, [1, 2, 3]);
    expect(outV).toEqual([1, 2, 3]);

    const outQ: Quat = [0, 0, 0, 1];
    copyQuatInto(outQ, [0.1, 0.2, 0.3, 0.9]);
    expect(outQ).toEqual([0.1, 0.2, 0.3, 0.9]);
  });
});

describe('lerpVec3Into', () => {
  const a: Vec3 = [0, 1, -2];
  const b: Vec3 = [4, -1, 2];

  it('returns a at t=0 and b at t=1', () => {
    const out: Vec3 = [0, 0, 0];
    lerpVec3Into(out, a, b, 0);
    expect(out).toEqual(a);
    lerpVec3Into(out, a, b, 1);
    expect(out).toEqual(b);
  });

  it('returns the midpoint at t=0.5', () => {
    const out: Vec3 = [0, 0, 0];
    lerpVec3Into(out, a, b, 0.5);
    expect(out).toEqual([2, 0, 0]);
  });

  it('is aliasing-safe when out === a', () => {
    const out: Vec3 = [0, 1, -2];
    lerpVec3Into(out, out, b, 0.5);
    expect(out).toEqual([2, 0, 0]);
  });
});

describe('slerpQuatInto', () => {
  const identity: Quat = [0, 0, 0, 1];
  const yaw90 = quatAboutY(Math.PI / 2);

  it('returns a at t=0 and b at t=1', () => {
    const out: Quat = [0, 0, 0, 0];
    slerpQuatInto(out, identity, yaw90, 0);
    expectQuatClose(out, identity);
    slerpQuatInto(out, identity, yaw90, 1);
    expectQuatClose(out, yaw90);
  });

  it('halves the rotation angle at t=0.5 (identity → 90° about Y gives 45°)', () => {
    const out: Quat = [0, 0, 0, 0];
    slerpQuatInto(out, identity, yaw90, 0.5);
    expectQuatClose(out, quatAboutY(Math.PI / 4));
  });

  it('takes the shortest path when b is the antipodal representation (-q)', () => {
    // -q는 q와 같은 회전. dot < 0이면 내부에서 b를 반전해 짧은 호로 보간해야 한다.
    const yaw90Neg: Quat = [-yaw90[0], -yaw90[1], -yaw90[2], -yaw90[3]];
    const out: Quat = [0, 0, 0, 0];
    slerpQuatInto(out, identity, yaw90Neg, 0.5);
    expectQuatClose(out, quatAboutY(Math.PI / 4));
  });

  it('stays unit-length across t for a large-angle slerp', () => {
    const yaw170 = quatAboutY((170 * Math.PI) / 180);
    const out: Quat = [0, 0, 0, 0];
    for (const t of [0, 0.1, 0.35, 0.5, 0.75, 0.9, 1]) {
      slerpQuatInto(out, identity, yaw170, t);
      expect(quatLength(out)).toBeCloseTo(1, DIGITS);
    }
  });

  it('falls back to normalized lerp for nearly identical rotations', () => {
    const tinyAngleRad = 1e-7;
    const nearIdentity = quatAboutY(tinyAngleRad);
    const out: Quat = [0, 0, 0, 0];
    slerpQuatInto(out, identity, nearIdentity, 0.5);
    expect(quatLength(out)).toBeCloseTo(1, DIGITS);
    // 결과는 두 입력 사이의 중간 미세 회전이어야 한다
    expectQuatClose(out, quatAboutY(tinyAngleRad / 2));
  });

  it('returns a (renormalized) for slerp of a rotation with itself', () => {
    const out: Quat = [0, 0, 0, 0];
    slerpQuatInto(out, yaw90, yaw90, 0.7);
    expectQuatClose(out, yaw90);
  });

  it('is aliasing-safe when out === a', () => {
    const out: Quat = [...identity];
    slerpQuatInto(out, out, yaw90, 0.5);
    expectQuatClose(out, quatAboutY(Math.PI / 4));
  });
});
