// core/ground-clamp.test.ts — 바닥 하한 계산 (순수 · core/ground-clamp.ts 계약)
//
// 이 테스트가 고정하는 것: "무엇이 최저점인가"가 회전·collider 오프셋·형상 종류에
// 따라 달라져도, 편집이 사물을 바닥 아래로 보내지 않는다는 규칙은 같다.
// 물리 없이 계산만 검증한다 — 실제 씬에서의 거동은 브라우저 게이트가 본다.

import { describe, expect, it } from 'vitest';
import {
  clampPositionAboveGround,
  entityDropBelowOrigin,
  GROUND_TOP_Y_M,
  groundedTransformForShape,
  localUpAxis,
  minGroundedY,
  rotatedY,
  shapeDropBelowOrigin,
  snapPositionToGround,
} from './ground-clamp';
import type { ColliderShape, EntitySpec, Quat, Vec3 } from '../schema/types';

// ── 테스트 상수 (매직넘버 금지 — CLAUDE.md §4) ──────────────────────

const IDENTITY: Quat = [0, 0, 0, 1];
const BOX_HALF: Vec3 = [0.1, 0.05, 0.2];
const SPHERE_RADIUS_M = 0.07;
const CAPSULE_RADIUS_M = 0.03;
const CAPSULE_HALF_HEIGHT_M = 0.08;
const CYLINDER_RADIUS_M = 0.04;
const CYLINDER_HALF_HEIGHT_M = 0.09;
/** 부동소수 비교 자릿수 — 씬 스케일(cm)에서 충분히 엄격하다 */
const PRECISION = 9;

/** 축 회전 쿼터니언 (deg) */
function quatAbout(axis: 'x' | 'y' | 'z', deg: number): Quat {
  const half = ((deg * Math.PI) / 180) / 2;
  const s = Math.sin(half);
  const c = Math.cos(half);
  if (axis === 'x') return [s, 0, 0, c];
  if (axis === 'y') return [0, s, 0, c];
  return [0, 0, s, c];
}

function boxEntity(shape: ColliderShape, position: Vec3 = [0, 0, 0], rotation?: Quat): EntitySpec {
  return {
    id: 'e',
    type: 'object',
    transform: rotation ? { position, rotation } : { position },
    visual: { kind: 'primitive', primitive: shape },
    physics: {
      bodyType: 'dynamic',
      colliders: [{ shape, group: 'OBJECT', collidesWith: ['ENV'] }],
    },
  };
}

// ── 1. 회전 수학 ────────────────────────────────────────────────────

describe('localUpAxis / rotatedY', () => {
  it('identity: up 축은 월드 Y 그대로', () => {
    expect(localUpAxis(IDENTITY)).toEqual([0, 1, 0]);
  });

  it('Z축 45° 회전: up 축이 XY 평면에서 45° 기운다', () => {
    const u = localUpAxis(quatAbout('z', 45));
    expect(u[0]).toBeCloseTo(Math.SQRT1_2, PRECISION);
    expect(u[1]).toBeCloseTo(Math.SQRT1_2, PRECISION);
    expect(u[2]).toBeCloseTo(0, PRECISION);
  });

  it('X축 +90° 회전: 로컬 +Z는 월드 −Y로, 로컬 +Y는 월드 +Z로 간다', () => {
    expect(rotatedY(quatAbout('x', 90), [0, 0, 1])).toBeCloseTo(-1, PRECISION);
    expect(rotatedY(quatAbout('x', 90), [0, 1, 0])).toBeCloseTo(0, PRECISION);
  });
});

// ── 2. 형상별 최저점 ────────────────────────────────────────────────

describe('shapeDropBelowOrigin', () => {
  it('box(축정렬): 세로 half extent가 곧 낙차', () => {
    const drop = shapeDropBelowOrigin({ kind: 'box', halfExtents: BOX_HALF });
    expect(drop).toBeCloseTo(BOX_HALF[1], PRECISION);
  });

  it('box(Z 45°): 모서리가 최저점 — (hx + hy)/√2', () => {
    const drop = shapeDropBelowOrigin(
      { kind: 'box', halfExtents: BOX_HALF },
      quatAbout('z', 45),
    );
    expect(drop).toBeCloseTo((BOX_HALF[0] + BOX_HALF[1]) * Math.SQRT1_2, PRECISION);
  });

  it('box(X 90°): 깊이 half extent가 세로가 된다', () => {
    const drop = shapeDropBelowOrigin(
      { kind: 'box', halfExtents: BOX_HALF },
      quatAbout('x', 90),
    );
    expect(drop).toBeCloseTo(BOX_HALF[2], PRECISION);
  });

  it('sphere: 회전과 무관하게 반지름', () => {
    const shape: ColliderShape = { kind: 'sphere', radius: SPHERE_RADIUS_M };
    expect(shapeDropBelowOrigin(shape)).toBeCloseTo(SPHERE_RADIUS_M, PRECISION);
    expect(shapeDropBelowOrigin(shape, quatAbout('z', 37))).toBeCloseTo(
      SPHERE_RADIUS_M,
      PRECISION,
    );
  });

  it('capsule: 축정렬은 halfHeight + r, 눕히면 r만', () => {
    const shape: ColliderShape = {
      kind: 'capsule',
      halfHeight: CAPSULE_HALF_HEIGHT_M,
      radius: CAPSULE_RADIUS_M,
    };
    expect(shapeDropBelowOrigin(shape)).toBeCloseTo(
      CAPSULE_HALF_HEIGHT_M + CAPSULE_RADIUS_M,
      PRECISION,
    );
    expect(shapeDropBelowOrigin(shape, quatAbout('z', 90))).toBeCloseTo(
      CAPSULE_RADIUS_M,
      PRECISION,
    );
  });

  it('cylinder: 눕히면 원판 반지름이 세로가 된다', () => {
    const shape: ColliderShape = {
      kind: 'cylinder',
      halfHeight: CYLINDER_HALF_HEIGHT_M,
      radius: CYLINDER_RADIUS_M,
    };
    expect(shapeDropBelowOrigin(shape)).toBeCloseTo(CYLINDER_HALF_HEIGHT_M, PRECISION);
    expect(shapeDropBelowOrigin(shape, quatAbout('x', 90))).toBeCloseTo(
      CYLINDER_RADIUS_M,
      PRECISION,
    );
  });

  it('측정 불가 형상(convexHull/trimesh/fromVisual)은 null — 호출자가 원점 폴백', () => {
    expect(shapeDropBelowOrigin({ kind: 'convexHull', ref: 'asset://1' })).toBeNull();
    expect(shapeDropBelowOrigin({ kind: 'trimesh', ref: 'asset://1' })).toBeNull();
    expect(shapeDropBelowOrigin({ kind: 'fromVisual' })).toBeNull();
  });
});

// ── 3. 엔티티 단위 ──────────────────────────────────────────────────

describe('entityDropBelowOrigin', () => {
  it('collider가 여럿이면 가장 깊은 것을 쓴다', () => {
    const entity: EntitySpec = {
      id: 'multi',
      type: 'static',
      transform: { position: [0, 0, 0] },
      visual: { kind: 'primitive', primitive: { kind: 'box', halfExtents: [0.1, 0.02, 0.1] } },
      physics: {
        bodyType: 'fixed',
        colliders: [
          { shape: { kind: 'box', halfExtents: [0.1, 0.02, 0.1] }, group: 'ENV', collidesWith: [] },
          { shape: { kind: 'sphere', radius: 0.2 }, group: 'ENV', collidesWith: [] },
        ],
      },
    };
    expect(entityDropBelowOrigin(entity)).toBeCloseTo(0.2, PRECISION);
  });

  it('collider offset이 위로 향하면 최저점도 그만큼 올라간다', () => {
    const entity: EntitySpec = {
      id: 'offset',
      type: 'static',
      transform: { position: [0, 0, 0] },
      visual: { kind: 'primitive', primitive: { kind: 'box', halfExtents: [0.1, 0.05, 0.1] } },
      physics: {
        bodyType: 'fixed',
        colliders: [
          {
            shape: { kind: 'box', halfExtents: [0.1, 0.05, 0.1] },
            offset: { position: [0, 0.3, 0] },
            group: 'ENV',
            collidesWith: [],
          },
        ],
      },
    };
    // 최저점 = 0.3 − 0.05 = 0.25 만큼 원점 **위** → 아래 방향 낙차는 음수
    expect(entityDropBelowOrigin(entity)).toBeCloseTo(-0.25, PRECISION);
  });

  it('physics가 없는 장식 엔티티는 visual.primitive를 잰다', () => {
    const entity: EntitySpec = {
      id: 'decor',
      type: 'static',
      transform: { position: [0, 0, 0] },
      visual: { kind: 'primitive', primitive: { kind: 'sphere', radius: SPHERE_RADIUS_M } },
    };
    expect(entityDropBelowOrigin(entity)).toBeCloseTo(SPHERE_RADIUS_M, PRECISION);
  });

  it('임포트 메시(convexHull)는 원점이 곧 하한 — 피벗이 bbox 바닥 중심이라 정확하다', () => {
    const entity: EntitySpec = {
      id: 'imported',
      type: 'object',
      transform: { position: [0, 0, 0] },
      visual: { kind: 'mesh', ref: 'asset://1' },
      physics: {
        bodyType: 'dynamic',
        colliders: [
          { shape: { kind: 'convexHull', ref: 'asset://1' }, group: 'OBJECT', collidesWith: [] },
        ],
      },
    };
    expect(entityDropBelowOrigin(entity)).toBe(0);
    expect(minGroundedY(entity)).toBe(GROUND_TOP_Y_M);
  });
});

// ── 4. 클램프 ───────────────────────────────────────────────────────

describe('clampPositionAboveGround', () => {
  const entity = boxEntity({ kind: 'box', halfExtents: BOX_HALF });

  it('지하로 내려간 위치를 최저점이 바닥에 닿는 y로 끌어올린다 (x/z는 보존)', () => {
    const result = clampPositionAboveGround(entity, [0.3, -5, -0.2]);
    expect(result.clamped).toBe(true);
    expect(result.position[0]).toBe(0.3);
    expect(result.position[2]).toBe(-0.2);
    expect(result.position[1]).toBeCloseTo(BOX_HALF[1], PRECISION);
  });

  it('바닥 슬래브 안(0 초과 아님)도 지하로 본다 — 반쯤 잠긴 배치를 허용하지 않는다', () => {
    const result = clampPositionAboveGround(entity, [0, 0, 0]);
    expect(result.clamped).toBe(true);
    expect(result.position[1]).toBeCloseTo(BOX_HALF[1], PRECISION);
  });

  it('바닥 위 위치는 손대지 않는다', () => {
    const result = clampPositionAboveGround(entity, [0, 1.5, 0]);
    expect(result.clamped).toBe(false);
    expect(result.position).toEqual([0, 1.5, 0]);
  });

  it('정확히 하한에 놓인 값은 오차 안에서 클램프로 치지 않는다 (토스트 반복 방지)', () => {
    const result = clampPositionAboveGround(entity, [0, BOX_HALF[1], 0]);
    expect(result.clamped).toBe(false);
  });

  it('이번 편집의 회전으로 하한을 잰다 — 회전 커밋이 모서리를 바닥에 박지 않는다', () => {
    const rotated = clampPositionAboveGround(entity, [0, 0.06, 0], quatAbout('z', 45));
    expect(rotated.clamped).toBe(true);
    expect(rotated.position[1]).toBeCloseTo(
      (BOX_HALF[0] + BOX_HALF[1]) * Math.SQRT1_2,
      PRECISION,
    );
  });

  it('로봇(형상 미측정)은 베이스 원점이 하한이다', () => {
    const robot: EntitySpec = {
      id: 'arm',
      type: 'robot',
      transform: { position: [0, -0.4, 0] },
      visual: { kind: 'urdf', ref: 'assets/robots/arm6/arm6.urdf' },
    };
    const result = clampPositionAboveGround(robot, [0, -0.4, 0]);
    expect(result.clamped).toBe(true);
    expect(result.position[1]).toBe(GROUND_TOP_Y_M);
  });
});

// ── 5. 치수 편집 후속 / 바닥에 붙이기 ───────────────────────────────

describe('groundedTransformForShape', () => {
  it('바닥에 놓인 박스를 키우면 중심을 올려 바닥에 계속 놓이게 한다', () => {
    const entity = boxEntity({ kind: 'box', halfExtents: [0.05, 0.05, 0.05] }, [0.2, 0.05, 0.1]);
    const next = groundedTransformForShape(entity, { kind: 'box', halfExtents: [0.05, 0.2, 0.05] });
    expect(next).not.toBeNull();
    expect(next?.position[1]).toBeCloseTo(0.2, PRECISION);
    expect(next?.position[0]).toBe(0.2);
    expect(next?.position[2]).toBe(0.1);
  });

  it('공중에 뜬 사물은 치수를 줄여도 그대로 둔다 (null = 후속 편집 없음)', () => {
    const entity = boxEntity({ kind: 'box', halfExtents: [0.05, 0.05, 0.05] }, [0, 1, 0]);
    expect(
      groundedTransformForShape(entity, { kind: 'box', halfExtents: [0.02, 0.02, 0.02] }),
    ).toBeNull();
  });

  it('측정 불가 형상은 후속 편집을 만들지 않는다', () => {
    const entity = boxEntity({ kind: 'box', halfExtents: [0.05, 0.05, 0.05] }, [0, 0.05, 0]);
    expect(groundedTransformForShape(entity, { kind: 'fromVisual' })).toBeNull();
  });
});

describe('snapPositionToGround', () => {
  it('떠 있는 사물을 바닥에 내린다 (클램프와 달리 내리기도 한다)', () => {
    const entity = boxEntity({ kind: 'box', halfExtents: BOX_HALF }, [0.4, 2.5, -0.3]);
    expect(snapPositionToGround(entity)).toEqual([0.4, BOX_HALF[1], -0.3]);
  });

  it('지하에 박힌 사물의 구제 경로가 된다', () => {
    const entity = boxEntity({ kind: 'box', halfExtents: BOX_HALF }, [0, -3, 0]);
    expect(snapPositionToGround(entity)[1]).toBeCloseTo(BOX_HALF[1], PRECISION);
  });
});
