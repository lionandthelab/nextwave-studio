// core/conveyor.test.ts — 컨베이어 표면 구동 + 재순환
//
// 두 층으로 나눠 검증한다:
//   1. 순수 기하 (물리 없음) — 방향 회전·벨트 길이 투영·재순환 판정/목적지
//   2. 실 Rapier 통합 — 벨트 위 박스가 **실제로 실려 가고**, 끝에서 되돌아오고,
//      로봇이 들어 올린 사물은 되돌아오지 않는다
//
// 물리 API 사용 전 initPhysics()를 await한다 (CLAUDE.md §2.7).

import { beforeAll, describe, expect, it } from 'vitest';
import {
  beltForward,
  beltLateral,
  ConveyorBinding,
  ConveyorRegistry,
  drivenVelocity,
  extentAlong,
  RECYCLE_GRACE_TICKS,
  RECYCLE_HEIGHT_ABOVE_BELT_M,
  RECYCLE_INSET_M,
  RECYCLE_OVERSHOOT_M,
  recycleTarget,
  rotateVec3,
  shouldRecycle,
} from './conveyor';
import type { BeltGeometry } from './conveyor';
import { initPhysics, RapierWorld } from './world';
import type { BodyId } from './types';
import type { ColliderSpec, ConveyorSpec, Quat, Vec3 } from '../schema/types';

// ── 테스트 상수 (매직넘버 금지 — CLAUDE.md §4) ──────────────────────

const TIMESTEP_HZ = 240;
const GRAVITY: Vec3 = [0, -9.81, 0];
const IDENTITY: Quat = [0, 0, 0, 1];
const PRECISION = 9;

/** 벨트: 진행축(+X) 1.0m × 두께 0.04m × 폭 0.3m */
const BELT_HALF: Vec3 = [0.5, 0.02, 0.15];
const BELT_CENTER: Vec3 = [0, 0.02, 0];
const BELT_TOP_Y = BELT_CENTER[1] + BELT_HALF[1];
const BELT_SPEED_MPS = 0.25;
const BELT_SPEC: ConveyorSpec = { direction: [1, 0, 0], speedMps: BELT_SPEED_MPS };

const ITEM_HALF_M = 0.03;
/** 벨트가 사물을 확실히 실어 나르는 데 필요한 시뮬 시간 */
const CARRY_SEC = 1.0;
/** 정착/이송 판정 허용 오차 */
const TOLERANCE_M = 0.02;
/** 이송 변위 하한 비율 — 안착 슬립을 흡수하되 "거의 안 움직임"은 잡는다 */
const CARRY_MIN_RATIO = 0.8;
/** "구동되지 않았다" 판정 상한 (m/s) — 굴러가는 잔여 속도는 허용 */
const NOT_DRIVEN_MAX_SPEED_MPS = 0.02;
/**
 * 관측 속도 / 선언 속도의 하한. 속도 지정은 스텝 **직전**이고 그 뒤 접촉 솔버가 정지한
 * 벨트 면과의 마찰로 사물을 되잡으므로, 관측값은 선언값의 85~90%가 된다
 * (실측 0.25 → 0.221 = 88.5%). 선언값은 상한이다 — conveyor.ts "알려진 특성" 참조.
 */
const DRIVEN_SPEED_MIN_RATIO = 0.75;

function quatAboutY(deg: number): Quat {
  const half = ((deg * Math.PI) / 180) / 2;
  return [0, Math.sin(half), 0, Math.cos(half)];
}

function geometryOf(forward: Vec3): BeltGeometry {
  const lateral = beltLateral(forward);
  return {
    forward,
    lateral,
    halfLength: extentAlong(BELT_HALF, forward, IDENTITY),
    halfWidth: extentAlong(BELT_HALF, lateral, IDENTITY),
    topY: BELT_TOP_Y,
  };
}

// ── 1. 순수 기하 ────────────────────────────────────────────────────

describe('rotateVec3', () => {
  it('identity는 벡터를 그대로 둔다', () => {
    expect(rotateVec3(IDENTITY, [1, 2, 3])).toEqual([1, 2, 3]);
  });

  it('Y축 90° 회전: +X가 −Z로 간다', () => {
    const r = rotateVec3(quatAboutY(90), [1, 0, 0]);
    expect(r[0]).toBeCloseTo(0, PRECISION);
    expect(r[2]).toBeCloseTo(-1, PRECISION);
  });
});

describe('beltForward', () => {
  it('회전이 없으면 로컬 방향이 곧 월드 방향 (정규화됨)', () => {
    const f = beltForward({ ...BELT_SPEC, direction: [3, 0, 0] }, IDENTITY);
    expect(f).toEqual([1, 0, 0]);
  });

  it('엔티티 회전이 흐름 방향을 함께 돌린다 (벨트를 돌리면 흐름도 돈다)', () => {
    const f = beltForward(BELT_SPEC, quatAboutY(90));
    expect(f?.[0]).toBeCloseTo(0, PRECISION);
    expect(f?.[2]).toBeCloseTo(-1, PRECISION);
  });

  it('수직 성분은 버리고 수평만 쓴다', () => {
    const f = beltForward({ ...BELT_SPEC, direction: [0, 5, 1] }, IDENTITY);
    expect(f).toEqual([0, 0, 1]);
  });

  it('수평 성분이 0이면 null — 조용한 "속도 0 벨트" 대신 비활성', () => {
    expect(beltForward({ ...BELT_SPEC, direction: [0, 1, 0] }, IDENTITY)).toBeNull();
  });
});

describe('extentAlong', () => {
  it('축정렬 벨트의 진행축 반길이는 그 축 half extent', () => {
    expect(extentAlong(BELT_HALF, [1, 0, 0], IDENTITY)).toBeCloseTo(BELT_HALF[0], PRECISION);
    expect(extentAlong(BELT_HALF, [0, 0, 1], IDENTITY)).toBeCloseTo(BELT_HALF[2], PRECISION);
  });

  it('벨트를 90° 돌리면 진행축 반길이가 바뀐 축을 따라간다', () => {
    // 월드 +X 방향으로 잰 반길이 = 로컬 Z half extent (벨트가 Y로 90° 돌았으므로)
    expect(extentAlong(BELT_HALF, [1, 0, 0], quatAboutY(90))).toBeCloseTo(
      BELT_HALF[2],
      PRECISION,
    );
  });
});

describe('drivenVelocity', () => {
  const forward: Vec3 = [1, 0, 0];
  const lateral = beltLateral(forward); // [0, 0, -1]

  it('진행축 성분을 벨트 속도로 지정한다', () => {
    const v = drivenVelocity([0, 0, 0], forward, lateral, BELT_SPEED_MPS);
    expect(v[0]).toBeCloseTo(BELT_SPEED_MPS, PRECISION);
  });

  it('★ 측면 속도를 보존한다 — 벨트가 로봇의 밀기를 지우지 않는다', () => {
    const pushedSideways: Vec3 = [0, 0, 0.4];
    const v = drivenVelocity(pushedSideways, forward, lateral, BELT_SPEED_MPS);
    expect(v[0]).toBeCloseTo(BELT_SPEED_MPS, PRECISION); // 진행축은 벨트가 지정
    expect(v[2]).toBeCloseTo(0.4, PRECISION); // 측면은 그대로 — 사물이 벨트를 벗어난다
  });

  it('수직 속도를 보존한다 (낙하·안착)', () => {
    const falling: Vec3 = [0, -1.5, 0];
    expect(drivenVelocity(falling, forward, lateral, BELT_SPEED_MPS)[1]).toBe(-1.5);
  });

  it('진행축 반대로 던져도 벨트 속도로 되돌린다 (역주행 불가)', () => {
    const v = drivenVelocity([-2, 0, 0], forward, lateral, BELT_SPEED_MPS);
    expect(v[0]).toBeCloseTo(BELT_SPEED_MPS, PRECISION);
  });
});

describe('shouldRecycle', () => {
  const geometry = geometryOf([1, 0, 0]);
  const onBeltY = BELT_TOP_Y + ITEM_HALF_M;

  it('벨트 위에 있는 동안은 되돌리지 않는다', () => {
    expect(shouldRecycle([0, onBeltY, 0], BELT_CENTER, geometry)).toBe(false);
    expect(shouldRecycle([BELT_HALF[0] - 0.01, onBeltY, 0], BELT_CENTER, geometry)).toBe(false);
  });

  it('끝을 확실히 지나야 되돌린다 (립에서 떨리지 않게)', () => {
    const justPast = BELT_CENTER[0] + geometry.halfLength + RECYCLE_OVERSHOOT_M / 2;
    expect(shouldRecycle([justPast, onBeltY, 0], BELT_CENTER, geometry)).toBe(false);
    const wellPast = BELT_CENTER[0] + geometry.halfLength + RECYCLE_OVERSHOOT_M * 2;
    expect(shouldRecycle([wellPast, onBeltY, 0], BELT_CENTER, geometry)).toBe(true);
  });

  it('옆으로 굴러 나간 사고품은 되돌리지 않는다 (사용자가 봐야 한다)', () => {
    const past = BELT_CENTER[0] + geometry.halfLength + RECYCLE_OVERSHOOT_M * 2;
    const farLateral = geometry.halfWidth * 5;
    expect(shouldRecycle([past, onBeltY, farLateral], BELT_CENTER, geometry)).toBe(false);
  });

  it('★ 로봇이 집어 든 사물은 되돌리지 않는다 (손에서 사라지면 안 된다)', () => {
    const past = BELT_CENTER[0] + geometry.halfLength + RECYCLE_OVERSHOOT_M * 2;
    const lifted = BELT_TOP_Y + RECYCLE_HEIGHT_ABOVE_BELT_M * 2;
    expect(shouldRecycle([past, lifted, 0], BELT_CENTER, geometry)).toBe(false);
  });

  it('회전한 벨트에서도 진행축 기준으로 판정한다', () => {
    const rotated = geometryOf([0, 0, -1]); // Y 90° 회전 벨트의 진행 방향
    const past: Vec3 = [0, BELT_TOP_Y + ITEM_HALF_M, -(rotated.halfLength + RECYCLE_OVERSHOOT_M * 2)];
    expect(shouldRecycle(past, BELT_CENTER, rotated)).toBe(true);
    // 반대쪽(시작점 뒤)은 되돌리지 않는다
    const behind: Vec3 = [0, BELT_TOP_Y + ITEM_HALF_M, rotated.halfLength + 0.5];
    expect(shouldRecycle(behind, BELT_CENTER, rotated)).toBe(false);
  });
});

describe('recycleTarget', () => {
  const geometry = geometryOf([1, 0, 0]);

  it('진행축만 시작점으로 되돌리고 높이·측면 위치는 보존한다', () => {
    const from: Vec3 = [1.2, BELT_TOP_Y + ITEM_HALF_M, 0.04];
    const to = recycleTarget(from, BELT_CENTER, geometry);
    expect(to[0]).toBeCloseTo(BELT_CENTER[0] - geometry.halfLength + RECYCLE_INSET_M, PRECISION);
    expect(to[1]).toBe(from[1]); // 높이 보존 — 벨트 위로 돌아온다
    expect(to[2]).toBeCloseTo(0.04, PRECISION);
  });

  it('되돌린 지점은 즉시 다시 재순환 조건에 걸리지 않는다 (무한 루프 방지)', () => {
    const from: Vec3 = [1.2, BELT_TOP_Y + ITEM_HALF_M, 0];
    const to = recycleTarget(from, BELT_CENTER, geometry);
    expect(shouldRecycle(to, BELT_CENTER, geometry)).toBe(false);
  });
});

// ── 2. 실 Rapier 통합 ───────────────────────────────────────────────

const BELT_COLLIDER: ColliderSpec = {
  shape: { kind: 'box', halfExtents: BELT_HALF },
  friction: 0.8,
  group: 'ENV',
  collidesWith: ['ROBOT', 'OBJECT'],
  emitEvents: true,
};

const ITEM_COLLIDER: ColliderSpec = {
  shape: { kind: 'box', halfExtents: [ITEM_HALF_M, ITEM_HALF_M, ITEM_HALF_M] },
  friction: 0.6,
  group: 'OBJECT',
  collidesWith: ['ENV', 'ROBOT', 'OBJECT'],
  emitEvents: true,
};

interface Rig {
  world: RapierWorld;
  conveyors: ConveyorRegistry;
  itemBody: BodyId;
  step(seconds: number): void;
}

function makeRig(spec: ConveyorSpec, itemStart: Vec3): Rig {
  const world = new RapierWorld(GRAVITY, TIMESTEP_HZ);
  const beltBody = world.createBody('belt', { bodyType: 'fixed', position: BELT_CENTER });
  world.createCollider(beltBody, BELT_COLLIDER, 'belt');

  const itemBody = world.createBody('item', { bodyType: 'dynamic', position: itemStart });
  world.createCollider(itemBody, ITEM_COLLIDER, 'item');

  const conveyors = new ConveyorRegistry();
  conveyors.add(
    new ConveyorBinding({
      world,
      entityId: 'belt',
      spec,
      halfExtents: BELT_HALF,
      pose: { position: BELT_CENTER, rotation: IDENTITY },
    }),
  );

  return {
    world,
    conveyors,
    itemBody,
    // 엔진과 같은 순서: preStep(conveyors.tickAll) → world.step()
    step: (seconds: number): void => {
      const ticks = Math.round(seconds * TIMESTEP_HZ);
      for (let i = 0; i < ticks; i += 1) {
        conveyors.tickAll();
        world.step();
      }
    },
  };
}

describe('ConveyorBinding — 실 Rapier', () => {
  beforeAll(async () => {
    await initPhysics();
  });

  it('벨트 위의 사물이 진행 방향으로 실려 간다 (속도 × 시간에 근접)', () => {
    const rig = makeRig(BELT_SPEC, [-0.3, BELT_TOP_Y + ITEM_HALF_M, 0]);
    try {
      rig.step(0.2); // 벨트에 안착
      const start = rig.world.getPose(rig.itemBody).position[0];
      rig.step(CARRY_SEC);
      const end = rig.world.getPose(rig.itemBody).position[0];
      const travelled = end - start;
      // 벨트가 **지정하는 것은 속도**다 — 그것을 직접 본다(정착 슬립에 흔들리지 않는다).
      // 선언값은 상한이고 솔버 마찰로 조금 낮게 관측된다 (DRIVEN_SPEED_MIN_RATIO 주석).
      const driven = rig.world.getLinearVelocity(rig.itemBody)[0];
      expect(driven).toBeGreaterThan(BELT_SPEED_MPS * DRIVEN_SPEED_MIN_RATIO);
      expect(driven).toBeLessThanOrEqual(BELT_SPEED_MPS + TOLERANCE_M);
      // 변위는 방향과 규모만 확인한다: 위쪽은 벨트 속도를 넘을 수 없고, 아래쪽은
      // 안착 슬립만큼 손해 볼 수 있으므로 칼날 같은 하한을 두지 않는다.
      expect(travelled).toBeGreaterThan(BELT_SPEED_MPS * CARRY_SEC * CARRY_MIN_RATIO);
      expect(travelled).toBeLessThan(BELT_SPEED_MPS * CARRY_SEC + TOLERANCE_M);
    } finally {
      rig.world.free();
    }
  });

  it('벨트를 돌리면 흐름 방향도 함께 돈다 (−Z로 실려 간다)', () => {
    const world = new RapierWorld(GRAVITY, TIMESTEP_HZ);
    try {
      const rotation = quatAboutY(90);
      const beltBody = world.createBody('belt', {
        bodyType: 'fixed',
        position: BELT_CENTER,
        rotation,
      });
      world.createCollider(beltBody, BELT_COLLIDER, 'belt');
      const itemBody = world.createBody('item', {
        bodyType: 'dynamic',
        position: [0, BELT_TOP_Y + ITEM_HALF_M, 0.3],
      });
      world.createCollider(itemBody, ITEM_COLLIDER, 'item');

      const conveyors = new ConveyorRegistry();
      conveyors.add(
        new ConveyorBinding({
          world,
          entityId: 'belt',
          spec: BELT_SPEC,
          halfExtents: BELT_HALF,
          pose: { position: BELT_CENTER, rotation },
        }),
      );
      const before = world.getPose(itemBody).position;
      for (let i = 0; i < TIMESTEP_HZ; i += 1) {
        conveyors.tickAll();
        world.step();
      }
      const after = world.getPose(itemBody).position;
      expect(after[2]).toBeLessThan(before[2] - 0.1); // −Z로 이동
      expect(Math.abs(after[0] - before[0])).toBeLessThan(TOLERANCE_M * 3); // X는 거의 그대로
    } finally {
      world.free();
    }
  });

  it('recycle이 꺼져 있으면 끝에서 떨어지고 돌아오지 않는다', () => {
    const rig = makeRig(BELT_SPEC, [0.3, BELT_TOP_Y + ITEM_HALF_M, 0]);
    try {
      rig.step(3);
      const p = rig.world.getPose(rig.itemBody).position;
      expect(p[0]).toBeGreaterThan(BELT_HALF[0]); // 벨트 끝 너머
      expect(p[1]).toBeLessThan(BELT_TOP_Y); // 바닥으로 떨어졌다
    } finally {
      rig.world.free();
    }
  });

  it('★ recycle이 켜져 있으면 끝에 도달한 사물이 시작점으로 돌아온다', () => {
    const rig = makeRig({ ...BELT_SPEC, recycle: true }, [0.3, BELT_TOP_Y + ITEM_HALF_M, 0]);
    try {
      rig.step(3);
      const p = rig.world.getPose(rig.itemBody).position;
      // 되돌아온 뒤 다시 실려 가는 중이므로 정확한 x는 관측 시점에 달렸다. 확실한 것은
      // **되돌아왔다**는 사실이다: 되돌리지 않았다면 3초 동안 계속 +X로 가 벨트 끝
      // (+0.5)을 훨씬 지났을 것이다. 그래서 "시작 쪽 절반에 있다"로 판정한다 —
      // 0을 경계로 쓰면 실측(−0.0148)이 15 mm 여유의 칼날 위에 놓인다.
      expect(p[0]).toBeLessThan(BELT_HALF[0] * 0.5);
      expect(p[1]).toBeGreaterThan(BELT_TOP_Y - TOLERANCE_M); // 벨트 위에 남아 있다
    } finally {
      rig.world.free();
    }
  });

  it('★ 재순환은 무한히 계속된다 (사물이 계속 온다)', () => {
    const rig = makeRig({ ...BELT_SPEC, recycle: true }, [-0.4, BELT_TOP_Y + ITEM_HALF_M, 0]);
    try {
      let laps = 0;
      let previousX = rig.world.getPose(rig.itemBody).position[0];
      for (let i = 0; i < 20; i += 1) {
        rig.step(0.5);
        const x = rig.world.getPose(rig.itemBody).position[0];
        if (x < previousX - BELT_HALF[0]) laps += 1; // 크게 뒤로 점프 = 재순환 1회
        previousX = x;
      }
      expect(laps).toBeGreaterThanOrEqual(2);
    } finally {
      rig.world.free();
    }
  });

  /**
   * ★ 이 테스트는 원래 **공허했다**: 물리 스텝 전에는 어떤 접촉도 존재하지 않으므로
   * 벨트 **위**에 놓은 사물로 바꿔도 그대로 통과했다(둘 다 속도 0). 지금은 충분히
   * 스텝을 돌린 뒤, 벨트 **옆**(닿을 수 없는 자리)의 사물과 벨트 **위**의 사물을
   * 같은 조건에서 비교한다 — 대조군이 있어야 "접촉이 진실"을 실제로 증명한다.
   */
  it('벨트에 닿지 않은 사물은 구동되지 않는다 — 벨트 위 사물과의 대조', () => {
    const world = new RapierWorld(GRAVITY, TIMESTEP_HZ);
    try {
      const beltBody = world.createBody('belt', { bodyType: 'fixed', position: BELT_CENTER });
      world.createCollider(beltBody, BELT_COLLIDER, 'belt');

      // 벨트 위 (대조군 — 구동되어야 한다)
      const onBelt = world.createBody('on', {
        bodyType: 'dynamic',
        position: [0, BELT_TOP_Y + ITEM_HALF_M, 0],
      });
      world.createCollider(onBelt, ITEM_COLLIDER, 'on');
      // 벨트 옆 — 폭 밖이라 바닥으로 떨어질 뿐 벨트에 닿지 않는다
      const beside = world.createBody('beside', {
        bodyType: 'dynamic',
        position: [0, BELT_TOP_Y + ITEM_HALF_M, BELT_HALF[2] + ITEM_HALF_M * 4],
      });
      world.createCollider(beside, ITEM_COLLIDER, 'beside');
      // 바닥 (옆 사물이 받쳐질 자리 — 없으면 영원히 낙하해 비교가 흐려진다)
      const floor = world.createBody('floor', { bodyType: 'fixed', position: [0, -0.05, 0] });
      world.createCollider(floor, { ...BELT_COLLIDER, shape: { kind: 'box', halfExtents: [2, 0.05, 2] } }, 'floor');

      const conveyors = new ConveyorRegistry();
      conveyors.add(
        new ConveyorBinding({
          world,
          entityId: 'belt',
          spec: BELT_SPEC,
          halfExtents: BELT_HALF,
          pose: { position: BELT_CENTER, rotation: IDENTITY },
        }),
      );
      for (let i = 0; i < TIMESTEP_HZ; i += 1) {
        conveyors.tickAll();
        world.step();
      }

      // 대조군은 벨트 속도로 구동된다 — 이 어서션이 없으면 위 검사는 다시 공허해진다
      expect(world.getLinearVelocity(onBelt)[0]).toBeGreaterThan(
        BELT_SPEED_MPS * DRIVEN_SPEED_MIN_RATIO,
      );
      // 닿지 않은 사물은 수평으로 밀리지 않는다
      const v = world.getLinearVelocity(beside);
      expect(Math.hypot(v[0], v[2])).toBeLessThan(NOT_DRIVEN_MAX_SPEED_MPS);
    } finally {
      world.free();
    }
  });

  it('수직 속도는 보존한다 — 사물이 벨트로 내려앉는 것을 막지 않는다', () => {
    const rig = makeRig(BELT_SPEC, [0, BELT_TOP_Y + ITEM_HALF_M, 0]);
    try {
      rig.step(0.5); // 접촉 성립
      rig.world.setLinearVelocity(rig.itemBody, [0, -1, 0]);
      rig.conveyors.tickAll();
      const v = rig.world.getLinearVelocity(rig.itemBody);
      expect(v[1]).toBeCloseTo(-1, 6); // 수직 성분 그대로
      expect(v[0]).toBeCloseTo(BELT_SPEED_MPS, 6); // 수평은 벨트 속도
    } finally {
      rig.world.free();
    }
  });

  it('퇴화한 방향(수직)은 완전한 no-op — 조용한 오작동 대신 무동작', () => {
    const rig = makeRig({ direction: [0, 1, 0], speedMps: BELT_SPEED_MPS }, [
      0,
      BELT_TOP_Y + ITEM_HALF_M,
      0,
    ]);
    try {
      expect(rig.conveyors.get('belt')?.beltGeometry).toBeNull();
      rig.step(0.5);
      const v = rig.world.getLinearVelocity(rig.itemBody);
      expect(Math.hypot(v[0], v[2])).toBeLessThan(NOT_DRIVEN_MAX_SPEED_MPS);
    } finally {
      rig.world.free();
    }
  });
});

describe('ConveyorRegistry', () => {
  it('중복 등록은 한국어 오류로 거부한다', () => {
    const world = new RapierWorld(GRAVITY, TIMESTEP_HZ);
    try {
      const make = (): ConveyorBinding =>
        new ConveyorBinding({
          world,
          entityId: 'belt',
          spec: BELT_SPEC,
          halfExtents: BELT_HALF,
          pose: { position: BELT_CENTER, rotation: IDENTITY },
        });
      const registry = new ConveyorRegistry();
      registry.add(make());
      expect(() => registry.add(make())).toThrow(/이미 등록/);
      expect(registry.ids()).toEqual(['belt']);
    } finally {
      world.free();
    }
  });

  it('미등록 id 제거는 no-op (dispose 경로 멱등성)', () => {
    const registry = new ConveyorRegistry();
    expect(() => registry.remove('없는-벨트')).not.toThrow();
    expect(registry.ids()).toEqual([]);
  });
});

// ── 3. 편집 후 정합 (SceneEditor 경로) ──────────────────────────────
//
// 벨트 기하는 빌드 시점 pose에 **고정**된다 — 성능을 위한 선택이지만, 그래서 배치가
// 바뀌는 경로마다 바인딩을 다시 만들어야 한다. 이 계약이 깨지면 화면의 벨트는 옮겨졌는데
// 사물은 옛 자리의 진행축·끝점으로 실려 가는, 눈으로는 원인을 알 수 없는 상태가 된다.

describe('ConveyorBinding — 배치 편집 후 기하 정합', () => {
  beforeAll(async () => {
    await initPhysics();
  });

  it('새 pose로 다시 만든 바인딩은 새 자리의 진행축/끝점을 쓴다', () => {
    const world = new RapierWorld(GRAVITY, TIMESTEP_HZ);
    try {
      const movedCenter: Vec3 = [1.5, BELT_CENTER[1], -0.8];
      const rotated = quatAboutY(90);
      const before = new ConveyorBinding({
        world,
        entityId: 'belt',
        spec: BELT_SPEC,
        halfExtents: BELT_HALF,
        pose: { position: BELT_CENTER, rotation: IDENTITY },
      });
      const after = new ConveyorBinding({
        world,
        entityId: 'belt',
        spec: BELT_SPEC,
        halfExtents: BELT_HALF,
        pose: { position: movedCenter, rotation: rotated },
      });

      // 회전이 반영돼 월드 진행축이 바뀐다
      expect(before.beltGeometry?.forward[0]).toBeCloseTo(1, PRECISION);
      expect(after.beltGeometry?.forward[2]).toBeCloseTo(-1, PRECISION);
      // 진행축 **반길이는 불변**이다: direction이 엔티티 로컬이라 벨트를 돌리면 상자와
      // 방향이 함께 돌아, "벨트가 자기 진행 방향으로 얼마나 긴가"는 달라지지 않는다.
      // (월드 축을 고정하고 벨트만 돌리면 달라진다 — extentAlong 테스트가 그 경우다.)
      expect(after.beltGeometry?.halfLength).toBeCloseTo(BELT_HALF[0], PRECISION);
      expect(after.beltGeometry?.halfWidth).toBeCloseTo(BELT_HALF[2], PRECISION);
    } finally {
      world.free();
    }
  });

  it('★ 레지스트리 재등록이 없으면 옛 기하가 남는다 — remove 후 add가 정상 경로', () => {
    const world = new RapierWorld(GRAVITY, TIMESTEP_HZ);
    try {
      const registry = new ConveyorRegistry();
      const make = (position: Vec3): ConveyorBinding =>
        new ConveyorBinding({
          world,
          entityId: 'belt',
          spec: BELT_SPEC,
          halfExtents: BELT_HALF,
          pose: { position, rotation: IDENTITY },
        });
      registry.add(make(BELT_CENTER));
      // 같은 id로 그냥 add하면 거부된다 — 반드시 remove가 먼저다 (SceneEditor.refreshConveyor)
      expect(() => registry.add(make([2, 0.02, 0]))).toThrow(/이미 등록/);
      registry.remove('belt');
      registry.add(make([2, 0.02, 0]));
      expect(registry.ids()).toEqual(['belt']);
    } finally {
      world.free();
    }
  });
});

// ── 4. 적대적 리뷰가 확인한 결함들의 회귀 ──────────────────────────

describe('ConveyorBinding — 리뷰 확인 결함 회귀', () => {
  beforeAll(async () => {
    await initPhysics();
  });

  /** 벨트 + 아이템 + (옵션) 두 번째 벨트를 만드는 최소 리그 */
  function twoBeltRig(): {
    world: RapierWorld;
    conveyors: ConveyorRegistry;
    itemBody: BodyId;
  } {
    const world = new RapierWorld(GRAVITY, TIMESTEP_HZ);
    const a = world.createBody('beltA', { bodyType: 'fixed', position: BELT_CENTER });
    world.createCollider(a, BELT_COLLIDER, 'beltA');
    // beltB: A에서 멀리 떨어져 있고 아이템은 한 번도 닿은 적이 없다
    const farCenter: Vec3 = [-1.6, BELT_CENTER[1], 0];
    const b = world.createBody('beltB', { bodyType: 'fixed', position: farCenter });
    world.createCollider(b, BELT_COLLIDER, 'beltB');

    const itemBody = world.createBody('item', {
      bodyType: 'dynamic',
      position: [0, BELT_TOP_Y + ITEM_HALF_M, 0],
    });
    world.createCollider(itemBody, ITEM_COLLIDER, 'item');

    const conveyors = new ConveyorRegistry();
    conveyors.add(
      new ConveyorBinding({
        world,
        entityId: 'beltA',
        spec: BELT_SPEC,
        halfExtents: BELT_HALF,
        pose: { position: BELT_CENTER, rotation: IDENTITY },
      }),
    );
    conveyors.add(
      new ConveyorBinding({
        world,
        entityId: 'beltB',
        spec: { ...BELT_SPEC, recycle: true },
        halfExtents: BELT_HALF,
        pose: { position: farCenter, rotation: IDENTITY },
      }),
    );
    return { world, conveyors, itemBody };
  }

  it('★ 다른 벨트 위의 사물을 끌어오지 않는다 (자기가 나른 것만 되돌린다)', () => {
    const rig = twoBeltRig();
    try {
      const before = rig.world.getPose(rig.itemBody).position;
      for (let i = 0; i < TIMESTEP_HZ; i += 1) {
        rig.conveyors.tickAll();
        rig.world.step();
      }
      const after = rig.world.getPose(rig.itemBody).position;
      // beltB의 시작점(x ≈ −2.08)으로 순간이동하지 않았다
      expect(after[0]).toBeGreaterThan(before[0] - TOLERANCE_M);
    } finally {
      rig.world.free();
    }
  });

  it('★ 리셋 직후 tick은 구동하지 않는다 (stale 접촉 그래프로 공중 사물 가속 금지)', () => {
    const rig = makeRig(BELT_SPEC, [0, BELT_TOP_Y + ITEM_HALF_M, 0]);
    try {
      rig.step(0.5); // 접촉 성립 + 구동 중
      expect(rig.world.getLinearVelocity(rig.itemBody)[0]).toBeGreaterThan(0);

      // 씬 리셋을 흉내 낸다: 공중으로 teleport(속도 0) + 스텝 없이 컨베이어 리셋
      rig.world.teleport(rig.itemBody, {
        position: [0, BELT_TOP_Y + 0.3, 0],
        rotation: IDENTITY,
      });
      rig.conveyors.resetAll();
      rig.conveyors.tickAll(); // 리셋 후 첫 preStep
      // stale 접촉으로 벨트 속도가 주입되면 안 된다 — 되감기 재생이 최초 재생과 달라진다
      expect(rig.world.getLinearVelocity(rig.itemBody)[0]).toBe(0);
    } finally {
      rig.world.free();
    }
  });

  it('★ 겹친 벨트가 속도를 합성하지 않는다 (먼저 선언된 벨트가 이긴다)', () => {
    const world = new RapierWorld(GRAVITY, TIMESTEP_HZ);
    try {
      // 같은 자리에 두 벨트: 하나는 +X, 하나는 +Z
      const a = world.createBody('beltA', { bodyType: 'fixed', position: BELT_CENTER });
      world.createCollider(a, BELT_COLLIDER, 'beltA');
      const b = world.createBody('beltB', { bodyType: 'fixed', position: BELT_CENTER });
      world.createCollider(b, BELT_COLLIDER, 'beltB');
      const itemBody = world.createBody('item', {
        bodyType: 'dynamic',
        position: [0, BELT_TOP_Y + ITEM_HALF_M, 0],
      });
      world.createCollider(itemBody, ITEM_COLLIDER, 'item');

      const conveyors = new ConveyorRegistry();
      for (const [id, dir] of [
        ['beltA', [1, 0, 0] as Vec3],
        ['beltB', [0, 0, 1] as Vec3],
      ] as const) {
        conveyors.add(
          new ConveyorBinding({
            world,
            entityId: id,
            spec: { direction: [...dir], speedMps: BELT_SPEED_MPS },
            halfExtents: BELT_HALF,
            pose: { position: BELT_CENTER, rotation: IDENTITY },
          }),
        );
      }
      for (let i = 0; i < TIMESTEP_HZ; i += 1) {
        conveyors.tickAll();
        world.step();
      }
      const v = world.getLinearVelocity(itemBody);
      // 합성되면 |v| ≈ √2 × speed. 중재가 있으면 먼저 선언된 beltA(+X)만 구동한다.
      expect(Math.hypot(v[0], v[2])).toBeLessThan(BELT_SPEED_MPS * 1.2);
      expect(v[0]).toBeGreaterThan(BELT_SPEED_MPS * DRIVEN_SPEED_MIN_RATIO);
      expect(Math.abs(v[2])).toBeLessThan(NOT_DRIVEN_MAX_SPEED_MPS);
    } finally {
      world.free();
    }
  });

  it('★ 벨트에서 손 뗀 지 오래된 사물은 되돌리지 않는다 (로봇이 든 물건 보호)', () => {
    const rig = makeRig({ ...BELT_SPEC, recycle: true }, [0, BELT_TOP_Y + ITEM_HALF_M, 0]);
    try {
      rig.step(0.5); // 벨트가 한 번 실어 나른다 (재순환 자격 획득)
      // 로봇이 집어 든 상황을 흉내: 벨트에서 떼어 공중에 유지하며 유효 기간을 넘긴다
      const heldAbove = BELT_TOP_Y + 0.10; // 12cm 상한보다 낮은 "정상 이송 높이"
      const holdTicks = RECYCLE_GRACE_TICKS * 2;
      for (let i = 0; i < holdTicks; i += 1) {
        rig.world.teleport(rig.itemBody, {
          position: [BELT_HALF[0] + 0.2, heldAbove, 0], // 끝을 지난 자리에 들고 있다
          rotation: IDENTITY,
        });
        rig.conveyors.tickAll();
        rig.world.step();
      }
      // 손에서 빼앗기지 않았다 — 벨트 시작점으로 순간이동하지 않았다
      expect(rig.world.getPose(rig.itemBody).position[0]).toBeGreaterThan(BELT_HALF[0]);
    } finally {
      rig.world.free();
    }
  });
});
