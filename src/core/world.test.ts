// core/world.test.ts — RapierWorld 단위/통합 테스트 (규범: docs/SIMULATION.md §4)
//
// rapier3d-compat는 WASM을 base64로 내장하므로 Node(vitest) 환경에서 그대로 돈다.
// 모든 물리 API 사용 전 initPhysics()를 await한다 (CLAUDE.md §2.7).

import { beforeAll, describe, expect, it } from 'vitest';
import {
  COLLISION_GROUP_BITS,
  RapierWorld,
  groupsToBits,
  initPhysics,
  interactionGroups,
} from './world';
import type { BodyId, ColliderId, ContactEvent, Pose } from './types';
import type { ColliderSpec, Quat, Vec3 } from '../schema/types';

// ── 테스트 상수 (매직넘버 금지 — CLAUDE.md §4) ──────────────────────

const GRAVITY: Vec3 = [0, -9.81, 0];
const TIMESTEP_HZ = 240;
const SETTLE_TICKS = 2 * TIMESTEP_HZ;          // 약 2초
const IDENTITY_QUAT: Quat = [0, 0, 0, 1];

const BOX_HALF_M = 0.05;
const DROP_HEIGHT_M = 1.0;
const GROUND_HALF_EXTENTS_M: Vec3 = [2, 0.1, 2];
const GROUND_CENTER_Y_M = -GROUND_HALF_EXTENTS_M[1]; // 윗면이 y=0

const REST_HEIGHT_TOLERANCE_M = 0.01;      // 정착 높이 허용 오차(솔버 침투 여유 포함)
const REST_TICK_DELTA_TOLERANCE_M = 1e-4;  // 정지 판정: 1 tick 이동량 상한

// ── 헬퍼 ────────────────────────────────────────────────────────────

function groundColliderSpec(overrides: Partial<ColliderSpec> = {}): ColliderSpec {
  return {
    shape: { kind: 'box', halfExtents: GROUND_HALF_EXTENTS_M },
    group: 'ENV',
    collidesWith: ['ROBOT', 'OBJECT'],
    ...overrides,
  };
}

function boxColliderSpec(overrides: Partial<ColliderSpec> = {}): ColliderSpec {
  return {
    shape: { kind: 'box', halfExtents: [BOX_HALF_M, BOX_HALF_M, BOX_HALF_M] },
    group: 'OBJECT',
    collidesWith: ['ENV', 'ROBOT', 'OBJECT'],
    ...overrides,
  };
}

interface FallingBoxWorld {
  world: RapierWorld;
  groundBody: BodyId;
  boxBody: BodyId;
  groundCollider: ColliderId;
  boxCollider: ColliderId;
}

/** 고정 바닥(윗면 y=0) + y=1에서 낙하하는 동적 박스 */
function buildFallingBoxWorld(opts: {
  boxCollider?: Partial<ColliderSpec>;
  boxRotation?: Quat;
} = {}): FallingBoxWorld {
  const world = new RapierWorld(GRAVITY, TIMESTEP_HZ);
  const groundBody = world.createBody('ground', {
    bodyType: 'fixed',
    position: [0, GROUND_CENTER_Y_M, 0],
  });
  const groundCollider = world.createCollider(groundBody, groundColliderSpec(), 'ground');
  const boxBody = world.createBody('box', {
    bodyType: 'dynamic',
    position: [0, DROP_HEIGHT_M, 0],
    rotation: opts.boxRotation,
  });
  const boxCollider = world.createCollider(boxBody, boxColliderSpec(opts.boxCollider), 'box');
  return { world, groundBody, boxBody, groundCollider, boxCollider };
}

function pairOf(e: ContactEvent): string {
  return [e.a, e.b].sort().join(',');
}

beforeAll(async () => {
  await initPhysics();
});

// ── 순수 헬퍼 테스트 ────────────────────────────────────────────────

describe('interactionGroups (pure)', () => {
  it('packs memberships into the high 16 bits and filter into the low 16 bits', () => {
    const memberships = COLLISION_GROUP_BITS.ENV;
    const filter = COLLISION_GROUP_BITS.ROBOT | COLLISION_GROUP_BITS.OBJECT;
    expect(interactionGroups(memberships, filter)).toBe((memberships << 16) | filter);
    expect(interactionGroups(memberships, filter)).toBe(0x0001_0006);
  });

  it('masks both fields to 16 bits', () => {
    expect(interactionGroups(0x1_0001, 0x2_0002)).toBe((1 << 16) | 2);
  });

  it('CLAUDE.md §5 bit assignments are fixed', () => {
    expect(COLLISION_GROUP_BITS.ENV).toBe(1 << 0);
    expect(COLLISION_GROUP_BITS.ROBOT).toBe(1 << 1);
    expect(COLLISION_GROUP_BITS.OBJECT).toBe(1 << 2);
    expect(COLLISION_GROUP_BITS.SENSOR_ZONE).toBe(1 << 3);
    expect(COLLISION_GROUP_BITS.DEBUG).toBe(1 << 15);
  });

  it('groupsToBits ORs the group bits', () => {
    expect(groupsToBits([])).toBe(0);
    expect(groupsToBits(['ENV', 'OBJECT'])).toBe((1 << 0) | (1 << 2));
    expect(groupsToBits(['ENV', 'ENV'])).toBe(1 << 0);
  });
});

// ── 물리 통합 테스트 ────────────────────────────────────────────────

describe('RapierWorld', () => {
  it('(a) falling box settles to rest on the ground', () => {
    const { world, boxBody } = buildFallingBoxWorld();
    try {
      for (let i = 0; i < SETTLE_TICKS; i++) world.step();

      // 정지 판정: 마지막 1 tick 동안의 위치 변화가 충분히 작다 (속도 ≈ delta/dt)
      const before = world.getPose(boxBody).position;
      world.step();
      const after = world.getPose(boxBody).position;
      const tickDeltaM = Math.hypot(
        after[0] - before[0], after[1] - before[1], after[2] - before[2],
      );
      expect(tickDeltaM).toBeLessThan(REST_TICK_DELTA_TOLERANCE_M);

      // 바닥 윗면(y=0) 위에 half-extent 높이로 정착 (터널링 없음)
      expect(after[1]).toBeGreaterThan(0);
      expect(Math.abs(after[1] - BOX_HALF_M)).toBeLessThan(REST_HEIGHT_TOLERANCE_M);
    } finally {
      world.free();
    }
  });

  it('(b) determinism: identical worlds produce bit-identical trajectories', () => {
    const N_TICKS = 500;
    const SAMPLE_EVERY_TICKS = 100;
    // 살짝 기울여 떨어뜨려 솔버 경로를 비자명하게 만든다 (z축 0.2rad 회전)
    const TILT_QUAT: Quat = [0, 0, Math.sin(0.1), Math.cos(0.1)];

    const run = (): Pose[] => {
      const { world, boxBody } = buildFallingBoxWorld({ boxRotation: TILT_QUAT });
      const samples: Pose[] = [];
      try {
        for (let tick = 1; tick <= N_TICKS; tick++) {
          world.step();
          if (tick % SAMPLE_EVERY_TICKS === 0) samples.push(world.getPose(boxBody));
        }
      } finally {
        world.free();
      }
      return samples;
    };

    const first = run();
    const second = run();
    expect(first).toHaveLength(N_TICKS / SAMPLE_EVERY_TICKS);
    expect(second).toStrictEqual(first); // 부동소수 완전 일치 (결정론)
  });

  it('(c) emits start then stop contact events between the right entity ids', () => {
    const { world, boxBody } = buildFallingBoxWorld({ boxCollider: { emitEvents: true } });
    try {
      const startEvents: ContactEvent[] = [];
      for (let i = 0; i < SETTLE_TICKS; i++) {
        for (const e of world.step()) {
          if (e.phase === 'start') startEvents.push(e);
        }
      }
      expect(startEvents.length).toBeGreaterThanOrEqual(1);
      for (const e of startEvents) {
        expect(pairOf(e)).toBe('box,ground');
        expect(e.kind).toBe('contact');
      }

      // 접촉 중인 박스를 들어올리면(teleport) 다음 스텝들에서 stop 이벤트가 나온다
      world.teleport(boxBody, { position: [0, DROP_HEIGHT_M, 0], rotation: IDENTITY_QUAT });
      const stopEvents: ContactEvent[] = [];
      for (let i = 0; i < TIMESTEP_HZ; i++) {
        for (const e of world.step()) {
          if (e.phase === 'stop') stopEvents.push(e);
        }
      }
      expect(stopEvents.some((e) => pairOf(e) === 'box,ground' && e.kind === 'contact')).toBe(true);
    } finally {
      world.free();
    }
  });

  it('(d) sensor collider reports kind sensor and does not physically block', () => {
    const SENSOR_CENTER_Y_M = 0.5;
    const SENSOR_HALF_EXTENTS_M: Vec3 = [0.2, 0.05, 0.2];

    const world = new RapierWorld(GRAVITY, TIMESTEP_HZ);
    try {
      const groundBody = world.createBody('ground', {
        bodyType: 'fixed',
        position: [0, GROUND_CENTER_Y_M, 0],
      });
      world.createCollider(groundBody, groundColliderSpec(), 'ground');

      const zoneBody = world.createBody('zone', {
        bodyType: 'fixed',
        position: [0, SENSOR_CENTER_Y_M, 0],
      });
      world.createCollider(zoneBody, {
        shape: { kind: 'box', halfExtents: SENSOR_HALF_EXTENTS_M },
        isSensor: true,
        group: 'SENSOR_ZONE',
        collidesWith: ['OBJECT'],
        emitEvents: true,
      }, 'zone');

      const boxBody = world.createBody('box', {
        bodyType: 'dynamic',
        position: [0, DROP_HEIGHT_M, 0],
      });
      world.createCollider(
        boxBody,
        boxColliderSpec({ collidesWith: ['ENV', 'SENSOR_ZONE'], emitEvents: true }),
        'box',
      );

      const events: ContactEvent[] = [];
      for (let i = 0; i < SETTLE_TICKS; i++) events.push(...world.step());

      const sensorEvents = events.filter((e) => e.kind === 'sensor' && pairOf(e) === 'box,zone');
      expect(sensorEvents.some((e) => e.phase === 'start')).toBe(true);
      expect(sensorEvents.some((e) => e.phase === 'stop')).toBe(true); // 통과 완료

      // 감지 영역에 막히지 않고 바닥까지 떨어져 정착했다
      const restY = world.getPose(boxBody).position[1];
      expect(Math.abs(restY - BOX_HALF_M)).toBeLessThan(REST_HEIGHT_TOLERANCE_M);
    } finally {
      world.free();
    }
  });

  it('(e) collision groups: a box whose collidesWith excludes ENV falls through the ground', () => {
    const { world, boxBody } = buildFallingBoxWorld({
      boxCollider: { collidesWith: ['OBJECT'], emitEvents: true },
    });
    try {
      const events: ContactEvent[] = [];
      for (let i = 0; i < SETTLE_TICKS; i++) events.push(...world.step());

      expect(events).toEqual([]); // 이벤트 없음
      // 정착하지 못하고 바닥을 관통해 계속 낙하 중
      expect(world.getPose(boxBody).position[1]).toBeLessThan(-DROP_HEIGHT_M);
    } finally {
      world.free();
    }
  });

  it('setKinematicPose drives a kinematicPosition body to the target next step', () => {
    const TARGET: Pose = { position: [0.1, 0.3, -0.2], rotation: IDENTITY_QUAT };
    const world = new RapierWorld(GRAVITY, TIMESTEP_HZ);
    try {
      const linkBody = world.createBody('link', {
        bodyType: 'kinematicPosition',
        position: [0, 0.2, 0],
      });
      world.setKinematicPose(linkBody, TARGET);
      world.step();
      const pose = world.getPose(linkBody);
      expect(pose.position[0]).toBeCloseTo(TARGET.position[0], 6);
      expect(pose.position[1]).toBeCloseTo(TARGET.position[1], 6);
      expect(pose.position[2]).toBeCloseTo(TARGET.position[2], 6);
    } finally {
      world.free();
    }
  });

  it('owns handle mappings; removeEntity and clear() reset them', () => {
    const { world, boxBody, boxCollider, groundBody, groundCollider } = buildFallingBoxWorld();
    try {
      expect(world.entityOfCollider(boxCollider)).toBe('box');
      expect(world.entityOfCollider(groundCollider)).toBe('ground');
      expect(world.bodiesOfEntity('box')).toEqual([boxBody]);
      expect(world.bodiesOfEntity('ground')).toEqual([groundBody]);
      expect(world.bodiesOfEntity('missing')).toEqual([]);

      world.removeEntity('box');
      expect(world.entityOfCollider(boxCollider)).toBeUndefined();
      expect(world.bodiesOfEntity('box')).toEqual([]);
      expect(world.entityOfCollider(groundCollider)).toBe('ground'); // 다른 엔티티는 무관

      world.clear();
      expect(world.entityOfCollider(groundCollider)).toBeUndefined();
      expect(world.bodiesOfEntity('ground')).toEqual([]);
      expect(world.step()).toEqual([]); // 빈 월드 스텝은 무해
    } finally {
      world.free();
    }
  });

  it('rejects unsupported collider shapes with a clear Phase 3+ error', () => {
    const world = new RapierWorld(GRAVITY, TIMESTEP_HZ);
    try {
      const bodyId = world.createBody('mesh-entity', { bodyType: 'fixed', position: [0, 0, 0] });
      expect(() => world.createCollider(bodyId, {
        shape: { kind: 'convexHull', ref: 'assets/mesh.obj' },
        group: 'OBJECT',
        collidesWith: ['ENV'],
      }, 'mesh-entity')).toThrow(/not yet supported \(Phase 3\+\)/);
    } finally {
      world.free();
    }
  });
});
