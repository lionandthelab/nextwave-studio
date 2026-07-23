// core/world.test.ts — RapierWorld 단위/통합 테스트 (규범: docs/SIMULATION.md §4)
//
// rapier3d-compat는 WASM을 base64로 내장하므로 Node(vitest) 환경에서 그대로 돈다.
// 모든 물리 API 사용 전 initPhysics()를 await한다 (CLAUDE.md §2.7).
//
// RAPIER 직접 import는 convexHull null 분기 모킹 전용이다 — 프로덕션 코드의
// "Rapier는 world.ts 밖으로 새지 않는다" 불변식(CLAUDE.md §3/§7)은 core 모듈에
// 적용되며, 이 파일은 그 경계 자체를 검증하는 테스트다.

import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import RAPIER from '@dimforge/rapier3d-compat';
import {
  COLLISION_GROUP_BITS,
  RapierWorld,
  groupsToBits,
  initPhysics,
  interactionGroups,
} from './world';
import type { BodyId, ColliderId, ContactEvent, Pose } from './types';
import type { MeshAssetResolver } from './scene-edit-types';
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

  it('resolver 없이 convexHull collider를 만들면 한국어 오류로 거부한다', () => {
    const world = new RapierWorld(GRAVITY, TIMESTEP_HZ);
    try {
      const bodyId = world.createBody('mesh-entity', { bodyType: 'fixed', position: [0, 0, 0] });
      expect(() => world.createCollider(bodyId, {
        shape: { kind: 'convexHull', ref: 'assets/mesh.obj' },
        group: 'OBJECT',
        collidesWith: ['ENV'],
      }, 'mesh-entity')).toThrow(/MeshAssetResolver가 필요합니다/);
    } finally {
      world.free();
    }
  });

  it("fromVisual collider는 robot 엔티티 전용 오류로 거부한다 (URDF 경로는 scene-loader 몫)", () => {
    const world = new RapierWorld(GRAVITY, TIMESTEP_HZ);
    try {
      const bodyId = world.createBody('deco', { bodyType: 'fixed', position: [0, 0, 0] });
      expect(() => world.createCollider(bodyId, {
        shape: { kind: 'fromVisual' },
        group: 'OBJECT',
        collidesWith: ['ENV'],
      }, 'deco')).toThrow(/robot 엔티티 전용/);
    } finally {
      world.free();
    }
  });
});

// ── 메시 collider (Phase 7 — MeshAssetResolver 주입 convexHull/trimesh) ──

/** 정육면체 8꼭짓점 point cloud (half-extent 기준) — convex hull 결과 = box와 동일 */
function cubePointCloud(halfM: number): Float32Array {
  const points: number[] = [];
  for (const x of [-halfM, halfM]) {
    for (const y of [-halfM, halfM]) {
      for (const z of [-halfM, halfM]) points.push(x, y, z);
    }
  }
  return new Float32Array(points);
}

const CUBE_ASSET_REF = 'asset://cube';
const QUAD_ASSET_REF = 'asset://quad';
/** 정점만 있고 인덱스가 없는 에셋 — trimesh 거부 검증용 */
const POINTS_ONLY_ASSET_REF = 'asset://points-only';

const QUAD_HALF_M = 2;
/** y=0 평면의 사각형(삼각형 2개) — trimesh 바닥 */
const QUAD_VERTICES = new Float32Array([
  -QUAD_HALF_M, 0, -QUAD_HALF_M,
  QUAD_HALF_M, 0, -QUAD_HALF_M,
  QUAD_HALF_M, 0, QUAD_HALF_M,
  -QUAD_HALF_M, 0, QUAD_HALF_M,
]);
const QUAD_INDICES = new Uint32Array([0, 1, 2, 0, 2, 3]);

const testResolver: MeshAssetResolver = {
  getPoints: (ref) => {
    if (ref === CUBE_ASSET_REF || ref === POINTS_ONLY_ASSET_REF) return cubePointCloud(BOX_HALF_M);
    if (ref === QUAD_ASSET_REF) return QUAD_VERTICES;
    return undefined;
  },
  getIndices: (ref) => (ref === QUAD_ASSET_REF ? QUAD_INDICES : undefined),
};

describe('RapierWorld — 메시 collider (convexHull/trimesh + resolver)', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('정육면체 point cloud convexHull 바디가 바닥에 정확히 정착하고 접촉 이벤트를 낸다', () => {
    const world = new RapierWorld(GRAVITY, TIMESTEP_HZ, testResolver);
    try {
      const groundBody = world.createBody('ground', {
        bodyType: 'fixed',
        position: [0, GROUND_CENTER_Y_M, 0],
      });
      world.createCollider(groundBody, groundColliderSpec(), 'ground');

      const hullBody = world.createBody('hull', {
        bodyType: 'dynamic',
        position: [0, DROP_HEIGHT_M, 0],
      });
      world.createCollider(hullBody, {
        shape: { kind: 'convexHull', ref: CUBE_ASSET_REF },
        group: 'OBJECT',
        collidesWith: ['ENV'],
        emitEvents: true,
      }, 'hull');

      const events: ContactEvent[] = [];
      for (let i = 0; i < SETTLE_TICKS; i++) events.push(...world.step());

      // 접촉 이벤트가 EventQueue 경유로 실제 감지된다 (CLAUDE.md §2.4)
      expect(
        events.some((e) => e.phase === 'start' && e.kind === 'contact' && pairOf(e) === 'ground,hull'),
      ).toBe(true);
      // 정육면체 hull은 box와 동일 — 바닥 위 half-extent 높이에 정착 (관통/터널링 없음)
      const restY = world.getPose(hullBody).position[1];
      expect(Math.abs(restY - BOX_HALF_M)).toBeLessThan(REST_HEIGHT_TOLERANCE_M);
    } finally {
      world.free();
    }
  });

  it('trimesh(fixed) 바닥 위에 동적 박스가 정착하고 접촉 이벤트를 낸다', () => {
    const world = new RapierWorld(GRAVITY, TIMESTEP_HZ, testResolver);
    try {
      const triBody = world.createBody('tri-ground', { bodyType: 'fixed', position: [0, 0, 0] });
      world.createCollider(triBody, {
        shape: { kind: 'trimesh', ref: QUAD_ASSET_REF },
        group: 'ENV',
        collidesWith: ['OBJECT'],
      }, 'tri-ground');

      const boxBody = world.createBody('box', {
        bodyType: 'dynamic',
        position: [0.2, DROP_HEIGHT_M, 0.1], // 삼각형 내부 임의 지점
      });
      world.createCollider(boxBody, boxColliderSpec({ emitEvents: true }), 'box');

      const events: ContactEvent[] = [];
      for (let i = 0; i < SETTLE_TICKS; i++) events.push(...world.step());

      expect(
        events.some((e) => e.phase === 'start' && e.kind === 'contact' && pairOf(e) === 'box,tri-ground'),
      ).toBe(true);
      const restY = world.getPose(boxBody).position[1];
      expect(Math.abs(restY - BOX_HALF_M)).toBeLessThan(REST_HEIGHT_TOLERANCE_M);
    } finally {
      world.free();
    }
  });

  it('등록되지 않은 에셋 ref는 한국어 오류로 거부한다', () => {
    const world = new RapierWorld(GRAVITY, TIMESTEP_HZ, testResolver);
    try {
      const bodyId = world.createBody('mesh-entity', { bodyType: 'fixed', position: [0, 0, 0] });
      expect(() => world.createCollider(bodyId, {
        shape: { kind: 'convexHull', ref: 'asset://ghost' },
        group: 'OBJECT',
        collidesWith: ['ENV'],
      }, 'mesh-entity')).toThrow(/해석할 수 없습니다/);
    } finally {
      world.free();
    }
  });

  it('인덱스 없는 에셋의 trimesh collider는 한국어 오류로 거부한다 (convexHull 안내)', () => {
    const world = new RapierWorld(GRAVITY, TIMESTEP_HZ, testResolver);
    try {
      const bodyId = world.createBody('mesh-entity', { bodyType: 'fixed', position: [0, 0, 0] });
      expect(() => world.createCollider(bodyId, {
        shape: { kind: 'trimesh', ref: POINTS_ONLY_ASSET_REF },
        group: 'ENV',
        collidesWith: ['OBJECT'],
      }, 'mesh-entity')).toThrow(/인덱스가 없습니다.*convexHull/);
    } finally {
      world.free();
    }
  });

  it("ColliderDesc.convexHull이 null을 반환하면 '볼록 껍질 생성 실패' 오류를 던진다", () => {
    // 현 rapier3d-compat 버전은 desc 생성 시점에 hull을 계산하지 않아 null 경로를
    // 실데이터로 유발할 수 없다 — 시그니처 계약(| null)의 방어 분기를 모킹으로 검증한다.
    vi.spyOn(RAPIER.ColliderDesc, 'convexHull').mockReturnValue(null);
    const world = new RapierWorld(GRAVITY, TIMESTEP_HZ, testResolver);
    try {
      const bodyId = world.createBody('mesh-entity', { bodyType: 'dynamic', position: [0, 1, 0] });
      expect(() => world.createCollider(bodyId, {
        shape: { kind: 'convexHull', ref: CUBE_ASSET_REF },
        group: 'OBJECT',
        collidesWith: ['ENV'],
      }, 'mesh-entity')).toThrow(/볼록 껍질 생성 실패/);
    } finally {
      world.free();
    }
  });
});
