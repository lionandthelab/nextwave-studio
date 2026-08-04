// core/scene-loader.test.ts — 샘플 씬 검증 + 헤드리스 물리 통합 테스트
//
// rapier3d-compat는 WASM을 base64로 내장해 Node(vitest)에서 그대로 돈다.
// 렌더 계층은 no-op RenderSceneApi로 대체한다 — scene-loader는 three 없이 완결적이다.
// 모든 물리 API 사용 전 initPhysics()를 await한다 (CLAUDE.md §2.7).

import { beforeAll, describe, expect, it, vi } from 'vitest';
import fallingBoxesSceneJson from '../assets/scenes/falling-boxes.scene.json';
import armAndBoxesSceneJson from '../assets/scenes/arm-and-boxes.scene.json';
import { validateScene } from '../schema/validate';
import { isRobotSpec } from '../schema/types';
import type { RobotSpec, SceneSpec, Vec3 } from '../schema/types';
import type {
  BodyId,
  ColliderId,
  ContactEvent,
  EntityId,
  PhysicsBodyInit,
  PhysicsWorld,
  Pose,
} from './types';
import type { JointInfo, LinkColliderDef } from './robot-types';
import { initPhysics, RapierWorld } from './world';
import { RenderSync } from './sync';
import { GROUND_ENTITY_ID, SceneLoader } from './scene-loader';
import type { RenderSceneApi, RobotHandle, RobotLoadRequest, VisualNode } from './scene-loader';

// ── 테스트 상수 (매직넘버 금지 — CLAUDE.md §4) ──────────────────────

const SIM_SECONDS = 3;
const RESET_POSITION_DECIMALS = 6;
const EXPECTED_DYNAMIC_ENTITY_COUNT = 5;
const EXPECTED_EMIT_EVENTS_MIN_COUNT = 2;

// robot 브랜치 픽스처 상수
const TEST_TIMESTEP_HZ = 240;
const ROBOT_ID = 'test-arm';
const FAKE_URDF_PATH = 'assets/robots/fake/fake.urdf';
/** 가짜 FK: L1/L2의 관절값 무관 기본 높이 (m) */
const L1_BASE_Y_M = 0.1;
const L2_BASE_Y_M = 0.2;
/** j1 revolute limits (rad) / j2 prismatic limits (m) */
const J1_LIMITS_RAD: [number, number] = [-1, 1];
const J2_LIMITS_M: [number, number] = [0, 0.05];
const HOME_J1_RAD = 0.5;
/** scene-loader 로봇 링크 collider 규약 기대값 (CLAUDE.md §5) */
const EXPECTED_ROBOT_LINK_FRICTION = 0.8;

// ── 헬퍼 ────────────────────────────────────────────────────────────

/** 시각 노드 스텁 — RenderSync.apply가 만지는 position/quaternion만 갖춘 no-op */
function stubNode(): VisualNode {
  const stub = {
    position: { set: () => undefined },
    quaternion: { set: () => undefined },
  };
  return stub as unknown as VisualNode;
}

interface CountingRenderApi extends RenderSceneApi {
  readonly counts: { addPrimitive: number; addGround: number; setPose: number; remove: number };
}

function makeRenderApi(loadRobot?: RenderSceneApi['loadRobot']): CountingRenderApi {
  const counts = { addPrimitive: 0, addGround: 0, setPose: 0, remove: 0 };
  return {
    counts,
    addPrimitive: () => {
      counts.addPrimitive += 1;
      return stubNode();
    },
    addGround: () => {
      counts.addGround += 1;
      return stubNode();
    },
    setPose: () => {
      counts.setPose += 1;
    },
    remove: () => {
      counts.remove += 1;
    },
    loadRobot:
      loadRobot ??
      (() => Promise.reject(new Error('이 테스트 씬에는 robot 엔티티가 없어야 합니다'))),
  };
}

function loadValidatedSpec(): SceneSpec {
  const result = validateScene(fallingBoxesSceneJson);
  if (!result.ok) throw new Error(`샘플 씬 검증 실패:\n${result.errors.join('\n')}`);
  return result.value;
}

function stepSeconds(world: RapierWorld, spec: SceneSpec, seconds: number): ContactEvent[] {
  const totalSteps = Math.round(seconds * spec.timestepHz);
  const events: ContactEvent[] = [];
  for (let i = 0; i < totalSteps; i += 1) {
    events.push(...world.step());
  }
  return events;
}

function dynamicEntitiesOf(spec: SceneSpec): SceneSpec['entities'] {
  return spec.entities.filter((e) => e.physics?.bodyType === 'dynamic');
}

/** 엔티티의 유일한 바디 id를 얻는다 (noUncheckedIndexedAccess 대응 명시적 가드) */
function soleBodyOf(world: RapierWorld, entityId: string): number {
  const bodies = world.bodiesOfEntity(entityId);
  expect(bodies).toHaveLength(1);
  const bodyId = bodies[0];
  if (bodyId === undefined) throw new Error(`엔티티 '${entityId}'의 바디가 없습니다`);
  return bodyId;
}

beforeAll(async () => {
  await initPhysics();
});

// ── 1. 샘플 씬 스키마 검증 ──────────────────────────────────────────

describe('falling-boxes.scene.json', () => {
  it('validateScene을 통과한다 (ok: true)', () => {
    const result = validateScene(fallingBoxesSceneJson);
    if (!result.ok) throw new Error(result.errors.join('\n'));
    expect(result.ok).toBe(true);
  });

  it('요구 구성을 갖춘다: 동적 5개(OBJECT) + 고정 벽 1개, emitEvents ≥ 2', () => {
    const spec = loadValidatedSpec();
    const dynamics = dynamicEntitiesOf(spec);
    expect(dynamics).toHaveLength(EXPECTED_DYNAMIC_ENTITY_COUNT);
    for (const entity of dynamics) {
      for (const collider of entity.physics?.colliders ?? []) {
        expect(collider.group).toBe('OBJECT');
        expect(collider.collidesWith).toEqual(['ENV', 'ROBOT', 'OBJECT']);
      }
      const y = entity.transform.position[1];
      expect(y).toBeGreaterThanOrEqual(0.5);
      expect(y).toBeLessThanOrEqual(1.5);
    }

    const emitters = dynamics.filter((e) =>
      (e.physics?.colliders ?? []).some((c) => c.emitEvents === true),
    );
    expect(emitters.length).toBeGreaterThanOrEqual(EXPECTED_EMIT_EVENTS_MIN_COUNT);

    const walls = spec.entities.filter(
      (e) => e.type === 'static' && e.physics?.bodyType === 'fixed',
    );
    expect(walls).toHaveLength(1);
    expect(spec.environment?.ground).toBe(true);
  });
});

// ── 2. 헤드리스 물리 통합 (SceneLoader + RapierWorld, 렌더 no-op) ───

describe('SceneLoader (headless integration)', () => {
  it('3초 시뮬 후 모든 동적 바디가 바닥 위(y > 0, 시작 높이 미만)에 있고 접촉 이벤트가 발생한다', async () => {
    const spec = loadValidatedSpec();
    const world = new RapierWorld(spec.gravity, spec.timestepHz);
    try {
      const sync = new RenderSync(world);
      const renderApi = makeRenderApi();
      const handle = await new SceneLoader(world, renderApi, sync).build(spec);

      // 핸들: ground 예약 id + 모든 스펙 엔티티 id
      expect(handle.entityIds).toContain(GROUND_ENTITY_ID);
      for (const entity of spec.entities) {
        expect(handle.entityIds).toContain(entity.id);
      }
      expect(renderApi.counts.addGround).toBe(1);
      expect(renderApi.counts.addPrimitive).toBe(spec.entities.length);

      const events = stepSeconds(world, spec, SIM_SECONDS);

      for (const entity of dynamicEntitiesOf(spec)) {
        const bodyId = soleBodyOf(world, entity.id);
        const y = world.getPose(bodyId).position[1];
        const startY = entity.transform.position[1];
        expect(y, `${entity.id}: 바닥을 관통하지 않아야 한다`).toBeGreaterThan(0);
        expect(y, `${entity.id}: 시작 높이 아래로 낙하해야 한다`).toBeLessThan(startY);
      }

      // 충돌 이벤트는 EventQueue 경유로 실제 감지되어야 한다 (CLAUDE.md §2.4, DoD §8)
      expect(events.some((e) => e.phase === 'start' && e.kind === 'contact')).toBe(true);
    } finally {
      world.free();
    }
  });

  it('reset()은 모든 바디를 초기 스펙 트랜스폼으로 되돌린다 (결정론적 재생)', async () => {
    const spec = loadValidatedSpec();
    const world = new RapierWorld(spec.gravity, spec.timestepHz);
    try {
      const sync = new RenderSync(world);
      const handle = await new SceneLoader(world, makeRenderApi(), sync).build(spec);

      stepSeconds(world, spec, 1);
      handle.reset();

      for (const entity of spec.entities) {
        if (!entity.physics) continue;
        const bodyId = soleBodyOf(world, entity.id);
        const pose = world.getPose(bodyId);
        const initial = entity.transform.position;
        expect(pose.position[0]).toBeCloseTo(initial[0], RESET_POSITION_DECIMALS);
        expect(pose.position[1]).toBeCloseTo(initial[1], RESET_POSITION_DECIMALS);
        expect(pose.position[2]).toBeCloseTo(initial[2], RESET_POSITION_DECIMALS);
      }

      // 리셋 직후 1초 재시뮬 — 속도가 0으로 초기화됐다면 다시 자유낙하해 정착 경로를 밟는다
      stepSeconds(world, spec, 1);
      for (const entity of dynamicEntitiesOf(spec)) {
        const bodyId = soleBodyOf(world, entity.id);
        expect(world.getPose(bodyId).position[1]).toBeGreaterThan(0);
      }
    } finally {
      world.free();
    }
  });

  it('dispose()는 물리 바디와 시각 노드를 전부 해제한다', async () => {
    const spec = loadValidatedSpec();
    const world = new RapierWorld(spec.gravity, spec.timestepHz);
    try {
      const sync = new RenderSync(world);
      const renderApi = makeRenderApi();
      const handle = await new SceneLoader(world, renderApi, sync).build(spec);
      const createdNodeCount = renderApi.counts.addPrimitive + renderApi.counts.addGround;

      handle.dispose();

      for (const id of handle.entityIds) {
        expect(world.bodiesOfEntity(id)).toEqual([]);
      }
      expect(renderApi.counts.remove).toBe(createdNodeCount);
      expect(world.step()).toEqual([]); // 빈 월드 스텝은 무해
    } finally {
      world.free();
    }
  });

});

// ── 3. 샘플 씬 검증: arm-and-boxes (Phase 3) ────────────────────────

describe('arm-and-boxes.scene.json', () => {
  it('validateScene을 통과하고 robot 1대 + 동적 박스 2개를 갖춘다', () => {
    const result = validateScene(armAndBoxesSceneJson);
    if (!result.ok) throw new Error(result.errors.join('\n'));
    const spec = result.value;

    expect(spec.name).toBe('arm-and-boxes');
    const robotEntities = spec.entities.filter(isRobotSpec);
    expect(robotEntities).toHaveLength(1);
    const arm = robotEntities[0];
    if (!arm) throw new Error('robot 엔티티가 없습니다');
    expect(arm.id).toBe('arm');
    // Phase 5: 시퀀스가 선언된 씬 — controller 'sequence', home에 joint1 포함
    // (arm-touch-box.sequence.json의 joint1 참조가 검증을 통과하기 위한 관절 명세)
    expect(arm.controller).toBe('sequence');
    expect(arm.linkColliders).toBe('fromVisual');
    expect(arm.selfCollision).toBe(false);
    expect(arm.home).toEqual({ joint1: 0.0, joint2: -0.6, joint3: 1.2, joint5: -0.6 });

    const boxes = dynamicEntitiesOf(spec);
    expect(boxes.map((e) => e.id)).toEqual(['box_a', 'box_b']);
    for (const box of boxes) {
      for (const collider of box.physics?.colliders ?? []) {
        expect(collider.group).toBe('OBJECT');
        expect(collider.collidesWith).toContain('ROBOT');
        expect(collider.emitEvents).toBe(true);
      }
    }
    expect(spec.environment?.ground).toBe(true);
  });
});

// ── 4. robot 브랜치 (가짜 RobotHandle + 기록형 PhysicsWorld) ─────────
// 물리 없이 scene-loader의 로봇 조립 계약만 검증한다: 바디/collider 생성 스펙,
// RobotRegistry 등록, home 적용, reset의 재-home. (실제 URDF/Rapier 결합은
// scripts/gate-browser.mjs --expect=arm 브라우저 게이트가 검증한다.)

const FAKE_LINK_COLLIDER_DEF: LinkColliderDef = {
  shape: { kind: 'box', halfExtents: [0.05, 0.05, 0.05] },
  offset: { position: [0, 0.01, 0] },
};

/** 가짜 FK 로봇: 2관절(j1 revolute, j2 prismatic) · 2링크(L1, L2, 각 1 collider).
 *  L1.position = [j1, L1_BASE_Y_M, 0], L2.position = [j1, L2_BASE_Y_M + j2, 0]. */
class FakeRobotHandle implements RobotHandle {
  readonly joints: readonly JointInfo[] = [
    { name: 'j1', type: 'revolute', limits: [...J1_LIMITS_RAD], initial: 0 },
    { name: 'j2', type: 'prismatic', limits: [...J2_LIMITS_M], initial: 0 },
  ];
  readonly linkColliders: ReadonlyMap<string, readonly LinkColliderDef[]> = new Map([
    ['L1', [FAKE_LINK_COLLIDER_DEF]],
    ['L2', [FAKE_LINK_COLLIDER_DEF]],
  ]);

  /** setRootTransform 호출 기록 (깊은 복사) */
  readonly rootTransforms: Pose[] = [];
  disposeCount = 0;

  private readonly values = new Map<string, number>([
    ['j1', 0],
    ['j2', 0],
  ]);

  setJointValues(values: Readonly<Record<string, number>>): void {
    for (const [name, value] of Object.entries(values)) {
      if (this.values.has(name)) this.values.set(name, value);
    }
  }

  readLinkPoses(): ReadonlyMap<string, Pose> {
    const j1 = this.values.get('j1') ?? 0;
    const j2 = this.values.get('j2') ?? 0;
    return new Map<string, Pose>([
      ['L1', { position: [j1, L1_BASE_Y_M, 0], rotation: [0, 0, 0, 1] }],
      ['L2', { position: [j1, L2_BASE_Y_M + j2, 0], rotation: [0, 0, 0, 1] }],
    ]);
  }

  setRootTransform(pose: Pose): void {
    this.rootTransforms.push({ position: [...pose.position], rotation: [...pose.rotation] });
  }

  dispose(): void {
    this.disposeCount += 1;
  }
}

interface CreatedBodyRecord {
  bodyId: BodyId;
  entityId: EntityId;
  init: PhysicsBodyInit;
}

interface CreatedColliderRecord {
  colliderId: ColliderId;
  bodyId: BodyId;
  entityId: EntityId;
  spec: Parameters<PhysicsWorld['createCollider']>[1];
}

/** createBody/createCollider/setKinematicPose/teleport 호출을 기록하는 PhysicsWorld 목 */
class RecordingWorld implements PhysicsWorld {
  readonly fixedDtSec = 1 / TEST_TIMESTEP_HZ;
  readonly bodies: CreatedBodyRecord[] = [];
  readonly colliders: CreatedColliderRecord[] = [];
  readonly kinematicCalls: { bodyId: BodyId; pose: Pose }[] = [];
  readonly teleportCalls: { bodyId: BodyId; pose: Pose }[] = [];
  readonly removedEntityIds: EntityId[] = [];
  /** setSelfContactEnabled 호출 기록 — 로봇 self-collision 배선 검증용 */
  readonly selfContactCalls: { entityId: EntityId; enabled: boolean }[] = [];
  private nextHandle = 1;

  setSelfContactEnabled(entityId: EntityId, enabled: boolean): void {
    this.selfContactCalls.push({ entityId, enabled });
  }

  createBody(entityId: EntityId, init: PhysicsBodyInit): BodyId {
    const bodyId = this.nextHandle;
    this.nextHandle += 1;
    this.bodies.push({
      bodyId,
      entityId,
      init: {
        ...init,
        position: [...init.position],
        rotation: init.rotation ? [...init.rotation] : undefined,
      },
    });
    return bodyId;
  }

  createCollider(
    bodyId: BodyId,
    spec: Parameters<PhysicsWorld['createCollider']>[1],
    entityId: EntityId,
  ): ColliderId {
    const colliderId = this.nextHandle;
    this.nextHandle += 1;
    this.colliders.push({ colliderId, bodyId, entityId, spec });
    return colliderId;
  }

  removeEntity(entityId: EntityId): void {
    this.removedEntityIds.push(entityId);
  }

  /** 되감기 전용 — 이 테스트들은 되감기를 호출하지 않으므로 no-op이면 충분하다 */
  clearContactState(): void {}

  setKinematicPose(bodyId: BodyId, pose: Pose): void {
    this.kinematicCalls.push({
      bodyId,
      pose: { position: [...pose.position], rotation: [...pose.rotation] },
    });
  }

  getPose(): Pose {
    return { position: [0, 0, 0], rotation: [0, 0, 0, 1] };
  }

  teleport(bodyId: BodyId, pose: Pose): void {
    this.teleportCalls.push({
      bodyId,
      pose: { position: [...pose.position], rotation: [...pose.rotation] },
    });
  }

  step(): ContactEvent[] {
    return [];
  }

  entityOfCollider(): EntityId | undefined {
    return undefined;
  }

  bodiesOfEntity(entityId: EntityId): readonly BodyId[] {
    return this.bodies.filter((b) => b.entityId === entityId).map((b) => b.bodyId);
  }

  // 컨베이어 표면 구동 계약 (core/types.ts) — 이 목은 접촉/속도를 다루지 않는다
  dynamicBodiesTouching(): readonly BodyId[] {
    return [];
  }
  dynamicBodies(): readonly BodyId[] {
    return [];
  }
  getLinearVelocity(): Vec3 {
    return [0, 0, 0];
  }
  setLinearVelocity(): void {}

  clear(): void {}
  free(): void {}
}

function makeRobotScene(overrides: Partial<RobotSpec> = {}): SceneSpec {
  const robotSpec: RobotSpec = {
    id: ROBOT_ID,
    type: 'robot',
    transform: { position: [0.1, 0, -0.2] },
    visual: { kind: 'urdf', ref: FAKE_URDF_PATH },
    urdf: FAKE_URDF_PATH,
    controller: 'manual',
    linkColliders: 'fromVisual',
    selfCollision: false,
    home: { j1: HOME_J1_RAD },
    ...overrides,
  };
  return {
    name: 'robot-branch-test',
    version: 1,
    gravity: [0, -9.81, 0],
    timestepHz: TEST_TIMESTEP_HZ,
    entities: [robotSpec],
  };
}

async function buildRobotScene(overrides: Partial<RobotSpec> = {}): Promise<{
  world: RecordingWorld;
  fake: FakeRobotHandle;
  loadRobot: ReturnType<typeof vi.fn>;
  handle: Awaited<ReturnType<SceneLoader['build']>>;
}> {
  const world = new RecordingWorld();
  const fake = new FakeRobotHandle();
  const loadRobot = vi.fn((_request: RobotLoadRequest) => Promise.resolve<RobotHandle>(fake));
  const renderApi = makeRenderApi(loadRobot);
  const handle = await new SceneLoader(world, renderApi, new RenderSync(world)).build(
    makeRobotScene(overrides),
  );
  return { world, fake, loadRobot, handle };
}

describe('SceneLoader — robot 브랜치', () => {
  it('collider 보유 링크마다 kinematicPosition 바디를 "home FK" pose로 만들고 ROBOT 그룹 + emitEvents collider를 붙인다', async () => {
    const { world, fake, loadRobot, handle } = await buildRobotScene();

    // URDF 로드 요청이 RobotSpec에서 유도됐다
    expect(loadRobot).toHaveBeenCalledTimes(1);
    expect(loadRobot).toHaveBeenCalledWith({ urdfPath: FAKE_URDF_PATH, packages: undefined });

    // 엔티티 트랜스폼 → 로봇 루트 pose
    expect(fake.rootTransforms).toEqual([
      { position: [0.1, 0, -0.2], rotation: [0, 0, 0, 1] },
    ]);

    // 링크당 kinematicPosition 바디 1개 — 생성 pose는 "home FK" pose다.
    // (리뷰 회귀: URDF 초기 자세로 만들면 첫 스텝에 초기→home 스윕이 생겨
    //  가짜 kinematic 속도가 잡힌다 — fresh load 스텝 전 상태 = home FK 이어야 한다)
    expect(world.bodies).toHaveLength(2);
    for (const body of world.bodies) {
      expect(body.entityId).toBe(ROBOT_ID);
      expect(body.init.bodyType).toBe('kinematicPosition');
    }
    expect(world.bodies[0]?.init.position).toEqual([HOME_J1_RAD, L1_BASE_Y_M, 0]);
    expect(world.bodies[1]?.init.position).toEqual([HOME_J1_RAD, L2_BASE_Y_M, 0]);

    // collider 규약 (CLAUDE.md §5): ROBOT 소속. 필터에는 **항상** ROBOT이 포함된다 —
    // 다른 로봇과의 충돌을 위해서다. 자기 링크 억제는 그룹이 아니라 엔티티 단위
    // (world.setSelfContactEnabled)로 처리한다.
    expect(world.colliders).toHaveLength(2);
    for (const collider of world.colliders) {
      expect(collider.entityId).toBe(ROBOT_ID);
      expect(collider.spec.group).toBe('ROBOT');
      expect(collider.spec.collidesWith).toEqual(['ENV', 'OBJECT', 'ROBOT']);
      expect(collider.spec.emitEvents).toBe(true);
      expect(collider.spec.friction).toBe(EXPECTED_ROBOT_LINK_FRICTION);
      expect(collider.spec.offset).toEqual(FAKE_LINK_COLLIDER_DEF.offset);
      expect(collider.spec.shape).toEqual(FAKE_LINK_COLLIDER_DEF.shape);
    }

    // 레지스트리 등록 + home 적용 (관절 상태의 진실은 core RobotBinding)
    expect(handle.robots.ids()).toEqual([ROBOT_ID]);
    expect(handle.robots.get(ROBOT_ID).readJoints()).toEqual({ j1: HOME_J1_RAD, j2: 0 });

    // build 마지막의 tick() — home FK pose가 링크 바디의 kinematic 목표로 push됐고,
    // 목표가 생성 pose와 동일하므로 첫 물리 스텝의 이동(스윕)이 0이다.
    expect(world.kinematicCalls).toHaveLength(2);
    expect(world.kinematicCalls[0]?.pose.position).toEqual([HOME_J1_RAD, L1_BASE_Y_M, 0]);
    expect(world.kinematicCalls[1]?.pose.position).toEqual([HOME_J1_RAD, L2_BASE_Y_M, 0]);
    expect(world.kinematicCalls.map((c) => c.pose.position)).toEqual(
      world.bodies.map((b) => b.init.position),
    );
  });

  it('reset()은 로봇 관절을 home으로 되돌리고, 링크 바디를 home FK pose로 teleport + kinematic 목표 재-push한다', async () => {
    const { world, handle } = await buildRobotScene();
    const binding = handle.robots.get(ROBOT_ID);

    binding.setJoint('j1', -0.25);
    binding.tick();
    expect(binding.readJoints()).toEqual({ j1: -0.25, j2: 0 });
    const callCountBefore = world.kinematicCalls.length;
    expect(world.teleportCalls).toHaveLength(0); // 재생 경로에서는 teleport 미사용

    handle.reset();

    expect(binding.readJoints()).toEqual({ j1: HOME_J1_RAD, j2: 0 });
    // 리뷰 회귀: tick()만으로는 다음 스텝까지 물리 pose가 리셋 전 상태로 남는다 —
    // 링크 바디가 home FK pose로 "즉시" teleport 되어야 리셋 직후(스텝 전) 상태가
    // fresh load와 동일하다 (결정론적 재생, SIMULATION.md §6).
    expect(world.teleportCalls.map((c) => c.pose.position)).toEqual([
      [HOME_J1_RAD, L1_BASE_Y_M, 0],
      [HOME_J1_RAD, L2_BASE_Y_M, 0],
    ]);
    // teleport 대상 = 링크 바디, 그리고 fresh load의 바디 생성 pose와 정확히 일치
    expect(world.teleportCalls.map((c) => c.bodyId)).toEqual(world.bodies.map((b) => b.bodyId));
    expect(world.teleportCalls.map((c) => c.pose.position)).toEqual(
      world.bodies.map((b) => b.init.position),
    );
    // 다음 스텝 kinematic 목표도 home FK로 재지정 — teleport pose와 동일(스윕 속도 0)
    const resetCalls = world.kinematicCalls.slice(callCountBefore);
    expect(resetCalls.map((c) => c.pose.position)).toEqual([
      [HOME_J1_RAD, L1_BASE_Y_M, 0],
      [HOME_J1_RAD, L2_BASE_Y_M, 0],
    ]);
  });

  it("linkColliders 'none'은 물리 바디를 전혀 만들지 않지만 로봇은 등록된다", async () => {
    const { world, handle } = await buildRobotScene({ linkColliders: 'none' });
    expect(world.bodies).toHaveLength(0);
    expect(world.colliders).toHaveLength(0);
    expect(world.kinematicCalls).toHaveLength(0); // push 대상 링크 바디 없음
    expect(handle.robots.has(ROBOT_ID)).toBe(true);
  });

  it('selfCollision: true면 해당 엔티티의 자기 접촉 발행이 켜진다 (그룹 필터는 불변)', async () => {
    const { world } = await buildRobotScene({ selfCollision: true });
    expect(world.colliders.length).toBeGreaterThan(0);
    // 그룹 필터는 selfCollision과 무관하게 항상 ROBOT을 포함한다(다른 로봇과의 충돌)
    for (const collider of world.colliders) {
      expect(collider.spec.collidesWith).toEqual(['ENV', 'OBJECT', 'ROBOT']);
    }
    // 자기 링크 접촉 여부는 엔티티 단위 스위치로 전달된다
    expect(world.selfContactCalls).toContainEqual({ entityId: ROBOT_ID, enabled: true });
  });

  it('selfCollision 미지정(기본)이면 자기 접촉 발행이 꺼진 채 배선된다', async () => {
    const { world } = await buildRobotScene();
    expect(world.selfContactCalls).toContainEqual({ entityId: ROBOT_ID, enabled: false });
  });

  it('dispose()는 로봇 물리 바디를 제거하고 렌더 핸들을 해제한다', async () => {
    const { world, fake, handle } = await buildRobotScene();
    handle.dispose();
    expect(world.removedEntityIds).toContain(ROBOT_ID);
    expect(fake.disposeCount).toBe(1);
  });

  // 리뷰 회귀: gripper.joints는 validateSequence(URDF를 못 봄)·checkJointNames(gripper
  // 미다룸)의 사각지대였다 — 존재하지 않는 관절이면 재생 중 resolveJoint 예외가
  // Engine rAF 루프를 죽였다. build 시점(URDF 관절 목록을 아는 최초 지점)에 거부한다.
  it('gripper.joints가 URDF에 없는 관절을 참조하면 build가 실패하고 자원을 정리한다', async () => {
    const world = new RecordingWorld();
    const fake = new FakeRobotHandle();
    const renderApi = makeRenderApi(() => Promise.resolve<RobotHandle>(fake));
    const loader = new SceneLoader(world, renderApi, new RenderSync(world));

    await expect(
      loader.build(
        makeRobotScene({ gripper: { joints: ['ghost_joint'], open: 0.03, close: 0 } }),
      ),
    ).rejects.toThrow(/gripper\.joints.*'ghost_joint'/);

    // 부분 생성 자원이 남지 않는다 (build의 teardown 계약)
    expect(world.removedEntityIds).toContain(ROBOT_ID);
    expect(fake.disposeCount).toBe(1);
  });

  it('gripper.joints는 URDF 원명·jointMap 논리명 모두 허용한다', async () => {
    const { handle } = await buildRobotScene({
      jointMap: { grip: 'j2' },
      gripper: { joints: ['grip', 'j2'], open: 0.05, close: 0 },
    });
    expect(handle.robots.has(ROBOT_ID)).toBe(true);
  });
});
