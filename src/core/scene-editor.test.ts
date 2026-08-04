// core/scene-editor.test.ts — SceneEditorImpl 통합 테스트 (Phase 7 Scene Builder 코어)
//
// 실제 Rapier(WASM base64 내장 — Node에서 그대로 동작) + 가짜 RenderSceneApi 조합으로
// scene-loader.test.ts와 같은 헤드리스 구성을 쓴다. 모든 물리 API 사용 전
// initPhysics()를 await한다 (CLAUDE.md §2.7).
//
// 검증 계약 (scene-edit-types.ts):
// - 모든 편집 연산은 [spec 갱신 + 물리/시각 동기 반영 + onChange 통지]가 원자적이다.
// - 실패한 연산은 spec을 바꾸지 않고 물리 자원을 새지 않는다 (롤백).
// - serialize()는 항상 validateScene을 통과하는 canonical spec을 반환한다.

import { beforeAll, describe, expect, it } from 'vitest';
import { validateScene } from '../schema/validate';
import type { EntitySpec, RobotSpec, SceneSpec, Vec3 } from '../schema/types';
import type { ContactEvent, Pose } from './types';
import type { JointInfo, LinkColliderDef } from './robot-types';
import type { MeshAssetResolver, SceneEditEvent } from './scene-edit-types';
import { initPhysics, RapierWorld } from './world';
import { RenderSync } from './sync';
import { SceneLoader } from './scene-loader';
import type { RenderSceneApi, RobotHandle, SceneHandle, VisualNode } from './scene-loader';
import { SceneEditorImpl } from './scene-editor';

// ── 테스트 상수 (매직넘버 금지 — CLAUDE.md §4) ──────────────────────

const TIMESTEP_HZ = 240;
const BOX_HALF_M = 0.05;
const BIG_BOX_HALF_M = 0.1;
const DROP_Y_M = 0.5;
/** 정착 높이 허용 오차 (솔버 침투 여유 포함) */
const REST_TOLERANCE_M = 0.01;
const POSE_DECIMALS = 6;
const SETTLE_SEC = 2;

const CUBE_ASSET_REF = 'asset://cube';
const GHOST_ASSET_REF = 'asset://ghost';

// robot 픽스처 상수 (scene-loader.test.ts의 FakeRobotHandle 계약과 동일)
const FAKE_URDF_PATH = 'assets/robots/fake/fake.urdf';
const L1_BASE_Y_M = 0.1;
const L2_BASE_Y_M = 0.2;
const J1_LIMITS_RAD: [number, number] = [-1, 1];
const J2_LIMITS_M: [number, number] = [0, 0.05];
const HOME_J1_RAD = 0.5;

// ── 헬퍼: 시각 스텁 + 기록형 RenderSceneApi ─────────────────────────

/** sync.apply가 마지막으로 쓴 position을 기록하는 시각 스텁 (stale-prev 회귀 검증용) */
interface RecordingStubNode {
  lastPosition: [number, number, number] | null;
  position: { set(x: number, y: number, z: number): void };
  quaternion: { set(x: number, y: number, z: number, w: number): void };
}

function stubNode(): VisualNode {
  const stub: RecordingStubNode = {
    lastPosition: null,
    position: {
      set: (x: number, y: number, z: number): void => {
        stub.lastPosition = [x, y, z];
      },
    },
    quaternion: { set: () => undefined },
  };
  return stub as unknown as VisualNode;
}

/** 스텁 노드의 기록 뷰 (VisualNode opaque 타입에서 되돌리는 테스트 전용 캐스트) */
function recordingView(node: VisualNode | undefined): RecordingStubNode {
  if (!node) throw new Error('시각 노드가 없습니다 (픽스처 오류)');
  return node as unknown as RecordingStubNode;
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
      (() => Promise.reject(new Error('이 테스트에는 robot 엔티티가 없어야 합니다'))),
  };
}

// ── 헬퍼: 씬 스펙 픽스처 ────────────────────────────────────────────

function boxEntity(id: string, position: Vec3, halfM: number = BOX_HALF_M): EntitySpec {
  return {
    id,
    type: 'object',
    transform: { position },
    visual: {
      kind: 'primitive',
      primitive: { kind: 'box', halfExtents: [halfM, halfM, halfM] },
      color: '#c0392b',
    },
    physics: {
      bodyType: 'dynamic',
      colliders: [
        {
          shape: { kind: 'box', halfExtents: [halfM, halfM, halfM] },
          friction: 0.6,
          group: 'OBJECT',
          collidesWith: ['ENV', 'ROBOT', 'OBJECT'],
          emitEvents: true,
        },
      ],
    },
  };
}

function robotEntity(id: string): RobotSpec {
  return {
    id,
    type: 'robot',
    transform: { position: [0.1, 0, -0.2] },
    visual: { kind: 'urdf', ref: FAKE_URDF_PATH },
    urdf: FAKE_URDF_PATH,
    controller: 'manual',
    linkColliders: 'fromVisual',
    selfCollision: false,
    home: { j1: HOME_J1_RAD },
  };
}

function baseScene(): SceneSpec {
  return {
    name: 'editor-test',
    version: 1,
    gravity: [0, -9.81, 0],
    timestepHz: TIMESTEP_HZ,
    environment: { ground: true },
    entities: [boxEntity('box_a', [0, DROP_Y_M, 0])],
  };
}

// ── 헬퍼: FakeRobotHandle (scene-loader.test.ts와 동일 계약의 최소본) ──

const FAKE_LINK_COLLIDER_DEF: LinkColliderDef = {
  shape: { kind: 'box', halfExtents: [0.05, 0.05, 0.05] },
  offset: { position: [0, 0.01, 0] },
};

/** 가짜 FK 로봇: L1.position = [j1, 0.1, 0], L2.position = [j1, 0.2 + j2, 0] */
class FakeRobotHandle implements RobotHandle {
  readonly joints: readonly JointInfo[] = [
    { name: 'j1', type: 'revolute', limits: [...J1_LIMITS_RAD], initial: 0 },
    { name: 'j2', type: 'prismatic', limits: [...J2_LIMITS_M], initial: 0 },
  ];
  readonly linkColliders: ReadonlyMap<string, readonly LinkColliderDef[]> = new Map([
    ['L1', [FAKE_LINK_COLLIDER_DEF]],
    ['L2', [FAKE_LINK_COLLIDER_DEF]],
  ]);

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

// ── 헬퍼: 픽스처 조립 ───────────────────────────────────────────────

const cubeResolver: MeshAssetResolver = {
  getPoints: (ref) => {
    if (ref !== CUBE_ASSET_REF) return undefined;
    const points: number[] = [];
    for (const x of [-BOX_HALF_M, BOX_HALF_M]) {
      for (const y of [-BOX_HALF_M, BOX_HALF_M]) {
        for (const z of [-BOX_HALF_M, BOX_HALF_M]) points.push(x, y, z);
      }
    }
    return new Float32Array(points);
  },
  getIndices: () => undefined,
};

interface Fixture {
  world: RapierWorld;
  sync: RenderSync;
  renderApi: CountingRenderApi;
  handle: SceneHandle;
  editor: SceneEditorImpl;
  /** onChange 수신 기록 */
  events: SceneEditEvent[];
  /** loadRobot이 만든 가짜 핸들들 (호출 순서) */
  fakes: FakeRobotHandle[];
}

async function makeFixture(
  spec: SceneSpec = baseScene(),
  opts: { withRobots?: boolean; resolver?: MeshAssetResolver } = {},
): Promise<Fixture> {
  const validated = validateScene(spec);
  if (!validated.ok) throw new Error(`픽스처 씬 검증 실패:\n${validated.errors.join('\n')}`);

  const world = new RapierWorld(
    validated.value.gravity,
    validated.value.timestepHz,
    opts.resolver,
  );
  const fakes: FakeRobotHandle[] = [];
  const renderApi = makeRenderApi(
    opts.withRobots
      ? () => {
          const fake = new FakeRobotHandle();
          fakes.push(fake);
          return Promise.resolve<RobotHandle>(fake);
        }
      : undefined,
  );
  const sync = new RenderSync(world);
  const handle = await new SceneLoader(world, renderApi, sync).build(validated.value);
  const editor = new SceneEditorImpl({
    spec: validated.value,
    world,
    sync,
    renderApi,
    robots: handle.robots,
    conveyors: handle.conveyors,
    builtEntities: handle.builtEntities,
  });
  const events: SceneEditEvent[] = [];
  editor.onChange((e) => events.push(e));
  return { world, sync, renderApi, handle, editor, events, fakes };
}

function stepSeconds(world: RapierWorld, seconds: number): ContactEvent[] {
  const totalSteps = Math.round(seconds * TIMESTEP_HZ);
  const events: ContactEvent[] = [];
  for (let i = 0; i < totalSteps; i += 1) events.push(...world.step());
  return events;
}

/** 엔티티의 유일한 바디 id (noUncheckedIndexedAccess 대응 명시적 가드) */
function soleBodyOf(world: RapierWorld, entityId: string): number {
  const bodies = world.bodiesOfEntity(entityId);
  expect(bodies).toHaveLength(1);
  const bodyId = bodies[0];
  if (bodyId === undefined) throw new Error(`엔티티 '${entityId}'의 바디가 없습니다`);
  return bodyId;
}

function entityById(spec: Readonly<SceneSpec>, id: string): EntitySpec {
  const entity = spec.entities.find((e) => e.id === id);
  if (!entity) throw new Error(`spec에 엔티티 '${id}'이(가) 없습니다`);
  return entity;
}

beforeAll(async () => {
  await initPhysics();
});

// ── 1. addEntity / removeEntity ─────────────────────────────────────

describe('SceneEditorImpl — addEntity / removeEntity', () => {
  it('addEntity는 spec·물리·시각을 함께 갱신하고 add 이벤트를 낸다 (새 바디는 실제 낙하·정착)', async () => {
    const f = await makeFixture();
    try {
      await f.editor.addEntity(boxEntity('box_b', [0.3, DROP_Y_M, 0]));

      expect(f.editor.spec.entities.map((e) => e.id)).toEqual(['box_a', 'box_b']);
      expect(f.renderApi.counts.addPrimitive).toBe(2);
      expect(f.events).toEqual([{ kind: 'add', entityId: 'box_b' }]);

      const serialized = f.editor.serialize();
      expect(validateScene(serialized).ok).toBe(true);

      // 새 엔티티가 물리에 실재한다 — 낙하해 바닥 위 half-extent 높이에 정착
      const bodyId = soleBodyOf(f.world, 'box_b');
      stepSeconds(f.world, SETTLE_SEC);
      const restY = f.world.getPose(bodyId).position[1];
      expect(Math.abs(restY - BOX_HALF_M)).toBeLessThan(REST_TOLERANCE_M);
    } finally {
      f.world.free();
    }
  });

  it('중복 id addEntity는 한국어 오류로 거부하고 아무것도 바꾸지 않는다', async () => {
    const f = await makeFixture();
    try {
      const before = f.editor.serialize();
      await expect(f.editor.addEntity(boxEntity('box_a', [1, 1, 1]))).rejects.toThrow(
        /이미 존재합니다/,
      );
      expect(f.editor.serialize()).toEqual(before);
      expect(f.renderApi.counts.addPrimitive).toBe(1); // 초기 box_a 몫만
      expect(f.events).toEqual([]);
    } finally {
      f.world.free();
    }
  });

  it('스키마 위반 addEntity는 검증 오류로 롤백된다 (spec 불변·바디 누수 없음)', async () => {
    const f = await makeFixture();
    try {
      const invalid = boxEntity('bad_box', [0, 1, 0]);
      // 교차 규칙 위반: emitEvents=true + collidesWith [] (validate.ts)
      invalid.physics!.colliders[0]!.collidesWith = [];
      const before = f.editor.serialize();

      await expect(f.editor.addEntity(invalid)).rejects.toThrow(/씬 검증 실패/);

      expect(f.editor.serialize()).toEqual(before);
      expect(f.world.bodiesOfEntity('bad_box')).toEqual([]);
      expect(f.renderApi.counts.addPrimitive).toBe(1);
      expect(f.events).toEqual([]);
    } finally {
      f.world.free();
    }
  });

  it('빌드 실패 addEntity(미등록 에셋 ref convexHull)는 부분 자원을 정리하고 롤백된다', async () => {
    const f = await makeFixture(baseScene(), { resolver: cubeResolver });
    try {
      const entity = boxEntity('hull_bad', [0, 1, 0]);
      entity.physics!.colliders[0]!.shape = { kind: 'convexHull', ref: GHOST_ASSET_REF };
      const before = f.editor.serialize();

      await expect(f.editor.addEntity(entity)).rejects.toThrow(/해석할 수 없습니다/);

      expect(f.editor.serialize()).toEqual(before);
      expect(f.world.bodiesOfEntity('hull_bad')).toEqual([]); // 바디 누수 없음
      // 시각 노드는 생성됐다가 빌드 실패 정리 경로에서 제거된다
      expect(f.renderApi.counts.addPrimitive).toBe(2);
      expect(f.renderApi.counts.remove).toBe(1);
      expect(f.events).toEqual([]);
    } finally {
      f.world.free();
    }
  });

  it('resolver가 주입된 월드에서 convexHull collider 엔티티를 추가하면 실제로 충돌한다', async () => {
    const f = await makeFixture(baseScene(), { resolver: cubeResolver });
    try {
      const entity = boxEntity('hull', [0.5, DROP_Y_M, 0]);
      entity.physics!.colliders[0]!.shape = { kind: 'convexHull', ref: CUBE_ASSET_REF };
      await f.editor.addEntity(entity);

      const bodyId = soleBodyOf(f.world, 'hull');
      const contactEvents = stepSeconds(f.world, SETTLE_SEC);

      // 정육면체 point cloud hull = box와 동일 — 바닥 위 정착 + 접촉 이벤트 발생
      const restY = f.world.getPose(bodyId).position[1];
      expect(Math.abs(restY - BOX_HALF_M)).toBeLessThan(REST_TOLERANCE_M);
      expect(
        contactEvents.some(
          (e) => e.phase === 'start' && (e.a === 'hull' || e.b === 'hull'),
        ),
      ).toBe(true);
    } finally {
      f.world.free();
    }
  });

  it('removeEntity는 물리 바디·시각 노드를 해제하고 spec에서 제거한다', async () => {
    const f = await makeFixture();
    try {
      f.editor.removeEntity('box_a');

      expect(f.world.bodiesOfEntity('box_a')).toEqual([]);
      expect(f.renderApi.counts.remove).toBe(1);
      expect(f.editor.spec.entities).toEqual([]);
      expect(f.events).toEqual([{ kind: 'remove', entityId: 'box_a' }]);
      expect(validateScene(f.editor.serialize()).ok).toBe(true);
      expect(f.world.step()).toEqual([]); // 남은 월드 스텝은 무해
    } finally {
      f.world.free();
    }
  });

  it('존재하지 않는 엔티티 removeEntity는 한국어 오류를 던진다', async () => {
    const f = await makeFixture();
    try {
      expect(() => f.editor.removeEntity('ghost')).toThrow(/없습니다/);
    } finally {
      f.world.free();
    }
  });

  it('편집(add/remove) 후 SceneHandle.dispose가 현재 상태를 정확히 해제한다 (builtEntities 공유)', async () => {
    const f = await makeFixture();
    try {
      await f.editor.addEntity(boxEntity('box_b', [0.3, DROP_Y_M, 0]));
      f.editor.removeEntity('box_a');

      f.handle.dispose();

      expect(f.world.bodiesOfEntity('box_b')).toEqual([]);
      expect(f.world.bodiesOfEntity('box_a')).toEqual([]);
      // 제거 합계 = 생성된 모든 노드 (box_a 편집 제거 1 + dispose의 ground/box_b)
      expect(f.renderApi.counts.remove).toBe(
        f.renderApi.counts.addPrimitive + f.renderApi.counts.addGround,
      );
    } finally {
      f.world.free();
    }
  });
});

// ── 2. updateTransform ──────────────────────────────────────────────

describe('SceneEditorImpl — updateTransform', () => {
  it('spec 갱신 + teleport(getPose로 검증) + 동적 바디 속도 0 초기화', async () => {
    const f = await makeFixture();
    try {
      stepSeconds(f.world, 0.2); // 낙하 중 — 속도 보유 상태에서 편집
      const target: Vec3 = [0.3, 0.7, -0.2];
      f.editor.updateTransform('box_a', { position: target });

      // 물리 pose가 즉시 새 배치와 일치 (teleport)
      const bodyId = soleBodyOf(f.world, 'box_a');
      const pose = f.world.getPose(bodyId);
      expect(pose.position[0]).toBeCloseTo(target[0], POSE_DECIMALS);
      expect(pose.position[1]).toBeCloseTo(target[1], POSE_DECIMALS);
      expect(pose.position[2]).toBeCloseTo(target[2], POSE_DECIMALS);

      // spec도 동일 값
      expect(entityById(f.editor.spec, 'box_a').transform.position).toEqual(target);
      expect(f.events).toEqual([{ kind: 'transform', entityId: 'box_a' }]);

      // 속도 0 초기화 증명: 1 tick 후 y가 거의 그대로다 (낙하 속도가 유지됐다면
      // ~0.2s 자유낙하 속도 1.96m/s × dt ≈ 8mm 하강해 0.695를 밑돈다)
      f.world.step();
      expect(f.world.getPose(bodyId).position[1]).toBeGreaterThan(0.695);
    } finally {
      f.world.free();
    }
  });

  it('reset()은 편집된 배치로 되돌아간다 (initialPose 갱신)', async () => {
    const f = await makeFixture();
    try {
      const target: Vec3 = [0.2, 0.9, 0.1];
      f.editor.updateTransform('box_a', { position: target });
      stepSeconds(f.world, 1); // 낙하·정착

      f.handle.reset();

      const pose = f.world.getPose(soleBodyOf(f.world, 'box_a'));
      expect(pose.position[0]).toBeCloseTo(target[0], POSE_DECIMALS);
      expect(pose.position[1]).toBeCloseTo(target[1], POSE_DECIMALS);
      expect(pose.position[2]).toBeCloseTo(target[2], POSE_DECIMALS);
    } finally {
      f.world.free();
    }
  });

  it('비유한 좌표는 검증 오류로 롤백된다', async () => {
    const f = await makeFixture();
    try {
      const before = f.editor.serialize();
      expect(() =>
        f.editor.updateTransform('box_a', { position: [Number.NaN, 0, 0] }),
      ).toThrow(/씬 검증 실패/);
      expect(f.editor.serialize()).toEqual(before);
      expect(f.events).toEqual([]);
    } finally {
      f.world.free();
    }
  });

  it('teleport 후 sync prev 스냅샷을 갱신한다 — paused 프레임의 apply(alpha=0)가 편집된 pose를 그린다', async () => {
    // f32 정확 표현값 사용 (Rapier 내부 f32 저장 — toEqual 정확 비교 가능)
    const target: Vec3 = [0.25, 0.75, -0.5];
    const f = await makeFixture();
    try {
      // 물리 tick의 commit()이 한 번 돈 상태를 재현 — prev = 편집 전 pose
      f.sync.commit();
      f.editor.updateTransform('box_a', { position: target });
      // 일시정지 프레임: alpha=0이면 apply는 prev를 그대로 그린다 — prev가 갱신되지
      // 않았다면 편집 전 pose로 시각이 되돌아간다 (stale-prev 회귀)
      f.sync.apply(0);
      const node = recordingView(f.handle.builtEntities.get('box_a')?.node);
      expect(node.lastPosition).toEqual(target);
    } finally {
      f.world.free();
    }
  });

  /**
   * ★ 회귀: 편집 결과가 **렌더 프레임을 기다리지 않고** 시각 노드에 반영돼야 한다.
   *
   * commit()은 prev만 갱신하고 Object3D는 렌더 루프의 apply()가 쓴다. 그런데 편집 중에는
   * 엔진이 일시정지돼 있어 다음 프레임이 언제 올지 모르고, 기즈모 앵커와 방향키 이동은
   * **시각 노드의 월드 행렬**을 기준점으로 삼는다 — 노드가 stale이면 다음 편집이 편집 전
   * pose에서 다시 출발한다.
   *
   * 실측 증상: 일시정지 상태에서 →(5cm) 직후 Shift+→(1cm)를 빠르게 누르면 순 이동이
   * 1cm가 아니라 4cm(= 5 − 1)로 나왔다. 프레임이 사이에 끼면 1cm — 같은 조작이 프레임
   * 타이밍에 따라 다른 결과를 냈다.
   */
  it('편집 직후 시각 노드가 즉시 새 pose를 갖는다 — 다음 편집의 기준점이 stale하지 않다', async () => {
    const first: Vec3 = [0.25, 0.5, 0];
    const second: Vec3 = [0.5, 0.5, 0];
    const f = await makeFixture();
    try {
      f.sync.commit();
      f.editor.updateTransform('box_a', { position: first });
      const node = recordingView(f.handle.builtEntities.get('box_a')?.node);
      // apply()를 **부르지 않고** 확인한다 — 렌더 프레임 없이도 반영돼 있어야 한다
      expect(node.lastPosition).toEqual(first);

      f.editor.updateTransform('box_a', { position: second });
      expect(node.lastPosition).toEqual(second);
    } finally {
      f.world.free();
    }
  });
});

// ── 3. updateDimensions / updatePhysics (재빌드) ─────────────────────

describe('SceneEditorImpl — updateDimensions / updatePhysics', () => {
  it('updateDimensions는 시각+collider를 함께 교체 재빌드한다 (새 정착 높이로 검증)', async () => {
    const f = await makeFixture();
    try {
      const oldBodyId = soleBodyOf(f.world, 'box_a');

      f.editor.updateDimensions('box_a', {
        kind: 'box',
        halfExtents: [BIG_BOX_HALF_M, BIG_BOX_HALF_M, BIG_BOX_HALF_M],
      });

      // 재빌드 — 핸들 매핑이 새 바디로 갱신된다
      const newBodyId = soleBodyOf(f.world, 'box_a');
      expect(newBodyId).not.toBe(oldBodyId);

      // spec: visual.primitive와 colliders[0].shape 동시 교체, 다른 필드 유지
      const entity = entityById(f.editor.spec, 'box_a');
      expect(entity.visual.primitive).toEqual({
        kind: 'box',
        halfExtents: [BIG_BOX_HALF_M, BIG_BOX_HALF_M, BIG_BOX_HALF_M],
      });
      const collider = entity.physics?.colliders[0];
      expect(collider?.shape).toEqual({
        kind: 'box',
        halfExtents: [BIG_BOX_HALF_M, BIG_BOX_HALF_M, BIG_BOX_HALF_M],
      });
      expect(collider?.friction).toBe(0.6); // 기존 collider 필드 보존
      expect(collider?.emitEvents).toBe(true);
      expect(collider?.group).toBe('OBJECT');

      // 시각 재생성 (이전 노드 제거 + 새 노드 추가)
      expect(f.renderApi.counts.remove).toBe(1);
      expect(f.renderApi.counts.addPrimitive).toBe(2);
      expect(f.events).toEqual([{ kind: 'dimensions', entityId: 'box_a' }]);

      // 물리 검증: 새 half-extent 높이로 정착 → collider가 실제로 커졌다
      stepSeconds(f.world, SETTLE_SEC);
      const restY = f.world.getPose(newBodyId).position[1];
      expect(Math.abs(restY - BIG_BOX_HALF_M)).toBeLessThan(REST_TOLERANCE_M);
      expect(validateScene(f.editor.serialize()).ok).toBe(true);
    } finally {
      f.world.free();
    }
  });

  it('updatePhysics의 bodyType 변경(dynamic→fixed)이 실제 물리 동작을 바꾼다', async () => {
    const f = await makeFixture();
    try {
      const entity = entityById(f.editor.serialize(), 'box_a');
      const physics = structuredClone(entity.physics);
      if (!physics) throw new Error('box_a에 physics가 있어야 합니다');
      physics.bodyType = 'fixed';

      f.editor.updatePhysics('box_a', physics);

      expect(entityById(f.editor.spec, 'box_a').physics?.bodyType).toBe('fixed');
      expect(f.events).toEqual([{ kind: 'physics', entityId: 'box_a' }]);

      // fixed 바디는 낙하하지 않는다 — 재빌드 위치(spec transform) 그대로
      const bodyId = soleBodyOf(f.world, 'box_a');
      stepSeconds(f.world, 1);
      const pose = f.world.getPose(bodyId);
      expect(pose.position[1]).toBeCloseTo(DROP_Y_M, POSE_DECIMALS);
      expect(validateScene(f.editor.serialize()).ok).toBe(true);
    } finally {
      f.world.free();
    }
  });

  it('updatePhysics 교차 규칙 위반(trimesh×dynamic)은 검증 오류로 롤백된다 — 엔티티 유지', async () => {
    const f = await makeFixture();
    try {
      const before = f.editor.serialize();
      const bodyIdBefore = soleBodyOf(f.world, 'box_a');

      expect(() =>
        f.editor.updatePhysics('box_a', {
          bodyType: 'dynamic',
          colliders: [
            {
              shape: { kind: 'trimesh', ref: 'asset://x' },
              group: 'OBJECT',
              collidesWith: ['ENV'],
            },
          ],
        }),
      ).toThrow(/trimesh/);

      expect(f.editor.serialize()).toEqual(before);
      expect(soleBodyOf(f.world, 'box_a')).toBe(bodyIdBefore); // 재빌드 미발생
      expect(f.events).toEqual([]);
    } finally {
      f.world.free();
    }
  });

  it('프리미티브가 아닌 엔티티의 updateDimensions는 한국어 오류로 거부한다', async () => {
    const f = await makeFixture(
      { ...baseScene(), entities: [robotEntity('arm')] },
      { withRobots: true },
    );
    try {
      expect(() =>
        f.editor.updateDimensions('arm', { kind: 'sphere', radius: 0.1 }),
      ).toThrow(/프리미티브 시각 엔티티가 아니어서/);
    } finally {
      f.world.free();
    }
  });

  it('collider가 2개 이상인 엔티티의 updateDimensions는 거부한다 (형상 드리프트 방지)', async () => {
    const spec = baseScene();
    const multi = boxEntity('multi', [0.3, DROP_Y_M, 0]);
    multi.physics?.colliders.push({
      shape: { kind: 'box', halfExtents: [BOX_HALF_M, BOX_HALF_M, BOX_HALF_M] },
      offset: { position: [0, BOX_HALF_M * 2, 0] },
      group: 'OBJECT',
      collidesWith: ['ENV'],
    });
    spec.entities.push(multi);
    const f = await makeFixture(spec);
    try {
      const before = f.editor.serialize();
      expect(() =>
        f.editor.updateDimensions('multi', {
          kind: 'box',
          halfExtents: [BIG_BOX_HALF_M, BIG_BOX_HALF_M, BIG_BOX_HALF_M],
        }),
      ).toThrow(/collider가 2개여서/);
      // spec 미변경 — colliders[1..]가 이전 형상으로 남는 조용한 드리프트가 없다
      expect(f.editor.serialize()).toEqual(before);
      expect(f.events).toEqual([]);
    } finally {
      f.world.free();
    }
  });
});

// ── 4. renameEntity ─────────────────────────────────────────────────

describe('SceneEditorImpl — renameEntity', () => {
  it('비로봇 엔티티: spec id + 월드 핸들 매핑을 새 id로 재구성하고 충돌 이벤트도 새 id로 나온다', async () => {
    const f = await makeFixture();
    try {
      f.editor.renameEntity('box_a', 'crate');

      expect(f.editor.spec.entities.map((e) => e.id)).toEqual(['crate']);
      expect(f.world.bodiesOfEntity('box_a')).toEqual([]);
      expect(f.world.bodiesOfEntity('crate')).toHaveLength(1);
      expect(f.events).toEqual([{ kind: 'rename', entityId: 'crate' }]);
      expect(validateScene(f.editor.serialize()).ok).toBe(true);

      // 핸들 매핑의 진실이 갱신됐다 — 접촉 이벤트가 새 id로 번역된다 (CLAUDE.md §2.4)
      const contactEvents = stepSeconds(f.world, SETTLE_SEC);
      expect(
        contactEvents.some((e) => e.phase === 'start' && (e.a === 'crate' || e.b === 'crate')),
      ).toBe(true);
      expect(contactEvents.some((e) => e.a === 'box_a' || e.b === 'box_a')).toBe(false);
    } finally {
      f.world.free();
    }
  });

  it('중복 id·예약 id로의 rename은 거부한다', async () => {
    const f = await makeFixture();
    try {
      await f.editor.addEntity(boxEntity('box_b', [0.3, DROP_Y_M, 0]));
      expect(() => f.editor.renameEntity('box_a', 'box_b')).toThrow(/이미 존재합니다/);
      expect(() => f.editor.renameEntity('box_a', '__ground')).toThrow(/예약한 id/);
      // 동일 id rename은 no-op (이벤트 없음)
      const eventCount = f.events.length;
      f.editor.renameEntity('box_a', 'box_a');
      expect(f.events).toHaveLength(eventCount);
    } finally {
      f.world.free();
    }
  });
});

// ── 5. robot 엔티티 편집 (가짜 RobotHandle) ─────────────────────────

describe('SceneEditorImpl — robot 편집', () => {
  it('addEntity(robot)는 URDF 로드 → 레지스트리 등록 + 링크 바디 생성', async () => {
    const f = await makeFixture(baseScene(), { withRobots: true });
    try {
      await f.editor.addEntity(robotEntity('arm'));

      expect(f.handle.robots.ids()).toEqual(['arm']);
      expect(f.world.bodiesOfEntity('arm')).toHaveLength(2); // L1/L2 링크 바디
      expect(f.fakes).toHaveLength(1);
      expect(f.fakes[0]?.rootTransforms).toEqual([
        { position: [0.1, 0, -0.2], rotation: [0, 0, 0, 1] },
      ]);
      expect(f.handle.robots.get('arm').readJoints()).toEqual({ j1: HOME_J1_RAD, j2: 0 });
      expect(f.events).toEqual([{ kind: 'add', entityId: 'arm' }]);
      expect(validateScene(f.editor.serialize()).ok).toBe(true);
    } finally {
      f.world.free();
    }
  });

  it('removeEntity(robot)는 레지스트리·링크 바디·렌더 핸들을 정리한다', async () => {
    const f = await makeFixture(baseScene(), { withRobots: true });
    try {
      await f.editor.addEntity(robotEntity('arm'));
      f.editor.removeEntity('arm');

      expect(f.handle.robots.has('arm')).toBe(false);
      expect(f.world.bodiesOfEntity('arm')).toEqual([]);
      expect(f.fakes[0]?.disposeCount).toBe(1);
      expect(f.editor.spec.entities.map((e) => e.id)).toEqual(['box_a']);
      // 제거된 로봇이 registry에 없으므로 이후 tickAll류 호출 경로가 죽은 바디를
      // 만지지 않는다 — 남은 월드 스텝이 무해하다
      stepSeconds(f.world, 0.1);
    } finally {
      f.world.free();
    }
  });

  it('updateTransform(robot)은 루트 트랜스폼 재설정 + FK 재-push를 수행한다', async () => {
    const f = await makeFixture(baseScene(), { withRobots: true });
    try {
      await f.editor.addEntity(robotEntity('arm'));
      f.editor.updateTransform('arm', { position: [1, 0, 1] });

      const fake = f.fakes[0];
      if (!fake) throw new Error('가짜 로봇 핸들이 없습니다');
      expect(fake.rootTransforms.at(-1)).toEqual({
        position: [1, 0, 1],
        rotation: [0, 0, 0, 1],
      });
      // 링크 바디는 FK pose로 즉시 정렬된다 (가짜 FK는 루트 무관 — home j1 기준).
      // Rapier는 좌표를 f32로 저장하므로 성분별 근사 비교를 쓴다.
      const linkPoses = f.world.bodiesOfEntity('arm').map((b) => f.world.getPose(b));
      const expectedLinkPositions: Vec3[] = [
        [HOME_J1_RAD, L1_BASE_Y_M, 0],
        [HOME_J1_RAD, L2_BASE_Y_M, 0],
      ];
      expect(linkPoses).toHaveLength(expectedLinkPositions.length);
      linkPoses.forEach((pose, i) => {
        const expected = expectedLinkPositions[i];
        if (!expected) throw new Error('기대 링크 pose가 없습니다');
        pose.position.forEach((v, axis) => {
          expect(v).toBeCloseTo(expected[axis] ?? 0, POSE_DECIMALS);
        });
      });
      expect(entityById(f.editor.spec, 'arm').transform.position).toEqual([1, 0, 1]);
      expect(f.events.at(-1)).toEqual({ kind: 'transform', entityId: 'arm' });
    } finally {
      f.world.free();
    }
  });

  it('renameEntity(robot)는 레지스트리·바디를 새 id로 재부착하고 관절 상태를 보존한다', async () => {
    const f = await makeFixture(baseScene(), { withRobots: true });
    try {
      await f.editor.addEntity(robotEntity('arm'));
      f.handle.robots.get('arm').setJoint('j1', -0.25); // home과 다른 상태로

      f.editor.renameEntity('arm', 'arm2');

      expect(f.handle.robots.has('arm')).toBe(false);
      expect(f.handle.robots.ids()).toEqual(['arm2']);
      expect(f.world.bodiesOfEntity('arm')).toEqual([]);
      expect(f.world.bodiesOfEntity('arm2')).toHaveLength(2);
      // 관절 상태 보존 (URDF 재로드·re-home 없음)
      expect(f.handle.robots.get('arm2').readJoints()).toEqual({ j1: -0.25, j2: 0 });
      // 링크 바디가 보존된 관절 상태의 FK pose로 정렬됐다
      const linkPoses = f.world.bodiesOfEntity('arm2').map((b) => f.world.getPose(b));
      expect(linkPoses[0]?.position[0]).toBeCloseTo(-0.25, POSE_DECIMALS);
      // 시각 핸들은 재사용된다 — dispose되지 않았다
      expect(f.fakes[0]?.disposeCount).toBe(0);
      expect(entityById(f.editor.spec, 'arm2').type).toBe('robot');
      expect(validateScene(f.editor.serialize()).ok).toBe(true);
    } finally {
      f.world.free();
    }
  });

  it('robot의 updatePhysics는 거부한다 (물리는 URDF에서 유도)', async () => {
    const f = await makeFixture(
      { ...baseScene(), entities: [robotEntity('arm')] },
      { withRobots: true },
    );
    try {
      expect(() =>
        f.editor.updatePhysics('arm', { bodyType: 'fixed', colliders: [] }),
      ).toThrow(/URDF에서 유도/);
    } finally {
      f.world.free();
    }
  });
});

// ── 6. serialize / onChange ─────────────────────────────────────────

describe('SceneEditorImpl — serialize / onChange', () => {
  it('serialize는 깊은 복사본을 반환한다 — 반환값 변형이 편집기 상태에 영향 없음', async () => {
    const f = await makeFixture();
    try {
      const snapshot = f.editor.serialize();
      snapshot.entities.pop();
      snapshot.name = 'mutated';
      expect(f.editor.spec.name).toBe('editor-test');
      expect(f.editor.spec.entities).toHaveLength(1);
    } finally {
      f.world.free();
    }
  });

  it('여러 연산을 거친 spec도 validateScene을 통과한다 (round-trip)', async () => {
    const f = await makeFixture();
    try {
      await f.editor.addEntity(boxEntity('box_b', [0.3, DROP_Y_M, 0]));
      f.editor.updateTransform('box_a', { position: [0, 1, 0], rotation: [0, 0, 0, 1] });
      f.editor.updateDimensions('box_b', { kind: 'sphere', radius: 0.07 });
      f.editor.renameEntity('box_a', 'crate');
      f.editor.removeEntity('box_b');

      const result = validateScene(f.editor.serialize());
      if (!result.ok) throw new Error(result.errors.join('\n'));
      expect(result.value.entities.map((e) => e.id)).toEqual(['crate']);
      expect(f.events.map((e) => e.kind)).toEqual([
        'add',
        'transform',
        'dimensions',
        'rename',
        'remove',
      ]);
    } finally {
      f.world.free();
    }
  });

  it('onChange 해제 함수는 이후 통지를 멈춘다', async () => {
    const f = await makeFixture();
    try {
      const seen: SceneEditEvent[] = [];
      const off = f.editor.onChange((e) => seen.push(e));
      f.editor.updateTransform('box_a', { position: [0, 1, 0] });
      off();
      f.editor.updateTransform('box_a', { position: [0, 2, 0] });
      expect(seen).toHaveLength(1);
    } finally {
      f.world.free();
    }
  });
});
