// core/robot-edit-contract.test.ts — 로봇 편집 경로의 계약 고정 테스트
//
// 배경: "오브젝트는 이동하는데 로봇이 이동을 안한다" 보고에서 출발해 로봇 분기와
// 비로봇 분기의 처리 차이를 전수 비교했고, 드러난 비대칭(로봇 레코드의 initialPose
// 부재, 커밋되지 않은 드래그 프리뷰를 되돌릴 지점 부재)을 해소한 뒤 **수정 후 계약**을
// 여기에 고정한다. ★회귀 표시가 붙은 케이스가 그 계약이다.
//
// scene-editor.test.ts의 FakeRobotHandle은 setRootTransform을 "기록만" 하고 FK에
// 반영하지 않아 루트 배치가 링크 pose에 미치는 영향을 볼 수 없다. 여기서는 실제
// render/urdf.ts UrdfRobotHandle의 구조(outer 그룹 트랜스폼 ∘ 로봇 프레임 링크 pose)를
// 순수 수학으로 재현하는 RootAwareFakeRobot을 쓴다 — three 의존 없이(vitest.config.ts
// 원칙) 루트 이동/회전이 링크 월드 pose로 합성되는 계약을 검증할 수 있다.

import { beforeAll, describe, expect, it } from 'vitest';
import { validateScene } from '../schema/validate';
import type { EntitySpec, Quat, RobotSpec, SceneSpec, Vec3 } from '../schema/types';
import type { Pose } from './types';
import type { JointInfo, LinkColliderDef } from './robot-types';
import { initPhysics, RapierWorld } from './world';
import { RenderSync } from './sync';
import { SceneLoader } from './scene-loader';
import type {
  BuiltEntityHandle,
  RenderSceneApi,
  RobotHandle,
  SceneHandle,
  VisualNode,
} from './scene-loader';
import { SceneEditorImpl } from './scene-editor';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4) ──────────────────────────────

const TIMESTEP_HZ = 240;
const BOX_HALF_M = 0.05;
const DROP_Y_M = 0.5;
const POSE_DECIMALS = 5;
const FAKE_URDF_PATH = 'assets/robots/fake/fake.urdf';
/** 로봇 프레임 기준 링크 위치 (이미 Y-up — axisFix 이후 좌표계) */
const L1_LOCAL: Vec3 = [0, 0.1, 0];
const L2_LOCAL_Y_M = 0.3;
/** 90° yaw (Y축) 쿼터니언 [x,y,z,w] */
const YAW_90: Quat = [0, Math.SQRT1_2, 0, Math.SQRT1_2];

// ── 순수 쿼터니언 헬퍼 (three 비의존 — 테스트 로컬) ──────────────────

function rotateByQuat(q: Readonly<Quat>, v: Readonly<Vec3>): Vec3 {
  const [qx, qy, qz, qw] = q;
  const [vx, vy, vz] = v;
  // t = 2 * (q_vec × v); v' = v + qw * t + q_vec × t
  const tx = 2 * (qy * vz - qz * vy);
  const ty = 2 * (qz * vx - qx * vz);
  const tz = 2 * (qx * vy - qy * vx);
  return [
    vx + qw * tx + (qy * tz - qz * ty),
    vy + qw * ty + (qz * tx - qx * tz),
    vz + qw * tz + (qx * ty - qy * tx),
  ];
}

function quatMul(a: Readonly<Quat>, b: Readonly<Quat>): Quat {
  const [ax, ay, az, aw] = a;
  const [bx, by, bz, bw] = b;
  return [
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  ];
}

// ── RootAwareFakeRobot (render/urdf.ts UrdfRobotHandle 구조 재현) ─────

const FAKE_LINK_COLLIDER_DEF: LinkColliderDef = {
  shape: { kind: 'box', halfExtents: [0.03, 0.03, 0.03] },
  offset: { position: [0, 0, 0] },
};

/**
 * outer 그룹 트랜스폼(= setRootTransform)이 링크 월드 pose에 합성되는 실제 계약을
 * 순수 수학으로 재현한 FK 뷰. readLinkPoses()는 root ∘ local을 돌려준다.
 */
class RootAwareFakeRobot implements RobotHandle {
  readonly joints: readonly JointInfo[] = [
    { name: 'j1', type: 'revolute', limits: [-1, 1], initial: 0 },
  ];
  readonly linkColliders: ReadonlyMap<string, readonly LinkColliderDef[]> = new Map([
    ['L1', [FAKE_LINK_COLLIDER_DEF]],
    ['L2', [FAKE_LINK_COLLIDER_DEF]],
  ]);

  /** setRootTransform 수신 이력 (호출 여부·값 검증용) */
  readonly rootTransforms: Pose[] = [];
  disposeCount = 0;

  private root: Pose = { position: [0, 0, 0], rotation: [0, 0, 0, 1] };
  private j1 = 0;

  setJointValues(values: Readonly<Record<string, number>>): void {
    const next = values['j1'];
    if (next !== undefined) this.j1 = next;
  }

  readLinkPoses(): ReadonlyMap<string, Pose> {
    const locals: ReadonlyArray<[string, Vec3]> = [
      ['L1', L1_LOCAL],
      ['L2', [this.j1, L2_LOCAL_Y_M, 0]],
    ];
    const out = new Map<string, Pose>();
    for (const [name, local] of locals) {
      const rotated = rotateByQuat(this.root.rotation, local);
      out.set(name, {
        position: [
          this.root.position[0] + rotated[0],
          this.root.position[1] + rotated[1],
          this.root.position[2] + rotated[2],
        ],
        rotation: quatMul(this.root.rotation, [0, 0, 0, 1]),
      });
    }
    return out;
  }

  setRootTransform(pose: Pose): void {
    this.root = {
      position: [pose.position[0], pose.position[1], pose.position[2]],
      rotation: [pose.rotation[0], pose.rotation[1], pose.rotation[2], pose.rotation[3]],
    };
    this.rootTransforms.push({ position: [...this.root.position], rotation: [...this.root.rotation] });
  }

  /** 테스트 전용: 현재 루트 (기즈모 드래그 프리뷰가 남긴 값 조회용) */
  currentRoot(): Pose {
    return { position: [...this.root.position], rotation: [...this.root.rotation] };
  }

  dispose(): void {
    this.disposeCount += 1;
  }
}

// ── 픽스처 ──────────────────────────────────────────────────────────

function stubNode(): VisualNode {
  return {
    position: { set: (): void => undefined },
    quaternion: { set: (): void => undefined },
  } as unknown as VisualNode;
}

function boxEntity(id: string, position: Vec3): EntitySpec {
  return {
    id,
    type: 'object',
    transform: { position },
    visual: {
      kind: 'primitive',
      primitive: { kind: 'box', halfExtents: [BOX_HALF_M, BOX_HALF_M, BOX_HALF_M] },
    },
    physics: {
      bodyType: 'dynamic',
      colliders: [
        {
          shape: { kind: 'box', halfExtents: [BOX_HALF_M, BOX_HALF_M, BOX_HALF_M] },
          group: 'OBJECT',
          collidesWith: ['ENV', 'ROBOT', 'OBJECT'],
          emitEvents: true,
        },
      ],
    },
  };
}

function robotEntity(id: string, position: Vec3 = [0, 0, 0]): RobotSpec {
  return {
    id,
    type: 'robot',
    transform: { position },
    visual: { kind: 'urdf', ref: FAKE_URDF_PATH },
    urdf: FAKE_URDF_PATH,
    controller: 'manual',
    linkColliders: 'fromVisual',
    selfCollision: false,
  };
}

function baseScene(entities: EntitySpec[]): SceneSpec {
  return {
    name: 'robot-edit-contract',
    version: 1,
    gravity: [0, -9.81, 0],
    timestepHz: TIMESTEP_HZ,
    environment: { ground: true },
    entities,
  };
}

interface Fixture {
  world: RapierWorld;
  sync: RenderSync;
  handle: SceneHandle;
  editor: SceneEditorImpl;
  fakes: RootAwareFakeRobot[];
}

async function makeFixture(spec: SceneSpec): Promise<Fixture> {
  const validated = validateScene(spec);
  if (!validated.ok) throw new Error(`픽스처 씬 검증 실패:\n${validated.errors.join('\n')}`);

  const world = new RapierWorld(validated.value.gravity, validated.value.timestepHz);
  const fakes: RootAwareFakeRobot[] = [];
  const renderApi: RenderSceneApi = {
    addPrimitive: () => stubNode(),
    addGround: () => stubNode(),
    setPose: () => undefined,
    remove: () => undefined,
    loadRobot: () => {
      const fake = new RootAwareFakeRobot();
      fakes.push(fake);
      return Promise.resolve<RobotHandle>(fake);
    },
  };
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
  return { world, sync, handle, editor, fakes };
}

function requireFake(f: Fixture, index: number): RootAwareFakeRobot {
  const fake = f.fakes[index];
  if (!fake) throw new Error(`가짜 로봇 핸들[${index}]이 없습니다`);
  return fake;
}

function requireRecord(f: Fixture, id: string): BuiltEntityHandle {
  const record = f.handle.builtEntities.get(id);
  if (!record) throw new Error(`빌드 레코드 '${id}'이(가) 없습니다`);
  return record;
}

/** 엔티티의 모든 바디 pose (world 매핑 순서) */
function posesOf(f: Fixture, id: string): Pose[] {
  return f.world.bodiesOfEntity(id).map((b) => f.world.getPose(b));
}

function expectVec3Close(actual: Readonly<Vec3>, expected: Readonly<Vec3>): void {
  actual.forEach((v, i) => {
    expect(v).toBeCloseTo(expected[i] ?? 0, POSE_DECIMALS);
  });
}

function stepWithRobots(f: Fixture, steps: number): void {
  for (let i = 0; i < steps; i += 1) {
    f.handle.robots.tickAll(); // Engine preStep 훅과 같은 순서
    f.world.step();
  }
}

beforeAll(async () => {
  await initPhysics();
});

// ── 1. updateTransform 로봇 분기 vs 비로봇 분기 ──────────────────────

describe('로봇 updateTransform — 물리 반영 및 initialPose 계약 (비로봇과 대칭)', () => {
  it('루트 이동이 링크 바디 월드 pose에 즉시 합성된다 (편집 직후 스텝 전)', async () => {
    const f = await makeFixture(baseScene([robotEntity('arm', [0, 0, 0])]));
    try {
      expectVec3Close(posesOf(f, 'arm')[0]?.position ?? [0, 0, 0], L1_LOCAL);

      f.editor.updateTransform('arm', { position: [1, 0, -2] });

      const poses = posesOf(f, 'arm');
      expect(poses).toHaveLength(2);
      expectVec3Close(poses[0]?.position ?? [0, 0, 0], [1, L1_LOCAL[1], -2]);
      expectVec3Close(poses[1]?.position ?? [0, 0, 0], [1, L2_LOCAL_Y_M, -2]);
    } finally {
      f.world.free();
    }
  });

  it('★회귀: 로봇 레코드의 initialPose도 빌드/편집 양쪽에서 설정된다 (비로봇과 대칭)', async () => {
    const f = await makeFixture(
      baseScene([robotEntity('arm', [0.25, 0, -0.25]), boxEntity('box', [0, DROP_Y_M, 0])]),
    );
    try {
      // 빌드 직후 — 로봇도 스펙 배치를 복원점으로 갖는다
      expect(requireRecord(f, 'arm').initialPose).toEqual({
        position: [0.25, 0, -0.25],
        rotation: [0, 0, 0, 1],
      });
      expect(requireRecord(f, 'box').initialPose).toEqual({
        position: [0, DROP_Y_M, 0],
        rotation: [0, 0, 0, 1],
      });

      f.editor.updateTransform('arm', { position: [1, 0, 1], rotation: YAW_90 });
      f.editor.updateTransform('box', { position: [2, DROP_Y_M, 2] });

      // 두 분기 모두 편집된 배치로 복원점을 갱신한다
      expect(requireRecord(f, 'arm').initialPose).toEqual({
        position: [1, 0, 1],
        rotation: [...YAW_90],
      });
      expect(requireRecord(f, 'box').initialPose).toEqual({
        position: [2, DROP_Y_M, 2],
        rotation: [0, 0, 0, 1],
      });
    } finally {
      f.world.free();
    }
  });

  it('★회귀: 한 번의 updateTransform이 spec·FK 루트·링크 바디·initialPose를 모두 갱신한다', async () => {
    const f = await makeFixture(baseScene([robotEntity('arm', [0, 0, 0])]));
    try {
      const fake = requireFake(f, 0);

      f.editor.updateTransform('arm', { position: [1, 0, -2] });

      // ① spec (편집의 단일 진실)
      expect(f.editor.spec.entities[0]?.transform.position).toEqual([1, 0, -2]);
      // ② 시각 FK 루트 (렌더 핸들이 소유 — 로봇의 시각은 물리의 거울이 아니라 FK 그래프)
      expect(fake.currentRoot().position).toEqual([1, 0, -2]);
      // ③ 링크 물리 바디 (스텝 전에 이미 정렬 — teleportLinksToFk)
      const poses = posesOf(f, 'arm');
      expectVec3Close(poses[0]?.position ?? [0, 0, 0], [1, L1_LOCAL[1], -2]);
      expectVec3Close(poses[1]?.position ?? [0, 0, 0], [1, L2_LOCAL_Y_M, -2]);
      // ④ 되감기 복원점
      expect(requireRecord(f, 'arm').initialPose?.position).toEqual([1, 0, -2]);
    } finally {
      f.world.free();
    }
  });

  it('그럼에도 reset()은 편집된 로봇 배치를 유지한다 (루트 진실이 렌더 핸들에 있으므로)', async () => {
    const f = await makeFixture(baseScene([robotEntity('arm')]));
    try {
      f.editor.updateTransform('arm', { position: [1.5, 0, 0.5] });
      stepWithRobots(f, 60);

      f.handle.reset();

      const poses = posesOf(f, 'arm');
      expectVec3Close(poses[0]?.position ?? [0, 0, 0], [1.5, L1_LOCAL[1], 0.5]);
      expectVec3Close(poses[1]?.position ?? [0, 0, 0], [1.5, L2_LOCAL_Y_M, 0.5]);
      // spec과도 일치 — 되감기 후 재실행(runFromNode)도 같은 배치에서 출발한다
      const entity = f.editor.spec.entities.find((e) => e.id === 'arm');
      expect(entity?.transform.position).toEqual([1.5, 0, 0.5]);
    } finally {
      f.world.free();
    }
  });

  it('★회귀: 커밋되지 않은 기즈모 드래그 프리뷰를 resyncTransform이 spec 진실로 되돌린다', async () => {
    // 시나리오: 기즈모 드래그가 outer 그룹을 시각적으로 옮겼지만(= setRootTransform 상당)
    // commit이 발행되지 않거나(no-op 드래그·스케일 거부) SceneEditor가 거부한 경우.
    // - 비로봇: onDraggingChanged(false)의 sync.bind 재바인딩 + 물리 pose가 진실이라
    //   다음 프레임에 시각이 원위치로 재수렴한다(자가 치유).
    // - 로봇: 재바인딩 대상이 없다(레코드에 bodyId/node가 없다) — 통합자가 명시적으로
    //   resyncTransform을 호출해 같은 재수렴을 만든다. 그 계약을 여기서 고정한다.
    const f = await makeFixture(baseScene([robotEntity('arm')]));
    try {
      const fake = requireFake(f, 0);
      const specBefore = f.editor.serialize();

      // 드래그 프리뷰 (TransformControls가 outer.position에 직접 쓰는 것과 동일 효과)
      fake.setRootTransform({ position: [3, 0, 3], rotation: [0, 0, 0, 1] });
      expect(f.editor.serialize()).toEqual(specBefore); // commit 없음 → spec 불변

      // 드래그 종료 훅: spec 진실로 재수렴 (spec은 건드리지 않는다)
      f.editor.resyncTransform('arm');

      expect(fake.currentRoot().position).toEqual([0, 0, 0]);
      expectVec3Close(posesOf(f, 'arm')[0]?.position ?? [0, 0, 0], L1_LOCAL);
      stepWithRobots(f, 1); // 다음 tick의 FK push도 프리뷰를 물리로 역류시키지 않는다
      expectVec3Close(posesOf(f, 'arm')[0]?.position ?? [0, 0, 0], L1_LOCAL);
      expect(f.editor.serialize()).toEqual(specBefore);
    } finally {
      f.world.free();
    }
  });

  it('★회귀: 되감기(reset)도 로봇 루트를 spec 배치로 복구한다 (프리뷰가 남아 있어도)', async () => {
    const f = await makeFixture(baseScene([robotEntity('arm', [0.5, 0, 0])]));
    try {
      const fake = requireFake(f, 0);

      // 커밋되지 않은 프리뷰가 남은 채 재생 → 되감기
      fake.setRootTransform({ position: [3, 0, 3], rotation: [0, 0, 0, 1] });
      stepWithRobots(f, 5);
      f.handle.reset();

      expect(fake.currentRoot().position).toEqual([0.5, 0, 0]);
      expectVec3Close(posesOf(f, 'arm')[0]?.position ?? [0, 0, 0], [0.5, L1_LOCAL[1], 0]);
      expect(f.editor.spec.entities[0]?.transform.position).toEqual([0.5, 0, 0]);
    } finally {
      f.world.free();
    }
  });
});

// ── 2. 회전(rotate) 커밋 ────────────────────────────────────────────

describe('로봇 회전 커밋', () => {
  it('쿼터니언이 변형 없이 setRootTransform으로 전달되고 링크 pose에 합성된다', async () => {
    const f = await makeFixture(baseScene([robotEntity('arm')]));
    try {
      const fake = requireFake(f, 0);
      f.handle.robots.get('arm').setJoint('j1', 0.4);
      f.handle.robots.get('arm').teleportLinksToFk();

      f.editor.updateTransform('arm', { position: [1, 0, 0], rotation: YAW_90 });

      // core는 축 변환을 하지 않는다 — 커밋 쿼터니언이 그대로 렌더 핸들에 도달
      expect(fake.rootTransforms.at(-1)?.rotation).toEqual([...YAW_90]);

      // yaw 90°: 로컬 [0.4, 0.3, 0] → 월드 [0, 0.3, -0.4] (+루트 [1,0,0])
      const poses = posesOf(f, 'arm');
      expectVec3Close(poses[1]?.position ?? [0, 0, 0], [1, L2_LOCAL_Y_M, -0.4]);
      const entity = f.editor.spec.entities[0];
      expect(entity?.transform.rotation).toEqual([...YAW_90]);
    } finally {
      f.world.free();
    }
  });

  it('회전 후 reset()도 회전을 유지한다 (관절만 home 복원)', async () => {
    const f = await makeFixture(baseScene([robotEntity('arm')]));
    try {
      f.editor.updateTransform('arm', { position: [0, 0, 0], rotation: YAW_90 });
      f.handle.robots.get('arm').setJoint('j1', 0.5);
      stepWithRobots(f, 10);

      f.handle.reset();

      expect(requireFake(f, 0).currentRoot().rotation).toEqual([...YAW_90]);
      // j1이 home(=initial 0)으로 복원 → L2 로컬 [0,0.3,0] → 회전해도 [0,0.3,0]
      expectVec3Close(posesOf(f, 'arm')[1]?.position ?? [0, 0, 0], [0, L2_LOCAL_Y_M, 0]);
    } finally {
      f.world.free();
    }
  });
});

// ── 3. updateDimensions / updatePhysics 거부 ────────────────────────

describe('로봇 치수/물리 편집 거부', () => {
  it('둘 다 한국어 오류로 거부하고 spec·물리를 바꾸지 않는다', async () => {
    const f = await makeFixture(baseScene([robotEntity('arm')]));
    try {
      const before = f.editor.serialize();
      const bodiesBefore = f.world.bodiesOfEntity('arm');

      expect(() => f.editor.updateDimensions('arm', { kind: 'sphere', radius: 0.1 })).toThrow(
        /프리미티브 시각 엔티티가 아니어서/,
      );
      expect(() => f.editor.updatePhysics('arm', { bodyType: 'fixed', colliders: [] })).toThrow(
        /URDF에서 유도/,
      );

      expect(f.editor.serialize()).toEqual(before);
      expect(f.world.bodiesOfEntity('arm')).toEqual(bodiesBefore);
    } finally {
      f.world.free();
    }
  });
});

// ── 4. 라이브러리 추가 로봇 vs 씬 JSON 로봇의 레코드 동형성 ──────────

describe('로봇 레코드 구조 동형성 (씬 JSON vs 라이브러리 추가)', () => {
  it('bodyId/node/bound/robot 필드 구성이 동일하다', async () => {
    const f = await makeFixture(baseScene([robotEntity('arm_scene', [0, 0, 0])]));
    try {
      await f.editor.addEntity(robotEntity('arm_added', [1, 0, 0]));

      const shapeOf = (record: BuiltEntityHandle): Record<string, boolean> => ({
        hasBodyId: record.bodyId !== undefined,
        hasNode: record.node !== undefined,
        bound: record.bound,
        hasRobot: record.robot !== undefined,
        hasInitialPose: record.initialPose !== undefined,
      });
      expect(shapeOf(requireRecord(f, 'arm_added'))).toEqual(shapeOf(requireRecord(f, 'arm_scene')));
      expect(shapeOf(requireRecord(f, 'arm_scene'))).toEqual({
        hasBodyId: false,
        hasNode: false,
        bound: false,
        hasRobot: true,
        hasInitialPose: true, // 로봇도 되감기 복원점을 갖는다 (비로봇과 대칭)
      });
      // 두 로봇 모두 레지스트리·링크 바디를 갖는다
      expect(f.handle.robots.ids()).toEqual(['arm_scene', 'arm_added']);
      expect(f.world.bodiesOfEntity('arm_added')).toHaveLength(2);
    } finally {
      f.world.free();
    }
  });
});

// ── 5. 다중 로봇 간섭 ───────────────────────────────────────────────

describe('로봇 2대 — 편집 간섭 없음', () => {
  it('한 로봇의 updateTransform이 다른 로봇의 루트/링크 바디에 영향을 주지 않는다', async () => {
    const f = await makeFixture(
      baseScene([robotEntity('arm_a', [-0.5, 0, 0]), robotEntity('arm_b', [0.5, 0, 0])]),
    );
    try {
      const [fakeA, fakeB] = [requireFake(f, 0), requireFake(f, 1)];
      const bBefore = posesOf(f, 'arm_b');

      f.editor.updateTransform('arm_a', { position: [-2, 0, 0] });

      expectVec3Close(posesOf(f, 'arm_a')[0]?.position ?? [0, 0, 0], [-2, L1_LOCAL[1], 0]);
      expect(fakeA.currentRoot().position).toEqual([-2, 0, 0]);
      // B는 그대로 (핸들·바디 매핑이 로봇마다 고유)
      expect(fakeB.currentRoot().position).toEqual([0.5, 0, 0]);
      expect(posesOf(f, 'arm_b')).toEqual(bBefore);
      // 링크 바디 집합이 서로 겹치지 않는다
      const aBodies = new Set(f.world.bodiesOfEntity('arm_a'));
      expect(f.world.bodiesOfEntity('arm_b').some((b) => aBodies.has(b))).toBe(false);
    } finally {
      f.world.free();
    }
  });

  it('로봇 2대가 실제로 서로 충돌 이벤트를 발행한다 (그룹 비트로 억제되지 않음)', async () => {
    const f = await makeFixture(
      baseScene([robotEntity('arm_a', [-0.5, 0, 0]), robotEntity('arm_b', [0.5, 0, 0])]),
    );
    try {
      // 두 로봇 링크가 부분 겹치도록 배치 — ROBOT × ROBOT 접촉 (완전 동심은 퇴화)
      f.editor.updateTransform('arm_b', { position: [-0.47, 0, 0] });
      const events = [];
      for (let i = 0; i < 20; i += 1) {
        f.handle.robots.tickAll();
        events.push(...f.world.step());
      }
      expect(
        events.some(
          (e) =>
            e.phase === 'start' &&
            ((e.a === 'arm_a' && e.b === 'arm_b') || (e.a === 'arm_b' && e.b === 'arm_a')),
        ),
      ).toBe(true);
    } finally {
      f.world.free();
    }
  });
});

// ── 6. 재생 중 편집 (scene-editor.ts 헤더의 "정지/일시정지 상태에서만") ──

describe('재생 중 로봇 updateTransform', () => {
  it('core는 재생 중 호출을 막지 않는다 — 링크 바디가 즉시 순간이동하고 스윕 속도는 0이다', async () => {
    const f = await makeFixture(baseScene([robotEntity('arm')]));
    try {
      stepWithRobots(f, 30); // "재생 중" 상태 모사

      f.editor.updateTransform('arm', { position: [5, 0, 0] });
      const rightAfter = posesOf(f, 'arm')[0]?.position ?? [0, 0, 0];
      expectVec3Close(rightAfter, [5, L1_LOCAL[1], 0]);

      // 다음 스텝에서 잔여 스윕(가짜 kinematic 속도)으로 밀리지 않는다
      stepWithRobots(f, 1);
      expectVec3Close(posesOf(f, 'arm')[0]?.position ?? [0, 0, 0], [5, L1_LOCAL[1], 0]);
    } finally {
      f.world.free();
    }
  });

  it('★재생 중 로봇을 사물 위로 순간이동시키면 사물을 통과(터널링)해 충돌이 감지되지 않는다', async () => {
    const f = await makeFixture(
      baseScene([robotEntity('arm', [0, 0, 0]), boxEntity('box', [3, 0.1, 0])]),
    );
    try {
      stepWithRobots(f, 120); // 박스 정착

      // 로봇을 박스 "너머"로 한 번에 옮긴다 (드래그 커밋 1회와 동일)
      f.editor.updateTransform('arm', { position: [6, 0, 0] });
      const events: string[] = [];
      for (let i = 0; i < 10; i += 1) {
        f.handle.robots.tickAll();
        for (const e of f.world.step()) {
          if ((e.a === 'arm' && e.b === 'box') || (e.a === 'box' && e.b === 'arm')) {
            events.push(`${e.a}-${e.b}:${e.phase}`);
          }
        }
      }
      // 통과 경로상의 박스와의 접촉은 한 번도 잡히지 않는다 — 편집을 정지 상태로
      // 강제하는 것은 UI(main.ts runEdit/pauseForEditIfPlaying) 몫이라는 계약의 근거
      expect(events).toEqual([]);
    } finally {
      f.world.free();
    }
  });
});
