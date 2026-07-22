// core/scene-robot-reset.test.ts — 로봇 링크 바디의 fresh load / reset 상태 동일성 회귀
//
// 리뷰 지적(determinism): (a) 링크 바디가 URDF 초기 자세(home 적용 전) FK pose로
// 생성되어 첫 스텝에 초기→home 스윕(가짜 kinematic 속도)이 생겼고, (b) reset()이
// tick()만 호출해 다음 스텝까지 물리 pose가 리셋 전 상태로 남았다 — fresh load와
// reset 후의 "스텝 전 상태"가 달라 결정론적 재생(SIMULATION.md §6, SceneHandle.reset
// 계약)이 깨졌다. 수정: 바디를 home FK pose에서 생성 + reset()이 teleportLinksToFk로
// 링크 바디를 즉시 정렬. 이 테스트는 실제 RapierWorld로 두 경로의 pose를 검증한다.
//
// FK는 가짜 RobotHandle(관절값 → 결정론적 링크 pose)로 대체 — render/three 비의존.

import { beforeAll, describe, expect, it } from 'vitest';
import { initPhysics, RapierWorld } from './world';
import { RenderSync } from './sync';
import { SceneLoader } from './scene-loader';
import type { RenderSceneApi, RobotHandle, VisualNode } from './scene-loader';
import type { Pose } from './types';
import type { JointInfo, LinkColliderDef } from './robot-types';
import type { RobotSpec, SceneSpec } from '../schema/types';

// ── 테스트 상수 (매직넘버 금지 — CLAUDE.md §4) ──────────────────────

const TIMESTEP_HZ = 240;
const ROBOT_ID = 'arm';
const FAKE_URDF_PATH = 'assets/robots/fake/fake.urdf';

/** L1 링크의 관절값 무관 기본 높이 (m) — FK: L1.position = [j1, L1_BASE_Y_M, 0] */
const L1_BASE_Y_M = 0.2;
const J1_LIMITS_RAD: [number, number] = [-1, 1];
const HOME_J1_RAD = 0.5;
/** reset 전 구동 목표 (home과 확실히 다른 값) */
const DRIVE_J1_RAD = -0.8;
/** 구동 반영 스텝 수 */
const DRIVE_STEPS = 5;
const POSITION_DECIMALS = 6;

const LINK_COLLIDER_DEF: LinkColliderDef = {
  shape: { kind: 'box', halfExtents: [0.05, 0.05, 0.05] },
};

// ── 가짜 FK RobotHandle (j1 revolute 1관절 · L1 1링크) ──────────────

class FakeArmHandle implements RobotHandle {
  readonly joints: readonly JointInfo[] = [
    { name: 'j1', type: 'revolute', limits: [...J1_LIMITS_RAD], initial: 0 },
  ];
  readonly linkColliders: ReadonlyMap<string, readonly LinkColliderDef[]> = new Map([
    ['L1', [LINK_COLLIDER_DEF]],
  ]);

  private j1 = 0;

  setJointValues(values: Readonly<Record<string, number>>): void {
    if (values['j1'] !== undefined) this.j1 = values['j1'];
  }

  readLinkPoses(): ReadonlyMap<string, Pose> {
    return new Map<string, Pose>([
      ['L1', { position: [this.j1, L1_BASE_Y_M, 0], rotation: [0, 0, 0, 1] }],
    ]);
  }

  setRootTransform(): void {}
  dispose(): void {}
}

// ── 픽스처 ──────────────────────────────────────────────────────────

const ROBOT_SPEC: RobotSpec = {
  id: ROBOT_ID,
  type: 'robot',
  transform: { position: [0, 0, 0] },
  visual: { kind: 'urdf', ref: FAKE_URDF_PATH },
  urdf: FAKE_URDF_PATH,
  controller: 'manual',
  linkColliders: 'fromVisual',
  selfCollision: false,
  home: { j1: HOME_J1_RAD },
};

const SCENE_SPEC: SceneSpec = {
  name: 'robot-reset-parity',
  version: 1,
  gravity: [0, -9.81, 0],
  timestepHz: TIMESTEP_HZ,
  entities: [ROBOT_SPEC],
};

function stubNode(): VisualNode {
  const stub = {
    position: { set: () => undefined },
    quaternion: { set: () => undefined },
  };
  return stub as unknown as VisualNode;
}

function makeRenderApi(): RenderSceneApi {
  return {
    addPrimitive: stubNode,
    addGround: stubNode,
    setPose: () => undefined,
    remove: () => undefined,
    loadRobot: () => Promise.resolve<RobotHandle>(new FakeArmHandle()),
  };
}

/** 로봇의 유일한 링크 바디 pose (noUncheckedIndexedAccess 대응 가드) */
function soleLinkPose(world: RapierWorld): Pose {
  const bodies = world.bodiesOfEntity(ROBOT_ID);
  expect(bodies).toHaveLength(1);
  const bodyId = bodies[0];
  if (bodyId === undefined) throw new Error('로봇 링크 바디가 없습니다');
  return world.getPose(bodyId);
}

beforeAll(async () => {
  await initPhysics();
});

// ── 회귀 테스트 ─────────────────────────────────────────────────────

describe('로봇 링크 바디 — fresh load / reset() 스텝 전 상태 동일성', () => {
  it('build 직후(스텝 전) 링크 바디가 home FK pose에 있다 (URDF 초기 자세 아님 — 첫 스텝 스윕 0)', async () => {
    const world = new RapierWorld(SCENE_SPEC.gravity, SCENE_SPEC.timestepHz);
    try {
      await new SceneLoader(world, makeRenderApi(), new RenderSync(world)).build(SCENE_SPEC);

      const pose = soleLinkPose(world);
      expect(pose.position[0]).toBeCloseTo(HOME_J1_RAD, POSITION_DECIMALS);
      expect(pose.position[1]).toBeCloseTo(L1_BASE_Y_M, POSITION_DECIMALS);
      expect(pose.position[2]).toBeCloseTo(0, POSITION_DECIMALS);
    } finally {
      world.free();
    }
  });

  it('reset() 직후(스텝 전) 물리 pose가 home FK로 즉시 복원된다 — fresh load와 동일', async () => {
    const world = new RapierWorld(SCENE_SPEC.gravity, SCENE_SPEC.timestepHz);
    try {
      const handle = await new SceneLoader(world, makeRenderApi(), new RenderSync(world))
        .build(SCENE_SPEC);
      const freshPose = soleLinkPose(world);

      // Engine의 preStep(tickAll) → step 순서를 재현해 관절을 home 밖으로 구동
      const binding = handle.robots.get(ROBOT_ID);
      binding.setJoint('j1', DRIVE_J1_RAD);
      for (let i = 0; i < DRIVE_STEPS; i += 1) {
        handle.robots.tickAll();
        world.step();
      }
      expect(soleLinkPose(world).position[0]).toBeCloseTo(DRIVE_J1_RAD, POSITION_DECIMALS);

      // 수정 전에는 reset() 후에도 물리 pose가 구동 pose에 남아(다음 스텝까지)
      // 시각(home)과 어긋났고, 다음 스텝이 구동 pose → home 스윕이 됐다.
      handle.reset();
      const resetPose = soleLinkPose(world);
      expect(resetPose.position[0]).toBeCloseTo(HOME_J1_RAD, POSITION_DECIMALS);
      expect(resetPose.position[1]).toBeCloseTo(L1_BASE_Y_M, POSITION_DECIMALS);
      expect(resetPose.position[2]).toBeCloseTo(0, POSITION_DECIMALS);

      // fresh load 직후와 reset 직후의 스텝 전 상태가 완전히 동일 (결정론적 재생)
      expect(resetPose).toStrictEqual(freshPose);
    } finally {
      world.free();
    }
  });
});
