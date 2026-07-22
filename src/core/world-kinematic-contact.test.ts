// core/world-kinematic-contact.test.ts — kinematic 바디 접촉 감지 회귀 테스트
//
// 리뷰 지적(physics): Rapier의 기본 ActiveCollisionTypes(DEFAULT=15)는 DYNAMIC_* 쌍만
// 활성화한다. 로봇 링크는 kinematicPosition, 바닥(ENV)은 fixed이므로 기본값 그대로면
// KINEMATIC_FIXED(8704) 쌍이 narrow-phase에서 통째로 건너뛰어져 — 그룹 필터와
// emitEvents가 다 맞아도 — CLAUDE.md §5의 핵심 쌍 ROBOT×ENV 접촉 이벤트가 절대
// 발행되지 않았다. selfCollision(KINEMATIC_KINEMATIC=52224)도 같은 원인의 죽은 옵션.
// 수정: RapierWorld.createCollider가 kinematic 바디의 collider에
// DEFAULT|KINEMATIC_FIXED|KINEMATIC_KINEMATIC을 설정한다 — "어떤 쌍이 상호작용하는가"는
// 충돌 그룹(collidesWith)이 유일하게 결정한다 (world.ts 상수 주석 참조).

import { beforeAll, describe, expect, it } from 'vitest';
import { initPhysics, RapierWorld } from './world';
import type { ContactEvent, Pose } from './types';
import type { ColliderSpec, Quat, Vec3 } from '../schema/types';

// ── 테스트 상수 (매직넘버 금지 — CLAUDE.md §4) ──────────────────────

const GRAVITY: Vec3 = [0, -9.81, 0];
const TIMESTEP_HZ = 240;
const IDENTITY_QUAT: Quat = [0, 0, 0, 1];

const GROUND_HALF_EXTENTS_M: Vec3 = [2, 0.1, 2];
const GROUND_CENTER_Y_M = -GROUND_HALF_EXTENTS_M[1]; // 윗면 y=0

const LINK_HALF_M = 0.05;
/** 접촉 전 호버링 높이 — 바닥과 확실히 떨어져 있다 */
const LINK_HOVER_Y_M = 0.5;
/** 접촉 목표 높이 — 바닥 윗면(y=0)을 2cm 관통 */
const LINK_TOUCH_Y_M = LINK_HALF_M - 0.02;
/** 목표 도달 + 이벤트 발행 여유 스텝 수 */
const SETTLE_STEPS = 10;

/** 로봇 링크 collider 규약 (scene-loader와 동일: ROBOT 그룹, emitEvents) */
const ROBOT_LINK_COLLIDER: ColliderSpec = {
  shape: { kind: 'box', halfExtents: [LINK_HALF_M, LINK_HALF_M, LINK_HALF_M] },
  group: 'ROBOT',
  collidesWith: ['ENV', 'OBJECT'],
  emitEvents: true,
};

/** selfCollision=true 규약: 필터에 ROBOT 포함 */
const ROBOT_LINK_COLLIDER_SELF: ColliderSpec = {
  ...ROBOT_LINK_COLLIDER,
  collidesWith: ['ENV', 'OBJECT', 'ROBOT'],
};

// ── 헬퍼 ────────────────────────────────────────────────────────────

function pairOf(e: ContactEvent): string {
  return [e.a, e.b].sort().join(',');
}

function poseAt(yM: number): Pose {
  return { position: [0, yM, 0], rotation: IDENTITY_QUAT };
}

function stepCollect(world: RapierWorld, steps: number): ContactEvent[] {
  const events: ContactEvent[] = [];
  for (let i = 0; i < steps; i += 1) events.push(...world.step());
  return events;
}

beforeAll(async () => {
  await initPhysics();
});

// ── 회귀 테스트 ─────────────────────────────────────────────────────

describe('RapierWorld — kinematic 바디 접촉 감지 (ActiveCollisionTypes 정규화)', () => {
  it('kinematic ROBOT 링크 × fixed ENV 바닥: start/stop 접촉 이벤트가 발행된다 (CLAUDE.md §5 핵심 쌍)', () => {
    const world = new RapierWorld(GRAVITY, TIMESTEP_HZ);
    try {
      const groundBody = world.createBody('ground', {
        bodyType: 'fixed',
        position: [0, GROUND_CENTER_Y_M, 0],
      });
      world.createCollider(groundBody, {
        shape: { kind: 'box', halfExtents: GROUND_HALF_EXTENTS_M },
        group: 'ENV',
        collidesWith: ['ROBOT', 'OBJECT'],
      }, 'ground');

      const linkBody = world.createBody('arm', {
        bodyType: 'kinematicPosition',
        position: [0, LINK_HOVER_Y_M, 0],
      });
      world.createCollider(linkBody, ROBOT_LINK_COLLIDER, 'arm');

      // 호버링 중에는 접촉 없음
      world.setKinematicPose(linkBody, poseAt(LINK_HOVER_Y_M));
      expect(stepCollect(world, SETTLE_STEPS)).toEqual([]);

      // 바닥으로 내려 접촉 — 수정 전에는 KINEMATIC_FIXED 쌍이 narrow-phase에서
      // 제외되어 이 start 이벤트가 절대 나오지 않았다 (그룹/emitEvents 무관)
      world.setKinematicPose(linkBody, poseAt(LINK_TOUCH_Y_M));
      const touchEvents = stepCollect(world, SETTLE_STEPS);
      const starts = touchEvents.filter((e) => e.phase === 'start');
      expect(starts.length).toBeGreaterThanOrEqual(1);
      for (const e of starts) {
        expect(pairOf(e)).toBe('arm,ground');
        expect(e.kind).toBe('contact');
      }

      // 다시 들어올리면 stop
      world.setKinematicPose(linkBody, poseAt(LINK_HOVER_Y_M));
      const liftEvents = stepCollect(world, SETTLE_STEPS);
      expect(
        liftEvents.some((e) => e.phase === 'stop' && pairOf(e) === 'arm,ground'),
      ).toBe(true);
    } finally {
      world.free();
    }
  });

  it('kinematic × kinematic (selfCollision=true 필터): 겹침에서 start 이벤트가 발행된다', () => {
    const world = new RapierWorld(GRAVITY, TIMESTEP_HZ);
    try {
      const bodyA = world.createBody('link_a', {
        bodyType: 'kinematicPosition',
        position: [0, LINK_HOVER_Y_M, 0],
      });
      world.createCollider(bodyA, ROBOT_LINK_COLLIDER_SELF, 'link_a');

      // 서로 떨어진 위치에서 시작
      const apartXM = 4 * LINK_HALF_M;
      const bodyB = world.createBody('link_b', {
        bodyType: 'kinematicPosition',
        position: [apartXM, LINK_HOVER_Y_M, 0],
      });
      world.createCollider(bodyB, ROBOT_LINK_COLLIDER_SELF, 'link_b');

      world.setKinematicPose(bodyA, poseAt(LINK_HOVER_Y_M));
      world.setKinematicPose(bodyB, {
        position: [apartXM, LINK_HOVER_Y_M, 0],
        rotation: IDENTITY_QUAT,
      });
      expect(stepCollect(world, SETTLE_STEPS)).toEqual([]);

      // B를 A 위치로 이동해 겹침 — KINEMATIC_KINEMATIC 쌍 활성화 회귀
      world.setKinematicPose(bodyB, poseAt(LINK_HOVER_Y_M));
      const events = stepCollect(world, SETTLE_STEPS);
      expect(
        events.some((e) => e.phase === 'start' && pairOf(e) === 'link_a,link_b'),
      ).toBe(true);
    } finally {
      world.free();
    }
  });

  it('selfCollision=false 필터(collidesWith에 ROBOT 없음)면 kinematic 링크끼리 겹쳐도 이벤트가 없다', () => {
    const world = new RapierWorld(GRAVITY, TIMESTEP_HZ);
    try {
      // 그룹 필터가 유일한 게이트임을 확인 — 쌍 타입은 활성화됐지만 필터가 거른다
      const bodyA = world.createBody('link_a', {
        bodyType: 'kinematicPosition',
        position: [0, LINK_HOVER_Y_M, 0],
      });
      world.createCollider(bodyA, ROBOT_LINK_COLLIDER, 'link_a');
      const bodyB = world.createBody('link_b', {
        bodyType: 'kinematicPosition',
        position: [0, LINK_HOVER_Y_M, 0], // 처음부터 겹침
      });
      world.createCollider(bodyB, ROBOT_LINK_COLLIDER, 'link_b');

      world.setKinematicPose(bodyA, poseAt(LINK_HOVER_Y_M));
      world.setKinematicPose(bodyB, poseAt(LINK_HOVER_Y_M));
      expect(stepCollect(world, SETTLE_STEPS)).toEqual([]);
    } finally {
      world.free();
    }
  });
});
