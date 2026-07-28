// core/world-robot-robot-contact.test.ts — 로봇↔로봇 충돌 회귀 테스트
//
// 결함(사용자 보고): 로봇팔 두 대를 서로 닿게 배치해도 그냥 겹치며 통과했다.
// 원인: scene-loader가 selfCollision=false(기본)일 때 로봇 링크 collider의
// collidesWith에서 ROBOT 그룹을 통째로 제외했다. 모든 로봇 링크는 같은 ROBOT 그룹이라
// 필터에 ROBOT이 없으면 "자기 링크"뿐 아니라 "다른 로봇"과도 narrow-phase에 도달하지
// 못한다 — 그룹 비트마스크는 로봇 개체를 구분하지 못하기 때문이다.
//
// 수정: 로봇 링크는 항상 ROBOT을 필터에 포함하고(다른 로봇과 충돌), 자기 링크 억제는
// 엔티티 단위로 처리한다(RapierWorld.setSelfContactEnabled). 로봇 한 대의 모든 링크는
// 하나의 EntityId를 공유하므로 "같은 엔티티 접촉" = self-collision이다.

import { beforeAll, describe, expect, it } from 'vitest';
import { initPhysics, RapierWorld } from './world';
import type { ContactEvent, Pose } from './types';
import type { ColliderSpec, Quat, Vec3 } from '../schema/types';

const GRAVITY: Vec3 = [0, -9.81, 0];
const TIMESTEP_HZ = 240;
const IDENTITY_QUAT: Quat = [0, 0, 0, 1];

const LINK_HALF_M = 0.05;
const HOVER_Y_M = 0.5;
/** 겹치지 않는 시작 간격 (반크기의 4배 — 확실히 분리) */
const APART_X_M = 4 * LINK_HALF_M;
const SETTLE_STEPS = 10;

/** 수정 후 scene-loader 규약과 동일: ROBOT 그룹 + 필터에 ROBOT 포함 */
const ROBOT_LINK_COLLIDER: ColliderSpec = {
  shape: { kind: 'box', halfExtents: [LINK_HALF_M, LINK_HALF_M, LINK_HALF_M] },
  group: 'ROBOT',
  collidesWith: ['ENV', 'OBJECT', 'ROBOT'],
  emitEvents: true,
};

function pairOf(e: ContactEvent): string {
  return [e.a, e.b].sort().join(',');
}

function poseAt(xM: number): Pose {
  return { position: [xM, HOVER_Y_M, 0], rotation: IDENTITY_QUAT };
}

function stepCollect(world: RapierWorld, steps: number): ContactEvent[] {
  const events: ContactEvent[] = [];
  for (let i = 0; i < steps; i += 1) events.push(...world.step());
  return events;
}

/** 엔티티 하나에 링크(kinematic 바디+collider) 하나를 만든다 */
function addLink(world: RapierWorld, entityId: string, xM: number): number {
  const bodyId = world.createBody(entityId, {
    bodyType: 'kinematicPosition',
    position: [xM, HOVER_Y_M, 0],
  });
  world.createCollider(bodyId, ROBOT_LINK_COLLIDER, entityId);
  return bodyId;
}

beforeAll(async () => {
  await initPhysics();
});

describe('RapierWorld — 로봇↔로봇 충돌과 자기 접촉 분리', () => {
  it('다른 로봇(다른 EntityId)의 링크가 겹치면 start 이벤트가 발행된다 ★ 회귀', () => {
    const world = new RapierWorld(GRAVITY, TIMESTEP_HZ);
    try {
      // 두 로봇 모두 기본값(자기 접촉 억제) — 그래도 서로는 충돌해야 한다
      world.setSelfContactEnabled('arm_a', false);
      world.setSelfContactEnabled('arm_b', false);

      const linkA = addLink(world, 'arm_a', 0);
      const linkB = addLink(world, 'arm_b', APART_X_M);

      // 떨어져 있을 땐 이벤트 없음
      world.setKinematicPose(linkA, poseAt(0));
      world.setKinematicPose(linkB, poseAt(APART_X_M));
      expect(stepCollect(world, SETTLE_STEPS)).toEqual([]);

      // arm_b의 링크를 arm_a 위치로 이동 → 겹침
      world.setKinematicPose(linkB, poseAt(0));
      const events = stepCollect(world, SETTLE_STEPS);

      expect(events.some((e) => e.phase === 'start' && pairOf(e) === 'arm_a,arm_b')).toBe(true);
      expect(events.every((e) => e.kind === 'contact')).toBe(true);
    } finally {
      world.free();
    }
  });

  it('분리되면 stop 이벤트로 접촉 해제가 보고된다', () => {
    const world = new RapierWorld(GRAVITY, TIMESTEP_HZ);
    try {
      const linkA = addLink(world, 'arm_a', 0);
      const linkB = addLink(world, 'arm_b', 0); // 처음부터 겹침
      world.setKinematicPose(linkA, poseAt(0));
      world.setKinematicPose(linkB, poseAt(0));
      const started = stepCollect(world, SETTLE_STEPS);
      expect(started.some((e) => e.phase === 'start' && pairOf(e) === 'arm_a,arm_b')).toBe(true);

      world.setKinematicPose(linkB, poseAt(APART_X_M)); // 떼어냄
      const stopped = stepCollect(world, SETTLE_STEPS);
      expect(stopped.some((e) => e.phase === 'stop' && pairOf(e) === 'arm_a,arm_b')).toBe(true);
    } finally {
      world.free();
    }
  });

  it('같은 로봇(같은 EntityId)의 링크끼리는 기본적으로 이벤트를 발행하지 않는다 (인접 링크 노이즈 억제)', () => {
    const world = new RapierWorld(GRAVITY, TIMESTEP_HZ);
    try {
      // 미등록 = 억제(기본). URDF 인접 링크는 구조상 항상 겹친다.
      const linkA = addLink(world, 'arm', 0);
      const linkB = addLink(world, 'arm', 0); // 같은 엔티티, 겹친 상태
      world.setKinematicPose(linkA, poseAt(0));
      world.setKinematicPose(linkB, poseAt(0));

      expect(stepCollect(world, SETTLE_STEPS)).toEqual([]);
    } finally {
      world.free();
    }
  });

  it('setSelfContactEnabled(true)면 같은 엔티티 링크 접촉도 발행된다 (selfCollision 옵트인)', () => {
    const world = new RapierWorld(GRAVITY, TIMESTEP_HZ);
    try {
      world.setSelfContactEnabled('arm', true);
      const linkA = addLink(world, 'arm', 0);
      const linkB = addLink(world, 'arm', 0);
      world.setKinematicPose(linkA, poseAt(0));
      world.setKinematicPose(linkB, poseAt(0));

      const events = stepCollect(world, SETTLE_STEPS);
      expect(events.some((e) => e.phase === 'start' && e.a === 'arm' && e.b === 'arm')).toBe(true);
    } finally {
      world.free();
    }
  });

  it('자기 접촉 옵트인은 해당 엔티티에만 적용된다', () => {
    const world = new RapierWorld(GRAVITY, TIMESTEP_HZ);
    try {
      world.setSelfContactEnabled('arm_a', true); // arm_b는 기본(억제)
      const a1 = addLink(world, 'arm_a', 0);
      const a2 = addLink(world, 'arm_a', 0);
      const b1 = addLink(world, 'arm_b', APART_X_M);
      const b2 = addLink(world, 'arm_b', APART_X_M);
      for (const [body, x] of [[a1, 0], [a2, 0], [b1, APART_X_M], [b2, APART_X_M]] as const) {
        world.setKinematicPose(body, poseAt(x));
      }

      const events = stepCollect(world, SETTLE_STEPS);
      expect(events.some((e) => e.a === 'arm_a' && e.b === 'arm_a')).toBe(true);
      expect(events.some((e) => e.a === 'arm_b' && e.b === 'arm_b')).toBe(false);
    } finally {
      world.free();
    }
  });

  it('clear()는 자기 접촉 등록을 초기화한다 (씬 재로드 시 이전 설정이 새지 않는다)', () => {
    const world = new RapierWorld(GRAVITY, TIMESTEP_HZ);
    try {
      world.setSelfContactEnabled('arm', true);
      world.clear();

      // 같은 id로 재구축 — clear 이후에는 기본(억제)이어야 한다
      const linkA = addLink(world, 'arm', 0);
      const linkB = addLink(world, 'arm', 0);
      world.setKinematicPose(linkA, poseAt(0));
      world.setKinematicPose(linkB, poseAt(0));

      expect(stepCollect(world, SETTLE_STEPS)).toEqual([]);
    } finally {
      world.free();
    }
  });
});

describe('RapierWorld — 접촉점 보강 (충돌 시각화용)', () => {
  it('start 접촉 이벤트에 월드 접촉점과 법선이 실린다', () => {
    const world = new RapierWorld(GRAVITY, TIMESTEP_HZ);
    try {
      const linkA = addLink(world, 'arm_a', 0);
      const linkB = addLink(world, 'arm_b', APART_X_M);
      world.setKinematicPose(linkA, poseAt(0));
      world.setKinematicPose(linkB, poseAt(APART_X_M));
      stepCollect(world, SETTLE_STEPS);

      // 살짝만 겹치게 이동 — 접촉점이 두 링크 사이 경계 근처여야 한다
      const overlapXM = 2 * LINK_HALF_M - 0.01;
      world.setKinematicPose(linkB, poseAt(overlapXM));
      const events = stepCollect(world, SETTLE_STEPS);

      const start = events.find((e) => e.phase === 'start' && pairOf(e) === 'arm_a,arm_b');
      expect(start).toBeDefined();
      expect(start?.point).toBeDefined();
      expect(start?.normal).toBeDefined();
      // 접촉점은 두 박스 사이(x ≈ 0.04~0.06), y는 호버 높이 근처
      const [px, py] = start!.point!;
      expect(px).toBeGreaterThan(0);
      expect(px).toBeLessThan(overlapXM + LINK_HALF_M);
      expect(Math.abs(py - HOVER_Y_M)).toBeLessThan(2 * LINK_HALF_M);
      // 법선은 단위 벡터
      const n = start!.normal!;
      const len = Math.hypot(n[0], n[1], n[2]);
      expect(Math.abs(len - 1)).toBeLessThan(1e-3);
    } finally {
      world.free();
    }
  });
});
