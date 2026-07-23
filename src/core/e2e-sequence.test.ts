// core/e2e-sequence.test.ts — Phase 4+5 통합: 실 Rapier + ControlPlayer + CollisionMonitor
//
// 검증 대상 (ROADMAP Phase 5 게이트 "waitForCollision이 실제 충돌에 동기화"):
// 1. 가짜 FK 로봇(단일 관절 j1 → 링크 x = j1 × 1m)을 moveJoints로 구동하고, 낙하하는
//    동적 박스가 배리어 활성화 "이후"에 링크 위로 떨어져 접촉 → waitForCollision이
//    실제 start 이벤트로 해제된다(timeout 경고 없음).
//    해제 시점: 접촉 start는 tick c의 world.step에서 발생·기록되고, player는 그 "다음"
//    tick(c+1)의 preStep에서 이력을 조회해 해제한다 — preStep → step 순서 계약
//    (ARCHITECTURE §5)에 내재한 1 tick 지연이며 결정론적이다.
// 2. 결정론적 재생: 전체 리셋(박스 teleport + applyHome/teleportLinksToFk + player.reset
//    + monitor.clear) 후 재실행하면 충돌 timeSec과 setJoints 호출 로그가 완전 동일하다
//    (SIMULATION §6).
// 3. CCD 관측 (ROADMAP Phase 4 게이트 항목 "빠른 링크 vs 얇은 사물"): 측정 결과를
//    정직하게 문서화한다 — Rapier 0.14에서 kinematicPosition 바디의 한 tick 대변위
//    스윕은 wall/링크 어느 쪽에 ccd:true를 줘도 얇은 벽 통과를 감지하지 못한다
//    (이벤트 0건 — 터널링). 끝 pose가 겹칠 때만 discrete narrow-phase가 잡는다.
//    회피책: 감지 벽/센서 존 두께를 최대 tick당 변위 이상으로 설계 (EXPERIMENTS.md).
//
// 엔진 tick 순서를 그대로 재현한다: player.step(t, dt) → binding.tick()(FK → kinematic
// push) → world.step() → monitor.dispatch(events, t) → t += dt.

import { beforeAll, describe, expect, it } from 'vitest';
import { CollisionMonitor } from './collision';
import { collisionQueryFromMonitor, robotApiFromRegistry } from './control/adapters';
import { ControlPlayer } from './control/player';
import type { RobotApi } from './control/steps';
import { RobotBinding, RobotRegistry } from './robots';
import type { RobotFkView } from './robot-types';
import { initPhysics, RapierWorld } from './world';
import type { BodyId, Pose } from './types';
import type { ColliderSpec, ControlSequence, Vec3 } from '../schema/types';

// ── 테스트 상수 (매직넘버 금지 — CLAUDE.md §4) ──────────────────────

const TIMESTEP_HZ = 240;
const GRAVITY: Vec3 = [0, -9.81, 0];
const ZERO_GRAVITY: Vec3 = [0, 0, 0];
const IDENTITY_QUAT: [number, number, number, number] = [0, 0, 0, 1];

/** 가짜 FK: 링크 x = j1 × JOINT_TO_X_M (단위 프리즘 관절) */
const JOINT_TO_X_M = 1.0;
const LINK_HALF_M = 0.05;
const BOX_HALF_M = 0.05;

/** moveJoints 목표/시간: j1 0 → 0.6 (링크 x 0 → 0.6m), 1초 */
const MOVE_TARGET_J1 = 0.6;
const MOVE_DURATION_SEC = 1.0;
/** moveJoints가 완료되는 tick 수 (elapsed = (k+1)·dt ≥ 1.0 → k = 239) */
const MOVE_TICKS = MOVE_DURATION_SEC * TIMESTEP_HZ;

/**
 * 낙하 박스: x = 링크 최종 위치(0.6) 위에서 gravityScale 0.1(유효 g=0.981)로 낙하.
 * 접촉(박스 중심 y = 링크 top + BOX_HALF = 0.1)까지 낙하 거리 0.706m → t ≈ 1.2s
 * — 배리어 활성화(t = 1.0s) "이후"에 접촉이 시작되도록 설계 (mark-이후 이벤트만
 * happenedSince에 잡히는 collision.ts 계약).
 */
const BOX_START: Vec3 = [MOVE_TARGET_J1 * JOINT_TO_X_M, 0.806, 0];
const BOX_GRAVITY_SCALE = 0.1;
/** 접촉 판정 높이: 링크 top(0.05) + 박스 half(0.05) */
const CONTACT_CENTER_Y_M = LINK_HALF_M + BOX_HALF_M;

/** 시퀀스 전체(≈1.4s) + 여유 실행 tick 수 */
const RUN_TICKS = 2 * TIMESTEP_HZ;
/** waitForCollision step 인덱스 (steps: [moveJoints, waitForCollision, wait]) */
const WAIT_INDEX = 1;
const TAIL_WAIT_SEC = 0.2;

const ROBOT_ID = 'bot';
const BOX_ID = 'box';

const LINK_COLLIDER: ColliderSpec = {
  shape: { kind: 'box', halfExtents: [LINK_HALF_M, LINK_HALF_M, LINK_HALF_M] },
  group: 'ROBOT',
  collidesWith: ['ENV', 'OBJECT'],
  emitEvents: true,
};

const BOX_COLLIDER: ColliderSpec = {
  shape: { kind: 'box', halfExtents: [BOX_HALF_M, BOX_HALF_M, BOX_HALF_M] },
  group: 'OBJECT',
  collidesWith: ['ENV', 'ROBOT'],
  emitEvents: true,
};

const SEQUENCE: ControlSequence = {
  id: 'e2e-touch',
  robot: ROBOT_ID,
  loop: false,
  steps: [
    { kind: 'moveJoints', targets: { j1: MOVE_TARGET_J1 }, durationSec: MOVE_DURATION_SEC },
    { kind: 'waitForCollision', between: [ROBOT_ID, BOX_ID] }, // timeout 없음 — 경고 불가 조건
    { kind: 'wait', durationSec: TAIL_WAIT_SEC },
  ],
};

// ── 가짜 FK 로봇 (단일 링크가 +x로 j1에 비례 이동) ──────────────────

function makeLineFk(): RobotFkView {
  let j1 = 0;
  return {
    joints: [{ name: 'j1', type: 'prismatic', initial: 0, limits: [0, 2] }],
    setJointValues: (values) => {
      const next = values['j1'];
      if (next !== undefined) j1 = next;
    },
    readLinkPoses: () =>
      new Map<string, Pose>([
        ['link', { position: [j1 * JOINT_TO_X_M, 0, 0], rotation: IDENTITY_QUAT }],
      ]),
    linkColliders: new Map([['link', [{ shape: LINK_COLLIDER.shape }]]]),
  };
}

// ── 실행 하네스 ─────────────────────────────────────────────────────

interface Rig {
  world: RapierWorld;
  binding: RobotBinding;
  boxBody: BodyId;
  monitor: CollisionMonitor;
  player: ControlPlayer;
  setJointsLog: Array<{ robot: string; values: Record<string, number> }>;
  warnings: string[];
}

function buildRig(): Rig {
  const world = new RapierWorld(GRAVITY, TIMESTEP_HZ);

  const linkBody = world.createBody(ROBOT_ID, {
    bodyType: 'kinematicPosition',
    position: [0, 0, 0],
  });
  world.createCollider(linkBody, LINK_COLLIDER, ROBOT_ID);

  const boxBody = world.createBody(BOX_ID, {
    bodyType: 'dynamic',
    position: BOX_START,
    gravityScale: BOX_GRAVITY_SCALE,
  });
  world.createCollider(boxBody, BOX_COLLIDER, BOX_ID);

  const binding = new RobotBinding({
    entityId: ROBOT_ID,
    fk: makeLineFk(),
    world,
    linkBodies: new Map([['link', linkBody]]),
  });
  const registry = new RobotRegistry();
  registry.add(binding);

  const monitor = new CollisionMonitor();
  // setJoints 호출 로그 래퍼 — 결정론 비교 대상 (main.ts와 동일 어댑터 위에 기록만 추가)
  const baseApi = robotApiFromRegistry(registry, () => undefined);
  const setJointsLog: Rig['setJointsLog'] = [];
  const robotApi: RobotApi = {
    readJoints: baseApi.readJoints,
    gripperConfig: baseApi.gripperConfig,
    setJoints: (robot, values) => {
      setJointsLog.push({ robot, values: { ...values } });
      baseApi.setJoints(robot, values);
    },
  };

  const warnings: string[] = [];
  const player = new ControlPlayer({
    robots: robotApi,
    collision: collisionQueryFromMonitor(monitor),
    warn: (msg) => warnings.push(msg),
  });

  return { world, binding, boxBody, monitor, player, setJointsLog, warnings };
}

interface RunResult {
  /** start 이벤트(쌍 무관 전부)의 timeSec 목록 */
  collisionStartTimes: number[];
  /** 접촉 start가 처음 관측된 tick (world.step 기준) */
  contactStartTick: number;
  /** player 커서가 waitForCollision을 지나간 tick (player.step 기준) */
  releaseTick: number;
  setJointsLog: Rig['setJointsLog'];
}

/** 엔진 tick 순서 재현: player.step → binding.tick → world.step → monitor.dispatch */
function runOnce(rig: Rig): RunResult {
  const dt = rig.world.fixedDtSec;
  const collisionStartTimes: number[] = [];
  const unsubscribe = rig.monitor.subscribe((e) => {
    if (e.phase === 'start') collisionStartTimes.push(e.timeSec);
  });

  let contactStartTick = -1;
  let releaseTick = -1;
  let t = 0;
  for (let tick = 0; tick < RUN_TICKS; tick += 1) {
    const indexBefore = rig.player.currentStepIndex;
    rig.player.step(t, dt);
    if (releaseTick < 0 && indexBefore === WAIT_INDEX && rig.player.currentStepIndex > WAIT_INDEX) {
      releaseTick = tick;
    }
    rig.binding.tick();
    const events = rig.world.step();
    rig.monitor.dispatch(events, t);
    if (contactStartTick < 0 && events.some((e) => e.phase === 'start')) {
      contactStartTick = tick;
    }
    t += dt;
  }

  unsubscribe();
  return {
    collisionStartTimes,
    contactStartTick,
    releaseTick,
    setJointsLog: [...rig.setJointsLog],
  };
}

beforeAll(async () => {
  await initPhysics();
});

// ── 1+2. 시퀀스 재생 + waitForCollision 배리어 + 결정론 ─────────────

describe('e2e — 실 Rapier + ControlPlayer + CollisionMonitor 통합 (Phase 4+5)', () => {
  it('waitForCollision이 실제 접촉 start의 다음 tick에 해제되고(timeout 경고 없음), 전체 리셋 후 재실행이 완전 동일하다', () => {
    const rig = buildRig();
    try {
      rig.player.load(SEQUENCE);
      const run1 = runOnce(rig);

      // 접촉은 배리어 활성화(moveJoints 완료 tick = MOVE_TICKS - 1) 이후에 시작된다
      expect(run1.contactStartTick).toBeGreaterThanOrEqual(MOVE_TICKS);
      // 해제는 접촉 start "다음" tick의 player.step — 파일 헤더의 1 tick 지연 계약
      expect(run1.releaseTick).toBe(run1.contactStartTick + 1);
      // timeout 경고 없음 — 배리어가 시간 초과가 아니라 실제 이벤트로 해제되었다
      expect(rig.warnings).toEqual([]);
      // 시퀀스 완주
      expect(rig.player.status).toBe('done');
      expect(rig.player.currentStepIndex).toBe(SEQUENCE.steps.length);
      expect(run1.collisionStartTimes.length).toBeGreaterThanOrEqual(1);

      // ── 전체 리셋: 박스 teleport + 로봇 home/FK 정렬 + player/monitor 리셋 ──
      rig.world.teleport(rig.boxBody, { position: BOX_START, rotation: IDENTITY_QUAT });
      rig.binding.applyHome();
      rig.binding.teleportLinksToFk();
      rig.binding.tick();
      rig.player.reset();
      rig.monitor.clear();
      rig.setJointsLog.length = 0;

      const run2 = runOnce(rig);

      // 결정론 (SIMULATION §6): 동일 시퀀스 + 동일 초기 상태 → 동일 충돌 시점·동일 로그
      expect(run2.collisionStartTimes).toEqual(run1.collisionStartTimes);
      expect(run2.contactStartTick).toBe(run1.contactStartTick);
      expect(run2.releaseTick).toBe(run1.releaseTick);
      expect(run2.setJointsLog).toEqual(run1.setJointsLog);
      expect(rig.warnings).toEqual([]);
      expect(rig.player.status).toBe('done');

      // 접촉 시점 물리 타당성: 박스가 링크 위에 실제로 닿는 높이까지 낙하했다
      const boxPose = rig.world.getPose(rig.boxBody);
      expect(boxPose.position[1]).toBeLessThanOrEqual(CONTACT_CENTER_Y_M + BOX_HALF_M);
    } finally {
      rig.world.free();
    }
  });
});

// ── 3. CCD 관측 (ROADMAP Phase 4 게이트 항목) — 측정된 한계의 문서화 ─

/** 얇은 벽 두께(half): 0.005m 벽 = half 0.0025m */
const THIN_WALL_HALF_X_M = 0.0025;
const WALL_X_M = 0.5;
/** 한 tick 스윕 목표 — 벽을 완전히 지나쳐 반대편으로 (1.5m/tick = 360m/s) */
const SWEEP_PAST_J1 = 1.5;
/** 끝 pose가 벽과 겹치는 목표 (discrete 검출 대조군) */
const SWEEP_INTO_J1 = 0.5;
const CCD_SETTLE_TICKS = 10;

const THIN_WALL_COLLIDER: ColliderSpec = {
  shape: { kind: 'box', halfExtents: [THIN_WALL_HALF_X_M, 0.5, 0.5] },
  group: 'ENV',
  collidesWith: ['ROBOT'],
  emitEvents: true,
  ccd: true, // 과제 조건: 벽 collider에 ccd — 아래 관측대로 kinematic 스윕에는 무효
};

function buildCcdRig(): { world: RapierWorld; binding: RobotBinding } {
  const world = new RapierWorld(ZERO_GRAVITY, TIMESTEP_HZ);
  const wallBody = world.createBody('wall', { bodyType: 'fixed', position: [WALL_X_M, 0, 0] });
  world.createCollider(wallBody, THIN_WALL_COLLIDER, 'wall');

  const linkBody = world.createBody(ROBOT_ID, {
    bodyType: 'kinematicPosition',
    position: [0, 0, 0],
    ccd: true, // 이동체 쪽에도 ccd — 역시 무효(아래 관측)
  });
  world.createCollider(linkBody, { ...LINK_COLLIDER, ccd: true }, ROBOT_ID);

  const binding = new RobotBinding({
    entityId: ROBOT_ID,
    fk: makeLineFk(),
    world,
    linkBodies: new Map([['link', linkBody]]),
  });
  return { world, binding };
}

describe('CCD — 빠른 kinematic 링크 vs 얇은 벽 (측정된 한계의 정직한 기록)', () => {
  it('측정: kinematic 한 tick 대변위 스윕은 ccd:true(벽·링크 모두)여도 얇은 벽 통과를 감지하지 못한다 — 터널링', () => {
    const { world, binding } = buildCcdRig();
    try {
      for (let i = 0; i < CCD_SETTLE_TICKS; i += 1) world.step();

      // 관절값 한 번에 1.5 → 링크가 한 tick에 벽(x=0.5)을 넘어 x=1.5로 스윕
      binding.setJoint('j1', SWEEP_PAST_J1);
      binding.tick();
      const events = [];
      for (let i = 0; i < CCD_SETTLE_TICKS; i += 1) events.push(...world.step());

      // ★ 측정된 한계 (Rapier 0.14, EXPERIMENTS.md 기록): CCD 스윕 이벤트 0건.
      // 이 어서션은 "기대 동작"이 아니라 현재 엔진의 실측 동작을 고정하는 회귀 문서다
      // — 향후 Rapier가 kinematic CCD를 지원하게 되면 이 테스트가 깨져 알려준다.
      // 회피책: 감지 벽/SENSOR_ZONE 두께 ≥ 최대 tick당 변위로 설계.
      expect(events).toEqual([]);
      // 링크는 실제로 벽 반대편으로 관통 이동했다 (터널링 확인)
      const linkPose = world.getPose(world.bodiesOfEntity(ROBOT_ID)[0] as BodyId);
      expect(linkPose.position[0]).toBeCloseTo(SWEEP_PAST_J1 * JOINT_TO_X_M, 5);
    } finally {
      world.free();
    }
  });

  it('대조군: 끝 pose가 벽과 겹치는 스윕은 discrete narrow-phase가 start 이벤트로 감지한다', () => {
    const { world, binding } = buildCcdRig();
    try {
      for (let i = 0; i < CCD_SETTLE_TICKS; i += 1) world.step();

      binding.setJoint('j1', SWEEP_INTO_J1); // 링크 끝 pose x=0.5 = 벽 위치(겹침)
      binding.tick();
      const events = [];
      for (let i = 0; i < CCD_SETTLE_TICKS; i += 1) events.push(...world.step());

      expect(
        events.some(
          (e) =>
            e.phase === 'start' &&
            [e.a, e.b].sort().join(',') === [ROBOT_ID, 'wall'].sort().join(','),
        ),
      ).toBe(true);
    } finally {
      world.free();
    }
  });
});
