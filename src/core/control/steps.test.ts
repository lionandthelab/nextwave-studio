// core/control/steps.test.ts — step 핸들러 순수 로직 단위 테스트
//
// 물리/렌더 없이 완결적으로 검증한다 (CLAUDE.md §4 "부수효과 격리"):
// - RobotApi는 setJoints 호출을 기록하고 내부 관절 상태에 반영하는 목(mock)
// - CollisionQuery는 happenedSince 응답을 스크립트할 수 있는 목
// 규범: docs/SIMULATION.md §3, docs/DATA_MODEL.md §6.

import { describe, expect, it } from 'vitest';
import { initStep, stepHandlers } from './steps';
import type { CollisionQuery, RobotApi, StepContext, StepOfKind } from './steps';

// ── 테스트 상수 (매직넘버 금지 — CLAUDE.md §4) ──────────────────────

const DEFAULT_ROBOT = 'arm';
const OTHER_ROBOT = 'arm2';

const DT_SEC = 0.1;

/** 그리퍼 설정: 평행 그리퍼 2관절, 열림 0.03m / 닫힘 0m */
const GRIPPER_OPEN_M = 0.03;
const GRIPPER_CLOSE_M = 0;
const GRIPPER_JOINTS = ['finger_l', 'finger_r'];

/** easeInOut(코사인 smoothstep) 기대값 — 구현과 같은 정의식에서 유도 (core/math.ts) */
const EASE_IN_OUT_AT_QUARTER = 0.5 - 0.5 * Math.cos(Math.PI * 0.25); // t=0.25
const FLOAT_DIGITS = 12;

// ── 목 RobotApi ─────────────────────────────────────────────────────

interface SetJointsCall {
  robot: string;
  values: Record<string, number>;
}

class MockRobotApi implements RobotApi {
  readonly setJointsCalls: SetJointsCall[] = [];
  readonly readJointsCalls: { robot: string; names?: string[] }[] = [];

  private readonly state = new Map<string, Map<string, number>>();
  private readonly grippers = new Map<
    string,
    { joints: string[]; open: number; close: number }
  >();

  constructor(initial: Record<string, Record<string, number>>) {
    for (const [robot, joints] of Object.entries(initial)) {
      this.state.set(robot, new Map(Object.entries(joints)));
    }
  }

  setGripper(robot: string, config: { joints: string[]; open: number; close: number }): void {
    this.grippers.set(robot, config);
  }

  readJoints(robot: string, names?: string[]): Record<string, number> {
    this.readJointsCalls.push({ robot, names: names ? [...names] : undefined });
    const joints = this.robotState(robot);
    if (!names) return Object.fromEntries(joints);
    const out: Record<string, number> = {};
    for (const name of names) {
      const value = joints.get(name);
      if (value === undefined) throw new Error(`mock: 관절 '${name}' 없음 (robot '${robot}')`);
      out[name] = value;
    }
    return out;
  }

  setJoints(robot: string, values: Record<string, number>): void {
    this.setJointsCalls.push({ robot, values: { ...values } });
    const joints = this.robotState(robot);
    for (const [name, value] of Object.entries(values)) joints.set(name, value);
  }

  gripperConfig(robot: string): { joints: string[]; open: number; close: number } | undefined {
    return this.grippers.get(robot);
  }

  /** 현재 관절값 조회 (검증용) */
  jointValue(robot: string, name: string): number {
    const value = this.robotState(robot).get(name);
    if (value === undefined) throw new Error(`mock: 관절 '${name}' 없음 (robot '${robot}')`);
    return value;
  }

  private robotState(robot: string): Map<string, number> {
    const joints = this.state.get(robot);
    if (!joints) throw new Error(`mock: 로봇 '${robot}' 없음`);
    return joints;
  }
}

// ── 목 CollisionQuery ───────────────────────────────────────────────

class MockCollisionQuery implements CollisionQuery {
  readonly markCalls: unknown[] = [];
  readonly happenedCalls: { mark: unknown; between: [string, string] }[] = [];
  /** 테스트가 스크립트하는 응답 (기본: 항상 미발생) */
  respond: (mark: unknown, between: [string, string]) => boolean = () => false;

  private nextMark = 0;

  mark(): unknown {
    const mark = this.nextMark;
    this.nextMark += 1;
    this.markCalls.push(mark);
    return mark;
  }

  happenedSince(mark: unknown, between: [string, string]): boolean {
    this.happenedCalls.push({ mark, between: [...between] });
    return this.respond(mark, between);
  }
}

// ── ctx 팩토리 ──────────────────────────────────────────────────────

interface TestRig {
  ctx: StepContext;
  robots: MockRobotApi;
  collision: MockCollisionQuery;
  warnings: string[];
}

function makeRig(initialJoints?: Record<string, Record<string, number>>): TestRig {
  const robots = new MockRobotApi(
    initialJoints ?? { [DEFAULT_ROBOT]: { j1: 0, j2: 0 }, [OTHER_ROBOT]: { j1: 0 } },
  );
  const collision = new MockCollisionQuery();
  const warnings: string[] = [];
  const ctx: StepContext = {
    robots,
    collision,
    defaultRobot: DEFAULT_ROBOT,
    warn: (msg) => warnings.push(msg),
  };
  return { ctx, robots, collision, warnings };
}

// ── setJoints ───────────────────────────────────────────────────────

describe('setJoints 핸들러', () => {
  it('한 번 적용하고 즉시 done — 기본 로봇(defaultRobot) 사용', () => {
    const { ctx, robots } = makeRig();
    const step: StepOfKind<'setJoints'> = { kind: 'setJoints', targets: { j1: 0.4, j2: -0.2 } };
    const state = stepHandlers.setJoints.init(step, ctx);

    expect(stepHandlers.setJoints.tick(state, ctx, DT_SEC)).toBe('done');
    expect(robots.setJointsCalls).toEqual([
      { robot: DEFAULT_ROBOT, values: { j1: 0.4, j2: -0.2 } },
    ]);
  });

  it('step.robot이 있으면 시퀀스 기본 로봇을 override한다', () => {
    const { ctx, robots } = makeRig();
    const step: StepOfKind<'setJoints'> = {
      kind: 'setJoints',
      robot: OTHER_ROBOT,
      targets: { j1: 1 },
    };
    const state = stepHandlers.setJoints.init(step, ctx);
    stepHandlers.setJoints.tick(state, ctx, DT_SEC);

    expect(robots.setJointsCalls[0]?.robot).toBe(OTHER_ROBOT);
  });
});

// ── moveJoints ──────────────────────────────────────────────────────

describe('moveJoints 핸들러', () => {
  it('linear: 알려진 dt에서 보간 궤적이 정확하다 (0.25/0.5/0.75 → 최종 정확값)', () => {
    const { ctx, robots } = makeRig();
    const step: StepOfKind<'moveJoints'> = {
      kind: 'moveJoints',
      targets: { j1: 1 },
      durationSec: 1,
      easing: 'linear',
    };
    const state = stepHandlers.moveJoints.init(step, ctx);

    const quarterDtSec = 0.25;
    expect(stepHandlers.moveJoints.tick(state, ctx, quarterDtSec)).toBe('running');
    expect(stepHandlers.moveJoints.tick(state, ctx, quarterDtSec)).toBe('running');
    expect(stepHandlers.moveJoints.tick(state, ctx, quarterDtSec)).toBe('running');
    expect(stepHandlers.moveJoints.tick(state, ctx, quarterDtSec)).toBe('done');

    const j1Trajectory = robots.setJointsCalls.map((c) => c.values['j1']);
    expect(j1Trajectory[0]).toBeCloseTo(0.25, FLOAT_DIGITS);
    expect(j1Trajectory[1]).toBeCloseTo(0.5, FLOAT_DIGITS);
    expect(j1Trajectory[2]).toBeCloseTo(0.75, FLOAT_DIGITS);
    expect(j1Trajectory[3]).toBe(1); // 마지막 tick은 목표값 그대로 (drift 없음)
  });

  it('easeInOut: 중간 지점 값이 코사인 곡선을 따른다 (t=0.25, t=0.5)', () => {
    const { ctx, robots } = makeRig();
    const step: StepOfKind<'moveJoints'> = {
      kind: 'moveJoints',
      targets: { j1: 1 },
      durationSec: 1,
      easing: 'easeInOut',
    };
    const state = stepHandlers.moveJoints.init(step, ctx);

    const quarterDtSec = 0.25;
    stepHandlers.moveJoints.tick(state, ctx, quarterDtSec); // t=0.25
    stepHandlers.moveJoints.tick(state, ctx, quarterDtSec); // t=0.5

    expect(robots.setJointsCalls[0]?.values['j1']).toBeCloseTo(
      EASE_IN_OUT_AT_QUARTER,
      FLOAT_DIGITS,
    );
    expect(robots.setJointsCalls[1]?.values['j1']).toBeCloseTo(0.5, FLOAT_DIGITS); // easeInOut(0.5)=0.5
  });

  it('마지막 tick은 부동소수 잔차 없이 정확한 목표값을 적용한다', () => {
    const { ctx, robots } = makeRig({ [DEFAULT_ROBOT]: { j1: 0.1 } });
    const exactTarget = 0.7777;
    const step: StepOfKind<'moveJoints'> = {
      kind: 'moveJoints',
      targets: { j1: exactTarget },
      durationSec: 0.3,
      easing: 'linear',
    };
    const state = stepHandlers.moveJoints.init(step, ctx);

    stepHandlers.moveJoints.tick(state, ctx, DT_SEC);
    stepHandlers.moveJoints.tick(state, ctx, DT_SEC);
    expect(stepHandlers.moveJoints.tick(state, ctx, DT_SEC)).toBe('done');

    const last = robots.setJointsCalls.at(-1);
    expect(last?.values['j1']).toBe(exactTarget); // toBeCloseTo가 아니라 완전 일치
  });

  it('durationSec 0 → 첫 tick(dt=0이어도)에서 즉시 목표값 + done', () => {
    const { ctx, robots } = makeRig();
    const step: StepOfKind<'moveJoints'> = {
      kind: 'moveJoints',
      targets: { j1: 0.9 },
      durationSec: 0,
    };
    const state = stepHandlers.moveJoints.init(step, ctx);

    expect(stepHandlers.moveJoints.tick(state, ctx, 0)).toBe('done');
    expect(robots.setJointsCalls).toEqual([{ robot: DEFAULT_ROBOT, values: { j1: 0.9 } }]);
  });

  it('init 시점에 target 이름들로 시작값을 스냅샷한다 — 이후 외부 변경에 영향받지 않음', () => {
    const { ctx, robots } = makeRig();
    const step: StepOfKind<'moveJoints'> = {
      kind: 'moveJoints',
      targets: { j1: 1 },
      durationSec: 1,
      easing: 'linear',
    };
    const state = stepHandlers.moveJoints.init(step, ctx);

    // readJoints가 target 이름으로 호출됐는지 (SIMULATION §3 스냅샷 계약)
    expect(robots.readJointsCalls).toEqual([{ robot: DEFAULT_ROBOT, names: ['j1'] }]);

    // init 후 외부에서 관절을 바꿔도 보간은 스냅샷 기준
    robots.setJoints(DEFAULT_ROBOT, { j1: 5 });
    robots.setJointsCalls.length = 0;

    const halfDtSec = 0.5;
    stepHandlers.moveJoints.tick(state, ctx, halfDtSec); // t=0.5 → lerp(0, 1, 0.5)
    expect(robots.setJointsCalls[0]?.values['j1']).toBeCloseTo(0.5, FLOAT_DIGITS);
  });

  it('여러 관절을 각자의 시작값에서 동시에 보간한다', () => {
    const { ctx, robots } = makeRig({ [DEFAULT_ROBOT]: { j1: 0, j2: 1 } });
    const step: StepOfKind<'moveJoints'> = {
      kind: 'moveJoints',
      targets: { j1: 1, j2: 0 },
      durationSec: 1,
      easing: 'linear',
    };
    const state = stepHandlers.moveJoints.init(step, ctx);

    const halfDtSec = 0.5;
    stepHandlers.moveJoints.tick(state, ctx, halfDtSec);
    expect(robots.setJointsCalls[0]?.values['j1']).toBeCloseTo(0.5, FLOAT_DIGITS);
    expect(robots.setJointsCalls[0]?.values['j2']).toBeCloseTo(0.5, FLOAT_DIGITS);
  });
});

// ── gripper ─────────────────────────────────────────────────────────

describe('gripper 핸들러', () => {
  function makeGripperRig(): TestRig {
    const rig = makeRig({
      [DEFAULT_ROBOT]: { finger_l: GRIPPER_CLOSE_M, finger_r: GRIPPER_CLOSE_M },
    });
    rig.robots.setGripper(DEFAULT_ROBOT, {
      joints: [...GRIPPER_JOINTS],
      open: GRIPPER_OPEN_M,
      close: GRIPPER_CLOSE_M,
    });
    return rig;
  }

  it("'open' → 모든 gripper 관절에 open 값 (durationSec 미지정 → 즉시 done)", () => {
    const { ctx, robots } = makeGripperRig();
    const step: StepOfKind<'gripper'> = { kind: 'gripper', state: 'open' };
    const state = stepHandlers.gripper.init(step, ctx);

    expect(stepHandlers.gripper.tick(state, ctx, DT_SEC)).toBe('done');
    expect(robots.setJointsCalls).toEqual([
      { robot: DEFAULT_ROBOT, values: { finger_l: GRIPPER_OPEN_M, finger_r: GRIPPER_OPEN_M } },
    ]);
  });

  it("'close' → close 값", () => {
    const rig = makeGripperRig();
    rig.robots.setJoints(DEFAULT_ROBOT, { finger_l: GRIPPER_OPEN_M, finger_r: GRIPPER_OPEN_M });
    rig.robots.setJointsCalls.length = 0;

    const step: StepOfKind<'gripper'> = { kind: 'gripper', state: 'close' };
    const state = stepHandlers.gripper.init(step, rig.ctx);
    stepHandlers.gripper.tick(state, rig.ctx, DT_SEC);

    expect(rig.robots.setJointsCalls).toEqual([
      { robot: DEFAULT_ROBOT, values: { finger_l: GRIPPER_CLOSE_M, finger_r: GRIPPER_CLOSE_M } },
    ]);
  });

  it('숫자 0.5 → close + (open-close)*0.5', () => {
    const { ctx, robots } = makeGripperRig();
    const step: StepOfKind<'gripper'> = { kind: 'gripper', state: 0.5 };
    const state = stepHandlers.gripper.init(step, ctx);
    stepHandlers.gripper.tick(state, ctx, DT_SEC);

    const midValue = GRIPPER_CLOSE_M + (GRIPPER_OPEN_M - GRIPPER_CLOSE_M) * 0.5;
    expect(robots.setJointsCalls[0]?.values['finger_l']).toBeCloseTo(midValue, FLOAT_DIGITS);
    expect(robots.setJointsCalls[0]?.values['finger_r']).toBeCloseTo(midValue, FLOAT_DIGITS);
  });

  it('숫자는 0..1로 클램프된다 (1.5→open, -1→close)', () => {
    const overRig = makeGripperRig();
    const overStep: StepOfKind<'gripper'> = { kind: 'gripper', state: 1.5 };
    const overState = stepHandlers.gripper.init(overStep, overRig.ctx);
    stepHandlers.gripper.tick(overState, overRig.ctx, DT_SEC);
    expect(overRig.robots.setJointsCalls[0]?.values['finger_l']).toBe(GRIPPER_OPEN_M);

    const underRig = makeGripperRig();
    const underStep: StepOfKind<'gripper'> = { kind: 'gripper', state: -1 };
    const underState = stepHandlers.gripper.init(underStep, underRig.ctx);
    stepHandlers.gripper.tick(underState, underRig.ctx, DT_SEC);
    expect(underRig.robots.setJointsCalls[0]?.values['finger_l']).toBe(GRIPPER_CLOSE_M);
  });

  it('durationSec 지정 시 moveJoints처럼 선형 보간한다 (중간값 → 최종 정확값)', () => {
    const { ctx, robots } = makeGripperRig();
    const step: StepOfKind<'gripper'> = { kind: 'gripper', state: 'open', durationSec: 0.2 };
    const state = stepHandlers.gripper.init(step, ctx);

    expect(stepHandlers.gripper.tick(state, ctx, DT_SEC)).toBe('running'); // t=0.5
    expect(robots.setJointsCalls[0]?.values['finger_l']).toBeCloseTo(
      GRIPPER_OPEN_M * 0.5,
      FLOAT_DIGITS,
    );

    expect(stepHandlers.gripper.tick(state, ctx, DT_SEC)).toBe('done'); // t=1
    expect(robots.setJointsCalls[1]?.values['finger_l']).toBe(GRIPPER_OPEN_M);
    expect(robots.setJointsCalls[1]?.values['finger_r']).toBe(GRIPPER_OPEN_M);
  });

  it('gripperConfig 없음 → init에서 한국어 경고 1회 + tick 즉시 done, setJoints 미호출', () => {
    const { ctx, robots, warnings } = makeRig(); // 그리퍼 미설정
    const step: StepOfKind<'gripper'> = { kind: 'gripper', state: 'open' };
    const state = stepHandlers.gripper.init(step, ctx);

    expect(warnings).toHaveLength(1);
    expect(warnings[0]).toContain('gripper');
    expect(warnings[0]).toContain(DEFAULT_ROBOT);

    expect(stepHandlers.gripper.tick(state, ctx, DT_SEC)).toBe('done');
    expect(robots.setJointsCalls).toHaveLength(0);
    expect(warnings).toHaveLength(1); // tick에서 추가 경고 없음
  });
});

// ── wait ────────────────────────────────────────────────────────────

describe('wait 핸들러', () => {
  it('경과 >= durationSec가 되는 tick에 done (경계 포함)', () => {
    const { ctx } = makeRig();
    const step: StepOfKind<'wait'> = { kind: 'wait', durationSec: 0.2 };
    const state = stepHandlers.wait.init(step, ctx);

    expect(stepHandlers.wait.tick(state, ctx, DT_SEC)).toBe('running'); // 0.1 < 0.2
    expect(stepHandlers.wait.tick(state, ctx, DT_SEC)).toBe('done'); // 0.2 >= 0.2
  });

  it('durationSec 0 → dt=0이어도 즉시 done', () => {
    const { ctx } = makeRig();
    const step: StepOfKind<'wait'> = { kind: 'wait', durationSec: 0 };
    const state = stepHandlers.wait.init(step, ctx);

    expect(stepHandlers.wait.tick(state, ctx, 0)).toBe('done');
  });
});

// ── waitForCollision ────────────────────────────────────────────────

describe('waitForCollision 핸들러', () => {
  const BETWEEN: [string, string] = ['arm', 'box_a'];

  it('init에서 mark를 발급하고, 같은 mark로 happenedSince를 조회한다', () => {
    const { ctx, collision } = makeRig();
    const step: StepOfKind<'waitForCollision'> = { kind: 'waitForCollision', between: BETWEEN };
    const state = stepHandlers.waitForCollision.init(step, ctx);

    expect(collision.markCalls).toHaveLength(1);
    stepHandlers.waitForCollision.tick(state, ctx, DT_SEC);
    expect(collision.happenedCalls).toEqual([{ mark: collision.markCalls[0], between: BETWEEN }]);
  });

  it('충돌 발생 시 done — 스크립트된 3번째 조회에서 해제', () => {
    const { ctx, collision } = makeRig();
    let queries = 0;
    const releaseOnQuery = 3;
    collision.respond = () => {
      queries += 1;
      return queries >= releaseOnQuery;
    };

    const step: StepOfKind<'waitForCollision'> = { kind: 'waitForCollision', between: BETWEEN };
    const state = stepHandlers.waitForCollision.init(step, ctx);

    expect(stepHandlers.waitForCollision.tick(state, ctx, DT_SEC)).toBe('running');
    expect(stepHandlers.waitForCollision.tick(state, ctx, DT_SEC)).toBe('running');
    expect(stepHandlers.waitForCollision.tick(state, ctx, DT_SEC)).toBe('done');
  });

  it('timeoutSec 초과 시 경고(쌍 + timeout 포함) 후 done — 경고 후 진행 (SIMULATION §3)', () => {
    const { ctx, warnings } = makeRig();
    const timeoutSec = 0.2;
    const step: StepOfKind<'waitForCollision'> = {
      kind: 'waitForCollision',
      between: BETWEEN,
      timeoutSec,
    };
    const state = stepHandlers.waitForCollision.init(step, ctx);

    expect(stepHandlers.waitForCollision.tick(state, ctx, DT_SEC)).toBe('running'); // 0.1 < 0.2
    expect(stepHandlers.waitForCollision.tick(state, ctx, DT_SEC)).toBe('done'); // 0.2 >= 0.2

    expect(warnings).toHaveLength(1);
    expect(warnings[0]).toContain(BETWEEN[0]);
    expect(warnings[0]).toContain(BETWEEN[1]);
    expect(warnings[0]).toContain(String(timeoutSec));
  });

  it('충돌 확인이 timeout 판정보다 우선한다', () => {
    const { ctx, collision, warnings } = makeRig();
    collision.respond = () => true;
    const shortTimeoutSec = 0.05;
    const step: StepOfKind<'waitForCollision'> = {
      kind: 'waitForCollision',
      between: BETWEEN,
      timeoutSec: shortTimeoutSec,
    };
    const state = stepHandlers.waitForCollision.init(step, ctx);

    expect(stepHandlers.waitForCollision.tick(state, ctx, DT_SEC)).toBe('done');
    expect(warnings).toHaveLength(0);
  });

  it('timeoutSec 미지정 → 무한 대기 (경고 없음)', () => {
    const { ctx, warnings } = makeRig();
    const step: StepOfKind<'waitForCollision'> = { kind: 'waitForCollision', between: BETWEEN };
    const state = stepHandlers.waitForCollision.init(step, ctx);

    const manyTicks = 100;
    for (let i = 0; i < manyTicks; i += 1) {
      expect(stepHandlers.waitForCollision.tick(state, ctx, DT_SEC)).toBe('running');
    }
    expect(warnings).toHaveLength(0);
  });
});

// ── label / goto / moveToPose ───────────────────────────────────────

describe('label 핸들러', () => {
  it('no-op 마커 — 즉시 done', () => {
    const { ctx, robots } = makeRig();
    const step: StepOfKind<'label'> = { kind: 'label', name: 'L' };
    const state = stepHandlers.label.init(step, ctx);

    expect(stepHandlers.label.tick(state, ctx, DT_SEC)).toBe('done');
    expect(robots.setJointsCalls).toHaveLength(0);
  });
});

describe('goto 핸들러', () => {
  it('player가 가로채야 하는 제어 흐름 — 핸들러 tick 도달 시 throw (내부 방어)', () => {
    const { ctx } = makeRig();
    const step: StepOfKind<'goto'> = { kind: 'goto', label: 'L' };
    const runtime = initStep(step, ctx);

    expect(() => runtime.tick(ctx, DT_SEC)).toThrow(/player/);
  });
});

describe('moveToPose 핸들러', () => {
  it('IK 백로그 — 정확한 한국어 경고 후 건너뜀 (흉내 내지 않음)', () => {
    const { ctx, robots, warnings } = makeRig();
    const step: StepOfKind<'moveToPose'> = {
      kind: 'moveToPose',
      target: { position: [0.1, 0.2, 0.3] },
      durationSec: 1,
    };
    const state = stepHandlers.moveToPose.init(step, ctx);

    expect(stepHandlers.moveToPose.tick(state, ctx, DT_SEC)).toBe('done');
    expect(warnings).toEqual(['moveToPose는 IK 백로그 — 건너뜀']);
    expect(robots.setJointsCalls).toHaveLength(0);
  });
});
