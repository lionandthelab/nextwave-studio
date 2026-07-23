// core/control/steps.ts — ControlStep 종류별 핸들러 (Phase 5)
// 규범: docs/SIMULATION.md §3 (player 해석 계약), docs/DATA_MODEL.md §6 (step 의미).
//
// 설계:
// - 순수 로직 — 물리(Rapier)/렌더(three)/DOM 비의존. 아래의 좁은 인터페이스
//   (RobotApi / CollisionQuery)에만 의존해 단위 테스트 가능하다 (CLAUDE.md §4).
//   통합 시 RobotRegistry/RobotBinding·CollisionMonitor가 이 인터페이스로 어댑트된다.
// - 핸들러 균일 형태: init(step, ctx) → state; tick(state, ctx, dtSec) → 'running' | 'done'.
//   상태는 핸들러가 만든 POJO에만 담긴다 — 핸들러 자체는 무상태(재사용 가능).
// - 로봇 해석: step.robot ?? 시퀀스 기본 로봇(ctx.defaultRobot) (DATA_MODEL §6).
// - 'goto'는 제어 흐름이라 PLAYER가 가로채 처리한다(핸들러 표에는 방어용 엔트리만 존재).
// - 'moveToPose'는 IK 백로그(ROADMAP) — 흉내 내지 않고 경고 후 건너뛴다.

import type { ControlStep, ControlStepKind, Easing } from '../../schema/types';
import { clamp01, ease, lerp } from '../math';

// ── 의존 인터페이스 (통합 계층이 어댑트) ─────────────────────────────

/** 로봇 관절 읽기/쓰기 표면 (RobotRegistry/RobotBinding 어댑터가 구현) */
export interface RobotApi {
  /**
   * 현재 관절값 조회. names 지정 시 "요청한 이름 그대로" 키로 반환한다
   * — moveJoints가 targets와 같은 키로 시작값을 스냅샷할 수 있게 (SIMULATION §3).
   */
  readJoints(robot: string, names?: string[]): Record<string, number>;
  /** 관절 setpoint 일괄 적용 (limits 클램프는 구현 소관) */
  setJoints(robot: string, values: Record<string, number>): void;
  /** RobotSpec.gripper 설정 (없으면 undefined — gripper step은 경고 후 건너뜀) */
  gripperConfig(robot: string): { joints: string[]; open: number; close: number } | undefined;
}

/** 충돌 이력 조회 표면 (CollisionMonitor 어댑터가 구현) */
export interface CollisionQuery {
  /** 현재 시점 마커 발급 — waitForCollision init 시점 기록용 (opaque) */
  mark(): unknown;
  /** mark 이후 between 쌍의 충돌이 있었는지 (순서 무관 여부는 구현 소관) */
  happenedSince(mark: unknown, between: [string, string]): boolean;
}

/** 핸들러 실행 컨텍스트 — player가 조립해 매 tick 전달한다 */
export interface StepContext {
  robots: RobotApi;
  collision: CollisionQuery;
  /** 시퀀스 기본 대상 로봇 (ControlSequence.robot) — step.robot이 없을 때 사용 */
  defaultRobot: string;
  /** 사용자 대면 경고 (한국어) — UI 콘솔/토스트로 전달된다 */
  warn(msg: string): void;
}

// ── 핸들러 계약 ─────────────────────────────────────────────────────

export type StepStatus = 'running' | 'done';

/** kind K의 step 타입 추출 */
export type StepOfKind<K extends ControlStepKind> = Extract<ControlStep, { kind: K }>;

/**
 * step 핸들러 균일 계약. 핸들러는 무상태 — 모든 진행 상태는 init이 만든 state에 담긴다.
 * tick은 매 물리 tick 호출되며, 같은 엔진 tick 안에서 이전 step이 완료된 직후 이어지는
 * step은 dtSec=0으로 tick된다(시간 이중 계상 방지 — player 계약).
 */
export interface StepHandler<S extends ControlStep, State> {
  init(step: S, ctx: StepContext): State;
  tick(state: State, ctx: StepContext, dtSec: number): StepStatus;
}

// ── 공용: 관절 보간 상태 기계 (moveJoints · gripper 공유) ────────────

/** 시작값→목표값 보간 진행 상태 */
interface JointMotionState {
  robot: string;
  /** 목표 관절값 (적용 키 순서 = 이 객체의 키 순서 — 결정론) */
  targets: Readonly<Record<string, number>>;
  /** init 시점 스냅샷 (targets와 동일한 키) */
  startJoints: Readonly<Record<string, number>>;
  durationSec: number;
  easing: Easing;
  elapsedSec: number;
}

function initJointMotion(
  ctx: StepContext,
  robot: string,
  targets: Readonly<Record<string, number>>,
  durationSec: number,
  easing: Easing,
): JointMotionState {
  return {
    robot,
    targets,
    startJoints: ctx.robots.readJoints(robot, Object.keys(targets)),
    durationSec,
    easing,
    elapsedSec: 0,
  };
}

/**
 * 보간 1 tick. t = clamp01(elapsed/durationSec) (durationSec 0 → 즉시 t=1).
 * t>=1이 되는 "마지막 tick"은 보간값이 아니라 목표값을 정확히 적용한다
 * — 부동소수 잔차 없이 최종 상태가 targets와 일치해야 한다 (결정론, SIMULATION §6).
 */
function tickJointMotion(state: JointMotionState, ctx: StepContext, dtSec: number): StepStatus {
  state.elapsedSec += dtSec;
  const t = state.durationSec <= 0 ? 1 : clamp01(state.elapsedSec / state.durationSec);
  if (t >= 1) {
    ctx.robots.setJoints(state.robot, { ...state.targets });
    return 'done';
  }
  const e = ease(state.easing, t);
  const values: Record<string, number> = {};
  for (const [name, target] of Object.entries(state.targets)) {
    // startJoints는 targets와 같은 키로 스냅샷된다 — ?? target은 타입 방어일 뿐
    values[name] = lerp(state.startJoints[name] ?? target, target, e);
  }
  ctx.robots.setJoints(state.robot, values);
  return 'running';
}

// ── 개별 핸들러 ─────────────────────────────────────────────────────

/** step.robot ?? 시퀀스 기본 로봇 (DATA_MODEL §6 해석 규칙) */
function resolveRobot(stepRobot: string | undefined, ctx: StepContext): string {
  return stepRobot ?? ctx.defaultRobot;
}

// setJoints — 즉시 적용 1회 → done
interface SetJointsState {
  robot: string;
  targets: Readonly<Record<string, number>>;
}

const setJointsHandler: StepHandler<StepOfKind<'setJoints'>, SetJointsState> = {
  init: (step, ctx) => ({ robot: resolveRobot(step.robot, ctx), targets: step.targets }),
  tick: (state, ctx) => {
    ctx.robots.setJoints(state.robot, { ...state.targets });
    return 'done';
  },
};

// moveJoints — 시작값 스냅샷 후 durationSec 동안 easing 보간
const moveJointsHandler: StepHandler<StepOfKind<'moveJoints'>, JointMotionState> = {
  init: (step, ctx) =>
    initJointMotion(
      ctx,
      resolveRobot(step.robot, ctx),
      step.targets,
      step.durationSec,
      step.easing ?? 'linear',
    ),
  tick: tickJointMotion,
};

// gripper — gripperConfig 기반 목표값 산출 + 선형 보간 (durationSec ?? 0)
type GripperState =
  | { kind: 'noConfig' }
  | { kind: 'motion'; motion: JointMotionState };

/**
 * 그리퍼 상태 → 관절 목표값. 'open'/'close'는 설정값 그대로,
 * 숫자 x(0..1 클램프)는 close↔open 선형 보간 (DATA_MODEL §4.1: x=1이 열림).
 */
function resolveGripperValue(
  state: 'open' | 'close' | number,
  config: { open: number; close: number },
): number {
  if (state === 'open') return config.open;
  if (state === 'close') return config.close;
  return lerp(config.close, config.open, clamp01(state));
}

const gripperHandler: StepHandler<StepOfKind<'gripper'>, GripperState> = {
  init: (step, ctx) => {
    const robot = resolveRobot(step.robot, ctx);
    const config = ctx.robots.gripperConfig(robot);
    if (!config) {
      ctx.warn(
        `control: 로봇 '${robot}'에 gripper 설정이 없습니다 — gripper step을 건너뜁니다 ` +
          '(RobotSpec.gripper를 정의하세요)',
      );
      return { kind: 'noConfig' };
    }
    const value = resolveGripperValue(step.state, config);
    const targets: Record<string, number> = {};
    for (const joint of config.joints) targets[joint] = value; // 평행 그리퍼 — 전 관절 동일값
    return {
      kind: 'motion',
      motion: initJointMotion(ctx, robot, targets, step.durationSec ?? 0, 'linear'),
    };
  },
  tick: (state, ctx, dtSec) => {
    if (state.kind === 'noConfig') return 'done';
    return tickJointMotion(state.motion, ctx, dtSec);
  },
};

// wait — 경과 시간이 durationSec 이상이면 done
interface WaitState {
  durationSec: number;
  elapsedSec: number;
}

const waitHandler: StepHandler<StepOfKind<'wait'>, WaitState> = {
  init: (step) => ({ durationSec: step.durationSec, elapsedSec: 0 }),
  tick: (state, _ctx, dtSec) => {
    state.elapsedSec += dtSec;
    return state.elapsedSec >= state.durationSec ? 'done' : 'running';
  },
};

// waitForCollision — init 시점 마커 이후의 충돌 발생을 배리어로 대기.
// timeoutSec 초과 시 경고 후 진행한다 (DATA_MODEL §6 "timeout 초과 시 경고 후 진행").

/**
 * waitForCollision timeout 경고의 **문구 계약** 조각. main.ts(handlePlayerWarn)가
 * 이 두 문자열의 substring 매칭으로 해당 노드를 'error'로 마킹한다 — 경고 문구를
 * 바꿀 때는 반드시 이 상수를 통해서만 바꿀 것 (steps.test.ts가 발행 문구에 두 조각이
 * 포함됨을 고정한다 — 임의 리워딩이 소비자를 조용히 끊는 것을 방지).
 */
export const WAIT_FOR_COLLISION_WARN_TAG = 'waitForCollision';
export const WAIT_FOR_COLLISION_TIMEOUT_MARKER = '감지되지 않았습니다';

interface WaitForCollisionState {
  collisionMark: unknown;
  between: [string, string];
  timeoutSec?: number;
  elapsedSec: number;
}

const waitForCollisionHandler: StepHandler<StepOfKind<'waitForCollision'>, WaitForCollisionState> =
  {
    init: (step, ctx) => ({
      collisionMark: ctx.collision.mark(),
      between: step.between,
      timeoutSec: step.timeoutSec,
      elapsedSec: 0,
    }),
    tick: (state, ctx, dtSec) => {
      if (ctx.collision.happenedSince(state.collisionMark, state.between)) return 'done';
      state.elapsedSec += dtSec;
      if (state.timeoutSec !== undefined && state.elapsedSec >= state.timeoutSec) {
        ctx.warn(
          `control: ${WAIT_FOR_COLLISION_WARN_TAG}('${state.between[0]}' × '${state.between[1]}')이(가) ` +
            `${state.timeoutSec}s 안에 ${WAIT_FOR_COLLISION_TIMEOUT_MARKER} — 경고 후 다음 step으로 진행합니다`,
        );
        return 'done';
      }
      return 'running';
    },
  };

// label — no-op 마커 (goto 점프 대상). 즉시 done.
const labelHandler: StepHandler<StepOfKind<'label'>, null> = {
  init: () => null,
  tick: () => 'done',
};

// goto — 제어 흐름은 PLAYER가 가로채 처리한다 (커서 점프 + times 카운터).
// 이 엔트리는 표의 완전성을 위한 방어용이다 — 도달하면 player 버그다.
const gotoHandler: StepHandler<StepOfKind<'goto'>, null> = {
  init: () => null,
  tick: () => {
    throw new Error(
      'control: goto step은 핸들러가 아니라 player가 처리해야 합니다 (내부 오류 — 커서 점프 로직 확인)',
    );
  },
};

// moveToPose — IK 백로그 (ROADMAP). 흉내 내지 않고 경고 후 건너뛴다.
const moveToPoseHandler: StepHandler<StepOfKind<'moveToPose'>, null> = {
  init: () => null,
  tick: (_state, ctx) => {
    ctx.warn('moveToPose는 IK 백로그 — 건너뜀');
    return 'done';
  },
};

// ── 핸들러 표 + 실행 래퍼 ───────────────────────────────────────────

/** kind → init이 만드는 state 타입 */
interface StepStateMap {
  moveJoints: JointMotionState;
  setJoints: SetJointsState;
  gripper: GripperState;
  wait: WaitState;
  waitForCollision: WaitForCollisionState;
  label: null;
  goto: null;
  moveToPose: null;
}

export type StepHandlers = {
  readonly [K in ControlStepKind]: StepHandler<StepOfKind<K>, StepStateMap[K]>;
};

/** step kind별 핸들러 표 (모든 kind 커버 — 컴파일 타임 보장) */
export const stepHandlers: StepHandlers = {
  moveJoints: moveJointsHandler,
  setJoints: setJointsHandler,
  gripper: gripperHandler,
  wait: waitHandler,
  waitForCollision: waitForCollisionHandler,
  label: labelHandler,
  goto: gotoHandler,
  moveToPose: moveToPoseHandler,
};

/**
 * init 완료된 활성 step 런타임 — state를 캡슐화하고 tick만 노출한다.
 * player는 step이 활성화될 때 initStep()으로 만들고, 완료/커서 이동 시 버린다.
 */
export interface StepRuntime {
  tick(ctx: StepContext, dtSec: number): StepStatus;
}

function makeRuntime<S extends ControlStep, State>(
  handler: StepHandler<S, State>,
  step: S,
  ctx: StepContext,
): StepRuntime {
  const state = handler.init(step, ctx);
  return { tick: (tickCtx, dtSec) => handler.tick(state, tickCtx, dtSec) };
}

/** step을 핸들러로 init하고 tick 가능한 런타임으로 감싼다 (kind별 분기 — 캐스트 없음) */
export function initStep(step: ControlStep, ctx: StepContext): StepRuntime {
  switch (step.kind) {
    case 'moveJoints':
      return makeRuntime(stepHandlers.moveJoints, step, ctx);
    case 'setJoints':
      return makeRuntime(stepHandlers.setJoints, step, ctx);
    case 'gripper':
      return makeRuntime(stepHandlers.gripper, step, ctx);
    case 'wait':
      return makeRuntime(stepHandlers.wait, step, ctx);
    case 'waitForCollision':
      return makeRuntime(stepHandlers.waitForCollision, step, ctx);
    case 'label':
      return makeRuntime(stepHandlers.label, step, ctx);
    case 'goto':
      return makeRuntime(stepHandlers.goto, step, ctx);
    case 'moveToPose':
      return makeRuntime(stepHandlers.moveToPose, step, ctx);
  }
}
