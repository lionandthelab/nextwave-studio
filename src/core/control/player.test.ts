// core/control/player.test.ts — ControlPlayer 순수 로직 단위 테스트
//
// 물리/렌더 없이 완결적으로 검증한다 (CLAUDE.md §4 "부수효과 격리"):
// - RobotApi 목: setJoints 호출 로그 + 내부 관절 상태 반영 (연쇄 step의 스냅샷 검증)
// - CollisionQuery 목: happenedSince 응답 스크립트 (waitForCollision 배리어)
// 규범: docs/SIMULATION.md §3 (player 계약), docs/DATA_MODEL.md §6 (step/goto 의미).

import { describe, expect, it } from 'vitest';
import { ControlPlayer, SAME_TICK_STEP_LIMIT } from './player';
import type { CollisionQuery, RobotApi } from './steps';
import type { ControlSequence, ControlStep } from '../../schema/types';

// ── 테스트 상수 (매직넘버 금지 — CLAUDE.md §4) ──────────────────────

const DEFAULT_ROBOT = 'arm';
const OTHER_ROBOT = 'arm2';
const SEQ_ID = 'test-seq';

const DT_SEC = 0.1;
/** step()의 simTime 인자 — player는 사용하지 않지만 엔진 계약상 전달된다 */
const SIM_TIME_SEC = 0;

const FLOAT_DIGITS = 12;

// ── 목 RobotApi / CollisionQuery ────────────────────────────────────

interface SetJointsCall {
  robot: string;
  values: Record<string, number>;
}

class MockRobotApi implements RobotApi {
  readonly setJointsCalls: SetJointsCall[] = [];

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

  private robotState(robot: string): Map<string, number> {
    const joints = this.state.get(robot);
    if (!joints) throw new Error(`mock: 로봇 '${robot}' 없음`);
    return joints;
  }
}

class MockCollisionQuery implements CollisionQuery {
  readonly markCalls: unknown[] = [];
  readonly happenedCalls: { mark: unknown; between: [string, string] }[] = [];
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

// ── 리그 ────────────────────────────────────────────────────────────

/** onStepChange 기록 항목 — kind 문자열로 비교해 가독성 확보 */
interface StepChangeRecord {
  index: number;
  kind: ControlStep['kind'] | null;
}

interface PlayerRig {
  player: ControlPlayer;
  robots: MockRobotApi;
  collision: MockCollisionQuery;
  warnings: string[];
  /** load 이전부터 구독된 onStepChange 로그 (load/reset/랩/끝 통지 포함) */
  events: StepChangeRecord[];
}

function makeRig(initialJoints?: Record<string, Record<string, number>>): PlayerRig {
  const robots = new MockRobotApi(
    initialJoints ?? { [DEFAULT_ROBOT]: { j1: 0, j2: 0 }, [OTHER_ROBOT]: { j1: 0 } },
  );
  const collision = new MockCollisionQuery();
  const warnings: string[] = [];
  const player = new ControlPlayer({ robots, collision, warn: (msg) => warnings.push(msg) });
  const events: StepChangeRecord[] = [];
  player.onStepChange((index, step) => events.push({ index, kind: step?.kind ?? null }));
  return { player, robots, collision, warnings, events };
}

function makeSeq(
  steps: ControlStep[],
  opts?: { robot?: string; loop?: boolean },
): ControlSequence {
  return { id: SEQ_ID, robot: opts?.robot ?? DEFAULT_ROBOT, loop: opts?.loop, steps };
}

/** n tick 진행 (고정 dt — 결정론) */
function runTicks(player: ControlPlayer, n: number): void {
  for (let i = 0; i < n; i += 1) player.step(SIM_TIME_SEC, DT_SEC);
}

// ── 로드/리셋/상태 표면 ─────────────────────────────────────────────

describe('ControlPlayer 상태 표면', () => {
  it('로드 전: idle, loaded=false, step()/reset()은 no-op', () => {
    const { player, robots, events } = makeRig();

    expect(player.status).toBe('idle');
    expect(player.loaded).toBe(false);
    expect(player.stepCount).toBe(0);

    player.step(SIM_TIME_SEC, DT_SEC);
    player.reset();

    expect(player.status).toBe('idle');
    expect(robots.setJointsCalls).toHaveLength(0);
    expect(events).toHaveLength(0);
  });

  it('load: running 전환 + 커서 0 + onStepChange(0, 첫 step) 통지', () => {
    const { player, events } = makeRig();
    player.load(makeSeq([{ kind: 'wait', durationSec: 1 }]));

    expect(player.status).toBe('running');
    expect(player.loaded).toBe(true);
    expect(player.stepCount).toBe(1);
    expect(player.currentStepIndex).toBe(0);
    expect(events).toEqual([{ index: 0, kind: 'wait' }]);
  });

  it('goto 대상 label이 없으면 load에서 한국어 오류로 throw (validateSequence 우회 방어)', () => {
    const { player } = makeRig();
    const invalid = makeSeq([{ kind: 'goto', label: 'nope' }]);

    expect(() => player.load(invalid)).toThrow(/label 'nope'/);
    expect(player.loaded).toBe(false);
    expect(player.status).toBe('idle');
  });

  it('onStepChange 해제 함수가 구독을 제거한다', () => {
    const { player } = makeRig();
    const extra: StepChangeRecord[] = [];
    const unsubscribe = player.onStepChange((index, step) =>
      extra.push({ index, kind: step?.kind ?? null }),
    );

    player.load(makeSeq([{ kind: 'wait', durationSec: 1 }]));
    expect(extra).toHaveLength(1);

    unsubscribe();
    player.reset();
    expect(extra).toHaveLength(1); // 해제 후 통지 없음
  });
});

// ── 기본 재생 흐름 ──────────────────────────────────────────────────

describe('ControlPlayer 재생', () => {
  it('zero-duration step 연쇄가 한 tick에 완료되고, onStepChange가 순서대로 발화한다', () => {
    const { player, robots, events } = makeRig();
    player.load(
      makeSeq([
        { kind: 'setJoints', targets: { j1: 0.5 } },
        { kind: 'wait', durationSec: 0.2 },
        { kind: 'setJoints', targets: { j1: 0 } },
      ]),
    );

    runTicks(player, 1); // setJoints 완료 → wait는 dt=0으로 활성화(시간 이중 계상 없음)
    expect(robots.setJointsCalls).toEqual([{ robot: DEFAULT_ROBOT, values: { j1: 0.5 } }]);
    expect(player.currentStepIndex).toBe(1);
    expect(player.status).toBe('running');

    runTicks(player, 1); // wait 경과 0.1 < 0.2
    expect(player.currentStepIndex).toBe(1);

    runTicks(player, 1); // wait 0.2 완료 → 마지막 setJoints 같은 tick 실행 → 시퀀스 끝
    expect(robots.setJointsCalls).toEqual([
      { robot: DEFAULT_ROBOT, values: { j1: 0.5 } },
      { robot: DEFAULT_ROBOT, values: { j1: 0 } },
    ]);
    expect(player.status).toBe('done');
    expect(player.currentStepIndex).toBe(player.stepCount);

    // 발화 순서: load(0) → 1 → 2 → 끝(3, null)
    expect(events).toEqual([
      { index: 0, kind: 'setJoints' },
      { index: 1, kind: 'wait' },
      { index: 2, kind: 'setJoints' },
      { index: 3, kind: null },
    ]);

    // done 이후 step()은 no-op
    runTicks(player, 3);
    expect(robots.setJointsCalls).toHaveLength(2);
    expect(events).toHaveLength(4);
  });

  it('moveJoints: 알려진 dt에서 보간 궤적 + 마지막 tick 정확한 목표값 + 종료 시 done', () => {
    const { player, robots } = makeRig();
    player.load(
      makeSeq([{ kind: 'moveJoints', targets: { j1: 1 }, durationSec: 0.2, easing: 'linear' }]),
    );

    runTicks(player, 1); // t=0.5
    expect(robots.setJointsCalls[0]?.values['j1']).toBeCloseTo(0.5, FLOAT_DIGITS);
    expect(player.status).toBe('running');

    runTicks(player, 1); // t=1 → 정확한 목표값 → 시퀀스 끝
    expect(robots.setJointsCalls[1]?.values['j1']).toBe(1);
    expect(player.status).toBe('done');
  });

  it('로봇 해석: step.robot ?? sequence.robot', () => {
    const { player, robots } = makeRig();
    player.load(
      makeSeq([
        { kind: 'setJoints', targets: { j1: 0.1 } }, // 기본 로봇
        { kind: 'setJoints', robot: OTHER_ROBOT, targets: { j1: 0.2 } }, // override
      ]),
    );

    runTicks(player, 1); // 두 step 모두 zero-duration — 한 tick에 완료
    expect(robots.setJointsCalls.map((c) => c.robot)).toEqual([DEFAULT_ROBOT, OTHER_ROBOT]);
    expect(player.status).toBe('done');
  });

  it('gripper: 설정 없는 로봇이면 경고 1회 후 건너뛰고 같은 tick에 다음 step 진행', () => {
    const { player, robots, warnings } = makeRig(); // 그리퍼 미설정
    player.load(
      makeSeq([
        { kind: 'gripper', state: 'open' },
        { kind: 'setJoints', targets: { j1: 0.3 } },
      ]),
    );

    runTicks(player, 1);
    expect(warnings).toHaveLength(1);
    expect(warnings[0]).toContain('gripper');
    expect(robots.setJointsCalls).toEqual([{ robot: DEFAULT_ROBOT, values: { j1: 0.3 } }]);
    expect(player.status).toBe('done');
  });

  it('gripper: 설정이 있으면 open 값으로 구동한다 (통합 경로)', () => {
    const { player, robots } = makeRig({ [DEFAULT_ROBOT]: { finger_l: 0, finger_r: 0 } });
    const openM = 0.03;
    robots.setGripper(DEFAULT_ROBOT, { joints: ['finger_l', 'finger_r'], open: openM, close: 0 });
    player.load(makeSeq([{ kind: 'gripper', state: 'open' }]));

    runTicks(player, 1);
    expect(robots.setJointsCalls).toEqual([
      { robot: DEFAULT_ROBOT, values: { finger_l: openM, finger_r: openM } },
    ]);
    expect(player.status).toBe('done');
  });

  it('moveToPose: 경고 후 건너뛰고 다음 step으로 진행 (IK 백로그 — 흉내 없음)', () => {
    const { player, robots, warnings } = makeRig();
    player.load(
      makeSeq([
        { kind: 'moveToPose', target: { position: [0.1, 0.2, 0.3] }, durationSec: 1 },
        { kind: 'setJoints', targets: { j1: 0.7 } },
      ]),
    );

    runTicks(player, 1);
    expect(warnings).toEqual(['moveToPose는 IK 백로그 — 건너뜀']);
    expect(robots.setJointsCalls).toEqual([{ robot: DEFAULT_ROBOT, values: { j1: 0.7 } }]);
    expect(player.status).toBe('done');
  });
});

// ── waitForCollision 배리어 ─────────────────────────────────────────

describe('ControlPlayer waitForCollision', () => {
  const BETWEEN: [string, string] = ['arm', 'box_a'];

  it('충돌 발생 시 해제되고 같은 tick에 다음 step이 실행된다', () => {
    const { player, robots, collision } = makeRig();
    let queries = 0;
    const releaseOnQuery = 2;
    collision.respond = () => {
      queries += 1;
      return queries >= releaseOnQuery;
    };
    player.load(
      makeSeq([
        { kind: 'waitForCollision', between: BETWEEN },
        { kind: 'setJoints', targets: { j1: 1 } },
      ]),
    );

    runTicks(player, 1); // 활성화 시 mark 발급, 1번째 조회 false
    expect(collision.markCalls).toHaveLength(1);
    expect(player.currentStepIndex).toBe(0);
    expect(robots.setJointsCalls).toHaveLength(0);

    runTicks(player, 1); // 2번째 조회 true → 해제 → 같은 tick에 setJoints
    expect(robots.setJointsCalls).toEqual([{ robot: DEFAULT_ROBOT, values: { j1: 1 } }]);
    expect(player.status).toBe('done');

    // 조회는 항상 init 시점 mark로 이루어진다
    for (const call of collision.happenedCalls) {
      expect(call.mark).toBe(collision.markCalls[0]);
      expect(call.between).toEqual(BETWEEN);
    }
  });

  it('timeout 초과 시 경고(쌍 + timeout 포함) 후 다음 step으로 진행한다', () => {
    const { player, robots, warnings } = makeRig();
    const timeoutSec = 0.2;
    player.load(
      makeSeq([
        { kind: 'waitForCollision', between: BETWEEN, timeoutSec },
        { kind: 'setJoints', targets: { j1: 1 } },
      ]),
    );

    runTicks(player, 1); // 경과 0.1 < 0.2
    expect(warnings).toHaveLength(0);
    expect(player.currentStepIndex).toBe(0);

    runTicks(player, 1); // 경과 0.2 → 경고 후 진행 (SIMULATION §3)
    expect(warnings).toHaveLength(1);
    expect(warnings[0]).toContain(BETWEEN[0]);
    expect(warnings[0]).toContain(BETWEEN[1]);
    expect(warnings[0]).toContain(String(timeoutSec));
    expect(robots.setJointsCalls).toEqual([{ robot: DEFAULT_ROBOT, values: { j1: 1 } }]);
    expect(player.status).toBe('done');
  });
});

// ── label / goto 흐름 제어 ──────────────────────────────────────────

describe('ControlPlayer label/goto', () => {
  it('goto times=2 → 정확히 2회 점프 후 통과 — 본문이 총 3회 실행된다', () => {
    const { player, robots, warnings } = makeRig();
    player.load(
      makeSeq([
        { kind: 'label', name: 'L' },
        { kind: 'setJoints', targets: { j1: 1 } }, // 본문
        { kind: 'goto', label: 'L', times: 2 },
      ]),
    );

    runTicks(player, 1); // 전부 zero-duration — 한 tick에 루프 완주
    expect(robots.setJointsCalls).toHaveLength(3);
    expect(player.status).toBe('done');
    expect(warnings).toHaveLength(0); // 한도 미도달 — 경고 없음
  });

  it('times 미지정(무한) goto라도 사이에 시간 소비 step이 있으면 한도에 걸리지 않는다', () => {
    const { player, warnings } = makeRig();
    player.load(
      makeSeq([
        { kind: 'label', name: 'L' },
        { kind: 'wait', durationSec: 0.1 },
        { kind: 'goto', label: 'L' },
      ]),
    );

    const manyTicks = 50;
    runTicks(player, manyTicks);
    expect(warnings).toHaveLength(0);
    expect(player.status).toBe('running'); // 무한 루프 — 정상 지속
  });

  it('label↔goto 무한 순환은 tick당 한도에서 경고 후 중단하고 다음 tick에 재개한다', () => {
    const { player, warnings } = makeRig();
    player.load(
      makeSeq([
        { kind: 'label', name: 'L' },
        { kind: 'goto', label: 'L' },
      ]),
    );

    runTicks(player, 1);
    expect(warnings).toHaveLength(1);
    expect(warnings[0]).toContain(String(SAME_TICK_STEP_LIMIT));
    expect(player.status).toBe('running'); // 시퀀스는 죽지 않는다 — tick만 중단

    runTicks(player, 1); // 다음 tick에도 같은 순환 — 다시 한도 경고
    expect(warnings).toHaveLength(2);
  });
});

// ── loop / reset 결정론 ─────────────────────────────────────────────

describe('ControlPlayer loop/reset 결정론', () => {
  /** 반복 재생해도 로그가 동일해지는 시퀀스 (최종 관절 상태 = 초기 상태) */
  const REPLAYABLE_STEPS: ControlStep[] = [
    { kind: 'setJoints', targets: { j1: 0.5 } },
    { kind: 'moveJoints', targets: { j1: 0 }, durationSec: 0.1, easing: 'linear' },
  ];
  /** 위 시퀀스 1회 완주에 걸리는 tick 수 (tick1: set+move활성, tick2: move 완료) */
  const TICKS_PER_PASS = 2;

  it('loop: 시퀀스 끝에서 처음으로 되감고(같은 tick 재실행 없음) 다음 랩 로그가 동일하다', () => {
    const { player, robots, events } = makeRig();
    player.load(makeSeq(REPLAYABLE_STEPS, { loop: true }));

    runTicks(player, TICKS_PER_PASS); // 1랩
    const firstPassLog = robots.setJointsCalls.map((c) => ({ ...c, values: { ...c.values } }));
    expect(player.status).toBe('running');
    expect(player.currentStepIndex).toBe(0); // 랩 후 커서 0 — 다음 tick부터 재생

    // 랩 통지: (0, 첫 step) — load 통지와 별개로 1회 더
    expect(events.filter((e) => e.index === 0 && e.kind === 'setJoints')).toHaveLength(2);

    runTicks(player, TICKS_PER_PASS); // 2랩
    const secondPassLog = robots.setJointsCalls.slice(firstPassLog.length);
    expect(secondPassLog).toEqual(firstPassLog); // 결정론적 재생 (SIMULATION §6)
  });

  it('loop 랩은 goto 발사 카운터도 초기화한다 — 랩마다 times가 다시 적용된다', () => {
    const { player, robots } = makeRig();
    player.load(
      makeSeq(
        [
          { kind: 'label', name: 'L' },
          { kind: 'setJoints', targets: { j1: 1 } },
          { kind: 'goto', label: 'L', times: 1 },
        ],
        { loop: true },
      ),
    );

    runTicks(player, 1); // 1랩: 본문 2회(점프 1회) 후 통과 → 랩
    expect(robots.setJointsCalls).toHaveLength(2);
    expect(player.status).toBe('running');

    runTicks(player, 1); // 2랩: 카운터 초기화 → 다시 본문 2회
    expect(robots.setJointsCalls).toHaveLength(4);
  });

  it('reset: 커서·경과·goto 카운터 초기화 → 재생 로그가 처음 재생과 동일하다', () => {
    const { player, robots, events } = makeRig();
    player.load(makeSeq(REPLAYABLE_STEPS)); // loop 없음

    runTicks(player, TICKS_PER_PASS);
    expect(player.status).toBe('done');
    const firstRunLog = robots.setJointsCalls.map((c) => ({ ...c, values: { ...c.values } }));

    player.reset();
    expect(player.status).toBe('running');
    expect(player.currentStepIndex).toBe(0);
    expect(events.at(-1)).toEqual({ index: 0, kind: 'setJoints' }); // reset 통지

    runTicks(player, TICKS_PER_PASS);
    expect(player.status).toBe('done');
    const secondRunLog = robots.setJointsCalls.slice(firstRunLog.length);
    expect(secondRunLog).toEqual(firstRunLog); // 결정론적 재생
  });

  it('진행 중 reset해도 처음부터 다시 결정론적으로 재생된다', () => {
    const { player, robots } = makeRig();
    player.load(makeSeq(REPLAYABLE_STEPS));

    runTicks(player, TICKS_PER_PASS);
    const fullRunLog = robots.setJointsCalls.map((c) => ({ ...c, values: { ...c.values } }));

    player.reset();
    runTicks(player, 1); // 중간까지만 진행 (관절이 중간 상태로 남는다)
    player.reset(); // 되감기 — setJoints는 절대값이라 다시 처음 로그가 재현돼야 한다
    robots.setJointsCalls.length = 0;

    runTicks(player, TICKS_PER_PASS);
    expect(robots.setJointsCalls).toEqual(fullRunLog);
  });
});

// ── enabled:false 스킵 (DATA_MODEL §6 StepCommon, UX_DESIGN §3.4) ───

describe('ControlPlayer enabled:false 스킵', () => {
  it('비활성 moveJoints는 관절을 건드리지 않고, 다음 step이 이번 tick의 dt를 온전히 받는다', () => {
    const { player, robots } = makeRig();
    player.load(
      makeSeq([
        { kind: 'moveJoints', targets: { j1: 1 }, durationSec: 1, enabled: false },
        { kind: 'wait', durationSec: DT_SEC }, // 스킵이 dt를 소비하지 않으면 1 tick에 완료
      ]),
    );

    runTicks(player, 1);
    expect(robots.setJointsCalls).toHaveLength(0); // 핸들러 미실행 — 관절 불변
    expect(player.status).toBe('done'); // wait가 전체 dt(0.1)를 받아 같은 tick에 완료
  });

  it('비활성 setJoints도 실행되지 않고 활성 step만 실행된다', () => {
    const { player, robots } = makeRig();
    player.load(
      makeSeq([
        { kind: 'setJoints', targets: { j1: 9 }, enabled: false },
        { kind: 'setJoints', targets: { j1: 1 } },
      ]),
    );

    runTicks(player, 1);
    expect(robots.setJointsCalls).toEqual([{ robot: DEFAULT_ROBOT, values: { j1: 1 } }]);
    expect(player.status).toBe('done');
  });

  it('비활성 goto는 점프하지 않고 통과한다(fall-through)', () => {
    const { player, robots } = makeRig();
    player.load(
      makeSeq([
        { kind: 'label', name: 'L' },
        { kind: 'setJoints', targets: { j1: 1 } },
        { kind: 'goto', label: 'L', times: 5, enabled: false }, // 활성이면 본문 6회
        { kind: 'setJoints', targets: { j1: 2 } },
      ]),
    );

    runTicks(player, 1);
    expect(robots.setJointsCalls).toEqual([
      { robot: DEFAULT_ROBOT, values: { j1: 1 } }, // 본문 1회 — 점프 없음
      { robot: DEFAULT_ROBOT, values: { j1: 2 } },
    ]);
    expect(player.status).toBe('done');
  });

  it('비활성 label도 goto 대상으로 해석된다(위치 마커) — load도 throw하지 않는다', () => {
    const { player, robots } = makeRig();
    player.load(
      makeSeq([
        { kind: 'goto', label: 'L', times: 1 }, // 전방 점프 → 사이 step을 건너뛴다
        { kind: 'setJoints', targets: { j1: 9 } }, // 점프로 건너뛰어짐
        { kind: 'label', name: 'L', enabled: false },
        { kind: 'setJoints', targets: { j1: 1 } },
      ]),
    );

    runTicks(player, 1);
    expect(robots.setJointsCalls).toEqual([{ robot: DEFAULT_ROBOT, values: { j1: 1 } }]);
    expect(player.status).toBe('done');
  });

  it('loop: 비활성 step이 있어도 랩이 정상 동작하고 랩마다 로그가 동일하다', () => {
    const { player, robots } = makeRig();
    player.load(
      makeSeq(
        [
          { kind: 'setJoints', targets: { j1: 1 } },
          { kind: 'wait', durationSec: 1, enabled: false }, // 활성이면 랩당 10+ tick
        ],
        { loop: true },
      ),
    );

    runTicks(player, 1); // 1랩: setJoints + 스킵 + 랩
    expect(robots.setJointsCalls).toHaveLength(1);
    expect(player.status).toBe('running');
    expect(player.currentStepIndex).toBe(0); // 랩 후 커서 0

    runTicks(player, 1); // 2랩 — 동일 로그 반복 (결정론)
    expect(robots.setJointsCalls).toHaveLength(2);
    expect(robots.setJointsCalls[1]).toEqual(robots.setJointsCalls[0]);
  });

  it('전부 비활성인 시퀀스(비루프)는 아무것도 실행하지 않고 done이 된다', () => {
    const { player, robots, warnings, events } = makeRig();
    player.load(
      makeSeq([
        { kind: 'setJoints', targets: { j1: 1 }, enabled: false },
        { kind: 'wait', durationSec: 5, enabled: false },
      ]),
    );

    runTicks(player, 1);
    expect(robots.setJointsCalls).toHaveLength(0);
    expect(warnings).toHaveLength(0);
    expect(player.status).toBe('done');
    // 커서는 스킵된 step들을 통과하며 통지된다: load(0) → 1 → 끝(2, null)
    expect(events).toEqual([
      { index: 0, kind: 'setJoints' },
      { index: 1, kind: 'wait' },
      { index: 2, kind: null },
    ]);
  });

  it('스킵도 같은 tick 진행 한도에 계상된다 — 한도 초과 시 경고 후 다음 tick 재개', () => {
    const { player, warnings } = makeRig();
    const extraSteps = 5;
    const steps: ControlStep[] = Array.from(
      { length: SAME_TICK_STEP_LIMIT + extraSteps },
      (): ControlStep => ({ kind: 'wait', durationSec: 1, enabled: false }),
    );
    player.load(makeSeq(steps));

    runTicks(player, 1); // 한도만큼 스킵 후 중단
    expect(warnings).toHaveLength(1);
    expect(warnings[0]).toContain(String(SAME_TICK_STEP_LIMIT));
    expect(player.currentStepIndex).toBe(SAME_TICK_STEP_LIMIT);
    expect(player.status).toBe('running');

    runTicks(player, 1); // 나머지 스킵 → 끝
    expect(player.status).toBe('done');
    expect(warnings).toHaveLength(1); // 추가 경고 없음
  });

  it('enabled:true 명시는 일반 실행과 동일하다', () => {
    const { player, robots } = makeRig();
    player.load(makeSeq([{ kind: 'setJoints', targets: { j1: 0.5 }, enabled: true }]));

    runTicks(player, 1);
    expect(robots.setJointsCalls).toEqual([{ robot: DEFAULT_ROBOT, values: { j1: 0.5 } }]);
    expect(player.status).toBe('done');
  });
});
