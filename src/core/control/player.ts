// core/control/player.ts — ControlSequence 해석기 (Phase 5)
// 규범: docs/SIMULATION.md §3, docs/DATA_MODEL.md §6, ARCHITECTURE §5 (preStep 훅).
//
// 계약:
// - 순수 로직 — steps.ts의 RobotApi/CollisionQuery 인터페이스에만 의존한다.
//   Engine preStep 훅에서 player.step(simTime, dt) → robots.tickAll() 순서로 호출된다
//   (player가 관절 "상태"를 쓰고, robots가 FK를 kinematic 바디로 push).
// - 같은 엔진 tick 안에서 여러 zero-duration step이 연쇄 완료될 수 있다.
//   tick의 첫 step만 전체 dt를 받고, 이어지는 step은 dt=0으로 처리한다
//   (시간 이중 계상 방지 — 시간 소비 step은 다음 tick부터 카운트 시작).
// - label/goto 순환 폭주 방지: 한 tick의 커서 진행(완료·점프)은
//   SAME_TICK_STEP_LIMIT회로 제한하고, 초과 시 한국어 경고 후 이번 tick을 중단한다.
// - goto는 제어 흐름이므로 핸들러가 아니라 player가 가로챈다: goto step 인덱스별
//   발사(fire) 카운터를 유지해 times 지정 시 최대 times회만 점프, 이후 통과(fall-through).
//   times 미지정은 무한 점프 (DATA_MODEL §6).
// - load/reset/루프 랩은 커서·경과·goto 카운터를 모두 초기화한다 — 결정론적 재생
//   (동일 시퀀스 → 동일 setJoints 호출 로그, SIMULATION §6).
// - enabled:false step은 핸들러 실행 없이 건너뛴다(순서 유지 — DATA_MODEL §6,
//   UX_DESIGN §3.4 활성/비활성). 스킵도 같은 tick 진행 한도에 계상된다.
//   · 비활성 goto는 점프하지 않고 통과한다(스킵이 goto 가로채기보다 먼저).
//   · 비활성 label은 여전히 goto 대상으로 해석된다 — label은 위치 마커이며,
//     비활성화는 "그 자리에서 아무것도 안 함"(원래 no-op)만 바꾸기 때문이다.
//   · 스킵은 시간을 소비하지 않으므로 dt를 보존한다 — 이번 tick의 첫 "실행" step이
//     전체 dt를 받는다(비활성 step은 순서만 남기고 없는 것처럼 동작).

import type { ControlSequence, ControlStep } from '../../schema/types';
import type { CollisionQuery, RobotApi, StepContext, StepRuntime } from './steps';
import { initStep } from './steps';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4) ─────────────────────────────

/**
 * 한 물리 tick 안에서 허용하는 최대 커서 진행 횟수(step 완료 + goto 점프).
 * label↔goto 무한 순환이 한 tick을 잠그는 것을 방지한다. 정상 흐름(시간 소비 step이
 * 사이에 있는 루프)은 이 한도에 도달하지 않는다.
 */
export const SAME_TICK_STEP_LIMIT = 64;

// ── 타입 ────────────────────────────────────────────────────────────

export interface ControlPlayerDeps {
  robots: RobotApi;
  collision: CollisionQuery;
  /** 경고 출력 (기본: console.warn). UI 콘솔/토스트로 라우팅 가능. */
  warn?: (msg: string) => void;
}

export type PlayerStatus = 'idle' | 'running' | 'done';

/**
 * 커서 이동 통지. 시퀀스 끝(비루프)에서는 (stepCount, null)로 1회 통지된다.
 * load/reset/루프 랩은 (0, steps[0])로 통지된다.
 */
export type StepChangeListener = (index: number, step: ControlStep | null) => void;

// ── ControlPlayer ───────────────────────────────────────────────────

export class ControlPlayer {
  private readonly robots: RobotApi;
  private readonly collision: CollisionQuery;
  private readonly warnFn: (msg: string) => void;

  private seq: ControlSequence | null = null;
  private ctx: StepContext | null = null;
  private cursor = 0;
  /** 현재 step의 init 완료 런타임 — 커서 이동 시 폐기 */
  private active: StepRuntime | null = null;
  private _status: PlayerStatus = 'idle';
  /** goto step 인덱스 → 지금까지 점프한 횟수 (reset/load/루프 랩에 초기화) */
  private readonly gotoFireCounts = new Map<number, number>();
  /** label 이름 → step 인덱스 (중복 label은 첫 번째가 이긴다 — load 시 구축) */
  private labelIndexByName = new Map<string, number>();
  private readonly stepChangeListeners = new Set<StepChangeListener>();

  constructor(deps: ControlPlayerDeps) {
    this.robots = deps.robots;
    this.collision = deps.collision;
    this.warnFn = deps.warn ?? ((msg: string) => console.warn(msg));
  }

  // ── 상태 표면 (UI/게이트용) ───────────────────────────────────────

  get loaded(): boolean {
    return this.seq !== null;
  }

  get status(): PlayerStatus {
    return this._status;
  }

  /** 현재 커서 인덱스. 시퀀스 끝(비루프 완료)에서는 stepCount와 같다. */
  get currentStepIndex(): number {
    return this.cursor;
  }

  get stepCount(): number {
    return this.seq?.steps.length ?? 0;
  }

  /** 커서 이동 통지 구독 (load/reset/루프 랩 포함). 반환값은 해제 함수. */
  onStepChange(fn: StepChangeListener): () => void {
    this.stepChangeListeners.add(fn);
    return () => {
      this.stepChangeListeners.delete(fn);
    };
  }

  // ── 로드 / 리셋 ───────────────────────────────────────────────────

  /**
   * 시퀀스 로드: 커서·goto 카운터 초기화 후 running 상태로 전환한다.
   * goto 대상 label은 로드 시점에 검증한다 — validateSequence(schema)가 이미 검사하지만
   * 검증을 우회한 입력(프로그래밍 오류)에 대해 방어한다.
   */
  load(seq: ControlSequence): void {
    // label은 위치 마커 — enabled와 무관하게 goto 대상으로 해석한다
    // (비활성 label로의 점프 허용; 도착 후 label 자체는 스킵되어 원래처럼 no-op).
    const labels = new Map<string, number>();
    seq.steps.forEach((step, index) => {
      if (step.kind === 'label' && !labels.has(step.name)) labels.set(step.name, index);
    });
    seq.steps.forEach((step, index) => {
      if (step.kind === 'goto' && !labels.has(step.label)) {
        throw new Error(
          `player: steps[${index}]의 goto 대상 label '${step.label}'이(가) 시퀀스에 존재하지 않습니다`,
        );
      }
    });

    this.seq = seq;
    this.labelIndexByName = labels;
    this.ctx = {
      robots: this.robots,
      collision: this.collision,
      defaultRobot: seq.robot,
      warn: this.warnFn,
    };
    this.resetInternal();
    this._status = 'running';
    this.notifyCursor();
  }

  /**
   * 처음으로 되감기: 커서 0, 경과 0(활성 런타임 폐기), goto 카운터 초기화.
   * 동일 시퀀스를 다시 재생하면 동일한 setJoints 호출 로그가 나온다(결정론적 재생).
   * 로드 전에는 no-op.
   */
  reset(): void {
    if (!this.seq) return;
    this.resetInternal();
    this._status = 'running';
    this.notifyCursor();
  }

  // ── 매 물리 tick (Engine preStep 훅에서 호출) ─────────────────────

  /**
   * 현재 step을 핸들러로 1 tick 해석하고, 완료 시 커서를 전진시킨다.
   * 같은 tick에서 연쇄 완료되는 step은 dt=0으로 이어 처리한다('running' step
   * 또는 시퀀스 끝에서 멈춤). simTimeSec은 엔진 계약 유지용이다(현재 미사용).
   */
  step(_simTimeSec: number, dtSec: number): void {
    const seq = this.seq;
    const ctx = this.ctx;
    if (!seq || !ctx || this._status !== 'running') return;

    // 이번 tick의 첫 step만 전체 dt — 이후 연쇄 step은 dt=0 (시간 이중 계상 방지)
    let dt = dtSec;
    let progressBudget = SAME_TICK_STEP_LIMIT;

    for (;;) {
      if (this.cursor >= seq.steps.length) {
        this.handleSequenceEnd(seq);
        return;
      }
      const current = seq.steps[this.cursor];
      if (!current) return; // noUncheckedIndexedAccess 방어 — 위 범위 검사로 도달 불가

      // enabled:false — 핸들러 실행 없이 건너뜀 (순서 유지, DATA_MODEL §6).
      // goto 가로채기보다 먼저이므로 비활성 goto는 점프하지 않고 통과한다.
      // dt는 보존한다 — 스킵은 시간을 소비하지 않으며, 이번 tick의 첫 실행 step이
      // 전체 dt를 받아야 "비활성 = 없는 것처럼" 시맨틱이 유지된다.
      if (current.enabled === false) {
        if (progressBudget <= 0) {
          this.warnSameTickLimit();
          return;
        }
        progressBudget -= 1;
        this.moveCursorTo(this.cursor + 1);
        continue;
      }

      // goto는 제어 흐름 — 핸들러 대신 player가 커서를 직접 움직인다
      if (current.kind === 'goto') {
        if (progressBudget <= 0) {
          this.warnSameTickLimit();
          return;
        }
        progressBudget -= 1;
        this.performGoto(current, this.cursor);
        dt = 0; // 점프는 시간을 소비하지 않는다
        continue;
      }

      if (this.active === null) this.active = initStep(current, ctx);
      if (this.active.tick(ctx, dt) === 'running') return;

      // 완료 → 커서 전진 (한도 초과 시 이번 tick 중단 — 다음 tick에 재개)
      if (progressBudget <= 0) {
        this.warnSameTickLimit();
        return;
      }
      progressBudget -= 1;
      this.moveCursorTo(this.cursor + 1);
      dt = 0;
    }
  }

  // ── 내부 ──────────────────────────────────────────────────────────

  /** 커서 0·활성 런타임 폐기·goto 카운터 초기화 (상태 통지는 호출자 소관) */
  private resetInternal(): void {
    this.cursor = 0;
    this.active = null;
    this.gotoFireCounts.clear();
  }

  /**
   * goto 처리: times 미지정이면 항상 점프, 지정 시 최대 times회 점프 후 통과.
   * 점프/통과 모두 커서 이동으로 통지된다.
   */
  private performGoto(step: Extract<ControlStep, { kind: 'goto' }>, index: number): void {
    const fired = this.gotoFireCounts.get(index) ?? 0;
    const shouldJump = step.times === undefined || fired < step.times;
    if (shouldJump) {
      this.gotoFireCounts.set(index, fired + 1);
      const target = this.labelIndexByName.get(step.label);
      if (target === undefined) {
        // load()에서 검증됨 — 도달하면 내부 불변식 위반
        throw new Error(
          `player: goto 대상 label '${step.label}'을(를) 찾을 수 없습니다 (내부 오류 — load 검증 우회)`,
        );
      }
      this.moveCursorTo(target);
    } else {
      this.moveCursorTo(index + 1);
    }
  }

  /**
   * 시퀀스 끝 처리: loop면 리셋 후 "다음 tick"부터 처음부터 재생(같은 tick에 이어
   * 실행하지 않는다 — SIMULATION §3 onSequenceEnd). 아니면 done으로 전환하고
   * (stepCount, null)을 1회 통지한다. 이후 step()은 no-op.
   */
  private handleSequenceEnd(seq: ControlSequence): void {
    if (seq.loop === true && seq.steps.length > 0) {
      this.resetInternal();
      this.notifyCursor(); // 루프 랩 통지 (0, steps[0])
      return;
    }
    this._status = 'done';
    this.notifyStepChange(seq.steps.length, null);
  }

  /** 커서 이동 + 활성 런타임 폐기. 유효 인덱스로의 이동만 즉시 통지한다
   *  (끝 도달 통지는 handleSequenceEnd가 담당 — 중복 통지 방지). */
  private moveCursorTo(index: number): void {
    this.cursor = index;
    this.active = null;
    if (this.seq && index < this.seq.steps.length) this.notifyCursor();
  }

  private warnSameTickLimit(): void {
    this.warnFn(
      `player: 한 tick 안에 step 진행이 ${SAME_TICK_STEP_LIMIT}회를 초과했습니다 — ` +
        'label/goto 무한 순환 가능성이 있습니다. 이번 tick 진행을 중단하고 다음 tick에 재개합니다',
    );
  }

  private notifyCursor(): void {
    const step = this.seq?.steps[this.cursor] ?? null;
    this.notifyStepChange(this.cursor, step);
  }

  private notifyStepChange(index: number, step: ControlStep | null): void {
    for (const listener of this.stepChangeListeners) {
      listener(index, step);
    }
  }
}
