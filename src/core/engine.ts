// core/engine.ts — 고정 timestep 시뮬레이션 루프 (규범: docs/SIMULATION.md §2,
// docs/ARCHITECTURE.md §5, CLAUDE.md §2.3)
//
// 계약:
// - 물리는 항상 정확히 fixedDtSec 단위로만 전진한다(accumulator 패턴). 가변 dt 유입 금지.
// - alpha 보간은 시각 전용이며 물리 상태에 영향을 주지 않는다.
// - paused/idle에서도 렌더는 계속 돈다(카메라 조작 유지). 물리만 멈춘다.
// - 이 모듈은 core/types(+ core/math)만 import한다. three.js/Rapier/DOM 금지
//   (requestAnimationFrame/performance는 SIMULATION §2에서 허용된 예외).

import type { ContactEvent, PhysicsWorld } from './types';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4) ─────────────────────────────

/** 프레임 dt 상한(초). 탭 백그라운드 복귀 시 물리 스텝 폭주(spiral of death) 방지 (CLAUDE.md §9). */
export const MAX_FRAME_DT_SEC = 0.1;

/** 허용되는 재생 속도 배율. UI 재생 컨트롤과 1:1 (docs/UX_DESIGN.md). */
export const ENGINE_SPEED_OPTIONS = [0.25, 0.5, 1, 2, 4] as const;
export type EngineSpeed = (typeof ENGINE_SPEED_OPTIONS)[number];

const DEFAULT_SPEED: EngineSpeed = 1;

/** ms → s 변환 계수 (performance.now()는 ms). */
const MS_PER_SEC = 1000;

/**
 * "최신 물리 pose를 그대로 표시"하는 보간 계수 (SIMULATION §2).
 * stepOnce 디버그 표시와 stop() 이후 idle 렌더에 쓴다 — alpha=1이면 sync.apply가
 * prev 스냅샷을 무시하고 물리 진실(최신 pose)만 그린다 (CLAUDE.md §2.1).
 */
const ALPHA_LATEST = 1;

/** Engine timestepHz와 world.fixedDtSec 불일치 감지 허용 오차(초). */
const FIXED_DT_MISMATCH_TOL_SEC = 1e-9;

// ── FixedStepper: 순수 accumulator (단위 테스트 대상) ────────────────

export interface StepperAdvance {
  /** 이번 프레임에 실행해야 할 물리 스텝 수 (0 이상의 정수) */
  steps: number;
  /** 시각 보간 계수 = 잔여 accumulator / fixedDt. 항상 [0, 1) */
  alpha: number;
}

/**
 * 고정 timestep accumulator (순수 로직 — 물리/렌더/DOM 비의존).
 *
 * 프레임 dt(가변)를 누적하고, fixedDtSec 단위의 정수 스텝 수와 잔여분 비율(alpha)을
 * 산출한다. 스텝 수는 accumulator만으로 결정되므로 렌더 프레임레이트와 무관하다
 * (결정론 — SIMULATION §6). frameDtSec은 MAX_FRAME_DT_SEC으로 클램프한다.
 */
export class FixedStepper {
  private readonly fixedDtSec: number;
  private accSec = 0;

  constructor(fixedDtSec: number) {
    if (!Number.isFinite(fixedDtSec) || fixedDtSec <= 0) {
      throw new RangeError(`FixedStepper: fixedDtSec must be a finite positive number, got ${fixedDtSec}`);
    }
    this.fixedDtSec = fixedDtSec;
  }

  advance(frameDtSec: number): StepperAdvance {
    // 비정상 입력(NaN/음수/무한)은 0으로, 정상 입력은 spiral-of-death 상한으로 클램프
    const safeDtSec =
      Number.isFinite(frameDtSec) && frameDtSec > 0
        ? Math.min(frameDtSec, MAX_FRAME_DT_SEC)
        : 0;

    this.accSec += safeDtSec;

    // 정수 나눗셈으로 스텝 수 산출 — 반복 감산 대비 부동소수 누적 오차가 작다
    let steps = Math.floor(this.accSec / this.fixedDtSec);
    if (steps > 0) {
      this.accSec -= steps * this.fixedDtSec;
    }
    // 부동소수 반올림 방어: 잔여는 항상 [0, fixedDtSec) 불변식을 유지한다
    if (this.accSec < 0) {
      this.accSec = 0;
    }
    while (this.accSec >= this.fixedDtSec) {
      this.accSec -= this.fixedDtSec;
      steps += 1;
    }

    return { steps, alpha: this.accSec / this.fixedDtSec };
  }

  /** accumulator를 비운다 (stop/재로드 시 사용). */
  reset(): void {
    this.accSec = 0;
  }
}

// ── Engine: 루프 오케스트레이션 ─────────────────────────────────────

export interface EngineHooks {
  /** 물리 스텝 직전 호출 — control player가 setpoint를 적용하는 지점 (ARCHITECTURE §5 ①) */
  preStep?(simTimeSec: number, dtSec: number): void;
  /** 물리 스텝이 반환한 접촉 이벤트 — CollisionMonitor 발행 지점 (ARCHITECTURE §5 ③) */
  onContacts?(events: ContactEvent[], simTimeSec: number): void;
}

export type EngineState = 'idle' | 'playing' | 'paused';

/** UI 구독용 tick 스냅샷 (rAF당 1회) */
export interface EngineTickInfo {
  state: EngineState;
  simTimeSec: number;
}

export type EngineTickListener = (info: EngineTickInfo) => void;

/** pose 스냅샷 커밋 + 보간 반영 (core/sync.ts의 RenderSync가 구현) */
export interface PoseSync {
  /** 물리 스텝 직전: prevPose ← 현재 pose 스냅샷 (SIMULATION §5) */
  commit(): void;
  /** 렌더 직전: prev→현재 pose를 alpha로 보간해 시각 객체에 반영 */
  apply(alpha: number): void;
}

/** 렌더 대상 (render/renderer.ts의 Renderer가 구조적으로 만족) */
export interface RenderTarget {
  draw(): void;
}

export interface EngineDeps {
  world: PhysicsWorld;
  sync: PoseSync;
  render: RenderTarget;
  hooks?: EngineHooks;
}

/**
 * 고정 timestep 엔진 (재생 상태 소유 — ARCHITECTURE §7).
 *
 * 루프(프레임당): frameDt×speed → stepper → 물리 tick × steps → sync.apply(alpha)
 * → render.draw() → onTick 통지. 물리 tick 순서:
 * hooks.preStep → sync.commit() → world.step() → hooks.onContacts → simTime += fixedDt.
 *
 * paused/idle에서도 sync.apply + render.draw는 계속된다(카메라 라이브 유지).
 * stop()은 엔진 시계(simTime/accumulator)만 리셋한다 — 씬 상태 리셋은 scene-loader 소관.
 */
export class Engine {
  private readonly fixedDtSec: number;
  private readonly stepper: FixedStepper;
  private readonly deps: EngineDeps;
  private readonly tickListeners = new Set<EngineTickListener>();
  /** 물리 tick당 통지 — onTick(rAF당 1회)과 별개다. onPhysicsTick 주석 참조. */
  private readonly physicsTickListeners = new Set<EngineTickListener>();

  private _state: EngineState = 'idle';
  private _simTimeSec = 0;
  private speedMult: number = DEFAULT_SPEED;
  private lastAlpha = 0;
  private lastFrameMs = 0;
  private running = false;

  constructor(deps: EngineDeps, timestepHz: number) {
    if (!Number.isFinite(timestepHz) || timestepHz <= 0) {
      throw new RangeError(`Engine: timestepHz must be a finite positive number, got ${timestepHz}`);
    }
    this.deps = deps;
    this.fixedDtSec = 1 / timestepHz;
    this.stepper = new FixedStepper(this.fixedDtSec);

    if (Math.abs(deps.world.fixedDtSec - this.fixedDtSec) > FIXED_DT_MISMATCH_TOL_SEC) {
      console.warn(
        `Engine: timestepHz(${timestepHz} → dt ${this.fixedDtSec}s) != world.fixedDtSec(${deps.world.fixedDtSec}s). ` +
          'Engine은 world가 자신의 fixedDtSec으로 스텝한다고 가정한다 — SceneSpec.timestepHz를 양쪽에 동일하게 전달할 것.',
      );
    }
  }

  get state(): EngineState {
    return this._state;
  }

  get simTimeSec(): number {
    return this._simTimeSec;
  }

  /** rAF 루프 시작 — 'idle'(렌더만) 상태로 진입. 이미 돌고 있으면 no-op. */
  start(): void {
    if (this.running) return;
    this.running = true;
    this.lastFrameMs = performance.now();
    requestAnimationFrame(this.frame);
  }

  /** 재생 시작/재개. 루프가 아직 안 돌고 있으면 start()도 수행한다. */
  play(): void {
    this._state = 'playing';
    if (!this.running) this.start();
  }

  /** 재생 일시정지 (playing에서만 의미 있음 — 렌더는 계속). */
  pause(): void {
    if (this._state === 'playing') {
      this._state = 'paused';
    }
  }

  /**
   * 정지: idle로 전환 + 엔진 시계 리셋 (simTime=0, accumulator 비움).
   * 씬(바디 pose) 리셋은 scene-loader의 책임 — 엔진은 자기 시계만 담당한다.
   *
   * lastAlpha는 ALPHA_LATEST(1)로 둔다: idle 프레임의 sync.apply가 stale prev
   * 스냅샷 대신 최신 물리 pose를 그리게 하여, 이후 SceneHandle.reset()이 바디를
   * 텔레포트해도 화면이 즉시 물리 진실을 비추도록 한다 (CLAUDE.md §2.1).
   */
  stop(): void {
    this._state = 'idle';
    this._simTimeSec = 0;
    this.stepper.reset();
    this.lastAlpha = ALPHA_LATEST;
  }

  /**
   * rAF 루프 완전 종료 — 씬 전환/앱 해제 전용 (stop()은 시계 리셋 후에도 렌더 루프가
   * 계속 도는 것과 달리, halt() 이후 이 엔진은 어떤 프레임도 처리하지 않는다).
   * 씬을 런타임에 교체할 때 이전 엔진 루프가 남아 이중 draw/tick을 만드는 것을
   * 막는다 (Phase 6 씬 전환 라이프사이클). halt된 엔진은 재사용하지 않는다 —
   * 새 씬은 새 Engine 인스턴스로 만든다 (결정론: 씬마다 fresh 시계/accumulator).
   */
  halt(): void {
    this.running = false;
    this._state = 'idle';
  }

  /** 일시정지/idle 상태에서 물리 1스텝만 진행(디버깅). playing 중에는 no-op. */
  stepOnce(): void {
    if (this._state === 'playing') return;
    this.tickPhysics();
    // 단일 스텝 결과를 즉시 그대로 표시 (SIMULATION §2 stepOnce 규범)
    this.lastAlpha = ALPHA_LATEST;
    this.deps.sync.apply(ALPHA_LATEST);
    this.deps.render.draw();
  }

  /** 재생 속도 배율 설정. ENGINE_SPEED_OPTIONS(0.25|0.5|1|2|4) 외 값은 거부. */
  setSpeed(mult: number): void {
    if (!(ENGINE_SPEED_OPTIONS as readonly number[]).includes(mult)) {
      throw new RangeError(
        `Engine.setSpeed: ${mult} is not an allowed speed (allowed: ${ENGINE_SPEED_OPTIONS.join(', ')})`,
      );
    }
    this.speedMult = mult;
  }

  /** rAF당 1회 호출되는 UI 콜백 등록. 반환값은 해제 함수. */
  onTick(listener: EngineTickListener): () => void {
    this.tickListeners.add(listener);
    return () => {
      this.tickListeners.delete(listener);
    };
  }

  /**
   * **물리 tick당** 1회 호출되는 콜백 등록 (onTick과 다르다). 반환값은 해제 함수.
   *
   * `onTick`은 rAF당 1회다 — 240Hz 물리에 60Hz 표본이므로 두 통지 사이에 4 tick이
   * 지나가고, 프레임이 느리면 `MAX_FRAME_DT_SEC` 클램프 때문에 최대 24 tick을 건너뛴다.
   * UI 리드아웃에는 충분하지만 **"특정 step의 첫 tick에서 무엇이 어디 있었나"** 같은
   * 계측에는 쓸 수 없다 — 그 표본은 프레임 타이밍에 따라 매번 다른 순간을 잡는다
   * (실측: 같은 씬·같은 초기 상태에서 놓기 거리가 0.019 / 0.021 / 0.044로 갈렸다).
   *
   * 통지 시점은 tick의 **끝**이다 — 리스너는 그 tick의 물리 결과(step 후 pose)를 본다.
   */
  onPhysicsTick(listener: EngineTickListener): () => void {
    this.physicsTickListeners.add(listener);
    return () => {
      this.physicsTickListeners.delete(listener);
    };
  }

  // ── 내부 루프 ─────────────────────────────────────────────────────

  private readonly frame = (nowMs: number): void => {
    if (!this.running) return;
    const frameDtSec = (nowMs - this.lastFrameMs) / MS_PER_SEC;
    this.lastFrameMs = nowMs;

    if (this._state === 'playing') {
      const { steps, alpha } = this.stepper.advance(frameDtSec * this.speedMult);
      for (let i = 0; i < steps; i += 1) {
        this.tickPhysics();
      }
      this.lastAlpha = alpha;
    }

    // paused/idle에서도 렌더는 계속 — 물리만 멈춘다 (SIMULATION §2)
    this.deps.sync.apply(this.lastAlpha);
    this.deps.render.draw();
    this.emitTick();

    requestAnimationFrame(this.frame);
  };

  /** 물리 1 tick: preStep → commit → step → onContacts → simTime 전진 */
  private tickPhysics(): void {
    const dtSec = this.fixedDtSec;
    this.deps.hooks?.preStep?.(this._simTimeSec, dtSec);
    this.deps.sync.commit();
    const contacts = this.deps.world.step();
    this.deps.hooks?.onContacts?.(contacts, this._simTimeSec);
    this._simTimeSec += dtSec;
    this.emitPhysicsTick();
  }

  private emitPhysicsTick(): void {
    if (this.physicsTickListeners.size === 0) return;
    const info: EngineTickInfo = { state: this._state, simTimeSec: this._simTimeSec };
    for (const listener of this.physicsTickListeners) {
      listener(info);
    }
  }

  private emitTick(): void {
    if (this.tickListeners.size === 0) return;
    const info: EngineTickInfo = { state: this._state, simTimeSec: this._simTimeSec };
    for (const listener of this.tickListeners) {
      listener(info);
    }
  }
}
