// ui/command-bar/playback.ts — 재생 트랜스포트 (UX_DESIGN §3.1 행 B 우측)
//
// ▶ 재생 / ⏸ 일시정지 / ⏹ 정지(리셋) / ⏭ 한 스텝 + 속도 선택 + 물리 상태·simTime 표시.
// 동작 배선(엔진·player·씬 리셋)은 글루(main.ts)가 PlaybackControls 콜백으로 주입한다 —
// 이 모듈은 core를 import하지 않는다(계층 규칙, CLAUDE.md §3). Play는 사람의 명시적
// 승인이다: 시퀀스는 자동 재생되지 않고 Play를 눌러야 시작된다(human-in-the-loop,
// 불변식 §2.9의 정신 — 플래너 이전 단계부터 동일 원칙 적용).
//
// 배치: 커맨드바 셸(scene-controls.ts의 mountCommandBarShell)의 **행 B 트랜스포트
// 슬롯**(`shell.rowBTransport`)에 들어가는 플렉스 아일랜드다. 아일랜드 자체는 수축하지
// 않는다 — 자연어 입력이 남는 폭을 먹고, P0 재생 버튼은 어떤 폭에서도 보존된다.
// 씬 전환 시 글루가 dispose 후 새 엔진에 다시 마운트한다(속도 select 등 뷰 상태가
// 씬을 가로질러 새지 않는다 — 결정론적 재생 준비).
//
// ── 상태 어휘 (UX_AUDIT C-18) ───────────────────────────────────────
// 구 구현은 여기(물리 루프 상태)와 뷰포트 오버레이(시퀀스 상태)에 **같은 어휘와 같은
// 시각 언어**(`● Running`)를 써서, 한 화면에 `● Running 2.66s`와 `● Idle · simTime
// 2.66s`가 동시에 떠 있었다. 사용자가 ▶를 누르기 **전부터** "Running"이 보여
// "이미 돌고 있나?"로 읽히던 지점이다.
//   · 여기(물리 루프)  → `물리 ● 가동 / 일시정지 / 정지`
//   · 뷰포트 오버레이  → `Running / Paused / Idle` (시퀀스 상태 전용 어휘)
// 두 리드아웃이 서로 다른 명사를 쓰므로 같은 화면에 있어도 모순이 아니다.
//
// 키보드: Space = 재생/일시정지 토글 (텍스트 입력·위젯 포커스 중에는 무시).

import { makeIconButton } from '../icons';
import { COLOR, SPACE, TYPE, applyType, ensureThemeStyles, styled } from '../theme';
import { COMMAND_BAR_PRIORITY, setCommandBarPriority } from './scene-controls';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4, 시각 토큰은 ui/theme.ts) ────

/** simTime 표시 소수 자릿수 */
const SIM_TIME_DECIMALS = 2;

/** 상태 텍스트 색 — 색 + 라벨 텍스트 병행이므로 색은 보조 채널이다 (UX_DESIGN §9) */
const STATE_COLORS: Readonly<Record<PlaybackStatusInfo['engineState'], string>> = {
  playing: COLOR.accentText,
  paused: COLOR.warnText,
  idle: COLOR.muted,
};

/**
 * **물리 루프** 상태 어휘. 시퀀스 상태(`Running/Paused/Idle`)와 같은 낱말을 쓰지 않는다
 * — 뷰포트 오버레이와 한 화면에 있어도 서로 다른 대상을 가리킨다는 게 읽힌다(C-18).
 * 리드아웃은 `물리 ● 가동` 형태로 조립된다(PHYSICS_READOUT_PREFIX 참조).
 */
const STATE_LABELS: Readonly<Record<PlaybackStatusInfo['engineState'], string>> = {
  playing: '가동',
  paused: '일시정지',
  idle: '정지',
};

/** 리드아웃이 가리키는 대상을 이름으로 못박는다 — 어휘 충돌의 근본 원인 제거 */
const PHYSICS_READOUT_PREFIX = '물리';

// ── 공개 타입 ───────────────────────────────────────────────────────

/** 글루(main.ts)가 엔진/player 위에서 구현해 주입하는 동작 표면 */
export interface PlaybackControls {
  play(): void;
  pause(): void;
  /** 정지 + 씬/시퀀스 리셋 (결정론적 재생 준비) */
  stop(): void;
  /** 물리 1스텝 (일시정지/idle 디버깅) */
  stepOnce(): void;
  setSpeed(speedMult: number): void;
}

export interface PlaybackStatusInfo {
  engineState: 'idle' | 'playing' | 'paused';
  simTimeSec: number;
  /** 시퀀스가 있는 씬이면 진행 상태 (없으면 미표시) */
  sequence?: {
    status: string;
    stepIndex: number;
    stepCount: number;
  };
}

export interface PlaybackBarHandle {
  readonly el: HTMLElement;
  /** 상태 표시 갱신 (engine.onTick에서 rAF당 1회) */
  update(info: PlaybackStatusInfo): void;
  /**
   * 재생/일시정지 토글. 전역 Space는 ui/shortcuts.ts 라우터가 소유하고 이걸 호출한다
   * — 키 바인딩이 모듈마다 흩어지면 Space를 3개가 나눠 갖는 상태로 수렴한다(C-6).
   */
  togglePlay(): void;
  dispose(): void;
}

// ── 순수 헬퍼 (DOM 비의존 — node 테스트 대상) ───────────────────────

/** 물리 루프 상태 → 리드아웃 텍스트 ("물리 가동" 등 — 점은 별도 요소) */
export function physicsStateLabel(state: PlaybackStatusInfo['engineState']): string {
  return `${PHYSICS_READOUT_PREFIX} ${STATE_LABELS[state]}`;
}

// ── 마운트 ──────────────────────────────────────────────────────────

export function mountPlaybackBar(
  host: HTMLElement,
  controls: PlaybackControls,
  speedOptions: readonly number[],
): PlaybackBarHandle {
  ensureThemeStyles();
  // 커맨드바 셸 행 B의 트랜스포트 아일랜드 — 배경/고정 배치는 셸이 소유한다.
  const bar = styled(document.createElement('div'), {
    display: 'flex',
    alignItems: 'center',
    gap: SPACE.xs,
    flexShrink: '0',
    color: COLOR.text,
    boxSizing: 'border-box',
    pointerEvents: 'auto',
  });
  applyType(bar, TYPE.body);
  bar.dataset.testid = 'playback-bar';
  // 바 위 상호작용이 뷰포트(OrbitControls)로 전파되지 않게 차단 (단독 마운트 대비 —
  // 커맨드바 셸 안에서는 셸의 동일 차단과 중복이지만 무해하다)
  for (const type of ['pointerdown', 'pointermove', 'pointerup', 'wheel', 'contextmenu']) {
    bar.addEventListener(type, (e) => {
      e.stopPropagation();
    });
  }

  // ▶ 재생은 이 화면 행 B의 **주요 액션**이다 — 액센트를 면으로 쓰는 유일한 버튼
  // (`--accent`는 보더만 칠해 ⏸/⏹/⏭과 시각 무게가 같았다: UX_AUDIT C-14).
  const playButton = makeIconButton('play', '재생', '재생 (Space)', 'playback-play', 'primary');
  const pauseButton = makeIconButton('pause', '일시정지', '일시정지 (Space)', 'playback-pause');
  const stopButton = makeIconButton('stop', '정지', '정지 + 씬/시퀀스 리셋', 'playback-stop');
  const stepButton = makeIconButton(
    'stepForward',
    '스텝',
    '물리 1스텝 (일시정지 중 디버깅)',
    'playback-step',
  );

  // P0 — 어떤 폭에서도 오버플로 메뉴로 밀려나지 않는다
  for (const b of [playButton, pauseButton, stopButton]) {
    setCommandBarPriority(b, COMMAND_BAR_PRIORITY.transport);
  }
  setCommandBarPriority(stepButton, COMMAND_BAR_PRIORITY.step);

  playButton.addEventListener('click', () => controls.play());
  pauseButton.addEventListener('click', () => controls.pause());
  stopButton.addEventListener('click', () => controls.stop());
  stepButton.addEventListener('click', () => controls.stepOnce());

  const speedSelect = styled(document.createElement('select'), { flexShrink: '0' });
  speedSelect.className = 'ui-select';
  speedSelect.dataset.testid = 'playback-speed';
  speedSelect.title = '재생 속도';
  speedSelect.setAttribute('aria-label', '재생 속도');
  setCommandBarPriority(speedSelect, COMMAND_BAR_PRIORITY.step);
  for (const speed of speedOptions) {
    const option = document.createElement('option');
    option.value = String(speed);
    option.textContent = `${speed}×`;
    if (speed === 1) option.selected = true;
    speedSelect.appendChild(option);
  }
  speedSelect.addEventListener('change', () => {
    controls.setSpeed(Number(speedSelect.value));
  });

  // 상태 리드아웃: "물리" + 펄스 점 + 상태 라벨 + mono simTime —
  // 색만으로 상태를 전달하지 않는다 (UX_DESIGN §9)
  const statusReadout = styled(document.createElement('span'), {
    marginLeft: SPACE.md,
    whiteSpace: 'nowrap',
    display: 'inline-flex',
    alignItems: 'center',
    flexShrink: '0',
  });
  statusReadout.dataset.testid = 'playback-status';
  setCommandBarPriority(statusReadout, COMMAND_BAR_PRIORITY.status);

  const statusScope = styled(document.createElement('span'), {
    color: COLOR.muted,
    marginRight: SPACE.xs,
  });
  applyType(statusScope, TYPE.caption);
  statusScope.textContent = PHYSICS_READOUT_PREFIX;

  const statusDot = document.createElement('span');
  statusDot.className = 'ui-dot ui-dot--idle';
  statusDot.setAttribute('aria-hidden', 'true');

  const statusLabel = styled(document.createElement('span'), { color: COLOR.muted });
  applyType(statusLabel, TYPE.caption);
  statusLabel.textContent = STATE_LABELS.idle;

  const statusTime = styled(document.createElement('span'), {
    color: COLOR.label,
    marginLeft: SPACE.sm,
  });
  applyType(statusTime, TYPE.monoReadout);

  statusReadout.appendChild(statusScope);
  statusReadout.appendChild(statusDot);
  statusReadout.appendChild(statusLabel);
  statusReadout.appendChild(statusTime);

  // 시퀀스 진행 — 물리 리드아웃과 다른 대상이므로 별도 요소로 분리해 둔다
  const sequenceReadout = styled(document.createElement('span'), {
    marginLeft: SPACE.md,
    color: COLOR.label,
    whiteSpace: 'nowrap',
    flexShrink: '0',
  });
  applyType(sequenceReadout, TYPE.caption);
  sequenceReadout.dataset.testid = 'playback-sequence';
  setCommandBarPriority(sequenceReadout, COMMAND_BAR_PRIORITY.status);

  bar.appendChild(playButton);
  bar.appendChild(pauseButton);
  bar.appendChild(stopButton);
  bar.appendChild(stepButton);
  bar.appendChild(speedSelect);
  bar.appendChild(statusReadout);
  bar.appendChild(sequenceReadout);
  host.appendChild(bar);

  // Space = 재생/일시정지 토글. 텍스트 입력뿐 아니라 **버튼/목록/다이얼로그 포커스
  // 중에도** 가로채지 않는다 — 구 isTypingTarget은 버튼 포커스 중 Space까지 삼켜
  // 충돌 로그 행을 키보드로 여는 순간 시뮬이 재생됐다 (UX_AUDIT C-6).
  let lastEngineState: PlaybackStatusInfo['engineState'] = 'idle';
  //
  // 키 바인딩 자체는 **여기서 하지 않는다.** 전역 단축키의 단일 소유자는 ui/shortcuts.ts의
  // 라우터이고(UX_AUDIT C-6), 이 모듈은 "무엇을 할지"만 제공한다. 통합자가
  // `shortcuts.register({ keys: 'Space', run: () => playbackBar.togglePlay() })`로 잇는다.
  const togglePlay = (): void => {
    // 낙관적 로컬 갱신: lastEngineState는 update()가 rAF당 1회만 새로고침하므로,
    // 같은 프레임 안의 연속 Space가 stale 상태를 읽어 토글 대신 play/pause를
    // 반복 호출하는 것을 막는다. 다음 update()가 엔진 진실로 되맞춘다.
    if (lastEngineState === 'playing') {
      controls.pause();
      lastEngineState = 'paused';
    } else {
      controls.play();
      lastEngineState = 'playing';
    }
  };

  const update = (info: PlaybackStatusInfo): void => {
    lastEngineState = info.engineState;
    statusDot.className = `ui-dot ui-dot--${info.engineState}`;
    statusLabel.textContent = STATE_LABELS[info.engineState];
    statusLabel.style.color = STATE_COLORS[info.engineState];
    statusTime.textContent = `${info.simTimeSec.toFixed(SIM_TIME_DECIMALS)}s`;
    statusReadout.setAttribute(
      'aria-label',
      `${physicsStateLabel(info.engineState)} · ${statusTime.textContent}`,
    );
    if (info.sequence) {
      const { status, stepIndex, stepCount } = info.sequence;
      const shown = Math.min(stepIndex + 1, stepCount);
      sequenceReadout.textContent = `시퀀스 ${status} · step ${shown}/${stepCount}`;
    } else {
      sequenceReadout.textContent = '';
    }
  };

  return {
    el: bar,
    update,
    togglePlay,
    dispose: (): void => {
      bar.remove();
    },
  };
}
