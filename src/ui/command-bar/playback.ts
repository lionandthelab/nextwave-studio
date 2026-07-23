// ui/command-bar/playback.ts — 재생 컨트롤 (UX_DESIGN §3.1 부분집합)
//
// ▶ Play / ⏸ Pause / ⏹ Stop(리셋) / ⏭ Step / 속도 선택 + 엔진 상태·simTime 표시.
// 동작 배선(엔진·player·씬 리셋)은 글루(main.ts)가 PlaybackControls 콜백으로 주입한다 —
// 이 모듈은 core를 import하지 않는다(계층 규칙, CLAUDE.md §3). Play는 사람의 명시적
// 승인이다: 시퀀스는 자동 재생되지 않고 Play를 눌러야 시작된다(human-in-the-loop,
// 불변식 §2.9의 정신 — 플래너 이전 단계부터 동일 원칙 적용).
//
// 배치: 자체 고정 오버레이가 아니라 커맨드바 셸(scene-controls.ts의
// mountCommandBarShell)의 "중앙 슬롯"에 들어가는 플렉스 행이다 — 상단 바는
// [좌: 씬 컨트롤 | 중앙: 재생 | 우: JSON 뷰어] 하나로 응집된다 (UX_DESIGN §3.1).
// 씬 전환 시 글루가 dispose 후 새 엔진에 다시 마운트한다(속도 select 등 뷰 상태가
// 씬을 가로질러 새지 않는다 — 결정론적 재생 준비).
//
// 키보드: Space = play/pause 토글 (input/textarea 등 입력 중에는 무시).

import {
  COLOR,
  FONT,
  SPACE,
  ensureThemeStyles,
  makeButton,
  styled,
} from '../theme';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4, 시각 토큰은 ui/theme.ts) ────

/** simTime 표시 소수 자릿수 */
const SIM_TIME_DECIMALS = 2;

/** 상태 텍스트 색 — 색 + 라벨 텍스트 병행이므로 색은 보조 채널이다 (UX_DESIGN §9) */
const STATE_COLORS: Readonly<Record<PlaybackStatusInfo['engineState'], string>> = {
  playing: COLOR.accentText,
  paused: COLOR.warn,
  idle: COLOR.muted,
};

const STATE_LABELS: Readonly<Record<PlaybackStatusInfo['engineState'], string>> = {
  playing: 'Running',
  paused: 'Paused',
  idle: 'Idle',
};

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
  dispose(): void;
}

// ── 내부 헬퍼 ───────────────────────────────────────────────────────

/** 입력 위젯에 포커스가 있으면 단축키를 무시한다 (타이핑 중 Space 오발 방지) */
function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  return (
    tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || target.isContentEditable
  );
}

// ── 마운트 ──────────────────────────────────────────────────────────

export function mountPlaybackBar(
  host: HTMLElement,
  controls: PlaybackControls,
  speedOptions: readonly number[],
): PlaybackBarHandle {
  ensureThemeStyles();
  // 커맨드바 셸 중앙 슬롯에 들어가는 플렉스 행 — 배경/고정 배치는 셸이 소유한다.
  // 단독 마운트도 가능하도록 폰트는 자체 지정한다(셸과 동일 값).
  const bar = styled(document.createElement('div'), {
    display: 'flex',
    alignItems: 'center',
    gap: SPACE.sm,
    color: COLOR.text,
    fontFamily: FONT.ui,
    fontSize: '12px',
    boxSizing: 'border-box',
    pointerEvents: 'auto',
  });
  bar.dataset.testid = 'playback-bar';
  // 바 위 상호작용이 뷰포트(OrbitControls)로 전파되지 않게 차단 (단독 마운트 대비 —
  // 커맨드바 셸 안에서는 셸의 동일 차단과 중복이지만 무해하다)
  for (const type of ['pointerdown', 'pointermove', 'pointerup', 'wheel', 'contextmenu']) {
    bar.addEventListener(type, (e) => {
      e.stopPropagation();
    });
  }

  // Play는 액센트 변형 — 사람의 명시적 승인(human-in-the-loop)이 가장 중요한 동작이다
  const playButton = makeButton('▶ Play', '재생 (Space)', 'playback-play', 'accent');
  const pauseButton = makeButton('⏸ Pause', '일시정지 (Space)', 'playback-pause');
  const stopButton = makeButton('⏹ Stop', '정지 + 씬/시퀀스 리셋', 'playback-stop');
  const stepButton = makeButton('⏭ Step', '물리 1스텝 (일시정지 중 디버깅)', 'playback-step');

  playButton.addEventListener('click', () => controls.play());
  pauseButton.addEventListener('click', () => controls.pause());
  stopButton.addEventListener('click', () => controls.stop());
  stepButton.addEventListener('click', () => controls.stepOnce());

  const speedSelect = document.createElement('select');
  speedSelect.className = 'ui-select';
  speedSelect.dataset.testid = 'playback-speed';
  speedSelect.title = '재생 속도';
  speedSelect.setAttribute('aria-label', '재생 속도');
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

  // 상태 리드아웃: 펄스 점(재생 중 액센트) + 텍스트 라벨 + mono simTime —
  // 색만으로 상태를 전달하지 않는다 (UX_DESIGN §9)
  const statusReadout = styled(document.createElement('span'), {
    marginLeft: '10px',
    whiteSpace: 'nowrap',
    display: 'inline-flex',
    alignItems: 'center',
  });
  statusReadout.dataset.testid = 'playback-status';
  const statusDot = document.createElement('span');
  statusDot.className = 'ui-dot ui-dot--idle';
  statusDot.setAttribute('aria-hidden', 'true');
  const statusLabel = styled(document.createElement('span'), { color: COLOR.muted });
  statusLabel.textContent = STATE_LABELS.idle;
  const statusTime = styled(document.createElement('span'), {
    fontFamily: FONT.mono,
    color: COLOR.label,
    marginLeft: SPACE.sm,
  });
  statusReadout.appendChild(statusDot);
  statusReadout.appendChild(statusLabel);
  statusReadout.appendChild(statusTime);

  const sequenceReadout = styled(document.createElement('span'), {
    marginLeft: '10px',
    color: COLOR.label,
    whiteSpace: 'nowrap',
  });
  sequenceReadout.dataset.testid = 'playback-sequence';

  bar.appendChild(playButton);
  bar.appendChild(pauseButton);
  bar.appendChild(stopButton);
  bar.appendChild(stepButton);
  bar.appendChild(speedSelect);
  bar.appendChild(statusReadout);
  bar.appendChild(sequenceReadout);
  host.appendChild(bar);

  // Space = play/pause 토글 (입력 위젯 포커스 중에는 무시)
  let lastEngineState: PlaybackStatusInfo['engineState'] = 'idle';
  const onKeyDown = (e: KeyboardEvent): void => {
    if (e.code !== 'Space' || isTypingTarget(e.target)) return;
    e.preventDefault(); // 페이지 스크롤·포커스 버튼 재클릭 방지
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
  window.addEventListener('keydown', onKeyDown);

  const update = (info: PlaybackStatusInfo): void => {
    lastEngineState = info.engineState;
    statusDot.className = `ui-dot ui-dot--${info.engineState}`;
    statusLabel.textContent = STATE_LABELS[info.engineState];
    statusLabel.style.color = STATE_COLORS[info.engineState];
    statusTime.textContent = `${info.simTimeSec.toFixed(SIM_TIME_DECIMALS)}s`;
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
    dispose: (): void => {
      window.removeEventListener('keydown', onKeyDown);
      bar.remove();
    },
  };
}
