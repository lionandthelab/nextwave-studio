// ui/command-bar/playback.ts — 재생 컨트롤 바 (UX_DESIGN §3.1 부분집합)
//
// ▶ Play / ⏸ Pause / ⏹ Stop(리셋) / ⏭ Step / 속도 선택 + 엔진 상태·simTime 표시.
// 동작 배선(엔진·player·씬 리셋)은 글루(main.ts)가 PlaybackControls 콜백으로 주입한다 —
// 이 모듈은 core를 import하지 않는다(계층 규칙, CLAUDE.md §3). Play는 사람의 명시적
// 승인이다: 시퀀스는 자동 재생되지 않고 Play를 눌러야 시작된다(human-in-the-loop,
// 불변식 §2.9의 정신 — 플래너 이전 단계부터 동일 원칙 적용).
//
// 키보드: Space = play/pause 토글 (input/textarea 등 입력 중에는 무시).

// ── 상수 (매직넘버 금지 — CLAUDE.md §4) ─────────────────────────────

/** 오류 오버레이(z 9999)·관절 패널(z 100)보다 아래, 캔버스보다 위 (dock과 동일 층) */
const BAR_Z_INDEX = '90';
/** simTime 표시 소수 자릿수 */
const SIM_TIME_DECIMALS = 2;

const STATE_COLORS: Readonly<Record<PlaybackStatusInfo['engineState'], string>> = {
  playing: '#27ae60',
  paused: '#e2b93d',
  idle: '#7a808a',
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

function styled<T extends HTMLElement>(el: T, style: Partial<CSSStyleDeclaration>): T {
  Object.assign(el.style, style);
  return el;
}

function makeButton(label: string, title: string, testId: string): HTMLButtonElement {
  const button = styled(document.createElement('button'), {
    background: '#22252b',
    color: '#cfd3d9',
    border: '1px solid #3a3f47',
    borderRadius: '4px',
    padding: '3px 12px',
    fontFamily: 'inherit',
    fontSize: '12px',
    cursor: 'pointer',
  });
  button.type = 'button';
  button.textContent = label;
  button.title = title;
  button.dataset.testid = testId;
  return button;
}

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
  const bar = styled(document.createElement('div'), {
    position: 'fixed',
    top: '0',
    left: '0',
    right: '0',
    zIndex: BAR_Z_INDEX,
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    padding: '6px 10px',
    background: 'rgba(16, 18, 22, 0.94)',
    borderBottom: '1px solid #2e3238',
    color: '#cfd3d9',
    fontFamily: 'ui-monospace, SFMono-Regular, Consolas, monospace',
    fontSize: '12px',
    boxSizing: 'border-box',
    pointerEvents: 'auto',
  });
  bar.dataset.testid = 'playback-bar';
  // 바 위 상호작용이 뷰포트(OrbitControls)로 전파되지 않게 차단
  for (const type of ['pointerdown', 'pointermove', 'pointerup', 'wheel', 'contextmenu']) {
    bar.addEventListener(type, (e) => {
      e.stopPropagation();
    });
  }

  const playButton = makeButton('▶ Play', '재생 (Space)', 'playback-play');
  const pauseButton = makeButton('⏸ Pause', '일시정지 (Space)', 'playback-pause');
  const stopButton = makeButton('⏹ Stop', '정지 + 씬/시퀀스 리셋', 'playback-stop');
  const stepButton = makeButton('⏭ Step', '물리 1스텝 (일시정지 중 디버깅)', 'playback-step');

  playButton.addEventListener('click', () => controls.play());
  pauseButton.addEventListener('click', () => controls.pause());
  stopButton.addEventListener('click', () => controls.stop());
  stepButton.addEventListener('click', () => controls.stepOnce());

  const speedSelect = styled(document.createElement('select'), {
    background: '#22252b',
    color: '#cfd3d9',
    border: '1px solid #3a3f47',
    borderRadius: '4px',
    padding: '3px 4px',
    fontFamily: 'inherit',
    fontSize: '12px',
  });
  speedSelect.dataset.testid = 'playback-speed';
  speedSelect.title = '재생 속도';
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

  const statusReadout = styled(document.createElement('span'), {
    marginLeft: '10px',
    whiteSpace: 'nowrap',
  });
  statusReadout.dataset.testid = 'playback-status';

  const sequenceReadout = styled(document.createElement('span'), {
    marginLeft: '10px',
    color: '#9aa0a8',
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
    statusReadout.textContent = `● ${STATE_LABELS[info.engineState]} · ${info.simTimeSec.toFixed(SIM_TIME_DECIMALS)}s`;
    statusReadout.style.color = STATE_COLORS[info.engineState];
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
