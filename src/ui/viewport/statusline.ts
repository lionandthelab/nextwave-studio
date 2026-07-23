// ui/viewport/statusline.ts — 뷰포트 실행 오버레이 (UX_DESIGN §3.3 "실행 오버레이")
//
// 뷰포트 좌하단에 [씬 이름 · ● 재생 상태 · simTime · (시퀀스 있으면) step n/N]을
// 표시한다. 순수 표시 전용 오버레이 — pointer-events: none으로 뷰포트 orbit/선택을
// 절대 가로막지 않는다. 데이터는 글루(main.ts)가 engine.onTick에서 update()로 공급한다
// (이 모듈은 core를 import하지 않는다 — 계층 규칙, CLAUDE.md §3).
//
// 배치 계약 (Phase 7 워크스페이스): host는 "positioned"(position:relative 등) 뷰포트
// 슬롯이어야 한다 — 오버레이는 host 기준 absolute로 좌하단/중앙에 앉는다
// (기존 document.body fixed 오버레이 방식은 워크스페이스 그리드 편입으로 대체됨).
//
// 빈 씬 안내 (UX_DESIGN §7 "빈 씬"): 엔티티가 0개인 씬이면 뷰포트 중앙에 안내 문구를
// 함께 띄운다 — 마운트 시 emptyScene 플래그로 결정되고, 씬 편집(Scene Builder)으로
// 엔티티가 생기면 통합자가 setEmptyHintVisible(false)로 숨긴다.

import {
  COLOR,
  FONT,
  RADIUS,
  SPACE,
  Z_INDEX,
  ensureThemeStyles,
  styled,
} from '../theme';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4) ─────────────────────────────

/** 뷰포트 슬롯 좌하단 오프셋 (독은 워크스페이스 그리드의 별도 행 — 겹치지 않는다) */
const STATUS_BOTTOM_PX = 12;
const STATUS_LEFT_PX = 12;
/** simTime 표시 소수 자릿수 (playback.ts와 동일 규약) */
const SIM_TIME_DECIMALS = 2;

const STATE_LABELS: Readonly<Record<ViewportStatusInfo['engineState'], string>> = {
  playing: 'Running',
  paused: 'Paused',
  idle: 'Idle',
};

const STATE_TEXT_COLORS: Readonly<Record<ViewportStatusInfo['engineState'], string>> = {
  playing: COLOR.accentText,
  paused: COLOR.warn,
  idle: COLOR.muted,
};

// ── 공개 타입 ───────────────────────────────────────────────────────

export interface ViewportStatusInfo {
  engineState: 'idle' | 'playing' | 'paused';
  simTimeSec: number;
  /** 시퀀스가 있는 씬이면 진행 배지 표시 (stepCount 0이면 미표시) */
  sequence?: {
    stepIndex: number;
    stepCount: number;
  };
}

export interface ViewportStatusOptions {
  /** 씬 이름 (spec.name) */
  sceneName: string;
  /** 엔티티 0개 씬 — 중앙 빈 씬 안내 표시 (UX_DESIGN §7) */
  emptyScene?: boolean;
}

export interface ViewportStatusHandle {
  readonly el: HTMLElement;
  /** 상태 갱신 (engine.onTick에서 rAF당 1회) */
  update(info: ViewportStatusInfo): void;
  /** 빈 씬 중앙 안내 표시/숨김 — Scene Builder로 첫 엔티티가 생기면 숨긴다 (통합자 호출) */
  setEmptyHintVisible(visible: boolean): void;
  dispose(): void;
}

// ── 마운트 ──────────────────────────────────────────────────────────

export function mountViewportStatus(
  host: HTMLElement,
  opts: ViewportStatusOptions,
): ViewportStatusHandle {
  ensureThemeStyles();

  const line = styled(document.createElement('div'), {
    position: 'absolute',
    left: `${STATUS_LEFT_PX}px`,
    bottom: `${STATUS_BOTTOM_PX}px`,
    zIndex: Z_INDEX.bar,
    display: 'flex',
    alignItems: 'center',
    gap: SPACE.md,
    padding: `3px ${SPACE.lg}`,
    background: COLOR.bgPanel,
    border: `1px solid ${COLOR.border}`,
    borderRadius: RADIUS.sm,
    color: COLOR.text,
    fontFamily: FONT.ui,
    fontSize: '12px',
    lineHeight: '1.6',
    whiteSpace: 'nowrap',
    pointerEvents: 'none', // 순수 표시 — 뷰포트 상호작용을 가로막지 않는다
    userSelect: 'none',
  });
  line.dataset.testid = 'viewport-status';

  const sceneNameEl = styled(document.createElement('span'), {
    color: COLOR.textStrong,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    maxWidth: '220px',
  });
  sceneNameEl.textContent = opts.sceneName;
  sceneNameEl.title = opts.sceneName;

  const sep = (): HTMLElement => {
    const dot = styled(document.createElement('span'), { color: COLOR.muted });
    dot.textContent = '·';
    return dot;
  };

  // 상태: 펄스 점(재생 중 액센트) + 텍스트 라벨 병행 (색만으로 전달 금지 — UX §9)
  const stateWrap = styled(document.createElement('span'), { display: 'inline-flex', alignItems: 'center' });
  const stateDot = document.createElement('span');
  stateDot.className = 'ui-dot ui-dot--idle';
  stateDot.setAttribute('aria-hidden', 'true');
  const stateLabel = styled(document.createElement('span'), { color: COLOR.muted });
  stateLabel.textContent = STATE_LABELS.idle;
  stateWrap.appendChild(stateDot);
  stateWrap.appendChild(stateLabel);

  const timeEl = styled(document.createElement('span'), {
    fontFamily: FONT.mono,
    color: COLOR.label,
  });
  timeEl.textContent = 'simTime 0.00s';

  const stepBadge = styled(document.createElement('span'), {
    fontFamily: FONT.mono,
    color: COLOR.label,
    display: 'none',
  });
  stepBadge.dataset.testid = 'viewport-status-step';

  line.appendChild(sceneNameEl);
  line.appendChild(sep());
  line.appendChild(stateWrap);
  line.appendChild(sep());
  line.appendChild(timeEl);
  line.appendChild(stepBadge);
  host.appendChild(line);

  // 빈 씬 중앙 안내 (UX_DESIGN §7) — 씬 수명 동안 고정, 상호작용 없음
  let emptyHint: HTMLElement | null = null;
  if (opts.emptyScene === true) {
    emptyHint = styled(document.createElement('div'), {
      position: 'absolute',
      left: '50%',
      top: '50%',
      transform: 'translate(-50%, -50%)',
      zIndex: Z_INDEX.bar,
      padding: `${SPACE.lg} 20px`,
      background: COLOR.bgPanel,
      border: `1px dashed ${COLOR.borderStrong}`,
      borderRadius: RADIUS.md,
      color: COLOR.muted,
      fontFamily: FONT.ui,
      fontSize: '13px',
      textAlign: 'center',
      lineHeight: '1.8',
      pointerEvents: 'none',
      userSelect: 'none',
    });
    emptyHint.dataset.testid = 'viewport-empty-hint';
    emptyHint.textContent =
      '빈 씬입니다 — 상단의 씬 프리셋을 선택하거나 씬 JSON을 업로드하세요';
    host.appendChild(emptyHint);
  }

  const update = (info: ViewportStatusInfo): void => {
    stateDot.className = `ui-dot ui-dot--${info.engineState}`;
    stateLabel.textContent = STATE_LABELS[info.engineState];
    stateLabel.style.color = STATE_TEXT_COLORS[info.engineState];
    timeEl.textContent = `simTime ${info.simTimeSec.toFixed(SIM_TIME_DECIMALS)}s`;
    if (info.sequence && info.sequence.stepCount > 0) {
      const shown = Math.min(info.sequence.stepIndex + 1, info.sequence.stepCount);
      stepBadge.textContent = `· step ${shown}/${info.sequence.stepCount}`;
      stepBadge.style.display = '';
    } else {
      stepBadge.style.display = 'none';
    }
  };

  return {
    el: line,
    update,
    setEmptyHintVisible: (visible): void => {
      if (emptyHint) emptyHint.style.display = visible ? '' : 'none';
    },
    dispose: (): void => {
      line.remove();
      emptyHint?.remove();
    },
  };
}
