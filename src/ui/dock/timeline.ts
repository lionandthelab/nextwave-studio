// ui/dock/timeline.ts — 시퀀스 진행 타임라인 (UX_DESIGN §3.6 "Timeline")
//
// step 마커(종류 라벨) 나열 + 활성 step 강조 + "n/총" · simTime 리드아웃.
// 마커 클릭은 현재 no-op — 재생 위치 이동(마커에서 재실행)은 Phase 10
// "노드/타임라인 마커에서 재실행(결정론적 재생)" 몫이다 (ROADMAP).
//
// 계층 규칙: POJO(step kind 문자열 목록)만 받는다 — core/schema를 import하지 않는다.
// 데이터 공급(player.onStepChange / engine.onTick)은 글루(main.ts)가 중계한다.

import { COLOR, FONT, SPACE, ensureThemeStyles, styled } from '../theme';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4, 시각 토큰은 ui/theme.ts) ────

/** simTime 표시 소수 자릿수 */
const SIM_TIME_DECIMALS = 2;

const MARKER_BG = COLOR.bgRaised;
/** 활성 마커: 액센트 배경 + 어두운 텍스트 (대비 ≥ 4.5:1) + bold — 색 외 채널 병행 */
const MARKER_BG_ACTIVE = COLOR.accent;
const MARKER_BG_DONE = COLOR.successSoft;
const MARKER_BORDER = COLOR.border;

// ── 공개 타입 ───────────────────────────────────────────────────────

export interface TimelinePanel {
  readonly el: HTMLElement;
  /** 시퀀스 로드/교체 — step 종류 라벨 목록으로 마커를 다시 그린다 */
  setSequence(stepKinds: readonly string[]): void;
  /**
   * 활성 step 인덱스 갱신. stepCount와 같으면 "시퀀스 끝(done)"으로 표시한다
   * (player.onStepChange의 (stepCount, null) 통지 계약).
   */
  setActiveIndex(index: number): void;
  /** simTime 리드아웃 갱신 (engine.onTick에서 rAF당 1회) */
  setSimTime(simTimeSec: number): void;
  dispose(): void;
}

// ── 패널 ────────────────────────────────────────────────────────────

export function createTimelinePanel(): TimelinePanel {
  ensureThemeStyles();
  const el = styled(document.createElement('div'), {
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
    boxSizing: 'border-box',
  });
  el.dataset.testid = 'timeline-panel';

  // 리드아웃 줄: "n/총 N" + simTime
  const readoutRow = styled(document.createElement('div'), {
    display: 'flex',
    alignItems: 'center',
    gap: SPACE.lg,
    padding: `${SPACE.xs} ${SPACE.md}`,
    borderBottom: `1px solid ${COLOR.border}`,
    flexShrink: '0',
    fontSize: '11px',
  });
  const progressReadout = styled(document.createElement('span'), {
    color: COLOR.textStrong,
    fontFamily: FONT.mono,
  });
  progressReadout.dataset.testid = 'timeline-progress';
  const timeReadout = styled(document.createElement('span'), {
    color: COLOR.label,
    fontFamily: FONT.mono,
  });
  readoutRow.appendChild(progressReadout);
  readoutRow.appendChild(timeReadout);
  el.appendChild(readoutRow);

  // 마커 행 (가로 스크롤)
  const markerRow = styled(document.createElement('div'), {
    flex: '1',
    display: 'flex',
    alignItems: 'center',
    gap: SPACE.xs,
    padding: `${SPACE.sm} ${SPACE.md}`,
    overflowX: 'auto',
  });
  markerRow.classList.add('ui-scroll');
  el.appendChild(markerRow);

  // 빈 시퀀스 안내 (UX_DESIGN §7 "빈 플로우") — setSequence가 마커로 대체한다
  const emptyHint = styled(document.createElement('span'), {
    color: COLOR.muted,
  });
  emptyHint.textContent =
    '이 씬에는 시퀀스가 없습니다 — {scene, sequence} 봉투 JSON을 업로드하면 재생할 수 있습니다';
  emptyHint.dataset.testid = 'timeline-empty';
  markerRow.appendChild(emptyHint);

  let markers: HTMLElement[] = [];
  let stepCount = 0;
  let activeIndex = -1;

  const paintMarkers = (): void => {
    markers.forEach((marker, i) => {
      const isActive = i === activeIndex;
      const isDone = activeIndex > i || activeIndex >= stepCount;
      marker.style.background = isActive ? MARKER_BG_ACTIVE : isDone ? MARKER_BG_DONE : MARKER_BG;
      marker.style.color = isActive
        ? COLOR.onAccent
        : isDone
          ? COLOR.successText
          : COLOR.label;
      // 색 외 채널 병행 (UX §9): 활성은 bold + 액센트 테두리, 완료는 초록 테두리
      marker.style.fontWeight = isActive ? '700' : '400';
      marker.style.borderColor = isActive
        ? COLOR.accent
        : isDone
          ? COLOR.successBorder
          : MARKER_BORDER;
    });
  };

  const paintProgress = (): void => {
    if (stepCount === 0) {
      progressReadout.textContent = '시퀀스 없음';
      return;
    }
    const shown = activeIndex >= stepCount ? stepCount : Math.max(activeIndex + 1, 1);
    const doneSuffix = activeIndex >= stepCount ? ' · 완료' : '';
    progressReadout.textContent = `${shown}/총 ${stepCount}${doneSuffix}`;
  };

  paintProgress(); // 초기 상태 표기 (시퀀스 없으면 "시퀀스 없음")

  return {
    el,
    setSequence: (stepKinds): void => {
      markerRow.replaceChildren();
      if (stepKinds.length === 0) markerRow.appendChild(emptyHint);
      markers = stepKinds.map((kind, i) => {
        const marker = styled(document.createElement('button'), {
          background: MARKER_BG,
          color: COLOR.label,
          border: `1px solid ${MARKER_BORDER}`,
          borderRadius: '3px',
          padding: '2px 8px',
          fontFamily: FONT.mono,
          fontSize: '10px',
          whiteSpace: 'nowrap',
          cursor: 'default',
          flexShrink: '0',
          transition: 'background-color 0.15s ease, color 0.15s ease, border-color 0.15s ease',
        });
        marker.type = 'button';
        marker.dataset.testid = 'timeline-marker';
        marker.textContent = `${i} ${kind}`;
        // TODO(Phase 10): 마커 클릭 → 해당 노드에서 재실행(결정론적 재생, ROADMAP
        // "노드/타임라인 마커에서 재실행"). 지금은 의도적 no-op.
        marker.addEventListener('click', () => {
          /* no-op — Phase 10 재실행 */
        });
        markerRow.appendChild(marker);
        return marker;
      });
      stepCount = stepKinds.length;
      activeIndex = stepCount > 0 ? 0 : -1;
      paintMarkers();
      paintProgress();
    },
    setActiveIndex: (index): void => {
      activeIndex = index;
      paintMarkers();
      paintProgress();
    },
    setSimTime: (simTimeSec): void => {
      timeReadout.textContent = `simTime ${simTimeSec.toFixed(SIM_TIME_DECIMALS)}s`;
    },
    dispose: (): void => {
      el.remove();
    },
  };
}
