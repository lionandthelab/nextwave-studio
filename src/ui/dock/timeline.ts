// ui/dock/timeline.ts — 시퀀스 진행 타임라인 (UX_DESIGN §3.6 "Timeline")
//
// step 마커(종류 라벨) 나열 + 활성 step 강조 + "n/총" · simTime 리드아웃.
// 마커 클릭 → onMarkerClick(index) 콜백(Phase 10 "노드/타임라인 마커에서 재실행" — 재생
// 위치 이동/노드 강조는 글루가 결정론적 재생으로 구현, ROADMAP). 마커 색은 캔버스
// (flow-graph node-render.statusColor)와 정합: active(액센트)·done(초록)·error(빨강)·
// pending(muted). 오류 마커는 통합자가 setErrorIndices로 지정한다(예: waitForCollision
// timeout이 그 노드를 error로 마킹 — Phase 8/10 §4).
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
/** 오류 마커: error 계열 배경 (캔버스 statusColor error와 정합) */
const MARKER_BG_ERROR = COLOR.errorSoft;
const MARKER_BORDER = COLOR.border;

/** 마커 상태의 텍스트 채널 (aria — 색만으로 전달 금지, UX §9) */
const MARKER_STATUS_KO = {
  error: '오류',
  active: '실행중',
  done: '완료',
  pending: '대기',
} as const;

// ── 공개 타입 ───────────────────────────────────────────────────────

export interface TimelinePanel {
  readonly el: HTMLElement;
  /** 시퀀스 로드/교체 — step 종류 라벨 목록으로 마커를 다시 그린다 */
  setSequence(stepKinds: readonly string[]): void;
  /**
   * 활성 step 인덱스 갱신. stepCount와 같으면 "시퀀스 끝(done)"으로 표시한다
   * (player.onStepChange의 (stepCount, null) 통지 계약). Phase 10 계약의
   * setActive(index) 역할을 겸한다(뷰포트 배지·캔버스 커서와 동기).
   */
  setActiveIndex(index: number): void;
  /**
   * 오류 상태 마커 인덱스 집합 (Phase 10 — 캔버스 status 'error'와 정합). error가
   * active/done보다 우선한다. 빈 배열이면 오류 표시를 모두 지운다. setSequence는
   * 새 시퀀스마다 오류 집합을 초기화한다(런 경계 리셋).
   */
  setErrorIndices(indices: readonly number[]): void;
  /**
   * 마커 클릭 콜백 등록 (Phase 10 — 마커에서 재실행/노드 점프). 통합자가 주입한다;
   * 재생 위치 이동(결정론적 재생)·그 노드 강조는 글루 몫이다. 재등록하면 대체된다.
   */
  onMarkerClick(fn: (index: number) => void): void;
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
  let markerKinds: string[] = [];
  let stepCount = 0;
  let activeIndex = -1;
  let errorIndices = new Set<number>();
  /** 마커 클릭 콜백 (통합자 주입 — 미주입이면 클릭은 무해한 no-op) */
  let markerClickListener: ((index: number) => void) | null = null;

  const paintMarkers = (): void => {
    markers.forEach((marker, i) => {
      const isError = errorIndices.has(i);
      const isActive = !isError && i === activeIndex;
      const isDone = !isError && !isActive && (activeIndex > i || activeIndex >= stepCount);
      // 색 우선순위: error > active > done > pending (캔버스 statusColor와 정합)
      if (isError) {
        marker.style.background = MARKER_BG_ERROR;
        marker.style.color = COLOR.errorText;
        marker.style.borderColor = COLOR.errorBorder;
      } else if (isActive) {
        marker.style.background = MARKER_BG_ACTIVE;
        marker.style.color = COLOR.onAccent;
        marker.style.borderColor = COLOR.accent;
      } else if (isDone) {
        marker.style.background = MARKER_BG_DONE;
        marker.style.color = COLOR.successText;
        marker.style.borderColor = COLOR.successBorder;
      } else {
        marker.style.background = MARKER_BG;
        marker.style.color = COLOR.label;
        marker.style.borderColor = MARKER_BORDER;
      }
      // 색 외 채널 병행 (UX §9): active/error는 bold, error는 ⚠ 접두 + aria 상태 라벨
      marker.style.fontWeight = isActive || isError ? '700' : '400';
      const kind = markerKinds[i] ?? '';
      const statusKo = isError
        ? MARKER_STATUS_KO.error
        : isActive
          ? MARKER_STATUS_KO.active
          : isDone
            ? MARKER_STATUS_KO.done
            : MARKER_STATUS_KO.pending;
      marker.textContent = `${isError ? '⚠ ' : ''}${i} ${kind}`;
      marker.setAttribute('aria-label', `${i + 1}/${stepCount} ${kind} · ${statusKo}`);
      if (isActive) marker.setAttribute('aria-current', 'step');
      else marker.removeAttribute('aria-current');
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
      markerKinds = [...stepKinds];
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
          cursor: 'pointer',
          flexShrink: '0',
          transition: 'background-color 0.15s ease, color 0.15s ease, border-color 0.15s ease',
        });
        marker.type = 'button';
        marker.dataset.testid = 'timeline-marker';
        marker.textContent = `${i} ${kind}`;
        // 마커 클릭 → 통합자 콜백 (Phase 10 "노드/타임라인 마커에서 재실행" — 결정론적
        // 재생/노드 강조는 글루 몫). 미주입이면 무해한 no-op.
        marker.addEventListener('click', () => {
          markerClickListener?.(i);
        });
        markerRow.appendChild(marker);
        return marker;
      });
      stepCount = stepKinds.length;
      activeIndex = stepCount > 0 ? 0 : -1;
      errorIndices = new Set(); // 새 시퀀스 = 새 런: 오류 표시 리셋
      paintMarkers();
      paintProgress();
    },
    setActiveIndex: (index): void => {
      activeIndex = index;
      paintMarkers();
      paintProgress();
    },
    setErrorIndices: (indices): void => {
      errorIndices = new Set(indices);
      paintMarkers();
    },
    onMarkerClick: (fn): void => {
      markerClickListener = fn;
    },
    setSimTime: (simTimeSec): void => {
      timeReadout.textContent = `simTime ${simTimeSec.toFixed(SIM_TIME_DECIMALS)}s`;
    },
    dispose: (): void => {
      el.remove();
    },
  };
}
