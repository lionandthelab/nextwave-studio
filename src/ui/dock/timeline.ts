// ui/dock/timeline.ts — 시퀀스 진행 타임라인 (UX_DESIGN §3.6 "타임라인")
//
// step 마커(종류 라벨) 나열 + 활성 step 강조 + "n/총" · simTime 리드아웃.
// 마커 클릭 → onMarkerClick(index) 콜백(Phase 10 "노드/타임라인 마커에서 재실행" — 재생
// 위치 이동/노드 강조는 글루가 결정론적 재생으로 구현, ROADMAP). 마커 색은 캔버스
// (flow-graph node-render.statusColor)와 정합: active(액센트)·done(초록)·error(충돌 램프)·
// pending(muted). 오류 마커는 통합자가 setErrorIndices로 지정한다(예: waitForCollision
// timeout이 그 노드를 error로 마킹 — Phase 8/10 §4).
//
// ── Phase 11에서 닫는 감사 항목 ─────────────────────────────────────
// C-10 마커를 **flex-wrap**으로 접는다. 노드 42개에서 31개만 보이고 11개가 우측
//      2797px 지점(가로 스크롤 뒤)에 있는데 정작 그 아래 독 여백 150px는 비어 있었다.
//      마커 히트 영역도 17px → ≥24px로 올린다(WCAG 2.2 SC 2.5.8).
//      방향키 탐색은 rovingTabindex — 마커 42개를 탭 42번으로 지나가게 하지 않는다.
// C-1  리드아웃(진행 · simTime)과 진행 바를 `stripEl`로 분리한다. 독이 이걸 탭바 우측에
//      심으면 **독이 접혀 있어도** 진행 상황이 보인다 → 접기의 정보 손실이 0이다.
//
// 계층 규칙: POJO(step kind 문자열 목록)만 받는다 — core/schema를 import하지 않는다.
// 데이터 공급(player.onStepChange / engine.onTick)은 글루(main.ts)가 중계한다.

import { rovingTabindex } from '../a11y';
import { icon } from '../icons';
import {
  BORDER,
  BORDER_WIDTH,
  COLLISION,
  COLOR,
  ICON,
  MOTION,
  RADIUS,
  SPACE,
  SURFACE,
  TYPE,
  applyType,
  ensureThemeStyles,
  styled,
  tr,
} from '../theme';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4, 시각 토큰은 ui/theme.ts) ────

/** simTime 표시 소수 자릿수 */
const SIM_TIME_DECIMALS = 2;
/** 인라인 스트립 진행 바 폭 */
const PROGRESS_BAR_WIDTH_PX = 96;
/** 진행 바 두께 */
const PROGRESS_BAR_HEIGHT_PX = 4;
/** 마커 최소 높이 — WCAG 2.2 SC 2.5.8 (24×24 표적) */
const MARKER_MIN_HEIGHT_PX = 24;

const MARKER_BG = SURFACE.raised;
/** 활성 마커: 액센트 배경 + 어두운 텍스트 (대비 ≥ 4.5:1) + bold — 색 외 채널 병행 */
const MARKER_BG_ACTIVE = COLOR.accent;
const MARKER_BG_DONE = COLOR.successSoft;
/** 오류 마커: 충돌 램프 배경 (캔버스 statusColor error와 정합 — C-7 단일 램프) */
const MARKER_BG_ERROR = COLLISION.soft;
const MARKER_BORDER = BORDER.default;

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
  /**
   * 탭바 인라인 스트립용 리드아웃 (진행 `1/7` · simTime `2.66s` · 진행 바).
   * 독의 `mountDock(host, tabs, { strip })`에 넘기면 **접힘 상태에서도** 보인다(C-1).
   * 넘기지 않으면 아무 데도 붙지 않으므로 통합자가 반드시 배선해야 한다.
   */
  readonly stripEl: HTMLElement;
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

  // ── 인라인 스트립: "1/7 · 2.66s ▓▓░░░" (독 탭바 우측에 심긴다 — C-1) ──
  const stripEl = styled(document.createElement('div'), {
    display: 'flex',
    alignItems: 'center',
    gap: SPACE.md,
    minWidth: '0',
  });
  stripEl.dataset.testid = 'timeline-strip';

  const progressReadout = styled(document.createElement('span'), {
    color: COLOR.textStrong,
    whiteSpace: 'nowrap',
  });
  applyType(progressReadout, TYPE.monoReadout);
  progressReadout.dataset.testid = 'timeline-progress';

  const separator = styled(document.createElement('span'), { color: COLOR.muted });
  separator.textContent = '·';
  separator.setAttribute('aria-hidden', 'true');

  const timeReadout = styled(document.createElement('span'), {
    color: COLOR.label,
    whiteSpace: 'nowrap',
  });
  applyType(timeReadout, TYPE.monoReadout);
  timeReadout.dataset.testid = 'timeline-time';
  timeReadout.title = '시뮬레이션 시간 (simTime)';

  // 진행 바 — 접힌 독에서도 "얼마나 왔는지"를 주변시로 잡히게 한다.
  // aria-live는 걸지 않는다(매 프레임 갱신 → 스크린리더 포화). role=progressbar는
  // 포커스/조회 시에만 읽히므로 안전하다.
  const progressBar = styled(document.createElement('div'), {
    position: 'relative',
    width: `${PROGRESS_BAR_WIDTH_PX}px`,
    height: `${PROGRESS_BAR_HEIGHT_PX}px`,
    borderRadius: RADIUS.full,
    background: SURFACE.sunken,
    overflow: 'hidden',
    flexShrink: '0',
  });
  progressBar.dataset.testid = 'timeline-progressbar';
  progressBar.setAttribute('role', 'progressbar');
  progressBar.setAttribute('aria-label', '시퀀스 진행');
  const progressFill = styled(document.createElement('div'), {
    height: '100%',
    width: '0%',
    background: COLOR.accent,
    borderRadius: RADIUS.full,
    transition: tr('width', MOTION.base),
  });
  progressBar.appendChild(progressFill);

  stripEl.appendChild(progressReadout);
  stripEl.appendChild(separator);
  stripEl.appendChild(timeReadout);
  stripEl.appendChild(progressBar);

  // ── 마커 영역 (flex-wrap — 가로 스크롤 대신 남는 세로 공간으로 접는다, C-10) ──
  const markerRow = styled(document.createElement('div'), {
    flex: '1 1 auto',
    display: 'flex',
    flexWrap: 'wrap',
    alignContent: 'flex-start',
    alignItems: 'center',
    gap: SPACE.xs,
    padding: `${SPACE.md} ${SPACE.lg}`,
    overflowY: 'auto',
    boxSizing: 'border-box',
    minHeight: '0',
  });
  markerRow.classList.add('ui-scroll');
  markerRow.setAttribute('role', 'toolbar');
  markerRow.setAttribute('aria-label', '시퀀스 단계 마커');
  el.appendChild(markerRow);

  // 빈 시퀀스 안내 (UX_DESIGN §7 "빈 플로우") — setSequence가 마커로 대체한다
  const emptyHint = styled(document.createElement('span'), { color: COLOR.muted });
  applyType(emptyHint, TYPE.body);
  emptyHint.textContent =
    '이 씬에는 시퀀스가 없습니다 — {scene, sequence} 봉투 JSON을 업로드하면 재생할 수 있습니다';
  emptyHint.dataset.testid = 'timeline-empty';
  markerRow.appendChild(emptyHint);

  interface Marker {
    readonly button: HTMLButtonElement;
    readonly labelEl: HTMLElement;
    iconEl: SVGSVGElement | null;
  }

  let markers: Marker[] = [];
  let markerKinds: string[] = [];
  let stepCount = 0;
  let activeIndex = -1;
  let errorIndices = new Set<number>();
  /** 마커 클릭 콜백 (통합자 주입 — 미주입이면 클릭은 무해한 no-op) */
  let markerClickListener: ((index: number) => void) | null = null;

  const roving = rovingTabindex(markerRow, [], {
    // wrap 레이아웃이라 ←→ 와 ↑↓ 를 모두 선형 이동으로 받는다
    orientation: 'both',
    onActivate: (_el, index) => {
      markerClickListener?.(index);
    },
  });

  const paintMarkers = (): void => {
    markers.forEach((marker, i) => {
      const isError = errorIndices.has(i);
      const isActive = !isError && i === activeIndex;
      const isDone = !isError && !isActive && (activeIndex > i || activeIndex >= stepCount);
      const { button } = marker;
      // 색 우선순위: error > active > done > pending (캔버스 statusColor와 정합)
      if (isError) {
        button.style.background = MARKER_BG_ERROR;
        button.style.color = COLLISION.text;
        button.style.borderColor = COLLISION.border;
      } else if (isActive) {
        button.style.background = MARKER_BG_ACTIVE;
        button.style.color = COLOR.onAccent;
        button.style.borderColor = COLOR.accent;
      } else if (isDone) {
        button.style.background = MARKER_BG_DONE;
        button.style.color = COLOR.successText;
        button.style.borderColor = COLOR.successBorder;
      } else {
        button.style.background = MARKER_BG;
        button.style.color = COLOR.label;
        button.style.borderColor = MARKER_BORDER;
      }
      // 색 외 채널 병행 (UX §9): active/error는 bold, error는 alert 아이콘 + aria 상태 라벨
      button.style.fontWeight = isActive || isError ? '700' : '400';
      if (isError && marker.iconEl === null) {
        const svg = icon('alert', ICON.sm);
        marker.iconEl = svg;
        button.insertBefore(svg, marker.labelEl);
      } else if (!isError && marker.iconEl !== null) {
        marker.iconEl.remove();
        marker.iconEl = null;
      }
      const kind = markerKinds[i] ?? '';
      const statusKo = isError
        ? MARKER_STATUS_KO.error
        : isActive
          ? MARKER_STATUS_KO.active
          : isDone
            ? MARKER_STATUS_KO.done
            : MARKER_STATUS_KO.pending;
      button.setAttribute('aria-label', `${i + 1}/${stepCount} ${kind} · ${statusKo}`);
      if (isActive) button.setAttribute('aria-current', 'step');
      else button.removeAttribute('aria-current');
    });
  };

  const paintProgress = (): void => {
    if (stepCount === 0) {
      progressReadout.textContent = '시퀀스 없음';
      progressReadout.removeAttribute('aria-label');
      separator.style.display = 'none';
      progressBar.style.display = 'none';
      return;
    }
    separator.style.display = '';
    progressBar.style.display = '';
    const shown = activeIndex >= stepCount ? stepCount : Math.max(activeIndex + 1, 1);
    const done = activeIndex >= stepCount;
    progressReadout.textContent = `${shown}/${stepCount}${done ? ' 완료' : ''}`;
    progressReadout.setAttribute(
      'aria-label',
      `${stepCount}단계 중 ${shown}단계${done ? ' · 완료' : ''}`,
    );
    const ratio = shown / stepCount;
    progressFill.style.width = `${(ratio * 100).toFixed(1)}%`;
    progressFill.style.background = done ? COLOR.success : COLOR.accent;
    progressBar.setAttribute('aria-valuemin', '0');
    progressBar.setAttribute('aria-valuemax', String(stepCount));
    progressBar.setAttribute('aria-valuenow', String(shown));
    progressBar.setAttribute('aria-valuetext', `${stepCount}단계 중 ${shown}단계`);
  };

  const setSimTime = (simTimeSec: number): void => {
    const text = `${simTimeSec.toFixed(SIM_TIME_DECIMALS)}s`;
    timeReadout.textContent = text;
    timeReadout.setAttribute('aria-label', `시뮬레이션 시간 ${text}`);
  };

  setSimTime(0);
  paintProgress(); // 초기 상태 표기 (시퀀스 없으면 "시퀀스 없음")

  return {
    el,
    stripEl,
    setSequence: (stepKinds): void => {
      markerRow.replaceChildren();
      if (stepKinds.length === 0) markerRow.appendChild(emptyHint);
      markerKinds = [...stepKinds];
      markers = stepKinds.map((kind, i) => {
        const button = styled(document.createElement('button'), {
          display: 'inline-flex',
          alignItems: 'center',
          gap: SPACE.xxs,
          boxSizing: 'border-box',
          background: MARKER_BG,
          color: COLOR.label,
          border: `${BORDER_WIDTH.hair} solid ${MARKER_BORDER}`,
          borderRadius: RADIUS.sm,
          // 2px 8px → 24px 이상 표적 (WCAG 2.2 SC 2.5.8)
          padding: `${SPACE.sm} ${SPACE.md}`,
          minHeight: `${MARKER_MIN_HEIGHT_PX}px`,
          whiteSpace: 'nowrap',
          cursor: 'pointer',
          flexShrink: '0',
          transition: `${tr('background-color', MOTION.instant)}, ${tr('color', MOTION.instant)}, ${tr('border-color', MOTION.instant)}`,
        });
        applyType(button, TYPE.monoMicro);
        button.type = 'button';
        button.dataset.testid = 'timeline-marker';
        // step kind는 도메인 식별자(영문)다 — 한국어 TTS가 철자 나열로 읽지 않게 lang 명시
        const labelEl = document.createElement('span');
        labelEl.setAttribute('lang', 'en');
        labelEl.textContent = `${i} ${kind}`;
        button.appendChild(labelEl);
        // 마커 클릭 → 통합자 콜백 (Phase 10 "노드/타임라인 마커에서 재실행" — 결정론적
        // 재생/노드 강조는 글루 몫). 미주입이면 무해한 no-op.
        button.addEventListener('click', () => {
          markerClickListener?.(i);
        });
        markerRow.appendChild(button);
        return { button, labelEl, iconEl: null };
      });
      stepCount = stepKinds.length;
      activeIndex = stepCount > 0 ? 0 : -1;
      errorIndices = new Set(); // 새 시퀀스 = 새 런: 오류 표시 리셋
      roving.setItems(markers.map((m) => m.button));
      roving.setActive(0);
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
    setSimTime,
    dispose: (): void => {
      roving.dispose();
      stripEl.remove();
      el.remove();
    },
  };
}
