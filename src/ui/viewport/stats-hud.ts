// ui/viewport/stats-hud.ts — 실행 계측 HUD (UX_AUDIT C-15)
//
// FPS · real-time factor(RTF) · 물리 스텝 시간 · 엔티티/콜라이더 수를 뷰포트 우상단에
// 접이식으로 표시한다. 이전에는 `src/ui` + `src/render` 전수 검색에서 `fps`/`rtf`/
// `realtime`/`substep` 문자열이 0건이었다 — 시뮬레이터가 신뢰를 얻는 숫자가 화면에
// 하나도 없었다는 뜻이다. 씬이 무거워져 물리가 실시간을 못 따라가도 사용자는
// "좀 느린가?"만 느꼈다.
//
// **RTF는 결정론 주장의 시각적 증거다.** PRD NFR-2("프레임 예산 내 유지")와 CLAUDE.md
// §2.3(고정 timestep)은 런타임에 검증 가능해야 의미가 있고, Gazebo/Isaac Sim 사용자는
// RTF를 습관적으로 먼저 본다.
//
// ── 계층 규칙 (CLAUDE.md §3) ─────────────────────────────────────────
// 이 모듈은 three.js/Rapier를 import하지 않는다. **순수 숫자만 받는다** — 통계 산출은
// accumulator를 소유한 `core/engine`이 콜백으로 노출하고, 여기서는 그리기만 한다.
//
// ── 자릿수 폭 고정 ───────────────────────────────────────────────────
// 매 프레임 갱신되는 수치가 좌우로 떨리면 그 자체가 신뢰도 문제다. 두 겹으로 막는다:
//   (a) TYPE.monoReadout — 모노 트랙 + tabular-nums로 **글리프 폭**을 고정
//   (b) 포매터의 padStart — 자릿수가 줄어도 **문자 수**를 고정 (59.8 → 9.8 전이에서 밀림 방지)
// 값 셀은 white-space: pre라 선행 공백이 접히지 않는다.

import {
  BORDER,
  BORDER_WIDTH,
  COLLISION,
  COLOR,
  ICON,
  MOTION,
  RADIUS,
  SHADOW,
  SPACE,
  SURFACE,
  TYPE,
  Z_INDEX,
  applyType,
  ensureThemeStyles,
  styled,
  tr,
} from '../theme';
import { icon, makeIconButton } from '../icons';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4) ─────────────────────────────

/** 뷰포트 슬롯 우상단 오프셋 */
const HUD_TOP_PX = 12;
const HUD_RIGHT_PX = 12;

/** 자체 rAF FPS 측정 창 — 짧으면 값이 튀고 길면 반응이 늦다 */
const FPS_SAMPLE_WINDOW_MS = 500;

const FPS_DECIMALS = 1;
const RTF_DECIMALS = 2;
const PHYSICS_MS_DECIMALS = 1;

/** 포매터 고정 폭 (자릿수가 줄어도 문자 수가 유지된다 — 파일 헤더 (b)) */
const FPS_FIELD_WIDTH = 5; // '120.0' / ' 59.8'
const RTF_FIELD_WIDTH = 5; // '12.34' / ' 1.00'
const PHYSICS_MS_FIELD_WIDTH = 5; // '100.0' / '  2.0'
const COUNT_FIELD_WIDTH = 4; // '1024' / '  63'

/**
 * RTF 건전성 경계. 1.00×가 실시간이고, 그 아래는 물리가 벽시계를 못 따라간다는 뜻이다.
 * 0.9 미만이면 경고, 0.5 미만이면 충돌 램프(가장 강한 경보 색)로 올린다.
 */
const RTF_WARN_BELOW = 0.9;
const RTF_CRITICAL_BELOW = 0.5;

/** 값이 아직 없음 (측정 전) */
const DASH = '—';

/** 세그먼트 구분자 (한 줄 요약 — run-overlay와 동일 규약) */
const SEGMENT_SEP = ' · ';

// ── 공개 타입 ───────────────────────────────────────────────────────

/**
 * 1회 계측 스냅샷. `update()`는 **부분 필드**를 받는다 — FPS는 HUD가 자체 rAF로 측정할
 * 수도, 통합자가 넣어 줄 수도 있다(둘 다 지원). 통합자가 `fps`를 한 번이라도 주면
 * 자체 측정은 멈춘다(이중 진실 제거).
 */
export interface StatsSample {
  /** 렌더 프레임레이트 (프레임/초) */
  fps: number;
  /** real-time factor — simTime 진행 / 벽시계 경과. 1.00이 실시간 */
  rtf: number;
  /** 프레임당 물리 스텝 소요 시간 (ms) */
  physicsMsPerFrame: number;
  entityCount: number;
  colliderCount: number;
}

export interface StatsHudOptions {
  /** 초기 접힘 상태 (기본 true — 접힌 gauge 아이콘 버튼) */
  readonly collapsed?: boolean;
  /** 자체 rAF FPS 측정 (기본 true). 통합자가 fps를 주기 시작하면 자동으로 꺼진다 */
  readonly measureFps?: boolean;
  /** 접기/펼치기 통지 (레이아웃 지속 등) */
  onToggle?(collapsed: boolean): void;
}

export interface StatsHudHandle {
  /** HUD 루트 (host 기준 absolute 우상단) */
  readonly el: HTMLElement;
  /** 계측 갱신 — 준 필드만 반영한다 (매 프레임 호출 안전) */
  update(s: Partial<StatsSample>): void;
  setCollapsed(collapsed: boolean): void;
  readonly collapsed: boolean;
  dispose(): void;
}

// ── 순수 헬퍼 (DOM 비의존 — stats-hud.test.ts 대상) ──────────────────

function padValue(text: string, width: number): string {
  return text.padStart(width, ' ');
}

/**
 * real-time factor = simTime 진행 / 벽시계 경과.
 *
 * 벽시계 경과가 0 이하이거나 입력이 비유한이면 0을 돌려준다 — 0으로 나눠 Infinity가
 * 리드아웃에 새는 것을 막는다(첫 프레임/일시정지 직후에 실제로 발생한다).
 */
export function computeRtf(simAdvanceSec: number, wallElapsedSec: number): number {
  if (!Number.isFinite(simAdvanceSec) || !Number.isFinite(wallElapsedSec)) return 0;
  if (wallElapsedSec <= 0 || simAdvanceSec < 0) return 0;
  return simAdvanceSec / wallElapsedSec;
}

/** FPS 리드아웃 — 소수 1자리, 폭 5 고정 ('  9.8' / ' 59.8' / '120.0') */
export function formatFps(fps: number): string {
  if (!Number.isFinite(fps) || fps < 0) return padValue(DASH, FPS_FIELD_WIDTH);
  return padValue(fps.toFixed(FPS_DECIMALS), FPS_FIELD_WIDTH);
}

/** RTF 리드아웃 — 소수 2자리 + '×', 폭 5 고정 (' 1.00×' / ' 0.42×') */
export function formatRtf(rtf: number): string {
  if (!Number.isFinite(rtf) || rtf < 0) return `${padValue(DASH, RTF_FIELD_WIDTH)}×`;
  return `${padValue(rtf.toFixed(RTF_DECIMALS), RTF_FIELD_WIDTH)}×`;
}

/** 물리 스텝 리드아웃 — 소수 1자리 + 'ms/f', 폭 5 고정 ('  2.0ms/f') */
export function formatPhysicsMs(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return `${padValue(DASH, PHYSICS_MS_FIELD_WIDTH)}ms/f`;
  return `${padValue(ms.toFixed(PHYSICS_MS_DECIMALS), PHYSICS_MS_FIELD_WIDTH)}ms/f`;
}

/** 개수 리드아웃 — 정수, 폭 4 고정 ('  63') */
export function formatCount(n: number): string {
  if (!Number.isFinite(n) || n < 0) return padValue(DASH, COUNT_FIELD_WIDTH);
  return padValue(String(Math.round(n)), COUNT_FIELD_WIDTH);
}

/**
 * RTF 값 → 텍스트 색 토큰. 실시간이면 기본 텍스트, 처지면 warn, 심하게 처지면 충돌 램프.
 * (색은 보조 채널이다 — 숫자 자체가 1차 채널이므로 색맹 사용자도 손실이 없다.)
 */
export function rtfColor(rtf: number): string {
  if (!Number.isFinite(rtf) || rtf < 0) return COLOR.muted;
  if (rtf < RTF_CRITICAL_BELOW) return COLLISION.text;
  if (rtf < RTF_WARN_BELOW) return COLOR.warnText;
  return COLOR.text;
}

/**
 * 한 줄 요약 (접힘 상태 툴팁 · 로그용). 패딩을 제거한 조밀 표기다 —
 * 'FPS 59.8 · RTF 1.00× · 물리 2.0ms/f · 엔티티 63 · 콜라이더 71'
 */
export function formatStatsLine(s: StatsSample): string {
  return [
    `FPS ${formatFps(s.fps).trim()}`,
    `RTF ${formatRtf(s.rtf).trim()}`,
    `물리 ${formatPhysicsMs(s.physicsMsPerFrame).trim()}`,
    `엔티티 ${formatCount(s.entityCount).trim()}`,
    `콜라이더 ${formatCount(s.colliderCount).trim()}`,
  ].join(SEGMENT_SEP);
}

// ── 마운트 ──────────────────────────────────────────────────────────

interface StatRow {
  readonly el: HTMLElement;
  readonly valueEl: HTMLElement;
}

function makeRow(labelText: string, testId: string): StatRow {
  const el = styled(document.createElement('div'), {
    display: 'flex',
    alignItems: 'baseline',
    justifyContent: 'space-between',
    gap: SPACE.lg,
  });
  const label = applyType(document.createElement('span'), TYPE.caption);
  label.style.color = COLOR.label;
  label.textContent = labelText;
  const valueEl = applyType(document.createElement('span'), TYPE.monoReadout);
  styled(valueEl, {
    color: COLOR.text,
    // 선행 패딩 공백이 접히지 않게 — 자릿수 폭 고정의 (b) 절반 (파일 헤더)
    whiteSpace: 'pre',
  });
  valueEl.dataset.testid = testId;
  el.appendChild(label);
  el.appendChild(valueEl);
  return { el, valueEl };
}

/**
 * 실행 계측 HUD를 host(positioned 뷰포트 슬롯)에 마운트한다.
 *
 * 접힘이 기본이다 — 시뮬레이터를 처음 보는 사람에게 숫자 5개를 들이밀지 않되,
 * "믿을 수 없다"고 느낀 순간 한 번의 클릭으로 근거에 도달하게 한다.
 */
export function mountStatsHud(host: HTMLElement, opts: StatsHudOptions = {}): StatsHudHandle {
  ensureThemeStyles();

  const root = styled(document.createElement('div'), {
    position: 'absolute',
    top: `${HUD_TOP_PX}px`,
    right: `${HUD_RIGHT_PX}px`,
    zIndex: Z_INDEX.bar,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'flex-end',
    gap: SPACE.xs,
    pointerEvents: 'auto',
    userSelect: 'none',
  });
  root.dataset.testid = 'stats-hud';
  // HUD 위 포인터/휠이 뷰포트 orbit으로 새지 않게 흡수 (패널 규약)
  for (const type of ['pointerdown', 'pointermove', 'pointerup', 'wheel', 'contextmenu']) {
    root.addEventListener(type, (e) => {
      e.stopPropagation();
    });
  }

  const toggleButton = makeIconButton(
    'gauge',
    '',
    '실행 계측 (FPS · RTF · 물리 스텝)',
    'stats-hud-toggle',
    'ghost',
    ICON.lg,
  );
  toggleButton.style.background = SURFACE.panel;
  toggleButton.style.borderColor = BORDER.default;
  root.appendChild(toggleButton);

  const panel = styled(document.createElement('div'), {
    display: 'flex',
    flexDirection: 'column',
    gap: SPACE.xs,
    minWidth: '168px',
    padding: `${SPACE.md} ${SPACE.lg}`,
    background: SURFACE.overlay,
    border: `${BORDER_WIDTH.hair} solid ${BORDER.default}`,
    borderRadius: RADIUS.md,
    boxShadow: SHADOW.overlay,
    transition: tr('opacity', MOTION.fast),
  });
  panel.dataset.testid = 'stats-hud-panel';

  const heading = applyType(document.createElement('div'), TYPE.micro);
  styled(heading, {
    display: 'flex',
    alignItems: 'center',
    gap: SPACE.sm,
    color: COLOR.label,
    textTransform: 'uppercase',
    letterSpacing: '0.06em',
    paddingBottom: SPACE.xs,
    borderBottom: `${BORDER_WIDTH.hair} solid ${BORDER.subtle}`,
    marginBottom: SPACE.xs,
  });
  const headingIcon = icon('gauge', ICON.sm);
  const headingText = document.createElement('span');
  headingText.textContent = '실행 계측';
  heading.appendChild(headingIcon);
  heading.appendChild(headingText);
  panel.appendChild(heading);

  const fpsRow = makeRow('FPS', 'stats-hud-fps');
  const rtfRow = makeRow('RTF', 'stats-hud-rtf');
  const physicsRow = makeRow('물리', 'stats-hud-physics');
  const entityRow = makeRow('엔티티', 'stats-hud-entities');
  const colliderRow = makeRow('콜라이더', 'stats-hud-colliders');
  for (const row of [fpsRow, rtfRow, physicsRow, entityRow, colliderRow]) {
    panel.appendChild(row.el);
  }
  root.appendChild(panel);
  host.appendChild(root);

  // ── 상태 ──────────────────────────────────────────────────────────

  const sample: StatsSample = {
    fps: Number.NaN,
    rtf: Number.NaN,
    physicsMsPerFrame: Number.NaN,
    entityCount: Number.NaN,
    colliderCount: Number.NaN,
  };

  let collapsed = opts.collapsed ?? true;

  const paint = (): void => {
    fpsRow.valueEl.textContent = formatFps(sample.fps);
    rtfRow.valueEl.textContent = formatRtf(sample.rtf);
    rtfRow.valueEl.style.color = rtfColor(sample.rtf);
    physicsRow.valueEl.textContent = formatPhysicsMs(sample.physicsMsPerFrame);
    entityRow.valueEl.textContent = formatCount(sample.entityCount);
    colliderRow.valueEl.textContent = formatCount(sample.colliderCount);
    // 접힘 상태에서도 호버로 근거에 닿게 한다. aria-label은 갱신하지 않는다 —
    // 포커스된 버튼의 접근 가능한 이름이 매 프레임 바뀌면 스크린리더가 범람한다.
    toggleButton.title = `실행 계측 — ${formatStatsLine(sample)}`;
  };

  const paintCollapsed = (): void => {
    panel.style.display = collapsed ? 'none' : 'flex';
    toggleButton.setAttribute('aria-expanded', String(!collapsed));
    toggleButton.classList.toggle('ui-btn--active', !collapsed);
  };

  toggleButton.addEventListener('click', () => {
    collapsed = !collapsed;
    paintCollapsed();
    opts.onToggle?.(collapsed);
  });

  // ── 자체 rAF FPS 측정 (통합자가 fps를 주면 중단) ───────────────────

  let selfMeasureFps = opts.measureFps ?? true;
  let rafId: number | null = null;
  let frameCount = 0;
  let windowStartMs = 0;

  const stopSelfMeasure = (): void => {
    selfMeasureFps = false;
    if (rafId !== null && typeof cancelAnimationFrame === 'function') cancelAnimationFrame(rafId);
    rafId = null;
  };

  const onFrame = (nowMs: number): void => {
    if (!selfMeasureFps) return;
    if (windowStartMs === 0) windowStartMs = nowMs;
    frameCount += 1;
    const elapsedMs = nowMs - windowStartMs;
    if (elapsedMs >= FPS_SAMPLE_WINDOW_MS) {
      sample.fps = (frameCount * 1000) / elapsedMs;
      frameCount = 0;
      windowStartMs = nowMs;
      paint();
    }
    rafId = requestAnimationFrame(onFrame);
  };

  if (selfMeasureFps && typeof requestAnimationFrame === 'function') {
    rafId = requestAnimationFrame(onFrame);
  }

  paint();
  paintCollapsed();

  return {
    el: root,
    update: (s: Partial<StatsSample>): void => {
      if (s.fps !== undefined) {
        // 통합자가 진실을 갖고 있다 — 자체 측정을 끄고 이중 진실을 제거한다
        if (selfMeasureFps) stopSelfMeasure();
        sample.fps = s.fps;
      }
      if (s.rtf !== undefined) sample.rtf = s.rtf;
      if (s.physicsMsPerFrame !== undefined) sample.physicsMsPerFrame = s.physicsMsPerFrame;
      if (s.entityCount !== undefined) sample.entityCount = s.entityCount;
      if (s.colliderCount !== undefined) sample.colliderCount = s.colliderCount;
      paint();
    },
    setCollapsed: (c: boolean): void => {
      if (c === collapsed) return;
      collapsed = c;
      paintCollapsed();
      opts.onToggle?.(collapsed);
    },
    get collapsed(): boolean {
      return collapsed;
    },
    dispose: (): void => {
      stopSelfMeasure();
      root.remove();
    },
  };
}
