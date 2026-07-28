// ui/workspace.ts — 워크스페이스 셸 (UX_DESIGN §2/§8, docs/UX_AUDIT.md C-1/C-8/C-9/C-12)
//
// CSS grid 레이아웃으로 슬롯(placement)만 소유한다 — 패널 내용물은 통합자(main.ts)가
// 각 슬롯에 마운트한다 (계층 규칙 CLAUDE.md §3: 이 모듈은 core/render를 모른다).
//
//   ┌───────────────────────────────────────────────┐
//   │ <header> commandBar (row 1, 전체 폭)           │
//   ├──────┬─┬───────────────────────────┬─┬────────┤
//   │<nav> │║│ <main> viewport           │║│<aside> │   ║ = 세로 스플리터
//   │ left │║│        ═══ fg 스플리터     │║│ right  │
//   │      │║│        <section> flowGraph│║│ stack  │
//   ├──────┴─┴───────────────────────────┴─┴────────┤
//   │ ═══ dock 스플리터                              │
//   │ <section> dock (row 4, 전체 폭)                │
//   └───────────────────────────────────────────────┘
//
// ── 공간 예산: 상수가 아니라 정책이다 (C-1) ────────────────────────
//
// 구 구현은 세로 크롬(바 44 + 독 211 + 그래프 241 + 스플리터 10 = **506px 고정**)과
// 가로 크롬(좌 240 + 우 280 + 스플리터 10 = **530px 고정**)이 창 크기와 무관한 상수라,
// 뷰포트 면적이 `(W−530)×(H−506)`이라는 결정론적 함수가 됐다. 창이 작아질수록
// **주인공(3D 뷰포트)만** 손해를 본다:
//   1920→38.5% / 1600→29.3% / 1440→27.7% / 1366→**20.9%** / 1280→**17.4%**
// 1366×768에서 뷰포트는 836×262(3.19:1 레터박스)로, 로봇의 세로 작업 공간을 관찰할 수 없다.
//
// 이제 규칙은 이렇다: **크롬이 먼저 양보하고 뷰포트가 마지막에 양보한다.**
//   - 세로 크롬은 `clamp()` CSS로 창 높이에 비례하되 상하한이 고정된다.
//   - 중앙은 `minmax(VIEWPORT_MIN, 1fr)`로 뷰포트 하한을 보장한다.
//   - 독은 **기본 접힘**이고, 접으면 그리드 슬롯 자체가 줄어 뷰포트가 픽셀을 돌려받는다.
//   - 브레이크포인트가 좌/우 패널을 자동으로 레일로 접는다.
//
// ── 실행 모드는 레이아웃의 1급 상태다 (C-9) ────────────────────────
// `FlowMode: 'full' | 'strip' | 'off'`. ▶Play가 그래프를 56px 노드 스트립으로 접고
// ⏹Stop이 복원한다 — UX_DESIGN §2 "실행 중 뷰포트 우선 확장, 그래프는 최소 높이 유지"와
// §8 "실행 중에는 Viewport+활성 노드 스트립을 유지"의 문자 그대로의 구현이다.
// §1의 동시 가시성 원칙을 버리지 않으면서 관찰이 필요한 순간에 뷰포트가 2배가 된다.

import {
  BORDER,
  BREAKPOINT,
  COLOR,
  FONT,
  LAYOUT,
  MOTION,
  SPACE,
  SURFACE,
  TYPE,
  ensureThemeStyles,
  makeButton,
  styled,
  tr,
} from './theme';
import { setButtonContent } from './icons';
import { STORAGE_PREFIX } from './brand';
import { SCOPE_ATTR } from './shortcuts';

// ── 크기 상수 (px — 매직넘버 금지, CLAUDE.md §4) ────────────────────

export interface PanelSizeLimits {
  readonly defaultPx: number;
  readonly minPx: number;
  readonly maxPx: number;
}

/**
 * 패널별 기본/최소/최대 크기 — 스플리터 클램프의 단일 진실 (테스트 대상).
 *
 * 하한은 "실사용 하한"으로 재산정됐다(C-1). 구 하한 합(374px 세로 / 261px 가로)은
 * 스플리터를 최대한 끌어도 1366×768에서 뷰포트가 **41.5%가 상한**이 되게 만들었다 —
 * 기본값 문제가 아니라 하한값 정책 문제였다.
 */
export const WORKSPACE_SIZES: Readonly<
  Record<'left' | 'right' | 'dock' | 'flowGraph', PanelSizeLimits>
> = {
  /** 좌 라이브러리 패널 폭 (UX §2 "~240px") */
  left: { defaultPx: 240, minPx: 180, maxPx: 420 },
  /** 우 인스펙터 스택 폭 (UX §2 "~280px" — theme LAYOUT.rightPanelWidthPx와 일치) */
  right: { defaultPx: LAYOUT.rightPanelWidthPx, minPx: 200, maxPx: 460 },
  /** 하단 독 높이 (펼침 상태). 접힘은 별도 상태(DOCK_COLLAPSED_PX) */
  dock: { defaultPx: 210, minPx: 112, maxPx: 440 },
  /**
   * flowGraph 페인 높이 (full 모드). 하한 148px은 `LAYOUT.flowHeightCss`의 clamp 하한과
   * 일치한다. 구 하한 200px은 "그래프가 항상 보여야 한다"는 Phase 7 요구를 지키려던 것인데,
   * 그 요구는 이제 **strip 모드(56px)** 가 더 싸게 만족시킨다.
   */
  flowGraph: { defaultPx: 240, minPx: 148, maxPx: 560 },
};

/** 접힌 좌/우 패널 아이콘 레일 폭 */
export const LEFT_RAIL_WIDTH_PX = 36;
export const RIGHT_RAIL_WIDTH_PX = 36;

/** 독 접힘 높이 (탭바 + 인라인 스트립만 — 정보 손실 0) */
export const DOCK_COLLAPSED_PX = LAYOUT.dockCollapsedPx;

/** flowGraph strip 모드 높이 (실행 중 — 활성 노드 스트립만) */
export const FLOW_STRIP_PX = LAYOUT.flowStripPx;

/** 뷰포트 절대 하한 — 어떤 크롬도 이 아래로 뷰포트를 먹지 못한다 */
export const VIEWPORT_MIN_HEIGHT_PX = LAYOUT.viewportMinHeightPx;

/** 스플리터 두께 / 키보드 조절 스텝 */
const SPLITTER_THICKNESS_PX = 5;
const SPLITTER_KEY_STEP_PX = 16;

/** 레이아웃 영속 키 */
const LAYOUT_STORAGE_KEY = `${STORAGE_PREFIX}.layout`;

// ── 순수 헬퍼 (DOM 비의존 — workspace.test.ts 대상) ─────────────────

/** 스플리터 클램프: min/max 안으로 강제 (NaN은 min으로 방어) */
export function clampPanelSize(px: number, limits: PanelSizeLimits): number {
  if (Number.isNaN(px)) return limits.minPx;
  return Math.min(Math.max(px, limits.minPx), limits.maxPx);
}

/**
 * 드래그 결과 크기: 드래그 시작 시점 크기 + (포인터 이동량 × 방향 부호)를 클램프.
 * sign=+1: 포인터가 +방향(오른쪽/아래)으로 갈수록 커진다. sign=-1: 반대(우패널/독).
 */
export function dragPanelSize(
  startPx: number,
  deltaPx: number,
  sign: 1 | -1,
  limits: PanelSizeLimits,
): number {
  return clampPanelSize(startPx + deltaPx * sign, limits);
}

/** flowGraph 페인의 3-상태 — 실행 상태에 반응하는 레이아웃의 축 */
export type FlowMode = 'full' | 'strip' | 'off';

/** flowMode → 페인 높이 CSS 값 */
export function flowModeHeightCss(mode: FlowMode, fullPx: number): string {
  if (mode === 'off') return '0px';
  if (mode === 'strip') return `${FLOW_STRIP_PX}px`;
  return `${fullPx}px`;
}

/**
 * 창 크기에 대한 자동 접힘 판정 (순수 — 테스트 대상).
 *
 * 반환값은 "자동 규칙이 원하는 상태"다. 사용자가 명시적으로 토글했다면 그 세션 동안은
 * 사용자 선택이 이긴다(mountWorkspace가 override 플래그로 처리).
 */
export interface AutoLayoutDecision {
  readonly collapseLeft: boolean;
  readonly collapseRight: boolean;
  readonly collapseDock: boolean;
  readonly compactBar: boolean;
}

export function decideAutoLayout(widthPx: number, heightPx: number): AutoLayoutDecision {
  return {
    collapseLeft: widthPx < BREAKPOINT.autoCollapseLeftPx,
    collapseRight: widthPx < BREAKPOINT.autoCollapseRightPx,
    collapseDock: heightPx < BREAKPOINT.autoCollapseDockPx,
    compactBar: widthPx < BREAKPOINT.compactBarPx,
  };
}

// ── 영속 상태 ───────────────────────────────────────────────────────

export interface PersistedLayout {
  leftPx: number;
  rightPx: number;
  dockPx: number;
  flowPx: number;
  leftCollapsed: boolean;
  rightCollapsed: boolean;
  dockCollapsed: boolean;
  flowMode: FlowMode;
}

function readPersistedLayout(): Partial<PersistedLayout> {
  try {
    const raw = localStorage.getItem(LAYOUT_STORAGE_KEY);
    if (raw === null) return {};
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed !== 'object' || parsed === null) return {};
    return parsed as Partial<PersistedLayout>;
  } catch {
    // 스키마 불일치·파싱 실패·저장소 차단은 조용히 기본값으로 (loadPlannerConfig와 같은 규약)
    return {};
  }
}

function writePersistedLayout(state: PersistedLayout): void {
  try {
    localStorage.setItem(LAYOUT_STORAGE_KEY, JSON.stringify(state));
  } catch {
    // 사파리 프라이빗 모드 등 — 영속 실패가 앱을 막으면 안 된다
  }
}

// ── 공개 타입 ───────────────────────────────────────────────────────

/** 통합자가 각 패널을 마운트하는 슬롯 (셸이 배치 소유, 내용물은 통합자 소유) */
export interface WorkspaceSlots {
  /** 상단 커맨드바 — <header role="banner"> */
  readonly commandBar: HTMLElement;
  /** 좌 라이브러리 패널 — <nav aria-label="라이브러리"> 안 */
  readonly left: HTMLElement;
  /** 중앙 3D 뷰포트 — 렌더러 캔버스 호스트를 reparent */
  readonly viewport: HTMLElement;
  /** 우 패널 스택 — <aside aria-label="인스펙터"> 안 */
  readonly rightStack: HTMLElement;
  /** 하단 독 — <section aria-label="독"> */
  readonly dock: HTMLElement;
  /** 중앙 하단 플로우 그래프 페인 — <section aria-label="플로우 그래프"> */
  readonly flowGraph: HTMLElement;
  /** 좌 레일(접힘 시 보이는 아이콘 열) — 통합자가 아이콘 버튼을 넣을 수 있다 */
  readonly leftRail: HTMLElement;
  /** 우 레일 */
  readonly rightRail: HTMLElement;
}

export interface WorkspaceHandle {
  /** 셸 루트 (host의 자식) */
  readonly el: HTMLElement;
  readonly slots: WorkspaceSlots;
  /** 좌 패널 접기/펼치기 — 접으면 36px 아이콘 레일만 남는다 */
  setLeftCollapsed(collapsed: boolean): void;
  /** 우 패널 접기/펼치기 (UX §2 "좌/**우** 패널은 접기 가능" — 구현이 없던 축) */
  setRightCollapsed(collapsed: boolean): void;
  /** 독 접기/펼치기 — 그리드 슬롯 자체가 줄어 뷰포트가 픽셀을 돌려받는다 */
  setDockCollapsed(collapsed: boolean): void;
  isDockCollapsed(): boolean;
  /** flowGraph 3-상태 */
  setFlowMode(mode: FlowMode): void;
  getFlowMode(): FlowMode;
  /** @deprecated setFlowMode를 쓸 것. visible=true → 'full', false → 'off' */
  setFlowGraphVisible(visible: boolean): void;
  /** 레이아웃 프리셋 적용 */
  applyPreset(preset: LayoutPresetName): void;
  /** 레이아웃 변경을 렌더러에 전파 — window 'resize' 이벤트 합성 발행 */
  notifyResize(): void;
  dispose(): void;
}

/** 워크플로 단계별 레이아웃 프리셋 (Blender Workspaces / Figma 관행) */
export type LayoutPresetName = 'default' | 'sceneBuild' | 'flowEdit' | 'runObserve';

const LAYOUT_PRESETS: Readonly<Record<LayoutPresetName, Partial<PersistedLayout>>> = {
  default: {
    leftCollapsed: false,
    rightCollapsed: false,
    dockCollapsed: true,
    flowMode: 'full',
  },
  sceneBuild: {
    leftPx: 300,
    rightPx: 320,
    leftCollapsed: false,
    rightCollapsed: false,
    dockCollapsed: true,
    flowMode: 'off',
  },
  flowEdit: {
    leftCollapsed: true,
    rightCollapsed: false,
    dockCollapsed: true,
    flowMode: 'full',
    flowPx: 380,
  },
  runObserve: {
    leftCollapsed: true,
    rightCollapsed: true,
    dockCollapsed: false,
    flowMode: 'strip',
  },
};

// ── 셸 전용 스타일 (스플리터 hover/드래그/포커스 — 토큰만 소비) ─────

const WORKSPACE_STYLE_ID = 'rsw-workspace-styles';

function ensureWorkspaceStyles(): void {
  if (document.getElementById(WORKSPACE_STYLE_ID) !== null) return;
  const style = document.createElement('style');
  style.id = WORKSPACE_STYLE_ID;
  style.textContent = `
.rsw-split {
  flex-shrink: 0;
  background: ${BORDER.subtle};
  transition: ${tr('background-color', MOTION.instant)};
  touch-action: none;
  user-select: none;
  position: relative;
}
/* 시각 두께는 5px 유지, 히트 영역만 넓힌다 (WCAG 2.2 SC 2.5.8 — Blender/Figma 관행).
   구 구현은 5px 표적을 정확히 물어야 했고, 실패하면 인접 패널을 클릭해 선택이 바뀌었다. */
.rsw-split--v { cursor: col-resize; width: ${SPLITTER_THICKNESS_PX}px; }
.rsw-split--v::before { content: ''; position: absolute; inset: 0 -5px; }
.rsw-split--h { cursor: row-resize; height: ${SPLITTER_THICKNESS_PX}px; }
.rsw-split--h::before { content: ''; position: absolute; inset: -5px 0; }
.rsw-split:hover, .rsw-split--drag { background: var(--rsw-accent); }
.rsw-split:focus-visible { outline: 2px solid var(--rsw-accent); outline-offset: -2px; }

/* 접힘 레일 */
.rsw-rail {
  display: none;
  flex-direction: column;
  align-items: center;
  gap: ${SPACE.xs};
  padding: ${SPACE.sm} 0;
  flex: 1 1 auto;
  min-height: 0;
}
.rsw-rail--visible { display: flex; }
.rsw-rail-label {
  writing-mode: vertical-rl;
  padding: ${SPACE.md} 0;
  letter-spacing: 0.08em;
  color: ${COLOR.muted};
  background: transparent;
  border: none;
  cursor: pointer;
  font: ${TYPE.caption.weight} ${TYPE.caption.sizePx}px/${TYPE.caption.lineHeightPx}px ${FONT.ui};
  transition: ${tr('color', MOTION.instant)};
}
.rsw-rail-label:hover { color: ${COLOR.textStrong}; }
.rsw-rail-label:focus-visible { outline: 2px solid var(--rsw-accent); outline-offset: 2px; }
`;
  document.head.appendChild(style);
}

// ── 스플리터 팩토리 ─────────────────────────────────────────────────

interface SplitterOptions {
  readonly orientation: 'vertical' | 'horizontal';
  /** 접근성 라벨 (role=separator) */
  readonly labelKo: string;
  readonly testId: string;
  /** 드래그 방향 부호 (dragPanelSize 참조) */
  readonly sign: 1 | -1;
  readonly limits: PanelSizeLimits;
  getSize(): number;
  setSize(px: number): void;
  /** 크기 반영 후 통지 (rAF 스로틀된 notifyResize) */
  onResized(): void;
}

function makeSplitter(opts: SplitterOptions): HTMLElement {
  const splitter = document.createElement('div');
  splitter.className =
    opts.orientation === 'vertical' ? 'rsw-split rsw-split--v' : 'rsw-split rsw-split--h';
  splitter.dataset.testid = opts.testId;
  splitter.setAttribute('role', 'separator');
  splitter.setAttribute('aria-label', opts.labelKo);
  splitter.setAttribute(
    'aria-orientation',
    opts.orientation === 'vertical' ? 'vertical' : 'horizontal',
  );
  splitter.setAttribute('aria-valuemin', String(opts.limits.minPx));
  splitter.setAttribute('aria-valuemax', String(opts.limits.maxPx));
  splitter.tabIndex = 0;

  const syncValueNow = (): void => {
    splitter.setAttribute('aria-valuenow', String(Math.round(opts.getSize())));
  };
  syncValueNow();

  let dragStartPos = 0;
  let dragStartSize = 0;
  let dragging = false;

  splitter.addEventListener('pointerdown', (e) => {
    if (e.button !== 0) return;
    dragging = true;
    dragStartPos = opts.orientation === 'vertical' ? e.clientX : e.clientY;
    dragStartSize = opts.getSize();
    splitter.classList.add('rsw-split--drag');
    splitter.setPointerCapture(e.pointerId);
    e.preventDefault();
  });
  splitter.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    const pos = opts.orientation === 'vertical' ? e.clientX : e.clientY;
    opts.setSize(dragPanelSize(dragStartSize, pos - dragStartPos, opts.sign, opts.limits));
    syncValueNow();
    opts.onResized();
  });
  const endDrag = (e: PointerEvent): void => {
    if (!dragging) return;
    dragging = false;
    splitter.classList.remove('rsw-split--drag');
    if (splitter.hasPointerCapture(e.pointerId)) splitter.releasePointerCapture(e.pointerId);
    opts.onResized();
  };
  splitter.addEventListener('pointerup', endDrag);
  splitter.addEventListener('pointercancel', endDrag);

  // 키보드 조절 (UX §9): 세로 스플리터 ←/→, 가로 스플리터 ↑/↓ — sign과 동일 규약.
  // stopPropagation이 중요하다: 같은 방향키를 뷰포트 nudge가 window에서 듣고 있어,
  // 막지 않으면 패널 폭 조절과 3D 오브젝트 이동이 동시에 일어난다 (C-6).
  splitter.addEventListener('keydown', (e) => {
    const growKey = opts.orientation === 'vertical' ? 'ArrowRight' : 'ArrowDown';
    const shrinkKey = opts.orientation === 'vertical' ? 'ArrowLeft' : 'ArrowUp';
    if (e.key !== growKey && e.key !== shrinkKey) return;
    const deltaPx = e.key === growKey ? SPLITTER_KEY_STEP_PX : -SPLITTER_KEY_STEP_PX;
    opts.setSize(dragPanelSize(opts.getSize(), deltaPx, opts.sign, opts.limits));
    syncValueNow();
    opts.onResized();
    e.preventDefault();
    e.stopPropagation();
  });

  return splitter;
}

// ── 마운트 ──────────────────────────────────────────────────────────

/**
 * 워크스페이스 셸을 host에 마운트한다. host는 뷰포트를 채우는 요소여야 한다
 * (루트는 position:absolute inset:0 — host가 static이어도 뷰포트 기준으로 찬다).
 */
export function mountWorkspace(host: HTMLElement): WorkspaceHandle {
  ensureThemeStyles();
  ensureWorkspaceStyles();

  // ── 레이아웃 상태 (뷰 상태는 ui 소유) ─────────────────────────────
  const saved = readPersistedLayout();
  let leftWidthPx = clampPanelSize(saved.leftPx ?? WORKSPACE_SIZES.left.defaultPx, WORKSPACE_SIZES.left);
  let rightWidthPx = clampPanelSize(
    saved.rightPx ?? WORKSPACE_SIZES.right.defaultPx,
    WORKSPACE_SIZES.right,
  );
  let dockHeightPx = clampPanelSize(saved.dockPx ?? WORKSPACE_SIZES.dock.defaultPx, WORKSPACE_SIZES.dock);
  let flowHeightPx = clampPanelSize(
    saved.flowPx ?? WORKSPACE_SIZES.flowGraph.defaultPx,
    WORKSPACE_SIZES.flowGraph,
  );
  let leftCollapsed = saved.leftCollapsed ?? false;
  let rightCollapsed = saved.rightCollapsed ?? false;
  // 독 기본 접힘 (C-1): 콘텐츠 ~58px에 211px를 상시 내주던 낭비를 되돌린다.
  // 접혀 있어도 탭바 + 인라인 진행 스트립이 보이므로 정보 손실은 0이다.
  let dockCollapsed = saved.dockCollapsed ?? true;
  let flowMode: FlowMode = saved.flowMode ?? 'off';

  // 자동 접힘이 건드린 축인지 추적 — 사용자가 명시적으로 토글하면 그 축은 자동 규칙에서 해방된다
  let leftAutoCollapsed = false;
  let rightAutoCollapsed = false;
  let dockAutoCollapsed = false;

  const persist = (): void => {
    writePersistedLayout({
      leftPx: leftWidthPx,
      rightPx: rightWidthPx,
      dockPx: dockHeightPx,
      flowPx: flowHeightPx,
      leftCollapsed,
      rightCollapsed,
      dockCollapsed,
      flowMode,
    });
  };
  let persistTimer: number | null = null;
  const persistDebounced = (): void => {
    if (persistTimer !== null) window.clearTimeout(persistTimer);
    persistTimer = window.setTimeout(() => {
      persistTimer = null;
      persist();
    }, 300);
  };

  // ── 리사이즈 통지 (rAF 스로틀 — 드래그 중 WebGL setSize 폭주 방지) ─
  let resizeRafId: number | null = null;
  const notifyResize = (): void => {
    window.dispatchEvent(new Event('resize'));
  };
  const notifyResizeThrottled = (): void => {
    if (resizeRafId !== null) return;
    resizeRafId = requestAnimationFrame(() => {
      resizeRafId = null;
      notifyResize();
    });
  };

  // ── 루트 그리드 ───────────────────────────────────────────────────
  const root = styled(document.createElement('div'), {
    position: 'absolute',
    inset: '0',
    display: 'grid',
    gridTemplateRows: 'auto minmax(0, 1fr) auto auto',
    gridTemplateColumns: 'auto auto minmax(0, 1fr) auto auto',
    background: SURFACE.base,
    color: COLOR.text,
    fontFamily: FONT.ui,
    fontSize: `${TYPE.body.sizePx}px`,
    overflow: 'hidden',
    boxSizing: 'border-box',
  });
  root.dataset.testid = 'workspace';

  // 행 1: 커맨드바 슬롯 — <header role="banner">
  const commandBarSlot = styled(document.createElement('header'), {
    gridRow: '1',
    gridColumn: '1 / -1',
    minHeight: '0',
    position: 'relative', // .rsw-run-strip 앵커
  });
  commandBarSlot.setAttribute('role', 'banner');
  commandBarSlot.dataset.testid = 'workspace-command-bar';
  commandBarSlot.setAttribute(SCOPE_ATTR, 'commandBar');
  // 전역 실행 상태 스트립 — body[data-run-state="running"]일 때 나타난다 (C-9)
  const runStrip = document.createElement('div');
  runStrip.className = 'rsw-run-strip';
  runStrip.setAttribute('aria-hidden', 'true');
  commandBarSlot.appendChild(runStrip);

  // ── 행 2 열 1: 좌 패널 — <nav> ────────────────────────────────────
  const leftWrap = styled(document.createElement('nav'), {
    gridRow: '2',
    gridColumn: '1',
    display: 'flex',
    flexDirection: 'column',
    minWidth: '0',
    minHeight: '0',
    overflow: 'hidden',
    background: SURFACE.panel,
    borderRight: `1px solid ${BORDER.subtle}`,
    boxSizing: 'border-box',
    transition: tr('width', MOTION.base),
  });
  leftWrap.setAttribute('aria-label', '라이브러리');
  leftWrap.dataset.testid = 'workspace-left-wrap';
  leftWrap.setAttribute(SCOPE_ATTR, 'inspector');

  const leftToggleRow = styled(document.createElement('div'), {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'flex-end',
    flexShrink: '0',
    padding: SPACE.xxs,
    borderBottom: `1px solid ${BORDER.subtle}`,
  });
  const leftToggle = makeButton('', '라이브러리 패널 접기', 'workspace-left-toggle', 'ghost');
  setButtonContent(leftToggle, 'chevronsLeft', '');
  styled(leftToggle, { padding: SPACE.xxs, minWidth: '24px', justifyContent: 'center' });
  leftToggleRow.appendChild(leftToggle);

  const leftRail = document.createElement('div');
  leftRail.className = 'rsw-rail';
  leftRail.dataset.testid = 'workspace-left-rail';
  const leftRailLabel = document.createElement('button');
  leftRailLabel.type = 'button';
  leftRailLabel.className = 'rsw-rail-label';
  leftRailLabel.textContent = '라이브러리';
  leftRailLabel.title = '라이브러리 패널 펼치기';
  leftRailLabel.setAttribute('aria-label', '라이브러리 패널 펼치기');
  leftRailLabel.dataset.testid = 'workspace-left-rail-label';
  leftRail.appendChild(leftRailLabel);

  // 좌 슬롯은 **세로 스택**이다: 라이브러리(위) + 씬 아웃라이너(아래).
  // 프로 3D 툴은 예외 없이 아웃라이너와 프로퍼티를 분리한다(Blender Outliner ≠ Properties,
  // Unity Hierarchy 좌 / Inspector 우). 목록은 *항상 보여야 하는 내비게이션 표면*이고
  // 속성은 *선택마다 통째로 바뀌며 세로를 독점하는 작업 표면*이라, 한 컨테이너에 두면
  // 선택을 바꾸려면 스크롤부터 해야 한다 (UX_AUDIT C-16).
  const leftSlot = styled(document.createElement('div'), {
    flex: '1 1 auto',
    minHeight: '0',
    position: 'relative',
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
  });
  leftSlot.dataset.testid = 'workspace-left';

  leftWrap.append(leftToggleRow, leftRail, leftSlot);

  // ── 행 2 열 3: 중앙 — <main> (뷰포트 상 / flowGraph 하) ───────────
  //
  // flex → grid로 바꾼 것이 C-1의 핵심이다. `minmax(VIEWPORT_MIN, 1fr)`가 뷰포트 하한을
  // 그리드 차원에서 보장하므로, 그래프/독이 아무리 커져도 뷰포트가 0으로 눌리지 않는다.
  const centerWrap = styled(document.createElement('main'), {
    gridRow: '2',
    gridColumn: '3',
    display: 'grid',
    gridTemplateRows: `minmax(${VIEWPORT_MIN_HEIGHT_PX}px, 1fr) auto var(--rsw-flow-h, 0px)`,
    minWidth: '0',
    minHeight: '0',
    overflow: 'hidden',
  });
  centerWrap.id = 'workcell-main'; // index.html의 스킵 링크 대상
  centerWrap.setAttribute('aria-label', '워크스페이스');
  centerWrap.dataset.testid = 'workspace-center';

  const viewportSlot = styled(document.createElement('div'), {
    gridRow: '1',
    minHeight: '0',
    position: 'relative',
    overflow: 'hidden',
  });
  viewportSlot.setAttribute('role', 'region');
  viewportSlot.setAttribute('aria-label', '3D 뷰포트');
  viewportSlot.dataset.testid = 'workspace-viewport';
  viewportSlot.setAttribute(SCOPE_ATTR, 'viewport');

  const flowGraphPane = styled(document.createElement('section'), {
    gridRow: '3',
    minHeight: '0',
    position: 'relative',
    overflow: 'hidden',
    background: SURFACE.panel,
    borderTop: `1px solid ${BORDER.default}`,
    transition: tr('height', MOTION.base),
  });
  flowGraphPane.setAttribute('aria-label', '플로우 그래프');
  flowGraphPane.dataset.testid = 'workspace-flow-graph';
  flowGraphPane.setAttribute(SCOPE_ATTR, 'graph');

  const flowGraphSplitter = makeSplitter({
    orientation: 'horizontal',
    labelKo: '뷰포트/플로우 그래프 분할 크기 조절',
    testId: 'workspace-split-flow-graph',
    sign: -1, // 위로 끌수록 flowGraph가 커진다
    limits: WORKSPACE_SIZES.flowGraph,
    getSize: () => flowHeightPx,
    setSize: (px) => {
      flowHeightPx = px;
      if (flowMode === 'full') root.style.setProperty('--rsw-flow-h', `${px}px`);
      persistDebounced();
    },
    onResized: notifyResizeThrottled,
  });
  styled(flowGraphSplitter, { gridRow: '2' });

  centerWrap.append(viewportSlot, flowGraphSplitter, flowGraphPane);

  // ── 행 2 열 5: 우 패널 — <aside> ──────────────────────────────────
  // width를 처음부터 확정한다: 열 5는 auto 트랙이라 폭을 비워 두면 스택 안 **내용의
  // max-content**가 열 폭이 된다 — 긴 안내 문구 한 줄이나 넓은 폼이 들어오는 순간
  // 패널이 UX §2의 "~280px"을 넘어 뷰포트를 잠식하고(실측: 로봇 선택 시 247 → 750 px),
  // 그리드는 좁아졌는데 캔버스는 resize 통지를 못 받아 패널 아래로 밀려 들어가
  // 그 띠에서 드롭·클릭이 패널에 먹힌다. 폭 변경은 스플리터/접기만 한다.
  const rightWrap = styled(document.createElement('aside'), {
    gridRow: '2',
    gridColumn: '5',
    display: 'flex',
    flexDirection: 'column',
    width: `${rightWidthPx}px`,
    minWidth: '0',
    minHeight: '0',
    overflow: 'hidden',
    background: SURFACE.panel,
    borderLeft: `1px solid ${BORDER.subtle}`,
    boxSizing: 'border-box',
    transition: tr('width', MOTION.base),
  });
  rightWrap.setAttribute('aria-label', '인스펙터');
  rightWrap.dataset.testid = 'workspace-right-wrap';
  rightWrap.setAttribute(SCOPE_ATTR, 'inspector');

  const rightToggleRow = styled(document.createElement('div'), {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'flex-start',
    flexShrink: '0',
    padding: SPACE.xxs,
    borderBottom: `1px solid ${BORDER.subtle}`,
  });
  const rightToggle = makeButton('', '인스펙터 패널 접기', 'workspace-right-toggle', 'ghost');
  setButtonContent(rightToggle, 'chevronsRight', '');
  styled(rightToggle, { padding: SPACE.xxs, minWidth: '24px', justifyContent: 'center' });
  rightToggleRow.appendChild(rightToggle);

  const rightRail = document.createElement('div');
  rightRail.className = 'rsw-rail';
  rightRail.dataset.testid = 'workspace-right-rail';
  const rightRailLabel = document.createElement('button');
  rightRailLabel.type = 'button';
  rightRailLabel.className = 'rsw-rail-label';
  rightRailLabel.textContent = '인스펙터';
  rightRailLabel.title = '인스펙터 패널 펼치기';
  rightRailLabel.setAttribute('aria-label', '인스펙터 패널 펼치기');
  rightRailLabel.dataset.testid = 'workspace-right-rail-label';
  rightRail.appendChild(rightRailLabel);

  const rightSlot = styled(document.createElement('div'), {
    flex: '1 1 auto',
    minHeight: '0',
    position: 'relative',
    overflowY: 'auto',
  });
  rightSlot.classList.add('ui-scroll');
  rightSlot.dataset.testid = 'workspace-right-stack';

  rightWrap.append(rightToggleRow, rightRail, rightSlot);

  // ── 세로 스플리터 (행 2 열 2/열 4) ────────────────────────────────
  const leftSplitter = makeSplitter({
    orientation: 'vertical',
    labelKo: '좌 패널 폭 조절',
    testId: 'workspace-split-left',
    sign: 1,
    limits: WORKSPACE_SIZES.left,
    getSize: () => leftWidthPx,
    setSize: (px) => {
      leftWidthPx = px;
      if (!leftCollapsed) leftWrap.style.width = `${px}px`;
      persistDebounced();
    },
    onResized: notifyResizeThrottled,
  });
  styled(leftSplitter, { gridRow: '2', gridColumn: '2' });

  const rightSplitter = makeSplitter({
    orientation: 'vertical',
    labelKo: '우 패널 폭 조절',
    testId: 'workspace-split-right',
    sign: -1, // 왼쪽으로 끌수록 우 패널이 커진다
    limits: WORKSPACE_SIZES.right,
    getSize: () => rightWidthPx,
    setSize: (px) => {
      rightWidthPx = px;
      if (!rightCollapsed) rightWrap.style.width = `${px}px`;
      persistDebounced();
    },
    onResized: notifyResizeThrottled,
  });
  styled(rightSplitter, { gridRow: '2', gridColumn: '4' });

  // ── 행 3/4: 독 스플리터 + 독 슬롯 — <section> ─────────────────────
  const dockSlot = styled(document.createElement('section'), {
    gridRow: '4',
    gridColumn: '1 / -1',
    height: `var(--rsw-dock-h, ${LAYOUT.dockHeightCss})`,
    minHeight: '0',
    position: 'relative',
    overflow: 'hidden',
    background: SURFACE.panel,
    borderTop: `1px solid ${BORDER.default}`,
    transition: tr('height', MOTION.base),
  });
  dockSlot.setAttribute('aria-label', '독 — 타임라인 · 충돌 로그 · 콘솔');
  dockSlot.dataset.testid = 'workspace-dock';
  dockSlot.setAttribute(SCOPE_ATTR, 'dock');

  const dockSplitter = makeSplitter({
    orientation: 'horizontal',
    labelKo: '독 높이 조절',
    testId: 'workspace-split-dock',
    sign: -1, // 위로 끌수록 독이 커진다
    limits: WORKSPACE_SIZES.dock,
    getSize: () => dockHeightPx,
    setSize: (px) => {
      dockHeightPx = px;
      if (!dockCollapsed) root.style.setProperty('--rsw-dock-h', `${px}px`);
      persistDebounced();
    },
    onResized: notifyResizeThrottled,
  });
  styled(dockSplitter, { gridRow: '3', gridColumn: '1 / -1' });

  // ── 조립 ──────────────────────────────────────────────────────────
  root.append(
    commandBarSlot,
    leftWrap,
    leftSplitter,
    centerWrap,
    rightSplitter,
    rightWrap,
    dockSplitter,
    dockSlot,
  );
  host.appendChild(root);

  // ── 페인 가시성 단일 헬퍼 ─────────────────────────────────────────
  //
  // 구 구현은 "숨김"의 의미가 컴포넌트마다 달랐다: flowGraph는 display:none으로 5+241px을
  // 온전히 회수했는데 독은 내부 body만 접혀 그리드 슬롯 211px이 그대로 남았다.
  // 이제 {슬롯 크기, 스플리터 display, notifyResize} 3종을 항상 함께 갱신한다.

  const paintLeft = (): void => {
    leftWrap.style.width = leftCollapsed ? `${LEFT_RAIL_WIDTH_PX}px` : `${leftWidthPx}px`;
    // 'flex'로 되돌린다 — ''는 CSS 기본값(block)이라 leftSlot의 세로 스택(라이브러리 +
    // 아웃라이너)이 무너지고 아웃라이너가 패널 밖으로 밀려 잘린다.
    leftSlot.style.display = leftCollapsed ? 'none' : 'flex';
    leftRail.classList.toggle('rsw-rail--visible', leftCollapsed);
    leftSplitter.style.display = leftCollapsed ? 'none' : '';
    leftToggleRow.style.justifyContent = leftCollapsed ? 'center' : 'flex-end';
    setButtonContent(leftToggle, leftCollapsed ? 'chevronsRight' : 'chevronsLeft', '');
    const title = leftCollapsed ? '라이브러리 패널 펼치기' : '라이브러리 패널 접기';
    leftToggle.title = title;
    leftToggle.setAttribute('aria-label', title);
    leftToggle.setAttribute('aria-expanded', String(!leftCollapsed));
  };

  const paintRight = (): void => {
    rightWrap.style.width = rightCollapsed ? `${RIGHT_RAIL_WIDTH_PX}px` : `${rightWidthPx}px`;
    rightSlot.style.display = rightCollapsed ? 'none' : '';
    rightRail.classList.toggle('rsw-rail--visible', rightCollapsed);
    rightSplitter.style.display = rightCollapsed ? 'none' : '';
    rightToggleRow.style.justifyContent = rightCollapsed ? 'center' : 'flex-start';
    setButtonContent(rightToggle, rightCollapsed ? 'chevronsLeft' : 'chevronsRight', '');
    const title = rightCollapsed ? '인스펙터 패널 펼치기' : '인스펙터 패널 접기';
    rightToggle.title = title;
    rightToggle.setAttribute('aria-label', title);
    rightToggle.setAttribute('aria-expanded', String(!rightCollapsed));
  };

  const paintDock = (): void => {
    root.style.setProperty(
      '--rsw-dock-h',
      dockCollapsed ? `${DOCK_COLLAPSED_PX}px` : `${dockHeightPx}px`,
    );
    dockSplitter.style.display = dockCollapsed ? 'none' : '';
  };

  const paintFlow = (): void => {
    root.style.setProperty('--rsw-flow-h', flowModeHeightCss(flowMode, flowHeightPx));
    flowGraphPane.style.display = flowMode === 'off' ? 'none' : '';
    flowGraphSplitter.style.display = flowMode === 'full' ? '' : 'none';
    flowGraphPane.dataset.flowMode = flowMode;
  };

  const setLeftCollapsed = (collapsed: boolean, fromAuto = false): void => {
    if (collapsed === leftCollapsed) return;
    leftCollapsed = collapsed;
    if (!fromAuto) leftAutoCollapsed = false;
    paintLeft();
    persistDebounced();
    notifyResize();
  };
  const setRightCollapsed = (collapsed: boolean, fromAuto = false): void => {
    if (collapsed === rightCollapsed) return;
    rightCollapsed = collapsed;
    if (!fromAuto) rightAutoCollapsed = false;
    paintRight();
    persistDebounced();
    notifyResize();
  };
  const setDockCollapsed = (collapsed: boolean, fromAuto = false): void => {
    if (collapsed === dockCollapsed) return;
    dockCollapsed = collapsed;
    if (!fromAuto) dockAutoCollapsed = false;
    paintDock();
    persistDebounced();
    notifyResize();
  };
  const setFlowMode = (mode: FlowMode): void => {
    if (mode === flowMode) return;
    flowMode = mode;
    paintFlow();
    persistDebounced();
    notifyResize();
  };

  leftToggle.addEventListener('click', () => {
    setLeftCollapsed(!leftCollapsed);
  });
  leftRailLabel.addEventListener('click', () => {
    setLeftCollapsed(false);
  });
  rightToggle.addEventListener('click', () => {
    setRightCollapsed(!rightCollapsed);
  });
  rightRailLabel.addEventListener('click', () => {
    setRightCollapsed(false);
  });

  paintLeft();
  paintRight();
  paintDock();
  paintFlow();

  // ── 반응형 (UX §8 — 구현이 0개이던 축) ────────────────────────────
  //
  // window.resize가 아니라 root를 관찰한다: 스플리터 드래그와 창 리사이즈를 같은 경로로
  // 처리하고, 임베드된 경우에도 동작한다. 자기 자신의 크기 변경으로 재진입하지 않도록
  // rAF로 한 프레임 미룬다.
  let autoRafId: number | null = null;
  const applyAutoLayout = (): void => {
    autoRafId = null;
    const rect = root.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;
    const want = decideAutoLayout(rect.width, rect.height);

    // 사용자가 직접 토글한 축은 건드리지 않는다. 자동으로 접은 축만 자동으로 되돌린다.
    if (want.collapseLeft && !leftCollapsed) {
      leftAutoCollapsed = true;
      setLeftCollapsed(true, true);
    } else if (!want.collapseLeft && leftCollapsed && leftAutoCollapsed) {
      leftAutoCollapsed = false;
      setLeftCollapsed(false, true);
    }

    if (want.collapseRight && !rightCollapsed) {
      rightAutoCollapsed = true;
      setRightCollapsed(true, true);
    } else if (!want.collapseRight && rightCollapsed && rightAutoCollapsed) {
      rightAutoCollapsed = false;
      setRightCollapsed(false, true);
    }

    if (want.collapseDock && !dockCollapsed) {
      dockAutoCollapsed = true;
      setDockCollapsed(true, true);
    } else if (!want.collapseDock && dockCollapsed && dockAutoCollapsed) {
      dockAutoCollapsed = false;
      setDockCollapsed(false, true);
    }

    // 좁은 화면에서 full 그래프는 뷰포트를 다시 압살한다 — strip으로 강등
    if (rect.height < BREAKPOINT.autoCollapseDockPx && flowMode === 'full') {
      setFlowMode('strip');
    }

    root.dataset.compactBar = String(want.compactBar);
  };
  const scheduleAutoLayout = (): void => {
    if (autoRafId !== null) return;
    autoRafId = requestAnimationFrame(applyAutoLayout);
  };

  const observer = new ResizeObserver(scheduleAutoLayout);
  observer.observe(root);
  scheduleAutoLayout();

  const applyPreset = (preset: LayoutPresetName): void => {
    const p = LAYOUT_PRESETS[preset];
    if (p.leftPx !== undefined) {
      leftWidthPx = clampPanelSize(p.leftPx, WORKSPACE_SIZES.left);
    }
    if (p.rightPx !== undefined) {
      rightWidthPx = clampPanelSize(p.rightPx, WORKSPACE_SIZES.right);
    }
    if (p.flowPx !== undefined) {
      flowHeightPx = clampPanelSize(p.flowPx, WORKSPACE_SIZES.flowGraph);
    }
    if (p.leftCollapsed !== undefined) leftCollapsed = p.leftCollapsed;
    if (p.rightCollapsed !== undefined) rightCollapsed = p.rightCollapsed;
    if (p.dockCollapsed !== undefined) dockCollapsed = p.dockCollapsed;
    if (p.flowMode !== undefined) flowMode = p.flowMode;
    // 프리셋은 명시적 사용자 의사다 — 자동 규칙에서 해방시킨다
    leftAutoCollapsed = false;
    rightAutoCollapsed = false;
    dockAutoCollapsed = false;
    paintLeft();
    paintRight();
    paintDock();
    paintFlow();
    persist();
    notifyResize();
  };

  return {
    el: root,
    slots: {
      commandBar: commandBarSlot,
      left: leftSlot,
      viewport: viewportSlot,
      rightStack: rightSlot,
      dock: dockSlot,
      flowGraph: flowGraphPane,
      leftRail,
      rightRail,
    },
    setLeftCollapsed: (c: boolean): void => {
      setLeftCollapsed(c);
    },
    setRightCollapsed: (c: boolean): void => {
      setRightCollapsed(c);
    },
    setDockCollapsed: (c: boolean): void => {
      setDockCollapsed(c);
    },
    isDockCollapsed: (): boolean => dockCollapsed,
    setFlowMode,
    getFlowMode: (): FlowMode => flowMode,
    setFlowGraphVisible: (visible: boolean): void => {
      setFlowMode(visible ? 'full' : 'off');
    },
    applyPreset,
    notifyResize,
    dispose: (): void => {
      observer.disconnect();
      if (resizeRafId !== null) {
        cancelAnimationFrame(resizeRafId);
        resizeRafId = null;
      }
      if (autoRafId !== null) {
        cancelAnimationFrame(autoRafId);
        autoRafId = null;
      }
      if (persistTimer !== null) {
        window.clearTimeout(persistTimer);
        persist();
      }
      root.remove();
    },
  };
}
