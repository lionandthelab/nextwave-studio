// ui/workspace.ts — 워크스페이스 셸 (UX_DESIGN §2: 커맨드바 + 3존 + 독 단일 페이지 스튜디오)
//
// CSS grid 레이아웃으로 슬롯(placement)만 소유한다 — 패널 내용물은 통합자(main.ts)가
// 각 슬롯에 마운트한다 (계층 규칙 CLAUDE.md §3: 이 모듈은 core/render를 모른다).
//
//   ┌───────────────────────────────────────────────┐
//   │ commandBar (row 1, 전체 폭)                    │
//   ├──────┬─┬───────────────────────────┬─┬────────┤
//   │ left │║│ center: viewport          │║│ right  │   ║ = 세로 스플리터
//   │      │║│         ═══ fg 스플리터    │║│ stack  │
//   │      │║│         flowGraph (숨김)   │║│        │
//   ├──────┴─┴───────────────────────────┴─┴────────┤
//   │ ═══ dock 스플리터                              │
//   │ dock (row 4, 전체 폭)                          │
//   └───────────────────────────────────────────────┘
//
// - 좌 패널 ~240px, 접으면 36px 아이콘 레일 (UX §2 "좌/우 패널은 접기 가능").
// - 중앙은 viewport(상)/flowGraph(하) 분할 페인 — flowGraph는 기본 숨김(Phase 8 몫),
//   보일 때 최소 높이 200px. 자리 표시 문구를 이 모듈이 그려 둔다.
// - 스플리터: 좌/우 폭 · 독 높이 · viewport/flowGraph 분할을 포인터 드래그로 조절
//   (min/max 클램프 + col/row-resize 커서 + 키보드 화살표 — UX §9 접근성).
//
// ── 뷰포트 슬롯과 렌더러 리사이즈 계약 ─────────────────────────────
// viewport 슬롯은 "position:relative + overflow:hidden"의 평범한 div다. 통합자는
// 기존 렌더러 호스트(#app 등)를 이 슬롯으로 reparent하고 width/height 100%로 채우면
// 된다 — 렌더러(render/renderer.ts)는 host.clientWidth/Height를 window 'resize'
// 이벤트에서 다시 읽으므로, 이 모듈은 스플리터 드래그/접기 후 notifyResize()로
// window resize 이벤트를 합성 발행해 캔버스 크기를 따라오게 한다.

import { COLOR, FONT, LAYOUT, ensureThemeStyles, makeButton, styled } from './theme';

// ── 크기 상수 (px — 매직넘버 금지, CLAUDE.md §4) ────────────────────

export interface PanelSizeLimits {
  readonly defaultPx: number;
  readonly minPx: number;
  readonly maxPx: number;
}

/** 패널별 기본/최소/최대 크기 — 스플리터 클램프의 단일 진실 (테스트 대상) */
export const WORKSPACE_SIZES: Readonly<
  Record<'left' | 'right' | 'dock' | 'flowGraph', PanelSizeLimits>
> = {
  /** 좌 라이브러리 패널 폭 (UX §2 "~240px") */
  left: { defaultPx: 240, minPx: 180, maxPx: 420 },
  /** 우 인스펙터 스택 폭 (UX §2 "~280px" — theme LAYOUT.rightPanelWidthPx와 일치) */
  right: { defaultPx: LAYOUT.rightPanelWidthPx, minPx: 220, maxPx: 460 },
  /** 하단 독 높이 (본문 180px + 탭바 여유 — theme LAYOUT.dockBodyHeightPx 기준) */
  dock: { defaultPx: LAYOUT.dockBodyHeightPx + 30, minPx: 120, maxPx: 440 },
  /** flowGraph 페인 높이 — 보일 때 최소 200px (Phase 7 요구) */
  flowGraph: { defaultPx: 240, minPx: 200, maxPx: 560 },
};

/** 접힌 좌 패널 아이콘 레일 폭 */
export const LEFT_RAIL_WIDTH_PX = 36;

/** 스플리터 두께 / 키보드 조절 스텝 */
const SPLITTER_THICKNESS_PX = 5;
const SPLITTER_KEY_STEP_PX = 16;

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

// ── 공개 타입 ───────────────────────────────────────────────────────

/** 통합자가 각 패널을 마운트하는 슬롯 div들 (셸이 배치 소유, 내용물은 통합자 소유) */
export interface WorkspaceSlots {
  /** 상단 커맨드바 (기존 커맨드바 셸이 이 안에 마운트) */
  readonly commandBar: HTMLElement;
  /** 좌 라이브러리 패널 (mountLibrary 대상) */
  readonly left: HTMLElement;
  /** 중앙 3D 뷰포트 — 렌더러 캔버스 호스트를 reparent (파일 헤더의 리사이즈 계약) */
  readonly viewport: HTMLElement;
  /** 우 패널 스택 (관절 패널 + 인스펙터) */
  readonly rightStack: HTMLElement;
  /** 하단 독 */
  readonly dock: HTMLElement;
  /** 중앙 하단 플로우 그래프 페인 (기본 숨김 — Phase 8이 채운다) */
  readonly flowGraph: HTMLElement;
}

export interface WorkspaceHandle {
  /** 셸 루트 (host의 자식) */
  readonly el: HTMLElement;
  readonly slots: WorkspaceSlots;
  /** 좌 패널 접기/펼치기 — 접으면 36px 아이콘 레일만 남는다 */
  setLeftCollapsed(collapsed: boolean): void;
  /** flowGraph 페인 표시/숨김 (보일 때 최소 높이 200px) */
  setFlowGraphVisible(visible: boolean): void;
  /** 레이아웃 변경을 렌더러에 전파 — window 'resize' 이벤트 합성 발행 */
  notifyResize(): void;
  dispose(): void;
}

// ── 셸 전용 스타일 (스플리터 hover/드래그/포커스 — 토큰만 소비) ─────

const WORKSPACE_STYLE_ID = 'rsw-workspace-styles';

function ensureWorkspaceStyles(): void {
  if (document.getElementById(WORKSPACE_STYLE_ID) !== null) return;
  const style = document.createElement('style');
  style.id = WORKSPACE_STYLE_ID;
  style.textContent = `
.rsw-split {
  flex-shrink: 0;
  background: ${COLOR.borderSoft};
  transition: background-color 0.12s ease;
  touch-action: none;
  user-select: none;
}
.rsw-split--v { cursor: col-resize; width: ${SPLITTER_THICKNESS_PX}px; }
.rsw-split--h { cursor: row-resize; height: ${SPLITTER_THICKNESS_PX}px; }
.rsw-split:hover, .rsw-split--drag { background: var(--rsw-accent); }
.rsw-split:focus-visible { outline: 2px solid var(--rsw-accent); outline-offset: -2px; }
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
  splitter.tabIndex = 0;

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

  // 키보드 조절 (UX §9): 세로 스플리터 ←/→, 가로 스플리터 ↑/↓ — sign과 동일 규약
  splitter.addEventListener('keydown', (e) => {
    const growKey = opts.orientation === 'vertical' ? 'ArrowRight' : 'ArrowDown';
    const shrinkKey = opts.orientation === 'vertical' ? 'ArrowLeft' : 'ArrowUp';
    if (e.key !== growKey && e.key !== shrinkKey) return;
    const deltaPx = e.key === growKey ? SPLITTER_KEY_STEP_PX : -SPLITTER_KEY_STEP_PX;
    opts.setSize(dragPanelSize(opts.getSize(), deltaPx, opts.sign, opts.limits));
    opts.onResized();
    e.preventDefault();
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
  let leftWidthPx = WORKSPACE_SIZES.left.defaultPx;
  let rightWidthPx = WORKSPACE_SIZES.right.defaultPx;
  let dockHeightPx = WORKSPACE_SIZES.dock.defaultPx;
  let flowGraphHeightPx = WORKSPACE_SIZES.flowGraph.defaultPx;
  let leftCollapsed = false;
  let flowGraphVisible = false;

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
    background: COLOR.bgApp,
    color: COLOR.text,
    fontFamily: FONT.ui,
    fontSize: '12px',
    overflow: 'hidden',
    boxSizing: 'border-box',
  });
  root.dataset.testid = 'workspace';

  // 행 1: 커맨드바 슬롯 (전체 폭)
  const commandBarSlot = styled(document.createElement('div'), {
    gridRow: '1',
    gridColumn: '1 / -1',
    minHeight: '0',
  });
  commandBarSlot.dataset.testid = 'workspace-command-bar';

  // ── 행 2 열 1: 좌 패널 (접기 레일 포함) ───────────────────────────
  const leftWrap = styled(document.createElement('div'), {
    gridRow: '2',
    gridColumn: '1',
    display: 'flex',
    flexDirection: 'column',
    minWidth: '0',
    minHeight: '0',
    overflow: 'hidden',
    background: COLOR.bgPanel,
    boxSizing: 'border-box',
  });
  leftWrap.dataset.testid = 'workspace-left-wrap';

  // 레일 헤더 (셸 소유): 펼침 상태 = 우측 정렬 « 버튼, 접힘 상태 = 세로 레일
  const leftRail = styled(document.createElement('div'), {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'flex-end',
    flexShrink: '0',
    padding: '2px',
    borderBottom: `1px solid ${COLOR.borderSoft}`,
  });
  const leftToggle = makeButton('«', '라이브러리 패널 접기', 'workspace-left-toggle', 'ghost');
  styled(leftToggle, { padding: '0 6px' });
  leftRail.appendChild(leftToggle);

  // 접힘 레일 안내 (세로 쓰기 라벨 — 클릭해도 펼쳐진다)
  const leftRailLabel = styled(document.createElement('button'), {
    display: 'none',
    background: 'transparent',
    border: 'none',
    color: COLOR.muted,
    font: `11px ${FONT.ui}`,
    writingMode: 'vertical-rl',
    padding: '10px 0',
    margin: '0 auto',
    cursor: 'pointer',
    letterSpacing: '0.08em',
  });
  leftRailLabel.type = 'button';
  leftRailLabel.textContent = '📚 Library';
  leftRailLabel.title = '라이브러리 패널 펼치기';
  leftRailLabel.setAttribute('aria-label', '라이브러리 패널 펼치기');
  leftRailLabel.dataset.testid = 'workspace-left-rail-label';

  const leftSlot = styled(document.createElement('div'), {
    flex: '1 1 auto',
    minHeight: '0',
    position: 'relative',
    overflow: 'hidden',
  });
  leftSlot.dataset.testid = 'workspace-left';

  leftWrap.appendChild(leftRail);
  leftWrap.appendChild(leftRailLabel);
  leftWrap.appendChild(leftSlot);

  // ── 행 2 열 3: 중앙 (viewport 상 / flowGraph 하 분할 페인) ────────
  const centerWrap = styled(document.createElement('div'), {
    gridRow: '2',
    gridColumn: '3',
    display: 'flex',
    flexDirection: 'column',
    minWidth: '0',
    minHeight: '0',
    overflow: 'hidden',
  });
  centerWrap.dataset.testid = 'workspace-center';

  // 뷰포트 슬롯 — 평범한 positioned div (파일 헤더의 reparent/리사이즈 계약)
  const viewportSlot = styled(document.createElement('div'), {
    flex: '1 1 auto',
    minHeight: '0',
    position: 'relative',
    overflow: 'hidden',
  });
  viewportSlot.dataset.testid = 'workspace-viewport';

  // flowGraph 페인 (기본 숨김 — Phase 8이 내용을 채운다)
  const flowGraphPane = styled(document.createElement('div'), {
    display: 'none',
    flexShrink: '0',
    height: `${flowGraphHeightPx}px`,
    minHeight: `${WORKSPACE_SIZES.flowGraph.minPx}px`,
    position: 'relative',
    overflow: 'hidden',
    background: COLOR.bgPanel,
    borderTop: `1px solid ${COLOR.border}`,
  });
  flowGraphPane.dataset.testid = 'workspace-flow-graph';

  // 자리 표시 문구 (은은한 안내 — 내용이 채워져도 방해하지 않게 pointer-events 없음)
  const flowGraphPlaceholder = styled(document.createElement('div'), {
    position: 'absolute',
    inset: '0',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    color: COLOR.muted,
    fontSize: '12px',
    pointerEvents: 'none',
    userSelect: 'none',
  });
  flowGraphPlaceholder.textContent = '플로우 그래프 — Phase 8';
  flowGraphPane.appendChild(flowGraphPlaceholder);

  const flowGraphSplitter = makeSplitter({
    orientation: 'horizontal',
    labelKo: '뷰포트/플로우 그래프 분할 크기 조절',
    testId: 'workspace-split-flow-graph',
    sign: -1, // 위로 끌수록 flowGraph가 커진다
    limits: WORKSPACE_SIZES.flowGraph,
    getSize: () => flowGraphHeightPx,
    setSize: (px) => {
      flowGraphHeightPx = px;
      flowGraphPane.style.height = `${px}px`;
    },
    onResized: notifyResizeThrottled,
  });
  flowGraphSplitter.style.display = 'none'; // flowGraph와 함께 표시

  centerWrap.appendChild(viewportSlot);
  centerWrap.appendChild(flowGraphSplitter);
  centerWrap.appendChild(flowGraphPane);

  // ── 행 2 열 5: 우 패널 스택 ───────────────────────────────────────
  // width를 처음부터 확정한다: 열 5는 auto 트랙이라 폭을 비워 두면 스택 안 **내용의
  // max-content**가 열 폭이 된다 — 긴 안내 문구 한 줄이나 넓은 폼이 들어오는 순간
  // 패널이 UX §2의 "~280px"을 넘어 뷰포트를 잠식하고(실측: 로봇 선택 시 247 → 750 px),
  // 그리드는 좁아졌는데 캔버스는 resize 통지를 못 받아 패널 아래로 밀려 들어가
  // 그 띠에서 드롭·클릭이 패널에 먹힌다. 폭 변경은 스플리터만 한다(setSize).
  const rightWrap = styled(document.createElement('div'), {
    gridRow: '2',
    gridColumn: '5',
    display: 'flex',
    flexDirection: 'column',
    width: `${rightWidthPx}px`,
    minWidth: '0',
    minHeight: '0',
    overflow: 'hidden',
    background: COLOR.bgPanel,
    boxSizing: 'border-box',
  });
  rightWrap.dataset.testid = 'workspace-right-wrap';
  const rightSlot = styled(document.createElement('div'), {
    flex: '1 1 auto',
    minHeight: '0',
    position: 'relative',
    overflowY: 'auto',
  });
  rightSlot.classList.add('ui-scroll');
  rightSlot.dataset.testid = 'workspace-right-stack';
  rightWrap.appendChild(rightSlot);

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
      rightWrap.style.width = `${px}px`;
    },
    onResized: notifyResizeThrottled,
  });
  styled(rightSplitter, { gridRow: '2', gridColumn: '4' });

  // ── 행 3/4: 독 스플리터 + 독 슬롯 ─────────────────────────────────
  const dockSlot = styled(document.createElement('div'), {
    gridRow: '4',
    gridColumn: '1 / -1',
    height: `${dockHeightPx}px`,
    minHeight: '0',
    position: 'relative',
    overflow: 'hidden',
    background: COLOR.bgBar,
    borderTop: `1px solid ${COLOR.border}`,
  });
  dockSlot.dataset.testid = 'workspace-dock';

  const dockSplitter = makeSplitter({
    orientation: 'horizontal',
    labelKo: '독 높이 조절',
    testId: 'workspace-split-dock',
    sign: -1, // 위로 끌수록 독이 커진다
    limits: WORKSPACE_SIZES.dock,
    getSize: () => dockHeightPx,
    setSize: (px) => {
      dockHeightPx = px;
      dockSlot.style.height = `${px}px`;
    },
    onResized: notifyResizeThrottled,
  });
  styled(dockSplitter, { gridRow: '3', gridColumn: '1 / -1' });

  // ── 조립 ──────────────────────────────────────────────────────────
  root.appendChild(commandBarSlot);
  root.appendChild(leftWrap);
  root.appendChild(leftSplitter);
  root.appendChild(centerWrap);
  root.appendChild(rightSplitter);
  root.appendChild(rightWrap);
  root.appendChild(dockSplitter);
  root.appendChild(dockSlot);
  host.appendChild(root);

  // ── 좌 패널 접기 (36px 아이콘 레일 — UX §2) ───────────────────────
  const paintLeftCollapse = (): void => {
    leftWrap.style.width = leftCollapsed ? `${LEFT_RAIL_WIDTH_PX}px` : `${leftWidthPx}px`;
    leftSlot.style.display = leftCollapsed ? 'none' : '';
    leftRailLabel.style.display = leftCollapsed ? 'block' : 'none';
    leftSplitter.style.display = leftCollapsed ? 'none' : '';
    leftRail.style.justifyContent = leftCollapsed ? 'center' : 'flex-end';
    leftToggle.textContent = leftCollapsed ? '»' : '«';
    const title = leftCollapsed ? '라이브러리 패널 펼치기' : '라이브러리 패널 접기';
    leftToggle.title = title;
    leftToggle.setAttribute('aria-label', title);
    leftToggle.setAttribute('aria-expanded', String(!leftCollapsed));
  };
  const setLeftCollapsed = (collapsed: boolean): void => {
    if (collapsed === leftCollapsed) return;
    leftCollapsed = collapsed;
    paintLeftCollapse();
    notifyResize();
  };
  leftToggle.addEventListener('click', () => {
    setLeftCollapsed(!leftCollapsed);
  });
  leftRailLabel.addEventListener('click', () => {
    setLeftCollapsed(false);
  });
  paintLeftCollapse();

  // ── flowGraph 표시/숨김 ───────────────────────────────────────────
  const setFlowGraphVisible = (visible: boolean): void => {
    if (visible === flowGraphVisible) return;
    flowGraphVisible = visible;
    flowGraphPane.style.display = visible ? 'block' : 'none';
    flowGraphSplitter.style.display = visible ? '' : 'none';
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
    },
    setLeftCollapsed,
    setFlowGraphVisible,
    notifyResize,
    dispose: (): void => {
      if (resizeRafId !== null) {
        cancelAnimationFrame(resizeRafId);
        resizeRafId = null;
      }
      root.remove();
    },
  };
}
