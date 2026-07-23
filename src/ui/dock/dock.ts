// ui/dock/dock.ts — 하단 독 셸: [Timeline | Collision Log | Console] 탭 (UX_DESIGN §3.6)
//
// 다크 미니멀·접기 가능·높이 ~180px. #app 캔버스 위 오버레이(fixed bottom)로 얹는다 —
// 독 내부 포인터/휠은 stopPropagation으로 흡수해 뷰포트 orbit 컨트롤로 새지 않게 한다.
// 탭 콘텐츠(el)는 각 패널 모듈(createTimelinePanel 등)이 만들고, 이 셸은 배치와
// 탭 전환만 담당한다.

import {
  COLOR,
  FONT,
  LAYOUT,
  SPACE,
  Z_INDEX,
  ensureThemeStyles,
  styled,
} from '../theme';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4, 시각 토큰은 ui/theme.ts) ────

/** 독 본문 높이(px) — UX_DESIGN §3.6 기준 컴팩트 독 */
const DOCK_BODY_HEIGHT_PX = LAYOUT.dockBodyHeightPx;

// ── 공개 타입 ───────────────────────────────────────────────────────

export interface DockTab {
  /** 탭 버튼 라벨 */
  readonly label: string;
  /** 탭 콘텐츠 루트 (패널 모듈이 생성) */
  readonly content: HTMLElement;
}

export interface DockHandle {
  readonly el: HTMLElement;
  /** 라벨로 탭 활성화 (미존재 라벨은 no-op) */
  activateTab(label: string): void;
  dispose(): void;
}

// ── 마운트 ──────────────────────────────────────────────────────────

/** 독을 host(보통 document.body)에 오버레이로 마운트한다. 첫 탭이 기본 활성. */
export function mountDock(host: HTMLElement, tabs: readonly DockTab[]): DockHandle {
  ensureThemeStyles();
  const dock = styled(document.createElement('div'), {
    position: 'fixed',
    left: '0',
    right: '0',
    bottom: '0',
    zIndex: Z_INDEX.dock,
    display: 'flex',
    flexDirection: 'column',
    background: COLOR.bgBar,
    borderTop: `1px solid ${COLOR.border}`,
    color: COLOR.text,
    fontFamily: FONT.ui,
    fontSize: '12px',
    boxSizing: 'border-box',
    pointerEvents: 'auto',
  });
  dock.dataset.testid = 'dock';
  // 독 위 상호작용이 뷰포트(OrbitControls)로 전파되지 않게 차단 (joint-panel과 동일 규약)
  for (const type of ['pointerdown', 'pointermove', 'pointerup', 'wheel', 'contextmenu']) {
    dock.addEventListener(type, (e) => {
      e.stopPropagation();
    });
  }

  // 탭 바
  const tabBar = styled(document.createElement('div'), {
    display: 'flex',
    alignItems: 'center',
    gap: '2px',
    padding: `0 ${SPACE.sm}`,
    flexShrink: '0',
    borderBottom: `1px solid ${COLOR.borderSoft}`,
  });

  // 본문 (탭 콘텐츠 컨테이너) — 접기는 height 트랜지션으로 부드럽게 (.ui-collapsible-h)
  const body = styled(document.createElement('div'), {
    height: `${DOCK_BODY_HEIGHT_PX}px`,
    overflow: 'hidden',
  });
  body.classList.add('ui-collapsible-h');

  let collapsed = false;
  let activeLabel = tabs[0]?.label ?? '';
  const tabButtons = new Map<string, HTMLButtonElement>();

  const paint = (): void => {
    body.style.height = collapsed ? '0px' : `${DOCK_BODY_HEIGHT_PX}px`;
    for (const tab of tabs) {
      const button = tabButtons.get(tab.label);
      const isActive = tab.label === activeLabel;
      if (button) {
        button.classList.toggle('ui-tab--active', isActive && !collapsed);
        button.setAttribute('aria-selected', String(isActive));
      }
      tab.content.style.display = isActive ? '' : 'none';
    }
  };

  for (const tab of tabs) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'ui-tab';
    button.textContent = tab.label;
    button.dataset.testid = `dock-tab-${tab.label.toLowerCase().replace(/\s+/g, '-')}`;
    button.addEventListener('click', () => {
      activeLabel = tab.label;
      if (collapsed) collapsed = false; // 접힌 상태에서 탭 클릭 = 펼치기
      paintCollapseLabel();
      paint();
    });
    tabBar.appendChild(button);
    tabButtons.set(tab.label, button);
    styled(tab.content, { height: '100%' });
    body.appendChild(tab.content);
  }

  // 접기/펼치기 토글 (우측 끝) — ▾/▴ 셰브론 규약 (joint-panel/inspector와 동일)
  const spacer = styled(document.createElement('span'), { flex: '1' });
  tabBar.appendChild(spacer);
  const collapseButton = document.createElement('button');
  collapseButton.type = 'button';
  collapseButton.className = 'ui-tab';
  collapseButton.dataset.testid = 'dock-collapse';
  const paintCollapseLabel = (): void => {
    collapseButton.textContent = collapsed ? '▴ 펼치기' : '▾ 접기';
    collapseButton.setAttribute('aria-expanded', String(!collapsed));
    collapseButton.setAttribute('aria-label', collapsed ? '독 펼치기' : '독 접기');
  };
  collapseButton.addEventListener('click', () => {
    collapsed = !collapsed;
    paintCollapseLabel();
    paint();
  });
  paintCollapseLabel();
  tabBar.appendChild(collapseButton);

  dock.appendChild(tabBar);
  dock.appendChild(body);
  host.appendChild(dock);
  paint();

  return {
    el: dock,
    activateTab: (label): void => {
      if (!tabButtons.has(label)) return;
      activeLabel = label;
      collapsed = false;
      paintCollapseLabel();
      paint();
    },
    dispose: (): void => {
      dock.remove();
    },
  };
}
