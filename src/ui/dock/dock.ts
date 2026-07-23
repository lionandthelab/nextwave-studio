// ui/dock/dock.ts — 하단 독 셸: [Timeline | Collision Log | Console] 탭 (UX_DESIGN §3.6)
//
// 다크 미니멀·접기 가능·높이 ~180px. #app 캔버스 위 오버레이(fixed bottom)로 얹는다 —
// 독 내부 포인터/휠은 stopPropagation으로 흡수해 뷰포트 orbit 컨트롤로 새지 않게 한다.
// 탭 콘텐츠(el)는 각 패널 모듈(createTimelinePanel 등)이 만들고, 이 셸은 배치와
// 탭 전환만 담당한다.

// ── 상수 (매직넘버 금지 — CLAUDE.md §4) ─────────────────────────────

/** 독 본문 높이(px) — UX_DESIGN §3.6 기준 컴팩트 독 */
const DOCK_BODY_HEIGHT_PX = 180;
/** 오류 오버레이(z 9999)·관절 패널(z 100)보다 아래, 캔버스보다 위 */
const DOCK_Z_INDEX = '90';

const TAB_ACTIVE_COLOR = '#e8eaed';
const TAB_INACTIVE_COLOR = '#7a808a';
const TAB_ACTIVE_BORDER = '#2e5db3';

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

// ── 내부 헬퍼 ───────────────────────────────────────────────────────

function styled<T extends HTMLElement>(el: T, style: Partial<CSSStyleDeclaration>): T {
  Object.assign(el.style, style);
  return el;
}

// ── 마운트 ──────────────────────────────────────────────────────────

/** 독을 host(보통 document.body)에 오버레이로 마운트한다. 첫 탭이 기본 활성. */
export function mountDock(host: HTMLElement, tabs: readonly DockTab[]): DockHandle {
  const dock = styled(document.createElement('div'), {
    position: 'fixed',
    left: '0',
    right: '0',
    bottom: '0',
    zIndex: DOCK_Z_INDEX,
    display: 'flex',
    flexDirection: 'column',
    background: 'rgba(16, 18, 22, 0.94)',
    borderTop: '1px solid #2e3238',
    color: '#cfd3d9',
    fontFamily: 'ui-monospace, SFMono-Regular, Consolas, monospace',
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
    padding: '0 6px',
    flexShrink: '0',
    borderBottom: '1px solid #22252b',
  });

  // 본문 (탭 콘텐츠 컨테이너)
  const body = styled(document.createElement('div'), {
    height: `${DOCK_BODY_HEIGHT_PX}px`,
    overflow: 'hidden',
  });

  let collapsed = false;
  let activeLabel = tabs[0]?.label ?? '';
  const tabButtons = new Map<string, HTMLButtonElement>();

  const paint = (): void => {
    body.style.display = collapsed ? 'none' : '';
    for (const tab of tabs) {
      const button = tabButtons.get(tab.label);
      const isActive = tab.label === activeLabel;
      if (button) {
        button.style.color = isActive ? TAB_ACTIVE_COLOR : TAB_INACTIVE_COLOR;
        button.style.borderBottom = `2px solid ${isActive && !collapsed ? TAB_ACTIVE_BORDER : 'transparent'}`;
      }
      tab.content.style.display = isActive ? '' : 'none';
    }
  };

  for (const tab of tabs) {
    const button = styled(document.createElement('button'), {
      background: 'transparent',
      color: TAB_INACTIVE_COLOR,
      border: 'none',
      borderBottom: '2px solid transparent',
      padding: '5px 10px',
      fontFamily: 'inherit',
      fontSize: '12px',
      cursor: 'pointer',
    });
    button.type = 'button';
    button.textContent = tab.label;
    button.dataset.testid = `dock-tab-${tab.label.toLowerCase().replace(/\s+/g, '-')}`;
    button.addEventListener('click', () => {
      activeLabel = tab.label;
      if (collapsed) collapsed = false; // 접힌 상태에서 탭 클릭 = 펼치기
      paint();
    });
    tabBar.appendChild(button);
    tabButtons.set(tab.label, button);
    styled(tab.content, { height: '100%' });
    body.appendChild(tab.content);
  }

  // 접기/펼치기 토글 (우측 끝)
  const spacer = styled(document.createElement('span'), { flex: '1' });
  tabBar.appendChild(spacer);
  const collapseButton = styled(document.createElement('button'), {
    background: 'transparent',
    color: TAB_INACTIVE_COLOR,
    border: 'none',
    padding: '5px 10px',
    fontFamily: 'inherit',
    fontSize: '12px',
    cursor: 'pointer',
  });
  collapseButton.type = 'button';
  collapseButton.dataset.testid = 'dock-collapse';
  const paintCollapseLabel = (): void => {
    collapseButton.textContent = collapsed ? '▴ 펼치기' : '▾ 접기';
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
