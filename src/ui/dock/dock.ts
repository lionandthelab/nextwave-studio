// ui/dock/dock.ts — 하단 독 셸: [타임라인 | 충돌 로그 | 콘솔] 탭 (UX_DESIGN §3.6)
//
// 다크 미니멀·접기 가능. #app 캔버스 위 오버레이(fixed bottom)로 얹거나 워크스페이스
// 그리드 슬롯에 심는다 — 독 내부 포인터/휠은 stopPropagation으로 흡수해 뷰포트 orbit
// 컨트롤로 새지 않게 한다. 탭 콘텐츠(el)는 각 패널 모듈(createTimelinePanel 등)이
// 만들고, 이 셸은 배치와 탭 전환만 담당한다.
//
// ── Phase 11에서 닫는 감사 항목 ─────────────────────────────────────
// C-7  탭 카운트 배지(`setBadge`) — 비활성 탭에 쌓인 충돌을 화면에 등장시킨다.
//      배지가 없던 탓에 waitForCollision을 포함한 시퀀스가 완주해도 "충돌이 있었다"는
//      신호가 화면 어디에도 없었다.
// C-1  접기가 픽셀을 실제로 돌려준다 — 내부 body만 0으로 만들던 것을
//      `onCollapseChange`로 워크스페이스(그리드 슬롯 높이)에 전파한다. 접힌 상태에서도
//      정보 손실이 0이 되도록 탭바 우측 **인라인 스트립**(`stripEl`)을 제공한다.
// C-18 WAI-ARIA APG Tabs 패턴 — role 없는 <button>에 aria-selected를 걸어 활성 탭이
//      스크린리더에 전혀 전달되지 않던 문제를 닫는다(tablist/tab/tabpanel + roving).

import { rovingTabindex } from '../a11y';
import { setButtonContent } from '../icons';
import {
  BORDER,
  BORDER_WIDTH,
  COLOR,
  LAYOUT,
  SPACE,
  SURFACE,
  TYPE,
  Z_INDEX,
  applyType,
  ensureThemeStyles,
  styled,
} from '../theme';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4, 시각 토큰은 ui/theme.ts) ────

/** 독 본문 높이(px) — UX_DESIGN §3.6 기준 컴팩트 독 */
const DOCK_BODY_HEIGHT_PX = LAYOUT.dockBodyHeightPx;

/** 배지 표시 상한 — 초과분은 `99+`로 접는다 (탭 폭 폭주 방지) */
const BADGE_MAX = 99;

/** 표준 탭 id — 통합자(main.ts)와 배지 소스(collision-log)가 공유하는 안정 식별자.
 *  라벨은 한국어로 바뀌지만 id는 바뀌지 않는다(선택자·배지 대상의 단일 진실). */
export const DOCK_TAB_ID = {
  timeline: 'timeline',
  collision: 'collision',
  console: 'console',
} as const;

/** 마운트 인스턴스 일련번호 — DOM id 충돌 방지 (aria-controls/labelledby 짝) */
let dockInstanceSeq = 0;

// ── 순수 헬퍼 (DOM 비의존 — 단위 테스트 대상) ──────────────────────

/** 라벨 → 안정 슬러그 (탭 id 미지정 시 폴백). 공백은 하이픈으로 접는다. */
export function dockTabSlug(label: string): string {
  return label.trim().toLowerCase().replace(/\s+/g, '-');
}

/**
 * 배지 카운트 → 표시 문자열. 0 이하·null·비유한수는 "배지 없음"(null)이다 —
 * 0을 그려 놓으면 "새 이벤트 0건"이라는 무의미한 시각 소음이 남는다.
 */
export function formatDockBadge(count: number | null | undefined): string | null {
  if (count === null || count === undefined) return null;
  if (!Number.isFinite(count)) return null;
  const n = Math.floor(count);
  if (n <= 0) return null;
  return n > BADGE_MAX ? `${BADGE_MAX}+` : String(n);
}

// ── 공개 타입 ───────────────────────────────────────────────────────

export interface DockTab {
  /** 안정 식별자 (배지·활성화 대상). 미지정이면 라벨 슬러그를 쓴다 */
  readonly id?: string;
  /** 탭 버튼 라벨 (한국어 UI 크롬) */
  readonly label: string;
  /** 탭 콘텐츠 루트 (패널 모듈이 생성) */
  readonly content: HTMLElement;
  /** 초기 배지 카운트 (보통 생략) */
  readonly badge?: number | null;
}

export interface MountDockOptions {
  /**
   * 탭바 우측에 심을 인라인 스트립 콘텐츠 (보통 timelinePanel.stripEl).
   * **접힌 상태에서도 보인다** — 접기가 정보 손실을 만들지 않게 하는 장치(C-1).
   */
  readonly strip?: HTMLElement;
  /** 초기 접힘 상태 (기본 false) */
  readonly initialCollapsed?: boolean;
  /**
   * 접힘 상태 변화 통지 — 워크스페이스가 이걸 받아 **그리드 슬롯 높이**를 줄인다.
   * 이게 없으면 독 내부 body만 0이 되고 뷰포트는 1px도 커지지 않는다(C-1).
   */
  onCollapseChange?(collapsed: boolean): void;
  /**
   * 탭 활성화 통지. `clearedBadge`는 활성화 직전 그 탭에 쌓여 있던 배지 카운트다
   * (통합자가 "첫 충돌 1회 토스트" 같은 단발 피드백에 쓴다).
   */
  onTabActivated?(tabId: string, clearedBadge: number | null): void;
}

export interface DockHandle {
  readonly el: HTMLElement;
  /**
   * 탭바 우측 인라인 스트립 컨테이너. 접힘 상태에서도 보이므로 리드아웃/진행 바를
   * 여기에 둔다(C-1). mountDock 옵션의 `strip`이 이미 여기에 들어가 있다.
   */
  readonly stripEl: HTMLElement;
  /** id 또는 라벨로 탭 활성화 (미존재면 no-op). 접혀 있으면 펼친다 */
  activateTab(tabIdOrLabel: string): void;
  /** 현재 활성 탭 id */
  activeTabId(): string;
  /** 탭 카운트 배지 설정. null/0 이하는 배지 제거 (C-7) */
  setBadge(tabIdOrLabel: string, count: number | null): void;
  /** 현재 배지 카운트 (없으면 null) */
  badgeOf(tabIdOrLabel: string): number | null;
  /** 외부에서 접기/펼치기 제어 (단축키·반응형 자동 접힘) */
  setCollapsed(collapsed: boolean): void;
  isCollapsed(): boolean;
  /** 탭 활성화 구독 (여러 개 등록 가능). 반환값은 해제 함수 */
  onTabActivated(cb: (tabId: string, clearedBadge: number | null) => void): () => void;
  dispose(): void;
}

// ── 내부 타입 ───────────────────────────────────────────────────────

interface TabEntry {
  readonly id: string;
  readonly label: string;
  readonly content: HTMLElement;
  readonly button: HTMLButtonElement;
  badge: number | null;
  badgeEl: HTMLElement | null;
  badgeSrEl: HTMLElement | null;
}

// ── 마운트 ──────────────────────────────────────────────────────────

/** 독을 host에 마운트한다. 첫 탭이 기본 활성. */
export function mountDock(
  host: HTMLElement,
  tabs: readonly DockTab[],
  opts: MountDockOptions = {},
): DockHandle {
  ensureThemeStyles();
  const seq = ++dockInstanceSeq;
  const tabDomId = (id: string): string => `rsw-dock${seq}-tab-${id}`;
  const panelDomId = (id: string): string => `rsw-dock${seq}-panel-${id}`;
  const bodyDomId = `rsw-dock${seq}-body`;

  const dock = styled(document.createElement('div'), {
    position: 'fixed',
    left: '0',
    right: '0',
    bottom: '0',
    zIndex: Z_INDEX.dock,
    display: 'flex',
    flexDirection: 'column',
    background: SURFACE.panel,
    borderTop: `${BORDER_WIDTH.hair} solid ${BORDER.default}`,
    color: COLOR.text,
    boxSizing: 'border-box',
    pointerEvents: 'auto',
  });
  applyType(dock, TYPE.body);
  dock.dataset.testid = 'dock';
  // 독 위 상호작용이 뷰포트(OrbitControls)로 전파되지 않게 차단 (joint-panel과 동일 규약)
  for (const type of ['pointerdown', 'pointermove', 'pointerup', 'wheel', 'contextmenu']) {
    dock.addEventListener(type, (e) => {
      e.stopPropagation();
    });
  }

  // 탭 바 = [tablist][인라인 스트립][접기 토글]
  // tablist에는 **탭만** 들어간다 — 접기 버튼을 tablist 안에 두면 role 계약이 깨진다.
  const tabBar = styled(document.createElement('div'), {
    display: 'flex',
    alignItems: 'center',
    gap: SPACE.md,
    padding: `0 ${SPACE.md}`,
    flexShrink: '0',
    borderBottom: `${BORDER_WIDTH.hair} solid ${BORDER.subtle}`,
  });
  tabBar.dataset.testid = 'dock-tabbar';

  const tabList = styled(document.createElement('div'), {
    display: 'flex',
    alignItems: 'center',
    gap: SPACE.xxs,
    flexShrink: '0',
  });
  tabList.setAttribute('role', 'tablist');
  tabList.setAttribute('aria-label', '독 패널');
  tabList.setAttribute('aria-orientation', 'horizontal');
  tabBar.appendChild(tabList);

  // 인라인 스트립 — 접혀 있어도 보이는 요약 영역 (타임라인 리드아웃 + 진행 바)
  const stripEl = styled(document.createElement('div'), {
    flex: '1 1 auto',
    minWidth: '0',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'flex-end',
    gap: SPACE.md,
    overflow: 'hidden',
  });
  stripEl.dataset.testid = 'dock-strip';
  if (opts.strip !== undefined) stripEl.appendChild(opts.strip);
  tabBar.appendChild(stripEl);

  // 본문 (탭 콘텐츠 컨테이너) — 접기는 height 트랜지션으로 부드럽게 (.ui-collapsible-h).
  // flex: 1 1 auto라 워크스페이스 슬롯에 심겼을 때(dock height:100%) 남는 높이를 채우고,
  // 단독 오버레이(height:auto)일 때는 위 height가 기준값으로 남는다.
  const body = styled(document.createElement('div'), {
    flex: '1 1 auto',
    height: `${DOCK_BODY_HEIGHT_PX}px`,
    minHeight: '0',
    overflow: 'hidden',
  });
  body.id = bodyDomId;
  body.classList.add('ui-collapsible-h');

  let collapsed = opts.initialCollapsed === true;
  const entries: TabEntry[] = [];
  const byId = new Map<string, TabEntry>();
  const byLabel = new Map<string, TabEntry>();
  const activatedListeners: ((tabId: string, cleared: number | null) => void)[] = [];

  for (const tab of tabs) {
    const id = tab.id ?? dockTabSlug(tab.label);
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'ui-tab';
    button.id = tabDomId(id);
    button.setAttribute('role', 'tab');
    button.setAttribute('aria-controls', panelDomId(id));
    styled(button, { display: 'inline-flex', alignItems: 'center' });
    const labelEl = document.createElement('span');
    labelEl.textContent = tab.label;
    button.appendChild(labelEl);
    button.dataset.testid = `dock-tab-${id}`;

    tab.content.id = panelDomId(id);
    tab.content.setAttribute('role', 'tabpanel');
    tab.content.setAttribute('aria-labelledby', tabDomId(id));
    tab.content.tabIndex = 0;
    styled(tab.content, { height: '100%' });

    const entry: TabEntry = {
      id,
      label: tab.label,
      content: tab.content,
      button,
      badge: formatDockBadge(tab.badge) === null ? null : (tab.badge ?? null),
      badgeEl: null,
      badgeSrEl: null,
    };
    entries.push(entry);
    byId.set(id, entry);
    byLabel.set(tab.label, entry);
    tabList.appendChild(button);
    body.appendChild(tab.content);
  }

  let activeId = entries[0]?.id ?? '';

  const resolve = (tabIdOrLabel: string): TabEntry | undefined =>
    byId.get(tabIdOrLabel) ?? byLabel.get(tabIdOrLabel);

  // ── 배지 (C-7) ────────────────────────────────────────────────────
  // 배지는 **비활성 탭**에 의미가 있다. 단, 접혀 있으면 활성 탭의 콘텐츠도 보이지
  // 않으므로 그때는 활성 탭에도 표시한다(접힘 = 전부 비활성과 같다).
  const paintBadge = (entry: TabEntry): void => {
    const visible = entry.id !== activeId || collapsed;
    const text = visible ? formatDockBadge(entry.badge) : null;
    if (text === null) {
      entry.badgeEl?.remove();
      entry.badgeSrEl?.remove();
      entry.badgeEl = null;
      entry.badgeSrEl = null;
      return;
    }
    if (entry.badgeEl === null || entry.badgeSrEl === null) {
      // 신규 생성 시에만 붙인다 — .ui-tab__badge의 스케일-인 애니메이션이 그때 1회 돈다
      const badgeEl = document.createElement('span');
      badgeEl.className = 'ui-tab__badge';
      badgeEl.setAttribute('aria-hidden', 'true');
      badgeEl.dataset.testid = `dock-badge-${entry.id}`;
      const srEl = document.createElement('span');
      srEl.className = 'sr-only';
      entry.button.appendChild(badgeEl);
      entry.button.appendChild(srEl);
      entry.badgeEl = badgeEl;
      entry.badgeSrEl = srEl;
    }
    entry.badgeEl.textContent = text;
    // 배지 숫자만으로는 "무엇이 3건인지" 전달되지 않는다 — 스크린리더용 문장을 병행
    entry.badgeSrEl.textContent = `, 새 항목 ${text}건`;
  };

  const paint = (): void => {
    body.style.height = collapsed ? '0px' : `${DOCK_BODY_HEIGHT_PX}px`;
    // 접힘은 flex 성장도 끈다 — 안 그러면 슬롯을 줄여도 본문이 다시 늘어난다
    body.style.flex = collapsed ? '0 0 0px' : '1 1 auto';
    dock.dataset.collapsed = String(collapsed);
    // 접힘 = 높이 0. 안쪽 포커스 가능 요소가 보조기술/탭 순서에 남지 않게 격리한다
    if (collapsed) body.setAttribute('inert', '');
    else body.removeAttribute('inert');
    for (const entry of entries) {
      const isActive = entry.id === activeId;
      entry.button.classList.toggle('ui-tab--active', isActive);
      entry.button.setAttribute('aria-selected', String(isActive));
      entry.content.style.display = isActive ? '' : 'none';
      paintBadge(entry);
    }
  };

  // ── 접기 토글 (우측 끝) ───────────────────────────────────────────
  const collapseButton = document.createElement('button');
  collapseButton.type = 'button';
  collapseButton.className = 'ui-btn ui-btn--ghost';
  styled(collapseButton, {
    padding: SPACE.xxs,
    minWidth: '24px',
    minHeight: '24px',
    justifyContent: 'center',
    flexShrink: '0',
  });
  collapseButton.dataset.testid = 'dock-collapse';
  collapseButton.setAttribute('aria-controls', bodyDomId);
  const paintCollapseButton = (): void => {
    setButtonContent(collapseButton, collapsed ? 'chevronUp' : 'chevronDown', '');
    const t = collapsed ? '독 펼치기' : '독 접기';
    collapseButton.title = t;
    collapseButton.setAttribute('aria-label', t);
    collapseButton.setAttribute('aria-expanded', String(!collapsed));
  };
  tabBar.appendChild(collapseButton);

  const applyCollapsed = (next: boolean, notify: boolean): void => {
    if (next === collapsed) return;
    collapsed = next;
    paintCollapseButton();
    paint();
    if (notify) opts.onCollapseChange?.(collapsed);
  };

  collapseButton.addEventListener('click', () => {
    applyCollapsed(!collapsed, true);
  });

  // ── 탭 활성화 ─────────────────────────────────────────────────────
  const activate = (entry: TabEntry): void => {
    const cleared = entry.badge;
    activeId = entry.id;
    entry.badge = null; // 활성 탭의 배지는 의미가 없다 — 진입 즉시 리셋
    // 접힌 상태에서 탭 클릭 = 펼치기 (워크스페이스에도 전파해야 픽셀이 돌아온다)
    applyCollapsed(false, true);
    paint();
    roving.setActive(entries.indexOf(entry));
    opts.onTabActivated?.(entry.id, cleared);
    for (const cb of [...activatedListeners]) cb(entry.id, cleared);
  };

  for (const entry of entries) {
    entry.button.addEventListener('click', () => {
      activate(entry);
    });
  }

  // ←→/Home/End 탐색 (APG Tabs — 수동 활성화: Enter/Space로 선택)
  const roving = rovingTabindex(
    tabList,
    entries.map((e) => e.button),
    {
      orientation: 'horizontal',
      onActivate: (_el, index) => {
        const entry = entries[index];
        if (entry !== undefined) activate(entry);
      },
    },
  );

  dock.appendChild(tabBar);
  dock.appendChild(body);
  host.appendChild(dock);
  paintCollapseButton();
  paint();
  roving.setActive(Math.max(entries.findIndex((e) => e.id === activeId), 0));

  return {
    el: dock,
    stripEl,
    activateTab: (tabIdOrLabel): void => {
      const entry = resolve(tabIdOrLabel);
      if (entry === undefined) return;
      activate(entry);
    },
    activeTabId: (): string => activeId,
    setBadge: (tabIdOrLabel, count): void => {
      const entry = resolve(tabIdOrLabel);
      if (entry === undefined) return;
      entry.badge = formatDockBadge(count) === null ? null : Math.floor(count ?? 0);
      paintBadge(entry);
    },
    badgeOf: (tabIdOrLabel): number | null => resolve(tabIdOrLabel)?.badge ?? null,
    setCollapsed: (next): void => {
      applyCollapsed(next, true);
    },
    isCollapsed: (): boolean => collapsed,
    onTabActivated: (cb): (() => void) => {
      activatedListeners.push(cb);
      return () => {
        const i = activatedListeners.indexOf(cb);
        if (i >= 0) activatedListeners.splice(i, 1);
      };
    },
    dispose: (): void => {
      roving.dispose();
      dock.remove();
    },
  };
}
