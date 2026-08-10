// ui/shell/shell.ts — 앱 셸: 2평면(콘솔/스튜디오) + 좌측 네비 레일 (docs/BACKEND.md Phase 12+)
//
// 콘솔 평면(공정·작업·블록·장비·기록·설정)은 스튜디오(기존 워크스페이스) DOM을
// **건드리지 않고 그 위를 덮는 전면 fixed 레이어**다. route가 'studio'면 콘텐트
// 레이어만 숨기고 레일은 얇은 오버레이로 남는다 — 어디서든 한 번의 탭으로 평면을
// 오간다. login/setup 라우트에서는 레일도 숨긴다(미인증 상태).
//
// ── 화면 계약 ───────────────────────────────────────────────────────
// deps.screens의 각 화면은 `(host) => { refresh, dispose }`로 **지연 마운트**된다 —
// 처음 방문할 때 마운트되고 이후 방문은 refresh()만 부른다. 화면 전환 시 포커스는
// 화면 제목으로 이동한다: 화면 모듈이 제목 요소에 `data-screen-title`을 달면 그 요소,
// 없으면 첫 h1/h2, 그것도 없으면 화면 호스트가 포커스를 받는다(WCAG 2.4.3).
// 전환은 createAnnouncer가 한국어로 발화한다.
//
// 계층 규칙: core/main을 모른다 — 라우터·화면 팩토리·사용자·연결 상태를 전부 주입받고
// 배선은 통합자가 한다. 전역 keydown을 걸지 않는다(라우팅은 hashchange가 소유).

import type { ConnectionState, ServerMode } from '../../api';
import type { UserInfo } from '../../schema/entities';
import { createAnnouncer } from '../a11y';
import { icon } from '../icons';
import type { IconName } from '../icons';
import {
  BORDER,
  BORDER_WIDTH,
  COLOR,
  ICON,
  MOTION,
  RADIUS,
  SPACE,
  SURFACE,
  TYPE,
  Z_INDEX,
  ensureThemeStyles,
  styled,
  tr,
} from '../theme';
import { ensureConsoleStyles, makeEmptyState } from '../console/primitives';
import { makeConnectionBadge } from './connection-badge';
import { initialOf } from './login';
import { CONSOLE_SCREEN_NAMES, isConsoleScreenName } from './router';
import type { ConsoleScreenName, Route, RouteName, RouterHandle } from './router';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4) ─────────────────────────────

/** 콘솔 레일 폭 (임무 명세 72px — 아이콘 + 한국어 라벨 세로) */
export const RAIL_WIDTH_PX = 72;
/** 스튜디오 위 얇은 오버레이 레일 폭 (아이콘 전용 — 터치 44px 이상 유지) */
export const RAIL_THIN_WIDTH_PX = 48;
/** 레일 버튼 최소 높이 — 장갑 낀 손 전제 (≥44px) */
export const NAV_BUTTON_MIN_HEIGHT_PX = 56;
/** 사용자 칩 아바타 지름 */
const CHIP_AVATAR_PX = 32;

// ── 순수 헬퍼 (DOM 비의존 — node 테스트 대상) ───────────────────────

/** 콘솔 화면 한국어 제목 — 레일 라벨·발화·플레이스홀더가 공유한다 */
export const SCREEN_TITLE_KO: Readonly<Record<ConsoleScreenName, string>> = {
  processes: '공정',
  tasks: '작업',
  blocks: '블록',
  devices: '장비',
  runs: '기록',
  settings: '설정',
};

export interface NavItem {
  readonly name: ConsoleScreenName;
  readonly labelKo: string;
  readonly iconName: IconName;
}

/** 레일 항목 — 순서 고정 (임무 명세: 공정·작업·블록·장비·기록·설정) */
export const NAV_ITEMS: readonly NavItem[] = [
  { name: 'processes', labelKo: SCREEN_TITLE_KO.processes, iconName: 'factory' },
  { name: 'tasks', labelKo: SCREEN_TITLE_KO.tasks, iconName: 'clipboard' },
  { name: 'blocks', labelKo: SCREEN_TITLE_KO.blocks, iconName: 'puzzle' },
  { name: 'devices', labelKo: SCREEN_TITLE_KO.devices, iconName: 'plug' },
  { name: 'runs', labelKo: SCREEN_TITLE_KO.runs, iconName: 'history' },
  { name: 'settings', labelKo: SCREEN_TITLE_KO.settings, iconName: 'settings' },
];

export type RailMode = 'full' | 'thin' | 'hidden';

/**
 * 라우트 → 레일 표시 모드. 콘솔=전체, 스튜디오=얇게, 인증 화면=숨김.
 *
 * **로컬 모드(서버 없음)의 스튜디오에서는 레일을 아예 감춘다.** 콘솔 6화면이 전부
 * 서버 개체를 다루므로 로컬에서는 빈 화면뿐이고, 거기로 가는 레일은 거짓 어포던스다.
 * 더 중요한 건 정적 배포 약속이다(docs/BACKEND.md §1): 서버가 없으면 앱이 **이전과
 * 똑같이** 보여야 하는데, 얇은 레일이 워크스페이스를 48px 밀어 뷰포트를 좁혔다.
 * 콘솔 라우트를 해시로 직접 열었다면 레일을 보여 돌아갈 길은 남긴다(막다른 길 금지).
 */
export function railModeForRoute(name: RouteName, serverMode: ServerMode = 'server'): RailMode {
  if (name === 'login' || name === 'setup') return 'hidden';
  if (name === 'studio') return serverMode === 'local' ? 'hidden' : 'thin';
  return 'full';
}

// ── 공개 계약 ───────────────────────────────────────────────────────

export interface ScreenHandle {
  refresh(): void;
  dispose(): void;
}

export type ScreenFactory = (host: HTMLElement) => ScreenHandle;

export interface ShellConnectionDeps {
  getState(): ConnectionState;
  /** 연결 상태 변경 구독. 반환값은 해제 함수 (ApiClient.onStateChange와 동일 형태). */
  subscribe(cb: (state: ConnectionState) => void): () => void;
}

export interface ShellDeps {
  readonly router: RouterHandle;
  /** 화면 팩토리 — 없는 화면은 "준비 중" 플레이스홀더가 뜬다 (막다른 빈 화면 금지) */
  readonly screens: Readonly<Partial<Record<ConsoleScreenName, ScreenFactory>>>;
  user(): UserInfo | null;
  readonly connection: ShellConnectionDeps;
  /** 사용자 칩 클릭 — 빠른 사용자 전환(BACKEND §3). 라우팅은 통합자 몫. */
  onSwitchUser(): void;
}

export interface ShellHandle {
  /** 사용자 칩·연결 배지·현재 화면을 다시 그린다 (로그인/전환 직후) */
  refresh(): void;
  /** outbox 동기화 대기 건수 주입 — 연결 배지에 병기된다 */
  setPendingCount(count: number): void;
  dispose(): void;
}

// ── 스타일 주입 (1회) ───────────────────────────────────────────────

const SHELL_STYLE_ID = 'rsw-shell-styles';

function ensureShellStyles(): void {
  if (document.getElementById(SHELL_STYLE_ID) !== null) return;
  const style = document.createElement('style');
  style.id = SHELL_STYLE_ID;
  style.textContent = `
.rsw-shell-rail {
  position: fixed;
  top: 0;
  bottom: 0;
  left: 0;
  width: ${RAIL_WIDTH_PX}px;
  box-sizing: border-box;
  display: flex;
  flex-direction: column;
  align-items: stretch;
  gap: ${SPACE.md};
  padding: ${SPACE.md} ${SPACE.xs};
  background: ${SURFACE.panel};
  border-right: ${BORDER_WIDTH.hair} solid ${BORDER.default};
  z-index: ${Z_INDEX.panel};
  overflow-y: auto;
  overflow-x: hidden;
  transition: ${tr('width', MOTION.base)};
}
.rsw-shell-rail--thin { width: ${RAIL_THIN_WIDTH_PX}px; }
.rsw-shell-rail__nav {
  display: flex;
  flex-direction: column;
  align-items: stretch;
  gap: ${SPACE.xs};
}
.rsw-shell-rail__spacer { flex: 1 1 auto; }
.rsw-shell-navbtn {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: ${SPACE.xxs};
  min-height: ${NAV_BUTTON_MIN_HEIGHT_PX}px;
  padding: ${SPACE.xs} ${SPACE.xxs};
  border: none;
  background: transparent;
  color: ${COLOR.muted};
  border-radius: ${RADIUS.sm};
  cursor: pointer;
  font-family: var(--rsw-font-ui);
  font-size: ${TYPE.micro.sizePx}px;
  line-height: ${TYPE.micro.lineHeightPx}px;
  font-weight: ${TYPE.micro.weight};
  transition: ${tr('color', MOTION.instant)}, ${tr('background-color', MOTION.instant)};
}
.rsw-shell-navbtn:hover {
  color: ${COLOR.textStrong};
  background: rgba(255, 255, 255, 0.06);
}
.rsw-shell-navbtn:focus-visible {
  outline: 2px solid var(--rsw-accent);
  outline-offset: -2px;
}
/* 현재 화면 — 토글 상태는 액센트 보더/틴트 축이다 (theme.ts 헤더 규약) */
.rsw-shell-navbtn--active {
  color: var(--rsw-accent-text);
  background: ${COLOR.accentSoft};
  box-shadow: inset ${BORDER_WIDTH.thick} 0 0 var(--rsw-accent);
}
.rsw-shell-navbtn__label {
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.rsw-shell-rail--thin .rsw-shell-navbtn__label { display: none; }
.rsw-shell-userchip__avatar {
  width: ${CHIP_AVATAR_PX}px;
  height: ${CHIP_AVATAR_PX}px;
  border-radius: ${RADIUS.full};
  display: grid;
  place-content: center;
  background: ${COLOR.accentSoft};
  color: ${COLOR.accentText};
  font-size: ${TYPE.bodyStrong.sizePx}px;
  font-weight: ${TYPE.bodyStrong.weight};
}
.rsw-shell-badgewrap {
  display: flex;
  justify-content: center;
  padding-bottom: ${SPACE.sm};
}
@media (forced-colors: active) {
  .rsw-shell-rail { border-right-color: CanvasText; }
  .rsw-shell-navbtn--active { background: Highlight; color: HighlightText; }
}
`;
  document.head.appendChild(style);
}

// ── 마운트 ──────────────────────────────────────────────────────────

export function mountShell(host: HTMLElement, deps: ShellDeps): ShellHandle {
  ensureThemeStyles();
  ensureConsoleStyles();
  ensureShellStyles();

  // 콘텐트 레이어 — 스튜디오 위를 덮는 전면 fixed 평면 (레일 폭만큼 비켜 선다)
  const contentLayer = styled(document.createElement('div'), {
    position: 'fixed',
    top: '0',
    right: '0',
    bottom: '0',
    left: `${RAIL_WIDTH_PX}px`,
    zIndex: Z_INDEX.panel,
    background: SURFACE.base,
    display: 'none',
    boxSizing: 'border-box',
  });
  contentLayer.dataset.testid = 'shell-content';

  // 레일 — 콘텐트보다 DOM 뒤에 두어 같은 z에서 위에 그려진다
  const rail = document.createElement('nav');
  rail.className = 'rsw-shell-rail';
  rail.setAttribute('aria-label', '콘솔 탐색');
  rail.dataset.testid = 'shell-rail';

  const navGroup = document.createElement('div');
  navGroup.className = 'rsw-shell-rail__nav';
  rail.appendChild(navGroup);

  const makeRailButton = (labelKo: string, iconName: IconName, testid: string): HTMLButtonElement => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'rsw-shell-navbtn';
    button.dataset.testid = testid;
    button.title = labelKo;
    button.setAttribute('aria-label', labelKo);
    button.appendChild(icon(iconName, ICON.lg));
    const label = document.createElement('span');
    label.className = 'rsw-shell-navbtn__label';
    label.textContent = labelKo;
    button.appendChild(label);
    return button;
  };

  const navButtons = new Map<ConsoleScreenName, HTMLButtonElement>();
  for (const item of NAV_ITEMS) {
    const button = makeRailButton(item.labelKo, item.iconName, `shell-nav-${item.name}`);
    button.addEventListener('click', () => deps.router.navigate({ name: item.name }));
    navButtons.set(item.name, button);
    navGroup.appendChild(button);
  }

  const spacer = document.createElement('div');
  spacer.className = 'rsw-shell-rail__spacer';
  rail.appendChild(spacer);

  const bottomGroup = document.createElement('div');
  bottomGroup.className = 'rsw-shell-rail__nav';
  rail.appendChild(bottomGroup);

  // 스튜디오 복귀
  const studioButton = makeRailButton('스튜디오', 'home', 'shell-nav-studio');
  studioButton.addEventListener('click', () => deps.router.navigate({ name: 'studio' }));
  bottomGroup.appendChild(studioButton);

  // 사용자 칩 — 클릭 = 빠른 사용자 전환
  const userChip = document.createElement('button');
  userChip.type = 'button';
  userChip.className = 'rsw-shell-navbtn';
  userChip.dataset.testid = 'shell-user-chip';
  const chipAvatar = document.createElement('span');
  chipAvatar.className = 'rsw-shell-userchip__avatar';
  chipAvatar.setAttribute('aria-hidden', 'true');
  userChip.appendChild(chipAvatar);
  const chipLabel = document.createElement('span');
  chipLabel.className = 'rsw-shell-navbtn__label';
  userChip.appendChild(chipLabel);
  userChip.addEventListener('click', () => deps.onSwitchUser());
  bottomGroup.appendChild(userChip);

  const syncUser = (): void => {
    const user = deps.user();
    chipAvatar.textContent = user === null ? '·' : initialOf(user.name);
    chipLabel.textContent = user === null ? '로그인' : user.name;
    const title = user === null ? '로그인' : `사용자 전환 — 현재 ${user.name}`;
    userChip.title = title;
    userChip.setAttribute('aria-label', title);
  };
  syncUser();

  // 연결 배지 — 상태는 항상 보인다 (BACKEND §6)
  const badgeWrap = document.createElement('div');
  badgeWrap.className = 'rsw-shell-badgewrap';
  const badge = makeConnectionBadge(deps.connection.getState(), { testid: 'shell-connection' });
  badgeWrap.appendChild(badge.el);
  bottomGroup.appendChild(badgeWrap);

  host.appendChild(contentLayer);
  host.appendChild(rail);

  const announcer = createAnnouncer(document.body);

  // ── 화면 지연 마운트 ──────────────────────────────────────────────

  interface MountedScreen {
    readonly host: HTMLElement;
    readonly handle: ScreenHandle;
  }
  const mounted = new Map<ConsoleScreenName, MountedScreen>();

  const mountPlaceholder = (screenHost: HTMLElement, name: ConsoleScreenName): ScreenHandle => {
    const title = document.createElement('h1');
    title.className = 'sr-only';
    title.dataset.screenTitle = 'true';
    title.tabIndex = -1;
    title.textContent = SCREEN_TITLE_KO[name];
    screenHost.appendChild(title);
    screenHost.appendChild(
      makeEmptyState({
        iconName: 'alert',
        titleKo: `${SCREEN_TITLE_KO[name]} 화면 준비 중`,
        hintKo: '이 화면은 아직 연결되지 않았습니다.',
        actions: [],
        testid: `shell-placeholder-${name}`,
      }),
    );
    return {
      refresh: (): void => {
        /* 플레이스홀더 — 갱신할 것이 없다 */
      },
      dispose: (): void => {
        screenHost.textContent = '';
      },
    };
  };

  const mountScreen = (name: ConsoleScreenName): MountedScreen => {
    const screenHost = document.createElement('div');
    screenHost.className = 'ui-scroll';
    styled(screenHost, { height: '100%', overflow: 'auto', boxSizing: 'border-box' });
    screenHost.dataset.testid = `shell-screen-${name}`;
    contentLayer.appendChild(screenHost);
    const factory = deps.screens[name];
    const handle = factory !== undefined ? factory(screenHost) : mountPlaceholder(screenHost, name);
    const entry: MountedScreen = { host: screenHost, handle };
    mounted.set(name, entry);
    return entry;
  };

  /** 화면 제목으로 포커스 이동 (data-screen-title → h1/h2 → 호스트 — WCAG 2.4.3) */
  const focusScreenTitle = (screenHost: HTMLElement): void => {
    const target = screenHost.querySelector<HTMLElement>('[data-screen-title], h1, h2') ?? screenHost;
    if (!target.hasAttribute('tabindex')) target.tabIndex = -1;
    target.focus();
  };

  // ── 라우트 반영 ───────────────────────────────────────────────────

  let firstApply = true;

  const applyRoute = (route: Route): void => {
    const mode = railModeForRoute(route.name, deps.connection.getState().mode);
    rail.style.display = mode === 'hidden' ? 'none' : 'flex';
    rail.classList.toggle('rsw-shell-rail--thin', mode === 'thin');
    // 레일은 두 모드 모두 좁다(72px / 48px) — 전체 라벨('서버 연결됨')은 어느 쪽에도
    // 들어가지 않아 넘치고 잘린다. 배지는 항상 compact(아이콘 + 대기 건수)이고 전체
    // 문구는 title/aria-label로 남는다.
    badge.setCompact(true);

    const onConsole = isConsoleScreenName(route.name);
    contentLayer.style.display = onConsole ? '' : 'none';

    for (const [name, button] of navButtons) {
      const active = onConsole && name === route.name;
      button.classList.toggle('rsw-shell-navbtn--active', active);
      if (active) button.setAttribute('aria-current', 'page');
      else button.removeAttribute('aria-current');
    }
    const onStudio = route.name === 'studio';
    studioButton.classList.toggle('rsw-shell-navbtn--active', onStudio);
    if (onStudio) studioButton.setAttribute('aria-current', 'page');
    else studioButton.removeAttribute('aria-current');

    if (onConsole) {
      const name = route.name;
      let entry = mounted.get(name);
      if (entry === undefined) {
        entry = mountScreen(name);
      } else {
        entry.handle.refresh();
      }
      for (const [n, e] of mounted) {
        e.host.style.display = n === name ? '' : 'none';
      }
      if (!firstApply) {
        announcer.announce(`${SCREEN_TITLE_KO[name]} 화면으로 이동했습니다`);
        focusScreenTitle(entry.host);
      }
    }
    firstApply = false;
  };

  const unsubscribeRoute = deps.router.subscribe(applyRoute);
  applyRoute(deps.router.current());

  const unsubscribeConnection = deps.connection.subscribe((state) => {
    badge.setState(state);
    // 서버가 늦게 나타나거나(로컬 → 서버) 사라지면 레일 표시가 따라와야 한다 —
    // 모드가 레일 가시성을 결정하므로 라우트를 다시 적용한다.
    applyRoute(deps.router.current());
  });

  return {
    refresh: (): void => {
      syncUser();
      badge.setState(deps.connection.getState());
      const route = deps.router.current();
      if (isConsoleScreenName(route.name)) mounted.get(route.name)?.handle.refresh();
    },
    setPendingCount: (count: number): void => {
      badge.setPendingCount(count);
    },
    dispose: (): void => {
      unsubscribeRoute();
      unsubscribeConnection();
      for (const entry of mounted.values()) entry.handle.dispose();
      mounted.clear();
      announcer.dispose();
      badge.dispose();
      rail.remove();
      contentLayer.remove();
    },
  };
}

// 재노출 — 통합자가 라우터와 셸을 한 배럴에서 배선할 수 있게 한다
export { CONSOLE_SCREEN_NAMES };
