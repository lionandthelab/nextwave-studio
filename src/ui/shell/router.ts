// ui/shell/router.ts — 콘솔 평면 해시 라우터 (docs/BACKEND.md Phase 12+)
//
// **의존성 0** — 이 파일은 아무것도 import하지 않는다. 순수 파싱/직렬화 함수와
// 얇은 구독 계층만 있어 node 환경 테스트가 브라우저 없이 전 경로를 검증한다
// (router.test.ts — RouterHost를 가짜로 주입).
//
// ── 계약 ────────────────────────────────────────────────────────────
// - 해시 형식: `#/name` 또는 `#/name/id` (id는 encodeURIComponent).
// - 알 수 없는 해시는 **tasks로 정규화**한다 — 죽은 링크가 빈 화면을 만들지 않는다.
//   정규화는 replaceHash(히스토리 오염 없음)로 주소창까지 교정한다.
// - id를 받는 라우트는 개체 화면 5종뿐이다(tasks/processes/blocks/devices/runs).
//   settings·studio·login·setup 뒤에 붙은 세그먼트는 조용히 버린다.
// - 전역 keydown을 걸지 않는다. hashchange 리스너가 이 모듈의 유일한 전역 구독이며
//   dispose()로 해제된다.

// ── 라우트 이름 ─────────────────────────────────────────────────────

/** 콘솔 평면 화면 6종 — shell의 네비 레일과 1:1 */
export const CONSOLE_SCREEN_NAMES = [
  'tasks',
  'processes',
  'blocks',
  'devices',
  'runs',
  'settings',
] as const;
export type ConsoleScreenName = (typeof CONSOLE_SCREEN_NAMES)[number];

/** 전체 라우트 — 콘솔 6종 + 스튜디오 복귀 + 인증 2종 */
export const ROUTE_NAMES = [...CONSOLE_SCREEN_NAMES, 'studio', 'login', 'setup'] as const;
export type RouteName = (typeof ROUTE_NAMES)[number];

export interface Route {
  readonly name: RouteName;
  /** 개체 상세(예: '#/runs/abc') — routeAcceptsId가 true인 라우트만 갖는다 */
  readonly id?: string;
}

/** 알 수 없는 해시의 정규화 목적지 — 설치기사의 홈은 작업 목록이다 */
export const DEFAULT_ROUTE: Route = { name: 'tasks' };

const ID_CAPABLE: ReadonlySet<RouteName> = new Set<RouteName>([
  'tasks',
  'processes',
  'blocks',
  'devices',
  'runs',
]);

export function isRouteName(value: string): value is RouteName {
  return (ROUTE_NAMES as readonly string[]).includes(value);
}

export function isConsoleScreenName(name: RouteName): name is ConsoleScreenName {
  return (CONSOLE_SCREEN_NAMES as readonly string[]).includes(name);
}

/** 이 라우트가 `/:id` 세그먼트를 갖는가 (개체 화면 5종만) */
export function routeAcceptsId(name: RouteName): boolean {
  return ID_CAPABLE.has(name);
}

// ── 순수 파싱 / 직렬화 ──────────────────────────────────────────────

/** 손상된 percent-encoding은 원문 그대로 둔다 — throw가 라우터를 죽이면 안 된다 */
function safeDecode(segment: string): string {
  try {
    return decodeURIComponent(segment);
  } catch {
    return segment;
  }
}

/**
 * 해시 문자열 → Route. 절대 throw하지 않는다.
 * 알 수 없는 이름·빈 해시는 DEFAULT_ROUTE, id 비대상 라우트의 꼬리는 버린다.
 */
export function parseHash(rawHash: string): Route {
  const raw = rawHash.startsWith('#') ? rawHash.slice(1) : rawHash;
  const segments = raw.split('/').filter((s) => s !== '');
  const nameSegment = segments[0];
  if (nameSegment === undefined || !isRouteName(nameSegment)) return DEFAULT_ROUTE;
  const idSegment = segments[1];
  if (idSegment === undefined || !routeAcceptsId(nameSegment)) {
    return { name: nameSegment };
  }
  return { name: nameSegment, id: safeDecode(idSegment) };
}

/** Route → 해시 문자열. id는 인코딩되고, id 비대상 라우트의 id는 버려진다. */
export function routeToHash(route: Route): string {
  const id = route.id;
  if (id !== undefined && id !== '' && routeAcceptsId(route.name)) {
    return `#/${route.name}/${encodeURIComponent(id)}`;
  }
  return `#/${route.name}`;
}

export function routesEqual(a: Route, b: Route): boolean {
  return a.name === b.name && (a.id ?? null) === (b.id ?? null);
}

// ── 라우터 (얇은 구독 계층 — 브라우저 결합은 RouterHost로 격리) ─────

/** window 결합 지점 — 테스트는 가짜 구현을 주입한다 */
export interface RouterHost {
  getHash(): string;
  /** 히스토리 항목을 만드는 이동 (뒤로가기 가능) */
  setHash(hash: string): void;
  /** 히스토리 항목 없는 교정 — hashchange를 발생시키지 않아야 한다 */
  replaceHash(hash: string): void;
  /** hashchange 구독. 반환값은 해제 함수. */
  addHashListener(listener: () => void): () => void;
}

export function browserRouterHost(win: Window = window): RouterHost {
  return {
    getHash: () => win.location.hash,
    setHash: (hash) => {
      win.location.hash = hash;
    },
    replaceHash: (hash) => {
      win.history.replaceState(null, '', hash);
    },
    addHashListener: (listener) => {
      win.addEventListener('hashchange', listener);
      return () => {
        win.removeEventListener('hashchange', listener);
      };
    },
  };
}

export interface RouterHandle {
  current(): Route;
  /** 라우트 이동 — 같은 라우트면 no-op. 구독자에게 동기 통지된다. */
  navigate(route: Route): void;
  /** 라우트 변경 구독. 반환값은 해제 함수. 현재 라우트는 current()로 읽는다. */
  subscribe(cb: (route: Route) => void): () => void;
  dispose(): void;
}

/**
 * 해시 라우터를 만든다. 생성 즉시 현재 해시를 읽어 정규화하고 hashchange를 구독한다.
 *
 * navigate는 브라우저 hashchange 이벤트(비동기)를 기다리지 않고 **즉시** 상태를
 * 전이·통지한다 — 이어지는 이벤트는 routesEqual 검사로 자연 무시된다(중복 통지 없음).
 */
export function createHashRouter(host: RouterHost = browserRouterHost()): RouterHandle {
  const listeners = new Set<(route: Route) => void>();

  const normalizeNow = (): Route => {
    const raw = host.getHash();
    const parsed = parseHash(raw);
    const canonical = routeToHash(parsed);
    if (raw !== canonical) host.replaceHash(canonical);
    return parsed;
  };

  let route: Route = normalizeNow();

  const apply = (next: Route): void => {
    if (routesEqual(route, next)) return;
    route = next;
    for (const cb of listeners) {
      try {
        cb(next);
      } catch {
        // 구독자 오류가 라우터를 깨지 않게 한다 (api/client.ts와 같은 규약)
      }
    }
  };

  const removeHashListener = host.addHashListener(() => {
    apply(normalizeNow());
  });

  return {
    current: () => route,
    navigate: (next: Route): void => {
      // 라운드트립으로 정규화 — settings에 id를 실어 보내는 실수를 여기서 흡수한다
      const canonical = parseHash(routeToHash(next));
      if (routesEqual(route, canonical)) return;
      host.setHash(routeToHash(canonical));
      apply(canonical);
    },
    subscribe: (cb: (route: Route) => void): (() => void) => {
      listeners.add(cb);
      return () => {
        listeners.delete(cb);
      };
    },
    dispose: (): void => {
      removeHashListener();
      listeners.clear();
    },
  };
}
