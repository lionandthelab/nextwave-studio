// ui/shell/router.test.ts — 해시 라우터 순수 로직 + 구독 계층 (node 환경, DOM 비의존)
//
// RouterHost를 가짜로 주입해 브라우저 없이 전 경로를 검증한다:
// - 파싱 ↔ 직렬화 왕복 (전 라우트 이름 · id 인코딩)
// - 알 수 없는 해시 → tasks 정규화 (주소창 교정은 replaceHash — 히스토리 오염 없음)
// - id 비대상 라우트(settings/studio/login/setup)의 꼬리 세그먼트 폐기
// - navigate/구독 통지 (중복 통지 없음 · 해제 · dispose)

import { describe, expect, it } from 'vitest';
import {
  CONSOLE_SCREEN_NAMES,
  DEFAULT_ROUTE,
  ROUTE_NAMES,
  createHashRouter,
  isConsoleScreenName,
  parseHash,
  routeAcceptsId,
  routeToHash,
  routesEqual,
} from './router';
import type { Route, RouterHost } from './router';

// ── 가짜 RouterHost ─────────────────────────────────────────────────

interface FakeHost {
  readonly host: RouterHost;
  getHash(): string;
  /** 외부 원인(주소창 직접 편집)으로 해시가 바뀐 상황을 흉내낸다 */
  setExternal(hash: string): void;
  readonly replaceCalls: string[];
}

function makeFakeHost(initialHash: string): FakeHost {
  let hash = initialHash;
  const listeners = new Set<() => void>();
  const replaceCalls: string[] = [];
  const fire = (): void => {
    for (const listener of [...listeners]) listener();
  };
  return {
    host: {
      getHash: () => hash,
      setHash: (next) => {
        if (next === hash) return;
        hash = next;
        fire();
      },
      replaceHash: (next) => {
        replaceCalls.push(next);
        hash = next; // history.replaceState — hashchange 미발생
      },
      addHashListener: (listener) => {
        listeners.add(listener);
        return () => {
          listeners.delete(listener);
        };
      },
    },
    getHash: () => hash,
    setExternal: (next) => {
      hash = next;
      fire();
    },
    replaceCalls,
  };
}

// ── 파싱 · 직렬화 왕복 ──────────────────────────────────────────────

describe('parseHash / routeToHash 왕복', () => {
  it('전 라우트 이름이 id 없이 왕복된다', () => {
    for (const name of ROUTE_NAMES) {
      const route: Route = { name };
      expect(parseHash(routeToHash(route))).toEqual({ name });
    }
  });

  it('id 대상 라우트는 id까지 왕복된다', () => {
    for (const name of ROUTE_NAMES.filter(routeAcceptsId)) {
      const route: Route = { name, id: 'abc-123' };
      expect(parseHash(routeToHash(route))).toEqual(route);
    }
  });

  it('id는 encodeURIComponent로 직렬화되고 파싱 시 복원된다 (한글·공백·슬래시)', () => {
    const id = '작업 A/1';
    const hash = routeToHash({ name: 'runs', id });
    expect(hash).toBe(`#/runs/${encodeURIComponent(id)}`);
    expect(parseHash(hash)).toEqual({ name: 'runs', id });
  });

  it('예시 형식이 계약과 일치한다: #/tasks · #/runs/abc', () => {
    expect(routeToHash({ name: 'tasks' })).toBe('#/tasks');
    expect(routeToHash({ name: 'runs', id: 'abc' })).toBe('#/runs/abc');
    expect(parseHash('#/tasks')).toEqual({ name: 'tasks' });
    expect(parseHash('#/runs/abc')).toEqual({ name: 'runs', id: 'abc' });
  });

  it('손상된 percent-encoding은 throw 없이 원문을 id로 남긴다', () => {
    expect(parseHash('#/runs/%zz')).toEqual({ name: 'runs', id: '%zz' });
  });
});

// ── 정규화 ──────────────────────────────────────────────────────────

describe('알 수 없는 해시 정규화', () => {
  it('알 수 없는 이름·빈 해시는 전부 tasks다', () => {
    for (const raw of ['', '#', '#/', '#/nope', '#/foo/bar', '#nonsense']) {
      expect(parseHash(raw)).toEqual(DEFAULT_ROUTE);
    }
    expect(DEFAULT_ROUTE).toEqual({ name: 'tasks' });
  });

  it('id 비대상 라우트의 꼬리 세그먼트는 버린다', () => {
    expect(parseHash('#/settings/x')).toEqual({ name: 'settings' });
    expect(parseHash('#/studio/x')).toEqual({ name: 'studio' });
    expect(parseHash('#/login/x')).toEqual({ name: 'login' });
    expect(routeToHash({ name: 'settings', id: 'x' })).toBe('#/settings');
  });

  it('셋째 세그먼트 이후는 무시한다 (#/runs/a/b → id=a)', () => {
    expect(parseHash('#/runs/a/b')).toEqual({ name: 'runs', id: 'a' });
  });

  it('빈 id는 id 없음으로 정규화된다', () => {
    expect(routeToHash({ name: 'runs', id: '' })).toBe('#/runs');
    expect(parseHash('#/runs/')).toEqual({ name: 'runs' });
  });
});

describe('routesEqual', () => {
  it('이름·id가 모두 같아야 참이다', () => {
    expect(routesEqual({ name: 'tasks' }, { name: 'tasks' })).toBe(true);
    expect(routesEqual({ name: 'runs', id: 'a' }, { name: 'runs', id: 'a' })).toBe(true);
    expect(routesEqual({ name: 'runs', id: 'a' }, { name: 'runs', id: 'b' })).toBe(false);
    expect(routesEqual({ name: 'runs', id: 'a' }, { name: 'runs' })).toBe(false);
    expect(routesEqual({ name: 'tasks' }, { name: 'blocks' })).toBe(false);
  });
});

describe('isConsoleScreenName', () => {
  it('콘솔 6종만 참이다', () => {
    for (const name of CONSOLE_SCREEN_NAMES) expect(isConsoleScreenName(name)).toBe(true);
    expect(isConsoleScreenName('studio')).toBe(false);
    expect(isConsoleScreenName('login')).toBe(false);
    expect(isConsoleScreenName('setup')).toBe(false);
  });
});

// ── 라우터 구독 계층 ────────────────────────────────────────────────

describe('createHashRouter', () => {
  it('생성 시 알 수 없는 해시를 tasks로 정규화하고 주소창을 replaceHash로 교정한다', () => {
    const fake = makeFakeHost('#/bogus');
    const router = createHashRouter(fake.host);
    expect(router.current()).toEqual({ name: 'tasks' });
    expect(fake.getHash()).toBe('#/tasks');
    expect(fake.replaceCalls).toEqual(['#/tasks']);
    router.dispose();
  });

  it('navigate는 해시를 쓰고 구독자에게 1회 통지한다 (이어지는 hashchange는 중복 통지 없음)', () => {
    const fake = makeFakeHost('#/tasks');
    const router = createHashRouter(fake.host);
    const seen: Route[] = [];
    router.subscribe((r) => seen.push(r));

    router.navigate({ name: 'runs', id: 'abc' });
    expect(fake.getHash()).toBe('#/runs/abc');
    expect(seen).toEqual([{ name: 'runs', id: 'abc' }]);
    expect(router.current()).toEqual({ name: 'runs', id: 'abc' });
    router.dispose();
  });

  it('같은 라우트로의 navigate는 no-op이다', () => {
    const fake = makeFakeHost('#/tasks');
    const router = createHashRouter(fake.host);
    const seen: Route[] = [];
    router.subscribe((r) => seen.push(r));
    router.navigate({ name: 'tasks' });
    expect(seen).toEqual([]);
    router.dispose();
  });

  it('navigate에 실린 비대상 id는 정규화되어 버려진다', () => {
    const fake = makeFakeHost('#/tasks');
    const router = createHashRouter(fake.host);
    router.navigate({ name: 'settings', id: 'x' });
    expect(fake.getHash()).toBe('#/settings');
    expect(router.current()).toEqual({ name: 'settings' });
    router.dispose();
  });

  it('외부 해시 변경(주소창 편집)도 정규화 + 통지된다', () => {
    const fake = makeFakeHost('#/tasks');
    const router = createHashRouter(fake.host);
    const seen: Route[] = [];
    router.subscribe((r) => seen.push(r));

    fake.setExternal('#/devices');
    expect(seen).toEqual([{ name: 'devices' }]);

    fake.setExternal('#/garbage');
    expect(seen).toEqual([{ name: 'devices' }, { name: 'tasks' }]);
    expect(fake.getHash()).toBe('#/tasks');
    router.dispose();
  });

  it('구독 해제·dispose 후에는 통지가 없다', () => {
    const fake = makeFakeHost('#/tasks');
    const router = createHashRouter(fake.host);
    const seen: Route[] = [];
    const unsubscribe = router.subscribe((r) => seen.push(r));

    unsubscribe();
    router.navigate({ name: 'blocks' });
    expect(seen).toEqual([]);

    const seenAfter: Route[] = [];
    router.subscribe((r) => seenAfter.push(r));
    router.dispose();
    fake.setExternal('#/devices');
    expect(seenAfter).toEqual([]);
  });

  it('구독자 오류가 라우터를 깨지 않는다', () => {
    const fake = makeFakeHost('#/tasks');
    const router = createHashRouter(fake.host);
    const seen: Route[] = [];
    router.subscribe(() => {
      throw new Error('subscriber crash');
    });
    router.subscribe((r) => seen.push(r));
    router.navigate({ name: 'runs' });
    expect(seen).toEqual([{ name: 'runs' }]);
    router.dispose();
  });
});
