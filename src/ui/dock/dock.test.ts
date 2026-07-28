// ui/dock/dock.test.ts — 하단 독 셸 계약 테스트 (Phase 11 — C-7 / C-1 / C-18)
//
// 검증 대상:
//   · 탭 ARIA (WAI-ARIA APG Tabs): tablist/tab/tabpanel + aria-selected/controls/labelledby
//   · 카운트 배지 (C-7): 비활성 탭에만 표시, 활성화 시 리셋, 상한 접기
//   · 접기 콜백 (C-1): onCollapseChange가 워크스페이스에 픽셀 반환을 위임한다
//
// ── 왜 DOM 셰임인가 ─────────────────────────────────────────────────
// vitest 환경은 `node`이고 jsdom/happy-dom이 의존성에 없다(vitest.config.ts). 독은
// "DOM 속성을 어떻게 세팅하는가"가 곧 계약이라 순수 함수 추출만으로는 ARIA를 검증할 수
// 없다. 그래서 이 파일 안에 **이 테스트가 실제로 쓰는 표면만** 갖는 최소 셰임을 둔다.
// 셰임은 테스트 로컬이며 프로덕션 코드는 전혀 알지 못한다.

import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { DOCK_TAB_ID, dockTabSlug, formatDockBadge, mountDock } from './dock';

// ── 최소 DOM 셰임 ───────────────────────────────────────────────────

interface FakeEvent {
  readonly type: string;
  readonly target: FakeElement;
  readonly key?: string;
  readonly ctrlKey?: boolean;
  readonly metaKey?: boolean;
  readonly altKey?: boolean;
  readonly shiftKey?: boolean;
  preventDefault(): void;
  stopPropagation(): void;
}

type Listener = (e: FakeEvent) => void;

class FakeClassList {
  private readonly names = new Set<string>();
  contains(name: string): boolean {
    return this.names.has(name);
  }
  add(...list: string[]): void {
    for (const n of list) if (n !== '') this.names.add(n);
  }
  remove(...list: string[]): void {
    for (const n of list) this.names.delete(n);
  }
  toggle(name: string, force?: boolean): boolean {
    const on = force ?? !this.names.has(name);
    if (on) this.names.add(name);
    else this.names.delete(name);
    return on;
  }
  reset(value: string): void {
    this.names.clear();
    for (const n of value.split(/\s+/)) if (n !== '') this.names.add(n);
  }
  toString(): string {
    return [...this.names].join(' ');
  }
}

class FakeElement {
  readonly style: Record<string, string> = {};
  readonly dataset: Record<string, string> = {};
  readonly children: FakeElement[] = [];
  readonly classList = new FakeClassList();
  readonly attributes = new Map<string, string>();
  parentNode: FakeElement | null = null;
  tabIndex = 0;
  id = '';
  type = '';
  title = '';
  disabled = false;
  private ownText = '';
  private readonly listeners = new Map<string, Listener[]>();

  constructor(readonly tagName: string) {}

  get className(): string {
    return this.classList.toString();
  }
  set className(value: string) {
    this.classList.reset(value);
  }

  get textContent(): string {
    return this.ownText + this.children.map((c) => c.textContent).join('');
  }
  set textContent(value: string) {
    this.children.splice(0, this.children.length);
    this.ownText = value;
  }

  get firstElementChild(): FakeElement | null {
    return this.children[0] ?? null;
  }
  get childElementCount(): number {
    return this.children.length;
  }

  appendChild(child: FakeElement): FakeElement {
    child.remove();
    child.parentNode = this;
    this.children.push(child);
    return child;
  }
  insertBefore(child: FakeElement, ref: FakeElement | null): FakeElement {
    child.remove();
    child.parentNode = this;
    const i = ref === null ? -1 : this.children.indexOf(ref);
    if (i < 0) this.children.push(child);
    else this.children.splice(i, 0, child);
    return child;
  }
  replaceChildren(...next: FakeElement[]): void {
    for (const c of [...this.children]) c.remove();
    for (const c of next) this.appendChild(c);
  }
  remove(): void {
    const parent = this.parentNode;
    if (parent === null) return;
    const i = parent.children.indexOf(this);
    if (i >= 0) parent.children.splice(i, 1);
    this.parentNode = null;
  }

  setAttribute(name: string, value: string): void {
    this.attributes.set(name, value);
  }
  getAttribute(name: string): string | null {
    return this.attributes.get(name) ?? null;
  }
  removeAttribute(name: string): void {
    this.attributes.delete(name);
  }
  hasAttribute(name: string): boolean {
    return this.attributes.has(name);
  }

  addEventListener(type: string, fn: Listener): void {
    const list = this.listeners.get(type);
    if (list === undefined) this.listeners.set(type, [fn]);
    else list.push(fn);
  }
  removeEventListener(type: string, fn: Listener): void {
    const list = this.listeners.get(type);
    if (list === undefined) return;
    const i = list.indexOf(fn);
    if (i >= 0) list.splice(i, 1);
  }

  /** 타깃 → 조상 순으로 전파 (stopPropagation 존중) */
  dispatch(type: string, init: { key?: string } = {}): void {
    let stopped = false;
    const event: FakeEvent = {
      type,
      target: this,
      ...init,
      preventDefault: () => undefined,
      stopPropagation: () => {
        stopped = true;
      },
    };
    const chain: FakeElement[] = [];
    chain.push(this);
    let node = this.parentNode;
    while (node !== null) {
      chain.push(node);
      node = node.parentNode;
    }
    for (const current of chain) {
      if (stopped) break;
      for (const fn of [...(current.listeners.get(type) ?? [])]) fn(event);
    }
  }

  focus(): void {
    fakeDocument.activeElement = this;
    this.dispatch('focusin');
  }

  /** 하위 트리에서 data-testid로 찾기 (테스트 편의) */
  find(testId: string): FakeElement | null {
    if (this.dataset.testid === testId) return this;
    for (const c of this.children) {
      const hit = c.find(testId);
      if (hit !== null) return hit;
    }
    return null;
  }
}

const created: FakeElement[] = [];
const fakeDocument = {
  head: new FakeElement('head'),
  body: new FakeElement('body'),
  activeElement: null as FakeElement | null,
  createElement(tag: string): FakeElement {
    const el = new FakeElement(tag);
    created.push(el);
    return el;
  },
  createElementNS(_ns: string, tag: string): FakeElement {
    return fakeDocument.createElement(tag);
  },
  getElementById(id: string): FakeElement | null {
    return created.find((el) => el.id === id) ?? null;
  },
};

beforeAll(() => {
  (globalThis as { document?: unknown }).document = fakeDocument;
});

// ── 테스트 헬퍼 ─────────────────────────────────────────────────────

const asHost = (el: FakeElement): HTMLElement => el as unknown as HTMLElement;

interface Rig {
  readonly host: FakeElement;
  readonly panels: Record<'timeline' | 'collision' | 'console', FakeElement>;
  readonly dock: ReturnType<typeof mountDock>;
  readonly root: FakeElement;
}

const mounted: Rig[] = [];

function makeRig(opts: Parameters<typeof mountDock>[2] = {}): Rig {
  const host = new FakeElement('div');
  const panels = {
    timeline: new FakeElement('div'),
    collision: new FakeElement('div'),
    console: new FakeElement('div'),
  };
  const dock = mountDock(
    asHost(host),
    [
      { id: DOCK_TAB_ID.timeline, label: '타임라인', content: asHost(panels.timeline) },
      { id: DOCK_TAB_ID.collision, label: '충돌 로그', content: asHost(panels.collision) },
      { id: DOCK_TAB_ID.console, label: '콘솔', content: asHost(panels.console) },
    ],
    opts,
  );
  const rig: Rig = { host, panels, dock, root: dock.el as unknown as FakeElement };
  mounted.push(rig);
  return rig;
}

function tabButton(rig: Rig, id: string): FakeElement {
  const el = rig.root.find(`dock-tab-${id}`);
  if (el === null) throw new Error(`탭 버튼을 찾을 수 없습니다: ${id}`);
  return el;
}

function dockBody(rig: Rig): FakeElement {
  // dock = [tabBar, body]
  const body = rig.root.children[1];
  if (body === undefined) throw new Error('독 본문을 찾을 수 없습니다');
  return body;
}

afterEach(() => {
  for (const rig of mounted.splice(0)) rig.dock.dispose();
});

// ── 순수 헬퍼 ───────────────────────────────────────────────────────

describe('formatDockBadge', () => {
  it('0 이하·null·비유한수는 배지 없음(null)', () => {
    expect(formatDockBadge(null)).toBeNull();
    expect(formatDockBadge(undefined)).toBeNull();
    expect(formatDockBadge(0)).toBeNull();
    expect(formatDockBadge(-3)).toBeNull();
    expect(formatDockBadge(Number.NaN)).toBeNull();
  });

  it('양수는 문자열, 상한 초과는 99+로 접는다 (탭 폭 폭주 방지)', () => {
    expect(formatDockBadge(1)).toBe('1');
    expect(formatDockBadge(99)).toBe('99');
    expect(formatDockBadge(100)).toBe('99+');
    expect(formatDockBadge(4820)).toBe('99+');
    expect(formatDockBadge(3.7)).toBe('3');
  });
});

describe('dockTabSlug', () => {
  it('공백을 하이픈으로 접고 소문자화한다', () => {
    expect(dockTabSlug('Collision Log')).toBe('collision-log');
    expect(dockTabSlug('  Console ')).toBe('console');
  });
});

// ── 탭 ARIA (C-18) ──────────────────────────────────────────────────

describe('독 탭 ARIA (WAI-ARIA APG Tabs)', () => {
  it('tablist / tab / tabpanel role이 전부 선언된다', () => {
    const rig = makeRig();
    const tabList = tabButton(rig, DOCK_TAB_ID.timeline).parentNode;
    expect(tabList?.getAttribute('role')).toBe('tablist');
    expect(tabList?.getAttribute('aria-orientation')).toBe('horizontal');
    for (const id of Object.values(DOCK_TAB_ID)) {
      expect(tabButton(rig, id).getAttribute('role')).toBe('tab');
    }
    for (const panel of Object.values(rig.panels)) {
      expect(panel.getAttribute('role')).toBe('tabpanel');
      expect(panel.tabIndex).toBe(0);
    }
  });

  it('tab ↔ tabpanel이 aria-controls / aria-labelledby로 상호 참조된다', () => {
    const rig = makeRig();
    const button = tabButton(rig, DOCK_TAB_ID.collision);
    const panel = rig.panels.collision;
    expect(button.getAttribute('aria-controls')).toBe(panel.id);
    expect(panel.getAttribute('aria-labelledby')).toBe(button.id);
    expect(button.id).not.toBe('');
    expect(panel.id).not.toBe('');
  });

  it('접기 버튼은 tablist 밖에 있다 (tablist 자식은 탭만)', () => {
    const rig = makeRig();
    const tabList = tabButton(rig, DOCK_TAB_ID.timeline).parentNode;
    const collapse = rig.root.find('dock-collapse');
    expect(collapse).not.toBeNull();
    expect(collapse?.parentNode).not.toBe(tabList);
    expect(tabList?.children.every((c) => c.getAttribute('role') === 'tab')).toBe(true);
  });

  it('활성 탭만 aria-selected=true이고, 클릭하면 따라 옮겨간다', () => {
    const rig = makeRig();
    expect(tabButton(rig, DOCK_TAB_ID.timeline).getAttribute('aria-selected')).toBe('true');
    expect(tabButton(rig, DOCK_TAB_ID.collision).getAttribute('aria-selected')).toBe('false');

    tabButton(rig, DOCK_TAB_ID.collision).dispatch('click');

    expect(rig.dock.activeTabId()).toBe(DOCK_TAB_ID.collision);
    expect(tabButton(rig, DOCK_TAB_ID.timeline).getAttribute('aria-selected')).toBe('false');
    expect(tabButton(rig, DOCK_TAB_ID.collision).getAttribute('aria-selected')).toBe('true');
    expect(rig.panels.timeline.style.display).toBe('none');
    expect(rig.panels.collision.style.display).toBe('');
  });

  it('roving tabindex — 활성 탭만 tabIndex 0, ←→로 이동 후 Enter로 활성화', () => {
    const rig = makeRig();
    const timeline = tabButton(rig, DOCK_TAB_ID.timeline);
    const collision = tabButton(rig, DOCK_TAB_ID.collision);
    expect(timeline.tabIndex).toBe(0);
    expect(collision.tabIndex).toBe(-1);

    timeline.dispatch('keydown', { key: 'ArrowRight' });
    expect(collision.tabIndex).toBe(0);
    expect(timeline.tabIndex).toBe(-1);
    // 수동 활성화 — 포커스 이동만으로 탭이 바뀌지 않는다
    expect(rig.dock.activeTabId()).toBe(DOCK_TAB_ID.timeline);

    collision.dispatch('keydown', { key: 'Enter' });
    expect(rig.dock.activeTabId()).toBe(DOCK_TAB_ID.collision);
  });
});

// ── 카운트 배지 (C-7) ───────────────────────────────────────────────

describe('탭 카운트 배지', () => {
  it('비활성 탭에 배지를 렌더한다 (.ui-tab__badge + 스크린리더 문장)', () => {
    const rig = makeRig();
    rig.dock.setBadge(DOCK_TAB_ID.collision, 3);

    const badge = rig.root.find(`dock-badge-${DOCK_TAB_ID.collision}`);
    expect(badge).not.toBeNull();
    expect(badge?.classList.contains('ui-tab__badge')).toBe(true);
    expect(badge?.textContent).toBe('3');
    // 숫자만으로는 "무엇이 3건인지" 전달되지 않는다 — sr-only 문장이 병행돼야 한다
    expect(tabButton(rig, DOCK_TAB_ID.collision).textContent).toContain('새 항목 3건');
    expect(rig.dock.badgeOf(DOCK_TAB_ID.collision)).toBe(3);
  });

  it('활성 탭에는 배지를 그리지 않는다 (펼쳐진 상태에서는 의미가 없다)', () => {
    const rig = makeRig();
    rig.dock.setBadge(DOCK_TAB_ID.timeline, 5); // timeline이 기본 활성
    expect(rig.root.find(`dock-badge-${DOCK_TAB_ID.timeline}`)).toBeNull();
  });

  it('접힌 상태에서는 활성 탭에도 배지가 보인다 (콘텐츠가 안 보이므로)', () => {
    const rig = makeRig();
    rig.dock.setCollapsed(true);
    rig.dock.setBadge(DOCK_TAB_ID.timeline, 2);
    expect(rig.root.find(`dock-badge-${DOCK_TAB_ID.timeline}`)?.textContent).toBe('2');
  });

  it('탭을 활성화하면 배지가 0으로 리셋된다', () => {
    const rig = makeRig();
    rig.dock.setBadge(DOCK_TAB_ID.collision, 7);
    tabButton(rig, DOCK_TAB_ID.collision).dispatch('click');
    expect(rig.dock.badgeOf(DOCK_TAB_ID.collision)).toBeNull();
    expect(rig.root.find(`dock-badge-${DOCK_TAB_ID.collision}`)).toBeNull();
  });

  it('0/null 배지는 요소 자체를 제거한다', () => {
    const rig = makeRig();
    rig.dock.setBadge(DOCK_TAB_ID.console, 4);
    expect(rig.root.find(`dock-badge-${DOCK_TAB_ID.console}`)).not.toBeNull();
    rig.dock.setBadge(DOCK_TAB_ID.console, 0);
    expect(rig.root.find(`dock-badge-${DOCK_TAB_ID.console}`)).toBeNull();
    rig.dock.setBadge(DOCK_TAB_ID.console, 4);
    rig.dock.setBadge(DOCK_TAB_ID.console, null);
    expect(rig.root.find(`dock-badge-${DOCK_TAB_ID.console}`)).toBeNull();
  });

  it('라벨로도 대상 탭을 지정할 수 있다 (id 미지정 통합 경로 호환)', () => {
    const rig = makeRig();
    rig.dock.setBadge('충돌 로그', 9);
    expect(rig.dock.badgeOf(DOCK_TAB_ID.collision)).toBe(9);
  });
});

// ── 접기 콜백 (C-1) ─────────────────────────────────────────────────

describe('독 접기', () => {
  it('접기 버튼이 onCollapseChange를 호출한다 (워크스페이스가 슬롯 높이를 줄인다)', () => {
    const onCollapseChange = vi.fn();
    const rig = makeRig({ onCollapseChange });
    const collapse = rig.root.find('dock-collapse');
    expect(collapse).not.toBeNull();

    collapse?.dispatch('click');
    expect(onCollapseChange).toHaveBeenCalledTimes(1);
    expect(onCollapseChange).toHaveBeenLastCalledWith(true);
    expect(rig.dock.isCollapsed()).toBe(true);
    expect(dockBody(rig).style.height).toBe('0px');
    expect(collapse?.getAttribute('aria-expanded')).toBe('false');
    // 접힌 본문은 탭 순서/보조기술에서 빠진다
    expect(dockBody(rig).hasAttribute('inert')).toBe(true);

    collapse?.dispatch('click');
    expect(onCollapseChange).toHaveBeenCalledTimes(2);
    expect(onCollapseChange).toHaveBeenLastCalledWith(false);
    expect(rig.dock.isCollapsed()).toBe(false);
    expect(dockBody(rig).hasAttribute('inert')).toBe(false);
  });

  it('setCollapsed로 외부에서 제어할 수 있고, 같은 값은 통지하지 않는다', () => {
    const onCollapseChange = vi.fn();
    const rig = makeRig({ onCollapseChange });
    rig.dock.setCollapsed(true);
    rig.dock.setCollapsed(true);
    expect(onCollapseChange).toHaveBeenCalledTimes(1);
    expect(rig.dock.isCollapsed()).toBe(true);
  });

  it('접힌 상태에서 탭 클릭 = 펼치기 (워크스페이스에도 전파된다)', () => {
    const onCollapseChange = vi.fn();
    const rig = makeRig({ initialCollapsed: true, onCollapseChange });
    expect(rig.dock.isCollapsed()).toBe(true);

    tabButton(rig, DOCK_TAB_ID.console).dispatch('click');
    expect(rig.dock.isCollapsed()).toBe(false);
    expect(onCollapseChange).toHaveBeenLastCalledWith(false);
    expect(rig.dock.activeTabId()).toBe(DOCK_TAB_ID.console);
  });

  it('인라인 스트립은 탭바 안에 있다 — 접혀도 리드아웃이 남는다', () => {
    const strip = new FakeElement('div');
    strip.dataset.testid = 'timeline-strip';
    const rig = makeRig({ strip: asHost(strip) });
    expect(strip.parentNode).toBe(rig.dock.stripEl as unknown as FakeElement);
    // 스트립은 본문(body)이 아니라 탭바에 있으므로 접혀도 사라지지 않는다
    rig.dock.setCollapsed(true);
    expect(dockBody(rig).find('timeline-strip')).toBeNull();
    expect(rig.root.find('timeline-strip')).not.toBeNull();
  });
});

// ── 탭 활성화 훅 (C-7 — 첫 충돌 토스트 트리거) ──────────────────────

describe('onTabActivated', () => {
  it('옵션 콜백과 구독 콜백 모두 tabId + 직전 배지 카운트를 받는다', () => {
    const optionCb = vi.fn();
    const rig = makeRig({ onTabActivated: optionCb });
    const subscribed = vi.fn();
    const off = rig.dock.onTabActivated(subscribed);

    rig.dock.setBadge(DOCK_TAB_ID.collision, 12);
    rig.dock.activateTab(DOCK_TAB_ID.collision);

    expect(optionCb).toHaveBeenLastCalledWith(DOCK_TAB_ID.collision, 12);
    expect(subscribed).toHaveBeenLastCalledWith(DOCK_TAB_ID.collision, 12);

    off();
    rig.dock.activateTab(DOCK_TAB_ID.console);
    expect(subscribed).toHaveBeenCalledTimes(1);
    expect(optionCb).toHaveBeenCalledTimes(2);
  });

  it('없는 탭은 no-op이다', () => {
    const optionCb = vi.fn();
    const rig = makeRig({ onTabActivated: optionCb });
    rig.dock.activateTab('없는-탭');
    expect(optionCb).not.toHaveBeenCalled();
    expect(rig.dock.activeTabId()).toBe(DOCK_TAB_ID.timeline);
  });
});
