// @vitest-environment jsdom
//
// ui/shortcuts.test.ts — 단축키 라우터 순수 로직 (DOM 최소 — jsdom)
//
// UX_AUDIT C-6: window에 keydown을 거는 곳이 5군데였고 Space를 3개가, 방향키를 2개가
// 나눠 가졌다. 여기서는 라우터의 **해석 규칙**을 고정한다 — 특히 "위젯이 키의 주인이면
// 라우터가 가로채지 않는다"는 계약(구 구현의 핵심 버그)을.

import { describe, expect, it, vi } from 'vitest';
import {
  SCOPE_ATTR,
  createShortcutRouter,
  formatKeys,
  keyHint,
  normalizeKey,
  resolveScope,
  widgetOwnsKey,
} from './shortcuts';
import type { ShortcutBinding } from './shortcuts';

// ── 키 정규화 ───────────────────────────────────────────────────────

describe('normalizeKey', () => {
  const base = { ctrlKey: false, metaKey: false, altKey: false, shiftKey: false };

  it('단일 문자는 대문자로 정규화된다', () => {
    expect(normalizeKey({ ...base, key: 'z' })).toBe('Z');
    expect(normalizeKey({ ...base, key: 'Z' })).toBe('Z');
  });

  it('스페이스는 code로도 key로도 잡힌다', () => {
    expect(normalizeKey({ ...base, key: ' ' })).toBe('Space');
    expect(normalizeKey({ ...base, key: 'Unidentified', code: 'Space' })).toBe('Space');
  });

  it('Ctrl과 Meta를 하나로 합친다 (Windows/macOS 통합)', () => {
    expect(normalizeKey({ ...base, key: 'z', ctrlKey: true })).toBe('Ctrl+Z');
    expect(normalizeKey({ ...base, key: 'z', metaKey: true })).toBe('Ctrl+Z');
  });

  it('수식어 순서는 Ctrl → Alt → Shift로 고정된다', () => {
    expect(
      normalizeKey({ ...base, key: 'z', ctrlKey: true, altKey: true, shiftKey: true }),
    ).toBe('Ctrl+Alt+Shift+Z');
  });

  it('명명 키는 그대로 유지된다', () => {
    expect(normalizeKey({ ...base, key: 'ArrowRight' })).toBe('ArrowRight');
    expect(normalizeKey({ ...base, key: 'Delete' })).toBe('Delete');
    expect(normalizeKey({ ...base, key: 'Escape' })).toBe('Escape');
  });
});

// ── 위젯 소유권 (구 구현의 핵심 버그) ───────────────────────────────

describe('widgetOwnsKey', () => {
  it('버튼에 포커스가 있으면 Space/Enter는 버튼의 것이다', () => {
    const btn = document.createElement('button');
    expect(widgetOwnsKey(btn, 'Space')).toBe(true);
    expect(widgetOwnsKey(btn, 'Enter')).toBe(true);
  });

  it('버튼이어도 방향키는 라우터가 가져도 된다', () => {
    const btn = document.createElement('button');
    expect(widgetOwnsKey(btn, 'ArrowRight')).toBe(false);
  });

  it('분할자·목록·메뉴는 방향키의 주인이다 — 패널 폭 조절과 3D nudge가 동시에 일어나던 버그', () => {
    const sep = document.createElement('div');
    sep.setAttribute('role', 'separator');
    expect(widgetOwnsKey(sep, 'ArrowLeft')).toBe(true);
    expect(widgetOwnsKey(sep, 'Home')).toBe(true);

    const listbox = document.createElement('div');
    listbox.setAttribute('role', 'listbox');
    expect(widgetOwnsKey(listbox, 'ArrowDown')).toBe(true);
  });

  it('조상까지 거슬러 판정한다', () => {
    const btn = document.createElement('button');
    const span = document.createElement('span');
    btn.appendChild(span);
    expect(widgetOwnsKey(span, 'Space')).toBe(true);
  });

  it('수식어가 붙은 조합은 위젯 소유가 아니다 (Ctrl+D 등)', () => {
    const btn = document.createElement('button');
    expect(widgetOwnsKey(btn, 'Ctrl+Space')).toBe(true); // bare가 Space라 활성화 계열
    expect(widgetOwnsKey(btn, 'Ctrl+D')).toBe(false);
  });

  it('평범한 div는 아무 키도 소유하지 않는다', () => {
    const div = document.createElement('div');
    expect(widgetOwnsKey(div, 'Space')).toBe(false);
    expect(widgetOwnsKey(div, 'ArrowUp')).toBe(false);
  });
});

// ── 스코프 해석 ─────────────────────────────────────────────────────

describe('resolveScope', () => {
  it('선언이 없으면 global', () => {
    expect(resolveScope(document.createElement('div'))).toBe('global');
    expect(resolveScope(null)).toBe('global');
  });

  it('가장 가까운 선언을 쓴다', () => {
    const outer = document.createElement('div');
    outer.setAttribute(SCOPE_ATTR, 'dock');
    const inner = document.createElement('div');
    inner.setAttribute(SCOPE_ATTR, 'graph');
    const leaf = document.createElement('span');
    inner.appendChild(leaf);
    outer.appendChild(inner);
    expect(resolveScope(leaf)).toBe('graph');
  });

  it('알 수 없는 값은 global로 방어한다', () => {
    const el = document.createElement('div');
    el.setAttribute(SCOPE_ATTR, 'nonsense');
    expect(resolveScope(el)).toBe('global');
  });
});

// ── 라우팅 ──────────────────────────────────────────────────────────

function makeBinding(over: Partial<ShortcutBinding> = {}): ShortcutBinding {
  return {
    id: over.id ?? 'test',
    keys: over.keys ?? 'Space',
    scope: over.scope ?? 'global',
    labelKo: over.labelKo ?? '테스트',
    group: over.group ?? '테스트',
    run: over.run ?? vi.fn(),
    ...over,
  } as ShortcutBinding;
}

function fireKey(target: EventTarget, init: KeyboardEventInit): KeyboardEvent {
  const e = new KeyboardEvent('keydown', { bubbles: true, cancelable: true, ...init });
  target.dispatchEvent(e);
  return e;
}

describe('createShortcutRouter', () => {
  it('등록된 키를 실행한다', () => {
    const host = document.createElement('div');
    const router = createShortcutRouter(host);
    const run = vi.fn();
    router.register(makeBinding({ keys: 'Space', run }));
    fireKey(host, { key: ' ' });
    expect(run).toHaveBeenCalledTimes(1);
    router.dispose();
  });

  it('해제하면 더 이상 실행되지 않는다', () => {
    const host = document.createElement('div');
    const router = createShortcutRouter(host);
    const run = vi.fn();
    const off = router.register(makeBinding({ keys: 'Space', run }));
    off();
    fireKey(host, { key: ' ' });
    expect(run).not.toHaveBeenCalled();
    router.dispose();
  });

  it('텍스트 입력 중에는 기본적으로 가로채지 않는다', () => {
    const host = document.createElement('div');
    const input = document.createElement('input');
    host.appendChild(input);
    document.body.appendChild(host);
    const router = createShortcutRouter(host);
    const run = vi.fn();
    router.register(makeBinding({ keys: 'Space', run }));
    fireKey(input, { key: ' ' });
    expect(run).not.toHaveBeenCalled();
    router.dispose();
    host.remove();
  });

  it('allowInTextEntry면 입력 중에도 동작한다 (Ctrl+S 등)', () => {
    const host = document.createElement('div');
    const input = document.createElement('input');
    host.appendChild(input);
    document.body.appendChild(host);
    const router = createShortcutRouter(host);
    const run = vi.fn();
    router.register(makeBinding({ keys: 'Ctrl+S', run, allowInTextEntry: true }));
    fireKey(input, { key: 's', ctrlKey: true });
    expect(run).toHaveBeenCalledTimes(1);
    router.dispose();
    host.remove();
  });

  it('포커스된 버튼의 Space를 가로채지 않는다 — 구 구현이 파괴하던 계약', () => {
    const host = document.createElement('div');
    const btn = document.createElement('button');
    host.appendChild(btn);
    document.body.appendChild(host);
    const router = createShortcutRouter(host);
    const run = vi.fn();
    router.register(makeBinding({ keys: 'Space', run }));
    fireKey(btn, { key: ' ' });
    expect(run).not.toHaveBeenCalled();
    router.dispose();
    host.remove();
  });

  it('스코프 바인딩이 global보다 우선한다', () => {
    const host = document.createElement('div');
    const zone = document.createElement('div');
    zone.setAttribute(SCOPE_ATTR, 'graph');
    host.appendChild(zone);
    document.body.appendChild(host);
    const router = createShortcutRouter(host);
    const globalRun = vi.fn();
    const graphRun = vi.fn();
    router.register(makeBinding({ id: 'g', keys: 'Delete', scope: 'global', run: globalRun }));
    router.register(makeBinding({ id: 'fg', keys: 'Delete', scope: 'graph', run: graphRun }));
    fireKey(zone, { key: 'Delete' });
    expect(graphRun).toHaveBeenCalledTimes(1);
    expect(globalRun).not.toHaveBeenCalled();
    router.dispose();
    host.remove();
  });

  it('isEnabled가 false면 다음 후보로 넘어간다 (재생 중 편집 잠금)', () => {
    const host = document.createElement('div');
    const zone = document.createElement('div');
    zone.setAttribute(SCOPE_ATTR, 'viewport');
    host.appendChild(zone);
    document.body.appendChild(host);
    const router = createShortcutRouter(host);
    const scopedRun = vi.fn();
    const globalRun = vi.fn();
    router.register(
      makeBinding({ id: 's', keys: 'X', scope: 'viewport', run: scopedRun, isEnabled: () => false }),
    );
    router.register(makeBinding({ id: 'g', keys: 'X', scope: 'global', run: globalRun }));
    fireKey(zone, { key: 'x' });
    expect(scopedRun).not.toHaveBeenCalled();
    expect(globalRun).toHaveBeenCalledTimes(1);
    router.dispose();
    host.remove();
  });

  it('setEnabled(false)면 전부 무시한다 (모달 열림)', () => {
    const host = document.createElement('div');
    const router = createShortcutRouter(host);
    const run = vi.fn();
    router.register(makeBinding({ keys: 'Space', run }));
    router.setEnabled(false);
    fireKey(host, { key: ' ' });
    expect(run).not.toHaveBeenCalled();
    router.setEnabled(true);
    fireKey(host, { key: ' ' });
    expect(run).toHaveBeenCalledTimes(1);
    router.dispose();
  });

  it('이미 preventDefault된 이벤트는 건드리지 않는다', () => {
    // 실제 구조를 재현한다: 라우터는 바깥(window)에 붙고, 안쪽 위젯이 먼저 처리한 뒤
    // 버블링으로 라우터에 도달한다. 안쪽이 소비한 키를 라우터가 다시 실행하면 안 된다.
    const host = document.createElement('div');
    const child = document.createElement('div');
    host.appendChild(child);
    document.body.appendChild(host);
    child.addEventListener('keydown', (e) => {
      e.preventDefault();
    });
    const router = createShortcutRouter(host);
    const run = vi.fn();
    router.register(makeBinding({ keys: 'Space', run }));
    fireKey(child, { key: ' ' });
    expect(run).not.toHaveBeenCalled();
    router.dispose();
    host.remove();
  });

  it('list()는 등록된 바인딩만 반환한다 — 도움말 시트가 없는 키를 광고하지 않는다', () => {
    const host = document.createElement('div');
    const router = createShortcutRouter(host);
    router.register(makeBinding({ id: 'a', keys: 'Space', labelKo: '재생' }));
    router.register(makeBinding({ id: 'b', keys: 'Ctrl+Z', labelKo: '실행 취소' }));
    expect(router.list().map((b) => b.id).sort()).toEqual(['a', 'b']);
    router.dispose();
    expect(router.list()).toHaveLength(0);
  });

  it('같은 id로 다시 등록하면 덮어쓴다 (중복 방지)', () => {
    const host = document.createElement('div');
    const router = createShortcutRouter(host);
    const first = vi.fn();
    const second = vi.fn();
    router.register(makeBinding({ id: 'dup', keys: 'Space', run: first }));
    router.register(makeBinding({ id: 'dup', keys: 'Space', run: second }));
    fireKey(host, { key: ' ' });
    expect(first).not.toHaveBeenCalled();
    expect(second).toHaveBeenCalledTimes(1);
    router.dispose();
  });
});

// ── 표시 포맷 ───────────────────────────────────────────────────────

describe('formatKeys / keyHint', () => {
  it('방향키·특수키를 기호로 바꾼다', () => {
    expect(formatKeys('ArrowRight')).toBe('→');
    expect(formatKeys('Ctrl+Shift+Z')).toBe('Ctrl + Shift + Z');
    expect(formatKeys('Delete')).toBe('Del');
    expect(formatKeys('Escape')).toBe('Esc');
  });

  it('keyHint는 툴팁에 붙일 접미사를 만든다', () => {
    expect(keyHint('Space')).toBe(' (Space)');
    expect(`재생/일시정지${keyHint('Space')}`).toBe('재생/일시정지 (Space)');
  });
});
