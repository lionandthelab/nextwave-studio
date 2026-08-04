// ui/shortcuts.ts — 단축키 단일 라우터 (docs/UX_AUDIT.md C-6)
//
// ── 왜 라우터인가 ───────────────────────────────────────────────────
//
// 구 구현은 `window`에 keydown을 거는 곳이 **5군데**였고 각자 `isTypingTarget`으로
// 방어했다. 소유자가 없는 키맵은 반드시 이 상태로 수렴한다:
//
//   · Space를 **3개가 나눠 가졌다** — 재생 토글 + 충돌 로그 행 활성화 + 그래프 팬 모드.
//     가드가 BUTTON을 제외하지 않아 **포커스된 버튼의 Space 활성화가 파괴**됐고,
//     충돌 로그 행을 키보드로 여는 순간 시뮬레이션이 재생됐다.
//   · 방향키를 **2개가 나눠 가졌다** — 스플리터 크기 조절 + 3D 오브젝트 nudge.
//     스플리터에 포커스를 두고 ←/→를 누르면 패널 폭 조절과 오브젝트 이동이 **동시에**
//     일어났고, 뷰포트가 20.9%로 눌린 화면에서 사용자는 그것을 인지하지 못했다.
//   · `docs/UX_DESIGN.md` §9가 규정한 `F`·`Home`·`←/→ Step`은 **아무도 구현하지 않았다.**
//     규정 8종 중 스펙대로 동작하는 건 Space와 W/E/R 둘뿐이었고, `←/→`는 스펙(Step)과
//     **반대 기능**(오브젝트 5cm 이동)에 배선되어 있었다 — Step 하려던 사용자가 물리
//     결과를 오염시키고 결정론적 재실행 전제를 깨뜨렸다.
//
// 이제 규칙은 하나다: **모든 전역 단축키는 이 라우터를 통과한다.**
//
// ── 스코프 ──────────────────────────────────────────────────────────
// 포커스가 어느 영역에 있는지로 바인딩을 고른다. 영역은 DOM의 `data-shortcut-scope`
// 속성으로 선언되고(workspace.ts가 슬롯에 부여), 라우터는 activeElement에서 위로 올라가며
// 가장 가까운 선언을 찾는다. 스코프 바인딩이 없으면 global로 폴백한다.

import { isTextEntryTarget } from './a11y';

export type ShortcutScope = 'global' | 'viewport' | 'graph' | 'dock' | 'inspector' | 'commandBar';

/** DOM에 스코프를 선언하는 속성 이름 */
export const SCOPE_ATTR = 'data-shortcut-scope';

export interface ShortcutBinding {
  /** 안정 id (도움말 시트 정렬·중복 검출용) */
  readonly id: string;
  /**
   * 키 조합. 정규화 규약: `[Ctrl+][Alt+][Shift+]<Key>`
   * `<Key>`는 KeyboardEvent.key를 쓰되 단일 문자는 대문자로("z" → "Z"),
   * Space는 "Space", 방향키는 "ArrowLeft" 등 그대로.
   * Ctrl은 macOS의 Meta도 함께 받는다(Ctrl/Cmd 통합).
   */
  readonly keys: string;
  readonly scope: ShortcutScope;
  /** 도움말 시트에 표시될 한국어 설명 */
  readonly labelKo: string;
  /** 도움말 시트의 그룹 제목 */
  readonly group: string;
  /**
   * 도움말 시트에서 숨긴다 (기본 false).
   *
   * 한 조작이 여러 키로 들어오는 경우(←→↑↓, Shift 미세 변형)를 위한 것이다. 라우터는
   * 키 하나당 바인딩 하나를 요구하므로 별칭이 필연적으로 생기는데, 그걸 전부 나열하면
   * 시트가 같은 조작 8줄로 덮인다. **대표 바인딩 하나만 보이고**(keysDisplay로 키
   * 묶음을 표기) 나머지는 숨긴다 — 시트가 "실제 등록된 것만 그린다"는 계약은 유지된다
   * (숨긴 별칭도 대표 줄의 표기에 포함되므로 광고와 구현이 어긋나지 않는다).
   */
  readonly hidden?: boolean;
  /** 시트에 표시할 키 토큰 override (예: ['←', '→', '↑', '↓']) — 없으면 formatKeys(keys) */
  readonly keysDisplay?: readonly string[];
  /** 텍스트 입력 중에도 동작해야 하는가 (기본 false) */
  readonly allowInTextEntry?: boolean;
  /** 지금 이 바인딩이 유효한가 (예: 재생 중에는 편집 단축키 비활성) */
  isEnabled?(): boolean;
  run(e: KeyboardEvent): void;
}

/** 이벤트를 정규화된 키 문자열로 (테스트 대상 — 순수) */
export function normalizeKey(e: {
  key: string;
  code?: string;
  ctrlKey: boolean;
  metaKey: boolean;
  altKey: boolean;
  shiftKey: boolean;
}): string {
  let key = e.key;
  if (key === ' ' || e.code === 'Space') key = 'Space';
  else if (key.length === 1) key = key.toUpperCase();
  // 문장부호는 Shift가 **이미 문자에 인코딩**돼 있다('?'는 Shift+'/'의 결과다).
  // Shift를 따로 붙이면 레이아웃마다 다른 조합이 되어 등록이 어긋난다.
  const shiftIsImplicit = key.length === 1 && !/[A-Z0-9]/.test(key);
  const parts: string[] = [];
  if (e.ctrlKey || e.metaKey) parts.push('Ctrl');
  if (e.altKey) parts.push('Alt');
  if (e.shiftKey && !shiftIsImplicit) parts.push('Shift');
  parts.push(key);
  return parts.join('+');
}

/**
 * 자체 키 처리를 갖는 위젯이 이 키를 먼저 가져야 하는가?
 *
 * 버튼/링크의 Space·Enter, 목록/메뉴/분할자의 방향키·Home·End는 위젯의 것이다.
 * 라우터가 가로채면 키보드 조작이 파괴된다(구 구현의 핵심 버그).
 */
export function widgetOwnsKey(target: EventTarget | null, normalized: string): boolean {
  if (!(target instanceof Element)) return false;
  const bare = normalized.replace(/^(Ctrl\+|Alt\+|Shift\+)+/, '');

  const activation = bare === 'Space' || bare === 'Enter';
  if (activation) {
    return (
      target.closest(
        'button, a[href], [role="button"], [role="menuitem"], [role="option"], [role="tab"], summary',
      ) !== null
    );
  }

  const navigation =
    bare === 'ArrowUp' ||
    bare === 'ArrowDown' ||
    bare === 'ArrowLeft' ||
    bare === 'ArrowRight' ||
    bare === 'Home' ||
    bare === 'End';
  if (navigation) {
    return (
      target.closest(
        '[role="separator"], [role="listbox"], [role="menu"], [role="tablist"], [role="grid"], [role="slider"]',
      ) !== null
    );
  }
  return false;
}

/** activeElement에서 위로 올라가며 가장 가까운 스코프 선언을 찾는다 */
export function resolveScope(target: EventTarget | null): ShortcutScope {
  if (!(target instanceof Element)) return 'global';
  const holder = target.closest(`[${SCOPE_ATTR}]`);
  if (holder === null) return 'global';
  const raw = holder.getAttribute(SCOPE_ATTR);
  switch (raw) {
    case 'viewport':
    case 'graph':
    case 'dock':
    case 'inspector':
    case 'commandBar':
      return raw;
    default:
      return 'global';
  }
}

export interface ShortcutRouterHandle {
  /** 바인딩 등록. 반환된 함수로 해제한다. */
  register(binding: ShortcutBinding): () => void;
  /** 여러 개 한 번에 */
  registerAll(bindings: readonly ShortcutBinding[]): () => void;
  /** 도움말 시트용 — 그룹별로 정리된 현재 **실제 구현된** 바인딩 목록 */
  list(): readonly ShortcutBinding[];
  /** 전체 비활성 (모달이 열려 있을 때 등) */
  setEnabled(enabled: boolean): void;
  isEnabled(): boolean;
  dispose(): void;
}

/**
 * 라우터를 만들어 target(기본 window)에 붙인다.
 *
 * 해석 순서:
 *   1. 라우터 자체가 비활성이면 무시
 *   2. 텍스트 입력 중이면 `allowInTextEntry` 바인딩만
 *   3. 포커스된 위젯이 이 키의 주인이면 무시 (버튼의 Space, 목록의 방향키 …)
 *   4. 현재 스코프 바인딩 → 없으면 global 바인딩
 *   5. `isEnabled()`가 false면 무시
 */
export function createShortcutRouter(
  target: EventTarget = window,
): ShortcutRouterHandle {
  const bindings = new Map<string, ShortcutBinding>();
  let enabled = true;

  const onKeyDown = (ev: Event): void => {
    if (!enabled) return;
    const e = ev as KeyboardEvent;
    if (e.defaultPrevented) return;
    // IME 조합 중에는 어떤 단축키도 가로채지 않는다 (한글 입력 중 오작동 방지)
    if (e.isComposing) return;

    const normalized = normalizeKey(e);
    const inTextEntry = isTextEntryTarget(e.target);
    const scope = resolveScope(e.target);

    // 스코프 우선, 그다음 global
    const candidates: ShortcutBinding[] = [];
    for (const b of bindings.values()) {
      if (b.keys !== normalized) continue;
      if (b.scope === scope) candidates.unshift(b);
      else if (b.scope === 'global') candidates.push(b);
    }
    if (candidates.length === 0) return;

    for (const binding of candidates) {
      if (inTextEntry && binding.allowInTextEntry !== true) continue;
      if (!inTextEntry && widgetOwnsKey(e.target, normalized)) continue;
      if (binding.isEnabled !== undefined && !binding.isEnabled()) continue;
      binding.run(e);
      return;
    }
  };

  target.addEventListener('keydown', onKeyDown);

  return {
    register: (binding: ShortcutBinding): (() => void) => {
      bindings.set(binding.id, binding);
      return () => {
        bindings.delete(binding.id);
      };
    },
    registerAll: (list: readonly ShortcutBinding[]): (() => void) => {
      for (const b of list) bindings.set(b.id, b);
      return () => {
        for (const b of list) bindings.delete(b.id);
      };
    },
    list: (): readonly ShortcutBinding[] => [...bindings.values()],
    setEnabled: (v: boolean): void => {
      enabled = v;
    },
    isEnabled: (): boolean => enabled,
    dispose: (): void => {
      target.removeEventListener('keydown', onKeyDown);
      bindings.clear();
    },
  };
}

// ── 표시용 포맷 ─────────────────────────────────────────────────────

const KEY_DISPLAY: Readonly<Record<string, string>> = {
  Ctrl: 'Ctrl',
  Alt: 'Alt',
  Shift: 'Shift',
  Space: 'Space',
  ArrowLeft: '←',
  ArrowRight: '→',
  ArrowUp: '↑',
  ArrowDown: '↓',
  Delete: 'Del',
  Escape: 'Esc',
  Enter: 'Enter',
  Home: 'Home',
};

/** `Ctrl+Shift+Z` → `Ctrl + Shift + Z` (도움말 시트/툴팁 표시용) */
export function formatKeys(keys: string): string {
  return keys
    .split('+')
    .map((k) => KEY_DISPLAY[k] ?? k)
    .join(' + ');
}

/** 버튼 title에 붙일 접미사 — `'재생/일시정지'` + `' (Space)'` */
export function keyHint(keys: string): string {
  return ` (${formatKeys(keys)})`;
}
