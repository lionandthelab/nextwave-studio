// ui/a11y.ts — 접근성 공용 프리미티브 (UX_AUDIT C-5 / C-12 / C-18)
//
// 같은 패턴이 코드베이스에 흩어져 있거나(clarify-card만 포커스 트랩을 구현했다) 아예
// 없어서(엔티티 목록에 roving tabindex가 없다) 생긴 부채를 한곳으로 모은다.
//
// 제공:
//   focusableWithin  — 컨테이너 안의 포커스 가능 요소 목록
//   trapFocus        — Tab/Shift+Tab 순환 트랩 + 포커스 복원 (모달·팝오버)
//   RovingTabindex   — 목록/그래프의 방향키 탐색 (listbox·menu·toolbar 패턴)
//   Announcer        — aria-live 요약 발화 (스로틀 — 스트림에 live를 걸지 않기 위해)
//
// ── aria-live에 관한 원칙 ───────────────────────────────────────────
// 충돌 로그처럼 초당 수십 건이 쏟아지는 스트림에는 **절대 aria-live를 걸지 않는다.**
// polite 큐는 취소되지 않고 누적되므로 사용자가 다른 조작을 해도 몇 분간 과거 이벤트만
// 읽는다. 링버퍼의 앞 행 제거도 변경으로 집계되어 중복 발화한다.
// 대신 Announcer로 **요약을 스로틀해** 알린다.

// ── 포커스 가능 요소 ────────────────────────────────────────────────

const FOCUSABLE_SELECTOR = [
  'a[href]',
  'button:not(:disabled)',
  'input:not(:disabled)',
  'select:not(:disabled)',
  'textarea:not(:disabled)',
  '[tabindex]:not([tabindex="-1"])',
].join(', ');

/** 컨테이너 안에서 실제로 포커스를 받을 수 있는 요소들 (숨김/크기 0 제외) */
export function focusableWithin(container: Element): HTMLElement[] {
  return Array.from(container.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR)).filter((el) => {
    if (el.hasAttribute('inert')) return false;
    if (el.getAttribute('aria-hidden') === 'true') return false;
    // offsetParent가 null이면 display:none 계열 (position:fixed는 예외 처리)
    if (el.offsetParent === null && getComputedStyle(el).position !== 'fixed') return false;
    return true;
  });
}

export interface FocusTrapHandle {
  /** 트랩 해제 + 진입 전 포커스 복원 */
  release(): void;
}

/**
 * 모달/팝오버에 Tab 순환 트랩을 건다.
 *
 * `aria-modal="true"`를 선언해 놓고 Tab이 배경으로 빠지면, 스크린리더에게는 "바깥은 없는
 * 셈"인데 실제 포커스는 읽히지 않는 요소에 가 있는 유령 상태가 된다(UX_AUDIT C-18).
 *
 * @param container 트랩 대상 (role="dialog" 등)
 * @param opts.initialFocus 진입 시 포커스할 요소. 없으면 첫 포커스 가능 요소 → container
 * @param opts.onEscape Escape 처리 (없으면 Escape를 가로채지 않는다)
 */
export function trapFocus(
  container: HTMLElement,
  opts: { initialFocus?: HTMLElement | null; onEscape?: () => void } = {},
): FocusTrapHandle {
  const previouslyFocused =
    document.activeElement instanceof HTMLElement ? document.activeElement : null;

  const onKeyDown = (e: KeyboardEvent): void => {
    if (e.key === 'Escape' && opts.onEscape !== undefined) {
      e.stopPropagation();
      e.preventDefault();
      opts.onEscape();
      return;
    }
    if (e.key !== 'Tab') return;
    const items = focusableWithin(container);
    const first = items[0];
    const last = items[items.length - 1];
    if (first === undefined || last === undefined) {
      e.preventDefault();
      container.focus();
      return;
    }
    const active = document.activeElement;
    if (e.shiftKey) {
      if (active === first || active === container || !container.contains(active)) {
        e.preventDefault();
        last.focus();
      }
    } else if (active === last || !container.contains(active)) {
      e.preventDefault();
      first.focus();
    }
  };

  container.addEventListener('keydown', onKeyDown);

  const target = opts.initialFocus ?? focusableWithin(container)[0] ?? container;
  // 컨테이너 자신에게 포커스를 줄 수 있어야 한다
  if (target === container && !container.hasAttribute('tabindex')) container.tabIndex = -1;
  target.focus();

  return {
    release: (): void => {
      container.removeEventListener('keydown', onKeyDown);
      if (previouslyFocused !== null && document.contains(previouslyFocused)) {
        previouslyFocused.focus();
      }
    },
  };
}

// ── Roving tabindex ─────────────────────────────────────────────────

export type RovingOrientation = 'vertical' | 'horizontal' | 'both';

export interface RovingTabindexOptions {
  readonly orientation?: RovingOrientation;
  /** 끝에서 반대편으로 순환할지 (기본 true) */
  readonly wrap?: boolean;
  /** Enter/Space로 항목을 "활성화"했을 때 */
  onActivate?(el: HTMLElement, index: number): void;
  /** 포커스가 이동했을 때 (선택 상태 미리보기 등) */
  onFocusChange?(el: HTMLElement, index: number): void;
}

export interface RovingTabindexHandle {
  /** 항목 목록을 교체한다 (재렌더 후 호출) */
  setItems(items: readonly HTMLElement[]): void;
  /** 활성(tabIndex=0) 항목을 지정한다 */
  setActive(index: number, focus?: boolean): void;
  readonly activeIndex: number;
  dispose(): void;
}

/**
 * 목록/툴바/그래프에 방향키 탐색을 붙인다 (WAI-ARIA APG roving tabindex).
 *
 * 항목 중 하나만 `tabIndex=0`이고 나머지는 `-1`이다 — 탭 한 번으로 목록에 들어오고,
 * 방향키로 이동하며, 목록을 빠져나갈 때도 탭 한 번이다. 항목이 100개라도 탭을
 * 100번 누르지 않는다.
 */
export function rovingTabindex(
  container: HTMLElement,
  initialItems: readonly HTMLElement[],
  opts: RovingTabindexOptions = {},
): RovingTabindexHandle {
  const orientation: RovingOrientation = opts.orientation ?? 'vertical';
  const wrap = opts.wrap ?? true;
  let items: HTMLElement[] = [...initialItems];
  let activeIndex = 0;

  const paint = (): void => {
    items.forEach((el, i) => {
      el.tabIndex = i === activeIndex ? 0 : -1;
    });
  };

  const move = (delta: number, focus: boolean): void => {
    if (items.length === 0) return;
    let next = activeIndex + delta;
    if (wrap) {
      next = ((next % items.length) + items.length) % items.length;
    } else {
      next = Math.min(Math.max(next, 0), items.length - 1);
    }
    activeIndex = next;
    paint();
    const target = items[activeIndex];
    if (focus && target !== undefined) {
      target.focus();
      opts.onFocusChange?.(target, activeIndex);
    }
  };

  const prevKeys =
    orientation === 'horizontal'
      ? ['ArrowLeft']
      : orientation === 'vertical'
        ? ['ArrowUp']
        : ['ArrowUp', 'ArrowLeft'];
  const nextKeys =
    orientation === 'horizontal'
      ? ['ArrowRight']
      : orientation === 'vertical'
        ? ['ArrowDown']
        : ['ArrowDown', 'ArrowRight'];

  const onKeyDown = (e: KeyboardEvent): void => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (prevKeys.includes(e.key)) {
      e.preventDefault();
      e.stopPropagation();
      move(-1, true);
    } else if (nextKeys.includes(e.key)) {
      e.preventDefault();
      e.stopPropagation();
      move(1, true);
    } else if (e.key === 'Home') {
      e.preventDefault();
      e.stopPropagation();
      activeIndex = 0;
      paint();
      items[0]?.focus();
    } else if (e.key === 'End') {
      e.preventDefault();
      e.stopPropagation();
      activeIndex = Math.max(items.length - 1, 0);
      paint();
      items[activeIndex]?.focus();
    } else if (e.key === 'Enter' || e.key === ' ') {
      const el = items[activeIndex];
      if (el === undefined) return;
      e.preventDefault();
      // Space가 전역 재생 토글로 새어 나가지 않게 한다 (UX_AUDIT C-6)
      e.stopPropagation();
      opts.onActivate?.(el, activeIndex);
    }
  };

  const onFocusIn = (e: FocusEvent): void => {
    const idx = items.indexOf(e.target as HTMLElement);
    if (idx >= 0 && idx !== activeIndex) {
      activeIndex = idx;
      paint();
    }
  };

  container.addEventListener('keydown', onKeyDown);
  container.addEventListener('focusin', onFocusIn);
  paint();

  return {
    setItems: (next: readonly HTMLElement[]): void => {
      items = [...next];
      if (activeIndex >= items.length) activeIndex = Math.max(items.length - 1, 0);
      paint();
    },
    setActive: (index: number, focus = false): void => {
      const target = items[index];
      if (target === undefined) return;
      activeIndex = index;
      paint();
      if (focus) target.focus();
    },
    get activeIndex(): number {
      return activeIndex;
    },
    dispose: (): void => {
      container.removeEventListener('keydown', onKeyDown);
      container.removeEventListener('focusin', onFocusIn);
    },
  };
}

// ── Announcer (스로틀된 aria-live 요약) ─────────────────────────────

export interface AnnouncerHandle {
  /**
   * 요약을 발화 예약한다. 마지막 발화 후 minIntervalMs가 지나지 않았으면 **덮어쓰고**
   * 타이머가 만료될 때 최신 문구 하나만 읽는다 — 큐가 쌓이지 않는다.
   */
  announce(text: string): void;
  /** 스로틀을 무시하고 즉시 발화 (완료·오류 같은 단발 사건) */
  announceNow(text: string): void;
  dispose(): void;
}

/**
 * 화면 밖 live 영역을 만들어 요약을 스로틀 발화한다.
 *
 * @param politeness 'polite'가 기본. 'assertive'는 진행 중 발화를 끊으므로 오류에만.
 * @param minIntervalMs 이 간격 안의 연속 호출은 마지막 문구로 합쳐진다.
 */
export function createAnnouncer(
  host: HTMLElement = document.body,
  politeness: 'polite' | 'assertive' = 'polite',
  minIntervalMs = 3000,
): AnnouncerHandle {
  const region = document.createElement('div');
  region.className = 'sr-only';
  region.setAttribute('role', politeness === 'assertive' ? 'alert' : 'status');
  region.setAttribute('aria-live', politeness);
  region.setAttribute('aria-atomic', 'true');
  host.appendChild(region);

  let lastSpokenAtMs = 0;
  let pendingText: string | null = null;
  let timerId: number | null = null;

  const flush = (): void => {
    timerId = null;
    if (pendingText === null) return;
    region.textContent = pendingText;
    pendingText = null;
    lastSpokenAtMs = Date.now();
  };

  return {
    announce: (text: string): void => {
      pendingText = text;
      const elapsed = Date.now() - lastSpokenAtMs;
      if (elapsed >= minIntervalMs && timerId === null) {
        flush();
        return;
      }
      if (timerId === null) {
        timerId = window.setTimeout(flush, Math.max(minIntervalMs - elapsed, 0));
      }
    },
    announceNow: (text: string): void => {
      if (timerId !== null) {
        window.clearTimeout(timerId);
        timerId = null;
      }
      pendingText = text;
      flush();
    },
    dispose: (): void => {
      if (timerId !== null) window.clearTimeout(timerId);
      region.remove();
    },
  };
}

// ── 입력 포커스 판정 (전역 단축키 가드 공용) ────────────────────────

/**
 * 이 요소에 포커스가 있을 때 전역 단축키가 가로채면 안 되는가?
 *
 * 구 `isTypingTarget`은 INPUT/TEXTAREA/SELECT/contenteditable만 걸러서, **버튼에 포커스가
 * 있어도 Space를 가로챘다** — 그 결과 포커스된 버튼의 Space 활성화가 파괴되고 충돌 로그
 * 행을 키보드로 여는 순간 시뮬레이션이 재생됐다(UX_AUDIT C-6).
 */
export function isTextEntryTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return true;
  return target.isContentEditable;
}

/** 자체 키 처리를 갖는 위젯(버튼·분할자·목록·메뉴) 안에 포커스가 있는가? */
export function isInteractiveWidgetTarget(target: EventTarget | null): boolean {
  if (!(target instanceof Element)) return false;
  return (
    target.closest(
      'button, [role="button"], [role="separator"], [role="menuitem"], [role="option"], [role="tab"], [role="listbox"], [role="menu"], [role="dialog"]',
    ) !== null
  );
}
