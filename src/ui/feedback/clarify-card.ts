// ui/feedback/clarify-card.ts — 명확화 카드 (UX_DESIGN §4.1 Flow 1, §7 "명확화 필요")
//
// 플래너가 { type:'clarify', question, options } 를 반환하면(PLANNER.md §3) 커맨드바
// 아래 중앙에 떠 있는 카드로 질문 + 옵션 버튼 + '직접 지정' 자유입력 + 취소를 보인다.
// 옵션 클릭 → onPick(옵션), 취소/Escape → onPick(null). 선택은 글루가 다시 생성에
// 반영한다(PLANNER.md §3 clarify 분기). 이 모듈은 core/planner를 import하지 않는다.
//
// 접근성(UX_DESIGN §9): role=dialog, 열릴 때 첫 포커스 이동, Escape=취소, Tab은 카드
// 안에서 순환(trap-ish). 파괴적 동작이 아니므로 확인 없이 즉시 선택/취소한다.

import {
  COLOR,
  FONT,
  LAYOUT,
  RADIUS,
  SHADOW,
  SPACE,
  Z_INDEX,
  ensureThemeStyles,
  makeButton,
  styled,
} from '../theme';

// ── 공개 타입 ───────────────────────────────────────────────────────

export interface ClarifyCardHandle {
  /** 명확화 질문 표시. options 미지정/빈 배열이면 '직접 지정' 자유입력만 노출한다. */
  show(question: string, options: string[] | undefined, onPick: (choice: string | null) => void): void;
  hide(): void;
  dispose(): void;
}

// ── 상수 (매직넘버 금지 — CLAUDE.md §4) ─────────────────────────────

/** 카드 상단 오프셋(px) — 커맨드바 바로 아래 여백 */
const CARD_TOP_PX = LAYOUT.belowBarTopPx;

// ── 내부 헬퍼 ───────────────────────────────────────────────────────

/** 카드 안 포커스 가능한 요소(순환 trap용) */
function focusable(root: HTMLElement): HTMLElement[] {
  const nodes = root.querySelectorAll<HTMLElement>('button, input, [tabindex]:not([tabindex="-1"])');
  return Array.from(nodes).filter((el) => !el.hasAttribute('disabled') && el.tabIndex !== -1);
}

// ── 마운트 ──────────────────────────────────────────────────────────

export function mountClarifyCard(host: HTMLElement): ClarifyCardHandle {
  ensureThemeStyles();

  const card = styled(document.createElement('div'), {
    position: 'fixed',
    top: `${CARD_TOP_PX}px`,
    left: '50%',
    transform: 'translateX(-50%)',
    zIndex: Z_INDEX.panel,
    width: 'min(460px, 92vw)',
    maxWidth: '92vw',
    display: 'none',
    flexDirection: 'column',
    gap: SPACE.md,
    background: COLOR.bgPanel,
    border: `1px solid ${COLOR.border}`,
    borderRadius: RADIUS.md,
    boxShadow: SHADOW.panel,
    padding: `${SPACE.lg} ${SPACE.lg}`,
    color: COLOR.text,
    fontFamily: FONT.ui,
    fontSize: '12px',
    boxSizing: 'border-box',
    pointerEvents: 'auto',
  });
  card.dataset.testid = 'clarify-card';
  card.setAttribute('role', 'dialog');
  card.setAttribute('aria-modal', 'false');
  const titleId = 'rsw-clarify-title';
  card.setAttribute('aria-labelledby', titleId);
  // 카드 위 상호작용이 뷰포트 orbit으로 새지 않게 (패널 규약)
  for (const type of ['pointerdown', 'pointermove', 'pointerup', 'wheel']) {
    card.addEventListener(type, (e) => {
      e.stopPropagation();
    });
  }

  // 질문 텍스트
  const question = styled(document.createElement('div'), {
    color: COLOR.textStrong,
    fontSize: '13px',
    fontWeight: '600',
    lineHeight: '1.5',
  });
  question.id = titleId;

  // 옵션 버튼 줄 (여러 개 — wrap)
  const optionsRow = styled(document.createElement('div'), {
    display: 'flex',
    flexWrap: 'wrap',
    gap: SPACE.sm,
  });
  optionsRow.dataset.testid = 'clarify-options';

  // '직접 지정' 자유입력 줄
  const freeRow = styled(document.createElement('div'), {
    display: 'flex',
    alignItems: 'center',
    gap: SPACE.sm,
  });
  const freeInput = document.createElement('input');
  freeInput.type = 'text';
  freeInput.className = 'ui-input';
  freeInput.dataset.testid = 'clarify-free';
  freeInput.placeholder = '직접 지정…';
  freeInput.setAttribute('aria-label', '직접 지정 입력');
  styled(freeInput, { flex: '1', minWidth: '0' });
  const freeSubmit = makeButton('확인', '직접 입력한 값으로 진행', 'clarify-submit');
  freeRow.appendChild(freeInput);
  freeRow.appendChild(freeSubmit);

  // 취소 줄 (우측 정렬)
  const footer = styled(document.createElement('div'), {
    display: 'flex',
    justifyContent: 'flex-end',
    gap: SPACE.sm,
  });
  const cancelButton = makeButton('취소', '명확화 취소', 'clarify-cancel', 'ghost');
  footer.appendChild(cancelButton);

  card.appendChild(question);
  card.appendChild(optionsRow);
  card.appendChild(freeRow);
  card.appendChild(footer);
  host.appendChild(card);

  // ── 상태 · 배선 ───────────────────────────────────────────────────

  /** 현재 대기 중인 콜백 — 한 번만 발화(중복 방지) */
  let pending: ((choice: string | null) => void) | null = null;

  const hideDom = (): void => {
    card.style.display = 'none';
    window.removeEventListener('keydown', onKeyDown, true);
    optionsRow.replaceChildren();
    freeInput.value = '';
  };

  /** 사용자 확정(옵션·자유입력·취소·Escape 공통) — pending을 소비해 정확히 1회 호출 */
  const resolve = (choice: string | null): void => {
    const cb = pending;
    pending = null;
    hideDom();
    if (cb) cb(choice);
  };

  function onKeyDown(e: KeyboardEvent): void {
    if (card.style.display === 'none') return;
    if (e.key === 'Escape') {
      e.preventDefault();
      resolve(null);
      return;
    }
    if (e.key === 'Tab') {
      // 카드 안에서 포커스 순환 (trap-ish)
      const items = focusable(card);
      const first = items[0];
      const last = items[items.length - 1];
      if (!first || !last) return;
      const active = document.activeElement as HTMLElement | null;
      if (e.shiftKey && active === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && active === last) {
        e.preventDefault();
        first.focus();
      }
    }
  }

  freeSubmit.addEventListener('click', () => {
    const value = freeInput.value.trim();
    if (value.length === 0) {
      freeInput.focus();
      return;
    }
    resolve(value);
  });
  freeInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.isComposing) {
      e.preventDefault();
      freeSubmit.click();
    }
  });
  cancelButton.addEventListener('click', () => {
    resolve(null);
  });

  const show = (
    q: string,
    options: string[] | undefined,
    onPick: (choice: string | null) => void,
  ): void => {
    // 이전 카드가 미해결이면 새 질문으로 교체하며 이전 콜백은 조용히 폐기한다
    // (콜백이라 대기 누수가 없다 — 새 clarify로 대체되었을 뿐이므로 취소 통지하지 않는다).
    pending = onPick;

    question.textContent = q;
    optionsRow.replaceChildren();
    const opts = options ?? [];
    for (const opt of opts) {
      const button = makeButton(opt, `옵션: ${opt}`, 'clarify-option');
      button.addEventListener('click', () => {
        resolve(opt);
      });
      optionsRow.appendChild(button);
    }
    optionsRow.style.display = opts.length > 0 ? 'flex' : 'none';
    freeInput.value = '';

    card.style.display = 'flex';
    window.addEventListener('keydown', onKeyDown, true);
    // 첫 포커스: 옵션이 있으면 첫 옵션, 없으면 자유입력
    const firstOption = optionsRow.querySelector<HTMLElement>('button');
    (firstOption ?? freeInput).focus();
  };

  return {
    show,
    hide: (): void => {
      // 프로그램적 dismiss는 조용히 닫는다(사용자 취소가 아니므로 onPick을 부르지 않음).
      // 사용자 취소는 취소 버튼/Escape가 onPick(null)로 통지한다.
      pending = null;
      hideDom();
    },
    dispose: (): void => {
      window.removeEventListener('keydown', onKeyDown, true);
      pending = null;
      card.remove();
    },
  };
}
