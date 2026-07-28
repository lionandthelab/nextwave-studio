// ui/feedback/clarify-card.ts — 명확화 카드 (UX_DESIGN §4.1 Flow 1, §7 "명확화 필요")
//
// 플래너가 { type:'clarify', question, options } 를 반환하면(PLANNER.md §3) 커맨드바
// 아래 중앙에 떠 있는 카드로 질문 + 옵션 버튼 + '직접 지정' 자유입력 + 취소를 보인다.
// 옵션 클릭 → onPick(옵션), 취소/Escape → onPick(null). 선택은 글루가 다시 생성에
// 반영한다(PLANNER.md §3 clarify 분기). 이 모듈은 core/planner를 import하지 않는다.
//
// 접근성(UX_DESIGN §9): role=dialog, 열릴 때 첫 포커스 이동, Escape=취소, Tab은 카드
// 안에서 순환. 파괴적 동작이 아니므로 확인 없이 즉시 선택/취소한다.
//
// ── 포커스 트랩 단일 구현 (UX_AUDIT C-18) ────────────────────────────
// 이 파일에만 있던 수제 focusable()/Tab 순환 로직은 `ui/a11y.ts`의 trapFocus로
// **승격·교체**됐다 — 같은 패턴을 두 곳에서 각자 구현하면(import-dialog는 아예 없었다)
// 규칙이 갈라진다. 동작 계약은 그대로다: 첫 포커스(옵션 있으면 첫 옵션, 없으면 자유입력)
// · Escape=취소 · Tab 순환. **추가로 얻은 것**은 닫을 때의 포커스 복원이다(trapFocus가
// 진입 전 활성 요소를 기억한다) — 명확화가 끝나면 커맨드바 입력으로 돌아온다.

import {
  BORDER,
  BORDER_WIDTH,
  COLOR,
  LAYOUT,
  RADIUS,
  SHADOW,
  SPACE,
  SURFACE,
  TYPE,
  Z_INDEX,
  applyType,
  ensureThemeStyles,
  makeButton,
  styled,
} from '../theme';
import { trapFocus } from '../a11y';
import type { FocusTrapHandle } from '../a11y';

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

// ── 마운트 ──────────────────────────────────────────────────────────

export function mountClarifyCard(host: HTMLElement): ClarifyCardHandle {
  ensureThemeStyles();

  const card = applyType(document.createElement('div'), TYPE.body);
  styled(card, {
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
    background: SURFACE.overlay,
    border: `${BORDER_WIDTH.hair} solid ${BORDER.default}`,
    borderRadius: RADIUS.md,
    boxShadow: SHADOW.overlay,
    padding: SPACE.xl,
    color: COLOR.text,
    boxSizing: 'border-box',
    pointerEvents: 'auto',
  });
  card.dataset.testid = 'clarify-card';
  card.setAttribute('role', 'dialog');
  // Tab이 실제로 카드 안에서 순환하므로 키보드/스크린리더에게는 이 카드가 모달이다.
  // 선언과 동작을 일치시킨다 — `aria-modal="false"`인데 Tab이 갇혀 있으면 "빠져나갈 수
  // 있다"고 안내한 뒤 빠져나가지 못하게 하는 셈이다 (UX_AUDIT C-18 모달 트랩 항목).
  card.setAttribute('aria-modal', 'true');
  const titleId = 'rsw-clarify-title';
  card.setAttribute('aria-labelledby', titleId);
  // 카드 위 상호작용이 뷰포트 orbit으로 새지 않게 (패널 규약)
  for (const type of ['pointerdown', 'pointermove', 'pointerup', 'wheel']) {
    card.addEventListener(type, (e) => {
      e.stopPropagation();
    });
  }

  // 질문 텍스트
  const question = applyType(document.createElement('div'), TYPE.subhead);
  question.style.color = COLOR.textStrong;
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
  /** 열려 있는 동안의 포커스 트랩 (release가 진입 전 포커스를 복원한다) */
  let trap: FocusTrapHandle | null = null;

  const hideDom = (): void => {
    card.style.display = 'none';
    trap?.release();
    trap = null;
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
    // 재진입(미해결 카드를 새 질문으로 교체)이면 이전 트랩을 먼저 걷는다 —
    // 그래야 "진입 전 포커스"가 이전 트랩 것으로 덮이지 않는다.
    trap?.release();
    // 첫 포커스: 옵션이 있으면 첫 옵션, 없으면 자유입력 (기존 계약 유지)
    const firstOption = optionsRow.querySelector<HTMLElement>('button');
    trap = trapFocus(card, {
      initialFocus: firstOption ?? freeInput,
      onEscape: () => {
        resolve(null);
      },
    });
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
      trap?.release();
      trap = null;
      pending = null;
      card.remove();
    },
  };
}
