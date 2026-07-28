// ui/feedback/toast.ts — 앱 전역 토스트 표면 (UX_DESIGN §3.1 상태 피드백, §7 상태 표)
//
// 우상단 스택에 시맨틱 토스트를 쌓는다: 성공·오류·정보·경고. 자동 사라짐(기본 4s,
// 오류 8s)과 닫기 버튼을 제공한다. 색만으로 상태를 전달하지 않는다 — SVG 아이콘 +
// 한국어 kind 라벨(aria-label) + 메시지가 병행된다(UX_DESIGN §9).
//
// ── 실행 취소 액션 (UX_AUDIT C-4) ────────────────────────────────────
// 파괴적 동작(노드 삭제·엔티티 삭제·시퀀스 교체)에 복구 경로가 없었다. `action`을 주면
// 토스트 안에 버튼으로 렌더된다 — "노드 삭제됨 [실행 취소]". 되돌릴 수 있다는 확신은
// 기능이 아니라 **탐색의 전제조건**이다: 없으면 사용자는 시도 자체를 줄인다.
//
// 액션이 있는 토스트는 자동 사라짐이 2배로 늘어난다(resolveToastDurationMs). 4초는
// 문장을 읽고 → 되돌릴지 판단하고 → 마우스를 옮기기에 부족하고, 그 사이에 사라지면
// 복구 경로가 있었다는 사실 자체가 전달되지 않는다.
//
// 계층 규칙(CLAUDE.md §3): core/planner를 import하지 않는다. 시각 토큰은 ui/theme.ts만
// 소비한다(ad-hoc 색 금지). 순수 헬퍼(resolveToastDurationMs·kindLabelKo·toastAriaLabel)는
// DOM 없이 node 환경에서 단위 테스트된다(toast.test.ts).
//
// 이 모듈이 앱 전역 토스트 서피스가 된다. 기존 ad-hoc 토스트(scene-controls 등)의
// 통합/이관은 글루(main.ts) 몫이며 이 모듈은 그것들을 건드리지 않는다.

import {
  BORDER,
  BORDER_WIDTH,
  COLLISION,
  COLOR,
  ICON,
  LAYOUT,
  MOTION,
  RADIUS,
  SHADOW,
  SPACE,
  SURFACE,
  TYPE,
  Z_INDEX,
  applyType,
  ensureThemeStyles,
  styled,
  tr,
} from '../theme';
import { icon, makeIconButton } from '../icons';
import type { IconName } from '../icons';

// ── 공개 타입 ───────────────────────────────────────────────────────

export type ToastKind = 'success' | 'error' | 'info' | 'warn';

/** 토스트 안의 단일 액션 (실행 취소 등) — 키보드로 도달 가능한 <button>으로 렌더된다 */
export interface ToastAction {
  readonly label: string;
  /** 아이콘 (기본 undo — 대표 용례가 실행 취소다). 재시도 등은 'refresh'를 준다 */
  readonly icon?: IconName;
  onClick(): void;
}

export interface ToastOptions {
  /** 메시지 아래 보조 설명(작은 글씨) */
  detail?: string;
  /**
   * 자동 사라짐(ms). 미지정 시 kind 기본값(오류 8s, 그 외 4s) —
   * **액션이 있으면 그 2배**(파일 헤더). 0 이하면 자동으로 사라지지 않는다.
   */
  durationMs?: number;
  /**
   * 복구 액션 (UX_AUDIT C-4). 예: `{ label: '실행 취소', onClick: () => history.undo() }`.
   * 클릭하면 콜백 실행 후 토스트가 닫힌다.
   */
  action?: ToastAction;
}

export interface ToastHandle {
  show(kind: ToastKind, message: string, opts?: ToastOptions): void;
  dispose(): void;
}

// ── 상수 (매직넘버 금지 — CLAUDE.md §4, 시각 토큰은 ui/theme.ts) ────

/** 기본 자동 사라짐(ms) */
export const TOAST_DURATION_DEFAULT_MS = 4000;
/** 오류 토스트 자동 사라짐(ms) — 사용자가 사유를 읽을 시간을 더 준다 */
export const TOAST_DURATION_ERROR_MS = 8000;
/** 액션이 있는 토스트의 시간 배수 — 읽고·판단하고·조준하는 3단계가 필요하다 */
export const TOAST_ACTION_DURATION_MULTIPLIER = 2;
/** 스택에 동시에 표시하는 최대 토스트 수 — 초과 시 가장 오래된 것을 제거 */
const MAX_TOASTS = 5;
/** 진입/이탈 슬라이드 거리 */
const TOAST_SLIDE_PX = 12;
/** 사라짐 애니메이션 후 실제 제거까지 여유(ms) — MOTION.exit보다 넉넉하게 */
const TOAST_REMOVE_DELAY_MS = 180;
/** 스택 우측 여백 */
const TOAST_RIGHT_PX = 10;

// ── kind 메타 (시각/의미 매핑 — 색은 보조 채널) ─────────────────────

interface ToastKindMeta {
  /** SVG 아이콘 이름 — 이모지/딩벳이 아니라 currentColor를 받는 도형이다 (C-13) */
  readonly icon: IconName;
  /** 좌측 스트라이프/아이콘 기본색 */
  readonly base: string;
  /** 메시지 강조/아이콘 텍스트색 (패널 대비 ≥ 4.5:1인 밝은 변형) */
  readonly text: string;
  /** SR 라벨 접두 + 굳이 색 없이도 의미가 전달되게 하는 한국어 kind 라벨 */
  readonly labelKo: string;
  /** 스크린리더 우선순위: 오류만 assertive(alert) */
  readonly role: 'status' | 'alert';
}

const TOAST_KIND_META: Readonly<Record<ToastKind, ToastKindMeta>> = {
  success: { icon: 'check', base: COLOR.success, text: COLOR.successText, labelKo: '성공', role: 'status' },
  error: { icon: 'alert', base: COLLISION.base, text: COLLISION.text, labelKo: '오류', role: 'alert' },
  warn: { icon: 'alert', base: COLOR.warn, text: COLOR.warnText, labelKo: '경고', role: 'status' },
  info: { icon: 'info', base: COLOR.info, text: COLOR.infoText, labelKo: '정보', role: 'status' },
} as const;

// ── 순수 헬퍼 (DOM 비의존 — node 테스트 대상) ───────────────────────

/** kind → 한국어 라벨(색 없이 의미 전달용 — SR·aria에 사용) */
export function kindLabelKo(kind: ToastKind): string {
  return TOAST_KIND_META[kind].labelKo;
}

/**
 * 자동 사라짐 시간 결정:
 * - opts.durationMs가 숫자면 그대로(0 이하 = 고정)
 * - 아니면 kind 기본값(오류 8s / 그 외 4s), **액션이 있으면 ×2**
 */
export function resolveToastDurationMs(kind: ToastKind, opts?: ToastOptions): number {
  if (opts && typeof opts.durationMs === 'number') return opts.durationMs;
  const base = kind === 'error' ? TOAST_DURATION_ERROR_MS : TOAST_DURATION_DEFAULT_MS;
  return opts?.action ? base * TOAST_ACTION_DURATION_MULTIPLIER : base;
}

/**
 * 토스트의 접근 가능한 이름. role=status/alert 컨테이너의 aria-label이 내용을 대체하므로
 * **액션의 존재도 이름에 담아야** 스크린리더 사용자가 복구 경로가 있음을 안다.
 */
export function toastAriaLabel(kind: ToastKind, message: string, actionLabel?: string): string {
  const head = `${kindLabelKo(kind)}: ${message}`;
  return actionLabel !== undefined && actionLabel !== ''
    ? `${head} — '${actionLabel}' 버튼 있음`
    : head;
}

// ── 마운트 ──────────────────────────────────────────────────────────

/**
 * 우상단 토스트 스택을 host(보통 document.body)에 마운트한다. 컨테이너 자체는
 * pointer-events:none(토스트 사이 빈틈이 뷰포트 orbit을 막지 않게)이고, 각 토스트
 * 카드만 pointer-events:auto다.
 */
export function mountToasts(host: HTMLElement): ToastHandle {
  ensureThemeStyles();
  const stack = styled(document.createElement('div'), {
    position: 'fixed',
    top: `${LAYOUT.belowBarTopPx}px`,
    right: `${TOAST_RIGHT_PX}px`,
    zIndex: Z_INDEX.toast,
    display: 'flex',
    flexDirection: 'column',
    gap: SPACE.md,
    maxWidth: 'min(400px, 88vw)',
    pointerEvents: 'none',
  });
  stack.dataset.testid = 'toast-stack';
  stack.setAttribute('aria-live', 'polite');
  host.appendChild(stack);

  /** 활성 타이머 — dispose에서 일괄 정리 */
  const timers = new Set<ReturnType<typeof setTimeout>>();

  const dismiss = (card: HTMLElement): void => {
    if (card.dataset.dismissing === '1') return;
    card.dataset.dismissing = '1';
    card.style.opacity = '0';
    card.style.transform = `translateX(${TOAST_SLIDE_PX}px)`;
    const removeTimer = setTimeout(() => {
      timers.delete(removeTimer);
      card.remove();
    }, TOAST_REMOVE_DELAY_MS);
    timers.add(removeTimer);
  };

  const show = (kind: ToastKind, message: string, opts?: ToastOptions): void => {
    const meta = TOAST_KIND_META[kind];

    const card = applyType(document.createElement('div'), TYPE.body);
    styled(card, {
      display: 'flex',
      alignItems: 'flex-start',
      gap: SPACE.md,
      minWidth: '0',
      background: SURFACE.overlay,
      border: `${BORDER_WIDTH.hair} solid ${BORDER.default}`,
      borderLeft: `3px solid ${meta.base}`,
      borderRadius: RADIUS.md,
      boxShadow: SHADOW.overlay,
      padding: `${SPACE.md} ${SPACE.md} ${SPACE.md} ${SPACE.lg}`,
      color: COLOR.text,
      pointerEvents: 'auto',
      opacity: '0',
      transform: `translateX(${TOAST_SLIDE_PX}px)`,
      // 이탈은 항상 진입보다 빠르다 (MOTION 규약)
      transition: `${tr('opacity', MOTION.fast)}, ${tr('transform', MOTION.fast)}`,
    });
    card.dataset.testid = `toast-${kind}`;
    card.setAttribute('role', meta.role);
    // 색 없이도 kind가 전달되도록 SR 라벨에 한국어 kind 접두를 붙인다 (UX_DESIGN §9)
    card.setAttribute('aria-label', toastAriaLabel(kind, message, opts?.action?.label));
    // 토스트 위 상호작용이 뷰포트(OrbitControls)로 새지 않게 차단 (dock/패널 규약)
    for (const type of ['pointerdown', 'pointermove', 'pointerup', 'wheel']) {
      card.addEventListener(type, (e) => {
        e.stopPropagation();
      });
    }

    const iconWrap = styled(document.createElement('span'), {
      flexShrink: '0',
      display: 'flex',
      alignItems: 'center',
      // 아이콘은 currentColor를 상속한다 — 색은 이 래퍼가 정한다 (icons.ts 규약)
      color: meta.text,
      paddingTop: SPACE.xxs,
    });
    iconWrap.appendChild(icon(meta.icon, ICON.lg));
    iconWrap.setAttribute('aria-hidden', 'true');

    const body = styled(document.createElement('div'), {
      flex: '1',
      minWidth: '0',
    });
    const messageEl = applyType(document.createElement('div'), TYPE.bodyStrong);
    styled(messageEl, { color: COLOR.textStrong, wordBreak: 'break-word' });
    messageEl.textContent = message;
    body.appendChild(messageEl);
    if (opts?.detail) {
      const detailEl = applyType(document.createElement('div'), TYPE.caption);
      styled(detailEl, {
        marginTop: SPACE.xs,
        color: COLOR.muted,
        wordBreak: 'break-word',
      });
      detailEl.textContent = opts.detail;
      body.appendChild(detailEl);
    }

    // 액션 버튼 (실행 취소 등) — <button>이라 Tab으로 도달하고 Enter/Space로 활성화된다.
    // role=status 컨테이너 안이어도 위젯 탐색으로 잡히고, aria-label에도 존재가 명시된다.
    if (opts?.action) {
      const action = opts.action;
      const actionRow = styled(document.createElement('div'), {
        display: 'flex',
        marginTop: SPACE.md,
      });
      const actionButton = makeIconButton(
        action.icon ?? 'undo',
        action.label,
        `${action.label} — ${message}`,
        'toast-action',
        'accent',
      );
      actionButton.addEventListener('click', () => {
        action.onClick();
        dismiss(card);
      });
      actionRow.appendChild(actionButton);
      body.appendChild(actionRow);
    }

    const closeButton = makeIconButton('close', '', '닫기', 'toast-close', 'ghost');
    styled(closeButton, { flexShrink: '0', padding: `0 ${SPACE.sm}` });
    closeButton.addEventListener('click', () => {
      dismiss(card);
    });

    card.appendChild(iconWrap);
    card.appendChild(body);
    card.appendChild(closeButton);
    stack.appendChild(card);

    // 스택 상한 초과분(가장 오래된 것)부터 제거
    while (stack.childElementCount > MAX_TOASTS && stack.firstElementChild) {
      const oldest = stack.firstElementChild;
      if (oldest instanceof HTMLElement && oldest !== card) dismiss(oldest);
      else break;
    }

    // 진입 애니메이션 (다음 프레임에 opacity/transform 복귀)
    requestAnimationFrame(() => {
      card.style.opacity = '1';
      card.style.transform = 'translateX(0)';
    });

    const durationMs = resolveToastDurationMs(kind, opts);
    if (durationMs > 0) {
      const autoTimer = setTimeout(() => {
        timers.delete(autoTimer);
        dismiss(card);
      }, durationMs);
      timers.add(autoTimer);
    }
  };

  return {
    show,
    dispose: (): void => {
      for (const timer of timers) clearTimeout(timer);
      timers.clear();
      stack.remove();
    },
  };
}
