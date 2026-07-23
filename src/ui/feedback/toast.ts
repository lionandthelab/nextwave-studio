// ui/feedback/toast.ts — 앱 전역 토스트 표면 (UX_DESIGN §3.1 상태 피드백, §7 상태 표)
//
// 우상단 스택에 시맨틱 토스트를 쌓는다: 성공(초록 체크)·오류(빨강)·정보·경고.
// 자동 사라짐(기본 4s, 오류 8s)과 닫기 버튼을 제공한다. 색만으로 상태를 전달하지
// 않는다 — 아이콘 + 한국어 kind 라벨(aria-label) + 메시지가 병행된다(UX_DESIGN §9).
//
// 계층 규칙(CLAUDE.md §3): core/planner를 import하지 않는다. 시각 토큰은 ui/theme.ts만
// 소비한다(ad-hoc 색 금지). 순수 헬퍼(resolveToastDurationMs·kindLabelKo)는 DOM 없이
// node 환경에서 단위 테스트된다(toast.test.ts).
//
// 이 모듈이 앱 전역 토스트 서피스가 된다. 기존 ad-hoc 토스트(scene-controls 등)의
// 통합/이관은 글루(main.ts) 몫이며 이 모듈은 그것들을 건드리지 않는다.

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

export type ToastKind = 'success' | 'error' | 'info' | 'warn';

export interface ToastOptions {
  /** 메시지 아래 보조 설명(작은 글씨) */
  detail?: string;
  /**
   * 자동 사라짐(ms). 미지정 시 kind 기본값(오류 8s, 그 외 4s).
   * 0 이하면 자동으로 사라지지 않는다(사용자가 닫기 버튼으로 닫음).
   */
  durationMs?: number;
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
/** 스택에 동시에 표시하는 최대 토스트 수 — 초과 시 가장 오래된 것을 제거 */
const MAX_TOASTS = 5;
/** 슬라이드-인/아웃 트랜지션 */
const TOAST_TRANSITION = 'opacity 0.16s ease, transform 0.16s ease';
/** 사라짐 애니메이션 후 실제 제거까지 여유(ms) */
const TOAST_REMOVE_DELAY_MS = 180;

// ── kind 메타 (시각/의미 매핑 — 색은 보조 채널) ─────────────────────

interface ToastKindMeta {
  readonly icon: string;
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
  success: { icon: '✓', base: COLOR.success, text: COLOR.successText, labelKo: '성공', role: 'status' },
  error: { icon: '⚠', base: COLOR.error, text: COLOR.errorText, labelKo: '오류', role: 'alert' },
  warn: { icon: '⚠', base: COLOR.warn, text: COLOR.warn, labelKo: '경고', role: 'status' },
  info: { icon: 'ℹ', base: COLOR.info, text: COLOR.info, labelKo: '정보', role: 'status' },
} as const;

// ── 순수 헬퍼 (DOM 비의존 — node 테스트 대상) ───────────────────────

/** kind → 한국어 라벨(색 없이 의미 전달용 — SR·aria에 사용) */
export function kindLabelKo(kind: ToastKind): string {
  return TOAST_KIND_META[kind].labelKo;
}

/**
 * 자동 사라짐 시간 결정: opts.durationMs가 숫자면 그대로(0 이하 = 고정), 없으면
 * kind 기본값(오류 8s / 그 외 4s).
 */
export function resolveToastDurationMs(kind: ToastKind, opts?: ToastOptions): number {
  if (opts && typeof opts.durationMs === 'number') return opts.durationMs;
  return kind === 'error' ? TOAST_DURATION_ERROR_MS : TOAST_DURATION_DEFAULT_MS;
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
    right: '10px',
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
    card.style.transform = 'translateX(12px)';
    const removeTimer = setTimeout(() => {
      timers.delete(removeTimer);
      card.remove();
    }, TOAST_REMOVE_DELAY_MS);
    timers.add(removeTimer);
  };

  const show = (kind: ToastKind, message: string, opts?: ToastOptions): void => {
    const meta = TOAST_KIND_META[kind];

    const card = styled(document.createElement('div'), {
      display: 'flex',
      alignItems: 'flex-start',
      gap: SPACE.md,
      minWidth: '0',
      background: COLOR.bgPanel,
      border: `1px solid ${COLOR.border}`,
      borderLeft: `3px solid ${meta.base}`,
      borderRadius: RADIUS.md,
      boxShadow: SHADOW.panel,
      padding: `${SPACE.md} ${SPACE.md} ${SPACE.md} ${SPACE.lg}`,
      color: COLOR.text,
      fontFamily: FONT.ui,
      fontSize: '12px',
      lineHeight: '1.55',
      pointerEvents: 'auto',
      opacity: '0',
      transform: 'translateX(12px)',
      transition: TOAST_TRANSITION,
    });
    card.dataset.testid = `toast-${kind}`;
    card.setAttribute('role', meta.role);
    // 색 없이도 kind가 전달되도록 SR 라벨에 한국어 kind 접두를 붙인다 (UX_DESIGN §9)
    card.setAttribute('aria-label', `${meta.labelKo}: ${message}`);
    // 토스트 위 상호작용이 뷰포트(OrbitControls)로 새지 않게 차단 (dock/패널 규약)
    for (const type of ['pointerdown', 'pointermove', 'pointerup', 'wheel']) {
      card.addEventListener(type, (e) => {
        e.stopPropagation();
      });
    }

    const icon = styled(document.createElement('span'), {
      flexShrink: '0',
      color: meta.text,
      fontSize: '13px',
      lineHeight: '1.5',
    });
    icon.textContent = meta.icon;
    icon.setAttribute('aria-hidden', 'true');

    const body = styled(document.createElement('div'), {
      flex: '1',
      minWidth: '0',
    });
    const messageEl = styled(document.createElement('div'), {
      color: COLOR.textStrong,
      wordBreak: 'break-word',
    });
    messageEl.textContent = message;
    body.appendChild(messageEl);
    if (opts?.detail) {
      const detailEl = styled(document.createElement('div'), {
        marginTop: SPACE.xs,
        color: COLOR.muted,
        fontSize: '11px',
        wordBreak: 'break-word',
      });
      detailEl.textContent = opts.detail;
      body.appendChild(detailEl);
    }

    const closeButton = makeButton('✕', '닫기', 'toast-close', 'ghost');
    closeButton.style.flexShrink = '0';
    closeButton.style.padding = '0 6px';
    closeButton.addEventListener('click', () => {
      dismiss(card);
    });

    card.appendChild(icon);
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
