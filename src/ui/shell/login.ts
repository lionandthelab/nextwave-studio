// ui/shell/login.ts — 로그인 / 최초 설정 화면 (docs/BACKEND.md §3)
//
// 1차 사용자는 로봇 설치기사다: 공유 단말 · 장갑 낀 손 · 서두름. 그래서
//   - 사용자 타일(아바타 48px+) → PIN 패드(키 ≥56px) 2단 흐름 — 타이핑이 없다
//   - 오류는 흔들림 애니메이션 대신 **텍스트 + role="alert"** (모션 민감성 + SR 병행)
//   - 423 잠금은 카운트다운으로 정직하게 보여준다 — 몇 초 남았는지 모르면 계속 누른다
//   - 서버가 없어도 막다른 길이 아니다: "서버 없이 로컬 모드로 계속" 경로 상시 제공
//
// 뷰 흐름(BACKEND §3): bootstrap → needsSetup이면 설정 마법사(이름 → PIN → PIN 확인),
// 아니면 사용자 타일 → PIN 패드 → login. 서버 불통이면 재시도 + 로컬 모드 안내.
//
// 계층 규칙: api 계층의 discriminated union만 소비한다(throw 없음). 시각은 theme 토큰과
// console/primitives만 쓴다. 전역 keydown을 걸지 않는다 — 패드는 버튼이라 키보드로도
// 자연히 동작한다. 순수 로직(pinAppend/pinBackspace/lockoutRemainingSec/initialOf)은
// DOM 없이 node 테스트된다(shell.test.ts).

import type { ApiResult, LoginResult } from '../../api';
import type { UserInfo, UserRole } from '../../schema/entities';
import { makeIconButton } from '../icons';
import {
  BORDER,
  BORDER_WIDTH,
  COLLISION,
  COLOR,
  MOTION,
  RADIUS,
  SPACE,
  SURFACE,
  TYPE,
  applyType,
  ensureThemeStyles,
  makeButton,
  styled,
  tr,
} from '../theme';
import {
  applyTouchTarget,
  ensureConsoleStyles,
  makeBadge,
  makeEmptyState,
} from '../console/primitives';
import { PRODUCT_NAME } from '../brand';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4) ─────────────────────────────

/** PIN 최대 자릿수 (schema/entities.ts pinSchema \d{4,8}와 짝) */
export const PIN_MAX_LEN = 8;
/** 로그인 최소 자릿수 (tech 4자리 허용 — BACKEND §3) */
export const LOGIN_PIN_MIN_LEN = 4;
/** 최초 설정(관리자) 최소 자릿수 — 서버가 6자리+를 강제한다 */
export const SETUP_PIN_MIN_LEN = 6;
/** PIN 패드 키 한 변 하한 — 장갑 낀 손 전제(임무 명세 ≥56px) */
export const PIN_PAD_KEY_MIN_PX = 56;
/** 잠금 카운트다운 갱신 주기 */
export const LOCKOUT_TICK_MS = 1000;
/** 사용자 타일 아바타 지름 (임무 명세 48px+) */
export const AVATAR_SIZE_PX = 48;
/** PIN 도트 지름 */
const DOT_SIZE_PX = 10;
/** 카드 폭 */
const CARD_WIDTH_PX = 420;
/** 타일 최소 폭 (auto-fill 기준) */
const TILE_MIN_WIDTH_PX = 150;

// ── 순수 헬퍼 (DOM 비의존 — node 테스트 대상) ───────────────────────

/** PIN 패드 키 12개 — 3×4 배열: 1-9 · 지우기 · 0 · 백스페이스 (임무 명세 고정) */
export type PadKey =
  | { readonly kind: 'digit'; readonly value: string }
  | { readonly kind: 'clear' }
  | { readonly kind: 'backspace' };

export const PIN_PAD_LAYOUT: readonly PadKey[] = [
  { kind: 'digit', value: '1' },
  { kind: 'digit', value: '2' },
  { kind: 'digit', value: '3' },
  { kind: 'digit', value: '4' },
  { kind: 'digit', value: '5' },
  { kind: 'digit', value: '6' },
  { kind: 'digit', value: '7' },
  { kind: 'digit', value: '8' },
  { kind: 'digit', value: '9' },
  { kind: 'clear' },
  { kind: 'digit', value: '0' },
  { kind: 'backspace' },
];

/** 자리 추가 — 숫자 1글자만, maxLen 초과 입력은 조용히 무시(도트가 안 늘면 충분) */
export function pinAppend(pin: string, digit: string, maxLen: number = PIN_MAX_LEN): string {
  if (!/^\d$/.test(digit)) return pin;
  if (pin.length >= maxLen) return pin;
  return pin + digit;
}

/** 마지막 한 자리 삭제 — 빈 문자열이면 그대로 */
export function pinBackspace(pin: string): string {
  return pin.slice(0, -1);
}

/** 잠금 해제까지 남은 초 (올림, 하한 0) — 카운트다운 표시용 */
export function lockoutRemainingSec(lockedUntilMs: number, nowMs: number): number {
  return Math.max(0, Math.ceil((lockedUntilMs - nowMs) / 1000));
}

export function lockoutMessageKo(remainSec: number): string {
  return `잠금됨 — ${remainSec}초 후 다시 시도할 수 있습니다`;
}

/** 아바타 이니셜 — 첫 코드포인트(한글/이모지 안전) 대문자. 빈 이름은 '?' */
export function initialOf(name: string): string {
  const first = [...name.trim()][0];
  return first === undefined ? '?' : first.toUpperCase();
}

export function roleLabelKo(role: UserRole): string {
  return role === 'admin' ? '관리자' : '설치기사';
}

// ── 스타일 주입 (1회 — theme/primitives와 같은 패턴) ────────────────

const LOGIN_STYLE_ID = 'rsw-login-styles';

function ensureLoginStyles(): void {
  if (document.getElementById(LOGIN_STYLE_ID) !== null) return;
  const style = document.createElement('style');
  style.id = LOGIN_STYLE_ID;
  style.textContent = `
.rsw-login {
  width: 100%;
  height: 100%;
  box-sizing: border-box;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: ${SPACE.xl};
  padding: ${SPACE.xl};
  background: ${SURFACE.base};
  overflow: auto;
}
.rsw-login__card {
  width: ${CARD_WIDTH_PX}px;
  max-width: 100%;
  box-sizing: border-box;
  display: flex;
  flex-direction: column;
  gap: ${SPACE.lg};
  background: ${SURFACE.panel};
  border: ${BORDER_WIDTH.hair} solid ${BORDER.default};
  border-radius: ${RADIUS.lg};
  padding: ${SPACE.xxl};
}
.rsw-login__error {
  min-height: ${TYPE.body.lineHeightPx}px;
  color: ${COLLISION.text};
}
.rsw-login-pad {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: ${SPACE.md};
}
.ui-btn.rsw-login-key {
  min-width: ${PIN_PAD_KEY_MIN_PX}px;
  min-height: ${PIN_PAD_KEY_MIN_PX}px;
  justify-content: center;
  font-size: ${TYPE.title.sizePx}px;
  line-height: ${TYPE.title.lineHeightPx}px;
  font-weight: ${TYPE.title.weight};
}
.rsw-login-dots {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: ${SPACE.sm};
  min-height: ${DOT_SIZE_PX * 2}px;
}
.rsw-login-dot {
  width: ${DOT_SIZE_PX}px;
  height: ${DOT_SIZE_PX}px;
  border-radius: ${RADIUS.full};
  background: ${COLOR.textStrong};
}
.rsw-login-tiles {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(${TILE_MIN_WIDTH_PX}px, 1fr));
  gap: ${SPACE.lg};
}
.rsw-login-tile {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: ${SPACE.sm};
  padding: ${SPACE.lg};
  background: ${SURFACE.raised};
  border: ${BORDER_WIDTH.hair} solid ${BORDER.strong};
  border-radius: ${RADIUS.md};
  color: ${COLOR.text};
  font-family: var(--rsw-font-ui);
  cursor: pointer;
  transition: ${tr('border-color', MOTION.instant)}, ${tr('background-color', MOTION.instant)};
}
.rsw-login-tile:hover {
  border-color: ${BORDER.hover};
  background: ${SURFACE.overlay};
}
.rsw-login-tile:focus-visible {
  outline: 2px solid var(--rsw-accent);
  outline-offset: 1px;
}
.rsw-login-avatar {
  width: ${AVATAR_SIZE_PX}px;
  height: ${AVATAR_SIZE_PX}px;
  border-radius: ${RADIUS.full};
  display: grid;
  place-content: center;
  background: ${COLOR.accentSoft};
  color: ${COLOR.accentText};
  font-size: ${TYPE.title.sizePx}px;
  font-weight: ${TYPE.title.weight};
}
@media (forced-colors: active) {
  .rsw-login-tile { border-color: ButtonBorder; }
  .rsw-login-dot { forced-color-adjust: none; }
}
`;
  document.head.appendChild(style);
}

// ── PIN 패드 위젯 (내부) ────────────────────────────────────────────

interface PinPadHandle {
  readonly el: HTMLElement;
  readonly firstKey: HTMLButtonElement;
  getPin(): string;
  clear(): void;
  setDisabled(disabled: boolean): void;
}

function makePinPad(opts: {
  readonly maxLen: number;
  readonly testidPrefix: string;
  onChange(pin: string): void;
}): PinPadHandle {
  const el = styled(document.createElement('div'), {
    display: 'flex',
    flexDirection: 'column',
    gap: SPACE.lg,
  });

  // 도트 표시 — 시각은 aria-hidden, 자릿수는 sr-only role=status가 발화한다
  const dots = document.createElement('div');
  dots.className = 'rsw-login-dots';
  dots.setAttribute('aria-hidden', 'true');
  el.appendChild(dots);

  const srCount = document.createElement('span');
  srCount.className = 'sr-only';
  srCount.setAttribute('role', 'status');
  el.appendChild(srCount);

  const grid = document.createElement('div');
  grid.className = 'rsw-login-pad';
  grid.setAttribute('aria-label', 'PIN 패드');
  el.appendChild(grid);

  let pin = '';
  const buttons: HTMLButtonElement[] = [];

  const paint = (): void => {
    dots.textContent = '';
    for (let i = 0; i < pin.length; i += 1) {
      const dot = document.createElement('span');
      dot.className = 'rsw-login-dot';
      dots.appendChild(dot);
    }
    srCount.textContent = pin.length === 0 ? 'PIN 비어 있음' : `PIN ${pin.length}자리 입력됨`;
  };

  const setPin = (next: string): void => {
    if (next === pin) return;
    pin = next;
    paint();
    opts.onChange(pin);
  };

  let firstKey: HTMLButtonElement | null = null;
  for (const key of PIN_PAD_LAYOUT) {
    let button: HTMLButtonElement;
    if (key.kind === 'digit') {
      button = makeButton(key.value, key.value, `${opts.testidPrefix}-key-${key.value}`);
      button.addEventListener('click', () => setPin(pinAppend(pin, key.value, opts.maxLen)));
      if (firstKey === null) firstKey = button;
    } else if (key.kind === 'clear') {
      button = makeButton('지우기', '모두 지우기', `${opts.testidPrefix}-key-clear`, 'ghost');
      button.addEventListener('click', () => setPin(''));
    } else {
      button = makeIconButton(
        'backspace',
        '',
        '한 자리 지우기',
        `${opts.testidPrefix}-key-backspace`,
        'ghost',
      );
      button.addEventListener('click', () => setPin(pinBackspace(pin)));
    }
    button.classList.add('rsw-login-key');
    buttons.push(button);
    grid.appendChild(button);
  }
  paint();

  // PIN_PAD_LAYOUT은 digit을 반드시 포함한다 — 방어적 폴백일 뿐이다
  const first = firstKey ?? buttons[0];
  if (first === undefined) throw new Error('PIN 패드 키가 없습니다');

  return {
    el,
    firstKey: first,
    getPin: () => pin,
    clear: (): void => {
      setPin('');
    },
    setDisabled: (disabled: boolean): void => {
      for (const b of buttons) b.disabled = disabled;
    },
  };
}

// ── 공개 계약 ───────────────────────────────────────────────────────

/** api 의존 — ApiClient의 인증 표면 부분집합 (통합자가 ApiClient를 그대로 넘겨도 맞는다) */
export interface LoginApi {
  bootstrap(): Promise<ApiResult<{ needsSetup: boolean; serverName: string }>>;
  loginUsers(): Promise<ApiResult<{ users: UserInfo[] }>>;
  setup(input: { name: string; pin: string }): Promise<LoginResult>;
  login(input: { userId: string; pin: string }): Promise<LoginResult>;
}

export interface LoginDeps {
  readonly api: LoginApi;
  onLoggedIn(user: UserInfo): void;
  /** "서버 없이 로컬 모드로 계속" — 라우팅/모드 전환은 통합자 몫 */
  onLocalMode(): void;
  /** 시각 주입점 (잠금 카운트다운) — 기본 Date.now */
  readonly nowMs?: () => number;
}

export interface LoginHandle {
  /** bootstrap부터 다시 — 서버 상태가 바뀌었을 때(재연결 등) */
  refresh(): void;
  dispose(): void;
}

// ── 마운트 ──────────────────────────────────────────────────────────

export function mountLogin(host: HTMLElement, deps: LoginDeps): LoginHandle {
  ensureThemeStyles();
  ensureConsoleStyles();
  ensureLoginStyles();
  const nowMs = deps.nowMs ?? ((): number => Date.now());

  const root = document.createElement('div');
  root.className = 'rsw-login';
  root.dataset.testid = 'login-screen';

  // 워드마크 — 제품명은 brand.ts 단일 진실, 영문이라 lang="en" (CLAUDE.md §4-b)
  const header = styled(document.createElement('div'), {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: SPACE.xs,
  });
  const wordmark = applyType(document.createElement('h1'), TYPE.display);
  styled(wordmark, { margin: '0', color: COLOR.textStrong });
  wordmark.setAttribute('lang', 'en');
  wordmark.textContent = PRODUCT_NAME;
  wordmark.dataset.screenTitle = 'true';
  wordmark.tabIndex = -1;
  header.appendChild(wordmark);
  const headerCaption = applyType(document.createElement('div'), TYPE.caption);
  styled(headerCaption, { color: COLOR.muted });
  headerCaption.textContent = '현장 콘솔 로그인';
  header.appendChild(headerCaption);
  root.appendChild(header);

  const card = document.createElement('div');
  card.className = 'rsw-login__card';
  root.appendChild(card);

  // 로컬 모드 탈출구 — 어떤 뷰에서도 막다른 길이 없다 (BACKEND §6)
  const footer = styled(document.createElement('div'), {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: SPACE.xs,
  });
  const localButton = makeButton(
    '서버 없이 로컬 모드로 계속',
    '서버 없이 로컬 모드로 계속',
    'login-local-mode',
    'ghost',
  );
  applyTouchTarget(localButton);
  localButton.addEventListener('click', () => deps.onLocalMode());
  footer.appendChild(localButton);
  const localCaption = applyType(document.createElement('div'), TYPE.caption);
  styled(localCaption, { color: COLOR.muted, textAlign: 'center' });
  localCaption.textContent =
    '작업물은 이 기기 브라우저에만 저장됩니다 — 서버가 연결되면 나중에 올릴 수 있습니다.';
  footer.appendChild(localCaption);
  root.appendChild(footer);

  host.appendChild(root);

  // ── 상태 ──────────────────────────────────────────────────────────
  let epoch = 0; // refresh/dispose 시 증가 — 늦게 도착한 응답을 무시한다
  let timerId: number | null = null;
  let busy = false;
  let cachedUsers: UserInfo[] = [];

  const clearTimer = (): void => {
    if (timerId !== null) {
      window.clearTimeout(timerId);
      timerId = null;
    }
  };

  const clearCard = (): void => {
    clearTimer();
    busy = false;
    card.textContent = '';
  };

  // ── 공용 빌더 ─────────────────────────────────────────────────────

  const sectionTitle = (textKo: string): HTMLElement => {
    const el = applyType(document.createElement('h2'), TYPE.subhead);
    styled(el, { margin: '0', color: COLOR.textStrong });
    el.textContent = textKo;
    return el;
  };

  const captionLine = (textKo: string): HTMLElement => {
    const el = applyType(document.createElement('p'), TYPE.caption);
    styled(el, { margin: '0', color: COLOR.muted });
    el.textContent = textKo;
    return el;
  };

  const errorLine = (testid: string): HTMLElement => {
    const el = applyType(document.createElement('div'), TYPE.body);
    el.className = 'rsw-login__error';
    el.setAttribute('role', 'alert'); // 흔들림 대신 텍스트 + aria-live (임무 명세)
    el.dataset.testid = testid;
    return el;
  };

  const makeAvatar = (name: string): HTMLElement => {
    const el = document.createElement('div');
    el.className = 'rsw-login-avatar';
    el.setAttribute('aria-hidden', 'true');
    el.textContent = initialOf(name);
    return el;
  };

  // ── 뷰: 로딩 ──────────────────────────────────────────────────────

  const renderLoading = (): void => {
    clearCard();
    card.appendChild(captionLine('서버 확인 중…'));
  };

  // ── 뷰: 서버 불통 ─────────────────────────────────────────────────

  const renderUnreachable = (messageKo: string): void => {
    clearCard();
    const retry = makeButton('다시 시도', '서버 연결 다시 시도', 'login-retry', 'primary');
    retry.addEventListener('click', () => {
      void boot();
    });
    card.appendChild(
      makeEmptyState({
        iconName: 'cloudOff',
        titleKo: '서버에 연결할 수 없습니다',
        hintKo: messageKo,
        actions: [retry],
        testid: 'login-unreachable',
      }),
    );
  };

  // ── 뷰: 최초 설정 마법사 (이름 → PIN → PIN 확인) ──────────────────

  const renderSetup = (): void => {
    let stage: 'name' | 'pin' | 'confirm' = 'name';
    let adminName = '';
    let firstPin = '';
    let stageErrorKo = '';

    const rerender = (): void => {
      clearCard();
      const errorEl = errorLine('setup-error');
      errorEl.textContent = stageErrorKo;
      stageErrorKo = '';

      if (stage === 'name') {
        card.appendChild(sectionTitle('처음 설정 — 관리자 계정 만들기'));
        card.appendChild(
          captionLine('이 서버의 첫 사용자입니다. 관리자 이름과 PIN을 정하면 시작됩니다.'),
        );
        const label = applyType(document.createElement('label'), TYPE.caption);
        styled(label, { color: COLOR.label, display: 'flex', flexDirection: 'column', gap: SPACE.xs });
        label.textContent = '이름';
        const input = document.createElement('input');
        input.type = 'text';
        input.className = 'ui-input';
        input.maxLength = 80;
        input.value = adminName;
        input.placeholder = '예: 김반장';
        input.dataset.testid = 'setup-name';
        applyTouchTarget(input);
        applyType(input, TYPE.subhead);
        label.appendChild(input);
        card.appendChild(label);
        card.appendChild(errorEl);

        const next = makeButton('다음', '다음 — PIN 입력', 'setup-next', 'primary');
        applyTouchTarget(next);
        next.disabled = input.value.trim() === '';
        input.addEventListener('input', () => {
          next.disabled = input.value.trim() === '';
        });
        next.addEventListener('click', () => {
          adminName = input.value.trim();
          if (adminName === '') return;
          stage = 'pin';
          rerender();
        });
        card.appendChild(next);
        input.focus();
        return;
      }

      // pin / confirm 공통 골격
      const isConfirm = stage === 'confirm';
      card.appendChild(
        sectionTitle(
          isConfirm ? 'PIN 확인 — 한 번 더 입력' : `관리자 PIN 입력 (${SETUP_PIN_MIN_LEN}~${PIN_MAX_LEN}자리)`,
        ),
      );
      card.appendChild(captionLine(`관리자: ${adminName}`));
      card.appendChild(errorEl);

      const submit = makeButton(
        isConfirm ? '설정 완료' : '다음',
        isConfirm ? '설정 완료' : '다음 — PIN 확인',
        isConfirm ? 'setup-submit' : 'setup-next',
        'primary',
      );
      applyTouchTarget(submit);
      submit.disabled = true;

      const pad = makePinPad({
        maxLen: PIN_MAX_LEN,
        testidPrefix: 'setup-pin',
        onChange: (pin) => {
          submit.disabled = busy || pin.length < SETUP_PIN_MIN_LEN;
        },
      });
      card.appendChild(pad.el);
      card.appendChild(submit);

      const back = makeButton('이전', '이전 단계로', 'setup-back', 'ghost');
      applyTouchTarget(back);
      back.addEventListener('click', () => {
        stage = isConfirm ? 'pin' : 'name';
        if (isConfirm) firstPin = '';
        rerender();
      });
      card.appendChild(back);

      submit.addEventListener('click', () => {
        const pin = pad.getPin();
        if (pin.length < SETUP_PIN_MIN_LEN) return;
        if (!isConfirm) {
          firstPin = pin;
          stage = 'confirm';
          rerender();
          return;
        }
        if (pin !== firstPin) {
          // 불일치 — PIN 단계부터 다시 (부분 상태를 남기지 않는다)
          firstPin = '';
          stage = 'pin';
          stageErrorKo = 'PIN이 일치하지 않습니다 — 다시 입력해 주세요';
          rerender();
          return;
        }
        void submitSetup(adminName, pin, errorEl, submit, pad);
      });
      pad.firstKey.focus();
    };

    const submitSetup = async (
      name: string,
      pin: string,
      errorEl: HTMLElement,
      submit: HTMLButtonElement,
      pad: PinPadHandle,
    ): Promise<void> => {
      if (busy) return;
      busy = true;
      submit.disabled = true;
      pad.setDisabled(true);
      const my = epoch;
      const result = await deps.api.setup({ name, pin });
      if (my !== epoch) return;
      busy = false;
      if (result.kind === 'ok') {
        deps.onLoggedIn(result.session.user);
        return;
      }
      pad.setDisabled(false);
      submit.disabled = pad.getPin().length < SETUP_PIN_MIN_LEN;
      errorEl.textContent = result.messageKo;
    };

    rerender();
  };

  // ── 뷰: 사용자 타일 ───────────────────────────────────────────────

  const renderTiles = (users: readonly UserInfo[]): void => {
    clearCard();
    if (users.length === 0) {
      const reload = makeButton('다시 불러오기', '사용자 목록 다시 불러오기', 'login-refresh');
      reload.addEventListener('click', () => {
        void boot();
      });
      card.appendChild(
        makeEmptyState({
          iconName: 'users',
          titleKo: '등록된 사용자가 없습니다',
          hintKo: '관리자에게 계정 생성을 요청하세요.',
          actions: [reload],
          testid: 'login-empty',
        }),
      );
      return;
    }
    card.appendChild(sectionTitle('사용자를 선택하세요'));
    const grid = document.createElement('div');
    grid.className = 'rsw-login-tiles';
    grid.setAttribute('aria-label', '사용자 목록');
    for (const user of users) {
      const tile = document.createElement('button');
      tile.type = 'button';
      tile.className = 'rsw-login-tile';
      tile.dataset.testid = `login-tile-${user.id}`;
      tile.setAttribute('aria-label', `${user.name} — ${roleLabelKo(user.role)}`);
      tile.appendChild(makeAvatar(user.name));
      const nameEl = applyType(document.createElement('span'), TYPE.bodyStrong);
      styled(nameEl, { color: COLOR.textStrong, maxWidth: '100%', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' });
      nameEl.textContent = user.name;
      tile.appendChild(nameEl);
      tile.appendChild(makeBadge(roleLabelKo(user.role), 'neutral'));
      tile.addEventListener('click', () => renderPad(user));
      grid.appendChild(tile);
    }
    card.appendChild(grid);
  };

  // ── 뷰: PIN 패드 (선택된 사용자) ──────────────────────────────────

  const renderPad = (user: UserInfo): void => {
    clearCard();

    const back = makeIconButton('chevronLeft', '다른 사용자', '다른 사용자 선택', 'login-back', 'ghost');
    applyTouchTarget(back);
    back.addEventListener('click', () => renderTiles(cachedUsers));
    card.appendChild(back);

    const identity = styled(document.createElement('div'), {
      display: 'flex',
      alignItems: 'center',
      gap: SPACE.md,
    });
    identity.appendChild(makeAvatar(user.name));
    const nameEl = applyType(document.createElement('span'), TYPE.subhead);
    styled(nameEl, { color: COLOR.textStrong, minWidth: '0', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' });
    nameEl.textContent = user.name;
    identity.appendChild(nameEl);
    identity.appendChild(makeBadge(roleLabelKo(user.role), 'neutral'));
    card.appendChild(identity);

    card.appendChild(captionLine('PIN을 입력하세요'));

    const errorEl = errorLine('login-error');
    card.appendChild(errorEl);

    // 카운트다운은 live 영역이 아니다 — 매초 발화되면 SR이 다른 조작을 못 한다.
    // 잠금 시작 알림은 errorEl(role=alert)이 1회 발화한다.
    const lockEl = applyType(document.createElement('div'), TYPE.caption);
    styled(lockEl, { color: COLOR.warnText, minHeight: `${TYPE.caption.lineHeightPx}px` });
    lockEl.dataset.testid = 'login-lock-remaining';
    card.appendChild(lockEl);

    const submit = makeButton('로그인', '로그인', 'login-submit', 'primary');
    applyTouchTarget(submit);
    submit.disabled = true;

    let lockedUntilMs = 0;

    const syncSubmit = (pin: string): void => {
      const locked = lockoutRemainingSec(lockedUntilMs, nowMs()) > 0;
      submit.disabled = busy || locked || pin.length < LOGIN_PIN_MIN_LEN;
    };

    const pad = makePinPad({
      maxLen: PIN_MAX_LEN,
      testidPrefix: 'login-pin',
      onChange: syncSubmit,
    });
    card.appendChild(pad.el);
    card.appendChild(submit);

    const tickLockout = (): void => {
      const remain = lockoutRemainingSec(lockedUntilMs, nowMs());
      if (remain > 0) {
        lockEl.textContent = lockoutMessageKo(remain);
        timerId = window.setTimeout(tickLockout, LOCKOUT_TICK_MS);
        return;
      }
      timerId = null;
      lockedUntilMs = 0;
      lockEl.textContent = '';
      errorEl.textContent = '다시 시도할 수 있습니다';
      pad.setDisabled(false);
      syncSubmit(pad.getPin());
    };

    const startLockout = (retryAfterSec: number, messageKo: string): void => {
      clearTimer();
      lockedUntilMs = nowMs() + retryAfterSec * 1000;
      errorEl.textContent = messageKo; // role=alert — 잠금 시작을 1회 발화
      pad.clear();
      pad.setDisabled(true);
      submit.disabled = true;
      tickLockout();
    };

    const attempt = async (): Promise<void> => {
      if (busy || lockoutRemainingSec(lockedUntilMs, nowMs()) > 0) return;
      const pin = pad.getPin();
      if (pin.length < LOGIN_PIN_MIN_LEN) return;
      busy = true;
      submit.disabled = true;
      pad.setDisabled(true);
      errorEl.textContent = '';
      const my = epoch;
      const result = await deps.api.login({ userId: user.id, pin });
      if (my !== epoch) return;
      busy = false;
      switch (result.kind) {
        case 'ok':
          deps.onLoggedIn(result.session.user);
          return;
        case 'invalid':
          pad.setDisabled(false);
          pad.clear(); // 오답은 도트를 비운다 — 몇 자리 틀렸는지 세게 하지 않는다
          errorEl.textContent = result.messageKo;
          return;
        case 'locked':
          startLockout(result.retryAfterSec, result.messageKo);
          return;
        case 'network':
        case 'error':
          pad.setDisabled(false);
          errorEl.textContent = result.messageKo;
          syncSubmit(pad.getPin());
          return;
      }
    };
    submit.addEventListener('click', () => {
      void attempt();
    });

    pad.firstKey.focus();
  };

  // ── 부트 흐름 ─────────────────────────────────────────────────────

  const loadUsers = async (my: number): Promise<void> => {
    const r = await deps.api.loginUsers();
    if (my !== epoch) return;
    if (r.kind === 'ok') {
      cachedUsers = r.data.users;
      renderTiles(cachedUsers);
      return;
    }
    renderUnreachable(r.messageKo);
  };

  const boot = async (): Promise<void> => {
    const my = ++epoch;
    renderLoading();
    const b = await deps.api.bootstrap();
    if (my !== epoch) return;
    if (b.kind === 'ok') {
      if (b.data.needsSetup) renderSetup();
      else await loadUsers(my);
      return;
    }
    renderUnreachable(b.messageKo);
  };

  void boot();

  return {
    refresh: (): void => {
      void boot();
    },
    dispose: (): void => {
      epoch += 1;
      clearTimer();
      root.remove();
    },
  };
}
