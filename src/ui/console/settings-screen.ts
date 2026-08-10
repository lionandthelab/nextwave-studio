// ui/console/settings-screen.ts — 설정 화면 (콘솔 평면 ⑨, docs/BACKEND.md Phase 12+)
//
// 흩어져 있던 설정(⚙ 플래너 다이얼로그 · 자동정지 토글 · 문서/데이터 관리)을 한 화면으로
// 모은다. 6개 섹션의 **세로 스택**이다 — 좌측 섹션 네비 없음. 설치기사(공유 단말 · 장갑 ·
// 서두름)에게 6섹션이면 스크롤이 탭 미로보다 빠르고, 어디에 무엇이 있는지 한 번의
// 스크롤로 전부 보인다.
//
//   1 내 계정        이름·역할 · PIN 변경 · 사용자 전환 · 로그아웃
//   2 사용자 관리    (admin만) 목록 표 + 역할 변경 · 비활성화/활성화 · 사용자 추가
//   3 생성 (AI)      현재 플래너 백엔드 표시 + 플래너 설정 다이얼로그 열기
//   4 실행 기본값    충돌 자동 정지 · 재생 속도 기본값
//   5 데이터         연결 상태 · 동기화(outbox) · 저장소 사용량 · 내보내기/가져오기
//   6 정보           제품 · 버전 · 서버 이름/버전
//
// ── 계층/시각 규칙 (CLAUDE.md §3 / §4-b) ────────────────────────────
// core/planner/render/main을 import하지 않는다 — deps는 좁은 인터페이스(콜백 + API
// 서브셋)뿐이고 배선은 통합자 몫이다. 시각 토큰은 ui/theme.ts, 아이콘은 ui/icons.ts,
// 공유 조립 블록은 primitives.ts만 소비한다. UI 크롬은 한국어, 도메인 식별자(모델명·
// 서버 버전 등)는 영문 원문 + lang="en"(WCAG 3.1.2).
//
// ── 서버 미연결 정직성 (BACKEND §1·§6) ──────────────────────────────
// local/offline에서도 화면은 깨지지 않는다. 서버가 필요한 동작(PIN 변경 · 사용자 관리 ·
// 동기화)은 **사유가 title로 보이는 비활성 버튼**이 된다 — 조용한 no-op 금지.
//
// ── 파괴적 동작의 되돌릴 경로 (CLAUDE.md §2.11) ─────────────────────
// 사용자 비활성화는 (a) 확인 바를 거치고 (b) 같은 행의 [활성화] 버튼이 즉시 보이는
// 되돌림 경로다 — soft토글이라 데이터 손실이 없고, 자기 자신 비활성화는 금지된다
// (마지막 관리자가 스스로를 잠그는 사고 방지).
//
// 모듈 top-level에서 DOM을 만지지 않는다 — 순수 헬퍼(PIN 검증 · 용량 포맷터 · 상태
// 매핑)는 node 환경 테스트가 import해도 안전하다(settings-screen.test.ts).

import { connectionLabelKo, lastSyncAgeKo } from '../../api';
import type { ConnectionState } from '../../api';
import { TRASH_RETENTION_DAYS } from '../../schema/entities';
import type { UserInfo, UserRole } from '../../schema/entities';
import { createAnnouncer } from '../a11y';
import { BRAND_NAME, PRODUCT_NAME } from '../brand';
import { icon, makeIconButton } from '../icons';
import type { IconName } from '../icons';
import {
  BORDER,
  BORDER_WIDTH,
  COLLISION,
  COLOR,
  ICON,
  RADIUS,
  SPACE,
  SURFACE,
  TYPE,
  applyType,
  ensureThemeStyles,
  makeButton,
  makePanelHeader,
  styled,
} from '../theme';
import type { StatusName } from '../theme';
import {
  applyTouchTarget,
  ensureConsoleStyles,
  makeBadge,
  makeConfirmBar,
  makeDataTable,
  makeEmptyState,
  makeModalShell,
} from './primitives';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4) ─────────────────────────────

/** 섹션 스택 최대 폭 — 넓은 모니터에서 행 길이가 무한정 늘어나지 않게 */
const CONTENT_MAX_WIDTH_PX = 840;
/** 정보 행 라벨 컬럼 폭 */
const INFO_LABEL_WIDTH_PX = 132;
/** 정보 행 최소 높이 — 배지 높이(BADGE_MIN_HEIGHT_PX)와 정렬 */
const INFO_ROW_MIN_HEIGHT_PX = 24;
/** 사용자 표 컬럼 폭 */
const USER_COL_ROLE_PX = 140;
const USER_COL_STATUS_PX = 96;
const USER_COL_ACTION_PX = 128;
/** 재생 속도 기본값 선택지 — processRules.speedLimitMult(1|2|4)와 같은 축 */
export const SPEED_MULT_OPTIONS = [1, 2, 4] as const;
export type SpeedMult = (typeof SPEED_MULT_OPTIONS)[number];

// ── 순수 헬퍼 (DOM 비의존 — node 테스트 대상) ───────────────────────

export interface PinRule {
  readonly minDigits: number;
  readonly maxDigits: number;
}

/** 역할별 PIN 자릿수 규칙 — 서버와 동일 (BACKEND §3: tech 4–8, admin 6–8) */
export function pinRuleForRole(role: UserRole): PinRule {
  return role === 'admin' ? { minDigits: 6, maxDigits: 8 } : { minDigits: 4, maxDigits: 8 };
}

export type ValidationResult = { readonly ok: true } | { readonly ok: false; readonly messageKo: string };

/** PIN 형식 검증 — 숫자만 + 역할별 자릿수 (pinSchema와 정합) */
export function checkPinFormat(pin: string, role: UserRole): ValidationResult {
  if (pin === '') return { ok: false, messageKo: 'PIN을 입력해 주세요' };
  if (!/^\d+$/.test(pin)) return { ok: false, messageKo: 'PIN은 숫자만 입력할 수 있습니다' };
  const rule = pinRuleForRole(role);
  if (pin.length < rule.minDigits || pin.length > rule.maxDigits) {
    return {
      ok: false,
      messageKo: `PIN은 ${rule.minDigits}~${rule.maxDigits}자리 숫자여야 합니다`,
    };
  }
  return { ok: true };
}

export interface PinChangeInput {
  readonly currentPin: string;
  readonly newPin: string;
  /** 새 PIN 재입력 (2회 입력 일치 확인) */
  readonly newPinRepeat: string;
  readonly role: UserRole;
}

/** PIN 변경 폼 검증 — 2회 불일치·자릿수·현재와 동일 여부를 서버 왕복 전에 잡는다 */
export function validatePinChange(input: PinChangeInput): ValidationResult {
  if (input.currentPin === '') return { ok: false, messageKo: '현재 PIN을 입력해 주세요' };
  const format = checkPinFormat(input.newPin, input.role);
  if (!format.ok) return format;
  if (input.newPin !== input.newPinRepeat) {
    return {
      ok: false,
      messageKo: '새 PIN 두 번이 서로 다릅니다 — 같은 번호를 다시 입력해 주세요',
    };
  }
  if (input.newPin === input.currentPin) {
    return { ok: false, messageKo: '새 PIN이 현재 PIN과 같습니다 — 다른 번호를 입력해 주세요' };
  }
  return { ok: true };
}

export interface NewUserInput {
  readonly name: string;
  readonly pin: string;
  readonly role: UserRole;
}

/** 사용자 추가 폼 검증 — displayNameSchema(공백만 금지·80자)와 정합 */
export function validateNewUser(input: NewUserInput): ValidationResult {
  if (input.name.trim() === '') return { ok: false, messageKo: '이름을 입력해 주세요' };
  if (input.name.trim().length > 80) return { ok: false, messageKo: '이름은 80자 이하여야 합니다' };
  return checkPinFormat(input.pin, input.role);
}

/**
 * 자기 자신 비활성화 금지 — 마지막 관리자가 스스로를 잠그는 사고를 막는다.
 * 로그인 정보가 없으면(null) 관리 자체가 불가하므로 false.
 */
export function canDeactivateUser(myId: string | null, targetId: string): boolean {
  return myId !== null && myId !== targetId;
}

/** 역할 한국어 라벨 (BACKEND §3: admin=관리자, tech=설치기사) */
export function roleLabelKo(role: UserRole): string {
  return role === 'admin' ? '관리자' : '설치기사';
}

/** 바이트 → 사람이 읽는 용량 (navigator.storage.estimate 표시용) */
export function formatBytesKo(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return '알 수 없음';
  if (bytes < 1024) return `${Math.round(bytes)} B`;
  const units = ['KB', 'MB', 'GB', 'TB'] as const;
  let value = bytes / 1024;
  let i = 0;
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024;
    i += 1;
  }
  const unit = units[i] ?? 'TB';
  const text = value >= 100 ? String(Math.round(value)) : value.toFixed(1);
  return `${text} ${unit}`;
}

export interface StorageEstimateInfo {
  readonly usageBytes: number;
  readonly quotaBytes: number;
}

/** 저장소 사용량 문구 — 추정 불가 환경(null)도 정직하게 말한다 */
export function formatStorageKo(estimate: StorageEstimateInfo | null): string {
  if (estimate === null) return '확인할 수 없음';
  return `${formatBytesKo(estimate.usageBytes)} 사용 (전체 ${formatBytesKo(estimate.quotaBytes)})`;
}

/** 연결 상태 → 배지 STATUS 축 매핑 (online=success, offline=warn, local=idle) */
export function connectionStatusName(state: ConnectionState): StatusName {
  if (state.mode === 'local') return 'idle';
  return state.online ? 'success' : 'warn';
}

/**
 * 서버가 필요한 동작의 차단 사유 — null이면 사용 가능.
 * 회색 버튼에는 반드시 이유가 title로 보인다(조용한 비활성 금지).
 */
export function serverBlockReasonKo(state: ConnectionState): string | null {
  if (state.mode === 'local') return '서버가 설정되지 않아 사용할 수 없습니다 (로컬 모드)';
  if (!state.online) return '오프라인 상태입니다 — 서버에 다시 연결되면 사용할 수 있습니다';
  return null;
}

/** 한글이 없는 문자열(모델명·버전 등)에만 lang="en"을 준다 (WCAG 3.1.2) */
export function textLang(text: string): 'en' | undefined {
  if (text === '') return undefined;
  return /[ㄱ-ㆎ가-힣]/.test(text) ? undefined : 'en';
}

/** 재생 속도 값 정규화 — 알 수 없는 값은 1×로 (localStorage 손상 방어) */
export function normalizeSpeedMult(value: number): SpeedMult {
  return (SPEED_MULT_OPTIONS as readonly number[]).includes(value) ? (value as SpeedMult) : 1;
}

export function speedMultLabel(mult: SpeedMult): string {
  return `${mult}×`;
}

export interface SyncReport {
  readonly sentCount: number;
  readonly conflictCount: number;
  readonly remainingCount: number;
}

/** 동기화 결과 요약 문구 — 충돌/잔여는 있을 때만 말한다 */
export function syncReportKo(report: SyncReport): string {
  if (report.sentCount === 0 && report.conflictCount === 0 && report.remainingCount === 0) {
    return '보낼 변경이 없습니다';
  }
  const parts = [`${report.sentCount}건 전송됨`];
  if (report.conflictCount > 0) parts.push(`충돌 ${report.conflictCount}건`);
  if (report.remainingCount > 0) parts.push(`대기 ${report.remainingCount}건`);
  return parts.join(' · ');
}

/** 마지막 동기화 문구 — 기록이 없으면 없다고 말한다 */
export function lastSyncTextKo(lastSyncIso: string | null, nowMs: number): string {
  if (lastSyncIso === null) return '동기화 기록 없음';
  return lastSyncAgeKo(lastSyncIso, nowMs);
}

// ── deps (좁은 인터페이스 — 배선은 통합자 몫) ───────────────────────

export type SettingsUsersListResult =
  | { readonly kind: 'ok'; readonly users: UserInfo[] }
  | { readonly kind: 'error'; readonly messageKo: string };

export type SettingsUserSaveResult =
  | { readonly kind: 'ok'; readonly user: UserInfo }
  | { readonly kind: 'error'; readonly messageKo: string };

export type ChangePinResult =
  | { readonly kind: 'ok' }
  | { readonly kind: 'error'; readonly messageKo: string };

export type SyncNowResult =
  | ({ readonly kind: 'ok' } & SyncReport)
  | { readonly kind: 'error'; readonly messageKo: string };

/** 실행 기본값 — 영속(localStorage)은 통합자 몫, 이 화면은 get/set만 안다 */
export interface ExecDefaults {
  readonly autoPauseOnCollision: boolean;
  readonly speedMult: SpeedMult;
}

export interface ServerHealthInfo {
  readonly name: string;
  readonly version: string;
}

export interface SettingsScreenDeps {
  /** 현재 로그인 사용자 — 로컬 모드/미로그인이면 null */
  me(): UserInfo | null;
  /** 사용자 관리 (admin 섹션) — GET/POST /users · PATCH /users/:id (BACKEND §4) */
  readonly users: {
    list(): Promise<SettingsUsersListResult>;
    create(input: NewUserInput): Promise<SettingsUserSaveResult>;
    patch(
      id: string,
      patch: { readonly role?: UserRole; readonly active?: boolean },
    ): Promise<SettingsUserSaveResult>;
  };
  /** 본인 PIN 변경 — PATCH /users/:id { pin } (비-admin도 본인은 허용, BACKEND §4) */
  changeMyPin(input: {
    readonly currentPin: string;
    readonly newPin: string;
  }): Promise<ChangePinResult>;
  /** 빠른 사용자 전환 — 로그인 타일 화면으로 (BACKEND §3) */
  onSwitchUser(): void;
  onLogout(): void;
  /** 현재 플래너 백엔드 요약 (예: '규칙 기반' | 'Anthropic · claude-…') */
  plannerSummary(): string;
  /** 기존 플래너 설정 다이얼로그 재사용 — 이 화면은 열기만 한다 */
  onOpenPlannerSettings(): void;
  readonly execDefaults: {
    get(): ExecDefaults;
    set(next: ExecDefaults): void;
  };
  readonly sync: {
    /** outbox 대기 건수 */
    pendingCount(): Promise<number>;
    syncNow(): Promise<SyncNowResult>;
    /** 마지막 성공 동기화 시각 — 없으면 null */
    lastSyncIso(): string | null;
  };
  /** navigator.storage.estimate 주입 — 미지원 환경은 null */
  storageEstimate(): Promise<StorageEstimateInfo | null>;
  onExportAll(): void;
  onImportAll(): void;
  /** GET /health — 실패/미연결이면 null */
  health(): Promise<ServerHealthInfo | null>;
  readonly connection: {
    get(): ConnectionState;
    /** 구독 — 반환값은 해제 함수 (ApiClient.onStateChange와 동일 계약) */
    onChange(cb: (state: ConnectionState) => void): () => void;
  };
  /** 앱(클라이언트 번들) 버전 표기 — 미주입 시 'dev' */
  readonly appVersion?: string;
}

export interface SettingsScreenHandle {
  refresh(): void;
  dispose(): void;
}

// ── 마운트 ──────────────────────────────────────────────────────────

export function mountSettingsScreen(
  host: HTMLElement,
  deps: SettingsScreenDeps,
): SettingsScreenHandle {
  ensureThemeStyles();
  ensureConsoleStyles();

  let disposed = false;
  let refreshSeq = 0;
  let conn: ConnectionState = deps.connection.get();
  let users: UserInfo[] = [];

  // ── 뼈대 ──────────────────────────────────────────────────────────

  const root = styled(document.createElement('div'), {
    height: '100%',
    overflowY: 'auto',
    boxSizing: 'border-box',
    background: SURFACE.base,
  });
  root.className = 'ui-scroll';
  root.dataset.testid = 'settings-screen';

  const column = styled(document.createElement('div'), {
    display: 'flex',
    flexDirection: 'column',
    gap: SPACE.xl,
    padding: SPACE.xl,
    maxWidth: `${CONTENT_MAX_WIDTH_PX}px`,
    margin: '0 auto',
    width: '100%',
    boxSizing: 'border-box',
  });
  root.appendChild(column);

  const heading = applyType(document.createElement('h2'), TYPE.display);
  styled(heading, { margin: '0', color: COLOR.textStrong });
  heading.textContent = '설정';
  column.appendChild(heading);

  const announcer = createAnnouncer(root);

  interface SectionParts {
    readonly el: HTMLElement;
    readonly body: HTMLElement;
    readonly actionsEl: HTMLElement | null;
  }

  const makeSection = (
    titleKo: string,
    iconName: IconName,
    testid: string,
    opts: { actions?: boolean } = {},
  ): SectionParts => {
    const el = styled(document.createElement('section'), {
      background: SURFACE.panel,
      border: `${BORDER_WIDTH.hair} solid ${BORDER.subtle}`,
      borderRadius: RADIUS.md,
      overflow: 'hidden',
    });
    el.dataset.testid = testid;

    const header = makePanelHeader(titleKo, {
      headingTag: 'h3',
      actions: opts.actions === true,
      testId: `${testid}-header`,
    });
    const iconWrap = styled(document.createElement('span'), {
      display: 'flex',
      color: COLOR.muted,
      flex: 'none',
    });
    iconWrap.appendChild(icon(iconName, ICON.lg));
    header.el.insertBefore(iconWrap, header.titleEl);
    header.titleEl.id = `${testid}-title`;
    el.setAttribute('aria-labelledby', `${testid}-title`);
    el.appendChild(header.el);

    const body = styled(document.createElement('div'), {
      display: 'flex',
      flexDirection: 'column',
      gap: SPACE.lg,
      padding: SPACE.xl,
    });
    el.appendChild(body);
    return { el, body, actionsEl: header.actionsEl };
  };

  /** 라벨 + 값 정보 행 */
  const infoRow = (labelKo: string, value: HTMLElement): HTMLElement => {
    const row = styled(document.createElement('div'), {
      display: 'flex',
      alignItems: 'center',
      gap: SPACE.lg,
      minHeight: `${INFO_ROW_MIN_HEIGHT_PX}px`,
    });
    const label = applyType(document.createElement('span'), TYPE.caption);
    styled(label, { color: COLOR.label, flex: `0 0 ${INFO_LABEL_WIDTH_PX}px` });
    label.textContent = labelKo;
    styled(value, { flex: '1 1 auto', minWidth: '0' });
    row.appendChild(label);
    row.appendChild(value);
    return row;
  };

  const hintLine = (textKo: string): HTMLElement => {
    const el = applyType(document.createElement('p'), TYPE.caption);
    styled(el, { margin: '0', color: COLOR.muted });
    el.textContent = textKo;
    return el;
  };

  /** 서버 의존 컨트롤 비활성 — 사유를 title/aria-label로 남긴다 */
  const setServerGated = (
    el: HTMLButtonElement | HTMLSelectElement,
    enabledTitle: string,
    extraReason: string | null = null,
  ): void => {
    const reason = extraReason ?? serverBlockReasonKo(conn);
    el.disabled = reason !== null;
    const title = reason ?? enabledTitle;
    el.title = title;
    el.setAttribute('aria-label', title);
  };

  // ══ 1. 내 계정 ════════════════════════════════════════════════════

  const accountSection = makeSection('내 계정', 'user', 'settings-account');

  // PIN 변경 모달 (1회 생성, 열 때 리셋)
  const pinModal = makeModalShell({
    titleKo: 'PIN 변경',
    onClose: () => {
      resetPinForm();
    },
    testid: 'settings-pin-modal',
  });
  root.appendChild(pinModal.root);

  const makePinField = (
    labelKo: string,
    testid: string,
  ): { row: HTMLElement; input: HTMLInputElement } => {
    const row = styled(document.createElement('div'), {
      display: 'flex',
      flexDirection: 'column',
      gap: SPACE.xs,
    });
    const label = applyType(document.createElement('label'), TYPE.caption);
    styled(label, { color: COLOR.label });
    label.textContent = labelKo;
    label.htmlFor = testid;
    const input = document.createElement('input');
    input.type = 'password';
    input.inputMode = 'numeric';
    input.autocomplete = 'off';
    input.maxLength = 8;
    input.className = 'ui-input';
    input.id = testid;
    input.dataset.testid = testid;
    applyTouchTarget(input);
    row.appendChild(label);
    row.appendChild(input);
    return { row, input };
  };

  const pinCurrent = makePinField('현재 PIN', 'settings-pin-current');
  const pinNew = makePinField('새 PIN', 'settings-pin-new');
  const pinNew2 = makePinField('새 PIN (다시 입력)', 'settings-pin-new2');
  const pinHint = hintLine('');
  const pinError = applyType(document.createElement('p'), TYPE.caption);
  styled(pinError, { margin: '0', color: COLLISION.text, display: 'none' });
  pinError.dataset.testid = 'settings-pin-error';
  pinModal.body.appendChild(pinCurrent.row);
  pinModal.body.appendChild(pinNew.row);
  pinModal.body.appendChild(pinNew2.row);
  pinModal.body.appendChild(pinHint);
  pinModal.body.appendChild(pinError);

  const pinCancelButton = makeButton('취소', '취소', 'settings-pin-cancel', 'ghost');
  applyTouchTarget(pinCancelButton);
  pinCancelButton.addEventListener('click', () => {
    pinModal.close();
    resetPinForm();
  });
  const pinSubmitButton = makeButton('변경', 'PIN 변경', 'settings-pin-submit', 'primary');
  applyTouchTarget(pinSubmitButton);
  pinModal.footer.appendChild(pinCancelButton);
  pinModal.footer.appendChild(pinSubmitButton);

  const pinInputs = [pinCurrent.input, pinNew.input, pinNew2.input];
  for (const input of pinInputs) {
    input.addEventListener('input', () => {
      pinModal.setDirty(pinInputs.some((i) => i.value !== ''));
      pinError.style.display = 'none';
    });
  }

  function resetPinForm(): void {
    for (const input of pinInputs) input.value = '';
    pinError.style.display = 'none';
    pinSubmitButton.disabled = false;
  }

  const showPinError = (messageKo: string): void => {
    pinError.textContent = messageKo;
    pinError.style.display = 'block';
    announcer.announceNow(`오류 — ${messageKo}`);
  };

  pinSubmitButton.addEventListener('click', () => {
    void submitPinChange();
  });

  async function submitPinChange(): Promise<void> {
    const me = deps.me();
    if (me === null) return;
    const result = validatePinChange({
      currentPin: pinCurrent.input.value,
      newPin: pinNew.input.value,
      newPinRepeat: pinNew2.input.value,
      role: me.role,
    });
    if (!result.ok) {
      showPinError(result.messageKo);
      return;
    }
    pinSubmitButton.disabled = true;
    const r = await deps.changeMyPin({
      currentPin: pinCurrent.input.value,
      newPin: pinNew.input.value,
    });
    if (disposed) return;
    pinSubmitButton.disabled = false;
    if (r.kind === 'ok') {
      pinModal.close();
      resetPinForm();
      accountStatusEl.textContent = 'PIN이 변경되었습니다';
      styled(accountStatusEl, { color: COLOR.successText });
      announcer.announceNow('PIN이 변경되었습니다');
      return;
    }
    showPinError(r.messageKo);
  }

  /** PIN 변경 성공 등 계정 섹션 상태 문구 (refresh 시 초기화) */
  let accountStatusEl = applyType(document.createElement('p'), TYPE.caption);

  const paintAccount = (): void => {
    accountSection.body.textContent = '';
    const me = deps.me();

    if (me === null) {
      const empty = applyType(document.createElement('p'), TYPE.body);
      styled(empty, { margin: '0', color: COLOR.muted });
      empty.textContent =
        conn.mode === 'local'
          ? '로그인되어 있지 않습니다 — 서버 없이 로컬 모드로 동작 중입니다'
          : '로그인되어 있지 않습니다';
      accountSection.body.appendChild(empty);

      const switchButton = makeIconButton(
        'user',
        '사용자 전환',
        '로그인 화면으로 이동',
        'settings-switch-user',
      );
      applyTouchTarget(switchButton);
      switchButton.addEventListener('click', () => deps.onSwitchUser());
      const row = styled(document.createElement('div'), {
        display: 'flex',
        flexWrap: 'wrap',
        gap: SPACE.md,
      });
      row.appendChild(switchButton);
      accountSection.body.appendChild(row);
      return;
    }

    const nameValue = applyType(document.createElement('span'), TYPE.bodyStrong);
    styled(nameValue, { color: COLOR.textStrong });
    nameValue.textContent = me.name;
    accountSection.body.appendChild(infoRow('이름', nameValue));

    const roleValue = styled(document.createElement('span'), { display: 'inline-flex' });
    roleValue.appendChild(makeBadge(roleLabelKo(me.role), 'neutral', { testid: 'settings-my-role' }));
    accountSection.body.appendChild(infoRow('역할', roleValue));

    const buttons = styled(document.createElement('div'), {
      display: 'flex',
      flexWrap: 'wrap',
      gap: SPACE.md,
    });

    const pinButton = makeIconButton('lock', 'PIN 변경', 'PIN 변경', 'settings-pin-change');
    applyTouchTarget(pinButton);
    setServerGated(pinButton, 'PIN 변경');
    pinButton.addEventListener('click', () => {
      const meNow = deps.me();
      if (meNow === null) return;
      const rule = pinRuleForRole(meNow.role);
      pinHint.textContent = `PIN은 ${rule.minDigits}~${rule.maxDigits}자리 숫자입니다`;
      resetPinForm();
      pinModal.open();
      pinCurrent.input.focus();
    });
    buttons.appendChild(pinButton);

    const switchButton = makeIconButton(
      'users',
      '사용자 전환',
      '사용자 전환 — 로그인 타일로 이동',
      'settings-switch-user',
    );
    applyTouchTarget(switchButton);
    switchButton.addEventListener('click', () => deps.onSwitchUser());
    buttons.appendChild(switchButton);

    const logoutButton = makeIconButton('logout', '로그아웃', '로그아웃', 'settings-logout');
    applyTouchTarget(logoutButton);
    logoutButton.addEventListener('click', () => deps.onLogout());
    buttons.appendChild(logoutButton);

    accountSection.body.appendChild(buttons);

    accountStatusEl = applyType(document.createElement('p'), TYPE.caption);
    styled(accountStatusEl, { margin: '0', color: COLOR.muted });
    accountStatusEl.dataset.testid = 'settings-account-status';
    accountSection.body.appendChild(accountStatusEl);
  };

  // ══ 2. 사용자 관리 (admin만) ══════════════════════════════════════

  const usersSection = makeSection('사용자 관리', 'users', 'settings-users', { actions: true });

  const userAddButton = makeIconButton('plus', '사용자 추가', '사용자 추가', 'settings-user-add');
  applyTouchTarget(userAddButton);
  usersSection.actionsEl?.appendChild(userAddButton);

  const usersError = applyType(document.createElement('p'), TYPE.caption);
  styled(usersError, { margin: '0', color: COLLISION.text, display: 'none' });
  usersError.dataset.testid = 'settings-users-error';

  const makeRoleSelect = (row: UserInfo): HTMLSelectElement => {
    const select = document.createElement('select');
    select.className = 'ui-select';
    applyTouchTarget(select);
    select.dataset.testid = `settings-user-role-${row.id}`;
    for (const role of ['admin', 'tech'] as const) {
      const option = document.createElement('option');
      option.value = role;
      option.textContent = roleLabelKo(role);
      select.appendChild(option);
    }
    select.value = row.role;
    const isSelf = deps.me()?.id === row.id;
    setServerGated(
      select,
      `${row.name} 역할 변경`,
      isSelf ? '자기 자신의 역할은 바꿀 수 없습니다' : null,
    );
    select.addEventListener('change', () => {
      void handleRoleChange(row, select);
    });
    return select;
  };

  async function handleRoleChange(row: UserInfo, select: HTMLSelectElement): Promise<void> {
    const nextRole: UserRole = select.value === 'admin' ? 'admin' : 'tech';
    if (nextRole === row.role) return;
    select.disabled = true;
    const r = await deps.users.patch(row.id, { role: nextRole });
    if (disposed) return;
    if (r.kind === 'ok') {
      announcer.announceNow(`'${row.name}' 역할이 ${roleLabelKo(nextRole)}(으)로 변경되었습니다`);
    } else {
      announcer.announceNow(`역할 변경 실패 — ${r.messageKo}`);
    }
    void loadUsers(); // 서버 진실로 재렌더 (실패 시 원래 값 복원 포함)
  }

  const deactivateConfirmHost = styled(document.createElement('div'), { display: 'none' });

  const clearDeactivateConfirm = (): void => {
    deactivateConfirmHost.textContent = '';
    deactivateConfirmHost.style.display = 'none';
  };

  const requestDeactivate = (row: UserInfo): void => {
    clearDeactivateConfirm();
    const bar = makeConfirmBar({
      messageKo: `'${row.name}' 사용자를 비활성화할까요? 로그인할 수 없게 됩니다 — 같은 행의 [활성화] 버튼으로 언제든 되돌릴 수 있습니다`,
      confirmLabelKo: '비활성화',
      danger: true,
      testid: 'settings-user-deactivate',
      onConfirm: () => {
        clearDeactivateConfirm();
        void setUserActive(row, false);
      },
      onCancel: clearDeactivateConfirm,
    });
    deactivateConfirmHost.style.display = 'block';
    deactivateConfirmHost.appendChild(bar.el);
    bar.focusConfirm();
  };

  async function setUserActive(row: UserInfo, active: boolean): Promise<void> {
    const r = await deps.users.patch(row.id, { active });
    if (disposed) return;
    if (r.kind === 'ok') {
      announcer.announceNow(
        active
          ? `'${row.name}' 사용자가 활성화되었습니다`
          : `'${row.name}' 사용자가 비활성화되었습니다 — [활성화] 버튼으로 되돌릴 수 있습니다`,
      );
    } else {
      announcer.announceNow(`변경 실패 — ${r.messageKo}`);
    }
    void loadUsers();
  }

  const makeActiveToggle = (row: UserInfo): HTMLButtonElement => {
    const myId = deps.me()?.id ?? null;
    const button = row.active
      ? makeButton('비활성화', `${row.name} 비활성화`, `settings-user-active-${row.id}`, 'danger')
      : makeButton('활성화', `${row.name} 활성화`, `settings-user-active-${row.id}`);
    applyTouchTarget(button);
    const selfReason =
      row.active && !canDeactivateUser(myId, row.id)
        ? '자기 자신은 비활성화할 수 없습니다'
        : null;
    setServerGated(button, button.title, selfReason);
    button.addEventListener('click', () => {
      if (row.active) requestDeactivate(row);
      else void setUserActive(row, true);
    });
    return button;
  };

  const usersTable = makeDataTable<UserInfo>({
    columns: [
      {
        key: 'name',
        labelKo: '이름',
        render: (row): HTMLElement => {
          const el = applyType(document.createElement('span'), TYPE.bodyStrong);
          styled(el, { color: COLOR.textStrong });
          el.textContent = deps.me()?.id === row.id ? `${row.name} (나)` : row.name;
          return el;
        },
      },
      {
        key: 'role',
        labelKo: '역할',
        width: `${USER_COL_ROLE_PX}px`,
        render: (row): HTMLElement => makeRoleSelect(row),
      },
      {
        key: 'active',
        labelKo: '상태',
        width: `${USER_COL_STATUS_PX}px`,
        render: (row): HTMLElement =>
          makeBadge(row.active ? '활성' : '비활성', row.active ? 'success' : 'idle', {
            testid: `settings-user-status-${row.id}`,
          }),
      },
      {
        key: 'actions',
        labelKo: '행동',
        width: `${USER_COL_ACTION_PX}px`,
        render: (row): HTMLElement => makeActiveToggle(row),
      },
    ],
    rows: [],
    rowTestid: (row) => `settings-user-row-${row.id}`,
    emptyState: makeEmptyState({
      iconName: 'users',
      titleKo: '사용자가 없습니다',
      hintKo: '[사용자 추가] 버튼으로 팀원을 등록하세요',
      actions: [],
      testid: 'settings-users-empty',
    }),
    ariaLabelKo: '사용자 목록',
  });

  usersSection.body.appendChild(usersError);
  usersSection.body.appendChild(usersTable.el);
  usersSection.body.appendChild(deactivateConfirmHost);

  async function loadUsers(): Promise<void> {
    const seq = refreshSeq;
    if (deps.me()?.role !== 'admin') return;
    if (serverBlockReasonKo(conn) !== null) {
      usersError.textContent = serverBlockReasonKo(conn) ?? '';
      usersError.style.display = 'block';
      return;
    }
    const r = await deps.users.list();
    if (disposed || seq !== refreshSeq) return;
    if (r.kind === 'ok') {
      users = r.users;
      usersError.style.display = 'none';
      usersTable.setRows(users);
      return;
    }
    usersError.textContent = r.messageKo;
    usersError.style.display = 'block';
  }

  // 사용자 추가 모달
  const addModal = makeModalShell({
    titleKo: '사용자 추가',
    onClose: () => {
      resetAddForm();
    },
    testid: 'settings-user-add-modal',
  });
  root.appendChild(addModal.root);

  const addNameRow = styled(document.createElement('div'), {
    display: 'flex',
    flexDirection: 'column',
    gap: SPACE.xs,
  });
  const addNameLabel = applyType(document.createElement('label'), TYPE.caption);
  styled(addNameLabel, { color: COLOR.label });
  addNameLabel.textContent = '이름';
  addNameLabel.htmlFor = 'settings-user-add-name';
  const addNameInput = document.createElement('input');
  addNameInput.type = 'text';
  addNameInput.maxLength = 80;
  addNameInput.className = 'ui-input';
  addNameInput.id = 'settings-user-add-name';
  addNameInput.dataset.testid = 'settings-user-add-name';
  applyTouchTarget(addNameInput);
  addNameRow.appendChild(addNameLabel);
  addNameRow.appendChild(addNameInput);

  const addRoleRow = styled(document.createElement('div'), {
    display: 'flex',
    flexDirection: 'column',
    gap: SPACE.xs,
  });
  const addRoleLabel = applyType(document.createElement('label'), TYPE.caption);
  styled(addRoleLabel, { color: COLOR.label });
  addRoleLabel.textContent = '역할';
  addRoleLabel.htmlFor = 'settings-user-add-role';
  const addRoleSelect = document.createElement('select');
  addRoleSelect.className = 'ui-select';
  addRoleSelect.id = 'settings-user-add-role';
  addRoleSelect.dataset.testid = 'settings-user-add-role';
  applyTouchTarget(addRoleSelect);
  for (const role of ['tech', 'admin'] as const) {
    const option = document.createElement('option');
    option.value = role;
    option.textContent = roleLabelKo(role);
    addRoleSelect.appendChild(option);
  }
  addRoleRow.appendChild(addRoleLabel);
  addRoleRow.appendChild(addRoleSelect);

  const addPin = makePinField('PIN', 'settings-user-add-pin');
  const addPinHint = hintLine('');
  const addError = applyType(document.createElement('p'), TYPE.caption);
  styled(addError, { margin: '0', color: COLLISION.text, display: 'none' });
  addError.dataset.testid = 'settings-user-add-error';

  const syncAddPinHint = (): void => {
    const role: UserRole = addRoleSelect.value === 'admin' ? 'admin' : 'tech';
    const rule = pinRuleForRole(role);
    addPinHint.textContent = `${roleLabelKo(role)} PIN은 ${rule.minDigits}~${rule.maxDigits}자리 숫자입니다`;
  };
  addRoleSelect.addEventListener('change', syncAddPinHint);

  addModal.body.appendChild(addNameRow);
  addModal.body.appendChild(addRoleRow);
  addModal.body.appendChild(addPin.row);
  addModal.body.appendChild(addPinHint);
  addModal.body.appendChild(addError);

  const addCancelButton = makeButton('취소', '취소', 'settings-user-add-cancel', 'ghost');
  applyTouchTarget(addCancelButton);
  addCancelButton.addEventListener('click', () => {
    addModal.close();
    resetAddForm();
  });
  const addSubmitButton = makeButton('추가', '사용자 추가', 'settings-user-add-submit', 'primary');
  applyTouchTarget(addSubmitButton);
  addModal.footer.appendChild(addCancelButton);
  addModal.footer.appendChild(addSubmitButton);

  for (const input of [addNameInput, addPin.input]) {
    input.addEventListener('input', () => {
      addModal.setDirty(addNameInput.value !== '' || addPin.input.value !== '');
      addError.style.display = 'none';
    });
  }

  function resetAddForm(): void {
    addNameInput.value = '';
    addPin.input.value = '';
    addRoleSelect.value = 'tech';
    addError.style.display = 'none';
    addSubmitButton.disabled = false;
    syncAddPinHint();
  }

  userAddButton.addEventListener('click', () => {
    resetAddForm();
    addModal.open();
    addNameInput.focus();
  });

  addSubmitButton.addEventListener('click', () => {
    void submitAddUser();
  });

  async function submitAddUser(): Promise<void> {
    const role: UserRole = addRoleSelect.value === 'admin' ? 'admin' : 'tech';
    const input: NewUserInput = { name: addNameInput.value.trim(), pin: addPin.input.value, role };
    const check = validateNewUser(input);
    if (!check.ok) {
      addError.textContent = check.messageKo;
      addError.style.display = 'block';
      announcer.announceNow(`오류 — ${check.messageKo}`);
      return;
    }
    addSubmitButton.disabled = true;
    const r = await deps.users.create(input);
    if (disposed) return;
    addSubmitButton.disabled = false;
    if (r.kind === 'ok') {
      addModal.close();
      resetAddForm();
      announcer.announceNow(`'${r.user.name}' 사용자가 추가되었습니다`);
      void loadUsers();
      return;
    }
    addError.textContent = r.messageKo;
    addError.style.display = 'block';
    announcer.announceNow(`오류 — ${r.messageKo}`);
  }

  // ══ 3. 생성 (AI) ══════════════════════════════════════════════════

  const aiSection = makeSection('생성 (AI)', 'wand', 'settings-ai');

  const aiBackendValue = applyType(document.createElement('span'), TYPE.bodyStrong);
  styled(aiBackendValue, { color: COLOR.textStrong });
  aiBackendValue.dataset.testid = 'settings-ai-backend';
  aiSection.body.appendChild(infoRow('현재 백엔드', aiBackendValue));

  const aiButtons = styled(document.createElement('div'), {
    display: 'flex',
    flexWrap: 'wrap',
    gap: SPACE.md,
  });
  const openPlannerButton = makeIconButton(
    'settings',
    '플래너 설정 열기',
    '플래너 설정 열기 — 백엔드·모델·API 키',
    'settings-open-planner',
  );
  applyTouchTarget(openPlannerButton);
  openPlannerButton.addEventListener('click', () => deps.onOpenPlannerSettings());
  aiButtons.appendChild(openPlannerButton);
  aiSection.body.appendChild(aiButtons);
  aiSection.body.appendChild(hintLine('API 키는 이 브라우저에만 저장됩니다 — 서버로 전송되지 않습니다'));

  const paintPlanner = (): void => {
    const summary = deps.plannerSummary();
    aiBackendValue.textContent = summary;
    const lang = textLang(summary);
    if (lang !== undefined) aiBackendValue.setAttribute('lang', lang);
    else aiBackendValue.removeAttribute('lang');
  };

  // ══ 4. 실행 기본값 ════════════════════════════════════════════════

  const execSection = makeSection('실행 기본값', 'gauge', 'settings-exec');

  const autoPauseLabel = document.createElement('label');
  autoPauseLabel.className = 'ui-check-label';
  applyTouchTarget(autoPauseLabel);
  const autoPauseCheck = document.createElement('input');
  autoPauseCheck.type = 'checkbox';
  autoPauseCheck.className = 'ui-check';
  autoPauseCheck.dataset.testid = 'settings-exec-autopause';
  const autoPauseText = document.createElement('span');
  autoPauseText.textContent = '예기치 않은 충돌 시 자동 정지';
  autoPauseLabel.appendChild(autoPauseCheck);
  autoPauseLabel.appendChild(autoPauseText);
  execSection.body.appendChild(autoPauseLabel);
  execSection.body.appendChild(
    hintLine('새 작업을 열 때의 기본값입니다 — 작업마다 실행 중에 바꿀 수 있습니다'),
  );

  const speedSelect = document.createElement('select');
  speedSelect.className = 'ui-select';
  speedSelect.dataset.testid = 'settings-exec-speed';
  applyTouchTarget(speedSelect);
  speedSelect.setAttribute('aria-label', '재생 속도 기본값');
  for (const mult of SPEED_MULT_OPTIONS) {
    const option = document.createElement('option');
    option.value = String(mult);
    option.textContent = speedMultLabel(mult);
    speedSelect.appendChild(option);
  }
  const speedWrap = styled(document.createElement('span'), { display: 'inline-flex' });
  speedWrap.appendChild(speedSelect);
  execSection.body.appendChild(infoRow('재생 속도 기본값', speedWrap));

  autoPauseCheck.addEventListener('change', () => {
    const cur = deps.execDefaults.get();
    deps.execDefaults.set({ ...cur, autoPauseOnCollision: autoPauseCheck.checked });
    announcer.announce(
      autoPauseCheck.checked ? '충돌 자동 정지 기본값 켬' : '충돌 자동 정지 기본값 끔',
    );
  });
  speedSelect.addEventListener('change', () => {
    const cur = deps.execDefaults.get();
    const mult = normalizeSpeedMult(Number(speedSelect.value));
    deps.execDefaults.set({ ...cur, speedMult: mult });
    announcer.announce(`재생 속도 기본값 ${speedMultLabel(mult)}`);
  });

  const paintExecDefaults = (): void => {
    const cur = deps.execDefaults.get();
    autoPauseCheck.checked = cur.autoPauseOnCollision;
    speedSelect.value = String(normalizeSpeedMult(cur.speedMult));
  };

  // ══ 5. 데이터 ═════════════════════════════════════════════════════

  const dataSection = makeSection('데이터', 'layers', 'settings-data');

  const connBadgeHost = styled(document.createElement('span'), { display: 'inline-flex' });
  dataSection.body.appendChild(infoRow('서버 연결', connBadgeHost));

  const lastSyncValue = applyType(document.createElement('span'), TYPE.body);
  styled(lastSyncValue, { color: COLOR.text });
  lastSyncValue.dataset.testid = 'settings-last-sync';
  dataSection.body.appendChild(infoRow('마지막 동기화', lastSyncValue));

  const pendingValue = applyType(document.createElement('span'), TYPE.body);
  styled(pendingValue, { color: COLOR.text });
  pendingValue.dataset.testid = 'settings-pending-count';
  pendingValue.textContent = '확인 중…';
  dataSection.body.appendChild(infoRow('동기화 대기', pendingValue));

  const syncRow = styled(document.createElement('div'), {
    display: 'flex',
    flexWrap: 'wrap',
    alignItems: 'center',
    gap: SPACE.md,
  });
  const syncNowButton = makeIconButton('sync', '지금 동기화', '대기 중인 변경을 서버로 전송', 'settings-sync-now');
  applyTouchTarget(syncNowButton);
  syncRow.appendChild(syncNowButton);
  const syncStatus = applyType(document.createElement('span'), TYPE.caption);
  styled(syncStatus, { color: COLOR.muted });
  syncStatus.dataset.testid = 'settings-sync-status';
  syncRow.appendChild(syncStatus);
  dataSection.body.appendChild(syncRow);

  syncNowButton.addEventListener('click', () => {
    void runSyncNow();
  });

  async function runSyncNow(): Promise<void> {
    syncNowButton.disabled = true;
    syncStatus.textContent = '동기화 중…';
    styled(syncStatus, { color: COLOR.muted });
    const r = await deps.sync.syncNow();
    if (disposed) return;
    setServerGated(syncNowButton, '대기 중인 변경을 서버로 전송');
    if (r.kind === 'ok') {
      syncStatus.textContent = syncReportKo(r);
      styled(syncStatus, { color: r.conflictCount > 0 ? COLOR.warnText : COLOR.successText });
      announcer.announceNow(`동기화 — ${syncReportKo(r)}`);
    } else {
      syncStatus.textContent = r.messageKo;
      styled(syncStatus, { color: COLLISION.text });
      announcer.announceNow(`동기화 실패 — ${r.messageKo}`);
    }
    void loadDataAsync();
  }

  const storageValue = applyType(document.createElement('span'), TYPE.body);
  styled(storageValue, { color: COLOR.text });
  storageValue.dataset.testid = 'settings-storage-usage';
  storageValue.textContent = '확인 중…';
  dataSection.body.appendChild(infoRow('로컬 저장소', storageValue));

  const dataButtons = styled(document.createElement('div'), {
    display: 'flex',
    flexWrap: 'wrap',
    gap: SPACE.md,
  });
  const exportButton = makeIconButton(
    'download',
    '전체 내보내기',
    '모든 작업물을 파일로 내보내기',
    'settings-export-all',
  );
  applyTouchTarget(exportButton);
  exportButton.addEventListener('click', () => deps.onExportAll());
  const importButton = makeIconButton(
    'upload',
    '가져오기',
    '내보낸 파일에서 작업물 가져오기',
    'settings-import-all',
  );
  applyTouchTarget(importButton);
  importButton.addEventListener('click', () => deps.onImportAll());
  dataButtons.appendChild(exportButton);
  dataButtons.appendChild(importButton);
  dataSection.body.appendChild(dataButtons);

  dataSection.body.appendChild(
    hintLine(`휴지통의 항목은 삭제 ${TRASH_RETENTION_DAYS}일 후 완전 삭제됩니다`),
  );

  const paintConnection = (): void => {
    connBadgeHost.textContent = '';
    connBadgeHost.appendChild(
      makeBadge(connectionLabelKo(conn), connectionStatusName(conn), {
        iconName: conn.mode === 'server' && conn.online ? 'check' : 'cloudOff',
        testid: 'settings-connection-badge',
      }),
    );
    lastSyncValue.textContent = lastSyncTextKo(deps.sync.lastSyncIso(), Date.now());
    setServerGated(syncNowButton, '대기 중인 변경을 서버로 전송');
    // 서버 의존 버튼/셀렉트 상태 재계산 (계정 · 사용자 관리)
    paintAccount();
    usersTable.setRows(users);
    setServerGated(userAddButton, '사용자 추가');
  };

  async function loadDataAsync(): Promise<void> {
    const seq = refreshSeq;
    const [pending, estimate] = await Promise.all([
      deps.sync.pendingCount().catch(() => 0),
      deps.storageEstimate().catch((): StorageEstimateInfo | null => null),
    ]);
    if (disposed || seq !== refreshSeq) return;
    pendingValue.textContent = pending === 0 ? '없음' : `${pending}건`;
    storageValue.textContent = formatStorageKo(estimate);
    lastSyncValue.textContent = lastSyncTextKo(deps.sync.lastSyncIso(), Date.now());
  }

  // ══ 6. 정보 ═══════════════════════════════════════════════════════

  const aboutSection = makeSection('정보', 'info', 'settings-about');

  const productValue = document.createElement('span');
  const productName = applyType(document.createElement('span'), TYPE.bodyStrong);
  styled(productName, { color: COLOR.textStrong });
  productName.textContent = PRODUCT_NAME;
  productName.setAttribute('lang', 'en');
  const brandSuffix = applyType(document.createElement('span'), TYPE.caption);
  styled(brandSuffix, { color: COLOR.muted, marginLeft: SPACE.sm });
  brandSuffix.textContent = `— ${BRAND_NAME}`;
  brandSuffix.setAttribute('lang', 'en');
  productValue.appendChild(productName);
  productValue.appendChild(brandSuffix);
  aboutSection.body.appendChild(infoRow('제품', productValue));

  const versionValue = applyType(document.createElement('span'), TYPE.body);
  styled(versionValue, { color: COLOR.text });
  versionValue.setAttribute('lang', 'en');
  versionValue.textContent = deps.appVersion ?? 'dev';
  aboutSection.body.appendChild(infoRow('버전', versionValue));

  const serverValue = applyType(document.createElement('span'), TYPE.body);
  styled(serverValue, { color: COLOR.text });
  serverValue.dataset.testid = 'settings-about-server';
  serverValue.textContent = '확인 중…';
  aboutSection.body.appendChild(infoRow('서버', serverValue));

  async function loadHealth(): Promise<void> {
    const seq = refreshSeq;
    if (conn.mode === 'local') {
      serverValue.textContent = '미연결 (로컬 모드)';
      serverValue.removeAttribute('lang');
      return;
    }
    const h = await deps.health().catch((): ServerHealthInfo | null => null);
    if (disposed || seq !== refreshSeq) return;
    if (h === null) {
      serverValue.textContent = '응답 없음 (오프라인)';
      serverValue.removeAttribute('lang');
      return;
    }
    serverValue.textContent = `${h.name} v${h.version}`;
    serverValue.setAttribute('lang', 'en');
  }

  // ── 조립 · 갱신 ───────────────────────────────────────────────────

  column.appendChild(accountSection.el);
  column.appendChild(usersSection.el);
  column.appendChild(aiSection.el);
  column.appendChild(execSection.el);
  column.appendChild(dataSection.el);
  column.appendChild(aboutSection.el);
  host.appendChild(root);

  const refresh = (): void => {
    if (disposed) return;
    refreshSeq += 1;
    conn = deps.connection.get();
    usersSection.el.style.display = deps.me()?.role === 'admin' ? '' : 'none';
    clearDeactivateConfirm();
    paintPlanner();
    paintExecDefaults();
    paintConnection(); // 계정 섹션 포함 — 서버 의존 컨트롤 상태를 함께 다시 그린다
    void loadUsers();
    void loadDataAsync();
    void loadHealth();
  };

  const unsubscribeConnection = deps.connection.onChange((state) => {
    if (disposed) return;
    conn = state;
    paintConnection();
  });

  refresh();

  return {
    refresh,
    dispose: (): void => {
      if (disposed) return;
      disposed = true;
      unsubscribeConnection();
      usersTable.dispose();
      pinModal.dispose();
      addModal.dispose();
      announcer.dispose();
      root.remove();
    },
  };
}
