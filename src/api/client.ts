// api/client.ts — 서버 API 클라이언트: fetch 래퍼 + 연결 상태 머신 + 세션 영속.
// 규범: docs/BACKEND.md §4(API)·§6(오프라인·동기화).
//
// ── 계층 규칙 ───────────────────────────────────────────────────────
// src/api는 src/schema만 안다 (planner와 같은 위치). ui/core/render를 import하지
// 않는다. UI는 이 계층의 discriminated union 결과를 받아 분기만 한다 — throw 기반
// 오류 처리는 UI가 try/catch를 흩뿌리게 만들므로 결과 타입을 기본으로 한다.
//
// ── 연결 상태 머신 (BACKEND §6) ─────────────────────────────────────
// 두 축의 곱이다:
//   mode:   'server' | 'local'  — 부트 시 GET /health 1회로 판정.
//                                 'local'이면 30s 간격으로 서버 출현을 재시도한다.
//   online: boolean             — 요청 성공/실패로 전이. offline이면 15s 헬스 폴링.
// UI 배지용 3상태(BACKEND §6)는 파생값이다: local→'local-only', 아니면 online 여부.
//
// ── 테스트 가능성 ───────────────────────────────────────────────────
// fetch/localStorage/타이머/시각을 전부 생성자 주입한다(기본값은 전역). 순수 전이
// 함수(nextConnectionState)를 분리해 브라우저 없이 node 환경에서 검증한다.

import type { UserInfo } from '../schema/entities';
import { API_PREFIX, PIN_LOCKOUT_SEC, userInfoSchema } from '../schema/entities';

// ── 상수 ────────────────────────────────────────────────────────────

/**
 * 세션 localStorage 키.
 * `src/ui/brand.ts`의 `STORAGE_PREFIX`('workcell') + '.session'과 일치해야 한다 —
 * api 계층은 ui를 import할 수 없어(계층 규칙) 리터럴로 유지한다.
 * brand의 STORAGE_PREFIX를 바꾸면 이 값도 함께 바꿀 것.
 */
export const SESSION_STORAGE_KEY = 'workcell.session';

/** 요청 타임아웃 — 현장 LAN 전제. 3초 무응답이면 오프라인으로 판정하는 편이 낫다. */
export const REQUEST_TIMEOUT_MS = 3000;
/** local 모드에서 서버 출현 재시도 간격 (BACKEND §6) */
export const LOCAL_RETRY_INTERVAL_MS = 30_000;
/** server 모드 offline일 때 헬스 폴링 간격 (BACKEND §6) */
export const OFFLINE_POLL_INTERVAL_MS = 15_000;

// 한국어 오류 문구 — 서버 messageKo가 있으면 그것을 우선한다 (오류 봉투 계약 §4).
export const MSG_NETWORK_KO = '서버에 연결할 수 없습니다 — 네트워크를 확인해 주세요';
export const MSG_TIMEOUT_KO = '서버 응답이 없습니다 (시간 초과)';
export const MSG_LOCAL_MODE_KO = '서버가 설정되지 않아 로컬 모드로 동작 중입니다';
export const MSG_UNAUTHORIZED_KO = '로그인이 만료되었습니다 — 다시 로그인해 주세요';
export const MSG_LOGIN_INVALID_KO = 'PIN이 올바르지 않습니다';
export const MSG_CONFLICT_KO = '다른 사용자가 먼저 저장했습니다 (버전 충돌)';
export const MSG_SERVER_ERROR_KO = '서버 오류가 발생했습니다';

// ── 연결 상태 머신 (순수) ───────────────────────────────────────────

export type ServerMode = 'server' | 'local';

export interface ConnectionState {
  readonly mode: ServerMode;
  readonly online: boolean;
}

/** UI 상단 배지용 3상태 (BACKEND §6) */
export type ConnectionLabel = 'online' | 'offline' | 'local-only';

export type ConnectionEvent =
  | 'boot-health-ok' // 부트 헬스 성공 → server/online
  | 'boot-health-fail' // 부트 헬스 실패 → local (자연 강등, BACKEND §1)
  | 'health-ok' // 폴링 헬스 성공 → server/online (local 탈출 포함)
  | 'health-fail' // 폴링 헬스 실패 → mode 유지, offline
  | 'request-reached' // HTTP 응답 수신(상태코드 무관 — 서버 도달) → online
  | 'request-unreachable'; // 네트워크/타임아웃 실패 → offline

/** 상태 전이 — 순수 함수. 스케줄링(타이머)은 ApiClient가 상태를 보고 결정한다. */
export function nextConnectionState(
  state: ConnectionState,
  event: ConnectionEvent,
): ConnectionState {
  switch (event) {
    case 'boot-health-ok':
    case 'health-ok':
    case 'request-reached':
      return { mode: 'server', online: true };
    case 'boot-health-fail':
      return { mode: 'local', online: false };
    case 'health-fail':
    case 'request-unreachable':
      return { mode: state.mode, online: false };
  }
}

export function connectionLabel(state: ConnectionState): ConnectionLabel {
  if (state.mode === 'local') return 'local-only';
  return state.online ? 'online' : 'offline';
}

/** 상태 배지 한국어 라벨 — 색 없이도 의미가 전달되어야 한다 (UX_DESIGN §9) */
export function connectionLabelKo(state: ConnectionState): string {
  switch (connectionLabel(state)) {
    case 'online':
      return '온라인';
    case 'offline':
      return '오프라인';
    case 'local-only':
      return '로컬 모드';
  }
}

// ── 주입 가능한 의존성 ──────────────────────────────────────────────

/** localStorage 부분집합 — 테스트에서 메모리 구현으로 대체한다 */
export interface SessionStore {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

/** localStorage가 없는 환경(node 테스트)용 폴백 */
export class MemorySessionStore implements SessionStore {
  private readonly map = new Map<string, string>();
  getItem(key: string): string | null {
    return this.map.get(key) ?? null;
  }
  setItem(key: string, value: string): void {
    this.map.set(key, value);
  }
  removeItem(key: string): void {
    this.map.delete(key);
  }
}

/** 타이머 주입점 — 테스트는 수동 타이머로 폴링/타임아웃을 결정론적으로 검증한다 */
export interface TimerHost {
  setTimeout(fn: () => void, ms: number): number;
  clearTimeout(handle: number): void;
}

/** fetch 호출에 실제로 쓰는 init 부분집합 — fake fetch가 이 형태만 구현하면 된다 */
export interface FetchInit {
  readonly method: string;
  readonly headers: Record<string, string>;
  readonly body?: string;
  readonly signal: AbortSignal;
}

export type FetchLike = (url: string, init: FetchInit) => Promise<Response>;

export interface ApiClientOptions {
  /** '' = 같은 오리진 (dev는 Vite 프록시, prod는 단일 프로세스 서빙 — BACKEND §2) */
  readonly baseUrl?: string;
  readonly fetchFn?: FetchLike;
  readonly storage?: SessionStore;
  readonly timers?: TimerHost;
  readonly nowIso?: () => string;
  readonly requestTimeoutMs?: number;
}

function defaultFetch(): FetchLike {
  return (url, init) =>
    globalThis.fetch(url, {
      method: init.method,
      headers: init.headers,
      body: init.body,
      signal: init.signal,
    });
}

function defaultStorage(): SessionStore {
  const g = globalThis as { localStorage?: SessionStore };
  return g.localStorage ?? new MemorySessionStore();
}

function defaultTimers(): TimerHost {
  return {
    // 핸들은 불투명 값 — 브라우저는 number, node는 Timeout 객체. 되돌려 줄 뿐이다.
    setTimeout: (fn, ms) => globalThis.setTimeout(fn, ms) as unknown as number,
    clearTimeout: (handle) => globalThis.clearTimeout(handle),
  };
}

// ── 결과 타입 (throw 대신 discriminated union — UI가 분기하기 쉽게) ──

export type ApiResult<T> =
  | { readonly kind: 'ok'; readonly status: number; readonly data: T }
  | { readonly kind: 'unauthorized'; readonly messageKo: string; readonly body: unknown }
  | {
      readonly kind: 'conflict';
      readonly status: 409;
      readonly body: unknown;
      readonly messageKo: string;
    }
  | { readonly kind: 'network'; readonly messageKo: string }
  | {
      readonly kind: 'error';
      readonly status: number;
      readonly error: string;
      readonly messageKo: string;
      readonly body: unknown;
    };

/** 서버 오류 봉투 `{ error, messageKo }`에서 한국어 문구를 우선 추출한다 */
export function pickMessageKo(body: unknown, fallbackKo: string): string {
  if (body !== null && typeof body === 'object' && 'messageKo' in body) {
    const m = (body as { messageKo: unknown }).messageKo;
    if (typeof m === 'string' && m.trim() !== '') return m;
  }
  return fallbackKo;
}

function pickErrorCode(body: unknown): string {
  if (body !== null && typeof body === 'object' && 'error' in body) {
    const e = (body as { error: unknown }).error;
    if (typeof e === 'string' && e !== '') return e;
  }
  return 'unknown';
}

function pickRetryAfterSec(body: unknown): number {
  if (body !== null && typeof body === 'object' && 'retryAfterSec' in body) {
    const v = (body as { retryAfterSec: unknown }).retryAfterSec;
    if (typeof v === 'number' && Number.isFinite(v) && v >= 0) return v;
  }
  return PIN_LOCKOUT_SEC;
}

// ── 세션 ────────────────────────────────────────────────────────────

export interface AuthSession {
  readonly token: string;
  readonly user: UserInfo;
}

/** 저장된 세션 문자열 해석 — 손상/구버전 형식은 조용히 폐기(null) */
export function parseStoredSession(raw: string | null): AuthSession | null {
  if (raw === null || raw === '') return null;
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return null;
  }
  if (parsed === null || typeof parsed !== 'object') return null;
  const obj = parsed as { token?: unknown; user?: unknown };
  if (typeof obj.token !== 'string' || obj.token === '') return null;
  const user = userInfoSchema.safeParse(obj.user);
  if (!user.success) return null;
  return { token: obj.token, user: user.data };
}

export type LoginResult =
  | { readonly kind: 'ok'; readonly session: AuthSession }
  | { readonly kind: 'invalid'; readonly messageKo: string }
  | { readonly kind: 'locked'; readonly retryAfterSec: number; readonly messageKo: string }
  | { readonly kind: 'network'; readonly messageKo: string }
  | { readonly kind: 'error'; readonly status: number; readonly messageKo: string };

// ── ApiClient ───────────────────────────────────────────────────────

export type HttpMethod = 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';

export interface RequestOptions {
  readonly body?: unknown;
  /** false = 인증 불필요 경로 (health/bootstrap/users/setup/login — BACKEND §4) */
  readonly auth?: boolean;
}

type RawFetchResult =
  | { readonly kind: 'http'; readonly status: number; readonly body: unknown }
  | { readonly kind: 'network' }
  | { readonly kind: 'timeout' };

export class ApiClient {
  private readonly baseUrl: string;
  private readonly fetchFn: FetchLike;
  private readonly storage: SessionStore;
  private readonly timers: TimerHost;
  private readonly requestTimeoutMs: number;

  private state: ConnectionState = { mode: 'server', online: true };
  private session: AuthSession | null;
  private readonly listeners = new Set<(state: ConnectionState) => void>();
  private pollHandle: number | null = null;
  private disposed = false;

  constructor(options: ApiClientOptions = {}) {
    this.baseUrl = options.baseUrl ?? '';
    this.fetchFn = options.fetchFn ?? defaultFetch();
    this.storage = options.storage ?? defaultStorage();
    this.timers = options.timers ?? defaultTimers();
    this.requestTimeoutMs = options.requestTimeoutMs ?? REQUEST_TIMEOUT_MS;
    this.session = parseStoredSession(this.storage.getItem(SESSION_STORAGE_KEY));
  }

  // ── 상태 ──────────────────────────────────────────────────────────

  getState(): ConnectionState {
    return this.state;
  }

  getSession(): AuthSession | null {
    return this.session;
  }

  /** 연결 상태 변경 구독. 반환값은 해제 함수. */
  onStateChange(cb: (state: ConnectionState) => void): () => void {
    this.listeners.add(cb);
    return () => {
      this.listeners.delete(cb);
    };
  }

  /**
   * 부트 시 1회 — GET /health로 mode를 판정한다 (BACKEND §6).
   * 실패하면 local 모드로 자연 강등되고 30s 간격 재시도가 시작된다.
   */
  async start(): Promise<ConnectionState> {
    const ok = await this.healthOk();
    this.applyEvent(ok ? 'boot-health-ok' : 'boot-health-fail');
    return this.state;
  }

  /**
   * 헬스 체크를 즉시 수행한다. 예약된 폴링과 같은 전이를 쓴다 —
   * UI의 "지금 다시 연결" 버튼이 이 메서드를 부른다.
   */
  async pollNow(): Promise<ConnectionState> {
    const ok = await this.healthOk();
    this.applyEvent(ok ? 'health-ok' : 'health-fail');
    return this.state;
  }

  /** 예약된 폴링 타이머를 해제한다 (테스트/페이지 종료용) */
  dispose(): void {
    this.disposed = true;
    if (this.pollHandle !== null) {
      this.timers.clearTimeout(this.pollHandle);
      this.pollHandle = null;
    }
  }

  private applyEvent(event: ConnectionEvent): void {
    const next = nextConnectionState(this.state, event);
    const changed = next.mode !== this.state.mode || next.online !== this.state.online;
    this.state = next;
    if (changed) {
      for (const cb of this.listeners) {
        try {
          cb(next);
        } catch {
          // 구독자 오류가 상태 머신을 깨지 않게 한다
        }
      }
    }
    this.schedule();
  }

  /** 상태를 보고 다음 폴링을 예약한다. server+online이면 폴링 없음. */
  private schedule(): void {
    if (this.pollHandle !== null) {
      this.timers.clearTimeout(this.pollHandle);
      this.pollHandle = null;
    }
    if (this.disposed) return;
    const delayMs =
      this.state.mode === 'local'
        ? LOCAL_RETRY_INTERVAL_MS
        : this.state.online
          ? null
          : OFFLINE_POLL_INTERVAL_MS;
    if (delayMs === null) return;
    this.pollHandle = this.timers.setTimeout(() => {
      this.pollHandle = null;
      void this.pollNow();
    }, delayMs);
  }

  private async healthOk(): Promise<boolean> {
    const raw = await this.rawFetch('GET', '/health', undefined, false);
    return raw.kind === 'http' && raw.status >= 200 && raw.status < 300;
  }

  // ── 요청 코어 ─────────────────────────────────────────────────────

  /**
   * JSON 요청을 보내고 결과 union을 돌려준다. 절대 throw하지 않는다.
   * - local 모드면 fetch 없이 즉시 network 결과 (30s 재시도가 복구를 소유한다)
   * - HTTP 응답 수신 = 서버 도달 → online 전이 / 네트워크·타임아웃 → offline 전이
   * - Bearer를 보낸 요청의 401 = 세션 만료 → 세션 폐기 (로그인 401과 구분된다)
   */
  async request<T>(
    method: HttpMethod,
    path: string,
    opts: RequestOptions = {},
  ): Promise<ApiResult<T>> {
    if (this.state.mode === 'local') {
      return { kind: 'network', messageKo: MSG_LOCAL_MODE_KO };
    }
    const auth = opts.auth ?? true;
    const bearerSent = auth && this.session !== null;
    const raw = await this.rawFetch(method, path, opts.body, auth);

    if (raw.kind === 'network' || raw.kind === 'timeout') {
      this.applyEvent('request-unreachable');
      return {
        kind: 'network',
        messageKo: raw.kind === 'timeout' ? MSG_TIMEOUT_KO : MSG_NETWORK_KO,
      };
    }

    this.applyEvent('request-reached');
    const { status, body } = raw;

    if (status >= 200 && status < 300) {
      return { kind: 'ok', status, data: body as T };
    }
    if (status === 401) {
      // 세션이 실제로 쓰였을 때만 폐기 — /auth/login의 401(PIN 오류)은 세션을 건드리지 않는다
      if (bearerSent) this.clearSession();
      return {
        kind: 'unauthorized',
        messageKo: pickMessageKo(body, MSG_UNAUTHORIZED_KO),
        body,
      };
    }
    if (status === 409) {
      return { kind: 'conflict', status: 409, body, messageKo: pickMessageKo(body, MSG_CONFLICT_KO) };
    }
    return {
      kind: 'error',
      status,
      error: pickErrorCode(body),
      messageKo: pickMessageKo(body, MSG_SERVER_ERROR_KO),
      body,
    };
  }

  private async rawFetch(
    method: HttpMethod,
    path: string,
    body: unknown,
    auth: boolean,
  ): Promise<RawFetchResult> {
    const ctrl = new AbortController();
    let timedOut = false;
    const handle = this.timers.setTimeout(() => {
      timedOut = true;
      ctrl.abort();
    }, this.requestTimeoutMs);
    try {
      const headers: Record<string, string> = { accept: 'application/json' };
      if (body !== undefined) headers['content-type'] = 'application/json';
      if (auth && this.session !== null) {
        headers['authorization'] = `Bearer ${this.session.token}`;
      }
      const res = await this.fetchFn(`${this.baseUrl}${API_PREFIX}${path}`, {
        method,
        headers,
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: ctrl.signal,
      });
      let parsed: unknown;
      try {
        parsed = (await res.json()) as unknown;
      } catch {
        parsed = undefined; // 빈 본문/비JSON — 상태 코드만으로 처리
      }
      return { kind: 'http', status: res.status, body: parsed };
    } catch {
      return timedOut ? { kind: 'timeout' } : { kind: 'network' };
    } finally {
      this.timers.clearTimeout(handle);
    }
  }

  // ── 세션 관리 ─────────────────────────────────────────────────────

  private setSession(session: AuthSession): void {
    this.session = session;
    try {
      this.storage.setItem(SESSION_STORAGE_KEY, JSON.stringify(session));
    } catch {
      // 저장 실패(쿼터 등)해도 메모리 세션으로 계속 동작한다
    }
  }

  private clearSession(): void {
    this.session = null;
    try {
      this.storage.removeItem(SESSION_STORAGE_KEY);
    } catch {
      // storage 불능이어도 메모리에서는 폐기됐다
    }
  }

  // ── 인증 API (BACKEND §3·§4) ──────────────────────────────────────

  async bootstrap(): Promise<ApiResult<{ needsSetup: boolean; serverName: string }>> {
    return this.request('GET', '/auth/bootstrap', { auth: false });
  }

  /** 로그인 화면의 사용자 타일 목록 (active만 — BACKEND §4) */
  async loginUsers(): Promise<ApiResult<{ users: UserInfo[] }>> {
    return this.request('GET', '/auth/users', { auth: false });
  }

  /** 최초 설정 마법사 — 사용자 0명일 때만 서버가 허용한다 */
  async setup(input: { name: string; pin: string }): Promise<LoginResult> {
    const r = await this.request<{ token: string; user: UserInfo }>('POST', '/auth/setup', {
      body: input,
      auth: false,
    });
    return this.toLoginResult(r);
  }

  async login(input: { userId: string; pin: string }): Promise<LoginResult> {
    const r = await this.request<{ token: string; user: UserInfo }>('POST', '/auth/login', {
      body: input,
      auth: false,
    });
    return this.toLoginResult(r);
  }

  /** 서버 통보는 최선 노력 — 결과와 무관하게 로컬 세션은 항상 폐기한다 */
  async logout(): Promise<ApiResult<{ ok: boolean }>> {
    const r = await this.request<{ ok: boolean }>('POST', '/auth/logout', { body: {} });
    this.clearSession();
    return r;
  }

  async me(): Promise<ApiResult<{ user: UserInfo }>> {
    return this.request('GET', '/auth/me');
  }

  private toLoginResult(r: ApiResult<{ token: string; user: UserInfo }>): LoginResult {
    switch (r.kind) {
      case 'ok': {
        const session: AuthSession = { token: r.data.token, user: r.data.user };
        this.setSession(session);
        return { kind: 'ok', session };
      }
      case 'unauthorized':
        // 로그인 맥락의 401 = PIN 오류 (세션 만료가 아니다)
        return { kind: 'invalid', messageKo: pickMessageKo(r.body, MSG_LOGIN_INVALID_KO) };
      case 'network':
        return { kind: 'network', messageKo: r.messageKo };
      case 'conflict':
        return { kind: 'error', status: 409, messageKo: r.messageKo };
      case 'error':
        if (r.status === 423) {
          const retryAfterSec = pickRetryAfterSec(r.body);
          return {
            kind: 'locked',
            retryAfterSec,
            messageKo: pickMessageKo(
              r.body,
              `잠시 후 다시 시도해 주세요 (${retryAfterSec}초 잠금)`,
            ),
          };
        }
        return { kind: 'error', status: r.status, messageKo: r.messageKo };
    }
  }
}
