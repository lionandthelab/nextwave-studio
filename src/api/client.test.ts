// api/client.test.ts — ApiClient 단위 테스트 (DOM 비의존, node 환경).
// fetch/storage/타이머를 전부 주입해 브라우저 없이 검증한다:
// - 연결 상태 머신 전이 (server/local × online/offline — BACKEND §6)
// - 폴링 스케줄 (local 30s 재시도 / offline 15s 헬스 폴링)
// - 세션 저장·복원·만료(Bearer 요청의 401 → 세션 폐기, 로그인 401과 구분)
// - 요청 타임아웃(AbortController) → network 결과 + offline 전이
// - 한국어 오류 문구: 서버 messageKo 우선, 없으면 자체 폴백

import { describe, expect, it } from 'vitest';
import {
  ApiClient,
  MemorySessionStore,
  SESSION_STORAGE_KEY,
  LOCAL_RETRY_INTERVAL_MS,
  OFFLINE_POLL_INTERVAL_MS,
  REQUEST_TIMEOUT_MS,
  MSG_LOCAL_MODE_KO,
  MSG_LOGIN_INVALID_KO,
  MSG_NETWORK_KO,
  MSG_TIMEOUT_KO,
  connectionLabel,
  connectionLabelKo,
  nextConnectionState,
  parseStoredSession,
  pickMessageKo,
} from './client';
import type { ConnectionState, FetchInit, FetchLike, TimerHost } from './client';
import { PIN_LOCKOUT_SEC } from '../schema/entities';
import type { UserInfo } from '../schema/entities';

// ── 테스트 대역 ─────────────────────────────────────────────────────

class ManualTimers implements TimerHost {
  private nextId = 1;
  readonly scheduled: Array<{ id: number; fn: () => void; delayMs: number }> = [];

  setTimeout(fn: () => void, ms: number): number {
    const id = this.nextId;
    this.nextId += 1;
    this.scheduled.push({ id, fn, delayMs: ms });
    return id;
  }

  clearTimeout(id: number): void {
    const i = this.scheduled.findIndex((s) => s.id === id);
    if (i >= 0) this.scheduled.splice(i, 1);
  }

  /** delayMs ≤ ms인 예약을 등록 순서대로 실행 */
  advance(ms: number): void {
    const due = this.scheduled.filter((s) => s.delayMs <= ms);
    for (const entry of due) {
      const i = this.scheduled.indexOf(entry);
      if (i >= 0) this.scheduled.splice(i, 1);
      entry.fn();
    }
  }

  delays(): number[] {
    return this.scheduled.map((s) => s.delayMs);
  }
}

type FetchStep = Response | Error | 'hang';

function makeFetch(steps: FetchStep[]): {
  fetchFn: FetchLike;
  calls: Array<{ url: string; init: FetchInit }>;
} {
  const calls: Array<{ url: string; init: FetchInit }> = [];
  const queue = [...steps];
  const fetchFn: FetchLike = (url, init) => {
    calls.push({ url, init });
    const step = queue.shift();
    if (step === undefined) return Promise.reject(new Error('fake fetch: 예약된 응답 없음'));
    if (step === 'hang') {
      return new Promise<Response>((_resolve, reject) => {
        init.signal.addEventListener('abort', () => reject(new Error('aborted')));
      });
    }
    if (step instanceof Error) return Promise.reject(step);
    return Promise.resolve(step);
  };
  return { fetchFn, calls };
}

function json(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

const USER: UserInfo = { id: 'user-0001', name: '김설치', role: 'tech', active: true };

function makeClient(steps: FetchStep[], storage = new MemorySessionStore()) {
  const timers = new ManualTimers();
  const { fetchFn, calls } = makeFetch(steps);
  const client = new ApiClient({ fetchFn, storage, timers });
  return { client, timers, storage, calls };
}

// ── 상태 머신 (순수 전이) ───────────────────────────────────────────

describe('nextConnectionState', () => {
  const server: ConnectionState = { mode: 'server', online: true };
  const local: ConnectionState = { mode: 'local', online: false };

  it('부트 헬스 성공 → server/online, 실패 → local/offline (자연 강등)', () => {
    expect(nextConnectionState(server, 'boot-health-ok')).toEqual({ mode: 'server', online: true });
    expect(nextConnectionState(server, 'boot-health-fail')).toEqual({
      mode: 'local',
      online: false,
    });
  });

  it('local에서 헬스 성공 → server/online (서버 출현 감지)', () => {
    expect(nextConnectionState(local, 'health-ok')).toEqual({ mode: 'server', online: true });
  });

  it('헬스/요청 실패는 mode를 유지하고 offline으로만 내린다', () => {
    expect(nextConnectionState(server, 'request-unreachable')).toEqual({
      mode: 'server',
      online: false,
    });
    expect(nextConnectionState(local, 'health-fail')).toEqual({ mode: 'local', online: false });
  });

  it('HTTP 응답 수신(상태코드 무관)은 서버 도달 → online', () => {
    expect(
      nextConnectionState({ mode: 'server', online: false }, 'request-reached'),
    ).toEqual({ mode: 'server', online: true });
  });
});

describe('connectionLabel / connectionLabelKo', () => {
  it('BACKEND §6의 3상태로 파생된다', () => {
    expect(connectionLabel({ mode: 'server', online: true })).toBe('online');
    expect(connectionLabel({ mode: 'server', online: false })).toBe('offline');
    expect(connectionLabel({ mode: 'local', online: false })).toBe('local-only');
  });

  it('상태 배지 한국어 라벨 (색 없이 의미 전달)', () => {
    expect(connectionLabelKo({ mode: 'server', online: true })).toBe('온라인');
    expect(connectionLabelKo({ mode: 'server', online: false })).toBe('오프라인');
    expect(connectionLabelKo({ mode: 'local', online: false })).toBe('로컬 모드');
  });
});

// ── 부트 판정 + 폴링 스케줄 ─────────────────────────────────────────

describe('ApiClient.start / 폴링 스케줄', () => {
  it('부트 헬스 성공 → server/online, 폴링 없음', async () => {
    const { client, timers, calls } = makeClient([json(200, { ok: true })]);
    const state = await client.start();
    expect(state).toEqual({ mode: 'server', online: true });
    expect(timers.delays()).toEqual([]);
    expect(calls[0]?.url).toBe('/api/v1/health');
  });

  it('부트 헬스 실패 → local + 30s 재시도 예약, pollNow 성공 시 server 복귀', async () => {
    const { client, timers } = makeClient([new TypeError('fetch failed'), json(200, { ok: true })]);
    const seen: ConnectionState[] = [];
    client.onStateChange((s) => seen.push(s));

    expect(await client.start()).toEqual({ mode: 'local', online: false });
    expect(timers.delays()).toEqual([LOCAL_RETRY_INTERVAL_MS]);

    expect(await client.pollNow()).toEqual({ mode: 'server', online: true });
    expect(timers.delays()).toEqual([]); // online이면 폴링 해제
    expect(seen).toEqual([
      { mode: 'local', online: false },
      { mode: 'server', online: true },
    ]);
  });

  it('local 모드의 일반 요청은 fetch 없이 즉시 network 결과 (30s 재시도가 복구를 소유)', async () => {
    const { client, calls } = makeClient([new TypeError('fetch failed')]);
    await client.start();
    expect(calls.length).toBe(1);

    const r = await client.request('GET', '/processes');
    expect(r).toEqual({ kind: 'network', messageKo: MSG_LOCAL_MODE_KO });
    expect(calls.length).toBe(1); // 추가 fetch 없음
  });

  it('server 모드 요청 실패 → offline + 15s 헬스 폴링 예약, 타이머 발화로 복귀', async () => {
    const { client, timers } = makeClient([
      json(200, { ok: true }), // start
      new TypeError('fetch failed'), // request
      json(200, { ok: true }), // 폴링 헬스
    ]);
    await client.start();

    const r = await client.request('GET', '/tasks');
    expect(r).toEqual({ kind: 'network', messageKo: MSG_NETWORK_KO });
    expect(client.getState()).toEqual({ mode: 'server', online: false });
    expect(timers.delays()).toEqual([OFFLINE_POLL_INTERVAL_MS]);

    expect(await client.pollNow()).toEqual({ mode: 'server', online: true });
  });

  it('dispose는 예약된 폴링을 해제한다', async () => {
    const { client, timers } = makeClient([new TypeError('fetch failed')]);
    await client.start();
    expect(timers.delays()).toEqual([LOCAL_RETRY_INTERVAL_MS]);
    client.dispose();
    expect(timers.delays()).toEqual([]);
  });
});

// ── 타임아웃 ────────────────────────────────────────────────────────

describe('요청 타임아웃', () => {
  it('3s 무응답이면 abort → network 결과(시간 초과 문구) + offline 전이', async () => {
    const { client, timers } = makeClient(['hang']);
    const pending = client.request('GET', '/auth/me', { auth: false });
    timers.advance(REQUEST_TIMEOUT_MS);
    const r = await pending;
    expect(r).toEqual({ kind: 'network', messageKo: MSG_TIMEOUT_KO });
    expect(client.getState()).toEqual({ mode: 'server', online: false });
  });
});

// ── 세션 저장·복원·만료 ─────────────────────────────────────────────

describe('세션', () => {
  it("로그인 성공 시 localStorage 'workcell.session'에 저장된다", async () => {
    const { client, storage } = makeClient([json(200, { token: 'tok-9', user: USER })]);
    const r = await client.login({ userId: USER.id, pin: '1234' });
    expect(r.kind).toBe('ok');
    expect(client.getSession()?.token).toBe('tok-9');
    const stored = parseStoredSession(storage.getItem(SESSION_STORAGE_KEY));
    expect(stored?.token).toBe('tok-9');
    expect(stored?.user.name).toBe('김설치');
  });

  it('새 클라이언트가 같은 storage에서 세션을 복원한다', () => {
    const storage = new MemorySessionStore();
    storage.setItem(SESSION_STORAGE_KEY, JSON.stringify({ token: 'tok-1', user: USER }));
    const { client } = makeClient([], storage);
    expect(client.getSession()?.token).toBe('tok-1');
  });

  it('Bearer 헤더는 세션 토큰으로 채워진다', async () => {
    const storage = new MemorySessionStore();
    storage.setItem(SESSION_STORAGE_KEY, JSON.stringify({ token: 'tok-1', user: USER }));
    const { client, calls } = makeClient([json(200, { user: USER })], storage);
    await client.me();
    expect(calls[0]?.url).toBe('/api/v1/auth/me');
    expect(calls[0]?.init.headers['authorization']).toBe('Bearer tok-1');
  });

  it('Bearer 요청의 401 = 세션 만료 → 세션 폐기 + unauthorized 결과', async () => {
    const storage = new MemorySessionStore();
    storage.setItem(SESSION_STORAGE_KEY, JSON.stringify({ token: 'tok-1', user: USER }));
    const { client } = makeClient(
      [json(401, { error: 'unauthorized', messageKo: '세션이 만료되었습니다' })],
      storage,
    );
    const r = await client.me();
    expect(r.kind).toBe('unauthorized');
    if (r.kind === 'unauthorized') expect(r.messageKo).toBe('세션이 만료되었습니다');
    expect(client.getSession()).toBeNull();
    expect(storage.getItem(SESSION_STORAGE_KEY)).toBeNull();
  });

  it('로그인 401(PIN 오류)은 기존 세션을 건드리지 않는다', async () => {
    const storage = new MemorySessionStore();
    storage.setItem(SESSION_STORAGE_KEY, JSON.stringify({ token: 'tok-1', user: USER }));
    const { client } = makeClient([json(401, {})], storage);
    const r = await client.login({ userId: 'user-0002', pin: '0000' });
    expect(r).toEqual({ kind: 'invalid', messageKo: MSG_LOGIN_INVALID_KO });
    expect(client.getSession()?.token).toBe('tok-1'); // 빠른 사용자 전환 보호
  });

  it('로그인 423 → locked + retryAfterSec (없으면 서버 상수 폴백)', async () => {
    const { client } = makeClient([
      json(423, { error: 'locked', messageKo: '5회 실패로 잠금', retryAfterSec: 42 }),
      json(423, { error: 'locked' }),
    ]);
    const r1 = await client.login({ userId: USER.id, pin: '0000' });
    expect(r1).toEqual({ kind: 'locked', retryAfterSec: 42, messageKo: '5회 실패로 잠금' });
    const r2 = await client.login({ userId: USER.id, pin: '0000' });
    expect(r2.kind).toBe('locked');
    if (r2.kind === 'locked') expect(r2.retryAfterSec).toBe(PIN_LOCKOUT_SEC);
  });

  it('logout은 서버 통보가 실패해도 로컬 세션을 폐기한다', async () => {
    const storage = new MemorySessionStore();
    storage.setItem(SESSION_STORAGE_KEY, JSON.stringify({ token: 'tok-1', user: USER }));
    const { client } = makeClient([new TypeError('offline')], storage);
    const r = await client.logout();
    expect(r.kind).toBe('network');
    expect(client.getSession()).toBeNull();
    expect(storage.getItem(SESSION_STORAGE_KEY)).toBeNull();
  });
});

describe('parseStoredSession', () => {
  it('손상/구버전 형식은 조용히 폐기한다', () => {
    expect(parseStoredSession(null)).toBeNull();
    expect(parseStoredSession('')).toBeNull();
    expect(parseStoredSession('not-json')).toBeNull();
    expect(parseStoredSession(JSON.stringify({ user: USER }))).toBeNull(); // token 없음
    expect(parseStoredSession(JSON.stringify({ token: 't' }))).toBeNull(); // user 없음
    expect(
      parseStoredSession(JSON.stringify({ token: 't', user: { ...USER, role: 'root' } })),
    ).toBeNull(); // 알 수 없는 role
  });

  it('유효한 세션은 복원된다', () => {
    const s = parseStoredSession(JSON.stringify({ token: 'tok-1', user: USER }));
    expect(s).toEqual({ token: 'tok-1', user: USER });
  });
});

// ── 한국어 오류 문구 ────────────────────────────────────────────────

describe('pickMessageKo', () => {
  it('서버 messageKo를 우선하고, 없거나 공백이면 폴백을 쓴다', () => {
    expect(pickMessageKo({ error: 'x', messageKo: '서버 문구' }, '폴백')).toBe('서버 문구');
    expect(pickMessageKo({ error: 'x' }, '폴백')).toBe('폴백');
    expect(pickMessageKo({ messageKo: '   ' }, '폴백')).toBe('폴백');
    expect(pickMessageKo(undefined, '폴백')).toBe('폴백');
    expect(pickMessageKo('문자열 본문', '폴백')).toBe('폴백');
  });
});

describe('서버 오류 봉투 매핑', () => {
  it('5xx는 error 결과로, messageKo는 서버 문구를 우선한다', async () => {
    const { client } = makeClient([
      json(500, { error: 'internal', messageKo: '서버 내부 오류' }),
    ]);
    const r = await client.request('GET', '/tasks', { auth: false });
    expect(r.kind).toBe('error');
    if (r.kind === 'error') {
      expect(r.status).toBe(500);
      expect(r.error).toBe('internal');
      expect(r.messageKo).toBe('서버 내부 오류');
    }
  });
});
