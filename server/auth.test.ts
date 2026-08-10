// server/auth.test.ts — 인증: 셋업 → 로그인 → 세션 만료·잠금 · 역할 권한 (BACKEND §3·§7)
//
// ':memory:' DB + app.inject 왕복. 시각은 makeClock 주입 — TTL/잠금을 실제 시간 대기
// 없이 검증한다. 순수 함수(hashPin·lockoutRemainingSec 등)는 DB 없이 직접 검증한다.

import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import {
  hashPin,
  lockoutRemainingSec,
  pinMeetsRolePolicy,
  verifyPin,
} from './auth';
import { parsePort } from './config';
import { bearer, createTestServer, createUserAndLogin, makeClock } from './test-util';
import type { TestServer } from './test-util';
import { buildApp } from './app';
import type { UserInfo } from '../src/schema/entities';

// ── 순수 함수 ───────────────────────────────────────────────────────

describe('hashPin / verifyPin', () => {
  it('같은 PIN + 같은 솔트는 같은 해시, 새 솔트는 다른 해시(사용자별 솔트)', () => {
    const a = hashPin('1234');
    const b = hashPin('1234');
    expect(a.hashB64).not.toBe(b.hashB64); // 솔트가 다르면 해시도 달라야 한다
    expect(hashPin('1234', a.saltB64).hashB64).toBe(a.hashB64);
  });

  it('올바른 PIN만 통과한다', () => {
    const stored = hashPin('482913');
    expect(verifyPin('482913', stored)).toBe(true);
    expect(verifyPin('482914', stored)).toBe(false);
    expect(verifyPin('', stored)).toBe(false);
  });
});

describe('lockoutRemainingSec', () => {
  it('잠기지 않았거나 이미 풀렸으면 null, 아니면 잔여 초 올림', () => {
    expect(lockoutRemainingSec(null, 1000)).toBeNull();
    expect(lockoutRemainingSec(1000, 1000)).toBeNull(); // 정확히 만료 시각 = 풀림
    expect(lockoutRemainingSec(1500, 1000)).toBe(1); // 0.5s → 1s 올림
    expect(lockoutRemainingSec(61_000, 1000)).toBe(60);
  });
});

describe('pinMeetsRolePolicy', () => {
  it('tech는 4자리부터, admin은 6자리부터', () => {
    expect(pinMeetsRolePolicy('1234', 'tech')).toBe(true);
    expect(pinMeetsRolePolicy('1234', 'admin')).toBe(false);
    expect(pinMeetsRolePolicy('123456', 'admin')).toBe(true);
  });
});

describe('parsePort', () => {
  it('숫자가 아니거나 범위 밖이면 null(기본값 강등)', () => {
    expect(parsePort(undefined)).toBeNull();
    expect(parsePort('')).toBeNull();
    expect(parsePort('abc')).toBeNull();
    expect(parsePort('0')).toBeNull();
    expect(parsePort('70000')).toBeNull();
    expect(parsePort('8787')).toBe(8787);
  });
});

// ── 부트스트랩 · 셋업 ───────────────────────────────────────────────

describe('부트스트랩과 셋업', () => {
  it('사용자 0명이면 needsSetup, 셋업은 딱 한 번만 열린다', async () => {
    const clock = makeClock();
    const app = buildApp({ dbPath: ':memory:', now: clock.now, serverName: '부트 서버' });
    await app.ready();
    try {
      const boot = await app.inject({ method: 'GET', url: '/api/v1/auth/bootstrap' });
      expect(boot.statusCode).toBe(200);
      expect(boot.json()).toEqual({ needsSetup: true, serverName: '부트 서버' });

      // admin PIN은 6자리 이상 — 5자리는 거부
      const weak = await app.inject({
        method: 'POST',
        url: '/api/v1/auth/setup',
        payload: { name: '관리자', pin: '12345' },
      });
      expect(weak.statusCode).toBe(400);

      const ok = await app.inject({
        method: 'POST',
        url: '/api/v1/auth/setup',
        payload: { name: '관리자', pin: '123456' },
      });
      expect(ok.statusCode).toBe(200);
      const body = ok.json() as { token: string; user: UserInfo };
      expect(body.token.length).toBeGreaterThanOrEqual(43); // 32바이트 base64url
      expect(body.user.role).toBe('admin');

      const bootAfter = await app.inject({ method: 'GET', url: '/api/v1/auth/bootstrap' });
      expect((bootAfter.json() as { needsSetup: boolean }).needsSetup).toBe(false);

      // 이미 설정된 서버 — 셋업 문은 닫혀 있다
      const again = await app.inject({
        method: 'POST',
        url: '/api/v1/auth/setup',
        payload: { name: '침입자', pin: '999999' },
      });
      expect(again.statusCode).toBe(403);
    } finally {
      await app.close();
    }
  });
});

// ── 로그인 · 세션 · 잠금 ────────────────────────────────────────────

describe('로그인과 세션', () => {
  let server: TestServer;

  beforeEach(async () => {
    server = await createTestServer();
  });
  afterEach(async () => {
    await server.app.close();
  });

  it('타일 목록은 active 사용자만, 로그인 성공 시 토큰과 사용자를 준다', async () => {
    const { user: tech } = await createUserAndLogin(server, '설치기사 김', '4321', 'tech');
    // 비활성 사용자는 타일에서 사라진다
    await server.app.inject({
      method: 'PATCH',
      url: `/api/v1/users/${tech.id}`,
      headers: bearer(server.adminToken),
      payload: { active: false },
    });
    const tiles = await server.app.inject({ method: 'GET', url: '/api/v1/auth/users' });
    const users = (tiles.json() as { users: UserInfo[] }).users;
    expect(users.map((u) => u.id)).toEqual([server.admin.id]);

    // 비활성 사용자는 로그인도 불가 — 존재 여부를 노출하지 않는 같은 401
    const login = await server.app.inject({
      method: 'POST',
      url: '/api/v1/auth/login',
      payload: { userId: tech.id, pin: '4321' },
    });
    expect(login.statusCode).toBe(401);
  });

  it('PIN 5회 실패 → 60초 잠금(423 retryAfterSec), 지나면 다시 로그인된다', async () => {
    const { user } = await createUserAndLogin(server, '설치기사 이', '7777', 'tech');
    const attempt = (pin: string) =>
      server.app.inject({ method: 'POST', url: '/api/v1/auth/login', payload: { userId: user.id, pin } });

    for (let i = 0; i < 4; i += 1) {
      expect((await attempt('0000')).statusCode).toBe(401);
    }
    // 5번째 실패에서 잠금이 걸린다
    const locked = await attempt('0000');
    expect(locked.statusCode).toBe(423);
    expect((locked.json() as { retryAfterSec: number }).retryAfterSec).toBe(60);

    // 잠금 중에는 올바른 PIN도 거부된다
    const during = await attempt('7777');
    expect(during.statusCode).toBe(423);
    expect((during.json() as { retryAfterSec: number }).retryAfterSec).toBeLessThanOrEqual(60);

    server.clock.advanceSec(61);
    expect((await attempt('7777')).statusCode).toBe(200);
  });

  it('세션은 30일 슬라이딩 — 쓰는 동안 연장되고, 안 쓰면 만료 후 401', async () => {
    const { token } = await createUserAndLogin(server, '설치기사 박', '5555', 'tech');
    const me = () =>
      server.app.inject({ method: 'GET', url: '/api/v1/auth/me', headers: bearer(token) });

    server.clock.advanceDays(20);
    expect((await me()).statusCode).toBe(200); // 20일 — 아직 유효, 여기서 TTL 재시작

    server.clock.advanceDays(20);
    expect((await me()).statusCode).toBe(200); // 마지막 사용에서 20일 — 슬라이딩 덕에 유효

    server.clock.advanceDays(31);
    expect((await me()).statusCode).toBe(401); // 31일 방치 — 만료
  });

  it('로그아웃하면 토큰이 즉시 폐기된다', async () => {
    const { token } = await createUserAndLogin(server, '설치기사 최', '9999', 'tech');
    const out = await server.app.inject({
      method: 'POST',
      url: '/api/v1/auth/logout',
      headers: bearer(token),
    });
    expect(out.statusCode).toBe(200);
    const me = await server.app.inject({
      method: 'GET',
      url: '/api/v1/auth/me',
      headers: bearer(token),
    });
    expect(me.statusCode).toBe(401);
  });

  it('Bearer 헤더가 없으면 보호 경로는 401 봉투를 돌려준다', async () => {
    const res = await server.app.inject({ method: 'GET', url: '/api/v1/auth/me' });
    expect(res.statusCode).toBe(401);
    const body = res.json() as { error: string; messageKo: string };
    expect(body.error).toBe('unauthorized');
    expect(body.messageKo.length).toBeGreaterThan(0);
  });
});

// ── 사용자 관리 권한 ────────────────────────────────────────────────

describe('사용자 관리 (admin 전용 + 본인 PIN 예외)', () => {
  let server: TestServer;

  beforeEach(async () => {
    server = await createTestServer();
  });
  afterEach(async () => {
    await server.app.close();
  });

  it('GET /users와 POST /users는 admin 전용(403)', async () => {
    const { token } = await createUserAndLogin(server, '설치기사', '4444', 'tech');
    const list = await server.app.inject({ method: 'GET', url: '/api/v1/users', headers: bearer(token) });
    expect(list.statusCode).toBe(403);
    const create = await server.app.inject({
      method: 'POST',
      url: '/api/v1/users',
      headers: bearer(token),
      payload: { name: '몰래 생성', pin: '1234', role: 'tech' },
    });
    expect(create.statusCode).toBe(403);

    // admin은 inactive 포함 전체 목록을 본다
    const adminList = await server.app.inject({
      method: 'GET',
      url: '/api/v1/users',
      headers: bearer(server.adminToken),
    });
    expect(adminList.statusCode).toBe(200);
    expect((adminList.json() as { users: UserInfo[] }).users).toHaveLength(2);
  });

  it('admin 역할 사용자 생성 시 PIN 6자리 미만은 400', async () => {
    const res = await server.app.inject({
      method: 'POST',
      url: '/api/v1/users',
      headers: bearer(server.adminToken),
      payload: { name: '부관리자', pin: '1234', role: 'admin' },
    });
    expect(res.statusCode).toBe(400);
  });

  it('비-admin은 본인 PIN만 변경할 수 있다 — 새 PIN으로 재로그인 확인', async () => {
    const { user, token } = await createUserAndLogin(server, '설치기사 A', '1111', 'tech');
    const { user: other } = await createUserAndLogin(server, '설치기사 B', '2222', 'tech');

    // 본인 PIN 변경 — 허용
    const own = await server.app.inject({
      method: 'PATCH',
      url: `/api/v1/users/${user.id}`,
      headers: bearer(token),
      payload: { pin: '8888' },
    });
    expect(own.statusCode).toBe(200);
    const relogin = await server.app.inject({
      method: 'POST',
      url: '/api/v1/auth/login',
      payload: { userId: user.id, pin: '8888' },
    });
    expect(relogin.statusCode).toBe(200);

    // 본인이라도 PIN 외 필드는 403
    const role = await server.app.inject({
      method: 'PATCH',
      url: `/api/v1/users/${user.id}`,
      headers: bearer(token),
      payload: { role: 'admin' },
    });
    expect(role.statusCode).toBe(403);

    // 타인 PIN 변경은 403
    const others = await server.app.inject({
      method: 'PATCH',
      url: `/api/v1/users/${other.id}`,
      headers: bearer(token),
      payload: { pin: '3333' },
    });
    expect(others.statusCode).toBe(403);
  });

  it('admin은 이름·역할·활성 상태를 바꿀 수 있고, admin 승격 시 PIN 정책이 적용된다', async () => {
    const { user } = await createUserAndLogin(server, '설치기사 C', '1212', 'tech');
    const rename = await server.app.inject({
      method: 'PATCH',
      url: `/api/v1/users/${user.id}`,
      headers: bearer(server.adminToken),
      payload: { name: '수석 설치기사 C' },
    });
    expect(rename.statusCode).toBe(200);
    expect((rename.json() as { user: UserInfo }).user.name).toBe('수석 설치기사 C');

    // 4자리 PIN인 채 admin 승격 + 4자리 새 PIN → 400 (변경 후 역할 기준)
    const promote = await server.app.inject({
      method: 'PATCH',
      url: `/api/v1/users/${user.id}`,
      headers: bearer(server.adminToken),
      payload: { role: 'admin', pin: '4545' },
    });
    expect(promote.statusCode).toBe(400);

    const missing = await server.app.inject({
      method: 'PATCH',
      url: '/api/v1/users/no-such-user-id',
      headers: bearer(server.adminToken),
      payload: { name: '유령' },
    });
    expect(missing.statusCode).toBe(404);
  });
});
