// server/auth.ts — PIN 해시(scrypt) · 세션(토큰 해시 저장) · 실패 카운터/잠금
//
// 보안 결정의 "왜" (docs/BACKEND.md §3):
// - PIN은 scrypt(N=16384, 사용자별 솔트)로 해시한다 — 숫자 4~8자리는 엔트로피가 낮아
//   빠른 해시(sha256 단독)면 오프라인 사전공격에 수 초면 뚫린다. 비교는 timingSafeEqual.
// - 세션 토큰은 32바이트 무작위(base64url)를 클라이언트에 주고, DB에는 SHA-256 해시만
//   저장한다 — DB 파일이 유출돼도(현장 PC는 물리 접근이 쉽다) 토큰을 재사용할 수 없다.
// - 온라인 무차별 대입은 실패 카운터 + 잠금(PIN_MAX_ATTEMPTS회 → PIN_LOCKOUT_SEC초)으로
//   막는다. 잠금 시점에 카운터를 리셋해 잠금 해제 후 다시 5회의 기회가 생기게 한다.

import { createHash, randomBytes, randomUUID, scryptSync, timingSafeEqual } from 'node:crypto';
import {
  PIN_LOCKOUT_SEC,
  PIN_MAX_ATTEMPTS,
  SESSION_TTL_DAYS,
  userRoleSchema,
} from '../src/schema/entities';
import type { UserInfo, UserRole } from '../src/schema/entities';
import { DAY_MS } from './db';
import type { DbHandle } from './db';

// ── 상수 (entities.ts 공유 상수의 파생 — 매직넘버 금지) ─────────────

export const SESSION_TTL_MS = SESSION_TTL_DAYS * DAY_MS;
export const PIN_LOCKOUT_MS = PIN_LOCKOUT_SEC * 1000;

/** scrypt 파라미터 — BACKEND §3 명시(N=16384). r/p는 scrypt 표준 권장값. */
const SCRYPT_N = 16384;
const SCRYPT_R = 8;
const SCRYPT_P = 1;
const SCRYPT_KEYLEN = 32;
const SALT_BYTES = 16;
const TOKEN_BYTES = 32;

// ── PIN 해시 (순수 — DB 비의존) ─────────────────────────────────────

export interface StoredPin {
  readonly saltB64: string;
  readonly hashB64: string;
}

/** PIN을 해시한다. 솔트를 생략하면 사용자별 새 솔트를 만든다. */
export function hashPin(pin: string, saltB64?: string): StoredPin {
  const salt = saltB64 !== undefined ? Buffer.from(saltB64, 'base64') : randomBytes(SALT_BYTES);
  const hash = scryptSync(pin, salt, SCRYPT_KEYLEN, { N: SCRYPT_N, r: SCRYPT_R, p: SCRYPT_P });
  return { saltB64: salt.toString('base64'), hashB64: hash.toString('base64') };
}

/** 저장된 해시와 비교한다 — 길이가 같을 때만 timingSafeEqual(길이 다름 = 즉시 불일치). */
export function verifyPin(pin: string, stored: StoredPin): boolean {
  const candidate = Buffer.from(hashPin(pin, stored.saltB64).hashB64, 'base64');
  const expected = Buffer.from(stored.hashB64, 'base64');
  if (candidate.length !== expected.length) return false;
  return timingSafeEqual(candidate, expected);
}

/**
 * 역할별 PIN 정책 (entities.ts pinSchema 주석의 서버 강제 부분):
 * tech 4~8자리, admin 6~8자리. 자릿수 상한/숫자 형식은 pinSchema가 이미 본다.
 */
export function pinMeetsRolePolicy(pin: string, role: UserRole): boolean {
  return role === 'admin' ? pin.length >= 6 : true;
}

// ── 사용자 행 ───────────────────────────────────────────────────────

export interface UserRow {
  readonly id: string;
  readonly name: string;
  readonly role: string;
  readonly pin_hash: string;
  readonly salt: string;
  readonly active: number;
  readonly failed_attempts: number;
  readonly locked_until: number | null;
  readonly created_at: number;
}

export function toUserInfo(row: UserRow): UserInfo {
  return {
    id: row.id,
    name: row.name,
    role: userRoleSchema.parse(row.role),
    active: row.active === 1,
  };
}

export function findUserById(db: DbHandle, userId: string): UserRow | undefined {
  return db.one<UserRow>('SELECT * FROM users WHERE id = ?', userId);
}

export function createUser(
  db: DbHandle,
  input: { name: string; pin: string; role: UserRole },
  nowMs: number,
): UserRow {
  const id = randomUUID();
  const stored = hashPin(input.pin);
  db.run(
    `INSERT INTO users (id, name, role, pin_hash, salt, active, failed_attempts, locked_until, created_at)
     VALUES (?, ?, ?, ?, ?, 1, 0, NULL, ?)`,
    id,
    input.name,
    input.role,
    stored.hashB64,
    stored.saltB64,
    nowMs,
  );
  // 방금 넣은 행이 없을 수 없다 — 있으면 프로그래밍 오류이므로 즉시 드러낸다
  const row = findUserById(db, id);
  if (row === undefined) throw new Error('사용자 생성 직후 조회 실패');
  return row;
}

// ── 세션 ────────────────────────────────────────────────────────────

/** 클라이언트에 주는 원문 토큰 — DB에는 절대 원문을 저장하지 않는다 */
export function generateSessionToken(): string {
  return randomBytes(TOKEN_BYTES).toString('base64url');
}

export function hashSessionToken(token: string): string {
  return createHash('sha256').update(token).digest('hex');
}

export function createSession(db: DbHandle, userId: string, nowMs: number): string {
  const token = generateSessionToken();
  db.run(
    `INSERT INTO sessions (token_hash, user_id, created_at, expires_at, last_seen_at)
     VALUES (?, ?, ?, ?, ?)`,
    hashSessionToken(token),
    userId,
    nowMs,
    nowMs + SESSION_TTL_MS,
    nowMs,
  );
  return token;
}

interface SessionJoinRow extends UserRow {
  readonly expires_at: number;
}

/**
 * 토큰 → 사용자. 만료·비활성은 null. 유효하면 슬라이딩 갱신(TTL을 지금부터 다시).
 * 슬라이딩인 이유: 현장 단말에서 매일 로그인시키지 않는다(SESSION_TTL_DAYS 주석).
 */
export function findSessionUser(db: DbHandle, token: string, nowMs: number): UserInfo | null {
  const tokenHash = hashSessionToken(token);
  const row = db.one<SessionJoinRow>(
    `SELECT u.*, s.expires_at FROM sessions s JOIN users u ON u.id = s.user_id
     WHERE s.token_hash = ?`,
    tokenHash,
  );
  if (row === undefined) return null;
  if (row.expires_at <= nowMs) {
    // 만료 세션은 조회 시점에 지연 삭제 — 별도 청소 작업이 필요 없다
    db.run('DELETE FROM sessions WHERE token_hash = ?', tokenHash);
    return null;
  }
  if (row.active !== 1) return null; // 비활성화된 사용자의 기존 세션도 즉시 무효
  db.run(
    'UPDATE sessions SET expires_at = ?, last_seen_at = ? WHERE token_hash = ?',
    nowMs + SESSION_TTL_MS,
    nowMs,
    tokenHash,
  );
  return toUserInfo(row);
}

export function revokeSession(db: DbHandle, token: string): void {
  db.run('DELETE FROM sessions WHERE token_hash = ?', hashSessionToken(token));
}

// ── 실패 카운터 · 잠금 ──────────────────────────────────────────────

/** 잠금 잔여 초 (순수). 잠기지 않았거나 이미 풀렸으면 null. */
export function lockoutRemainingSec(lockedUntilMs: number | null, nowMs: number): number | null {
  if (lockedUntilMs === null || lockedUntilMs <= nowMs) return null;
  return Math.ceil((lockedUntilMs - nowMs) / 1000);
}

/**
 * PIN 실패 1회 기록. 임계에 도달하면 잠금을 걸고 잔여 초를 반환한다(아니면 null).
 * 잠금 설정 시 카운터를 0으로 되돌린다 — 잠금 해제 후 곧바로 재잠금되지 않게.
 */
export function registerPinFailure(db: DbHandle, row: UserRow, nowMs: number): number | null {
  const nextCount = row.failed_attempts + 1;
  if (nextCount >= PIN_MAX_ATTEMPTS) {
    db.run(
      'UPDATE users SET failed_attempts = 0, locked_until = ? WHERE id = ?',
      nowMs + PIN_LOCKOUT_MS,
      row.id,
    );
    return PIN_LOCKOUT_SEC;
  }
  db.run('UPDATE users SET failed_attempts = ? WHERE id = ?', nextCount, row.id);
  return null;
}

export function resetPinFailures(db: DbHandle, userId: string): void {
  db.run('UPDATE users SET failed_attempts = 0, locked_until = NULL WHERE id = ?', userId);
}
