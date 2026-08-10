// server/routes-auth.ts — bootstrap/setup/users(타일)/login/logout/me + /users CRUD
//
// 로그인 UX의 "왜"(docs/BACKEND.md §3): 현장 공유 단말 + 장갑 전제라 아이디 타이핑이
// 없다 — 타일(GET /auth/users, active만) → PIN 패드 → POST /auth/login. 실패 응답은
// "사용자 없음"과 "PIN 불일치"를 구분하지 않는다(계정 존재 여부를 노출하지 않는다).

import type { FastifyInstance } from 'fastify';
import { z } from 'zod';
import { displayNameSchema, pinSchema, userRoleSchema } from '../src/schema/entities';
import { parseWith, requireAdmin, requireUser, sendError } from './app';
import {
  createSession,
  createUser,
  findUserById,
  hashPin,
  lockoutRemainingSec,
  pinMeetsRolePolicy,
  registerPinFailure,
  resetPinFailures,
  revokeSession,
  toUserInfo,
  verifyPin,
} from './auth';
import type { UserRow } from './auth';
import type { RouteContext } from './db';

const setupBodySchema = z.object({ name: displayNameSchema, pin: pinSchema });

// 로그인 본문은 형식만 본다 — PIN 형식 오류도 401로 수렴시켜 정보를 흘리지 않는다
const loginBodySchema = z.object({ userId: z.string().min(1), pin: z.string().min(1) });

const createUserBodySchema = z.object({
  name: displayNameSchema,
  pin: pinSchema,
  role: userRoleSchema,
});

const patchUserBodySchema = z
  .object({
    name: displayNameSchema.optional(),
    pin: pinSchema.optional(),
    role: userRoleSchema.optional(),
    active: z.boolean().optional(),
  })
  .refine((body) => Object.keys(body).length > 0, { message: '변경할 필드가 없습니다' });

function countUsers(ctx: RouteContext): number {
  return ctx.db.one<{ n: number }>('SELECT COUNT(*) AS n FROM users')?.n ?? 0;
}

export function registerAuthRoutes(api: FastifyInstance, ctx: RouteContext): void {
  const { db } = ctx;

  // ── 부트스트랩 · 셋업 ─────────────────────────────────────────────

  api.get('/auth/bootstrap', async () => ({
    needsSetup: countUsers(ctx) === 0,
    serverName: ctx.serverName,
  }));

  api.post('/auth/setup', async (req, reply) => {
    const body = parseWith(reply, setupBodySchema, req.body);
    if (body === null) return reply;
    // 첫 관리자는 admin이다 — admin PIN 정책(6자리+)을 셋업에서도 강제
    if (!pinMeetsRolePolicy(body.pin, 'admin')) {
      return sendError(reply, 400, 'weak-pin', '관리자 PIN은 6자리 이상이어야 합니다');
    }
    const nowMs = ctx.now().getTime();
    // 사용자 0명일 때만 열린 문 — 검사와 생성을 한 트랜잭션으로 묶어 동시 셋업 경합 차단
    const result = db.transaction((): { token: string; row: UserRow } | null => {
      if (countUsers(ctx) > 0) return null;
      const row = createUser(db, { name: body.name, pin: body.pin, role: 'admin' }, nowMs);
      return { token: createSession(db, row.id, nowMs), row };
    });
    if (result === null) {
      return sendError(reply, 403, 'setup-closed', '이미 설정이 완료된 서버입니다 — 관리자에게 계정을 요청하세요');
    }
    return reply.send({ token: result.token, user: toUserInfo(result.row) });
  });

  // ── 로그인 타일 · 로그인/로그아웃 ─────────────────────────────────

  api.get('/auth/users', async () => {
    const rows = db.all<UserRow>('SELECT * FROM users WHERE active = 1 ORDER BY created_at ASC');
    return { users: rows.map(toUserInfo) };
  });

  api.post('/auth/login', async (req, reply) => {
    const body = parseWith(reply, loginBodySchema, req.body);
    if (body === null) return reply;
    const nowMs = ctx.now().getTime();

    let row = findUserById(db, body.userId);
    if (row === undefined || row.active !== 1) {
      return sendError(reply, 401, 'invalid-credentials', '사용자 또는 PIN이 올바르지 않습니다');
    }

    // 잠금 확인 — 만료된 잠금은 여기서 청소해 카운터를 새로 시작한다
    const remaining = lockoutRemainingSec(row.locked_until, nowMs);
    if (remaining !== null) {
      return sendError(reply, 423, 'locked', `PIN 오류가 반복되어 잠겼습니다 — ${remaining}초 후 다시 시도하세요`, {
        retryAfterSec: remaining,
      });
    }
    if (row.locked_until !== null) {
      resetPinFailures(db, row.id);
      const refreshed = findUserById(db, row.id);
      if (refreshed !== undefined) row = refreshed;
    }

    if (!verifyPin(body.pin, { saltB64: row.salt, hashB64: row.pin_hash })) {
      const lockedForSec = registerPinFailure(db, row, nowMs);
      if (lockedForSec !== null) {
        return sendError(reply, 423, 'locked', `PIN 오류가 반복되어 잠겼습니다 — ${lockedForSec}초 후 다시 시도하세요`, {
          retryAfterSec: lockedForSec,
        });
      }
      return sendError(reply, 401, 'invalid-credentials', '사용자 또는 PIN이 올바르지 않습니다');
    }

    resetPinFailures(db, row.id);
    const token = createSession(db, row.id, nowMs);
    return reply.send({ token, user: toUserInfo(row) });
  });

  api.post('/auth/logout', async (req, reply) => {
    // preHandler를 통과했으므로 Bearer 헤더는 반드시 있다
    const header = req.headers.authorization ?? '';
    const token = header.startsWith('Bearer ') ? header.slice('Bearer '.length).trim() : '';
    if (token !== '') revokeSession(db, token);
    return reply.send({ ok: true });
  });

  api.get('/auth/me', async (req) => ({ user: requireUser(req) }));

  // ── 사용자 관리 (admin) ───────────────────────────────────────────

  api.get('/users', async (req, reply) => {
    if (requireAdmin(req, reply) === null) return reply;
    const rows = db.all<UserRow>('SELECT * FROM users ORDER BY created_at ASC');
    return reply.send({ users: rows.map(toUserInfo) });
  });

  api.post('/users', async (req, reply) => {
    if (requireAdmin(req, reply) === null) return reply;
    const body = parseWith(reply, createUserBodySchema, req.body);
    if (body === null) return reply;
    if (!pinMeetsRolePolicy(body.pin, body.role)) {
      return sendError(reply, 400, 'weak-pin', '관리자 PIN은 6자리 이상이어야 합니다');
    }
    const row = createUser(db, body, ctx.now().getTime());
    return reply.status(201).send({ user: toUserInfo(row) });
  });

  api.patch<{ Params: { id: string } }>('/users/:id', async (req, reply) => {
    const actor = requireUser(req);
    const body = parseWith(reply, patchUserBodySchema, req.body);
    if (body === null) return reply;

    // 비-admin은 "본인 PIN 변경"만 허용된다 (docs/BACKEND.md §4)
    if (actor.role !== 'admin') {
      if (actor.id !== req.params.id) {
        return sendError(reply, 403, 'forbidden', '다른 사용자의 정보는 관리자만 변경할 수 있습니다');
      }
      const extraKeys = Object.keys(body).filter((k) => k !== 'pin');
      if (extraKeys.length > 0) {
        return sendError(reply, 403, 'forbidden', '본인 계정은 PIN만 변경할 수 있습니다');
      }
    }

    const row = findUserById(db, req.params.id);
    if (row === undefined) {
      return sendError(reply, 404, 'not-found', '해당 사용자가 없습니다');
    }

    // PIN을 바꾼다면 변경 "후" 역할 기준으로 정책을 본다(admin 승격 + 짧은 PIN 동시 차단)
    const nextRole = body.role ?? userRoleSchema.parse(row.role);
    if (body.pin !== undefined && !pinMeetsRolePolicy(body.pin, nextRole)) {
      return sendError(reply, 400, 'weak-pin', '관리자 PIN은 6자리 이상이어야 합니다');
    }

    if (body.name !== undefined) db.run('UPDATE users SET name = ? WHERE id = ?', body.name, row.id);
    if (body.role !== undefined) db.run('UPDATE users SET role = ? WHERE id = ?', body.role, row.id);
    if (body.active !== undefined) {
      db.run('UPDATE users SET active = ? WHERE id = ?', body.active ? 1 : 0, row.id);
    }
    if (body.pin !== undefined) {
      const stored = hashPin(body.pin);
      db.run('UPDATE users SET pin_hash = ?, salt = ? WHERE id = ?', stored.hashB64, stored.saltB64, row.id);
      resetPinFailures(db, row.id); // 새 PIN에는 이전 실패 이력이 무의미하다
    }

    const updated = findUserById(db, row.id);
    if (updated === undefined) {
      return sendError(reply, 500, 'internal', '사용자 갱신 직후 조회에 실패했습니다');
    }
    return reply.send({ user: toUserInfo(updated) });
  });
}
