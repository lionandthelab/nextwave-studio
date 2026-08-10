// server/app.ts — Fastify 앱 조립: 인증 preHandler · 오류 봉투 · 라우트 등록
//
// listen하지 않는다 — 테스트는 app.inject로, 실행은 index.ts가 담당한다(관심사 분리).
// 시각(now)은 주입 가능하다: 세션 TTL·잠금 만료·휴지통 퍼지를 테스트가 시계 조작 없이
// 검증하기 위한 계약이다(docs/BACKEND.md §7).

import Fastify from 'fastify';
import type { FastifyInstance, FastifyReply, FastifyRequest } from 'fastify';
import type { z } from 'zod';
import pkg from '../package.json';
import { API_PREFIX } from '../src/schema/entities';
import type { UserInfo } from '../src/schema/entities';
import { findSessionUser } from './auth';
import { DEFAULT_SERVER_NAME } from './config';
import { openDb } from './db';
import type { RouteContext } from './db';
import { registerAuthRoutes } from './routes-auth';
import { registerEntityRoutes } from './routes-entities';
import { registerRunRoutes } from './routes-runs';

declare module 'fastify' {
  interface FastifyRequest {
    /** 인증 preHandler가 채운다 — 공개 경로에서는 null */
    authUser: UserInfo | null;
  }
}

// ── 공용 응답 헬퍼 (라우트 모듈이 재사용) ───────────────────────────

/**
 * 오류 봉투 { error, messageKo } + 상태 코드 (docs/BACKEND.md §4).
 * extra는 스펙이 요구하는 추가 필드(retryAfterSec, lock, current 등)를 싣는다.
 */
export function sendError(
  reply: FastifyReply,
  status: number,
  code: string,
  messageKo: string,
  extra: Record<string, unknown> = {},
): FastifyReply {
  return reply.status(status).send({ error: code, messageKo, ...extra });
}

export function zodFirstIssue(error: z.ZodError): string {
  const issue = error.issues[0];
  if (issue === undefined) return '알 수 없는 검증 오류';
  const path = issue.path.join('.');
  return path === '' ? issue.message : `${path}: ${issue.message}`;
}

/** zod 검증 실패 시 400 봉투를 보내고 null을 반환한다(호출부는 즉시 return reply). */
export function parseWith<S extends z.ZodTypeAny>(
  reply: FastifyReply,
  schema: S,
  data: unknown,
): z.infer<S> | null {
  const result = schema.safeParse(data);
  if (!result.success) {
    sendError(reply, 400, 'validation', `요청이 스키마를 통과하지 못했습니다 — ${zodFirstIssue(result.error)}`);
    return null;
  }
  return result.data as z.infer<S>;
}

/** 인증 훅 이후에만 호출 — 훅이 보장하므로 null이면 프로그래밍 오류다. */
export function requireUser(req: FastifyRequest): UserInfo {
  if (req.authUser === null) throw new Error('requireUser는 인증 훅 이후에만 호출해야 합니다');
  return req.authUser;
}

/** admin이 아니면 403 봉투를 보내고 null을 반환한다. */
export function requireAdmin(req: FastifyRequest, reply: FastifyReply): UserInfo | null {
  const user = requireUser(req);
  if (user.role !== 'admin') {
    sendError(reply, 403, 'forbidden', '관리자(admin) 전용 기능입니다');
    return null;
  }
  return user;
}

// ── 앱 빌드 ─────────────────────────────────────────────────────────

export interface BuildAppOptions {
  /** SQLite 파일 경로 또는 ':memory:' (테스트) */
  readonly dbPath: string;
  /** 시각 주입 (기본: 실제 시계) */
  readonly now?: () => Date;
  readonly serverName?: string;
}

/** 인증 없이 접근 가능한 경로 (docs/BACKEND.md §4 첫 문단) — 그 외 전부 Bearer 필수 */
const PUBLIC_ROUTES: ReadonlySet<string> = new Set([
  'GET /health',
  'GET /auth/bootstrap',
  'GET /auth/users',
  'POST /auth/setup',
  'POST /auth/login',
]);

export function buildApp(opts: BuildAppOptions): FastifyInstance {
  const db = openDb(opts.dbPath);
  const now = opts.now ?? ((): Date => new Date());
  const ctx: RouteContext = { db, now, serverName: opts.serverName ?? DEFAULT_SERVER_NAME };
  const startedAtMs = now().getTime();

  const app = Fastify({
    logger: false,
    // task 문서는 씬 전체 사본 + 썸네일/임포트 메시 data URI를 봉투로 나른다 —
    // fastify 기본 1MB로는 정상 문서도 413이 난다.
    bodyLimit: 32 * 1024 * 1024,
  });

  app.decorateRequest('authUser', null);

  // 모든 오류를 봉투로 통일 — JSON 파싱 실패(400)든 핸들러 예외(500)든
  // 클라이언트는 항상 { error, messageKo }를 받는다. 던져진 값은 무엇이든 올 수 있어
  // unknown으로 받고 필요한 필드만 좁혀 읽는다.
  app.setErrorHandler((err: unknown, _req, reply) => {
    const e = err as { statusCode?: unknown; code?: unknown; message?: unknown };
    const status =
      typeof e.statusCode === 'number' && e.statusCode >= 400 && e.statusCode <= 599
        ? e.statusCode
        : 500;
    if (status >= 500) console.error('[workcell-server] 처리되지 않은 오류:', err);
    void reply.status(status).send({
      error: status >= 500 ? 'internal' : typeof e.code === 'string' ? e.code : 'bad-request',
      messageKo:
        status >= 500
          ? '서버 내부 오류가 발생했습니다'
          : typeof e.message === 'string' && e.message !== ''
            ? e.message
            : '잘못된 요청입니다',
    });
  });

  void app.register(
    async (api) => {
      // 인증 게이트 — 공개 경로 외 전부 Bearer 토큰 검증(401), 유효하면 슬라이딩 갱신
      api.addHook('preHandler', async (req, reply) => {
        const path = (req.url.split('?')[0] ?? '').slice(API_PREFIX.length);
        if (PUBLIC_ROUTES.has(`${req.method} ${path}`)) return;
        const header = req.headers.authorization;
        if (header === undefined || !header.startsWith('Bearer ')) {
          sendError(reply, 401, 'unauthorized', '로그인이 필요합니다');
          return reply;
        }
        const token = header.slice('Bearer '.length).trim();
        const user = findSessionUser(db, token, now().getTime());
        if (user === null) {
          sendError(reply, 401, 'unauthorized', '세션이 유효하지 않습니다 — 다시 로그인하세요');
          return reply;
        }
        req.authUser = user;
        return;
      });

      api.get('/health', async () => ({
        ok: true,
        name: ctx.serverName,
        version: pkg.version,
        uptimeSec: Math.max(0, Math.round((now().getTime() - startedAtMs) / 1000)),
      }));

      registerAuthRoutes(api, ctx);
      registerEntityRoutes(api, ctx);
      registerRunRoutes(api, ctx);

      // API 접두 아래의 미등록 경로 — SPA 폴백(index.ts)이 아닌 JSON 404 봉투
      api.setNotFoundHandler((_req, reply) => {
        sendError(reply, 404, 'not-found', '요청한 API 경로가 없습니다');
      });
    },
    { prefix: API_PREFIX },
  );

  app.addHook('onClose', (_instance, done) => {
    db.close();
    done();
  });

  return app;
}
