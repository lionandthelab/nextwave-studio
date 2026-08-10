// server/routes-runs.ts — 실행 기록(append-only) + 작업 통계
//
// 왜 append-only인가(docs/BACKEND.md §1): Run은 "무슨 일이 있었나"의 감사 기록이다.
// 수정·삭제 API가 아예 없어야 기록을 신뢰할 수 있다 — 라우트 자체를 만들지 않는다.
// operator는 클라이언트 주장을 믿지 않고 세션 토큰의 사용자로 강제 주입한다.

import type { FastifyInstance } from 'fastify';
import { runRecordSchema } from '../src/schema/entities';
import type { RunRecord, TaskStats } from '../src/schema/entities';
import { parseWith, requireUser, sendError } from './app';
import type { RouteContext } from './db';

// ── 순수 로직 (vitest node 환경에서 직접 테스트) ────────────────────

export const RUNS_DEFAULT_LIMIT = 50;
export const RUNS_MAX_LIMIT = 200;
export const TOP_COLLISION_NODES_MAX = 5;

/**
 * TaskStats 계산. topCollisionNodes는 "예기치 않은" 충돌만 센다 — 의도된 접촉(파지 등)
 * 은 결함 신호가 아니다(core/collision-classify와 같은 구분 축). 동률은 nodeId 사전순
 * 으로 고정해 결과를 결정론적으로 만든다.
 */
export function computeTaskStats(records: readonly RunRecord[]): TaskStats {
  const runCount = records.length;
  const successCount = records.filter((r) => r.result === 'completed').length;
  const avgDurationSec =
    runCount === 0 ? null : records.reduce((sum, r) => sum + r.wallTimeSec, 0) / runCount;

  const counts = new Map<string, number>();
  for (const record of records) {
    for (const collision of record.collisions) {
      if (collision.classification !== 'unexpected' || collision.nodeId === null) continue;
      counts.set(collision.nodeId, (counts.get(collision.nodeId) ?? 0) + 1);
    }
  }
  const topCollisionNodes = [...counts.entries()]
    .map(([nodeId, count]) => ({ nodeId, count }))
    .sort((a, b) => b.count - a.count || a.nodeId.localeCompare(b.nodeId))
    .slice(0, TOP_COLLISION_NODES_MAX);

  return { runCount, successCount, avgDurationSec, topCollisionNodes };
}

/** 목록 limit/offset 정규화 — 잘못된 값은 조용히 기본값/경계로 강등 */
export function normalizePagination(
  rawLimit: string | undefined,
  rawOffset: string | undefined,
): { limit: number; offset: number } {
  const limitNum = Number(rawLimit);
  const offsetNum = Number(rawOffset);
  const limit =
    Number.isInteger(limitNum) && limitNum >= 1 ? Math.min(limitNum, RUNS_MAX_LIMIT) : RUNS_DEFAULT_LIMIT;
  const offset = Number.isInteger(offsetNum) && offsetNum >= 0 ? offsetNum : 0;
  return { limit, offset };
}

// ── 라우트 등록 ─────────────────────────────────────────────────────

interface RunRow {
  readonly id: string;
  readonly payload: string;
  readonly started_at: number;
}

export function registerRunRoutes(api: FastifyInstance, ctx: RouteContext): void {
  const { db } = ctx;

  api.post('/runs', async (req, reply) => {
    const user = requireUser(req);
    const parsed = parseWith(reply, runRecordSchema, req.body);
    if (parsed === null) return reply;
    // operator는 세션이 진실 — 클라이언트가 보낸 값은 무엇이든 덮어쓴다
    const record: RunRecord = { ...parsed, operatorId: user.id, operatorName: user.name };
    const existing = db.one<{ id: string }>('SELECT id FROM runs WHERE id = ?', record.id);
    if (existing !== undefined) {
      return sendError(reply, 409, 'duplicate-id', '이미 같은 id의 실행 기록이 있습니다');
    }
    db.run(
      'INSERT INTO runs (id, task_id, payload, started_at, operator_id) VALUES (?, ?, ?, ?, ?)',
      record.id,
      record.taskId,
      JSON.stringify(record),
      Date.parse(record.startedAtIso),
      user.id,
    );
    return reply.status(201).send({ id: record.id });
  });

  api.get('/runs', async (req, reply) => {
    const query = req.query as Record<string, string | undefined>;
    const { limit, offset } = normalizePagination(query['limit'], query['offset']);
    const taskId = query['taskId']?.trim();
    const hasFilter = taskId !== undefined && taskId !== '';

    const where = hasFilter ? 'WHERE task_id = ?' : '';
    const filterParams: unknown[] = hasFilter ? [taskId] : [];
    const total = db.one<{ n: number }>(`SELECT COUNT(*) AS n FROM runs ${where}`, ...filterParams)?.n ?? 0;
    const rows = db.all<RunRow>(
      `SELECT id, payload, started_at FROM runs ${where} ORDER BY started_at DESC LIMIT ? OFFSET ?`,
      ...filterParams,
      limit,
      offset,
    );
    // payload는 저장 시점에 runRecordSchema를 통과했다 — 재검증 없이 역직렬화만 한다
    const items = rows.map((row) => JSON.parse(row.payload) as RunRecord);
    return reply.send({ items, total });
  });

  api.get<{ Params: { id: string } }>('/runs/:id', async (req, reply) => {
    const row = db.one<RunRow>('SELECT id, payload, started_at FROM runs WHERE id = ?', req.params.id);
    if (row === undefined) return sendError(reply, 404, 'not-found', '해당 실행 기록이 없습니다');
    return reply.send(JSON.parse(row.payload) as RunRecord);
  });

  // 작업 통계 — 존재하지 않는 taskId도 0건 통계로 답한다(오프라인 생성 작업의 기록이
  // 문서보다 먼저 동기화될 수 있다). 404로 막을 이유가 없다.
  api.get<{ Params: { id: string } }>('/tasks/:id/stats', async (req, reply) => {
    const rows = db.all<RunRow>(
      'SELECT id, payload, started_at FROM runs WHERE task_id = ? ORDER BY started_at DESC',
      req.params.id,
    );
    const records = rows.map((row) => JSON.parse(row.payload) as RunRecord);
    return reply.send(computeTaskStats(records));
  });
}
