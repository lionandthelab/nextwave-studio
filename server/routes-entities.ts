// server/routes-entities.ts — 개체 4종 공통 CRUD + 조언적 잠금(locks)
//
// 계약의 "왜"(docs/BACKEND.md §1·§4):
// - 저장은 낙관적 버전: baseVersion이 서버 버전과 다르면 409 + 서버 현재본을 돌려준다.
//   자동 병합은 하지 않는다 — 수작업 존중, 해결은 사람이 한다(클라이언트 UI 몫).
// - 삭제는 soft-delete(휴지통 TRASH_RETENTION_DAYS일) + restore. 완전 삭제는 목록 조회
//   시점의 지연 퍼지뿐이다 — 파괴는 언제나 되돌릴 수 있다(§2.11 정신).
// - 잠금은 조언적(advisory)이다: TTL이 지나면 자연 해제, 강탈 API는 없다. 잠금이
//   저장을 막지도 않는다 — 최종 안전망은 버전 충돌 409다.
// - 서버는 payload(scene 등)를 해석하지 않는다 — entities.ts 스키마 검증까지만.

import type { FastifyInstance } from 'fastify';
import { z } from 'zod';
import {
  LOCK_TTL_SEC,
  TRASH_RETENTION_DAYS,
  blockDocSchema,
  deviceDocSchema,
  lockInfoSchema,
  processDocSchema,
  runResultSchema,
  taskDocSchema,
  versionSchema,
} from '../src/schema/entities';
import type {
  EntityMeta,
  LockInfo,
  RecordMeta,
  TaskSummary,
} from '../src/schema/entities';
import { parseWith, requireUser, sendError, zodFirstIssue } from './app';
import { DAY_MS, msToIso } from './db';
import type { RouteContext } from './db';

// ── kind 정의 ───────────────────────────────────────────────────────

type EntityKind = 'process' | 'task' | 'block' | 'device';

interface KindDef {
  /** URL 세그먼트 (복수형) */
  readonly plural: string;
  /** entities.kind 컬럼 값 (단수형 — locks의 kind와 동일 축) */
  readonly kind: EntityKind;
  readonly docSchema: z.ZodTypeAny;
}

const KIND_DEFS: readonly KindDef[] = [
  { plural: 'processes', kind: 'process', docSchema: processDocSchema },
  { plural: 'tasks', kind: 'task', docSchema: taskDocSchema },
  { plural: 'blocks', kind: 'block', docSchema: blockDocSchema },
  { plural: 'devices', kind: 'device', docSchema: deviceDocSchema },
];

/** 모든 개체 문서가 공유하는 최소 형태 — docSchema 통과 후 안전하게 읽는다 */
interface EntityDocBase {
  readonly id: string;
  readonly name: string;
}

const saveRequestSchema = z.object({
  doc: z.unknown(),
  baseVersion: versionSchema.nullable(),
});

export const TRASH_RETENTION_MS = TRASH_RETENTION_DAYS * DAY_MS;
export const LOCK_TTL_MS = LOCK_TTL_SEC * 1000;

// ── 순수 헬퍼 (payload 요약 — vitest node 환경에서 직접 테스트) ─────

/** payload.sequence.steps 길이 — 서버는 시퀀스를 해석하지 않으므로 구조만 안전하게 센다 */
export function extractStepCount(payload: unknown): number {
  if (typeof payload !== 'object' || payload === null) return 0;
  const seq = (payload as Record<string, unknown>)['sequence'];
  if (typeof seq !== 'object' || seq === null) return 0;
  const steps = (seq as Record<string, unknown>)['steps'];
  return Array.isArray(steps) ? steps.length : 0;
}

/** payload.thumbnail이 비어 있지 않은 문자열인가 */
export function extractHasThumbnail(payload: unknown): boolean {
  if (typeof payload !== 'object' || payload === null) return false;
  const thumb = (payload as Record<string, unknown>)['thumbnail'];
  return typeof thumb === 'string' && thumb.length > 0;
}

/**
 * 최근 실행 payload → TaskSummary.lastRun. 실행 종료 시각(endedAtIso)을 우선하고,
 * payload가 깨졌으면(정상 경로에서는 없음) DB의 started_at으로 강등한다.
 */
export function extractLastRun(
  runPayload: unknown,
  fallbackAtMs: number,
): TaskSummary['lastRun'] {
  if (typeof runPayload !== 'object' || runPayload === null) return null;
  const rec = runPayload as Record<string, unknown>;
  const result = runResultSchema.safeParse(rec['result']);
  if (!result.success) return null;
  const ended = rec['endedAtIso'];
  const atIso =
    typeof ended === 'string' && !Number.isNaN(Date.parse(ended)) ? ended : msToIso(fallbackAtMs);
  return { atIso, result: result.data };
}

/** soft-delete 시각 → 복원 마감(ms) */
export function restoreDeadlineMs(deletedAtMs: number): number {
  return deletedAtMs + TRASH_RETENTION_MS;
}

// ── DB 행 → API 봉투 ────────────────────────────────────────────────

interface EntityRow {
  readonly kind: string;
  readonly id: string;
  readonly name: string;
  readonly process_id: string | null;
  readonly payload: string;
  readonly version: number;
  readonly created_at: number;
  readonly created_by: string;
  readonly updated_at: number;
  readonly updated_by: string;
  readonly deleted_at: number | null;
  readonly deleted_by: string | null;
  readonly created_by_name: string | null;
  readonly updated_by_name: string | null;
  readonly deleted_by_name: string | null;
}

/** 감사 이름은 조회 시점 조인 — users는 비활성화만 있고 삭제가 없어 조인이 안전하다 */
const SELECT_ENTITY = `
  SELECT e.*,
         cu.name AS created_by_name,
         uu.name AS updated_by_name,
         du.name AS deleted_by_name
    FROM entities e
    LEFT JOIN users cu ON cu.id = e.created_by
    LEFT JOIN users uu ON uu.id = e.updated_by
    LEFT JOIN users du ON du.id = e.deleted_by`;

const UNKNOWN_USER_NAME = '알 수 없음';

function rowToMeta(row: EntityRow): RecordMeta {
  return {
    version: row.version,
    createdAtIso: msToIso(row.created_at),
    createdBy: row.created_by,
    createdByName: row.created_by_name ?? UNKNOWN_USER_NAME,
    updatedAtIso: msToIso(row.updated_at),
    updatedBy: row.updated_by,
    updatedByName: row.updated_by_name ?? UNKNOWN_USER_NAME,
    deletedAtIso: row.deleted_at === null ? null : msToIso(row.deleted_at),
    deletedByName: row.deleted_by === null ? null : (row.deleted_by_name ?? UNKNOWN_USER_NAME),
  };
}

function rowToEnvelope(row: EntityRow): { doc: unknown; meta: RecordMeta } {
  return { doc: JSON.parse(row.payload) as unknown, meta: rowToMeta(row) };
}

// ── 잠금 ────────────────────────────────────────────────────────────

const lockKindSchema = lockInfoSchema.shape.entityKind;
const lockActionSchema = z.object({ action: z.enum(['acquire', 'heartbeat', 'release']) });

interface LockRow {
  readonly kind: string;
  readonly id: string;
  readonly user_id: string;
  readonly acquired_at: number;
  readonly expires_at: number;
  readonly user_name: string | null;
}

const SELECT_LOCK = `
  SELECT l.*, u.name AS user_name
    FROM locks l
    LEFT JOIN users u ON u.id = l.user_id
   WHERE l.kind = ? AND l.id = ?`;

function lockRowToInfo(row: LockRow): LockInfo {
  return {
    entityKind: lockKindSchema.parse(row.kind),
    entityId: row.id,
    userId: row.user_id,
    userName: row.user_name ?? UNKNOWN_USER_NAME,
    acquiredAtIso: msToIso(row.acquired_at),
    expiresAtIso: msToIso(row.expires_at),
  };
}

// ── 라우트 등록 ─────────────────────────────────────────────────────

export function registerEntityRoutes(api: FastifyInstance, ctx: RouteContext): void {
  const { db } = ctx;

  const fetchRow = (kind: EntityKind, id: string): EntityRow | undefined =>
    db.one<EntityRow>(`${SELECT_ENTITY} WHERE e.kind = ? AND e.id = ?`, kind, id);

  /** 목록 조회 시 지연 퍼지 — 휴지통 보존 기한이 지난 행을 그때 완전 삭제한다 */
  const purgeExpiredTrash = (nowMs: number): void => {
    db.run('DELETE FROM entities WHERE deleted_at IS NOT NULL AND deleted_at <= ?', nowMs - TRASH_RETENTION_MS);
  };

  const taskSummaryFor = (row: EntityRow): TaskSummary => {
    const payload = JSON.parse(row.payload) as unknown;
    const lastRunRow = db.one<{ payload: string; started_at: number }>(
      'SELECT payload, started_at FROM runs WHERE task_id = ? ORDER BY started_at DESC LIMIT 1',
      row.id,
    );
    return {
      stepCount: extractStepCount(payload),
      hasThumbnail: extractHasThumbnail(payload),
      lastRun:
        lastRunRow === undefined
          ? null
          : extractLastRun(JSON.parse(lastRunRow.payload) as unknown, lastRunRow.started_at),
    };
  };

  for (const def of KIND_DEFS) {
    // ── 목록 ──────────────────────────────────────────────────────
    api.get(`/${def.plural}`, async (req, reply) => {
      const nowMs = ctx.now().getTime();
      purgeExpiredTrash(nowMs);

      const query = req.query as Record<string, string | undefined>;
      const includeDeleted = query['includeDeleted'] === '1' || query['includeDeleted'] === 'true';
      const clauses = ['e.kind = ?'];
      const params: unknown[] = [def.kind];
      if (!includeDeleted) clauses.push('e.deleted_at IS NULL');
      const q = query['q']?.trim();
      if (q !== undefined && q !== '') {
        clauses.push('e.name LIKE ?');
        params.push(`%${q}%`);
      }
      const processId = query['processId']?.trim();
      if (def.kind === 'task' && processId !== undefined && processId !== '') {
        clauses.push('e.process_id = ?');
        params.push(processId);
      }

      const rows = db.all<EntityRow>(
        `${SELECT_ENTITY} WHERE ${clauses.join(' AND ')} ORDER BY e.updated_at DESC`,
        ...params,
      );
      const items: EntityMeta[] = rows.map((row) => ({
        id: row.id,
        name: row.name,
        meta: rowToMeta(row),
        taskSummary: def.kind === 'task' ? taskSummaryFor(row) : null,
        processId: row.process_id,
      }));
      return reply.send({ items });
    });

    // ── 생성 (id는 클라이언트 발급 uuid — 오프라인 생성 지원) ─────
    api.post(`/${def.plural}`, async (req, reply) => {
      const user = requireUser(req);
      const body = parseWith(reply, saveRequestSchema, req.body);
      if (body === null) return reply;
      if (body.baseVersion !== null) {
        return sendError(reply, 400, 'validation', '신규 생성은 baseVersion이 null이어야 합니다');
      }
      const parsedDoc = def.docSchema.safeParse(body.doc);
      if (!parsedDoc.success) {
        return sendError(reply, 400, 'validation', `문서가 스키마를 통과하지 못했습니다 — ${zodFirstIssue(parsedDoc.error)}`);
      }
      const doc = parsedDoc.data as EntityDocBase & Record<string, unknown>;
      if (fetchRow(def.kind, doc.id) !== undefined) {
        return sendError(reply, 409, 'duplicate-id', '이미 같은 id의 문서가 있습니다 — 새로 만들려면 다른 id가 필요합니다');
      }
      const nowMs = ctx.now().getTime();
      const processId = def.kind === 'task' ? ((doc['processId'] as string | null | undefined) ?? null) : null;
      db.run(
        `INSERT INTO entities
           (kind, id, name, process_id, payload, version,
            created_at, created_by, updated_at, updated_by, deleted_at, deleted_by)
         VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, ?, NULL, NULL)`,
        def.kind,
        doc.id,
        doc.name,
        processId,
        JSON.stringify(doc),
        nowMs,
        user.id,
        nowMs,
        user.id,
      );
      const row = fetchRow(def.kind, doc.id);
      if (row === undefined) return sendError(reply, 500, 'internal', '생성 직후 조회에 실패했습니다');
      return reply.status(201).send(rowToEnvelope(row));
    });

    // ── 단건 조회 (휴지통 행도 meta.deletedAtIso로 표시해 반환) ───
    api.get<{ Params: { id: string } }>(`/${def.plural}/:id`, async (req, reply) => {
      const row = fetchRow(def.kind, req.params.id);
      if (row === undefined) return sendError(reply, 404, 'not-found', '해당 문서가 없습니다');
      return reply.send(rowToEnvelope(row));
    });

    // ── 저장 (낙관적 버전 — 불일치면 409 + 현재본) ────────────────
    api.put<{ Params: { id: string } }>(`/${def.plural}/:id`, async (req, reply) => {
      const user = requireUser(req);
      const body = parseWith(reply, saveRequestSchema, req.body);
      if (body === null) return reply;
      if (body.baseVersion === null) {
        return sendError(reply, 400, 'validation', '기존 문서 저장에는 baseVersion(정수)이 필요합니다');
      }
      const parsedDoc = def.docSchema.safeParse(body.doc);
      if (!parsedDoc.success) {
        return sendError(reply, 400, 'validation', `문서가 스키마를 통과하지 못했습니다 — ${zodFirstIssue(parsedDoc.error)}`);
      }
      const doc = parsedDoc.data as EntityDocBase & Record<string, unknown>;
      if (doc.id !== req.params.id) {
        return sendError(reply, 400, 'validation', '문서 id와 URL id가 다릅니다');
      }
      const row = fetchRow(def.kind, req.params.id);
      if (row === undefined) return sendError(reply, 404, 'not-found', '해당 문서가 없습니다');
      if (row.version !== body.baseVersion) {
        // 자동 병합하지 않는다 — 현재본을 실어 보내고 해결은 사람이 한다(BACKEND §6)
        return sendError(reply, 409, 'version-conflict', '다른 사용자가 먼저 저장했습니다 — 서버 현재본을 확인하세요', {
          current: rowToEnvelope(row),
        });
      }
      const nowMs = ctx.now().getTime();
      const processId = def.kind === 'task' ? ((doc['processId'] as string | null | undefined) ?? null) : null;
      db.run(
        `UPDATE entities
            SET name = ?, process_id = ?, payload = ?, version = version + 1,
                updated_at = ?, updated_by = ?
          WHERE kind = ? AND id = ?`,
        doc.name,
        processId,
        JSON.stringify(doc),
        nowMs,
        user.id,
        def.kind,
        req.params.id,
      );
      const updated = fetchRow(def.kind, req.params.id);
      if (updated === undefined) return sendError(reply, 500, 'internal', '저장 직후 조회에 실패했습니다');
      return reply.send(rowToEnvelope(updated));
    });

    // ── soft-delete (완전 삭제 아님 — 휴지통 이동) ────────────────
    api.delete<{ Params: { id: string } }>(`/${def.plural}/:id`, async (req, reply) => {
      const user = requireUser(req);
      const row = fetchRow(def.kind, req.params.id);
      if (row === undefined) return sendError(reply, 404, 'not-found', '해당 문서가 없습니다');
      // 이미 휴지통이면 시각을 덮어쓰지 않는다(반복 삭제로 보존 기한이 늘어나면 안 된다)
      const deletedAtMs = row.deleted_at ?? ctx.now().getTime();
      if (row.deleted_at === null) {
        db.run(
          'UPDATE entities SET deleted_at = ?, deleted_by = ? WHERE kind = ? AND id = ?',
          deletedAtMs,
          user.id,
          def.kind,
          req.params.id,
        );
      }
      return reply.send({ restoreUntilIso: msToIso(restoreDeadlineMs(deletedAtMs)) });
    });

    // ── 복원 ──────────────────────────────────────────────────────
    api.post<{ Params: { id: string } }>(`/${def.plural}/:id/restore`, async (req, reply) => {
      const row = fetchRow(def.kind, req.params.id);
      if (row === undefined) {
        return sendError(reply, 404, 'not-found', '해당 문서가 없습니다 — 휴지통 보존 기한이 지나 완전 삭제되었을 수 있습니다');
      }
      if (row.deleted_at !== null) {
        db.run(
          'UPDATE entities SET deleted_at = NULL, deleted_by = NULL WHERE kind = ? AND id = ?',
          def.kind,
          req.params.id,
        );
      }
      const restored = fetchRow(def.kind, req.params.id);
      if (restored === undefined) return sendError(reply, 500, 'internal', '복원 직후 조회에 실패했습니다');
      return reply.send(rowToEnvelope(restored));
    });
  }

  // ── 조언적 잠금 (acquire/heartbeat/release, TTL 자연 해제) ────────

  const fetchLock = (kind: string, id: string, nowMs: number): LockRow | undefined => {
    const row = db.one<LockRow>(SELECT_LOCK, kind, id);
    if (row === undefined) return undefined;
    if (row.expires_at <= nowMs) {
      // 만료 잠금은 조회 시점에 지연 삭제 — heartbeat가 끊긴 편집자는 자연 해제된다
      db.run('DELETE FROM locks WHERE kind = ? AND id = ?', kind, id);
      return undefined;
    }
    return row;
  };

  api.get<{ Params: { kind: string; id: string } }>('/locks/:kind/:id', async (req, reply) => {
    const kind = lockKindSchema.safeParse(req.params.kind);
    if (!kind.success) {
      return sendError(reply, 400, 'validation', '잠금 대상 kind는 task·process·block 중 하나입니다');
    }
    const row = fetchLock(kind.data, req.params.id, ctx.now().getTime());
    return reply.send({ lock: row === undefined ? null : lockRowToInfo(row) });
  });

  api.post<{ Params: { kind: string; id: string } }>('/locks/:kind/:id', async (req, reply) => {
    const user = requireUser(req);
    const kind = lockKindSchema.safeParse(req.params.kind);
    if (!kind.success) {
      return sendError(reply, 400, 'validation', '잠금 대상 kind는 task·process·block 중 하나입니다');
    }
    const body = parseWith(reply, lockActionSchema, req.body);
    if (body === null) return reply;
    const nowMs = ctx.now().getTime();
    const existing = fetchLock(kind.data, req.params.id, nowMs);

    // 타인의 유효 잠금 — 어떤 action도 423(강탈 없음, 만료 대기 또는 소유자 해제뿐)
    if (existing !== undefined && existing.user_id !== user.id) {
      return sendError(reply, 423, 'locked', `${existing.user_name ?? UNKNOWN_USER_NAME} 님이 편집 중입니다`, {
        lock: lockRowToInfo(existing),
      });
    }

    if (body.action === 'release') {
      if (existing !== undefined) db.run('DELETE FROM locks WHERE kind = ? AND id = ?', kind.data, req.params.id);
      return reply.send({ lock: null });
    }

    // acquire와 heartbeat는 같은 갱신이다 — heartbeat가 만료 직후 도착해도(잠금 소실)
    // 소유자 연속성이 유지되도록 재획득으로 처리한다. acquired_at은 최초 획득 시각 유지.
    const acquiredAt = existing?.acquired_at ?? nowMs;
    db.run(
      `INSERT INTO locks (kind, id, user_id, acquired_at, expires_at) VALUES (?, ?, ?, ?, ?)
       ON CONFLICT(kind, id) DO UPDATE SET user_id = excluded.user_id,
         acquired_at = excluded.acquired_at, expires_at = excluded.expires_at`,
      kind.data,
      req.params.id,
      user.id,
      acquiredAt,
      nowMs + LOCK_TTL_MS,
    );
    const fresh = fetchLock(kind.data, req.params.id, nowMs);
    if (fresh === undefined) return sendError(reply, 500, 'internal', '잠금 획득 직후 조회에 실패했습니다');
    return reply.send({ lock: lockRowToInfo(fresh) });
  });
}
