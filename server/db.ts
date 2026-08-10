// server/db.ts — better-sqlite3 오픈 · 전진 전용 마이그레이션 · prepared statement 캐시
//
// 왜 단일 테이블 entities인가: 문서 개체 4종(process/task/block/device)은 봉투 구조가
// 동일하다(payload JSON + 감사 메타 + soft-delete). kind 컬럼으로 구분하면 개체 종류가
// 늘어도 마이그레이션 표면이 늘지 않는다(docs/BACKEND.md §5).
//
// 왜 ms 정수 타임스탬프인가: TTL/퍼지 계산이 전부 산술이 된다. ISO 문자열은 API 경계
// (routes)에서만 만든다 — DB와 시각 표현을 분리해야 시각 주입 테스트가 단순해진다.

import { mkdirSync } from 'node:fs';
import { dirname } from 'node:path';
import Database from 'better-sqlite3';

export type SqliteDb = Database.Database;

// ── 마이그레이션 (전진 전용 — 버전 정수 + SQL, docs/BACKEND.md §5) ──

export interface Migration {
  readonly version: number;
  readonly sql: string;
}

export const MIGRATIONS: readonly Migration[] = [
  {
    version: 1,
    sql: `
      CREATE TABLE users (
        id              TEXT PRIMARY KEY,
        name            TEXT NOT NULL,
        role            TEXT NOT NULL,
        pin_hash        TEXT NOT NULL,
        salt            TEXT NOT NULL,
        active          INTEGER NOT NULL DEFAULT 1,
        failed_attempts INTEGER NOT NULL DEFAULT 0,
        locked_until    INTEGER,
        created_at      INTEGER NOT NULL
      );

      CREATE TABLE sessions (
        token_hash   TEXT PRIMARY KEY,
        user_id      TEXT NOT NULL REFERENCES users(id),
        created_at   INTEGER NOT NULL,
        expires_at   INTEGER NOT NULL,
        last_seen_at INTEGER NOT NULL
      );
      CREATE INDEX idx_sessions_user ON sessions(user_id);

      CREATE TABLE entities (
        kind       TEXT NOT NULL,
        id         TEXT NOT NULL,
        name       TEXT NOT NULL,
        process_id TEXT,
        payload    TEXT NOT NULL,
        version    INTEGER NOT NULL,
        created_at INTEGER NOT NULL,
        created_by TEXT NOT NULL,
        updated_at INTEGER NOT NULL,
        updated_by TEXT NOT NULL,
        deleted_at INTEGER,
        deleted_by TEXT,
        PRIMARY KEY (kind, id)
      );
      CREATE INDEX idx_entities_updated ON entities(kind, updated_at DESC);

      CREATE TABLE locks (
        kind        TEXT NOT NULL,
        id          TEXT NOT NULL,
        user_id     TEXT NOT NULL,
        acquired_at INTEGER NOT NULL,
        expires_at  INTEGER NOT NULL,
        PRIMARY KEY (kind, id)
      );

      -- append-only: UPDATE/DELETE 경로가 코드에 존재하지 않는다 (docs/BACKEND.md §1)
      CREATE TABLE runs (
        id          TEXT PRIMARY KEY,
        task_id     TEXT NOT NULL,
        payload     TEXT NOT NULL,
        started_at  INTEGER NOT NULL,
        operator_id TEXT NOT NULL
      );
      CREATE INDEX idx_runs_task ON runs(task_id, started_at DESC);
    `,
  },
];

/** user_version pragma 기준으로 미적용 마이그레이션만 순서대로 적용한다(전진 전용). */
export function migrate(db: SqliteDb, migrations: readonly Migration[] = MIGRATIONS): void {
  const current = db.pragma('user_version', { simple: true }) as number;
  const pending = [...migrations]
    .sort((a, b) => a.version - b.version)
    .filter((m) => m.version > current);
  for (const m of pending) {
    // 마이그레이션 하나 = 트랜잭션 하나 — 중간 실패 시 반쯤 적용된 스키마를 남기지 않는다
    db.transaction(() => {
      db.exec(m.sql);
      db.pragma(`user_version = ${m.version}`);
    })();
  }
}

// ── DB 핸들 (prepared statement 캐시 포함) ──────────────────────────

export interface DbHandle {
  readonly raw: SqliteDb;
  /** 단건 조회 — 없으면 undefined */
  one<Row>(sql: string, ...params: unknown[]): Row | undefined;
  /** 전체 조회 */
  all<Row>(sql: string, ...params: unknown[]): Row[];
  /** 쓰기 실행 */
  run(sql: string, ...params: unknown[]): Database.RunResult;
  /** 즉시 실행 트랜잭션 래퍼 */
  transaction<T>(fn: () => T): T;
  close(): void;
}

/**
 * DB를 연다(파일 경로 또는 ':memory:').
 * - 파일 DB는 WAL 모드 — 읽기(목록 폴링)와 쓰기(저장)가 서로를 막지 않는다.
 * - 같은 SQL은 한 번만 prepare한다 — 라우트 핸들러가 문자열 SQL을 반복 호출해도
 *   프레임 예산(단일 파일 DB, 동기 API)을 지킨다.
 */
export function openDb(dbPath: string): DbHandle {
  const isMemory = dbPath === ':memory:';
  if (!isMemory) mkdirSync(dirname(dbPath), { recursive: true });
  const raw = new Database(dbPath);
  // WAL은 파일 DB 전용 — :memory:에는 의미가 없어 명시적으로 건너뛴다
  if (!isMemory) raw.pragma('journal_mode = WAL');
  raw.pragma('foreign_keys = ON');
  migrate(raw);

  const cache = new Map<string, Database.Statement<unknown[], unknown>>();
  const stmt = (sql: string): Database.Statement<unknown[], unknown> => {
    const hit = cache.get(sql);
    if (hit !== undefined) return hit;
    const prepared = raw.prepare(sql);
    cache.set(sql, prepared);
    return prepared;
  };

  return {
    raw,
    one: <Row>(sql: string, ...params: unknown[]): Row | undefined =>
      stmt(sql).get(...params) as Row | undefined,
    all: <Row>(sql: string, ...params: unknown[]): Row[] => stmt(sql).all(...params) as Row[],
    run: (sql: string, ...params: unknown[]): Database.RunResult => stmt(sql).run(...params),
    transaction: <T>(fn: () => T): T => raw.transaction(fn)(),
    close: (): void => {
      raw.close();
    },
  };
}

// ── 라우트 공유 컨텍스트 ────────────────────────────────────────────

/**
 * 라우트 모듈이 공유하는 최소 컨텍스트. now는 주입 가능해야 한다 — TTL(세션·잠금)과
 * 휴지통 퍼지를 테스트가 시계 조작(fake timer) 없이 검증하기 위한 계약이다.
 */
export interface RouteContext {
  readonly db: DbHandle;
  readonly now: () => Date;
  readonly serverName: string;
}

export const DAY_MS = 86_400_000;

/** ms 타임스탬프 → ISO 8601 (API 경계 전용 — DB에는 ISO를 저장하지 않는다) */
export function msToIso(ms: number): string {
  return new Date(ms).toISOString();
}
