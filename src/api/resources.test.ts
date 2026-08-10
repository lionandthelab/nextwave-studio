// api/resources.test.ts — 타입드 리소스 클라이언트 단위 테스트 (node 환경).
// fake fetch 주입으로 경로/메서드/본문(SaveRequest)과 결과 union 매핑을 검증한다:
// - 409 → 'conflict' + 서버 현재본 추출 (자동 병합 금지 — BACKEND §6)
// - 423 → 'held' + 잠금 정보 (강탈 없음 — BACKEND §1.5)
// - 쿼리 문자열 빌더(순수), 실패 3종(network/unauthorized/error) 접기

import { describe, expect, it } from 'vitest';
import { ApiClient, MemorySessionStore } from './client';
import type { FetchInit, FetchLike, TimerHost } from './client';
import {
  buildListQuery,
  buildRunsQuery,
  createWorkcellApi,
  extractConflictCurrent,
  extractLock,
} from './resources';
import type {
  LockInfo,
  RecordMeta,
  RunRecord,
  SaveRequest,
  TaskDoc,
  TaskStats,
} from '../schema/entities';

// ── 테스트 대역 ─────────────────────────────────────────────────────

type FetchStep = Response | Error;

/** 발화하지 않는 타이머 — 실제 setTimeout이 테스트 프로세스를 붙잡지 않게 한다 */
const inertTimers: TimerHost = { setTimeout: () => 0, clearTimeout: () => undefined };

function makeApi(steps: FetchStep[]) {
  const calls: Array<{ url: string; init: FetchInit }> = [];
  const queue = [...steps];
  const fetchFn: FetchLike = (url, init) => {
    calls.push({ url, init });
    const step = queue.shift();
    if (step === undefined) return Promise.reject(new Error('fake fetch: 예약된 응답 없음'));
    if (step instanceof Error) return Promise.reject(step);
    return Promise.resolve(step);
  };
  const storage = new MemorySessionStore();
  storage.setItem(
    'workcell.session',
    JSON.stringify({
      token: 'tok-1',
      user: { id: 'user-0001', name: '김설치', role: 'tech', active: true },
    }),
  );
  const client = new ApiClient({ fetchFn, storage, timers: inertTimers });
  return { api: createWorkcellApi(client), calls };
}

function json(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

const T0 = '2026-08-07T09:00:00.000Z';

const META: RecordMeta = {
  version: 3,
  createdAtIso: T0,
  createdBy: 'user-0001',
  createdByName: '김설치',
  updatedAtIso: T0,
  updatedBy: 'user-0001',
  updatedByName: '김설치',
  deletedAtIso: null,
  deletedByName: null,
};

const TASK: TaskDoc = {
  id: 'task-0001',
  name: '픽앤플레이스',
  processId: null,
  sceneOrigin: null,
  scene: {},
  sequence: null,
  assets: {},
  thumbnail: null,
  notes: '',
};

const LOCK: LockInfo = {
  entityKind: 'task',
  entityId: 'task-0001',
  userId: 'user-0002',
  userName: '박기사',
  acquiredAtIso: T0,
  expiresAtIso: '2026-08-07T09:01:30.000Z',
};

// ── 쿼리 빌더 (순수) ────────────────────────────────────────────────

describe('buildListQuery', () => {
  it('빈 옵션이면 빈 문자열 (쿼리 없음)', () => {
    expect(buildListQuery()).toBe('');
    expect(buildListQuery({})).toBe('');
    expect(buildListQuery({ q: '   ', includeDeleted: false, processId: '' })).toBe('');
  });

  it('설정된 옵션만 포함하고 q는 트림한다', () => {
    expect(buildListQuery({ q: ' 로봇 ' })).toBe(`?${new URLSearchParams({ q: '로봇' })}`);
    expect(buildListQuery({ includeDeleted: true })).toBe('?includeDeleted=1');
    expect(buildListQuery({ processId: 'proc-0001' })).toBe('?processId=proc-0001');
    expect(buildListQuery({ q: 'a b', includeDeleted: true, processId: 'proc-0001' })).toBe(
      '?q=a+b&includeDeleted=1&processId=proc-0001',
    );
  });
});

describe('buildRunsQuery', () => {
  it('taskId/limit/offset 조합', () => {
    expect(buildRunsQuery()).toBe('');
    expect(buildRunsQuery({ taskId: 'task-0001', limit: 10, offset: 20 })).toBe(
      '?taskId=task-0001&limit=10&offset=20',
    );
    expect(buildRunsQuery({ limit: 0, offset: 0 })).toBe('?limit=0&offset=0');
  });
});

// ── 방어적 추출기 (순수) ────────────────────────────────────────────

describe('extractConflictCurrent', () => {
  it('ConflictResponse 봉투에서 현재본을 추출한다', () => {
    const current = extractConflictCurrent<TaskDoc>({
      error: 'version-conflict',
      current: { doc: TASK, meta: META },
    });
    expect(current?.doc.id).toBe('task-0001');
    expect(current?.meta.version).toBe(3);
  });

  it('형태가 어긋나면 null (throw하지 않는다)', () => {
    expect(extractConflictCurrent(null)).toBeNull();
    expect(extractConflictCurrent({})).toBeNull();
    expect(extractConflictCurrent({ current: null })).toBeNull();
    expect(extractConflictCurrent({ current: { doc: TASK } })).toBeNull(); // meta 없음
  });
});

describe('extractLock', () => {
  it('423 본문에서 잠금 정보를 추출한다', () => {
    expect(extractLock({ error: 'locked', lock: LOCK })?.userName).toBe('박기사');
  });

  it('형태가 어긋나면 null', () => {
    expect(extractLock(null)).toBeNull();
    expect(extractLock({ lock: null })).toBeNull();
    expect(extractLock({ lock: { userId: 'u' } })).toBeNull(); // expiresAtIso 없음
  });
});

// ── 개체 CRUD ───────────────────────────────────────────────────────

describe('EntityClient', () => {
  it('list — GET /E + 쿼리, EntityMeta 행 반환', async () => {
    const meta = { id: 'task-0001', name: '픽앤플레이스', meta: META, taskSummary: null, processId: null };
    const { api, calls } = makeApi([json(200, { items: [meta] })]);
    const r = await api.tasks.list({ processId: 'proc-0001' });
    expect(calls[0]?.url).toBe('/api/v1/tasks?processId=proc-0001');
    expect(calls[0]?.init.method).toBe('GET');
    expect(r.kind).toBe('ok');
    if (r.kind === 'ok') expect(r.items[0]?.id).toBe('task-0001');
  });

  it('create — POST /E, 본문은 SaveRequest{doc, baseVersion: null}', async () => {
    const { api, calls } = makeApi([json(201, { doc: TASK, meta: META })]);
    const r = await api.tasks.create(TASK);
    expect(calls[0]?.url).toBe('/api/v1/tasks');
    expect(calls[0]?.init.method).toBe('POST');
    const body = JSON.parse(calls[0]?.init.body ?? '{}') as SaveRequest<TaskDoc>;
    expect(body.baseVersion).toBeNull();
    expect(body.doc.id).toBe('task-0001');
    expect(r.kind).toBe('ok');
    if (r.kind === 'ok') expect(r.record.meta.version).toBe(3);
  });

  it('update — PUT /E/:id, baseVersion 동봉', async () => {
    const { api, calls } = makeApi([json(200, { doc: TASK, meta: { ...META, version: 4 } })]);
    const r = await api.tasks.update('task-0001', TASK, 3);
    expect(calls[0]?.url).toBe('/api/v1/tasks/task-0001');
    expect(calls[0]?.init.method).toBe('PUT');
    const body = JSON.parse(calls[0]?.init.body ?? '{}') as SaveRequest<TaskDoc>;
    expect(body.baseVersion).toBe(3);
    expect(r.kind).toBe('ok');
  });

  it('update 409 → conflict + 서버 현재본 (판별 가능한 union)', async () => {
    const serverDoc = { ...TASK, name: '다른 사용자가 고친 이름' };
    const { api } = makeApi([
      json(409, {
        error: 'version-conflict',
        messageKo: '다른 사용자가 먼저 저장했습니다',
        current: { doc: serverDoc, meta: { ...META, version: 5 } },
      }),
    ]);
    const r = await api.tasks.update('task-0001', TASK, 3);
    expect(r.kind).toBe('conflict');
    if (r.kind === 'conflict') {
      expect(r.current?.doc.name).toBe('다른 사용자가 고친 이름');
      expect(r.current?.meta.version).toBe(5);
      expect(r.messageKo).toBe('다른 사용자가 먼저 저장했습니다');
    }
  });

  it('update 409 본문이 손상돼도 conflict로 분류된다 (current만 null)', async () => {
    const { api } = makeApi([json(409, { error: 'version-conflict' })]);
    const r = await api.tasks.update('task-0001', TASK, 3);
    expect(r.kind).toBe('conflict');
    if (r.kind === 'conflict') expect(r.current).toBeNull();
  });

  it('remove — DELETE /E/:id → restoreUntilIso (soft-delete)', async () => {
    const { api, calls } = makeApi([json(200, { restoreUntilIso: '2026-09-06T09:00:00.000Z' })]);
    const r = await api.blocks.remove('block-001');
    expect(calls[0]?.url).toBe('/api/v1/blocks/block-001');
    expect(calls[0]?.init.method).toBe('DELETE');
    expect(r).toEqual({ kind: 'ok', restoreUntilIso: '2026-09-06T09:00:00.000Z' });
  });

  it('restore — POST /E/:id/restore', async () => {
    const { api, calls } = makeApi([json(200, { doc: TASK, meta: META })]);
    const r = await api.tasks.restore('task-0001');
    expect(calls[0]?.url).toBe('/api/v1/tasks/task-0001/restore');
    expect(calls[0]?.init.method).toBe('POST');
    expect(r.kind).toBe('ok');
  });

  it('네트워크 실패는 network 실패로 접힌다', async () => {
    const { api } = makeApi([new TypeError('fetch failed')]);
    const r = await api.processes.list();
    expect(r.kind).toBe('network');
  });

  it('401은 unauthorized 실패로 접힌다', async () => {
    const { api } = makeApi([json(401, { error: 'unauthorized', messageKo: '만료' })]);
    const r = await api.devices.get('dev-0001');
    expect(r).toEqual({ kind: 'unauthorized', messageKo: '만료' });
  });
});

// ── 잠금 ────────────────────────────────────────────────────────────

describe('locks', () => {
  it('acquire 성공 — POST /locks/:kind/:id + action 본문', async () => {
    const { api, calls } = makeApi([json(200, { lock: LOCK })]);
    const r = await api.locks('task', 'task-0001', 'acquire');
    expect(calls[0]?.url).toBe('/api/v1/locks/task/task-0001');
    expect(JSON.parse(calls[0]?.init.body ?? '{}')).toEqual({ action: 'acquire' });
    expect(r.kind).toBe('ok');
    if (r.kind === 'ok') expect(r.lock?.userName).toBe('박기사');
  });

  it('423 → held + 보유자 정보 (강탈 없음)', async () => {
    const { api } = makeApi([
      json(423, { error: 'locked', messageKo: '박기사 님이 편집 중입니다', lock: LOCK }),
    ]);
    const r = await api.locks('task', 'task-0001', 'acquire');
    expect(r.kind).toBe('held');
    if (r.kind === 'held') {
      expect(r.lock?.userId).toBe('user-0002');
      expect(r.messageKo).toBe('박기사 님이 편집 중입니다');
    }
  });

  it('getLock — GET /locks/:kind/:id (없으면 null)', async () => {
    const { api, calls } = makeApi([json(200, { lock: null })]);
    const r = await api.getLock('process', 'proc-0001');
    expect(calls[0]?.url).toBe('/api/v1/locks/process/proc-0001');
    expect(r).toEqual({ kind: 'ok', lock: null });
  });
});

// ── 실행 기록 · 통계 ────────────────────────────────────────────────

const RUN: RunRecord = {
  id: 'run-00001',
  taskId: 'task-0001',
  taskName: '픽앤플레이스',
  taskVersion: 3,
  processId: null,
  operatorId: 'user-0001',
  operatorName: '김설치',
  startedAtIso: T0,
  endedAtIso: '2026-08-07T09:02:00.000Z',
  result: 'completed',
  stepsTotal: 5,
  stepsDone: 5,
  simTimeSec: 12.5,
  wallTimeSec: 13.1,
  collisions: [],
  interventions: [],
};

describe('runs / taskStats', () => {
  it('runs.create — POST /runs (append-only) → id', async () => {
    const { api, calls } = makeApi([json(201, { id: 'run-00001' })]);
    const r = await api.runs.create(RUN);
    expect(calls[0]?.url).toBe('/api/v1/runs');
    expect(r).toEqual({ kind: 'ok', id: 'run-00001' });
  });

  it('runs.list — 쿼리 + total', async () => {
    const { api, calls } = makeApi([json(200, { items: [RUN], total: 37 })]);
    const r = await api.runs.list({ taskId: 'task-0001', limit: 10, offset: 20 });
    expect(calls[0]?.url).toBe('/api/v1/runs?taskId=task-0001&limit=10&offset=20');
    expect(r.kind).toBe('ok');
    if (r.kind === 'ok') {
      expect(r.total).toBe(37);
      expect(r.items[0]?.result).toBe('completed');
    }
  });

  it('runs.get — GET /runs/:id', async () => {
    const { api, calls } = makeApi([json(200, RUN)]);
    const r = await api.runs.get('run-00001');
    expect(calls[0]?.url).toBe('/api/v1/runs/run-00001');
    expect(r.kind).toBe('ok');
    if (r.kind === 'ok') expect(r.run.taskName).toBe('픽앤플레이스');
  });

  it('taskStats — GET /tasks/:id/stats', async () => {
    const stats: TaskStats = {
      runCount: 10,
      successCount: 8,
      avgDurationSec: 12.5,
      topCollisionNodes: [{ nodeId: 'node-3', count: 2 }],
    };
    const { api, calls } = makeApi([json(200, stats)]);
    const r = await api.taskStats('task-0001');
    expect(calls[0]?.url).toBe('/api/v1/tasks/task-0001/stats');
    expect(r.kind).toBe('ok');
    if (r.kind === 'ok') expect(r.stats.successCount).toBe(8);
  });
});
