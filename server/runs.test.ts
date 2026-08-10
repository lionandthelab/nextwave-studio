// server/runs.test.ts — 실행 기록 append-only · operator 강제 주입 · 목록/통계 (BACKEND §4·§7)
//
// Run은 감사 기록이다: 삭제 라우트가 아예 없어야 하고(404), operator는 클라이언트
// 주장이 아니라 세션 토큰의 사용자여야 한다. 통계 계산(computeTaskStats)은 순수
// 함수로도 직접 검증한다.

import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import type { EntityMeta, RunRecord, TaskStats } from '../src/schema/entities';
import { computeTaskStats, normalizePagination } from './routes-runs';
import {
  bearer,
  createTestServer,
  createUserAndLogin,
  makeCollision,
  makeRunRecord,
  makeTaskDoc,
} from './test-util';
import type { TestServer } from './test-util';

let server: TestServer;

beforeEach(async () => {
  server = await createTestServer();
});
afterEach(async () => {
  await server.app.close();
});

const post = (url: string, payload: object, token = server.adminToken) =>
  server.app.inject({ method: 'POST', url: `/api/v1${url}`, headers: bearer(token), payload });
const get = (url: string, token = server.adminToken) =>
  server.app.inject({ method: 'GET', url: `/api/v1${url}`, headers: bearer(token) });

// ── 순수 로직 ───────────────────────────────────────────────────────

describe('computeTaskStats (순수)', () => {
  it('빈 기록 → 0 통계 + avgDurationSec null', () => {
    expect(computeTaskStats([])).toEqual({
      runCount: 0,
      successCount: 0,
      avgDurationSec: null,
      topCollisionNodes: [],
    });
  });

  it('completed만 성공으로 세고, 평균은 wallTimeSec, 충돌 상위는 unexpected+nodeId만', () => {
    const records: RunRecord[] = [
      makeRunRecord('run-pure-001', 'task-pure-01', {
        result: 'completed',
        wallTimeSec: 10,
        collisions: [
          makeCollision('node-a', 'unexpected'),
          makeCollision('node-a', 'unexpected'),
          makeCollision('node-b', 'unexpected'),
          makeCollision('node-c', 'intended'), // 의도된 접촉은 결함 신호가 아니다
          makeCollision(null, 'unexpected'), // 노드 밖 충돌은 노드 통계에서 제외
        ],
      }),
      makeRunRecord('run-pure-002', 'task-pure-01', { result: 'error', wallTimeSec: 20 }),
      makeRunRecord('run-pure-003', 'task-pure-01', { result: 'completed', wallTimeSec: 30 }),
    ];
    expect(computeTaskStats(records)).toEqual({
      runCount: 3,
      successCount: 2,
      avgDurationSec: 20,
      topCollisionNodes: [
        { nodeId: 'node-a', count: 2 },
        { nodeId: 'node-b', count: 1 },
      ],
    });
  });

  it('상위 노드는 최대 5개, 동률은 nodeId 사전순으로 결정론적', () => {
    const collisions = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6'].map((n) => makeCollision(n, 'unexpected'));
    const stats = computeTaskStats([makeRunRecord('run-pure-004', 'task-pure-01', { collisions })]);
    expect(stats.topCollisionNodes).toHaveLength(5);
    expect(stats.topCollisionNodes.map((n) => n.nodeId)).toEqual(['n1', 'n2', 'n3', 'n4', 'n5']);
  });
});

describe('normalizePagination (순수)', () => {
  it('기본 50, 상한 200, 음수/쓰레기는 강등', () => {
    expect(normalizePagination(undefined, undefined)).toEqual({ limit: 50, offset: 0 });
    expect(normalizePagination('10', '5')).toEqual({ limit: 10, offset: 5 });
    expect(normalizePagination('999', '-1')).toEqual({ limit: 200, offset: 0 });
    expect(normalizePagination('abc', 'xyz')).toEqual({ limit: 50, offset: 0 });
  });
});

// ── HTTP 왕복 ───────────────────────────────────────────────────────

describe('runs API', () => {
  it('operator는 세션 사용자로 강제 주입된다 — 클라이언트 스푸핑 무시', async () => {
    const tech = await createUserAndLogin(server, '설치기사 정', '6060', 'tech');
    const record = makeRunRecord('run-op-00001', 'task-op-0001'); // 팩토리가 스푸핑 값을 넣는다
    const created = await post('/runs', record, tech.token);
    expect(created.statusCode).toBe(201);
    expect(created.json()).toEqual({ id: 'run-op-00001' });

    const fetched = await get('/runs/run-op-00001');
    expect(fetched.statusCode).toBe(200);
    const stored = fetched.json() as RunRecord;
    expect(stored.operatorId).toBe(tech.user.id);
    expect(stored.operatorName).toBe('설치기사 정');
  });

  it('append-only — 같은 id 재기록은 409, 삭제 라우트는 존재하지 않는다(404)', async () => {
    await post('/runs', makeRunRecord('run-ap-00001', 'task-ap-0001'));
    expect((await post('/runs', makeRunRecord('run-ap-00001', 'task-ap-0001'))).statusCode).toBe(409);

    const deleted = await server.app.inject({
      method: 'DELETE',
      url: '/api/v1/runs/run-ap-00001',
      headers: bearer(server.adminToken),
    });
    expect(deleted.statusCode).toBe(404);
    // 여전히 조회된다 — 기록은 지워지지 않았다
    expect((await get('/runs/run-ap-00001')).statusCode).toBe(200);
  });

  it('스키마 위반 기록은 400 — 검증 없이 저장되지 않는다', async () => {
    const broken = { ...makeRunRecord('run-bad-0001', 'task-bad-001') } as Record<string, unknown>;
    delete broken['endedAtIso'];
    expect((await post('/runs', broken)).statusCode).toBe(400);
  });

  it('목록 — startedAt 내림차순, taskId 필터, limit/offset과 total', async () => {
    const t0 = '2026-08-07T00:00:00.000Z';
    const t1 = '2026-08-07T00:00:10.000Z';
    const t2 = '2026-08-07T00:00:20.000Z';
    await post('/runs', makeRunRecord('run-ls-00001', 'task-ls-0001', { startedAtIso: t0 }));
    await post('/runs', makeRunRecord('run-ls-00002', 'task-ls-0001', { startedAtIso: t1 }));
    await post('/runs', makeRunRecord('run-ls-00003', 'task-ls-0001', { startedAtIso: t2 }));
    await post('/runs', makeRunRecord('run-ls-other1', 'task-ls-9999', { startedAtIso: t2 }));

    const all = (await get('/runs')).json() as { items: RunRecord[]; total: number };
    expect(all.total).toBe(4);

    const filtered = (await get('/runs?taskId=task-ls-0001')).json() as { items: RunRecord[]; total: number };
    expect(filtered.total).toBe(3);
    expect(filtered.items.map((r) => r.id)).toEqual(['run-ls-00003', 'run-ls-00002', 'run-ls-00001']);

    const page = (await get('/runs?taskId=task-ls-0001&limit=2')).json() as { items: RunRecord[]; total: number };
    expect(page.total).toBe(3); // total은 페이지와 무관하게 전체 건수
    expect(page.items.map((r) => r.id)).toEqual(['run-ls-00003', 'run-ls-00002']);

    const rest = (await get('/runs?taskId=task-ls-0001&limit=2&offset=2')).json() as { items: RunRecord[] };
    expect(rest.items.map((r) => r.id)).toEqual(['run-ls-00001']);

    expect((await get('/runs/run-none-0001')).statusCode).toBe(404);
  });

  it('GET /tasks/:id/stats — 기록 집계, 기록이 없으면 0 통계', async () => {
    await post('/runs', makeRunRecord('run-st-00001', 'task-st-0001', {
      result: 'completed',
      wallTimeSec: 10,
      startedAtIso: '2026-08-07T00:00:00.000Z',
      collisions: [makeCollision('node-x', 'unexpected'), makeCollision('node-x', 'unexpected')],
    }));
    await post('/runs', makeRunRecord('run-st-00002', 'task-st-0001', {
      result: 'error',
      wallTimeSec: 20,
      startedAtIso: '2026-08-07T00:01:00.000Z',
      collisions: [makeCollision('node-y', 'unexpected')],
    }));

    const stats = (await get('/tasks/task-st-0001/stats')).json() as TaskStats;
    expect(stats).toEqual({
      runCount: 2,
      successCount: 1,
      avgDurationSec: 15,
      topCollisionNodes: [
        { nodeId: 'node-x', count: 2 },
        { nodeId: 'node-y', count: 1 },
      ],
    });

    const empty = (await get('/tasks/task-st-none/stats')).json() as TaskStats;
    expect(empty).toEqual({ runCount: 0, successCount: 0, avgDurationSec: null, topCollisionNodes: [] });
  });

  it('작업 목록의 taskSummary.lastRun이 최근 실행을 반영한다', async () => {
    await post('/tasks', { doc: makeTaskDoc('task-lr-0001'), baseVersion: null });
    await post('/runs', makeRunRecord('run-lr-00001', 'task-lr-0001', {
      startedAtIso: '2026-08-07T00:00:00.000Z',
      result: 'completed',
    }));
    await post('/runs', makeRunRecord('run-lr-00002', 'task-lr-0001', {
      startedAtIso: '2026-08-07T00:05:00.000Z',
      result: 'stopped',
      wallTimeSec: 30,
    }));

    const items = ((await get('/tasks')).json() as { items: EntityMeta[] }).items;
    const item = items.find((i) => i.id === 'task-lr-0001');
    expect(item?.taskSummary?.lastRun).toEqual({
      atIso: '2026-08-07T00:05:30.000Z', // 최근 실행의 endedAtIso (started + wallTimeSec)
      result: 'stopped',
    });
  });
});
