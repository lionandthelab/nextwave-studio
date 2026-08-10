// api/offline.test.ts — 오프라인 캐시 + outbox 단위 테스트 (node 환경, 인메모리 스토리지).
// 검증 대상 (BACKEND §6):
// - 읽기 캐시 저장/서빙 — 오프라인이면 캐시로 읽고 fetchedAtIso를 보존한다
// - outbox 순서 보존·재전송 — seq 오름차순, 네트워크 실패 시 중단 후 이어서 재전송
// - 409는 자동 병합하지 않고 conflict 레코드로 보존 (엔티티 id별 조회)
// - sendOutboxOp — outbox 항목 → 실제 HTTP 호출 매핑

import { describe, expect, it } from 'vitest';
import { ApiClient, MemorySessionStore } from './client';
import type { FetchInit, FetchLike, TimerHost } from './client';
import { createWorkcellApi } from './resources';
import type { EntityKind } from './resources';
import {
  ApiCache,
  MemoryKVStorage,
  OfflineOutbox,
  MSG_OUTBOX_CORRUPT_KO,
  conflictKey,
  outboxKey,
  lastSyncAgeKo,
  listCacheKey,
  listThroughCache,
  getThroughCache,
  sendOutboxOp,
} from './offline';
import type { FlushReport, OutboxOp, OutboxSendOutcome } from './offline';
import type { EntityMeta, RecordMeta, SaveRequest, TaskDoc } from '../schema/entities';

// ── 픽스처 ──────────────────────────────────────────────────────────

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

function metaRow(id: string): EntityMeta {
  return { id, name: `작업 ${id}`, meta: META, taskSummary: null, processId: null };
}

const fixedNow = (): string => T0;

// ── KVStorage ───────────────────────────────────────────────────────

describe('MemoryKVStorage', () => {
  it('put/get/remove 왕복 + keys는 오름차순 정렬', async () => {
    const kv = new MemoryKVStorage();
    await kv.put('meta', 'b', 2);
    await kv.put('meta', 'a', 1);
    expect(await kv.get('meta', 'a')).toBe(1);
    expect(await kv.keys('meta')).toEqual(['a', 'b']);
    await kv.remove('meta', 'a');
    expect(await kv.get('meta', 'a')).toBeUndefined();
    expect(await kv.keys('meta')).toEqual(['b']);
  });

  it('스토어는 서로 격리된다', async () => {
    const kv = new MemoryKVStorage();
    await kv.put('outbox', 'k', 'outbox-value');
    expect(await kv.get('conflicts', 'k')).toBeUndefined();
  });
});

// ── 캐시 키 · 나이 표기 (순수) ──────────────────────────────────────

describe('listCacheKey', () => {
  it('같은 질의는 같은 키로 수렴한다 (q 트림 포함)', () => {
    expect(listCacheKey()).toBe('q=|del=0|proc=');
    expect(listCacheKey({ q: ' 로봇 ' })).toBe(listCacheKey({ q: '로봇' }));
  });

  it('다른 질의는 다른 키', () => {
    expect(listCacheKey({ includeDeleted: true })).not.toBe(listCacheKey({}));
    expect(listCacheKey({ processId: 'p1' })).not.toBe(listCacheKey({ processId: 'p2' }));
  });
});

describe('lastSyncAgeKo', () => {
  const now = Date.parse(T0);
  it('분/시간/일 단위로 접는다', () => {
    expect(lastSyncAgeKo(T0, now + 30_000)).toBe('방금 전');
    expect(lastSyncAgeKo(T0, now + 5 * 60_000)).toBe('5분 전');
    expect(lastSyncAgeKo(T0, now + 3 * 3_600_000)).toBe('3시간 전');
    expect(lastSyncAgeKo(T0, now + 49 * 3_600_000)).toBe('2일 전');
  });

  it('시각을 해석할 수 없으면 정직하게 표기한다', () => {
    expect(lastSyncAgeKo('not-a-date', now)).toBe('알 수 없음');
  });
});

// ── 읽기 캐시 ───────────────────────────────────────────────────────

describe('ApiCache', () => {
  it('목록 저장/읽기 왕복 — fetchedAtIso는 주입 시각', async () => {
    const cache = new ApiCache(new MemoryKVStorage(), fixedNow);
    await cache.saveList('tasks', listCacheKey(), [metaRow('task-0001')]);
    const hit = await cache.readList('tasks', listCacheKey());
    expect(hit?.items[0]?.id).toBe('task-0001');
    expect(hit?.fetchedAtIso).toBe(T0);
  });

  it('미스는 null (다른 kind/키는 격리)', async () => {
    const cache = new ApiCache(new MemoryKVStorage(), fixedNow);
    await cache.saveList('tasks', listCacheKey(), [metaRow('task-0001')]);
    expect(await cache.readList('processes', listCacheKey())).toBeNull();
    expect(await cache.readList('tasks', listCacheKey({ q: 'x' }))).toBeNull();
  });

  it('단건 저장/읽기 왕복', async () => {
    const cache = new ApiCache(new MemoryKVStorage(), fixedNow);
    await cache.saveRecord('tasks', TASK.id, { doc: TASK, meta: META });
    const hit = await cache.readRecord('tasks', TASK.id);
    expect((hit?.record.doc as TaskDoc).name).toBe('픽앤플레이스');
    expect(await cache.readRecord('tasks', 'task-9999')).toBeNull();
  });
});

// ── outbox 키 (순수) ────────────────────────────────────────────────

describe('outboxKey / conflictKey', () => {
  it('0채움으로 사전순 == 숫자순', () => {
    expect(outboxKey(7)).toBe('000000000007');
    expect(outboxKey(12) > outboxKey(9)).toBe(true); // 문자열 비교로도 순서 유지
    expect(conflictKey('task-0001', 7)).toBe('task-0001::000000000007');
  });
});

// ── outbox 순서 보존 · 재전송 · 충돌 보존 ──────────────────────────

function makeOps(): Array<{
  opKind: OutboxOp['opKind'];
  entityKind: EntityKind;
  entityId: string;
  request?: SaveRequest<unknown> | null;
}> {
  return [
    { opKind: 'create', entityKind: 'tasks', entityId: 'task-0001', request: { doc: TASK, baseVersion: null } },
    { opKind: 'update', entityKind: 'tasks', entityId: 'task-0002', request: { doc: TASK, baseVersion: 3 } },
    { opKind: 'remove', entityKind: 'blocks', entityId: 'block-001' },
  ];
}

describe('OfflineOutbox', () => {
  it('enqueue는 seq를 단조 증가로 발급하고 pending은 적재 순서를 보존한다', async () => {
    const kv = new MemoryKVStorage();
    const outbox = new OfflineOutbox(kv, fixedNow);
    for (const input of makeOps()) await outbox.enqueue(input);
    const pending = await outbox.pending();
    expect(pending.map((op) => op.seq)).toEqual([1, 2, 3]);
    expect(pending.map((op) => op.opKind)).toEqual(['create', 'update', 'remove']);
    expect(pending[0]?.enqueuedAtIso).toBe(T0);
  });

  it('오프라인 2회 저장 → drop 없이 flush하면 자기 자신과 409 (드롭이 필요한 이유)', async () => {
    // 회귀 가드: 같은 baseVersion을 가진 update op 2개를 순서대로 보내면, 첫 op가 서버
    // 버전을 올려 **두 번째 op(사용자의 최신본)** 가 409로 격리된다. UI는 "다른 사용자가
    // 먼저 저장했습니다"를 띄우지만 실제로 먼저 저장한 것은 자기 자신이다.
    const outbox = new OfflineOutbox(new MemoryKVStorage(), fixedNow);
    for (const doc of ['첫 저장', '최신본']) {
      await outbox.enqueue({
        opKind: 'update',
        entityKind: 'tasks',
        entityId: 'task-0001',
        request: { doc: { name: doc }, baseVersion: 7 },
      });
    }
    let serverVersion = 7;
    const report = await outbox.flush((op) => {
      const base = op.request?.baseVersion ?? null;
      if (base !== serverVersion) return Promise.resolve({ kind: 'conflict' as const, current: null });
      serverVersion += 1;
      return Promise.resolve({ kind: 'ok' as const });
    });
    expect(report.sentCount).toBe(1);
    expect(report.conflictCount).toBe(1); // ← 가짜 충돌

    // dropPendingUpdates를 쓰면 최신본 하나만 남아 충돌이 사라진다
    const fixed = new OfflineOutbox(new MemoryKVStorage(), fixedNow);
    for (const doc of ['첫 저장', '최신본']) {
      await fixed.dropPendingUpdates('tasks', 'task-0001');
      await fixed.enqueue({
        opKind: 'update',
        entityKind: 'tasks',
        entityId: 'task-0001',
        request: { doc: { name: doc }, baseVersion: 7 },
      });
    }
    let v = 7;
    const sentDocs: unknown[] = [];
    const fixedReport = await fixed.flush((op) => {
      if ((op.request?.baseVersion ?? null) !== v) {
        return Promise.resolve({ kind: 'conflict' as const, current: null });
      }
      v += 1;
      sentDocs.push(op.request?.doc);
      return Promise.resolve({ kind: 'ok' as const });
    });
    expect(fixedReport).toMatchObject({ sentCount: 1, conflictCount: 0 });
    expect(sentDocs).toEqual([{ name: '최신본' }]); // 마지막 저장이 이긴다
  });

  it('dropPendingUpdates — 같은 개체의 대기 update만 걷어낸다 (가짜 자기-충돌 방지)', async () => {
    const outbox = new OfflineOutbox(new MemoryKVStorage(), fixedNow);
    await outbox.enqueue({
      opKind: 'update',
      entityKind: 'tasks',
      entityId: 'task-0001',
      request: { doc: { name: '옛 저장' }, baseVersion: 7 },
    });
    await outbox.enqueue({
      opKind: 'create',
      entityKind: 'tasks',
      entityId: 'task-0001',
      request: { doc: {}, baseVersion: null },
    });
    await outbox.enqueue({
      opKind: 'update',
      entityKind: 'tasks',
      entityId: 'task-0002', // 다른 개체 — 남아야 한다
      request: { doc: {}, baseVersion: 3 },
    });
    const dropped = await outbox.dropPendingUpdates('tasks', 'task-0001');
    expect(dropped).toBe(1);
    const remaining = await outbox.pending();
    expect(remaining.map((op) => [op.opKind, op.entityId])).toEqual([
      ['create', 'task-0001'],
      ['update', 'task-0002'],
    ]);
  });

  it('seq는 스토리지에 영속된다 — 재기동해도 이어서 발급', async () => {
    const kv = new MemoryKVStorage();
    const first = new OfflineOutbox(kv, fixedNow);
    await first.enqueue(makeOps()[0]!);
    const second = new OfflineOutbox(kv, fixedNow); // 앱 재기동 시뮬레이션
    const op = await second.enqueue(makeOps()[1]!);
    expect(op.seq).toBe(2);
  });

  it('flush — 전부 성공하면 순서대로 보내고 큐를 비운다', async () => {
    const outbox = new OfflineOutbox(new MemoryKVStorage(), fixedNow);
    for (const input of makeOps()) await outbox.enqueue(input);
    const sentSeqs: number[] = [];
    const report = await outbox.flush((op) => {
      sentSeqs.push(op.seq);
      return Promise.resolve({ kind: 'ok' });
    });
    expect(sentSeqs).toEqual([1, 2, 3]);
    expect(report).toEqual({ sentCount: 3, conflictCount: 0, remainingCount: 0, stoppedBy: null });
    expect(await outbox.pending()).toEqual([]);
  });

  it('flush — 네트워크 실패에서 중단하고, 다음 flush가 남은 항목을 이어서 재전송한다', async () => {
    const outbox = new OfflineOutbox(new MemoryKVStorage(), fixedNow);
    for (const input of makeOps()) await outbox.enqueue(input);

    const first = await outbox.flush((op) =>
      Promise.resolve<OutboxSendOutcome>(op.seq === 2 ? { kind: 'network' } : { kind: 'ok' }),
    );
    expect(first).toEqual({ sentCount: 1, conflictCount: 0, remainingCount: 2, stoppedBy: 'network' });
    expect((await outbox.pending()).map((op) => op.seq)).toEqual([2, 3]);

    const resent: number[] = [];
    const second = await outbox.flush((op) => {
      resent.push(op.seq);
      return Promise.resolve({ kind: 'ok' });
    });
    expect(resent).toEqual([2, 3]); // 순서 보존 재전송
    expect(second.stoppedBy).toBeNull();
    expect(await outbox.pending()).toEqual([]);
  });

  it('flush — 409는 자동 병합 없이 conflict 레코드로 보존하고 다음 항목을 계속 보낸다', async () => {
    const outbox = new OfflineOutbox(new MemoryKVStorage(), fixedNow);
    for (const input of makeOps()) await outbox.enqueue(input);
    const serverCurrent = { doc: { ...TASK, name: '서버본' }, meta: { ...META, version: 9 } };

    const report = await outbox.flush((op) =>
      Promise.resolve<OutboxSendOutcome>(
        op.entityId === 'task-0002' ? { kind: 'conflict', current: serverCurrent } : { kind: 'ok' },
      ),
    );
    expect(report).toEqual({ sentCount: 2, conflictCount: 1, remainingCount: 0, stoppedBy: null });
    expect(await outbox.pending()).toEqual([]); // 충돌 항목은 큐가 아니라 conflict 저장소에

    const conflicts = await outbox.conflictsFor('task-0002');
    expect(conflicts.length).toBe(1);
    expect(conflicts[0]?.opKind).toBe('update');
    expect(conflicts[0]?.request?.baseVersion).toBe(3); // 내 것(사본 저장용) 보존
    expect((conflicts[0]?.current?.doc as TaskDoc).name).toBe('서버본'); // 서버본 열기용
    expect(await outbox.conflictsFor('task-0001')).toEqual([]);

    await outbox.removeConflict('task-0002', conflicts[0]!.seq);
    expect(await outbox.conflictsFor('task-0002')).toEqual([]);
  });

  it('flush 재진입은 busy로 거절된다 (이중 전송 방지)', async () => {
    const outbox = new OfflineOutbox(new MemoryKVStorage(), fixedNow);
    await outbox.enqueue(makeOps()[0]!);
    let inner: FlushReport | null = null;
    await outbox.flush(async () => {
      inner = await outbox.flush(() => Promise.resolve({ kind: 'ok' }));
      return { kind: 'ok' };
    });
    expect(inner).not.toBeNull();
    expect((inner as unknown as FlushReport).stoppedBy).toBe('busy');
  });
});

// ── 캐시 경유 읽기 (서버 우선 → 오프라인 캐시 서빙) ────────────────

/** 발화하지 않는 타이머 — 실제 setTimeout이 테스트 프로세스를 붙잡지 않게 한다 */
const inertTimers: TimerHost = { setTimeout: () => 0, clearTimeout: () => undefined };

function makeApi(steps: Array<Response | Error>) {
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

describe('listThroughCache', () => {
  it('서버 성공 → source=server + 캐시에 기록', async () => {
    const cache = new ApiCache(new MemoryKVStorage(), fixedNow);
    const { api } = makeApi([json(200, { items: [metaRow('task-0001')] })]);
    const r = await listThroughCache(api.tasks, cache, {});
    expect(r.kind).toBe('ok');
    if (r.kind === 'ok') {
      expect(r.source).toBe('server');
      expect(r.fetchedAtIso).toBe(T0);
    }
    expect((await cache.readList('tasks', listCacheKey({})))?.items.length).toBe(1);
  });

  it('네트워크 실패 → 캐시 서빙 (source=cache, 마지막 동기화 시각 동봉)', async () => {
    const kv = new MemoryKVStorage();
    const cache = new ApiCache(kv, fixedNow);
    const warm = makeApi([json(200, { items: [metaRow('task-0001')] })]);
    await listThroughCache(warm.api.tasks, cache, {});

    const cold = makeApi([new TypeError('fetch failed')]);
    const r = await listThroughCache(cold.api.tasks, cache, {});
    expect(r.kind).toBe('ok');
    if (r.kind === 'ok') {
      expect(r.source).toBe('cache');
      expect(r.items[0]?.id).toBe('task-0001');
      expect(r.fetchedAtIso).toBe(T0);
    }
  });

  it('캐시도 없으면 network 실패를 그대로 돌려준다', async () => {
    const cache = new ApiCache(new MemoryKVStorage(), fixedNow);
    const { api } = makeApi([new TypeError('fetch failed')]);
    const r = await listThroughCache(api.tasks, cache, {});
    expect(r.kind).toBe('network');
  });

  it('인증 만료는 캐시로 가리지 않는다 (연결 문제가 아니다)', async () => {
    const cache = new ApiCache(new MemoryKVStorage(), fixedNow);
    await cache.saveList('tasks', listCacheKey({}), [metaRow('task-0001')]);
    const { api } = makeApi([json(401, { error: 'unauthorized', messageKo: '만료' })]);
    const r = await listThroughCache(api.tasks, cache, {});
    expect(r.kind).toBe('unauthorized');
  });
});

describe('getThroughCache', () => {
  it('서버 성공 후 오프라인이면 단건도 캐시로 읽는다', async () => {
    const kv = new MemoryKVStorage();
    const cache = new ApiCache(kv, fixedNow);
    const warm = makeApi([json(200, { doc: TASK, meta: META })]);
    const first = await getThroughCache(warm.api.tasks, cache, TASK.id);
    expect(first.kind).toBe('ok');
    if (first.kind === 'ok') expect(first.source).toBe('server');

    const cold = makeApi([new TypeError('fetch failed')]);
    const second = await getThroughCache(cold.api.tasks, cache, TASK.id);
    expect(second.kind).toBe('ok');
    if (second.kind === 'ok') {
      expect(second.source).toBe('cache');
      expect(second.record.doc.name).toBe('픽앤플레이스');
      expect(second.fetchedAtIso).toBe(T0);
    }
  });
});

// ── sendOutboxOp — outbox 항목 → HTTP 매핑 ─────────────────────────

describe('sendOutboxOp', () => {
  function op(partial: Partial<OutboxOp> & Pick<OutboxOp, 'opKind'>): OutboxOp {
    return {
      seq: 1,
      entityKind: 'tasks',
      entityId: TASK.id,
      request: null,
      enqueuedAtIso: T0,
      ...partial,
    };
  }

  it('create → POST /E (SaveRequest 그대로)', async () => {
    const { api, calls } = makeApi([json(201, { doc: TASK, meta: META })]);
    const outcome = await sendOutboxOp(
      api,
      op({ opKind: 'create', request: { doc: TASK, baseVersion: null } }),
    );
    expect(outcome).toEqual({ kind: 'ok' });
    expect(calls[0]?.url).toBe('/api/v1/tasks');
    expect(calls[0]?.init.method).toBe('POST');
  });

  it('update 409 → conflict 결과 + 서버 현재본', async () => {
    const { api } = makeApi([
      json(409, {
        error: 'version-conflict',
        current: { doc: { ...TASK, name: '서버본' }, meta: { ...META, version: 9 } },
      }),
    ]);
    const outcome = await sendOutboxOp(
      api,
      op({ opKind: 'update', request: { doc: TASK, baseVersion: 3 } }),
    );
    expect(outcome.kind).toBe('conflict');
    if (outcome.kind === 'conflict') {
      expect((outcome.current?.doc as TaskDoc).name).toBe('서버본');
    }
  });

  it('remove/restore → DELETE·POST 경로 매핑', async () => {
    const removeApi = makeApi([json(200, { restoreUntilIso: T0 })]);
    expect(await sendOutboxOp(removeApi.api, op({ opKind: 'remove' }))).toEqual({ kind: 'ok' });
    expect(removeApi.calls[0]?.init.method).toBe('DELETE');

    const restoreApi = makeApi([json(200, { doc: TASK, meta: META })]);
    expect(await sendOutboxOp(restoreApi.api, op({ opKind: 'restore' }))).toEqual({ kind: 'ok' });
    expect(restoreApi.calls[0]?.url).toBe('/api/v1/tasks/task-0001/restore');
  });

  it('손상된 항목(request 없음)은 fetch 없이 error — 조용한 no-op 금지', async () => {
    const { api, calls } = makeApi([]);
    const created = await sendOutboxOp(api, op({ opKind: 'create', request: null }));
    expect(created).toEqual({ kind: 'error', messageKo: MSG_OUTBOX_CORRUPT_KO });
    const updated = await sendOutboxOp(
      api,
      op({ opKind: 'update', request: { doc: TASK, baseVersion: null } }),
    );
    expect(updated).toEqual({ kind: 'error', messageKo: MSG_OUTBOX_CORRUPT_KO });
    expect(calls.length).toBe(0);
  });
});
