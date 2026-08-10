// server/entities.test.ts — 개체 CRUD 왕복 · 버전 충돌 409 · 휴지통/퍼지 · 잠금 (BACKEND §4·§7)
//
// 핵심 계약:
// - 저장은 낙관적 버전 — 불일치 409에는 반드시 서버 "현재본"이 실려 있어야 한다
//   (클라이언트 충돌 해결 UI의 재료).
// - 삭제는 soft — 목록에서 사라지되 restore로 돌아오고, 30일이 지나서야
//   목록 조회 시점에 완전 삭제된다(시각 주입으로 검증).
// - 잠금은 조언적 TTL 90초 — 타인 보유는 423, 만료는 자연 해제.

import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import type { EntityMeta, LockInfo, RecordMeta } from '../src/schema/entities';
import { extractHasThumbnail, extractLastRun, extractStepCount } from './routes-entities';
import {
  ADMIN_PIN,
  bearer,
  createTestServer,
  createUserAndLogin,
  makeBlockDoc,
  makeDeviceDoc,
  makeProcessDoc,
  makeTaskDoc,
} from './test-util';
import type { TestServer } from './test-util';

interface Envelope {
  doc: Record<string, unknown>;
  meta: RecordMeta;
}

let server: TestServer;

beforeEach(async () => {
  server = await createTestServer();
});
afterEach(async () => {
  await server.app.close();
});

const post = (url: string, payload: object, token = server.adminToken) =>
  server.app.inject({ method: 'POST', url: `/api/v1${url}`, headers: bearer(token), payload });
const put = (url: string, payload: object, token = server.adminToken) =>
  server.app.inject({ method: 'PUT', url: `/api/v1${url}`, headers: bearer(token), payload });
const get = (url: string, token = server.adminToken) =>
  server.app.inject({ method: 'GET', url: `/api/v1${url}`, headers: bearer(token) });
const del = (url: string, token = server.adminToken) =>
  server.app.inject({ method: 'DELETE', url: `/api/v1${url}`, headers: bearer(token) });

// ── 순수 요약 헬퍼 ──────────────────────────────────────────────────

describe('payload 요약 헬퍼 (순수)', () => {
  it('extractStepCount — sequence.steps 배열 길이, 구조가 다르면 0', () => {
    expect(extractStepCount({ sequence: { steps: [1, 2, 3] } })).toBe(3);
    expect(extractStepCount({ sequence: null })).toBe(0);
    expect(extractStepCount({})).toBe(0);
    expect(extractStepCount(null)).toBe(0);
    expect(extractStepCount({ sequence: { steps: '아님' } })).toBe(0);
  });

  it('extractHasThumbnail — 비어 있지 않은 문자열만 true', () => {
    expect(extractHasThumbnail({ thumbnail: 'data:image/png;base64,AAAA' })).toBe(true);
    expect(extractHasThumbnail({ thumbnail: '' })).toBe(false);
    expect(extractHasThumbnail({ thumbnail: null })).toBe(false);
    expect(extractHasThumbnail(undefined)).toBe(false);
  });

  it('extractLastRun — endedAtIso와 result를 뽑고, 깨진 payload는 null/폴백', () => {
    const run = { endedAtIso: '2026-08-07T01:00:00.000Z', result: 'completed' };
    expect(extractLastRun(run, 0)).toEqual({ atIso: '2026-08-07T01:00:00.000Z', result: 'completed' });
    expect(extractLastRun({ result: 'stopped' }, 1000)).toEqual({
      atIso: '1970-01-01T00:00:01.000Z',
      result: 'stopped',
    });
    expect(extractLastRun({ result: '이상한값' }, 0)).toBeNull();
    expect(extractLastRun(null, 0)).toBeNull();
  });
});

// ── CRUD 왕복 (4종 공통) ────────────────────────────────────────────

describe('개체 CRUD 왕복', () => {
  it('processes/tasks/blocks/devices 각각 생성 → 조회 → 저장(버전 증가)', async () => {
    const cases: { plural: string; doc: Record<string, unknown> }[] = [
      { plural: 'processes', doc: makeProcessDoc('proc-rt-0001') as unknown as Record<string, unknown> },
      { plural: 'tasks', doc: makeTaskDoc('task-rt-0001') as unknown as Record<string, unknown> },
      { plural: 'blocks', doc: makeBlockDoc('block-rt-0001') as unknown as Record<string, unknown> },
      { plural: 'devices', doc: makeDeviceDoc('device-rt-0001') as unknown as Record<string, unknown> },
    ];
    for (const { plural, doc } of cases) {
      const created = await post(`/${plural}`, { doc, baseVersion: null });
      expect(created.statusCode).toBe(201);
      const createdBody = created.json() as Envelope;
      expect(createdBody.meta.version).toBe(1);
      expect(createdBody.meta.createdByName).toBe('관리자');
      expect(createdBody.doc).toEqual(doc); // 왕복 손실 없음

      const fetched = await get(`/${plural}/${String(doc['id'])}`);
      expect(fetched.statusCode).toBe(200);
      expect((fetched.json() as Envelope).doc).toEqual(doc);

      const renamed = { ...doc, name: '개정판' };
      const saved = await put(`/${plural}/${String(doc['id'])}`, { doc: renamed, baseVersion: 1 });
      expect(saved.statusCode).toBe(200);
      const savedBody = saved.json() as Envelope;
      expect(savedBody.meta.version).toBe(2);
      expect(savedBody.doc['name']).toBe('개정판');

      // 목록은 payload 없는 메타만 — doc 키가 없어야 한다
      const list = await get(`/${plural}`);
      const items = (list.json() as { items: EntityMeta[] }).items;
      const item = items.find((i) => i.id === doc['id']);
      expect(item).toBeDefined();
      expect(item?.name).toBe('개정판');
      expect('doc' in (item as unknown as Record<string, unknown>)).toBe(false);
    }
  });

  it('중복 id 생성은 409, 스키마 위반 문서는 400, 없는 문서 PUT은 404', async () => {
    const doc = makeProcessDoc('proc-dup-001');
    expect((await post('/processes', { doc, baseVersion: null })).statusCode).toBe(201);
    expect((await post('/processes', { doc, baseVersion: null })).statusCode).toBe(409);

    const invalid = await post('/processes', { doc: { id: 'proc-bad-001' }, baseVersion: null });
    expect(invalid.statusCode).toBe(400);

    const missing = await put('/processes/proc-none-01', {
      doc: makeProcessDoc('proc-none-01'),
      baseVersion: 1,
    });
    expect(missing.statusCode).toBe(404);

    // 문서 id와 URL id 불일치 — 잘못 겨눈 저장은 거부
    const mismatch = await put('/processes/proc-dup-001', {
      doc: makeProcessDoc('proc-other-01'),
      baseVersion: 1,
    });
    expect(mismatch.statusCode).toBe(400);
  });

  it('인증 없이 개체 API 접근은 401 봉투', async () => {
    const res = await server.app.inject({ method: 'GET', url: '/api/v1/processes' });
    expect(res.statusCode).toBe(401);
    expect((res.json() as { error: string }).error).toBe('unauthorized');
  });
});

// ── 버전 충돌 ───────────────────────────────────────────────────────

describe('낙관적 버전', () => {
  it('낡은 baseVersion 저장은 409 + 서버 현재본 포함', async () => {
    const doc = makeProcessDoc('proc-ver-001');
    await post('/processes', { doc, baseVersion: null });

    const first = await put('/processes/proc-ver-001', {
      doc: { ...doc, name: '개정판 A' },
      baseVersion: 1,
    });
    expect(first.statusCode).toBe(200);

    // 같은 baseVersion 1로 다시 저장 — 다른 편집자가 이미 v2를 만든 상황
    const stale = await put('/processes/proc-ver-001', {
      doc: { ...doc, name: '개정판 B' },
      baseVersion: 1,
    });
    expect(stale.statusCode).toBe(409);
    const body = stale.json() as { error: string; current: Envelope };
    expect(body.error).toBe('version-conflict');
    expect(body.current.meta.version).toBe(2);
    expect(body.current.doc['name']).toBe('개정판 A'); // 서버 현재본 = 먼저 저장된 쪽
  });

  it('신규 생성에 baseVersion 숫자를 넣으면 400', async () => {
    const res = await post('/processes', { doc: makeProcessDoc('proc-ver-002'), baseVersion: 3 });
    expect(res.statusCode).toBe(400);
  });
});

// ── 휴지통 (soft-delete → restore → 지연 퍼지) ──────────────────────

describe('휴지통', () => {
  it('삭제 → 목록 제외(includeDeleted로는 보임) → 복원 → 30일 경과 시 지연 완전삭제', async () => {
    const doc = makeProcessDoc('proc-trash-01');
    await post('/processes', { doc, baseVersion: null });

    const deleted = await del('/processes/proc-trash-01');
    expect(deleted.statusCode).toBe(200);
    const expectedDeadline = new Date(server.clock.now().getTime() + 30 * 86_400_000).toISOString();
    expect((deleted.json() as { restoreUntilIso: string }).restoreUntilIso).toBe(expectedDeadline);

    // 기본 목록에서는 제외, includeDeleted=1이면 deletedAtIso가 채워져 보인다
    const normal = await get('/processes');
    expect((normal.json() as { items: EntityMeta[] }).items).toHaveLength(0);
    const withTrash = await get('/processes?includeDeleted=1');
    const trashed = (withTrash.json() as { items: EntityMeta[] }).items;
    expect(trashed).toHaveLength(1);
    expect(trashed[0]?.meta.deletedAtIso).not.toBeNull();
    expect(trashed[0]?.meta.deletedByName).toBe('관리자');

    // 단건 조회는 휴지통 행도 돌려준다(표시는 meta가 한다)
    const single = await get('/processes/proc-trash-01');
    expect(single.statusCode).toBe(200);
    expect((single.json() as Envelope).meta.deletedAtIso).not.toBeNull();

    // 복원 — 다시 목록에 나타난다
    const restored = await post('/processes/proc-trash-01/restore', {});
    expect(restored.statusCode).toBe(200);
    expect((restored.json() as Envelope).meta.deletedAtIso).toBeNull();
    expect((await get('/processes')).json()).toMatchObject({ items: [{ id: 'proc-trash-01' }] });

    // 다시 삭제하고 31일 경과 — 목록 조회가 퍼지를 유발한다
    await del('/processes/proc-trash-01');
    server.clock.advanceDays(31);
    // 기존 세션은 30일 TTL로 만료됐다 — 관리자로 재로그인
    const relogin = await server.app.inject({
      method: 'POST',
      url: '/api/v1/auth/login',
      payload: { userId: server.admin.id, pin: ADMIN_PIN },
    });
    expect(relogin.statusCode).toBe(200);
    const freshToken = (relogin.json() as { token: string }).token;

    const purgedList = await get('/processes?includeDeleted=1', freshToken);
    expect((purgedList.json() as { items: EntityMeta[] }).items).toHaveLength(0);
    expect((await get('/processes/proc-trash-01', freshToken)).statusCode).toBe(404);
    expect((await post('/processes/proc-trash-01/restore', {}, freshToken)).statusCode).toBe(404);
  });
});

// ── 목록 필터 · 정렬 · taskSummary ──────────────────────────────────

describe('목록', () => {
  it('updatedAt 내림차순 — 최근에 저장된 문서가 위로 온다', async () => {
    await post('/blocks', { doc: makeBlockDoc('block-ord-aa', '먼저'), baseVersion: null });
    server.clock.advanceSec(10);
    await post('/blocks', { doc: makeBlockDoc('block-ord-bb', '나중'), baseVersion: null });

    let items = ((await get('/blocks')).json() as { items: EntityMeta[] }).items;
    expect(items.map((i) => i.id)).toEqual(['block-ord-bb', 'block-ord-aa']);

    server.clock.advanceSec(10);
    await put('/blocks/block-ord-aa', {
      doc: makeBlockDoc('block-ord-aa', '먼저-개정'),
      baseVersion: 1,
    });
    items = ((await get('/blocks')).json() as { items: EntityMeta[] }).items;
    expect(items.map((i) => i.id)).toEqual(['block-ord-aa', 'block-ord-bb']);
  });

  it('tasks — q·processId 필터와 taskSummary(stepCount·hasThumbnail)', async () => {
    await post('/processes', { doc: makeProcessDoc('proc-line-01'), baseVersion: null });
    await post('/tasks', {
      doc: makeTaskDoc('task-aaa-001', {
        name: '팔레타이징 A',
        processId: 'proc-line-01',
        stepCount: 3,
        thumbnail: 'data:image/png;base64,AAAA',
      }),
      baseVersion: null,
    });
    await post('/tasks', { doc: makeTaskDoc('task-bbb-002', { name: '검사 B' }), baseVersion: null });

    const all = ((await get('/tasks')).json() as { items: EntityMeta[] }).items;
    expect(all).toHaveLength(2);
    const t1 = all.find((i) => i.id === 'task-aaa-001');
    expect(t1?.processId).toBe('proc-line-01');
    expect(t1?.taskSummary).toEqual({ stepCount: 3, hasThumbnail: true, lastRun: null });
    const t2 = all.find((i) => i.id === 'task-bbb-002');
    expect(t2?.processId).toBeNull();
    expect(t2?.taskSummary).toEqual({ stepCount: 0, hasThumbnail: false, lastRun: null });

    const filtered = ((await get('/tasks?processId=proc-line-01')).json() as { items: EntityMeta[] }).items;
    expect(filtered.map((i) => i.id)).toEqual(['task-aaa-001']);

    const searched = ((await get(`/tasks?q=${encodeURIComponent('팔레')}`)).json() as { items: EntityMeta[] }).items;
    expect(searched.map((i) => i.id)).toEqual(['task-aaa-001']);
  });
});

// ── 조언적 잠금 ─────────────────────────────────────────────────────

describe('잠금 (advisory lock)', () => {
  it('타 사용자 보유 시 423, heartbeat 연장, TTL 만료 시 자연 해제', async () => {
    const tech = await createUserAndLogin(server, '설치기사', '4321', 'tech');
    await post('/tasks', { doc: makeTaskDoc('task-lock-01'), baseVersion: null });

    // admin이 획득
    const acquired = await post('/locks/task/task-lock-01', { action: 'acquire' });
    expect(acquired.statusCode).toBe(200);
    const lock = (acquired.json() as { lock: LockInfo }).lock;
    expect(lock.userId).toBe(server.admin.id);
    expect(lock.userName).toBe('관리자');
    expect(Date.parse(lock.expiresAtIso) - Date.parse(lock.acquiredAtIso)).toBe(90_000);

    // 타 사용자 — acquire도 release도 423 (강탈 없음)
    const stolen = await post('/locks/task/task-lock-01', { action: 'acquire' }, tech.token);
    expect(stolen.statusCode).toBe(423);
    expect((stolen.json() as { lock: LockInfo }).lock.userId).toBe(server.admin.id);
    const releaseByOther = await post('/locks/task/task-lock-01', { action: 'release' }, tech.token);
    expect(releaseByOther.statusCode).toBe(423);

    // heartbeat — 만료 시각이 지금 기준으로 다시 연장된다 (acquired_at은 유지)
    server.clock.advanceSec(60);
    const beat = await post('/locks/task/task-lock-01', { action: 'heartbeat' });
    expect(beat.statusCode).toBe(200);
    const beatLock = (beat.json() as { lock: LockInfo }).lock;
    expect(beatLock.acquiredAtIso).toBe(lock.acquiredAtIso);
    expect(Date.parse(beatLock.expiresAtIso)).toBe(server.clock.now().getTime() + 90_000);

    // TTL 경과 — 자연 해제되고 다른 사용자가 획득할 수 있다
    server.clock.advanceSec(91);
    const after = await get('/locks/task/task-lock-01');
    expect((after.json() as { lock: LockInfo | null }).lock).toBeNull();
    const reacquired = await post('/locks/task/task-lock-01', { action: 'acquire' }, tech.token);
    expect(reacquired.statusCode).toBe(200);
    expect((reacquired.json() as { lock: LockInfo }).lock.userId).toBe(tech.user.id);

    // 소유자 해제 — { lock: null }
    const released = await post('/locks/task/task-lock-01', { action: 'release' }, tech.token);
    expect(released.statusCode).toBe(200);
    expect((released.json() as { lock: LockInfo | null }).lock).toBeNull();
  });

  it('잠금 kind는 task·process·block만 — 그 외 400', async () => {
    const res = await post('/locks/device/dev-lock-01', { action: 'acquire' });
    expect(res.statusCode).toBe(400);
  });
});
