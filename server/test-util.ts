// server/test-util.ts — 서버 테스트 공용 하네스 (':memory:' DB + 주입 시계 + 문서 팩토리)
//
// 실제 HTTP 소켓을 열지 않는다 — buildApp은 listen하지 않고 테스트는 app.inject로
// 왕복한다(docs/BACKEND.md §7). 시각은 makeClock으로 주입해 TTL(세션·잠금)·휴지통
// 퍼지를 시계 조작(fake timer) 없이 결정론적으로 검증한다.

import type { FastifyInstance } from 'fastify';
import type {
  BlockDoc,
  DeviceDoc,
  ProcessDoc,
  RunCollision,
  RunRecord,
  TaskDoc,
  UserInfo,
  UserRole,
} from '../src/schema/entities';
import { buildApp } from './app';

// ── 주입 시계 ───────────────────────────────────────────────────────

export interface TestClock {
  readonly now: () => Date;
  set(iso: string): void;
  advanceSec(sec: number): void;
  advanceDays(days: number): void;
}

export const CLOCK_START_ISO = '2026-08-07T00:00:00.000Z';

export function makeClock(startIso: string = CLOCK_START_ISO): TestClock {
  let currentMs = Date.parse(startIso);
  return {
    now: (): Date => new Date(currentMs),
    set: (iso: string): void => {
      currentMs = Date.parse(iso);
    },
    advanceSec: (sec: number): void => {
      currentMs += sec * 1000;
    },
    advanceDays: (days: number): void => {
      currentMs += days * 86_400_000;
    },
  };
}

// ── 앱 하네스 ───────────────────────────────────────────────────────

export const ADMIN_PIN = '123456';

export interface TestServer {
  readonly app: FastifyInstance;
  readonly clock: TestClock;
  readonly adminToken: string;
  readonly admin: UserInfo;
}

/** ':memory:' DB로 앱을 만들고 셋업(관리자 생성 + 로그인)까지 마친 하네스를 돌려준다 */
export async function createTestServer(): Promise<TestServer> {
  const clock = makeClock();
  const app = buildApp({ dbPath: ':memory:', now: clock.now, serverName: '테스트 서버' });
  await app.ready();
  const res = await app.inject({
    method: 'POST',
    url: '/api/v1/auth/setup',
    payload: { name: '관리자', pin: ADMIN_PIN },
  });
  if (res.statusCode !== 200) throw new Error(`셋업 실패: ${res.statusCode} ${res.body}`);
  const body = res.json() as { token: string; user: UserInfo };
  return { app, clock, adminToken: body.token, admin: body.user };
}

export function bearer(token: string): { authorization: string } {
  return { authorization: `Bearer ${token}` };
}

/** admin 권한으로 사용자를 만들고 그 사용자로 로그인까지 마친다 */
export async function createUserAndLogin(
  server: TestServer,
  name: string,
  pin: string,
  role: UserRole,
): Promise<{ user: UserInfo; token: string }> {
  const created = await server.app.inject({
    method: 'POST',
    url: '/api/v1/users',
    headers: bearer(server.adminToken),
    payload: { name, pin, role },
  });
  if (created.statusCode !== 201) throw new Error(`사용자 생성 실패: ${created.statusCode} ${created.body}`);
  const { user } = created.json() as { user: UserInfo };
  const login = await server.app.inject({
    method: 'POST',
    url: '/api/v1/auth/login',
    payload: { userId: user.id, pin },
  });
  if (login.statusCode !== 200) throw new Error(`로그인 실패: ${login.statusCode} ${login.body}`);
  return { user, token: (login.json() as { token: string }).token };
}

// ── 문서 팩토리 (id는 entityIdSchema min 8자를 충족해야 한다) ───────

export function makeProcessDoc(id: string, name = '용접 공정'): ProcessDoc {
  return {
    id,
    name,
    descriptionKo: '',
    scene: { version: 1, entities: [] },
    deviceIds: [],
    rules: { autoPauseOnCollision: true, speedLimitMult: null },
  };
}

export function makeTaskDoc(
  id: string,
  opts: {
    name?: string;
    processId?: string | null;
    stepCount?: number;
    thumbnail?: string | null;
  } = {},
): TaskDoc {
  const stepCount = opts.stepCount ?? 0;
  return {
    id,
    name: opts.name ?? '픽앤플레이스 작업',
    processId: opts.processId ?? null,
    sceneOrigin: null,
    scene: { version: 1, entities: [] },
    sequence:
      stepCount === 0
        ? null
        : { steps: Array.from({ length: stepCount }, () => ({ kind: 'setJoints', targets: {} })) },
    assets: {},
    thumbnail: opts.thumbnail ?? null,
    notes: '',
  };
}

export function makeBlockDoc(id: string, name = '집기 블록'): BlockDoc {
  return {
    id,
    name,
    descriptionKo: '',
    steps: [{ kind: 'setJoints', targets: { joint1: 0.5 } }],
    params: [],
    robotHint: null,
  };
}

export function makeDeviceDoc(id: string, name = '6축 로봇'): DeviceDoc {
  return {
    id,
    name,
    kind: 'robot',
    templateKey: 'arm-6',
    connection: { mode: 'virtual', endpoint: null },
    notes: '',
  };
}

export function makeCollision(
  nodeId: string | null,
  classification: 'intended' | 'unexpected',
): RunCollision {
  return {
    atSimSec: 1.5,
    entityA: 'robot-arm',
    entityB: 'box-entity',
    phase: 'start',
    nodeId,
    classification,
  };
}

export function makeRunRecord(
  id: string,
  taskId: string,
  opts: {
    startedAtIso?: string;
    result?: RunRecord['result'];
    wallTimeSec?: number;
    collisions?: RunCollision[];
  } = {},
): RunRecord {
  const startedAtIso = opts.startedAtIso ?? CLOCK_START_ISO;
  const wallTimeSec = opts.wallTimeSec ?? 10;
  return {
    id,
    taskId,
    taskName: '테스트 작업',
    taskVersion: 1,
    processId: null,
    // 일부러 스푸핑된 값 — 서버가 세션 사용자로 덮어써야 한다(routes-runs 계약)
    operatorId: 'spoofed-operator-id',
    operatorName: '스푸핑된 작업자',
    startedAtIso,
    endedAtIso: new Date(Date.parse(startedAtIso) + wallTimeSec * 1000).toISOString(),
    result: opts.result ?? 'completed',
    stepsTotal: 5,
    stepsDone: 5,
    simTimeSec: 8,
    wallTimeSec,
    collisions: opts.collisions ?? [],
    interventions: [],
  };
}
