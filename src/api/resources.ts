// api/resources.ts — 개체별 타입드 클라이언트 (processes/tasks/blocks/devices +
// locks/runs/stats). 규범: docs/BACKEND.md §4, 타입 진실: src/schema/entities.ts.
//
// ── 설계 ────────────────────────────────────────────────────────────
// - 결과는 전부 discriminated union — 409는 'conflict'로 판별 가능하게 분리한다
//   (BACKEND §1.5: 저장은 baseVersion 검사, 불일치 409 + 서버 현재본 반환).
// - 응답 본문은 타입 단언만 한다: 서버는 같은 저장소의 코드이고 entities.ts라는
//   단일 zod 진실을 공유한다. scene/sequence의 도메인 검증은 클라이언트 로드
//   시점의 몫이다 (불변식 §2.9 — 이 계층의 책임이 아니다).
// - 캐시/outbox는 offline.ts가 소유한다. 이 파일은 순수 HTTP 표면만 담당한다.

import type { ApiClient, ApiResult } from './client';
import type {
  BlockDoc,
  DeviceDoc,
  EntityMeta,
  LockInfo,
  ProcessDoc,
  RecordEnvelope,
  RunRecord,
  SaveRequest,
  TaskDoc,
  TaskStats,
} from '../schema/entities';

// ── 개체 종류 ↔ 문서 타입 매핑 ──────────────────────────────────────

export interface EntityDocMap {
  processes: ProcessDoc;
  tasks: TaskDoc;
  blocks: BlockDoc;
  devices: DeviceDoc;
}

export type EntityKind = keyof EntityDocMap;

export const ENTITY_KINDS: readonly EntityKind[] = ['processes', 'tasks', 'blocks', 'devices'];

/** 잠금 대상 종류 — lockInfoSchema(entities.ts)와 일치 */
export type LockKind = 'task' | 'process' | 'block';
export type LockAction = 'acquire' | 'heartbeat' | 'release';

// ── 결과 union ──────────────────────────────────────────────────────

/** 성공 외 공통 실패 3종 — UI는 이 셋만 알면 모든 리소스 호출을 분기할 수 있다 */
export type ApiFailure =
  | { readonly kind: 'network'; readonly messageKo: string }
  | { readonly kind: 'unauthorized'; readonly messageKo: string }
  | { readonly kind: 'error'; readonly status: number; readonly messageKo: string };

export type ListResult = { readonly kind: 'ok'; readonly items: EntityMeta[] } | ApiFailure;

export type GetResult<T> =
  | { readonly kind: 'ok'; readonly record: RecordEnvelope<T> }
  | ApiFailure;

export type SaveResult<T> =
  | { readonly kind: 'ok'; readonly record: RecordEnvelope<T> }
  | {
      readonly kind: 'conflict';
      /** 서버 현재본 (ConflictResponse.current) — 봉투가 없으면 null */
      readonly current: RecordEnvelope<T> | null;
      readonly messageKo: string;
    }
  | ApiFailure;

export type RemoveResult =
  | { readonly kind: 'ok'; readonly restoreUntilIso: string }
  | ApiFailure;

export type LockResult =
  | { readonly kind: 'ok'; readonly lock: LockInfo | null }
  | {
      /** 423 — 타인이 보유 중 (강탈 없음, BACKEND §1.5) */
      readonly kind: 'held';
      readonly lock: LockInfo | null;
      readonly messageKo: string;
    }
  | ApiFailure;

export type RunCreateResult = { readonly kind: 'ok'; readonly id: string } | ApiFailure;
export type RunListResult =
  | { readonly kind: 'ok'; readonly items: RunRecord[]; readonly total: number }
  | ApiFailure;
export type RunGetResult = { readonly kind: 'ok'; readonly run: RunRecord } | ApiFailure;
export type TaskStatsResult = { readonly kind: 'ok'; readonly stats: TaskStats } | ApiFailure;

/** ok/conflict가 아닌 ApiResult를 공통 실패로 접는다 */
function asFailure(r: ApiResult<unknown>): ApiFailure {
  switch (r.kind) {
    case 'network':
      return { kind: 'network', messageKo: r.messageKo };
    case 'unauthorized':
      return { kind: 'unauthorized', messageKo: r.messageKo };
    case 'conflict':
      // 저장 이외 경로의 409 (예: 중복 잠금 이외 서버 사정) — 일반 오류로 접는다
      return { kind: 'error', status: 409, messageKo: r.messageKo };
    case 'error':
      return { kind: 'error', status: r.status, messageKo: r.messageKo };
    case 'ok':
      throw new Error('asFailure: ok 결과는 실패가 아니다');
  }
}

// ── 순수 헬퍼 (node 테스트 대상) ────────────────────────────────────

export interface ListOptions {
  readonly q?: string;
  readonly includeDeleted?: boolean;
  /** tasks 전용 필터 — 다른 종류에서는 서버가 무시한다 */
  readonly processId?: string;
}

/** 목록 쿼리 문자열 — 빈 옵션이면 ''(쿼리 없음) */
export function buildListQuery(opts: ListOptions = {}): string {
  const params = new URLSearchParams();
  const q = opts.q?.trim() ?? '';
  if (q !== '') params.set('q', q);
  if (opts.includeDeleted === true) params.set('includeDeleted', '1');
  if (opts.processId !== undefined && opts.processId !== '') {
    params.set('processId', opts.processId);
  }
  const s = params.toString();
  return s === '' ? '' : `?${s}`;
}

export interface RunListOptions {
  readonly taskId?: string;
  readonly limit?: number;
  readonly offset?: number;
}

export function buildRunsQuery(opts: RunListOptions = {}): string {
  const params = new URLSearchParams();
  if (opts.taskId !== undefined && opts.taskId !== '') params.set('taskId', opts.taskId);
  if (opts.limit !== undefined) params.set('limit', String(opts.limit));
  if (opts.offset !== undefined) params.set('offset', String(opts.offset));
  const s = params.toString();
  return s === '' ? '' : `?${s}`;
}

/** 409 본문(ConflictResponse)에서 서버 현재본을 방어적으로 추출한다 */
export function extractConflictCurrent<T>(body: unknown): RecordEnvelope<T> | null {
  if (body === null || typeof body !== 'object' || !('current' in body)) return null;
  const current = (body as { current: unknown }).current;
  if (current === null || typeof current !== 'object') return null;
  if (!('doc' in current) || !('meta' in current)) return null;
  return current as RecordEnvelope<T>;
}

/** 423 본문 `{ lock }`에서 잠금 정보를 방어적으로 추출한다 */
export function extractLock(body: unknown): LockInfo | null {
  if (body === null || typeof body !== 'object' || !('lock' in body)) return null;
  const lock = (body as { lock: unknown }).lock;
  if (lock === null || typeof lock !== 'object') return null;
  if (!('userId' in lock) || !('expiresAtIso' in lock)) return null;
  return lock as LockInfo;
}

// ── 개체 클라이언트 ─────────────────────────────────────────────────

export class EntityClient<K extends EntityKind> {
  constructor(
    private readonly api: ApiClient,
    readonly kind: K,
  ) {}

  /** 목록 — payload 없는 EntityMeta 행 (BACKEND §4) */
  async list(opts: ListOptions = {}): Promise<ListResult> {
    const r = await this.api.request<{ items: EntityMeta[] }>(
      'GET',
      `/${this.kind}${buildListQuery(opts)}`,
    );
    if (r.kind === 'ok') return { kind: 'ok', items: r.data.items };
    return asFailure(r);
  }

  /** 단건 — 휴지통 행도 반환된다 (meta.deletedAtIso로 표시) */
  async get(id: string): Promise<GetResult<EntityDocMap[K]>> {
    const r = await this.api.request<RecordEnvelope<EntityDocMap[K]>>(
      'GET',
      `/${this.kind}/${encodeURIComponent(id)}`,
    );
    if (r.kind === 'ok') return { kind: 'ok', record: r.data };
    return asFailure(r);
  }

  /** 생성 — id는 클라이언트 발급 uuid (오프라인 생성 지원). 중복 id는 conflict. */
  async create(doc: EntityDocMap[K]): Promise<SaveResult<EntityDocMap[K]>> {
    const body: SaveRequest<EntityDocMap[K]> = { doc, baseVersion: null };
    return this.toSaveResult(await this.api.request('POST', `/${this.kind}`, { body }));
  }

  /** 저장 — baseVersion 불일치면 conflict + 서버 현재본 (자동 병합 금지, BACKEND §6) */
  async update(
    id: string,
    doc: EntityDocMap[K],
    baseVersion: number,
  ): Promise<SaveResult<EntityDocMap[K]>> {
    const body: SaveRequest<EntityDocMap[K]> = { doc, baseVersion };
    return this.toSaveResult(
      await this.api.request('PUT', `/${this.kind}/${encodeURIComponent(id)}`, { body }),
    );
  }

  /** soft-delete — 휴지통 30일. 완전 삭제 API는 없다 (BACKEND §1.4). */
  async remove(id: string): Promise<RemoveResult> {
    const r = await this.api.request<{ restoreUntilIso: string }>(
      'DELETE',
      `/${this.kind}/${encodeURIComponent(id)}`,
    );
    if (r.kind === 'ok') return { kind: 'ok', restoreUntilIso: r.data.restoreUntilIso };
    return asFailure(r);
  }

  async restore(id: string): Promise<GetResult<EntityDocMap[K]>> {
    const r = await this.api.request<RecordEnvelope<EntityDocMap[K]>>(
      'POST',
      `/${this.kind}/${encodeURIComponent(id)}/restore`,
    );
    if (r.kind === 'ok') return { kind: 'ok', record: r.data };
    return asFailure(r);
  }

  private toSaveResult(
    r: ApiResult<RecordEnvelope<EntityDocMap[K]>>,
  ): SaveResult<EntityDocMap[K]> {
    if (r.kind === 'ok') return { kind: 'ok', record: r.data };
    if (r.kind === 'conflict') {
      return {
        kind: 'conflict',
        current: extractConflictCurrent<EntityDocMap[K]>(r.body),
        messageKo: r.messageKo,
      };
    }
    return asFailure(r);
  }
}

// ── 실행 기록 (append-only — 삭제 API 없음) ─────────────────────────

export class RunsClient {
  constructor(private readonly api: ApiClient) {}

  async create(record: RunRecord): Promise<RunCreateResult> {
    const r = await this.api.request<{ id: string }>('POST', '/runs', { body: record });
    if (r.kind === 'ok') return { kind: 'ok', id: r.data.id };
    return asFailure(r);
  }

  async list(opts: RunListOptions = {}): Promise<RunListResult> {
    const r = await this.api.request<{ items: RunRecord[]; total: number }>(
      'GET',
      `/runs${buildRunsQuery(opts)}`,
    );
    if (r.kind === 'ok') return { kind: 'ok', items: r.data.items, total: r.data.total };
    return asFailure(r);
  }

  async get(id: string): Promise<RunGetResult> {
    const r = await this.api.request<RunRecord>('GET', `/runs/${encodeURIComponent(id)}`);
    if (r.kind === 'ok') return { kind: 'ok', run: r.data };
    return asFailure(r);
  }
}

// ── 표면 파사드 ─────────────────────────────────────────────────────

export class WorkcellApi {
  readonly processes: EntityClient<'processes'>;
  readonly tasks: EntityClient<'tasks'>;
  readonly blocks: EntityClient<'blocks'>;
  readonly devices: EntityClient<'devices'>;
  readonly runs: RunsClient;
  private readonly byKind: { readonly [K in EntityKind]: EntityClient<K> };

  constructor(private readonly api: ApiClient) {
    this.processes = new EntityClient(api, 'processes');
    this.tasks = new EntityClient(api, 'tasks');
    this.blocks = new EntityClient(api, 'blocks');
    this.devices = new EntityClient(api, 'devices');
    this.runs = new RunsClient(api);
    this.byKind = {
      processes: this.processes,
      tasks: this.tasks,
      blocks: this.blocks,
      devices: this.devices,
    };
  }

  /** 런타임 kind → 타입드 클라이언트 (outbox 재전송이 쓴다) */
  entity<K extends EntityKind>(kind: K): EntityClient<K> {
    return this.byKind[kind];
  }

  /** 편집 잠금 — acquire/heartbeat/release. 타인 보유면 'held'(423). */
  async locks(kind: LockKind, id: string, action: LockAction): Promise<LockResult> {
    const r = await this.api.request<{ lock: LockInfo | null }>(
      'POST',
      `/locks/${kind}/${encodeURIComponent(id)}`,
      { body: { action } },
    );
    if (r.kind === 'ok') return { kind: 'ok', lock: r.data.lock };
    if (r.kind === 'error' && r.status === 423) {
      return { kind: 'held', lock: extractLock(r.body), messageKo: r.messageKo };
    }
    return asFailure(r);
  }

  async getLock(kind: LockKind, id: string): Promise<LockResult> {
    const r = await this.api.request<{ lock: LockInfo | null }>(
      'GET',
      `/locks/${kind}/${encodeURIComponent(id)}`,
    );
    if (r.kind === 'ok') return { kind: 'ok', lock: r.data.lock };
    return asFailure(r);
  }

  async taskStats(taskId: string): Promise<TaskStatsResult> {
    const r = await this.api.request<TaskStats>(
      'GET',
      `/tasks/${encodeURIComponent(taskId)}/stats`,
    );
    if (r.kind === 'ok') return { kind: 'ok', stats: r.data };
    return asFailure(r);
  }
}

export function createWorkcellApi(client: ApiClient): WorkcellApi {
  return new WorkcellApi(client);
}
