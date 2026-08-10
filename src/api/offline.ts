// api/offline.ts — 오프라인 읽기 캐시 + 쓰기 outbox. 규범: docs/BACKEND.md §6.
//
// ── 정책 ────────────────────────────────────────────────────────────
// 읽기: 성공 응답을 IndexedDB 'workcell-api-cache'에 기록하고, 오프라인이면
//   캐시를 서빙한다 ("오프라인 — 마지막 동기화 n분 전" 배너는 UI 몫,
//   n분 계산은 lastSyncAgeKo가 제공).
// 쓰기: 실패한 쓰기를 outbox에 적재하고 재연결 시 **순서대로** 재전송한다.
//   409 충돌은 자동 병합하지 않는다 — conflict 레코드로 보존하고 사용자가
//   "서버본 열기 / 내 것으로 덮어쓰기 / 사본으로 저장"을 고른다(§2.11 정신).
//   서버가 거부한(4xx/5xx) 항목도 버리지 않는다 — 작업 손실 금지가 우선이므로
//   flush를 멈추고 큐에 남긴다(비우는 결정은 사람이 한다).
//
// ── 스토리지는 인터페이스 뒤에 ──────────────────────────────────────
// KVStorage 인터페이스 + IndexedDB 구현 + 인메모리 구현(테스트용).
// IndexedDB factory도 주입 가능 — node 테스트는 MemoryKVStorage만 쓴다.

import type { EntityMeta, RecordEnvelope, SaveRequest } from '../schema/entities';
import type {
  EntityClient,
  EntityDocMap,
  EntityKind,
  ListOptions,
  ApiFailure,
  GetResult,
  RemoveResult,
  SaveResult,
  WorkcellApi,
} from './resources';

// ── 상수 ────────────────────────────────────────────────────────────

/** IndexedDB 데이터베이스 이름 (BACKEND §6) */
export const API_CACHE_DB_NAME = 'workcell-api-cache';
export const API_CACHE_DB_VERSION = 1;

export const MSG_OUTBOX_CORRUPT_KO = '보관된 항목이 손상되어 보낼 수 없습니다';

// ── KVStorage 인터페이스 ────────────────────────────────────────────

export type StoreName = 'lists' | 'records' | 'outbox' | 'conflicts' | 'meta';

export const KV_STORES: readonly StoreName[] = [
  'lists',
  'records',
  'outbox',
  'conflicts',
  'meta',
];

export interface KVStorage {
  get(store: StoreName, key: string): Promise<unknown>;
  put(store: StoreName, key: string, value: unknown): Promise<void>;
  remove(store: StoreName, key: string): Promise<void>;
  /** 키 오름차순 정렬 반환 — outbox 순서 보존이 이 계약에 의존한다 */
  keys(store: StoreName): Promise<string[]>;
}

/** 테스트용 인메모리 구현 — 브라우저 없이 vitest node 환경에서 돈다 */
export class MemoryKVStorage implements KVStorage {
  private readonly stores = new Map<StoreName, Map<string, unknown>>();

  private store(name: StoreName): Map<string, unknown> {
    let m = this.stores.get(name);
    if (m === undefined) {
      m = new Map<string, unknown>();
      this.stores.set(name, m);
    }
    return m;
  }

  get(store: StoreName, key: string): Promise<unknown> {
    return Promise.resolve(this.store(store).get(key));
  }

  put(store: StoreName, key: string, value: unknown): Promise<void> {
    this.store(store).set(key, value);
    return Promise.resolve();
  }

  remove(store: StoreName, key: string): Promise<void> {
    this.store(store).delete(key);
    return Promise.resolve();
  }

  keys(store: StoreName): Promise<string[]> {
    return Promise.resolve([...this.store(store).keys()].sort());
  }
}

/** IndexedDB 구현 — factory 주입 가능(기본 globalThis.indexedDB), 지연 오픈 */
export class IndexedDbKVStorage implements KVStorage {
  private dbPromise: Promise<IDBDatabase> | null = null;

  constructor(
    private readonly factory: IDBFactory | null = (globalThis as { indexedDB?: IDBFactory })
      .indexedDB ?? null,
    private readonly dbName: string = API_CACHE_DB_NAME,
  ) {}

  private db(): Promise<IDBDatabase> {
    if (this.factory === null) {
      return Promise.reject(new Error('IndexedDB를 사용할 수 없는 환경입니다'));
    }
    const factory = this.factory;
    this.dbPromise ??= new Promise<IDBDatabase>((resolve, reject) => {
      const req = factory.open(this.dbName, API_CACHE_DB_VERSION);
      req.onupgradeneeded = () => {
        const db = req.result;
        for (const store of KV_STORES) {
          if (!db.objectStoreNames.contains(store)) db.createObjectStore(store);
        }
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error ?? new Error('IndexedDB open 실패'));
    });
    return this.dbPromise;
  }

  private async run<T>(
    store: StoreName,
    mode: IDBTransactionMode,
    op: (s: IDBObjectStore) => IDBRequest<T>,
  ): Promise<T> {
    const db = await this.db();
    return new Promise<T>((resolve, reject) => {
      const tx = db.transaction(store, mode);
      const req = op(tx.objectStore(store));
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error ?? new Error('IndexedDB 요청 실패'));
    });
  }

  async get(store: StoreName, key: string): Promise<unknown> {
    return this.run<unknown>(store, 'readonly', (s) => s.get(key));
  }

  async put(store: StoreName, key: string, value: unknown): Promise<void> {
    await this.run<IDBValidKey>(store, 'readwrite', (s) => s.put(value, key));
  }

  async remove(store: StoreName, key: string): Promise<void> {
    await this.run<undefined>(store, 'readwrite', (s) => s.delete(key));
  }

  async keys(store: StoreName): Promise<string[]> {
    const raw = await this.run<IDBValidKey[]>(store, 'readonly', (s) => s.getAllKeys());
    return raw.map((k) => String(k)).sort();
  }
}

// ── 읽기 캐시 ───────────────────────────────────────────────────────

export interface CachedList {
  readonly items: EntityMeta[];
  readonly fetchedAtIso: string;
}

export interface CachedRecord {
  readonly record: RecordEnvelope<unknown>;
  readonly fetchedAtIso: string;
}

/** 목록 캐시 키 — 같은 질의는 같은 키로 수렴해야 한다 (순수, 테스트 대상) */
export function listCacheKey(opts: ListOptions = {}): string {
  const q = opts.q?.trim() ?? '';
  const del = opts.includeDeleted === true ? '1' : '0';
  const proc = opts.processId ?? '';
  return `q=${q}|del=${del}|proc=${proc}`;
}

export class ApiCache {
  constructor(
    private readonly storage: KVStorage,
    private readonly nowIso: () => string = () => new Date().toISOString(),
  ) {}

  async saveList(kind: EntityKind, queryKey: string, items: EntityMeta[]): Promise<CachedList> {
    const value: CachedList = { items, fetchedAtIso: this.nowIso() };
    await this.storage.put('lists', `${kind}/${queryKey}`, value);
    return value;
  }

  async readList(kind: EntityKind, queryKey: string): Promise<CachedList | null> {
    const raw = await this.storage.get('lists', `${kind}/${queryKey}`);
    if (raw === null || raw === undefined || typeof raw !== 'object') return null;
    const value = raw as CachedList;
    return Array.isArray(value.items) && typeof value.fetchedAtIso === 'string' ? value : null;
  }

  async saveRecord(
    kind: EntityKind,
    id: string,
    record: RecordEnvelope<unknown>,
  ): Promise<CachedRecord> {
    const value: CachedRecord = { record, fetchedAtIso: this.nowIso() };
    await this.storage.put('records', `${kind}/${id}`, value);
    return value;
  }

  async readRecord(kind: EntityKind, id: string): Promise<CachedRecord | null> {
    const raw = await this.storage.get('records', `${kind}/${id}`);
    if (raw === null || raw === undefined || typeof raw !== 'object') return null;
    const value = raw as CachedRecord;
    return typeof value.fetchedAtIso === 'string' &&
      value.record !== null &&
      typeof value.record === 'object'
      ? value
      : null;
  }
}

/** "마지막 동기화 n분 전" 배너의 나이 표기 (순수) — 배너 조립은 UI 몫 */
export function lastSyncAgeKo(fetchedAtIso: string, nowMs: number): string {
  const fetchedMs = Date.parse(fetchedAtIso);
  if (!Number.isFinite(fetchedMs)) return '알 수 없음';
  const diffMin = Math.floor(Math.max(0, nowMs - fetchedMs) / 60_000);
  if (diffMin < 1) return '방금 전';
  if (diffMin < 60) return `${diffMin}분 전`;
  const hours = Math.floor(diffMin / 60);
  if (hours < 24) return `${hours}시간 전`;
  return `${Math.floor(hours / 24)}일 전`;
}

// ── 캐시 경유 읽기 (server 우선, 오프라인이면 캐시 서빙 — BACKEND §6) ──

export type ListThroughCacheResult =
  | {
      readonly kind: 'ok';
      readonly source: 'server' | 'cache';
      readonly items: EntityMeta[];
      readonly fetchedAtIso: string;
    }
  | ApiFailure;

export type GetThroughCacheResult<T> =
  | {
      readonly kind: 'ok';
      readonly source: 'server' | 'cache';
      readonly record: RecordEnvelope<T>;
      readonly fetchedAtIso: string;
    }
  | ApiFailure;

export async function listThroughCache<K extends EntityKind>(
  entity: EntityClient<K>,
  cache: ApiCache,
  opts: ListOptions = {},
): Promise<ListThroughCacheResult> {
  const key = listCacheKey(opts);
  const r = await entity.list(opts);
  if (r.kind === 'ok') {
    const saved = await cache.saveList(entity.kind, key, r.items);
    return { kind: 'ok', source: 'server', items: r.items, fetchedAtIso: saved.fetchedAtIso };
  }
  // 캐시 서빙은 연결 문제일 때만 — 인증 만료/서버 거부를 캐시로 가리지 않는다
  if (r.kind === 'network') {
    const hit = await cache.readList(entity.kind, key);
    if (hit !== null) {
      return { kind: 'ok', source: 'cache', items: hit.items, fetchedAtIso: hit.fetchedAtIso };
    }
  }
  return r;
}

export async function getThroughCache<K extends EntityKind>(
  entity: EntityClient<K>,
  cache: ApiCache,
  id: string,
): Promise<GetThroughCacheResult<EntityDocMap[K]>> {
  const r = await entity.get(id);
  if (r.kind === 'ok') {
    const saved = await cache.saveRecord(entity.kind, id, r.record);
    return { kind: 'ok', source: 'server', record: r.record, fetchedAtIso: saved.fetchedAtIso };
  }
  if (r.kind === 'network') {
    const hit = await cache.readRecord(entity.kind, id);
    if (hit !== null) {
      return {
        kind: 'ok',
        source: 'cache',
        // 캐시는 저장 시점의 형태를 보존한다 — 도메인 검증은 로드 시점 몫 (§2.9)
        record: hit.record as RecordEnvelope<EntityDocMap[K]>,
        fetchedAtIso: hit.fetchedAtIso,
      };
    }
  }
  return r;
}

// ── Outbox (쓰기 큐) ────────────────────────────────────────────────

export type OutboxOpKind = 'create' | 'update' | 'remove' | 'restore';

export interface OutboxOp {
  /** 단조 증가 — 재전송 순서의 진실 */
  readonly seq: number;
  readonly opKind: OutboxOpKind;
  readonly entityKind: EntityKind;
  readonly entityId: string;
  /** create/update의 SaveRequest — remove/restore는 null */
  readonly request: SaveRequest<unknown> | null;
  readonly enqueuedAtIso: string;
}

/** 409로 보존된 항목 — 자동 병합하지 않는다. 해소(UI 3택)는 사람이 한다. */
export interface ConflictRecord {
  readonly seq: number;
  readonly opKind: OutboxOpKind;
  readonly entityKind: EntityKind;
  readonly entityId: string;
  readonly request: SaveRequest<unknown> | null;
  readonly enqueuedAtIso: string;
  readonly detectedAtIso: string;
  /** 충돌 시점 서버 현재본 (없으면 null) */
  readonly current: RecordEnvelope<unknown> | null;
}

export type OutboxSendOutcome =
  | { readonly kind: 'ok' }
  | { readonly kind: 'conflict'; readonly current: RecordEnvelope<unknown> | null }
  | { readonly kind: 'network' }
  | { readonly kind: 'unauthorized' }
  | { readonly kind: 'error'; readonly messageKo: string };

export interface FlushReport {
  readonly sentCount: number;
  readonly conflictCount: number;
  readonly remainingCount: number;
  /** null = 큐 소진까지 진행. 그 외 = 해당 사유로 중단(항목은 큐에 남는다). */
  readonly stoppedBy: 'network' | 'unauthorized' | 'error' | 'busy' | null;
}

const OUTBOX_SEQ_KEY = 'outbox-next-seq';
const SEQ_PAD = 12;

/** outbox 키 — 0채움으로 사전순 == 숫자순을 보장한다 */
export function outboxKey(seq: number): string {
  return String(seq).padStart(SEQ_PAD, '0');
}

/** conflict 키 — 엔티티 id별 조회(prefix)와 개별 삭제가 모두 가능해야 한다 */
export function conflictKey(entityId: string, seq: number): string {
  return `${entityId}::${outboxKey(seq)}`;
}

function isOutboxOp(v: unknown): v is OutboxOp {
  if (v === null || typeof v !== 'object') return false;
  const op = v as { seq?: unknown; opKind?: unknown; entityKind?: unknown; entityId?: unknown };
  return (
    typeof op.seq === 'number' &&
    typeof op.opKind === 'string' &&
    typeof op.entityKind === 'string' &&
    typeof op.entityId === 'string'
  );
}

function isConflictRecord(v: unknown): v is ConflictRecord {
  return isOutboxOp(v) && 'detectedAtIso' in (v as object);
}

export class OfflineOutbox {
  private flushing = false;

  constructor(
    private readonly storage: KVStorage,
    private readonly nowIso: () => string = () => new Date().toISOString(),
  ) {}

  async enqueue(input: {
    opKind: OutboxOpKind;
    entityKind: EntityKind;
    entityId: string;
    request?: SaveRequest<unknown> | null;
  }): Promise<OutboxOp> {
    const rawNext = await this.storage.get('meta', OUTBOX_SEQ_KEY);
    const seq =
      typeof rawNext === 'number' && Number.isInteger(rawNext) && rawNext >= 1 ? rawNext : 1;
    await this.storage.put('meta', OUTBOX_SEQ_KEY, seq + 1);
    const op: OutboxOp = {
      seq,
      opKind: input.opKind,
      entityKind: input.entityKind,
      entityId: input.entityId,
      request: input.request ?? null,
      enqueuedAtIso: this.nowIso(),
    };
    await this.storage.put('outbox', outboxKey(seq), op);
    return op;
  }

  /**
   * 같은 개체의 대기 중 update op를 걷어낸다 — 새 update를 적재하기 **직전**에 부른다.
   *
   * 오프라인 동안 같은 문서를 여러 번 저장하면 모든 op가 같은 baseVersion을 갖는다.
   * 그대로 순서 재전송하면 첫 op가 서버 버전을 올리고 **자기 자신의 두 번째 op가 409**가
   * 된다 — 사용자의 최신본이 "다른 사용자가 먼저 저장했습니다"로 격리되는 가짜 충돌.
   * 로컬에서는 마지막 저장이 이기는 것이 사용자의 의도이므로 이전 update만 대체한다
   * (create/remove/restore는 의미가 다르므로 건드리지 않는다). 반환은 제거한 op 수.
   */
  async dropPendingUpdates(entityKind: EntityKind, entityId: string): Promise<number> {
    const keys = await this.storage.keys('outbox');
    let dropped = 0;
    for (const key of keys) {
      const raw = await this.storage.get('outbox', key);
      if (!isOutboxOp(raw)) continue;
      if (raw.opKind !== 'update') continue;
      if (raw.entityKind !== entityKind || raw.entityId !== entityId) continue;
      await this.storage.remove('outbox', key);
      dropped += 1;
    }
    return dropped;
  }

  /** 적재 순서(seq 오름차순)대로 반환 */
  async pending(): Promise<OutboxOp[]> {
    const keys = await this.storage.keys('outbox');
    const ops: OutboxOp[] = [];
    for (const key of keys) {
      const raw = await this.storage.get('outbox', key);
      if (isOutboxOp(raw)) ops.push(raw);
    }
    return ops.sort((a, b) => a.seq - b.seq);
  }

  /**
   * 재연결 시 순서 재전송. send는 통합자가 주입한다(보통 sendOutboxOp).
   * - ok       → 큐에서 제거, 계속
   * - conflict → conflict 레코드로 보존 + 큐에서 제거, 계속 (자동 병합 금지)
   * - network/unauthorized/error → 중단, 항목은 큐에 남는다 (작업 손실 금지)
   */
  async flush(send: (op: OutboxOp) => Promise<OutboxSendOutcome>): Promise<FlushReport> {
    if (this.flushing) {
      const remaining = await this.pending();
      return { sentCount: 0, conflictCount: 0, remainingCount: remaining.length, stoppedBy: 'busy' };
    }
    this.flushing = true;
    let sentCount = 0;
    let conflictCount = 0;
    let stoppedBy: FlushReport['stoppedBy'] = null;
    try {
      const ops = await this.pending();
      for (const op of ops) {
        const outcome = await send(op);
        if (outcome.kind === 'ok') {
          await this.storage.remove('outbox', outboxKey(op.seq));
          sentCount += 1;
          continue;
        }
        if (outcome.kind === 'conflict') {
          const record: ConflictRecord = {
            ...op,
            detectedAtIso: this.nowIso(),
            current: outcome.current,
          };
          await this.storage.put('conflicts', conflictKey(op.entityId, op.seq), record);
          await this.storage.remove('outbox', outboxKey(op.seq));
          conflictCount += 1;
          continue;
        }
        stoppedBy = outcome.kind;
        break;
      }
    } finally {
      this.flushing = false;
    }
    const remaining = await this.pending();
    return { sentCount, conflictCount, remainingCount: remaining.length, stoppedBy };
  }

  async conflicts(): Promise<ConflictRecord[]> {
    const keys = await this.storage.keys('conflicts');
    const records: ConflictRecord[] = [];
    for (const key of keys) {
      const raw = await this.storage.get('conflicts', key);
      if (isConflictRecord(raw)) records.push(raw);
    }
    return records.sort((a, b) => a.seq - b.seq);
  }

  /** 엔티티 id별 충돌 조회 — UI가 목록 카드에 충돌 배지를 붙일 때 쓴다 */
  async conflictsFor(entityId: string): Promise<ConflictRecord[]> {
    const all = await this.conflicts();
    return all.filter((c) => c.entityId === entityId);
  }

  /** 사용자가 충돌을 해소(3택)한 뒤 레코드를 지운다 */
  async removeConflict(entityId: string, seq: number): Promise<void> {
    await this.storage.remove('conflicts', conflictKey(entityId, seq));
  }
}

// ── outbox 항목 → 실제 API 호출 매핑 (flush의 기본 send) ────────────

function saveOutcome<T>(r: SaveResult<T>): OutboxSendOutcome {
  switch (r.kind) {
    case 'ok':
      return { kind: 'ok' };
    case 'conflict':
      return { kind: 'conflict', current: r.current };
    case 'network':
      return { kind: 'network' };
    case 'unauthorized':
      return { kind: 'unauthorized' };
    case 'error':
      return { kind: 'error', messageKo: r.messageKo };
  }
}

function plainOutcome(r: RemoveResult | GetResult<unknown>): OutboxSendOutcome {
  switch (r.kind) {
    case 'ok':
      return { kind: 'ok' };
    case 'network':
      return { kind: 'network' };
    case 'unauthorized':
      return { kind: 'unauthorized' };
    case 'error':
      return { kind: 'error', messageKo: r.messageKo };
  }
}

/** outbox 항목 하나를 리소스 클라이언트로 재전송한다 — `outbox.flush((op) => sendOutboxOp(api, op))` */
export async function sendOutboxOp(api: WorkcellApi, op: OutboxOp): Promise<OutboxSendOutcome> {
  const client = api.entity(op.entityKind);
  switch (op.opKind) {
    case 'create': {
      if (op.request === null) return { kind: 'error', messageKo: MSG_OUTBOX_CORRUPT_KO };
      // 적재 시점 doc은 unknown으로 보존된다 — 종류는 op.entityKind가 보증한다
      const doc = op.request.doc as EntityDocMap[EntityKind];
      return saveOutcome(await client.create(doc));
    }
    case 'update': {
      if (op.request === null || op.request.baseVersion === null) {
        return { kind: 'error', messageKo: MSG_OUTBOX_CORRUPT_KO };
      }
      const doc = op.request.doc as EntityDocMap[EntityKind];
      return saveOutcome(await client.update(op.entityId, doc, op.request.baseVersion));
    }
    case 'remove':
      return plainOutcome(await client.remove(op.entityId));
    case 'restore':
      return plainOutcome(await client.restore(op.entityId));
  }
}
