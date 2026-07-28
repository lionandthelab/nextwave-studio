// ui/document.ts — 워크셀 문서 모델 (docs/UX_AUDIT.md C-3)
//
// ── 왜 "문서"인가 ───────────────────────────────────────────────────
//
// 구 상태에는 사용자가 소유하는 대상이 없었다. 씬은 프리셋이고, 시퀀스는 화면 상태이며,
// 임포트한 메시는 세션 부산물이었다. 그 결과 **작업물이 3중으로 소실**됐다:
//
//   1. 💾 저장이 `SceneSpec`만 직렬화했다 — 30분간 플래너와 그래프로 만든 **시퀀스가
//      "저장했다"는 감각과 함께 사라졌다.** 업로드 쪽은 `{scene, sequence}` 봉투를 받는데
//      저장 쪽은 봉투를 만들지 않는 비대칭이었다.
//   2. `localStorage`/`sessionStorage`가 **완전히 비어 있었다**(플래너 설정 1건 제외).
//      새로고침 한 번에 씬·시퀀스·레이아웃·선택·줌·undo 스택이 전부 휘발했다.
//   3. `confirm(` / `beforeunload` / dirty 표시가 repo 전체에 **0건**이었다. 씬 프리셋을
//      "잠깐 보려고" 바꾸면 경고 없이 모든 것이 증발했다.
//
// 이 모듈이 세 가지를 한 번에 닫는다: **봉투 저장 · IndexedDB 자동저장 · dirty 가드.**
//
// 이 제품의 핵심 산출물은 씬이 아니라 **시퀀스**다(PRD §4 시나리오 2·3). 프로 툴에서
// 가장 용서받지 못하는 실패 유형이고, 한 번 겪으면 사용자는 돌아오지 않는다.
//
// ── 계층 ────────────────────────────────────────────────────────────
// `schema`를 알지만 `core`/`render`를 모른다. 스키마 검증은 로드 시 통합자가 수행한다
// (이 모듈은 **형태만** 다루고 유효성을 판정하지 않는다 — 불변식 §2.9는 통합자 책임).

import { DOCUMENT_EXTENSION, PRODUCT_NAME, STORAGE_PREFIX } from './brand';

// ── 문서 형태 ───────────────────────────────────────────────────────

export const DOCUMENT_VERSION = 1 as const;

/** 자체 완결 워크셀 문서 — "이 파일이 전부다"를 약속한다 */
export interface WorkcellDocument {
  readonly version: typeof DOCUMENT_VERSION;
  /** 사람이 읽는 문서명 (탭 제목·파일명의 기준) */
  readonly name: string;
  readonly savedAtIso: string;
  /** SceneSpec (형태만 — 검증은 로드 시) */
  readonly scene: unknown;
  /** ControlSequence | null */
  readonly sequence: unknown;
  /**
   * 임포트 메시 등 외부 에셋. id → data URI.
   * 없으면 다른 세션에서 열 때 해당 엔티티가 복원되지 않는다(구 동작).
   */
  readonly assets?: Readonly<Record<string, string>>;
  /** 생성 도구 표기 — 포렌식용 */
  readonly generator?: string;
}

/** 로드 시 해석 결과 — 레거시 형식도 여기로 정규화된다 */
export interface ParsedDocument {
  readonly scene: unknown;
  readonly sequence: unknown;
  readonly assets: Readonly<Record<string, string>>;
  /** 문서명 (없으면 null — 통합자가 SceneSpec.name으로 폴백) */
  readonly name: string | null;
  /** 어떤 형식으로 들어왔는가 (진단용) */
  readonly kind: 'document' | 'envelope' | 'bare-scene';
}

/**
 * 업로드/복원 payload를 해석한다. 세 형식을 모두 받는다:
 *   (a) `WorkcellDocument`      — `version` + `scene` 보유
 *   (b) `{ scene, sequence? }`  — 구 봉투 (하위 호환)
 *   (c) `SceneSpec` 단독        — 구 저장 파일 (하위 호환)
 *
 * `SceneSpec`에는 `scene` 필드가 없으므로 세 형식은 모호하지 않다.
 */
export function parseDocument(payload: unknown): ParsedDocument {
  if (payload === null || typeof payload !== 'object') {
    return { scene: payload, sequence: null, assets: {}, name: null, kind: 'bare-scene' };
  }
  const obj = payload as Record<string, unknown>;

  if ('scene' in obj) {
    const isDoc = typeof obj['version'] === 'number';
    const rawAssets = obj['assets'];
    const assets =
      rawAssets !== null && typeof rawAssets === 'object'
        ? (rawAssets as Record<string, string>)
        : {};
    const rawName = obj['name'];
    return {
      scene: obj['scene'],
      sequence: obj['sequence'] ?? null,
      assets,
      name: typeof rawName === 'string' && rawName !== '' ? rawName : null,
      kind: isDoc ? 'document' : 'envelope',
    };
  }
  return { scene: payload, sequence: null, assets: {}, name: null, kind: 'bare-scene' };
}

/** 저장용 문서를 만든다 */
export function createDocument(input: {
  name: string;
  scene: unknown;
  sequence: unknown;
  assets?: Readonly<Record<string, string>>;
  nowIso: string;
}): WorkcellDocument {
  const doc: WorkcellDocument = {
    version: DOCUMENT_VERSION,
    name: input.name,
    savedAtIso: input.nowIso,
    scene: input.scene,
    sequence: input.sequence ?? null,
    generator: PRODUCT_NAME,
  };
  if (input.assets !== undefined && Object.keys(input.assets).length > 0) {
    return { ...doc, assets: input.assets };
  }
  return doc;
}

/** 파일명으로 안전한 문자열로 정규화 */
export function sanitizeDocumentName(name: string): string {
  const cleaned = name
    .trim()
    .replace(/[\\/:*?"<>|]/g, '-')
    .replace(/\s+/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '');
  return cleaned === '' ? 'workcell' : cleaned;
}

/** 다운로드 파일명 — `{이름}.workcell.json` */
export function documentFileName(name: string): string {
  return `${sanitizeDocumentName(name)}${DOCUMENT_EXTENSION}`;
}

/** 문서를 파일로 내려받는다 (정적 호스팅 — 백엔드 불필요) */
export function downloadDocument(doc: WorkcellDocument): string {
  const json = JSON.stringify(doc, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = documentFileName(doc.name);
  anchor.click();
  window.setTimeout(() => {
    URL.revokeObjectURL(url);
  }, 1000);
  return anchor.download;
}

// ── dirty 추적 ──────────────────────────────────────────────────────

/**
 * 저장 상태 추적기.
 *
 * 판정은 직렬화 문자열 비교다 — 씬/시퀀스는 수백 KB 수준이라 편집 커밋 경계에서만
 * 계산하면 충분히 싸다(매 프레임 호출 금지).
 */
export interface DirtyTracker {
  /** 현재 상태를 "저장됨"으로 표시 (저장/로드 직후) */
  markSaved(scene: unknown, sequence: unknown): void;
  /** 현재 상태가 마지막 저장과 다른가 */
  check(scene: unknown, sequence: unknown): boolean;
  isDirty(): boolean;
  /** 상태 변화 시에만 호출된다 (매번이 아니라) */
  onChange(fn: (dirty: boolean) => void): void;
  reset(): void;
}

export function createDirtyTracker(): DirtyTracker {
  let baseline: string | null = null;
  let dirty = false;
  const listeners: ((d: boolean) => void)[] = [];

  const snapshot = (scene: unknown, sequence: unknown): string => {
    try {
      return JSON.stringify({ scene, sequence });
    } catch {
      // 순환 참조 등 — 비교 불가면 항상 dirty로 보수적 판정
      return `__unserializable__${Date.now()}`;
    }
  };

  const setDirty = (next: boolean): void => {
    if (next === dirty) return;
    dirty = next;
    for (const fn of listeners) fn(dirty);
  };

  return {
    markSaved: (scene, sequence): void => {
      baseline = snapshot(scene, sequence);
      setDirty(false);
    },
    check: (scene, sequence): boolean => {
      if (baseline === null) return dirty;
      setDirty(snapshot(scene, sequence) !== baseline);
      return dirty;
    },
    isDirty: (): boolean => dirty,
    onChange: (fn): void => {
      listeners.push(fn);
    },
    reset: (): void => {
      baseline = null;
      setDirty(false);
    },
  };
}

/**
 * 미저장 변경이 있을 때만 이탈 경고를 건다.
 *
 * 항상 거는 것은 사용자 적대적이다 — 아무것도 안 고쳤는데 탭을 닫을 때마다 물으면
 * 사용자는 경고를 학습적으로 무시하게 되고, 정작 중요한 순간에도 무시한다.
 */
export function installUnloadGuard(isDirty: () => boolean): () => void {
  const onBeforeUnload = (e: BeforeUnloadEvent): void => {
    if (!isDirty()) return;
    e.preventDefault();
    // 최신 브라우저는 문구를 무시하고 자체 메시지를 쓴다. returnValue 설정이 트리거다.
    e.returnValue = '';
  };
  window.addEventListener('beforeunload', onBeforeUnload);
  return () => {
    window.removeEventListener('beforeunload', onBeforeUnload);
  };
}

// ── IndexedDB 저장소 ────────────────────────────────────────────────
//
// localStorage는 5MB 상한 + 동기 API라 씬/시퀀스/에셋 봉투에 부적합하다.
// IndexedDB는 백엔드 없이(정적 호스팅 유지) 수십 MB를 비동기로 다룬다.

const DB_NAME = `${STORAGE_PREFIX}-store`;
const DB_VERSION = 1;
const DRAFT_STORE = 'drafts';
const DOCS_STORE = 'documents';
/** 활성 초안의 고정 키 — 항상 1개만 유지한다 */
const DRAFT_KEY = 'active';

export interface DraftRecord {
  readonly key: string;
  readonly doc: WorkcellDocument;
  readonly updatedAtIso: string;
  /** 어느 프리셋/문서에서 파생됐는지 (복원 배너 문구용) */
  readonly originLabel: string | null;
}

export interface DocumentStore {
  saveDraft(doc: WorkcellDocument, originLabel: string | null, nowIso: string): Promise<void>;
  loadDraft(): Promise<DraftRecord | null>;
  clearDraft(): Promise<void>;
  /** 이름 붙인 문서 보관 (문서 라이브러리) */
  putDocument(doc: WorkcellDocument): Promise<void>;
  listDocuments(): Promise<readonly WorkcellDocument[]>;
  deleteDocument(name: string): Promise<void>;
  readonly available: boolean;
}

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = (): void => {
      const db = req.result;
      if (!db.objectStoreNames.contains(DRAFT_STORE)) {
        db.createObjectStore(DRAFT_STORE, { keyPath: 'key' });
      }
      if (!db.objectStoreNames.contains(DOCS_STORE)) {
        db.createObjectStore(DOCS_STORE, { keyPath: 'name' });
      }
    };
    req.onsuccess = (): void => {
      resolve(req.result);
    };
    req.onerror = (): void => {
      reject(req.error ?? new Error('IndexedDB 열기 실패'));
    };
  });
}

function txDone(tx: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    tx.oncomplete = (): void => {
      resolve();
    };
    tx.onerror = (): void => {
      reject(tx.error ?? new Error('IndexedDB 트랜잭션 실패'));
    };
    tx.onabort = (): void => {
      reject(tx.error ?? new Error('IndexedDB 트랜잭션 중단'));
    };
  });
}

/** 저장소를 연다. IndexedDB가 없거나 차단되면 **무해한 no-op 저장소**를 준다. */
export function createDocumentStore(): DocumentStore {
  const available = typeof indexedDB !== 'undefined';
  if (!available) {
    return {
      available: false,
      saveDraft: async (): Promise<void> => undefined,
      loadDraft: async (): Promise<DraftRecord | null> => null,
      clearDraft: async (): Promise<void> => undefined,
      putDocument: async (): Promise<void> => undefined,
      listDocuments: async (): Promise<readonly WorkcellDocument[]> => [],
      deleteDocument: async (): Promise<void> => undefined,
    };
  }

  let dbPromise: Promise<IDBDatabase> | null = null;
  const db = async (): Promise<IDBDatabase> => {
    dbPromise ??= openDb();
    return dbPromise;
  };

  return {
    available: true,
    saveDraft: async (doc, originLabel, nowIso): Promise<void> => {
      const conn = await db();
      const tx = conn.transaction(DRAFT_STORE, 'readwrite');
      const record: DraftRecord = { key: DRAFT_KEY, doc, updatedAtIso: nowIso, originLabel };
      tx.objectStore(DRAFT_STORE).put(record);
      await txDone(tx);
    },
    loadDraft: async (): Promise<DraftRecord | null> => {
      const conn = await db();
      const tx = conn.transaction(DRAFT_STORE, 'readonly');
      const req = tx.objectStore(DRAFT_STORE).get(DRAFT_KEY);
      await txDone(tx);
      const value: unknown = req.result;
      if (value === undefined || value === null) return null;
      return value as DraftRecord;
    },
    clearDraft: async (): Promise<void> => {
      const conn = await db();
      const tx = conn.transaction(DRAFT_STORE, 'readwrite');
      tx.objectStore(DRAFT_STORE).delete(DRAFT_KEY);
      await txDone(tx);
    },
    putDocument: async (doc): Promise<void> => {
      const conn = await db();
      const tx = conn.transaction(DOCS_STORE, 'readwrite');
      tx.objectStore(DOCS_STORE).put(doc);
      await txDone(tx);
    },
    listDocuments: async (): Promise<readonly WorkcellDocument[]> => {
      const conn = await db();
      const tx = conn.transaction(DOCS_STORE, 'readonly');
      const req = tx.objectStore(DOCS_STORE).getAll();
      await txDone(tx);
      const rows: unknown = req.result;
      if (!Array.isArray(rows)) return [];
      return (rows as WorkcellDocument[]).sort((a, b) =>
        b.savedAtIso.localeCompare(a.savedAtIso),
      );
    },
    deleteDocument: async (name): Promise<void> => {
      const conn = await db();
      const tx = conn.transaction(DOCS_STORE, 'readwrite');
      tx.objectStore(DOCS_STORE).delete(name);
      await txDone(tx);
    },
  };
}

// ── 자동저장 ────────────────────────────────────────────────────────

export interface AutosaveDeps {
  readonly store: DocumentStore;
  /** 현재 문서를 만든다. null이면 저장할 게 없다(부팅 중 등). */
  snapshot(): WorkcellDocument | null;
  /** 복원 배너 문구용 라벨 (프리셋명 등) */
  originLabel(): string | null;
  /** 저장 실패 알림 (조용히 실패하면 사용자가 안심하고 잃는다) */
  onError?(message: string): void;
  /** 디바운스 간격 (기본 4초) */
  readonly debounceMs?: number;
}

export interface AutosaveHandle {
  /** 편집이 일어났음을 알린다 — 디바운스되어 저장된다 */
  schedule(): void;
  /** 즉시 저장 (탭 숨김·저장 버튼 등) */
  flush(): Promise<void>;
  dispose(): void;
}

/**
 * 편집 커밋마다 초안을 IndexedDB에 디바운스 저장한다.
 *
 * 브라우저 완결형이라는 장점이 반대로 작동하던 지점이다 — 데스크톱 앱은 크래시해도
 * 복구 파일이 있지만, 이 앱은 탭 실수 하나로 0이 됐다.
 */
export function createAutosave(deps: AutosaveDeps): AutosaveHandle {
  const debounceMs = deps.debounceMs ?? 4000;
  let timer: number | null = null;
  let disposed = false;

  const doSave = async (): Promise<void> => {
    if (disposed) return;
    const doc = deps.snapshot();
    if (doc === null) return;
    try {
      await deps.store.saveDraft(doc, deps.originLabel(), new Date().toISOString());
    } catch (err) {
      deps.onError?.(err instanceof Error ? err.message : String(err));
    }
  };

  const schedule = (): void => {
    if (disposed) return;
    if (timer !== null) window.clearTimeout(timer);
    timer = window.setTimeout(() => {
      timer = null;
      void doSave();
    }, debounceMs);
  };

  // 탭을 숨기거나 닫을 때는 디바운스를 기다리지 않는다 — 마지막 몇 초가 가장 아깝다
  const onVisibility = (): void => {
    if (document.visibilityState === 'hidden') void doSave();
  };
  document.addEventListener('visibilitychange', onVisibility);

  return {
    schedule,
    flush: async (): Promise<void> => {
      if (timer !== null) {
        window.clearTimeout(timer);
        timer = null;
      }
      await doSave();
    },
    dispose: (): void => {
      disposed = true;
      if (timer !== null) window.clearTimeout(timer);
      document.removeEventListener('visibilitychange', onVisibility);
    },
  };
}

// ── 복원 배너 문구 ──────────────────────────────────────────────────

/** "3분 전" 같은 상대 시간 (배너 문구용 — 순수, 테스트 대상) */
export function describeAge(fromIso: string, nowMs: number): string {
  const thenMs = Date.parse(fromIso);
  if (Number.isNaN(thenMs)) return '알 수 없는 시각';
  const deltaSec = Math.max(0, Math.round((nowMs - thenMs) / 1000));
  if (deltaSec < 60) return '방금 전';
  const min = Math.round(deltaSec / 60);
  if (min < 60) return `${min}분 전`;
  const hour = Math.round(min / 60);
  if (hour < 24) return `${hour}시간 전`;
  const day = Math.round(hour / 24);
  return `${day}일 전`;
}
