// ui/console/glue.ts — 콘솔 평면 조립 글루 (Phase 12, docs/BACKEND.md)
//
// main.ts(스튜디오 글루)가 이 모듈 하나를 마운트하면 콘솔 평면 전체가 선다:
// ApiClient·오프라인 계층 → 해시 라우터 → 셸(네비 레일) → 로그인/셋업 → 화면 6종.
// 스튜디오와의 결합은 **StudioBridge 좁은 인터페이스**로만 이뤄진다 — 이 모듈은
// core/render를 모르고, main.ts는 콘솔 화면 내부를 모른다.
//
// ── 책임 분할 ───────────────────────────────────────────────────────
// - 이 모듈: 인증 수명(로그인·전환·로그아웃), 작업/공정 컨텍스트(열기·서버 저장·충돌
//   해결·편집 잠금 heartbeat), 화면 deps 배선, outbox/실행기록 동기화, 라우트별
//   스튜디오 단축키 정지(shortcuts.setEnabled).
// - main.ts: 씬 로드/직렬화, RunRecorder 축적(재생 경로 소유자가 기록도 소유),
//   Ctrl+S 라우팅(consoleRef.saveActive() 우선 → 파일 다운로드 폴백).
//
// ── 오프라인 정책 (BACKEND §6) ──────────────────────────────────────
// 읽기는 listThroughCache/getThroughCache(연결 문제일 때만 캐시 서빙), 쓰기는 outbox
// 순서 보존 재전송 + 충돌 보존(자동 병합 금지 — 사용자가 서버본/덮어쓰기/사본 선택).
// 실행 기록(Run)은 outbox 개체 축이 아니므로 meta 스토어의 자체 대기열로 보존한다.

import {
  ApiCache,
  ApiClient,
  IndexedDbKVStorage,
  OfflineOutbox,
  createWorkcellApi,
  getThroughCache,
  listThroughCache,
  sendOutboxOp,
} from '../../api';
import type {
  ApiResult,
  ConnectionState,
  KVStorage,
  ListOptions,
  WorkcellApi,
} from '../../api';
import type {
  BlockDoc,
  ControlSequence,
  ControlStep,
  ProcessDoc,
  RunRecord,
  TaskDoc,
  UserInfo,
  UserRole,
} from '../../schema';
import { captureBlock, expandBlock } from '../../schema/blocks';
import { runRecordSchema } from '../../schema/entities';
import { createHashRouter, isConsoleScreenName } from '../shell/router';
import type { ConsoleScreenName, Route } from '../shell/router';
import { RAIL_THIN_WIDTH_PX, mountShell } from '../shell/shell';
import type { ScreenFactory, ScreenHandle, ShellHandle } from '../shell/shell';
import { mountLogin } from '../shell/login';
import type { LoginHandle } from '../shell/login';
import { mountTasksScreen } from './tasks-screen';
import { mountProcessesScreen } from './processes-screen';
import { mountBlocksScreen } from './blocks-screen';
import { mountDevicesScreen } from './devices-screen';
import { mountRunsScreen } from './runs-screen';
import type { RunsScreenHandle } from './runs-screen';
import { mountSettingsScreen } from './settings-screen';
import type { ExecDefaults } from './settings-screen';
import { TOUCH_TARGET_MIN_PX, makeModalShell } from './primitives';
import { mountToasts } from '../feedback/toast';
import type { ToastHandle } from '../feedback/toast';
import { LIBRARY_TEMPLATES } from '../library/templates';
import type { LibraryBlockSummary } from '../library/library';
import { parseDocument } from '../document';
import {
  COLLISION,
  SPACE,
  SURFACE,
  TYPE,
  COLOR,
  Z_INDEX,
  applyType,
  makeButton,
  styled,
} from '../theme';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4) ─────────────────────────────

/** 편집 잠금 heartbeat 주기 (ms) — LOCK_TTL_SEC(90s)의 1/3 */
const LOCK_HEARTBEAT_MS = 30_000;
/** 실행 기본값 localStorage 키 */
const EXEC_DEFAULTS_KEY = 'workcell.execDefaults';
/** 마지막 성공 동기화 시각 localStorage 키 */
const LAST_SYNC_KEY = 'workcell.lastSyncIso';
/** meta 스토어의 실행 기록 대기열 키 접두 (outbox는 개체 4종 전용 — 파일 헤더) */
const PENDING_RUN_PREFIX = 'pending-run:';
/** 실행 기록 대기열 상한 — 초과 시 가장 오래된 것부터 폐기 (장기 오프라인 방어) */
const PENDING_RUN_CAP = 500;

const EXEC_DEFAULTS_FALLBACK: ExecDefaults = { autoPauseOnCollision: false, speedMult: 1 };

// ── 스튜디오 브리지 (main.ts가 구현하는 좁은 표면) ──────────────────

export interface TaskRunContext {
  readonly taskId: string;
  readonly taskName: string;
  readonly taskVersion: number;
  readonly processId: string | null;
}

export interface StudioBridge {
  /** 문서(씬+시퀀스)를 스튜디오에 로드하고 저장 기준선/탭 제목을 label로 맞춘다 */
  loadDocument(input: {
    scene: unknown;
    sequence: unknown;
    label: string;
  }): Promise<{ ok: boolean; errors: readonly string[] }>;
  /** 현재 편집 상태의 SceneSpec 깊은 복사 (활성 씬 없으면 null) */
  serializeScene(): unknown | null;
  /** 현재 라이브 시퀀스 (검증 통과본 — 없으면 null) */
  currentSequence(): ControlSequence | null;
  /** 현재 씬의 로봇 엔티티 id 목록 (블록 삽입 대상 선택) */
  currentRobotIds(): readonly string[];
  /** 전개된 블록 step들을 현재 그래프 뒤에 이어 붙인다 (§2.8 파이프라인 경유) */
  insertSteps(steps: readonly ControlStep[]): { ok: boolean; errors: readonly string[] };
  /** 결정론적 재실행 — 기록 화면의 "이 노드부터 재현" */
  runFromNode(nodeId: string): void;
  /** 예제 문서 (새 작업 온보딩용 — 기본 프리셋의 {scene, sequence}) */
  sampleDocument(): { scene: unknown; sequence: unknown };
  /** 빈 바닥 SceneSpec (새 작업/공정 시작점) */
  emptySceneSpec(name: string): unknown;
  openPlannerSettings(): void;
  plannerSummary(): string;
  /** 실행 기본값을 활성 씬에 즉시 적용 (자동 정지 토글·속도) */
  applyExecDefaults(defaults: ExecDefaults): void;
  /** 현재 문서를 파일로 내려받는다 (기존 저장 경로) */
  exportCurrentDocument(): void;
  resetCamera(): void;
  /** 라이브러리 '블록' 섹션 다시 읽기 (블록 저장/삭제 후) */
  reloadLibraryBlocks(): void;
  /** 스튜디오 좌측 여백 — 얇은 레일이 라이브러리를 덮지 않게 한다 */
  setStudioInset(px: number): void;
  /**
   * 서버 저장 성공 → "저장됨" 기준선 갱신 (dirty 추적 동기). snapshot을 주면 **전송한
   * 스냅샷**이 기준선이 된다 — 저장 왕복 중의 편집이 dirty로 남아야 유실되지 않는다.
   * 생략하면 현재 편집 상태를 기준선으로 삼는다.
   */
  markSavedBaseline(snapshot?: { scene: unknown; sequence: unknown }): void;
}

export interface ConsolePlaneDeps {
  readonly bridge: StudioBridge;
  /** 스튜디오 전역 단축키 정지 스위치 — 콘솔 화면에서 Space가 재생을 건드리지 않게 */
  readonly shortcuts: { setEnabled(enabled: boolean): void };
  appLog(level: 'info' | 'warn' | 'error', message: string): void;
}

export interface ConsolePlaneHandle {
  /** 현재 열려 있는 작업 컨텍스트 — RunRecorder begin의 재료 (없으면 기록하지 않는다) */
  currentTaskInfo(): TaskRunContext | null;
  /** 실행 기록의 operator — 서버 세션 사용자, 로컬 모드면 '로컬 사용자' */
  operator(): { id: string; name: string };
  /** 완료된 RunRecord 제출 — 온라인이면 즉시, 아니면 대기열 보존 */
  submitRun(record: RunRecord): void;
  /** Ctrl+S 라우팅: 작업/공정 컨텍스트가 있으면 서버 저장을 처리하고 true */
  saveActive(): Promise<boolean>;
  hasSaveContext(): boolean;
  /** 새 논리 씬 로드 시 작업/공정 컨텍스트 해제 — stale 컨텍스트 덮어쓰기 사고 방지 */
  clearDocumentContext(): void;
  execDefaults(): ExecDefaults;
  /** '블록으로 저장' 다이얼로그 (커맨드바 버튼 → 현재 시퀀스에서 캡처) */
  openBlockCapture(): void;
  /** 라이브러리 '블록' 섹션 공급자 */
  libraryBlocks(): Promise<readonly LibraryBlockSummary[]>;
  /** 라이브러리 블록 카드 클릭 → 파라미터/로봇 다이얼로그 → 그래프 삽입 */
  onLibraryInsertBlock(id: string): void;
  dispose(): void;
}

// ── 실행 기본값 영속 (localStorage) ─────────────────────────────────

function loadExecDefaults(): ExecDefaults {
  try {
    const raw = localStorage.getItem(EXEC_DEFAULTS_KEY);
    if (raw !== null) {
      const parsed = JSON.parse(raw) as Partial<ExecDefaults>;
      const speed = parsed.speedMult;
      return {
        autoPauseOnCollision: parsed.autoPauseOnCollision === true,
        speedMult: speed === 1 || speed === 2 || speed === 4 ? speed : 1,
      };
    }
  } catch {
    // 손상/불가 — 기본값
  }
  return EXEC_DEFAULTS_FALLBACK;
}

function saveExecDefaults(next: ExecDefaults): void {
  try {
    localStorage.setItem(EXEC_DEFAULTS_KEY, JSON.stringify(next));
  } catch {
    // 세션 한정
  }
}

// ── 마운트 ──────────────────────────────────────────────────────────

export function mountConsolePlane(deps: ConsolePlaneDeps): ConsolePlaneHandle {
  const { bridge, appLog } = deps;

  // ── 서비스 계층 ───────────────────────────────────────────────────
  const client = new ApiClient();
  const api: WorkcellApi = createWorkcellApi(client);
  const storage: KVStorage = new IndexedDbKVStorage();
  const cache = new ApiCache(storage);
  const outbox = new OfflineOutbox(storage);

  // ── 콘솔 전용 토스트 (셸 레이어 위 — 스튜디오 토스트 z와 분리) ────
  // 부모 stacking context 트릭: fixed + zIndex 래퍼 안의 fixed 스택은 뷰포트 기준
  // 좌표를 유지하면서 z는 래퍼의 것을 따른다 → 콘솔(panel)·모달(modal) 위에 뜬다.
  const toastLayer = styled(document.createElement('div'), {
    position: 'fixed',
    inset: '0',
    zIndex: String(Z_INDEX.modal + 1),
    pointerEvents: 'none',
  });
  toastLayer.dataset.testid = 'console-toast-layer';
  document.body.appendChild(toastLayer);
  const toasts: ToastHandle = mountToasts(toastLayer);

  // ── 상태 ──────────────────────────────────────────────────────────
  let currentUser: UserInfo | null = null;
  let taskCtx: (TaskRunContext & { lockHeld: boolean }) | null = null;
  let processCtx: { id: string; name: string; version: number } | null = null;
  let heartbeatTimer: ReturnType<typeof setInterval> | null = null;
  let localModeNoticed = false;
  let disposed = false;
  let runsHandle: RunsScreenHandle | null = null;
  /** 연결 모드 판정 후에 마운트된다 (부트 절 참조) — 그 전에는 null */
  let shell: ShellHandle | null = null;
  let login: LoginHandle | null = null;
  const screenHandles = new Map<ConsoleScreenName, ScreenHandle>();

  const conn = (): ConnectionState => client.getState();
  const isServerOnline = (): boolean => {
    const s = conn();
    return s.mode === 'server' && s.online;
  };

  // ── 라우터 · 셸 ───────────────────────────────────────────────────
  const router = createHashRouter();

  const track = (name: ConsoleScreenName, handle: ScreenHandle): ScreenHandle => {
    screenHandles.set(name, handle);
    return handle;
  };

  const refreshScreens = (): void => {
    for (const handle of screenHandles.values()) handle.refresh();
    shell?.refresh();
  };

  // ── 잠금 heartbeat ────────────────────────────────────────────────
  const stopHeartbeat = (): void => {
    if (heartbeatTimer !== null) {
      clearInterval(heartbeatTimer);
      heartbeatTimer = null;
    }
  };

  const releaseTaskLock = (): void => {
    const ctx = taskCtx;
    stopHeartbeat();
    if (ctx?.lockHeld === true && isServerOnline()) {
      void api.locks('task', ctx.taskId, 'release');
    }
  };

  const startHeartbeat = (taskId: string): void => {
    stopHeartbeat();
    heartbeatTimer = setInterval(() => {
      if (!isServerOnline()) return;
      void api.locks('task', taskId, 'heartbeat');
    }, LOCK_HEARTBEAT_MS);
  };

  /**
   * 빠른 사용자 전환 — 문서 컨텍스트·잠금을 **반드시** 내려놓고 타일 화면으로 간다.
   * 안 그러면 사용자 B가 로그인한 뒤 A가 잡은 잠금이 B의 토큰으로 heartbeat되고,
   * B의 Ctrl+S가 A의 작업에 저장된다 (공유 단말의 실제 사고 경로).
   */
  const goSwitchUser = (): void => {
    releaseTaskLock();
    taskCtx = null;
    processCtx = null;
    router.navigate({ name: 'login' });
  };

  // ── 실행 기록 대기열 (meta 스토어 — outbox는 개체 4종 전용) ───────
  const queuePendingRun = async (record: RunRecord): Promise<void> => {
    // 장기 오프라인에서 무한 점유를 막는 상한 — 초과 시 가장 오래된 것부터 폐기하고
    // 사유를 남긴다 (키가 startedAtIso 접두라 사전순 = 시간순).
    const runKeys = (await storage.keys('meta'))
      .filter((k) => k.startsWith(PENDING_RUN_PREFIX))
      .sort();
    while (runKeys.length >= PENDING_RUN_CAP) {
      const oldest = runKeys.shift();
      if (oldest === undefined) break;
      await storage.remove('meta', oldest);
      appLog('warn', `실행 기록 대기열 상한(${PENDING_RUN_CAP}건) — 가장 오래된 1건 폐기`);
    }
    await storage.put('meta', `${PENDING_RUN_PREFIX}${record.startedAtIso}:${record.id}`, record);
  };

  const flushPendingRuns = async (): Promise<number> => {
    const keys = await storage.keys('meta');
    let sent = 0;
    for (const key of keys) {
      if (!key.startsWith(PENDING_RUN_PREFIX)) continue;
      const raw = await storage.get('meta', key);
      const parsed = runRecordSchema.safeParse(raw);
      if (!parsed.success) {
        await storage.remove('meta', key); // 손상 레코드는 보존 가치가 없다
        continue;
      }
      const r = await api.runs.create(parsed.data);
      if (r.kind === 'ok') {
        await storage.remove('meta', key);
        sent += 1;
      } else if (r.kind === 'network' || r.kind === 'unauthorized') {
        // 연결이 죽었거나 세션 만료 — 재시도 가능하다. 대기열에 남기고 중단한다
        // (runs는 append-only 감사 데이터 — outbox.flush의 중단 정책과 동일).
        break;
      } else if (r.status >= 500) {
        break; // 서버 일시 장애 — 다음 동기화에서 재시도
      } else {
        // 4xx 스키마 거부 — 재시도해도 같다. 버리고 사유를 남긴다.
        await storage.remove('meta', key);
        appLog('warn', `실행 기록 1건 폐기 — 서버 거부: ${r.messageKo}`);
      }
    }
    return sent;
  };

  // ── 동기화 (outbox + 실행 기록) ───────────────────────────────────
  const updatePendingBadge = (): void => {
    void (async (): Promise<void> => {
      const ops = await outbox.pending();
      const runKeys = (await storage.keys('meta')).filter((k) =>
        k.startsWith(PENDING_RUN_PREFIX),
      );
      shell?.setPendingCount(ops.length + runKeys.length);
    })();
  };

  let syncing = false;
  /** 저장 in-flight 가드 (saveActive) — Ctrl+S 연타의 자기-409 방지 */
  let saving = false;
  const syncNow = async (): Promise<{ sent: number; conflicts: number; remaining: number }> => {
    if (syncing || !isServerOnline()) return { sent: 0, conflicts: 0, remaining: 0 };
    syncing = true;
    try {
      const report = await outbox.flush((op) => sendOutboxOp(api, op));
      const runsSent = await flushPendingRuns();
      const sent = report.sentCount + runsSent;
      if (sent > 0 || report.conflictCount > 0) {
        try {
          localStorage.setItem(LAST_SYNC_KEY, new Date().toISOString());
        } catch {
          // 세션 한정
        }
      }
      // flush가 만든 충돌 레코드를 화면 갱신 **전에** 반영한다 — 배지 없는 충돌은
      // 사용자가 3선택 다이얼로그에 도달할 방법이 없다 (BACKEND §6).
      await refreshConflictIds();
      updatePendingBadge();
      refreshScreens();
      return { sent, conflicts: report.conflictCount, remaining: report.remainingCount };
    } finally {
      syncing = false;
    }
  };

  // ── 개체 리소스 (읽기는 캐시 경유 — 오프라인에서도 목록이 선다) ───
  const cachedTasks = {
    list: (opts?: ListOptions) => listThroughCache(api.tasks, cache, opts),
    get: (id: string) => getThroughCache(api.tasks, cache, id),
    create: (doc: TaskDoc) => api.tasks.create(doc),
    update: (id: string, doc: TaskDoc, baseVersion: number) =>
      api.tasks.update(id, doc, baseVersion),
    remove: (id: string) => api.tasks.remove(id),
    restore: (id: string) => api.tasks.restore(id),
  };
  const cachedProcesses = {
    list: (opts?: ListOptions) => listThroughCache(api.processes, cache, opts),
    get: (id: string) => getThroughCache(api.processes, cache, id),
    create: (doc: ProcessDoc) => api.processes.create(doc),
    update: (id: string, doc: ProcessDoc, baseVersion: number) =>
      api.processes.update(id, doc, baseVersion),
    remove: (id: string) => api.processes.remove(id),
    restore: (id: string) => api.processes.restore(id),
  };
  const cachedBlocks = {
    list: (opts?: ListOptions) => listThroughCache(api.blocks, cache, opts),
    get: (id: string) => getThroughCache(api.blocks, cache, id),
    create: (doc: BlockDoc) => api.blocks.create(doc),
    update: (id: string, doc: BlockDoc, baseVersion: number) =>
      api.blocks.update(id, doc, baseVersion),
    remove: (id: string) => api.blocks.remove(id),
    restore: (id: string) => api.blocks.restore(id),
  };

  // ── 작업 열기 / 새 작업 ───────────────────────────────────────────

  const openTaskInStudio = async (id: string, opts?: { replayNodeId?: string }): Promise<void> => {
    const r = await cachedTasks.get(id);
    if (r.kind !== 'ok') {
      toasts.show('error', '작업을 불러오지 못했습니다', { detail: r.messageKo });
      return;
    }
    const doc = r.record.doc;
    const loaded = await bridge.loadDocument({
      scene: doc.scene,
      sequence: doc.sequence,
      label: doc.name,
    });
    if (!loaded.ok) {
      toasts.show('error', `'${doc.name}' 열기 실패 — 씬 검증을 통과하지 못했습니다`, {
        detail: loaded.errors.join('\n'),
      });
      return;
    }
    releaseTaskLock();
    processCtx = null;
    const myCtx: TaskRunContext & { lockHeld: boolean } = {
      taskId: doc.id,
      taskName: doc.name,
      taskVersion: r.record.meta.version,
      processId: doc.processId,
      lockHeld: false,
    };
    taskCtx = myCtx;
    if (isServerOnline()) {
      const lock = await api.locks('task', doc.id, 'acquire');
      // await 사이에 다른 작업이 열렸을 수 있다 — 그 사이 taskCtx가 바뀌었으면 이 응답은
      // 낡은 것이다. 그대로 반영하면 (a) 새 작업이 잠금을 잡은 것으로 오표시되고 (b) 옛
      // 작업의 heartbeat가 새 타이머를 덮어써 **타인이 옛 작업을 영영 못 여는** 누수가 된다.
      if (taskCtx !== myCtx) {
        if (lock.kind === 'ok' && lock.lock !== null) {
          void api.locks('task', doc.id, 'release'); // 방금 잡힌 낡은 잠금은 즉시 반납
        }
        return;
      }
      if (lock.kind === 'ok') {
        myCtx.lockHeld = true;
        startHeartbeat(doc.id);
      } else if (lock.kind === 'held') {
        toasts.show('info', `${lock.lock?.userName ?? '다른 사용자'}님이 이 작업을 편집 중입니다`, {
          detail: '저장 시 버전 충돌이 감지되면 선택지를 드립니다.',
        });
      }
    }
    router.navigate({ name: 'studio' });
    appLog('info', `작업 열기: '${doc.name}' (v${r.record.meta.version})`);
    if (opts?.replayNodeId !== undefined) {
      const nodeId = opts.replayNodeId;
      // 씬 로드 직후 그래프가 서 있는 프레임에서 결정론적 재실행을 건다
      window.setTimeout(() => {
        bridge.runFromNode(nodeId);
      }, 0);
    }
  };

  /**
   * 새 작업 생성 + 열기. 반환은 **서버 개체가 실제로 만들어졌는지** — 충돌 해결의
   * "사본으로 저장"이 이 값으로 충돌 레코드 삭제 여부를 결정한다(false면 사용자의
   * 유일본이 conflicts 스토어에 남아 있어야 한다 — 작업 손실 금지).
   */
  const createTaskAndOpen = async (input: {
    name: string;
    scene: unknown;
    sequence: unknown;
    processId: string | null;
    sceneOrigin: { processId: string; processVersion: number } | null;
  }): Promise<boolean> => {
    const doc: TaskDoc = {
      id: crypto.randomUUID(),
      name: input.name,
      processId: input.processId,
      sceneOrigin: input.sceneOrigin,
      scene: input.scene,
      sequence: input.sequence,
      assets: {},
      thumbnail: null,
      notes: '',
    };
    if (!isServerOnline()) {
      // 로컬/오프라인 — 서버 개체 없이 스튜디오만 연다 (컨텍스트 없는 자유 편집).
      // 서버 개체는 만들어지지 않았다 — false (호출자가 원본 보존 여부를 판단한다).
      const loaded = await bridge.loadDocument({
        scene: doc.scene,
        sequence: doc.sequence,
        label: doc.name,
      });
      if (loaded.ok) {
        router.navigate({ name: 'studio' });
        toasts.show('info', '서버 미연결 — 작업은 파일/자동저장으로만 보존됩니다');
      }
      return false;
    }
    const r = await cachedTasks.create(doc);
    if (r.kind !== 'ok') {
      toasts.show('error', '작업을 만들지 못했습니다', {
        detail: 'messageKo' in r ? r.messageKo : undefined,
      });
      return false;
    }
    await openTaskInStudio(doc.id);
    screenHandles.get('tasks')?.refresh();
    return true;
  };

  // ── 이름 입력 소형 다이얼로그 (새 작업/블록 캡처 공용) ────────────

  // ── 글루 모달 공통: 열림 동안 스튜디오 단축키 정지 ────────────────
  // 스튜디오 라우트에서 뜨는 모달(블록 저장/삽입·충돌 해결)의 버튼에 포커스가 있으면
  // Ctrl+Z(undo)·?가 뒤의 씬에 먹는다 — 캡처해 둔 시퀀스와 씬이 어긋난 채 저장되는
  // 사고 경로다. 닫히면 현재 라우트 기준으로 복원한다 (shortcuts.ts의 문서화된 용도).
  let openGlueModals = 0;
  const modalOpened = (): void => {
    openGlueModals += 1;
    deps.shortcuts.setEnabled(false);
  };
  const modalClosed = (): void => {
    openGlueModals = Math.max(0, openGlueModals - 1);
    if (openGlueModals === 0) applyRoute(router.current());
  };

  const promptName = (opts: {
    titleKo: string;
    placeholderKo: string;
    initial: string;
    onSubmit(name: string): void;
  }): void => {
    const modal = makeModalShell({
      titleKo: opts.titleKo,
      onClose: () => {
        modal.root.remove();
        modalClosed();
      },
      testid: 'console-name-dialog',
    });
    const input = document.createElement('input');
    input.type = 'text';
    input.value = opts.initial;
    input.placeholder = opts.placeholderKo;
    input.className = 'ui-input';
    input.dataset.testid = 'console-name-input';
    styled(input, { width: '100%', boxSizing: 'border-box', minHeight: `${TOUCH_TARGET_MIN_PX}px` });
    applyType(input, TYPE.body);
    modal.body.appendChild(input);
    const submit = makeButton('만들기', opts.titleKo, 'console-name-submit', 'primary');
    const cancel = makeButton('취소', '취소', 'console-name-cancel', 'ghost');
    cancel.addEventListener('click', () => {
      modal.close();
      modal.root.remove();
      modalClosed();
    });
    const commit = (): void => {
      const name = input.value.trim();
      if (name === '') {
        input.focus();
        return;
      }
      modal.close();
      modal.root.remove();
      modalClosed();
      opts.onSubmit(name);
    };
    submit.addEventListener('click', commit);
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') commit();
    });
    input.addEventListener('input', () => {
      modal.setDirty(input.value.trim() !== opts.initial.trim());
    });
    modal.footer.append(cancel, submit);
    document.body.appendChild(modal.root);
    modal.open();
    modalOpened();
    input.focus();
    input.select();
  };

  // ── 서버 저장 (작업/공정 컨텍스트) + 충돌 해결 ────────────────────

  /** 버전 충돌 3선택 다이얼로그 — 자동 병합은 없다 (BACKEND §6) */
  const openConflictDialog = (opts: {
    nameKo: string;
    onOpenServer(): void;
    /** null이면 그 선택지를 비활성 + 사유 표기 (내 편집분이 없는 충돌 — remove/restore op) */
    onOverwrite: (() => void) | null;
    onSaveCopy: (() => void) | null;
  }): void => {
    const modal = makeModalShell({
      titleKo: '저장 충돌 — 다른 사용자가 먼저 저장했습니다',
      onClose: () => {
        modal.root.remove();
        modalClosed();
      },
      testid: 'save-conflict-dialog',
    });
    const desc = applyType(document.createElement('p'), TYPE.body);
    styled(desc, { margin: '0', color: COLOR.text });
    desc.textContent =
      `'${opts.nameKo}'의 서버 버전이 내가 연 뒤에 바뀌었습니다. ` +
      '자동으로 합치지 않습니다 — 어떻게 할지 선택하세요.';
    modal.body.appendChild(desc);
    const mk = (
      labelKo: string,
      hintKo: string,
      testid: string,
      run: (() => void) | null,
      variant: 'primary' | 'default' | 'danger',
    ): void => {
      const row = styled(document.createElement('div'), {
        display: 'flex',
        alignItems: 'center',
        gap: SPACE.md,
        marginTop: SPACE.md,
      });
      const btn = makeButton(labelKo, hintKo, testid, variant);
      styled(btn, { minHeight: `${TOUCH_TARGET_MIN_PX}px` });
      if (run === null) {
        // 무음 종료 금지 — 왜 못 누르는지 버튼 자신이 말한다
        btn.disabled = true;
        btn.title = '이 충돌에는 되돌릴 편집 내용이 없습니다 (삭제/복원 요청)';
      } else {
        btn.addEventListener('click', () => {
          modal.close();
          modal.root.remove();
          modalClosed();
          run();
        });
      }
      const hint = applyType(document.createElement('span'), TYPE.caption);
      hint.style.color = COLOR.muted;
      hint.textContent = run === null ? btn.title : hintKo;
      row.append(btn, hint);
      modal.body.appendChild(row);
    };
    mk('사본으로 저장', '내 작업을 새 작업으로 보존한다 (가장 안전)', 'conflict-save-copy', opts.onSaveCopy, 'primary');
    mk('서버본 열기', '내 미저장 변경을 버리고 서버 버전을 연다', 'conflict-open-server', opts.onOpenServer, 'default');
    mk('내 것으로 덮어쓰기', '서버 버전을 내 작업으로 교체한다', 'conflict-overwrite', opts.onOverwrite, 'danger');
    const cancel = makeButton('취소', '아무것도 하지 않는다', 'conflict-cancel', 'ghost');
    cancel.addEventListener('click', () => {
      modal.close();
      modal.root.remove();
      modalClosed();
    });
    modal.footer.appendChild(cancel);
    document.body.appendChild(modal.root);
    modal.open();
    modalOpened();
  };

  const buildTaskDocFromStudio = (ctx: TaskRunContext): TaskDoc | null => {
    const scene = bridge.serializeScene();
    if (scene === null) return null;
    return {
      id: ctx.taskId,
      name: ctx.taskName,
      processId: ctx.processId,
      sceneOrigin: null, // 유지: 서버 현재본의 sceneOrigin을 보존한다 (아래 update 시 재부착)
      scene,
      sequence: bridge.currentSequence(),
      assets: {},
      thumbnail: null,
      notes: '',
    };
  };

  const saveTaskCtx = async (): Promise<boolean> => {
    const ctx = taskCtx;
    if (ctx === null) return false;
    const doc = buildTaskDocFromStudio(ctx);
    if (doc === null) {
      toasts.show('warn', '저장할 씬이 없습니다');
      return true;
    }
    // 서버 현재본의 sceneOrigin/notes/thumbnail을 보존한다 — 스튜디오는 씬·시퀀스만 진실
    const current = await cachedTasks.get(ctx.taskId);
    const preserved: TaskDoc =
      current.kind === 'ok'
        ? {
            ...doc,
            sceneOrigin: current.record.doc.sceneOrigin,
            notes: current.record.doc.notes,
            thumbnail: current.record.doc.thumbnail,
            assets: current.record.doc.assets,
          }
        : doc;
    const r = await api.tasks.update(ctx.taskId, preserved, ctx.taskVersion);
    if (r.kind === 'ok') {
      if (taskCtx?.taskId === ctx.taskId) {
        taskCtx = { ...ctx, taskVersion: r.record.meta.version, taskName: preserved.name };
      }
      // 기준선은 **전송한 스냅샷** — 저장 왕복 중의 편집이 dirty로 남아야 유실되지 않는다
      bridge.markSavedBaseline({ scene: preserved.scene, sequence: preserved.sequence });
      toasts.show('success', `서버에 저장됨 — '${preserved.name}' (v${r.record.meta.version})`);
      screenHandles.get('tasks')?.refresh();
      return true;
    }
    if (r.kind === 'conflict') {
      const serverCurrent = r.current;
      openConflictDialog({
        nameKo: ctx.taskName,
        onOpenServer: () => {
          void openTaskInStudio(ctx.taskId);
        },
        onOverwrite: () => {
          void (async (): Promise<void> => {
            // 서버 현재 버전을 모르면 덮어쓰기를 진행하지 않는다 — 임의 폴백(1 등)은
            // 검사 없는 무단 교체가 된다
            let base = serverCurrent?.meta.version ?? null;
            if (base === null) {
              const cur = await api.tasks.get(ctx.taskId);
              if (cur.kind === 'ok') base = cur.record.meta.version;
            }
            if (base === null) {
              toasts.show('error', '서버 버전을 확인할 수 없어 덮어쓰기를 중단했습니다');
              return;
            }
            const again = await api.tasks.update(ctx.taskId, preserved, base);
            if (again.kind === 'ok') {
              if (taskCtx?.taskId === ctx.taskId) {
                taskCtx = { ...ctx, taskVersion: again.record.meta.version };
              }
              bridge.markSavedBaseline({ scene: preserved.scene, sequence: preserved.sequence });
              screenHandles.get('tasks')?.refresh();
              toasts.show('success', `덮어쓰기 저장됨 (v${again.record.meta.version})`);
            } else {
              toasts.show('error', '덮어쓰기에 실패했습니다', {
                detail: 'messageKo' in again ? again.messageKo : undefined,
              });
            }
          })();
        },
        onSaveCopy: () => {
          promptName({
            titleKo: '사본으로 저장',
            placeholderKo: '새 작업 이름',
            initial: `${ctx.taskName} (사본)`,
            onSubmit: (name) => {
              void createTaskAndOpen({
                name,
                scene: preserved.scene,
                sequence: preserved.sequence,
                processId: preserved.processId,
                sceneOrigin: preserved.sceneOrigin,
              });
            },
          });
        },
      });
      return true;
    }
    if (r.kind === 'network') {
      // 같은 작업의 이전 대기 update를 걷어낸다 — 안 그러면 재연결 flush에서 첫 op가
      // 서버 버전을 올리고 **자기 자신의 두 번째 op가 409**(가짜 충돌)가 된다 (offline.ts).
      await outbox.dropPendingUpdates('tasks', ctx.taskId);
      await outbox.enqueue({
        opKind: 'update',
        entityKind: 'tasks',
        entityId: ctx.taskId,
        request: { doc: preserved, baseVersion: ctx.taskVersion },
      });
      updatePendingBadge();
      toasts.show('warn', '오프라인 — 변경이 동기화 대기열에 저장되었습니다');
      return true;
    }
    toasts.show('error', '저장에 실패했습니다', {
      detail: 'messageKo' in r ? r.messageKo : undefined,
    });
    return true;
  };

  const saveProcessCtx = async (): Promise<boolean> => {
    const ctx = processCtx;
    if (ctx === null) return false;
    const scene = bridge.serializeScene();
    if (scene === null) return true;
    const current = await cachedProcesses.get(ctx.id);
    if (current.kind !== 'ok') {
      toasts.show('error', '공정을 다시 읽지 못해 저장할 수 없습니다');
      return true;
    }
    const doc: ProcessDoc = { ...current.record.doc, scene };
    const r = await api.processes.update(ctx.id, doc, ctx.version);
    if (r.kind === 'ok') {
      if (processCtx?.id === ctx.id) processCtx = { ...ctx, version: r.record.meta.version };
      bridge.markSavedBaseline({ scene, sequence: bridge.currentSequence() });
      toasts.show(
        'success',
        `공정 씬 저장됨 — '${ctx.name}' (v${r.record.meta.version}). 기존 작업들은 바뀌지 않습니다`,
      );
      screenHandles.get('processes')?.refresh();
      return true;
    }
    if (r.kind === 'conflict') {
      toasts.show('error', '공정이 다른 곳에서 먼저 저장되었습니다 — 공정 화면에서 다시 여세요');
      return true;
    }
    if (r.kind === 'network') {
      // 작업 저장과 동일한 오프라인 정책 — 조용한 소실 금지 (BACKEND §6)
      await outbox.dropPendingUpdates('processes', ctx.id);
      await outbox.enqueue({
        opKind: 'update',
        entityKind: 'processes',
        entityId: ctx.id,
        request: { doc, baseVersion: ctx.version },
      });
      updatePendingBadge();
      toasts.show('warn', '오프라인 — 공정 변경이 동기화 대기열에 저장되었습니다');
      return true;
    }
    toasts.show('error', '공정 저장 실패', { detail: 'messageKo' in r ? r.messageKo : undefined });
    return true;
  };

  // ── 블록: 삽입(전개) · 캡처 ──────────────────────────────────────

  const openBlockInsertDialog = (block: BlockDoc): void => {
    const robots = bridge.currentRobotIds();
    const modal = makeModalShell({
      titleKo: `블록 삽입 — ${block.name}`,
      onClose: () => {
        modal.root.remove();
        modalClosed();
      },
      testid: 'block-insert-dialog',
    });

    const info = applyType(document.createElement('p'), TYPE.caption);
    styled(info, { margin: '0', color: COLOR.muted });
    info.textContent = `step ${block.steps.length}개가 현재 시퀀스 뒤에 이어 붙습니다 (사본 — 블록 수정은 전파되지 않음)`;
    modal.body.appendChild(info);

    // 대상 로봇 (씬에 로봇이 있을 때만 — 없으면 삽입 자체가 검증에서 거부된다)
    let robotSelect: HTMLSelectElement | null = null;
    if (robots.length > 0) {
      const label = applyType(document.createElement('label'), TYPE.caption);
      styled(label, { display: 'block', marginTop: SPACE.lg, color: COLOR.label });
      label.textContent = '대상 로봇';
      robotSelect = document.createElement('select');
      robotSelect.className = 'ui-select';
      robotSelect.dataset.testid = 'block-insert-robot';
      robotSelect.lang = 'en';
      styled(robotSelect, { width: '100%', minHeight: `${TOUCH_TARGET_MIN_PX}px`, marginTop: SPACE.xs });
      for (const id of robots) {
        const opt = document.createElement('option');
        opt.value = id;
        opt.textContent = id;
        robotSelect.appendChild(opt);
      }
      label.appendChild(robotSelect);
      modal.body.appendChild(label);
    }

    // 파라미터 입력
    const paramInputs = new Map<string, HTMLInputElement>();
    for (const param of block.params) {
      const label = applyType(document.createElement('label'), TYPE.caption);
      styled(label, { display: 'block', marginTop: SPACE.lg, color: COLOR.label });
      label.textContent = param.labelKo;
      const input = document.createElement('input');
      input.className = 'ui-input';
      input.dataset.testid = `block-param-${param.key}`;
      styled(input, { width: '100%', minHeight: `${TOUCH_TARGET_MIN_PX}px`, marginTop: SPACE.xs, boxSizing: 'border-box' });
      if (param.kind === 'number') {
        input.type = 'number';
        input.required = true; // 빈 칸을 0으로 조용히 해석하지 않는다 (아래 Number.NaN 승격)
        if (param.min !== undefined) input.min = String(param.min);
        if (param.max !== undefined) input.max = String(param.max);
        input.value = String(param.defaultValue);
      } else if (param.kind === 'boolean') {
        input.type = 'checkbox';
        input.checked = param.defaultValue === true;
        styled(input, { width: 'auto', minHeight: '24px' });
      } else {
        input.type = 'text';
        input.value = String(param.defaultValue);
      }
      paramInputs.set(param.key, input);
      label.appendChild(input);
      modal.body.appendChild(label);
    }

    const errorLine = applyType(document.createElement('p'), TYPE.caption);
    styled(errorLine, { margin: `${SPACE.md} 0 0`, color: COLLISION.text, whiteSpace: 'pre-wrap' });
    errorLine.setAttribute('role', 'alert');
    modal.body.appendChild(errorLine);

    const cancel = makeButton('취소', '취소', 'block-insert-cancel', 'ghost');
    cancel.addEventListener('click', () => {
      modal.close();
      modal.root.remove();
      modalClosed();
    });
    const insert = makeButton('시퀀스에 삽입', '전개하여 현재 그래프 뒤에 추가', 'block-insert-confirm', 'primary');
    styled(insert, { minHeight: `${TOUCH_TARGET_MIN_PX}px` });
    insert.addEventListener('click', () => {
      const paramValues: Record<string, unknown> = {};
      for (const param of block.params) {
        const input = paramInputs.get(param.key);
        if (!input) continue;
        if (param.kind === 'number') {
          // 빈 칸은 `Number('') === 0`이라 검증을 통과해 **사용자가 비웠다고 생각한 값이
          // 0으로 삽입**된다. NaN으로 승격해 expandBlock이 한국어 사유로 거부하게 한다.
          paramValues[param.key] = input.value.trim() === '' ? Number.NaN : Number(input.value);
        } else if (param.kind === 'boolean') {
          paramValues[param.key] = input.checked;
        } else {
          paramValues[param.key] = input.value;
        }
      }
      const expanded = expandBlock(block, {
        targetRobotId: robotSelect?.value ?? null,
        paramValues,
      });
      if (!expanded.ok) {
        errorLine.textContent = expanded.errors.join('\n');
        return;
      }
      const inserted = bridge.insertSteps(expanded.steps);
      if (!inserted.ok) {
        errorLine.textContent = inserted.errors.join('\n');
        return;
      }
      modal.close();
      modal.root.remove();
      modalClosed();
      toasts.show('success', `'${block.name}' 블록 삽입됨 — step ${expanded.steps.length}개`);
      router.navigate({ name: 'studio' });
    });
    modal.footer.append(cancel, insert);
    document.body.appendChild(modal.root);
    modal.open();
    modalOpened();
  };

  const insertBlockById = (blockId: string): void => {
    void (async (): Promise<void> => {
      const r = await cachedBlocks.get(blockId);
      if (r.kind !== 'ok') {
        toasts.show('error', '블록을 불러오지 못했습니다', { detail: r.messageKo });
        return;
      }
      openBlockInsertDialog(r.record.doc);
    })();
  };

  const openBlockCapture = (): void => {
    const seq = bridge.currentSequence();
    if (seq === null || seq.steps.length === 0) {
      toasts.show('warn', '캡처할 시퀀스가 없습니다 — 플로우에 노드를 먼저 만드세요');
      return;
    }
    if (!isServerOnline()) {
      toasts.show('warn', '서버 미연결 — 블록은 서버에 저장됩니다. 연결 후 다시 시도하세요');
      return;
    }
    promptName({
      titleKo: '현재 시퀀스를 블록으로 저장',
      placeholderKo: '블록 이름 (예: 픽업 후 존 적재)',
      initial: '',
      onSubmit: (name) => {
        const captured = captureBlock(seq.steps, { name, descriptionKo: '' });
        if (!captured.ok) {
          toasts.show('error', '블록 캡처 실패', { detail: captured.errors.join('\n') });
          return;
        }
        void (async (): Promise<void> => {
          const r = await cachedBlocks.create(captured.block);
          if (r.kind === 'ok') {
            toasts.show('success', `블록 '${name}' 저장됨 — 라이브러리와 블록 화면에서 재사용`);
            bridge.reloadLibraryBlocks();
            screenHandles.get('blocks')?.refresh();
          } else {
            toasts.show('error', '블록 저장 실패', {
              detail: 'messageKo' in r ? r.messageKo : undefined,
            });
          }
        })();
      },
    });
  };

  // ── 사용자 관리 API (settings — resources에 없는 /users 표면) ─────

  const usersApi = {
    list: async (): Promise<
      { kind: 'ok'; users: UserInfo[] } | { kind: 'error'; messageKo: string }
    > => {
      const r: ApiResult<{ users: UserInfo[] }> = await client.request('GET', '/users');
      if (r.kind === 'ok') return { kind: 'ok', users: r.data.users };
      return { kind: 'error', messageKo: r.messageKo };
    },
    create: async (input: {
      name: string;
      pin: string;
      role: UserRole;
    }): Promise<{ kind: 'ok'; user: UserInfo } | { kind: 'error'; messageKo: string }> => {
      const r: ApiResult<{ user: UserInfo }> = await client.request('POST', '/users', {
        body: input,
      });
      if (r.kind === 'ok') return { kind: 'ok', user: r.data.user };
      return { kind: 'error', messageKo: r.messageKo };
    },
    patch: async (
      id: string,
      patch: { role?: UserRole; active?: boolean; pin?: string },
    ): Promise<{ kind: 'ok'; user: UserInfo } | { kind: 'error'; messageKo: string }> => {
      const r: ApiResult<{ user: UserInfo }> = await client.request('PATCH', `/users/${id}`, {
        body: patch,
      });
      if (r.kind === 'ok') return { kind: 'ok', user: r.data.user };
      return { kind: 'error', messageKo: r.messageKo };
    },
  };

  // ── 화면 팩토리 ───────────────────────────────────────────────────

  const screens: Partial<Record<ConsoleScreenName, ScreenFactory>> = {
    tasks: (host) =>
      track(
        'tasks',
        mountTasksScreen(host, {
          tasks: cachedTasks,
          processes: cachedProcesses,
          locks: (taskId) => api.getLock('task', taskId),
          conflicts: () => conflictTaskIds,
          onOpenTask: (id) => {
            void openTaskInStudio(id);
          },
          onNewTask: () => {
            promptName({
              titleKo: '새 작업',
              placeholderKo: '작업 이름 (예: 3번 라인 팔레타이징)',
              initial: '',
              onSubmit: (name) => {
                void createTaskAndOpen({
                  name,
                  scene: bridge.emptySceneSpec(name),
                  sequence: null,
                  processId: null,
                  sceneOrigin: null,
                });
              },
            });
          },
          onNewFromSample: () => {
            const sample = bridge.sampleDocument();
            promptName({
              titleKo: '예제에서 시작',
              placeholderKo: '작업 이름',
              initial: '예제 — 로봇팔과 상자',
              onSubmit: (name) => {
                void createTaskAndOpen({
                  name,
                  scene: sample.scene,
                  sequence: sample.sequence,
                  processId: null,
                  sceneOrigin: null,
                });
              },
            });
          },
          onResolveConflict: (id) => {
            void resolveConflictFor(id);
          },
          connection: conn,
          toast: toasts,
          // getter — 화면은 1회 마운트되므로 로그인 후의 사용자를 렌더 시점마다 읽어야 한다
          currentUserId: () => currentUser?.id ?? null,
        }),
      ),
    processes: (host) =>
      track(
        'processes',
        mountProcessesScreen(host, {
          processes: cachedProcesses,
          devices: { list: () => listThroughCache(api.devices, cache) },
          tasks: { list: (opts) => listThroughCache(api.tasks, cache, opts) },
          onCreateTask: (processId) => {
            void (async (): Promise<void> => {
              const p = await cachedProcesses.get(processId);
              if (p.kind !== 'ok') {
                toasts.show('error', '공정을 불러오지 못했습니다');
                return;
              }
              promptName({
                titleKo: `'${p.record.doc.name}' 공정으로 새 작업`,
                placeholderKo: '작업 이름',
                initial: '',
                onSubmit: (name) => {
                  // 복사본 의미론 (CLAUDE.md §3 Phase 12+): 씬 전체 사본 + 출처 기록
                  void createTaskAndOpen({
                    name,
                    scene: structuredClone(p.record.doc.scene),
                    sequence: null,
                    processId,
                    sceneOrigin: { processId, processVersion: p.record.meta.version },
                  });
                },
              });
            })();
          },
          onEditScene: (processId) => {
            void (async (): Promise<void> => {
              const p = await cachedProcesses.get(processId);
              if (p.kind !== 'ok') {
                toasts.show('error', '공정을 불러오지 못했습니다');
                return;
              }
              const scene =
                p.record.doc.scene ?? bridge.emptySceneSpec(p.record.doc.name);
              const loaded = await bridge.loadDocument({
                scene,
                sequence: null,
                label: `공정 — ${p.record.doc.name}`,
              });
              if (!loaded.ok) {
                toasts.show('error', '공정 씬을 열지 못했습니다', {
                  detail: loaded.errors.join('\n'),
                });
                return;
              }
              releaseTaskLock();
              taskCtx = null;
              processCtx = {
                id: processId,
                name: p.record.doc.name,
                version: p.record.meta.version,
              };
              router.navigate({ name: 'studio' });
              toasts.show('info', 'Ctrl+S가 공정 씬을 저장합니다 — 기존 작업들은 바뀌지 않습니다');
            })();
          },
          onCaptureCurrentScene: () => Promise.resolve(bridge.serializeScene()),
          onOpenTask: (taskId) => {
            void openTaskInStudio(taskId);
          },
          connection: conn,
          toast: toasts,
        }),
      ),
    blocks: (host) =>
      track(
        'blocks',
        mountBlocksScreen(host, {
          blocks: cachedBlocks,
          onInsertBlock: insertBlockById,
          connection: conn,
          toast: toasts,
        }),
      ),
    devices: (host) =>
      track(
        'devices',
        mountDevicesScreen(host, {
          devices: {
            list: (opts) => listThroughCache(api.devices, cache, opts),
            get: (id) => getThroughCache(api.devices, cache, id),
            create: (doc) => api.devices.create(doc),
            update: (id, doc, baseVersion) => api.devices.update(id, doc, baseVersion),
            remove: (id) => api.devices.remove(id),
            restore: (id) => api.devices.restore(id),
          },
          processes: {
            list: () => listThroughCache(api.processes, cache),
            get: (id) => getThroughCache(api.processes, cache, id),
          },
          templatesProvider: () => LIBRARY_TEMPLATES,
          connection: conn,
          toast: toasts,
          onApplyCameraPreset: () => {
            bridge.resetCamera();
            router.navigate({ name: 'studio' });
            toasts.show('info', '뷰포트 기본 시점 적용 — 카메라 장비 시점은 준비 중입니다');
          },
        }),
      ),
    runs: (host) => {
      const handle = mountRunsScreen(host, {
        runs: api.runs,
        stats: (taskId) => api.taskStats(taskId),
        onOpenTask: (taskId) => {
          void openTaskInStudio(taskId);
        },
        onReplayFromNode: (taskId, nodeId) => {
          void openTaskInStudio(taskId, { replayNodeId: nodeId });
        },
        getConnection: conn,
        initialTaskId: router.current().name === 'runs' ? (router.current().id ?? null) : null,
      });
      runsHandle = handle;
      return track('runs', handle);
    },
    settings: (host) =>
      track(
        'settings',
        mountSettingsScreen(host, {
          me: () => currentUser,
          users: usersApi,
          changeMyPin: async ({ currentPin, newPin }) => {
            const user = currentUser;
            if (user === null) return { kind: 'error', messageKo: '로그인이 필요합니다' };
            // 현재 PIN 검증은 실제 로그인으로 — 서버에 별도 확인 API를 늘리지 않는다
            const verify = await client.login({ userId: user.id, pin: currentPin });
            if (verify.kind === 'invalid')
              return { kind: 'error', messageKo: '현재 PIN이 일치하지 않습니다' };
            if (verify.kind !== 'ok')
              return { kind: 'error', messageKo: verify.messageKo };
            const r = await usersApi.patch(user.id, { pin: newPin });
            if (r.kind === 'ok') return { kind: 'ok' };
            return { kind: 'error', messageKo: r.messageKo };
          },
          onSwitchUser: goSwitchUser,
          onLogout: () => {
            void (async (): Promise<void> => {
              releaseTaskLock();
              taskCtx = null;
              processCtx = null;
              await client.logout();
              currentUser = null;
              shell?.refresh();
              router.navigate({ name: 'login' });
            })();
          },
          plannerSummary: () => bridge.plannerSummary(),
          onOpenPlannerSettings: () => bridge.openPlannerSettings(),
          execDefaults: {
            get: loadExecDefaults,
            set: (next) => {
              saveExecDefaults(next);
              bridge.applyExecDefaults(next);
            },
          },
          sync: {
            pendingCount: async () => {
              const ops = await outbox.pending();
              const runKeys = (await storage.keys('meta')).filter((k) =>
                k.startsWith(PENDING_RUN_PREFIX),
              );
              return ops.length + runKeys.length;
            },
            syncNow: async () => {
              if (!isServerOnline()) {
                return { kind: 'error', messageKo: '서버에 연결되어 있지 않습니다' };
              }
              const r = await syncNow();
              return {
                kind: 'ok',
                sentCount: r.sent,
                conflictCount: r.conflicts,
                remainingCount: r.remaining,
              };
            },
            lastSyncIso: () => {
              try {
                return localStorage.getItem(LAST_SYNC_KEY);
              } catch {
                return null;
              }
            },
          },
          storageEstimate: async () => {
            try {
              const est = await navigator.storage.estimate();
              if (est.usage === undefined || est.quota === undefined) return null;
              return { usageBytes: est.usage, quotaBytes: est.quota };
            } catch {
              return null;
            }
          },
          onExportAll: () => {
            bridge.exportCurrentDocument();
          },
          onImportAll: () => {
            importFileInput.click();
          },
          health: async () => {
            const r: ApiResult<{ ok: boolean; name: string; version: string }> =
              await client.request('GET', '/health', { auth: false });
            if (r.kind !== 'ok') return null;
            return { name: r.data.name, version: r.data.version };
          },
          connection: {
            get: conn,
            onChange: (cb) => client.onStateChange(cb),
          },
        }),
      ),
  };

  // ── 동기화 충돌 해결 (tasks 화면 배지 → 3선택) ────────────────────

  let conflictTaskIds: readonly string[] = [];

  const refreshConflictIds = async (): Promise<void> => {
    const records = await outbox.conflicts();
    conflictTaskIds = records
      .filter((c) => c.entityKind === 'tasks')
      .map((c) => c.entityId);
  };

  const resolveConflictFor = async (taskId: string): Promise<void> => {
    const records = await outbox.conflicts();
    const record = records.find((c) => c.entityKind === 'tasks' && c.entityId === taskId);
    if (record === undefined) {
      await refreshConflictIds();
      screenHandles.get('tasks')?.refresh();
      return;
    }
    const mine = record.request?.doc as TaskDoc | undefined;
    const nameKo = mine?.name ?? taskId;
    openConflictDialog({
      nameKo,
      onOpenServer: () => {
        void (async (): Promise<void> => {
          await outbox.removeConflict(record.entityId, record.seq);
          await refreshConflictIds();
          await openTaskInStudio(taskId);
        })();
      },
      // 내 편집분이 없는 충돌(remove/restore op)에서는 두 선택지를 비활성 + 사유 표기한다
      onOverwrite: mine === undefined ? null : () => {
        void (async (): Promise<void> => {
          const cur = await api.tasks.get(taskId);
          if (cur.kind !== 'ok') {
            // 서버 현재 버전을 모른 채 임의 base로 덮어쓰지 않는다
            toasts.show('error', '서버 버전을 확인할 수 없어 덮어쓰기를 중단했습니다', {
              detail: cur.messageKo,
            });
            return;
          }
          const r = await api.tasks.update(taskId, mine, cur.record.meta.version);
          if (r.kind === 'ok') {
            await outbox.removeConflict(record.entityId, record.seq);
            toasts.show('success', '내 변경으로 덮어썼습니다');
          } else {
            toasts.show('error', '덮어쓰기 실패', {
              detail: 'messageKo' in r ? r.messageKo : undefined,
            });
          }
          await refreshConflictIds();
          screenHandles.get('tasks')?.refresh();
        })();
      },
      onSaveCopy: mine === undefined ? null : () => {
        promptName({
          titleKo: '사본으로 저장',
          placeholderKo: '새 작업 이름',
          initial: `${nameKo} (사본)`,
          onSubmit: (name) => {
            void (async (): Promise<void> => {
              const created = await createTaskAndOpen({
                name,
                scene: mine.scene,
                sequence: mine.sequence,
                processId: mine.processId,
                sceneOrigin: mine.sceneOrigin,
              });
              // 사본 생성이 **성공했을 때만** 충돌 레코드를 지운다 — 실패(401·5xx·
              // 네트워크) 시 이 레코드가 오프라인 편집분의 유일본이다 (작업 손실 금지).
              if (created) {
                await outbox.removeConflict(record.entityId, record.seq);
                await refreshConflictIds();
                screenHandles.get('tasks')?.refresh();
              } else {
                toasts.show('warn', '사본이 만들어지지 않아 충돌 항목을 보존했습니다');
              }
            })();
          },
        });
      },
    });
  };

  // ── 설정 import 파일 입력 (숨김) ──────────────────────────────────

  const importFileInput = document.createElement('input');
  importFileInput.type = 'file';
  importFileInput.accept = '.json,.workcell.json';
  importFileInput.style.display = 'none';
  importFileInput.dataset.testid = 'console-import-input';
  document.body.appendChild(importFileInput);
  importFileInput.addEventListener('change', () => {
    const file = importFileInput.files?.[0];
    importFileInput.value = '';
    if (!file) return;
    void (async (): Promise<void> => {
      try {
        const parsed = parseDocument(JSON.parse(await file.text()));
        const loaded = await bridge.loadDocument({
          scene: parsed.scene,
          sequence: parsed.sequence,
          label: parsed.name ?? file.name,
        });
        if (loaded.ok) {
          releaseTaskLock();
          taskCtx = null;
          processCtx = null;
          router.navigate({ name: 'studio' });
        } else {
          toasts.show('error', '가져오기 실패 — 씬 검증을 통과하지 못했습니다', {
            detail: loaded.errors.join('\n'),
          });
        }
      } catch (err) {
        toasts.show('error', '가져오기 실패 — JSON을 읽을 수 없습니다', {
          detail: err instanceof Error ? err.message : String(err),
        });
      }
    })();
  });

  // ── 셸 · 로그인 (연결 판정 **후** 마운트 — 순서가 계약이다) ───────
  //
  // 셸을 즉시 마운트하면 기본 라우트(#/tasks)의 화면이 연결 모드 판정 전에 API를
  // 발사한다. 정적 배포에서 그 응답(404 또는 SPA 폴백 200 HTML)이 연결 상태 머신을
  // "서버 도달"로 되돌려, 서버가 없는데 온라인 배지가 뜨는 고착이 생긴다. 그래서
  // client.start()가 모드를 판정한 뒤에야 셸을 마운트하고 첫 라우트를 적용한다 —
  // 그 전까지는 스튜디오가 그대로 보인다(정적 배포/게이트 경로 무변경, BACKEND §1).

  // 로그인 레이어 — 셸(panel)보다 위, 모달과 같은 층 (인증은 다른 모든 것을 가린다)
  const loginLayer = styled(document.createElement('div'), {
    position: 'fixed',
    inset: '0',
    zIndex: String(Z_INDEX.modal),
    display: 'none',
    background: SURFACE.base,
    overflow: 'auto',
  });
  loginLayer.dataset.testid = 'login-layer';
  document.body.appendChild(loginLayer);

  /** 로그인 화면은 auth 라우트에 처음 진입할 때 마운트한다 (마운트 자체가 bootstrap을 부른다) */
  const ensureLoginShown = (): void => {
    if (login !== null) {
      login.refresh(); // 재진입 — 서버 상태가 바뀌었을 수 있다
      return;
    }
    login = mountLogin(loginLayer, {
      api: client,
      onLoggedIn: (user) => {
        currentUser = user;
        shell?.refresh();
        refreshScreens();
        router.navigate({ name: 'tasks' });
        void syncNow();
        void refreshConflictIds(); // 이전 세션의 미해결 충돌 배지 복원
      },
      onLocalMode: () => {
        router.navigate({ name: 'studio' });
        if (!localModeNoticed) {
          localModeNoticed = true;
          toasts.show('info', '로컬 모드 — 작업물은 이 브라우저(자동저장·파일)에만 보존됩니다');
        }
      },
    });
  };

  // ── 라우트 반응 (단축키 정지 · 로그인 레이어 · 스튜디오 여백) ─────

  const applyRoute = (route: Route): void => {
    const authRoute = route.name === 'login' || route.name === 'setup';
    const consoleOpen = isConsoleScreenName(route.name);
    // 콘솔/로그인에서는 스튜디오 전역 단축키(Space 재생 등)를 정지한다
    deps.shortcuts.setEnabled(!consoleOpen && !authRoute);
    loginLayer.style.display = authRoute ? 'block' : 'none';
    if (authRoute) ensureLoginShown();
    // 레일이 실제로 보일 때만 워크스페이스를 비켜 세운다. 로컬 모드에서는 레일이
    // 숨겨지므로(shell.railModeForRoute) 여백도 0이어야 한다 — 서버가 없으면 화면이
    // 이전과 픽셀 단위로 같아야 한다는 정적 배포 약속(BACKEND §1).
    const railVisible = route.name === 'studio' && conn().mode !== 'local';
    bridge.setStudioInset(railVisible ? RAIL_THIN_WIDTH_PX : 0);
    if (route.name === 'runs') runsHandle?.setTaskFilter(route.id ?? null);
  };
  const offRoute = router.subscribe(applyRoute);

  // ── 연결 상태 반응 (재연결 → 자동 동기화) ─────────────────────────

  let lastOnline = false;
  const offState = client.onStateChange((state) => {
    const nowOnline = state.mode === 'server' && state.online;
    if (nowOnline && !lastOnline) {
      void (async (): Promise<void> => {
        const r = await syncNow(); // 내부에서 refreshConflictIds까지 수행
        if (r.sent > 0) {
          toasts.show('success', `다시 연결됨 — 대기 중이던 ${r.sent}건을 서버에 보냈습니다`);
        }
        if (r.conflicts > 0) {
          toasts.show('warn', `동기화 충돌 ${r.conflicts}건 — 작업 목록에서 해결하세요`);
        }
      })();
    }
    lastOnline = nowOnline;
    shell?.refresh();
    // 모드 전환(로컬 ↔ 서버)은 레일 가시성과 스튜디오 여백을 바꾼다 — 라우트 재적용
    applyRoute(router.current());
  });

  // ── 부트 (연결 판정 → 인증 판정 → 셸 마운트 → 초기 라우트) ────────

  void (async (): Promise<void> => {
    const state = await client.start();
    if (disposed) return;
    if (state.mode === 'local') {
      // 서버 없음 — 기존 정적 배포/게이트 경로 그대로: 스튜디오가 첫 화면이다
      router.navigate({ name: 'studio' });
    } else {
      lastOnline = state.online;
      const session = client.getSession();
      let authed = false;
      if (session !== null) {
        const me = await client.me();
        if (me.kind === 'ok') {
          currentUser = me.data.user;
          authed = true;
        }
      }
      if (!authed) router.navigate({ name: 'login' });
    }
    if (disposed) return;
    shell = mountShell(document.body, {
      router,
      screens,
      user: () => currentUser,
      connection: {
        getState: conn,
        subscribe: (cb) => client.onStateChange(cb),
      },
      onSwitchUser: goSwitchUser,
    });
    applyRoute(router.current());
    updatePendingBadge();
    if (currentUser !== null) {
      void syncNow();
      void refreshConflictIds();
    }
  })();

  // ── 핸들 ──────────────────────────────────────────────────────────

  return {
    currentTaskInfo: () =>
      taskCtx === null
        ? null
        : {
            taskId: taskCtx.taskId,
            taskName: taskCtx.taskName,
            taskVersion: taskCtx.taskVersion,
            processId: taskCtx.processId,
          },
    operator: () =>
      currentUser === null
        ? { id: 'local-user', name: '로컬 사용자' }
        : { id: currentUser.id, name: currentUser.name },
    submitRun: (record) => {
      void (async (): Promise<void> => {
        if (isServerOnline()) {
          const r = await api.runs.create(record);
          if (r.kind === 'ok') {
            screenHandles.get('runs')?.refresh();
            return;
          }
          if (r.kind === 'error' && r.status < 500) {
            // 4xx 스키마 거부만 폐기 — 재시도해도 같다
            appLog('warn', `실행 기록 저장 거부: ${r.messageKo}`);
            return;
          }
          // network / unauthorized(세션 만료) / 5xx — 재시도 가능. 대기열에 보존한다.
        }
        await queuePendingRun(record);
        updatePendingBadge();
      })();
    },
    saveActive: async () => {
      if (taskCtx === null && processCtx === null) return false;
      // in-flight 가드 — Ctrl+S 연타가 같은 baseVersion으로 요청 2개를 만들면 두 번째가
      // **자기 자신과 409**를 일으켜 존재하지 않는 동시 편집자를 상대하게 된다.
      if (saving) return true;
      saving = true;
      try {
        if (taskCtx !== null) return await saveTaskCtx();
        return await saveProcessCtx();
      } finally {
        saving = false;
      }
    },
    hasSaveContext: () => taskCtx !== null || processCtx !== null,
    clearDocumentContext: () => {
      releaseTaskLock();
      taskCtx = null;
      processCtx = null;
    },
    execDefaults: loadExecDefaults,
    openBlockCapture,
    libraryBlocks: async () => {
      if (!isServerOnline() && conn().mode === 'local') return [];
      const r = await cachedBlocks.list();
      if (r.kind !== 'ok') return [];
      const summaries: LibraryBlockSummary[] = [];
      for (const item of r.items) {
        const detail = await cachedBlocks.get(item.id);
        if (detail.kind === 'ok') {
          summaries.push({
            id: item.id,
            name: item.name,
            stepCount: detail.record.doc.steps.length,
            robotHint: detail.record.doc.robotHint,
          });
        }
      }
      return summaries;
    },
    onLibraryInsertBlock: insertBlockById,
    dispose: () => {
      disposed = true;
      releaseTaskLock();
      offRoute();
      offState();
      login?.dispose();
      shell?.dispose();
      router.dispose();
      toasts.dispose();
      toastLayer.remove();
      loginLayer.remove();
      importFileInput.remove();
      client.dispose();
    },
  };
}
