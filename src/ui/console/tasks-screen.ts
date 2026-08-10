// ui/console/tasks-screen.ts — ② 작업 목록 화면: 콘솔 평면의 홈 (docs/BACKEND.md)
//
// 설치기사가 로그인 후 처음 보는 화면. 목표는 하나다 — **"내가 만들던 것을 3초 안에
// 다시 연다."** 그래서 목록은 수정 내림차순 고정이고(마지막 작업이 항상 맨 위),
// 검색·필터는 클라이언트에서 즉시 돌며(서버 왕복 없음 — 오프라인 캐시 서빙과도 맞물린다),
// 모든 행동 버튼은 터치 타깃 ≥44px(장갑 전제)다.
//
// ── 구성 (임무 스펙) ────────────────────────────────────────────────
// - 상단: 제목 '작업' + 검색 + 공정 필터 + 밀도 토글(카드↔표, 20건 초과 시 표 기본,
//   localStorage 기억) + [＋ 새 작업](primary)
// - 카드: 썸네일(로더 주입 시)·이름·공정명·노드 수·수정 상대시각·마지막 실행 배지
// - 표: 이름 | 공정 | 노드 수 | 수정 | 마지막 실행 | 행동(더보기 메뉴: 열기·복제·삭제)
// - 잠금: 다른 사용자가 편집 중이면 lock 아이콘 + '{이름} 편집 중' 배지 (내 잠금 제외)
// - 충돌: outbox 동기화 충돌 id 목록에 있으면 warn 배지, 클릭 시 onResolveConflict
// - 삭제 = soft: 액션 토스트 '삭제됨 — 실행취소'(restore). 휴지통 보기 + 복원.
//
// ── 계층/관례 (CLAUDE.md §3·§4-b, primitives.ts 헤더) ───────────────
// core/planner/render를 import하지 않는다. deps는 좁은 인터페이스(리소스 + 콜백)만 —
// 배선은 통합자 몫이고 main.ts는 손대지 않는다. 시각 토큰은 ui/theme.ts, 아이콘은
// ui/icons.ts, 조립 블록은 ./primitives만 소비한다. UI 크롬은 한국어, 도메인 식별자
// (엔티티 id 폴백 표기)는 영문 원문 + lang="en".
// 전역 window keydown 없음 — 더보기 메뉴의 Escape/방향키는 메뉴 요소 로컬 리스너이고,
// 바깥 클릭 닫기는 메뉴가 열려 있는 동안만 document pointerdown(캡처)을 걸었다 뗀다.
//
// 서버 미연결(local/offline)에서도 깨지지 않는다: 목록 실패는 빈 상태 + 다시 시도로
// 수렴하고, 서버 쓰기(복제·삭제·복원)는 **사유가 title에 적힌 회색 버튼**이 된다.
//
// 모듈 top-level에서 DOM을 만지지 않는다 — 순수 헬퍼(밀도 결정·필터·정렬·잠금 표시·
// 사본 이름)는 node 환경 테스트가 import해도 안전하다(tasks-screen.test.ts).

import type { ConnectionState, GetResult, ListResult, LockResult, RemoveResult, SaveResult } from '../../api';
import type { EntityMeta, LockInfo, RunResult, TaskDoc } from '../../schema/entities';
import { describeAge } from '../document';
import type { ToastHandle } from '../feedback/toast';
import { makeIconButton } from '../icons';
import type { IconName } from '../icons';
import {
  BORDER,
  BORDER_WIDTH,
  COLOR,
  ICON,
  RADIUS,
  SHADOW,
  SPACE,
  SURFACE,
  TYPE,
  Z_INDEX,
  applyType,
  ensureThemeStyles,
  makeButton,
  styled,
} from '../theme';
import type { StatusName } from '../theme';
import {
  applyTouchTarget,
  ensureConsoleStyles,
  filterRowsByQuery,
  makeBadge,
  makeCard,
  makeCardGrid,
  makeDataTable,
  makeEmptyState,
  makeRowComparator,
  makeSearchField,
} from './primitives';
import type { CardSubline } from './primitives';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4) ─────────────────────────────

/** 밀도 선택 기억 키 — localStorage (brand STORAGE_PREFIX 계열의 콘솔 평면 키) */
export const DENSITY_STORAGE_KEY = 'workcell.tasksScreen.density';
/** 이 건수를 **초과**하면 기본 밀도가 표(list)가 된다 (임무 명세: 20건 초과 시 표) */
export const DENSITY_TABLE_THRESHOLD = 20;
/** 공정 필터 특수값 — entityIdSchema는 min 8자라 실제 id와 충돌하지 않는다 */
export const PROCESS_FILTER_ALL = 'all';
export const PROCESS_FILTER_NONE = 'none';
/** displayNameSchema의 이름 길이 상한 (schema/entities.ts와 짝) */
export const TASK_NAME_MAX_LEN = 80;

const DUPLICATE_SUFFIX_KO = ' (사본)';
const TRASH_HINT_KO = '삭제된 작업은 30일 동안 보관됩니다';

/** 검색 필드 레이아웃 (flex-basis / min / max) */
const SEARCH_BASIS_PX = 240;
const SEARCH_MIN_WIDTH_PX = 200;
const SEARCH_MAX_WIDTH_PX = 420;
/** 공정 필터 select 최소 폭 */
const FILTER_MIN_WIDTH_PX = 150;
/** 더보기 메뉴 치수/여백 */
const MENU_MIN_WIDTH_PX = 190;
const MENU_GAP_PX = 4;
const MENU_VIEWPORT_MARGIN_PX = 8;
/** 표 컬럼 폭 */
const COL_STEPS_WIDTH = '72px';
const COL_UPDATED_WIDTH = '132px';
const COL_LAST_RUN_WIDTH = '116px';
const COL_ACTIONS_WIDTH = '96px';

// ── 순수 헬퍼 (DOM 비의존 — node 테스트 대상) ───────────────────────

export type TasksDensity = 'grid' | 'list';

/**
 * 밀도 결정 — 사용자가 고른 값(localStorage)이 있으면 그것이 이긴다.
 * 없으면 건수 기준: 20건 초과 → 표(스캔 효율), 이하 → 카드(재인식 효율).
 */
export function decideDensity(stored: string | null, taskCount: number): TasksDensity {
  if (stored === 'grid' || stored === 'list') return stored;
  return taskCount > DENSITY_TABLE_THRESHOLD ? 'list' : 'grid';
}

/** 공정 필터 매처 — 'all' | 'none'(무소속) | 공정 id */
export function matchesProcessFilter(
  row: { readonly processId: string | null },
  filter: string,
): boolean {
  if (filter === PROCESS_FILTER_ALL) return true;
  if (filter === PROCESS_FILTER_NONE) return row.processId === null;
  return row.processId === filter;
}

/**
 * 잠금 → "누가 편집 중인가" 표시값. 내 잠금(다른 탭)과 만료된 잠금은 null이다 —
 * TTL 90초가 지난 잠금을 계속 보여 주면 "아무도 없는데 잠겨 있다"가 된다.
 * 만료 시각이 파싱 불가면 **표시하는 쪽으로**(fail-open) — 감춰서 동시 편집을
 * 유도하는 것보다 낫다.
 */
export function activeLockHolder(
  lock: LockInfo | null | undefined,
  currentUserId: string | null,
  nowMs: number,
): string | null {
  if (lock === null || lock === undefined) return null;
  if (currentUserId !== null && lock.userId === currentUserId) return null;
  const expMs = Date.parse(lock.expiresAtIso);
  if (!Number.isNaN(expMs) && expMs <= nowMs) return null;
  return lock.userName;
}

export interface RunBadgeInfo {
  readonly labelKo: string;
  readonly status: StatusName;
}

/** 마지막 실행 결과 → 배지 (STATUS 축 — runResultSchema 4종 전부 커버) */
export const RUN_RESULT_BADGE: Readonly<Record<RunResult, RunBadgeInfo>> = {
  completed: { labelKo: '완료', status: 'success' },
  stopped: { labelKo: '정지', status: 'idle' },
  error: { labelKo: '오류', status: 'error' },
  autoPaused: { labelKo: '자동 정지', status: 'warn' },
};

/** 목록 행 뷰모델 — EntityMeta + 잠금/충돌/공정명을 화면이 쓰는 형태로 접은 것 */
export interface TaskRowVM {
  readonly id: string;
  readonly name: string;
  readonly processId: string | null;
  /** 공정 이름 (목록에 없으면 null — 표시는 id로 폴백, lang="en") */
  readonly processName: string | null;
  readonly stepCount: number | null;
  readonly hasThumbnail: boolean;
  readonly updatedAtIso: string;
  readonly updatedByName: string;
  readonly lastRun: { readonly atIso: string; readonly result: RunResult } | null;
  readonly deletedAtIso: string | null;
  /** 다른 사용자가 편집 중이면 그 이름 (activeLockHolder) */
  readonly lockHolder: string | null;
  /** 동기화 충돌(outbox) 있음 */
  readonly conflict: boolean;
}

export interface TaskRowContext {
  readonly processNames: ReadonlyMap<string, string>;
  readonly locks: ReadonlyMap<string, LockInfo | null>;
  readonly conflictIds: ReadonlySet<string>;
  readonly currentUserId: string | null;
  readonly nowMs: number;
}

/** 서버 목록(EntityMeta) → 행 뷰모델 (순수 — 정렬은 visibleTaskRows 몫) */
export function buildTaskRows(items: readonly EntityMeta[], ctx: TaskRowContext): TaskRowVM[] {
  return items.map((item) => {
    const summary = item.taskSummary;
    return {
      id: item.id,
      name: item.name,
      processId: item.processId,
      processName:
        item.processId !== null ? (ctx.processNames.get(item.processId) ?? null) : null,
      stepCount: summary !== null ? summary.stepCount : null,
      hasThumbnail: summary !== null && summary.hasThumbnail,
      updatedAtIso: item.meta.updatedAtIso,
      updatedByName: item.meta.updatedByName,
      lastRun: summary !== null ? summary.lastRun : null,
      deletedAtIso: item.meta.deletedAtIso,
      lockHolder: activeLockHolder(ctx.locks.get(item.id) ?? null, ctx.currentUserId, ctx.nowMs),
      conflict: ctx.conflictIds.has(item.id),
    };
  });
}

/** 정렬 키 — 수정 시각(ms). 파싱 불가면 null(비교기가 방향 무관 마지막으로 보낸다) */
export function updatedAtSortKey(iso: string): number | null {
  const ms = Date.parse(iso);
  return Number.isNaN(ms) ? null : ms;
}

/** 검색 대상 문자열 — 이름·공정명·공정 id 어느 쪽으로도 걸린다 */
export function taskSearchText(row: TaskRowVM): string {
  return `${row.name} ${row.processName ?? ''} ${row.processId ?? ''}`;
}

export interface TasksViewState {
  readonly query: string;
  readonly processFilter: string;
  /** true면 휴지통(삭제된 행만), false면 일반(삭제 안 된 행만) */
  readonly trash: boolean;
}

/**
 * 화면에 실제로 그릴 행: 휴지통 분리 → 공정 필터 → 검색 → 수정 내림차순.
 * "마지막으로 만지던 작업이 항상 맨 위" — 3초 안에 다시 열기의 전제다.
 */
export function visibleTaskRows(rows: readonly TaskRowVM[], view: TasksViewState): TaskRowVM[] {
  const inView = rows.filter((r) => (r.deletedAtIso !== null) === view.trash);
  const byProcess = inView.filter((r) => matchesProcessFilter(r, view.processFilter));
  const byQuery = filterRowsByQuery(byProcess, view.query, taskSearchText);
  return byQuery.sort(makeRowComparator<TaskRowVM>((r) => updatedAtSortKey(r.updatedAtIso), 'desc'));
}

/** 복제 이름 — ' (사본)' 접미. 이름 상한(80자)을 넘지 않게 원본을 자른다. */
export function duplicateName(name: string): string {
  const budget = TASK_NAME_MAX_LEN - DUPLICATE_SUFFIX_KO.length;
  const base = name.length > budget ? name.slice(0, budget).trimEnd() : name;
  return `${base}${DUPLICATE_SUFFIX_KO}`;
}

/**
 * 서버 쓰기(복제·삭제·복원)가 지금 불가한 이유 — null이면 가능.
 * 회색 버튼에는 반드시 이 문구가 title로 붙는다(왜 안 되는지 말하지 않는 회색은 결함).
 */
export function unavailableReasonKo(conn: ConnectionState): string | null {
  if (conn.mode === 'local') return '로컬 모드 — 서버가 연결되면 사용할 수 있습니다';
  if (!conn.online) return '오프라인 — 서버에 다시 연결되면 사용할 수 있습니다';
  return null;
}

/** 개체 id 발급 — 클라이언트 uuid (BACKEND §4: 오프라인 생성 지원, 중복은 서버가 409) */
export function makeEntityId(): string {
  const c = globalThis.crypto;
  if (c !== undefined && typeof c.randomUUID === 'function') return c.randomUUID();
  // crypto 미지원 폴백 — entityIdSchema(min 8자)를 만족하는 유일-확률 id
  return `task-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

/** 상태 라인 문구 — 필터로 줄었으면 "N개 중 M개 표시"로 정직하게 말한다 */
export function taskCountLabelKo(
  visibleCount: number,
  datasetCount: number,
  trash: boolean,
): string {
  const noun = trash ? '휴지통' : '작업';
  if (visibleCount === datasetCount) return `${noun} ${datasetCount}개`;
  return `${noun} ${datasetCount}개 중 ${visibleCount}개 표시`;
}

// ── deps (좁은 인터페이스 — 배선은 통합자 몫) ───────────────────────

/** tasks 리소스 표면 — api/resources.ts EntityClient<'tasks'>가 구조적으로 만족한다 */
export interface TasksResource {
  list(opts?: {
    readonly q?: string;
    readonly includeDeleted?: boolean;
    readonly processId?: string;
  }): Promise<ListResult>;
  get(id: string): Promise<GetResult<TaskDoc>>;
  create(doc: TaskDoc): Promise<SaveResult<TaskDoc>>;
  remove(id: string): Promise<RemoveResult>;
  restore(id: string): Promise<GetResult<TaskDoc>>;
}

/** processes 리소스 표면 — 이름 매핑용 목록만 쓴다 */
export interface ProcessesResource {
  list(opts?: { readonly q?: string }): Promise<ListResult>;
}

export interface TasksScreenDeps {
  readonly tasks: TasksResource;
  readonly processes: ProcessesResource;
  /** 작업 편집 잠금 조회 (표시 전용 — 획득하지 않는다). GET /locks/task/:id 배선. */
  locks(taskId: string): Promise<LockResult>;
  /** 동기화 충돌(outbox conflicts)이 있는 작업 id 목록 */
  conflicts(): readonly string[];
  onOpenTask(id: string): void;
  onNewTask(): void;
  onNewFromSample(): void;
  onResolveConflict(id: string): void;
  /** 연결 상태 getter — 화면은 refresh/렌더 시점에 읽는다 (변화 시 통합자가 refresh 호출) */
  connection(): ConnectionState;
  /** 액션(실행취소) 토스트 표면 — ui/feedback/toast.ts의 mountToasts 산출물 */
  readonly toast: Pick<ToastHandle, 'show'>;
  /**
   * 현재 로그인 사용자 id getter — 내 잠금은 "편집 중"으로 표시하지 않는다 (null = 미로그인).
   * 값 스냅샷이 아니라 getter인 이유: 화면은 셸이 1회만 마운트하고 이후 refresh()만
   * 부르므로, 로그인 **후**의 사용자로 판정하려면 렌더 시점마다 다시 읽어야 한다.
   */
  currentUserId(): string | null;
  /**
   * 카드 썸네일 로더 (선택) — 목록 API(EntityMeta)에는 data URI가 없다.
   * 없으면 카드는 썸네일 없이 그려진다(밀도 우선 — primitives makeCard 규약).
   */
  getThumbnail?(id: string): Promise<string | null>;
  /** 시각 주입 (테스트) — 기본 Date.now */
  nowMs?(): number;
  /** 밀도 기억 저장소 주입 (테스트) — 기본 localStorage */
  readonly densityStorage?: Pick<Storage, 'getItem' | 'setItem'>;
}

export interface TasksScreenHandle {
  /** 목록·잠금·충돌을 다시 불러온다 (연결 상태 변화·충돌 해결 후 통합자가 부른다) */
  refresh(): void;
  dispose(): void;
}

// ── 마운트 ──────────────────────────────────────────────────────────

export function mountTasksScreen(host: HTMLElement, deps: TasksScreenDeps): TasksScreenHandle {
  ensureThemeStyles();
  ensureConsoleStyles();

  const now = (): number => (deps.nowMs !== undefined ? deps.nowMs() : Date.now());
  const storage: Pick<Storage, 'getItem' | 'setItem'> | null =
    deps.densityStorage ?? (typeof localStorage === 'undefined' ? null : localStorage);

  // ── 상태 ──
  let disposed = false;
  let generation = 0;
  let loading = false;
  let lastErrorKo: string | null = null;
  let items: TaskRowVM[] = [];
  let processOptions: { readonly id: string; readonly name: string }[] = [];
  let query = '';
  let processFilter: string = PROCESS_FILTER_ALL;
  let trashMode = false;
  let offlineReason: string | null = null;
  let storedDensity: string | null = null;
  try {
    storedDensity = storage === null ? null : storage.getItem(DENSITY_STORAGE_KEY);
  } catch {
    storedDensity = null; // 저장소 접근 불가(프라이빗 모드 등) — 건수 기본값으로
  }
  /** 썸네일 캐시 — null은 "없음 확인됨"(재요청 방지) */
  const thumbCache = new Map<string, string | null>();
  const cardEls = new Map<string, HTMLElement>();

  // ── 뼈대 ──
  const root = styled(document.createElement('section'), {
    display: 'flex',
    flexDirection: 'column',
    gap: SPACE.lg,
    padding: SPACE.xl,
    height: '100%',
    boxSizing: 'border-box',
    minHeight: '0',
  });
  root.dataset.testid = 'tasks-screen';
  root.setAttribute('aria-label', '작업 목록');
  host.appendChild(root);

  // ── 제목 행 ──
  const titleRow = styled(document.createElement('div'), {
    display: 'flex',
    alignItems: 'center',
    gap: SPACE.lg,
    flexWrap: 'wrap',
  });
  const titleEl = applyType(document.createElement('h2'), TYPE.display);
  styled(titleEl, { margin: '0', color: COLOR.textStrong });
  titleEl.textContent = '작업';
  const statusEl = applyType(document.createElement('span'), TYPE.caption);
  styled(statusEl, { color: COLOR.muted });
  statusEl.setAttribute('role', 'status');
  statusEl.dataset.testid = 'tasks-count';
  const connHost = styled(document.createElement('span'), { display: 'inline-flex' });
  const titleSpacer = styled(document.createElement('div'), { flex: '1 1 auto' });
  const newButton = makeIconButton('plus', '새 작업', '새 작업 만들기', 'tasks-new', 'primary', ICON.lg);
  applyTouchTarget(newButton);
  applyType(newButton, TYPE.subhead);
  styled(newButton, { paddingLeft: SPACE.xl, paddingRight: SPACE.xl });
  newButton.addEventListener('click', () => deps.onNewTask());
  titleRow.append(titleEl, statusEl, connHost, titleSpacer, newButton);
  root.appendChild(titleRow);

  // ── 도구 행 ──
  const toolbar = styled(document.createElement('div'), {
    display: 'flex',
    alignItems: 'center',
    gap: SPACE.md,
    flexWrap: 'wrap',
  });
  const search = makeSearchField({
    placeholderKo: '작업 검색 (이름·공정)',
    testid: 'tasks-search',
    onInput: (q) => {
      query = q;
      renderBody();
    },
  });
  styled(search.el, {
    flex: `1 1 ${SEARCH_BASIS_PX}px`,
    minWidth: `${SEARCH_MIN_WIDTH_PX}px`,
    maxWidth: `${SEARCH_MAX_WIDTH_PX}px`,
  });

  const processSelect = document.createElement('select');
  processSelect.className = 'ui-select';
  processSelect.dataset.testid = 'tasks-process-filter';
  processSelect.setAttribute('aria-label', '공정 필터');
  applyTouchTarget(processSelect);
  styled(processSelect, { minWidth: `${FILTER_MIN_WIDTH_PX}px` });
  processSelect.addEventListener('change', () => {
    processFilter = processSelect.value;
    renderBody();
  });

  const densityGroup = styled(document.createElement('div'), {
    display: 'inline-flex',
    gap: SPACE.xs,
  });
  densityGroup.setAttribute('role', 'group');
  densityGroup.setAttribute('aria-label', '보기 밀도');
  const gridButton = makeIconButton('grid', '', '카드 보기', 'tasks-density-grid', 'ghost', ICON.lg);
  const listButton = makeIconButton('list', '', '표 보기', 'tasks-density-list', 'ghost', ICON.lg);
  applyTouchTarget(gridButton, { square: true });
  applyTouchTarget(listButton, { square: true });
  gridButton.addEventListener('click', () => setDensity('grid'));
  listButton.addEventListener('click', () => setDensity('list'));
  densityGroup.append(gridButton, listButton);

  const trashButton = makeIconButton(
    'trash',
    '휴지통',
    `휴지통 보기 — ${TRASH_HINT_KO}`,
    'tasks-trash-toggle',
    'ghost',
  );
  applyTouchTarget(trashButton);
  trashButton.setAttribute('aria-pressed', 'false');
  trashButton.addEventListener('click', () => setTrash(!trashMode));

  toolbar.append(search.el, processSelect, densityGroup, trashButton);
  root.appendChild(toolbar);

  // ── 본문 (표 / 카드 / 빈 상태) ──
  const bodyHost = styled(document.createElement('div'), {
    flex: '1 1 auto',
    minHeight: '0',
    overflowY: 'auto',
  });
  bodyHost.className = 'ui-scroll';
  const tableEmptyHost = document.createElement('div');
  const gridEmptyHost = document.createElement('div');
  const grid = makeCardGrid({ testid: 'tasks-card-grid' });

  const table = makeDataTable<TaskRowVM>({
    columns: [
      { key: 'name', labelKo: '이름', render: renderNameCell },
      { key: 'process', labelKo: '공정', render: renderProcessCell },
      { key: 'steps', labelKo: '노드 수', width: COL_STEPS_WIDTH, render: renderStepsCell },
      { key: 'updated', labelKo: '수정', width: COL_UPDATED_WIDTH, render: renderUpdatedCell },
      { key: 'lastRun', labelKo: '마지막 실행', width: COL_LAST_RUN_WIDTH, render: renderLastRunCell },
      { key: 'actions', labelKo: '행동', width: COL_ACTIONS_WIDTH, render: renderActionsCell },
    ],
    rows: [],
    onRowClick: (row) => openRow(row),
    rowTestid: (row) => `task-row-${row.id}`,
    emptyState: tableEmptyHost,
    ariaLabelKo: '작업 목록',
  });
  bodyHost.append(table.el, grid.el, gridEmptyHost);
  root.appendChild(bodyHost);

  // ── 셀 렌더러 ──

  function renderNameCell(row: TaskRowVM): HTMLElement {
    const wrap = styled(document.createElement('div'), {
      display: 'flex',
      flexDirection: 'column',
      gap: SPACE.xs,
      minWidth: '0',
    });
    const nameEl = applyType(document.createElement('span'), TYPE.bodyStrong);
    styled(nameEl, {
      color: COLOR.textStrong,
      overflow: 'hidden',
      textOverflow: 'ellipsis',
      whiteSpace: 'nowrap',
    });
    nameEl.textContent = row.name;
    wrap.appendChild(nameEl);
    const badges = taskBadges(row);
    if (badges.childElementCount > 0) wrap.appendChild(badges);
    return wrap;
  }

  function renderProcessCell(row: TaskRowVM): HTMLElement {
    const span = document.createElement('span');
    if (row.processId === null) {
      span.textContent = '무소속';
      styled(span, { color: COLOR.muted });
    } else if (row.processName !== null) {
      span.textContent = row.processName;
    } else {
      // 공정 목록에 없는 id — 도메인 식별자 원문 그대로 (lang="en", §4-b)
      applyType(span, TYPE.monoBody);
      styled(span, { color: COLOR.muted });
      span.setAttribute('lang', 'en');
      span.textContent = row.processId;
    }
    return span;
  }

  function renderStepsCell(row: TaskRowVM): HTMLElement {
    const span = applyType(document.createElement('span'), TYPE.monoReadout);
    if (row.stepCount === null) {
      span.textContent = '—';
      styled(span, { color: COLOR.muted });
    } else {
      span.textContent = String(row.stepCount);
    }
    return span;
  }

  function renderUpdatedCell(row: TaskRowVM): HTMLElement {
    const span = document.createElement('span');
    span.textContent = describeAge(row.updatedAtIso, now());
    span.title = `${fullTimeKo(row.updatedAtIso)} · ${row.updatedByName}`;
    return span;
  }

  function renderLastRunCell(row: TaskRowVM): HTMLElement {
    if (row.lastRun === null) {
      const span = styled(document.createElement('span'), { color: COLOR.muted });
      span.textContent = '—';
      return span;
    }
    const info = RUN_RESULT_BADGE[row.lastRun.result];
    const badge = makeBadge(info.labelKo, info.status);
    badge.title = `마지막 실행 ${describeAge(row.lastRun.atIso, now())}`;
    return badge;
  }

  function renderActionsCell(row: TaskRowVM): HTMLElement {
    const cell = styled(document.createElement('div'), {
      display: 'flex',
      gap: SPACE.xs,
      justifyContent: 'flex-end',
    });
    // 행 클릭(열기)·roving 활성화로 새지 않게 차단
    cell.addEventListener('click', (e) => e.stopPropagation());
    cell.addEventListener('pointerdown', (e) => e.stopPropagation());
    if (row.deletedAtIso !== null) {
      cell.appendChild(restoreButton(row));
    } else {
      cell.appendChild(moreButton(row));
    }
    return cell;
  }

  // ── 공용 조각 ──

  function fullTimeKo(iso: string): string {
    const ms = Date.parse(iso);
    return Number.isNaN(ms) ? iso : new Date(ms).toLocaleString('ko-KR');
  }

  /** 이름 옆 배지 묶음 — 휴지통 · 동기화 충돌(클릭 가능) · 편집 중 잠금 */
  function taskBadges(row: TaskRowVM): HTMLElement {
    const wrap = styled(document.createElement('span'), {
      display: 'inline-flex',
      alignItems: 'center',
      gap: SPACE.xs,
      flexWrap: 'wrap',
    });
    if (row.deletedAtIso !== null) {
      const badge = makeBadge('휴지통', 'idle', { iconName: 'trash' });
      badge.title = `삭제 ${describeAge(row.deletedAtIso, now())} — ${TRASH_HINT_KO}`;
      wrap.appendChild(badge);
    }
    if (row.conflict) wrap.appendChild(conflictBadgeButton(row));
    if (row.lockHolder !== null) {
      const badge = makeBadge(`${row.lockHolder} 편집 중`, 'running', {
        iconName: 'lock',
        testid: `task-lock-${row.id}`,
      });
      badge.title = '다른 사용자가 편집 중입니다 — 잠금이 풀리면 편집할 수 있습니다';
      wrap.appendChild(badge);
    }
    return wrap;
  }

  /** 동기화 충돌 배지 — 클릭하면 해결 다이얼로그 콜백 (배지 자체가 행동 진입점) */
  function conflictBadgeButton(row: TaskRowVM): HTMLElement {
    const btn = document.createElement('button');
    btn.type = 'button';
    styled(btn, {
      background: 'transparent',
      border: 'none',
      padding: '0',
      margin: '0',
      cursor: 'pointer',
      display: 'inline-flex',
      alignItems: 'center',
    });
    applyTouchTarget(btn);
    btn.dataset.testid = `task-conflict-${row.id}`;
    const label = `'${row.name}' 동기화 충돌 해결`;
    btn.title = label;
    btn.setAttribute('aria-label', label);
    btn.appendChild(makeBadge('동기화 충돌', 'warn', { iconName: 'alert' }));
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      deps.onResolveConflict(row.id);
    });
    return btn;
  }

  function restoreButton(row: TaskRowVM): HTMLButtonElement {
    const btn = makeIconButton('undo', '복원', `'${row.name}' 복원`, `task-restore-${row.id}`);
    applyTouchTarget(btn);
    if (offlineReason !== null) {
      btn.disabled = true;
      btn.title = `복원 — ${offlineReason}`;
      btn.setAttribute('aria-label', btn.title);
    } else {
      btn.addEventListener('click', () => {
        void restoreTask(row);
      });
    }
    return btn;
  }

  function moreButton(row: TaskRowVM): HTMLButtonElement {
    const btn = makeIconButton('more', '', `'${row.name}' 행동 메뉴`, `task-more-${row.id}`, 'ghost', ICON.lg);
    applyTouchTarget(btn, { square: true });
    btn.setAttribute('aria-haspopup', 'menu');
    btn.addEventListener('click', () => openRowMenu(btn, row));
    return btn;
  }

  function openRow(row: TaskRowVM): void {
    if (row.deletedAtIso !== null) {
      deps.toast.show('info', '휴지통의 작업입니다', { detail: '복원한 뒤 열 수 있습니다.' });
      return;
    }
    deps.onOpenTask(row.id);
  }

  // ── 더보기 메뉴 (열기·복제·삭제 · 충돌 해결) ──

  interface RowMenuState {
    readonly root: HTMLElement;
    readonly anchor: HTMLElement;
    readonly onDocPointer: (e: Event) => void;
  }
  let rowMenu: RowMenuState | null = null;

  function closeRowMenu(restoreFocus: boolean): void {
    if (rowMenu === null) return;
    document.removeEventListener('pointerdown', rowMenu.onDocPointer, true);
    rowMenu.root.remove();
    const { anchor } = rowMenu;
    rowMenu = null;
    if (restoreFocus && anchor.isConnected) anchor.focus();
  }

  function menuItem(
    iconName: IconName,
    labelKo: string,
    testid: string,
    opts: { readonly danger?: boolean; readonly reason?: string | null },
    onPick: () => void,
  ): HTMLButtonElement {
    const btn = makeIconButton(iconName, labelKo, labelKo, testid, opts.danger === true ? 'danger' : 'ghost');
    btn.setAttribute('role', 'menuitem');
    applyTouchTarget(btn);
    styled(btn, { width: '100%', justifyContent: 'flex-start' });
    const reason = opts.reason ?? null;
    if (reason !== null) {
      btn.disabled = true;
      btn.title = `${labelKo} — ${reason}`;
      btn.setAttribute('aria-label', btn.title);
    } else {
      btn.addEventListener('click', () => {
        closeRowMenu(true);
        onPick();
      });
    }
    return btn;
  }

  function openRowMenu(anchor: HTMLElement, row: TaskRowVM): void {
    closeRowMenu(false);
    const menuRoot = styled(document.createElement('div'), {
      position: 'fixed',
      zIndex: Z_INDEX.modal,
      display: 'flex',
      flexDirection: 'column',
      gap: SPACE.xxs,
      minWidth: `${MENU_MIN_WIDTH_PX}px`,
      padding: SPACE.xs,
      background: SURFACE.overlay,
      border: `${BORDER_WIDTH.hair} solid ${BORDER.default}`,
      borderRadius: RADIUS.md,
      boxShadow: SHADOW.overlay,
      visibility: 'hidden',
    });
    menuRoot.setAttribute('role', 'menu');
    menuRoot.setAttribute('aria-label', `'${row.name}' 행동 메뉴`);
    menuRoot.dataset.testid = 'tasks-row-menu';

    const menuItems: HTMLButtonElement[] = [
      menuItem('folderOpen', '열기', 'tasks-menu-open', {}, () => deps.onOpenTask(row.id)),
      menuItem('copy', '복제', 'tasks-menu-duplicate', { reason: offlineReason }, () => {
        void duplicateTask(row);
      }),
    ];
    if (row.conflict) {
      menuItems.push(
        menuItem('alert', '동기화 충돌 해결', 'tasks-menu-conflict', {}, () =>
          deps.onResolveConflict(row.id),
        ),
      );
    }
    menuItems.push(
      menuItem('trash', '삭제', 'tasks-menu-remove', { danger: true, reason: offlineReason }, () => {
        void removeTask(row);
      }),
    );
    for (const item of menuItems) menuRoot.appendChild(item);

    // 메뉴 로컬 키 처리 — 전역 라우터 소관이 아닌 위젯 내부 키다 (shortcuts 규약의 위젯 소유권)
    menuRoot.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        e.stopPropagation();
        closeRowMenu(true);
        return;
      }
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault();
        const enabled = menuItems.filter((b) => !b.disabled);
        if (enabled.length === 0) return;
        const idx = enabled.findIndex((b) => b === document.activeElement);
        const delta = e.key === 'ArrowDown' ? 1 : -1;
        const next = enabled[(idx + delta + enabled.length) % enabled.length];
        next?.focus();
      } else if (e.key === 'Tab') {
        closeRowMenu(false);
      }
    });

    const onDocPointer = (e: Event): void => {
      const target = e.target;
      if (target instanceof Node && menuRoot.contains(target)) return;
      closeRowMenu(false);
    };
    document.addEventListener('pointerdown', onDocPointer, true);

    root.appendChild(menuRoot);
    const rect = anchor.getBoundingClientRect();
    const menuW = menuRoot.offsetWidth;
    const menuH = menuRoot.offsetHeight;
    let left = Math.min(rect.right - menuW, window.innerWidth - menuW - MENU_VIEWPORT_MARGIN_PX);
    left = Math.max(MENU_VIEWPORT_MARGIN_PX, left);
    let top = rect.bottom + MENU_GAP_PX;
    if (top + menuH > window.innerHeight - MENU_VIEWPORT_MARGIN_PX) {
      top = Math.max(MENU_VIEWPORT_MARGIN_PX, rect.top - menuH - MENU_GAP_PX);
    }
    menuRoot.style.left = `${left}px`;
    menuRoot.style.top = `${top}px`;
    menuRoot.style.visibility = 'visible';
    rowMenu = { root: menuRoot, anchor, onDocPointer };
    const first = menuItems.find((b) => !b.disabled) ?? menuItems[0];
    first?.focus();
  }

  // ── 카드 ──

  function makeTaskCard(row: TaskRowVM): HTMLElement {
    const badges = taskBadges(row);
    if (row.lastRun !== null) {
      const info = RUN_RESULT_BADGE[row.lastRun.result];
      const badge = makeBadge(info.labelKo, info.status);
      badge.title = `마지막 실행 ${describeAge(row.lastRun.atIso, now())}`;
      badges.appendChild(badge);
    }

    const sublines: CardSubline[] = [];
    if (row.processId === null) sublines.push('무소속');
    else if (row.processName !== null) sublines.push(row.processName);
    else sublines.push({ text: row.processId, lang: 'en' });
    sublines.push(`노드 ${row.stepCount ?? 0}개 · ${describeAge(row.updatedAtIso, now())} 수정`);

    const actions: HTMLElement[] = [];
    if (row.deletedAtIso !== null) actions.push(restoreButton(row));
    else actions.push(moreButton(row));

    const thumb = thumbCache.get(row.id);
    const card = makeCard({
      title: row.name,
      sublines,
      badge: badges.childElementCount > 0 ? badges : undefined,
      thumbnailDataUri: typeof thumb === 'string' ? thumb : undefined,
      onClick: () => openRow(row),
      actions,
      testid: `task-card-${row.id}`,
    });
    cardEls.set(row.id, card);
    return card;
  }

  function renderCards(visible: readonly TaskRowVM[]): void {
    grid.setCards(visible.map((row) => makeTaskCard(row)));
  }

  /** 썸네일 지연 로딩 — 목록 표시를 막지 않고, 도착한 카드만 제자리 교체한다 */
  function loadThumbnails(visible: readonly TaskRowVM[], gen: number): void {
    const getThumbnail = deps.getThumbnail;
    if (getThumbnail === undefined) return;
    for (const row of visible) {
      if (!row.hasThumbnail || row.deletedAtIso !== null || thumbCache.has(row.id)) continue;
      getThumbnail(row.id)
        .then((uri) => {
          if (disposed || gen !== generation) return;
          thumbCache.set(row.id, uri);
          if (typeof uri !== 'string') return;
          const current = cardEls.get(row.id);
          if (current === undefined || !current.isConnected) return;
          current.replaceWith(makeTaskCard(row));
        })
        .catch(() => {
          thumbCache.set(row.id, null);
        });
    }
  }

  // ── 서버 행동 (soft-delete · 복원 · 복제) ──

  async function removeTask(row: TaskRowVM): Promise<void> {
    const res = await deps.tasks.remove(row.id);
    if (disposed) return;
    if (res.kind === 'ok') {
      // 파괴적 동작 → 되돌릴 경로 (CLAUDE.md §2.11): 실행취소 = restore
      deps.toast.show('success', `'${row.name}' 삭제됨`, {
        detail: TRASH_HINT_KO,
        action: {
          label: '실행취소',
          onClick: (): void => {
            void restoreTask(row);
          },
        },
      });
      refresh();
    } else {
      deps.toast.show('error', '삭제하지 못했습니다', { detail: res.messageKo });
    }
  }

  async function restoreTask(row: TaskRowVM): Promise<void> {
    const res = await deps.tasks.restore(row.id);
    if (disposed) return;
    if (res.kind === 'ok') {
      deps.toast.show('success', `'${row.name}' 복원됨`);
      refresh();
    } else {
      deps.toast.show('error', '복원하지 못했습니다', { detail: res.messageKo });
    }
  }

  async function duplicateTask(row: TaskRowVM): Promise<void> {
    const got = await deps.tasks.get(row.id);
    if (disposed) return;
    if (got.kind !== 'ok') {
      deps.toast.show('error', '복제하지 못했습니다', { detail: got.messageKo });
      return;
    }
    const src = got.record.doc;
    const copy: TaskDoc = { ...src, id: makeEntityId(), name: duplicateName(src.name) };
    const created = await deps.tasks.create(copy);
    if (disposed) return;
    if (created.kind === 'ok') {
      deps.toast.show('success', `'${copy.name}' 만들어짐`);
      refresh();
    } else {
      deps.toast.show('error', '복제하지 못했습니다', { detail: created.messageKo });
    }
  }

  // ── 빈 상태 ──

  function renderEmptyInto(hostEl: HTMLElement, datasetCount: number): void {
    hostEl.textContent = '';
    let content: HTMLElement;
    if (loading) {
      content = makeEmptyState({
        iconName: 'sync',
        titleKo: '불러오는 중…',
        actions: [],
        testid: 'tasks-empty-loading',
      });
    } else if (lastErrorKo !== null) {
      const retry = makeIconButton('refresh', '다시 시도', '목록 다시 불러오기', 'tasks-retry');
      retry.addEventListener('click', () => refresh());
      content = makeEmptyState({
        iconName: 'cloudOff',
        titleKo: '목록을 불러오지 못했습니다',
        hintKo: lastErrorKo,
        actions: [retry],
        testid: 'tasks-empty-error',
      });
    } else if (trashMode) {
      const back = makeButton('휴지통 닫기', '휴지통 보기 종료', 'tasks-trash-close');
      back.addEventListener('click', () => setTrash(false));
      content = makeEmptyState({
        iconName: 'trash',
        titleKo: '휴지통이 비어 있습니다',
        hintKo: TRASH_HINT_KO,
        actions: [back],
        testid: 'tasks-empty-trash',
      });
    } else if (datasetCount === 0) {
      const create = makeIconButton('plus', '새 작업', '새 작업 만들기', 'tasks-empty-new', 'primary');
      create.addEventListener('click', () => deps.onNewTask());
      const sample = makeIconButton('wand', '예제에서 시작', '예제 작업으로 시작하기', 'tasks-empty-sample');
      sample.addEventListener('click', () => deps.onNewFromSample());
      content = makeEmptyState({
        iconName: 'clipboard',
        titleKo: '아직 작업이 없습니다',
        hintKo: '새 작업을 만들거나, 예제 작업으로 시작해 보세요.',
        actions: [create, sample],
        testid: 'tasks-empty-onboarding',
      });
    } else {
      const clear = makeButton('필터 지우기', '검색어와 공정 필터 초기화', 'tasks-clear-filter');
      clear.addEventListener('click', () => {
        query = '';
        search.setValue('', { silent: true });
        processFilter = PROCESS_FILTER_ALL;
        processSelect.value = PROCESS_FILTER_ALL;
        renderBody();
      });
      content = makeEmptyState({
        iconName: 'search',
        titleKo: '조건에 맞는 작업이 없습니다',
        hintKo: '검색어나 공정 필터를 바꿔 보세요.',
        actions: [clear],
        testid: 'tasks-empty-filtered',
      });
    }
    hostEl.appendChild(content);
  }

  // ── 페인트 ──

  function paintConnection(conn: ConnectionState): void {
    connHost.textContent = '';
    const reason = unavailableReasonKo(conn);
    if (reason === null) return;
    const badge = makeBadge(conn.mode === 'local' ? '로컬 모드' : '오프라인', 'warn', {
      iconName: 'cloudOff',
      testid: 'tasks-conn-badge',
    });
    badge.title = reason;
    connHost.appendChild(badge);
  }

  function paintTrashToggle(): void {
    trashButton.setAttribute('aria-pressed', String(trashMode));
    trashButton.classList.toggle('ui-btn--active', trashMode);
  }

  function paintDensityButtons(density: TasksDensity): void {
    const lockNote = trashMode ? ' (휴지통에서는 표 보기로 고정)' : '';
    gridButton.disabled = trashMode;
    listButton.disabled = trashMode;
    gridButton.title = `카드 보기${lockNote}`;
    gridButton.setAttribute('aria-label', gridButton.title);
    listButton.title = `표 보기${lockNote}`;
    listButton.setAttribute('aria-label', listButton.title);
    gridButton.classList.toggle('ui-btn--active', density === 'grid');
    gridButton.setAttribute('aria-pressed', String(density === 'grid'));
    listButton.classList.toggle('ui-btn--active', density === 'list');
    listButton.setAttribute('aria-pressed', String(density === 'list'));
  }

  function syncProcessOptions(): void {
    const prev = processFilter;
    processSelect.textContent = '';
    const addOption = (value: string, label: string): void => {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = label;
      processSelect.appendChild(option);
    };
    addOption(PROCESS_FILTER_ALL, '전체 공정');
    addOption(PROCESS_FILTER_NONE, '무소속');
    for (const p of processOptions) addOption(p.id, p.name);
    const valid = [PROCESS_FILTER_ALL, PROCESS_FILTER_NONE, ...processOptions.map((p) => p.id)];
    processFilter = valid.includes(prev) ? prev : PROCESS_FILTER_ALL;
    processSelect.value = processFilter;
  }

  function renderBody(): void {
    const conn = deps.connection();
    offlineReason = unavailableReasonKo(conn);
    paintConnection(conn);
    paintTrashToggle();

    const datasetCount = items.filter((r) => (r.deletedAtIso !== null) === trashMode).length;
    const visible = visibleTaskRows(items, { query, processFilter, trash: trashMode });
    // 휴지통은 표 고정 — 복원 버튼이 항상 같은 자리에 보인다
    const density: TasksDensity = trashMode ? 'list' : decideDensity(storedDensity, datasetCount);
    paintDensityButtons(density);
    statusEl.textContent = loading
      ? '불러오는 중…'
      : taskCountLabelKo(visible.length, datasetCount, trashMode);

    closeRowMenu(false); // 재렌더 시 메뉴 앵커가 교체된다
    cardEls.clear();

    if (density === 'list') {
      grid.el.style.display = 'none';
      gridEmptyHost.style.display = 'none';
      gridEmptyHost.textContent = '';
      table.el.style.display = '';
      table.setRows(visible);
      if (visible.length === 0) renderEmptyInto(tableEmptyHost, datasetCount);
      else tableEmptyHost.textContent = '';
    } else {
      table.el.style.display = 'none';
      tableEmptyHost.textContent = '';
      if (visible.length === 0) {
        grid.el.style.display = 'none';
        gridEmptyHost.style.display = '';
        renderEmptyInto(gridEmptyHost, datasetCount);
      } else {
        gridEmptyHost.style.display = 'none';
        gridEmptyHost.textContent = '';
        grid.el.style.display = '';
        renderCards(visible);
        loadThumbnails(visible, generation);
      }
    }
  }

  // ── 상태 전이 ──

  function setDensity(density: TasksDensity): void {
    storedDensity = density;
    try {
      storage?.setItem(DENSITY_STORAGE_KEY, density);
    } catch {
      // 저장 불가 — 세션 내에서만 유지
    }
    renderBody();
  }

  function setTrash(on: boolean): void {
    if (trashMode === on) return;
    trashMode = on;
    paintTrashToggle();
    refresh();
  }

  // ── 데이터 로드 ──

  async function doRefresh(): Promise<void> {
    const gen = ++generation;
    loading = true;
    lastErrorKo = null;
    renderBody();

    const conn = deps.connection();
    // 로컬 모드 — 서버 개체가 존재하지 않으므로 요청 자체를 보내지 않는다. 정적 배포에서
    // /api 경로가 SPA 폴백(200 HTML)에 맞으면 연결 상태 머신이 "서버 도달"로 오판할 수
    // 있다(BACKEND §1의 로컬 강등 계약 훼손). 빈 목록 + 헤더의 '로컬 모드' 배지로 수렴.
    if (conn.mode === 'local') {
      items = [];
      loading = false;
      lastErrorKo = null;
      renderBody();
      return;
    }
    const [taskRes, procRes] = await Promise.all([
      deps.tasks.list(trashMode ? { includeDeleted: true } : {}),
      deps.processes.list(),
    ]);
    if (disposed || gen !== generation) return;

    const processNames = new Map<string, string>();
    processOptions = [];
    if (procRes.kind === 'ok') {
      for (const p of procRes.items) {
        processNames.set(p.id, p.name);
        processOptions.push({ id: p.id, name: p.name });
      }
    }
    syncProcessOptions();

    if (taskRes.kind !== 'ok') {
      lastErrorKo = taskRes.messageKo;
      items = [];
      loading = false;
      renderBody();
      return;
    }

    // 잠금 조회 — 서버 온라인일 때만 (표시 전용, 실패는 "잠금 없음"으로 강등)
    const locksMap = new Map<string, LockInfo | null>();
    if (unavailableReasonKo(conn) === null && !trashMode) {
      const lockResults = await Promise.all(
        taskRes.items.map(async (item): Promise<readonly [string, LockInfo | null]> => {
          try {
            const lr = await deps.locks(item.id);
            return [item.id, lr.kind === 'ok' || lr.kind === 'held' ? lr.lock : null];
          } catch {
            return [item.id, null];
          }
        }),
      );
      if (disposed || gen !== generation) return;
      for (const [id, lock] of lockResults) locksMap.set(id, lock);
    }

    items = buildTaskRows(taskRes.items, {
      processNames,
      locks: locksMap,
      conflictIds: new Set(deps.conflicts()),
      currentUserId: deps.currentUserId(),
      nowMs: now(),
    });
    loading = false;
    renderBody();
  }

  function refresh(): void {
    void doRefresh();
  }

  // 첫 로드
  refresh();

  return {
    refresh,
    dispose: (): void => {
      disposed = true;
      generation += 1;
      closeRowMenu(false);
      search.dispose();
      table.dispose();
      root.remove();
    },
  };
}
