// ui/console/runs-screen.ts — 실행 기록 화면 (Phase 12+ 콘솔 평면, docs/BACKEND.md §4)
//
// 지난 실행(RunRecord)을 돌아보는 화면이다. 좌측은 실행 목록(작업명·결과 배지·시각·
// 소요·작업자), 우측은 선택한 실행의 상세: 결과 요약 헤더 + 개입 타임라인 + 충돌 목록.
// **킬러 기능은 "실패 지점 재현"** — 예기치 않은 충돌 행의 [이 노드부터 재현] 버튼이
// deps.onReplayFromNode(taskId, nodeId)를 부르고, 통합자가 이를 워크스페이스의
// runFromNode에 배선한다. 기록은 읽기 전용이다(runs는 append-only — 삭제 UI 없음).
//
// ── 통계 카드 (작업 필터 시) ────────────────────────────────────────
// runCount·성공률·평균 소요를 보여주되, **runCount < STATS_MIN_RUN_COUNT(20)이면
// 숨긴다** — 3번 실행에 "성공률 67%"는 정보가 아니라 노이즈이고, 설치기사가 작은
// 표본의 요동을 실제 악화로 오독하게 만든다.
//
// ── 배지 축 결정 ────────────────────────────────────────────────────
// autoPaused(예기치 않은 충돌 자동 정지)는 STATUS.error(=COLLISION 램프)를 쓴다 —
// STATUS.warn은 "충돌 아님"이 명시된 축이다(theme.ts). error 결과와는 라벨('충돌
// 정지' vs '오류')이 구분한다 — 색만으로 상태를 말하지 않는다(UX §9).
//
// ── 계층/모드 규칙 ──────────────────────────────────────────────────
// deps는 좁은 인터페이스(runs 목록 + stats + 콜백)만 받는다 — core/main import 금지.
// local 모드(서버 미설정)에서는 실행 기록이 존재하지 않으므로 목록 대신 사유를
// 정직하게 표시하고 새로 고침을 비활성한다(회색 버튼에는 title로 이유 — 공통 규약).
// 순수 헬퍼(배지 매핑·소요 포맷·통계 임계·개입 정렬)는 node 테스트 대상이다
// (runs-screen.test.ts — 모듈 top-level에서 DOM을 만지지 않는다).

import type { RunListResult, TaskStatsResult } from '../../api';
import type {
  RunCollision,
  RunIntervention,
  RunRecord,
  RunResult,
  TaskStats,
} from '../../schema/entities';
import { createAnnouncer } from '../a11y';
import { makeIconButton } from '../icons';
import {
  BORDER,
  BORDER_WIDTH,
  COLLISION,
  COLOR,
  RADIUS,
  SPACE,
  SURFACE,
  TYPE,
  applyType,
  ensureThemeStyles,
  styled,
} from '../theme';
import type { StatusName } from '../theme';
import {
  applyTouchTarget,
  ensureConsoleStyles,
  makeBadge,
  makeDataTable,
  makeEmptyState,
} from './primitives';
import type { BadgeStatus } from './primitives';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4) ─────────────────────────────

/** 목록 페이지 크기 — 서버 기본값(BACKEND §4 GET /runs limit=50)과 동일 */
export const RUNS_PAGE_LIMIT = 50;
/** 통계 카드 표시 하한 — 이 미만의 표본에서 성공률은 노이즈다 (임무 명세) */
export const STATS_MIN_RUN_COUNT = 20;
/** local 모드에서 비활성 버튼에 붙는 사유 (회색 버튼에는 반드시 title로 이유) */
export const MSG_RUNS_LOCAL_KO = '서버 미연결 — 실행 기록은 서버에 저장됩니다';

/** 좌측 목록 페인 폭 비율 */
const LIST_PANE_FLEX = '0 1 46%';
/** 타임라인 시각 컬럼 폭 — 초 단위 mono 리드아웃 정렬용 */
const TIMELINE_TIME_WIDTH_PX = 56;

// ── 순수 헬퍼 (DOM 비의존 — node 테스트 대상) ───────────────────────

export interface ResultBadgeSpec {
  readonly labelKo: string;
  readonly status: StatusName;
}

/**
 * 실행 결과 → 배지 매핑. autoPaused는 COLLISION 램프(STATUS.error)다 — warn은
 * "충돌 아님" 축이라 쓰지 않는다(파일 헤더). error와는 라벨이 구분한다.
 */
const RESULT_BADGE: Readonly<Record<RunResult, ResultBadgeSpec>> = {
  completed: { labelKo: '완주', status: 'success' },
  stopped: { labelKo: '정지됨', status: 'idle' },
  error: { labelKo: '오류', status: 'error' },
  autoPaused: { labelKo: '충돌 정지', status: 'error' },
};

export function resultBadgeSpec(result: RunResult): ResultBadgeSpec {
  return RESULT_BADGE[result];
}

/** 개입 종류 → 한국어 라벨 (UI 크롬은 한국어 — §4-b) */
const INTERVENTION_KIND_KO: Readonly<Record<RunIntervention['kind'], string>> = {
  play: '재생',
  pause: '일시정지',
  stop: '정지',
  stepNode: '한 노드 실행',
  runFromNode: '이 노드부터 실행',
  autoPause: '자동 정지',
};

export function interventionKindKo(kind: RunIntervention['kind']): string {
  return INTERVENTION_KIND_KO[kind];
}

/** 개입을 시뮬 시각 오름차순으로 정렬한 사본 (동시각은 기록 순서 유지 — 안정 정렬) */
export function sortInterventionsBySimTime(list: readonly RunIntervention[]): RunIntervention[] {
  return [...list].sort((a, b) => a.atSimSec - b.atSimSec);
}

/**
 * 소요 시간(초) → 한국어 표기. 10초 미만은 0.1초 단위, 60초 미만은 초, 그 위는
 * "m분 s초" / "h시간 m분"(0인 하위 단위는 생략). 음수/비수치는 '0초'로 방어.
 */
export function formatDurationKo(sec: number): string {
  const safe = Number.isFinite(sec) && sec > 0 ? sec : 0;
  if (safe < 10) {
    const tenth = Math.round(safe * 10) / 10;
    if (tenth < 10) return `${tenth}초`;
  }
  const whole = Math.round(safe);
  if (whole < 60) return `${whole}초`;
  const totalMin = Math.floor(whole / 60);
  const restSec = whole % 60;
  if (totalMin < 60) return restSec === 0 ? `${totalMin}분` : `${totalMin}분 ${restSec}초`;
  const hours = Math.floor(totalMin / 60);
  const restMin = totalMin % 60;
  return restMin === 0 ? `${hours}시간` : `${hours}시간 ${restMin}분`;
}

/** 시뮬 시각 리드아웃 — '12.3s' (도메인 표기라 영문 접미사, 표시 요소에 lang="en") */
export function formatSimClock(sec: number): string {
  const safe = Number.isFinite(sec) && sec > 0 ? sec : 0;
  return `${Math.round(safe * 10) / 10}s`;
}

/** ISO 시각 → 'YYYY-MM-DD HH:mm' (로컬 시간대). 파싱 불가면 원문 그대로(정보 손실 금지). */
export function formatDateTimeKo(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const pad = (n: number): string => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

/** 통계 카드 표시 여부 — 작은 표본의 성공률은 노이즈다 (파일 헤더) */
export function shouldShowStats(runCount: number): boolean {
  return runCount >= STATS_MIN_RUN_COUNT;
}

/** 성공률(%) — 표본 0이면 null (0%로 오독시키지 않는다) */
export function successRatePercent(runCount: number, successCount: number): number | null {
  if (runCount <= 0) return null;
  return Math.round((successCount / runCount) * 100);
}

export interface CollisionBadgeSpec {
  readonly labelKo: string;
  readonly status: BadgeStatus;
}

/** 충돌 분류 → 배지. 의도된 접촉은 상태 축 바깥(neutral) — 결함이 아니다. */
export function collisionBadgeSpec(
  classification: RunCollision['classification'],
): CollisionBadgeSpec {
  return classification === 'unexpected'
    ? { labelKo: '예기치 않은 충돌', status: 'error' }
    : { labelKo: '의도된 접촉', status: 'neutral' };
}

/**
 * 이 충돌 행에서 재현 가능한 노드 id — **예기치 않은** 충돌이고 발생 노드가 기록된
 * 경우에만. 의도된 접촉은 실패가 아니고, 노드 없는 충돌은 재현 지점이 없다.
 */
export function replayTarget(collision: RunCollision): string | null {
  return collision.classification === 'unexpected' ? collision.nodeId : null;
}

// ── deps / handle ───────────────────────────────────────────────────

export interface RunsScreenConnection {
  readonly mode: 'server' | 'local';
  readonly online: boolean;
}

export interface RunsListOpts {
  readonly taskId?: string;
  readonly limit?: number;
  readonly offset?: number;
}

/** RunsClient(src/api/resources.ts)의 목록 표면만 — 좁은 구조적 인터페이스 */
export interface RunsResourceLike {
  list(opts?: RunsListOpts): Promise<RunListResult>;
}

export interface RunsScreenDeps {
  readonly runs: RunsResourceLike;
  /** WorkcellApi.taskStats — 작업 필터가 걸렸을 때만 호출된다 */
  stats(taskId: string): Promise<TaskStatsResult>;
  /** [작업 열기] — 통합자가 작업 화면/워크스페이스로 배선 */
  onOpenTask(taskId: string): void;
  /** [이 노드부터 재현] — 통합자가 워크스페이스 runFromNode로 배선 (킬러 기능) */
  onReplayFromNode(taskId: string, nodeId: string): void;
  /** 연결 상태 — local 모드면 목록을 요청하지 않고 사유를 표시한다. 없으면 server로 간주. */
  getConnection?(): RunsScreenConnection;
  /** 딥링크 runs?taskId= — 초기 작업 필터 */
  readonly initialTaskId?: string | null;
}

export interface RunsScreenHandle {
  refresh(): void;
  /** 작업 필터 교체 (딥링크·작업 화면에서 진입). null이면 해제. 즉시 다시 불러온다. */
  setTaskFilter(taskId: string | null): void;
  dispose(): void;
}

// ── 마운트 ──────────────────────────────────────────────────────────

export function mountRunsScreen(host: HTMLElement, deps: RunsScreenDeps): RunsScreenHandle {
  ensureThemeStyles();
  ensureConsoleStyles();

  const root = styled(document.createElement('section'), {
    display: 'flex',
    flexDirection: 'column',
    gap: SPACE.lg,
    height: '100%',
    minHeight: '0',
    minWidth: '0',
    boxSizing: 'border-box',
  });
  root.dataset.testid = 'runs-screen';
  root.setAttribute('aria-label', '실행 기록');

  const announcer = createAnnouncer(root);

  // ── 상태 ──
  let taskFilter: string | null = deps.initialTaskId ?? null;
  let rows: RunRecord[] = [];
  let selectedId: string | null = null;
  let requestSeq = 0;
  let disposed = false;

  const connection = (): RunsScreenConnection =>
    deps.getConnection?.() ?? { mode: 'server', online: true };

  // ── 헤더 행 ──
  const headerRow = styled(document.createElement('div'), {
    display: 'flex',
    alignItems: 'center',
    gap: SPACE.md,
    flex: 'none',
  });
  const heading = applyType(document.createElement('h2'), TYPE.title);
  styled(heading, { margin: '0', flex: '1 1 auto', minWidth: '0', color: COLOR.textStrong });
  heading.textContent = '실행 기록';
  headerRow.appendChild(heading);

  const REFRESH_TITLE = '실행 기록 새로 고침';
  const refreshButton = makeIconButton('refresh', '새로 고침', REFRESH_TITLE, 'runs-refresh');
  applyTouchTarget(refreshButton);
  refreshButton.addEventListener('click', () => {
    void refresh();
  });
  headerRow.appendChild(refreshButton);
  root.appendChild(headerRow);

  // ── 작업 필터 행 (필터 걸렸을 때만 표시) ──
  const filterRow = styled(document.createElement('div'), {
    display: 'none',
    alignItems: 'center',
    gap: SPACE.md,
    flex: 'none',
  });
  filterRow.dataset.testid = 'runs-filter';
  const filterLabel = applyType(document.createElement('span'), TYPE.caption);
  styled(filterLabel, { color: COLOR.label });
  filterLabel.textContent = '작업 필터';
  filterRow.appendChild(filterLabel);

  const filterValue = applyType(document.createElement('span'), TYPE.bodyStrong);
  styled(filterValue, {
    color: COLOR.textStrong,
    minWidth: '0',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  });
  filterValue.dataset.testid = 'runs-filter-value';
  filterRow.appendChild(filterValue);

  const filterClearButton = makeIconButton(
    'close',
    '필터 해제',
    '작업 필터 해제',
    'runs-filter-clear',
    'ghost',
  );
  applyTouchTarget(filterClearButton);
  filterClearButton.addEventListener('click', () => {
    setTaskFilter(null);
  });
  filterRow.appendChild(filterClearButton);
  root.appendChild(filterRow);

  // ── 상태/오류 표시 ──
  const statusLine = applyType(document.createElement('div'), TYPE.caption);
  styled(statusLine, { display: 'none', color: COLOR.muted, flex: 'none' });
  statusLine.dataset.testid = 'runs-status';
  statusLine.setAttribute('role', 'status');
  root.appendChild(statusLine);

  const errorBar = styled(document.createElement('div'), {
    display: 'none',
    alignItems: 'center',
    flexWrap: 'wrap',
    gap: SPACE.md,
    padding: `${SPACE.sm} ${SPACE.lg}`,
    background: COLLISION.surface,
    border: `${BORDER_WIDTH.hair} solid ${COLLISION.border}`,
    borderRadius: RADIUS.md,
    flex: 'none',
  });
  errorBar.dataset.testid = 'runs-error';
  errorBar.setAttribute('role', 'alert');
  const errorMessage = applyType(document.createElement('span'), TYPE.body);
  styled(errorMessage, { flex: '1 1 auto', minWidth: '0', color: COLLISION.text });
  errorBar.appendChild(errorMessage);
  const retryButton = makeIconButton('refresh', '재시도', '다시 불러오기', 'runs-retry');
  applyTouchTarget(retryButton);
  retryButton.addEventListener('click', () => {
    void refresh();
  });
  errorBar.appendChild(retryButton);
  root.appendChild(errorBar);

  // ── 본문: 좌 목록 / 우 상세 ──
  const content = styled(document.createElement('div'), {
    display: 'flex',
    gap: SPACE.lg,
    flex: '1 1 auto',
    minHeight: '0',
    minWidth: '0',
  });
  root.appendChild(content);

  const listPane = styled(document.createElement('div'), {
    display: 'flex',
    flexDirection: 'column',
    gap: SPACE.lg,
    flex: LIST_PANE_FLEX,
    minWidth: '0',
    minHeight: '0',
  });
  content.appendChild(listPane);

  // 통계 카드 (작업 필터 + 표본 충분 시)
  const statsCard = styled(document.createElement('div'), {
    display: 'none',
    gap: SPACE.xl,
    padding: `${SPACE.lg} ${SPACE.xl}`,
    background: SURFACE.raised,
    border: `${BORDER_WIDTH.hair} solid ${BORDER.subtle}`,
    borderRadius: RADIUS.md,
    flex: 'none',
  });
  statsCard.dataset.testid = 'runs-stats-card';
  listPane.appendChild(statsCard);

  // local 모드 안내 (실행 기록은 서버 기능)
  const localNotice = makeEmptyState({
    iconName: 'cloudOff',
    titleKo: '서버 미연결',
    hintKo: '실행 기록은 서버에 저장됩니다. 서버 연결 후 다시 열어 주세요.',
    actions: [],
    testid: 'runs-local-notice',
  });
  localNotice.style.display = 'none';
  listPane.appendChild(localNotice);

  const emptyRefreshButton = makeIconButton(
    'refresh',
    '새로 고침',
    REFRESH_TITLE,
    'runs-empty-refresh',
  );
  emptyRefreshButton.addEventListener('click', () => {
    void refresh();
  });
  const listEmptyState = makeEmptyState({
    iconName: 'history',
    titleKo: '실행 기록이 없습니다',
    hintKo: '작업을 재생하면 실행 결과가 여기에 쌓입니다.',
    actions: [emptyRefreshButton],
    testid: 'runs-empty',
  });

  const table = makeDataTable<RunRecord>({
    columns: [
      { key: 'task', labelKo: '작업', width: '26%', render: (r): string => r.taskName },
      {
        key: 'result',
        labelKo: '결과',
        render: (r): HTMLElement => {
          const spec = resultBadgeSpec(r.result);
          return makeBadge(spec.labelKo, spec.status);
        },
      },
      {
        key: 'startedAt',
        labelKo: '시각',
        lang: 'en',
        render: (r): HTMLElement => {
          const el = applyType(document.createElement('span'), TYPE.monoBody);
          el.textContent = formatDateTimeKo(r.startedAtIso);
          return el;
        },
      },
      { key: 'duration', labelKo: '소요', render: (r): string => formatDurationKo(r.wallTimeSec) },
      { key: 'operator', labelKo: '작업자', render: (r): string => r.operatorName },
    ],
    rows: [],
    onRowClick: (run): void => {
      selectRun(run);
    },
    rowTestid: (r): string => `run-row-${r.id}`,
    emptyState: listEmptyState,
    ariaLabelKo: '실행 기록 목록',
  });
  styled(table.el, { flex: '1 1 auto', minHeight: '0' });
  listPane.appendChild(table.el);

  // 우측 상세
  const detailPane = styled(document.createElement('div'), {
    flex: '1 1 54%',
    minWidth: '0',
    minHeight: '0',
    overflowY: 'auto',
    background: SURFACE.panel,
    border: `${BORDER_WIDTH.hair} solid ${BORDER.subtle}`,
    borderRadius: RADIUS.md,
    padding: SPACE.xl,
    boxSizing: 'border-box',
  });
  detailPane.className = 'ui-scroll';
  detailPane.dataset.testid = 'run-detail';
  content.appendChild(detailPane);

  // ── 상세 렌더 ──

  const renderDetailPlaceholder = (): void => {
    detailPane.textContent = '';
    detailPane.appendChild(
      makeEmptyState({
        iconName: 'clipboard',
        titleKo: '실행을 선택하세요',
        hintKo: '왼쪽 목록에서 실행을 선택하면 개입 타임라인과 충돌 내역이 표시됩니다.',
        actions: [],
        testid: 'run-detail-placeholder',
      }),
    );
  };

  const sectionTitle = (textKo: string): HTMLElement => {
    const h = applyType(document.createElement('h3'), TYPE.subhead);
    styled(h, { margin: '0', color: COLOR.textStrong });
    h.textContent = textKo;
    return h;
  };

  const emptyCaption = (textKo: string): HTMLElement => {
    const el = applyType(document.createElement('div'), TYPE.caption);
    styled(el, { color: COLOR.muted });
    el.textContent = textKo;
    return el;
  };

  /** 도메인 식별자 칩 (노드 id 등) — 영문 원문 + lang="en" (§4-b) */
  const idChip = (id: string): HTMLElement => {
    const chip = applyType(document.createElement('span'), TYPE.monoBody);
    styled(chip, {
      background: SURFACE.sunken,
      border: `${BORDER_WIDTH.hair} solid ${BORDER.subtle}`,
      borderRadius: RADIUS.xs,
      padding: `0 ${SPACE.xs}`,
      color: COLOR.text,
      whiteSpace: 'nowrap',
    });
    chip.setAttribute('lang', 'en');
    chip.textContent = id;
    return chip;
  };

  const metaRow = (labelKo: string, value: HTMLElement | string): HTMLElement[] => {
    const label = applyType(document.createElement('dt'), TYPE.caption);
    styled(label, { margin: '0', color: COLOR.label });
    label.textContent = labelKo;
    const valueEl = applyType(document.createElement('dd'), TYPE.body);
    styled(valueEl, { margin: '0', color: COLOR.text, minWidth: '0' });
    if (typeof value === 'string') valueEl.textContent = value;
    else valueEl.appendChild(value);
    return [label, valueEl];
  };

  const renderDetail = (run: RunRecord): void => {
    detailPane.textContent = '';
    const wrap = styled(document.createElement('div'), {
      display: 'flex',
      flexDirection: 'column',
      gap: SPACE.xl,
    });
    detailPane.appendChild(wrap);

    // ── 결과 요약 헤더 ──
    const head = styled(document.createElement('div'), {
      display: 'flex',
      flexDirection: 'column',
      gap: SPACE.md,
    });
    const titleRow = styled(document.createElement('div'), {
      display: 'flex',
      alignItems: 'center',
      gap: SPACE.md,
      minWidth: '0',
    });
    const taskName = applyType(document.createElement('span'), TYPE.title);
    styled(taskName, {
      flex: '1 1 auto',
      minWidth: '0',
      overflow: 'hidden',
      textOverflow: 'ellipsis',
      whiteSpace: 'nowrap',
      color: COLOR.textStrong,
    });
    taskName.textContent = run.taskName;
    titleRow.appendChild(taskName);
    const spec = resultBadgeSpec(run.result);
    titleRow.appendChild(makeBadge(spec.labelKo, spec.status, { testid: 'run-detail-result' }));
    head.appendChild(titleRow);

    const meta = styled(document.createElement('dl'), {
      display: 'grid',
      gridTemplateColumns: 'auto 1fr auto 1fr',
      alignItems: 'baseline',
      gap: `${SPACE.xs} ${SPACE.lg}`,
      margin: '0',
    });
    const stepsEl = applyType(document.createElement('span'), TYPE.monoReadout);
    stepsEl.setAttribute('lang', 'en');
    stepsEl.textContent = `${run.stepsDone}/${run.stepsTotal}`;
    const simEl = applyType(document.createElement('span'), TYPE.monoReadout);
    simEl.setAttribute('lang', 'en');
    simEl.textContent = formatSimClock(run.simTimeSec);
    meta.append(
      ...metaRow('노드 진행', stepsEl),
      ...metaRow('시뮬 시간', simEl),
      ...metaRow('실제 소요', formatDurationKo(run.wallTimeSec)),
      ...metaRow('작업자', run.operatorName),
      ...metaRow('시작', formatDateTimeKo(run.startedAtIso)),
    );
    head.appendChild(meta);

    const actionsRow = styled(document.createElement('div'), {
      display: 'flex',
      gap: SPACE.md,
      flexWrap: 'wrap',
    });
    const openTaskButton = makeIconButton(
      'folderOpen',
      '작업 열기',
      '이 실행의 작업 열기',
      'run-open-task',
    );
    applyTouchTarget(openTaskButton);
    openTaskButton.addEventListener('click', () => {
      deps.onOpenTask(run.taskId);
    });
    actionsRow.appendChild(openTaskButton);
    head.appendChild(actionsRow);
    wrap.appendChild(head);

    // ── 개입 타임라인 (시각순) ──
    const timelineSection = styled(document.createElement('div'), {
      display: 'flex',
      flexDirection: 'column',
      gap: SPACE.sm,
    });
    timelineSection.appendChild(sectionTitle('개입 타임라인'));
    const interventions = sortInterventionsBySimTime(run.interventions);
    if (interventions.length === 0) {
      timelineSection.appendChild(emptyCaption('개입 없음'));
    } else {
      const list = styled(document.createElement('ol'), {
        display: 'flex',
        flexDirection: 'column',
        gap: SPACE.xs,
        margin: '0',
        padding: '0',
        listStyle: 'none',
      });
      list.dataset.testid = 'run-interventions';
      for (const iv of interventions) {
        const row = styled(document.createElement('li'), {
          display: 'flex',
          alignItems: 'center',
          gap: SPACE.md,
          minWidth: '0',
        });
        const time = applyType(document.createElement('span'), TYPE.monoReadout);
        styled(time, {
          width: `${TIMELINE_TIME_WIDTH_PX}px`,
          flex: 'none',
          textAlign: 'right',
          color: COLOR.muted,
        });
        time.setAttribute('lang', 'en');
        time.textContent = formatSimClock(iv.atSimSec);
        row.appendChild(time);
        const kindEl = applyType(document.createElement('span'), TYPE.body);
        styled(kindEl, { color: COLOR.text });
        kindEl.textContent = interventionKindKo(iv.kind);
        row.appendChild(kindEl);
        if (iv.nodeId !== null) row.appendChild(idChip(iv.nodeId));
        list.appendChild(row);
      }
      timelineSection.appendChild(list);
    }
    wrap.appendChild(timelineSection);

    // ── 충돌 목록 (+ 실패 지점 재현 — 킬러 기능) ──
    const collisionSection = styled(document.createElement('div'), {
      display: 'flex',
      flexDirection: 'column',
      gap: SPACE.sm,
    });
    collisionSection.appendChild(sectionTitle('충돌'));
    if (run.collisions.length === 0) {
      collisionSection.appendChild(emptyCaption('충돌 없음'));
    } else {
      const list = styled(document.createElement('ol'), {
        display: 'flex',
        flexDirection: 'column',
        margin: '0',
        padding: '0',
        listStyle: 'none',
      });
      list.dataset.testid = 'run-collisions';
      run.collisions.forEach((col, i) => {
        const row = styled(document.createElement('li'), {
          display: 'flex',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: SPACE.md,
          padding: `${SPACE.sm} 0`,
          borderBottom: i === run.collisions.length - 1
            ? 'none'
            : `${BORDER_WIDTH.hair} solid ${BORDER.subtle}`,
        });
        row.dataset.testid = `run-collision-${i}`;
        const colSpec = collisionBadgeSpec(col.classification);
        row.appendChild(makeBadge(colSpec.labelKo, colSpec.status));
        const pair = applyType(document.createElement('span'), TYPE.monoBody);
        styled(pair, { color: COLOR.text, minWidth: '0', overflowWrap: 'anywhere' });
        pair.setAttribute('lang', 'en');
        pair.textContent = `${col.entityA} × ${col.entityB}`;
        row.appendChild(pair);
        const at = applyType(document.createElement('span'), TYPE.monoReadout);
        styled(at, { color: COLOR.muted });
        at.setAttribute('lang', 'en');
        at.textContent = formatSimClock(col.atSimSec);
        row.appendChild(at);
        if (col.nodeId !== null) row.appendChild(idChip(col.nodeId));

        const target = replayTarget(col);
        if (target !== null) {
          const replayButton = makeIconButton(
            'play',
            '이 노드부터 재현',
            `이 노드부터 재현 — ${target}`,
            `run-replay-${i}`,
            'accent',
          );
          applyTouchTarget(replayButton);
          replayButton.addEventListener('click', () => {
            deps.onReplayFromNode(run.taskId, target);
          });
          row.appendChild(replayButton);
        }
        list.appendChild(row);
      });
      collisionSection.appendChild(list);
    }
    wrap.appendChild(collisionSection);
  };

  const selectRun = (run: RunRecord): void => {
    selectedId = run.id;
    table.setSelected((r) => r.id === selectedId);
    renderDetail(run);
  };

  // ── 통계 카드 렌더 ──

  const renderStats = (stats: TaskStats): void => {
    statsCard.textContent = '';
    statsCard.style.display = 'flex';
    const tile = (labelKo: string, valueText: string, mono: boolean): HTMLElement => {
      const box = styled(document.createElement('div'), {
        display: 'flex',
        flexDirection: 'column',
        gap: SPACE.xxs,
        minWidth: '0',
      });
      const label = applyType(document.createElement('span'), TYPE.caption);
      styled(label, { color: COLOR.label });
      label.textContent = labelKo;
      const value = applyType(
        document.createElement('span'),
        mono ? TYPE.monoReadout : TYPE.bodyStrong,
      );
      styled(value, { color: COLOR.textStrong });
      value.textContent = valueText;
      box.append(label, value);
      return box;
    };
    const rate = successRatePercent(stats.runCount, stats.successCount);
    statsCard.append(
      tile('실행 횟수', `${stats.runCount}`, true),
      tile('성공률', rate === null ? '—' : `${rate}%`, true),
      tile(
        '평균 소요',
        stats.avgDurationSec === null ? '—' : formatDurationKo(stats.avgDurationSec),
        false,
      ),
    );
  };

  const hideStats = (): void => {
    statsCard.style.display = 'none';
    statsCard.textContent = '';
  };

  // ── 연결/오류 표면 ──

  const setLoading = (loading: boolean): void => {
    statusLine.style.display = loading ? '' : 'none';
    statusLine.textContent = loading ? '불러오는 중…' : '';
  };

  const showError = (messageKo: string): void => {
    errorMessage.textContent = messageKo;
    errorBar.style.display = 'flex';
  };

  const hideError = (): void => {
    errorBar.style.display = 'none';
  };

  /** 연결 상태를 UI에 반영한다. false면 목록 요청을 하지 않는다(local 모드). */
  const applyConnectionUi = (): boolean => {
    const conn = connection();
    const local = conn.mode === 'local';
    refreshButton.disabled = local;
    const reason = local ? MSG_RUNS_LOCAL_KO : REFRESH_TITLE;
    refreshButton.title = reason;
    refreshButton.setAttribute('aria-label', reason);
    localNotice.style.display = local ? '' : 'none';
    table.el.style.display = local ? 'none' : '';
    if (local) {
      hideStats();
      hideError();
      setLoading(false);
      renderDetailPlaceholder();
    }
    return !local;
  };

  const updateFilterRow = (): void => {
    if (taskFilter === null) {
      filterRow.style.display = 'none';
      return;
    }
    filterRow.style.display = 'flex';
    // 작업 이름은 기록의 스냅샷에서 얻는다 — 아직 없으면 id 원문(lang="en")
    const name = rows[0]?.taskName;
    if (name !== undefined) {
      filterValue.textContent = name;
      filterValue.removeAttribute('lang');
    } else {
      filterValue.textContent = taskFilter;
      filterValue.setAttribute('lang', 'en');
    }
  };

  // ── 불러오기 ──

  async function refresh(): Promise<void> {
    const seq = ++requestSeq;
    if (!applyConnectionUi()) return;
    hideError();
    setLoading(true);

    const filter = taskFilter;
    const statsPromise = filter === null ? null : deps.stats(filter);
    const listRes = await deps.runs.list(
      filter === null ? { limit: RUNS_PAGE_LIMIT } : { taskId: filter, limit: RUNS_PAGE_LIMIT },
    );
    if (disposed || seq !== requestSeq) return;
    setLoading(false);

    if (listRes.kind !== 'ok') {
      showError(listRes.messageKo);
      announcer.announceNow(`실행 기록 불러오기 실패 — ${listRes.messageKo}`);
      return;
    }

    rows = listRes.items;
    table.setRows(rows);
    updateFilterRow();
    announcer.announce(`실행 기록 ${rows.length}건`);

    // 선택 유지 — 새 목록에 없으면 해제
    const selected = rows.find((r) => r.id === selectedId);
    if (selected === undefined) {
      selectedId = null;
      table.setSelected(null);
      renderDetailPlaceholder();
    } else {
      table.setSelected((r) => r.id === selectedId);
      renderDetail(selected);
    }

    // 통계 (작업 필터 시에만 — 표본이 작으면 숨긴다)
    if (statsPromise === null) {
      hideStats();
      return;
    }
    const statsRes = await statsPromise;
    if (disposed || seq !== requestSeq) return;
    if (statsRes.kind === 'ok' && shouldShowStats(statsRes.stats.runCount)) {
      renderStats(statsRes.stats);
    } else {
      hideStats(); // 통계 실패는 목록을 막지 않는다 — 부가 정보다
    }
  }

  function setTaskFilter(taskId: string | null): void {
    if (taskId === taskFilter) return;
    taskFilter = taskId;
    selectedId = null;
    renderDetailPlaceholder();
    updateFilterRow();
    void refresh();
  }

  // ── 초기화 ──
  renderDetailPlaceholder();
  updateFilterRow();
  host.appendChild(root);
  void refresh();

  return {
    refresh: (): void => {
      void refresh();
    },
    setTaskFilter,
    dispose: (): void => {
      disposed = true;
      requestSeq += 1;
      announcer.dispose();
      table.dispose();
      root.remove();
    },
  };
}
