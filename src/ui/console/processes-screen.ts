// ui/console/processes-screen.ts — 콘솔 평면 ① 공정 목록 화면 (docs/BACKEND.md Phase 12+)
//
// 공정(Process) = 라인 레이아웃(씬) + 규칙 + 소속 장비/작업이다. 이 화면은 공정을
// 카드 그리드로 보여주고, 만들기/규칙 편집/소속 장비 체크/소속 작업 목록/삭제(soft)를
// 제공한다. 씬 사본 생성(새 작업)·스튜디오 이동은 **콜백만** 노출한다 — 배선은 통합자 몫.
//
// ── 계층 규칙 (CLAUDE.md §3) ────────────────────────────────────────
// ui/console → src/api(리소스 결과 union) + src/schema(문서 타입)만 안다.
// core/render/main을 import하지 않는다. deps는 좁은 구조적 인터페이스다 —
// EntityClient 클래스가 아니라 메서드 시그니처만 요구한다(테스트에서 가짜 주입 가능).
//
// ── 1차 사용자: 로봇 설치기사 (BACKEND §1) ──────────────────────────
// 공유 단말·장갑·서두름 전제: 터치 타깃 ≥44px(applyTouchTarget), 상태는 배지로,
// 빈 상태는 다음 행동을 안내, 파괴적 동작(삭제)은 soft + 실행취소 토스트(§2.11).
// 서버 미연결(local/offline)에서는 쓰기 버튼을 **사유 title과 함께** 비활성한다.
//
// 순수 로직(규칙 폼 정규화·작업 카운트·카드 보조행·차단 사유)은 DOM 없이 export되어
// node 환경 vitest가 검증한다(processes-screen.test.ts — toast.test.ts 관례).

import { lastSyncAgeKo } from '../../api';
import type { ConnectionState, GetResult, ListResult, RemoveResult, SaveResult } from '../../api';
import type { EntityMeta, ProcessDoc, ProcessRules, RecordEnvelope } from '../../schema/entities';
import { createAnnouncer } from '../a11y';
import { makeIconButton } from '../icons';
import { COLOR, ICON, SPACE, TYPE, applyType, makeButton, styled } from '../theme';
import type { ToastHandle } from '../feedback/toast';
import {
  TOUCH_TARGET_MIN_PX,
  applyTouchTarget,
  filterRowsByQuery,
  makeBadge,
  makeCard,
  makeCardGrid,
  makeEmptyState,
  makeModalShell,
  makeSearchField,
} from './primitives';
import type { ModalShellHandle } from './primitives';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4) ─────────────────────────────

/** 헤더 검색 필드 폭 — 카드 그리드 헤더에서 만들기 버튼과 공존하는 고정 폭 */
const SEARCH_FIELD_WIDTH_PX = 240;

// ── 순수 상수/헬퍼 (DOM 비의존 — node 테스트 대상) ──────────────────

/**
 * 새 공정의 규칙 기본값 — 충돌 자동 정지는 **켜짐**이 안전 기본이다(현장 라인에서
 * 예기치 않은 접촉을 조용히 지나치는 쪽이 위험). 속도 상한은 기본 없음(null).
 */
export const DEFAULT_PROCESS_RULES: ProcessRules = {
  autoPauseOnCollision: true,
  speedLimitMult: null,
};

export interface SpeedLimitOption {
  /** select의 value 문자열 */
  readonly value: string;
  readonly labelKo: string;
  /** ProcessRules.speedLimitMult 값 (null = 제한 없음) */
  readonly mult: number | null;
}

/** 재생 속도 상한 선택지 — 스키마 주석(1|2|4|null)과 1:1 (entities.ts processRulesSchema) */
export const SPEED_LIMIT_OPTIONS: readonly SpeedLimitOption[] = [
  { value: '1', labelKo: '1×', mult: 1 },
  { value: '2', labelKo: '2×', mult: 2 },
  { value: '4', labelKo: '4×', mult: 4 },
  { value: 'none', labelKo: '제한 없음', mult: null },
];

/** 알 수 없는 값의 폴백은 **가장 보수적인 1×**다 — 속도 상한에서 관대한 폴백은 위험하다 */
const SPEED_LIMIT_FALLBACK_MULT = 1;
const SPEED_LIMIT_FALLBACK_VALUE = '1';

/** select value → speedLimitMult. 미지 값은 1×(보수적 폴백). */
export function parseSpeedLimitValue(value: string): number | null {
  const opt = SPEED_LIMIT_OPTIONS.find((o) => o.value === value);
  return opt !== undefined ? opt.mult : SPEED_LIMIT_FALLBACK_MULT;
}

/** speedLimitMult → select value. 선택지에 없는 수치(예: 3)도 1×로 접는다(보수적). */
export function speedLimitOptionValue(mult: number | null): string {
  const opt = SPEED_LIMIT_OPTIONS.find((o) => o.mult === mult);
  return opt !== undefined ? opt.value : SPEED_LIMIT_FALLBACK_VALUE;
}

/** 규칙 폼 입력(체크박스 + select 문자열) → ProcessRules 정규화 */
export function normalizeProcessRules(input: {
  readonly autoPauseOnCollision: boolean;
  readonly speedLimitValue: string;
}): ProcessRules {
  return {
    autoPauseOnCollision: input.autoPauseOnCollision,
    speedLimitMult: parseSpeedLimitValue(input.speedLimitValue),
  };
}

/** 작업 목록 행 → 공정별 작업 수 (processId null = 무소속, 집계 제외) */
export function countTasksByProcess(
  items: readonly { readonly processId: string | null }[],
): Map<string, number> {
  const counts = new Map<string, number>();
  for (const item of items) {
    if (item.processId === null) continue;
    counts.set(item.processId, (counts.get(item.processId) ?? 0) + 1);
  }
  return counts;
}

/** 장비 체크리스트 토글 — 중복 없이 추가/제거, 기존 순서 보존 (순수) */
export function toggleDeviceId(
  deviceIds: readonly string[],
  deviceId: string,
  checked: boolean,
): string[] {
  const without = deviceIds.filter((id) => id !== deviceId);
  return checked ? [...without, deviceId] : without;
}

export interface ProcessDraft {
  readonly id: string;
  readonly name: string;
  readonly descriptionKo: string;
  /** SceneSpec 봉투(unknown) — null이면 "빈 바닥에서 시작"(스튜디오가 기본 씬으로 해석) */
  readonly scene: unknown;
  readonly deviceIds: readonly string[];
  readonly rules: ProcessRules;
}

/** 폼 초안 → ProcessDoc — 이름/설명 trim, deviceIds 중복 제거 */
export function buildProcessDoc(draft: ProcessDraft): ProcessDoc {
  return {
    id: draft.id,
    name: draft.name.trim(),
    descriptionKo: draft.descriptionKo.trim(),
    scene: draft.scene,
    deviceIds: [...new Set(draft.deviceIds)],
    rules: { ...draft.rules },
  };
}

export interface ProcessCardInfo {
  readonly descriptionKo: string;
  /** 문서 로드 실패 시 null — 카드는 '?'로 정직하게 표기 */
  readonly deviceCount: number | null;
  readonly taskCount: number;
  readonly updatedAtIso: string;
  readonly updatedByName: string;
}

/** 카드 보조행 텍스트 — 설명(있으면) · '장비 n대 · 작업 m개' · '수정 x분 전 · 이름' */
export function processCardSublines(info: ProcessCardInfo, nowMs: number): string[] {
  const lines: string[] = [];
  const desc = info.descriptionKo.trim();
  if (desc !== '') lines.push(desc);
  const deviceLabel = info.deviceCount === null ? '?' : String(info.deviceCount);
  lines.push(`장비 ${deviceLabel}대 · 작업 ${info.taskCount}개`);
  lines.push(`수정 ${lastSyncAgeKo(info.updatedAtIso, nowMs)} · ${info.updatedByName}`);
  return lines;
}

/**
 * 서버 쓰기 차단 사유 — null이면 사용 가능. 비활성 버튼의 title에 **반드시** 이 사유를
 * 병기한다(회색 버튼이 이유 없이 죽어 있으면 설치기사는 고장으로 읽는다).
 */
export function serverBlockReasonKo(state: ConnectionState): string | null {
  if (state.mode === 'local') {
    return '서버가 설정되지 않았습니다 (로컬 모드) — 서버 연결 후 사용할 수 있습니다';
  }
  if (!state.online) {
    return '오프라인입니다 — 서버 연결이 복구되면 사용할 수 있습니다';
  }
  return null;
}

// ── 공용 폼 조립 헬퍼 (콘솔 화면 공용 — devices-screen이 재사용) ────

/** 세로 라벨 + 컨트롤 폼 행 — <label>로 감싸 클릭·포커스 연결을 공짜로 얻는다 */
export function formRow(labelKo: string, control: HTMLElement): HTMLLabelElement {
  const row = styled(document.createElement('label'), {
    display: 'flex',
    flexDirection: 'column',
    gap: SPACE.xs,
  });
  const label = applyType(document.createElement('span'), TYPE.caption);
  styled(label, { color: COLOR.label });
  label.textContent = labelKo;
  row.appendChild(label);
  row.appendChild(control);
  return row;
}

/** 텍스트 입력 — .ui-input + 터치 타깃 44px */
export function makeTextInput(testid: string, value: string, placeholderKo: string): HTMLInputElement {
  const input = document.createElement('input');
  input.type = 'text';
  input.className = 'ui-input';
  input.value = value;
  input.placeholder = placeholderKo;
  input.dataset.testid = testid;
  applyTouchTarget(input);
  return input;
}

/** 모달 내부 섹션 제목 */
export function sectionTitle(textKo: string): HTMLElement {
  const el = applyType(document.createElement('h3'), TYPE.subhead);
  styled(el, { margin: '0', color: COLOR.textStrong });
  el.textContent = textKo;
  return el;
}

/** 보조 설명 캡션 */
function captionText(textKo: string): HTMLElement {
  const el = applyType(document.createElement('p'), TYPE.caption);
  styled(el, { margin: '0', color: COLOR.muted });
  el.textContent = textKo;
  return el;
}

// ── deps 계약 (좁은 구조적 인터페이스 — EntityClient가 구조적으로 만족) ──

export interface ProcessesResource {
  list(opts?: { readonly q?: string }): Promise<ListResult>;
  get(id: string): Promise<GetResult<ProcessDoc>>;
  create(doc: ProcessDoc): Promise<SaveResult<ProcessDoc>>;
  update(id: string, doc: ProcessDoc, baseVersion: number): Promise<SaveResult<ProcessDoc>>;
  remove(id: string): Promise<RemoveResult>;
  restore(id: string): Promise<GetResult<ProcessDoc>>;
}

/** 소속 장비 체크리스트용 — 이름만 필요하므로 목록만 요구한다 */
export interface DevicesResourceLite {
  list(): Promise<ListResult>;
}

/** 소속 작업 목록/카운트용 */
export interface TasksResourceLite {
  list(opts?: { readonly processId?: string }): Promise<ListResult>;
}

export interface ProcessesScreenDeps {
  readonly processes: ProcessesResource;
  readonly devices: DevicesResourceLite;
  readonly tasks: TasksResourceLite;
  /** [이 공정으로 새 작업] — 씬 사본 생성·이동은 통합자 몫(entities.ts 복사본 의미론) */
  onCreateTask(processId: string): void;
  /** [공정 씬을 스튜디오에서 편집] */
  onEditScene(processId: string): void;
  /** '현재 스튜디오 씬 사용' — SceneSpec 봉투(unknown)를 돌려준다 */
  onCaptureCurrentScene(): Promise<unknown>;
  /** 소속 작업 [열기] — 없으면 열기 버튼을 그리지 않는다 */
  onOpenTask?(taskId: string): void;
  /** 연결 상태 getter — refresh 시점마다 재평가한다 (BACKEND §6 3상태) */
  connection(): ConnectionState;
  /** 앱 전역 토스트 표면 (실행취소 액션 포함 — CLAUDE.md §2.11) */
  readonly toast: ToastHandle;
}

export interface ConsoleScreenHandle {
  refresh(): void;
  dispose(): void;
}

// ── 마운트 ──────────────────────────────────────────────────────────

export function mountProcessesScreen(
  host: HTMLElement,
  deps: ProcessesScreenDeps,
): ConsoleScreenHandle {
  const root = styled(document.createElement('div'), {
    display: 'flex',
    flexDirection: 'column',
    gap: SPACE.xl,
    padding: SPACE.xl,
    boxSizing: 'border-box',
    height: '100%',
    minHeight: '0',
    overflow: 'hidden',
  });
  root.dataset.testid = 'processes-screen';
  host.appendChild(root);

  const announcer = createAnnouncer(root);

  // ── 헤더: 제목 · 연결 배지 · 검색 · 만들기 ────────────────────────
  const header = styled(document.createElement('div'), {
    display: 'flex',
    alignItems: 'center',
    gap: SPACE.lg,
    flexWrap: 'wrap',
  });
  const heading = applyType(document.createElement('h2'), TYPE.title);
  styled(heading, { margin: '0', color: COLOR.textStrong });
  heading.textContent = '공정';
  header.appendChild(heading);

  const connHost = styled(document.createElement('span'), { display: 'inline-flex' });
  header.appendChild(connHost);

  const spacer = styled(document.createElement('span'), { flex: '1 1 auto' });
  header.appendChild(spacer);

  let queryText = '';
  const search = makeSearchField({
    placeholderKo: '공정 검색',
    testid: 'processes-search',
    onInput: (q) => {
      queryText = q;
      renderList();
    },
  });
  styled(search.el, { width: `${SEARCH_FIELD_WIDTH_PX}px`, maxWidth: '100%' });
  header.appendChild(search.el);

  const createButton = makeIconButton('plus', '공정 만들기', '공정 만들기', 'processes-create', 'primary');
  applyTouchTarget(createButton);
  createButton.addEventListener('click', () => {
    openCreateDialog();
  });
  header.appendChild(createButton);
  root.appendChild(header);

  // ── 목록 영역 ─────────────────────────────────────────────────────
  const listHost = styled(document.createElement('div'), {
    flex: '1 1 auto',
    minHeight: '0',
    overflowY: 'auto',
  });
  listHost.className = 'ui-scroll';
  root.appendChild(listHost);

  const grid = makeCardGrid({ testid: 'processes-grid' });

  // ── 상태 ──────────────────────────────────────────────────────────
  let disposed = false;
  let loadSeq = 0;
  let detailBusy = false;
  let items: EntityMeta[] = [];
  const docs = new Map<string, RecordEnvelope<ProcessDoc>>();
  let taskCounts: Map<string, number> = new Map();
  let activeModal: ModalShellHandle | null = null;

  const closeActiveModal = (): void => {
    if (activeModal !== null) {
      activeModal.dispose();
      activeModal = null;
    }
  };

  /** 모달을 활성으로 등록 — 이전 모달은 정리한다 (한 번에 하나) */
  const adoptModal = (modal: ModalShellHandle): void => {
    closeActiveModal();
    activeModal = modal;
    root.appendChild(modal.root);
    modal.open();
  };

  /** 저장 성공 등 프로그램적 닫기 (onClose를 부르지 않는 close + 정리) */
  const finishModal = (modal: ModalShellHandle): void => {
    modal.close();
    if (activeModal === modal) activeModal = null;
    modal.dispose();
  };

  const blockReason = (): string | null => serverBlockReasonKo(deps.connection());

  const paintConnection = (): void => {
    connHost.textContent = '';
    const state = deps.connection();
    const reason = serverBlockReasonKo(state);
    if (reason !== null) {
      const badge = makeBadge(
        state.mode === 'local' ? '로컬 모드' : '오프라인',
        state.mode === 'local' ? 'neutral' : 'warn',
        { testid: 'processes-conn-badge' },
      );
      badge.title = reason;
      connHost.appendChild(badge);
    }
    createButton.disabled = reason !== null;
    createButton.title = reason !== null ? `공정 만들기 — ${reason}` : '공정 만들기';
  };

  // ── 목록 렌더 ─────────────────────────────────────────────────────

  const showOnly = (el: HTMLElement): void => {
    listHost.textContent = '';
    listHost.appendChild(el);
  };

  const showLoading = (): void => {
    const el = applyType(document.createElement('p'), TYPE.body);
    styled(el, { margin: '0', color: COLOR.muted });
    el.dataset.testid = 'processes-loading';
    el.textContent = '불러오는 중…';
    showOnly(el);
  };

  const showError = (messageKo: string): void => {
    const retry = makeIconButton('refresh', '다시 시도', '목록 다시 불러오기', 'processes-retry');
    retry.addEventListener('click', () => {
      refresh();
    });
    showOnly(
      makeEmptyState({
        iconName: 'alert',
        titleKo: '목록을 불러오지 못했습니다',
        hintKo: messageKo,
        actions: [retry],
        testid: 'processes-error',
      }),
    );
  };

  const makeCreateActionButton = (testid: string): HTMLButtonElement => {
    const btn = makeIconButton('plus', '공정 만들기', '공정 만들기', testid, 'primary');
    const reason = blockReason();
    if (reason !== null) {
      btn.disabled = true;
      btn.title = `공정 만들기 — ${reason}`;
    }
    btn.addEventListener('click', () => {
      openCreateDialog();
    });
    return btn;
  };

  const renderList = (): void => {
    if (items.length === 0) {
      showOnly(
        makeEmptyState({
          iconName: 'factory',
          titleKo: '아직 공정이 없습니다',
          hintKo:
            '공정은 라인의 설비 배치와 규칙입니다. 공정을 만들어 두면 같은 라인의 작업들이 배치와 규칙을 공유합니다.',
          actions: [makeCreateActionButton('processes-create-empty')],
          testid: 'processes-empty',
        }),
      );
      return;
    }
    const filtered = filterRowsByQuery(items, queryText, (m) => {
      const doc = docs.get(m.id)?.doc;
      return `${m.name} ${doc?.descriptionKo ?? ''}`;
    });
    if (filtered.length === 0) {
      const clear = makeButton('검색어 지우기', '검색어 지우기', 'processes-search-clear-empty', 'ghost');
      applyTouchTarget(clear);
      clear.addEventListener('click', () => {
        search.setValue('');
      });
      showOnly(
        makeEmptyState({
          iconName: 'search',
          titleKo: '검색 결과가 없습니다',
          actions: [clear],
          testid: 'processes-search-empty',
        }),
      );
      return;
    }

    const nowMs = Date.now();
    const reason = blockReason();
    const cards = filtered.map((m) => {
      const env = docs.get(m.id);
      const sublines = processCardSublines(
        {
          descriptionKo: env?.doc.descriptionKo ?? '',
          deviceCount: env !== undefined ? env.doc.deviceIds.length : null,
          taskCount: taskCounts.get(m.id) ?? 0,
          updatedAtIso: m.meta.updatedAtIso,
          updatedByName: m.meta.updatedByName,
        },
        nowMs,
      );
      const del = makeIconButton('trash', '', `공정 '${m.name}' 삭제`, `process-delete-${m.id}`, 'ghost');
      applyTouchTarget(del, { square: true });
      if (reason !== null) {
        del.disabled = true;
        del.title = `삭제 — ${reason}`;
      }
      del.addEventListener('click', () => {
        void removeProcess(m);
      });
      return makeCard({
        title: m.name,
        sublines,
        onClick: () => {
          void openDetail(m);
        },
        actions: [del],
        testid: `process-card-${m.id}`,
      });
    });
    grid.setCards(cards);
    showOnly(grid.el);
  };

  // ── 데이터 로드 ───────────────────────────────────────────────────

  const load = async (): Promise<void> => {
    const seq = ++loadSeq;
    paintConnection();
    // 로컬 모드 — 서버 개체가 없다. 요청을 보내지 않는다(정적 배포에서 /api가 SPA 폴백에
    // 맞으면 연결 상태 머신이 "서버 도달"로 오판할 수 있다 — tasks-screen과 동일 계약).
    if (deps.connection().mode === 'local') {
      items = [];
      taskCounts = new Map();
      docs.clear();
      renderList();
      return;
    }
    showLoading();
    const [listRes, tasksRes] = await Promise.all([deps.processes.list(), deps.tasks.list()]);
    if (disposed || seq !== loadSeq) return;
    if (listRes.kind !== 'ok') {
      showError(listRes.messageKo);
      return;
    }
    items = listRes.items;
    taskCounts = tasksRes.kind === 'ok' ? countTasksByProcess(tasksRes.items) : new Map();
    docs.clear();
    // 카드에 장비 수·설명이 필요하다 — 목록 메타에는 payload가 없어 문서를 병렬로 당긴다.
    // (현장 공정 수는 소수라는 전제. 개별 실패는 카드에 '?'로 정직하게 표기된다.)
    await Promise.all(
      listRes.items.map(async (m) => {
        const r = await deps.processes.get(m.id);
        if (r.kind === 'ok') docs.set(m.id, r.record);
      }),
    );
    if (disposed || seq !== loadSeq) return;
    renderList();
    announcer.announce(`공정 ${items.length}개`);
  };

  const refresh = (): void => {
    void load();
  };

  // ── 삭제 (soft + 실행취소 토스트 — CLAUDE.md §2.11) ───────────────

  const removeProcess = async (m: EntityMeta): Promise<void> => {
    const r = await deps.processes.remove(m.id);
    if (disposed) return;
    if (r.kind !== 'ok') {
      deps.toast.show('error', '공정을 삭제하지 못했습니다', { detail: r.messageKo });
      return;
    }
    deps.toast.show('success', `공정 '${m.name}' 삭제됨`, {
      detail: '30일 안에 복원할 수 있습니다',
      action: {
        label: '실행 취소',
        onClick: () => {
          void restoreProcess(m);
        },
      },
    });
    announcer.announceNow(`공정 '${m.name}' 삭제됨`);
    refresh();
  };

  const restoreProcess = async (m: EntityMeta): Promise<void> => {
    const r = await deps.processes.restore(m.id);
    if (disposed) return;
    if (r.kind !== 'ok') {
      deps.toast.show('error', '공정을 복원하지 못했습니다', { detail: r.messageKo });
      return;
    }
    deps.toast.show('success', `공정 '${m.name}' 복원됨`);
    refresh();
  };

  // ── 만들기 다이얼로그 ─────────────────────────────────────────────

  const openCreateDialog = (): void => {
    const reason = blockReason();
    if (reason !== null) return; // 버튼이 비활성이라 정상 경로로는 도달하지 않는다 (방어)

    const modal = makeModalShell({
      titleKo: '공정 만들기',
      testid: 'process-create-modal',
      onClose: () => {
        if (activeModal === modal) activeModal = null;
        modal.dispose();
      },
    });

    const nameInput = makeTextInput('process-name-input', '', '예: 1라인 포장');
    const descInput = document.createElement('textarea');
    descInput.className = 'ui-input';
    descInput.rows = 3;
    descInput.placeholder = '이 공정이 하는 일 (선택)';
    descInput.dataset.testid = 'process-desc-input';
    styled(descInput, { resize: 'vertical' });

    const markDirty = (): void => {
      modal.setDirty(true);
    };
    nameInput.addEventListener('input', markDirty);
    descInput.addEventListener('input', markDirty);

    // 씬 출처 — 라디오 2택 (기본: 현재 스튜디오 씬)
    const sourceGroup = styled(document.createElement('div'), {
      display: 'flex',
      flexDirection: 'column',
      gap: SPACE.xs,
    });
    sourceGroup.setAttribute('role', 'radiogroup');
    sourceGroup.setAttribute('aria-label', '씬 출처');
    const sourceLabel = applyType(document.createElement('span'), TYPE.caption);
    styled(sourceLabel, { color: COLOR.label });
    sourceLabel.textContent = '씬 출처';
    sourceGroup.appendChild(sourceLabel);

    const makeSourceRadio = (
      value: 'capture' | 'blank',
      labelKo: string,
      checked: boolean,
      testid: string,
    ): { row: HTMLLabelElement; input: HTMLInputElement } => {
      const row = document.createElement('label');
      row.className = 'ui-check-label';
      applyTouchTarget(row);
      const input = document.createElement('input');
      input.type = 'radio';
      input.name = 'process-scene-source';
      input.value = value;
      input.checked = checked;
      input.dataset.testid = testid;
      styled(input, {
        width: `${ICON.lg}px`,
        height: `${ICON.lg}px`,
        accentColor: 'var(--rsw-accent)',
      });
      input.addEventListener('change', markDirty);
      const text = applyType(document.createElement('span'), TYPE.body);
      styled(text, { color: COLOR.text });
      text.textContent = labelKo;
      row.appendChild(input);
      row.appendChild(text);
      return { row, input };
    };
    const captureRadio = makeSourceRadio(
      'capture',
      '현재 스튜디오 씬 사용',
      true,
      'process-scene-capture',
    );
    const blankRadio = makeSourceRadio('blank', '빈 바닥에서 시작', false, 'process-scene-blank');
    sourceGroup.appendChild(captureRadio.row);
    sourceGroup.appendChild(blankRadio.row);

    modal.body.appendChild(formRow('이름', nameInput));
    modal.body.appendChild(formRow('설명', descInput));
    modal.body.appendChild(sourceGroup);

    const submit = makeButton('만들기', '공정 만들기', 'process-create-submit', 'primary');
    applyTouchTarget(submit);
    submit.addEventListener('click', () => {
      void submitCreate();
    });
    modal.footer.appendChild(submit);

    const submitCreate = async (): Promise<void> => {
      if (nameInput.value.trim() === '') {
        deps.toast.show('warn', '이름을 입력하세요');
        nameInput.focus();
        return;
      }
      submit.disabled = true;
      try {
        const scene = captureRadio.input.checked ? await deps.onCaptureCurrentScene() : null;
        const doc = buildProcessDoc({
          id: crypto.randomUUID(),
          name: nameInput.value,
          descriptionKo: descInput.value,
          scene,
          deviceIds: [],
          rules: { ...DEFAULT_PROCESS_RULES },
        });
        const r = await deps.processes.create(doc);
        if (disposed) return;
        if (r.kind === 'ok') {
          deps.toast.show('success', `공정 '${doc.name}' 만들어짐`);
          finishModal(modal);
          refresh();
        } else if (r.kind === 'conflict') {
          deps.toast.show('error', r.messageKo);
        } else {
          deps.toast.show('error', '공정을 만들지 못했습니다', { detail: r.messageKo });
        }
      } catch {
        deps.toast.show('error', '현재 씬을 캡처하지 못했습니다');
      } finally {
        submit.disabled = false;
      }
    };

    adoptModal(modal);
    nameInput.focus();
  };

  // ── 상세 모달 (이름/설명/규칙/장비/작업) ──────────────────────────

  const openDetail = async (m: EntityMeta): Promise<void> => {
    if (detailBusy) return;
    detailBusy = true;
    const [docRes, devRes, taskRes] = await Promise.all([
      deps.processes.get(m.id),
      deps.devices.list(),
      deps.tasks.list({ processId: m.id }),
    ]);
    detailBusy = false;
    if (disposed) return;
    if (docRes.kind !== 'ok') {
      deps.toast.show('error', '공정을 열지 못했습니다', { detail: docRes.messageKo });
      return;
    }
    buildDetailModal(
      docRes.record,
      devRes.kind === 'ok' ? devRes.items : null,
      taskRes.kind === 'ok' ? taskRes.items : [],
    );
  };

  const buildDetailModal = (
    env: RecordEnvelope<ProcessDoc>,
    deviceItems: EntityMeta[] | null,
    taskItems: EntityMeta[],
  ): void => {
    const doc = env.doc;
    const reason = blockReason();
    const modal = makeModalShell({
      titleKo: `공정 · ${doc.name}`,
      widthPx: 560,
      testid: 'process-detail-modal',
      onClose: () => {
        if (activeModal === modal) activeModal = null;
        modal.dispose();
      },
    });
    const markDirty = (): void => {
      modal.setDirty(true);
    };

    // 이름/설명
    const nameInput = makeTextInput('process-detail-name', doc.name, '공정 이름');
    nameInput.addEventListener('input', markDirty);
    const descInput = document.createElement('textarea');
    descInput.className = 'ui-input';
    descInput.rows = 3;
    descInput.value = doc.descriptionKo;
    descInput.placeholder = '이 공정이 하는 일 (선택)';
    descInput.dataset.testid = 'process-detail-desc';
    styled(descInput, { resize: 'vertical' });
    descInput.addEventListener('input', markDirty);
    modal.body.appendChild(formRow('이름', nameInput));
    modal.body.appendChild(formRow('설명', descInput));

    // 규칙 — 실행 기본값 (작업 열 때 초기값으로 복사된다 — 참조 아님)
    modal.body.appendChild(sectionTitle('규칙'));
    const autoPauseRow = document.createElement('label');
    autoPauseRow.className = 'ui-check-label';
    applyTouchTarget(autoPauseRow);
    const autoPauseBox = document.createElement('input');
    autoPauseBox.type = 'checkbox';
    autoPauseBox.className = 'ui-check';
    autoPauseBox.checked = doc.rules.autoPauseOnCollision;
    autoPauseBox.dataset.testid = 'process-rule-autopause';
    autoPauseBox.addEventListener('change', markDirty);
    const autoPauseText = applyType(document.createElement('span'), TYPE.body);
    styled(autoPauseText, { color: COLOR.text });
    autoPauseText.textContent = '예기치 않은 충돌 시 자동 일시정지';
    autoPauseRow.appendChild(autoPauseBox);
    autoPauseRow.appendChild(autoPauseText);
    modal.body.appendChild(autoPauseRow);

    const speedSelect = document.createElement('select');
    speedSelect.className = 'ui-select';
    speedSelect.dataset.testid = 'process-rule-speed';
    applyTouchTarget(speedSelect);
    for (const opt of SPEED_LIMIT_OPTIONS) {
      const option = document.createElement('option');
      option.value = opt.value;
      option.textContent = opt.labelKo;
      speedSelect.appendChild(option);
    }
    speedSelect.value = speedLimitOptionValue(doc.rules.speedLimitMult);
    speedSelect.addEventListener('change', markDirty);
    modal.body.appendChild(formRow('재생 속도 상한', speedSelect));

    // 소속 장비 체크리스트
    modal.body.appendChild(sectionTitle('소속 장비'));
    let deviceIds: string[] = [...doc.deviceIds];
    if (deviceItems === null) {
      modal.body.appendChild(captionText('장비 목록을 불러오지 못했습니다 — 장비 소속은 저장 시 그대로 유지됩니다'));
    } else if (deviceItems.length === 0) {
      modal.body.appendChild(captionText('등록된 장비가 없습니다 — 장비 화면에서 먼저 추가하세요'));
    } else {
      const checklist = styled(document.createElement('div'), {
        display: 'flex',
        flexDirection: 'column',
        gap: SPACE.xs,
      });
      checklist.setAttribute('role', 'group');
      checklist.setAttribute('aria-label', '소속 장비');
      for (const d of deviceItems) {
        const row = document.createElement('label');
        row.className = 'ui-check-label';
        applyTouchTarget(row);
        const box = document.createElement('input');
        box.type = 'checkbox';
        box.className = 'ui-check';
        box.checked = deviceIds.includes(d.id);
        box.dataset.testid = `process-device-${d.id}`;
        box.addEventListener('change', () => {
          deviceIds = toggleDeviceId(deviceIds, d.id, box.checked);
          markDirty();
        });
        const text = applyType(document.createElement('span'), TYPE.body);
        styled(text, { color: COLOR.text });
        text.textContent = d.name;
        row.appendChild(box);
        row.appendChild(text);
        checklist.appendChild(row);
      }
      modal.body.appendChild(checklist);
    }

    // 소속 작업 목록
    modal.body.appendChild(sectionTitle('소속 작업'));
    if (taskItems.length === 0) {
      modal.body.appendChild(captionText('이 공정으로 만든 작업이 없습니다'));
    } else {
      const taskList = styled(document.createElement('div'), {
        display: 'flex',
        flexDirection: 'column',
        gap: SPACE.xs,
      });
      taskList.setAttribute('role', 'list');
      taskList.setAttribute('aria-label', '소속 작업');
      const onOpenTask = deps.onOpenTask;
      for (const t of taskItems) {
        const row = styled(document.createElement('div'), {
          display: 'flex',
          alignItems: 'center',
          gap: SPACE.md,
          minHeight: `${TOUCH_TARGET_MIN_PX}px`,
        });
        row.setAttribute('role', 'listitem');
        const name = applyType(document.createElement('span'), TYPE.body);
        styled(name, {
          flex: '1 1 auto',
          minWidth: '0',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
          color: COLOR.text,
        });
        name.textContent = t.name;
        row.appendChild(name);
        if (onOpenTask !== undefined) {
          const open = makeIconButton(
            'folderOpen',
            '열기',
            `작업 '${t.name}' 열기`,
            `process-task-open-${t.id}`,
            'ghost',
          );
          applyTouchTarget(open);
          open.addEventListener('click', () => {
            finishModal(modal);
            onOpenTask(t.id);
          });
          row.appendChild(open);
        }
        taskList.appendChild(row);
      }
      modal.body.appendChild(taskList);
    }

    // 이동 액션 (씬 사본 생성/스튜디오 편집 — 콜백만, 배선은 통합자)
    const actionsRow = styled(document.createElement('div'), {
      display: 'flex',
      gap: SPACE.md,
      flexWrap: 'wrap',
      marginTop: SPACE.md,
    });
    const newTaskBtn = makeButton(
      '이 공정으로 새 작업',
      '이 공정의 씬 사본으로 새 작업 만들기',
      'process-new-task',
    );
    applyTouchTarget(newTaskBtn);
    newTaskBtn.addEventListener('click', () => {
      finishModal(modal);
      deps.onCreateTask(doc.id);
    });
    const editSceneBtn = makeButton(
      '공정 씬을 스튜디오에서 편집',
      '공정 씬을 스튜디오에서 편집',
      'process-edit-scene',
    );
    applyTouchTarget(editSceneBtn);
    editSceneBtn.addEventListener('click', () => {
      finishModal(modal);
      deps.onEditScene(doc.id);
    });
    actionsRow.appendChild(newTaskBtn);
    actionsRow.appendChild(editSceneBtn);
    modal.body.appendChild(actionsRow);

    // 푸터: 삭제(좌) · 저장(우)
    const deleteBtn = makeButton('삭제', `공정 '${doc.name}' 삭제`, 'process-detail-delete', 'danger');
    applyTouchTarget(deleteBtn);
    styled(deleteBtn, { marginRight: 'auto' });
    const saveBtn = makeButton('저장', '공정 저장', 'process-save', 'primary');
    applyTouchTarget(saveBtn);
    if (reason !== null) {
      deleteBtn.disabled = true;
      deleteBtn.title = `삭제 — ${reason}`;
      saveBtn.disabled = true;
      saveBtn.title = `저장 — ${reason}`;
    }
    deleteBtn.addEventListener('click', () => {
      finishModal(modal);
      void removeProcess(envMeta(env));
    });
    saveBtn.addEventListener('click', () => {
      void save();
    });
    modal.footer.appendChild(deleteBtn);
    modal.footer.appendChild(saveBtn);

    const save = async (): Promise<void> => {
      if (nameInput.value.trim() === '') {
        deps.toast.show('warn', '이름을 입력하세요');
        nameInput.focus();
        return;
      }
      saveBtn.disabled = true;
      const next = buildProcessDoc({
        id: doc.id,
        name: nameInput.value,
        descriptionKo: descInput.value,
        scene: doc.scene,
        deviceIds,
        rules: normalizeProcessRules({
          autoPauseOnCollision: autoPauseBox.checked,
          speedLimitValue: speedSelect.value,
        }),
      });
      const r = await deps.processes.update(doc.id, next, env.meta.version);
      if (disposed) return;
      saveBtn.disabled = reason !== null;
      if (r.kind === 'ok') {
        deps.toast.show('success', `공정 '${next.name}' 저장됨`);
        finishModal(modal);
        refresh();
      } else if (r.kind === 'conflict') {
        // 자동 병합 금지 (BACKEND §6) — 사유를 알리고 사람이 다시 연다
        deps.toast.show('error', r.messageKo, {
          detail: '모달을 닫고 다시 열면 최신 내용을 볼 수 있습니다',
        });
      } else {
        deps.toast.show('error', '공정을 저장하지 못했습니다', { detail: r.messageKo });
      }
    };

    adoptModal(modal);
  };

  /** 상세 모달 → removeProcess에 넘길 EntityMeta 최소 형태 (삭제 토스트 라벨용) */
  const envMeta = (env: RecordEnvelope<ProcessDoc>): EntityMeta => ({
    id: env.doc.id,
    name: env.doc.name,
    meta: env.meta,
    taskSummary: null,
    processId: null,
  });

  // 초기 로드
  refresh();

  return {
    refresh,
    dispose: (): void => {
      disposed = true;
      loadSeq += 1;
      closeActiveModal();
      search.dispose();
      announcer.dispose();
      root.remove();
    },
  };
}
