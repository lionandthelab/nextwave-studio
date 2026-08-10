// ui/console/blocks-screen.ts — 재사용 블록 화면 (콘솔 평면, docs/BACKEND.md Phase 12+)
//
// 팀이 저장한 재사용 블록(BlockDoc)을 카드 그리드로 보여 주고, 미리보기(step kind
// 체인 — flow-graph/node-render의 kindMeta 색·라벨 재사용)·이름/설명 인라인 편집·
// soft-delete(+실행취소 토스트)·[스튜디오에서 삽입]을 제공한다.
//
// ── 전개(inline expansion) 모델과의 관계 ────────────────────────────
// 이 화면은 블록을 **선택**하는 곳까지만 책임진다 — onInsertBlock(blockId) 콜백으로
// 통합자에게 넘기면, 통합자가 파라미터 다이얼로그를 띄우고 schema/blocks.ts의
// expandBlock으로 전개해 그래프에 삽입한다(검증 통과분만 — §2.8). 블록 원본을 고쳐도
// 이미 전개된 작업은 바뀌지 않는다(entities.ts 복사본 의미론).
//
// ── 계층/시각 규칙 ─────────────────────────────────────────────────
// deps는 좁은 인터페이스(BlocksResource + 콜백)만 받는다 — core/main import 금지.
// 시각은 theme.ts 토큰 + console/primitives만 소비한다. 목록 API(EntityMeta)에는
// step 수·robotHint가 없으므로 카드 상세는 get(id)로 채운다 — 실패한 개별 항목은
// 화면 전체를 죽이지 않고 "상세 없음" 카드로 강등된다.
//
// ── 오프라인/로컬 모드 (BACKEND §6) ────────────────────────────────
// 로컬 모드(서버 미설정)면 블록 저장소 자체가 없다 — 빈 상태로 사유를 안내한다.
// 오프라인이면 저장/삭제 버튼을 비활성하고 **title로 이유를 밝힌다**(회색 버튼에
// 이유 없는 침묵 금지). 파괴적 동작(삭제)은 soft-delete + 실행취소 토스트다(§2.11).
//
// 순수 헬퍼(카드 모델·검색 텍스트·연결 사유·미리보기 체인)는 DOM 없이 node 환경에서
// 테스트된다(blocks-screen.test.ts — primitives.test.ts와 같은 관례).

import type { ConnectionState, GetResult, ListResult, RemoveResult, SaveResult } from '../../api';
import { connectionLabelKo } from '../../api';
import type { BlockDoc, BlockParam, RecordEnvelope } from '../../schema/entities';
import { kindMeta } from '../flow-graph/node-render';
import type { ToastHandle } from '../feedback/toast';
import { icon, makeIconButton } from '../icons';
import {
  BORDER,
  BORDER_WIDTH,
  COLOR,
  ICON,
  RADIUS,
  SPACE,
  SURFACE,
  TYPE,
  applyType,
  ensureThemeStyles,
  makeButton,
  styled,
} from '../theme';
import {
  applyTouchTarget,
  ensureConsoleStyles,
  filterRowsByQuery,
  makeBadge,
  makeCard,
  makeCardGrid,
  makeEmptyState,
  makeModalShell,
  makeSearchField,
} from './primitives';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4) ─────────────────────────────

/** 미리보기 모달 폭 — step 체인이 한눈에 들어오는 넓이 */
const PREVIEW_MODAL_WIDTH_PX = 560;
/** 미리보기 체인 칩의 범주 색 스트립 두께 (node-render 좌측 스트립과 동일 의미) */
const CHAIN_CHIP_STRIP_PX = 3;
/** 설명 textarea 행 수 */
const DESCRIPTION_ROWS = 3;
/** 이름 최대 길이 (entities.ts displayNameSchema.max와 동일) */
const NAME_MAX_CHARS = 80;
/** 설명 최대 길이 (entities.ts blockDocSchema.descriptionKo max와 동일) */
const DESCRIPTION_MAX_CHARS = 500;

/** 로컬 모드 빈 상태 안내 (사유 — 회색 화면에 이유 없는 침묵 금지) */
export const LOCAL_MODE_BLOCKS_HINT_KO =
  '서버에 연결되면 팀이 저장한 재사용 블록이 여기 표시됩니다.';

// ── deps (좁은 인터페이스 — 통합자가 배선) ──────────────────────────

/** blocks 개체 클라이언트의 부분집합 — api/resources.ts EntityClient<'blocks'>가 만족한다 */
export interface BlocksResource {
  list(opts?: { readonly q?: string; readonly includeDeleted?: boolean }): Promise<ListResult>;
  get(id: string): Promise<GetResult<BlockDoc>>;
  update(id: string, doc: BlockDoc, baseVersion: number): Promise<SaveResult<BlockDoc>>;
  remove(id: string): Promise<RemoveResult>;
  restore(id: string): Promise<GetResult<BlockDoc>>;
}

export interface BlocksScreenDeps {
  readonly blocks: BlocksResource;
  /** [스튜디오에서 삽입] — 파라미터 다이얼로그·expandBlock·그래프 삽입은 통합자 몫 */
  onInsertBlock(blockId: string): void;
  /** 현재 연결 상태 — refresh/동작 시점마다 재평가한다 */
  connection(): ConnectionState;
  /** 전역 토스트 (삭제 실행취소 등 — CLAUDE.md §2.11 되돌릴 경로) */
  readonly toast: Pick<ToastHandle, 'show'>;
}

export interface BlocksScreenHandle {
  refresh(): void;
  dispose(): void;
}

// ── 순수 헬퍼 (DOM 비의존 — node 테스트 대상) ───────────────────────

export interface BlockCardModel {
  readonly id: string;
  readonly name: string;
  readonly descriptionKo: string;
  /** null = 상세(get) 로드 실패 — 카드가 "상세 없음"으로 강등된다 */
  readonly stepCount: number | null;
  readonly robotHint: string | null;
  readonly paramCount: number;
  /** step kind 체인 (미리보기 표시 순서 그대로) */
  readonly kinds: readonly string[];
}

/** 상세 봉투 → 카드 모델 */
export function blockCardModel(record: RecordEnvelope<BlockDoc>): BlockCardModel {
  return {
    id: record.doc.id,
    name: record.doc.name,
    descriptionKo: record.doc.descriptionKo,
    stepCount: record.doc.steps.length,
    robotHint: record.doc.robotHint,
    paramCount: record.doc.params.length,
    kinds: record.doc.steps.map((step) => step.kind),
  };
}

/** get(id) 실패 시 목록 메타만으로 만드는 강등 카드 모델 */
export function degradedCardModel(id: string, name: string): BlockCardModel {
  return {
    id,
    name,
    descriptionKo: '',
    stepCount: null,
    robotHint: null,
    paramCount: 0,
    kinds: [],
  };
}

/** 카드 보조행 — 도메인 식별자 조합이라 lang="en"으로 렌더된다 */
export function blockSubline(stepCount: number | null, robotHint: string | null): string {
  const parts: string[] = [];
  if (stepCount !== null) parts.push(`${stepCount} steps`);
  if (robotHint !== null && robotHint !== '') parts.push(robotHint);
  return parts.join(' · ');
}

/** 검색 대상 텍스트 (이름 + 설명 + robotHint) */
export function blockSearchText(model: BlockCardModel): string {
  return `${model.name} ${model.descriptionKo} ${model.robotHint ?? ''}`;
}

/**
 * 저장/삭제가 불가능한 사유 (null = 가능). 회색 버튼에는 반드시 이 문자열이 title로
 * 붙는다 — 이유 없는 비활성은 사용자를 시행착오로 내몬다.
 */
export function writeDisabledReasonKo(state: ConnectionState): string | null {
  if (state.mode === 'local') {
    return '서버가 설정되지 않아(로컬 모드) 블록을 저장/삭제할 수 없습니다';
  }
  if (!state.online) {
    return '오프라인 상태입니다 — 서버에 다시 연결되면 저장/삭제할 수 있습니다';
  }
  return null;
}

/** 연결 배지 상태 — 정상만 success, 그 외(오프라인·로컬)는 warn (색+텍스트 병행) */
export function connectionBadgeStatus(state: ConnectionState): 'success' | 'warn' {
  return state.mode === 'server' && state.online ? 'success' : 'warn';
}

export interface PreviewChainItem {
  readonly kind: string;
  /** kindMeta 표시명 (PascalCase — lang="en"으로 렌더) */
  readonly label: string;
  /** kindMeta 범주 색 (CATEGORY 토큰 — 색을 여기서 발명하지 않는다) */
  readonly color: string;
}

/** step kind 체인 → 미리보기 칩 데이터 (알 수 없는 kind도 kindMeta가 안전 처리) */
export function previewChain(kinds: readonly string[]): PreviewChainItem[] {
  return kinds.map((kind) => {
    const meta = kindMeta(kind);
    return { kind, label: meta.label, color: meta.color };
  });
}

// ── 마운트 ──────────────────────────────────────────────────────────

export function mountBlocksScreen(host: HTMLElement, deps: BlocksScreenDeps): BlocksScreenHandle {
  ensureThemeStyles();
  ensureConsoleStyles();

  // ── 상태 (ui 소유 — 뷰 상태만) ──────────────────────────────────
  let query = '';
  let cardModels: BlockCardModel[] | null = null;
  let loadErrorKo: string | null = null;
  let loading = false;
  let loadSeq = 0;
  let conn: ConnectionState = deps.connection();

  // ── 루트 ────────────────────────────────────────────────────────
  const root = applyType(document.createElement('section'), TYPE.body);
  styled(root, {
    display: 'flex',
    flexDirection: 'column',
    gap: SPACE.lg,
    width: '100%',
    height: '100%',
    minHeight: '0',
    boxSizing: 'border-box',
    padding: SPACE.xl,
    overflowY: 'auto',
    color: COLOR.text,
  });
  root.className = 'ui-scroll';
  root.dataset.testid = 'blocks-screen';
  root.setAttribute('aria-label', '블록');

  // ── 툴바: 제목 + 연결 배지 + 개수 + 검색 + 새로 고침 ─────────────
  const toolbar = styled(document.createElement('div'), {
    display: 'flex',
    alignItems: 'center',
    flexWrap: 'wrap',
    gap: SPACE.md,
    flex: 'none',
  });

  const title = applyType(document.createElement('h2'), TYPE.display);
  styled(title, { margin: '0', color: COLOR.textStrong, flex: 'none' });
  title.textContent = '블록';
  toolbar.appendChild(title);

  const connBadgeHost = styled(document.createElement('span'), { display: 'inline-flex' });
  toolbar.appendChild(connBadgeHost);

  const countBadgeHost = styled(document.createElement('span'), { display: 'inline-flex' });
  toolbar.appendChild(countBadgeHost);

  const search = makeSearchField({
    placeholderKo: '블록 검색…',
    onInput: (q): void => {
      query = q;
      renderContent();
    },
    testid: 'blocks-search',
  });
  styled(search.el, { flex: '1 1 240px', minWidth: '180px', maxWidth: '420px' });
  toolbar.appendChild(search.el);

  const refreshButton = makeIconButton('refresh', '새로 고침', '블록 목록 새로 고침', 'blocks-refresh', 'ghost');
  applyTouchTarget(refreshButton);
  refreshButton.addEventListener('click', () => {
    void load();
  });
  toolbar.appendChild(refreshButton);
  root.appendChild(toolbar);

  // ── 본문 (그리드 / 빈 상태 / 오류) ──────────────────────────────
  const contentHost = styled(document.createElement('div'), {
    flex: '1 1 auto',
    minHeight: '0',
  });
  root.appendChild(contentHost);

  const grid = makeCardGrid({ testid: 'blocks-grid' });

  // ── 미리보기 모달 (이름/설명 인라인 편집 + step 체인 + 삽입) ─────
  const modal = makeModalShell({
    titleKo: '블록 미리보기',
    onClose: (): void => {
      /* 열람 상태는 buildPreview가 매번 새로 만든다 — 정리할 잔여 없음 */
    },
    widthPx: PREVIEW_MODAL_WIDTH_PX,
    testid: 'blocks-preview',
  });
  root.appendChild(modal.root);

  // ── 렌더 헬퍼 ───────────────────────────────────────────────────

  const renderConnBadge = (): void => {
    connBadgeHost.replaceChildren(
      makeBadge(connectionLabelKo(conn), connectionBadgeStatus(conn), {
        testid: 'blocks-conn-badge',
      }),
    );
  };

  const renderCountBadge = (): void => {
    countBadgeHost.replaceChildren();
    if (cardModels !== null) {
      countBadgeHost.appendChild(makeBadge(`${cardModels.length}개`, 'neutral'));
    }
  };

  const mutedLine = (text: string): HTMLElement => {
    const el = applyType(document.createElement('div'), TYPE.body);
    styled(el, { color: COLOR.muted, padding: SPACE.lg });
    el.textContent = text;
    return el;
  };

  const makeCardEl = (model: BlockCardModel): HTMLElement => {
    const writeReason = writeDisabledReasonKo(conn);

    const insertButton = makeButton(
      '스튜디오에서 삽입',
      `'${model.name}' 블록을 스튜디오 플로우 그래프에 삽입`,
      `block-insert-${model.id}`,
      'primary',
    );
    applyTouchTarget(insertButton);
    styled(insertButton, { flex: '1 1 auto' });
    insertButton.addEventListener('click', () => {
      deps.onInsertBlock(model.id);
    });

    const deleteButton = makeIconButton(
      'trash',
      '',
      `'${model.name}' 블록 삭제`,
      `block-delete-${model.id}`,
      'ghost',
    );
    applyTouchTarget(deleteButton, { square: true });
    if (writeReason !== null) {
      deleteButton.disabled = true;
      deleteButton.title = writeReason;
    }
    deleteButton.addEventListener('click', () => {
      void doDelete(model);
    });

    const sublines: ({ text: string; lang?: string } | string)[] = [];
    if (model.descriptionKo !== '') sublines.push(model.descriptionKo);
    const subline = blockSubline(model.stepCount, model.robotHint);
    if (subline !== '') sublines.push({ text: subline, lang: 'en' });
    if (model.stepCount === null) sublines.push('상세 정보를 불러오지 못했습니다');

    return makeCard({
      title: model.name,
      sublines,
      badge:
        model.paramCount > 0
          ? makeBadge(`파라미터 ${model.paramCount}`, 'neutral', {
              testid: `block-params-badge-${model.id}`,
            })
          : undefined,
      onClick: (): void => {
        void openPreview(model.id);
      },
      actions: [insertButton, deleteButton],
      testid: `block-card-${model.id}`,
    });
  };

  const renderContent = (): void => {
    renderConnBadge();
    renderCountBadge();
    contentHost.replaceChildren();

    if (conn.mode === 'local') {
      contentHost.appendChild(
        makeEmptyState({
          iconName: 'cloudOff',
          titleKo: '로컬 모드 — 블록 저장소 없음',
          hintKo: LOCAL_MODE_BLOCKS_HINT_KO,
          actions: [],
          testid: 'blocks-empty-local',
        }),
      );
      return;
    }

    if (loadErrorKo !== null) {
      const retry = makeButton('다시 시도', '블록 목록 다시 불러오기', 'blocks-retry', 'primary');
      retry.addEventListener('click', () => {
        void load();
      });
      contentHost.appendChild(
        makeEmptyState({
          iconName: 'alert',
          titleKo: '블록 목록을 불러오지 못했습니다',
          hintKo: loadErrorKo,
          actions: [retry],
          testid: 'blocks-empty-error',
        }),
      );
      return;
    }

    if (cardModels === null) {
      if (loading) contentHost.appendChild(mutedLine('블록 불러오는 중…'));
      return;
    }

    if (cardModels.length === 0) {
      contentHost.appendChild(
        makeEmptyState({
          iconName: 'puzzle',
          titleKo: '저장된 블록이 없습니다',
          hintKo: '스튜디오의 플로우 그래프에서 step 묶음을 블록으로 저장하면 여기 나타납니다.',
          actions: [],
          testid: 'blocks-empty',
        }),
      );
      return;
    }

    const filtered = filterRowsByQuery(cardModels, query, blockSearchText);
    if (filtered.length === 0) {
      contentHost.appendChild(
        makeEmptyState({
          iconName: 'search',
          titleKo: '검색 결과 없음',
          hintKo: '다른 검색어로 다시 시도해 보세요.',
          actions: [],
          testid: 'blocks-empty-search',
        }),
      );
      return;
    }

    grid.setCards(filtered.map(makeCardEl));
    contentHost.appendChild(grid.el);
  };

  // ── 데이터 로드 ─────────────────────────────────────────────────

  const load = async (): Promise<void> => {
    conn = deps.connection();
    if (conn.mode === 'local') {
      cardModels = null;
      loadErrorKo = null;
      loading = false;
      renderContent();
      return;
    }
    const mySeq = ++loadSeq;
    loading = true;
    loadErrorKo = null;
    renderContent();

    const listed = await deps.blocks.list();
    if (mySeq !== loadSeq) return; // 최신 요청이 이겼다 — 낡은 응답 폐기
    if (listed.kind !== 'ok') {
      loading = false;
      cardModels = null;
      loadErrorKo = listed.messageKo;
      conn = deps.connection(); // 실패로 오프라인 전환됐을 수 있다 — 배지 최신화
      renderContent();
      return;
    }

    // 목록 메타에는 step 수/robotHint가 없다 — 상세를 병렬로 채운다 (파일 헤더)
    const items = listed.items.filter((item) => item.meta.deletedAtIso === null);
    const details = await Promise.all(
      items.map(async (item) => ({ item, got: await deps.blocks.get(item.id) })),
    );
    if (mySeq !== loadSeq) return;
    loading = false;
    cardModels = details.map(({ item, got }) =>
      got.kind === 'ok' ? blockCardModel(got.record) : degradedCardModel(item.id, item.name),
    );
    conn = deps.connection();
    renderContent();
  };

  // ── 삭제 (soft + 실행취소 토스트 — §2.11 되돌릴 경로) ────────────

  const doDelete = async (model: BlockCardModel): Promise<void> => {
    const reason = writeDisabledReasonKo(deps.connection());
    if (reason !== null) {
      deps.toast.show('warn', reason);
      return;
    }
    const removed = await deps.blocks.remove(model.id);
    if (removed.kind !== 'ok') {
      deps.toast.show('error', removed.messageKo);
      return;
    }
    deps.toast.show('info', `'${model.name}' 블록을 삭제했습니다`, {
      detail: '휴지통에서 30일간 보관됩니다',
      action: {
        label: '실행 취소',
        onClick: (): void => {
          void undoDelete(model);
        },
      },
    });
    void load();
  };

  const undoDelete = async (model: BlockCardModel): Promise<void> => {
    const restored = await deps.blocks.restore(model.id);
    if (restored.kind === 'ok') {
      deps.toast.show('success', `'${model.name}' 블록을 복원했습니다`);
    } else {
      deps.toast.show('error', restored.messageKo);
    }
    void load();
  };

  // ── 미리보기 모달 (이름/설명 편집 + step 체인 + 파라미터 + 삽입) ──

  const fieldLabel = (text: string): HTMLElement => {
    const el = applyType(document.createElement('div'), TYPE.caption);
    styled(el, { color: COLOR.label });
    el.textContent = text;
    return el;
  };

  const chainChip = (item: PreviewChainItem): HTMLElement => {
    const chip = applyType(document.createElement('span'), TYPE.caption);
    styled(chip, {
      display: 'inline-flex',
      alignItems: 'center',
      padding: `${SPACE.xs} ${SPACE.sm}`,
      background: SURFACE.raised,
      border: `${BORDER_WIDTH.hair} solid ${BORDER.subtle}`,
      borderLeft: `${CHAIN_CHIP_STRIP_PX}px solid ${item.color}`,
      borderRadius: RADIUS.sm,
      color: COLOR.text,
      whiteSpace: 'nowrap',
    });
    chip.setAttribute('lang', 'en');
    chip.textContent = item.label;
    return chip;
  };

  const buildPreview = (record: RecordEnvelope<BlockDoc>): void => {
    const doc = record.doc;
    let baseVersion = record.meta.version;
    let currentDoc = doc;
    let dirty = false;

    modal.body.replaceChildren();
    modal.footer.replaceChildren();

    // 이름 (인라인 편집)
    modal.body.appendChild(fieldLabel('이름'));
    const nameInput = document.createElement('input');
    nameInput.type = 'text';
    nameInput.className = 'ui-input';
    nameInput.value = doc.name;
    nameInput.maxLength = NAME_MAX_CHARS;
    nameInput.setAttribute('aria-label', '블록 이름');
    nameInput.dataset.testid = 'blocks-preview-name';
    modal.body.appendChild(nameInput);

    // 설명 (인라인 편집)
    modal.body.appendChild(fieldLabel('설명'));
    const descInput = document.createElement('textarea');
    descInput.className = 'ui-input';
    descInput.rows = DESCRIPTION_ROWS;
    descInput.value = doc.descriptionKo;
    descInput.maxLength = DESCRIPTION_MAX_CHARS;
    descInput.setAttribute('aria-label', '블록 설명');
    descInput.dataset.testid = 'blocks-preview-desc';
    styled(descInput, { resize: 'vertical', fontFamily: 'inherit' });
    modal.body.appendChild(descInput);

    // 메타 (수정자 · 버전 — 다중 사용자의 "누가 언제")
    const metaLine = applyType(document.createElement('div'), TYPE.caption);
    styled(metaLine, { color: COLOR.muted });
    const robotHintText = doc.robotHint !== null && doc.robotHint !== '' ? ` · ${doc.robotHint}` : '';
    metaLine.textContent = `수정: ${record.meta.updatedByName} · v${record.meta.version}${robotHintText}`;
    modal.body.appendChild(metaLine);

    // step 체인 (kindMeta 색·라벨 재사용 — 노드 그래프와 같은 시각 어휘)
    modal.body.appendChild(fieldLabel(`step 체인 (${doc.steps.length}개)`));
    const chainRow = styled(document.createElement('div'), {
      display: 'flex',
      alignItems: 'center',
      flexWrap: 'wrap',
      gap: SPACE.sm,
    });
    chainRow.dataset.testid = 'blocks-preview-chain';
    const chain = previewChain(doc.steps.map((step) => step.kind));
    chain.forEach((item, index) => {
      if (index > 0) {
        const sep = styled(document.createElement('span'), {
          display: 'inline-flex',
          color: COLOR.muted,
        });
        sep.setAttribute('aria-hidden', 'true');
        sep.appendChild(icon('chevronRight', ICON.sm));
        chainRow.appendChild(sep);
      }
      chainRow.appendChild(chainChip(item));
    });
    modal.body.appendChild(chainRow);

    // 파라미터 (삽입 시 채우는 인자 — 값 편집은 삽입 다이얼로그 몫)
    if (doc.params.length > 0) {
      modal.body.appendChild(fieldLabel(`파라미터 (${doc.params.length}개)`));
      const paramsCol = styled(document.createElement('div'), {
        display: 'flex',
        flexDirection: 'column',
        gap: SPACE.sm,
      });
      for (const param of doc.params as readonly BlockParam[]) {
        const row = styled(document.createElement('div'), {
          display: 'flex',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: SPACE.sm,
        });
        const keyBadge = makeBadge(param.key, 'neutral');
        keyBadge.setAttribute('lang', 'en');
        row.appendChild(keyBadge);
        const labelSpan = applyType(document.createElement('span'), TYPE.caption);
        styled(labelSpan, { color: COLOR.text });
        labelSpan.textContent = param.labelKo;
        row.appendChild(labelSpan);
        const kindSpan = applyType(document.createElement('span'), TYPE.caption);
        styled(kindSpan, { color: COLOR.muted });
        kindSpan.setAttribute('lang', 'en');
        kindSpan.textContent = param.kind;
        row.appendChild(kindSpan);
        const defaultSpan = applyType(document.createElement('span'), TYPE.caption);
        styled(defaultSpan, { color: COLOR.muted });
        defaultSpan.textContent = `기본값 ${String(param.defaultValue)}`;
        row.appendChild(defaultSpan);
        paramsCol.appendChild(row);
      }
      modal.body.appendChild(paramsCol);
    }

    // ── footer: 저장(dirty 시) + 스튜디오에서 삽입 ─────────────────
    const saveButton = makeButton('저장', '이름/설명 변경 저장', 'blocks-preview-save', 'default');
    applyTouchTarget(saveButton);

    const syncSave = (): void => {
      const reason = writeDisabledReasonKo(deps.connection());
      saveButton.disabled = !dirty || reason !== null;
      saveButton.title = reason ?? '이름/설명 변경 저장';
    };
    syncSave();

    const markDirty = (): void => {
      dirty = true;
      modal.setDirty(true);
      syncSave();
    };
    nameInput.addEventListener('input', markDirty);
    descInput.addEventListener('input', markDirty);

    saveButton.addEventListener('click', () => {
      void (async (): Promise<void> => {
        const name = nameInput.value.trim();
        if (name === '') {
          deps.toast.show('error', '블록 이름이 비어 있습니다');
          nameInput.focus();
          return;
        }
        const nextDoc: BlockDoc = { ...currentDoc, name, descriptionKo: descInput.value };
        const saved = await deps.blocks.update(doc.id, nextDoc, baseVersion);
        if (saved.kind === 'ok') {
          baseVersion = saved.record.meta.version;
          currentDoc = saved.record.doc;
          dirty = false;
          modal.setDirty(false);
          syncSave();
          deps.toast.show('success', `'${name}' 블록을 저장했습니다`);
          void load();
          return;
        }
        if (saved.kind === 'conflict') {
          // 자동 병합 금지 (BACKEND §6) — 사용자가 서버본을 보고 결정한다
          deps.toast.show('error', saved.messageKo, {
            action: {
              label: '서버본 불러오기',
              icon: 'refresh',
              onClick: (): void => {
                void openPreview(doc.id);
              },
            },
          });
          return;
        }
        deps.toast.show('error', saved.messageKo);
      })();
    });

    const insertButton = makeButton(
      '스튜디오에서 삽입',
      `'${doc.name}' 블록을 스튜디오 플로우 그래프에 삽입`,
      'blocks-preview-insert',
      'primary',
    );
    applyTouchTarget(insertButton);
    insertButton.addEventListener('click', () => {
      modal.close(); // 프로그램적 닫기 — dirty 확인 없이 (편집은 저장 버튼이 소유)
      deps.onInsertBlock(doc.id);
    });

    modal.footer.appendChild(saveButton);
    modal.footer.appendChild(insertButton);
  };

  const openPreview = async (id: string): Promise<void> => {
    const got = await deps.blocks.get(id);
    if (got.kind !== 'ok') {
      deps.toast.show('error', got.messageKo);
      return;
    }
    buildPreview(got.record);
    modal.open();
  };

  // ── 초기 로드 + 핸들 ────────────────────────────────────────────

  host.appendChild(root);
  void load();

  return {
    refresh: (): void => {
      void load();
    },
    dispose: (): void => {
      loadSeq += 1; // 진행 중 응답 폐기
      search.dispose();
      modal.dispose();
      grid.clear();
      root.remove();
    },
  };
}
