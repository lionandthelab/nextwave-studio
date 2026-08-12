// ui/library/library.ts — 라이브러리 패널 (UX_DESIGN §3.2: Objects/Robots 카드 + 검색)
//
// 템플릿 카드를 섹션(Objects/Robots)으로 나눠 그리드로 보여 주고, 두 가지 배치 경로를
// 제공한다 (UX §4.2 Flow 2):
//   (a) 카드 클릭/Enter → deps.onPlace(spec, null)
//       — 통합자가 뷰포트 중앙 레이캐스트 지점에 배치한다.
//   (b) HTML5 드래그 → dataTransfer에 "템플릿 키"만 싣는다 (아래 드롭 계약).
//
// ── 드래그앤드롭 계약 (통합자 뷰포트 측 구현 규범) ───────────────────
// - MIME: TEMPLATE_MIME('application/x-robotsim-template'), 값 = LibraryTemplate.key.
//   ('text/plain'에도 같은 키를 실어 외부 드롭 대상 호환을 유지한다.)
// - 뷰포트(통합자)가 dragover(preventDefault + dropEffect='copy')와 drop을 등록한다:
//   drop 시 getData(TEMPLATE_MIME) → templateByKey(key) → template.create(uniquify)
//   → onPlace(spec, { x: e.clientX, y: e.clientY }) — dropClient를 바닥 레이캐스트로
//   변환해 배치하는 것은 뷰포트 몫이다 (UX §3.3 "라이브러리 드래그 프리뷰").
//
// ── 드래그 피드백 (UX_AUDIT C-17) ────────────────────────────────────
// 이전 헤더 주석은 "별도 프리뷰는 뷰포트의 반투명 고스트가 담당"이라며 책임을 넘겼지만
// **뷰포트에 그 구현이 없었다** — 카드를 끄는 동안 화면에 아무 일도 일어나지 않았다.
// 이제 `deps.onDragState(active, label)`로 드래그 시작/종료를 통지한다: 통합자가 이를
// `viewport/drop-hint.ts`의 가장자리 하이라이트에 연결하면 "놓을 수 있다"는 어포던스가
// 즉시 생긴다. **바닥 레이캐스트 지점의 반투명 고스트는 여전히 render 계층 몫**이다
// (ui는 three를 모른다 — CLAUDE.md §3). 카드 요소 자체는 기본 drag image로 쓰인다.
//
// 계층 규칙 (CLAUDE.md §3): core/render를 모른다. 엔티티 생성(spec)은 templates.ts의
// 순수 팩토리가, 실제 씬 편입은 통합자(SceneEditor)가 담당한다. id 유일화(uniquify)도
// 씬 상태를 아는 통합자가 주입한다.
//
// 접근성 (UX §9): 카드는 <button>이라 Tab 포커스·Enter/Space 배치가 기본 동작이고,
// 설명은 title(호버)과 aria-label로 병행 노출된다. 섹션 접기 토글은 aria-expanded.

import { LIBRARY_TEMPLATES, filterTemplates } from './templates';
import type { LibraryTemplate, TemplateSection } from './templates';
import {
  BORDER,
  BORDER_WIDTH,
  COLOR,
  ICON,
  MOTION,
  RADIUS,
  SPACE,
  SURFACE,
  TYPE,
  applyType,
  ensureThemeStyles,
  makeButton,
  makePanelHeader,
  styled,
  tr,
} from '../theme';
import { icon } from '../icons';
import type { EntitySpec } from '../../schema';

// ── 드래그 페이로드 MIME (통합자 뷰포트 드롭 핸들러와의 계약) ───────

export const TEMPLATE_MIME = 'application/x-robotsim-template';

// ── 공개 타입 ───────────────────────────────────────────────────────

/**
 * 블록 섹션 카드 요약 (Phase 12+ 재사용 블록 — schema/entities.ts BlockDoc의 표시
 * 부분집합). 목록 데이터의 진실은 통합자(blocks API)가 소유하고 이 패널은 표시만 한다.
 */
export interface LibraryBlockSummary {
  readonly id: string;
  readonly name: string;
  readonly stepCount: number;
  readonly robotHint: string | null;
}

export interface LibraryDeps {
  /**
   * 템플릿 배치 요청. dropClient가 null이면 카드 클릭/Enter 경로 — 통합자가 뷰포트
   * 중앙 레이캐스트로 위치를 정한다. 드롭 경로에서는 통합자(뷰포트)가 좌표와 함께
   * 직접 호출한다 (파일 헤더의 드롭 계약).
   */
  onPlace(spec: EntitySpec, dropClient: { x: number; y: number } | null): void;
  /** idBase → 씬-유일 id (예: 'box' → 'box_2') — 유일성 진실은 통합자/SceneEditor */
  uniquify(base: string): string;
  /**
   * Import 섹션의 카드 클릭 (UX §3.2 "Import: 카드 클릭 = 파일 선택").
   * 파일 선택기/임포트 다이얼로그는 통합자 소유다. 미주입 시 Import 섹션은 렌더되지
   * 않는다(헤드리스/테스트 호환).
   */
  onImportRequest?(): void;
  /**
   * 카드 드래그 시작/종료 통지 (UX_AUDIT C-17 — 파일 헤더 "드래그 피드백").
   * label은 끌고 있는 템플릿 이름(종료 시 null). 통합자가 뷰포트 드롭 힌트에 연결한다.
   */
  onDragState?(active: boolean, label: string | null): void;
  /**
   * 재사용 블록 목록 공급자 (Phase 12+ — 삽입 시 전개 모델). **미주입 시 블록 섹션은
   * 렌더되지 않는다** — 기존 호출부는 수정 없이 그대로 동작한다.
   */
  blocksProvider?(): Promise<readonly LibraryBlockSummary[]>;
  /**
   * 블록 카드 클릭 = 삽입 요청. 파라미터 다이얼로그·expandBlock·그래프 삽입은 통합자
   * 몫이다 (schema/blocks.ts — 이 패널은 블록을 해석하지 않는다). 블록은 씬 배치물이
   * 아니라 시퀀스 조각이므로 템플릿 카드와 달리 뷰포트 드래그를 지원하지 않는다.
   */
  onInsertBlock?(id: string): void;
}

export interface LibraryHandle {
  /** 패널 루트 (host를 채운다) — 통합자 재배치용 */
  readonly el: HTMLElement;
  /** 검색어 프로그램 설정 (빈 문자열 = 전체 표시) */
  setSearch(query: string): void;
  /** 블록 목록 다시 읽기 (블록 저장/삭제 후 통합자가 호출) — blocksProvider 미주입 시 no-op */
  reloadBlocks(): void;
  dispose(): void;
}

// ── 섹션 정의 (UX §3.2 — Import 섹션은 3D 임포트 다이얼로그 소유자 몫) ──
//
// 라벨의 영/한 이중 표기는 **라이브러리 카드 한정 예외**다 (UX_AUDIT C-18 언어 정책):
// 패널 크롬은 한국어로 통일하되, 프리미티브 이름(Box/Sphere)은 3D 도메인 식별자라
// 영문을 유지하고 한국어를 병기한다.

/**
 * 섹션 순서 — **로봇이 먼저다.**
 *
 * 사물(7종)을 앞에 두면 로봇 3종이 패널 밖으로 밀린다(실측: 1440×900에서 카드 10장 중
 * 로봇 3장이 전부 잘림 — 패널 bottom 548px, 로봇 카드 y=547·617). 로봇 셀을 세우는
 * 도구인데 **로봇이 화면에 하나도 없는** 상태로 첫 화면이 열렸고, 설치기사의 첫 행동이
 * 정확히 "로봇 놓기"다. 스크롤하면 나오지만, 첫 화면에 없는 것은 없는 것이다.
 */
const SECTIONS: ReadonlyArray<{ readonly id: TemplateSection; readonly labelKo: string }> = [
  { id: 'robots', labelKo: 'Robots · 로봇' },
  { id: 'objects', labelKo: 'Objects · 사물' },
];

/** 카드 그리드 최소 열 폭 (auto-fill) */
const CARD_MIN_WIDTH_PX = 88;

// ── 패널 전용 스타일 (카드 hover/포커스/드래그 — 토큰만 소비) ───────

const LIBRARY_STYLE_ID = 'rsw-library-styles';

function ensureLibraryStyles(): void {
  if (document.getElementById(LIBRARY_STYLE_ID) !== null) return;
  const style = document.createElement('style');
  style.id = LIBRARY_STYLE_ID;
  style.textContent = `
.rsw-lib-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: ${SPACE.sm};
  padding: ${SPACE.md} ${SPACE.xs} ${SPACE.sm} ${SPACE.xs};
  background: ${SURFACE.raised};
  color: ${COLOR.text};
  border: ${BORDER_WIDTH.hair} solid ${BORDER.strong};
  border-radius: ${RADIUS.sm};
  font-family: var(--rsw-font-ui);
  font-size: ${TYPE.caption.sizePx}px;
  line-height: ${TYPE.caption.lineHeightPx}px;
  font-weight: ${TYPE.caption.weight};
  cursor: grab;
  user-select: none;
  transition: ${tr('background-color', MOTION.instant)}, ${tr('border-color', MOTION.instant)}, ${tr('color', MOTION.instant)};
}
.rsw-lib-card:hover {
  background: ${SURFACE.overlay};
  border-color: ${BORDER.hover};
  color: ${COLOR.textStrong};
}
.rsw-lib-card:focus-visible {
  outline: 2px solid var(--rsw-accent);
  outline-offset: 1px;
}
.rsw-lib-card:active { cursor: grabbing; }
.rsw-lib-card--dragging { opacity: 0.5; border-color: var(--rsw-select); }
/* 아이콘 크기는 SVG의 width/height 속성(ICON.xl)이 소유한다 — font-size로 크기를
   정하던 이모지 시절의 잔재를 없앤다 (UX_AUDIT C-13). 색만 여기서 준다. */
.rsw-lib-card__icon {
  display: flex;
  align-items: center;
  justify-content: center;
  color: ${COLOR.label};
}
.rsw-lib-card:hover .rsw-lib-card__icon { color: ${COLOR.textStrong}; }
.rsw-lib-card__label {
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.rsw-lib-section {
  display: flex;
  align-items: center;
  gap: ${SPACE.sm};
  width: 100%;
  margin: ${SPACE.xl} 0 ${SPACE.sm} 0;
  padding: ${SPACE.xxs} ${SPACE.xs};
  text-align: left;
}
.rsw-lib-section > svg { transition: ${tr('transform', MOTION.fast)}; }
.rsw-lib-section--collapsed > svg { transform: rotate(-90deg); }
`;
  document.head.appendChild(style);
}

// ── 순수 헬퍼 ───────────────────────────────────────────────────────

/** 검색어로 블록 요약 필터 (이름·robotHint 부분 일치, 대소문자 무시 — filterTemplates와 동형) */
export function filterBlockSummaries(
  items: readonly LibraryBlockSummary[],
  query: string,
): readonly LibraryBlockSummary[] {
  const q = query.trim().toLowerCase();
  if (q === '') return items;
  return items.filter(
    (block) =>
      block.name.toLowerCase().includes(q) ||
      (block.robotHint ?? '').toLowerCase().includes(q),
  );
}

// ── 내부 DOM 헬퍼 ───────────────────────────────────────────────────

/** muted 안내 줄 (빈 검색 결과 등) */
function mutedLine(text: string): HTMLElement {
  const el = applyType(document.createElement('div'), TYPE.body);
  styled(el, {
    color: COLOR.muted,
    padding: `${SPACE.xs} ${SPACE.xxs}`,
  });
  el.textContent = text;
  return el;
}

// ── 마운트 ──────────────────────────────────────────────────────────

/** 라이브러리 패널을 host(워크스페이스 좌 슬롯)에 마운트한다. 패널은 host를 채운다. */
export function mountLibrary(host: HTMLElement, deps: LibraryDeps): LibraryHandle {
  ensureThemeStyles();
  ensureLibraryStyles();

  const panel = applyType(document.createElement('div'), TYPE.body);
  styled(panel, {
    width: '100%',
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
    minHeight: '0',
    color: COLOR.text,
    boxSizing: 'border-box',
  });
  panel.dataset.testid = 'library';
  // 패널 위 포인터/휠이 뷰포트 orbit으로 새지 않게 흡수 (기존 패널 규약 — dock/inspector).
  // stopPropagation은 네이티브 HTML5 drag 동작(별도 drag 이벤트)을 막지 않는다.
  for (const type of ['pointerdown', 'pointermove', 'pointerup', 'wheel', 'contextmenu']) {
    panel.addEventListener(type, (e) => {
      e.stopPropagation();
    });
  }

  // 헤더: 공용 팩토리(UX_AUDIT C-18 — 수제 헤더 8곳 통일) + 검색 (UX §3.2 상단 고정)
  const header = styled(document.createElement('div'), {
    display: 'flex',
    flexDirection: 'column',
    flexShrink: '0',
    borderBottom: `${BORDER_WIDTH.hair} solid ${BORDER.subtle}`,
  });
  const panelHeader = makePanelHeader('라이브러리', {
    headingTag: 'h2',
    collapsible: true,
    testId: 'library-header',
  });
  header.appendChild(panelHeader.el);

  const searchRow = styled(document.createElement('div'), {
    padding: `0 ${SPACE.lg} ${SPACE.md} ${SPACE.lg}`,
  });
  const search = document.createElement('input');
  search.type = 'search';
  search.className = 'ui-input';
  search.placeholder = '템플릿 검색…';
  search.setAttribute('aria-label', '템플릿 검색');
  search.dataset.testid = 'library-search';
  styled(search, { width: '100%', boxSizing: 'border-box' });
  searchRow.appendChild(search);
  header.appendChild(searchRow);
  panel.appendChild(header);

  // 본문 (스크롤 영역): 섹션 + 카드 그리드
  const body = styled(document.createElement('div'), {
    flex: '1 1 auto',
    minHeight: '0',
    overflowY: 'auto',
    padding: `0 ${SPACE.lg} ${SPACE.lg} ${SPACE.lg}`,
  });
  body.classList.add('ui-scroll');
  panel.appendChild(body);

  // ── 뷰 상태 (ui 소유) ─────────────────────────────────────────────
  let query = '';
  const sectionCollapsed = new Map<TemplateSection, boolean>();

  // ── 블록 섹션 상태 (blocksProvider 주입 시에만 사용 — 파일 헤더) ──
  let blockItems: readonly LibraryBlockSummary[] = [];
  let blocksState: 'loading' | 'ready' | 'error' = 'loading';
  let disposed = false;

  const loadBlocks = (): void => {
    const provider = deps.blocksProvider;
    if (provider === undefined) return;
    blocksState = 'loading';
    render();
    provider().then(
      (items) => {
        if (disposed) return;
        blockItems = items;
        blocksState = 'ready';
        render();
      },
      () => {
        if (disposed) return;
        blocksState = 'error';
        render();
      },
    );
  };

  // ── 카드 생성 ─────────────────────────────────────────────────────
  const makeCard = (template: LibraryTemplate): HTMLButtonElement => {
    const card = document.createElement('button');
    card.type = 'button';
    card.className = 'rsw-lib-card';
    card.draggable = true;
    card.dataset.testid = `library-card-${template.key}`;
    card.dataset.templateKey = template.key;
    card.title = template.descriptionKo; // 호버 설명 (UX §3.2 "호버 시 간단 설명")
    card.setAttribute('aria-label', `${template.labelKo} 추가 — ${template.descriptionKo}`);

    const iconWrap = document.createElement('span');
    iconWrap.className = 'rsw-lib-card__icon';
    iconWrap.setAttribute('aria-hidden', 'true');
    iconWrap.appendChild(icon(template.icon, ICON.xl));
    const label = document.createElement('span');
    label.className = 'rsw-lib-card__label';
    label.textContent = template.labelKo;
    card.appendChild(iconWrap);
    card.appendChild(label);

    // (a) 클릭/Enter/Space → 뷰포트 중앙 배치 요청 (<button>이라 키보드는 click으로 온다)
    card.addEventListener('click', () => {
      deps.onPlace(template.create(deps.uniquify), null);
    });

    // (b) 드래그 → 페이로드는 "템플릿 키"만 (spec 생성은 드롭 시점 통합자 몫 — 파일 헤더)
    card.addEventListener('dragstart', (e) => {
      if (!e.dataTransfer) return;
      e.dataTransfer.setData(TEMPLATE_MIME, template.key);
      e.dataTransfer.setData('text/plain', template.key);
      e.dataTransfer.effectAllowed = 'copy';
      card.classList.add('rsw-lib-card--dragging');
      deps.onDragState?.(true, template.labelKo);
    });
    card.addEventListener('dragend', () => {
      card.classList.remove('rsw-lib-card--dragging');
      deps.onDragState?.(false, null);
    });

    return card;
  };

  // ── 블록 카드 (클릭 = 삽입 요청 — 드래그 없음, deps 주석 참조) ────
  const makeBlockCard = (block: LibraryBlockSummary): HTMLButtonElement => {
    const card = document.createElement('button');
    card.type = 'button';
    card.className = 'rsw-lib-card';
    card.dataset.testid = `library-block-card-${block.id}`;
    const hint = block.robotHint !== null && block.robotHint !== '' ? ` · ${block.robotHint}` : '';
    card.title = `${block.name} — step ${block.stepCount}개${hint} (클릭해 시퀀스에 삽입)`;
    card.setAttribute('aria-label', `${block.name} 블록 삽입 — step ${block.stepCount}개${hint}`);
    styled(card, { cursor: 'pointer' }); // grab 커서는 드래그 카드 전용

    const iconWrap = document.createElement('span');
    iconWrap.className = 'rsw-lib-card__icon';
    iconWrap.setAttribute('aria-hidden', 'true');
    iconWrap.appendChild(icon('puzzle', ICON.xl));
    const label = document.createElement('span');
    label.className = 'rsw-lib-card__label';
    label.textContent = block.name;
    card.appendChild(iconWrap);
    card.appendChild(label);

    card.addEventListener('click', () => {
      deps.onInsertBlock?.(block.id);
    });
    return card;
  };

  // ── 섹션/그리드 렌더 (검색·접기 변경 시 전체 재구축 — 카드 수가 작다) ─
  const render = (): void => {
    body.replaceChildren();
    const visible = filterTemplates(LIBRARY_TEMPLATES, query);

    for (const section of SECTIONS) {
      const templates = visible.filter((t) => t.section === section.id);
      const collapsed = sectionCollapsed.get(section.id) ?? false;

      // 섹션 헤더 (접기 토글 — UX §3.2). 셰브론 글리프 2종(▾/▸) 대신 **한 셰브론의
      // 회전**으로 통일한다 (UX_AUDIT C-18 — 한 화면에 규약이 둘이면 규약이 아니다).
      const sectionHeader = applyType(document.createElement('button'), TYPE.bodyStrong);
      sectionHeader.type = 'button';
      sectionHeader.className = collapsed
        ? 'ui-btn ui-btn--ghost rsw-lib-section rsw-lib-section--collapsed'
        : 'ui-btn ui-btn--ghost rsw-lib-section';
      sectionHeader.dataset.testid = `library-section-${section.id}`;
      sectionHeader.appendChild(icon('chevronDown', ICON.md));
      const sectionLabel = document.createElement('span');
      sectionLabel.textContent = `${section.labelKo} (${templates.length})`;
      sectionHeader.appendChild(sectionLabel);
      const sectionTitle = `${section.labelKo} — ${collapsed ? '펼치기' : '접기'}`;
      sectionHeader.title = sectionTitle;
      sectionHeader.setAttribute('aria-label', sectionTitle);
      sectionHeader.setAttribute('aria-expanded', String(!collapsed));
      sectionHeader.addEventListener('click', () => {
        sectionCollapsed.set(section.id, !collapsed);
        render();
      });
      body.appendChild(sectionHeader);
      if (collapsed) continue;

      if (templates.length === 0) {
        body.appendChild(mutedLine(query.trim() === '' ? '템플릿 없음' : '검색 결과 없음'));
        continue;
      }

      const grid = styled(document.createElement('div'), {
        display: 'grid',
        gridTemplateColumns: `repeat(auto-fill, minmax(${CARD_MIN_WIDTH_PX}px, 1fr))`,
        gap: SPACE.md,
      });
      grid.dataset.testid = `library-grid-${section.id}`;
      for (const template of templates) grid.appendChild(makeCard(template));
      body.appendChild(grid);
    }

    // 블록 섹션 (Phase 12+ 재사용 블록 — 삽입 시 전개) — blocksProvider 주입 시에만 렌더.
    // 섹션 라벨의 영/한 병기는 라이브러리 카드 한정 예외의 연장이다 (CLAUDE.md §4-b.4).
    if (deps.blocksProvider !== undefined) {
      const visibleBlocks = filterBlockSummaries(blockItems, query);
      const blocksHeader = applyType(document.createElement('div'), TYPE.bodyStrong);
      styled(blocksHeader, {
        color: COLOR.label,
        margin: `${SPACE.xl} 0 ${SPACE.sm} 0`,
        padding: `${SPACE.xxs} ${SPACE.xs}`,
      });
      blocksHeader.textContent =
        blocksState === 'ready' ? `Blocks · 블록 (${visibleBlocks.length})` : 'Blocks · 블록';
      blocksHeader.dataset.testid = 'library-section-blocks';
      body.appendChild(blocksHeader);

      if (blocksState === 'loading') {
        body.appendChild(mutedLine('블록 불러오는 중…'));
      } else if (blocksState === 'error') {
        // 막다른 길 금지 — 사유와 함께 재시도 경로를 준다
        body.appendChild(mutedLine('블록을 불러오지 못했습니다'));
        const retry = makeButton('다시 시도', '블록 목록 다시 불러오기', 'library-blocks-retry', 'ghost');
        retry.addEventListener('click', () => {
          loadBlocks();
        });
        body.appendChild(retry);
      } else if (visibleBlocks.length === 0) {
        body.appendChild(mutedLine(query.trim() === '' ? '저장된 블록 없음' : '검색 결과 없음'));
      } else {
        const blocksGrid = styled(document.createElement('div'), {
          display: 'grid',
          gridTemplateColumns: `repeat(auto-fill, minmax(${CARD_MIN_WIDTH_PX}px, 1fr))`,
          gap: SPACE.md,
        });
        blocksGrid.dataset.testid = 'library-grid-blocks';
        for (const block of visibleBlocks) blocksGrid.appendChild(makeBlockCard(block));
        body.appendChild(blocksGrid);
      }
    }

    // Import 섹션 (UX §3.2) — 통합자가 onImportRequest를 주입한 경우에만 렌더.
    // 검색어와 무관하게 항상 노출한다(진입점 상실 방지 — 카드가 아닌 "동작" 섹션).
    if (deps.onImportRequest !== undefined) {
      const importHeader = applyType(document.createElement('div'), TYPE.bodyStrong);
      styled(importHeader, {
        color: COLOR.label,
        margin: `${SPACE.xl} 0 ${SPACE.sm} 0`,
        padding: `${SPACE.xxs} ${SPACE.xs}`,
      });
      importHeader.textContent = '가져오기 · 3D 파일';
      body.appendChild(importHeader);

      const importCard = document.createElement('button');
      importCard.type = 'button';
      importCard.className = 'rsw-lib-card';
      importCard.dataset.testid = 'library-import-card';
      importCard.title =
        '3D 파일 임포트 — glTF/glb · STL · OBJ (클릭해 파일 선택, 또는 파일을 뷰포트로 드래그)';
      importCard.setAttribute('aria-label', '3D 파일 임포트 — 클릭해 파일 선택');
      styled(importCard, { width: '100%', cursor: 'pointer' });
      const importIcon = document.createElement('span');
      importIcon.className = 'rsw-lib-card__icon';
      importIcon.setAttribute('aria-hidden', 'true');
      importIcon.appendChild(icon('upload', ICON.xl));
      const importLabel = document.createElement('span');
      importLabel.className = 'rsw-lib-card__label';
      importLabel.textContent = '3D 파일 임포트';
      importCard.appendChild(importIcon);
      importCard.appendChild(importLabel);
      importCard.addEventListener('click', () => {
        deps.onImportRequest?.();
      });
      body.appendChild(importCard);
    }
  };

  search.addEventListener('input', () => {
    query = search.value;
    render();
  });

  // 패널 헤더 접기 → 본문/검색 숨김 (좌 패널 세로 공간 회수)
  panelHeader.onToggle((collapsed) => {
    body.style.display = collapsed ? 'none' : '';
    searchRow.style.display = collapsed ? 'none' : '';
  });

  host.appendChild(panel);
  render();
  loadBlocks(); // blocksProvider 미주입 시 no-op — 기존 호출부 무수정 호환

  return {
    el: panel,
    setSearch: (q): void => {
      query = q;
      search.value = q;
      render();
    },
    reloadBlocks: (): void => {
      loadBlocks();
    },
    dispose: (): void => {
      disposed = true;
      panel.remove();
    },
  };
}
