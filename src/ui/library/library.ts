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
// - 드래그 고스트: 카드 요소 자체가 기본 drag image로 쓰인다 (별도 프리뷰는 뷰포트의
//   반투명 고스트가 담당 — 이 모듈은 페이로드만 책임진다).
//
// 계층 규칙 (CLAUDE.md §3): core/render를 모른다. 엔티티 생성(spec)은 templates.ts의
// 순수 팩토리가, 실제 씬 편입은 통합자(SceneEditor)가 담당한다. id 유일화(uniquify)도
// 씬 상태를 아는 통합자가 주입한다.
//
// 접근성 (UX §9): 카드는 <button>이라 Tab 포커스·Enter/Space 배치가 기본 동작이고,
// 설명은 title(호버)과 aria-label로 병행 노출된다. 섹션 접기 토글은 aria-expanded.

import { LIBRARY_TEMPLATES, filterTemplates } from './templates';
import type { LibraryTemplate, TemplateSection } from './templates';
import { COLOR, FONT, RADIUS, SPACE, ensureThemeStyles, styled } from '../theme';
import type { EntitySpec } from '../../schema';

// ── 드래그 페이로드 MIME (통합자 뷰포트 드롭 핸들러와의 계약) ───────

export const TEMPLATE_MIME = 'application/x-robotsim-template';

// ── 공개 타입 ───────────────────────────────────────────────────────

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
   * [Phase 7] Import 섹션의 ⬆ 카드 클릭 (UX §3.2 "Import ⬆: 카드 클릭 = 파일 선택").
   * 파일 선택기/임포트 다이얼로그는 통합자 소유다. 미주입 시 Import 섹션은 렌더되지
   * 않는다(헤드리스/테스트 호환).
   */
  onImportRequest?(): void;
}

export interface LibraryHandle {
  /** 패널 루트 (host를 채운다) — 통합자 재배치용 */
  readonly el: HTMLElement;
  /** 검색어 프로그램 설정 (빈 문자열 = 전체 표시) */
  setSearch(query: string): void;
  dispose(): void;
}

// ── 섹션 정의 (UX §3.2 — Import 섹션은 3D 임포트 다이얼로그 소유자 몫) ──

const SECTIONS: ReadonlyArray<{ readonly id: TemplateSection; readonly labelKo: string }> = [
  { id: 'objects', labelKo: 'Objects · 사물' },
  { id: 'robots', labelKo: 'Robots · 로봇' },
];

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
  gap: 4px;
  padding: 8px 4px 6px 4px;
  background: ${COLOR.bgRaised};
  color: ${COLOR.text};
  border: 1px solid ${COLOR.borderStrong};
  border-radius: ${RADIUS.sm};
  font-family: inherit;
  font-size: 11px;
  line-height: 1.3;
  cursor: grab;
  user-select: none;
  transition: background-color 0.12s ease, border-color 0.12s ease, color 0.12s ease;
}
.rsw-lib-card:hover {
  background: #2b2f37;
  border-color: #4a5058;
  color: ${COLOR.textStrong};
}
.rsw-lib-card:focus-visible {
  outline: 2px solid var(--rsw-accent);
  outline-offset: 1px;
}
.rsw-lib-card:active { cursor: grabbing; }
.rsw-lib-card--dragging { opacity: 0.5; border-color: var(--rsw-accent); }
.rsw-lib-card__icon { font-size: 20px; line-height: 1.2; }
.rsw-lib-card__label {
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
`;
  document.head.appendChild(style);
}

// ── 내부 DOM 헬퍼 ───────────────────────────────────────────────────

/** muted 안내 줄 (빈 검색 결과 등) */
function mutedLine(text: string): HTMLElement {
  const el = styled(document.createElement('div'), {
    color: COLOR.muted,
    padding: '4px 2px',
  });
  el.textContent = text;
  return el;
}

// ── 마운트 ──────────────────────────────────────────────────────────

/** 라이브러리 패널을 host(워크스페이스 좌 슬롯)에 마운트한다. 패널은 host를 채운다. */
export function mountLibrary(host: HTMLElement, deps: LibraryDeps): LibraryHandle {
  ensureThemeStyles();
  ensureLibraryStyles();

  const panel = styled(document.createElement('div'), {
    width: '100%',
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
    minHeight: '0',
    color: COLOR.text,
    fontFamily: FONT.ui,
    fontSize: '12px',
    lineHeight: '1.5',
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

  // 헤더: 타이틀 + 검색 (UX §3.2 "검색/필터 상단 고정")
  const header = styled(document.createElement('div'), {
    display: 'flex',
    flexDirection: 'column',
    gap: SPACE.sm,
    padding: `${SPACE.sm} 10px`,
    borderBottom: `1px solid ${COLOR.borderSoft}`,
    flexShrink: '0',
  });
  const title = styled(document.createElement('strong'), {
    color: COLOR.textStrong,
    fontSize: '13px',
  });
  title.textContent = 'Library';
  header.appendChild(title);

  const search = document.createElement('input');
  search.type = 'search';
  search.className = 'ui-input';
  search.placeholder = '템플릿 검색…';
  search.setAttribute('aria-label', '템플릿 검색');
  search.dataset.testid = 'library-search';
  styled(search, { width: '100%', boxSizing: 'border-box' });
  header.appendChild(search);
  panel.appendChild(header);

  // 본문 (스크롤 영역): 섹션 + 카드 그리드
  const body = styled(document.createElement('div'), {
    flex: '1 1 auto',
    minHeight: '0',
    overflowY: 'auto',
    padding: `${SPACE.sm} 10px 10px 10px`,
  });
  body.classList.add('ui-scroll');
  panel.appendChild(body);

  // ── 뷰 상태 (ui 소유) ─────────────────────────────────────────────
  let query = '';
  const sectionCollapsed = new Map<TemplateSection, boolean>();

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

    const icon = document.createElement('span');
    icon.className = 'rsw-lib-card__icon';
    icon.textContent = template.icon;
    icon.setAttribute('aria-hidden', 'true');
    const label = document.createElement('span');
    label.className = 'rsw-lib-card__label';
    label.textContent = template.labelKo;
    card.appendChild(icon);
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
    });
    card.addEventListener('dragend', () => {
      card.classList.remove('rsw-lib-card--dragging');
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

      // 섹션 헤더 (접기 토글 — UX §3.2 "섹션 접기/펼치기")
      const sectionHeader = document.createElement('button');
      sectionHeader.type = 'button';
      sectionHeader.className = 'ui-btn ui-btn--ghost';
      sectionHeader.dataset.testid = `library-section-${section.id}`;
      sectionHeader.textContent = `${collapsed ? '▸' : '▾'} ${section.labelKo} (${templates.length})`;
      sectionHeader.setAttribute('aria-expanded', String(!collapsed));
      styled(sectionHeader, {
        display: 'block',
        width: '100%',
        textAlign: 'left',
        margin: `${SPACE.sm} 0 ${SPACE.xs} 0`,
        padding: '2px 4px',
      });
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
        gridTemplateColumns: 'repeat(auto-fill, minmax(88px, 1fr))',
        gap: SPACE.sm,
      });
      grid.dataset.testid = `library-grid-${section.id}`;
      for (const template of templates) grid.appendChild(makeCard(template));
      body.appendChild(grid);
    }

    // Import 섹션 (UX §3.2) — 통합자가 onImportRequest를 주입한 경우에만 렌더.
    // 검색어와 무관하게 항상 노출한다(진입점 상실 방지 — 카드가 아닌 "동작" 섹션).
    if (deps.onImportRequest !== undefined) {
      const importHeader = styled(document.createElement('div'), {
        color: COLOR.muted,
        margin: `${SPACE.sm} 0 ${SPACE.xs} 0`,
        padding: '2px 4px',
        fontWeight: '600',
      });
      importHeader.textContent = 'Import · 3D 파일';
      body.appendChild(importHeader);

      const importCard = document.createElement('button');
      importCard.type = 'button';
      importCard.className = 'rsw-lib-card';
      importCard.dataset.testid = 'library-import-card';
      importCard.title = '3D 파일 임포트 — glTF/glb · STL · OBJ (클릭해 파일 선택, 또는 파일을 뷰포트로 드래그)';
      importCard.setAttribute('aria-label', '3D 파일 임포트 — 클릭해 파일 선택');
      styled(importCard, { width: '100%', cursor: 'pointer' });
      const importIcon = document.createElement('span');
      importIcon.className = 'rsw-lib-card__icon';
      importIcon.textContent = '⬆';
      importIcon.setAttribute('aria-hidden', 'true');
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

  host.appendChild(panel);
  render();

  return {
    el: panel,
    setSearch: (q): void => {
      query = q;
      search.value = q;
      render();
    },
    dispose: (): void => {
      panel.remove();
    },
  };
}
