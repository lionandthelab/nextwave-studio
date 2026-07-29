// ui/dock/collision-log.ts — 충돌 로그 패널 (UX_DESIGN §3.6 "충돌 로그")
//
// 표 형식: 시각(s.mmm) · a × b · phase 배지 · kind · 링크 어포던스. 최신이 아래
// (자동 스크롤 — 사용자가 위로 스크롤해 두면 자동 스크롤을 멈춘다). 엔티티 텍스트 필터
// 입력. 행 활성화 → onFocusEntity(id)(하이라이트) + onRowClick({a,b,timeSec})
// (Phase 10 "행 클릭→오브젝트 포커스+당시 활성 노드 강조" — timeSec으로 통합자가 그
// 시점 활성 노드를 강조, ROADMAP §3.6).
//
// ── Phase 11에서 닫는 감사 항목 ─────────────────────────────────────
// C-18 표 시맨틱 복구. 구현은 `<tr role="button">`이었다 — role을 덮어쓰면 그 행은
//      표에서 이탈하고 자식 `<td>`들이 presentational로 강등되어 열 의미가 통째로
//      사라진다. `<thead>`/`<th scope="col">`도 없어 "0.512 / arm × box_a / start /
//      contact"가 무엇의 나열인지 어디에도 없었다.
//      → **표 시맨틱을 살리고**(thead + th scope=col + caption), 행 활성화는 첫 셀 안의
//        실제 `<button>`이 담당한다. grid 재해석(role=grid/row/gridcell) 대신 이 쪽을
//        고른 이유: 이 표는 셀 단위 편집/2차원 탐색이 없는 **읽기 전용 목록**이고,
//        네이티브 table 시맨틱은 무료로 정확한 반면 grid는 셀 로빙까지 직접 구현해야
//        하며 그 순간 "행 열기"라는 단일 동작이 2차원 모델 뒤로 숨는다.
// C-7  `unseenCount()`/`resetUnseen()` — 비활성 탭에 쌓인 충돌 건수를 독 배지에 공급한다.
//      + CSV 내보내기: 현재 필터가 적용된 결과를 파일로 낸다. 검증 도구의 산출물이
//        화면 밖으로 나갈 수 없으면 워크플로의 종점이 아니라 구경거리에 머문다.
//      **aria-live는 걸지 않는다** — 물리 스텝마다 이벤트가 쏟아지면 polite 큐가 포화되어
//      사용자가 다른 조작을 해도 몇 분간 과거 충돌만 읽는다(링버퍼의 앞 행 제거도 변경으로
//      집계되어 중복 발화한다). 요약 발화는 통합자가 createAnnouncer로 처리한다.
//
// 계층 규칙: schema의 CollisionEvent POJO만 안다. 이벤트 공급(모니터 구독)은
// 글루(main.ts)가 addEvent 호출로 중계한다 — 이 모듈은 core를 import하지 않는다.

import { CONTACT_CLASS_LABEL_KO } from '../../core/collision-classify';
import type { ContactClass } from '../../core/collision-classify';
import { rovingTabindex } from '../a11y';
import { icon, makeIconButton } from '../icons';
import {
  BORDER,
  BORDER_WIDTH,
  COLLISION,
  COLOR,
  ICON,
  MOTION,
  SPACE,
  SURFACE,
  TYPE,
  applyType,
  ensureThemeStyles,
  styled,
  tr,
} from '../theme';
import type { CollisionEvent } from '../../schema/types';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4, 시각 토큰은 ui/theme.ts) ────

/** DOM/메모리에 유지할 최대 행 수 — 초과 시 가장 오래된 행부터 제거 */
const MAX_ROWS = 500;
/** 자동 스크롤 판정: 바닥에서 이 px 이내면 "바닥에 붙어 있음"으로 본다 */
const AUTOSCROLL_THRESHOLD_PX = 8;
/** 시각 표시 소수 자릿수 (s.mmm) */
const TIME_DECIMALS = 3;
/** CSV 헤더 — 도메인 식별자라 영문 유지 (스프레드시트/스크립트 소비 대상) */
const CSV_HEADER = 'timeSec,a,b,phase,kind';
/** UTF-8 BOM — Excel이 UTF-8 CSV를 ANSI로 오해하지 않게 (엔티티 id에 한글이 섞일 수 있다) */
const UTF8_BOM = '\uFEFF';

// ── 로그 전용 스타일 (hover/focus 어포던스 — 토큰만 소비, 1회 주입) ──
// theme.ts를 건드리지 않고 이 패널 몫의 인터랙션(:hover / :focus-visible)만 주입한다
// (canvas.ts의 ensureCanvasStyles와 같은 국소 스타일 패턴). 색은 theme 토큰만 쓴다.

const COLLISION_LOG_STYLE_ID = 'rsw-collision-log-styles';

function ensureCollisionLogStyles(): void {
  if (document.getElementById(COLLISION_LOG_STYLE_ID) !== null) return;
  const style = document.createElement('style');
  style.id = COLLISION_LOG_STYLE_ID;
  style.textContent = `
.rsw-clog-row { cursor: pointer; }
/* 행 열기 버튼 — 시각적으로는 첫 셀의 텍스트, 시맨틱으로는 진짜 버튼 */
.rsw-clog-open {
  display: block;
  width: 100%;
  padding: 0;
  margin: 0;
  border: 0;
  background: none;
  color: inherit;
  font: inherit;
  text-align: left;
  cursor: pointer;
}
.rsw-clog-open:focus-visible { outline: 2px solid ${COLOR.accent}; outline-offset: 1px; }
.rsw-clog-link {
  color: ${COLOR.muted};
  opacity: 0.45;
  transition: ${tr('color', MOTION.instant)}, ${tr('opacity', MOTION.instant)};
}
.rsw-clog-row:hover .rsw-clog-link,
.rsw-clog-row:focus-within .rsw-clog-link { color: ${COLOR.accentText}; opacity: 1; }
`;
  document.head.appendChild(style);
}

/**
 * phase/kind → 배지 클래스 (텍스트가 의미를 전달하고 색은 보조 — UX_DESIGN §9).
 * sensor 이벤트는 phase와 무관하게 파랑 계열로 구분한다(감지 전용 — 물리 반응 없음).
 */
function badgeClassOf(e: CollisionEvent): string {
  if (e.kind === 'sensor') return 'ui-badge ui-badge--sensor';
  return e.phase === 'start' ? 'ui-badge ui-badge--start' : 'ui-badge ui-badge--stop';
}

// ── 공개 타입 ───────────────────────────────────────────────────────

export interface CollisionLogOptions {
  /** 행/엔티티 클릭 → 관련 오브젝트 포커스 요청 (하이라이트 등 — 글루가 구현) */
  onFocusEntity(entityId: string): void;
  /**
   * 행 클릭 → 충돌 이벤트 좌표 통지 (Phase 10, §3.6 "당시 활성 노드 강조"). onFocusEntity와
   * 함께 호출된다(엔티티 포커스와 노드 강조는 별개 관심사). timeSec으로 통합자가 그 시점의
   * 활성 노드를 찾아 강조한다(예: run-overlay.timeSecToNodeIndex). 선택 — 미주입이면 무시.
   */
  onRowClick?(info: { a: string; b: string; timeSec: number }): void;
}

export interface CollisionLogPanel {
  readonly el: HTMLElement;
  /**
   * 새 접촉 이벤트 1건 추가 (모니터 구독을 글루가 중계).
   *
   * `contactClass`는 통합자가 core/collision-classify로 판정해 넘긴다. 미주입이면
   * 분류 없이 기록만 한다(하위 호환). **`unexpected`만 미확인 카운트에 들어간다** —
   * 의도한 파지가 독 배지를 올리면 진짜 사고가 소음에 묻힌다.
   */
  addEvent(e: CollisionEvent, contactClass?: ContactClass): void;
  /** 모든 행 제거 (Stop/씬 리셋 시 — 결정론적 재생과 짝). 미확인 카운트도 0이 된다 */
  clear(): void;
  /**
   * 마지막 resetUnseen 이후 도착한 이벤트 수 — 독 탭 배지의 소스 (C-7).
   * 표시 여부/상한 판단은 독(formatDockBadge)이 한다.
   */
  unseenCount(): number;
  /** 미확인 카운트 리셋 (독이 이 탭을 활성화했을 때) */
  resetUnseen(): void;
  dispose(): void;
}

// ── 내부 헬퍼 ───────────────────────────────────────────────────────

/** 필터 문자열이 이벤트의 a/b 엔티티 id에 부분 일치하는가 (대소문자 무시) */
function matchesFilter(a: string, b: string, filter: string): boolean {
  if (filter === '') return true;
  const f = filter.toLowerCase();
  return a.toLowerCase().includes(f) || b.toLowerCase().includes(f);
}

/** CSV 필드 이스케이프 (RFC 4180 — 쉼표/따옴표/개행 포함 시 인용) */
function csvField(value: string): string {
  return /[",\r\n]/.test(value) ? `"${value.replace(/"/g, '""')}"` : value;
}

/** 이벤트 목록 → CSV 본문 (헤더 포함). 순수 — 단위 테스트 가능 */
export function collisionEventsToCsv(events: readonly CollisionEvent[]): string {
  const lines = [CSV_HEADER];
  for (const e of events) {
    lines.push(
      [
        e.timeSec.toFixed(TIME_DECIMALS),
        csvField(e.a),
        csvField(e.b),
        csvField(e.phase),
        csvField(e.kind),
      ].join(','),
    );
  }
  return `${lines.join('\r\n')}\r\n`;
}

// ── 패널 ────────────────────────────────────────────────────────────

export function createCollisionLogPanel(opts: CollisionLogOptions): CollisionLogPanel {
  ensureThemeStyles();
  ensureCollisionLogStyles();
  const el = styled(document.createElement('div'), {
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
    boxSizing: 'border-box',
  });
  el.dataset.testid = 'collision-log';

  // ── 헤더: 필터 + CSV 내보내기 ─────────────────────────────────────
  const headerRow = styled(document.createElement('div'), {
    display: 'flex',
    alignItems: 'center',
    gap: SPACE.sm,
    padding: `${SPACE.sm} ${SPACE.md}`,
    borderBottom: `${BORDER_WIDTH.hair} solid ${BORDER.default}`,
    flexShrink: '0',
  });
  const filterLabel = styled(document.createElement('span'), { color: COLOR.muted });
  applyType(filterLabel, TYPE.caption);
  filterLabel.textContent = '필터';
  const filterInput = styled(document.createElement('input'), { width: '140px' });
  applyType(filterInput, TYPE.body);
  filterInput.className = 'ui-input';
  filterInput.type = 'text';
  filterInput.placeholder = '엔티티 id…';
  filterInput.dataset.testid = 'collision-filter';
  filterInput.setAttribute('aria-label', '충돌 로그 엔티티 필터');

  const headerSpacer = styled(document.createElement('span'), { flex: '1 1 auto' });
  const exportButton = makeIconButton(
    'download',
    'CSV',
    'CSV 내보내기 — 현재 필터가 적용된 행만',
    'collision-export',
  );

  headerRow.appendChild(filterLabel);
  headerRow.appendChild(filterInput);
  headerRow.appendChild(headerSpacer);
  headerRow.appendChild(exportButton);
  el.appendChild(headerRow);

  // ── 스크롤 영역 + 표 ──────────────────────────────────────────────
  const scroller = styled(document.createElement('div'), {
    flex: '1 1 auto',
    minHeight: '0',
    overflowY: 'auto',
    padding: `0 ${SPACE.xs}`,
  });
  scroller.classList.add('ui-scroll');
  const table = styled(document.createElement('table'), {
    width: '100%',
    borderCollapse: 'collapse',
  });
  applyType(table, TYPE.monoBody);

  const caption = document.createElement('caption');
  caption.className = 'sr-only';
  caption.textContent = '충돌 이벤트 로그 — 시각 · 엔티티 쌍 · phase · kind';
  table.appendChild(caption);

  // 열 의미를 표 자체가 전달한다 (구현에는 thead가 아예 없었다 — C-18)
  const thead = document.createElement('thead');
  const headTr = document.createElement('tr');
  const COLUMNS: readonly { readonly text: string; readonly en?: boolean }[] = [
    { text: '시각' },
    { text: '엔티티 쌍' },
    { text: 'phase', en: true },
    { text: '분류', en: false },
    { text: '연동' },
  ];
  COLUMNS.forEach((col, i) => {
    const th = styled(document.createElement('th'), {
      position: 'sticky',
      top: '0',
      zIndex: '1',
      background: SURFACE.panel,
      color: COLOR.label,
      textAlign: 'left',
      whiteSpace: 'nowrap',
      padding: `${SPACE.xs} ${SPACE.md} ${SPACE.xs} ${SPACE.xs}`,
      borderBottom: `${BORDER_WIDTH.hair} solid ${BORDER.default}`,
    });
    applyType(th, TYPE.caption);
    th.setAttribute('scope', 'col');
    if (col.en === true) th.setAttribute('lang', 'en');
    // 마지막 열(링크 어포던스)은 시각적으로 글리프뿐이라 제목을 화면에서 감춘다
    if (i === COLUMNS.length - 1) {
      const sr = document.createElement('span');
      sr.className = 'sr-only';
      sr.textContent = col.text;
      th.appendChild(sr);
    } else {
      th.textContent = col.text;
    }
    headTr.appendChild(th);
  });
  thead.appendChild(headTr);
  table.appendChild(thead);

  const tbody = document.createElement('tbody');
  table.appendChild(tbody);
  scroller.appendChild(table);
  el.appendChild(scroller);

  // 빈 상태 안내 (UX_DESIGN §7) — 행이 생기면 숨긴다
  const emptyState = styled(document.createElement('div'), {
    padding: `${SPACE.xl} ${SPACE.md}`,
    color: COLOR.muted,
    textAlign: 'center',
  });
  applyType(emptyState, TYPE.body);
  emptyState.textContent = '아직 충돌 이벤트가 없습니다 — 재생 후 감지 즉시 여기 기록됩니다';
  emptyState.dataset.testid = 'collision-empty';
  scroller.appendChild(emptyState);

  /** 표시 중인 이벤트(링버퍼) — CSV 내보내기의 소스 */
  const events: CollisionEvent[] = [];
  /** 마지막 resetUnseen 이후 도착 건수 (독 배지 소스 — C-7) */
  let unseen = 0;

  // 행 열기 버튼 방향키 탐색 — 행 500개를 탭 500번으로 지나가게 하지 않는다.
  // 이벤트가 쏟아질 때 매 건 O(n) 재계산을 피하려고 마이크로태스크 1회로 합친다.
  const roving = rovingTabindex(tbody, [], {
    orientation: 'vertical',
    onActivate: (target) => {
      if (target instanceof HTMLButtonElement) target.click();
    },
  });
  let rovingDirty = false;
  const scheduleRovingRefresh = (): void => {
    if (rovingDirty) return;
    rovingDirty = true;
    queueMicrotask(() => {
      rovingDirty = false;
      const visible: HTMLElement[] = [];
      for (const row of tbody.children) {
        if (!(row instanceof HTMLTableRowElement)) continue;
        if (row.style.display === 'none') continue;
        const button = row.querySelector<HTMLButtonElement>('button.rsw-clog-open');
        if (button !== null) visible.push(button);
      }
      roving.setItems(visible);
    });
  };

  const currentFilter = (): string => filterInput.value.trim();

  /** 현재 필터를 통과하는 이벤트만 (CSV 내보내기와 행 표시가 같은 술어를 쓴다) */
  const filteredEvents = (): CollisionEvent[] => {
    const filter = currentFilter();
    return events.filter((e) => matchesFilter(e.a, e.b, filter));
  };

  const paintEmptyState = (): void => {
    emptyState.style.display = tbody.childElementCount === 0 ? '' : 'none';
    thead.style.display = tbody.childElementCount === 0 ? 'none' : '';
  };

  const paintExportButton = (): void => {
    const n = filteredEvents().length;
    exportButton.disabled = n === 0;
    const t =
      n === 0
        ? 'CSV 내보내기 — 내보낼 행이 없습니다'
        : `CSV 내보내기 — 현재 필터 기준 ${n}건`;
    exportButton.title = t;
    exportButton.setAttribute('aria-label', t);
  };

  /** 행 표시/숨김을 현재 필터로 갱신 */
  const applyFilter = (): void => {
    const filter = currentFilter();
    for (const row of tbody.children) {
      if (!(row instanceof HTMLTableRowElement)) continue;
      const a = row.dataset.entityA ?? '';
      const b = row.dataset.entityB ?? '';
      row.style.display = matchesFilter(a, b, filter) ? '' : 'none';
    }
    paintExportButton();
    scheduleRovingRefresh();
  };
  filterInput.addEventListener('input', applyFilter);

  exportButton.addEventListener('click', () => {
    const rows = filteredEvents();
    if (rows.length === 0) return;
    const blob = new Blob([UTF8_BOM, collisionEventsToCsv(rows)], {
      type: 'text/csv;charset=utf-8',
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `collisions-${new Date().toISOString().replace(/[:.]/g, '-')}.csv`;
    anchor.click();
    URL.revokeObjectURL(url);
  });

  const addEvent = (e: CollisionEvent, contactClass?: ContactClass): void => {
    // 자동 스크롤: 사용자가 위로 스크롤해 둔 상태면 유지(멈춤), 바닥이면 따라간다
    const stick =
      scroller.scrollTop + scroller.clientHeight >=
      scroller.scrollHeight - AUTOSCROLL_THRESHOLD_PX;

    const row = document.createElement('tr');
    row.dataset.testid = 'collision-row';
    row.dataset.entityA = e.a;
    row.dataset.entityB = e.b;
    if (contactClass !== undefined) row.dataset.contactClass = contactClass;
    row.classList.add('rsw-hover-row', 'rsw-clog-row');
    styled(row, { borderBottom: `${BORDER_WIDTH.hair} solid ${BORDER.subtle}` });

    // 첫 셀 = 시각 + 행 활성화 버튼. `<tr role="button">`이 아니라 진짜 버튼이라
    // 표 구조가 유지되고 열 헤더(scope=col)가 각 셀에 정상 연결된다.
    const timeCell = styled(document.createElement('td'), {
      color: COLOR.muted,
      padding: `${SPACE.xxs} ${SPACE.md} ${SPACE.xxs} ${SPACE.xs}`,
      whiteSpace: 'nowrap',
      width: '1%',
    });
    const openButton = document.createElement('button');
    openButton.type = 'button';
    openButton.className = 'rsw-clog-open';
    openButton.tabIndex = -1; // roving이 활성 1개만 0으로 올린다
    openButton.textContent = e.timeSec.toFixed(TIME_DECIMALS);
    openButton.dataset.testid = 'collision-row-open';
    openButton.setAttribute(
      'aria-label',
      `충돌 ${e.timeSec.toFixed(TIME_DECIMALS)}초 · ${e.a} × ${e.b} · ${e.phase} · ${e.kind}` +
        ' — 오브젝트 포커스 및 당시 노드 강조',
    );
    openButton.title = '오브젝트 포커스 + 당시 활성 노드 강조';
    timeCell.appendChild(openButton);

    const pairCell = styled(document.createElement('td'), {
      color: COLOR.text,
      padding: `${SPACE.xxs} ${SPACE.md}`,
    });
    pairCell.textContent = `${e.a} × ${e.b}`;
    pairCell.setAttribute('lang', 'en'); // 엔티티 id는 도메인 식별자(영문)

    const phaseCell = styled(document.createElement('td'), {
      padding: `${SPACE.xxs} ${SPACE.md}`,
      width: '1%',
    });
    const badge = document.createElement('span');
    // 배지 텍스트가 의미(phase)를 전달하고 색(kind/phase)은 보조 — sensor는 파랑 계열
    badge.className = badgeClassOf(e);
    badge.textContent = e.phase;
    badge.setAttribute('lang', 'en');
    phaseCell.appendChild(badge);

    // 분류 셀 — 물리 kind(contact/sensor)가 아니라 **사용자에게 의미 있는 분류**를 보인다.
    // "arm × box_a / contact"는 그것이 성공인지 사고인지 말해 주지 않는다.
    const kindCell = styled(document.createElement('td'), {
      padding: `${SPACE.xxs} ${SPACE.xs}`,
      width: '1%',
      whiteSpace: 'nowrap',
    });
    if (contactClass === undefined) {
      kindCell.style.color = e.kind === 'sensor' ? COLOR.infoText : COLOR.label;
      kindCell.textContent = e.kind;
      kindCell.setAttribute('lang', 'en');
    } else {
      kindCell.style.color =
        contactClass === 'unexpected'
          ? COLLISION.text
          : contactClass === 'sensor'
            ? COLOR.infoText
            : COLOR.muted;
      kindCell.textContent = CONTACT_CLASS_LABEL_KO[contactClass];
    }

    // 연동 어포던스: 이 행이 뷰포트/노드와 이어져 있음을 알리는 표적 아이콘
    // (hover/focus-within 시 강조 — ensureCollisionLogStyles). 이름은 버튼 aria-label이
    // 이미 전달하므로 글리프는 장식이다.
    const linkCell = styled(document.createElement('td'), {
      padding: `${SPACE.xxs} ${SPACE.xs} ${SPACE.xxs} ${SPACE.xxs}`,
      width: '1%',
      textAlign: 'right',
    });
    const linkGlyph = icon('target', ICON.sm);
    linkGlyph.classList.add('rsw-clog-link');
    linkGlyph.style.display = 'inline-block';
    linkCell.appendChild(linkGlyph);

    row.appendChild(timeCell);
    row.appendChild(pairCell);
    row.appendChild(phaseCell);
    row.appendChild(kindCell);
    row.appendChild(linkCell);

    // 행 활성화 → 관련 오브젝트 포커스 + 이벤트 좌표 통지(§3.6).
    // 포커스 대상은 바닥 같은 예약 엔티티('__' 접두)보다 사용자 엔티티를 우선한다
    // (하이라이트 대상으로 의미가 있는 쪽). timeSec은 통합자가 당시 노드 강조에 쓴다.
    openButton.addEventListener('click', () => {
      const preferred = e.a.startsWith('__') && !e.b.startsWith('__') ? e.b : e.a;
      opts.onFocusEntity(preferred);
      opts.onRowClick?.({ a: e.a, b: e.b, timeSec: e.timeSec });
    });

    // 현재 필터 즉시 적용
    if (!matchesFilter(e.a, e.b, currentFilter())) row.style.display = 'none';

    tbody.appendChild(row);
    events.push(e);
    while (tbody.childElementCount > MAX_ROWS) tbody.firstElementChild?.remove();
    while (events.length > MAX_ROWS) events.shift();
    // 배지 소스: **충돌의 시작(start)만** 센다. 의도된 접촉은 세지 않고, stop은 같은
    // 사건의 끝이라 새 건수가 아니다 — 상태줄 '충돌 N'과 같은 기준이어야 두 표시가
    // 어긋나지 않는다. (분류 미주입 시엔 하위 호환으로 전부 센다.)
    if (contactClass === undefined || (contactClass === 'unexpected' && e.phase === 'start')) {
      unseen += 1;
    }
    paintEmptyState();
    paintExportButton();
    scheduleRovingRefresh();
    if (stick) scroller.scrollTop = scroller.scrollHeight;
  };

  paintEmptyState();
  paintExportButton();

  return {
    el,
    addEvent,
    clear: (): void => {
      tbody.replaceChildren();
      events.length = 0;
      unseen = 0;
      paintEmptyState();
      paintExportButton();
      scheduleRovingRefresh();
    },
    unseenCount: (): number => unseen,
    resetUnseen: (): void => {
      unseen = 0;
    },
    dispose: (): void => {
      roving.dispose();
      el.remove();
    },
  };
}
