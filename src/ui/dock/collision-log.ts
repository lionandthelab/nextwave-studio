// ui/dock/collision-log.ts — 충돌 로그 패널 (UX_DESIGN §3.6 "Collision Log")
//
// 표 형식: 시간(s.mmm) · a × b · phase 배지 · kind. 최신이 아래(자동 스크롤 —
// 사용자가 위로 스크롤해 두면 자동 스크롤을 멈춘다). 엔티티 텍스트 필터 입력,
// 행 클릭 → onFocusEntity(id) 콜백(하이라이트 + 콘솔 노트 — 카메라 포커스는
// Phase 10 "행 클릭→오브젝트 포커스+당시 노드 강조"에서 확장, ROADMAP).
//
// 계층 규칙: schema의 CollisionEvent POJO만 안다. 이벤트 공급(모니터 구독)은
// 글루(main.ts)가 addEvent 호출로 중계한다 — 이 모듈은 core를 import하지 않는다.

import type { CollisionEvent } from '../../schema/types';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4) ─────────────────────────────

/** DOM에 유지할 최대 행 수 — 초과 시 가장 오래된 행부터 제거 */
const MAX_ROWS = 500;
/** 자동 스크롤 판정: 바닥에서 이 px 이내면 "바닥에 붙어 있음"으로 본다 */
const AUTOSCROLL_THRESHOLD_PX = 8;
/** 시간 표시 소수 자릿수 (s.mmm) */
const TIME_DECIMALS = 3;

const PHASE_COLORS: Readonly<Record<CollisionEvent['phase'], string>> = {
  start: '#e74c3c',
  stop: '#5d6470',
};

// ── 공개 타입 ───────────────────────────────────────────────────────

export interface CollisionLogOptions {
  /** 행/엔티티 클릭 → 관련 오브젝트 포커스 요청 (하이라이트 등 — 글루가 구현) */
  onFocusEntity(entityId: string): void;
}

export interface CollisionLogPanel {
  readonly el: HTMLElement;
  /** 새 충돌 이벤트 1건 추가 (모니터 구독을 글루가 중계) */
  addEvent(e: CollisionEvent): void;
  /** 모든 행 제거 (Stop/씬 리셋 시 — 결정론적 재생과 짝) */
  clear(): void;
  dispose(): void;
}

// ── 내부 헬퍼 ───────────────────────────────────────────────────────

function styled<T extends HTMLElement>(el: T, style: Partial<CSSStyleDeclaration>): T {
  Object.assign(el.style, style);
  return el;
}

/** 필터 문자열이 이벤트의 a/b 엔티티 id에 부분 일치하는가 (대소문자 무시) */
function matchesFilter(a: string, b: string, filter: string): boolean {
  if (filter === '') return true;
  const f = filter.toLowerCase();
  return a.toLowerCase().includes(f) || b.toLowerCase().includes(f);
}

// ── 패널 ────────────────────────────────────────────────────────────

export function createCollisionLogPanel(opts: CollisionLogOptions): CollisionLogPanel {
  const el = styled(document.createElement('div'), {
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
    boxSizing: 'border-box',
  });
  el.dataset.testid = 'collision-log';

  // 필터 입력 줄
  const filterRow = styled(document.createElement('div'), {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    padding: '4px 8px',
    borderBottom: '1px solid #2e3238',
    flexShrink: '0',
  });
  const filterLabel = styled(document.createElement('span'), { color: '#5d6470', fontSize: '11px' });
  filterLabel.textContent = '필터';
  const filterInput = styled(document.createElement('input'), {
    background: '#101216',
    color: '#cfd3d9',
    border: '1px solid #2e3238',
    borderRadius: '3px',
    padding: '1px 6px',
    fontFamily: 'inherit',
    fontSize: '11px',
    width: '140px',
  });
  filterInput.type = 'text';
  filterInput.placeholder = '엔티티 id…';
  filterInput.dataset.testid = 'collision-filter';
  filterRow.appendChild(filterLabel);
  filterRow.appendChild(filterInput);
  el.appendChild(filterRow);

  // 스크롤 영역 + 표
  const scroller = styled(document.createElement('div'), {
    flex: '1',
    overflowY: 'auto',
    padding: '0 4px',
  });
  const table = styled(document.createElement('table'), {
    width: '100%',
    borderCollapse: 'collapse',
    fontSize: '11px',
  });
  const tbody = document.createElement('tbody');
  table.appendChild(tbody);
  scroller.appendChild(table);
  el.appendChild(scroller);

  /** 행 표시/숨김을 현재 필터로 갱신 */
  const applyFilter = (): void => {
    const filter = filterInput.value.trim();
    for (const row of tbody.children) {
      if (!(row instanceof HTMLTableRowElement)) continue;
      const a = row.dataset.entityA ?? '';
      const b = row.dataset.entityB ?? '';
      row.style.display = matchesFilter(a, b, filter) ? '' : 'none';
    }
  };
  filterInput.addEventListener('input', applyFilter);

  const addEvent = (e: CollisionEvent): void => {
    // 자동 스크롤: 사용자가 위로 스크롤해 둔 상태면 유지(멈춤), 바닥이면 따라간다
    const stick =
      scroller.scrollTop + scroller.clientHeight >=
      scroller.scrollHeight - AUTOSCROLL_THRESHOLD_PX;

    const row = document.createElement('tr');
    row.dataset.testid = 'collision-row';
    row.dataset.entityA = e.a;
    row.dataset.entityB = e.b;
    styled(row, { cursor: 'pointer', borderBottom: '1px solid #22252b' });

    const timeCell = styled(document.createElement('td'), {
      color: '#5d6470',
      padding: '2px 8px 2px 4px',
      whiteSpace: 'nowrap',
      width: '1%',
    });
    timeCell.textContent = e.timeSec.toFixed(TIME_DECIMALS);

    const pairCell = styled(document.createElement('td'), {
      color: '#cfd3d9',
      padding: '2px 8px',
    });
    pairCell.textContent = `${e.a} × ${e.b}`;

    const phaseCell = styled(document.createElement('td'), { padding: '2px 8px', width: '1%' });
    const badge = styled(document.createElement('span'), {
      color: '#fff',
      background: PHASE_COLORS[e.phase],
      borderRadius: '3px',
      padding: '0 6px',
      fontSize: '10px',
    });
    badge.textContent = e.phase;
    phaseCell.appendChild(badge);

    const kindCell = styled(document.createElement('td'), {
      color: '#9aa0a8',
      padding: '2px 4px',
      width: '1%',
      whiteSpace: 'nowrap',
    });
    kindCell.textContent = e.kind;

    row.appendChild(timeCell);
    row.appendChild(pairCell);
    row.appendChild(phaseCell);
    row.appendChild(kindCell);

    // 행 클릭 → 관련 오브젝트 포커스. 바닥 같은 예약 엔티티('__' 접두)보다
    // 사용자 엔티티를 우선한다 (하이라이트 대상으로 의미가 있는 쪽).
    row.addEventListener('click', () => {
      const preferred = e.a.startsWith('__') && !e.b.startsWith('__') ? e.b : e.a;
      opts.onFocusEntity(preferred);
    });

    // 현재 필터 즉시 적용
    if (!matchesFilter(e.a, e.b, filterInput.value.trim())) row.style.display = 'none';

    tbody.appendChild(row);
    while (tbody.childElementCount > MAX_ROWS) tbody.firstElementChild?.remove();
    if (stick) scroller.scrollTop = scroller.scrollHeight;
  };

  return {
    el,
    addEvent,
    clear: (): void => {
      tbody.replaceChildren();
    },
    dispose: (): void => {
      el.remove();
    },
  };
}
