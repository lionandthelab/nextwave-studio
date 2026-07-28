// ui/command-bar/layout.test.ts — 커맨드바 레이아웃·어휘 계약 테스트 (UX_AUDIT C-2/C-18)
//
// ── 이 테스트가 막는 회귀 ───────────────────────────────────────────
// 구 셸은 44px 한 줄에 22개 요소(인터랙티브 18개)를 넣고 중앙 슬롯에만
// `justify-content: center`를 걸었다. 중앙만 `min-width: 0`으로 0까지 수축하는데
// `overflow` 지정이 없어 내용이 **좌우 대칭으로 상자 밖에 그려졌고**, DOM 뒤 형제가
// 위에 깔려 포인터 이벤트를 가로챘다. 실측 겹침: 1920→1 / 1600→8 / 1440→13 /
// 1366→17 / 1280→20 / 1024→24건. 1280×720에서 ↶/↷는 100% 피복되어 클릭 불가였다.
//
// 원인은 스타일 세 줄이었고, 고치는 것도 세 줄이다. 그래서 이 테스트는 "겹침이
// 없는지"가 아니라 **겹침을 불가능하게 만드는 스타일 계약이 살아 있는지**를 본다.
//
// ── 왜 계약 검증인가 ────────────────────────────────────────────────
// 실제 겹침 측정에는 레이아웃 엔진이 필요한데, 이 저장소의 vitest 환경은 node이고
// jsdom도 설치돼 있지 않다(jsdom이 있어도 `getBoundingClientRect`가 전부 0이라
// 겹침 측정은 불가능하다). 실측은 브라우저 게이트(scripts/gate-browser.mjs)의
// 몫이고, 여기서는 **DOM 없이 검증 가능한 스타일 계약**을 고정한다.

import { describe, expect, it } from 'vitest';
import {
  COMMAND_BAR_LEGACY_SLOT,
  COMMAND_BAR_PRIORITY,
  COMMAND_BAR_ROW_HEIGHT_PX,
  COMMAND_BAR_SLOT_STYLE,
  MIN_NL_INPUT_WIDTH_PX,
  commandBarDensity,
  overflowMoveOrder,
} from './scene-controls';
import type { CommandBarSlotName } from './scene-controls';
import { physicsStateLabel } from './playback';
import { BREAKPOINT, LAYOUT } from '../theme';

// ── 슬롯 스타일 계약 ────────────────────────────────────────────────

/** 남는 폭을 흡수해 수축할 수 있는 슬롯 — 넘침이 형제 위로 새면 안 된다 */
const SHRINKABLE_SLOTS: readonly CommandBarSlotName[] = ['row', 'rowAStart', 'rowBCommand'];
/** 수축하지 않는 슬롯 — P0 액션(재생·주요 토글)을 폭 압박에서 보호한다 */
const FIXED_SLOTS: readonly CommandBarSlotName[] = ['rowAEnd', 'rowBTransport'];
const CONTENT_SLOTS: readonly CommandBarSlotName[] = [...SHRINKABLE_SLOTS, ...FIXED_SLOTS];

describe('커맨드바 슬롯 스타일 계약 (C-2)', () => {
  it('바 자신: flex-wrap nowrap + overflow hidden — 내용이 바 밖으로 그려지지 않는다', () => {
    expect(COMMAND_BAR_SLOT_STYLE.bar.flexWrap).toBe('nowrap');
    expect(COMMAND_BAR_SLOT_STYLE.bar.overflow).toBe('hidden');
    // 2행 구조 — 행 A(문서/앱) 위에 행 B(명령/트랜스포트)
    expect(COMMAND_BAR_SLOT_STYLE.bar.flexDirection).toBe('column');
  });

  it.each(CONTENT_SLOTS)('%s: overflow hidden — 넘침은 형제 위가 아니라 잘린다', (name) => {
    expect(COMMAND_BAR_SLOT_STYLE[name].overflow).toBe('hidden');
  });

  it.each(CONTENT_SLOTS)(
    '%s: justify-content flex-start — 넘침이 좌우 양방향으로 유출되지 않는다',
    (name) => {
      // 구 center 슬롯의 `center`가 정확히 이 버그의 원인이었다.
      expect(COMMAND_BAR_SLOT_STYLE[name].justifyContent).toBe('flex-start');
      expect(COMMAND_BAR_SLOT_STYLE[name].justifyContent).not.toBe('center');
    },
  );

  it.each(CONTENT_SLOTS)('%s: min-width 0 — 내용 최소폭이 슬롯을 밀어내지 못한다', (name) => {
    expect(COMMAND_BAR_SLOT_STYLE[name].minWidth).toBe('0');
  });

  it.each(SHRINKABLE_SLOTS)('%s: flex-shrink 1 — 폭 압박을 흡수한다', (name) => {
    expect(COMMAND_BAR_SLOT_STYLE[name].flexShrink).toBe('1');
  });

  it.each(FIXED_SLOTS)('%s: flex-shrink 0 — P0 액션은 수축하지 않는다', (name) => {
    expect(COMMAND_BAR_SLOT_STYLE[name].flexShrink).toBe('0');
  });

  it('자연어 입력 슬롯은 기준 폭 220px를 확보하고 남는 폭을 먹는다 (헤드라인 기능)', () => {
    expect(MIN_NL_INPUT_WIDTH_PX).toBe(220);
    expect(COMMAND_BAR_SLOT_STYLE.rowBCommand.flexGrow).toBe('1');
    expect(COMMAND_BAR_SLOT_STYLE.rowBCommand.flexBasis).toBe(`${MIN_NL_INPUT_WIDTH_PX}px`);
  });

  it('행 A 좌측이 남는 폭을 먹어 행 A 우측을 오른쪽 끝으로 민다', () => {
    expect(COMMAND_BAR_SLOT_STYLE.rowAStart.flexGrow).toBe('1');
    expect(COMMAND_BAR_SLOT_STYLE.rowAEnd.flexGrow).toBe('0');
  });

  it('행 높이 × 2 = 커맨드바 높이 (theme.LAYOUT과 어긋나지 않는다)', () => {
    expect(COMMAND_BAR_ROW_HEIGHT_PX * 2).toBe(LAYOUT.barHeightPx);
  });
});

describe('레거시 슬롯 별칭 (점진 이행)', () => {
  it('left/center/right는 새 슬롯의 별칭으로 계속 존재한다', () => {
    expect(COMMAND_BAR_LEGACY_SLOT.left).toBe('rowAStart');
    expect(COMMAND_BAR_LEGACY_SLOT.center).toBe('row');
    expect(COMMAND_BAR_LEGACY_SLOT.right).toBe('rowAEnd');
  });

  it('레거시 이름이 가리키는 슬롯도 같은 겹침 계약을 지킨다', () => {
    for (const name of Object.values(COMMAND_BAR_LEGACY_SLOT)) {
      const style = COMMAND_BAR_SLOT_STYLE[name];
      expect(style.overflow).toBe('hidden');
      expect(style.justifyContent).toBe('flex-start');
      expect(style.minWidth).toBe('0');
    }
  });
});

// ── 반응형 밀도 ─────────────────────────────────────────────────────

describe('commandBarDensity (C-2/C-8)', () => {
  it('1180px 이상은 라벨을 유지한다', () => {
    expect(commandBarDensity(1920)).toBe('full');
    expect(commandBarDensity(BREAKPOINT.iconOnlyBarPx)).toBe('full');
  });

  it('1180px 미만은 아이콘 전용 — 라벨만 숨고 aria-label은 남는다', () => {
    expect(commandBarDensity(BREAKPOINT.iconOnlyBarPx - 1)).toBe('iconOnly');
    // 태블릿 가로(1024)에서 회수되는 실측 폭: 트랜스포트 4개 240→112px,
    // 파일 2개 142→56px = 214px
    expect(commandBarDensity(1024)).toBe('iconOnly');
  });

  it('860px 미만은 압축 — 오버플로 메뉴가 열린다', () => {
    expect(commandBarDensity(BREAKPOINT.compactBarPx - 1)).toBe('compact');
    expect(commandBarDensity(640)).toBe('compact');
  });

  it('경계에서 밀도가 뒤집히지 않는다 (폭이 줄면 밀도는 단조 증가한다)', () => {
    const rank = { full: 0, iconOnly: 1, compact: 2 } as const;
    let prev = -1;
    for (let w = 1920; w >= 320; w -= 4) {
      const r = rank[commandBarDensity(w)];
      expect(r).toBeGreaterThanOrEqual(prev);
      prev = r;
    }
  });
});

// ── 오버플로 우선순위 ───────────────────────────────────────────────

describe('overflowMoveOrder (C-2)', () => {
  it('P0(재생 트랜스포트)은 절대 밀려나지 않는다', () => {
    const order = overflowMoveOrder([0, 0, 0, 6, 1]);
    expect(order).not.toContain(0);
    expect(order).not.toContain(1);
    expect(order).not.toContain(2);
  });

  it('우선순위 숫자가 큰 것(=덜 중요한 것)부터 밀려난다', () => {
    // [자연어(P1), 씬(P2), 상태(P3), Step(P4), 뷰(P5), 기타(P6)]
    const order = overflowMoveOrder([1, 2, 3, 4, 5, 6]);
    expect(order).toEqual([5, 4, 3, 2, 1, 0]);
  });

  it('동률은 DOM 순서를 유지한다 (안정 정렬 — 배치가 튀지 않는다)', () => {
    expect(overflowMoveOrder([6, 6, 6])).toEqual([0, 1, 2]);
  });

  it('선언된 우선순위가 명세(P0...P6) 밖으로 벗어나지 않는다', () => {
    const values = Object.values(COMMAND_BAR_PRIORITY);
    expect(Math.min(...values)).toBe(0);
    expect(Math.max(...values)).toBe(6);
    // ▶⏸⏹ > 자연어+생성 > 씬 > 상태 > Step·속도 > 뷰 토글 > 기타
    expect(COMMAND_BAR_PRIORITY.transport).toBeLessThan(COMMAND_BAR_PRIORITY.command);
    expect(COMMAND_BAR_PRIORITY.command).toBeLessThan(COMMAND_BAR_PRIORITY.scene);
    expect(COMMAND_BAR_PRIORITY.scene).toBeLessThan(COMMAND_BAR_PRIORITY.status);
    expect(COMMAND_BAR_PRIORITY.status).toBeLessThan(COMMAND_BAR_PRIORITY.step);
    expect(COMMAND_BAR_PRIORITY.step).toBeLessThan(COMMAND_BAR_PRIORITY.view);
    expect(COMMAND_BAR_PRIORITY.view).toBeLessThan(COMMAND_BAR_PRIORITY.misc);
  });
});

// ── 상태 어휘 (C-18) ────────────────────────────────────────────────
//
// 구 구현은 커맨드바(물리 루프)와 뷰포트 오버레이(시퀀스)가 **같은 어휘**를 써서
// 한 화면에 `● Running 2.66s`와 `● Idle · simTime 2.66s`가 동시에 떠 있었다.
// 커맨드바는 이제 자기가 가리키는 대상을 이름으로 못박는다.

describe('물리 루프 상태 어휘 (C-18)', () => {
  it('시퀀스 어휘(Running/Paused/Idle)를 재사용하지 않는다', () => {
    for (const state of ['playing', 'paused', 'idle'] as const) {
      const label = physicsStateLabel(state);
      expect(label).toMatch(/^물리 /);
      expect(label).not.toMatch(/Running|Paused|Idle/);
    }
  });

  it('세 상태가 서로 다른 낱말을 쓴다 (색만으로 구분하지 않는다 — UX_DESIGN §9)', () => {
    const labels = (['playing', 'paused', 'idle'] as const).map(physicsStateLabel);
    expect(new Set(labels).size).toBe(3);
    expect(labels).toEqual(['물리 가동', '물리 일시정지', '물리 정지']);
  });
});
