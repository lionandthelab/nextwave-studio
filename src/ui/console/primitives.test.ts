// ui/console/primitives.test.ts — 콘솔 프리미티브 순수 로직 단위 테스트 (DOM 비의존, node)
//
// makeBadge/makeDataTable/makeModalShell 등의 DOM 조립·배선은 브라우저 게이트 몫이다
// (toast.test.ts와 같은 관례). 여기서는 이 슬라이스의 순수 계약만 검증한다:
// - 설치기사 치수 계약 (터치 44 · 배지 24 · 표 행 48 · 디바운스 200)
// - 배지 상태 → STATUS 토큰 매핑 (error = COLLISION 정합, neutral은 축 바깥)
// - 디바운스 타이밍 (가짜 타이머 — coalesce · 리셋 · flush · cancel)
// - 표 정렬 비교기 (빈 값은 방향 무관 마지막) · 검색 매처(한글 NFC · 대소문자 무시)
// - 모달 dirty 닫기 의도 (dismissIntent)

import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  BADGE_MIN_HEIGHT_PX,
  SEARCH_DEBOUNCE_MS,
  TABLE_ROW_MIN_HEIGHT_PX,
  TOUCH_TARGET_MIN_PX,
  compareCellValues,
  createDebounced,
  dismissIntent,
  filterRowsByQuery,
  makeRowComparator,
  matchesQuery,
  normalizeQuery,
  resolveBadgeToken,
} from './primitives';
import { BORDER, COLLISION, COLOR, STATUS } from '../theme';

// ── 치수 계약 (BACKEND §1 — 공유 단말·장갑) ─────────────────────────

describe('콘솔 치수 상수', () => {
  it('터치 타깃 ≥ 44px, 배지 ≥ 24px, 표 행 ≥ 48px', () => {
    expect(TOUCH_TARGET_MIN_PX).toBeGreaterThanOrEqual(44);
    expect(BADGE_MIN_HEIGHT_PX).toBeGreaterThanOrEqual(24);
    expect(TABLE_ROW_MIN_HEIGHT_PX).toBeGreaterThanOrEqual(48);
  });

  it('검색 디바운스는 200ms (임무 명세 고정값)', () => {
    expect(SEARCH_DEBOUNCE_MS).toBe(200);
  });
});

// ── 배지 상태 → 토큰 매핑 ───────────────────────────────────────────

describe('resolveBadgeToken', () => {
  it('STATUS 5종은 theme의 STATUS 토큰을 그대로 돌려준다 (색을 발명하지 않는다)', () => {
    for (const name of Object.keys(STATUS) as (keyof typeof STATUS)[]) {
      expect(resolveBadgeToken(name)).toBe(STATUS[name]);
    }
  });

  it('모든 토큰은 fg/bg/border 3튜플이 비어 있지 않다', () => {
    const statuses = [...(Object.keys(STATUS) as (keyof typeof STATUS)[]), 'neutral'] as const;
    for (const name of statuses) {
      const token = resolveBadgeToken(name);
      expect(token.fg.length).toBeGreaterThan(0);
      expect(token.bg.length).toBeGreaterThan(0);
      expect(token.border.length).toBeGreaterThan(0);
    }
  });

  it('error는 COLLISION 램프와 시각 정합이다 (C-7 단일 램프 — 새 빨강 hue 금지)', () => {
    const token = resolveBadgeToken('error');
    expect(token.fg).toBe(COLLISION.text);
    expect(token.bg).toBe(COLLISION.soft);
    expect(token.border).toBe(COLLISION.border);
  });

  it("'neutral'은 STATUS 축 바깥의 muted 라벨 칩이다", () => {
    const token = resolveBadgeToken('neutral');
    expect(token.fg).toBe(COLOR.text);
    expect(token.bg).toBe(COLOR.mutedSoft);
    expect(token.border).toBe(BORDER.strong);
    // 어느 STATUS 토큰과도 같은 객체가 아니다
    for (const name of Object.keys(STATUS) as (keyof typeof STATUS)[]) {
      expect(token).not.toBe(STATUS[name]);
    }
  });
});

// ── 디바운스 타이밍 (가짜 타이머) ───────────────────────────────────

describe('createDebounced', () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it('연속 호출은 마지막 값 1회로 합쳐진다', () => {
    vi.useFakeTimers();
    const fn = vi.fn();
    const d = createDebounced<string>(fn, SEARCH_DEBOUNCE_MS);
    d.call('a');
    d.call('ab');
    d.call('abc');
    vi.advanceTimersByTime(SEARCH_DEBOUNCE_MS - 1);
    expect(fn).not.toHaveBeenCalled();
    vi.advanceTimersByTime(1);
    expect(fn).toHaveBeenCalledTimes(1);
    expect(fn).toHaveBeenCalledWith('abc');
  });

  it('호출마다 타이머가 리셋된다 (trailing — 타자 중에는 발화하지 않는다)', () => {
    vi.useFakeTimers();
    const fn = vi.fn();
    const d = createDebounced<string>(fn, 200);
    d.call('a');
    vi.advanceTimersByTime(150);
    d.call('ab');
    vi.advanceTimersByTime(150); // 마지막 호출 후 150ms — 아직
    expect(fn).not.toHaveBeenCalled();
    vi.advanceTimersByTime(50); // 마지막 호출 후 200ms
    expect(fn).toHaveBeenCalledTimes(1);
    expect(fn).toHaveBeenCalledWith('ab');
  });

  it('flush는 대기 값을 즉시 발화하고 타이머를 걷는다 (Enter 확정)', () => {
    vi.useFakeTimers();
    const fn = vi.fn();
    const d = createDebounced<string>(fn, 200);
    d.call('질의');
    expect(d.pending).toBe(true);
    d.flush();
    expect(fn).toHaveBeenCalledTimes(1);
    expect(fn).toHaveBeenCalledWith('질의');
    expect(d.pending).toBe(false);
    vi.advanceTimersByTime(500); // 이중 발화 없음
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it('대기 값이 없으면 flush는 no-op이다', () => {
    vi.useFakeTimers();
    const fn = vi.fn();
    const d = createDebounced<string>(fn, 200);
    d.flush();
    expect(fn).not.toHaveBeenCalled();
  });

  it('cancel은 대기 값을 폐기한다 (지우기 버튼)', () => {
    vi.useFakeTimers();
    const fn = vi.fn();
    const d = createDebounced<string>(fn, 200);
    d.call('버릴 값');
    d.cancel();
    expect(d.pending).toBe(false);
    vi.advanceTimersByTime(500);
    expect(fn).not.toHaveBeenCalled();
  });
});

// ── 표 정렬 비교기 ──────────────────────────────────────────────────

describe('compareCellValues', () => {
  it('숫자쌍은 수치 비교', () => {
    expect(compareCellValues(1, 2)).toBeLessThan(0);
    expect(compareCellValues(10, 2)).toBeGreaterThan(0);
    expect(compareCellValues(3, 3)).toBe(0);
  });

  it('문자열은 로캘 비교 (한글 포함)', () => {
    expect(compareCellValues('가', '나')).toBeLessThan(0);
    expect(compareCellValues('b', 'a')).toBeGreaterThan(0);
    expect(compareCellValues('같음', '같음')).toBe(0);
  });

  it('숫자·문자열 혼합은 문자열로 강등해 비교한다 (타입 혼합 컬럼 안전망)', () => {
    expect(compareCellValues(2, '2')).toBe(0);
  });
});

describe('makeRowComparator', () => {
  interface R {
    name: string;
    lastRunSec: number | null;
  }
  const rows: R[] = [
    { name: 'b-작업', lastRunSec: 30 },
    { name: 'a-작업', lastRunSec: null },
    { name: 'c-작업', lastRunSec: 10 },
  ];

  it('오름차순 정렬 + 빈 값(null)은 마지막', () => {
    const sorted = [...rows].sort(makeRowComparator((r) => r.lastRunSec, 'asc'));
    expect(sorted.map((r) => r.name)).toEqual(['c-작업', 'b-작업', 'a-작업']);
  });

  it('내림차순이어도 빈 값은 여전히 마지막이다 ("기록 없음"이 맨 위로 튀지 않는다)', () => {
    const sorted = [...rows].sort(makeRowComparator((r) => r.lastRunSec, 'desc'));
    expect(sorted.map((r) => r.name)).toEqual(['b-작업', 'c-작업', 'a-작업']);
  });

  it('문자열 키 정렬', () => {
    const sorted = [...rows].sort(makeRowComparator((r) => r.name, 'asc'));
    expect(sorted.map((r) => r.name)).toEqual(['a-작업', 'b-작업', 'c-작업']);
  });

  it('dir 기본값은 asc', () => {
    const sorted = [...rows].sort(makeRowComparator((r) => r.name));
    expect(sorted[0]?.name).toBe('a-작업');
  });

  it('모두 빈 값이면 순서 유지(0)', () => {
    const cmp = makeRowComparator<R>(() => null);
    expect(cmp(rows[0] as R, rows[1] as R)).toBe(0);
  });
});

// ── 검색 매처 ───────────────────────────────────────────────────────

describe('normalizeQuery / matchesQuery', () => {
  it('빈/공백 검색어는 전부 매치 — 필터 없음', () => {
    expect(matchesQuery('픽앤플레이스 작업', '')).toBe(true);
    expect(matchesQuery('픽앤플레이스 작업', '   ')).toBe(true);
  });

  it('대소문자 무시 부분 일치 (도메인 식별자는 영문)', () => {
    expect(matchesQuery('MoveJoints 노드', 'movejoints')).toBe(true);
    expect(matchesQuery('arm-6 로봇', 'ARM')).toBe(true);
    expect(matchesQuery('conveyor-1', 'belt')).toBe(false);
  });

  it('한국어 부분 일치 + 검색어 양끝 공백 무시', () => {
    expect(matchesQuery('컨베이어 픽앤플레이스', '픽앤')).toBe(true);
    expect(matchesQuery('컨베이어 픽앤플레이스', '  픽앤  ')).toBe(true); // 양끝 공백은 trim
    expect(matchesQuery('컨베이어 픽앤플레이스', '픽 앤')).toBe(false); // 내부 공백은 리터럴
  });

  it('normalizeQuery는 NFC + trim + 소문자', () => {
    expect(normalizeQuery('  ABC  ')).toBe('abc');
    // NFD(조합형)로 들어온 한글도 NFC로 접힌다
    expect(normalizeQuery('가나'.normalize('NFD'))).toBe('가나');
  });

  it('NFD 텍스트도 NFC 검색어와 매치된다 (macOS 파일명 유래 문자열 방어)', () => {
    expect(matchesQuery('공정A'.normalize('NFD'), '공정')).toBe(true);
  });
});

describe('filterRowsByQuery', () => {
  const rows = [
    { id: 'task-pick', name: '픽앤플레이스' },
    { id: 'task-weld', name: '용접 셀' },
    { id: 'task-conv', name: '컨베이어 이송' },
  ];
  const textOf = (r: (typeof rows)[number]): string => `${r.id} ${r.name}`;

  it('이름·id 어느 쪽으로도 걸러진다', () => {
    expect(filterRowsByQuery(rows, '용접', textOf).map((r) => r.id)).toEqual(['task-weld']);
    expect(filterRowsByQuery(rows, 'CONV', textOf).map((r) => r.id)).toEqual(['task-conv']);
  });

  it('빈 검색어는 사본 전체를 돌려준다 (원본 배열과 다른 참조)', () => {
    const out = filterRowsByQuery(rows, '', textOf);
    expect(out).toEqual(rows);
    expect(out).not.toBe(rows);
  });

  it('매치 없음 → 빈 배열 (화면은 emptyState로 전환)', () => {
    expect(filterRowsByQuery(rows, '존재하지 않음', textOf)).toEqual([]);
  });
});

// ── 모달 dirty 닫기 의도 ────────────────────────────────────────────

describe('dismissIntent', () => {
  it('dirty가 아니면 바로 닫는다', () => {
    expect(dismissIntent(false)).toBe('close');
  });

  it('dirty면 확인을 거친다 — 입력 손실은 파괴적 동작이다 (§2.11)', () => {
    expect(dismissIntent(true)).toBe('confirm');
  });
});
