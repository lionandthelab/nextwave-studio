// ui/console/primitives.ts — 콘솔 평면 공유 UI 프리미티브 (docs/BACKEND.md, Phase 12+)
//
// 콘솔 평면(공정·작업·블록·장비·실행 기록·사용자 — 목록/표 중심 화면 6종)이 공유하는
// 조립 블록이다. 화면 모듈은 이 파일의 팩토리만 소비한다 — 화면마다 배지/표/카드를
// 수제로 만들면 스타일·a11y·터치 타깃 규약이 화면 수만큼 분열한다.
//
// ── 1차 사용자: 로봇 설치기사 (BACKEND §1) ──────────────────────────
// 공유 단말 · 장갑 낀 손 · 서두름 전제. 그래서:
//   - 터치 타깃 ≥ 44px (TOUCH_TARGET_MIN_PX — 워크스페이스의 24px 최소보다 크다)
//   - 표 행 높이 ≥ 48px(TABLE_ROW_MIN_HEIGHT_PX), 배지 높이 ≥ 24px(BADGE_MIN_HEIGHT_PX)
//   - 상태는 항상 배지(한국어 라벨 + STATUS 토큰)로 — 색만으로 전달하지 않는다(UX §9)
//   - 빈 상태는 막다른 길이 아니라 다음 행동(actions 버튼)을 안내한다
//
// ── 계층/시각 규칙 (CLAUDE.md §2.9 / §3 / §4-b) ─────────────────────
// core/planner/render를 import하지 않는다. 시각 토큰은 ui/theme.ts만 소비한다 —
// STATUS(상태 배지, 액센트 3분할과 분리된 4번째 의미축)가 이 평면의 핵심 토큰이다.
// 아이콘은 ui/icons.ts. 전역 window keydown을 걸지 않는다 — Escape는 trapFocus(로컬
// 컨테이너 리스너), 그 외 단축키는 통합자가 shortcuts 라우터에 배선한다.
// UI 크롬은 한국어, 도메인 식별자(id·kind)는 영문 원문 — 표 컬럼의 `lang`으로 셀에
// `lang="en"`을 부여한다(한국어 TTS가 CamelCase를 철자로 읽지 않게 — WCAG 3.1.2).
//
// 모듈 top-level에서 DOM을 만지지 않는다 — 순수 헬퍼(디바운스·정렬 비교기·검색 매처·
// 배지 토큰 매핑·dismissIntent)는 node 환경 테스트가 import해도 안전하다
// (primitives.test.ts — toast.test.ts와 같은 관례. DOM 조립 검증은 브라우저 게이트 몫).

import { rovingTabindex, trapFocus } from '../a11y';
import type { FocusTrapHandle, RovingTabindexHandle } from '../a11y';
import { icon, makeIconButton } from '../icons';
import type { IconName } from '../icons';
import {
  BORDER,
  BORDER_WIDTH,
  COLLISION,
  COLOR,
  ICON,
  MOTION,
  RADIUS,
  SHADOW,
  SPACE,
  STATUS,
  SURFACE,
  TYPE,
  Z_INDEX,
  applyType,
  ensureThemeStyles,
  makeButton,
  styled,
  tr,
} from '../theme';
import type { StatusName, StatusToken } from '../theme';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4, 시각 토큰은 ui/theme.ts) ────

/** 콘솔 평면 터치 타깃 하한 — 장갑 낀 손 전제 (BACKEND §1, WCAG 2.5.5 AAA급) */
export const TOUCH_TARGET_MIN_PX = 44;
/** 상태 배지 최소 높이 */
export const BADGE_MIN_HEIGHT_PX = 24;
/** 데이터 표 행 최소 높이 */
export const TABLE_ROW_MIN_HEIGHT_PX = 48;
/** 검색 입력 디바운스 — 타자마다 목록 필터를 다시 돌리지 않는다 */
export const SEARCH_DEBOUNCE_MS = 200;
/** dirty 모달을 닫으려 할 때 확인 문구 (검증 게이트가 문구 회귀를 잡는다) */
export const MODAL_DIRTY_MESSAGE_KO = '저장되지 않은 변경이 있습니다. 닫을까요?';

/** 모달 스크림 — 배경이 조작 불가임을 시각으로도 알린다 (import-dialog와 동일 규약) */
const MODAL_SCRIM = 'rgba(0, 0, 0, 0.45)';
/** 모달 기본 폭 */
const MODAL_WIDTH_DEFAULT_PX = 480;
/** 카드 그리드 최소 카드 폭 (auto-fill 기준) */
const CARD_MIN_WIDTH_PX = 220;
/** 카드 썸네일 종횡비 (뷰포트 캡처와 유사한 가로형) */
const CARD_THUMB_ASPECT = '16 / 9';
/** 빈 상태 아이콘 원판 지름 */
const EMPTY_ICON_BOX_PX = 48;
/** 빈 상태 힌트 문단 최대 폭 (행 길이 가독 상한) */
const EMPTY_HINT_MAX_WIDTH_PX = 340;

// ── 순수 헬퍼 (DOM 비의존 — node 테스트 대상) ───────────────────────

/** 배지 상태 — STATUS 5종 + 'neutral'(의미 없는 라벨 칩: 종류 표기 등) */
export type BadgeStatus = StatusName | 'neutral';

/** neutral 칩 — 상태 의미 없이 정보만 담는 칩 (STATUS 축 바깥, muted 계열) */
const NEUTRAL_BADGE_TOKEN: StatusToken = {
  fg: COLOR.text,
  bg: COLOR.mutedSoft,
  border: BORDER.strong,
};

/** 배지 상태 → {fg, bg, border} 토큰 (STATUS 축 소비 — 색을 여기서 발명하지 않는다) */
export function resolveBadgeToken(status: BadgeStatus): StatusToken {
  return status === 'neutral' ? NEUTRAL_BADGE_TOKEN : STATUS[status];
}

export interface Debounced<T> {
  /** 값을 예약한다 — delayMs 안에 다시 부르면 타이머가 리셋되고 마지막 값만 남는다 */
  call(value: T): void;
  /** 대기 중인 값이 있으면 즉시 발화 (Enter 확정 등) */
  flush(): void;
  /** 대기 중인 값 폐기 (지우기 버튼·dispose) */
  cancel(): void;
  readonly pending: boolean;
}

/** trailing 디바운스 — 마지막 호출 후 delayMs가 지나야 fn이 마지막 값으로 1회 불린다 */
export function createDebounced<T>(fn: (value: T) => void, delayMs: number): Debounced<T> {
  let timer: ReturnType<typeof setTimeout> | null = null;
  let pendingBox: { value: T } | null = null;

  const clearTimer = (): void => {
    if (timer !== null) {
      clearTimeout(timer);
      timer = null;
    }
  };
  const fire = (): void => {
    timer = null;
    const box = pendingBox;
    pendingBox = null;
    if (box !== null) fn(box.value);
  };

  return {
    call(value: T): void {
      pendingBox = { value };
      clearTimer();
      timer = setTimeout(fire, delayMs);
    },
    flush(): void {
      if (pendingBox === null) return;
      clearTimer();
      fire();
    },
    cancel(): void {
      clearTimer();
      pendingBox = null;
    },
    get pending(): boolean {
      return pendingBox !== null;
    },
  };
}

/** 표 셀 정렬 값 — null/undefined는 "값 없음"으로 방향과 무관하게 마지막 */
export type CellValue = string | number | null | undefined;

/** 값 비교 — 숫자쌍은 수치로, 그 외는 한국어 로캘 문자열로 */
export function compareCellValues(a: string | number, b: string | number): number {
  if (typeof a === 'number' && typeof b === 'number') {
    return a === b ? 0 : a < b ? -1 : 1;
  }
  return String(a).localeCompare(String(b), 'ko');
}

/**
 * 행 정렬 비교기. 빈 값(null/undefined)은 **방향과 무관하게 마지막**이다 —
 * 내림차순이라고 "실행 기록 없음"이 맨 위로 튀면 목록이 거꾸로 읽힌다.
 */
export function makeRowComparator<Row>(
  get: (row: Row) => CellValue,
  dir: 'asc' | 'desc' = 'asc',
): (a: Row, b: Row) => number {
  const sign = dir === 'desc' ? -1 : 1;
  return (rowA: Row, rowB: Row): number => {
    const a = get(rowA);
    const b = get(rowB);
    const aEmpty = a === null || a === undefined;
    const bEmpty = b === null || b === undefined;
    if (aEmpty && bEmpty) return 0;
    if (aEmpty) return 1;
    if (bEmpty) return -1;
    return sign * compareCellValues(a, b);
  };
}

/** 검색어 정규화 — NFC(한글 조합) + trim + 소문자 */
export function normalizeQuery(raw: string): string {
  return raw.normalize('NFC').trim().toLowerCase();
}

/** 부분 문자열 매치 (대소문자 무시). 빈 검색어는 전부 매치 — "필터 없음"이다. */
export function matchesQuery(text: string, query: string): boolean {
  const nq = normalizeQuery(query);
  if (nq === '') return true;
  return text.normalize('NFC').toLowerCase().includes(nq);
}

/** 행 목록을 검색어로 거른다 (textOf가 행의 검색 대상 문자열을 만든다) */
export function filterRowsByQuery<Row>(
  rows: readonly Row[],
  query: string,
  textOf: (row: Row) => string,
): Row[] {
  const nq = normalizeQuery(query);
  if (nq === '') return [...rows];
  return rows.filter((row) => textOf(row).normalize('NFC').toLowerCase().includes(nq));
}

export type DismissIntent = 'close' | 'confirm';

/**
 * 모달 닫기 요청(배경 클릭·Escape)의 처리 — dirty면 바로 닫지 않고 확인을 거친다.
 * 파괴적 동작(입력 손실)에는 되돌릴 경로가 있어야 한다(CLAUDE.md §2.11).
 */
export function dismissIntent(dirty: boolean): DismissIntent {
  return dirty ? 'confirm' : 'close';
}

// ── 공용 스타일시트 주입 (1회 — theme.ts와 같은 패턴) ───────────────

const CONSOLE_STYLE_ID = 'rsw-console-styles';

/** 콘솔 평면 공용 스타일을 <head>에 1회 주입한다 (멱등, 모든 팩토리가 먼저 호출) */
export function ensureConsoleStyles(): void {
  if (document.getElementById(CONSOLE_STYLE_ID) !== null) return;
  const style = document.createElement('style');
  style.id = CONSOLE_STYLE_ID;
  style.textContent = `
/* ── 터치 타깃 (장갑 전제 — BACKEND §1) ─────────────────────────── */
.rsw-c-touch { min-height: ${TOUCH_TARGET_MIN_PX}px; }
.rsw-c-touch--square { min-width: ${TOUCH_TARGET_MIN_PX}px; justify-content: center; }

/* ── 상태 배지 (STATUS 토큰 — 색은 인스턴스가 인라인으로 주입) ──── */
.rsw-cbadge {
  display: inline-flex;
  align-items: center;
  gap: ${SPACE.xs};
  box-sizing: border-box;
  min-height: ${BADGE_MIN_HEIGHT_PX}px;
  padding: 0 ${SPACE.md};
  border-radius: ${RADIUS.full};
  border: ${BORDER_WIDTH.hair} solid transparent;
  font-family: var(--rsw-font-ui);
  font-size: ${TYPE.caption.sizePx}px;
  line-height: ${TYPE.caption.lineHeightPx}px;
  font-weight: ${TYPE.micro.weight};
  white-space: nowrap;
}

/* ── 빈 상태 (다음 행동 안내 — 막다른 길 금지) ──────────────────── */
.rsw-cempty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: ${SPACE.lg};
  padding: ${SPACE.xxxl} ${SPACE.xl};
  text-align: center;
}
.rsw-cempty__iconbox {
  width: ${EMPTY_ICON_BOX_PX}px;
  height: ${EMPTY_ICON_BOX_PX}px;
  border-radius: ${RADIUS.full};
  background: ${SURFACE.raised};
  border: ${BORDER_WIDTH.hair} solid ${BORDER.subtle};
  display: grid;
  place-content: center;
  color: ${COLOR.muted};
}
.rsw-cempty__title {
  margin: 0;
  color: ${COLOR.textStrong};
  font-family: var(--rsw-font-ui);
  font-size: ${TYPE.display.sizePx}px;
  line-height: ${TYPE.display.lineHeightPx}px;
  font-weight: ${TYPE.display.weight};
  letter-spacing: ${TYPE.display.letterSpacing};
}
.rsw-cempty__hint {
  margin: 0;
  max-width: ${EMPTY_HINT_MAX_WIDTH_PX}px;
  color: ${COLOR.muted};
  font-family: var(--rsw-font-ui);
  font-size: ${TYPE.body.sizePx}px;
  line-height: ${TYPE.body.lineHeightPx}px;
}
.rsw-cempty__actions {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: ${SPACE.md};
  margin-top: ${SPACE.md};
}

/* ── 검색 필드 ──────────────────────────────────────────────────── */
.rsw-csearch { position: relative; display: block; }
.rsw-csearch__icon {
  position: absolute;
  left: ${SPACE.md};
  top: 50%;
  transform: translateY(-50%);
  display: flex;
  color: ${COLOR.muted};
  pointer-events: none;
}
.rsw-csearch__input {
  width: 100%;
  box-sizing: border-box;
  min-height: ${TOUCH_TARGET_MIN_PX}px;
  padding-left: calc(${SPACE.md} + ${ICON.lg}px + ${SPACE.sm});
  padding-right: ${TOUCH_TARGET_MIN_PX}px;
  font-size: ${TYPE.subhead.sizePx}px;
  line-height: ${TYPE.subhead.lineHeightPx}px;
}
.rsw-csearch__clear {
  position: absolute;
  right: 0;
  top: 50%;
  transform: translateY(-50%);
  min-width: ${TOUCH_TARGET_MIN_PX}px;
  min-height: ${TOUCH_TARGET_MIN_PX}px;
  justify-content: center;
}

/* ── 데이터 표 (테이블 시맨틱 — th scope, 행 ≥48px) ─────────────── */
.rsw-ctable-wrap {
  overflow: auto;
  background: ${SURFACE.panel};
  border: ${BORDER_WIDTH.hair} solid ${BORDER.subtle};
  border-radius: ${RADIUS.md};
}
.rsw-ctable {
  width: 100%;
  border-collapse: collapse;
  color: ${COLOR.text};
  font-family: var(--rsw-font-ui);
  font-size: ${TYPE.body.sizePx}px;
  line-height: ${TYPE.body.lineHeightPx}px;
}
.rsw-ctable th {
  position: sticky;
  top: 0;
  z-index: 1;
  background: ${SURFACE.raised};
  color: ${COLOR.label};
  text-align: left;
  font-size: ${TYPE.caption.sizePx}px;
  line-height: ${TYPE.caption.lineHeightPx}px;
  font-weight: ${TYPE.bodyStrong.weight};
  padding: ${SPACE.sm} ${SPACE.lg};
  border-bottom: ${BORDER_WIDTH.hair} solid ${BORDER.default};
  white-space: nowrap;
}
.rsw-ctable td {
  padding: ${SPACE.sm} ${SPACE.lg};
  border-bottom: ${BORDER_WIDTH.hair} solid ${BORDER.subtle};
  vertical-align: middle;
}
.rsw-ctable tbody tr {
  height: ${TABLE_ROW_MIN_HEIGHT_PX}px;
  transition: ${tr('background-color', MOTION.instant)};
}
.rsw-ctable tbody tr:last-child td { border-bottom: none; }
.rsw-ctable tbody tr[data-clickable='true'] { cursor: pointer; }
.rsw-ctable tbody tr[data-clickable='true']:hover { background: rgba(255, 255, 255, 0.045); }
/* 선택은 SELECT(청색) 축 — 액센트와 분리 (theme.ts 헤더 규약) */
.rsw-ctable tbody tr.rsw-ctable-row--selected {
  background: var(--rsw-select-soft);
  box-shadow: inset ${BORDER_WIDTH.thick} 0 0 var(--rsw-select);
}
.rsw-ctable tbody tr:focus-visible,
.rsw-ccard__main:focus-visible {
  outline: 2px solid var(--rsw-accent);
  outline-offset: -2px;
}

/* ── 카드 그리드 ────────────────────────────────────────────────── */
.rsw-ccardgrid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(${CARD_MIN_WIDTH_PX}px, 1fr));
  gap: ${SPACE.lg};
}
.rsw-ccard {
  display: flex;
  flex-direction: column;
  background: ${SURFACE.raised};
  border: ${BORDER_WIDTH.hair} solid ${BORDER.strong};
  border-radius: ${RADIUS.md};
  overflow: hidden;
  transition: ${tr('border-color', MOTION.instant)}, ${tr('background-color', MOTION.instant)};
}
.rsw-ccard:hover { border-color: ${BORDER.hover}; }
/* 카드의 클릭 표면은 내부 <button> — article 자체에 role을 겹치지 않는다(중첩 인터랙티브 방지) */
.rsw-ccard__main {
  display: flex;
  flex-direction: column;
  align-items: stretch;
  width: 100%;
  min-height: ${TOUCH_TARGET_MIN_PX}px;
  margin: 0;
  padding: 0;
  background: none;
  border: none;
  color: inherit;
  font: inherit;
  text-align: left;
  cursor: pointer;
}
.rsw-ccard__thumb {
  display: block;
  width: 100%;
  aspect-ratio: ${CARD_THUMB_ASPECT};
  object-fit: cover;
  background: ${SURFACE.sunken};
  border-bottom: ${BORDER_WIDTH.hair} solid ${BORDER.subtle};
}
.rsw-ccard__body {
  display: flex;
  flex-direction: column;
  gap: ${SPACE.xs};
  padding: ${SPACE.lg};
  min-width: 0;
}
.rsw-ccard__titlerow {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: ${SPACE.md};
  min-width: 0;
}
.rsw-ccard__title {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: ${COLOR.textStrong};
  font-size: ${TYPE.subhead.sizePx}px;
  line-height: ${TYPE.subhead.lineHeightPx}px;
  font-weight: ${TYPE.subhead.weight};
}
.rsw-ccard__subline {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: ${COLOR.muted};
  font-size: ${TYPE.caption.sizePx}px;
  line-height: ${TYPE.caption.lineHeightPx}px;
}
.rsw-ccard__actions {
  display: flex;
  gap: ${SPACE.sm};
  padding: 0 ${SPACE.lg} ${SPACE.lg};
}

/* ── Windows 고대비 (theme.ts 규약과 짝) ────────────────────────── */
@media (forced-colors: active) {
  .rsw-cbadge { border-color: CanvasText; }
  .rsw-ctable tbody tr.rsw-ctable-row--selected { background: Highlight; color: HighlightText; }
}
`;
  document.head.appendChild(style);
}

// ── 터치 타깃 헬퍼 ──────────────────────────────────────────────────

/**
 * 버튼/컨트롤을 콘솔 터치 타깃(≥44px)으로 키운다. 핵심 동작 버튼(makeButton /
 * makeIconButton 산출물)에 걸어 쓴다 — square는 아이콘 전용 정사각 버튼용.
 */
export function applyTouchTarget<T extends HTMLElement>(
  el: T,
  opts: { square?: boolean } = {},
): T {
  ensureConsoleStyles();
  el.classList.add('rsw-c-touch');
  if (opts.square === true) el.classList.add('rsw-c-touch--square');
  return el;
}

// ── 배지 ────────────────────────────────────────────────────────────

export interface BadgeOptions {
  readonly iconName?: IconName;
  readonly testid?: string;
}

/**
 * 상태 배지 — STATUS 토큰(4번째 의미축) 소비. 라벨이 의미를 전달하고 색은 보조다
 * (UX §9 — 색만으로 상태를 말하지 않는다). 높이 ≥ 24px.
 */
export function makeBadge(
  labelKo: string,
  status: BadgeStatus,
  opts: BadgeOptions = {},
): HTMLSpanElement {
  ensureThemeStyles();
  ensureConsoleStyles();
  const token = resolveBadgeToken(status);
  const el = document.createElement('span');
  el.className = 'rsw-cbadge';
  el.dataset.status = status;
  if (opts.testid !== undefined) el.dataset.testid = opts.testid;
  el.style.color = token.fg;
  el.style.background = token.bg;
  el.style.borderColor = token.border;
  if (opts.iconName !== undefined) el.appendChild(icon(opts.iconName, ICON.sm));
  const text = document.createElement('span');
  text.textContent = labelKo;
  el.appendChild(text);
  return el;
}

// ── 빈 상태 ─────────────────────────────────────────────────────────

export interface EmptyStateOptions {
  readonly iconName: IconName;
  readonly titleKo: string;
  readonly hintKo?: string;
  /** 다음 행동 버튼들 (호출자가 makeButton/makeIconButton으로 만들어 넘긴다) */
  readonly actions: readonly HTMLElement[];
  readonly testid?: string;
}

/**
 * 빈 상태 — "없다"가 아니라 **무엇을 하면 되는지**를 안내한다(설치기사 온보딩).
 * actions의 첫 버튼이 화면의 주요 액션(primary)인 것이 관례다.
 */
export function makeEmptyState(opts: EmptyStateOptions): HTMLElement {
  ensureThemeStyles();
  ensureConsoleStyles();
  const el = document.createElement('div');
  el.className = 'rsw-cempty';
  if (opts.testid !== undefined) el.dataset.testid = opts.testid;

  const iconBox = document.createElement('div');
  iconBox.className = 'rsw-cempty__iconbox';
  iconBox.appendChild(icon(opts.iconName, ICON.xl));
  el.appendChild(iconBox);

  const title = document.createElement('h3');
  title.className = 'rsw-cempty__title';
  title.textContent = opts.titleKo;
  el.appendChild(title);

  if (opts.hintKo !== undefined) {
    const hint = document.createElement('p');
    hint.className = 'rsw-cempty__hint';
    hint.textContent = opts.hintKo;
    el.appendChild(hint);
  }

  if (opts.actions.length > 0) {
    const actions = document.createElement('div');
    actions.className = 'rsw-cempty__actions';
    for (const action of opts.actions) {
      applyTouchTarget(action as HTMLElement);
      actions.appendChild(action);
    }
    el.appendChild(actions);
  }
  return el;
}

// ── 검색 필드 ───────────────────────────────────────────────────────

export interface SearchFieldOptions {
  readonly placeholderKo: string;
  /** 디바운스(200ms) 후 호출. 지우기·setValue는 즉시 호출된다. */
  onInput(query: string): void;
  readonly testid: string;
  /** 기본 SEARCH_DEBOUNCE_MS */
  readonly debounceMs?: number;
}

export interface SearchFieldHandle {
  readonly el: HTMLElement;
  readonly input: HTMLInputElement;
  getValue(): string;
  /** silent:true면 onInput을 부르지 않는다 (외부 상태 복원용) */
  setValue(value: string, opts?: { silent?: boolean }): void;
  focus(): void;
  dispose(): void;
}

/**
 * 검색 필드 — 디바운스 내장(기본 200ms), 지우기 버튼(즉시 반영), Enter로 즉시 확정.
 * Escape는 내용이 있을 때만 지우고 전파를 멈춘다 — 비어 있으면 모달 trapFocus의
 * Escape(닫기)로 자연스럽게 흘러간다.
 */
export function makeSearchField(opts: SearchFieldOptions): SearchFieldHandle {
  ensureThemeStyles();
  ensureConsoleStyles();
  const debounced = createDebounced<string>(
    (q) => opts.onInput(q),
    opts.debounceMs ?? SEARCH_DEBOUNCE_MS,
  );

  const el = document.createElement('div');
  el.className = 'rsw-csearch';
  el.setAttribute('role', 'search');
  el.setAttribute('aria-label', opts.placeholderKo);

  const iconWrap = document.createElement('span');
  iconWrap.className = 'rsw-csearch__icon';
  iconWrap.appendChild(icon('search', ICON.lg));
  el.appendChild(iconWrap);

  const input = document.createElement('input');
  input.type = 'text';
  input.className = 'ui-input rsw-csearch__input';
  input.placeholder = opts.placeholderKo;
  input.setAttribute('aria-label', opts.placeholderKo);
  input.dataset.testid = opts.testid;
  el.appendChild(input);

  const clearButton = makeIconButton('close', '', '검색어 지우기', `${opts.testid}-clear`, 'ghost');
  clearButton.classList.add('rsw-csearch__clear');
  clearButton.style.display = 'none';
  el.appendChild(clearButton);

  const syncClear = (): void => {
    clearButton.style.display = input.value === '' ? 'none' : 'inline-flex';
  };
  const clearNow = (): void => {
    input.value = '';
    debounced.cancel();
    syncClear();
    opts.onInput('');
    input.focus();
  };

  input.addEventListener('input', () => {
    syncClear();
    debounced.call(input.value);
  });
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      debounced.flush();
    } else if (e.key === 'Escape' && input.value !== '') {
      e.preventDefault();
      e.stopPropagation();
      clearNow();
    }
  });
  clearButton.addEventListener('click', clearNow);

  return {
    el,
    input,
    getValue: (): string => input.value,
    setValue: (value: string, o: { silent?: boolean } = {}): void => {
      input.value = value;
      debounced.cancel();
      syncClear();
      if (o.silent !== true) opts.onInput(value);
    },
    focus: (): void => {
      input.focus();
    },
    dispose: (): void => {
      debounced.cancel();
      el.remove();
    },
  };
}

// ── 데이터 표 ───────────────────────────────────────────────────────

export interface ConsoleTableColumn<Row> {
  /** 컬럼 식별자 (정렬 키 등 화면 로직용) */
  readonly key: string;
  /** 헤더 라벨 (한국어 — 도메인 식별자 컬럼이면 영문 원문 그대로도 허용) */
  readonly labelKo: string;
  /** CSS width (예: '120px' | '20%') */
  readonly width?: string;
  /** 셀 내용 언어 — 도메인 식별자(id·kind) 컬럼은 'en' (WCAG 3.1.2) */
  readonly lang?: string;
  render(row: Row): HTMLElement | string;
}

export interface ConsoleTableOptions<Row> {
  readonly columns: readonly ConsoleTableColumn<Row>[];
  readonly rows: readonly Row[];
  /** 있으면 행이 클릭·키보드(roving tabindex + Enter/Space)로 열린다 */
  onRowClick?(row: Row): void;
  rowTestid(row: Row): string;
  /** rows가 비면 표 대신 이 요소를 보인다 (makeEmptyState 산출물) */
  readonly emptyState: HTMLElement;
  /** 표의 접근 가능한 이름 (예: '작업 목록') */
  readonly ariaLabelKo?: string;
}

export interface ConsoleTableHandle<Row> {
  readonly el: HTMLElement;
  setRows(rows: readonly Row[]): void;
  /** 선택 표면(SELECT 축) — null이면 선택 해제 */
  setSelected(match: ((row: Row) => boolean) | null): void;
  dispose(): void;
}

/**
 * 데이터 표 — 진짜 <table> 시맨틱(thead/th scope="col")을 쓴다. 행 높이 ≥ 48px,
 * hover/선택(SELECT 축) 표면, 클릭 가능 행은 roving tabindex로 방향키 탐색.
 */
export function makeDataTable<Row>(opts: ConsoleTableOptions<Row>): ConsoleTableHandle<Row> {
  ensureThemeStyles();
  ensureConsoleStyles();

  const el = document.createElement('div');
  el.className = 'rsw-ctable-wrap ui-scroll';

  const table = document.createElement('table');
  table.className = 'rsw-ctable';
  if (opts.ariaLabelKo !== undefined) table.setAttribute('aria-label', opts.ariaLabelKo);

  const thead = document.createElement('thead');
  const headRow = document.createElement('tr');
  for (const col of opts.columns) {
    const th = document.createElement('th');
    th.scope = 'col';
    th.textContent = col.labelKo;
    if (col.width !== undefined) th.style.width = col.width;
    headRow.appendChild(th);
  }
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = document.createElement('tbody');
  table.appendChild(tbody);
  el.appendChild(table);

  const emptyHost = document.createElement('div');
  emptyHost.appendChild(opts.emptyState);
  emptyHost.style.display = 'none';
  el.appendChild(emptyHost);

  let rows: Row[] = [...opts.rows];
  let rowEls: HTMLElement[] = [];
  let selectedMatch: ((row: Row) => boolean) | null = null;
  let roving: RovingTabindexHandle | null = null;
  const onRowClick = opts.onRowClick;

  const paintSelection = (): void => {
    rowEls.forEach((trEl, i) => {
      const row = rows[i];
      const selected = row !== undefined && selectedMatch !== null && selectedMatch(row);
      trEl.classList.toggle('rsw-ctable-row--selected', selected);
    });
  };

  const renderRows = (): void => {
    tbody.textContent = '';
    rowEls = [];
    for (const row of rows) {
      const trEl = document.createElement('tr');
      trEl.dataset.testid = opts.rowTestid(row);
      if (onRowClick !== undefined) {
        trEl.dataset.clickable = 'true';
        trEl.addEventListener('click', () => onRowClick(row));
      }
      for (const col of opts.columns) {
        const td = document.createElement('td');
        if (col.lang !== undefined) td.setAttribute('lang', col.lang);
        const out = col.render(row);
        if (typeof out === 'string') td.textContent = out;
        else td.appendChild(out);
        trEl.appendChild(td);
      }
      tbody.appendChild(trEl);
      rowEls.push(trEl);
    }
    const empty = rows.length === 0;
    table.style.display = empty ? 'none' : '';
    emptyHost.style.display = empty ? '' : 'none';
    if (onRowClick !== undefined) {
      if (roving === null) {
        roving = rovingTabindex(tbody, rowEls, {
          orientation: 'vertical',
          onActivate: (_el, index) => {
            const row = rows[index];
            if (row !== undefined) onRowClick(row);
          },
        });
      } else {
        roving.setItems(rowEls);
      }
    }
    paintSelection();
  };
  renderRows();

  return {
    el,
    setRows: (next: readonly Row[]): void => {
      rows = [...next];
      renderRows();
    },
    setSelected: (match: ((row: Row) => boolean) | null): void => {
      selectedMatch = match;
      paintSelection();
    },
    dispose: (): void => {
      roving?.dispose();
      el.remove();
    },
  };
}

// ── 카드 그리드 ─────────────────────────────────────────────────────

export interface CardGridHandle {
  readonly el: HTMLElement;
  setCards(cards: readonly HTMLElement[]): void;
  append(card: HTMLElement): void;
  clear(): void;
}

/** 카드 그리드 컨테이너 — 목록↔카드 전환은 화면이 소유한다(이 모듈은 표면만 제공) */
export function makeCardGrid(opts: { testid?: string } = {}): CardGridHandle {
  ensureThemeStyles();
  ensureConsoleStyles();
  const el = document.createElement('div');
  el.className = 'rsw-ccardgrid';
  if (opts.testid !== undefined) el.dataset.testid = opts.testid;
  return {
    el,
    setCards: (cards: readonly HTMLElement[]): void => {
      el.textContent = '';
      for (const card of cards) el.appendChild(card);
    },
    append: (card: HTMLElement): void => {
      el.appendChild(card);
    },
    clear: (): void => {
      el.textContent = '';
    },
  };
}

/** 카드 보조행 — 도메인 식별자면 lang:'en'을 준다 */
export type CardSubline = string | { readonly text: string; readonly lang?: string };

export interface ConsoleCardOptions {
  readonly title: string;
  readonly sublines: readonly CardSubline[];
  /** makeBadge 산출물 (제목행 우측) */
  readonly badge?: HTMLElement;
  /** 썸네일 data URI (TaskDoc.thumbnail) — 없으면 썸네일 영역 생략(밀도 우선) */
  readonly thumbnailDataUri?: string;
  onClick(): void;
  /** 카드 하단 보조 액션 (클릭 전파가 카드 열기로 새지 않는다) */
  readonly actions?: readonly HTMLElement[];
  readonly testid: string;
}

/**
 * 카드 — 클릭 표면은 내부 <button>(썸네일+본문)이고 actions는 그 바깥이다.
 * role="button"을 article에 겹쳐 중첩 인터랙티브를 만들지 않는다.
 */
export function makeCard(opts: ConsoleCardOptions): HTMLElement {
  ensureThemeStyles();
  ensureConsoleStyles();
  const root = document.createElement('article');
  root.className = 'rsw-ccard';

  const main = document.createElement('button');
  main.type = 'button';
  main.className = 'rsw-ccard__main';
  main.dataset.testid = opts.testid;
  main.title = opts.title;
  main.addEventListener('click', () => opts.onClick());

  if (opts.thumbnailDataUri !== undefined) {
    const img = document.createElement('img');
    img.className = 'rsw-ccard__thumb';
    img.src = opts.thumbnailDataUri;
    img.alt = '';
    main.appendChild(img);
  }

  const body = document.createElement('div');
  body.className = 'rsw-ccard__body';

  const titleRow = document.createElement('div');
  titleRow.className = 'rsw-ccard__titlerow';
  const titleEl = document.createElement('span');
  titleEl.className = 'rsw-ccard__title';
  titleEl.textContent = opts.title;
  titleRow.appendChild(titleEl);
  if (opts.badge !== undefined) titleRow.appendChild(opts.badge);
  body.appendChild(titleRow);

  for (const subline of opts.sublines) {
    const div = document.createElement('div');
    div.className = 'rsw-ccard__subline';
    if (typeof subline === 'string') {
      div.textContent = subline;
    } else {
      div.textContent = subline.text;
      if (subline.lang !== undefined) div.setAttribute('lang', subline.lang);
    }
    body.appendChild(div);
  }
  main.appendChild(body);
  root.appendChild(main);

  if (opts.actions !== undefined && opts.actions.length > 0) {
    const actions = document.createElement('div');
    actions.className = 'rsw-ccard__actions';
    actions.addEventListener('click', (e) => {
      e.stopPropagation();
    });
    for (const action of opts.actions) actions.appendChild(action);
    root.appendChild(actions);
  }
  return root;
}

// ── 확인 바 ─────────────────────────────────────────────────────────

export interface ConfirmBarOptions {
  readonly messageKo: string;
  readonly confirmLabelKo: string;
  /** 기본 '취소' */
  readonly cancelLabelKo?: string;
  /** 파괴적 확인이면 true(기본) — danger 버튼 + 충돌 램프 표면 */
  readonly danger?: boolean;
  onConfirm(): void;
  onCancel?(): void;
  readonly testid?: string;
}

export interface ConfirmBarHandle {
  readonly el: HTMLElement;
  focusConfirm(): void;
}

/**
 * 인라인 확인 바 — window.confirm 대신 쓰는 파괴적 동작 2단 확인. 모달 dirty 닫기,
 * 행 삭제(soft-delete — 되돌리기는 toast의 실행취소가 맡는다) 등에서 재사용한다.
 */
export function makeConfirmBar(opts: ConfirmBarOptions): ConfirmBarHandle {
  ensureThemeStyles();
  ensureConsoleStyles();
  const danger = opts.danger ?? true;
  const testid = opts.testid ?? 'confirm-bar';

  const el = styled(document.createElement('div'), {
    display: 'flex',
    alignItems: 'center',
    flexWrap: 'wrap',
    gap: SPACE.md,
    padding: `${SPACE.md} ${SPACE.xl}`,
    background: danger ? COLLISION.surface : SURFACE.overlay,
    borderTop: `${BORDER_WIDTH.hair} solid ${danger ? COLLISION.border : BORDER.subtle}`,
    boxSizing: 'border-box',
  });
  el.setAttribute('role', 'group');
  el.setAttribute('aria-label', opts.messageKo);
  el.dataset.testid = testid;

  const message = applyType(document.createElement('span'), TYPE.body);
  styled(message, {
    flex: '1 1 auto',
    minWidth: '0',
    color: danger ? COLLISION.text : COLOR.text,
  });
  message.textContent = opts.messageKo;
  el.appendChild(message);

  const onCancel = opts.onCancel;
  if (onCancel !== undefined) {
    const cancelLabel = opts.cancelLabelKo ?? '취소';
    const cancelButton = makeButton(cancelLabel, cancelLabel, `${testid}-cancel`, 'ghost');
    applyTouchTarget(cancelButton);
    cancelButton.addEventListener('click', onCancel);
    el.appendChild(cancelButton);
  }

  const confirmButton = makeButton(
    opts.confirmLabelKo,
    opts.confirmLabelKo,
    `${testid}-confirm`,
    danger ? 'danger' : 'primary',
  );
  applyTouchTarget(confirmButton);
  confirmButton.addEventListener('click', opts.onConfirm);
  el.appendChild(confirmButton);

  return {
    el,
    focusConfirm: (): void => {
      confirmButton.focus();
    },
  };
}

// ── 모달 셸 ─────────────────────────────────────────────────────────

export interface ModalShellOptions {
  readonly titleKo: string;
  /** 사용자 상호작용(배경/Esc/닫기 버튼/확인)으로 실제 닫혔을 때 통지 */
  onClose(): void;
  /** 기본 480 */
  readonly widthPx?: number;
  readonly testid?: string;
}

export interface ModalShellHandle {
  /** 오버레이 루트 — 호출자가 host에 append한다 */
  readonly root: HTMLElement;
  /** 내용 영역 (세로 스크롤) */
  readonly body: HTMLElement;
  /** 하단 버튼 영역 */
  readonly footer: HTMLElement;
  /** 연다 — dirty는 false로 리셋된다(새 폼 시작) */
  open(): void;
  /** 프로그램적 닫기 — onClose를 부르지 않는다 (저장 성공 후 화면이 스스로 닫을 때) */
  close(): void;
  /** true면 배경 클릭·Escape가 바로 닫지 않고 확인 바를 거친다 */
  setDirty(dirty: boolean): void;
  isOpen(): boolean;
  dispose(): void;
}

/**
 * 모달 셸 — trapFocus(Tab 순환 + Escape + 진입 전 포커스 복원), 배경 클릭 닫기,
 * **dirty면 확인 바를 거친다**(dismissIntent — 입력 손실은 파괴적 동작이다 §2.11).
 * import-dialog와 동일한 오버레이 규약: 포인터 이벤트를 흡수해 뷰포트로 새지 않는다.
 */
export function makeModalShell(opts: ModalShellOptions): ModalShellHandle {
  ensureThemeStyles();
  ensureConsoleStyles();
  const testid = opts.testid ?? 'console-modal';

  const root = styled(document.createElement('div'), {
    position: 'fixed',
    inset: '0',
    zIndex: Z_INDEX.modal,
    display: 'none',
    alignItems: 'center',
    justifyContent: 'center',
    background: MODAL_SCRIM,
    pointerEvents: 'auto',
  });
  root.dataset.testid = testid;
  for (const type of ['pointerdown', 'pointermove', 'pointerup', 'wheel', 'contextmenu']) {
    root.addEventListener(type, (e) => {
      e.stopPropagation();
    });
  }

  const panel = applyType(document.createElement('div'), TYPE.body);
  styled(panel, {
    width: `${opts.widthPx ?? MODAL_WIDTH_DEFAULT_PX}px`,
    maxWidth: '94vw',
    maxHeight: '88vh',
    display: 'flex',
    flexDirection: 'column',
    background: SURFACE.modal,
    border: `${BORDER_WIDTH.hair} solid ${BORDER.default}`,
    borderRadius: RADIUS.lg,
    boxShadow: SHADOW.modal,
    color: COLOR.text,
    boxSizing: 'border-box',
    overflow: 'hidden',
  });
  panel.setAttribute('role', 'dialog');
  panel.setAttribute('aria-modal', 'true');
  panel.setAttribute('aria-labelledby', `${testid}-title`);
  panel.tabIndex = -1;
  root.appendChild(panel);

  const header = styled(document.createElement('div'), {
    display: 'flex',
    alignItems: 'center',
    gap: SPACE.md,
    padding: `${SPACE.lg} ${SPACE.xl}`,
    borderBottom: `${BORDER_WIDTH.hair} solid ${BORDER.subtle}`,
    flex: 'none',
  });
  const titleEl = applyType(document.createElement('h2'), TYPE.display);
  styled(titleEl, {
    margin: '0',
    flex: '1 1 auto',
    minWidth: '0',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    color: COLOR.textStrong,
  });
  titleEl.id = `${testid}-title`;
  titleEl.textContent = opts.titleKo;
  header.appendChild(titleEl);

  const closeButton = makeIconButton('close', '', '닫기 (Esc)', `${testid}-close`, 'ghost');
  applyTouchTarget(closeButton, { square: true });
  header.appendChild(closeButton);
  panel.appendChild(header);

  const body = styled(document.createElement('div'), {
    display: 'flex',
    flexDirection: 'column',
    gap: SPACE.lg,
    padding: SPACE.xl,
    overflowY: 'auto',
    flex: '1 1 auto',
    minHeight: '0',
  });
  body.className = 'ui-scroll';
  panel.appendChild(body);

  /** dirty 확인 바가 footer 위에 끼워지는 자리 */
  const confirmHost = styled(document.createElement('div'), { display: 'none', flex: 'none' });
  panel.appendChild(confirmHost);

  const footer = styled(document.createElement('div'), {
    display: 'flex',
    justifyContent: 'flex-end',
    gap: SPACE.md,
    padding: `${SPACE.lg} ${SPACE.xl}`,
    borderTop: `${BORDER_WIDTH.hair} solid ${BORDER.subtle}`,
    flex: 'none',
  });
  panel.appendChild(footer);

  let dirty = false;
  let trap: FocusTrapHandle | null = null;
  let confirmBar: ConfirmBarHandle | null = null;
  let open = false;

  const removeConfirm = (): void => {
    if (confirmBar !== null) {
      confirmBar.el.remove();
      confirmBar = null;
    }
    confirmHost.style.display = 'none';
  };

  const hide = (): void => {
    open = false;
    root.style.display = 'none';
    trap?.release();
    trap = null;
    removeConfirm();
  };

  const userClose = (): void => {
    hide();
    opts.onClose();
  };

  const showDirtyConfirm = (): void => {
    if (confirmBar === null) {
      confirmBar = makeConfirmBar({
        messageKo: MODAL_DIRTY_MESSAGE_KO,
        confirmLabelKo: '닫기',
        cancelLabelKo: '계속 편집',
        danger: true,
        testid: `${testid}-dirty`,
        onConfirm: userClose,
        onCancel: (): void => {
          removeConfirm();
          panel.focus();
        },
      });
      confirmHost.appendChild(confirmBar.el);
    }
    confirmHost.style.display = 'block';
    confirmBar.focusConfirm();
  };

  const requestClose = (): void => {
    if (dismissIntent(dirty) === 'close') {
      userClose();
      return;
    }
    showDirtyConfirm();
  };

  root.addEventListener('click', (e) => {
    if (e.target === root) requestClose();
  });
  closeButton.addEventListener('click', requestClose);

  return {
    root,
    body,
    footer,
    open: (): void => {
      dirty = false;
      removeConfirm();
      root.style.display = 'flex';
      open = true;
      trap?.release();
      trap = trapFocus(panel, { onEscape: requestClose });
    },
    close: hide,
    setDirty: (d: boolean): void => {
      dirty = d;
    },
    isOpen: (): boolean => open,
    dispose: (): void => {
      hide();
      root.remove();
    },
  };
}
