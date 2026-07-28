// ui/command-bar/scene-controls.ts — 커맨드바 셸(2행) + 씬 관리 컨트롤 (UX_DESIGN §2/§3.1)
//
// 이 모듈이 소유하는 것은 두 가지다.
//   (1) 커맨드바 셸 — 2행 구조 + 반응형 축약(아이콘 전용 / 우선순위 오버플로)
//   (2) 행 A 좌측 섹션 — 브랜드 워드마크 · 씬 프리셋 <select> · 열기 · 저장 · 저장 상태
//
// ── 왜 2행인가 (UX_AUDIT C-2) ──────────────────────────────────────
// 구 셸은 44px 한 줄에 22개 요소(인터랙티브 18개)를 넣고 center 슬롯에만
// `justify-content: center`를 걸었다. center는 `min-width: 0`으로 0까지 수축하는데
// 내용은 **좌우 대칭으로 넘쳐** 양옆 형제 위에 그려졌고, DOM 뒤 형제가 위에 깔려
// 포인터 이벤트를 가로챘다. 실측: 1280×720에서 겹침 20건, ↶/↷는 100% 피복되어
// 클릭 자체가 불가능했다.
//
// 겹침을 "조정"이 아니라 **구조적으로 불가능**하게 만든다:
//   · 모든 슬롯에 `overflow: hidden` — 넘친 내용은 형제 위가 아니라 잘린다
//   · `justify-content: flex-start` — 넘침이 한쪽으로만 발생한다(양방향 유출 제거)
//   · 수축 가능 슬롯은 `flex-shrink: 1` + `min-width: 0`, 고정 슬롯만 `flex-shrink: 0`
//   · 바 자신도 `flex-wrap: nowrap` + `overflow: hidden`
// 이 계약은 COMMAND_BAR_SLOT_STYLE로 노출되고 layout.test.ts가 단위 검증한다.
//
// ── 반응형 축약 (UX_AUDIT C-2/C-8) ────────────────────────────────
// 폭 감시는 `window.resize`가 아니라 바 자신의 ResizeObserver다 — 스플리터 드래그로
// 바 폭이 바뀌는 경우도 잡아야 한다.
//   < BREAKPOINT.iconOnlyBarPx(1180) → 아이콘 전용(.ui-btn--icon-only, aria-label 유지)
//   < BREAKPOINT.compactBarPx(860)   → 낮은 우선순위부터 '더보기' 팝오버로 이동
// 오버플로는 `data-cmdbar-priority`(0=P0 최우선 … 6=P6) 하나로 선언한다. 속성이 없거나
// 0이면 절대 이동하지 않는다 — 아직 배선되지 않은 요소가 갑자기 사라지지 않는다.
//
// 계층 규칙 (CLAUDE.md §3): core를 import하지 않는다. 씬 전환·검증·저장 대상 spec은
// 글루가 SceneControlsApi 콜백으로 주입한다 — 이 모듈은 파일 I/O(파싱·다운로드)와
// 표시 상태(select 동기화·오류 토스트·저장 상태)만 소유한다.
//
// ── 업로드 JSON 형식 규약 (DATA_MODEL §5/§6) ─────────────────────────
// (a) SceneSpec 단독:            { "name": …, "version": 1, "entities": […], … }
// (b) 씬+시퀀스 봉투(envelope):  { "scene": <SceneSpec>, "sequence": <ControlSequence> }
// 봉투 해석(unwrap)과 스키마 검증은 글루(main.ts) 몫이다 — 이 모듈은 JSON.parse까지만
// 수행하고 파싱 결과를 그대로 넘긴다. 검증 실패는 콘솔 패널 + 인라인 오류로 표시된다.

import { appLog } from '../dock/console-panel';
import { trapFocus } from '../a11y';
import type { FocusTrapHandle } from '../a11y';
import { PRODUCT_NAME } from '../brand';
import { icon, makeIconButton } from '../icons';
import {
  BORDER,
  BREAKPOINT,
  COLLISION,
  COLOR,
  ICON,
  LAYOUT,
  MOTION,
  RADIUS,
  SHADOW,
  SPACE,
  SURFACE,
  TYPE,
  Z_INDEX,
  applyType,
  ensureThemeStyles,
  makeButton,
  styled,
  tr,
} from '../theme';
import type { SceneSpec } from '../../schema';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4, 시각 토큰은 ui/theme.ts) ────

/** 커맨드바 아래 여백(px) — 오류 토스트가 바를 가리지 않는 위치 */
const BELOW_BAR_TOP_PX = LAYOUT.belowBarTopPx;
/** 오류 토스트 자동 숨김 시간(ms) */
const ERROR_AUTO_HIDE_MS = 8000;
/** 오류 토스트 최대 표시 길이 — 초과분은 말줄임 (전체 사유는 콘솔 탭) */
const ERROR_MAX_CHARS = 400;
/** Blob URL 해제 지연(ms) — 다운로드 클릭이 URL을 소비할 시간 여유 */
const BLOB_URL_REVOKE_DELAY_MS = 1000;
/** 업로드 씬(비프리셋) 표시용 select 옵션 값 — 프리셋 이름과 충돌하지 않는 예약값 */
const UPLOADED_OPTION_VALUE = '__uploaded';
/** 활성 씬 없음 표시용 select 옵션 값 — build 단계 실패로 이전 씬까지 잃었을 때 */
const NO_SCENE_OPTION_VALUE = '__none';

/** 커맨드바 한 행의 높이(px) — LAYOUT.barHeightPx(2행)의 절반 */
export const COMMAND_BAR_ROW_HEIGHT_PX = LAYOUT.barHeightPx / 2;

/**
 * 자연어 입력 슬롯의 기본 폭(px). 이 제품의 헤드라인 기능이므로 행 B에서 flex-grow로
 * 남는 폭을 먹되, 기준 폭(flex-basis)을 이 값으로 잡아 트랜스포트 아일랜드에 밀려
 * 사라지지 않게 한다.
 */
export const MIN_NL_INPUT_WIDTH_PX = 220;

/** 오버플로 우선순위 선언 속성 — 값이 클수록 먼저 '더보기'로 밀려난다 */
const PRIORITY_ATTR = 'data-cmdbar-priority';
/** 압축 밀도에서 시각적으로만 숨기는 요소(접근 가능한 이름은 .sr-only로 유지) */
const HIDE_COMPACT_ATTR = 'data-cmdbar-hide-compact';

/**
 * 커맨드바 오버플로 우선순위 (UX_AUDIT C-2).
 * 0(P0)은 절대 숨기지 않는다 — 재생 트랜스포트는 어떤 폭에서도 손에 닿아야 한다.
 */
export const COMMAND_BAR_PRIORITY = {
  /** P0 ▶ ⏸ ⏹ — 절대 숨기지 않는다 */
  transport: 0,
  /** P1 자연어 입력 + 생성 */
  command: 1,
  /** P2 씬 선택 */
  scene: 2,
  /** P3 상태 리드아웃 */
  status: 3,
  /** P4 ⏭ Step · 재생 속도 */
  step: 4,
  /** P5 플로우 · {} JSON */
  view: 5,
  /** P6 열기·저장·↶↷·교체/이어서·자동 정지·설정 */
  misc: 6,
} as const;

/** 압축 밀도(<860px)에서 폭 측정과 무관하게 무조건 밀어내는 최소 우선순위 */
const COMPACT_FORCED_FROM = COMMAND_BAR_PRIORITY.step;

/**
 * 프리셋 씬 한국어 라벨 (라벨 미등록 이름은 이름 그대로 표시).
 * 새 프리셋은 main.ts의 SCENE_REGISTRY에 데이터로 추가된다 — 여기 라벨이 없어도
 * select에는 자동으로 나타난다 (새 씬 = 새 데이터, CLAUDE.md §2.5).
 */
const SCENE_LABELS_KO: Readonly<Record<string, string>> = {
  'falling-boxes': '낙하 박스',
  'arm-and-boxes': '로봇팔·박스',
  'pick-and-place': '픽앤플레이스',
  'obstacle-avoidance': '장애물 회피',
  'collision-testbed': '충돌 테스트베드',
};

// ── 슬롯 스타일 계약 (C-2 — layout.test.ts가 단위 검증한다) ─────────

export type SlotStyle = Readonly<Partial<CSSStyleDeclaration>>;

/** 수축 가능 슬롯의 공통 계약 — 이 네 값이 겹침을 물리적으로 불가능하게 만든다 */
const SHRINKABLE: SlotStyle = {
  flexShrink: '1',
  minWidth: '0',
  overflow: 'hidden',
  justifyContent: 'flex-start',
};

/** 고정 슬롯의 공통 계약 — 수축하지 않지만 넘침은 여전히 잘린다 */
const FIXED: SlotStyle = {
  flexShrink: '0',
  minWidth: '0',
  overflow: 'hidden',
  justifyContent: 'flex-start',
};

export type CommandBarSlotName =
  | 'bar'
  | 'row'
  | 'rowAStart'
  | 'rowAEnd'
  | 'rowBCommand'
  | 'rowBTransport';

/**
 * 커맨드바 슬롯 스타일 계약.
 *
 * **구 버그의 정확한 위치**: 구 `center`는 `justifyContent: 'center'` + `flex: 1` +
 * `minWidth: 0`인데 `overflow` 지정이 없었다. 0까지 수축한 뒤 내용이 좌우로 넘쳐
 * 이웃 슬롯 위에 그려졌다. `justify-content`를 `flex-start`로 되돌리고 `overflow:
 * hidden`을 거는 것만으로 겹침이 사라진다.
 */
export const COMMAND_BAR_SLOT_STYLE: Readonly<Record<CommandBarSlotName, SlotStyle>> = {
  bar: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'stretch',
    flexWrap: 'nowrap',
    overflow: 'hidden',
  },
  /** 행 컨테이너 (행 A / 행 B 공통) — 레거시 `center` 별칭이 가리키는 요소 */
  row: {
    display: 'flex',
    alignItems: 'center',
    flexWrap: 'nowrap',
    gap: SPACE.xl,
    ...SHRINKABLE,
  },
  /** 행 A 좌측 — 워드마크·씬·파일·히스토리 (레거시 `left`) */
  rowAStart: { display: 'flex', alignItems: 'center', gap: SPACE.lg, flexGrow: '1', ...SHRINKABLE },
  /** 행 A 우측 — 뷰 토글·설정·도움말 (레거시 `right`). P0 액션 보호를 위해 고정 폭 */
  rowAEnd: { display: 'flex', alignItems: 'center', gap: SPACE.xs, flexGrow: '0', ...FIXED },
  /** 행 B 좌측 — 자연어 입력(헤드라인 기능이므로 남는 폭을 먹는다) */
  rowBCommand: {
    display: 'flex',
    alignItems: 'center',
    gap: SPACE.sm,
    flexGrow: '1',
    flexBasis: `${MIN_NL_INPUT_WIDTH_PX}px`,
    ...SHRINKABLE,
  },
  /** 행 B 우측 — 재생 트랜스포트 아일랜드 (수축 금지 — P0 보호) */
  rowBTransport: {
    display: 'flex',
    alignItems: 'center',
    gap: SPACE.sm,
    flexGrow: '0',
    ...FIXED,
  },
};

/**
 * 레거시 슬롯 이름 → 새 슬롯. `main.ts`가 아직 `left`/`center`/`right`를 쓰므로
 * 셸은 세 이름을 별칭으로 계속 노출한다(점진 이행).
 */
export const COMMAND_BAR_LEGACY_SLOT = {
  left: 'rowAStart',
  center: 'row',
  right: 'rowAEnd',
} as const satisfies Readonly<Record<'left' | 'center' | 'right', CommandBarSlotName>>;

// ── 반응형 밀도 (순수 — 단위 테스트 대상) ───────────────────────────

/** 커맨드바 밀도 — full(라벨 표시) / iconOnly(라벨 숨김) / compact(+오버플로 메뉴) */
export type CommandBarDensity = 'full' | 'iconOnly' | 'compact';

/**
 * 바 폭 → 밀도. 실측 회수량(아이콘 전용): 트랜스포트 4개 240→112px,
 * 파일 2개 142→56px = 214px. 라벨은 `.ui-btn__label`만 숨으므로 `aria-label`/`title`은
 * 그대로 남는다 — 접근성 손실 0 (UX_AUDIT C-2).
 */
export function commandBarDensity(widthPx: number): CommandBarDensity {
  if (widthPx < BREAKPOINT.compactBarPx) return 'compact';
  if (widthPx < BREAKPOINT.iconOnlyBarPx) return 'iconOnly';
  return 'full';
}

/**
 * 오버플로로 밀어낼 순서(인덱스). 우선순위 숫자가 큰 것(=덜 중요한 것)부터,
 * 동률이면 DOM 순서대로. 우선순위 0(P0)은 절대 포함되지 않는다.
 */
export function overflowMoveOrder(priorities: readonly number[]): number[] {
  return priorities
    .map((priority, index) => ({ priority, index }))
    .filter((it) => it.priority > 0)
    .sort((a, b) => b.priority - a.priority || a.index - b.index)
    .map((it) => it.index);
}

/** 요소에 오버플로 우선순위를 선언한다 (COMMAND_BAR_PRIORITY 참조) */
export function setCommandBarPriority(el: HTMLElement, priority: number): void {
  el.setAttribute(PRIORITY_ATTR, String(priority));
}

/** 압축 밀도에서 시각적으로만 숨긴다(접근 가능한 이름은 .sr-only로 유지) */
export function setCommandBarHideOnCompact(el: HTMLElement): void {
  el.setAttribute(HIDE_COMPACT_ATTR, 'true');
}

// ── 공개 타입 ───────────────────────────────────────────────────────

/** 씬 전환 결과 — 글루(main.ts)의 loadScene 결과를 그대로 전달받는다 */
export interface SceneSwitchResult {
  readonly ok: boolean;
  /** 실패 시 사람이 읽을 수 있는 한국어 오류 목록 */
  readonly errors?: readonly string[];
  /**
   * 실패가 build 단계에서 나 "이전 활성 씬까지 해제된" 경우 true (글루의 loadScene은
   * 검증 실패 시 이전 씬을 보존하지만, build 실패 시에는 이미 teardown이 끝난 뒤다).
   * true면 select를 이전 씬 이름으로 되돌리지 않고 '씬 없음' 상태를 표시한다.
   */
  readonly sceneLost?: boolean;
}

/** 글루(main.ts)가 씬 라이프사이클 위에서 구현해 주입하는 동작 표면 */
export interface SceneControlsApi {
  /** 프리셋 씬으로 전환 (검증 → 이전 씬 해제 → 새로 빌드) */
  switchToPreset(name: string): Promise<SceneSwitchResult>;
  /** 업로드 JSON으로 전환 — payload는 SceneSpec 단독 또는 {scene, sequence} 봉투 */
  switchToUpload(payload: unknown, fileName: string): Promise<SceneSwitchResult>;
  /** 현재 활성 씬의 SceneSpec (저장용 — 활성 씬이 없으면 null) */
  currentSpec(): SceneSpec | null;
  /**
   * 도움말/단축키 표시 (선택). 주입하면 `?` 버튼이 만들어진다 — 배치는 통합자가
   * `opts.helpHost`(보통 shell.rowAEnd)로 지정하거나 handle.helpButton을 직접 붙인다.
   */
  onShowHelp?(): void;
  /**
   * 저장 동작 대체 (선택). 주입하면 기본 SceneSpec 다운로드 대신 이 함수가 실행된다.
   *
   * 기본 동작은 `SceneSpec`만 내보내므로 **시퀀스가 유실된다** — 업로드는
   * `{scene, sequence}` 봉투를 받는데 저장은 봉투를 만들지 않는 비대칭이었다.
   * 통합자가 `ui/document.ts`의 봉투 저장을 주입해 이 비대칭을 닫는다 (UX_AUDIT C-3).
   */
  saveDocument?(): void;
}

export interface SceneControlsOptions {
  /** `?` 도움말 버튼을 마운트할 호스트 (보통 shell.rowAEnd) */
  readonly helpHost?: HTMLElement;
}

export interface SceneControlsHandle {
  readonly el: HTMLElement;
  /** `api.onShowHelp`가 주어졌을 때만 존재하는 `?` 버튼 */
  readonly helpButton: HTMLButtonElement | null;
  /** select 표시 동기화 — 프리셋 이름 또는 null(업로드 씬). 부트 초기화에 사용. */
  setCurrent(presetName: string | null): void;
  /** 저장 상태 표시 (UX_DESIGN §7) — 통합자가 문서 변경/저장 시점에 호출한다 */
  setDirty(dirty: boolean): void;
  /** 인라인 오류 토스트 표시 (한국어) — 커맨드바 아래에 나타나고 자동/클릭으로 닫힌다 */
  showError(message: string): void;
  dispose(): void;
}

/** 커맨드바 셸 — 2행 슬롯 + 반응형 축약 (UX_DESIGN §2/§3.1) */
export interface CommandBarShell {
  readonly el: HTMLElement;
  /** 행 A — 문서/앱 */
  readonly rowA: HTMLElement;
  /** 행 A 좌측: 워드마크 · 씬 ▾ · 열기 · 저장 · ↶ ↷ */
  readonly rowAStart: HTMLElement;
  /** 행 A 우측: 플로우 · {} JSON · ⚙ · ? */
  readonly rowAEnd: HTMLElement;
  /** 행 B — 명령/트랜스포트 */
  readonly rowB: HTMLElement;
  /** 행 B 좌측: 자연어 입력 · 교체/이어서 · 생성 */
  readonly rowBCommand: HTMLElement;
  /** 행 B 우측: ▶ ⏸ ⏹ ⏭ · 속도 · 상태 · 충돌 시 자동 정지 */
  readonly rowBTransport: HTMLElement;
  /** @deprecated `rowAStart` 별칭 (레거시 배선 호환) */
  readonly left: HTMLElement;
  /** @deprecated `rowB` 별칭 (레거시 배선 호환) */
  readonly center: HTMLElement;
  /** @deprecated `rowAEnd` 별칭 (레거시 배선 호환) */
  readonly right: HTMLElement;
  /** 현재 밀도 */
  readonly density: CommandBarDensity;
  /** 콘텐츠를 추가/제거한 뒤 밀도·오버플로를 즉시 재계산 (내부 감시가 있으므로 보통 불필요) */
  refresh(): void;
  dispose(): void;
}

// ── 내부 헬퍼 ───────────────────────────────────────────────────────

/** 프리셋 이름 → select 표시 라벨: "한국어 라벨 (이름)" 또는 이름 그대로 */
function sceneOptionLabel(name: string): string {
  const ko = SCENE_LABELS_KO[name];
  return ko ? `${ko} (${name})` : name;
}

/** 씬 이름 → 안전한 다운로드 파일명 조각 (경로/예약 문자 제거) */
function sanitizeFileName(name: string): string {
  const cleaned = name.replace(/[\\/:*?"<>|\s]+/g, '-').replace(/^-+|-+$/g, '');
  return cleaned.length > 0 ? cleaned : 'scene';
}

const SVG_NS = 'http://www.w3.org/2000/svg';

/**
 * '더보기(⋯)' 글리프. icons.ts에 없는 유일한 형상이라 여기서 같은 규약(16×16 그리드,
 * currentColor, aria-hidden)으로 만든다 — 딩벳 문자를 되살리지 않기 위해서다(C-13).
 */
function moreIcon(sizePx: number = ICON.md): SVGSVGElement {
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('viewBox', '0 0 16 16');
  svg.setAttribute('width', String(sizePx));
  svg.setAttribute('height', String(sizePx));
  svg.setAttribute('fill', 'currentColor');
  svg.setAttribute('aria-hidden', 'true');
  svg.setAttribute('focusable', 'false');
  svg.style.flex = 'none';
  svg.style.display = 'block';
  for (const cx of [3.6, 8, 12.4]) {
    const c = document.createElementNS(SVG_NS, 'circle');
    c.setAttribute('cx', String(cx));
    c.setAttribute('cy', '8');
    c.setAttribute('r', '1.35');
    svg.appendChild(c);
  }
  return svg;
}

/** favicon.svg와 같은 형상의 브랜드 마크 (20×20 인라인 SVG) */
const WORDMARK_MARK_SVG = `
<svg viewBox="0 0 32 32" width="20" height="20" aria-hidden="true" focusable="false" style="display:block;flex:none">
  <defs>
    <linearGradient id="rsw-wordmark-grad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#8B7BFF"/>
      <stop offset="1" stop-color="#5B48E0"/>
    </linearGradient>
  </defs>
  <rect width="32" height="32" rx="8" fill="url(#rsw-wordmark-grad)"/>
  <path d="M7 24h8.5" stroke="#F2F0FF" stroke-width="2.2" stroke-linecap="round" fill="none"/>
  <path d="M11.2 24V16.4L18.6 9" stroke="#F2F0FF" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
  <path d="M18.6 9h4.6" stroke="#F2F0FF" stroke-width="2.6" stroke-linecap="round" fill="none"/>
  <circle cx="11.2" cy="16.4" r="2.5" fill="#0B0D13"/>
  <circle cx="11.2" cy="16.4" r="1.15" fill="#F2F0FF"/>
  <circle cx="18.6" cy="9" r="2.5" fill="#0B0D13"/>
  <circle cx="18.6" cy="9" r="1.15" fill="#F2F0FF"/>
  <rect x="19" y="18.8" width="5.6" height="5.6" rx="1.2" fill="#38BDF8"/>
</svg>`.trim();

/** 얇은 세로 구분선 — 행 안에서 기능 그룹 경계를 보이게 한다 */
function groupDivider(): HTMLElement {
  const el = styled(document.createElement('span'), {
    width: '1px',
    height: '16px',
    background: BORDER.subtle,
    flexShrink: '0',
  });
  el.setAttribute('aria-hidden', 'true');
  return el;
}

// ── 커맨드바 셸 ─────────────────────────────────────────────────────

/** 오버플로로 이동 가능한 항목 1건 */
interface OverflowItem {
  readonly el: HTMLElement;
  readonly priority: number;
  /** 원위치 표식 — 복원은 항상 이 자리로 돌아온다 */
  readonly anchor: Comment;
  readonly prevWidth: string;
  readonly prevJustify: string;
  moved: boolean;
}

/**
 * 응집된 상단 커맨드바(2행)를 host에 마운트하고 슬롯을 돌려준다.
 * 바 위 포인터/휠 상호작용은 stopPropagation으로 흡수한다 — 뷰포트 orbit으로 새지 않게
 * (dock/joint-panel과 동일 규약).
 */
export function mountCommandBarShell(host: HTMLElement): CommandBarShell {
  ensureThemeStyles();

  const bar = styled(document.createElement('div'), {
    position: 'fixed',
    top: '0',
    left: '0',
    right: '0',
    zIndex: Z_INDEX.bar,
    height: `${LAYOUT.barHeightPx}px`,
    padding: `0 ${SPACE.lg}`,
    background: COLOR.bgBar,
    borderBottom: `1px solid ${COLOR.border}`,
    boxShadow: SHADOW.bar,
    color: COLOR.text,
    boxSizing: 'border-box',
    pointerEvents: 'auto',
    ...COMMAND_BAR_SLOT_STYLE.bar,
  });
  applyType(bar, TYPE.body);
  bar.dataset.testid = 'command-bar';
  for (const type of ['pointerdown', 'pointermove', 'pointerup', 'wheel', 'contextmenu']) {
    bar.addEventListener(type, (e) => {
      e.stopPropagation();
    });
  }

  const makeRow = (testId: string, withBorder: boolean): HTMLElement => {
    const row = styled(document.createElement('div'), {
      height: `${COMMAND_BAR_ROW_HEIGHT_PX}px`,
      boxSizing: 'border-box',
      borderBottom: withBorder ? `1px solid ${BORDER.subtle}` : 'none',
      ...COMMAND_BAR_SLOT_STYLE.row,
    });
    row.dataset.testid = testId;
    return row;
  };

  const rowA = makeRow('command-bar-row-a', true);
  const rowB = makeRow('command-bar-row-b', false);

  const makeSlot = (name: CommandBarSlotName, testId: string): HTMLElement => {
    const slot = styled(document.createElement('div'), COMMAND_BAR_SLOT_STYLE[name]);
    slot.dataset.testid = testId;
    return slot;
  };

  const rowAStart = makeSlot('rowAStart', 'command-bar-a-start');
  const rowAEnd = makeSlot('rowAEnd', 'command-bar-a-end');
  const rowBCommand = makeSlot('rowBCommand', 'command-bar-b-command');
  const rowBTransport = makeSlot('rowBTransport', 'command-bar-b-transport');

  rowA.appendChild(rowAStart);
  rowA.appendChild(rowAEnd);
  rowB.appendChild(rowBCommand);
  rowB.appendChild(rowBTransport);
  bar.appendChild(rowA);
  bar.appendChild(rowB);
  host.appendChild(bar);

  // ── '더보기' 오버플로 팝오버 ──────────────────────────────────────
  // 바는 overflow:hidden이므로 팝오버를 바 안에 두면 잘린다 — body에 fixed로 띄운다.
  // 버튼은 항상 바 우상단에 있으므로 좌표는 (바 아래, 우측 정렬) 고정으로 충분하다.

  const overflowButton = makeButton('', '더 많은 도구 보기', 'command-bar-overflow', 'ghost');
  overflowButton.appendChild(moreIcon());
  overflowButton.setAttribute('aria-haspopup', 'true');
  overflowButton.setAttribute('aria-expanded', 'false');
  overflowButton.style.display = 'none';
  rowAEnd.appendChild(overflowButton);

  const popover = styled(document.createElement('div'), {
    position: 'fixed',
    top: `${BELOW_BAR_TOP_PX}px`,
    right: SPACE.lg,
    zIndex: Z_INDEX.slidePanel,
    display: 'none',
    flexDirection: 'column',
    alignItems: 'stretch',
    gap: SPACE.xs,
    minWidth: '208px',
    maxWidth: 'min(320px, 92vw)',
    padding: SPACE.md,
    background: SURFACE.overlay,
    border: `1px solid ${BORDER.default}`,
    borderRadius: RADIUS.lg,
    boxShadow: SHADOW.overlay,
    color: COLOR.text,
    boxSizing: 'border-box',
  });
  applyType(popover, TYPE.body);
  popover.dataset.testid = 'command-bar-overflow-menu';
  popover.setAttribute('role', 'dialog');
  popover.setAttribute('aria-modal', 'true');
  popover.setAttribute('aria-label', '커맨드바 추가 도구');
  for (const type of ['pointerdown', 'pointermove', 'pointerup', 'wheel', 'contextmenu']) {
    popover.addEventListener(type, (e) => {
      e.stopPropagation();
    });
  }
  document.body.appendChild(popover);

  let items: OverflowItem[] = [];
  let density: CommandBarDensity = 'full';
  let popoverOpen = false;
  let trap: FocusTrapHandle | null = null;

  const closePopover = (): void => {
    if (!popoverOpen) return;
    popoverOpen = false;
    popover.style.display = 'none';
    overflowButton.setAttribute('aria-expanded', 'false');
    trap?.release();
    trap = null;
    document.removeEventListener('pointerdown', onDocumentPointerDown, true);
  };

  function onDocumentPointerDown(e: PointerEvent): void {
    const target = e.target;
    if (!(target instanceof Node)) return;
    if (popover.contains(target) || overflowButton.contains(target)) return;
    closePopover();
  }

  const openPopover = (): void => {
    if (popoverOpen) return;
    popoverOpen = true;
    popover.style.display = 'flex';
    overflowButton.setAttribute('aria-expanded', 'true');
    trap = trapFocus(popover, { onEscape: closePopover });
    document.addEventListener('pointerdown', onDocumentPointerDown, true);
  };

  overflowButton.addEventListener('click', () => {
    if (popoverOpen) closePopover();
    else openPopover();
  });

  // ── 밀도 · 오버플로 적용 ─────────────────────────────────────────

  const slots = [rowAStart, rowAEnd, rowBCommand, rowBTransport];

  /** 어느 슬롯이든 내용이 상자를 넘겼는가 (jsdom/node에서는 항상 false — 결정론) */
  const overflowing = (): boolean => slots.some((s) => s.scrollWidth - s.clientWidth > 1);

  const restoreItem = (it: OverflowItem): void => {
    if (!it.moved) return;
    it.el.style.width = it.prevWidth;
    it.el.style.justifyContent = it.prevJustify;
    it.anchor.replaceWith(it.el);
    it.moved = false;
  };

  const moveItem = (it: OverflowItem): void => {
    if (it.moved) return;
    it.el.replaceWith(it.anchor);
    it.el.classList.remove('ui-btn--icon-only'); // 메뉴 안에서는 항상 라벨을 보인다
    it.el.style.width = '100%';
    it.el.style.justifyContent = 'flex-start';
    popover.appendChild(it.el);
    it.moved = true;
  };

  /**
   * 바에 남아 있는 우선순위 선언 요소를 전부 원위치로 되돌린 뒤 다시 수집한다.
   * 씬 전환마다 재생 컨트롤이 dispose/재마운트되므로 목록을 캐시할 수 없다.
   * 결과는 **밀어낼 순서(우선순위 내림차순)** 로 정렬된다.
   */
  const collect = (): void => {
    for (const it of items) restoreItem(it);
    const collected: OverflowItem[] = Array.from(
      bar.querySelectorAll<HTMLElement>(`[${PRIORITY_ATTR}]`),
    )
      .map((el) => ({
        el,
        priority: Number(el.getAttribute(PRIORITY_ATTR) ?? '0'),
        anchor: document.createComment('cmdbar-slot'),
        prevWidth: el.style.width,
        prevJustify: el.style.justifyContent,
        moved: false,
      }))
      .filter((it) => Number.isFinite(it.priority) && it.priority > 0);
    items = overflowMoveOrder(collected.map((it) => it.priority))
      .map((i) => collected[i])
      .filter((it): it is OverflowItem => it !== undefined);
  };

  const applyIconOnly = (iconOnly: boolean): void => {
    for (const btn of bar.querySelectorAll<HTMLButtonElement>('button.ui-btn')) {
      // 라벨 스팬과 아이콘을 둘 다 가진 버튼만 축약할 수 있다 — 아이콘 없는 버튼의
      // 라벨을 숨기면 빈 사각형이 된다.
      const canShrink =
        btn.querySelector('svg') !== null && btn.querySelector('.ui-btn__label') !== null;
      if (canShrink) btn.classList.toggle('ui-btn--icon-only', iconOnly);
    }
  };

  const applyHideOnCompact = (compact: boolean): void => {
    for (const el of bar.querySelectorAll<HTMLElement>(`[${HIDE_COMPACT_ATTR}]`)) {
      el.classList.toggle('sr-only', compact);
    }
  };

  let applying = false;
  const apply = (): void => {
    if (applying) return;
    applying = true;
    try {
      const width = bar.clientWidth > 0 ? bar.clientWidth : window.innerWidth;
      density = commandBarDensity(width);

      // 순서가 중요하다: 먼저 전부 원위치로 되돌리고(폭이 넓어졌을 수 있다) 밀도를
      // 적용해 실제 폭을 확정한 뒤에 넘침을 측정한다.
      collect();
      applyIconOnly(density !== 'full');
      applyHideOnCompact(density === 'compact');

      const forcedFrom = density === 'compact' ? COMPACT_FORCED_FROM : Number.POSITIVE_INFINITY;
      for (const it of items) {
        // items는 우선순위 내림차순 — forcedFrom 미만을 만나면 이후도 전부 미만이다
        if (it.priority >= forcedFrom || overflowing()) moveItem(it);
        else break;
      }

      const movedCount = items.filter((it) => it.moved).length;
      // 셸 마운트 시점에는 rowAEnd가 비어 있어 '더보기'가 첫 자식이 된다 — 통합자가
      // 붙인 도구들 뒤로 항상 밀어 둔다(메뉴 버튼은 그룹의 끝에 있어야 읽힌다).
      if (movedCount > 0 && rowAEnd.lastElementChild !== overflowButton) {
        rowAEnd.appendChild(overflowButton);
      }
      overflowButton.style.display = movedCount > 0 ? '' : 'none';
      overflowButton.title = `더 많은 도구 보기 (${movedCount})`;
      overflowButton.setAttribute('aria-label', overflowButton.title);
      if (movedCount === 0) closePopover();
    } finally {
      applying = false;
      mutations?.takeRecords(); // 자기 DOM 변경으로 재진입하지 않게
    }
  };

  let scheduled = 0;
  const schedule = (): void => {
    if (scheduled !== 0 || applying) return;
    scheduled = requestAnimationFrame(() => {
      scheduled = 0;
      apply();
    });
  };

  // 폭 감시는 window.resize가 아니라 바 자신의 ResizeObserver — 스플리터 드래그로
  // 바 폭이 바뀌는 경우도 잡는다 (UX_AUDIT C-2/C-8).
  const resizeObserver =
    typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(() => schedule());
  resizeObserver?.observe(bar);

  // 씬 전환마다 재생 컨트롤이 재마운트되므로 자식 변화도 감시한다.
  const mutations =
    typeof MutationObserver === 'undefined' ? null : new MutationObserver(() => schedule());
  mutations?.observe(bar, { childList: true, subtree: true });

  schedule();

  return {
    el: bar,
    rowA,
    rowAStart,
    rowAEnd,
    rowB,
    rowBCommand,
    rowBTransport,
    left: rowAStart,
    center: rowB,
    right: rowAEnd,
    get density(): CommandBarDensity {
      return density;
    },
    refresh: apply,
    dispose: (): void => {
      if (scheduled !== 0) cancelAnimationFrame(scheduled);
      resizeObserver?.disconnect();
      mutations?.disconnect();
      closePopover();
      for (const it of items) restoreItem(it);
      items = [];
      popover.remove();
      bar.remove();
    },
  };
}

// ── 씬 컨트롤 (행 A 좌측 섹션) ──────────────────────────────────────

export function mountSceneControls(
  host: HTMLElement,
  presetNames: readonly string[],
  api: SceneControlsApi,
  opts: SceneControlsOptions = {},
): SceneControlsHandle {
  ensureThemeStyles();
  const section = styled(document.createElement('div'), {
    display: 'flex',
    alignItems: 'center',
    gap: SPACE.lg,
    minWidth: '0',
    flexShrink: '1',
    overflow: 'hidden',
  });
  section.dataset.testid = 'scene-controls';

  // ── 브랜드 워드마크 (C-12) ────────────────────────────────────────
  // 실사용 <h1>이 0개라 문서 아웃라인이 비어 있었다(WCAG 2.4.2/1.3.1). 시각 크기는
  // TYPE.title 그대로 두고 태그만 h1로 올린다 — 마크업 한 글자로 닫히는 문제다.
  const wordmark = styled(document.createElement('h1'), {
    display: 'inline-flex',
    alignItems: 'center',
    gap: SPACE.sm,
    margin: '0',
    color: COLOR.textStrong,
    whiteSpace: 'nowrap',
    flexShrink: '0',
  });
  applyType(wordmark, TYPE.title);
  wordmark.dataset.testid = 'app-wordmark';
  const mark = document.createElement('span');
  mark.style.display = 'inline-flex';
  mark.style.flex = 'none';
  mark.innerHTML = WORDMARK_MARK_SVG;
  const wordmarkText = document.createElement('span');
  wordmarkText.textContent = PRODUCT_NAME;
  setCommandBarHideOnCompact(wordmarkText); // 압축 밀도: 시각만 숨기고 이름은 남긴다
  wordmark.appendChild(mark);
  wordmark.appendChild(wordmarkText);
  section.appendChild(wordmark);
  section.appendChild(groupDivider());

  // ── 씬 프리셋 select ──────────────────────────────────────────────
  const sceneGroup = styled(document.createElement('div'), {
    display: 'flex',
    alignItems: 'center',
    gap: SPACE.xs,
    minWidth: '0',
  });
  setCommandBarPriority(sceneGroup, COMMAND_BAR_PRIORITY.scene);

  const select = styled(document.createElement('select'), {
    flexShrink: '1',
    minWidth: '80px',
    maxWidth: '200px',
  });
  select.className = 'ui-select';
  select.dataset.testid = 'scene-select';
  select.title = '씬 프리셋 선택';
  select.setAttribute('aria-label', '씬 프리셋 선택');
  for (const name of presetNames) {
    const option = document.createElement('option');
    option.value = name;
    option.textContent = sceneOptionLabel(name);
    select.appendChild(option);
  }
  // 업로드 씬(비프리셋) 표시용 옵션 — 활성일 때만 보인다
  const uploadedOption = document.createElement('option');
  uploadedOption.value = UPLOADED_OPTION_VALUE;
  uploadedOption.textContent = '업로드 씬';
  uploadedOption.disabled = true;
  uploadedOption.hidden = true;
  select.appendChild(uploadedOption);
  // 활성 씬 없음 표시용 옵션 — build 실패로 이전 씬까지 잃었을 때만 보인다
  const noneOption = document.createElement('option');
  noneOption.value = NO_SCENE_OPTION_VALUE;
  noneOption.textContent = '씬 없음';
  noneOption.disabled = true;
  noneOption.hidden = true;
  select.appendChild(noneOption);
  sceneGroup.appendChild(select);
  section.appendChild(sceneGroup);

  // ── 파일 그룹: 열기 · 저장 · 저장 상태 ────────────────────────────
  const fileGroup = styled(document.createElement('div'), {
    display: 'flex',
    alignItems: 'center',
    gap: SPACE.xs,
    flexShrink: '0',
  });

  const fileInput = document.createElement('input');
  fileInput.type = 'file';
  fileInput.accept = '.json,application/json';
  fileInput.style.display = 'none';
  fileInput.dataset.testid = 'scene-upload-input';
  fileGroup.appendChild(fileInput);

  const uploadButton = makeIconButton(
    'folderOpen',
    '열기',
    '워크셀 열기 (Ctrl+O) — SceneSpec 단독 또는 {scene, sequence} 봉투 JSON',
    'scene-upload',
  );
  setCommandBarPriority(uploadButton, COMMAND_BAR_PRIORITY.misc);
  fileGroup.appendChild(uploadButton);

  const saveButton = makeIconButton('save', '저장', '워크셀 저장 (Ctrl+S)', 'scene-save');
  setCommandBarPriority(saveButton, COMMAND_BAR_PRIORITY.misc);
  fileGroup.appendChild(saveButton);

  // 저장 상태 (UX_DESIGN §7) — 저장 버튼 옆 아이콘 + caption
  const dirtyIndicator = styled(document.createElement('span'), {
    display: 'inline-flex',
    alignItems: 'center',
    gap: SPACE.xs,
    color: COLOR.muted,
    whiteSpace: 'nowrap',
    flexShrink: '0',
    transition: tr('color', MOTION.fast),
  });
  applyType(dirtyIndicator, TYPE.caption);
  dirtyIndicator.dataset.testid = 'scene-dirty';
  dirtyIndicator.setAttribute('role', 'status');
  setCommandBarPriority(dirtyIndicator, COMMAND_BAR_PRIORITY.misc);
  fileGroup.appendChild(dirtyIndicator);
  section.appendChild(fileGroup);

  const setDirty = (dirty: boolean): void => {
    dirtyIndicator.textContent = '';
    const label = document.createElement('span');
    if (dirty) {
      const dot = document.createElement('span');
      dot.className = 'ui-dot ui-dot--paused';
      dot.setAttribute('aria-hidden', 'true');
      dirtyIndicator.appendChild(dot);
      label.textContent = '저장 안 됨';
      dirtyIndicator.style.color = COLOR.warnText;
    } else {
      dirtyIndicator.appendChild(icon('check', ICON.sm));
      label.textContent = '저장됨';
      dirtyIndicator.style.color = COLOR.muted;
    }
    dirtyIndicator.appendChild(label);
  };
  setDirty(false);

  // ── 도움말 버튼 (선택 — 통합자가 helpHost로 배치) ─────────────────
  let helpButton: HTMLButtonElement | null = null;
  const onShowHelp = api.onShowHelp;
  if (onShowHelp !== undefined) {
    helpButton = makeIconButton('help', '', '도움말 · 단축키 (?)', 'help-open', 'ghost');
    helpButton.addEventListener('click', () => {
      onShowHelp();
    });
    opts.helpHost?.appendChild(helpButton);
  }

  // ── 인라인 오류 토스트 (커맨드바 아래, 클릭/자동 닫힘) ────────────
  const errorToast = styled(document.createElement('div'), {
    position: 'fixed',
    top: `${BELOW_BAR_TOP_PX}px`,
    left: SPACE.lg,
    zIndex: Z_INDEX.toast,
    maxWidth: 'min(560px, 80vw)',
    background: COLLISION.surface,
    border: `1px solid ${COLLISION.border}`,
    borderLeft: `3px solid ${COLLISION.base}`,
    borderRadius: RADIUS.md,
    boxShadow: SHADOW.panel,
    padding: `${SPACE.md} ${SPACE.lg}`,
    color: COLLISION.text,
    whiteSpace: 'pre-wrap',
    cursor: 'pointer',
    display: 'none',
  });
  applyType(errorToast, TYPE.monoBody);
  errorToast.dataset.testid = 'scene-error';
  errorToast.title = '클릭하여 닫기';
  errorToast.setAttribute('role', 'alert');
  document.body.appendChild(errorToast);

  let errorHideTimer: ReturnType<typeof setTimeout> | null = null;
  const hideError = (): void => {
    errorToast.style.display = 'none';
    if (errorHideTimer !== null) {
      clearTimeout(errorHideTimer);
      errorHideTimer = null;
    }
  };
  errorToast.addEventListener('click', hideError);

  const showError = (message: string): void => {
    const shown =
      message.length > ERROR_MAX_CHARS
        ? `${message.slice(0, ERROR_MAX_CHARS)}… (전체 사유는 콘솔 탭)`
        : message;
    errorToast.textContent = shown;
    errorToast.style.display = 'block';
    if (errorHideTimer !== null) clearTimeout(errorHideTimer);
    errorHideTimer = setTimeout(hideError, ERROR_AUTO_HIDE_MS);
  };

  // ── 표시 상태 ─────────────────────────────────────────────────────

  /** 마지막으로 "적용에 성공한" select 값 — 전환 실패 시 이 값으로 되돌린다 */
  let lastAppliedValue: string = presetNames[0] ?? UPLOADED_OPTION_VALUE;

  const setCurrent = (presetName: string | null): void => {
    noneOption.hidden = true; // 씬이 다시 활성화됨 — '씬 없음' 상태 해제
    if (presetName === null) {
      uploadedOption.hidden = false;
      select.value = UPLOADED_OPTION_VALUE;
      lastAppliedValue = UPLOADED_OPTION_VALUE;
    } else {
      uploadedOption.hidden = true;
      select.value = presetName;
      lastAppliedValue = presetName;
    }
  };

  /**
   * 전환 실패 후 select 표시 복원: 검증 단계 실패면 이전 씬이 그대로 살아 있으므로
   * 이전 값으로 되돌리고, build 단계 실패(sceneLost)면 이전 씬이 이미 해제되어
   * 활성 씬이 없다 — '씬 없음'을 표시해 실상과 표시를 일치시킨다.
   */
  const restoreAfterFailure = (result: SceneSwitchResult): void => {
    if (result.sceneLost) {
      noneOption.hidden = false;
      select.value = NO_SCENE_OPTION_VALUE;
      lastAppliedValue = NO_SCENE_OPTION_VALUE;
    } else {
      select.value = lastAppliedValue;
    }
  };

  /** 전환 진행 중 컨트롤 잠금 (이중 전환 방지 — 글루의 loadScene도 재진입을 거른다) */
  const setBusy = (busy: boolean): void => {
    select.disabled = busy;
    uploadButton.disabled = busy;
    saveButton.disabled = busy;
  };

  // ── 동작 배선 ─────────────────────────────────────────────────────

  select.addEventListener('change', () => {
    const name = select.value;
    if (name === UPLOADED_OPTION_VALUE || name === NO_SCENE_OPTION_VALUE) return;
    hideError();
    setBusy(true);
    void api
      .switchToPreset(name)
      .then((result) => {
        if (result.ok) {
          setCurrent(name);
        } else {
          restoreAfterFailure(result); // 이전 씬 표시 복귀 또는 '씬 없음'
          showError(
            `씬 전환 실패 (${name}):\n${(result.errors ?? ['알 수 없는 오류']).join('\n')}`,
          );
        }
      })
      .finally(() => {
        setBusy(false);
      });
  });

  uploadButton.addEventListener('click', () => {
    fileInput.click();
  });

  fileInput.addEventListener('change', () => {
    const file = fileInput.files?.[0];
    fileInput.value = ''; // 같은 파일 재선택도 change가 다시 발화하도록 초기화
    if (!file) return;
    hideError();
    setBusy(true);
    void (async (): Promise<void> => {
      let payload: unknown;
      try {
        payload = JSON.parse(await file.text()) as unknown;
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        appLog('error', `씬 업로드 JSON 파싱 실패 (${file.name}): ${msg}`);
        showError(`JSON 파싱 실패 (${file.name}):\n${msg}`);
        return;
      }
      const result = await api.switchToUpload(payload, file.name);
      if (result.ok) {
        setCurrent(null);
      } else {
        restoreAfterFailure(result);
        showError(
          `업로드 씬 로드 실패 (${file.name}):\n${(result.errors ?? ['알 수 없는 오류']).join('\n')}`,
        );
      }
    })().finally(() => {
      setBusy(false);
    });
  });

  saveButton.addEventListener('click', () => {
    if (api.saveDocument !== undefined) {
      api.saveDocument();
      return;
    }
    const spec = api.currentSpec();
    if (!spec) {
      showError('저장할 씬이 없습니다 — 활성 씬이 로드된 뒤 다시 시도하세요');
      return;
    }
    const json = JSON.stringify(spec, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `${sanitizeFileName(spec.name)}.scene.json`;
    anchor.click();
    setTimeout(() => {
      URL.revokeObjectURL(url);
    }, BLOB_URL_REVOKE_DELAY_MS);
    appLog('info', `씬 '${spec.name}' SceneSpec 저장 (${anchor.download})`);
  });

  host.appendChild(section);

  return {
    el: section,
    helpButton,
    setCurrent,
    setDirty,
    showError,
    dispose: (): void => {
      hideError();
      errorToast.remove();
      helpButton?.remove();
      section.remove();
    },
  };
}
