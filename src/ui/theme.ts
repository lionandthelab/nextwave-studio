// ui/theme.ts — 디자인 시스템 단일 진실 (UX_DESIGN §1-2 원칙, docs/UX_AUDIT.md C-11/C-14)
//
// 모든 패널(command-bar / dock / inspector / joint-panel / viewport 오버레이 / flow-graph)이
// 이 모듈의 토큰만 소비한다 — 모듈별 ad-hoc 색상·크기 하드코딩 금지. 시각 언어는 한 곳에서
// 결정한다.
//
// ── 토큰 축 (7개) ──────────────────────────────────────────────────
//   SURFACE  표면 고도 6단계 (각 단계가 이전 대비 대비비 ≥1.14:1 — 계산 검증됨)
//   BORDER   경계 4단계 (항상 얹힌 표면보다 밝다 — 구분선이 표면에 먹히지 않는다)
//   TYPE     타입 스케일 7 + 모노 3 (size/lineHeight/weight 3-튜플, 4px 그리드 정렬)
//   SPACE    간격 9단계 (1/2/4/6/8/12/16/24/32)
//   RADIUS   반경 5단계
//   MOTION   duration/easing 7단계 (이탈은 항상 진입보다 빠르다)
//   ICON     아이콘 치수 4단계 (icons.ts와 짝)
//
// ── 시각 언어 ──────────────────────────────────────────────────────
// - 표면: 차가운 블루-바이올렛 언더톤의 근-블랙. 중성 회색이 아니다 — 뷰포트 안의
//   따뜻한 로봇/오브젝트와 보색 대비를 이뤄 크롬이 뒤로 물러난다.
// - 액센트: 일렉트릭 바이올렛 #7C6AF6. **역할을 3갈래로 분리**한다(UX_AUDIT C-14):
//     · 주요 액션(1화면 1개) → accent 면(fill) + onAccent 텍스트
//     · 활성/토글 상태      → accent 보더 + accentText
//     · 선택(selection)     → **accent를 쓰지 않는다.** SELECT(스카이블루)
//   3D 뷰포트에서 "선택됨"과 "실행 중"이 같은 색이면 구분이 불가능하다.
//   청색 선택은 Blender/Onshape/Isaac Sim 공통 관행이다.
// - 충돌: COLLISION 램프 1개에서 UI·3D가 모두 파생된다(UX_AUDIT C-7). 이 제품의
//   핵심 사건에 빨강이 4개 hue로 흩어져 있던 문제를 닫는다.
// - 폰트: Pretendard Variable(한글+라틴 통합 메트릭, 자체 호스팅 동적 서브셋) /
//   수치 리드아웃은 JetBrains Mono + **tabular-nums 고정**.
//
// ── 접근성 계약 (UX_DESIGN §9) ─────────────────────────────────────
// - 본문/라벨/muted 텍스트 토큰은 **가장 밝은 표면(SURFACE.modal) 위에서도** 대비 ≥ 4.5:1.
//   (검산값은 각 토큰 주석에 기재 — 값을 바꾸면 반드시 다시 계산할 것)
// - 상태는 색만으로 전달하지 않는다 — 항상 라벨/아이콘 텍스트가 병행된다(각 패널 책임).
// - ensureThemeStyles()가 prefers-reduced-motion / forced-colors 대응을 주입한다.
// - 인라인 스타일로 불가능한 것(:hover / :focus-visible / 스크롤바 / 의사요소 / 미디어
//   쿼리)은 전부 주입 스타일시트가 소유한다. 모듈 top-level에서는 DOM을 만지지 않는다
//   — node 단위 테스트가 import해도 안전하다.

// ── 표면 고도 (elevation) ───────────────────────────────────────────
//
// 상대휘도 L과 이전 단계 대비 대비비:
//   sunken  L=0.00338          base  L=0.00702 (×1.068)
//   panel   L=0.01618 (×1.161) raised L=0.02550 (×1.141)
//   overlay L=0.03569 (×1.135) modal  L=0.04823 (×1.146)
//
// Z_INDEX의 각 층이 이 사다리에 1:1로 대응한다(아래 Z_INDEX 주석 참조).
// overlay 이상은 **불투명**이다 — 반투명 + blur 없음 조합이 하위 콘텐츠를 글자 단위로
// 통과시키던 문제(UX_AUDIT C-11)를 닫는다.

export const SURFACE = {
  /** 입력 필드·코드면 — 눌린 표면 */
  sunken: '#090B11',
  /** 앱 바닥 · 뷰포트 뒤 */
  base: '#11141C',
  /** 도킹 패널 · 커맨드바 · 독 */
  panel: '#1D2230',
  /** 버튼 · 카드 · 플로우 노드 — 한 단계 떠 있는 표면 */
  raised: '#262C3D',
  /** 슬라이드 패널 · 팝오버 · 토스트 */
  overlay: '#2E3546',
  /** 다이얼로그 */
  modal: '#363E51',
} as const;

// ── 경계 ────────────────────────────────────────────────────────────
//
// 규약: 경계는 **항상 얹힌 표면보다 밝다**. 구 borderSoft(#22252b)가 bgRaised와 동일
// 헥스라 raised 버튼 위에서 구분선이 사라지던 버그(UX_AUDIT C-11)의 재발을 막는다.

export const BORDER = {
  /** 행 구분 등 옅은 경계 — raised(#262C3D)보다 밝다 */
  subtle: '#2A3140',
  /** 패널 경계 */
  default: '#333B4D',
  /** 버튼·입력 테두리 */
  strong: '#414A5F',
  /** 버튼·입력·노드 hover 테두리 */
  hover: '#57627C',
} as const;

// ── 색상 토큰 ───────────────────────────────────────────────────────
//
// 기존 소비처와의 호환을 위해 키 이름은 유지한다(292곳). 값만 새 시스템으로 교체된다.
// 신규 코드는 SURFACE / BORDER / SELECT / COLLISION / CATEGORY를 직접 쓸 것.

export const COLOR = {
  /** 앱 최하층 배경 = SURFACE.base */
  bgApp: SURFACE.base,
  /** 상단 바/독 = SURFACE.panel (불투명) */
  bgBar: SURFACE.panel,
  /** 도킹 패널 = SURFACE.panel (불투명 — 투과 제거) */
  bgPanel: SURFACE.panel,
  /** 슬라이드 패널·팝오버·토스트 = SURFACE.overlay */
  bgOverlay: SURFACE.overlay,
  /** 다이얼로그 = SURFACE.modal */
  bgModal: SURFACE.modal,
  /** 입력 필드 배경 = SURFACE.sunken */
  bgField: SURFACE.sunken,
  /** 버튼 등 한 단계 떠 있는 표면 = SURFACE.raised */
  bgRaised: SURFACE.raised,
  /** 패널 경계선 */
  border: BORDER.default,
  /** 행 구분 등 옅은 경계선 */
  borderSoft: BORDER.subtle,
  /** 버튼/입력 테두리 */
  borderStrong: BORDER.strong,
  /** 버튼/입력/노드 hover 테두리 */
  borderHover: BORDER.hover,
  /** 본문 텍스트 — modal 위 6.76:1 */
  text: '#C7CEDB',
  /** 제목/값 강조 텍스트 — modal 위 9.44:1 */
  textStrong: '#EEF1F6',
  /** 폼 라벨 — modal 위 4.85:1 */
  label: '#A6AFC2',
  /** 보조/빈 상태 텍스트 — modal 위 4.68:1 (가장 밝은 표면 기준 하한) */
  muted: '#A2ACC0',
  /** muted 계열 배경 틴트 (비활성 배지 등) */
  mutedSoft: 'rgba(162, 172, 192, 0.14)',
  /** 캔버스 배경 그리드 점 */
  gridDot: 'rgba(255, 255, 255, 0.05)',

  /** 액센트 — 일렉트릭 바이올렛. 면(fill)으로 쓸 때만 주요 액션이다 */
  accent: '#7C6AF6',
  /** 어두운 배경 위 액센트 "텍스트"용 밝은 변형 — modal 위 5.00:1 */
  accentText: '#B3A6FF',
  /** 액센트 배경 틴트 (활성 배지·토글) */
  accentSoft: 'rgba(124, 106, 246, 0.16)',
  /** 액센트 면 위에 올리는 어두운 텍스트 — accent 대비 4.86:1 */
  onAccent: '#0B0D13',

  success: '#34D399',
  /** modal 위 7.01:1 */
  successText: '#6EE7B7',
  successSoft: 'rgba(52, 211, 153, 0.14)',
  /** success 계열 테두리 (타임라인 done 마커 등) */
  successBorder: 'rgba(52, 211, 153, 0.42)',

  warn: '#FBBF24',
  /** modal 위 7.41:1 */
  warnText: '#FCD34D',
  warnSoft: 'rgba(251, 191, 36, 0.14)',

  /** 오류/충돌 — COLLISION 램프의 별칭 (신규 코드는 COLLISION을 쓸 것) */
  error: '#FF5D55',
  /** modal 위 4.99:1 */
  errorText: '#FF938C',
  errorSoft: 'rgba(255, 93, 85, 0.16)',
  /** 오류 토스트 등 error 계열 어두운 표면 */
  errorSurface: '#2A1418',
  /** error 계열 테두리 (errorSurface 위) */
  errorBorder: 'rgba(255, 93, 85, 0.45)',

  /** sensor/정보성 강조 */
  info: '#60A5FA',
  /** modal 위 5.93:1 */
  infoText: '#93C5FD',
  infoSoft: 'rgba(96, 165, 250, 0.14)',
} as const;

// ── 선택(selection) ─────────────────────────────────────────────────
//
// **액센트와 분리된 축이다.** 3D 뷰포트에서 "선택된 오브젝트"와 "실행 중"이 같은 색이면
// 구분이 불가능하다. render 계층(interaction/highlight)도 이 값을 쓴다 — 계층 규칙상
// render는 ui/theme를 import할 수 없으므로 숫자 상수를 중복 선언하고 상호참조 주석을 단다.

export const SELECT = {
  /** 선택 하이라이트 기준색 (3D 아웃라인 = 0x38BDF8) */
  base: '#38BDF8',
  /** 어두운 표면 위 선택 텍스트 — modal 위 6.41:1 */
  text: '#7DD3FC',
  /** 선택 행 배경 틴트 */
  soft: 'rgba(56, 189, 248, 0.14)',
  /** 선택 행 좌측 보더 */
  border: 'rgba(56, 189, 248, 0.55)',
} as const;

/** 3D 선택 아웃라인 색 — render/interaction.ts가 같은 값을 숫자로 중복 선언한다 */
export const SELECT_HEX_3D = 0x38bdf8;

// ── 충돌 램프 ───────────────────────────────────────────────────────
//
// 이 제품의 핵심 사건. UI 배지 · 3D 펄스 · 접촉점 마커 · 토스트가 **한 램프에서** 나온다.
// (구: error #ff6b6b / errorText #ff8a80 / errorSoft #e74c3c / 펄스 0xe74c3c /
//  마커 0xff3b30 — 4개 hue로 흩어져 있었다. UX_AUDIT C-7)

export const COLLISION = {
  base: '#FF5D55',
  text: '#FF938C',
  soft: 'rgba(255, 93, 85, 0.16)',
  border: 'rgba(255, 93, 85, 0.45)',
  surface: '#2A1418',
} as const;

/** 3D 충돌 펄스/접촉점 마커 색 — render/highlight.ts·contact-marker.ts가 중복 선언한다 */
export const COLLISION_HEX_3D = 0xff5d55;

// ── 범주형 색 (플로우 노드 종류) ────────────────────────────────────
//
// **시맨틱 토큰(success/warn/info)을 범주형으로 재사용하지 않는다.** 재사용하면 "초록
// 노드"가 성공인지 흐름 제어인지 구분되지 않는다(UX_AUDIT C-14). 흐름은 보라 계열로
// 빼서 완료-초록과 충돌을 피한다.

export const CATEGORY = {
  /** moveJoints / setJoints / gripper — 동작 */
  motion: '#7C6AF6',
  /** wait — 시간 */
  time: '#60A5FA',
  /** waitForCollision — 충돌 */
  collision: '#FBBF24',
  /** label / goto — 흐름 제어 */
  flow: '#C084FC',
} as const;

// ── 타이포그래피 ────────────────────────────────────────────────────

export const FONT = {
  /** UI 텍스트 (라벨·버튼·제목) — Pretendard가 한글+라틴을 통합 메트릭으로 커버 */
  ui: "'Pretendard Variable', Pretendard, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif",
  /** 수치 리드아웃·로그·JSON — 라틴은 JetBrains Mono, 한글은 Pretendard로 폴백 */
  mono: "'JetBrains Mono', ui-monospace, SFMono-Regular, 'Cascadia Mono', Consolas, 'Pretendard Variable', monospace",
} as const;

export interface TypeToken {
  readonly sizePx: number;
  readonly lineHeightPx: number;
  readonly weight: number;
  /** 모노 트랙이면 true — applyType이 FONT.mono + tabular-nums를 건다 */
  readonly mono?: boolean;
  /** 광학 타이트닝/트래킹 */
  readonly letterSpacing?: string;
}

/**
 * 타입 스케일. 행간은 4px 그리드에 정렬된다.
 *
 * 구 시스템은 크기 축이 아예 없어 81곳이 인라인 리터럴이었고, 정보를 담는 최대 활자가
 * 13px이라 **위계 상단이 잘려** 있었다 — 그래서 위계를 색이 대신 감당했고, 액센트 한
 * 색이 8가지 의미를 겸직하게 됐다(UX_AUDIT C-14). title/display가 그 부하를 덜어낸다.
 */
export const TYPE = {
  /** 배지 · 단위 접미사 · 축 라벨(X/Y/Z) */
  micro: { sizePx: 10, lineHeightPx: 14, weight: 500 },
  /** 폼 라벨 · 보조 설명 · 로그 메타 */
  caption: { sizePx: 11, lineHeightPx: 16, weight: 400 },
  /** 기본 본문 — 버튼 · 입력 · 목록 행 */
  body: { sizePx: 12, lineHeightPx: 18, weight: 400 },
  /** 값 강조 · 선택된 행 */
  bodyStrong: { sizePx: 12, lineHeightPx: 18, weight: 600 },
  /** 패널 내부 섹션 제목 (Transform, 관절) */
  subhead: { sizePx: 13, lineHeightPx: 18, weight: 600 },
  /** 패널 헤더 (인스펙터 / 라이브러리 / 엔티티 편집) */
  title: { sizePx: 15, lineHeightPx: 22, weight: 600, letterSpacing: '-0.01em' },
  /** 모달 제목 · 빈 상태 헤드라인 · 브랜드 워드마크 */
  display: { sizePx: 19, lineHeightPx: 26, weight: 650, letterSpacing: '-0.015em' },

  /** 타임라인 칩 · 링크 글리프 */
  monoMicro: { sizePx: 10, lineHeightPx: 14, weight: 400, mono: true },
  /** 콘솔 · 충돌 로그 · JSON */
  monoBody: { sizePx: 11, lineHeightPx: 16, weight: 400, mono: true },
  /** 관절값 · simTime · 좌표 입력 — 자릿수 폭 고정이 필수인 곳 */
  monoReadout: { sizePx: 12, lineHeightPx: 16, weight: 400, mono: true },
} as const satisfies Record<string, TypeToken>;

/** 타입 토큰을 요소에 적용한다. 모노 토큰은 tabular-nums가 함께 걸린다. */
export function applyType<T extends HTMLElement | SVGElement>(el: T, token: TypeToken): T {
  const s = el.style;
  s.fontFamily = token.mono === true ? FONT.mono : FONT.ui;
  s.fontSize = `${token.sizePx}px`;
  s.lineHeight = `${token.lineHeightPx}px`;
  s.fontWeight = String(token.weight);
  if (token.letterSpacing !== undefined) s.letterSpacing = token.letterSpacing;
  if (token.mono === true) s.fontVariantNumeric = 'tabular-nums';
  return el;
}

// ── 간격 / 반경 / 아이콘 ────────────────────────────────────────────
//
// 구 SPACE는 4단계(4/6/8/12)뿐이라 (a) 밀집 컨트롤용 1~3px과 (b) 섹션용 16px 이상이
// 없었고, 개발자가 두 결핍을 raw 리터럴 88곳으로 메웠다(UX_AUDIT C-18).

export const SPACE = {
  /** 보더 보정 · 광학 정렬 nudge */
  hair: '1px',
  /** 아이콘↔텍스트 · 배지 세로 패딩 */
  xxs: '2px',
  /** 인라인 갭 · 밀집 행 세로 패딩 */
  xs: '4px',
  /** 컨트롤 세로 패딩 · 폼 필드 간 */
  sm: '6px',
  /** 컴포넌트 간 갭 · 패널 좌우 패딩 */
  md: '8px',
  /** 섹션 내부 · 패널 헤더 좌우 */
  lg: '12px',
  /** 섹션 경계 */
  xl: '16px',
  /** 모달 내부 · 빈 상태 상하 */
  xxl: '24px',
  /** 빈 상태 여백 */
  xxxl: '32px',
} as const;

export const RADIUS = {
  /** 배지 */
  xs: '3px',
  /** 버튼 · 입력 · 칩 */
  sm: '4px',
  /** 카드 · 플로우 노드 */
  md: '6px',
  /** 패널 · 모달 */
  lg: '10px',
  /** 원형 */
  full: '9999px',
  /** @deprecated RADIUS.sm을 쓸 것 (구 소비처 호환) */
  badge: '3px',
} as const;

export const BORDER_WIDTH = { hair: '1px', thick: '2px' } as const;

/** 아이콘 치수 — icons.ts의 icon(name, size)와 짝 */
export const ICON = { sm: 12, md: 14, lg: 16, xl: 24 } as const;

// ── 그림자 (SURFACE 사다리와 짝) ────────────────────────────────────

export const SHADOW = {
  raised: '0 1px 2px rgba(0, 0, 0, 0.5)',
  panel: '0 4px 16px rgba(0, 0, 0, 0.4)',
  overlay: '0 8px 28px rgba(0, 0, 0, 0.55)',
  modal: '0 24px 64px rgba(0, 0, 0, 0.65)',
  bar: '0 1px 0 rgba(0, 0, 0, 0.4)',
} as const;

// ── 모션 ────────────────────────────────────────────────────────────
//
// 규약: **이탈은 항상 진입보다 빠르다.** 구 시스템은 duration 6종이 인라인 문자열로
// 흩어져 있고 easing이 전부 브라우저 기본 `ease`였으며, 키프레임 2개가 둘 다 무한
// 펄스라 상태 *전이*를 표현하는 모션이 0개였다(UX_AUDIT C-18).

export const MOTION = {
  /** hover/active 색 전환 */
  instant: { ms: 80, easing: 'cubic-bezier(.2,0,0,1)' },
  /** 토글 · 배지 상태 변화 · 포커스 링 */
  fast: { ms: 140, easing: 'cubic-bezier(.2,0,0,1)' },
  /** 패널 접기/펴기 · 슬라이드 패널 진입 */
  base: { ms: 220, easing: 'cubic-bezier(.2,0,0,1)' },
  /** 모달 진입 */
  slow: { ms: 320, easing: 'cubic-bezier(.4,0,.2,1)' },
  /** 모든 이탈 */
  exit: { ms: 120, easing: 'cubic-bezier(.4,0,1,1)' },
  /** 실행 커서 이동 · 노드 활성화 · 충돌 감지 순간 */
  emphasis: { ms: 480, easing: 'cubic-bezier(.34,1.28,.64,1)' },
  /** 실행 중 펄스 (무한 반복) */
  loop: { ms: 1400, easing: 'ease-out' },
} as const;

/** `transition` 속성 문자열을 만든다 — `transition: ${tr('background-color', MOTION.fast)}` */
export function tr(property: string, token: { ms: number; easing: string }): string {
  return `${property} ${token.ms}ms ${token.easing}`;
}

// ── z-index 층 규약 (SURFACE 사다리와 1:1) ──────────────────────────

export const Z_INDEX = {
  /** 커맨드바 · 독 — SURFACE.panel */
  bar: '90',
  dock: '90',
  /** 우측 패널 스택 — SURFACE.panel */
  rightStack: '94',
  /** {} JSON 슬라이드 패널 · 토스트 — SURFACE.overlay */
  slidePanel: '95',
  toast: '95',
  /** 단독 마운트 패널 기본값 — SURFACE.overlay */
  panel: '100',
  /** 다이얼로그 — SURFACE.modal */
  modal: '120',
  /** 오류 오버레이 */
  overlay: '9999',
} as const;

// ── 레이아웃 상수 ───────────────────────────────────────────────────
//
// 고정 px가 창 높이와 무관하게 뷰포트를 압살하던 문제(UX_AUDIT C-1)를 닫기 위해,
// 세로 크롬은 clamp() CSS 문자열로 노출한다. 스플리터가 조절하면 CSS 변수를 덮어쓴다.

export const LAYOUT = {
  /** 상단 커맨드바 높이 — 2행(행당 38px) */
  barHeightPx: 76,
  /** 1행 축약 모드(좁은 화면) 높이 */
  barHeightCompactPx: 40,
  /** 바 아래 요소(토스트·우측 스택)의 top 오프셋 */
  belowBarTopPx: 84,
  /** 독 높이 — 창 높이에 비례하되 상하한 고정 */
  dockHeightCss: 'clamp(112px, 16vh, 280px)',
  /** 독 접힘 높이 (탭바 + 인라인 스트립만) */
  dockCollapsedPx: 30,
  /** 독 본문 기본 높이 (dock.ts 내부 접기 트랜지션 기준값) */
  dockBodyHeightPx: 180,
  /** 플로우 그래프 페인 높이 */
  flowHeightCss: 'clamp(148px, 22vh, 380px)',
  /** 플로우 그래프 스트립 모드 높이 (실행 중 — 활성 노드만) */
  flowStripPx: 56,
  /** 뷰포트 절대 하한 — 이 아래로는 어떤 크롬도 뷰포트를 먹지 못한다 */
  viewportMinHeightPx: 240,
  /** 우측 패널 폭 */
  rightPanelWidthPx: 280,
} as const;

/** 반응형 브레이크포인트 (UX_DESIGN §8 — 구현은 workspace.ts의 ResizeObserver) */
export const BREAKPOINT = {
  /** 이 아래로 좌 패널 자동 접힘 */
  autoCollapseLeftPx: 1100,
  /** 이 아래로 우 패널 자동 접힘 */
  autoCollapseRightPx: 900,
  /** 이 아래로 커맨드바 아이콘 전용 */
  iconOnlyBarPx: 1180,
  /** 이 아래로 커맨드바 1행 + 오버플로 메뉴 */
  compactBarPx: 860,
  /** 이 아래로 독 자동 접힘 (세로) */
  autoCollapseDockPx: 620,
} as const;

// ── 공용 DOM 헬퍼 ───────────────────────────────────────────────────

export function styled<T extends HTMLElement>(el: T, style: Partial<CSSStyleDeclaration>): T {
  Object.assign(el.style, style);
  return el;
}

export type ButtonVariant = 'default' | 'primary' | 'accent' | 'ghost' | 'danger';

/**
 * 공용 버튼 — 시각은 .ui-btn 클래스(hover/active/focus-visible 포함)가 소유한다.
 * 인라인으로 배경/테두리를 덮지 않는다(덮으면 hover가 죽는다).
 *
 * `title`은 접근 가능한 이름 겸 툴팁이다. 단축키가 있으면 반드시 병기할 것
 * (예: '재생/일시정지 (Space)') — 아이콘 전용 모드에서 유일한 이름이 된다.
 */
export function makeButton(
  label: string,
  title: string,
  testId: string,
  variant: ButtonVariant = 'default',
): HTMLButtonElement {
  ensureThemeStyles();
  const button = document.createElement('button');
  button.type = 'button';
  button.className = variant === 'default' ? 'ui-btn' : `ui-btn ui-btn--${variant}`;
  button.textContent = label;
  button.title = title;
  button.setAttribute('aria-label', title);
  button.dataset.testid = testId;
  return button;
}

export interface PanelHeaderOptions {
  /** 접기 버튼을 단다 */
  readonly collapsible?: boolean;
  /** 헤더 우측 액션 영역을 만든다 */
  readonly actions?: boolean;
  /** 제목 요소 태그 — 문서 아웃라인을 위해 h2를 기본으로 한다 */
  readonly headingTag?: 'h2' | 'h3' | 'div';
  readonly testId?: string;
}

export interface PanelHeaderHandle {
  readonly el: HTMLElement;
  readonly titleEl: HTMLElement;
  /** actions:true일 때만 존재 */
  readonly actionsEl: HTMLElement | null;
  /** collapsible:true일 때만 존재 */
  readonly collapseButton: HTMLButtonElement | null;
  setTitle(text: string): void;
  setCollapsed(collapsed: boolean): void;
  onToggle(fn: (collapsed: boolean) => void): void;
}

/**
 * 공용 패널 헤더 팩토리.
 *
 * 구 시스템은 헤더를 8곳에서 각자 수제로 만들어 보더 토큰 3종·제목 굵기 2종
 * (`<strong>` 700 vs span 600)·셰브론 규약 2종(▾/▴ vs ▾/▸)이 **한 화면에 동시 노출**
 * 되고 있었다(UX_AUDIT C-18). 접기는 글리프 교체가 아니라 **단일 셰브론의 회전**으로
 * 통일한다 — 규약 논쟁이 사라지고 상태 전이에 모션이 붙는다.
 */
export function makePanelHeader(
  titleText: string,
  opts: PanelHeaderOptions = {},
): PanelHeaderHandle {
  ensureThemeStyles();
  const el = document.createElement('div');
  el.className = 'ui-panel-header';
  if (opts.testId !== undefined) el.dataset.testid = opts.testId;

  const titleEl = document.createElement(opts.headingTag ?? 'h2');
  titleEl.className = 'ui-panel-header__title';
  titleEl.textContent = titleText;
  el.appendChild(titleEl);

  let actionsEl: HTMLElement | null = null;
  if (opts.actions === true) {
    actionsEl = document.createElement('div');
    actionsEl.className = 'ui-panel-header__actions';
    el.appendChild(actionsEl);
  }

  let collapseButton: HTMLButtonElement | null = null;
  let collapsed = false;
  const listeners: ((c: boolean) => void)[] = [];

  const paint = (): void => {
    if (collapseButton === null) return;
    collapseButton.classList.toggle('ui-panel-header__chevron--collapsed', collapsed);
    collapseButton.setAttribute('aria-expanded', String(!collapsed));
    const t = collapsed ? `${titleText} 펼치기` : `${titleText} 접기`;
    collapseButton.title = t;
    collapseButton.setAttribute('aria-label', t);
  };

  if (opts.collapsible === true) {
    collapseButton = document.createElement('button');
    collapseButton.type = 'button';
    collapseButton.className = 'ui-btn ui-btn--ghost ui-panel-header__chevron';
    collapseButton.dataset.testid =
      opts.testId !== undefined ? `${opts.testId}-collapse` : 'panel-collapse';
    // 셰브론 1개를 회전시킨다 (▾/▴ vs ▾/▸ 규약 분열 제거)
    collapseButton.innerHTML =
      '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" focusable="false">' +
      '<path d="M4 6.5 8 10.5 12 6.5" fill="none" stroke="currentColor" stroke-width="1.5" ' +
      'stroke-linecap="round" stroke-linejoin="round"/></svg>';
    collapseButton.addEventListener('click', () => {
      collapsed = !collapsed;
      paint();
      for (const fn of listeners) fn(collapsed);
    });
    el.appendChild(collapseButton);
    paint();
  }

  return {
    el,
    titleEl,
    actionsEl,
    collapseButton,
    setTitle: (text: string): void => {
      titleEl.textContent = text;
    },
    setCollapsed: (c: boolean): void => {
      if (c === collapsed) return;
      collapsed = c;
      paint();
      for (const fn of listeners) fn(collapsed);
    },
    onToggle: (fn: (c: boolean) => void): void => {
      listeners.push(fn);
    },
  };
}

// ── 공용 스타일시트 주입 (1회) ──────────────────────────────────────

const THEME_STYLE_ID = 'rsw-theme-styles';

/** 공용 스타일시트를 <head>에 1회 주입한다. 모든 마운트 함수가 먼저 호출한다(멱등). */
export function ensureThemeStyles(): void {
  if (document.getElementById(THEME_STYLE_ID) !== null) return;
  const style = document.createElement('style');
  style.id = THEME_STYLE_ID;
  style.textContent = `
:root {
  --rsw-surface-sunken: ${SURFACE.sunken};
  --rsw-surface-base: ${SURFACE.base};
  --rsw-surface-panel: ${SURFACE.panel};
  --rsw-surface-raised: ${SURFACE.raised};
  --rsw-surface-overlay: ${SURFACE.overlay};
  --rsw-surface-modal: ${SURFACE.modal};
  --rsw-border-subtle: ${BORDER.subtle};
  --rsw-border: ${BORDER.default};
  --rsw-border-strong: ${BORDER.strong};
  --rsw-border-hover: ${BORDER.hover};
  --rsw-text: ${COLOR.text};
  --rsw-text-strong: ${COLOR.textStrong};
  --rsw-muted: ${COLOR.muted};
  --rsw-accent: ${COLOR.accent};
  --rsw-accent-text: ${COLOR.accentText};
  --rsw-accent-soft: ${COLOR.accentSoft};
  --rsw-on-accent: ${COLOR.onAccent};
  --rsw-select: ${SELECT.base};
  --rsw-select-soft: ${SELECT.soft};
  --rsw-select-border: ${SELECT.border};
  --rsw-collision: ${COLLISION.base};
  --rsw-font-ui: ${FONT.ui};
  --rsw-font-mono: ${FONT.mono};
  /* 레이아웃 — workspace.ts의 스플리터가 덮어쓴다 */
  --rsw-dock-h: ${LAYOUT.dockHeightCss};
  --rsw-flow-h: ${LAYOUT.flowHeightCss};
}

/* ── 스크린리더 전용 (랜드마크 제목·스킵 링크) ────────────────────── */
.sr-only {
  position: absolute;
  width: 1px; height: 1px;
  padding: 0; margin: -1px;
  overflow: hidden;
  clip: rect(0 0 0 0);
  white-space: nowrap;
  border: 0;
}
.rsw-skip-link {
  position: fixed;
  top: ${SPACE.md}; left: ${SPACE.md};
  z-index: 10000;
  transform: translateY(-200%);
  background: var(--rsw-accent);
  color: var(--rsw-on-accent);
  font: ${TYPE.bodyStrong.weight} ${TYPE.bodyStrong.sizePx}px/${TYPE.bodyStrong.lineHeightPx}px var(--rsw-font-ui);
  padding: ${SPACE.md} ${SPACE.lg};
  border-radius: ${RADIUS.sm};
  text-decoration: none;
  transition: ${tr('transform', MOTION.fast)};
}
.rsw-skip-link:focus { transform: translateY(0); }

/* ── 버튼 ─────────────────────────────────────────────────────────── */
.ui-btn {
  display: inline-flex;
  align-items: center;
  gap: ${SPACE.xs};
  background: ${SURFACE.raised};
  color: ${COLOR.text};
  border: ${BORDER_WIDTH.hair} solid ${BORDER.strong};
  border-radius: ${RADIUS.sm};
  padding: ${SPACE.xs} ${SPACE.md};
  min-height: 24px;
  font-family: var(--rsw-font-ui);
  font-size: ${TYPE.body.sizePx}px;
  line-height: ${TYPE.body.lineHeightPx}px;
  font-weight: ${TYPE.body.weight};
  cursor: pointer;
  white-space: nowrap;
  transition: ${tr('background-color', MOTION.instant)}, ${tr('border-color', MOTION.instant)}, ${tr('color', MOTION.instant)};
}
.ui-btn > svg { flex: none; }
.ui-btn:hover:not(:disabled) {
  background: ${SURFACE.overlay};
  border-color: ${BORDER.hover};
  color: ${COLOR.textStrong};
}
.ui-btn:active:not(:disabled) { background: ${SURFACE.panel}; }
.ui-btn:disabled { opacity: 0.42; cursor: default; }

/* 주요 액션 — 1화면 1개. 액센트를 **면**으로 쓰는 유일한 곳이다 */
.ui-btn--primary {
  background: ${COLOR.accent};
  color: ${COLOR.onAccent};
  border-color: ${COLOR.accent};
  font-weight: ${TYPE.bodyStrong.weight};
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.14), ${SHADOW.raised};
}
.ui-btn--primary:hover:not(:disabled) {
  background: #8B7BFF;
  border-color: #8B7BFF;
  color: ${COLOR.onAccent};
}
.ui-btn--primary:active:not(:disabled) { background: #6A57E8; border-color: #6A57E8; }

/* 활성/토글 상태 — 보더 + 텍스트만 (면은 primary의 몫) */
.ui-btn--accent { border-color: rgba(124, 106, 246, 0.55); color: ${COLOR.accentText}; }
.ui-btn--accent:hover:not(:disabled) {
  background: var(--rsw-accent-soft);
  border-color: var(--rsw-accent);
  color: #C9BFFF;
}

.ui-btn--ghost { background: transparent; border-color: transparent; color: ${COLOR.muted}; }
.ui-btn--ghost:hover:not(:disabled) {
  background: rgba(255, 255, 255, 0.06);
  border-color: transparent;
  color: ${COLOR.textStrong};
}

.ui-btn--danger { border-color: ${COLLISION.border}; color: ${COLLISION.text}; }
.ui-btn--danger:hover:not(:disabled) {
  background: ${COLLISION.soft};
  border-color: ${COLLISION.base};
  color: #FFB0AA;
}

/* 토글형 버튼의 "켜짐" 상태 ({} JSON 등) — aria-pressed와 짝으로 쓴다 */
.ui-btn--active { border-color: var(--rsw-accent); color: var(--rsw-accent-text); background: var(--rsw-accent-soft); }

/* 아이콘 전용 모드 — 라벨 텍스트를 숨기되 접근 가능한 이름(aria-label)은 유지한다 */
.ui-btn--icon-only { padding-left: ${SPACE.sm}; padding-right: ${SPACE.sm}; }
.ui-btn--icon-only .ui-btn__label { display: none; }

/* ── 셀렉트 / 입력 ────────────────────────────────────────────────── */
.ui-select, .ui-input {
  background: ${SURFACE.raised};
  color: ${COLOR.text};
  border: ${BORDER_WIDTH.hair} solid ${BORDER.strong};
  border-radius: ${RADIUS.sm};
  padding: ${SPACE.xs} ${SPACE.sm};
  min-height: 24px;
  font-family: var(--rsw-font-ui);
  font-size: ${TYPE.body.sizePx}px;
  line-height: ${TYPE.body.lineHeightPx}px;
  transition: ${tr('border-color', MOTION.instant)}, ${tr('color', MOTION.instant)};
}
.ui-input { background: ${SURFACE.sunken}; box-shadow: inset 0 1px 0 rgba(0, 0, 0, 0.45); }
.ui-input--mono { font-family: var(--rsw-font-mono); font-variant-numeric: tabular-nums; }
.ui-select:hover:not(:disabled), .ui-input:hover:not(:disabled) { border-color: ${BORDER.hover}; }
.ui-select:disabled, .ui-input:disabled { opacity: 0.42; }

/* ── 포커스 링 (키보드 조작 가시성 — UX_DESIGN §9) ─────────────────── */
.ui-btn:focus-visible, .ui-select:focus-visible, .ui-input:focus-visible,
.ui-tab:focus-visible, .ui-check:focus-visible, .ui-range:focus-visible,
[role="option"]:focus-visible, [role="separator"]:focus-visible,
.rsw-fg-node:focus-visible, .rsw-fg-plus:focus-visible,
input[type="range"]:focus-visible, input[type="checkbox"]:focus-visible {
  outline: 2px solid var(--rsw-accent);
  outline-offset: 1px;
}

/* ── 패널 헤더 (makePanelHeader 전용 — 수제 구현 8곳을 대체) ───────── */
.ui-panel-header {
  display: flex;
  align-items: center;
  gap: ${SPACE.md};
  padding: ${SPACE.md} ${SPACE.lg};
  border-bottom: ${BORDER_WIDTH.hair} solid ${BORDER.subtle};
  flex: none;
}
.ui-panel-header__title {
  flex: 1 1 auto;
  min-width: 0;
  margin: 0;
  color: ${COLOR.textStrong};
  font-family: var(--rsw-font-ui);
  font-size: ${TYPE.title.sizePx}px;
  line-height: ${TYPE.title.lineHeightPx}px;
  font-weight: ${TYPE.title.weight};
  letter-spacing: ${TYPE.title.letterSpacing};
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.ui-panel-header__actions { display: flex; align-items: center; gap: ${SPACE.xs}; flex: none; }
.ui-panel-header__chevron {
  padding: ${SPACE.xxs};
  min-width: 24px;
  min-height: 24px;
  justify-content: center;
}
.ui-panel-header__chevron svg { transition: ${tr('transform', MOTION.fast)}; }
.ui-panel-header__chevron--collapsed svg { transform: rotate(-90deg); }

/* ── 독 탭 ────────────────────────────────────────────────────────── */
.ui-tab {
  position: relative;
  background: transparent;
  color: ${COLOR.muted};
  border: none;
  border-bottom: ${BORDER_WIDTH.thick} solid transparent;
  padding: ${SPACE.sm} ${SPACE.lg};
  min-height: 30px;
  font-family: var(--rsw-font-ui);
  font-size: ${TYPE.body.sizePx}px;
  font-weight: ${TYPE.body.weight};
  cursor: pointer;
  transition: ${tr('color', MOTION.instant)}, ${tr('border-color', MOTION.instant)};
}
.ui-tab:hover { color: ${COLOR.textStrong}; }
.ui-tab--active { color: ${COLOR.textStrong}; border-bottom-color: var(--rsw-accent); font-weight: ${TYPE.bodyStrong.weight}; }

/* 탭 카운트 배지 — 비활성 탭에 새 이벤트가 쌓였을 때 (UX_AUDIT C-7) */
.ui-tab__badge {
  display: inline-block;
  margin-left: ${SPACE.sm};
  padding: 0 ${SPACE.sm};
  border-radius: ${RADIUS.full};
  background: ${COLLISION.soft};
  color: ${COLLISION.text};
  border: ${BORDER_WIDTH.hair} solid ${COLLISION.border};
  font-family: var(--rsw-font-mono);
  font-size: ${TYPE.micro.sizePx}px;
  line-height: 15px;
  font-variant-numeric: tabular-nums;
  animation: rsw-badge-in ${MOTION.fast.ms}ms ${MOTION.fast.easing};
}
@keyframes rsw-badge-in {
  from { transform: scale(0.6); opacity: 0; }
  to   { transform: scale(1);   opacity: 1; }
}

/* ── 패널 스크롤바 ────────────────────────────────────────────────── */
.ui-scroll { scrollbar-width: thin; scrollbar-color: ${BORDER.strong} transparent; }
.ui-scroll::-webkit-scrollbar { width: 10px; height: 10px; }
.ui-scroll::-webkit-scrollbar-track { background: transparent; }
.ui-scroll::-webkit-scrollbar-thumb {
  background: ${BORDER.strong};
  border-radius: ${RADIUS.full};
  border: 3px solid transparent;
  background-clip: padding-box;
}
.ui-scroll::-webkit-scrollbar-thumb:hover { background: ${BORDER.hover}; background-clip: padding-box; }

/* ── 상태 점 (재생 상태 — 색 + 라벨 텍스트 병행) ───────────────────── */
.ui-dot {
  display: inline-block;
  width: 7px;
  height: 7px;
  border-radius: ${RADIUS.full};
  margin-right: ${SPACE.sm};
  vertical-align: middle;
  background: ${COLOR.muted};
}
.ui-dot--playing { background: var(--rsw-accent); animation: rsw-pulse ${MOTION.loop.ms}ms ${MOTION.loop.easing} infinite; }
.ui-dot--paused { background: ${COLOR.warn}; }
.ui-dot--idle { background: ${COLOR.muted}; }
@keyframes rsw-pulse {
  0%   { box-shadow: 0 0 0 0 rgba(124, 106, 246, 0.55); }
  70%  { box-shadow: 0 0 0 6px rgba(124, 106, 246, 0); }
  100% { box-shadow: 0 0 0 0 rgba(124, 106, 246, 0); }
}

/* 전역 실행 상태 — 화면 어디를 보고 있어도 주변시로 잡힌다 (UX_AUDIT C-9) */
body[data-run-state="running"] .rsw-run-strip {
  transform: scaleX(1);
  opacity: 1;
}
.rsw-run-strip {
  position: absolute;
  left: 0; right: 0; bottom: -1px;
  height: 2px;
  background: linear-gradient(90deg, transparent, var(--rsw-accent) 35%, var(--rsw-select) 65%, transparent);
  transform: scaleX(0);
  opacity: 0;
  transform-origin: left center;
  pointer-events: none;
  transition: ${tr('transform', MOTION.base)}, ${tr('opacity', MOTION.base)};
}

/* ── 배지 (충돌 로그 phase/kind — 텍스트가 의미를 전달, 색은 보조) ── */
.ui-badge {
  display: inline-block;
  padding: 0 ${SPACE.sm};
  border-radius: ${RADIUS.xs};
  font-family: var(--rsw-font-ui);
  font-size: ${TYPE.micro.sizePx}px;
  font-weight: ${TYPE.micro.weight};
  line-height: 16px;
  border: ${BORDER_WIDTH.hair} solid transparent;
  white-space: nowrap;
}
.ui-badge--start { background: ${COLLISION.soft}; color: ${COLLISION.text}; border-color: ${COLLISION.border}; }
.ui-badge--stop { background: ${COLOR.mutedSoft}; color: ${COLOR.muted}; border-color: rgba(162, 172, 192, 0.3); }
.ui-badge--sensor { background: ${COLOR.infoSoft}; color: ${COLOR.infoText}; border-color: rgba(96, 165, 250, 0.4); }
.ui-badge--accent { background: ${COLOR.accentSoft}; color: ${COLOR.accentText}; border-color: rgba(124, 106, 246, 0.4); }

/* ── 행 hover / 선택 (목록·로그) ──────────────────────────────────── */
/* 선택은 SELECT(스카이블루) — 액센트와 분리된 축이다 */
.rsw-hover-row { transition: ${tr('background-color', MOTION.instant)}; }
.rsw-hover-row:hover { background: rgba(255, 255, 255, 0.045); }
.rsw-entity-row {
  border-left: ${BORDER_WIDTH.thick} solid transparent;
  border-radius: ${RADIUS.sm};
  cursor: pointer;
  transition: ${tr('background-color', MOTION.instant)}, ${tr('border-color', MOTION.instant)};
}
.rsw-entity-row:hover { background: rgba(255, 255, 255, 0.045); }
.rsw-entity-row--selected { background: var(--rsw-select-soft); border-left-color: var(--rsw-select); }
.rsw-entity-row--selected:hover { background: var(--rsw-select-soft); }

/* ── 체크박스 (네이티브 렌더 제거 — 디자인 시스템 편입) ───────────── */
.ui-check {
  appearance: none;
  -webkit-appearance: none;
  flex: none;
  width: 16px; height: 16px;
  margin: 0;
  border: ${BORDER_WIDTH.hair} solid ${BORDER.strong};
  border-radius: ${RADIUS.xs};
  background: ${SURFACE.sunken};
  cursor: pointer;
  display: inline-grid;
  place-content: center;
  transition: ${tr('background-color', MOTION.instant)}, ${tr('border-color', MOTION.instant)};
}
.ui-check::before {
  content: '';
  width: 10px; height: 10px;
  transform: scale(0);
  transform-origin: center;
  background: ${COLOR.onAccent};
  clip-path: polygon(14% 47%, 0 61%, 39% 100%, 100% 21%, 86% 7%, 39% 72%);
  transition: ${tr('transform', MOTION.fast)};
}
.ui-check:hover:not(:disabled) { border-color: ${BORDER.hover}; }
.ui-check:checked { background: var(--rsw-accent); border-color: var(--rsw-accent); }
.ui-check:checked::before { transform: scale(1); }
.ui-check:disabled { opacity: 0.42; cursor: default; }
/* 체크박스를 감싼 라벨 전체가 히트 영역이 되도록 */
.ui-check-label {
  display: inline-flex;
  align-items: center;
  gap: ${SPACE.sm};
  min-height: 24px;
  cursor: pointer;
  color: ${COLOR.muted};
  font-size: ${TYPE.caption.sizePx}px;
  line-height: ${TYPE.caption.lineHeightPx}px;
  white-space: nowrap;
}
.ui-check-label:hover { color: ${COLOR.text}; }

/* ── 슬라이더 ─────────────────────────────────────────────────────── */
.ui-range {
  appearance: none;
  -webkit-appearance: none;
  width: 100%;
  height: 20px;
  margin: 0;
  background: transparent;
  cursor: pointer;
}
.ui-range::-webkit-slider-runnable-track {
  height: 4px;
  border-radius: ${RADIUS.full};
  background: var(--rsw-range-track, ${SURFACE.sunken});
  box-shadow: inset 0 1px 0 rgba(0, 0, 0, 0.5);
}
.ui-range::-moz-range-track {
  height: 4px;
  border-radius: ${RADIUS.full};
  background: var(--rsw-range-track, ${SURFACE.sunken});
}
.ui-range::-webkit-slider-thumb {
  appearance: none;
  -webkit-appearance: none;
  width: 12px; height: 12px;
  margin-top: -4px;
  border-radius: ${RADIUS.full};
  background: ${COLOR.textStrong};
  border: ${BORDER_WIDTH.hair} solid ${BORDER.hover};
  box-shadow: ${SHADOW.raised};
  transition: ${tr('background-color', MOTION.instant)}, ${tr('transform', MOTION.instant)};
}
.ui-range::-moz-range-thumb {
  width: 12px; height: 12px;
  border-radius: ${RADIUS.full};
  background: ${COLOR.textStrong};
  border: ${BORDER_WIDTH.hair} solid ${BORDER.hover};
}
.ui-range:hover::-webkit-slider-thumb { background: var(--rsw-accent-text); transform: scale(1.12); }
.ui-range:disabled { opacity: 0.42; cursor: default; }
/* 부호 있는 값(관절 limits가 0을 걸칠 때) — 채움이 0에서 썸까지 그려진다.
   구현: joint-panel/node-editor가 --rsw-range-track에 gradient를 써 넣는다.
   (min→thumb 채움은 -0.6rad를 "35% 진행"처럼 읽히게 만든다 — UX_AUDIT C-18) */
.ui-range--bipolar::-webkit-slider-runnable-track { background: var(--rsw-range-track); }
.ui-range--bipolar::-moz-range-track { background: var(--rsw-range-track); }

/* ── 접기 트랜지션 (max-height/height는 각 패널이 인라인으로 구동) ── */
.ui-collapsible { transition: ${tr('max-height', MOTION.base)}, ${tr('opacity', MOTION.fast)}, ${tr('padding', MOTION.base)}; }
.ui-collapsible-h { transition: ${tr('height', MOTION.base)}; }

/* ── 히트 영역 확장 (WCAG 2.2 SC 2.5.8 — 시각 크기는 유지) ────────── */
/* 스플리터 5px, flow-plus 15px 같이 시각적으로 얇아야 하는 표적의 클릭 영역만 넓힌다.
   Blender/Figma가 쓰는 방식이다. */
.rsw-hit-x { position: relative; }
.rsw-hit-x::before { content: ''; position: absolute; inset: 0 -5px; }
.rsw-hit-y { position: relative; }
.rsw-hit-y::before { content: ''; position: absolute; inset: -5px 0; }

/* ── 모션 민감성 (WCAG 2.3.3 / OS 설정 존중) ───────────────────────── */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
    scroll-behavior: auto !important;
  }
}

/* ── Windows 고대비 모드 ───────────────────────────────────────────── */
/* SVG의 fill/stroke 프레젠테이션 속성은 강제 치환 대상이 아니므로 명시 대응이 필요하다
   — 대응이 없으면 주변 UI만 흑백이 되고 플로우 그래프만 원래 색이 남는다. */
@media (forced-colors: active) {
  .ui-btn, .ui-input, .ui-select, .ui-check, .ui-panel-header {
    border: ${BORDER_WIDTH.hair} solid ButtonBorder;
  }
  .ui-btn:focus-visible, .ui-select:focus-visible, .ui-input:focus-visible {
    outline: 2px solid Highlight;
  }
  .ui-tab--active { border-bottom-color: Highlight; }
  .rsw-entity-row--selected { background: Highlight; color: HighlightText; }
  .rsw-fg-node-body { fill: Canvas; stroke: CanvasText; }
  .rsw-fg-node text { fill: CanvasText !important; }
  .rsw-fg-edge { stroke: CanvasText !important; }
  .ui-dot { forced-color-adjust: none; }
  .ui-badge { border-color: CanvasText; }
}
`;
  document.head.appendChild(style);
}
