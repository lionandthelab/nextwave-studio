// ui/theme.ts — UI 디자인 토큰 + 공용 스타일시트 (UX_DESIGN §1-2 원칙, frontend-design)
//
// 모든 패널(command-bar / dock / inspector / joint-panel / viewport 오버레이)이 이 모듈의
// 토큰만 소비한다 — 모듈별 ad-hoc 색상 하드코딩 금지. 시각 언어는 한 곳에서 결정한다.
//
// - 다크 테마: #14161a 계열 배경 레이어 + rgba 글래스 패널
// - 액센트: 주황 #e67e22 (로봇 팔 색과 일관 — 실행/활성/선택 강조)
// - 시맨틱: success / warn / error / info(sensor)
// - 폰트: UI 텍스트는 system-ui 스택, 수치 리드아웃·로그는 ui-monospace 스택
// - 상호작용(:hover / :active / :focus-visible / 스크롤바 / 펄스 애니메이션)은
//   인라인 스타일로 불가능하므로 ensureThemeStyles()가 <style> 1장을 주입한다.
//   (모듈 top-level에서는 DOM을 만지지 않는다 — node 단위 테스트가 import해도 안전)
//
// 접근성 (UX_DESIGN §9): 본문 텍스트 토큰은 패널 배경 대비 ≥ 4.5:1.
// 상태는 색만으로 전달하지 않는다 — 항상 라벨/아이콘 텍스트가 병행된다(각 패널 책임).

// ── 색상 토큰 ───────────────────────────────────────────────────────

export const COLOR = {
  /** 앱 최하층 배경 */
  bgApp: '#14161a',
  /** 상단 바/독 — 반투명 글래스 (blur는 GPU 없는 환경 배려로 미사용) */
  bgBar: 'rgba(17, 19, 24, 0.94)',
  /** 떠 있는 패널(인스펙터·관절·JSON 뷰어) 글래스 */
  bgPanel: 'rgba(19, 22, 27, 0.93)',
  /** 입력 필드 배경 */
  bgField: '#101216',
  /** 버튼 등 한 단계 떠 있는 표면 */
  bgRaised: '#22252b',
  /** 패널 경계선 */
  border: '#2e3238',
  /** 행 구분 등 옅은 경계선 */
  borderSoft: '#22252b',
  /** 버튼/입력 테두리 */
  borderStrong: '#3a3f47',
  /** 본문 텍스트 (≥ 4.5:1) */
  text: '#cfd4db',
  /** 제목/값 강조 텍스트 */
  textStrong: '#eceef1',
  /** 폼 라벨 */
  label: '#9aa1ab',
  /** 보조/빈 상태 텍스트 — 4.5:1을 지키는 최저 명도 */
  muted: '#8b93a1',
  /** 액센트 (주황 — 로봇 팔 색과 일관) */
  accent: '#e67e22',
  /** 어두운 배경 위 액센트 "텍스트"용 밝은 변형 */
  accentText: '#f0a860',
  /** 액센트 배경 틴트 (선택 행·활성 배지) */
  accentSoft: 'rgba(230, 126, 34, 0.13)',
  /** 액센트 위에 올리는 어두운 텍스트 (accent 배경 대비 ≥ 4.5:1) */
  onAccent: '#14161a',
  success: '#3dbb72',
  successText: '#7fd6a2',
  successSoft: 'rgba(61, 187, 114, 0.14)',
  /** success 계열 테두리 (타임라인 done 마커 등) */
  successBorder: 'rgba(61, 187, 114, 0.4)',
  warn: '#e2b93d',
  error: '#ff6b6b',
  errorText: '#ff8a80',
  errorSoft: 'rgba(231, 76, 60, 0.16)',
  /** 오류 토스트 등 error 계열 어두운 표면 */
  errorSurface: 'rgba(42, 18, 20, 0.97)',
  /** error 계열 테두리 (errorSurface 위) */
  errorBorder: '#a33a3a',
  /** sensor/정보성 강조 (충돌 로그 sensor 배지 등) */
  info: '#7fb3ef',
  infoSoft: 'rgba(90, 157, 232, 0.14)',
} as const;

// ── 타이포그래피 ────────────────────────────────────────────────────

export const FONT = {
  /** UI 텍스트 (라벨·버튼·제목) */
  ui: "system-ui, -apple-system, 'Segoe UI', Roboto, 'Noto Sans KR', sans-serif",
  /** 수치 리드아웃·로그·JSON (자릿수 정렬) */
  mono: "ui-monospace, SFMono-Regular, 'Cascadia Mono', Consolas, monospace",
} as const;

// ── 간격 / 반경 / 그림자 ────────────────────────────────────────────

export const SPACE = { xs: '4px', sm: '6px', md: '8px', lg: '12px' } as const;

export const RADIUS = {
  /** 버튼·입력·마커 */
  sm: '5px',
  /** 패널 */
  md: '8px',
  /** 배지 */
  badge: '3px',
} as const;

export const SHADOW = {
  panel: '0 10px 26px rgba(0, 0, 0, 0.38)',
  bar: '0 1px 8px rgba(0, 0, 0, 0.35)',
} as const;

// ── z-index 층 규약 (기존 레이어링 유지 — 숫자만 중앙화) ─────────────

export const Z_INDEX = {
  /** 커맨드바 · 독 — 같은 층 */
  bar: '90',
  dock: '90',
  /** 우측 패널 스택 (독/바보다 위, 슬라이드 패널보다 아래) */
  rightStack: '94',
  /** {} JSON 슬라이드 패널 · 오류 토스트 */
  slidePanel: '95',
  toast: '95',
  /** 단독 마운트 패널 기본값 */
  panel: '100',
  /** 오류 오버레이 */
  overlay: '9999',
} as const;

// ── 레이아웃 상수 ───────────────────────────────────────────────────

export const LAYOUT = {
  /** 상단 커맨드바 높이 */
  barHeightPx: 44,
  /** 바 아래 요소(토스트·우측 스택)의 top 오프셋 */
  belowBarTopPx: 52,
  /** 독 본문 높이 */
  dockBodyHeightPx: 180,
  /** 우측 패널 폭 */
  rightPanelWidthPx: 280,
} as const;

// ── 공용 DOM 헬퍼 ───────────────────────────────────────────────────

export function styled<T extends HTMLElement>(el: T, style: Partial<CSSStyleDeclaration>): T {
  Object.assign(el.style, style);
  return el;
}

export type ButtonVariant = 'default' | 'accent' | 'ghost';

/**
 * 공용 버튼 — 시각은 .ui-btn 클래스(hover/active/focus-visible 포함)가 소유한다.
 * 인라인으로 배경/테두리를 덮지 않는다(덮으면 hover가 죽는다).
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
  button.className =
    variant === 'default' ? 'ui-btn' : `ui-btn ui-btn--${variant}`;
  button.textContent = label;
  button.title = title;
  button.setAttribute('aria-label', title);
  button.dataset.testid = testId;
  return button;
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
  --rsw-accent: ${COLOR.accent};
  --rsw-accent-text: ${COLOR.accentText};
  --rsw-accent-soft: ${COLOR.accentSoft};
}

/* ── 버튼 ─────────────────────────────────────────────────────────── */
.ui-btn {
  background: ${COLOR.bgRaised};
  color: ${COLOR.text};
  border: 1px solid ${COLOR.borderStrong};
  border-radius: ${RADIUS.sm};
  padding: 3px 10px;
  font-family: inherit;
  font-size: 12px;
  line-height: 1.5;
  cursor: pointer;
  white-space: nowrap;
  transition: background-color 0.12s ease, border-color 0.12s ease, color 0.12s ease;
}
.ui-btn:hover:not(:disabled) {
  background: #2b2f37;
  border-color: #4a5058;
  color: ${COLOR.textStrong};
}
.ui-btn:active:not(:disabled) { background: #1d2026; }
.ui-btn:disabled { opacity: 0.45; cursor: default; }

.ui-btn--accent { border-color: rgba(230, 126, 34, 0.55); color: ${COLOR.accentText}; }
.ui-btn--accent:hover:not(:disabled) {
  background: var(--rsw-accent-soft);
  border-color: var(--rsw-accent);
  color: #ffb877;
}

.ui-btn--ghost { background: transparent; border-color: transparent; color: ${COLOR.muted}; }
.ui-btn--ghost:hover:not(:disabled) {
  background: rgba(255, 255, 255, 0.06);
  border-color: transparent;
  color: ${COLOR.textStrong};
}

/* 토글형 버튼의 "켜짐" 상태 ({} JSON 등) — aria-pressed와 짝으로 쓴다 */
.ui-btn--active { border-color: var(--rsw-accent); color: var(--rsw-accent-text); }

/* ── 셀렉트 / 입력 ────────────────────────────────────────────────── */
.ui-select, .ui-input {
  background: ${COLOR.bgRaised};
  color: ${COLOR.text};
  border: 1px solid ${COLOR.borderStrong};
  border-radius: ${RADIUS.sm};
  padding: 3px 6px;
  font-family: inherit;
  font-size: 12px;
  transition: border-color 0.12s ease, color 0.12s ease;
}
.ui-input { background: ${COLOR.bgField}; }
.ui-select:hover:not(:disabled), .ui-input:hover:not(:disabled) { border-color: #4a5058; }
.ui-select:disabled { opacity: 0.45; }

/* ── 포커스 링 (키보드 조작 가시성 — UX_DESIGN §9) ─────────────────── */
.ui-btn:focus-visible, .ui-select:focus-visible, .ui-input:focus-visible,
.ui-tab:focus-visible, input[type="range"]:focus-visible {
  outline: 2px solid var(--rsw-accent);
  outline-offset: 1px;
}

/* ── 독 탭 ────────────────────────────────────────────────────────── */
.ui-tab {
  background: transparent;
  color: ${COLOR.muted};
  border: none;
  border-bottom: 2px solid transparent;
  padding: 5px 10px;
  font-family: inherit;
  font-size: 12px;
  cursor: pointer;
  transition: color 0.12s ease, border-color 0.12s ease;
}
.ui-tab:hover { color: ${COLOR.textStrong}; }
.ui-tab--active { color: ${COLOR.textStrong}; border-bottom-color: var(--rsw-accent); }

/* ── 패널 스크롤바 ────────────────────────────────────────────────── */
.ui-scroll { scrollbar-width: thin; scrollbar-color: #3a3f47 transparent; }
.ui-scroll::-webkit-scrollbar { width: 8px; height: 8px; }
.ui-scroll::-webkit-scrollbar-track { background: transparent; }
.ui-scroll::-webkit-scrollbar-thumb {
  background: #343941;
  border-radius: 4px;
  border: 2px solid transparent;
  background-clip: padding-box;
}
.ui-scroll::-webkit-scrollbar-thumb:hover { background: #454b55; background-clip: padding-box; }

/* ── 상태 점 (재생 상태 — 색 + 라벨 텍스트 병행) ───────────────────── */
.ui-dot {
  display: inline-block;
  width: 7px;
  height: 7px;
  border-radius: 50%;
  margin-right: 6px;
  vertical-align: middle;
  background: ${COLOR.muted};
}
.ui-dot--playing { background: var(--rsw-accent); animation: rsw-pulse 1.4s ease-out infinite; }
.ui-dot--paused { background: ${COLOR.warn}; }
.ui-dot--idle { background: ${COLOR.muted}; }
@keyframes rsw-pulse {
  0% { box-shadow: 0 0 0 0 rgba(230, 126, 34, 0.55); }
  70% { box-shadow: 0 0 0 6px rgba(230, 126, 34, 0); }
  100% { box-shadow: 0 0 0 0 rgba(230, 126, 34, 0); }
}

/* ── 배지 (충돌 로그 phase/kind — 텍스트가 의미를 전달, 색은 보조) ── */
.ui-badge {
  display: inline-block;
  padding: 0 6px;
  border-radius: ${RADIUS.badge};
  font-size: 10px;
  line-height: 16px;
  border: 1px solid transparent;
  white-space: nowrap;
}
.ui-badge--start { background: ${COLOR.errorSoft}; color: ${COLOR.errorText}; border-color: rgba(231, 76, 60, 0.4); }
.ui-badge--stop { background: rgba(139, 147, 161, 0.12); color: #aab2bf; border-color: rgba(139, 147, 161, 0.3); }
.ui-badge--sensor { background: ${COLOR.infoSoft}; color: ${COLOR.info}; border-color: rgba(90, 157, 232, 0.4); }

/* ── 행 hover / 선택 (목록·로그) ──────────────────────────────────── */
.rsw-hover-row { transition: background-color 0.1s ease; }
.rsw-hover-row:hover { background: rgba(255, 255, 255, 0.045); }
.rsw-entity-row {
  border-left: 2px solid transparent;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.1s ease, border-color 0.1s ease;
}
.rsw-entity-row:hover { background: rgba(255, 255, 255, 0.045); }
.rsw-entity-row--selected { background: var(--rsw-accent-soft); border-left-color: var(--rsw-accent); }
.rsw-entity-row--selected:hover { background: var(--rsw-accent-soft); }

/* ── 접기 트랜지션 (max-height/height는 각 패널이 인라인으로 구동) ── */
.ui-collapsible { transition: max-height 0.2s ease, opacity 0.16s ease, padding 0.2s ease; }
.ui-collapsible-h { transition: height 0.2s ease; }
`;
  document.head.appendChild(style);
}
