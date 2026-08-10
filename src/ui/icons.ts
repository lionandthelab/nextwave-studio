// ui/icons.ts — SVG 아이콘 세트 (UX_AUDIT C-13)
//
// 구 시스템은 컬러 이모지 10종(📂 💾 🦾 🗣 ✨ 📚 🎯 ✋ 💥 🔖)과 모노크롬 딩벳 24종
// (▶ ⏸ ⏹ ⏭ ⚙ ⚠ ● ▣ ◍ ⬭ ▾ ▴ ▸ «)을 **같은 슬롯에 섞어** 썼다. 그 대가:
//   (a) OS/브라우저마다 다른 그림 → 스크린샷·문서·데모가 재현되지 않는다
//   (b) currentColor를 못 받아 hover/disabled/selected에서 텍스트만 밝아진다(상태 변화 반쪽)
//   (c) 베이스라인·advance width가 텍스트 폰트와 달라 광학 정렬이 불가능하다
//   (d) Windows 고대비 모드에서 강제 치환 대상이 아니라 색이 그대로 남는다
//
// ── 규약 ────────────────────────────────────────────────────────────
// - 16×16 viewBox 그리드. 모든 경로는 이 그리드 안에서 광학적으로 정렬된다.
// - `stroke: currentColor` + `fill: none` + `stroke-width: 1.5` + round cap/join 이 기본.
//   면으로 그려야 하는 것(play/stop 등)만 `filled: true`로 `fill: currentColor`를 쓴다.
// - **아이콘 자체에 색을 칠하지 않는다.** 종류 색은 노드 좌측 스트립 같은 별도 요소가
//   담당한다(theme.ts의 CATEGORY 참조).
// - `aria-hidden="true"` + `focusable="false"` — 아이콘은 항상 장식이고, 접근 가능한
//   이름은 감싸는 버튼의 `aria-label`이 제공한다(makeButton 규약).

import { ICON, makeButton } from './theme';
import type { ButtonVariant } from './theme';

interface IconDef {
  readonly paths: readonly string[];
  /** true면 fill: currentColor + stroke 없음 */
  readonly filled?: boolean;
  /** 추가 원 요소 [cx, cy, r][] */
  readonly circles?: readonly (readonly [number, number, number])[];
}

const ICONS = {
  // ── 트랜스포트 (▶ ⏸ ⏹ ⏭ 대체) ─────────────────────────────────
  play: { paths: ['M5.4 3.5 12.8 8 5.4 12.5Z'], filled: true },
  pause: { paths: ['M5.6 3.4h2.1v9.2H5.6zM8.9 3.4H11v9.2H8.9z'], filled: true },
  stop: { paths: ['M4.3 4.3h7.4v7.4H4.3z'], filled: true },
  stepForward: { paths: ['M4.2 3.5 10.4 8 4.2 12.5Z', 'M11.2 3.5h1.6v9h-1.6z'], filled: true },
  stepBack: { paths: ['M11.8 3.5 5.6 8l6.2 4.5Z', 'M3.2 3.5h1.6v9H3.2z'], filled: true },

  // ── 파일 / 히스토리 (📂 💾 ↶ ↷ 대체) ──────────────────────────
  folderOpen: {
    paths: [
      'M2 5.4a1.2 1.2 0 0 1 1.2-1.2h2.9l1.4 1.7H12a1.2 1.2 0 0 1 1.2 1.2v.7',
      'M2 6.6h12.3l-1.5 5.1a1.2 1.2 0 0 1-1.15.8H3.2A1.2 1.2 0 0 1 2 11.3z',
    ],
  },
  save: {
    paths: [
      'M3 4.2a1.2 1.2 0 0 1 1.2-1.2h6.1L13 5.7v6.1A1.2 1.2 0 0 1 11.8 13H4.2A1.2 1.2 0 0 1 3 11.8z',
      'M5.6 3v3.1h4.6V3',
      'M5.6 13V9.5h4.8V13',
    ],
  },
  undo: { paths: ['M5.9 4.1 2.9 7.1l3 3', 'M2.9 7.1h6.2a3.6 3.6 0 1 1 0 7.2H6.6'] },
  redo: { paths: ['M10.1 4.1l3 3-3 3', 'M13.1 7.1H6.9a3.6 3.6 0 1 0 0 7.2h2.5'] },
  upload: { paths: ['M8 11.6V3.1', 'M4.8 6.3 8 3.1l3.2 3.2', 'M2.8 13.4h10.4'] },
  download: { paths: ['M8 3.1v8.5', 'M4.8 8.4 8 11.6l3.2-3.2', 'M2.8 13.4h10.4'] },
  copy: {
    paths: [
      'M5.9 5.9h6.3a1.1 1.1 0 0 1 1.1 1.1v6.3a1.1 1.1 0 0 1-1.1 1.1H5.9a1.1 1.1 0 0 1-1.1-1.1V7a1.1 1.1 0 0 1 1.1-1.1z',
      'M2.7 10.1V3.8a1.1 1.1 0 0 1 1.1-1.1h6.3',
    ],
  },
  trash: {
    paths: [
      'M3.2 4.4h9.6',
      'M6.4 4.4V3.1a.8.8 0 0 1 .8-.8h1.6a.8.8 0 0 1 .8.8v1.3',
      'M4.6 4.4l.6 8.4a1 1 0 0 0 1 .9h3.6a1 1 0 0 0 1-.9l.6-8.4',
      'M6.9 7v4M9.1 7v4',
    ],
  },

  // ── 생성 / 설정 (✨ ⚙ 대체) ────────────────────────────────────
  wand: {
    paths: [
      'M9.6 2 10.7 4.9 13.6 6l-2.9 1.1L9.6 10 8.5 7.1 5.6 6l2.9-1.1Z',
      'M4.1 9.9l.6 1.5 1.5.6-1.5.6-.6 1.5-.6-1.5L2 12l1.5-.6z',
    ],
  },
  settings: {
    paths: [
      'M8 1.9v1.7M8 12.4v1.7M14.1 8h-1.7M3.6 8H1.9',
      'M12.31 3.69 11.1 4.9M4.9 11.1l-1.21 1.21M12.31 12.31 11.1 11.1M4.9 4.9 3.69 3.69',
    ],
    circles: [[8, 8, 2.3]],
  },
  help: { paths: ['M6.1 6.2a1.95 1.95 0 1 1 2.5 2.2c-.63.28-.95.72-.95 1.35v.35'], circles: [[8, 8, 6.1], [8, 12.1, 0.35]] },
  keyboard: {
    paths: [
      'M2 4.6a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v6.8a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1z',
      'M5.4 10.1h5.2',
    ],
    circles: [[4.6, 6.4, 0.42], [7, 6.4, 0.42], [9.4, 6.4, 0.42], [11.6, 6.4, 0.42], [5.6, 8.3, 0.42], [8, 8.3, 0.42], [10.4, 8.3, 0.42]],
  },

  // ── 도메인 (🦾 ✋ ⏱ 💥 🔖 대체) ────────────────────────────────
  robotArm: {
    paths: ['M2.6 13.6h4.6', 'M4.9 13.6V9.3L8.7 5.5', 'M8.7 5.5l2.6-2.6'],
    circles: [[4.9, 9.3, 1.25], [8.7, 5.5, 1.25], [11.7, 2.6, 1.1]],
  },
  gripper: {
    paths: [
      'M5.2 2.4v5.4M10.8 2.4v5.4',
      'M3.4 7.8h9.2v2.1a3 3 0 0 1-3 3H6.4a3 3 0 0 1-3-3z',
    ],
  },
  timer: { paths: ['M8 6.2v3.2l2.1 1.4', 'M6.1 1.9h3.8'], circles: [[8, 9.2, 5.3]] },
  impact: {
    paths: ['M8 1.6 9.5 5.6 13.4 4 11.6 7.9 15 10.1l-4.2.5.5 4.2-3.3-2.6-3.3 2.6.5-4.2L1 10.1l3.4-2.2L2.6 4l3.9 1.6Z'],
  },
  bookmark: { paths: ['M4.2 2.9a.9.9 0 0 1 .9-.9h5.8a.9.9 0 0 1 .9.9v11l-3.8-2.9-3.8 2.9z'] },
  target: { paths: ['M8 1.4v2.4M8 12.2v2.4M14.6 8h-2.4M3.8 8H1.4'], circles: [[8, 8, 4.6], [8, 8, 1.5]] },

  // ── 프리미티브 (▣ ● ◍ ⬭ ▭ 대체) ───────────────────────────────
  shapeBox: { paths: ['M8 1.9 14 5.1v5.8L8 14.1 2 10.9V5.1Z', 'M2 5.1 8 8.3l6-3.2', 'M8 8.3v5.8'] },
  shapeSphere: { paths: ['M3.1 5.9a9 9 0 0 0 9.8 0M3.1 10.1a9 9 0 0 1 9.8 0'], circles: [[8, 8, 6.1]] },
  shapeCylinder: {
    paths: [
      'M3.4 4.4a4.6 2.1 0 0 1 9.2 0a4.6 2.1 0 0 1-9.2 0',
      'M3.4 4.4v7.2a4.6 2.1 0 0 0 9.2 0V4.4',
    ],
  },
  shapeCapsule: {
    paths: ['M8 1.9a3.3 3.3 0 0 1 3.3 3.3v5.6a3.3 3.3 0 0 1-6.6 0V5.2A3.3 3.3 0 0 1 8 1.9z'],
  },
  shapePlane: { paths: ['M1.5 10.6 8 6.9l6.5 3.7L8 14.3z', 'M4.7 8.8 11.2 12.5'] },
  /**
   * 컨베이어: 양끝 롤러 2개 + 그 위를 잇는 벨트 상판 + 진행 방향 화살표.
   * 화살표가 있어야 "표면이 흐른다"는 것이 정지 화면에서도 읽힌다.
   */
  conveyor: {
    paths: ['M3.4 9.4h9.2', 'M6.2 5.6h4.4', 'M9.1 4.2 10.6 5.6 9.1 7'],
    circles: [
      [3.4, 11.2, 1.8],
      [12.6, 11.2, 1.8],
    ],
  },

  // ── 셰브론 / 화살표 (▾ ▴ ▸ « » 대체) ──────────────────────────
  chevronDown: { paths: ['M4 6.4 8 10.4 12 6.4'] },
  chevronUp: { paths: ['M4 9.6 8 5.6l4 4'] },
  chevronRight: { paths: ['M6.4 4 10.4 8l-4 4'] },
  chevronLeft: { paths: ['M9.6 4 5.6 8l4 4'] },
  chevronsLeft: { paths: ['M7.4 4 3.4 8l4 4', 'M12.6 4 8.6 8l4 4'] },
  chevronsRight: { paths: ['M8.6 4 12.6 8l-4 4', 'M3.4 4 7.4 8l-4 4'] },

  // ── 상태 / 피드백 (✕ ⚠ ✓ 대체) ────────────────────────────────
  close: { paths: ['M4.2 4.2 11.8 11.8M11.8 4.2 4.2 11.8'] },
  alert: { paths: ['M8 2.2 14.6 13.4H1.4z', 'M8 6.4v3.1'], circles: [[8, 11.4, 0.42]] },
  check: { paths: ['M3.4 8.4 6.4 11.4 12.6 4.6'] },
  info: { paths: ['M8 7.6v3.9'], circles: [[8, 8, 6.1], [8, 5.2, 0.42]] },

  // ── 뷰 / 도구 ─────────────────────────────────────────────────
  braces: {
    paths: [
      'M6.2 2.4c-1.4 0-2.1.7-2.1 2v1.5c0 .95-.5 1.5-1.3 1.8v.6c.8.3 1.3.85 1.3 1.8v1.5c0 1.3.7 2 2.1 2',
      'M9.8 2.4c1.4 0 2.1.7 2.1 2v1.5c0 .95.5 1.5 1.3 1.8v.6c-.8.3-1.3.85-1.3 1.8v1.5c0 1.3-.7 2-2.1 2',
    ],
  },
  workflow: {
    paths: [
      'M1.6 5.4h3.2v3.2H1.6zM11.2 7.4h3.2v3.2h-3.2z',
      'M6.2 7h2.2a1.4 1.4 0 0 1 1.4 1.4v.6',
      'M4.8 7v3.6a1.4 1.4 0 0 0 1.4 1.4h2.2',
    ],
  },
  library: { paths: ['M3 2.9h3.1v10.2H3zM7.3 2.9h3.1v10.2H7.3z', 'M11.6 3.3l1.9.5-1.6 9.4-1.4-.4'] },
  search: { paths: ['M10.6 10.6 13.8 13.8'], circles: [[7.1, 7.1, 4.4]] },
  fit: {
    paths: [
      'M2.4 6V3.6a1.2 1.2 0 0 1 1.2-1.2H6',
      'M10 2.4h2.4a1.2 1.2 0 0 1 1.2 1.2V6',
      'M13.6 10v2.4a1.2 1.2 0 0 1-1.2 1.2H10',
      'M6 13.6H3.6a1.2 1.2 0 0 1-1.2-1.2V10',
    ],
  },
  plus: { paths: ['M8 3.4v9.2M3.4 8h9.2'] },
  minus: { paths: ['M3.4 8h9.2'] },
  home: { paths: ['M2.4 7.5 8 2.7l5.6 4.8', 'M4 7v6.3h8V7'] },
  refresh: { paths: ['M13.2 6.6A5.5 5.5 0 1 0 13.4 9.6', 'M13.6 2.7v3.9h-3.9'] },
  gauge: { paths: ['M2.4 12.1a5.6 5.6 0 1 1 11.2 0', 'M8 12.1 11.1 7.6'] },
  eye: { paths: ['M1.6 8S4 3.7 8 3.7 14.4 8 14.4 8 12 12.3 8 12.3 1.6 8 1.6 8z'], circles: [[8, 8, 1.9]] },
  layers: { paths: ['M8 1.9 14 5.1 8 8.3 2 5.1Z', 'M2 8.3 8 11.5l6-3.2', 'M2 11 8 14.2 14 11'] },
  speech: {
    paths: [
      'M2.4 4.4a1.4 1.4 0 0 1 1.4-1.4h8.4a1.4 1.4 0 0 1 1.4 1.4v5a1.4 1.4 0 0 1-1.4 1.4H7.2L3.9 13.6v-2.8h-.1a1.4 1.4 0 0 1-1.4-1.4z',
    ],
  },
  panelLeft: { paths: ['M2.2 3.6a1.2 1.2 0 0 1 1.2-1.2h9.2a1.2 1.2 0 0 1 1.2 1.2v8.8a1.2 1.2 0 0 1-1.2 1.2H3.4a1.2 1.2 0 0 1-1.2-1.2z', 'M6.4 2.4v11.2'] },
  panelRight: { paths: ['M2.2 3.6a1.2 1.2 0 0 1 1.2-1.2h9.2a1.2 1.2 0 0 1 1.2 1.2v8.8a1.2 1.2 0 0 1-1.2 1.2H3.4a1.2 1.2 0 0 1-1.2-1.2z', 'M9.6 2.4v11.2'] },
  panelBottom: { paths: ['M2.2 3.6a1.2 1.2 0 0 1 1.2-1.2h9.2a1.2 1.2 0 0 1 1.2 1.2v8.8a1.2 1.2 0 0 1-1.2 1.2H3.4a1.2 1.2 0 0 1-1.2-1.2z', 'M2.4 9.8h11.2'] },
  maximize: { paths: ['M9.4 2.6h4v4', 'M6.6 13.4h-4v-4', 'M13.4 2.6 9.1 6.9', 'M2.6 13.4l4.3-4.3'] },
  minimize: { paths: ['M13.1 2.9 8.8 7.2M8.8 3.4v3.8h3.8', 'M2.9 13.1 7.2 8.8M7.2 12.6V8.8H3.4'] },

  // ── 기즈모 모드 ────────────────────────────────────────────────
  move: { paths: ['M8 1.9v12.2M1.9 8h12.2', 'M8 1.9 6.2 3.9M8 1.9l1.8 2M8 14.1l-1.8-2M8 14.1l1.8-2', 'M1.9 8l2-1.8M1.9 8l2 1.8M14.1 8l-2-1.8M14.1 8l-2 1.8'] },
  rotate: { paths: ['M13.1 8A5.1 5.1 0 1 1 11 3.9', 'M13.4 1.5v3.4H10'] },
  scale: { paths: ['M2.6 13.4V9.7M2.6 13.4h3.7', 'M2.6 13.4 7.4 8.6', 'M9.6 2.6h3.8v3.8h-3.8z'] },
  magnet: { paths: ['M4 3v5a4 4 0 0 0 8 0V3', 'M4 3h3.1v5a.9.9 0 0 0 1.8 0V3H12'] },

  // ── 콘솔 평면 (docs/BACKEND.md — 목록·표·계정·연결 상태) ────────
  // search · plus · check · alert · chevronRight/Left · refresh · home은 위에 이미 있다.
  /** 필터 깔때기 */
  filter: { paths: ['M2.6 3.2h10.8l-3.6 5.2v3.4L6.2 14V8.4z'] },
  /** 목록 보기 (불릿 3행) */
  list: {
    paths: ['M5.8 4.2h7.6', 'M5.8 8h7.6', 'M5.8 11.8h7.6'],
    circles: [[3, 4.2, 0.55], [3, 8, 0.55], [3, 11.8, 0.55]],
  },
  /** 카드/그리드 보기 (2×2) */
  grid: {
    paths: ['M2.6 2.6h4.6v4.6H2.6z', 'M8.8 2.6h4.6v4.6H8.8z', 'M2.6 8.8h4.6v4.6H2.6z', 'M8.8 8.8h4.6v4.6H8.8z'],
  },
  /** 통계 막대 차트 */
  chart: { paths: ['M2.4 13.4h11.2', 'M4.6 13.4V8.6', 'M8 13.4V4.6', 'M11.4 13.4V6.8'] },
  /** 시각 (timer와 달리 스톱워치 꼭지 없는 일반 시계) */
  clock: { paths: ['M8 4.8V8l2.2 1.5'], circles: [[8, 8, 5.9]] },
  /** 더 보기 (⋯ 대체) */
  more: { paths: [], filled: true, circles: [[3.4, 8, 1.1], [8, 8, 1.1], [12.6, 8, 1.1]] },
  /** 사용자 1인 (계정 타일) */
  user: { paths: ['M3.2 13.6a4.8 4.8 0 0 1 9.6 0'], circles: [[8, 5.4, 2.7]] },
  /** 사용자 여럿 (팀/사용자 관리) */
  users: {
    paths: ['M1.8 13.6a4.2 4.2 0 0 1 8.4 0', 'M11.4 9.7a4.2 4.2 0 0 1 2.9 3.9'],
    circles: [[6, 5.9, 2.4], [11.1, 4.9, 1.9]],
  },
  /** 공정 (톱니 지붕 공장) */
  factory: {
    paths: ['M2.6 13.4V6.2l3.4 2.4V6.2l3.4 2.4V3.4h4.2v10z', 'M6.2 13.4v-2.4h3.6v2.4'],
  },
  /** 작업 (클립보드) */
  clipboard: {
    paths: [
      'M5.6 3.2H4.4a1.1 1.1 0 0 0-1.1 1.1v8.2a1.1 1.1 0 0 0 1.1 1.1h7.2a1.1 1.1 0 0 0 1.1-1.1V4.3a1.1 1.1 0 0 0-1.1-1.1h-1.2',
      'M5.6 2.2h4.8v2H5.6z',
      'M5.8 8h4.4',
      'M5.8 10.6h3',
    ],
  },
  /** 재사용 블록 (퍼즐 조각 — 위 꼭지 + 우 꼭지) */
  puzzle: { paths: ['M3.4 13.4V6.2h2.3a1.7 1.7 0 1 1 3.4 0H12v2.3a1.7 1.7 0 1 1 0 3.4v1.5z'] },
  /** 장비 연결 (플러그) */
  plug: {
    paths: ['M5.4 1.8v3.4M10.6 1.8v3.4', 'M3.8 5.2h8.4v2.2a4.2 4.2 0 0 1-8.4 0z', 'M8 11.6v2.6'],
  },
  /** 카메라 (장비 종류 · 썸네일 캡처) */
  camera: {
    paths: [
      'M2.4 6a1.2 1.2 0 0 1 1.2-1.2h1.7l1.2-1.6h3l1.2 1.6h1.7A1.2 1.2 0 0 1 13.6 6v5.6a1.2 1.2 0 0 1-1.2 1.2H3.6a1.2 1.2 0 0 1-1.2-1.2z',
    ],
    circles: [[8, 8.7, 2.3]],
  },
  /** 실행 기록 (뒤로 도는 시계 — refresh의 좌우 반전 + 시계바늘) */
  history: {
    paths: ['M2.8 6.6A5.5 5.5 0 1 1 2.6 9.6', 'M2.4 2.7v3.9h3.9', 'M8 5.6V8l1.9 1.3'],
  },
  /** 로그아웃 (문 밖 화살표) */
  logout: {
    paths: [
      'M6.6 13.4H3.8a1.2 1.2 0 0 1-1.2-1.2V3.8a1.2 1.2 0 0 1 1.2-1.2h2.8',
      'M10.2 4.9 13.3 8l-3.1 3.1',
      'M13.3 8H6.2',
    ],
  },
  /** 동기화 (양방향 순환 화살표 — outbox 재전송) */
  sync: {
    paths: [
      'M3.2 5.6A5.8 5.8 0 0 1 12.8 5.6',
      'M12.8 2.5v3.1h-3.1',
      'M12.8 10.4A5.8 5.8 0 0 1 3.2 10.4',
      'M3.2 13.5v-3.1h3.1',
    ],
  },
  /** 오프라인 (구름 + 사선 — BACKEND §6 연결 상태 배지) */
  cloudOff: {
    paths: ['M12 6.7h-.9A5.3 5.3 0 1 0 6 13.3h6a2.7 2.7 0 0 0 0-5.4z', 'M2.6 2.6l10.8 10.8'],
  },
  /** PIN 패드 지우기 (백스페이스 키) */
  backspace: {
    paths: [
      'M5.8 3.2h6.6a1.2 1.2 0 0 1 1.2 1.2v7.2a1.2 1.2 0 0 1-1.2 1.2H5.8L2 8z',
      'M7.4 6.2 11 9.8M11 6.2 7.4 9.8',
    ],
  },
  /** 잠금 (편집 잠금 · PIN 잠금) */
  lock: {
    paths: [
      'M5.2 7.2V5.4a2.8 2.8 0 0 1 5.6 0v1.8',
      'M4 7.2h8a1 1 0 0 1 1 1v4.2a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V8.2a1 1 0 0 1 1-1z',
    ],
    circles: [[8, 10.3, 0.5]],
  },
} as const satisfies Record<string, IconDef>;

export type IconName = keyof typeof ICONS;

const SVG_NS = 'http://www.w3.org/2000/svg';

/**
 * SVG 아이콘 요소를 만든다.
 *
 * 항상 `aria-hidden="true"`다 — 접근 가능한 이름은 감싸는 버튼의 `aria-label`이 준다.
 * `currentColor`를 상속하므로 hover/disabled/selected에서 텍스트와 **함께** 변한다.
 */
export function icon(name: IconName, sizePx: number = ICON.md): SVGSVGElement {
  const def: IconDef = ICONS[name];
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('viewBox', '0 0 16 16');
  svg.setAttribute('width', String(sizePx));
  svg.setAttribute('height', String(sizePx));
  svg.setAttribute('aria-hidden', 'true');
  svg.setAttribute('focusable', 'false');
  svg.style.flex = 'none';
  svg.style.display = 'block';

  const filled = def.filled === true;
  if (filled) {
    svg.setAttribute('fill', 'currentColor');
    svg.setAttribute('stroke', 'none');
  } else {
    svg.setAttribute('fill', 'none');
    svg.setAttribute('stroke', 'currentColor');
    svg.setAttribute('stroke-width', '1.5');
    svg.setAttribute('stroke-linecap', 'round');
    svg.setAttribute('stroke-linejoin', 'round');
  }

  for (const d of def.paths) {
    const path = document.createElementNS(SVG_NS, 'path');
    path.setAttribute('d', d);
    svg.appendChild(path);
  }
  if (def.circles !== undefined) {
    for (const [cx, cy, r] of def.circles) {
      const c = document.createElementNS(SVG_NS, 'circle');
      c.setAttribute('cx', String(cx));
      c.setAttribute('cy', String(cy));
      c.setAttribute('r', String(r));
      // 작은 원(점)은 면으로, 큰 원(윤곽)은 선으로
      if (!filled && r < 0.8) c.setAttribute('fill', 'currentColor');
      svg.appendChild(c);
    }
  }
  return svg;
}

/** innerHTML 컨텍스트용 마크업 문자열 (SVG 안에서 조립할 때) */
export function iconMarkup(name: IconName, sizePx: number = ICON.md): string {
  return icon(name, sizePx).outerHTML;
}

/**
 * 버튼 내용을 `[아이콘] 라벨` 구조로 채운다.
 *
 * 라벨은 `.ui-btn__label`로 감싸므로, 좁은 화면에서 `.ui-btn--icon-only` 클래스 하나로
 * 텍스트만 숨길 수 있다 — `aria-label`/`title`은 그대로라 **접근성 손실이 0**이다
 * (UX_AUDIT C-2의 아이콘 전용 모드).
 */
export function setButtonContent(
  button: HTMLButtonElement,
  iconName: IconName,
  label: string,
  sizePx: number = ICON.md,
): void {
  button.textContent = '';
  button.appendChild(icon(iconName, sizePx));
  if (label !== '') {
    const span = document.createElement('span');
    span.className = 'ui-btn__label';
    span.textContent = label;
    button.appendChild(span);
  }
}

/**
 * 아이콘 + 라벨 버튼을 한 번에 만든다.
 *
 * `theme.ts`는 `icons.ts`를 import하지 않으므로 이 방향은 순환이 아니다
 * (계층: icons → theme).
 */
export function makeIconButton(
  iconName: IconName,
  label: string,
  title: string,
  testId: string,
  variant: ButtonVariant = 'default',
  sizePx: number = ICON.md,
): HTMLButtonElement {
  const button = makeButton(label, title, testId, variant);
  setButtonContent(button, iconName, label, sizePx);
  return button;
}
