// ui/inspector/joint-panel.ts — 관절 수동 제어 패널 (ROADMAP Phase 3 "슬라이더 수동 제어")
//
// 계층 규칙 (CLAUDE.md §3): ui는 core 공개 API만 쓴다. 이 패널은 core 내부를 import하지
// 않고, 글루(main.ts)가 주입하는 콜백(JointPanelApi)과 POJO 관절 정보(JointInfo 타입)만
// 사용한다. 뷰 상태(슬라이더 값)는 ui 소유지만 관절 진실은 core(RobotBinding)가 소유한다
// — 슬라이더 input은 setJoint 콜백으로 core에 "요청"할 뿐이고, Home 등 core 쪽 상태
// 변화 후에는 readJoints로 다시 읽어 표시를 맞춘다 (물리/코어 → 뷰 단방향).
//
// ── 슬라이더 채움 규약 (UX_AUDIT C-18) ──────────────────────────────
// 네이티브 accentColor는 트랙을 **min에서 썸까지** 채운다. 관절 limits가 부호를 걸치면
// (예: ±2.0) 이 채움은 의미가 틀리다 — -0.600이 "35% 진행"처럼, 0.000이 "절반 열림"처럼
// 읽힌다. 관절값은 이 앱의 핵심 수치이므로 부호와 0 기준점이 시각적으로 정확해야 한다.
// 그래서 `.ui-range`(+ 부호 있는 범위면 `.ui-range--bipolar`)를 쓰고 `--rsw-range-track`에
// gradient를 써 넣어 **0에서 썸까지만** 채우고 0 지점에 1px 틱을 그린다
// (inspector.ts의 attachRangeFill/rangeTrackGradient — 노드 에디터와 공용).

import {
  BORDER_WIDTH,
  COLOR,
  ICON,
  LAYOUT,
  RADIUS,
  SHADOW,
  SPACE,
  TYPE,
  Z_INDEX,
  applyType,
  ensureThemeStyles,
  makePanelHeader,
  styled,
} from '../theme';
import { makeIconButton } from '../icons';
import { attachRangeFill, formatFixed } from './inspector';
import type { JointInfo } from '../../core/robot-types';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4, 시각 토큰은 ui/theme.ts) ────

/** limits 없는 revolute/continuous 관절의 슬라이더 폴백 범위 (±π 근사, rad) */
const FALLBACK_REVOLUTE_LIMIT_RAD = 3.1416;
/** limits 없는 prismatic 관절의 슬라이더 폴백 범위 (m) */
const FALLBACK_PRISMATIC_RANGE_M: readonly [number, number] = [0, 0.05];
/** 슬라이더 최소 증분 (rad 또는 m) */
const SLIDER_STEP = 0.001;
/** 값 표시 소수 자릿수 */
const READOUT_DECIMALS = 3;
/** 값 리드아웃 고정 폭 (자릿수가 늘어도 슬라이더가 흔들리지 않게) */
const READOUT_MIN_WIDTH_PX = 56;
/** 상단 커맨드바 아래에 배치 (단독 마운트 기본값 — 스택 편입 시 통합자가 덮는다) */
const PANEL_TOP_PX = LAYOUT.belowBarTopPx;
/** 본문 스크롤 영역 상한 (접기 트랜지션의 펼침 목표값) */
const BODY_MAX_HEIGHT = '70vh';

// ── 공개 타입 ───────────────────────────────────────────────────────

export interface JointPanelRobot {
  readonly id: string;
  /** 유효 limits가 반영된 관절 목록 (core RobotBinding.joints — POJO) */
  readonly joints: readonly JointInfo[];
}

/** 글루(main.ts)가 core 위에서 구현해 주입하는 콜백 표면 — 패널은 이것만 안다 */
export interface JointPanelApi {
  setJoint(robotId: string, jointName: string, value: number): void;
  readJoints(robotId: string): Record<string, number>;
  /** home 관절 포즈 재적용 (core RobotBinding.applyHome) */
  applyHome(robotId: string): void;
}

export interface JointPanelHandle {
  /** 패널 루트 — 글루(main.ts)가 우측 스택 등으로 재배치할 때 사용 (inspector.ts와 동일 규약) */
  readonly el: HTMLElement;
  /** 패널 DOM 제거 (씬 재로드 시) */
  dispose(): void;
}

// ── 내부 헬퍼 ───────────────────────────────────────────────────────

/** 관절의 슬라이더 [min, max] — 유효 limits 우선, 없으면 타입별 폴백 */
function sliderRangeOf(joint: JointInfo): readonly [number, number] {
  if (joint.limits) return joint.limits;
  if (joint.type === 'prismatic') return FALLBACK_PRISMATIC_RANGE_M;
  return [-FALLBACK_REVOLUTE_LIMIT_RAD, FALLBACK_REVOLUTE_LIMIT_RAD];
}

/** 관절값 단위 (CLAUDE.md §4 단위 규약 — prismatic만 미터) */
function unitLabelOf(joint: JointInfo): string {
  return joint.type === 'prismatic' ? '미터' : '라디안';
}

function formatValue(value: number): string {
  return formatFixed(value, READOUT_DECIMALS);
}

// ── 마운트 ──────────────────────────────────────────────────────────

/**
 * 관절 패널을 host에 마운트한다 (우상단 절대 배치).
 * 패널 내부 포인터/휠 이벤트는 stopPropagation으로 흡수한다 — 슬라이더 조작이
 * 뷰포트 orbit 컨트롤로 새어 나가지 않게 한다.
 */
export function mountJointPanel(
  host: HTMLElement,
  robots: readonly JointPanelRobot[],
  api: JointPanelApi,
): JointPanelHandle {
  ensureThemeStyles();
  const panel = styled(document.createElement('div'), {
    position: 'absolute',
    top: `${PANEL_TOP_PX}px`,
    right: '12px',
    zIndex: Z_INDEX.panel,
    maxHeight: `calc(100vh - ${PANEL_TOP_PX + 12}px)`,
    display: 'flex',
    flexDirection: 'column',
    background: COLOR.bgPanel,
    border: `${BORDER_WIDTH.hair} solid ${COLOR.border}`,
    borderRadius: RADIUS.md,
    boxShadow: SHADOW.panel,
    color: COLOR.text,
    minWidth: '240px',
    boxSizing: 'border-box',
    userSelect: 'none',
  });
  applyType(panel, TYPE.body);
  panel.dataset.testid = 'joint-panel';
  // 패널 위 상호작용이 뷰포트(OrbitControls)로 전파되지 않게 차단
  for (const type of ['pointerdown', 'pointermove', 'pointerup', 'wheel', 'contextmenu']) {
    panel.addEventListener(type, (e) => {
      e.stopPropagation();
    });
  }

  // 헤더 — 셰브론 회전 규약은 makePanelHeader가 소유한다(수제 ▾/▴ 제거, UX_AUDIT C-18)
  const header = makePanelHeader('관절 제어', {
    collapsible: true,
    testId: 'joint-panel-header',
  });
  // 구 testId 유지 (측정 스크립트/문서 참조 호환)
  if (header.collapseButton !== null) header.collapseButton.dataset.testid = 'joint-panel-collapse';
  panel.appendChild(header.el);

  // 본문 (스크롤 영역): 로봇별 관절 섹션 — 접기는 max-height 트랜지션 (.ui-collapsible)
  const body = styled(document.createElement('div'), {
    overflowY: 'auto',
    minHeight: '0',
    maxHeight: BODY_MAX_HEIGHT,
    opacity: '1',
    padding: `${SPACE.md} ${SPACE.lg} ${SPACE.lg} ${SPACE.lg}`,
  });
  body.classList.add('ui-scroll', 'ui-collapsible');
  panel.appendChild(body);

  header.onToggle((collapsed) => {
    body.style.maxHeight = collapsed ? '0' : BODY_MAX_HEIGHT;
    body.style.opacity = collapsed ? '0' : '1';
    body.style.paddingTop = collapsed ? '0' : SPACE.md;
    body.style.paddingBottom = collapsed ? '0' : SPACE.lg;
    body.style.overflowY = collapsed ? 'hidden' : 'auto';
  });

  for (const robot of robots) {
    body.appendChild(buildRobotSection(robot, api));
  }

  host.appendChild(panel);
  return {
    el: panel,
    dispose: (): void => {
      panel.remove();
    },
  };
}

function buildRobotSection(robot: JointPanelRobot, api: JointPanelApi): HTMLElement {
  const section = styled(document.createElement('section'), { margin: `0 0 ${SPACE.xl} 0` });

  /** Home 후 슬라이더/표시를 core 진실(readJoints)로 되맞추는 갱신 훅 목록 */
  const refreshers: Array<() => void> = [];

  // 로봇 섹션 헤더 — 패널 헤더와 같은 팩토리(제목 굵기/보더/액션 정렬 규약 단일화)
  const header = makePanelHeader(robot.id, {
    actions: true,
    headingTag: 'h3',
    testId: `joint-section-${robot.id}`,
  });
  styled(header.el, { padding: `${SPACE.xs} 0 ${SPACE.sm} 0` });
  applyType(header.titleEl, TYPE.subhead);
  header.titleEl.title = robot.id;

  const homeButton = makeIconButton(
    'home',
    'Home',
    `'${robot.id}' home 포즈 재적용`,
    `joint-home-${robot.id}`,
    'default',
    ICON.sm,
  );
  homeButton.addEventListener('click', () => {
    api.applyHome(robot.id);
    for (const refresh of refreshers) refresh();
  });
  header.actionsEl?.appendChild(homeButton);
  section.appendChild(header.el);

  const initialValues = api.readJoints(robot.id);
  for (const joint of robot.joints) {
    const { row, refresh } = buildJointRow(robot.id, joint, initialValues[joint.name] ?? 0, api);
    refreshers.push(refresh);
    section.appendChild(row);
  }
  return section;
}

function buildJointRow(
  robotId: string,
  joint: JointInfo,
  initialValue: number,
  api: JointPanelApi,
): { row: HTMLElement; refresh: () => void } {
  const [minValue, maxValue] = sliderRangeOf(joint);
  const unit = unitLabelOf(joint);

  const row = styled(document.createElement('div'), {
    display: 'grid',
    gridTemplateColumns: '1fr auto',
    alignItems: 'center',
    columnGap: SPACE.md,
    margin: `${SPACE.sm} 0 0 0`,
  });

  const label = applyType(document.createElement('span'), TYPE.caption);
  styled(label, {
    color: COLOR.label,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  });
  label.textContent = joint.name;
  label.title = joint.name;

  const readout = applyType(document.createElement('span'), TYPE.monoReadout);
  styled(readout, {
    color: COLOR.textStrong,
    textAlign: 'right',
    minWidth: `${READOUT_MIN_WIDTH_PX}px`,
  });
  readout.textContent = formatValue(initialValue);

  const slider = styled(document.createElement('input'), {
    width: '100%',
    gridColumn: '1 / 3',
  });
  slider.type = 'range';
  slider.min = String(minValue);
  slider.max = String(maxValue);
  slider.step = String(SLIDER_STEP);
  slider.value = String(initialValue);
  slider.dataset.testid = 'joint-slider';
  slider.dataset.joint = joint.name;
  slider.setAttribute('aria-label', `${robotId} ${joint.name} 관절 목표값`);
  // 채움은 0 기준 (부호 있는 범위) — 네이티브 accentColor 대체 (UX_AUDIT C-18)
  const paintFill = attachRangeFill(slider, minValue, maxValue);

  /** 값 하나로 리드아웃·트랙 채움·aria-valuetext를 함께 갱신한다 (셋이 어긋나지 않게) */
  const paintValue = (value: number): void => {
    readout.textContent = formatValue(value);
    paintFill(value);
    // 숫자만 읽히면 rad인지 m인지 알 수 없다 — 단위를 말로 붙인다 (UX_AUDIT C-16)
    slider.setAttribute('aria-valuetext', `${formatValue(value)} ${unit}`);
  };
  paintValue(initialValue);

  slider.addEventListener('input', () => {
    const value = Number(slider.value);
    api.setJoint(robotId, joint.name, value);
    paintValue(value);
  });

  row.appendChild(label);
  row.appendChild(readout);
  row.appendChild(slider);

  return {
    row,
    refresh: (): void => {
      // core 쪽 상태 변화(Home 등) 후 진실을 다시 읽어 뷰를 맞춘다 (core → 뷰 단방향)
      const value = api.readJoints(robotId)[joint.name];
      if (value === undefined) return;
      slider.value = String(value);
      paintValue(value);
    },
  };
}
