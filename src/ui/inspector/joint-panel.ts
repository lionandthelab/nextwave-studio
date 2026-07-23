// ui/inspector/joint-panel.ts — 임시 관절 수동 제어 패널 (ROADMAP Phase 3 "슬라이더 수동 제어")
//
// 계층 규칙 (CLAUDE.md §3): ui는 core 공개 API만 쓴다. 이 패널은 core 내부를 import하지
// 않고, 글루(main.ts)가 주입하는 콜백(JointPanelApi)과 POJO 관절 정보(JointInfo 타입)만
// 사용한다. 뷰 상태(슬라이더 값)는 ui 소유지만 관절 진실은 core(RobotBinding)가 소유한다
// — 슬라이더 input은 setJoint 콜백으로 core에 "요청"할 뿐이고, Home 등 core 쪽 상태
// 변화 후에는 readJoints로 다시 읽어 표시를 맞춘다 (물리/코어 → 뷰 단방향).
//
// Phase 6 인스펙터로 대체될 임시 UI다 — 스타일은 인라인 최소 구성(다크·모노스페이스).

import {
  COLOR,
  FONT,
  LAYOUT,
  RADIUS,
  SHADOW,
  SPACE,
  Z_INDEX,
  ensureThemeStyles,
  makeButton,
  styled,
} from '../theme';
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
/** 상단 커맨드바 아래에 배치 (단독 마운트 기본값 — 스택 편입 시 통합자가 덮는다) */
const PANEL_TOP_PX = LAYOUT.belowBarTopPx;

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

function formatValue(value: number): string {
  return value.toFixed(READOUT_DECIMALS);
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
    border: `1px solid ${COLOR.border}`,
    borderRadius: RADIUS.md,
    boxShadow: SHADOW.panel,
    color: COLOR.text,
    fontFamily: FONT.ui,
    fontSize: '12px',
    lineHeight: '1.5',
    minWidth: '240px',
    boxSizing: 'border-box',
    userSelect: 'none',
  });
  panel.dataset.testid = 'joint-panel';
  // 패널 위 상호작용이 뷰포트(OrbitControls)로 전파되지 않게 차단
  for (const type of ['pointerdown', 'pointermove', 'pointerup', 'wheel', 'contextmenu']) {
    panel.addEventListener(type, (e) => {
      e.stopPropagation();
    });
  }

  // 헤더: 타이틀 + 접기 토글 (inspector.ts/dock과 동일한 ▾/▴ 규약)
  const header = styled(document.createElement('div'), {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: SPACE.md,
    padding: `${SPACE.sm} 10px`,
    borderBottom: `1px solid ${COLOR.borderSoft}`,
    flexShrink: '0',
  });
  const panelTitle = styled(document.createElement('strong'), {
    color: COLOR.textStrong,
    fontSize: '13px',
  });
  panelTitle.textContent = '관절 제어';
  header.appendChild(panelTitle);

  const collapseButton = makeButton('▾ 접기', '패널 접기/펼치기', 'joint-panel-collapse', 'ghost');
  styled(collapseButton, { padding: '0 4px' });
  header.appendChild(collapseButton);
  panel.appendChild(header);

  // 본문 (스크롤 영역): 로봇별 관절 섹션 — 접기는 max-height 트랜지션 (.ui-collapsible)
  const body = styled(document.createElement('div'), {
    overflowY: 'auto',
    minHeight: '0',
    maxHeight: '70vh',
    opacity: '1',
    padding: '8px 12px 10px 12px',
  });
  body.classList.add('ui-scroll', 'ui-collapsible');
  panel.appendChild(body);

  let collapsed = false;
  const paintCollapse = (): void => {
    body.style.maxHeight = collapsed ? '0' : '70vh';
    body.style.opacity = collapsed ? '0' : '1';
    body.style.paddingTop = collapsed ? '0' : '8px';
    body.style.paddingBottom = collapsed ? '0' : '10px';
    body.style.overflowY = collapsed ? 'hidden' : 'auto';
    collapseButton.textContent = collapsed ? '▴ 펼치기' : '▾ 접기';
    collapseButton.setAttribute('aria-expanded', String(!collapsed));
  };
  collapseButton.addEventListener('click', () => {
    collapsed = !collapsed;
    paintCollapse();
  });
  paintCollapse();

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
  const section = styled(document.createElement('section'), { margin: '0 0 10px 0' });

  const header = styled(document.createElement('div'), {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: SPACE.md,
    margin: '0 0 6px 0',
  });
  const title = styled(document.createElement('strong'), {
    color: COLOR.textStrong,
    fontSize: '13px',
  });
  title.textContent = robot.id;
  header.appendChild(title);

  /** Home 후 슬라이더/표시를 core 진실(readJoints)로 되맞추는 갱신 훅 목록 */
  const refreshers: Array<() => void> = [];

  const homeButton = makeButton('Home', `'${robot.id}' home 포즈 재적용`, `joint-home-${robot.id}`);
  homeButton.addEventListener('click', () => {
    api.applyHome(robot.id);
    for (const refresh of refreshers) refresh();
  });
  header.appendChild(homeButton);
  section.appendChild(header);

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

  const row = styled(document.createElement('div'), {
    display: 'grid',
    gridTemplateColumns: '1fr auto',
    alignItems: 'center',
    columnGap: '8px',
    margin: '2px 0',
  });

  const label = styled(document.createElement('span'), {
    color: COLOR.label,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  });
  label.textContent = joint.name;
  label.title = joint.name;

  const readout = styled(document.createElement('span'), {
    color: COLOR.textStrong,
    fontFamily: FONT.mono,
    textAlign: 'right',
    minWidth: '56px',
  });
  readout.textContent = formatValue(initialValue);

  const slider = styled(document.createElement('input'), {
    width: '100%',
    gridColumn: '1 / 3',
    accentColor: COLOR.accent, // 네이티브 range 트랙/썸을 액센트로
  });
  slider.type = 'range';
  slider.min = String(minValue);
  slider.max = String(maxValue);
  slider.step = String(SLIDER_STEP);
  slider.value = String(initialValue);
  slider.setAttribute('aria-label', `${robotId} ${joint.name} 관절 목표값`);
  slider.addEventListener('input', () => {
    const value = Number(slider.value);
    api.setJoint(robotId, joint.name, value);
    readout.textContent = formatValue(value);
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
      readout.textContent = formatValue(value);
    },
  };
}
