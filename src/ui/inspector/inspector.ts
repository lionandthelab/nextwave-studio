// ui/inspector/inspector.ts — MVP 인스펙터 패널
// (ROADMAP Phase 6 "인스펙터(엔티티 목록·선택·트랜스폼/관절 상태 표시)", UX_DESIGN §3.5 (A)의 읽기 부분집합)
//
// READ-heavy MVP: 엔티티 목록 + 선택 + 선택 대상의 트랜스폼/관절 상태 "표시"만 한다.
// 편집(치수·Physics·Transform 쓰기)은 Phase 7 Scene Builder의 몫이다. 이 패널은 물리를
// 절대 변경하지 않는다 — 값은 InspectorDeps 콜백으로 읽기만 한다(뷰에서 물리 pose를
// 역으로 쓰지 않는다 — CLAUDE.md §6 "UI/UX", 불변식 §2.1).
//
// 계층 규칙 (CLAUDE.md §3): ui는 core를 import하지 않는다. 글루(통합자)가 core 파사드
// 위에서 InspectorDeps를 구현해 주입한다. 값 갱신은 통합자가 refresh()를 적당한 주기
// (예: ~150ms 인터벌 또는 engine.onTick 스로틀)로 호출해 일어난다 — 이 모듈은 타이머를
// 소유하지 않는다(주기 결정권은 통합자).
//
// 회전 표시 규약: 내부 진실은 쿼터니언 [x,y,z,w]다 (CLAUDE.md §4 "회전 표현").
// 이 패널의 오일러(deg, XYZ 순서) 변환(quatToEulerDegXYZ)은 **표시 전용**이며
// 역변환·저장 경로가 없다. 행 title(툴팁)로 원본 쿼터니언을 함께 노출한다.
//
// 선택 통지: 목록 행 클릭(같은 행 재클릭 = 해제) 또는 핸들 select() 호출로 선택이
// "바뀔 때만" deps.onSelect가 불린다 — 통합자가 onSelect 안에서 같은 id로 select()를
// 되돌려 불러도 루프가 생기지 않는다(변경 가드). 통합자는 onSelect에서
// render/highlight.ts 펄스 등으로 반응할 수 있다.
//
// 스타일: joint-panel.ts·dock과 같은 시각 언어(다크·모노스페이스·#2e3238 보더·접기
// 토글)를 쓴다. joint-panel은 이 인스펙터로 대체될 임시 UI다(그 파일 헤더 참조) —
// 통합자가 두 패널을 한 컨테이너로 재배치할 수 있도록 핸들에 el을 노출한다.

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
import type { Quat, Vec3 } from '../../schema/types';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4, 시각 토큰은 ui/theme.ts) ────

/** 상단 커맨드바 아래로 내려 배치 (단독 마운트 기본값 — 스택 편입 시 통합자가 덮는다) */
const PANEL_TOP_PX = LAYOUT.belowBarTopPx;
const PANEL_RIGHT_PX = 12;
const PANEL_WIDTH_PX = LAYOUT.rightPanelWidthPx;
/** 하단 독(본문 180px + 탭바 + 여백)을 침범하지 않는 최대 높이 여유 */
const PANEL_BOTTOM_CLEARANCE_PX = 220;

/** position/관절 값 표시 소수 자릿수 (미터/라디안, CLAUDE.md §4 단위 규약) */
export const POSITION_DECIMALS = 3;
export const JOINT_VALUE_DECIMALS = 3;
/** 오일러 각 표시 소수 자릿수 (deg — 표시 전용 변환) */
export const EULER_DEG_DECIMALS = 1;
/** 툴팁에 노출하는 원본 쿼터니언 소수 자릿수 */
export const QUAT_TITLE_DECIMALS = 4;

/** 라디안 → 도 */
export const RAD_TO_DEG = 180 / Math.PI;

/** 오일러 추출 짐벌락 판정 임계 (|sin(y)| 기준 — three.js Euler 'XYZ' 추출과 동일) */
const GIMBAL_LOCK_EPS = 0.9999999;
/** 쿼터니언 노름이 이보다 작으면 무효로 보고 [0,0,0]을 표시한다 (표시 전용 방어) */
const QUAT_NORM_EPS = 1e-12;

/** 관절 표의 고정 컬럼 폭 (행별 grid의 컬럼 정렬 유지) */
const JOINT_VALUE_COL_PX = 52;
const JOINT_LIMITS_COL_PX = 104;

// 시각 토큰은 ui/theme.ts가 소유한다 — 이 모듈은 토큰만 소비한다
const COLOR_TEXT = COLOR.text;
const COLOR_TEXT_STRONG = COLOR.textStrong;
const COLOR_MUTED = COLOR.muted;
const COLOR_LABEL = COLOR.label;
const COLOR_BORDER = COLOR.border;
const COLOR_ROW_BORDER = COLOR.borderSoft;

// ── 공개 타입 ───────────────────────────────────────────────────────

/** 목록 항목 — EntitySpec의 id/type 부분집합 (미지 type은 라벨로 그대로 표시) */
export interface InspectorEntity {
  id: string;
  type: string;
}

/** 물리(진실)에서 읽어 온 pose 스냅샷 — rotation은 쿼터니언 [x,y,z,w] */
export interface InspectorPose {
  position: Vec3;
  rotation: Quat;
}

/** 관절 상태 1건 — valueRad는 revolute면 rad, prismatic이면 m (원시값 그대로 표시) */
export interface InspectorJoint {
  name: string;
  valueRad: number;
  limits?: [number, number];
}

/**
 * 글루(통합자)가 core 파사드 위에서 구현해 주입하는 읽기 전용 데이터 표면.
 * getPose/getJoints가 null이면 해당 섹션은 "없음"으로 표시된다
 * (예: 물리 바디 없는 엔티티의 pose, 로봇이 아닌 엔티티의 관절).
 */
export interface InspectorDeps {
  listEntities(): Array<{ id: string; type: string }>;
  getPose(
    id: string,
  ): { position: [number, number, number]; rotation: [number, number, number, number] } | null;
  getJoints(id: string): Array<{ name: string; valueRad: number; limits?: [number, number] }> | null;
  /** 선택이 "바뀔 때만" 통지 (클릭·select() 공통, 변경 가드 — 파일 헤더 참조) */
  onSelect?(id: string | null): void;
}

export interface InspectorHandle {
  /** 패널 루트 — 통합자가 재배치/컨테이너 편입에 쓸 수 있다 (스타일 소유는 통합자로 이양 가능) */
  readonly el: HTMLElement;
  /** 목록/선택 상태값 재독(통합자가 주기 호출 — 예: ~150ms). 이 모듈은 타이머가 없다 */
  refresh(): void;
  /** 프로그램적 선택 (null = 해제). 미존재 id는 해제로 정규화. 변경 시 onSelect 통지 */
  select(id: string | null): void;
  /** 패널 DOM 제거 */
  dispose(): void;
}

// ── 순수 헬퍼 (DOM 비의존 — node 단위 테스트 대상) ───────────────────

/**
 * 쿼터니언 [x,y,z,w] → 오일러 각(deg) [x,y,z] — Tait-Bryan **XYZ 순서**
 * (three.js `Euler` 기본 순서와 동일한 행렬 추출식).
 *
 * **표시 전용**이다: 내부 진실은 쿼터니언이며(CLAUDE.md §4) 이 값을 되쓰는 경로는 없다.
 * 입력은 정규화하지 않아도 된다(내부 정규화). 노름이 0에 가까우면 [0,0,0]을 돌려준다.
 * |y|=90° 짐벌락에서는 z=0으로 고정하고 나머지를 x에 흡수한다(three.js와 동일 규약).
 */
export function quatToEulerDegXYZ(quat: Quat): [number, number, number] {
  const [qx, qy, qz, qw] = quat;
  const norm = Math.hypot(qx, qy, qz, qw);
  if (!(norm > QUAT_NORM_EPS)) return [0, 0, 0];
  const x = qx / norm;
  const y = qy / norm;
  const z = qz / norm;
  const w = qw / norm;

  // 회전 행렬 성분 (열벡터 규약 — R = Rx·Ry·Rz 분해 기준 추출)
  const m11 = 1 - 2 * (y * y + z * z);
  const m12 = 2 * (x * y - z * w);
  const m13 = 2 * (x * z + y * w);
  const m22 = 1 - 2 * (x * x + z * z);
  const m23 = 2 * (y * z - x * w);
  const m32 = 2 * (y * z + x * w);
  const m33 = 1 - 2 * (x * x + y * y);

  const sinY = Math.min(Math.max(m13, -1), 1);
  const eulerYRad = Math.asin(sinY);
  let eulerXRad: number;
  let eulerZRad: number;
  if (Math.abs(m13) < GIMBAL_LOCK_EPS) {
    eulerXRad = Math.atan2(-m23, m33);
    eulerZRad = Math.atan2(-m12, m11);
  } else {
    // 짐벌락 (y = ±90°): x/z 축이 겹친다 — z=0으로 고정
    eulerXRad = Math.atan2(m32, m22);
    eulerZRad = 0;
  }
  return [eulerXRad * RAD_TO_DEG, eulerYRad * RAD_TO_DEG, eulerZRad * RAD_TO_DEG];
}

/**
 * 고정 소수점 포맷. 반올림 결과가 0이면 음의 부호를 지운다
 * ("-0.000" 방지 — 리드아웃이 0 근처에서 깜빡이며 부호가 튀지 않게).
 */
export function formatFixed(value: number, decimals: number): string {
  const text = value.toFixed(decimals);
  return /^-0(?:\.0+)?$/.test(text) ? text.slice(1) : text;
}

/** position 리드아웃: "X 0.400 · Y 0.050 · Z 0.000" (미터, 소수 3자리) */
export function formatPosition(position: Vec3): string {
  const [x, y, z] = position;
  const f = (v: number): string => formatFixed(v, POSITION_DECIMALS);
  return `X ${f(x)} · Y ${f(y)} · Z ${f(z)}`;
}

/** 오일러(deg) 리드아웃: "RX 0.0° · RY 90.0° · RZ 0.0°" (표시 전용 변환 결과) */
export function formatEulerDeg(eulerDeg: [number, number, number]): string {
  const [x, y, z] = eulerDeg;
  const f = (v: number): string => formatFixed(v, EULER_DEG_DECIMALS);
  return `RX ${f(x)}° · RY ${f(y)}° · RZ ${f(z)}°`;
}

/** 원본 쿼터니언 툴팁 텍스트: "quat [0.0000, 0.7071, 0.0000, 0.7071]" */
export function quatReadout(quat: Quat): string {
  const [x, y, z, w] = quat;
  const f = (v: number): string => formatFixed(v, QUAT_TITLE_DECIMALS);
  return `quat [${f(x)}, ${f(y)}, ${f(z)}, ${f(w)}]`;
}

/** 관절 limits 리드아웃: "[-1.571, 1.571]" · 없으면 "—" */
export function formatLimits(limits?: [number, number]): string {
  if (!limits) return '—';
  const [lower, upper] = limits;
  return `[${formatFixed(lower, JOINT_VALUE_DECIMALS)}, ${formatFixed(upper, JOINT_VALUE_DECIMALS)}]`;
}

/** 목록 행 클릭 의미론: 이미 선택된 행을 다시 클릭하면 해제(null), 아니면 그 행 선택 */
export function nextSelection(current: string | null, clickedId: string): string | null {
  return current === clickedId ? null : clickedId;
}

/** 요청 선택 id를 현재 엔티티 목록에 대해 정규화 — 목록에 없으면 해제(null) */
export function resolveSelection(
  entityIds: readonly string[],
  requested: string | null,
): string | null {
  if (requested === null) return null;
  return entityIds.includes(requested) ? requested : null;
}

/** 엔티티 목록의 변경 감지 키 (id/type/순서에 민감 — 같으면 목록 DOM 재구축 생략) */
export function entityListKey(entities: ReadonlyArray<{ id: string; type: string }>): string {
  // 구분자는 id에 등장하지 않는 제어 문자 — 항목 경계 모호성("a b"+"c" vs "a"+"b c") 제거
  const FIELD_SEP = String.fromCharCode(0);
  const ENTRY_SEP = String.fromCharCode(1);
  return entities.map((e) => `${e.id}${FIELD_SEP}${e.type}`).join(ENTRY_SEP);
}

/** 엔티티 type → 목록 아이콘/한국어 라벨 (미지 type은 원문 그대로 라벨) */
export function entityTypeMeta(type: string): { icon: string; label: string } {
  switch (type) {
    case 'robot':
      return { icon: '🦾', label: '로봇' };
    case 'object':
      return { icon: '▣', label: '사물' };
    case 'static':
      return { icon: '▤', label: '정적' };
    default:
      return { icon: '◌', label: type };
  }
}

// ── 내부 DOM 헬퍼 ───────────────────────────────────────────────────

/** 섹션 캡션 (muted 소제목) */
function caption(text: string): HTMLElement {
  const el = styled(document.createElement('div'), {
    color: COLOR_MUTED,
    fontSize: '11px',
    margin: '8px 0 2px 0',
  });
  el.textContent = text;
  return el;
}

/** muted 안내/플레이스홀더 줄 */
function mutedLine(text: string): HTMLElement {
  const el = styled(document.createElement('div'), {
    color: COLOR_MUTED,
    padding: '2px 0',
  });
  el.textContent = text;
  return el;
}

// ── 마운트 ──────────────────────────────────────────────────────────

/**
 * 인스펙터를 host에 마운트한다 (우측 절대 배치 — 상단 재생 바 아래).
 * 패널 내부 포인터/휠은 stopPropagation으로 흡수한다 — 뷰포트 orbit 컨트롤로
 * 새지 않게 하는 기존 패널 규약(joint-panel/dock)과 동일.
 */
export function mountInspector(host: HTMLElement, deps: InspectorDeps): InspectorHandle {
  ensureThemeStyles();
  const panel = styled(document.createElement('div'), {
    position: 'absolute',
    top: `${PANEL_TOP_PX}px`,
    right: `${PANEL_RIGHT_PX}px`,
    zIndex: Z_INDEX.panel,
    width: `${PANEL_WIDTH_PX}px`,
    maxHeight: `calc(100vh - ${PANEL_TOP_PX + PANEL_BOTTOM_CLEARANCE_PX}px)`,
    display: 'flex',
    flexDirection: 'column',
    background: COLOR.bgPanel,
    border: `1px solid ${COLOR_BORDER}`,
    borderRadius: RADIUS.md,
    boxShadow: SHADOW.panel,
    color: COLOR_TEXT,
    fontFamily: FONT.ui,
    fontSize: '12px',
    lineHeight: '1.5',
    boxSizing: 'border-box',
    userSelect: 'none',
  });
  panel.dataset.testid = 'inspector';
  for (const type of ['pointerdown', 'pointermove', 'pointerup', 'wheel', 'contextmenu']) {
    panel.addEventListener(type, (e) => {
      e.stopPropagation();
    });
  }

  // 헤더: 타이틀 + 접기 토글 (dock과 동일한 ▾/▴ 규약)
  const header = styled(document.createElement('div'), {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: SPACE.md,
    padding: `${SPACE.sm} 10px`,
    borderBottom: `1px solid ${COLOR_ROW_BORDER}`,
    flexShrink: '0',
  });
  const title = styled(document.createElement('strong'), {
    color: COLOR_TEXT_STRONG,
    fontSize: '13px',
  });
  title.textContent = 'Inspector';
  header.appendChild(title);

  const collapseButton = makeButton('▾ 접기', '패널 접기/펼치기', 'inspector-collapse', 'ghost');
  styled(collapseButton, { padding: '0 4px' });
  header.appendChild(collapseButton);
  panel.appendChild(header);

  // 본문 (스크롤 영역): 엔티티 목록 + 상세 — 접기는 max-height 트랜지션 (.ui-collapsible)
  const body = styled(document.createElement('div'), {
    overflowY: 'auto',
    minHeight: '0',
    maxHeight: '70vh',
    opacity: '1',
    padding: '6px 10px 10px 10px',
  });
  body.classList.add('ui-scroll', 'ui-collapsible');
  panel.appendChild(body);

  const listCaption = caption('엔티티');
  const listContainer = document.createElement('div');
  listContainer.dataset.testid = 'inspector-entities';
  const detailContainer = document.createElement('div');
  detailContainer.dataset.testid = 'inspector-detail';
  body.appendChild(listCaption);
  body.appendChild(listContainer);
  body.appendChild(detailContainer);

  // ── 상태 (뷰 상태는 ui 소유 — 시뮬 진실은 deps 콜백 너머의 core) ──
  let selectedId: string | null = null;
  let collapsed = false;
  let renderedListKey: string | null = null;
  const rowById = new Map<string, HTMLElement>();

  const paintCollapse = (): void => {
    body.style.maxHeight = collapsed ? '0' : '70vh';
    body.style.opacity = collapsed ? '0' : '1';
    body.style.paddingTop = collapsed ? '0' : '6px';
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

  const paintListSelection = (): void => {
    for (const [id, row] of rowById) {
      // 선택 시각은 클래스 토글 — hover 스타일과 공존한다 (theme .rsw-entity-row--selected)
      row.classList.toggle('rsw-entity-row--selected', id === selectedId);
      row.setAttribute('aria-selected', String(id === selectedId));
    }
  };

  // ── 상세 렌더 (읽기 전용 리드아웃 — refresh마다 재구축) ────────────
  const renderDetail = (): void => {
    detailContainer.replaceChildren();
    if (selectedId === null) {
      detailContainer.appendChild(caption('상세'));
      detailContainer.appendChild(mutedLine('엔티티를 선택하면 상태가 표시됩니다'));
      return;
    }

    const entity = deps.listEntities().find((e) => e.id === selectedId);
    const meta = entityTypeMeta(entity?.type ?? '');

    detailContainer.appendChild(caption('상세'));
    const idRow = styled(document.createElement('div'), {
      display: 'flex',
      alignItems: 'baseline',
      justifyContent: 'space-between',
      gap: '8px',
      padding: '1px 0',
    });
    const idText = styled(document.createElement('span'), {
      color: COLOR_TEXT_STRONG,
      overflow: 'hidden',
      textOverflow: 'ellipsis',
      whiteSpace: 'nowrap',
    });
    idText.textContent = `${meta.icon} ${selectedId}`;
    const typeBadge = styled(document.createElement('span'), {
      color: COLOR_LABEL,
      fontSize: '11px',
      flexShrink: '0',
    });
    typeBadge.textContent = meta.label;
    idRow.appendChild(idText);
    idRow.appendChild(typeBadge);
    detailContainer.appendChild(idRow);

    // Transform — 물리(진실)에서 읽은 pose 리드아웃 (편집 없음)
    detailContainer.appendChild(caption('Transform'));
    const pose = deps.getPose(selectedId);
    if (pose === null) {
      detailContainer.appendChild(mutedLine('pose 없음 (물리 바디 미등록)'));
    } else {
      const positionLabel = styled(document.createElement('div'), {
        color: COLOR_LABEL,
        fontSize: '11px',
      });
      positionLabel.textContent = 'position (m)';
      const positionValue = styled(document.createElement('div'), {
        color: COLOR_TEXT_STRONG,
        fontFamily: FONT.mono,
        padding: '0 0 2px 8px',
      });
      positionValue.dataset.testid = 'inspector-position';
      positionValue.textContent = formatPosition(pose.position);

      const rotationLabel = styled(document.createElement('div'), {
        color: COLOR_LABEL,
        fontSize: '11px',
      });
      rotationLabel.textContent = 'rotation (deg)';
      const rotationValue = styled(document.createElement('div'), {
        color: COLOR_TEXT_STRONG,
        fontFamily: FONT.mono,
        padding: '0 0 2px 8px',
      });
      rotationValue.dataset.testid = 'inspector-rotation';
      // 오일러는 표시 전용 — 원본(진실) 쿼터니언은 툴팁으로 노출 (CLAUDE.md §4)
      rotationValue.textContent = formatEulerDeg(quatToEulerDegXYZ(pose.rotation));
      rotationValue.title = `${quatReadout(pose.rotation)} — 내부 진실은 쿼터니언, 오일러(XYZ)는 표시 전용`;

      detailContainer.appendChild(positionLabel);
      detailContainer.appendChild(positionValue);
      detailContainer.appendChild(rotationLabel);
      detailContainer.appendChild(rotationValue);
    }

    // 관절 (로봇 엔티티만 — getJoints가 null이면 섹션 자체를 생략)
    const joints = deps.getJoints(selectedId);
    if (joints !== null) {
      detailContainer.appendChild(caption(`관절 (${joints.length})`));
      if (joints.length === 0) {
        detailContainer.appendChild(mutedLine('없음'));
      } else {
        const gridColumns = `1fr ${JOINT_VALUE_COL_PX}px ${JOINT_LIMITS_COL_PX}px`;
        const head = styled(document.createElement('div'), {
          display: 'grid',
          gridTemplateColumns: gridColumns,
          columnGap: '6px',
          color: COLOR_MUTED,
          fontSize: '11px',
          borderBottom: `1px solid ${COLOR_ROW_BORDER}`,
        });
        for (const [text, align] of [
          ['이름', 'left'],
          ['값', 'right'],
          ['limits', 'right'],
        ] as const) {
          const cell = styled(document.createElement('span'), { textAlign: align });
          cell.textContent = text;
          head.appendChild(cell);
        }
        detailContainer.appendChild(head);

        for (const joint of joints) {
          const row = styled(document.createElement('div'), {
            display: 'grid',
            gridTemplateColumns: gridColumns,
            columnGap: '6px',
            padding: '1px 0',
          });
          row.dataset.testid = 'inspector-joint';
          row.dataset.joint = joint.name;

          const nameCell = styled(document.createElement('span'), {
            color: COLOR_LABEL,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          });
          nameCell.textContent = joint.name;
          nameCell.title = joint.name;

          const valueCell = styled(document.createElement('span'), {
            color: COLOR_TEXT_STRONG,
            fontFamily: FONT.mono,
            textAlign: 'right',
          });
          valueCell.textContent = formatFixed(joint.valueRad, JOINT_VALUE_DECIMALS);

          const limitsCell = styled(document.createElement('span'), {
            color: COLOR_MUTED,
            fontFamily: FONT.mono,
            textAlign: 'right',
            whiteSpace: 'nowrap',
          });
          limitsCell.textContent = formatLimits(joint.limits);

          row.appendChild(nameCell);
          row.appendChild(valueCell);
          row.appendChild(limitsCell);
          detailContainer.appendChild(row);
        }
      }
    }
  };

  // ── 선택 적용 (클릭·select() 공통 경로 — 변경 시에만 onSelect 통지) ─
  const applySelection = (requested: string | null): void => {
    const ids = deps.listEntities().map((e) => e.id);
    const resolved = resolveSelection(ids, requested);
    const changed = resolved !== selectedId;
    selectedId = resolved;
    paintListSelection();
    renderDetail();
    if (changed) deps.onSelect?.(resolved);
  };

  // ── 목록 렌더 (id/type/순서가 바뀔 때만 재구축 — refresh의 DOM 변동 최소화) ─
  const renderList = (entities: ReadonlyArray<{ id: string; type: string }>): void => {
    listContainer.replaceChildren();
    rowById.clear();
    listCaption.textContent = `엔티티 (${entities.length})`;
    if (entities.length === 0) {
      listContainer.appendChild(mutedLine('엔티티 없음'));
      return;
    }
    for (const entity of entities) {
      const meta = entityTypeMeta(entity.type);
      const row = styled(document.createElement('div'), {
        display: 'grid',
        gridTemplateColumns: 'auto 1fr auto',
        alignItems: 'baseline',
        columnGap: '6px',
        padding: '1px 6px',
      });
      // hover/선택 시각은 theme 클래스 소유 (.rsw-entity-row[--selected])
      row.classList.add('rsw-entity-row');
      row.dataset.testid = 'inspector-entity';
      row.dataset.entityId = entity.id;

      const icon = styled(document.createElement('span'), { fontSize: '11px' });
      icon.textContent = meta.icon;
      const idCell = styled(document.createElement('span'), {
        color: COLOR_TEXT,
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        whiteSpace: 'nowrap',
      });
      idCell.textContent = entity.id;
      idCell.title = entity.id;
      const typeCell = styled(document.createElement('span'), {
        color: COLOR_MUTED,
        fontSize: '11px',
      });
      typeCell.textContent = meta.label;

      row.appendChild(icon);
      row.appendChild(idCell);
      row.appendChild(typeCell);
      row.addEventListener('click', () => {
        applySelection(nextSelection(selectedId, entity.id));
      });
      listContainer.appendChild(row);
      rowById.set(entity.id, row);
    }
  };

  const refresh = (): void => {
    const entities = deps.listEntities();
    const key = entityListKey(entities);
    if (key !== renderedListKey) {
      renderedListKey = key;
      renderList(entities);
      // 선택 엔티티가 목록에서 사라졌으면 해제 (변경 시 onSelect 통지)
      const resolved = resolveSelection(
        entities.map((e) => e.id),
        selectedId,
      );
      if (resolved !== selectedId) {
        selectedId = resolved;
        deps.onSelect?.(resolved);
      }
      paintListSelection();
    }
    renderDetail();
  };

  host.appendChild(panel);
  refresh(); // 초기 1회 — 이후 주기는 통합자 소유

  return {
    el: panel,
    refresh,
    select: (id): void => {
      applySelection(id);
    },
    dispose: (): void => {
      panel.remove();
    },
  };
}
