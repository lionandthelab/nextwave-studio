// ui/inspector/inspector.ts — 인스펙터 패널 + 씬 아웃라이너 (UX_DESIGN §3.5 (A), UX_AUDIT C-16/C-10/C-5)
//
// 이 모듈은 두 개의 마운트 지점을 낸다:
//   mountSceneOutliner — **엔티티 목록(아웃라이너)** 단독 패널. 검색 + 타입 필터 칩 +
//                        listbox(방향키 탐색). 통합자가 좌 패널로 이관할 수 있다.
//   mountInspector     — 선택 대상 상세(관절 요약). options.showList=false면 목록을
//                        그리지 않는다(목록은 아웃라이너가 소유).
//
// ── 왜 나뉘었나 (UX_AUDIT C-16) ─────────────────────────────────────
// 목록은 *항상 보여야 하는 내비게이션 표면*이고, 속성은 *선택마다 통째로 바뀌며 세로
// 공간을 독점하는 작업 표면*이다. 한 스크롤 컨테이너에 두면 선택을 바꾸려면 스크롤부터
// 해야 한다(Blender/Isaac Sim/Unity/Figma 모두 분리한다). 같은 이유로 **읽기 전용
// Transform 리드아웃을 제거**했다 — 편집 폼(entity-editor)이 같은 값을 편집 가능한 형태로
// 이미 보여주므로 중복이었다. 이 패널의 상세는 편집 폼이 다루지 않는 것(관절 요약)만 남긴다.
//
// READ-ONLY 계약: 이 패널은 물리를 절대 변경하지 않는다 — 값은 InspectorDeps 콜백으로
// 읽기만 한다(뷰에서 물리 pose를 역으로 쓰지 않는다 — CLAUDE.md 불변식 §2.1).
//
// 계층 규칙 (CLAUDE.md §3): ui는 core를 import하지 않는다. 글루(통합자)가 core 파사드
// 위에서 InspectorDeps를 구현해 주입한다. 값 갱신은 통합자가 refresh()를 적당한 주기
// (예: ~150ms 인터벌 또는 engine.onTick 스로틀)로 호출해 일어난다 — 이 모듈은 타이머를
// 소유하지 않는다(주기 결정권은 통합자).
//
// 회전 표시 규약: 내부 진실은 쿼터니언 [x,y,z,w]다 (CLAUDE.md §4 "회전 표현").
// 오일러(deg, XYZ) 변환(quatToEulerDegXYZ)은 **표시/입력 경계 전용**이며 이 파일에서는
// entity-editor가 재사용하는 순수 헬퍼로만 남는다.
//
// 선택 통지: 목록 행 클릭/Enter(같은 행 재활성화 = 해제) 또는 핸들 select() 호출로 선택이
// "바뀔 때만" deps.onSelect가 불린다 — 통합자가 onSelect 안에서 같은 id로 select()를
// 되돌려 불러도 루프가 생기지 않는다(변경 가드).
//
// 접근성 (UX_AUDIT C-5): 엔티티 목록은 role="listbox" + role="option" + roving tabindex다.
// 이전에는 행이 role도 tabIndex도 없는 div라 **키보드로 엔티티를 선택할 방법이 없었고**,
// 선택이 안 되면 트랜스폼 편집·기즈모·관절 패널이 전부 닫혔다(WCAG 2.1.1 Level A).

import {
  BORDER_WIDTH,
  COLOR,
  ICON,
  LAYOUT,
  RADIUS,
  SHADOW,
  SPACE,
  SURFACE,
  TYPE,
  Z_INDEX,
  applyType,
  ensureThemeStyles,
  makeButton,
  makePanelHeader,
  styled,
} from '../theme';
import { icon } from '../icons';
import type { IconName } from '../icons';
import { rovingTabindex } from '../a11y';
import type { Quat, Vec3 } from '../../schema/types';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4, 시각 토큰은 ui/theme.ts) ────

/** 상단 커맨드바 아래로 내려 배치 (단독 마운트 기본값 — 스택 편입 시 통합자가 덮는다) */
const PANEL_TOP_PX = LAYOUT.belowBarTopPx;
const PANEL_RIGHT_PX = 12;
const PANEL_WIDTH_PX = LAYOUT.rightPanelWidthPx;
/** 하단 독(본문 180px + 탭바 + 여백)을 침범하지 않는 최대 높이 여유 */
const PANEL_BOTTOM_CLEARANCE_PX = 220;
/** 본문 스크롤 영역 상한 (접기 트랜지션의 펼침 목표값) */
const BODY_MAX_HEIGHT = '70vh';

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

/**
 * 선택 대상이 없을 때의 **단일** 빈 상태 문구 (UX_AUDIT C-16).
 * 인스펙터와 편집 폼이 각자 다른 문구를 동시에 노출하던 중복을 닫는다 —
 * entity-editor가 이 상수를 import해 쓴다(문구의 단일 진실).
 */
export const EMPTY_SELECTION_HINT = '뷰포트나 아웃라이너에서 대상을 선택하세요';

/** 아웃라이너 타입 필터 칩 (UX_AUDIT C-10 — 라이브러리엔 검색이 있고 목록엔 없던 비대칭) */
export const ENTITY_TYPE_FILTERS: ReadonlyArray<{ type: string; label: string }> = [
  { type: 'robot', label: '로봇' },
  { type: 'object', label: '사물' },
  { type: 'static', label: '정적' },
];

/** 인스턴스별 고유 id 접두사 (label htmlFor 연결이 인스턴스 간 충돌하지 않게) */
let instanceSeq = 0;

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
 * getJoints가 null이면 관절 섹션이 생략된다(로봇이 아닌 엔티티).
 */
export interface InspectorDeps {
  listEntities(): Array<{ id: string; type: string }>;
  /**
   * @deprecated 읽기 전용 Transform 리드아웃은 편집 폼과 중복이라 제거됐다(UX_AUDIT C-16).
   * 시그니처 호환을 위해 남긴다 — 이 패널은 더 이상 호출하지 않는다.
   */
  getPose(
    id: string,
  ): { position: [number, number, number]; rotation: [number, number, number, number] } | null;
  getJoints(id: string): Array<{ name: string; valueRad: number; limits?: [number, number] }> | null;
  /** 선택이 "바뀔 때만" 통지 (활성화·select() 공통, 변경 가드 — 파일 헤더 참조) */
  onSelect?(id: string | null): void;
}

/** mountInspector 선택 옵션 — 기존 2인자 호출과 호환된다 */
export interface InspectorOptions {
  /**
   * 엔티티 목록(아웃라이너)을 이 패널 안에 그릴지. 기본 true.
   * false면 목록은 mountSceneOutliner가 소유한다(좌 패널 이관 — UX_AUDIT C-16).
   */
  showList?: boolean;
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

/** 아웃라이너가 요구하는 최소 표면 — InspectorDeps가 구조적으로 그대로 대입된다 */
export interface SceneOutlinerDeps {
  listEntities(): Array<{ id: string; type: string }>;
  /** 선택이 "바뀔 때만" 통지 */
  onSelect?(id: string | null): void;
}

export interface SceneOutlinerHandle {
  readonly el: HTMLElement;
  /** 현재 선택 id (없으면 null) */
  readonly selectedId: string | null;
  refresh(): void;
  select(id: string | null): void;
  dispose(): void;
}

// ── 순수 헬퍼 (DOM 비의존 — node 단위 테스트 대상) ───────────────────

/**
 * 쿼터니언 [x,y,z,w] → 오일러 각(deg) [x,y,z] — Tait-Bryan **XYZ 순서**
 * (three.js `Euler` 기본 순서와 동일한 행렬 추출식).
 *
 * **표시/입력 경계 전용**이다: 내부 진실은 쿼터니언이며(CLAUDE.md §4) 이 값을 되쓰는
 * 경로는 entity-editor의 eulerDegToQuat(역변환 쌍) 하나뿐이다.
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

/** 목록 행 활성화 의미론: 이미 선택된 행을 다시 활성화하면 해제(null), 아니면 그 행 선택 */
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

/**
 * 아웃라이너 필터 판정 (UX_AUDIT C-10).
 * 타입 칩이 하나도 켜져 있지 않으면 타입 제한 없음, 검색어는 id/type 부분 일치(대소문자 무시).
 */
export function matchesEntityFilter(
  entity: { id: string; type: string },
  query: string,
  types: ReadonlySet<string>,
): boolean {
  if (types.size > 0 && !types.has(entity.type)) return false;
  const q = query.trim().toLowerCase();
  if (q === '') return true;
  return entity.id.toLowerCase().includes(q) || entity.type.toLowerCase().includes(q);
}

/** 엔티티 type → 목록 아이콘(SVG 이름)/한국어 라벨 (미지 type은 원문 그대로 라벨) */
export function entityTypeMeta(type: string): { icon: IconName; label: string } {
  switch (type) {
    case 'robot':
      return { icon: 'robotArm', label: '로봇' };
    case 'object':
      return { icon: 'shapeBox', label: '사물' };
    case 'static':
      return { icon: 'shapePlane', label: '정적' };
    default:
      return { icon: 'layers', label: type };
  }
}

// ── 슬라이더 트랙 채움 (UX_AUDIT C-18 — joint-panel/node-editor 공용) ─

/** 부호를 걸치는 범위인가 (min < 0 < max) — 채움 기준점이 min이 아니라 0이어야 한다 */
export function isBipolarRange(minValue: number, maxValue: number): boolean {
  return minValue < 0 && maxValue > 0;
}

/**
 * `--rsw-range-track`에 써 넣을 배경(linear-gradient) 문자열.
 *
 * 단극(0..max, 예: 그리퍼)은 기존대로 min→thumb 채움이 옳다. 그러나 부호를 걸치는
 * 관절 범위에서 min→thumb 채움은 **의미가 틀리다**: limits ±2.0에서 -0.600이 "35% 진행"
 * 처럼 읽히고 0.000이 "절반 열림"으로 보인다(UX_AUDIT C-18). 이 함수는 부호 있는 범위에서
 * **0에서 썸까지만** 채우고 0 지점에 1px 틱을 그린다(틱 레이어가 채움 레이어 위에 온다).
 */
export function rangeTrackGradient(value: number, minValue: number, maxValue: number): string {
  const span = maxValue - minValue;
  const track = SURFACE.sunken;
  const fillColor = COLOR.accent;
  if (!Number.isFinite(span) || span <= 0) return track;
  const pct = (v: number): number => {
    const ratio = (v - minValue) / span;
    return Math.min(Math.max(ratio, 0), 1) * 100;
  };
  const valuePct = pct(Number.isFinite(value) ? value : minValue);
  if (!isBipolarRange(minValue, maxValue)) {
    return (
      `linear-gradient(to right, ${fillColor} 0 ${valuePct}%, ` +
      `${track} ${valuePct}% 100%)`
    );
  }
  const zeroPct = pct(0);
  const lo = Math.min(zeroPct, valuePct);
  const hi = Math.max(zeroPct, valuePct);
  const tick =
    `linear-gradient(to right, transparent 0 calc(${zeroPct}% - 0.5px), ` +
    `${COLOR.muted} calc(${zeroPct}% - 0.5px) calc(${zeroPct}% + 0.5px), ` +
    `transparent calc(${zeroPct}% + 0.5px) 100%)`;
  const fill =
    `linear-gradient(to right, ${track} 0 ${lo}%, ` +
    `${fillColor} ${lo}% ${hi}%, ${track} ${hi}% 100%)`;
  return `${tick}, ${fill}`;
}

/**
 * range 입력을 디자인 시스템 슬라이더로 만든다(채움 갱신 함수를 돌려준다).
 * 값이 바뀔 때마다 paint(value)를 호출해야 트랙 채움이 값과 동기화된다.
 */
export function attachRangeFill(
  input: HTMLInputElement,
  minValue: number,
  maxValue: number,
): (value: number) => void {
  input.classList.add('ui-range');
  if (isBipolarRange(minValue, maxValue)) input.classList.add('ui-range--bipolar');
  return (value: number): void => {
    input.style.setProperty('--rsw-range-track', rangeTrackGradient(value, minValue, maxValue));
  };
}

// ── 내부 DOM 헬퍼 ───────────────────────────────────────────────────

/** 섹션 캡션 (muted 소제목) */
function caption(text: string): HTMLElement {
  const el = applyType(document.createElement('div'), TYPE.caption);
  styled(el, { color: COLOR.muted, margin: `${SPACE.xl} 0 ${SPACE.xxs} 0` });
  el.textContent = text;
  return el;
}

/** muted 안내/플레이스홀더 줄 */
function mutedLine(text: string): HTMLElement {
  const el = applyType(document.createElement('div'), TYPE.body);
  styled(el, { color: COLOR.muted, padding: `${SPACE.xs} 0`, whiteSpace: 'normal' });
  el.textContent = text;
  return el;
}

// ── 엔티티 목록 뷰 (아웃라이너 본체 — 두 마운트가 공유한다) ──────────

interface EntityListView {
  readonly el: HTMLElement;
  /** 목록 재독 (변경 없으면 DOM 재구축 생략) */
  refresh(): void;
  /** 선택 시각 갱신 */
  setSelected(id: string | null): void;
  dispose(): void;
}

/**
 * 검색 + 타입 필터 칩 + listbox(방향키 탐색)로 이뤄진 엔티티 목록.
 * 선택 상태는 소유자가 갖는다 — 이 뷰는 활성화된 id를 onActivate로 올려보낼 뿐이다.
 */
function createEntityListView(
  deps: SceneOutlinerDeps,
  onActivate: (id: string) => void,
): EntityListView {
  const idPrefix = `rsw-outliner-${(instanceSeq += 1)}`;
  const root = styled(document.createElement('div'), {
    display: 'flex',
    flexDirection: 'column',
    gap: SPACE.sm,
    minWidth: '0',
  });

  // 검색
  const searchRow = styled(document.createElement('div'), {
    display: 'flex',
    alignItems: 'center',
    gap: SPACE.xs,
  });
  const searchIcon = styled(document.createElement('span'), {
    color: COLOR.muted,
    display: 'flex',
    flex: 'none',
  });
  searchIcon.appendChild(icon('search', ICON.sm));
  const searchId = `${idPrefix}-search`;
  const searchInput = document.createElement('input');
  searchInput.type = 'search';
  searchInput.className = 'ui-input';
  searchInput.id = searchId;
  searchInput.placeholder = '엔티티 검색…';
  searchInput.dataset.testid = 'outliner-search';
  styled(searchInput, { flex: '1 1 auto', minWidth: '0', boxSizing: 'border-box' });
  // 아이콘+플레이스홀더만으로는 가시 라벨이 없다 — 이름은 sr-only 라벨이 htmlFor로 준다
  const searchLabel = document.createElement('label');
  searchLabel.className = 'sr-only';
  searchLabel.htmlFor = searchId;
  searchLabel.textContent = '엔티티 검색';
  searchRow.appendChild(searchLabel);
  searchRow.appendChild(searchIcon);
  searchRow.appendChild(searchInput);
  root.appendChild(searchRow);

  // 타입 필터 칩 (다중 선택 — 아무것도 안 켜면 전체)
  const chipRow = styled(document.createElement('div'), {
    display: 'flex',
    flexWrap: 'wrap',
    gap: SPACE.xs,
  });
  chipRow.setAttribute('role', 'group');
  chipRow.setAttribute('aria-label', '엔티티 타입 필터');
  const activeTypes = new Set<string>();
  for (const filter of ENTITY_TYPE_FILTERS) {
    const button = makeButton(
      filter.label,
      `${filter.label}만 보기 (다시 누르면 해제)`,
      `outliner-filter-${filter.type}`,
      'ghost',
    );
    styled(button, { padding: `${SPACE.xxs} ${SPACE.sm}`, minHeight: '24px' });
    button.setAttribute('aria-pressed', 'false');
    button.addEventListener('click', () => {
      if (activeTypes.has(filter.type)) activeTypes.delete(filter.type);
      else activeTypes.add(filter.type);
      const on = activeTypes.has(filter.type);
      button.setAttribute('aria-pressed', String(on));
      button.classList.toggle('ui-btn--active', on);
      render(true);
    });
    chipRow.appendChild(button);
  }
  root.appendChild(chipRow);

  const listCaption = caption('엔티티');
  styled(listCaption, { margin: `${SPACE.xxs} 0 ${SPACE.xxs} 0` });
  root.appendChild(listCaption);

  // listbox (WAI-ARIA APG — 행이 role="option"일 때만 aria-selected가 유효하다)
  const listEl = styled(document.createElement('div'), {
    display: 'flex',
    flexDirection: 'column',
    gap: SPACE.hair,
    minWidth: '0',
  });
  listEl.dataset.testid = 'inspector-entities';
  listEl.setAttribute('role', 'listbox');
  listEl.setAttribute('aria-label', '씬 엔티티');
  root.appendChild(listEl);

  // 빈 상태는 listbox **바깥**에 둔다 — role="listbox"의 자식은 option이어야 한다
  const emptyLine = mutedLine('엔티티 없음');
  emptyLine.dataset.testid = 'outliner-empty';
  emptyLine.style.display = 'none';
  root.appendChild(emptyLine);

  const rowById = new Map<string, HTMLElement>();
  let renderedKey: string | null = null;
  let selectedId: string | null = null;

  const roving = rovingTabindex(listEl, [], {
    orientation: 'vertical',
    onActivate: (el) => {
      const id = el.dataset.entityId;
      if (id !== undefined) onActivate(id);
    },
  });

  const paintSelection = (): void => {
    for (const [id, row] of rowById) {
      // 선택 시각은 클래스 토글 — hover 스타일과 공존한다 (theme .rsw-entity-row--selected)
      row.classList.toggle('rsw-entity-row--selected', id === selectedId);
      row.setAttribute('aria-selected', String(id === selectedId));
    }
  };

  const buildRow = (entity: { id: string; type: string }): HTMLElement => {
    const meta = entityTypeMeta(entity.type);
    const row = styled(document.createElement('div'), {
      display: 'grid',
      gridTemplateColumns: 'auto 1fr auto',
      alignItems: 'center',
      columnGap: SPACE.sm,
      padding: `${SPACE.xxs} ${SPACE.sm}`,
    });
    // hover/선택 시각은 theme 클래스 소유 (.rsw-entity-row[--selected])
    row.classList.add('rsw-entity-row');
    row.dataset.testid = 'inspector-entity';
    row.dataset.entityId = entity.id;
    row.setAttribute('role', 'option');
    row.setAttribute('aria-selected', 'false');

    const iconCell = styled(document.createElement('span'), {
      color: COLOR.muted,
      display: 'flex',
      flex: 'none',
    });
    iconCell.appendChild(icon(meta.icon, ICON.sm));

    const idCell = applyType(document.createElement('span'), TYPE.body);
    styled(idCell, {
      color: COLOR.text,
      overflow: 'hidden',
      textOverflow: 'ellipsis',
      whiteSpace: 'nowrap',
    });
    idCell.textContent = entity.id;
    idCell.title = entity.id;

    const typeCell = applyType(document.createElement('span'), TYPE.micro);
    styled(typeCell, { color: COLOR.muted, flex: 'none' });
    typeCell.textContent = meta.label;

    row.appendChild(iconCell);
    row.appendChild(idCell);
    row.appendChild(typeCell);
    row.addEventListener('click', () => {
      onActivate(entity.id);
    });
    return row;
  };

  /** @param force 필터 변경 등 목록 키가 같아도 다시 그려야 할 때 */
  function render(force = false): void {
    const entities = deps.listEntities();
    const visible = entities.filter((e) =>
      matchesEntityFilter(e, searchInput.value, activeTypes),
    );
    const key = entityListKey(visible);
    listCaption.textContent =
      visible.length === entities.length
        ? `엔티티 (${entities.length})`
        : `엔티티 (${visible.length}/${entities.length})`;
    if (!force && key === renderedKey) return;
    renderedKey = key;

    listEl.replaceChildren();
    rowById.clear();
    if (visible.length === 0) {
      emptyLine.textContent = entities.length === 0 ? '엔티티 없음' : '검색 결과 없음';
      emptyLine.style.display = '';
      listEl.style.display = 'none';
      roving.setItems([]);
      return;
    }
    emptyLine.style.display = 'none';
    listEl.style.display = '';
    const rows: HTMLElement[] = [];
    for (const entity of visible) {
      const row = buildRow(entity);
      listEl.appendChild(row);
      rowById.set(entity.id, row);
      rows.push(row);
    }
    roving.setItems(rows);
    paintSelection();
  }

  searchInput.addEventListener('input', () => {
    render(true);
  });

  render(true);

  return {
    el: root,
    refresh: (): void => {
      render();
    },
    setSelected: (id): void => {
      selectedId = id;
      paintSelection();
    },
    dispose: (): void => {
      roving.dispose();
      root.remove();
    },
  };
}

// ── 씬 아웃라이너 (독립 마운트 — 통합자가 좌 패널로 배치) ────────────

/**
 * 엔티티 목록(아웃라이너)을 host에 마운트한다.
 *
 * 우측 인스펙터에서 분리된 표면이다(UX_AUDIT C-16) — 통합자가 좌 패널(라이브러리 아래)에
 * 두면 "선택을 바꾸려면 스크롤부터 해야 한다"가 사라진다. 배치는 통합자가 el로 결정한다.
 */
export function mountSceneOutliner(
  host: HTMLElement,
  deps: SceneOutlinerDeps,
): SceneOutlinerHandle {
  ensureThemeStyles();
  const panel = styled(document.createElement('div'), {
    width: '100%',
    boxSizing: 'border-box',
    display: 'flex',
    flexDirection: 'column',
    minHeight: '0',
    background: COLOR.bgPanel,
    border: `${BORDER_WIDTH.hair} solid ${COLOR.border}`,
    borderRadius: RADIUS.md,
    color: COLOR.text,
    userSelect: 'none',
  });
  applyType(panel, TYPE.body);
  panel.dataset.testid = 'scene-outliner';

  const header = makePanelHeader('씬 아웃라이너', {
    collapsible: true,
    testId: 'scene-outliner-header',
  });
  panel.appendChild(header.el);

  const body = styled(document.createElement('div'), {
    overflowY: 'auto',
    minHeight: '0',
    maxHeight: BODY_MAX_HEIGHT,
    opacity: '1',
    padding: `${SPACE.md} ${SPACE.lg} ${SPACE.lg} ${SPACE.lg}`,
  });
  body.classList.add('ui-scroll', 'ui-collapsible');
  panel.appendChild(body);

  let selectedId: string | null = null;

  const applySelection = (requested: string | null): void => {
    const ids = deps.listEntities().map((e) => e.id);
    const resolved = resolveSelection(ids, requested);
    const changed = resolved !== selectedId;
    selectedId = resolved;
    view.setSelected(selectedId);
    if (changed) deps.onSelect?.(resolved);
  };

  const view = createEntityListView(deps, (id) => {
    applySelection(nextSelection(selectedId, id));
  });
  body.appendChild(view.el);

  header.onToggle((collapsed) => {
    body.style.maxHeight = collapsed ? '0' : BODY_MAX_HEIGHT;
    body.style.opacity = collapsed ? '0' : '1';
    body.style.paddingTop = collapsed ? '0' : SPACE.md;
    body.style.paddingBottom = collapsed ? '0' : SPACE.lg;
    body.style.overflowY = collapsed ? 'hidden' : 'auto';
  });

  host.appendChild(panel);

  return {
    el: panel,
    get selectedId(): string | null {
      return selectedId;
    },
    refresh: (): void => {
      view.refresh();
      // 선택 엔티티가 목록에서 사라졌으면 해제 (변경 시 onSelect 통지)
      applySelection(selectedId);
    },
    select: (id): void => {
      applySelection(id);
    },
    dispose: (): void => {
      view.dispose();
      panel.remove();
    },
  };
}

// ── 인스펙터 마운트 ─────────────────────────────────────────────────

/**
 * 인스펙터를 host에 마운트한다 (우측 절대 배치 — 상단 재생 바 아래).
 * 패널 내부 포인터/휠은 stopPropagation으로 흡수한다 — 뷰포트 orbit 컨트롤로
 * 새지 않게 하는 기존 패널 규약(joint-panel/dock)과 동일.
 */
export function mountInspector(
  host: HTMLElement,
  deps: InspectorDeps,
  options: InspectorOptions = {},
): InspectorHandle {
  ensureThemeStyles();
  const showList = options.showList ?? true;
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
    border: `${BORDER_WIDTH.hair} solid ${COLOR.border}`,
    borderRadius: RADIUS.md,
    boxShadow: SHADOW.panel,
    color: COLOR.text,
    boxSizing: 'border-box',
    userSelect: 'none',
  });
  applyType(panel, TYPE.body);
  panel.dataset.testid = 'inspector';
  for (const type of ['pointerdown', 'pointermove', 'pointerup', 'wheel', 'contextmenu']) {
    panel.addEventListener(type, (e) => {
      e.stopPropagation();
    });
  }

  // 헤더 — 셰브론 회전 규약은 makePanelHeader가 소유한다(수제 ▾/▴ 제거, UX_AUDIT C-18)
  const header = makePanelHeader('인스펙터', { collapsible: true, testId: 'inspector-header' });
  // 구 testId 유지 (측정 스크립트/문서 참조 호환)
  if (header.collapseButton !== null) header.collapseButton.dataset.testid = 'inspector-collapse';
  panel.appendChild(header.el);

  // 본문 (스크롤 영역): (선택적) 엔티티 목록 + 상세 — 접기는 max-height 트랜지션
  const body = styled(document.createElement('div'), {
    overflowY: 'auto',
    minHeight: '0',
    maxHeight: BODY_MAX_HEIGHT,
    opacity: '1',
    padding: `${SPACE.md} ${SPACE.lg} ${SPACE.lg} ${SPACE.lg}`,
  });
  body.classList.add('ui-scroll', 'ui-collapsible');
  panel.appendChild(body);

  const detailContainer = document.createElement('div');
  detailContainer.dataset.testid = 'inspector-detail';

  // ── 상태 (뷰 상태는 ui 소유 — 시뮬 진실은 deps 콜백 너머의 core) ──
  let selectedId: string | null = null;

  const listView: EntityListView | null = showList
    ? createEntityListView(deps, (id) => {
        applySelection(nextSelection(selectedId, id));
      })
    : null;
  if (listView !== null) body.appendChild(listView.el);
  body.appendChild(detailContainer);

  header.onToggle((collapsed) => {
    body.style.maxHeight = collapsed ? '0' : BODY_MAX_HEIGHT;
    body.style.opacity = collapsed ? '0' : '1';
    body.style.paddingTop = collapsed ? '0' : SPACE.md;
    body.style.paddingBottom = collapsed ? '0' : SPACE.lg;
    body.style.overflowY = collapsed ? 'hidden' : 'auto';
  });

  // ── 상세 렌더 (편집 폼이 다루지 않는 것만 — Transform 중복 제거, C-16) ─
  const renderDetail = (): void => {
    detailContainer.replaceChildren();
    if (selectedId === null) {
      // 빈 상태 문구는 편집 폼이 하나만 낸다 (EMPTY_SELECTION_HINT — 중복 제거)
      detailContainer.style.display = 'none';
      return;
    }
    detailContainer.style.display = '';

    const entity = deps.listEntities().find((e) => e.id === selectedId);
    const meta = entityTypeMeta(entity?.type ?? '');

    // 선택 대상 헤드라인 (아이콘 + id + 타입)
    const idRow = styled(document.createElement('div'), {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      gap: SPACE.md,
      padding: `${SPACE.xs} 0`,
    });
    const idText = applyType(document.createElement('span'), TYPE.bodyStrong);
    styled(idText, {
      color: COLOR.textStrong,
      display: 'flex',
      alignItems: 'center',
      gap: SPACE.xs,
      overflow: 'hidden',
      minWidth: '0',
    });
    const idIcon = styled(document.createElement('span'), {
      display: 'flex',
      flex: 'none',
      color: COLOR.muted,
    });
    idIcon.appendChild(icon(meta.icon, ICON.md));
    const idLabel = styled(document.createElement('span'), {
      overflow: 'hidden',
      textOverflow: 'ellipsis',
      whiteSpace: 'nowrap',
    });
    idLabel.textContent = selectedId;
    idText.appendChild(idIcon);
    idText.appendChild(idLabel);
    const typeBadge = applyType(document.createElement('span'), TYPE.micro);
    styled(typeBadge, { color: COLOR.label, flex: 'none' });
    typeBadge.textContent = meta.label;
    idRow.appendChild(idText);
    idRow.appendChild(typeBadge);
    detailContainer.appendChild(idRow);

    // 관절 (로봇 엔티티만 — getJoints가 null이면 섹션 자체를 생략)
    const joints = deps.getJoints(selectedId);
    if (joints !== null) {
      detailContainer.appendChild(caption(`관절 (${joints.length})`));
      if (joints.length === 0) {
        detailContainer.appendChild(mutedLine('없음'));
      } else {
        const gridColumns = `1fr ${JOINT_VALUE_COL_PX}px ${JOINT_LIMITS_COL_PX}px`;
        const head = applyType(document.createElement('div'), TYPE.micro);
        styled(head, {
          display: 'grid',
          gridTemplateColumns: gridColumns,
          columnGap: SPACE.sm,
          color: COLOR.muted,
          borderBottom: `${BORDER_WIDTH.hair} solid ${COLOR.borderSoft}`,
          padding: `0 0 ${SPACE.xxs} 0`,
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
            columnGap: SPACE.sm,
            padding: `${SPACE.hair} 0`,
          });
          row.dataset.testid = 'inspector-joint';
          row.dataset.joint = joint.name;

          const nameCell = applyType(document.createElement('span'), TYPE.body);
          styled(nameCell, {
            color: COLOR.label,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          });
          nameCell.textContent = joint.name;
          nameCell.title = joint.name;

          const valueCell = applyType(document.createElement('span'), TYPE.monoReadout);
          styled(valueCell, { color: COLOR.textStrong, textAlign: 'right' });
          valueCell.textContent = formatFixed(joint.valueRad, JOINT_VALUE_DECIMALS);

          const limitsCell = applyType(document.createElement('span'), TYPE.monoMicro);
          styled(limitsCell, {
            color: COLOR.muted,
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

  // ── 선택 적용 (활성화·select() 공통 경로 — 변경 시에만 onSelect 통지) ─
  function applySelection(requested: string | null): void {
    const ids = deps.listEntities().map((e) => e.id);
    const resolved = resolveSelection(ids, requested);
    const changed = resolved !== selectedId;
    selectedId = resolved;
    listView?.setSelected(selectedId);
    renderDetail();
    if (changed) deps.onSelect?.(resolved);
  }

  const refresh = (): void => {
    listView?.refresh();
    // 선택 엔티티가 목록에서 사라졌으면 해제 (변경 시 onSelect 통지)
    applySelection(selectedId);
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
      listView?.dispose();
      panel.remove();
    },
  };
}
