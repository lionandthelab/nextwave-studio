// ui/inspector/entity-editor.ts — 편집형 인스펙터 폼 (UX_DESIGN §3.5 (A) 오브젝트, ROADMAP Phase 7)
//
// 선택된 엔티티의 이름/ID · Transform · Dimensions · Physics를 편집하는 폼을 host에
// 마운트한다. 모든 커밋은 **명시적**이다(Enter/blur/select·checkbox change/스크럽 종료)
// — 키 입력마다 물리를 갱신하지 않는다(물리 churn 방지, 요구 §3).
//
// 계층 규칙 (CLAUDE.md §3): ui는 core를 import하지 않는다. 통합자(main.ts)가
// SceneEditor(core/scene-edit-types.ts) 위에서 EntityEditorDeps를 구현해 주입한다.
// 편집의 진실은 SceneEditor의 spec이다 — 이 폼은 커밋 후 클램프된 값 되쓰기(시각
// 피드백)만 하고, 실제 재동기화는 통합자가 SceneEditor.onChange에서 refreshFrom(spec)을
// 호출해 수행한다(뷰에서 물리 pose를 역으로 쓰지 않는다 — 불변식 §2.1).
//
// 회전 규약 (CLAUDE.md §4): 내부 진실은 쿼터니언 [x,y,z,w]. 오일러(deg, XYZ)는 UI 입력
// 경계 전용이며 커밋 시 eulerDegToQuat로 즉시 변환한다. 표시 변환은 inspector.ts의
// quatToEulerDegXYZ(동일 XYZ 규약의 역변환 쌍 — 테스트로 고정)를 재사용한다.
//
// 치수 규약 (UX §3.5): UI는 **전체 치수(m)**를 보여주고 내부 halfExtents(/2)로 변환해
// 커밋한다. 모든 치수는 DIMENSION_MIN_M(0.005m)로 하한 클램프.
//
// 순수 헬퍼(eulerDegToQuat · clamp류 · shape/physics 폼 상태 변환 · scrubValue)는
// DOM 없이 export되어 node에서 단위 테스트된다(entity-editor.test.ts). DOM은 얇다.

import {
  COLOR,
  FONT,
  RADIUS,
  SHADOW,
  SPACE,
  ensureThemeStyles,
  makeButton,
  styled,
} from '../theme';
import { formatFixed, quatToEulerDegXYZ } from './inspector';
import type {
  BodyType,
  ColliderGroup,
  ColliderShape,
  EntitySpec,
  PhysicsSpec,
  Quat,
  Transform,
  Vec3,
} from '../../schema/types';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4) ─────────────────────────────

/** 치수(전체 치수·반지름·높이) 하한 클램프 (m) — 요구 §2 Dimensions */
export const DIMENSION_MIN_M = 0.005;
/** friction/restitution/density 클램프 범위 — 요구 §2 Physics */
export const PHYSICS_PARAM_MIN = 0;
export const PHYSICS_PARAM_MAX = 2;

const DEG_TO_RAD = Math.PI / 180;
const IDENTITY_QUAT: Quat = [0, 0, 0, 1];

/** position 입력 step (m) — 요구 §2 Transform */
const POSITION_STEP_M = 0.01;
/** rotation 입력 step (deg) */
const ROTATION_STEP_DEG = 1;
/** 치수 입력 step (m) */
const DIMENSION_STEP_M = 0.005;
/** physics 파라미터 입력 step */
const PHYSICS_STEP = 0.01;

/** 입력값 표시 소수 자릿수 (populate/커밋 되쓰기 공통) */
const POSITION_DECIMALS_EDIT = 3;
const ROTATION_DECIMALS_EDIT = 2;
const DIMENSION_DECIMALS_EDIT = 3;
const PHYSICS_DECIMALS_EDIT = 2;

/** 라벨 드래그 스크럽: 1px당 증분 (단위는 필드별) */
const SCRUB_POSITION_PER_PX = 0.005;
const SCRUB_ROTATION_PER_PX_DEG = 0.5;
const SCRUB_DIMENSION_PER_PX = 0.002;
const SCRUB_PHYSICS_PER_PX = 0.005;
/** 스크럽 시작 판정 데드존 (px) — 단순 클릭과 구분 */
const SCRUB_DEADZONE_PX = 3;

/** collider 기본값 (DATA_MODEL §2 — 폼 초기값/파싱 폴백) */
const DEFAULT_FRICTION = 0.5;
const DEFAULT_RESTITUTION = 0;
const DEFAULT_DENSITY = 1;
const DEFAULT_GROUP: ColliderGroup = 'OBJECT';
const DEFAULT_COLLIDES_WITH: readonly ColliderGroup[] = ['ENV', 'ROBOT', 'OBJECT'];

/** 폼에 노출하는 bodyType 후보 (요구 §2 — kinematicVelocity는 현재값일 때만 추가) */
const BODY_TYPE_CHOICES: readonly BodyType[] = ['dynamic', 'fixed', 'kinematicPosition'];
/** 폼에 노출하는 그룹 후보 (CLAUDE.md §5 — DEBUG는 현재값일 때만 추가) */
const GROUP_CHOICES: readonly ColliderGroup[] = ['ENV', 'ROBOT', 'OBJECT', 'SENSOR_ZONE'];

// ── 공개 타입 ───────────────────────────────────────────────────────

/**
 * 통합자(main.ts)가 SceneEditor 위에서 구현해 주입하는 편집 표면.
 * 이 폼은 spec 스냅샷을 읽고, 변경은 반드시 이 콜백으로만 요청한다.
 */
export interface EntityEditorDeps {
  /** 현재 spec 스냅샷 (없으면 null) */
  getEntity(id: string): EntitySpec | null;
  isRobot(id: string): boolean;
  /** → SceneEditor.updateTransform (통합자 바인딩) */
  updateTransform(id: string, transform: Transform): void;
  /** → SceneEditor.updateDimensions — 메시+collider 동시 재생성 */
  updateDimensions(id: string, shape: ColliderShape): void;
  /** → SceneEditor.updatePhysics — 바디+collider 재생성 */
  updatePhysics(id: string, physics: PhysicsSpec): void;
  /** null = 성공, string = 한국어 오류 메시지(중복 등 — 인라인 표시) */
  renameEntity(id: string, newId: string): string | null;
}

export interface EntityEditorHandle {
  /** 폼 루트 — 통합자가 우측 스택 등으로 재배치할 때 사용 */
  readonly el: HTMLElement;
  /** 선택 변경: id의 spec으로 폼 재구축. null(또는 미존재 id) = 플레이스홀더 */
  showFor(id: string | null): void;
  /** 값 재동기화 (통합자가 SceneEditor.onChange·선택 변경 시 호출). 편집 중 필드는 건너뛴다 */
  refreshFrom(spec: EntitySpec): void;
  dispose(): void;
}

// ── 순수 헬퍼 (DOM 비의존 — node 단위 테스트 대상) ───────────────────

/**
 * 오일러 각(deg, Tait-Bryan **XYZ 순서**) → 쿼터니언 [x,y,z,w].
 * inspector.ts의 quatToEulerDegXYZ와 동일 규약의 역변환 쌍이다(q = qx ⊗ qy ⊗ qz,
 * three.js Euler 'XYZ'와 동일 폐형식). 라운드트립은 테스트로 고정한다.
 */
export function eulerDegToQuat(eulerDeg: readonly [number, number, number]): Quat {
  const halfX = (eulerDeg[0] * DEG_TO_RAD) / 2;
  const halfY = (eulerDeg[1] * DEG_TO_RAD) / 2;
  const halfZ = (eulerDeg[2] * DEG_TO_RAD) / 2;
  const sx = Math.sin(halfX);
  const cx = Math.cos(halfX);
  const sy = Math.sin(halfY);
  const cy = Math.cos(halfY);
  const sz = Math.sin(halfZ);
  const cz = Math.cos(halfZ);
  return [
    sx * cy * cz + cx * sy * sz,
    cx * sy * cz - sx * cy * sz,
    cx * cy * sz + sx * sy * cz,
    cx * cy * cz - sx * sy * sz,
  ];
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

/** 치수 하한 클램프 — 비유한(NaN/∞) 입력은 최소값으로 방어 */
export function clampDimensionM(value: number): number {
  if (!Number.isFinite(value)) return DIMENSION_MIN_M;
  return Math.max(value, DIMENSION_MIN_M);
}

/** friction/restitution/density 0..2 클램프 — 비유한 입력은 0으로 방어 */
export function clampPhysicsParam(value: number): number {
  if (!Number.isFinite(value)) return PHYSICS_PARAM_MIN;
  return clamp(value, PHYSICS_PARAM_MIN, PHYSICS_PARAM_MAX);
}

/** 입력 문자열 → 유한 수. 빈 문자열/NaN/∞는 null (호출자가 폴백 결정) */
export function parseNumeric(text: string): number | null {
  const trimmed = text.trim();
  if (trimmed === '') return null;
  const value = Number(trimmed);
  return Number.isFinite(value) ? value : null;
}

/** 라벨 드래그 스크럽 값 계산: 시작값 + 수평 이동량 × 증분, [min, max] 클램프 */
export function scrubValue(
  startValue: number,
  deltaPx: number,
  stepPerPx: number,
  min?: number,
  max?: number,
): number {
  let value = startValue + deltaPx * stepPerPx;
  if (min !== undefined) value = Math.max(value, min);
  if (max !== undefined) value = Math.min(value, max);
  return value;
}

/** 이름 커밋 전 로컬 검증 — deps 호출 전 빈 이름 차단 (그 외 검증은 deps가 소유) */
export function localNameError(name: string): string | null {
  return name.trim() === '' ? '이름을 입력하세요' : null;
}

/**
 * 프리미티브 치수 폼 상태 — UI는 전체 치수(m), 내부 저장은 halfExtents/halfHeight.
 * heightM: cylinder/capsule 모두 원통부 **전체** 높이(halfHeight × 2)다.
 */
export type ShapeFormState =
  | { kind: 'box'; widthM: number; heightM: number; lengthM: number }
  | { kind: 'sphere'; radiusM: number }
  | { kind: 'cylinder'; radiusM: number; heightM: number }
  | { kind: 'capsule'; radiusM: number; heightM: number };

/** ColliderShape → 폼 상태 (전체 치수). 프리미티브 4종 외(convexHull 등)는 null */
export function shapeToFormState(shape: ColliderShape): ShapeFormState | null {
  switch (shape.kind) {
    case 'box': {
      const [halfX, halfY, halfZ] = shape.halfExtents;
      return { kind: 'box', widthM: halfX * 2, heightM: halfY * 2, lengthM: halfZ * 2 };
    }
    case 'sphere':
      return { kind: 'sphere', radiusM: shape.radius };
    case 'cylinder':
      return { kind: 'cylinder', radiusM: shape.radius, heightM: shape.halfHeight * 2 };
    case 'capsule':
      return { kind: 'capsule', radiusM: shape.radius, heightM: shape.halfHeight * 2 };
    default:
      return null;
  }
}

/** 폼 상태(전체 치수) → ColliderShape — DIMENSION_MIN_M 하한 클램프 후 /2 변환 */
export function formStateToShape(form: ShapeFormState): ColliderShape {
  switch (form.kind) {
    case 'box': {
      const width = clampDimensionM(form.widthM);
      const height = clampDimensionM(form.heightM);
      const length = clampDimensionM(form.lengthM);
      return { kind: 'box', halfExtents: [width / 2, height / 2, length / 2] };
    }
    case 'sphere':
      return { kind: 'sphere', radius: clampDimensionM(form.radiusM) };
    case 'cylinder':
      return {
        kind: 'cylinder',
        radius: clampDimensionM(form.radiusM),
        halfHeight: clampDimensionM(form.heightM) / 2,
      };
    case 'capsule':
      return {
        kind: 'capsule',
        radius: clampDimensionM(form.radiusM),
        halfHeight: clampDimensionM(form.heightM) / 2,
      };
  }
}

/**
 * 엔티티의 편집 대상 프리미티브 shape.
 * collider(물리가 진실)를 우선하고, 없거나 프리미티브가 아니면 visual.primitive 폴백.
 */
export function primitiveShapeOf(spec: EntitySpec): ColliderShape | null {
  const colliderShape = spec.physics?.colliders[0]?.shape;
  if (colliderShape !== undefined && shapeToFormState(colliderShape) !== null) {
    return colliderShape;
  }
  const visualShape = spec.visual.kind === 'primitive' ? spec.visual.primitive : undefined;
  if (visualShape !== undefined && shapeToFormState(visualShape) !== null) {
    return visualShape;
  }
  return null;
}

/** primitiveShapeOf + shapeToFormState 합성 (폼 초기값/populate용) */
export function primitiveFormStateOf(spec: EntitySpec): ShapeFormState | null {
  const shape = primitiveShapeOf(spec);
  return shape === null ? null : shapeToFormState(shape);
}

/** Physics 폼 상태 — 첫 collider에서 유도 (커밋 시 모든 collider에 동일 적용) */
export interface PhysicsFormState {
  bodyType: BodyType;
  friction: number;
  restitution: number;
  density: number;
  group: ColliderGroup;
  collidesWith: ColliderGroup[];
  isSensor: boolean;
  emitEvents: boolean;
  ccd: boolean;
}

/** PhysicsSpec → 폼 상태. collider가 없으면 DATA_MODEL 기본값 */
export function physicsFormStateOf(physics: PhysicsSpec): PhysicsFormState {
  const collider = physics.colliders[0];
  return {
    bodyType: physics.bodyType,
    friction: collider?.friction ?? DEFAULT_FRICTION,
    restitution: collider?.restitution ?? DEFAULT_RESTITUTION,
    density: collider?.density ?? DEFAULT_DENSITY,
    group: collider?.group ?? DEFAULT_GROUP,
    collidesWith: collider !== undefined ? [...collider.collidesWith] : [...DEFAULT_COLLIDES_WITH],
    isSensor: collider?.isSensor ?? false,
    emitEvents: collider?.emitEvents ?? false,
    ccd: collider?.ccd ?? false,
  };
}

/**
 * 폼 상태를 원본 PhysicsSpec에 적용해 **전체 PhysicsSpec을 재구성**한다.
 * 미지 필드는 스프레드로 보존(damping/gravityScale/offset/shape 등 — 요구 §2 Physics).
 * friction/restitution/density는 0..2 클램프. 폼 값은 모든 collider에 동일 적용된다.
 */
export function applyPhysicsFormState(original: PhysicsSpec, form: PhysicsFormState): PhysicsSpec {
  const friction = clampPhysicsParam(form.friction);
  const restitution = clampPhysicsParam(form.restitution);
  const density = clampPhysicsParam(form.density);
  return {
    ...original,
    bodyType: form.bodyType,
    colliders: original.colliders.map((collider) => ({
      ...collider,
      friction,
      restitution,
      density,
      group: form.group,
      collidesWith: [...form.collidesWith],
      isSensor: form.isSensor,
      emitEvents: form.emitEvents,
      ccd: form.ccd,
    })),
  };
}

/** bodyType 셀렉트 후보 — 현재값이 기본 후보 밖(kinematicVelocity)이면 뒤에 추가(데이터 보존) */
export function bodyTypeChoices(current: BodyType): BodyType[] {
  return BODY_TYPE_CHOICES.includes(current)
    ? [...BODY_TYPE_CHOICES]
    : [...BODY_TYPE_CHOICES, current];
}

/** 그룹 셀렉트 후보 — 현재값이 기본 후보 밖(DEBUG)이면 뒤에 추가(데이터 보존) */
export function groupChoices(current: ColliderGroup): ColliderGroup[] {
  return GROUP_CHOICES.includes(current) ? [...GROUP_CHOICES] : [...GROUP_CHOICES, current];
}

/** collidesWith 체크박스 후보 — 기본 4종 + 현재값의 추가 그룹(중복 제거, 드롭 방지) */
export function collidesWithChoices(current: readonly ColliderGroup[]): ColliderGroup[] {
  const extras = current.filter((group, index) => {
    return !GROUP_CHOICES.includes(group) && current.indexOf(group) === index;
  });
  return [...GROUP_CHOICES, ...extras];
}

/** bodyType 옵션 한국어 병기 라벨 */
export function bodyTypeLabel(bodyType: BodyType): string {
  switch (bodyType) {
    case 'dynamic':
      return 'dynamic — 동적';
    case 'fixed':
      return 'fixed — 고정';
    case 'kinematicPosition':
      return 'kinematicPosition — 위치 구동';
    case 'kinematicVelocity':
      return 'kinematicVelocity — 속도 구동';
  }
}

/** 충돌 그룹 옵션 한국어 병기 라벨 (CLAUDE.md §5) */
export function groupLabel(group: ColliderGroup): string {
  switch (group) {
    case 'ENV':
      return 'ENV — 환경';
    case 'ROBOT':
      return 'ROBOT — 로봇';
    case 'OBJECT':
      return 'OBJECT — 사물';
    case 'SENSOR_ZONE':
      return 'SENSOR_ZONE — 센서 존';
    case 'DEBUG':
      return 'DEBUG — 디버그';
  }
}

// ── 스크럽 (재사용 DOM 헬퍼 — 순수 계산은 scrubValue) ────────────────

export interface ScrubOptions {
  /** 1px당 증분 */
  stepPerPx: number;
  min?: number;
  max?: number;
  /** 드래그 중 입력창 표시 소수 자릿수 */
  decimals: number;
  /** 드래그 종료(pointerup/cancel) 시 1회 호출 — 명시적 커밋 */
  onCommit(): void;
  /** 스크럽 시작/종료 통지 (populate 건너뛰기 가드용) */
  onActiveChange?(active: boolean): void;
}

/**
 * 라벨에 좌우 드래그 스크럽을 붙인다: pointerdown 후 수평 드래그로 input 값을 조정하고,
 * 드래그 종료 시에만 onCommit 1회(명시적 커밋 — 드래그 중에는 표시만 갱신).
 * 포인터 캡처를 사용하므로 document 레벨 리스너가 없다(dispose 불요).
 */
export function attachScrub(
  label: HTMLElement,
  input: HTMLInputElement,
  options: ScrubOptions,
): void {
  styled(label, { cursor: 'ew-resize', touchAction: 'none' });
  let active = false;
  let moved = false;
  let startX = 0;
  let startValue = 0;

  label.addEventListener('pointerdown', (e: PointerEvent) => {
    active = true;
    moved = false;
    startX = e.clientX;
    startValue = parseNumeric(input.value) ?? 0;
    label.setPointerCapture(e.pointerId);
    options.onActiveChange?.(true);
    e.preventDefault();
  });
  label.addEventListener('pointermove', (e: PointerEvent) => {
    if (!active) return;
    const deltaPx = e.clientX - startX;
    if (!moved && Math.abs(deltaPx) < SCRUB_DEADZONE_PX) return;
    moved = true;
    const value = scrubValue(startValue, deltaPx, options.stepPerPx, options.min, options.max);
    input.value = formatFixed(value, options.decimals);
  });
  const finish = (e: PointerEvent): void => {
    if (!active) return;
    active = false;
    if (label.hasPointerCapture(e.pointerId)) label.releasePointerCapture(e.pointerId);
    options.onActiveChange?.(false);
    if (moved) options.onCommit();
  };
  label.addEventListener('pointerup', finish);
  label.addEventListener('pointercancel', finish);
}

// ── 내부 DOM 헬퍼 ───────────────────────────────────────────────────

function subCaption(text: string): HTMLElement {
  const el = styled(document.createElement('div'), {
    color: COLOR.muted,
    fontSize: '11px',
    margin: '6px 0 2px 0',
  });
  el.textContent = text;
  return el;
}

function mutedLine(text: string): HTMLElement {
  const el = styled(document.createElement('div'), {
    color: COLOR.muted,
    padding: '4px 0',
  });
  el.textContent = text;
  return el;
}

function tripleRow(): HTMLElement {
  return styled(document.createElement('div'), {
    display: 'grid',
    gridTemplateColumns: 'repeat(3, minmax(0, 1fr))',
    gap: '4px 6px',
  });
}

interface NumberCellOpts {
  label: string;
  testId: string;
  step: number;
  min?: number;
  max?: number;
  decimals: number;
  scrubPerPx: number;
  onCommit(): void;
  onScrubActive(active: boolean, input: HTMLInputElement): void;
}

/** 라벨(스크럽 가능) + number 입력 셀. 커밋은 change(Enter/blur/스피너)와 스크럽 종료만 */
function numberCell(opts: NumberCellOpts): { cell: HTMLElement; input: HTMLInputElement } {
  const cell = styled(document.createElement('div'), {
    display: 'flex',
    flexDirection: 'column',
    gap: '2px',
    minWidth: '0',
  });
  const label = styled(document.createElement('span'), {
    color: COLOR.label,
    fontSize: '11px',
    userSelect: 'none',
  });
  label.textContent = opts.label;
  label.title = `${opts.label} — 라벨을 좌우로 드래그해 조정`;

  const input = document.createElement('input');
  input.type = 'number';
  input.className = 'ui-input';
  input.step = String(opts.step);
  if (opts.min !== undefined) input.min = String(opts.min);
  if (opts.max !== undefined) input.max = String(opts.max);
  styled(input, {
    width: '100%',
    boxSizing: 'border-box',
    fontFamily: FONT.mono,
    textAlign: 'right',
  });
  input.dataset.testid = opts.testId;
  input.setAttribute('aria-label', opts.label);
  input.addEventListener('change', () => {
    opts.onCommit();
  });
  input.addEventListener('keydown', (e: KeyboardEvent) => {
    if (e.key === 'Enter') input.blur();
  });
  attachScrub(label, input, {
    stepPerPx: opts.scrubPerPx,
    min: opts.min,
    max: opts.max,
    decimals: opts.decimals,
    onCommit: opts.onCommit,
    onActiveChange: (active) => {
      opts.onScrubActive(active, input);
    },
  });

  cell.appendChild(label);
  cell.appendChild(input);
  return { cell, input };
}

function addOption(select: HTMLSelectElement, value: string, label: string): void {
  const option = document.createElement('option');
  option.value = value;
  option.textContent = label;
  select.appendChild(option);
}

function ensureOption(select: HTMLSelectElement, value: string, label: string): void {
  for (const option of select.options) {
    if (option.value === value) return;
  }
  addOption(select, value, label);
}

// ── 마운트 ──────────────────────────────────────────────────────────

interface FieldBinding {
  /** populate 건너뛰기 판정 대상 (포커스/스크럽 중이면 값 갱신 생략) */
  control: HTMLElement;
  populate(spec: EntitySpec): void;
}

/**
 * 엔티티 편집 폼을 host에 마운트한다. 배치(단독/우측 스택 편입)는 통합자가 el로 결정한다.
 * 폼 내부 포인터/휠/키 이벤트는 stopPropagation으로 흡수한다 — 뷰포트 orbit·앱 전역
 * 단축키로 새지 않게 하는 기존 패널 규약(inspector/dock)과 동일.
 */
export function mountEntityEditor(host: HTMLElement, deps: EntityEditorDeps): EntityEditorHandle {
  ensureThemeStyles();
  const root = styled(document.createElement('div'), {
    width: '100%',
    boxSizing: 'border-box',
    background: COLOR.bgPanel,
    border: `1px solid ${COLOR.border}`,
    borderRadius: RADIUS.md,
    boxShadow: SHADOW.panel,
    color: COLOR.text,
    fontFamily: FONT.ui,
    fontSize: '12px',
    lineHeight: '1.5',
    userSelect: 'none',
  });
  root.dataset.testid = 'entity-editor';
  for (const type of ['pointerdown', 'pointermove', 'pointerup', 'wheel', 'contextmenu', 'keydown']) {
    root.addEventListener(type, (e) => {
      e.stopPropagation();
    });
  }

  const header = styled(document.createElement('div'), {
    display: 'flex',
    alignItems: 'center',
    gap: SPACE.md,
    padding: `${SPACE.sm} 10px`,
    borderBottom: `1px solid ${COLOR.borderSoft}`,
  });
  const title = styled(document.createElement('strong'), {
    color: COLOR.textStrong,
    fontSize: '13px',
  });
  title.textContent = '엔티티 편집';
  header.appendChild(title);
  root.appendChild(header);

  const content = styled(document.createElement('div'), {
    padding: '6px 10px 10px 10px',
    maxHeight: '65vh',
    overflowY: 'auto',
    minHeight: '0',
  });
  content.classList.add('ui-scroll');
  root.appendChild(content);

  // ── 상태 (뷰 상태는 ui 소유 — 편집 진실은 deps 너머 SceneEditor의 spec) ─
  let currentId: string | null = null;
  let formSignature: string | null = null;
  /** 스크럽 드래그 중인 입력 (populate 건너뛰기 가드) */
  let scrubControl: HTMLInputElement | null = null;
  const fieldBindings: FieldBinding[] = [];

  const specOf = (): EntitySpec | null => (currentId === null ? null : deps.getEntity(currentId));

  const onScrubActive = (active: boolean, input: HTMLInputElement): void => {
    scrubControl = active ? input : null;
  };

  const isBusy = (control: HTMLElement): boolean => {
    const active = document.activeElement;
    if (active instanceof HTMLElement && control.contains(active)) return true;
    return scrubControl !== null && control.contains(scrubControl);
  };

  const populateAll = (spec: EntitySpec): void => {
    for (const binding of fieldBindings) {
      if (isBusy(binding.control)) continue;
      binding.populate(spec);
    }
  };

  const bindNumber = (
    input: HTMLInputElement,
    get: (spec: EntitySpec) => number | null,
    decimals: number,
  ): void => {
    fieldBindings.push({
      control: input,
      populate: (spec): void => {
        const value = get(spec);
        if (value !== null) input.value = formatFixed(value, decimals);
      },
    });
  };

  // ── 폼 시그니처: 섹션 구성이 바뀌는 조건 (바뀌면 재구축, 아니면 값만 populate) ─
  const formSignatureOf = (spec: EntitySpec): string => {
    const robot = deps.isRobot(spec.id);
    const shapeKind = robot ? null : (primitiveFormStateOf(spec)?.kind ?? null);
    return [robot ? 'robot' : 'entity', shapeKind ?? '-', spec.physics !== undefined ? 'phys' : '-'].join('|');
  };

  // ── 섹션 골격 (접기 토글 — dock/inspector와 같은 ▾/▸ 규약) ─────────
  const makeSection = (
    titleText: string,
    testId: string,
  ): { section: HTMLElement; body: HTMLElement } => {
    const section = styled(document.createElement('section'), {
      borderBottom: `1px solid ${COLOR.borderSoft}`,
      padding: '2px 0 8px 0',
      margin: '0 0 4px 0',
    });
    section.dataset.testid = testId;
    const toggle = makeButton(`▾ ${titleText}`, `${titleText} 섹션 접기/펼치기`, `${testId}-toggle`, 'ghost');
    styled(toggle, { width: '100%', textAlign: 'left', padding: '2px 4px', fontWeight: '600' });
    const body = styled(document.createElement('div'), { padding: '4px 4px 0 4px' });
    let collapsed = false;
    const paint = (): void => {
      body.style.display = collapsed ? 'none' : '';
      toggle.textContent = `${collapsed ? '▸' : '▾'} ${titleText}`;
      toggle.setAttribute('aria-expanded', String(!collapsed));
    };
    toggle.addEventListener('click', () => {
      collapsed = !collapsed;
      paint();
    });
    paint();
    section.appendChild(toggle);
    section.appendChild(body);
    return { section, body };
  };

  // ── 이름/ID 섹션 ───────────────────────────────────────────────────
  const buildNameSection = (): HTMLElement => {
    const { section, body } = makeSection('이름/ID', 'ee-sec-name');
    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'ui-input';
    styled(input, { width: '100%', boxSizing: 'border-box' });
    input.dataset.testid = 'ee-name';
    input.setAttribute('aria-label', '엔티티 이름/ID');

    const errorLine = styled(document.createElement('div'), {
      color: COLOR.errorText,
      fontSize: '11px',
      padding: '2px 0 0 0',
      display: 'none',
      whiteSpace: 'normal',
    });
    errorLine.dataset.testid = 'ee-name-error';
    const showError = (message: string | null): void => {
      errorLine.style.display = message === null ? 'none' : '';
      errorLine.textContent = message ?? '';
    };

    const commit = (): void => {
      if (currentId === null) return;
      const next = input.value.trim();
      if (next === currentId) {
        showError(null);
        return;
      }
      const localError = localNameError(next);
      if (localError !== null) {
        showError(localError);
        return;
      }
      const error = deps.renameEntity(currentId, next);
      if (error !== null) {
        showError(error);
        return;
      }
      // 성공: 이후 커밋/refreshFrom이 새 id 기준으로 동작하도록 즉시 갱신
      currentId = next;
      input.value = next;
      showError(null);
    };
    input.addEventListener('change', commit);
    input.addEventListener('keydown', (e: KeyboardEvent) => {
      if (e.key === 'Enter') input.blur();
    });

    body.appendChild(input);
    body.appendChild(errorLine);
    fieldBindings.push({
      control: input,
      populate: (spec): void => {
        input.value = spec.id;
        showError(null);
      },
    });
    return section;
  };

  // ── Transform 섹션 (position + rotation 오일러 입력 → 쿼터니언 커밋) ─
  const buildTransformSection = (): HTMLElement => {
    const { section, body } = makeSection('Transform', 'ee-sec-transform');

    function commitPosition(): void {
      const spec = specOf();
      if (spec === null) return;
      const base = spec.transform.position;
      const next: Vec3 = [
        parseNumeric(px.input.value) ?? base[0],
        parseNumeric(py.input.value) ?? base[1],
        parseNumeric(pz.input.value) ?? base[2],
      ];
      deps.updateTransform(spec.id, { ...spec.transform, position: next });
      px.input.value = formatFixed(next[0], POSITION_DECIMALS_EDIT);
      py.input.value = formatFixed(next[1], POSITION_DECIMALS_EDIT);
      pz.input.value = formatFixed(next[2], POSITION_DECIMALS_EDIT);
    }

    function commitRotation(): void {
      const spec = specOf();
      if (spec === null) return;
      const baseEuler = quatToEulerDegXYZ(spec.transform.rotation ?? IDENTITY_QUAT);
      const eulerDeg: [number, number, number] = [
        parseNumeric(rx.input.value) ?? baseEuler[0],
        parseNumeric(ry.input.value) ?? baseEuler[1],
        parseNumeric(rz.input.value) ?? baseEuler[2],
      ];
      deps.updateTransform(spec.id, { ...spec.transform, rotation: eulerDegToQuat(eulerDeg) });
      rx.input.value = formatFixed(eulerDeg[0], ROTATION_DECIMALS_EDIT);
      ry.input.value = formatFixed(eulerDeg[1], ROTATION_DECIMALS_EDIT);
      rz.input.value = formatFixed(eulerDeg[2], ROTATION_DECIMALS_EDIT);
    }

    const posOpts = {
      step: POSITION_STEP_M,
      decimals: POSITION_DECIMALS_EDIT,
      scrubPerPx: SCRUB_POSITION_PER_PX,
      onCommit: commitPosition,
      onScrubActive,
    } as const;
    const px = numberCell({ label: 'X', testId: 'ee-pos-x', ...posOpts });
    const py = numberCell({ label: 'Y', testId: 'ee-pos-y', ...posOpts });
    const pz = numberCell({ label: 'Z', testId: 'ee-pos-z', ...posOpts });

    const rotOpts = {
      step: ROTATION_STEP_DEG,
      decimals: ROTATION_DECIMALS_EDIT,
      scrubPerPx: SCRUB_ROTATION_PER_PX_DEG,
      onCommit: commitRotation,
      onScrubActive,
    } as const;
    const rx = numberCell({ label: 'RX°', testId: 'ee-rot-x', ...rotOpts });
    const ry = numberCell({ label: 'RY°', testId: 'ee-rot-y', ...rotOpts });
    const rz = numberCell({ label: 'RZ°', testId: 'ee-rot-z', ...rotOpts });

    body.appendChild(subCaption('position (m)'));
    const posRow = tripleRow();
    posRow.appendChild(px.cell);
    posRow.appendChild(py.cell);
    posRow.appendChild(pz.cell);
    body.appendChild(posRow);

    body.appendChild(subCaption('rotation (deg · XYZ — 커밋 시 쿼터니언 변환)'));
    const rotRow = tripleRow();
    rotRow.appendChild(rx.cell);
    rotRow.appendChild(ry.cell);
    rotRow.appendChild(rz.cell);
    body.appendChild(rotRow);

    bindNumber(px.input, (s) => s.transform.position[0], POSITION_DECIMALS_EDIT);
    bindNumber(py.input, (s) => s.transform.position[1], POSITION_DECIMALS_EDIT);
    bindNumber(pz.input, (s) => s.transform.position[2], POSITION_DECIMALS_EDIT);
    bindNumber(
      rx.input,
      (s) => quatToEulerDegXYZ(s.transform.rotation ?? IDENTITY_QUAT)[0],
      ROTATION_DECIMALS_EDIT,
    );
    bindNumber(
      ry.input,
      (s) => quatToEulerDegXYZ(s.transform.rotation ?? IDENTITY_QUAT)[1],
      ROTATION_DECIMALS_EDIT,
    );
    bindNumber(
      rz.input,
      (s) => quatToEulerDegXYZ(s.transform.rotation ?? IDENTITY_QUAT)[2],
      ROTATION_DECIMALS_EDIT,
    );
    return section;
  };

  // ── Dimensions 섹션 (프리미티브 전용 — 전체 치수 UI, halfExtents/2 내부 변환) ─
  const buildDimensionsSection = (initial: ShapeFormState): HTMLElement => {
    const { section, body } = makeSection('Dimensions', 'ee-sec-dims');
    body.appendChild(
      subCaption(`전체 치수(m) 기준 — 내부는 halfExtents/2로 저장 · 최소 ${DIMENSION_MIN_M} m`),
    );

    const dimOpts = {
      step: DIMENSION_STEP_M,
      min: DIMENSION_MIN_M,
      decimals: DIMENSION_DECIMALS_EDIT,
      scrubPerPx: SCRUB_DIMENSION_PER_PX,
      onScrubActive,
    } as const;
    const row = tripleRow();
    body.appendChild(row);

    const commitShape = (form: ShapeFormState): void => {
      const spec = specOf();
      if (spec === null) return;
      deps.updateDimensions(spec.id, formStateToShape(form));
    };

    if (initial.kind === 'box') {
      function commit(): void {
        const spec = specOf();
        if (spec === null) return;
        const fallback = primitiveFormStateOf(spec);
        if (fallback === null || fallback.kind !== 'box') return;
        const form: ShapeFormState = {
          kind: 'box',
          widthM: parseNumeric(w.input.value) ?? fallback.widthM,
          heightM: parseNumeric(h.input.value) ?? fallback.heightM,
          lengthM: parseNumeric(l.input.value) ?? fallback.lengthM,
        };
        commitShape(form);
        const norm = shapeToFormState(formStateToShape(form));
        if (norm !== null && norm.kind === 'box') {
          w.input.value = formatFixed(norm.widthM, DIMENSION_DECIMALS_EDIT);
          h.input.value = formatFixed(norm.heightM, DIMENSION_DECIMALS_EDIT);
          l.input.value = formatFixed(norm.lengthM, DIMENSION_DECIMALS_EDIT);
        }
      }
      const w = numberCell({ label: 'W 너비', testId: 'ee-dim-w', onCommit: commit, ...dimOpts });
      const h = numberCell({ label: 'H 높이', testId: 'ee-dim-h', onCommit: commit, ...dimOpts });
      const l = numberCell({ label: 'L 길이', testId: 'ee-dim-l', onCommit: commit, ...dimOpts });
      row.appendChild(w.cell);
      row.appendChild(h.cell);
      row.appendChild(l.cell);
      const boxForm = (s: EntitySpec): Extract<ShapeFormState, { kind: 'box' }> | null => {
        const f = primitiveFormStateOf(s);
        return f !== null && f.kind === 'box' ? f : null;
      };
      bindNumber(w.input, (s) => boxForm(s)?.widthM ?? null, DIMENSION_DECIMALS_EDIT);
      bindNumber(h.input, (s) => boxForm(s)?.heightM ?? null, DIMENSION_DECIMALS_EDIT);
      bindNumber(l.input, (s) => boxForm(s)?.lengthM ?? null, DIMENSION_DECIMALS_EDIT);
    } else if (initial.kind === 'sphere') {
      function commit(): void {
        const spec = specOf();
        if (spec === null) return;
        const fallback = primitiveFormStateOf(spec);
        if (fallback === null || fallback.kind !== 'sphere') return;
        const form: ShapeFormState = {
          kind: 'sphere',
          radiusM: parseNumeric(r.input.value) ?? fallback.radiusM,
        };
        commitShape(form);
        const norm = shapeToFormState(formStateToShape(form));
        if (norm !== null && norm.kind === 'sphere') {
          r.input.value = formatFixed(norm.radiusM, DIMENSION_DECIMALS_EDIT);
        }
      }
      const r = numberCell({ label: '반지름', testId: 'ee-dim-radius', onCommit: commit, ...dimOpts });
      row.appendChild(r.cell);
      bindNumber(
        r.input,
        (s) => {
          const f = primitiveFormStateOf(s);
          return f !== null && f.kind === 'sphere' ? f.radiusM : null;
        },
        DIMENSION_DECIMALS_EDIT,
      );
    } else {
      // cylinder | capsule — 반지름 + 전체 높이
      const kind = initial.kind;
      function commit(): void {
        const spec = specOf();
        if (spec === null) return;
        const fallback = primitiveFormStateOf(spec);
        if (fallback === null || fallback.kind !== kind) return;
        const form: ShapeFormState = {
          kind,
          radiusM: parseNumeric(r.input.value) ?? fallback.radiusM,
          heightM: parseNumeric(h.input.value) ?? fallback.heightM,
        };
        commitShape(form);
        const norm = shapeToFormState(formStateToShape(form));
        if (norm !== null && (norm.kind === 'cylinder' || norm.kind === 'capsule')) {
          r.input.value = formatFixed(norm.radiusM, DIMENSION_DECIMALS_EDIT);
          h.input.value = formatFixed(norm.heightM, DIMENSION_DECIMALS_EDIT);
        }
      }
      const r = numberCell({ label: '반지름', testId: 'ee-dim-radius', onCommit: commit, ...dimOpts });
      const h = numberCell({ label: '높이', testId: 'ee-dim-height', onCommit: commit, ...dimOpts });
      row.appendChild(r.cell);
      row.appendChild(h.cell);
      if (kind === 'capsule') {
        body.appendChild(subCaption('캡슐 높이 = 원통부 전체 높이 (반구 캡 제외)'));
      }
      const cylForm = (
        s: EntitySpec,
      ): Extract<ShapeFormState, { kind: 'cylinder' | 'capsule' }> | null => {
        const f = primitiveFormStateOf(s);
        return f !== null && (f.kind === 'cylinder' || f.kind === 'capsule') ? f : null;
      };
      bindNumber(r.input, (s) => cylForm(s)?.radiusM ?? null, DIMENSION_DECIMALS_EDIT);
      bindNumber(h.input, (s) => cylForm(s)?.heightM ?? null, DIMENSION_DECIMALS_EDIT);
    }
    return section;
  };

  // ── Physics 섹션 ───────────────────────────────────────────────────
  const buildPhysicsSection = (initialPhysics: PhysicsSpec): HTMLElement => {
    const initial = physicsFormStateOf(initialPhysics);
    const { section, body } = makeSection('Physics', 'ee-sec-physics');

    function commitPhysics(): void {
      const spec = specOf();
      if (spec === null || spec.physics === undefined) return;
      const fallback = physicsFormStateOf(spec.physics);
      const form: PhysicsFormState = {
        // 옵션 값은 bodyTypeChoices/groupChoices 후보에서만 유래 — 안전한 좁히기
        bodyType: bodySelect.value as BodyType,
        friction: parseNumeric(friction.input.value) ?? fallback.friction,
        restitution: parseNumeric(restitution.input.value) ?? fallback.restitution,
        density: parseNumeric(density.input.value) ?? fallback.density,
        group: groupSelect.value as ColliderGroup,
        collidesWith: [...collideChecks.entries()]
          .filter(([, box]) => box.checked)
          .map(([group]) => group),
        isSensor: sensorBox.checked,
        emitEvents: emitBox.checked,
        ccd: ccdBox.checked,
      };
      const applied = applyPhysicsFormState(spec.physics, form);
      deps.updatePhysics(spec.id, applied);
      // 클램프 반영 값 되쓰기 (재동기화 자체는 통합자의 refreshFrom 몫)
      const norm = physicsFormStateOf(applied);
      friction.input.value = formatFixed(norm.friction, PHYSICS_DECIMALS_EDIT);
      restitution.input.value = formatFixed(norm.restitution, PHYSICS_DECIMALS_EDIT);
      density.input.value = formatFixed(norm.density, PHYSICS_DECIMALS_EDIT);
    }

    // bodyType
    body.appendChild(subCaption('bodyType'));
    const bodySelect = document.createElement('select');
    bodySelect.className = 'ui-select';
    styled(bodySelect, { width: '100%', boxSizing: 'border-box' });
    bodySelect.dataset.testid = 'ee-body-type';
    bodySelect.setAttribute('aria-label', 'bodyType');
    for (const choice of bodyTypeChoices(initial.bodyType)) {
      addOption(bodySelect, choice, bodyTypeLabel(choice));
    }
    bodySelect.value = initial.bodyType;
    bodySelect.addEventListener('change', commitPhysics);
    body.appendChild(bodySelect);

    // friction / restitution / density (0..2)
    const paramOpts = {
      step: PHYSICS_STEP,
      min: PHYSICS_PARAM_MIN,
      max: PHYSICS_PARAM_MAX,
      decimals: PHYSICS_DECIMALS_EDIT,
      scrubPerPx: SCRUB_PHYSICS_PER_PX,
      onCommit: commitPhysics,
      onScrubActive,
    } as const;
    const friction = numberCell({ label: '마찰', testId: 'ee-friction', ...paramOpts });
    const restitution = numberCell({ label: '반발', testId: 'ee-restitution', ...paramOpts });
    const density = numberCell({ label: '밀도', testId: 'ee-density', ...paramOpts });
    body.appendChild(subCaption(`계수 (${PHYSICS_PARAM_MIN}–${PHYSICS_PARAM_MAX})`));
    const paramRow = tripleRow();
    paramRow.appendChild(friction.cell);
    paramRow.appendChild(restitution.cell);
    paramRow.appendChild(density.cell);
    body.appendChild(paramRow);

    // 충돌 그룹 (CLAUDE.md §5)
    body.appendChild(subCaption('그룹'));
    const groupSelect = document.createElement('select');
    groupSelect.className = 'ui-select';
    styled(groupSelect, { width: '100%', boxSizing: 'border-box' });
    groupSelect.dataset.testid = 'ee-group';
    groupSelect.setAttribute('aria-label', '충돌 그룹');
    for (const choice of groupChoices(initial.group)) {
      addOption(groupSelect, choice, groupLabel(choice));
    }
    groupSelect.value = initial.group;
    groupSelect.addEventListener('change', commitPhysics);
    body.appendChild(groupSelect);

    // collidesWith 체크박스 그룹
    body.appendChild(subCaption('충돌 대상 (collidesWith)'));
    const collidesBox = styled(document.createElement('div'), {
      display: 'flex',
      flexWrap: 'wrap',
      gap: '2px 12px',
      padding: '2px 0',
    });
    collidesBox.dataset.testid = 'ee-collides';
    body.appendChild(collidesBox);
    const collideChecks = new Map<ColliderGroup, HTMLInputElement>();
    const makeCheckbox = (
      labelText: string,
      testId: string,
      checked: boolean,
    ): { row: HTMLElement; box: HTMLInputElement } => {
      const rowEl = styled(document.createElement('label'), {
        display: 'flex',
        alignItems: 'center',
        gap: '6px',
        padding: '1px 0',
        cursor: 'pointer',
        color: COLOR.text,
      });
      const box = document.createElement('input');
      box.type = 'checkbox';
      box.checked = checked;
      box.dataset.testid = testId;
      styled(box, { accentColor: COLOR.accent, margin: '0' });
      box.addEventListener('change', commitPhysics);
      const text = document.createElement('span');
      text.textContent = labelText;
      rowEl.appendChild(box);
      rowEl.appendChild(text);
      return { row: rowEl, box };
    };
    const rebuildCollides = (
      choices: readonly ColliderGroup[],
      checked: readonly ColliderGroup[],
    ): void => {
      collidesBox.replaceChildren();
      collideChecks.clear();
      for (const group of choices) {
        const { row, box } = makeCheckbox(group, `ee-collides-${group}`, checked.includes(group));
        styled(row, { fontSize: '11px' });
        collidesBox.appendChild(row);
        collideChecks.set(group, box);
      }
    };
    rebuildCollides(collidesWithChoices(initial.collidesWith), initial.collidesWith);

    // sensor / emitEvents / ccd
    body.appendChild(subCaption('플래그'));
    const sensor = makeCheckbox('센서 (isSensor — 물리 반응 없음)', 'ee-sensor', initial.isSensor);
    const emit = makeCheckbox('충돌 이벤트 발행 (emitEvents)', 'ee-emit-events', initial.emitEvents);
    const ccd = makeCheckbox('CCD (터널링 방지)', 'ee-ccd', initial.ccd);
    const sensorBox = sensor.box;
    const emitBox = emit.box;
    const ccdBox = ccd.box;
    body.appendChild(sensor.row);
    body.appendChild(emit.row);
    body.appendChild(ccd.row);

    // populate 바인딩 (외부 변경 재동기화 — 포커스/스크럽 중이면 건너뜀)
    const physicsForm = (s: EntitySpec): PhysicsFormState | null =>
      s.physics !== undefined ? physicsFormStateOf(s.physics) : null;
    fieldBindings.push({
      control: bodySelect,
      populate: (s): void => {
        const f = physicsForm(s);
        if (f === null) return;
        ensureOption(bodySelect, f.bodyType, bodyTypeLabel(f.bodyType));
        bodySelect.value = f.bodyType;
      },
    });
    bindNumber(friction.input, (s) => physicsForm(s)?.friction ?? null, PHYSICS_DECIMALS_EDIT);
    bindNumber(restitution.input, (s) => physicsForm(s)?.restitution ?? null, PHYSICS_DECIMALS_EDIT);
    bindNumber(density.input, (s) => physicsForm(s)?.density ?? null, PHYSICS_DECIMALS_EDIT);
    fieldBindings.push({
      control: groupSelect,
      populate: (s): void => {
        const f = physicsForm(s);
        if (f === null) return;
        ensureOption(groupSelect, f.group, groupLabel(f.group));
        groupSelect.value = f.group;
      },
    });
    fieldBindings.push({
      control: collidesBox,
      populate: (s): void => {
        const f = physicsForm(s);
        if (f === null) return;
        const choices = collidesWithChoices(f.collidesWith);
        const known = [...collideChecks.keys()];
        const sameChoices =
          choices.length === known.length && choices.every((group, i) => known[i] === group);
        if (!sameChoices) {
          rebuildCollides(choices, f.collidesWith);
        } else {
          for (const [group, box] of collideChecks) box.checked = f.collidesWith.includes(group);
        }
      },
    });
    fieldBindings.push({
      control: sensorBox,
      populate: (s): void => {
        const f = physicsForm(s);
        if (f !== null) sensorBox.checked = f.isSensor;
      },
    });
    fieldBindings.push({
      control: emitBox,
      populate: (s): void => {
        const f = physicsForm(s);
        if (f !== null) emitBox.checked = f.emitEvents;
      },
    });
    fieldBindings.push({
      control: ccdBox,
      populate: (s): void => {
        const f = physicsForm(s);
        if (f !== null) ccdBox.checked = f.ccd;
      },
    });
    return section;
  };

  // ── 폼 (재)구축 ────────────────────────────────────────────────────
  const buildForm = (spec: EntitySpec | null): void => {
    fieldBindings.length = 0;
    scrubControl = null;
    content.replaceChildren();
    if (spec === null) {
      content.appendChild(mutedLine('엔티티를 선택하면 편집할 수 있습니다'));
      return;
    }
    content.appendChild(buildNameSection());
    content.appendChild(buildTransformSection());
    if (deps.isRobot(spec.id)) {
      // 로봇: 물리/치수는 URDF 유도 — 이름/Transform만 편집 (요구 §2 Robot)
      const note = styled(document.createElement('div'), {
        color: COLOR.muted,
        fontSize: '11px',
        padding: '4px',
      });
      note.dataset.testid = 'ee-robot-note';
      note.textContent = '로봇 물리는 URDF에서 유도됩니다 — 이름/Transform만 편집 가능합니다';
      content.appendChild(note);
    } else {
      const dimForm = primitiveFormStateOf(spec);
      if (dimForm !== null) content.appendChild(buildDimensionsSection(dimForm));
      if (spec.physics !== undefined) content.appendChild(buildPhysicsSection(spec.physics));
    }
    populateAll(spec);
  };

  buildForm(null);
  host.appendChild(root);

  return {
    el: root,
    showFor: (id): void => {
      const spec = id === null ? null : deps.getEntity(id);
      currentId = spec?.id ?? null;
      formSignature = spec === null ? null : formSignatureOf(spec);
      buildForm(spec);
    },
    refreshFrom: (spec): void => {
      if (spec.id !== currentId) {
        // 다른 id의 spec 통지(외부 rename 등) — 그 spec을 채택해 재구축
        currentId = spec.id;
        formSignature = formSignatureOf(spec);
        buildForm(spec);
        return;
      }
      const signature = formSignatureOf(spec);
      if (signature !== formSignature) {
        formSignature = signature;
        buildForm(spec);
        return;
      }
      populateAll(spec);
    },
    dispose: (): void => {
      fieldBindings.length = 0;
      currentId = null;
      root.remove();
    },
  };
}
