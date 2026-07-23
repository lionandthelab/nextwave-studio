// ui/library/templates.ts — 라이브러리 템플릿 레지스트리 (UX_DESIGN §3.2 Objects/Robots)
//
// 순수 데이터 + 팩토리 모듈이다: DOM/물리/렌더에 의존하지 않고 schema 타입만 안다
// (CLAUDE.md §3 계층 규칙 — node 단위 테스트 대상). 각 템플릿의 create()는 씬에
// 그대로 추가 가능한 EntitySpec을 만들며, 최소 씬으로 감싸면 validateScene을 통과한다
// (templates.test.ts가 계약으로 검증).
//
// id 발급: create(uniquify)의 uniquify 콜백이 idBase('box' 등)로부터 'box_1' 형식의
// 씬-유일 id를 만든다 — 유일성의 진실(현재 씬의 entities)은 통합자/SceneEditor가
// 소유하므로 이 모듈은 카운터를 들지 않는다.
//
// 충돌 그룹 규약 (CLAUDE.md §5 + phase6 샘플 씬):
// - 동적 프리미티브: OBJECT 그룹, collidesWith [ENV, ROBOT, OBJECT, SENSOR_ZONE],
//   emitEvents true. SENSOR_ZONE을 포함하는 이유 — Rapier 쌍 필터는 양방향이라
//   (core/phase6-scenes.test.ts 규약) OBJECT 측 필터에 SENSOR_ZONE이 없으면 라이브러리로
//   추가한 Sensor Zone이 사물을 감지하지 못한다. pick-and-place 샘플 씬과 동일한 배정.
// - Sensor Zone: static + fixed + isSensor, SENSOR_ZONE 그룹, collidesWith
//   [OBJECT, ROBOT], emitEvents true — 감지만 하고 튕기지 않는다 (CLAUDE.md §5).
// - 얇은 Plane: 박스로 구현한 정적 환경 판 — ENV 그룹(바닥/벽 등 정적 환경 규약),
//   fixed 바디. 사물 낙하 받침/작업대 용도가 기본이므로 동적 OBJECT가 아니라 ENV로 둔다.
//
// 시각 색상은 UI 스타일이 아니라 "씬 데이터"다 — ui/theme 토큰이 아닌 기존 샘플 씬
// (arm-and-boxes/pick-and-place)·render/meshes 기본 팔레트와 일관된 값을 상수로 쓴다.

import type { ColliderGroup, ColliderShape, EntitySpec, RobotSpec, Vec3 } from '../../schema';

// ── 공개 타입 ───────────────────────────────────────────────────────

/** idBase → 씬-유일 id (예: 'box' → 'box_1'). 유일성 판단은 호출자(통합자) 몫. */
export type Uniquify = (base: string) => string;

export type TemplateSection = 'objects' | 'robots';

export interface LibraryTemplate {
  /** 안정 키 — 드래그 페이로드(dataTransfer)·검색·테스트 id로 쓰인다 */
  readonly key: string;
  readonly section: TemplateSection;
  /** 카드 표기 라벨 (한국어 병기) */
  readonly labelKo: string;
  /** 카드 아이콘 (이모지/기호 — UX_DESIGN §2 와이어프레임 표기와 일치) */
  readonly icon: string;
  /** 호버 설명 (한국어) */
  readonly descriptionKo: string;
  /** 기본 id 접두 — create()가 uniquify(idBase)로 id를 발급한다 */
  readonly idBase: string;
  /** 씬에 추가 가능한 EntitySpec 생성 (호출마다 새 객체 — 공유 참조 없음) */
  create(uniquify: Uniquify): EntitySpec;
}

// ── 기본 치수 상수 (미터 — 매직넘버 금지, CLAUDE.md §4) ─────────────

/** Box 기본 half extents — 과제 규약의 기본 치수 */
export const DEFAULT_BOX_HALF_EXTENTS: Vec3 = [0.05, 0.05, 0.05];
const DEFAULT_SPHERE_RADIUS_M = 0.05;
const DEFAULT_CYLINDER_RADIUS_M = 0.05;
const DEFAULT_CYLINDER_HALF_HEIGHT_M = 0.05;
const DEFAULT_CAPSULE_RADIUS_M = 0.03;
const DEFAULT_CAPSULE_HALF_HEIGHT_M = 0.05;
/** 얇은 판 (박스로): 50cm × 2cm × 50cm */
const PLANE_HALF_EXTENTS: Vec3 = [0.25, 0.01, 0.25];
/** 감지 영역 기본 크기: 20cm × 10cm × 20cm */
const SENSOR_ZONE_HALF_EXTENTS: Vec3 = [0.1, 0.05, 0.1];

/** 동적 사물 기본 마찰 (arm-and-boxes 샘플 씬과 동일) */
const OBJECT_FRICTION = 0.6;

// ── 충돌 그룹 배정 (파일 헤더의 근거 참조) ──────────────────────────

const OBJECT_COLLIDES_WITH: readonly ColliderGroup[] = ['ENV', 'ROBOT', 'OBJECT', 'SENSOR_ZONE'];
const PLANE_COLLIDES_WITH: readonly ColliderGroup[] = ['ROBOT', 'OBJECT'];
const SENSOR_COLLIDES_WITH: readonly ColliderGroup[] = ['OBJECT', 'ROBOT'];

// ── 시각 색 (씬 데이터 — 기존 샘플 씬/meshes 기본 팔레트와 일관) ────

const BOX_COLOR = '#c0392b';      // 붉은 벽돌 (DATA_MODEL 예시·arm-and-boxes box_a)
const SPHERE_COLOR = '#2980b9';   // 청색 (arm-and-boxes box_b)
const CYLINDER_COLOR = '#d4930f'; // 호박색 (meshes 기본 실린더 색)
const CAPSULE_COLOR = '#27ae60';  // 녹색 (meshes 기본 캡슐 색)
const PLANE_COLOR = '#7f8c8d';    // 회색 판
/** 반투명 느낌의 어두운 틴트 — pick-and-place drop_zone과 동일 (render가 알파 미지원) */
const SENSOR_ZONE_COLOR = '#1e4d40';

// ── Arm-6 로봇 템플릿 상수 (arm-and-boxes 씬의 로봇 정의와 동일) ────

const ARM6_URDF_PATH = 'assets/robots/arm6/arm6.urdf';
const ARM6_HOME: Readonly<Record<string, number>> = {
  joint1: 0.0,
  joint2: -0.6,
  joint3: 1.2,
  joint5: -0.6,
};
const ARM6_GRIPPER_JOINTS: readonly string[] = ['finger_left_joint', 'finger_right_joint'];
const ARM6_GRIPPER_OPEN = 0.03;
const ARM6_GRIPPER_CLOSE = 0.0;

// ── 팩토리 헬퍼 ─────────────────────────────────────────────────────

/** 프리미티브 형상의 "바닥 접지" 스폰 높이 (형상 최저점이 y=0에 오는 중심 y) */
function restingY(shape: ColliderShape): number {
  switch (shape.kind) {
    case 'box':
      return shape.halfExtents[1];
    case 'sphere':
      return shape.radius;
    case 'capsule':
      return shape.halfHeight + shape.radius;
    case 'cylinder':
      return shape.halfHeight;
    default:
      return 0;
  }
}

/** 동적 사물 프리미티브 EntitySpec — 시각 primitive와 collider 형상은 동일(단일 치수 진실) */
function dynamicObjectEntity(
  uniquify: Uniquify,
  idBase: string,
  shape: ColliderShape,
  color: string,
): EntitySpec {
  return {
    id: uniquify(idBase),
    type: 'object',
    transform: { position: [0, restingY(shape), 0] },
    visual: { kind: 'primitive', primitive: shape, color },
    physics: {
      bodyType: 'dynamic',
      colliders: [
        {
          shape,
          friction: OBJECT_FRICTION,
          group: 'OBJECT',
          collidesWith: [...OBJECT_COLLIDES_WITH],
          emitEvents: true,
        },
      ],
    },
  };
}

/** 얇은 판 (정적 ENV — 파일 헤더의 그룹 배정 근거 참조) */
function planeEntity(uniquify: Uniquify): EntitySpec {
  const shape: ColliderShape = { kind: 'box', halfExtents: [...PLANE_HALF_EXTENTS] };
  return {
    id: uniquify('plane'),
    type: 'static',
    transform: { position: [0, restingY(shape), 0] },
    visual: { kind: 'primitive', primitive: shape, color: PLANE_COLOR },
    physics: {
      bodyType: 'fixed',
      colliders: [
        {
          shape,
          group: 'ENV',
          collidesWith: [...PLANE_COLLIDES_WITH],
          emitEvents: true,
        },
      ],
    },
  };
}

/** 감지 전용 Sensor Zone — 물리 반응 없이 교차만 감지 (CLAUDE.md §5 SENSOR_ZONE) */
function sensorZoneEntity(uniquify: Uniquify): EntitySpec {
  const shape: ColliderShape = { kind: 'box', halfExtents: [...SENSOR_ZONE_HALF_EXTENTS] };
  return {
    id: uniquify('sensor_zone'),
    type: 'static',
    transform: { position: [0, restingY(shape), 0] },
    visual: { kind: 'primitive', primitive: shape, color: SENSOR_ZONE_COLOR },
    physics: {
      bodyType: 'fixed',
      colliders: [
        {
          shape,
          isSensor: true,
          group: 'SENSOR_ZONE',
          collidesWith: [...SENSOR_COLLIDES_WITH],
          emitEvents: true,
        },
      ],
    },
  };
}

/** Arm-6 로봇 — arm-and-boxes 씬과 동일한 home/gripper 구성, home 포즈로 배치 (UX §3.2) */
function arm6Entity(uniquify: Uniquify): RobotSpec {
  return {
    id: uniquify('arm'),
    type: 'robot',
    transform: { position: [0, 0, 0] },
    visual: { kind: 'urdf', ref: ARM6_URDF_PATH },
    urdf: ARM6_URDF_PATH,
    controller: 'sequence',
    linkColliders: 'fromVisual',
    selfCollision: false,
    home: { ...ARM6_HOME },
    gripper: {
      joints: [...ARM6_GRIPPER_JOINTS],
      open: ARM6_GRIPPER_OPEN,
      close: ARM6_GRIPPER_CLOSE,
    },
  };
}

// ── 템플릿 레지스트리 ───────────────────────────────────────────────

export const LIBRARY_TEMPLATES: readonly LibraryTemplate[] = [
  {
    key: 'box',
    section: 'objects',
    labelKo: 'Box · 박스',
    icon: '▣',
    descriptionKo: '동적 박스 (10cm) — 드롭 후 인스펙터에서 W/H/L 조정',
    idBase: 'box',
    create: (uniquify) =>
      dynamicObjectEntity(
        uniquify,
        'box',
        { kind: 'box', halfExtents: [...DEFAULT_BOX_HALF_EXTENTS] },
        BOX_COLOR,
      ),
  },
  {
    key: 'sphere',
    section: 'objects',
    labelKo: 'Sphere · 구',
    icon: '●',
    descriptionKo: '동적 구 (반지름 5cm)',
    idBase: 'sphere',
    create: (uniquify) =>
      dynamicObjectEntity(
        uniquify,
        'sphere',
        { kind: 'sphere', radius: DEFAULT_SPHERE_RADIUS_M },
        SPHERE_COLOR,
      ),
  },
  {
    key: 'cylinder',
    section: 'objects',
    labelKo: 'Cylinder · 원통',
    icon: '◍',
    descriptionKo: '동적 원통 (반지름 5cm · 높이 10cm)',
    idBase: 'cylinder',
    create: (uniquify) =>
      dynamicObjectEntity(
        uniquify,
        'cylinder',
        {
          kind: 'cylinder',
          radius: DEFAULT_CYLINDER_RADIUS_M,
          halfHeight: DEFAULT_CYLINDER_HALF_HEIGHT_M,
        },
        CYLINDER_COLOR,
      ),
  },
  {
    key: 'capsule',
    section: 'objects',
    labelKo: 'Capsule · 캡슐',
    icon: '⬭',
    descriptionKo: '동적 캡슐 (반지름 3cm · 몸통 10cm)',
    idBase: 'capsule',
    create: (uniquify) =>
      dynamicObjectEntity(
        uniquify,
        'capsule',
        {
          kind: 'capsule',
          radius: DEFAULT_CAPSULE_RADIUS_M,
          halfHeight: DEFAULT_CAPSULE_HALF_HEIGHT_M,
        },
        CAPSULE_COLOR,
      ),
  },
  {
    key: 'plane',
    section: 'objects',
    labelKo: 'Plane · 판',
    icon: '▭',
    descriptionKo: '얇은 정적 판 (50cm 박스, ENV) — 받침/작업대 용도',
    idBase: 'plane',
    create: planeEntity,
  },
  {
    key: 'sensor-zone',
    section: 'objects',
    labelKo: 'Sensor Zone · 감지 영역',
    icon: '⌖',
    descriptionKo: '물리 반응 없이 진입/이탈만 감지하는 영역 (sensor)',
    idBase: 'sensor_zone',
    create: sensorZoneEntity,
  },
  {
    key: 'arm-6',
    section: 'robots',
    labelKo: 'Arm-6 · 6축 로봇팔',
    icon: '🦾',
    descriptionKo: '6-DOF 로봇팔 + 평행 그리퍼 — home 포즈로 배치',
    idBase: 'arm',
    create: arm6Entity,
  },
];

/** 드래그 드롭 페이로드(템플릿 키) → 템플릿. 미지 키는 undefined (통합자 드롭 핸들러용) */
export function templateByKey(key: string): LibraryTemplate | undefined {
  return LIBRARY_TEMPLATES.find((t) => t.key === key);
}

// ── 검색 필터 (순수 — library.ts 검색 입력이 소비) ──────────────────

/** 대소문자 무시 부분 일치: labelKo · key · descriptionKo 중 하나라도 맞으면 통과 */
export function matchesQuery(template: LibraryTemplate, query: string): boolean {
  const q = query.trim().toLowerCase();
  if (q === '') return true;
  return (
    template.labelKo.toLowerCase().includes(q) ||
    template.key.toLowerCase().includes(q) ||
    template.descriptionKo.toLowerCase().includes(q)
  );
}

/** 검색어로 템플릿 목록 필터 (빈/공백 검색어 = 전체) */
export function filterTemplates(
  templates: readonly LibraryTemplate[],
  query: string,
): readonly LibraryTemplate[] {
  return templates.filter((t) => matchesQuery(t, query));
}
