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

import { DETECTION_ZONE_TAG } from '../../schema';
import type { ColliderGroup, ColliderShape, EntitySpec, RobotSpec, Vec3 } from '../../schema';
import type { IconName } from '../icons';

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
  /**
   * 카드 아이콘 — **SVG 아이콘 이름**이다(UX_AUDIT C-13).
   *
   * 이전에는 `string`이라 이모지·기하문자(`▣ ● ◍ ⬭ ▭ ⌖ 🦾`)를 담았고, 그 결과 타일 6개는
   * 밝은 회색 딩벳인데 `Arm-6`만 풀컬러 이모지라 크기·광학무게·색·베이스라인이 전부
   * 달랐다. 타입을 `IconName`으로 좁히면 **컴파일러가 누락과 오타를 잡는다** — 새 템플릿을
   * 추가하며 이모지를 다시 밀수입하는 경로가 구조적으로 막힌다.
   */
  readonly icon: IconName;
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
/** 컨베이어 벨트 기본 크기: 길이 80cm × 두께 4cm × 폭 24cm */
const CONVEYOR_HALF_EXTENTS: Vec3 = [0.4, 0.02, 0.12];
/** 컨베이어 기본 속도 (m/s) — 로봇이 따라잡을 수 있는 라인 속도 */
const CONVEYOR_SPEED_MPS = 0.12;
/** 컨베이어 기본 진행 방향 (엔티티 로컬 +X) */
const CONVEYOR_DIRECTION: Vec3 = [1, 0, 0];
/** 벨트 표면 마찰 — 사물이 미끄러지지 않고 실려 가도록 높게 */
const CONVEYOR_FRICTION = 0.8;

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
const CONVEYOR_COLOR = '#4a5568'; // 짙은 회청 벨트 (판보다 어둡게 — 표면이 흐른다는 구분)
/**
 * 감지 영역 색 + 표현. **반투명 + 모서리 선**이 규약이다 (VisualSpec.opacity/edges):
 * sensor는 통과 가능한데 불투명 상자로 그리면 화면이 "단단한 벽"이라고 말해, 사용자가
 * 관통을 결함으로 읽는다(실제 보고). 통과 가능한 것은 통과 가능해 보여야 한다.
 */
const SENSOR_ZONE_COLOR = '#2fbf8f';
const SENSOR_ZONE_OPACITY = 0.22;


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

// ── SCARA-4 로봇 템플릿 상수 (public/assets/robots/scara4/scara4.urdf) ──
//
// arm6(6축 관절팔)와 다른 것은 **구조**다: 수평면 회전 2축 + 수직 직동 1축 + 손목.
// 조립 라인의 상하 픽앤플레이스에 쓰이는 형태이고, 손도 평행 손가락이 아니라 흡착 패드다.
//
// home 실측(헤드리스 FK): joint2=-1.0으로 팔꿈치를 접어 두 수평 링크가 겹쳐 보이지
// 않게 했다(완전 신장은 옆에서 막대 하나로 보여 SCARA의 2단 구조가 읽히지 않는다).
// 이 포즈의 최저점은 y=0.0(받침이 바닥에 접지)이고 어떤 링크도 y<0으로 가지 않는다 —
// 로봇 링크는 kinematic이라 자가 교정이 없다(CLAUDE.md §2.11).
const SCARA4_URDF_PATH = 'assets/robots/scara4/scara4.urdf';
const SCARA4_HOME: Readonly<Record<string, number>> = {
  joint1: 0.0,
  joint2: -1.0,
  joint3: 0.0,
  joint4: 0.0, // 손목 — home에 없으면 시퀀스/플래너가 이 축을 주소지정할 수 없다
};
/**
 * joint3은 **prismatic(미터)** 인데 플래너의 씬 컨텍스트는 관절을 revolute(라디안)로
 * 가정한다. 범위를 명시해 두지 않으면 rad 스케일 값이 생성돼 조용히 클램프된다.
 * 값 규약: 관절값 = 하강 깊이(m). 0 = 최상단, 0.18 = 최하단.
 */
const SCARA4_JOINT_LIMITS: Readonly<Record<string, [number, number]>> = {
  joint3: [0, 0.18],
};
/**
 * 흡착 그리퍼 — 평행 손가락 2개가 아니라 **패드를 밀어 내리는 관절 1개**다.
 * open=0.0(수축·떨어짐), close=0.012(12mm 내려가 밀착). gripper step은 관절값 구동이
 * 전부이므로 손 모양이 달라도 기존 player가 그대로 동작한다.
 */
const SCARA4_GRIPPER_JOINTS: readonly string[] = ['suction_joint'];
const SCARA4_GRIPPER_OPEN = 0.0;
const SCARA4_GRIPPER_CLOSE = 0.012;

// ── Cobot-7 로봇 템플릿 상수 (public/assets/robots/cobot7/cobot7.urdf) ──
//
// 7축 여유자유도 협동로봇 + 3지 클로. arm6·SCARA와 손 모양이 또 다르다 —
// 손가락 3개가 120° 간격으로 동시에 굽는다(gripper.joints에 셋을 모두 나열).
//
// home 실측: 피치 3개의 합이 π라 공구축이 수직 하향(0.002, -1.000, -0.000)이다.
// 손끝 최저 y=0.0961, 전 링크 collider 최저 y=0.0 — 바닥 위 물체를 내려다보는 자세다.
const COBOT7_URDF_PATH = 'assets/robots/cobot7/cobot7.urdf';
const COBOT7_HOME: Readonly<Record<string, number>> = {
  joint1: 0.0,
  joint2: 0.7,
  joint3: 0.0,
  joint4: 1.1,
  joint5: 0.0,
  joint6: 1.34,
  joint7: 0.0,
};
/**
 * 3지 클로 — open/close는 **기계적 양 끝**이다(내접 지름 143.3mm ↔ 17.3mm).
 * 실제 파지에는 중간 숫자 state를 쓴다: `state: 'close'`를 5cm 상자에 쓰면 kinematic
 * 손가락이 물체를 뚫고 지나가 사출시킨다. arm6 pick-and-place가 `state: 0.6`을 쓰는
 * 것과 같은 이유다 (관절값 = lerp(close, open, state)).
 */
const COBOT7_GRIPPER_JOINTS: readonly string[] = [
  'finger_a_joint',
  'finger_b_joint',
  'finger_c_joint',
];
const COBOT7_GRIPPER_OPEN = -0.65;
/**
 * 0.05는 **패드 3장이 서로 닿는 지점**이다. 원래 값 0.1은 URDF 한계 안이었지만 세 패드가
 * 서로 3.3mm 파고들었다(OBB-SAT 실측 — src/render/robot-self-clearance.test.ts).
 * 손가락 URDF limit도 같은 값으로 조여 수동 조작으로도 넘지 못하게 했다.
 */
const COBOT7_GRIPPER_CLOSE = 0.05;

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

/**
 * 감지 전용 Sensor Zone — 물리 반응 없이 교차만 감지 (CLAUDE.md §5 SENSOR_ZONE).
 *
 * `detection-zone` 태그는 "통과 가능한 것이 의도"라는 **명시적 선언**이다. 태그 없는
 * sensor 정적 엔티티는 단단해 보이는데 모든 것이 지나가는 유령 장애물이고,
 * core/sample-scenes.test.ts가 그것을 결함으로 잡는다 (pick-and-place drop_zone 회귀).
 */
function sensorZoneEntity(uniquify: Uniquify): EntitySpec {
  const shape: ColliderShape = { kind: 'box', halfExtents: [...SENSOR_ZONE_HALF_EXTENTS] };
  return {
    id: uniquify('sensor_zone'),
    type: 'static',
    tags: [DETECTION_ZONE_TAG],
    transform: { position: [0, restingY(shape), 0] },
    visual: {
      kind: 'primitive',
      primitive: shape,
      color: SENSOR_ZONE_COLOR,
      opacity: SENSOR_ZONE_OPACITY,
      edges: true,
    },
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

/**
 * 컨베이어 벨트 — 고정 ENV 표면이 위에 닿은 동적 사물을 실어 나른다 (DATA_MODEL §4.2).
 *
 * `Plane · 판`과 같은 정적 ENV 계열이지만 `conveyor` 블록이 붙는다. recycle은 기본
 * **꺼짐**이다: 라이브러리로 하나 놓았을 뿐인 벨트가 사물을 순간이동시키면 놀랍다.
 * 순환 라인은 씬 데이터에서 명시적으로 켠다(conveyor-pick-place 샘플).
 */
function conveyorEntity(uniquify: Uniquify): EntitySpec {
  const shape: ColliderShape = { kind: 'box', halfExtents: [...CONVEYOR_HALF_EXTENTS] };
  return {
    id: uniquify('conveyor'),
    type: 'static',
    transform: { position: [0, restingY(shape), 0] },
    visual: { kind: 'primitive', primitive: shape, color: CONVEYOR_COLOR },
    physics: {
      bodyType: 'fixed',
      colliders: [
        {
          shape,
          friction: CONVEYOR_FRICTION,
          group: 'ENV',
          collidesWith: [...PLANE_COLLIDES_WITH],
          emitEvents: true,
        },
      ],
    },
    conveyor: {
      direction: [...CONVEYOR_DIRECTION],
      speedMps: CONVEYOR_SPEED_MPS,
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

/** SCARA-4 — 수평 2축 + 수직 직동 + 손목, 흡착 그리퍼. home 포즈로 배치 (UX §3.2) */
function scara4Entity(uniquify: Uniquify): RobotSpec {
  return {
    id: uniquify('scara'),
    type: 'robot',
    transform: { position: [0, 0, 0] },
    visual: { kind: 'urdf', ref: SCARA4_URDF_PATH },
    urdf: SCARA4_URDF_PATH,
    controller: 'sequence',
    linkColliders: 'fromVisual',
    selfCollision: false,
    home: { ...SCARA4_HOME },
    jointLimits: { ...SCARA4_JOINT_LIMITS },
    gripper: {
      joints: [...SCARA4_GRIPPER_JOINTS],
      open: SCARA4_GRIPPER_OPEN,
      close: SCARA4_GRIPPER_CLOSE,
    },
  };
}

/** Cobot-7 — 7축 여유자유도 협동로봇 + 3지 클로. home 포즈로 배치 (UX §3.2) */
function cobot7Entity(uniquify: Uniquify): RobotSpec {
  return {
    id: uniquify('cobot'),
    type: 'robot',
    transform: { position: [0, 0, 0] },
    visual: { kind: 'urdf', ref: COBOT7_URDF_PATH },
    urdf: COBOT7_URDF_PATH,
    controller: 'sequence',
    linkColliders: 'fromVisual',
    selfCollision: false,
    home: { ...COBOT7_HOME },
    gripper: {
      joints: [...COBOT7_GRIPPER_JOINTS],
      open: COBOT7_GRIPPER_OPEN,
      close: COBOT7_GRIPPER_CLOSE,
    },
  };
}

// ── 템플릿 레지스트리 ───────────────────────────────────────────────

export const LIBRARY_TEMPLATES: readonly LibraryTemplate[] = [
  {
    key: 'box',
    section: 'objects',
    labelKo: 'Box · 박스',
    icon: 'shapeBox',
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
    icon: 'shapeSphere',
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
    icon: 'shapeCylinder',
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
    icon: 'shapeCapsule',
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
    icon: 'shapePlane',
    descriptionKo: '얇은 정적 판 (50cm 박스, ENV) — 받침/작업대 용도',
    idBase: 'plane',
    create: planeEntity,
  },
  {
    key: 'conveyor',
    section: 'objects',
    labelKo: 'Conveyor · 컨베이어',
    icon: 'conveyor',
    descriptionKo: '벨트 위에 닿은 사물을 실어 나르는 정적 표면 (기본 0.12 m/s, +X 방향)',
    idBase: 'conveyor',
    create: conveyorEntity,
  },
  {
    key: 'sensor-zone',
    section: 'objects',
    labelKo: 'Sensor Zone · 감지 영역',
    icon: 'target',
    descriptionKo: '물리 반응 없이 진입/이탈만 감지하는 영역 (sensor)',
    idBase: 'sensor_zone',
    create: sensorZoneEntity,
  },
  {
    key: 'arm-6',
    section: 'robots',
    labelKo: 'Arm-6 · 6축 로봇팔',
    icon: 'robotArm',
    descriptionKo: '6-DOF 로봇팔 + 평행 그리퍼 — home 포즈로 배치',
    idBase: 'arm',
    create: arm6Entity,
  },
  {
    key: 'scara-4',
    section: 'robots',
    labelKo: 'SCARA-4 · 4축 스카라',
    icon: 'robotArm',
    descriptionKo: '수평 2축 + 수직 직동 + 손목 (작업반경 0.40m) — 흡착 패드는 누르기 전용',
    idBase: 'scara',
    create: scara4Entity,
  },
  {
    key: 'cobot-7',
    section: 'robots',
    labelKo: 'Cobot-7 · 7축 협동로봇',
    icon: 'robotArm',
    descriptionKo: '7축 여유자유도 팔 + 3지 클로 — 개구 13cm, 5cm 이하 물체용',
    idBase: 'cobot',
    create: cobot7Entity,
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
