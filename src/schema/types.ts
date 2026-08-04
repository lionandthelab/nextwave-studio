// schema/types.ts — 데이터 모델 타입 정의 (규범: docs/DATA_MODEL.md)
// 모든 좌표는 Y-up, 미터, 라디안. 회전은 쿼터니언 [x, y, z, w].
// 이 모듈은 순수 타입(POJO)만 담는다 — 어떤 런타임 의존성도 없다.

// ── 1. 공통 원시 타입 ────────────────────────────────────────────────

export type Vec3 = [number, number, number];
export type Quat = [number, number, number, number]; // x, y, z, w (기본 [0,0,0,1])

export interface Transform {
  position: Vec3;
  rotation?: Quat; // 기본 identity
  scale?: Vec3;    // 시각 전용. 물리 collider 크기에는 영향 없음
}

// ── 2. Collider / Physics ───────────────────────────────────────────

export type ColliderShape =
  | { kind: 'box'; halfExtents: Vec3 }
  | { kind: 'sphere'; radius: number }
  | { kind: 'capsule'; halfHeight: number; radius: number }
  | { kind: 'cylinder'; halfHeight: number; radius: number }
  | { kind: 'convexHull'; ref: string }  // 에셋 메시로부터 볼록 껍질
  | { kind: 'trimesh'; ref: string }     // 정적 지오메트리 전용(동적 금지)
  | { kind: 'fromVisual' };              // 시각 메시에서 AABB/hull 자동 유도

export type BodyType =
  | 'dynamic'            // 힘/중력에 반응
  | 'fixed'              // 정적(환경)
  | 'kinematicPosition'  // 위치 지정으로 구동(로봇 기본)
  | 'kinematicVelocity'; // 속도 지정으로 구동

// CLAUDE.md §5 충돌 그룹 규약과 1:1
export type ColliderGroup = 'ENV' | 'ROBOT' | 'OBJECT' | 'SENSOR_ZONE' | 'DEBUG';

export interface ColliderSpec {
  shape: ColliderShape;
  offset?: Transform;            // 바디 로컬 기준 오프셋
  density?: number;              // 기본 1.0
  friction?: number;             // 기본 0.5
  restitution?: number;          // 반발계수, 기본 0.0
  isSensor?: boolean;            // true면 물리 반응 없이 교차만 감지
  group: ColliderGroup;          // 소속 그룹
  collidesWith: ColliderGroup[]; // 상호작용 대상 그룹
  ccd?: boolean;                 // 고속 이동체 터널링 방지
  emitEvents?: boolean;          // 충돌 이벤트 발행 여부(ActiveEvents)
}

export interface PhysicsSpec {
  bodyType: BodyType;
  colliders: ColliderSpec[];
  linearDamping?: number;
  angularDamping?: number;
  gravityScale?: number; // 기본 1.0
}

// ── 3. Visual ───────────────────────────────────────────────────────

export interface VisualSpec {
  kind: 'urdf' | 'mesh' | 'primitive';
  ref?: string;                       // urdf/mesh 파일 경로 (kind에 따라)
  primitive?: ColliderShape;          // kind==='primitive'일 때 형상
  color?: string;                     // primitive/mesh 색 (예: '#c0392b')
  packages?: Record<string, string>;  // URDF: ROS 패키지명 → 경로 매핑
  /**
   * 불투명도 0..1 (기본 1 = 불투명).
   *
   * **감지 존을 눈으로 구분하기 위해 존재한다.** sensor collider는 물리적으로 통과
   * 가능한데(§5) 불투명한 상자로 그리면 화면은 "단단한 벽"이라고 말한다 — 사용자가
   * 관통을 결함으로 읽는다(실제 사용자 보고). 통과 가능한 것은 통과 가능해 보여야 한다.
   */
  opacity?: number;
  /**
   * 형상의 모서리를 선으로 덧그린다 (기본 false).
   *
   * 반투명만으로는 얇은 부피가 배경에 묻힌다. 모서리 선이 있으면 "여기까지가 이 영역"의
   * 경계가 읽히고, 반투명 면과 합쳐져 **부피(volume)** 로 보인다 — 트리거 볼륨을 그리는
   * 3D 도구들의 공통 관례(Unity trigger gizmo, Isaac Sim trigger volume).
   */
  edges?: boolean;
}

// ── 4. Entity ───────────────────────────────────────────────────────

/**
 * 컨베이어 벨트 (DATA_MODEL §4.2).
 *
 * 정적(fixed) 벨트 **표면**이 위에 닿아 있는 동적 사물을 일정 속도로 실어 나른다.
 * 벨트 자체는 움직이지 않는다 — 실제 컨베이어와 같이 "표면 속도"만 갖는다. 물리적으로는
 * 매 스텝 접촉 중인 동적 바디의 **수평 속도**를 벨트 속도로 지정하는 액추에이터다
 * (수직 성분은 보존 — 중력/낙하가 그대로 살아 있다).
 *
 * 왜 바디를 움직이지 않는가: 벨트를 kinematicVelocity로 굴리면 벨트가 씬 밖으로 날아간다.
 * 무한궤도를 흉내 내려면 표면만 흐르게 해야 하고, Rapier에는 표면 속도 개념이 없으므로
 * 접촉 바디를 직접 구동한다 (core/conveyor.ts).
 */
export interface ConveyorSpec {
  /**
   * 벨트 진행 방향 — **엔티티 로컬** 기준. 엔티티 회전이 적용되므로 벨트를 돌리면
   * 흐름도 함께 돈다. 수평(XZ) 성분만 쓰며 내부에서 정규화한다.
   */
  direction: Vec3;
  /** 벨트 표면 속도 (m/s, > 0) */
  speedMps: number;
  /**
   * 끝에 도달한 사물을 시작점으로 되돌린다 (기본 false).
   *
   * 켜면 씬에 선언된 N개의 사물이 무한히 순환한다 — **런타임에 엔티티를 생성하지 않고**
   * "물건이 계속 오는 라인"을 표현한다. 런타임 스폰은 SceneSpec이 진실이라는 §2.5와
   * 정면으로 부딪히고(스펙에 없는 엔티티가 씬에 생긴다), 충돌 로그·Undo·저장이 모두
   * 그 엔티티를 설명할 수 없게 된다.
   */
  recycle?: boolean;
}

export interface EntitySpec {
  id: string;                    // 씬 내 유일. 충돌 로그/제어 참조 키
  type: 'robot' | 'object' | 'static';
  transform: Transform;          // 초기 배치
  visual: VisualSpec;
  physics?: PhysicsSpec;         // static 장식은 생략 가능
  tags?: string[];               // 그룹핑/쿼리용 자유 태그 (예약 태그: DETECTION_ZONE_TAG)
  conveyor?: ConveyorSpec;       // 벨트 표면 구동 (static + fixed + box collider 전용)
}

/**
 * 예약 태그 — "이 정적 엔티티는 **통과 가능한 것이 의도**다"라는 명시적 선언.
 *
 * 그룹(SENSOR_ZONE)이나 isSensor 플래그만으로는 의도와 실수를 구분할 수 없다. 단단해야
 * 할 장애물을 sensor로 만들면 화면엔 상자가 보이는데 로봇도 사물도 그냥 지나가고,
 * collidesWith에 ROBOT이 없으면 **이벤트조차 나지 않아** 충돌 로그에도 흔적이 없다
 * (pick-and-place drop_zone의 실제 결함). 태그는 사람이 "그렇게 하려던 것"이라고 적는
 * 유일한 지점이고, 태그 없는 sensor 정적 엔티티는 결함으로 간주한다
 * (core/sample-scenes.test.ts가 강제).
 */
export const DETECTION_ZONE_TAG = 'detection-zone';

export interface RobotSpec extends EntitySpec {
  type: 'robot';
  visual: VisualSpec & { kind: 'urdf' };
  urdf: string;                                    // .urdf 경로
  urdfPackages?: Record<string, string>;           // 메시 경로 매핑
  jointMap?: Record<string, string>;               // 논리명 → URDF joint명
  home?: Record<string, number>;                   // 초기 관절값(rad 또는 m)
  jointLimits?: Record<string, [number, number]>;  // 안전 클램프(옵션)
  controller: 'sequence' | 'manual';
  linkColliders?: 'fromVisual' | 'primitive' | 'none';
  selfCollision?: boolean;                         // 기본 false (인접 링크 무시)
  /** gripper 제어 step이 구동할 관절 + open/close 값 (DATA_MODEL §4.1) */
  gripper?: {
    joints: string[];
    open: number;
    close: number;
  };
}

export function isRobotSpec(e: EntitySpec): e is RobotSpec {
  return e.type === 'robot';
}

// ── 5. Scene ────────────────────────────────────────────────────────

export interface SceneSpec {
  name: string;
  version: 1;
  gravity: Vec3;      // 예: [0, -9.81, 0]
  timestepHz: number; // 고정 스텝 주파수. 예: 240
  entities: EntitySpec[]; // RobotSpec 포함
  camera?: {
    position: Vec3;
    target: Vec3;
    fov?: number;
  };
  environment?: {
    ground?: boolean; // 기본 대형 평면 바닥 생성
    skyColor?: string;
  };
}

// ── 6. Control Sequence ─────────────────────────────────────────────

export type Easing = 'linear' | 'easeInOut' | 'step';

/**
 * 모든 step 공통 옵션 (Flow Graph 무손실 왕복 — UX_DESIGN §6, DATA_MODEL §6):
 * enabled: false면 player가 실행 없이 건너뜀(순서 유지, 기본 true). note: 사용자 메모.
 */
export interface StepCommon {
  enabled?: boolean;
  note?: string;
}

export type ControlStep = StepCommon & (
  // 관절을 duration 동안 목표값으로 보간 이동
  | { kind: 'moveJoints'; robot?: string; targets: Record<string, number>;
      durationSec: number; easing?: Easing }
  // 즉시 관절값 설정(보간 없음)
  | { kind: 'setJoints'; robot?: string; targets: Record<string, number> }
  // 그리퍼 상태 (열림/닫힘 또는 0..1)
  | { kind: 'gripper'; robot?: string; state: 'open' | 'close' | number;
      durationSec?: number }
  // 지정 시간 대기
  | { kind: 'wait'; durationSec: number }
  // 두 엔티티의 충돌이 감지될 때까지 대기(안전/동기화용)
  | { kind: 'waitForCollision'; between: [string, string]; timeoutSec?: number }
  // 흐름 제어
  | { kind: 'label'; name: string }
  | { kind: 'goto'; label: string; times?: number } // times 미지정 시 무한
  // (로드맵) 카테시안 목표 — IK 솔버 주입 지점
  | { kind: 'moveToPose'; robot?: string; target: Transform; durationSec: number }
);

export type ControlStepKind = ControlStep['kind'];

export interface ControlSequence {
  id: string;
  robot: string;   // 기본 대상 로봇(step에서 override 가능)
  loop?: boolean;  // 전체 반복
  steps: ControlStep[];
}

// ── 7. Collision Event (런타임 출력 규범) ────────────────────────────

export interface CollisionEvent {
  timeSec: number;          // simTime
  a: string;                // EntityId
  b: string;                // EntityId
  phase: 'start' | 'stop';
  kind: 'contact' | 'sensor';
  point?: Vec3;             // 접촉점(가능할 때)
  normal?: Vec3;
}
