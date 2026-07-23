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
}

// ── 4. Entity ───────────────────────────────────────────────────────

export interface EntitySpec {
  id: string;                    // 씬 내 유일. 충돌 로그/제어 참조 키
  type: 'robot' | 'object' | 'static';
  transform: Transform;          // 초기 배치
  visual: VisualSpec;
  physics?: PhysicsSpec;         // static 장식은 생략 가능
  tags?: string[];               // 그룹핑/쿼리용 자유 태그
}

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

export type ControlStep =
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
  | { kind: 'moveToPose'; robot?: string; target: Transform; durationSec: number };

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
