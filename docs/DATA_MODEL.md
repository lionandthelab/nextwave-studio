# DATA_MODEL — robot-sim-web

씬과 제어 시퀀스의 선언적 스키마를 정의한다. 이 문서가 `src/schema`의 규범(source of
truth)이다. 모든 좌표는 **Y-up, 미터, 라디안**, 회전은 쿼터니언 `[x,y,z,w]`.

---

## 1. 공통 원시 타입

```ts
type Vec3 = [number, number, number];
type Quat = [number, number, number, number];   // x, y, z, w (기본 [0,0,0,1])

interface Transform {
  position: Vec3;
  rotation?: Quat;      // 기본 identity
  scale?: Vec3;         // 시각 전용. 물리 collider 크기에는 영향 없음
}
```

---

## 2. Collider / Physics

```ts
type ColliderShape =
  | { kind: 'box';      halfExtents: Vec3 }
  | { kind: 'sphere';   radius: number }
  | { kind: 'capsule';  halfHeight: number; radius: number }
  | { kind: 'cylinder'; halfHeight: number; radius: number }
  | { kind: 'convexHull'; ref: string }   // 에셋 메시로부터 볼록 껍질
  | { kind: 'trimesh';    ref: string }    // 정적 지오메트리 전용(동적 금지 권장)
  | { kind: 'fromVisual' };                // 시각 메시에서 AABB/hull 자동 유도

type BodyType =
  | 'dynamic'                  // 힘/중력에 반응
  | 'fixed'                    // 정적(환경)
  | 'kinematicPosition'        // 위치 지정으로 구동(로봇 기본)
  | 'kinematicVelocity';       // 속도 지정으로 구동

interface ColliderSpec {
  shape: ColliderShape;
  offset?: Transform;          // 바디 로컬 기준 오프셋
  density?: number;            // 기본 1.0
  friction?: number;           // 기본 0.5
  restitution?: number;        // 반발계수, 기본 0.0
  isSensor?: boolean;          // true면 물리 반응 없이 교차만 감지
  group: ColliderGroup;        // 소속 그룹 (CLAUDE.md §5)
  collidesWith: ColliderGroup[]; // 상호작용 대상 그룹
  ccd?: boolean;               // 고속 이동체 터널링 방지
  emitEvents?: boolean;        // 충돌 이벤트 발행 여부(ActiveEvents)
}

type ColliderGroup =
  | 'ENV' | 'ROBOT' | 'OBJECT' | 'SENSOR_ZONE' | 'DEBUG';  // §5 표와 1:1

interface PhysicsSpec {
  bodyType: BodyType;
  colliders: ColliderSpec[];
  linearDamping?: number;
  angularDamping?: number;
  gravityScale?: number;       // 기본 1.0
}
```

**규칙**
- `trimesh`는 동적 바디에 쓰지 않는다(안정성/성능). 동적은 프리미티브나 `convexHull`.
- 충돌을 감지하려는 collider는 `emitEvents: true` + 상대 그룹을 `collidesWith`에 포함.
- 감지만 하고 튕기지 않으려면 `isSensor: true`(그룹은 보통 `SENSOR_ZONE`).

---

## 3. Visual

```ts
interface VisualSpec {
  kind: 'urdf' | 'mesh' | 'primitive';
  ref?: string;                // urdf/mesh 파일 경로 (kind에 따라)
  primitive?: ColliderShape;   // kind==='primitive'일 때 형상
  color?: string;              // primitive/mesh 색 (예: '#c0392b')
  packages?: Record<string, string>; // URDF: ROS 패키지명 → 경로 매핑
}
```

---

## 4. Entity

```ts
interface EntitySpec {
  id: string;                  // 씬 내 유일. 충돌 로그/제어 참조 키
  type: 'robot' | 'object' | 'static';
  transform: Transform;        // 초기 배치
  visual: VisualSpec;
  physics?: PhysicsSpec;       // static 장식은 생략 가능
  tags?: string[];             // 그룹핑/쿼리용 자유 태그
}
```

### 4.1 Robot (Entity 확장)

```ts
interface RobotSpec extends EntitySpec {
  type: 'robot';
  visual: VisualSpec & { kind: 'urdf' };
  urdf: string;                          // .urdf 경로
  urdfPackages?: Record<string, string>; // 메시 경로 매핑
  jointMap?: Record<string, string>;     // 논리명 → URDF joint명
  home?: Record<string, number>;         // 초기 관절값(rad 또는 m)
  jointLimits?: Record<string, [number, number]>; // 안전 클램프(옵션)
  controller: 'sequence' | 'manual';
  // 관절 자동 collider 생성 정책
  linkColliders?: 'fromVisual' | 'primitive' | 'none';
  selfCollision?: boolean;               // 기본 false (인접 링크 무시)
  // 그리퍼 구성: `gripper` 제어 step이 구동할 관절과 open/close 값 매핑.
  // 값은 각 관절에 동일하게 적용된다(평행 그리퍼). 0..1 상태는 close↔open 선형 보간.
  gripper?: {
    joints: string[];                    // URDF 관절명(또는 jointMap 논리명)
    open: number;                        // 열림 상태 관절값 (예: 0.03)
    close: number;                       // 닫힘 상태 관절값 (예: 0.0)
  };
}
```

---

## 5. Scene

```ts
interface SceneSpec {
  name: string;
  version: 1;
  gravity: Vec3;               // 예: [0, -9.81, 0]
  timestepHz: number;          // 고정 스텝 주파수. 예: 240
  entities: EntitySpec[];      // RobotSpec 포함
  camera?: {
    position: Vec3; target: Vec3; fov?: number;
  };
  environment?: {
    ground?: boolean;          // 기본 무한/대형 평면 바닥 생성
    skyColor?: string;
  };
}
```

---

## 6. Control Sequence

제어는 **선언적 step 배열**. player가 커서와 경과시간을 들고 해석한다(SIMULATION.md §3).

```ts
type Easing = 'linear' | 'easeInOut' | 'step';

type ControlStep =
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
  | { kind: 'goto'; label: string; times?: number }   // times 미지정 시 무한
  // (로드맵) 카테시안 목표 — IK 솔버 주입 지점
  | { kind: 'moveToPose'; robot?: string; target: Transform; durationSec: number };

interface ControlSequence {
  id: string;
  robot: string;               // 기본 대상 로봇(step에서 override 가능)
  loop?: boolean;              // 전체 반복
  steps: ControlStep[];
}
```

**해석 규칙**
- `moveJoints`는 시작값→목표값을 `durationSec` 동안 `easing`으로 보간해 매 tick setpoint 산출.
- kinematic 로봇: setpoint를 관절 위치로 직접 반영. dynamic 로봇: PD 목표로 사용.
- `waitForCollision`은 `collision` 이벤트 스트림을 구독하는 배리어. timeout 초과 시 경고 후 진행.
- `goto/label`은 최소 루프 제어만 제공(복잡 로직은 여러 시퀀스로 분리 권장).
- 미지정 값은 관절 이름이 로봇의 `jointMap`/URDF joint와 일치해야 한다(검증 시 확인).

---

## 7. Collision Event (런타임 출력)

스키마 입력이 아니라 `collision` 모듈이 발행하는 이벤트 형태(구독자/로그 규범).

```ts
interface CollisionEvent {
  timeSec: number;             // simTime
  a: string;                   // EntityId
  b: string;                   // EntityId
  phase: 'start' | 'stop';
  kind: 'contact' | 'sensor';
  point?: Vec3;                // 접촉점(가능할 때)
  normal?: Vec3;
}
```

---

## 8. 런타임 검증

- `src/schema`는 위 타입 + zod(또는 동등) 스키마로 **로드 시 검증**한다.
- 검증 항목: id 유일성, 참조 무결성(제어 시퀀스의 robot/joint가 씬에 존재),
  그룹/`collidesWith` 유효성, `trimesh`+dynamic 금지, timestepHz>0, gravity 유한.
- 검증 실패는 씬 로드를 중단하고 사람이 읽을 수 있는 오류를 UI에 표시한다.

---

## 9. 최소 예시

### 9.1 SceneSpec (로봇 + 박스 + 바닥)

```json
{
  "name": "collision-testbed",
  "version": 1,
  "gravity": [0, -9.81, 0],
  "timestepHz": 240,
  "environment": { "ground": true, "skyColor": "#1b1e23" },
  "camera": { "position": [1.5, 1.2, 1.5], "target": [0, 0.4, 0] },
  "entities": [
    {
      "id": "arm",
      "type": "robot",
      "transform": { "position": [0, 0, 0] },
      "visual": { "kind": "urdf", "ref": "assets/arm/arm.urdf" },
      "urdf": "assets/arm/arm.urdf",
      "urdfPackages": { "arm_description": "assets/arm" },
      "home": { "joint1": 0.0, "joint2": -0.6, "joint3": 1.2 },
      "controller": "sequence",
      "linkColliders": "fromVisual",
      "selfCollision": false
    },
    {
      "id": "box_a",
      "type": "object",
      "transform": { "position": [0.4, 0.05, 0.0] },
      "visual": { "kind": "primitive", "primitive": { "kind": "box", "halfExtents": [0.05,0.05,0.05] }, "color": "#c0392b" },
      "physics": {
        "bodyType": "dynamic",
        "colliders": [{
          "shape": { "kind": "box", "halfExtents": [0.05,0.05,0.05] },
          "friction": 0.6, "group": "OBJECT",
          "collidesWith": ["ENV","ROBOT","OBJECT"],
          "emitEvents": true
        }]
      }
    }
  ]
}
```

### 9.2 ControlSequence (픽 동작 + 충돌 대기)

```json
{
  "id": "reach-and-touch",
  "robot": "arm",
  "loop": false,
  "steps": [
    { "kind": "moveJoints", "targets": { "joint2": 0.2, "joint3": 0.6 }, "durationSec": 2.0, "easing": "easeInOut" },
    { "kind": "waitForCollision", "between": ["arm", "box_a"], "timeoutSec": 5.0 },
    { "kind": "gripper", "state": "close", "durationSec": 0.5 },
    { "kind": "wait", "durationSec": 1.0 },
    { "kind": "moveJoints", "targets": { "joint2": -0.6, "joint3": 1.2 }, "durationSec": 2.0 }
  ]
}
```
