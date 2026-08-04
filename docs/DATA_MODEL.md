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
  opacity?: number;            // 0..1 (기본 1). 감지 존을 통과 가능해 보이게 — §4.1-b
  edges?: boolean;             // 모서리 선을 덧그린다 (반투명 부피의 경계) — §4.1-b
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
  tags?: string[];             // 그룹핑/쿼리용 자유 태그 (예약: 'detection-zone')
}
```

**예약 태그 `'detection-zone'`** (`schema/types.ts`의 `DETECTION_ZONE_TAG`) — "이 정적
엔티티는 **통과 가능한 것이 의도**"라는 명시적 선언이다.

`isSensor`나 그룹만으로는 의도와 실수를 구분할 수 없다. 단단해야 할 장애물을 sensor로
만들면 화면엔 상자가 보이는데 로봇도 사물도 그냥 지나가고, `collidesWith`에 `ROBOT`이
없으면 **접촉 이벤트조차 나지 않아** 충돌 로그에도 흔적이 남지 않는다 — pick-and-place의
`drop_zone`이 정확히 그 상태였다(하나의 상자가 "도착 감지"와 "선반" 두 역할을 겸했다).
지금은 둘로 나뉘어 있다: 감지는 `drop_zone`(sensor + 태그), 실체는 `drop_shelf`(ENV 고체).

규칙(`core/sample-scenes.test.ts`가 모든 샘플 씬에 강제):
- 태그가 붙은 엔티티의 collider는 **반드시** `isSensor: true` + `SENSOR_ZONE` 그룹이다.
- 태그가 없는 `fixed` 엔티티의 collider는 sensor이면 **안 되고**, 로봇이 있는 씬이라면
  `collidesWith`에 `ROBOT`이 있고 `emitEvents: true`여야 한다.

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

### 4.1-b 감지 존의 표현 규약 (VisualSpec.opacity / edges)

`isSensor` collider는 **통과 가능**하다(§5). 그것을 불투명한 상자로 그리면 화면이
"단단한 벽"이라고 말하고, 사용자는 관통을 결함으로 읽는다(실제 보고). 그래서 감지 존은
**반투명(`opacity`) + 모서리 선(`edges`)** 으로 그린다 — 트리거 볼륨을 표현하는 3D
도구들의 공통 관례다.

```jsonc
"visual": {
  "kind": "primitive",
  "primitive": { "kind": "box", "halfExtents": [0.09, 0.006, 0.07] },
  "color": "#2fbf8f",
  "opacity": 0.22,   // 통과 가능함을 형태로 말한다
  "edges": true      // 반투명 면만으로는 묻히는 경계를 세운다
}
```

두 필드는 **순수 표현**이며 물리에 영향하지 않는다. 렌더 구현은 반투명일 때
`depthWrite`를 끄고(뒷면이 앞면을 지워 속 빈 껍데기가 되는 것 방지), 양면을 그리며,
그림자를 드리우지 않는다(그림자가 있으면 다시 단단해 보인다).

**역할이 다르면 형태와 색도 달라야 한다.** 샘플 씬의 규약: 통과 감지(포토아이)는
호박색 세로 빔 + 장식 기둥, 도착 감지는 청록색 바닥 패드. 장식 기둥은 `physics`를
갖지 않는다 — 물리를 주면 로봇 작업 반경에 새 장애물이 생긴다.

### 4.2 Conveyor (Entity 옵션 블록)

```ts
interface ConveyorSpec {
  direction: Vec3;    // 벨트 진행 방향 — **엔티티 로컬**, 수평(XZ) 성분만 사용(정규화)
  speedMps: number;   // 벨트 표면 속도 (m/s, > 0)
  recycle?: boolean;  // 끝에 도달한 사물을 시작점으로 되돌린다 (기본 false)
}
```

정적(fixed) 벨트 **표면**이 위에 닿은 동적 사물을 실어 나른다. 벨트 자신은 움직이지 않는다 —
실제 컨베이어처럼 프레임은 고정이고 표면만 흐른다.

**왜 표면 속도인가.** Rapier에는 surface velocity 개념이 없다. 대안 둘은 모두 탈락한다:
벨트 바디를 `kinematicVelocity`로 굴리면 벨트가 씬 밖으로 날아가고, 무한궤도를 여러 조각의
바디로 만들면 접촉 수가 폭발하고 조각 경계마다 사물이 걸린다. 그래서 매 물리 스텝 **직전**에
접촉 중인 동적 바디를 직접 구동한다 (`core/conveyor.ts`).

**진행축만 지정한다.** 수평 속도를 통째로 덮어쓰면 벨트가 로봇을 이긴다 — 측면 성분이 매
스텝 0으로 지워져, 팔이 사물을 벨트 밖으로 밀어도 240 Hz로 되돌려진다(실측: 2초를 밀어도
z가 5 mm). 진행축은 벨트가 지정하고 **측면·수직은 보존**한다. 실제 컨베이어도 그렇다.

**재순환은 스폰이 아니다.** "물건이 계속 온다"를 런타임 엔티티 생성으로 구현하면 SceneSpec이
진실이라는 불변식(§2.5)이 깨진다 — 스펙에 없는 엔티티가 씬에 생기고, 충돌 로그·Undo·저장·
인스펙터가 그것을 설명할 수 없다. 대신 씬에 선언된 N개가 벨트 끝에서 시작점으로 돌아간다.
되돌리기 판정은 세 조건을 **모두** 만족할 때만이다: 진행축으로 끝을 확실히 지났고, 측면으로
벨트 폭 근처이며, 벨트 상면보다 크게 높지 않다. 세 번째가 없으면 **로봇이 집어 든 사물이
손에서 사라진다**.

교차 규칙 (`validateScene`이 강제 — 전부 조용한 오작동을 막기 위한 것):

| 규칙 | 어기면 |
|-----|-------|
| `type: 'static'` | 벨트 자신이 물리에 밀려 씬을 떠난다 |
| `physics.bodyType: 'fixed'` | 같음 |
| `colliders[0].shape.kind === 'box'` | 벨트 길이/폭을 몰라 재순환 지점을 계산할 수 없다 |
| `colliders[0].isSensor !== true` | 접촉이 성립하지 않아 실어 나를 대상이 영원히 0개 |
| `direction`의 수평 성분 ≠ 0 | 정규화 불가 — 조용히 "속도 0인 벨트"가 된다 |

**기하는 빌드 시점에 고정된다** (진행축·반길이·상면 높이). 성능을 위한 선택이므로, 배치가
바뀌는 편집 경로는 바인딩을 다시 만들어야 한다 — `SceneEditor.updateTransform`이
`refreshConveyor`를, `updateDimensions/updatePhysics/renameEntity`는 재빌드가 처리한다.
이 계약이 깨지면 화면의 벨트는 옮겨졌는데 사물은 옛 자리의 진행축으로 실려 간다.

예시는 `src/assets/scenes/conveyor-pick-place.scene.json`.

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

// 모든 step 공통 옵션 필드 (Flow Graph 뷰 무손실 왕복 지원 — UX_DESIGN §6):
// - enabled: false면 player가 실행하지 않고 건너뛴다(순서 유지). 기본 true.
// - note: 사용자 메모(실행 무관).
type StepCommon = { enabled?: boolean; note?: string };

type ControlStep = StepCommon & (
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
  | { kind: 'moveToPose'; robot?: string; target: Transform; durationSec: number }
);

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
