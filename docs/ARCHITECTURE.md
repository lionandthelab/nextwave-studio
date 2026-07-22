# ARCHITECTURE — robot-sim-web

## 1. 설계 원칙

- **데이터 주도**: 씬과 제어는 선언적 데이터, 엔진은 해석기. (CLAUDE.md §2.5–2.6)
- **물리가 진실**: 트랜스폼의 단일 원천은 Rapier. three.js는 물리 결과를 비추는 거울.
- **계층 격리**: 물리 엔진 의존을 한 계층에 가둬 MuJoCo로 교체 가능하게 유지.
- **결정론 지향**: 고정 timestep + accumulator로 프레임레이트와 물리를 분리.

## 2. 계층 구조

```
┌────────────────────────────────────────────────────────────┐
│  App / UI 계층  (src/ui)                                     │
│  재생 컨트롤 · 씬/시퀀스 로더 · 충돌 로그 패널 · 인스펙터     │
│  프레임워크: React 또는 vanilla (core에 침투 금지)           │
└───────────────┬────────────────────────────────────────────┘
                │  명령(load/play/pause/step) · 이벤트 구독
┌───────────────▼────────────────────────────────────────────┐
│  Core / 오케스트레이션 계층  (src/core)  ── 프레임워크 비의존 │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────┐   │
│  │ engine        │  │ control/player│  │ collision        │   │
│  │ (fixed loop)  │  │ (sequence)    │  │ (EventQueue 소비)│   │
│  └──────┬───────┘  └───────┬───────┘  └────────┬─────────┘   │
│         │                  │                    │             │
│  ┌──────▼──────────────────▼────────────────────▼─────────┐  │
│  │ world (Rapier 래퍼)  ·  sync (pose → Object3D)          │  │
│  │ scene-loader (SceneSpec → world + render + 핸들 매핑)   │  │
│  └──────┬──────────────────────────────────┬──────────────┘  │
└─────────┼──────────────────────────────────┼─────────────────┘
          │                                  │
┌─────────▼────────────┐          ┌──────────▼───────────────────┐
│  Physics 계층         │          │  Render 계층  (src/render)    │
│  Rapier (WASM)        │          │  three.js scene/camera/light  │
│  강체·collider·스텝    │          │  urdf-loader (URDF→Object3D)  │
│  EventQueue·CCD·groups │          │  OrbitControls · 디버그 헬퍼  │
└──────────────────────┘          └───────────────────────────────┘
          ▲                                  ▲
          └──────────  schema (src/schema)  ─┘
              SceneSpec · EntitySpec · ControlSequence 타입 + 런타임 검증
```

**의존 방향(엄수)**: `ui → core → {render, schema}`.
`core`는 UI 프레임워크를 import하지 않는다. Rapier를 아는 곳은 `world`/`collision`,
three.js를 아는 곳은 `render`/`sync`뿐이다.

## 3. 핵심 모듈 책임

| 모듈 | 책임 | 알아도 되는 것 |
|-----|------|--------------|
| `schema` | 데이터 타입·런타임 검증(zod 등). 순수 | 없음 (POJO) |
| `render` | three.js 씬 구성, URDF 로드, 카메라, 렌더 | three.js |
| `core/world` | Rapier world·강체·collider 생성, 스텝, **핸들↔엔티티 매핑 소유** | Rapier |
| `core/scene-loader` | `SceneSpec` → world 바디 + render 노드 생성·연결 | schema, world, render |
| `core/sync` | 스텝 후 RigidBody pose → 대응 `Object3D` 단방향 반영, 보간 | world(핸들), three.js |
| `core/control/player` | `ControlSequence` 커서 진행, step 라우팅, 관절 setpoint 산출 | schema |
| `core/control/steps` | step 종류별 핸들러(moveJoints/gripper/wait/…) | schema, robot 제어 API |
| `core/collision` | `EventQueue` 소비 → 엔티티 쌍 이벤트로 변환·발행 | world(핸들 매핑) |
| `core/engine` | 부트스트랩·고정 timestep 루프·모듈 조립·상태(play/pause) | 위 core 모듈 |
| `ui` | 명령 발행, 이벤트 구독, 시각화 패널 | core 공개 API |

## 4. 부트스트랩 순서 (고정)

WASM 비동기 초기화 때문에 순서가 중요하다. (CLAUDE.md §2.7)

```
1. await RAPIER.init()                    // compat 빌드 WASM 로드
2. const spec = validate(loadSceneJson()) // 스키마 검증
3. world = new PhysicsWorld(spec.gravity) // Rapier world 생성
4. render = new Renderer(canvas)          // three.js 셋업
5. sceneLoader.build(spec, world, render) // 바디+메시 생성, 핸들 매핑
6. player.load(controlSequence)           // 시퀀스 로드
7. engine.start()                         // 고정 timestep 루프 진입
```

## 5. 런타임 데이터 흐름 (한 프레임)

```
requestAnimationFrame(now)
  frameDt = clamp(now - last, ≤0.1s)
  accumulator += frameDt
  while accumulator ≥ FIXED_DT:
      player.step(simTime, FIXED_DT)      # ① setpoint 산출 → 로봇 관절/토크 적용
      world.step(eventQueue)              # ② Rapier 물리 1스텝
      collision.drain(eventQueue, simTime)# ③ 충돌 이벤트 → 발행(UI 로그/하이라이트)
      simTime += FIXED_DT
      accumulator -= FIXED_DT
  alpha = accumulator / FIXED_DT
  sync.apply(alpha)                       # ④ RigidBody pose → Object3D (보간)
  render.draw()                           # ⑤ three.js 렌더
```

- ①에서 로봇 종류에 따라: **kinematic(position-based)** → 관절 위치 직접 지정,
  **dynamic** → PD 컨트롤러가 목표를 향해 토크 산출(MuJoCo 경로에서 주로 사용).
  MVP 기본은 kinematic articulation으로 단순화한다.
- ③의 충돌 이벤트는 `core`에서 발행만 하고, 하이라이트 같은 시각 반응은 `ui`가 구독해 처리.

## 6. 물리 추상화 경계 (교체 가능성)

MuJoCo 교체를 위해 `core`는 Rapier 타입을 직접 노출하지 않고 얇은 인터페이스로 감싼다.

```ts
interface PhysicsWorld {
  createBody(spec: BodySpec): BodyId;
  createCollider(bodyId: BodyId, spec: ColliderSpec): ColliderId;
  setJointTarget(bodyId: BodyId, joint: string, value: number): void;
  step(): ContactEvent[];        // 엔진 종류 무관한 접촉 이벤트
  getPose(bodyId: BodyId): Pose; // position + quaternion
}

interface ContactEvent {
  a: EntityId; b: EntityId;
  phase: 'start' | 'stop';
  kind: 'contact' | 'sensor';
  point?: Vec3; normal?: Vec3;   // 가능할 때
}
```

- Rapier 구현: `RapierWorld implements PhysicsWorld` (`world.ts`).
- 미래 MuJoCo 구현: `MujocoWorld implements PhysicsWorld` + SceneSpec→MJCF 변환기.
- **불변식**: `core/engine`·`control`·`sync`·`ui`는 이 인터페이스만 참조하고
  Rapier 심볼을 직접 import하지 않는다.

## 7. 상태 관리

- **시뮬 상태**(pose, 관절값, simTime)의 진실은 물리 world. UI는 스냅샷을 구독/폴링.
- **재생 상태**(play/pause/step/speed)는 `engine`이 소유하는 유한 상태.
- **선택/하이라이트** 등 순수 UI 상태는 `ui`가 소유(물리에 영향 없음).

## 8. 확장 포인트 (열어두는 이음매)

- **실기 연동**: `PhysicsWorld` 대신/병행으로 websocket 브리지를 두면 digital-twin 가능
  (Non-goal이지만 인터페이스가 이를 허용).
- **IK**: `control/steps`에 `moveToPose`(카테시안) step + IK 솔버 주입 지점.
- **센서 시뮬**: `render`의 오프스크린 카메라로 depth/RGB 캡처 훅.
- **씬 편집기**: `ui`가 `SceneSpec`을 편집→`scene-loader` 재빌드하는 왕복 루프.
