# SIMULATION — robot-sim-web

시뮬레이션 루프, 제어 시퀀스 재생, 충돌 감지의 구현 규범. Rapier(`@dimforge/rapier3d-compat`)
API에 밀착해 기술한다. 코드는 참조용 스켈레톤이며 구현 시 이 계약을 지킨다.

---

## 1. 부트스트랩 (WASM 비동기)

```ts
import RAPIER from '@dimforge/rapier3d-compat';

export async function boot(canvas: HTMLCanvasElement, sceneJson: unknown) {
  await RAPIER.init();                          // ★ 반드시 먼저. 이후 물리 API 사용 가능
  const spec = validateScene(sceneJson);        // schema 검증
  const world = new RapierWorld(spec.gravity);  // PhysicsWorld 구현
  const render = new Renderer(canvas, spec);    // three.js
  sceneLoader.build(spec, world, render);       // 바디+메시 생성, 핸들 매핑
  const engine = new Engine(world, render, spec.timestepHz);
  return engine;                                // engine.load(sequence) 후 engine.start()
}
```

> compat 빌드를 쓰는 이유: WASM을 base64로 내장해 Vite 등 번들러의 top-level await/
> ESM-WASM 이슈를 피한다. (CLAUDE.md §9)

---

## 2. 고정 timestep 루프

프레임레이트(가변)와 물리 스텝(고정)을 분리한다. accumulator 패턴.

```ts
class Engine {
  private readonly fixedDt: number;   // 1 / timestepHz
  private acc = 0;
  private simTime = 0;
  private last = 0;
  private state: 'idle' | 'playing' | 'paused' = 'idle';

  constructor(private world: PhysicsWorld,
              private render: Renderer,
              timestepHz: number,
              private player: ControlPlayer,
              private collision: CollisionMonitor,
              private sync: RenderSync) {
    this.fixedDt = 1 / timestepHz;
  }

  start() { this.state = 'playing'; this.last = performance.now(); this.frame(this.last); }
  pause() { this.state = 'paused'; }
  resume(){ this.state = 'playing'; this.last = performance.now(); }

  /** 일시정지 상태에서 물리 1스텝만 진행(디버깅) */
  stepOnce() {
    this.tickPhysics();
    this.sync.apply(1);
    this.render.draw();
  }

  private frame = (now: number) => {
    const frameDt = Math.min((now - this.last) / 1000, 0.1); // spiral-of-death 방지
    this.last = now;

    if (this.state === 'playing') {
      this.acc += frameDt;
      while (this.acc >= this.fixedDt) {
        this.tickPhysics();
        this.acc -= this.fixedDt;
      }
    }
    const alpha = this.acc / this.fixedDt;   // 남은 시간 비율로 시각 보간
    this.sync.apply(alpha);
    this.render.draw();
    requestAnimationFrame(this.frame);
  };

  private tickPhysics() {
    this.player.step(this.simTime, this.fixedDt);   // ① 제어 → setpoint 적용
    const contacts = this.world.step();             // ② 물리 1스텝 (+EventQueue 소비)
    this.collision.dispatch(contacts, this.simTime);// ③ 충돌 이벤트 발행
    this.simTime += this.fixedDt;
  }
}
```

**핵심 규칙**
- `world.step()`에 가변 dt를 넣지 않는다. Rapier의 timestep은 `integrationParameters.dt`로
  `fixedDt`에 고정한다.
- `alpha` 보간은 **시각 전용**. 물리 상태는 항상 정확히 `fixedDt` 단위로만 전진.
- `state==='paused'`여도 렌더는 계속(카메라 조작 가능), 물리만 멈춘다.

---

## 3. 제어 시퀀스 Player

`ControlSequence`(DATA_MODEL §6)를 tick 단위로 해석한다. 순수 로직으로 단위 테스트 가능.

```ts
interface ActiveStep {
  index: number;
  elapsedSec: number;
  startJoints?: Record<string, number>;  // moveJoints 보간 시작값 스냅샷
}

class ControlPlayer {
  private cur: ActiveStep = { index: 0, elapsedSec: 0 };
  private loopCounters = new Map<number, number>(); // goto용

  constructor(private seq: ControlSequence,
              private robots: RobotHandleMap,   // 관절 setpoint 적용 인터페이스
              private events: CollisionStream) {}

  /** 매 물리 tick 호출 */
  step(simTime: number, dt: number) {
    const step = this.seq.steps[this.cur.index];
    if (!step) { this.onSequenceEnd(); return; }

    switch (step.kind) {
      case 'setJoints':
        this.applyJoints(step.robot, step.targets);
        this.advance();
        break;

      case 'moveJoints': {
        if (!this.cur.startJoints)
          this.cur.startJoints = this.robots.readJoints(step.robot ?? this.seq.robot, step.targets);
        this.cur.elapsedSec += dt;
        const t = clamp01(this.cur.elapsedSec / step.durationSec);
        const e = ease(step.easing ?? 'linear', t);
        const interp = lerpJoints(this.cur.startJoints, step.targets, e);
        this.applyJoints(step.robot, interp);
        if (t >= 1) this.advance();
        break;
      }

      case 'gripper':
        this.applyGripper(step.robot, step.state, step.durationSec ?? 0, dt);
        // durationSec 경과 시 advance (내부 상태로 판정)
        break;

      case 'wait':
        this.cur.elapsedSec += dt;
        if (this.cur.elapsedSec >= step.durationSec) this.advance();
        break;

      case 'waitForCollision':
        if (this.events.happenedSince(this.cur.startMark, step.between)) this.advance();
        else if (this.timedOut(step.timeoutSec, dt)) { warn('collision timeout'); this.advance(); }
        break;

      case 'label':
        this.advance();
        break;

      case 'goto':
        this.handleGoto(step);
        break;

      case 'moveToPose':   // 로드맵: IK 솔버 주입
        this.applyIkTarget(step);
        break;
    }
  }

  private advance() {
    this.cur = { index: this.cur.index + 1, elapsedSec: 0 };
  }

  private onSequenceEnd() {
    if (this.seq.loop) this.cur = { index: 0, elapsedSec: 0 };
    // else: idle 유지
  }
}
```

**setpoint 적용 방식 (로봇 종류별)**
- **kinematicPosition (MVP 기본)**: `robots.setJoint(name, value)`가 관절 각을 직접 설정.
  URDF joint를 kinematic으로 구동하므로 물리 반력은 받지 않지만 충돌은 감지된다.
- **dynamic (MuJoCo 경로 등)**: setpoint를 PD 목표로 사용.
  `torque = kp*(target - q) - kd*qd`. 접촉 반력이 동역학에 반영된다.

> MVP는 kinematicPosition로 단순화해 "동작 재생 + 충돌 감지"에 집중한다.
> 파지 물리가 필요해지면 dynamic + MuJoCo로 확장한다(CLAUDE.md §7).

---

## 4. 충돌 감지 (Rapier)

### 4.1 collider 셋업 (씬 로드 시)

```ts
import RAPIER from '@dimforge/rapier3d-compat';

// 그룹 비트마스크: 상위 16비트=소속(membership), 하위 16비트=필터(filter)
function interactionGroups(memberships: number, filter: number): number {
  return ((memberships & 0xffff) << 16) | (filter & 0xffff);
}
const G = { ENV:1<<0, ROBOT:1<<1, OBJECT:1<<2, SENSOR_ZONE:1<<3, DEBUG:1<<15 };

function buildCollider(world: RAPIER.World, body: RAPIER.RigidBody, c: ColliderSpec) {
  const desc = shapeToDesc(c.shape)               // box/sphere/capsule/…
    .setFriction(c.friction ?? 0.5)
    .setRestitution(c.restitution ?? 0.0)
    .setDensity(c.density ?? 1.0)
    .setSensor(c.isSensor ?? false)
    .setCollisionGroups(interactionGroups(groupBit(c.group), filterBits(c.collidesWith)));

  if (c.emitEvents) desc.setActiveEvents(RAPIER.ActiveEvents.COLLISION_EVENTS);
  if (c.ccd)        body.enableCcd(true);         // 또는 RigidBodyDesc.setCcdEnabled(true)

  const collider = world.createCollider(desc, body);
  handleToEntity.set(collider.handle, entityIdOf(body));  // ★ 핸들 매핑 등록
  return collider;
}
```

**규칙**
- 감지하려는 쌍 양쪽에 그룹/필터가 맞아야 한다. 예: 로봇 링크는 `ROBOT` 소속 +
  `collidesWith: [ENV, OBJECT]`, 박스는 `OBJECT` 소속 + `[ENV, ROBOT, OBJECT]`.
- 로봇 self-collision은 인접 링크 필터에서 제외(기본 off).
- sensor는 물리 반응 없이 교차만 이벤트로 준다(감지 영역, 게이트 등).

### 4.2 이벤트 소비 (매 스텝)

```ts
class RapierWorld implements PhysicsWorld {
  private eventQueue = new RAPIER.EventQueue(true);

  step(): ContactEvent[] {
    this.world.timestep = this.fixedDt;          // 고정
    this.world.step(this.eventQueue);            // ★ EventQueue 전달

    const out: ContactEvent[] = [];
    // (1) 충돌 시작/종료 이벤트
    this.eventQueue.drainCollisionEvents((h1, h2, started) => {
      const a = handleToEntity.get(h1);
      const b = handleToEntity.get(h2);
      if (!a || !b) return;
      const sensor = this.isSensor(h1) || this.isSensor(h2);
      out.push({ a, b, phase: started ? 'start' : 'stop',
                 kind: sensor ? 'sensor' : 'contact' });
    });

    // (2) 접촉점/법선이 필요하면 접촉 force 이벤트에서 보강(옵션)
    this.eventQueue.drainContactForceEvents((e) => {
      const a = handleToEntity.get(e.collider1());
      const b = handleToEntity.get(e.collider2());
      // out의 해당 쌍에 point/normal/magnitude 보강 …
    });

    return out;
  }
}
```

> 주의: broad-phase가 접촉 후보를 찾고 narrow-phase가 접촉점/이벤트를 만든다. 접촉점이
> 1개→다수로 바뀌는 것은 이벤트가 아니다(시작/종료 전환만 이벤트). 지속 접촉을 알려면
> `world.contactPairsWith(collider, cb)`로 폴링하거나 force 이벤트를 활용한다.

### 4.3 이벤트 발행 (CollisionMonitor)

```ts
class CollisionMonitor {
  private subscribers = new Set<(e: CollisionEvent) => void>();
  subscribe(fn: (e: CollisionEvent) => void) { this.subscribers.add(fn); }

  dispatch(contacts: ContactEvent[], simTime: number) {
    for (const c of contacts) {
      const evt: CollisionEvent = { timeSec: simTime, ...c };
      this.record(evt);                    // waitForCollision 배리어용 이력
      this.subscribers.forEach(fn => fn(evt));  // UI 로그/하이라이트, 시퀀스 배리어
    }
  }
}
```

`ui`는 여기 구독해 로그 패널 갱신 + 충돌 오브젝트 하이라이트. `player`는 이력을 조회해
`waitForCollision`을 해제.

---

## 5. Pose 동기화 (Rapier → three.js)

```ts
class RenderSync {
  // bodyId → three.Object3D, 이전/현재 pose 스냅샷 보관(보간용)
  apply(alpha: number) {
    for (const [bodyId, obj] of this.bindings) {
      const p = this.world.getPose(bodyId);         // 최신 물리 pose
      const prev = this.prevPose.get(bodyId) ?? p;
      // 위치 선형보간 + 회전 slerp
      obj.position.lerpVectors(vec(prev.position), vec(p.position), alpha);
      obj.quaternion.slerpQuaternions(quat(prev.rotation), quat(p.rotation), alpha);
    }
  }
  commit() { /* tickPhysics 후 prevPose ← curPose 갱신 */ }
}
```

- **단방향**: 물리 → 시각. three.js에서 물리 바디 pose를 역으로 쓰지 않는다.
- URDF 로봇은 링크별 Object3D가 대응 강체(또는 kinematic joint 상태)에 바인딩된다.
- 보간은 시각 매끄러움 전용이며 물리 정확도에 영향 없음.

---

## 6. 결정론 체크리스트

- [ ] `world.timestep = fixedDt` 고정, 가변 dt 미사용
- [ ] 물리 스텝 횟수는 accumulator만으로 결정(렌더 프레임레이트 무관)
- [ ] 난수 사용 시 시드 고정
- [ ] 이벤트 처리 순서 안정적(핸들 매핑 결정적)
- [ ] 초기 상태(home 관절값, 배치) 스키마로 고정

동일 SceneSpec + 동일 ControlSequence → 동일 궤적/충돌 시점을 목표로 한다.

---

## 7. MuJoCo 교체 시 바뀌는 것 / 안 바뀌는 것

| 구성요소 | Rapier(MVP) | MuJoCo 교체 후 |
|---------|-------------|----------------|
| 물리 world/스텝 | `RapierWorld` | `MujocoWorld` (`mj_step`) |
| 씬 표현 | SceneSpec→Rapier 빌더 | SceneSpec→MJCF 변환기 |
| 충돌 읽기 | `EventQueue` drain | `MjData.contact` 순회 |
| 제어 | kinematic setpoint | dynamic PD/토크 |
| **렌더/스키마/UI/시퀀스** | — | **그대로 재사용** |

이 경계를 지키는 것이 불변식이다(CLAUDE.md §7). `core`는 `PhysicsWorld`/`ContactEvent`
인터페이스만 참조하고 Rapier 심볼을 직접 노출하지 않는다.
