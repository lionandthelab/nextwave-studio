// core/conveyor.ts — 컨베이어 벨트 표면 구동 + 재순환 (규범: docs/DATA_MODEL.md §4.2)
//
// ── 왜 "표면 속도"인가 ──────────────────────────────────────────────
//
// 실제 컨베이어는 프레임은 고정이고 벨트 **표면**만 흐른다. 이것을 물리로 흉내 내는
// 방법은 셋뿐이다:
//
//   (a) 벨트 바디를 kinematicVelocity로 굴린다 → 벨트가 씬 밖으로 날아간다. 탈락.
//   (b) 무한궤도를 여러 조각의 바디로 만들어 돌린다 → 접촉 수가 폭발하고 조각 경계마다
//       사물이 걸린다. 프레임 예산(§2.2)을 지킬 수 없다. 탈락.
//   (c) **접촉 중인 동적 바디의 "진행축" 속도를 벨트 속도로 지정**한다 ← 채택.
//
// (c)는 Rapier에 표면 속도(surface velocity) 개념이 없기 때문에 필요한 우회다. 벨트는
// 고정 바디 그대로 두고, 매 물리 스텝 **직전**에 접촉 목록을 물리 엔진에 물어(narrow-phase가
// 진실 — AABB 추정이 아니다) 그 바디들을 구동한다.
//
// **진행축만** 건드리는 것이 핵심이다(drivenVelocity). 수평 속도를 통째로 덮어쓰면 벨트가
// 로봇을 이겨서, 팔이 사물을 벨트 밖으로 밀어도 매 스텝 되돌려진다(실측: 2초를 밀어도
// z가 5 mm). 측면·수직 성분을 보존해야 중력·낙하·로봇의 밀기가 모두 살아 있다.
//
// ── 재순환(recycle): 왜 스폰이 아니라 되돌리기인가 ──────────────────
//
// "물건이 계속 온다"를 런타임 엔티티 생성으로 구현하면 SceneSpec이 진실이라는 §2.5가
// 깨진다 — 스펙에 없는 엔티티가 씬에 생기고, 충돌 로그·Undo·저장·인스펙터가 모두 그것을
// 설명할 수 없게 된다. 대신 씬에 선언된 N개가 벨트 끝에서 시작점으로 **되돌아간다**.
// 사용자에게 보이는 결과는 같고, 데이터 모델은 온전하다.
//
// 되돌리기는 물리 스텝 **직전**(preStep)에 일어나므로 그 뒤의 sync.commit()이 새 pose를
// prev로 스냅샷한다 — 화면에 순간이동 궤적이 그려지지 않는다 (engine.ts tick 순서).
//
// ── 계층 ────────────────────────────────────────────────────────────
// core 전용: three.js/Rapier 심볼 없음. 물리는 PhysicsWorld 인터페이스로만 만진다
// (CLAUDE.md §7 — MuJoCo 교체 시 이 파일은 그대로 재사용된다).
// 아래 "순수 기하" 절의 함수들은 물리 없이 단위 테스트된다 (conveyor.test.ts).

import type { ConveyorSpec, Quat, Vec3 } from '../schema/types';
import type { BodyId, EntityId, PhysicsWorld, Pose } from './types';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4) ─────────────────────────────

const IDENTITY_QUAT: Quat = [0, 0, 0, 1];

/**
 * 벨트 끝을 이만큼 지나야 재순환한다 (m). 0으로 두면 끝에 걸친 사물이 매 스텝
 * 되돌아가며 떨리므로, 확실히 벗어난 뒤에만 되돌린다.
 */
export const RECYCLE_OVERSHOOT_M = 0.03;

/**
 * 재순환 대상의 측면 허용 폭 배수 (벨트 반폭 기준). 벨트를 벗어나 옆으로 굴러간
 * 사물까지 되돌리지는 않는다 — 그건 사고이고, 사용자가 봐야 한다.
 */
export const RECYCLE_LATERAL_FACTOR = 1.5;

/**
 * 재순환 대상의 높이 **상한** (벨트 상면 기준, m). 크게 들어 올려진 것은 벨트를 떠난
 * 것이지 끝에 도달한 것이 아니다. 다만 이 높이만으로 "로봇이 들고 있음"을 판정할 수는
 * 없다 — 10 cm만 들어 옮기는 정상 이송도 이 안에 들어온다(리뷰에서 실측으로 확인).
 * 로봇이 든 사물을 지키는 실제 장치는 아래 `RECYCLE_GRACE_TICKS`다.
 */
export const RECYCLE_HEIGHT_ABOVE_BELT_M = 0.12;

/**
 * 재순환 자격의 **유효 기간** (물리 tick). 벨트가 마지막으로 닿은 뒤 이 tick 안에 끝을
 * 지난 사물만 되돌린다.
 *
 * ★ 이것이 "로봇이 집어 든 물건을 손에서 빼앗지 않는" 진짜 규칙이다. 벨트에서 굴러
 * 떨어지는 사물은 접촉이 끊긴 직후(수 tick 안에) 끝을 지나지만, 로봇이 집어 옮기는
 * 사물은 접촉이 끊기고 **한참 뒤에** 끝을 지난다. 높이 임계값처럼 임의의 거리를 고르는
 * 대신, 두 상황이 실제로 다른 축(경과 시간)에서 판정한다.
 * 240 Hz에서 60 tick = 0.25초 — 벨트를 벗어난 사물이 낙하하는 시간보다 넉넉하고,
 * 로봇이 집어 옮기는 동작(수 초)보다는 훨씬 짧다.
 */
export const RECYCLE_GRACE_TICKS = 60;

/**
 * 재순환 대상의 높이 **하한** (벨트 상면 기준 아래로, m).
 *
 * 상한만 두면 벨트 끝을 지나 **바닥까지 떨어진** 사물도 조건을 만족해, 벨트 높이로
 * 되돌려지면서 벨트 판 안이나 아래에 박힌다. "아직 벨트 높이에 있는 것만 되돌린다"가
 * 올바른 규칙이다 — 이미 바닥에 떨어진 것은 라인을 벗어난 것이다.
 */
export const RECYCLE_HEIGHT_BELOW_BELT_M = 0.06;

/** 재순환 시 벨트 시작점에서 안쪽으로 들어오는 여유 (m) — 시작 립에 걸치지 않게 */
export const RECYCLE_INSET_M = 0.02;

// ── 순수 기하 (물리 비의존 — 단위 테스트 대상) ──────────────────────

/** 쿼터니언 q로 로컬 벡터 v를 회전한 월드 벡터 */
export function rotateVec3(q: Readonly<Quat>, v: Readonly<Vec3>): Vec3 {
  const [x, y, z, w] = q;
  const [vx, vy, vz] = v;
  // t = 2 * (q_vec × v); v' = v + w*t + q_vec × t
  const tx = 2 * (y * vz - z * vy);
  const ty = 2 * (z * vx - x * vz);
  const tz = 2 * (x * vy - y * vx);
  return [
    vx + w * tx + (y * tz - z * ty),
    vy + w * ty + (z * tx - x * tz),
    vz + w * tz + (x * ty - y * tx),
  ];
}

/**
 * 벨트 진행 방향(월드, 수평 단위벡터).
 * 로컬 direction을 엔티티 회전으로 돌린 뒤 XZ 평면에 투영해 정규화한다.
 * 수평 성분이 0이면 null — validateScene이 이미 거부하지만 런타임 방어를 남긴다.
 */
export function beltForward(spec: ConveyorSpec, rotation: Readonly<Quat>): Vec3 | null {
  const world = rotateVec3(rotation, spec.direction);
  const len = Math.hypot(world[0], world[2]);
  if (len === 0) return null;
  return [world[0] / len, 0, world[2] / len];
}

/** 수평 좌우 축 = forward × up (Y-up) — 재순환의 측면 판정에 쓴다 */
export function beltLateral(forward: Readonly<Vec3>): Vec3 {
  // forward × (0,1,0) = (fz, 0, -fx)
  return [forward[2], 0, -forward[0]];
}

/**
 * 축정렬 박스 half extents를 임의의 수평 단위축 u에 투영한 반길이.
 * 회전한 벨트에서도 "진행 방향으로 얼마나 긴가"를 정확히 준다.
 */
export function extentAlong(
  halfExtents: Readonly<Vec3>,
  axis: Readonly<Vec3>,
  rotation: Readonly<Quat>,
): number {
  // 월드 축 u를 벨트 로컬로 되돌린 뒤(Rᵀu) half extents와 |·| 내적
  const inv: Quat = [-rotation[0], -rotation[1], -rotation[2], rotation[3]];
  const local = rotateVec3(inv, axis);
  return (
    Math.abs(local[0]) * halfExtents[0] +
    Math.abs(local[1]) * halfExtents[1] +
    Math.abs(local[2]) * halfExtents[2]
  );
}

/** 벨트 로컬 좌표계에서 본 한 점 (진행축 s, 측면축 t, 월드 높이 y) */
export interface BeltCoords {
  /** 벨트 중심 기준 진행 방향 좌표 (m) */
  along: number;
  /** 벨트 중심 기준 측면 좌표 (m) */
  lateral: number;
  /** 월드 y (m) */
  worldY: number;
}

export function beltCoordsOf(
  point: Readonly<Vec3>,
  beltCenter: Readonly<Vec3>,
  forward: Readonly<Vec3>,
  lateral: Readonly<Vec3>,
): BeltCoords {
  const dx = point[0] - beltCenter[0];
  const dz = point[2] - beltCenter[2];
  return {
    along: dx * forward[0] + dz * forward[2],
    lateral: dx * lateral[0] + dz * lateral[2],
    worldY: point[1],
  };
}

/** 벨트 기하 스냅샷 (빌드 시점에 확정 — 매 스텝 재계산하지 않는다) */
export interface BeltGeometry {
  /** 진행 방향 (월드, 수평 단위) */
  readonly forward: Vec3;
  /** 측면 축 (월드, 수평 단위) */
  readonly lateral: Vec3;
  /** 진행축 반길이 (m) */
  readonly halfLength: number;
  /** 측면축 반폭 (m) */
  readonly halfWidth: number;
  /** 벨트 상면의 월드 y (m) */
  readonly topY: number;
}

/**
 * 재순환 판정 (순수). 조건 셋을 **모두** 만족해야 되돌린다:
 *
 * 1. 진행축으로 벨트 끝을 확실히 지났다 (`RECYCLE_OVERSHOOT_M`)
 * 2. 측면으로 벨트 폭 근처에 있다 — 옆으로 굴러 나간 사고품은 되돌리지 않는다
 * 3. **아직 벨트 높이에 있다** (상면 기준 위아래 밴드 안)
 *
 * 3번의 상한이 없으면 로봇이 사물을 들고 벨트 끝을 지나는 순간 손에서 사물이 사라지고,
 * 하한이 없으면 벨트 끝에서 바닥까지 떨어진 사물이 벨트 높이로 되돌려져 판에 박힌다.
 */
export function shouldRecycle(
  position: Readonly<Vec3>,
  beltCenter: Readonly<Vec3>,
  geometry: BeltGeometry,
): boolean {
  const coords = beltCoordsOf(position, beltCenter, geometry.forward, geometry.lateral);
  if (coords.along <= geometry.halfLength + RECYCLE_OVERSHOOT_M) return false;
  if (Math.abs(coords.lateral) > geometry.halfWidth * RECYCLE_LATERAL_FACTOR) return false;
  if (position[1] > geometry.topY + RECYCLE_HEIGHT_ABOVE_BELT_M) return false;
  return position[1] >= geometry.topY - RECYCLE_HEIGHT_BELOW_BELT_M;
}

/**
 * 재순환 목적지 (순수): 같은 측면 위치·같은 높이를 유지한 채 진행축만 벨트 시작점으로.
 * 높이를 보존하므로 벨트 위를 흐르던 사물은 벨트 위로 돌아온다.
 */
export function recycleTarget(
  position: Readonly<Vec3>,
  beltCenter: Readonly<Vec3>,
  geometry: BeltGeometry,
): Vec3 {
  const coords = beltCoordsOf(position, beltCenter, geometry.forward, geometry.lateral);
  const startAlong = -geometry.halfLength + RECYCLE_INSET_M;
  return [
    beltCenter[0] + geometry.forward[0] * startAlong + geometry.lateral[0] * coords.lateral,
    position[1],
    beltCenter[2] + geometry.forward[2] * startAlong + geometry.lateral[2] * coords.lateral,
  ];
}

/**
 * 벨트가 구동한 뒤의 속도 — **진행축 성분만** 벨트 속도로 지정하고 나머지는 보존한다.
 *
 * ★ 이 분해가 없으면 벨트가 로봇을 이긴다. 수평 속도를 통째로 벨트 속도로 덮어쓰면
 * 측면(lateral) 성분이 매 스텝 0으로 지워져, 로봇이 사물을 벨트 밖으로 밀어내려 해도
 * 240 Hz로 되돌려진다 — 실측에서 팔이 2초를 밀어도 z가 5 mm밖에 움직이지 않았다.
 *
 * 실제 컨베이어도 진행 방향으로만 끌고, 옆으로 밀면 (마찰을 이기는 한) 옆으로 간다.
 * 그래서 축을 셋으로 나눈다: 진행축 = 벨트가 지정, 측면축 = 보존, 수직 = 보존.
 *
 * ── 알려진 특성: 실제 속도는 선언값보다 조금 낮다 ──────────────────
 * 속도 지정은 `preStep`(스텝 **직전**)에 일어나고, 그 뒤 `world.step()`의 접촉 솔버가
 * **정지해 있는 벨트 면과의 마찰**로 사물을 다시 붙잡는다. 그래서 스텝 후 관측되는
 * 속도는 선언 속도의 약 85~90%다(실측: 0.25 m/s 선언 → 0.221 m/s 관측).
 * 실제 컨베이어는 표면 자체가 움직여 이 손실이 없지만, Rapier에 표면 속도가 없는 한
 * 우회의 대가로 남는다. **선언값은 상한**으로 읽는 것이 정확하다.
 */
export function drivenVelocity(
  current: Readonly<Vec3>,
  forward: Readonly<Vec3>,
  lateral: Readonly<Vec3>,
  speedMps: number,
): Vec3 {
  // 현재 수평 속도의 측면 성분만 남긴다 (진행축 성분은 벨트 값으로 대체)
  const lateralSpeed = current[0] * lateral[0] + current[2] * lateral[2];
  return [
    forward[0] * speedMps + lateral[0] * lateralSpeed,
    current[1],
    forward[2] * speedMps + lateral[2] * lateralSpeed,
  ];
}

// ── 바인딩 / 레지스트리 (물리 접촉) ─────────────────────────────────

export interface ConveyorBindingDeps {
  readonly world: PhysicsWorld;
  readonly entityId: EntityId;
  readonly spec: ConveyorSpec;
  /** 벨트 표면 박스의 half extents (collider[0].shape — validateScene이 box를 보장) */
  readonly halfExtents: Vec3;
  /** 벨트 엔티티의 배치 (회전 포함) */
  readonly pose: Pose;
}

/**
 * 벨트 한 대. `tick()`이 매 물리 스텝 직전에 (1) 접촉 중인 동적 바디를 구동하고
 * (2) recycle이 켜져 있으면 끝을 지난 사물을 시작점으로 되돌린다.
 *
 * 벨트 자신의 pose는 **빌드/편집 시점에 고정**된 값을 쓴다 — 벨트는 fixed 바디이므로
 * 시뮬 중 움직이지 않고, 매 스텝 pose를 다시 읽는 것은 낭비다. 편집으로 벨트가 움직이면
 * 통합자가 엔티티를 재빌드하므로 새 바인딩이 새 pose로 만들어진다.
 */
export class ConveyorBinding {
  readonly entityId: EntityId;
  private readonly world: PhysicsWorld;
  private readonly spec: ConveyorSpec;
  private readonly center: Vec3;
  private readonly geometry: BeltGeometry | null;
  /**
   * 이 벨트가 마지막으로 실어 나른 tick (바디별) — **재순환 대상은 여기 있는 것뿐**이다.
   *
   * 전체 동적 바디를 후보로 삼으면, 벨트 끝 너머에 놓인 다른 사물(다른 벨트 위의 물건,
   * 배출 구역에 쌓인 상자)이 이 벨트의 시작점으로 순간이동한다 — 씬에 벨트가 둘 이상이면
   * 바로 재현된다(리뷰 실측: 다른 벨트 위 사물이 첫 tick에 끌려왔다).
   * tick 번호까지 기억하는 이유는 `RECYCLE_GRACE_TICKS` 주석 참조.
   */
  private readonly lastCarriedTick = new Map<BodyId, number>();
  /** 이 바인딩이 돈 tick 수 (재순환 자격의 유효 기간 판정 기준) */
  private tickIndex = 0;
  /**
   * 다음 tick의 **구동**을 한 번 건너뛴다. 씬 리셋 직후에 켠다.
   *
   * 접촉 그래프는 마지막 `world.step()`의 결과다. reset()은 바디를 teleport하지만
   * 스텝을 돌리지 않으므로, 바로 다음 preStep의 접촉 목록은 **리셋 전 배치**의 것이다.
   * 그대로 구동하면 공중에 있는 사물에 벨트 속도가 주입돼 되감기 재생이 최초 재생과
   * 달라진다(리뷰 실측: 26 cm 상공의 사물에 0.1 m/s 주입 → 결정론 위반).
   */
  private skipDriveOnce = false;

  constructor(deps: ConveyorBindingDeps) {
    this.world = deps.world;
    this.entityId = deps.entityId;
    this.spec = deps.spec;
    const rotation = deps.pose.rotation ?? IDENTITY_QUAT;
    this.center = [deps.pose.position[0], deps.pose.position[1], deps.pose.position[2]];

    const forward = beltForward(deps.spec, rotation);
    if (forward === null) {
      // validateScene이 막지만, 파사드/편집 경로로 들어온 값에 대한 런타임 방어.
      // 기하가 없으면 tick()은 완전한 no-op이 된다 (조용한 오작동 대신 무동작).
      this.geometry = null;
      return;
    }
    const lateral = beltLateral(forward);
    this.geometry = {
      forward,
      lateral,
      halfLength: extentAlong(deps.halfExtents, forward, rotation),
      halfWidth: extentAlong(deps.halfExtents, lateral, rotation),
      topY: deps.pose.position[1] + extentAlong(deps.halfExtents, [0, 1, 0], rotation),
    };
  }

  /** 벨트 기하 스냅샷 (검증/게이트용 — null이면 방향이 퇴화해 비활성 상태다) */
  get beltGeometry(): BeltGeometry | null {
    return this.geometry;
  }

  /**
   * 씬 리셋 훅 — 실어 나른 이력을 버리고 다음 tick의 구동을 한 번 건너뛴다
   * (`skipDriveOnce` 주석의 stale 접촉 그래프 문제).
   */
  reset(): void {
    this.lastCarriedTick.clear();
    this.skipDriveOnce = true;
  }

  /**
   * Engine preStep 훅 대상 — 매 물리 tick, world.step() 전에 1회.
   * @param claimed 이번 tick에 **다른 벨트가 이미 구동한** 바디 (겹친 벨트 중재)
   */
  tick(claimed?: Set<BodyId>): void {
    this.tickIndex += 1;
    const geometry = this.geometry;
    if (geometry === null) return;

    // (1) 표면 구동: 접촉 중인 동적 바디의 **진행축 속도만** 벨트 속도로 지정한다.
    //     측면·수직은 보존 — 로봇이 사물을 벨트 밖으로 밀 수 있고(drivenVelocity 주석),
    //     벨트에 내려앉는 중이면 계속 내려앉는다.
    if (this.skipDriveOnce) {
      this.skipDriveOnce = false;
    } else {
      for (const bodyId of this.world.dynamicBodiesTouching(this.entityId)) {
        // 겹친 벨트 중재: 이번 tick에 다른 벨트가 이미 구동했으면 건드리지 않는다.
        // 그러지 않으면 앞 벨트의 구동이 이 벨트의 "보존할 측면 성분"으로 오인돼
        // 두 속도가 합성된다 (리뷰 실측: 0.25 m/s 벨트 둘이 만나 0.336 m/s).
        // 승자는 레지스트리 삽입 순서 = 씬 선언 순서로 결정론적이다.
        if (claimed?.has(bodyId) === true) continue;
        claimed?.add(bodyId);
        this.lastCarriedTick.set(bodyId, this.tickIndex);
        const current = this.world.getLinearVelocity(bodyId);
        this.world.setLinearVelocity(
          bodyId,
          drivenVelocity(current, geometry.forward, geometry.lateral, this.spec.speedMps),
        );
      }
    }

    // (2) 재순환: 끝을 지난 사물을 시작점으로. 되돌릴 시점에는 접촉이 이미 끊긴 뒤라
    //     (1)의 목록에 없다 — 그래서 "이 벨트가 실어 본 적 있는" 바디를 후보로 쓴다.
    if (this.spec.recycle !== true) return;
    for (const [bodyId, touchedAt] of this.lastCarriedTick) {
      // 자격 만료: 벨트에서 손을 뗀 지 오래된 사물은 이 벨트의 소관이 아니다
      // (로봇이 집어 옮기는 중인 물건 — RECYCLE_GRACE_TICKS 주석)
      if (this.tickIndex - touchedAt > RECYCLE_GRACE_TICKS) continue;
      const pose = this.world.getPose(bodyId);
      if (!shouldRecycle(pose.position, this.center, geometry)) continue;
      this.world.teleport(bodyId, {
        position: recycleTarget(pose.position, this.center, geometry),
        rotation: pose.rotation,
      });
    }
  }
}

/**
 * 씬의 컨베이어 보관소. RobotRegistry와 같은 계약이다 — SceneHandle이 소유하고
 * Engine preStep 훅이 tickAll()을 부른다.
 */
export class ConveyorRegistry {
  private readonly bindings = new Map<EntityId, ConveyorBinding>();

  add(binding: ConveyorBinding): void {
    if (this.bindings.has(binding.entityId)) {
      throw new Error(`conveyor: 컨베이어 '${binding.entityId}'이(가) 이미 등록되어 있습니다`);
    }
    this.bindings.set(binding.entityId, binding);
  }

  get(entityId: EntityId): ConveyorBinding | undefined {
    return this.bindings.get(entityId);
  }

  /** 등록 해제 — 미등록 id는 no-op (dispose 경로 멱등성, RobotRegistry.remove와 동일) */
  remove(entityId: EntityId): void {
    this.bindings.delete(entityId);
  }

  /** 등록된 컨베이어 엔티티 id 목록 (삽입 순서) */
  ids(): string[] {
    return [...this.bindings.keys()];
  }

  /**
   * Engine preStep 훅 대상 — 매 물리 tick, world.step() 전에 1회 호출한다.
   *
   * 겹친 벨트가 같은 사물을 두 번 구동하지 않도록 tick마다 "이미 구동된 바디" 집합을
   * 공유한다 — 먼저 선언된 벨트가 이긴다(삽입 순서 = 씬 선언 순서, 결정론적).
   */
  tickAll(): void {
    const claimed = new Set<BodyId>();
    for (const binding of this.bindings.values()) binding.tick(claimed);
  }

  /** 씬 리셋 훅 — 모든 벨트의 이송 이력을 버린다 (ConveyorBinding.reset 주석) */
  resetAll(): void {
    for (const binding of this.bindings.values()) binding.reset();
  }
}
