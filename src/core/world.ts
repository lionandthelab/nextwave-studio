// core/world.ts — Rapier 물리 월드 래퍼 (PhysicsWorld 구현)
//
// 이 파일만 Rapier를 import할 수 있다 — 불변식 (CLAUDE.md §3, §7). core/collision.ts는
// 엔진 비의존 순수 모듈이다(step()이 핸들→EntityId 변환을 이미 끝낸 ContactEvent만 소비).
// WASM 초기화(initPhysics)가 완료되기 전에는 어떤 물리 API도 호출하지 않는다 (§2.7).
//
// Phase 1: 바디/collider 생성, 고정 timestep 스텝, EventQueue 기반 충돌 이벤트,
// collider handle ↔ EntityId 매핑 소유 (CLAUDE.md §2.4, §4). 규범: docs/SIMULATION.md §4.

import RAPIER from '@dimforge/rapier3d-compat';
import type {
  BodyId, ColliderId, ContactEvent, EntityId, PhysicsBodyInit, PhysicsWorld, Pose,
} from './types';
import type {
  BodyType, ColliderGroup, ColliderShape, ColliderSpec, Quat, Vec3,
} from '../schema/types';

let rapierReady = false;

/** WASM 로드. 부트스트랩 최초 1회, 모든 물리 API 사용 전 반드시 await. */
export async function initPhysics(): Promise<void> {
  if (rapierReady) return;
  await RAPIER.init();
  rapierReady = true;
}

// ── 충돌 그룹 규약 (CLAUDE.md §5) ───────────────────────────────────
// Rapier interaction group: 상위 16비트 = 소속(membership), 하위 16비트 = 필터(filter).

/** 그룹 이름 → 비트 (CLAUDE.md §5 표와 1:1. 예약 슬롯 4–14는 미배정) */
export const COLLISION_GROUP_BITS: Readonly<Record<ColliderGroup, number>> = {
  ENV: 1 << 0,
  ROBOT: 1 << 1,
  OBJECT: 1 << 2,
  SENSOR_ZONE: 1 << 3,
  DEBUG: 1 << 15,
};

/** 그룹 필드는 16비트만 유효 */
const GROUP_FIELD_MASK = 0xffff;
/** membership 필드가 차지하는 상위 비트 오프셋 */
const MEMBERSHIP_SHIFT_BITS = 16;

/**
 * Rapier InteractionGroups 패킹 (순수 함수 — 단위 테스트 대상).
 * 상위 16비트 = memberships, 하위 16비트 = filter.
 */
export function interactionGroups(memberships: number, filter: number): number {
  return ((memberships & GROUP_FIELD_MASK) << MEMBERSHIP_SHIFT_BITS) | (filter & GROUP_FIELD_MASK);
}

/** 그룹 이름 목록 → 필터 비트마스크 (순수 함수) */
export function groupsToBits(groups: readonly ColliderGroup[]): number {
  let bits = 0;
  for (const g of groups) bits |= COLLISION_GROUP_BITS[g];
  return bits;
}

/** ColliderSpec의 group/collidesWith → Rapier InteractionGroups 값 (순수 함수) */
export function collisionGroupsOf(spec: Pick<ColliderSpec, 'group' | 'collidesWith'>): number {
  return interactionGroups(COLLISION_GROUP_BITS[spec.group], groupsToBits(spec.collidesWith));
}

// ── 기본값 상수 (DATA_MODEL.md §2 — 매직넘버 금지, CLAUDE.md §4) ────

const DEFAULT_DENSITY = 1.0;
const DEFAULT_FRICTION = 0.5;
const DEFAULT_RESTITUTION = 0.0;
const IDENTITY_QUAT: Quat = [0, 0, 0, 1];

/**
 * kinematic 바디에 붙는 collider의 ActiveCollisionTypes.
 *
 * Rapier 기본(DEFAULT=15)은 DYNAMIC_* 쌍만 활성화한다 — kinematic↔fixed(ROBOT×ENV),
 * kinematic↔kinematic(selfCollision, 로봇 간) 쌍은 그룹 필터·emitEvents와 무관하게
 * narrow-phase에서 통째로 건너뛰어 접촉 이벤트가 절대 나오지 않는다.
 * 이 프로젝트에서 "어떤 쌍이 상호작용하는가"의 유일한 결정권은 충돌 그룹
 * (ColliderSpec.group/collidesWith — CLAUDE.md §5)이므로, kinematic 바디 collider에는
 * 해당 쌍 타입을 켜서 그룹 필터가 유일한 게이트가 되게 정규화한다.
 * (쌍 판정은 두 collider 타입의 합집합 기준 — 한쪽만 켜도 쌍이 활성화된다.)
 * 비트 OR 결과는 number로 넓혀지므로 enum 타입으로 되돌리는 캐스트만 수행한다.
 */
const KINEMATIC_ACTIVE_COLLISION_TYPES = (RAPIER.ActiveCollisionTypes.DEFAULT |
  RAPIER.ActiveCollisionTypes.KINEMATIC_FIXED |
  RAPIER.ActiveCollisionTypes.KINEMATIC_KINEMATIC) as RAPIER.ActiveCollisionTypes;

// ── 내부 변환 헬퍼 ──────────────────────────────────────────────────

function toVector(v: Vec3): { x: number; y: number; z: number } {
  return { x: v[0], y: v[1], z: v[2] };
}

function toRotation(q: Quat): { x: number; y: number; z: number; w: number } {
  return { x: q[0], y: q[1], z: q[2], w: q[3] };
}

function rigidBodyDescFor(bodyType: BodyType): RAPIER.RigidBodyDesc {
  switch (bodyType) {
    case 'dynamic': return RAPIER.RigidBodyDesc.dynamic();
    case 'fixed': return RAPIER.RigidBodyDesc.fixed();
    case 'kinematicPosition': return RAPIER.RigidBodyDesc.kinematicPositionBased();
    case 'kinematicVelocity': return RAPIER.RigidBodyDesc.kinematicVelocityBased();
  }
}

// Rapier capsule/cylinder는 Y축 정렬 — 프로젝트 Y-up 규약과 일치 (CLAUDE.md §4)
function colliderDescFor(shape: ColliderShape): RAPIER.ColliderDesc {
  switch (shape.kind) {
    case 'box':
      return RAPIER.ColliderDesc.cuboid(shape.halfExtents[0], shape.halfExtents[1], shape.halfExtents[2]);
    case 'sphere':
      return RAPIER.ColliderDesc.ball(shape.radius);
    case 'capsule':
      return RAPIER.ColliderDesc.capsule(shape.halfHeight, shape.radius);
    case 'cylinder':
      return RAPIER.ColliderDesc.cylinder(shape.halfHeight, shape.radius);
    case 'convexHull':
    case 'trimesh':
    case 'fromVisual':
      throw new Error(`RapierWorld: collider shape '${shape.kind}' not yet supported (Phase 3+)`);
  }
}

// ── PhysicsWorld 구현 ───────────────────────────────────────────────

export class RapierWorld implements PhysicsWorld {
  readonly fixedDtSec: number;
  private readonly world: RAPIER.World;
  private readonly eventQueue: RAPIER.EventQueue;

  // 핸들 매핑의 유일한 진실 (CLAUDE.md §4). Map은 삽입 순서를 보존한다(결정론).
  private readonly colliderToEntity = new Map<ColliderId, EntityId>();
  private readonly entityToBodies = new Map<EntityId, BodyId[]>();
  // stop 이벤트 시점엔 collider가 이미 제거됐을 수 있으므로 sensor 여부를 자체 보관.
  // (제거된 핸들의 엔트리는 다음 step()의 drain 후 flushRemovedColliders가 정리한다.)
  private readonly sensorColliders = new Set<ColliderId>();
  // 제거된 collider 핸들 → 엔티티 id tombstone. removeEntity가 등록하고 다음 step()의
  // drain이 끝나면 비운다. Rapier는 활성 접촉 중이던 collider가 제거되면 다음 스텝에서
  // 'stop' 이벤트를 발행하므로, 이 매핑이 없으면 상대 엔티티가 start/stop 짝을 잃고
  // 접촉 상태가 '고착'된다 (Phase 4 CollisionMonitor·waitForCollision 이력 소비자 보호).
  private readonly removedColliderToEntity = new Map<ColliderId, EntityId>();

  constructor(gravity: Vec3, timestepHz: number) {
    if (!rapierReady) {
      throw new Error('RapierWorld created before initPhysics() — bootstrap order violation (CLAUDE.md §2.7)');
    }
    if (timestepHz <= 0) throw new Error(`timestepHz must be > 0, got ${timestepHz}`);
    this.fixedDtSec = 1 / timestepHz;
    this.world = new RAPIER.World(toVector(gravity));
    this.world.timestep = this.fixedDtSec; // 고정 timestep — 이후 절대 변경하지 않는다
    this.eventQueue = new RAPIER.EventQueue(true);
  }

  createBody(entityId: EntityId, init: PhysicsBodyInit): BodyId {
    const desc = rigidBodyDescFor(init.bodyType)
      .setTranslation(init.position[0], init.position[1], init.position[2])
      .setRotation(toRotation(init.rotation ?? IDENTITY_QUAT));
    if (init.linearDamping !== undefined) desc.setLinearDamping(init.linearDamping);
    if (init.angularDamping !== undefined) desc.setAngularDamping(init.angularDamping);
    if (init.gravityScale !== undefined) desc.setGravityScale(init.gravityScale);
    if (init.ccd) desc.setCcdEnabled(true);

    const body = this.world.createRigidBody(desc);
    const bodies = this.entityToBodies.get(entityId);
    if (bodies) bodies.push(body.handle);
    else this.entityToBodies.set(entityId, [body.handle]);
    return body.handle;
  }

  createCollider(bodyId: BodyId, spec: ColliderSpec, entityId: EntityId): ColliderId {
    const body = this.requireBody(bodyId, 'createCollider');
    const desc = colliderDescFor(spec.shape)
      .setDensity(spec.density ?? DEFAULT_DENSITY)
      .setFriction(spec.friction ?? DEFAULT_FRICTION)
      .setRestitution(spec.restitution ?? DEFAULT_RESTITUTION)
      .setSensor(spec.isSensor ?? false)
      .setCollisionGroups(collisionGroupsOf(spec));
    // kinematic 바디(로봇 링크 등): kinematic↔fixed/kinematic 쌍도 그룹 필터가
    // 결정하도록 활성화 — 기본값이면 ROBOT×ENV 접촉이 감지되지 않는다 (상수 주석 참조)
    if (body.isKinematic()) desc.setActiveCollisionTypes(KINEMATIC_ACTIVE_COLLISION_TYPES);

    if (spec.offset) {
      const p = spec.offset.position;
      desc.setTranslation(p[0], p[1], p[2]); // 부모 바디 로컬 기준
      if (spec.offset.rotation) desc.setRotation(toRotation(spec.offset.rotation));
    }
    if (spec.emitEvents) desc.setActiveEvents(RAPIER.ActiveEvents.COLLISION_EVENTS);
    if (spec.ccd) body.enableCcd(true); // 터널링 방지 (CLAUDE.md §9)

    const collider = this.world.createCollider(desc, body);
    this.colliderToEntity.set(collider.handle, entityId); // ★ 핸들 매핑 등록
    if (spec.isSensor) this.sensorColliders.add(collider.handle);
    return collider.handle;
  }

  removeEntity(entityId: EntityId): void {
    const bodies = this.entityToBodies.get(entityId);
    if (!bodies) return;
    for (const bodyId of bodies) {
      const body: RAPIER.RigidBody | undefined = this.world.getRigidBody(bodyId);
      if (!body) continue;
      // removeRigidBody가 부속 collider도 함께 제거하므로, 매핑을 먼저 정리한다.
      // 단, 살아있는 매핑에서 지우되 tombstone으로 옮겨 둔다 — 제거로 유발되는
      // 'stop' 이벤트(다음 step의 drain)를 엔티티 쌍으로 번역해야 start/stop 짝이
      // 보존된다. sensorColliders 엔트리도 그때까지 유지한다(kind 판정용).
      const numColliders = body.numColliders();
      for (let i = 0; i < numColliders; i++) {
        const handle = body.collider(i).handle;
        this.colliderToEntity.delete(handle);
        this.removedColliderToEntity.set(handle, entityId);
      }
      this.world.removeRigidBody(body);
    }
    this.entityToBodies.delete(entityId);
  }

  setKinematicPose(bodyId: BodyId, pose: Pose): void {
    const body = this.requireBody(bodyId, 'setKinematicPose');
    body.setNextKinematicTranslation(toVector(pose.position));
    body.setNextKinematicRotation(toRotation(pose.rotation));
  }

  getPose(bodyId: BodyId): Pose {
    const body = this.requireBody(bodyId, 'getPose');
    const t = body.translation();
    const r = body.rotation();
    return { position: [t.x, t.y, t.z], rotation: [r.x, r.y, r.z, r.w] };
  }

  teleport(bodyId: BodyId, pose: Pose): void {
    const body = this.requireBody(bodyId, 'teleport');
    body.setTranslation(toVector(pose.position), true);
    body.setRotation(toRotation(pose.rotation), true);
    if (body.isDynamic()) {
      body.setLinvel(toVector([0, 0, 0]), true);
      body.setAngvel(toVector([0, 0, 0]), true);
    }
  }

  step(): ContactEvent[] {
    // timestep은 생성자에서 fixedDtSec으로 고정됨 — 가변 dt 유입 금지 (CLAUDE.md §2.3)
    this.world.step(this.eventQueue);

    const out: ContactEvent[] = [];
    this.eventQueue.drainCollisionEvents((h1, h2, started) => {
      // 살아있는 매핑 우선, 없으면 제거 tombstone(이번 drain까지 유효)에서 조회 —
      // 제거된 collider의 잔여 'stop' 이벤트도 엔티티 쌍으로 번역된다.
      const a = this.colliderToEntity.get(h1) ?? this.removedColliderToEntity.get(h1);
      const b = this.colliderToEntity.get(h2) ?? this.removedColliderToEntity.get(h2);
      if (a === undefined || b === undefined) return; // 매핑 없는 쌍은 무시
      const sensor = this.sensorColliders.has(h1) || this.sensorColliders.has(h2);
      out.push({
        a, b,
        phase: started ? 'start' : 'stop',
        kind: sensor ? 'sensor' : 'contact',
      });
    });
    this.flushRemovedColliders();
    return out;
  }

  /** 제거 tombstone 정리 — 제거 유발 'stop' 이벤트는 제거 후 첫 step()의 drain에만 온다. */
  private flushRemovedColliders(): void {
    if (this.removedColliderToEntity.size === 0) return;
    for (const handle of this.removedColliderToEntity.keys()) {
      this.sensorColliders.delete(handle);
    }
    this.removedColliderToEntity.clear();
  }

  entityOfCollider(collider: ColliderId): EntityId | undefined {
    return this.colliderToEntity.get(collider);
  }

  bodiesOfEntity(entityId: EntityId): readonly BodyId[] {
    return this.entityToBodies.get(entityId) ?? [];
  }

  clear(): void {
    const bodies: RAPIER.RigidBody[] = [];
    this.world.forEachRigidBody((body) => bodies.push(body));
    for (const body of bodies) this.world.removeRigidBody(body); // 부속 collider 포함 제거
    // 부모 없는 collider는 이 래퍼로는 생성 불가하지만, 방어적으로 잔여분도 제거한다.
    const strays: RAPIER.Collider[] = [];
    this.world.forEachCollider((collider) => strays.push(collider));
    for (const collider of strays) this.world.removeCollider(collider, false);

    this.colliderToEntity.clear();
    this.entityToBodies.clear();
    this.sensorColliders.clear();
    this.removedColliderToEntity.clear();
  }

  free(): void {
    this.eventQueue.free();
    this.world.free();
    this.colliderToEntity.clear();
    this.entityToBodies.clear();
    this.sensorColliders.clear();
    this.removedColliderToEntity.clear();
  }

  private requireBody(bodyId: BodyId, op: string): RAPIER.RigidBody {
    const body: RAPIER.RigidBody | undefined = this.world.getRigidBody(bodyId);
    if (!body) throw new Error(`RapierWorld.${op}: unknown bodyId ${bodyId}`);
    return body;
  }
}
