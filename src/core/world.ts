// core/world.ts — Rapier 물리 월드 래퍼 (PhysicsWorld 구현)
//
// 이 파일과 core/collision.ts만 Rapier를 import할 수 있다 — 불변식 (CLAUDE.md §3, §7).
// WASM 초기화(initPhysics)가 완료되기 전에는 어떤 물리 API도 호출하지 않는다 (§2.7).
//
// Phase 0: init + 빈 world 생성/스텝만. Phase 1에서 바디/collider/이벤트 구현 확장.

import RAPIER from '@dimforge/rapier3d-compat';
import type {
  BodyId, ColliderId, ContactEvent, EntityId, PhysicsBodyInit, PhysicsWorld, Pose,
} from './types';
import type { ColliderSpec, Vec3 } from '../schema/types';

let rapierReady = false;

/** WASM 로드. 부트스트랩 최초 1회, 모든 물리 API 사용 전 반드시 await. */
export async function initPhysics(): Promise<void> {
  if (rapierReady) return;
  await RAPIER.init();
  rapierReady = true;
}

export class RapierWorld implements PhysicsWorld {
  readonly fixedDtSec: number;
  private readonly world: RAPIER.World;

  constructor(gravity: Vec3, timestepHz: number) {
    if (!rapierReady) {
      throw new Error('RapierWorld created before initPhysics() — bootstrap order violation (CLAUDE.md §2.7)');
    }
    if (timestepHz <= 0) throw new Error(`timestepHz must be > 0, got ${timestepHz}`);
    this.fixedDtSec = 1 / timestepHz;
    this.world = new RAPIER.World({ x: gravity[0], y: gravity[1], z: gravity[2] });
    this.world.timestep = this.fixedDtSec; // 고정 timestep — 가변 dt 유입 금지
  }

  // Phase 1에서 구현 — 지금은 명시적 미구현 (호출 시 즉시 실패)
  createBody(_entityId: EntityId, _init: PhysicsBodyInit): BodyId {
    throw new Error('RapierWorld.createBody: Phase 1에서 구현');
  }
  createCollider(_bodyId: BodyId, _spec: ColliderSpec, _entityId: EntityId): ColliderId {
    throw new Error('RapierWorld.createCollider: Phase 1에서 구현');
  }
  removeEntity(_entityId: EntityId): void {
    throw new Error('RapierWorld.removeEntity: Phase 1에서 구현');
  }
  setKinematicPose(_bodyId: BodyId, _pose: Pose): void {
    throw new Error('RapierWorld.setKinematicPose: Phase 1에서 구현');
  }
  getPose(_bodyId: BodyId): Pose {
    throw new Error('RapierWorld.getPose: Phase 1에서 구현');
  }
  teleport(_bodyId: BodyId, _pose: Pose): void {
    throw new Error('RapierWorld.teleport: Phase 1에서 구현');
  }

  step(): ContactEvent[] {
    this.world.step();
    return [];
  }

  entityOfCollider(_collider: ColliderId): EntityId | undefined {
    return undefined;
  }
  bodiesOfEntity(_entityId: EntityId): readonly BodyId[] {
    return [];
  }
  clear(): void {
    // Phase 1에서 구현
  }
  free(): void {
    this.world.free();
  }
}
