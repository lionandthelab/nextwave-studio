// core/types.ts — 물리 추상화 경계 (규범: docs/ARCHITECTURE.md §6, CLAUDE.md §7)
//
// core의 다른 모듈(engine/control/sync/scene-loader)과 ui는 이 인터페이스만 참조한다.
// Rapier 심볼은 core/world.ts·core/collision.ts 밖으로 새지 않는다 — 불변식.
// 미래 MuJoCo 교체 시 이 인터페이스의 새 구현(MujocoWorld)만 추가한다.

import type { BodyType, ColliderSpec, Vec3, Quat } from '../schema/types';

/** 씬 엔티티 id (SceneSpec.entities[].id) */
export type EntityId = string;

/** 물리 엔진 내부 바디 핸들 (opaque — 엔진 구현이 발급) */
export type BodyId = number;

/** 물리 엔진 내부 collider 핸들 (opaque) */
export type ColliderId = number;

export interface Pose {
  position: Vec3;
  rotation: Quat; // [x, y, z, w]
}

/** 엔진 종류 무관한 접촉 이벤트 (Rapier EventQueue / MuJoCo MjData.contact 공통 표현) */
export interface ContactEvent {
  a: EntityId;
  b: EntityId;
  phase: 'start' | 'stop';
  kind: 'contact' | 'sensor';
  point?: Vec3;
  normal?: Vec3;
}

/** 바디 생성 초기화 스펙 (schema의 PhysicsSpec에서 유도) */
export interface PhysicsBodyInit {
  bodyType: BodyType;
  position: Vec3;
  rotation?: Quat;
  linearDamping?: number;
  angularDamping?: number;
  gravityScale?: number;
  ccd?: boolean;
}

/**
 * 물리 월드 추상화. 구현: RapierWorld(core/world.ts), 미래 MujocoWorld.
 *
 * 계약:
 * - step()은 반드시 고정 dt(1/timestepHz)로만 전진한다. 가변 dt 유입 금지.
 * - step()이 반환하는 ContactEvent는 EventQueue(또는 동등 기구)에서 유도한다.
 *   메시 겹침 추정 금지 (CLAUDE.md §2.4).
 * - collider handle ↔ EntityId 매핑은 구현이 소유한 유일한 진실이다.
 */
export interface PhysicsWorld {
  /** 고정 timestep (초). 1 / timestepHz */
  readonly fixedDtSec: number;

  createBody(entityId: EntityId, init: PhysicsBodyInit): BodyId;
  createCollider(bodyId: BodyId, spec: ColliderSpec, entityId: EntityId): ColliderId;
  removeEntity(entityId: EntityId): void;

  /** kinematicPosition 바디 구동 — 다음 스텝에서 도달할 pose 지정 */
  setKinematicPose(bodyId: BodyId, pose: Pose): void;

  /** 현재 물리 pose 조회 (렌더 동기화·스냅샷용) */
  getPose(bodyId: BodyId): Pose;

  /** 강제 텔레포트 (씬 리셋/편집 전용 — 시뮬 중 사용 금지) */
  teleport(bodyId: BodyId, pose: Pose): void;

  /** 물리 1스텝 전진 + 충돌 이벤트 반환 */
  step(): ContactEvent[];

  /** collider 핸들 → 엔티티 id (충돌 이벤트 변환용) */
  entityOfCollider(collider: ColliderId): EntityId | undefined;

  /** 엔티티 id → 바디 핸들 목록 (로봇은 링크당 1개) */
  bodiesOfEntity(entityId: EntityId): readonly BodyId[];

  /** 모든 바디/collider 제거 (씬 재로드·리셋) */
  clear(): void;

  /** WASM 자원 해제 */
  free(): void;
}
