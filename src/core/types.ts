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

  /**
   * 접촉 상태(워밍스타트 임펄스·접촉 매니폴드)를 버린다 — **되감기 전용**.
   *
   * 바디를 teleport해도 솔버의 접촉 매니폴드는 남는다. 거기 누적된 임펄스가 다음 스텝의
   * 워밍스타트 초기값이 되므로, "같은 좌표에서 다시 시작"해도 **직전에 무슨 일이
   * 있었는지에 따라 결과가 달라진다**. 실측(A/B, 각 3회): 새 월드에서 재생하면 최종 좌표가
   * 소수점 5자리까지 3회 동일한데, 15초 돌린 뒤 되감아 재생하면 3회 중 2회가 달랐다
   * (한 번은 완주 54.4초, 정상 24.9초).
   *
   * 구현은 collider를 제거·재생성해 좁은 단계(narrow phase)를 비우는 것이다.
   *
   * **이것이 보장하는 것과 아닌 것**: 되감기 후 재생이 **매번 같은 결과**를 낸다
   * (수정 후 실측 3/3 동일, 수정 전 3회 중 2회 상이). 그러나 그 결과가 **새로 연 페이지의
   * 재생과 같다는 보장은 아니다** — collider 재생성이 Rapier 내부 핸들·순회 순서를 바꾸므로
   * 되감기 재생은 자기들끼리만 재현된다(실측: 컨베이어 씬에서 되감기 재생이 54.4s,
   * 새 월드가 24.8s — item_c의 배리어 두 개에서 갈린다). 완전 동등성은 월드 재빌드가 필요하고,
   * 그건 별도 작업이다. 자동화·게이트가 **새 월드와 같은 결과**를 원하면 페이지를 새로 열어야 한다.
   */
  clearContactState(): void;

  /** kinematicPosition 바디 구동 — 다음 스텝에서 도달할 pose 지정 */
  setKinematicPose(bodyId: BodyId, pose: Pose): void;

  /** 현재 물리 pose 조회 (렌더 동기화·스냅샷용) */
  getPose(bodyId: BodyId): Pose;

  /** 강제 텔레포트 (씬 리셋/편집 전용 — 시뮬 중 사용 금지) */
  teleport(bodyId: BodyId, pose: Pose): void;

  /** 물리 1스텝 전진 + 충돌 이벤트 반환 */
  step(): ContactEvent[];

  /**
   * 같은 엔티티에 속한 collider끼리의 접촉 이벤트 발행 여부 (기본 false).
   *
   * 로봇 한 대의 모든 링크는 하나의 EntityId를 공유하므로, "같은 엔티티 접촉" = 자기
   * 충돌(self-collision)이다. 기본 false는 URDF 인접 링크가 구조상 항상 겹쳐 발생시키는
   * 상시 접촉 노이즈를 억제한다 (CLAUDE.md §5 "self-collision 기본 비활성").
   *
   * **서로 다른 엔티티(다른 로봇) 간 충돌은 이 설정과 무관하게 항상 발행된다** —
   * 그룹 비트마스크는 로봇 개체를 구분하지 못하므로 그룹이 아닌 엔티티 단위로 가른다.
   */
  setSelfContactEnabled(entityId: EntityId, enabled: boolean): void;

  /** collider 핸들 → 엔티티 id (충돌 이벤트 변환용) */
  entityOfCollider(collider: ColliderId): EntityId | undefined;

  /** 엔티티 id → 바디 핸들 목록 (로봇은 링크당 1개) */
  bodiesOfEntity(entityId: EntityId): readonly BodyId[];

  /**
   * 이 엔티티의 collider와 **현재 접촉 중인** 다른 엔티티의 동적 바디 목록.
   *
   * 컨베이어처럼 "표면에 닿은 것을 구동하는" 액추에이터를 위한 조회다. 접촉 판정은
   * 물리 엔진의 narrow-phase가 유일한 진실이며, 메시/AABB 겹침 추정이 아니다
   * (CLAUDE.md §2.4의 정신 — 다만 이것은 이벤트가 아니라 상태 조회다).
   *
   * 같은 엔티티에 속한 바디(로봇 링크 등)는 제외한다 — 자기 자신을 구동하지 않는다.
   * 반환 순서는 결정론적이어야 한다(엔진 내부 순회 순서 고정).
   */
  dynamicBodiesTouching(entityId: EntityId): readonly BodyId[];

  /** 씬의 모든 **동적** 바디 (재순환 후보 훑기 등 — 순서 결정론적) */
  dynamicBodies(): readonly BodyId[];

  /** 선형 속도 조회 (m/s) */
  getLinearVelocity(bodyId: BodyId): Vec3;

  /**
   * 선형 속도 지정 (m/s) — 표면 구동 액추에이터 전용.
   * kinematic/fixed 바디에는 의미가 없으므로 구현은 동적 바디에만 적용한다.
   */
  setLinearVelocity(bodyId: BodyId, velocity: Vec3): void;

  /** 모든 바디/collider 제거 (씬 재로드·리셋) */
  clear(): void;

  /** WASM 자원 해제 */
  free(): void;
}
