// core/scene-loader.ts — SceneSpec → 물리 바디 + 시각 노드 생성·연결
// (규범: docs/ARCHITECTURE.md §3 "core/scene-loader", CLAUDE.md §2.5 "씬은 데이터로 선언한다")
//
// ── 계층 규칙 ────────────────────────────────────────────────────────
// 이 모듈은 schema 타입 + PhysicsWorld(core/types) + RenderSync(core/sync)만 안다.
// three.js는 직접 import하지 않는다 — core에서 three를 알아도 되는 곳은 core/sync.ts뿐
// (CLAUDE.md §3). 시각 노드 생성은 호출자가 주입하는 좁은 RenderSceneApi를 통해서만
// 요청하고, 반환된 노드는 opaque 타입(VisualNode)으로만 다뤄 sync.bind로 흘려보낸다.
// VisualNode는 RenderSync.bind의 파라미터 타입에서 유도하므로(three 경계는 sync가 소유)
// 이 파일에는 three 심볼이 등장하지 않는다.
//
// ── 책임 ─────────────────────────────────────────────────────────────
// - EntitySpec → 물리 바디(PhysicsBodyInit) + collider들 + 프리미티브 시각 메시 생성
// - 물리 바디 ↔ 시각 노드를 RenderSync에 바인딩 (물리 → 시각 단방향, 불변식 §2.1)
// - environment.ground → ENV 고정 바닥(collider 상면 y=0) + 바닥 시각 메시
// - SceneHandle 반환: reset(초기 트랜스폼 텔레포트 — 결정론적 재생), dispose(전체 해제)
//
// robot 엔티티(URDF 로딩·링크 바디)는 Phase 3에서 지원한다 — 지금은 명시적 오류.

import type {
  BodyId,
  EntityId,
  PhysicsBodyInit,
  PhysicsWorld,
  Pose,
} from './types';
import type { RenderSync } from './sync';
import type {
  ColliderShape,
  ColliderSpec,
  EntitySpec,
  Quat,
  SceneSpec,
  Vec3,
} from '../schema/types';

// ── 시각 노드 opaque 타입 ───────────────────────────────────────────

/**
 * RenderSync.bind가 받는 시각 노드 타입(three.Object3D)을 파라미터 위치에서 유도한
 * opaque 별칭. scene-loader는 이 타입의 내부를 들여다보지 않고 그대로 통과시킨다.
 */
export type VisualNode = Parameters<RenderSync['bind']>[1];

// ── 렌더 계층에 요청하는 좁은 인터페이스 ─────────────────────────────
// 구현은 render 계층 위에서 조립하는 글루(main.ts)가 제공한다. 헤드리스 테스트에서는
// no-op 구현으로 대체 가능하다 — scene-loader는 three 없이도 완결적으로 동작한다.

export interface RenderSceneApi {
  /** 프리미티브 형상 시각 메시를 생성해 씬 루트의 직접 자식으로 추가한다 (sync 바인딩 계약). */
  addPrimitive(shape: ColliderShape, color: string | undefined): VisualNode;
  /** 바닥 시각 메시를 생성해 씬 루트에 추가한다. 메시 스스로 상면 y=0에 배치된다. */
  addGround(): VisualNode;
  /** 물리 바디가 없는 순수 시각 노드의 1회 배치 (불변식 §2.1의 "순수 시각 요소" 예외). */
  setPose(node: VisualNode, position: Vec3, rotation: Quat): void;
  /** 노드를 씬에서 제거한다. */
  remove(node: VisualNode): void;
}

// ── 바닥(ground) 규약 상수 (매직넘버 금지 — CLAUDE.md §4) ────────────

/** environment.ground가 만드는 예약 엔티티 id (SceneSpec.entities와 충돌 금지) */
export const GROUND_ENTITY_ID: EntityId = '__ground';

/** 바닥 collider half extents: 20m × 0.1m × 20m 슬래브 */
export const GROUND_HALF_EXTENTS_M: Vec3 = [10, 0.05, 10];

/** 바닥 바디 중심 y — collider 상면이 정확히 y=0이 되도록 half-height만큼 내린다 */
const GROUND_CENTER_Y_M = -GROUND_HALF_EXTENTS_M[1];

/**
 * 바닥 collider 규약: ENV 그룹, ROBOT/OBJECT와 상호작용 (CLAUDE.md §5).
 * emitEvents는 기본 false — 바닥 접촉 이벤트는 상대(OBJECT/ROBOT) collider가 발행한다.
 */
const GROUND_COLLIDER_SPEC: ColliderSpec = {
  shape: { kind: 'box', halfExtents: GROUND_HALF_EXTENTS_M },
  group: 'ENV',
  collidesWith: ['ROBOT', 'OBJECT'],
};

const IDENTITY_QUAT: Quat = [0, 0, 0, 1];

// ── 반환 핸들 ───────────────────────────────────────────────────────

export interface SceneHandle {
  /** 생성된 엔티티 id 목록 (environment.ground 사용 시 GROUND_ENTITY_ID 포함) */
  readonly entityIds: readonly EntityId[];
  /**
   * 모든 바디를 초기 스펙 트랜스폼으로 텔레포트하고 속도를 0으로 만든다
   * — 동일 SceneSpec에서 동일 궤적을 재현하는 결정론적 재생용 (SIMULATION.md §6).
   * 마지막에 sync의 prev 스냅샷도 갱신하므로, 직후 렌더 프레임부터 화면이
   * 리셋된 물리 pose를 비춘다 (stale prev 보간 방지 — CLAUDE.md §2.1).
   */
  reset(): void;
  /** 물리 바디·시각 노드·sync 바인딩을 전부 해제한다. */
  dispose(): void;
}

// ── 내부 레코드 ─────────────────────────────────────────────────────

interface BuiltEntity {
  readonly entityId: EntityId;
  /** 물리 바디가 없는 순수 장식이면 undefined */
  readonly bodyId?: BodyId;
  /** reset()이 되돌릴 초기 pose (스펙에서 복사 — 이후 스펙 변형과 무관) */
  readonly initialPose?: Pose;
  readonly node: VisualNode;
  /** sync.bind 등록 여부 (바닥 시각 메시는 정적이라 바인딩하지 않는다) */
  readonly bound: boolean;
}

function cloneVec3(v: Readonly<Vec3>): Vec3 {
  return [v[0], v[1], v[2]];
}

function cloneQuat(q: Readonly<Quat>): Quat {
  return [q[0], q[1], q[2], q[3]];
}

// ── SceneLoader ─────────────────────────────────────────────────────

export class SceneLoader {
  constructor(
    private readonly world: PhysicsWorld,
    private readonly renderApi: RenderSceneApi,
    private readonly sync: RenderSync,
  ) {}

  /**
   * SceneSpec을 해석해 물리 바디 + 시각 노드를 생성하고 바인딩한다.
   * 중간에 실패하면 부분 생성된 자원을 정리한 뒤 오류를 다시 던진다
   * — 반쯤 로드된 씬을 남기지 않는다.
   */
  build(spec: SceneSpec): SceneHandle {
    const built: BuiltEntity[] = [];
    try {
      if (spec.environment?.ground) {
        if (spec.entities.some((e) => e.id === GROUND_ENTITY_ID)) {
          throw new Error(
            `scene-loader: 엔티티 id '${GROUND_ENTITY_ID}'는 environment.ground가 예약한 id입니다 — 다른 id를 사용하세요`,
          );
        }
        built.push(this.buildGround());
      }
      for (const entity of spec.entities) {
        built.push(this.buildEntity(entity));
      }
    } catch (err) {
      this.teardown(built);
      throw err;
    }

    const entityIds: readonly EntityId[] = built.map((b) => b.entityId);
    return {
      entityIds,
      reset: (): void => {
        for (const b of built) {
          if (b.bodyId !== undefined && b.initialPose) {
            // teleport는 리셋/편집 전용 API — 시뮬 재생 중 pose 주입에는 쓰지 않는다
            this.world.teleport(b.bodyId, b.initialPose);
          }
        }
        // prev 스냅샷을 텔레포트된 pose로 갱신 — 갱신하지 않으면 다음 물리 tick의
        // commit() 전까지 sync.apply(alpha<1)가 리셋 전 pose를 계속 그린다
        // ("three.js는 물리의 거울" 불변식 위반, CLAUDE.md §2.1).
        this.sync.commit();
      },
      dispose: (): void => {
        this.teardown(built);
      },
    };
  }

  // ── 내부 빌더 ─────────────────────────────────────────────────────

  /** ENV 고정 바닥: collider 상면이 정확히 y=0. 시각 메시는 자체 배치(바인딩 불필요). */
  private buildGround(): BuiltEntity {
    const bodyId = this.world.createBody(GROUND_ENTITY_ID, {
      bodyType: 'fixed',
      position: [0, GROUND_CENTER_Y_M, 0],
    });
    this.world.createCollider(bodyId, GROUND_COLLIDER_SPEC, GROUND_ENTITY_ID);
    const node = this.renderApi.addGround();
    return {
      entityId: GROUND_ENTITY_ID,
      bodyId,
      initialPose: { position: [0, GROUND_CENTER_Y_M, 0], rotation: cloneQuat(IDENTITY_QUAT) },
      node,
      bound: false,
    };
  }

  private buildEntity(entity: EntitySpec): BuiltEntity {
    if (entity.type === 'robot') {
      throw new Error(
        `scene-loader: robot 엔티티 '${entity.id}'는 아직 지원되지 않습니다 — URDF 로딩·링크 바디 생성은 Phase 3`,
      );
    }

    const node = this.buildVisual(entity);
    const physics = entity.physics;

    if (!physics) {
      // 물리 없는 순수 장식: 스펙 트랜스폼으로 1회 배치 (시각 전용 — 불변식 §2.1 예외)
      this.renderApi.setPose(
        node,
        cloneVec3(entity.transform.position),
        cloneQuat(entity.transform.rotation ?? IDENTITY_QUAT),
      );
      return { entityId: entity.id, node, bound: false };
    }

    const init: PhysicsBodyInit = {
      bodyType: physics.bodyType,
      position: cloneVec3(entity.transform.position),
      rotation: entity.transform.rotation ? cloneQuat(entity.transform.rotation) : undefined,
      linearDamping: physics.linearDamping,
      angularDamping: physics.angularDamping,
      gravityScale: physics.gravityScale,
    };
    const bodyId = this.world.createBody(entity.id, init);
    try {
      for (const collider of physics.colliders) {
        this.world.createCollider(bodyId, collider, entity.id);
      }
      // 물리 → 시각 단방향 동기화 등록 (트랜스폼의 진실은 물리 — 불변식 §2.1)
      this.sync.bind(bodyId, node);
    } catch (err) {
      // 바디는 만들었으나 collider/바인딩에 실패 — 이 엔티티 몫만 되돌리고 재던짐
      this.world.removeEntity(entity.id);
      this.renderApi.remove(node);
      throw err;
    }

    return {
      entityId: entity.id,
      bodyId,
      initialPose: {
        position: cloneVec3(entity.transform.position),
        rotation: cloneQuat(entity.transform.rotation ?? IDENTITY_QUAT),
      },
      node,
      bound: true,
    };
  }

  private buildVisual(entity: EntitySpec): VisualNode {
    const visual = entity.visual;
    switch (visual.kind) {
      case 'primitive': {
        if (!visual.primitive) {
          throw new Error(
            `scene-loader: 엔티티 '${entity.id}'의 visual.kind가 'primitive'인데 visual.primitive 형상이 없습니다`,
          );
        }
        return this.renderApi.addPrimitive(visual.primitive, visual.color);
      }
      case 'mesh':
      case 'urdf':
        throw new Error(
          `scene-loader: visual.kind '${visual.kind}'(엔티티 '${entity.id}')는 아직 지원되지 않습니다 — 에셋/URDF 로딩은 Phase 3`,
        );
    }
  }

  /** 생성된 자원 해제: sync 바인딩 → 물리 바디 → 시각 노드 순서 */
  private teardown(built: BuiltEntity[]): void {
    for (const b of built) {
      if (b.bodyId !== undefined) {
        if (b.bound) this.sync.unbind(b.bodyId);
        this.world.removeEntity(b.entityId);
      }
      this.renderApi.remove(b.node);
    }
    built.length = 0;
  }
}
