// core/scene-editor.ts — SceneEditor 구현 (Phase 7 Scene Builder 코어 편집 엔진)
// (계약: core/scene-edit-types.ts · 규범: docs/UX_DESIGN.md §3.2/§3.3/§3.5, ROADMAP Phase 7)
//
// ── 설계 (scene-edit-types.ts 헤더의 원칙 구현) ──────────────────────
// - 편집의 단일 진실은 SceneSpec이다 (불변식 §2.5). 모든 연산은
//   [초안(deep clone) 변형 → validateScene → 물리/시각 반영 → spec 커밋 → onChange]
//   순서로 진행한다. 검증 실패 시 spec·월드 모두 변경 전 상태 그대로다(롤백 =
//   커밋하지 않음). 빌드 실패 시에도 spec은 커밋되지 않고, 재빌드 연산은 이전
//   스펙으로 엔티티를 복원한 뒤 원인 오류를 다시 던진다.
// - 엔티티 생성 로직은 scene-loader의 buildEntity/buildObjectEntity/attachRobotPhysics를
//   그대로 재사용한다 — SceneLoader.build와 편집기의 구성 결과가 드리프트하지 않는다.
// - builtEntities Map은 SceneHandle과 공유하는 살아있는 레코드다: 편집기가 add/remove
//   하면 SceneHandle.reset()/dispose()가 항상 현재 편집 상태를 순회한다.
// - 물리 pose 조작은 씬 리셋/편집 전용 API(PhysicsWorld.teleport ·
//   RobotBinding.teleportLinksToFk)만 사용한다 — 시뮬 재생 중 호출 금지 (§2.1).
//   편집 UI(ui 계층)는 엔진이 정지/일시정지 상태일 때만 편집 연산을 호출할 책임이 있다.
// - Undo/Redo는 ui 계층이 serialize() 스냅샷 스택으로 구현한다 (scene-edit-types 헤더).
//
// ── 계층 규칙 ────────────────────────────────────────────────────────
// core 전용: three.js/Rapier 심볼 없음. 물리는 PhysicsWorld, 시각은 RenderSceneApi,
// 검증은 schema/validate만 사용한다 (CLAUDE.md §3).

import type { SceneEditEvent, SceneEditKind, SceneEditor } from './scene-edit-types';
import type { EntityId, PhysicsWorld, Pose } from './types';
import type { RobotBinding, RobotRegistry } from './robots';
import { attachRobotPhysics, buildEntity, buildObjectEntity, GROUND_ENTITY_ID } from './scene-loader';
import type {
  BuiltEntityHandle,
  EntityBuildDeps,
  RenderSceneApi,
  RenderSyncLike,
} from './scene-loader';
import { validateScene } from '../schema/validate';
import { isRobotSpec } from '../schema/types';
import type {
  ColliderShape,
  EntitySpec,
  PhysicsSpec,
  Quat,
  RobotSpec,
  SceneSpec,
  Transform,
  Vec3,
} from '../schema/types';

// ── 내부 헬퍼 (순수) ─────────────────────────────────────────────────

const IDENTITY_QUAT: Quat = [0, 0, 0, 1];

function cloneVec3(v: Readonly<Vec3>): Vec3 {
  return [v[0], v[1], v[2]];
}

function cloneQuat(q: Readonly<Quat>): Quat {
  return [q[0], q[1], q[2], q[3]];
}

function poseFromTransform(transform: Transform): Pose {
  return {
    position: cloneVec3(transform.position),
    rotation: cloneQuat(transform.rotation ?? IDENTITY_QUAT),
  };
}

// ── 생성자 의존성 ────────────────────────────────────────────────────

export interface SceneEditorDeps {
  /** 편집 시작 상태 — 생성 시 deep clone되어 편집기의 진실이 된다 (외부 변형과 절연) */
  readonly spec: SceneSpec;
  readonly world: PhysicsWorld;
  /** RenderSync의 bind/unbind 표면 (엔티티 재빌드 시 바인딩 교체) */
  readonly sync: RenderSyncLike;
  /**
   * 시각 노드 생성/제거 + URDF 로드 표면. 구현은 main.ts 글루가 제공한다
   * (SceneLoader.build에 주입한 것과 같은 인스턴스를 넘길 것 — 노드가 같은 씬 루트에
   * 추가/제거되어야 한다).
   */
  readonly renderApi: RenderSceneApi;
  /** 씬의 로봇 레지스트리 (SceneHandle.robots와 같은 인스턴스) — robot add/remove 갱신 */
  readonly robots: RobotRegistry;
  /**
   * SceneLoader.build가 반환한 살아있는 빌드 레코드 Map (SceneHandle.builtEntities).
   * 편집기가 add/remove 시 함께 갱신해 reset()/dispose()가 항상 현재 상태를 본다.
   */
  readonly builtEntities: Map<EntityId, BuiltEntityHandle>;
}

// ── SceneEditorImpl ─────────────────────────────────────────────────

export class SceneEditorImpl implements SceneEditor {
  private currentSpec: SceneSpec;
  private readonly world: PhysicsWorld;
  private readonly renderApi: RenderSceneApi;
  private readonly robots: RobotRegistry;
  private readonly sync: RenderSyncLike;
  private readonly built: Map<EntityId, BuiltEntityHandle>;
  private readonly buildDeps: EntityBuildDeps;
  private readonly listeners = new Set<(e: SceneEditEvent) => void>();

  constructor(deps: SceneEditorDeps) {
    this.currentSpec = structuredClone(deps.spec);
    this.world = deps.world;
    this.renderApi = deps.renderApi;
    this.robots = deps.robots;
    this.sync = deps.sync;
    this.built = deps.builtEntities;
    this.buildDeps = {
      world: deps.world,
      renderApi: deps.renderApi,
      sync: deps.sync,
      robots: deps.robots,
    };
  }

  get spec(): Readonly<SceneSpec> {
    return this.currentSpec;
  }

  /**
   * 엔티티 추가. robot이면 URDF 로드로 async — await 중 다른 편집 연산을 호출하지
   * 않는 것은 호출자(ui) 책임이다(단일 사용자 편집 모델).
   * 실패 시(중복 id·검증 실패·빌드 실패) spec은 변경되지 않고 물리/시각 자원도 새지
   * 않는다 — buildEntity가 자기 몫의 부분 실패를 정리한다.
   */
  async addEntity(entitySpec: EntitySpec): Promise<void> {
    const id = entitySpec.id;
    this.guardReservedId(id);
    if (this.currentSpec.entities.some((e) => e.id === id) || this.built.has(id)) {
      throw new Error(
        `scene-editor: 엔티티 id '${id}'이(가) 이미 존재합니다 — 씬 내에서 유일해야 합니다`,
      );
    }
    const next = this.validatedDraft((draft) => {
      draft.entities.push(structuredClone(entitySpec));
    });
    const validated = this.entityAt(next, next.entities.length - 1);

    // 빌드가 성공해야만 spec을 커밋한다 — 실패 시 spec/월드 모두 변경 전 상태
    const record = await buildEntity(validated, this.buildDeps);
    this.built.set(id, record);
    this.currentSpec = next;
    this.emit('add', id);
  }

  /** 엔티티 제거: 물리 바디·sync 바인딩·시각 노드·로봇 레지스트리/핸들 해제 + spec 반영 */
  removeEntity(id: string): void {
    const index = this.requireEntityIndex(id);
    const record = this.requireRecord(id);
    record.dispose(); // sync.unbind → world.removeEntity → 시각 제거 → (robot) registry/핸들
    this.built.delete(id);
    this.currentSpec.entities.splice(index, 1);
    this.emit('remove', id);
  }

  /**
   * 배치 편집: spec.transform 갱신 + 물리 teleport(편집 전용 API — dynamic은
   * RapierWorld.teleport가 속도를 0으로 초기화) + 시각 반영.
   * robot: 루트 트랜스폼 재설정 후 FK를 즉시 재-push한다(링크 바디 teleport +
   * 다음 스텝 kinematic 목표) — 편집 직후 물리 pose가 새 배치와 일치한다.
   */
  updateTransform(id: string, transform: Transform): void {
    const index = this.requireEntityIndex(id);
    const record = this.requireRecord(id);
    const next = this.validatedDraft((draft) => {
      this.entityAt(draft, index).transform = structuredClone(transform);
    });
    const pose = poseFromTransform(this.entityAt(next, index).transform);

    if (record.robot) {
      this.placeRobotRoot(record, pose);
      // reset()이 편집된 배치로 돌아가도록 레코드의 초기 pose도 갱신 — 비로봇과 대칭.
      // (로봇 루트는 물리가 아니라 렌더 핸들이 소유하므로 이 스냅샷이 유일한 복원점이다)
      record.initialPose = poseFromTransform(this.entityAt(next, index).transform);
    } else if (record.bodyId !== undefined) {
      for (const bodyId of this.world.bodiesOfEntity(id)) {
        this.world.teleport(bodyId, pose);
      }
      // reset()이 편집된 배치로 돌아가도록 레코드의 초기 pose도 갱신
      record.initialPose = poseFromTransform(this.entityAt(next, index).transform);
      // prev 스냅샷을 텔레포트된 pose로 갱신 — 갱신하지 않으면 paused/idle 프레임의
      // sync.apply(alpha<1)가 다음 물리 tick의 commit()까지 편집 전 pose를 계속
      // 보간해 그린다 (SceneHandle.reset()이 잡는 것과 동일한 stale-prev 함정,
      // CLAUDE.md §2.1). 재빌드 경로(updateDimensions/updatePhysics)는 bind()가
      // prev를 재스냅샷하므로 이 처리가 필요 없다.
      this.sync.commit();
    } else if (record.node) {
      // 물리 없는 순수 장식 — 1회 배치 (불변식 §2.1의 시각 전용 예외)
      this.renderApi.setPose(record.node, pose.position, pose.rotation);
    }

    this.currentSpec = next;
    this.emit('transform', id);
  }

  /**
   * spec을 바꾸지 않고 물리/시각만 spec.transform으로 재수렴시킨다 (편집 통지 없음).
   *
   * 용도: 기즈모 드래그는 드래그 "중" 시각 노드를 직접 움직이는 순수 프리뷰인데
   * (render/interaction.ts 헤더), 그 드래그가 commit으로 이어지지 않는 경우가 있다 —
   * 움직임 없는 no-op 드래그, 로봇의 스케일 커밋 거부, runEdit이 삼킨 편집 실패.
   * 비로봇은 통합자가 RenderSync 바인딩을 되살리면 다음 프레임에 물리 진실이 시각을
   * 덮어써 자가 치유되지만, **로봇의 시각 루트는 RenderSync 대상이 아니라 FK 그래프
   * 자체**여서 되돌리는 주체가 없다 — 프리뷰가 그대로 남고 다음 tickAll에서 kinematic
   * 링크 바디로 역류한다. 그 비대칭을 없애는 것이 이 연산이다.
   *
   * spec은 진실 그대로이므로 검증/undo 스냅샷/onChange가 필요 없다(순수 복원).
   */
  resyncTransform(id: string): void {
    const index = this.requireEntityIndex(id);
    const record = this.requireRecord(id);
    const pose = poseFromTransform(this.entityAt(this.currentSpec, index).transform);

    if (record.robot) {
      this.placeRobotRoot(record, pose);
    } else if (record.bodyId !== undefined) {
      for (const bodyId of this.world.bodiesOfEntity(id)) {
        this.world.teleport(bodyId, pose);
      }
      this.sync.commit();
    } else if (record.node) {
      this.renderApi.setPose(record.node, pose.position, pose.rotation);
    }
  }

  /**
   * 로봇 루트 배치의 단일 경로 (updateTransform / resyncTransform 공용):
   * 렌더 핸들의 루트 트랜스폼 → 링크 바디 즉시 정렬(편집 전용 teleport) → 다음 스텝
   * kinematic 목표 재-push(스윕 속도 0) → sync prev 스냅샷 갱신.
   */
  private placeRobotRoot(record: BuiltEntityHandle, pose: Pose): void {
    if (!record.robot) return;
    record.robot.handle.setRootTransform(pose);
    record.robot.binding.teleportLinksToFk();
    record.robot.binding.tick();
    // prev 스냅샷 갱신 — SceneHandle.reset()과 동일 계약 (비로봇 분기 주석 참조)
    this.sync.commit();
  }

  /**
   * 치수 편집 (UX_DESIGN §3.5 Dimensions): 프리미티브 시각 엔티티 전용.
   * visual.primitive와 physics.colliders[0].shape를 함께 교체하고(다른 collider
   * 필드는 유지) 엔티티를 제자리(spec 순서 유지)에서 재빌드한다 — 메시와 collider가
   * 항상 같은 형상을 갖는다.
   *
   * collider가 2개 이상인 엔티티(수작성 씬 JSON)는 거부한다 — colliders[0]만 바꾸면
   * 나머지 collider가 이전 형상으로 남아 시각 메시와 물리 footprint가 조용히
   * 어긋난다("단일 치수 진실" 재빌드가 막으려는 바로 그 드리프트).
   */
  updateDimensions(id: string, shape: ColliderShape): void {
    const index = this.requireEntityIndex(id);
    const current = this.entityAt(this.currentSpec, index);
    if (isRobotSpec(current) || current.visual.kind !== 'primitive') {
      throw new Error(
        `scene-editor: 엔티티 '${id}'은(는) 프리미티브 시각 엔티티가 아니어서 치수를 편집할 수 없습니다 ` +
          `(visual.kind: '${current.visual.kind}')`,
      );
    }
    const colliderCount = current.physics?.colliders.length ?? 0;
    if (colliderCount > 1) {
      throw new Error(
        `scene-editor: 엔티티 '${id}'은(는) collider가 ${colliderCount}개여서 치수를 편집할 수 없습니다 ` +
          '— 치수 편집은 단일 collider 엔티티만 지원합니다 (메시·collider 형상 드리프트 방지)',
      );
    }
    const next = this.validatedDraft((draft) => {
      const entity = this.entityAt(draft, index);
      entity.visual.primitive = structuredClone(shape);
      const firstCollider = entity.physics?.colliders[0];
      if (firstCollider) firstCollider.shape = structuredClone(shape);
    });
    this.rebuildInPlace(id, index, next, 'dimensions');
  }

  /**
   * 물리 속성 편집: physics 블록 전체 교체(bodyType/collider/damping 등) 후 재빌드.
   * robot의 물리는 URDF <collision>에서 유도되므로 편집 대상이 아니다.
   */
  updatePhysics(id: string, physics: PhysicsSpec): void {
    const index = this.requireEntityIndex(id);
    const current = this.entityAt(this.currentSpec, index);
    if (isRobotSpec(current)) {
      throw new Error(
        `scene-editor: 로봇 엔티티 '${id}'의 물리는 URDF에서 유도됩니다 — physics 블록을 편집할 수 없습니다`,
      );
    }
    const next = this.validatedDraft((draft) => {
      this.entityAt(draft, index).physics = structuredClone(physics);
    });
    this.rebuildInPlace(id, index, next, 'physics');
  }

  /**
   * 엔티티 id 변경.
   *
   * 범위: SceneSpec 내부의 id(entities[].id)와 런타임 레지스트리(월드 핸들 매핑·
   * 로봇 레지스트리·빌드 레코드)만 갱신한다. **ControlSequence 등 씬 밖 문서의
   * 참조(robot/between 등)는 범위 밖이다** — 시퀀스는 별도 데이터이고(불변식 §2.6),
   * 참조 정합은 시퀀스 검증(validateSequence(scene))이 로드/재생 시점에 다시 잡는다.
   *
   * 물리 핸들 매핑(collider↔EntityId)은 world가 소유한 유일한 진실이고(CLAUDE.md §2.4)
   * PhysicsWorld에는 rename 연산이 없으므로, 새 id로 바디를 재생성한다:
   * - 비로봇: 엔티티 재빌드 (spec 트랜스폼 기준 — 편집 모드 의미론).
   * - robot: URDF 재로드 없이(동기 계약) 같은 RobotHandle에 attachRobotPhysics로
   *   새 id의 링크 바디/바인딩을 재부착하고, 현재 관절 상태를 보존한다.
   */
  renameEntity(id: string, newId: string): void {
    if (newId === id) return;
    const index = this.requireEntityIndex(id);
    this.guardReservedId(newId);
    if (this.currentSpec.entities.some((e) => e.id === newId) || this.built.has(newId)) {
      throw new Error(
        `scene-editor: 엔티티 id '${newId}'이(가) 이미 존재합니다 — 씬 내에서 유일해야 합니다`,
      );
    }
    const record = this.requireRecord(id);
    const oldEntity = this.entityAt(this.currentSpec, index);
    const next = this.validatedDraft((draft) => {
      this.entityAt(draft, index).id = newId;
    });
    const newEntity = this.entityAt(next, index);

    if (record.robot) {
      this.renameRobotRecord(id, newId, record, oldEntity, newEntity);
    } else {
      // 비로봇: 이전 레코드 해제 후 새 id로 재빌드 (실패 시 이전 스펙으로 복원)
      record.dispose();
      this.built.delete(id);
      let rebuilt: BuiltEntityHandle;
      try {
        rebuilt = buildObjectEntity(newEntity, this.buildDeps);
      } catch (err) {
        this.built.set(id, buildObjectEntity(oldEntity, this.buildDeps));
        throw this.wrapRebuildError(id, err);
      }
      this.built.set(newId, rebuilt);
    }

    this.currentSpec = next;
    this.emit('rename', newId);
  }

  /** 저장/Undo 스냅샷용 깊은 복사본 — 항상 validateScene을 통과하는 canonical spec */
  serialize(): SceneSpec {
    return structuredClone(this.currentSpec);
  }

  onChange(fn: (e: SceneEditEvent) => void): () => void {
    this.listeners.add(fn);
    return () => {
      this.listeners.delete(fn);
    };
  }

  // ── 내부 구현 ─────────────────────────────────────────────────────

  private emit(kind: SceneEditKind, entityId: string): void {
    // 순회 중 구독 해제가 안전하도록 스냅샷 배열로 통지
    for (const fn of [...this.listeners]) fn({ kind, entityId });
  }

  /**
   * 현재 spec의 깊은 복사 초안에 mutate를 적용하고 validateScene으로 검증한다.
   * 실패 시 사람이 읽을 수 있는 한국어 오류를 던진다(spec 미변경 — 롤백 완료 상태).
   * 성공 시 zod가 정규화한 canonical spec을 반환한다(커밋은 호출자 몫).
   */
  private validatedDraft(mutate: (draft: SceneSpec) => void): SceneSpec {
    const draft = structuredClone(this.currentSpec);
    mutate(draft);
    const result = validateScene(draft);
    if (!result.ok) {
      throw new Error(
        `scene-editor: 씬 검증 실패 — 변경이 취소되었습니다\n${result.errors.join('\n')}`,
      );
    }
    return result.value;
  }

  /**
   * 비로봇 엔티티를 제자리에서 재빌드한다: 이전 레코드 해제 → 새 스펙으로 빌드 →
   * 레코드/spec 커밋 → 통지. 새 스펙 빌드 실패 시 이전 스펙으로 엔티티를 복원하고
   * 원인 오류를 다시 던진다 (spec 미커밋 — 데이터·월드 정합 유지).
   * spec.entities 배열의 순서는 index 위치를 그대로 쓰므로 보존된다.
   */
  private rebuildInPlace(id: string, index: number, next: SceneSpec, kind: SceneEditKind): void {
    const record = this.requireRecord(id);
    const oldEntity = this.entityAt(this.currentSpec, index);
    const newEntity = this.entityAt(next, index);

    record.dispose();
    this.built.delete(id);
    let rebuilt: BuiltEntityHandle;
    try {
      rebuilt = buildObjectEntity(newEntity, this.buildDeps);
    } catch (err) {
      // 이전 스펙은 직전까지 성공적으로 빌드돼 있었다 — 복원은 성공을 기대할 수 있다
      this.built.set(id, buildObjectEntity(oldEntity, this.buildDeps));
      throw this.wrapRebuildError(id, err);
    }
    this.built.set(id, rebuilt);
    this.currentSpec = next;
    this.emit(kind, id);
  }

  /** robot 엔티티의 id 변경 — renameEntity의 robot 분기 (같은 RobotHandle 재사용) */
  private renameRobotRecord(
    id: string,
    newId: string,
    record: BuiltEntityHandle,
    oldEntity: EntitySpec,
    newEntity: EntitySpec,
  ): void {
    if (!record.robot || !isRobotSpec(oldEntity) || !isRobotSpec(newEntity)) {
      throw new Error(`scene-editor: 엔티티 '${id}'의 로봇 레코드가 유효하지 않습니다 (내부 오류)`);
    }
    const { handle } = record.robot;
    // 현재 관절 상태(URDF 원명 키)를 보존 — attachRobotPhysics는 home을 적용하므로
    const jointValues = record.robot.binding.readJoints();

    // 이전 id의 물리·레지스트리 해제 (RobotHandle/시각은 유지 — URDF 재로드 없음)
    this.world.removeEntity(id);
    this.robots.remove(id);
    this.built.delete(id);

    let newBinding: RobotBinding;
    try {
      newBinding = attachRobotPhysics(newEntity, handle, this.buildDeps);
    } catch (err) {
      // 복원: 이전 id로 재부착 (직전까지 같은 내용으로 성공했던 스펙)
      const restored = this.reattachRobot(oldEntity, handle, jointValues);
      this.built.set(id, this.makeRobotRecord(oldEntity.id, handle, restored));
      throw this.wrapRebuildError(id, err);
    }
    // 관절 상태 복원 + 링크 바디 즉시 정렬 (편집 전용 teleport 경로)
    newBinding.setJoints(jointValues);
    newBinding.teleportLinksToFk();
    newBinding.tick();

    this.built.set(newId, this.makeRobotRecord(newId, handle, newBinding));
  }

  /** 로봇 재부착 + 관절 상태 복원 (rename 실패 복원 경로 공용) */
  private reattachRobot(
    spec: RobotSpec,
    handle: BuiltRobotHandle,
    jointValues: Record<string, number>,
  ): RobotBinding {
    const binding = attachRobotPhysics(spec, handle, this.buildDeps);
    binding.setJoints(jointValues);
    binding.teleportLinksToFk();
    binding.tick();
    return binding;
  }

  private makeRobotRecord(
    entityId: string,
    handle: BuiltRobotHandle,
    binding: RobotBinding,
  ): BuiltEntityHandle {
    return {
      entityId,
      bound: false,
      robot: { handle, binding },
      dispose: (): void => {
        this.world.removeEntity(entityId);
        this.robots.remove(entityId);
        handle.dispose();
      },
    };
  }

  private wrapRebuildError(id: string, err: unknown): Error {
    const msg = err instanceof Error ? err.message : String(err);
    return new Error(
      `scene-editor: 엔티티 '${id}' 재빌드 실패 — 변경이 취소되고 이전 상태로 복원되었습니다: ${msg}`,
    );
  }

  /** spec.entities[index] 안전 접근 (noUncheckedIndexedAccess 대응 — 인덱스는 내부 유도값) */
  private entityAt(spec: SceneSpec, index: number): EntitySpec {
    const entity = spec.entities[index];
    if (!entity) {
      throw new Error(`scene-editor: 엔티티 인덱스 ${index}가 유효하지 않습니다 (내부 오류)`);
    }
    return entity;
  }

  private requireEntityIndex(id: string): number {
    const index = this.currentSpec.entities.findIndex((e) => e.id === id);
    if (index < 0) {
      const ids = this.currentSpec.entities.map((e) => e.id);
      throw new Error(
        `scene-editor: 씬에 엔티티 '${id}'이(가) 없습니다` +
          (ids.length > 0 ? ` — 현재 엔티티: ${ids.join(', ')}` : ''),
      );
    }
    return index;
  }

  private requireRecord(id: string): BuiltEntityHandle {
    const record = this.built.get(id);
    if (!record) {
      throw new Error(
        `scene-editor: 엔티티 '${id}'의 빌드 레코드가 없습니다 — spec과 월드가 어긋났습니다 (내부 오류)`,
      );
    }
    return record;
  }

  /** environment.ground 예약 id 보호 — SceneLoader.build의 가드와 동일 규약 */
  private guardReservedId(id: string): void {
    if (id === GROUND_ENTITY_ID) {
      throw new Error(
        `scene-editor: 엔티티 id '${GROUND_ENTITY_ID}'는 environment.ground가 예약한 id입니다 — 다른 id를 사용하세요`,
      );
    }
  }
}

/** BuiltRobot.handle 타입 별칭 (내부 가독성용) */
type BuiltRobotHandle = NonNullable<BuiltEntityHandle['robot']>['handle'];
