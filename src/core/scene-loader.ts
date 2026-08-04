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
// - robot 엔티티(Phase 3): renderApi.loadRobot(비동기 URDF 로드) → 링크별
//   kinematicPosition 바디 + ROBOT 그룹 collider 생성 → RobotBinding/RobotRegistry 등록.
//   로봇 링크는 RenderSync에 바인딩하지 않는다 — 시각 로봇(urdf 씬 그래프)은 관절 상태에서
//   직접 갱신되는 FK 그래프 그 자체이고, 물리 바디는 매 tick 같은 FK 결과를 받는다
//   (단일 진실 = 관절 상태, core/robot-types.ts 헤더 참조).
// - SceneHandle 반환: reset(초기 트랜스폼 텔레포트 + 로봇 home 복원 — 결정론적 재생),
//   dispose(전체 해제, 로봇 핸들 포함)
//
// ── 엔티티 빌드 루틴 공유 (Phase 7 Scene Builder) ────────────────────
// 엔티티 1개 몫의 생성 로직은 exported 함수(buildEntity/buildObjectEntity/
// attachRobotPhysics)로 추출되어 SceneLoader.build와 SceneEditor(core/scene-editor.ts)가
// **같은 코드**를 쓴다 — 엔티티 구성의 단일 진실(로직 드리프트 방지). 각 BuiltEntityHandle
// 은 자기 몫의 해제(dispose)를 소유하고, SceneHandle.builtEntities는 이 레코드들의
// "살아있는" Map이다(SceneEditor가 add/remove 시 함께 갱신 — reset/dispose가 항상
// 현재 편집 상태를 본다).

import type {
  BodyId,
  EntityId,
  PhysicsBodyInit,
  PhysicsWorld,
  Pose,
} from './types';
import { ConveyorBinding, ConveyorRegistry, rotateVec3 } from './conveyor';
import type { RobotFkView } from './robot-types';
import { RobotBinding, RobotRegistry } from './robots';
import type { RenderSync } from './sync';
import { isRobotSpec } from '../schema/types';
import type {
  ColliderGroup,
  ColliderShape,
  ColliderSpec,
  EntitySpec,
  Quat,
  RobotSpec,
  SceneSpec,
  Vec3,
} from '../schema/types';

// ── 시각 노드 opaque 타입 ───────────────────────────────────────────

/**
 * RenderSync.bind가 받는 시각 노드 타입(three.Object3D)을 파라미터 위치에서 유도한
 * opaque 별칭. scene-loader는 이 타입의 내부를 들여다보지 않고 그대로 통과시킨다.
 */
export type VisualNode = Parameters<RenderSync['bind']>[1];

/**
 * 엔티티 빌드·편집이 필요로 하는 sync 표면. commit은 SceneEditor의 편집 teleport가
 * prev 스냅샷을 갱신하는 데 쓴다 — 갱신하지 않으면 paused 프레임의 apply(alpha<1)가
 * 편집 전 pose를 계속 보간해 그린다 (SceneHandle.reset()과 동일 계약, CLAUDE.md §2.1).
 */
export type RenderSyncLike = Pick<RenderSync, 'bind' | 'unbind' | 'commit' | 'apply'>;

// ── 로봇 핸들 (core 쪽 구조적 계약) ─────────────────────────────────
// render/urdf.ts의 RobotHandle과 구조적으로 동일하지만, core → render import를 만들지
// 않기 위해 여기서 독립 선언한다 (계층 의존 방향 ui → core → render, CLAUDE.md §3).
// render 구현체는 이 타입을 구조적으로 만족하므로 글루(main.ts)가 그대로 넘길 수 있다.

export type RobotHandle = RobotFkView & {
  /** 엔티티 배치(SceneSpec transform) 반영 — Y-up 월드 기준 루트 pose 설정 */
  setRootTransform(pose: Pose): void;
  /** 씬에서 제거하고 렌더 자원을 해제한다. */
  dispose(): void;
};

/** renderApi.loadRobot 요청 스펙 (RobotSpec에서 유도 — render는 스키마를 몰라도 된다) */
export interface RobotLoadRequest {
  urdfPath: string;
  packages?: Record<string, string>;
}

// ── 렌더 계층에 요청하는 좁은 인터페이스 ─────────────────────────────
// 구현은 render 계층 위에서 조립하는 글루(main.ts)가 제공한다. 헤드리스 테스트에서는
// no-op 구현으로 대체 가능하다 — scene-loader는 three 없이도 완결적으로 동작한다.

export interface RenderSceneApi {
  /**
   * 프리미티브 형상 시각 메시를 생성해 씬 루트의 직접 자식으로 추가한다 (sync 바인딩 계약).
   * `style`은 순수 표현 옵션(불투명도·모서리 선)이며 물리에 영향하지 않는다 — 감지 존을
   * "통과 가능해 보이게" 그리는 용도다 (VisualSpec.opacity/edges 주석).
   */
  addPrimitive(
    shape: ColliderShape,
    color: string | undefined,
    style?: { opacity?: number; edges?: boolean },
  ): VisualNode;
  /** 바닥 시각 메시를 생성해 씬 루트에 추가한다. 메시 스스로 상면 y=0에 배치된다. */
  addGround(): VisualNode;
  /** 물리 바디가 없는 순수 시각 노드의 1회 배치 (불변식 §2.1의 "순수 시각 요소" 예외). */
  setPose(node: VisualNode, position: Vec3, rotation: Quat): void;
  /** 노드를 씬에서 제거한다. */
  remove(node: VisualNode): void;
  /** URDF 로봇을 비동기 로드해 씬 루트에 부착하고 FK 뷰 핸들을 돌려준다 (Phase 3). */
  loadRobot(request: RobotLoadRequest): Promise<RobotHandle>;
  /**
   * [Phase 7] 임포트 에셋 시각 노드를 씬 루트의 직접 자식으로 추가한다
   * (visual.kind==='mesh', ref='asset://…'). 글루(main.ts)는 MeshAssetStore 원형의
   * clone을 추가하고, 미등록 ref는 한국어 오류로 던진다. 선택 멤버 — 헤드리스
   * 테스트/프리미티브 전용 환경은 구현하지 않아도 되며, 미구현 상태로 mesh visual을
   * 만나면 buildVisual이 한국어 오류를 던진다.
   */
  addMeshAsset?(ref: string): VisualNode;
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

// ── 로봇 링크 collider 규약 상수 (CLAUDE.md §5, 매직넘버 금지 §4) ────

/** 로봇 링크 collider 마찰 계수 — 파지 이전 MVP에서 사물이 잘 미끄러지지 않게 높게 잡는다 */
const ROBOT_LINK_FRICTION = 0.8;

/**
 * 로봇 링크 collider의 상호작용 대상 — **항상 ROBOT을 포함한다**.
 *
 * 로봇 링크는 전부 ROBOT 그룹이므로, 여기서 ROBOT을 빼면 자기 링크뿐 아니라
 * **다른 로봇과의 충돌까지 통째로 사라진다**(로봇 두 대가 서로 통과). 그룹 비트는
 * "어느 로봇인지"를 구분하지 못하므로, 자기 링크 억제는 그룹이 아니라 엔티티 단위로
 * 처리한다 — 로봇 한 대의 모든 링크는 같은 EntityId를 공유하므로 "같은 엔티티 접촉"이
 * 곧 self-collision이다 (world.setSelfContactEnabled, CLAUDE.md §5).
 */
const ROBOT_COLLIDES_WITH: readonly ColliderGroup[] = ['ENV', 'OBJECT', 'ROBOT'];

/** RobotSpec.linkColliders 미지정 시 기본 정책 (DATA_MODEL.md §4.1) */
const DEFAULT_LINK_COLLIDER_POLICY: NonNullable<RobotSpec['linkColliders']> = 'fromVisual';

// ── 엔티티 빌드 계약 (SceneLoader + SceneEditor 공용) ────────────────

/** 엔티티 1개를 만드는 데 필요한 의존성 묶음 — SceneLoader.build가 조립하고,
 *  SceneEditor는 같은 인스턴스들로 이 묶음을 재구성해 동일 루틴을 호출한다. */
export interface EntityBuildDeps {
  readonly world: PhysicsWorld;
  readonly renderApi: RenderSceneApi;
  readonly sync: RenderSyncLike;
  /** 씬의 로봇 바인딩 보관소 — robot 엔티티 빌드가 등록/해제한다 */
  readonly robots: RobotRegistry;
  /** 씬의 컨베이어 보관소 — conveyor 블록이 있는 엔티티 빌드가 등록/해제한다 */
  readonly conveyors: ConveyorRegistry;
}

/** robot 엔티티 전용 레코드 (reset/편집이 사용) */
export interface BuiltRobot {
  readonly handle: RobotHandle;
  readonly binding: RobotBinding;
}

/**
 * 생성된 엔티티 1개 몫의 살아있는 레코드 + 해제 책임.
 * dispose()는 이 엔티티 몫의 sync 바인딩 → 물리 바디 → 시각 노드/로봇 핸들·레지스트리를
 * 해제한다 (멱등 아님 — 호출자는 1회만 부른다).
 */
export interface BuiltEntityHandle {
  readonly entityId: EntityId;
  /** 물리 바디가 없는 순수 장식이면 undefined (로봇은 링크당 바디 — world 매핑이 소유) */
  readonly bodyId?: BodyId;
  /**
   * reset()이 되돌릴 pose. 빌드 시점 스펙에서 복사되며, 씬 편집(SceneEditor의
   * updateTransform)이 스펙과 함께 갱신한다 — reset은 "현재 편집된 배치"로 돌아간다.
   */
  initialPose?: Pose;
  /** 프리미티브/바닥 시각 노드. 로봇은 핸들이 시각을 소유하므로 undefined */
  readonly node?: VisualNode;
  /** sync.bind 등록 여부 (바닥 시각 메시·로봇 링크는 바인딩하지 않는다) */
  readonly bound: boolean;
  /** robot 엔티티 전용 레코드 */
  readonly robot?: BuiltRobot;
  /** 이 엔티티 몫의 자원 전체 해제 */
  dispose(): void;
}

function cloneVec3(v: Readonly<Vec3>): Vec3 {
  return [v[0], v[1], v[2]];
}

function cloneQuat(q: Readonly<Quat>): Quat {
  return [q[0], q[1], q[2], q[3]];
}

// ── 엔티티 빌드 루틴 (단일 진실 — SceneLoader.build와 SceneEditor 공용) ──

/**
 * EntitySpec 1개 → 물리 + 시각 + 바인딩. robot이면 URDF 로드로 async.
 * 중간 실패 시 자기 몫의 부분 자원을 정리한 뒤 다시 던진다.
 */
export async function buildEntity(
  entity: EntitySpec,
  deps: EntityBuildDeps,
): Promise<BuiltEntityHandle> {
  if (isRobotSpec(entity)) return buildRobotEntity(entity, deps);
  return buildObjectEntity(entity, deps);
}

/**
 * 비로봇(object/static) 엔티티 빌더 — 동기.
 * SceneEditor의 updateDimensions/updatePhysics(동기 계약)가 재빌드에 직접 사용한다.
 */
export function buildObjectEntity(entity: EntitySpec, deps: EntityBuildDeps): BuiltEntityHandle {
  if (isRobotSpec(entity)) {
    throw new Error(
      `scene-loader: buildObjectEntity는 robot 엔티티('${entity.id}')를 처리하지 않습니다 — buildEntity를 사용하세요`,
    );
  }
  const { world, renderApi, sync, conveyors } = deps;
  const node = buildVisual(entity, renderApi);
  const physics = entity.physics;

  if (!physics) {
    // 물리 없는 순수 장식: 스펙 트랜스폼으로 1회 배치 (시각 전용 — 불변식 §2.1 예외)
    renderApi.setPose(
      node,
      cloneVec3(entity.transform.position),
      cloneQuat(entity.transform.rotation ?? IDENTITY_QUAT),
    );
    return {
      entityId: entity.id,
      node,
      bound: false,
      dispose: (): void => {
        renderApi.remove(node);
      },
    };
  }

  const init: PhysicsBodyInit = {
    bodyType: physics.bodyType,
    position: cloneVec3(entity.transform.position),
    rotation: entity.transform.rotation ? cloneQuat(entity.transform.rotation) : undefined,
    linearDamping: physics.linearDamping,
    angularDamping: physics.angularDamping,
    gravityScale: physics.gravityScale,
  };
  const bodyId = world.createBody(entity.id, init);
  try {
    for (const collider of physics.colliders) {
      world.createCollider(bodyId, collider, entity.id);
    }
    // 물리 → 시각 단방향 동기화 등록 (트랜스폼의 진실은 물리 — 불변식 §2.1)
    sync.bind(bodyId, node);
    // 컨베이어 표면 구동 등록 (DATA_MODEL §4.2). 벨트 기하는 빌드 시점 pose/collider에서
    // 고정되므로, 편집으로 벨트가 움직이면 재빌드가 새 바인딩을 만든다.
    registerConveyor(entity, deps);
  } catch (err) {
    // 바디는 만들었으나 collider/바인딩/컨베이어 등록에 실패 — 이 엔티티 몫만 되돌린다.
    // ★ unbind가 반드시 먼저다: sync.bind가 성공한 뒤 registerConveyor가 던지면
    //   제거된 바디를 가리키는 바인딩이 남고, 다음 sync.commit()이 죽은 핸들로
    //   world.getPose를 불러 씬 전체가 멈춘다. (bind가 마지막 문장이던 시절에는
    //   생길 수 없던 경로다 — 뒤에 던질 수 있는 문장을 추가하면서 생겼다.)
    sync.unbind(bodyId);
    world.removeEntity(entity.id);
    renderApi.remove(node);
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
    dispose: (): void => {
      sync.unbind(bodyId);
      conveyors.remove(entity.id);
      world.removeEntity(entity.id);
      renderApi.remove(node);
    },
  };
}

/**
 * conveyor 블록이 있으면 벨트 바인딩을 등록한다 (없으면 no-op).
 *
 * SceneEditor도 이 함수를 쓴다: 벨트 기하(진행축·반길이·상면 높이)는 **빌드 시점의
 * pose/collider에 고정**되므로, 배치가 바뀌면(updateTransform) 바인딩을 새 pose로 다시
 * 만들어야 한다. 그렇지 않으면 화면의 벨트는 옮겨졌는데 사물은 옛 자리의 기하로 실려 간다.
 *
 * 벨트 표면은 colliders[0]의 box여야 한다 — validateScene이 이미 강제하지만
 * (§4.2 교차 규칙), 파사드/편집 경로로 들어온 스펙에도 같은 전제가 필요하므로
 * 여기서 한 번 더 확인하고 사람이 읽을 오류를 던진다.
 */
export function registerConveyor(entity: EntitySpec, deps: EntityBuildDeps): void {
  const spec = entity.conveyor;
  if (spec === undefined) return;
  const shape = entity.physics?.colliders[0]?.shape;
  if (shape === undefined || shape.kind !== 'box') {
    throw new Error(
      `scene-loader: 컨베이어 '${entity.id}'의 첫 collider는 box여야 합니다 ` +
        `(현재 '${shape?.kind ?? '없음'}') — 벨트 길이/폭을 알 수 없으면 재순환 지점을 계산할 수 없습니다`,
    );
  }
  // collider offset은 바디 로컬이다 — 벨트 판이 엔티티 원점에서 떨어져 있으면 기하
  // (상면 높이·끝점)가 그만큼 어긋난다. 회전을 적용해 월드 중심으로 환산한다.
  const rotation = cloneQuat(entity.transform.rotation ?? IDENTITY_QUAT);
  const offset = entity.physics?.colliders[0]?.offset?.position;
  const origin = entity.transform.position;
  const center: Vec3 =
    offset === undefined
      ? cloneVec3(origin)
      : (() => {
          const world = rotateVec3(rotation, offset);
          return [origin[0] + world[0], origin[1] + world[1], origin[2] + world[2]];
        })();

  deps.conveyors.add(
    new ConveyorBinding({
      world: deps.world,
      entityId: entity.id,
      spec,
      halfExtents: cloneVec3(shape.halfExtents),
      pose: { position: center, rotation },
    }),
  );
}

function buildVisual(entity: EntitySpec, renderApi: RenderSceneApi): VisualNode {
  const visual = entity.visual;
  switch (visual.kind) {
    case 'primitive': {
      if (!visual.primitive) {
        throw new Error(
          `scene-loader: 엔티티 '${entity.id}'의 visual.kind가 'primitive'인데 visual.primitive 형상이 없습니다`,
        );
      }
      return renderApi.addPrimitive(visual.primitive, visual.color, {
        opacity: visual.opacity,
        edges: visual.edges,
      });
    }
    case 'urdf':
      throw new Error(
        `scene-loader: visual.kind 'urdf'는 robot 엔티티 전용입니다 (엔티티 '${entity.id}'의 type: '${entity.type}')`,
      );
    case 'mesh': {
      // Phase 7 임포트 에셋 — ref('asset://…')는 renderApi.addMeshAsset(글루)이 해석한다
      if (visual.ref === undefined || visual.ref === '') {
        throw new Error(
          `scene-loader: 엔티티 '${entity.id}'의 visual.kind가 'mesh'인데 visual.ref가 없습니다`,
        );
      }
      if (renderApi.addMeshAsset === undefined) {
        throw new Error(
          `scene-loader: 엔티티 '${entity.id}'의 mesh 시각을 만들 수 없습니다 — ` +
            'RenderSceneApi.addMeshAsset 미구현 (임포트 에셋을 지원하지 않는 환경)',
        );
      }
      return renderApi.addMeshAsset(visual.ref);
    }
  }
}

/**
 * 로드된 RobotHandle 위에 물리(바인딩 + kinematic 링크 바디 + collider)를 부착하고
 * 레지스트리에 등록한다 — buildRobotEntity의 "동기" 후반부.
 *
 * SceneEditor.renameEntity가 URDF 재로드 없이(동기 계약) 같은 RobotHandle로 새 id의
 * 물리를 다시 부착할 때도 이 루틴을 재사용한다 — 바디/collider 규약의 단일 진실.
 * 루트 트랜스폼은 손대지 않는다(호출자가 setRootTransform을 이미 반영한 상태).
 *
 * 순서(설계 — core/robot-types.ts 헤더):
 *   1. RobotBinding 생성(관절 상태 진실 소유, 생성자에서 home 적용)
 *   2. gripper.joints 존재 검증 — URDF 관절 목록과 스키마 gripper 설정을 모두 아는
 *      최초 지점 (여기서 거르지 않으면 재생 중 resolveJoint 예외가 Engine rAF 루프를
 *      죽인다 — 로드 시점의 한국어 오류로 앞당긴다)
 *   3. tick() 1회 — FK 뷰를 home 상태로 갱신 (linkBodies가 아직 비어 물리 push 없음)
 *   4. "home FK" 링크 pose를 읽어 collider 보유 링크마다 kinematicPosition 바디 생성
 *      + URDF <collision> 유래 collider(ROBOT 그룹, emitEvents) 부착
 *   5. Registry 등록 + tick() 1회 — home FK를 kinematic 목표로 push(생성 pose와
 *      동일하므로 첫 스텝 이동/스윕 속도 0)
 *
 * 바디를 URDF 초기 자세(모든 관절 0)에서 만들면 첫 스텝에 초기→home 스윕이 생겨
 * |Δpose|/dt의 가짜 kinematic 속도가 잡힌다(접촉 사물에 비현실적 임펄스) — 바디를
 * 처음부터 home FK pose에서 생성해 fresh load와 reset()의 스텝 전 상태를 동일하게
 * 만든다(결정론적 재생, SIMULATION.md §6).
 *
 * 실패 시 자원 정리는 호출자 책임이다 (world.removeEntity + robots.remove).
 */
export function attachRobotPhysics(
  spec: RobotSpec,
  handle: RobotHandle,
  deps: EntityBuildDeps,
): RobotBinding {
  const { world, robots } = deps;

  // linkBodies는 binding과 공유하는 라이브 Map — binding을 먼저 만들어 home 관절
  // 상태를 확정한 뒤, 아래에서 home FK pose로 생성한 바디들을 채워 넣는다.
  const linkBodies = new Map<string, BodyId>();
  const binding = new RobotBinding({
    entityId: spec.id,
    fk: handle,
    world,
    linkBodies,
    jointMap: spec.jointMap,
    jointLimitOverrides: spec.jointLimits,
    home: spec.home,
  });
  if (spec.gripper) {
    for (const jointName of spec.gripper.joints) {
      try {
        binding.resolveJoint(jointName);
      } catch (err) {
        throw new Error(
          `scene-loader: 로봇 '${spec.id}'의 gripper.joints에 존재하지 않는 관절 ` +
            `'${jointName}'이(가) 있습니다 — ${err instanceof Error ? err.message : String(err)}`,
        );
      }
    }
  }

  // FK 뷰(시각 로봇)를 home 관절 상태로 갱신 — linkBodies가 비어 있어 물리 push 없음
  binding.tick();

  // linkColliders 정책: 'none'이면 물리 바디를 만들지 않는다(시각 전용 로봇).
  // 'fromVisual'/'primitive'는 현재 동일하게 URDF <collision> 태그 유래 정의를 쓴다
  // — urdf-loader가 collision 프리미티브를 파싱해 handle.linkColliders로 제공한다.
  const policy = spec.linkColliders ?? DEFAULT_LINK_COLLIDER_POLICY;
  if (policy !== 'none') {
    const homePoses = handle.readLinkPoses();
    const collidesWith: ColliderGroup[] = [...ROBOT_COLLIDES_WITH];
    // 자기 링크(같은 EntityId) 접촉 발행 여부 — 기본 false로 URDF 인접 링크의 상시
    // 접촉 노이즈를 억제한다. 다른 로봇과의 충돌은 이 설정과 무관하게 항상 발행된다.
    world.setSelfContactEnabled(spec.id, spec.selfCollision === true);
    for (const [linkName, colliderDefs] of handle.linkColliders) {
      const pose = homePoses.get(linkName);
      if (!pose) {
        throw new Error(
          `scene-loader: 로봇 '${spec.id}' 링크 '${linkName}'의 FK pose를 읽을 수 없습니다 ` +
            '— RobotFkView.readLinkPoses는 collider 보유 링크를 모두 포함해야 합니다',
        );
      }
      const bodyId = world.createBody(spec.id, {
        bodyType: 'kinematicPosition',
        position: cloneVec3(pose.position),
        rotation: cloneQuat(pose.rotation),
      });
      for (const def of colliderDefs) {
        world.createCollider(
          bodyId,
          {
            shape: def.shape,
            offset: def.offset,
            group: 'ROBOT',
            collidesWith,
            emitEvents: true, // 로봇–사물/환경 충돌 감지가 프로젝트 핵심 (ROADMAP Phase 4)
            friction: ROBOT_LINK_FRICTION,
          },
          spec.id,
        );
      }
      linkBodies.set(linkName, bodyId);
    }
  }

  robots.add(binding);
  // home FK를 kinematic 목표로 push — 링크 바디 생성 pose와 동일하므로
  // 첫 물리 스텝에서 이동(스윕 속도) 없이 home pose가 유지된다.
  binding.tick();
  return binding;
}

/**
 * robot 엔티티 빌더 (Phase 3): renderApi.loadRobot(URDF 로드, Z-up→Y-up 변환은 render 몫)
 * → setRootTransform(엔티티 배치) → attachRobotPhysics(물리 부착 + 레지스트리 등록).
 *
 * 로봇 링크는 RenderSync에 바인딩하지 않는다 — 시각 로봇은 관절 상태에서 직접
 * 갱신되는 FK 그래프 그 자체다(파일 헤더 참조).
 */
async function buildRobotEntity(
  spec: RobotSpec,
  deps: EntityBuildDeps,
): Promise<BuiltEntityHandle> {
  const { world, renderApi, robots } = deps;
  const handle = await renderApi.loadRobot({
    urdfPath: spec.urdf,
    packages: spec.urdfPackages,
  });
  try {
    const rootPose: Pose = {
      position: cloneVec3(spec.transform.position),
      rotation: cloneQuat(spec.transform.rotation ?? IDENTITY_QUAT),
    };
    handle.setRootTransform(rootPose);
    const binding = attachRobotPhysics(spec, handle, deps);
    return {
      entityId: spec.id,
      // 로봇 루트에도 "진실로 되돌리는 지점"을 남긴다 — 비로봇과 대칭.
      // 로봇의 루트 배치는 물리가 아니라 렌더 핸들(URDF outer 그룹)이 들고 있어서,
      // 이 스냅샷이 없으면 커밋되지 않은 드래그 프리뷰를 reset()도 되돌리지 못한다.
      initialPose: { position: cloneVec3(rootPose.position), rotation: cloneQuat(rootPose.rotation) },
      bound: false,
      robot: { handle, binding },
      dispose: (): void => {
        // 로봇은 링크당 바디가 여럿이지만 entityId 하나로 전부 제거된다 (world 매핑 소유)
        world.removeEntity(spec.id);
        robots.remove(spec.id);
        handle.dispose();
      },
    };
  } catch (err) {
    // 바디/binding 중간 실패 — 이 로봇 몫의 물리·렌더 자원만 되돌리고 재던짐
    world.removeEntity(spec.id);
    robots.remove(spec.id);
    handle.dispose();
    throw err;
  }
}

/** ENV 고정 바닥: collider 상면이 정확히 y=0. 시각 메시는 자체 배치(바인딩 불필요). */
function buildGroundEntity(deps: EntityBuildDeps): BuiltEntityHandle {
  const { world, renderApi } = deps;
  const bodyId = world.createBody(GROUND_ENTITY_ID, {
    bodyType: 'fixed',
    position: [0, GROUND_CENTER_Y_M, 0],
  });
  world.createCollider(bodyId, GROUND_COLLIDER_SPEC, GROUND_ENTITY_ID);
  const node = renderApi.addGround();
  return {
    entityId: GROUND_ENTITY_ID,
    bodyId,
    initialPose: { position: [0, GROUND_CENTER_Y_M, 0], rotation: cloneQuat(IDENTITY_QUAT) },
    node,
    bound: false,
    dispose: (): void => {
      world.removeEntity(GROUND_ENTITY_ID);
      renderApi.remove(node);
    },
  };
}

// ── 반환 핸들 ───────────────────────────────────────────────────────

export interface SceneHandle {
  /** 빌드 시점에 생성된 엔티티 id 목록 (environment.ground 사용 시 GROUND_ENTITY_ID 포함) */
  readonly entityIds: readonly EntityId[];
  /** 씬의 로봇 바인딩 보관소 — Engine preStep 훅이 robots.tickAll()을 호출한다. */
  readonly robots: RobotRegistry;
  /**
   * 씬의 컨베이어 보관소 — Engine preStep 훅이 conveyors.tickAll()을 호출한다.
   * 로봇 FK push **뒤**에 돌아야 한다: 같은 tick에서 로봇이 사물을 밀고 벨트가 실어
   * 나르면, 나중에 지정한 속도가 이긴다. 벨트 위에서는 벨트가 이기는 것이 자연스럽다.
   */
  readonly conveyors: ConveyorRegistry;
  /**
   * 엔티티 id → 시각 노드 (프리미티브/바닥 등 비로봇 엔티티만 — 충돌 하이라이트용,
   * UX_DESIGN §3.3). 로봇의 시각은 RobotHandle이 소유하므로 여기 포함되지 않는다 —
   * 로봇 노드 매핑은 글루(main.ts)가 loadRobot 시점에 수집한다. 읽기 전용(순수 시각
   * 소비 전용 — 불변식 §2.1: 물리 pose를 역으로 쓰는 용도 금지).
   * 빌드 시점 스냅샷 — 씬 편집 이후의 최신 상태는 builtEntities를 본다.
   */
  readonly visualNodes: ReadonlyMap<EntityId, VisualNode>;
  /**
   * 엔티티 id → 살아있는 빌드 레코드 (Phase 7). SceneEditor(core/scene-editor.ts)가
   * 이 Map을 공유해 add/remove 시 함께 갱신한다 — reset()/dispose()는 항상 이 Map의
   * "현재" 내용을 순회하므로 편집 후에도 정확하게 동작한다. UI는 이 Map을 직접
   * 변형하지 않는다 (변형은 SceneEditor 연산으로만).
   */
  readonly builtEntities: Map<EntityId, BuiltEntityHandle>;
  /**
   * 모든 바디를 초기 스펙 트랜스폼으로 텔레포트하고 속도를 0으로 만든다
   * — 동일 SceneSpec에서 동일 궤적을 재현하는 결정론적 재생용 (SIMULATION.md §6).
   * 로봇은 applyHome()으로 생성 시점 관절 포즈를 복원하고, 링크 바디를 home FK
   * pose로 즉시 teleport + kinematic 목표 재-push한다 — 리셋 직후(스텝 전)의
   * 물리 상태가 fresh load와 동일하고, 첫 스텝의 스윕 속도가 0이다.
   * 마지막에 sync의 prev 스냅샷도 갱신하므로, 직후 렌더 프레임부터 화면이
   * 리셋된 물리 pose를 비춘다 (stale prev 보간 방지 — CLAUDE.md §2.1).
   */
  reset(): void;
  /** 물리 바디·시각 노드·sync 바인딩·로봇 핸들을 전부 해제한다. */
  dispose(): void;
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
   * robot 엔티티의 URDF 로드가 비동기이므로 build 전체가 async다 (Phase 3).
   * 중간에 실패하면 부분 생성된 자원을 정리한 뒤 오류를 다시 던진다
   * — 반쯤 로드된 씬을 남기지 않는다.
   */
  async build(spec: SceneSpec): Promise<SceneHandle> {
    const robots = new RobotRegistry();
    const conveyors = new ConveyorRegistry();
    const deps: EntityBuildDeps = {
      world: this.world,
      renderApi: this.renderApi,
      sync: this.sync,
      robots,
      conveyors,
    };
    const built = new Map<EntityId, BuiltEntityHandle>();
    try {
      if (spec.environment?.ground) {
        if (spec.entities.some((e) => e.id === GROUND_ENTITY_ID)) {
          throw new Error(
            `scene-loader: 엔티티 id '${GROUND_ENTITY_ID}'는 environment.ground가 예약한 id입니다 — 다른 id를 사용하세요`,
          );
        }
        built.set(GROUND_ENTITY_ID, buildGroundEntity(deps));
      }
      for (const entity of spec.entities) {
        if (built.has(entity.id)) {
          // validateScene이 이미 걸러야 하지만, Map 키 충돌로 레코드가 새는 것을 방어
          throw new Error(`scene-loader: 엔티티 id '${entity.id}'가 중복됩니다 — 씬 내에서 유일해야 합니다`);
        }
        built.set(entity.id, await buildEntity(entity, deps));
      }
    } catch (err) {
      teardown(built);
      throw err;
    }

    const entityIds: readonly EntityId[] = [...built.keys()];
    const visualNodes = new Map<EntityId, VisualNode>();
    for (const b of built.values()) {
      if (b.node) visualNodes.set(b.entityId, b.node);
    }
    return {
      entityIds,
      robots,
      conveyors,
      visualNodes,
      builtEntities: built,
      reset: (): void => {
        for (const b of built.values()) {
          if (b.robot) {
            // 0) 루트 배치를 initialPose(= 스펙의 편집 상태)로 복원한다. 로봇의 루트는
            //    렌더 핸들이 소유하므로 물리 teleport로는 되돌릴 수 없다 — 커밋되지 않은
            //    기즈모 드래그 프리뷰가 남았더라도 되감기가 진실로 되돌린다.
            if (b.initialPose) b.robot.handle.setRootTransform(b.initialPose);
            // 관절 상태(단일 진실)를 home으로 복원한 뒤:
            // 1) teleportLinksToFk — 링크 바디를 home FK pose로 "즉시" 정렬. tick()만
            //    쓰면 다음 스텝까지 물리 pose가 리셋 전 상태로 남고, 첫 스텝이
            //    이전 pose → home 스윕이 되어 가짜 kinematic 속도/임펄스가 생긴다.
            // 2) tick — 다음 스텝 kinematic 목표도 home FK로 지정(스윕 속도 0).
            // 결과: 리셋 직후(스텝 전) 상태 = fresh load 직후 상태 (결정론적 재생).
            b.robot.binding.applyHome();
            b.robot.binding.teleportLinksToFk();
            b.robot.binding.tick();
            continue;
          }
          if (b.bodyId !== undefined && b.initialPose) {
            // teleport는 리셋/편집 전용 API — 시뮬 재생 중 pose 주입에는 쓰지 않는다
            this.world.teleport(b.bodyId, b.initialPose);
          }
        }
        // 컨베이어: 이송 이력을 버리고 다음 tick 구동을 한 번 건너뛴다. 접촉 그래프는
        // 마지막 step()의 결과라 teleport 직후에는 **리셋 전 배치**를 가리킨다 —
        // 그대로 구동하면 공중의 사물에 벨트 속도가 주입돼 되감기 재생이 최초 재생과
        // 달라진다 (ConveyorBinding.skipDriveOnce 주석).
        conveyors.resetAll();
        // ★ 접촉 매니폴드를 버린다. teleport는 좌표만 되돌리고 솔버의 워밍스타트 임펄스는
        // 남기므로, 그것 없이는 "같은 좌표에서 다시 시작"해도 직전 이력에 따라 결과가
        // 갈린다 — 되감기가 결정론적 재생을 준비한다는 계약이 성립하지 않는다
        // (실측 A/B: 새 월드 3/3 동일 vs 되감기 3회 중 2회 다름). PhysicsWorld 계약 주석 참조.
        this.world.clearContactState();
        // prev 스냅샷을 텔레포트된 pose로 갱신 — 갱신하지 않으면 다음 물리 tick의
        // commit() 전까지 sync.apply(alpha<1)가 리셋 전 pose를 계속 그린다
        // ("three.js는 물리의 거울" 불변식 위반, CLAUDE.md §2.1).
        this.sync.commit();
      },
      dispose: (): void => {
        teardown(built);
      },
    };
  }
}

/** 생성된 자원 해제: 각 레코드의 dispose (sync 바인딩 → 물리 바디 → 시각/로봇 순서) */
function teardown(built: Map<EntityId, BuiltEntityHandle>): void {
  for (const b of built.values()) b.dispose();
  built.clear();
}
