// render/urdf.ts — URDF 로딩 래퍼 (urdf-loader → RobotFkView/RobotHandle)
//
// render 계층에서 urdf-loader를 아는 유일한 지점이다 (CLAUDE.md §3).
// core는 여기서 반환하는 RobotHandle(RobotFkView 확장)의 POJO 표면만 본다 —
// three 타입은 RobotHandle의 public 표면에 노출되지 않는다 (core/robot-types.ts 계약).
//
// ── 구조 ─────────────────────────────────────────────────────────────
//   scene ─ outer Group(엔티티 트랜스폼 대상 — setRootTransform가 씀)
//             └─ axisFix Group(rotation.x = -π/2 — URDF Z-up → 내부 Y-up 축 변환의
//                "유일한" 지점, CLAUDE.md §4. 다른 어디서도 축 변환하지 않는다)
//                  └─ URDFRobot (urdf-loader 씬 그래프 = FK 그래프)
//
// ── FK 흐름 (core/robot-types.ts 설계) ───────────────────────────────
// core가 setJointValues로 관절 상태를 밀어 넣으면 urdf-loader 씬 그래프가 FK를 수행하고,
// readLinkPoses()가 링크 월드 pose(Y-up, 엔티티 배치 포함)를 POJO로 돌려준다.
// 시각 로봇은 같은 그래프 그 자체이므로 RenderSync 바인딩 없이 즉시 갱신된다 —
// 로봇 링크는 RenderSync 대상이 아니다.
//
// ── urdf-loader 프리미티브 구현 메모 (^0.12 소스 확인) ───────────────
// <collision>의 box/sphere/cylinder는 "단위 지오메트리 + mesh.scale"로 생성된다:
//   box      → BoxGeometry(1,1,1),        scale = (sizeX, sizeY, sizeZ)
//   sphere   → SphereGeometry(1,…),       scale = (r, r, r)
//   cylinder → CylinderGeometry(1,1,1,…), scale = (r, length, r), rotation.x = π/2
// 따라서 형상 크기는 지오메트리 파라미터 × 링크-상대 TRS의 scale로 산출하고,
// offset(Transform)에는 scale을 제외한 position/rotation만 담는다.

import * as THREE from 'three';
import URDFLoader from 'urdf-loader';
import type {
  URDFCollider,
  URDFJoint,
  URDFLink,
  URDFMimicJoint,
  URDFRobot,
} from 'urdf-loader';
import type { Pose } from '../core/types';
import type { JointInfo, LinkColliderDef, RobotFkView } from '../core/robot-types';
import type { ColliderShape, Transform, Vec3 } from '../schema/types';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4) ─────────────────────────────

/**
 * URDF(Z-up) → 내부 표준 Y-up 축 변환 회전각.
 * 이 회전을 적용하는 axisFix 그룹이 프로젝트 전체에서 유일한 축 변환 지점이다 (CLAUDE.md §4).
 */
const ZUP_TO_YUP_ROTATION_X_RAD = -Math.PI / 2;

// ── 공개 타입 ───────────────────────────────────────────────────────

/**
 * URDF 로봇 핸들. core에 넘기는 RobotFkView + render 전용 수명 관리.
 * public 표면은 POJO만 다룬다 — three 타입 비노출 (core/robot-types.ts 경계).
 */
export interface RobotHandle extends RobotFkView {
  /** 엔티티 배치(SceneSpec transform) 반영 — outer 그룹 pose 설정 (Y-up 월드 기준) */
  setRootTransform(pose: Pose): void;
  /** 씬에서 제거하고 로봇 서브트리의 geometry/material을 해제한다. */
  dispose(): void;
}

export interface LoadUrdfRobotOptions {
  /** URDF 경로. SceneSpec 관례: 'assets/robots/arm6/arm6.urdf' (public/ 루트에서 서빙) */
  urdfPath: string;
  /** ROS 패키지명 → URL 프리픽스 매핑 (urdf-loader의 loader.packages) */
  packages?: Record<string, string>;
}

// ── 순수 변환 헬퍼 (작게 유지 — 미래 단위 테스트 대상) ──────────────

/**
 * SceneSpec의 상대 경로('assets/…')를 사이트 루트 기준 URL('/assets/…')로 정규화한다.
 * 이미 절대 URL(스킴 포함)이거나 선행 슬래시가 있으면 그대로 둔다. (순수 함수)
 */
export function toRootUrl(path: string): string {
  if (path.startsWith('/') || /^[a-z][a-z0-9+.-]*:\/\//i.test(path)) return path;
  return `/${path}`;
}

/**
 * three 프리미티브 지오메트리 + 링크-상대 scale → ColliderShape. (순수 함수)
 *
 * urdf-loader는 단위 지오메트리를 mesh.scale로 키우므로(파일 상단 메모) 실제 치수는
 * "지오메트리 파라미터 × |scale|"이다. 지원 외 지오메트리(파일 메시 등)는 null.
 *
 * cylinder: three와 Rapier 모두 Y축 정렬 실린더다. urdf-loader는 URDF의 Z축 정렬
 * 실린더를 노드 회전(rotation.x = π/2)으로 실현하는데, 그 회전은 링크-상대 offset
 * (decomposeMatrixToTransform의 rotation)에 포함되므로 형상 자체는 Y축 정렬
 * { halfHeight, radius }로 두면 Rapier에서 그대로 맞아떨어진다.
 */
export function primitiveGeometryToShape(
  geometry: THREE.BufferGeometry,
  scale: Vec3,
): ColliderShape | null {
  const sx = Math.abs(scale[0]);
  const sy = Math.abs(scale[1]);
  const sz = Math.abs(scale[2]);

  if (geometry instanceof THREE.BoxGeometry) {
    const { width, height, depth } = geometry.parameters;
    return {
      kind: 'box',
      halfExtents: [(width * sx) / 2, (height * sy) / 2, (depth * sz) / 2],
    };
  }
  if (geometry instanceof THREE.SphereGeometry) {
    // URDF 구는 항상 균일 스케일(r,r,r) — 대표값으로 x축 스케일 사용
    return { kind: 'sphere', radius: geometry.parameters.radius * sx };
  }
  if (geometry instanceof THREE.CylinderGeometry) {
    const { radiusTop, height } = geometry.parameters;
    return {
      kind: 'cylinder',
      halfHeight: (height * sy) / 2,
      radius: radiusTop * sx,
    };
  }
  return null;
}

/** decomposeMatrixToTransform 결과 — offset용 TR와 형상 크기 산정용 scale 분리 */
export interface DecomposedTransform {
  /** position + rotation (scale 제외) — LinkColliderDef.offset에 사용 */
  transform: Transform;
  /** TRS의 scale 성분 — primitiveGeometryToShape에 전달 */
  scale: Vec3;
}

/**
 * TRS 행렬 → { Transform(position/rotation), scale } 분해. (순수 함수)
 * 링크-상대 collider 행렬은 leaf mesh에만 scale이 있으므로 TRS 분해가 정확하다.
 */
export function decomposeMatrixToTransform(matrix: THREE.Matrix4): DecomposedTransform {
  const position = new THREE.Vector3();
  const rotation = new THREE.Quaternion();
  const scale = new THREE.Vector3();
  matrix.decompose(position, rotation, scale);
  return {
    transform: {
      position: [position.x, position.y, position.z],
      rotation: [rotation.x, rotation.y, rotation.z, rotation.w],
    },
    scale: [scale.x, scale.y, scale.z],
  };
}

// ── 내부 타입 가드 (urdf-loader/three 런타임 플래그 기반) ────────────

function isUrdfCollider(obj: THREE.Object3D): obj is URDFCollider {
  return (obj as Partial<URDFCollider>).isURDFCollider === true;
}

function isUrdfLink(obj: THREE.Object3D): obj is URDFLink {
  return (obj as Partial<URDFLink>).isURDFLink === true;
}

function isUrdfJoint(obj: THREE.Object3D): obj is URDFJoint {
  return (obj as Partial<URDFJoint>).isURDFJoint === true;
}

/** mimic 관절 여부 — mimic 대상 관절 구동 시 urdf-loader가 자동 갱신하므로 직접 구동 제외 */
function isMimicJoint(joint: URDFJoint): joint is URDFMimicJoint {
  return typeof (joint as Partial<URDFMimicJoint>).mimicJoint === 'string';
}

function isMesh(obj: THREE.Object3D): obj is THREE.Mesh {
  return (obj as Partial<THREE.Mesh>).isMesh === true;
}

// ── 내부 빌더 ───────────────────────────────────────────────────────

interface BuiltJoints {
  infos: JointInfo[];
  /** setJointValues 라우팅 대상 (구동 가능 관절만 — fixed/mimic/planar/floating 제외) */
  drivable: Map<string, URDFJoint>;
}

/** robot.joints → JointInfo 목록 + 구동 가능 관절 맵 (Object.keys 삽입 순서 = 결정론) */
function buildJointInfos(robot: URDFRobot): BuiltJoints {
  const infos: JointInfo[] = [];
  const drivable = new Map<string, URDFJoint>();
  for (const name of Object.keys(robot.joints)) {
    const joint = robot.joints[name];
    if (!joint) continue;
    const type = joint.jointType;
    // revolute/prismatic/continuous만 구동 대상 — fixed는 물론 planar/floating(URDF
    // 희귀 타입)도 MVP 범위 밖이라 제외한다.
    if (type !== 'revolute' && type !== 'prismatic' && type !== 'continuous') continue;
    if (isMimicJoint(joint)) continue;

    const info: JointInfo = { name, type, initial: joint.angle };
    // continuous는 limits 없음(robot-types 계약). revolute/prismatic은 유한할 때만.
    if (
      type !== 'continuous' &&
      Number.isFinite(joint.limit.lower) &&
      Number.isFinite(joint.limit.upper)
    ) {
      info.limits = [joint.limit.lower, joint.limit.upper];
    }
    infos.push(info);
    drivable.set(name, joint);
  }
  return { infos, drivable };
}

/**
 * 링크 "소유" URDFCollider 노드 수집. 자식 링크로 이어지는 joint/link 경계는 넘지
 * 않는다 — 자손 링크의 collider는 그 링크 몫이다.
 */
function colliderNodesOfLink(link: URDFLink): URDFCollider[] {
  const found: URDFCollider[] = [];
  const visit = (obj: THREE.Object3D): void => {
    for (const child of obj.children) {
      if (isUrdfCollider(child)) {
        found.push(child);
        continue;
      }
      if (isUrdfJoint(child) || isUrdfLink(child)) continue;
      visit(child);
    }
  };
  visit(link);
  return found;
}

/**
 * 링크별 LinkColliderDef 목록을 로드 직후 1회 구축한다.
 *
 * offset은 "링크 프레임 기준 mesh의 상대 변환"(linkWorld⁻¹ × meshWorld)이다.
 * 로드 시점 포즈에서 계산하지만, collider는 링크에 강체 부착이라 링크-상대 변환은
 * 관절값/루트 배치와 무관(pose-independent)하므로 1회 계산으로 안정적이다.
 * 상대 변환에서 조상 체인(axisFix 포함)은 상쇄되므로 링크 프레임 로컬 값이 나온다.
 */
function buildLinkColliders(robot: URDFRobot): Map<string, readonly LinkColliderDef[]> {
  robot.updateMatrixWorld(true); // matrixWorld 유효화 — 이후 링크-상대 계산에 사용
  const linkInverse = new THREE.Matrix4();
  const relative = new THREE.Matrix4();
  const out = new Map<string, readonly LinkColliderDef[]>();

  for (const linkName of Object.keys(robot.links)) {
    const link = robot.links[linkName];
    if (!link) continue;
    linkInverse.copy(link.matrixWorld).invert();

    const defs: LinkColliderDef[] = [];
    for (const colliderNode of colliderNodesOfLink(link)) {
      colliderNode.traverse((obj) => {
        if (!isMesh(obj)) return;
        relative.multiplyMatrices(linkInverse, obj.matrixWorld);
        const { transform, scale } = decomposeMatrixToTransform(relative);
        const shape = primitiveGeometryToShape(obj.geometry, scale);
        if (!shape) {
          console.warn(
            `urdf: 링크 '${linkName}'의 collision 지오메트리(${obj.geometry.type})는 ` +
              'MVP에서 지원되지 않아 건너뜁니다 — box/sphere/cylinder 프리미티브만 지원합니다',
          );
          return;
        }
        defs.push({ shape, offset: transform });
      });
    }
    if (defs.length > 0) out.set(linkName, defs);
  }
  return out;
}

// ── RobotHandle 구현 (비공개 — 공개 표면은 인터페이스뿐) ─────────────

class UrdfRobotHandle implements RobotHandle {
  readonly joints: readonly JointInfo[];
  readonly linkColliders: ReadonlyMap<string, readonly LinkColliderDef[]>;

  private readonly outer: THREE.Group;
  private readonly robot: URDFRobot;
  private readonly drivableJoints: ReadonlyMap<string, URDFJoint>;
  /**
   * readLinkPoses 대상 링크명 — Object.keys(robot.links) 순서(결정론) 중 collider가
   * 하나라도 있는 링크만. 선택 이유: core는 collider 있는 링크에만 kinematic 바디를
   * 만들므로(물리 관련 링크), collider 없는 링크의 pose는 소비처가 없다. 매 tick
   * 호출되는 핫 패스에서 불필요한 pose 산출/할당을 줄인다.
   */
  private readonly physicsLinkNames: readonly string[];

  // 월드 pose 읽기용 재사용 temp (매 tick 신규 할당 방지)
  private readonly tmpPosition = new THREE.Vector3();
  private readonly tmpQuaternion = new THREE.Quaternion();

  constructor(scene: THREE.Object3D, robot: URDFRobot) {
    this.robot = robot;

    // 구조: outer(엔티티 트랜스폼) → axisFix(Z-up→Y-up "유일" 변환 지점) → URDFRobot
    this.outer = new THREE.Group();
    this.outer.name = `urdf:${robot.robotName || 'robot'}`;
    const axisFix = new THREE.Group();
    axisFix.name = 'urdf-axis-zup-to-yup';
    // ★ URDF Z-up → 내부 Y-up 축 변환은 여기 한 곳뿐이다 (CLAUDE.md §4)
    axisFix.rotation.x = ZUP_TO_YUP_ROTATION_X_RAD;
    axisFix.add(robot);
    this.outer.add(axisFix);
    scene.add(this.outer);

    const { infos, drivable } = buildJointInfos(robot);
    this.joints = infos;
    this.drivableJoints = drivable;
    this.linkColliders = buildLinkColliders(robot);
    this.physicsLinkNames = Object.keys(robot.links).filter((name) =>
      this.linkColliders.has(name),
    );
  }

  setJointValues(values: Readonly<Record<string, number>>): void {
    for (const [name, value] of Object.entries(values)) {
      const joint = this.drivableJoints.get(name);
      if (!joint) continue; // 미지/비구동 관절명은 조용히 무시 — 클램프·검증은 core 몫
      joint.setJointValue(value);
    }
  }

  readLinkPoses(): ReadonlyMap<string, Pose> {
    // matrixWorldAutoUpdate는 기본값 유지 — 관절/루트 변경 후 여기서 명시적으로
    // 서브트리 월드 행렬을 갱신한 뒤 읽는다 (렌더 프레임과 무관하게 정확한 FK 보장).
    this.outer.updateMatrixWorld(true);

    const out = new Map<string, Pose>();
    for (const linkName of this.physicsLinkNames) {
      const link = this.robot.links[linkName];
      if (!link) continue;
      link.getWorldPosition(this.tmpPosition);
      link.getWorldQuaternion(this.tmpQuaternion);
      out.set(linkName, {
        position: [this.tmpPosition.x, this.tmpPosition.y, this.tmpPosition.z],
        rotation: [
          this.tmpQuaternion.x,
          this.tmpQuaternion.y,
          this.tmpQuaternion.z,
          this.tmpQuaternion.w,
        ],
      });
    }
    return out;
  }

  setRootTransform(pose: Pose): void {
    this.outer.position.set(pose.position[0], pose.position[1], pose.position[2]);
    this.outer.quaternion.set(
      pose.rotation[0],
      pose.rotation[1],
      pose.rotation[2],
      pose.rotation[3],
    );
  }

  dispose(): void {
    this.outer.removeFromParent();
    this.outer.traverse((obj) => {
      if (!isMesh(obj)) return;
      obj.geometry.dispose();
      const material = obj.material;
      if (Array.isArray(material)) {
        for (const m of material) m.dispose();
      } else {
        material.dispose();
      }
    });
  }
}

// ── 로더 진입점 ─────────────────────────────────────────────────────

/**
 * URDF를 로드해 scene 루트 아래에 부착하고 RobotHandle을 돌려준다.
 *
 * @param scene three 씬 루트(또는 로봇을 담을 컨테이너). outer 그룹이 직접 자식으로
 *              추가된다. readLinkPoses의 "월드" pose는 이 노드의 좌표계 기준이므로
 *              변환 없는 씬 루트를 넘길 것.
 * @throws Error(한국어) — 파일 없음/파싱 실패 등 로드 실패 시.
 */
export async function loadUrdfRobot(
  scene: THREE.Object3D,
  opts: LoadUrdfRobotOptions,
): Promise<RobotHandle> {
  const loader = new URDFLoader();
  // <collision> 태그 파싱 활성 — linkColliders의 원천 (기본값 false라 명시 필수)
  loader.parseCollision = true;
  if (opts.packages) {
    // 패키지 프리픽스도 사이트 루트 기준으로 정규화 (public/ 정적 서빙 관례)
    const normalized: Record<string, string> = {};
    for (const [pkg, prefix] of Object.entries(opts.packages)) {
      normalized[pkg] = toRootUrl(prefix);
    }
    loader.packages = normalized;
  }

  const url = toRootUrl(opts.urdfPath);
  let robot: URDFRobot;
  try {
    robot = await loader.loadAsync(url);
  } catch (err) {
    const reason = err instanceof Error ? err.message : String(err);
    throw new Error(
      `URDF 로드 실패: '${opts.urdfPath}' — 파일이 없거나 파싱에 실패했습니다 (${reason})`,
    );
  }
  return new UrdfRobotHandle(scene, robot);
}
