// planner/scene-context.ts — SceneSpec(+실시간 상태) → SceneContext 그라운딩 직렬화.
// 규범: docs/PLANNER.md §2. 이 모듈은 순수하다 — schema 타입만 안다(../schema).
// 물리/렌더/UI/DOM에 의존하지 않는다 (CLAUDE.md §3 계층 규칙: planner → schema).
//
// 목적: LLM(또는 규칙 기반 플래너)이 존재하지 않는 로봇/관절/물체를 지어내지 않도록,
// 현재 씬을 구조화한 컨텍스트로 전달한다. 방향어 해석 규약도 함께 노출한다(§2, §4.1).

import type { EntitySpec, RobotSpec, SceneSpec } from '../schema';
import { isRobotSpec } from '../schema';

// ── 규약 상수 ────────────────────────────────────────────────────────

/**
 * 방향어 해석 규약 (내부 Y-up, 미터, 라디안 기준). 컨텍스트에 포함되고, 플래너 결과의
 * assumptions로도 사용자에게 다시 노출된다 (PLANNER.md §2/§4.1).
 */
export const DIRECTION_CONVENTIONS = {
  left: '-X',
  right: '+X',
  up: '+Y',
  forward: '+Z',
  back: '-Z',
} as const;

export type DirectionConventions = typeof DIRECTION_CONVENTIONS;

/**
 * 플래너가 생성할 수 있는 ControlStep 종류 (capabilities.stepKinds). schema의
 * CONTROL_STEP_KINDS와 달리 `moveToPose`는 제외된다 — IK 솔버 미구현(백로그)이라
 * 플래너가 카테시안 목표를 지어내지 못하게 한다 (ROADMAP 백로그).
 */
export const PLANNER_STEP_KINDS = [
  'moveJoints',
  'setJoints',
  'gripper',
  'wait',
  'waitForCollision',
  'label',
  'goto',
] as const;

export type PlannerStepKind = (typeof PLANNER_STEP_KINDS)[number];

/**
 * jointLimits가 스펙에 없을 때 사용하는 기본 관절 범위 (rad).
 * 한계(문서화): SceneSpec만으로는 URDF의 실제 limits/joint type을 알 수 없다. URDF를
 * 로드하기 전까지 revolute 기본 범위로 추정한다 — buildContext(live) 경로에서
 * 실제 관절값(current)은 채워지지만 limits/type의 정밀도는 이 추정에 머문다.
 */
export const DEFAULT_REVOLUTE_LIMIT: readonly [number, number] = [-Math.PI, Math.PI];

// ── 입력: 실시간 상태 스냅샷 (POJO — main 글루가 레지스트리에서 채운다) ──

/**
 * 현재 실행 중인 월드의 상태. main 글루(ui)가 로봇 레지스트리·물리 pose에서 채워
 * 전달한다. 없으면 buildContext는 home 포즈/spec transform을 사용한다.
 */
export interface WorldSnapshot {
  /** 로봇 id → { 관절명 → 현재값(rad/m) } */
  jointValuesByRobot?: Record<string, Record<string, number>>;
  /** 엔티티 id → 현재 월드 위치 [x,y,z] */
  positionsByEntity?: Record<string, [number, number, number]>;
}

// ── SceneContext (PLANNER.md §2와 1:1, directions 확장) ───────────────

export interface SceneContextJoint {
  name: string;
  type: 'revolute' | 'prismatic';
  /** [min, max] — rad(revolute) 또는 m(prismatic) */
  limits: [number, number];
  /** 현재값 (live 없으면 home, home 없으면 0) */
  current: number;
}

export interface SceneContextGripper {
  name: string;
  open: number;
  close: number;
}

export interface SceneContextRobot {
  id: string;
  joints: SceneContextJoint[];
  gripper?: SceneContextGripper;
  homePose: Record<string, number>;
}

export interface SceneContextObject {
  id: string;
  type: 'object' | 'static';
  position: [number, number, number];
  /** box → {w,h,l}, sphere → {radius}, cylinder/capsule → {radius,height} */
  dimensions?: Record<string, number>;
  tags?: string[];
}

export interface SceneContext {
  robots: SceneContextRobot[];
  objects: SceneContextObject[];
  capabilities: {
    stepKinds: string[];
  };
  frame: 'y-up, meters, radians';
  /** 방향어 해석 규약 — 컨텍스트에 포함 + assumptions로 노출 */
  directions: DirectionConventions;
}

// ── 빌더 ─────────────────────────────────────────────────────────────

/**
 * SceneSpec(+선택적 실시간 스냅샷) → SceneContext.
 * live가 없으면 관절은 home/0, 물체 위치는 spec transform을 사용한다.
 */
export function buildContext(scene: SceneSpec, live?: WorldSnapshot): SceneContext {
  const robots: SceneContextRobot[] = [];
  const objects: SceneContextObject[] = [];

  for (const entity of scene.entities) {
    if (isRobotSpec(entity)) {
      robots.push(buildRobotContext(entity, live));
    } else if (entity.type === 'object' || entity.type === 'static') {
      objects.push(buildObjectContext(entity, live));
    }
  }

  return {
    robots,
    objects,
    capabilities: { stepKinds: [...PLANNER_STEP_KINDS] },
    frame: 'y-up, meters, radians',
    directions: DIRECTION_CONVENTIONS,
  };
}

/**
 * moveJoints/setJoints로 제어 가능한 관절 이름 목록 유도:
 * home ∪ jointMap ∪ jointLimits (등장 순서 보존).
 * SceneSpec만으로는 URDF 관절 전체를 알 수 없으므로 스펙에 나타난 관절명만 노출한다.
 *
 * gripper.joints는 **의도적으로 제외**한다. 그리퍼는 별도 제어면이며 `gripper` step
 * (state: open/close/0..1)으로 구동되고, 컨텍스트에는 SceneContextRobot.gripper로
 * 노출된다 — moveJoints/setJoints 대상이 아니다. 또한 validateSequence의 checkJointNames는
 * gripper.joints를 knownJoints에서 제외하므로(그리퍼 관절 존재 검증은 URDF를 아는 로드
 * 시점, scene-loader.attachRobotPhysics 담당), 이 관절을 joints 배열로 광고하면 LLM이
 * moveJoints 대상으로 지어내 검증 거부되는 그라운딩⇄검증 비대칭이 생긴다. 여기서
 * joints 배열을 checkJointNames의 knownJoints와 정확히 일치시켜 비대칭을 제거한다.
 * (gripper.joints가 home 등에도 선언되면 그 경로로 자연히 포함되어 검증도 통과한다.)
 */
function robotJointNames(robot: RobotSpec): string[] {
  const names: string[] = [];
  const add = (key: string): void => {
    if (!names.includes(key)) names.push(key);
  };
  if (robot.home) for (const key of Object.keys(robot.home)) add(key);
  if (robot.jointMap) for (const key of Object.keys(robot.jointMap)) add(key);
  if (robot.jointLimits) for (const key of Object.keys(robot.jointLimits)) add(key);
  return names;
}

function buildRobotContext(robot: RobotSpec, live?: WorldSnapshot): SceneContextRobot {
  const home = robot.home ?? {};
  const liveJoints = live?.jointValuesByRobot?.[robot.id];

  const joints: SceneContextJoint[] = robotJointNames(robot).map((name) => {
    const specLimit = robot.jointLimits?.[name];
    const limits: [number, number] = specLimit
      ? [specLimit[0], specLimit[1]]
      : [DEFAULT_REVOLUTE_LIMIT[0], DEFAULT_REVOLUTE_LIMIT[1]];
    const current = liveJoints?.[name] ?? home[name] ?? 0;
    // type: SceneSpec은 관절 종류를 담지 않는다 — revolute 기본값(문서화된 한계).
    return { name, type: 'revolute', limits, current };
  });

  const result: SceneContextRobot = {
    id: robot.id,
    joints,
    homePose: { ...home },
  };
  if (robot.gripper) {
    result.gripper = { name: 'gripper', open: robot.gripper.open, close: robot.gripper.close };
  }
  return result;
}

function buildObjectContext(entity: EntitySpec, live?: WorldSnapshot): SceneContextObject {
  const livePos = live?.positionsByEntity?.[entity.id];
  const specPos = entity.transform.position;
  const position: [number, number, number] = livePos
    ? [livePos[0], livePos[1], livePos[2]]
    : [specPos[0], specPos[1], specPos[2]];

  const object: SceneContextObject = {
    id: entity.id,
    type: entity.type === 'static' ? 'static' : 'object',
    position,
  };
  const dimensions = dimensionsOf(entity);
  if (dimensions) object.dimensions = dimensions;
  if (entity.tags && entity.tags.length > 0) object.tags = [...entity.tags];
  return object;
}

/**
 * visual.primitive에서 사람이 읽을 치수를 유도한다.
 * box: {w,h,l} = 2*halfExtents · sphere: {radius} · cylinder/capsule: {radius, height}.
 * convexHull/trimesh/fromVisual/메시/urdf: 치수 불명 → undefined.
 */
function dimensionsOf(entity: EntitySpec): Record<string, number> | undefined {
  const primitive = entity.visual.primitive;
  if (!primitive) return undefined;
  switch (primitive.kind) {
    case 'box':
      return {
        w: 2 * primitive.halfExtents[0],
        h: 2 * primitive.halfExtents[1],
        l: 2 * primitive.halfExtents[2],
      };
    case 'sphere':
      return { radius: primitive.radius };
    case 'cylinder':
    case 'capsule':
      return { radius: primitive.radius, height: 2 * primitive.halfHeight };
    default:
      return undefined;
  }
}
