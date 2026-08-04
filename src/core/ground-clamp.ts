// core/ground-clamp.ts — 편집 배치의 바닥 하한 (순수 · CLAUDE.md §3 core 계층)
//
// ── 왜 필요한가 ─────────────────────────────────────────────────────
//
// 바닥은 상면이 정확히 y=0인 두께 0.1 m의 고정 슬래브다(scene-loader GROUND_*).
// 편집으로 사물을 그 아래로 내리면 되돌아올 길이 없다:
//
//   · **static/fixed** 바디는 물리가 밀어내지 않는다 — 지하에 영구히 박힌다.
//   · **dynamic** 바디도 슬래브(-0.1 m) 밑으로 내려가면 접촉이 성립하지 않아
//     그대로 무한 낙하한다 — 화면에서 "사라진다".
//   · **robot** 링크는 kinematicPosition이라 자가 교정이 아예 없다.
//
// 즉 지하 배치는 "잘못 놓았다"가 아니라 **작업물 손실**이다. 되돌릴 경로가 있어야
// 사용자가 탐색을 멈추지 않는다(불변식 §2.11)는 규칙의 예방 쪽 짝이다.
//
// ── 무엇을 클램프하는가 ─────────────────────────────────────────────
//
// 엔티티의 **collider가 실제로 차지하는 최저점**이 바닥 상면 위에 오도록 origin y의
// 하한을 계산한다. 중심이 아니라 최저점이 기준이므로, 회전한 박스도 모서리가 바닥을
// 뚫지 않고, 치수를 키워도 아래로 가라앉지 않는다.
//
// 형상을 측정할 수 없는 경우(convexHull/trimesh/fromVisual — ref만 있고 정점은 렌더
// 계층 소유)에는 **원점 자체를 하한**으로 삼는다. 임포트 메시의 피벗은 bbox 바닥
// 중심이고(render/mesh-import.ts) 로봇 루트의 원점은 베이스 링크 발밑이므로, 두 경우
// 모두 "원점 y ≥ 바닥"이 곧 정확한 접지 규칙이다.
//
// ── 계층 ────────────────────────────────────────────────────────────
// schema 타입만 안다. 물리·렌더·UI 비의존 — node 단위 테스트 대상(ground-clamp.test.ts).
// 적용 지점은 ui 통합자다: 직접 조작(기즈모·방향키·인스펙터 입력)과 배치/치수 편집처럼
// **사람이 유발한 편집**만 클램프한다. 씬 JSON 로드는 손대지 않는다 — 수작성 씬의
// 좌표를 조용히 고쳐 쓰면 파일과 화면이 어긋나고, 데이터가 진실이라는 §2.5가 깨진다.

import type { ColliderShape, EntitySpec, Quat, Transform, Vec3 } from '../schema/types';

/** 바닥 상면 y (scene-loader의 GROUND 슬래브 상면과 같은 값) */
export const GROUND_TOP_Y_M = 0;

/**
 * 클램프 판정 허용 오차 (m). 부동소수 오차로 매 커밋마다 "바닥에 맞췄습니다" 토스트가
 * 뜨는 것을 막는다 — 1 µm은 이 씬 스케일(cm 단위 사물)에서 시각적으로 0이다.
 */
export const GROUND_CLAMP_EPS_M = 1e-6;

const IDENTITY_QUAT: Quat = [0, 0, 0, 1];

// ── 회전 수학 (순수) ────────────────────────────────────────────────

/**
 * 쿼터니언 q(바디→월드)에 대해 **월드 Y축을 바디 로컬로 되돌린** 단위벡터 = Rᵀ·ŷ.
 * 회전행렬 R의 2번째 **행**이다. 지지함수(support)를 로컬 좌표에서 계산하기 위한 축.
 */
export function localUpAxis(q: Readonly<Quat>): Vec3 {
  const [x, y, z, w] = q;
  return [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)];
}

/**
 * 로컬 벡터 v를 쿼터니언 q로 회전한 월드 벡터의 **y 성분**만 (R·v의 y).
 * (R·v)_y = R의 2번째 행 · v = (Rᵀ·ŷ) · v 이므로 localUpAxis를 그대로 재사용한다.
 */
export function rotatedY(q: Readonly<Quat>, v: Readonly<Vec3>): number {
  const u = localUpAxis(q);
  return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}

/**
 * 형상의 원점에서 **아래쪽 최저점까지의 거리** (회전 반영, m).
 *
 * box/sphere/capsule/cylinder는 원점 대칭이므로 지지함수 support(u) = support(−u)다 —
 * 로컬 up 축 u = Rᵀ·ŷ 하나로 위·아래 반쪽을 동시에 얻는다. Rapier의 capsule/cylinder는
 * 로컬 Y축 정렬이다(core/world.ts primitiveColliderDescFor).
 *
 * 측정 불가능한 형상(convexHull/trimesh/fromVisual)은 null — 호출자가 "원점이 곧 최저점"
 * 폴백을 쓴다 (파일 헤더).
 */
export function shapeDropBelowOrigin(
  shape: Readonly<ColliderShape>,
  rotation: Readonly<Quat> = IDENTITY_QUAT,
): number | null {
  const u = localUpAxis(rotation);
  switch (shape.kind) {
    case 'box':
      return (
        Math.abs(u[0]) * shape.halfExtents[0] +
        Math.abs(u[1]) * shape.halfExtents[1] +
        Math.abs(u[2]) * shape.halfExtents[2]
      );
    case 'sphere':
      return shape.radius;
    case 'capsule':
      // 원통부 축(로컬 Y) 투영 + 반구 캡은 방향 무관하게 반지름만큼
      return Math.abs(u[1]) * shape.halfHeight + shape.radius;
    case 'cylinder':
      // 축 투영 + 원판 테두리의 축 직교 성분 투영
      return (
        Math.abs(u[1]) * shape.halfHeight +
        Math.hypot(u[0], u[2]) * shape.radius
      );
    default:
      return null; // convexHull | trimesh | fromVisual — 정점은 render 계층 소유
  }
}

/**
 * 엔티티 원점에서 최저점까지의 거리 (모든 collider 중 가장 깊은 것, m).
 *
 * collider의 `offset.position`은 바디 로컬이므로 회전을 적용해 월드 y 기여로 환산한다.
 * physics가 없는 순수 장식 엔티티는 `visual.primitive`를 대신 재는다 — 사용자 눈에는
 * 그 메시가 곧 그 사물이고, 시각만 지하에 잠기는 것도 같은 손실이다.
 *
 * 측정 가능한 형상이 하나도 없으면 0 (= 원점이 곧 최저점 — 파일 헤더의 폴백).
 */
export function entityDropBelowOrigin(
  entity: Readonly<EntitySpec>,
  rotation: Readonly<Quat> = entity.transform.rotation ?? IDENTITY_QUAT,
): number {
  // −Infinity에서 시작한다: 0을 초기값으로 두면 "원점보다 **위**에 있는 collider"
  // (offset이 위로 큰 경우)가 0으로 잘려, 원점을 지하에 둬도 되는 배치를 막게 된다.
  let deepest = Number.NEGATIVE_INFINITY;

  const colliders = entity.physics?.colliders ?? [];
  for (const collider of colliders) {
    const drop = shapeDropBelowOrigin(collider.shape, rotation);
    if (drop === null) continue;
    const offset = collider.offset?.position;
    // 오프셋이 위로 향하면 최저점도 그만큼 올라간다 (부호 반전 — 아래 방향 거리 기준)
    const offsetDrop = offset === undefined ? 0 : -rotatedY(rotation, offset);
    deepest = Math.max(deepest, drop + offsetDrop);
  }

  if (deepest === Number.NEGATIVE_INFINITY && entity.visual.kind === 'primitive' && entity.visual.primitive) {
    const drop = shapeDropBelowOrigin(entity.visual.primitive, rotation);
    if (drop !== null) deepest = drop;
  }

  return deepest === Number.NEGATIVE_INFINITY ? 0 : deepest;
}

/** 엔티티가 바닥 위에 서기 위한 origin y의 하한 (m) */
export function minGroundedY(
  entity: Readonly<EntitySpec>,
  rotation?: Readonly<Quat>,
  groundTopY: number = GROUND_TOP_Y_M,
): number {
  return groundTopY + entityDropBelowOrigin(entity, rotation ?? entity.transform.rotation ?? IDENTITY_QUAT);
}

// ── 클램프 결과 ─────────────────────────────────────────────────────

export interface GroundClampResult {
  /** 하한을 적용한 위치 (클램프되지 않았으면 입력과 같은 값의 새 튜플) */
  readonly position: Vec3;
  /** 실제로 끌어올렸는가 — 통합자가 이때만 사유를 안내한다 */
  readonly clamped: boolean;
  /** 적용된 하한 y (안내 문구에 그대로 쓴다) */
  readonly minY: number;
}

/**
 * 편집으로 제안된 위치를 바닥 위로 끌어올린다.
 *
 * 회전은 **이번 편집이 적용될 회전**을 넘긴다(회전 기즈모 커밋은 위치와 회전을 함께
 * 보내므로, 이전 회전으로 재면 하한이 어긋난다). 생략하면 엔티티의 현재 회전을 쓴다.
 */
export function clampPositionAboveGround(
  entity: Readonly<EntitySpec>,
  position: Readonly<Vec3>,
  rotation?: Readonly<Quat>,
  groundTopY: number = GROUND_TOP_Y_M,
): GroundClampResult {
  const minY = minGroundedY(entity, rotation, groundTopY);
  if (position[1] >= minY - GROUND_CLAMP_EPS_M) {
    return { position: [position[0], position[1], position[2]], clamped: false, minY };
  }
  return { position: [position[0], minY, position[2]], clamped: true, minY };
}

/**
 * 치수/형상이 바뀐 뒤 가라앉은 엔티티를 바닥 위로 되올린다 (updateDimensions 후속).
 *
 * 중심을 고정한 채 크기만 키우면 아래쪽 절반이 그대로 바닥을 뚫는다 — 대부분의 씬
 * 편집기가 "바닥에 놓인 사물은 커져도 바닥에 놓여 있다"로 동작하는 이유다.
 * 새 형상 기준의 하한만 적용하고 x/z와 회전은 건드리지 않는다.
 */
export function groundedTransformForShape(
  entity: Readonly<EntitySpec>,
  nextShape: Readonly<ColliderShape>,
  groundTopY: number = GROUND_TOP_Y_M,
): Transform | null {
  const rotation = entity.transform.rotation ?? IDENTITY_QUAT;
  const drop = shapeDropBelowOrigin(nextShape, rotation);
  if (drop === null) return null;
  const minY = groundTopY + drop;
  const [x, y, z] = entity.transform.position;
  if (y >= minY - GROUND_CLAMP_EPS_M) return null;
  return { ...entity.transform, position: [x, minY, z] };
}

/**
 * "바닥에 붙이기" — 최저점이 바닥 상면에 **정확히** 닿도록 y를 맞춘다.
 * 클램프(하한)와 달리 떠 있는 사물을 내리기도 한다. Unity/Unreal의 snap-to-floor,
 * Blender의 Align to Ground와 같은 조작이며, 지하에 박힌 사물의 구제 경로이기도 하다.
 */
export function snapPositionToGround(
  entity: Readonly<EntitySpec>,
  rotation?: Readonly<Quat>,
  groundTopY: number = GROUND_TOP_Y_M,
): Vec3 {
  const [x, , z] = entity.transform.position;
  return [x, minGroundedY(entity, rotation, groundTopY), z];
}
