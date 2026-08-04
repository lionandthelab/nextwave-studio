// render/meshes.ts — 프리미티브 시각 메시 팩토리 (three.js)
//
// ColliderShape 프리미티브(box/sphere/capsule/cylinder) → THREE.Mesh 생성.
// 시각 메시는 물리 collider와 독립 객체다(CLAUDE.md §2.2) — 여기서 만든 메시를
// 물리에 쓰는 일은 없고, scene-loader가 바디를 만들고 RenderSync가 pose만 밀어 넣는다.
//
// 치수 대응 (Y-up, 미터 — Rapier와 three.js 모두 Y축 정렬 프리미티브):
// - box:      Rapier halfExtents → THREE.BoxGeometry는 전체 길이 = 2 × halfExtents
// - sphere:   radius 동일
// - capsule:  Rapier capsule(halfHeight, radius)의 halfHeight는 원통부의 절반.
//             THREE.CapsuleGeometry(radius, length)의 length는 원통부 전체 길이
//             → length = 2 × halfHeight (전체 높이 = length + 2 × radius, 양쪽 동일)
// - cylinder: THREE.CylinderGeometry(r, r, height) → height = 2 × halfHeight
//
// convexHull/trimesh/fromVisual은 에셋 파이프라인(Phase 3+)에서 처리 — 여기서는 명시 오류.

import * as THREE from 'three';
import type { ColliderShape } from '../schema/types';

// ── 기본 색 (타입별 구분 가능한 차분한 팔레트) ──────────────────────
const DEFAULT_BOX_COLOR = '#c0392b';      // 붉은 벽돌 (DATA_MODEL 예시와 일치)
const DEFAULT_SPHERE_COLOR = '#2980b9';   // 청색
const DEFAULT_CAPSULE_COLOR = '#27ae60';  // 녹색
const DEFAULT_CYLINDER_COLOR = '#d4930f'; // 호박색

// ── 재질/세그먼트 상수 (매직넘버 금지, CLAUDE.md §4) ────────────────
const PRIMITIVE_ROUGHNESS = 0.55;
const PRIMITIVE_METALNESS = 0.05;
const SPHERE_WIDTH_SEGMENTS = 32;
const SPHERE_HEIGHT_SEGMENTS = 16;
const CAPSULE_CAP_SEGMENTS = 8;
const CAPSULE_RADIAL_SEGMENTS = 24;
const CYLINDER_RADIAL_SEGMENTS = 32;

// ── 감지 존(트리거 볼륨) 표현 상수 ──────────────────────────────────
/**
 * 모서리 선의 밝기 배수. 면보다 밝아야 반투명 면 위에서 경계가 읽힌다.
 * 색 자체는 씬 데이터(visual.color)에서 오고 여기서는 명도만 올린다.
 */
const EDGE_LIGHTEN = 0.55;
/** 모서리 선의 최소 명도 — 어두운 씬 색이 배경에 묻히지 않게 */
const EDGE_MIN_LIGHTNESS = 0.45;

// ── 바닥 상수 ───────────────────────────────────────────────────────
const DEFAULT_GROUND_SIZE_M = 20;
const GROUND_THICKNESS_M = 0.02;
/** 그리드 헬퍼(y=0 라인)와의 z-fighting 회피용 시각적 침하량. 물리 ENV 면은 y=0 유지. */
const GROUND_VISUAL_SINK_M = 0.001;
const GROUND_COLOR = '#23262b';
const GROUND_ROUGHNESS = 0.95;
const GROUND_METALNESS = 0.0;

/**
 * 프리미티브 collider 형상에 대응하는 시각 메시를 만든다.
 * 그림자 캐스트/수신 활성. 색 미지정 시 타입별 기본색.
 *
 * @throws convexHull / trimesh / fromVisual — 프리미티브가 아니므로 명시적으로 실패
 *         (에셋 메시 파이프라인은 Phase 3+).
 */
export interface PrimitiveMeshStyle {
  /** 0..1 (기본 1 = 불투명) — 감지 존처럼 통과 가능한 부피에 쓴다 */
  readonly opacity?: number;
  /** 모서리 선을 덧그린다 (반투명 부피의 경계를 읽히게) */
  readonly edges?: boolean;
}

export function primitiveMesh(
  shape: ColliderShape,
  color?: string,
  style?: PrimitiveMeshStyle,
): THREE.Mesh {
  switch (shape.kind) {
    case 'box':
      return buildMesh(
        new THREE.BoxGeometry(
          2 * shape.halfExtents[0], 2 * shape.halfExtents[1], 2 * shape.halfExtents[2],
        ),
        color ?? DEFAULT_BOX_COLOR,
        shape.kind,
        style,
      );
    case 'sphere':
      return buildMesh(
        new THREE.SphereGeometry(shape.radius, SPHERE_WIDTH_SEGMENTS, SPHERE_HEIGHT_SEGMENTS),
        color ?? DEFAULT_SPHERE_COLOR,
        shape.kind,
        style,
      );
    case 'capsule':
      // CapsuleGeometry의 length 인자는 원통부 길이(캡 제외) — Rapier halfHeight×2와 1:1.
      // 둘 다 Y축 정렬이라 추가 회전 불필요.
      return buildMesh(
        new THREE.CapsuleGeometry(
          shape.radius, 2 * shape.halfHeight, CAPSULE_CAP_SEGMENTS, CAPSULE_RADIAL_SEGMENTS,
        ),
        color ?? DEFAULT_CAPSULE_COLOR,
        shape.kind,
        style,
      );
    case 'cylinder':
      return buildMesh(
        new THREE.CylinderGeometry(
          shape.radius, shape.radius, 2 * shape.halfHeight, CYLINDER_RADIAL_SEGMENTS,
        ),
        color ?? DEFAULT_CYLINDER_COLOR,
        shape.kind,
        style,
      );
    case 'convexHull':
    case 'trimesh':
    case 'fromVisual':
      throw new Error(
        `primitiveMesh: '${shape.kind}' is not a primitive shape — ` +
        `asset-derived meshes are Phase 3+ (only box/sphere/capsule/cylinder supported)`,
      );
  }
}

/**
 * 바닥 시각 메시: y=0에 상면이 오는 대형 얇은 박스. 그림자 수신 전용.
 *
 * 시각 전용이다 — 물리 ENV 바디(fixed + ENV 그룹 collider) 생성은 scene-loader의
 * 책임이며 이 메시와 무관하다(CLAUDE.md §2.2). 정적이므로 RenderSync 바인딩도 불필요.
 * 상면은 그리드 헬퍼와의 z-fighting을 피해 GROUND_VISUAL_SINK_M만큼 살짝 내려 둔다.
 */
export function groundMesh(sizeM: number = DEFAULT_GROUND_SIZE_M): THREE.Mesh {
  const geometry = new THREE.BoxGeometry(sizeM, GROUND_THICKNESS_M, sizeM);
  const material = new THREE.MeshStandardMaterial({
    color: GROUND_COLOR,
    roughness: GROUND_ROUGHNESS,
    metalness: GROUND_METALNESS,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.y = -(GROUND_THICKNESS_M / 2) - GROUND_VISUAL_SINK_M;
  mesh.receiveShadow = true;
  mesh.castShadow = false;
  mesh.name = 'ground';
  return mesh;
}

/**
 * 씬에서 제거된 시각 노드의 GPU 자원(geometry/material)을 해제한다.
 * primitiveMesh/groundMesh는 메시마다 독립 geometry/material을 만들므로 공유 자원
 * 이중 해제 위험이 없다. 로봇 URDF 메시는 RobotHandle.dispose(render/urdf.ts)가
 * 동일한 방식으로 자기 몫을 해제한다 — 이 함수는 비로봇 노드용 대칭 짝이다.
 */
export function disposeMeshResources(root: THREE.Object3D): void {
  root.traverse((obj) => {
    // Mesh와 LineSegments(모서리 선) 모두 geometry/material을 소유한다 — 선만 빠뜨리면
    // 씬을 전환할 때마다 감지 존 개수만큼 GPU 자원이 샌다.
    const drawable = obj as Partial<THREE.Mesh & THREE.LineSegments>;
    if (drawable.isMesh !== true && drawable.isLineSegments !== true) return;
    drawable.geometry?.dispose();
    const material = drawable.material;
    if (material === undefined) return;
    if (Array.isArray(material)) {
      for (const m of material) m.dispose();
    } else {
      material.dispose();
    }
  });
}

function buildMesh(
  geometry: THREE.BufferGeometry,
  colorHex: string,
  kind: string,
  style?: PrimitiveMeshStyle,
): THREE.Mesh {
  const opacity = style?.opacity ?? 1;
  const translucent = opacity < 1;
  const material = new THREE.MeshStandardMaterial({
    color: colorHex,
    roughness: PRIMITIVE_ROUGHNESS,
    metalness: PRIMITIVE_METALNESS,
    transparent: translucent,
    opacity,
    // 반투명 부피는 깊이를 쓰지 않는다 — 쓰면 자기 뒷면이 앞면을 지워 속이 빈 껍데기로
    // 보이고, 안에 든 사물이 존에 들어간 순간 사라진 것처럼 보인다.
    depthWrite: !translucent,
    side: translucent ? THREE.DoubleSide : THREE.FrontSide,
  });
  const mesh = new THREE.Mesh(geometry, material);
  // 통과 가능한 부피가 그림자를 드리우면 다시 "단단한 물체"로 읽힌다
  mesh.castShadow = !translucent;
  mesh.receiveShadow = !translucent;
  mesh.name = `primitive:${kind}`;

  if (style?.edges === true) {
    const edgeColor = new THREE.Color(colorHex);
    // 면보다 밝게 — 반투명 면 위에서 경계가 보이도록 (HSL 명도만 올린다)
    const hsl = { h: 0, s: 0, l: 0 };
    edgeColor.getHSL(hsl);
    edgeColor.setHSL(hsl.h, hsl.s, Math.max(EDGE_MIN_LIGHTNESS, hsl.l + EDGE_LIGHTEN));
    const lines = new THREE.LineSegments(
      new THREE.EdgesGeometry(geometry),
      new THREE.LineBasicMaterial({ color: edgeColor, transparent: true, opacity: 1 }),
    );
    lines.name = `primitive:${kind}:edges`;
    // 메시의 자식이므로 pose 동기화(RenderSync)와 선택 아웃라인이 그대로 따라온다
    mesh.add(lines);
  }
  return mesh;
}
