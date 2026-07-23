// render/mesh-import.ts — 3D 파일 임포트 파이프라인 (UX_DESIGN §4.4, ROADMAP Phase 7)
//
// render 계층에서 three 메시 로더(GLTF/STL/OBJ)를 아는 유일한 지점이다 (CLAUDE.md §3).
// core 타입은 MeshAssetResolver(core/scene-edit-types) "타입"만 참조한다 — render/urdf.ts가
// core/robot-types를 타입으로만 참조하는 것과 같은 패턴이다.
//
// ── 책임 ─────────────────────────────────────────────────────────────
// - parseModelFile: 확장자 기반 포맷 감지 → three 로더로 ArrayBuffer 파싱 →
//   Object3D + 삼각형 수 + 원본 단위 AABB (다이얼로그 표시용).
// - prepareForScene: 균일 스케일 + (선택) Z-up→Y-up 축 변환 적용, 피벗을 bbox
//   "바닥 중심"으로 재정렬(엔티티 position y=0에서 지면 위에 안착). convex hull용
//   정점 배열(≤ MAX_HULL_POINTS, 그리드 해시 데시메이션)과 AABB half extents /
//   중심 오프셋을 함께 산출한다.
// - extractTrimeshGeometry: trimesh collider용 병합 삼각형 지오메트리(정점+인덱스).
// - MeshAssetStore: 임포트 에셋 저장소 — core MeshAssetResolver 구현.
//   'asset://<n>' ref 발급·해석 지점.
//
// ── 축 변환 규약 ─────────────────────────────────────────────────────
// Z-up→Y-up 변환은 URDF 경로(render/urdf.ts axisFix)와 동일한 -π/2 X축 회전 규약이다.
// URDF는 축 규약이 고정이라 로더가 항상 변환하지만, 임포트 메시는 원본 축을 알 수
// 없으므로 사용자가 다이얼로그에서 선택한다(UX_DESIGN §4.4). 임포트 파이프라인의
// 축 변환 지점은 prepareForScene 한 곳뿐이다.
//
// ── 세션 한정 에셋 (한계 — EXPERIMENTS 기록) ─────────────────────────
// 'asset://' ref는 이 세션의 MeshAssetStore에서만 해석된다. SceneSpec을 저장(💾)해도
// 메시 데이터는 파일에 내장되지 않으므로, 저장 경로는 collectAssetRefs()가 비어 있지
// 않으면 ASSET_SAVE_WARNING_KO를 사용자에게 경고해야 한다(통합자 몫 — main/scene-controls).
// data-URI 내장(scene-edit-types의 serializeRef 방향)은 백로그다.

import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';
import { disposeMeshResources } from './meshes';
import type { MeshAssetResolver } from '../core/scene-edit-types';
import type { SceneSpec, Vec3 } from '../schema/types';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4) ─────────────────────────────

/** convex hull에 넘기는 최대 정점 수 — hull 계산 비용 상한 (그리드 데시메이션 목표) */
export const MAX_HULL_POINTS = 2048;

/** 그리드 데시메이션 시작 분할 수(축당). 40³ = 64,000셀에서 시작해 점차 줄인다. */
const DECIMATION_START_DIVISIONS = 40;
/** 분할 수 축소 비율(3/4) — maxPoints 초과 시 곱해 내려간다 */
const DECIMATION_SHRINK_NUM = 3;
const DECIMATION_SHRINK_DEN = 4;

/**
 * Z-up → Y-up 축 변환 회전각 — render/urdf.ts의 URDF axisFix와 동일 규약(-π/2 X).
 * (urdf.ts의 상수는 비공개라 여기서 같은 값을 독립 선언한다 — 값 계약은 위 헤더 참조)
 */
export const ZUP_TO_YUP_ROTATION_X_RAD = -Math.PI / 2;

/** 임포트 에셋 ref 프리픽스 (core/scene-edit-types의 "asset://…" 규약) */
export const ASSET_REF_PREFIX = 'asset://';

// 임포트 메시 기본 재질 (STL은 재질 정보가 없다) — meshes.ts 프리미티브 재질과 같은 계열
const IMPORTED_MESH_COLOR = '#8f9aa8';
const IMPORTED_MESH_ROUGHNESS = 0.6;
const IMPORTED_MESH_METALNESS = 0.05;

// ── 포맷 감지 ───────────────────────────────────────────────────────

export type ImportFormat = 'glb' | 'gltf' | 'stl' | 'obj';

/** 다이얼로그 "형식 감지" 라벨 (UX_DESIGN §4.4 목업과 동일 표기) */
const FORMAT_LABELS: Readonly<Record<ImportFormat, string>> = {
  glb: 'glTF (.glb)',
  gltf: 'glTF (.gltf)',
  stl: 'STL (.stl)',
  obj: 'OBJ (.obj)',
};

/** 파일 입력 accept 속성용 지원 확장자 (UX_DESIGN §4.4 "지원 형식(MVP)") */
export const SUPPORTED_IMPORT_EXTENSIONS: readonly string[] = ['.glb', '.gltf', '.stl', '.obj'];

/** 파일명 확장자 → 임포트 포맷. 미지원이면 null. (순수 함수) */
export function detectImportFormat(fileName: string): ImportFormat | null {
  const match = /\.([a-z0-9]+)$/i.exec(fileName);
  const ext = match?.[1]?.toLowerCase();
  if (ext === 'glb' || ext === 'gltf' || ext === 'stl' || ext === 'obj') return ext;
  return null;
}

// ── 파싱 ────────────────────────────────────────────────────────────

export interface ModelFileInput {
  name: string;
  buffer: ArrayBuffer;
}

export interface ImportedModel {
  object3D: THREE.Object3D;
  triangleCount: number;
  /** 원본 단위 기준 AABB (스케일/축 변환 "전") — 다이얼로그 표시용 */
  bbox: { min: Vec3; max: Vec3 };
  /** 다이얼로그 "형식 감지" 표시 문자열 (예: 'glTF (.glb)') */
  formatLabel: string;
}

/**
 * 3D 파일을 확장자 기반으로 파싱한다.
 * - .glb/.gltf → GLTFLoader.parseAsync (외부 .bin/텍스처 참조 .gltf는 실패 — 단일 파일 전용)
 * - .stl → STLLoader.parse (재질 없음 → 기본 재질 부여)
 * - .obj → OBJLoader.parse (MTL 미지원 — 단일 파일 임포트라 재질은 로더 기본값)
 *
 * @throws Error(한국어) — 미지원 확장자 / 파싱 실패 / 삼각형 메시 없음.
 */
export async function parseModelFile(file: ModelFileInput): Promise<ImportedModel> {
  const format = detectImportFormat(file.name);
  if (format === null) {
    throw new Error(
      `mesh-import: 지원되지 않는 파일 형식입니다 ('${file.name}') — ` +
        `지원 형식: ${SUPPORTED_IMPORT_EXTENSIONS.join(', ')}`,
    );
  }

  const object3D = await parseByFormat(format, file);
  object3D.traverse((obj) => {
    if (isMesh(obj)) {
      obj.castShadow = true;
      obj.receiveShadow = true;
    }
  });

  const triangleCount = countTriangles(object3D);
  if (triangleCount === 0) {
    throw new Error(
      `mesh-import: '${file.name}'에서 삼각형 메시를 찾지 못했습니다 — 메시가 누락되었거나 빈 파일입니다`,
    );
  }

  const { min, max } = boundsOfPositions(collectVertexPositions(object3D));
  return { object3D, triangleCount, bbox: { min, max }, formatLabel: FORMAT_LABELS[format] };
}

async function parseByFormat(format: ImportFormat, file: ModelFileInput): Promise<THREE.Object3D> {
  switch (format) {
    case 'glb':
    case 'gltf': {
      try {
        // GLTFLoader가 매직 헤더로 glb(바이너리)/gltf(JSON)를 스스로 판별한다
        const gltf = await new GLTFLoader().parseAsync(file.buffer, '');
        return gltf.scene;
      } catch (err) {
        throw parseFailure(
          file.name,
          'glTF',
          err,
          '외부 .bin/텍스처를 참조하는 .gltf는 지원되지 않습니다 — 단일 파일(.glb)을 권장합니다.',
        );
      }
    }
    case 'stl': {
      try {
        const geometry = new STLLoader().parse(file.buffer);
        if (!geometry.hasAttribute('normal')) geometry.computeVertexNormals();
        const material = new THREE.MeshStandardMaterial({
          color: IMPORTED_MESH_COLOR,
          roughness: IMPORTED_MESH_ROUGHNESS,
          metalness: IMPORTED_MESH_METALNESS,
        });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.name = 'imported:stl';
        return mesh;
      } catch (err) {
        throw parseFailure(file.name, 'STL', err);
      }
    }
    case 'obj': {
      try {
        return new OBJLoader().parse(new TextDecoder().decode(file.buffer));
      } catch (err) {
        throw parseFailure(file.name, 'OBJ', err);
      }
    }
  }
}

function parseFailure(fileName: string, formatName: string, err: unknown, hint?: string): Error {
  const reason = err instanceof Error ? err.message : String(err);
  return new Error(
    `mesh-import: ${formatName} 파싱 실패 ('${fileName}') — ${reason}${hint ? ` · ${hint}` : ''}`,
  );
}

// ── 씬 편입 준비 (스케일·축 변환·피벗 재정렬) ────────────────────────

export interface PrepareOptions {
  /** 균일 스케일 (단위 변환 — 예: mm 원본이면 0.001). 0보다 큰 유한한 수. */
  scale: number;
  /** 원본 up 축 — 'z'면 Z-up→Y-up 회전(-π/2 X, URDF 경로와 동일 규약) 적용 */
  upAxis: 'y' | 'z';
}

export interface PreparedModel {
  /**
   * 스케일·축 변환·피벗 재정렬이 적용된 시각 루트.
   * 피벗은 bbox "바닥 중심" — 엔티티 transform.position y=0에서 지면 위에 앉는다.
   */
  object3D: THREE.Object3D;
  /**
   * 변환 적용 후 루트(피벗)-로컬 정점 [x,y,z,...] — convex hull collider용.
   * 중복 제거/데시메이션으로 ≤ MAX_HULL_POINTS (decimatePoints 문서 참조).
   */
  points: Float32Array;
  /** 변환 적용 후 AABB half extents (box collider용) */
  aabbHalfExtents: Vec3;
  /** 피벗(바닥 중심) 기준 AABB 중심 오프셋 — box collider offset용 (≈ [0, halfY, 0]) */
  aabbCenterOffset: Vec3;
}

/**
 * 임포트 모델을 씬 편입 가능한 형태로 준비한다.
 *
 * 구조: root(피벗 — 씬 배치 대상) → fix(스케일 → 축 회전 → 피벗 이동) → 원본 Object3D.
 * 원본 Object3D의 자체 트랜스폼은 건드리지 않는다(재호출 안전) — 모든 보정은 fix
 * 그룹이 소유한다. 반환 points/aabb는 물리 collider가 그대로 소비하는 루트-로컬 값이다.
 *
 * @throws Error(한국어) — scale이 0 이하/비유한, 또는 정점 없는 모델.
 */
export function prepareForScene(model: ImportedModel, opts: PrepareOptions): PreparedModel {
  if (!Number.isFinite(opts.scale) || opts.scale <= 0) {
    throw new Error(
      `mesh-import: 스케일은 0보다 큰 유한한 수여야 합니다 (입력: ${String(opts.scale)})`,
    );
  }

  const root = new THREE.Group();
  root.name = 'imported-asset';
  const fix = new THREE.Group();
  fix.name = 'import-scale-axis-pivot';
  fix.scale.setScalar(opts.scale);
  // ★ 임포트 경로의 Z-up→Y-up 축 변환은 여기 한 곳뿐이다 (파일 헤더의 축 변환 규약)
  if (opts.upAxis === 'z') fix.rotation.x = ZUP_TO_YUP_ROTATION_X_RAD;
  fix.add(model.object3D);
  root.add(fix);
  root.updateMatrixWorld(true);

  // 변환 적용 후 정점(루트-로컬) → bbox → 피벗을 바닥 중심으로 재정렬
  const positions = collectVertexPositions(root);
  const before = boundsOfPositions(positions);
  const shift: Vec3 = [
    -(before.min[0] + before.max[0]) / 2,
    -before.min[1],
    -(before.min[2] + before.max[2]) / 2,
  ];
  // fix.position은 root 공간 이동(자신의 회전·스케일 "이후" 적용)이라 shift와 1:1
  fix.position.set(shift[0], shift[1], shift[2]);
  root.updateMatrixWorld(true);
  for (let i = 0; i < positions.length; i += 3) {
    positions[i] = (positions[i] ?? 0) + shift[0];
    positions[i + 1] = (positions[i + 1] ?? 0) + shift[1];
    positions[i + 2] = (positions[i + 2] ?? 0) + shift[2];
  }

  const after = boundsOfPositions(positions);
  const aabbHalfExtents: Vec3 = [
    (after.max[0] - after.min[0]) / 2,
    (after.max[1] - after.min[1]) / 2,
    (after.max[2] - after.min[2]) / 2,
  ];
  const aabbCenterOffset: Vec3 = [
    (after.max[0] + after.min[0]) / 2,
    (after.max[1] + after.min[1]) / 2,
    (after.max[2] + after.min[2]) / 2,
  ];

  return {
    object3D: root,
    points: decimatePoints(positions, MAX_HULL_POINTS),
    aabbHalfExtents,
    aabbCenterOffset,
  };
}

// ── 정점 수집 / 데시메이션 ──────────────────────────────────────────

/**
 * root 기준 로컬 좌표로 변환한 모든 메시 정점을 평탄 배열 [x,y,z,...]로 수집한다.
 * root가 씬에 부착된 상태라도 root.matrixWorld 역행렬 곱으로 상쇄되므로 항상 root-로컬.
 */
function collectVertexPositions(root: THREE.Object3D): Float32Array {
  root.updateMatrixWorld(true);
  const rootInverse = new THREE.Matrix4().copy(root.matrixWorld).invert();
  const relative = new THREE.Matrix4();
  const v = new THREE.Vector3();

  const meshes: THREE.Mesh[] = [];
  let total = 0;
  root.traverse((obj) => {
    if (!isMesh(obj) || !obj.geometry.hasAttribute('position')) return;
    meshes.push(obj);
    total += obj.geometry.getAttribute('position').count;
  });

  const out = new Float32Array(total * 3);
  let cursor = 0;
  for (const mesh of meshes) {
    const position = mesh.geometry.getAttribute('position');
    relative.multiplyMatrices(rootInverse, mesh.matrixWorld);
    for (let i = 0; i < position.count; i += 1) {
      v.fromBufferAttribute(position, i).applyMatrix4(relative);
      out[cursor] = v.x;
      out[cursor + 1] = v.y;
      out[cursor + 2] = v.z;
      cursor += 3;
    }
  }
  return out;
}

function boundsOfPositions(positions: Float32Array): { min: Vec3; max: Vec3 } {
  if (positions.length === 0) {
    throw new Error('mesh-import: 정점이 없는 모델입니다 — 메시 누락');
  }
  const min: Vec3 = [Infinity, Infinity, Infinity];
  const max: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < positions.length; i += 3) {
    const x = positions[i] ?? 0;
    const y = positions[i + 1] ?? 0;
    const z = positions[i + 2] ?? 0;
    if (x < min[0]) min[0] = x;
    if (x > max[0]) max[0] = x;
    if (y < min[1]) min[1] = y;
    if (y > max[1]) max[1] = y;
    if (z < min[2]) min[2] = z;
    if (z > max[2]) max[2] = z;
  }
  return { min, max };
}

/**
 * 그리드 해시 데시메이션 (convex hull 입력용 정점 상한 보장):
 *
 * 1. 정점 수 ≤ maxPoints면 "정확 중복"만 제거해 그대로 반환한다(입력 순서 유지).
 * 2. 초과 시 AABB를 축당 `divisions`칸의 균등 그리드로 나누고, 점유 셀마다
 *    **AABB 중심에서 가장 먼 정점 1개**를 대표로 남긴다(같은 거리면 최초 등장 —
 *    결정론. 출력 순서는 셀 최초 점유 순). 최초 등장 정점을 그대로 쓰면 셀 안의
 *    비극점이 hull 극점(모서리 정점)을 밀어내 hull이 셀 대각선만큼 시각 메시보다
 *    작아질 수 있다 — 중심 최원점 선택은 극점 손실을 셀 이하 오목부로 한정한다.
 *    대표점은 항상 입력 정점 자체다(셀 중심으로 스냅하지 않는다 — hull이 원본
 *    형상을 벗어나지 않는다: 과대 근사 없음).
 * 3. 여전히 초과면 divisions에 3/4을 곱해 재시도한다. divisions ≤ ⌊∛maxPoints⌋에서는
 *    점유 셀 수 ≤ divisions³ ≤ maxPoints가 보장되므로 루프는 항상 종료한다
 *    (maxPoints=2048이면 ⌊∛2048⌋=12, 12³=1728 ≤ 2048).
 *
 * 한계(문서화): 셀 내부의 오목 세부는 뭉개진다(convex hull 입력이므로 무영향).
 * 볼록 극점도 "중심 최원점이 아닌" 극점이 같은 셀의 더 먼 정점에 밀려날 수는 있으나
 * 그 오차는 셀 크기 이하로 유계다(최소 divisions 12에서 축당 AABB의 ~8%).
 */
export function decimatePoints(
  positions: Float32Array,
  maxPoints: number = MAX_HULL_POINTS,
): Float32Array {
  if (maxPoints < 1) {
    throw new Error(`mesh-import: maxPoints는 1 이상이어야 합니다 (입력: ${maxPoints})`);
  }
  const count = Math.floor(positions.length / 3);
  if (count === 0) return new Float32Array(0);
  if (count <= maxPoints) return dedupExact(positions, count);

  const { min, max } = boundsOfPositions(positions);
  const minDivisions = Math.max(1, Math.floor(Math.cbrt(maxPoints)));
  let divisions = DECIMATION_START_DIVISIONS;
  let selected = selectGridRepresentatives(positions, count, min, max, divisions);
  while (selected.length / 3 > maxPoints && divisions > minDivisions) {
    divisions = Math.max(
      minDivisions,
      Math.floor((divisions * DECIMATION_SHRINK_NUM) / DECIMATION_SHRINK_DEN),
    );
    selected = selectGridRepresentatives(positions, count, min, max, divisions);
  }
  return selected;
}

/** 정확히 같은 좌표의 중복 정점 제거 (입력 순서 유지 — 결정론) */
function dedupExact(positions: Float32Array, count: number): Float32Array {
  const seen = new Set<string>();
  const out: number[] = [];
  for (let i = 0; i < count; i += 1) {
    const x = positions[3 * i] ?? 0;
    const y = positions[3 * i + 1] ?? 0;
    const z = positions[3 * i + 2] ?? 0;
    const key = `${x}|${y}|${z}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(x, y, z);
  }
  return Float32Array.from(out);
}

/**
 * AABB 균등 그리드에서 점유 셀당 AABB 중심 최원점 1개 선택 (decimatePoints 2단계).
 * 결정론: 거리가 정확히 같으면 최초 등장 정점을 유지하고, 출력 순서는 셀 최초 점유
 * 순서다(Map 삽입 순서 — 기존 대표가 더 먼 정점으로 교체돼도 순서는 불변).
 */
function selectGridRepresentatives(
  positions: Float32Array,
  count: number,
  min: Vec3,
  max: Vec3,
  divisions: number,
): Float32Array {
  const cellIndex = (value: number, lo: number, hi: number): number => {
    const extent = hi - lo;
    if (extent <= 0) return 0; // 납작한 축(두께 0)은 단일 셀
    const index = Math.floor(((value - lo) / extent) * divisions);
    return Math.min(divisions - 1, Math.max(0, index));
  };
  const centerX = (min[0] + max[0]) / 2;
  const centerY = (min[1] + max[1]) / 2;
  const centerZ = (min[2] + max[2]) / 2;
  /** 셀 key → 대표 정점 index (삽입 순서 = 셀 최초 점유 순서) */
  const bestIndex = new Map<number, number>();
  /** 셀 key → 대표 정점의 중심 거리² (교체 판정) */
  const bestDistSq = new Map<number, number>();
  for (let i = 0; i < count; i += 1) {
    const x = positions[3 * i] ?? 0;
    const y = positions[3 * i + 1] ?? 0;
    const z = positions[3 * i + 2] ?? 0;
    const key =
      cellIndex(x, min[0], max[0]) +
      divisions * (cellIndex(y, min[1], max[1]) + divisions * cellIndex(z, min[2], max[2]));
    const dx = x - centerX;
    const dy = y - centerY;
    const dz = z - centerZ;
    const distSq = dx * dx + dy * dy + dz * dz;
    const prev = bestDistSq.get(key);
    if (prev === undefined || distSq > prev) {
      bestIndex.set(key, i); // 기존 key 재-set은 Map 삽입 순서를 바꾸지 않는다
      bestDistSq.set(key, distSq);
    }
  }
  const out = new Float32Array(bestIndex.size * 3);
  let cursor = 0;
  for (const i of bestIndex.values()) {
    out[cursor] = positions[3 * i] ?? 0;
    out[cursor + 1] = positions[3 * i + 1] ?? 0;
    out[cursor + 2] = positions[3 * i + 2] ?? 0;
    cursor += 3;
  }
  return out;
}

// ── trimesh 지오메트리 추출 ─────────────────────────────────────────

/**
 * trimesh collider용 병합 삼각형 지오메트리(정점 + 인덱스)를 루트-로컬 좌표로 추출한다.
 * 데시메이션하지 않는다 — trimesh는 정적(Static) 환경 전용이라(DATA_MODEL §2) hull만큼
 * 정점 수에 민감하지 않고 형상 충실도가 우선이다.
 *
 * @throws Error(한국어) — 삼각형이 하나도 없을 때.
 */
export function extractTrimeshGeometry(root: THREE.Object3D): {
  positions: Float32Array;
  indices: Uint32Array;
} {
  root.updateMatrixWorld(true);
  const rootInverse = new THREE.Matrix4().copy(root.matrixWorld).invert();
  const relative = new THREE.Matrix4();
  const v = new THREE.Vector3();

  const positionChunks: number[] = [];
  const indexChunks: number[] = [];
  let base = 0;

  const meshes: THREE.Mesh[] = [];
  root.traverse((obj) => {
    if (isMesh(obj) && obj.geometry.hasAttribute('position')) meshes.push(obj);
  });

  for (const mesh of meshes) {
    const geometry = mesh.geometry;
    const position = geometry.getAttribute('position');
    relative.multiplyMatrices(rootInverse, mesh.matrixWorld);
    for (let i = 0; i < position.count; i += 1) {
      v.fromBufferAttribute(position, i).applyMatrix4(relative);
      positionChunks.push(v.x, v.y, v.z);
    }
    const index = geometry.getIndex();
    if (index !== null) {
      const triCount = Math.floor(index.count / 3) * 3;
      for (let i = 0; i < triCount; i += 1) indexChunks.push(base + index.getX(i));
    } else {
      const triVertexCount = Math.floor(position.count / 3) * 3;
      for (let i = 0; i < triVertexCount; i += 1) indexChunks.push(base + i);
    }
    base += position.count;
  }

  if (indexChunks.length === 0) {
    throw new Error('mesh-import: trimesh로 쓸 삼각형이 없습니다 — 메시 누락');
  }
  return {
    positions: Float32Array.from(positionChunks),
    indices: Uint32Array.from(indexChunks),
  };
}

// ── MeshAssetStore (MeshAssetResolver 구현) ─────────────────────────

export interface MeshAssetBundle {
  /** 시각 원형(prototype) — 씬 인스턴싱 시 clone(true)로 복제해 사용할 것 (아래 클래스 문서) */
  object3D: THREE.Object3D;
  /** convex hull 정점 (prepareForScene의 points — ≤ MAX_HULL_POINTS) */
  points: Float32Array;
  /** trimesh collider 전용 데이터 (extractTrimeshGeometry 산출 — trimesh 전략일 때만) */
  trimesh?: { positions: Float32Array; indices: Uint32Array };
}

/**
 * 세션 한정 임포트 에셋 저장소 — core MeshAssetResolver(scene-edit-types) 구현.
 *
 * - register()가 'asset://<n>' ref를 발급한다 (세션 내 단조 증가 — 재사용 없음).
 * - getPoints: trimesh 데이터가 있으면 trimesh 정점(getIndices와 쌍을 이룸), 없으면
 *   hull 정점. trimesh로 등록된 ref에 convexHull collider를 써도 전체 정점의 hull로
 *   여전히 기하학적으로 정확하다(정점 수만 많아진다).
 * - getObject: 시각 인스턴싱용 원형. 반환 노드는 저장소가 소유한다 — 씬에 넣을 때는
 *   반드시 clone(true)로 복제할 것. (Object3D는 동시에 한 부모만 가질 수 있고,
 *   원형을 직접 씬에 넣으면 씬 해제 시 원형이 함께 파괴된다.)
 * - 자원 수명: clone은 geometry/material을 원형과 "공유"하므로, 씬에서 임포트 시각
 *   노드를 제거할 때 disposeMeshResources를 호출하면 안 된다(공유 자원 파괴 — 다른
 *   인스턴스·원형까지 깨진다). 제거는 removeFromParent까지만, GPU 자원 해제는 저장소
 *   clear()가 일괄 수행한다.
 * - serializeRef(영속화·data-URI 내장)는 이 Phase에 구현하지 않는다 — 파일 헤더의
 *   "세션 한정 에셋" 한계 참조.
 */
export class MeshAssetStore implements MeshAssetResolver {
  private nextId = 1;
  private readonly assets = new Map<string, MeshAssetBundle>();

  /** 준비된 에셋 번들 등록 → 'asset://<n>' ref 발급 */
  register(bundle: MeshAssetBundle): string {
    const ref = `${ASSET_REF_PREFIX}${this.nextId}`;
    this.nextId += 1;
    this.assets.set(ref, bundle);
    return ref;
  }

  /** 시각 인스턴싱용 원형 Object3D (씬 투입 시 clone(true) 필수 — 클래스 문서 참조) */
  getObject(ref: string): THREE.Object3D | undefined {
    return this.assets.get(ref)?.object3D;
  }

  getPoints(ref: string): Float32Array | undefined {
    const asset = this.assets.get(ref);
    if (!asset) return undefined;
    return asset.trimesh ? asset.trimesh.positions : asset.points;
  }

  getIndices(ref: string): Uint32Array | undefined {
    return this.assets.get(ref)?.trimesh?.indices;
  }

  /** 등록된 ref 목록 (등록 순서 — 디버깅/테스트용) */
  refs(): readonly string[] {
    return [...this.assets.keys()];
  }

  /** 모든 에셋과 원형의 geometry/material을 해제한다 (앱 종료/전체 초기화 전용) */
  clear(): void {
    for (const bundle of this.assets.values()) disposeMeshResources(bundle.object3D);
    this.assets.clear();
  }
}

// ── 씬 저장 경고 (세션 한정 에셋 한계 — 통합자 소비용) ───────────────

/** 씬 저장(💾) 시 asset:// 참조가 있으면 사용자에게 보여줄 경고 (한국어) */
export const ASSET_SAVE_WARNING_KO =
  '이 씬에는 임포트된 3D 메시(asset:// 참조)가 포함되어 있습니다. ' +
  '현재 버전은 메시 데이터를 씬 파일에 내장하지 않으므로(세션 한정 에셋), ' +
  '저장된 JSON을 다른 세션에서 불러오면 해당 엔티티는 복원되지 않습니다.';

/**
 * SceneSpec 안의 asset:// 참조를 수집한다(중복 제거, 발견 순서 — 순수 함수).
 * 저장 경로(💾)는 결과가 비어 있지 않으면 ASSET_SAVE_WARNING_KO를 표시해야 한다
 * (통합자 몫 — 파일 헤더의 "세션 한정 에셋").
 */
export function collectAssetRefs(spec: SceneSpec): string[] {
  const refs: string[] = [];
  const add = (ref: string | undefined): void => {
    if (ref !== undefined && ref.startsWith(ASSET_REF_PREFIX) && !refs.includes(ref)) {
      refs.push(ref);
    }
  };
  for (const entity of spec.entities) {
    if (entity.visual.kind === 'mesh') add(entity.visual.ref);
    for (const collider of entity.physics?.colliders ?? []) {
      const shape = collider.shape;
      if (shape.kind === 'convexHull' || shape.kind === 'trimesh') add(shape.ref);
    }
  }
  return refs;
}

// ── 내부 타입 가드 / 집계 ───────────────────────────────────────────

function isMesh(obj: THREE.Object3D): obj is THREE.Mesh {
  return (obj as Partial<THREE.Mesh>).isMesh === true;
}

/** 서브트리의 총 삼각형 수 (인덱스 지오메트리는 index/3, 비인덱스는 position/3) */
function countTriangles(root: THREE.Object3D): number {
  let total = 0;
  root.traverse((obj) => {
    if (!isMesh(obj)) return;
    const geometry = obj.geometry;
    const index = geometry.getIndex();
    if (index !== null) {
      total += Math.floor(index.count / 3);
      return;
    }
    if (geometry.hasAttribute('position')) {
      total += Math.floor(geometry.getAttribute('position').count / 3);
    }
  });
  return total;
}
