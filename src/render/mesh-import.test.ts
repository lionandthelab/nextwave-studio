// render/mesh-import.test.ts — 3D 임포트 파이프라인 순수 부분 단위 테스트 (node, DOM 비의존)
//
// 대상 (Phase 7 임포트 요구사항):
// - parseModelFile: STL/OBJ 파싱(three 로더는 node에서 순수 동작)·형식 라벨·오류 경로.
//   glTF는 로더 특성상 브라우저 자원(fetch/이미지)이 얽힐 수 있어 브라우저 게이트 몫.
// - decimatePoints: ≤ MAX_HULL_POINTS 상한·중복 제거·대표점이 항상 입력 정점.
// - prepareForScene: bbox 수학·스케일·피벗 바닥-중심 재정렬·Z-up→Y-up 회전 수학.
// - extractTrimeshGeometry: 병합 인덱스 무결성.
// - MeshAssetStore: ref 발급('asset://<n>')·resolver 계약(getPoints/getIndices).
// - collectAssetRefs: 저장 경고 대상 수집.
// - import-dialog 순수 함수: 폼 상태 → EntitySpec 매핑·trimesh→Static 강제
//   (다이얼로그 DOM 조립은 얇은 글루 — 브라우저 게이트 몫, inspector.test.ts와 동일 방침).

import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import {
  ASSET_REF_PREFIX,
  MAX_HULL_POINTS,
  MeshAssetStore,
  collectAssetRefs,
  decimatePoints,
  detectImportFormat,
  extractTrimeshGeometry,
  parseModelFile,
  prepareForScene,
} from './mesh-import';
import type { ImportedModel } from './mesh-import';
import {
  buildImportedEntitySpec,
  defaultEntityIdFromFileName,
  forcedEntityKind,
  formatBboxSizeM,
} from '../ui/library/import-dialog';
import type { ImportFormState } from '../ui/library/import-dialog';
import { templateByKey } from '../ui/library/templates';
import { validateScene } from '../schema';
import type { EntitySpec, SceneSpec, Vec3 } from '../schema';

// ── 테스트 헬퍼 ─────────────────────────────────────────────────────

function toBuffer(text: string): ArrayBuffer {
  const bytes = new TextEncoder().encode(text);
  // Uint8Array.buffer는 ArrayBufferLike — 이 테스트에서는 항상 실제 ArrayBuffer다
  return bytes.buffer as ArrayBuffer;
}

/** prepareForScene 입력용 최소 ImportedModel (bbox/count는 prepare가 재계산 — 더미) */
function importedModelOf(object3D: THREE.Object3D): ImportedModel {
  return {
    object3D,
    triangleCount: 1,
    bbox: { min: [0, 0, 0], max: [0, 0, 0] },
    formatLabel: 'test',
  };
}

function meshOfGeometry(geometry: THREE.BufferGeometry): THREE.Mesh {
  return new THREE.Mesh(geometry, new THREE.MeshBasicMaterial());
}

/** Float32Array [x,y,z,...] → 좌표 트리플 목록 (결정론 비교용 정렬) */
function toSortedTriples(points: Float32Array): Array<[number, number, number]> {
  const triples: Array<[number, number, number]> = [];
  for (let i = 0; i < points.length; i += 3) {
    triples.push([points[i] ?? 0, points[i + 1] ?? 0, points[i + 2] ?? 0]);
  }
  triples.sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
  return triples;
}

function sceneWith(entities: EntitySpec[]): SceneSpec {
  return {
    name: 'import-test',
    version: 1,
    gravity: [0, -9.81, 0],
    timestepHz: 240,
    entities,
  };
}

const ASCII_STL = `solid tri
facet normal 0 0 1
  outer loop
    vertex 0 0 0
    vertex 1 0 0
    vertex 0 1 0
  endloop
endfacet
endsolid tri
`;

const OBJ_TRIANGLE = 'v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n';

// ── detectImportFormat ──────────────────────────────────────────────

describe('detectImportFormat', () => {
  it('지원 확장자를 대소문자 무관하게 감지한다', () => {
    expect(detectImportFormat('model.glb')).toBe('glb');
    expect(detectImportFormat('Model.GLTF')).toBe('gltf');
    expect(detectImportFormat('part.STL')).toBe('stl');
    expect(detectImportFormat('a.b.obj')).toBe('obj');
  });

  it('미지원/무확장자는 null', () => {
    expect(detectImportFormat('model.fbx')).toBeNull();
    expect(detectImportFormat('model')).toBeNull();
  });
});

// ── parseModelFile (STL/OBJ — node 순수 경로) ───────────────────────

describe('parseModelFile', () => {
  it('ASCII STL: 삼각형 1개 + 형식 라벨 + 원본 단위 bbox', async () => {
    const model = await parseModelFile({ name: 'tri.stl', buffer: toBuffer(ASCII_STL) });
    expect(model.formatLabel).toBe('STL (.stl)');
    expect(model.triangleCount).toBe(1);
    expect(model.bbox.min).toEqual([0, 0, 0]);
    expect(model.bbox.max).toEqual([1, 1, 0]);
  });

  it('OBJ: 삼각형 1개 + 형식 라벨', async () => {
    const model = await parseModelFile({ name: 'tri.obj', buffer: toBuffer(OBJ_TRIANGLE) });
    expect(model.formatLabel).toBe('OBJ (.obj)');
    expect(model.triangleCount).toBe(1);
  });

  it('미지원 확장자는 한국어 오류로 거부한다', async () => {
    await expect(
      parseModelFile({ name: 'model.fbx', buffer: toBuffer('') }),
    ).rejects.toThrow(/지원되지 않는 파일 형식/);
  });

  it('삼각형 없는 OBJ(정점만)는 "메시 누락" 오류로 거부한다', async () => {
    await expect(
      parseModelFile({ name: 'points.obj', buffer: toBuffer('v 0 0 0\nv 1 0 0\n') }),
    ).rejects.toThrow(/삼각형 메시를 찾지 못했습니다/);
  });
});

// ── decimatePoints (그리드 해시 데시메이션) ─────────────────────────

describe('decimatePoints', () => {
  it('상한 이하 입력은 정확 중복만 제거하고 순서를 유지한다', () => {
    const input = Float32Array.from([0, 0, 0, 1, 1, 1, 0, 0, 0, 2, 2, 2]);
    const out = decimatePoints(input, MAX_HULL_POINTS);
    expect(toSortedTriples(out)).toEqual([
      [0, 0, 0],
      [1, 1, 1],
      [2, 2, 2],
    ]);
  });

  it('큰 입력도 항상 MAX_HULL_POINTS 이하로 줄이고, 대표점은 전부 입력 정점이다', () => {
    // 30×30×30 격자 = 27,000 정점 (> 2048)
    const n = 30;
    const input = new Float32Array(n * n * n * 3);
    let cursor = 0;
    const inputKeys = new Set<string>();
    for (let x = 0; x < n; x += 1) {
      for (let y = 0; y < n; y += 1) {
        for (let z = 0; z < n; z += 1) {
          const px = x / (n - 1);
          const py = y / (n - 1);
          const pz = z / (n - 1);
          input[cursor] = px;
          input[cursor + 1] = py;
          input[cursor + 2] = pz;
          cursor += 3;
          inputKeys.add(`${Math.fround(px)}|${Math.fround(py)}|${Math.fround(pz)}`);
        }
      }
    }
    const out = decimatePoints(input, MAX_HULL_POINTS);
    expect(out.length % 3).toBe(0);
    expect(out.length / 3).toBeGreaterThan(0);
    expect(out.length / 3).toBeLessThanOrEqual(MAX_HULL_POINTS);
    for (let i = 0; i < out.length; i += 3) {
      const key = `${out[i] ?? 0}|${out[i + 1] ?? 0}|${out[i + 2] ?? 0}`;
      expect(inputKeys.has(key)).toBe(true);
    }
  });

  it('셀을 공유하는 극점(모서리)이 비극점보다 뒤에 와도 대표로 살아남는다 (hull 과소근사 방지)', () => {
    // AABB [0,1]³, 시작 divisions(40)에서 [0.98…]과 [1,1,1]은 같은 셀에 든다.
    // 최초 등장 정점을 대표로 남기면 hull 극점인 모서리 [1,1,1]이 탈락한다 —
    // AABB 중심 최원점 선택이 극점을 보존해야 한다.
    const input = Float32Array.from([
      0, 0, 0,
      0.98, 0.98, 0.98,
      1, 1, 1,
      0.5, 0.5, 0.5,
      0.2, 0.2, 0.2,
    ]);
    const out = decimatePoints(input, 4);
    expect(out.length / 3).toBeLessThanOrEqual(4);
    const triples = toSortedTriples(out);
    expect(triples).toContainEqual([1, 1, 1]);
    // 같은 셀의 비극점([0.98…], f32 반올림 고려)은 대표가 아니다
    expect(triples.some((t) => Math.abs(t[0] - 0.98) < 1e-3)).toBe(false);
  });

  it('퇴화 입력(전부 같은 점)은 대표점 1개로 준다', () => {
    const count = MAX_HULL_POINTS + 100;
    const input = new Float32Array(count * 3);
    for (let i = 0; i < count; i += 1) {
      input[3 * i] = 0.5;
      input[3 * i + 1] = -1;
      input[3 * i + 2] = 2;
    }
    const out = decimatePoints(input, MAX_HULL_POINTS);
    expect(Array.from(out)).toEqual([0.5, -1, 2]);
  });

  it('빈 입력은 빈 배열', () => {
    expect(decimatePoints(new Float32Array(0), MAX_HULL_POINTS).length).toBe(0);
  });
});

// ── prepareForScene (bbox·스케일·피벗·축 변환) ──────────────────────

describe('prepareForScene', () => {
  it('스케일 + bbox 수학: halfExtents/centerOffset이 스케일 반영값이다', () => {
    // BoxGeometry(1,2,3)을 원점에서 멀리 옮겨 두어도 결과는 위치 무관해야 한다
    const mesh = meshOfGeometry(new THREE.BoxGeometry(1, 2, 3));
    mesh.position.set(10, 5, -2);
    const prepared = prepareForScene(importedModelOf(mesh), { scale: 2, upAxis: 'y' });

    expect(prepared.aabbHalfExtents[0]).toBeCloseTo(1, 6);
    expect(prepared.aabbHalfExtents[1]).toBeCloseTo(2, 6);
    expect(prepared.aabbHalfExtents[2]).toBeCloseTo(3, 6);
    // 피벗은 bbox 바닥 중심 → 중심 오프셋은 [0, halfY, 0]
    expect(prepared.aabbCenterOffset[0]).toBeCloseTo(0, 6);
    expect(prepared.aabbCenterOffset[1]).toBeCloseTo(2, 6);
    expect(prepared.aabbCenterOffset[2]).toBeCloseTo(0, 6);
  });

  it('피벗 재정렬: 반환 Object3D의 bbox가 바닥 중심 피벗(y=0, x/z 중앙)에 앉는다', () => {
    const mesh = meshOfGeometry(new THREE.BoxGeometry(1, 2, 3));
    mesh.position.set(10, 5, -2);
    const prepared = prepareForScene(importedModelOf(mesh), { scale: 2, upAxis: 'y' });

    prepared.object3D.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(prepared.object3D, true);
    expect(box.min.y).toBeCloseTo(0, 6);
    expect(box.max.y).toBeCloseTo(4, 6);
    expect((box.min.x + box.max.x) / 2).toBeCloseTo(0, 6);
    expect((box.min.z + box.max.z) / 2).toBeCloseTo(0, 6);
  });

  it('Z-up→Y-up 회전 수학: (x,y,z) → (x,z,-y) — URDF 경로와 동일 규약(-π/2 X)', () => {
    // 단일 삼각형: (0,0,0), (1,0,0), (0,0,2) — Z축으로 2만큼 뻗은 형상
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      'position',
      new THREE.Float32BufferAttribute([0, 0, 0, 1, 0, 0, 0, 0, 2], 3),
    );
    const prepared = prepareForScene(importedModelOf(meshOfGeometry(geometry)), {
      scale: 1,
      upAxis: 'z',
    });

    // 회전 후 Z 뻗음(2)이 Y 뻗음이 된다
    expect(prepared.aabbHalfExtents[0]).toBeCloseTo(0.5, 6);
    expect(prepared.aabbHalfExtents[1]).toBeCloseTo(1, 6);
    expect(prepared.aabbHalfExtents[2]).toBeCloseTo(0, 6);

    // 피벗 재정렬 포함 최종 정점: (-0.5,0,0), (0.5,0,0), (-0.5,2,0)
    const triples = toSortedTriples(prepared.points);
    expect(triples.length).toBe(3);
    const expected: Array<[number, number, number]> = [
      [-0.5, 0, 0],
      [-0.5, 2, 0],
      [0.5, 0, 0],
    ];
    triples.forEach((triple, i) => {
      const want = expected[i];
      expect(want).toBeDefined();
      triple.forEach((value, axis) => {
        expect(value).toBeCloseTo((want as [number, number, number])[axis] ?? 0, 5);
      });
    });
  });

  it('points는 항상 MAX_HULL_POINTS 이하로 데시메이션된다', () => {
    // SphereGeometry(1, 64, 64) — 정점 4,225개 (> 2048)
    const mesh = meshOfGeometry(new THREE.SphereGeometry(1, 64, 64));
    const prepared = prepareForScene(importedModelOf(mesh), { scale: 1, upAxis: 'y' });
    expect(prepared.points.length / 3).toBeLessThanOrEqual(MAX_HULL_POINTS);
    expect(prepared.points.length / 3).toBeGreaterThan(0);
  });

  it('유효하지 않은 스케일(0/음수/NaN)은 한국어 오류로 거부한다', () => {
    const model = importedModelOf(meshOfGeometry(new THREE.BoxGeometry(1, 1, 1)));
    expect(() => prepareForScene(model, { scale: 0, upAxis: 'y' })).toThrow(/스케일/);
    expect(() => prepareForScene(model, { scale: -1, upAxis: 'y' })).toThrow(/스케일/);
    expect(() => prepareForScene(model, { scale: Number.NaN, upAxis: 'y' })).toThrow(/스케일/);
  });
});

// ── extractTrimeshGeometry ──────────────────────────────────────────

describe('extractTrimeshGeometry', () => {
  it('인덱스 지오메트리를 병합하고 인덱스 무결성을 유지한다', () => {
    const prepared = prepareForScene(
      importedModelOf(meshOfGeometry(new THREE.BoxGeometry(1, 1, 1))),
      { scale: 1, upAxis: 'y' },
    );
    const tri = extractTrimeshGeometry(prepared.object3D);
    // BoxGeometry: 정점 24개 · 삼각형 12개(인덱스 36)
    expect(tri.positions.length).toBe(24 * 3);
    expect(tri.indices.length).toBe(36);
    expect(tri.indices.length % 3).toBe(0);
    const vertexCount = tri.positions.length / 3;
    for (const index of tri.indices) {
      expect(index).toBeLessThan(vertexCount);
    }
  });
});

// ── MeshAssetStore (MeshAssetResolver 계약) ─────────────────────────

describe('MeshAssetStore', () => {
  const bundleOf = (points: Float32Array): { object3D: THREE.Object3D; points: Float32Array } => ({
    object3D: new THREE.Group(),
    points,
  });

  it("register는 'asset://<n>' ref를 단조 증가로 발급한다", () => {
    const store = new MeshAssetStore();
    const a = store.register(bundleOf(Float32Array.from([0, 0, 0])));
    const b = store.register(bundleOf(Float32Array.from([1, 1, 1])));
    expect(a).toBe(`${ASSET_REF_PREFIX}1`);
    expect(b).toBe(`${ASSET_REF_PREFIX}2`);
    expect(store.refs()).toEqual([a, b]);
  });

  it('getObject/getPoints 왕복 — trimesh 없는 에셋의 getIndices는 undefined(hull 전용)', () => {
    const store = new MeshAssetStore();
    const object3D = new THREE.Group();
    const points = Float32Array.from([0, 0, 0, 1, 0, 0]);
    const ref = store.register({ object3D, points });
    expect(store.getObject(ref)).toBe(object3D);
    expect(store.getPoints(ref)).toBe(points);
    expect(store.getIndices(ref)).toBeUndefined();
  });

  it('trimesh 에셋: getPoints는 인덱스와 쌍을 이루는 trimesh 정점을 준다', () => {
    const store = new MeshAssetStore();
    const hullPoints = Float32Array.from([0, 0, 0]);
    const trimeshPositions = Float32Array.from([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const indices = Uint32Array.from([0, 1, 2]);
    const ref = store.register({
      object3D: new THREE.Group(),
      points: hullPoints,
      trimesh: { positions: trimeshPositions, indices },
    });
    expect(store.getPoints(ref)).toBe(trimeshPositions);
    expect(store.getIndices(ref)).toBe(indices);
  });

  it('미등록 ref는 전부 undefined', () => {
    const store = new MeshAssetStore();
    expect(store.getObject('asset://999')).toBeUndefined();
    expect(store.getPoints('asset://999')).toBeUndefined();
    expect(store.getIndices('asset://999')).toBeUndefined();
  });
});

// ── collectAssetRefs (씬 저장 경고 대상) ────────────────────────────

describe('collectAssetRefs', () => {
  it('visual.ref + collider ref에서 asset:// 참조를 중복 없이 수집한다', () => {
    const spec = sceneWith([
      {
        id: 'imported',
        type: 'object',
        transform: { position: [0, 0, 0] },
        visual: { kind: 'mesh', ref: 'asset://1' },
        physics: {
          bodyType: 'dynamic',
          colliders: [
            {
              shape: { kind: 'convexHull', ref: 'asset://1' },
              group: 'OBJECT',
              collidesWith: ['ENV', 'ROBOT', 'OBJECT'],
              emitEvents: true,
            },
          ],
        },
      },
      {
        id: 'box',
        type: 'object',
        transform: { position: [0, 0, 0] },
        visual: {
          kind: 'primitive',
          primitive: { kind: 'box', halfExtents: [0.05, 0.05, 0.05] },
        },
      },
    ]);
    expect(collectAssetRefs(spec)).toEqual(['asset://1']);
  });

  it('asset 참조가 없는 씬은 빈 배열 (파일 경로 ref는 수집하지 않는다)', () => {
    const spec = sceneWith([
      {
        id: 'deco',
        type: 'static',
        transform: { position: [0, 0, 0] },
        visual: { kind: 'mesh', ref: 'assets/models/rock.glb' },
      },
    ]);
    expect(collectAssetRefs(spec)).toEqual([]);
  });
});

// ── import-dialog 순수 함수 (폼 상태 → EntitySpec 매핑) ─────────────

describe('import-dialog 순수 함수', () => {
  const derived = {
    assetRef: 'asset://7',
    aabbHalfExtents: [0.5, 1, 0.25] as Vec3,
    aabbCenterOffset: [0, 1, 0] as Vec3,
  };
  const baseState: ImportFormState = {
    entityId: 'part',
    scale: 1,
    upAxis: 'y',
    collider: 'convexHull',
    entityKind: 'object',
  };

  it('forcedEntityKind: trimesh만 Static으로 강제한다', () => {
    expect(forcedEntityKind('trimesh', 'object')).toBe('static');
    expect(forcedEntityKind('trimesh', 'static')).toBe('static');
    expect(forcedEntityKind('convexHull', 'object')).toBe('object');
    expect(forcedEntityKind('aabb', 'object')).toBe('object');
  });

  it('convexHull + Object → 동적 OBJECT 엔티티 (감지 규약 포함)', () => {
    const entity = buildImportedEntitySpec(baseState, derived);
    expect(entity.id).toBe('part');
    expect(entity.type).toBe('object');
    expect(entity.visual).toEqual({ kind: 'mesh', ref: 'asset://7' });
    expect(entity.physics?.bodyType).toBe('dynamic');
    const collider = entity.physics?.colliders[0];
    expect(collider?.shape).toEqual({ kind: 'convexHull', ref: 'asset://7' });
    expect(collider?.offset).toBeUndefined();
    expect(collider?.group).toBe('OBJECT');
    // SENSOR_ZONE 포함 — 없으면 양방향 쌍 필터 때문에 Sensor Zone이 임포트 사물을
    // 감지하지 못한다 (templates.ts OBJECT_COLLIDES_WITH와 동일 배정)
    expect(collider?.collidesWith).toEqual(['ENV', 'ROBOT', 'OBJECT', 'SENSOR_ZONE']);
    expect(collider?.emitEvents).toBe(true);
  });

  it('동적 임포트 엔티티의 group/collidesWith는 라이브러리 동적 템플릿과 동일하다', () => {
    const imported = buildImportedEntitySpec(baseState, derived).physics?.colliders[0];
    const template = templateByKey('box')?.create((base) => `${base}_1`);
    const templateCollider = template?.physics?.colliders[0];
    expect(templateCollider).toBeDefined();
    expect(imported?.group).toBe(templateCollider?.group);
    expect(imported?.collidesWith).toEqual(templateCollider?.collidesWith);
    expect(imported?.emitEvents).toBe(templateCollider?.emitEvents);
  });

  it('AABB → box collider (halfExtents + 피벗 기준 중심 offset)', () => {
    const entity = buildImportedEntitySpec({ ...baseState, collider: 'aabb' }, derived);
    const collider = entity.physics?.colliders[0];
    expect(collider?.shape).toEqual({ kind: 'box', halfExtents: [0.5, 1, 0.25] });
    expect(collider?.offset).toEqual({ position: [0, 1, 0] });
  });

  it('trimesh + Object 요청 → Static/fixed/ENV로 강제된다 (동적 trimesh 금지)', () => {
    const entity = buildImportedEntitySpec(
      { ...baseState, collider: 'trimesh', entityKind: 'object' },
      derived,
    );
    expect(entity.type).toBe('static');
    expect(entity.physics?.bodyType).toBe('fixed');
    const collider = entity.physics?.colliders[0];
    expect(collider?.shape).toEqual({ kind: 'trimesh', ref: 'asset://7' });
    expect(collider?.group).toBe('ENV');
  });

  it('생성된 EntitySpec은 세 전략 모두 SceneSpec 스키마 검증을 통과한다', () => {
    const strategies: ReadonlyArray<ImportFormState['collider']> = [
      'convexHull',
      'aabb',
      'trimesh',
    ];
    for (const collider of strategies) {
      const entity = buildImportedEntitySpec({ ...baseState, collider }, derived);
      const result = validateScene(sceneWith([entity]));
      expect(result.ok, `${collider} 전략 검증 실패: ${result.ok ? '' : result.errors.join('; ')}`).toBe(
        true,
      );
    }
  });

  it('defaultEntityIdFromFileName: 소문자·안전 문자만, 전부 걸러지면 fallback', () => {
    expect(defaultEntityIdFromFileName('My Model V2.glb')).toBe('my-model-v2');
    expect(defaultEntityIdFromFileName('part.obj')).toBe('part');
    expect(defaultEntityIdFromFileName('한글모델.stl')).toBe('imported-mesh');
  });

  it('formatBboxSizeM: 스케일 반영 W × H × L 표시', () => {
    expect(formatBboxSizeM({ min: [0, 0, 0], max: [1, 0.5, 0.25] }, 2)).toBe(
      '2.000 × 1.000 × 0.500 m',
    );
  });
});
