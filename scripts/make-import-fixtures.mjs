// scripts/make-import-fixtures.mjs — mesh-import 게이트의 **계측용** 픽스처 생성기
//
// 실행: node scripts/make-import-fixtures.mjs
// 산출: public/assets/models/gate-box.{glb,stl,obj} (커밋 대상 — CI에서 돌 필요 없다)
//
// ── 왜 손으로 만든 픽스처인가 ────────────────────────────────────────
// public/assets/models/의 다운로드 모델(avocado, teacup 등)은 **카탈로그 스모크**용이다.
// 수치 어서션을 거기에 걸면 에셋을 갱신하는 순간 게이트가 통째로 무너지고, 무엇보다
// 그 모델들의 치수는 우리가 통제하지 않는다.
//
// ── 형상 제약 (이걸 어기면 게이트가 거짓 초록이 된다) ────────────────
// 1. **세 변의 길이가 모두 다르다.** 정육면체면 Z-up→Y-up 회전(y↔z 교환)이 관측되지
//    않아, 임포트가 upAxis를 통째로 무시해도 어서션이 통과한다.
// 2. **bbox가 원점 중심이 아니다.** 원점 중심이면 prepareForScene의 피벗 재정렬에서
//    x/z 성분이 항등이라 검증되지 않는다.
// 3. **3종이 같은 솔리드다.** 삼각형 수·bbox가 같아야 포맷 간 교차 검증이 성립한다.

import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const OUT_DIR = resolve(dirname(fileURLToPath(import.meta.url)), '../public/assets/models');
const BASE_NAME = 'gate-box';

/** 원본(Y-up) bbox. 변 0.30(x) × 0.20(y) × 0.10(z), 원점에서 어긋나 있다. */
const MIN = [0.1, 0.05, -0.3];
const MAX = [0.4, 0.25, -0.2];

const [minX, minY, minZ] = MIN;
const [maxX, maxY, maxZ] = MAX;

/** 8 정점 — 아래면(y=minY) 0..3, 윗면(y=maxY) 4..7 */
const VERTICES = [
  [minX, minY, minZ], [maxX, minY, minZ], [maxX, minY, maxZ], [minX, minY, maxZ],
  [minX, maxY, minZ], [maxX, maxY, minZ], [maxX, maxY, maxZ], [minX, maxY, maxZ],
];

/** 12 삼각형 (바깥쪽 CCW) */
const TRIANGLES = [
  [0, 2, 1], [0, 3, 2],   // 바닥 (-y)
  [4, 5, 6], [4, 6, 7],   // 천장 (+y)
  [0, 1, 5], [0, 5, 4],   // -z
  [2, 3, 7], [2, 7, 6],   // +z
  [3, 0, 4], [3, 4, 7],   // -x
  [1, 2, 6], [1, 6, 5],   // +x
];

function faceNormal([a, b, c]) {
  const p = VERTICES[a], q = VERTICES[b], r = VERTICES[c];
  const u = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
  const v = [r[0] - p[0], r[1] - p[1], r[2] - p[2]];
  const n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
  const len = Math.hypot(...n) || 1;
  return n.map((x) => x / len);
}

// ── OBJ (텍스트, 1-base 인덱스) ─────────────────────────────────────

function makeObj() {
  const lines = [
    `# ${BASE_NAME} — mesh-import 게이트 계측 픽스처 (생성: scripts/make-import-fixtures.mjs)`,
    `o ${BASE_NAME}`,
    ...VERTICES.map(([x, y, z]) => `v ${x} ${y} ${z}`),
    ...TRIANGLES.map(([a, b, c]) => `f ${a + 1} ${b + 1} ${c + 1}`),
  ];
  return `${lines.join('\n')}\n`;
}

// ── STL (ASCII — STLLoader가 매직으로 판별하므로 diff 읽히는 쪽을 쓴다) ──

function makeStl() {
  const out = [`solid ${BASE_NAME}`];
  for (const tri of TRIANGLES) {
    const [nx, ny, nz] = faceNormal(tri);
    out.push(`  facet normal ${nx} ${ny} ${nz}`, '    outer loop');
    for (const idx of tri) {
      const [x, y, z] = VERTICES[idx];
      out.push(`      vertex ${x} ${y} ${z}`);
    }
    out.push('    endloop', '  endfacet');
  }
  out.push(`endsolid ${BASE_NAME}`);
  return `${out.join('\n')}\n`;
}

// ── GLB (glTF 2.0 바이너리, 손으로 조립) ────────────────────────────

const GLB_MAGIC = 0x46546c67;      // 'glTF'
const GLB_VERSION = 2;
const CHUNK_JSON = 0x4e4f534a;     // 'JSON'
const CHUNK_BIN = 0x004e4942;      // 'BIN\0'
const COMPONENT_UNSIGNED_SHORT = 5123;
const COMPONENT_FLOAT = 5126;
const TARGET_ELEMENT_ARRAY = 34963;
const TARGET_ARRAY = 34962;

function pad4(n) {
  return (4 - (n % 4)) % 4;
}

function makeGlb() {
  const indices = TRIANGLES.flat();
  const indexBytes = indices.length * 2;          // uint16
  const positionOffset = indexBytes + pad4(indexBytes);
  const positionBytes = VERTICES.length * 3 * 4;  // float32 vec3

  const bin = Buffer.alloc(positionOffset + positionBytes);
  indices.forEach((v, i) => bin.writeUInt16LE(v, i * 2));
  VERTICES.flat().forEach((v, i) => bin.writeFloatLE(v, positionOffset + i * 4));

  const json = {
    asset: { version: '2.0', generator: 'workcell make-import-fixtures' },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0, name: BASE_NAME }],
    meshes: [{ name: BASE_NAME, primitives: [{ attributes: { POSITION: 1 }, indices: 0 }] }],
    accessors: [
      { bufferView: 0, componentType: COMPONENT_UNSIGNED_SHORT, count: indices.length, type: 'SCALAR' },
      // POSITION accessor는 min/max가 필수다 (glTF 2.0 스펙)
      { bufferView: 1, componentType: COMPONENT_FLOAT, count: VERTICES.length, type: 'VEC3', min: MIN, max: MAX },
    ],
    bufferViews: [
      { buffer: 0, byteOffset: 0, byteLength: indexBytes, target: TARGET_ELEMENT_ARRAY },
      { buffer: 0, byteOffset: positionOffset, byteLength: positionBytes, target: TARGET_ARRAY },
    ],
    buffers: [{ byteLength: bin.length }],
  };

  const jsonRaw = Buffer.from(JSON.stringify(json), 'utf8');
  const jsonChunk = Buffer.concat([jsonRaw, Buffer.alloc(pad4(jsonRaw.length), 0x20)]); // 공백 패딩
  const binChunk = Buffer.concat([bin, Buffer.alloc(pad4(bin.length), 0x00)]);          // 0 패딩

  const header = Buffer.alloc(12);
  header.writeUInt32LE(GLB_MAGIC, 0);
  header.writeUInt32LE(GLB_VERSION, 4);
  header.writeUInt32LE(12 + 8 + jsonChunk.length + 8 + binChunk.length, 8);

  const jsonHeader = Buffer.alloc(8);
  jsonHeader.writeUInt32LE(jsonChunk.length, 0);
  jsonHeader.writeUInt32LE(CHUNK_JSON, 4);

  const binHeader = Buffer.alloc(8);
  binHeader.writeUInt32LE(binChunk.length, 0);
  binHeader.writeUInt32LE(CHUNK_BIN, 4);

  return Buffer.concat([header, jsonHeader, jsonChunk, binHeader, binChunk]);
}

// ── 쓰기 ────────────────────────────────────────────────────────────

mkdirSync(OUT_DIR, { recursive: true });
const written = [
  [`${BASE_NAME}.obj`, makeObj()],
  [`${BASE_NAME}.stl`, makeStl()],
  [`${BASE_NAME}.glb`, makeGlb()],
];
for (const [name, data] of written) {
  writeFileSync(resolve(OUT_DIR, name), data);
  console.log(`${name}: ${typeof data === 'string' ? Buffer.byteLength(data) : data.length} bytes`);
}
console.log(`삼각형 ${TRIANGLES.length} · 정점 ${VERTICES.length} · bbox ${JSON.stringify(MIN)}~${JSON.stringify(MAX)}`);
