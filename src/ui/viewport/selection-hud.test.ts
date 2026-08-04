// ui/viewport/selection-hud.test.ts — 선택 HUD 순수 헬퍼 계약
//
// HUD는 "한 칸씩" 조정 표면이다. 이 테스트가 고정하는 것:
// (1) 형상 종류마다 노출되는 치수 축이 맞고, (2) ± 한 칸이 **전체 치수** 기준으로
// 정확히 움직이며(내부 halfExtents 표현이 새지 않는다), (3) 하한 아래로는 못 내려가고,
// (4) 편집할 수 없는 대상(로봇·임포트 메시·다중 collider)에는 스테퍼를 그리지 않는다.

import { describe, expect, it } from 'vitest';
import {
  dimensionFieldsOf,
  editableShapeOf,
  entityKindLabelKo,
  formatPosition,
  HUD_DIMENSION_MIN_M,
  HUD_DIMENSION_STEP_M,
  stepShapeDimension,
} from './selection-hud';
import type { ColliderShape, EntitySpec } from '../../schema/types';

// ── 테스트 상수 (매직넘버 금지 — CLAUDE.md §4) ──────────────────────

const BOX: ColliderShape = { kind: 'box', halfExtents: [0.05, 0.03, 0.1] };
const SPHERE: ColliderShape = { kind: 'sphere', radius: 0.07 };
const CYLINDER: ColliderShape = { kind: 'cylinder', radius: 0.04, halfHeight: 0.09 };
const PRECISION = 9;

function primitiveEntity(shape: ColliderShape, colliderCount = 1): EntitySpec {
  return {
    id: 'obj',
    type: 'object',
    transform: { position: [0, 0, 0] },
    visual: { kind: 'primitive', primitive: shape },
    physics: {
      bodyType: 'dynamic',
      colliders: Array.from({ length: colliderCount }, () => ({
        shape,
        group: 'OBJECT' as const,
        collidesWith: ['ENV' as const],
      })),
    },
  };
}

// ── 1. 치수 필드 ────────────────────────────────────────────────────

describe('dimensionFieldsOf', () => {
  it('box: 너비·높이·길이를 **전체 치수**로 노출한다 (halfExtents × 2)', () => {
    const fields = dimensionFieldsOf(BOX);
    expect(fields.map((f) => f.axis)).toEqual(['width', 'height', 'length']);
    expect(fields.map((f) => f.valueM)).toEqual([0.1, 0.06, 0.2]);
  });

  it('sphere: 반지름 한 줄 (반지름은 전체 치수 변환 없이 그대로)', () => {
    expect(dimensionFieldsOf(SPHERE)).toEqual([
      { axis: 'radius', labelKo: '반지름', valueM: SPHERE.radius },
    ]);
  });

  it('cylinder/capsule: 반지름 + 전체 높이', () => {
    const fields = dimensionFieldsOf(CYLINDER);
    expect(fields.map((f) => f.axis)).toEqual(['radius', 'height']);
    expect(fields[1]?.valueM).toBeCloseTo(0.18, PRECISION);
  });

  it('편집 불가 형상은 빈 배열 — 스테퍼를 그리지 않는다', () => {
    expect(dimensionFieldsOf({ kind: 'convexHull', ref: 'asset://1' })).toEqual([]);
    expect(dimensionFieldsOf({ kind: 'trimesh', ref: 'asset://1' })).toEqual([]);
    expect(dimensionFieldsOf({ kind: 'fromVisual' })).toEqual([]);
  });
});

// ── 2. ± 한 칸 ──────────────────────────────────────────────────────

describe('stepShapeDimension', () => {
  it('box 높이 +1칸: 전체 치수가 step만큼 커진다 (halfExtents는 절반만)', () => {
    const next = stepShapeDimension(BOX, 'height', HUD_DIMENSION_STEP_M);
    expect(next.kind).toBe('box');
    if (next.kind !== 'box') return;
    expect(next.halfExtents[1]).toBeCloseTo((0.06 + HUD_DIMENSION_STEP_M) / 2, PRECISION);
    // 건드리지 않은 축은 그대로
    expect(next.halfExtents[0]).toBeCloseTo(BOX.kind === 'box' ? 0.05 : 0, PRECISION);
  });

  it('box 너비 −1칸', () => {
    const next = stepShapeDimension(BOX, 'width', -HUD_DIMENSION_STEP_M);
    if (next.kind !== 'box') throw new Error('box여야 한다');
    expect(next.halfExtents[0]).toBeCloseTo((0.1 - HUD_DIMENSION_STEP_M) / 2, PRECISION);
  });

  it('하한 아래로는 내려가지 않는다 (0 이하·음수 치수 금지)', () => {
    const tiny: ColliderShape = { kind: 'box', halfExtents: [0.005, 0.005, 0.005] };
    const next = stepShapeDimension(tiny, 'width', -HUD_DIMENSION_STEP_M);
    if (next.kind !== 'box') throw new Error('box여야 한다');
    expect(next.halfExtents[0]).toBeCloseTo(HUD_DIMENSION_MIN_M / 2, PRECISION);
  });

  it('sphere 반지름 ±', () => {
    const next = stepShapeDimension(SPHERE, 'radius', HUD_DIMENSION_STEP_M);
    if (next.kind !== 'sphere') throw new Error('sphere여야 한다');
    expect(next.radius).toBeCloseTo(SPHERE.radius + HUD_DIMENSION_STEP_M, PRECISION);
  });

  it('cylinder 높이는 전체 높이 기준으로 움직인다', () => {
    const next = stepShapeDimension(CYLINDER, 'height', HUD_DIMENSION_STEP_M);
    if (next.kind !== 'cylinder') throw new Error('cylinder여야 한다');
    expect(next.halfHeight).toBeCloseTo((0.18 + HUD_DIMENSION_STEP_M) / 2, PRECISION);
    expect(next.radius).toBeCloseTo(CYLINDER.radius, PRECISION);
  });

  it('형상이 갖지 않는 축은 무해한 no-op (원 형상 그대로)', () => {
    expect(stepShapeDimension(SPHERE, 'width', HUD_DIMENSION_STEP_M)).toEqual(SPHERE);
    expect(stepShapeDimension(BOX, 'radius', HUD_DIMENSION_STEP_M)).toEqual(BOX);
  });
});

// ── 3. 편집 대상 판정 ───────────────────────────────────────────────

describe('editableShapeOf', () => {
  it('단일 collider 프리미티브는 편집 대상', () => {
    expect(editableShapeOf(primitiveEntity(BOX))).toEqual(BOX);
  });

  it('로봇은 대상이 아니다 (물리가 URDF에서 유도된다)', () => {
    const robot: EntitySpec = {
      id: 'arm',
      type: 'robot',
      transform: { position: [0, 0, 0] },
      visual: { kind: 'urdf', ref: 'assets/robots/arm6/arm6.urdf' },
    };
    expect(editableShapeOf(robot)).toBeNull();
  });

  it('임포트 메시(visual.kind mesh)는 대상이 아니다', () => {
    const imported: EntitySpec = {
      id: 'mesh',
      type: 'object',
      transform: { position: [0, 0, 0] },
      visual: { kind: 'mesh', ref: 'asset://1' },
    };
    expect(editableShapeOf(imported)).toBeNull();
  });

  it('collider 2개 이상은 대상이 아니다 — scene-editor가 거부할 버튼을 그리지 않는다', () => {
    expect(editableShapeOf(primitiveEntity(BOX, 2))).toBeNull();
  });
});

// ── 4. 표시 포맷 ────────────────────────────────────────────────────

describe('표시 포맷', () => {
  it('위치 리드아웃은 축 라벨과 소수 3자리', () => {
    expect(formatPosition([0.35, 0.025, -0.15])).toBe('x 0.350 · y 0.025 · z -0.150');
  });

  it('유형 배지는 한국어 (UI 크롬은 한국어 — CLAUDE.md §4-b)', () => {
    expect(entityKindLabelKo(primitiveEntity(BOX))).toBe('사물');
    expect(entityKindLabelKo({ ...primitiveEntity(BOX), type: 'static' })).toBe('정적');
    expect(entityKindLabelKo({ ...primitiveEntity(BOX), type: 'robot' })).toBe('로봇');
  });
});
