// ui/inspector/entity-editor.test.ts — 편집 폼 순수 헬퍼 단위 테스트 (DOM 비의존, node)
//
// mountEntityEditor의 DOM 조립은 얇은 글루라 여기서 다루지 않는다(브라우저 게이트 몫).
// 여기서는 오일러(deg)→쿼터니언 변환(inspector.ts quatToEulerDegXYZ와의 역변환 쌍),
// 클램프/파싱/스크럽 계산, 치수·Physics 폼 상태 매핑(전체 치수 ↔ halfExtents,
// 미지 필드 보존)의 계약을 검증한다.

import { describe, expect, it } from 'vitest';
import {
  DIMENSION_MIN_M,
  PHYSICS_PARAM_MAX,
  PHYSICS_PARAM_MIN,
  applyPhysicsFormState,
  bodyTypeChoices,
  bodyTypeLabel,
  clamp,
  clampDimensionM,
  clampPhysicsParam,
  collidesWithChoices,
  eulerDegToQuat,
  formStateToShape,
  groupChoices,
  groupLabel,
  localNameError,
  parseNumeric,
  physicsFormStateOf,
  primitiveFormStateOf,
  primitiveShapeOf,
  scrubValue,
  shapeToFormState,
} from './entity-editor';
import { quatToEulerDegXYZ } from './inspector';
import type { ColliderShape, EntitySpec, PhysicsSpec, Quat } from '../../schema/types';

// ── 테스트 전용 쿼터니언 헬퍼 (Hamilton 곱, [x,y,z,w] — inspector.test와 동일 규약) ─

function axisQuat(axis: 'x' | 'y' | 'z', deg: number): Quat {
  const halfRad = (deg * Math.PI) / 180 / 2;
  const s = Math.sin(halfRad);
  const c = Math.cos(halfRad);
  if (axis === 'x') return [s, 0, 0, c];
  if (axis === 'y') return [0, s, 0, c];
  return [0, 0, s, c];
}

function mulQuat(a: Quat, b: Quat): Quat {
  const [ax, ay, az, aw] = a;
  const [bx, by, bz, bw] = b;
  return [
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  ];
}

function expectQuatClose(actual: Quat, expected: Quat, digits = 9): void {
  for (let i = 0; i < 4; i += 1) {
    expect(actual[i]).toBeCloseTo(expected[i] ?? 0, digits);
  }
}

// ── eulerDegToQuat — 입력 경계 전용 변환 (내부 진실은 쿼터니언, CLAUDE.md §4) ─

describe('eulerDegToQuat', () => {
  it('[0,0,0] → identity 쿼터니언', () => {
    expectQuatClose(eulerDegToQuat([0, 0, 0]), [0, 0, 0, 1]);
  });

  it('단일 축 90° 회전을 축별 쿼터니언으로 변환한다', () => {
    expectQuatClose(eulerDegToQuat([90, 0, 0]), axisQuat('x', 90));
    expectQuatClose(eulerDegToQuat([0, 90, 0]), axisQuat('y', 90));
    expectQuatClose(eulerDegToQuat([0, 0, 90]), axisQuat('z', 90));
  });

  it('합성 회전이 XYZ 순서 Hamilton 곱(qx ⊗ qy ⊗ qz)과 일치한다', () => {
    const expected = mulQuat(mulQuat(axisQuat('x', 30), axisQuat('y', 40)), axisQuat('z', 20));
    expectQuatClose(eulerDegToQuat([30, 40, 20]), expected);
  });

  it('quatToEulerDegXYZ와 라운드트립한다 (역변환 쌍 계약)', () => {
    const [x, y, z] = quatToEulerDegXYZ(eulerDegToQuat([30, 40, 20]));
    expect(x).toBeCloseTo(30, 6);
    expect(y).toBeCloseTo(40, 6);
    expect(z).toBeCloseTo(20, 6);
  });

  it('음수/큰 각도에서도 단위 노름을 유지한다', () => {
    const [x, y, z, w] = eulerDegToQuat([-170, 85, 200]);
    expect(Math.hypot(x, y, z, w)).toBeCloseTo(1, 9);
  });
});

// ── 클램프 / 파싱 / 스크럽 계산 ──────────────────────────────────────

describe('clamp / clampDimensionM / clampPhysicsParam', () => {
  it('clamp: [min, max] 범위로 자른다', () => {
    expect(clamp(5, 0, 2)).toBe(2);
    expect(clamp(-1, 0, 2)).toBe(0);
    expect(clamp(1.5, 0, 2)).toBe(1.5);
  });

  it(`치수는 최소 ${DIMENSION_MIN_M}m로 클램프한다 (요구 §2 Dimensions)`, () => {
    expect(clampDimensionM(0)).toBe(DIMENSION_MIN_M);
    expect(clampDimensionM(-1)).toBe(DIMENSION_MIN_M);
    expect(clampDimensionM(0.1)).toBe(0.1);
  });

  it('치수: 비유한 입력(NaN/∞)은 최소값으로 방어한다', () => {
    expect(clampDimensionM(Number.NaN)).toBe(DIMENSION_MIN_M);
    expect(clampDimensionM(Number.POSITIVE_INFINITY)).toBe(DIMENSION_MIN_M);
  });

  it(`physics 계수는 ${PHYSICS_PARAM_MIN}..${PHYSICS_PARAM_MAX}로 클램프한다`, () => {
    expect(clampPhysicsParam(-0.5)).toBe(PHYSICS_PARAM_MIN);
    expect(clampPhysicsParam(3)).toBe(PHYSICS_PARAM_MAX);
    expect(clampPhysicsParam(1.2)).toBe(1.2);
    expect(clampPhysicsParam(Number.NaN)).toBe(PHYSICS_PARAM_MIN);
  });
});

describe('parseNumeric — 입력 문자열 파싱', () => {
  it('유한 수 문자열을 파싱한다 (공백 허용, 지수 표기 포함)', () => {
    expect(parseNumeric('1.5')).toBe(1.5);
    expect(parseNumeric('  -0.25 ')).toBe(-0.25);
    expect(parseNumeric('1e-3')).toBe(0.001);
    expect(parseNumeric('0')).toBe(0);
  });

  it('빈 문자열/비수치/비유한은 null (호출자가 폴백 결정)', () => {
    expect(parseNumeric('')).toBeNull();
    expect(parseNumeric('   ')).toBeNull();
    expect(parseNumeric('abc')).toBeNull();
    expect(parseNumeric('Infinity')).toBeNull();
  });
});

describe('scrubValue — 라벨 드래그 스크럽 계산', () => {
  it('시작값 + 이동량 × 증분', () => {
    expect(scrubValue(1, 10, 0.01)).toBeCloseTo(1.1, 9);
    expect(scrubValue(1, -10, 0.01)).toBeCloseTo(0.9, 9);
    expect(scrubValue(0.4, 0, 0.01)).toBe(0.4);
  });

  it('min/max가 주어지면 클램프한다', () => {
    expect(scrubValue(0.01, -100, 0.01, DIMENSION_MIN_M)).toBe(DIMENSION_MIN_M);
    expect(scrubValue(1.9, 100, 0.01, undefined, 2)).toBe(2);
  });
});

describe('localNameError — 이름 커밋 로컬 검증', () => {
  it('빈/공백 이름은 한국어 오류를 돌려준다', () => {
    expect(localNameError('')).toBe('이름을 입력하세요');
    expect(localNameError('   ')).toBe('이름을 입력하세요');
  });

  it('내용 있는 이름은 통과(null) — 중복 등은 deps.renameEntity가 판정', () => {
    expect(localNameError('box_a')).toBeNull();
  });
});

// ── 치수 폼 상태 매핑 (전체 치수 UI ↔ halfExtents 내부 — UX §3.5) ────

describe('shapeToFormState — 내부 half 값 → 전체 치수', () => {
  it('box: halfExtents × 2 → W/H/L', () => {
    expect(shapeToFormState({ kind: 'box', halfExtents: [0.05, 0.1, 0.15] })).toEqual({
      kind: 'box',
      widthM: 0.1,
      heightM: 0.2,
      lengthM: 0.3,
    });
  });

  it('sphere: 반지름 그대로', () => {
    expect(shapeToFormState({ kind: 'sphere', radius: 0.07 })).toEqual({
      kind: 'sphere',
      radiusM: 0.07,
    });
  });

  it('cylinder/capsule: halfHeight × 2 → 전체 높이', () => {
    expect(shapeToFormState({ kind: 'cylinder', halfHeight: 0.2, radius: 0.05 })).toEqual({
      kind: 'cylinder',
      radiusM: 0.05,
      heightM: 0.4,
    });
    expect(shapeToFormState({ kind: 'capsule', halfHeight: 0.15, radius: 0.04 })).toEqual({
      kind: 'capsule',
      radiusM: 0.04,
      heightM: 0.3,
    });
  });

  it('프리미티브 4종 외(convexHull/trimesh/fromVisual)는 null', () => {
    expect(shapeToFormState({ kind: 'convexHull', ref: 'asset://x' })).toBeNull();
    expect(shapeToFormState({ kind: 'trimesh', ref: 'asset://x' })).toBeNull();
    expect(shapeToFormState({ kind: 'fromVisual' })).toBeNull();
  });
});

describe('formStateToShape — 전체 치수 → 내부 half 값 (+ 최소 클램프)', () => {
  it('box: W/H/L ÷ 2 → halfExtents', () => {
    expect(
      formStateToShape({ kind: 'box', widthM: 0.1, heightM: 0.2, lengthM: 0.3 }),
    ).toEqual({ kind: 'box', halfExtents: [0.05, 0.1, 0.15] });
  });

  it(`최소 치수 ${DIMENSION_MIN_M}m 클램프 후 변환한다 (0/음수 입력 방어)`, () => {
    const shape = formStateToShape({ kind: 'box', widthM: 0, heightM: -1, lengthM: 0.3 });
    expect(shape).toEqual({
      kind: 'box',
      halfExtents: [DIMENSION_MIN_M / 2, DIMENSION_MIN_M / 2, 0.15],
    });
    expect(formStateToShape({ kind: 'sphere', radiusM: 0.001 })).toEqual({
      kind: 'sphere',
      radius: DIMENSION_MIN_M,
    });
  });

  it('cylinder/capsule: 전체 높이 ÷ 2 → halfHeight', () => {
    expect(formStateToShape({ kind: 'cylinder', radiusM: 0.05, heightM: 0.4 })).toEqual({
      kind: 'cylinder',
      radius: 0.05,
      halfHeight: 0.2,
    });
    expect(formStateToShape({ kind: 'capsule', radiusM: 0.04, heightM: 0.3 })).toEqual({
      kind: 'capsule',
      radius: 0.04,
      halfHeight: 0.15,
    });
  });

  it('유효 치수는 shape ↔ form 라운드트립이 무손실이다', () => {
    const original: ColliderShape = { kind: 'box', halfExtents: [0.05, 0.1, 0.15] };
    const form = shapeToFormState(original);
    expect(form).not.toBeNull();
    if (form !== null) expect(formStateToShape(form)).toEqual(original);
  });
});

// ── 엔티티에서 편집 대상 shape 추출 ─────────────────────────────────

const BOX_PHYSICS: PhysicsSpec = {
  bodyType: 'dynamic',
  colliders: [
    {
      shape: { kind: 'box', halfExtents: [0.05, 0.05, 0.05] },
      friction: 0.6,
      group: 'OBJECT',
      collidesWith: ['ENV', 'ROBOT', 'OBJECT'],
      emitEvents: true,
    },
  ],
};

const BOX_ENTITY: EntitySpec = {
  id: 'box_a',
  type: 'object',
  transform: { position: [0.4, 0.05, 0] },
  visual: {
    kind: 'primitive',
    primitive: { kind: 'box', halfExtents: [0.06, 0.06, 0.06] },
    color: '#c0392b',
  },
  physics: BOX_PHYSICS,
};

describe('primitiveShapeOf / primitiveFormStateOf', () => {
  it('collider shape(물리가 진실)를 visual보다 우선한다', () => {
    expect(primitiveShapeOf(BOX_ENTITY)).toEqual({
      kind: 'box',
      halfExtents: [0.05, 0.05, 0.05],
    });
    expect(primitiveFormStateOf(BOX_ENTITY)).toEqual({
      kind: 'box',
      widthM: 0.1,
      heightM: 0.1,
      lengthM: 0.1,
    });
  });

  it('collider가 프리미티브가 아니면(fromVisual 등) visual.primitive로 폴백한다', () => {
    const entity: EntitySpec = {
      ...BOX_ENTITY,
      physics: {
        bodyType: 'dynamic',
        colliders: [
          { shape: { kind: 'fromVisual' }, group: 'OBJECT', collidesWith: ['ENV'] },
        ],
      },
    };
    expect(primitiveShapeOf(entity)).toEqual({ kind: 'box', halfExtents: [0.06, 0.06, 0.06] });
  });

  it('physics가 없으면 visual.primitive를 쓴다', () => {
    const { physics: _physics, ...rest } = BOX_ENTITY;
    expect(primitiveShapeOf(rest)).toEqual({ kind: 'box', halfExtents: [0.06, 0.06, 0.06] });
  });

  it('프리미티브가 전혀 없으면(mesh 엔티티 등) null', () => {
    const entity: EntitySpec = {
      id: 'mesh_a',
      type: 'object',
      transform: { position: [0, 0, 0] },
      visual: { kind: 'mesh', ref: 'assets/part.glb' },
    };
    expect(primitiveShapeOf(entity)).toBeNull();
    expect(primitiveFormStateOf(entity)).toBeNull();
  });
});

// ── Physics 폼 상태 매핑 ────────────────────────────────────────────

describe('physicsFormStateOf', () => {
  it('첫 collider에서 값을 추출한다 (미지정은 DATA_MODEL 기본값)', () => {
    const form = physicsFormStateOf(BOX_PHYSICS);
    expect(form).toEqual({
      bodyType: 'dynamic',
      friction: 0.6,
      restitution: 0,
      density: 1,
      group: 'OBJECT',
      collidesWith: ['ENV', 'ROBOT', 'OBJECT'],
      isSensor: false,
      emitEvents: true,
      ccd: false,
    });
  });

  it('collider가 없으면 전체 기본값을 돌려준다', () => {
    const form = physicsFormStateOf({ bodyType: 'fixed', colliders: [] });
    expect(form.friction).toBe(0.5);
    expect(form.restitution).toBe(0);
    expect(form.density).toBe(1);
    expect(form.group).toBe('OBJECT');
    expect(form.collidesWith).toEqual(['ENV', 'ROBOT', 'OBJECT']);
    expect(form.isSensor).toBe(false);
  });
});

describe('applyPhysicsFormState — 전체 PhysicsSpec 재구성', () => {
  const ORIGINAL: PhysicsSpec = {
    bodyType: 'dynamic',
    linearDamping: 0.1,
    angularDamping: 0.2,
    gravityScale: 0.5,
    colliders: [
      {
        shape: { kind: 'box', halfExtents: [0.05, 0.05, 0.05] },
        offset: { position: [0, 0.1, 0] },
        friction: 0.6,
        group: 'OBJECT',
        collidesWith: ['ENV', 'ROBOT', 'OBJECT'],
        emitEvents: true,
      },
      {
        shape: { kind: 'sphere', radius: 0.03 },
        friction: 0.3,
        group: 'OBJECT',
        collidesWith: ['ENV'],
      },
    ],
  };
  const FORM = {
    bodyType: 'fixed',
    friction: 0.9,
    restitution: 0.4,
    density: 1.5,
    group: 'SENSOR_ZONE',
    collidesWith: ['ROBOT'],
    isSensor: true,
    emitEvents: false,
    ccd: true,
  } as const;

  it('bodyType/계수/그룹/플래그를 적용하고 모든 collider에 동일 반영한다', () => {
    const next = applyPhysicsFormState(ORIGINAL, { ...FORM, collidesWith: ['ROBOT'] });
    expect(next.bodyType).toBe('fixed');
    expect(next.colliders).toHaveLength(2);
    for (const collider of next.colliders) {
      expect(collider.friction).toBe(0.9);
      expect(collider.restitution).toBe(0.4);
      expect(collider.density).toBe(1.5);
      expect(collider.group).toBe('SENSOR_ZONE');
      expect(collider.collidesWith).toEqual(['ROBOT']);
      expect(collider.isSensor).toBe(true);
      expect(collider.emitEvents).toBe(false);
      expect(collider.ccd).toBe(true);
    }
  });

  it('폼에 없는 필드는 스프레드로 보존한다 (damping/gravityScale/shape/offset)', () => {
    const next = applyPhysicsFormState(ORIGINAL, { ...FORM, collidesWith: ['ROBOT'] });
    expect(next.linearDamping).toBe(0.1);
    expect(next.angularDamping).toBe(0.2);
    expect(next.gravityScale).toBe(0.5);
    expect(next.colliders[0]?.shape).toEqual({ kind: 'box', halfExtents: [0.05, 0.05, 0.05] });
    expect(next.colliders[0]?.offset).toEqual({ position: [0, 0.1, 0] });
    expect(next.colliders[1]?.shape).toEqual({ kind: 'sphere', radius: 0.03 });
  });

  it(`계수를 ${PHYSICS_PARAM_MIN}..${PHYSICS_PARAM_MAX}로 클램프한다`, () => {
    const next = applyPhysicsFormState(ORIGINAL, {
      ...FORM,
      collidesWith: ['ROBOT'],
      friction: 5,
      restitution: -1,
      density: 2.5,
    });
    expect(next.colliders[0]?.friction).toBe(PHYSICS_PARAM_MAX);
    expect(next.colliders[0]?.restitution).toBe(PHYSICS_PARAM_MIN);
    expect(next.colliders[0]?.density).toBe(PHYSICS_PARAM_MAX);
  });

  it('원본 spec을 변경하지 않는다 (불변 재구성)', () => {
    applyPhysicsFormState(ORIGINAL, { ...FORM, collidesWith: ['ROBOT'] });
    expect(ORIGINAL.bodyType).toBe('dynamic');
    expect(ORIGINAL.colliders[0]?.friction).toBe(0.6);
    expect(ORIGINAL.colliders[1]?.collidesWith).toEqual(['ENV']);
  });
});

// ── 셀렉트/체크박스 후보 목록 ────────────────────────────────────────

describe('bodyTypeChoices / groupChoices / collidesWithChoices', () => {
  it('bodyType 기본 후보는 dynamic/fixed/kinematicPosition (요구 §2)', () => {
    expect(bodyTypeChoices('dynamic')).toEqual(['dynamic', 'fixed', 'kinematicPosition']);
  });

  it('현재값이 후보 밖이면 뒤에 추가한다 (kinematicVelocity 데이터 보존)', () => {
    expect(bodyTypeChoices('kinematicVelocity')).toEqual([
      'dynamic',
      'fixed',
      'kinematicPosition',
      'kinematicVelocity',
    ]);
  });

  it('그룹 기본 후보는 ENV/ROBOT/OBJECT/SENSOR_ZONE — DEBUG는 현재값일 때만', () => {
    expect(groupChoices('OBJECT')).toEqual(['ENV', 'ROBOT', 'OBJECT', 'SENSOR_ZONE']);
    expect(groupChoices('DEBUG')).toEqual(['ENV', 'ROBOT', 'OBJECT', 'SENSOR_ZONE', 'DEBUG']);
  });

  it('collidesWith 후보: 기본 4종 + 현재값의 추가 그룹(중복 제거 — 드롭 방지)', () => {
    expect(collidesWithChoices(['ENV', 'ROBOT'])).toEqual([
      'ENV',
      'ROBOT',
      'OBJECT',
      'SENSOR_ZONE',
    ]);
    expect(collidesWithChoices(['ENV', 'DEBUG', 'DEBUG'])).toEqual([
      'ENV',
      'ROBOT',
      'OBJECT',
      'SENSOR_ZONE',
      'DEBUG',
    ]);
  });
});

describe('bodyTypeLabel / groupLabel — 원본 값 + 한국어 병기', () => {
  it('라벨에 원본 값이 포함된다 (스키마 값과의 대응 가시성)', () => {
    expect(bodyTypeLabel('dynamic')).toContain('dynamic');
    expect(bodyTypeLabel('kinematicPosition')).toContain('kinematicPosition');
    expect(groupLabel('ENV')).toContain('ENV');
    expect(groupLabel('SENSOR_ZONE')).toContain('SENSOR_ZONE');
  });
});
