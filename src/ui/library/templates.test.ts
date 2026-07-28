// ui/library/templates.test.ts — 라이브러리 템플릿 팩토리 단위 테스트 (DOM 비의존, node)
//
// 계약 (templates.ts 헤더):
// - 모든 템플릿의 create() 출력은 최소 씬으로 감싸면 validateScene을 통과한다.
// - id는 uniquify(idBase)로 발급된다 — 팩토리는 자체 카운터를 들지 않는다.
// - 그룹/필터 배정은 CLAUDE.md §5 + phase6 샘플 씬 규약(OBJECT 측 SENSOR_ZONE 포함,
//   센서 양쪽 필터 규칙)과 일치한다.
// mountLibrary의 DOM 조립은 얇은 글루라 여기서 다루지 않는다(브라우저 게이트 몫).

import { describe, expect, it } from 'vitest';
import {
  DEFAULT_BOX_HALF_EXTENTS,
  LIBRARY_TEMPLATES,
  filterTemplates,
  matchesQuery,
  templateByKey,
} from './templates';
import type { LibraryTemplate, Uniquify } from './templates';
import { isRobotSpec, validateScene } from '../../schema';
import type { EntitySpec, SceneSpec } from '../../schema';

// ── 테스트 헬퍼 ─────────────────────────────────────────────────────

/** 'box' → 'box_1' 형식의 단순 uniquifier (통합자 구현의 최소 모형) */
const uniquify1: Uniquify = (base) => `${base}_1`;

/** 엔티티들을 최소 유효 씬으로 감싼다 (DATA_MODEL §5 필수 필드만) */
function wrapInScene(entities: EntitySpec[]): SceneSpec {
  return {
    name: 'template-test',
    version: 1,
    gravity: [0, -9.81, 0],
    timestepHz: 240,
    entities,
  };
}

function mustTemplate(key: string): LibraryTemplate {
  const template = templateByKey(key);
  if (!template) throw new Error(`template '${key}' not found`);
  return template;
}

// ── 전 템플릿 공통 계약 ─────────────────────────────────────────────

describe('LIBRARY_TEMPLATES 공통 계약', () => {
  it('키는 레지스트리 내에서 유일하다', () => {
    const keys = LIBRARY_TEMPLATES.map((t) => t.key);
    expect(new Set(keys).size).toBe(keys.length);
  });

  it.each(LIBRARY_TEMPLATES.map((t) => [t.key, t] as const))(
    "'%s' create() 출력은 최소 씬으로 감싸면 validateScene을 통과한다",
    (_key, template) => {
      const spec = template.create(uniquify1);
      const result = validateScene(wrapInScene([spec]));
      expect(result.ok, JSON.stringify(!result.ok ? result.errors : [])).toBe(true);
    },
  );

  it.each(LIBRARY_TEMPLATES.map((t) => [t.key, t] as const))(
    "'%s' id는 uniquify(idBase)로 발급된다",
    (_key, template) => {
      const spec = template.create(uniquify1);
      expect(spec.id).toBe(`${template.idBase}_1`);
    },
  );

  it('create()는 호출마다 독립 객체를 만든다 (공유 참조 없음)', () => {
    for (const template of LIBRARY_TEMPLATES) {
      const a = template.create(uniquify1);
      const b = template.create(uniquify1);
      expect(a).not.toBe(b);
      expect(a).toEqual(b); // 같은 uniquify면 내용은 동일
      // 한쪽 변형이 다른 쪽에 새지 않는다 (깊은 독립성)
      a.transform.position[0] = 999;
      expect(b.transform.position[0]).not.toBe(999);
    }
  });

  it('라벨/설명/아이콘은 비어 있지 않다 (카드 표기 계약)', () => {
    for (const template of LIBRARY_TEMPLATES) {
      expect(template.labelKo.length).toBeGreaterThan(0);
      expect(template.descriptionKo.length).toBeGreaterThan(0);
      expect(template.icon.length).toBeGreaterThan(0);
    }
  });

  it('아이콘은 SVG 아이콘 이름이다 — 이모지/딩벳 회귀 차단 (UX_AUDIT C-13)', () => {
    // 타입(IconName)이 1차 방어지만, 문자열 리터럴 유니온이라 "그럴듯한" 오타는
    // 컴파일러가 잡아도 **이모지를 다시 밀수입하는 습관**은 리뷰가 놓칠 수 있다.
    // 아이콘 이름은 예외 없이 ASCII 식별자다.
    for (const template of LIBRARY_TEMPLATES) {
      expect(template.icon).toMatch(/^[a-zA-Z][a-zA-Z0-9]*$/);
    }
  });

  it('아이콘은 형상/도메인 의미와 일치한다 (카드 6종 + 로봇)', () => {
    const byKey = new Map(LIBRARY_TEMPLATES.map((t) => [t.key, t.icon]));
    expect(byKey.get('box')).toBe('shapeBox');
    expect(byKey.get('sphere')).toBe('shapeSphere');
    expect(byKey.get('cylinder')).toBe('shapeCylinder');
    expect(byKey.get('capsule')).toBe('shapeCapsule');
    expect(byKey.get('plane')).toBe('shapePlane');
    expect(byKey.get('sensor-zone')).toBe('target');
    expect(byKey.get('arm-6')).toBe('robotArm');
  });

  it('전 템플릿을 한 씬에 넣어도 유효하다 (uniquify로 id 충돌 없음)', () => {
    const entities = LIBRARY_TEMPLATES.map((t) => t.create(uniquify1));
    const result = validateScene(wrapInScene(entities));
    expect(result.ok).toBe(true);
  });
});

// ── 동적 프리미티브 (Box 대표 + 시각/collider 일치) ─────────────────

describe('동적 프리미티브 템플릿', () => {
  it('box: 기본 치수 halfExtents [0.05,0.05,0.05] + OBJECT 그룹 배정', () => {
    const spec = mustTemplate('box').create(uniquify1);
    expect(spec.type).toBe('object');
    expect(spec.physics?.bodyType).toBe('dynamic');

    const collider = spec.physics?.colliders[0];
    expect(collider?.shape).toEqual({ kind: 'box', halfExtents: DEFAULT_BOX_HALF_EXTENTS });
    expect(collider?.group).toBe('OBJECT');
    expect(collider?.emitEvents).toBe(true);
    // 과제 규약의 최소 집합 [ENV, ROBOT, OBJECT] + 센서 양쪽 필터 규칙의 SENSOR_ZONE
    // (phase6-scenes.test.ts: OBJECT 측이 SENSOR_ZONE을 나열해야 센서 쌍이 성립)
    expect(collider?.collidesWith).toEqual(
      expect.arrayContaining(['ENV', 'ROBOT', 'OBJECT', 'SENSOR_ZONE']),
    );
  });

  it.each([['box'], ['sphere'], ['cylinder'], ['capsule']] as const)(
    "'%s': 시각 primitive 형상과 collider 형상이 동일하다 (단일 치수 진실)",
    (key) => {
      const spec = mustTemplate(key).create(uniquify1);
      expect(spec.visual.kind).toBe('primitive');
      expect(spec.visual.primitive).toEqual(spec.physics?.colliders[0]?.shape);
    },
  );

  it('동적 프리미티브는 바닥 접지 높이로 스폰된다 (최저점 y=0)', () => {
    const box = mustTemplate('box').create(uniquify1);
    expect(box.transform.position[1]).toBeCloseTo(0.05, 9);
    const sphere = mustTemplate('sphere').create(uniquify1);
    expect(sphere.transform.position[1]).toBeCloseTo(0.05, 9);
    const capsule = mustTemplate('capsule').create(uniquify1);
    // 캡슐 최저점 = 중심 - (halfHeight + radius)
    expect(capsule.transform.position[1]).toBeCloseTo(0.05 + 0.03, 9);
  });
});

// ── 얇은 Plane (정적 ENV) ───────────────────────────────────────────

describe('plane 템플릿', () => {
  it('정적 fixed 바디 + ENV 그룹의 얇은 박스다', () => {
    const spec = mustTemplate('plane').create(uniquify1);
    expect(spec.type).toBe('static');
    expect(spec.physics?.bodyType).toBe('fixed');
    const collider = spec.physics?.colliders[0];
    expect(collider?.group).toBe('ENV');
    expect(collider?.shape.kind).toBe('box');
    if (collider?.shape.kind === 'box') {
      // "얇은" 판: 높이 half가 가로/세로 half보다 훨씬 작다
      expect(collider.shape.halfExtents[1]).toBeLessThan(collider.shape.halfExtents[0] / 5);
    }
  });
});

// ── Sensor Zone (감지 전용) ─────────────────────────────────────────

describe('sensor-zone 템플릿', () => {
  it('static + fixed + isSensor + SENSOR_ZONE 그룹 + [OBJECT, ROBOT] 필터 + emitEvents', () => {
    const spec = mustTemplate('sensor-zone').create(uniquify1);
    expect(spec.type).toBe('static');
    expect(spec.physics?.bodyType).toBe('fixed');
    const collider = spec.physics?.colliders[0];
    expect(collider?.isSensor).toBe(true);
    expect(collider?.group).toBe('SENSOR_ZONE');
    expect(collider?.collidesWith).toEqual(expect.arrayContaining(['OBJECT', 'ROBOT']));
    expect(collider?.emitEvents).toBe(true);
  });
});

// ── Arm-6 로봇 ──────────────────────────────────────────────────────

describe('arm-6 템플릿', () => {
  it('arm-and-boxes 씬과 동일한 URDF/home/gripper 구성의 RobotSpec이다', () => {
    const spec = mustTemplate('arm-6').create(uniquify1);
    expect(isRobotSpec(spec)).toBe(true);
    if (!isRobotSpec(spec)) return;
    expect(spec.urdf).toBe('assets/robots/arm6/arm6.urdf');
    expect(spec.visual).toEqual({ kind: 'urdf', ref: 'assets/robots/arm6/arm6.urdf' });
    expect(spec.controller).toBe('sequence');
    expect(spec.linkColliders).toBe('fromVisual');
    expect(spec.selfCollision).toBe(false);
    expect(spec.home).toEqual({ joint1: 0.0, joint2: -0.6, joint3: 1.2, joint5: -0.6 });
    expect(spec.gripper).toEqual({
      joints: ['finger_left_joint', 'finger_right_joint'],
      open: 0.03,
      close: 0.0,
    });
  });
});

// ── 검색 필터 ───────────────────────────────────────────────────────

describe('filterTemplates / matchesQuery', () => {
  it('빈/공백 검색어는 전체를 반환한다', () => {
    expect(filterTemplates(LIBRARY_TEMPLATES, '')).toHaveLength(LIBRARY_TEMPLATES.length);
    expect(filterTemplates(LIBRARY_TEMPLATES, '   ')).toHaveLength(LIBRARY_TEMPLATES.length);
  });

  it('키/라벨/설명에 대소문자 무시 부분 일치한다', () => {
    const byKey = filterTemplates(LIBRARY_TEMPLATES, 'BOX');
    expect(byKey.map((t) => t.key)).toContain('box');

    const byKoreanLabel = filterTemplates(LIBRARY_TEMPLATES, '로봇팔');
    expect(byKoreanLabel.map((t) => t.key)).toEqual(['arm-6']);

    const byDescription = filterTemplates(LIBRARY_TEMPLATES, '감지');
    expect(byDescription.map((t) => t.key)).toContain('sensor-zone');
  });

  it('일치 항목이 없으면 빈 배열', () => {
    expect(filterTemplates(LIBRARY_TEMPLATES, 'no-such-template')).toEqual([]);
  });

  it('matchesQuery는 앞뒤 공백을 무시한다', () => {
    const box = mustTemplate('box');
    expect(matchesQuery(box, '  box  ')).toBe(true);
  });
});

// ── templateByKey (드롭 페이로드 해석 계약) ─────────────────────────

describe('templateByKey', () => {
  it('등록된 키는 해당 템플릿을, 미지 키는 undefined를 반환한다', () => {
    expect(templateByKey('box')?.key).toBe('box');
    expect(templateByKey('unknown')).toBeUndefined();
  });
});
