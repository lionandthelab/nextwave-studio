// core/phase6-scenes.test.ts — Phase 6 샘플 씬 3종 + 시퀀스 2종 데이터 검증
//
// ROADMAP Phase 6 "샘플 씬 3종: 픽앤플레이스 / 장애물 회피 / 충돌 테스트베드" 자산이
// 스키마 검증(validateScene/validateSequence)과 프로젝트 규약(CLAUDE.md §5 충돌 그룹,
// DATA_MODEL §2 emitEvents 규칙)을 만족하는지 데이터 수준에서 고정한다.
// 물리/렌더 없이 순수 검증만 수행한다 — 실제 물리 거동(밀기 픽앤플레이스, 기둥 무접촉,
// 낙하/미끄럼/전도)은 scripts/gate-browser.mjs의 씬별 게이트가 검증한다.
//
// 핵심 규약(양쪽 필터 규칙): Rapier 쌍 필터는 양방향이다 — SENSOR_ZONE collider가
// OBJECT를 나열해도, OBJECT collider가 SENSOR_ZONE을 나열하지 않으면 쌍이 성립하지
// 않는다. 이 테스트가 양쪽 모두를 어서션한다 (EXPERIMENTS.md Phase 6 항목).

import { describe, expect, it } from 'vitest';
import pickAndPlaceSceneJson from '../assets/scenes/pick-and-place.scene.json';
import obstacleAvoidanceSceneJson from '../assets/scenes/obstacle-avoidance.scene.json';
import collisionTestbedSceneJson from '../assets/scenes/collision-testbed.scene.json';
import pickAndPlaceSequenceJson from '../assets/sequences/pick-and-place.sequence.json';
import obstacleAvoidanceSequenceJson from '../assets/sequences/obstacle-avoidance.sequence.json';
import { validateScene, validateSequence } from '../schema/validate';
import { isRobotSpec } from '../schema/types';
import type { ColliderSpec, EntitySpec, SceneSpec } from '../schema/types';

// ── 테스트 상수 (매직넘버 금지 — CLAUDE.md §4) ──────────────────────

/** 시퀀스별 step 수 (gate-browser PNP_STEP_COUNT/OA_STEP_COUNT와 일치해야 함) */
const PNP_SEQUENCE_STEP_COUNT = 9;
const OA_SEQUENCE_STEP_COUNT = 8;
/** collision-testbed 동적 바디 수 (bouncer, slider, wrecker, stack×3) */
const TESTBED_DYNAMIC_ENTITY_COUNT = 6;
/** bouncer 반발계수 — "튀는 공" 쇼케이스 계약 */
const TESTBED_BOUNCER_RESTITUTION = 0.8;

// ── 헬퍼 ────────────────────────────────────────────────────────────

function mustValidateScene(json: unknown, name: string): SceneSpec {
  const result = validateScene(json);
  if (!result.ok) throw new Error(`${name} 검증 실패:\n${result.errors.join('\n')}`);
  return result.value;
}

function entityOf(spec: SceneSpec, id: string): EntitySpec {
  const entity = spec.entities.find((e) => e.id === id);
  if (!entity) throw new Error(`씬 '${spec.name}'에 '${id}' 엔티티가 없습니다`);
  return entity;
}

function collidersOf(entity: EntitySpec): readonly ColliderSpec[] {
  const colliders = entity.physics?.colliders;
  if (!colliders || colliders.length === 0) {
    throw new Error(`엔티티 '${entity.id}'에 collider가 없습니다`);
  }
  return colliders;
}

function dynamicEntitiesOf(spec: SceneSpec): SceneSpec['entities'] {
  return spec.entities.filter((e) => e.physics?.bodyType === 'dynamic');
}

// ── 1. pick-and-place ───────────────────────────────────────────────

describe('pick-and-place.scene.json', () => {
  const spec = mustValidateScene(pickAndPlaceSceneJson, 'pick-and-place.scene.json');

  it('validateScene을 통과하고 robot 1대 + cargo + drop_zone을 갖춘다', () => {
    expect(spec.name).toBe('pick-and-place');
    const robots = spec.entities.filter(isRobotSpec);
    expect(robots).toHaveLength(1);
    expect(robots[0]?.id).toBe('arm');
    expect(robots[0]?.controller).toBe('sequence');
    expect(spec.environment?.ground).toBe(true);
  });

  it('cargo: 동적 OBJECT, SENSOR_ZONE 필터 포함(양쪽 규칙의 OBJECT 측) + emitEvents', () => {
    const cargo = entityOf(spec, 'cargo');
    expect(cargo.physics?.bodyType).toBe('dynamic');
    for (const collider of collidersOf(cargo)) {
      expect(collider.group).toBe('OBJECT');
      expect(collider.collidesWith).toContain('ROBOT');
      expect(collider.collidesWith).toContain('ENV');
      // 센서 쌍 성립 조건 — OBJECT 측도 SENSOR_ZONE을 나열해야 한다
      expect(collider.collidesWith).toContain('SENSOR_ZONE');
      expect(collider.emitEvents).toBe(true);
    }
  });

  it('drop_zone: 고정 SENSOR_ZONE sensor collider(양쪽 규칙의 SENSOR 측) + emitEvents', () => {
    const zone = entityOf(spec, 'drop_zone');
    expect(zone.physics?.bodyType).toBe('fixed');
    for (const collider of collidersOf(zone)) {
      expect(collider.isSensor).toBe(true);
      expect(collider.group).toBe('SENSOR_ZONE');
      expect(collider.collidesWith).toContain('OBJECT');
      expect(collider.emitEvents).toBe(true);
    }
  });

  it('pick-and-place.sequence.json이 씬 대비 validateSequence를 통과한다', () => {
    const result = validateSequence(pickAndPlaceSequenceJson, spec);
    if (!result.ok) throw new Error(result.errors.join('\n'));
    expect(result.value.robot).toBe('arm');
    expect(result.value.steps).toHaveLength(PNP_SEQUENCE_STEP_COUNT);
    // 밀기(push) 설계 계약: waitForCollision 배리어가 arm×cargo 접촉에 동기화된다
    const barrier = result.value.steps.find((s) => s.kind === 'waitForCollision');
    expect(barrier).toBeDefined();
    if (barrier?.kind === 'waitForCollision') {
      expect([...barrier.between].sort()).toEqual(['arm', 'cargo']);
    }
  });
});

// ── 2. obstacle-avoidance ───────────────────────────────────────────

describe('obstacle-avoidance.scene.json', () => {
  const spec = mustValidateScene(obstacleAvoidanceSceneJson, 'obstacle-avoidance.scene.json');

  it('validateScene을 통과하고 robot + pillar + 박스 2개를 갖춘다', () => {
    expect(spec.name).toBe('obstacle-avoidance');
    expect(spec.entities.filter(isRobotSpec)).toHaveLength(1);
    expect(dynamicEntitiesOf(spec).map((e) => e.id).sort()).toEqual([
      'target_box',
      'waypoint_box',
    ]);
    expect(spec.environment?.ground).toBe(true);
  });

  it('pillar: 고정 ENV + emitEvents — 닿았다면 반드시 이력에 남는다(무접촉 어서션의 전제)', () => {
    const pillar = entityOf(spec, 'pillar');
    expect(pillar.physics?.bodyType).toBe('fixed');
    for (const collider of collidersOf(pillar)) {
      expect(collider.group).toBe('ENV');
      expect(collider.collidesWith).toContain('ROBOT');
      expect(collider.emitEvents).toBe(true);
    }
  });

  it('target_box: 동적 OBJECT + ROBOT 필터 + emitEvents', () => {
    const target = entityOf(spec, 'target_box');
    expect(target.physics?.bodyType).toBe('dynamic');
    for (const collider of collidersOf(target)) {
      expect(collider.group).toBe('OBJECT');
      expect(collider.collidesWith).toContain('ROBOT');
      expect(collider.emitEvents).toBe(true);
    }
  });

  it('obstacle-avoidance.sequence.json이 씬 대비 validateSequence를 통과한다', () => {
    const result = validateSequence(obstacleAvoidanceSequenceJson, spec);
    if (!result.ok) throw new Error(result.errors.join('\n'));
    expect(result.value.robot).toBe('arm');
    expect(result.value.steps).toHaveLength(OA_SEQUENCE_STEP_COUNT);
    const barrier = result.value.steps.find((s) => s.kind === 'waitForCollision');
    expect(barrier).toBeDefined();
    if (barrier?.kind === 'waitForCollision') {
      expect([...barrier.between].sort()).toEqual(['arm', 'target_box']);
    }
  });
});

// ── 3. collision-testbed ────────────────────────────────────────────

describe('collision-testbed.scene.json', () => {
  const spec = mustValidateScene(collisionTestbedSceneJson, 'collision-testbed.scene.json');

  it('validateScene을 통과하고 로봇 없이 동적 6개 + 경사로 2개 + 센서 게이트를 갖춘다', () => {
    expect(spec.name).toBe('collision-testbed');
    expect(spec.entities.filter(isRobotSpec)).toHaveLength(0);
    expect(dynamicEntitiesOf(spec)).toHaveLength(TESTBED_DYNAMIC_ENTITY_COUNT);
    expect(spec.environment?.ground).toBe(true);
    // 기울인 고정 경사로 — rotation이 실제로 지정되어 있다(회전된 fixed box 계약)
    for (const rampId of ['slide_ramp', 'roll_ramp']) {
      const ramp = entityOf(spec, rampId);
      expect(ramp.physics?.bodyType).toBe('fixed');
      expect(ramp.transform.rotation).toBeDefined();
    }
  });

  it('모든 동적 OBJECT collider가 emitEvents:true다 (게이트의 접촉 쌍 어서션 전제)', () => {
    for (const entity of dynamicEntitiesOf(spec)) {
      for (const collider of collidersOf(entity)) {
        expect(collider.group, entity.id).toBe('OBJECT');
        expect(collider.emitEvents, entity.id).toBe(true);
      }
    }
  });

  it('bouncer: restitution 0.8 (튀는 공 쇼케이스)', () => {
    const bouncer = entityOf(spec, 'bouncer');
    for (const collider of collidersOf(bouncer)) {
      expect(collider.restitution).toBe(TESTBED_BOUNCER_RESTITUTION);
    }
  });

  it('slide_gate ↔ slider 센서 쌍이 양쪽 필터를 모두 갖춘다', () => {
    const gate = entityOf(spec, 'slide_gate');
    for (const collider of collidersOf(gate)) {
      expect(collider.isSensor).toBe(true);
      expect(collider.group).toBe('SENSOR_ZONE');
      expect(collider.collidesWith).toContain('OBJECT');
      expect(collider.emitEvents).toBe(true);
    }
    const slider = entityOf(spec, 'slider');
    for (const collider of collidersOf(slider)) {
      expect(collider.collidesWith).toContain('SENSOR_ZONE');
    }
  });
});
