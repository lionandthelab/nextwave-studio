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
import conveyorPickPlaceSceneJson from '../assets/scenes/conveyor-pick-place.scene.json';
import lLineCellSceneJson from '../assets/scenes/l-line-cell.scene.json';
import conveyorPickPlaceSequenceJson from '../assets/sequences/conveyor-pick-place.sequence.json';
import pickAndPlaceSequenceJson from '../assets/sequences/pick-and-place.sequence.json';
import obstacleAvoidanceSequenceJson from '../assets/sequences/obstacle-avoidance.sequence.json';
import { validateScene, validateSequence } from '../schema/validate';
import { isRobotSpec } from '../schema/types';
import type {
  ColliderSpec,
  ControlSequence,
  ControlStep,
  EntitySpec,
  SceneSpec,
  Vec3,
} from '../schema/types';

// ── 테스트 상수 (매직넘버 금지 — CLAUDE.md §4) ──────────────────────

/** 시퀀스별 step 수 (gate-browser PNP_STEP_COUNT/OA_STEP_COUNT와 일치해야 함) */
const PNP_SEQUENCE_STEP_COUNT = 11;
const OA_SEQUENCE_STEP_COUNT = 8;
/**
 * cargo 마찰 하한. 이 프로젝트의 그리퍼는 실제 파지를 모사하지 않고 평행 손가락의 접촉
 * 마찰만 있으므로, 들어 올려 옮기려면 기본값(0.5)보다 높은 마찰이 필요하다.
 *
 * 상한도 실질적으로 존재한다(그래서 "높을수록 좋다"가 아니다): 마찰이 너무 높으면 첫
 * 접촉에서 상자가 곧바로 물려 **접촉이 다시 시작되지 않고**, 그러면 waitForCollision
 * 배리어가 새 start 이벤트를 못 봐 timeout까지 6초를 버린다(실측 1.0·1.8에서 재현).
 * 0.8이 "들어 올릴 만큼 강하고 배리어를 굶기지 않을 만큼" 되는 지점이다.
 */
const PNP_MIN_GRASP_FRICTION = 0.7;
/** conveyor-pick-place step 수 (gate-browser CPP_STEP_COUNT와 일치해야 함) */
const CPP_SEQUENCE_STEP_COUNT = 33;
/** 라인에서 연속으로 처리하는 사이클 수 = 컨베이어에서 나오는 상자 수 */
const CPP_CYCLE_COUNT = 3;
/** 사이클 순서대로의 대상 아이템 */
const CPP_CYCLE_ITEMS = ['item_a', 'item_b', 'item_c'] as const;
/**
 * 이송 호(joint1 변화량) 상한 (rad).
 *
 * 마찰 파지는 **호 길이**에 진다 — 회전이 길수록 원심 성분이 오래 걸려 상자가 손가락
 * 사이에서 미끄러져 나간다. 브라우저 실측: 0.21–0.30 rad는 3회 모두 성공, 0.395 rad
 * 이상은 이송 중 낙하("가다가 떨어뜨린다" 사용자 보고). 상한을 실측 성공 구간 끝에 건다.
 */
const CPP_MAX_TRANSPORT_ARC_RAD = 0.3;
/** 파지에 필요한 아이템 마찰 하한 (실측: 1.6에서 3사이클 모두 파지 성공) */
const CPP_MIN_ITEM_FRICTION = 1.0;
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

/** 첫 collider의 box halfExtents — box가 아니면 계약 위반이므로 던진다 */
function boxHalfExtentsOf(entity: EntitySpec): Vec3 {
  const shape = collidersOf(entity)[0]?.shape;
  if (shape?.kind !== 'box') {
    throw new Error(`엔티티 '${entity.id}'의 첫 collider가 box가 아닙니다`);
  }
  return shape.halfExtents;
}

/** moveJoints/setJoints의 관절 목표값, 그 외 step은 undefined */
function jointTargetsOf(step: ControlStep | undefined): Record<string, number> | undefined {
  if (!step) return undefined;
  return step.kind === 'moveJoints' || step.kind === 'setJoints' ? step.targets : undefined;
}

function dynamicEntitiesOf(spec: SceneSpec): SceneSpec['entities'] {
  return spec.entities.filter((e) => e.physics?.bodyType === 'dynamic');
}

// ── 1. pick-and-place ───────────────────────────────────────────────

describe('pick-and-place.scene.json', () => {
  const spec = mustValidateScene(pickAndPlaceSceneJson, 'pick-and-place.scene.json');

  it('validateScene을 통과하고 robot 1대 + cargo + drop_zone + drop_shelf를 갖춘다', () => {
    expect(spec.name).toBe('pick-and-place');
    const robots = spec.entities.filter(isRobotSpec);
    expect(robots).toHaveLength(1);
    expect(robots[0]?.id).toBe('arm');
    expect(robots[0]?.controller).toBe('sequence');
    expect(spec.environment?.ground).toBe(true);
    expect(spec.entities.map((e) => e.id).sort()).toEqual([
      'arm',
      'cargo',
      'drop_shelf',
      'drop_zone',
    ]);
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

  /**
   * 감지 존과 선반의 **역할 분리** (사용자 보고: "선반에 부딪히는데 충돌 표시 안 뜨고 관통").
   *
   * 원인은 하나의 상자가 두 역할을 겸한 것이었다: 도착 감지를 하려고 sensor로 만들었더니
   * (물리 반응 없음 = 관통) collidesWith도 [OBJECT]뿐이라 **로봇과는 쌍조차 성립하지 않아**
   * 이벤트가 0건이었다. 눈에는 단단한 상자인데 물리적으로는 존재하지 않았던 셈이다.
   *
   * 이제 둘로 나눈다: 감지는 sensor(drop_zone)가, 실체는 ENV 고체(drop_shelf)가 맡는다.
   * 선반의 collidesWith에 ROBOT이 반드시 있어야 접촉 이벤트가 나오고, 그래야
   * classifyContact가 'unexpected'(= 충돌)로 승격할 재료가 생긴다.
   */
  it('drop_shelf: 실체 있는 고정 ENV — sensor가 아니고 ROBOT과 쌍이 성립한다', () => {
    const shelf = entityOf(spec, 'drop_shelf');
    expect(shelf.physics?.bodyType).toBe('fixed');
    for (const collider of collidersOf(shelf)) {
      expect(collider.isSensor).not.toBe(true);
      expect(collider.group).toBe('ENV');
      // ROBOT이 빠지면 팔이 선반을 지나가도 이벤트가 0건이다 — 이 씬의 핵심 회귀
      expect(collider.collidesWith).toContain('ROBOT');
      expect(collider.collidesWith).toContain('OBJECT');
      expect(collider.emitEvents).toBe(true);
    }
  });

  it('감지 존과 선반은 서로 다른 엔티티다 — 한 상자가 두 역할을 겸하지 않는다', () => {
    const zone = entityOf(spec, 'drop_zone');
    const shelf = entityOf(spec, 'drop_shelf');
    expect(zone.id).not.toBe(shelf.id);
    const zoneCollider = collidersOf(zone)[0];
    const shelfCollider = collidersOf(shelf)[0];
    expect(zoneCollider?.isSensor).toBe(true);
    expect(shelfCollider?.isSensor).not.toBe(true);
  });

  it('pick-and-place.sequence.json이 씬 대비 validateSequence를 통과한다', () => {
    const result = validateSequence(pickAndPlaceSequenceJson, spec);
    if (!result.ok) throw new Error(result.errors.join('\n'));
    expect(result.value.robot).toBe('arm');
    expect(result.value.steps).toHaveLength(PNP_SEQUENCE_STEP_COUNT);
    // 파지 설계 계약: waitForCollision 배리어가 arm×cargo 접촉에 동기화된다
    const barrier = result.value.steps.find((s) => s.kind === 'waitForCollision');
    expect(barrier).toBeDefined();
    if (barrier?.kind === 'waitForCollision') {
      expect([...barrier.between].sort()).toEqual(['arm', 'cargo']);
    }
  });

  /**
   * ★ 사용자 요청: "상자를 잡고 조금 위로 올려서 플레이싱". 밀기(push)가 아니라
   * **들어 올려 옮기는** 시퀀스임을 데이터 수준에서 고정한다 — 파지 → 상승 → 이송 →
   * 하강 → 놓기 순서가 유지되어야 한다. 실제로 상자가 떠오르는지는 브라우저 게이트가 잰다.
   */
  it('파지 후 들어 올렸다가 내려놓는 순서다 (grip → lift → 이송 → lower → release)', () => {
    const result = validateSequence(pickAndPlaceSequenceJson, spec);
    if (!result.ok) throw new Error(result.errors.join('\n'));
    const steps = result.value.steps;
    const gripIndex = steps.findIndex((s) => s.kind === 'gripper' && typeof s.state === 'number');
    const releaseIndex = steps.findIndex(
      (s, i) => i > gripIndex && s.kind === 'gripper' && s.state === 'open',
    );
    expect(gripIndex).toBeGreaterThanOrEqual(0);
    expect(releaseIndex).toBeGreaterThan(gripIndex);

    // 파지와 놓기 사이에 **상승 → 이송 → 하강** 3개의 moveJoints가 있어야 한다
    const between = steps.slice(gripIndex + 1, releaseIndex);
    const moves = between.filter((s) => s.kind === 'moveJoints');
    expect(moves.length).toBeGreaterThanOrEqual(3);

    // 상승과 하강은 같은 관절 집합을 반대로 움직인다 (joint2가 대표)
    const lift = moves[0];
    const lower = moves[moves.length - 1];
    if (lift?.kind !== 'moveJoints' || lower?.kind !== 'moveJoints') throw new Error('moveJoints 기대');
    const liftJ2 = lift.targets['joint2'];
    const lowerJ2 = lower.targets['joint2'];
    expect(liftJ2, '상승 step이 joint2를 조정해야 한다').toBeDefined();
    expect(lowerJ2, '하강 step이 joint2를 조정해야 한다').toBeDefined();
    // joint2가 작을수록 팔이 펴져 그리퍼가 높다 — 상승 목표가 하강 목표보다 작아야 한다
    expect(liftJ2 ?? 0).toBeLessThan(lowerJ2 ?? 0);
  });

  it('cargo 마찰이 파지를 버틸 만큼 높다 (그리퍼가 실제 파지를 모사하지 않는 보정)', () => {
    const cargo = entityOf(spec, 'cargo');
    for (const collider of collidersOf(cargo)) {
      expect(collider.friction ?? 0).toBeGreaterThanOrEqual(PNP_MIN_GRASP_FRICTION);
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

// ── 4. conveyor-pick-place (컨베이어 라인) ──────────────────────────
//
// 이 씬이 고정하는 새 개념: **표면 구동 컨베이어**(DATA_MODEL §4.2)와 **포토아이 게이트**.
// 게이트가 없으면 로봇이 라인 위에서 물건을 기다려야 하고, 지나가는 다른 물건이 팔에
// 부딪혀 정상 실행이 매번 충돌을 보고한다(실측). 감지로 트리거하면 팔은 라인 밖에서
// 대기하다가 필요한 순간에만 내려온다 — 실제 컨베이어 셀의 동작이기도 하다.

describe('conveyor-pick-place.scene.json', () => {
  const spec = mustValidateScene(conveyorPickPlaceSceneJson, 'conveyor-pick-place.scene.json');

  it('validateScene 통과 + 로봇/벨트/게이트/감지존/아이템 3개 + 센서 기둥 장식', () => {
    expect(spec.name).toBe('conveyor-pick-place');
    expect(spec.entities.filter(isRobotSpec)).toHaveLength(1);
    expect(spec.entities.map((e) => e.id).sort()).toEqual([
      'arm',
      'belt',
      'drop_zone',
      'gate_post_l',
      'gate_post_r',
      ...CPP_CYCLE_ITEMS,
      'pick_gate',
    ]);
    expect(spec.environment?.ground).toBe(true);
  });

  /**
   * 아이템 수 = 사이클 수. 하나라도 모자라면 시퀀스의 마지막 사이클이 영원히 오지 않는
   * 물건을 기다리다 배리어 timeout으로 죽는다 — 실행은 "done"이 아니라 오류로 끝난다.
   */
  it('라인 위 아이템 수가 시퀀스 사이클 수와 같다', () => {
    const items = spec.entities.filter((e) => e.id.startsWith('item_'));
    expect(items).toHaveLength(CPP_CYCLE_COUNT);
  });

  /**
   * 광전 센서 기둥은 **순수 장식**이다 (§2.1 "시각 전용 요소" 예외).
   * 물리를 주면 로봇 작업 반경에 새 장애물이 생겨, 정상 예제가 매 실행 충돌을 보고한다.
   */
  it('gate_post_*: 물리 없는 순수 장식 — 로봇 경로에 장애물을 추가하지 않는다', () => {
    for (const id of ['gate_post_l', 'gate_post_r']) {
      expect(entityOf(spec, id).physics).toBeUndefined();
    }
  });

  /**
   * ★ 사용자 보고: "칸막이? 문? 과 드랍존이 그냥 녹색 직사각형으로 보여서 관통하는 것처럼
   * 보여." 통과 가능한 것(sensor)은 **통과 가능해 보여야** 한다 — 불투명 상자로 그리면
   * 화면이 "단단한 벽"이라고 말해 사용자가 관통을 결함으로 읽는다.
   */
  it('감지 존은 반투명 + 모서리 선으로 그려지고, 빔과 바닥 패드는 색으로 구분된다', () => {
    const gate = entityOf(spec, 'pick_gate');
    const zone = entityOf(spec, 'drop_zone');
    for (const zoneEntity of [gate, zone]) {
      const { opacity, edges } = zoneEntity.visual;
      expect(opacity, `'${zoneEntity.id}'가 불투명하면 단단한 벽으로 읽힌다`).toBeDefined();
      expect(opacity).toBeLessThan(1);
      expect(edges, `'${zoneEntity.id}'의 경계가 없으면 부피로 읽히지 않는다`).toBe(true);
    }
    // 두 존은 서로 다른 색이어야 구분된다 (사용자가 "구분이 안 된다"고 보고한 지점)
    expect(gate.visual.color).not.toBe(zone.visual.color);
  });

  it('belt: conveyor 블록을 가진 고정 ENV 박스 — sensor가 아니고 ROBOT과 쌍이 성립한다', () => {
    const belt = entityOf(spec, 'belt');
    expect(belt.type).toBe('static');
    expect(belt.physics?.bodyType).toBe('fixed');
    expect(belt.conveyor).toBeDefined();
    for (const collider of collidersOf(belt)) {
      expect(collider.shape.kind).toBe('box'); // 재순환 지점 계산의 전제
      expect(collider.isSensor).not.toBe(true); // sensor면 접촉이 없어 아무것도 못 싣는다
      expect(collider.group).toBe('ENV');
      expect(collider.collidesWith).toContain('ROBOT');
      expect(collider.collidesWith).toContain('OBJECT');
      expect(collider.emitEvents).toBe(true);
    }
  });

  it('belt.conveyor: 수평 방향 + 양수 속도 + 재순환 켜짐 ("물건이 계속 온다"의 근거)', () => {
    const conveyor = entityOf(spec, 'belt').conveyor;
    expect(conveyor).toBeDefined();
    if (!conveyor) return;
    expect(Math.hypot(conveyor.direction[0], conveyor.direction[2])).toBeGreaterThan(0);
    expect(conveyor.speedMps).toBeGreaterThan(0);
    expect(conveyor.recycle).toBe(true);
  });

  it('item_*: 동적 OBJECT + SENSOR_ZONE 필터(양쪽 규칙) + emitEvents + 파지 마찰', () => {
    for (const id of CPP_CYCLE_ITEMS) {
      const item = entityOf(spec, id);
      expect(item.physics?.bodyType).toBe('dynamic');
      for (const collider of collidersOf(item)) {
        expect(collider.group).toBe('OBJECT');
        expect(collider.collidesWith).toContain('ENV'); // 벨트에 얹히려면 필요
        expect(collider.collidesWith).toContain('ROBOT');
        // 포토아이 게이트·감지 존이 성립하려면 OBJECT 측에도 SENSOR_ZONE이 있어야 한다
        expect(collider.collidesWith).toContain('SENSOR_ZONE');
        expect(collider.emitEvents).toBe(true);
        // 이 예제의 파지는 순수 마찰이다(흡착/조인트 없음) — 마찰이 낮으면 이송 중 낙하한다
        expect(collider.friction ?? 0).toBeGreaterThanOrEqual(CPP_MIN_ITEM_FRICTION);
      }
    }
  });

  /**
   * ★ 사용자 보고: "컨베이어에서 나오는 상자 한 세 개 정도 연속으로 드랍존에 안착."
   *
   * 세 개를 같은 지점에 놓으면 나중 상자가 먼저 놓인 상자를 +x로 밀어낸다(실측: 놓기
   * 지점 x≈0.335 → 밀려서 0.596까지). 감지 존은 **놓는 한 점**이 아니라 밀려 쌓이는
   * **적재 레인 전체**를 덮어야 한다 — 안 그러면 정상 실행인데 "존에 안 들어갔다"가 된다.
   * (실측 안착 범위 x 0.335–0.596 / z 0.041–0.060.)
   */
  it('drop_zone: 세 상자가 밀려 쌓이는 레인 전체를 덮고, 벨트와 겹치지 않는다', () => {
    const zone = entityOf(spec, 'drop_zone');
    const [zx, , zz] = zone.transform.position;
    const [hx, , hz] = boxHalfExtentsOf(zone);

    // 실측 안착 범위를 여유 없이라도 덮는가
    expect(zx - hx).toBeLessThanOrEqual(0.335);
    expect(zx + hx).toBeGreaterThanOrEqual(0.596);
    expect(zz - hz).toBeLessThanOrEqual(0.041);
    expect(zz + hz).toBeGreaterThanOrEqual(0.06);

    // 벨트와 XZ가 겹치면 라인 위를 지나는 상자가 매번 존 진입 이벤트를 낸다(오탐)
    const belt = entityOf(spec, 'belt');
    const beltNearZ = belt.transform.position[2] - boxHalfExtentsOf(belt)[2];
    expect(zz + hz, '감지 존이 벨트 아래까지 뻗으면 통과하는 상자를 안착으로 센다').toBeLessThan(
      beltNearZ,
    );
  });

  it('pick_gate/drop_zone: 태그 붙은 감지 존 (sensor + SENSOR_ZONE)', () => {
    for (const id of ['pick_gate', 'drop_zone']) {
      const zone = entityOf(spec, id);
      expect(zone.tags).toContain('detection-zone');
      for (const collider of collidersOf(zone)) {
        expect(collider.isSensor).toBe(true);
        expect(collider.group).toBe('SENSOR_ZONE');
        expect(collider.collidesWith).toContain('OBJECT');
        expect(collider.emitEvents).toBe(true);
      }
    }
  });

  // 모듈 스코프에서 검증하면 씬이 깨졌을 때 suite 자체가 수집에 실패해 나머지 테스트가
  // 전부 사라진다 — 원인 한 줄이 28개의 침묵을 만든다. 테스트 안에서 지연 호출한다.
  const loadSequence = (): ControlSequence => {
    const result = validateSequence(conveyorPickPlaceSequenceJson, spec);
    if (!result.ok) throw new Error(result.errors.join('\n'));
    return result.value;
  };

  it('시퀀스가 씬 대비 검증을 통과하고 사이클마다 배리어 2개(포토아이 → 픽)를 갖는다', () => {
    const sequence = loadSequence();
    expect(sequence.robot).toBe('arm');
    expect(sequence.steps).toHaveLength(CPP_SEQUENCE_STEP_COUNT);

    const pairs = sequence.steps.flatMap((s) =>
      s.kind === 'waitForCollision' ? [[...s.between].sort().join('×')] : [],
    );
    // 사이클마다: 1) 포토아이가 도착을 감지 → 2) 라인이 물건을 그리퍼까지 실어 온다
    expect(pairs).toHaveLength(CPP_CYCLE_COUNT * 2);
    for (const item of CPP_CYCLE_ITEMS) {
      expect(pairs).toContain(`${item}×pick_gate`);
      expect(pairs).toContain(`arm×${item}`);
    }
  });

  /**
   * 사이클은 아이템 순서대로 정확히 한 번씩 돈다. 순서가 섞이면 배리어가 아직 게이트에
   * 닿지 않은 물건을 기다리며 timeout으로 죽는다(라인은 선입선출이다).
   */
  it('사이클이 아이템 순서대로 정확히 한 번씩 돈다', () => {
    const sequence = loadSequence();
    const gateOrder = sequence.steps.flatMap((s) =>
      s.kind === 'waitForCollision' && s.between.includes('pick_gate')
        ? [s.between.find((id) => id !== 'pick_gate')]
        : [],
    );
    expect(gateOrder).toEqual([...CPP_CYCLE_ITEMS]);
  });

  /**
   * ★ 사용자 보고: "컨베이 픽앤플레이스는 여전히 가다가 떨어뜨린다."
   *
   * 원인은 마찰이 아니라 **이송 호 길이**였다(실측 §CPP_MAX_TRANSPORT_ARC_RAD). 파지 후
   * joint1을 크게 돌리면 상자가 손가락에서 빠져나온다. 호 상한을 데이터로 고정해 둔다 —
   * 나중에 놓는 위치를 옮기려고 joint1 목표만 바꾸면 이 테스트가 먼저 막는다.
   */
  it('이송 호(joint1 변화량)가 파지가 버티는 범위 안에 있다', () => {
    const sequence = loadSequence();
    let currentJoint1: number | undefined;
    const arcs: number[] = [];
    for (const step of sequence.steps) {
      const target = jointTargetsOf(step)?.joint1;
      if (target === undefined) continue;
      if (currentJoint1 !== undefined) arcs.push(Math.abs(target - currentJoint1));
      currentJoint1 = target;
    }
    // 파지 상태의 이송 호만 문제지만, 어느 호가 파지 중인지는 아래 테스트가 보장하므로
    // 여기서는 home 복귀를 제외한 모든 회전이 안전 범위인지를 본다.
    const transportArcs = arcs.slice(0, -1);
    expect(transportArcs.length).toBeGreaterThanOrEqual(CPP_CYCLE_COUNT);
    for (const arc of transportArcs) {
      expect(arc).toBeLessThanOrEqual(CPP_MAX_TRANSPORT_ARC_RAD);
    }
  });

  /**
   * 파지 직후에는 **반드시 들어 올린 뒤** 회전한다. 든 상자의 바닥이 대기 중인 상자의
   * 윗면보다 낮으면 이송 중 라인 위의 물건을 들이받는다. 반대로 놓은 뒤 복귀할 때도
   * 먼저 들어야 한다 — 낮게 쓸고 지나가면 대기 중인 상자를 라인 밖으로 쳐낸다(실측).
   *
   * "들어 올린다"의 판정: joint2 목표가 작아지는 것 = 팔이 펴지며 손끝이 올라간다.
   */
  it('각 사이클이 파지→상승→이송, 놓기→상승→복귀 순서를 지킨다', () => {
    const steps = loadSequence().steps;
    const isRaise = (index: number): boolean => {
      if (steps[index]?.kind !== 'moveJoints') return false;
      const next = jointTargetsOf(steps[index])?.joint2;
      if (next === undefined) return false;
      for (let i = index - 1; i >= 0; i -= 1) {
        const before = jointTargetsOf(steps[i])?.joint2;
        if (before !== undefined) return next < before;
      }
      return false;
    };
    const isRotate = (index: number): boolean =>
      steps[index]?.kind === 'moveJoints' && jointTargetsOf(steps[index])?.joint1 !== undefined;

    const grips = steps.flatMap((s, i) => (s.kind === 'gripper' && s.state !== 'open' ? [i] : []));
    expect(grips).toHaveLength(CPP_CYCLE_COUNT);
    for (const index of grips) {
      expect(isRaise(index + 1), `step ${index + 1}: 파지 직후 상승이 없다`).toBe(true);
      expect(isRotate(index + 2), `step ${index + 2}: 상승 뒤 이송 회전이 없다`).toBe(true);
    }

    // 놓기(open) 뒤에도 상승이 먼저 — 마지막 open은 시퀀스 끝(home)이라 제외
    const releases = steps.flatMap((s, i) => (s.kind === 'gripper' && s.state === 'open' ? [i] : []));
    for (const index of releases.slice(1)) {
      expect(isRaise(index + 1), `step ${index + 1}: 놓은 뒤 낮게 복귀하면 대기 상자를 쳐낸다`).toBe(
        true,
      );
    }
  });
});

// ── 5. l-line-cell (ㄱ자 라인 + 로봇 3종) ───────────────────────────
//
// 이 씬이 고정하는 새 개념: **직각으로 이어진 벨트 2개**와 **손이 서로 다른 로봇 3대의
// 스테이션 분담**. 코너 이송은 겹침 중재(conveyor.ts의 claimed Set)에 기대는데, 그 동작이
// 아래 기하 조건 넷에 전부 의존한다 — 하나만 어겨도 상자가 코너에서 낙하하거나 시작점으로
// 순간이동한다. 브라우저 실측으로 하나하나 확인한 값이고, 여기서 데이터로 고정한다.

/** 하류 벨트가 상류 벨트의 끝을 넘겨야 하는 최소 길이 (m). 실측: 0.06이면 낙하, 0.12면 통과 */
const LL_MIN_OVERRUN_M = 0.12;
/** 라인의 로봇 3대 — 손 기구가 서로 다른 것이 이 예제의 핵심이다 */
const LL_ROBOTS = ['press', 'picker', 'palletizer'] as const;

describe('l-line-cell.scene.json', () => {
  const spec = mustValidateScene(lLineCellSceneJson, 'l-line-cell.scene.json');
  const ids = spec.entities.map((e) => e.id);

  it('validateScene 통과 + 벨트 2개 · 가이드 2개 · 로봇 3대', () => {
    expect(spec.name).toBe('l-line-cell');
    expect(spec.entities.filter(isRobotSpec).map((e) => e.id).sort()).toEqual([...LL_ROBOTS].sort());
    for (const id of ['belt_in', 'belt_out', 'rail_outer', 'rail_inner']) {
      expect(ids, `'${id}'가 없다`).toContain(id);
    }
    expect(spec.environment?.ground).toBe(true);
  });

  /**
   * ★ 로봇 3대의 URDF가 **서로 달라야** 이 예제가 의미를 갖는다. 같은 팔 3대를 세우면
   * "여러 종류의 로봇 손"이라는 이 씬의 존재 이유가 사라진다.
   */
  it('로봇 3대가 서로 다른 URDF(= 서로 다른 손)를 쓴다', () => {
    const urdfs = spec.entities.filter(isRobotSpec).map((r) => r.urdf);
    expect(new Set(urdfs).size).toBe(LL_ROBOTS.length);
  });

  /**
   * ★ 선언 순서가 **동작의 일부**다. ConveyorRegistry.tickAll()의 claimed 중재는
   * bindings 삽입 순서(= 선언 순서)로 승자를 정한다(core/conveyor.ts). 상류가 먼저여야
   * 코너 전까지 직진하고, 뒤집으면 상자 앞모서리가 하류에 닿는 순간 꺾여 몸통 대부분이
   * 아직 상류 위인 채로 하류 앞모서리 밖으로 떨어진다(실측).
   */
  it('belt_in이 belt_out보다 먼저 선언된다 (중재 승자 = 선언 순서)', () => {
    expect(ids.indexOf('belt_in')).toBeLessThan(ids.indexOf('belt_out'));
  });

  /**
   * ★ 하류 벨트가 상류 벨트의 끝을 충분히 넘겨야 한다. 상류의 접촉은 깔끔하게 끊기지 않고
   * 흔들리며 깜빡이고, 그동안 두 벨트가 번갈아 속도를 주입해 상자가 대각선으로 표류한다.
   * 하류가 그 표류를 받아낼 만큼 넓지 않으면 상자가 모서리 밖으로 나간다
   * (실측: 오버런 0.06 → 정지/낙하, 0.12 → 통과).
   */
  it('belt_out이 belt_in의 끝을 충분히 넘긴다 (코너 표류를 받아낸다)', () => {
    const beltIn = entityOf(spec, 'belt_in');
    const beltOut = entityOf(spec, 'belt_out');
    const inEndX = beltIn.transform.position[0] + boxHalfExtentsOf(beltIn)[0];
    const outEndX = beltOut.transform.position[0] + boxHalfExtentsOf(beltOut)[0];
    expect(outEndX - inEndX).toBeGreaterThanOrEqual(LL_MIN_OVERRUN_M);
  });

  /**
   * ★ 하류 벨트의 레인이 상류 레인을 **완전히** 덮어야 한다. 절반만 덮으면 상자가
   * 코너에서 덮이지 않은 쪽으로 빠져나간다(실측).
   */
  it('belt_out의 레인이 belt_in의 레인을 완전히 덮는다', () => {
    const beltIn = entityOf(spec, 'belt_in');
    const beltOut = entityOf(spec, 'belt_out');
    const inNearZ = beltIn.transform.position[2] - boxHalfExtentsOf(beltIn)[2];
    const inFarZ = beltIn.transform.position[2] + boxHalfExtentsOf(beltIn)[2];
    const outNearZ = beltOut.transform.position[2] - boxHalfExtentsOf(beltOut)[2];
    const outFarZ = beltOut.transform.position[2] + boxHalfExtentsOf(beltOut)[2];
    expect(outNearZ).toBeLessThanOrEqual(inNearZ);
    expect(outFarZ).toBeGreaterThanOrEqual(inFarZ);
  });

  /**
   * ★ 상류 벨트의 recycle은 반드시 꺼져 있어야 한다. 켜면 코너 지점이 이미 상류의 재순환
   * 조건(진행축 길이 초과)을 만족해, 상자가 코너에 닿는 순간 **시작점으로 순간이동**하고
   * 이어서 라인 밖으로 떨어진다 — 코너 자체가 성립하지 않는다(실측).
   */
  it('이어진 벨트는 재순환을 켜지 않는다 (코너 지점이 재순환 조건을 만족한다)', () => {
    for (const id of ['belt_in', 'belt_out']) {
      expect(entityOf(spec, id).conveyor?.recycle ?? false, `'${id}'`).toBe(false);
    }
  });

  /**
   * ★ 가이드 레일이 **상류 레인 위에 서면 안 된다**. 실제로 겪은 결함이다: rail_inner의
   * z 시작을 벨트 A의 +z 모서리보다 앞에 두었더니, 레일이 라인 한가운데를 막아 지나가는
   * 상자를 옆으로 쳐냈다(item_a·item_b가 z=-0.085로 밀려 벨트 밖 낙하).
   * 레일은 코너 **바깥쪽**에서 안내해야지 레인을 침범하면 안 된다.
   */
  it('가이드 레일이 belt_in 레인을 침범하지 않는다', () => {
    const beltIn = entityOf(spec, 'belt_in');
    const [bx, , bz] = beltIn.transform.position;
    const [bhx, , bhz] = boxHalfExtentsOf(beltIn);
    for (const id of ['rail_outer', 'rail_inner']) {
      const rail = entityOf(spec, id);
      const [rx, , rz] = rail.transform.position;
      const [rhx, , rhz] = boxHalfExtentsOf(rail);
      const overlapsX = Math.abs(rx - bx) < rhx + bhx;
      const overlapsZ = Math.abs(rz - bz) < rhz + bhz;
      expect(overlapsX && overlapsZ, `'${id}'이 belt_in 레인 위에 서 있다`).toBe(false);
    }
  });

  /** 가이드 레일은 실체다 — sensor면 상자가 그대로 통과해 안내를 못 한다 */
  it('가이드 레일은 sensor가 아닌 고정 ENV다', () => {
    for (const id of ['rail_outer', 'rail_inner']) {
      const rail = entityOf(spec, id);
      expect(rail.physics?.bodyType).toBe('fixed');
      for (const collider of collidersOf(rail)) {
        expect(collider.isSensor).not.toBe(true);
        expect(collider.group).toBe('ENV');
        expect(collider.collidesWith).toContain('OBJECT');
      }
    }
  });
});
