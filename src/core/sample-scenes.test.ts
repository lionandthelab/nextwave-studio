// core/sample-scenes.test.ts — 제공 예제 **전체**에 걸리는 데이터 계약
//
// phase6-scenes.test.ts가 씬 3종을 개별적으로 자세히 보는 반면, 이 파일은 SCENE_REGISTRY와
// 같은 목록을 한 번에 훑으며 **모든 예제가 공통으로 만족해야 하는 규칙**을 건다.
// 새 예제를 추가하면 아래 SAMPLES에 한 줄 append하는 것만으로 같은 그물이 걸린다.
//
// ── 이 그물이 잡는 실제 결함 ─────────────────────────────────────────
//
// 1. **보이지만 존재하지 않는 장애물** (사용자 보고: "선반에 부딪히는데 충돌 표시 안 뜨고
//    관통해"). pick-and-place의 drop_zone은 화면엔 단단한 상자였지만 sensor였고
//    collidesWith가 [OBJECT]뿐이라, 로봇과는 **쌍조차 성립하지 않아** 이벤트가 0건이었다.
//    → 로봇이 있는 씬의 비-sensor 정적 collider는 ROBOT과 쌍이 성립하고 emitEvents여야 한다.
//
// 2. **지하 스폰** (사용자 보고: "물체가 바닥 아래 지하로 위치돼 사라진다"). 편집 경로는
//    core/ground-clamp가 막지만, 샘플 데이터가 애초에 지하에서 시작하면 사용자는 같은
//    증상을 첫 화면에서 본다. → 모든 엔티티의 최저점이 바닥 위에 있어야 한다.
//
// 3. **한쪽만 선언한 쌍 필터**. Rapier 쌍 필터는 양방향이다 — sensor 쪽이 OBJECT를
//    나열해도 OBJECT 쪽이 SENSOR_ZONE을 나열하지 않으면 감지되지 않는다.
//
// 물리 거동은 scripts/gate-browser.mjs의 씬별 게이트가 본다 (npm run gate:samples).

import { describe, expect, it } from 'vitest';
import armAndBoxesSceneJson from '../assets/scenes/arm-and-boxes.scene.json';
import collisionTestbedSceneJson from '../assets/scenes/collision-testbed.scene.json';
import conveyorPickPlaceSceneJson from '../assets/scenes/conveyor-pick-place.scene.json';
import fallingBoxesSceneJson from '../assets/scenes/falling-boxes.scene.json';
import lLineCellSceneJson from '../assets/scenes/l-line-cell.scene.json';
import lLineCellSequenceJson from '../assets/sequences/l-line-cell.sequence.json';
import obstacleAvoidanceSceneJson from '../assets/scenes/obstacle-avoidance.scene.json';
import pickAndPlaceSceneJson from '../assets/scenes/pick-and-place.scene.json';
import twoArmsCollisionSceneJson from '../assets/scenes/two-arms-collision.scene.json';
import armTouchBoxSequenceJson from '../assets/sequences/arm-touch-box.sequence.json';
import conveyorPickPlaceSequenceJson from '../assets/sequences/conveyor-pick-place.sequence.json';
import obstacleAvoidanceSequenceJson from '../assets/sequences/obstacle-avoidance.sequence.json';
import pickAndPlaceSequenceJson from '../assets/sequences/pick-and-place.sequence.json';
import twoArmsCollisionSequenceJson from '../assets/sequences/two-arms-collision.sequence.json';
import { validateScene, validateSequence } from '../schema/validate';
import { DETECTION_ZONE_TAG, isRobotSpec } from '../schema/types';
import type { ConveyorSpec, EntitySpec, Quat, SceneSpec } from '../schema/types';
import { GROUND_CLAMP_EPS_M, entityDropBelowOrigin, minGroundedY } from './ground-clamp';
import { beltCoordsOf, beltForward, beltLateral, extentAlong } from './conveyor';

// ── 예제 목록 (main.ts SCENE_REGISTRY와 같은 구성) ──────────────────

interface Sample {
  readonly name: string;
  readonly scene: unknown;
  readonly sequence?: unknown;
}

const SAMPLES: readonly Sample[] = [
  { name: 'falling-boxes', scene: fallingBoxesSceneJson },
  { name: 'arm-and-boxes', scene: armAndBoxesSceneJson, sequence: armTouchBoxSequenceJson },
  { name: 'pick-and-place', scene: pickAndPlaceSceneJson, sequence: pickAndPlaceSequenceJson },
  {
    name: 'obstacle-avoidance',
    scene: obstacleAvoidanceSceneJson,
    sequence: obstacleAvoidanceSequenceJson,
  },
  { name: 'collision-testbed', scene: collisionTestbedSceneJson },
  {
    name: 'conveyor-pick-place',
    scene: conveyorPickPlaceSceneJson,
    sequence: conveyorPickPlaceSequenceJson,
  },
  {
    name: 'two-arms-collision',
    scene: twoArmsCollisionSceneJson,
    sequence: twoArmsCollisionSequenceJson,
  },
  { name: 'l-line-cell', scene: lLineCellSceneJson, sequence: lLineCellSequenceJson },
];

/** main.ts SCENE_REGISTRY의 항목 수 — 예제를 늘리면 여기도 함께 늘어야 한다 */
const EXPECTED_SAMPLE_COUNT = 8;

const IDENTITY_QUAT: Quat = [0, 0, 0, 1];
/** "벨트 위에 얹혀 있다" 판정 여유 (m) — 첫 스텝에 내려앉는 미세 간격을 허용한다 */
const ON_BELT_TOLERANCE_M = 0.02;


function mustValidate(sample: Sample): SceneSpec {
  const result = validateScene(sample.scene);
  if (!result.ok) throw new Error(`${sample.name} 검증 실패:\n${result.errors.join('\n')}`);
  return result.value;
}

// ── 0. 목록 자체 ────────────────────────────────────────────────────

describe('예제 목록', () => {
  it('제공 예제는 8종이고 이름이 중복되지 않는다', () => {
    expect(SAMPLES).toHaveLength(EXPECTED_SAMPLE_COUNT);
    expect(new Set(SAMPLES.map((s) => s.name)).size).toBe(EXPECTED_SAMPLE_COUNT);
  });

  it('예제 이름과 씬 name 필드가 일치한다 (?scene= 라우팅의 전제)', () => {
    for (const sample of SAMPLES) {
      expect(mustValidate(sample).name).toBe(sample.name);
    }
  });
});

// ── 1. 스키마 검증 ──────────────────────────────────────────────────

describe.each(SAMPLES.map((s) => [s.name, s] as const))('%s', (_name, sample) => {
  const spec = mustValidate(sample);

  it('씬이 validateScene을 통과한다', () => {
    expect(spec.entities.length).toBeGreaterThan(0);
    expect(spec.timestepHz).toBeGreaterThan(0);
  });

  it('엔티티 id가 씬 안에서 유일하다', () => {
    const ids = spec.entities.map((e) => e.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  if (sample.sequence !== undefined) {
    it('시퀀스가 이 씬 대비 validateSequence를 통과한다 (참조 정합)', () => {
      const result = validateSequence(sample.sequence, spec);
      if (!result.ok) throw new Error(result.errors.join('\n'));
      expect(result.value.steps.length).toBeGreaterThan(0);
      // 시퀀스의 기본 로봇이 씬에 실재해야 한다
      expect(spec.entities.some((e) => e.id === result.value.robot && isRobotSpec(e))).toBe(true);
    });

    it('waitForCollision.between이 씬의 실재 엔티티를 가리킨다', () => {
      const result = validateSequence(sample.sequence, spec);
      if (!result.ok) throw new Error(result.errors.join('\n'));
      const ids = new Set(spec.entities.map((e) => e.id));
      for (const step of result.value.steps) {
        if (step.kind !== 'waitForCollision') continue;
        for (const id of step.between) {
          expect(ids.has(id), `waitForCollision가 없는 엔티티 '${id}'를 기다린다`).toBe(true);
        }
      }
    });
  }

  // ── 2. 지하 스폰 금지 (사용자 보고 #1의 데이터 쪽 짝) ────────────
  it('바닥이 있는 씬이면 어떤 엔티티도 지하에서 시작하지 않는다', () => {
    if (spec.environment?.ground !== true) return;
    for (const entity of spec.entities) {
      const minY = minGroundedY(entity);
      expect(
        entity.transform.position[1],
        `'${entity.id}'가 바닥 아래에서 시작한다 (y=${entity.transform.position[1]}, 하한 ${minY})`,
      ).toBeGreaterThanOrEqual(minY - GROUND_CLAMP_EPS_M);
    }
  });

  // ── 3. 보이지만 존재하지 않는 장애물 금지 (사용자 보고 #3) ────────
  //
  // 핵심은 **의도의 선언**이다. "이건 sensor다"를 태그로 명시하지 않으면, 정적 엔티티를
  // sensor로 만들어 이 검사를 조용히 빠져나갈 수 있다 — 그게 정확히 원래 결함의 모양이다
  // (단단해 보이는 상자가 아무도 모르게 통과 가능했다). 태그가 없으면 "단단한 장애물"로
  // 간주하고, 단단한 장애물의 조건(비-sensor · ROBOT 쌍 성립 · emitEvents)을 강제한다.
  it('감지 존이 아닌 정적 엔티티는 실제로 단단하고, 로봇과 쌍이 성립한다', () => {
    const hasRobot = spec.entities.some(isRobotSpec);
    for (const entity of spec.entities) {
      if (entity.physics?.bodyType !== 'fixed') continue;
      if (entity.tags?.includes(DETECTION_ZONE_TAG) === true) continue; // 의도된 감지 존
      for (const collider of entity.physics.colliders) {
        expect(
          collider.isSensor === true,
          `'${entity.id}'가 sensor인데 '${DETECTION_ZONE_TAG}' 태그가 없다 — ` +
            '단단해 보이는데 모든 것이 통과하는 유령 장애물이 된다. ' +
            '감지가 목적이면 태그로 선언하고, 장애물이면 sensor를 끄라',
        ).toBe(false);
        if (!hasRobot) continue; // 로봇이 없으면 로봇-장애물 쌍이라는 개념 자체가 없다
        expect(
          collider.collidesWith,
          `'${entity.id}'는 단단한 정적 장애물인데 ROBOT과 쌍이 성립하지 않는다 ` +
            '— 로봇이 지나가도 이벤트가 0건이 된다',
        ).toContain('ROBOT');
        expect(
          collider.emitEvents,
          `'${entity.id}'가 emitEvents를 켜지 않아 닿아도 로그에 남지 않는다`,
        ).toBe(true);
      }
    }
  });

  /**
   * ★ 장식용 벨트 금지. 컨베이어를 선언해 놓고 그 위에 아무것도 없으면 예제는 "아무 일도
   * 일어나지 않는 화면"이 된다 — 실제로 이 씬을 처음 만들 때 item_b를 벨트 시작점보다
   * 앞(x=-0.02)에 두어, 그 사물이 바닥에 떨어진 채 한 번도 실려 가지 않았다.
   *
   * 판정에는 core/conveyor.ts의 **검증된 순수 기하**를 그대로 쓴다 — 테스트가 자체
   * 기하 계산을 다시 구현하면 구현과 테스트가 같이 틀릴 수 있다.
   *
   * **다중 벨트 라인 예외**: ㄱ자 라인처럼 벨트가 이어지면 하류 벨트 위에서 시작하는
   * 사물이 없는 것이 정상이다 — 상류에서 코너를 돌아 실려 온다. 그래서 판정은
   * "그 위에서 시작한다" 또는 "다른 벨트의 레인과 겹쳐 공급받는다" 둘 중 하나다.
   * 어느 쪽도 아니면 그 벨트는 아무것도 실을 수 없고, 그때가 진짜 장식이다.
   */
  it('컨베이어가 있으면 그 위에서 시작하거나 상류에서 공급받는 동적 사물이 있다', () => {
    const belts = spec.entities.filter((e) => e.conveyor !== undefined);
    if (belts.length === 0) return;

    /** 벨트의 월드 AABB (XZ 평면) — 레인 겹침 판정용 */
    const laneOf = (belt: EntitySpec): { minX: number; maxX: number; minZ: number; maxZ: number } | null => {
      const shape = belt.physics?.colliders[0]?.shape;
      if (shape === undefined || shape.kind !== 'box') return null;
      const [cx, , cz] = belt.transform.position;
      const [hx, , hz] = shape.halfExtents;
      return { minX: cx - hx, maxX: cx + hx, minZ: cz - hz, maxZ: cz + hz };
    };
    const lanesOverlap = (a: EntitySpec, b: EntitySpec): boolean => {
      const la = laneOf(a);
      const lb = laneOf(b);
      if (la === null || lb === null) return false;
      return la.minX < lb.maxX && lb.minX < la.maxX && la.minZ < lb.maxZ && lb.minZ < la.maxZ;
    };
    const dynamics = spec.entities.filter((e) => e.physics?.bodyType === 'dynamic');

    for (const belt of belts) {
      const shape = belt.physics?.colliders[0]?.shape;
      if (shape === undefined || shape.kind !== 'box') {
        throw new Error(`'${belt.id}' 벨트 collider가 box가 아니다 (validateScene이 막았어야 한다)`);
      }
      const rotation = belt.transform.rotation ?? IDENTITY_QUAT;
      const forward = beltForward(belt.conveyor as ConveyorSpec, rotation);
      expect(forward, `'${belt.id}'의 벨트 방향이 퇴화했다`).not.toBeNull();
      if (forward === null) continue;
      const lateral = beltLateral(forward);
      const halfLength = extentAlong(shape.halfExtents, forward, rotation);
      const halfWidth = extentAlong(shape.halfExtents, lateral, rotation);
      const topY = belt.transform.position[1] + extentAlong(shape.halfExtents, [0, 1, 0], rotation);

      const carried = dynamics.filter((item) => {
        const c = beltCoordsOf(item.transform.position, belt.transform.position, forward, lateral);
        const onDeck = Math.abs(c.along) <= halfLength && Math.abs(c.lateral) <= halfWidth;
        const drop = entityDropBelowOrigin(item);
        const bottom = item.transform.position[1] - drop;
        // 벨트 상면에 얹혀 있어야 한다 (약간의 여유는 첫 스텝에 내려앉는다)
        return onDeck && bottom >= topY - ON_BELT_TOLERANCE_M && bottom <= topY + ON_BELT_TOLERANCE_M;
      });
      // 상류 공급: 다른 벨트와 레인이 겹치면 그쪽에서 실려 온다 (코너 이송)
      const fedByUpstream = belts.some((other) => other.id !== belt.id && lanesOverlap(belt, other));
      expect(
        carried.length > 0 || fedByUpstream,
        `컨베이어 '${belt.id}'는 위에서 시작하는 사물도 없고 상류 벨트와 레인도 겹치지 않는다 ` +
          `— 아무것도 실을 수 없는 장식 벨트다. ` +
          `동적 사물: ${dynamics.map((d) => `${d.id}@${JSON.stringify(d.transform.position)}`).join(', ')}`,
      ).toBe(true);
    }
  });

  it('감지 존 태그가 붙은 엔티티는 실제로 sensor이고 SENSOR_ZONE 그룹이다', () => {
    for (const entity of spec.entities) {
      if (entity.tags?.includes(DETECTION_ZONE_TAG) !== true) continue;
      const colliders = entity.physics?.colliders ?? [];
      expect(colliders.length, `'${entity.id}'에 collider가 없다`).toBeGreaterThan(0);
      for (const collider of colliders) {
        expect(collider.isSensor, `'${entity.id}'는 감지 존 태그인데 sensor가 아니다`).toBe(true);
        expect(collider.group).toBe('SENSOR_ZONE');
      }
    }
  });

  // ── 4. 양방향 쌍 필터 (sensor ↔ OBJECT) ─────────────────────────
  it('sensor 존이 있으면 OBJECT 측 필터에도 SENSOR_ZONE이 있다 (쌍 필터는 양방향)', () => {
    const sensorIds = spec.entities
      .filter((e) => e.physics?.colliders.some((c) => c.isSensor === true))
      .map((e) => e.id);
    if (sensorIds.length === 0) return;
    const dynamicObjects = spec.entities.filter((e) => e.physics?.bodyType === 'dynamic');
    // 감지 대상이 하나도 없는 sensor 존은 이 규칙의 대상이 아니다(순수 시각 마커)
    const detectable = dynamicObjects.filter((e) =>
      e.physics?.colliders.some((c) => c.collidesWith.includes('SENSOR_ZONE')),
    );
    expect(
      detectable.length,
      `sensor 존(${sensorIds.join(', ')})이 있는데 SENSOR_ZONE을 나열한 동적 사물이 없다 ` +
        '— 쌍이 성립하지 않아 아무것도 감지되지 않는다',
    ).toBeGreaterThan(0);
  });

  // ── 5. 감지 존은 sensor여야 한다 (역할 혼동 방지) ────────────────
  it('SENSOR_ZONE 그룹 collider는 반드시 isSensor다 (튕기는 감지 영역 금지)', () => {
    for (const entity of spec.entities) {
      for (const collider of entity.physics?.colliders ?? []) {
        if (collider.group !== 'SENSOR_ZONE') continue;
        expect(collider.isSensor, `'${entity.id}'의 SENSOR_ZONE collider가 sensor가 아니다`).toBe(
          true,
        );
      }
    }
  });
});
