// core/scene-loader.test.ts — 샘플 씬 검증 + 헤드리스 물리 통합 테스트
//
// rapier3d-compat는 WASM을 base64로 내장해 Node(vitest)에서 그대로 돈다.
// 렌더 계층은 no-op RenderSceneApi로 대체한다 — scene-loader는 three 없이 완결적이다.
// 모든 물리 API 사용 전 initPhysics()를 await한다 (CLAUDE.md §2.7).

import { beforeAll, describe, expect, it } from 'vitest';
import fallingBoxesSceneJson from '../assets/scenes/falling-boxes.scene.json';
import { validateScene } from '../schema/validate';
import type { RobotSpec, SceneSpec } from '../schema/types';
import type { ContactEvent } from './types';
import { initPhysics, RapierWorld } from './world';
import { RenderSync } from './sync';
import { GROUND_ENTITY_ID, SceneLoader } from './scene-loader';
import type { RenderSceneApi, VisualNode } from './scene-loader';

// ── 테스트 상수 (매직넘버 금지 — CLAUDE.md §4) ──────────────────────

const SIM_SECONDS = 3;
const RESET_POSITION_DECIMALS = 6;
const EXPECTED_DYNAMIC_ENTITY_COUNT = 5;
const EXPECTED_EMIT_EVENTS_MIN_COUNT = 2;

// ── 헬퍼 ────────────────────────────────────────────────────────────

/** 시각 노드 스텁 — RenderSync.apply가 만지는 position/quaternion만 갖춘 no-op */
function stubNode(): VisualNode {
  const stub = {
    position: { set: () => undefined },
    quaternion: { set: () => undefined },
  };
  return stub as unknown as VisualNode;
}

interface CountingRenderApi extends RenderSceneApi {
  readonly counts: { addPrimitive: number; addGround: number; setPose: number; remove: number };
}

function makeRenderApi(): CountingRenderApi {
  const counts = { addPrimitive: 0, addGround: 0, setPose: 0, remove: 0 };
  return {
    counts,
    addPrimitive: () => {
      counts.addPrimitive += 1;
      return stubNode();
    },
    addGround: () => {
      counts.addGround += 1;
      return stubNode();
    },
    setPose: () => {
      counts.setPose += 1;
    },
    remove: () => {
      counts.remove += 1;
    },
  };
}

function loadValidatedSpec(): SceneSpec {
  const result = validateScene(fallingBoxesSceneJson);
  if (!result.ok) throw new Error(`샘플 씬 검증 실패:\n${result.errors.join('\n')}`);
  return result.value;
}

function stepSeconds(world: RapierWorld, spec: SceneSpec, seconds: number): ContactEvent[] {
  const totalSteps = Math.round(seconds * spec.timestepHz);
  const events: ContactEvent[] = [];
  for (let i = 0; i < totalSteps; i += 1) {
    events.push(...world.step());
  }
  return events;
}

function dynamicEntitiesOf(spec: SceneSpec): SceneSpec['entities'] {
  return spec.entities.filter((e) => e.physics?.bodyType === 'dynamic');
}

/** 엔티티의 유일한 바디 id를 얻는다 (noUncheckedIndexedAccess 대응 명시적 가드) */
function soleBodyOf(world: RapierWorld, entityId: string): number {
  const bodies = world.bodiesOfEntity(entityId);
  expect(bodies).toHaveLength(1);
  const bodyId = bodies[0];
  if (bodyId === undefined) throw new Error(`엔티티 '${entityId}'의 바디가 없습니다`);
  return bodyId;
}

beforeAll(async () => {
  await initPhysics();
});

// ── 1. 샘플 씬 스키마 검증 ──────────────────────────────────────────

describe('falling-boxes.scene.json', () => {
  it('validateScene을 통과한다 (ok: true)', () => {
    const result = validateScene(fallingBoxesSceneJson);
    if (!result.ok) throw new Error(result.errors.join('\n'));
    expect(result.ok).toBe(true);
  });

  it('요구 구성을 갖춘다: 동적 5개(OBJECT) + 고정 벽 1개, emitEvents ≥ 2', () => {
    const spec = loadValidatedSpec();
    const dynamics = dynamicEntitiesOf(spec);
    expect(dynamics).toHaveLength(EXPECTED_DYNAMIC_ENTITY_COUNT);
    for (const entity of dynamics) {
      for (const collider of entity.physics?.colliders ?? []) {
        expect(collider.group).toBe('OBJECT');
        expect(collider.collidesWith).toEqual(['ENV', 'ROBOT', 'OBJECT']);
      }
      const y = entity.transform.position[1];
      expect(y).toBeGreaterThanOrEqual(0.5);
      expect(y).toBeLessThanOrEqual(1.5);
    }

    const emitters = dynamics.filter((e) =>
      (e.physics?.colliders ?? []).some((c) => c.emitEvents === true),
    );
    expect(emitters.length).toBeGreaterThanOrEqual(EXPECTED_EMIT_EVENTS_MIN_COUNT);

    const walls = spec.entities.filter(
      (e) => e.type === 'static' && e.physics?.bodyType === 'fixed',
    );
    expect(walls).toHaveLength(1);
    expect(spec.environment?.ground).toBe(true);
  });
});

// ── 2. 헤드리스 물리 통합 (SceneLoader + RapierWorld, 렌더 no-op) ───

describe('SceneLoader (headless integration)', () => {
  it('3초 시뮬 후 모든 동적 바디가 바닥 위(y > 0, 시작 높이 미만)에 있고 접촉 이벤트가 발생한다', () => {
    const spec = loadValidatedSpec();
    const world = new RapierWorld(spec.gravity, spec.timestepHz);
    try {
      const sync = new RenderSync(world);
      const renderApi = makeRenderApi();
      const handle = new SceneLoader(world, renderApi, sync).build(spec);

      // 핸들: ground 예약 id + 모든 스펙 엔티티 id
      expect(handle.entityIds).toContain(GROUND_ENTITY_ID);
      for (const entity of spec.entities) {
        expect(handle.entityIds).toContain(entity.id);
      }
      expect(renderApi.counts.addGround).toBe(1);
      expect(renderApi.counts.addPrimitive).toBe(spec.entities.length);

      const events = stepSeconds(world, spec, SIM_SECONDS);

      for (const entity of dynamicEntitiesOf(spec)) {
        const bodyId = soleBodyOf(world, entity.id);
        const y = world.getPose(bodyId).position[1];
        const startY = entity.transform.position[1];
        expect(y, `${entity.id}: 바닥을 관통하지 않아야 한다`).toBeGreaterThan(0);
        expect(y, `${entity.id}: 시작 높이 아래로 낙하해야 한다`).toBeLessThan(startY);
      }

      // 충돌 이벤트는 EventQueue 경유로 실제 감지되어야 한다 (CLAUDE.md §2.4, DoD §8)
      expect(events.some((e) => e.phase === 'start' && e.kind === 'contact')).toBe(true);
    } finally {
      world.free();
    }
  });

  it('reset()은 모든 바디를 초기 스펙 트랜스폼으로 되돌린다 (결정론적 재생)', () => {
    const spec = loadValidatedSpec();
    const world = new RapierWorld(spec.gravity, spec.timestepHz);
    try {
      const sync = new RenderSync(world);
      const handle = new SceneLoader(world, makeRenderApi(), sync).build(spec);

      stepSeconds(world, spec, 1);
      handle.reset();

      for (const entity of spec.entities) {
        if (!entity.physics) continue;
        const bodyId = soleBodyOf(world, entity.id);
        const pose = world.getPose(bodyId);
        const initial = entity.transform.position;
        expect(pose.position[0]).toBeCloseTo(initial[0], RESET_POSITION_DECIMALS);
        expect(pose.position[1]).toBeCloseTo(initial[1], RESET_POSITION_DECIMALS);
        expect(pose.position[2]).toBeCloseTo(initial[2], RESET_POSITION_DECIMALS);
      }

      // 리셋 직후 1초 재시뮬 — 속도가 0으로 초기화됐다면 다시 자유낙하해 정착 경로를 밟는다
      stepSeconds(world, spec, 1);
      for (const entity of dynamicEntitiesOf(spec)) {
        const bodyId = soleBodyOf(world, entity.id);
        expect(world.getPose(bodyId).position[1]).toBeGreaterThan(0);
      }
    } finally {
      world.free();
    }
  });

  it('dispose()는 물리 바디와 시각 노드를 전부 해제한다', () => {
    const spec = loadValidatedSpec();
    const world = new RapierWorld(spec.gravity, spec.timestepHz);
    try {
      const sync = new RenderSync(world);
      const renderApi = makeRenderApi();
      const handle = new SceneLoader(world, renderApi, sync).build(spec);
      const createdNodeCount = renderApi.counts.addPrimitive + renderApi.counts.addGround;

      handle.dispose();

      for (const id of handle.entityIds) {
        expect(world.bodiesOfEntity(id)).toEqual([]);
      }
      expect(renderApi.counts.remove).toBe(createdNodeCount);
      expect(world.step()).toEqual([]); // 빈 월드 스텝은 무해
    } finally {
      world.free();
    }
  });

  it('robot 엔티티는 명확한 Phase 3 오류를 던진다', () => {
    const robotEntity: RobotSpec = {
      id: 'arm',
      type: 'robot',
      transform: { position: [0, 0, 0] },
      visual: { kind: 'urdf', ref: 'assets/arm/arm.urdf' },
      urdf: 'assets/arm/arm.urdf',
      controller: 'sequence',
    };
    const spec: SceneSpec = {
      name: 'robot-not-yet',
      version: 1,
      gravity: [0, -9.81, 0],
      timestepHz: 240,
      entities: [robotEntity],
    };
    const world = new RapierWorld(spec.gravity, spec.timestepHz);
    try {
      const loader = new SceneLoader(world, makeRenderApi(), new RenderSync(world));
      expect(() => loader.build(spec)).toThrow(/Phase 3/);
    } finally {
      world.free();
    }
  });
});
