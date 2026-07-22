// core/scene-reset-sync.test.ts — SceneHandle.reset() 후 시각 동기화 회귀 테스트
//
// 리뷰 지적(determinism): reset()이 바디를 초기 pose로 텔레포트해도 RenderSync의
// prev 스냅샷이 갱신되지 않아, 다음 물리 tick의 commit() 전까지 apply(alpha<1)가
// 리셋 전 pose를 계속 그렸다 — "three.js는 물리의 거울" 불변식 위반 (CLAUDE.md §2.1).
// 수정: reset() 마지막에 sync.commit()으로 prev ← 텔레포트된 pose 갱신.
// (Engine.stop()도 lastAlpha를 ALPHA_LATEST(1)로 두어 idle 렌더가 최신 pose를 그린다.)
//
// 렌더 계층은 기록형 스텁 노드로 대체한다 — scene-loader는 three 없이 완결적이다.

import { beforeAll, describe, expect, it } from 'vitest';
import { initPhysics, RapierWorld } from './world';
import { RenderSync } from './sync';
import { SceneLoader } from './scene-loader';
import type { RenderSceneApi, VisualNode } from './scene-loader';
import type { SceneSpec, Vec3 } from '../schema/types';

// ── 테스트 상수 (매직넘버 금지 — CLAUDE.md §4) ──────────────────────

const TIMESTEP_HZ = 240;
const FALL_TICKS = TIMESTEP_HZ;            // 1초 낙하 — prev/cur가 초기 pose에서 확실히 벗어남
const POSITION_DECIMALS = 6;

const BOX_ID = 'box';
const BOX_HALF_M = 0.05;
const BOX_START_Y_M = 1.0;

/** prev 스냅샷만 그대로 표시하는 보간 계수 (버그 재현 경로: idle 프레임의 apply(0)) */
const ALPHA_PREV_ONLY = 0;

const SCENE_SPEC: SceneSpec = {
  name: 'reset-sync-regression',
  version: 1,
  gravity: [0, -9.81, 0],
  timestepHz: TIMESTEP_HZ,
  environment: { ground: true },
  entities: [
    {
      id: BOX_ID,
      type: 'object',
      transform: { position: [0, BOX_START_Y_M, 0] },
      visual: {
        kind: 'primitive',
        primitive: { kind: 'box', halfExtents: [BOX_HALF_M, BOX_HALF_M, BOX_HALF_M] },
      },
      physics: {
        bodyType: 'dynamic',
        colliders: [
          {
            shape: { kind: 'box', halfExtents: [BOX_HALF_M, BOX_HALF_M, BOX_HALF_M] },
            group: 'OBJECT',
            collidesWith: ['ENV', 'ROBOT', 'OBJECT'],
          },
        ],
      },
    },
  ],
};

// ── 기록형 시각 노드 스텁 ───────────────────────────────────────────

interface RecordingNode {
  /** apply()가 마지막으로 쓴 position (없으면 undefined) */
  lastPosition?: Vec3;
}

function makeRecordingRenderApi(): { api: RenderSceneApi; nodes: RecordingNode[] } {
  const nodes: RecordingNode[] = [];
  const makeNode = (): VisualNode => {
    const rec: RecordingNode = {};
    nodes.push(rec);
    const stub = {
      position: {
        set: (x: number, y: number, z: number) => {
          rec.lastPosition = [x, y, z];
        },
      },
      quaternion: { set: () => undefined },
    };
    return stub as unknown as VisualNode;
  };
  return {
    nodes,
    api: {
      addPrimitive: makeNode,
      addGround: makeNode,
      setPose: () => undefined,
      remove: () => undefined,
      loadRobot: () =>
        Promise.reject(new Error('이 테스트 씬에는 robot 엔티티가 없어야 합니다')),
    },
  };
}

beforeAll(async () => {
  await initPhysics();
});

// ── 회귀 테스트 ─────────────────────────────────────────────────────

describe('SceneHandle.reset() — 시각 prev 스냅샷 갱신 (CLAUDE.md §2.1)', () => {
  it('reset() 직후 apply(0)이 리셋 전 pose가 아니라 초기 pose를 그린다', async () => {
    const world = new RapierWorld(SCENE_SPEC.gravity, SCENE_SPEC.timestepHz);
    try {
      const sync = new RenderSync(world);
      const { api, nodes } = makeRecordingRenderApi();
      const handle = await new SceneLoader(world, api, sync).build(SCENE_SPEC);

      // Engine의 물리 tick 순서를 재현: commit() → world.step() (SIMULATION.md §5)
      for (let i = 0; i < FALL_TICKS; i += 1) {
        sync.commit();
        world.step();
      }

      // 낙하 후 apply(0) — prev 스냅샷이 그려지며 시작 높이보다 낮다 (정상 동작 확인)
      sync.apply(ALPHA_PREV_ONLY);
      const boxNode = nodes.find((n) => n.lastPosition !== undefined);
      if (!boxNode?.lastPosition) throw new Error('바인딩된 박스 노드가 apply를 받지 못했다');
      expect(boxNode.lastPosition[1]).toBeLessThan(BOX_START_Y_M);
      const fallenY = boxNode.lastPosition[1];

      // 리셋 후 apply(0): prev가 갱신됐다면 초기 pose가 그려진다.
      // (수정 전에는 fallenY가 그대로 남았다 — stale prev 보간)
      handle.reset();
      sync.apply(ALPHA_PREV_ONLY);
      expect(boxNode.lastPosition[1]).not.toBeCloseTo(fallenY, POSITION_DECIMALS);
      expect(boxNode.lastPosition[0]).toBeCloseTo(0, POSITION_DECIMALS);
      expect(boxNode.lastPosition[1]).toBeCloseTo(BOX_START_Y_M, POSITION_DECIMALS);
      expect(boxNode.lastPosition[2]).toBeCloseTo(0, POSITION_DECIMALS);
    } finally {
      world.free();
    }
  });
});
