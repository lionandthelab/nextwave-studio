// core/world-remove-contact.test.ts — removeEntity 후 잔여 'stop' 이벤트 회귀 테스트
//
// 리뷰 지적(physics): removeEntity가 collider 매핑을 즉시 지우면, Rapier가 제거된
// collider의 활성 접촉에 대해 다음 스텝에 발행하는 'stop' 이벤트가 매핑 가드에서
// 버려져, 살아남은 엔티티가 start/stop 짝을 잃고 접촉 상태가 '고착'된다.
// 수정: removeEntity는 핸들 매핑을 tombstone으로 옮기고, step()의 drain이 이를
// 참조해 제거 유발 stop 이벤트도 엔티티 쌍으로 번역한 뒤 비운다.
//
// rapier3d-compat는 WASM을 base64로 내장하므로 Node(vitest)에서 그대로 돈다.
// 모든 물리 API 사용 전 initPhysics()를 await한다 (CLAUDE.md §2.7).

import { beforeAll, describe, expect, it } from 'vitest';
import { initPhysics, RapierWorld } from './world';
import type { ContactEvent } from './types';
import type { ColliderSpec, Vec3 } from '../schema/types';

// ── 테스트 상수 (매직넘버 금지 — CLAUDE.md §4) ──────────────────────

const GRAVITY: Vec3 = [0, -9.81, 0];
const TIMESTEP_HZ = 240;
const SETTLE_TICKS = 2 * TIMESTEP_HZ;       // 접촉 시작까지 최대 2초
const POST_REMOVE_TICKS = TIMESTEP_HZ;      // 제거 후 stop 이벤트 대기 최대 1초

const BOX_HALF_M = 0.05;
const DROP_HEIGHT_M = 0.3;
const GROUND_HALF_EXTENTS_M: Vec3 = [2, 0.1, 2];
const GROUND_CENTER_Y_M = -GROUND_HALF_EXTENTS_M[1]; // 윗면 y=0

const ZONE_CENTER_Y_M = BOX_HALF_M;         // 정착한 박스를 계속 포함하는 감지 영역
const ZONE_HALF_EXTENTS_M: Vec3 = [0.2, 0.1, 0.2];

// ── 헬퍼 ────────────────────────────────────────────────────────────

const GROUND_COLLIDER: ColliderSpec = {
  shape: { kind: 'box', halfExtents: GROUND_HALF_EXTENTS_M },
  group: 'ENV',
  collidesWith: ['ROBOT', 'OBJECT'],
  emitEvents: true,
};

const BOX_COLLIDER: ColliderSpec = {
  shape: { kind: 'box', halfExtents: [BOX_HALF_M, BOX_HALF_M, BOX_HALF_M] },
  group: 'OBJECT',
  collidesWith: ['ENV', 'ROBOT', 'OBJECT', 'SENSOR_ZONE'],
  emitEvents: true,
};

const ZONE_COLLIDER: ColliderSpec = {
  shape: { kind: 'box', halfExtents: ZONE_HALF_EXTENTS_M },
  isSensor: true,
  group: 'SENSOR_ZONE',
  collidesWith: ['OBJECT'],
  emitEvents: true,
};

function pairOf(e: ContactEvent): string {
  return [e.a, e.b].sort().join(',');
}

/** 고정 바닥(윗면 y=0) + 낙하 박스. withZone이면 바닥 위 sensor 영역도 추가. */
function buildWorld(withZone: boolean): RapierWorld {
  const world = new RapierWorld(GRAVITY, TIMESTEP_HZ);
  const groundBody = world.createBody('ground', {
    bodyType: 'fixed',
    position: [0, GROUND_CENTER_Y_M, 0],
  });
  world.createCollider(groundBody, GROUND_COLLIDER, 'ground');

  if (withZone) {
    const zoneBody = world.createBody('zone', {
      bodyType: 'fixed',
      position: [0, ZONE_CENTER_Y_M, 0],
    });
    world.createCollider(zoneBody, ZONE_COLLIDER, 'zone');
  }

  const boxBody = world.createBody('box', {
    bodyType: 'dynamic',
    position: [0, DROP_HEIGHT_M, 0],
  });
  world.createCollider(boxBody, BOX_COLLIDER, 'box');
  return world;
}

/** phase='start'인 (kind, pair) 이벤트가 나올 때까지 스텝 (최대 SETTLE_TICKS) */
function stepUntilStart(world: RapierWorld, kind: ContactEvent['kind'], pair: string): void {
  for (let i = 0; i < SETTLE_TICKS; i += 1) {
    const hit = world
      .step()
      .some((e) => e.phase === 'start' && e.kind === kind && pairOf(e) === pair);
    if (hit) return;
  }
  throw new Error(`${SETTLE_TICKS}틱 안에 '${pair}' ${kind} start 이벤트가 없었다`);
}

function collectSteps(world: RapierWorld, ticks: number): ContactEvent[] {
  const events: ContactEvent[] = [];
  for (let i = 0; i < ticks; i += 1) events.push(...world.step());
  return events;
}

beforeAll(async () => {
  await initPhysics();
});

// ── 회귀 테스트 ─────────────────────────────────────────────────────

describe('RapierWorld.removeEntity — 접촉 중 제거 시 start/stop 짝 보존', () => {
  it('활성 contact 중 엔티티를 제거하면 다음 step()에서 해당 쌍의 stop 이벤트가 나온다', () => {
    const world = buildWorld(false);
    try {
      stepUntilStart(world, 'contact', 'box,ground');

      world.removeEntity('box');
      // 제거 직후에도 공개 조회는 즉시 무효 (tombstone은 drain 내부 전용)
      expect(world.bodiesOfEntity('box')).toEqual([]);

      const events = collectSteps(world, POST_REMOVE_TICKS);
      const stops = events.filter((e) => e.phase === 'stop' && pairOf(e) === 'box,ground');
      expect(stops.length).toBeGreaterThanOrEqual(1);
      for (const e of stops) expect(e.kind).toBe('contact');
    } finally {
      world.free();
    }
  });

  it('sensor 영역을 접촉(교차) 중 제거해도 stop 이벤트의 kind가 sensor로 유지된다', () => {
    const world = buildWorld(true);
    try {
      stepUntilStart(world, 'sensor', 'box,zone');

      world.removeEntity('zone');
      const events = collectSteps(world, POST_REMOVE_TICKS);
      const stops = events.filter((e) => e.phase === 'stop' && pairOf(e) === 'box,zone');
      expect(stops.length).toBeGreaterThanOrEqual(1);
      for (const e of stops) expect(e.kind).toBe('sensor');
    } finally {
      world.free();
    }
  });

  it('tombstone은 다음 step()의 drain까지만 유효하다 (그 뒤 스텝은 잔여 이벤트 없음)', () => {
    const world = buildWorld(false);
    try {
      stepUntilStart(world, 'contact', 'box,ground');
      world.removeEntity('box');

      // 제거 후 첫 스텝의 drain에서 stop을 소비하고 tombstone은 비워진다
      const first = world.step();
      expect(first.some((e) => e.phase === 'stop' && pairOf(e) === 'box,ground')).toBe(true);

      // 이후 스텝들에선 제거된 쌍의 이벤트가 더 나오지 않는다
      const later = collectSteps(world, POST_REMOVE_TICKS);
      expect(later.filter((e) => pairOf(e) === 'box,ground')).toEqual([]);
    } finally {
      world.free();
    }
  });
});
