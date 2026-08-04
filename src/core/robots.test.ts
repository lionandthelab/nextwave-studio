// core/robots.test.ts — RobotBinding / RobotRegistry 순수 로직 단위 테스트
//
// 물리/렌더 없이 완결적으로 검증한다 (CLAUDE.md §4 "부수효과 격리"):
// - RobotFkView는 관절값에서 결정론적으로 링크 pose를 유도하는 수제 목(mock)
// - PhysicsWorld는 setKinematicPose 호출을 기록하는 목 — Rapier 비의존

import { afterEach, describe, expect, it, vi } from 'vitest';
import { RobotBinding, RobotRegistry } from './robots';
import type { RobotBindingOptions } from './robots';
import type { BodyId, ColliderId, ContactEvent, EntityId, PhysicsWorld, Pose } from './types';
import type { Vec3 } from '../schema/types';
import type { JointInfo, LinkColliderDef, RobotFkView } from './robot-types';

// ── 테스트 상수 (매직넘버 금지 — CLAUDE.md §4) ──────────────────────

const ROBOT_ID = 'arm';
const TEST_TIMESTEP_HZ = 240;

/** j1: revolute, rad */
const J1_LIMITS_RAD: [number, number] = [-1, 1];
/** j2: prismatic, m */
const J2_LIMITS_M: [number, number] = [0, 0.03];

const L1_BODY_ID: BodyId = 11;
const L2_BODY_ID: BodyId = 22;
const GHOST_BODY_ID: BodyId = 33;

/** 목 FK에서 L2 pose의 고정 z 오프셋 (관절값과 무관한 상수 성분) */
const L2_BASE_Z_M = 1;

const IDENTITY_QUAT: readonly number[] = [0, 0, 0, 1];

const HOME_SHOULDER_RAD = 0.5;

// ── 목 RobotFkView ──────────────────────────────────────────────────
// 2관절(j1 revolute [-1,1], j2 prismatic [0,0.03]) · 2링크(L1, L2).
// readLinkPoses는 현재 관절값에서 결정론적으로 pose를 유도한다:
//   L1.position = [j1, 0, 0], L2.position = [j1, j2, L2_BASE_Z_M].

class MockFkView implements RobotFkView {
  readonly joints: readonly JointInfo[] = [
    { name: 'j1', type: 'revolute', limits: [...J1_LIMITS_RAD], initial: 0 },
    { name: 'j2', type: 'prismatic', limits: [...J2_LIMITS_M], initial: 0 },
  ];
  readonly linkColliders: ReadonlyMap<string, readonly LinkColliderDef[]> = new Map();

  /** setJointValues 호출 기록 (각 호출의 values 복사본) */
  readonly setJointValuesCalls: Record<string, number>[] = [];

  private current: Record<string, number> = { j1: 0, j2: 0 };

  setJointValues(values: Readonly<Record<string, number>>): void {
    this.setJointValuesCalls.push({ ...values });
    this.current = { ...this.current, ...values };
  }

  readLinkPoses(): ReadonlyMap<string, Pose> {
    const j1 = this.current['j1'] ?? 0;
    const j2 = this.current['j2'] ?? 0;
    return new Map<string, Pose>([
      ['L1', { position: [j1, 0, 0], rotation: [0, 0, 0, 1] }],
      ['L2', { position: [j1, j2, L2_BASE_Z_M], rotation: [0, 0, 0, 1] }],
    ]);
  }
}

// ── 목 PhysicsWorld ─────────────────────────────────────────────────

interface KinematicPoseCall {
  bodyId: BodyId;
  pose: Pose;
}

class MockPhysicsWorld implements PhysicsWorld {
  readonly fixedDtSec = 1 / TEST_TIMESTEP_HZ;

  /** setKinematicPose 호출 기록 (pose 깊은 복사 — 이후 변형과 무관) */
  readonly kinematicPoseCalls: KinematicPoseCall[] = [];

  setKinematicPose(bodyId: BodyId, pose: Pose): void {
    this.kinematicPoseCalls.push({
      bodyId,
      pose: { position: [...pose.position], rotation: [...pose.rotation] },
    });
  }

  /** 자기 접촉 스위치 — 이 테스트에서는 기록만 하고 동작에 영향 없다 */
  readonly selfContactCalls: { entityId: EntityId; enabled: boolean }[] = [];
  setSelfContactEnabled(entityId: EntityId, enabled: boolean): void {
    this.selfContactCalls.push({ entityId, enabled });
  }

  // 이하 이 테스트에서 쓰지 않는 계약 — 호출되면 테스트 실패로 드러나게 던진다
  createBody(): BodyId {
    throw new Error('MockPhysicsWorld.createBody: 이 테스트에서 호출되면 안 됩니다');
  }
  createCollider(): ColliderId {
    throw new Error('MockPhysicsWorld.createCollider: 이 테스트에서 호출되면 안 됩니다');
  }
  removeEntity(): void {
    throw new Error('MockPhysicsWorld.removeEntity: 이 테스트에서 호출되면 안 됩니다');
  }

  /** 되감기 전용 — 이 테스트들은 되감기를 호출하지 않으므로 no-op이면 충분하다 */
  clearContactState(): void {}
  getPose(): Pose {
    throw new Error('MockPhysicsWorld.getPose: 이 테스트에서 호출되면 안 됩니다');
  }
  teleport(): void {
    throw new Error('MockPhysicsWorld.teleport: 이 테스트에서 호출되면 안 됩니다');
  }
  step(): ContactEvent[] {
    throw new Error('MockPhysicsWorld.step: 이 테스트에서 호출되면 안 됩니다');
  }
  entityOfCollider(): EntityId | undefined {
    return undefined;
  }
  bodiesOfEntity(): readonly BodyId[] {
    return [];
  }
  // 컨베이어 표면 구동 계약 (core/types.ts) — 이 목은 접촉/속도를 다루지 않는다
  dynamicBodiesTouching(): readonly BodyId[] {
    return [];
  }
  dynamicBodies(): readonly BodyId[] {
    return [];
  }
  getLinearVelocity(): Vec3 {
    return [0, 0, 0];
  }
  setLinearVelocity(): void {}
  clear(): void {}
  free(): void {}
}

// ── 픽스처 ──────────────────────────────────────────────────────────

type FixtureOverrides = Partial<
  Pick<RobotBindingOptions, 'entityId' | 'linkBodies' | 'jointMap' | 'jointLimitOverrides' | 'home'>
>;

function makeFixture(overrides: FixtureOverrides = {}): {
  fk: MockFkView;
  world: MockPhysicsWorld;
  binding: RobotBinding;
} {
  const fk = new MockFkView();
  const world = new MockPhysicsWorld();
  const binding = new RobotBinding({
    entityId: ROBOT_ID,
    fk,
    world,
    linkBodies: new Map<string, BodyId>([
      ['L1', L1_BODY_ID],
      ['L2', L2_BODY_ID],
    ]),
    ...overrides,
  });
  return { fk, world, binding };
}

/** console.warn을 침묵시키고 스파이를 돌려준다 (클램프 경고 검증용) */
function spyWarn(): ReturnType<typeof vi.spyOn> {
  return vi.spyOn(console, 'warn').mockImplementation(() => undefined);
}

afterEach(() => {
  vi.restoreAllMocks();
});

// ── RobotBinding ────────────────────────────────────────────────────

describe('RobotBinding — 생성·home', () => {
  it('관절 상태를 fk.joints initial로 초기화한다', () => {
    const { binding } = makeFixture();
    expect(binding.readJoints()).toEqual({ j1: 0, j2: 0 });
  });

  it('home을 생성 시 적용한다 — jointMap 논리명(shoulder → j1) 간접 참조 포함', () => {
    const { binding } = makeFixture({
      jointMap: { shoulder: 'j1' },
      home: { shoulder: HOME_SHOULDER_RAD },
    });
    expect(binding.readJoints()).toEqual({ j1: HOME_SHOULDER_RAD, j2: 0 });
  });

  it('applyHome()이 생성 시점 포즈(initial + home)를 복원한다', () => {
    const { binding } = makeFixture({
      jointMap: { shoulder: 'j1' },
      home: { shoulder: HOME_SHOULDER_RAD },
    });
    binding.setJoint('j1', 0.9);
    binding.setJoint('j2', 0.01);
    binding.applyHome();
    expect(binding.readJoints()).toEqual({ j1: HOME_SHOULDER_RAD, j2: 0 });
  });
});

describe('RobotBinding — limits 클램프', () => {
  it('URDF limits 밖 값을 양방향으로 클램프하고 한국어 경고를 낸다', () => {
    const warn = spyWarn();
    const { binding } = makeFixture();

    binding.setJoint('j1', 2);
    expect(binding.readJoints()['j1']).toBe(J1_LIMITS_RAD[1]);

    binding.setJoint('j1', -2);
    expect(binding.readJoints()['j1']).toBe(J1_LIMITS_RAD[0]);

    expect(warn).toHaveBeenCalledTimes(2);
    expect(String(warn.mock.calls[0]?.[0])).toContain('클램프');
  });

  it('override limits가 더 좁으면 override가 이긴다 (교집합)', () => {
    const warn = spyWarn();
    const narrower: [number, number] = [-0.5, 0.8];
    const { binding } = makeFixture({ jointLimitOverrides: { j1: narrower } });

    binding.setJoint('j1', 2);
    expect(binding.readJoints()['j1']).toBe(narrower[1]);

    binding.setJoint('j1', -2);
    expect(binding.readJoints()['j1']).toBe(narrower[0]);

    expect(warn).toHaveBeenCalledTimes(2);
  });

  it('override가 URDF보다 넓으면 URDF limits가 이긴다 (교집합)', () => {
    spyWarn();
    const wider: [number, number] = [-1, 1];
    const { binding } = makeFixture({ jointLimitOverrides: { j2: wider } });

    binding.setJoint('j2', 0.05);
    expect(binding.readJoints()['j2']).toBe(J2_LIMITS_M[1]);

    binding.setJoint('j2', -0.5);
    expect(binding.readJoints()['j2']).toBe(J2_LIMITS_M[0]);
  });

  it('joints getter는 유효 limits(URDF ∩ override)를 노출한다', () => {
    const narrower: [number, number] = [-0.5, 0.8];
    const { binding } = makeFixture({ jointLimitOverrides: { j1: narrower } });

    const j1 = binding.joints.find((j) => j.name === 'j1');
    const j2 = binding.joints.find((j) => j.name === 'j2');
    expect(j1?.limits).toEqual(narrower);
    expect(j2?.limits).toEqual(J2_LIMITS_M);
    expect(binding.joints.map((j) => j.name)).toEqual(['j1', 'j2']);
  });

  it('limits 범위 안 값은 경고 없이 그대로 반영된다', () => {
    const warn = spyWarn();
    const { binding } = makeFixture();
    binding.setJoint('j1', 0.25);
    expect(binding.readJoints()['j1']).toBe(0.25);
    expect(warn).not.toHaveBeenCalled();
  });
});

describe('RobotBinding — 관절 해석·조회', () => {
  it('알 수 없는 관절은 사용 가능한 관절 목록과 함께 던진다', () => {
    const { binding } = makeFixture({ jointMap: { shoulder: 'j1' } });
    expect(() => binding.setJoint('elbow', 0)).toThrowError(/j1, j2/);
    expect(() => binding.resolveJoint('elbow')).toThrowError(/관절 'elbow'/);
    expect(() => binding.readJoints(['elbow'])).toThrowError(/사용 가능한 관절/);
  });

  it('readJoints()는 라이브 참조가 아닌 복사본을 돌려준다', () => {
    const { binding } = makeFixture();
    const snapshot = binding.readJoints();
    snapshot['j1'] = 999;
    expect(binding.readJoints()['j1']).toBe(0);
  });

  it('readJoints(names)는 논리명/원명을 해석하고 요청한 이름 그대로 키를 쓴다', () => {
    const { binding } = makeFixture({
      jointMap: { shoulder: 'j1' },
      home: { shoulder: HOME_SHOULDER_RAD },
    });
    expect(binding.readJoints(['shoulder', 'j2'])).toEqual({
      shoulder: HOME_SHOULDER_RAD,
      j2: 0,
    });
    expect(binding.readJoints(['j1'])).toEqual({ j1: HOME_SHOULDER_RAD });
  });

  it('setJoints는 여러 관절을 일괄 설정한다 (논리명 혼용 가능)', () => {
    const { binding } = makeFixture({ jointMap: { shoulder: 'j1' } });
    binding.setJoints({ shoulder: 0.3, j2: 0.02 });
    expect(binding.readJoints()).toEqual({ j1: 0.3, j2: 0.02 });
  });
});

describe('RobotBinding — tick (kinematic 구동)', () => {
  it('linkBodies 엔트리의 pose를 Map 순서대로 world에 push하고, 최신 관절값을 반영한다', () => {
    const { fk, world, binding } = makeFixture();
    binding.setJoint('j1', 0.25);
    binding.setJoint('j2', 0.02);

    binding.tick();

    // FK에 최신 관절 상태가 전달됐다
    expect(fk.setJointValuesCalls).toHaveLength(1);
    expect(fk.setJointValuesCalls[0]).toEqual({ j1: 0.25, j2: 0.02 });

    // linkBodies 삽입 순서(L1 → L2) 그대로, 관절값에서 유도된 pose가 push됐다
    expect(world.kinematicPoseCalls.map((c) => c.bodyId)).toEqual([L1_BODY_ID, L2_BODY_ID]);
    expect(world.kinematicPoseCalls[0]?.pose).toEqual({
      position: [0.25, 0, 0],
      rotation: IDENTITY_QUAT,
    });
    expect(world.kinematicPoseCalls[1]?.pose).toEqual({
      position: [0.25, 0.02, L2_BASE_Z_M],
      rotation: IDENTITY_QUAT,
    });
  });

  it('관절이 안 바뀌어도 매 tick 전체를 다시 push한다 (kinematic 연속 목표)', () => {
    const { world, binding } = makeFixture();
    binding.tick();
    binding.tick();
    expect(world.kinematicPoseCalls).toHaveLength(4); // 2링크 × 2틱
  });

  it('FK pose에 없는 링크(linkBodies에만 있는 항목)는 건너뛴다', () => {
    const { world, binding } = makeFixture({
      linkBodies: new Map<string, BodyId>([
        ['L1', L1_BODY_ID],
        ['ghost', GHOST_BODY_ID], // MockFkView.readLinkPoses에 없는 링크
        ['L2', L2_BODY_ID],
      ]),
    });
    binding.tick();
    expect(world.kinematicPoseCalls.map((c) => c.bodyId)).toEqual([L1_BODY_ID, L2_BODY_ID]);
  });
});

// ── RobotRegistry ───────────────────────────────────────────────────

describe('RobotRegistry', () => {
  it('ids()는 등록(삽입) 순서를 유지한다', () => {
    const registry = new RobotRegistry();
    registry.add(makeFixture({ entityId: 'arm-b' }).binding);
    registry.add(makeFixture({ entityId: 'arm-a' }).binding);
    expect(registry.ids()).toEqual(['arm-b', 'arm-a']);
  });

  it('tickAll()은 등록된 모든 바인딩을 tick한다 (엔진 preStep 훅 대상)', () => {
    const registry = new RobotRegistry();
    const first = makeFixture({ entityId: 'arm-1' });
    const second = makeFixture({ entityId: 'arm-2' });
    registry.add(first.binding);
    registry.add(second.binding);

    registry.tickAll();

    expect(first.fk.setJointValuesCalls).toHaveLength(1);
    expect(second.fk.setJointValuesCalls).toHaveLength(1);
    expect(first.world.kinematicPoseCalls.map((c) => c.bodyId)).toEqual([L1_BODY_ID, L2_BODY_ID]);
    expect(second.world.kinematicPoseCalls.map((c) => c.bodyId)).toEqual([L1_BODY_ID, L2_BODY_ID]);
  });

  it('get()은 미등록 id에 한국어 오류를 던지고, has()는 false를 돌려준다', () => {
    const registry = new RobotRegistry();
    registry.add(makeFixture({ entityId: 'arm-1' }).binding);

    expect(registry.has('arm-1')).toBe(true);
    expect(registry.has('nope')).toBe(false);
    expect(() => registry.get('nope')).toThrowError(/찾을 수 없습니다/);
    expect(registry.get('arm-1').entityId).toBe('arm-1');
  });

  it('같은 entityId를 중복 등록하면 던진다', () => {
    const registry = new RobotRegistry();
    registry.add(makeFixture({ entityId: 'arm-1' }).binding);
    expect(() => registry.add(makeFixture({ entityId: 'arm-1' }).binding)).toThrowError(
      /이미 등록/,
    );
  });
});
