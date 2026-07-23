// schema/validate.test.ts — 런타임 검증 단위 테스트 (규범: docs/DATA_MODEL.md §8/§9)
// 픽스처는 DATA_MODEL.md §9.1(SceneSpec) / §9.2(ControlSequence) 예시를 그대로 사용한다.

import { describe, expect, it } from 'vitest';
import type { ControlSequence, EntitySpec, RobotSpec, SceneSpec } from './types';
import { isRobotSpec } from './types';
import { validateScene, validateSequence, type ValidationResult } from './validate';

// ── 헬퍼 ────────────────────────────────────────────────────────────

function expectOk<T>(result: ValidationResult<T>): T {
  if (!result.ok) {
    throw new Error(`검증 통과를 기대했지만 실패: ${result.errors.join(' | ')}`);
  }
  return result.value;
}

function expectFail<T>(result: ValidationResult<T>): string[] {
  if (result.ok) throw new Error('검증 실패를 기대했지만 통과했습니다');
  return result.errors;
}

/** 오류 목록을 한 문자열로 합쳐 substring 단언을 쉽게 한다 */
function joined<T>(result: ValidationResult<T>): string {
  return expectFail(result).join('\n');
}

// ── 픽스처: DATA_MODEL.md §9.1 SceneSpec (로봇 + 박스 + 바닥) ────────

const armRobot: RobotSpec = {
  id: 'arm',
  type: 'robot',
  transform: { position: [0, 0, 0] },
  visual: { kind: 'urdf', ref: 'assets/arm/arm.urdf' },
  urdf: 'assets/arm/arm.urdf',
  urdfPackages: { arm_description: 'assets/arm' },
  home: { joint1: 0.0, joint2: -0.6, joint3: 1.2 },
  controller: 'sequence',
  linkColliders: 'fromVisual',
  selfCollision: false,
};

const boxA: EntitySpec = {
  id: 'box_a',
  type: 'object',
  transform: { position: [0.4, 0.05, 0.0] },
  visual: {
    kind: 'primitive',
    primitive: { kind: 'box', halfExtents: [0.05, 0.05, 0.05] },
    color: '#c0392b',
  },
  physics: {
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
  },
};

const exampleScene: SceneSpec = {
  name: 'collision-testbed',
  version: 1,
  gravity: [0, -9.81, 0],
  timestepHz: 240,
  environment: { ground: true, skyColor: '#1b1e23' },
  camera: { position: [1.5, 1.2, 1.5], target: [0, 0.4, 0] },
  entities: [armRobot, boxA],
};

// ── 픽스처: DATA_MODEL.md §9.2 ControlSequence (픽 동작 + 충돌 대기) ─

const exampleSequence: ControlSequence = {
  id: 'reach-and-touch',
  robot: 'arm',
  loop: false,
  steps: [
    {
      kind: 'moveJoints',
      targets: { joint2: 0.2, joint3: 0.6 },
      durationSec: 2.0,
      easing: 'easeInOut',
    },
    { kind: 'waitForCollision', between: ['arm', 'box_a'], timeoutSec: 5.0 },
    { kind: 'gripper', state: 'close', durationSec: 0.5 },
    { kind: 'wait', durationSec: 1.0 },
    { kind: 'moveJoints', targets: { joint2: -0.6, joint3: 1.2 }, durationSec: 2.0 },
  ],
};

// ── validateScene: 유효 입력 ────────────────────────────────────────

describe('validateScene — 유효한 씬', () => {
  it('최소 유효 씬이 통과한다', () => {
    const value = expectOk(
      validateScene({
        name: 'minimal',
        version: 1,
        gravity: [0, -9.81, 0],
        timestepHz: 240,
        entities: [],
      }),
    );
    expect(value.name).toBe('minimal');
    expect(value.timestepHz).toBe(240);
  });

  it('DATA_MODEL §9.1 예시 씬이 통과한다', () => {
    const value = expectOk(validateScene(exampleScene));
    expect(value.name).toBe('collision-testbed');
    expect(value.entities).toHaveLength(2);
  });

  it('파싱된 출력에 로봇 전용 필드(urdf/home/controller)가 보존된다', () => {
    const value = expectOk(validateScene(exampleScene));
    const arm = value.entities.find((e) => e.id === 'arm');
    expect(arm).toBeDefined();
    if (!arm || !isRobotSpec(arm)) throw new Error('arm이 RobotSpec으로 파싱되지 않았습니다');
    expect(arm.urdf).toBe('assets/arm/arm.urdf');
    expect(arm.controller).toBe('sequence');
    expect(arm.home).toEqual({ joint1: 0.0, joint2: -0.6, joint3: 1.2 });
  });

  it('fixed 바디의 trimesh collider는 허용된다', () => {
    const fixedTrimesh = structuredClone(boxA);
    fixedTrimesh.id = 'terrain';
    fixedTrimesh.physics!.bodyType = 'fixed';
    fixedTrimesh.physics!.colliders[0]!.shape = { kind: 'trimesh', ref: 'assets/terrain.obj' };
    expectOk(validateScene({ ...exampleScene, entities: [armRobot, fixedTrimesh] }));
  });
});

// ── validateScene: 구조 오류 ────────────────────────────────────────

describe('validateScene — 구조 오류', () => {
  it('객체가 아닌 입력은 (root) 경로의 오류를 낸다', () => {
    const text = joined(validateScene(null));
    expect(text).toContain('(root)');
  });

  it('version이 1이 아니면 실패한다', () => {
    const text = joined(validateScene({ ...exampleScene, version: 2 }));
    expect(text).toContain('version');
    expect(text).toContain('1');
  });

  it('timestepHz가 0이면 실패한다', () => {
    const text = joined(validateScene({ ...exampleScene, timestepHz: 0 }));
    expect(text).toContain('timestepHz: timestepHz는 0보다 커야 합니다');
  });

  it('timestepHz가 무한대면 실패한다', () => {
    const text = joined(
      validateScene({ ...exampleScene, timestepHz: Number.POSITIVE_INFINITY }),
    );
    expect(text).toContain('timestepHz: timestepHz는 유한한 수여야 합니다');
  });

  it('gravity 성분이 무한대면 실패한다', () => {
    const text = joined(
      validateScene({ ...exampleScene, gravity: [0, Number.NEGATIVE_INFINITY, 0] }),
    );
    expect(text).toContain('gravity[1]');
    expect(text).toContain('유한한 수');
  });

  it('알 수 없는 entity type은 실패한다', () => {
    const text = joined(
      validateScene({ ...exampleScene, entities: [{ ...boxA, type: 'vehicle' }] }),
    );
    expect(text).toContain('entities[0].type');
    expect(text).toContain('허용');
  });

  it('알 수 없는 collider group은 실패한다', () => {
    const badBox = structuredClone(boxA);
    const text = joined(
      validateScene({
        ...exampleScene,
        entities: [
          armRobot,
          {
            ...badBox,
            physics: {
              bodyType: 'dynamic',
              colliders: [{ ...badBox.physics!.colliders[0]!, group: 'LASER' }],
            },
          },
        ],
      }),
    );
    expect(text).toContain('entities[1].physics.colliders[0].group');
    expect(text).toContain('허용');
  });
});

// ── validateScene: 교차 필드 규칙 ───────────────────────────────────

describe('validateScene — 교차 필드 규칙', () => {
  it('엔티티 id가 중복되면 실패한다', () => {
    const text = joined(
      validateScene({ ...exampleScene, entities: [boxA, structuredClone(boxA)] }),
    );
    expect(text).toContain('entities[1].id');
    expect(text).toContain('중복');
  });

  it('dynamic 바디의 trimesh collider는 실패한다', () => {
    const badBox = structuredClone(boxA);
    badBox.physics!.colliders[0]!.shape = { kind: 'trimesh', ref: 'assets/box.obj' };
    const text = joined(
      validateScene({ ...exampleScene, entities: [armRobot, badBox] }),
    );
    expect(text).toContain('entities[1].physics.colliders[0].shape');
    expect(text).toContain('trimesh');
    expect(text).toContain("'fixed'");
  });

  it('kinematicPosition 바디의 trimesh collider도 실패한다', () => {
    const badBox = structuredClone(boxA);
    badBox.physics!.bodyType = 'kinematicPosition';
    badBox.physics!.colliders[0]!.shape = { kind: 'trimesh', ref: 'assets/box.obj' };
    const text = joined(
      validateScene({ ...exampleScene, entities: [armRobot, badBox] }),
    );
    expect(text).toContain('entities[1].physics.colliders[0].shape');
    expect(text).toContain('trimesh');
  });

  it('emitEvents=true인데 collidesWith가 비어 있으면 실패한다', () => {
    const badBox = structuredClone(boxA);
    badBox.physics!.colliders[0]!.collidesWith = [];
    const text = joined(
      validateScene({ ...exampleScene, entities: [armRobot, badBox] }),
    );
    expect(text).toContain('entities[1].physics.colliders[0].collidesWith');
    expect(text).toContain('emitEvents');
  });

  it('robot 엔티티에 urdf가 없으면 실패한다', () => {
    const { urdf: _urdf, ...robotWithoutUrdf } = armRobot;
    const text = joined(
      validateScene({ ...exampleScene, entities: [robotWithoutUrdf, boxA] }),
    );
    expect(text).toContain('entities[0].urdf');
    expect(text).toContain('누락');
  });

  it('robot 엔티티에 controller가 없으면 실패한다', () => {
    const { controller: _controller, ...robotWithoutController } = armRobot;
    const text = joined(
      validateScene({ ...exampleScene, entities: [robotWithoutController, boxA] }),
    );
    expect(text).toContain('entities[0].controller');
    expect(text).toContain('누락');
  });

  it("robot 엔티티의 visual.kind가 'urdf'가 아니면 실패한다", () => {
    const text = joined(
      validateScene({
        ...exampleScene,
        entities: [{ ...armRobot, visual: { kind: 'primitive' } }, boxA],
      }),
    );
    expect(text).toContain('entities[0].visual.kind');
    expect(text).toContain('urdf');
  });
});

// ── validateSequence: 유효 입력 ─────────────────────────────────────

describe('validateSequence — 유효한 시퀀스', () => {
  it('DATA_MODEL §9.2 예시 시퀀스가 (씬 없이) 통과한다', () => {
    const value = expectOk(validateSequence(exampleSequence));
    expect(value.id).toBe('reach-and-touch');
    expect(value.steps).toHaveLength(5);
  });

  it('DATA_MODEL §9.2 예시 시퀀스가 §9.1 씬과 함께 통과한다', () => {
    const value = expectOk(validateSequence(exampleSequence, exampleScene));
    expect(value.robot).toBe('arm');
  });

  it('label을 참조하는 goto는 통과한다', () => {
    expectOk(
      validateSequence({
        id: 'loop-demo',
        robot: 'arm',
        steps: [
          { kind: 'label', name: 'start' },
          { kind: 'wait', durationSec: 0.5 },
          { kind: 'goto', label: 'start', times: 3 },
        ],
      }),
    );
  });

  it('관절 정보가 전혀 없는 로봇은 임의의 관절 이름을 허용한다', () => {
    const { home: _home, ...armWithoutMaps } = armRobot;
    const sceneWithoutMaps: SceneSpec = {
      ...exampleScene,
      entities: [armWithoutMaps, boxA],
    };
    expectOk(
      validateSequence(
        {
          id: 'free-joints',
          robot: 'arm',
          steps: [{ kind: 'moveJoints', targets: { anything: 1.0 }, durationSec: 1.0 }],
        },
        sceneWithoutMaps,
      ),
    );
  });
});

// ── validateSequence: 구조·교차 규칙 오류 ───────────────────────────

describe('validateSequence — 시퀀스 자체 규칙', () => {
  it('steps가 비어 있으면 실패한다', () => {
    const text = joined(validateSequence({ id: 'empty', robot: 'arm', steps: [] }));
    expect(text).toContain('steps');
    expect(text).toContain('최소 1개');
  });

  it('존재하지 않는 label로의 goto는 실패한다', () => {
    const text = joined(
      validateSequence({
        id: 'bad-goto',
        robot: 'arm',
        steps: [
          { kind: 'label', name: 'start' },
          { kind: 'goto', label: 'nowhere' },
        ],
      }),
    );
    expect(text).toContain('steps[1].label');
    expect(text).toContain("'nowhere'");
    expect(text).toContain('존재하지 않습니다');
  });

  it('durationSec가 음수면 실패한다', () => {
    const text = joined(
      validateSequence({
        id: 'neg-duration',
        robot: 'arm',
        steps: [{ kind: 'wait', durationSec: -1 }],
      }),
    );
    expect(text).toContain('steps[0].durationSec: durationSec는 0 이상이어야 합니다');
  });

  it('moveJoints의 durationSec가 음수면 실패한다', () => {
    const text = joined(
      validateSequence({
        id: 'neg-move',
        robot: 'arm',
        steps: [{ kind: 'moveJoints', targets: { joint1: 0 }, durationSec: -0.5 }],
      }),
    );
    expect(text).toContain('steps[0].durationSec: durationSec는 0 이상이어야 합니다');
  });

  it('timeoutSec가 0이면 실패한다', () => {
    const text = joined(
      validateSequence({
        id: 'zero-timeout',
        robot: 'arm',
        steps: [{ kind: 'waitForCollision', between: ['arm', 'box_a'], timeoutSec: 0 }],
      }),
    );
    expect(text).toContain('steps[0].timeoutSec: timeoutSec는 0보다 커야 합니다');
  });

  it('goto times가 음수면 실패한다', () => {
    const text = joined(
      validateSequence({
        id: 'neg-times',
        robot: 'arm',
        steps: [
          { kind: 'label', name: 'start' },
          { kind: 'goto', label: 'start', times: -3.5 },
        ],
      }),
    );
    expect(text).toContain('steps[1].times');
    expect(text).toContain('times는 0 이상이어야 합니다');
  });

  it('goto times가 소수면 실패한다', () => {
    const text = joined(
      validateSequence({
        id: 'frac-times',
        robot: 'arm',
        steps: [
          { kind: 'label', name: 'start' },
          { kind: 'goto', label: 'start', times: 1.5 },
        ],
      }),
    );
    expect(text).toContain('steps[1].times');
    expect(text).toContain('times는 정수여야 합니다');
  });

  it('goto times 0(점프 안 함)은 허용한다', () => {
    expectOk(
      validateSequence({
        id: 'zero-times',
        robot: 'arm',
        steps: [
          { kind: 'label', name: 'start' },
          { kind: 'goto', label: 'start', times: 0 },
        ],
      }),
    );
  });

  it('gripper state 숫자값이 0..1 밖이면 실패한다 (런타임 clamp에 가려지는 데이터 오류)', () => {
    const over = joined(
      validateSequence({
        id: 'gripper-over',
        robot: 'arm',
        steps: [{ kind: 'gripper', state: 42 }],
      }),
    );
    expect(over).toContain('steps[0].state');
    expect(over).toContain('0 이상 1 이하');

    const under = joined(
      validateSequence({
        id: 'gripper-under',
        robot: 'arm',
        steps: [{ kind: 'gripper', state: -0.1 }],
      }),
    );
    expect(under).toContain('steps[0].state');
    expect(under).toContain('0 이상 1 이하');
  });

  it('gripper state 경계값 0/1과 중간값은 허용한다', () => {
    for (const state of [0, 0.5, 1]) {
      expectOk(
        validateSequence({
          id: `gripper-${state}`,
          robot: 'arm',
          steps: [{ kind: 'gripper', state }],
        }),
      );
    }
  });

  it('알 수 없는 step kind는 실패한다', () => {
    const text = joined(
      validateSequence({
        id: 'bad-kind',
        robot: 'arm',
        steps: [{ kind: 'teleport' }],
      }),
    );
    expect(text).toContain('steps[0].kind');
    expect(text).toContain('허용');
  });
});

describe('validateSequence — 씬 참조 무결성 (scene 제공 시)', () => {
  it('시퀀스의 기본 robot이 씬에 없으면 실패한다', () => {
    const text = joined(
      validateSequence({ ...exampleSequence, robot: 'ghost' }, exampleScene),
    );
    expect(text).toContain("robot: 씬에 'ghost' 엔티티가 존재하지 않습니다");
  });

  it('시퀀스의 robot이 로봇 엔티티가 아니면 실패한다', () => {
    const text = joined(
      validateSequence(
        {
          id: 'not-a-robot',
          robot: 'box_a',
          steps: [{ kind: 'wait', durationSec: 1 }],
        },
        exampleScene,
      ),
    );
    expect(text).toContain('robot:');
    expect(text).toContain('로봇 엔티티가 아닙니다');
  });

  it('step의 robot override가 씬에 없으면 실패한다', () => {
    const text = joined(
      validateSequence(
        {
          id: 'bad-override',
          robot: 'arm',
          steps: [{ kind: 'gripper', robot: 'ghost', state: 'open' }],
        },
        exampleScene,
      ),
    );
    expect(text).toContain("steps[0].robot: 씬에 'ghost' 엔티티가 존재하지 않습니다");
  });

  it('waitForCollision.between의 id가 씬에 없으면 실패한다', () => {
    const text = joined(
      validateSequence(
        {
          id: 'bad-between',
          robot: 'arm',
          steps: [{ kind: 'waitForCollision', between: ['arm', 'ghost_box'] }],
        },
        exampleScene,
      ),
    );
    expect(text).toContain("steps[0].between[1]: 씬에 'ghost_box' 엔티티가 존재하지 않습니다");
  });

  it('moveJoints의 관절 이름이 로봇의 home/jointMap/jointLimits에 없으면 실패한다', () => {
    const text = joined(
      validateSequence(
        {
          id: 'bad-joint',
          robot: 'arm',
          steps: [{ kind: 'moveJoints', targets: { joint9: 1.0 }, durationSec: 1.0 }],
        },
        exampleScene,
      ),
    );
    expect(text).toContain('steps[0].targets.joint9');
    expect(text).toContain('정의되지 않은 관절');
    expect(text).toContain('joint1');
  });

  it('setJoints의 관절 이름도 동일하게 검사한다', () => {
    const text = joined(
      validateSequence(
        {
          id: 'bad-setjoints',
          robot: 'arm',
          steps: [{ kind: 'setJoints', targets: { joint9: 0.0 } }],
        },
        exampleScene,
      ),
    );
    expect(text).toContain('steps[0].targets.joint9');
    expect(text).toContain('정의되지 않은 관절');
  });

  it('jointMap 키에 있는 관절 이름은 허용한다', () => {
    const armWithJointMap: RobotSpec = { ...armRobot, jointMap: { shoulder: 'joint1' } };
    const scene: SceneSpec = { ...exampleScene, entities: [armWithJointMap, boxA] };
    expectOk(
      validateSequence(
        {
          id: 'logical-joint',
          robot: 'arm',
          steps: [{ kind: 'setJoints', targets: { shoulder: 0.4 } }],
        },
        scene,
      ),
    );
  });
});
