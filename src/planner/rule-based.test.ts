// planner/rule-based.test.ts — 결정론적 규칙 기반 플래너 단위 테스트 (LLM 없이 — PLANNER.md §7).
// 생성 sequence는 실제 씬(arm-and-boxes)에 대해 validateSequence를 통과해야 한다.

import { describe, expect, it } from 'vitest';
import { validateScene, validateSequence } from '../schema';
import type { ControlSequence, SceneSpec } from '../schema';
import armAndBoxesJson from '../assets/scenes/arm-and-boxes.scene.json';
import { buildContext, type SceneContext } from './scene-context';
import { generateRuleBased, RULE_BASED_SEQUENCE_ID } from './adapters/rule-based';
import type { PlannerResult } from './llm-adapter';

function loadScene(json: unknown): SceneSpec {
  const result = validateScene(json);
  if (!result.ok) throw new Error(`픽스처 씬 검증 실패: ${result.errors.join(' | ')}`);
  return result.value;
}

const armScene = loadScene(armAndBoxesJson);
const armCtx = buildContext(armScene);

function expectSequence(result: PlannerResult): ControlSequence {
  if (result.type !== 'sequence') {
    throw new Error(`sequence를 기대했지만 ${result.type}: ${JSON.stringify(result)}`);
  }
  const validation = validateSequence(result.sequence, armScene);
  if (!validation.ok) {
    throw new Error(`생성 시퀀스가 검증 실패: ${validation.errors.join(' | ')}`);
  }
  return result.sequence;
}

describe('generateRuleBased — 집기(touch)', () => {
  it('알려진 대상을 집으면 유효한 시퀀스를 만든다 (open→approach→nudge→waitForCollision→close→wait→home)', () => {
    const result = generateRuleBased('box_a를 집어', armCtx);
    const seq = expectSequence(result);
    expect(seq.id).toBe(RULE_BASED_SEQUENCE_ID);
    expect(seq.robot).toBe('arm');
    // 2단 접근(approach moveJoints → nudge setJoints)은 결정론적 접촉 계약 (arm-touch-box와 동일)
    expect(seq.steps.map((s) => s.kind)).toEqual([
      'gripper',
      'moveJoints',
      'setJoints',
      'waitForCollision',
      'gripper',
      'wait',
      'moveJoints',
    ]);
  });

  it('waitForCollision 쌍이 [robot, entity]로 정확하다', () => {
    const result = generateRuleBased('box_a를 집어', armCtx);
    const seq = expectSequence(result);
    const wfc = seq.steps.find((s) => s.kind === 'waitForCollision');
    expect(wfc).toMatchObject({ between: ['arm', 'box_a'], timeoutSec: 6 });
  });

  it('reach 자세는 joint1 = -atan2(z,x) 규약을 따르고 관절 이름이 유효하다', () => {
    const result = generateRuleBased('box_a를 집어', armCtx);
    const seq = expectSequence(result);
    const reach = seq.steps[1];
    if (reach?.kind !== 'moveJoints') throw new Error('두 번째 step은 moveJoints여야 함');
    // box_a[0.35,·,0.15] → -atan2(0.15,0.35) ≈ -0.405
    expect(reach.targets['joint1']).toBeCloseTo(-0.405, 3);
    expect(Object.keys(reach.targets).sort()).toEqual(['joint1', 'joint2', 'joint3', 'joint5']);
  });

  it('assumptions에 대상 해석과 휴리스틱 경고를 노출한다', () => {
    const result = generateRuleBased('box_a를 집어', armCtx);
    if (result.type !== 'sequence') throw new Error('sequence 기대');
    expect(result.assumptions).toEqual(['box_a를 대상으로 해석', '도달 자세는 휴리스틱(정확한 IK 아님)']);
  });

  it('영어 명령(pick/grab/touch)도 인식한다', () => {
    expect(generateRuleBased('grab box_b', armCtx).type).toBe('sequence');
    expect(generateRuleBased('touch box_a', armCtx).type).toBe('sequence');
  });

  it('관절 목표를 컨텍스트 limits로 클램프한다', () => {
    // joint3 limits를 [-1,1]로 좁힌 컨텍스트 — 휴리스틱 approach elbow(mid 1.414)가 1.0으로 클램프
    const tightCtx: SceneContext = {
      robots: [
        {
          id: 'arm',
          joints: [
            { name: 'joint1', type: 'revolute', limits: [-3.14, 3.14], current: 0 },
            { name: 'joint2', type: 'revolute', limits: [-2, 2], current: 0 },
            { name: 'joint3', type: 'revolute', limits: [-1, 1], current: 0 },
            { name: 'joint5', type: 'revolute', limits: [-1.9, 1.9], current: 0 },
          ],
          gripper: { name: 'gripper', open: 0.03, close: 0 },
          homePose: { joint1: 0, joint2: -0.6, joint3: 1.2, joint5: -0.6 },
        },
      ],
      objects: [{ id: 'box_a', type: 'object', position: [0.35, 0.03, 0.15] }],
      capabilities: { stepKinds: [] },
      frame: 'y-up, meters, radians',
      directions: { left: '-X', right: '+X', up: '+Y', forward: '+Z', back: '-Z' },
    };
    const result = generateRuleBased('box_a를 집어', tightCtx);
    if (result.type !== 'sequence') throw new Error('sequence 기대');
    const reach = result.sequence.steps[1];
    if (reach?.kind !== 'moveJoints') throw new Error('moveJoints 기대');
    expect(reach.targets['joint3']).toBe(1.0);
  });
});

describe('generateRuleBased — 모호/미지 대상', () => {
  it('대상 미지정 + 물체 여럿 → clarify(옵션 목록)', () => {
    const result = generateRuleBased('박스를 집어', armCtx);
    expect(result).toEqual({ type: 'clarify', question: '어느 것을 집을까요?', options: ['box_a', 'box_b'] });
  });

  it('존재하지 않는 대상 지정 → clarify(available ids)', () => {
    const result = generateRuleBased('box_z를 집어', armCtx);
    expect(result).toMatchObject({ type: 'clarify', options: ['box_a', 'box_b'] });
  });

  it('clarify 선택 토큰 [선택: box_b]을 되먹이면 그 대상으로 확정한다', () => {
    const result = generateRuleBased('박스를 집어 [선택: box_b]', armCtx);
    const wfc =
      result.type === 'sequence'
        ? result.sequence.steps.find((s) => s.kind === 'waitForCollision')
        : undefined;
    expect(result.type).toBe('sequence');
    expect(wfc).toMatchObject({ between: ['arm', 'box_b'] });
  });

  it('물체가 정확히 1개면 대상 미지정이어도 그것으로 확정한다', () => {
    const oneObjectCtx: SceneContext = {
      ...armCtx,
      objects: [{ id: 'box_a', type: 'object', position: [0.35, 0.03, 0.15] }],
    };
    const result = generateRuleBased('집어', oneObjectCtx);
    expect(result.type).toBe('sequence');
  });
});

describe('generateRuleBased — 에러 경계', () => {
  it('로봇이 없으면 error', () => {
    const noRobotCtx: SceneContext = { ...armCtx, robots: [] };
    const result = generateRuleBased('box_a를 집어', noRobotCtx);
    expect(result.type).toBe('error');
  });

  it('집을 물체가 없으면 error', () => {
    const noObjectCtx: SceneContext = { ...armCtx, objects: [] };
    const result = generateRuleBased('집어', noObjectCtx);
    expect(result.type).toBe('error');
  });

  it('해석할 수 없는 명령은 지원 패턴을 담은 error', () => {
    const result = generateRuleBased('날씨 어때', armCtx);
    if (result.type !== 'error') throw new Error('error 기대');
    expect(result.message).toContain('지원하는 명령');
  });
});

describe('generateRuleBased — 단일 동작(그리퍼/대기/홈)', () => {
  it('"열어" → gripper open', () => {
    const seq = expectSequence(generateRuleBased('열어', armCtx));
    expect(seq.steps).toHaveLength(1);
    expect(seq.steps[0]).toMatchObject({ kind: 'gripper', state: 'open' });
  });

  it('"닫아" → gripper close', () => {
    const seq = expectSequence(generateRuleBased('그리퍼 닫아', armCtx));
    expect(seq.steps[0]).toMatchObject({ kind: 'gripper', state: 'close' });
  });

  it('"3초 기다려" → wait 3', () => {
    const seq = expectSequence(generateRuleBased('3초 기다려', armCtx));
    expect(seq.steps[0]).toEqual({ kind: 'wait', durationSec: 3 });
  });

  it('"홈으로" → moveJoints(homePose)', () => {
    const seq = expectSequence(generateRuleBased('홈으로', armCtx));
    expect(seq.steps).toHaveLength(1);
    expect(seq.steps[0]).toMatchObject({
      kind: 'moveJoints',
      targets: { joint1: 0, joint2: -0.6, joint3: 1.2, joint5: -0.6 },
    });
  });
});

describe('generateRuleBased — 다중 절', () => {
  it('"~ 그리고 ~"로 이어진 절의 step을 이어 붙인다', () => {
    const seq = expectSequence(generateRuleBased('box_a를 집어 그리고 홈으로', armCtx));
    // touch(7 step: approach+nudge 2단) + home(1 step)
    expect(seq.steps.map((s) => s.kind)).toEqual([
      'gripper',
      'moveJoints',
      'setJoints',
      'waitForCollision',
      'gripper',
      'wait',
      'moveJoints',
      'moveJoints',
    ]);
  });

  it('쉼표로도 절을 나눈다 ("열어, 닫아")', () => {
    const seq = expectSequence(generateRuleBased('열어, 닫아', armCtx));
    expect(seq.steps.map((s) => s.kind)).toEqual(['gripper', 'gripper']);
    expect(seq.steps[0]).toMatchObject({ state: 'open' });
    expect(seq.steps[1]).toMatchObject({ state: 'close' });
  });

  it('한 절이라도 clarify면 전체가 clarify (부분 성공 미채택)', () => {
    const result = generateRuleBased('열어 그리고 박스를 집어', armCtx);
    expect(result.type).toBe('clarify');
  });
});
