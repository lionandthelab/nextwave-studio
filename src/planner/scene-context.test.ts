// planner/scene-context.test.ts — buildContext 단위 테스트 (LLM 없이 — PLANNER.md §7).
// 픽스처: 실제 샘플 씬 arm-and-boxes.scene.json (arm 로봇 + box_a + box_b).

import { describe, expect, it } from 'vitest';
import { validateScene, validateSequence } from '../schema';
import type { SceneSpec } from '../schema';
import armAndBoxesJson from '../assets/scenes/arm-and-boxes.scene.json';
import obstacleAvoidanceJson from '../assets/scenes/obstacle-avoidance.scene.json';
import {
  buildContext,
  DEFAULT_REVOLUTE_LIMIT,
  DIRECTION_CONVENTIONS,
  PLANNER_STEP_KINDS,
  type WorldSnapshot,
} from './scene-context';

function loadScene(json: unknown): SceneSpec {
  const result = validateScene(json);
  if (!result.ok) throw new Error(`픽스처 씬 검증 실패: ${result.errors.join(' | ')}`);
  return result.value;
}

const armScene = loadScene(armAndBoxesJson);
const obstacleScene = loadScene(obstacleAvoidanceJson);

describe('buildContext — robots', () => {
  it('로봇을 하나 추출하고 id를 유지한다', () => {
    const ctx = buildContext(armScene);
    expect(ctx.robots).toHaveLength(1);
    expect(ctx.robots[0]!.id).toBe('arm');
  });

  it('관절을 home ∪ jointMap ∪ jointLimits 합집합에서 유도한다(등장 순서)', () => {
    const ctx = buildContext(armScene);
    const names = ctx.robots[0]!.joints.map((j) => j.name);
    expect(names).toEqual(['joint1', 'joint2', 'joint3', 'joint5']);
  });

  // 그라운딩⇄검증 정렬(안전 리뷰 회귀): gripper.joints는 별도 제어면(gripper step +
  // SceneContextRobot.gripper)이고 checkJointNames의 knownJoints에서 제외되므로,
  // joints 배열로 광고하면 LLM이 moveJoints 대상으로 지어내 검증 거부되는 비대칭이 생긴다.
  // joints 배열은 오직 moveJoints/setJoints로 제어 가능한 관절만 담아야 한다.
  it('gripper.joints만 선언된 관절은 joints 배열에서 제외한다(gripper로만 노출)', () => {
    const ctx = buildContext(armScene);
    const names = ctx.robots[0]!.joints.map((j) => j.name);
    expect(names).not.toContain('finger_left_joint');
    expect(names).not.toContain('finger_right_joint');
    // 그리퍼 자체는 여전히 gripper 객체로 완전히 노출된다.
    expect(ctx.robots[0]!.gripper).toEqual({ name: 'gripper', open: 0.03, close: 0.0 });
  });

  it('current를 home에서 채우고 없으면 0으로 둔다', () => {
    const ctx = buildContext(armScene);
    const byName = new Map(ctx.robots[0]!.joints.map((j) => [j.name, j]));
    expect(byName.get('joint2')!.current).toBe(-0.6);
    expect(byName.get('joint3')!.current).toBe(1.2);
    // jointLimits가 없어 revolute 기본 범위를 쓰는 관절도 home값을 current로 반영한다.
    expect(byName.get('joint1')!.current).toBe(0.0);
  });

  it('jointLimits가 없으면 revolute 기본 범위를 쓴다', () => {
    const ctx = buildContext(armScene);
    const joint1 = ctx.robots[0]!.joints.find((j) => j.name === 'joint1')!;
    expect(joint1.limits).toEqual([DEFAULT_REVOLUTE_LIMIT[0], DEFAULT_REVOLUTE_LIMIT[1]]);
    expect(joint1.type).toBe('revolute');
  });

  it('gripper를 RobotSpec.gripper에서 open/close로 노출한다', () => {
    const ctx = buildContext(armScene);
    expect(ctx.robots[0]!.gripper).toEqual({ name: 'gripper', open: 0.03, close: 0.0 });
  });

  it('homePose를 로봇 home의 복사본으로 노출한다', () => {
    const ctx = buildContext(armScene);
    expect(ctx.robots[0]!.homePose).toEqual({ joint1: 0.0, joint2: -0.6, joint3: 1.2, joint5: -0.6 });
  });
});

describe('buildContext — 그라운딩⇄검증 정렬', () => {
  // 광고한 관절이 곧 moveJoints/setJoints로 검증 통과 가능한 관절이어야 한다 —
  // 두 모듈이 미래에 어긋나지 않도록 buildContext 출력과 validateSequence를 함께 검사.
  it('joints 배열의 모든 관절은 moveJoints 대상으로 validateSequence를 통과한다', () => {
    const ctx = buildContext(armScene);
    const targets: Record<string, number> = {};
    for (const joint of ctx.robots[0]!.joints) targets[joint.name] = 0;
    const result = validateSequence(
      { id: 'grounding-align', robot: 'arm', steps: [{ kind: 'moveJoints', targets, durationSec: 1 }] },
      armScene,
    );
    expect(result.ok).toBe(true);
  });

  it('gripper 전용 관절을 moveJoints 대상으로 쓰면 validateSequence가 거부한다', () => {
    const result = validateSequence(
      {
        id: 'gripper-as-joint',
        robot: 'arm',
        steps: [{ kind: 'moveJoints', targets: { finger_left_joint: 0.02 }, durationSec: 1 }],
      },
      armScene,
    );
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.errors.join(' | ')).toContain('finger_left_joint');
    }
  });
});

describe('buildContext — objects', () => {
  it('object/static 엔티티만 objects로, 로봇은 제외한다', () => {
    const ctx = buildContext(armScene);
    expect(ctx.objects.map((o) => o.id)).toEqual(['box_a', 'box_b']);
  });

  it('box 치수를 2*halfExtents = {w,h,l}로 계산한다', () => {
    const ctx = buildContext(armScene);
    const boxA = ctx.objects.find((o) => o.id === 'box_a')!;
    expect(boxA.dimensions).toEqual({ w: 0.06, h: 0.06, l: 0.06 });
    expect(boxA.type).toBe('object');
    expect(boxA.position).toEqual([0.35, 0.03, 0.15]);
  });

  it('static 엔티티는 type static으로 노출한다', () => {
    const ctx = buildContext(obstacleScene);
    const pillar = ctx.objects.find((o) => o.id === 'pillar')!;
    expect(pillar.type).toBe('static');
    expect(pillar.dimensions).toEqual({ w: 0.04, h: 0.65, l: 0.04 });
  });
});

describe('buildContext — capabilities/frame/directions', () => {
  it('stepKinds에 구현된 종류만 노출하고 moveToPose는 제외한다', () => {
    const ctx = buildContext(armScene);
    expect(ctx.capabilities.stepKinds).toEqual([...PLANNER_STEP_KINDS]);
    expect(ctx.capabilities.stepKinds).not.toContain('moveToPose');
  });

  it('frame과 directions 규약을 노출한다', () => {
    const ctx = buildContext(armScene);
    expect(ctx.frame).toBe('y-up, meters, radians');
    expect(ctx.directions).toEqual(DIRECTION_CONVENTIONS);
    expect(ctx.directions.left).toBe('-X');
    expect(ctx.directions.forward).toBe('+Z');
  });
});

describe('buildContext — live WorldSnapshot', () => {
  it('live 관절값이 있으면 current를 덮어쓴다', () => {
    const live: WorldSnapshot = { jointValuesByRobot: { arm: { joint2: 0.9 } } };
    const ctx = buildContext(armScene, live);
    const joint2 = ctx.robots[0]!.joints.find((j) => j.name === 'joint2')!;
    expect(joint2.current).toBe(0.9);
    // 스냅샷에 없는 관절은 home로 유지
    const joint3 = ctx.robots[0]!.joints.find((j) => j.name === 'joint3')!;
    expect(joint3.current).toBe(1.2);
  });

  it('live 위치가 있으면 물체 position을 덮어쓴다', () => {
    const live: WorldSnapshot = { positionsByEntity: { box_a: [0.1, 0.2, 0.3] } };
    const ctx = buildContext(armScene, live);
    const boxA = ctx.objects.find((o) => o.id === 'box_a')!;
    expect(boxA.position).toEqual([0.1, 0.2, 0.3]);
  });
});
