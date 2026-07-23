// planner/planner.ts — 플래너 오케스트레이션. 규범: docs/PLANNER.md §4, §7.
//
// PlannerService는 backend에 따라 라우팅한다:
// - 'rule-based': 결정론적 오프라인 플래너(데모/게이트 기본값)로 직접 생성 후 방어적 재검증.
// - { adapter }: 시스템 프롬프트(스키마+컨텍스트+규약+few-shot) 구성 → repairLoop(검증/복구).
//
// 어떤 경로든 반환 전에 sequence 변형은 반드시 검증을 통과한다(불변식 §2.9 — 미검증 출력
// 실행 금지). 자동 실행은 하지 않는다 — 사용자가 검토 후 Play를 눌러야 한다(ui의 몫).

import type { SceneSpec } from '../schema';
import { validateSequence } from '../schema';
import type { LlmAdapter, PlannerResult } from './llm-adapter';
import { generateRuleBased } from './adapters/rule-based';
import type { SceneContext } from './scene-context';
import { repairLoop } from './validate-repair';

export type PlannerBackend = 'rule-based' | { adapter: LlmAdapter };

export interface PlannerServiceOptions {
  backend: PlannerBackend;
}

export class PlannerService {
  private readonly backend: PlannerBackend;

  constructor(opts: PlannerServiceOptions) {
    this.backend = opts.backend;
  }

  /** 현재 backend의 사람이 읽을 이름 */
  get backendName(): string {
    return this.backend === 'rule-based' ? '규칙 기반 (오프라인 데모)' : this.backend.adapter.name;
  }

  /**
   * 자연어 → PlannerResult. scene은 씬 참조 무결성 검증에 쓰인다(ctx와 같은 씬이어야 한다).
   * sequence 변형은 반환 전 반드시 검증을 통과한다.
   */
  async generate(nl: string, ctx: SceneContext, scene: SceneSpec): Promise<PlannerResult> {
    if (this.backend === 'rule-based') {
      const result = generateRuleBased(nl, ctx);
      return ensureValidated(result, scene);
    }
    const systemPrompt = buildSystemPrompt(ctx);
    const userPrompt = buildUserPrompt(nl);
    return repairLoop(this.backend.adapter, systemPrompt, userPrompt, scene);
  }
}

/**
 * sequence 변형이면 방어적으로 재검증한다 — 규칙 기반 출력도 실행 노출 전 검증을 거친다.
 * clarify/error는 그대로 통과. 검증 실패(정상 경로에선 발생하지 않음)는 error로 강등한다.
 */
function ensureValidated(result: PlannerResult, scene: SceneSpec): PlannerResult {
  if (result.type !== 'sequence') return result;
  const validation = validateSequence(result.sequence, scene);
  if (validation.ok) return result;
  return {
    type: 'error',
    message: `생성된 시퀀스가 검증을 통과하지 못했습니다: ${validation.errors.join('; ')}`,
  };
}

// ── 프롬프트 빌더 (순수 함수 — LLM 없이 테스트 가능) ─────────────────

/** 사용자 프롬프트: 자연어 지시를 그대로 전달 */
export function buildUserPrompt(nl: string): string {
  return nl;
}

/**
 * 시스템 프롬프트: 역할 + 사용 가능한 step 스키마 요약 + SceneContext(JSON) + 규약 +
 * few-shot 1개. 규약(PLANNER.md §4.1): 오직 JSON만, SceneContext id만 사용, 없는 대상은
 * clarify, limits 준수, assumptions 노출, 방향어 규약.
 */
export function buildSystemPrompt(ctx: SceneContext): string {
  const directionLines = [
    `- 왼쪽 = ${ctx.directions.left}, 오른쪽 = ${ctx.directions.right}`,
    `- 위 = ${ctx.directions.up}, 앞 = ${ctx.directions.forward}, 뒤 = ${ctx.directions.back}`,
  ].join('\n');

  return [
    '당신은 브라우저 로봇 시뮬레이터의 자연어 → 제어 시퀀스 플래너다.',
    '주어진 씬 컨텍스트에 그라운딩된 ControlSequence 초안을 생성한다.',
    '',
    '## 출력 계약',
    '아래 3가지 중 정확히 하나의 JSON을 반환한다:',
    '1. { "type": "sequence", "sequence": <ControlSequence>, "assumptions"?: string[] }',
    '2. { "type": "clarify", "question": string, "options"?: string[] }',
    '3. { "type": "error", "message": string }',
    '',
    `## 사용 가능한 step 종류 (kind)`,
    ctx.capabilities.stepKinds.map((kind) => `- ${kind}`).join('\n'),
    'ControlSequence = { id: string, robot: string, steps: ControlStep[] }.',
    'moveJoints: { kind, targets: {관절명: number}, durationSec, easing? }.',
    'gripper: { kind, state: "open"|"close"|0..1, durationSec? }.',
    'waitForCollision: { kind, between: [entityId, entityId], timeoutSec? }.',
    'wait: { kind, durationSec }. setJoints: { kind, targets }. label/goto: 흐름 제어.',
    '',
    '## 좌표 규약',
    `프레임: ${ctx.frame}. 회전은 라디안.`,
    directionLines,
    '',
    '## 씬 컨텍스트 (이 안의 id만 사용할 것)',
    JSON.stringify({ robots: ctx.robots, objects: ctx.objects }, null, 2),
    '',
    '## 규약 (반드시 준수)',
    '- 오직 유효한 JSON만 출력한다. 서문/설명/코드펜스 금지.',
    '- robots/joints/objects는 위 SceneContext의 id만 사용한다. 없는 대상을 지어내지 말 것.',
    '- 필요한 대상이 모호하거나 씬에 없으면 값을 지어내지 말고 clarify를 반환한다.',
    '- 관절 목표는 각 관절의 limits [min, max] 범위 안으로 한다.',
    '- 방향어·기본 duration 등 해석한 가정을 assumptions에 한국어로 요약해 노출한다.',
    '',
    '## 예시',
    'user: box_a를 집어',
    'assistant:',
    JSON.stringify(FEW_SHOT_EXAMPLE, null, 2),
  ].join('\n');
}

/** few-shot 예시 출력 (arm이 box_a를 집는 기본 패턴) */
const FEW_SHOT_EXAMPLE = {
  type: 'sequence',
  sequence: {
    id: 'planned',
    robot: 'arm',
    steps: [
      { kind: 'gripper', state: 'open', durationSec: 0.4 },
      {
        kind: 'moveJoints',
        targets: { joint1: -0.405, joint2: 0.639, joint3: 1.414, joint5: 1.089 },
        durationSec: 2.0,
        easing: 'easeInOut',
      },
      { kind: 'setJoints', targets: { joint2: 0.683, joint3: 1.442, joint5: 1.017 } },
      { kind: 'waitForCollision', between: ['arm', 'box_a'], timeoutSec: 6 },
      { kind: 'gripper', state: 'close', durationSec: 0.5 },
      { kind: 'wait', durationSec: 0.3 },
      {
        kind: 'moveJoints',
        targets: { joint1: 0, joint2: -0.6, joint3: 1.2, joint5: -0.6 },
        durationSec: 2.0,
        easing: 'easeInOut',
      },
    ],
  },
  assumptions: ['box_a를 대상으로 해석', '도달 자세는 휴리스틱(정확한 IK 아님)'],
} as const;
