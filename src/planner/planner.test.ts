// planner/planner.test.ts — PlannerService 라우팅 + 프롬프트 빌더 단위 테스트 (LLM 없이).

import { describe, expect, it } from 'vitest';
import { validateScene, validateSequence } from '../schema';
import type { SceneSpec } from '../schema';
import armAndBoxesJson from '../assets/scenes/arm-and-boxes.scene.json';
import { buildContext } from './scene-context';
import { PlannerService, buildSystemPrompt, buildUserPrompt } from './planner';
import type { LlmAdapter } from './llm-adapter';

function loadScene(json: unknown): SceneSpec {
  const result = validateScene(json);
  if (!result.ok) throw new Error(`픽스처 씬 검증 실패: ${result.errors.join(' | ')}`);
  return result.value;
}

const armScene = loadScene(armAndBoxesJson);
const armCtx = buildContext(armScene);

/** 고정 응답을 돌려주는 어댑터 */
class StubAdapter implements LlmAdapter {
  readonly name = 'stub';
  constructor(private readonly response: string) {}
  complete(): Promise<string> {
    return Promise.resolve(this.response);
  }
}

describe('PlannerService — rule-based backend', () => {
  it('규칙 기반으로 검증된 시퀀스를 생성한다', async () => {
    const service = new PlannerService({ backend: 'rule-based' });
    const result = await service.generate('box_a를 집어', armCtx, armScene);
    expect(result.type).toBe('sequence');
    if (result.type === 'sequence') {
      expect(validateSequence(result.sequence, armScene).ok).toBe(true);
    }
  });

  it('모호 입력은 clarify로 통과시킨다', async () => {
    const service = new PlannerService({ backend: 'rule-based' });
    const result = await service.generate('박스를 집어', armCtx, armScene);
    expect(result).toMatchObject({ type: 'clarify', options: ['box_a', 'box_b'] });
  });

  it('backendName은 정직한 오프라인 라벨', () => {
    const service = new PlannerService({ backend: 'rule-based' });
    expect(service.backendName).toContain('규칙 기반');
  });
});

describe('PlannerService — LLM adapter backend', () => {
  it('어댑터 출력을 검증 후 sequence로 반환한다', async () => {
    const json = JSON.stringify({
      type: 'sequence',
      sequence: { id: 'planned', robot: 'arm', steps: [{ kind: 'wait', durationSec: 1 }] },
    });
    const service = new PlannerService({ backend: { adapter: new StubAdapter(json) } });
    const result = await service.generate('아무 문장', armCtx, armScene);
    expect(result.type).toBe('sequence');
  });

  it('어댑터가 무효 출력만 내면 error로 강등한다', async () => {
    const service = new PlannerService({ backend: { adapter: new StubAdapter('not json') } });
    const result = await service.generate('아무 문장', armCtx, armScene);
    expect(result.type).toBe('error');
  });

  it('backendName은 어댑터 이름', () => {
    const service = new PlannerService({ backend: { adapter: new StubAdapter('{}') } });
    expect(service.backendName).toBe('stub');
  });
});

describe('buildUserPrompt', () => {
  it('자연어를 그대로 전달한다', () => {
    expect(buildUserPrompt('box_a를 집어')).toBe('box_a를 집어');
  });
});

describe('buildSystemPrompt', () => {
  const prompt = buildSystemPrompt(armCtx);

  it('SceneContext의 로봇/물체 id를 포함한다', () => {
    expect(prompt).toContain('arm');
    expect(prompt).toContain('box_a');
    expect(prompt).toContain('box_b');
  });

  it('방향어 규약을 포함한다', () => {
    expect(prompt).toContain('-X');
    expect(prompt).toContain('+Z');
    expect(prompt).toContain('왼쪽');
  });

  it('JSON-only 규약과 그라운딩 규칙을 포함한다', () => {
    expect(prompt).toContain('오직 유효한 JSON만');
    expect(prompt).toContain('id만 사용');
    expect(prompt).toContain('clarify');
  });

  it('사용 가능한 stepKinds를 포함하고 moveToPose는 없다', () => {
    expect(prompt).toContain('moveJoints');
    expect(prompt).toContain('waitForCollision');
    expect(prompt).not.toContain('moveToPose');
  });

  it('few-shot 예시를 포함한다', () => {
    expect(prompt).toContain('예시');
    expect(prompt).toContain('box_a를 집어');
  });
});
