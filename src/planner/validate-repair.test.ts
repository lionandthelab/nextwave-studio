// planner/validate-repair.test.ts — 검증·복구 루프 단위 테스트 (LLM 없이 — PLANNER.md §7).
// FakeAdapter로 LLM 응답을 스크립트한다(진짜 API 호출 없음).

import { describe, expect, it } from 'vitest';
import { validateScene } from '../schema';
import type { SceneSpec } from '../schema';
import armAndBoxesJson from '../assets/scenes/arm-and-boxes.scene.json';
import type { LlmAdapter } from './llm-adapter';
import { extractJson, repairLoop, validatePlannerOutput } from './validate-repair';

function loadScene(json: unknown): SceneSpec {
  const result = validateScene(json);
  if (!result.ok) throw new Error(`픽스처 씬 검증 실패: ${result.errors.join(' | ')}`);
  return result.value;
}

const armScene = loadScene(armAndBoxesJson);

/** 스크립트된 응답을 순서대로 돌려주는 어댑터 (마지막 응답을 이후에도 반복) */
class ScriptedAdapter implements LlmAdapter {
  readonly name = 'scripted';
  calls = 0;
  readonly prompts: string[] = [];
  constructor(private readonly responses: string[]) {}
  complete(_system: string, user: string): Promise<string> {
    this.prompts.push(user);
    const index = Math.min(this.calls, this.responses.length - 1);
    this.calls += 1;
    return Promise.resolve(this.responses[index] ?? '');
  }
}

/** 항상 throw하는 어댑터 (인증/네트워크 오류 모사) */
class ThrowingAdapter implements LlmAdapter {
  readonly name = 'throwing';
  calls = 0;
  complete(): Promise<string> {
    this.calls += 1;
    return Promise.reject(new Error('API 키가 유효하지 않습니다.'));
  }
}

const VALID_SEQUENCE_JSON = JSON.stringify({
  type: 'sequence',
  sequence: { id: 'planned', robot: 'arm', steps: [{ kind: 'wait', durationSec: 1 }] },
});
const SCHEMA_INVALID_JSON = JSON.stringify({
  type: 'sequence',
  sequence: { id: 'x', robot: 'arm', steps: [{ kind: 'wait', durationSec: -1 }] },
});
const NOT_JSON = 'this is not json {';

describe('validatePlannerOutput', () => {
  it('유효한 sequence를 수용하고 검증된 값을 돌려준다', () => {
    const result = validatePlannerOutput(JSON.parse(VALID_SEQUENCE_JSON), armScene);
    expect(result.ok).toBe(true);
    if (result.ok && result.result.type === 'sequence') {
      expect(result.result.sequence.robot).toBe('arm');
    }
  });

  it('스키마 위반 sequence는 오류 목록으로 거부한다', () => {
    const result = validatePlannerOutput(JSON.parse(SCHEMA_INVALID_JSON), armScene);
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.errors.length).toBeGreaterThan(0);
  });

  it('씬에 없는 로봇을 참조하면 참조 무결성으로 거부한다', () => {
    const parsed = { type: 'sequence', sequence: { id: 'x', robot: 'ghost', steps: [{ kind: 'wait', durationSec: 1 }] } };
    const result = validatePlannerOutput(parsed, armScene);
    expect(result.ok).toBe(false);
  });

  it('clarify는 구조만 확인하고 통과한다', () => {
    const result = validatePlannerOutput({ type: 'clarify', question: '어느 것?', options: ['box_a'] }, armScene);
    expect(result).toEqual({
      ok: true,
      result: { type: 'clarify', question: '어느 것?', options: ['box_a'] },
    });
  });

  it('error는 구조만 확인하고 통과한다', () => {
    const result = validatePlannerOutput({ type: 'error', message: '실패' }, armScene);
    expect(result).toEqual({ ok: true, result: { type: 'error', message: '실패' } });
  });

  it('알 수 없는 type은 거부한다', () => {
    const result = validatePlannerOutput({ type: 'nope' }, armScene);
    expect(result.ok).toBe(false);
  });

  it('객체가 아니면 거부한다', () => {
    expect(validatePlannerOutput('문자열', armScene).ok).toBe(false);
    expect(validatePlannerOutput(null, armScene).ok).toBe(false);
  });
});

describe('extractJson', () => {
  it('순수 JSON을 파싱한다', () => {
    expect(extractJson('{"a":1}')).toEqual({ a: 1 });
  });

  it('코드펜스(```json)를 벗겨낸다', () => {
    expect(extractJson('```json\n{"a":1}\n```')).toEqual({ a: 1 });
    expect(extractJson('```\n{"b":2}\n```')).toEqual({ b: 2 });
  });

  it('서문/후문에 둘러싸인 JSON을 잘라낸다', () => {
    expect(extractJson('여기 결과: {"type":"error","message":"x"} 이상입니다')).toEqual({
      type: 'error',
      message: 'x',
    });
  });

  it('JSON이 없으면 null', () => {
    expect(extractJson('not json at all')).toBeNull();
    expect(extractJson('')).toBeNull();
  });
});

describe('repairLoop', () => {
  it('[비 JSON, 스키마 위반, 유효] → 복구 후 성공하며 3회 호출한다', async () => {
    const adapter = new ScriptedAdapter([NOT_JSON, SCHEMA_INVALID_JSON, VALID_SEQUENCE_JSON]);
    const result = await repairLoop(adapter, 'sys', 'box_a를 집어', armScene);
    expect(result.type).toBe('sequence');
    expect(adapter.calls).toBe(3);
    // 재시도 프롬프트에 이전 오류가 되먹여졌는지 확인
    expect(adapter.prompts[1]).toContain('이전 출력의 오류');
    expect(adapter.prompts[2]).toContain('이전 출력의 오류');
  });

  it('계속 무효하면 N회 후 error(마지막 오류 포함)', async () => {
    const adapter = new ScriptedAdapter([SCHEMA_INVALID_JSON]);
    const result = await repairLoop(adapter, 'sys', 'x', armScene, 3);
    expect(result.type).toBe('error');
    if (result.type === 'error') expect(result.message).toContain('3회');
    expect(adapter.calls).toBe(3);
  });

  it('모델이 clarify를 반환하면 그대로 통과(재시도 없음)', async () => {
    const clarifyJson = JSON.stringify({ type: 'clarify', question: '어느 박스?', options: ['box_a', 'box_b'] });
    const adapter = new ScriptedAdapter([clarifyJson]);
    const result = await repairLoop(adapter, 'sys', 'x', armScene);
    expect(result).toMatchObject({ type: 'clarify', question: '어느 박스?' });
    expect(adapter.calls).toBe(1);
  });

  it('어댑터가 throw하면 즉시 error로 변환(재시도 안 함)', async () => {
    const adapter = new ThrowingAdapter();
    const result = await repairLoop(adapter, 'sys', 'x', armScene);
    expect(result).toEqual({ type: 'error', message: 'API 키가 유효하지 않습니다.' });
    expect(adapter.calls).toBe(1);
  });
});
