// planner/adapters/anthropic.ts — Anthropic Claude LlmAdapter 구현.
// 이 파일만 @anthropic-ai/sdk에 의존한다(계층 규칙의 유일한 예외). 브라우저에서 직접
// 호출하므로 dangerouslyAllowBrowser를 켠다(교육/프로토타입 — PRD §6). 구조화 출력으로
// JSON 형식을 강제하고, 거부(refusal)·API 오류를 한국어 메시지로 변환한다.

import Anthropic from '@anthropic-ai/sdk';
import type { LlmAdapter } from '../llm-adapter';

export interface AnthropicAdapterOptions {
  apiKey: string;
  /** 모델 id (기본 'claude-opus-5'). 사용자가 설정에서 바꿀 수 있다 */
  model?: string;
}

/** 기본 모델 — 사용자 설정 가능 문자열 */
export const DEFAULT_ANTHROPIC_MODEL = 'claude-opus-5';

/** ControlSequence 하나에는 충분한 출력 예산 */
const MAX_TOKENS = 4096;

export class AnthropicAdapter implements LlmAdapter {
  readonly name = 'Anthropic Claude';
  private readonly client: Anthropic;
  private readonly model: string;

  constructor(opts: AnthropicAdapterOptions) {
    // 설정(apiKey/model)은 생성자로 주입 — planner 계층은 localStorage/DOM을 모른다(ui의 몫).
    this.client = new Anthropic({ apiKey: opts.apiKey, dangerouslyAllowBrowser: true });
    this.model = opts.model ?? DEFAULT_ANTHROPIC_MODEL;
  }

  async complete(
    systemPrompt: string,
    userPrompt: string,
    opts?: { jsonSchema?: object },
  ): Promise<string> {
    let response: Anthropic.Message;
    try {
      response = await this.client.messages.create({
        model: this.model,
        max_tokens: MAX_TOKENS,
        // 이 모델에서는 temperature/top_p/top_k를 보내지 않는다(제거됨 — 400). adaptive thinking 사용.
        thinking: { type: 'adaptive' },
        system: systemPrompt,
        messages: [{ role: 'user', content: userPrompt }],
        ...(opts?.jsonSchema
          ? {
              output_config: {
                format: { type: 'json_schema', schema: opts.jsonSchema as Record<string, unknown> },
              },
            }
          : {}),
      });
    } catch (error) {
      throw mapAnthropicError(error);
    }

    // refusal이면 content가 비어 있을 수 있으므로 먼저 검사한다.
    if (response.stop_reason === 'refusal') {
      throw new Error('요청이 정책상 거부되었습니다. 다른 문장으로 다시 시도해 주세요.');
    }

    const textBlock = response.content.find(
      (block): block is Anthropic.TextBlock => block.type === 'text',
    );
    if (!textBlock) {
      throw new Error('모델 응답에 텍스트가 없습니다.');
    }
    return textBlock.text;
  }
}

/** SDK의 타입 있는 오류 클래스를 한국어 메시지로 변환한다 (문자열 매칭 금지) */
function mapAnthropicError(error: unknown): Error {
  if (error instanceof Anthropic.AuthenticationError) {
    return new Error('API 키가 유효하지 않습니다.');
  }
  if (error instanceof Anthropic.RateLimitError) {
    return new Error('요청이 많아 잠시 후 다시 시도해 주세요 (요청 한도 초과).');
  }
  if (error instanceof Anthropic.APIError) {
    return new Error(`Anthropic API 오류: ${error.message}`);
  }
  if (error instanceof Error) return error;
  return new Error(String(error));
}
