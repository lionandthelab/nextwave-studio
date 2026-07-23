// planner/llm-adapter.ts — LLM 어댑터 경계 + 플래너 출력 계약.
// 규범: docs/PLANNER.md §3(PlannerResult), §7(어댑터 격리). 순수 타입/인터페이스만
// 담는다 — 구현(규칙 기반/Anthropic)은 각 어댑터 파일에 격리된다.

import type { ControlSequence } from '../schema';

/**
 * LLM 호출을 격리하는 어댑터. 모델/엔드포인트를 교체 가능하게 한다 (PLANNER.md §7).
 * 순수 문자열 in/out — planner의 검증/복구 로직은 이 인터페이스만 보고 LLM 없이 테스트된다.
 */
export interface LlmAdapter {
  /** 사람이 읽을 어댑터 이름 (UI 배지/로그용) */
  readonly name: string;
  /**
   * 시스템/사용자 프롬프트로 1회 완성을 요청하고 응답 텍스트를 돌려준다.
   * jsonSchema를 주면 구조화 출력(structured output)으로 JSON 형식을 강제한다.
   * 실패는 사람이 읽을 수 있는 한국어 메시지의 Error로 throw한다.
   */
  complete(
    systemPrompt: string,
    userPrompt: string,
    opts?: { jsonSchema?: object },
  ): Promise<string>;
}

/**
 * 플래너 출력 계약 (PLANNER.md §3). UX는 이 3분기로 나눠 처리한다:
 * - sequence: 검증 통과한 ControlSequence + 어떤 가정을 했는지(방향·대상 물체 등).
 * - clarify: 대상/의도가 모호할 때 사용자에게 되묻는다(옵션 버튼 카드).
 * - error: 복구 재시도 후에도 실패. 사람이 읽을 수 있는 사유.
 */
export type PlannerResult =
  | { type: 'sequence'; sequence: ControlSequence; assumptions?: string[] }
  | { type: 'clarify'; question: string; options?: string[] }
  | { type: 'error'; message: string };
