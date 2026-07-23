// planner/validate-repair.ts — 플래너 출력 검증 + 복구 루프. 규범: docs/PLANNER.md §4.
//
// 신뢰 경계(불변식 §2.9): 미검증/무효 출력은 절대 실행에 노출되지 않는다. LLM raw 출력을
// (1) JSON으로 추출 → (2) PlannerResult 봉투 + ControlSequence 스키마·씬 참조 무결성
// 검증 → 실패 시 오류를 되먹여 재생성(상한 N회) → 초과 시 error 변형.
//
// schema 계층의 validateSequence(zod + 참조 무결성)를 그대로 재사용한다 — planner는
// 검증 규칙을 다시 구현하지 않는다(단일 진실).

import type { SceneSpec } from '../schema';
import { validateSequence } from '../schema';
import type { LlmAdapter, PlannerResult } from './llm-adapter';

/** 기본 복구 재시도 상한 (PLANNER.md §4.2) */
export const DEFAULT_MAX_REPAIR_ATTEMPTS = 3;

// ── 검증 ─────────────────────────────────────────────────────────────

export type ValidatePlannerOutputResult =
  | { ok: true; result: PlannerResult }
  | { ok: false; errors: string[] };

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function asStringArray(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) return undefined;
  if (!value.every((item) => typeof item === 'string')) return undefined;
  return [...(value as string[])];
}

/**
 * 파싱된 값이 유효한 PlannerResult 봉투인지 검사한다(3분기 수용).
 * - sequence: parsed.sequence가 반드시 validateSequence(+scene 참조 무결성)를 통과해야 한다.
 * - clarify/error: 구조(question/options, message)만 확인한다.
 * 실패 시 "경로: 한국어" 오류 목록을 반환한다(모델에 되먹임 가능한 형태).
 */
export function validatePlannerOutput(
  parsed: unknown,
  scene: SceneSpec,
): ValidatePlannerOutputResult {
  if (!isRecord(parsed)) {
    return { ok: false, errors: ['출력이 객체(JSON object)가 아닙니다'] };
  }
  const type = parsed['type'];

  if (type === 'sequence') {
    const validation = validateSequence(parsed['sequence'], scene);
    if (!validation.ok) return { ok: false, errors: validation.errors };
    const result: PlannerResult = { type: 'sequence', sequence: validation.value };
    const assumptions = asStringArray(parsed['assumptions']);
    if (assumptions) result.assumptions = assumptions;
    return { ok: true, result };
  }

  if (type === 'clarify') {
    const question = parsed['question'];
    if (typeof question !== 'string' || question.length === 0) {
      return { ok: false, errors: ["clarify에는 비어 있지 않은 'question' 문자열이 필요합니다"] };
    }
    const result: PlannerResult = { type: 'clarify', question };
    const options = asStringArray(parsed['options']);
    if (options) result.options = options;
    return { ok: true, result };
  }

  if (type === 'error') {
    const message = parsed['message'];
    if (typeof message !== 'string' || message.length === 0) {
      return { ok: false, errors: ["error에는 비어 있지 않은 'message' 문자열이 필요합니다"] };
    }
    return { ok: true, result: { type: 'error', message } };
  }

  return {
    ok: false,
    errors: [
      `알 수 없는 결과 type입니다: ${JSON.stringify(type)} (허용: 'sequence' | 'clarify' | 'error')`,
    ],
  };
}

// ── JSON 추출 (구조화 출력이 실패할 때의 방어선) ─────────────────────

/**
 * raw 문자열에서 JSON 값을 추출한다. 구조화 출력이면 이미 순수 JSON이지만, 코드펜스/서문이
 * 섞여 들어오는 경우를 방어적으로 벗겨낸다. 파싱 불가 시 null.
 */
export function extractJson(raw: string): unknown | null {
  const trimmed = raw.trim();
  if (trimmed === '') return null;

  // 1) 그대로 파싱 시도
  const direct = tryParse(trimmed);
  if (direct.ok) return direct.value;

  // 2) 코드펜스 제거 (```json ... ``` 또는 ``` ... ```)
  const fenceMatch = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fenceMatch) {
    const inner = tryParse(fenceMatch[1]!.trim());
    if (inner.ok) return inner.value;
  }

  // 3) 첫 '{'..마지막 '}' 구간을 잘라 파싱 (서문/후문 제거)
  const first = trimmed.indexOf('{');
  const last = trimmed.lastIndexOf('}');
  if (first >= 0 && last > first) {
    const slice = tryParse(trimmed.slice(first, last + 1));
    if (slice.ok) return slice.value;
  }

  return null;
}

function tryParse(text: string): { ok: true; value: unknown } | { ok: false } {
  try {
    return { ok: true, value: JSON.parse(text) as unknown };
  } catch {
    return { ok: false };
  }
}

// ── 복구 루프 ────────────────────────────────────────────────────────

/**
 * 어댑터를 최대 maxAttempts회 호출한다. 매 시도마다 raw → extractJson → validatePlannerOutput.
 * 실패하면 구체적 오류를 사용자 프롬프트에 되먹여 재생성한다. N회 초과 시 마지막 오류를 담은
 * error 변형을 반환한다. 어댑터가 throw하면(인증/네트워크 등) 즉시 error로 변환한다.
 */
export async function repairLoop(
  adapter: LlmAdapter,
  systemPrompt: string,
  userPrompt: string,
  scene: SceneSpec,
  maxAttempts: number = DEFAULT_MAX_REPAIR_ATTEMPTS,
): Promise<PlannerResult> {
  let lastErrors: string[] = ['출력을 받지 못했습니다'];
  let currentUserPrompt = userPrompt;

  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    let raw: string;
    try {
      raw = await adapter.complete(systemPrompt, currentUserPrompt, {
        jsonSchema: PLANNER_RESULT_JSON_SCHEMA,
      });
    } catch (error) {
      // 인증/레이트리밋/네트워크 오류는 재시도로 해결되지 않는다 — 즉시 사용자에게 알린다.
      return { type: 'error', message: error instanceof Error ? error.message : String(error) };
    }

    const parsed = extractJson(raw);
    if (parsed === null) {
      lastErrors = ['출력을 JSON으로 파싱할 수 없습니다'];
    } else {
      const validation = validatePlannerOutput(parsed, scene);
      if (validation.ok) return validation.result;
      lastErrors = validation.errors;
    }

    currentUserPrompt =
      `${userPrompt}\n\n이전 출력의 오류: ${lastErrors.join('; ')}\n` +
      '오류를 수정한 완전한 JSON을 다시 출력하라. 설명·코드펜스 없이 JSON만.';
  }

  return {
    type: 'error',
    message:
      `플래너가 유효한 출력을 생성하지 못했습니다 (${maxAttempts}회 시도). ` +
      `마지막 오류: ${lastErrors.join('; ')}`,
  };
}

// ── PlannerResult JSON 스키마 (구조화 출력용, 손수 작성) ──────────────
// 규칙(구조화 출력 계약): 모든 닫힌 객체에 additionalProperties:false, 재귀 금지,
// 수치 min/max 제약 금지, 유니온은 anyOf. targets(관절 맵)만 동적 키가 필요하므로
// additionalProperties를 number 스키마로 둔다(맵 타입의 표준 표현).

/** 모든 step 공통 옵션 필드 */
const STEP_COMMON_PROPERTIES = {
  enabled: { type: 'boolean' },
  note: { type: 'string' },
} as const;

/** 관절 목표 맵 — 동적 키(관절명) → number */
const JOINT_TARGETS_SCHEMA = {
  type: 'object',
  additionalProperties: { type: 'number' },
} as const;

const CONTROL_STEP_SCHEMA = {
  anyOf: [
    {
      type: 'object',
      properties: {
        kind: { const: 'moveJoints' },
        robot: { type: 'string' },
        targets: JOINT_TARGETS_SCHEMA,
        durationSec: { type: 'number' },
        easing: { enum: ['linear', 'easeInOut', 'step'] },
        ...STEP_COMMON_PROPERTIES,
      },
      required: ['kind', 'targets', 'durationSec'],
      additionalProperties: false,
    },
    {
      type: 'object',
      properties: {
        kind: { const: 'setJoints' },
        robot: { type: 'string' },
        targets: JOINT_TARGETS_SCHEMA,
        ...STEP_COMMON_PROPERTIES,
      },
      required: ['kind', 'targets'],
      additionalProperties: false,
    },
    {
      type: 'object',
      properties: {
        kind: { const: 'gripper' },
        robot: { type: 'string' },
        state: { anyOf: [{ const: 'open' }, { const: 'close' }, { type: 'number' }] },
        durationSec: { type: 'number' },
        ...STEP_COMMON_PROPERTIES,
      },
      required: ['kind', 'state'],
      additionalProperties: false,
    },
    {
      type: 'object',
      properties: {
        kind: { const: 'wait' },
        durationSec: { type: 'number' },
        ...STEP_COMMON_PROPERTIES,
      },
      required: ['kind', 'durationSec'],
      additionalProperties: false,
    },
    {
      type: 'object',
      properties: {
        kind: { const: 'waitForCollision' },
        between: { type: 'array', items: { type: 'string' } },
        timeoutSec: { type: 'number' },
        ...STEP_COMMON_PROPERTIES,
      },
      required: ['kind', 'between'],
      additionalProperties: false,
    },
    {
      type: 'object',
      properties: {
        kind: { const: 'label' },
        name: { type: 'string' },
        ...STEP_COMMON_PROPERTIES,
      },
      required: ['kind', 'name'],
      additionalProperties: false,
    },
    {
      type: 'object',
      properties: {
        kind: { const: 'goto' },
        label: { type: 'string' },
        times: { type: 'number' },
        ...STEP_COMMON_PROPERTIES,
      },
      required: ['kind', 'label'],
      additionalProperties: false,
    },
  ],
} as const;

const CONTROL_SEQUENCE_SCHEMA = {
  type: 'object',
  properties: {
    id: { type: 'string' },
    robot: { type: 'string' },
    loop: { type: 'boolean' },
    steps: { type: 'array', items: CONTROL_STEP_SCHEMA },
  },
  required: ['id', 'robot', 'steps'],
  additionalProperties: false,
} as const;

/** 3분기 PlannerResult 봉투 스키마 (sequence | clarify | error) */
export const PLANNER_RESULT_JSON_SCHEMA = {
  anyOf: [
    {
      type: 'object',
      properties: {
        type: { const: 'sequence' },
        sequence: CONTROL_SEQUENCE_SCHEMA,
        assumptions: { type: 'array', items: { type: 'string' } },
      },
      required: ['type', 'sequence'],
      additionalProperties: false,
    },
    {
      type: 'object',
      properties: {
        type: { const: 'clarify' },
        question: { type: 'string' },
        options: { type: 'array', items: { type: 'string' } },
      },
      required: ['type', 'question'],
      additionalProperties: false,
    },
    {
      type: 'object',
      properties: {
        type: { const: 'error' },
        message: { type: 'string' },
      },
      required: ['type', 'message'],
      additionalProperties: false,
    },
  ],
} as const;
