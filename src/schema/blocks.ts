// schema/blocks.ts — 재사용 블록 캡처·전개 (inline expansion)
//
// ── 설계 결정 (확정) ────────────────────────────────────────────────
// 블록은 참조가 아니라 **삽입 시 전개**된다. 블록 원본을 나중에 고쳐도 이미 전개된
// 작업은 바뀌지 않는다 — 검증 끝난 작업이 조용히 바뀌는 것은 로봇 셀에서 사고다
// (entities.ts 헤더의 "복사본 의미론"과 동일 원칙). 전개 흔적은 표시용 라벨
// (blockRefLabel)로만 남기고, 전개된 step의 실행 의미는 일반 step과 완전히 동일하다.
//
// ── 계층/검증 규칙 ─────────────────────────────────────────────────
// schema 내부(types/validate/entities)만 안다 — DOM/물리/LLM 없음(순수, node 테스트
// 대상). 전개 결과는 **반드시 controlStepSchema 배열로 재검증**된다(CLAUDE.md §2.8
// 정신 — 검증 없이는 실행에 노출하지 않는다). 모든 실패는 throw가 아니라
// { ok:false, errors[](한국어) } 유니온으로 반환한다 — UI가 그대로 표시한다.
//
// ── binding 경로 규약 ──────────────────────────────────────────────
// BlockParam.bindings[].path는 steps[stepIndex] **step 객체 기준** 경로다.
// - 'durationSec'          → step.durationSec        (1단계)
// - 'targets.joint1'       → step.targets.joint1     (2단계 — 최대 깊이)
// - 'params.durationSec'   → step.durationSec        (Flow Graph 뷰(FlowNode.params)
//   관례의 별칭 — 선두 'params.' 구획은 step 루트를 가리킨다. entities.ts의 예시 표기)
// 배열 내부(between[0] 등)·kind 치환·미지의 필드는 거부한다 — zod strip이 모르는 키를
// 조용히 벗겨 "대입했는데 아무 일도 없는" 무음 no-op이 되는 것을 막는다.

import { z } from 'zod';
import type { ControlStep, ControlStepKind } from './types';
import { controlStepSchema } from './validate';
import { blockDocSchema } from './entities';
import type { BlockDoc, BlockParam } from './entities';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4) ─────────────────────────────

/** binding 경로 최대 깊이 ('params.' 별칭 정규화 후 구획 수) */
export const BINDING_PATH_MAX_SEGMENTS = 2;
/** Flow Graph 뷰 관례의 경로 별칭 — 선두 구획이 이 값이면 step 루트로 해석한다 */
export const BINDING_PARAMS_ALIAS = 'params';
/** robotHint 최대 길이 (entities.ts blockDocSchema.robotHint의 max와 동일) */
export const ROBOT_HINT_MAX_CHARS = 60;

/** robot 필드를 갖는 step 종류 — 전개 시 targetRobotId 일괄 재매핑 대상 */
export const ROBOT_STEP_KINDS: readonly ControlStepKind[] = [
  'moveJoints',
  'setJoints',
  'gripper',
  'moveToPose',
];

/**
 * step 종류별 binding 허용 필드 (kind 제외 — 종류 치환은 전개가 아니라 다른 블록이다).
 * Record<ControlStepKind, ...>이므로 새 step kind를 추가하면 **여기가 컴파일에서 깨진다**
 * — flow-graph.ts의 STEP_KIND_FLAGS와 같은 동기화 장치.
 */
const STEP_BINDABLE_FIELDS: Readonly<Record<ControlStepKind, readonly string[]>> = {
  moveJoints: ['robot', 'targets', 'durationSec', 'easing'],
  setJoints: ['robot', 'targets'],
  gripper: ['robot', 'state', 'durationSec'],
  wait: ['durationSec'],
  waitForCollision: ['between', 'timeoutSec'],
  label: ['name'],
  goto: ['label', 'times'],
  moveToPose: ['robot', 'target', 'durationSec'],
};

/** 모든 step 공통(StepCommon) binding 허용 필드 */
const COMMON_BINDABLE_FIELDS: readonly string[] = ['enabled', 'note'];

// ── 결과 타입 ───────────────────────────────────────────────────────

export type CaptureBlockResult =
  | { readonly ok: true; readonly block: BlockDoc }
  | { readonly ok: false; readonly errors: string[] };

export type ExpandBlockResult =
  | { readonly ok: true; readonly steps: ControlStep[] }
  | { readonly ok: false; readonly errors: string[] };

export interface CaptureBlockOptions {
  readonly name: string;
  readonly descriptionKo?: string;
  /**
   * 로봇 힌트. **미지정(undefined)이면 steps에서 수집한 첫 로봇 id**가 힌트가 된다
   * (임무 명세 — "robot 참조를 수집해 첫 로봇을 기준으로 기록"). 명시적 null은
   * "힌트 없음"으로 존중한다.
   */
  readonly robotHint?: string | null;
  /**
   * 개체 id — 미지정 시 crypto.randomUUID() (BACKEND §4: id는 클라이언트 발급).
   * 테스트/호출자 주입점.
   */
  readonly id?: string;
}

export interface ExpandBlockOptions {
  /**
   * 전개 대상 로봇. 문자열이면 robot 필드를 갖는 모든 step에 **일괄 재매핑**된다
   * (robot이 없던 step에도 명시적으로 박힌다 — 삽입 결과가 시퀀스 기본 로봇에
   * 조용히 좌우되지 않게). null이면 캡처 당시 robot 참조를 그대로 유지한다.
   */
  readonly targetRobotId: string | null;
  /**
   * 파라미터 값 (key → 값). 없는 key는 defaultValue를 쓴다. block.params에 없는
   * 잉여 key는 오류다(무음 무시 금지). 재매핑 후에 대입되므로 'robot' 경로 binding은
   * targetRobotId보다 우선한다(파라미터가 더 구체적인 의도다).
   */
  readonly paramValues: Record<string, unknown>;
}

// ── 한국어 오류 메시지 (validate.ts의 koreanErrorMap은 비공개 — 최소 부분집합 소유) ──

const koreanErrorMap: z.ZodErrorMap = (issue, ctx) => {
  switch (issue.code) {
    case z.ZodIssueCode.invalid_type:
      if (issue.received === z.ZodParsedType.undefined) {
        return { message: `필수 항목이 누락되었습니다 (기대: ${issue.expected})` };
      }
      return {
        message: `잘못된 타입입니다 (기대: ${issue.expected}, 실제: ${issue.received})`,
      };
    case z.ZodIssueCode.invalid_union:
      return { message: '허용되는 값 형식이 아닙니다' };
    case z.ZodIssueCode.invalid_union_discriminator:
      return {
        message: `허용되지 않는 값입니다 (허용: ${issue.options.map(String).join(', ')})`,
      };
    case z.ZodIssueCode.invalid_enum_value:
      return {
        message: `허용되지 않는 값 '${String(issue.received)}'입니다 (허용: ${issue.options.join(', ')})`,
      };
    case z.ZodIssueCode.too_small: {
      const min = String(issue.minimum);
      if (issue.type === 'array') return { message: `최소 ${min}개의 항목이 필요합니다` };
      if (issue.type === 'string') return { message: `최소 ${min}자 이상이어야 합니다` };
      return {
        message: issue.inclusive ? `${min} 이상이어야 합니다` : `${min}보다 커야 합니다`,
      };
    }
    case z.ZodIssueCode.too_big: {
      const max = String(issue.maximum);
      if (issue.type === 'array') return { message: `최대 ${max}개의 항목만 허용됩니다` };
      if (issue.type === 'string') return { message: `최대 ${max}자까지 허용됩니다` };
      return {
        message: issue.inclusive ? `${max} 이하여야 합니다` : `${max}보다 작아야 합니다`,
      };
    }
    case z.ZodIssueCode.custom:
      return { message: issue.message ?? '유효하지 않은 값입니다' };
    default:
      return { message: ctx.defaultError };
  }
};

/** zod 이슈 경로 → "steps[0].durationSec" 형태 (validate.ts와 동일 표기) */
function formatPath(path: readonly (string | number)[]): string {
  if (path.length === 0) return '(root)';
  let out = '';
  for (const seg of path) {
    if (typeof seg === 'number') out += `[${seg}]`;
    else out += out === '' ? seg : `.${seg}`;
  }
  return out;
}

function formatIssues(
  issues: readonly z.ZodIssue[],
  pathPrefix: readonly (string | number)[] = [],
): string[] {
  return issues.map(
    (issue) => `${formatPath([...pathPrefix, ...issue.path])}: ${issue.message}`,
  );
}

// ── 순수 헬퍼 ───────────────────────────────────────────────────────

/** steps에 등장하는 robot 참조를 등장 순서대로(중복 제거) 수집한다 */
export function collectRobotIds(steps: readonly ControlStep[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const step of steps) {
    if ('robot' in step && typeof step.robot === 'string' && !seen.has(step.robot)) {
      seen.add(step.robot);
      out.push(step.robot);
    }
  }
  return out;
}

/**
 * 전개 흔적(blockRef) 표시용 라벨. 전개된 노드의 note/카드 배지 등에 통합자가 쓴다 —
 * 실행 의미에는 어떤 영향도 없다(step 스키마에 blockRef 필드를 넣지 않는 이유:
 * zod strip이 벗겨내는 임시 키는 저장·재검증을 오가며 조용히 사라진다).
 */
export function blockRefLabel(block: Pick<BlockDoc, 'id' | 'name'>): string {
  return `블록 '${block.name}'에서 전개 (block:${block.id})`;
}

export type BindingPathParse =
  | { readonly ok: true; readonly head: string; readonly tail: string | null }
  | { readonly ok: false; readonly errorKo: string };

/**
 * binding 경로 해석 (파일 헤더의 경로 규약). 선두 'params.' 구획은 Flow Graph 뷰
 * 별칭으로 제거되고, 남은 구획은 1~2단계여야 한다.
 */
export function parseBindingPath(path: string): BindingPathParse {
  const raw = path.split('.');
  if (raw.some((segment) => segment === '')) {
    return { ok: false, errorKo: `경로 '${path}'에 빈 구획이 있습니다` };
  }
  const segments = raw[0] === BINDING_PARAMS_ALIAS ? raw.slice(1) : raw;
  const head = segments[0];
  if (head === undefined) {
    return { ok: false, errorKo: `경로 '${path}'가 대입 지점을 가리키지 않습니다` };
  }
  if (segments.length > BINDING_PATH_MAX_SEGMENTS) {
    return {
      ok: false,
      errorKo: `경로 '${path}'가 최대 깊이(${BINDING_PATH_MAX_SEGMENTS}단계)를 초과합니다`,
    };
  }
  if (head === 'kind') {
    return {
      ok: false,
      errorKo: `경로 '${path}'는 step 종류(kind)를 바꿀 수 없습니다 — 종류가 다르면 다른 블록입니다`,
    };
  }
  return { ok: true, head, tail: segments[1] ?? null };
}

export type ParamValueCheck =
  | { readonly ok: true; readonly value: number | string | boolean }
  | { readonly ok: false; readonly errorKo: string };

function describeType(value: unknown): string {
  if (value === null) return 'null';
  if (Array.isArray(value)) return 'array';
  return typeof value;
}

/** 파라미터 값의 종류(kind)·범위(min/max) 검사 — 타입 불일치는 오류(무음 강제 변환 금지) */
export function checkParamValue(param: BlockParam, value: unknown): ParamValueCheck {
  const label = `파라미터 '${param.key}' (${param.labelKo})`;
  const kindOk =
    (param.kind === 'number' && typeof value === 'number' && Number.isFinite(value)) ||
    (param.kind === 'string' && typeof value === 'string') ||
    (param.kind === 'boolean' && typeof value === 'boolean');
  if (!kindOk) {
    return {
      ok: false,
      errorKo: `${label}: ${param.kind} 값이 필요합니다 (실제: ${describeType(value)})`,
    };
  }
  if (param.kind === 'number' && typeof value === 'number') {
    if (param.min !== undefined && value < param.min) {
      return { ok: false, errorKo: `${label}: ${param.min} 이상이어야 합니다 (실제: ${value})` };
    }
    if (param.max !== undefined && value > param.max) {
      return { ok: false, errorKo: `${label}: ${param.max} 이하여야 합니다 (실제: ${value})` };
    }
  }
  return { ok: true, value: value as number | string | boolean };
}

/** step kind에 대해 binding head 필드가 허용되는지 검사 */
function checkBindableField(kind: string, head: string): string | null {
  const fields = (STEP_BINDABLE_FIELDS as Record<string, readonly string[] | undefined>)[kind];
  if (fields === undefined) return `알 수 없는 step 종류 '${kind}'입니다`;
  if (!fields.includes(head) && !COMMON_BINDABLE_FIELDS.includes(head)) {
    const allowed = [...fields, ...COMMON_BINDABLE_FIELDS].join(', ');
    return `step 종류 '${kind}'에 '${head}' 필드가 없습니다 (허용: ${allowed})`;
  }
  return null;
}

/** 해석된 경로에 값을 대입한다 (깊은 복사본을 제자리 변형). 실패 시 한국어 사유. */
function applyBinding(
  step: ControlStep,
  head: string,
  tail: string | null,
  value: number | string | boolean,
): string | null {
  // ControlStep은 닫힌 유니온이라 동적 경로 대입은 Record 관점이 필요하다 — 어떤 키가
  // 허용되는지는 checkBindableField가, 값의 정합성은 재검증(controlStepSchema)이 보증한다.
  const record = step as unknown as Record<string, unknown>;
  if (tail === null) {
    record[head] = value;
    return null;
  }
  const container = record[head];
  if (container === null || typeof container !== 'object' || Array.isArray(container)) {
    return `'${head}'이(가) 객체가 아니어서 '${head}.${tail}'에 대입할 수 없습니다`;
  }
  (container as Record<string, unknown>)[tail] = value;
  return null;
}

// ── captureBlock — 선택한 step들을 재사용 블록으로 갈무리 ────────────

/**
 * steps를 깊은 복사해 BlockDoc으로 만든다. robot 참조를 수집해 **첫 로봇을 힌트로
 * 기록**한다(opts.robotHint 명시가 항상 우선). 결과는 blockDocSchema 전체 검증을
 * 통과해야만 ok다 — 빈 블록·무효 step·빈 이름은 여기서 거부된다.
 */
export function captureBlock(
  steps: readonly ControlStep[],
  opts: CaptureBlockOptions,
): CaptureBlockResult {
  if (steps.length === 0) {
    return {
      ok: false,
      errors: ['빈 블록은 만들 수 없습니다 — 최소 1개의 step이 필요합니다'],
    };
  }
  const copies = steps.map((step) => structuredClone(step));
  const firstRobot = collectRobotIds(copies)[0] ?? null;
  const robotHint =
    opts.robotHint !== undefined
      ? opts.robotHint
      : firstRobot === null
        ? null
        : firstRobot.slice(0, ROBOT_HINT_MAX_CHARS); // 스키마 max(60) — 잘라서 힌트로

  const parsed = blockDocSchema.safeParse(
    {
      id: opts.id ?? crypto.randomUUID(),
      name: opts.name,
      descriptionKo: opts.descriptionKo ?? '',
      steps: copies,
      params: [],
      robotHint,
    },
    { errorMap: koreanErrorMap },
  );
  if (!parsed.success) return { ok: false, errors: formatIssues(parsed.error.issues) };
  return { ok: true, block: parsed.data };
}

// ── expandBlock — 블록을 일반 step 배열로 전개 ──────────────────────

/**
 * 블록을 ControlStep[]로 전개한다: 깊은 복사 → robot 일괄 재매핑(targetRobotId) →
 * 파라미터 binding 대입(paramValues, 재매핑보다 우선) → **controlStepSchema 배열
 * 재검증**. 어떤 실패도 부분 결과 없이 { ok:false, errors[] }다 — 반쯤 전개된 step을
 * 그래프에 흘리지 않는다.
 */
export function expandBlock(block: BlockDoc, opts: ExpandBlockOptions): ExpandBlockResult {
  if (block.steps.length === 0) {
    return { ok: false, errors: ['빈 블록은 전개할 수 없습니다 — step이 없습니다'] };
  }
  const errors: string[] = [];
  const steps = block.steps.map((step) => structuredClone(step));

  // 1) robot 일괄 재매핑 — robot 필드를 갖는 모든 step에 명시적으로 기록한다.
  if (opts.targetRobotId !== null) {
    for (const step of steps) {
      switch (step.kind) {
        case 'moveJoints':
        case 'setJoints':
        case 'gripper':
        case 'moveToPose':
          step.robot = opts.targetRobotId;
          break;
        default:
          break; // robot 필드 없는 종류 (ROBOT_STEP_KINDS 바깥)
      }
    }
  }

  // 2) 파라미터 대입 — 잉여 key는 오류 (무음 무시 금지)
  const knownKeys = new Set(block.params.map((param) => param.key));
  for (const key of Object.keys(opts.paramValues)) {
    if (!knownKeys.has(key)) {
      errors.push(`알 수 없는 파라미터 '${key}'입니다 — 블록에 정의되지 않았습니다`);
    }
  }

  for (const param of block.params) {
    const provided = Object.prototype.hasOwnProperty.call(opts.paramValues, param.key)
      ? opts.paramValues[param.key]
      : param.defaultValue;
    const checked = checkParamValue(param, provided);
    if (!checked.ok) {
      errors.push(checked.errorKo);
      continue;
    }
    param.bindings.forEach((binding, bindingIndex) => {
      const where = `파라미터 '${param.key}' bindings[${bindingIndex}]`;
      const target = steps[binding.stepIndex];
      if (target === undefined) {
        errors.push(
          `${where}: stepIndex ${binding.stepIndex}이(가) 범위(0..${steps.length - 1})를 벗어났습니다`,
        );
        return;
      }
      const parsedPath = parseBindingPath(binding.path);
      if (!parsedPath.ok) {
        errors.push(`${where}: ${parsedPath.errorKo}`);
        return;
      }
      const fieldError = checkBindableField(target.kind, parsedPath.head);
      if (fieldError !== null) {
        errors.push(`${where}: ${fieldError}`);
        return;
      }
      const applyError = applyBinding(target, parsedPath.head, parsedPath.tail, checked.value);
      if (applyError !== null) errors.push(`${where}: ${applyError}`);
    });
  }

  if (errors.length > 0) return { ok: false, errors };

  // 3) 재검증 (§2.8 — 전개 결과는 검증을 통과해야만 실행/편집 표면에 나간다)
  const out: ControlStep[] = [];
  steps.forEach((step, index) => {
    const parsed = controlStepSchema.safeParse(step, { errorMap: koreanErrorMap });
    if (parsed.success) out.push(parsed.data);
    else errors.push(...formatIssues(parsed.error.issues, ['steps', index]));
  });
  if (errors.length > 0) return { ok: false, errors };
  return { ok: true, steps: out };
}
