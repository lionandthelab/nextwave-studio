// planner/adapters/rule-based.ts — 결정론적 오프라인 플래너 (데모/게이트 기본값).
// 규범: docs/PLANNER.md. LLM 없이 한국어+영어 명령을 SceneContext에 그라운딩해
// ControlSequence 초안을 만든다. LlmAdapter가 아니라 직접 Planner 구현이다 —
// planner.ts(PlannerService)가 backend에 따라 이 구현으로 라우팅한다.
//
// 순수 함수형: SceneContext(그라운딩된 씬) + 자연어 → PlannerResult. 물리/렌더/DOM
// 의존 없음. 생성한 sequence는 구성상 항상 유효하며, PlannerService가 방어적으로
// validateSequence를 한 번 더 통과시킨다(불변식 §2.9 — 미검증 출력 실행 금지).

import type { ControlStep, ControlSequence } from '../../schema';
import type { PlannerResult } from '../llm-adapter';
import type { SceneContext, SceneContextObject, SceneContextRobot } from '../scene-context';

/** UI 배지/선택용 정직한 라벨 — 이 플래너는 LLM이 아님을 드러낸다 */
export const RULE_BASED_LABEL = '규칙 기반 (오프라인 데모)';

/** 규칙 기반이 만드는 시퀀스의 id */
export const RULE_BASED_SEQUENCE_ID = 'planned';

// ── step 지속시간/타임아웃 상수 (매직넘버 금지 — CLAUDE.md §4) ────────
const REACH_DURATION_SEC = 2.0;
const HOME_DURATION_SEC = 2.0;
const GRIPPER_OPEN_DURATION_SEC = 0.4;
const GRIPPER_CLOSE_DURATION_SEC = 0.5;
const SETTLE_WAIT_SEC = 0.3;
const COLLISION_TIMEOUT_SEC = 6;
const DEFAULT_WAIT_SEC = 1;

// ── 도달 자세 휴리스틱 밴드 경계 (반경 r = hypot(x,z), 미터) ──────────
const REACH_NEAR_R = 0.3;
const REACH_MID_R = 0.45;

// ── 명령 키워드 (한국어 stem + 영어) ─────────────────────────────────
const TOUCH_KEYWORDS = [
  '집어', '집는', '집기', '집을', '집게',
  '잡아', '잡는', '잡기', '잡을',
  '터치', '눌러', '누르', '만져', '만지',
  'pick', 'grab', 'grasp', 'touch',
] as const;
const OPEN_KEYWORDS = ['열어', '열기', '펼쳐', 'open'] as const;
const CLOSE_KEYWORDS = ['닫아', '닫기', '오므', '쥐어', '쥐기', 'close'] as const;
const WAIT_KEYWORDS = ['기다려', '기다리', '대기', 'wait'] as const;
const HOME_KEYWORDS = ['홈', '원위치', '초기 자세', '초기자세', '처음', 'home'] as const;

/** 다중 절 구분자 (~하고/그리고/then/쉼표). PLANNER 규약의 접속사만 분리한다. */
const CLAUSE_SEPARATORS =
  /\s*(?:그리고|그다음에|그다음|그 다음에|그 다음|하고서|하고|then|,|;)\s*/gi;

/** 사용자가 clarify 옵션을 고른 뒤 UI가 되돌려 붙이는 토큰: "... [선택: box_a]" */
const SELECTION_TOKEN = /\[선택:\s*([^\]]+)\]/;

/** 해석 불가 시 사용자에게 보이는 지원 패턴 안내 (한국어) */
export const UNDERSTAND_ERROR_MESSAGE = [
  '이해하지 못했습니다. 지원하는 명령 예시:',
  '- "박스를 집어" · "box_a를 잡아" · "터치해" (집기/접촉)',
  '- "그리퍼 열어" · "닫아" (그리퍼)',
  '- "3초 기다려" (대기)',
  '- "홈으로" · "원위치" (홈 자세)',
  '여러 동작은 "그리고"/"하고"/","로 이어 붙일 수 있습니다.',
].join('\n');

// ── 공개 API ─────────────────────────────────────────────────────────

/**
 * 결정론적 규칙 기반 플래너. planner.ts가 소유하고 라우팅한다.
 * generate는 동기 함수지만 PlannerService.generate가 Promise로 감싼다.
 */
export class RuleBasedPlanner {
  readonly name = RULE_BASED_LABEL;

  generate(nl: string, ctx: SceneContext): PlannerResult {
    return generateRuleBased(nl, ctx);
  }
}

/**
 * 자연어 + SceneContext → PlannerResult (결정론).
 * 다중 절은 각각 계획해 step을 이어 붙인다. 한 절이라도 clarify/error면 전체를 그 결과로
 * 반환한다(부분 성공 미채택 — PLANNER.md §4.2).
 */
export function generateRuleBased(nl: string, ctx: SceneContext): PlannerResult {
  if (ctx.robots.length === 0) {
    return {
      type: 'error',
      message: '이 씬에는 로봇이 없어 제어 시퀀스를 만들 수 없습니다. 먼저 로봇을 추가하세요.',
    };
  }

  const { text, selectedId } = extractSelection(nl);
  const clauses = splitClauses(text);

  const steps: ControlStep[] = [];
  const assumptions: string[] = [];
  let matchedAny = false;

  for (const clause of clauses) {
    const result = planClause(clause, ctx, selectedId);
    if (result.type === 'clarify' || result.type === 'error') return result;
    matchedAny = true;
    for (const step of result.steps) steps.push(step);
    for (const assumption of result.assumptions) {
      if (!assumptions.includes(assumption)) assumptions.push(assumption);
    }
  }

  if (!matchedAny || steps.length === 0) {
    return { type: 'error', message: UNDERSTAND_ERROR_MESSAGE };
  }

  const robotId = ctx.robots[0]!.id;
  const sequence: ControlSequence = { id: RULE_BASED_SEQUENCE_ID, robot: robotId, steps };
  const result: PlannerResult = { type: 'sequence', sequence };
  if (assumptions.length > 0) result.assumptions = assumptions;
  return result;
}

// ── 절 파싱 ──────────────────────────────────────────────────────────

/** 성공 절은 step 목록 + 가정을, 실패 절은 clarify/error를 낸다 */
type ClauseResult =
  | { type: 'steps'; steps: ControlStep[]; assumptions: string[] }
  | { type: 'clarify'; question: string; options?: string[] }
  | { type: 'error'; message: string };

function planClause(rawClause: string, ctx: SceneContext, selectedId: string | null): ClauseResult {
  const clause = rawClause.trim();
  if (clause === '') return { type: 'steps', steps: [], assumptions: [] };

  const norm = clause.toLowerCase();
  const has = (keywords: readonly string[]): boolean => keywords.some((k) => norm.includes(k));

  // 우선순위: 집기(복합) > 홈 > 그리퍼 열기/닫기 > 대기.
  if (has(TOUCH_KEYWORDS)) return planTouch(clause, ctx, selectedId);
  if (has(HOME_KEYWORDS)) return planHome(ctx);
  if (has(OPEN_KEYWORDS)) return planGripper('open');
  if (has(CLOSE_KEYWORDS)) return planGripper('close');
  if (has(WAIT_KEYWORDS)) return planWait(clause);

  return { type: 'error', message: UNDERSTAND_ERROR_MESSAGE };
}

/**
 * '<대상>을 집어/잡아/터치' → open → approach(moveJoints) → nudge(setJoints) →
 * waitForCollision → close → settle → home.
 *
 * 접근/눌러내림을 2단으로 나누는 것은 결정론적 접촉을 위한 계약이다 (arm-touch-box와
 * 동일 패턴): waitForCollision 배리어는 init 시점 마커 "이후"의 충돌만 감지한다
 * (steps.ts). 단일 moveJoints로 박스에 닿으면 접촉 start가 배리어 시작 "전"에 나
 * happenedSince가 놓치고 timeout으로 흐른다. 따라서 moveJoints로 박스 바로 위까지
 * 접근한 뒤, setJoints로 눌러 내려 접촉을 배리어 직후에 발생시켜 이벤트로 해제한다.
 */
function planTouch(clause: string, ctx: SceneContext, selectedId: string | null): ClauseResult {
  const robot = ctx.robots[0]!;
  const resolution = resolveTarget(clause, ctx, selectedId);
  if (resolution.kind === 'none') {
    return { type: 'error', message: '이 씬에는 집을 수 있는 사물이 없습니다.' };
  }
  if (resolution.kind === 'clarify') {
    return { type: 'clarify', question: '어느 것을 집을까요?', options: resolution.options };
  }

  const targetId = resolution.id;
  const target = ctx.objects.find((object) => object.id === targetId)!;
  const approachTargets = reachPose(target, robot);
  const contactTargets = contactNudge(target, robot);
  const homeTargets = homePose(robot);

  const steps: ControlStep[] = [
    { kind: 'gripper', state: 'open', durationSec: GRIPPER_OPEN_DURATION_SEC },
    { kind: 'moveJoints', targets: approachTargets, durationSec: REACH_DURATION_SEC, easing: 'easeInOut' },
    { kind: 'setJoints', targets: contactTargets },
    { kind: 'waitForCollision', between: [robot.id, targetId], timeoutSec: COLLISION_TIMEOUT_SEC },
    { kind: 'gripper', state: 'close', durationSec: GRIPPER_CLOSE_DURATION_SEC },
    { kind: 'wait', durationSec: SETTLE_WAIT_SEC },
    { kind: 'moveJoints', targets: homeTargets, durationSec: HOME_DURATION_SEC, easing: 'easeInOut' },
  ];
  const assumptions = [`${targetId}를 대상으로 해석`, '도달 자세는 휴리스틱(정확한 IK 아님)'];
  return { type: 'steps', steps, assumptions };
}

function planHome(ctx: SceneContext): ClauseResult {
  const robot = ctx.robots[0]!;
  const targets = homePose(robot);
  return {
    type: 'steps',
    steps: [{ kind: 'moveJoints', targets, durationSec: HOME_DURATION_SEC, easing: 'easeInOut' }],
    assumptions: [],
  };
}

function planGripper(state: 'open' | 'close'): ClauseResult {
  const durationSec = state === 'open' ? GRIPPER_OPEN_DURATION_SEC : GRIPPER_CLOSE_DURATION_SEC;
  return { type: 'steps', steps: [{ kind: 'gripper', state, durationSec }], assumptions: [] };
}

function planWait(clause: string): ClauseResult {
  const seconds = parseSeconds(clause) ?? DEFAULT_WAIT_SEC;
  return { type: 'steps', steps: [{ kind: 'wait', durationSec: seconds }], assumptions: [] };
}

// ── 대상 물체 해석 ───────────────────────────────────────────────────

type TargetResolution =
  | { kind: 'resolved'; id: string }
  | { kind: 'clarify'; options: string[] }
  | { kind: 'none' };

/**
 * 절에서 조작 대상 물체를 결정한다.
 * - clarify 후 되돌아온 선택(selectedId) 우선. 유효하지 않으면 available ids로 다시 clarify.
 * - 절에 물체 id가 명시되면 사용(2개 이상이면 clarify).
 * - id처럼 보이나 씬에 없는 토큰(밑줄 포함)이 있으면 available ids로 clarify.
 * - 그 외: 물체가 정확히 1개면 그것으로, 여러 개면 clarify.
 */
function resolveTarget(
  clause: string,
  ctx: SceneContext,
  selectedId: string | null,
): TargetResolution {
  const objectIds = ctx.objects.map((object) => object.id);
  if (objectIds.length === 0) return { kind: 'none' };

  if (selectedId !== null) {
    if (objectIds.includes(selectedId)) return { kind: 'resolved', id: selectedId };
    return { kind: 'clarify', options: objectIds };
  }

  const mentioned = ctx.objects.filter((object) => clause.includes(object.id));
  if (mentioned.length === 1) return { kind: 'resolved', id: mentioned[0]!.id };
  if (mentioned.length > 1) return { kind: 'clarify', options: mentioned.map((o) => o.id) };

  // 밑줄 포함 id 형태의 미지 토큰(존재하지 않는 대상 지정) → clarify
  const idLike = clause.match(/[A-Za-z][A-Za-z0-9_]*/g) ?? [];
  const unknown = idLike.filter((token) => token.includes('_') && !objectIds.includes(token));
  if (unknown.length > 0) return { kind: 'clarify', options: objectIds };

  if (objectIds.length === 1) return { kind: 'resolved', id: objectIds[0]! };
  return { kind: 'clarify', options: objectIds };
}

// ── 도달 자세 휴리스틱 (arm6 6-DOF 기하, 문서화된 근사) ───────────────

/**
 * 접근(approach) 자세 → 관절 목표(joint1/2/3/5). 정확한 IK가 아니라 arm6 팔 형상에 맞춘 휴리스틱.
 * - joint1 = -atan2(z, x): 베이스 회전으로 물체 방위각을 향한다. (검증: box_a[0.35,·,0.15] →
 *   -atan2(0.15,0.35) ≈ -0.405, arm-touch-box.sequence.json과 일치)
 * - r = hypot(x, z): 수평 도달 반경. near/mid/far 밴드로 어깨/팔꿈치/손목 각을 고른다.
 * mid 밴드(0.3≤r<0.45)의 접근 값(joint2 0.639 · joint3 1.414 · joint5 1.089)은
 * arm-touch-box.sequence.json의 검증된 접근 자세다 — 박스 바로 위까지 접근한다. 이어지는
 * contactNudge(setJoints)가 눌러 내려 배리어 직후 접촉을 발생시킨다. box_a/box_b 모두 이 밴드.
 * 모든 값은 컨텍스트 limits로 클램프하고, 로봇에 없는 관절은 건너뛴다(검증 통과 보장).
 */
function reachPose(target: SceneContextObject, robot: SceneContextRobot): Record<string, number> {
  const [x, , z] = target.position;
  const joint1 = -Math.atan2(z, x);
  const r = Math.hypot(x, z);

  let shoulder: number;
  let elbow: number;
  let wrist: number;
  if (r < REACH_NEAR_R) {
    shoulder = 0.5;
    elbow = 1.85;
    wrist = 1.16;
  } else if (r < REACH_MID_R) {
    // arm-touch-box 검증 접근 자세 (box_a/box_b 모두 이 밴드)
    shoulder = 0.639;
    elbow = 1.414;
    wrist = 1.089;
  } else {
    shoulder = 0.78;
    elbow = 1.2;
    wrist = 0.95;
  }

  const raw: Record<string, number> = { joint1, joint2: shoulder, joint3: elbow, joint5: wrist };
  return clampToRobot(raw, robot);
}

/**
 * 접촉(nudge) 자세 → 관절 목표(joint2/3/5). setJoints로 즉시 적용해 그리퍼를 박스 위로
 * 눌러 내린다 — joint1(방위각)은 접근에서 이미 맞췄으므로 건드리지 않는다.
 * mid 밴드 값(joint2 0.683 · joint3 1.442 · joint5 1.017)은 arm-touch-box의 검증된 접촉
 * 자세다. 이 눌러 내림이 waitForCollision 배리어 init "직후"에 접촉 start를 만들어
 * happenedSince(mark)가 이벤트로 배리어를 해제하게 한다 (timeout 경로 회피).
 */
function contactNudge(target: SceneContextObject, robot: SceneContextRobot): Record<string, number> {
  const r = Math.hypot(target.position[0], target.position[2]);

  let shoulder: number;
  let elbow: number;
  let wrist: number;
  if (r < REACH_NEAR_R) {
    shoulder = 0.55;
    elbow = 1.9;
    wrist = 1.08;
  } else if (r < REACH_MID_R) {
    // arm-touch-box 검증 접촉 자세 (box_a/box_b 모두 이 밴드)
    shoulder = 0.683;
    elbow = 1.442;
    wrist = 1.017;
  } else {
    shoulder = 0.82;
    elbow = 1.2;
    wrist = 0.9;
  }

  const raw: Record<string, number> = { joint2: shoulder, joint3: elbow, joint5: wrist };
  return clampToRobot(raw, robot);
}

/** 홈 자세: homePose가 있으면 그대로, 없으면 reach 관절을 0으로 (검증 통과용 fallback) */
function homePose(robot: SceneContextRobot): Record<string, number> {
  if (Object.keys(robot.homePose).length > 0) return { ...robot.homePose };
  const fallback: Record<string, number> = {};
  for (const name of ['joint1', 'joint2', 'joint3', 'joint5']) {
    if (robot.joints.some((joint) => joint.name === name)) fallback[name] = 0;
  }
  return fallback;
}

/** 목표값을 각 관절의 컨텍스트 limits로 클램프하고, 로봇에 없는 관절은 제외한다 */
function clampToRobot(
  targets: Record<string, number>,
  robot: SceneContextRobot,
): Record<string, number> {
  const out: Record<string, number> = {};
  for (const [name, value] of Object.entries(targets)) {
    const joint = robot.joints.find((j) => j.name === name);
    if (!joint) continue;
    out[name] = clamp(value, joint.limits[0], joint.limits[1]);
  }
  return out;
}

function clamp(value: number, lo: number, hi: number): number {
  return Math.min(Math.max(value, lo), hi);
}

// ── 텍스트 유틸 ──────────────────────────────────────────────────────

/** "... [선택: box_a]" 토큰을 떼어내 selectedId로 반환하고 본문에서 제거한다 */
function extractSelection(nl: string): { text: string; selectedId: string | null } {
  const match = nl.match(SELECTION_TOKEN);
  if (!match) return { text: nl, selectedId: null };
  const selectedId = match[1]!.trim();
  const text = nl.replace(match[0], ' ').trim();
  return { text, selectedId };
}

function splitClauses(text: string): string[] {
  return text
    .split(CLAUSE_SEPARATORS)
    .map((part) => part.trim())
    .filter((part) => part.length > 0);
}

/** 절에서 첫 숫자를 초 단위로 뽑는다 ("3초 기다려" → 3, "wait 2.5" → 2.5) */
function parseSeconds(clause: string): number | null {
  const match = clause.match(/(\d+(?:\.\d+)?)/);
  if (!match) return null;
  const value = Number.parseFloat(match[1]!);
  return Number.isFinite(value) ? value : null;
}
