// planner/index.ts — 플래너 계층 배럴.
// 규범: docs/PLANNER.md. 계층 규칙: planner → schema (+ adapters/anthropic만 SDK).
// ui 계층은 이 배럴에서 import한다.

export type {
  SceneContext,
  SceneContextJoint,
  SceneContextGripper,
  SceneContextRobot,
  SceneContextObject,
  DirectionConventions,
  PlannerStepKind,
  WorldSnapshot,
} from './scene-context';
export {
  buildContext,
  DIRECTION_CONVENTIONS,
  PLANNER_STEP_KINDS,
  DEFAULT_REVOLUTE_LIMIT,
} from './scene-context';

export type { LlmAdapter, PlannerResult } from './llm-adapter';

export {
  PlannerService,
  buildSystemPrompt,
  buildUserPrompt,
} from './planner';
export type { PlannerBackend, PlannerServiceOptions } from './planner';

export {
  RuleBasedPlanner,
  generateRuleBased,
  RULE_BASED_LABEL,
  RULE_BASED_SEQUENCE_ID,
  UNDERSTAND_ERROR_MESSAGE,
} from './adapters/rule-based';

export { AnthropicAdapter, DEFAULT_ANTHROPIC_MODEL } from './adapters/anthropic';
export type { AnthropicAdapterOptions } from './adapters/anthropic';

export {
  validatePlannerOutput,
  extractJson,
  repairLoop,
  PLANNER_RESULT_JSON_SCHEMA,
  DEFAULT_MAX_REPAIR_ATTEMPTS,
} from './validate-repair';
export type { ValidatePlannerOutputResult } from './validate-repair';
