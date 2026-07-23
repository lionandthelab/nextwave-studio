// schema/validate.ts — SceneSpec / ControlSequence 런타임 검증 (규범: docs/DATA_MODEL.md §8)
//
// - zod 스키마는 schema/types.ts의 타입을 1:1로 미러링한다. 파일 하단의 MustAssign
//   검사가 "zod 출력 타입 → types.ts 타입" 대입 가능성을 컴파일 타임에 보장한다.
// - 구조 검증(zod) + 교차 필드 규칙(id 유일성, trimesh×bodyType, 참조 무결성 등)을
//   모두 사람이 읽을 수 있는 한국어 메시지("경로: 메시지")로 반환한다 — UI 직접 표시용.
// - 이 모듈은 순수하다: 물리/렌더/UI에 의존하지 않고 zod만 사용한다 (ARCHITECTURE §3).

import { z } from 'zod';
import type {
  BodyType,
  ColliderGroup,
  ColliderShape,
  ColliderSpec,
  ControlSequence,
  ControlStep,
  Easing,
  EntitySpec,
  PhysicsSpec,
  RobotSpec,
  SceneSpec,
  Transform,
  VisualSpec,
} from './types';
import { isRobotSpec } from './types';

// ── 상수 ────────────────────────────────────────────────────────────

/** SceneSpec 스키마 버전 (DATA_MODEL.md §5). version 필드는 이 값과 일치해야 한다. */
export const SCENE_SPEC_VERSION = 1;

/** ControlSequence가 가져야 하는 최소 step 수 */
const MIN_SEQUENCE_STEP_COUNT = 1;

// ── 결과 타입 ───────────────────────────────────────────────────────

export type ValidationResult<T> =
  | { ok: true; value: T }
  | { ok: false; errors: string[] };

// ── 한국어 오류 메시지 ──────────────────────────────────────────────

/**
 * zod 기본(영어) 메시지를 한국어로 대체하는 컨텍스트 에러 맵.
 * `.min()/.positive()` 등에 직접 지정한 메시지가 항상 우선한다.
 */
const koreanErrorMap: z.ZodErrorMap = (issue, ctx) => {
  switch (issue.code) {
    case z.ZodIssueCode.invalid_type:
      if (issue.received === z.ZodParsedType.undefined) {
        return { message: `필수 항목이 누락되었습니다 (기대: ${issue.expected})` };
      }
      return {
        message: `잘못된 타입입니다 (기대: ${issue.expected}, 실제: ${issue.received})`,
      };
    case z.ZodIssueCode.invalid_literal:
      return { message: `${JSON.stringify(issue.expected)} 값이어야 합니다` };
    case z.ZodIssueCode.invalid_enum_value:
      return {
        message: `허용되지 않는 값 '${String(issue.received)}'입니다 (허용: ${issue.options.join(', ')})`,
      };
    case z.ZodIssueCode.invalid_union_discriminator:
      return {
        message: `허용되지 않는 값입니다 (허용: ${issue.options.map(String).join(', ')})`,
      };
    case z.ZodIssueCode.invalid_union:
      return { message: '허용되는 값 형식이 아닙니다' };
    case z.ZodIssueCode.not_finite:
      return { message: '유한한 수여야 합니다' };
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
    case z.ZodIssueCode.unrecognized_keys:
      return { message: `허용되지 않는 키입니다: ${issue.keys.join(', ')}` };
    case z.ZodIssueCode.custom:
      return { message: issue.message ?? '유효하지 않은 값입니다' };
    default:
      return { message: ctx.defaultError };
  }
};

/** zod 이슈 경로 → "entities[2].physics.colliders[0].shape" 형태의 JSON 경로 문자열 */
function formatPath(path: readonly (string | number)[]): string {
  if (path.length === 0) return '(root)';
  let out = '';
  for (const seg of path) {
    if (typeof seg === 'number') out += `[${seg}]`;
    else out += out === '' ? seg : `.${seg}`;
  }
  return out;
}

function formatIssues(issues: readonly z.ZodIssue[]): string[] {
  return issues.map((issue) => `${formatPath(issue.path)}: ${issue.message}`);
}

// ── 1. 공통 원시 타입 ───────────────────────────────────────────────

const finiteNumber = z.number().finite();

const vec3Schema = z.tuple([finiteNumber, finiteNumber, finiteNumber]);
const quatSchema = z.tuple([finiteNumber, finiteNumber, finiteNumber, finiteNumber]);

export const transformSchema = z.object({
  position: vec3Schema,
  rotation: quatSchema.optional(),
  scale: vec3Schema.optional(),
});

// ── 2. Collider / Physics ───────────────────────────────────────────

export const colliderShapeSchema = z.discriminatedUnion('kind', [
  z.object({ kind: z.literal('box'), halfExtents: vec3Schema }),
  z.object({ kind: z.literal('sphere'), radius: finiteNumber }),
  z.object({ kind: z.literal('capsule'), halfHeight: finiteNumber, radius: finiteNumber }),
  z.object({ kind: z.literal('cylinder'), halfHeight: finiteNumber, radius: finiteNumber }),
  z.object({ kind: z.literal('convexHull'), ref: z.string() }),
  z.object({ kind: z.literal('trimesh'), ref: z.string() }),
  z.object({ kind: z.literal('fromVisual') }),
]);

export const bodyTypeSchema = z.enum([
  'dynamic',
  'fixed',
  'kinematicPosition',
  'kinematicVelocity',
]);

// CLAUDE.md §5 충돌 그룹 규약과 1:1
export const colliderGroupSchema = z.enum(['ENV', 'ROBOT', 'OBJECT', 'SENSOR_ZONE', 'DEBUG']);

export const colliderSpecSchema = z
  .object({
    shape: colliderShapeSchema,
    offset: transformSchema.optional(),
    density: finiteNumber.optional(),
    friction: finiteNumber.optional(),
    restitution: finiteNumber.optional(),
    isSensor: z.boolean().optional(),
    group: colliderGroupSchema,
    collidesWith: z.array(colliderGroupSchema),
    ccd: z.boolean().optional(),
    emitEvents: z.boolean().optional(),
  })
  .superRefine((collider, ctx) => {
    // 교차 규칙: 충돌 이벤트를 발행하려면 감지 대상 그룹이 최소 1개 필요 (DATA_MODEL §2)
    if (collider.emitEvents === true && collider.collidesWith.length === 0) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ['collidesWith'],
        message:
          'emitEvents가 true인 collider는 collidesWith에 상대 그룹을 1개 이상 지정해야 합니다',
      });
    }
  });

export const physicsSpecSchema = z
  .object({
    bodyType: bodyTypeSchema,
    colliders: z.array(colliderSpecSchema),
    linearDamping: finiteNumber.optional(),
    angularDamping: finiteNumber.optional(),
    gravityScale: finiteNumber.optional(),
  })
  .superRefine((physics, ctx) => {
    // 교차 규칙: trimesh는 정적(fixed) 바디 전용 (DATA_MODEL §2 규칙)
    if (physics.bodyType === 'fixed') return;
    physics.colliders.forEach((collider, index) => {
      if (collider.shape.kind === 'trimesh') {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ['colliders', index, 'shape'],
          message: `trimesh collider는 bodyType 'fixed'에서만 사용할 수 있습니다 (현재 bodyType: '${physics.bodyType}')`,
        });
      }
    });
  });

// ── 3. Visual ───────────────────────────────────────────────────────

export const visualSpecSchema = z.object({
  kind: z.enum(['urdf', 'mesh', 'primitive']),
  ref: z.string().optional(),
  primitive: colliderShapeSchema.optional(),
  color: z.string().optional(),
  packages: z.record(z.string(), z.string()).optional(),
});

// ── 4. Entity ───────────────────────────────────────────────────────

const entityIdSchema = z.string().min(1, 'id는 비어 있지 않은 문자열이어야 합니다');

/** robot/object/static 공통 필드 (type/visual은 변형별로 재정의) */
const entityBaseShape = {
  id: entityIdSchema,
  transform: transformSchema,
  visual: visualSpecSchema,
  physics: physicsSpecSchema.optional(),
  tags: z.array(z.string()).optional(),
};

const objectEntitySchema = z.object({ ...entityBaseShape, type: z.literal('object') });
const staticEntitySchema = z.object({ ...entityBaseShape, type: z.literal('static') });

/** 로봇 엔티티: urdf + controller 필수, visual.kind는 'urdf' 고정 (DATA_MODEL §4.1) */
export const robotSpecSchema = z.object({
  ...entityBaseShape,
  type: z.literal('robot'),
  visual: visualSpecSchema.extend({ kind: z.literal('urdf') }),
  urdf: z.string().min(1, 'urdf 경로는 비어 있지 않은 문자열이어야 합니다'),
  urdfPackages: z.record(z.string(), z.string()).optional(),
  jointMap: z.record(z.string(), z.string()).optional(),
  home: z.record(z.string(), finiteNumber).optional(),
  jointLimits: z.record(z.string(), z.tuple([finiteNumber, finiteNumber])).optional(),
  controller: z.enum(['sequence', 'manual']),
  linkColliders: z.enum(['fromVisual', 'primitive', 'none']).optional(),
  selfCollision: z.boolean().optional(),
  gripper: z
    .object({
      joints: z.array(z.string().min(1)).min(1, 'gripper.joints는 최소 1개 관절이 필요합니다'),
      open: finiteNumber,
      close: finiteNumber,
    })
    .optional(),
});

export const entitySpecSchema = z.discriminatedUnion('type', [
  robotSpecSchema,
  objectEntitySchema,
  staticEntitySchema,
]);

// ── 5. Scene ────────────────────────────────────────────────────────

export const sceneSpecSchema = z
  .object({
    name: z.string(),
    version: z.literal(SCENE_SPEC_VERSION),
    gravity: vec3Schema,
    timestepHz: z
      .number()
      .finite('timestepHz는 유한한 수여야 합니다')
      .positive('timestepHz는 0보다 커야 합니다'),
    entities: z.array(entitySpecSchema),
    camera: z
      .object({
        position: vec3Schema,
        target: vec3Schema,
        fov: finiteNumber.optional(),
      })
      .optional(),
    environment: z
      .object({
        ground: z.boolean().optional(),
        skyColor: z.string().optional(),
      })
      .optional(),
  })
  .superRefine((scene, ctx) => {
    // 교차 규칙: 엔티티 id는 씬 내에서 유일 (DATA_MODEL §4, §8)
    const firstIndexById = new Map<string, number>();
    scene.entities.forEach((entity, index) => {
      const firstIndex = firstIndexById.get(entity.id);
      if (firstIndex !== undefined) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ['entities', index, 'id'],
          message: `엔티티 id '${entity.id}'가 중복됩니다 (entities[${firstIndex}]와 동일) — 씬 내에서 유일해야 합니다`,
        });
      } else {
        firstIndexById.set(entity.id, index);
      }
    });
  });

// ── 6. Control Sequence ─────────────────────────────────────────────

export const easingSchema = z.enum(['linear', 'easeInOut', 'step']);

const jointTargetsSchema = z.record(z.string(), finiteNumber);

const durationSecSchema = z
  .number()
  .finite('durationSec는 유한한 수여야 합니다')
  .nonnegative('durationSec는 0 이상이어야 합니다');

const timeoutSecSchema = z
  .number()
  .finite('timeoutSec는 유한한 수여야 합니다')
  .positive('timeoutSec는 0보다 커야 합니다');

export const controlStepSchema = z.discriminatedUnion('kind', [
  z.object({
    kind: z.literal('moveJoints'),
    robot: entityIdSchema.optional(),
    targets: jointTargetsSchema,
    durationSec: durationSecSchema,
    easing: easingSchema.optional(),
  }),
  z.object({
    kind: z.literal('setJoints'),
    robot: entityIdSchema.optional(),
    targets: jointTargetsSchema,
  }),
  z.object({
    kind: z.literal('gripper'),
    robot: entityIdSchema.optional(),
    // 숫자 형태는 0..1 (0=close, 1=open — DATA_MODEL §6). 범위 밖 값은 런타임
    // clamp01(steps.ts)이 조용히 삼켜 데이터 오류를 가리므로 검증에서 미리 거부한다.
    state: z
      .union([z.literal('open'), z.literal('close'), finiteNumber])
      .superRefine((state, ctx) => {
        if (typeof state === 'number' && (state < 0 || state > 1)) {
          ctx.addIssue({
            code: z.ZodIssueCode.custom,
            message: 'gripper state 숫자값은 0 이상 1 이하여야 합니다 (0=close, 1=open)',
          });
        }
      }),
    durationSec: durationSecSchema.optional(),
  }),
  z.object({ kind: z.literal('wait'), durationSec: durationSecSchema }),
  z.object({
    kind: z.literal('waitForCollision'),
    between: z.tuple([entityIdSchema, entityIdSchema]),
    timeoutSec: timeoutSecSchema.optional(),
  }),
  z.object({
    kind: z.literal('label'),
    name: z.string().min(1, 'label 이름은 비어 있지 않은 문자열이어야 합니다'),
  }),
  z.object({
    kind: z.literal('goto'),
    label: z.string().min(1, 'goto 대상 label은 비어 있지 않은 문자열이어야 합니다'),
    // "N회 반복" 시맨틱 (DATA_MODEL §6): 음수는 영구 no-op, 소수는 ceil처럼 동작해
    // (player.ts 'fired < times') 의도와 조용히 어긋난다 — 0 이상의 정수만 허용.
    times: z
      .number()
      .int('times는 정수여야 합니다')
      .nonnegative('times는 0 이상이어야 합니다')
      .optional(),
  }),
  z.object({
    kind: z.literal('moveToPose'),
    robot: entityIdSchema.optional(),
    target: transformSchema,
    durationSec: durationSecSchema,
  }),
]);

export const controlSequenceSchema = z
  .object({
    id: z.string().min(1, '시퀀스 id는 비어 있지 않은 문자열이어야 합니다'),
    robot: entityIdSchema,
    loop: z.boolean().optional(),
    steps: z
      .array(controlStepSchema)
      .min(MIN_SEQUENCE_STEP_COUNT, `steps에는 최소 ${MIN_SEQUENCE_STEP_COUNT}개의 step이 필요합니다`),
  })
  .superRefine((sequence, ctx) => {
    // 교차 규칙: goto 대상 label은 시퀀스 안에 존재해야 한다 (DATA_MODEL §6)
    const labelNames = new Set<string>();
    for (const step of sequence.steps) {
      if (step.kind === 'label') labelNames.add(step.name);
    }
    sequence.steps.forEach((step, index) => {
      if (step.kind === 'goto' && !labelNames.has(step.label)) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ['steps', index, 'label'],
          message: `goto 대상 label '${step.label}'이(가) 시퀀스에 존재하지 않습니다`,
        });
      }
    });
  });

// ── 검증 API ────────────────────────────────────────────────────────

/**
 * SceneSpec 검증. 구조(zod) + 교차 필드 규칙을 모두 검사한다.
 * 실패 시 "경로: 한국어 메시지" 형식의 오류 목록을 반환한다 (UI 직접 표시용).
 */
export function validateScene(input: unknown): ValidationResult<SceneSpec> {
  const parsed = sceneSpecSchema.safeParse(input, { errorMap: koreanErrorMap });
  if (!parsed.success) return { ok: false, errors: formatIssues(parsed.error.issues) };
  return { ok: true, value: parsed.data };
}

/**
 * ControlSequence 검증. scene을 함께 주면 참조 무결성(robot/entity/joint 존재)까지 검사한다.
 */
export function validateSequence(
  input: unknown,
  scene?: SceneSpec,
): ValidationResult<ControlSequence> {
  const parsed = controlSequenceSchema.safeParse(input, { errorMap: koreanErrorMap });
  if (!parsed.success) return { ok: false, errors: formatIssues(parsed.error.issues) };

  const sequence: ControlSequence = parsed.data;
  if (scene) {
    const errors = checkSequenceAgainstScene(sequence, scene);
    if (errors.length > 0) return { ok: false, errors };
  }
  return { ok: true, value: sequence };
}

// ── 씬 참조 무결성 (scene이 제공된 경우) ─────────────────────────────

function checkSequenceAgainstScene(sequence: ControlSequence, scene: SceneSpec): string[] {
  const errors: string[] = [];
  const entityById = new Map<string, EntitySpec>();
  for (const entity of scene.entities) entityById.set(entity.id, entity);

  /** 로봇 참조 확인: 존재 + type==='robot'. 실패 시 오류를 쌓고 undefined 반환. */
  const resolveRobot = (id: string, path: string): RobotSpec | undefined => {
    const entity = entityById.get(id);
    if (!entity) {
      errors.push(`${path}: 씬에 '${id}' 엔티티가 존재하지 않습니다`);
      return undefined;
    }
    if (!isRobotSpec(entity)) {
      errors.push(`${path}: '${id}'은(는) 로봇 엔티티가 아닙니다 (type: '${entity.type}')`);
      return undefined;
    }
    return entity;
  };

  const defaultRobot = resolveRobot(sequence.robot, 'robot');

  sequence.steps.forEach((step, index) => {
    switch (step.kind) {
      case 'moveJoints':
      case 'setJoints': {
        const robot =
          step.robot !== undefined
            ? resolveRobot(step.robot, `steps[${index}].robot`)
            : defaultRobot;
        if (robot) checkJointNames(robot, step.targets, `steps[${index}].targets`, errors);
        break;
      }
      case 'gripper':
      case 'moveToPose': {
        if (step.robot !== undefined) resolveRobot(step.robot, `steps[${index}].robot`);
        break;
      }
      case 'waitForCollision': {
        step.between.forEach((entityId, betweenIndex) => {
          if (!entityById.has(entityId)) {
            errors.push(
              `steps[${index}].between[${betweenIndex}]: 씬에 '${entityId}' 엔티티가 존재하지 않습니다`,
            );
          }
        });
        break;
      }
      default:
        // wait / label / goto — 씬 참조 없음
        break;
    }
  });

  return errors;
}

/**
 * 관절 이름 검사: 로봇 스펙의 home/jointMap/jointLimits 키 합집합에 있어야 한다.
 * 로봇에 관절 정보가 전혀 없으면 통과시킨다 — URDF 로드 시점 검증에 위임 (DATA_MODEL §6).
 */
function checkJointNames(
  robot: RobotSpec,
  targets: Record<string, number>,
  pathPrefix: string,
  errors: string[],
): void {
  const knownJoints = new Set<string>();
  for (const source of [robot.home, robot.jointMap, robot.jointLimits]) {
    if (source) for (const key of Object.keys(source)) knownJoints.add(key);
  }
  if (knownJoints.size === 0) return;

  const knownList = [...knownJoints].sort().join(', ');
  for (const jointName of Object.keys(targets)) {
    if (!knownJoints.has(jointName)) {
      errors.push(
        `${pathPrefix}.${jointName}: 로봇 '${robot.id}'에 정의되지 않은 관절 이름입니다 (알려진 관절: ${knownList})`,
      );
    }
  }
}

// ── 타입 계약 검증 (컴파일 타임 전용) ────────────────────────────────
// zod 출력 타입이 schema/types.ts의 동결된 계약에 대입 가능함을 보장한다.
// 아래 선언이 컴파일되지 않으면 스키마와 타입이 어긋난 것이다 — types.ts를 고치지 말고
// 이 파일의 스키마를 고칠 것.

type MustAssign<A extends B, B> = A;

type _TransformOut = MustAssign<z.infer<typeof transformSchema>, Transform>;
type _ColliderShapeOut = MustAssign<z.infer<typeof colliderShapeSchema>, ColliderShape>;
type _BodyTypeOut = MustAssign<z.infer<typeof bodyTypeSchema>, BodyType>;
type _ColliderGroupOut = MustAssign<z.infer<typeof colliderGroupSchema>, ColliderGroup>;
type _ColliderSpecOut = MustAssign<z.infer<typeof colliderSpecSchema>, ColliderSpec>;
type _PhysicsSpecOut = MustAssign<z.infer<typeof physicsSpecSchema>, PhysicsSpec>;
type _VisualSpecOut = MustAssign<z.infer<typeof visualSpecSchema>, VisualSpec>;
type _EntitySpecOut = MustAssign<z.infer<typeof entitySpecSchema>, EntitySpec>;
type _RobotSpecOut = MustAssign<z.infer<typeof robotSpecSchema>, RobotSpec>;
type _SceneSpecOut = MustAssign<z.infer<typeof sceneSpecSchema>, SceneSpec>;
type _EasingOut = MustAssign<z.infer<typeof easingSchema>, Easing>;
type _ControlStepOut = MustAssign<z.infer<typeof controlStepSchema>, ControlStep>;
type _ControlSequenceOut = MustAssign<z.infer<typeof controlSequenceSchema>, ControlSequence>;
