// schema/entities.ts — 협업 개체(도메인 문서) 스키마 · 클라이언트/서버 공유 계약
//
// ── 왜 이 파일인가 ──────────────────────────────────────────────────
// Phase 12+(콘솔 평면·백엔드·다중 사용자)가 다루는 최상위 개체 5종을 정의한다:
//   Process(공정) · Task(작업) · Block(재사용 블록) · Device(장비) · Run(실행 기록)
// 서버(server/)와 클라이언트(src/)가 **같은 zod 스키마**로 검증한다 — 계약이 한 곳에
// 있어야 API 양끝이 어긋나지 않는다 (docs/BACKEND.md가 이 파일을 규범으로 참조).
//
// ── 계층 규칙 ───────────────────────────────────────────────────────
// schema 내부 참조만 허용(validate.ts의 controlStepSchema). core/render/ui를 모른다.
// scene/sequence 필드는 여기서 **의도적으로 z.unknown()** 이다 — 서버는 씬의 내용을
// 해석하지 않는다(봉투만 검증). 도메인 검증(sceneSpecSchema 등)은 클라이언트 로드
// 시점의 몫이다(불변식 §2.9와 동일한 원칙: 검증 없이는 실행에 노출하지 않는다).
//
// ── 복사본 의미론 (사용자 결정: "복사본 유지") ──────────────────────
// Task.scene은 공정 씬의 **전체 사본**이다. 참조가 아니다 — 공정 레이아웃을 고쳐도
// 검증이 끝난 작업이 조용히 바뀌지 않는다(로봇 셀에서 무단 전파는 사고다).
// 대신 sceneOrigin{processId, processVersion}을 기록해, 공정이 갱신되면 UI가
// "공정 레이아웃 갱신됨 — 차이 보기/가져오기"를 **알리고 사람이 적용**한다.
// Block 삽입도 같은 원칙: 노드로 전개(inline expansion)하고 blockRef 흔적만 남긴다.

import { z } from 'zod';
import { controlStepSchema } from './validate';

// ── 공통 원자 ───────────────────────────────────────────────────────

/** 개체 id — 클라이언트가 crypto.randomUUID()로 발급한다(오프라인 생성 지원) */
export const entityIdSchema = z.string().min(8).max(64);

/** 사람이 읽는 이름 — 공백만인 이름 금지 */
export const displayNameSchema = z
  .string()
  .min(1)
  .max(80)
  .refine((s) => s.trim().length > 0, { message: '이름이 비어 있습니다' });

/** ISO 8601 타임스탬프 (서버가 발급하는 시각은 서버가 진실) */
export const isoTimeSchema = z.string().datetime({ offset: true }).or(z.string().datetime());

/** 낙관적 동시성 버전 — 서버가 증가시킨다. 클라이언트는 절대 임의 조작하지 않는다. */
export const versionSchema = z.number().int().min(1);

// ── 사용자 / 인증 ───────────────────────────────────────────────────

/** 역할 2종 — 현장 팀 모델. admin=관리자(사용자 관리·완전 삭제), tech=설치기사 */
export const userRoleSchema = z.enum(['admin', 'tech']);
export type UserRole = z.infer<typeof userRoleSchema>;

/** PIN — 공유 단말·장갑 낀 손 전제의 숫자 PIN. tech 4–8자리, admin 6–8자리(서버 강제) */
export const pinSchema = z.string().regex(/^\d{4,8}$/, 'PIN은 4~8자리 숫자입니다');

export const userInfoSchema = z.object({
  id: entityIdSchema,
  name: displayNameSchema,
  role: userRoleSchema,
  active: z.boolean(),
});
export type UserInfo = z.infer<typeof userInfoSchema>;

// ── 기록 메타 (서버가 소유하는 봉투) ────────────────────────────────

/** 서버가 붙이는 감사 메타 — 다중 사용자에서 "누가 언제"의 진실 */
export const recordMetaSchema = z.object({
  version: versionSchema,
  createdAtIso: isoTimeSchema,
  createdBy: entityIdSchema,
  createdByName: z.string(),
  updatedAtIso: isoTimeSchema,
  updatedBy: entityIdSchema,
  updatedByName: z.string(),
  deletedAtIso: isoTimeSchema.nullable(),
  deletedByName: z.string().nullable(),
});
export type RecordMeta = z.infer<typeof recordMetaSchema>;

// ── Process (공정) ──────────────────────────────────────────────────

/** 공정 규칙 — 실행 기본값. 작업 열 때 초기값으로 복사된다(참조 아님). */
export const processRulesSchema = z.object({
  /** 예기치 않은 충돌 시 자동 정지 기본값 */
  autoPauseOnCollision: z.boolean(),
  /** 재생 속도 상한 (1|2|4, null=제한 없음) */
  speedLimitMult: z.number().int().positive().nullable(),
});
export type ProcessRules = z.infer<typeof processRulesSchema>;

export const processDocSchema = z.object({
  id: entityIdSchema,
  name: displayNameSchema,
  descriptionKo: z.string().max(500).default(''),
  /** SceneSpec — 서버는 내용을 해석하지 않는다(클라이언트가 sceneSpecSchema로 검증) */
  scene: z.unknown(),
  /** 이 공정 소속 장비 id 목록 (Device.id 참조) */
  deviceIds: z.array(entityIdSchema).default([]),
  rules: processRulesSchema,
});
export type ProcessDoc = z.infer<typeof processDocSchema>;

// ── Task (작업) ─────────────────────────────────────────────────────

/** 씬 사본의 출처 — 공정 갱신 감지용("차이 보기/가져오기"는 사람이 결정) */
export const sceneOriginSchema = z.object({
  processId: entityIdSchema,
  processVersion: versionSchema,
});
export type SceneOrigin = z.infer<typeof sceneOriginSchema>;

export const taskDocSchema = z.object({
  id: entityIdSchema,
  name: displayNameSchema,
  /** 소속 공정 (null = 공정 무소속 자유 작업) */
  processId: entityIdSchema.nullable(),
  /** 씬 사본의 출처 기록 (processId가 null이면 null) */
  sceneOrigin: sceneOriginSchema.nullable(),
  /** SceneSpec 전체 사본 (복사본 의미론 — 파일 헤더 참조) */
  scene: z.unknown(),
  /** ControlSequence | null — 검증은 클라이언트 로드 시(불변식 §2.9) */
  sequence: z.unknown().nullable(),
  /** 임포트 메시 등 — id → data URI (WorkcellDocument.assets와 동일 형태) */
  assets: z.record(z.string()).default({}),
  /** 목록 카드 썸네일 (뷰포트 캡처 data URI, 선택) */
  thumbnail: z.string().nullable().default(null),
  notes: z.string().max(2000).default(''),
});
export type TaskDoc = z.infer<typeof taskDocSchema>;

// ── Block (재사용 블록) ─────────────────────────────────────────────

/**
 * 블록 파라미터 — 삽입 시 사용자가 채우는 인자. bindings가 가리키는 step 경로에
 * 값이 대입된 뒤 전개된다(예: durationSec을 블록 인자로 승격).
 */
export const blockParamSchema = z.object({
  key: z.string().min(1).max(40),
  labelKo: z.string().min(1).max(60),
  kind: z.enum(['number', 'string', 'boolean']),
  defaultValue: z.union([z.number(), z.string(), z.boolean()]),
  min: z.number().optional(),
  max: z.number().optional(),
  /** 대입 지점: steps[stepIndex]의 경로(예: "params.durationSec") */
  bindings: z.array(
    z.object({ stepIndex: z.number().int().min(0), path: z.string().min(1) }),
  ),
});
export type BlockParam = z.infer<typeof blockParamSchema>;

export const blockDocSchema = z.object({
  id: entityIdSchema,
  name: displayNameSchema,
  descriptionKo: z.string().max(500).default(''),
  /** 전개될 step 목록 — 깊은 검증(controlStepSchema). 빈 블록은 금지. */
  steps: z.array(controlStepSchema).min(1),
  params: z.array(blockParamSchema).default([]),
  /** 어떤 로봇 종류를 전제로 만들었는지 힌트 (라이브러리 카드 표기용, 자유 문자열) */
  robotHint: z.string().max(60).nullable().default(null),
});
export type BlockDoc = z.infer<typeof blockDocSchema>;

// ── Device (장비) ───────────────────────────────────────────────────

export const deviceKindSchema = z.enum(['robot', 'camera', 'plc']);
export type DeviceKind = z.infer<typeof deviceKindSchema>;

/**
 * 연결 형상 — 현재 구현은 'virtual'만 실동작(씬의 로봇 엔티티에 대응).
 * 'real'은 어댑터 경계 예약(endpoint = 브리지 URL). UI는 미구현 사유를 정직하게 표기한다.
 */
export const deviceConnectionSchema = z.object({
  mode: z.enum(['virtual', 'real']),
  endpoint: z.string().max(200).nullable(),
});
export type DeviceConnection = z.infer<typeof deviceConnectionSchema>;

export const deviceDocSchema = z.object({
  id: entityIdSchema,
  name: displayNameSchema,
  kind: deviceKindSchema,
  /** 라이브러리 로봇 템플릿 key (robot 종류일 때 — 'arm-6'|'scara-4'|'cobot-7' 등) */
  templateKey: z.string().max(40).nullable(),
  connection: deviceConnectionSchema,
  notes: z.string().max(500).default(''),
});
export type DeviceDoc = z.infer<typeof deviceDocSchema>;

// ── Run (실행 기록 — append-only) ───────────────────────────────────

export const runResultSchema = z.enum([
  'completed', // 시퀀스 완주
  'stopped', // 사람이 ⏹
  'error', // waitForCollision timeout 등 노드 오류
  'autoPaused', // 예기치 않은 충돌 자동 정지 후 종료
]);
export type RunResult = z.infer<typeof runResultSchema>;

export const runCollisionSchema = z.object({
  atSimSec: z.number(),
  entityA: z.string(),
  entityB: z.string(),
  phase: z.enum(['start', 'end']),
  /** 발생 시 활성 노드 (없으면 null) */
  nodeId: z.string().nullable(),
  classification: z.enum(['intended', 'unexpected']),
});
export type RunCollision = z.infer<typeof runCollisionSchema>;

export const runInterventionSchema = z.object({
  atSimSec: z.number(),
  kind: z.enum(['play', 'pause', 'stop', 'stepNode', 'runFromNode', 'autoPause']),
  nodeId: z.string().nullable(),
});
export type RunIntervention = z.infer<typeof runInterventionSchema>;

export const runRecordSchema = z.object({
  id: entityIdSchema,
  taskId: entityIdSchema,
  /** 실행 시점의 작업 이름/버전 스냅샷 — 작업이 개명/삭제돼도 기록은 읽힌다 */
  taskName: z.string(),
  taskVersion: versionSchema,
  processId: entityIdSchema.nullable(),
  operatorId: entityIdSchema,
  operatorName: z.string(),
  startedAtIso: isoTimeSchema,
  endedAtIso: isoTimeSchema,
  result: runResultSchema,
  stepsTotal: z.number().int().min(0),
  stepsDone: z.number().int().min(0),
  simTimeSec: z.number().min(0),
  wallTimeSec: z.number().min(0),
  collisions: z.array(runCollisionSchema),
  interventions: z.array(runInterventionSchema),
});
export type RunRecord = z.infer<typeof runRecordSchema>;

// ── 목록 메타 (payload 없는 가벼운 행 — 서버 목록 API 응답) ─────────

export const taskSummarySchema = z.object({
  stepCount: z.number().int().min(0),
  hasThumbnail: z.boolean(),
  lastRun: z
    .object({ atIso: isoTimeSchema, result: runResultSchema })
    .nullable(),
});
export type TaskSummary = z.infer<typeof taskSummarySchema>;

/** 목록 행 — kind별 summary는 tasks만 확장 필드를 갖는다 */
export const entityMetaSchema = z.object({
  id: entityIdSchema,
  name: z.string(),
  meta: recordMetaSchema,
  /** tasks 목록에만 존재 */
  taskSummary: taskSummarySchema.nullable().default(null),
  /** tasks: 소속 공정 id (필터용) */
  processId: entityIdSchema.nullable().default(null),
});
export type EntityMeta = z.infer<typeof entityMetaSchema>;

// ── API 봉투 (요청/응답 DTO — docs/BACKEND.md의 규범 타입) ──────────

/** 저장 요청 — baseVersion 불일치면 서버가 409 + 현재본을 돌려준다 */
export interface SaveRequest<T> {
  readonly doc: T;
  /** 신규 생성이면 null */
  readonly baseVersion: number | null;
}

export interface RecordEnvelope<T> {
  readonly doc: T;
  readonly meta: RecordMeta;
}

export interface ConflictResponse<T> {
  readonly error: 'version-conflict';
  readonly current: RecordEnvelope<T>;
}

/** 편집 잠금(advisory) — TTL이 지나면 자연 해제. 강탈은 없다(만료 대기 또는 소유자 해제). */
export const lockInfoSchema = z.object({
  entityKind: z.enum(['task', 'process', 'block']),
  entityId: entityIdSchema,
  userId: entityIdSchema,
  userName: z.string(),
  acquiredAtIso: isoTimeSchema,
  expiresAtIso: isoTimeSchema,
});
export type LockInfo = z.infer<typeof lockInfoSchema>;

export const taskStatsSchema = z.object({
  runCount: z.number().int().min(0),
  successCount: z.number().int().min(0),
  avgDurationSec: z.number().nullable(),
  /** 예기치 않은 충돌 상위 노드 (nodeId → 횟수, 내림차순 최대 5개) */
  topCollisionNodes: z.array(z.object({ nodeId: z.string(), count: z.number().int() })),
});
export type TaskStats = z.infer<typeof taskStatsSchema>;

// ── 서버 상수 (양끝 공유 — 매직넘버 금지) ───────────────────────────

/** 세션 수명 (일) — 현장 단말에서 매일 로그인시키지 않는다. 슬라이딩 갱신. */
export const SESSION_TTL_DAYS = 30;
/** PIN 연속 실패 허용 횟수 — 초과 시 잠금 */
export const PIN_MAX_ATTEMPTS = 5;
/** PIN 잠금 시간 (초) */
export const PIN_LOCKOUT_SEC = 60;
/** 편집 잠금 TTL (초) — heartbeat가 이 안에 갱신해야 유지된다 */
export const LOCK_TTL_SEC = 90;
/** 휴지통 보존 (일) — 지나면 목록 조회 시 지연 완전삭제 */
export const TRASH_RETENTION_DAYS = 30;
/** API 경로 접두 */
export const API_PREFIX = '/api/v1';
