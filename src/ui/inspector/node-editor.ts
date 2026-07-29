// ui/inspector/node-editor.ts — 노드 파라미터 인스펙터 폼 (UX_DESIGN §3.5 (B), ROADMAP Phase 8)
//
// Flow Graph에서 선택된 노드(FlowNode)의 step 파라미터를 종류별 폼으로 편집한다.
// 모든 편집은 deps.commitParams로만 커밋된다 — 통합자가 updateNodeParams op +
// serializeGraph(toSequence→validateSequence) 검증 위에서 구현한다. 검증 실패 시
// 한국어 오류를 인라인으로 표시하고 폼을 마지막 유효 값(진실 = deps.getNode)으로
// 되돌린다 — 그래프를 직렬화 불가능한 상태로 만들 수 없다(불변식 §2.8).
//
// 커밋 정책 (요구 §3): select/checkbox/세그먼트는 change 즉시, 슬라이더는 드래그 중
// 숫자 리드아웃만 갱신(라이브 프리뷰)하고 release(change)에 커밋, number/text 입력은
// blur/Enter(change)에 커밋한다. 키 입력마다 커밋하지 않는다.
//
// 계층 규칙 (CLAUDE.md §3): ui는 core를 import하지 않고 schema 타입만 안다.
// NodeEditorFlowNode는 schema/flow-graph.ts FlowNode의 **구조적 부분집합**이다
// (id/kind/params/enabled/note — ui.x/y·status·origin은 이 폼의 관심사가 아니다).
// 통합자가 FlowNode를 그대로 넘기면 구조적 타이핑으로 수용된다.
//
// 순수 헬퍼(params↔폼 상태 변환 · 관절 행 모델 · 클램프/정규화)는 DOM 없이 export되어
// node에서 단위 테스트된다(node-editor.test.ts). DOM 조립은 얇다.

import {
  BORDER_WIDTH,
  COLLISION,
  COLOR,
  ICON,
  RADIUS,
  SHADOW,
  SPACE,
  TYPE,
  applyType,
  ensureThemeStyles,
  makeButton,
  makePanelHeader,
  styled,
} from '../theme';
import { icon, makeIconButton } from '../icons';
import type { IconName } from '../icons';
import { attachRangeFill, formatFixed } from './inspector';
import { clamp, parseNumeric } from './entity-editor';
import type { Easing } from '../../schema/types';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4, 시각 토큰은 ui/theme.ts) ────

/** 관절 limits가 없을 때 슬라이더 범위 폴백: ±π rad (요구 §2 moveJoints/setJoints) */
export const JOINT_RANGE_FALLBACK_RAD = Math.PI;
/** gripper 숫자 상태의 열림/닫힘 극값 (DATA_MODEL §6: 0=close, 1=open) */
export const GRIPPER_OPEN_RATIO = 1;
export const GRIPPER_CLOSE_RATIO = 0;

/** 관절 값 표시 소수 자릿수 (rad — 요구 §2 "number input (rad, 3 decimals)") */
const JOINT_DECIMALS = 3;
/** 관절 슬라이더 step (rad) */
const JOINT_SLIDER_STEP_RAD = 0.001;
/** durationSec/timeoutSec 입력 step·소수 자릿수 */
const DURATION_STEP_SEC = 0.1;
const DURATION_DECIMALS = 2;
/** gripper 0..1 슬라이더 step·리드아웃 소수 자릿수 */
const GRIPPER_SLIDER_STEP = 0.01;
const GRIPPER_DECIMALS = 2;
/** 노트 textarea 행 수 */
const NOTE_ROWS = 2;
/** 그리퍼 값 리드아웃 고정 폭 (자릿수가 늘어도 슬라이더가 흔들리지 않게) */
const GRIPPER_READOUT_COL_PX = 44;
/** 관절 행의 숫자 입력 컬럼 폭 */
const JOINT_INPUT_COL_PX = 68;

/** 커밋 오류 상자의 DOM id — 실패한 입력이 aria-describedby로 가리킨다 (UX_AUDIT C-16) */
const ERROR_BOX_ID = 'ne-errors';

/** DOM id 발급기 — label htmlFor / aria-describedby 연결용 (재구축마다 유일) */
let idSeq = 0;
function nextDomId(prefix: string): string {
  idSeq += 1;
  return `${prefix}-${idSeq}`;
}

// ── 공개 타입 ───────────────────────────────────────────────────────

/** 그래프 노드의 구조적 부분집합 — schema/flow-graph.ts FlowNode가 그대로 대입된다 */
export interface NodeEditorFlowNode {
  id: string;
  kind: string;
  params: Record<string, unknown>;
  enabled: boolean;
  note?: string;
}

/** 씬 그라운딩 컨텍스트 — 통합자가 SceneSpec/RobotSpec에서 유도해 공급 */
export interface NodeEditorSceneContext {
  /** 시퀀스 기본 대상 로봇 id (단일 로봇 MVP — 폼에서는 읽기 전용 표시) */
  robot: string;
  robotJoints: Array<{ name: string; limits?: [number, number] }>;
  entityIds: string[];
  /** 시퀀스 안의 label 노드 이름들 (goto 대상 후보) */
  labels: string[];
  gripperAvailable: boolean;
}

/**
 * 통합자가 flow-graph 편집 op 위에서 구현해 주입하는 편집 표면.
 * commitParams는 updateNodeParams + serializeGraph 검증을 수행하고,
 * null(성공) 또는 한국어 오류 목록(실패 — 그래프 미변경)을 돌려준다(불변식 §2.8).
 */
export interface NodeEditorDeps {
  getNode(id: string): NodeEditorFlowNode | null;
  sceneContext(): NodeEditorSceneContext;
  commitParams(id: string, params: Record<string, unknown>): string[] | null;
  setEnabled(id: string, enabled: boolean): void;
  setNote(id: string, note: string): void;
}

export interface NodeEditorHandle {
  /** 폼 루트 — 통합자가 우측 스택 등으로 배치할 때 사용 */
  readonly el: HTMLElement;
  /** 선택 변경: id 노드의 폼으로 재구축. null(또는 미존재 id) = 플레이스홀더 */
  showFor(id: string | null): void;
  /** 외부 변경 재동기화 (통합자가 그래프 변경/주기마다 호출). 폼 편집 중이면 건너뛴다 */
  refresh(): void;
  dispose(): void;
}

// ── 순수 헬퍼: 공통 파싱/정규화 (DOM 비의존 — node 단위 테스트 대상) ──

/** unknown → 유한 수. 숫자가 아니거나 NaN/∞면 null */
export function toFiniteNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

/** durationSec 정규화: null/비유한 → 0, 음수 → 0 (스키마 nonnegative와 정합) */
export function clampDurationSec(value: number | null): number {
  if (value === null || !Number.isFinite(value)) return 0;
  return Math.max(0, value);
}

/** 0..1 클램프 (gripper 숫자 상태) — 비유한 입력은 0(닫힘)으로 방어 */
export function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return clamp(value, 0, 1);
}

/** unknown → Easing. 허용 값이 아니면 undefined (기본 easing에 위임) */
export function easingOf(value: unknown): Easing | undefined {
  return value === 'linear' || value === 'easeInOut' || value === 'step' ? value : undefined;
}

/** 노드의 대상 로봇 표시값: params.robot(비어 있지 않은 문자열) 우선, 없으면 기본 로봇 */
export function robotDisplayOf(params: Record<string, unknown>, defaultRobot: string): string {
  const robot = params['robot'];
  return typeof robot === 'string' && robot !== '' ? robot : defaultRobot;
}

/**
 * step 종류 → 헤더 아이콘(SVG 이름)/라벨 (요구 §2 "kind 헤더 with icon"). 미지 kind는 원문 라벨.
 *
 * 아이콘은 이모지가 아니라 `icons.ts`의 이름이다 (UX_AUDIT C-13) — 컬러 이모지는
 * currentColor를 못 받아 hover/disabled에서 텍스트만 밝아지고, OS마다 그림이 달라
 * 스크린샷·문서가 재현되지 않는다.
 */
export function nodeKindMeta(kind: string): { icon: IconName; label: string } {
  switch (kind) {
    case 'moveJoints':
      return { icon: 'robotArm', label: 'MoveJoints — 관절 이동' };
    case 'setJoints':
      return { icon: 'rotate', label: 'SetJoints — 관절 즉시 설정' };
    case 'gripper':
      return { icon: 'gripper', label: 'Gripper — 그리퍼' };
    case 'wait':
      return { icon: 'timer', label: 'Wait — 대기' };
    case 'waitForCollision':
      return { icon: 'impact', label: 'WaitForCollision — 접촉 대기' };
    case 'label':
      return { icon: 'bookmark', label: 'Label — 라벨' };
    case 'goto':
      return { icon: 'undo', label: 'Goto — 라벨로 이동' };
    case 'moveToPose':
      return { icon: 'target', label: 'MoveToPose — 포즈 이동' };
    default:
      return { icon: 'workflow', label: kind };
  }
}

// ── 순수 헬퍼: 관절 행 모델 (moveJoints/setJoints) ───────────────────

/** 관절 행 1건 — 폼 상태의 단위. limits는 슬라이더 범위/클램프에 쓰인다 */
export interface JointRowState {
  name: string;
  valueRad: number;
  limits?: [number, number];
}

/** moveJoints/setJoints 폼 상태 (durationSec/easing은 moveJoints 전용) */
export interface JointsFormState {
  rows: JointRowState[];
  durationSec?: number;
  easing?: Easing;
}

/** limits 튜플 방어 검증: [lo, hi] 유한 + lo ≤ hi 일 때만 유효 */
export function validJointLimits(limits: unknown): [number, number] | undefined {
  if (!Array.isArray(limits) || limits.length !== 2) return undefined;
  const [lo, hi] = limits;
  if (typeof lo !== 'number' || typeof hi !== 'number') return undefined;
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || lo > hi) return undefined;
  return [lo, hi];
}

/** 슬라이더 범위: 유효 limits 사용, 없으면 ±π 폴백 (요구 §2) */
export function jointSliderRange(
  limits?: readonly [number, number],
): { minRad: number; maxRad: number } {
  const valid = validJointLimits(limits);
  if (valid !== undefined) return { minRad: valid[0], maxRad: valid[1] };
  return { minRad: -JOINT_RANGE_FALLBACK_RAD, maxRad: JOINT_RANGE_FALLBACK_RAD };
}

/** 관절 값을 슬라이더 범위로 클램프. 비유한 입력은 0을 클램프해 방어 */
export function clampJointValueRad(value: number, limits?: readonly [number, number]): number {
  const { minRad, maxRad } = jointSliderRange(limits);
  const base = Number.isFinite(value) ? value : 0;
  return clamp(base, minRad, maxRad);
}

/** robotJoints → 이름별 유효 limits 맵 (무효 limits는 제외 — ±π 폴백 대상) */
export function jointLimitsByName(
  robotJoints: ReadonlyArray<{ name: string; limits?: [number, number] }>,
): Map<string, [number, number]> {
  const map = new Map<string, [number, number]>();
  for (const joint of robotJoints) {
    const valid = validJointLimits(joint.limits);
    if (valid !== undefined) map.set(joint.name, valid);
  }
  return map;
}

/**
 * params.targets → 관절 행 목록 (삽입 순서 유지). 숫자가 아닌 값은 건너뛰고,
 * robotJoints의 유효 limits를 행에 부착한다. durationSec/easing도 함께 읽는다.
 */
export function jointsFormStateOf(
  params: Record<string, unknown>,
  robotJoints: ReadonlyArray<{ name: string; limits?: [number, number] }>,
): JointsFormState {
  const limitsMap = jointLimitsByName(robotJoints);
  const rows: JointRowState[] = [];
  const targets = params['targets'];
  if (typeof targets === 'object' && targets !== null && !Array.isArray(targets)) {
    for (const [name, raw] of Object.entries(targets as Record<string, unknown>)) {
      const valueRad = toFiniteNumber(raw);
      if (valueRad === null) continue;
      const limits = limitsMap.get(name);
      rows.push(limits !== undefined ? { name, valueRad, limits } : { name, valueRad });
    }
  }
  const durationSec = toFiniteNumber(params['durationSec']);
  const easing = easingOf(params['easing']);
  return {
    rows,
    ...(durationSec !== null ? { durationSec } : {}),
    ...(easing !== undefined ? { easing } : {}),
  };
}

/** 아직 행으로 쓰이지 않은 관절 이름 (추가 드롭다운 후보 — 요구 §2) */
export function availableJointNames(
  robotJoints: ReadonlyArray<{ name: string; limits?: [number, number] }>,
  rows: ReadonlyArray<JointRowState>,
): string[] {
  const used = new Set(rows.map((row) => row.name));
  return robotJoints.map((joint) => joint.name).filter((name) => !used.has(name));
}

/** 관절 행 → targets 레코드. 값은 행별 limits(없으면 ±π)로 클램프 */
export function targetsFromJointRows(rows: ReadonlyArray<JointRowState>): Record<string, number> {
  const targets: Record<string, number> = {};
  for (const row of rows) targets[row.name] = clampJointValueRad(row.valueRad, row.limits);
  return targets;
}

/**
 * 관절 폼 상태 → params. targets를 재구성하고, moveJoints면 durationSec(0 이상
 * 클램프)과 easing(undefined면 키 제거 — 기본값 위임)을 관리한다.
 * 관리하지 않는 키(robot 등)는 원본을 보존한다(무손실 왕복 — UX §6).
 */
export function applyJointsFormToParams(
  original: Record<string, unknown>,
  form: JointsFormState,
  kind: 'moveJoints' | 'setJoints',
): Record<string, unknown> {
  const params: Record<string, unknown> = { ...original, targets: targetsFromJointRows(form.rows) };
  if (kind === 'moveJoints') {
    params['durationSec'] = clampDurationSec(form.durationSec ?? null);
    if (form.easing !== undefined) params['easing'] = form.easing;
    else delete params['easing'];
  }
  return params;
}

// ── 순수 헬퍼: gripper ──────────────────────────────────────────────

/** 세그먼트 컨트롤 모드: 열기 | 닫기 | 값(0..1 슬라이더) */
export type GripperMode = 'open' | 'close' | 'value';

export interface GripperFormState {
  mode: GripperMode;
  /** 값 모드의 0..1 비율. open/close 모드에서도 모드 전환 시 초기값으로 쓰인다 */
  valueRatio: number;
  durationSec?: number;
}

/** params.state → 폼 상태. 미지 값은 open으로 폴백 (0=닫힘, 1=열림 — DATA_MODEL §6) */
export function gripperFormStateOf(params: Record<string, unknown>): GripperFormState {
  const durationSec = toFiniteNumber(params['durationSec']);
  const common = durationSec !== null ? { durationSec: clampDurationSec(durationSec) } : {};
  const state = params['state'];
  if (state === 'close') return { mode: 'close', valueRatio: GRIPPER_CLOSE_RATIO, ...common };
  const ratio = toFiniteNumber(state);
  if (ratio !== null) return { mode: 'value', valueRatio: clamp01(ratio), ...common };
  return { mode: 'open', valueRatio: GRIPPER_OPEN_RATIO, ...common };
}

/** gripper 폼 상태 → params. durationSec undefined면 키 제거(즉시 동작 위임) */
export function applyGripperFormToParams(
  original: Record<string, unknown>,
  form: GripperFormState,
): Record<string, unknown> {
  const params: Record<string, unknown> = { ...original };
  params['state'] = form.mode === 'value' ? clamp01(form.valueRatio) : form.mode;
  if (form.durationSec !== undefined) params['durationSec'] = clampDurationSec(form.durationSec);
  else delete params['durationSec'];
  return params;
}

// ── 순수 헬퍼: wait / waitForCollision ──────────────────────────────

/** params.durationSec → 표시값 (누락/무효는 0) */
export function waitDurationOf(params: Record<string, unknown>): number {
  return clampDurationSec(toFiniteNumber(params['durationSec']));
}

export function applyWaitFormToParams(
  original: Record<string, unknown>,
  durationSec: number | null,
): Record<string, unknown> {
  return { ...original, durationSec: clampDurationSec(durationSec) };
}

export interface CollisionFormState {
  a: string;
  b: string;
  /** undefined = 무제한 대기 (timeoutSec 키 생략) */
  timeoutSec?: number;
}

/** timeoutSec 정규화: 양수만 유효, 그 외(빈값·0·음수·비유한)는 무제한(undefined) */
export function normalizeTimeoutSec(value: number | null): number | undefined {
  if (value === null || !Number.isFinite(value) || value <= 0) return undefined;
  return value;
}

/** params.between/timeoutSec → 폼 상태. 무효 항목은 빈 문자열/무제한으로 방어 */
export function collisionFormStateOf(params: Record<string, unknown>): CollisionFormState {
  const betweenRaw = params['between'];
  const pair: unknown[] = Array.isArray(betweenRaw) ? betweenRaw : [];
  const first = pair[0];
  const second = pair[1];
  const timeoutSec = normalizeTimeoutSec(toFiniteNumber(params['timeoutSec']));
  return {
    a: typeof first === 'string' ? first : '',
    b: typeof second === 'string' ? second : '',
    ...(timeoutSec !== undefined ? { timeoutSec } : {}),
  };
}

export function applyCollisionFormToParams(
  original: Record<string, unknown>,
  form: CollisionFormState,
): Record<string, unknown> {
  const params: Record<string, unknown> = { ...original, between: [form.a, form.b] };
  const timeoutSec = normalizeTimeoutSec(form.timeoutSec ?? null);
  if (timeoutSec !== undefined) params['timeoutSec'] = timeoutSec;
  else delete params['timeoutSec'];
  return params;
}

// ── 순수 헬퍼: label / goto ─────────────────────────────────────────

export function labelNameOf(params: Record<string, unknown>): string {
  const name = params['name'];
  return typeof name === 'string' ? name : '';
}

/**
 * label 이름 커밋: trim만 하고 그대로 넘긴다 — 빈 이름/중복 등은 다운스트림
 * (commitParams의 serializeGraph 검증)이 한국어 오류로 거부한다 (요구 §2 label).
 */
export function applyLabelFormToParams(
  original: Record<string, unknown>,
  name: string,
): Record<string, unknown> {
  return { ...original, name: name.trim() };
}

export interface GotoFormState {
  label: string;
  /** undefined = 무한 반복 (times 키 생략 — DATA_MODEL §6) */
  times?: number;
}

/** times 정규화: 빈값/비유한 → 무한(undefined), 그 외 0 이상의 정수로 반올림 */
export function normalizeGotoTimes(value: number | null): number | undefined {
  if (value === null || !Number.isFinite(value)) return undefined;
  return Math.max(0, Math.round(value));
}

export function gotoFormStateOf(params: Record<string, unknown>): GotoFormState {
  const labelRaw = params['label'];
  const times = normalizeGotoTimes(toFiniteNumber(params['times']));
  return {
    label: typeof labelRaw === 'string' ? labelRaw : '',
    ...(times !== undefined ? { times } : {}),
  };
}

export function applyGotoFormToParams(
  original: Record<string, unknown>,
  form: GotoFormState,
): Record<string, unknown> {
  const params: Record<string, unknown> = { ...original, label: form.label };
  const times = normalizeGotoTimes(form.times ?? null);
  if (times !== undefined) params['times'] = times;
  else delete params['times'];
  return params;
}

// ── 내부 DOM 헬퍼 (얇은 조립 — 시각 언어는 entity-editor와 동일) ─────

function subCaption(text: string): HTMLElement {
  const el = applyType(document.createElement('div'), TYPE.caption);
  styled(el, { color: COLOR.muted, margin: `${SPACE.lg} 0 ${SPACE.xxs} 0`, whiteSpace: 'normal' });
  el.textContent = text;
  return el;
}

function mutedLine(text: string): HTMLElement {
  const el = applyType(document.createElement('div'), TYPE.body);
  styled(el, { color: COLOR.muted, padding: `${SPACE.xs} 0`, whiteSpace: 'normal' });
  el.textContent = text;
  return el;
}

/** 아이콘 + 문구 한 줄 (경고/안내 박스 본문 — 이모지·딩벳 대체, UX_AUDIT C-13) */
function iconLine(name: IconName, text: string): HTMLElement {
  const row = styled(document.createElement('div'), {
    display: 'flex',
    alignItems: 'flex-start',
    gap: SPACE.sm,
    padding: `${SPACE.hair} 0`,
  });
  const glyph = styled(document.createElement('span'), { display: 'flex', flex: 'none' });
  glyph.appendChild(icon(name, ICON.sm));
  const body = styled(document.createElement('span'), { whiteSpace: 'normal', minWidth: '0' });
  body.textContent = text;
  row.appendChild(glyph);
  row.appendChild(body);
  return row;
}

function addOption(select: HTMLSelectElement, value: string, label: string): void {
  const option = document.createElement('option');
  option.value = value;
  option.textContent = label;
  select.appendChild(option);
}

function ensureOption(select: HTMLSelectElement, value: string, label: string): void {
  for (const option of select.options) {
    if (option.value === value) return;
  }
  addOption(select, value, label);
}

function makeSelect(testId: string, ariaLabel: string): HTMLSelectElement {
  const select = document.createElement('select');
  select.className = 'ui-select';
  styled(select, { width: '100%', boxSizing: 'border-box' });
  select.dataset.testid = testId;
  select.setAttribute('aria-label', ariaLabel);
  return select;
}

/**
 * 슬라이더 + 값 갱신 함수. 채움은 **0 기준**이다(부호 있는 범위) — 네이티브 accentColor의
 * min→thumb 채움은 limits ±2.0에서 -0.600을 "35% 진행"처럼 읽히게 만든다(UX_AUDIT C-18).
 * `aria-valuetext`도 함께 갱신한다 — 숫자만 읽히면 단위(rad)가 전달되지 않는다.
 */
function makeSlider(
  minValue: number,
  maxValue: number,
  stepValue: number,
  testId: string,
  ariaLabel: string,
  unitText: string,
  decimals: number,
): { slider: HTMLInputElement; paint: (value: number) => void } {
  const slider = document.createElement('input');
  slider.type = 'range';
  slider.min = String(minValue);
  slider.max = String(maxValue);
  slider.step = String(stepValue);
  styled(slider, { width: '100%', minWidth: '0', margin: '0' });
  slider.dataset.testid = testId;
  slider.setAttribute('aria-label', ariaLabel);
  const paintFill = attachRangeFill(slider, minValue, maxValue);
  const paint = (value: number): void => {
    paintFill(value);
    slider.setAttribute('aria-valuetext', `${formatFixed(value, decimals)} ${unitText}`);
  };
  return { slider, paint };
}

function bareNumberInput(testId: string, ariaLabel: string, stepValue: number): HTMLInputElement {
  const input = document.createElement('input');
  input.type = 'number';
  input.className = 'ui-input ui-input--mono';
  input.step = String(stepValue);
  styled(input, {
    width: '100%',
    boxSizing: 'border-box',
    textAlign: 'right',
  });
  input.dataset.testid = testId;
  input.setAttribute('aria-label', ariaLabel);
  input.addEventListener('keydown', (e: KeyboardEvent) => {
    if (e.key === 'Enter') input.blur();
  });
  return input;
}

interface NumberFieldOptions {
  label: string;
  testId: string;
  step: number;
  min?: number;
  max?: number;
  placeholder?: string;
  /** change(Enter/blur/스피너)에서 1회 호출 — 커밋 지점 */
  onCommit(input: HTMLInputElement): void;
}

/**
 * 라벨 + number 입력 셀 (커밋은 change만 — 요구 §3).
 * 라벨은 `<label for>`로 입력에 연결된다 — 가시 텍스트가 곧 접근 가능한 이름이다
 * (WCAG 2.5.3 Label in Name, UX_AUDIT C-16).
 */
function numberField(options: NumberFieldOptions): { cell: HTMLElement; input: HTMLInputElement } {
  const cell = styled(document.createElement('div'), {
    display: 'flex',
    flexDirection: 'column',
    gap: SPACE.xxs,
    minWidth: '0',
  });
  const inputId = nextDomId(`rsw-${options.testId}`);
  const label = applyType(document.createElement('label'), TYPE.caption);
  styled(label, { color: COLOR.label, userSelect: 'none', whiteSpace: 'normal' });
  label.htmlFor = inputId;
  label.textContent = options.label;
  const input = bareNumberInput(options.testId, options.label, options.step);
  input.id = inputId;
  input.removeAttribute('aria-label');
  if (options.min !== undefined) input.min = String(options.min);
  if (options.max !== undefined) input.max = String(options.max);
  if (options.placeholder !== undefined) input.placeholder = options.placeholder;
  input.addEventListener('change', () => {
    options.onCommit(input);
  });
  cell.appendChild(label);
  cell.appendChild(input);
  return { cell, input };
}

/** 접기 가능한 섹션 (셰브론 회전 규약은 makePanelHeader가 소유 — UX_AUDIT C-18) */
function makeSection(
  titleText: string,
  testId: string,
): { section: HTMLElement; body: HTMLElement } {
  const section = styled(document.createElement('section'), {
    borderBottom: `${BORDER_WIDTH.hair} solid ${COLOR.borderSoft}`,
    padding: `0 0 ${SPACE.lg} 0`,
    // 섹션 경계는 16px 이상 — 4~8px로 뭉개면 그룹이 눈에 보이지 않는다
    margin: `0 0 ${SPACE.xl} 0`,
  });
  section.dataset.testid = testId;
  const header = makePanelHeader(titleText, { collapsible: true, headingTag: 'h3', testId });
  // section이 이미 testId를 갖는다 — 헤더의 중복 testid는 제거하고 토글 id만 남긴다
  delete header.el.dataset.testid;
  if (header.collapseButton !== null) header.collapseButton.dataset.testid = `${testId}-toggle`;
  styled(header.el, { padding: `${SPACE.xs} 0`, borderBottom: 'none' });
  applyType(header.titleEl, TYPE.subhead);
  // 영문 도메인 식별자 제목은 lang을 명시한다 — 한국어 TTS 철자 나열 방지 (WCAG 3.1.2)
  if (/^[\x20-\x7E]+$/.test(titleText)) header.titleEl.setAttribute('lang', 'en');
  const body = styled(document.createElement('div'), { padding: `${SPACE.xs} 0 0 0` });
  header.onToggle((collapsed) => {
    body.style.display = collapsed ? 'none' : '';
  });
  section.appendChild(header.el);
  section.appendChild(body);
  return { section, body };
}

// ── 마운트 ──────────────────────────────────────────────────────────

/**
 * 노드 파라미터 편집 폼을 host에 마운트한다. 배치(우측 스택 편입 등)는 통합자가
 * el로 결정한다. 폼 내부 포인터/휠/키 이벤트는 stopPropagation으로 흡수한다 —
 * 뷰포트 orbit·전역 단축키(Del 등)로 새지 않게 하는 기존 패널 규약과 동일.
 */
export function mountNodeEditor(host: HTMLElement, deps: NodeEditorDeps): NodeEditorHandle {
  ensureThemeStyles();
  const root = styled(document.createElement('div'), {
    width: '100%',
    boxSizing: 'border-box',
    background: COLOR.bgPanel,
    border: `${BORDER_WIDTH.hair} solid ${COLOR.border}`,
    borderRadius: RADIUS.md,
    boxShadow: SHADOW.panel,
    color: COLOR.text,
    userSelect: 'none',
  });
  applyType(root, TYPE.body);
  root.dataset.testid = 'node-editor';
  for (const type of ['pointerdown', 'pointermove', 'pointerup', 'wheel', 'contextmenu', 'keydown']) {
    root.addEventListener(type, (e) => {
      e.stopPropagation();
    });
  }

  // 헤더: kind 아이콘 + 타입명 + 노드 id (요구 §2 "kind 헤더 with icon").
  // 수제 헤더 대신 공용 팩토리 — 제목 굵기/보더/액션 정렬 규약이 한 곳에서 결정된다.
  const header = makePanelHeader('노드 편집', { actions: true, testId: 'node-editor-header' });
  const headerIcon = styled(document.createElement('span'), {
    display: 'flex',
    flex: 'none',
    color: COLOR.muted,
  });
  headerIcon.dataset.testid = 'ne-kind-icon';
  header.el.insertBefore(headerIcon, header.titleEl);
  const headerTitle = header.titleEl;
  headerTitle.dataset.testid = 'ne-kind';
  // kind 라벨은 영문 도메인 식별자를 포함한다 — 한국어 TTS가 철자 나열로 읽지 않도록
  // lang을 명시한다 (WCAG 3.1.2)
  headerTitle.setAttribute('lang', 'en');
  const headerId = applyType(document.createElement('span'), TYPE.monoMicro);
  styled(headerId, {
    color: COLOR.muted,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    minWidth: '0',
  });
  headerId.dataset.testid = 'ne-id';
  if (header.actionsEl !== null) {
    styled(header.actionsEl, { minWidth: '0', overflow: 'hidden' });
    header.actionsEl.appendChild(headerId);
  }
  root.appendChild(header.el);

  const content = styled(document.createElement('div'), {
    padding: `${SPACE.md} ${SPACE.lg} ${SPACE.lg} ${SPACE.lg}`,
    maxHeight: '65vh',
    overflowY: 'auto',
    minHeight: '0',
  });
  content.classList.add('ui-scroll');
  root.appendChild(content);

  // 빈 상태 플레이스홀더
  const placeholder = mutedLine('노드를 선택하면 파라미터를 편집할 수 있습니다');
  placeholder.dataset.testid = 'ne-placeholder';
  content.appendChild(placeholder);

  // 파라미터 섹션 (내용은 노드 종류별로 재구축)
  const paramsSection = makeSection('파라미터', 'ne-sec-params');
  const paramsBody = paramsSection.body;
  content.appendChild(paramsSection.section);

  // 커밋 오류 (인라인 한국어 — 불변식 §2.8: 무효 편집은 거부되고 폼은 복원된다)
  const errorBox = applyType(document.createElement('div'), TYPE.caption);
  styled(errorBox, {
    display: 'none',
    color: COLLISION.text,
    background: COLLISION.soft,
    border: `${BORDER_WIDTH.hair} solid ${COLLISION.border}`,
    borderRadius: RADIUS.sm,
    padding: `${SPACE.sm} ${SPACE.md}`,
    margin: `0 0 ${SPACE.lg} 0`,
    whiteSpace: 'normal',
  });
  // id는 aria-describedby의 대상이다 — 실패한 입력이 이 상자를 가리킨다 (UX_AUDIT C-16)
  errorBox.id = ERROR_BOX_ID;
  errorBox.dataset.testid = 'ne-errors';
  errorBox.setAttribute('role', 'alert');
  content.appendChild(errorBox);

  // 공통 섹션: 대상 로봇(읽기 전용) · 활성 토글 · 노트 (요구 §2 공통)
  const commonSection = makeSection('공통', 'ne-sec-common');
  const commonBody = commonSection.body;
  content.appendChild(commonSection.section);

  const robotRow = styled(document.createElement('div'), {
    display: 'flex',
    alignItems: 'baseline',
    gap: SPACE.sm,
    padding: `${SPACE.hair} 0`,
  });
  const robotLabel = applyType(document.createElement('span'), TYPE.caption);
  styled(robotLabel, { color: COLOR.label, flex: 'none' });
  robotLabel.textContent = '대상 로봇';
  const robotValue = applyType(document.createElement('span'), TYPE.monoReadout);
  styled(robotValue, {
    color: COLOR.textStrong,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  });
  robotValue.dataset.testid = 'ne-robot';
  const robotHint = applyType(document.createElement('span'), TYPE.micro);
  styled(robotHint, { color: COLOR.muted, flex: 'none' });
  robotHint.textContent = '(단일 로봇 MVP — 읽기 전용)';
  robotRow.appendChild(robotLabel);
  robotRow.appendChild(robotValue);
  robotRow.appendChild(robotHint);
  commonBody.appendChild(robotRow);

  const enabledRow = document.createElement('label');
  enabledRow.className = 'ui-check-label';
  styled(enabledRow, { color: COLOR.text, margin: `${SPACE.sm} 0 0 0`, whiteSpace: 'normal' });
  const enabledBox = document.createElement('input');
  enabledBox.type = 'checkbox';
  enabledBox.className = 'ui-check';
  enabledBox.dataset.testid = 'ne-enabled';
  const enabledText = document.createElement('span');
  enabledText.textContent = '활성 — 해제 시 실행에서 제외(순서 유지)';
  enabledRow.appendChild(enabledBox);
  enabledRow.appendChild(enabledText);
  commonBody.appendChild(enabledRow);

  const noteId = nextDomId('rsw-ne-note');
  const noteCaption = subCaption('노트');
  const noteLabel = document.createElement('label');
  noteLabel.htmlFor = noteId;
  noteLabel.textContent = '노트';
  noteCaption.replaceChildren(noteLabel);
  commonBody.appendChild(noteCaption);
  const noteArea = applyType(document.createElement('textarea'), TYPE.body);
  noteArea.className = 'ui-input';
  noteArea.id = noteId;
  noteArea.rows = NOTE_ROWS;
  noteArea.placeholder = '노드 메모 (실행 무관)';
  styled(noteArea, {
    width: '100%',
    boxSizing: 'border-box',
    resize: 'vertical',
  });
  noteArea.dataset.testid = 'ne-note';
  commonBody.appendChild(noteArea);

  // ── 상태 (뷰 상태는 ui 소유 — 그래프 진실은 deps 너머 flow-graph) ──
  let currentId: string | null = null;

  /** 마지막 커밋이 거부됐는가 — 폼 재구축 후 aria-invalid를 다시 입히기 위한 상태 */
  let commitRejected = false;

  /**
   * 커밋 실패를 **시각(빨강)뿐 아니라 프로그램적으로도** 알린다 (UX_AUDIT C-16):
   * 파라미터 컨트롤에 aria-invalid를 걸고 aria-describedby로 오류 상자를 가리킨다.
   * 폼은 커밋마다 재구축되므로 renderAll 뒤에 다시 호출해야 한다.
   */
  const paintInvalidState = (): void => {
    const controls = paramsBody.querySelectorAll<HTMLElement>('input, select, textarea');
    for (const control of controls) {
      if (commitRejected) {
        control.setAttribute('aria-invalid', 'true');
        control.setAttribute('aria-describedby', ERROR_BOX_ID);
      } else {
        control.removeAttribute('aria-invalid');
        control.removeAttribute('aria-describedby');
      }
    }
  };

  const showErrors = (errors: readonly string[]): void => {
    errorBox.replaceChildren();
    commitRejected = errors.length > 0;
    if (errors.length === 0) {
      errorBox.style.display = 'none';
      return;
    }
    errorBox.style.display = '';
    for (const message of errors) {
      errorBox.appendChild(iconLine('alert', message));
    }
  };

  /**
   * 커밋 공통 경로: 진실(getNode)의 params 위에 폼 상태를 적용해 commitParams.
   * 성공이든 실패든 renderAll()로 진실을 다시 그린다 — 성공 시 정규화 값 반영,
   * 실패 시 마지막 유효 값으로 복귀(불변식 §2.8: 그래프는 변하지 않았다).
   */
  const commitWith = (
    build: (original: Record<string, unknown>) => Record<string, unknown>,
  ): void => {
    if (currentId === null) return;
    const node = deps.getNode(currentId);
    if (node === null) return;
    const errors = deps.commitParams(currentId, build(node.params));
    showErrors(errors ?? []);
    renderAll();
    // 재구축된 컨트롤에 오류 상태를 다시 입힌다 (renderAll이 paramsBody를 갈아 끼운다)
    paintInvalidState();
  };

  // ── 재구축 중 포커스 복원 (커밋 → 재구축이 키보드 흐름을 끊지 않게) ─
  interface FocusKey {
    testId: string;
    joint: string | null;
  }
  const captureFocus = (): FocusKey | null => {
    const active = document.activeElement;
    if (!(active instanceof HTMLElement) || !root.contains(active)) return null;
    const testId = active.dataset.testid;
    if (testId === undefined) return null;
    const row = active.closest('[data-joint]');
    const joint = row instanceof HTMLElement ? (row.dataset.joint ?? null) : null;
    return { testId, joint };
  };
  const restoreFocus = (key: FocusKey | null): void => {
    if (key === null) return;
    const scope =
      key.joint !== null ? root.querySelector(`[data-joint="${CSS.escape(key.joint)}"]`) : root;
    if (!(scope instanceof Element)) return;
    const target = scope.querySelector(`[data-testid="${CSS.escape(key.testId)}"]`);
    if (target instanceof HTMLElement) target.focus();
  };

  // ── 종류별 파라미터 폼 ─────────────────────────────────────────────

  const buildJointsForm = (
    node: NodeEditorFlowNode,
    kind: 'moveJoints' | 'setJoints',
    context: NodeEditorSceneContext,
  ): void => {
    const form = jointsFormStateOf(node.params, context.robotJoints);
    const limitsMap = jointLimitsByName(context.robotJoints);

    interface RowRef {
      name: string;
      limits?: [number, number];
      input: HTMLInputElement;
      slider: HTMLInputElement;
    }
    const rowRefs: RowRef[] = [];

    /** 현재 DOM 입력값 → 관절 행 모델 (커밋 시점 스냅샷) */
    const readRows = (): JointRowState[] =>
      rowRefs.map((ref) => {
        const raw = parseNumeric(ref.input.value) ?? parseNumeric(ref.slider.value) ?? 0;
        const valueRad = clampJointValueRad(raw, ref.limits);
        return ref.limits !== undefined
          ? { name: ref.name, valueRad, limits: ref.limits }
          : { name: ref.name, valueRad };
      });

    // moveJoints 전용 컨트롤 (commitJoints가 읽는다 — setJoints면 null 유지)
    let durationInput: HTMLInputElement | null = null;
    let easingSelect: HTMLSelectElement | null = null;

    const commitJoints = (rows: JointRowState[]): void => {
      const jointsForm: JointsFormState = { rows };
      if (kind === 'moveJoints') {
        const fallback = form.durationSec ?? 0;
        jointsForm.durationSec =
          durationInput !== null ? (parseNumeric(durationInput.value) ?? fallback) : fallback;
        const easing = easingOf(easingSelect?.value);
        if (easing !== undefined) jointsForm.easing = easing;
      }
      commitWith((original) => applyJointsFormToParams(original, jointsForm, kind));
    };

    paramsBody.appendChild(subCaption(`관절 목표값 (rad · ${form.rows.length}개)`));
    if (form.rows.length === 0) {
      paramsBody.appendChild(mutedLine('관절이 없습니다 — 아래에서 추가하세요'));
    }
    for (const row of form.rows) {
      const range = jointSliderRange(row.limits);
      const container = styled(document.createElement('div'), {
        padding: `${SPACE.sm} 0`,
        borderBottom: `${BORDER_WIDTH.hair} solid ${COLOR.borderSoft}`,
      });
      container.dataset.testid = 'ne-joint-row';
      container.dataset.joint = row.name;

      const head = styled(document.createElement('div'), {
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: SPACE.sm,
      });
      const name = applyType(document.createElement('span'), TYPE.caption);
      styled(name, {
        color: COLOR.label,
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        whiteSpace: 'nowrap',
      });
      name.textContent = row.name;
      name.title =
        `${row.name} — 범위 [${formatFixed(range.minRad, JOINT_DECIMALS)}, ` +
        `${formatFixed(range.maxRad, JOINT_DECIMALS)}] rad` +
        (row.limits === undefined ? ' (limits 없음 — ±π 폴백)' : '');
      const removeButton = makeIconButton(
        'close',
        '',
        `관절 ${row.name} 행 제거`,
        'ne-joint-remove',
        'ghost',
        ICON.sm,
      );
      styled(removeButton, { flex: 'none' });
      head.appendChild(name);
      head.appendChild(removeButton);

      const controls = styled(document.createElement('div'), {
        display: 'grid',
        gridTemplateColumns: `minmax(0, 1fr) ${JOINT_INPUT_COL_PX}px`,
        gap: SPACE.sm,
        alignItems: 'center',
      });
      const { slider, paint: paintSlider } = makeSlider(
        range.minRad,
        range.maxRad,
        JOINT_SLIDER_STEP_RAD,
        'ne-joint-slider',
        `${row.name} 목표값 슬라이더 (rad)`,
        '라디안',
        JOINT_DECIMALS,
      );
      const input = bareNumberInput('ne-joint-value', `${row.name} 목표값 (rad)`, JOINT_SLIDER_STEP_RAD);
      const initial = clampJointValueRad(row.valueRad, row.limits);
      slider.value = String(initial);
      input.value = formatFixed(initial, JOINT_DECIMALS);
      paintSlider(initial);

      // 드래그 중에는 리드아웃만 갱신(라이브 프리뷰), 커밋은 release/blur (요구 §3)
      slider.addEventListener('input', () => {
        const value = parseNumeric(slider.value);
        if (value === null) return;
        input.value = formatFixed(value, JOINT_DECIMALS);
        // 채움(0 기준)과 aria-valuetext를 드래그 중에도 값과 동기화한다 (UX_AUDIT C-18)
        paintSlider(value);
      });
      slider.addEventListener('change', () => {
        commitJoints(readRows());
      });
      input.addEventListener('change', () => {
        commitJoints(readRows());
      });
      removeButton.addEventListener('click', () => {
        commitJoints(readRows().filter((r) => r.name !== row.name));
      });

      controls.appendChild(slider);
      controls.appendChild(input);
      container.appendChild(head);
      container.appendChild(controls);
      paramsBody.appendChild(container);
      rowRefs.push(
        row.limits !== undefined
          ? { name: row.name, limits: row.limits, input, slider }
          : { name: row.name, input, slider },
      );
    }

    // 미사용 관절 추가 드롭다운 (요구 §2)
    const available = availableJointNames(context.robotJoints, form.rows);
    const addSelect = makeSelect('ne-joint-add', '관절 추가');
    styled(addSelect, { marginTop: SPACE.md });
    addOption(addSelect, '', available.length > 0 ? '관절 추가…' : '추가할 관절 없음');
    for (const jointName of available) addOption(addSelect, jointName, jointName);
    addSelect.disabled = available.length === 0;
    addSelect.addEventListener('change', () => {
      const jointName = addSelect.value;
      if (jointName === '') return;
      const limits = limitsMap.get(jointName);
      const newRow: JointRowState =
        limits !== undefined
          ? { name: jointName, valueRad: clampJointValueRad(0, limits), limits }
          : { name: jointName, valueRad: 0 };
      commitJoints([...readRows(), newRow]);
    });
    paramsBody.appendChild(addSelect);

    if (kind === 'moveJoints') {
      const grid = styled(document.createElement('div'), {
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: `${SPACE.xs} ${SPACE.sm}`,
        marginTop: SPACE.lg,
      });
      const duration = numberField({
        label: 'durationSec (s)',
        testId: 'ne-duration',
        step: DURATION_STEP_SEC,
        min: 0,
        onCommit: (): void => {
          commitJoints(readRows());
        },
      });
      duration.input.value = formatFixed(form.durationSec ?? 0, DURATION_DECIMALS);
      durationInput = duration.input;

      const easingCell = styled(document.createElement('div'), {
        display: 'flex',
        flexDirection: 'column',
        gap: SPACE.xxs,
        minWidth: '0',
      });
      const easingId = nextDomId('rsw-ne-easing');
      const easingLabel = applyType(document.createElement('label'), TYPE.caption);
      styled(easingLabel, { color: COLOR.label });
      easingLabel.htmlFor = easingId;
      easingLabel.textContent = 'easing';
      const select = makeSelect('ne-easing', 'easing');
      select.id = easingId;
      select.removeAttribute('aria-label');
      addOption(select, '', '(기본 — linear)');
      addOption(select, 'linear', 'linear — 등속');
      addOption(select, 'easeInOut', 'easeInOut — 완급');
      addOption(select, 'step', 'step — 즉시');
      select.value = form.easing ?? '';
      select.addEventListener('change', () => {
        commitJoints(readRows());
      });
      easingSelect = select;
      easingCell.appendChild(easingLabel);
      easingCell.appendChild(select);

      grid.appendChild(duration.cell);
      grid.appendChild(easingCell);
      paramsBody.appendChild(grid);
    }
  };

  const buildGripperForm = (node: NodeEditorFlowNode, context: NodeEditorSceneContext): void => {
    const form = gripperFormStateOf(node.params);

    if (!context.gripperAvailable) {
      const warn = applyType(document.createElement('div'), TYPE.caption);
      styled(warn, {
        color: COLOR.warnText,
        border: `${BORDER_WIDTH.hair} solid ${COLOR.warn}`,
        borderRadius: RADIUS.sm,
        background: COLOR.warnSoft,
        padding: `${SPACE.sm} ${SPACE.md}`,
        margin: `0 0 ${SPACE.lg} 0`,
        whiteSpace: 'normal',
      });
      warn.dataset.testid = 'ne-grip-warn';
      warn.appendChild(iconLine('alert', '이 로봇엔 gripper 설정이 없습니다'));
      paramsBody.appendChild(warn);
    }

    let sliderRef: HTMLInputElement | null = null;
    let durationRef: HTMLInputElement | null = null;

    const commitGripper = (mode: GripperMode): void => {
      const valueRatio =
        sliderRef !== null ? clamp01(parseNumeric(sliderRef.value) ?? form.valueRatio) : form.valueRatio;
      const durationSec = durationRef !== null ? parseNumeric(durationRef.value) : null;
      const nextForm: GripperFormState = { mode, valueRatio };
      if (durationSec !== null) nextForm.durationSec = durationSec;
      commitWith((original) => applyGripperFormToParams(original, nextForm));
    };

    // 세그먼트 컨트롤 [열기|닫기|값] (요구 §2)
    paramsBody.appendChild(subCaption('상태'));
    const seg = styled(document.createElement('div'), { display: 'flex', gap: SPACE.xs });
    seg.setAttribute('role', 'group');
    seg.setAttribute('aria-label', '그리퍼 상태');
    const segEntries: Array<{ mode: GripperMode; label: string; testId: string }> = [
      { mode: 'open', label: '열기', testId: 'ne-grip-open' },
      { mode: 'close', label: '닫기', testId: 'ne-grip-close' },
      { mode: 'value', label: '값', testId: 'ne-grip-value' },
    ];
    for (const entry of segEntries) {
      const button = makeButton(entry.label, `그리퍼 상태: ${entry.label}`, entry.testId);
      styled(button, { flex: '1' });
      const active = entry.mode === form.mode;
      button.setAttribute('aria-pressed', String(active));
      if (active) button.classList.add('ui-btn--active');
      button.addEventListener('click', () => {
        if (entry.mode === form.mode) return;
        commitGripper(entry.mode);
      });
      seg.appendChild(button);
    }
    paramsBody.appendChild(seg);

    // 값 모드: 0..1 슬라이더 + 리드아웃 (드래그 중 프리뷰, release 커밋)
    if (form.mode === 'value') {
      const sliderRow = styled(document.createElement('div'), {
        display: 'grid',
        gridTemplateColumns: `minmax(0, 1fr) ${GRIPPER_READOUT_COL_PX}px`,
        gap: SPACE.sm,
        alignItems: 'center',
        marginTop: SPACE.lg,
      });
      // 0..1 단극 범위 — 채움은 min→thumb이 옳다(0이 곧 하한이므로 기준점이 맞다)
      const { slider, paint: paintSlider } = makeSlider(
        0,
        1,
        GRIPPER_SLIDER_STEP,
        'ne-grip-slider',
        '그리퍼 값 (0=닫힘, 1=열림)',
        '(0=닫힘, 1=열림)',
        GRIPPER_DECIMALS,
      );
      slider.value = String(form.valueRatio);
      paintSlider(form.valueRatio);
      const readout = applyType(document.createElement('span'), TYPE.monoReadout);
      styled(readout, { color: COLOR.textStrong, textAlign: 'right' });
      readout.dataset.testid = 'ne-grip-readout';
      readout.textContent = formatFixed(form.valueRatio, GRIPPER_DECIMALS);
      slider.addEventListener('input', () => {
        const value = parseNumeric(slider.value);
        if (value === null) return;
        readout.textContent = formatFixed(value, GRIPPER_DECIMALS);
        paintSlider(value);
      });
      slider.addEventListener('change', () => {
        commitGripper('value');
      });
      sliderRef = slider;
      sliderRow.appendChild(slider);
      sliderRow.appendChild(readout);
      paramsBody.appendChild(sliderRow);
      paramsBody.appendChild(subCaption('0 = 닫힘 · 1 = 열림'));
    }

    const duration = numberField({
      label: 'durationSec (s · 빈값=즉시)',
      testId: 'ne-grip-duration',
      step: DURATION_STEP_SEC,
      min: 0,
      placeholder: '(생략)',
      onCommit: (): void => {
        commitGripper(form.mode);
      },
    });
    if (form.durationSec !== undefined) {
      duration.input.value = formatFixed(form.durationSec, DURATION_DECIMALS);
    }
    durationRef = duration.input;
    styled(duration.cell, { marginTop: SPACE.lg });
    paramsBody.appendChild(duration.cell);
  };

  const buildWaitForm = (node: NodeEditorFlowNode): void => {
    const field = numberField({
      label: 'durationSec (s)',
      testId: 'ne-wait-duration',
      step: DURATION_STEP_SEC,
      min: 0,
      onCommit: (input): void => {
        commitWith((original) => applyWaitFormToParams(original, parseNumeric(input.value)));
      },
    });
    field.input.value = formatFixed(waitDurationOf(node.params), DURATION_DECIMALS);
    paramsBody.appendChild(field.cell);
  };

  const buildCollisionForm = (node: NodeEditorFlowNode, context: NodeEditorSceneContext): void => {
    const form = collisionFormStateOf(node.params);

    const makeEntitySelect = (
      labelText: string,
      testId: string,
      current: string,
    ): { cell: HTMLElement; select: HTMLSelectElement } => {
      const cell = styled(document.createElement('div'), {
        display: 'flex',
        flexDirection: 'column',
        gap: SPACE.xxs,
        minWidth: '0',
      });
      const selectId = nextDomId(`rsw-${testId}`);
      const label = applyType(document.createElement('label'), TYPE.caption);
      styled(label, { color: COLOR.label });
      label.htmlFor = selectId;
      label.textContent = labelText;
      const select = makeSelect(testId, labelText);
      select.id = selectId;
      select.removeAttribute('aria-label');
      if (current === '') addOption(select, '', '(선택)');
      for (const entityId of context.entityIds) addOption(select, entityId, entityId);
      if (current !== '') ensureOption(select, current, current);
      select.value = current;
      cell.appendChild(label);
      cell.appendChild(select);
      return { cell, select };
    };

    const a = makeEntitySelect('엔티티 A', 'ne-wfc-a', form.a);
    const b = makeEntitySelect('엔티티 B', 'ne-wfc-b', form.b);

    const commitPair = (): void => {
      const timeoutSec = normalizeTimeoutSec(parseNumeric(timeout.input.value));
      commitWith((original) =>
        applyCollisionFormToParams(original, {
          a: a.select.value,
          b: b.select.value,
          ...(timeoutSec !== undefined ? { timeoutSec } : {}),
        }),
      );
    };
    a.select.addEventListener('change', commitPair);
    b.select.addEventListener('change', commitPair);

    const timeout = numberField({
      label: 'timeoutSec (s · 빈값=무제한)',
      testId: 'ne-wfc-timeout',
      step: DURATION_STEP_SEC,
      min: 0,
      placeholder: '(무제한)',
      onCommit: (): void => {
        commitPair();
      },
    });
    if (form.timeoutSec !== undefined) {
      timeout.input.value = formatFixed(form.timeoutSec, DURATION_DECIMALS);
    }

    const grid = styled(document.createElement('div'), {
      display: 'grid',
      gridTemplateColumns: '1fr 1fr',
      gap: `${SPACE.xs} ${SPACE.sm}`,
    });
    grid.appendChild(a.cell);
    grid.appendChild(b.cell);
    paramsBody.appendChild(subCaption('접촉 대기 엔티티 쌍 (= 조작 대상 선언)'));
    paramsBody.appendChild(grid);
    styled(timeout.cell, { marginTop: SPACE.lg });
    paramsBody.appendChild(timeout.cell);
  };

  const buildLabelForm = (node: NodeEditorFlowNode): void => {
    const cell = styled(document.createElement('div'), {
      display: 'flex',
      flexDirection: 'column',
      gap: SPACE.xxs,
    });
    const inputId = nextDomId('rsw-ne-label-name');
    const label = applyType(document.createElement('label'), TYPE.caption);
    styled(label, { color: COLOR.label });
    label.htmlFor = inputId;
    label.textContent = 'label 이름';
    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'ui-input';
    input.id = inputId;
    styled(input, { width: '100%', boxSizing: 'border-box' });
    input.dataset.testid = 'ne-label-name';
    input.value = labelNameOf(node.params);
    input.addEventListener('change', () => {
      commitWith((original) => applyLabelFormToParams(original, input.value));
    });
    input.addEventListener('keydown', (e: KeyboardEvent) => {
      if (e.key === 'Enter') input.blur();
    });
    cell.appendChild(label);
    cell.appendChild(input);
    paramsBody.appendChild(cell);
    paramsBody.appendChild(
      subCaption('goto가 이 이름을 참조합니다 — 빈 이름/중복은 커밋 시 검증됩니다'),
    );
  };

  const buildGotoForm = (node: NodeEditorFlowNode, context: NodeEditorSceneContext): void => {
    const form = gotoFormStateOf(node.params);

    const labelCell = styled(document.createElement('div'), {
      display: 'flex',
      flexDirection: 'column',
      gap: SPACE.xxs,
      minWidth: '0',
    });
    const gotoSelectId = nextDomId('rsw-ne-goto-label');
    const labelCaption = applyType(document.createElement('label'), TYPE.caption);
    styled(labelCaption, { color: COLOR.label });
    labelCaption.htmlFor = gotoSelectId;
    labelCaption.textContent = '대상 label';
    const select = makeSelect('ne-goto-label', '대상 label');
    select.id = gotoSelectId;
    select.removeAttribute('aria-label');
    if (form.label === '') addOption(select, '', '(label 선택)');
    const seen = new Set<string>();
    for (const labelName of context.labels) {
      if (seen.has(labelName)) continue;
      seen.add(labelName);
      addOption(select, labelName, labelName);
    }
    if (form.label !== '') ensureOption(select, form.label, form.label);
    select.value = form.label;
    select.disabled = context.labels.length === 0 && form.label === '';
    labelCell.appendChild(labelCaption);
    labelCell.appendChild(select);

    const commitGoto = (): void => {
      const times = normalizeGotoTimes(parseNumeric(timesField.input.value));
      commitWith((original) =>
        applyGotoFormToParams(original, {
          label: select.value,
          ...(times !== undefined ? { times } : {}),
        }),
      );
    };
    select.addEventListener('change', commitGoto);

    const timesField = numberField({
      label: '반복 횟수 (빈값=무한)',
      testId: 'ne-goto-times',
      step: 1,
      min: 0,
      placeholder: '무한',
      onCommit: (): void => {
        commitGoto();
      },
    });
    if (form.times !== undefined) timesField.input.value = String(form.times);

    const grid = styled(document.createElement('div'), {
      display: 'grid',
      gridTemplateColumns: '1fr 1fr',
      gap: `${SPACE.xs} ${SPACE.sm}`,
    });
    grid.appendChild(labelCell);
    grid.appendChild(timesField.cell);
    paramsBody.appendChild(grid);
    if (context.labels.length === 0) {
      paramsBody.appendChild(mutedLine('시퀀스에 label 노드가 없습니다'));
    }
  };

  const buildPoseNotice = (): void => {
    const notice = applyType(document.createElement('div'), TYPE.caption);
    styled(notice, {
      color: COLOR.infoText,
      background: COLOR.infoSoft,
      borderRadius: RADIUS.sm,
      padding: `${SPACE.sm} ${SPACE.md}`,
      whiteSpace: 'normal',
    });
    notice.dataset.testid = 'ne-pose-note';
    notice.appendChild(iconLine('info', 'IK 백로그 — 실행 시 건너뜀'));
    paramsBody.appendChild(notice);
  };

  const buildParamsFor = (node: NodeEditorFlowNode): void => {
    paramsBody.replaceChildren();
    const context = deps.sceneContext();
    switch (node.kind) {
      case 'moveJoints':
      case 'setJoints':
        buildJointsForm(node, node.kind, context);
        break;
      case 'gripper':
        buildGripperForm(node, context);
        break;
      case 'wait':
        buildWaitForm(node);
        break;
      case 'waitForCollision':
        buildCollisionForm(node, context);
        break;
      case 'label':
        buildLabelForm(node);
        break;
      case 'goto':
        buildGotoForm(node, context);
        break;
      case 'moveToPose':
        buildPoseNotice();
        break;
      default:
        paramsBody.appendChild(
          mutedLine(`'${node.kind}' 노드는 파라미터 편집을 지원하지 않습니다`),
        );
        break;
    }
  };

  const paintCommon = (node: NodeEditorFlowNode): void => {
    const context = deps.sceneContext();
    robotValue.textContent = robotDisplayOf(node.params, context.robot);
    robotValue.title = robotValue.textContent;
    enabledBox.checked = node.enabled;
    if (document.activeElement !== noteArea) noteArea.value = node.note ?? '';
  };

  const renderAll = (): void => {
    const node = currentId === null ? null : deps.getNode(currentId);
    if (node === null) {
      headerIcon.replaceChildren(icon('workflow', ICON.md));
      headerTitle.textContent = '노드 편집';
      headerTitle.removeAttribute('lang');
      headerId.textContent = '';
      placeholder.style.display = '';
      paramsSection.section.style.display = 'none';
      commonSection.section.style.display = 'none';
      errorBox.style.display = 'none';
      return;
    }
    placeholder.style.display = 'none';
    paramsSection.section.style.display = '';
    commonSection.section.style.display = '';
    const meta = nodeKindMeta(node.kind);
    headerIcon.replaceChildren(icon(meta.icon, ICON.md));
    headerTitle.textContent = meta.label;
    // 라벨이 영문 도메인 식별자(MoveJoints…)를 포함한다 — 한국어 TTS 철자 나열 방지
    headerTitle.setAttribute('lang', 'en');
    headerTitle.title = meta.label;
    headerId.textContent = node.id;
    headerId.title = node.id;
    paintCommon(node);
    const focusKey = captureFocus();
    buildParamsFor(node);
    restoreFocus(focusKey);
  };

  enabledBox.addEventListener('change', () => {
    if (currentId === null) return;
    deps.setEnabled(currentId, enabledBox.checked);
    renderAll();
  });
  noteArea.addEventListener('change', () => {
    if (currentId === null) return;
    deps.setNote(currentId, noteArea.value);
  });

  renderAll(); // 초기 빈 상태
  host.appendChild(root);

  return {
    el: root,
    showFor: (id): void => {
      currentId = id;
      showErrors([]);
      renderAll();
    },
    refresh: (): void => {
      // 폼 편집 중(포커스가 폼 안)이면 재구축을 건너뛴다 — 입력 중 값 클로버 방지.
      // 사용자의 다음 커밋이 어차피 renderAll로 진실을 다시 그린다.
      const active = document.activeElement;
      if (active instanceof HTMLElement && root.contains(active)) return;
      renderAll();
    },
    dispose: (): void => {
      currentId = null;
      root.remove();
    },
  };
}
