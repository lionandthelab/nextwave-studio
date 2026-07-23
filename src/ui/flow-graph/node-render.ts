// ui/flow-graph/node-render.ts — step 종류별 노드 표현 헬퍼 (Phase 8, UX_DESIGN §3.4)
//
// 캔버스(canvas.ts)가 노드를 그릴 때 쓰는 순수 데이터/문자열 헬퍼만 담는다 — DOM 비의존,
// node 환경 단위 테스트 대상(canvas.test.ts). 시각 색은 ui/theme.ts 토큰만 소비한다.
//
// - kindMeta: step 종류 → 아이콘/표시명/팔레트 그룹/색 (UX §3.4 "좌측 아이콘 + 타입명")
// - nodeSummary: step params → 요약 한 줄 (예: moveJoints → 'joint2→0.2 외 1 · 2.0s')
//   params는 신뢰하지 않는 Record<string, unknown>이다 — 형태가 어긋나도 throw하지 않고
//   그릴 수 있는 부분만 그린다. 값 정합성 보증은 schema/flow-graph(직렬화 검증, §2.8)의
//   소관이며, 이 모듈은 "표시"만 책임진다.
// - statusColor/statusLabelKo: 실행 상태 점 색 + 텍스트 라벨(색만으로 전달 금지 — UX §9)
// - originBadge: 출처 배지 텍스트 (generated → 'AI', modified → '수정됨' — UX §3.4)
// - PALETTE_GROUPS: ＋ 삽입 팔레트의 step 분류 (동작/시간/충돌/흐름 — Phase 8 요구)

import type { ControlStepKind } from '../../schema/types';
import { COLOR } from '../theme';

// ── 숫자/시간 포맷 (요약 전용 — 데이터 값은 건드리지 않는다) ─────────

/** 요약 숫자 반올림 자릿수 — 부동소수 잡음 제거 (0.30000000000000004 → '0.3') */
const SUMMARY_NUM_DECIMALS = 3;

/** 요약용 숫자 문자열: 소수 3자리 반올림 + 뒤 0 제거 */
export function formatNum(value: number): string {
  const factor = 10 ** SUMMARY_NUM_DECIMALS;
  return String(Math.round(value * factor) / factor);
}

/**
 * duration 표기: 0.1s 해상도로 떨어지면 소수 1자리 고정('2.0s', '1.0s'),
 * 더 세밀한 값은 그대로('0.25s'). UX §3.4 예시("MoveJoints · 2.0s")와 일치.
 */
export function formatDurationSec(sec: number): string {
  const tenth = Math.round(sec * 10) / 10;
  if (Math.abs(tenth - sec) < 1e-9) return `${tenth.toFixed(1)}s`;
  return `${formatNum(sec)}s`;
}

// ── step 종류 메타 (아이콘 · 표시명 · 팔레트 그룹 · 색) ──────────────

export type PaletteGroupKo = '동작' | '시간' | '충돌' | '흐름';

export interface StepKindMeta {
  /** 노드 좌측 아이콘 글리프 */
  readonly icon: string;
  /** 노드 타입 표시명 (UX §2 목업의 'MoveJoint' 류 — kind의 PascalCase) */
  readonly label: string;
  /** ＋ 팔레트 분류 */
  readonly groupKo: PaletteGroupKo;
  /** 종류 색 (노드 좌측 컬러 스트립) — theme 토큰 */
  readonly color: string;
  /** 팔레트 툴팁/빈 요약 대체용 한국어 설명 */
  readonly descriptionKo: string;
}

const KIND_META: Readonly<Record<string, StepKindMeta>> = {
  moveJoints: {
    icon: '🦾',
    label: 'MoveJoints',
    groupKo: '동작',
    color: COLOR.accent,
    descriptionKo: '관절을 목표값으로 보간 이동',
  },
  setJoints: {
    icon: '⚙',
    label: 'SetJoints',
    groupKo: '동작',
    color: COLOR.accent,
    descriptionKo: '관절값 즉시 설정',
  },
  gripper: {
    icon: '✋',
    label: 'Gripper',
    groupKo: '동작',
    color: COLOR.accent,
    descriptionKo: '그리퍼 열기/닫기',
  },
  moveToPose: {
    icon: '🎯',
    label: 'MoveToPose',
    groupKo: '동작',
    color: COLOR.accent,
    descriptionKo: '카테시안 목표 이동 (로드맵 — 현재 실행 시 건너뜀)',
  },
  wait: {
    icon: '⏱',
    label: 'Wait',
    groupKo: '시간',
    color: COLOR.info,
    descriptionKo: '지정 시간 대기',
  },
  waitForCollision: {
    icon: '💥',
    label: 'WaitForCollision',
    groupKo: '충돌',
    color: COLOR.warn,
    descriptionKo: '두 엔티티의 충돌이 감지될 때까지 대기',
  },
  label: {
    icon: '🔖',
    label: 'Label',
    groupKo: '흐름',
    color: COLOR.success,
    descriptionKo: 'goto 점프 대상 라벨',
  },
  goto: {
    icon: '↩',
    label: 'Goto',
    groupKo: '흐름',
    color: COLOR.success,
    descriptionKo: '라벨로 점프 (반복)',
  },
};

/** 알 수 없는 kind의 대체 메타 (label은 kind 문자열 그대로) */
const UNKNOWN_KIND_META: Omit<StepKindMeta, 'label'> = {
  icon: '□',
  groupKo: '동작',
  color: COLOR.muted,
  descriptionKo: '알 수 없는 step 종류',
};

/** step kind → 표현 메타. 알 수 없는 kind도 안전하게 표시한다(throw 없음). */
export function kindMeta(kind: string): StepKindMeta {
  return KIND_META[kind] ?? { ...UNKNOWN_KIND_META, label: kind };
}

// ── ＋ 팔레트 그룹 (UX §3.4 "step 종류별 분류") ─────────────────────
//
// moveToPose는 의도적으로 제외한다 — IK 백로그(ROADMAP)라 player가 경고 후 건너뛰는
// step을 팔레트에서 새로 만들도록 권하지 않는다. (기존 시퀀스/플래너 출력에 들어 있는
// moveToPose 노드는 kindMeta로 정상 표시된다.)

export interface PaletteGroup {
  readonly labelKo: PaletteGroupKo;
  readonly kinds: readonly ControlStepKind[];
}

export const PALETTE_GROUPS: readonly PaletteGroup[] = [
  { labelKo: '동작', kinds: ['moveJoints', 'setJoints', 'gripper'] },
  { labelKo: '시간', kinds: ['wait'] },
  { labelKo: '충돌', kinds: ['waitForCollision'] },
  { labelKo: '흐름', kinds: ['label', 'goto'] },
];

// ── params 안전 접근 (unknown → 표시 가능한 값만) ────────────────────

function numField(params: Record<string, unknown>, key: string): number | undefined {
  const value = params[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function strField(params: Record<string, unknown>, key: string): string | undefined {
  const value = params[key];
  return typeof value === 'string' ? value : undefined;
}

/** targets 객체에서 유한한 숫자 값 엔트리만 (선언 순서 유지 — 결정론) */
function jointTargets(params: Record<string, unknown>): [string, number][] {
  const value = params['targets'];
  if (value === null || typeof value !== 'object' || Array.isArray(value)) return [];
  const out: [string, number][] = [];
  for (const [name, target] of Object.entries(value as Record<string, unknown>)) {
    if (typeof target === 'number' && Number.isFinite(target)) out.push([name, target]);
  }
  return out;
}

/** 'joint2→0.2 외 1' — 첫 관절 + 나머지 개수 (UX §3.4 요약 형식) */
function targetsText(targets: [string, number][]): string {
  const first = targets[0];
  if (first === undefined) return '';
  const extra = targets.length - 1;
  const head = `${first[0]}→${formatNum(first[1])}`;
  return extra > 0 ? `${head} 외 ${extra}` : head;
}

// ── 노드 요약 한 줄 ─────────────────────────────────────────────────

/** 요약 조각 구분자 */
const SUMMARY_SEP = ' · ';

/**
 * step 종류별 요약 텍스트. 형태가 어긋난 params에도 throw하지 않고 그릴 수 있는
 * 부분만 조합한다 (없으면 빈 문자열).
 */
export function nodeSummary(kind: string, params: Record<string, unknown>): string {
  switch (kind) {
    case 'moveJoints': {
      const parts: string[] = [];
      const targets = targetsText(jointTargets(params));
      if (targets !== '') parts.push(targets);
      const duration = numField(params, 'durationSec');
      if (duration !== undefined) parts.push(formatDurationSec(duration));
      return parts.join(SUMMARY_SEP);
    }
    case 'setJoints':
      return targetsText(jointTargets(params));
    case 'gripper': {
      const parts: string[] = [];
      const state = params['state'];
      if (state === 'open') parts.push('열기');
      else if (state === 'close') parts.push('닫기');
      else if (typeof state === 'number' && Number.isFinite(state)) parts.push(formatNum(state));
      const duration = numField(params, 'durationSec');
      if (duration !== undefined) parts.push(formatDurationSec(duration));
      return parts.join(SUMMARY_SEP);
    }
    case 'wait': {
      const duration = numField(params, 'durationSec');
      return duration !== undefined ? formatDurationSec(duration) : '';
    }
    case 'waitForCollision': {
      const parts: string[] = [];
      const between = params['between'];
      if (
        Array.isArray(between) &&
        typeof between[0] === 'string' &&
        typeof between[1] === 'string'
      ) {
        parts.push(`${between[0]} × ${between[1]}`);
      }
      const timeout = numField(params, 'timeoutSec');
      if (timeout !== undefined) parts.push(`${formatNum(timeout)}s`);
      return parts.join(SUMMARY_SEP);
    }
    case 'label':
      return strField(params, 'name') ?? '';
    case 'goto': {
      const label = strField(params, 'label');
      if (label === undefined) return '';
      const times = numField(params, 'times');
      // times 미지정 = 무한 반복 (DATA_MODEL §6)
      return times === undefined ? `→ ${label} ∞` : `→ ${label} ×${formatNum(times)}`;
    }
    case 'moveToPose': {
      const parts: string[] = [];
      const target = params['target'];
      if (target !== null && typeof target === 'object' && !Array.isArray(target)) {
        const position = (target as Record<string, unknown>)['position'];
        if (
          Array.isArray(position) &&
          position.length === 3 &&
          position.every((v): v is number => typeof v === 'number' && Number.isFinite(v))
        ) {
          parts.push(`[${position.map((v) => formatNum(v)).join(', ')}]`);
        }
      }
      const duration = numField(params, 'durationSec');
      if (duration !== undefined) parts.push(formatDurationSec(duration));
      return parts.join(SUMMARY_SEP);
    }
    default:
      return '';
  }
}

// ── 실행 상태 (UX §3.4 상태 점 — 색 + 텍스트 병행) ──────────────────

export type NodeRunStatus = 'pending' | 'active' | 'done' | 'error';

export function statusColor(status: NodeRunStatus): string {
  switch (status) {
    case 'active':
      return COLOR.accent;
    case 'done':
      return COLOR.success;
    case 'error':
      return COLOR.error;
    case 'pending':
      return COLOR.muted;
  }
}

/** 상태의 텍스트 채널 (aria/title — 색만으로 전달 금지, UX §9) */
export function statusLabelKo(status: NodeRunStatus): string {
  switch (status) {
    case 'active':
      return '실행중';
    case 'done':
      return '완료';
    case 'error':
      return '오류';
    case 'pending':
      return '대기';
  }
}

// ── 출처 배지 (UX §3.4) ─────────────────────────────────────────────

/** origin → 배지 텍스트. 'manual'은 배지 없음(null). */
export function originBadge(origin: string): string | null {
  if (origin === 'generated') return 'AI';
  if (origin === 'modified') return '수정됨';
  return null;
}

// ── 텍스트 자르기 (SVG <text>에는 ellipsis가 없다 — 수동 절단) ───────

export function truncateText(text: string, maxChars: number): string {
  if (text.length <= maxChars) return text;
  if (maxChars <= 1) return '…';
  return `${text.slice(0, maxChars - 1)}…`;
}
