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
// - nodeLod: 줌 배율 → 노드 상세도 정책 (UX_AUDIT C-10 확장성 천장)
//
// ── 이 모듈은 DOM에 의존하지 않는다 (canvas.test.ts는 node 환경에서 돈다) ──────
// LOD는 **정책만** 여기서 소유하고(nodeLod/LOD_ZOOM_THRESHOLD), SVG 분기 그리기는
// canvas.ts의 drawNode가 이 정책을 소비해 수행한다. icons.ts에서 가져오는 것도
// `IconName` 타입뿐이다 — 값 import가 아니므로 node 환경에서 안전하다.

import type { ControlStepKind } from '../../schema/types';
import type { IconName } from '../icons';
import { CATEGORY, COLLISION, COLOR } from '../theme';

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
  /**
   * 노드 좌측 아이콘 (icons.ts의 SVG 세트 — UX_AUDIT C-13).
   * 이모지가 아니라 `IconName`이므로 kind를 추가하면서 아이콘을 빠뜨리면 **컴파일이 깨진다.**
   * 아이콘 자체에는 색을 칠하지 않는다(currentColor 상속) — 범주 색은 좌측 스트립이 든다.
   */
  readonly icon: IconName;
  /** 노드 타입 표시명 (UX §2 목업의 'MoveJoint' 류 — kind의 PascalCase) */
  readonly label: string;
  /** ＋ 팔레트 분류 */
  readonly groupKo: PaletteGroupKo;
  /**
   * 범주 색 (노드 좌측 컬러 스트립) — **CATEGORY 토큰만 쓴다.**
   * 시맨틱 토큰(success/warn/info)을 범주형으로 재사용하면 "초록 노드"가 성공인지 흐름
   * 제어인지 구분되지 않고, 그 초록이 타임라인 완료 칩의 초록과 같은 값이 된다(C-14).
   */
  readonly color: string;
  /** 팔레트 툴팁/빈 요약 대체용 한국어 설명 */
  readonly descriptionKo: string;
}

const KIND_META: Readonly<Record<string, StepKindMeta>> = {
  moveJoints: {
    icon: 'robotArm',
    label: 'MoveJoints',
    groupKo: '동작',
    color: CATEGORY.motion,
    descriptionKo: '관절을 목표값으로 보간 이동',
  },
  setJoints: {
    icon: 'settings',
    label: 'SetJoints',
    groupKo: '동작',
    color: CATEGORY.motion,
    descriptionKo: '관절값 즉시 설정',
  },
  gripper: {
    icon: 'gripper',
    label: 'Gripper',
    groupKo: '동작',
    color: CATEGORY.motion,
    descriptionKo: '그리퍼 열기/닫기',
  },
  moveToPose: {
    icon: 'target',
    label: 'MoveToPose',
    groupKo: '동작',
    color: CATEGORY.motion,
    descriptionKo: '카테시안 목표 이동 (로드맵 — 현재 실행 시 건너뜀)',
  },
  wait: {
    icon: 'timer',
    label: 'Wait',
    groupKo: '시간',
    color: CATEGORY.time,
    descriptionKo: '지정 시간 대기',
  },
  waitForCollision: {
    icon: 'impact',
    label: 'WaitForCollision',
    groupKo: '충돌',
    color: CATEGORY.collision,
    descriptionKo: '두 엔티티의 충돌이 감지될 때까지 대기',
  },
  label: {
    icon: 'bookmark',
    label: 'Label',
    groupKo: '흐름',
    color: CATEGORY.flow,
    descriptionKo: 'goto 점프 대상 라벨',
  },
  goto: {
    icon: 'refresh',
    label: 'Goto',
    groupKo: '흐름',
    color: CATEGORY.flow,
    descriptionKo: '라벨로 점프 (반복)',
  },
};

/** 알 수 없는 kind의 대체 메타 (label은 kind 문자열 그대로) */
const UNKNOWN_KIND_META: Omit<StepKindMeta, 'label'> = {
  icon: 'help',
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

/**
 * 상태 점 색. 4단계는 서로 다른 축의 토큰을 쓴다 —
 * pending은 중립(muted), active는 액센트(펄스가 병행), done은 성공, error는 **충돌 램프**
 * (이 제품의 오류는 곧 충돌 사건이며 3D 펄스·접촉점 마커와 같은 램프를 써야 한다 — C-7).
 */
export function statusColor(status: NodeRunStatus): string {
  switch (status) {
    case 'active':
      return COLOR.accent;
    case 'done':
      return COLOR.success;
    case 'error':
      return COLLISION.base;
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

// ── LOD 정책 (UX_AUDIT C-10 — 확장성 천장) ──────────────────────────
//
// 노드 라벨 예산은 15자다. 줌 0.5에서 11px 활자는 화면상 5.5px가 되어 **어차피 판독이
// 불가능하다** — 그 배율 아래에서 텍스트를 계속 그리는 것은 프레임 예산만 태우고
// 형태 인지를 방해한다. 그래서 아이콘 + 범주 색 칩으로 축약한다(compact).
//
// 축약은 **시각 채널만** 줄인다. `aria-label`은 두 LOD에서 동일한 전체 요약을 유지하므로
// 스크린리더 사용자에게는 줌 배율이 아무 영향도 주지 않는다.

export type NodeLod = 'compact' | 'full';

/** 이 배율 미만에서 노드를 아이콘 칩으로 축약한다 */
export const LOD_ZOOM_THRESHOLD = 0.5;

/** 줌 배율 → 노드 상세도. NaN은 안전하게 full로 본다. */
export function nodeLod(zoom: number): NodeLod {
  return zoom < LOD_ZOOM_THRESHOLD ? 'compact' : 'full';
}
