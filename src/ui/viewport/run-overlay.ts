// ui/viewport/run-overlay.ts — 뷰포트 좌하단 실행 오버레이 (UX_DESIGN §3.3 "실행 오버레이",
// §5 "동기 강조"). Phase 10 오케스트레이션 배지.
//
// 뷰포트 좌하단에 '● Running · node 3/7 · simTime 2.34s · <scene>' 형태의 실행 상태를
// 표시한다. Phase-6 statusline(mountViewportStatus)의 후신/보강판이다 — 통합자(main.ts)가
// engine.onTick + player.onStepChange를 엮어 매 tick setState(...)로 상태를 밀어넣는다.
// 이 모듈은 순수 뷰다: core/schema를 import하지 않고(계층 규칙 CLAUDE.md §3), 상태를
// 스스로 계산하지 않는다 — 통합자가 이미 계산한 스냅샷만 그린다(§5 "활성 노드 강조 ↔
// 로봇 동작 ↔ Timeline 커서가 항상 일치"의 뷰포트 한 축).
//
// 순수 표시 전용: pointer-events:none으로 뷰포트 orbit/선택을 절대 가로막지 않는다.
// 배치 계약: host는 positioned(position:relative 등) 뷰포트 슬롯이어야 한다 — 오버레이는
// host 기준 absolute 좌하단에 앉는다.
//
// 접근성 (UX §9): 상태는 색만으로 전달하지 않는다 — 펄스 점(색) + 텍스트 라벨(Running/
// Paused/Idle)을 병행한다. 시각 라인은 aria-hidden으로 두고, 별도 sr-only live 영역
// (role=status, aria-live=polite)이 "의미 있는 전이"(재생 상태·노드 진행 변화)에서만
// 갱신되어 스크린리더를 rAF마다(simTime 변화) 범람시키지 않는다.
//
// ── 충돌 발화 (UX_AUDIT C-7) ─────────────────────────────────────────
// 충돌 시 일어나는 일이 (a) 로그 표 행 추가 (b) 3D 빨강 펄스 (c) 접촉점 마커 — 셋 다
// 순수 시각이라 비시각 사용자에게는 이 제품의 핵심 사건이 **존재하지 않았다**
// (WCAG 4.1.3 Status Messages 실패). 그래서 오버레이가 충돌 요약을 발화한다.
//
// **충돌 로그 스트림(<tbody>)에 aria-live를 거는 것은 금물이다.** 물리 스텝마다
// 이벤트가 쏟아지면 polite 큐는 취소되지 않고 누적되므로, 사용자가 다른 조작을 해도
// 몇 분간 과거 충돌만 읽는다. 링버퍼의 앞 행 제거도 "변경"으로 집계되어 중복 발화한다.
// 스트림이 아니라 **요약을 스로틀해** 알린다 — a11y.ts의 createAnnouncer(3초 게이트).
// 같은 이유로 충돌 건수는 announceSignature(전이 축)에 **넣지 않는다**: 시그니처에
// 들어가면 충돌 1건마다 즉시 발화가 되어 스로틀이 무력해진다.

import {
  BORDER,
  BORDER_WIDTH,
  COLLISION,
  COLOR,
  RADIUS,
  SELECT,
  SPACE,
  SURFACE,
  TYPE,
  Z_INDEX,
  applyType,
  ensureThemeStyles,
  styled,
} from '../theme';
import { createAnnouncer } from '../a11y';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4) ─────────────────────────────

/** 뷰포트 슬롯 좌하단 오프셋 (statusline.ts와 동일 규약 — 독 행과 겹치지 않는다) */
const OVERLAY_BOTTOM_PX = 12;
const OVERLAY_LEFT_PX = 12;
/** simTime 표시 소수 자릿수 (statusline/playback과 동일 규약) */
const SIM_TIME_DECIMALS = 2;
/** 세그먼트 구분자 (시각 라인 · 요약 텍스트 공용) */
const SEGMENT_SEP = ' · ';
/** 선택 리드아웃 접두어 — '선택 arm_left' */
const SELECTION_PREFIX = '선택 ';
/** 씬 이름/선택 id 리드아웃 최대 폭 */
const READOUT_MAX_WIDTH_PX = 220;
/**
 * 충돌 요약 발화 최소 간격(ms). 이 안의 연속 충돌은 마지막 문구 하나로 합쳐진다
 * (a11y.ts createAnnouncer 계약) — polite 큐 포화를 구조적으로 막는 유일한 장치다.
 */
export const COLLISION_ANNOUNCE_MIN_INTERVAL_MS = 3000;

// ── 공개 타입 (통합자 계약) ─────────────────────────────────────────

/** 정규화된 재생 상태 (engineState 문자열은 이 3값으로 접힌다 — 미지 값은 idle) */
export type OverlayEngineState = 'idle' | 'playing' | 'paused';

/**
 * 오버레이 1틱 상태 스냅샷. engineState는 느슨한 string으로 받는다(Engine의
 * EngineState와 구조적으로 호환되나 core 타입에 결합하지 않는다 — 순수 뷰).
 * 시퀀스가 없는 씬은 nodeIndex/nodeCount/activeNodeLabel이 모두 null이다.
 */
export interface RunOverlayState {
  engineState: string;
  simTimeSec: number;
  /** 현재 활성 노드의 표시 라벨 (예: 'MoveJoints'). 시퀀스 없음/미재생이면 null */
  activeNodeLabel: string | null;
  /** 현재 노드 인덱스(0-기반). 시퀀스 없으면 null */
  nodeIndex: number | null;
  /** 총 노드 수. 시퀀스 없으면 null */
  nodeCount: number | null;
  /** 씬 이름 (spec.name) */
  sceneName: string;
  /**
   * 현재 뷰포트 선택 엔티티 id (없으면 null). 선택 상태를 알 수 있는 곳이 스크롤될 수
   * 있는 우측 패널뿐이면, 빈 곳 클릭으로 선택이 풀린 줄 모른 채 방향키만 누르는
   * 실패 연쇄가 생긴다 — 뷰포트에 상시 표시한다 (UX_DESIGN §3.3 선택 피드백).
   */
  selectedEntityId?: string | null;
  /**
   * 누적 충돌 건수 (UX_AUDIT C-7). 이 제품의 존재 이유가 화면에 등장하는 축이다.
   * 미주입/0이면 세그먼트도 발화도 없다 — 기존 소비자의 요약 문자열 계약이 바뀌지 않는다.
   */
  collisionCount?: number;
  /** 최근 충돌 쌍 표시 문자열 (예: 'arm × box_a'). 없으면 null */
  lastCollisionPair?: string | null;
  /**
   * 시퀀스가 방금 끝났는가. 통합자가 알면 명시하고, 미주입이면 playing→idle 전이로
   * 추정한다 — 완주/정지 어느 쪽이든 "무슨 일이 있었는지"를 1회 요약 발화하기 위함이다.
   */
  sequenceDone?: boolean;
}

export interface RunOverlayHandle {
  /** 상태 갱신 (통합자가 engine.onTick 등에서 매 tick 호출) */
  setState(s: RunOverlayState): void;
  dispose(): void;
}

// ── 순수 헬퍼 (DOM 비의존 — run-overlay.test.ts 대상) ────────────────

const STATE_LABELS: Readonly<Record<OverlayEngineState, string>> = {
  playing: 'Running',
  paused: 'Paused',
  idle: 'Idle',
};

/** 느슨한 engineState 문자열을 3상태로 정규화한다 (미지 값은 idle으로 방어) */
export function normalizeEngineState(state: string): OverlayEngineState {
  return state === 'playing' || state === 'paused' ? state : 'idle';
}

/** 상태 라벨 텍스트 (색만으로 전달 금지 — UX §9의 텍스트 채널) */
export function stateLabel(state: string): string {
  return STATE_LABELS[normalizeEngineState(state)];
}

/**
 * 상태 → 라벨 텍스트 색 토큰 (theme). playing=액센트(밝은 변형)·paused=warn·idle=muted.
 * 펄스는 dot 클래스가, 색+텍스트 병행은 이 값 + stateLabel이 담당한다.
 */
export function stateColor(state: string): string {
  switch (normalizeEngineState(state)) {
    case 'playing':
      return COLOR.accentText;
    case 'paused':
      return COLOR.warn;
    case 'idle':
      return COLOR.muted;
  }
}

/** 상태 → 펄스 점 클래스 (theme .ui-dot 계열 — playing만 펄스 애니메이션) */
export function stateDotClass(state: string): string {
  return `ui-dot ui-dot--${normalizeEngineState(state)}`;
}

/**
 * 'node N/M' 진행 텍스트. nodeIndex(0-기반)를 1-기반 표시로 올리고 [1, nodeCount]로
 * 클램프한다(시퀀스 끝 인덱스 == nodeCount여도 'node M/M'로 보인다). 시퀀스가 없거나
 * (index/count null) count가 양수가 아니면 null(세그먼트 미표시).
 */
export function formatNodeProgress(
  nodeIndex: number | null,
  nodeCount: number | null,
): string | null {
  if (nodeIndex === null || nodeCount === null) return null;
  if (!Number.isFinite(nodeIndex) || !Number.isFinite(nodeCount) || nodeCount <= 0) return null;
  const count = Math.floor(nodeCount);
  const shown = Math.min(Math.max(Math.floor(nodeIndex) + 1, 1), count);
  return `node ${shown}/${count}`;
}

/** simTime 리드아웃 텍스트 (비유한 입력은 0으로 방어) */
export function formatSimTime(simTimeSec: number): string {
  const safe = Number.isFinite(simTimeSec) ? simTimeSec : 0;
  return `simTime ${safe.toFixed(SIM_TIME_DECIMALS)}s`;
}

/**
 * 충돌 시각(timeSec) → 그 시점에 활성이던 노드 인덱스 (Collision Log 행 → 노드 강조 연동,
 * §3.6). boundaries[i] = 노드 i가 active가 된 simTime(오름차순 가정). t 이하인 마지막
 * 경계의 인덱스를 돌려준다:
 * - t가 첫 경계보다 앞서면 0(재생 시작 전 — 가장 가까운 노드는 첫 노드),
 * - 경계가 비어 있으면 -1(노드 없음).
 * 오름차순이므로 첫 초과 경계에서 멈춘다(선형 스캔 — MVP 규모 ≤ 수십 노드).
 */
export function timeSecToNodeIndex(timeSec: number, boundaries: readonly number[]): number {
  if (boundaries.length === 0) return -1;
  let index = 0;
  for (let i = 0; i < boundaries.length; i += 1) {
    const boundary = boundaries[i];
    if (boundary === undefined) continue; // noUncheckedIndexedAccess 방어
    if (boundary <= timeSec) index = i;
    else break;
  }
  return index;
}

/** '선택 <id>' 리드아웃 세그먼트 (선택 없으면 null — 세그먼트 미표시) */
export function formatSelection(selectedEntityId: string | null | undefined): string | null {
  if (selectedEntityId === null || selectedEntityId === undefined || selectedEntityId === '') {
    return null;
  }
  return `${SELECTION_PREFIX}${selectedEntityId}`;
}

/**
 * '충돌 N건' 세그먼트.
 *
 * **0건도 표시한다** — 0은 "감지가 돌고 있는데 아직 없다"는 정보이고, 사라지는 필드는
 * 그 정보를 지운다(UX_AUDIT C-7). 이 제품의 존재 이유가 화면에서 조건부로 사라지면
 * 안 된다.
 *
 * 반대로 **미주입(undefined)이면 null**이다 — 통합자가 충돌 축을 아직 배선하지 않은
 * 소비자의 요약 문자열 계약을 바꾸지 않기 위함이다(기존 파사드 overlayText 검사).
 */
export function formatCollisionCount(count: number | undefined): string | null {
  if (count === undefined || !Number.isFinite(count) || count < 0) return null;
  return `충돌 ${Math.floor(count)}건`;
}

/**
 * 충돌 요약 발화 문구 — '충돌 12건 · 최근 arm × box_a'.
 *
 * 표시와 달리 **0건은 발화하지 않는다**(null): 아무 일도 일어나지 않았다는 사실을
 * 반복해 읽는 것은 정보가 아니라 소음이다. 화면은 상태를 유지하고 음성은 사건을 알린다.
 */
export function formatCollisionAnnouncement(
  count: number | undefined,
  lastCollisionPair: string | null | undefined,
): string | null {
  if (count === undefined || !Number.isFinite(count) || count <= 0) return null;
  const countText = formatCollisionCount(count);
  if (countText === null) return null;
  if (lastCollisionPair === null || lastCollisionPair === undefined || lastCollisionPair === '') {
    return countText;
  }
  return `${countText}${SEGMENT_SEP}최근 ${lastCollisionPair}`;
}

/** 재생 종료 1회 발화 — '시퀀스 완료 · 충돌 총 12건' (0건도 명시한다: 결과가 곧 정보다) */
export function formatRunSummary(count: number | undefined): string {
  const safe = count !== undefined && Number.isFinite(count) && count > 0 ? Math.floor(count) : 0;
  return `시퀀스 완료${SEGMENT_SEP}충돌 총 ${safe}건`;
}

/**
 * 시각 라인/aria 요약 텍스트(선행 ● 점 제외). 세그먼트: 상태 · [node N/M] ·
 * [activeNodeLabel] · simTime · scene · [선택 id] · [충돌 N건]. 진행이 없으면 노드/라벨
 * 세그먼트는 생략한다. 선택/충돌 세그먼트는 맨 뒤 — 기존 접두 세그먼트를 보는 소비자
 * (파사드 overlayText 검사)의 계약을 바꾸지 않는다.
 */
export function overlaySummary(s: RunOverlayState): string {
  const parts: string[] = [stateLabel(s.engineState)];
  const progress = formatNodeProgress(s.nodeIndex, s.nodeCount);
  if (progress !== null) {
    parts.push(progress);
    if (s.activeNodeLabel !== null && s.activeNodeLabel !== '') parts.push(s.activeNodeLabel);
  }
  parts.push(formatSimTime(s.simTimeSec));
  if (s.sceneName !== '') parts.push(s.sceneName);
  const selection = formatSelection(s.selectedEntityId);
  if (selection !== null) parts.push(selection);
  const collisions = formatCollisionCount(s.collisionCount);
  if (collisions !== null) parts.push(collisions);
  return parts.join(SEGMENT_SEP);
}

/**
 * 스크린리더 announce 시그니처 — simTime을 제외한 "의미 있는 전이"만 담는다(재생 상태·
 * 노드 진행·활성 노드·선택 변화). 이 값이 바뀔 때만 live 영역을 갱신해 rAF마다의 범람을 막는다.
 *
 * **충돌 건수는 의도적으로 제외한다** — 시그니처에 넣으면 충돌 1건마다 즉시 발화가 되어
 * 스로틀이 무력해진다. 충돌은 별도 Announcer(3초 게이트)가 요약으로 담당한다(파일 헤더).
 */
export function announceSignature(s: RunOverlayState): string {
  const progress = formatNodeProgress(s.nodeIndex, s.nodeCount);
  return (
    `${normalizeEngineState(s.engineState)}|${progress ?? ''}|${s.activeNodeLabel ?? ''}` +
    `|${s.selectedEntityId ?? ''}`
  );
}

/** announce 텍스트 (시그니처가 바뀐 순간 live 영역에 넣는 concise 문구 — simTime 없음) */
export function announceText(s: RunOverlayState): string {
  const parts: string[] = [stateLabel(s.engineState)];
  const progress = formatNodeProgress(s.nodeIndex, s.nodeCount);
  if (progress !== null) {
    parts.push(progress);
    if (s.activeNodeLabel !== null && s.activeNodeLabel !== '') parts.push(s.activeNodeLabel);
  }
  const selection = formatSelection(s.selectedEntityId);
  if (selection !== null) parts.push(selection);
  return parts.join(SEGMENT_SEP);
}

// ── 마운트 ──────────────────────────────────────────────────────────

/** 시각적으로 숨기되 접근성 트리에는 남기는 sr-only 스타일 (live 영역 전용) */
const SR_ONLY_STYLE: Partial<CSSStyleDeclaration> = {
  position: 'absolute',
  width: '1px',
  height: '1px',
  padding: '0',
  margin: '-1px',
  overflow: 'hidden',
  clip: 'rect(0 0 0 0)',
  whiteSpace: 'nowrap',
  border: '0',
};

export function mountRunOverlay(host: HTMLElement): RunOverlayHandle {
  ensureThemeStyles();

  // 루트: 뷰포트 슬롯 좌하단 absolute. 순수 표시 — 뷰포트 상호작용을 가로막지 않는다.
  const root = styled(document.createElement('div'), {
    position: 'absolute',
    left: `${OVERLAY_LEFT_PX}px`,
    bottom: `${OVERLAY_BOTTOM_PX}px`,
    zIndex: Z_INDEX.bar,
    pointerEvents: 'none',
    userSelect: 'none',
  });
  root.dataset.testid = 'run-overlay';

  // 시각 라인 (aria-hidden — 접근성 텍스트는 아래 live 영역이 담당, 이중 낭독 방지)
  const line = applyType(document.createElement('div'), TYPE.body);
  styled(line, {
    display: 'flex',
    alignItems: 'center',
    gap: SPACE.md,
    padding: `${SPACE.xxs} ${SPACE.lg}`,
    background: SURFACE.panel,
    border: `${BORDER_WIDTH.hair} solid ${BORDER.default}`,
    borderRadius: RADIUS.sm,
    color: COLOR.text,
    whiteSpace: 'nowrap',
  });
  line.setAttribute('aria-hidden', 'true');

  // 상태: 펄스 점(색) + 라벨 텍스트 병행
  const stateWrap = styled(document.createElement('span'), {
    display: 'inline-flex',
    alignItems: 'center',
  });
  const stateDot = document.createElement('span');
  stateDot.className = 'ui-dot ui-dot--idle';
  const stateLabelEl = styled(document.createElement('span'), { color: COLOR.muted });
  stateLabelEl.textContent = STATE_LABELS.idle;
  stateWrap.appendChild(stateDot);
  stateWrap.appendChild(stateLabelEl);

  const sep = (): HTMLElement => {
    const dot = styled(document.createElement('span'), { color: COLOR.muted });
    dot.textContent = '·';
    return dot;
  };

  // 노드 진행 'node N/M' (+ 활성 노드 라벨) — 시퀀스 없으면 숨김
  const nodeSep = sep();
  const nodeEl = applyType(document.createElement('span'), TYPE.monoReadout);
  nodeEl.style.color = COLOR.label;
  nodeEl.dataset.testid = 'run-overlay-node';

  const timeSep = sep();
  const timeEl = applyType(document.createElement('span'), TYPE.monoReadout);
  timeEl.style.color = COLOR.label;
  timeEl.textContent = formatSimTime(0);

  const sceneSep = sep();
  const sceneEl = styled(document.createElement('span'), {
    color: COLOR.textStrong,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    maxWidth: `${READOUT_MAX_WIDTH_PX}px`,
  });

  // 선택 리드아웃 '선택 arm_left' — 선택이 없으면 숨김 (RunOverlayState 주석)
  const selectionSep = sep();
  const selectionEl = applyType(document.createElement('span'), TYPE.monoReadout);
  styled(selectionEl, {
    // 선택은 액센트가 아니라 SELECT(스카이블루) 램프다 — 3D 뷰포트에서 "선택됨"과
    // "실행 중"이 같은 색이면 구분이 불가능하다 (UX_AUDIT C-14 액센트 3분할).
    color: SELECT.text,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    maxWidth: `${READOUT_MAX_WIDTH_PX}px`,
  });
  selectionEl.dataset.testid = 'run-overlay-selection';

  // 충돌 리드아웃 '충돌 12건' — 통합자가 축을 배선했으면 0건도 상설한다(C-7).
  // 색은 setState에서 건수에 따라 바꾼다(0건 muted / 1건 이상 충돌 램프).
  const collisionSep = sep();
  const collisionEl = applyType(document.createElement('span'), TYPE.monoReadout);
  collisionEl.style.color = COLLISION.text;
  collisionEl.dataset.testid = 'run-overlay-collisions';

  line.appendChild(stateWrap);
  line.appendChild(nodeSep);
  line.appendChild(nodeEl);
  line.appendChild(timeSep);
  line.appendChild(timeEl);
  line.appendChild(sceneSep);
  line.appendChild(sceneEl);
  line.appendChild(selectionSep);
  line.appendChild(selectionEl);
  line.appendChild(collisionSep);
  line.appendChild(collisionEl);

  // sr-only live 영역 — "의미 있는 전이"에서만 갱신 (aria-live polite, 범람 방지)
  const liveRegion = styled(document.createElement('span'), SR_ONLY_STYLE);
  liveRegion.setAttribute('role', 'status');
  liveRegion.setAttribute('aria-live', 'polite');
  liveRegion.dataset.testid = 'run-overlay-live';

  root.appendChild(line);
  root.appendChild(liveRegion);
  host.appendChild(root);

  // 충돌 전용 스로틀 Announcer — 전이 live 영역과 분리한다. 전이는 "드물고 즉시"가
  // 맞고 충돌은 "잦고 요약"이 맞아서, 한 영역에 섞으면 둘 중 하나가 반드시 손해를 본다.
  const collisionAnnouncer = createAnnouncer(root, 'polite', COLLISION_ANNOUNCE_MIN_INTERVAL_MS);
  root.lastElementChild?.setAttribute('data-testid', 'run-overlay-collision-live');

  let lastSignature: string | null = null;
  /** 마지막으로 발화 예약한 충돌 문구 — 같은 문구를 반복 예약하지 않는다 */
  let lastCollisionText: string | null = null;
  /** 직전 틱이 재생 중이었는가 (재생 종료 에지 검출) */
  let wasPlaying = false;
  /** 이번 종료 에피소드에서 완료 요약을 이미 발화했는가 */
  let runSummaryAnnounced = false;

  const setState = (s: RunOverlayState): void => {
    // 상태 점 + 라벨 (색 + 텍스트 병행)
    stateDot.className = stateDotClass(s.engineState);
    stateLabelEl.textContent = stateLabel(s.engineState);
    stateLabelEl.style.color = stateColor(s.engineState);

    // 노드 진행 (+ 활성 노드 라벨)
    const progress = formatNodeProgress(s.nodeIndex, s.nodeCount);
    if (progress !== null) {
      const label =
        s.activeNodeLabel !== null && s.activeNodeLabel !== ''
          ? `${progress}${SEGMENT_SEP}${s.activeNodeLabel}`
          : progress;
      nodeEl.textContent = label;
      nodeEl.style.display = '';
      nodeSep.style.display = '';
    } else {
      nodeEl.style.display = 'none';
      nodeSep.style.display = 'none';
    }

    timeEl.textContent = formatSimTime(s.simTimeSec);

    if (s.sceneName !== '') {
      sceneEl.textContent = s.sceneName;
      sceneEl.title = s.sceneName;
      sceneEl.style.display = '';
      sceneSep.style.display = '';
    } else {
      sceneEl.style.display = 'none';
      sceneSep.style.display = 'none';
    }

    const selection = formatSelection(s.selectedEntityId);
    if (selection !== null) {
      selectionEl.textContent = selection;
      selectionEl.title = selection;
      selectionEl.style.display = '';
      selectionSep.style.display = '';
    } else {
      selectionEl.style.display = 'none';
      selectionSep.style.display = 'none';
    }

    const collisions = formatCollisionCount(s.collisionCount);
    if (collisions !== null) {
      collisionEl.textContent = collisions;
      // 색은 보조 채널 — 숫자가 1차 채널이므로 색맹 사용자도 손실이 없다
      collisionEl.style.color =
        s.collisionCount !== undefined && s.collisionCount > 0 ? COLLISION.text : COLOR.muted;
      collisionEl.style.display = '';
      collisionSep.style.display = '';
    } else {
      collisionEl.style.display = 'none';
      collisionSep.style.display = 'none';
    }

    // 스크린리더: 시그니처(상태/노드/라벨/선택) 변화 시에만 갱신 (simTime rAF 범람 차단)
    const signature = announceSignature(s);
    if (signature !== lastSignature) {
      lastSignature = signature;
      liveRegion.textContent = announceText(s);
    }

    // 충돌 요약: 문구가 바뀌었을 때만 예약하고, 발화 자체는 3초 게이트가 합친다
    const collisionText = formatCollisionAnnouncement(s.collisionCount, s.lastCollisionPair);
    if (collisionText !== null && collisionText !== lastCollisionText) {
      lastCollisionText = collisionText;
      collisionAnnouncer.announce(collisionText);
    }

    // 재생 종료 1회 요약 — "무슨 일이 있었나"는 끝난 직후가 가장 필요한 순간이다.
    // 일시정지(playing→paused)는 종료가 아니다 — idle 전이만 에지로 본다.
    const normalized = normalizeEngineState(s.engineState);
    const playing = normalized === 'playing';
    if (playing) runSummaryAnnounced = false;
    const ended = s.sequenceDone === true || (wasPlaying && normalized === 'idle');
    if (ended && !runSummaryAnnounced) {
      runSummaryAnnounced = true;
      collisionAnnouncer.announceNow(formatRunSummary(s.collisionCount));
    }
    wasPlaying = playing;
  };

  return {
    setState,
    dispose: (): void => {
      collisionAnnouncer.dispose();
      root.remove();
    },
  };
}
