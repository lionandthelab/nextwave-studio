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

import { COLOR, FONT, RADIUS, SPACE, Z_INDEX, ensureThemeStyles, styled } from '../theme';

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
 * 시각 라인/aria 요약 텍스트(선행 ● 점 제외). 세그먼트: 상태 · [node N/M] ·
 * [activeNodeLabel] · simTime · scene · [선택 id]. 진행이 없으면 노드/라벨 세그먼트는
 * 생략한다. 선택 세그먼트는 맨 뒤 — 기존 접두 세그먼트를 보는 소비자(파사드 overlayText
 * 검사)의 계약을 바꾸지 않는다.
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
  return parts.join(SEGMENT_SEP);
}

/**
 * 스크린리더 announce 시그니처 — simTime을 제외한 "의미 있는 전이"만 담는다(재생 상태·
 * 노드 진행·활성 노드·선택 변화). 이 값이 바뀔 때만 live 영역을 갱신해 rAF마다의 범람을 막는다.
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
  const line = styled(document.createElement('div'), {
    display: 'flex',
    alignItems: 'center',
    gap: SPACE.md,
    padding: `3px ${SPACE.lg}`,
    background: COLOR.bgPanel,
    border: `1px solid ${COLOR.border}`,
    borderRadius: RADIUS.sm,
    color: COLOR.text,
    fontFamily: FONT.ui,
    fontSize: '12px',
    lineHeight: '1.6',
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
  const nodeEl = styled(document.createElement('span'), {
    fontFamily: FONT.mono,
    color: COLOR.label,
  });
  nodeEl.dataset.testid = 'run-overlay-node';

  const timeSep = sep();
  const timeEl = styled(document.createElement('span'), {
    fontFamily: FONT.mono,
    color: COLOR.label,
  });
  timeEl.textContent = formatSimTime(0);

  const sceneSep = sep();
  const sceneEl = styled(document.createElement('span'), {
    color: COLOR.textStrong,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    maxWidth: '220px',
  });

  // 선택 리드아웃 '선택 arm_left' — 선택이 없으면 숨김 (RunOverlayState 주석)
  const selectionSep = sep();
  const selectionEl = styled(document.createElement('span'), {
    fontFamily: FONT.mono,
    color: COLOR.accentText,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    maxWidth: '220px',
  });
  selectionEl.dataset.testid = 'run-overlay-selection';

  line.appendChild(stateWrap);
  line.appendChild(nodeSep);
  line.appendChild(nodeEl);
  line.appendChild(timeSep);
  line.appendChild(timeEl);
  line.appendChild(sceneSep);
  line.appendChild(sceneEl);
  line.appendChild(selectionSep);
  line.appendChild(selectionEl);

  // sr-only live 영역 — "의미 있는 전이"에서만 갱신 (aria-live polite, 범람 방지)
  const liveRegion = styled(document.createElement('span'), SR_ONLY_STYLE);
  liveRegion.setAttribute('role', 'status');
  liveRegion.setAttribute('aria-live', 'polite');
  liveRegion.dataset.testid = 'run-overlay-live';

  root.appendChild(line);
  root.appendChild(liveRegion);
  host.appendChild(root);

  let lastSignature: string | null = null;

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

    // 스크린리더: 시그니처(상태/노드/라벨/선택) 변화 시에만 갱신 (simTime rAF 범람 차단)
    const signature = announceSignature(s);
    if (signature !== lastSignature) {
      lastSignature = signature;
      liveRegion.textContent = announceText(s);
    }
  };

  return {
    setState,
    dispose: (): void => {
      root.remove();
    },
  };
}
