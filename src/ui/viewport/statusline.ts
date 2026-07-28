// ui/viewport/statusline.ts — 뷰포트 실행 오버레이 (UX_DESIGN §3.3 "실행 오버레이")
//
// 뷰포트 좌하단에 [씬 이름 · ● 재생 상태 · simTime · 충돌 N · (시퀀스 있으면) step n/N]을
// 표시한다. 데이터는 글루(main.ts)가 engine.onTick에서 update()로 공급한다
// (이 모듈은 core를 import하지 않는다 — 계층 규칙, CLAUDE.md §3).
//
// ── 충돌 카운터 (UX_AUDIT C-7) ───────────────────────────────────────
// 이 제품의 존재 이유는 충돌 감지인데, 7/7 시퀀스가 waitForCollision을 통과해 완주해도
// 화면 어디에도 "충돌이 있었다"는 표시가 없었다. 충돌 카운터는 **0건일 때도 상시 표시**
// 한다 — 0은 "감지가 돌고 있는데 아직 없다"는 정보이고, 사라지는 필드는 그 정보를 지운다.
// 클릭하면 충돌 로그 탭으로 데려간다(onFocusCollisionLog).
//
// 배치 계약 (Phase 7 워크스페이스): host는 "positioned"(position:relative 등) 뷰포트
// 슬롯이어야 한다 — 오버레이는 host 기준 absolute로 좌하단/중앙에 앉는다.
//
// 순수 표시 전용: 라인 자체는 pointer-events:none으로 뷰포트 orbit/선택을 가로막지
// 않는다. **예외는 충돌 카운터 버튼 하나**로, 이 요소만 pointer-events:auto다.
//
// 빈 씬 안내 (UX_DESIGN §7 "빈 씬"): 엔티티가 0개인 씬이면 뷰포트 중앙에 안내 문구를
// 함께 띄운다 — 마운트 시 emptyScene 플래그로 결정되고, 씬 편집(Scene Builder)으로
// 엔티티가 생기면 통합자가 setEmptyHintVisible(false)로 숨긴다.

import {
  BORDER,
  BORDER_WIDTH,
  COLLISION,
  COLOR,
  ICON,
  RADIUS,
  SPACE,
  SURFACE,
  TYPE,
  Z_INDEX,
  applyType,
  ensureThemeStyles,
  styled,
} from '../theme';
import { icon } from '../icons';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4) ─────────────────────────────

/** 뷰포트 슬롯 좌하단 오프셋 (독은 워크스페이스 그리드의 별도 행 — 겹치지 않는다) */
const STATUS_BOTTOM_PX = 12;
const STATUS_LEFT_PX = 12;
/** simTime 표시 소수 자릿수 (playback.ts와 동일 규약) */
const SIM_TIME_DECIMALS = 2;
/** 씬 이름 리드아웃 최대 폭 */
const SCENE_NAME_MAX_WIDTH_PX = 220;

/**
 * 빈 씬 중앙 안내 (UX_DESIGN §7 · §1 "직접 조작").
 *
 * 이전 문구는 "상단의 씬 프리셋을 선택하거나 씬 JSON을 업로드하세요"였다 — 규정과
 * 정반대로 JSON 업로드를 권해, 좌측 라이브러리 카드 8종이 첫 사용자에게 죽은 UI가 됐다
 * (UX_AUDIT C-18 "인앱 도움말 0"). 첫 안내는 반드시 직접 조작 경로를 가리켜야 한다.
 */
export const EMPTY_SCENE_HINT_KO =
  '라이브러리에서 로봇/물체를 드래그하거나 3D 파일을 놓으세요';

const STATE_LABELS: Readonly<Record<ViewportStatusInfo['engineState'], string>> = {
  playing: 'Running',
  paused: 'Paused',
  idle: 'Idle',
};

const STATE_TEXT_COLORS: Readonly<Record<ViewportStatusInfo['engineState'], string>> = {
  playing: COLOR.accentText,
  paused: COLOR.warn,
  idle: COLOR.muted,
};

// ── 공개 타입 ───────────────────────────────────────────────────────

export interface ViewportStatusInfo {
  engineState: 'idle' | 'playing' | 'paused';
  simTimeSec: number;
  /** 시퀀스가 있는 씬이면 진행 배지 표시 (stepCount 0이면 미표시) */
  sequence?: {
    stepIndex: number;
    stepCount: number;
  };
  /** 누적 충돌 건수 — 미주입이면 setCollisionCount로 넣은 마지막 값이 유지된다 */
  collisionCount?: number;
}

export interface ViewportStatusOptions {
  /** 씬 이름 (spec.name) */
  sceneName: string;
  /** 엔티티 0개 씬 — 중앙 빈 씬 안내 표시 (UX_DESIGN §7) */
  emptyScene?: boolean;
  /**
   * 충돌 카운터 클릭 → 충돌 로그 탭 활성화 (UX_AUDIT C-7 제안).
   * 미주입이면 카운터는 비상호작용 텍스트로 렌더된다(포커스 순서를 오염시키지 않는다).
   */
  onFocusCollisionLog?(): void;
}

export interface ViewportStatusHandle {
  readonly el: HTMLElement;
  /** 상태 갱신 (engine.onTick에서 rAF당 1회) */
  update(info: ViewportStatusInfo): void;
  /** 누적 충돌 건수 갱신 (CollisionMonitor 구독에서 호출 — 0도 표시된다) */
  setCollisionCount(n: number): void;
  /** 빈 씬 중앙 안내 표시/숨김 — Scene Builder로 첫 엔티티가 생기면 숨긴다 (통합자 호출) */
  setEmptyHintVisible(visible: boolean): void;
  dispose(): void;
}

// ── 순수 헬퍼 (DOM 비의존) ──────────────────────────────────────────

/** 상태줄 충돌 카운터 텍스트 — 0건도 표시한다(감지가 돌고 있음을 알린다) */
export function formatCollisionBadge(count: number): string {
  const safe = Number.isFinite(count) && count > 0 ? Math.floor(count) : 0;
  return `충돌 ${safe}`;
}

// ── 마운트 ──────────────────────────────────────────────────────────

export function mountViewportStatus(
  host: HTMLElement,
  opts: ViewportStatusOptions,
): ViewportStatusHandle {
  ensureThemeStyles();

  const line = applyType(document.createElement('div'), TYPE.body);
  styled(line, {
    position: 'absolute',
    left: `${STATUS_LEFT_PX}px`,
    bottom: `${STATUS_BOTTOM_PX}px`,
    zIndex: Z_INDEX.bar,
    display: 'flex',
    alignItems: 'center',
    gap: SPACE.md,
    padding: `${SPACE.xxs} ${SPACE.lg}`,
    background: SURFACE.panel,
    border: `${BORDER_WIDTH.hair} solid ${BORDER.default}`,
    borderRadius: RADIUS.sm,
    color: COLOR.text,
    whiteSpace: 'nowrap',
    pointerEvents: 'none', // 순수 표시 — 뷰포트 상호작용을 가로막지 않는다
    userSelect: 'none',
  });
  line.dataset.testid = 'viewport-status';

  const sceneNameEl = styled(document.createElement('span'), {
    color: COLOR.textStrong,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    maxWidth: `${SCENE_NAME_MAX_WIDTH_PX}px`,
  });
  sceneNameEl.textContent = opts.sceneName;
  sceneNameEl.title = opts.sceneName;

  const sep = (): HTMLElement => {
    const dot = styled(document.createElement('span'), { color: COLOR.muted });
    dot.textContent = '·';
    return dot;
  };

  // 상태: 펄스 점(재생 중 액센트) + 텍스트 라벨 병행 (색만으로 전달 금지 — UX §9)
  const stateWrap = styled(document.createElement('span'), {
    display: 'inline-flex',
    alignItems: 'center',
  });
  const stateDot = document.createElement('span');
  stateDot.className = 'ui-dot ui-dot--idle';
  stateDot.setAttribute('aria-hidden', 'true');
  const stateLabel = styled(document.createElement('span'), { color: COLOR.muted });
  stateLabel.textContent = STATE_LABELS.idle;
  stateWrap.appendChild(stateDot);
  stateWrap.appendChild(stateLabel);

  const timeEl = applyType(document.createElement('span'), TYPE.monoReadout);
  timeEl.style.color = COLOR.label;
  timeEl.textContent = 'simTime 0.00s';

  // ── 충돌 카운터 (C-7) ────────────────────────────────────────────
  // 콜백이 있으면 버튼(클릭 → 충돌 로그), 없으면 순수 텍스트. 버튼일 때만 라인의
  // pointer-events:none을 국소적으로 되돌린다 — 뷰포트 orbit을 가로막는 면적을
  // 카운터 하나로 제한한다.
  const hasFocusCallback = typeof opts.onFocusCollisionLog === 'function';
  const collisionEl: HTMLElement = hasFocusCallback
    ? document.createElement('button')
    : document.createElement('span');
  applyType(collisionEl, TYPE.monoReadout);
  collisionEl.dataset.testid = 'viewport-status-collisions';
  collisionEl.appendChild(icon('impact', ICON.sm));
  const collisionText = document.createElement('span');
  collisionText.textContent = formatCollisionBadge(0);
  collisionEl.appendChild(collisionText);

  if (collisionEl instanceof HTMLButtonElement) {
    collisionEl.type = 'button';
    collisionEl.className = 'ui-btn ui-btn--ghost rsw-hit-y';
    collisionEl.title = '충돌 로그 열기';
    collisionEl.setAttribute('aria-label', '충돌 로그 열기');
    styled(collisionEl, {
      pointerEvents: 'auto',
      padding: `0 ${SPACE.sm}`,
      gap: SPACE.xs,
    });
    collisionEl.addEventListener('click', (e) => {
      e.stopPropagation();
      opts.onFocusCollisionLog?.();
    });
  } else {
    styled(collisionEl, {
      display: 'inline-flex',
      alignItems: 'center',
      gap: SPACE.xs,
      color: COLOR.muted,
    });
  }

  const stepBadge = applyType(document.createElement('span'), TYPE.monoReadout);
  styled(stepBadge, { color: COLOR.label, display: 'none' });
  stepBadge.dataset.testid = 'viewport-status-step';

  line.appendChild(sceneNameEl);
  line.appendChild(sep());
  line.appendChild(stateWrap);
  line.appendChild(sep());
  line.appendChild(timeEl);
  line.appendChild(sep());
  line.appendChild(collisionEl);
  line.appendChild(stepBadge);
  host.appendChild(line);

  // 빈 씬 중앙 안내 (UX_DESIGN §7) — 씬 수명 동안 고정, 상호작용 없음
  let emptyHint: HTMLElement | null = null;
  if (opts.emptyScene === true) {
    emptyHint = applyType(document.createElement('div'), TYPE.subhead);
    styled(emptyHint, {
      position: 'absolute',
      left: '50%',
      top: '50%',
      transform: 'translate(-50%, -50%)',
      zIndex: Z_INDEX.bar,
      display: 'flex',
      alignItems: 'center',
      gap: SPACE.md,
      padding: `${SPACE.lg} ${SPACE.xl}`,
      background: SURFACE.panel,
      border: `${BORDER_WIDTH.hair} dashed ${BORDER.strong}`,
      borderRadius: RADIUS.md,
      color: COLOR.muted,
      textAlign: 'center',
      pointerEvents: 'none',
      userSelect: 'none',
    });
    emptyHint.dataset.testid = 'viewport-empty-hint';
    emptyHint.appendChild(icon('library', ICON.lg));
    const hintText = document.createElement('span');
    hintText.textContent = EMPTY_SCENE_HINT_KO;
    emptyHint.appendChild(hintText);
    host.appendChild(emptyHint);
  }

  const setCollisionCount = (n: number): void => {
    const safe = Number.isFinite(n) && n > 0 ? Math.floor(n) : 0;
    collisionText.textContent = formatCollisionBadge(safe);
    // 0건은 muted(감지 중), 1건 이상은 충돌 램프 — 색은 보조 채널이고 숫자가 1차 채널이다
    collisionEl.style.color = safe > 0 ? COLLISION.text : COLOR.muted;
  };

  const update = (info: ViewportStatusInfo): void => {
    stateDot.className = `ui-dot ui-dot--${info.engineState}`;
    stateLabel.textContent = STATE_LABELS[info.engineState];
    stateLabel.style.color = STATE_TEXT_COLORS[info.engineState];
    timeEl.textContent = `simTime ${info.simTimeSec.toFixed(SIM_TIME_DECIMALS)}s`;
    if (info.collisionCount !== undefined) setCollisionCount(info.collisionCount);
    if (info.sequence && info.sequence.stepCount > 0) {
      const shown = Math.min(info.sequence.stepIndex + 1, info.sequence.stepCount);
      stepBadge.textContent = `· step ${shown}/${info.sequence.stepCount}`;
      stepBadge.style.display = '';
    } else {
      stepBadge.style.display = 'none';
    }
  };

  setCollisionCount(0);

  return {
    el: line,
    update,
    setCollisionCount,
    setEmptyHintVisible: (visible): void => {
      if (emptyHint) emptyHint.style.display = visible ? 'flex' : 'none';
    },
    dispose: (): void => {
      line.remove();
      emptyHint?.remove();
    },
  };
}
