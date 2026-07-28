// ui/dock/console-panel.ts — 앱 콘솔 패널 + appLog pub/sub (UX_DESIGN §3.6 "콘솔")
//
// 역할: 스키마 검증 오류·player 경고·(추후) 플래너 메시지 등 사람이 읽어야 하는
// 앱 로그를 하단 독 콘솔 탭에 표시한다. appLog()는 ui 계층 전용 pub/sub이다 —
// core/planner는 이 모듈을 모른다(계층 의존 방향 ui → core, CLAUDE.md §3). core의
// 경고는 글루(main.ts)가 콜백(warn 주입 등)으로 받아 appLog로 중계한다.
//
// 패널이 마운트되기 전에 발생한 로그도 잃지 않도록 유계 버퍼에 보관하고,
// 패널 생성 시 버퍼를 재생(replay)한다.
//
// ── Phase 11에서 닫는 감사 항목 ─────────────────────────────────────
// C-18 구현은 `MAX_LOG_LINES = 300`이 전부였다 — 레벨 필터도, 지우기도, 복사도 없어서
//      오류 한 줄을 찾으려면 300줄을 눈으로 훑고, 찾아도 밖으로 가져갈 방법이 없었다.
//      → 레벨 토글 3종 + 복사 + 지우기를 헤더에 둔다. 토글은 **표시**만 바꾸고 줄 자체는
//        남는다(껐다 켜면 과거 줄이 그대로 돌아온다).
//      로그 영역에 aria-live는 걸지 않는다 — 검증 오류가 수십 줄로 쏟아지면 polite 큐가
//      포화되어 사용자가 다른 조작을 해도 몇 분간 과거 줄만 읽는다.

import { icon, makeIconButton } from '../icons';
import {
  BORDER,
  BORDER_WIDTH,
  COLOR,
  ICON,
  SPACE,
  SURFACE,
  TYPE,
  applyType,
  ensureThemeStyles,
  styled,
} from '../theme';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4, 시각 토큰은 ui/theme.ts) ────

/** 보관/표시할 최대 로그 줄 수 — 초과 시 가장 오래된 줄부터 버린다 */
const MAX_LOG_LINES = 300;
/** 자동 스크롤 판정: 바닥에서 이 px 이내면 "바닥에 붙어 있음"으로 본다 */
const AUTOSCROLL_THRESHOLD_PX = 8;
/** "복사됨" 피드백 유지 시간(ms) */
const COPY_FEEDBACK_MS = 1400;

/** 레벨별 색 — 라벨 태그([INFO]/[WARN]/[ERR])가 병행되므로 색은 보조 채널 (UX §9) */
const LEVEL_COLORS: Readonly<Record<AppLogLevel, string>> = {
  info: COLOR.label,
  warn: COLOR.warnText,
  error: COLOR.errorText,
};

const LEVEL_TAGS: Readonly<Record<AppLogLevel, string>> = {
  info: 'INFO',
  warn: 'WARN',
  error: 'ERR ',
};

/** 헤더 토글 순서 (표시 순서 고정) */
const LEVELS: readonly AppLogLevel[] = ['info', 'warn', 'error'];

// ── appLog pub/sub (ui 계층 전용) ───────────────────────────────────

export type AppLogLevel = 'info' | 'warn' | 'error';

export interface AppLogEntry {
  readonly level: AppLogLevel;
  readonly msg: string;
  /** 벽시계 시각 (표시용 — 시뮬 시간이 아니다) */
  readonly atMs: number;
}

type AppLogListener = (entry: AppLogEntry) => void;

const logBuffer: AppLogEntry[] = [];
const logListeners = new Set<AppLogListener>();

/**
 * 앱 로그 발행 — 검증 오류·경고·안내를 콘솔 탭(+구독자)으로 보낸다.
 * ui 계층 어디서든 호출 가능. 패널 마운트 전 로그도 버퍼에 남아 재생된다.
 */
export function appLog(level: AppLogLevel, msg: string): void {
  const entry: AppLogEntry = { level, msg, atMs: Date.now() };
  logBuffer.push(entry);
  if (logBuffer.length > MAX_LOG_LINES) logBuffer.shift();
  for (const listener of [...logListeners]) listener(entry);
}

/** 로그 구독 (반환값은 해제 함수). replayBuffer=true면 기존 버퍼를 즉시 재생한다. */
export function subscribeAppLog(fn: AppLogListener, replayBuffer = false): () => void {
  if (replayBuffer) for (const entry of logBuffer) fn(entry);
  logListeners.add(fn);
  return () => {
    logListeners.delete(fn);
  };
}

/**
 * 재생 버퍼를 비운다 — "지우기"가 화면만 비우면, 씬을 다시 로드해 패널이 재생성될 때
 * 지웠던 줄이 되살아난다(패널은 마운트 시 버퍼를 재생한다). 둘을 함께 비운다.
 */
export function clearAppLog(): void {
  logBuffer.length = 0;
}

// ── 패널 ────────────────────────────────────────────────────────────

export interface ConsolePanel {
  readonly el: HTMLElement;
  dispose(): void;
}

function formatClock(atMs: number): string {
  const d = new Date(atMs);
  const hh = String(d.getHours()).padStart(2, '0');
  const mm = String(d.getMinutes()).padStart(2, '0');
  const ss = String(d.getSeconds()).padStart(2, '0');
  return `${hh}:${mm}:${ss}`;
}

/** 콘솔 탭 콘텐츠 생성. dock이 el을 탭 콘텐츠로 마운트한다. */
export function createConsolePanel(): ConsolePanel {
  ensureThemeStyles();
  const el = styled(document.createElement('div'), {
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
    boxSizing: 'border-box',
  });
  el.dataset.testid = 'console-panel';

  // ── 로그 영역 (먼저 만든다 — 헤더 핸들러가 참조한다) ───────────────
  const scroller = styled(document.createElement('div'), {
    flex: '1 1 auto',
    minHeight: '0',
    overflowY: 'auto',
    padding: `${SPACE.xs} ${SPACE.md}`,
    boxSizing: 'border-box',
    background: SURFACE.sunken,
  });
  applyType(scroller, TYPE.monoBody);
  scroller.classList.add('ui-scroll');

  const emptyState = styled(document.createElement('div'), {
    padding: `${SPACE.lg} 0`,
    color: COLOR.muted,
    textAlign: 'center',
  });
  applyType(emptyState, TYPE.body);
  emptyState.textContent = '로그가 없습니다';
  emptyState.dataset.testid = 'console-empty';

  /** 로그 줄 컨테이너 — 빈 상태와 분리해 두면 줄 연산에 예외 처리가 필요 없다 */
  const linesEl = document.createElement('div');
  linesEl.dataset.testid = 'console-lines';

  scroller.appendChild(emptyState);
  scroller.appendChild(linesEl);

  const enabledLevels = new Set<AppLogLevel>(LEVELS);

  const paintEmptyState = (): void => {
    emptyState.style.display = linesEl.childElementCount === 0 ? '' : 'none';
  };

  const paintLevels = (): void => {
    for (const line of linesEl.children) {
      if (!(line instanceof HTMLElement)) continue;
      const level = line.dataset.level;
      line.style.display =
        level !== undefined && enabledLevels.has(level as AppLogLevel) ? '' : 'none';
    }
  };

  // ── 헤더: 레벨 토글 + 복사 + 지우기 ──────────────────────────────
  const header = styled(document.createElement('div'), {
    display: 'flex',
    alignItems: 'center',
    gap: SPACE.lg,
    padding: `${SPACE.xs} ${SPACE.md}`,
    borderBottom: `${BORDER_WIDTH.hair} solid ${BORDER.default}`,
    flexShrink: '0',
  });

  const levelGroup = styled(document.createElement('div'), {
    display: 'flex',
    alignItems: 'center',
    gap: SPACE.md,
  });
  levelGroup.setAttribute('role', 'group');
  levelGroup.setAttribute('aria-label', '로그 레벨 표시');

  for (const level of LEVELS) {
    const label = document.createElement('label');
    label.className = 'ui-check-label';
    const input = document.createElement('input');
    input.type = 'checkbox';
    input.className = 'ui-check';
    input.checked = true;
    input.dataset.testid = `console-level-${level}`;
    input.setAttribute('aria-label', `${LEVEL_TAGS[level].trim()} 레벨 표시`);
    input.addEventListener('change', () => {
      if (input.checked) enabledLevels.add(level);
      else enabledLevels.delete(level);
      paintLevels();
    });
    const text = styled(document.createElement('span'), { color: LEVEL_COLORS[level] });
    applyType(text, TYPE.monoMicro);
    // 레벨 태그는 로그 본문과 같은 도메인 어휘(영문) — 한국어 TTS 철자 나열 방지
    text.setAttribute('lang', 'en');
    text.textContent = LEVEL_TAGS[level].trim();
    label.appendChild(input);
    label.appendChild(text);
    levelGroup.appendChild(label);
  }
  header.appendChild(levelGroup);
  header.appendChild(styled(document.createElement('span'), { flex: '1 1 auto' }));

  const copyButton = makeIconButton(
    'copy',
    '복사',
    '보이는 로그를 클립보드로 복사',
    'console-copy',
  );
  const clearButton = makeIconButton('trash', '지우기', '콘솔 로그 지우기', 'console-clear');
  header.appendChild(copyButton);
  header.appendChild(clearButton);

  el.appendChild(header);
  el.appendChild(scroller);

  // ── 복사 피드백 (아이콘/라벨 일시 교체 — 토스트를 띄우지 않는다) ────
  let copyFeedbackTimer: number | null = null;
  const paintCopyButton = (copied: boolean): void => {
    copyButton.replaceChildren();
    copyButton.appendChild(icon(copied ? 'check' : 'copy', ICON.md));
    const span = document.createElement('span');
    span.className = 'ui-btn__label';
    span.textContent = copied ? '복사됨' : '복사';
    copyButton.appendChild(span);
  };

  const visibleLines = (): string[] => {
    const out: string[] = [];
    for (const line of linesEl.children) {
      if (!(line instanceof HTMLElement)) continue;
      if (line.style.display === 'none') continue;
      out.push(line.textContent ?? '');
    }
    return out;
  };

  copyButton.addEventListener('click', () => {
    const lines = visibleLines();
    if (lines.length === 0) return;
    const clipboard: Clipboard | undefined = navigator.clipboard;
    if (clipboard === undefined) {
      appLog('warn', '이 환경에서는 클립보드를 쓸 수 없습니다 — 로그를 직접 선택해 복사하세요');
      return;
    }
    void clipboard.writeText(lines.join('\n')).then(
      () => {
        if (copyFeedbackTimer !== null) window.clearTimeout(copyFeedbackTimer);
        paintCopyButton(true);
        copyFeedbackTimer = window.setTimeout(() => {
          copyFeedbackTimer = null;
          paintCopyButton(false);
        }, COPY_FEEDBACK_MS);
      },
      () => {
        appLog('warn', '클립보드 복사가 거부되었습니다 — 로그를 직접 선택해 복사하세요');
      },
    );
  });

  clearButton.addEventListener('click', () => {
    linesEl.replaceChildren();
    clearAppLog(); // 재생 버퍼도 함께 — 씬 리로드로 지운 줄이 되살아나지 않게
    paintEmptyState();
  });

  // ── 로그 수신 ─────────────────────────────────────────────────────
  const append = (entry: AppLogEntry): void => {
    const stick =
      scroller.scrollTop + scroller.clientHeight >= scroller.scrollHeight - AUTOSCROLL_THRESHOLD_PX;
    const line = styled(document.createElement('div'), {
      whiteSpace: 'pre-wrap',
      color: LEVEL_COLORS[entry.level],
    });
    applyType(line, TYPE.monoBody);
    line.dataset.level = entry.level;
    line.textContent = `${formatClock(entry.atMs)} [${LEVEL_TAGS[entry.level].trim()}] ${entry.msg}`;
    if (!enabledLevels.has(entry.level)) line.style.display = 'none';
    linesEl.appendChild(line);
    while (linesEl.childElementCount > MAX_LOG_LINES) linesEl.firstElementChild?.remove();
    paintEmptyState();
    if (stick) scroller.scrollTop = scroller.scrollHeight;
  };

  paintEmptyState();
  const unsubscribe = subscribeAppLog(append, true);
  return {
    el,
    dispose: (): void => {
      if (copyFeedbackTimer !== null) window.clearTimeout(copyFeedbackTimer);
      unsubscribe();
      el.remove();
    },
  };
}
