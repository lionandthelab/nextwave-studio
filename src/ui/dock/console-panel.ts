// ui/dock/console-panel.ts — 앱 콘솔 패널 + appLog pub/sub (UX_DESIGN §3.6 "Console")
//
// 역할: 스키마 검증 오류·player 경고·(추후) 플래너 메시지 등 사람이 읽어야 하는
// 앱 로그를 하단 독 Console 탭에 표시한다. appLog()는 ui 계층 전용 pub/sub이다 —
// core/planner는 이 모듈을 모른다(계층 의존 방향 ui → core, CLAUDE.md §3). core의
// 경고는 글루(main.ts)가 콜백(warn 주입 등)으로 받아 appLog로 중계한다.
//
// 패널이 마운트되기 전에 발생한 로그도 잃지 않도록 유계 버퍼에 보관하고,
// 패널 생성 시 버퍼를 재생(replay)한다.

// ── 상수 (매직넘버 금지 — CLAUDE.md §4) ─────────────────────────────

/** 보관/표시할 최대 로그 줄 수 — 초과 시 가장 오래된 줄부터 버린다 */
const MAX_LOG_LINES = 300;
/** 자동 스크롤 판정: 바닥에서 이 px 이내면 "바닥에 붙어 있음"으로 본다 */
const AUTOSCROLL_THRESHOLD_PX = 8;

const LEVEL_COLORS: Readonly<Record<AppLogLevel, string>> = {
  info: '#9aa0a8',
  warn: '#e2b93d',
  error: '#ff6b6b',
};

const LEVEL_TAGS: Readonly<Record<AppLogLevel, string>> = {
  info: 'INFO',
  warn: 'WARN',
  error: 'ERR ',
};

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
 * 앱 로그 발행 — 검증 오류·경고·안내를 Console 탭(+구독자)으로 보낸다.
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

// ── 패널 ────────────────────────────────────────────────────────────

export interface ConsolePanel {
  readonly el: HTMLElement;
  dispose(): void;
}

function styled<T extends HTMLElement>(el: T, style: Partial<CSSStyleDeclaration>): T {
  Object.assign(el.style, style);
  return el;
}

function formatClock(atMs: number): string {
  const d = new Date(atMs);
  const hh = String(d.getHours()).padStart(2, '0');
  const mm = String(d.getMinutes()).padStart(2, '0');
  const ss = String(d.getSeconds()).padStart(2, '0');
  return `${hh}:${mm}:${ss}`;
}

/** Console 탭 콘텐츠 생성. dock이 el을 탭 콘텐츠로 마운트한다. */
export function createConsolePanel(): ConsolePanel {
  const el = styled(document.createElement('div'), {
    height: '100%',
    overflowY: 'auto',
    padding: '4px 8px',
    boxSizing: 'border-box',
    fontSize: '11px',
    lineHeight: '1.6',
  });
  el.dataset.testid = 'console-panel';

  const append = (entry: AppLogEntry): void => {
    const stick =
      el.scrollTop + el.clientHeight >= el.scrollHeight - AUTOSCROLL_THRESHOLD_PX;
    const line = styled(document.createElement('div'), {
      whiteSpace: 'pre-wrap',
      color: LEVEL_COLORS[entry.level],
    });
    line.textContent = `${formatClock(entry.atMs)} [${LEVEL_TAGS[entry.level].trim()}] ${entry.msg}`;
    el.appendChild(line);
    while (el.childElementCount > MAX_LOG_LINES) el.firstElementChild?.remove();
    if (stick) el.scrollTop = el.scrollHeight;
  };

  const unsubscribe = subscribeAppLog(append, true);
  return {
    el,
    dispose: (): void => {
      unsubscribe();
      el.remove();
    },
  };
}
