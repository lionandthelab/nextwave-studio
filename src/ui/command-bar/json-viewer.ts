// ui/command-bar/json-viewer.ts — '{} JSON' 토글 + ControlSequence 편집기 · 버전 이력
// (UX_DESIGN §3.1 "{} JSON")
//
// 커맨드바 우측의 토글 버튼과, 화면 우측에서 슬라이드로 나타나는 패널. 탭 2개다:
//   [JSON]  — 시퀀스 원본을 **직접 편집**하고 [적용]으로 커밋 (읽기 + 복사도 그대로)
//   [버전]  — 커밋 이력을 시각·라벨과 함께 나열하고 임의 버전으로 되돌린다
//
// ── 편집 계약 (불변식 §2.8 / §2.9) ──────────────────────────────────
// 이 모듈은 **텍스트만 다룬다.** 파싱·스키마 검증·그래프 반영은 전부 통합자가
// deps.applyJson(text)에서 수행하고, 결과를 { ok } | { errors } 로 돌려준다 —
// 검증을 통과하지 못한 JSON은 그래프/실행 어디에도 닿지 않는다. 실패 시 텍스트는
// 사용자가 쓴 그대로 남는다(고쳐 쓸 수 있어야 한다 — 지워버리면 다시 타이핑이다).
//
// ── 편집 중 외부 변경 (덮어쓰기 금지) ───────────────────────────────
// 캔버스/인스펙터 편집이 커밋되면 통합자가 refresh()를 부른다. 이때 사용자가 이미
// 텍스트를 고치고 있으면(dirty) **덮어쓰지 않고** 상단에 "밖에서 시퀀스가 바뀌었습니다
// — [현재 값 불러오기]" 배너를 띄운다. 30초 타이핑을 말없이 날리는 것이 이 패널에서
// 가능한 가장 나쁜 일이다.
//
// 계층 규칙 (CLAUDE.md §3): schema 타입(POJO)과 ui/theme·icons만 안다. core 비의존.
// 전역 keydown은 열려 있을 때의 Escape 하나뿐이며, Ctrl+Enter(적용)는 패널 내부에만
// 건다 — 전역 단축키의 단일 소유자는 ui/shortcuts.ts 라우터다 (§2.10).

import { rovingTabindex } from '../a11y';
import { makeIconButton } from '../icons';
import {
  BORDER,
  COLLISION,
  COLOR,
  LAYOUT,
  SHADOW,
  SPACE,
  STATUS,
  SURFACE,
  TYPE,
  Z_INDEX,
  applyType,
  ensureThemeStyles,
  makeButton,
  styled,
} from '../theme';
import { COMMAND_BAR_PRIORITY, setCommandBarPriority } from './scene-controls';
import type { ControlSequence } from '../../schema';
import type { SequenceVersion } from '../sequence-versions';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4, 시각 토큰은 ui/theme.ts) ────

/** 패널 폭(px) — 편집 가능해지며 넓혔다(80열 JSON이 접히지 않게) */
const PANEL_WIDTH_PX = 460;
/** 패널 상단 오프셋(px) — 커맨드바 바로 아래 */
const PANEL_TOP_PX = LAYOUT.barHeightPx;
/** 패널 하단 오프셋(px) — 독(본문 180 + 탭 바)과 겹치지 않는 여유 */
const PANEL_BOTTOM_PX = 216;
/** 슬라이드 트랜지션 */
const PANEL_TRANSITION = 'transform 0.18s ease';
/** 복사 버튼 피드백 표시 시간(ms) */
const COPY_FLASH_MS = 1500;
/** JSON pretty-print 들여쓰기 폭 */
const JSON_INDENT = 2;
/** 되돌리기 확인 바가 떠 있는 동안의 대상 버전 없음 표시 */
const NO_PENDING_RESTORE = -1;
/** 확인 바 testid 접두 — 중복 생성 방지 조회에 쓴다 */
const RESTORE_CONFIRM_PREFIX = 'json-restore-confirm-';

/** 마운트 인스턴스 일련번호 — DOM id 충돌 방지 (aria-controls/labelledby 짝) */
let mountSerial = 0;

// ── 공개 타입 ───────────────────────────────────────────────────────

/** 적용 결과 — 실패는 **사람이 읽을 한국어 사유**를 반드시 동반한다 (§2.8 피드백) */
export type ApplyJsonResult = { readonly ok: true } | { readonly ok: false; readonly errors: readonly string[] };

export interface JsonViewerDeps {
  /** 현재 씬의 검증된 ControlSequence (없으면 null) */
  getSequence(): ControlSequence | null;
  /**
   * 텍스트 적용 — 파싱 + 스키마/씬 참조 검증 + 그래프 커밋을 통합자가 수행한다.
   * 미주입이면 패널은 읽기 전용으로 동작한다(편집 UI가 렌더되지 않는다).
   */
  applyJson?(text: string): ApplyJsonResult;
  /** 버전 이력 (최신 우선). 미주입이면 [버전] 탭이 렌더되지 않는다. */
  listVersions?(): readonly SequenceVersion[];
  /** 특정 버전으로 되돌리기 — 통합자가 새 버전으로 append 커밋한다 */
  restoreVersion?(version: number): ApplyJsonResult;
  /** 상대 시각 문구 ('5분 전') — 통합자가 ui/document.ts describeAge를 넘긴다 */
  describeAge?(atIso: string): string;
}

export interface JsonViewerHandle {
  /** 토글 버튼 요소 (커맨드바 우측 슬롯에 마운트됨) */
  readonly el: HTMLElement;
  /**
   * 시퀀스가 바뀌었을 때(씬 전환·그래프 편집) 호출 — 패널 내용을 현재 진실로 다시 그린다.
   * **사용자가 편집 중(dirty)이면 텍스트를 덮어쓰지 않고** 외부 변경 배너만 띄운다.
   */
  refresh(): void;
  /** 편집 중인 미적용 텍스트가 있는가 (씬 전환 가드 등 통합자 판단용) */
  isDirty(): boolean;
  dispose(): void;
}

// ── 순수 헬퍼 (DOM 비의존 — node 테스트 대상) ───────────────────────

/** 시퀀스 → 패널에 그릴 텍스트 (null이면 빈 문자열) */
export function sequenceToText(seq: ControlSequence | null): string {
  return seq === null ? '' : JSON.stringify(seq, null, JSON_INDENT);
}

/** 문자 오프셋 → 1-based 줄 번호 (텍스트 길이로 클램프) */
function lineOfOffset(text: string, offset: number): number {
  return text.slice(0, Math.max(0, Math.min(offset, text.length))).split('\n').length;
}

/**
 * JSON 파싱 오류를 **사람이 고칠 수 있는 한국어 한 줄**로 바꾼다.
 *
 * 엔진 메시지를 그대로 보이면 (a) 영문이고 (b) 위치가 몇 번째 줄인지 사용자가 세어야
 * 한다. 문제는 엔진마다 형식이 다르고 **최근 V8은 `at position N`을 더 이상 주지
 * 않는다**(`Unexpected token '}', ..."c":\n}" is not valid JSON`). 그래서 세 형식을
 * 모두 시도한다:
 *   1. `position N`      — 구 V8 · 일부 엔진
 *   2. `line N column M` — Firefox
 *   3. 인용된 원문 발췌  — 최근 V8. 발췌를 본문에서 찾아 그 **끝** 지점을 오류 위치로 본다
 * 셋 다 실패하면 위치 없이 무엇을 확인할지라도 한국어로 말한다 (§4-b).
 */
export function jsonErrorKo(err: unknown, text: string): string {
  const raw = err instanceof Error ? err.message : String(err);
  const at = (line: number): string => `JSON 형식 오류 — ${line}번째 줄 근처를 확인하세요`;

  const byPosition = /position (\d+)/i.exec(raw);
  if (byPosition?.[1] !== undefined) return at(lineOfOffset(text, Number(byPosition[1])));

  const byLine = /line (\d+)/i.exec(raw);
  if (byLine?.[1] !== undefined) return at(Math.max(1, Number(byLine[1])));

  // 최근 V8: `... ..."<발췌>" is not valid JSON`. 발췌는 오류 지점 **직전까지의** 원문이다.
  const byExcerpt = /"([\s\S]*)"\s+is not valid JSON/.exec(raw);
  const excerpt = byExcerpt?.[1];
  if (excerpt !== undefined && excerpt !== '') {
    const idx = text.indexOf(excerpt);
    if (idx >= 0) return at(lineOfOffset(text, idx + excerpt.length));
  }

  // 입력이 도중에 끝난 경우 — 위치는 항상 본문 끝이다
  if (/unexpected end of (json )?input/i.test(raw)) {
    return `JSON 형식 오류 — 입력이 끝까지 닫히지 않았습니다 (${lineOfOffset(text, text.length)}번째 줄까지 확인)`;
  }

  return `JSON 형식 오류 — 괄호·쉼표·따옴표를 확인하세요`;
}

/** 버전 행 표기: 'v3 · 5분 전 · 노드 12개' */
export function versionSummaryKo(entry: SequenceVersion, ageText: string): string {
  return `v${entry.version} · ${ageText} · 노드 ${entry.stepCount}개`;
}

// ── 마운트 ──────────────────────────────────────────────────────────

/**
 * '{} JSON' 토글 버튼을 buttonHost(커맨드바 우측 슬롯)에, 슬라이드 패널을
 * panelHost(보통 document.body)에 마운트한다.
 */
export function mountJsonViewer(
  buttonHost: HTMLElement,
  panelHost: HTMLElement,
  deps: JsonViewerDeps,
): JsonViewerHandle {
  ensureThemeStyles();
  const editable = deps.applyJson !== undefined;
  const hasVersions = deps.listVersions !== undefined && deps.restoreVersion !== undefined;
  const ageOf = (atIso: string): string => deps.describeAge?.(atIso) ?? atIso;

  // 토글 버튼 — 열림 상태는 .ui-btn--active + aria-pressed로 표현한다.
  // 액센트는 **토글 상태 전용**이다: 액센트 면(primary)은 ▶ 재생/생성의 몫 (C-14).
  const toggleButton = makeIconButton(
    'braces',
    'JSON',
    editable ? 'ControlSequence JSON 편집 · 버전 이력' : '현재 ControlSequence 원본 JSON 보기',
    'json-toggle',
  );
  toggleButton.setAttribute('aria-pressed', 'false');
  setCommandBarPriority(toggleButton, COMMAND_BAR_PRIORITY.view);
  buttonHost.appendChild(toggleButton);

  // 슬라이드 패널 (우측, 기본 닫힘)
  const panel = styled(document.createElement('div'), {
    position: 'fixed',
    top: `${PANEL_TOP_PX}px`,
    right: '0',
    bottom: `${PANEL_BOTTOM_PX}px`,
    width: `${PANEL_WIDTH_PX}px`,
    maxWidth: '90vw',
    zIndex: Z_INDEX.slidePanel,
    display: 'flex',
    flexDirection: 'column',
    background: COLOR.bgPanel,
    borderLeft: `1px solid ${COLOR.border}`,
    borderBottom: `1px solid ${COLOR.border}`,
    boxShadow: SHADOW.panel,
    color: COLOR.text,
    boxSizing: 'border-box',
    transform: 'translateX(100%)',
    transition: PANEL_TRANSITION,
    pointerEvents: 'auto',
  });
  applyType(panel, TYPE.body);
  panel.dataset.testid = 'json-viewer';
  panel.setAttribute('aria-hidden', 'true');
  // 닫힌(화면 밖) 패널의 버튼이 Tab 순서에 남지 않게 — aria-hidden 영역으로
  // 포커스가 들어가는 WAI-ARIA 위반 방지 (키보드 가시성, UX_DESIGN §9)
  panel.inert = true;
  // 패널 위 상호작용이 뷰포트(OrbitControls)로 전파되지 않게 차단 (dock과 동일 규약)
  for (const type of ['pointerdown', 'pointermove', 'pointerup', 'wheel', 'contextmenu']) {
    panel.addEventListener(type, (e) => {
      e.stopPropagation();
    });
  }

  // ── 헤더: 제목 + 복사 + 닫기 ──────────────────────────────────────
  const header = styled(document.createElement('div'), {
    display: 'flex',
    alignItems: 'center',
    gap: SPACE.sm,
    padding: `${SPACE.sm} ${SPACE.md}`,
    borderBottom: `1px solid ${COLOR.border}`,
    flexShrink: '0',
  });
  const headerTitle = styled(document.createElement('span'), {
    color: COLOR.textStrong,
    flex: '1',
    minWidth: '0',
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  });
  applyType(headerTitle, TYPE.bodyStrong);
  // 'ControlSequence'는 도메인 원문이라 번역하지 않는다 — 한국어 TTS가 CamelCase를
  // 철자로 읽지 않도록 영문 조각만 lang="en" span으로 감싼다 (§4-b 규칙 3).
  const titleEn = document.createElement('span');
  titleEn.lang = 'en';
  titleEn.textContent = 'ControlSequence JSON';
  headerTitle.appendChild(titleEn);
  if (!editable) headerTitle.append(' (읽기 전용)');

  const copyButton = makeIconButton('copy', '복사', 'JSON을 클립보드에 복사', 'json-copy');
  const closeButton = makeIconButton('close', '', 'JSON 패널 닫기 (Esc)', 'json-close', 'ghost');
  header.append(headerTitle, copyButton, closeButton);
  panel.appendChild(header);

  // ── 탭 (JSON / 버전) ──────────────────────────────────────────────
  type TabId = 'json' | 'versions';
  let tab: TabId = 'json';

  const tabBar = styled(document.createElement('div'), {
    display: hasVersions ? 'flex' : 'none',
    gap: SPACE.xs,
    padding: `${SPACE.xs} ${SPACE.md}`,
    borderBottom: `1px solid ${COLOR.border}`,
    flexShrink: '0',
  });
  tabBar.setAttribute('role', 'tablist');

  // DOM id 충돌 방지 (aria-controls/labelledby 짝 — dock.ts와 동일 관례)
  const uid = `json-viewer-${(mountSerial += 1)}`;
  const tabDomId = (id: TabId): string => `${uid}-tab-${id}`;
  const paneDomId = (id: TabId): string => `${uid}-pane-${id}`;

  const makeTab = (id: TabId, labelKo: string, langEn = false): HTMLButtonElement => {
    const btn = makeButton(labelKo, `${labelKo} 탭`, `json-tab-${id}`, 'ghost');
    btn.setAttribute('role', 'tab');
    btn.id = tabDomId(id);
    btn.setAttribute('aria-controls', paneDomId(id));
    if (langEn) btn.lang = 'en'; // 도메인 원문 — 한국어 TTS가 철자로 읽지 않게 (§4-b 규칙 3)
    btn.addEventListener('click', () => {
      setTab(id);
    });
    tabBar.appendChild(btn);
    return btn;
  };
  const jsonTab = makeTab('json', 'JSON', true);
  const versionsTab = makeTab('versions', '버전');
  // 방향키 소유 — shortcuts.ts의 widgetOwnsKey가 [role="tablist"] 안의 ←/→를 위젯
  // 소유로 판정해 라우터를 양보시킨다. 여기에 핸들러가 없으면 그 키는 **어디서도 처리되지
  // 않는 사장 키**가 된다(라우터 헤더가 못 박은 계약의 반대편 실패).
  const tabRoving = rovingTabindex(tabBar, [jsonTab, versionsTab], { orientation: 'horizontal' });
  panel.appendChild(tabBar);

  // ── 외부 변경 배너 (편집 중 refresh) ──────────────────────────────
  const staleBanner = styled(document.createElement('div'), {
    display: 'none',
    alignItems: 'center',
    gap: SPACE.sm,
    padding: `${SPACE.sm} ${SPACE.md}`,
    background: STATUS.warn.bg,
    color: STATUS.warn.fg,
    borderBottom: `1px solid ${STATUS.warn.border}`,
    flexShrink: '0',
  });
  applyType(staleBanner, TYPE.caption);
  staleBanner.dataset.testid = 'json-stale-banner';
  staleBanner.setAttribute('role', 'status');
  const staleText = document.createElement('span');
  staleText.style.flex = '1';
  staleText.textContent = '밖에서 시퀀스가 바뀌었습니다 — 내 편집은 그대로 두었습니다';
  const staleReload = makeButton('현재 값 불러오기', '편집을 버리고 현재 시퀀스를 불러온다', 'json-stale-reload', 'ghost');
  staleBanner.append(staleText, staleReload);
  panel.appendChild(staleBanner);

  // ── JSON 탭 본문 ──────────────────────────────────────────────────
  const jsonPane = styled(document.createElement('div'), {
    flex: '1',
    minHeight: '0',
    display: 'flex',
    flexDirection: 'column',
  });

  /** 읽기 전용 모드의 <pre> — 편집 가능하면 textarea가 대신한다 */
  const pre = styled(document.createElement('pre'), {
    flex: '1',
    margin: '0',
    padding: `${SPACE.md} ${SPACE.lg}`,
    overflow: 'auto',
    whiteSpace: 'pre',
    color: COLOR.text,
    display: editable ? 'none' : '',
  });
  applyType(pre, TYPE.monoBody);
  pre.dataset.testid = 'json-content';
  pre.lang = 'en'; // 내용은 도메인 원문(step kind·필드명) — §4-b 규칙 3
  pre.classList.add('ui-scroll');

  const editor = styled(document.createElement('textarea'), {
    flex: '1',
    minHeight: '0',
    margin: '0',
    padding: `${SPACE.md} ${SPACE.lg}`,
    border: 'none',
    outline: 'none',
    resize: 'none',
    background: SURFACE.sunken,
    color: COLOR.text,
    display: editable ? '' : 'none',
    whiteSpace: 'pre',
    overflowWrap: 'normal',
    overflow: 'auto',
    tabSize: '2',
  });
  applyType(editor, TYPE.monoBody);
  editor.dataset.testid = 'json-editor';
  editor.spellcheck = false;
  editor.setAttribute('aria-label', 'ControlSequence JSON 편집');
  editor.lang = 'en'; // 내용은 도메인 원문(step kind·필드명) — §4-b 규칙 3
  editor.classList.add('ui-scroll');

  /** 인라인 오류 (파싱/검증) — 색만이 아니라 문구로 전달한다 (§9 a11y) */
  const errorBox = styled(document.createElement('div'), {
    display: 'none',
    padding: `${SPACE.sm} ${SPACE.md}`,
    borderTop: `1px solid ${COLLISION.border}`,
    background: COLLISION.surface,
    color: COLLISION.text,
    whiteSpace: 'pre-wrap',
    maxHeight: '30%',
    overflow: 'auto',
    flexShrink: '0',
  });
  applyType(errorBox, TYPE.caption);
  errorBox.dataset.testid = 'json-error';
  errorBox.setAttribute('role', 'alert');

  /** 하단 액션 바 — 적용 / 편집 취소 + 상태 문구 */
  const actionBar = styled(document.createElement('div'), {
    display: editable ? 'flex' : 'none',
    alignItems: 'center',
    gap: SPACE.sm,
    padding: `${SPACE.sm} ${SPACE.md}`,
    borderTop: `1px solid ${COLOR.border}`,
    flexShrink: '0',
  });
  const statusText = styled(document.createElement('span'), {
    flex: '1',
    minWidth: '0',
    color: COLOR.muted,
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  });
  applyType(statusText, TYPE.caption);
  statusText.dataset.testid = 'json-status';
  const revertButton = makeButton('편집 취소', '편집을 버리고 현재 시퀀스로 되돌린다', 'json-revert', 'ghost');
  const applyButton = makeButton('적용', '이 JSON을 시퀀스에 적용한다 (Ctrl+Enter)', 'json-apply', 'primary');
  actionBar.append(statusText, revertButton, applyButton);

  // 빈 시퀀스 안내 (UX_DESIGN §7 "빈 플로우")
  const emptyState = styled(document.createElement('div'), {
    flex: '1',
    display: 'none',
    alignItems: 'center',
    justifyContent: 'center',
    padding: `0 ${SPACE.xxl}`,
    color: COLOR.muted,
    textAlign: 'center',
  });
  applyType(emptyState, TYPE.body);
  emptyState.textContent =
    '이 씬에는 시퀀스가 없습니다 — 플로우에서 노드를 만들거나 {scene, sequence} 봉투 JSON을 업로드하세요';
  emptyState.dataset.testid = 'json-empty';

  jsonPane.append(pre, editor, emptyState, errorBox, actionBar);
  if (hasVersions) {
    jsonPane.setAttribute('role', 'tabpanel');
    jsonPane.id = paneDomId('json');
    jsonPane.setAttribute('aria-labelledby', tabDomId('json'));
  }
  panel.appendChild(jsonPane);

  // ── 버전 탭 본문 ──────────────────────────────────────────────────
  const versionsPane = styled(document.createElement('div'), {
    flex: '1',
    minHeight: '0',
    display: 'none',
    flexDirection: 'column',
    overflow: 'auto',
  });
  versionsPane.classList.add('ui-scroll');
  versionsPane.dataset.testid = 'json-versions';
  if (hasVersions) {
    versionsPane.setAttribute('role', 'tabpanel');
    versionsPane.id = paneDomId('versions');
    versionsPane.setAttribute('aria-labelledby', tabDomId('versions'));
  }

  const versionsNote = styled(document.createElement('p'), {
    margin: '0',
    padding: `${SPACE.sm} ${SPACE.md}`,
    color: COLOR.muted,
    borderBottom: `1px solid ${COLOR.border}`,
  });
  applyType(versionsNote, TYPE.caption);
  versionsNote.textContent =
    '시퀀스 편집 이력입니다. 되돌리기도 새 버전으로 남아 다시 되돌릴 수 있습니다. (이 이력은 이 세션에만 유지됩니다 — 영구 보존은 저장을 쓰세요)';

  const versionsList = styled(document.createElement('div'), {
    display: 'flex',
    flexDirection: 'column',
  });
  versionsPane.append(versionsNote, versionsList);
  panel.appendChild(versionsPane);
  panelHost.appendChild(panel);

  // ── 상태 ──────────────────────────────────────────────────────────

  let open = false;
  let copyFlashTimer: ReturnType<typeof setTimeout> | null = null;
  /** 마지막으로 패널에 실어 넣은 "현재 진실" 텍스트 — dirty 판정 기준 */
  let baselineText = '';
  /** 편집 중 외부 변경이 있었는가 (배너 표시) */
  let stale = false;
  /** 되돌리기 확인 대기 중인 버전 (없으면 NO_PENDING_RESTORE) */
  let pendingRestore = NO_PENDING_RESTORE;

  const isDirty = (): boolean => editable && editor.value !== baselineText;

  const showErrors = (errors: readonly string[]): void => {
    errorBox.textContent = errors.join('\n');
    errorBox.style.display = errors.length > 0 ? '' : 'none';
  };

  const paintActions = (): void => {
    const dirty = isDirty();
    applyButton.disabled = !dirty;
    revertButton.disabled = !dirty;
    statusText.textContent = dirty ? '수정됨 — 적용하지 않았습니다' : '현재 시퀀스와 같습니다';
    statusText.style.color = dirty ? STATUS.warn.fg : COLOR.muted;
    staleBanner.style.display = stale && dirty ? 'flex' : 'none';
    if (!dirty) stale = false;
  };

  /**
   * 현재 시퀀스 진실을 패널에 싣는다 (편집 내용을 덮어쓴다 — 호출자가 판단).
   *
   * **불변식: `baselineText`를 바꾸는 모든 지점은 `editor.value`와 짝을 맞춘다.**
   * 한쪽만 바꾸면 `isDirty()`가 영구히 true로 굳어 (a) Escape가 죽고(편집 중 보호가
   * 오작동) (b) 이후 refresh가 전부 stale 배너로 빠져 **시퀀스가 있는 씬인데 "시퀀스가
   * 없습니다"를 계속 보여준다.** 시퀀스 없는 씬(falling-boxes 등)을 지나갈 때 실제로
   * 발생하던 경로다.
   */
  const loadFromTruth = (): void => {
    const seq = deps.getSequence();
    const text = sequenceToText(seq);
    baselineText = text;
    stale = false;
    if (editable) editor.value = text; // seq === null이면 '' — 기준선과 짝을 맞춘다
    else pre.textContent = text;
    showErrors([]);
    if (seq === null) {
      pre.style.display = 'none';
      editor.style.display = 'none';
      actionBar.style.display = 'none';
      emptyState.style.display = 'flex';
      copyButton.disabled = true;
      paintActions();
      return;
    }
    emptyState.style.display = 'none';
    copyButton.disabled = false;
    if (editable) {
      editor.style.display = '';
      actionBar.style.display = 'flex';
    } else {
      pre.style.display = '';
    }
    paintActions();
  };

  const renderVersions = (): void => {
    if (!hasVersions) return;
    versionsList.textContent = '';
    pendingRestore = NO_PENDING_RESTORE;
    const entries = deps.listVersions?.() ?? [];
    if (entries.length === 0) {
      const empty = styled(document.createElement('p'), {
        margin: '0',
        padding: `${SPACE.xl} ${SPACE.md}`,
        color: COLOR.muted,
        textAlign: 'center',
      });
      applyType(empty, TYPE.body);
      empty.textContent = '아직 편집 이력이 없습니다';
      empty.dataset.testid = 'json-versions-empty';
      versionsList.appendChild(empty);
      return;
    }
    entries.forEach((entry, index) => {
      const isCurrent = index === 0; // 목록은 최신 우선
      const row = styled(document.createElement('div'), {
        display: 'flex',
        alignItems: 'center',
        gap: SPACE.sm,
        padding: `${SPACE.sm} ${SPACE.md}`,
        borderBottom: `1px solid ${BORDER.subtle}`,
        background: isCurrent ? STATUS.running.bg : 'transparent',
      });
      row.dataset.testid = `json-version-${entry.version}`;

      const textCol = styled(document.createElement('div'), { flex: '1', minWidth: '0' });
      const line1 = applyType(document.createElement('div'), TYPE.bodyStrong);
      styled(line1, { color: COLOR.textStrong, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' });
      line1.textContent = entry.labelKo;
      const line2 = applyType(document.createElement('div'), TYPE.caption);
      styled(line2, { color: COLOR.muted });
      line2.textContent = versionSummaryKo(entry, ageOf(entry.atIso)) + (isCurrent ? ' · 현재' : '');
      textCol.append(line1, line2);
      row.appendChild(textCol);

      if (!isCurrent) {
        const restore = makeButton(
          '되돌리기',
          `v${entry.version}로 되돌리기`, // 보이는 라벨을 포함한다 (WCAG 2.5.3 Label in Name)
          `json-restore-${entry.version}`,
          'ghost',
        );
        restore.addEventListener('click', () => {
          // 되돌리기는 시퀀스를 통째로 바꾼다 — 확인 한 단계를 둔다. 확인 자체도
          // 되돌릴 수 있으므로(새 버전으로 append) 무거운 모달까지는 쓰지 않는다.
          // 같은 행 재클릭은 **토글**이다 — 조용한 no-op을 만들지 않는다 (§6).
          if (pendingRestore === entry.version) {
            clearRestoreConfirm();
            return;
          }
          renderRestoreConfirm(row, entry);
        });
        row.appendChild(restore);
      }
      versionsList.appendChild(row);
    });
  };

  /**
   * 떠 있는 확인 바를 걷어낸다. 스칼라 `pendingRestore`만으로는 직전 바를 잊어
   * **서로 다른 버전을 가리키는 확인 바가 동시에 두 개** 뜰 수 있다 — 시퀀스를 통째로
   * 바꾸는 조작에서 가장 위험한 종류의 애매함이다. DOM을 진실로 삼아 정리한다.
   */
  const clearRestoreConfirm = (): void => {
    pendingRestore = NO_PENDING_RESTORE;
    for (const el of versionsPane.querySelectorAll(`[data-testid^="${RESTORE_CONFIRM_PREFIX}"]`)) {
      el.remove();
    }
  };

  /** 행 안에서 확인 → 실행. 인라인 확인 바(모달 없이 되돌릴 수 있는 조작) */
  const renderRestoreConfirm = (row: HTMLElement, entry: SequenceVersion): void => {
    clearRestoreConfirm(); // 확인 바는 언제나 하나뿐이다
    pendingRestore = entry.version;
    const bar = styled(document.createElement('div'), {
      display: 'flex',
      alignItems: 'center',
      gap: SPACE.sm,
      padding: `${SPACE.sm} ${SPACE.md}`,
      background: STATUS.warn.bg,
      color: STATUS.warn.fg,
      borderBottom: `1px solid ${STATUS.warn.border}`,
    });
    applyType(bar, TYPE.caption);
    bar.dataset.testid = `${RESTORE_CONFIRM_PREFIX}${entry.version}`;
    const msg = document.createElement('span');
    msg.style.flex = '1';
    msg.textContent = `v${entry.version}(${entry.labelKo})로 되돌립니다 — 지금 시퀀스는 이력에 남습니다`;
    // title은 보이는 라벨을 포함한다 (WCAG 2.5.3 — 음성 제어가 '되돌리기'로 매칭되게)
    const yes = makeButton('되돌리기', `v${entry.version}로 되돌리기 확인`, `json-restore-yes-${entry.version}`, 'primary');
    const no = makeButton('취소', '되돌리기 취소', `json-restore-no-${entry.version}`, 'ghost');
    no.addEventListener('click', clearRestoreConfirm);
    yes.addEventListener('click', () => {
      clearRestoreConfirm();
      const result = deps.restoreVersion?.(entry.version) ?? { ok: false, errors: ['되돌리기를 지원하지 않습니다'] };
      if (!result.ok) {
        // 목록을 다시 그린다 — cap 폐기로 실패한 경우 사라진 행이 남아 다시 눌리면 안 된다
        renderVersions();
        setTab('json');
        showErrors(result.errors);
        return;
      }
      // 성공 — 통합자가 refresh()를 부르지만, 편집 중이 아니었다면 즉시 반영한다
      loadFromTruth();
      renderVersions();
    });
    bar.append(msg, yes, no);
    row.insertAdjacentElement('afterend', bar);
  };

  function setTab(next: TabId): void {
    tab = hasVersions ? next : 'json';
    jsonPane.style.display = tab === 'json' ? 'flex' : 'none';
    versionsPane.style.display = tab === 'versions' ? 'flex' : 'none';
    jsonTab.classList.toggle('ui-btn--active', tab === 'json');
    versionsTab.classList.toggle('ui-btn--active', tab === 'versions');
    jsonTab.setAttribute('aria-selected', String(tab === 'json'));
    versionsTab.setAttribute('aria-selected', String(tab === 'versions'));
    tabRoving.setActive(tab === 'json' ? 0 : 1);
    if (tab === 'versions') renderVersions();
  }

  const paintToggle = (): void => {
    // 인라인이 아닌 클래스 토글 — hover 등 상태 스타일이 살아 있게 유지한다
    toggleButton.classList.toggle('ui-btn--active', open);
    toggleButton.setAttribute('aria-pressed', String(open));
  };

  /** 열려 있을 때만 Escape로 닫는다 (버튼 title의 '(Esc)' 약속을 실제로 지킨다) */
  const onKeyDown = (e: KeyboardEvent): void => {
    if (e.key !== 'Escape' || !open) return;
    // 편집 중이면 Escape는 "닫기"가 아니라 아무것도 하지 않는다 — 30초 타이핑이
    // 습관적인 Escape 한 번에 사라지는 것을 막는다(편집 취소는 명시 버튼).
    if (isDirty()) return;
    e.preventDefault();
    setOpen(false);
    toggleButton.focus();
  };

  function setOpen(next: boolean): void {
    open = next;
    if (open && !isDirty()) loadFromTruth(); // 열 때마다 현재 진실로 갱신 (편집 중이면 보존)
    if (open && tab === 'versions') renderVersions();
    panel.style.transform = open ? 'translateX(0)' : 'translateX(100%)';
    panel.setAttribute('aria-hidden', String(!open));
    panel.inert = !open; // 닫힘 = 포커스/상호작용 불가 (트랜스폼 애니메이션은 유지)
    if (open) window.addEventListener('keydown', onKeyDown);
    else window.removeEventListener('keydown', onKeyDown);
    paintToggle();
  }

  // ── 편집 배선 ─────────────────────────────────────────────────────

  const doApply = (): void => {
    if (!editable || !isDirty()) return;
    const text = editor.value;
    // 1) 형식 — 파싱 실패는 위치를 줄 번호로 환산해 알려준다
    try {
      JSON.parse(text) as unknown;
    } catch (err) {
      showErrors([jsonErrorKo(err, text)]);
      return;
    }
    // 2) 스키마 + 씬 참조 검증 + 커밋 — 통합자 몫 (§2.8/§2.9)
    const result = deps.applyJson?.(text) ?? { ok: false, errors: ['편집이 지원되지 않습니다'] };
    if (!result.ok) {
      showErrors(result.errors);
      return; // 텍스트는 사용자가 쓴 그대로 남는다 — 고쳐 쓸 수 있어야 한다
    }
    showErrors([]);
    // 커밋 성공 — 통합자의 refresh()가 곧 오지만, 기준선을 즉시 맞춰 dirty를 내린다.
    // (통합자가 정규화한 결과와 다를 수 있으므로 진실에서 다시 싣는다.)
    loadFromTruth();
  };

  editor.addEventListener('input', () => {
    paintActions();
    if (errorBox.style.display !== 'none') showErrors([]); // 고치기 시작하면 오류를 지운다
  });
  editor.addEventListener('keydown', (e) => {
    // 패널 내부 한정 — 전역 키맵을 건드리지 않는다 (§2.10)
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      doApply();
      return;
    }
    if (e.key === 'Tab') {
      // 코드 편집기에서 Tab은 들여쓰기다. 다만 포커스 탈출 경로를 없애면 키보드
      // 사용자가 갇히므로 Shift+Tab은 그대로 흘려보낸다(이전 요소로 이동).
      if (e.shiftKey) return;
      e.preventDefault();
      const indent = ' '.repeat(JSON_INDENT);
      // execCommand는 폐기 예정이지만 **textarea의 네이티브 undo 스택을 보존하는 유일한
      // 방법**이다. `editor.value = ...` 대입은 스택을 파기해, Tab을 한 번 누른 뒤로는
      // 편집기 안에서 Ctrl+Z가 이전 타이핑을 되돌리지 못한다(전역 Ctrl+Z는 텍스트 입력
      // 대상을 제외하므로 브라우저 undo가 이 안의 유일한 축이다). 실패 시 대입으로 폴백.
      let inserted = false;
      try {
        inserted = document.execCommand('insertText', false, indent);
      } catch {
        inserted = false;
      }
      if (!inserted) {
        const start = editor.selectionStart;
        const end = editor.selectionEnd;
        editor.value = `${editor.value.slice(0, start)}${indent}${editor.value.slice(end)}`;
        editor.selectionStart = editor.selectionEnd = start + indent.length;
      }
      paintActions();
    }
  });
  applyButton.addEventListener('click', doApply);
  revertButton.addEventListener('click', () => {
    loadFromTruth();
    editor.focus();
  });
  staleReload.addEventListener('click', () => {
    loadFromTruth();
    editor.focus();
  });

  toggleButton.addEventListener('click', () => {
    setOpen(!open);
  });
  closeButton.addEventListener('click', () => {
    setOpen(false);
  });

  // 라벨 스팬만 갈아 끼운다 — 버튼 내용을 통째로 바꾸면 아이콘이 사라지고
  // `.ui-btn--icon-only` 축약도 깨진다 (UX_AUDIT C-13).
  const copyLabel = copyButton.querySelector<HTMLElement>('.ui-btn__label');
  const flashCopyLabel = (label: string): void => {
    if (copyLabel !== null) copyLabel.textContent = label;
    copyButton.title = label;
    if (copyFlashTimer !== null) clearTimeout(copyFlashTimer);
    copyFlashTimer = setTimeout(() => {
      if (copyLabel !== null) copyLabel.textContent = '복사';
      copyButton.title = 'JSON을 클립보드에 복사';
      copyFlashTimer = null;
    }, COPY_FLASH_MS);
  };

  copyButton.addEventListener('click', () => {
    // 편집 중이면 화면에 보이는 것(편집본)을 복사한다 — 보이는 것과 다른 것을
    // 복사하는 것은 사용자가 알 방법이 없는 배신이다.
    const text = editable ? editor.value : sequenceToText(deps.getSequence());
    if (text === '') return;
    // 클립보드 API는 secure context 전용 — 불가하면 실패를 그대로 표시한다
    if (!('clipboard' in navigator)) {
      flashCopyLabel('복사 실패');
      return;
    }
    navigator.clipboard.writeText(text).then(
      () => {
        flashCopyLabel('복사됨');
      },
      () => {
        flashCopyLabel('복사 실패');
      },
    );
  });

  setTab('json');
  loadFromTruth();
  paintToggle();

  return {
    el: toggleButton,
    refresh: (): void => {
      if (isDirty()) {
        // 사용자가 편집 중이다 — 덮어쓰지 않고 배너로만 알린다 (파일 헤더의 계약)
        stale = true;
        paintActions();
        if (tab === 'versions') renderVersions();
        return;
      }
      loadFromTruth();
      if (tab === 'versions') renderVersions();
    },
    isDirty,
    dispose: (): void => {
      if (copyFlashTimer !== null) clearTimeout(copyFlashTimer);
      window.removeEventListener('keydown', onKeyDown);
      tabRoving.dispose();
      toggleButton.remove();
      panel.remove();
    },
  };
}
