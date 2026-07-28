// ui/command-bar/json-viewer.ts — '{} JSON' 토글 + ControlSequence 원본 뷰어
// (UX_DESIGN §3.1 "{} JSON": 현재 ControlSequence 원본 JSON 뷰어 — 읽기 + 복사)
//
// 커맨드바 우측의 토글 버튼과, 화면 우측에서 슬라이드로 나타나는 읽기 전용 패널.
// 시퀀스 진실은 글루(main.ts)가 getSequence 콜백으로 공급한다 — 이 모듈은 core를
// import하지 않고 schema 타입(POJO)만 안다 (계층 규칙, CLAUDE.md §3).
//
// 갱신 정책: 열 때마다 + refresh() 호출 시 다시 그린다. 씬 전환으로 시퀀스가 바뀌면
// 글루가 refresh()를 호출한다(그래프 편집이 생기는 Phase 8에서 구독 훅으로 확장).
// 시퀀스가 없는 씬은 '시퀀스 없음' 빈 상태를 보인다 (UX_DESIGN §7).

import { makeIconButton } from '../icons';
import {
  COLOR,
  LAYOUT,
  SHADOW,
  SPACE,
  TYPE,
  Z_INDEX,
  applyType,
  ensureThemeStyles,
  styled,
} from '../theme';
import { COMMAND_BAR_PRIORITY, setCommandBarPriority } from './scene-controls';
import type { ControlSequence } from '../../schema';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4, 시각 토큰은 ui/theme.ts) ────

/** 패널 폭(px) */
const PANEL_WIDTH_PX = 380;
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

// ── 공개 타입 ───────────────────────────────────────────────────────

export interface JsonViewerHandle {
  /** 토글 버튼 요소 (커맨드바 우측 슬롯에 마운트됨) */
  readonly el: HTMLElement;
  /** 시퀀스가 바뀌었을 때(씬 전환 등) 호출 — 패널 내용을 현재 진실로 다시 그린다 */
  refresh(): void;
  dispose(): void;
}

// ── 마운트 ──────────────────────────────────────────────────────────

/**
 * '{} JSON' 토글 버튼을 buttonHost(커맨드바 우측 슬롯)에, 슬라이드 패널을
 * panelHost(보통 document.body)에 마운트한다. getSequence는 현재 씬의 검증된
 * ControlSequence(없으면 null)를 돌려주는 글루 콜백이다.
 */
export function mountJsonViewer(
  buttonHost: HTMLElement,
  panelHost: HTMLElement,
  getSequence: () => ControlSequence | null,
): JsonViewerHandle {
  ensureThemeStyles();
  // 토글 버튼 — 열림 상태는 .ui-btn--active + aria-pressed로 표현한다.
  // 액센트는 **토글 상태 전용**이다: 액센트 면(primary)은 ▶ 재생/생성의 몫 (C-14).
  const toggleButton = makeIconButton(
    'braces',
    'JSON',
    '현재 ControlSequence 원본 JSON 보기 (읽기 전용)',
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
  // 닫힌(화면 밖) 패널의 복사/닫기 버튼이 Tab 순서에 남지 않게 — aria-hidden 영역으로
  // 포커스가 들어가는 WAI-ARIA 위반 방지 (키보드 가시성, UX_DESIGN §9)
  panel.inert = true;
  // 패널 위 상호작용이 뷰포트(OrbitControls)로 전파되지 않게 차단 (dock과 동일 규약)
  for (const type of ['pointerdown', 'pointermove', 'pointerup', 'wheel', 'contextmenu']) {
    panel.addEventListener(type, (e) => {
      e.stopPropagation();
    });
  }

  // 헤더: 제목 + 복사 + 닫기
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
  headerTitle.textContent = 'ControlSequence JSON (읽기 전용)';

  const copyButton = makeIconButton('copy', '복사', 'JSON을 클립보드에 복사', 'json-copy');

  const closeButton = makeIconButton('close', '', 'JSON 패널 닫기 (Esc)', 'json-close', 'ghost');

  header.appendChild(headerTitle);
  header.appendChild(copyButton);
  header.appendChild(closeButton);
  panel.appendChild(header);

  // 본문: JSON pre / 빈 상태
  const pre = styled(document.createElement('pre'), {
    flex: '1',
    margin: '0',
    padding: `${SPACE.md} ${SPACE.lg}`,
    overflow: 'auto',
    whiteSpace: 'pre',
    color: COLOR.text,
  });
  applyType(pre, TYPE.monoBody);
  pre.dataset.testid = 'json-content';
  pre.classList.add('ui-scroll');

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
    '이 씬에는 시퀀스가 없습니다 — {scene, sequence} 봉투 JSON을 업로드하면 여기 표시됩니다';
  emptyState.dataset.testid = 'json-empty';

  panel.appendChild(pre);
  panel.appendChild(emptyState);
  panelHost.appendChild(panel);

  // ── 상태 · 렌더 ───────────────────────────────────────────────────

  let open = false;
  let copyFlashTimer: ReturnType<typeof setTimeout> | null = null;

  /** 현재 시퀀스 진실(getSequence)로 본문을 다시 그린다 */
  const renderContent = (): void => {
    const seq = getSequence();
    if (seq === null) {
      pre.style.display = 'none';
      emptyState.style.display = 'flex';
      copyButton.disabled = true;
      return;
    }
    pre.textContent = JSON.stringify(seq, null, JSON_INDENT);
    pre.style.display = '';
    emptyState.style.display = 'none';
    copyButton.disabled = false;
  };

  const paintToggle = (): void => {
    // 인라인이 아닌 클래스 토글 — hover 등 상태 스타일이 살아 있게 유지한다
    toggleButton.classList.toggle('ui-btn--active', open);
    toggleButton.setAttribute('aria-pressed', String(open));
  };

  /** 열려 있을 때만 Escape로 닫는다 (버튼 title의 '(Esc)' 약속을 실제로 지킨다) */
  const onKeyDown = (e: KeyboardEvent): void => {
    if (e.key !== 'Escape' || !open) return;
    e.preventDefault();
    setOpen(false);
    toggleButton.focus();
  };

  function setOpen(next: boolean): void {
    open = next;
    if (open) renderContent(); // 열 때마다 현재 진실로 갱신
    panel.style.transform = open ? 'translateX(0)' : 'translateX(100%)';
    panel.setAttribute('aria-hidden', String(!open));
    panel.inert = !open; // 닫힘 = 포커스/상호작용 불가 (트랜스폼 애니메이션은 유지)
    if (open) window.addEventListener('keydown', onKeyDown);
    else window.removeEventListener('keydown', onKeyDown);
    paintToggle();
  }

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
    const seq = getSequence();
    if (seq === null) return;
    const text = JSON.stringify(seq, null, JSON_INDENT);
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

  renderContent();
  paintToggle();

  return {
    el: toggleButton,
    refresh: (): void => {
      // 닫혀 있어도 갱신 비용이 미미하므로 항상 다시 그린다 (열 때 또 한 번 갱신됨)
      renderContent();
    },
    dispose: (): void => {
      if (copyFlashTimer !== null) clearTimeout(copyFlashTimer);
      window.removeEventListener('keydown', onKeyDown);
      toggleButton.remove();
      panel.remove();
    },
  };
}
