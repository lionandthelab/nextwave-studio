// ui/help-sheet.ts — 인앱 도움말 · 단축키 시트 (docs/UX_AUDIT.md C-12, C-18)
//
// 구 상태: 앱 안에 도움말·단축키 표·투어가 **0건**이었다. 단축키는 4개 파일에 흩어져
// 존재하지만 그것을 알려주는 화면이 없어, 이미 구현된 기능의 가치조차 회수하지 못했다.
//
// ── 중요한 설계 결정 ────────────────────────────────────────────────
// 이 시트는 `docs/UX_DESIGN.md` §9의 규정 목록을 옮겨 적지 **않는다.**
// `ShortcutRouter.list()`가 반환하는 **실제 등록된 바인딩**만 렌더한다.
// 문서를 그대로 베끼면 구현되지 않은 키(`F`, `Home`, `←/→ Step`이 실제로 없었다)를
// 광고하게 되고, 그것은 도움말이 없는 것보다 나쁘다.

import {
  BORDER,
  BORDER_WIDTH,
  COLOR,
  RADIUS,
  SHADOW,
  SPACE,
  SURFACE,
  TYPE,
  Z_INDEX,
  applyType,
  ensureThemeStyles,
  makeButton,
  makePanelHeader,
  styled,
} from './theme';
import { setButtonContent } from './icons';
import { trapFocus } from './a11y';
import type { FocusTrapHandle } from './a11y';
import { BRAND_NAME, PRODUCT_NAME, PRODUCT_TAGLINE } from './brand';
import { formatKeys } from './shortcuts';
import type { ShortcutBinding } from './shortcuts';

export interface HelpSheetDeps {
  /** 현재 **실제 등록된** 바인딩 (호출 시점에 조회 — 동적 등록/해제 반영) */
  listShortcuts(): readonly ShortcutBinding[];
  /** 사용법 문서 링크 (없으면 링크를 그리지 않는다) */
  readonly usageDocUrl?: string;
}

export interface HelpSheetHandle {
  open(): void;
  close(): void;
  isOpen(): boolean;
  dispose(): void;
}

const HELP_STYLE_ID = 'rsw-help-styles';

function ensureHelpStyles(): void {
  if (document.getElementById(HELP_STYLE_ID) !== null) return;
  const style = document.createElement('style');
  style.id = HELP_STYLE_ID;
  style.textContent = `
.rsw-kbd {
  display: inline-block;
  min-width: 20px;
  padding: 0 ${SPACE.sm};
  border: ${BORDER_WIDTH.hair} solid ${BORDER.strong};
  border-bottom-width: 2px;
  border-radius: ${RADIUS.sm};
  background: ${SURFACE.raised};
  color: ${COLOR.textStrong};
  font-family: var(--rsw-font-mono);
  font-size: ${TYPE.micro.sizePx}px;
  line-height: 18px;
  text-align: center;
  white-space: nowrap;
}
.rsw-help-row {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: ${SPACE.lg};
  padding: ${SPACE.xs} 0;
}
.rsw-help-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 0 ${SPACE.xxl};
}
`;
  document.head.appendChild(style);
}

/**
 * `?` 도움말 시트를 마운트한다(초기엔 닫힘).
 *
 * `role="dialog"` + `aria-modal="true"`를 선언하므로 **반드시 포커스 트랩을 건다** —
 * 선언만 하고 Tab이 배경으로 빠지면 스크린리더에게는 "바깥은 없는 셈"인데 실제 포커스는
 * 읽히지 않는 요소에 가 있는 유령 상태가 된다.
 */
export function mountHelpSheet(host: HTMLElement, deps: HelpSheetDeps): HelpSheetHandle {
  ensureThemeStyles();
  ensureHelpStyles();

  const scrim = styled(document.createElement('div'), {
    position: 'fixed',
    inset: '0',
    background: 'rgba(6, 8, 12, 0.62)',
    zIndex: Z_INDEX.modal,
    display: 'none',
    alignItems: 'center',
    justifyContent: 'center',
    padding: SPACE.xxl,
    boxSizing: 'border-box',
  });
  scrim.dataset.testid = 'help-scrim';

  const panel = styled(document.createElement('div'), {
    display: 'flex',
    flexDirection: 'column',
    width: 'min(760px, 100%)',
    maxHeight: '100%',
    background: SURFACE.modal,
    border: `${BORDER_WIDTH.hair} solid ${BORDER.default}`,
    borderRadius: RADIUS.lg,
    boxShadow: SHADOW.modal,
    overflow: 'hidden',
    boxSizing: 'border-box',
  });
  panel.setAttribute('role', 'dialog');
  panel.setAttribute('aria-modal', 'true');
  panel.setAttribute('aria-label', `${PRODUCT_NAME} 도움말 · 단축키`);
  panel.dataset.testid = 'help-panel';
  panel.tabIndex = -1;

  const header = makePanelHeader(`${PRODUCT_NAME} 도움말`, {
    actions: true,
    headingTag: 'h2',
    testId: 'help-header',
  });
  const closeBtn = makeButton('', '닫기 (Esc)', 'help-close', 'ghost');
  setButtonContent(closeBtn, 'close', '');
  styled(closeBtn, { minWidth: '28px', minHeight: '28px', justifyContent: 'center' });
  header.actionsEl?.appendChild(closeBtn);
  panel.appendChild(header.el);

  const body = styled(document.createElement('div'), {
    flex: '1 1 auto',
    minHeight: '0',
    overflowY: 'auto',
    padding: SPACE.xxl,
  });
  body.classList.add('ui-scroll');
  panel.appendChild(body);

  scrim.appendChild(panel);
  host.appendChild(scrim);

  let trap: FocusTrapHandle | null = null;
  let open = false;

  const renderBody = (): void => {
    body.textContent = '';

    // 제품 소개 — 첫 사용자가 "이게 뭘 하는 도구인가"를 여기서 안다
    const intro = document.createElement('p');
    applyType(intro, TYPE.body);
    styled(intro, { margin: `0 0 ${SPACE.xxl} 0`, color: COLOR.muted });
    intro.textContent = `${PRODUCT_TAGLINE}. 라이브러리에서 로봇과 사물을 뷰포트로 끌어다 놓고, 자연어나 노드 그래프로 제어 시퀀스를 만든 뒤 재생해 충돌을 관찰합니다.`;
    body.appendChild(intro);

    // 그룹별 단축키 — 실제 등록된 것만
    // hidden 별칭(←→↑↓의 나머지 3방향, Shift 미세 변형 등)은 대표 줄이 대신 표기한다
    const bindings = deps.listShortcuts().filter((b) => b.hidden !== true);
    const groups = new Map<string, ShortcutBinding[]>();
    for (const b of bindings) {
      const list = groups.get(b.group);
      if (list === undefined) groups.set(b.group, [b]);
      else list.push(b);
    }

    if (groups.size === 0) {
      const empty = document.createElement('p');
      applyType(empty, TYPE.body);
      styled(empty, { color: COLOR.muted, margin: '0' });
      empty.textContent = '등록된 단축키가 없습니다.';
      body.appendChild(empty);
    } else {
      const grid = document.createElement('div');
      grid.className = 'rsw-help-grid';

      for (const [groupName, list] of groups) {
        const section = document.createElement('section');
        styled(section, { marginBottom: SPACE.xxl, breakInside: 'avoid' });
        section.setAttribute('aria-label', groupName);

        const h = document.createElement('h3');
        applyType(h, TYPE.subhead);
        styled(h, {
          margin: `0 0 ${SPACE.md} 0`,
          color: COLOR.accentText,
          paddingBottom: SPACE.xs,
          borderBottom: `${BORDER_WIDTH.hair} solid ${BORDER.subtle}`,
        });
        h.textContent = groupName;
        section.appendChild(h);

        for (const b of list) {
          const row = document.createElement('div');
          row.className = 'rsw-help-row';

          const label = document.createElement('span');
          applyType(label, TYPE.body);
          styled(label, { color: COLOR.text, minWidth: '0' });
          label.textContent = b.labelKo;

          const keys = document.createElement('span');
          styled(keys, { flex: 'none', display: 'flex', gap: SPACE.xs });
          for (const part of b.keysDisplay ?? formatKeys(b.keys).split(' + ')) {
            const kbd = document.createElement('kbd');
            kbd.className = 'rsw-kbd';
            kbd.textContent = part;
            keys.appendChild(kbd);
          }

          row.append(label, keys);
          section.appendChild(row);
        }
        grid.appendChild(section);
      }
      body.appendChild(grid);
    }

    // 푸터 — 문서 링크 + 브랜드
    const footer = document.createElement('div');
    styled(footer, {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      gap: SPACE.lg,
      marginTop: SPACE.xl,
      paddingTop: SPACE.lg,
      borderTop: `${BORDER_WIDTH.hair} solid ${BORDER.subtle}`,
    });

    const brand = document.createElement('span');
    applyType(brand, TYPE.caption);
    styled(brand, { color: COLOR.muted });
    brand.textContent = `${PRODUCT_NAME} · ${BRAND_NAME}`;
    footer.appendChild(brand);

    if (deps.usageDocUrl !== undefined) {
      const link = document.createElement('a');
      applyType(link, TYPE.caption);
      styled(link, { color: COLOR.accentText, textDecoration: 'none' });
      link.href = deps.usageDocUrl;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      link.textContent = '사용법 문서 열기 ↗';
      footer.appendChild(link);
    }
    body.appendChild(footer);
  };

  const doClose = (): void => {
    if (!open) return;
    open = false;
    scrim.style.display = 'none';
    trap?.release();
    trap = null;
  };

  const doOpen = (): void => {
    if (open) return;
    renderBody();
    scrim.style.display = 'flex';
    open = true;
    trap = trapFocus(panel, { initialFocus: closeBtn, onEscape: doClose });
  };

  closeBtn.addEventListener('click', doClose);
  // 스크림 클릭으로 닫기 (패널 안 클릭은 통과시키지 않는다)
  scrim.addEventListener('mousedown', (e) => {
    if (e.target === scrim) doClose();
  });

  return {
    open: doOpen,
    close: doClose,
    isOpen: (): boolean => open,
    dispose: (): void => {
      trap?.release();
      scrim.remove();
    },
  };
}
