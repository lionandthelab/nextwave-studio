// ui/viewport/drop-hint.ts — 드래그 배치 힌트 오버레이 (UX_AUDIT C-17)
//
// 라이브러리 카드를 어두운 뷰포트 위로 끌고 가는 동안 **아무 시각 피드백이 없었다.**
// dragover 핸들러는 preventDefault()와 dropEffect='copy'만 했고, library.ts:16 주석은
// "별도 프리뷰는 뷰포트의 반투명 고스트가 담당"이라며 책임을 넘겼지만 뷰포트에 그
// 구현이 없었다. 사용자는 놓고 → 확인하고 → 기즈모로 다시 옮긴다(3단계).
//
// ── 이 모듈의 범위 (그리고 범위가 아닌 것) ───────────────────────────
// `docs/UX_DESIGN.md` §3.3이 명시한 "바닥 레이캐스트 지점의 반투명 고스트"는 **render
// 계층 몫이다** — 레이캐스트도 메시도 ui가 만질 수 없다(CLAUDE.md §3). 여기서는
// ui가 정당하게 소유할 수 있는 절반만 제공한다: **드롭 대상 영역의 가장자리 하이라이트
// + "여기에 놓기" 힌트**. 즉 "놓을 수 있다"는 어포던스는 즉시 해결되고, "여기에 놓인다"는
// 정밀도는 render의 beginPlacementPreview/updatePlacementPreview가 채워야 한다.
//
// 배치 계약: host는 positioned(position:relative 등) 뷰포트 슬롯이어야 한다.
// **항상 pointer-events:none이다** — 드롭 대상 위에 얹힌 요소가 포인터를 먹으면
// dragover/drop 이벤트가 끊겨 힌트가 드롭 자체를 망가뜨린다.

import {
  BORDER_WIDTH,
  COLOR,
  ICON,
  MOTION,
  RADIUS,
  SELECT,
  SPACE,
  SURFACE,
  TYPE,
  Z_INDEX,
  applyType,
  ensureThemeStyles,
  styled,
  tr,
} from '../theme';
import { icon } from '../icons';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4) ─────────────────────────────

/** 가장자리 하이라이트 두께 */
const EDGE_WIDTH = BORDER_WIDTH.thick;
/** 하이라이트 안쪽 여백 — 뷰포트 경계선과 겹쳐 두꺼워 보이지 않게 */
const EDGE_INSET_PX = 4;

const DEFAULT_HINT_KO = '여기에 놓기';

// ── 공개 타입 ───────────────────────────────────────────────────────

export interface DropHintHandle {
  readonly el: HTMLElement;
  /**
   * 드래그 시작/종료. label은 끌고 있는 템플릿 이름(예: 'Box · 박스') — 없으면 기본 문구만.
   * 라이브러리 카드의 dragstart/dragend에서 그대로 켜고 끈다.
   */
  setActive(active: boolean, label?: string | null): void;
  dispose(): void;
}

// ── 순수 헬퍼 (DOM 비의존) ──────────────────────────────────────────

/** 힌트 문구 — 무엇을 놓는지 알면 함께 보여 준다 ('여기에 놓기 — Box · 박스') */
export function formatDropHintText(label: string | null | undefined): string {
  if (label === null || label === undefined || label.trim() === '') return DEFAULT_HINT_KO;
  return `${DEFAULT_HINT_KO} — ${label.trim()}`;
}

// ── 마운트 ──────────────────────────────────────────────────────────

/** 드래그 배치 힌트를 host(positioned 뷰포트 슬롯)에 마운트한다 (초기 비활성). */
export function mountDropHint(host: HTMLElement): DropHintHandle {
  ensureThemeStyles();

  const root = styled(document.createElement('div'), {
    position: 'absolute',
    inset: `${EDGE_INSET_PX}px`,
    zIndex: Z_INDEX.bar,
    display: 'none',
    alignItems: 'center',
    justifyContent: 'center',
    // 선택(SELECT) 램프를 쓴다 — "이 표면이 지금 대상이다"는 선택 계열 의미이고,
    // 액센트(실행 중)와 색이 겹치면 재생 중 드래그에서 두 상태가 구분되지 않는다.
    border: `${EDGE_WIDTH} dashed ${SELECT.border}`,
    borderRadius: RADIUS.md,
    background: SELECT.soft,
    opacity: '0',
    transition: `${tr('opacity', MOTION.fast)}`,
    // 드롭 대상 위 오버레이는 절대 포인터를 먹으면 안 된다 (파일 헤더)
    pointerEvents: 'none',
    userSelect: 'none',
  });
  root.dataset.testid = 'viewport-drop-hint';
  root.setAttribute('aria-hidden', 'true'); // 순수 시각 어포던스 — 키보드 경로는 카드 Enter다

  const pill = applyType(document.createElement('div'), TYPE.subhead);
  styled(pill, {
    display: 'flex',
    alignItems: 'center',
    gap: SPACE.md,
    padding: `${SPACE.md} ${SPACE.xl}`,
    background: SURFACE.overlay,
    border: `${BORDER_WIDTH.hair} solid ${SELECT.border}`,
    borderRadius: RADIUS.lg,
    color: COLOR.textStrong,
    whiteSpace: 'nowrap',
  });
  const pillIcon = icon('target', ICON.lg);
  const pillText = document.createElement('span');
  pillText.textContent = formatDropHintText(null);
  pill.appendChild(pillIcon);
  pill.appendChild(pillText);
  pill.dataset.testid = 'viewport-drop-hint-label';
  root.appendChild(pill);
  host.appendChild(root);

  let rafId: number | null = null;

  const setActive = (active: boolean, label?: string | null): void => {
    if (rafId !== null && typeof cancelAnimationFrame === 'function') {
      cancelAnimationFrame(rafId);
      rafId = null;
    }
    if (active) {
      pillText.textContent = formatDropHintText(label);
      root.style.display = 'flex';
      // 다음 프레임에 opacity를 올려야 transition이 실제로 돈다(display 전환 직후 무시됨)
      if (typeof requestAnimationFrame === 'function') {
        rafId = requestAnimationFrame(() => {
          rafId = null;
          root.style.opacity = '1';
        });
      } else {
        root.style.opacity = '1';
      }
      return;
    }
    root.style.opacity = '0';
    root.style.display = 'none';
  };

  return {
    el: root,
    setActive,
    dispose: (): void => {
      if (rafId !== null && typeof cancelAnimationFrame === 'function') cancelAnimationFrame(rafId);
      root.remove();
    },
  };
}
