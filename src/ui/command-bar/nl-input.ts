// ui/command-bar/nl-input.ts — 자연어 입력 + 생성 트리거 (UX_DESIGN §3.1, §4.1 Flow 1, §7)
//
// 커맨드바 중앙-좌 슬롯에 들어가는 플렉스 행: 🗣 입력 + 교체/이어서 세그먼트 토글 +
// ✨ 생성 버튼. 생성 요청만 발행하고, 플래너 실행·결과 라우팅은 글루(main.ts)가
// deps.generate에서 수행한다 — 이 모듈은 core/planner를 import하지 않는다(CLAUDE.md §3).
//
// 불변식 §2.9(human-in-the-loop): "생성"은 실행이 아니다. 이 입력은 초안 생성을 요청할
// 뿐이며, 시뮬레이터 자동 재생을 하지 않는다. 검증·그래프 로드·▶Play는 글루/사용자 몫.
//
// 상태(UX_DESIGN §7):
//   generating → 입력 비활성 + 버튼에 활동 표시(펄스 점) + "생성 중…"
//   success    → 잠깐 초록 체크 후 idle 복귀
//   clarify    → idle과 동일하게 재활성(명확화 카드는 별도 모듈이 띄운다)
//   error      → 입력에 빨강 테두리 + detail을 title로(토스트는 글루가 띄운다)

import {
  COLOR,
  FONT,
  SPACE,
  ensureThemeStyles,
  makeButton,
  styled,
} from '../theme';

// ── 공개 타입 ───────────────────────────────────────────────────────

/** 생성 모드: 새 플로우로 교체 | 기존 뒤에 이어서 추가 (UX_DESIGN §3.1, 기본 교체) */
export type GenerateMode = 'replace' | 'append';

export type NlInputState = 'idle' | 'generating' | 'success' | 'clarify' | 'error';

export interface NlInputDeps {
  /** 자연어 명령 생성 요청 — 플래너 실행·검증·결과 라우팅은 글루가 담당 */
  generate(nl: string, mode: GenerateMode): Promise<void>;
  /** 다른 생성/전환이 진행 중인지 — 중복 트리거 가드 */
  isBusy(): boolean;
}

export interface NlInputHandle {
  readonly el: HTMLElement;
  /** 외부(글루)에서 상태 전이 표시 — detail은 error/clarify 사유(선택) */
  setState(state: NlInputState, detail?: string): void;
  dispose(): void;
}

// ── 상수 (매직넘버 금지 — CLAUDE.md §4) ─────────────────────────────

/** 성공 체크 표시 유지 시간(ms) — 이후 idle 시각으로 복귀 */
const SUCCESS_FLASH_MS = 1400;

/** 모드 → 한국어 라벨 (세그먼트 토글) */
const MODE_LABEL_KO: Readonly<Record<GenerateMode, string>> = {
  replace: '교체',
  append: '이어서',
};

const MODE_TITLE_KO: Readonly<Record<GenerateMode, string>> = {
  replace: '새 플로우로 교체',
  append: '기존 플로우 뒤에 이어서 추가',
};

// ── 순수 헬퍼 (DOM 비의존 — node 테스트 대상) ───────────────────────

/**
 * 생성 가능 여부: 진행 중이 아니고(자동 재트리거 방지), 입력이 공백만은 아니어야 한다.
 * 빈 입력으로는 플래너를 호출하지 않는다(불필요한 요청·비용 방지).
 */
export function canGenerate(text: string, busy: boolean): boolean {
  return !busy && text.trim().length > 0;
}

/** 모드 → 한국어 라벨 (세그먼트 버튼 텍스트) */
export function modeLabelKo(mode: GenerateMode): string {
  return MODE_LABEL_KO[mode];
}

// ── 마운트 ──────────────────────────────────────────────────────────

export function mountNlInput(host: HTMLElement, deps: NlInputDeps): NlInputHandle {
  ensureThemeStyles();

  const row = styled(document.createElement('div'), {
    display: 'flex',
    alignItems: 'center',
    gap: SPACE.sm,
    flex: '1',
    minWidth: '0',
    color: COLOR.text,
    fontFamily: FONT.ui,
    fontSize: '12px',
    boxSizing: 'border-box',
    pointerEvents: 'auto',
  });
  row.dataset.testid = 'nl-input';
  // 입력 위 상호작용이 뷰포트 orbit으로 새지 않게 (커맨드바 셸 규약과 중복이지만 무해)
  for (const type of ['pointerdown', 'pointermove', 'pointerup', 'wheel']) {
    row.addEventListener(type, (e) => {
      e.stopPropagation();
    });
  }

  // 🗣 아이콘 (장식 — SR 무시)
  const icon = styled(document.createElement('span'), { flexShrink: '0', opacity: '0.85' });
  icon.textContent = '🗣';
  icon.setAttribute('aria-hidden', 'true');

  // 자연어 입력 (flex-grow)
  const input = document.createElement('input');
  input.type = 'text';
  input.className = 'ui-input';
  input.dataset.testid = 'nl-text';
  input.placeholder = '로봇에게 시킬 일을 자연어로…';
  input.setAttribute('aria-label', '로봇에게 시킬 일을 자연어로 입력');
  styled(input, { flex: '1', minWidth: '80px' });

  // 교체/이어서 세그먼트 토글
  let mode: GenerateMode = 'replace';
  const modeGroup = styled(document.createElement('div'), {
    display: 'inline-flex',
    flexShrink: '0',
    gap: '0',
  });
  modeGroup.setAttribute('role', 'group');
  modeGroup.setAttribute('aria-label', '생성 모드');

  const makeModeButton = (m: GenerateMode): HTMLButtonElement => {
    const b = makeButton(MODE_LABEL_KO[m], MODE_TITLE_KO[m], `nl-mode-${m}`);
    b.style.borderRadius = m === 'replace' ? '5px 0 0 5px' : '0 5px 5px 0';
    return b;
  };
  const replaceBtn = makeModeButton('replace');
  const appendBtn = makeModeButton('append');
  // 인접 버튼 사이 이중 테두리 제거
  appendBtn.style.marginLeft = '-1px';
  modeGroup.appendChild(replaceBtn);
  modeGroup.appendChild(appendBtn);

  const paintMode = (): void => {
    replaceBtn.classList.toggle('ui-btn--active', mode === 'replace');
    appendBtn.classList.toggle('ui-btn--active', mode === 'append');
    replaceBtn.setAttribute('aria-pressed', String(mode === 'replace'));
    appendBtn.setAttribute('aria-pressed', String(mode === 'append'));
  };
  replaceBtn.addEventListener('click', () => {
    mode = 'replace';
    paintMode();
  });
  appendBtn.addEventListener('click', () => {
    mode = 'append';
    paintMode();
  });
  paintMode();

  // ✨ 생성 버튼 (액센트) — 내부에 활동 점 + 라벨 스팬
  const generateBtn = makeButton('', '자연어 명령으로 제어 플로우 생성', 'nl-generate', 'accent');
  generateBtn.textContent = '';
  const busyDot = document.createElement('span');
  busyDot.className = 'ui-dot ui-dot--playing';
  busyDot.setAttribute('aria-hidden', 'true');
  busyDot.style.display = 'none';
  busyDot.style.marginRight = '4px';
  const generateLabel = document.createElement('span');
  generateLabel.textContent = '✨ 생성';
  generateBtn.appendChild(busyDot);
  generateBtn.appendChild(generateLabel);

  // ── 상태 렌더 ─────────────────────────────────────────────────────

  let flashTimer: ReturnType<typeof setTimeout> | null = null;
  const clearFlash = (): void => {
    if (flashTimer !== null) {
      clearTimeout(flashTimer);
      flashTimer = null;
    }
  };

  /** 입력 빨강 테두리 on/off (error 상태) — 테두리색만 인라인으로 덮고 나머진 클래스 유지 */
  const setErrorRing = (on: boolean): void => {
    input.style.borderColor = on ? COLOR.error : '';
    input.style.boxShadow = on ? `0 0 0 1px ${COLOR.error}` : '';
  };

  const render = (state: NlInputState, detail?: string): void => {
    clearFlash();
    input.title = detail ?? '';

    switch (state) {
      case 'generating':
        input.disabled = true;
        generateBtn.disabled = true;
        busyDot.style.display = 'inline-block';
        generateLabel.textContent = '생성 중…';
        generateBtn.setAttribute('aria-busy', 'true');
        setErrorRing(false);
        break;
      case 'success':
        input.disabled = false;
        generateBtn.disabled = false;
        busyDot.style.display = 'none';
        generateLabel.textContent = '✓ 완료';
        generateBtn.style.color = COLOR.successText;
        generateBtn.removeAttribute('aria-busy');
        setErrorRing(false);
        flashTimer = setTimeout(() => {
          flashTimer = null;
          render('idle');
        }, SUCCESS_FLASH_MS);
        break;
      case 'error':
        input.disabled = false;
        generateBtn.disabled = false;
        busyDot.style.display = 'none';
        generateLabel.textContent = '✨ 생성';
        generateBtn.style.color = '';
        generateBtn.removeAttribute('aria-busy');
        setErrorRing(true);
        break;
      case 'clarify':
      case 'idle':
      default:
        input.disabled = false;
        generateBtn.disabled = false;
        busyDot.style.display = 'none';
        generateLabel.textContent = '✨ 생성';
        generateBtn.style.color = '';
        generateBtn.removeAttribute('aria-busy');
        setErrorRing(false);
        break;
    }
  };

  // ── 동작 배선 ─────────────────────────────────────────────────────

  const runGenerate = (): void => {
    if (!canGenerate(input.value, deps.isBusy())) return;
    setErrorRing(false); // 재시도 시 이전 오류 표시 해제
    void deps.generate(input.value.trim(), mode);
  };

  generateBtn.addEventListener('click', runGenerate);
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.isComposing) {
      e.preventDefault();
      runGenerate();
    }
  });

  row.appendChild(icon);
  row.appendChild(input);
  row.appendChild(modeGroup);
  row.appendChild(generateBtn);
  host.appendChild(row);

  return {
    el: row,
    setState: render,
    dispose: (): void => {
      clearFlash();
      row.remove();
    },
  };
}
