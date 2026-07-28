// ui/command-bar/nl-input.ts — 자연어 입력 + 생성 트리거 (UX_DESIGN §3.1, §4.1 Flow 1, §7)
//
// 커맨드바 **행 B 명령 슬롯**(`shell.rowBCommand`)에 들어가는 플렉스 행:
// 말풍선 아이콘 + 입력 + 교체/이어서 세그먼트 토글 + 생성 버튼.
// 생성 요청만 발행하고, 플래너 실행·결과 라우팅은 글루(main.ts)가 deps.generate에서
// 수행한다 — 이 모듈은 core/planner를 import하지 않는다(CLAUDE.md §3).
//
// 이 입력은 **이 제품의 헤드라인 기능**이다. 구 배치에서는 44px 한 줄의 중앙 슬롯에
// 재생 컨트롤과 함께 들어가 1280px에서 재생 버튼 뒤로 사라졌다(UX_AUDIT C-2).
// 행 B에서 남는 폭을 전부 먹되(flex-grow) 기준 폭 220px를 확보한다.
//
// 불변식 §2.9(human-in-the-loop): "생성"은 실행이 아니다. 이 입력은 초안 생성을 요청할
// 뿐이며, 시뮬레이터 자동 재생을 하지 않는다. 검증·그래프 로드·▶재생은 글루/사용자 몫.
//
// 상태(UX_DESIGN §7):
//   generating → 입력 비활성 + 버튼에 활동 표시(펄스 점) + "생성 중…"
//   success    → 잠깐 초록 체크 후 idle 복귀
//   clarify    → idle과 동일하게 재활성(명확화 카드는 별도 모듈이 띄운다)
//   error      → 입력에 충돌 램프 테두리 + detail을 title로(토스트는 글루가 띄운다)

import { icon, makeIconButton } from '../icons';
import {
  COLLISION,
  COLOR,
  ICON,
  SPACE,
  TYPE,
  applyType,
  ensureThemeStyles,
  makeButton,
  styled,
} from '../theme';
import {
  COMMAND_BAR_PRIORITY,
  MIN_NL_INPUT_WIDTH_PX,
  setCommandBarPriority,
} from './scene-controls';

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

/** 생성 버튼 라벨 (idle/error) */
const GENERATE_LABEL = '생성';

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
    flexGrow: '1',
    flexShrink: '1',
    flexBasis: `${MIN_NL_INPUT_WIDTH_PX}px`,
    minWidth: '0',
    overflow: 'hidden',
    color: COLOR.text,
    boxSizing: 'border-box',
    pointerEvents: 'auto',
  });
  applyType(row, TYPE.body);
  row.dataset.testid = 'nl-input';
  setCommandBarPriority(row, COMMAND_BAR_PRIORITY.command);
  // 입력 위 상호작용이 뷰포트 orbit으로 새지 않게 (커맨드바 셸 규약과 중복이지만 무해)
  for (const type of ['pointerdown', 'pointermove', 'pointerup', 'wheel']) {
    row.addEventListener(type, (e) => {
      e.stopPropagation();
    });
  }

  // 말풍선 아이콘 (장식 — SR 무시). 컬러 이모지는 currentColor를 못 받아 hover/disabled
  // 에서 텍스트만 밝아졌다 (UX_AUDIT C-13).
  const speechIcon = icon('speech', ICON.md);
  speechIcon.style.color = COLOR.muted;

  // 자연어 입력 (남는 폭 전부)
  const input = styled(document.createElement('input'), {
    flexGrow: '1',
    flexShrink: '1',
    minWidth: '0',
  });
  input.type = 'text';
  input.className = 'ui-input';
  input.dataset.testid = 'nl-text';
  input.placeholder = '로봇에게 시킬 일을 자연어로…';
  input.title = '자연어 명령 (Enter = 생성)';
  input.setAttribute('aria-label', '로봇에게 시킬 일을 자연어로 입력');

  // 교체/이어서 세그먼트 토글
  let mode: GenerateMode = 'replace';
  const modeGroup = styled(document.createElement('div'), {
    display: 'inline-flex',
    flexShrink: '0',
    gap: '0',
  });
  modeGroup.setAttribute('role', 'group');
  modeGroup.setAttribute('aria-label', '생성 모드');
  modeGroup.dataset.testid = 'nl-mode-group';
  setCommandBarPriority(modeGroup, COMMAND_BAR_PRIORITY.misc);

  const makeModeButton = (m: GenerateMode): HTMLButtonElement => {
    const b = makeButton(MODE_LABEL_KO[m], MODE_TITLE_KO[m], `nl-mode-${m}`);
    b.style.borderRadius = m === 'replace' ? '4px 0 0 4px' : '0 4px 4px 0';
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

  // 생성 버튼 — 행 B의 주요 액션이므로 액센트 **면**(primary)이다. 구 `--accent`는
  // 보더만 칠해 이웃 버튼과 시각 무게가 같았다 (UX_AUDIT C-14).
  const generateBtn = makeIconButton(
    'wand',
    GENERATE_LABEL,
    '자연어 명령으로 제어 플로우 생성 (Enter)',
    'nl-generate',
    'primary',
  );
  generateBtn.style.flexShrink = '0';
  const generateLabel = generateBtn.querySelector<HTMLElement>('.ui-btn__label');
  const busyDot = document.createElement('span');
  busyDot.className = 'ui-dot ui-dot--playing';
  busyDot.setAttribute('aria-hidden', 'true');
  busyDot.style.display = 'none';
  generateBtn.insertBefore(busyDot, generateBtn.firstChild);

  /** 라벨 텍스트만 교체한다 (아이콘/활동 점 구조는 유지 — 아이콘 전용 모드 호환) */
  const setGenerateLabel = (text: string): void => {
    if (generateLabel !== null) generateLabel.textContent = text;
  };

  // ── 상태 렌더 ─────────────────────────────────────────────────────

  let flashTimer: ReturnType<typeof setTimeout> | null = null;
  const clearFlash = (): void => {
    if (flashTimer !== null) {
      clearTimeout(flashTimer);
      flashTimer = null;
    }
  };

  /** 입력 오류 테두리 on/off — 테두리색만 인라인으로 덮고 나머진 클래스 유지 */
  const setErrorRing = (on: boolean): void => {
    input.style.borderColor = on ? COLLISION.base : '';
    input.style.boxShadow = on ? `0 0 0 1px ${COLLISION.base}` : '';
  };

  const render = (state: NlInputState, detail?: string): void => {
    clearFlash();
    input.title = detail ?? '자연어 명령 (Enter = 생성)';

    switch (state) {
      case 'generating':
        input.disabled = true;
        generateBtn.disabled = true;
        busyDot.style.display = 'inline-block';
        setGenerateLabel('생성 중…');
        generateBtn.setAttribute('aria-busy', 'true');
        setErrorRing(false);
        break;
      case 'success':
        input.disabled = false;
        generateBtn.disabled = false;
        busyDot.style.display = 'none';
        setGenerateLabel('완료');
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
        setGenerateLabel(GENERATE_LABEL);
        generateBtn.removeAttribute('aria-busy');
        setErrorRing(true);
        break;
      case 'clarify':
      case 'idle':
      default:
        input.disabled = false;
        generateBtn.disabled = false;
        busyDot.style.display = 'none';
        setGenerateLabel(GENERATE_LABEL);
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

  row.appendChild(speechIcon);
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
