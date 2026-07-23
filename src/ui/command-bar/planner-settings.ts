// ui/command-bar/planner-settings.ts — 플래너 백엔드 설정 (UX_DESIGN §3.1 [⚙])
//
// 커맨드바 우측 ⚙ 버튼 → 작은 모달: 백엔드(규칙 기반 오프라인 데모 | Anthropic API),
// Anthropic 선택 시 API 키(password) + 모델명 입력 + 저장 위치/사용 범위 고지.
// 저장/취소. 실제 영속화(localStorage)와 플래너 재구성은 글루(main.ts)가 deps.set에서
// 수행한다 — 이 모듈은 core/planner를 import하지 않는다(CLAUDE.md §3). 설정 형상은
// 여기서 정의해 노출한다(planner 모듈이 이 타입을 재사용/정렬).
//
// 보안 고지(불변식 §2.9 정신의 투명성): 키는 이 브라우저에만 저장되고 Anthropic 호출에만
// 쓰인다는 점을 명시하고, 공용 PC 경고를 항상 보인다.

import {
  COLOR,
  FONT,
  RADIUS,
  SHADOW,
  SPACE,
  Z_INDEX,
  ensureThemeStyles,
  makeButton,
  styled,
} from '../theme';

// ── 공개 타입 (플래너 백엔드 설정 형상) ─────────────────────────────

export type PlannerBackendKind = 'rule-based' | 'anthropic';

export interface PlannerBackendConfig {
  backend: PlannerBackendKind;
  apiKey: string;
  model: string;
}

export interface PlannerSettingsDeps {
  /** 현재 설정 조회 (모달 열 때 폼 채움) */
  get(): PlannerBackendConfig;
  /** 저장 — 글루가 localStorage 영속화 + 플래너 재구성 */
  set(cfg: PlannerBackendConfig): void;
}

export interface PlannerSettingsHandle {
  /** 프로그램적으로 모달 열기 (⚙ 버튼도 내부에서 이 함수를 호출) */
  open(): void;
  dispose(): void;
}

// ── 상수 (매직넘버 금지 — CLAUDE.md §4) ─────────────────────────────

/** 모델명 기본값 (Anthropic) */
export const DEFAULT_PLANNER_MODEL = 'claude-opus-4-8';

/** 키 저장/사용 범위 고지 (한국어) */
const KEY_NOTICE_KO =
  'API 키는 이 브라우저(localStorage)에만 저장되며 Anthropic API 호출에만 사용됩니다. ' +
  '공용 PC에서는 사용하지 마세요.';

/** radio 그룹 name */
const BACKEND_RADIO_NAME = 'rsw-planner-backend';

// ── 순수 헬퍼 (DOM 비의존 — node 테스트 대상) ───────────────────────

/**
 * 저장 가능 여부: Anthropic 백엔드는 API 키가 비어 있으면 저장 불가.
 * 규칙 기반(오프라인)은 키 없이 항상 저장 가능.
 */
export function canSavePlannerConfig(cfg: PlannerBackendConfig): boolean {
  if (cfg.backend === 'anthropic') return cfg.apiKey.trim().length > 0;
  return true;
}

/**
 * 저장 직전 정규화: 모델명 공백 트림 후 비면 기본값, API 키 트림.
 * (규칙 기반이어도 이후 Anthropic 전환에 대비해 입력값은 보존/정규화만 한다.)
 */
export function normalizePlannerConfig(raw: PlannerBackendConfig): PlannerBackendConfig {
  const model = raw.model.trim();
  return {
    backend: raw.backend,
    apiKey: raw.apiKey.trim(),
    model: model.length > 0 ? model : DEFAULT_PLANNER_MODEL,
  };
}

// ── 내부 DOM 헬퍼 ───────────────────────────────────────────────────

function fieldLabel(text: string): HTMLLabelElement {
  const label = styled(document.createElement('label'), {
    display: 'block',
    color: COLOR.label,
    fontSize: '11px',
    marginBottom: SPACE.xs,
  });
  label.textContent = text;
  return label;
}

// ── 마운트 ──────────────────────────────────────────────────────────

export function mountPlannerSettings(
  host: HTMLElement,
  deps: PlannerSettingsDeps,
): PlannerSettingsHandle {
  ensureThemeStyles();

  // ⚙ 버튼 (커맨드바 우측)
  const gearButton = makeButton('⚙', '플래너 설정 (백엔드·API 키·모델)', 'planner-settings-open');
  host.appendChild(gearButton);

  // 백드롭 + 다이얼로그 (기본 숨김)
  const backdrop = styled(document.createElement('div'), {
    position: 'fixed',
    inset: '0',
    zIndex: Z_INDEX.overlay,
    display: 'none',
    alignItems: 'center',
    justifyContent: 'center',
    background: 'rgba(0, 0, 0, 0.5)',
    fontFamily: FONT.ui,
  });
  backdrop.dataset.testid = 'planner-settings-backdrop';

  const dialog = styled(document.createElement('div'), {
    width: 'min(420px, 92vw)',
    maxHeight: '86vh',
    overflowY: 'auto',
    display: 'flex',
    flexDirection: 'column',
    gap: SPACE.lg,
    background: COLOR.bgPanel,
    border: `1px solid ${COLOR.border}`,
    borderRadius: RADIUS.md,
    boxShadow: SHADOW.panel,
    padding: `${SPACE.lg} ${SPACE.lg}`,
    color: COLOR.text,
    fontSize: '12px',
    boxSizing: 'border-box',
  });
  dialog.className = 'ui-scroll';
  dialog.dataset.testid = 'planner-settings';
  dialog.setAttribute('role', 'dialog');
  dialog.setAttribute('aria-modal', 'true');
  const dlgTitleId = 'rsw-planner-settings-title';
  dialog.setAttribute('aria-labelledby', dlgTitleId);
  // 다이얼로그 위 상호작용이 뷰포트로 새지 않게
  for (const type of ['pointerdown', 'pointermove', 'pointerup', 'wheel']) {
    dialog.addEventListener(type, (e) => {
      e.stopPropagation();
    });
  }

  const heading = styled(document.createElement('div'), {
    color: COLOR.textStrong,
    fontSize: '14px',
    fontWeight: '600',
  });
  heading.id = dlgTitleId;
  heading.textContent = '플래너 설정';
  dialog.appendChild(heading);

  // ── 백엔드 radio ──────────────────────────────────────────────────
  const backendGroup = styled(document.createElement('div'), {
    display: 'flex',
    flexDirection: 'column',
    gap: SPACE.sm,
  });
  backendGroup.setAttribute('role', 'radiogroup');
  backendGroup.setAttribute('aria-label', '플래너 백엔드');
  backendGroup.appendChild(fieldLabel('백엔드'));

  const makeRadio = (
    value: PlannerBackendKind,
    text: string,
    testId: string,
  ): { wrap: HTMLLabelElement; input: HTMLInputElement } => {
    const wrap = styled(document.createElement('label'), {
      display: 'flex',
      alignItems: 'center',
      gap: SPACE.sm,
      cursor: 'pointer',
      color: COLOR.text,
    });
    const input = document.createElement('input');
    input.type = 'radio';
    input.name = BACKEND_RADIO_NAME;
    input.value = value;
    input.dataset.testid = testId;
    input.style.accentColor = COLOR.accent;
    const span = document.createElement('span');
    span.textContent = text;
    wrap.appendChild(input);
    wrap.appendChild(span);
    return { wrap, input };
  };

  const ruleRadio = makeRadio('rule-based', '규칙 기반 (오프라인 데모)', 'planner-backend-rule');
  const anthropicRadio = makeRadio('anthropic', 'Anthropic API', 'planner-backend-anthropic');
  backendGroup.appendChild(ruleRadio.wrap);
  backendGroup.appendChild(anthropicRadio.wrap);
  dialog.appendChild(backendGroup);

  // ── Anthropic 전용 섹션 (키 + 모델 + 고지) ────────────────────────
  const anthropicSection = styled(document.createElement('div'), {
    display: 'none',
    flexDirection: 'column',
    gap: SPACE.md,
    paddingTop: SPACE.sm,
    borderTop: `1px solid ${COLOR.borderSoft}`,
  });
  anthropicSection.dataset.testid = 'planner-anthropic-section';

  const keyField = document.createElement('div');
  const keyLabel = fieldLabel('API 키');
  const keyInput = document.createElement('input');
  keyInput.type = 'password';
  keyInput.className = 'ui-input';
  keyInput.dataset.testid = 'planner-api-key';
  keyInput.placeholder = 'sk-ant-…';
  keyInput.autocomplete = 'off';
  keyInput.setAttribute('aria-label', 'Anthropic API 키');
  styled(keyInput, { width: '100%', boxSizing: 'border-box' });
  keyLabel.htmlFor = 'rsw-planner-api-key';
  keyInput.id = 'rsw-planner-api-key';
  keyField.appendChild(keyLabel);
  keyField.appendChild(keyInput);

  const modelField = document.createElement('div');
  const modelLabel = fieldLabel('모델');
  const modelInput = document.createElement('input');
  modelInput.type = 'text';
  modelInput.className = 'ui-input';
  modelInput.dataset.testid = 'planner-model';
  modelInput.placeholder = DEFAULT_PLANNER_MODEL;
  modelInput.setAttribute('aria-label', '모델명');
  styled(modelInput, { width: '100%', boxSizing: 'border-box' });
  modelLabel.htmlFor = 'rsw-planner-model';
  modelInput.id = 'rsw-planner-model';
  modelField.appendChild(modelLabel);
  modelField.appendChild(modelInput);

  const notice = styled(document.createElement('p'), {
    margin: '0',
    padding: `${SPACE.sm} ${SPACE.md}`,
    background: COLOR.infoSoft,
    border: `1px solid ${COLOR.border}`,
    borderRadius: RADIUS.sm,
    color: COLOR.muted,
    fontSize: '11px',
    lineHeight: '1.6',
  });
  notice.dataset.testid = 'planner-key-notice';
  notice.textContent = KEY_NOTICE_KO;

  anthropicSection.appendChild(keyField);
  anthropicSection.appendChild(modelField);
  anthropicSection.appendChild(notice);
  dialog.appendChild(anthropicSection);

  // 검증 오류(인라인)
  const errorText = styled(document.createElement('div'), {
    display: 'none',
    color: COLOR.errorText,
    fontSize: '11px',
    lineHeight: '1.5',
  });
  errorText.dataset.testid = 'planner-settings-error';
  errorText.setAttribute('role', 'alert');
  dialog.appendChild(errorText);

  // ── 푸터 (저장/취소) ──────────────────────────────────────────────
  const footer = styled(document.createElement('div'), {
    display: 'flex',
    justifyContent: 'flex-end',
    gap: SPACE.sm,
  });
  const cancelButton = makeButton('취소', '변경 취소', 'planner-settings-cancel', 'ghost');
  const saveButton = makeButton('저장', '설정 저장', 'planner-settings-save', 'accent');
  footer.appendChild(cancelButton);
  footer.appendChild(saveButton);
  dialog.appendChild(footer);

  backdrop.appendChild(dialog);
  document.body.appendChild(backdrop);

  // ── 상태 · 배선 ───────────────────────────────────────────────────

  const currentBackend = (): PlannerBackendKind =>
    anthropicRadio.input.checked ? 'anthropic' : 'rule-based';

  /** 백엔드 선택에 따라 Anthropic 섹션 표시/숨김 + 오류 초기화 */
  const syncBackendVisibility = (): void => {
    anthropicSection.style.display = currentBackend() === 'anthropic' ? 'flex' : 'none';
    errorText.style.display = 'none';
  };
  ruleRadio.input.addEventListener('change', syncBackendVisibility);
  anthropicRadio.input.addEventListener('change', syncBackendVisibility);

  const close = (): void => {
    backdrop.style.display = 'none';
    window.removeEventListener('keydown', onKeyDown, true);
  };

  function onKeyDown(e: KeyboardEvent): void {
    if (backdrop.style.display === 'none') return;
    if (e.key === 'Escape') {
      e.preventDefault();
      close();
    }
  }

  const open = (): void => {
    const cfg = deps.get();
    ruleRadio.input.checked = cfg.backend !== 'anthropic';
    anthropicRadio.input.checked = cfg.backend === 'anthropic';
    keyInput.value = cfg.apiKey;
    modelInput.value = cfg.model;
    errorText.style.display = 'none';
    syncBackendVisibility();
    backdrop.style.display = 'flex';
    window.addEventListener('keydown', onKeyDown, true);
    // 첫 포커스: 현재 백엔드 radio
    (cfg.backend === 'anthropic' ? anthropicRadio.input : ruleRadio.input).focus();
  };

  gearButton.addEventListener('click', open);
  cancelButton.addEventListener('click', close);
  // 백드롭(다이얼로그 바깥) 클릭 = 취소
  backdrop.addEventListener('click', (e) => {
    if (e.target === backdrop) close();
  });

  saveButton.addEventListener('click', () => {
    const draft: PlannerBackendConfig = {
      backend: currentBackend(),
      apiKey: keyInput.value,
      model: modelInput.value,
    };
    if (!canSavePlannerConfig(draft)) {
      errorText.textContent = 'Anthropic API를 선택하면 API 키를 입력해야 합니다.';
      errorText.style.display = 'block';
      keyInput.focus();
      return;
    }
    deps.set(normalizePlannerConfig(draft));
    close();
  });

  return {
    open,
    dispose: (): void => {
      window.removeEventListener('keydown', onKeyDown, true);
      gearButton.remove();
      backdrop.remove();
    },
  };
}
