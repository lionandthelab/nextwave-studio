// ui/command-bar/scene-controls.ts — 씬 관리 컨트롤 + 커맨드바 셸 (UX_DESIGN §3.1 부분집합)
//
// 커맨드바 좌측 섹션: 앱 타이틀 · 씬 프리셋 <select> · 📂 업로드 · 💾 저장.
// 하나의 응집된 상단 커맨드바(셸)를 이 모듈이 제공하고, 글루(main.ts)가
// 좌(씬 컨트롤)/중앙(재생 컨트롤)/우(JSON 뷰어) 슬롯에 각 모듈을 마운트한다.
//
// 계층 규칙 (CLAUDE.md §3): core를 import하지 않는다. 씬 전환·검증·저장 대상 spec은
// 글루가 SceneControlsApi 콜백으로 주입한다 — 이 모듈은 파일 I/O(파싱·다운로드)와
// 표시 상태(select 동기화·오류 토스트)만 소유한다.
//
// ── 업로드 JSON 형식 규약 (DATA_MODEL §5/§6) ─────────────────────────
// (a) SceneSpec 단독:            { "name": …, "version": 1, "entities": […], … }
// (b) 씬+시퀀스 봉투(envelope):  { "scene": <SceneSpec>, "sequence": <ControlSequence> }
// 봉투 해석(unwrap)과 스키마 검증은 글루(main.ts) 몫이다 — 이 모듈은 JSON.parse까지만
// 수행하고 파싱 결과를 그대로 넘긴다. 검증 실패는 콘솔 패널 + 인라인 오류로 표시된다.

import { appLog } from '../dock/console-panel';
import {
  COLOR,
  FONT,
  LAYOUT,
  RADIUS,
  SHADOW,
  SPACE,
  Z_INDEX,
  ensureThemeStyles,
  makeButton,
  styled,
} from '../theme';
import type { SceneSpec } from '../../schema';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4, 시각 토큰은 ui/theme.ts) ────

/** 커맨드바 아래 여백(px) — 오류 토스트가 바를 가리지 않는 위치 */
const BELOW_BAR_TOP_PX = LAYOUT.belowBarTopPx;
/** 오류 토스트 자동 숨김 시간(ms) */
const ERROR_AUTO_HIDE_MS = 8000;
/** 오류 토스트 최대 표시 길이 — 초과분은 말줄임 (전체 사유는 Console 탭) */
const ERROR_MAX_CHARS = 400;
/** Blob URL 해제 지연(ms) — 다운로드 클릭이 URL을 소비할 시간 여유 */
const BLOB_URL_REVOKE_DELAY_MS = 1000;
/** 업로드 씬(비프리셋) 표시용 select 옵션 값 — 프리셋 이름과 충돌하지 않는 예약값 */
const UPLOADED_OPTION_VALUE = '__uploaded';
/** 활성 씬 없음 표시용 select 옵션 값 — build 단계 실패로 이전 씬까지 잃었을 때 */
const NO_SCENE_OPTION_VALUE = '__none';

/**
 * 프리셋 씬 한국어 라벨 (라벨 미등록 이름은 이름 그대로 표시).
 * 새 프리셋은 main.ts의 SCENE_REGISTRY에 데이터로 추가된다 — 여기 라벨이 없어도
 * select에는 자동으로 나타난다 (새 씬 = 새 데이터, CLAUDE.md §2.5).
 */
const SCENE_LABELS_KO: Readonly<Record<string, string>> = {
  'falling-boxes': '낙하 박스',
  'arm-and-boxes': '로봇팔·박스',
  'pick-and-place': '픽앤플레이스',
  'obstacle-avoidance': '장애물 회피',
  'collision-testbed': '충돌 테스트베드',
};

// ── 공개 타입 ───────────────────────────────────────────────────────

/** 씬 전환 결과 — 글루(main.ts)의 loadScene 결과를 그대로 전달받는다 */
export interface SceneSwitchResult {
  readonly ok: boolean;
  /** 실패 시 사람이 읽을 수 있는 한국어 오류 목록 */
  readonly errors?: readonly string[];
  /**
   * 실패가 build 단계에서 나 "이전 활성 씬까지 해제된" 경우 true (글루의 loadScene은
   * 검증 실패 시 이전 씬을 보존하지만, build 실패 시에는 이미 teardown이 끝난 뒤다).
   * true면 select를 이전 씬 이름으로 되돌리지 않고 '씬 없음' 상태를 표시한다.
   */
  readonly sceneLost?: boolean;
}

/** 글루(main.ts)가 씬 라이프사이클 위에서 구현해 주입하는 동작 표면 */
export interface SceneControlsApi {
  /** 프리셋 씬으로 전환 (검증 → 이전 씬 해제 → 새로 빌드) */
  switchToPreset(name: string): Promise<SceneSwitchResult>;
  /** 업로드 JSON으로 전환 — payload는 SceneSpec 단독 또는 {scene, sequence} 봉투 */
  switchToUpload(payload: unknown, fileName: string): Promise<SceneSwitchResult>;
  /** 현재 활성 씬의 SceneSpec (💾 저장용 — 활성 씬이 없으면 null) */
  currentSpec(): SceneSpec | null;
}

export interface SceneControlsHandle {
  readonly el: HTMLElement;
  /** select 표시 동기화 — 프리셋 이름 또는 null(업로드 씬). 부트 초기화에 사용. */
  setCurrent(presetName: string | null): void;
  /** 인라인 오류 토스트 표시 (한국어) — 커맨드바 아래에 나타나고 자동/클릭으로 닫힌다 */
  showError(message: string): void;
  dispose(): void;
}

/** 커맨드바 셸 — 좌/중앙/우 슬롯을 가진 하나의 응집된 상단 바 (UX_DESIGN §3.1) */
export interface CommandBarShell {
  readonly el: HTMLElement;
  readonly left: HTMLElement;
  readonly center: HTMLElement;
  readonly right: HTMLElement;
  dispose(): void;
}

// ── 내부 헬퍼 ───────────────────────────────────────────────────────

/** 프리셋 이름 → select 표시 라벨: "한국어 라벨 (이름)" 또는 이름 그대로 */
function sceneOptionLabel(name: string): string {
  const ko = SCENE_LABELS_KO[name];
  return ko ? `${ko} (${name})` : name;
}

/** 씬 이름 → 안전한 다운로드 파일명 조각 (경로/예약 문자 제거) */
function sanitizeFileName(name: string): string {
  const cleaned = name.replace(/[\\/:*?"<>|\s]+/g, '-').replace(/^-+|-+$/g, '');
  return cleaned.length > 0 ? cleaned : 'scene';
}

// ── 커맨드바 셸 ─────────────────────────────────────────────────────

/**
 * 응집된 상단 커맨드바(고정 오버레이)를 host에 마운트하고 좌/중앙/우 슬롯을 돌려준다.
 * 바 위 포인터/휠 상호작용은 stopPropagation으로 흡수한다 — 뷰포트 orbit으로 새지 않게
 * (dock/joint-panel과 동일 규약).
 */
export function mountCommandBarShell(host: HTMLElement): CommandBarShell {
  ensureThemeStyles();
  const bar = styled(document.createElement('div'), {
    position: 'fixed',
    top: '0',
    left: '0',
    right: '0',
    zIndex: Z_INDEX.bar,
    height: `${LAYOUT.barHeightPx}px`,
    display: 'flex',
    alignItems: 'center',
    gap: '14px',
    padding: `0 ${SPACE.lg}`,
    background: COLOR.bgBar,
    borderBottom: `1px solid ${COLOR.border}`,
    boxShadow: SHADOW.bar,
    color: COLOR.text,
    fontFamily: FONT.ui,
    fontSize: '12px',
    boxSizing: 'border-box',
    pointerEvents: 'auto',
  });
  bar.dataset.testid = 'command-bar';
  for (const type of ['pointerdown', 'pointermove', 'pointerup', 'wheel', 'contextmenu']) {
    bar.addEventListener(type, (e) => {
      e.stopPropagation();
    });
  }

  const left = styled(document.createElement('div'), {
    display: 'flex',
    alignItems: 'center',
    gap: SPACE.md,
    flexShrink: '0',
  });
  const center = styled(document.createElement('div'), {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: SPACE.sm,
    flex: '1',
    minWidth: '0',
  });
  const right = styled(document.createElement('div'), {
    display: 'flex',
    alignItems: 'center',
    gap: SPACE.sm,
    flexShrink: '0',
  });

  bar.appendChild(left);
  bar.appendChild(center);
  bar.appendChild(right);
  host.appendChild(bar);

  return {
    el: bar,
    left,
    center,
    right,
    dispose: (): void => {
      bar.remove();
    },
  };
}

// ── 씬 컨트롤 (커맨드바 좌측 섹션) ──────────────────────────────────

export function mountSceneControls(
  host: HTMLElement,
  presetNames: readonly string[],
  api: SceneControlsApi,
): SceneControlsHandle {
  ensureThemeStyles();
  const section = styled(document.createElement('div'), {
    display: 'flex',
    alignItems: 'center',
    gap: SPACE.md,
  });
  section.dataset.testid = 'scene-controls';

  // 앱 타이틀 (액센트 마커 + 이름 — 시각 위계의 시작점)
  const title = styled(document.createElement('strong'), {
    display: 'inline-flex',
    alignItems: 'center',
    gap: SPACE.sm,
    color: COLOR.textStrong,
    fontSize: '13px',
    letterSpacing: '0.2px',
    whiteSpace: 'nowrap',
  });
  const titleMark = styled(document.createElement('span'), {
    width: '8px',
    height: '8px',
    borderRadius: '2px',
    background: COLOR.accent,
    flexShrink: '0',
  });
  titleMark.setAttribute('aria-hidden', 'true');
  title.appendChild(titleMark);
  title.appendChild(document.createTextNode('robot-sim-web'));
  section.appendChild(title);

  // 씬 프리셋 select
  const select = styled(document.createElement('select'), { maxWidth: '220px' });
  select.className = 'ui-select';
  select.dataset.testid = 'scene-select';
  select.title = '씬 프리셋 선택';
  select.setAttribute('aria-label', '씬 프리셋 선택');
  for (const name of presetNames) {
    const option = document.createElement('option');
    option.value = name;
    option.textContent = sceneOptionLabel(name);
    select.appendChild(option);
  }
  // 업로드 씬(비프리셋) 표시용 옵션 — 활성일 때만 보인다
  const uploadedOption = document.createElement('option');
  uploadedOption.value = UPLOADED_OPTION_VALUE;
  uploadedOption.textContent = '업로드 씬';
  uploadedOption.disabled = true;
  uploadedOption.hidden = true;
  select.appendChild(uploadedOption);
  // 활성 씬 없음 표시용 옵션 — build 실패로 이전 씬까지 잃었을 때만 보인다
  const noneOption = document.createElement('option');
  noneOption.value = NO_SCENE_OPTION_VALUE;
  noneOption.textContent = '씬 없음';
  noneOption.disabled = true;
  noneOption.hidden = true;
  select.appendChild(noneOption);
  section.appendChild(select);

  // 📂 업로드 (숨은 파일 입력 + 버튼)
  const fileInput = document.createElement('input');
  fileInput.type = 'file';
  fileInput.accept = '.json,application/json';
  fileInput.style.display = 'none';
  fileInput.dataset.testid = 'scene-upload-input';
  section.appendChild(fileInput);
  const uploadButton = makeButton(
    '📂 업로드',
    '씬 JSON 업로드 — SceneSpec 단독 또는 {scene, sequence} 봉투',
    'scene-upload',
  );
  section.appendChild(uploadButton);

  // 💾 저장 (현재 SceneSpec 다운로드)
  const saveButton = makeButton('💾 저장', '현재 SceneSpec JSON 다운로드', 'scene-save');
  section.appendChild(saveButton);

  // 인라인 오류 토스트 (커맨드바 아래, 클릭/자동 닫힘)
  const errorToast = styled(document.createElement('div'), {
    position: 'fixed',
    top: `${BELOW_BAR_TOP_PX}px`,
    left: '10px',
    zIndex: Z_INDEX.toast,
    maxWidth: 'min(560px, 80vw)',
    background: COLOR.errorSurface,
    border: `1px solid ${COLOR.errorBorder}`,
    borderLeft: `3px solid ${COLOR.error}`,
    borderRadius: RADIUS.md,
    boxShadow: SHADOW.panel,
    padding: `${SPACE.md} ${SPACE.lg}`,
    color: COLOR.errorText,
    fontFamily: FONT.mono,
    fontSize: '12px',
    lineHeight: '1.6',
    whiteSpace: 'pre-wrap',
    cursor: 'pointer',
    display: 'none',
  });
  errorToast.dataset.testid = 'scene-error';
  errorToast.title = '클릭하여 닫기';
  errorToast.setAttribute('role', 'alert');
  document.body.appendChild(errorToast);

  let errorHideTimer: ReturnType<typeof setTimeout> | null = null;
  const hideError = (): void => {
    errorToast.style.display = 'none';
    if (errorHideTimer !== null) {
      clearTimeout(errorHideTimer);
      errorHideTimer = null;
    }
  };
  errorToast.addEventListener('click', hideError);

  const showError = (message: string): void => {
    const shown =
      message.length > ERROR_MAX_CHARS
        ? `${message.slice(0, ERROR_MAX_CHARS)}… (전체 사유는 Console 탭)`
        : message;
    errorToast.textContent = shown;
    errorToast.style.display = 'block';
    if (errorHideTimer !== null) clearTimeout(errorHideTimer);
    errorHideTimer = setTimeout(hideError, ERROR_AUTO_HIDE_MS);
  };

  // ── 표시 상태 ─────────────────────────────────────────────────────

  /** 마지막으로 "적용에 성공한" select 값 — 전환 실패 시 이 값으로 되돌린다 */
  let lastAppliedValue: string = presetNames[0] ?? UPLOADED_OPTION_VALUE;

  const setCurrent = (presetName: string | null): void => {
    noneOption.hidden = true; // 씬이 다시 활성화됨 — '씬 없음' 상태 해제
    if (presetName === null) {
      uploadedOption.hidden = false;
      select.value = UPLOADED_OPTION_VALUE;
      lastAppliedValue = UPLOADED_OPTION_VALUE;
    } else {
      uploadedOption.hidden = true;
      select.value = presetName;
      lastAppliedValue = presetName;
    }
  };

  /**
   * 전환 실패 후 select 표시 복원: 검증 단계 실패면 이전 씬이 그대로 살아 있으므로
   * 이전 값으로 되돌리고, build 단계 실패(sceneLost)면 이전 씬이 이미 해제되어
   * 활성 씬이 없다 — '씬 없음'을 표시해 실상과 표시를 일치시킨다.
   */
  const restoreAfterFailure = (result: SceneSwitchResult): void => {
    if (result.sceneLost) {
      noneOption.hidden = false;
      select.value = NO_SCENE_OPTION_VALUE;
      lastAppliedValue = NO_SCENE_OPTION_VALUE;
    } else {
      select.value = lastAppliedValue;
    }
  };

  /** 전환 진행 중 컨트롤 잠금 (이중 전환 방지 — 글루의 loadScene도 재진입을 거른다) */
  const setBusy = (busy: boolean): void => {
    select.disabled = busy;
    uploadButton.disabled = busy;
    saveButton.disabled = busy;
  };

  // ── 동작 배선 ─────────────────────────────────────────────────────

  select.addEventListener('change', () => {
    const name = select.value;
    if (name === UPLOADED_OPTION_VALUE || name === NO_SCENE_OPTION_VALUE) return;
    hideError();
    setBusy(true);
    void api
      .switchToPreset(name)
      .then((result) => {
        if (result.ok) {
          setCurrent(name);
        } else {
          restoreAfterFailure(result); // 이전 씬 표시 복귀 또는 '씬 없음'
          showError(
            `씬 전환 실패 (${name}):\n${(result.errors ?? ['알 수 없는 오류']).join('\n')}`,
          );
        }
      })
      .finally(() => {
        setBusy(false);
      });
  });

  uploadButton.addEventListener('click', () => {
    fileInput.click();
  });

  fileInput.addEventListener('change', () => {
    const file = fileInput.files?.[0];
    fileInput.value = ''; // 같은 파일 재선택도 change가 다시 발화하도록 초기화
    if (!file) return;
    hideError();
    setBusy(true);
    void (async (): Promise<void> => {
      let payload: unknown;
      try {
        payload = JSON.parse(await file.text()) as unknown;
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        appLog('error', `씬 업로드 JSON 파싱 실패 (${file.name}): ${msg}`);
        showError(`JSON 파싱 실패 (${file.name}):\n${msg}`);
        return;
      }
      const result = await api.switchToUpload(payload, file.name);
      if (result.ok) {
        setCurrent(null);
      } else {
        restoreAfterFailure(result);
        showError(
          `업로드 씬 로드 실패 (${file.name}):\n${(result.errors ?? ['알 수 없는 오류']).join('\n')}`,
        );
      }
    })().finally(() => {
      setBusy(false);
    });
  });

  saveButton.addEventListener('click', () => {
    const spec = api.currentSpec();
    if (!spec) {
      showError('저장할 씬이 없습니다 — 활성 씬이 로드된 뒤 다시 시도하세요');
      return;
    }
    const json = JSON.stringify(spec, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `${sanitizeFileName(spec.name)}.scene.json`;
    anchor.click();
    setTimeout(() => {
      URL.revokeObjectURL(url);
    }, BLOB_URL_REVOKE_DELAY_MS);
    appLog('info', `씬 '${spec.name}' SceneSpec 저장 (${anchor.download})`);
  });

  host.appendChild(section);

  return {
    el: section,
    setCurrent,
    showError,
    dispose: (): void => {
      hideError();
      errorToast.remove();
      section.remove();
    },
  };
}
