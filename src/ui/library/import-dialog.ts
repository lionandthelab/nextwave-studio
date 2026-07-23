// ui/library/import-dialog.ts — 3D 파일 임포트 다이얼로그 (UX_DESIGN §4.4, ROADMAP Phase 7)
//
// 파일 → parseModelFile(형식 감지·삼각형 수·바운딩) → 사용자 설정(스케일·Up-axis·
// Collider 전략·유형) → prepareForScene → registerAsset('asset://<n>') →
// EntitySpec 조립 → onConfirm. 씬 편입(SceneEditor.addEntity)·배치는 통합자 몫이다.
//
// ── 계층 규칙 (CLAUDE.md §3) ─────────────────────────────────────────
// ui는 core를 import하지 않는다. render(mesh-import)는 하위 계층이므로 직접 사용한다
// — 의존 방향은 ui → render 하향이며(역전 없음), three 객체는 불투명하게 통과만
// 시킨다(직접 조작 없음 — parseModelFile/prepareForScene 결과를 registerAsset으로
// 흘려보낼 뿐). 에셋 저장소는 콜백(registerAsset)으로 주입받아 store 구현에
// 비의존적이다.
//
// ── 안전 규칙 (DATA_MODEL §2, UX §4.4) ──────────────────────────────
// 동적 바디에 trimesh 금지 — Collider 전략이 'Trimesh·정적'이면 유형을 Static으로
// 강제하고(Object 버튼 비활성) EntitySpec도 bodyType 'fixed'로 조립한다. 이 강제는
// 순수 함수 forcedEntityKind가 소유한다(단위 테스트 — render/mesh-import.test.ts).
//
// ── 세션 한정 에셋 안내 ──────────────────────────────────────────────
// 임포트 메시는 세션 한정(asset:// ref — mesh-import.ts 헤더)이다. 다이얼로그에
// 상시 안내 문구를 노출하고, 씬 저장(💾) 경로의 경고 배선(collectAssetRefs +
// ASSET_SAVE_WARNING_KO)은 통합자 몫이다.
//
// 모듈 top-level에서는 DOM을 만지지 않는다 — 순수 함수(forcedEntityKind /
// buildImportedEntitySpec / defaultEntityIdFromFileName / formatBboxSizeM)는 node
// 테스트가 import해도 안전하다(theme.ts와 같은 규약).

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
import {
  extractTrimeshGeometry,
  parseModelFile,
  prepareForScene,
} from '../../render/mesh-import';
import type { ImportedModel, MeshAssetBundle } from '../../render/mesh-import';
import type {
  ColliderShape,
  ColliderSpec,
  EntitySpec,
  Transform,
  Vec3,
} from '../../schema/types';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4, 시각 토큰은 ui/theme.ts) ────

const DIALOG_WIDTH_PX = 360;
/** 폼 라벨 컬럼 폭 — 인스펙터 계열 폼과 유사한 좌측 정렬 */
const LABEL_COL_PX = 92;
/** 스케일 기본값 (UX §4.4 목업 "단위/스케일: [1.0] (m)") */
const DEFAULT_SCALE = 1.0;
/** 바운딩 크기 표시 소수 자릿수 (m) */
const SIZE_DECIMALS = 3;
/** 스케일 입력 step/최소값 */
const SCALE_INPUT_STEP = '0.1';
const SCALE_INPUT_MIN = '0';

// ── 폼 상태 (순수 부분 — 테스트 대상) ────────────────────────────────

export type ImportColliderStrategy = 'convexHull' | 'aabb' | 'trimesh';
export type ImportEntityKind = 'object' | 'static';

export interface ImportFormState {
  /** 씬 엔티티 id (씬 내 유일성 검사는 SceneEditor.addEntity 몫) */
  entityId: string;
  scale: number;
  upAxis: 'y' | 'z';
  collider: ImportColliderStrategy;
  entityKind: ImportEntityKind;
}

/**
 * trimesh는 동적 바디 금지(DATA_MODEL §2) — 전략이 trimesh면 요청과 무관하게
 * Static으로 강제한다. UI(유형 토글 비활성)와 EntitySpec 조립이 모두 이 함수를 쓴다.
 */
export function forcedEntityKind(
  collider: ImportColliderStrategy,
  requested: ImportEntityKind,
): ImportEntityKind {
  return collider === 'trimesh' ? 'static' : requested;
}

export interface ImportDerived {
  /** registerAsset이 발급한 'asset://<n>' ref */
  assetRef: string;
  /** prepareForScene 산출 AABB half extents (AABB 전략의 box collider용) */
  aabbHalfExtents: Vec3;
  /** 피벗(바닥 중심) 기준 AABB 중심 오프셋 (box collider offset용) */
  aabbCenterOffset: Vec3;
}

/**
 * 폼 상태 → EntitySpec 조립 (순수 함수 — 테스트 대상).
 *
 * - visual: { kind: 'mesh', ref: assetRef } — 시각은 에셋 저장소가 해석
 * - collider: convexHull{ref} | box(aabbHalfExtents + 중심 offset) | trimesh{ref}
 * - 그룹 규약(CLAUDE.md §5): 동적 Object → OBJECT, Static → ENV. emitEvents true —
 *   로봇–사물 충돌 감지가 프로젝트 핵심이므로 임포트 엔티티도 기본 감지 대상이다.
 *   동적 Object의 collidesWith는 라이브러리 동적 템플릿(templates.ts
 *   OBJECT_COLLIDES_WITH)과 동일하게 [ENV, ROBOT, OBJECT, SENSOR_ZONE]이다 —
 *   Rapier 쌍 필터는 양방향이라 OBJECT 측 필터에 SENSOR_ZONE이 없으면 Sensor Zone이
 *   임포트 사물을 감지하지 못한다. Static(ENV)은 [ENV, ROBOT, OBJECT] 유지.
 * - transform은 [0,0,0] — 배치(드롭 위치/인스펙터)는 호출자 몫. 피벗이 바닥 중심이라
 *   y=0이 곧 지면 안착이다(prepareForScene 계약).
 */
export function buildImportedEntitySpec(
  state: ImportFormState,
  derived: ImportDerived,
): EntitySpec {
  const kind = forcedEntityKind(state.collider, state.entityKind);
  const isStatic = kind === 'static';

  let shape: ColliderShape;
  let offset: Transform | undefined;
  switch (state.collider) {
    case 'convexHull':
      shape = { kind: 'convexHull', ref: derived.assetRef };
      break;
    case 'trimesh':
      shape = { kind: 'trimesh', ref: derived.assetRef };
      break;
    case 'aabb':
      // hull/trimesh 정점은 피벗 재정렬이 이미 반영돼 offset이 불필요하지만,
      // box는 피벗(바닥 중심)과 AABB 중심이 다르므로 중심 오프셋을 명시한다.
      shape = { kind: 'box', halfExtents: [...derived.aabbHalfExtents] };
      offset = { position: [...derived.aabbCenterOffset] };
      break;
  }

  const collider: ColliderSpec = {
    shape,
    ...(offset ? { offset } : {}),
    group: isStatic ? 'ENV' : 'OBJECT',
    // 동적 OBJECT는 SENSOR_ZONE 포함 — 라이브러리 동적 템플릿과 동일 배정 (파일 상단 근거)
    collidesWith: isStatic
      ? ['ENV', 'ROBOT', 'OBJECT']
      : ['ENV', 'ROBOT', 'OBJECT', 'SENSOR_ZONE'],
    emitEvents: true,
  };

  return {
    id: state.entityId,
    type: isStatic ? 'static' : 'object',
    transform: { position: [0, 0, 0] },
    visual: { kind: 'mesh', ref: derived.assetRef },
    physics: {
      bodyType: isStatic ? 'fixed' : 'dynamic',
      colliders: [collider],
    },
  };
}

/** 파일명 → 기본 엔티티 id ('My Model.glb' → 'my-model'). 전부 걸러지면 'imported-mesh'. */
export function defaultEntityIdFromFileName(fileName: string): string {
  const base = fileName.replace(/\.[^.]*$/, '');
  const cleaned = base
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, '-')
    .replace(/^-+|-+$/g, '');
  return cleaned.length > 0 ? cleaned : 'imported-mesh';
}

/** 원본 bbox × 스케일 → "W × H × L m" 표시 문자열 (순수 함수 — 다이얼로그 리드아웃) */
export function formatBboxSizeM(bbox: { min: Vec3; max: Vec3 }, scale: number): string {
  const size = (axis: 0 | 1 | 2): string =>
    ((bbox.max[axis] - bbox.min[axis]) * scale).toFixed(SIZE_DECIMALS);
  return `${size(0)} × ${size(1)} × ${size(2)} m`;
}

// ── 공개 타입 ───────────────────────────────────────────────────────

/** 통합자(글루)가 주입하는 store-비의존 콜백 표면 */
export interface ImportDialogDeps {
  /** 준비된 에셋 번들 등록 → 'asset://<n>' ref (글루가 MeshAssetStore.register를 배선) */
  registerAsset(bundle: MeshAssetBundle): string;
  /**
   * [추가] 확정 통지 — 스키마 검증 가능한 EntitySpec + 시각 에셋 ref.
   * 씬 편입(SceneEditor.addEntity)·드롭 위치 반영은 통합자 몫이다.
   */
  onConfirm(entity: EntitySpec, visualObjectRef: string): void;
}

export interface ImportDialogHandle {
  /** 파일을 파싱해 다이얼로그를 연다 (파싱 실패 시 다이얼로그 안에 사유 표시 — UX §7) */
  openWith(file: File): void;
  dispose(): void;
}

// ── 내부: 세그먼트 토글 (라디오 대체 — 버튼 + aria-pressed) ──────────

interface SegmentedOption<T extends string> {
  value: T;
  label: string;
  title: string;
  testId: string;
}

interface SegmentedControl<T extends string> {
  el: HTMLElement;
  get(): T;
  set(value: T): void;
  setOptionDisabled(value: T, disabled: boolean): void;
  setAllDisabled(disabled: boolean): void;
}

function makeSegmented<T extends string>(
  options: readonly SegmentedOption<T>[],
  initial: T,
  onChange: (value: T) => void,
): SegmentedControl<T> {
  const el = styled(document.createElement('div'), {
    display: 'flex',
    flexWrap: 'wrap',
    gap: SPACE.xs,
  });
  el.setAttribute('role', 'group');
  let current = initial;
  const buttons = new Map<T, HTMLButtonElement>();

  const render = (): void => {
    for (const [value, button] of buttons) {
      const active = value === current;
      button.classList.toggle('ui-btn--active', active);
      button.setAttribute('aria-pressed', String(active));
    }
  };

  for (const option of options) {
    const button = makeButton(option.label, option.title, option.testId);
    button.addEventListener('click', () => {
      if (button.disabled || current === option.value) return;
      current = option.value;
      render();
      onChange(current);
    });
    buttons.set(option.value, button);
    el.appendChild(button);
  }
  render();

  return {
    el,
    get: () => current,
    set: (value: T): void => {
      if (current === value) return;
      current = value;
      render();
      onChange(current);
    },
    setOptionDisabled: (value: T, disabled: boolean): void => {
      const button = buttons.get(value);
      if (button) button.disabled = disabled;
    },
    setAllDisabled: (disabled: boolean): void => {
      for (const button of buttons.values()) button.disabled = disabled;
    },
  };
}

// ── 마운트 ──────────────────────────────────────────────────────────

type DialogPhase = 'loading' | 'ready' | 'error';

/**
 * 임포트 다이얼로그를 host에 마운트한다(초기 숨김). 프로젝트 mount* 규약에 따라
 * host를 첫 인자로 받는다 — 라이브러리 패널(⬆ 카드)이나 뷰포트 드롭 핸들러가
 * openWith(file)로 연다.
 */
export function mountImportDialog(
  host: HTMLElement,
  deps: ImportDialogDeps,
): ImportDialogHandle {
  ensureThemeStyles();

  // ── 오버레이 (모달 — 배경 클릭/Esc = 취소) ────────────────────────
  const overlay = styled(document.createElement('div'), {
    position: 'fixed',
    inset: '0',
    zIndex: Z_INDEX.panel,
    display: 'none',
    alignItems: 'center',
    justifyContent: 'center',
    background: 'transparent',
    pointerEvents: 'auto',
  });
  overlay.dataset.testid = 'import-dialog';
  // 다이얼로그 위 상호작용이 뷰포트 orbit으로 새지 않게 흡수 (커맨드바/독과 동일 규약)
  for (const type of ['pointerdown', 'pointermove', 'pointerup', 'wheel', 'contextmenu']) {
    overlay.addEventListener(type, (e) => {
      e.stopPropagation();
    });
  }

  const panel = styled(document.createElement('div'), {
    width: `${DIALOG_WIDTH_PX}px`,
    maxWidth: '92vw',
    maxHeight: '86vh',
    overflowY: 'auto',
    background: COLOR.bgPanel,
    border: `1px solid ${COLOR.border}`,
    borderRadius: RADIUS.md,
    boxShadow: SHADOW.panel,
    color: COLOR.text,
    fontFamily: FONT.ui,
    fontSize: '12px',
    lineHeight: '1.6',
    padding: SPACE.lg,
    boxSizing: 'border-box',
    display: 'flex',
    flexDirection: 'column',
    gap: SPACE.md,
  });
  panel.className = 'ui-scroll';
  panel.setAttribute('role', 'dialog');
  panel.setAttribute('aria-modal', 'true');
  panel.setAttribute('aria-label', '3D 파일 임포트');
  panel.tabIndex = -1;
  overlay.appendChild(panel);

  // ── 헤더 ─────────────────────────────────────────────────────────
  const title = styled(document.createElement('strong'), {
    color: COLOR.textStrong,
    fontSize: '13px',
  });
  title.textContent = '3D 파일 임포트';
  const fileNameEl = styled(document.createElement('div'), {
    color: COLOR.muted,
    fontFamily: FONT.mono,
    fontSize: '11px',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  });
  panel.appendChild(title);
  panel.appendChild(fileNameEl);

  // ── 행 헬퍼 ──────────────────────────────────────────────────────
  const row = (labelText: string, control: HTMLElement, forId?: string): HTMLElement => {
    const rowEl = styled(document.createElement('div'), {
      display: 'flex',
      alignItems: 'center',
      gap: SPACE.md,
    });
    const label = styled(document.createElement('label'), {
      width: `${LABEL_COL_PX}px`,
      flexShrink: '0',
      color: COLOR.label,
    });
    label.textContent = labelText;
    if (forId) label.htmlFor = forId;
    rowEl.appendChild(label);
    rowEl.appendChild(control);
    return rowEl;
  };
  const readout = (testId: string): HTMLElement => {
    const el = styled(document.createElement('span'), {
      fontFamily: FONT.mono,
      color: COLOR.textStrong,
      minWidth: '0',
      overflow: 'hidden',
      textOverflow: 'ellipsis',
      whiteSpace: 'nowrap',
    });
    el.dataset.testid = testId;
    return el;
  };

  // ── 정보 리드아웃 (형식 감지 · 삼각형 수 · 바운딩 크기) ──────────
  const formatReadout = readout('import-format');
  const triangleReadout = readout('import-triangles');
  const sizeReadout = readout('import-size');
  panel.appendChild(row('형식 감지', formatReadout));
  panel.appendChild(row('삼각형 수', triangleReadout));
  panel.appendChild(row('바운딩 크기', sizeReadout));

  // ── 폼 컨트롤 ────────────────────────────────────────────────────
  const idInput = document.createElement('input');
  idInput.type = 'text';
  idInput.className = 'ui-input';
  idInput.id = 'rsw-import-entity-id';
  idInput.dataset.testid = 'import-id';
  idInput.title = '씬 엔티티 id (씬 내 유일해야 합니다)';
  styled(idInput, { flex: '1', minWidth: '0' });
  panel.appendChild(row('이름', idInput, idInput.id));

  const scaleWrap = styled(document.createElement('div'), {
    display: 'flex',
    alignItems: 'center',
    gap: SPACE.sm,
  });
  const scaleInput = document.createElement('input');
  scaleInput.type = 'number';
  scaleInput.className = 'ui-input';
  scaleInput.id = 'rsw-import-scale';
  scaleInput.dataset.testid = 'import-scale';
  scaleInput.step = SCALE_INPUT_STEP;
  scaleInput.min = SCALE_INPUT_MIN;
  scaleInput.title = '균일 스케일 — 원본 단위 → 미터 (예: mm 원본이면 0.001)';
  styled(scaleInput, { width: '80px' });
  const scaleUnit = styled(document.createElement('span'), { color: COLOR.muted });
  scaleUnit.textContent = '(m)';
  scaleWrap.appendChild(scaleInput);
  scaleWrap.appendChild(scaleUnit);
  panel.appendChild(row('단위/스케일', scaleWrap, scaleInput.id));

  // Up-axis 토글 (UX §4.4: ( Y-up ) ( Z-up→변환 ))
  const upAxisSeg = makeSegmented<'y' | 'z'>(
    [
      { value: 'y', label: 'Y-up', title: '원본이 Y-up — 변환 없음', testId: 'import-upaxis-y' },
      {
        value: 'z',
        label: 'Z-up→변환',
        title: '원본이 Z-up — Y-up으로 회전 (-90° X, URDF 경로와 동일 규약)',
        testId: 'import-upaxis-z',
      },
    ],
    'y',
    () => {
      /* 표시 전용 상태 — 확정 시 폼 스냅샷으로 읽는다 */
    },
  );
  panel.appendChild(row('Up-axis', upAxisSeg.el));

  // Collider 전략 (Convex Hull 기본 / AABB / Trimesh-정적)
  const trimeshNote = styled(document.createElement('div'), {
    display: 'none',
    color: COLOR.muted,
    fontSize: '11px',
    paddingLeft: `${LABEL_COL_PX}px`,
  });
  trimeshNote.dataset.testid = 'import-trimesh-note';
  trimeshNote.textContent = '※ Trimesh는 Static(고정) 전용 — 동적 바디에 trimesh 금지 (DATA_MODEL §2)';

  const kindSeg = makeSegmented<ImportEntityKind>(
    [
      {
        value: 'object',
        label: 'Object · 동적',
        title: '중력/힘에 반응하는 조작 대상 (OBJECT 그룹)',
        testId: 'import-kind-object',
      },
      {
        value: 'static',
        label: 'Static · 고정',
        title: '움직이지 않는 환경 요소 (ENV 그룹)',
        testId: 'import-kind-static',
      },
    ],
    'object',
    () => {
      /* 표시 전용 상태 */
    },
  );

  const applyTrimeshForcing = (strategy: ImportColliderStrategy): void => {
    const isTrimesh = strategy === 'trimesh';
    trimeshNote.style.display = isTrimesh ? 'block' : 'none';
    if (isTrimesh) {
      kindSeg.set(forcedEntityKind(strategy, kindSeg.get())); // → 'static'
      kindSeg.setOptionDisabled('object', true);
    } else {
      kindSeg.setOptionDisabled('object', false);
    }
  };

  const colliderSeg = makeSegmented<ImportColliderStrategy>(
    [
      {
        value: 'convexHull',
        label: 'Convex Hull',
        title: '볼록 껍질 collider — 동적 바디 기본 권장 (UX §4.4 안전 규칙)',
        testId: 'import-collider-hull',
      },
      {
        value: 'aabb',
        label: 'AABB',
        title: '경계 상자(box) collider — 가장 단순·빠름',
        testId: 'import-collider-aabb',
      },
      {
        value: 'trimesh',
        label: 'Trimesh·정적',
        title: '삼각형 메시 collider — Static(고정) 전용 (DATA_MODEL §2)',
        testId: 'import-collider-trimesh',
      },
    ],
    'convexHull',
    applyTrimeshForcing,
  );
  panel.appendChild(row('Collider', colliderSeg.el));
  panel.appendChild(trimeshNote);
  panel.appendChild(row('유형', kindSeg.el));

  // 세션 한정 에셋 상시 안내 (파일 헤더 참조)
  const sessionNote = styled(document.createElement('div'), {
    color: COLOR.muted,
    fontSize: '11px',
  });
  sessionNote.textContent =
    '※ 임포트 메시는 세션 한정입니다 — 씬 저장 시 메시 데이터가 파일에 내장되지 않습니다.';
  panel.appendChild(sessionNote);

  // ── 오류 표시 (파싱 실패 사유 / 폼 검증 — UX §7 "임포트 실패") ────
  const errorBox = styled(document.createElement('div'), {
    display: 'none',
    background: COLOR.errorSoft,
    border: `1px solid ${COLOR.errorBorder}`,
    borderRadius: RADIUS.sm,
    color: COLOR.errorText,
    fontFamily: FONT.mono,
    fontSize: '11px',
    padding: `${SPACE.sm} ${SPACE.md}`,
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-word',
  });
  errorBox.dataset.testid = 'import-error';
  errorBox.setAttribute('role', 'alert');
  panel.appendChild(errorBox);

  const showError = (message: string): void => {
    errorBox.textContent = message;
    errorBox.style.display = 'block';
  };
  const hideError = (): void => {
    errorBox.style.display = 'none';
  };

  // ── 푸터 [취소][추가] ────────────────────────────────────────────
  const footer = styled(document.createElement('div'), {
    display: 'flex',
    justifyContent: 'flex-end',
    gap: SPACE.sm,
    marginTop: SPACE.xs,
  });
  const cancelButton = makeButton('취소', '임포트 취소', 'import-cancel');
  const confirmButton = makeButton('추가', '씬에 엔티티로 추가', 'import-confirm', 'accent');
  footer.appendChild(cancelButton);
  footer.appendChild(confirmButton);
  panel.appendChild(footer);

  host.appendChild(overlay);

  // ── 상태/수명 ────────────────────────────────────────────────────
  let currentModel: ImportedModel | null = null;
  /** openWith 재호출 경합 가드 — 낡은 비동기 파싱 결과를 폐기한다 */
  let openToken = 0;

  const onKeyDown = (e: KeyboardEvent): void => {
    if (e.key === 'Escape') close();
  };

  const setFormDisabled = (disabled: boolean): void => {
    idInput.disabled = disabled;
    scaleInput.disabled = disabled;
    upAxisSeg.setAllDisabled(disabled);
    colliderSeg.setAllDisabled(disabled);
    kindSeg.setAllDisabled(disabled);
    if (!disabled) applyTrimeshForcing(colliderSeg.get()); // trimesh면 Object 다시 잠금
  };

  const setPhase = (phase: DialogPhase, message?: string): void => {
    if (phase === 'loading') {
      hideError();
      formatReadout.textContent = message ?? '분석 중…';
      triangleReadout.textContent = '—';
      sizeReadout.textContent = '—';
      setFormDisabled(true);
      confirmButton.disabled = true;
      return;
    }
    if (phase === 'error') {
      showError(message ?? '알 수 없는 오류');
      formatReadout.textContent = '실패';
      setFormDisabled(true);
      confirmButton.disabled = true;
      return;
    }
    hideError();
    setFormDisabled(false);
    confirmButton.disabled = false;
  };

  const refreshSizeReadout = (): void => {
    if (!currentModel) return;
    const scale = Number(scaleInput.value);
    sizeReadout.textContent =
      Number.isFinite(scale) && scale > 0
        ? formatBboxSizeM(currentModel.bbox, scale)
        : '— (스케일이 유효하지 않음)';
  };
  scaleInput.addEventListener('input', refreshSizeReadout);

  const resetForm = (fileName: string): void => {
    fileNameEl.textContent = fileName;
    fileNameEl.title = fileName;
    idInput.value = defaultEntityIdFromFileName(fileName);
    scaleInput.value = String(DEFAULT_SCALE);
    upAxisSeg.set('y');
    colliderSeg.set('convexHull');
    kindSeg.set('object');
    applyTrimeshForcing('convexHull');
  };

  const show = (): void => {
    overlay.style.display = 'flex';
    document.addEventListener('keydown', onKeyDown);
    panel.focus();
  };

  const close = (): void => {
    openToken += 1; // 진행 중 파싱 결과 폐기
    overlay.style.display = 'none';
    document.removeEventListener('keydown', onKeyDown);
    currentModel = null; // 미확정 모델은 등록되지 않는다 — GC에 맡김
  };

  // 배경 클릭 = 취소 (패널 내부 클릭은 제외)
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) close();
  });
  cancelButton.addEventListener('click', close);

  // ── [추가] 확정 ──────────────────────────────────────────────────
  confirmButton.addEventListener('click', () => {
    const model = currentModel;
    if (!model) return;

    const entityId = idInput.value.trim();
    if (entityId.length === 0) {
      showError('이름(엔티티 id)을 입력하세요');
      return;
    }
    const scale = Number(scaleInput.value);
    if (!Number.isFinite(scale) || scale <= 0) {
      showError('스케일은 0보다 큰 수여야 합니다');
      return;
    }

    const state: ImportFormState = {
      entityId,
      scale,
      upAxis: upAxisSeg.get(),
      collider: colliderSeg.get(),
      entityKind: kindSeg.get(),
    };

    try {
      const prepared = prepareForScene(model, { scale: state.scale, upAxis: state.upAxis });
      const bundle: MeshAssetBundle = {
        object3D: prepared.object3D,
        points: prepared.points,
        ...(state.collider === 'trimesh'
          ? { trimesh: extractTrimeshGeometry(prepared.object3D) }
          : {}),
      };
      const assetRef = deps.registerAsset(bundle);
      const entity = buildImportedEntitySpec(state, {
        assetRef,
        aabbHalfExtents: prepared.aabbHalfExtents,
        aabbCenterOffset: prepared.aabbCenterOffset,
      });
      deps.onConfirm(entity, assetRef);
      close();
    } catch (err) {
      showError(err instanceof Error ? err.message : String(err));
    }
  });

  // ── openWith (파일 → 파싱 → 폼 표시) ─────────────────────────────
  const openWith = (file: File): void => {
    openToken += 1;
    const token = openToken;
    currentModel = null;
    resetForm(file.name);
    setPhase('loading', `'${file.name}' 분석 중…`);
    show();

    void (async (): Promise<void> => {
      try {
        const buffer = await file.arrayBuffer();
        const model = await parseModelFile({ name: file.name, buffer });
        if (token !== openToken) return; // 그 사이 닫혔거나 다른 파일이 열림
        currentModel = model;
        formatReadout.textContent = model.formatLabel;
        triangleReadout.textContent = model.triangleCount.toLocaleString('en-US');
        refreshSizeReadout();
        setPhase('ready');
        idInput.focus();
      } catch (err) {
        if (token !== openToken) return;
        setPhase('error', err instanceof Error ? err.message : String(err));
      }
    })();
  };

  return {
    openWith,
    dispose: (): void => {
      close();
      overlay.remove();
    },
  };
}
