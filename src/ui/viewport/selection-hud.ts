// ui/viewport/selection-hud.ts — 선택 오브젝트 조작 HUD (UX_DESIGN §3.3)
//
// ── 왜 필요한가 ─────────────────────────────────────────────────────
//
// 오브젝트를 클릭하면 아웃라인과 기즈모는 나타나지만, **그 다음에 무엇을 할 수 있는지**
// 알려주는 것이 화면에 없었다. 이동 방향키(←→↑↓ · Shift 미세 · PageUp/Dn)와 치수 편집은
// 이미 구현돼 있었는데도, 그 사실이 오른쪽 인스펙터를 펼쳐 Dimensions 섹션까지 스크롤한
// 사용자에게만 도달했다. 구현된 기능이 발견되지 않으면 없는 것과 같다.
//
// ── 참조한 관례 ─────────────────────────────────────────────────────
//
// - **Blender N-패널(Item 탭)**: 선택 대상의 트랜스폼과 **Dimensions를 뷰포트 안에서**
//   바로 읽고 고친다. 이 HUD의 골격이다 — 시선을 3D에서 떼지 않고 크기를 조정한다.
// - **Unity/Isaac Sim**: W/E/R 모드 전환, 스냅 토글이 뷰포트 툴바에 상주한다.
//   (이 저장소는 좌상단 기즈모 바가 이미 담당 — HUD는 중복하지 않는다.)
// - **Unity/Unreal의 snap-to-floor(End)**: 떠 있거나 지하에 박힌 사물을 바닥에 앉힌다.
//   바닥 클램프(core/ground-clamp.ts)의 짝이 되는 복구 조작이다.
//
// ── 설계 ────────────────────────────────────────────────────────────
//
// - 선택이 없으면 **아예 그리지 않는다**(display:none) — 빈 상태 카드를 띄우면 3D 화면의
//   1/6이 상시 잠식된다. 선택 순간 나타나는 것 자체가 "이 대상에 무언가 할 수 있다"는 신호다.
// - 치수 스테퍼(− / +)는 **프리미티브에만** 그린다. 로봇·임포트 메시는 치수 편집 대상이
//   아니므로(scene-editor.updateDimensions 계약) 누르면 실패할 버튼을 보여주지 않는다.
// - 값 표시는 **읽기 전용 리드아웃**이다. 정밀 입력·스크럽은 인스펙터가 소유한다 —
//   같은 편집기를 두 곳에 두면 포커스/커밋 경합이 생긴다. HUD는 "한 칸씩" 담당이다.
// - 모든 편집은 통합자가 준 콜백으로만 나간다. 이 모듈은 core/물리/three를 모른다
//   (CLAUDE.md §3). 클램프·undo·검증은 전부 통합자 경로에서 이미 처리된다.
//
// 배치 계약: host는 positioned 뷰포트 슬롯이어야 한다(우하단 absolute). 하단 좌측의
// 상태줄과 겹치지 않는다.

import {
  BORDER,
  BORDER_WIDTH,
  COLOR,
  ICON,
  RADIUS,
  SELECT,
  SPACE,
  SURFACE,
  TYPE,
  Z_INDEX,
  applyType,
  ensureThemeStyles,
  makeButton,
  styled,
} from '../theme';
import { icon, makeIconButton } from '../icons';
import type { ColliderShape, EntitySpec, Vec3 } from '../../schema/types';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4) ─────────────────────────────

const HUD_RIGHT_PX = 12;
const HUD_BOTTOM_PX = 12;
const HUD_MIN_WIDTH_PX = 232;
/** 위치 리드아웃 소수 자릿수 (인스펙터 Transform과 같은 규약) */
const POSITION_DECIMALS = 3;
/** 치수 리드아웃 소수 자릿수 */
const DIMENSION_DECIMALS = 3;

/**
 * 치수 스테퍼 1회 증감 (m, 전체 치수 기준). 인스펙터 입력의 step과 같은 1 cm —
 * 두 경로에서 같은 크기로 움직여야 "같은 조작의 다른 표면"으로 학습된다.
 */
export const HUD_DIMENSION_STEP_M = 0.01;
/** 치수 하한 (m) — entity-editor의 DIMENSION_MIN_M과 같은 값 */
export const HUD_DIMENSION_MIN_M = 0.01;

// ── 순수 헬퍼 (DOM 비의존 — node 단위 테스트 대상) ──────────────────

/** 치수 축 — 형상 종류에 따라 노출되는 조합이 다르다 */
export type DimensionAxis = 'width' | 'height' | 'length' | 'radius';

export interface DimensionField {
  readonly axis: DimensionAxis;
  /** 스테퍼 라벨 (한국어 — UI 크롬은 한국어, CLAUDE.md §4-b) */
  readonly labelKo: string;
  /** 현재 **전체 치수** (m) */
  readonly valueM: number;
}

/**
 * 형상 → HUD가 그릴 치수 필드 목록 (전체 치수 기준 — halfExtents는 내부 표현이다).
 * 편집 불가 형상(convexHull/trimesh/fromVisual)은 빈 배열 → 스테퍼를 그리지 않는다.
 */
export function dimensionFieldsOf(shape: Readonly<ColliderShape>): readonly DimensionField[] {
  switch (shape.kind) {
    case 'box':
      return [
        { axis: 'width', labelKo: '너비', valueM: shape.halfExtents[0] * 2 },
        { axis: 'height', labelKo: '높이', valueM: shape.halfExtents[1] * 2 },
        { axis: 'length', labelKo: '길이', valueM: shape.halfExtents[2] * 2 },
      ];
    case 'sphere':
      return [{ axis: 'radius', labelKo: '반지름', valueM: shape.radius }];
    case 'capsule':
    case 'cylinder':
      return [
        { axis: 'radius', labelKo: '반지름', valueM: shape.radius },
        { axis: 'height', labelKo: '높이', valueM: shape.halfHeight * 2 },
      ];
    default:
      return [];
  }
}

/**
 * 한 축을 deltaM만큼 키운/줄인 새 형상 (전체 치수 기준, 하한 클램프 포함).
 * 형상 종류가 갖지 않는 축이 들어오면 원 형상을 그대로 돌려준다(무해한 no-op).
 */
export function stepShapeDimension(
  shape: Readonly<ColliderShape>,
  axis: DimensionAxis,
  deltaM: number,
  minM: number = HUD_DIMENSION_MIN_M,
): ColliderShape {
  const bump = (fullM: number): number => Math.max(minM, fullM + deltaM);
  switch (shape.kind) {
    case 'box': {
      const full: Vec3 = [
        shape.halfExtents[0] * 2,
        shape.halfExtents[1] * 2,
        shape.halfExtents[2] * 2,
      ];
      if (axis === 'width') full[0] = bump(full[0]);
      else if (axis === 'height') full[1] = bump(full[1]);
      else if (axis === 'length') full[2] = bump(full[2]);
      else return shape;
      return { kind: 'box', halfExtents: [full[0] / 2, full[1] / 2, full[2] / 2] };
    }
    case 'sphere':
      if (axis !== 'radius') return shape;
      return { kind: 'sphere', radius: bump(shape.radius) };
    case 'capsule':
    case 'cylinder': {
      if (axis === 'radius') return { ...shape, radius: bump(shape.radius) };
      if (axis === 'height') return { ...shape, halfHeight: bump(shape.halfHeight * 2) / 2 };
      return shape;
    }
    default:
      return shape;
  }
}

/** 엔티티의 편집 가능한 프리미티브 형상 (없으면 null — 로봇·임포트 메시) */
export function editableShapeOf(entity: Readonly<EntitySpec>): ColliderShape | null {
  if (entity.type === 'robot') return null;
  if (entity.visual.kind !== 'primitive' || !entity.visual.primitive) return null;
  const colliderCount = entity.physics?.colliders.length ?? 0;
  // scene-editor.updateDimensions가 collider 2개 이상을 거부한다 — 실패할 버튼을 그리지 않는다
  if (colliderCount > 1) return null;
  const shape = entity.visual.primitive;
  return dimensionFieldsOf(shape).length > 0 ? shape : null;
}

/** 위치 리드아웃 문자열 (`x 0.350 · y 0.025 · z 0.150`) */
export function formatPosition(position: Readonly<Vec3>): string {
  const f = (n: number): string => n.toFixed(POSITION_DECIMALS);
  return `x ${f(position[0])} · y ${f(position[1])} · z ${f(position[2])}`;
}

/** 엔티티 유형 → 한국어 배지 라벨 */
export function entityKindLabelKo(entity: Readonly<EntitySpec>): string {
  switch (entity.type) {
    case 'robot':
      return '로봇';
    case 'static':
      return '정적';
    default:
      return '사물';
  }
}

// ── 공개 타입 ───────────────────────────────────────────────────────

export interface SelectionHudDeps {
  /** 치수 한 칸 조정 → 통합자의 치수 편집 경로 (클램프·undo 포함) */
  stepDimension(id: string, shape: ColliderShape): void;
  /** 바닥에 붙이기 (End 키와 같은 함수를 부를 것 — 동작 분기 금지) */
  snapToGround(id: string): void;
  /** 선택 해제 */
  clearSelection(): void;
}

export interface SelectionHudHandle {
  readonly el: HTMLElement;
  /** 선택/스펙 변경 시 호출 (null = 선택 없음 → 숨김) */
  setSelection(entity: Readonly<EntitySpec> | null): void;
  dispose(): void;
}

// ── 마운트 ──────────────────────────────────────────────────────────

export function mountSelectionHud(
  host: HTMLElement,
  deps: SelectionHudDeps,
): SelectionHudHandle {
  ensureThemeStyles();

  const panel = styled(document.createElement('div'), {
    position: 'absolute',
    right: `${HUD_RIGHT_PX}px`,
    bottom: `${HUD_BOTTOM_PX}px`,
    zIndex: Z_INDEX.bar,
    display: 'none',
    flexDirection: 'column',
    gap: SPACE.sm,
    minWidth: `${HUD_MIN_WIDTH_PX}px`,
    padding: `${SPACE.sm} ${SPACE.md}`,
    background: SURFACE.panel,
    // 선택은 청색 축(SELECT)이 소유한다 — 액센트(토글 상태)와 섞지 않는다 (§2.9)
    border: `${BORDER_WIDTH.hair} solid ${SELECT.border}`,
    borderRadius: RADIUS.md,
    userSelect: 'none',
  });
  panel.dataset.testid = 'selection-hud';
  panel.setAttribute('role', 'group');
  panel.setAttribute('aria-label', '선택 오브젝트 조작');

  // ── 헤더: 대상 id + 유형 배지 + 선택 해제 ────────────────────────
  const header = styled(document.createElement('div'), {
    display: 'flex',
    alignItems: 'center',
    gap: SPACE.sm,
  });

  const nameEl = applyType(document.createElement('span'), TYPE.monoReadout);
  styled(nameEl, {
    color: SELECT.text,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    flex: '1 1 auto',
  });
  // 엔티티 id는 도메인 식별자(영문 원문 유지) — 한국어 TTS가 철자로 읽지 않게 (§4-b.3)
  nameEl.lang = 'en';
  nameEl.dataset.testid = 'selection-hud-name';

  const kindBadge = applyType(document.createElement('span'), TYPE.micro);
  styled(kindBadge, {
    flex: 'none',
    padding: `0 ${SPACE.sm}`,
    borderRadius: RADIUS.sm,
    background: SURFACE.raised,
    color: COLOR.muted,
  });
  kindBadge.dataset.testid = 'selection-hud-kind';

  const closeButton = makeIconButton('close', '', '선택 해제', 'selection-hud-clear', 'ghost');
  closeButton.addEventListener('click', () => {
    deps.clearSelection();
  });

  header.appendChild(nameEl);
  header.appendChild(kindBadge);
  header.appendChild(closeButton);

  // ── 위치 리드아웃 ────────────────────────────────────────────────
  const positionRow = applyType(document.createElement('div'), TYPE.monoReadout);
  styled(positionRow, { color: COLOR.label, whiteSpace: 'nowrap' });
  positionRow.lang = 'en';
  positionRow.dataset.testid = 'selection-hud-position';

  // ── 치수 스테퍼 (프리미티브 전용) ────────────────────────────────
  const dimensionsWrap = styled(document.createElement('div'), {
    display: 'flex',
    flexDirection: 'column',
    gap: SPACE.xxs,
  });
  dimensionsWrap.dataset.testid = 'selection-hud-dimensions';

  const dimensionCaption = applyType(document.createElement('div'), TYPE.micro);
  styled(dimensionCaption, { color: COLOR.muted });
  dimensionCaption.textContent = `치수 (m) — ± ${HUD_DIMENSION_STEP_M * 100}cm`;

  // ── 바닥에 붙이기 ────────────────────────────────────────────────
  const groundButton = makeButton(
    '바닥에 붙이기',
    '선택 오브젝트를 바닥에 앉힌다 (End)',
    'selection-hud-snap-ground',
  );
  styled(groundButton, { width: '100%', justifyContent: 'center' });
  groundButton.prepend(icon('shapePlane', ICON.sm));
  groundButton.addEventListener('click', () => {
    if (currentId !== null) deps.snapToGround(currentId);
  });

  // ── 키보드 안내 ──────────────────────────────────────────────────
  // 이 HUD의 존재 이유의 절반이다: 이미 동작하는 키를 화면에서 알려준다.
  const hint = applyType(document.createElement('div'), TYPE.micro);
  styled(hint, {
    color: COLOR.muted,
    borderTop: `${BORDER_WIDTH.hair} solid ${BORDER.subtle}`,
    paddingTop: SPACE.xs,
    lineHeight: '1.5',
  });
  hint.dataset.testid = 'selection-hud-hint';
  hint.textContent =
    '← → ↑ ↓ 이동 · Shift 미세 · PageUp/PageDown 높이 · W/E/R 기즈모 · F 포커스';

  panel.appendChild(header);
  panel.appendChild(positionRow);
  panel.appendChild(dimensionsWrap);
  panel.appendChild(groundButton);
  panel.appendChild(hint);
  host.appendChild(panel);

  let currentId: string | null = null;

  /** 치수 행 (라벨 + − 값 +) 한 줄을 그린다 */
  const buildDimensionRow = (
    entityId: string,
    shape: ColliderShape,
    field: DimensionField,
  ): HTMLElement => {
    const row = styled(document.createElement('div'), {
      display: 'flex',
      alignItems: 'center',
      gap: SPACE.xs,
    });

    const label = applyType(document.createElement('span'), TYPE.micro);
    styled(label, { color: COLOR.label, flex: '1 1 auto' });
    label.textContent = field.labelKo;

    const value = applyType(document.createElement('span'), TYPE.monoReadout);
    styled(value, { color: COLOR.text, minWidth: '52px', textAlign: 'right' });
    value.textContent = field.valueM.toFixed(DIMENSION_DECIMALS);
    value.dataset.testid = `selection-hud-dim-${field.axis}`;

    const makeStep = (deltaM: number, iconName: 'plus' | 'minus'): HTMLButtonElement => {
      const sign = deltaM > 0 ? '늘리기' : '줄이기';
      const button = makeIconButton(
        iconName,
        '',
        `${field.labelKo} ${sign} (${HUD_DIMENSION_STEP_M * 100}cm)`,
        `selection-hud-${field.axis}-${deltaM > 0 ? 'inc' : 'dec'}`,
        'ghost',
      );
      button.addEventListener('click', () => {
        deps.stepDimension(entityId, stepShapeDimension(shape, field.axis, deltaM));
      });
      return button;
    };

    row.appendChild(label);
    row.appendChild(makeStep(-HUD_DIMENSION_STEP_M, 'minus'));
    row.appendChild(value);
    row.appendChild(makeStep(HUD_DIMENSION_STEP_M, 'plus'));
    return row;
  };

  const setSelection = (entity: Readonly<EntitySpec> | null): void => {
    if (entity === null) {
      currentId = null;
      panel.style.display = 'none';
      dimensionsWrap.replaceChildren();
      return;
    }
    currentId = entity.id;
    panel.style.display = 'flex';
    nameEl.textContent = entity.id;
    nameEl.title = entity.id;
    kindBadge.textContent = entityKindLabelKo(entity);
    positionRow.textContent = formatPosition(entity.transform.position);

    const shape = editableShapeOf(entity);
    if (shape === null) {
      dimensionsWrap.replaceChildren();
      return;
    }
    // 매 갱신마다 재구축한다 — 행 수가 형상 종류(박스 3줄 / 구 1줄)에 따라 달라지고,
    // HUD는 포커스를 들고 있지 않은 리드아웃이라 재구축의 대가가 없다.
    const rows = dimensionFieldsOf(shape).map((field) =>
      buildDimensionRow(entity.id, shape, field),
    );
    dimensionsWrap.replaceChildren(dimensionCaption, ...rows);
  };

  return {
    el: panel,
    setSelection,
    dispose: (): void => {
      panel.remove();
    },
  };
}
