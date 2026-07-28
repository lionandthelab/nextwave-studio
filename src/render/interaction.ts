// render/interaction.ts — 뷰포트 3D 상호작용: 선택·아웃라인·Transform 기즈모·바닥 레이캐스트
// (UX_DESIGN §3.3, ROADMAP Phase 7 "Viewport 상호작용")
//
// ── 책임 ─────────────────────────────────────────────────────────────
// - 클릭 픽킹: pickables(entityId → 시각 루트 Object3D)에 대한 재귀 raycast로 선택,
//   빈 곳 클릭이면 선택 해제. orbit 드래그와의 구분은 pointerdown↔up 이동 거리 임계값.
//   메시를 빗나간 클릭은 **CLICK_TOLERANCE_PX 반경의 8방위 재조준**으로 한 번 더
//   판정한다 — 로봇 링크처럼 얇은 메시는 몇 px 차이로 빗나가기 때문이다. 여유는 화면
//   좌표에서만 주므로 "실루엣 근처"에서만 관대해지고, 빈 곳 클릭 = 선택 해제라는 규범
//   (UX_DESIGN §3.3)은 유지된다. ★ 월드 AABB와 광선을 교차시키는 방식은 쓰지 않는다:
//   AABB는 3D 상자여서 비스듬한 카메라에서 실루엣보다 훨씬 넓은 화면 영역을 덮고,
//   로봇에서 한참 떨어진 빈 하늘/바닥 클릭까지 선택으로 만든다(실측: 캔버스 격자
//   1000점 중 229점이 팔 하나에 흡수, 두 팔 사이 빈 공간에서는 엉뚱한 팔이 선택됐다).
// - 선택 하이라이트: emissive 틴트 + THREE.BoxHelper 아웃라인. 액센트 색은 생성자
//   opts로 주입받는다 — render 계층은 ui/theme을 import하지 않는다(CLAUDE.md §3).
//   기본값 0xe67e22는 ui/theme COLOR.accent와 같은 값을 "의도적으로" 중복 정의한 것.
// - Transform 기즈모: three/examples TransformControls(three 패키지 내장 — 신규 런타임
//   의존성 아님). 이동/회전/스케일 모드(setMode + 키보드 W/E/R, 입력 필드 타이핑 중엔
//   무시), 스냅(0.05 m / 15°), 드래그 중 OrbitControls 비활성화(dragging-changed).
//   기즈모는 선택 루트에 직접 붙지 않고 **프록시 앵커 노드**에 붙는다 — 아래 참조.
//
// ── 기즈모 프록시 앵커 (로봇도 오브젝트처럼 잡히게 하는 핵심) ────────
// TransformControls는 attach한 객체의 **원점**에 핸들을 그린다. 프리미티브는 원점이
// 메시 중심이라 문제가 없지만, URDF 로봇의 시각 루트(outer Group) 원점은 베이스 링크
// = 바닥(y=0)이다. 그래서 화면에서 로봇 몸통 위에 그려질 핸들이 발밑에 그려지고
// (실측 73~201 px 이격), 사용자가 보이는 몸통을 드래그하면 핸들 유효 반경(~70 px)
// 밖이라 pointerdown이 OrbitControls로 흘러 **카메라만 돌고 로봇은 안 움직인다**.
// → 씬에 상주하는 빈 Object3D(앵커)를 선택 대상의 **시각 AABB 중심**(= 선택 아웃라인
//   BoxHelper가 그리는 박스의 중심)에 놓고 기즈모를 거기 붙인다. 앵커 자리는 드래그
//   밖에서 매 프레임 갱신한다 — 재생 중 FK로 몸이 펴져도 핸들이 몸통을 따라간다.
//
// 앵커 → 루트 전달 규칙 (**델타 전달**, 궤도 회전 금지):
//   pos:   rootWorld = rootStart + (anchorNow − anchorStart)   ← 드래그 시작 시점 기준
//   rot/scale: 앵커 값을 그대로 복사 (드래그 시작 시 앵커 = 루트의 월드 회전/스케일)
// 즉 "핸들만 다른 자리에 그린" 것과 같고, attach(root)였을 때의 의미를 정확히 보존한다.
// ★ 앵커를 피벗으로 삼아 루트를 역산하면(root = anchor − R(anchorRot)·offset) **회전만
//   해도 루트가 앵커를 중심으로 공전**해 바닥에 서 있던 로봇이 최대 0.7 m 떠오른다
//   (three r169 TransformControls는 rotate 모드에서 object.quaternion만 건드리고
//   position은 그대로 두므로, attach(root)에서는 위치가 절대 변하지 않았다).
//   회전 피벗은 루트 원점(= 로봇 베이스)이어야 한다 — 바닥에 고정된 팔의 자연스러운
//   피벗이기도 하다. 이 델타 규칙이 그 계약을 코드로 고정한다.
// commit 페이로드(= 루트의 월드 트랜스폼) 계약은 그대로여서 통합자(main.ts)·SceneEditor는
// 앵커의 존재를 알 필요가 없다.
// - 바닥 레이캐스트: raycastGround(clientX, clientY) → y=0 평면 교점 [x, 0, z]
//   (라이브러리 카드 드롭 배치용 — UX_DESIGN §3.2/§4.2).
//
// ── 물리와의 관계 (불변식 §2.1과의 정합) ─────────────────────────────
// 기즈모는 드래그 "중"에만 시각 Object3D를 직접 움직인다 — 순수 시각 프리뷰이며
// 물리 진실을 건드리지 않는다(예외: 로봇 루트는 살아있는 FK 그래프라, playing 중이면
// preStep의 tickAll이 드래그 중간 pose를 kinematic 바디로 push한다 — 그래서 통합자는
// 드래그 시작 시 엔진을 일시정지한다, 아래 참조). 드래그가 끝나면 onTransformCommit
// (id, { mode, position, rotation, scale })로 보고한다. commit pose는 라이브
// matrixWorld가 아니라 **드래그 이벤트(objectChange)마다 캡처한 최신 pose**다 —
// RenderSync.apply가 rAF마다 바인딩된 노드의 position/quaternion을 물리 pose로
// 덮어쓰므로, pointerup 시점의 matrixWorld는 이미 물리 pose로 리셋돼 있을 수 있다
// (덮어쓰기 경합 — 캡처 스냅샷은 이 경합에 면역이다). 움직임이 없던 드래그(기즈모
// 축 위 단순 클릭)는 commit을 발행하지 않는다 — 보간된 렌더 pose가 편집 경로로
// 물리에 역류하는 no-op teleport를 만들지 않는다.
// 통합자가 commit을 SceneEditor.updateTransform(물리 teleport, scene-edit-types.ts)
// 으로 라우팅하면 다음 sync에서 시각이 물리 진실로 재수렴한다. 드래그 프리뷰가
// RenderSync 덮어쓰기와 싸우지 않도록, 통합자는 onDraggingChanged로 드래그 동안
// 대상 바디의 sync 바인딩을 해제하고 커밋 후 재바인딩한다(dragging=false 통지는
// commit 발행 "이후"에 온다 — 재바인딩이 teleport된 pose를 스냅샷하는 순서 보장).
// playing 중 드래그 시작은 통합자가 엔진을 일시정지한다(main.ts 편집 정책).
// - scale 모드: 원시 scale 배율을 그대로 보고한다(commit.mode === 'scale'로 구분).
//   프리미티브는 통합자가 scale을 치수(Dimensions) 편집으로 변환하고(UX §3.3 "스케일
//   vs 정밀 치수"), 비프리미티브(URDF 로봇·임포트 메시)는 통합자가 commit을 거부하고
//   시각을 원복할 때까지 시각 전용 효과다.
//
// ── 통합 방법 (main.ts 글루) ─────────────────────────────────────────
//   const vi = new ViewportInteraction({ renderer, domElement, orbitControls }, opts);
//   vi.setPickables(map);            // 씬 빌드 후: entityId → 시각 루트(로봇은 outer Group,
//                                    //   프리미티브는 mesh — 모두 씬 루트의 직접 자식)
//   vi.onSelect(...); vi.onTransformCommit(...);
//   engine.onTick(() => vi.update()); // 매 프레임: BoxHelper가 선택 대상을 따라간다
//   vi.dispose();                     // 씬 teardown 시
// 주의: Renderer는 OrbitControls·캔버스(webgl.domElement)를 공개하지 않으므로 통합자가
// domElement(캔버스)와 orbitControls를 직접 주입해야 한다 — Renderer에 getter를 추가
// 하는 것이 권장 경로다(frictions 보고 참조).
//
// ── 하이라이트 상호작용 주의 ─────────────────────────────────────────
// highlight.ts(pulseEntity)도 emissive를 일시 변조한다. 펄스는 시작 시점 emissive를
// 스냅샷해 복원하므로, 선택 틴트 위에 펄스가 겹치면 펄스 종료 후 "틴트된" 값으로
// 복원된다(선택 중이면 올바른 값). 선택 해제가 펄스보다 먼저면 틴트 복원 후 펄스가
// 자기 스냅샷으로 되돌리는 짧은 잔상이 있을 수 있으나 ~600ms 내 자연 소멸하는 시각
// 전용 현상이다 — 물리·데이터에는 영향 없다.

import * as THREE from 'three';
import { TransformControls } from 'three/examples/jsm/controls/TransformControls.js';
import type { Quat, Vec3 } from '../schema/types';
import type { Renderer } from './renderer';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4) ─────────────────────────────

/** 이동 스냅 격자 (m) — UX_DESIGN §3.3 "스냅" */
export const TRANSLATION_SNAP_M = 0.05;
/** 회전 스냅 각도 (deg) */
export const ROTATION_SNAP_DEG = 15;
/** 회전 스냅 각도 (rad) — TransformControls.rotationSnap에 들어가는 값 */
export const ROTATION_SNAP_RAD = (ROTATION_SNAP_DEG * Math.PI) / 180;

/** 방향키 1회 이동 거리 (m) — 스냅 격자와 같게 맞춰 반복 이동이 격자에 정렬된다 */
export const NUDGE_STEP_M = TRANSLATION_SNAP_M;
/** Shift 병용 시 미세 이동 거리 (m) */
export const NUDGE_FINE_STEP_M = 0.01;

/** 클릭 판정 최대 이동 거리 (px) — 이보다 크면 orbit 드래그로 보고 선택을 바꾸지 않는다 */
const DEFAULT_CLICK_MAX_DISTANCE_PX = 5;
/**
 * no-op 드래그 판정 오차 — 시작·종료 트랜스폼의 성분별 차가 전부 이 이하이면
 * "움직임 없음"으로 보고 commit을 발행하지 않는다 (파일 헤더 "물리와의 관계").
 */
export const COMMIT_MIN_DELTA = 1e-6;
/** 선택 시 emissive를 액센트 색으로 끌어올리는 보간 비율 (0..1) */
const DEFAULT_SELECT_EMISSIVE_TINT = 0.35;
/**
 * 클릭 재조준 여유 반경 (px) — 메시를 살짝 빗나간 클릭을 이 반경 안에서만 보정한다.
 * 얇은 로봇 링크를 노렸을 때의 몇 px 오차를 흡수하되, 그 밖은 빈 곳(선택 해제)이다.
 */
export const CLICK_TOLERANCE_PX = 6;
/** 여유 반경에서 재조준할 방향 수 (8방위) */
const CLICK_TOLERANCE_SAMPLES = 8;
/** 기본 액센트 색 — ui/theme COLOR.accent(#e67e22)와 동일 값 (import 금지로 중복 정의) */
const DEFAULT_ACCENT_COLOR_HEX = 0xe67e22;
/** 바닥 평면 레이캐스트에서 시선이 평면과 평행하다고 볼 임계값 */
const RAY_PARALLEL_EPS = 1e-9;

// ── 순수 헬퍼 (three/DOM 비의존 — node vitest 대상, interaction.test.ts) ──

export type GizmoMode = 'translate' | 'rotate' | 'scale';

/** 클라이언트 좌표 → NDC 변환에 필요한 뷰포트 사각형 (DOMRect의 구조적 부분집합) */
export interface NdcRect {
  left: number;
  top: number;
  width: number;
  height: number;
}

/**
 * 클라이언트(px) 좌표 → NDC [-1, 1]. y는 화면 아래가 -1이 되도록 반전한다.
 * 크기가 0 이하인 퇴화 사각형이면 null (division-by-zero 방어).
 */
export function clientToNdc(
  clientX: number,
  clientY: number,
  rect: NdcRect,
): [number, number] | null {
  if (rect.width <= 0 || rect.height <= 0) return null;
  const x = ((clientX - rect.left) / rect.width) * 2 - 1;
  const y = -(((clientY - rect.top) / rect.height) * 2 - 1);
  return [x, y];
}

/** NDC [-1, 1] → 클라이언트(px) 좌표 — clientToNdc의 역 (기즈모 핸들 화면 위치 계산) */
export function ndcToClient(
  ndcX: number,
  ndcY: number,
  rect: NdcRect,
): [number, number] | null {
  if (rect.width <= 0 || rect.height <= 0) return null;
  return [
    rect.left + ((ndcX + 1) / 2) * rect.width,
    rect.top + ((1 - ndcY) / 2) * rect.height,
  ];
}

/**
 * 클릭 재조준 오프셋(px) — 반경 radiusPx의 원 위 samples개 방위.
 * 메시 raycast가 빗나갔을 때만 쓰는 화면 좌표 여유이므로, 반경 밖은 여전히 "빈 곳"이다.
 * 반경/개수가 유효하지 않으면 빈 배열(= 여유 없음, 정확 판정만).
 */
export function clickToleranceOffsets(
  radiusPx: number,
  samples: number = CLICK_TOLERANCE_SAMPLES,
): Array<[number, number]> {
  if (!(radiusPx > 0) || !Number.isFinite(radiusPx) || samples <= 0) return [];
  const out: Array<[number, number]> = [];
  for (let i = 0; i < samples; i += 1) {
    const angleRad = (2 * Math.PI * i) / samples;
    out.push([Math.cos(angleRad) * radiusPx, Math.sin(angleRad) * radiusPx]);
  }
  return out;
}

/**
 * 값을 step 격자의 최근접 배수로 스냅한다. step이 0 이하이거나 value가 유한하지
 * 않으면 원값을 그대로 돌려준다 (스냅 비활성 경로와 동일 의미).
 */
export function snapToStep(value: number, step: number): number {
  if (!(step > 0) || !Number.isFinite(value)) return value;
  return Math.round(value / step) * step;
}

/**
 * 원점 origin, 방향 dir인 광선이 y=0 평면과 만나는 파라미터 t (≥ 0).
 * 평면과 평행(|dirY| < eps)하거나 교점이 광선 뒤쪽(t < 0)이면 null.
 */
export function rayGroundT(
  originY: number,
  dirY: number,
  eps: number = RAY_PARALLEL_EPS,
): number | null {
  if (Math.abs(dirY) < eps) return null;
  const t = -originY / dirY;
  return t >= 0 ? t : null;
}

/** 광선(origin, dir 튜플)과 y=0 평면의 교점. y 성분은 정확히 0으로 고정한다. */
export function rayGroundPoint(origin: Readonly<Vec3>, dir: Readonly<Vec3>): Vec3 | null {
  const t = rayGroundT(origin[1], dir[1]);
  if (t === null) return null;
  return [origin[0] + dir[0] * t, 0, origin[2] + dir[2] * t];
}

/**
 * 방향키 이동 축.
 * - `right`: 카메라 기준 좌우 (←/→)
 * - `forward`: 카메라 기준 앞뒤 (↑/↓)
 * - `vertical`: 월드 Y (PageUp/PageDown)
 */
export interface NudgeAxis {
  kind: 'right' | 'forward' | 'vertical';
  /** +1 또는 -1 */
  sign: 1 | -1;
}

/**
 * 키보드 키 → 이동 축 (순수 함수 — 단위 테스트 대상).
 * 방향키는 카메라 기준 수평, PageUp/PageDown은 월드 수직.
 */
export function keyToNudgeAxis(key: string): NudgeAxis | null {
  switch (key) {
    case 'ArrowRight':
      return { kind: 'right', sign: 1 };
    case 'ArrowLeft':
      return { kind: 'right', sign: -1 };
    case 'ArrowUp':
      return { kind: 'forward', sign: 1 };
    case 'ArrowDown':
      return { kind: 'forward', sign: -1 };
    case 'PageUp':
      return { kind: 'vertical', sign: 1 };
    case 'PageDown':
      return { kind: 'vertical', sign: -1 };
    default:
      return null;
  }
}

/** 키보드 키 → 기즈모 모드 (W/E/R, 대소문자 무관 — UX_DESIGN §3.3/§9) */
export function keyToGizmoMode(key: string): GizmoMode | null {
  switch (key.toLowerCase()) {
    case 'w':
      return 'translate';
    case 'e':
      return 'rotate';
    case 'r':
      return 'scale';
    default:
      return null;
  }
}

/** 이벤트 타깃의 "타이핑 중" 판정에 필요한 구조적 부분집합 (EventTarget 캐스트용) */
export interface TypingTargetLike {
  tagName?: string;
  isContentEditable?: boolean;
}

/** 입력 필드/편집 가능 요소에 포커스가 있으면 true — 단축키(W/E/R)를 무시해야 한다 */
export function isTypingTarget(target: TypingTargetLike | null): boolean {
  if (target === null) return false;
  const tag = target.tagName?.toUpperCase();
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return true;
  return target.isContentEditable === true;
}

/** 월드 트랜스폼 스냅샷 튜플 묶음 (드래그 캡처·no-op 판정 공용) */
export interface TransformSnapshot {
  position: Vec3;
  rotation: Quat;
  scale: Vec3;
}

/**
 * 두 트랜스폼 스냅샷이 오차 내 동일한지 — no-op 드래그 commit 스킵 판정.
 * 회전은 q와 -q가 같은 회전이므로 부호 반전도 동일로 본다.
 */
export function transformsAlmostEqual(
  a: TransformSnapshot,
  b: TransformSnapshot,
  eps: number = COMMIT_MIN_DELTA,
): boolean {
  const vec3Eq = (x: Readonly<Vec3>, y: Readonly<Vec3>): boolean =>
    Math.abs(x[0] - y[0]) <= eps &&
    Math.abs(x[1] - y[1]) <= eps &&
    Math.abs(x[2] - y[2]) <= eps;
  const quatEq = (x: Readonly<Quat>, y: Readonly<Quat>, sign: 1 | -1): boolean =>
    Math.abs(x[0] - sign * y[0]) <= eps &&
    Math.abs(x[1] - sign * y[1]) <= eps &&
    Math.abs(x[2] - sign * y[2]) <= eps &&
    Math.abs(x[3] - sign * y[3]) <= eps;
  return (
    vec3Eq(a.position, b.position) &&
    vec3Eq(a.scale, b.scale) &&
    (quatEq(a.rotation, b.rotation, 1) || quatEq(a.rotation, b.rotation, -1))
  );
}

// ── 기즈모 프록시 앵커 수학 (순수 — three 비의존, interaction.test.ts 대상) ──

/**
 * 드래그 중 앵커의 이동량을 루트에 그대로 전달한 월드 위치
 * (파일 헤더 "앵커 → 루트 전달 규칙" — 앵커를 피벗으로 한 역산이 아니다).
 *
 * 회전/스케일 모드에서 TransformControls는 앵커의 position을 건드리지 않으므로
 * 델타가 0 → 루트 위치도 그대로다. 즉 **회전은 루트 원점(로봇 베이스)을 피벗**으로
 * 하고, "핸들만 몸통 위에 그린" 효과만 남는다.
 */
export function rootPositionFromAnchorDrag(
  rootStartPosition: Readonly<Vec3>,
  anchorStartPosition: Readonly<Vec3>,
  anchorPosition: Readonly<Vec3>,
): Vec3 {
  return [
    rootStartPosition[0] + (anchorPosition[0] - anchorStartPosition[0]),
    rootStartPosition[1] + (anchorPosition[1] - anchorStartPosition[1]),
    rootStartPosition[2] + (anchorPosition[2] - anchorStartPosition[2]),
  ];
}

/** pointerdown→up 이동 거리가 임계값 이하면 "클릭"으로 판정 (orbit 드래그 배제) */
export function withinClickThreshold(
  downX: number,
  downY: number,
  upX: number,
  upY: number,
  maxDistancePx: number,
): boolean {
  const dx = upX - downX;
  const dy = upY - downY;
  return dx * dx + dy * dy <= maxDistancePx * maxDistancePx;
}

// ── 공개 타입 ───────────────────────────────────────────────────────

/** 드래그 종료 시 보고되는 오브젝트 루트의 월드 트랜스폼 (튜플 — 프로젝트 표준 표현) */
export interface TransformCommit {
  /** 어떤 기즈모 모드의 결과인지 — scale은 통합자가 치수 편집으로 변환/거부 */
  mode: GizmoMode;
  position: Vec3;
  /** [x, y, z, w] (CLAUDE.md §4) */
  rotation: Quat;
  /** 원시 scale 배율 (기본 [1,1,1]) — 물리 collider에는 직접 적용되지 않는다 */
  scale: Vec3;
}

/**
 * 방향키 이동이 조용히 무시된 이유 — 통합자가 사람이 읽을 문구로 표면화한다.
 * (render 계층은 UI 문구를 갖지 않는다: 사유만 방출하고 번역은 ui 몫 — CLAUDE.md §3)
 */
export type NudgeBlockedReason = 'no-selection';

/** OrbitControls에서 필요한 최소 표면 — 드래그 중 비활성화 용도뿐 */
export interface OrbitControlsLike {
  enabled: boolean;
}

export interface ViewportInteractionDeps {
  /** scene·camera 접근 (public readonly 표면만 사용) */
  renderer: Renderer;
  /**
   * 포인터 이벤트·TransformControls를 붙일 캔버스 요소. Renderer가 webgl.domElement를
   * 공개하지 않으므로 통합자가 주입한다 (host.querySelector('canvas') 또는 Renderer에
   * getter 추가 — frictions 참조).
   */
  domElement: HTMLElement;
  /** 기즈모 드래그 중 비활성화할 orbit 컨트롤 (Renderer.controls가 private — 주입) */
  orbitControls?: OrbitControlsLike;
}

export interface ViewportInteractionOptions {
  /** 선택 아웃라인(BoxHelper)·emissive 틴트 색 (기본: ui/theme accent와 동일 값) */
  accentColorHex?: number;
  /** 선택 emissive 틴트 강도 0..1 (기본 0.35) */
  emissiveTintStrength?: number;
  /** 클릭 판정 최대 이동 거리 px (기본 5) */
  clickMaxDistancePx?: number;
}

// ── 내부 타입/스크래치 ──────────────────────────────────────────────

/** emissive를 가진 재질 (MeshStandardMaterial 등) — highlight.ts와 동일한 구조 판정 */
interface EmissiveMaterial extends THREE.Material {
  emissive: THREE.Color;
}

interface TintedMaterial {
  readonly material: EmissiveMaterial;
  readonly originalEmissiveHex: number;
}

/** 프레임/이벤트 핫 패스 재사용 버퍼 (신규 할당 최소화 — highlight.ts 관례) */
const _ndc = new THREE.Vector2();
const _worldPos = new THREE.Vector3();
const _worldQuat = new THREE.Quaternion();
const _worldScale = new THREE.Vector3();
/** 방향키 이동 계산용 재사용 버퍼 (할당 없는 키 핸들러) */
const _nudgeForward = new THREE.Vector3();
const _nudgeRight = new THREE.Vector3();
const _WORLD_UP = new THREE.Vector3(0, 1, 0);
const _tintScratch = new THREE.Color();
const _accentScratch = new THREE.Color();
/** 앵커 AABB 계산 버퍼 (선택 대상 1개 — BoxHelper.update와 같은 비용의 프레임 작업) */
const _selectionBox = new THREE.Box3();
const _boxCenter = new THREE.Vector3();
/** 드래그 중 루트의 목표 월드 행렬 → 부모 로컬 변환 버퍼 */
const _targetMatrix = new THREE.Matrix4();
const _parentInverse = new THREE.Matrix4();
const _scratchPos = new THREE.Vector3();
const _scratchQuat = new THREE.Quaternion();
const _scratchScale = new THREE.Vector3();

function hasEmissive(material: THREE.Material): material is EmissiveMaterial {
  return (material as Partial<EmissiveMaterial>).emissive instanceof THREE.Color;
}

/** 서브트리의 emissive 재질을 중복 없이 수집 (공유 재질 1회 — 이중 복원 방지) */
function collectEmissiveMaterials(root: THREE.Object3D): TintedMaterial[] {
  const seen = new Set<THREE.Material>();
  const out: TintedMaterial[] = [];
  root.traverse((obj) => {
    const mesh = obj as Partial<THREE.Mesh>;
    if (mesh.isMesh !== true || mesh.material === undefined) return;
    const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    for (const material of materials) {
      if (seen.has(material) || !hasEmissive(material)) continue;
      seen.add(material);
      out.push({ material, originalEmissiveHex: material.emissive.getHex() });
    }
  });
  return out;
}

// ── ViewportInteraction ─────────────────────────────────────────────

export class ViewportInteraction {
  private readonly scene: THREE.Scene;
  private readonly camera: THREE.Camera;
  private readonly domElement: HTMLElement;
  private readonly orbit: OrbitControlsLike | null;

  private readonly accentColorHex: number;
  private readonly emissiveTintStrength: number;
  private readonly clickMaxDistancePx: number;

  private readonly raycaster = new THREE.Raycaster();
  private readonly gizmo: TransformControls;
  /** getHelper()가 돌려주는 기즈모 시각 루트 — 씬에 add/remove하는 대상 (three r169+) */
  private readonly gizmoRoot: THREE.Object3D;
  /**
   * 기즈모 프록시 앵커 (파일 헤더 "기즈모 프록시 앵커"). 씬 루트의 직접 자식인 빈
   * Object3D — 렌더되지 않고 픽킹 대상도 아니다(pickables 밖). 기즈모는 항상 여기 붙는다.
   */
  private readonly gizmoAnchor = new THREE.Object3D();

  private pickables = new Map<string, THREE.Object3D>();
  private idByRoot = new Map<THREE.Object3D, string>();

  private currentId: string | null = null;
  private tinted: readonly TintedMaterial[] = [];
  private boxHelper: THREE.BoxHelper | null = null;

  private readonly selectListeners = new Set<(id: string | null) => void>();
  private readonly commitListeners = new Set<(id: string, commit: TransformCommit) => void>();
  private readonly dragListeners = new Set<(dragging: boolean, id: string | null) => void>();
  private readonly nudgeBlockedListeners = new Set<(reason: NudgeBlockedReason) => void>();

  /** 드래그 시작 시점 트랜스폼 (no-op 드래그 판정 기준 + 앵커 델타 전달의 기준점) */
  private dragStart: TransformSnapshot | null = null;
  /** 드래그 시작 시점 앵커의 월드 위치 — 이후 이동량만큼 루트를 옮긴다 (파일 헤더) */
  private dragAnchorStart: Vec3 | null = null;
  /**
   * 드래그 중 objectChange마다 캡처한 최신 트랜스폼 — commit의 진실.
   * 라이브 matrixWorld는 RenderSync.apply가 rAF마다 물리 pose로 덮어쓸 수 있어
   * pointerup 시점 decompose는 신뢰할 수 없다 (파일 헤더 "물리와의 관계").
   */
  private dragLatest: TransformSnapshot | null = null;

  private pointerDownX = 0;
  private pointerDownY = 0;
  private pointerDownArmed = false;
  private disposed = false;

  // 이벤트 핸들러는 add/removeEventListener 짝을 위해 인스턴스 필드로 고정한다
  private readonly handlePointerDown = (event: PointerEvent): void => {
    if (event.button !== 0) return;
    this.pointerDownArmed = true;
    this.pointerDownX = event.clientX;
    this.pointerDownY = event.clientY;
  };

  private readonly handlePointerUp = (event: PointerEvent): void => {
    if (event.button !== 0 || !this.pointerDownArmed) return;
    this.pointerDownArmed = false;
    // 기즈모 조작(드래그 직후 포함) 또는 기즈모 핸들 호버 중이면 선택을 바꾸지 않는다.
    // TransformControls의 pointerup이 먼저 돌아 dragging은 이미 false일 수 있으므로
    // axis(호버 중인 핸들)로도 판정한다.
    if (this.gizmo.dragging || this.gizmo.axis !== null) return;
    if (
      !withinClickThreshold(
        this.pointerDownX,
        this.pointerDownY,
        event.clientX,
        event.clientY,
        this.clickMaxDistancePx,
      )
    ) {
      return; // orbit 드래그 — 선택 유지
    }
    this.select(this.pickAt(event.clientX, event.clientY));
  };

  private readonly handleKeyDown = (event: KeyboardEvent): void => {
    // Ctrl+R(새로고침) 등 브라우저/앱 단축키를 가로채지 않는다
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    if (isTypingTarget(event.target as TypingTargetLike | null)) return;
    const mode = keyToGizmoMode(event.key);
    if (mode !== null) {
      this.setMode(mode);
      return;
    }
    if (this.nudgeByKey(event)) event.preventDefault(); // 방향키 페이지 스크롤 방지
  };

  /**
   * 방향키로 선택 오브젝트를 이동한다 (UX_DESIGN §3.3 "단축키로 배치 조정").
   *
   * - ←/→/↑/↓: **카메라 기준** 수평 이동 — 화면에서 보이는 방향과 일치해 직관적이다.
   * - PageUp/PageDown: 월드 Y(수직) 이동.
   * - Shift 병용: 미세 이동(1cm).
   *
   * 기즈모 드래그와 **같은 커밋 경로**(onTransformCommit)를 쓴다 — 통합자가 물리
   * teleport까지 동일하게 처리하므로 이동 수단에 따라 동작이 갈리지 않는다.
   * @returns 이동을 처리했으면 true
   */
  private nudgeByKey(event: KeyboardEvent): boolean {
    const axis = keyToNudgeAxis(event.key);
    if (axis === null) return false;
    if (this.currentId === null) {
      // 조용한 무시 금지: "왜 아무 일도 안 일어나는지"를 통합자가 알려야 한다.
      // (빈 곳 클릭으로 선택이 풀린 줄 모르는 사용자가 방향키만 계속 누르는 실패 연쇄)
      for (const fn of [...this.nudgeBlockedListeners]) fn('no-selection');
      return false;
    }
    const snapshot = this.decomposeSelectedRoot();
    if (snapshot === null) return false;

    const step = event.shiftKey ? NUDGE_FINE_STEP_M : NUDGE_STEP_M;
    const delta = this.nudgeDelta(axis, step);

    const position: Vec3 = [
      snapshot.position[0] + delta[0],
      snapshot.position[1] + delta[1],
      snapshot.position[2] + delta[2],
    ];
    // 커밋만 발행한다 — 시각 노드는 통합자의 물리 teleport → sync 경로로 갱신된다
    // (드래그와 동일. 여기서 노드를 직접 옮기면 물리와 어긋난다 — 불변식 §2.1)
    const commit: TransformCommit = {
      mode: 'translate',
      position,
      rotation: snapshot.rotation,
      scale: snapshot.scale,
    };
    for (const fn of this.commitListeners) fn(this.currentId, commit);
    return true;
  }

  /** 이동 축(카메라 기준 right/forward, 월드 up) → 월드 델타 벡터 */
  private nudgeDelta(axis: NudgeAxis, step: number): Vec3 {
    if (axis.kind === 'vertical') return [0, axis.sign * step, 0];

    // 카메라 기준 방향을 바닥 평면(y=0)에 투영한다 — 화면에서 보이는 대로 움직인다
    this.camera.getWorldDirection(_nudgeForward);
    _nudgeForward.y = 0;
    if (_nudgeForward.lengthSq() < 1e-6) {
      // 카메라가 수직으로 내려다보는 특수 상황 — 월드 축으로 폴백
      _nudgeForward.set(0, 0, -1);
    }
    _nudgeForward.normalize();

    if (axis.kind === 'forward') {
      return [_nudgeForward.x * axis.sign * step, 0, _nudgeForward.z * axis.sign * step];
    }
    // right = forward × up (Y-up 좌표계)
    _nudgeRight.copy(_nudgeForward).cross(_WORLD_UP).normalize();
    return [_nudgeRight.x * axis.sign * step, 0, _nudgeRight.z * axis.sign * step];
  }

  private readonly handleDraggingChanged = (event: { value: unknown }): void => {
    const dragging = event.value === true;
    // CRITICAL: 기즈모 드래그 중 카메라 orbit이 함께 돌지 않게 잠근다 (UX §3.3)
    if (this.orbit !== null) this.orbit.enabled = !dragging;
    if (dragging) {
      this.dragStart = this.decomposeSelectedRoot();
      // 앵커의 시작 자리를 고정한다 — 이후 objectChange마다 "여기서 얼마나 움직였나"만
      // 루트에 전달한다(파일 헤더의 델타 규칙). TransformControls는 pointerDown에서
      // 객체를 아직 건드리지 않으므로 이 시점 값이 곧 드래그 시작 pose다.
      this.gizmoAnchor.updateMatrixWorld(true);
      this.gizmoAnchor.matrixWorld.decompose(_scratchPos, _scratchQuat, _scratchScale);
      this.dragAnchorStart = [_scratchPos.x, _scratchPos.y, _scratchPos.z];
      this.dragLatest = null;
      this.notifyDragging(true);
    } else {
      this.emitCommit(); // 드래그 종료 = commit 시점 (no-op 드래그는 스킵)
      this.dragStart = null;
      this.dragAnchorStart = null;
      this.dragLatest = null;
      // commit(teleport) "이후" 통지 — 통합자의 sync 재바인딩 순서 보장 (파일 헤더)
      this.notifyDragging(false);
    }
  };

  /**
   * 드래그 중 기즈모가 **앵커**를 움직일 때마다 그 pose를 선택 루트로 역변환해 적용하고
   * (프리뷰), 루트의 월드 트랜스폼을 최신 pose로 캡처한다 (파일 헤더 "기즈모 프록시 앵커").
   */
  private readonly handleObjectChange = (): void => {
    if (!this.gizmo.dragging) return;
    this.applyAnchorToRoot();
    const snapshot = this.decomposeSelectedRoot();
    if (snapshot !== null) this.dragLatest = snapshot;
  };

  constructor(deps: ViewportInteractionDeps, opts: ViewportInteractionOptions = {}) {
    this.scene = deps.renderer.scene;
    this.camera = deps.renderer.camera;
    this.domElement = deps.domElement;
    this.orbit = deps.orbitControls ?? null;

    this.accentColorHex = opts.accentColorHex ?? DEFAULT_ACCENT_COLOR_HEX;
    this.emissiveTintStrength = opts.emissiveTintStrength ?? DEFAULT_SELECT_EMISSIVE_TINT;
    this.clickMaxDistancePx = opts.clickMaxDistancePx ?? DEFAULT_CLICK_MAX_DISTANCE_PX;

    this.gizmoAnchor.name = 'gizmo-anchor';
    this.scene.add(this.gizmoAnchor);

    this.gizmo = new TransformControls(this.camera, this.domElement);
    this.gizmoRoot = this.gizmo.getHelper();
    this.scene.add(this.gizmoRoot);
    this.gizmo.addEventListener('dragging-changed', this.handleDraggingChanged);
    this.gizmo.addEventListener('objectChange', this.handleObjectChange);
    this.setSnap(false); // 기본: 스냅 꺼짐 (통합자가 UI 토글로 켠다)

    this.domElement.addEventListener('pointerdown', this.handlePointerDown);
    this.domElement.addEventListener('pointerup', this.handlePointerUp);
    window.addEventListener('keydown', this.handleKeyDown);
  }

  // ── 픽킹 대상 ─────────────────────────────────────────────────────

  /**
   * entityId → 시각 루트 매핑을 교체한다 (씬 빌드/재빌드 후 통합자가 공급 — 로봇은
   * outer Group, 프리미티브는 mesh). 현재 선택이 새 매핑에 없으면 해제(리스너 통지),
   * 같은 id가 새 루트로 바뀌었으면 하이라이트·기즈모를 조용히 재부착한다.
   */
  setPickables(map: Map<string, THREE.Object3D>): void {
    this.pickables = new Map(map);
    this.idByRoot = new Map();
    for (const [id, root] of this.pickables) this.idByRoot.set(root, id);

    const id = this.currentId;
    if (id === null) return;
    const root = this.pickables.get(id);
    if (root === undefined) {
      this.select(null);
    } else {
      this.clearSelectionVisuals();
      this.applySelectionVisuals(root);
    }
  }

  // ── 선택 ─────────────────────────────────────────────────────────

  /** 현재 선택된 엔티티 id (없으면 null) */
  get selectedId(): string | null {
    return this.currentId;
  }

  /** 선택 변경 구독 — 해제 함수 반환. 클릭·select() 호출 모두에서 발화한다. */
  onSelect(fn: (id: string | null) => void): () => void {
    this.selectListeners.add(fn);
    return () => this.selectListeners.delete(fn);
  }

  /**
   * 선택을 설정한다 (null = 해제). pickables에 없는 id는 null로 정규화된다 —
   * 씬에서 사라진 엔티티를 가리키는 stale 선택을 만들 수 없다.
   */
  select(id: string | null): void {
    const resolved = id !== null && this.pickables.has(id) ? id : null;
    if (resolved === this.currentId) return; // 변경 없음 (재선택 루프 방지)

    this.clearSelectionVisuals();
    this.currentId = resolved;
    if (resolved !== null) {
      const root = this.pickables.get(resolved);
      if (root !== undefined) this.applySelectionVisuals(root);
    }
    for (const fn of this.selectListeners) fn(resolved);
  }

  // ── 기즈모 ────────────────────────────────────────────────────────

  /** 기즈모 모드 전환 (이동/회전/스케일 — W/E/R 키와 동일 경로) */
  setMode(mode: GizmoMode): void {
    this.gizmo.setMode(mode);
  }

  /** 현재 기즈모 모드 */
  getMode(): GizmoMode {
    return this.gizmo.getMode();
  }

  /**
   * 검증용 스냅샷 (월드 좌표 튜플): 기즈모 앵커 · 선택 대상의 시각 AABB 중심 · 루트 원점
   * + 기즈모가 실제로 앵커에 붙어 있는지(attachedToAnchor). 선택/지오메트리가 없으면 null.
   *
   * "핸들이 보이는 몸통에 붙는가"는 화면 기하 문제여서 순수 단위 테스트로는 잡히지
   * 않는다 — 브라우저 게이트가 이 수치로 회귀를 감시한다(three 타입을 파사드로 흘리지
   * 않기 위해 튜플로 돌려준다 — CLAUDE.md §3/§7). 좌표만으로는 attach 대상이 바뀐
   * 회귀를 못 잡으므로(앵커 좌표는 그대로다) attachedToAnchor를 함께 싣는다.
   */
  anchorProbe(): {
    anchor: Vec3;
    visualCenter: Vec3;
    rootOrigin: Vec3;
    attachedToAnchor: boolean;
  } | null {
    const id = this.currentId;
    if (id === null) return null;
    const root = this.pickables.get(id);
    if (root === undefined) return null;
    root.updateMatrixWorld(true);
    _selectionBox.setFromObject(root);
    if (_selectionBox.isEmpty()) return null;
    _selectionBox.getCenter(_boxCenter);
    root.matrixWorld.decompose(_worldPos, _worldQuat, _worldScale);
    return {
      anchor: [this.gizmoAnchor.position.x, this.gizmoAnchor.position.y, this.gizmoAnchor.position.z],
      visualCenter: [_boxCenter.x, _boxCenter.y, _boxCenter.z],
      rootOrigin: [_worldPos.x, _worldPos.y, _worldPos.z],
      attachedToAnchor: this.gizmo.object === this.gizmoAnchor,
    };
  }

  /**
   * 검증용: 기즈모 핸들의 화면(클라이언트 px) 좌표 — 앵커의 월드 위치를 카메라로 투영.
   * 선택이 없거나 캔버스가 퇴화 크기면 null.
   *
   * 사용자가 실제로 하는 제스처("보이는 몸통을 끌기")를 브라우저 게이트가 합성 마우스로
   * 재현하려면 핸들이 화면 어디에 있는지 알아야 한다 — 그 한 점만 노출한다.
   */
  anchorScreenPoint(): [number, number] | null {
    if (this.currentId === null) return null;
    this.gizmoAnchor.updateMatrixWorld(true);
    _scratchPos.setFromMatrixPosition(this.gizmoAnchor.matrixWorld).project(this.camera);
    if (!Number.isFinite(_scratchPos.x) || !Number.isFinite(_scratchPos.y)) return null;
    return ndcToClient(_scratchPos.x, _scratchPos.y, this.domElement.getBoundingClientRect());
  }

  /** 스냅 토글: 이동 0.05 m / 회전 15° (상수 TRANSLATION_SNAP_M / ROTATION_SNAP_RAD) */
  setSnap(enabled: boolean): void {
    this.gizmo.setTranslationSnap(enabled ? TRANSLATION_SNAP_M : null);
    this.gizmo.setRotationSnap(enabled ? ROTATION_SNAP_RAD : null);
  }

  /**
   * 드래그 종료 commit 구독 — 해제 함수 반환. 통합자는 이 commit을
   * SceneEditor.updateTransform(물리 teleport)으로 라우팅한다(파일 헤더 참조).
   * 움직임이 없던 드래그(기즈모 축 위 단순 클릭)는 발행되지 않는다.
   */
  onTransformCommit(fn: (id: string, commit: TransformCommit) => void): () => void {
    this.commitListeners.add(fn);
    return () => this.commitListeners.delete(fn);
  }

  /**
   * 기즈모 드래그 시작/종료 구독 — 해제 함수 반환. 통합자 훅:
   * - dragging=true: playing이면 엔진 일시정지 + 대상 바디 sync 바인딩 해제(프리뷰가
   *   RenderSync 덮어쓰기와 싸우지 않게).
   * - dragging=false: commit 발행 "이후"에 통지된다 — teleport된 pose를 스냅샷하도록
   *   sync를 재바인딩하는 순서가 보장된다 (파일 헤더 "물리와의 관계").
   */
  onDraggingChanged(fn: (dragging: boolean, id: string | null) => void): () => void {
    this.dragListeners.add(fn);
    return () => this.dragListeners.delete(fn);
  }

  /**
   * 방향키 이동이 거부된 순간 구독 — 해제 함수 반환. 통합자가 한국어 안내로 표면화한다
   * (조용한 무시 금지). 키를 누를 때마다 발화하므로 통합자가 스로틀한다.
   */
  onNudgeBlocked(fn: (reason: NudgeBlockedReason) => void): () => void {
    this.nudgeBlockedListeners.add(fn);
    return () => this.nudgeBlockedListeners.delete(fn);
  }

  // ── 바닥 레이캐스트 (라이브러리 드롭 배치) ────────────────────────

  /** 클라이언트 좌표에서 카메라 광선을 쏴 y=0 바닥 평면 교점 [x, 0, z]를 구한다 */
  raycastGround(clientX: number, clientY: number): Vec3 | null {
    const ndc = clientToNdc(clientX, clientY, this.domElement.getBoundingClientRect());
    if (ndc === null) return null;
    this.raycaster.setFromCamera(_ndc.set(ndc[0], ndc[1]), this.camera);
    const { origin, direction } = this.raycaster.ray;
    return rayGroundPoint(
      [origin.x, origin.y, origin.z],
      [direction.x, direction.y, direction.z],
    );
  }

  // ── 프레임 갱신 / 해제 ────────────────────────────────────────────

  /**
   * 매 프레임 호출 (통합자: engine.onTick 또는 렌더 루프의 draw 직전) —
   * BoxHelper 아웃라인이 이동/애니메이션 중인 선택 대상을 따라가게 한다.
   */
  update(): void {
    if (this.boxHelper !== null && this.boxHelper.visible) this.boxHelper.update();
    // 기즈모 앵커도 선택 대상의 **현재** 시각 중심을 따라간다 — 물리 teleport로 루트가
    // 움직이거나 재생 중 FK로 팔이 펴져도 핸들이 보이는 몸통 위에 남는다(선택 시점에
    // 오프셋을 고정하면 재생 후 핸들이 몸에서 0.4 m 넘게 떨어지는 원래 결함이 재현된다).
    // 드래그 중에는 앵커가 진실이므로 건드리지 않는다.
    if (this.gizmo.dragging || this.currentId === null) return;
    const root = this.pickables.get(this.currentId);
    if (root !== undefined) this.placeAnchorAtVisualCenter(root);
  }

  /** 리스너·기즈모·헬퍼 해제 (씬 teardown 시 — 멱등) */
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;

    this.domElement.removeEventListener('pointerdown', this.handlePointerDown);
    this.domElement.removeEventListener('pointerup', this.handlePointerUp);
    window.removeEventListener('keydown', this.handleKeyDown);

    this.clearSelectionVisuals(); // emissive 원복 + 기즈모 detach
    this.currentId = null;

    this.gizmo.removeEventListener('dragging-changed', this.handleDraggingChanged);
    this.gizmo.removeEventListener('objectChange', this.handleObjectChange);
    this.scene.remove(this.gizmoRoot);
    this.scene.remove(this.gizmoAnchor); // 프록시 앵커도 씬에서 제거 (지오메트리 없음)
    // three r169의 TransformControls.dispose()는 Controls 리팩터링에서 남은
    // this.traverse 호출 때문에 TypeError를 던진다(업스트림 버그 — r170에서
    // this._root.traverse로 수정). 같은 의도를 안전하게 수행한다:
    // 포인터 리스너 해제(disconnect) + 헬퍼 서브트리 geometry/material 해제.
    this.gizmo.disconnect();
    this.gizmoRoot.traverse((child) => {
      const mesh = child as Partial<THREE.Mesh>;
      mesh.geometry?.dispose();
      const material = mesh.material;
      if (material === undefined) return;
      if (Array.isArray(material)) for (const m of material) m.dispose();
      else material.dispose();
    });

    if (this.boxHelper !== null) {
      this.scene.remove(this.boxHelper);
      this.boxHelper.dispose();
      this.boxHelper = null;
    }

    if (this.orbit !== null) this.orbit.enabled = true; // 드래그 중 dispose 방어
    this.dragStart = null;
    this.dragAnchorStart = null;
    this.dragLatest = null;
    this.selectListeners.clear();
    this.commitListeners.clear();
    this.dragListeners.clear();
    this.nudgeBlockedListeners.clear();
  }

  // ── 내부 구현 ─────────────────────────────────────────────────────

  /**
   * 클릭 지점의 엔티티 id — 정확 판정 후, 빗나가면 CLICK_TOLERANCE_PX 반경 8방위로
   * 한 번 더 재조준한다 (파일 헤더 "클릭 픽킹"). 여유 반경 밖은 빈 곳 = 선택 해제.
   */
  private pickAt(clientX: number, clientY: number): string | null {
    const direct = this.pickExact(clientX, clientY);
    if (direct !== null) return direct;
    for (const [dx, dy] of clickToleranceOffsets(CLICK_TOLERANCE_PX)) {
      const nearby = this.pickExact(clientX + dx, clientY + dy);
      if (nearby !== null) return nearby;
    }
    return null;
  }

  /** 클라이언트 좌표에서 pickables 재귀 raycast → 가장 가까운 히트의 엔티티 id */
  private pickExact(clientX: number, clientY: number): string | null {
    const ndc = clientToNdc(clientX, clientY, this.domElement.getBoundingClientRect());
    if (ndc === null) return null;
    this.raycaster.setFromCamera(_ndc.set(ndc[0], ndc[1]), this.camera);
    // pickables 서브트리만 검사 — 그리드/기즈모/BoxHelper는 자연히 배제된다
    const hits = this.raycaster.intersectObjects([...this.pickables.values()], true);
    for (const hit of hits) {
      const id = this.rootIdOf(hit.object);
      if (id !== null) return id;
    }
    return null;
  }

  /** 히트된 노드에서 조상 체인을 따라 pickable 루트의 엔티티 id를 찾는다 */
  private rootIdOf(obj: THREE.Object3D): string | null {
    let cursor: THREE.Object3D | null = obj;
    while (cursor !== null) {
      const id = this.idByRoot.get(cursor);
      if (id !== undefined) return id;
      cursor = cursor.parent;
    }
    return null;
  }

  /** 선택 시각(틴트 + 아웃라인) 적용 + 기즈모 부착 */
  private applySelectionVisuals(root: THREE.Object3D): void {
    const tinted = collectEmissiveMaterials(root);
    _accentScratch.setHex(this.accentColorHex);
    for (const entry of tinted) {
      _tintScratch
        .setHex(entry.originalEmissiveHex)
        .lerp(_accentScratch, this.emissiveTintStrength);
      entry.material.emissive.copy(_tintScratch);
    }
    this.tinted = tinted;

    if (this.boxHelper === null) {
      this.boxHelper = new THREE.BoxHelper(root, this.accentColorHex);
      this.scene.add(this.boxHelper);
    } else {
      this.boxHelper.setFromObject(root);
    }
    this.boxHelper.visible = true;

    // 기즈모는 루트 원점이 아니라 **시각 AABB 중심**(= 위 아웃라인 박스의 중심)에 붙는다.
    // 로봇처럼 루트 원점이 발밑인 대상에서도 핸들이 보이는 몸통 위에 온다 (파일 헤더).
    this.placeAnchorAtVisualCenter(root);
    this.gizmo.attach(this.gizmoAnchor);
  }

  /**
   * 앵커를 선택 루트의 시각 AABB 중심(= 선택 아웃라인 박스의 중심)으로 옮긴다.
   * 회전/스케일은 루트의 월드 값을 그대로 복사한다 — 드래그 시작 시 앵커와 루트의
   * 회전/스케일이 같아야 델타 전달 규칙(파일 헤더)이 성립한다.
   * 드래그 밖에서 매 프레임 호출된다: 재생 중 FK로 몸이 펴져도 핸들이 몸통을 따라간다.
   */
  private placeAnchorAtVisualCenter(root: THREE.Object3D): void {
    root.updateMatrixWorld(true);
    root.matrixWorld.decompose(_worldPos, _worldQuat, _worldScale);
    _selectionBox.setFromObject(root);
    // 지오메트리가 없으면 루트 원점 폴백 (빈 Group 등)
    if (_selectionBox.isEmpty()) _boxCenter.copy(_worldPos);
    else _selectionBox.getCenter(_boxCenter);
    this.gizmoAnchor.position.copy(_boxCenter);
    this.gizmoAnchor.quaternion.copy(_worldQuat);
    this.gizmoAnchor.scale.copy(_worldScale);
    this.gizmoAnchor.updateMatrixWorld(true);
  }

  /**
   * 드래그된 앵커 pose → 선택 루트에 적용 (파일 헤더 "앵커 → 루트 전달 규칙").
   * 위치는 **앵커의 이동량만** 전달하고(회전 피벗은 루트 원점), 회전/스케일은 앵커
   * 값을 그대로 쓴다. 루트가 씬 루트의 직접 자식이 아닌 경우도 부모 역행렬로 처리한다.
   */
  private applyAnchorToRoot(): void {
    const id = this.currentId;
    const dragStart = this.dragStart;
    const anchorStart = this.dragAnchorStart;
    if (id === null || dragStart === null || anchorStart === null) return;
    const root = this.pickables.get(id);
    if (root === undefined) return;

    this.gizmoAnchor.updateMatrixWorld(true);
    this.gizmoAnchor.matrixWorld.decompose(_scratchPos, _scratchQuat, _scratchScale);
    const rootWorld = rootPositionFromAnchorDrag(dragStart.position, anchorStart, [
      _scratchPos.x,
      _scratchPos.y,
      _scratchPos.z,
    ]);
    _targetMatrix.compose(
      _scratchPos.set(rootWorld[0], rootWorld[1], rootWorld[2]),
      _scratchQuat,
      _scratchScale,
    );
    const parent = root.parent;
    if (parent !== null) {
      parent.updateMatrixWorld(true);
      _targetMatrix.premultiply(_parentInverse.copy(parent.matrixWorld).invert());
    }
    _targetMatrix.decompose(root.position, root.quaternion, root.scale);
    root.updateMatrixWorld(true);
  }

  /** 선택 시각 원복 + 기즈모 분리 (선택 해제·교체·dispose 공용 경로) */
  private clearSelectionVisuals(): void {
    for (const entry of this.tinted) {
      entry.material.emissive.setHex(entry.originalEmissiveHex);
    }
    this.tinted = [];
    if (this.boxHelper !== null) this.boxHelper.visible = false;
    this.gizmo.detach();
  }

  /** 선택 루트의 월드 트랜스폼 스냅샷 (선택/루트가 없으면 null) */
  private decomposeSelectedRoot(): TransformSnapshot | null {
    const id = this.currentId;
    if (id === null) return null;
    const root = this.pickables.get(id);
    if (root === undefined) return null;
    root.updateMatrixWorld(true);
    root.matrixWorld.decompose(_worldPos, _worldQuat, _worldScale);
    return {
      position: [_worldPos.x, _worldPos.y, _worldPos.z],
      rotation: [_worldQuat.x, _worldQuat.y, _worldQuat.z, _worldQuat.w],
      scale: [_worldScale.x, _worldScale.y, _worldScale.z],
    };
  }

  private notifyDragging(dragging: boolean): void {
    // 순회 중 구독 해제가 안전하도록 스냅샷 배열로 통지
    for (const fn of [...this.dragListeners]) fn(dragging, this.currentId);
  }

  /**
   * 드래그 종료 시 드래그 중 캡처된 최신 트랜스폼을 튜플로 발행.
   * objectChange가 한 번도 없었거나(순수 클릭) 시작 트랜스폼과 오차 내 동일하면
   * 발행하지 않는다 — no-op teleport·불필요한 undo 스냅샷 방지 (파일 헤더).
   */
  private emitCommit(): void {
    const id = this.currentId;
    const latest = this.dragLatest;
    if (id === null || latest === null) return;
    if (!this.pickables.has(id)) return;
    if (this.dragStart !== null && transformsAlmostEqual(this.dragStart, latest)) return;

    const commit: TransformCommit = {
      mode: this.getMode(),
      position: latest.position,
      rotation: latest.rotation,
      scale: latest.scale,
    };
    for (const fn of this.commitListeners) fn(id, commit);
  }
}
