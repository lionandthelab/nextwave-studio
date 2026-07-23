// main.ts — 부트스트랩 진입점 + 씬 라이프사이클 (Phase 1–7: 데이터 주도 씬 + URDF 로봇
// + 충돌/시퀀스 UI + 런타임 씬 전환 + Scene Builder 워크스페이스)
//
// 순서 고정 (docs/ARCHITECTURE.md §4, CLAUDE.md §2.7):
//   1. await initPhysics()      — WASM 로드 완료 전 물리 API 호출 금지
//   2. SceneSpec JSON 검증      — 실패 시 사람이 읽을 수 있는 오류 오버레이 (DATA_MODEL §8)
//   3. RapierWorld 생성         — spec.gravity / spec.timestepHz (+ MeshAssetStore resolver)
//   4. Renderer 생성/재사용     — spec.camera / spec.environment 반영
//   5. await SceneLoader.build  — 바디 + 메시 + URDF 로봇 생성, sync/robot 바인딩
//   6. 시퀀스 검증(있으면)      — 검증 실패 시 실행에 노출하지 않는다 (불변식 §2.9)
//   7. Engine 루프 시작 — preStep: player.step → robots.tickAll (ARCHITECTURE §5 ①),
//      onContacts: CollisionMonitor.dispatch (③)
//
// main은 조립 글루다: core(엔진·월드·로더·편집기·player·monitor)와 render(three)·
// ui(워크스페이스/라이브러리/커맨드바/독/우측 패널 스택)를 여기서 잇는다.
// scene-loader가 요구하는 RenderSceneApi의 three 구현(loadRobot·addMeshAsset 포함)도
// 여기서 제공한다 (core는 three를 모른 채 이 좁은 인터페이스만 호출한다 — CLAUDE.md §3).
// 부트 씬 선택은 ?scene= 쿼리 파라미터로 한다 — 씬도 시퀀스도 데이터다 (§2.5/§2.6).
//
// ── 워크스페이스 셸 (Phase 7, UX_DESIGN §2) ─────────────────────────
// mountWorkspace가 커맨드바/좌 라이브러리/중앙 뷰포트/우 스택/하단 독 슬롯을 CSS grid로
// 배치한다. 렌더러 호스트(#app)는 뷰포트 슬롯으로 reparent되고(Renderer는 리사이즈 시
// host.clientWidth/Height를 다시 읽는다 — workspace.notifyResize 계약), 기존 fixed
// 오버레이 패널(커맨드바·독·우측 스택·뷰포트 상태선)은 슬롯 흐름(static)으로 편입된다.
//
// ── 씬 편집 (Phase 7 Scene Builder) ─────────────────────────────────
// 씬마다 SceneEditorImpl을 SceneLoader.build와 **같은** world/sync/renderApi/robots/
// builtEntities 인스턴스 위에 세운다 — 편집과 로드가 같은 엔티티 빌드 루틴을 공유한다.
// 편집 경로: Library 드래그/클릭 → 바닥 레이캐스트 → addEntity / ViewportInteraction
// 기즈모 commit → updateTransform(이동·회전) 또는 updateDimensions(스케일→치수 변환,
// 오브젝트 스케일은 1로 리셋 — "기즈모 스케일은 빠른 조정, 진실은 치수" UX §3.3) /
// 엔티티 편집 폼 → update*/rename. 모든 편집은 SceneEditor.onChange로 통지되어
// 픽킹 맵·인스펙터·히스토리가 재동기화된다.
//
// ── Undo/Redo (UX §7) ───────────────────────────────────────────────
// ui/history.ts의 스냅샷 스택(serialize() 디바운스 기록, cap 50). 복원은 **전체 씬
// 재로드**(loadScene fromHistory) — 정확성 우선. 재생 중 복원은 engine.stop 후 진행
// (한국어 토스트). 시퀀스는 원본 JSON을 새 스펙에 재검증해 유효할 때만 유지된다.
// 로봇 URDF 재로드는 브라우저 캐시로 흡수된다 (EXPERIMENTS 기록).
//
// ── 임포트 에셋 (세션 한정) ──────────────────────────────────────────
// MeshAssetStore는 앱 수명 단일 인스턴스 — 씬 전환/undo 재로드를 가로질러 asset://
// ref가 계속 해석된다. RapierWorld ctor에 resolver로 주입되어 convexHull/trimesh
// collider를 만들고, RenderSceneApi.addMeshAsset이 시각 clone을 만든다. clone은
// geometry/material을 원형과 공유하므로 제거 시 disposeMeshResources를 호출하지
// 않는다(저장소 clear()가 일괄 해제 — mesh-import.ts MeshAssetStore 계약).
// 씬 저장(💾) 시 asset:// 참조가 있으면 세션 한정 한계를 경고한다.
//
// ── 시퀀스 재생 정책 (human-in-the-loop) ─────────────────────────────
// 씬 레지스트리에 시퀀스가 선언된 씬이라도 시퀀스를 자동 재생하지 않는다 — 검증을
// 통과한 시퀀스는 "로드 가능" 상태로만 두고, 사용자가 ▶ Play를 눌러야 player에
// 로드/시작된다(불변식 §2.9의 원칙을 플래너 이전 단계부터 적용). 물리 루프 자체는
// 씬 로드 직후 시작한다(낙하 등 씬 자체 물리는 재생 컨트롤과 무관하게 관찰 가능해야 함
// — 기존 falling-boxes 게이트 계약 유지).

import { CollisionMonitor } from './core/collision';
import {
  collisionQueryFromMonitor,
  robotApiFromRegistry,
} from './core/control/adapters';
import { ControlPlayer } from './core/control/player';
import type { PlayerStatus } from './core/control/player';
import { Engine, ENGINE_SPEED_OPTIONS } from './core/engine';
import type { EngineSpeed, EngineState } from './core/engine';
import { GROUND_ENTITY_ID, SceneLoader } from './core/scene-loader';
import type { RenderSceneApi, RobotHandle, SceneHandle, VisualNode } from './core/scene-loader';
import { SceneEditorImpl } from './core/scene-editor';
import type { SceneEditor } from './core/scene-edit-types';
import type { JointInfo } from './core/robot-types';
import { RenderSync } from './core/sync';
import type { PhysicsWorld, Pose } from './core/types';
import { initPhysics, RapierWorld } from './core/world';
import { pulseEntity } from './render/highlight';
import {
  isTypingTarget,
  ROTATION_SNAP_DEG,
  TRANSLATION_SNAP_M,
  ViewportInteraction,
} from './render/interaction';
import type { GizmoMode, TransformCommit, TypingTargetLike } from './render/interaction';
import {
  ASSET_SAVE_WARNING_KO,
  collectAssetRefs,
  MeshAssetStore,
  SUPPORTED_IMPORT_EXTENSIONS,
} from './render/mesh-import';
import { disposeMeshResources, groundMesh, primitiveMesh } from './render/meshes';
import { Renderer } from './render/renderer';
import { loadUrdfRobot } from './render/urdf';
import { mountJsonViewer } from './ui/command-bar/json-viewer';
import { mountPlaybackBar } from './ui/command-bar/playback';
import {
  mountCommandBarShell,
  mountSceneControls,
} from './ui/command-bar/scene-controls';
import type { SceneSwitchResult } from './ui/command-bar/scene-controls';
import { createCollisionLogPanel } from './ui/dock/collision-log';
import { appLog, createConsolePanel } from './ui/dock/console-panel';
import { mountDock } from './ui/dock/dock';
import { createTimelinePanel } from './ui/dock/timeline';
import { SceneHistory } from './ui/history';
import { DIMENSION_MIN_M, mountEntityEditor } from './ui/inspector/entity-editor';
import { mountInspector } from './ui/inspector/inspector';
import type { InspectorHandle } from './ui/inspector/inspector';
import { mountJointPanel } from './ui/inspector/joint-panel';
import { mountImportDialog } from './ui/library/import-dialog';
import { mountLibrary, TEMPLATE_MIME } from './ui/library/library';
import { templateByKey } from './ui/library/templates';
import { mountViewportStatus } from './ui/viewport/statusline';
import { mountWorkspace } from './ui/workspace';
import {
  COLOR,
  FONT,
  LAYOUT,
  RADIUS,
  SHADOW,
  SPACE,
  Z_INDEX,
  makeButton,
} from './ui/theme';
import { isRobotSpec, validateScene, validateSequence } from './schema';
import type {
  ColliderShape,
  CollisionEvent,
  ControlSequence,
  EntitySpec,
  PhysicsSpec,
  SceneSpec,
  Transform,
  Vec3,
} from './schema';
import fallingBoxesSceneJson from './assets/scenes/falling-boxes.scene.json';
import armAndBoxesSceneJson from './assets/scenes/arm-and-boxes.scene.json';
import pickAndPlaceSceneJson from './assets/scenes/pick-and-place.scene.json';
import obstacleAvoidanceSceneJson from './assets/scenes/obstacle-avoidance.scene.json';
import collisionTestbedSceneJson from './assets/scenes/collision-testbed.scene.json';
import armTouchBoxSequenceJson from './assets/sequences/arm-touch-box.sequence.json';
import pickAndPlaceSequenceJson from './assets/sequences/pick-and-place.sequence.json';
import obstacleAvoidanceSequenceJson from './assets/sequences/obstacle-avoidance.sequence.json';

// ── 씬 레지스트리 (?scene= 파라미터/프리셋 select로 선택 — 새 씬 = 새 데이터) ──
// 시퀀스가 선언된 씬은 검증 후 ▶ Play로만 재생된다(파일 헤더의 재생 정책).
// 새 프리셋 추가 = 아래에 항목 한 줄 append (scene-controls의 select에 자동 반영).

interface SceneRegistryEntry {
  readonly scene: unknown;
  /** 이 씬에서 재생할 ControlSequence JSON (선택) */
  readonly sequence?: unknown;
}

const SCENE_REGISTRY: Readonly<Record<string, SceneRegistryEntry>> = {
  'falling-boxes': { scene: fallingBoxesSceneJson },
  'arm-and-boxes': { scene: armAndBoxesSceneJson, sequence: armTouchBoxSequenceJson },
  // Phase 6 샘플 씬 3종 (ROADMAP Phase 6, PRD §7) — 전부 데이터 전용 (CLAUDE.md §2.5/§2.6)
  'pick-and-place': { scene: pickAndPlaceSceneJson, sequence: pickAndPlaceSequenceJson },
  'obstacle-avoidance': {
    scene: obstacleAvoidanceSceneJson,
    sequence: obstacleAvoidanceSequenceJson,
  },
  'collision-testbed': { scene: collisionTestbedSceneJson },
};

const DEFAULT_SCENE_NAME = 'arm-and-boxes';

/** 충돌 로그에서 최근 이벤트를 조회할 때의 기본 상한 (파사드 recent의 인자와 무관) */
const COLLISION_RECENT_DEFAULT_LIMIT = 50;

/** 우측 패널 스택 내 패널 사이 간격 (워크스페이스 우 슬롯 안 — UX_DESIGN §2 우측 존) */
const RIGHT_STACK_GAP_PX = 8;
/** 우측 패널 스택 안쪽 여백 */
const RIGHT_STACK_PADDING_PX = 8;
/** 인스펙터 값 갱신 스로틀 주기 (playing 중 — inspector.ts 헤더의 "주기 결정권은 통합자") */
const INSPECTOR_REFRESH_INTERVAL_MS = 150;
/** 앱 토스트 자동 숨김 (ms) */
const TOAST_AUTO_HIDE_MS = 5000;
/** 스케일 기즈모 → 치수 변환 시 half 치수 하한 (entity-editor 전체 치수 하한의 절반) */
const MIN_HALF_DIMENSION_M = DIMENSION_MIN_M / 2;

// ── 자동화/AI-native 훅 (Playwright 게이트 · 추후 ui 계층이 사용) ────

/**
 * 로봇 조작용 좁은 파사드 — 게이트/임시 UI가 필요로 하는 표면만 함수로 감싼
 * "편의 표면"이다. 주의: SimHandle.sceneHandle(공개 core API)을 통해서도
 * sceneHandle.robots(RobotRegistry → RobotBinding)에 도달할 수 있다 — Engine
 * preStep 훅·자동화를 위한 의도된 노출이다. 게이트/UI 코드는 core 내부 표면에
 * 의존이 자라지 않도록 가능하면 이 파사드만 사용할 것.
 */
export interface SimRobotsFacade {
  /** 등록된 로봇 엔티티 id 목록 */
  ids(): string[];
  /** 유효 limits(URDF ∩ override)가 반영된 관절 목록 */
  joints(robotId: string): readonly JointInfo[];
  /** 관절 목표값 설정 (limits 클램프는 core가 수행) */
  setJoint(robotId: string, jointName: string, value: number): void;
  /** 현재 관절값 스냅샷 (키: URDF joint명) */
  readJoints(robotId: string): Record<string, number>;
  /** 링크 kinematic 바디들의 현재 "물리" pose — 물리가 진실 (CLAUDE.md §2.1) */
  linkPoses(robotId: string): Pose[];
}

/** 시퀀스 재생 파사드 (씬에 검증된 시퀀스가 있는 경우에만 노출) */
export interface SimPlayerFacade {
  /** player 상태 — 시퀀스가 아직 Play로 로드되지 않았으면 'idle' */
  readonly status: PlayerStatus;
  readonly currentStepIndex: number;
  readonly stepCount: number;
  /** ▶ Play와 동일: 최초 호출 시 검증된 시퀀스를 로드(arm)하고 엔진을 재생한다 */
  play(): void;
  pause(): void;
  /** ⏹ Stop과 동일: 엔진 정지 + 씬/player/충돌 이력 리셋 (결정론적 재생 준비) */
  stop(): void;
}

/** 충돌 이력 조회 파사드 (게이트/디버깅용 — 진실은 CollisionMonitor) */
export interface SimCollisionFacade {
  historyCount(): number;
  /** 최근 n건 (오래된 것 → 최신 순) */
  recent(n?: number): readonly CollisionEvent[];
}

/**
 * 씬 편집 파사드 (Phase 7 Scene Builder — 게이트/자동화용).
 * 진실은 SceneEditor(core/scene-edit-types.ts) — 이 파사드는 선택 상태(interaction)와
 * 라이브러리 템플릿 배치를 함께 엮은 얇은 표면이다.
 */
export interface SimEditorFacade {
  /** 현재 편집 스펙의 엔티티 id 목록 (환경 바닥 예약 엔티티 제외) */
  entityIds(): string[];
  /** 현재 편집 상태의 SceneSpec 깊은 복사본 */
  serialize(): SceneSpec;
  /** 엔티티 추가 + 선택 (robot이면 URDF 로드로 async) */
  addEntity(spec: EntitySpec): Promise<void>;
  /** 라이브러리 템플릿을 지정 위치에 추가 + 선택 — 발급된 씬-유일 id 반환 */
  placeTemplate(templateKey: string, position: Vec3): Promise<string>;
  updateTransform(id: string, transform: Transform): void;
  updateDimensions(id: string, shape: ColliderShape): void;
  updatePhysics(id: string, physics: PhysicsSpec): void;
  renameEntity(id: string, newId: string): void;
  removeEntity(id: string): void;
  /** 뷰포트 픽킹 대상 id 목록 (선택 가능 엔티티 — 바닥 제외) */
  pickableIds(): string[];
  selectedId(): string | null;
  select(id: string | null): void;
}

/** Undo/Redo 파사드 (앱 수명 SceneHistory 위의 얇은 표면) */
export interface SimHistoryFacade {
  undo(): Promise<boolean>;
  redo(): Promise<boolean>;
  readonly canUndo: boolean;
  readonly canRedo: boolean;
}

/** window.__sim으로 노출되는 시뮬 핸들. Rapier 타입은 새지 않는다(PhysicsWorld 경계). */
export interface SimHandle {
  readonly engine: Engine;
  readonly world: PhysicsWorld;
  readonly sceneHandle: SceneHandle;
  readonly spec: SceneSpec;
  readonly robots: SimRobotsFacade;
  readonly collision: SimCollisionFacade;
  readonly editor: SimEditorFacade;
  readonly history: SimHistoryFacade;
  readonly player?: SimPlayerFacade;
}

declare global {
  interface Window {
    /** 자동화 훅 (scripts/gate-browser.mjs가 검증에 사용) — 항상 "현재 활성 씬"을 가리킨다 */
    __sim?: SimHandle;
  }
}

// ── 오류 오버레이 (검증 실패·부트스트랩 실패 표시용, 한국어) ─────────

const OVERLAY_Z_INDEX = Z_INDEX.overlay;

function showErrorOverlay(title: string, lines: readonly string[]): void {
  const overlay = document.createElement('div');
  Object.assign(overlay.style, {
    position: 'fixed',
    inset: '0',
    zIndex: OVERLAY_Z_INDEX,
    background: 'rgba(12, 14, 18, 0.96)',
    color: '#e8eaed',
    fontFamily: 'ui-monospace, SFMono-Regular, Consolas, monospace',
    padding: '32px',
    overflow: 'auto',
    boxSizing: 'border-box',
  } satisfies Partial<CSSStyleDeclaration>);

  const heading = document.createElement('h1');
  heading.textContent = title;
  Object.assign(heading.style, {
    color: '#ff6b6b',
    fontSize: '18px',
    margin: '0 0 16px 0',
  } satisfies Partial<CSSStyleDeclaration>);
  overlay.appendChild(heading);

  const list = document.createElement('pre');
  // textContent 사용 — 오류 문자열을 마크업으로 해석하지 않는다
  list.textContent = lines.join('\n');
  Object.assign(list.style, {
    whiteSpace: 'pre-wrap',
    lineHeight: '1.7',
    fontSize: '13px',
    margin: '0',
  } satisfies Partial<CSSStyleDeclaration>);
  overlay.appendChild(list);

  document.body.appendChild(overlay);
}

// ── 순수 헬퍼 ───────────────────────────────────────────────────────

/**
 * 스케일 기즈모 배율 → 프리미티브 치수 변환 (UX §3.3 "스케일 vs 정밀 치수").
 * box는 축별 배율, sphere는 평균 배율, cylinder/capsule은 xz 평균(반지름)·y(높이).
 * 결과 half 치수는 MIN_HALF_DIMENSION_M로 하한 클램프한다(퇴화 collider 방지).
 * 변환 후 통합자가 updateDimensions로 커밋하면 엔티티가 스케일 1의 새 메시+collider로
 * 재생성된다 — 오브젝트 스케일은 항상 1로 되돌아간다(물리 collider에 스케일 없음).
 */
function scaleShape(shape: ColliderShape, scale: Vec3): ColliderShape {
  const sx = Math.abs(scale[0]);
  const sy = Math.abs(scale[1]);
  const sz = Math.abs(scale[2]);
  const half = (value: number): number => Math.max(value, MIN_HALF_DIMENSION_M);
  switch (shape.kind) {
    case 'box':
      return {
        kind: 'box',
        halfExtents: [
          half(shape.halfExtents[0] * sx),
          half(shape.halfExtents[1] * sy),
          half(shape.halfExtents[2] * sz),
        ],
      };
    case 'sphere':
      return { kind: 'sphere', radius: Math.max(shape.radius * ((sx + sy + sz) / 3), DIMENSION_MIN_M) };
    case 'cylinder':
      return {
        kind: 'cylinder',
        radius: Math.max(shape.radius * ((sx + sz) / 2), DIMENSION_MIN_M),
        halfHeight: half(shape.halfHeight * sy),
      };
    case 'capsule':
      return {
        kind: 'capsule',
        radius: Math.max(shape.radius * ((sx + sz) / 2), DIMENSION_MIN_M),
        halfHeight: half(shape.halfHeight * sy),
      };
    default:
      // convexHull/trimesh/fromVisual — 스케일 편집 미지원 (호출부에서 걸러진다)
      return shape;
  }
}

// ── 씬 라이프사이클 타입 ────────────────────────────────────────────

/** 씬 전환 요청 — 프리셋(레지스트리) 또는 업로드 JSON */
interface SceneLoadRequest {
  readonly scene: unknown;
  readonly sequence?: unknown;
}

interface SceneLoadOptions {
  /** true면 히스토리 복원 재로드 — 히스토리 스택을 리셋하지 않는다 */
  readonly fromHistory?: boolean;
}

type SceneLoadResult =
  | { readonly ok: true }
  | {
      readonly ok: false;
      /** validate: SceneSpec 검증 실패 · build: 빌드(URDF 로드 등) 실패 · busy: 전환 진행 중 */
      readonly stage: 'validate' | 'build' | 'busy';
      readonly errors: readonly string[];
    };

/** 활성 씬 1개 몫의 런타임 — dispose()가 이 씬의 모든 자원·구독·UI를 해제한다 */
interface ActiveScene {
  readonly spec: SceneSpec;
  /** 검증을 통과한 시퀀스 (없거나 무효면 null — 실행에 노출되지 않음, §2.9) */
  readonly validSequence: ControlSequence | null;
  /** 히스토리 재로드 시 새 스펙에 재검증할 원본 시퀀스 JSON */
  readonly sequenceJson: unknown;
  readonly editor: SceneEditor;
  readonly engine: Engine;
  /** idBase → 현재 편집 스펙 기준 씬-유일 id (라이브러리 배치용) */
  uniquifyId(base: string): string;
  /** 엔티티를 드롭 좌표(null이면 뷰포트 중앙)의 바닥 레이캐스트 지점에 추가 + 선택 */
  placeEntity(entity: EntitySpec, dropClient: { x: number; y: number } | null): Promise<string>;
  dispose(): void;
}

/**
 * 업로드 JSON 봉투 해석 (형식 규약은 scene-controls.ts 헤더):
 * (a) SceneSpec 단독, (b) { scene, sequence? } 봉투. 'scene' 키가 있으면 봉투로
 * 해석한다 — SceneSpec에는 'scene' 필드가 없으므로 두 형식은 모호하지 않다.
 */
function unwrapUploadEnvelope(payload: unknown): { scene: unknown; sequence?: unknown } {
  if (payload !== null && typeof payload === 'object' && 'scene' in payload) {
    const envelope = payload as { scene: unknown; sequence?: unknown };
    return { scene: envelope.scene, sequence: envelope.sequence };
  }
  return { scene: payload };
}

/** 프리셋 전환을 URL ?scene=에 반영(딥링크 공유용) — 업로드 씬은 파라미터를 지운다 */
function updateUrlSceneParam(presetName: string | null): void {
  const url = new URL(window.location.href);
  if (presetName !== null) url.searchParams.set('scene', presetName);
  else url.searchParams.delete('scene');
  window.history.replaceState(null, '', url);
}

// ── 부트스트랩 ──────────────────────────────────────────────────────

async function boot(): Promise<void> {
  const host = document.getElementById('app');
  if (!host) throw new Error('#app host element not found');

  // 부트 씬 선택: ?scene=<이름> (미지정 시 기본 씬). 알 수 없는 이름은 명확한 오류로 중단.
  const bootSceneName =
    new URLSearchParams(window.location.search).get('scene') ?? DEFAULT_SCENE_NAME;
  const bootEntry = SCENE_REGISTRY[bootSceneName];
  if (bootEntry === undefined) {
    showErrorOverlay('알 수 없는 씬', [
      `'${bootSceneName}' 씬을 찾을 수 없습니다.`,
      `사용 가능한 씬: ${Object.keys(SCENE_REGISTRY).join(', ')}`,
      `예: ?scene=${DEFAULT_SCENE_NAME}`,
    ]);
    return;
  }

  await initPhysics();
  console.log('Rapier ready');

  // ── 워크스페이스 셸 (UX_DESIGN §2 — 앱 수명) ──────────────────────
  // 렌더러 호스트(#app)를 뷰포트 슬롯으로 통째로 reparent한다 — Renderer는 리사이즈
  // 이벤트에서 host.clientWidth/Height를 다시 읽으므로(workspace.notifyResize가 스플리터
  // 드래그/접기 후 window resize를 합성 발행) 캔버스가 슬롯 크기를 따라온다.
  const workspace = mountWorkspace(document.body);
  workspace.slots.viewport.appendChild(host);
  Object.assign(host.style, {
    position: 'absolute',
    inset: '0',
    width: '100%',
    height: '100%',
  } satisfies Partial<CSSStyleDeclaration>);

  // ── 앱 수명 상태 (씬 전환을 가로질러 유지) ────────────────────────

  let renderer: Renderer | null = null;
  let active: ActiveScene | null = null;
  let switching = false;
  /** 임포트 3D 에셋 저장소 — 앱 수명 (씬 전환/undo 재로드에도 asset:// 해석 유지) */
  const assetStore = new MeshAssetStore();

  /** 렌더러는 캔버스째로 재사용하고, 씬별 카메라/환경 옵션만 재적용한다 */
  const ensureRenderer = (spec: SceneSpec): Renderer => {
    const opts = {
      skyColor: spec.environment?.skyColor,
      cameraPosition: spec.camera?.position,
      cameraTarget: spec.camera?.target,
      cameraFov: spec.camera?.fov,
    };
    if (renderer === null) renderer = new Renderer(host, opts);
    else renderer.applySceneOptions(opts);
    return renderer;
  };

  // ── 상단 커맨드바 셸 [좌: 씬 컨트롤·↶↷ | 중앙: 재생 | 우: {} JSON] ─
  // 씬 전환을 가로질러 유지된다 (UX_DESIGN §3.1 — 하나의 응집된 커맨드바).
  // fixed 오버레이 기본값을 슬롯 흐름으로 중화한다 (workspace 편입).

  const commandBar = mountCommandBarShell(workspace.slots.commandBar);
  Object.assign(commandBar.el.style, {
    position: 'static',
    top: 'auto',
    left: 'auto',
    right: 'auto',
  } satisfies Partial<CSSStyleDeclaration>);
  const jsonViewer = mountJsonViewer(
    commandBar.right,
    document.body,
    () => active?.validSequence ?? null,
  );

  // ── 앱 토스트 (히스토리/저장 경고 등 — 오류 토스트는 scene-controls 소유) ──

  const toastEl = document.createElement('div');
  Object.assign(toastEl.style, {
    position: 'fixed',
    top: `${LAYOUT.belowBarTopPx}px`,
    left: '50%',
    transform: 'translateX(-50%)',
    zIndex: Z_INDEX.toast,
    maxWidth: 'min(560px, 80vw)',
    background: COLOR.bgPanel,
    border: `1px solid ${COLOR.borderStrong}`,
    borderLeft: `3px solid ${COLOR.accent}`,
    borderRadius: RADIUS.md,
    boxShadow: SHADOW.panel,
    padding: `${SPACE.md} ${SPACE.lg}`,
    color: COLOR.text,
    fontFamily: FONT.ui,
    fontSize: '12px',
    lineHeight: '1.6',
    whiteSpace: 'pre-wrap',
    cursor: 'pointer',
    display: 'none',
  } satisfies Partial<CSSStyleDeclaration>);
  toastEl.dataset.testid = 'app-toast';
  toastEl.title = '클릭하여 닫기';
  document.body.appendChild(toastEl);
  let toastTimer: ReturnType<typeof setTimeout> | null = null;
  const hideToast = (): void => {
    toastEl.style.display = 'none';
    if (toastTimer !== null) {
      clearTimeout(toastTimer);
      toastTimer = null;
    }
  };
  toastEl.addEventListener('click', hideToast);
  const showToast = (message: string, kind: 'info' | 'warn'): void => {
    toastEl.style.borderLeftColor = kind === 'warn' ? COLOR.warn : COLOR.accent;
    toastEl.textContent = message;
    toastEl.style.display = 'block';
    if (toastTimer !== null) clearTimeout(toastTimer);
    toastTimer = setTimeout(hideToast, TOAST_AUTO_HIDE_MS);
  };

  // ── Undo/Redo (UX §7) — 앱 수명 SceneHistory + ↶↷ 버튼 + Ctrl/Cmd+Z ──

  const undoButton = makeButton('↶', '되돌리기 (Ctrl+Z)', 'history-undo');
  const redoButton = makeButton('↷', '다시하기 (Ctrl+Shift+Z)', 'history-redo');
  undoButton.disabled = true;
  redoButton.disabled = true;

  const history = new SceneHistory({
    restore: (spec) => restoreSceneFromHistory(spec),
    onStateChange: (canUndo, canRedo) => {
      undoButton.disabled = !canUndo;
      redoButton.disabled = !canRedo;
    },
  });
  undoButton.addEventListener('click', () => {
    void history.undo();
  });
  redoButton.addEventListener('click', () => {
    void history.redo();
  });
  window.addEventListener('keydown', (e: KeyboardEvent) => {
    if (!(e.ctrlKey || e.metaKey) || e.altKey) return;
    const key = e.key.toLowerCase();
    if (key !== 'z' && key !== 'y') return;
    if (isTypingTarget(e.target as TypingTargetLike | null)) return; // 입력 필드는 브라우저 기본 undo
    e.preventDefault();
    if (key === 'y' || e.shiftKey) void history.redo();
    else void history.undo();
  });

  const historyFacade: SimHistoryFacade = {
    undo: () => history.undo(),
    redo: () => history.redo(),
    get canUndo() {
      return history.canUndo;
    },
    get canRedo() {
      return history.canRedo;
    },
  };

  /**
   * 히스토리 스냅샷 복원 = 전체 씬 재로드 (검증 → teardown → 클린 빌드 — 결정론).
   * 재생 중이면 엔진을 먼저 정지한다(edit-during-play 정책, EXPERIMENTS 기록) —
   * 복원 결과가 재생 상태에 오염되지 않는다. 시퀀스는 현재 씬의 원본 JSON을
   * 새 스펙에 재검증해 유효할 때만 유지된다(무효 → 미로드 + 콘솔 경고).
   */
  async function restoreSceneFromHistory(spec: SceneSpec): Promise<boolean> {
    const scene = active;
    if (scene && scene.engine.state === 'playing') {
      scene.engine.stop();
      showToast('되돌리기: 시뮬 정지됨', 'info');
    }
    const result = await loadScene(
      { scene: spec, sequence: scene?.sequenceJson },
      { fromHistory: true },
    );
    if (!result.ok) {
      for (const error of result.errors) appLog('error', `히스토리 복원 실패: ${error}`);
      showToast('되돌리기/다시하기 실패 — Console 탭을 확인하세요', 'warn');
      return false;
    }
    appLog('info', '씬 히스토리 복원 완료 (전체 클린 재로드)');
    return true;
  }

  // ── 씬 1개 빌드 (검증된 spec → 월드/씬/에디터/엔진/UI — 파일 헤더의 3–7단계) ──

  async function buildScene(
    spec: SceneSpec,
    sequenceJson: unknown,
  ): Promise<ActiveScene> {
    const render = ensureRenderer(spec);
    // MeshAssetStore를 resolver로 주입 — convexHull/trimesh collider의 asset:// 해석 지점
    const world = new RapierWorld(spec.gravity, spec.timestepHz, assetStore);
    const sync = new RenderSync(world);

    // 로봇 핸들 → 시각 노드. RobotHandle은 three 노드를 노출하지 않으므로(경계 계약)
    // loadRobot 호출 전후의 씬 자식 차이로 캡처한다. 핸들 동일성 기준이라 씬 편집으로
    // 로봇이 추가/개명되어도(같은 핸들 재사용 — scene-editor rename 계약) 매핑이 유지된다.
    const robotNodeByHandle = new Map<RobotHandle, VisualNode>();

    // scene-loader(core)가 three를 모르도록, 좁은 RenderSceneApi를 여기서 구현해 주입
    const renderApi: RenderSceneApi = {
      addPrimitive: (shape, color) => {
        const mesh = primitiveMesh(shape, color);
        render.scene.add(mesh); // 씬 루트 직접 자식 — RenderSync 바인딩 계약
        return mesh;
      },
      addGround: () => {
        const mesh = groundMesh();
        render.scene.add(mesh);
        return mesh;
      },
      setPose: (node, position, rotation) => {
        node.position.set(position[0], position[1], position[2]);
        node.quaternion.set(rotation[0], rotation[1], rotation[2], rotation[3]);
      },
      remove: (node) => {
        node.removeFromParent();
        // 임포트 에셋 clone은 geometry/material을 저장소 원형과 "공유"한다 — dispose 금지
        // (다른 인스턴스/원형까지 파괴됨). GPU 자원 수명은 MeshAssetStore.clear() 소유
        // (render/mesh-import.ts MeshAssetStore 계약).
        if ((node.userData as Record<string, unknown>)['rswAssetRef'] !== undefined) return;
        // 씬 전환마다 GPU 버퍼가 GC 대기 상태로 쌓이지 않게 즉시 해제 —
        // RobotHandle.dispose(render/urdf.ts)가 URDF 메시에 하는 것과 대칭
        disposeMeshResources(node);
      },
      // Phase 7 임포트 에셋 시각 — 저장소 원형의 clone을 씬 루트 직접 자식으로 추가
      addMeshAsset: (ref) => {
        const proto = assetStore.getObject(ref);
        if (!proto) {
          throw new Error(
            `main: 메시 에셋 '${ref}'을(를) 해석할 수 없습니다 — ` +
              '이 세션에서 임포트한 에셋만 사용할 수 있습니다 (세션 한정 에셋)',
          );
        }
        const node = proto.clone(true);
        (node.userData as Record<string, unknown>)['rswAssetRef'] = ref;
        render.scene.add(node);
        return node;
      },
      // URDF 로봇 로드 — render/urdf.ts의 RobotHandle은 core의 RobotHandle 타입을
      // 구조적으로 만족한다 (three 심볼은 public 표면에 노출되지 않음)
      loadRobot: async (request) => {
        const childrenBefore = new Set(render.scene.children);
        const handle = await loadUrdfRobot(render.scene, {
          urdfPath: request.urdfPath,
          packages: request.packages,
        });
        const added = render.scene.children.find((c) => !childrenBefore.has(c));
        if (added) robotNodeByHandle.set(handle, added);
        return handle;
      },
    };

    let sceneHandle: SceneHandle;
    try {
      sceneHandle = await new SceneLoader(world, renderApi, sync).build(spec);
    } catch (err) {
      // 빌드 실패 — SceneLoader가 자기 몫의 부분 자원을 이미 정리했으므로 월드만 해제
      world.free();
      throw err;
    }

    // ── 씬 편집기 (Phase 7) — 로더와 같은 인스턴스 위에 세운다 ────────
    // builtEntities/robots를 공유하므로 편집 후에도 reset()/dispose()가 현재 상태를 본다.
    const editor: SceneEditor = new SceneEditorImpl({
      spec,
      world,
      sync,
      renderApi,
      robots: sceneHandle.robots,
      builtEntities: sceneHandle.builtEntities,
    });

    /** 엔티티의 현재 시각 노드 — 편집 이후에도 정확 (builtEntities가 살아있는 진실) */
    const visualNodeOf = (entityId: string): VisualNode | undefined => {
      const record = sceneHandle.builtEntities.get(entityId);
      if (!record) return undefined;
      if (record.node) return record.node;
      if (record.robot) return robotNodeByHandle.get(record.robot.handle);
      return undefined;
    };

    // ── 빌드 이후 조립 가드 ──────────────────────────────────────────
    // SceneLoader.build 이후의 조립(에디터 배선/모니터/플레이어/엔진/독/우측 스택)이
    // 중간에 던지면 이미 만들어진 몫만 teardown 순서 계약(EXPERIMENTS 2026-07-23:
    // halt → 구독 해제 → UI 제거 → 씬 자원 → sync → world)대로 되감고 재던진다.
    const built: {
      engine?: Engine;
      interaction?: ViewportInteraction;
      gizmoBar?: HTMLDivElement;
      timelinePanel?: ReturnType<typeof createTimelinePanel>;
      collisionPanel?: ReturnType<typeof createCollisionLogPanel>;
      consolePanel?: ReturnType<typeof createConsolePanel>;
      dock?: ReturnType<typeof mountDock>;
      offMonitor?: () => void;
      playbackBar?: ReturnType<typeof mountPlaybackBar>;
      viewportStatus?: ReturnType<typeof mountViewportStatus>;
      offStepChange?: () => void;
      offTick?: () => void;
      offEditorChange?: () => void;
      rightStack?: HTMLDivElement;
      jointPanel?: ReturnType<typeof mountJointPanel>;
      inspector?: InspectorHandle;
      entityEditor?: ReturnType<typeof mountEntityEditor>;
    } = {};

    // 이 씬 몫의 전부를 해제한다 — 다음 빌드에 어떤 상태도 새지 않는다(전체 클린 빌드).
    // 순서: 엔진 루프 완전 정지 → 구독 해제 → 상호작용(기즈모/하이라이트) 해제 →
    // UI 제거 → 씬 자원(물리 바디·시각 노드·로봇 핸들) 해제 → sync 바인딩 정리 →
    // 월드 free. 렌더러/캔버스·워크스페이스 셸은 유지된다.
    const teardownBuilt = (): void => {
      built.engine?.halt();
      built.offTick?.();
      built.offStepChange?.();
      built.offMonitor?.();
      built.offEditorChange?.();
      history.cancelPending(); // 해제될 editor를 캡처하는 pending 스냅샷 폐기
      built.interaction?.dispose(); // 씬 노드가 살아있는 동안 emissive 원복 + 기즈모 분리
      built.gizmoBar?.remove();
      built.playbackBar?.dispose();
      built.viewportStatus?.dispose();
      built.entityEditor?.dispose();
      built.inspector?.dispose();
      built.jointPanel?.dispose();
      built.rightStack?.remove();
      built.timelinePanel?.dispose();
      built.collisionPanel?.dispose();
      built.consolePanel?.dispose();
      built.dock?.dispose();
      sceneHandle.dispose();
      sync.clear();
      world.free();
      if (window.__sim && built.engine && window.__sim.engine === built.engine) {
        window.__sim = undefined;
      }
    };

    try {
      // ── 충돌 모니터 + 시퀀스 player (Phase 4/5 코어를 앱에 배선) ────

      const monitor = new CollisionMonitor();
      const robotApi = robotApiFromRegistry(sceneHandle.robots, (robotId) => {
        // 그리퍼 설정은 "현재 편집 스펙"에서 읽는다 — 로봇 rename/추가에도 정합
        const entity = editor.spec.entities.find((e) => e.id === robotId);
        return entity && isRobotSpec(entity) ? entity.gripper : undefined;
      });
      const player = new ControlPlayer({
        robots: robotApi,
        collision: collisionQueryFromMonitor(monitor),
        warn: (msg) => appLog('warn', msg),
      });

      // 시퀀스 검증 — 미검증/무효 시퀀스는 실행(player)에 노출하지 않는다 (불변식 §2.9)
      let validSequence: ControlSequence | null = null;
      if (sequenceJson !== undefined) {
        const sequenceValidation = validateSequence(sequenceJson, spec);
        if (sequenceValidation.ok) {
          validSequence = sequenceValidation.value;
          appLog(
            'info',
            `시퀀스 '${validSequence.id}' 검증 통과 (${validSequence.steps.length}개 step) — ▶ Play로 재생`,
          );
        } else {
          // 히스토리 복원으로 엔티티 참조가 깨진 경우 등 — 시퀀스는 언로드 상태로 남는다
          console.warn('Sequence validation failed:', sequenceValidation.errors);
          for (const error of sequenceValidation.errors) {
            appLog('error', `시퀀스 검증 실패: ${error}`);
          }
        }
      }

      const engine = new Engine(
        {
          world,
          sync,
          render,
          hooks: {
            // 매 물리 tick, world.step() 직전 (ARCHITECTURE §5 ①):
            // ① player가 관절 "상태"를 갱신하고 → ② robots가 FK를 kinematic 바디로 push.
            // player는 시퀀스 미로드 시 no-op — 순서 계약은 모든 씬에서 동일하다.
            preStep: (simTimeSec, dtSec) => {
              player.step(simTimeSec, dtSec);
              sceneHandle.robots.tickAll();
            },
            // 접촉 이벤트 발행 (ARCHITECTURE §5 ③) — 이력 기록 + UI 구독자 통지
            onContacts: (events, simTimeSec) => {
              monitor.dispatch(events, simTimeSec);
            },
          },
        },
        spec.timestepHz, // world와 동일한 timestepHz — 이중 소스 불일치 경고 방지
      );
      built.engine = engine;

      // ── 뷰포트 상호작용 (Phase 7): 선택·아웃라인·기즈모·바닥 레이캐스트 ──

      const interaction = new ViewportInteraction(
        {
          renderer: render,
          domElement: render.domElement,
          orbitControls: render.orbitControls,
        },
        {
          // 액센트 색은 조립 지점(여기)이 ui/theme 토큰에서 주입한다 — render 계층은
          // ui를 import하지 않는다(CLAUDE.md §3). interaction.ts의 동일 값 기본치는
          // 진짜 fallback으로만 남는다.
          accentColorHex: parseInt(COLOR.accent.slice(1), 16),
        },
      );
      built.interaction = interaction;

      /** 픽킹 맵 (entityId → 시각 루트) — 편집마다 builtEntities에서 재구축한다 */
      const pickables = new Map<string, VisualNode>();
      const rebuildPickables = (): void => {
        pickables.clear();
        for (const [id, record] of sceneHandle.builtEntities) {
          if (id === GROUND_ENTITY_ID) continue; // 바닥은 선택/기즈모 대상이 아니다
          const node =
            record.node ?? (record.robot ? robotNodeByHandle.get(record.robot.handle) : undefined);
          if (node) pickables.set(id, node);
        }
        interaction.setPickables(pickables);
      };
      rebuildPickables();

      /**
       * 살아있는 바디를 변형하는 편집(teleport/재빌드) 전 재생 자동 일시정지 —
       * core 계약(scene-editor.ts 헤더: 편집 연산은 정지/일시정지 상태에서만 호출)을
       * UI 경로에서 집행한다. restoreSceneFromHistory의 정지 정책과 같은 계열.
       * __sim.editor 파사드(게이트/자동화)는 의도적으로 이 게이트를 거치지 않는다 —
       * teleport는 속도를 0으로 초기화해 spec/initialPose 정합이 유지되므로 재생 중
       * 파사드 편집도 리플레이 결정론은 깨지 않는다 (EXPERIMENTS 기록).
       */
      const pauseForEditIfPlaying = (reasonKo: string): void => {
        if (engine.state !== 'playing') return;
        engine.pause();
        showToast(`${reasonKo}: 시뮬 일시정지됨 — ▶ Play로 재개`, 'info');
      };

      /**
       * 편집 연산 실행 + 실패 표면화. SceneEditor의 실패 복원 경로는 시각 노드를
       * 재생성할 수 있으므로(rebuildInPlace 복원) 실패 시에도 픽킹 맵을 재동기화한다.
       * 재생 중이면 먼저 자동 일시정지한다 (위 pauseForEditIfPlaying 정책).
       */
      const runEdit = (op: () => void): void => {
        pauseForEditIfPlaying('편집');
        try {
          op();
        } catch (err) {
          rebuildPickables();
          const msg = err instanceof Error ? err.message : String(err);
          appLog('error', msg);
          showToast(msg, 'warn');
        }
      };

      // 기즈모 모드/스냅 오버레이 (UX §3.3 — W/E/R 단축키는 interaction이 처리)
      const gizmoBar = document.createElement('div');
      Object.assign(gizmoBar.style, {
        position: 'absolute',
        top: '10px',
        left: '12px',
        display: 'flex',
        gap: SPACE.xs,
        zIndex: '5',
      } satisfies Partial<CSSStyleDeclaration>);
      gizmoBar.dataset.testid = 'gizmo-bar';
      const modeButtons: ReadonlyArray<{ mode: GizmoMode; button: HTMLButtonElement }> = [
        { mode: 'translate', button: makeButton('W 이동', '이동 기즈모 (단축키 W)', 'gizmo-translate') },
        { mode: 'rotate', button: makeButton('E 회전', '회전 기즈모 (단축키 E)', 'gizmo-rotate') },
        { mode: 'scale', button: makeButton('R 스케일', '스케일 기즈모 (단축키 R) — 프리미티브는 치수로 변환', 'gizmo-scale') },
      ];
      let paintedMode: GizmoMode | null = null;
      const paintGizmoBar = (): void => {
        const current = interaction.getMode();
        if (current === paintedMode) return;
        paintedMode = current;
        for (const { mode, button } of modeButtons) {
          button.classList.toggle('ui-btn--active', mode === current);
          button.setAttribute('aria-pressed', String(mode === current));
        }
      };
      for (const { mode, button } of modeButtons) {
        button.addEventListener('click', () => {
          interaction.setMode(mode);
          paintGizmoBar();
        });
        gizmoBar.appendChild(button);
      }
      let snapEnabled = false;
      const snapButton = makeButton(
        '스냅',
        `격자 스냅 토글 — 이동 ${TRANSLATION_SNAP_M} m · 회전 ${ROTATION_SNAP_DEG}°`,
        'gizmo-snap',
      );
      snapButton.setAttribute('aria-pressed', 'false');
      snapButton.addEventListener('click', () => {
        snapEnabled = !snapEnabled;
        interaction.setSnap(snapEnabled);
        snapButton.classList.toggle('ui-btn--active', snapEnabled);
        snapButton.setAttribute('aria-pressed', String(snapEnabled));
      });
      gizmoBar.appendChild(snapButton);
      workspace.slots.viewport.appendChild(gizmoBar);
      built.gizmoBar = gizmoBar;
      paintGizmoBar();

      // 기즈모 드래그 수명 훅 (render/interaction.ts 헤더 "물리와의 관계"):
      // - 시작: playing이면 일시정지(로봇 루트 드래그가 preStep tickAll로 물리에
      //   새는 것 방지) + 대상 바디의 sync 바인딩 해제(드래그 프리뷰가 RenderSync
      //   덮어쓰기와 싸우지 않게 — 프리뷰가 실제로 포인터를 따라온다).
      // - 종료: commit(teleport) "이후" 통지되므로 재바인딩의 prev 스냅샷이 곧
      //   teleport된 물리 pose다 — 다음 프레임부터 시각이 물리 진실로 재수렴.
      //   commit이 실패/스킵된 경우에도 재바인딩이 시각을 물리 pose로 되돌린다.
      interaction.onDraggingChanged((dragging, id) => {
        if (dragging) pauseForEditIfPlaying('기즈모 편집');
        if (id === null) return;
        const record = sceneHandle.builtEntities.get(id);
        if (!record || record.bodyId === undefined || !record.bound || !record.node) return;
        if (dragging) sync.unbind(record.bodyId);
        else sync.bind(record.bodyId, record.node);
      });

      // 기즈모 commit → SceneEditor 라우팅 (render/interaction.ts 헤더 계약):
      // 드래그 중은 순수 시각 프리뷰, commit 시점에 물리(teleport)로 정합된다.
      interaction.onTransformCommit((id, commit: TransformCommit) => {
        if (commit.mode === 'scale') {
          const entity = editor.spec.entities.find((e) => e.id === id);
          const primitive =
            entity && !isRobotSpec(entity) && entity.visual.kind === 'primitive'
              ? entity.visual.primitive
              : undefined;
          if (entity === undefined || primitive === undefined) {
            // 비프리미티브(로봇·임포트 메시): 스케일 커밋 거부 + 시각 스케일 원복 (UX §3.3)
            const node = pickables.get(id);
            if (node) node.scale.set(1, 1, 1);
            appLog('warn', `'${id}': 스케일 기즈모는 프리미티브 전용입니다 — 원복했습니다`);
            return;
          }
          // 스케일 배율 → 치수 편집으로 변환. updateDimensions가 엔티티를 스케일 1의
          // 새 메시+collider로 재생성하므로 드래그로 커진 시각 스케일은 자연히 소거된다.
          runEdit(() => editor.updateDimensions(id, scaleShape(primitive, commit.scale)));
          return;
        }
        runEdit(() =>
          editor.updateTransform(id, { position: commit.position, rotation: commit.rotation }),
        );
      });

      // ── UI: 하단 독 (Timeline | Collision Log | Console) — 워크스페이스 독 슬롯 ──

      const timelinePanel = createTimelinePanel();
      built.timelinePanel = timelinePanel;
      const collisionPanel = createCollisionLogPanel({
        onFocusEntity: (entityId) => {
          const node = visualNodeOf(entityId);
          if (node && entityId !== GROUND_ENTITY_ID) pulseEntity(node);
          // 카메라 포커스/당시 노드 강조는 Phase 10 (ROADMAP "Collision Log 연동")
          appLog('info', `충돌 로그: '${entityId}' 하이라이트 (카메라 포커스는 Phase 10)`);
        },
      });
      built.collisionPanel = collisionPanel;
      const consolePanel = createConsolePanel();
      built.consolePanel = consolePanel;
      built.dock = mountDock(workspace.slots.dock, [
        { label: 'Timeline', content: timelinePanel.el },
        { label: 'Collision Log', content: collisionPanel.el },
        { label: 'Console', content: consolePanel.el },
      ]);
      // 독의 fixed 오버레이 기본값을 슬롯 흐름으로 중화 (workspace 편입)
      Object.assign(built.dock.el.style, {
        position: 'static',
        left: 'auto',
        right: 'auto',
        bottom: 'auto',
        height: '100%',
      } satisfies Partial<CSSStyleDeclaration>);

      // 충돌 → 로그 패널 행 추가 + start 시 관련 오브젝트 빨강 펄스 (UX_DESIGN §3.3/§3.6)
      built.offMonitor = monitor.subscribe((e) => {
        collisionPanel.addEvent(e);
        if (e.phase !== 'start') return;
        for (const entityId of [e.a, e.b]) {
          if (entityId === GROUND_ENTITY_ID) continue; // 바닥 전체 펄스는 소음 — 제외
          const node = visualNodeOf(entityId);
          if (node) pulseEntity(node);
        }
      });

      // 인스펙터 핸들 — 아래 재생 컨트롤·onTick 클로저가 참조하므로 먼저 선언한다
      // (마운트는 우측 패널 스택 조립 시점 — window.__sim 배선 이후)
      let inspectorRef: InspectorHandle | null = null;
      let inspectorLastRefreshMs = 0;
      let inspectorLastEngineState: EngineState = 'idle';

      // 시퀀스 arm(최초 Play 시 player 로드) — 파일 헤더의 human-in-the-loop 정책
      let sequenceArmed = false;
      const armSequenceIfAvailable = (): void => {
        if (!validSequence || sequenceArmed) return;
        player.load(validSequence);
        sequenceArmed = true;
        appLog('info', `시퀀스 '${validSequence.id}' 재생 시작 (${validSequence.steps.length}개 step)`);
      };

      const playbackControls = {
        play: (): void => {
          armSequenceIfAvailable();
          engine.play();
        },
        pause: (): void => {
          engine.pause();
        },
        // 정지 = 결정론적 재생 준비: 엔진 시계 → 씬 pose → player 커서 → 충돌 이력 순.
        // monitor.clear()로 이전 mark가 모두 무효화되지만 player.reset()이 활성 런타임을
        // 폐기하므로 stale mark 소비자는 남지 않는다 (collision.ts clear 계약).
        stop: (): void => {
          engine.stop();
          sceneHandle.reset();
          if (sequenceArmed) player.reset();
          monitor.clear();
          collisionPanel.clear();
        },
        stepOnce: (): void => {
          engine.stepOnce();
          // 단일 스텝 결과를 인스펙터에 즉시 반영 (paused/idle 중 유일한 pose 변화 경로)
          inspectorRef?.refresh();
        },
        setSpeed: (speedMult: number): void => {
          // select 옵션은 ENGINE_SPEED_OPTIONS에서 생성되므로 항상 유효하다
          engine.setSpeed(speedMult as EngineSpeed);
        },
      };

      // 재생 컨트롤은 커맨드바 중앙 슬롯에 — 씬마다 재마운트한다 (속도 select 등
      // 뷰 상태가 씬을 가로질러 새지 않는다: 새 엔진의 기본 속도 1×와 표시가 일치)
      const playbackBar = mountPlaybackBar(
        commandBar.center,
        playbackControls,
        ENGINE_SPEED_OPTIONS,
      );
      built.playbackBar = playbackBar;

      // 뷰포트 좌하단 실행 오버레이 (UX_DESIGN §3.3) — 뷰포트 슬롯 안 absolute 배치.
      // 표시 전용(pointer-events 없음), 씬 수명과 함께 마운트/해제된다.
      const viewportStatus = mountViewportStatus(workspace.slots.viewport, {
        sceneName: spec.name,
        emptyScene: spec.entities.length === 0,
      });
      built.viewportStatus = viewportStatus;

      // 타임라인: 검증된 시퀀스의 step 마커 + player 커서 연동
      if (validSequence) {
        timelinePanel.setSequence(validSequence.steps.map((step) => step.kind));
        built.offStepChange = player.onStepChange((index) => {
          timelinePanel.setActiveIndex(index);
        });
      }

      // rAF당 1회: 재생 바 + 타임라인 리드아웃 + 상호작용 헬퍼 갱신 (물리 tick과 분리)
      built.offTick = engine.onTick((info) => {
        interaction.update(); // BoxHelper 아웃라인이 이동/애니메이션 중 선택 대상을 따라간다
        paintGizmoBar(); // W/E/R 단축키로 바뀐 모드를 버튼 상태에 반영 (변경 시에만 DOM)
        playbackBar.update({
          engineState: info.state,
          simTimeSec: info.simTimeSec,
          sequence: validSequence
            ? {
                // 엔진 idle에서는 armed 여부와 무관하게 대기 라벨을 보인다 — ⏹ Stop 후
                // player.reset()은 커서를 되감으며 'running'으로 두지만(ControlPlayer.reset
                // 계약) 엔진 tick이 없어 진행되지 않는 상태다. 'running' 표기는 오해를
                // 부르므로 실제 재개 수단(▶ Play)을 안내한다.
                status:
                  sequenceArmed && info.state !== 'idle' ? player.status : '대기 (▶ Play)',
                stepIndex: player.currentStepIndex,
                stepCount: player.stepCount,
              }
            : undefined,
        });
        timelinePanel.setSimTime(info.simTimeSec);
        viewportStatus.update({
          engineState: info.state,
          simTimeSec: info.simTimeSec,
          sequence: validSequence
            ? { stepIndex: player.currentStepIndex, stepCount: player.stepCount }
            : undefined,
        });

        // 인스펙터 값 갱신 정책: playing 중 ~150ms 스로틀 + 상태 전이(pause/stop) 시 1회
        // — 멈춘 화면이 항상 최신 물리 진실을 비추고, 재생 중 rAF마다 DOM을 다시 만들지
        // 않는다 (선택 시 1회 갱신은 onSelect에서, stepOnce는 재생 컨트롤에서 별도 호출)
        const inspectorStateChanged = info.state !== inspectorLastEngineState;
        inspectorLastEngineState = info.state;
        if (info.state === 'playing') {
          const nowMs = performance.now();
          if (nowMs - inspectorLastRefreshMs >= INSPECTOR_REFRESH_INTERVAL_MS) {
            inspectorLastRefreshMs = nowMs;
            inspectorRef?.refresh();
          }
        } else if (inspectorStateChanged) {
          inspectorRef?.refresh();
        }
      });

      // paused/idle 중 관절 변경(슬라이더·Home)도 시각 로봇에 즉시 반영한다 — tick()은
      // 시각 FK 갱신 + "다음 스텝" kinematic 목표 지정뿐이라 preStep 밖 호출이 무해하다
      // (다음 preStep의 tickAll이 동일 목표를 다시 push — 물리 스텝 결과·결정론에 영향
      // 없음). playing 중에는 매 물리 tick이 어차피 반영하므로 생략한다.
      const refreshRobotVisualWhenNotPlaying = (robotId: string): void => {
        if (engine.state !== 'playing') sceneHandle.robots.get(robotId).tick();
      };

      // 로봇 파사드 — core 내부 객체 대신 좁은 함수 표면만 노출 (게이트·임시 UI 공용)
      const robots: SimRobotsFacade = {
        ids: () => sceneHandle.robots.ids(),
        joints: (robotId) => sceneHandle.robots.get(robotId).joints,
        setJoint: (robotId, jointName, value) => {
          sceneHandle.robots.get(robotId).setJoint(jointName, value);
          refreshRobotVisualWhenNotPlaying(robotId);
        },
        readJoints: (robotId) => sceneHandle.robots.get(robotId).readJoints(),
        // 물리 바디 pose가 진실 — FK 캐시가 아니라 world에서 읽는다 (CLAUDE.md §2.1)
        linkPoses: (robotId) =>
          world.bodiesOfEntity(robotId).map((bodyId) => world.getPose(bodyId)),
      };

      const collisionFacade: SimCollisionFacade = {
        historyCount: () => monitor.history().length,
        recent: (n) => monitor.history({ limit: n ?? COLLISION_RECENT_DEFAULT_LIMIT }),
      };

      const armedSequence = validSequence; // 클로저용 non-null 별칭 (아래 파사드에서 사용)
      const playerFacade: SimPlayerFacade | undefined = armedSequence
        ? {
            get status() {
              return player.status;
            },
            get currentStepIndex() {
              return player.currentStepIndex;
            },
            get stepCount() {
              // Play 전(미로드)에도 검증된 시퀀스의 step 수를 보고한다 — "로드 가능" 상태 표면
              return player.loaded ? player.stepCount : armedSequence.steps.length;
            },
            play: playbackControls.play,
            pause: playbackControls.pause,
            stop: playbackControls.stop,
          }
        : undefined;

      // ── 씬 편집 글루: id 발급·배치·파사드 (Phase 7) ──────────────────

      /** idBase → 씬-유일 id ('box' → 'box_1'). 유일성 진실은 현재 편집 스펙이다. */
      const uniquifyId = (base: string): string => {
        const used = new Set(editor.spec.entities.map((e) => e.id));
        used.add(GROUND_ENTITY_ID);
        let n = 1;
        while (used.has(`${base}_${n}`)) n += 1;
        return `${base}_${n}`;
      };

      const viewportCenterClient = (): { x: number; y: number } => {
        const rect = workspace.slots.viewport.getBoundingClientRect();
        return { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 };
      };

      /** 엔티티 추가 + 선택 (실패는 한국어 오류 throw — 호출자가 표면화) */
      const addEntityAndSelect = async (entity: EntitySpec): Promise<string> => {
        await editor.addEntity(entity);
        interaction.select(entity.id); // onChange가 이미 픽킹 맵을 재구축한 뒤다
        return entity.id;
      };

      /** 드롭 좌표(null = 뷰포트 중앙)의 바닥 레이캐스트 지점에 배치 — y는 템플릿 유지 */
      const placeEntity = async (
        entity: EntitySpec,
        dropClient: { x: number; y: number } | null,
      ): Promise<string> => {
        const client = dropClient ?? viewportCenterClient();
        const ground = interaction.raycastGround(client.x, client.y);
        if (ground) {
          entity.transform.position = [ground[0], entity.transform.position[1], ground[2]];
        }
        return addEntityAndSelect(entity);
      };

      const editorFacade: SimEditorFacade = {
        entityIds: () => editor.spec.entities.map((e) => e.id),
        serialize: () => editor.serialize(),
        addEntity: async (entitySpec) => {
          await addEntityAndSelect(structuredClone(entitySpec));
        },
        placeTemplate: async (templateKey, position) => {
          const template = templateByKey(templateKey);
          if (!template) {
            throw new Error(`main: 라이브러리 템플릿 '${templateKey}'이(가) 없습니다`);
          }
          const entity = template.create(uniquifyId);
          entity.transform.position = [position[0], position[1], position[2]];
          return addEntityAndSelect(entity);
        },
        updateTransform: (id, transform) => editor.updateTransform(id, transform),
        updateDimensions: (id, shape) => editor.updateDimensions(id, shape),
        updatePhysics: (id, physics) => editor.updatePhysics(id, physics),
        renameEntity: (id, newId) => editor.renameEntity(id, newId),
        removeEntity: (id) => {
          // 안전 가드: 재생 중 로봇 제거는 player가 다음 tick에 사라진 로봇을 구동하려다
          // 죽을 수 있다 — 시퀀스가 arm된 상태면 정지(리셋) 후 제거한다 (편집 정책 §아래).
          if (sequenceArmed && sceneHandle.robots.ids().includes(id)) {
            playbackControls.stop();
            appLog('warn', `로봇 '${id}' 제거: 재생 중이던 시퀀스를 정지했습니다`);
          }
          editor.removeEntity(id);
        },
        pickableIds: () => [...pickables.keys()],
        selectedId: () => interaction.selectedId,
        select: (id) => {
          interaction.select(id);
        },
      };

      // 씬 전환 후 게이트/자동화가 보는 핸들은 항상 "이" 씬의 새 인스턴스들이다
      window.__sim = {
        engine,
        world,
        sceneHandle,
        spec,
        robots,
        collision: collisionFacade,
        editor: editorFacade,
        history: historyFacade,
        ...(playerFacade ? { player: playerFacade } : {}),
      };

      // ── 우측 패널 스택: 관절 패널 + 인스펙터 + 엔티티 편집 (UX_DESIGN §2 우측 존) ──
      // 워크스페이스 우 슬롯(스크롤 소유) 안의 세로 스택 — fixed 오버레이가 아니다.
      const rightStack = document.createElement('div');
      Object.assign(rightStack.style, {
        display: 'flex',
        flexDirection: 'column',
        gap: `${RIGHT_STACK_GAP_PX}px`,
        padding: `${RIGHT_STACK_PADDING_PX}px`,
        boxSizing: 'border-box',
        width: '100%',
      } satisfies Partial<CSSStyleDeclaration>);
      workspace.slots.rightStack.appendChild(rightStack);
      built.rightStack = rightStack;

      /** 패널을 스택 흐름(static)으로 편입 — 모듈 기본 절대 배치/자체 폭 제약을 해제 */
      const adoptIntoStack = (panelEl: HTMLElement): void => {
        Object.assign(panelEl.style, {
          position: 'static',
          top: 'auto',
          right: 'auto',
          width: '100%',
          minWidth: '0',
          maxHeight: 'none',
          minHeight: '0',
          flex: '0 1 auto',
        } satisfies Partial<CSSStyleDeclaration>);
      };

      // 로봇이 있는 씬이면 임시 관절 패널 마운트 (ROADMAP Phase 3 "슬라이더 수동 제어").
      // 주의: 씬 편집으로 나중에 추가된 로봇은 이 패널에 나타나지 않는다 — 관절 확인은
      // 인스펙터가, 정식 관절 편집 UI는 후속 Phase가 맡는다 (빌드 시점 스냅샷 UI).
      const jointPanel =
        sceneHandle.robots.ids().length > 0
          ? mountJointPanel(
              rightStack,
              sceneHandle.robots.ids().map((robotId) => ({
                id: robotId,
                joints: sceneHandle.robots.get(robotId).joints,
              })),
              {
                setJoint: robots.setJoint,
                readJoints: robots.readJoints,
                applyHome: (robotId) => {
                  sceneHandle.robots.get(robotId).applyHome();
                  refreshRobotVisualWhenNotPlaying(robotId);
                },
              },
            )
          : null;
      if (jointPanel) {
        built.jointPanel = jointPanel;
        adoptIntoStack(jointPanel.el);
      }

      // 인스펙터 (읽기: 엔티티 목록·선택·트랜스폼/관절 상태) — 목록은 편집 스펙 기준
      const inspector = mountInspector(rightStack, {
        // 편집의 진실은 editor.spec — 추가/삭제/개명이 즉시 목록에 반영된다.
        // environment.ground의 예약 엔티티(GROUND_ENTITY_ID)는 스펙에 없으므로 제외.
        listEntities: () => editor.spec.entities.map((e) => ({ id: e.id, type: e.type })),
        // 물리 바디 pose가 진실 (CLAUDE.md §2.1) — 첫 바디(단일 바디 엔티티는 그 바디,
        // 로봇은 링크 생성 순서상 루트 링크)를 읽는다
        getPose: (id) => {
          const firstBody = world.bodiesOfEntity(id)[0];
          if (firstBody === undefined) return null;
          const pose = world.getPose(firstBody);
          return { position: pose.position, rotation: pose.rotation };
        },
        // 로봇 엔티티만 관절 섹션 — limits는 URDF ∩ override 유효값 (RobotBinding.joints)
        getJoints: (id) => {
          if (!sceneHandle.robots.ids().includes(id)) return null;
          const binding = sceneHandle.robots.get(id);
          const values = binding.readJoints();
          return binding.joints.map((joint) => ({
            name: joint.name,
            valueRad: values[joint.name] ?? joint.initial,
            ...(joint.limits ? { limits: joint.limits } : {}),
          }));
        },
        // 인스펙터 행 클릭 → 뷰포트 선택으로 위임 — 선택 동기화의 단일 경로는
        // interaction.onSelect 리스너다 (아래). 변경 가드 덕에 루프가 생기지 않는다.
        onSelect: (id) => {
          interaction.select(id);
        },
      });
      built.inspector = inspector;
      adoptIntoStack(inspector.el);
      inspectorRef = inspector;

      // 엔티티 편집 폼 (UX §3.5 (A) — 이름/Transform/Dimensions/Physics 쓰기)
      const entityEditor = mountEntityEditor(rightStack, {
        getEntity: (id) => {
          const entity = editor.spec.entities.find((e) => e.id === id);
          return entity ? structuredClone(entity) : null;
        },
        isRobot: (id) => sceneHandle.robots.ids().includes(id),
        updateTransform: (id, transform) => {
          runEdit(() => editor.updateTransform(id, transform));
        },
        updateDimensions: (id, shape) => {
          runEdit(() => editor.updateDimensions(id, shape));
        },
        updatePhysics: (id, physics) => {
          runEdit(() => editor.updatePhysics(id, physics));
        },
        renameEntity: (id, newId) => {
          try {
            editor.renameEntity(id, newId);
            return null;
          } catch (err) {
            rebuildPickables(); // rename 실패 복원 경로가 노드를 재생성했을 수 있다
            return err instanceof Error ? err.message : String(err);
          }
        },
      });
      built.entityEditor = entityEditor;
      adoptIntoStack(entityEditor.el);

      // ── 선택 동기화 (단일 경로): interaction이 진실, 나머지가 따라온다 ──
      // 클릭 픽킹·프로그램 select·픽킹 맵 재구축(setPickables의 stale 해제) 모두
      // 이 리스너를 통해 인스펙터/편집 폼에 전파된다. 각 모듈의 변경 가드가 루프를 막는다.
      interaction.onSelect((id) => {
        inspector.select(id);
        entityEditor.showFor(id);
        if (id !== null) {
          const node = visualNodeOf(id);
          if (node) pulseEntity(node);
        }
        inspector.refresh();
      });

      // ── 편집 통지 → 파생 상태 재동기화 + 히스토리 기록 ────────────────
      built.offEditorChange = editor.onChange((e) => {
        rebuildPickables(); // add/remove/재빌드(치수·물리·개명)로 바뀐 노드 매핑 재구축
        inspector.refresh();
        viewportStatus.setEmptyHintVisible(editor.spec.entities.length === 0);
        if (e.kind === 'rename') {
          // 이전 id의 선택은 rebuildPickables가 해제했다 — 새 id를 재선택해 연속성 유지
          interaction.select(e.entityId);
        }
        const selectedId = interaction.selectedId;
        if (selectedId !== null) {
          const entity = editor.spec.entities.find((s) => s.id === selectedId);
          if (entity) entityEditor.refreshFrom(structuredClone(entity));
        }
        // Undo 스냅샷: 연속 조정(transform/dimensions/physics) burst는 디바운스로
        // 1장에 합쳐지고, 구조 변경(add/remove/rename)은 flushPending으로 경계를 세워
        // 개별 스냅샷이 된다 — "undo 1회 = 구조 변경 1개" (ui/history.ts 계약).
        const discrete = e.kind === 'add' || e.kind === 'remove' || e.kind === 'rename';
        if (discrete) history.flushPending(); // 직전 연속 burst를 먼저 확정
        history.noteChange(() => editor.serialize());
        if (discrete) history.flushPending();
      });

      engine.start();
      engine.play(); // 물리 루프 자동 시작 — 시퀀스는 ▶ Play로만 (파일 헤더의 재생 정책)
      console.log(
        `Scene '${spec.name}' loaded — entities: [${sceneHandle.entityIds.join(', ')}], ${spec.timestepHz}Hz`,
      );

      return {
        spec,
        validSequence,
        sequenceJson,
        editor,
        engine,
        uniquifyId,
        placeEntity,
        dispose: teardownBuilt,
      };
    } catch (err) {
      // 조립 도중 실패 — 이미 만들어진 몫만 되감고 재던진다 (loadScene이 표면화)
      teardownBuilt();
      throw err;
    }
  }

  // ── 씬 전환 (항상 전체 클린 빌드 — 파일 헤더의 씬 라이프사이클) ───

  async function loadScene(
    request: SceneLoadRequest,
    opts: SceneLoadOptions = {},
  ): Promise<SceneLoadResult> {
    if (switching) {
      return {
        ok: false,
        stage: 'busy',
        errors: ['이미 씬 전환이 진행 중입니다 — 잠시 후 다시 시도하세요'],
      };
    }
    // 검증을 teardown보다 먼저 — 무효 씬 때문에 잘 돌던 씬을 잃지 않는다 (DATA_MODEL §8)
    const validation = validateScene(request.scene);
    if (!validation.ok) {
      console.error('Scene validation failed:', validation.errors);
      return { ok: false, stage: 'validate', errors: validation.errors };
    }

    switching = true;
    try {
      active?.dispose();
      active = null;
      active = await buildScene(validation.value, request.sequence);
      jsonViewer.refresh(); // 시퀀스 뷰어를 새 씬 진실로 갱신
      // 히스토리: 새 "논리 씬"이면 스택 리셋 + 기준 스냅샷. 히스토리 복원 재로드면
      // 스택을 건드리지 않는다 (SceneHistory가 스택 이동을 소유).
      if (!opts.fromHistory) history.reset(active.editor.serialize());
      return { ok: true };
    } catch (err) {
      // 빌드 도중 실패 — 이전 씬은 이미 해제되어 빈 상태로 남는다. 사유는 호출자
      // (부트: 오버레이 / 전환: 토스트+콘솔 패널)가 표면화한다.
      console.error('Scene build failed:', err);
      const msg = err instanceof Error ? (err.stack ?? err.message) : String(err);
      jsonViewer.refresh();
      return { ok: false, stage: 'build', errors: [msg] };
    } finally {
      switching = false;
    }
  }

  // ── 커맨드바 좌측: 타이틀 · 씬 프리셋 · 📂 업로드 · 💾 저장 · ↶↷ (UX_DESIGN §3.1) ──

  const sceneControls = mountSceneControls(
    commandBar.left,
    Object.keys(SCENE_REGISTRY),
    {
      switchToPreset: async (name): Promise<SceneSwitchResult> => {
        const entry = SCENE_REGISTRY[name];
        if (entry === undefined) {
          return { ok: false, errors: [`'${name}' 씬을 찾을 수 없습니다`] };
        }
        const result = await loadScene({ scene: entry.scene, sequence: entry.sequence });
        if (result.ok) {
          updateUrlSceneParam(name);
          appLog('info', `씬 전환: '${name}'`);
          return { ok: true };
        }
        for (const error of result.errors) {
          appLog('error', `씬 전환 실패 (${name}): ${error}`);
        }
        // build 실패는 이전 씬 teardown 이후다 — select가 죽은 씬 이름을 보이지 않게
        return { ok: false, errors: result.errors, sceneLost: result.stage === 'build' };
      },
      switchToUpload: async (payload, fileName): Promise<SceneSwitchResult> => {
        const { scene, sequence } = unwrapUploadEnvelope(payload);
        const result = await loadScene({ scene, sequence });
        if (result.ok) {
          updateUrlSceneParam(null); // 업로드 씬은 딥링크 불가 — 파라미터 제거
          appLog('info', `업로드 씬 로드: '${active?.spec.name ?? fileName}' (${fileName})`);
          return { ok: true };
        }
        for (const error of result.errors) {
          appLog('error', `업로드 씬 로드 실패 (${fileName}): ${error}`);
        }
        return { ok: false, errors: result.errors, sceneLost: result.stage === 'build' };
      },
      // 💾 저장 대상은 "현재 편집 상태"다 (build 시점 스냅샷이 아니라 editor.serialize).
      // 임포트 에셋(asset:// 참조)이 있으면 세션 한정 한계를 경고한다 — 저장된 JSON을
      // 다른 세션에서 열면 해당 엔티티는 복원되지 않는다 (mesh-import.ts 헤더의 한계).
      currentSpec: () => {
        const scene = active;
        if (!scene) return null;
        const spec = scene.editor.serialize();
        const assetRefs = collectAssetRefs(spec);
        if (assetRefs.length > 0) {
          showToast(`⚠ ${ASSET_SAVE_WARNING_KO}`, 'warn');
          appLog('warn', ASSET_SAVE_WARNING_KO);
        }
        return spec;
      },
    },
  );
  commandBar.left.appendChild(undoButton);
  commandBar.left.appendChild(redoButton);

  // ── 라이브러리 + 3D 임포트 (UX §3.2/§4.4 — 앱 수명, 활성 씬에 위임) ──

  /** 뷰포트 파일 드롭 좌표 — 임포트 확정 시 그 지점에 배치 (없으면 뷰포트 중앙) */
  let pendingImportDrop: { x: number; y: number } | null = null;

  const importDialog = mountImportDialog(document.body, {
    registerAsset: (bundle) => assetStore.register(bundle),
    onConfirm: (entity) => {
      const scene = active;
      if (!scene) {
        appLog('error', '임포트 실패: 활성 씬이 없습니다');
        return;
      }
      const drop = pendingImportDrop;
      pendingImportDrop = null;
      scene.placeEntity(entity, drop).then(
        (id) => {
          appLog('info', `임포트 엔티티 '${id}' 추가 (세션 한정 에셋 — 저장 시 경고)`);
        },
        (err: unknown) => {
          const msg = err instanceof Error ? err.message : String(err);
          appLog('error', `임포트 엔티티 추가 실패: ${msg}`);
          showToast(`임포트 실패: ${msg}`, 'warn');
        },
      );
    },
  });

  // Import ⬆ 카드/파일 드롭 공용 파일 선택기
  const importFileInput = document.createElement('input');
  importFileInput.type = 'file';
  importFileInput.accept = SUPPORTED_IMPORT_EXTENSIONS.join(',');
  importFileInput.style.display = 'none';
  importFileInput.dataset.testid = 'import-file-input';
  document.body.appendChild(importFileInput);
  importFileInput.addEventListener('change', () => {
    const file = importFileInput.files?.[0];
    importFileInput.value = ''; // 같은 파일 재선택도 change가 다시 발화하도록 초기화
    if (!file) return;
    pendingImportDrop = null; // 파일 선택 경로는 뷰포트 중앙 배치
    importDialog.openWith(file);
  });

  mountLibrary(workspace.slots.left, {
    onPlace: (entitySpec, dropClient) => {
      const scene = active;
      if (!scene) return;
      scene.placeEntity(entitySpec, dropClient).catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : String(err);
        appLog('error', `라이브러리 배치 실패: ${msg}`);
        showToast(`배치 실패: ${msg}`, 'warn');
      });
    },
    uniquify: (base) => active?.uniquifyId(base) ?? base,
    onImportRequest: () => importFileInput.click(),
  });

  // 뷰포트 드롭: 라이브러리 카드(TEMPLATE_MIME) 또는 3D 파일 (UX §3.2 드롭 계약, §4.4)
  workspace.slots.viewport.addEventListener('dragover', (e: DragEvent) => {
    if (!e.dataTransfer) return;
    const types = e.dataTransfer.types;
    if (types.includes(TEMPLATE_MIME) || types.includes('Files')) {
      e.preventDefault();
      e.dataTransfer.dropEffect = 'copy';
    }
  });
  workspace.slots.viewport.addEventListener('drop', (e: DragEvent) => {
    if (!e.dataTransfer) return;
    const templateKey = e.dataTransfer.getData(TEMPLATE_MIME);
    if (templateKey !== '') {
      e.preventDefault();
      const scene = active;
      const template = templateByKey(templateKey);
      if (!scene || !template) return;
      const entity = template.create((base) => scene.uniquifyId(base));
      scene.placeEntity(entity, { x: e.clientX, y: e.clientY }).catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : String(err);
        appLog('error', `라이브러리 드롭 배치 실패: ${msg}`);
        showToast(`배치 실패: ${msg}`, 'warn');
      });
      return;
    }
    const file = e.dataTransfer.files[0];
    if (file) {
      e.preventDefault();
      pendingImportDrop = { x: e.clientX, y: e.clientY };
      importDialog.openWith(file);
    }
  });

  // ── 부트 씬 로드 (?scene= 딥링크 — 기존 게이트 계약 그대로) ───────

  const bootResult = await loadScene({
    scene: bootEntry.scene,
    sequence: bootEntry.sequence,
  });
  if (!bootResult.ok) {
    if (bootResult.stage === 'validate') {
      showErrorOverlay(`씬 검증 실패 — ${bootSceneName}.scene.json`, bootResult.errors);
    } else {
      showErrorOverlay('부트스트랩 실패', bootResult.errors);
    }
    return;
  }
  sceneControls.setCurrent(bootSceneName);
  workspace.notifyResize(); // 슬롯 편입 직후 캔버스 크기를 워크스페이스 그리드에 맞춘다
}

boot().catch((err: unknown) => {
  console.error('Bootstrap failed:', err);
  const msg = err instanceof Error ? (err.stack ?? err.message) : String(err);
  showErrorOverlay('부트스트랩 실패', [msg]);
});
