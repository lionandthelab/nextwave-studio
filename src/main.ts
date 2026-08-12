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
//
// ── Flow Graph (Phase 8, UX_DESIGN §3.4/§6) ─────────────────────────
// 그래프 상태(FlowGraph)의 단일 소유자는 이 글루다: 씬 로드 시 fromSequence로 만들고,
// 모든 편집은 runFlowOp 파이프라인(op → 구조 검증 → 씬 참조 무결성 serializeGraph →
// 커밋)을 거친다 — 편집으로 직렬화 불가능한 상태를 만들 수 없다(불변식 §2.8). 커밋은
// 라이브 ControlSequence를 교체하고 JSON 뷰어·타임라인을 갱신한다.
// **시퀀스 편집 정책**: 재생(armed) 중 편집이 커밋되면 재생을 정지한다 — 엔진(씬 물리)
// 은 계속 돌고, player만 unarm + 커서 0으로 리셋된다(한국어 토스트 '시퀀스 수정됨 —
// 처음부터 재생됩니다'). 다음 ▶ Play가 편집된 시퀀스를 처음부터 재생한다.
// edges는 노드 배열에서 파생되는 상태다(schema/flow-graph deriveEdges) — 진실은
// 노드 순서 + goto params뿐이며, 이 글루는 그래프를 직접 변형하지 않고 op만 적용한다.
// 시퀀스 없는 씬도 '플로우' 토글로 빈 그래프에서 시퀀스를 만들 수 있다(기본 로봇 =
// 씬의 첫 로봇; 로봇 없는 씬은 시퀀스 검증(robot 참조)이 편집을 거부 — 한국어 안내).

import { CollisionMonitor } from './core/collision';
import { classifyContact, isCollision } from './core/collision-classify';
import type { ContactClass } from './core/collision-classify';
import {
  collisionQueryFromMonitor,
  robotApiFromRegistry,
} from './core/control/adapters';
import { ControlPlayer } from './core/control/player';
import type { PlayerStatus } from './core/control/player';
import {
  WAIT_FOR_COLLISION_TIMEOUT_MARKER,
  WAIT_FOR_COLLISION_WARN_TAG,
} from './core/control/steps';
import { Engine, ENGINE_SPEED_OPTIONS } from './core/engine';
import type { EngineSpeed, EngineState } from './core/engine';
import {
  clampPositionAboveGround,
  groundedTransformForShape,
  snapPositionToGround,
} from './core/ground-clamp';
import { GROUND_ENTITY_ID, SceneLoader } from './core/scene-loader';
import type { RenderSceneApi, RobotHandle, SceneHandle, VisualNode } from './core/scene-loader';
import { SceneEditorImpl } from './core/scene-editor';
import type { SceneEditEvent, SceneEditor } from './core/scene-edit-types';
import type { JointInfo } from './core/robot-types';
import { RenderSync } from './core/sync';
import type { PhysicsWorld, Pose } from './core/types';
import { initPhysics, RapierWorld } from './core/world';
import { pulseEntity } from './render/highlight';
import { mountContactMarkers } from './render/contact-marker';
import type { ContactMarkers } from './render/contact-marker';
import {
  isTypingTarget,
  NUDGE_FINE_STEP_M,
  NUDGE_STEP_M,
  ROTATION_SNAP_DEG,
  TRANSLATION_SNAP_M,
  ViewportInteraction,
} from './render/interaction';
import type {
  GizmoMode,
  NudgeAxis,
  TransformCommit,
  TypingTargetLike,
} from './render/interaction';
import {
  ASSET_SAVE_WARNING_KO,
  collectAssetRefs,
  MeshAssetStore,
  SUPPORTED_IMPORT_EXTENSIONS,
} from './render/mesh-import';
import { disposeMeshResources, groundMesh, primitiveMesh } from './render/meshes';
import { Renderer } from './render/renderer';
import { loadUrdfRobot } from './render/urdf';
import { jsonErrorKo, mountJsonViewer } from './ui/command-bar/json-viewer';
import type { ApplyJsonResult } from './ui/command-bar/json-viewer';
import { SequenceVersions } from './ui/sequence-versions';
import { mountPlaybackBar } from './ui/command-bar/playback';
import { mountFlowCanvas } from './ui/flow-graph/canvas';
import type { FlowCanvasOpResult } from './ui/flow-graph/canvas';
import { kindMeta } from './ui/flow-graph/node-render';
import type { NodeRunStatus } from './ui/flow-graph/node-render';
import { Orchestrator } from './ui/orchestrator';
import type {
  OrchestratorDeps,
  OrchestratorEngine,
  OrchestratorPlayer,
} from './ui/orchestrator';
import { mountNodeEditor } from './ui/inspector/node-editor';
import {
  COMMAND_BAR_PRIORITY,
  mountCommandBarShell,
  mountSceneControls,
  setCommandBarPriority,
} from './ui/command-bar/scene-controls';
import type { SceneSwitchResult } from './ui/command-bar/scene-controls';
import { createCollisionLogPanel } from './ui/dock/collision-log';
import { appLog, createConsolePanel } from './ui/dock/console-panel';
import { DOCK_TAB_ID, mountDock } from './ui/dock/dock';
import { createTimelinePanel } from './ui/dock/timeline';
import { SceneHistory } from './ui/history';
import type { HistorySnapshot } from './ui/history';
import { DIMENSION_MIN_M, mountEntityEditor } from './ui/inspector/entity-editor';
import { mountInspector, mountSceneOutliner } from './ui/inspector/inspector';
import type { InspectorHandle, SceneOutlinerHandle } from './ui/inspector/inspector';
import { mountJointPanel } from './ui/inspector/joint-panel';
import { mountImportDialog } from './ui/library/import-dialog';
import { mountLibrary, TEMPLATE_MIME } from './ui/library/library';
import { templateByKey } from './ui/library/templates';
import { mountSelectionHud } from './ui/viewport/selection-hud';
import { mountViewportStatus } from './ui/viewport/statusline';
import type { ViewportStatusHandle } from './ui/viewport/statusline';
import { computeRtf, mountStatsHud } from './ui/viewport/stats-hud';
import type { StatsHudHandle } from './ui/viewport/stats-hud';
import { mountDropHint } from './ui/viewport/drop-hint';
import type { DropHintHandle } from './ui/viewport/drop-hint';
import {
  mountRunOverlay,
  overlaySummary,
  timeSecToNodeIndex,
} from './ui/viewport/run-overlay';
import type { RunOverlayState } from './ui/viewport/run-overlay';
import { mountWorkspace } from './ui/workspace';
import { makeIconButton } from './ui/icons';
import { SCOPE_ATTR, createShortcutRouter } from './ui/shortcuts';
import { mountHelpSheet } from './ui/help-sheet';
import { RunRecorder } from './ui/run-recorder';
import { mountConsolePlane } from './ui/console/glue';
import type { ConsolePlaneHandle } from './ui/console/glue';
import type { ExecDefaults } from './ui/console/settings-screen';
import {
  createAutosave,
  createDirtyTracker,
  createDocument,
  createDocumentStore,
  describeAge,
  downloadDocument,
  installUnloadGuard,
  parseDocument,
} from './ui/document';
import { setDocumentTitle } from './ui/brand';
import { mountNlInput } from './ui/command-bar/nl-input';
import type { GenerateMode, NlInputHandle } from './ui/command-bar/nl-input';
import type { WorkcellDocument } from './ui/document';
import { DEFAULT_PLANNER_MODEL, mountPlannerSettings } from './ui/command-bar/planner-settings';
import type { PlannerBackendConfig } from './ui/command-bar/planner-settings';
import { mountClarifyCard } from './ui/feedback/clarify-card';
import { mountToasts } from './ui/feedback/toast';
import { AnthropicAdapter, PlannerService, buildContext } from './planner';
import type { PlannerResult, WorldSnapshot } from './planner';
import {
  BORDER,
  COLOR,
  FONT,
  LAYOUT,
  RADIUS,
  SHADOW,
  SPACE,
  SURFACE,
  TYPE,
  Z_INDEX,
  applyType,
  ensureThemeStyles,
  makeButton,
} from './ui/theme';
import {
  FLOW_GRAPH_SEQUENCE_ID,
  defaultNodeFor,
  deriveEdges,
  fromSequence,
  insertNode,
  isRobotSpec,
  moveNode,
  remapEntityId,
  removeNode,
  serializeGraph,
  setNodeEnabled,
  updateNodeParams,
  validateScene,
  validateSequence,
} from './schema';
import type {
  ColliderShape,
  CollisionEvent,
  ConveyorSpec,
  ControlSequence,
  ControlStep,
  EntitySpec,
  FlowGraph,
  FlowNode,
  PhysicsSpec,
  Quat,
  RunResult,
  SceneSpec,
  Transform,
  Vec3,
} from './schema';
import conveyorPickPlaceSceneJson from './assets/scenes/conveyor-pick-place.scene.json';
import lLineCellSceneJson from './assets/scenes/l-line-cell.scene.json';
import lLineCellSequenceJson from './assets/sequences/l-line-cell.sequence.json';
import conveyorPickPlaceSequenceJson from './assets/sequences/conveyor-pick-place.sequence.json';
import fallingBoxesSceneJson from './assets/scenes/falling-boxes.scene.json';
import armAndBoxesSceneJson from './assets/scenes/arm-and-boxes.scene.json';
import pickAndPlaceSceneJson from './assets/scenes/pick-and-place.scene.json';
import obstacleAvoidanceSceneJson from './assets/scenes/obstacle-avoidance.scene.json';
import collisionTestbedSceneJson from './assets/scenes/collision-testbed.scene.json';
import twoArmsCollisionSceneJson from './assets/scenes/two-arms-collision.scene.json';
import twoArmsCollisionSequenceJson from './assets/sequences/two-arms-collision.sequence.json';
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
  // 컨베이어 라인 — 벨트가 물건을 계속 실어 오고 로봇이 집어 감지 존에 넣는다 (§4.2)
  'conveyor-pick-place': {
    scene: conveyorPickPlaceSceneJson,
    sequence: conveyorPickPlaceSequenceJson,
  },
  // ㄱ자 라인 셀 — 직각으로 이어진 벨트 2개 위에서 로봇 3종이 각자 스테이션을 맡는다
  'l-line-cell': { scene: lLineCellSceneJson, sequence: lLineCellSequenceJson },
  // 로봇↔로봇 충돌 데모 — 두 팔이 중앙에서 만나 접촉한다 (ROBOT×ROBOT 회귀 시연)
  'two-arms-collision': {
    scene: twoArmsCollisionSceneJson,
    sequence: twoArmsCollisionSequenceJson,
  },
};

const DEFAULT_SCENE_NAME = 'arm-and-boxes';

/** 충돌 로그에서 최근 이벤트를 조회할 때의 기본 상한 (파사드 recent의 인자와 무관) */
const COLLISION_RECENT_DEFAULT_LIMIT = 50;

/** 우측 패널 스택 내 패널 사이 간격 (워크스페이스 우 슬롯 안 — UX_DESIGN §2 우측 존) */
const RIGHT_STACK_GAP_PX = 8;
/** 우측 패널 스택 안쪽 여백 */
const RIGHT_STACK_PADDING_PX = 8;
/**
 * 우측 스택 표시 순서 (flex order — DOM 마운트 순서와 분리).
 * 선택 대상을 "편집"하는 폼이 항상 맨 위다: 로봇을 선택하면 관절 슬라이더 패널과
 * 인스펙터의 관절 표가 길어져 편집 폼이 화면 밖으로 밀려나기 때문이다(adoptIntoStack 주석).
 */
const RIGHT_STACK_ORDER = { editForm: 0, inspector: 1, jointPanel: 2 } as const;
/** 인스펙터 값 갱신 스로틀 주기 (playing 중 — inspector.ts 헤더의 "주기 결정권은 통합자") */
const INSPECTOR_REFRESH_INTERVAL_MS = 150;
/** 앱 토스트 자동 숨김 (ms) */
const TOAST_AUTO_HIDE_MS = 5000;
/** 스케일 기즈모 → 치수 변환 시 half 치수 하한 (entity-editor 전체 치수 하한의 절반) */
const MIN_HALF_DIMENSION_M = DIMENSION_MIN_M / 2;
/** 방향키 안내 토스트 최소 간격 (ms) — 키를 누르고 있어도 토스트가 쌓이지 않게 */
const NUDGE_HINT_THROTTLE_MS = 2000;

/**
 * 기즈모 모드 키 (뷰포트 스코프). Unity/Isaac Sim의 W/E/R 관례를 그대로 따른다 —
 * 3D 편집기에서 가장 널리 학습된 매핑이라 별도 안내 없이도 손이 먼저 안다.
 */
const GIZMO_MODE_KEYS: ReadonlyArray<{ key: string; mode: GizmoMode; labelKo: string }> = [
  { key: 'W', mode: 'translate', labelKo: '기즈모 모드 — 이동(W) · 회전(E) · 스케일(R)' },
  { key: 'E', mode: 'rotate', labelKo: '회전 기즈모' },
  { key: 'R', mode: 'scale', labelKo: '스케일 기즈모' },
];

/**
 * 선택 오브젝트 이동 키 (뷰포트 스코프). 방향키는 **카메라 기준** 수평,
 * PageUp/PageDown은 월드 수직 — 화면에서 보이는 방향과 일치시키는 것이 3D 편집기의
 * 보편 규약이다. `canonical`인 항목만 도움말 시트에 한 줄로 나오고 나머지는 별칭이다.
 */
const NUDGE_KEYS: ReadonlyArray<{
  key: string;
  axis: NudgeAxis;
  canonical: boolean;
  labelKo: string;
  display?: readonly string[];
}> = [
  {
    key: 'ArrowLeft',
    axis: { kind: 'right', sign: -1 },
    canonical: true,
    labelKo: `선택 오브젝트 이동 — ${NUDGE_STEP_M * 100}cm · Shift ${NUDGE_FINE_STEP_M * 100}cm (카메라 기준)`,
    display: ['←', '→', '↑', '↓'],
  },
  { key: 'ArrowRight', axis: { kind: 'right', sign: 1 }, canonical: false, labelKo: '오른쪽 이동' },
  { key: 'ArrowUp', axis: { kind: 'forward', sign: 1 }, canonical: false, labelKo: '앞쪽 이동' },
  { key: 'ArrowDown', axis: { kind: 'forward', sign: -1 }, canonical: false, labelKo: '뒤쪽 이동' },
  {
    key: 'PageUp',
    axis: { kind: 'vertical', sign: 1 },
    canonical: true,
    labelKo: '선택 오브젝트 높이 조절 (월드 Y)',
    display: ['PageUp', 'PageDown'],
  },
  { key: 'PageDown', axis: { kind: 'vertical', sign: -1 }, canonical: false, labelKo: '아래로 이동' },
];
/** 바닥 클램프 안내 토스트 최소 간격 (ms) — ↓를 누르고 있어도 토스트가 쌓이지 않게 */
const GROUND_CLAMP_HINT_THROTTLE_MS = 2000;

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
  /** ⏹ Stop과 동일: 엔진 정지 + 씬/player/충돌 이력 + 물리 접촉 상태 리셋 */
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
  updateConveyor(id: string, conveyor: ConveyorSpec): void;
  renameEntity(id: string, newId: string): void;
  removeEntity(id: string): void;
  /** 뷰포트 픽킹 대상 id 목록 (선택 가능 엔티티 — 바닥 제외) */
  pickableIds(): string[];
  selectedId(): string | null;
  select(id: string | null): void;
  /**
   * 검증용: 현재 선택의 기즈모 앵커 · 시각 AABB 중심 · 루트 원점(월드 좌표) +
   * 기즈모가 실제로 그 앵커에 붙어 있는지. "기즈모 핸들이 보이는 몸통에 붙는가"의
   * 회귀 가드 — 로봇처럼 루트 원점이 발밑인 대상에서 앵커가 시각 중심에 오는지 확인한다.
   */
  anchorProbe(): {
    anchor: Vec3;
    visualCenter: Vec3;
    rootOrigin: Vec3;
    attachedToAnchor: boolean;
  } | null;

  /**
   * 검증용: 기즈모 핸들의 화면(클라이언트 px) 좌표. 게이트가 사용자의 실제 제스처
   * (보이는 몸통을 마우스로 끌기)를 합성 마우스로 재현하는 조준점이다.
   */
  anchorScreenPoint(): [number, number] | null;
}

/** Undo/Redo 파사드 (앱 수명 SceneHistory 위의 얇은 표면) */
export interface SimHistoryFacade {
  undo(): Promise<boolean>;
  redo(): Promise<boolean>;
  readonly canUndo: boolean;
  readonly canRedo: boolean;
}

/**
 * Flow Graph 편집 파사드 (Phase 8 — 게이트/자동화용 최소 표면).
 * 모든 편집은 UI와 동일한 runFlowOp 파이프라인(§2.8)을 거친다 — 성공 시 라이브
 * 시퀀스가 교체되고, 실패(반환 false) 시 그래프/시퀀스는 변하지 않는다.
 */
export interface SimFlowGraphFacade {
  /** flowGraph 페인 표시 여부 ('플로우' 토글 상태) */
  visible(): boolean;
  nodeCount(): number;
  /** 체인 순서의 노드 id 목록 ('n1','n2',... — fromSequence 안정 id 규약) */
  nodeIds(): string[];
  /** 체인 순서의 step kind 목록 */
  kinds(): string[];
  /** 노드 params 깊은 복사본 (없으면 null) */
  params(nodeId: string): Record<string, unknown> | null;
  reorder(nodeId: string, toIndex: number): boolean;
  /** defaultNodeFor('wait') 기본값 노드를 atIndex에 삽입 */
  insertWait(atIndex: number): boolean;
  remove(nodeId: string): boolean;
  setEnabled(nodeId: string, enabled: boolean): boolean;
  /** 현재 라이브 ControlSequence의 JSON 문자열 (커밋된 진실 — 항상 검증 통과본) */
  sequenceJson(): string;
  /** 마지막 편집 파이프라인 결과: 'ok' 또는 한국어 오류 목록 */
  lastValidation(): 'ok' | string[];
  /** 이번 재생 런에서 'active'로 표시된 적 있는 노드 id (스킵 검증용 — arm 시 리셋) */
  everActiveNodeIds(): string[];
  /** 현재 캔버스 실행 상태 맵 (nodeId → pending|active|done|error) */
  nodeStatuses(): Record<string, string>;
}

/**
 * 자연어 Planner 파사드 (Phase 9 — 게이트/자동화용). 플래너 서비스는 앱 수명이지만,
 * 생성 결과는 "현재 활성 씬"의 그래프에 로드되므로 이 파사드는 씬별 __sim에 실린다.
 * generate는 UI와 완전히 동일한 흐름을 탄다: buildContext → 생성 → 검증 → 그래프 로드.
 * 자동 재생하지 않는다(§2.9) — 사용자가(또는 게이트가) ▶ Play를 눌러야 실행된다.
 */
export interface SimPlannerFacade {
  /** 자연어 → 생성 (현재 backend, 교체 모드). 결과 type만 요약해 돌려준다. */
  generate(nl: string): Promise<{ type: string }>;
  /** 마지막 생성 결과 요약 (없으면 null) */
  lastResult(): {
    type: string;
    stepCount?: number;
    assumptions?: string[];
    question?: string;
    options?: string[];
  } | null;
  /** 마지막 생성 시퀀스가 현재 그래프에 로드되었는지 (origin 'generated' 노드 존재 여부) */
  isLoadedIntoGraph(): boolean;
  /** 현재 player 상태 문자열 ('idle'|'running'|'done') — §2.9 무자동재생 증명용 */
  playerStatus(): string;
}

/**
 * 실행 오케스트레이션 파사드 (Phase 10 — 게이트/자동화용). 노드 단위 재생·상태·재실행을
 * 노출한다. 재생 컨트롤은 playbackControls(=UI 재생 바)와 동일한 orchestrator 경로를 탄다 —
 * 파사드와 사람 조작이 같은 진실(§5 동기 강조)을 본다.
 */
export interface SimOrchestratorFacade {
  /** ▶ 연속 재생 (필요 시 시퀀스 arm 후) */
  play(): void;
  /** ⏸ 즉시 일시정지 */
  pause(): void;
  /** ⏹ 정지 + 재생 준비 (씬/player/충돌 이력 + 물리 접촉 상태 리셋) */
  stop(): void;
  /** ⏭ 노드 1개 전진 (물리 1 tick이 아니라 — §5) */
  stepNode(): void;
  /** '예기치 않은 충돌 시 자동 정지' 토글 (§5, 기본 off) */
  setAutoPause(enabled: boolean): void;
  /** 자동 정지 토글 현재 상태 */
  autoPause(): boolean;
  /** 현재 활성(실행 중) 노드 id, 없으면 null (트라이페인 진실) */
  activeNodeId(): string | null;
  /** 노드 실행 상태 맵 스냅샷 (nodeId → pending|active|done|error) */
  statuses(): Record<string, string>;
  /** 노드/타임라인 마커에서 결정론적 재실행 (처음부터 되감아 빨리감기 — §5) */
  runFromNode(nodeId: string): void;
  /** 실행 오버레이 요약 텍스트 (뷰포트 배지와 동일 — 게이트 트라이페인 검증용) */
  overlayText(): string;
}

/**
 * 3D 임포트 파사드 (게이트/자동화용 — UX_DESIGN §4.4).
 *
 * 다이얼로그를 **여는 것만** 노출한다. 파일 → 파싱 → 폼 → [추가]의 나머지 구간은
 * 게이트가 다이얼로그 DOM(data-testid)으로 조작해 **사람이 쓰는 경로와 같은 코드**를
 * 타게 한다. 파사드가 폼을 우회해 EntitySpec을 직접 조립하면 "다이얼로그는 망가졌는데
 * 게이트는 초록"이 된다 — orchestrator 파사드가 playbackControls(=UI 재생 바)를 위임
 * 호출하는 것과 같은 규약(§5 동기 강조).
 */
export interface SimMeshImportFacade {
  /** 임포트 다이얼로그를 연다 — 라이브러리 ⬆ / 뷰포트 파일 드롭과 동일 진입점 */
  open(file: File): void;
  /** 이 세션에 등록된 임포트 에셋 ref 목록 ('asset://<n>', 등록 순서) */
  assetRefs(): readonly string[];
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
  readonly flowGraph: SimFlowGraphFacade;
  readonly planner: SimPlannerFacade;
  readonly orchestrator: SimOrchestratorFacade;
  readonly meshImport: SimMeshImportFacade;
  readonly player?: SimPlayerFacade;
}

declare global {
  interface Window {
    /** 자동화 훅 (scripts/gate-browser.mjs가 검증에 사용) — 항상 "현재 활성 씬"을 가리킨다 */
    __sim?: SimHandle;
  }
}

// ── 이전 작업 복원 배너 (UX_AUDIT C-3) ──────────────────────────────
//
// 자동저장된 초안이 있을 때만 나타난다. 사용자가 명시적으로 선택하기 전에는 현재 씬을
// 건드리지 않는다 — 복원이 또 다른 데이터 손실이 되면 안 된다.

interface RestoreBannerDeps {
  readonly ageText: string;
  readonly originLabel: string | null;
  onRestore(): void;
  onDismiss(): void;
}

function mountRestoreBanner(host: HTMLElement, deps: RestoreBannerDeps): { remove(): void } {
  ensureThemeStyles();
  const bar = document.createElement('div');
  Object.assign(bar.style, {
    position: 'fixed',
    left: '50%',
    bottom: SPACE.xxl,
    transform: 'translateX(-50%)',
    zIndex: Z_INDEX.toast,
    display: 'flex',
    alignItems: 'center',
    gap: SPACE.lg,
    padding: `${SPACE.lg} ${SPACE.xl}`,
    background: SURFACE.overlay,
    border: `1px solid ${BORDER.default}`,
    borderRadius: RADIUS.lg,
    boxShadow: SHADOW.overlay,
    maxWidth: 'min(560px, calc(100vw - 32px))',
  } satisfies Partial<CSSStyleDeclaration>);
  bar.setAttribute('role', 'status');
  bar.dataset.testid = 'restore-banner';

  const text = document.createElement('div');
  applyType(text, TYPE.body);
  text.style.color = COLOR.text;
  const origin = deps.originLabel === null ? '' : ` · ${deps.originLabel}`;
  text.textContent = `저장하지 않은 이전 작업이 있습니다 (${deps.ageText}${origin})`;

  const restore = makeButton('복원', '이전 작업 복원', 'restore-apply', 'primary');
  const dismiss = makeButton('버리기', '이전 작업 버리기', 'restore-dismiss', 'ghost');

  const close = (): void => {
    bar.remove();
  };
  restore.addEventListener('click', () => {
    close();
    deps.onRestore();
  });
  dismiss.addEventListener('click', () => {
    close();
    deps.onDismiss();
  });

  bar.append(text, restore, dismiss);
  host.appendChild(bar);
  return { remove: close };
}

// ── 부트 오버레이 해제 ───────────────────────────────────────────────
//
// index.html의 #boot는 첫 페인트 ~ WASM 초기화 사이의 빈 화면을 덮는다. 부트가 성공하든
// 실패하든 반드시 걷어야 한다 — 실패 시엔 오류 오버레이가 그 아래에 있기 때문이다.

function dismissBoot(): void {
  const boot = document.getElementById('boot');
  if (boot === null) return;
  boot.dataset.hiding = 'true';
  const remove = (): void => {
    boot.remove();
  };
  boot.addEventListener('transitionend', remove, { once: true });
  // 트랜지션이 비활성(prefers-reduced-motion)이면 transitionend가 오지 않는다
  window.setTimeout(remove, 400);
}

// ── 오류 오버레이 (검증 실패·부트스트랩 실패 표시용, 한국어) ─────────

const OVERLAY_Z_INDEX = Z_INDEX.overlay;

/** 오류 화면에서 현재 작업을 파일로 구제하는 콜백 (부트 완료 후 배선) */
let rescueDocumentDownload: (() => void) | null = null;

/**
 * 전역 오류 안전망 (UX_AUDIT C-18).
 *
 * 구 상태: `window.onerror`/`unhandledrejection` 핸들러가 0건이라, WASM 초기화 실패나
 * URDF 메시 404(CLAUDE.md §9의 흔한 함정)가 전부 "화면이 안 뜬다"로 수렴했고 사용자는
 * 원인을 알 수 없었다. 신뢰성 인상은 "에러가 없다"가 아니라 **"에러를 잘 설명한다"** 에서 온다.
 */
function installGlobalErrorHandlers(): void {
  let shown = false;
  const report = (title: string, detail: string): void => {
    if (shown) return; // 첫 오류만 — 연쇄 오류로 화면을 덮지 않는다
    shown = true;
    showErrorOverlay(title, [detail]);
  };
  window.addEventListener('error', (e) => {
    const err: unknown = e.error;
    report('예기치 못한 오류', err instanceof Error ? (err.stack ?? err.message) : e.message);
  });
  window.addEventListener('unhandledrejection', (e) => {
    const reason: unknown = e.reason;
    report(
      '처리되지 않은 비동기 오류',
      reason instanceof Error ? (reason.stack ?? reason.message) : String(reason),
    );
  });
}

function showErrorOverlay(title: string, lines: readonly string[]): void {
  dismissBoot();
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

  // 오류 상황에서도 **작업물을 구제**하는 경로를 준다 — 이게 핵심이다.
  // 다시 시도할 수 있다는 것보다, 만든 것이 안전하다는 것이 먼저다.
  const actions = document.createElement('div');
  Object.assign(actions.style, {
    display: 'flex',
    gap: SPACE.md,
    marginTop: SPACE.xxl,
    flexWrap: 'wrap',
  } satisfies Partial<CSSStyleDeclaration>);

  if (rescueDocumentDownload !== null) {
    const rescue = makeButton('현재 작업 내려받기', '현재 워크셀을 파일로 저장', 'error-rescue', 'primary');
    rescue.addEventListener('click', () => {
      rescueDocumentDownload?.();
    });
    actions.appendChild(rescue);
  }
  const reload = makeButton('새로고침', '페이지를 다시 불러온다', 'error-reload');
  reload.addEventListener('click', () => {
    window.location.reload();
  });
  actions.appendChild(reload);
  overlay.appendChild(actions);

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
  /**
   * 현재 라이브 시퀀스 (검증 통과본, 없으면 null — 실행에 노출되지 않음, §2.9).
   * Phase 8: 그래프 편집이 커밋될 때마다 교체된다 — JSON 뷰어가 이 값을 그린다.
   */
  readonly validSequence: ControlSequence | null;
  /**
   * 히스토리 재로드 시 새 스펙에 재검증할 시퀀스 JSON. 그래프 편집으로 라이브
   * 시퀀스가 생겼으면 그것이 우선한다 — 씬 undo/redo가 시퀀스 편집을 잃지 않는다.
   */
  readonly sequenceJson: unknown;
  /** '플로우' 토글로 페인이 열렸을 때 — 로봇 없는 씬 안내 등 컨텍스트 피드백 */
  onFlowPaneShown(): void;
  readonly editor: SceneEditor;
  readonly engine: Engine;
  /** idBase → 현재 편집 스펙 기준 씬-유일 id (라이브러리 배치용) */
  uniquifyId(base: string): string;
  /** 엔티티를 드롭 좌표(null이면 뷰포트 중앙)의 바닥 레이캐스트 지점에 추가 + 선택 */
  placeEntity(entity: EntitySpec, dropClient: { x: number; y: number } | null): Promise<string>;
  /**
   * 플래너 생성 시퀀스를 그래프에 로드한다 (Phase 9 — 검증 통과본만, §2.9). seq는 호출부가
   * 이미 validateSequence를 통과시킨 값이다. replace는 교체(전부 origin 'generated'),
   * append는 기존 뒤에 이어 붙이며 새 step만 'generated'로 표시하고 label 충돌은 개명한다.
   * 자동 재생하지 않는다 — currentSequence만 교체하고 player는 로드하지 않는다.
   */
  loadGeneratedSequence(seq: ControlSequence, mode: GenerateMode): { ok: boolean; errors?: string[] };
  /** 현재 그래프에 origin 'generated' 노드가 있는지 (planner 파사드 isLoadedIntoGraph용) */
  hasGeneratedNodes(): boolean;
  /** 현재 player 상태 문자열 (planner 파사드 playerStatus용) */
  playerStatus(): string;
  dispose(): void;
}


/** 프리셋 전환을 URL ?scene=에 반영(딥링크 공유용) — 업로드 씬은 파라미터를 지운다 */
function updateUrlSceneParam(presetName: string | null): void {
  const url = new URL(window.location.href);
  if (presetName !== null) url.searchParams.set('scene', presetName);
  else url.searchParams.delete('scene');
  window.history.replaceState(null, '', url);
}

// ── 자연어 Planner 설정 영속화 (Phase 9 — localStorage) ──────────────
// 키는 이 브라우저에만 저장된다(공용 PC 경고는 설정 다이얼로그가 표시). 기본은 규칙 기반
// (오프라인, 네트워크 없음). Anthropic 선택 + 키가 있을 때만 SDK 어댑터로 라우팅한다.

/** localStorage 키 — { backend, apiKey, model } */
const PLANNER_CONFIG_KEY = 'robotSimWeb.planner';

/** localStorage에서 플래너 설정을 읽는다 (없거나 손상 시 규칙 기반 기본값) */
function loadPlannerConfig(): PlannerBackendConfig {
  try {
    const raw = localStorage.getItem(PLANNER_CONFIG_KEY);
    if (raw !== null) {
      const parsed = JSON.parse(raw) as Partial<PlannerBackendConfig>;
      return {
        backend: parsed.backend === 'anthropic' ? 'anthropic' : 'rule-based',
        apiKey: typeof parsed.apiKey === 'string' ? parsed.apiKey : '',
        model:
          typeof parsed.model === 'string' && parsed.model.trim() !== ''
            ? parsed.model
            : DEFAULT_PLANNER_MODEL,
      };
    }
  } catch {
    // 손상된 저장값/localStorage 불가 — 기본값으로 진행
  }
  return { backend: 'rule-based', apiKey: '', model: DEFAULT_PLANNER_MODEL };
}

/** 플래너 설정을 localStorage에 저장한다 (프라이빗 모드 등 실패는 조용히 무시) */
function savePlannerConfig(cfg: PlannerBackendConfig): void {
  try {
    localStorage.setItem(PLANNER_CONFIG_KEY, JSON.stringify(cfg));
  } catch {
    // 세션 한정으로만 동작 (localStorage 불가)
  }
}

/** 설정 → PlannerService. anthropic + 키가 있으면 SDK 어댑터, 그 외엔 규칙 기반(방어적 폴백) */
function buildPlannerService(cfg: PlannerBackendConfig): PlannerService {
  if (cfg.backend === 'anthropic' && cfg.apiKey.trim() !== '') {
    return new PlannerService({
      backend: { adapter: new AnthropicAdapter({ apiKey: cfg.apiKey, model: cfg.model }) },
    });
  }
  return new PlannerService({ backend: 'rule-based' });
}

/**
 * '이어서(append)' 병합: 기존 step 뒤에 incoming step을 이어 붙인다. incoming의 label
 * 이름이 기존 label과 충돌하면 suffix('_2','_3'...)로 개명하고, 같은 incoming 세그먼트의
 * goto가 그 label을 가리키면 참조도 함께 갱신한다 (교체/이어서 모드 계약 — Phase 9).
 */
function appendStepsWithLabelRename(
  existing: readonly ControlStep[],
  incoming: readonly ControlStep[],
): ControlStep[] {
  const used = new Set<string>();
  for (const step of existing) {
    if (step.kind === 'label') used.add(step.name);
  }
  const rename = new Map<string, string>();
  for (const step of incoming) {
    if (step.kind !== 'label') continue;
    if (!used.has(step.name)) {
      used.add(step.name);
      continue;
    }
    let n = 2;
    let candidate = `${step.name}_${n}`;
    while (used.has(candidate)) {
      n += 1;
      candidate = `${step.name}_${n}`;
    }
    rename.set(step.name, candidate);
    used.add(candidate);
  }
  const renamedIncoming = incoming.map((step) => {
    if (step.kind === 'label' && rename.has(step.name)) {
      return { ...step, name: rename.get(step.name)! };
    }
    if (step.kind === 'goto' && rename.has(step.label)) {
      return { ...step, label: rename.get(step.label)! };
    }
    return step;
  });
  return [...existing, ...renamedIncoming];
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

  installGlobalErrorHandlers();
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
  /** `active`를 선언 타입 그대로 읽는다 — 대입이 클로저 안에서만 일어나 CFA가 null로 좁힌다 */
  const getActiveScene = (): ActiveScene | null => active;
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
  commandBar.el.setAttribute(SCOPE_ATTR, 'commandBar');

  // ── 단축키 라우터 (UX_AUDIT C-6) ──────────────────────────────────
  //
  // 구 구현은 window에 keydown을 거는 곳이 5군데였고 Space를 3개가, 방향키를 2개가
  // 나눠 가졌다. 이제 전역 단축키는 전부 이 라우터를 통과한다 — 스코프 판정과
  // "위젯이 키의 주인이면 가로채지 않는다"는 규칙이 한곳에 있다.
  const shortcuts = createShortcutRouter(window);
  /** 현재 씬의 재생 컨트롤 — 씬 빌드가 대입한다(단축키는 씬 수명을 가로지른다) */
  let currentPlayback: {
    play(): void;
    pause(): void;
    stop(): void;
    stepOnce(): void;
  } | null = null;
  /** 현재 씬의 카메라 포커스 — F/Home 키가 쓴다 */
  let focusSelectedEntity: (() => void) | null = null;
  /** 현재 씬의 상태줄·드롭 힌트 — 씬 수명을 가로지르는 소비자(충돌 카운터/라이브러리 드래그)용 */
  let viewportStatusRef: ViewportStatusHandle | null = null;
  let activePlaybackBar: { togglePlay(): void } | null = null;
  let resetCameraView: (() => void) | null = null;
  /** '플로우' 토글 — 버튼과 단축키가 같은 함수를 부른다(동작 분기 금지) */
  let flowPaneToggler: (() => void) | null = null;
  let activeDropHint: DropHintHandle | null = null;
  /** 콘솔 평면 핸들 (Phase 12) — 부트 말미에 마운트. 씬 빌드가 실행 기록·저장 라우팅에 사용 */
  let consoleRef: ConsolePlaneHandle | null = null;
  /** 현재 씬의 결정론적 재실행 — 기록 화면 "이 노드부터 재현"이 사용 (씬 빌드가 대입) */
  let orchestratorRunFromNode: ((nodeId: string) => void) | null = null;
  /** 현재 씬의 실행 기본값 적용기 — 설정 화면의 변경을 즉시 반영 (씬 빌드가 대입) */
  let applyExecDefaultsToScene: ((defaults: ExecDefaults) => void) | null = null;
  /** 문서 표시명 override — 작업/공정 문서를 열면 탭 제목이 씬 이름 대신 이것을 쓴다 */
  let docLabelOverride: string | null = null;
  /**
   * 현재 씬의 뷰포트 편집 명령 — 기즈모 모드 전환 · 선택 오브젝트 이동 · 바닥에 붙이기.
   * 키 소유는 라우터(§2.10)에 있고 render/interaction은 명령만 제공한다.
   */
  let viewportEdit: {
    setGizmoMode(mode: GizmoMode): void;
    nudge(axis: NudgeAxis, fine: boolean): void;
    snapToGround(): void;
    hasSelection(): boolean;
  } | null = null;

  // ── 도움말 · 단축키 시트 (UX_AUDIT C-12) ──────────────────────────
  //
  // 시트는 UX_DESIGN §9의 규정 목록이 아니라 **실제 등록된 바인딩**만 그린다.
  // 문서를 베끼면 구현되지 않은 키를 광고하게 되고, 그건 도움말이 없는 것보다 나쁘다.
  const helpSheet = mountHelpSheet(document.body, {
    listShortcuts: () => shortcuts.list(),
  });

  // ── 문서 모델: 저장 상태 · 자동저장 · 이탈 가드 (UX_AUDIT C-3) ────
  //
  // 구 상태에는 사용자가 소유하는 대상이 없었다. 💾 저장은 SceneSpec만 직렬화해
  // **시퀀스를 버렸고**, localStorage/IndexedDB가 비어 있어 새로고침 한 번에 전부
  // 휘발했으며, beforeunload·dirty 표시가 0건이었다.
  const dirtyTracker = createDirtyTracker();
  const documentStore = createDocumentStore();

  const snapshotDocument = (): WorkcellDocument | null => {
    const scene = active;
    if (!scene) return null;
    return createDocument({
      name: scene.spec.name,
      scene: scene.editor.serialize(),
      sequence: scene.validSequence ?? null,
      nowIso: new Date().toISOString(),
    });
  };

  const autosave = createAutosave({
    store: documentStore,
    snapshot: snapshotDocument,
    originLabel: () => active?.spec.name ?? null,
    onError: (message) => {
      appLog('warn', `자동저장 실패: ${message}`);
    },
  });

  /** 편집이 일어났음을 문서 계층에 알린다 (dirty 재판정 + 자동저장 예약) */
  const markDocumentChanged = (): void => {
    const scene = active;
    if (!scene) return;
    dirtyTracker.check(scene.editor.serialize(), scene.validSequence ?? null);
    autosave.schedule();
  };

  /** 현재 문서를 봉투로 저장한다 — 씬과 시퀀스가 **함께** 나간다 */
  rescueDocumentDownload = (): void => {
    const doc = snapshotDocument();
    if (doc !== null) downloadDocument(doc);
  };

  const saveDocumentToFile = (): void => {
    const doc = snapshotDocument();
    if (doc === null) {
      showToast('저장할 워크셀이 없습니다 — 씬이 로드된 뒤 다시 시도하세요', 'warn');
      return;
    }
    const fileName = downloadDocument(doc);
    dirtyTracker.markSaved(doc.scene, doc.sequence);
    appLog('info', `워크셀 저장: ${fileName}`);
    showToast(`저장됨 — ${fileName}`, 'info');
  };

  dirtyTracker.onChange((dirty) => {
    setDocumentTitle(docLabelOverride ?? active?.spec.name ?? null, dirty);
  });
  installUnloadGuard(() => dirtyTracker.isDirty());

  // ── 단축키 등록표 (UX_DESIGN §9) ──────────────────────────────────
  //
  // 이 표가 **구현의 단일 진실**이고, 도움말 시트는 이 표만 그린다. 문서에만 있고
  // 구현되지 않은 키를 광고하지 않기 위해서다 — 구 상태는 규정 8종 중 2종만
  // 스펙대로 동작했고 ←/→는 스펙과 반대 기능에 배선돼 있었다.
  shortcuts.registerAll([
    {
      id: 'playback.toggle',
      keys: 'Space',
      scope: 'global',
      group: '재생',
      labelKo: '재생 / 일시정지',
      run: (e) => {
        e.preventDefault(); // 페이지 스크롤 방지
        activePlaybackBar?.togglePlay();
      },
    },
    {
      id: 'playback.step',
      keys: 'ArrowRight',
      scope: 'global',
      group: '재생',
      labelKo: '노드 하나 실행 (Step)',
      run: (e) => {
        e.preventDefault();
        currentPlayback?.stepOnce();
      },
    },
    {
      id: 'playback.stop',
      keys: 'Escape',
      scope: 'viewport',
      group: '재생',
      labelKo: '정지 (처음으로 되감기)',
      run: () => {
        currentPlayback?.stop();
      },
    },
    {
      id: 'edit.undo',
      keys: 'Ctrl+Z',
      scope: 'global',
      group: '편집',
      labelKo: '실행 취소',
      run: (e) => {
        e.preventDefault();
        void history.undo();
      },
    },
    {
      id: 'edit.redo',
      keys: 'Ctrl+Shift+Z',
      scope: 'global',
      group: '편집',
      labelKo: '다시 실행',
      run: (e) => {
        e.preventDefault();
        void history.redo();
      },
    },
    {
      id: 'file.save',
      keys: 'Ctrl+S',
      scope: 'global',
      group: '파일',
      labelKo: '워크셀 저장 (씬 + 시퀀스)',
      // 입력창에 타이핑 중이어도 저장은 되어야 한다
      allowInTextEntry: true,
      run: (e) => {
        e.preventDefault(); // 브라우저 '페이지 저장' 차단
        // 작업/공정 문서 컨텍스트가 있으면 서버 저장(충돌 처리 포함), 아니면 파일 다운로드
        void (async (): Promise<void> => {
          const handled = await consoleRef?.saveActive();
          if (handled !== true) saveDocumentToFile();
        })();
      },
    },
    // ── 뷰포트 편집 (scope: viewport — 3D 화면에 포커스가 있을 때만) ──
    //
    // 이전에는 render/interaction.ts가 window에 keydown을 직접 걸어 W/E/R·방향키를
    // 처리했다. 라우터가 모르는 두 번째 키맵이었고, 그 결과 `→`는 **재생 Step과
    // 오브젝트 이동을 동시에** 일으켰다(라우터의 preventDefault는 다른 리스너를 막지
    // 못한다). 이제 두 조작 모두 이 표에 있고, 스코프가 소유권을 가른다:
    // 뷰포트에 포커스 + 선택 있음 → 이동, 그 밖 → Step.
    ...GIZMO_MODE_KEYS.map(({ key, mode, labelKo }, index) => ({
      id: `edit.gizmo.${mode}`,
      keys: key,
      scope: 'viewport' as const,
      group: '뷰포트 편집',
      labelKo,
      // 대표 줄 하나로 3종 모드를 표기한다 (W/E/R는 같은 조작의 세 상태)
      hidden: index > 0,
      keysDisplay: index === 0 ? GIZMO_MODE_KEYS.map((k) => k.key) : undefined,
      run: (e: KeyboardEvent): void => {
        e.preventDefault();
        viewportEdit?.setGizmoMode(mode);
      },
    })),
    ...NUDGE_KEYS.map(({ key, axis, canonical, labelKo, display }) =>
      [false, true].map((fine) => ({
        id: `edit.nudge.${key}${fine ? '.fine' : ''}`,
        keys: fine ? `Shift+${key}` : key,
        scope: 'viewport' as const,
        group: '뷰포트 편집',
        labelKo,
        hidden: !canonical || fine,
        keysDisplay: canonical && !fine ? display : undefined,
        // 선택이 없으면 이 바인딩은 성립하지 않는다 — 라우터가 global의 Step으로
        // 넘어가므로 "아무것도 선택하지 않은 채 →"는 규정대로 Step이 된다.
        isEnabled: (): boolean => viewportEdit?.hasSelection() === true,
        run: (e: KeyboardEvent): void => {
          e.preventDefault(); // 방향키 페이지 스크롤 방지
          viewportEdit?.nudge(axis, fine);
        },
      })),
    ).flat(),
    {
      id: 'edit.snapGround',
      keys: 'End',
      scope: 'viewport',
      group: '뷰포트 편집',
      labelKo: '선택 오브젝트를 바닥에 붙이기',
      run: (e) => {
        e.preventDefault();
        viewportEdit?.snapToGround();
      },
    },
    {
      id: 'camera.focus',
      keys: 'F',
      scope: 'global',
      group: '카메라',
      labelKo: '선택 대상 포커스',
      run: (e) => {
        e.preventDefault();
        focusSelectedEntity?.();
      },
    },
    {
      id: 'camera.reset',
      keys: 'Home',
      scope: 'global',
      group: '카메라',
      labelKo: '카메라 리셋',
      run: (e) => {
        e.preventDefault();
        resetCameraView?.();
      },
    },
    {
      id: 'view.flow',
      keys: 'Ctrl+Shift+F',
      scope: 'global',
      group: '보기',
      labelKo: '플로우 그래프 표시 / 숨김',
      run: (e) => {
        e.preventDefault();
        flowPaneToggler?.();
      },
    },
    {
      id: 'view.dock',
      keys: 'Ctrl+Shift+D',
      scope: 'global',
      group: '보기',
      labelKo: '하단 독 펼치기 / 접기',
      run: (e) => {
        e.preventDefault();
        workspace.setDockCollapsed(!workspace.isDockCollapsed());
      },
    },
    {
      id: 'help.open',
      keys: '?',
      scope: 'global',
      group: '일반',
      labelKo: '도움말 · 단축키',
      run: (e) => {
        e.preventDefault();
        if (helpSheet.isOpen()) helpSheet.close();
        else helpSheet.open();
      },
    },
  ]);
  // '플로우' 토글 (Phase 8) — 중앙 하단 flowGraph 페인 표시/숨김. 표시 상태는 앱 수명이
  // 소유하고, 씬 로드가 시퀀스 유무로 재설정한다(시퀀스 있는 씬 = 자동 표시). 시퀀스
  // 없는 씬도 페인을 열어 빈 그래프에서 시퀀스를 만들 수 있다(파일 헤더의 Flow Graph 절).
  let flowPaneVisible = false;
  const flowToggleButton = makeIconButton(
    'workflow',
    '플로우',
    '플로우 그래프 페인 표시/숨김 (Ctrl+Shift+F)',
    'flow-toggle',
  );
  setCommandBarPriority(flowToggleButton, COMMAND_BAR_PRIORITY.view);
  flowToggleButton.setAttribute('aria-pressed', 'false');
  const setFlowPaneVisible = (visible: boolean): void => {
    flowPaneVisible = visible;
    workspace.setFlowGraphVisible(visible); // 변경 시 notifyResize → 캔버스 fit 추종
    flowToggleButton.classList.toggle('ui-btn--active', visible);
    flowToggleButton.setAttribute('aria-pressed', String(visible));
  };
  flowPaneToggler = (): void => {
    setFlowPaneVisible(!flowPaneVisible);
    if (flowPaneVisible) active?.onFlowPaneShown();
  };
  flowToggleButton.addEventListener('click', () => {
    flowPaneToggler?.();
  });
  commandBar.right.appendChild(flowToggleButton);

  // '블록 저장' — 현재 시퀀스를 재사용 블록으로 (Phase 12 ⑤, 다이얼로그는 콘솔 글루 소유)
  const blockCaptureButton = makeIconButton(
    'puzzle',
    '블록 저장',
    '현재 시퀀스를 재사용 블록으로 저장',
    'block-capture',
  );
  setCommandBarPriority(blockCaptureButton, COMMAND_BAR_PRIORITY.view);
  blockCaptureButton.addEventListener('click', () => {
    consoleRef?.openBlockCapture();
  });
  commandBar.right.appendChild(blockCaptureButton);

  // ── {} JSON 패널 (보기 · **직접 편집** · 버전 이력) ────────────────
  //
  // 편집/되돌리기는 씬 수명 함수(applyJsonToSequence / restoreSequenceVersion)에 위임한다 —
  // 패널은 앱 수명이고 시퀀스 진실은 씬마다 새로 서기 때문이다. 씬이 없으면 편집을 거부한다.
  // 모든 적용은 그래프 편집과 **같은 §2.8 파이프라인**(검증 → 커밋)을 지난다.
  /**
   * 현재 씬의 JSON 적용기/되돌리기 — 씬 빌드가 대입하고 **teardown이 반드시 null로
   * 되돌린다**. 이 정리가 없으면 빌드 실패(손상 파일 업로드)나 씬 전환 중(URDF 로딩
   * 수백 ms) 사용자가 [적용]/[되돌리기]를 눌렀을 때 **해제된 씬의 클로저가 실행**되어,
   * 아무 일도 안 일어났는데 성공 토스트가 뜨고 죽은 씬 스냅샷이 Undo 스택에 들어간다.
   */
  let applyJsonToSequence: ((text: string) => ApplyJsonResult) | null = null;
  let restoreSequenceVersion: ((version: number) => ApplyJsonResult) | null = null;

  /**
   * 시퀀스 버전 스택 — **앱 수명**이다.
   *
   * 씬 수명으로 두면 Ctrl+Z(씬 Undo)가 전체 재빌드를 거치므로 이력이 통째로 날아갔다:
   * JSON 편집으로 v2~v4를 쌓고 Ctrl+Z를 한 번 누르면 [버전] 탭에 v1만 남아, 방금
   * 떠나온 상태로도 중간 상태로도 돌아갈 수 없었다. **이 기능의 존재 이유(안전망)를
   * 다른 되돌리기 축이 지우는** 구조였다. 대신 loadScene이 "새 논리 씬"일 때만 clear()
   * 한다 — history.reset()과 정확히 같은 조건(§ loadScene의 fromHistory 분기)이다.
   */
  const sequenceVersions = new SequenceVersions();

  const NO_ACTIVE_SCENE_RESULT: ApplyJsonResult = {
    ok: false,
    errors: ['활성 씬이 없어 시퀀스를 적용할 수 없습니다'],
  };

  const jsonViewer = mountJsonViewer(commandBar.right, document.body, {
    getSequence: () => active?.validSequence ?? null,
    applyJson: (text) => applyJsonToSequence?.(text) ?? NO_ACTIVE_SCENE_RESULT,
    listVersions: () => sequenceVersions.list(),
    restoreVersion: (version) => restoreSequenceVersion?.(version) ?? NO_ACTIVE_SCENE_RESULT,
    describeAge: (atIso) => describeAge(atIso, Date.now()),
  });

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

  const undoButton = makeIconButton('undo', '', '실행 취소 (Ctrl+Z)', 'history-undo');
  const redoButton = makeIconButton('redo', '', '다시 실행 (Ctrl+Shift+Z)', 'history-redo');
  setCommandBarPriority(undoButton, COMMAND_BAR_PRIORITY.misc);
  setCommandBarPriority(redoButton, COMMAND_BAR_PRIORITY.misc);
  undoButton.disabled = true;
  redoButton.disabled = true;

  const history = new SceneHistory({
    restore: (snapshot) => restoreSceneFromHistory(snapshot),
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
  async function restoreSceneFromHistory(snapshot: HistorySnapshot): Promise<boolean> {
    const scene = active;
    if (scene && scene.engine.state === 'playing') {
      scene.engine.stop();
      showToast('되돌리기: 시뮬 정지됨', 'info');
    }
    // 시퀀스도 스냅샷에서 복원한다 — 구 구현은 현재 씬의 sequenceJson을 그대로
    // 재사용해서, 플로우 그래프 편집이 Undo 대상에서 통째로 빠져 있었다 (C-4).
    const result = await loadScene(
      { scene: snapshot.scene, sequence: snapshot.sequence ?? scene?.sequenceJson },
      { fromHistory: true },
    );
    if (!result.ok) {
      for (const error of result.errors) appLog('error', `히스토리 복원 실패: ${error}`);
      showToast('되돌리기/다시하기 실패 — 콘솔 탭을 확인하세요', 'warn');
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
      addPrimitive: (shape, color, style) => {
        const mesh = primitiveMesh(shape, color, style);
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
        conveyors: sceneHandle.conveyors,
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
      autoPauseControl?: HTMLElement;
      viewportStatus?: ReturnType<typeof mountViewportStatus>;
      selectionHud?: ReturnType<typeof mountSelectionHud>;
      runOverlay?: ReturnType<typeof mountRunOverlay>;
      contactMarkers?: ContactMarkers;
      orchestrator?: Orchestrator;
      offTick?: () => void;
      offEditorChange?: () => void;
      rightStack?: HTMLDivElement;
      jointPanel?: ReturnType<typeof mountJointPanel>;
      inspector?: InspectorHandle;
      sceneOutliner?: SceneOutlinerHandle;
      statsHud?: StatsHudHandle;
      dropHint?: DropHintHandle;
      entityEditor?: ReturnType<typeof mountEntityEditor>;
      flowNodeEditor?: ReturnType<typeof mountNodeEditor>;
      flowCanvas?: ReturnType<typeof mountFlowCanvas>;
      flowPaneHost?: HTMLDivElement;
    } = {};

    // ── 실행 기록 (Phase 12 ⑦ — RunRecorder는 순수 축적, 제출은 콘솔 글루) ──
    // 기록은 **작업 컨텍스트가 있을 때만** 남긴다: 데모/프리셋 재생을 기록하면 실행 기록
    // 화면이 열 수 없는 작업 id로 오염된다. 헬퍼는 teardownBuilt보다 먼저 선언한다 —
    // 조립 실패 catch가 teardownBuilt를 부를 때 TDZ로 원인 오류를 가리지 않기 위해서다.
    // (finishRun은 isActive 가드로 조기 반환하므로 engine/flowGraph 미선언 시점에도 안전.)
    const runRecorder = new RunRecorder();
    let runWallStartMs = 0;
    let runSawAutoPause = false;
    /** simTime 게터 — engine은 try 블록에서 생성되므로 생성 후 대입된다 */
    let runSimTimeSec: () => number = () => 0;

    const finishRun = (result: RunResult): void => {
      if (!runRecorder.isActive()) return;
      const record = runRecorder.finish(result, {
        endedAtIso: new Date().toISOString(),
        simTimeSec: runSimTimeSec(),
        wallTimeSec: (Date.now() - runWallStartMs) / 1000,
      });
      if (record !== null) consoleRef?.submitRun(record);
    };

    // 이 씬 몫의 전부를 해제한다 — 다음 빌드에 어떤 상태도 새지 않는다(전체 클린 빌드).
    // 순서: 엔진 루프 완전 정지 → 구독 해제 → 상호작용(기즈모/하이라이트) 해제 →
    // UI 제거 → 씬 자원(물리 바디·시각 노드·로봇 핸들) 해제 → sync 바인딩 정리 →
    // 월드 free. 렌더러/캔버스·워크스페이스 셸은 유지된다.
    const teardownBuilt = (): void => {
      // 앱 수명 패널({} JSON)이 잡고 있는 **이 씬의 클로저를 먼저 끊는다.** 남겨두면
      // 빌드 실패 후나 전환 중(URDF 로딩)에 [적용]/[되돌리기]가 해제된 씬에 커밋되어,
      // 아무 일도 안 일어났는데 성공 토스트가 뜨고 죽은 스냅샷이 Undo 스택에 들어간다.
      applyJsonToSequence = null;
      restoreSequenceVersion = null;
      // 진행 중이던 실행 기록을 먼저 닫는다 (씬 전환/문서 열기로 재생이 끊긴 경우 —
      // simTimeSec을 엔진 정지 전에 읽는다). 기록이 없으면 no-op.
      finishRun(runSawAutoPause ? 'autoPaused' : 'stopped');
      built.engine?.halt();
      built.offTick?.();
      built.orchestrator?.dispose(); // player.onStepChange + monitor 구독 해제 (Phase 10)
      built.offMonitor?.();
      built.offEditorChange?.();
      history.cancelPending(); // 해제될 editor를 캡처하는 pending 스냅샷 폐기
      built.interaction?.dispose(); // 씬 노드가 살아있는 동안 emissive 원복 + 기즈모 분리
      built.gizmoBar?.remove();
      built.playbackBar?.dispose();
      built.autoPauseControl?.remove();
      built.viewportStatus?.dispose();
      built.selectionHud?.dispose();
      built.runOverlay?.dispose();
      built.contactMarkers?.dispose();
      // flow 캔버스는 앱 수명 페인(workspace.slots.flowGraph) 안에 산다 — 씬 몫만 제거
      built.flowCanvas?.dispose();
      built.flowPaneHost?.remove();
      built.flowNodeEditor?.dispose();
      built.entityEditor?.dispose();
      built.inspector?.dispose();
      built.sceneOutliner?.dispose();
      built.statsHud?.dispose();
      built.dropHint?.dispose();
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
      // player 경고 → 콘솔 로그 + flow 캔버스 상태 배선(아래 flow 섹션이 구현을 주입:
      // waitForCollision timeout 경고 시 해당 노드를 'error'로 마킹 — Phase 8 §4)
      let handlePlayerWarn: (msg: string) => void = () => {};
      const player = new ControlPlayer({
        robots: robotApi,
        collision: collisionQueryFromMonitor(monitor),
        warn: (msg) => {
          appLog('warn', msg);
          handlePlayerWarn(msg);
        },
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
          // 히스토리 복원으로 엔티티 참조가 깨진 경우 등 — 시퀀스는 언로드 상태로 남는다.
          // 콘솔 로그만으로는 플로우가 사라진 이유가 보이지 않으므로 토스트로도 알린다
          // (undo 직후의 조용한 플로우 소실 방지 — UX §7 검증 오류 표면화).
          console.warn('Sequence validation failed:', sequenceValidation.errors);
          for (const error of sequenceValidation.errors) {
            appLog('error', `시퀀스 검증 실패: ${error}`);
          }
          showToast(
            '시퀀스 검증 실패 — 플로우를 로드하지 않았습니다. 콘솔 탭에서 사유를 확인하세요',
            'warn',
          );
        }
      }

      // ── 라이브 시퀀스 + Flow Graph 상태 (Phase 8 — 파일 헤더의 Flow Graph 절) ──
      // currentSequence가 "지금 재생 가능한" 진실이다. 그래프 편집 커밋마다 교체되고,
      // JSON 뷰어·타임라인·Play arm이 전부 이 값을 본다. sequenceArmed는 최초 Play
      // 시 player.load 여부(human-in-the-loop) + preStep 진행 게이트를 겸한다.
      let currentSequence: ControlSequence | null = validSequence;
      let sequenceArmed = false;
      /**
       * 마지막 재생 이후 시퀀스가 편집됐는가 — ▶가 "이어서"가 아니라 **"처음부터"** 를
       * 뜻해야 하는 상태다.
       *
       * 구 동작: 편집은 player만 unarm하고 씬 물리는 그대로 뒀다. 그래서 완주 후 노드를
       * 재정렬하고 ▶를 누르면 로봇이 **이전 런의 끝 포즈에서** 시작했다 — moveJoints
       * 목표가 이미 달성돼 있어 아무 움직임이 없거나, waitForCollision이 이전 런의 잔여
       * 접촉으로 즉시 통과하거나 반대로 타임아웃까지 6초를 세웠다. 사용자에게는 정확히
       * "순서를 바꿨는데 로봇이 예전처럼 군다"로 보인다.
       */
      let sequenceDirtySinceRun = false;
      /** toSequence 복원용 시퀀스 메타 (그래프에는 id/loop가 실리지 않는다 — F1 계약) */
      const flowSeqMeta: { id: string; loop: boolean | undefined } = {
        id: validSequence?.id ?? FLOW_GRAPH_SEQUENCE_ID,
        loop: validSequence?.loop,
      };
      /** 그래프 진실 (단일 소유: 이 글루) — 시퀀스 없는 씬은 빈 그래프 + 첫 로봇 기본 */
      let flowGraph: FlowGraph = validSequence
        ? fromSequence(validSequence, { origin: 'manual' })
        : { nodes: [], edges: [], robot: sceneHandle.robots.ids()[0] ?? '' };
      let lastFlowValidation: 'ok' | string[] = 'ok';
      // 로드 시점을 버전으로 남긴다 — "열었을 때로 되돌리기"가 항상 가능해야 한다.
      // 스택이 비어 있으면 새 씬(loadScene이 clear했다), 아니면 Undo/Redo 복원이다.
      if (validSequence !== null) {
        sequenceVersions.record(validSequence, {
          labelKo: sequenceVersions.size() === 0 ? '열었을 때' : '되돌리기(Ctrl+Z)로 복원',
        });
      }
      /** 캔버스 실행 상태 (nodeId → 상태) — Orchestrator onNodeStatus가 다시 그린다 (Phase 10) */
      let flowStatuses: Record<string, NodeRunStatus> = {};
      /** 이번 재생 런에서 active로 표시된 노드 (게이트의 스킵 검증용 — arm 시 리셋) */
      const flowEverActiveNodeIds = new Set<string>();
      /**
       * 현재 활성(실행 중) 노드 id 캐시 — Orchestrator onActiveNode에서 파생한다(트라이페인
       * 진실: 그래프 활성 노드 ↔ 뷰포트 배지 ↔ Timeline 커서의 단일 소스). unexpectedCollision
       * 판정·run-overlay 활성 라벨이 이 값을 읽는다.
       */
      let activeFlowNodeId: string | null = null;
      /**
       * 노드가 active가 된 시점의 simTime(체인 인덱스별) — Collision Log 행 클릭 시
       * timeSec → 당시 활성 노드 강조(§3.6)에 쓴다. arm 시 비우고 노드 활성 통지마다 기록한다.
       * loop 시퀀스는 나중 활성이 이전 값을 덮으므로 근사다(run-overlay.timeSecToNodeIndex 계약).
       */
      const nodeActiveStartSimSec: number[] = [];

      const engine = new Engine(
        {
          world,
          sync,
          render,
          hooks: {
            // 매 물리 tick, world.step() 직전 (ARCHITECTURE §5 ①):
            // ① player가 관절 "상태"를 갱신하고 → ② robots가 FK를 kinematic 바디로 push.
            // sequenceArmed 게이트: 미로드 시 no-op이던 기존 계약에 더해, 그래프 편집이
            // 재생을 정지(unarm)하면 로드된 이전 시퀀스도 더 진행하지 않는다 (Phase 8
            // 시퀀스 편집 정책 — 파일 헤더).
            preStep: (simTimeSec, dtSec) => {
              if (sequenceArmed) player.step(simTimeSec, dtSec);
              sceneHandle.robots.tickAll();
              // ③ 벨트 표면 구동 + 재순환 (DATA_MODEL §4.2). 로봇 FK push **뒤**에
              //    돌아야 한다 — 같은 tick에서 둘 다 사물에 손을 대면 나중 것이 이기고,
              //    벨트 위에서는 벨트가 이기는 것이 자연스럽다. 재순환의 teleport도
              //    여기서 일어나므로 뒤따르는 sync.commit()이 새 pose를 prev로 잡는다
              //    (순간이동 궤적이 그려지지 않는다 — engine.ts tick 순서).
              sceneHandle.conveyors.tickAll();
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
        // 성공/실패 무관 — 실패 복원도 스펙을 바꿀 수 있다 (UX_AUDIT C-3)
        markDocumentChanged();
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

      // 단축키 라우터가 부를 뷰포트 편집 명령 (씬 수명 — 다음 씬 빌드가 덮어쓴다).
      // snapToGround는 아래 clampAboveGround 블록에서 정의되므로 지연 호출로 묶는다.
      viewportEdit = {
        setGizmoMode: (mode) => {
          interaction.setMode(mode);
          paintGizmoBar();
        },
        nudge: (axis, fine) => {
          interaction.nudgeSelected(axis, fine);
        },
        snapToGround: () => {
          snapSelectionToGround();
        },
        hasSelection: () => interaction.selectedId !== null,
      };

      const isRobotEntity = (id: string): boolean => sceneHandle.robots.ids().includes(id);

      /**
       * 커밋되지 않은 드래그 프리뷰를 버리고 시각/물리를 spec 진실로 되돌린다.
       * 비로봇은 sync 재바인딩이 자가 치유하지만 로봇의 시각 루트는 RenderSync 대상이
       * 아니라 FK 그래프 자체여서 되돌리는 주체가 없다 (scene-editor.resyncTransform 주석).
       * 실패는 조용히 삼키지 않고 콘솔에 남긴다 — 여기서 던지면 드래그 훅이 죽는다.
       */
      const resyncFromSpec = (id: string): void => {
        try {
          editor.resyncTransform(id);
        } catch (err) {
          appLog('error', err instanceof Error ? err.message : String(err));
        }
      };

      // 기즈모 드래그 수명 훅 (render/interaction.ts 헤더 "물리와의 관계"):
      // - 시작: playing이면 일시정지(로봇 루트 드래그가 preStep tickAll로 물리에
      //   새는 것 방지) + 대상 바디의 sync 바인딩 해제(드래그 프리뷰가 RenderSync
      //   덮어쓰기와 싸우지 않게 — 프리뷰가 실제로 포인터를 따라온다).
      // - 종료: commit(teleport) "이후" 통지되므로 재바인딩의 prev 스냅샷이 곧
      //   teleport된 물리 pose다 — 다음 프레임부터 시각이 물리 진실로 재수렴.
      //   commit이 실패/스킵된 경우에도 재바인딩이 시각을 물리 pose로 되돌린다.
      //   로봇은 그 자가 치유 경로가 없으므로 spec으로 명시 재수렴시킨다(대칭화).
      interaction.onDraggingChanged((dragging, id) => {
        if (dragging) pauseForEditIfPlaying('기즈모 편집');
        if (id === null) return;
        const record = sceneHandle.builtEntities.get(id);
        if (!record) return;
        if (record.robot) {
          // commit이 발행됐으면 spec이 이미 새 값이라 무해한 재적용(no-op)이 된다.
          if (!dragging) resyncFromSpec(id);
          return;
        }
        if (record.bodyId === undefined || !record.bound || !record.node) return;
        if (dragging) sync.unbind(record.bodyId);
        else sync.bind(record.bodyId, record.node);
      });

      /**
       * 편집으로 제안된 위치를 **바닥 위**로 클램프한다 (core/ground-clamp.ts).
       *
       * 지하 배치는 되돌릴 길이 없는 작업물 손실이다: static은 물리가 밀어내지 않고,
       * dynamic은 바닥 슬래브 밑으로 빠지면 무한 낙하해 화면에서 사라지며, 로봇 링크는
       * kinematicPosition이라 자가 교정이 아예 없다. 직접 조작 경로(기즈모·방향키·
       * 인스펙터 입력)에서 UI가 막고 이유를 알린다.
       *
       * 바닥이 없는 씬(environment.ground !== true)에서는 설 자리가 없으므로 클램프하지
       * 않는다. 파사드 __sim.editor.updateTransform은 자동화용이라 그대로 통과시킨다.
       */
      let lastGroundClampHintMs = 0;
      const clampAboveGround = (id: string, position: Vec3, rotation?: Quat): Vec3 => {
        if (editor.spec.environment?.ground !== true) return position;
        const entity = editor.spec.entities.find((e) => e.id === id);
        if (entity === undefined) return position;
        const result = clampPositionAboveGround(
          entity,
          position,
          rotation ?? entity.transform.rotation,
        );
        if (!result.clamped) return result.position;
        const now = performance.now();
        if (now - lastGroundClampHintMs >= GROUND_CLAMP_HINT_THROTTLE_MS) {
          lastGroundClampHintMs = now;
          showToast(
            `'${id}': 바닥 아래로는 내려갈 수 없습니다 — y를 ${result.minY.toFixed(3)} m로 맞췄습니다`,
            'warn',
          );
        }
        return result.position;
      };

      /**
       * 치수 편집 + 바닥 재안착. 중심을 고정한 채 크기만 키우면 아래쪽 절반이 그대로
       * 바닥을 뚫는다 — "바닥에 놓인 사물은 커져도 바닥에 놓여 있다"가 씬 편집기의
       * 보편 동작이다. 두 연산을 한 runEdit 안에서 처리해 undo 한 번으로 되돌아간다.
       */
      const applyDimensions = (id: string, shape: ColliderShape): void => {
        runEdit(() => {
          const before = editor.spec.entities.find((e) => e.id === id);
          editor.updateDimensions(id, shape);
          if (before === undefined || editor.spec.environment?.ground !== true) return;
          const grounded = groundedTransformForShape(before, shape);
          if (grounded !== null) editor.updateTransform(id, grounded);
        });
      };

      /**
       * "바닥에 붙이기" — 최저점을 바닥 상면에 정확히 맞춘다 (Unity/Unreal의 snap-to-floor).
       * 클램프와 달리 떠 있는 사물을 내리기도 하고, 이미 지하에 박힌 사물의 구제 경로다.
       */
      const snapSelectionToGround = (): void => {
        const id = interaction.selectedId;
        if (id === null) {
          showToast('바닥에 붙일 대상이 없습니다 — 뷰포트에서 오브젝트를 먼저 선택하세요', 'info');
          return;
        }
        const entity = editor.spec.entities.find((e) => e.id === id);
        if (entity === undefined) return;
        const position = snapPositionToGround(entity);
        runEdit(() => editor.updateTransform(id, { ...entity.transform, position }));
        showToast(`'${id}': 바닥에 붙였습니다 (y ${position[1].toFixed(3)} m)`, 'info');
      };

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
            // 비프리미티브(로봇·임포트 메시): 스케일 커밋 거부 + 시각 원복 (UX §3.3).
            // 원복은 scale뿐 아니라 position/rotation까지 spec으로 되돌린다 — 스케일
            // 기즈모 드래그도 대상을 조금 움직일 수 있기 때문이다.
            const node = pickables.get(id);
            if (node) node.scale.set(1, 1, 1);
            resyncFromSpec(id);
            const msg = `'${id}': 스케일 기즈모는 프리미티브 전용입니다 — 원복했습니다`;
            appLog('warn', msg);
            showToast(msg, 'warn'); // 콘솔 탭에만 남기면 사용자는 이유를 모른다
            return;
          }
          // 스케일 배율 → 치수 편집으로 변환. updateDimensions가 엔티티를 스케일 1의
          // 새 메시+collider로 재생성하므로 드래그로 커진 시각 스케일은 자연히 소거된다.
          applyDimensions(id, scaleShape(primitive, commit.scale));
          return;
        }
        const position = clampAboveGround(id, commit.position, commit.rotation);
        runEdit(() => editor.updateTransform(id, { position, rotation: commit.rotation }));
      });

      // 방향키 이동이 거부된 이유를 한국어로 표면화 (조용한 무시 금지 — UX §9).
      // 키를 누르는 동안 매번 발화하므로 스로틀한다.
      let lastNudgeHintMs = 0;
      interaction.onNudgeBlocked(() => {
        const now = performance.now();
        if (now - lastNudgeHintMs < NUDGE_HINT_THROTTLE_MS) return;
        lastNudgeHintMs = now;
        showToast('이동할 대상이 없습니다 — 뷰포트에서 클릭하거나 인스펙터 목록에서 선택하세요', 'info');
      });

      // ── UI: 하단 독 (Timeline | Collision Log | Console) — 워크스페이스 독 슬롯 ──

      const timelinePanel = createTimelinePanel();
      built.timelinePanel = timelinePanel;
      const collisionPanel = createCollisionLogPanel({
        onFocusEntity: (entityId) => {
          const node = visualNodeOf(entityId);
          if (node && entityId !== GROUND_ENTITY_ID) {
            pulseEntity(node);
            render.frameObject(node); // 라벨이 약속한 '포커스'를 실제로 수행한다
          }
          appLog('info', `충돌 로그: '${entityId}' 포커스`);
        },
        // 행 클릭 → 그 충돌 시점(timeSec)에 active였던 노드를 강조 (§3.6, Phase 10).
        // 기록된 노드 활성 시작 simTime 경계로 timeSec을 노드 인덱스로 접는다(근사 — loop 주석).
        // 주의: 이는 **읽기 전용 과거-시점 조사 하이라이트**다 — Orchestrator의 activeIndex(재생
        // 상태의 단일 진실)를 바꾸지 않으며, 다음 onNodeStatus/onActiveNode 방출에서 진실로
        // 되돌아간다. 재생 상태의 두 번째 소스가 아니다(정지/일시정지 중 일시적 조사 강조).
        onRowClick: ({ timeSec }) => {
          const idx = timeSecToNodeIndex(timeSec, nodeActiveStartSimSec);
          if (idx < 0 || idx >= flowGraph.nodes.length) return;
          const node = flowGraph.nodes[idx];
          if (node === undefined) return;
          flowCanvas.selectNode(node.id); // 캔버스 아웃라인 강조 (onSelectNode 에코 없음)
          timelinePanel.setActiveIndex(idx); // Timeline 커서도 그 노드로 (트라이페인 정합)
          appLog('info', `충돌 @${timeSec.toFixed(3)}s → 당시 노드 '${node.id}' (${node.kind}) 강조`);
        },
      });
      built.collisionPanel = collisionPanel;
      // 첫 충돌 안내 1회 + 오버레이/상태줄이 읽는 누적 카운트 (씬 리셋 시 함께 되돌린다)
      let firstCollisionNoticed = false;
      let collisionCountForOverlay = 0;
      let lastCollisionPairForOverlay: string | null = null;
      const consolePanel = createConsolePanel();
      built.consolePanel = consolePanel;
      built.dock = mountDock(
        workspace.slots.dock,
        [
          { id: DOCK_TAB_ID.timeline, label: '타임라인', content: timelinePanel.el },
          { id: DOCK_TAB_ID.collision, label: '충돌 로그', content: collisionPanel.el },
          { id: DOCK_TAB_ID.console, label: '콘솔', content: consolePanel.el },
        ],
        {
          // 접혀 있어도 "1/7 · 2.66s ▓▓░░"가 탭바에 남는다 — 접기의 정보 손실이 0이다
          strip: timelinePanel.stripEl,
          initialCollapsed: workspace.isDockCollapsed(),
          // 독은 자기 본문만 접는다 — 그리드 슬롯을 실제로 줄이는 건 워크스페이스다.
          // 이 배선이 없으면 접어도 뷰포트가 1px도 커지지 않는다 (UX_AUDIT C-1).
          onCollapseChange: (collapsed) => {
            workspace.setDockCollapsed(collapsed);
          },
          // 충돌 로그를 열면 미확인 카운트를 리셋해 배지 소스와 동기화한다
          onTabActivated: (tabId) => {
            if (tabId === DOCK_TAB_ID.collision) {
              collisionPanel.resetUnseen();
              built.dock?.setBadge(DOCK_TAB_ID.collision, 0);
            }
          },
        },
      );
      // 독의 fixed 오버레이 기본값을 슬롯 흐름으로 중화 (workspace 편입)
      Object.assign(built.dock.el.style, {
        position: 'static',
        left: 'auto',
        right: 'auto',
        bottom: 'auto',
        height: '100%',
      } satisfies Partial<CSSStyleDeclaration>);

      // 충돌 접촉점 마커 — 씬 루트에 부착(월드 좌표 그대로). 씬 수명과 함께 정리된다.
      const contactMarkers = mountContactMarkers(render.scene);
      built.contactMarkers = contactMarkers;

      // 충돌 → 로그 패널 행 추가 + start 시 관련 오브젝트 빨강 펄스 + 접촉점 마커
      // (UX_DESIGN §3.3 "충돌 오브젝트 하이라이트 + 접촉점 마커" / §3.6 로그)
      built.offMonitor = monitor.subscribe((e) => {
        const contactClass = classifyCollision(e);
        const isRealCollision = isCollision(contactClass);
        collisionPanel.addEvent(e, contactClass);
        // 실행 기록 (Phase 12 ⑦) — begin 전이면 레코더가 무시한다. 물리 phase 'stop'은
        // 기록 스키마의 'end'로 매핑한다 (runCollisionSchema — entities.ts).
        runRecorder.recordCollision({
          atSimSec: e.timeSec,
          entityA: e.a,
          entityB: e.b,
          phase: e.phase === 'start' ? 'start' : 'end',
          nodeId: activeFlowNodeId,
          classification: isRealCollision ? 'unexpected' : 'intended',
        });
        // 비활성 탭에 쌓인 충돌을 탭 배지로 표면화한다 (UX_AUDIT C-7): 구 구현은
        // waitForCollision을 포함한 시퀀스가 완주해도 충돌이 있었다는 표시가 화면
        // 어디에도 없었다 — 이 제품의 존재 이유가 3번째 탭 뒤에 숨어 있었다.
        // 배지·카운터·토스트·펄스는 **진짜 충돌만** 센다. 집으려는 박스에 손을 대는 것은
        // 시퀀스가 의도한 성공이지 사고가 아니다 — 그것까지 세면 진짜 사고가 소음에 묻힌다.
        built.dock?.setBadge(DOCK_TAB_ID.collision, collisionPanel.unseenCount());
        if (e.phase === 'start' && isRealCollision) {
          collisionCountForOverlay += 1;
          lastCollisionPairForOverlay = `${e.a} × ${e.b}`;
        }
        viewportStatusRef?.setCollisionCount(collisionCountForOverlay);
        if (e.phase !== 'start') return;
        // 첫 충돌 1회만 안내한다 — 매번 띄우면 학습적으로 무시된다
        if (isRealCollision && !firstCollisionNoticed) {
          firstCollisionNoticed = true;
          showToast(
            `예기치 않은 충돌 — ${e.a} × ${e.b}. 하단 독의 «충돌 로그»를 확인하세요`,
            'warn',
          );
        }
        // "어디서" 부딪혔는지 — 물리에서 온 월드 접촉점 (sensor는 접촉점이 없다)
        if (e.point) contactMarkers.spawn(e.point, e.normal);
        // "무엇이" 부딪혔는지 — 관련 엔티티 펄스. **진짜 충돌만** 빨갛게 깜빡인다.
        if (!isRealCollision) return;
        for (const entityId of [e.a, e.b]) {
          if (entityId === GROUND_ENTITY_ID) continue; // 바닥 전체 펄스는 소음 — 제외
          const node = visualNodeOf(entityId);
          if (node) pulseEntity(node);
        }
      });

      // 인스펙터 핸들 — 아래 재생 컨트롤·onTick 클로저가 참조하므로 먼저 선언한다
      // (마운트는 우측 패널 스택 조립 시점 — window.__sim 배선 이후)
      let inspectorRef: InspectorHandle | null = null;
      let sceneOutlinerRef: SceneOutlinerHandle | null = null;
      let inspectorLastRefreshMs = 0;
      let inspectorLastEngineState: EngineState = 'idle';

      // ── 시퀀스 arm + 결정론적 되감기 (Phase 10 Orchestrator가 재생을 제어) ──
      // 실행 직전 재검증 (§2.9 "검증 통과본만 실행"): 마지막 커밋 이후의 씬 편집(로봇
      // rename/제거 등)으로 참조가 깨진 시퀀스를 arm하면 엔진 preStep의 RobotRegistry.get이
      // 던져 tick 루프가 죽는다 — arm을 거부하고 한국어 오류를 표면화한다. 엔진 재생(씬
      // 물리)은 시퀀스와 무관하게 진행된다.
      const armFromStart = (): boolean => {
        if (!currentSequence) {
          sequenceArmed = false;
          return false;
        }
        const revalidation = validateSequence(currentSequence, editor.spec);
        if (!revalidation.ok) {
          const detail = revalidation.errors.join('\n');
          appLog('error', `시퀀스 재검증 실패 — 재생을 거부합니다:\n${detail}`);
          showToast(`시퀀스가 현재 씬과 맞지 않아 재생할 수 없습니다:\n${detail}`, 'warn');
          sequenceArmed = false;
          return false;
        }
        // 새 재생 런 — 이전 런의 active 이력·노드 시작 시각을 리셋 (load 통지가 상태를 다시 그린다)
        flowEverActiveNodeIds.clear();
        nodeActiveStartSimSec.length = 0;
        player.load(currentSequence);
        sequenceArmed = true;
        sequenceDirtySinceRun = false; // 새 런이 편집본을 처음부터 싣는다
        return true;
      };

      // ▶ Play용: 이미 armed(재개)면 no-op — 처음부터 재로드하지 않는다(mid-run 재개).
      // Phase 8: 그래프 편집이 unarm하므로, 다음 Play가 "현재" 시퀀스를 새로 로드한다.
      const armSequenceIfAvailable = (): void => {
        if (sequenceArmed || !currentSequence) return;
        const seqId = currentSequence.id;
        const count = currentSequence.steps.length;
        if (armFromStart()) appLog('info', `시퀀스 '${seqId}' 재생 시작 (${count}개 step)`);
      };

      // Orchestrator resetScene 훅 — 결정론적 되감기: 씬 pose → 충돌 이력 → player 커서.
      // orchestrator.stop()/runFromNode()가 자신의 resetting 가드(withReset) 안에서 호출하므로
      // armFromStart의 player.load 커서 통지는 무시되고, 이후 명시적 recompute가 상태를 그린다.
      // 매 되감기마다 재검증(armFromStart)하므로 stop 이후의 편집도 여기서 걸러진다.
      const resetScene = (): void => {
        sceneHandle.reset();
        monitor.clear();
        collisionPanel.clear();
        contactMarkers.clear(); // 이전 실행의 접촉 마커가 되감기 후 남지 않게
        collisionCountForOverlay = 0;
        lastCollisionPairForOverlay = null;
        firstCollisionNoticed = false;
        built.dock?.setBadge(DOCK_TAB_ID.collision, 0);
        viewportStatusRef?.setCollisionCount(0);
        armFromStart();
      };

      // 재생 컨트롤은 Orchestrator를 경유한다(§5): Play/Pause/Stop/Step/속도가 모두 노드 단위
      // 오케스트레이션 계층을 통과해 파사드·사람 조작이 같은 진실을 본다. ⏭ Step은 물리 1
      // tick이 아니라 "노드 1개"다(§5) — 물리-tick 프레임 스텝(engine.stepOnce)은 UI에서 내린다.
      /**
       * 실행 상태를 body에 반영한다 (커맨드바 하단 액센트 진행 스트립의 소스).
       *
       * **재생이 그래프 페인 크기를 바꾸지 않는다.** 이전 구현은 ▶Play에서 flowMode를
       * strip(56px)으로 접었는데, 노드가 갑자기 쪼그라들어 읽던 내용을 잃는 대가가
       * 뷰포트 몇십 px보다 컸다. 페인 크기는 사용자가 스플리터로 정한 값 그대로 둔다.
       *
       * 그리고 이 함수는 **Play/Pause/Stop 콜백이 아니라 tick의 진실에서 호출된다** —
       * 콜백에만 걸면 시퀀스가 자연 종료될 때(사용자가 아무것도 누르지 않았을 때)
       * 아무도 상태를 되돌리지 않아 영원히 '실행 중'으로 남는다.
       */
      const paintRunState = (
        seqRunning: boolean,
        seqDone: boolean,
        engineState: EngineState,
      ): void => {
        // 완주는 '일시정지'가 아니라 '끝'이다 — 완료 시 엔진을 세우므로 engineState는
        // 'paused'지만, 사용자에게 이 상태의 의미는 idle(다시 실행 가능)이다.
        document.body.dataset.runState = seqRunning
          ? 'running'
          : !seqDone && engineState === 'paused'
            ? 'paused'
            : 'idle';
      };

      /**
       * 시퀀스가 방금 완주했는가 (running → done 전이) — 완료 처리를 1회만 하기 위한 래치.
       */
      let seqCompletionHandled = false;

      // 실행 기록 시작 — 작업 컨텍스트 + 시퀀스가 있을 때만 (finishRun과 한 쌍, 위 선언부)
      runSimTimeSec = () => engine.simTimeSec;
      const beginRunIfPossible = (): void => {
        if (runRecorder.isActive()) return;
        const ctx = consoleRef?.currentTaskInfo() ?? null;
        if (ctx === null) return;
        if (flowGraph.nodes.length === 0) return;
        const operator = consoleRef?.operator() ?? { id: 'local-user', name: '로컬 사용자' };
        runSawAutoPause = false;
        runWallStartMs = Date.now();
        runRecorder.begin({
          taskId: ctx.taskId,
          taskName: ctx.taskName,
          taskVersion: ctx.taskVersion,
          processId: ctx.processId,
          operatorId: operator.id,
          operatorName: operator.name,
          stepsTotal: flowGraph.nodes.filter((n) => n.enabled).length,
          startedAtIso: new Date().toISOString(),
        });
      };

      const playbackControls = {
        play: (): void => {
          // 완주 후 ▶는 "다시 실행"이다. armSequenceIfAvailable은 sequenceArmed가 true면
          // 조기 반환하므로, 완주 상태(player.status==='done')에서는 아무 일도 일어나지
          // 않았다 — 정지를 눌러야만 동작하던 원인. 처음부터 결정론적으로 되감는다.
          // "새 런을 시작해야 하는가"의 단일 판정: 완주 후 ▶(다시 실행) **또는** 마지막
          // 재생 이후 편집됨. 둘 다 ⏹ → ▶와 같은 경로(씬·충돌 이력·player 전부 되감기)를
          // 타야 편집본이 깨끗한 초기 상태에서 돈다.
          if (sequenceDirtySinceRun || (sequenceArmed && player.status === 'done')) {
            // ⏹ 정지 → ▶ 재생과 **같은 경로**다(게이트가 이미 증명하는 결정론적 리플레이).
            // resetScene()만 부르면 씬 바디는 되감기지만 simTimeSec이 이어져,
            // 같은 "처음부터"인데 정지 경로와 시간 표시가 달라진다.
            orchestrator.stop();
            seqCompletionHandled = false;
            appLog('info', '처음부터 실행 — 편집본을 초기 상태에서 재생합니다');
          } else {
            armSequenceIfAvailable();
          }
          beginRunIfPossible(); // 작업 컨텍스트가 있으면 이 재생부터 실행 기록 (Phase 12 ⑦)
          runRecorder.recordIntervention('play', activeFlowNodeId, engine.simTimeSec);
          orchestrator.play();
          refreshOverlay(); // engine.play() 직후 동기 갱신 — rAF 지연 없이 'Running · node k/n' 전이 (§5)
        },
        pause: (): void => {
          runRecorder.recordIntervention('pause', activeFlowNodeId, engine.simTimeSec);
          orchestrator.pause();
          refreshOverlay(); // 'Paused' 전이를 즉시 반영 (정지에는 onActiveNode 방출이 없다)
        },
        stop: (): void => {
          // 기록 마감은 orchestrator.stop() **전에** — 정지가 simTime을 되감는다
          runRecorder.recordIntervention('stop', activeFlowNodeId, engine.simTimeSec);
          finishRun(runSawAutoPause ? 'autoPaused' : 'stopped');
          orchestrator.stop();
          refreshOverlay(); // 'Idle' 전이를 즉시 반영
        },
        stepOnce: (): void => {
          // ▶와 같은 판정 — 편집 후 첫 ⏭도 이전 런의 끝 상태에서 이어지면 안 된다
          if (sequenceDirtySinceRun) {
            orchestrator.stop();
            seqCompletionHandled = false;
          }
          armSequenceIfAvailable(); // 시퀀스 arm 없이는 노드 경계가 없어 무한 재생 → 먼저 arm
          // 완주 상태에서 ⏭는 orchestrator.step()이 no-op이다 — 기록만 열면 stepsDone 0의
          // 유령 런이 append-only 로그에 남아 통계(성공률·runCount)를 영구 오염시킨다.
          if (sequenceArmed && player.status === 'done') return;
          beginRunIfPossible();
          runRecorder.recordIntervention('stepNode', activeFlowNodeId, engine.simTimeSec);
          orchestrator.step();
          refreshOverlay(); // 노드 스텝 재생 전이를 즉시 반영 (경계 정지는 onTick이 뒤따라 반영)
          // 노드 스텝은 엔진을 잠시 재생 후 경계에서 멈춘다 — 인스펙터는 onTick 상태 전이로 갱신됨
          inspectorRef?.refresh();
        sceneOutlinerRef?.refresh();
        },
        setSpeed: (speedMult: number): void => {
          // select 옵션은 ENGINE_SPEED_OPTIONS에서 생성되므로 항상 유효하다
          orchestrator.setSpeed(speedMult as EngineSpeed);
        },
      };

      // 재생 컨트롤은 커맨드바 중앙 슬롯에 — 씬마다 재마운트한다 (속도 select 등
      // 뷰 상태가 씬을 가로질러 새지 않는다: 새 엔진의 기본 속도 1×와 표시가 일치)
      const playbackBar = mountPlaybackBar(
        commandBar.rowBTransport,
        playbackControls,
        ENGINE_SPEED_OPTIONS,
      );
      built.playbackBar = playbackBar;
      currentPlayback = playbackControls;
      activePlaybackBar = playbackBar;

      // ⚙ '충돌 시 자동 정지' 토글 (§5 "예기치 않은 충돌 시 자동 ⏸") — 재생 바 옆, 기본 off.
      // 켜면 로봇–환경/사물의 비의도 충돌(waitForCollision 대상이 아닌 로봇 접촉)에서 자동
      // 일시정지 + 해당 노드 강조. 파사드 setAutoPause가 이 체크박스를 동기화한다(게이트 가시 플래그).
      const autoPauseLabel = document.createElement('label');
      Object.assign(autoPauseLabel.style, {
        display: 'inline-flex',
        alignItems: 'center',
        gap: SPACE.xs,
        marginLeft: '10px',
        color: COLOR.label,
        fontFamily: FONT.ui,
        fontSize: '11px',
        whiteSpace: 'nowrap',
        cursor: 'pointer',
        userSelect: 'none',
      } satisfies Partial<CSSStyleDeclaration>);
      const autoPauseCheckbox = document.createElement('input');
      autoPauseCheckbox.type = 'checkbox';
      autoPauseCheckbox.dataset.testid = 'autopause-toggle';
      autoPauseCheckbox.setAttribute('aria-label', '충돌 시 자동 정지');
      autoPauseCheckbox.addEventListener('change', () => {
        orchestrator.setAutoPauseOnCollision(autoPauseCheckbox.checked);
      });
      const autoPauseText = document.createElement('span');
      autoPauseText.textContent = '충돌 시 자동 정지';
      autoPauseLabel.appendChild(autoPauseCheckbox);
      autoPauseLabel.appendChild(autoPauseText);
      autoPauseLabel.title = '예기치 않은 로봇 충돌에서 자동 ⏸ + 해당 노드 강조 (§5)';
      setCommandBarPriority(autoPauseLabel, COMMAND_BAR_PRIORITY.misc);
      commandBar.rowBTransport.appendChild(autoPauseLabel);
      built.autoPauseControl = autoPauseLabel;

      // 뷰포트 좌하단 실행 오버레이 (UX_DESIGN §3.3/§5) — Phase 10 오케스트레이션 배지가
      // 기존 statusline의 상태 라인을 대체한다. statusline은 "빈 씬 중앙 안내"만 담당하도록
      // 상태 라인 el을 숨겨 남긴다(중앙 안내는 setEmptyHintVisible로 씬 편집과 계속 연동).
      const viewportStatus: ViewportStatusHandle = mountViewportStatus(workspace.slots.viewport, {
        onFocusCollisionLog: () => {
          built.dock?.activateTab(DOCK_TAB_ID.collision);
        },
        sceneName: spec.name,
        emptyScene: spec.entities.length === 0,
      });
      viewportStatus.el.style.display = 'none'; // run-overlay가 상태 라인을 대체 (중앙 빈 씬 안내만 유지)
      built.viewportStatus = viewportStatus;
      viewportStatusRef = viewportStatus;

      // F 키 / 충돌 로그 행 클릭이 쓰는 카메라 프레이밍 (UX_DESIGN §9).
      // 구 구현에는 프레이밍이 아예 없어서, 충돌 로그의 "오브젝트 포커스" 안내가
      // 실제로는 펄스만 시켰다 — 대상이 화면 밖이면 사용자는 아무것도 보지 못했다.
      resetCameraView = (): void => {
        render.resetCamera();
      };
      focusSelectedEntity = (): void => {
        const id = interaction.selectedId;
        if (id === null) {
          render.resetCamera();
          return;
        }
        const node = visualNodeOf(id);
        if (node) render.frameObject(node);
      };

      // 선택 조작 HUD (뷰포트 우하단) — 이동 키 안내 + 치수 한 칸 조정 + 바닥에 붙이기.
      // 이미 구현돼 있던 방향키 이동과 치수 편집이 화면 어디에도 드러나지 않아 발견되지
      // 않던 문제를 메운다 (ui/viewport/selection-hud.ts 헤더).
      const selectionHud = mountSelectionHud(workspace.slots.viewport, {
        stepDimension: (id, shape) => {
          applyDimensions(id, shape);
        },
        snapToGround: () => {
          snapSelectionToGround(); // End 키와 같은 함수 — 동작 분기 금지
        },
        clearSelection: () => {
          interaction.select(null);
        },
      });
      built.selectionHud = selectionHud;

      /** HUD를 현재 선택/스펙으로 재동기화 (선택 변경·편집 통지 양쪽에서 호출) */
      const refreshSelectionHud = (): void => {
        const id = interaction.selectedId;
        const entity =
          id === null ? undefined : editor.spec.entities.find((e) => e.id === id);
        selectionHud.setSelection(entity ?? null);
      };

      const runOverlay = mountRunOverlay(workspace.slots.viewport);

      // 실행 계측 HUD (UX_AUDIT C-15) — 시뮬레이터를 자처하는데 FPS/RTF/스텝 지표가
      // 하나도 없었다. CLAUDE.md §2.3의 고정 timestep 결정론과 PRD NFR-2의 프레임 예산이
      // 런타임에 **관측 가능한 사실**이 된다. Gazebo/Isaac Sim 사용자는 RTF를 먼저 본다.
      const statsHud = mountStatsHud(workspace.slots.viewport);
      built.statsHud = statsHud;
      let lastStatsSimSec = engine.simTimeSec;
      let lastStatsWallMs = performance.now();
      const specColliderCount = (): number =>
        editor.spec.entities.reduce((n, e) => n + (e.physics?.colliders.length ?? 0), 0);

      // 드래그 배치 힌트 (UX_AUDIT C-17) — 라이브러리 카드를 끌 때 "여기에 놓을 수 있다"
      const dropHint = mountDropHint(workspace.slots.viewport);
      built.dropHint = dropHint;
      activeDropHint = dropHint;
      built.runOverlay = runOverlay;
      /** 마지막 오버레이 스냅샷 — 파사드 overlayText()가 순수 요약(run-overlay.overlaySummary)으로 그린다. */
      let lastOverlayState: RunOverlayState = {
        engineState: 'idle',
        simTimeSec: 0,
        activeNodeLabel: null,
        nodeIndex: null,
        nodeCount: null,
        sceneName: spec.name,
      };

      /**
       * 오버레이 1틱 스냅샷 계산 (순수 — run-overlay.overlaySummary가 그린다). 시퀀스 실행
       * 상태를 반영한다: 물리만 도는 대기(미arm/done)는 'Idle', armed+running이면 엔진 상태
       * (playing/paused)를 비춘다. 노드 진행(node k/n)은 armed일 때만, 활성 라벨은 Orchestrator
       * 활성 노드(activeFlowNodeId)에서 파생한다.
       */
      const computeOverlayState = (
        engineStateStr: string,
        simTimeSec: number,
      ): RunOverlayState => {
        const seqRunning = sequenceArmed && player.status === 'running';
        const activeNode =
          activeFlowNodeId !== null
            ? flowGraph.nodes.find((n) => n.id === activeFlowNodeId)
            : undefined;
        return {
          engineState: seqRunning ? engineStateStr : 'idle',
          simTimeSec,
          activeNodeLabel: activeNode ? kindMeta(activeNode.kind).label : null,
          nodeIndex: sequenceArmed && currentSequence ? player.currentStepIndex : null,
          nodeCount: sequenceArmed && currentSequence ? player.stepCount : null,
          sceneName: spec.name,
          // 선택 상태를 뷰포트에 상시 노출 — 우측 패널은 스크롤로 가려질 수 있고,
          // 선택이 풀린 줄 모른 채 방향키를 누르는 실패 연쇄가 여기서 끊긴다.
          selectedEntityId: interaction.selectedId,
          // 충돌 축 (UX_AUDIT C-7) — '충돌 N건' 세그먼트 + 3초 스로틀 요약 발화.
          // 0건도 표시한다: "감지가 돌고 있다"는 신호 자체가 정보다.
          collisionCount: collisionCountForOverlay,
          lastCollisionPair: lastCollisionPairForOverlay,
        };
      };

      /**
       * 뷰포트 오버레이를 현재 진실(엔진 상태·시계)로 다시 그린다. rAF onTick 외에도
       * Orchestrator 방출(onActiveNode)·재생 컨트롤(play/pause/stop/step) **직후** 호출해,
       * 오버레이의 Running/Paused/node-progress 전이가 그래프 active dot·Timeline 커서와
       * **같은 동기 지점**에서 일어나게 한다 — §5 "활성 노드 ↔ 로봇 동작 ↔ Timeline 커서
       * 항상 일치"의 뷰포트 축을 rAF 한 프레임 지연 없이 잠근다 (Phase 10). 인자 생략 시
       * engine 라이브 값을 읽는다(engine.play/pause/stop이 상태를 동기적으로 전이시킨 뒤).
       */
      const refreshOverlay = (
        engineStateStr: string = engine.state,
        simTimeSec: number = engine.simTimeSec,
      ): void => {
        lastOverlayState = computeOverlayState(engineStateStr, simTimeSec);
        runOverlay.setState(lastOverlayState);
      };

      // 타임라인: 검증된 시퀀스의 step 마커 (player 커서 연동은 flow 섹션의
      // onStepChange 구독이 타임라인 + 캔버스 상태를 함께 갱신한다 — Phase 8)
      if (validSequence) {
        timelinePanel.setSequence(validSequence.steps.map((step) => step.kind));
      }

      // rAF당 1회: 재생 바 + 타임라인 리드아웃 + 상호작용 헬퍼 갱신 (물리 tick과 분리)
      built.offTick = engine.onTick((info) => {
        {
          // 벽시계 0.25s 창으로 RTF를 갱신한다 — 매 tick 갱신은 숫자가 떨려 읽을 수 없다
          const nowMs = performance.now();
          const wallSec = (nowMs - lastStatsWallMs) / 1000;
          if (wallSec >= 0.25) {
            statsHud.update({
              rtf: computeRtf(info.simTimeSec - lastStatsSimSec, wallSec),
              entityCount: editor.spec.entities.length,
              colliderCount: specColliderCount(),
            });
            lastStatsSimSec = info.simTimeSec;
            lastStatsWallMs = nowMs;
          }
        }
        interaction.update(); // BoxHelper 아웃라인이 이동/애니메이션 중 선택 대상을 따라간다
        // 접촉점 마커 감쇠 — 벽시계 기준이라 일시정지 중에도 자연스럽게 사라진다
        contactMarkers.update(performance.now());
        paintGizmoBar(); // W/E/R 단축키로 바뀐 모드를 버튼 상태에 반영 (변경 시에만 DOM)
        // 시퀀스 진행의 단일 진실 — 물리 엔진 상태와 분리된다. 물리는 시퀀스가 done이
        // 된 뒤에도 계속 돌기 때문에(오브젝트가 계속 정착해야 한다), 이걸 engineState로
        // 판정하면 완주 후에도 영원히 '실행 중'으로 남는다.
        const seqRunningNow =
          sequenceArmed && info.state === 'playing' && player.status === 'running';
        const seqDoneNow = sequenceArmed && player.status === 'done';

        // 시퀀스가 끝나면 **실행도 끝난다.** 구 동작은 물리 루프가 계속 돌아 simTime이
        // 영원히 올라갔고, 사용자에게는 "끝났는데 안 멈춘다"로 보였다.
        // 씬 상태는 보존한다(⏹ 정지와 달리 리셋하지 않는다) — 결과를 그대로 관찰할 수 있다.
        if (seqDoneNow && !seqCompletionHandled) {
          seqCompletionHandled = true;
          // 완주 시점에 실행 기록 마감 — 오류 노드가 있었으면 'error'로 남긴다 (Phase 12 ⑦)
          finishRun(Object.values(flowStatuses).includes('error') ? 'error' : 'completed');
          if (info.state === 'playing') {
            orchestrator.pause();
            appLog('info', '시퀀스 완주 — 실행 종료 (▶ 다시 실행으로 처음부터)');
          }
        } else if (!seqDoneNow) {
          seqCompletionHandled = false;
        }

        paintRunState(seqRunningNow, seqDoneNow, info.state);
        playbackBar.update({
          engineState: info.state,
          simTimeSec: info.simTimeSec,
          sequence: currentSequence
            ? {
                // 엔진 idle에서는 armed 여부와 무관하게 대기 라벨을 보인다 — ⏹ Stop 후
                // player.reset()은 커서를 되감으며 'running'으로 두지만(ControlPlayer.reset
                // 계약) 엔진 tick이 없어 진행되지 않는 상태다. 'running' 표기는 오해를
                // 부르므로 실제 재개 수단(▶ Play)을 안내한다.
                status:
                  sequenceArmed && info.state !== 'idle' ? player.status : '대기 (▶ Play)',
                stepIndex: player.currentStepIndex,
                // 미arm(편집 직후 포함) 상태에서는 player가 이전 시퀀스를 물고 있을 수
                // 있다 — 총 step 수는 항상 현재 라이브 시퀀스 기준으로 보인다 (Phase 8)
                stepCount: sequenceArmed ? player.stepCount : currentSequence.steps.length,
                running: seqRunningNow,
                done: seqDoneNow,
              }
            : undefined,
        });
        timelinePanel.setSimTime(info.simTimeSec);
        // 실행 오버레이 (§5 트라이페인 뷰포트 축): rAF마다 simTime을 흘려보낸다. 재생 상태·
        // 노드 진행 전이는 refreshOverlay(=computeOverlayState)가 Orchestrator 방출(onActiveNode)·
        // 재생 컨트롤(play/pause/stop/step)과 같은 동기 지점에서도 밀어넣으므로, 뷰포트 오버레이가
        // 그래프 active dot·Timeline 커서와 한 프레임도 어긋나지 않는다(Play 순간 포함 — Phase 10).
        refreshOverlay(info.state, info.simTimeSec);

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
        sceneOutlinerRef?.refresh();
          }
        } else if (inspectorStateChanged) {
          inspectorRef?.refresh();
        sceneOutlinerRef?.refresh();
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

      const playerFacade: SimPlayerFacade | undefined = validSequence
        ? {
            get status() {
              return player.status;
            },
            get currentStepIndex() {
              return player.currentStepIndex;
            },
            get stepCount() {
              // Play 전(미arm — 그래프 편집 직후 포함)에도 "현재 라이브 시퀀스"의
              // step 수를 보고한다 — "로드 가능" 상태 표면 (Phase 8)
              return sequenceArmed ? player.stepCount : (currentSequence?.steps.length ?? 0);
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

      /**
       * 드롭 좌표(null = 뷰포트 중앙)의 바닥 레이캐스트 지점에 배치 — y는 템플릿 유지.
       * 템플릿/임포트가 어떤 y를 들고 오든 최저점이 바닥 위에 오도록 마지막에 클램프한다
       * (지하 스폰 = 되돌릴 길 없는 손실 — core/ground-clamp.ts 헤더).
       */
      const placeEntity = async (
        entity: EntitySpec,
        dropClient: { x: number; y: number } | null,
      ): Promise<string> => {
        const client = dropClient ?? viewportCenterClient();
        const ground = interaction.raycastGround(client.x, client.y);
        if (ground) {
          entity.transform.position = [ground[0], entity.transform.position[1], ground[2]];
        }
        if (editor.spec.environment?.ground === true) {
          entity.transform.position = clampPositionAboveGround(
            entity,
            entity.transform.position,
          ).position;
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
        updateConveyor: (id, conveyor) => editor.updateConveyor(id, conveyor),
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
        anchorProbe: () => interaction.anchorProbe(),
        anchorScreenPoint: () => interaction.anchorScreenPoint(),
      };

      // (window.__sim 배선은 flow 파사드까지 조립된 뒤 — 아래 Flow Graph 섹션 끝)

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

      /** 패널을 스택 흐름(static)으로 편입 — 모듈 기본 절대 배치/자체 폭 제약을 해제.
       *  zIndex도 auto로 되돌린다: 단독 마운트 기본값(Z_INDEX.panel=100)이 flex 아이템
       *  으로 남으면 {} JSON 슬라이드 패널(95) 위에 그려져 클릭을 가로챈다 — 우측
       *  스택은 슬라이드 패널보다 아래가 규약이다 (ui/theme.ts Z_INDEX 주석).
       *
       *  order: DOM 순서와 무관하게 **편집 폼이 항상 스택 맨 위**에 오게 한다. 마운트
       *  순서(관절 패널 → 인스펙터 → 편집 폼)를 그대로 그리면 로봇 선택 시 관절 슬라이더
       *  와 읽기 전용 관절 표가 편집 폼을 화면 밖으로 밀어낸다(실측: 1600×950에서
       *  ee-pos-x까지 최대 865 px 스크롤 — 로봇의 Transform 입력이 사실상 도달 불가).
       *  오브젝트는 같은 자리에서 스크롤 0 px이었다 — 그 비대칭을 없앤다. */
      const adoptIntoStack = (panelEl: HTMLElement, order: number): void => {
        Object.assign(panelEl.style, {
          position: 'static',
          top: 'auto',
          right: 'auto',
          width: '100%',
          minWidth: '0',
          maxHeight: 'none',
          minHeight: '0',
          flex: '0 1 auto',
          zIndex: 'auto',
          order: String(order),
        } satisfies Partial<CSSStyleDeclaration>);
      };

      // 로봇이 있는 씬이면 임시 관절 패널 마운트 (ROADMAP Phase 3 "슬라이더 수동 제어").
      // 로봇 구성이 바뀌면(라이브러리 드롭으로 추가·삭제·개명) 패널을 다시 만든다 —
      // 빌드 시점 스냅샷으로 두면 "빈 씬에 팔 2대를 놓고 관절을 움직여 충돌시킨다"는
      // 시나리오가 UI만으로는 끝까지 갈 수 없다(패널 자체가 생기지 않았다).
      // 슬라이더 초기값은 core 진실(readJoints)에서 읽으므로 재마운트로 값이 튀지 않는다.
      let robotSetSignature = '';
      const syncJointPanel = (): void => {
        const ids = sceneHandle.robots.ids();
        const signature = ids.join(' ');
        if (signature === robotSetSignature) return;
        robotSetSignature = signature;
        built.jointPanel?.dispose();
        built.jointPanel = undefined;
        if (ids.length === 0) return;
        const panel = mountJointPanel(
          rightStack,
          ids.map((robotId) => ({
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
        );
        built.jointPanel = panel;
        // DOM 순서가 아니라 flex order가 스택 위치를 정한다 — 나중에 붙어도 맨 아래다
        adoptIntoStack(panel.el, RIGHT_STACK_ORDER.jointPanel);
      };
      syncJointPanel();

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
        // 목록(아웃라이너)은 좌 패널이 소유한다 — 아웃라이너 ≠ 프로퍼티 (UX_AUDIT C-16).
        // 한 컬럼에 두면 속성 폼이 길어질 때 목록이 화면 밖으로 밀려, 선택을 바꾸려면
        // 스크롤부터 해야 했다(로봇 선택 시 최대 865px).
      }, { showList: false });
      built.inspector = inspector;

      // 씬 아웃라이너 — 좌 패널 하단(라이브러리 아래). 좌 패널 콘텐츠는 y≈540에서 끝나고
      // 329px가 비어 있었으므로 추가 화면 비용은 0이다.
      const sceneOutliner = mountSceneOutliner(workspace.slots.left, {
        listEntities: () => editor.spec.entities.map((e) => ({ id: e.id, type: e.type })),
        onSelect: (id) => {
          interaction.select(id); // 선택 진실은 interaction — 변경 가드가 루프를 막는다
        },
      });
      Object.assign(sceneOutliner.el.style, {
        flex: '1 1 42%',
        minHeight: '120px',
        borderRadius: '0',
        borderLeft: 'none',
        borderRight: 'none',
        borderBottom: 'none',
      } satisfies Partial<CSSStyleDeclaration>);
      built.sceneOutliner = sceneOutliner;
      sceneOutlinerRef = sceneOutliner;
      adoptIntoStack(inspector.el, RIGHT_STACK_ORDER.inspector);
      inspectorRef = inspector;

      // 엔티티 편집 폼 (UX §3.5 (A) — 이름/Transform/Dimensions/Physics 쓰기)
      const entityEditor = mountEntityEditor(rightStack, {
        // 오브젝트를 지울 방법이 UI에 아예 없었다 — core에 removeEntity가 완전 구현돼
        // 있는데 호출부가 window.__sim 자동화 파사드뿐이라, 라이브러리에서 넣을 수만
        // 있고 뺄 수 없는 add-only 함정이었다 (UX_AUDIT C-4).
        onDeleteEntity: (id) => {
          runEdit(() => {
            editorFacade.removeEntity(id); // 재생 중 로봇 제거 가드가 안에 있다
          });
          interaction.select(null);
          toasts.show('success', `'${id}' 삭제됨`, {
            action: {
              label: '실행 취소',
              onClick: () => {
                void history.undo();
              },
            },
          });
        },
        getEntity: (id) => {
          const entity = editor.spec.entities.find((e) => e.id === id);
          return entity ? structuredClone(entity) : null;
        },
        isRobot: isRobotEntity,
        updateTransform: (id, transform) => {
          // 기즈모/방향키와 같은 규칙 — 바닥 하한 (clampAboveGround 주석)
          const position = clampAboveGround(id, transform.position, transform.rotation);
          runEdit(() => editor.updateTransform(id, { ...transform, position }));
        },
        updateDimensions: (id, shape) => {
          applyDimensions(id, shape);
        },
        updatePhysics: (id, physics) => {
          runEdit(() => editor.updatePhysics(id, physics));
        },
        updateConveyor: (id, conveyor) => {
          runEdit(() => editor.updateConveyor(id, conveyor));
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
      adoptIntoStack(entityEditor.el, RIGHT_STACK_ORDER.editForm);

      // ── Flow Graph 글루 (Phase 8): 편집 파이프라인 + 캔버스/노드 폼 + 상태 동기 ──
      // 파일 헤더의 Flow Graph 절이 규범이다. 그래프 상태는 위 라이브 시퀀스 블록의
      // flowGraph가 진실이고, 여기서는 op 적용·검증·커밋·UI 재동기화만 조립한다.

      /** 그래프의 label 노드 이름 목록 (팔레트/goto 대상 후보 — 중복 제거, 등장 순) */
      const flowLabelNames = (): string[] => {
        const names: string[] = [];
        for (const node of flowGraph.nodes) {
          if (node.kind !== 'label') continue;
          const name = node.params['name'];
          if (typeof name === 'string' && !names.includes(name)) names.push(name);
        }
        return names;
      };

      /** defaultNodeFor 컨텍스트 — 캔버스 팔레트의 비활성 판정도 이 값을 쓴다 */
      const flowPaletteContext = (): { robot: string; entityIds: string[]; labels: string[] } => ({
        robot: flowGraph.robot,
        entityIds: editor.spec.entities.map((e) => e.id),
        labels: flowLabelNames(),
      });

      /**
       * 내용이 바뀐 노드에 '수정됨' 배지(origin 'modified')를 단다 — 로드된 시퀀스
       * JSON과 달라졌음을 표시 (UX §3.4). F1의 op는 generated→modified만 승격하므로,
       * fromSequence가 'manual'로 로드한 노드의 편집은 여기(diff)서 승격한다.
       * 순서 이동만으로는 내용이 변하지 않으므로 배지가 붙지 않는다.
       */
      const withEditBadges = (prev: FlowGraph, next: FlowGraph): FlowGraph => {
        const prevById = new Map(prev.nodes.map((node) => [node.id, node]));
        const nodes = next.nodes.map((node) => {
          if (node.origin === 'modified') return node;
          const before = prevById.get(node.id);
          if (!before) return node; // 새 노드(삽입/복제)는 'manual' 유지 — 사용자 작성
          const changed =
            before.enabled !== node.enabled ||
            before.note !== node.note ||
            JSON.stringify(before.params) !== JSON.stringify(node.params);
          return changed ? { ...node, origin: 'modified' as const } : node;
        });
        return { nodes, edges: next.edges, robot: next.robot };
      };

      /**
       * 편집 커밋: 라이브 시퀀스 교체 + 재생 정지 정책 집행 + 파생 UI 재동기화.
       * 시퀀스 편집 정책(파일 헤더): armed였다면 unarm + player 커서 0 — 엔진(씬 물리)
       * 은 계속 돈다. 다음 ▶ Play가 편집본을 처음부터 재생한다.
       */
      const commitFlowSequence = (seq: ControlSequence, versionLabelKo?: string): void => {
        // 시퀀스만 바뀌어도 문서는 미저장이다 — 구 저장은 이 축을 통째로 버렸다 (C-3)
        queueMicrotask(markDocumentChanged);
        // 버전 이력에 append (라벨은 이전/이후 비교로 자동 도출 — 편집 호출부를 건드리지
        // 않아도 새 경로에 자동으로 이름이 붙는다. JSON 직접 편집/되돌리기만 명시 지정).
        sequenceVersions.record(seq, versionLabelKo === undefined ? {} : { labelKo: versionLabelKo });
        // 노드 편집(삭제·재정렬·복제·파라미터·교체 생성)을 Undo 대상으로 만든다 (C-4).
        // 구 구현에서는 15분간 손질한 시퀀스가 Del 한 번에 복구 불가로 사라졌다.
        history.flushPending(); // 직전 씬 편집 burst와 경계를 세운다
        history.noteChange(() => ({ scene: editor.serialize(), sequence: seq }));
        history.flushPending();
        const wasArmed = sequenceArmed;
        currentSequence = seq;
        // 씬이 이미 "쓰인" 상태(재생했거나 시간이 흐른 뒤)에서의 편집은 다음 ▶가
        // 처음부터 되감아야 한다 — 안 그러면 편집본이 이전 런의 끝 상태 위에서 돌아
        // "순서를 바꿨는데 로봇이 예전처럼 군다"가 된다 (sequenceDirtySinceRun 헤더).
        if (wasArmed || engine.simTimeSec > 0) sequenceDirtySinceRun = true;
        if (wasArmed) {
          sequenceArmed = false; // preStep 게이트 — 로드된 이전 시퀀스는 더 진행하지 않음
          // 런 표현 초기화(엔진/씬 무영향): player 커서 되감기(resetting 가드로 통지 무시) +
          // 상태 전부 pending.
          orchestrator.resetForEdit();
          // 재생 중이었다면 엔진도 세운다. 시퀀스만 unarm하면 **엔진은 playing인데 로봇은
          // 얼어붙고 오버레이는 Idle**인 모순 상태가 남는다 — 사용자에겐 "편집이 재생을
          // 깨뜨렸다"로 보인다.
          if (engine.state === 'playing') orchestrator.pause();
          showToast('시퀀스 수정됨 — ▶ 재생하면 처음부터 새 순서로 실행합니다', 'info');
        }
        flowEverActiveNodeIds.clear();
        timelinePanel.setSequence(seq.steps.map((step) => step.kind));
        // 새 노드 집합으로 상태 맵/타임라인 재동기 (idle이면 전부 pending — orchestrator 진실)
        orchestrator.refresh();
        jsonViewer.refresh();
      };

      /**
       * 편집 파이프라인 코어 (§2.8 게이트): op(구조 검증 포함) → 씬 참조 무결성
       * serializeGraph(scene) → '수정됨' 배지 → 커밋 → 캔버스/노드 폼 재동기화.
       * 성공 null, 실패 시 한국어 오류 목록 반환 — 그래프/시퀀스는 변하지 않는다.
       */
      const runFlowOp = (op: (g: FlowGraph) => FlowCanvasOpResult): string[] | null => {
        const structural = op(flowGraph);
        if (!structural.ok || structural.graph === undefined) {
          const errors =
            structural.errors && structural.errors.length > 0
              ? structural.errors
              : ['플로우 편집이 거부되었습니다'];
          lastFlowValidation = errors;
          return errors;
        }
        // 씬 참조 무결성(로봇/엔티티/관절)까지 — F1 op의 구조 검증(씬 없음)을 보강하는
        // UI 경로의 최종 §2.8 게이트. 원본 시퀀스의 id/loop를 복원해 직렬화한다.
        const serialized = serializeGraph(structural.graph, editor.spec, {
          id: flowSeqMeta.id,
          ...(flowSeqMeta.loop !== undefined ? { loop: flowSeqMeta.loop } : {}),
        });
        if (!serialized.ok) {
          lastFlowValidation = serialized.errors;
          return serialized.errors;
        }
        flowGraph = withEditBadges(flowGraph, structural.graph);
        lastFlowValidation = 'ok';
        // 라벨은 넘기지 않는다 — describeSequenceChange가 이전/이후 비교로 도출한다
        // (편집 호출부마다 라벨을 넘기게 하면 한 곳만 빠뜨려도 '알 수 없음'이 남는다).
        commitFlowSequence(serialized.sequence);
        flowCanvas.render();
        nodeEditor.refresh(); // 폼 편집 중(포커스)이면 내부 가드가 건너뛴다
        return null;
      };

      /**
       * 검증된 시퀀스를 그래프 진실로 **통째로 교체**한다 (JSON 직접 편집 · 버전 되돌리기).
       *
       * 노드 단위 op가 아니라 전체 교체이므로 runFlowOp를 타지 않지만, **같은 게이트를
       * 같은 순서로** 지난다: validateSequence(스키마 + 씬 참조) → fromSequence →
       * serializeGraph(재직렬화로 왕복 무결성 확인) → commit. 마지막 재직렬화는
       * 형식적으로 보이지만 §2.8의 "편집 결과는 항상 직렬화 가능"을 이 경로에서도
       * 기계적으로 보증한다 — 사람이 쓴 JSON이 들어오는 유일한 경로라 더 필요하다.
       */
      const replaceSequenceFromValidated = (
        candidate: unknown,
        versionLabelKo: string,
      ): { readonly ok: true; readonly recorded: boolean } | { readonly ok: false; readonly errors: readonly string[] } => {
        const validated = validateSequence(candidate, editor.spec);
        if (!validated.ok) return { ok: false, errors: validated.errors };
        const nextGraph = fromSequence(validated.value, { origin: 'manual' });
        const serialized = serializeGraph(nextGraph, editor.spec, {
          id: validated.value.id,
          ...(validated.value.loop !== undefined ? { loop: validated.value.loop } : {}),
        });
        if (!serialized.ok) {
          lastFlowValidation = serialized.errors;
          return { ok: false, errors: serialized.errors };
        }
        flowGraph = nextGraph;
        // 메타는 제자리 갱신한다(다른 클로저들이 이 객체를 캡처하고 있다 — 재대입 금지).
        // 사람이 쓴 JSON이 id/loop를 바꿀 수 있으므로 여기서 진실을 따라간다.
        flowSeqMeta.id = validated.value.id;
        flowSeqMeta.loop = validated.value.loop;
        lastFlowValidation = 'ok';
        const versionCountBefore = sequenceVersions.size();
        commitFlowSequence(serialized.sequence, versionLabelKo);
        flowCanvas.render();
        nodeEditor.refresh();
        // 시퀀스가 통째로 바뀌면 페인이 닫혀 있을 이유가 없다 — 결과를 보여준다
        if (serialized.sequence.steps.length > 0 && !flowPaneVisible) setFlowPaneVisible(true);
        // 내용이 직전 버전과 같으면 record가 기록하지 않는다(공백·키 순서만 바꾼 경우).
        // 안내 문구가 "이력에 남았다"고 거짓말하지 않도록 실제 기록 여부를 돌려준다.
        return { ok: true, recorded: sequenceVersions.size() > versionCountBefore };
      };

      /** {} JSON 패널의 [적용] — 텍스트 파싱은 패널이 이미 했지만 진실은 여기서 다시 판정한다 */
      applyJsonToSequence = (text): ApplyJsonResult => {
        let parsed: unknown;
        try {
          parsed = JSON.parse(text);
        } catch (err) {
          // 패널이 이미 선파싱해 줄 번호로 안내하므로 여기 도달은 드물지만,
          // 사용자에게 보이는 문자열은 어느 경로든 한국어다 (§4-b).
          return { ok: false, errors: [jsonErrorKo(err, text)] };
        }
        const result = replaceSequenceFromValidated(parsed, 'JSON 직접 편집');
        if (!result.ok) {
          appLog('error', `JSON 적용 거부: ${result.errors.join(' / ')}`);
          return result;
        }
        appLog('info', 'JSON 직접 편집 적용 — 시퀀스 교체');
        showToast(
          result.recorded
            ? 'JSON 적용됨 — 되돌리려면 {} JSON의 [버전] 탭을 쓰세요'
            : 'JSON 적용됨 (내용이 같아 새 버전은 만들지 않았습니다)',
          'info',
        );
        return { ok: true };
      };

      /** [버전] 탭의 되돌리기 — 되돌리기도 새 버전으로 append된다(되돌리기를 되돌릴 수 있다) */
      restoreSequenceVersion = (version): ApplyJsonResult => {
        const entry = sequenceVersions.get(version);
        if (entry === null) {
          return { ok: false, errors: [`v${version}은 이력에서 사라졌습니다 (상한 초과로 폐기)`] };
        }
        const result = replaceSequenceFromValidated(entry.sequence, `v${version}으로 되돌림`);
        if (!result.ok) {
          appLog('error', `되돌리기 거부(v${version}): ${result.errors.join(' / ')}`);
          return result;
        }
        appLog('info', `시퀀스 v${version}으로 되돌림 (${entry.labelKo})`);
        showToast(
          result.recorded
            ? `v${version}으로 되돌렸습니다 — 지금 상태도 이력에 남아 있습니다`
            : `v${version}과 내용이 같아 그대로입니다`,
          'info',
        );
        return { ok: true };
      };

      /** 캔버스/파사드용 래퍼: 실패를 한국어 토스트 + 콘솔 로그로 표면화 (§2.8 피드백) */
      const applyFlowOpWithToast = (op: (g: FlowGraph) => FlowCanvasOpResult): boolean => {
        const errors = runFlowOp(op);
        if (errors === null) return true;
        const detail = errors.join('\n');
        appLog('error', `플로우 편집 거부: ${detail}`);
        showToast(`플로우 편집 거부됨:\n${detail}`, 'warn');
        return false;
      };

      /**
       * 노트 편집 op — F1(schema/flow-graph)에 setNodeNote가 없어 글루가 보완한다.
       * note는 노드 필드(params 제외)이므로 updateNodeParams로는 닿지 않는다. 같은
       * §2.8 파이프라인(직렬화 검증)을 거치며, 빈 문자열은 note 키 제거로 정규화한다.
       */
      const setNodeNoteOp =
        (nodeId: string, note: string) =>
        (g: FlowGraph): FlowCanvasOpResult => {
          if (!g.nodes.some((n) => n.id === nodeId)) {
            return { ok: false, errors: [`그래프에 id '${nodeId}' 노드가 없습니다`] };
          }
          const nodes = g.nodes.map((n) => {
            if (n.id !== nodeId) return n;
            const next: FlowNode = { ...n, params: structuredClone(n.params), ui: { ...n.ui } };
            if (note === '') delete next.note;
            else next.note = note;
            return next;
          });
          const graph: FlowGraph = { nodes, edges: deriveEdges(nodes), robot: g.robot };
          const serialized = serializeGraph(graph);
          if (!serialized.ok) return { ok: false, errors: serialized.errors };
          return { ok: true, graph };
        };

      // 노드 파라미터 인스펙터 폼 (UX §3.5 (B)) — 우측 스택, 기본 숨김(선택 중재가 표시)
      const nodeEditor = mountNodeEditor(rightStack, {
        getNode: (id) => flowGraph.nodes.find((n) => n.id === id) ?? null,
        sceneContext: () => {
          const robotId = flowGraph.robot;
          const robotJoints = sceneHandle.robots.ids().includes(robotId)
            ? sceneHandle.robots.get(robotId).joints.map((joint) => ({
                name: joint.name,
                ...(joint.limits ? { limits: joint.limits } : {}),
              }))
            : [];
          const robotEntity = editor.spec.entities.find((e) => e.id === robotId);
          return {
            robot: robotId,
            robotJoints,
            entityIds: editor.spec.entities.map((e) => e.id),
            labels: flowLabelNames(),
            gripperAvailable:
              robotEntity !== undefined &&
              isRobotSpec(robotEntity) &&
              robotEntity.gripper !== undefined,
          };
        },
        // 폼 커밋: 오류는 폼 인라인 표시로 되돌린다 (토스트 아님 — node-editor 계약)
        commitParams: (id, params) => runFlowOp((g) => updateNodeParams(g, id, params)),
        setEnabled: (id, enabled) => {
          applyFlowOpWithToast((g) => setNodeEnabled(g, id, enabled));
        },
        setNote: (id, note) => {
          applyFlowOpWithToast(setNodeNoteOp(id, note));
        },
      });
      built.flowNodeEditor = nodeEditor;
      adoptIntoStack(nodeEditor.el, RIGHT_STACK_ORDER.editForm);

      /** 우측 스택 중재: 마지막 선택이 이긴다 — 노드 폼 ↔ 엔티티 폼 표시 전환 */
      const showRightPanelFor = (panel: 'entity' | 'node'): void => {
        nodeEditor.el.style.display = panel === 'node' ? '' : 'none';
        entityEditor.el.style.display = panel === 'entity' ? '' : 'none';
      };
      showRightPanelFor('entity'); // 기본: 엔티티 폼 (기존 동작 유지)

      // 캔버스 호스트 — workspace flowGraph 페인의 자리 표시 문구를 덮는 불투명 배경
      // (캔버스 svg는 투명 — 배경이 없으면 문구가 비쳐 보인다). 씬 수명과 함께 제거.
      const flowPaneHost = document.createElement('div');
      Object.assign(flowPaneHost.style, {
        position: 'absolute',
        inset: '0',
        background: COLOR.bgPanel,
      } satisfies Partial<CSSStyleDeclaration>);
      workspace.slots.flowGraph.appendChild(flowPaneHost);
      built.flowPaneHost = flowPaneHost;

      const flowCanvas = mountFlowCanvas(flowPaneHost, {
        getGraph: () => flowGraph,
        applyOp: applyFlowOpWithToast,
        // 선택 중재 (마지막 선택 승리): 노드 선택 → 뷰포트 선택 해제 + 노드 폼 표시.
        // interaction.select(null)의 onSelect(null) 에코는 패널을 바꾸지 않는다(아래).
        onSelectNode: (id) => {
          if (id !== null) {
            interaction.select(null);
            nodeEditor.showFor(id);
            showRightPanelFor('node');
          } else {
            nodeEditor.showFor(null);
            showRightPanelFor('entity');
          }
        },
        // 제자리 드롭 / 페인 밖 드롭 — 조용히 되돌리면 "드래그가 먹히지 않았다"로 읽힌다.
        // 사용자가 실제로 겪은 결함이 이것이었다(순서가 안 바뀐 채 ▶를 눌러 이전 순서 실행).
        onReorderNoop: () => {
          showToast('순서가 그대로입니다 — 옮길 자리의 노드 위로 끌어다 놓으세요', 'info');
        },
        onReorderCancelled: () => {
          showToast('재정렬 취소됨 — 플로우 그래프 안에서 놓아야 순서가 바뀝니다', 'warn');
        },
        paletteContext: flowPaletteContext,
        // 빈 플로우의 '자연어로 만들기' → 커맨드바 자연어 입력에 포커스 (UX_AUDIT C-12).
        // 구 구현은 여기서 "Phase 9에서 제공됩니다"라고 **거짓 안내**를 했다 — 플래너는
        // 이미 출시되어 있었고, 첫 사용자가 가장 도움이 필요한 순간에 이탈시켰다.
        onRequestNlFocus: () => {
          const field = nlInput?.el.querySelector<HTMLInputElement>('[data-testid="nl-text"]');
          if (field === null || field === undefined) return;
          field.focus();
          field.select();
        },
      });
      built.flowCanvas = flowCanvas;

      // ── 플래너 생성 시퀀스 로드 (Phase 9 — human-in-the-loop, §2.9) ────
      // 검증은 호출부(main의 handlePlannerResult)가 이미 수행했다. 여기서는 그래프에
      // origin 'generated'로 로드하고 §2.8 파이프라인(serializeGraph(scene))으로 한 번
      // 더 재직렬화 검증한 뒤 commit한다. player는 로드하지 않는다 — sequenceArmed=false
      // 유지 → 다음 ▶ Play가 armSequenceIfAvailable에서 재검증→로드(무자동재생 증명).
      const loadGeneratedSequence = (
        seq: ControlSequence,
        mode: GenerateMode,
      ): { ok: boolean; errors?: string[] } => {
        const useAppend =
          mode === 'append' && currentSequence !== null && flowGraph.nodes.length > 0;

        let nextGraph: FlowGraph;
        let nextId: string;
        let nextLoop: boolean | undefined;

        if (useAppend) {
          const base = currentSequence!;
          const appendedSteps = appendStepsWithLabelRename(base.steps, seq.steps);
          const mergedSeq: ControlSequence = {
            id: flowSeqMeta.id,
            robot: base.robot,
            ...(flowSeqMeta.loop !== undefined ? { loop: flowSeqMeta.loop } : {}),
            steps: appendedSteps,
          };
          // 기존 노드의 origin은 보존하고, 새로 이어 붙인 step만 'generated'로 표시한다
          // (fromSequence는 step 순서와 노드가 1:1 — 앞 k개가 기존, 뒤가 새 step).
          const oldOrigins = flowGraph.nodes.map((node) => node.origin);
          const rebuilt = fromSequence(mergedSeq, { origin: 'manual' });
          const k = oldOrigins.length;
          const nodes: FlowNode[] = rebuilt.nodes.map((node, i) =>
            i < k
              ? { ...node, origin: oldOrigins[i] ?? 'manual' }
              : { ...node, origin: 'generated' },
          );
          nextGraph = { nodes, edges: deriveEdges(nodes), robot: rebuilt.robot };
          nextId = flowSeqMeta.id;
          nextLoop = flowSeqMeta.loop;
        } else {
          nextGraph = fromSequence(seq, { origin: 'generated' });
          nextId = seq.id;
          nextLoop = seq.loop;
        }

        const serialized = serializeGraph(nextGraph, editor.spec, {
          id: nextId,
          ...(nextLoop !== undefined ? { loop: nextLoop } : {}),
        });
        if (!serialized.ok) {
          lastFlowValidation = serialized.errors;
          return { ok: false, errors: serialized.errors };
        }
        flowSeqMeta.id = nextId;
        flowSeqMeta.loop = nextLoop;
        flowGraph = nextGraph;
        lastFlowValidation = 'ok';
        commitFlowSequence(serialized.sequence);
        setFlowPaneVisible(true); // 생성된 플로우는 페인을 열어 검토 대상으로 노출
        flowCanvas.render();
        nodeEditor.refresh();
        return { ok: true };
      };

      const hasGeneratedNodes = (): boolean =>
        flowGraph.nodes.some((node) => node.origin === 'generated');

      // ── 실행 오케스트레이터 배선 (Phase 10, UX_DESIGN §5 — THE NORM) ──────────
      // Phase 8이 배선한 player.onStepChange → 캔버스 상태를 일급 오케스트레이션으로 심화한다:
      // 노드 경계 컨트롤(Play/Pause/Stop/Step) · 트라이페인 동기(활성 노드 ↔ 뷰포트 배지 ↔
      // Timeline 커서) · 충돌 인지 정지 · 결정론적 재실행. player 커서가 유일한 진실이고,
      // 오케스트레이터는 그것을 관찰해 표현 상태(노드 상태 맵·활성 노드)만 파생·방출한다.

      /**
       * 상태 맵 → Timeline 커서/오류 마커 (트라이페인 정합): active 인덱스에 커서, done
       * 노드는 완료색, error 노드는 오류 마커. active가 없고 전부 done이면 커서를 끝(노드
       * 수)에 둔다(모두 done 표기), 그 외 active 없음은 -1(대기).
       */
      const paintTimelineFromStatuses = (map: Record<string, NodeRunStatus>): void => {
        const nodes = flowGraph.nodes;
        let activeIdx = -1;
        let allDone = nodes.length > 0;
        const errorIdx: number[] = [];
        nodes.forEach((node, i) => {
          const status = map[node.id] ?? 'pending';
          if (status === 'active') activeIdx = i;
          else if (status === 'error') errorIdx.push(i);
          if (status !== 'done') allDone = false;
        });
        timelinePanel.setActiveIndex(activeIdx >= 0 ? activeIdx : allDone ? nodes.length : -1);
        timelinePanel.setErrorIndices(errorIdx);
      };

      /** 현재 그래프의 waitForCollision 배리어 쌍 목록 (비활성 배리어도 "조작 대상"으로 포함). */
      const awaitedCollisionPairs = (): Array<readonly [string, string]> => {
        const pairs: Array<readonly [string, string]> = [];
        for (const node of flowGraph.nodes) {
          if (node.kind !== 'waitForCollision') continue;
          const between = node.params['between'];
          if (
            Array.isArray(between) &&
            typeof between[0] === 'string' &&
            typeof between[1] === 'string'
          ) {
            pairs.push([between[0], between[1]]);
          }
        }
        return pairs;
      };

      /**
       * 접촉 분류 — core/collision-classify의 순수 판정에 위임한다.
       *
       * 타겟(= 시퀀스가 접촉 대기 노드로 선언한 쌍)·바닥·감지 영역·사물끼리의 접촉은
       * 충돌이 아니다. **로봇이 타겟 아닌 것에 부딪힌 경우만** 충돌이다.
       *
       * 구 판정은 "동적 사물과의 접촉은 정상 조작(밀기/파지)"이라며 옆 물건과의 충돌까지
       * 통째로 면제했는데, 그러면 "타겟 외의 것에 부딪혔다"는 사고를 놓친다.
       */
      const classifyCollision = (e: CollisionEvent): ContactClass =>
        classifyContact(e, {
          robotIds: new Set(sceneHandle.robots.ids()),
          targetPairs: awaitedCollisionPairs(),
          groundId: GROUND_ENTITY_ID,
        });

      /** §5 충돌 인지 정지 — start phase의 진짜 충돌만 자동 정지를 트리거한다 */
      const unexpectedCollision = (e: CollisionEvent): boolean =>
        e.phase === 'start' && isCollision(classifyCollision(e));

      // core Engine/Player를 오케스트레이터의 좁은 표면으로 감싼다 (Rapier/three 비노출).
      // onTick의 state 타입 완화(EngineState → string), setSpeed 배율 검증은 engine이 수행.
      const orchEngine: OrchestratorEngine = {
        play: () => engine.play(),
        pause: () => engine.pause(),
        stop: () => engine.stop(),
        setSpeed: (mult) => engine.setSpeed(mult as EngineSpeed),
        get state() {
          return engine.state;
        },
        get simTimeSec() {
          return engine.simTimeSec;
        },
        onTick: (fn) =>
          engine.onTick((info) => fn({ state: info.state, simTimeSec: info.simTimeSec })),
      };
      const orchPlayer: OrchestratorPlayer = {
        load: () => {}, // 실 arm은 armFromStart가 담당 — 오케스트레이터는 커서만 관찰한다
        reset: () => player.reset(),
        get status() {
          return player.status;
        },
        get currentStepIndex() {
          return player.currentStepIndex;
        },
        // 노드 맵은 항상 현재 그래프와 1:1 — 미arm이어도 전체 노드가 표현된다(전부 pending)
        get stepCount() {
          return flowGraph.nodes.length;
        },
        onStepChange: (fn) => player.onStepChange((index) => fn(index, null)),
      };

      const orchestratorDeps: OrchestratorDeps = {
        engine: orchEngine,
        player: orchPlayer,
        monitor,
        // 상태 맵 방출 → 캔버스 상태 점 + Timeline 마커 (트라이페인 동기 ①)
        onNodeStatus: (map) => {
          flowStatuses = map;
          flowCanvas.setStatuses(map);
          paintTimelineFromStatuses(map);
          // 실행 기록 진행 카운터 — 레코더는 [0, stepsTotal] 클램프 후 **마지막 값**을
          // 남긴다(단조 증가를 보장하지 않는다). 되감기(runFromNode)로 done이 줄어드는
          // 것은 실제 진행이 줄어든 것이므로 마지막 값이 맞는 의미다.
          runRecorder.noteStepDone(
            Object.values(map).filter((status) => status === 'done').length,
          );
        },
        // 활성 노드 방출 → 캔버스 아웃라인 강조 + 뷰포트 배지 라벨(캐시) (트라이페인 동기 ②)
        onActiveNode: (nodeId) => {
          activeFlowNodeId = nodeId;
          flowCanvas.selectNode(nodeId); // 외부 주도 선택 — onSelectNode 에코 없음(노드 폼 미개방)
          if (nodeId !== null) {
            flowEverActiveNodeIds.add(nodeId);
            const idx = flowGraph.nodes.findIndex((n) => n.id === nodeId);
            if (idx >= 0) nodeActiveStartSimSec[idx] = engine.simTimeSec; // 충돌 로그 연동 경계
          }
          // 뷰포트 오버레이(node k/n·활성 라벨)를 그래프 active dot·Timeline 커서와 같은
          // 동기 지점에서 잠근다 — 노드 경계에서 세 뷰가 한 프레임도 어긋나지 않는다 (§5).
          refreshOverlay();
        },
        resetScene,
        nodeIdByStepIndex: (index) => flowGraph.nodes[index]?.id ?? null,
        stepIndexByNodeId: (id) => {
          const i = flowGraph.nodes.findIndex((n) => n.id === id);
          return i >= 0 ? i : null;
        },
        enabledStepIndices: () =>
          flowGraph.nodes.reduce<number[]>((acc, node, i) => {
            if (node.enabled) acc.push(i);
            return acc;
          }, []),
        unexpectedCollision,
      };

      const orchestrator = new Orchestrator(orchestratorDeps);
      built.orchestrator = orchestrator;

      // 오류 이벤트 → 콘솔 (예기치 않은 충돌만 — waitForCollision 타임아웃은 player.warn이
      // 이미 콘솔에 로그하고, 아래 handlePlayerWarn이 markError로 라우팅한다)
      orchestrator.onError((e) => {
        if (e.reason !== 'collision') return;
        const pair = e.collision ? ` (${e.collision.a}×${e.collision.b})` : '';
        appLog(
          'warn',
          `예기치 않은 충돌${pair} — 노드 '${e.nodeId}' 오류 표시` +
            (orchestrator.autoPauseOnCollision ? ' + 자동 정지' : ''),
        );
        // 자동 정지가 실제로 걸렸을 때만 개입으로 기록 — 이후 ⏹은 'autoPaused' 결과가 된다
        if (orchestrator.autoPauseOnCollision) {
          runSawAutoPause = true;
          runRecorder.recordIntervention('autoPause', e.nodeId, engine.simTimeSec);
        }
      });

      /** 노드/마커에서 결정론적 재실행 (§5) — 처음부터 되감아 목표 노드 경계까지 빨리감기. */
      const runFromNodeWithToast = (nodeId: string): void => {
        const node = flowGraph.nodes.find((n) => n.id === nodeId);
        if (node === undefined) return;
        showToast(
          `'${kindMeta(node.kind).label}' 노드부터 다시 재생합니다 (처음부터 되감아 빨리감기)`,
          'info',
        );
        beginRunIfPossible();
        runRecorder.recordIntervention('runFromNode', nodeId, engine.simTimeSec);
        orchestrator.runFromNode(nodeId);
      };
      // 기록 화면 "이 노드부터 재현"이 쓰는 앱 수명 참조 (씬 전환 시 새 씬 것으로 교체)
      orchestratorRunFromNode = runFromNodeWithToast;

      // Timeline 마커 클릭 → 그 노드부터 재실행 (§5 "마커/노드 클릭 → 재실행")
      timelinePanel.onMarkerClick((index) => {
        const node = flowGraph.nodes[index];
        if (node !== undefined) runFromNodeWithToast(node.id);
      });

      // waitForCollision timeout 경고 → 해당 노드 markError (문구 계약: steps.ts가 상수로
      // 고정 — WAIT_FOR_COLLISION_WARN_TAG/TIMEOUT_MARKER. 발행측 리워딩이 이 매칭을 조용히
      // 끊지 못하도록 공유 상수로만 매칭한다. steps.test.ts가 문구를 핀한다.) markError는
      // onNodeStatus로 캔버스+타임라인에 error를 그리고 오류 이벤트를 방출한다.
      handlePlayerWarn = (msg): void => {
        if (
          !msg.includes(WAIT_FOR_COLLISION_WARN_TAG) ||
          !msg.includes(WAIT_FOR_COLLISION_TIMEOUT_MARKER)
        ) {
          return;
        }
        const node = flowGraph.nodes[player.currentStepIndex];
        if (node === undefined) return;
        orchestrator.markError(node.id);
      };

      // 초기 페인트 — 전부 pending을 캔버스/타임라인에 그린다 (부트 시 커서 통지가 아직 없다)
      orchestrator.refresh();

      // 실행 기본값 (설정 화면 — localStorage) 적용 + 씬 수명 적용기 노출 (Phase 12 ⑨).
      // 속도는 **엔진과 표시를 항상 함께** 맞춘다 — 한쪽만 바꾸면 select는 2×인데 실제
      // 재생이 1×인 조용한 no-op이 된다(CLAUDE.md §6 "조용한 no-op 금지").
      applyExecDefaultsToScene = (defaults) => {
        orchestrator.setAutoPauseOnCollision(defaults.autoPauseOnCollision);
        autoPauseCheckbox.checked = defaults.autoPauseOnCollision;
        orchestrator.setSpeed(defaults.speedMult as EngineSpeed);
        playbackBar.setSpeedDisplay(defaults.speedMult);
      };
      const bootExecDefaults = consoleRef?.execDefaults();
      if (bootExecDefaults !== undefined) applyExecDefaultsToScene(bootExecDefaults);

      // 페인 표시 정책: 시퀀스 있는 씬 = 자동 표시, 없는 씬 = 숨김('플로우' 토글로 열기)
      setFlowPaneVisible(validSequence !== null);

      /** '플로우' 토글로 페인이 열렸을 때의 컨텍스트 안내 (로봇 없는 씬) */
      const onFlowPaneShown = (): void => {
        if (flowGraph.nodes.length === 0 && sceneHandle.robots.ids().length === 0) {
          showToast(
            '이 씬에는 로봇이 없습니다 — 시퀀스를 만들려면 먼저 로봇을 추가하세요',
            'warn',
          );
        }
      };

      // ── 씬 편집 ↔ 플로우 재동기 (Phase 8 보강 — 편집/재생 잠김 방지) ──
      // flowGraph.robot과 step 참조는 빌드 시점 스냅샷이라 씬 편집(rename/로봇 추가)과
      // 어긋날 수 있다: (a) 로봇 rename 후 모든 플로우 편집이 "씬에 없는 엔티티"로
      // 거부되고, (b) 로봇 없는 씬(robot '')에 로봇을 추가해도 편집이 영구 거부된다.
      // SceneEditEvent에는 새 id만 실리므로 rename의 옛 id는 직전 통지 시점 id 목록과의
      // 차집합으로 복원한다 (rename은 단일 엔티티 연산 — scene-editor.ts 계약).
      let flowPrevEntityIds: string[] = editor.spec.entities.map((e) => e.id);

      /** 재동기 그래프를 §2.8 파이프라인(씬 참조 무결성 직렬화)으로 커밋한다.
       *  시스템 동기화이므로 '수정됨' 배지 diff는 걷지 않는다. 실패 시 그래프/시퀀스
       *  불변 + 콘솔 오류 (이미 씬과 어긋나 있던 시퀀스 — 편집 거부/arm 거부가 막는다). */
      const commitResyncedFlowGraph = (nextGraph: FlowGraph, reasonKo: string): void => {
        const serialized = serializeGraph(nextGraph, editor.spec, {
          id: flowSeqMeta.id,
          ...(flowSeqMeta.loop !== undefined ? { loop: flowSeqMeta.loop } : {}),
        });
        if (!serialized.ok) {
          appLog(
            'error',
            `${reasonKo} — 시퀀스 재직렬화 실패: ${serialized.errors.join(' / ')}`,
          );
          return;
        }
        flowGraph = nextGraph;
        lastFlowValidation = 'ok';
        commitFlowSequence(serialized.sequence);
        flowCanvas.render();
        nodeEditor.refresh();
        appLog('info', reasonKo);
      };

      /** 씬 편집 통지 → 플로우 재동기 (editor.onChange 구독에서 호출) */
      const resyncFlowWithSceneEdit = (e: SceneEditEvent): void => {
        const currentIds = editor.spec.entities.map((en) => en.id);
        const prevIds = flowPrevEntityIds;
        flowPrevEntityIds = currentIds;

        if (e.kind === 'rename') {
          const currentSet = new Set(currentIds);
          const oldId = prevIds.find((id) => !currentSet.has(id));
          if (oldId === undefined) return;
          const next = remapEntityId(flowGraph, oldId, e.entityId);
          if (next === flowGraph) return; // 그래프에 옛 id 참조 없음 — 동일 참조 계약
          if (next.nodes.length === 0) {
            flowGraph = next; // 빈 그래프 — 기본 로봇만 추종 (직렬화 대상 없음)
            return;
          }
          commitResyncedFlowGraph(
            next,
            `플로우 시퀀스의 '${oldId}' 참조를 '${e.entityId}'(으)로 갱신했습니다`,
          );
          return;
        }

        if (e.kind === 'add') {
          // 기본 로봇이 무효(빈 문자열/죽은 참조)면 로봇 추가 시 첫 로봇을 채택해
          // 편집을 되살린다 (유효한 기본 로봇은 절대 바꾸지 않는다).
          const robotEntity = editor.spec.entities.find((en) => en.id === flowGraph.robot);
          if (robotEntity !== undefined && isRobotSpec(robotEntity)) return;
          const candidate = sceneHandle.robots.ids()[0];
          if (candidate === undefined || candidate === flowGraph.robot) return;
          if (flowGraph.nodes.length === 0) {
            flowGraph = { ...flowGraph, robot: candidate };
            appLog('info', `플로우 기본 로봇: '${candidate}' (씬에 추가된 첫 로봇 채택)`);
            return;
          }
          commitResyncedFlowGraph(
            { nodes: flowGraph.nodes, edges: flowGraph.edges, robot: candidate },
            `플로우 기본 로봇을 '${candidate}'(으)로 채택했습니다`,
          );
        }
      };

      // Flow Graph 파사드 (게이트/자동화 — UI와 같은 runFlowOp 파이프라인만 사용)
      const flowFacade: SimFlowGraphFacade = {
        visible: () => flowPaneVisible,
        nodeCount: () => flowGraph.nodes.length,
        nodeIds: () => flowGraph.nodes.map((n) => n.id),
        kinds: () => flowGraph.nodes.map((n) => n.kind),
        params: (nodeId) => {
          const node = flowGraph.nodes.find((n) => n.id === nodeId);
          return node ? structuredClone(node.params) : null;
        },
        reorder: (nodeId, toIndex) => applyFlowOpWithToast((g) => moveNode(g, nodeId, toIndex)),
        insertWait: (atIndex) =>
          applyFlowOpWithToast((g) =>
            insertNode(g, defaultNodeFor('wait', flowPaletteContext()), atIndex),
          ),
        remove: (nodeId) => applyFlowOpWithToast((g) => removeNode(g, nodeId)),
        setEnabled: (nodeId, enabled) =>
          applyFlowOpWithToast((g) => setNodeEnabled(g, nodeId, enabled)),
        sequenceJson: () => JSON.stringify(currentSequence),
        lastValidation: () =>
          lastFlowValidation === 'ok' ? 'ok' : [...lastFlowValidation],
        everActiveNodeIds: () => [...flowEverActiveNodeIds],
        nodeStatuses: () => ({ ...flowStatuses }),
      };

      // 플래너 파사드 (Phase 9): 생성은 앱 수명 runGenerate를 위임 호출한다 — UI와
      // 완전히 같은 흐름(검증 → 그래프 로드 → 무자동재생). 로드 여부/player 상태는 이
      // 씬의 상태에서 읽는다. lastResult는 앱 수명(마지막 생성)이라 boot 클로저를 본다.
      const plannerFacade: SimPlannerFacade = {
        generate: (nl) => runGenerate(nl, 'replace').then((result) => ({ type: result.type })),
        lastResult: () => lastPlannerResult,
        isLoadedIntoGraph: () => hasGeneratedNodes(),
        playerStatus: () => player.status,
      };

      // 실행 오케스트레이션 파사드 (Phase 10) — 재생/노드 상태/재실행을 UI와 같은 경로로 노출.
      // 재생 컨트롤은 playbackControls(=UI 재생 바)를 위임 호출해 파사드/사람 조작이 동일한
      // orchestrator 진실을 본다 (§5 동기 강조).
      const orchestratorFacade: SimOrchestratorFacade = {
        play: () => playbackControls.play(),
        pause: () => playbackControls.pause(),
        stop: () => playbackControls.stop(),
        stepNode: () => playbackControls.stepOnce(),
        setAutoPause: (enabled) => {
          orchestrator.setAutoPauseOnCollision(enabled);
          autoPauseCheckbox.checked = enabled; // UI 토글과 동기 (게이트 가시 플래그)
        },
        autoPause: () => orchestrator.autoPauseOnCollision,
        activeNodeId: () => orchestrator.activeNodeId,
        statuses: () => ({ ...orchestrator.statuses }),
        runFromNode: (nodeId) => runFromNodeWithToast(nodeId),
        overlayText: () => overlaySummary(lastOverlayState),
      };

      // 3D 임포트 파사드 — 다이얼로그·에셋 저장소는 **앱 수명**이지만 게이트가 보는 훅은
      // __sim 하나뿐이라 씬별 핸들에 실어 위임한다(plannerFacade가 앱 수명 runGenerate를
      // 위임하는 것과 동일). importDialog는 이 함수 정의보다 뒤에 선언되지만 buildScene의
      // 첫 호출은 그보다 뒤라 TDZ가 아니다.
      const meshImportFacade: SimMeshImportFacade = {
        open: (file) => {
          pendingImportDrop = null; // 파일 선택 경로와 동일 — 뷰포트 중앙 배치
          importDialog.openWith(file);
        },
        assetRefs: () => assetStore.refs(),
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
        flowGraph: flowFacade,
        planner: plannerFacade,
        orchestrator: orchestratorFacade,
        meshImport: meshImportFacade,
        ...(playerFacade ? { player: playerFacade } : {}),
      };

      // ── 선택 동기화 (단일 경로): interaction이 진실, 나머지가 따라온다 ──
      // 클릭 픽킹·프로그램 select·픽킹 맵 재구축(setPickables의 stale 해제) 모두
      // 이 리스너를 통해 인스펙터/편집 폼에 전파된다. 각 모듈의 변경 가드가 루프를 막는다.
      interaction.onSelect((id) => {
        inspector.select(id);
        sceneOutliner.select(id);
        entityEditor.showFor(id);
        refreshSelectionHud();
        refreshOverlay(); // 선택 리드아웃('선택 <id>')을 rAF 지연 없이 즉시 반영
        if (id !== null) {
          // 우측 스택 중재 (마지막 선택 승리): 엔티티 선택 → 노드 선택 해제 + 엔티티 폼.
          // null(해제) 에코는 패널을 바꾸지 않는다 — 노드 선택이 유발한 해제와 공존.
          flowCanvas.selectNode(null);
          nodeEditor.showFor(null);
          showRightPanelFor('entity');
          const node = visualNodeOf(id);
          if (node) pulseEntity(node);
        }
        inspector.refresh();
        sceneOutliner.refresh();
      });

      // ── 편집 통지 → 파생 상태 재동기화 + 히스토리 기록 ────────────────
      built.offEditorChange = editor.onChange((e) => {
        rebuildPickables(); // add/remove/재빌드(치수·물리·개명)로 바뀐 노드 매핑 재구축
        syncJointPanel(); // 로봇 추가/삭제/개명 → 관절 슬라이더 패널 재구성
        resyncFlowWithSceneEdit(e); // rename 참조 리매핑 · 기본 로봇 채택 (플로우 잠김 방지)
        inspector.refresh();
        sceneOutliner.refresh();
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
        refreshSelectionHud(); // 위치·치수 리드아웃이 편집 결과를 즉시 반영
        // Undo 스냅샷: 연속 조정(transform/dimensions/physics) burst는 디바운스로
        // 1장에 합쳐지고, 구조 변경(add/remove/rename)은 flushPending으로 경계를 세워
        // 개별 스냅샷이 된다 — "undo 1회 = 구조 변경 1개" (ui/history.ts 계약).
        const discrete = e.kind === 'add' || e.kind === 'remove' || e.kind === 'rename';
        if (discrete) history.flushPending(); // 직전 연속 burst를 먼저 확정
        history.noteChange(() => ({
          scene: editor.serialize(),
          sequence: currentSequence,
        }));
        if (discrete) history.flushPending();
      });

      engine.start();
      engine.play(); // 물리 루프 자동 시작 — 시퀀스는 ▶ Play로만 (파일 헤더의 재생 정책)
      console.log(
        `Scene '${spec.name}' loaded — entities: [${sceneHandle.entityIds.join(', ')}], ${spec.timestepHz}Hz`,
      );

      return {
        spec,
        // 그래프 편집이 커밋될 때마다 라이브 시퀀스가 바뀐다 — getter로 항상 현재 진실
        get validSequence() {
          return currentSequence;
        },
        // 히스토리 재로드용: 편집된 라이브 시퀀스가 있으면 그것을(씬 undo가 시퀀스
        // 편집을 잃지 않게), 없으면 원본 JSON을 재검증 대상으로 넘긴다
        get sequenceJson() {
          return currentSequence ?? sequenceJson;
        },
        onFlowPaneShown,
        editor,
        engine,
        uniquifyId,
        placeEntity,
        loadGeneratedSequence,
        hasGeneratedNodes,
        playerStatus: () => player.status,
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
      if (!opts.fromHistory) {
        // 새 논리 씬 — 작업/공정 문서 컨텍스트와 표시명을 해제한다 (stale 컨텍스트로
        // Ctrl+S가 엉뚱한 작업을 덮어쓰는 사고 방지). 문서 열기(콘솔 글루의 bridge.
        // loadDocument)는 로드가 끝난 뒤 컨텍스트를 다시 세운다 — 순서가 계약이다.
        consoleRef?.clearDocumentContext();
        docLabelOverride = null;
        sceneControls.setDocumentContext(null); // 프리셋/업로드로 돌아왔다 — select가 다시 진실
        // 시퀀스 버전 이력도 여기서만 비운다 — history.reset()과 같은 조건이다.
        // Undo/Redo 복원(fromHistory)은 이력을 유지해야 [버전] 탭이 안전망으로 남는다.
        sequenceVersions.clear();
      }
      active?.dispose();
      active = null;
      active = await buildScene(validation.value, request.sequence);
      jsonViewer.refresh(); // 시퀀스 뷰어를 새 씬 진실로 갱신
      // 히스토리: 새 "논리 씬"이면 스택 리셋 + 기준 스냅샷. 히스토리 복원 재로드면
      // 스택을 건드리지 않는다 (SceneHistory가 스택 이동을 소유).
      if (!opts.fromHistory) {
        history.reset({ scene: active.editor.serialize(), sequence: active.validSequence ?? null });
      }
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
        const { scene, sequence } = parseDocument(payload);
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
      // 저장은 **씬 + 시퀀스 봉투**로 나간다 (UX_AUDIT C-3).
      // 작업/공정 문서가 열려 있으면 서버 저장이 우선한다 (Ctrl+S와 같은 라우팅)
      saveDocument: () => {
        void (async (): Promise<void> => {
          const handled = await consoleRef?.saveActive();
          if (handled !== true) saveDocumentToFile();
        })();
      },
      onShowHelp: () => {
        helpSheet.open();
      },
    },
    { helpHost: commandBar.rowAEnd },
  );
  dirtyTracker.onChange((dirty) => {
    sceneControls.setDirty(dirty);
  });
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

  const library = mountLibrary(workspace.slots.left, {
    onDragState: (dragActive, label) => {
      activeDropHint?.setActive(dragActive, label);
    },
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
    // 재사용 블록 섹션 (Phase 12 ⑤) — 콘솔 글루가 목록/삽입을 소유한다.
    // consoleRef는 부트 말미에 마운트되지만 provider는 렌더 시점에 호출되므로 안전.
    blocksProvider: () => consoleRef?.libraryBlocks() ?? Promise.resolve([]),
    onInsertBlock: (id) => {
      consoleRef?.onLibraryInsertBlock(id);
    },
  });
  // 좌 슬롯 세로 배분: 라이브러리(카드 그리드) 60 / 씬 아웃라이너 40.
  // 아웃라이너는 씬 빌드마다 재마운트되므로 배분은 슬롯 쪽에서 고정한다.
  Object.assign(library.el.style, {
    flex: '1 1 58%',
    minHeight: '0',
  } satisfies Partial<CSSStyleDeclaration>);

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

  // ── 자연어 Planner (Phase 9, UX_DESIGN §3.1/§4.1 Flow 1, PLANNER.md — 앱 수명) ──
  // 설정(backend/apiKey/model)은 localStorage 영속. 규칙 기반(오프라인)이 기본이며,
  // Anthropic 선택 시 이 세션 브라우저에서 직접 호출한다(교육/프로토타입 — PRD §6).
  // 생성 흐름은 §2.9를 매 출구에서 집행한다: (1) 플래너가 검증한 sequence를 실행 노출
  // 직전 한 번 더 validateSequence(심층 방어) → 실패면 로드하지 않음, (2) 검증 통과본만
  // 그래프에 로드하고 자동 재생하지 않음(사용자 ▶ Play), (3) clarify/error/예외는
  // 시뮬레이터로 보내지 않고 명확화 카드·토스트·콘솔로 표면화한다.

  const toasts = mountToasts(document.body);
  const clarifyCard = mountClarifyCard(document.body);

  let plannerConfig = loadPlannerConfig();
  let plannerService = buildPlannerService(plannerConfig);

  /** 중복 생성 가드 + nl-input busy 표시 */
  let generating = false;
  /** nl-input 핸들 (mount 후 할당 — runGenerate가 상태를 표시) */
  let nlInput: NlInputHandle | null = null;
  /** planner 파사드 lastResult()용 — 마지막 생성 결과 요약 (앱 수명, 마지막 생성) */
  let lastPlannerResult:
    | {
        type: string;
        stepCount?: number;
        assumptions?: string[];
        question?: string;
        options?: string[];
      }
    | null = null;

  /**
   * 현재 활성 씬의 실시간 상태 스냅샷 (관절값 + 물체 월드 위치) — 그라운딩 정확도를
   * 높인다(밀려 이동한 물체는 현재 위치로). 진실은 물리(world) — public 파사드로만 읽는다.
   */
  const buildLiveSnapshot = (): WorldSnapshot => {
    const sim = window.__sim;
    if (!sim) return {};
    const jointValuesByRobot: Record<string, Record<string, number>> = {};
    for (const robotId of sim.robots.ids()) {
      jointValuesByRobot[robotId] = sim.robots.readJoints(robotId);
    }
    const positionsByEntity: Record<string, [number, number, number]> = {};
    for (const id of sim.editor.entityIds()) {
      const bodies = sim.world.bodiesOfEntity(id);
      if (bodies.length > 0) {
        const p = sim.world.getPose(bodies[0]!).position;
        positionsByEntity[id] = [p[0], p[1], p[2]];
      }
    }
    return { jointValuesByRobot, positionsByEntity };
  };

  /** 플래너 결과 3분기 처리 (§2.9 집행 — 검증 통과본만 그래프 로드, 무자동재생) */
  const handlePlannerResult = (
    result: PlannerResult,
    nl: string,
    mode: GenerateMode,
    scene: ActiveScene,
  ): void => {
    if (result.type === 'sequence') {
      // 심층 방어(§2.9): 플래너가 이미 검증했지만 실행 노출 직전 현재 씬에 한 번 더 검증
      const validation = validateSequence(result.sequence, scene.editor.spec);
      if (!validation.ok) {
        const detail = validation.errors.join('\n');
        appLog('error', `생성 시퀀스 재검증 실패 — 로드하지 않습니다:\n${detail}`);
        toasts.show('error', '생성된 시퀀스가 검증을 통과하지 못했습니다', { detail });
        nlInput?.setState('error', detail);
        lastPlannerResult = { type: 'error' };
        return;
      }
      const loaded = scene.loadGeneratedSequence(validation.value, mode);
      if (!loaded.ok) {
        const detail = (loaded.errors ?? ['알 수 없는 오류']).join('\n');
        appLog('error', `생성 시퀀스 그래프 로드 실패:\n${detail}`);
        toasts.show('error', '생성된 시퀀스를 그래프에 로드하지 못했습니다', { detail });
        nlInput?.setState('error', detail);
        lastPlannerResult = { type: 'error' };
        return;
      }
      const assumptions = result.assumptions ?? [];
      if (assumptions.length > 0) {
        toasts.show('info', `가정: ${assumptions.join(' · ')}`);
        appLog('info', `플래너 가정: ${assumptions.join(' / ')}`);
      }
      const modeKo = mode === 'append' ? '이어서' : '교체';
      appLog(
        'info',
        `플래너 생성 완료 (${validation.value.steps.length}개 step · ${modeKo}) — ▶ Play로 재생`,
      );
      lastPlannerResult = {
        type: 'sequence',
        stepCount: validation.value.steps.length,
        ...(assumptions.length > 0 ? { assumptions } : {}),
      };
      nlInput?.setState('success');
      return;
    }

    if (result.type === 'clarify') {
      lastPlannerResult = {
        type: 'clarify',
        question: result.question,
        ...(result.options ? { options: result.options } : {}),
      };
      nlInput?.setState('clarify');
      clarifyCard.show(result.question, result.options, (choice) => {
        if (choice === null) {
          nlInput?.setState('idle');
          return;
        }
        // P1 규약: 선택을 원문에 되붙여 재생성 ("... [선택: box_b]")
        void runGenerate(`${nl} [선택: ${choice}]`, mode);
      });
      return;
    }

    // error
    lastPlannerResult = { type: 'error' };
    appLog('error', `플래너 오류: ${result.message}`);
    toasts.show('error', result.message);
    nlInput?.setState('error', result.message);
  };

  /**
   * 자연어 → 생성. buildContext(현재 씬 + 라이브 스냅샷) → plannerService.generate →
   * handlePlannerResult. 어댑터/네트워크 예외는 한국어 토스트로 표면화한다. 결과를
   * 돌려주어 파사드(게이트)가 type을 검사할 수 있게 한다.
   */
  const runGenerate = async (nl: string, mode: GenerateMode): Promise<PlannerResult> => {
    clarifyCard.hide(); // 새 생성은 대기 중인 명확화 카드를 조용히 대체
    const scene = active;
    if (!scene) {
      const message = '활성 씬이 없어 생성할 수 없습니다.';
      lastPlannerResult = { type: 'error' };
      toasts.show('error', message);
      nlInput?.setState('error', message);
      return { type: 'error', message };
    }
    if (generating) {
      return { type: 'error', message: '생성이 이미 진행 중입니다.' };
    }
    generating = true;
    nlInput?.setState('generating');
    try {
      const ctx = buildContext(scene.editor.spec, buildLiveSnapshot());
      const result = await plannerService.generate(nl, ctx, scene.editor.spec);
      // 비동기 생성 중 사용자가 씬을 전환(loadScene)했다면 캡처한 scene은 이미
      // dispose됐다. 폐기된 씬에 결과를 적용하지 않는다(§2.9와 무관한 견고성 방어).
      if (active !== scene) {
        appLog('warn', '생성 결과 폐기: 생성 도중 씬이 전환되었습니다.');
        return { type: 'error', message: '생성 도중 씬이 전환되어 결과를 취소했습니다.' };
      }
      handlePlannerResult(result, nl, mode, scene);
      return result;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      appLog('error', `플래너 예외: ${message}`);
      toasts.show('error', '생성 중 오류가 발생했습니다', { detail: message });
      nlInput?.setState('error', message);
      lastPlannerResult = { type: 'error' };
      return { type: 'error', message };
    } finally {
      generating = false;
    }
  };

  // nl-input (커맨드바 중앙-좌 — 재생 컨트롤 앞). 생성 요청만 발행하고 실행·검증·그래프
  // 로드는 runGenerate가 담당한다 (nl-input은 core/planner를 모른다 — CLAUDE.md §3).
  nlInput = mountNlInput(commandBar.rowBCommand, {
    generate: (nl, mode) => runGenerate(nl, mode).then(() => undefined),
    isBusy: () => generating,
  });

  // ⚙ 플래너 설정 (커맨드바 우측) — 저장 시 localStorage 영속 + 서비스 재구성
  const plannerSettingsHandle = mountPlannerSettings(commandBar.right, {
    get: () => plannerConfig,
    set: (cfg) => {
      plannerConfig = cfg;
      savePlannerConfig(cfg);
      plannerService = buildPlannerService(cfg);
      appLog('info', `플래너 백엔드 설정: ${plannerService.backendName}`);
      toasts.show('success', `플래너 설정 저장됨 — ${plannerService.backendName}`);
    },
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

  // 부트 직후를 "저장됨" 기준선으로 삼는다 — 이후의 편집만 미저장으로 센다
  {
    // 명시 타입: `active`는 클로저(loadScene) 안에서만 대입되므로 여기서 CFA가 null로
    // 좁혀 있다 — 선언 타입으로 되돌려 읽는다.
    const scene = getActiveScene();
    if (scene !== null) {
      dirtyTracker.markSaved(scene.editor.serialize(), scene.validSequence ?? null);
    }
    setDocumentTitle(scene?.spec.name ?? null, false);
  }
  dismissBoot();

  // ── 이전 작업 복원 (UX_AUDIT C-3) ────────────────────────────────
  //
  // 브라우저 완결형이라는 장점이 정확히 반대로 작동하던 지점이다 — 데스크톱 앱은
  // 크래시해도 복구 파일이 있지만, 이 앱은 탭 실수 하나로 0이 됐다.
  void (async (): Promise<void> => {
    const draft = await documentStore.loadDraft().catch(() => null);
    if (draft === null) return;
    const banner = mountRestoreBanner(document.body, {
      ageText: describeAge(draft.updatedAtIso, Date.now()),
      originLabel: draft.originLabel,
      onRestore: () => {
        void (async (): Promise<void> => {
          const result = await loadScene({
            scene: draft.doc.scene,
            sequence: draft.doc.sequence,
          });
          if (result.ok) {
            updateUrlSceneParam(null);
            appLog('info', `이전 작업 복원: '${draft.doc.name}'`);
            showToast(`'${draft.doc.name}' 복원됨`, 'info');
          } else {
            appLog('error', `복원 실패: ${result.errors.join(' / ')}`);
            showToast('복원에 실패했습니다 — 콘솔 탭을 확인하세요', 'warn');
          }
        })();
      },
      onDismiss: () => {
        void documentStore.clearDraft();
      },
    });
    void banner;
  })();

  // ── 콘솔 평면 (Phase 12 — 2평면 IA: 공정·작업·블록·장비·기록·설정) ──
  //
  // 스튜디오 위 전면 레이어로 마운트된다. 서버가 없으면 ApiClient가 로컬 모드로
  // 판정하고 스튜디오가 첫 화면이 된다(기존 정적 배포/게이트 경로 무변경 — BACKEND §1).
  // 스튜디오와의 결합은 StudioBridge 좁은 표면뿐이다 — 콘솔은 core/render를 모른다.
  consoleRef = mountConsolePlane({
    bridge: {
      loadDocument: async ({ scene, sequence, label }) => {
        // sequence는 "없음"이 null로 온다(TaskDoc.sequence). buildScene의 관문은
        // `!== undefined`라서 null이 그대로 validateSequence로 들어가면 **새 작업을 만들
        // 때마다** "시퀀스 검증 실패" 오류 토스트가 뜬다 — 경계에서 undefined로 정규화한다.
        const result = await loadScene({ scene, sequence: sequence ?? undefined });
        if (!result.ok) return { ok: false, errors: result.errors };
        updateUrlSceneParam(null); // 문서는 딥링크 대상이 아니다 (?scene=은 프리셋 전용)
        sceneControls.setCurrent(null);
        // 상단이 "지금 무엇을 편집 중인지"와 "저장이 어디로 가는지"를 말한다
        sceneControls.setDocumentContext(label);
        docLabelOverride = label;
        const scRef = getActiveScene();
        if (scRef !== null) {
          dirtyTracker.markSaved(scRef.editor.serialize(), scRef.validSequence ?? null);
        }
        setDocumentTitle(label, false);
        appLog('info', `문서 로드: '${label}'`);
        return { ok: true, errors: [] };
      },
      serializeScene: () => getActiveScene()?.editor.serialize() ?? null,
      currentSequence: () => getActiveScene()?.validSequence ?? null,
      currentRobotIds: () => {
        const scRef = getActiveScene();
        if (scRef === null) return [];
        return scRef.editor.spec.entities.filter(isRobotSpec).map((e) => e.id);
      },
      insertSteps: (steps) => {
        const scRef = getActiveScene();
        if (scRef === null) return { ok: false, errors: ['활성 씬이 없습니다'] };
        const spec = scRef.editor.spec;
        const robots = spec.entities.filter(isRobotSpec).map((e) => e.id);
        const baseRobot = scRef.validSequence?.robot ?? robots[0];
        if (baseRobot === undefined) {
          return {
            ok: false,
            errors: ['이 씬에는 로봇이 없습니다 — 블록을 삽입하려면 로봇을 먼저 추가하세요'],
          };
        }
        const candidate: ControlSequence = {
          id: 'block-insert',
          robot: baseRobot,
          steps: [...steps],
        };
        const validated = validateSequence(candidate, spec);
        if (!validated.ok) return { ok: false, errors: validated.errors };
        const applied = scRef.loadGeneratedSequence(validated.value, 'append');
        if (!applied.ok) return { ok: false, errors: applied.errors ?? ['삽입 실패'] };
        return { ok: true, errors: [] };
      },
      runFromNode: (nodeId) => {
        orchestratorRunFromNode?.(nodeId);
      },
      sampleDocument: () => {
        const entry = SCENE_REGISTRY[DEFAULT_SCENE_NAME];
        return { scene: entry?.scene ?? null, sequence: entry?.sequence ?? null };
      },
      emptySceneSpec: (name) => ({
        name,
        version: 1,
        gravity: [0, -9.81, 0],
        timestepHz: 240,
        environment: { ground: true, skyColor: '#1b1e23' },
        camera: { position: [2.2, 1.8, 2.2], target: [0, 0.3, 0] },
        entities: [],
      }),
      openPlannerSettings: () => {
        plannerSettingsHandle.open();
      },
      plannerSummary: () => plannerService.backendName,
      applyExecDefaults: (defaults) => {
        applyExecDefaultsToScene?.(defaults);
      },
      exportCurrentDocument: () => {
        saveDocumentToFile();
      },
      resetCamera: () => {
        resetCameraView?.();
      },
      reloadLibraryBlocks: () => {
        library.reloadBlocks();
      },
      setStudioInset: (px) => {
        workspace.el.style.left = px > 0 ? `${px}px` : '0px';
        workspace.notifyResize();
      },
      markSavedBaseline: (snapshot) => {
        // 전송한 스냅샷이 있으면 그것이 기준선이다 — 저장 왕복(수백 ms) 중의 편집은
        // 전송되지 않았으므로 dirty로 남아야 한다(그러지 않으면 조용히 유실된다).
        if (snapshot !== undefined) {
          dirtyTracker.markSaved(snapshot.scene, snapshot.sequence);
          return;
        }
        const scRef = getActiveScene();
        if (scRef !== null) {
          dirtyTracker.markSaved(scRef.editor.serialize(), scRef.validSequence ?? null);
        }
      },
    },
    shortcuts: {
      setEnabled: (enabled) => {
        shortcuts.setEnabled(enabled);
      },
    },
    appLog: (level, message) => {
      appLog(level, message);
    },
  });
  // 실행 기본값(설정 화면 localStorage)을 부트 씬에도 적용한다.
  // (applyExecDefaultsToScene은 buildScene 클로저 안에서만 대입되어 CFA가 null로 좁힌다 —
  //  getActiveScene과 같은 이유로 함수 간접 참조로 선언 타입으로 되돌려 읽는다.)
  const getApplyExecDefaults = (): ((defaults: ExecDefaults) => void) | null =>
    applyExecDefaultsToScene;
  getApplyExecDefaults()?.(consoleRef.execDefaults());
}

boot().catch((err: unknown) => {
  console.error('Bootstrap failed:', err);
  const msg = err instanceof Error ? (err.stack ?? err.message) : String(err);
  showErrorOverlay('부트스트랩 실패', [msg]);
});
