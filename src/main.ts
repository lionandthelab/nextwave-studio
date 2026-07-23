// main.ts — 부트스트랩 진입점 + 씬 라이프사이클 (Phase 1–6: 데이터 주도 씬 + URDF 로봇
// + 충돌/시퀀스 UI + 런타임 씬 전환)
//
// 순서 고정 (docs/ARCHITECTURE.md §4, CLAUDE.md §2.7):
//   1. await initPhysics()      — WASM 로드 완료 전 물리 API 호출 금지
//   2. SceneSpec JSON 검증      — 실패 시 사람이 읽을 수 있는 오류 오버레이 (DATA_MODEL §8)
//   3. RapierWorld 생성         — spec.gravity / spec.timestepHz
//   4. Renderer 생성/재사용     — spec.camera / spec.environment 반영
//   5. await SceneLoader.build  — 바디 + 메시 + URDF 로봇 생성, sync/robot 바인딩
//   6. 시퀀스 검증(있으면)      — 검증 실패 시 실행에 노출하지 않는다 (불변식 §2.9)
//   7. Engine 루프 시작 — preStep: player.step → robots.tickAll (ARCHITECTURE §5 ①),
//      onContacts: CollisionMonitor.dispatch (③)
//
// main은 조립 글루다: core(엔진·월드·로더·player·monitor)와 render(three)·
// ui(커맨드바/독/우측 패널 스택[관절 패널+인스펙터])를 여기서 잇는다.
// scene-loader가 요구하는 RenderSceneApi의 three 구현(loadRobot 포함)도
// 여기서 제공한다 (core는 three를 모른 채 이 좁은 인터페이스만 호출한다 — CLAUDE.md §3).
// 부트 씬 선택은 ?scene= 쿼리 파라미터로 한다 — 씬도 시퀀스도 데이터다 (§2.5/§2.6).
//
// ── 씬 라이프사이클 (Phase 6 "씬·시퀀스 업로드 로더, 프리셋 선택 UI") ─
// 위 3–7단계는 buildScene()으로 추출되어 페이지 리로드 없이 반복 실행된다. 씬 전환
// (loadScene)은 항상 [검증 → 이전 씬 완전 해제(teardown) → 새로 빌드]의 전체 클린
// 빌드다 — 월드/sync/monitor/player/엔진/독 패널을 씬마다 새로 만들어 어떤 상태도
// 씬을 가로질러 새지 않는다(결정론, §2.3). 렌더러(three 캔버스)와 상단 커맨드바 셸·
// JSON 뷰어만 앱 수명으로 유지되고, 씬별 카메라/환경 옵션은 Renderer.applySceneOptions
// 로 재적용된다. 전환 후 window.__sim은 항상 "새" 엔진/월드/핸들을 가리킨다(게이트 계약).
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
import type { RenderSceneApi, SceneHandle, VisualNode } from './core/scene-loader';
import type { JointInfo } from './core/robot-types';
import { RenderSync } from './core/sync';
import type { PhysicsWorld, Pose } from './core/types';
import { initPhysics, RapierWorld } from './core/world';
import { pulseEntity } from './render/highlight';
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
import { mountInspector } from './ui/inspector/inspector';
import type { InspectorHandle } from './ui/inspector/inspector';
import { mountJointPanel } from './ui/inspector/joint-panel';
import { mountViewportStatus } from './ui/viewport/statusline';
import { LAYOUT, Z_INDEX } from './ui/theme';
import { isRobotSpec, validateScene, validateSequence } from './schema';
import type { CollisionEvent, ControlSequence, SceneSpec } from './schema';
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

// ── 우측 패널 스택 상수 (관절 패널 + 인스펙터 — UX_DESIGN §2 우측 존) ──
// 층/폭/z-index는 ui/theme 토큰에서 유도한다 — 테마가 바뀌어도 바/독과 어긋나지 않는다.

/** 커맨드바 아래 오프셋 — scene-controls의 오류 토스트와 동일 층 기준 */
const RIGHT_STACK_TOP_PX = LAYOUT.belowBarTopPx;
const RIGHT_STACK_RIGHT_PX = 12;
const RIGHT_STACK_WIDTH_PX = LAYOUT.rightPanelWidthPx;
/** 스택 내 패널 사이 간격 */
const RIGHT_STACK_GAP_PX = 8;
/** 하단 독(본문 180px + 탭바 + 여백)을 침범하지 않는 여유 */
const RIGHT_STACK_BOTTOM_CLEARANCE_PX = 230;
/** 독/커맨드바보다 위, {} JSON 슬라이드 패널보다 아래 — 뷰어가 열리면 스택을 덮는다 */
const RIGHT_STACK_Z_INDEX = Z_INDEX.rightStack;
/** 인스펙터 값 갱신 스로틀 주기 (playing 중 — inspector.ts 헤더의 "주기 결정권은 통합자") */
const INSPECTOR_REFRESH_INTERVAL_MS = 150;

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

/** window.__sim으로 노출되는 시뮬 핸들. Rapier 타입은 새지 않는다(PhysicsWorld 경계). */
export interface SimHandle {
  readonly engine: Engine;
  readonly world: PhysicsWorld;
  readonly sceneHandle: SceneHandle;
  readonly spec: SceneSpec;
  readonly robots: SimRobotsFacade;
  readonly collision: SimCollisionFacade;
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

// ── 씬 라이프사이클 타입 ────────────────────────────────────────────

/** 씬 전환 요청 — 프리셋(레지스트리) 또는 업로드 JSON */
interface SceneLoadRequest {
  readonly scene: unknown;
  readonly sequence?: unknown;
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

  // ── 앱 수명 상태 (씬 전환을 가로질러 유지) ────────────────────────

  let renderer: Renderer | null = null;
  let active: ActiveScene | null = null;
  let switching = false;

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

  // ── 상단 커맨드바 셸 [좌: 씬 컨트롤 | 중앙: 재생 | 우: {} JSON] ───
  // 씬 전환을 가로질러 유지된다 (UX_DESIGN §3.1 — 하나의 응집된 커맨드바).

  const commandBar = mountCommandBarShell(document.body);
  const jsonViewer = mountJsonViewer(
    commandBar.right,
    document.body,
    () => active?.validSequence ?? null,
  );

  // ── 씬 1개 빌드 (검증된 spec → 월드/씬/엔진/UI — 파일 헤더의 3–7단계) ──

  async function buildScene(
    spec: SceneSpec,
    sequenceJson: unknown,
  ): Promise<ActiveScene> {
    const render = ensureRenderer(spec);
    const world = new RapierWorld(spec.gravity, spec.timestepHz);
    const sync = new RenderSync(world);

    // 로봇 시각 노드 수집 버퍼 — RobotHandle은 three 노드를 노출하지 않으므로(경계 계약)
    // loadRobot 호출 전후의 씬 자식 차이로 캡처한다. SceneLoader.build는 엔티티를
    // 순서대로 await하므로 캡처 순서 = RobotRegistry 등록 순서(robots.ids())와 같다.
    const robotVisualNodesInLoadOrder: VisualNode[] = [];

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
        // 씬 전환마다 GPU 버퍼가 GC 대기 상태로 쌓이지 않게 즉시 해제 —
        // RobotHandle.dispose(render/urdf.ts)가 URDF 메시에 하는 것과 대칭
        disposeMeshResources(node);
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
        if (added) robotVisualNodesInLoadOrder.push(added);
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

    // 로봇 엔티티 id → 시각 노드 (등록 순서 = 로드 순서 — 위 캡처 버퍼 주석 참조)
    const robotVisualNodeById = new Map<string, VisualNode>();
    sceneHandle.robots.ids().forEach((robotId, index) => {
      const node = robotVisualNodesInLoadOrder[index];
      if (node) robotVisualNodeById.set(robotId, node);
    });
    const visualNodeOf = (entityId: string): VisualNode | undefined =>
      robotVisualNodeById.get(entityId) ?? sceneHandle.visualNodes.get(entityId);

    // ── 빌드 이후 조립 가드 ──────────────────────────────────────────
    // SceneLoader.build 이후의 조립(모니터/플레이어/엔진/독/우측 스택)이 중간에 던지면
    // 이미 만들어진 몫만 teardown 순서 계약(EXPERIMENTS 2026-07-23: halt → 구독 해제 →
    // UI 제거 → 씬 자원 → sync → world)대로 되감고 재던진다 — 실패해도 월드/DOM 패널/
    // 전역 키 리스너가 새지 않는다. 성공 시 반환되는 dispose()도 같은 함수를 쓴다
    // (해제 경로는 하나다).
    const built: {
      engine?: Engine;
      timelinePanel?: ReturnType<typeof createTimelinePanel>;
      collisionPanel?: ReturnType<typeof createCollisionLogPanel>;
      consolePanel?: ReturnType<typeof createConsolePanel>;
      dock?: ReturnType<typeof mountDock>;
      offMonitor?: () => void;
      playbackBar?: ReturnType<typeof mountPlaybackBar>;
      viewportStatus?: ReturnType<typeof mountViewportStatus>;
      offStepChange?: () => void;
      offTick?: () => void;
      rightStack?: HTMLDivElement;
      jointPanel?: ReturnType<typeof mountJointPanel>;
      inspector?: InspectorHandle;
    } = {};

    // 이 씬 몫의 전부를 해제한다 — 다음 빌드에 어떤 상태도 새지 않는다(전체 클린 빌드).
    // 순서: 엔진 루프 완전 정지 → 구독 해제 → UI 제거 → 씬 자원(물리 바디·시각 노드·
    // 로봇 핸들) 해제 → sync 바인딩 정리 → 월드 free. 렌더러/캔버스는 유지된다.
    const teardownBuilt = (): void => {
      built.engine?.halt();
      built.offTick?.();
      built.offStepChange?.();
      built.offMonitor?.();
      built.playbackBar?.dispose();
      built.viewportStatus?.dispose();
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
        const entity = spec.entities.find((e) => e.id === robotId);
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
          console.error('Sequence validation failed:', sequenceValidation.errors);
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

      // ── UI: 하단 독 (Timeline | Collision Log | Console) + 재생 컨트롤 ──

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
      built.dock = mountDock(document.body, [
        { label: 'Timeline', content: timelinePanel.el },
        { label: 'Collision Log', content: collisionPanel.el },
        { label: 'Console', content: consolePanel.el },
      ]);

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

      // 뷰포트 좌하단 실행 오버레이 (UX_DESIGN §3.3): 씬 이름 · 상태 · simTime · step —
      // 표시 전용(pointer-events 없음), 씬 수명과 함께 마운트/해제된다
      const viewportStatus = mountViewportStatus(document.body, {
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

      // rAF당 1회: 재생 바 + 타임라인 리드아웃 갱신 (물리 tick과 분리된 뷰 갱신)
      built.offTick = engine.onTick((info) => {
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

      // 씬 전환 후 게이트/자동화가 보는 핸들은 항상 "이" 씬의 새 인스턴스들이다
      window.__sim = {
        engine,
        world,
        sceneHandle,
        spec,
        robots,
        collision: collisionFacade,
        ...(playerFacade ? { player: playerFacade } : {}),
      };

      // ── 우측 패널 스택: 관절 패널(위) + 인스펙터(아래) (UX_DESIGN §2 우측 존) ──
      // 두 패널 모두 자체 절대 배치 기본값을 갖지만, 여기서는 하나의 스택 컨테이너에
      // 편입해 세로로 쌓는다 — 위 패널을 접으면 아래 패널이 자연히 올라오고, 컨테이너
      // maxHeight가 하단 독과의 겹침을 막는다. 재배치는 각 핸들의 el로 한다
      // (inspector.ts 헤더의 "통합자 재배치" 규약 — 두 모듈 모두 el을 노출한다).
      const rightStack = document.createElement('div');
      Object.assign(rightStack.style, {
        position: 'fixed',
        top: `${RIGHT_STACK_TOP_PX}px`,
        right: `${RIGHT_STACK_RIGHT_PX}px`,
        zIndex: RIGHT_STACK_Z_INDEX,
        width: `${RIGHT_STACK_WIDTH_PX}px`,
        maxHeight: `calc(100vh - ${RIGHT_STACK_TOP_PX + RIGHT_STACK_BOTTOM_CLEARANCE_PX}px)`,
        display: 'flex',
        flexDirection: 'column',
        gap: `${RIGHT_STACK_GAP_PX}px`,
      } satisfies Partial<CSSStyleDeclaration>);
      document.body.appendChild(rightStack);
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

      // 로봇이 있는 씬이면 임시 관절 패널 마운트 (ROADMAP Phase 3 "슬라이더 수동 제어",
      // Phase 6 인스펙터가 "표시"를 대체하기 전까지 "쓰기(슬라이더)" 경로로 공존)
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

      // 인스펙터 (ROADMAP Phase 6 "엔티티 목록·선택·트랜스폼/관절 상태 표시") —
      // 읽기 전용 deps를 core 파사드 위에서 구현해 주입한다 (ui는 core를 모른다, §3)
      const inspector = mountInspector(rightStack, {
        // 목록은 spec.entities 기준 — environment.ground가 만드는 예약 엔티티
        // GROUND_ENTITY_ID('__ground')는 스펙에 없는 내부 id이므로 자연히 제외된다
        listEntities: () => spec.entities.map((e) => ({ id: e.id, type: e.type })),
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
        // 선택 변경 → 해당 시각 노드 붉은 펄스(render/highlight) + 상세 1회 재독.
        // inspector의 변경 가드 덕에 여기서 refresh를 되불러도 루프가 생기지 않는다.
        onSelect: (id) => {
          if (id !== null) {
            const node = visualNodeOf(id);
            if (node) pulseEntity(node);
          }
          inspectorRef?.refresh();
        },
      });
      built.inspector = inspector;
      adoptIntoStack(inspector.el);
      inspectorRef = inspector;

      engine.start();
      engine.play(); // 물리 루프 자동 시작 — 시퀀스는 ▶ Play로만 (파일 헤더의 재생 정책)
      console.log(
        `Scene '${spec.name}' loaded — entities: [${sceneHandle.entityIds.join(', ')}], ${spec.timestepHz}Hz`,
      );

      return {
        spec,
        validSequence,
        dispose: teardownBuilt,
      };
    } catch (err) {
      // 조립 도중 실패 — 이미 만들어진 몫만 되감고 재던진다 (loadScene이 표면화)
      teardownBuilt();
      throw err;
    }
  }

  // ── 씬 전환 (항상 전체 클린 빌드 — 파일 헤더의 씬 라이프사이클) ───

  async function loadScene(request: SceneLoadRequest): Promise<SceneLoadResult> {
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

  // ── 커맨드바 좌측: 타이틀 · 씬 프리셋 · 📂 업로드 · 💾 저장 (UX_DESIGN §3.1) ──

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
      currentSpec: () => active?.spec ?? null,
    },
  );

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
}

boot().catch((err: unknown) => {
  console.error('Bootstrap failed:', err);
  const msg = err instanceof Error ? (err.stack ?? err.message) : String(err);
  showErrorOverlay('부트스트랩 실패', [msg]);
});
