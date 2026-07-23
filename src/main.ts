// main.ts — 부트스트랩 진입점 (Phase 1–5: 데이터 주도 씬 + URDF 로봇 + 충돌/시퀀스 UI)
//
// 순서 고정 (docs/ARCHITECTURE.md §4, CLAUDE.md §2.7):
//   1. await initPhysics()      — WASM 로드 완료 전 물리 API 호출 금지
//   2. SceneSpec JSON 검증      — 실패 시 사람이 읽을 수 있는 오류 오버레이 (DATA_MODEL §8)
//   3. RapierWorld 생성         — spec.gravity / spec.timestepHz
//   4. Renderer 생성            — spec.camera / spec.environment 반영
//   5. await SceneLoader.build  — 바디 + 메시 + URDF 로봇 생성, sync/robot 바인딩
//   6. 시퀀스 검증(있으면)      — 검증 실패 시 실행에 노출하지 않는다 (불변식 §2.9)
//   7. Engine 루프 시작 — preStep: player.step → robots.tickAll (ARCHITECTURE §5 ①),
//      onContacts: CollisionMonitor.dispatch (③)
//
// main은 조립 글루다: core(엔진·월드·로더·player·monitor)와 render(three)·ui(독/재생바)를
// 여기서 잇는다. scene-loader가 요구하는 RenderSceneApi의 three 구현(loadRobot 포함)도
// 여기서 제공한다 (core는 three를 모른 채 이 좁은 인터페이스만 호출한다 — CLAUDE.md §3).
// 씬 선택은 ?scene= 쿼리 파라미터로 한다 — 씬도 시퀀스도 데이터다 (CLAUDE.md §2.5/§2.6).
//
// ── 시퀀스 재생 정책 (human-in-the-loop) ─────────────────────────────
// 씬 레지스트리에 시퀀스가 선언된 씬이라도 시퀀스를 자동 재생하지 않는다 — 검증을
// 통과한 시퀀스는 "로드 가능" 상태로만 두고, 사용자가 ▶ Play를 눌러야 player에
// 로드/시작된다(불변식 §2.9의 원칙을 플래너 이전 단계부터 적용). 물리 루프 자체는
// 부팅 직후 시작한다(낙하 등 씬 자체 물리는 재생 컨트롤과 무관하게 관찰 가능해야 함
// — 기존 falling-boxes 게이트 계약 유지).

import { CollisionMonitor } from './core/collision';
import {
  collisionQueryFromMonitor,
  robotApiFromRegistry,
} from './core/control/adapters';
import { ControlPlayer } from './core/control/player';
import type { PlayerStatus } from './core/control/player';
import { Engine, ENGINE_SPEED_OPTIONS } from './core/engine';
import type { EngineSpeed } from './core/engine';
import { GROUND_ENTITY_ID, SceneLoader } from './core/scene-loader';
import type { RenderSceneApi, SceneHandle, VisualNode } from './core/scene-loader';
import type { JointInfo } from './core/robot-types';
import { RenderSync } from './core/sync';
import type { PhysicsWorld, Pose } from './core/types';
import { initPhysics, RapierWorld } from './core/world';
import { pulseEntity } from './render/highlight';
import { groundMesh, primitiveMesh } from './render/meshes';
import { Renderer } from './render/renderer';
import { loadUrdfRobot } from './render/urdf';
import { mountPlaybackBar } from './ui/command-bar/playback';
import { createCollisionLogPanel } from './ui/dock/collision-log';
import { appLog, createConsolePanel } from './ui/dock/console-panel';
import { mountDock } from './ui/dock/dock';
import { createTimelinePanel } from './ui/dock/timeline';
import { mountJointPanel } from './ui/inspector/joint-panel';
import { isRobotSpec, validateScene, validateSequence } from './schema';
import type { CollisionEvent, ControlSequence, SceneSpec } from './schema';
import fallingBoxesSceneJson from './assets/scenes/falling-boxes.scene.json';
import armAndBoxesSceneJson from './assets/scenes/arm-and-boxes.scene.json';
import armTouchBoxSequenceJson from './assets/sequences/arm-touch-box.sequence.json';

// ── 씬 레지스트리 (?scene= 파라미터로 선택 — 새 씬 = 새 데이터) ──────
// 시퀀스가 선언된 씬은 검증 후 ▶ Play로만 재생된다(파일 헤더의 재생 정책).

interface SceneRegistryEntry {
  readonly scene: unknown;
  /** 이 씬에서 재생할 ControlSequence JSON (선택) */
  readonly sequence?: unknown;
}

const SCENE_REGISTRY: Readonly<Record<string, SceneRegistryEntry>> = {
  'falling-boxes': { scene: fallingBoxesSceneJson },
  'arm-and-boxes': { scene: armAndBoxesSceneJson, sequence: armTouchBoxSequenceJson },
};

const DEFAULT_SCENE_NAME = 'arm-and-boxes';

/** 충돌 로그에서 최근 이벤트를 조회할 때의 기본 상한 (파사드 recent의 인자와 무관) */
const COLLISION_RECENT_DEFAULT_LIMIT = 50;

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

/** 시퀀스 재생 파사드 (씬 레지스트리에 시퀀스가 선언된 씬에서만 노출) */
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
    /** 자동화 훅 (scripts/gate-browser.mjs가 검증에 사용) */
    __sim?: SimHandle;
  }
}

// ── 오류 오버레이 (검증 실패·부트스트랩 실패 표시용, 한국어) ─────────

const OVERLAY_Z_INDEX = '9999';

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

// ── 부트스트랩 ──────────────────────────────────────────────────────

async function boot(): Promise<void> {
  const host = document.getElementById('app');
  if (!host) throw new Error('#app host element not found');

  // 씬 선택: ?scene=<이름> (미지정 시 기본 씬). 알 수 없는 이름은 명확한 오류로 중단.
  const sceneName =
    new URLSearchParams(window.location.search).get('scene') ?? DEFAULT_SCENE_NAME;
  const registryEntry = SCENE_REGISTRY[sceneName];
  if (registryEntry === undefined) {
    showErrorOverlay('알 수 없는 씬', [
      `'${sceneName}' 씬을 찾을 수 없습니다.`,
      `사용 가능한 씬: ${Object.keys(SCENE_REGISTRY).join(', ')}`,
      `예: ?scene=${DEFAULT_SCENE_NAME}`,
    ]);
    return;
  }

  await initPhysics();
  console.log('Rapier ready');

  // 씬은 데이터다 — 코드가 아니라 JSON이 씬을 정의한다 (CLAUDE.md §2.5)
  const validation = validateScene(registryEntry.scene);
  if (!validation.ok) {
    console.error('Scene validation failed:', validation.errors);
    showErrorOverlay(`씬 검증 실패 — ${sceneName}.scene.json`, validation.errors);
    return;
  }
  const spec = validation.value;

  const world = new RapierWorld(spec.gravity, spec.timestepHz);
  const render = new Renderer(host, {
    skyColor: spec.environment?.skyColor,
    cameraPosition: spec.camera?.position,
    cameraTarget: spec.camera?.target,
    cameraFov: spec.camera?.fov,
  });
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

  const sceneHandle = await new SceneLoader(world, renderApi, sync).build(spec);

  // 로봇 엔티티 id → 시각 노드 (등록 순서 = 로드 순서 — 위 캡처 버퍼 주석 참조)
  const robotVisualNodeById = new Map<string, VisualNode>();
  sceneHandle.robots.ids().forEach((robotId, index) => {
    const node = robotVisualNodesInLoadOrder[index];
    if (node) robotVisualNodeById.set(robotId, node);
  });
  const visualNodeOf = (entityId: string): VisualNode | undefined =>
    robotVisualNodeById.get(entityId) ?? sceneHandle.visualNodes.get(entityId);

  // ── 충돌 모니터 + 시퀀스 player (Phase 4/5 코어를 앱에 배선) ──────

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
  if (registryEntry.sequence !== undefined) {
    const sequenceValidation = validateSequence(registryEntry.sequence, spec);
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

  // ── UI: 하단 독 (Timeline | Collision Log | Console) + 재생 바 ────

  const timelinePanel = createTimelinePanel();
  const collisionPanel = createCollisionLogPanel({
    onFocusEntity: (entityId) => {
      const node = visualNodeOf(entityId);
      if (node && entityId !== GROUND_ENTITY_ID) pulseEntity(node);
      // 카메라 포커스/당시 노드 강조는 Phase 10 (ROADMAP "Collision Log 연동")
      appLog('info', `충돌 로그: '${entityId}' 하이라이트 (카메라 포커스는 Phase 10)`);
    },
  });
  const consolePanel = createConsolePanel();
  mountDock(document.body, [
    { label: 'Timeline', content: timelinePanel.el },
    { label: 'Collision Log', content: collisionPanel.el },
    { label: 'Console', content: consolePanel.el },
  ]);

  // 충돌 → 로그 패널 행 추가 + start 시 관련 오브젝트 빨강 펄스 (UX_DESIGN §3.3/§3.6)
  monitor.subscribe((e) => {
    collisionPanel.addEvent(e);
    if (e.phase !== 'start') return;
    for (const entityId of [e.a, e.b]) {
      if (entityId === GROUND_ENTITY_ID) continue; // 바닥 전체 펄스는 소음 — 제외
      const node = visualNodeOf(entityId);
      if (node) pulseEntity(node);
    }
  });

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
    },
    setSpeed: (speedMult: number): void => {
      // select 옵션은 ENGINE_SPEED_OPTIONS에서 생성되므로 항상 유효하다
      engine.setSpeed(speedMult as EngineSpeed);
    },
  };

  const playbackBar = mountPlaybackBar(document.body, playbackControls, ENGINE_SPEED_OPTIONS);

  // 타임라인: 검증된 시퀀스의 step 마커 + player 커서 연동
  if (validSequence) {
    timelinePanel.setSequence(validSequence.steps.map((step) => step.kind));
    player.onStepChange((index) => {
      timelinePanel.setActiveIndex(index);
    });
  }

  // rAF당 1회: 재생 바 + 타임라인 리드아웃 갱신 (물리 tick과 분리된 뷰 갱신)
  engine.onTick((info) => {
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

  window.__sim = {
    engine,
    world,
    sceneHandle,
    spec,
    robots,
    collision: collisionFacade,
    ...(playerFacade ? { player: playerFacade } : {}),
  };

  // 로봇이 있는 씬이면 임시 관절 패널 마운트 (ROADMAP Phase 3 "슬라이더 수동 제어")
  if (sceneHandle.robots.ids().length > 0) {
    mountJointPanel(
      document.body,
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
    );
  }

  engine.start();
  engine.play(); // 물리 루프 자동 시작 — 시퀀스는 ▶ Play로만 (파일 헤더의 재생 정책)
  console.log(
    `Scene '${spec.name}' loaded — entities: [${sceneHandle.entityIds.join(', ')}], ${spec.timestepHz}Hz`,
  );
}

boot().catch((err: unknown) => {
  console.error('Bootstrap failed:', err);
  const msg = err instanceof Error ? (err.stack ?? err.message) : String(err);
  showErrorOverlay('부트스트랩 실패', [msg]);
});
