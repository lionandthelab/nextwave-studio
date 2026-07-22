// main.ts — 부트스트랩 진입점 (Phase 1–3: 데이터 주도 씬 + URDF 로봇)
//
// 순서 고정 (docs/ARCHITECTURE.md §4, CLAUDE.md §2.7):
//   1. await initPhysics()      — WASM 로드 완료 전 물리 API 호출 금지
//   2. SceneSpec JSON 검증      — 실패 시 사람이 읽을 수 있는 오류 오버레이 (DATA_MODEL §8)
//   3. RapierWorld 생성         — spec.gravity / spec.timestepHz
//   4. Renderer 생성            — spec.camera / spec.environment 반영
//   5. await SceneLoader.build  — 바디 + 메시 + URDF 로봇 생성, sync/robot 바인딩
//   6. (Phase 5+) 시퀀스 로드
//   7. Engine 루프 시작 (preStep 훅에서 robots.tickAll — FK → kinematic 바디 push)
//
// main은 조립 글루다: core(엔진·월드·로더)와 render(three)를 여기서 잇는다.
// scene-loader가 요구하는 RenderSceneApi의 three 구현(loadRobot 포함)도 여기서 제공한다
// (core는 three를 모른 채 이 좁은 인터페이스만 호출한다 — CLAUDE.md §3).
// 씬 선택은 ?scene= 쿼리 파라미터로 한다 — 씬은 데이터다 (CLAUDE.md §2.5).

import { Engine } from './core/engine';
import { SceneLoader } from './core/scene-loader';
import type { RenderSceneApi, SceneHandle } from './core/scene-loader';
import type { JointInfo } from './core/robot-types';
import { RenderSync } from './core/sync';
import type { PhysicsWorld, Pose } from './core/types';
import { initPhysics, RapierWorld } from './core/world';
import { groundMesh, primitiveMesh } from './render/meshes';
import { Renderer } from './render/renderer';
import { loadUrdfRobot } from './render/urdf';
import { mountJointPanel } from './ui/inspector/joint-panel';
import { validateScene } from './schema';
import type { SceneSpec } from './schema';
import fallingBoxesSceneJson from './assets/scenes/falling-boxes.scene.json';
import armAndBoxesSceneJson from './assets/scenes/arm-and-boxes.scene.json';

// ── 씬 레지스트리 (?scene= 파라미터로 선택 — 새 씬 = 새 데이터) ──────

const SCENE_JSONS: Readonly<Record<string, unknown>> = {
  'falling-boxes': fallingBoxesSceneJson,
  'arm-and-boxes': armAndBoxesSceneJson,
};

const DEFAULT_SCENE_NAME = 'arm-and-boxes';

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

/** window.__sim으로 노출되는 시뮬 핸들. Rapier 타입은 새지 않는다(PhysicsWorld 경계). */
export interface SimHandle {
  readonly engine: Engine;
  readonly world: PhysicsWorld;
  readonly sceneHandle: SceneHandle;
  readonly spec: SceneSpec;
  readonly robots: SimRobotsFacade;
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
  const sceneJson = SCENE_JSONS[sceneName];
  if (sceneJson === undefined) {
    showErrorOverlay('알 수 없는 씬', [
      `'${sceneName}' 씬을 찾을 수 없습니다.`,
      `사용 가능한 씬: ${Object.keys(SCENE_JSONS).join(', ')}`,
      `예: ?scene=${DEFAULT_SCENE_NAME}`,
    ]);
    return;
  }

  await initPhysics();
  console.log('Rapier ready');

  // 씬은 데이터다 — 코드가 아니라 JSON이 씬을 정의한다 (CLAUDE.md §2.5)
  const validation = validateScene(sceneJson);
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
    loadRobot: (request) =>
      loadUrdfRobot(render.scene, {
        urdfPath: request.urdfPath,
        packages: request.packages,
      }),
  };

  const sceneHandle = await new SceneLoader(world, renderApi, sync).build(spec);

  const engine = new Engine(
    {
      world,
      sync,
      render,
      hooks: {
        // 매 물리 tick, world.step() 직전: 관절 상태 → FK → kinematic 링크 바디 push
        // (core/robot-types.ts 설계 — 로봇이 없으면 no-op)
        preStep: () => {
          sceneHandle.robots.tickAll();
        },
        // Phase 4에서 CollisionMonitor + UI 충돌 로그 패널로 대체 — 지금은 콘솔 확인 (DoD §8)
        onContacts: (events, simTimeSec) => {
          for (const e of events) {
            console.log(
              `[collision] t=${simTimeSec.toFixed(3)}s ${e.a} <-> ${e.b} ${e.phase} (${e.kind})`,
            );
          }
        },
      },
    },
    spec.timestepHz, // world와 동일한 timestepHz — 이중 소스 불일치 경고 방지
  );

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

  window.__sim = { engine, world, sceneHandle, spec, robots };

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
  engine.play();
  console.log(
    `Scene '${spec.name}' loaded — entities: [${sceneHandle.entityIds.join(', ')}], ${spec.timestepHz}Hz`,
  );
}

boot().catch((err: unknown) => {
  console.error('Bootstrap failed:', err);
  const msg = err instanceof Error ? (err.stack ?? err.message) : String(err);
  showErrorOverlay('부트스트랩 실패', [msg]);
});
