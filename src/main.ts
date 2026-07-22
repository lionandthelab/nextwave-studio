// main.ts — 부트스트랩 진입점 (Phase 1+2: 데이터 주도 falling-boxes 데모)
//
// 순서 고정 (docs/ARCHITECTURE.md §4, CLAUDE.md §2.7):
//   1. await initPhysics()      — WASM 로드 완료 전 물리 API 호출 금지
//   2. SceneSpec JSON 검증      — 실패 시 사람이 읽을 수 있는 오류 오버레이 (DATA_MODEL §8)
//   3. RapierWorld 생성         — spec.gravity / spec.timestepHz
//   4. Renderer 생성            — spec.camera / spec.environment 반영
//   5. SceneLoader.build        — 바디 + 메시 생성, sync 바인딩
//   6. (Phase 5+) 시퀀스 로드
//   7. Engine 루프 시작
//
// main은 조립 글루다: core(엔진·월드·로더)와 render(three)를 여기서 잇는다.
// scene-loader가 요구하는 RenderSceneApi의 three 구현도 여기서 제공한다
// (core는 three를 모른 채 이 좁은 인터페이스만 호출한다 — CLAUDE.md §3).

import { Engine } from './core/engine';
import { SceneLoader } from './core/scene-loader';
import type { RenderSceneApi, SceneHandle } from './core/scene-loader';
import { RenderSync } from './core/sync';
import type { PhysicsWorld } from './core/types';
import { initPhysics, RapierWorld } from './core/world';
import { groundMesh, primitiveMesh } from './render/meshes';
import { Renderer } from './render/renderer';
import { validateScene } from './schema';
import type { SceneSpec } from './schema';
import fallingBoxesSceneJson from './assets/scenes/falling-boxes.scene.json';

// ── 자동화/AI-native 훅 (Playwright 게이트 · 추후 ui 계층이 사용) ────

/** window.__sim으로 노출되는 시뮬 핸들. Rapier 타입은 새지 않는다(PhysicsWorld 경계). */
export interface SimHandle {
  readonly engine: Engine;
  readonly world: PhysicsWorld;
  readonly sceneHandle: SceneHandle;
  readonly spec: SceneSpec;
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

  await initPhysics();
  console.log('Rapier ready');

  // 씬은 데이터다 — 코드가 아니라 JSON이 씬을 정의한다 (CLAUDE.md §2.5)
  const validation = validateScene(fallingBoxesSceneJson);
  if (!validation.ok) {
    console.error('Scene validation failed:', validation.errors);
    showErrorOverlay('씬 검증 실패 — falling-boxes.scene.json', validation.errors);
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
  };

  const sceneHandle = new SceneLoader(world, renderApi, sync).build(spec);

  const engine = new Engine(
    {
      world,
      sync,
      render,
      hooks: {
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

  window.__sim = { engine, world, sceneHandle, spec };

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
