// main.ts — 부트스트랩 진입점
//
// 순서 고정 (docs/ARCHITECTURE.md §4, CLAUDE.md §2.7):
//   1. await initPhysics()  — WASM 로드 완료 전 물리 API 호출 금지
//   2. (Phase 2+) SceneSpec 검증·로드
//   3. PhysicsWorld 생성
//   4. Renderer 생성
//   5. (Phase 2+) scene-loader로 바디+메시 생성
//   6. (Phase 5+) 시퀀스 로드
//   7. 루프 시작
//
// Phase 0: 빈 world + 빈 3D 씬(그리드) 렌더 확인.

import { initPhysics, RapierWorld } from './core/world';
import { Renderer } from './render/renderer';

const DEFAULT_GRAVITY: [number, number, number] = [0, -9.81, 0];
const DEFAULT_TIMESTEP_HZ = 240;

async function boot(): Promise<void> {
  const host = document.getElementById('app');
  if (!host) throw new Error('#app host element not found');

  await initPhysics();
  console.log('Rapier ready');

  const world = new RapierWorld(DEFAULT_GRAVITY, DEFAULT_TIMESTEP_HZ);
  console.log(`Physics world created (fixed dt = ${world.fixedDtSec.toFixed(5)}s)`);

  const render = new Renderer(host);

  // Phase 0 임시 루프: 렌더만. Phase 1에서 Engine(고정 timestep accumulator)으로 대체.
  const frame = (): void => {
    render.draw();
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

boot().catch((err: unknown) => {
  console.error('Bootstrap failed:', err);
  const host = document.getElementById('app');
  if (host) {
    const msg = err instanceof Error ? err.message : String(err);
    host.innerHTML = `<pre style="color:#ff6b6b;padding:16px;font-family:monospace">Bootstrap failed:\n${msg}</pre>`;
  }
});
