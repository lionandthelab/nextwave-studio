import { defineConfig } from 'vitest/config';

// 순수 로직(core 진행/보간, schema 검증, planner 검증 루프)은 node 환경에서 돈다.
// rapier3d-compat는 WASM을 base64로 내장하므로 Node에서도 구동된다 — 물리 통합
// 테스트(world.test / scene-loader.test)는 initPhysics() await 후 vitest에서 직접
// 수행한다. three.js "렌더" 의존 테스트는 만들지 않는다(render/sync-math처럼 순수
// 수학으로 추출해 테스트). 브라우저 전체 파이프라인 검증은 scripts/gate-browser.mjs.
export default defineConfig({
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts', 'server/**/*.test.ts'],
    passWithNoTests: true,
  },
});
