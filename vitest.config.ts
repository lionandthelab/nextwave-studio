import { defineConfig } from 'vitest/config';

// 순수 로직(core 진행/보간, schema 검증, planner 검증 루프)은 node 환경에서 돈다.
// three.js/Rapier(WASM)에 의존하는 테스트는 만들지 않는 것이 기본 원칙이며,
// 물리 통합 검증은 브라우저 게이트(ROADMAP)에서 수행한다.
export default defineConfig({
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts'],
    passWithNoTests: true,
  },
});
