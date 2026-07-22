import { defineConfig } from 'vite';

// robot-sim-web — Vite 설정
//
// Rapier는 `@dimforge/rapier3d-compat`를 쓴다. compat 빌드는 WASM을 base64로 내장해
// top-level await / ESM-WASM 관련 번들러 이슈를 피한다(CLAUDE.md §9). 따라서 별도
// wasm 플러그인 없이도 동작하는 것이 기본 기대다. 아래는 안전장치성 설정.
//
// 만약 비-compat `@dimforge/rapier3d`로 전환한다면 top-level await 지원 target과
// wasm 로딩 플러그인이 추가로 필요하다. 그 전까지는 compat 유지가 불변식에 가깝다.

export default defineConfig({
  build: {
    target: 'es2022',            // top-level await 등 최신 기능 여지 확보
  },
  optimizeDeps: {
    // compat 빌드가 사전 번들 과정에서 문제될 경우 제외(필요 시 주석 해제)
    // exclude: ['@dimforge/rapier3d-compat'],
  },
  worker: {
    format: 'es',
  },
  server: {
    // 필요 시 COOP/COEP 헤더로 SharedArrayBuffer 활성화(멀티스레드 WASM 사용 시)
    // headers: {
    //   'Cross-Origin-Opener-Policy': 'same-origin',
    //   'Cross-Origin-Embedder-Policy': 'require-corp',
    // },
  },
});
