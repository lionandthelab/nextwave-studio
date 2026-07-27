import { defineConfig } from 'vite';

// robot-sim-web — Vite 설정
//
// Rapier는 `@dimforge/rapier3d-compat`를 쓴다. compat 빌드는 WASM을 base64로 내장해
// top-level await / ESM-WASM 관련 번들러 이슈를 피한다(CLAUDE.md §9). 따라서 별도
// wasm 플러그인 없이도 동작하는 것이 기본 기대다. 아래는 안전장치성 설정.
//
// 만약 비-compat `@dimforge/rapier3d`로 전환한다면 top-level await 지원 target과
// wasm 로딩 플러그인이 추가로 필요하다. 그 전까지는 compat 유지가 불변식에 가깝다.

// Docker 컨테이너에서 개발 서버를 돌릴 때만 폴링 감시를 켠다. bind mount는 호스트의
// 파일 변경이 inotify 이벤트로 전파되지 않아 HMR이 죽는다(Windows/macOS Docker Desktop
// 에서 특히). docker-compose의 dev 서비스가 VITE_USE_POLLING=true를 주입한다.
// @types/node를 추가하지 않기 위해 필요한 만큼만 앰비언트 선언한다.
declare const process: { env: Record<string, string | undefined> };
const useContainerPolling = process.env.VITE_USE_POLLING === 'true';

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
    // 컨테이너 안에서만 폴링 — 호스트 개발 시에는 기본(네이티브 감시) 그대로다.
    ...(useContainerPolling ? { watch: { usePolling: true, interval: 300 } } : {}),
    // 필요 시 COOP/COEP 헤더로 SharedArrayBuffer 활성화(멀티스레드 WASM 사용 시)
    // headers: {
    //   'Cross-Origin-Opener-Policy': 'same-origin',
    //   'Cross-Origin-Embedder-Policy': 'require-corp',
    // },
  },
});
