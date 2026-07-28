# TECH_STACK — robot-sim-web 구현 기술 스택

실제 구현에 쓰인 기술의 한 장 요약. 설계 근거는 `docs/ARCHITECTURE.md`,
결정 이력은 `EXPERIMENTS.md`.

> **핵심 제약**: 백엔드 없음(PRD NFR-1). 브라우저에서 물리·렌더·로봇 로딩·LLM 호출이
> 모두 끝나고, 산출물은 정적 파일이다.

---

## 1. 런타임 의존성 — 5개

| 패키지 | 버전 | 역할 | 선택 이유 |
|-------|------|------|----------|
| `@dimforge/rapier3d-compat` | 0.14.0 | 물리 엔진 (WASM) | 충돌 이벤트·센서·CCD가 1급 기능. **compat** 빌드는 WASM을 base64로 내장해 Vite의 top-level await / ESM-WASM 이슈를 회피 |
| `three` | 0.169.0 | 3D 렌더링 | 웹 3D 사실상 표준. 예제 모듈(컨트롤·로더)까지 한 패키지로 해결 |
| `urdf-loader` | 0.12.7 | URDF → three.js | 로봇 기술 표준 포맷 파싱 + 관절 FK 그래프 제공 |
| `zod` | 3.25.76 | 런타임 스키마 검증 | 외부 JSON(씬·시퀀스·LLM 출력)의 신뢰 경계. TS 타입과 1:1 미러링 |
| `@anthropic-ai/sdk` | 0.113.0 | LLM 어댑터 (선택적) | 자연어 플래너의 Anthropic 백엔드. **기본 백엔드는 오프라인 규칙 기반**이라 이 의존성 없이도 전 기능 동작 |

의존성을 5개로 묶은 것이 의도적 선택이다 — 상태관리·UI 프레임워크·유틸리티 라이브러리
없이 vanilla TypeScript로 구현했다(§5).

### three.js 서브모듈 사용

| 모듈 | 용도 |
|-----|------|
| `controls/OrbitControls` | 카메라 orbit/pan/zoom |
| `controls/TransformControls` | 씬 편집 기즈모(이동·회전·스케일) |
| `loaders/GLTFLoader` · `STLLoader` · `OBJLoader` | 3D 파일 임포트 (`.glb/.gltf` · `.stl` · `.obj`) |

---

## 2. 개발 도구

| 도구 | 버전 | 역할 |
|-----|------|------|
| `typescript` | 5.9.3 | strict 모드. `any` 금지(ESLint error), `noUncheckedIndexedAccess` |
| `vite` | 5.4.21 | 개발 서버(HMR) + 정적 번들링 (target `es2022`) |
| `vitest` | 2.1.9 | 단위·통합 테스트 **724개**. node 환경에서 Rapier WASM 실제 구동 |
| `eslint` + `typescript-eslint` | 9.39.5 / 8.65.0 | flat config. 경고 0 유지 |
| `playwright` | 1.61.1 | 실브라우저 게이트(물리 어서션) — 컨테이너 이미지에는 미포함 |
| `nginx` (Docker) | 1.27-alpine | 프로덕션 정적 서빙 |

**검증 계층**
`npm run verify`(tsc + eslint + vitest) → `scripts/gate-browser.mjs`(실브라우저 게이트
11종) → `scripts/verify-container.mjs`(도커 배포 형상).

---

## 3. 계층별 기술 매핑 — "어느 계층이 무엇을 아는가"

이 표가 아키텍처의 핵심이다. 각 기술은 **정확히 한 계층에만** 갇혀 있다.

```
ui → {core, planner} → {render, schema}
```

| 계층 | 파일/LOC | 아는 기술 | 모르는 기술 |
|-----|---------|----------|------------|
| `schema/` | 4 / 1.3k | zod | 물리·렌더·DOM 전부 |
| `core/` | 14 / 3.4k | **Rapier**(`world.ts` 단독) | three.js (예외: `sync.ts`는 타입만), DOM |
| `planner/` | 7 / 1.2k | **Anthropic SDK**(`adapters/anthropic.ts` 단독) | 물리·렌더·DOM·localStorage |
| `render/` | 7 / 2.2k | **three.js**, urdf-loader | 물리·UI 상태 |
| `ui/` | 26 / 11.3k | DOM | Rapier·three 심볼(주입된 인터페이스로만 접근) |

**격리의 대가로 얻는 것**

| 경계 | 교체 가능해지는 것 |
|-----|------------------|
| `PhysicsWorld` 인터페이스 (`core/types.ts`) | Rapier → **MuJoCo WASM** (물리 계층만, 렌더·스키마·UI·시퀀스 재사용) |
| `LlmAdapter` 인터페이스 (`planner/llm-adapter.ts`) | Anthropic → 다른 LLM/로컬 모델 |
| `RenderSceneApi` (`core/scene-loader.ts`) | three.js → 다른 렌더러 |
| `ui/` 전체 | vanilla → React 등 (core/render 무변경) |

---

## 4. 규모

| 항목 | 수치 |
|-----|------|
| 소스 | 59 파일 / 22,356 LOC (테스트 제외) |
| 테스트 | 34 파일 / **724개** (vitest) |
| 브라우저 게이트 | 11종 (Playwright) |
| 프로덕션 번들 | 3.25 MB (gzip **1.09 MB**) — 대부분 Rapier WASM base64 |
| 런타임 의존성 | 5개 |

---

## 5. 채택하지 않은 것

| 후보 | 대신 선택 | 이유 |
|-----|----------|------|
| React / Vue | vanilla TS + DOM/SVG | core·render는 어차피 프레임워크 비의존이어야 하고, 플로우 그래프는 커스텀 캔버스가 필요해 프레임워크 이점이 제한적. UI만을 위한 번들·의존성 증가 회피 |
| 상태관리 라이브러리 | 계층별 소유권 규칙 | 시뮬 진실은 물리(Rapier), 뷰 상태는 ui가 소유 — 전역 스토어가 오히려 진실을 흐림 |
| `@dimforge/rapier3d` (비-compat) | `rapier3d-compat` | top-level await / ESM-WASM 번들러 이슈 회피 |
| CSS 프레임워크 | `ui/theme.ts` 디자인 토큰 | 토큰 + 주입 스타일시트로 충분, 빌드 파이프라인 단순 유지 |
| 서버 사이드 LLM 프록시 | 브라우저 직접 호출(사용자 키) | 백엔드 없음 제약. 대신 기본은 오프라인 규칙 기반이고, 키는 localStorage에만 저장(교육·프로토타입 전제 — PRD §6) |

---

## 6. 배포 형상

| 방식 | 구성 |
|-----|------|
| **Docker** (권장) | 멀티스테이지: `node:22-alpine`(빌드) → `nginx:1.27-alpine`(서빙). 런타임 이미지에 Node 없음. Windows/Linux/macOS 동일 |
| **정적 호스팅** | `npm run build` → `dist/`를 아무 정적 호스트에 업로드 |

`docker-compose.yml`은 `app`(프로덕션 8080) · `dev`(Vite HMR 5173) · `verify`(CI) 세
프로파일을 제공한다. 자세한 실행법은 `docs/USAGE.md` §10.2.

---

## 7. 브라우저 요구사항

- **WebGL2** (three.js 렌더링)
- **WebAssembly** (Rapier 물리)
- ES2022 문법 지원

GPU 없는 일반 노트북에서 동작한다(PRD §7 성공 기준). 모바일은 뷰어 수준.

---

## 관련 문서

`docs/ARCHITECTURE.md`(계층·데이터 흐름 상세) · `docs/DATA_MODEL.md`(스키마 규범) ·
`docs/SIMULATION.md`(Rapier API 사용 규범) · `docs/PLANNER.md`(LLM 어댑터 설계) ·
`docs/USAGE.md`(사용법) · `CLAUDE.md` §7(MuJoCo 교체 경로) · `EXPERIMENTS.md`(결정 이력).
