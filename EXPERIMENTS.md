# EXPERIMENTS.md — robot-sim-web

설계 결정·실험·트레이드오프의 **append-only 로그**. "왜"를 여기 남긴다("무엇"은
CLAUDE.md/docs에 반영). 새 항목은 아래에 추가하고 기존 항목은 수정하지 않는다.
결정이 뒤집히면 새 항목으로 supersede를 기록한다.

형식:

```
## [YYYY-MM-DD] 제목
- 상태: 결정됨 | 실험중 | 폐기됨 | supersedes #N | superseded by #N
- 맥락: 무엇을 정하려 했나
- 선택: 무엇을 택했나
- 근거: 왜
- 트레이드오프: 무엇을 포기했나
- 측정치(있으면): 수치·벤치마크
- 영향받는 파일/문서:
```

---

## [초기] 물리 엔진: Rapier(compat) 채택

- 상태: 결정됨
- 맥락: 브라우저 완결형 시뮬레이터의 물리 계층 선택
- 선택: `@dimforge/rapier3d-compat` (WASM). 렌더 three.js, 로봇 로딩 urdf-loader
- 근거: 충돌 이벤트/센서/CCD가 1급 기능, 웹 통합 성숙, 백엔드 불필요.
  compat 빌드는 WASM 내장으로 Vite 번들 이슈 회피
- 트레이드오프: 접촉 물리 정밀도는 MuJoCo보다 낮음 → 파지 사실성은 후순위
- 영향: CLAUDE.md §1/§9, docs/ARCHITECTURE.md, docs/SIMULATION.md

## [초기] 물리 계층 격리로 MuJoCo 교체 경로 확보

- 상태: 결정됨
- 맥락: 향후 접촉 정밀도 요구 시 물리 엔진 교체 가능성
- 선택: `core`는 `PhysicsWorld`/`ContactEvent` 인터페이스만 참조, Rapier 심볼 비노출
- 근거: 렌더/스키마/UI/시퀀스 재사용하며 물리만 교체 가능하게
- 트레이드오프: 얇은 추상화 계층 유지 비용
- 영향: CLAUDE.md §7, docs/ARCHITECTURE.md §6, docs/SIMULATION.md §7

## [초기] MVP 제어: kinematicPosition 로봇

- 상태: 결정됨
- 맥락: 로봇 구동 방식(kinematic vs dynamic)
- 선택: MVP는 kinematicPosition(관절 위치 직접 지정). 충돌은 감지되나 반력 없음
- 근거: "동작 재생 + 충돌 감지" 목표에 최소 복잡도. PD 튜닝 부담 제거
- 트레이드오프: 접촉 반력 동역학 없음(파지·밀기 물리 부정확) → MuJoCo 경로로 이관
- 영향: docs/SIMULATION.md §3, docs/DATA_MODEL.md §4.1

## [2026-07-23] 전면 재구축: 구버전 archived 보존, 설계서 기반 재시작

- 상태: 결정됨
- 맥락: 기존 autogrip-sim(백엔드+Isaac Sim 연동) 구현을 폐기하고 브라우저 완결형으로 전환
- 선택: 기존 main HEAD를 `archived` 브랜치로 보존, main은 설계서(docs/) 기반 처음부터 재개발
- 근거: 아키텍처가 근본적으로 다름(백엔드 필수 → 정적 호스팅). 점진 마이그레이션 이득 없음
- 트레이드오프: 구버전의 STL 프리셋·테스트 자산 재사용은 필요 시 archived에서 선별 복사
- 영향: 저장소 전체, docs/ROADMAP.md Phase 0부터 진행

## [2026-07-23] UI 프레임워크: vanilla TypeScript 채택 (React 미도입)

- 상태: 결정됨
- 맥락: ARCHITECTURE.md는 "React 또는 vanilla"를 허용. UX_DESIGN.md의 워크스페이스
  (커맨드바·라이브러리·인스펙터·플로우그래프·독)를 구현할 프레임워크 선택
- 선택: vanilla TypeScript + DOM/SVG. 프레임워크 미도입
- 근거: (1) package.json 고정 의존성 원칙 유지, (2) core/render 계층은 어차피 프레임워크
  비의존이어야 함 — UI만을 위한 React 도입은 번들 크기·의존성 대비 이득이 작음,
  (3) 플로우그래프는 커스텀 캔버스/SVG 구현이 필요해 프레임워크 이점이 제한적
- 트레이드오프: 인스펙터 폼 등 상태→DOM 바인딩 보일러플레이트 증가. 필요 시 얇은
  컴포넌트 헬퍼로 흡수. 훗날 React 전환 시 ui/ 계층만 교체(경계 규칙 덕분에 가능)
- 영향: src/ui 전반, package.json

## [2026-07-23] lint/test 하네스 관례

- 상태: 결정됨
- 맥락: ESLint 9 flat config + vitest 초기 설정
- 선택: `typescript-eslint` 메타 패키지(flat config), 미사용 인자는 `_` 접두사 허용,
  vitest `passWithNoTests`(Phase 0 시점 테스트 부재 허용), `npm run verify` =
  typecheck + lint + test 통합 게이트 스크립트
- 근거: DoD(CLAUDE.md §8) 검증을 단일 커맨드로 반복 가능하게
- 영향: eslint.config.js, vitest.config.ts, package.json

<!-- 이후 결정은 여기에 추가 -->
