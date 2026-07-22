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

## [2026-07-23] RenderSync 보간: three 내장 slerp 대신 순수 튜플 수학 추출

- 상태: 결정됨
- 맥락: Phase 1+2 pose 동기화(core/sync.ts) — prev/cur pose를 alpha로 보간해
  Object3D에 쓰는 핫 패스의 구현 방식 선택
- 선택: lerp/slerp/clamp를 `render/sync-math.ts` 순수 함수(튜플 in/out 파라미터)로
  구현하고 core/sync.ts는 이를 사용. three는 `import type { Object3D }`만 남김.
  slerp는 최단 경로(dot<0 시 부호 반전) + 근접 회전 시 nlerp 폴백
- 근거: (1) node 환경 vitest로 three 없이 단위 테스트 가능(vitest.config 원칙),
  (2) out-파라미터 스타일로 프레임당 신규 할당 0 보장, (3) core/sync의 런타임
  three 의존이 사라져 경계가 더 얇아짐(타입 전용). Engine의 구조적 PoseSync
  인터페이스(commit/apply)와 일치 확인
- 트레이드오프: three의 검증된 slerp 재구현 비용 → 13개 단위 테스트(항등·중간각·
  antipodal·단위길이·aliasing)로 상쇄. core/math.ts의 스칼라 clamp01과 이름 중복
  (모듈 순수성 각자 유지 목적, 통합은 추후 판단)
- 영향받는 파일/문서: src/core/sync.ts, src/render/sync-math.ts,
  src/render/sync-math.test.ts, docs/SIMULATION.md §5

## [2026-07-23] 바닥 시각 메시 1mm 침하 (z-fighting 회피)

- 상태: 결정됨
- 맥락: groundMesh(render/meshes.ts)의 상면을 y=0에 두면 Renderer의 y=0 그리드
  헬퍼 라인과 동일 평면이 되어 z-fighting 발생
- 선택: 시각 바닥 상면을 `GROUND_VISUAL_SINK_M = 0.001`(1mm)만큼 내림.
  물리 ENV collider 면은 정확히 y=0 유지(scene-loader 소관, 시각과 무관)
- 근거: 시각 전용 오프셋은 불변식 §2.1 예외 범주(순수 시각 요소)이고 1mm는
  지각 불가. 물리 정합성에는 영향 없음
- 트레이드오프: 시각-물리 1mm 불일치(문서화로 명시)
- 영향받는 파일/문서: src/render/meshes.ts

## [2026-07-23] schema 검증: 컨텍스트 한국어 에러 맵 + 명시적 체크 메시지

- 상태: 결정됨
- 맥락: DATA_MODEL §8 "사람이 읽을 수 있는 오류를 UI에 표시" — zod 오류의 한국어화 방식
- 선택: safeParse에 컨텍스트 errorMap(전역 한국어 번역)을 전달. 필드 특정 메시지는
  명시적 체크 메시지(`.min(1, msg)`, `.positive(msg)`, superRefine message)로만 지정.
  엔티티 type / step kind는 `z.discriminatedUnion`으로 구조 강제. 모든 숫자 리프는
  유한성(finite) 필수. 오류는 "경로: 메시지" 형식(예: `entities[0].urdf: …`)
- 근거: zod v3에서 safeParse에 넘긴 컨텍스트 errorMap이 스키마 바인딩 errorMap /
  `required_error` 파라미터를 **덮어쓴다**(우선순위 규칙). 명시적 체크 메시지만 살아남으므로
  거기에만 한국어를 직접 지정하고, 나머지는 전역 맵의 일반 문구로 통일
- 트레이드오프: 필수 필드 누락은 범용 문구("필수 항목이 누락되었습니다 (기대: <type>)")로
  떨어짐. robot 규칙(urdf/controller/visual.kind)은 별도 문장이 아닌 discriminated union의
  일반 문구로 보고됨. 씬 참조 오류는 구조 파싱 성공 후에만 검사되므로 구조/참조 오류가
  별도 라운드로 도착 — 호출자는 단일 완전 오류 목록을 가정하지 말 것
- 영향받는 파일/문서: src/schema/validate.ts, src/schema/index.ts(배럴 추가)

## [2026-07-23] RapierWorld Phase 1 구현 결정

- 상태: 결정됨
- 맥락: PhysicsWorld(core/types.ts 동결 계약) Rapier 구현의 세부 선택
- 선택: (1) timestep은 생성자에서 `world.timestep = 1/timestepHz`로 1회 고정, 이후 불변.
  (2) stop 이벤트 시점엔 collider가 이미 제거됐을 수 있으므로 sensor 여부를
  `Set<ColliderId>`로 자체 보관해 contact/sensor 분류. (3) `ColliderSpec.ccd`는 collider별
  플래그지만 Rapier CCD는 바디 단위 → 부모 바디에 `enableCcd(true)` 매핑(SIMULATION §4.1).
  (4) `interactionGroups`는 규약 공식 그대로 — DEBUG(비트 15) 소속 시 부호 있는 32비트
  시프트로 JS 음수가 되지만 비트 패턴은 올바른 u32(비교는 비트 동등으로 할 것)
- 근거: 계약(core/types.ts)이 유일한 권위 — ARCHITECTURE §6 스케치의 setJointTarget은
  계약에 없으므로 미구현(관절 제어는 후속 Phase에서 계약 변경으로 도입)
- 트레이드오프: 한 collider의 ccd:true가 같은 바디의 모든 collider에 적용됨.
  getPose는 호출마다 새 튜플 반환(계약상) — 완전 무할당 경로가 필요해지면
  getPoseInto 변형을 계약에 추가 검토
- 측정치: rapier3d-compat는 Node(vitest)에서 구동됨 — 물리 통합 12 테스트 ~65ms
- 영향받는 파일/문서: src/core/world.ts, src/core/world.test.ts, vitest.config.ts(주석 갱신)

## [2026-07-23] Engine 루프: 의존성 객체 생성자 + hooks 주입, floor-division accumulator

- 상태: 결정됨
- 맥락: SIMULATION §2 스켈레톤은 Engine(world, render, hz, player, collision, sync)
  위치 인자 + player/collision 직접 참조. Phase 1엔 player/collision이 아직 없음
- 선택: `new Engine({world, sync, render, hooks?}, timestepHz)` — ControlPlayer.step은
  hooks.preStep으로, CollisionMonitor.dispatch는 hooks.onContacts로 후속 Phase에 주입.
  accumulator는 반복 감산 대신 정수 나눗셈(floor)으로 스텝 수 산출(부동소수 누적 오차 최소화).
  easing 'easeInOut'은 코사인식(0.5-0.5cos(πt)), 'step'은 duration 종료 순간에만 점프.
  stop()은 'paused'가 아닌 'idle' 전환 + 시계 리셋(simTime=0) — 씬 리셋은 scene-loader 소관.
  onContacts는 simTime 증가 "이전"(tick 시작 시각)으로 호출 — 충돌 타임스탬프 규범
- 근거: 아직 없는 모듈에 대한 위치 인자 대신 구조적 인터페이스(PoseSync/RenderTarget)와
  선택적 hooks로 결합도 최소화. fixedDt 이중 소스(world/Engine)는 불일치 시 console.warn
- 트레이드오프: SIMULATION §2 스켈레톤과 생성자 시그니처가 다름(계약 문서는 유지, 코드가
  주입 지점을 명시). 'stopped-but-paused' 상태가 필요하면 후속 결정 필요
- 영향받는 파일/문서: src/core/engine.ts, src/core/math.ts, src/core/stepper.test.ts

## [2026-07-23] scene-loader: 좁은 RenderSceneApi 주입 + opaque VisualNode

- 상태: 결정됨
- 맥락: Phase 1+2 통합 — SceneSpec → 물리 바디 + 시각 메시 생성을 core에서 하되
  "core는 three를 모른다"(sync.ts 예외) 규칙을 지키는 방법
- 선택: scene-loader가 `RenderSceneApi`(addPrimitive/addGround/setPose/remove) 인터페이스를
  정의하고 구현은 main.ts 글루가 three(render/meshes) 위에서 제공. 시각 노드 타입은
  `Parameters<RenderSync['bind']>[1]`로 유도한 opaque `VisualNode` — three 심볼 직접 import 없음.
  environment.ground → 예약 id `__ground`의 ENV 고정 바디(halfExtents [10,0.05,10],
  상면 y=0) + groundMesh 시각(자체 배치, sync 미바인딩). 물리 있는 엔티티는 전부
  sync.bind(고정 벽 포함 — pose 불변이라 무해·균일). SceneHandle.reset()은 초기 스펙
  트랜스폼으로 teleport(속도 0) — 결정론적 재생. 실패 시 부분 생성 자원 teardown 후 재던짐.
  robot 엔티티는 Phase 3까지 명시적 오류
- 근거: 렌더 구현을 인터페이스 뒤로 밀면 헤드리스(Node) 통합 테스트가 no-op 구현으로
  가능해짐 — falling-boxes 씬을 vitest에서 3초 시뮬해 정착/충돌 이벤트 검증
- 트레이드오프: 시각 전용 처리(그림자·색)는 main 글루에 있음 — ui 계층 도입 시 이동 예정.
  reset() 직후 1프레임은 prev pose 보간이 남아 시각적으로만 이전 위치와 섞일 수 있음
- 측정치: 브라우저 게이트(scripts/gate-browser.mjs --expect=falling-boxes) ALL PASS —
  5개 동적 바디 정착 높이 0.029–0.059m(각 형상 반높이/반지름과 일치), 페이지 에러 0
- 영향받는 파일/문서: src/core/scene-loader.ts, src/main.ts,
  src/assets/scenes/falling-boxes.scene.json, src/core/scene-loader.test.ts

## [2026-07-23] 리뷰 수정: removeEntity stop-이벤트 tombstone + stop/reset 시각 정합

- 상태: 결정됨
- 맥락: Phase 1+2 리뷰 지적 반영. (1) removeEntity가 collider 매핑을 즉시 지워,
  Rapier가 제거된 collider의 활성 접촉에 대해 다음 스텝에 발행하는 'stop' 이벤트가
  step()의 매핑 가드에서 버려짐 — start/stop 짝 소비자(Phase 4 CollisionMonitor,
  waitForCollision 이력)가 '고착' 접촉을 보게 될 잠재 결함. (2) Engine.stop()이
  lastAlpha=0을 남겨 idle 프레임이 prev 스냅샷(리셋 전 pose)을 계속 그리고,
  SceneHandle.reset()이 sync prev를 갱신하지 않아 stop+reset 후 화면이 물리 진실과
  어긋남 (CLAUDE.md §2.1 위반 — 위 항목의 "reset() 직후 1프레임" 트레이드오프의 일반화)
- 선택: (1) RapierWorld에 `removedColliderToEntity` tombstone 맵 추가 — removeEntity가
  살아있는 매핑에서 지우면서 tombstone에 등록, step()의 drain이 live 매핑 우선 +
  tombstone 폴백으로 번역, drain 종료 시 flush(sensorColliders 엔트리도 그때 정리 —
  "stop 이벤트 시점엔 collider가 이미 제거됐을 수 있으므로" 주석이 이제 실제로 참).
  제거 유발 stop 이벤트가 제거 후 첫 step()에 오는 것을 probe로 실증(rapier 0.14 compat).
  (2) Engine.stop()은 lastAlpha=ALPHA_LATEST(1) — idle 렌더가 최신 물리 pose를 그림
  (stepOnce와 동일 규범). SceneHandle.reset()은 텔레포트 후 sync.commit()으로 prev 갱신
- 근거: 공개 API(entityOfCollider)는 제거 즉시 undefined 유지(계약 불변), tombstone은
  drain 내부 전용이라 핸들 매핑 진실성·결정론에 영향 없음. alpha=1 렌더는 보간 생략일
  뿐 물리 상태 불변(시각 전용 — §2.3)
- 트레이드오프: Rapier가 핸들 번호를 세대 없이 즉시 재사용한다면 tombstone 오귀속
  가능성이 이론상 있으나, 0.14는 세대 인코딩 핸들이라 스텝 사이 재사용이 없다
- 검증: src/core/world-remove-contact.test.ts(contact/sensor 제거 stop + tombstone 수명),
  src/core/scene-reset-sync.test.ts(reset 후 apply(0)이 초기 pose를 그림) 회귀 테스트 추가
- 영향받는 파일/문서: src/core/world.ts, src/core/engine.ts, src/core/scene-loader.ts

<!-- 이후 결정은 여기에 추가 -->
