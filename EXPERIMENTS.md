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

## [2026-07-23] Phase 3: 로봇 관절 구동 설계 — 관절 진실은 core, FK는 render, kinematic push

- 상태: 결정됨
- 맥락: URDF 로봇을 "물리가 진실"(CLAUDE.md §2.1) 불변식과 충돌 없이 구동해야 한다.
  로봇 링크 pose의 원천이 관절값(FK)인데, FK 계산기는 render의 urdf-loader 씬 그래프다.
- 선택: 관절 상태(joint values)의 유일한 진실은 core의 `RobotBinding`이 소유한다.
  FK(관절값 → 링크 월드 pose)는 render의 urdf-loader 씬 그래프가 수행하고, core는
  `RobotFkView` 인터페이스(POJO만 — three 심볼 비노출)로만 읽는다. 매 물리 tick,
  world.step() "직전"(Engine `preStep` 훅)에 `RobotRegistry.tickAll()`이 FK 링크 pose를
  kinematicPosition 바디에 `setKinematicPose`(다음 스텝 목표)로 밀어 넣는다. 시각 로봇
  (urdf Object3D)은 같은 관절 상태에서 직접 갱신된다 — 그 자체가 FK 그래프이므로
  로봇 링크는 RenderSync에 바인딩하지 않는다. URDF Z-up → 내부 Y-up 변환은 render의
  URDF 래퍼(axisFix 그룹, rotation.x = -π/2) 한 곳에서만 수행한다(CLAUDE.md §4).
- §2.1 불변식 해석: kinematic 로봇에서 "물리가 트랜스폼의 진실"은 링크 바디가 관절
  상태라는 단일 원천에서 유도된 pose를 매 스텝 물리 목표로 받아들인다는 뜻이다.
  물리(충돌·접촉 상대)가 보는 pose와 시각 pose가 같은 관절 상태에서 유도되므로 진실이
  갈라지지 않고, three.js 쪽에서 물리와 "어긋나게" 값을 쓰는 역방향 경로도 없다
  (setJointValues → FK → 물리 push의 단방향 파이프라인).
- 부속 결정: (1) `readLinkPoses`는 collider 보유 링크만 반환(240Hz 핫 패스에서 소비처
  없는 pose 산출 회피). (2) urdf-loader 0.12는 collision 프리미티브를 "단위 지오메트리
  × mesh.scale"로 만들므로 형상 치수 = 지오메트리 파라미터 × |링크-상대 TRS scale|,
  offset은 position/rotation만 담는다(URDF Z축 실린더의 rotation.x=π/2도 offset에 포함
  — Rapier 실린더는 Y축 정렬이라 그대로 맞아떨어진다). (3) mimic/planar/floating 관절은
  구동 대상에서 제외(mimic은 urdf-loader가 자동 추종). (4) `applyHome()`은 home 키만이
  아니라 전 관절을 fk initial로 되돌린 뒤 home을 적용 — home에 없는 관절도 결정론적으로
  복원(SIMULATION.md §6). (5) 유효 limits = URDF ∩ jointLimitOverrides 교집합, 빈
  교집합은 생성 시점 오류. (6) 비유한 관절값은 한국어 오류로 거부 — NaN이 FK/물리
  경로에 들어가지 않게. (7) `readJoints(names)`는 요청한 이름(논리명 허용) 그대로 키를
  써서 반환 — player의 moveJoints 시작값 스냅샷 계약(SIMULATION §3)과 맞춘다.
- 트레이드오프: readLinkPoses가 호출마다 Map/Pose를 새로 할당(240Hz) — 계약에 소유권
  규정이 없어 안전 우선. GC 압력이 측정되면 robot-types 계약에 버퍼 재사용을 추가 검토.
  tick()의 Object.fromEntries 할당도 동일 판단(MVP 규모에서 무시 가능). limit 태그 없는
  revolute는 urdf-loader가 lower=upper=0을 주므로 [0,0]으로 고정된 관절이 된다(URDF
  명세상 revolute의 limit는 필수 — 오류 대신 명세대로 해석).
- 영향받는 파일/문서: src/core/robot-types.ts, src/core/robots.ts, src/render/urdf.ts,
  src/core/engine.ts(preStep 훅 사용), src/core/scene-loader.ts, src/main.ts

## [2026-07-23] Phase 3: linkColliders 'fromVisual'은 URDF <collision> 태그로 해석

- 상태: 결정됨
- 맥락: RobotSpec.linkColliders 정책(DATA_MODEL §4.1)의 'fromVisual'을 구현해야 한다.
  이름은 "시각 메시에서 유도"를 시사하지만, URDF는 이미 링크별 <collision> 태그로
  단순화된 충돌 형상을 별도 제공한다.
- 선택: 'fromVisual'(기본값)을 "URDF <collision> 태그의 프리미티브(box/sphere/cylinder)
  사용"으로 해석한다(`loader.parseCollision = true` → `RobotFkView.linkColliders`).
  'primitive'도 현재 동일 경로다. 'none'은 물리 바디를 아예 만들지 않는다(시각 전용 —
  로봇은 레지스트리에 등록되어 관절 조작은 가능). collider 없는 링크는 바디도 없다.
  파일 메시 collision 지오메트리는 MVP 미지원 — 경고 후 건너뜀.
- 근거: URDF <collision>이 이미 "단순 프리미티브 collider"(CLAUDE.md §2.2)의 규범적
  원천이다. 시각 메시 AABB/hull 자동 유도는 메시 임포트 파이프라인(Phase 7)과 함께
  도입해도 늦지 않다.
- 트레이드오프: <collision>이 없는 URDF는 충돌 감지가 안 된다 — 자체 제작 arm6.urdf는
  전 링크에 collision을 명시했다. visual-유래 자동 생성이 필요해지면 'fromVisual'과
  'primitive'를 분화시킨다.
- 영향받는 파일/문서: src/render/urdf.ts, src/core/scene-loader.ts(robot 브랜치),
  src/assets/scenes/arm-and-boxes.scene.json, public/assets/robots/arm6/arm6.urdf

## [2026-07-23] Phase 3: selfCollision=false는 그룹 필터로 구현 (인접 쌍 필터링 불필요)

- 상태: 결정됨
- 맥락: URDF self-collision 기본 비활성(CLAUDE.md §5 "인접 링크 무시") 구현 방법.
  통상 구현은 인접 링크 쌍별 예외 필터인데, Rapier에서 쌍별 예외는 번거롭다.
- 선택: 로봇 링크 collider는 전부 ROBOT 그룹 소속이므로, selfCollision=false(기본)면
  `collidesWith`에서 ROBOT을 제외한다(['ENV','OBJECT']). 필터에 ROBOT이 없으면 링크끼리는
  broad-phase에서 걸러져 인접이든 아니든 충돌 자체가 없다 — 인접 쌍 열거가 불필요.
  selfCollision=true면 ['ENV','OBJECT','ROBOT'] — 이때는 인접 링크(관절부에서 형상이
  겹침)도 충돌 대상이 되므로 주의(현 MVP에 true 사용처 없음). 링크 collider 스펙:
  emitEvents=true(로봇–사물/환경 충돌 감지가 프로젝트 핵심 — ROADMAP Phase 4), friction 0.8.
- 트레이드오프: 이 방식은 "비인접 링크 간 self-collision만 감지"가 불가능하다(전부 or
  전무). 필요해지면 로봇별 예약 그룹(§5 슬롯 4–14) 배정 또는 쌍별 필터 훅으로 확장.
  다중 로봇도 현재는 서로 같은 ROBOT 그룹이라 selfCollision=false면 로봇 간 충돌도
  감지되지 않는다 — 로봇 간 충돌 감지가 요구되면 그룹 분리로 해결(백로그).
- 검증: `npm test` 115개 통과(scene-loader robot 브랜치 — 바디/collider 스펙·registry·
  reset 재-home 어서션 포함), `node scripts/gate-browser.mjs --expect=arm` ALL PASS
  (관절 8개, home joint2=-0.6 적용, 링크 y 최대 0.682(기립 — 축 변환 정상), setJoint로
  링크 바디 x/z 변위 > 0.02), `--expect=falling-boxes` 회귀 ALL PASS.
- 영향받는 파일/문서: src/core/scene-loader.ts, src/core/scene-loader.test.ts,
  scripts/gate-browser.mjs, src/main.ts(?scene= 씬 레지스트리·robots 파사드·관절 패널
  마운트), src/ui/inspector/joint-panel.ts

## [2026-07-23] Phase 3 리뷰 수정: kinematic 쌍 ActiveCollisionTypes 정규화 + 로봇 링크 바디 build/reset 상태 동일성

- 상태: 결정됨
- 맥락: Phase 3 리뷰 지적 반영. (1) **[major]** Rapier 0.14의 collider 기본
  ActiveCollisionTypes(DEFAULT=15)는 DYNAMIC_* 쌍만 활성화한다. 로봇 링크는
  kinematicPosition, 바닥은 fixed이므로 KINEMATIC_FIXED(8704) 쌍이 narrow-phase에서
  통째로 건너뛰어져 — 그룹 필터·emitEvents가 다 맞아도 — CLAUDE.md §5 핵심 쌍
  ROBOT×ENV의 접촉 이벤트가 절대 발행되지 않았다(Phase 4 게이트 선제 결함).
  selfCollision=true(KINEMATIC_KINEMATIC=52224 필요)도 같은 원인의 죽은 옵션이었다.
  (2) **[major]** 링크 바디가 home 적용 "전" URDF 초기 FK pose로 생성되어 첫 스텝에
  초기→home 스윕(|Δpose|×240Hz의 가짜 kinematic 속도 — 접촉 사물에 비현실적 임펄스)이
  생겼고, reset()은 tick()(다음 스텝 목표 지정)만 호출해 다음 스텝까지 물리 pose가
  리셋 전 상태로 남았다 — fresh load와 reset 후의 "스텝 전 상태"가 달라 결정론적
  재생(SceneHandle.reset 계약, SIMULATION.md §6)이 깨졌다. (3) [minor] paused/idle 중
  UI 관절 변경이 다음 물리 tick까지 시각 로봇에 반영되지 않음(Phase 5 재생 컨트롤에서
  표면화될 잠재 결함). (4) [minor] CLAUDE.md §4/§9·ROADMAP Phase 3이 축 변환 지점을
  scene-loader로 표기(코드·frozen 계약은 render URDF 래퍼) — 문서 표류.
- 선택: (1) RapierWorld.createCollider가 부모 바디가 kinematic이면
  `DEFAULT|KINEMATIC_FIXED|KINEMATIC_KINEMATIC`을 설정한다 — 이 프로젝트에서 "어떤
  쌍이 상호작용하는가"의 유일한 결정권은 충돌 그룹(ColliderSpec.collidesWith)이고,
  엔진별 쌍-타입 기본값은 PhysicsWorld 래퍼가 정규화한다(쌍 판정은 두 collider 타입의
  합집합 — 한쪽만 켜도 활성). selfCollision=false는 여전히 그룹 필터로 걸러진다
  (kinematic-kinematic 쌍이 켜져도 ROBOT 미포함 필터가 거름 — 회귀 테스트로 확인).
  (2) buildRobot이 RobotBinding을 먼저 만들어(공유 linkBodies Map, 생성자에서 home
  적용) tick() 1회로 FK 뷰를 home으로 갱신한 뒤 "home FK pose"에서 링크 바디를
  생성한다 — 첫 스텝 스윕 0. reset()은 applyHome() 후 새 API
  `RobotBinding.teleportLinksToFk()`(world.teleport — 리셋 전용 계약)로 링크 바디를
  home FK로 즉시 정렬하고 tick()으로 다음 스텝 목표를 재지정한다. 결과: fresh load와
  reset 직후의 스텝 전 물리 상태가 완전 동일(실 Rapier 회귀 테스트로 toStrictEqual 검증).
  (3) main.ts 글루의 파사드 setJoint/applyHome이 engine.state !== 'playing'일 때
  binding.tick()으로 시각 FK를 즉시 갱신 — tick은 시각 갱신 + 다음 스텝 목표 지정뿐이라
  preStep 밖 호출이 무해(다음 preStep이 동일 목표를 재-push, 결정론 영향 없음).
  (4) CLAUDE.md §4/§9·ROADMAP Phase 3을 "축 변환은 render URDF 래퍼(axisFix) 한 곳"으로
  정정(§10 규칙 — 코드가 아닌 문서를 결정에 맞춤). main.ts 파사드 주석도
  sceneHandle.robots를 통한 RobotRegistry 노출이 의도된 것임을 명시하도록 정정
  (표면 축소 대신 — 게이트/향후 UI가 sceneHandle 공개 API를 쓰기 때문).
- 트레이드오프: kinematic collider의 쌍 타입 상시 활성화는 넓은 활성 범위지만 실제
  쌍 생성은 그룹 필터가 게이트하므로 비용 증가는 broad-phase 후보 판정뿐(MVP 규모
  무시 가능). 링크 바디를 home에서 생성하므로 "URDF 초기 자세의 물리 바디"는 어떤
  시점에도 존재하지 않는다(소비처 없음 확인). arm-and-boxes에서 ROBOT×ENV가 이제
  실제 감지되므로 home 자세가 바닥에 닿는 씬은 부팅 직후 start 이벤트가 로그에 남을
  수 있다 — 정상 동작(감지가 프로젝트 목적).
- 검증: src/core/world-kinematic-contact.test.ts(kinematic×fixed start/stop,
  kinematic×kinematic self start, selfCollision=false 필터 무이벤트),
  src/core/scene-robot-reset.test.ts(build 직후 home FK pose, reset 직후 스텝 전
  pose = fresh load와 toStrictEqual), scene-loader.test.ts 로봇 브랜치 갱신(바디 생성
  pose = home FK = kinematic 목표, reset의 teleport 호출 검증). 전체 게이트:
  typecheck + lint + 전체 테스트 + build + gate-browser --expect=arm ALL PASS.
- 영향받는 파일/문서: src/core/world.ts, src/core/robots.ts, src/core/scene-loader.ts,
  src/main.ts, CLAUDE.md §4/§9, docs/ROADMAP.md Phase 3, src/core/scene-loader.test.ts,
  src/core/world-kinematic-contact.test.ts(신규), src/core/scene-robot-reset.test.ts(신규)

<!-- 이후 결정은 여기에 추가 -->
