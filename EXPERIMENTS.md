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

## [2026-07-23] RobotSpec.gripper 스키마 확장 (Phase 5 선행)

- 상태: 결정됨
- 맥락: `gripper` 제어 step(DATA_MODEL §6)이 어떤 관절을 구동할지 로봇 스펙에 정의 필요.
  SceneContext(PLANNER §2)도 gripper 정보를 요구하나 RobotSpec에 대응 필드가 없었음
- 선택: `RobotSpec.gripper?: { joints: string[]; open: number; close: number }` 추가.
  값은 모든 gripper.joints에 동일 적용(평행 그리퍼), 0..1 상태는 close↔open 선형 보간
- 근거: 특정 로봇 동작을 엔진에 하드코딩하지 않는다(§2.6) — 그리퍼 구성도 데이터.
  문서 우선 원칙(§10)에 따라 DATA_MODEL.md §4.1 먼저 갱신 후 schema 반영
- 트레이드오프: 비대칭 그리퍼(관절별 다른 값)는 미지원 — 필요 시 per-joint 매핑으로 확장
- 영향받는 파일/문서: docs/DATA_MODEL.md §4.1, src/schema/types.ts, src/schema/validate.ts,
  src/assets/scenes/arm-and-boxes.scene.json, public/assets/robots/arm6/arm6.urdf(finger 관절)

## [2026-07-23] Phase 4+5 통합: 충돌 이벤트 + 시퀀스 재생을 앱/UI에 배선

- 상태: 결정됨
- 맥락: CollisionMonitor(core/collision.ts)·ControlPlayer(core/control)를 main 글루 +
  하단 독 UI(Timeline/Collision Log/Console) + 재생 바로 통합. 샘플 시퀀스
  arm-touch-box으로 ROADMAP Phase 4/5 게이트를 브라우저에서 검증.
- 선택/근거:
  - **무자동재생 (human-in-the-loop, 플래너 이전부터)**: 씬 레지스트리에 시퀀스가
    선언돼 있어도 자동 재생하지 않는다 — 검증 통과 시퀀스는 "로드 가능" 상태로만 두고
    사용자가 ▶ Play를 눌러야 player.load()가 일어난다(불변식 §2.9의 원칙을 선적용).
    물리 루프 자체는 부팅 즉시 시작한다(falling-boxes 게이트 계약 유지 — 낙하 등 씬
    자체 물리는 재생 컨트롤과 무관).
  - **Stop = 결정론적 재생 준비**: engine.stop() → sceneHandle.reset() →
    player.reset() → monitor.clear() → 충돌 로그 DOM 클리어 순. monitor.clear()로
    이전 CollisionMark가 전부 무효화되지만 player.reset()이 활성 step 런타임을
    폐기하므로 stale mark 소비자는 남지 않는다.
  - **시퀀스 데이터 설계 — 배리어 mark 시맨틱과의 정합**: waitForCollision은 "step
    활성화 시점 mark 이후"의 이벤트만 본다(collision.ts 계약). 접근 동작이 접촉을
    만들며 끝나면 그 start는 mark보다 앞서 배리어가 영원히 못 본다(timeout 경로).
    해결: moveJoints는 박스 위 1.5cm 호버까지만 접근하고, 즉시형 setJoints가 2.5cm
    하강을 지시 → player의 same-tick 연쇄(setJoints 완료 → dt=0로 waitForCollision
    init/mark)가 같은 tick의 world.step "이전"에 mark를 발급하므로, 그 tick의 하강
    스윕이 만드는 start 이벤트를 다음 tick에 정확히 잡는다(경고 없는 이벤트 해제 —
    게이트에서 Play→done 5.97s로 확인, timeout 경로면 ≈11.9s).
  - **도달 자세는 FK로 계산·실측 검증**: arm6.urdf 평면 기하(어깨 y=0.13, 상완 0.28,
    전완+링크4 0.28, 손목→핑거팁 0.15)로 IK 계산 — box_a(0.35, 0.03, 0.15) 위:
    joint1 = −atan2(0.15, 0.35) = **−0.405**(부호는 axisFix Z-up→Y-up 변환의 yaw 방향
    실측으로 확정 — +0.405는 box_b 쪽으로 감), 호버 (0.639, 1.414, 1.089), 하강
    (0.683, 1.442, 1.017). Playwright 실측: 핑거팁 y=0.05 예측 대비 오차 ≈1mm,
    arm×box_a start 이벤트 확인 후 시퀀스에 채택. scene home에 joint1 추가(시퀀스
    joint 참조 검증용), controller를 'sequence'로 변경.
  - **gripper 매핑 시맨틱**: RobotApi.gripperConfig는 글루가 SceneSpec의
    RobotSpec.gripper를 그대로 돌려준다(core는 스키마 엔티티 조회를 소유하지 않음).
    gripper.joints의 논리명 해석(jointMap)은 RobotBinding.resolveJoint가 이미
    수행하므로 어댑터에서 중복 해석하지 않는다.
  - **어댑터 공유 모듈(core/control/adapters.ts)**: RobotApi.readJoints(robot, names)
    ↔ RobotBinding.readJoints(names)의 시그니처 갭, CollisionQuery의 opaque mark ↔
    branded CollisionMark 복원을 한 곳에서 닫는다 — main과 e2e 테스트가 같은 배선을
    쓴다(이중 구현 방지).
  - **로봇 하이라이트 노드 수집**: RobotHandle은 three 노드를 노출하지 않으므로(경계
    계약) 글루가 loadRobot 전후 씬 자식 diff로 캡처하고, 비로봇 엔티티는
    SceneHandle.visualNodes(신규, 읽기 전용)로 노출한다. 펄스(render/highlight.ts)는
    emissive 스냅샷→복원 방식의 순수 시각 효과(불변식 §2.1 예외).
  - **Timeline 마커 클릭은 no-op**: 마커에서 재실행(재생 위치 이동)은 Phase 10
    "노드/타임라인 마커에서 재실행" 몫 — TODO 주석으로 명시.
  - (Phase 4/5 코어 에이전트 결정의 기록 대행) CollisionMonitor는 배열 인덱스가 아닌
    **전역 단조 증가 시퀀스 커서**로 mark를 발급하고 clear()가 이력+커서를 함께
    리셋한다(이전 mark 전부 무효). 이력 초과는 O(n) shift 드롭-oldest(상한 1000,
    측정 후에만 링 버퍼 최적화). player는 same-tick 연쇄 진행을
    SAME_TICK_STEP_LIMIT(64)로 제한하고 연쇄 step은 dt=0을 받는다(시간 이중 계상
    방지). timeout류 경계는 elapsed >= 기준으로 통일. goto는 인덱스별 fire 카운터로
    times 제한을 구현한다.
- **CCD 실측 (ROADMAP Phase 4 게이트 항목 — 측정된 한계)**: Rapier 0.14에서
  kinematicPosition 바디의 한 tick 대변위 스윕(1.5m/tick)은 얇은 벽(0.005m) 통과를
  **감지하지 못한다** — 벽 collider ccd:true, 링크 바디/collider ccd:true, 양쪽 모두
  켜도 이벤트 0건(조용한 터널링). 끝 pose가 겹칠 때만 discrete narrow-phase가 잡는다.
  가짜 통과 처리 대신 실측 동작을 e2e-sequence.test.ts에 회귀로 고정했다(Rapier가
  향후 kinematic CCD를 지원하면 테스트가 깨져 알려줌). **회피책**: 감지용 벽/
  SENSOR_ZONE 두께 ≥ 최대 tick당 변위(240Hz 기준 링크 속도 × 1/240)로 설계.
- 트레이드오프: 시퀀스의 "호버 + 즉시 하강" 2단 구성은 mark 시맨틱에 대한 데이터
  수준 우회다 — 접촉을 만드는 접근 동작과 배리어를 한 step으로 쓰고 싶다면
  waitForCollision에 "직전 N초 이력 포함" 같은 옵션이 필요하다(스키마 확장 백로그).
  로봇 시각 노드 diff 캡처는 loadRobot의 순차 await 계약에 기댄다(SceneLoader.build가
  병렬화되면 재검토).
- 검증: vitest 195개 전부 통과(신규: e2e-sequence.test.ts 3개 — 실 Rapier + 가짜 FK
  로봇 + 실 ControlPlayer/CollisionMonitor 통합, 배리어가 접촉 start 다음 tick에 해제
  (경고 0)·전체 리셋 후 재실행의 충돌 timeSec/setJoints 로그 완전 동일·CCD 실측 2건).
  gate-browser --expect=arm-sequence ALL PASS(무자동재생, Play 재생, arm×box_a start
  이력, waitForCollision 통과, 그리퍼 열림→닫힘, Play 후 5.97s done(이벤트 해제 경로),
  충돌 로그 DOM 행), --expect=arm / --expect=falling-boxes 회귀 ALL PASS.
- 영향받는 파일/문서: src/main.ts, src/core/scene-loader.ts(SceneHandle.visualNodes),
  src/core/control/adapters.ts(신규), src/core/e2e-sequence.test.ts(신규),
  src/ui/dock/{dock,collision-log,timeline,console-panel}.ts(신규),
  src/ui/command-bar/playback.ts(신규), src/render/highlight.ts(신규),
  src/assets/sequences/arm-touch-box.sequence.json(신규),
  src/assets/scenes/arm-and-boxes.scene.json(home.joint1, controller),
  src/core/scene-loader.test.ts(씬 데이터 미러 어서션 갱신), scripts/gate-browser.mjs

## 2026-07-23 — 리뷰 후속 수정 (Phase 4/5 마이너 결함 6건)

- 배경: physics/determinism/schema 리뷰어가 보고한 마이너 결함 8건(중복 2건 제외 6건)을
  코드로 재검증 — 전부 실재. 결정론/불변식 위반은 없고 상태 표면·검증 사각지대·하네스
  견고성 이슈였다. 전건 수정.
- **⏹ Stop 후 재생 바 라벨 (main.ts)**: Stop은 player.reset()을 호출하고 reset은
  커서를 되감으며 status를 'running'으로 두는 계약(ControlPlayer.reset)이라, 엔진
  idle인데 바가 "시퀀스 running"을 표시했다. PlayerStatus에 'ready'를 추가하는 안은
  reset 계약·게이트·테스트 파급이 커서 기각 — 표시 계층(main.ts onTick)에서 엔진
  idle이면 armed 여부와 무관하게 '대기 (▶ Play)'를 보이도록 매핑했다(진실은 그대로,
  뷰만 정정). idle 중 ⏭ Step의 시퀀스 단일 스텝 진행은 의도된 디버깅 동작으로 유지.
- **Space 토글 stale 상태 (playback.ts)**: lastEngineState가 rAF당 1회만 갱신되어 한
  프레임 안의 연속 Space가 토글 대신 play/pause를 반복 호출했다. 키 입력 시 낙관적
  로컬 갱신(pause→'paused', play→'playing')으로 수정 — 다음 update()가 엔진 진실로
  되맞추므로 어긋나도 1프레임 이내 자기 교정.
- **goto times 검증 강화 (validate.ts)**: 임의 유한수 허용 → 0 이상 정수만 허용.
  음수는 'fired < times'로 영구 no-op, 소수는 ceil처럼 동작해 "N회 반복" 의도와
  조용히 어긋났다. times=0(점프 안 함)은 유효로 유지.
- **gripper state 숫자 범위 검증 (validate.ts)**: DATA_MODEL §6의 0..1 계약을
  스키마가 강제하지 않아 state=42가 통과 후 런타임 clamp01이 조용히 1.0으로
  삼켰다. union에 superRefine을 얹어 0..1 밖 숫자를 한국어 메시지로 거부
  (zod union 기본 오류에 묻히지 않게 custom issue 사용).
- **gripper.joints 존재 검증 (scene-loader.ts)**: validateSequence는 URDF 관절을 못
  보고 checkJointNames는 gripper step을 다루지 않는 사각지대 — 씬이 URDF에 없는
  gripper 관절을 선언하면 재생 중 resolveJoint 예외가 Engine rAF 루프를 통째로
  죽였다(다음 requestAnimationFrame 미도달). URDF 관절 목록과 스키마 gripper 설정을
  모두 아는 최초 지점인 buildRobot에서 binding.resolveJoint로 검증해 로드 시점의
  한국어 오류(부트스트랩 오버레이)로 앞당겼다. 논리명(jointMap)도 resolveJoint가
  해석하므로 허용. player.step try/catch 대안은 오류 시맨틱 전반을 바꾸는 큰 변경이라
  기각(필요해지면 별도 결정).
- **arm-sequence 게이트 견고성 (gate-browser.mjs)**: player 파사드 부재 시 후속
  evaluate가 TypeError로 거부되어 하네스가 exit 2로 죽고 PASS/FAIL 표를 잃었다 —
  파사드 체크 후 상호작용 어서션 블록(2~8)을 가드하고 부재 시 FAIL 1건으로 기록
  후 건너뛴다(종료 코드는 여전히 비-0 — false-pass 불가).
- 검증: 신규 회귀 테스트 — validate.test.ts(goto times 음수/소수/0, gripper state
  범위 밖/경계값), scene-loader.test.ts(gripper.joints 부재 시 build 실패+자원 정리,
  원명/논리명 허용). typecheck/lint/vitest/build 및 gate-browser
  --expect=arm-sequence/arm/falling-boxes 전부 통과 확인 후 완료 보고.
- 영향 파일: src/main.ts, src/ui/command-bar/playback.ts, src/schema/validate.ts,
  src/core/scene-loader.ts, scripts/gate-browser.mjs,
  src/schema/validate.test.ts, src/core/scene-loader.test.ts

## [2026-07-23] Phase 6: 씬 관리 UI — 런타임 씬 전환 라이프사이클 + 커맨드바 통합

- **결정: 씬 전환은 "전체 클린 빌드"** — 프리셋 select/📂 업로드로 씬을 바꿀 때
  월드·sync·monitor·player·엔진·독 패널을 전부 새로 만들고(이전 것은 dispose),
  렌더러(three 캔버스)와 상단 커맨드바 셸·JSON 뷰어만 앱 수명으로 유지한다.
  대안(단일 월드 재사용 + clear)은 timestepHz가 씬마다 다를 수 있어 기각 —
  Engine/RapierWorld의 고정 dt는 생성 시 결정되므로(§2.3) 씬마다 fresh 인스턴스가
  결정론적으로 안전하다. main.ts의 부트 3–7단계를 buildScene()으로 추출, loadScene()이
  [검증 → 이전 씬 dispose → 빌드]를 수행한다. 검증은 teardown보다 먼저 — 무효 씬
  때문에 잘 돌던 씬을 잃지 않는다. 전환 후 window.__sim은 항상 새 핸들(게이트 계약).
- **Engine.halt() 추가 (core/engine.ts)**: stop()은 시계만 리셋하고 rAF 루프는 계속
  돈다(idle 렌더) — 씬 전환 시 이전 엔진 루프가 남으면 이중 draw/onTick 누수.
  halt()는 running=false로 루프를 완전 종료한다. halted 엔진은 재사용하지 않는다.
- **Renderer.applySceneOptions() 추가 (render/renderer.ts)**: 캔버스/OrbitControls를
  유지한 채 씬별 skyColor·카메라 position/target/fov만 재적용. 미지정 옵션은 생성자
  기본값으로 되돌린다(이전 씬 설정 누수 방지). 기본값 상수를 생성자와 공용화.
- **커맨드바 셸 (ui/command-bar/scene-controls.ts)**: [좌: 타이틀·씬 select·📂·💾 |
  중앙: 재생 컨트롤 | 우: {} JSON] 하나의 고정 상단 바(UX_DESIGN §3.1). playback.ts는
  자체 fixed 오버레이에서 셸 중앙 슬롯의 플렉스 행으로 변경, 씬마다 재마운트해
  속도 select 등 뷰 상태가 씬을 가로질러 새지 않는다. joint-panel은 바 아래(44px)로.
- **업로드 형식 규약**: (a) SceneSpec 단독 (b) `{ scene, sequence }` 봉투 — 'scene'
  키 존재로 판별(SceneSpec에 'scene' 필드가 없어 모호하지 않음). 파싱은
  scene-controls, 봉투 해석·검증은 main. 실패 시 Console 탭 appLog + 인라인 토스트
  (한국어), 활성 씬은 유지. 💾 저장은 현재 SceneSpec을 `<name>.scene.json`으로 Blob
  다운로드(시퀀스는 포함하지 않음 — 씬 파일 왕복 규약 유지).
- **{} JSON 뷰어 (ui/command-bar/json-viewer.ts)**: 우측 슬라이드 읽기 전용 패널,
  pretty-print + 복사(클립보드). 열 때마다 + 씬 전환 시 refresh()로 갱신(그래프
  편집이 생기는 Phase 8에서 구독 훅으로 확장). 시퀀스 없으면 '시퀀스 없음' 빈 상태.
- **URL 동기화**: 프리셋 전환은 ?scene=을 replaceState로 갱신(딥링크 공유), 업로드
  씬은 파라미터 제거. 부트 시 ?scene= 해석·오류 동작은 기존과 동일(게이트 계약).
- 검증: typecheck/lint/vitest(237) 통과 + vite build + gate-browser
  --expect=arm-sequence/arm/falling-boxes 전 항목 PASS (아래 게이트 실행 기록).
- 영향 파일: src/main.ts, src/ui/command-bar/scene-controls.ts(신규),
  src/ui/command-bar/json-viewer.ts(신규), src/ui/command-bar/playback.ts,
  src/ui/inspector/joint-panel.ts, src/core/engine.ts, src/render/renderer.ts

## 2026-07-23 — Phase 6 샘플 씬 3종 (데이터 전용 저작 — 코드 변경 없음)

- 배경: ROADMAP Phase 6 / PRD §7 "샘플 씬 3종(픽앤플레이스·장애물 회피·충돌 테스트베드)".
  new-scene.md 절차대로 **순수 데이터**(SceneSpec/ControlSequence JSON)만으로 저작했다 —
  엔진/로더/스키마 코드 변경 0줄 (불변식 §2.5/§2.6 실증). main.ts는 레지스트리 항목
  추가만, gate-browser.mjs는 씬별 어서션 블록 추가만.
- **픽앤플레이스는 "잡아 들기"가 아니라 "밀기(push/drag)"로 정직하게 설계했다**:
  MVP의 kinematic 링크 + Rapier 접촉으로는 파지-리프트가 물리적으로 신뢰 불가
  (PRD §6 non-goal, MuJoCo 경로 몫). 대신 kinematic 링크가 dynamic 바디를 미는
  신뢰 가능한 상호작용만 사용한다. 실측으로 확정한 세부 설계:
  - **완전 close는 금물**: arm6 그리퍼의 닫힘 간격은 4mm라 5cm 카고를 close하면
    쥐어짜서 튕겨낸다(1차 실측: close 후 스윕 0.4s 만에 contact stop — 카고 유실).
    gripper state **0.7**(간격 ≈46mm)로 5cm 카고를 느슨한 케이지로 감싼다.
  - **joint6(손목 롤)=π/2로 손가락을 진행 방향 앞/뒤에 배치**: 손가락 오프셋 축은
    joint6=0에서 반경 방향(실측 확인) — π/2 롤로 접선(스윕) 방향으로 돌려 뒤쪽
    손가락이 카고 뒷면을 **양의 접촉으로 밀게** 한다(마찰 의존 없음).
  - **배리어 해제는 검증된 hover+즉시 nudge 패턴**: 스트래들 하강(무접촉) 후
    setJoints로 joint1 +0.03(접선 ~11mm) — 다음 tick에 뒤 손가락이 카고를 눌러
    waitForCollision(mark 이후) 이벤트로 해제된다. 게이트가 이벤트 경로를 시간
    상한(done<10s)으로 강제한다(timeout 경로 ≈14.1s와 구분).
  - IK 산수 실수 교훈: 손끝 목표 y를 0.15m 높게 계산해 1차 시도에서 접촉 0건 —
    window.__sim.robots 파사드로 링크 pose를 실측(probe)해 교정했다. FK 체인 실측:
    link5 원점 = 어깨(0.13) + 0.28·u(φ2) + 0.28·u(φ3), 손끝 = link5 + 0.15·u(φ5).
- **SENSOR_ZONE 페어링은 양쪽 필터 규칙**: Rapier 쌍 필터는 양방향이다 — 센서
  collider가 collidesWith:[OBJECT]를 선언해도, **OBJECT 쪽 collider가 SENSOR_ZONE을
  나열하지 않으면 이벤트가 발생하지 않는다**. pick-and-place의 cargo와 testbed의
  slider만 SENSOR_ZONE을 필터에 추가했다(다른 OBJECT는 센서와 무관 — 불필요한
  narrow-phase 페어 방지). phase6-scenes.test.ts가 양쪽 모두를 데이터 수준에서 고정.
- 장애물 회피: 검증된 arm-touch-box 기하를 재사용(target_box=[0.35,0.03,0.15],
  hover θ2=0.639 → nudge θ2=0.683). 회피 경로는 "접힌(홈) 자세의 최대 반경
  ≈0.17m ≪ 기둥 반경 0.40m" 성질을 이용 — 측면 A 하강 → 홈 자세로 리프트 →
  joint1 회전(기둥 위/안쪽 통과) → 측면 B 하강. pillar는 emitEvents:true라
  "닿았다면 반드시 이력에 남는다" — 게이트의 무접촉 어서션이 공허하지 않다.
- 충돌 테스트베드(로봇 없음): 반발 0.8 공 바운스(ENV 바닥과 combine avg → 유효
  0.4), 마찰 0.05 경사 미끄럼 → SENSOR_ZONE 게이트 통과(비행 중 감지), 마찰 0.8
  경사 구름 공이 3단 스택을 전도. **스택 전도 실측 교훈 2건**: (1) 굴러온 공이
  바닥 경유로 도달하면 램프 립 착지(반발 0, 마찰 0.8)가 수평 속도를 흡수해 스택이
  버틴다 — 스택을 램프 출구 비행경로 안(x=−0.68)으로 옮겨 ≈1.3m/s로 직격시켰다.
  (2) EntitySpec에 초기 속도 필드가 없어 "굴러 들어오는" 연출은 경사로 데이터로만
  표현 가능(스키마 확장 백로그 후보).
- 검증: vitest 237개 전부 통과(신규 phase6-scenes.test.ts 12개 — 씬/시퀀스 검증 +
  그룹/센서 페어링/emitEvents 데이터 고정). 브라우저 게이트 6종 ALL PASS —
  pick-and-place(arm×cargo start, cargo×drop_zone sensor start@7.36s, done 8.22s),
  obstacle-avoidance(arm×pillar 0건, arm×target_box start@9.96s, done 9.35s),
  collision-testbed(접촉 쌍 12종/센서@0.67s/전 바디 y>0 — 스택 3개 전도 후 y≈0.029),
  회귀 arm-sequence/arm/falling-boxes. 신규 게이트는 2회 연속 실행으로 결정론 확인
  (동일 PASS 표). 정적 검사: tsc/eslint 경고 0, vite build 성공.
- 영향 파일: src/assets/scenes/{pick-and-place,obstacle-avoidance,collision-testbed}
  .scene.json(신규), src/assets/sequences/{pick-and-place,obstacle-avoidance}
  .sequence.json(신규), src/main.ts(레지스트리 항목만), scripts/gate-browser.mjs
  (씬별 어서션 3블록 + 공용 playAndAwaitDone), src/core/phase6-scenes.test.ts(신규)

## 2026-07-23 — Phase 6 통합: 씬 전환 teardown 순서 확정 + 인스펙터 배선

- **씬 전환 teardown 순서(ActiveScene.dispose)를 다음으로 고정한다** — 순서가 계약이다:
  1. `engine.halt()` — rAF 루프 **완전 종료**. `stop()`은 시계만 리셋하고 루프는 계속
     돌므로, halt 없이는 이전 씬 엔진이 곧 해제될 world를 계속 step/draw한다
     (Rapier WASM use-after-free 크래시 + 이중 draw 경로). 반드시 가장 먼저.
  2. 구독 해제(`offTick`/`offStepChange`/`offMonitor`) — 곧 제거될 UI로의 콜백 유입
     차단. 인스펙터 refresh는 엔진 tick에서만 구동되므로 1–2단계 이후에는 해제된
     world를 읽는 stale 콜백이 구조적으로 불가능하다.
  3. UI 제거(playbackBar → inspector → jointPanel → rightStack → 독 패널들 → dock)
     — 물리/모니터 참조를 가진 뷰를 물리 해제보다 먼저 걷는다.
  4. `sceneHandle.dispose()` — 물리 바디·시각 노드·로봇 핸들(URDF 씬 그래프) 해제.
  5. `sync.clear()` — 물리↔시각 바인딩 매핑 정리(바디가 사라진 뒤 남은 참조 제거).
  6. `world.free()` — Rapier WASM 메모리 반환. 모든 소비자가 사라진 **마지막**에.
  - `window.__sim` 해제는 "여전히 이 엔진을 가리킬 때만" 수행 — 새 씬이 이미
    재할당했으면 건드리지 않는다(전환 경쟁 시 새 핸들 오염 방지). 렌더러(three
    캔버스)·커맨드바 셸·JSON 뷰어는 앱 수명으로 유지, 씬별 카메라/환경 옵션만
    `Renderer.applySceneOptions`로 재적용.
- **런타임 전환 게이트 신설**: `gate-browser.mjs --expect=scene-switch` —
  arm-and-boxes 부트 → **UI select(change 이벤트) 경유** collision-testbed 전환 →
  spec.name 갱신·엔티티 수(spec.entities+ground) 일치·robots 파사드 빈 목록(누수
  검출)·sim 전진 → arm-and-boxes 복귀(URDF 재로드 경로) → 동일 검증 + select 표시
  동기 + 페이지 에러 0건. 전환 중 __sim이 잠시 undefined인 것은 정상(해제→재빌드)
  이라 폴링으로 흡수한다.
- **인스펙터(MVP) 배선** (ui/inspector/inspector.ts — 읽기 전용, 물리 불변):
  - deps는 core 파사드 위 글루 구현: 목록=spec.entities(예약 `__ground` 자연 제외),
    pose=`world.bodiesOfEntity(id)[0]`→`getPose`(물리가 진실, §2.1), 관절=robots
    레지스트리(로봇 엔티티만, limits는 URDF ∩ override 유효값).
  - 갱신 주기(주기 결정권은 통합자): playing 중 engine.onTick에서 **150ms 스로틀**,
    playing→paused/idle **전이 시 1회**, 선택 변경 시 1회(onSelect), stepOnce 후 1회.
  - onSelect → `render/highlight.pulseEntity`로 해당 시각 노드 붉은 펄스. inspector의
    선택 변경 가드 덕에 onSelect 안 refresh 재호출이 루프를 만들지 않는다.
  - 표시 규약: 오일러(deg, XYZ)는 **표시 전용** 변환이고 원본 쿼터니언은 툴팁 노출
    (내부 진실은 쿼터니언, CLAUDE.md §4). getJoints의 valueRad 필드는 prismatic이면
    m — 컬럼 라벨은 일반형 '값'.
  - 배치: 우측 **패널 스택**(fixed, top 44/right 12/폭 280) — 관절 패널(위, 쓰기
    경로) + 인스펙터(아래, 읽기 경로), 둘 다 접기 가능. 편입은 각 핸들의 `el`로
    절대 배치를 스택 흐름(static)으로 전환(모듈 기본값은 standalone 배치 유지).
    z-index 94 — 독/커맨드바(90) 위, {} JSON 슬라이드 패널(95) 아래(뷰어가 열리면
    스택을 덮어 읽기 가능). joint-panel.ts에는 접기 헤더와 핸들 `el` 노출만 추가
    (Phase 7 Scene Builder 인스펙터로 대체 예정인 임시 UI 지위는 그대로).
- 검증: tsc/eslint 경고 0, vitest 237개 전부 통과, vite build 성공. 브라우저 게이트
  7종 ALL PASS — falling-boxes/arm/arm-sequence(회귀), pick-and-place(**2회 연속**
  동일 PASS 표 — done 8.25s/8.18s, 결정론), obstacle-avoidance, collision-testbed,
  scene-switch(신규).
- 영향 파일: src/main.ts(인스펙터 배선 + 우측 스택), src/ui/inspector/joint-panel.ts
  (접기 헤더·el 노출), scripts/gate-browser.mjs(scene-switch 모드), EXPERIMENTS.md

## [2026-07-28] 로봇↔로봇 충돌 결함 수정: 그룹 필터 ≠ 자기 충돌

- 상태: 결정됨 (supersedes Phase 3의 "selfCollision=false → 필터에서 ROBOT 제외")
- 맥락: 사용자가 로봇팔 두 대를 닿게 배치했으나 충돌 이벤트가 전혀 발행되지 않고
  그대로 겹쳐 지나갔다
- 원인: scene-loader가 `selfCollision=false`(기본)일 때 링크 collider의 collidesWith에서
  ROBOT 그룹을 통째로 제외했다. 모든 로봇 링크는 같은 ROBOT 그룹이므로, 필터에 ROBOT이
  없으면 **자기 링크뿐 아니라 다른 로봇과도** narrow-phase에 도달하지 못한다.
  "self-collision"과 "robot-robot collision"을 그룹 하나로 뭉뚱그린 설계 오류
- 선택: 두 개념을 분리한다.
  (1) 링크 collider는 **항상** ROBOT을 필터에 포함 → 다른 로봇과 충돌한다.
  (2) 자기 링크 억제는 **엔티티 단위**로 처리 — 로봇 한 대의 모든 링크는 하나의
      EntityId를 공유하므로 "같은 엔티티 접촉(a===b)" = self-collision이다.
      `PhysicsWorld.setSelfContactEnabled(entityId, enabled)` 추가, 기본 억제.
- 근거: Rapier 그룹 비트마스크는 16개뿐이고 **로봇 개체를 구분하지 못한다**. 예약 슬롯
  (4–14)을 로봇마다 배정하는 대안은 로봇 수 상한(~11)이 생기고, Phase 7 런타임 씬 편집으로
  로봇이 추가될 때마다 모든 로봇의 필터를 재계산해야 해 취약하다. 엔티티 id 기반 판정은
  개수 제한이 없고 핸들↔엔티티 매핑(§2.4 "유일한 진실")을 그대로 재사용한다.
- 트레이드오프: 인접 링크 쌍이 narrow-phase까지는 도달해 접촉 계산 비용이 남는다(이벤트만
  억제). 링크 8개 규모에서는 무시할 수준이며, 필요해지면 인접 쌍 필터를 별도 도입한다.
  `PhysicsWorld` 계약에 메서드 1개 추가(테스트 목 2곳 갱신).
- 측정치: 신규 게이트 `--expect=two-arms` — arm_left×arm_right start 이벤트 t≈6.7s 감지,
  접촉점 [0.190, 0.300, -0.015] 보강, 자기 링크 접촉 0건. 단위 테스트 7종 추가.
  기존 게이트 12종 전부 회귀 없음(특히 obstacle-avoidance의 "arm×pillar 0건" 유지).
- 영향받는 파일/문서: src/core/types.ts, src/core/world.ts, src/core/scene-loader.ts,
  src/core/world-robot-robot-contact.test.ts, src/core/scene-loader.test.ts,
  src/assets/scenes/two-arms-collision.scene.json, scripts/gate-browser.mjs, docs/USAGE.md

## [2026-07-28] 충돌 시각화 강화: 물리 접촉점 + 3D 마커

- 상태: 결정됨
- 맥락: 충돌이 감지돼도 "어디서" 부딪혔는지 알 수 없었다(엔티티 전체 펄스만 존재).
  로봇 링크처럼 큰 물체끼리는 전체 펄스가 위치 정보를 주지 못한다
- 선택: (1) `RapierWorld.step()`이 start 접촉에 한해 `world.contactPair`로 매니폴드를
  조회해 월드 접촉점·법선을 `ContactEvent.point/normal`에 채운다(DATA_MODEL §7의 선택 필드가
  드디어 채워진다). (2) `render/contact-marker.ts` — 접촉점에 빨강 구 + 퍼지는 링을 띄운다.
  `depthTest: false`로 로봇 내부 접촉도 가려지지 않고, 24개 고정 풀을 재사용해 할당이 없다.
- 근거: sensor는 매니폴드가 없고 stop 시점엔 이미 분리돼 조회 불가 — start·contact에만
  보강한다. 조회 실패해도 감지 자체에는 영향이 없다(부가 정보)
- 트레이드오프: 접촉 시작마다 매니폴드 1회 조회 비용. 대표 접촉점 1개만 쓴다(전체 매니폴드
  시각화는 과함). 마커 감쇠는 물리 시간이 아닌 벽시계 기준 — 일시정지 중에도 사라진다
- 영향받는 파일/문서: src/core/world.ts, src/render/contact-marker.ts, src/main.ts, docs/USAGE.md

## [2026-07-28] 방향키 오브젝트 이동 (기즈모 커밋 경로 재사용)

- 상태: 결정됨
- 맥락: 사용자가 "물체를 옮기는 방법이 없다"고 보고. 기즈모는 정상 동작했으나(선택 API
  검증됨) 클릭→모드 전환→축 드래그를 알아야 해 발견성이 낮았다
- 선택: 방향키(카메라 기준 수평)·PageUp/PageDown(월드 Y)로 선택 오브젝트를 이동한다.
  기본 5cm(스냅 격자와 동일), Shift 병용 시 1cm. **기즈모 드래그와 같은 커밋 경로**
  (`onTransformCommit`)로 발행해 통합자의 물리 teleport 처리가 갈리지 않는다
- 근거: 카메라 기준 이동은 화면에서 보이는 방향과 일치해 직관적이다(월드축 고정은 카메라를
  돌리면 어긋난다). 시각 노드를 직접 옮기지 않고 커밋만 발행 — 물리가 진실이라는 §2.1 유지
- 트레이드오프: 방향키가 브라우저 스크롤을 막는다(선택 상태에서만 preventDefault)
- 영향받는 파일/문서: src/render/interaction.ts, src/render/interaction.test.ts, docs/USAGE.md

<!-- 이후 결정은 여기에 추가 -->

## [2026-07-23] Phase 6: UI 디자인 폴리시 패스 — 디자인 토큰(ui/theme.ts) + 시각 언어 통일

- **결정: 시각 토큰을 `src/ui/theme.ts` 한 곳으로 중앙화** (ROADMAP Phase 6
  "frontend-design 스킬 원칙으로 UI 정리"). 기존에는 각 ui 모듈이 색/보더/폰트를
  인라인으로 하드코딩(#2e5db3 파랑 액센트, #5d6470 muted 등 산발) — 이제 모든
  패널이 COLOR/FONT/SPACE/RADIUS/SHADOW/Z_INDEX/LAYOUT 토큰만 소비한다.
  동작·계층 규칙 불변: core/schema 무수정, ui는 여전히 core를 import하지 않는다
  (theme는 ui 내부 모듈).
- **액센트를 파랑(#2e5db3) → 주황(#e67e22)으로 교체** — 로봇 팔 링크 색과 일관.
  적용점: Play 버튼(accent 변형), 재생 중 펄스 점, 활성 독 탭 밑줄, 타임라인 활성
  마커(주황 배경 + 어두운 텍스트 — 대비 6.4:1), 인스펙터 선택 행, {} JSON 열림
  상태, 관절 슬라이더 accent-color, 포커스 링.
- **hover/active/focus-visible/스크롤바/펄스는 클래스로**: 인라인 스타일로는 상태
  셀렉터가 불가능 — `ensureThemeStyles()`가 `<style id="rsw-theme-styles">` 1장을
  멱등 주입한다(.ui-btn/.ui-select/.ui-input/.ui-tab/.ui-scroll/.ui-dot/.ui-badge/
  .rsw-entity-row 등). 모듈 top-level에서는 DOM을 만지지 않아 node 단위 테스트가
  ui 모듈을 import해도 안전. 상태 표시는 인라인 색 대신 **클래스 토글**로 바꿔
  hover와 공존시켰다(독 탭 활성, 인스펙터 선택 행, JSON 토글).
- **레이아웃 정리**: 커맨드바 명시 높이 44px(3존 유지), 바 아래 층은 52px 기준
  (RIGHT_STACK_TOP_PX 44→52, 토스트 동일, JSON 패널 top 44). 우측 패널 폭 280
  유지, 패널 radius 8px + 그림자 통일, 접기 셰브론(▾/▴) 규약 유지 + aria-expanded.
  접기는 display 토글 → max-height/height 트랜지션(0.2s)으로 부드럽게(내용 숨김
  의미는 동일).
- **뷰포트 statusline 신설** (`ui/viewport/statusline.ts`, UX_DESIGN §3.3): 좌하단
  오버레이 "씬 이름 · ● 상태 · simTime · step n/N". pointer-events:none(뷰포트
  상호작용 무간섭), 씬 수명과 함께 마운트/해제. 글루(main.ts)가 engine.onTick에서
  update — ui는 POJO만 받는다. 빈 씬(엔티티 0)이면 중앙 안내 문구(UX §7).
- **충돌 로그 배지 규약**: kind==='sensor'면 파랑 계열, 아니면 phase start=빨강
  틴트/stop=회색 틴트. 배지 텍스트(phase)와 kind 셀 텍스트가 의미를 전달하고 색은
  보조(§9 "색만으로 전달 금지"). 빈 상태 안내 추가("아직 충돌 이벤트가 없습니다…").
  타임라인·JSON 뷰어에도 빈 시퀀스 안내 문구.
- **접근성**: muted 텍스트 #5d6470(3.1:1) → #8b93a1(5.8:1)로 상향 — 본문 텍스트
  대비 ≥ 4.5:1. 모든 버튼 aria-label, 토글 aria-pressed/aria-expanded, 슬라이더
  aria-label(관절명), 토스트 role=alert, :focus-visible 링(주황 2px).
- **폰트 정책**: UI 텍스트 system-ui 스택 / 수치·로그·JSON은 ui-monospace 스택
  (simTime·position·관절값·limits·타임라인 리드아웃·콘솔·충돌 시간). 글래스 패널은
  rgba 반투명만 사용 — backdrop-filter(blur)는 GPU 없는 노트북(PRD §7) 배려로 배제.
- 검증: tsc/eslint 경고 0, vitest 237개 전부 통과, vite build 성공. 브라우저 게이트
  7종 ALL PASS — falling-boxes / arm / arm-sequence / pick-and-place /
  obstacle-avoidance / collision-testbed / scene-switch (동작·testid 계약 무변경).
  gate-screenshot.png 육안 확인: 44px 상단 바 3존, 주황 액센트 일관, 우측 스택
  정렬, statusline·빈 시퀀스 안내 표시.
- 영향 파일: src/ui/theme.ts(신설), src/ui/viewport/statusline.ts(신설),
  src/ui/command-bar/{scene-controls,playback,json-viewer}.ts,
  src/ui/dock/{dock,collision-log,console-panel,timeline}.ts,
  src/ui/inspector/{inspector,joint-panel}.ts, src/main.ts(statusline 배선 +
  스택 top 52), index.html(베이스 스타일), EXPERIMENTS.md

## [2026-07-23] Phase 6 완료 — 리뷰 반영(파이널라이저) + README 재작성 + 최종 게이트

- **리뷰 findings 반영** (minor 9건 중 8건 적용, 1건 보류 — 사유는 아래):
  - `renderApi.remove`가 프리미티브/바닥 메시의 geometry/material을 `dispose()`하도록
    `render/meshes.ts`에 `disposeMeshResources` 신설(URDF `RobotHandle.dispose`와 대칭)
    — 씬 전환 반복 시 GPU 버퍼가 GC 대기로 쌓이지 않는다.
  - `buildScene` 조립 가드: SceneLoader.build 이후의 조립(엔진/독/우측 스택/재생 바)을
    try/catch로 감싸고, 부분 조립 실패 시 이미 만들어진 몫만 **기존 teardown 순서
    계약(halt → 구독 해제 → UI 제거 → 씬 자원 → sync → world)** 그대로 되감고
    재던진다. 성공 시 반환되는 `dispose()`와 실패 복구가 같은 함수(`teardownBuilt`)
    를 쓴다 — 해제 경로는 하나. 실패해도 월드/DOM 패널/Space 키 리스너가 새지 않는다.
  - 씬 전환 실패 표시 정합: `SceneSwitchResult.sceneLost`(build 단계 실패 = 이전 씬
    이미 해제됨)를 글루가 전달하고, scene-controls는 이때 select를 죽은 씬 이름으로
    되돌리는 대신 '씬 없음' 옵션을 표시한다(validate 실패는 기존대로 이전 값 복귀).
  - 테마 토큰 드리프트 제거: COLOR에 `errorSurface`/`errorBorder`/`successBorder`
    추가 — scene-controls 오류 토스트와 timeline done 마커의 ad-hoc rgba/hex를 토큰
    소비로 교체(토스트 텍스트 #ffb3b3 → COLOR.errorText #ff8a80, 시각 차이 미미).
  - main.ts 우측 스택/오버레이 상수를 `ui/theme`의 LAYOUT/Z_INDEX 토큰에서 유도
    (52/280/'94'/'9999' 수치 중복 제거 — 테마 변경 시 자동 동기).
  - {} JSON 슬라이드 패널: 닫힘 상태에 `inert` 적용 — aria-hidden 영역으로 Tab
    포커스가 들어가던 WAI-ARIA 위반 수정(트랜스폼 애니메이션은 유지).
  - **보류 1건**: collision-testbed의 동적 collider `collidesWith`에 ROBOT 포함 —
    로봇 없는 씬에서 해당 비트는 매치되지 않아 무해하고, 로봇 씬과의 템플릿 통일
    (복사-붙여넣기 저작)을 유지하는 편이 낫다고 판단. 동작 차이 0.
- **README.md 재작성** (ROADMAP Phase 6 "README 실행법"): 데모 씬 5종 표(내용·게이트
  어서션), 실행법(dev/build/verify/gate), 조작법(카메라·Space·패널), 아키텍처 한 장
  요약(계층 다이어그램 + 불변식 3개 + docs 링크), AI-native 하네스 소개(CLAUDE.md
  헌법·AGENTS.md 리뷰 5역할·EXPERIMENTS.md·검증 게이트), /new-scene 데이터 전용
  씬 추가법, 한계 명시(파지 물리 없음·kinematic 스윕 CCD 한계·안전 인증 부적합·
  IK/센서/저작 UI 미구현 — 과장 없이 현재 상태 그대로).
- **최종 게이트 측정치** (전부 이 커밋 시점 실측):
  - `tsc --noEmit` 통과, eslint 경고 0, vitest **237/237 통과**, `vite build` 성공
    (번들 2,807.82 kB / gzip 960.04 kB — Rapier WASM base64 내장 단일 청크).
  - 브라우저 게이트 7종 **ALL PASS**: falling-boxes(5 바디 정착 y 0.029–0.059) /
    arm(관절 8, home joint2=-0.6, 링크 max y 0.682) / arm-sequence(arm×box_a
    start@5.371s, done 6.02s < 9s 이벤트 해제 경로, 충돌 로그 5행) /
    pick-and-place(arm×cargo start@5.571s, drop_zone sensor@7.483s, done 8.25s) /
    obstacle-avoidance(arm×pillar 0건, arm×target_box@9.942s, done 9.42s) /
    collision-testbed(접촉 쌍 12종, sensor@0.67s, 전 바디 y>0 정착) /
    scene-switch(왕복 전환·로봇 파사드 누수 0·select 동기·페이지 에러 0).
- 영향 파일: src/main.ts(조립 가드·토큰 유도 상수·remove dispose·sceneLost),
  src/render/meshes.ts(disposeMeshResources), src/ui/theme.ts(토큰 3종),
  src/ui/command-bar/scene-controls.ts(씬 없음 상태·토스트 토큰),
  src/ui/command-bar/json-viewer.ts(inert), src/ui/dock/timeline.ts(successBorder),
  README.md(재작성), EXPERIMENTS.md

## [2026-07-23] Phase 7 — 3D 파일 임포트 파이프라인 (mesh-import + import-dialog)

- **범위**: `src/render/mesh-import.ts`(파싱·prepareForScene·MeshAssetStore),
  `src/ui/library/import-dialog.ts`(UX §4.4 다이얼로그), `src/render/mesh-import.test.ts`.
  씬 편입 배선(RenderSceneApi.addMeshAsset / world의 convexHull·trimesh 지원 /
  저장 경고)은 통합자 몫으로 남김 — 아래 "통합 필요" 참조.
- **포맷**: glTF(.glb/.gltf)·STL·OBJ — 확장자 기반 감지, GLTFLoader.parseAsync /
  STLLoader.parse / OBJLoader.parse(text). 단일 파일 전용(외부 .bin/텍스처 참조
  .gltf 미지원 — 오류 안내에 .glb 권장 명시, MTL 미지원 — 재질 기본값).
- **축 변환**: Z-up→Y-up은 URDF 경로(render/urdf.ts axisFix)와 동일한 -π/2 X 회전
  규약. URDF는 축이 고정 규약이라 로더가 항상 변환하지만, 임포트 메시는 원본 축을
  알 수 없어 사용자가 다이얼로그에서 선택한다(UX §4.4). 임포트 경로의 축 변환
  지점은 prepareForScene 한 곳.
- **피벗 규약**: prepareForScene이 피벗을 bbox "바닥 중심"으로 재정렬 — 엔티티
  position y=0이 곧 지면 안착. AABB 전략의 box collider만 중심 오프셋(offset)이
  필요하고 hull/trimesh 정점에는 재정렬이 이미 반영된다.
- **데시메이션**: convex hull 입력 정점을 그리드 해시로 ≤ 2048(MAX_HULL_POINTS)
  보장. AABB 균등 그리드에서 점유 셀당 최초 등장 정점 1개(입력 순서 — 결정론,
  셀 중심 스냅 없음 → hull이 원본을 벗어나지 않음). 분할 수 40에서 시작해 3/4씩
  축소, ⌊∛2048⌋=12에서 12³=1728 ≤ 2048로 종료 보장.
- **trimesh 안전 규칙**(DATA_MODEL §2): 다이얼로그가 Trimesh 전략 선택 시 유형을
  Static으로 강제(Object 버튼 비활성 + 안내 문구). 강제 로직은 순수 함수
  `forcedEntityKind`가 소유하고 EntitySpec 조립(`buildImportedEntitySpec`)도 같은
  함수를 소비 — UI 우회 경로가 없다. trimesh 데이터는 데시메이션 없이 전체
  삼각형(병합 정점+인덱스)을 쓴다(정적 전용 — 형상 충실도 우선).
- **MeshAssetStore 결정**: MeshAssetResolver(core/scene-edit-types) 구현.
  ref = 'asset://<n>' 단조 증가. trimesh 등록 에셋의 getPoints는 (getIndices와
  쌍을 이루도록) trimesh 정점을 돌려준다 — 같은 ref에 convexHull을 써도 전체
  정점 hull로 여전히 정확. getObject는 "원형"을 주며 씬 투입은 clone(true) 필수,
  clone은 geometry/material을 공유하므로 임포트 시각 노드에 disposeMeshResources
  금지(해제는 store.clear() 일괄) — 통합자 주의 사항.
- **한계(세션 한정 에셋)**: asset:// ref는 현 세션 MeshAssetStore에서만 해석된다.
  씬 저장(💾) 시 메시 데이터가 JSON에 내장되지 않으므로 다른 세션에서 불러오면
  해당 엔티티 복원 불가. 저장 경로는 `collectAssetRefs(spec)`가 비어 있지 않으면
  `ASSET_SAVE_WARNING_KO`를 표시해야 한다(통합자 배선). **백로그**: data-URI 내장
  (scene-edit-types serializeRef 방향)으로 승격.
- **통합 필요(다른 소유 파일 — 이번 변경에서 손대지 않음)**:
  1) `RenderSceneApi`(core/scene-loader.ts)에 `addMeshAsset(ref: string): VisualNode`
     추가 + buildVisual의 'mesh' 케이스 라우팅, main.ts 글루가
     `store.getObject(ref)` → `clone(true)` → 씬 추가로 구현.
  2) `core/world.ts` createCollider의 convexHull/trimesh 분기 — MeshAssetResolver
     주입(RAPIER.ColliderDesc.convexHull(points) / trimesh(points, indices)).
  3) 💾 저장 경로에 collectAssetRefs + ASSET_SAVE_WARNING_KO 경고.
- 검증: tsc 통과, eslint 경고 0, vitest 381/381 통과(신규 29건 — 데시메이션 상한·
  bbox/피벗/Z-up 회전 수학·STL/OBJ 파싱·store 계약·폼→EntitySpec 매핑·검증 통과·
  trimesh→Static 강제).
- 영향 파일: src/render/mesh-import.ts(신설), src/ui/library/import-dialog.ts(신설),
  src/render/mesh-import.test.ts(신설), EXPERIMENTS.md

## [2026-07-23] Phase 7 — Scene Builder 통합 (워크스페이스 조립 + 편집 배선 + Undo/Redo)

- **범위**: `src/main.ts`(워크스페이스 조립·SceneEditor/상호작용/편집 폼/임포트 배선),
  `src/ui/history.ts`(+테스트, 신설), `src/ui/workspace.ts` 슬롯 편입, 게이트
  `--expect=scene-builder` 추가. 병렬 랜딩된 Phase 7 모듈(코어 SceneEditor · 워크스페이스
  셸 · ViewportInteraction · entity-editor · mesh-import)의 통합 지점.

### 병렬 워크스트림의 결정 사항 (통합자 대리 기록 — 각 소유 랜인 EXPERIMENTS 미기재분)
- **엔티티 빌드 루틴 추출**: SceneLoader의 엔티티 1개 몫 생성이 exported 함수
  (`buildEntity`/`buildObjectEntity`/`attachRobotPhysics`)로 추출되어 SceneLoader.build와
  SceneEditorImpl이 **같은 코드**를 쓴다 — 엔티티 구성의 단일 진실(로직 드리프트 방지).
- **SceneHandle.builtEntities 라이브 맵**: 빌드 레코드 Map을 SceneHandle과 SceneEditor가
  공유한다 — 편집(add/remove/재빌드) 후에도 reset()/dispose()가 항상 현재 상태를 순회.
  기존 `visualNodes`는 빌드 시점 스냅샷으로 남고(호환), 최신 시각 노드는 builtEntities가
  진실이다 — main의 `visualNodeOf`/픽킹 맵이 이를 소비한다.
- **RobotRegistry.remove(entityId)** 추가(9줄, 미등록 id no-op): 로봇 제거/개명 시
  바인딩을 해제하지 않으면 Engine preStep tickAll()이 삭제된 바디에 setKinematicPose를
  시도해 던진다.
- **RapierWorld 3번째 ctor 인자 `MeshAssetResolver`**(선택): convexHull/trimesh collider의
  asset:// ref 해석 지점. 미주입 시 해당 shape은 한국어 오류(프리미티브 전용 씬은 무영향).
  참고: 현 rapier3d-compat은 ColliderDesc.convexHull이 desc 시점에 null을 돌려주지 않아
  '볼록 껍질 생성 실패' 분기는 mock으로만 테스트된다(경계 자체의 테스트).
- **로봇 rename = 동기 재부착**: renameEntity는 URDF 재로드 없이 기존 RobotHandle에
  attachRobotPhysics로 새 id의 물리를 재부착한다(관절 상태 보존, world 핸들 매핑은
  새 id로 재생성 — PhysicsWorld에 rename 연산이 없다). ControlSequence의 id 참조 갱신은
  범위 밖 — 시퀀스 검증이 로드/재생 시점에 다시 잡는다.
- **라이브러리 템플릿 그룹 배정 편차(의도)**: 동적 오브젝트 collidesWith에 SENSOR_ZONE
  포함(과제 문안엔 없음) — Rapier 쌍 필터는 양방향이라 OBJECT 측에 없으면 라이브러리
  Sensor Zone이 사물을 감지하지 못한다(pick-and-place 샘플과 동일 배정). Plane 템플릿은
  dynamic OBJECT가 아니라 **static ENV**(받침/작업대 용도 — CLAUDE.md §5 정적 환경 규약).
  Sensor Zone의 ROBOT 감지는 로봇 링크 필터(ROBOT_COLLIDES_WITH)에 SENSOR_ZONE이 없어
  아직 발화하지 않는다 — 데이터로는 전방 호환(코어 소유자 백로그).
- **interaction 계층 색상**: render는 ui/theme을 import하지 않으므로 액센트 색은 생성자
  opts 기본값(0xe67e22 — COLOR.accent와 동일 값 의도적 중복)으로 갖는다.

### 통합 결정
- **Undo/Redo = serialize() 스냅샷 + 전체 씬 재로드** (`ui/history.ts`, cap 50):
  부분 diff 적용 대신 검증된 SceneSpec 전체를 loadScene 경로(검증→teardown→클린 빌드)로
  복원한다 — 정확성 우선, 물리/시각/레지스트리 정합이 구조적으로 보장된다. 로봇 URDF
  재로드는 브라우저 HTTP 캐시로 흡수(실측 재로드 ~수백 ms). 시퀀스는 원본 JSON을 새
  스펙에 재검증해 유효할 때만 유지(무효 → 미로드 + 콘솔 경고, 불변식 §2.9 유지).
  - 스냅샷은 noteChange 시점 **즉시 캡처**(eager) + 트레일링 디바운스(300ms)로 연속
    조정 burst(스크럽/기즈모)를 1장에 합친다. 지연 캡처였다면 burst 도중 구조 변경이
    끼면 이전 상태가 유실된다(게이트에서 실제 발견) — 구조 변경(add/remove/rename)은
    flushPending으로 경계를 세워 개별 스냅샷("undo 1회 = 구조 변경 1개").
  - 복원 실패(전환 busy/빌드 실패) 시 스택 이동을 되돌린다 — 스택과 씬이 어긋나지 않음.
  - 히스토리는 앱 수명(재로드를 가로질러 유지), 프리셋 전환/업로드 시에만 reset.
- **기즈모 commit 의미론** (render/interaction.ts 계약 수용): 드래그 "중"은 순수 시각
  프리뷰(물리 불변), 드래그 종료 commit 시점에 SceneEditor로 라우팅되어 물리(teleport)와
  정합된다 — 불변식 §2.1(물리가 진실)을 편집 UX와 양립시키는 유일한 지점. scale 모드는
  프리미티브만 **치수 편집으로 변환**(box: 축별 배율, sphere: 평균, cylinder/capsule:
  xz평균 반지름·y 높이, 하한 클램프)하고 updateDimensions 재빌드가 오브젝트 스케일을
  1로 소거한다("기즈모 스케일=빠른 조정, 진실은 치수" — UX §3.3). 비프리미티브는 거부
  + 시각 원복.
- **edit-during-play 정책**: 편집은 언제든 허용 — 동적 엔티티는 teleport(속도 0 초기화)
  후 물리로 반응한다. 재생 중 기즈모 드래그는 RenderSync가 매 프레임 시각을 덮어써
  시각적으로 싸우므로 일시정지/idle 편집을 권장(다음 sync에서 물리 진실로 재수렴).
  Undo/Redo는 재생 중이면 engine.stop 후 진행(결정론적 재로드) + 한국어 토스트
  '되돌리기: 시뮬 정지됨'. 안전 가드: arm된 시퀀스 재생 중 로봇 제거는 먼저 ⏹ Stop
  (사라진 로봇을 player가 구동하다 rAF 루프가 죽는 것 방지).
- **asset:// 세션 한정 한계(정직한 제약)**: 임포트 메시는 앱 수명 MeshAssetStore에서만
  해석된다(씬 전환/undo 재로드 유지). 씬 저장(💾)은 이제 editor.serialize()(현재 편집
  상태)를 내보내며, asset:// 참조가 있으면 ASSET_SAVE_WARNING_KO 경고 토스트+콘솔 로그
  — 다른 세션에서 열면 해당 엔티티는 복원 불가. 백로그: data-URI 내장(serializeRef).
  RenderSceneApi에 **선택 멤버** `addMeshAsset?(ref)` 추가(필수로 하면 헤드리스 테스트
  fake 전부가 깨진다 — 미구현 환경은 mesh visual에서 한국어 오류). 임포트 clone은
  geometry/material을 원형과 공유 — renderApi.remove가 userData.rswAssetRef 마커로
  disposeMeshResources를 건너뛴다(해제는 store.clear() 일괄).
- **워크스페이스 편입**: 커맨드바/독/우측 스택/뷰포트 상태선의 fixed 오버레이 배치를
  슬롯 흐름(static/absolute-in-slot)으로 중화. 렌더러 호스트(#app)를 뷰포트 슬롯으로
  reparent(absolute inset 0) — 스플리터 드래그/접기 후 notifyResize(window resize 합성)
  로 캔버스가 따라온다. Renderer에 domElement/orbitControls getter 추가(ViewportInteraction
  주입 — querySelector 의존 제거). 선택 동기화는 단일 경로: interaction.onSelect →
  inspector.select + entity-editor.showFor(각 모듈 변경 가드로 루프 없음).
- **three r169 TransformControls.dispose 버그 우회**: r169의 dispose()는 Controls
  리팩터링에서 남은 this.traverse 호출로 TypeError를 던진다(r170에서 수정). 씬 전환/
  undo 재로드가 전부 죽는 실제 크래시로 발견 — disconnect() + 헬퍼 서브트리 수동 해제로
  대체 (render/interaction.ts dispose).
- **한계(기록)**: 임시 관절 패널은 빌드 시점 로봇 스냅샷 UI — 편집으로 나중에 추가된
  로봇은 나타나지 않는다(관절 확인은 인스펙터, 정식 편집 UI는 후속 Phase). 라이브러리
  Import ⬆ 카드는 LibraryDeps.onImportRequest(선택 dep)로 추가 — 미주입 시 미렌더.
- **검증(전부 실측)**: `tsc --noEmit` 통과, eslint 경고 0, vitest **418/418** 통과
  (신규 12건 — history 스택/디바운스/경계/실패 복원), `vite build` 성공. 브라우저 게이트
  8종 **ALL PASS**: 기존 7종(falling-boxes/arm/arm-sequence/pick-and-place/
  obstacle-avoidance/collision-testbed/scene-switch) + **scene-builder**(라이브러리 카드
  7종 렌더, placeTemplate로 box_1 추가·바디 생성·인스펙터 목록 반영·자동 선택,
  updateTransform teleport 즉시 반영, updateDimensions 0.05→0.1 후 정착 y=0.095,
  removeEntity 정리, undo로 box_1 복원(전체 재로드) 후 sim 전진, 페이지 에러 0).
- 영향 파일: src/main.ts(재작성), src/ui/history.ts·history.test.ts(신설),
  src/render/renderer.ts(getter 2종), src/render/interaction.ts(dispose 우회),
  src/core/scene-loader.ts(addMeshAsset 선택 멤버 + mesh visual 분기),
  src/ui/viewport/statusline.ts(슬롯 absolute + 빈 씬 힌트 토글),
  src/ui/library/library.ts(Import ⬆ 카드), scripts/gate-browser.mjs(scene-builder),
  EXPERIMENTS.md

## [2026-07-23] Phase 7 리뷰 후속 수정 — 기즈모 드래그/커밋 경합 + 편집 정책 집행

### 배경 (리뷰 발견)
- **기즈모 commit 경합(major)**: Engine.frame은 playing/paused/idle 모두에서 rAF마다
  sync.apply를 호출한다 — 드래그 중 TransformControls가 움직인 시각 노드를 매 프레임
  물리 pose로 되돌려 (1) 드래그 프리뷰가 포인터를 따라오지 않고, (2) pointerup 시점
  matrixWorld decompose가 대개 물리 pose(≈드래그 전)로 리셋된 값을 읽어 드래그가
  조용히 유실됐다(release-while-moving일 때만 우연히 성공).
- **편집 teleport 후 stale prev(major)**: SceneEditor.updateTransform이 바디를
  teleport해도 RenderSync prev 스냅샷을 갱신하지 않아, paused 프레임의
  apply(alpha<1)가 편집 전 pose를 계속 보간해 그렸다(SceneHandle.reset()은 같은
  함정을 sync.commit()으로 이미 잡고 있었다 — 편집 경로만 누락).
- **임포트 동적 collider 필터(major)**: SENSOR_ZONE 누락 — Rapier 쌍 필터는
  양방향이라 임포트 사물이 Sensor Zone에 감지되지 않았다(라이브러리 템플릿과 불일치).

### 결정
- **드래그 pose 캡처 = objectChange 스냅샷**: commit은 라이브 matrixWorld가 아니라
  드래그 중 objectChange마다 캡처한 최신 트랜스폼을 발행한다(RenderSync 덮어쓰기에
  면역). 움직임 없는 드래그(기즈모 축 단순 클릭)는 commit을 발행하지 않는다 —
  보간 렌더 pose가 편집 경로로 물리에 역류하는 no-op teleport·잉여 undo 스냅샷 제거
  (transformsAlmostEqual, q≈-q 동일 처리, COMMIT_MIN_DELTA=1e-6).
- **드래그 수명 훅 onDraggingChanged**: dragging=false 통지는 commit 발행 "이후" —
  main.ts가 드래그 시작 시 대상 바디 sync 바인딩을 해제(프리뷰가 실제로 포인터를
  따라옴), 종료 시 재바인딩(prev = teleport된 물리 pose, 다음 프레임부터 재수렴).
- **edit-during-play 정책 변경 (기존 "언제든 허용" 항목을 대체)**: 살아있는 바디를
  변형하는 UI 편집(기즈모 드래그 시작·runEdit 경유 transform/dimensions/physics)은
  재생 중이면 **자동 일시정지** + 한국어 토스트('… 시뮬 일시정지됨 — ▶ Play로 재개').
  근거: core 계약(scene-editor.ts/types.ts "teleport는 씬 리셋/편집 전용 — 시뮬 중
  사용 금지")과 구현이 상충했고, 특히 로봇 루트 드래그는 playing 중 preStep tickAll이
  드래그 중간 pose를 kinematic 링크로 push해 "드래그=시각 프리뷰" 계약을 깼다(포인터
  속도 스윕 임펄스·waitForCollision 오발 가능). 추가(add/place)·remove·rename은 기존
  정책 유지(재생 중 허용 — 기존 궤적을 편집 API로 변형하지 않음). __sim.editor
  파사드(게이트/자동화)는 의도적으로 게이트를 우회한다 — teleport는 속도 0 초기화로
  spec/initialPose 정합이 유지되어 리플레이 결정론을 깨지 않는다(scene-builder 게이트는
  재생 중 파사드 편집 후 정착 y를 검증한다).
- **updateDimensions 다중 collider 거부**: colliders[0]만 교체하면 나머지 collider가
  이전 형상으로 남아 시각·물리가 조용히 어긋난다 — collider 2개 이상이면 한국어
  오류로 거부(수작성 씬 JSON만 해당, 라이브러리/임포트는 항상 1개).
- **hull 데시메이션 대표점 = AABB 중심 최원점**: 셀당 최초 등장 정점 유지는 비극점이
  hull 극점(모서리)을 밀어내 hull이 시각 메시보다 최대 셀 대각선(최소 divisions 12에서
  축당 ~8%)만큼 작아질 수 있었다 — 셀당 중심 최원점 선택(동거리 시 최초 등장,
  출력 순서 = 셀 최초 점유 순 — 결정론 유지)으로 극점 손실을 셀 이하 오목부로 한정.
  과대 근사는 여전히 없음(대표점은 항상 입력 정점).
- **RenderSyncLike에 commit 추가** (Pick bind|unbind → +commit): SceneEditor.
  updateTransform이 teleport 후 sync.commit()으로 prev를 갱신한다(reset()과 동일 계약).
- **액센트 색 주입**: main.ts가 ViewportInteraction 생성 시 ui/theme COLOR.accent를
  accentColorHex로 주입 — interaction.ts의 중복 상수는 진짜 fallback으로 강등.
- **임포트 동적 collider**: collidesWith에 SENSOR_ZONE 추가(템플릿 OBJECT_COLLIDES_WITH
  와 동일 배정 — 정렬 회귀 테스트 추가). Static(ENV)은 기존 유지.
- **ui→render 헤더 주석 정정**: import-dialog.ts의 "ui/viewport가 render를 래핑하는
  것과 같은 방향" 주장 삭제(부정확 — statusline은 render를 import하지 않음). 실제
  근거(하향 의존·역전 없음·three 불투명 통과)로 교체. 의존 주입 리팩터링은 보류
  (리뷰어 판정: non-blocking, 기존 ui→schema 관례와 일관).

### 검증 (전부 실측)
- `tsc --noEmit` 통과, eslint 경고 0, vitest **425/425** 통과 (신규 7건 —
  transformsAlmostEqual 3, teleport 후 sync prev 갱신 1, 다중 collider 거부 1,
  데시메이션 극점 보존 1, 임포트↔템플릿 collider 정렬 1), `vite build` 성공.
- 브라우저 게이트 8종 **ALL PASS**: falling-boxes / arm / arm-sequence /
  pick-and-place / obstacle-avoidance / collision-testbed / scene-switch /
  scene-builder (재생 중 파사드 편집·teleport·정착 y·undo 재로드 포함 — 편집
  일시정지 게이트가 파사드를 우회함을 실측 확인).
- 영향 파일: src/render/interaction.ts(드래그 캡처·no-op 스킵·onDraggingChanged),
  src/render/mesh-import.ts(대표점 선택), src/core/scene-editor.ts(sync.commit +
  다중 collider 가드), src/core/scene-loader.ts(RenderSyncLike +commit),
  src/ui/library/import-dialog.ts(SENSOR_ZONE + 헤더 정정), src/main.ts(액센트 주입 ·
  pauseForEditIfPlaying · 드래그 sync 서스펜드), 테스트 3파일(+7건), EXPERIMENTS.md

## [2026-07-23] Phase 8 — Flow Graph 에디터 (스키마 뷰모델 + 캔버스 + 노드 폼 + 통합)

### 배경
ControlSequence를 n8n형 노드 그래프로 시각화·편집한다 (UX_DESIGN §3.4/§6, ROADMAP
Phase 8). 병렬 작업 3건(F1 schema/flow-graph.ts + player 스킵, F2 ui/flow-graph 캔버스,
F3 ui/inspector/node-editor.ts 폼)을 main.ts 글루로 통합했다. 핵심 게이트는 불변식
§2.8 — 그래프 편집으로 직렬화 불가능한 상태를 만들 수 없다.

### 결정 (스키마 계층 — F1)
- **edges는 파생 상태다**: FlowGraph의 단일 진실은 노드 배열(순서 + goto params)이며,
  seq/loop 엣지는 모든 편집 후 deriveEdges로 재유도된다. 엣지를 독립 편집 대상으로
  두면 "노드 순서 ↔ steps 배열 1:1" 불변식(UX §6)과 이중 진실이 생긴다.
- **loop 엣지는 첫 label 승리**: goto → 같은 이름의 첫 번째 label 노드. player의
  "중복 label은 첫 번째가 이긴다" 규칙과 동일한 해석을 뷰에도 적용.
- **정규형: enabled:true는 직렬화에서 생략**: toSequence는 enabled가 false일 때만
  키를 남긴다 — 편집 왕복이 원본 JSON에 잡음을 더하지 않는다.
- **편집 연산의 구조 게이트는 씬 없이 돈다**: F1 op의 finishEdit는 serializeGraph
  (scene 없음 — zod 구조 검증)만 통과시킨다. 씬 참조 무결성(로봇/엔티티/관절 존재)은
  씬을 아는 통합 글루의 소관(아래).
- **시퀀스 id/loop는 그래프에 실리지 않는다**: FlowGraph는 steps의 뷰다(§6 고정 계약
  — nodes/edges/robot). 원본 id/loop는 ToSequenceOptions로 복원한다 — main 글루가
  flowSeqMeta로 보관·전달.
- **player 스킵 시맨틱**: enabled:false step은 핸들러 실행 없이 건너뛴다(순서 유지).
  스킵은 dt를 보존한다 — 이번 tick의 첫 "실행" step이 전체 dt를 받아 "비활성 = 없는
  것처럼" 동작. 비활성 goto는 점프하지 않고 통과, 비활성 label은 여전히 goto 대상
  (위치 마커). 스킵도 same-tick 진행 한도(64)에 계상.

### 결정 (캔버스 — F2)
- **캔버스는 ui.x/y를 무시하고 체인 순서에서 결정론 배치**: 렌더가 (graph, statuses,
  selection, viewport)의 순수 함수가 되고, 드래그 재정렬의 "drop x → 삽입 인덱스"
  계산이 1행 체인을 전제로 단순해진다. ui.x/y는 표현 전용(UX §6)이라 실행/직렬화에
  무영향 — 자유 배치는 필요해질 때 재검토.
- **리스너는 마운트 시 1세트 위임 부착**: render()는 DOM만 재구축한다(전체 재렌더,
  MVP ≤ 64 노드) — 리스너 누수 없음.
- **캔버스는 그래프를 직접 변형하지 않는다**: 모든 편집이 deps.applyOp(순수 op)로
  나가고, 거부 피드백(한국어 토스트)은 통합자 몫.

### 결정 (통합 — main.ts 글루)
- **편집 파이프라인 단일 경로 runFlowOp**: op(구조 검증 포함) → serializeGraph(graph,
  editor.spec, {id, loop}) 씬 참조 무결성 → '수정됨' 배지 diff → 커밋(라이브 시퀀스
  교체 + JSON 뷰어/타임라인 갱신) → 캔버스/폼 재동기화. 실패는 한국어 오류로 거부되고
  그래프/시퀀스는 불변 — UI(캔버스/팔레트/폼)와 __sim.flowGraph 파사드가 전부 이
  경로만 쓴다(§2.8).
- **시퀀스 편집은 재생을 정지한다**: armed 중 편집 커밋 → unarm + player 커서 0 +
  토스트 '시퀀스 수정됨 — 처음부터 재생됩니다'. 엔진(씬 물리)은 계속 돈다. preStep이
  sequenceArmed로 player.step을 게이트한다(ControlPlayer에 unload가 없어 로드된 이전
  시퀀스의 진행을 글루가 차단). 다음 ▶ Play가 "현재" 시퀀스를 새로 load — human-in-
  the-loop(§2.9) 유지: 편집이 자동 재생을 유발하지 않는다.
- **'수정됨' 배지는 내용 diff로 승격**: fromSequence 로드는 origin 'manual'인데 F1
  op는 generated→modified만 승격한다 — 글루가 커밋 시 params/enabled/note 내용이
  바뀐 노드를 'modified'로 승격해 "로드된 JSON과 달라짐"을 표시. 순서 이동만으로는
  배지가 붙지 않는다(내용 불변).
- **우측 스택 선택 중재 (마지막 선택 승리)**: 노드 선택 → 뷰포트 선택 해제 + 노드
  폼 표시, 엔티티 선택 → 노드 선택 해제 + 엔티티 폼 표시. 해제(null) 에코는 패널을
  바꾸지 않아 루프가 없다. 두 폼 모두 마운트 유지, display만 전환.
- **player 상태 동기(기본형)**: onStepChange → 커서 앞 done · 현재 active · 뒤
  pending. 비활성 노드는 active로 표시하지 않는다(스킵은 같은 tick 통과).
  waitForCollision timeout 경고(warn 콜백 문구 매칭)는 해당 노드를 'error'로 마킹.
  Phase 10이 노드 경계 정지·재실행으로 심화한다.
- **setNote op는 글루가 보완**: note는 노드 필드(params 제외)라 updateNodeParams로
  닿지 않는다 — 같은 §2.8 파이프라인을 거치는 로컬 op(deriveEdges + serializeGraph).
- **adoptIntoStack이 zIndex도 auto로 리셋**: 단독 마운트 기본값(Z_INDEX.panel=100)이
  flex 아이템으로 남아 {} JSON 슬라이드 패널(95)을 가려 클릭을 가로챘다(게이트에서
  실측 발견) — 우측 스택은 슬라이드 패널보다 아래가 theme 규약.
- **씬 undo/redo가 시퀀스 편집을 보존**: ActiveScene.sequenceJson이 편집된 라이브
  시퀀스를 우선 반환 — 히스토리 재로드가 그래프 편집을 잃지 않는다.
- **시퀀스 없는 씬도 '플로우' 토글로 빈 그래프 시작**: 기본 로봇 = 씬의 첫 로봇.
  로봇 없는 씬은 시퀀스 스키마(robot 참조)가 모든 편집을 거부한다 — 페인 열 때
  한국어 안내 토스트.

### 알려진 한계 (Phase 10 백로그)
- Space 단축키가 재생 토글(playback)과 캔버스 팬 수식키에 동시 배정 — 전역 단축키
  중재 미구현. Del은 선택 중재(한쪽만 선택 유지)로 실질 충돌 없음.
- 씬 편집(엔티티 삭제/개명)이 라이브 시퀀스 참조를 깨는 경우의 재검증은 로드/편집
  시점에만 이뤄진다(다음 그래프 편집 시 거부로 표면화).
- moveToPose는 팔레트에서 제외(IK 백로그 — player가 경고 후 스킵). 기존 시퀀스의
  해당 노드는 정상 표시된다.

### 검증 (전부 실측)
- `tsc --noEmit` 통과, eslint 경고 0, vitest 578/578 통과, `vite build` 성공.
- 브라우저 게이트: **flow-graph 신규 ALL PASS** (7노드 렌더 → insertWait(2) 8노드 +
  직렬화 검증 → JSON 뷰어 동기 → reorder 순서 갱신 → waitForCollision 비활성 재생
  5.92s done(배리어/timeout 미경유, 스킵 노드 무active) → 삭제 7→6 + 유효 + 페이지
  에러 0) + 기존 게이트 재실행(arm-sequence/scene-builder/scene-switch/pick-and-place).
- 영향 파일: src/main.ts(글루 통합 + adoptIntoStack zIndex), scripts/gate-browser.mjs
  (flow-graph 시나리오), EXPERIMENTS.md. (F1/F2/F3 산출물은 병렬 에이전트 몫.)

## [2026-07-23] Phase 8 리뷰 후속 수정 — 씬 편집 ↔ 플로우 정합 (arm 재검증 · rename 재동기)

### 배경 (리뷰 지적)
Phase 8 통합의 "알려진 한계"였던 "씬 편집이 라이브 시퀀스 참조를 깨는 경우"가
실제로는 두 가지 실질 결함이었다:
1. **Play arm이 재검증 없이 stale 시퀀스를 로드**: 커밋 이후 로봇 rename/제거가
   일어나면 다음 Play에서 Engine preStep(try/catch 없음)의 RobotRegistry.get이
   던져 tick 루프가 죽는다 — "검증 통과본만 실행"(§2.9) 계약이 실행 시점에 깨짐.
2. **flowGraph.robot이 빌드 시점 스냅샷으로 고정**: (a) 로봇 rename 후 모든 플로우
   편집이 "씬에 없는 엔티티"로 거부, (b) 로봇 없는 씬(robot '')에 로봇을 추가해도
   편집 영구 거부 — §2.8은 지켜지지만(거부+한국어 피드백) 에디터가 세션 내 잠긴다.

### 결정
- **arm 직전 재검증**: armSequenceIfAvailable이 player.load 전에
  `validateSequence(currentSequence, editor.spec)`를 실행 — 실패 시 arm 거부 +
  한국어 토스트/콘솔 오류. 엔진 재생(씬 물리)은 시퀀스와 무관하게 계속된다.
- **rename 재동기 remapEntityId(schema/flow-graph)**: 순수 치환 함수(기본 robot ·
  step params.robot · waitForCollision.between). 시스템 동기화라 origin 배지를
  바꾸지 않고, 참조가 없으면 동일 참조를 반환한다(변경 판별 계약). 글루는
  SceneEditEvent(rename)에서 옛 id를 "직전 통지 시점 id 목록과의 차집합"으로
  복원해(이벤트에 new id만 실림) §2.8 파이프라인(serializeGraph(scene))으로 커밋.
- **기본 로봇 채택**: 'add' 통지에서 flowGraph.robot이 무효(빈 문자열/죽은 참조)면
  첫 로봇을 채택 — 유효한 기본 로봇은 절대 바꾸지 않는다.
- **Stop = unarm + 캔버스 런 상태 클리어**: 이전 런의 'error'/everActive가 Stop→Play
  재실행에 잔존하고 player.reset() 통지가 idle 중 노드 0을 'active'(펄스)로 남기던
  문제 — Stop이 unarm하여 다음 Play가 재검증→재로드(arm) 단일 경로를 타고,
  statuses/everActive를 비워 idle 캔버스는 전부 pending(재생 바 '대기' 표기와 일관).
- **waitForCollision timeout 문구 계약을 상수로 고정**: steps.ts가
  WAIT_FOR_COLLISION_WARN_TAG/TIMEOUT_MARKER를 export하고 발행 문구를 이 상수로
  조립, main.ts 매칭도 동일 상수 사용 — 발행측 리워딩이 노드 'error' 마킹을 조용히
  끊을 수 없다(steps.test.ts가 문구 포함을 핀).
- **시퀀스 언로드 토스트**: 씬 빌드에서 시퀀스 검증 실패 시(히스토리 undo로 참조가
  깨진 경우 포함) 콘솔 로그에 더해 토스트로 표면화 — undo 직후 플로우가 조용히
  사라지는 UX 방지.
- **테마 토큰 3종 추가**: COLOR.borderHover('#4a5058')/gridDot/mutedSoft — flow
  캔버스의 하드코딩 색 리터럴 3곳을 토큰 소비로 교체(값 동일 — 시각 변화 없음).
- **UX_DESIGN §6 문서 정정**: FlowNode.id 주석 "= ControlStep.id" →
  그래프-로컬 안정 id('n1','n2',…)로 — ControlStep에는 id 필드가 없다(구현·파사드
  nodeIds 계약과 문서 일치, CLAUDE.md §10).

### 명시적 보류 (백로그 유지)
- **플로우 편집의 Undo/Redo 미포함**: SceneHistory는 SceneSpec 스냅샷만 기록한다 —
  UX §7("씬·그래프 편집 전반")의 그래프 몫은 {spec, sequence} 쌍 스냅샷으로의
  히스토리 일반화가 필요해 Phase 8 범위에서 보류(ROADMAP Phase 8 체크리스트에도
  없음). 부작용: 플로우 편집 직후 Ctrl+Z는 마지막 "씬" 편집을 되돌린다(전체 재로드,
  라이브 시퀀스는 sequenceJson 경유로 보존). Phase 10 전후 재검토.
- **로봇 제거 시 재동기는 하지 않는다**: 죽은 참조의 자동 치환은 의미 변경이라
  거부 피드백(편집)과 arm 재검증(재생)으로만 막는다 — 로봇을 다시 추가하면
  기본 로봇 채택이 편집을 되살린다.

### 검증 (전부 실측)
- `tsc --noEmit` 통과, eslint 경고 0, vitest 586/586 통과(+8: remapEntityId 7 ·
  timeout 문구 계약 1), `vite build` 성공.
- 브라우저 게이트 ALL PASS: flow-graph(신규 어서션 2종 포함 — Stop 런 상태 리셋 +
  재-Play 완주 6.00s done, 로봇 rename 후 seq.robot/between 재동기 + insertWait
  성공) · arm-sequence · scene-builder · scene-switch.
- 영향 파일: src/schema/flow-graph.ts(remapEntityId), src/main.ts(arm 재검증 ·
  Stop 리셋 · rename/add 재동기 글루 · 문구 상수 매칭 · 언로드 토스트),
  src/core/control/steps.ts(문구 계약 상수), src/ui/theme.ts(+토큰 3종),
  src/ui/flow-graph/canvas.ts(토큰 소비), docs/UX_DESIGN.md §6(FlowNode.id 정정),
  scripts/gate-browser.mjs(flow-graph 어서션 2종), 테스트 2파일.

---

## [2026-07-23] Phase 9 — 자연어 Planner 앱 통합 (설정·생성 흐름·그래프 로드·게이트)

- 상태: 결정됨
- 맥락: P1(src/planner/* — PlannerService·buildContext·규칙기반/Anthropic 어댑터·복구
  루프)와 P2(nl-input·clarify-card·toast·planner-settings ui) 랜딩분을 앱에 배선한다.
  자연어 → 검증된 ControlSequence 초안 → Flow Graph 로드까지 §2.9(미검증/무자동재생)를
  매 출구에서 집행하는 것이 목표. 규범: PLANNER.md, UX_DESIGN §4.1 Flow 1, ROADMAP Phase 9.

### 선택 — (a) 플래너 백엔드: 규칙 기반 오프라인 기본 + Anthropic SDK 어댑터
- **기본은 규칙 기반(오프라인)** — 네트워크·API 키 없이 결정론적으로 동작하므로 데모/
  게이트가 재현 가능하다. Anthropic은 설정에서 명시적으로 켜야 활성(키 필요).
- **구조화 출력(structured outputs, output_config.json_schema)으로 JSON 형식을 보장**
  한다. PLANNER.md §4.1의 "결정성 힌트: 온도 낮게"는 이 모델(claude-opus-4-8)에서
  temperature/top_p/top_k가 제거되어(전송 시 400) 쓸 수 없다 — 대신 **구조화 출력 +
  few-shot 예시 1개 + 명시적 그라운딩 규약**으로 안정성을 확보한다(supersedes PLANNER.md
  §4.1의 온도 힌트 in practice). adaptive thinking 사용.
- 어댑터 격리(LlmAdapter): main은 설정에서 AnthropicAdapter(apiKey/model)를 주입만 하고,
  planner 계층은 localStorage/DOM을 모른다(CLAUDE.md §3). anthropic+키가 있을 때만 SDK
  경로, 그 외엔 규칙 기반으로 방어적 폴백(buildPlannerService).

### 선택 — (b) API 키 localStorage 저장 UX와 경고
- 설정은 `localStorage['robotSimWeb.planner']`에 { backend, apiKey, model }로 저장.
  손상값/localStorage 불가(프라이빗 모드)는 조용히 규칙 기반 기본으로 폴백한다.
- **투명성 고지**(planner-settings): "키는 이 브라우저에만 저장되고 Anthropic 호출에만
  쓰이며 공용 PC에서 쓰지 말 것" — 항상 표시. 교육/프로토타입 도구임을 재확인(PRD §6).
- 게이트는 **fresh chromium(빈 localStorage) = 규칙 기반**이라 네트워크 없이 결정론적.

### 선택 — (c) 이어서(append) 모드 label 충돌 처리
- append 병합은 기존 step 뒤에 incoming step을 이어 붙이되, incoming의 label 이름이
  기존 label과 충돌하면 suffix('_2','_3'...)로 개명하고 **같은 세그먼트의 goto 참조도
  함께 갱신**한다(appendStepsWithLabelRename). 서로 다른 세그먼트로 향하는 goto는 건드리지
  않는다(edge case, 문서화).
- append는 **새로 이어 붙인 step만 origin 'generated'**로 표시하고 기존 노드 origin은
  보존한다(fromSequence는 step↔노드 1:1이므로 앞 k개=기존, 뒤=신규). 병합본은 §2.8
  파이프라인(serializeGraph(scene) 참조 무결성)으로 재검증 후에만 커밋된다.

### 선택 — 규칙 기반 reach를 2단(approach+nudge)으로 — 결정론적 접촉 계약
- **맥락**: waitForCollision 배리어는 init 마커 "이후"의 충돌만 감지한다(steps.ts
  happenedSince). 단일 moveJoints로 박스에 닿으면 접촉 start가 배리어 시작 "전"에 발생해
  놓치고 timeout(6s)으로 흘러 done이 늦고 노드가 'error'로 마킹된다.
- **선택**: planTouch를 open→**moveJoints(approach, 박스 바로 위)**→**setJoints(nudge,
  눌러 내림)**→waitForCollision→close→wait→home = 7 step으로 바꿨다(기존 6 → 7).
  이는 arm-touch-box.sequence.json의 검증된 2단 패턴과 동일하다 — nudge가 배리어 직후
  접촉 start를 만들어 이벤트로 해제한다. mid 밴드(0.3≤r<0.45) 값은 arm-touch-box의 검증
  자세(approach joint2 0.639·joint3 1.414·joint5 1.089 / nudge 0.683·1.442·1.017)를 채택.
  box_a[0.35,·,0.15]·box_b[0.3,·,-0.2] 모두 이 밴드.
- **측정치**: planner 게이트에서 arm×box_a 충돌 start @5.03s, done @play+5.27s(이벤트
  해제 — timeout≈12s 경로 아님). rule-based.test.ts 20건 통과(구조 6→7 step 반영).

### 선택 — 파사드/글루 경계
- **__sim.planner 파사드**(씬별): generate(nl)은 앱 수명 runGenerate를 위임 호출한다 —
  UI와 완전히 같은 흐름(buildContext→생성→심층 방어 재검증→그래프 로드→무자동재생).
  lastResult/isLoadedIntoGraph/playerStatus로 게이트가 §2.9(무자동재생)를 증명한다.
- **앱 수명 vs 씬별**: 플래너 서비스·설정·nl-input·clarify·toast·⚙는 앱 수명(boot),
  생성 시퀀스는 "현재 씬"의 그래프에 로드되므로 ActiveScene.loadGeneratedSequence로
  위임(replace/append). 씬 전환을 가로질러 nl-input은 커맨드바 중앙-좌에 유지된다.
- **AI 배지 DOM 노출**: canvas.ts 노드 <g>에 data-origin=node.origin 추가(배지 텍스트와
  병행) — 게이트가 generated 노드를 DOM으로 검증. 시각 변화 없음(배지는 기존대로 렌더).
- **심층 방어(§2.9)**: planner가 이미 validateSequence를 통과시키지만, main도 실행 노출
  직전 현재 씬에 한 번 더 validateSequence — 실패 시 로드하지 않고 토스트/콘솔로 표면화.

### 트레이드오프
- 규칙 기반 reach는 정확한 IK가 아니라 밴드 테이블 휴리스틱 — near/far 밴드는 mid의
  형상 추세를 유지한 근사이며 게이트로 검증되는 것은 mid(box_a/box_b)뿐. IK 솔버는 백로그.
- append origin 승격은 기존 노드를 'manual'로 재빌드 후 인덱스로 이전 origin을 복원한다 —
  노드 id는 fromSequence가 새로 발급('n1'..) 하므로 선택/상태는 commit이 리셋(무해).

### 검증 (전부 실측)
- `tsc --noEmit` 통과, eslint 경고 0, vitest **662/662 통과**(+2 파일·+기존 대비 planner
  게이트 반영), `vite build` 성공.
- 브라우저 게이트 **ALL PASS**: planner(신규 — 생성/AI배지/무자동재생/실제 접촉/clarify→
  box_b/견고성) · flow-graph · arm-sequence · scene-builder · scene-switch · pick-and-place.
- **빌드 마찰(환경)**: `vite build`가 Bash(git-bash)의 소문자 드라이브레터 cwd(`c:\...`)
  에서 `[vite:html-inline-proxy] No matching HTML proxy module found`(대소문자 불일치
  `C:` vs `c:`)로 실패 — PowerShell(대문자 `C:\...`)에서는 정상. 코드/설정 무관한 Vite
  Windows 드라이브레터 케이싱 버그. 빌드/게이트는 PowerShell에서 실행할 것.
- 영향 파일: src/main.ts(플래너 boot 배선 · runGenerate/handlePlannerResult · 설정
  영속화 · loadGeneratedSequence/append 병합 · __sim.planner 파사드), src/planner/
  adapters/rule-based.ts(2단 접근+nudge reach), src/planner/planner.ts(few-shot 2단),
  src/ui/flow-graph/canvas.ts(data-origin), scripts/gate-browser.mjs(--expect=planner),
  src/planner/rule-based.test.ts(7 step 반영).

## [2026-07-23] 그라운딩⇄검증 정렬: gripper 전용 관절을 SceneContext.joints에서 제외

- 상태: 결정됨
- 맥락: 안전 리뷰 지적 — scene-context.ts의 robotJointNames가 joints 배열을
  home ∪ jointMap ∪ jointLimits ∪ gripper.joints로 유도했으나, validate.ts의
  checkJointNames는 knownJoints를 home ∪ jointMap ∪ jointLimits로만 구성한다.
  gripper 전용 관절(예: finger_left_joint)은 LLM에 "사용 가능 관절"로 광고되지만
  moveJoints/setJoints 대상으로 쓰면 검증에서 거부되는 비대칭이 있었다.
- 선택: robotJointNames에서 gripper.joints 포함을 제거해 joints 배열을 checkJointNames의
  knownJoints와 정확히 일치시킨다(Option B). validator는 손대지 않는다.
- 근거: 그리퍼는 별도 제어면이다 — `gripper` step(state: open/close/0..1)으로 구동되고
  SceneContextRobot.gripper로 이미 완전히 노출된다. gripper.joints 존재 검증은 URDF를
  아는 최초 지점인 scene-loader.attachRobotPhysics가 담당한다(기존 리뷰 회귀 결정).
  따라서 gripper.joints는 moveJoints/setJoints 대상이 아니며, joints 배열로 광고하면
  잘못된 메타데이터(revolute/[-π,π]/current 0인 평행 그리퍼 prismatic 관절)까지 함께
  노출된다. validator를 느슨하게 하는 Option A는 평행 그리퍼 추상(양 손가락 동시 보간)을
  깨므로 채택하지 않았다. §2.9 위반 아님(fail-safe) — 그라운딩 정확도·재시도 낭비 개선.
- 트레이드오프: gripper 관절명은 이제 joints 배열이 아닌 gripper 객체로만 LLM에 보인다
  (의도된 제어 경로). gripper.joints가 home 등에도 선언되면 그 경로로 자연히 포함된다.
- 영향 파일: src/planner/scene-context.ts(robotJointNames + docstring),
  src/planner/scene-context.test.ts(joints 배열 기대치 갱신 + 그라운딩⇄검증 정렬
  회귀 테스트 2건 추가: validateSequence 통과/거부).

## [2026-07-23] Phase 10 실행 오케스트레이션 — Orchestrator를 Engine/Player 위에 배선

- 상태: 결정됨 (게이트 통과)
- 맥락: Phase 8이 배선한 `player.onStepChange → canvas.setStatuses`를 UX_DESIGN §5의
  일급 오케스트레이션(노드 경계 제어·트라이페인 동기·충돌 인지 정지·결정론적 재실행)으로
  심화한다. 코어(Engine/ControlPlayer/CollisionMonitor)는 손대지 않고, `ui/orchestrator.ts`
  Orchestrator를 그 위에 얇게 감싸 main.ts 글루가 재생 컨트롤·상태·재실행을 통과시킨다.

### 결정 1 — 노드 단위 제어를 코어 위 표현 계층으로 (진실은 player 커서)
- Orchestrator는 player를 직접 tick하지 않는다. player는 Engine preStep 훅이 구동하고,
  Orchestrator는 `player.onStepChange` 커서 통지를 관찰해 **표현 상태**(노드 상태 맵·활성
  노드)만 파생·방출한다. 재생 제어는 `engine.play/pause/stop/setSpeed`로만 한다.
- ⏸/⏭는 "노드 경계" 단위다: PausePolicy(none/atNext/stepOne/runTo)를 순수 함수로 두고
  다음 onStepChange 경계에서 `engine.pause()`를 결정한다. ⏭ Step은 "물리 1 tick"이 아니라
  "노드 1개"다 — 물리-tick 프레임 스텝(engine.stepOnce)은 UI에서 내렸다(디버그용으로 코어엔 잔존).
- 트라이페인 소스: **player step 인덱스가 유일한 진실**. 그래프 활성 노드(캔버스 아웃라인+
  active 점) ↔ 뷰포트 run-overlay 'node k/n'+활성 라벨 ↔ Timeline 커서 마커가 모두 이 인덱스
  에서 파생된다. Timeline 활성/오류 마커는 상태 맵에서 역산한다(active→커서, error→오류 마커,
  전부 done→끝). 게이트가 세 뷰가 같은 노드를 가리키는지 한 순간에 대조해 검증한다(PASS).

### 결정 2 — 재실행(runFromNode) 결정론: per-node 스냅샷 없이 "처음부터 되감아 빨리감기"
- per-node 씬 스냅샷이 없으므로 재실행은 **항상 resetScene로 처음으로 되감고** 목표 앞 노드를
  4×로 재생 후 목표 경계에서 정지(+속도 복원)한다. 같은 시퀀스는 같은 setJoints 로그를 낸다
  (ControlPlayer.load/reset 계약 — 결정론).
- 한계(정직한 MVP): Engine 프레임 루프가 한 프레임 안 물리 tick 사이에 pause를 재검사하지
  않아 목표 노드가 "실행 직전"이 아니라 "경계 ±1 tick"에서 멈춘다. ⏭ Step의 <1프레임 노드
  오버스텝도 같은 계열이다. 정밀 재개점은 per-node 스냅샷 도입 시 개선(백로그).
- resetScene는 매 되감기마다 `armFromStart`(validateSequence→player.load)로 **재검증 후
  재장전**한다. player.load의 커서 통지는 Orchestrator의 resetting 가드(withReset) 안에서
  무시되고, 이후 명시적 recompute가 상태를 그린다. 편집으로 unarm되는 케이스는 stop이 아닌
  `resetForEdit`(엔진/씬 무영향, 표현만 pending)로 분리했다 — edit-during-play 정책 유지.

### 결정 3 — "예기치 않은 충돌" 판정 휴리스틱 (보수적 — 오검출로 정상 실행을 물들이지 않음)
- 승격 조건을 좁혔다: start phase의 로봇×비로봇 접촉 중 (a) 바닥이 아니고, (b) 어떤
  waitForCollision 배리어 대상 쌍도 아니며, (c) 상대가 **동적 사물이 아닐** 때만 예기치 않음.
  즉 robot × 정적 환경(벽·기둥)의 비의도 접촉만 오류로 승격한다.
- 근거: 로봇이 바닥에 서 있는 접촉·동적 사물 조작(밀기/파지)·배리어 대상 접촉은 전부 정상이다.
  이를 오류로 물들이면 완주 시퀀스가 "전부 done"이 아니게 되어 재생이 오해를 부른다. 비활성
  배리어의 쌍도 "조작 대상"으로 포함해(enabled 무관) 배리어를 꺼도 그 접촉을 오류로 보지 않는다.
- 자동 정지 토글('충돌 시 자동 정지', 기본 off, 재생 바 옆 체크박스): 켜면 위 판정에서 자동 ⏸.
  결정적으로 강제하기 어려워(정적 벽이 있는 씬 필요) 게이트는 토글이 facade+체크박스에 반영되는
  배선만 검증한다(§5 문구대로 "at minimum assert the toggle flips a facade-visible flag").

### 결정 4 — run-overlay가 statusline의 상태 라인을 대체(빈 씬 중앙 안내는 유지)
- 뷰포트 좌하단 실행 오버레이를 `mountRunOverlay`로 교체하되, 기존 statusline은 "빈 씬 중앙
  안내"만 담당하도록 상태 라인 el만 숨겨 남긴다(setEmptyHintVisible 연동 유지 — 회귀 없음).
- 오버레이 engineState는 **시퀀스 실행 상태**를 비춘다: 물리만 도는 대기(미arm/done)는 'Idle',
  armed+running이면 엔진 상태(playing/paused). 물리 자동재생 씬에서 부트 오버레이가 'Running'으로
  오인되지 않게 한 선택 — 실행 오케스트레이션 배지의 의미와 정합.

### 검증 (전부 실측 — PowerShell, Windows 드라이브레터 케이싱 이슈 회피)
- `tsc --noEmit` 통과 · eslint 경고 0 · vitest **724/724 통과** · `vite build` 성공.
- 브라우저 게이트 **11종 ALL PASS**: orchestration(신규 — 초기 pending·상태 진행·arm×box_a·
  트라이페인 일관·완주 done·stepNode 1노드·autoPause 토글·runFromNode) · flow-graph ·
  arm-sequence · planner · pick-and-place · scene-builder · scene-switch · obstacle-avoidance ·
  falling-boxes · arm · collision-testbed.
- 영향 파일: src/ui/orchestrator.ts(resetForEdit 추가 — 편집 unarm 표현 리셋), src/main.ts
  (Orchestrator 배선: armFromStart/resetScene/playbackControls를 orchestrator 경유 · onNodeStatus/
  onActiveNode/unexpectedCollision · run-overlay 스왑 · 충돌 로그 onRowClick 노드 강조 · Timeline
  onMarkerClick 재실행 · __sim.orchestrator 파사드), scripts/gate-browser.mjs(--expect=orchestration),
  README.md(Flow 1 워크스루 + Phase 0–10 완료), EXPERIMENTS.md(이 항목).

### 결정 5 — 뷰포트 오버레이를 rAF가 아니라 **동기 진실**에서 갱신 (Phase 10 사후 수정)
- 문제(자기비판 리뷰): run-overlay의 `lastOverlayState`가 `engine.onTick`(rAF cadence)에서만
  재계산됐다. 반면 그래프 active dot·Timeline 커서·__sim 파사드는 `orchestrator.onActiveNode/
  onNodeStatus`로 **동기적으로** active가 된다(▶ Play가 player를 arm하는 그 순간). 결과적으로
  뷰포트 오버레이가 나머지 두 페인보다 **최대 1 rAF 늦어**, §5 "항상 일치"가 Play 순간에는
  문자 그대로 성립하지 않았다. orchestration 게이트가 이를 결정론적 red로 노출했다("Idle ·
  simTime …"를 running 오버레이로 캡처).
- 수정: 오버레이 계산을 순수 헬퍼 `computeOverlayState(engineState, simTime)`로 뽑고,
  `refreshOverlay()`를 (a) onTick(연속 simTime), (b) `onActiveNode`(노드 경계 — 그래프/타임라인과
  같은 동기 지점), (c) 재생 컨트롤 play/pause/stop/step **직후**(engine.state 전이는 동기적 —
  engine.ts 확인)에서 호출한다. 이제 오버레이의 Running/Paused/node-progress 전이가 세 페인과
  한 프레임도 어긋나지 않는다. `activeIndex`(재생 진실)는 여전히 player 커서가 유일 소유 —
  오버레이는 그 파생 뷰일 뿐이다(불변식 유지).
- 게이트 정직성(자기비판 후속): (1) `runningOverlay`를 매 폴에서 재래치해 1프레임 지연에
  영구 실패하지 않게 하고, (2) 트라이페인 등식에 **오버레이 파싱 노드('node k/n'→id)를 독립
  항으로 추가**해 "graph == overlay == timeline == facade"가 라벨대로 실제 4항을 대조하게 했다
  (이전엔 오버레이가 등식에서 빠져 있었다). 3회 연속 결정론 PASS 확인.
- 부수 정리: OrchestratorEngine 주입 계약에서 미사용 `stepOnce()` 제거(표면 최소화 — 노드
  경계 러너는 play()+PausePolicy로 구동, engine.stepOnce는 코어에 디버그용으로 잔존).
  edit-during-play 토스트를 '시퀀스를 처음부터 …(씬은 유지 · 완전 되감기는 ⏹ Stop)'로 명확화
  (씬 물리는 리셋 안 됨 — resetForEdit 스코프 정직화). package.json에 `gate*`/`gate` 스크립트
  추가(브라우저 DoD 게이트 재현 가능·강제 가능).
- 검증: tsc·eslint·vitest 724/724 통과, vite build 성공, 게이트 [orchestration·planner·
  flow-graph·arm-sequence·scene-builder·scene-switch] ALL PASS(orchestration 3회 결정론).
- 영향 파일: src/main.ts(computeOverlayState/refreshOverlay + 배선), src/ui/orchestrator.ts
  (stepOnce 제거), src/ui/orchestrator.test.ts(목 정리), scripts/gate-browser.mjs(오버레이 재래치
  + 트라이페인 오버레이 항), package.json(gate 스크립트), EXPERIMENTS.md(이 항목).


---

## 2026-07-28 — 버그 수정: "오브젝트는 이동하는데 로봇이 이동을 안한다"

사용자 보고: 라이브러리에서 로봇팔 2대를 배치해 충돌 실험을 하던 중, 오브젝트는 드래그로
옮겨지는데 로봇은 옮겨지지 않았다.

### 결정 1 — 근본 원인은 이동 파이프라인이 아니라 **기즈모 앵커 기하**였다
- 실측(?scene=two-arms-collision, 1600×950): `witness_box`는 기즈모 원점과 시각 메시 중심의
  이격이 0 px인데, `arm_left`는 **376 mm(화면 115~201 px)** 떨어져 있었다. TransformControls는
  attach한 객체의 **원점**에 핸들을 그리고, URDF 로봇의 시각 루트(outer Group) 원점은 베이스
  링크 = 바닥(y=0)이기 때문이다. 핸들 유효 반경은 실측 ~70 px이라, 사용자가 보이는 몸통을
  누르면 pointerdown이 기즈모에 잡히지 않고 OrbitControls로 흘러 **카메라만 돌았다**.
- 즉 이동 경로(interaction.emitCommit → main.onTransformCommit → SceneEditor.updateTransform →
  setRootTransform + teleportLinksToFk + tick)에는 결함이 없었다. 프로그램 select + 방향키가
  정상 동작했던 것도, 조사자가 기즈모 **원점**을 계산해 끌었을 때 정상이었던 것도 이와 정합한다.
- 수정: 기즈모를 선택 루트가 아니라 **씬에 상주하는 프록시 앵커 Object3D**에 붙이고, 앵커를
  선택 시 계산한 시각 AABB 중심(선택 아웃라인 BoxHelper와 같은 박스)에 둔다. 드래그 중
  objectChange마다 앵커 pose를 루트 pose로 역변환해 적용하므로 **commit 페이로드(=루트의 월드
  트랜스폼) 계약은 그대로**다 — main.ts 글루와 core(SceneEditor)는 손대지 않았다.
  앵커 오프셋은 선택 시 1회 고정한다(매 프레임 재계산하면 재생 중 FK 변화로 핸들이 흔들린다).
- 대안 기각: `TransformControls.size` 확대는 376 mm 이격 자체를 해소하지 못해 단독 해법이 아니다.
  선택 메시 자체를 잡고 끄는 지면 드래그는 orbit 제스처와 모호해져 보류(후속 검토).
- 검증: 실브라우저 신뢰 마우스 드래그로 **보이는 몸통 중심**에서 90 px 끌기 —
  수정 전 NOMOVE, 수정 후 `arm_left [-0.5,0,0] → [-0.2415,0,-0.031]`(0.26 m MOVED),
  대조군 `witness_box` 0.19 m MOVED, pageErrors 0.

### 결정 2 — 얇은 메시 픽킹은 **월드 AABB 2차 패스**로 관대하게
- 로봇 링크는 얇아 화면 bbox 안을 클릭해도 광선이 링크 사이 빈틈을 지나간다(실측 명중률 26%,
  박스 80%). 빗나간 클릭은 그대로 **조용한 선택 해제**가 되어, 이어지는 방향키가 아무 일도
  하지 않는 실패 연쇄를 만들었다.
- 수정: `pickAt`의 메시 raycast가 실패하면 각 pickable의 월드 AABB와 광선을 교차시켜 가장
  가까운 것을 채택한다. AABB는 선택 아웃라인이 그리는 박스와 같으므로 "보이는 상자를
  클릭하면 잡힌다"와 의미가 일치하고, 완전히 빈 곳 클릭의 선택 해제 동작은 유지된다.
- 검증: 로봇 화면 bbox 격자 클릭 명중률 **26% → 100%**(121/121).

### 결정 3 — 로봇 루트에도 "진실로 되돌리는 지점"을 만든다 (비로봇과 대칭화)
- 발견된 비대칭: 로봇 빌드 레코드에는 `initialPose`가 없었고(`scene-loader` robot 분기),
  `SceneEditor.updateTransform`의 로봇 분기도 이를 갱신하지 않았다. 로봇의 **루트 배치는 물리가
  아니라 렌더 핸들(URDF outer 그룹)이 소유**하므로, 커밋되지 않은 드래그 프리뷰를 되돌릴 주체가
  코드 어디에도 없었다 — 프리뷰가 다음 `tickAll`에서 kinematic 링크 바디로 역류하고 `reset()`
  조차 복구하지 못했다(재현 테스트로 확인).
- 수정: (a) `buildRobotEntity`가 `initialPose`를 설정, (b) `updateTransform` 로봇 분기가 이를
  갱신, (c) `SceneHandle.reset()`의 로봇 분기가 `applyHome` 앞에 `setRootTransform(initialPose)`,
  (d) `SceneEditor.resyncTransform(id)` 신설 — spec을 바꾸지 않고 물리/시각만 spec으로 재수렴
  (검증·통지·undo 스냅샷 없음). main.ts의 `onDraggingChanged(false)`가 로봇에 대해 이를 호출해
  비로봇의 `sync.bind` 재바인딩과 **같은 자가 치유**를 만든다. 스케일 거부 경로도 이를 재사용한다.
- 왜 비로봇에는 resync를 걸지 않는가: 비로봇의 물리 pose는 살아있는 진실이다(낙하 중인 박스를
  spec 위치로 되돌리면 오히려 물리를 깬다). 로봇 루트는 편집으로만 바뀌므로 spec이 곧 진실이다.

### 결정 4 — 로봇 베이스 y는 UI가 0으로 클램프한다
- 로봇 링크는 kinematicPosition이라 바닥(fixed ENV)과 겹쳐도 물리가 밀어내지 못한다. 실측:
  y=−0.1로 커밋 후 2.5 s 재생·⏹ Stop 모두 −0.1000 유지(같은 조건의 dynamic 박스는 +0.0287로
  자가 교정). 즉 "오브젝트는 잘 옮겨진 것처럼 보이고 로봇만 이상해 보이는" 상태가 된다.
- 수정: 직접 조작 경로(기즈모 커밋·방향키 커밋·인스펙터 Transform 입력)에서 y를 0으로 클램프하고
  한국어 토스트로 이유를 알린다. `__sim.editor.updateTransform` 파사드는 자동화용이라 그대로
  통과시킨다(게이트가 임의 pose를 주입할 수 있어야 한다) — 이 비대칭은 의도된 것이다.
- 대안 기각: core(SceneEditor)에서 클램프하면 데이터 계층이 요청과 다른 값을 커밋해 파사드
  결정론이 깨진다. 규칙은 UI 정책으로 둔다.

### 결정 5 — 발견성·피드백 (실패 연쇄를 끊는 쪽)
- 우측 스택에 flex `order` 도입: **편집 폼이 항상 맨 위**. 로봇을 선택하면 관절 슬라이더 패널과
  인스펙터의 관절 표가 길어져 `ee-pos-x`까지 최대 865 px 스크롤이 필요했다(오브젝트는 0 px) —
  "로봇은 숫자로도 못 옮긴다"는 오해의 실제 원인이었다.
- 뷰포트 실행 오버레이에 `선택 <id>` 리드아웃 상시 표시(선택 상태를 알 수 있는 곳이 스크롤될 수
  있는 우측 패널뿐이었다). 선택 변경 시 `refreshOverlay()`로 rAF 지연 없이 갱신.
- 선택 없이 방향키를 누르면 한국어 토스트(2 s 스로틀). 이전에는 완전 무음이었다.
- 로봇 스케일 거부를 콘솔 로그뿐 아니라 토스트로도 표면화(다른 편집 실패는 이미 토스트였다).
- 엔티티 편집 폼의 로봇 안내 문구를 제약형 → 능력형으로("로봇도 위 Transform으로 옮기고 회전할
  수 있습니다 …").

### 결정 6 — 인스펙터 Transform 커밋의 대상 고정 (별건 잠재 결함)
- `commitPosition`/`commitRotation`이 `specOf()`(**현재 선택**)를 다시 조회했다. 입력을 커밋하지
  않은 채 다른 엔티티를 선택하면 `content.replaceChildren()`가 포커스된 입력을 DOM에서 떼면서
  지연 발화하는 native `change`가 **새로 선택된 엔티티**로 값을 커밋했다(엔티티 간 값 누출).
- 수정: 폼이 만들어질 때의 id를 클로저로 고정하고, 그 폼의 대상이 아니면 조용히 무시한다.

### 검증 (전부 실측)
- `npm run verify`: tsc --noEmit 통과 · ESLint 경고 0 · vitest **36 files / 758 tests 전부 통과**
  (기존 736 + 로봇 편집 계약 14 + 기즈모 앵커/픽킹 순수수학 8).
- `npm run build` 성공.
- 브라우저 게이트 **6종 ALL PASS**: two-arms(로봇 이동 3종 신규 어서션 포함) · arm ·
  arm-sequence · scene-builder · orchestration · planner.
- 임시 실브라우저 프로브(드래그·명중률)는 조사 후 삭제, 임시 디버그 훅도 원복(git status 확인).

### 영향 파일
`src/render/interaction.ts`(프록시 앵커 + 앵커 수학 순수 헬퍼 + AABB 픽킹 2차 패스 +
onNudgeBlocked + anchorProbe), `src/render/interaction.test.ts`, `src/core/scene-loader.ts`
(로봇 initialPose + reset 루트 복원), `src/core/scene-editor.ts`(로봇 initialPose 갱신 +
resyncTransform + placeRobotRoot), `src/core/scene-edit-types.ts`(resyncTransform 계약),
`src/core/robot-edit-contract.test.ts`(신규 — 수정 후 계약 고정), `src/main.ts`(드래그 훅 로봇
재수렴 · y 클램프 · 스케일 거부 토스트 · 방향키 안내 · 우측 스택 order · 선택 리드아웃 ·
anchorProbe 파사드), `src/ui/inspector/entity-editor.ts`(폼 대상 고정 + 안내 문구),
`src/ui/viewport/run-overlay.ts`(선택 세그먼트), `scripts/gate-browser.mjs`(two-arms 로봇 이동·
되감기 유지·앵커 위치 회귀), `docs/USAGE.md` §5.2/§5.3, `EXPERIMENTS.md`(이 항목).

## 2026-07-28 — 위 수정의 적대적 재검증 후속 (회전 궤도 회귀 · 픽킹 과포획 · 게이트 판별력)

앞 항목의 수정을 적대적 검증자들이 실브라우저로 재검증하면서 **그 수정이 만든 새 결함 2건과,
회귀 가드가 사실상 비어 있었다는 점**이 드러났다. 아래는 그 후속 결정이다.

### 결정 7 — 앵커는 "핸들 위치"일 뿐, 회전 피벗이 아니다 (궤도 회전 회귀 제거)
- 회귀: 기즈모를 시각 AABB 중심의 프록시 앵커에 붙인 뒤, 루트를 `root = anchor − R(anchorRot)·offset`
  으로 역산했다. three r169 `TransformControls`는 rotate 모드에서 `object.quaternion`만 바꾸므로
  (`TransformControls.js` pointerMove의 rotate 분기 — position은 절대 건드리지 않는다) 이전
  `attach(root)`에서는 회전이 위치를 바꾸지 않았는데, 역산 식에서는 앵커 회전만 바뀌어도 루트가
  앵커를 중심으로 **공전**한다. 변위 = 2·offset·sin(θ/2) → 로봇(offset 0.376 m) 기준 15° ≈ 0.10 m,
  90° ≈ 0.53 m. 실측: 회전 드래그 한 번에 베이스가 y=0.72 m까지 떠올랐다(바닥 클램프는 y<0만 막는다).
- 수정: 앵커 → 루트 전달을 **델타 전달**로 바꿨다.
  `rootPos = rootStart + (anchorNow − anchorStart)`, 회전/스케일은 앵커 값 복사.
  드래그 시작 시 앵커의 회전/스케일 = 루트의 월드 회전/스케일이므로 이는 `attach(root)`의 의미를
  정확히 보존한다 — "핸들만 다른 자리에 그린" 것과 같다. 회전 피벗은 루트 원점(로봇 베이스)이며,
  바닥에 고정된 팔의 자연스러운 피벗이기도 하다.
- 부수 효과: 앵커 오프셋을 로컬 좌표로 들고 다닐 이유가 사라져 `anchorOffsetLocal` /
  `anchorPositionFromRoot` / `anchorWorldDelta` / `rotateVec3ByQuat`을 제거하고, 앵커는 매 프레임
  시각 AABB 중심에 직접 놓는다(`placeAnchorAtVisualCenter`). 이로써 "선택 시 1회 고정" 때문에
  재생 중 핸들이 몸통에서 0.435 m 떨어지던 별건 결함도 함께 사라졌다(BoxHelper.update와 같은
  비용의 프레임 작업 1회 추가).

### 결정 8 — 월드 AABB 픽킹 2차 패스를 화면 좌표 여유(6 px)로 교체
- 회귀: `pickByBounds()`는 각 pickable의 **월드 AABB**와 광선을 교차시켰다. AABB는 3D 상자여서
  비스듬한 카메라에서 실루엣보다 훨씬 넓은 화면 영역을 덮는다 — 실측으로 캔버스 격자 1000점 중
  229점이 팔 하나에 흡수됐고, 두 팔 사이의 빈 공간에서 **엉뚱한 팔**이 선택됐다. "빈 곳 클릭 =
  선택 해제"(UX_DESIGN §3.3)라는 규범이 깨지고, 조준하지 않은 로봇이 방향키에 반응하게 된다.
- 수정: 정확 raycast가 빗나가면 `CLICK_TOLERANCE_PX`(6 px) 반경 8방위로만 재조준한다. 여유가
  화면 좌표에 갇히므로 실루엣 근처에서만 관대해지고 규범은 유지된다. 실측(수정 후, 캔버스 격자
  960점): null 932 · arm_left 15 · arm_right 12 · box 1 — 빈 곳은 전부 선택 해제.
- 기각한 대안: AABB 유지 + 거리 가중치(여전히 빈 하늘을 선택), 실루엣 마스크 렌더(비용 과다).

### 결정 9 — 브라우저 게이트에 "사용자의 실제 제스처"를 넣는다 (판별력 확보)
- 문제: 앞 항목의 신규 어서션 3건이 전부 판별력이 없었다. 앵커 어서션은 `anchorProbe().anchor`와
  `visualCenter`를 비교했는데 둘 다 같은 코드로 같은 박스에서 나온 값이라 **동어반복**이었고,
  나머지 2건은 방향키/파사드 경로만 타서 드래그 편집이 통째로 죽어도 초록불이었다.
- 수정(게이트 two-arms):
  1. `anchorScreenPoint()`(신규 검증 표면)로 **핸들 화면 좌표를 얻어 합성 마우스로 드래그**하고
     spec·물리 이동량을 잰다 — 사용자가 보고한 그 제스처다.
  2. 회전 링을 반경·방위로 훑어 충분한 회전(Δq > 0.1 ≈ 11.5°)을 만든 뒤 **위치가 안 변했는지**
     확인한다(공전 회귀가 남아 있으면 0.075 m 이상 밀린다 — 허용치 0.01 m).
  3. 되감기 어서션은 **시각 루트만 0.4 m 어긋뜨린 뒤** reset을 부른다(원 결함의 재현 조건).
  4. 앵커 어서션에 `attachedToAnchor`(gizmo.object === 앵커)를 추가 — 좌표만으로는 attach 대상이
     루트로 되돌아간 회귀를 잡을 수 없었다.
- 되돌림 실험으로 판별력을 확인했다(전부 실측):
  `attach(root)` 복원 → 3건 FAIL(드래그 이동 0 m) · reset 루트 복원 제거 → 1건 FAIL(drift 0.566 m) ·
  궤도 회전식 복원 → 1건 FAIL(posShift 0.159 m). 수정본은 12건 ALL PASS.

### 결정 10 — 우 패널 폭을 확정한다 (내용이 레이아웃을 밀지 않게)
- 원인: 워크스페이스 그리드의 열 5는 `auto` 트랙인데 `rightWrap`에 width가 없어 **스택 내용의
  max-content**가 열 폭이 됐다(좌 패널은 `paintLeftCollapse`가 처음부터 width를 준다 — 우측만
  누락). 로봇 안내 문구를 한 줄 늘리자 폭이 247 → 750 px로 벌어졌고, 그리드는 좁아졌는데 캔버스는
  resize 통지를 못 받아 패널 아래로 밀려 들어가 그 띠에서 라이브러리 드롭이 조용히 무시됐다.
- 수정: `rightWrap`에 `WORKSPACE_SIZES.right.defaultPx`(UX §2 "~280px")를 처음부터 부여한다.
  폭 변경은 스플리터만 한다. 안내 문구도 짧게 줄이고 `whiteSpace: normal`을 명시했다.
  실측(수정 후, 1600×950): 선택 없음/로봇/오브젝트 **모두 280 px**, 캔버스 가림 0 px.

### 결정 11 — 관절 슬라이더 패널을 로봇 구성 변화에 재구성한다
- 문제: 패널이 빌드 시점 스냅샷이라 라이브러리로 추가한 로봇은 슬라이더가 없었다 — 사용자
  시나리오("빈 씬 → 팔 2대 → 관절로 충돌")의 **종착점이 UI만으로는 막혀 있었다**.
- 수정: `editor.onChange`에서 로봇 id 목록 시그니처가 바뀌면 패널을 다시 만든다. 슬라이더 초기값은
  core 진실(`readJoints`)에서 읽으므로 재마운트로 값이 튀지 않고, flex `order` 덕에 DOM에 나중에
  붙어도 스택 위치는 그대로다. 실측: 빈 씬 → 카드 2회 실제 드래그앤드롭 → 슬라이더 0 → **16개**,
  그 슬라이더 DOM 조작으로 관절이 구동되고 충돌 로그가 쌓인다(pageErrors 0).

### 검증 (전부 실측)
- `npm run verify`: tsc --noEmit 통과 · ESLint 경고 0 · vitest **36 files / 759 tests 전부 통과**.
- `npm run build` 성공.
- 브라우저 게이트 **7종 ALL PASS**: two-arms(12건 — 신규 제스처 어서션 포함) · arm ·
  arm-sequence · scene-builder · orchestration · planner · scene-switch.
- 임시 실브라우저 프로브는 저장소 밖(scratchpad)에서 실행하고 삭제했다.

### 영향 파일
`src/render/interaction.ts`(델타 전달 · 앵커 매 프레임 갱신 · 6 px 재조준 픽킹 ·
`anchorScreenPoint`/`attachedToAnchor`), `src/render/interaction.test.ts`(궤도 회전을 계약으로
고정하던 테스트를 **회전은 위치를 바꾸지 않는다**로 뒤집음 + ndcToClient/여유 반경),
`src/ui/workspace.ts`(우 패널 폭 확정), `src/ui/inspector/entity-editor.ts`(안내 문구 축약 + 줄바꿈),
`src/main.ts`(관절 패널 재구성 · anchorScreenPoint 파사드), `scripts/gate-browser.mjs`(two-arms
제스처/회전/되감기 판별력), `docs/USAGE.md` §5.2, `EXPERIMENTS.md`(이 항목).

---

## Phase 11 — 제품화 (Studio Hardening) · 2026-07-28

`docs/UX_AUDIT.md`(5인 디자인 팀 진단, 통합 이슈 C-1~C-18)의 실행. **새 능력을 추가하지 않고**
이미 있는 능력이 신뢰·보존·발견되게 만든다. 진단 근거 수치는 전부 실행 중인 앱을 Playwright로
계측한 것이며 증거는 `docs/ux-audit/`에 있다.

### 결정 1 — 공간 예산을 상수에서 정책으로 (C-1)

- 문제: 세로 크롬(바 44 + 독 211 + 그래프 241 + 스플리터 10 = **506px**)과 가로 크롬(**530px**)이
  창 크기와 무관한 상수라, 뷰포트 면적이 `(W−530)×(H−506)`이라는 결정론적 함수였다. 실측 4개 지점이
  이 식과 정확히 일치했다(1080−506=574 / 900−506=394 / 768−506=262 / 720−506=214). 1366×768에서
  뷰포트는 화면의 **20.9%**, 3.19:1 레터박스였다. 스플리터를 최대로 끌어도 41.5%가 상한이었다 —
  기본값 문제가 아니라 **하한값 정책 문제**였다.
- 수정: 세로 크롬을 `clamp()`로, 중앙을 `grid-template-rows: minmax(240px,1fr) auto var(--rsw-flow-h)`로
  바꿔 **뷰포트 하한을 그리드 차원에서 보장**한다. 독을 기본 접힘으로 두되 탭바에 진행 스트립을 남겨
  정보 손실을 0으로 만들었다. 하한 재산정(dock 120→112, flowGraph 200→148, right 220→200).
- 트레이드오프: `flowGraph.minPx` 200 → 148은 Phase 7의 "그래프가 항상 보인다" 요구를 약화시키는
  것처럼 보이지만, 그 요구의 **의도**는 strip 모드(56px)가 더 싸게 만족시킨다.
- 실측(전/후 뷰포트 점유율): 1920 38.5→**48.8%** / 1600 29.3→**40.7%** / 1440 27.7→**38.5%** /
  1366 20.9→**33.2%** / 1280 17.4→**29.9%** / 1024 16.5→**37.2%** / 768폭 뷰포트 240px→**696px**.

### 결정 2 — 커맨드바 겹침은 세 줄의 문제였다 (C-2)

- 문제: 겹침이 미관이 아니라 **기능 소실**이었다. 겹친 영역에서 DOM 뒤 형제가 포인터를 가로채,
  1280×720에서 되돌리기/다시하기 버튼(폭 28px)이 100% 피복되어 클릭 불가였고 자연어 입력창과
  생성 버튼이 재생 버튼 뒤로 사라졌다. 실측 겹침: 1920→1 / 1600→8 / 1440→13 / 1366→17 /
  1280→20 / 1024→24.
- 원인: `scene-controls.ts`의 세 줄. `left`/`right`가 `flexShrink:0`인데 `center`만 `minWidth:0` +
  **`justifyContent:'center'`** 라, center 컨텐츠가 좌우 **대칭으로** 넘쳐 양쪽 이웃 위에 얹혔다.
  `bar`에 `overflow` 미지정이라 밖으로 그려지고 `.ui-btn`은 `nowrap`이라 줄바꿈도 없었다.
- 수정: `justify-content: flex-start` + `bar{overflow:hidden}` + 세 슬롯 모두 `flexShrink:1/minWidth:0`.
  이것만으로 **겹침이 물리적으로 불가능**해진다. 그 위에 2행 구조(UX_DESIGN §2 다이어그램이 이미
  그렇게 그려 놓았다) + P0~P6 우선순위 오버플로 + 아이콘 전용 모드를 얹었다.
- 실측: 전 해상도에서 실제 겹침 **0**(남은 1건은 라벨이 자기 체크박스를 감싼 부모-자식 포함).

### 결정 3 — 문서(Document) 모델 도입 (C-3)

- 문제: 작업물이 3중으로 소실됐다. (a) 저장이 `SceneSpec`만 직렬화해 **시퀀스를 버렸다** —
  업로드는 `{scene, sequence}` 봉투를 받는데 저장은 봉투를 만들지 않는 비대칭. (b) `localStorage`/
  `sessionStorage`가 완전히 비어 있어(플래너 설정 1건 제외) 새로고침 한 번에 전부 휘발.
  (c) `confirm(`/`beforeunload`/dirty 표시가 repo 전체 **0건**.
- 수정: `src/ui/document.ts` — `{version, name, scene, sequence, assets}` 봉투 + IndexedDB 디바운스
  자동저장 + `DirtyTracker`(직렬화 비교, 상태 변화 시에만 통지) + `beforeunload`(미저장일 때만).
  확장자를 `.workcell.json`으로 바꿔 "이 파일이 전부다"를 이름이 약속하게 했다.
- 하위 호환: `parseDocument`가 세 형식(문서 / 구 봉투 / SceneSpec 단독)을 전부 받는다.
  `SceneSpec`에는 `scene` 필드가 없어 모호하지 않다.

### 결정 4 — 되돌릴 수 있다는 확신은 탐색의 전제조건이다 (C-4)

- 문제: (a) `editor.removeEntity`가 core에 완전 구현돼 있는데 호출부가 `window.__sim` 자동화
  파사드뿐이라 **오브젝트를 지울 UI가 없었다**(add-only 함정). (b) `SceneHistory`가 `SceneSpec`만
  저장해 플로우 그래프 편집(노드 삭제·재정렬·복제·파라미터·교체 생성)이 전부 복구 불가였다.
  `Ctrl+Z`가 "때때로만 동작하는" 것은 아예 없는 것보다 나쁘다 — 잘못된 안전감을 준다.
- 수정: 스냅샷 타입을 `HistorySnapshot = {scene, sequence}`로 넓히고 `commitFlowSequence`에서도
  `noteChange`를 부른다. 복원 경로(`restoreSceneFromHistory`)가 이미 `{scene, sequence}` 봉투를
  받고 있어 배선 비용이 작았다. 엔티티 삭제는 인스펙터 헤더 버튼 + 토스트 실행취소 액션.

### 결정 5 — 전역 키 리스너 5개를 단일 라우터로 (C-6)

- 문제: `window`에 keydown을 거는 곳이 5군데였고 각자 방어했다. **Space를 3개가** 나눠 가졌고
  (재생·충돌로그 행·그래프 팬), 가드가 `BUTTON`을 제외하지 않아 포커스된 버튼의 Space 활성화가
  파괴되고 충돌 로그 행을 키보드로 여는 순간 시뮬이 재생됐다. **방향키를 2개가** 나눠 가져
  스플리터 조절과 3D nudge가 동시에 일어났다. `docs/UX_DESIGN.md` §9 규정 8종 중 스펙대로 동작하는
  것은 2종뿐이었고 좌우 방향키는 스펙(Step)과 **반대 기능**에 배선돼 있었다.
- 수정: `src/ui/shortcuts.ts` — 스코프(`data-shortcut-scope`) 판정 + **위젯 소유권 규칙**
  (버튼의 Space/Enter, 목록·분할자의 방향키는 위젯의 것이다)을 한곳에 두고, 모든 전역 단축키가
  이 라우터를 통과한다. `playback.ts`는 `togglePlay()`만 제공하고 키 바인딩을 갖지 않는다.
- 도움말 시트는 **`router.list()`가 반환하는 실제 등록 바인딩만** 그린다 — 문서를 베끼면 구현되지
  않은 키를 광고하게 되고 그건 도움말이 없는 것보다 나쁘다.

### 결정 6 — 액센트를 주황에서 바이올렛으로, 역할을 3분할 (C-11/C-14)

- 문제: (a) `bgApp`와 `bgPanel`의 대비비가 **1.001:1**, `borderSoft`와 `bgRaised`가 **동일 헥스**
  (#22252b)라 표면 고도가 색으로 존재하지 않았다. 반투명(0.93)인데 blur가 없어 슬라이드 패널이
  하위 콘텐츠를 글자 단위로 통과시켰다. (b) 액센트 한 색이 **8가지 의미를 겸직**했고,
  `.ui-btn--accent`가 `background`를 건드리지 않아 재생 버튼이 나머지 트랜스포트와 같은 시각 무게였다.
- 수정: `SURFACE` 6단계 사다리(각 단계 ≥1.14:1, 계산 검증) + `BORDER` 4단계(항상 얹힌 표면보다
  밝다) + 액센트 역할 3분할(면=주요 액션 / 보더=토글 / **선택은 별도 축 `SELECT` 스카이블루**).
- **액센트 색 변경 근거**: 구 주황(#e67e22)은 "로봇 팔 색과 일관"을 의도했으나, 뷰포트 안의 따뜻한
  오브젝트와 크롬이 같은 온도라 크롬이 앞으로 나왔다. 차가운 바이올렛(#7C6AF6)은 보색 대비로 크롬을
  뒤로 물리고, 3D 선택(청색)과도 명확히 갈린다. 청색 선택은 Blender/Onshape/Isaac Sim 공통 관행이다.
- 부작용 처리: 가장 밝은 `SURFACE.modal` 위에서도 `text`/`label`/`muted`가 전부 AA를 넘도록
  재산정했다(muted 4.68:1이 하한). 값 변경 시 재계산 필요 — 각 토큰 주석에 검산값을 남겼다.

### 결정 7 — snake 레이아웃의 perRow는 폭이 아니라 fit 줌으로 고른다 (C-10)

- 문제: `ZOOM_MIN=0.4` + 노드 피치 224px 때문에 "맞춤"이 전체를 담지 못하는 지점이 1366폭에서
  **9노드**였다(데모 시퀀스가 이미 7노드). snake 도입 후 `nodesPerRow`가 **폭만** 봐서 836×240
  페인에서 k=2가 뽑혔고, 7노드가 4줄로 접혀 세로가 구속 조건이 되며 fit 줌이 **33%** 로 떨어졌다 —
  단일 행(47%)보다 나빴다.
- 수정: `bestPerRow(n, paneW, paneH)`가 `k=1..n` 각각의 fit 배율을 계산해 argmax를 취한다.
  **단일 행(k=n)이 후보에 포함되므로 결과가 단일 행보다 나쁠 수 없다** — 이 회귀가 구조적으로
  재발 불가능해진다. 클램프 전 원값으로 비교해야 상·하한 동률이 argmax를 무의미하게 만들지 않는다.
- 실측(836×240): 7노드 33%→**55%**(LOD full 복귀) · 12노드 28%→55% · 30노드 12%→33% ·
  80노드 12%→23%. LOD full이 유지되는 상한이 n=7 미만에서 **n=12**까지 올라갔다.

### 결정 8 — 충돌을 화면에 등장시킨다 (C-7)

- 문제: `waitForCollision`을 포함한 7/7 시퀀스가 완주해도 **충돌이 있었다는 표시가 화면 어디에도
  없었다**(`docs/ux-audit/07-sequence-done-1920.png`). 독 탭에 배지 API 자체가 없었고, 비시각
  사용자에게는 시각 신호 3종(로그 행·펄스·마커)뿐이라 아무 신호도 만들지 않았다(WCAG 4.1.3 실패).
  빨강이 **6개 값 · 4개 hue**로 흩어져 로그 행과 3D 마커를 눈으로 잇는 것이 색으로 뒷받침되지 않았다.
- 수정: 독 탭 카운트 배지 + 상태줄 `충돌 N` 상시 표시(0건 포함 — "감지가 돌고 있다"가 정보다) +
  첫 충돌 1회 토스트 + `COLLISION` 램프 단일화.
- **로그 스트림에 `aria-live`를 걸지 않는다**: 물리 스텝마다 이벤트가 쏟아지면 polite 큐가 포화되어
  사용자가 다른 조작을 해도 몇 분간 과거 충돌만 읽고, 링버퍼의 앞 행 제거도 변경으로 집계돼 중복
  발화한다. 대신 3초 스로틀 **요약**을 별도 live 영역으로 발화한다.

### 결정 9 — 실행은 레이아웃의 1급 상태다 (C-9)

- 문제: `docs/UX_DESIGN.md` §2가 "실행 중 뷰포트 우선 확장"을 규정했는데 엔진 상태에 반응하는
  레이아웃 코드가 0줄이었다. Idle과 Running 화면의 시각 차이가 7px 점의 펄스 + 칩 하나의 보더 +
  텍스트뿐이었다.
- 수정: `FlowMode: 'full' | 'strip' | 'off'`. 재생이 그래프를 56px 노드 스트립으로 접고 정지가
  복원한다 — §1의 동시 가시성을 버리지 않으면서 관찰이 필요한 순간에 뷰포트가 넓어진다.
  `body[data-run-state]`로 커맨드바 하단에 액센트 진행 스트립을 띄워 주변시로도 잡히게 했다.

### 결정 10 — 아웃라이너를 좌 패널로, 프로퍼티는 우 패널에 (C-16)

- 문제: 280px 한 컬럼에 카드 3개가 쌓이고 그중 하나가 아웃라이너와 프로퍼티를 동시에 담아,
  Transform이 편집 가능/읽기 전용으로 **중복** 표시되고 빈 상태 문구가 2개 동시 노출됐다.
  `main.ts` 주석이 대가를 자백하고 있었다: "로봇 선택 시 ee-pos-x까지 최대 865px 스크롤".
- 수정: 목록을 `mountSceneOutliner`로 분리해 좌 패널 하단에 배치(좌 패널 콘텐츠는 y≈540에서 끝나고
  **329px가 비어 있었다** — 추가 화면 비용 0). 프로 3D 툴은 예외 없이 둘을 분리한다(Blender
  Outliner ≠ Properties, Unity Hierarchy 좌 / Inspector 우).
- 함정: `paintLeft()`가 펼침 시 `display = ''`로 되돌려 인라인 `flex`를 지웠다 — 좌 슬롯의 세로
  스택이 무너져 아웃라이너가 패널 밖으로 밀려 잘렸다. `''` 대신 `'flex'`로 명시해야 한다.

### 결정 11 — 제품 셸 (C-12)

- 문제: `<title>`이 패키지 slug, 파비콘 없음, 실사용 `<h1>` 0개(유일한 h1은 **부팅 실패 오버레이
  전용**), 랜드마크 0개. 빈 플로우가 **이미 출시된 기능**을 "Phase 9에서 제공됩니다"라고 안내해,
  첫 사용자가 가장 도움이 필요한 순간에 거짓 정보로 이탈시켰다.
- 수정: 제품명 **Workcell**(`ui/brand.ts` 단일 진실) · SVG 파비콘 · 동적 `document.title`(미저장 시
  점 접두) · 워드마크 `<h1>` 승격 · `header/nav/main/aside/section` 랜드마크 · 스킵 링크 ·
  `?` 도움말 시트. **사용자 노출 문자열에 내부 로드맵 어휘(Phase N) 금지**를 규칙으로 세웠다.
- 네이밍 근거: **workcell**은 "로봇+지그+부품이 한 셀에 놓인 작업 단위"를 뜻하는 로보틱스 표준
  용어로 `SceneSpec`이 기술하는 대상과 1:1 일치한다. 상위 브랜드는 워크스페이스 경로의 NextWave가 흡수.

### 결정 12 — 폰트 자체 호스팅 (C-14)

- 문제: `FONT.ui` 스택에 Noto Sans KR이 있었지만 **어디서도 로드하지 않았다**(`@font-face` 0건).
  Windows에서 라틴은 Segoe UI, 한글은 Malgun Gothic으로 갈려 x-height·수직 메트릭이 어긋났고,
  같은 UI가 OS마다 다른 타이포그래피를 가졌다.
- 수정: Pretendard Variable(한글+라틴 통합 메트릭, OFL) **동적 서브셋 92개**를 `public/fonts/`에
  자체 호스팅. 전체 3.1MB지만 브라우저는 실제 쓰는 unicode-range만 받는다(한국어 UI 통상 100~250KB).
  수치 리드아웃은 JetBrains Mono(92KB) + **`tabular-nums` 고정** — 구 시스템은 `tabular-nums`가
  전 코드베이스 0회라 매 프레임 갱신되는 simTime·관절값이 자릿수 폭 지터를 냈다.
- 대안 기각: 전체 variable 단일 파일(2.06MB)은 첫 로드 비용이 커서, "설치 없음"이 셀링 포인트인
  제품에 맞지 않는다.

### 진행 방식

기반 모듈(theme/icons/a11y/workspace/shortcuts/document/brand/help-sheet)을 먼저 세우고, 그 위에서
UI 디렉터리 5곳(인스펙터 / 플로우그래프 / 독 / 커맨드바 / 라이브러리·피드백·뷰포트)을 **병렬로**
이행한 뒤 `main.ts`에서 통합했다. 병렬 작업자는 `main.ts`·기반 모듈을 건드리지 않고 배선 요청을
코드로 제출하는 규약을 썼다 — 공유 파일 충돌 없이 5개 디렉터리를 동시에 옮길 수 있었다.

### 검증 (전부 실측)

- `tsc --noEmit` 통과 · ESLint 오류 0 · vitest **41 files / 978 tests 전부 통과**(Phase 10 대비 +219).
- `npm run build` 성공.
- 브라우저 게이트 **6종 ALL PASS**: orchestration · planner · flow-graph · arm-sequence ·
  scene-builder · scene-switch.
- 해상도 7종 실측(전/후)은 위 결정 1·2에 기재. 페이지 오류 전 해상도 0건.
- `<h1>` 1개 · 랜드마크 `header:1 nav:1 main:1 aside:1 section:5` · `title="arm-and-boxes — Workcell"`.
- 증거: `docs/ux-audit/`(진단 전 22장) + `docs/ux-audit/after/`(수정 후 7장) + `measurements.json`.

### 남은 것 (Phase 11 범위 밖으로 명시)

- **물리 스텝 시간(ms/frame)**: Stats HUD가 자리를 잡아 뒀으나 `core/engine`이
  `EngineTickInfo.physicsMsPerFrame`을 노출하기 전까지 대시(—)로 표시된다. 계층 규칙상 `ui`가
  측정할 수 없다.
- **3D 배치 고스트**: 드롭 힌트(가장자리 하이라이트 + "여기에 놓기")까지만 구현했다.
  UX_DESIGN §3.3의 "바닥 레이캐스트 지점 반투명 고스트"는 `render`에
  `beginPlacementPreview/updatePlacementPreview/endPlacementPreview`가 추가되어야 완결된다.
- **문서 라이브러리 UI**: `document.ts`에 IndexedDB 문서 컬렉션(`putDocument`/`listDocuments`)이
  구현돼 있으나 씬 select에 "내 문서" optgroup을 붙이는 UI는 미구현이다.
- **레이아웃 프리셋 UI**: `workspace.applyPreset('default'|'sceneBuild'|'flowEdit'|'runObserve')`가
  구현돼 있으나 메뉴 노출이 없다.
- **URL fragment 공유 링크**: 정적 호스팅에서 가능한 최대치로 백로그에 남긴다.

### 영향 파일

**신규**: `src/ui/icons.ts` · `a11y.ts` · `shortcuts.ts` · `document.ts` · `brand.ts` · `help-sheet.ts` ·
`viewport/stats-hud.ts` · `viewport/drop-hint.ts` · `public/favicon.svg` · `public/fonts/**` ·
`docs/UX_AUDIT.md` · `docs/ux-audit/**`.
**전면 개편**: `src/ui/theme.ts` · `workspace.ts` · `command-bar/scene-controls.ts` ·
`flow-graph/canvas.ts` · `inspector/inspector.ts` · `dock/dock.ts` · `index.html`.
**수정**: 나머지 `src/ui/**` 전량 · `src/main.ts` · `src/ui/history.ts`(HistorySnapshot) ·
`src/render/renderer.ts`(frameObject/resetCamera) · 대응 테스트 전량.

---

## 2026-08-03 — 사용자 보고 3건: 지하 배치 · 조작 발견성 · 유령 선반

사용자 보고를 그대로 옮기면 이렇다.

1. "물체를 넣을 때 바닥 아래 지하로 위치되는 경우가 계속 생겨서 사물이 사라지는 불편함이 있어."
2. "물체를 눌렀을 때 어떻게 방향키로 움직일 수 있는지, 너비 조정을 할 수 있는지가 좀 더 편한
   UX로 되었으면 좋겠어. 타 시뮬레이터를 참조하여 제공해줘."
3. "픽앤플레이스에서 사물은 집는데 옆에있는 선반에 부딪히는데 그냥 충돌 표시 안 뜨고 관통해."

셋을 조사하다 **네 번째, 보고되지 않은 결함**이 나왔다(아래 결정 3).

### 결정 1 — 지하 배치는 실수가 아니라 **작업물 손실**이다 (`core/ground-clamp.ts` 신설)

바닥은 상면이 정확히 y=0인 두께 0.1 m 고정 슬래브다(`scene-loader` GROUND_*). 그 아래로
내려간 사물은 되돌아올 길이 없다:

- **static/fixed**는 물리가 밀어내지 않는다 — 지하에 영구히 박힌다.
- **dynamic**도 슬래브(-0.1 m) 밑으로 내려가면 접촉이 성립하지 않아 무한 낙하한다 = "사라진다".
- **robot** 링크는 kinematicPosition이라 자가 교정이 아예 없다.

그래서 §2.11("파괴적 동작에는 되돌릴 경로")의 **예방 쪽 짝**으로 하한을 건다. 클램프 기준은
중심이 아니라 **collider의 실제 최저점**이다 — 회전한 박스의 모서리도, 치수를 키운 뒤의
아래쪽 절반도 바닥을 뚫지 않는다. 지지함수(support)를 로컬 up 축 `u = Rᵀŷ` 하나로 계산한다
(box/sphere/capsule/cylinder는 원점 대칭이라 위·아래를 한 번에 얻는다).

측정할 수 없는 형상(convexHull/trimesh/fromVisual)은 **원점을 하한**으로 삼는다. 임포트
메시의 피벗은 bbox 바닥 중심이고(`render/mesh-import.ts`) 로봇 루트 원점은 베이스 링크
발밑이므로, 두 경우 모두 "원점 y ≥ 바닥"이 우연이 아니라 정확한 접지 규칙이다.

**적용 범위를 편집 경로로 한정한다** — 기즈모 커밋 · 방향키 · 인스펙터 입력 · 라이브러리
배치 · 치수 편집. 씬 JSON 로드는 손대지 않는다: 수작성 좌표를 조용히 고쳐 쓰면 파일과
화면이 어긋나고 "데이터가 진실"(§2.5)이 깨진다. `environment.ground !== true`인 씬에서는
설 자리가 없으므로 클램프하지 않는다. `__sim.editor` 파사드는 자동화용이라 기존 정책대로
통과시킨다.

부수 결정: 치수를 키우면 **중심을 올려 바닥에 계속 놓이게** 한다(`groundedTransformForShape`).
중심 고정으로 키우면 아래쪽 절반이 그대로 바닥을 뚫는데, 대부분의 씬 편집기가 "바닥에 놓인
사물은 커져도 바닥에 놓여 있다"로 동작하는 이유가 이것이다. 치수+재안착을 한 `runEdit`에
묶어 undo 한 번으로 되돌아간다.

복구 경로로 **`End` = 바닥에 붙이기**를 추가했다(Unity/Unreal의 snap-to-floor). 클램프와
달리 떠 있는 사물을 내리기도 하고, 이미 지하에 박힌 기존 씬의 구제 수단이 된다.

한 번 잡은 자체 결함: `entityDropBelowOrigin`의 누적 초기값을 0으로 두면 **원점보다 위에
있는 collider**(offset이 큰 경우)가 0으로 잘려, 원점을 지하에 둬도 되는 배치를 막게 된다.
`-Infinity`에서 시작하도록 고쳤다(테스트가 잡았다).

### 결정 2 — 조작 발견성: 선택 HUD (Blender N-패널 관례)

방향키 이동(5cm / Shift 1cm / PageUp·Dn)과 치수 편집은 **이미 구현돼 있었다.** 문제는 그
사실이 화면 어디에도 없었다는 것이다 — 오른쪽 인스펙터를 펼쳐 Dimensions까지 스크롤한
사용자만 발견했다. 구현된 기능이 발견되지 않으면 없는 것과 같다.

참조한 관례:

- **Blender N-패널(Item 탭)** — 선택 대상의 트랜스폼과 Dimensions를 **뷰포트 안에서** 바로
  고친다. HUD의 골격으로 삼았다(시선을 3D에서 떼지 않고 크기를 조정한다).
- **Unity/Isaac Sim** — W/E/R 모드 전환이 뷰포트 툴바에 상주. 좌상단 기즈모 바가 이미
  담당하므로 HUD는 중복하지 않는다.
- **Unity/Unreal snap-to-floor** — `End`(결정 1).

`ui/viewport/selection-hud.ts`: 우하단에 대상 id·유형·위치·**치수 ± 스테퍼**·바닥에 붙이기·
키 안내. 설계 원칙 셋:

- 선택이 없으면 **그리지 않는다**. 빈 상태 카드를 띄우면 3D 화면의 1/6을 상시 잠식한다.
- 치수 스테퍼는 **프리미티브에만**. 로봇·임포트 메시·다중 collider 엔티티는
  `updateDimensions`가 거부하므로 실패할 버튼을 보여주지 않는다.
- 값은 **읽기 전용 리드아웃**. 정밀 입력·스크럽은 인스펙터가 소유한다 — 같은 편집기를 두
  곳에 두면 포커스/커밋 경합이 생긴다. HUD는 "한 칸씩" 담당이다.

### 결정 3 — ★ 보고되지 않았던 결함: `→`가 Step과 이동을 **동시에** 일으켰다

보고 2를 조사하다 발견했다. `render/interaction.ts`가 `window`에 keydown을 직접 걸어
W/E/R·방향키를 처리하고 있었다 — **불변식 §2.10의 정면 위반**이고, 라우터가 모르는 두 번째
키맵이었다. 라우터의 `playback.step`(ArrowRight)이 `preventDefault()`를 불러도 다른 리스너는
막히지 않으므로, `→` 한 번이 **시퀀스를 한 노드 진행시키면서 동시에 선택 오브젝트를 5cm
움직였다.** 아이러니하게도 `shortcuts.ts`의 파일 헤더가 "방향키를 2개가 나눠 가졌다"를 이미
과거형으로 기록하고 있었는데, 그 중 하나가 살아남아 있었다.

수정: `interaction`은 **명령만 제공**하고(`setMode` / `nudgeSelected`) 키 매핑은 전부 라우터
등록표로 옮겼다. 소유권은 **스코프가 가른다**:

- 3D 뷰포트에 포커스 + 선택 있음 → 이동
- 그 밖(또는 선택 없음) → 규정대로 Step (UX_DESIGN §9)

이를 성립시키려면 뷰포트가 포커스를 받아야 했다. 슬롯에 `tabIndex=-1` + `pointerdown` 포커스
훅을 달았다 — "3D 화면을 클릭했다 = 여기서 작업한다"는 선언으로 본다(포커스 링은 그리지
않는다: 클릭할 때마다 화면 전체가 테두리로 둘러싸이면 시각 소음이다).

부수 변경: 한 조작이 여러 키로 들어오는 경우(←→↑↓, Shift 변형)를 위해 `ShortcutBinding`에
`hidden`/`keysDisplay`를 추가했다. 라우터는 키 하나당 바인딩 하나를 요구하므로 별칭이
필연적인데, 전부 나열하면 도움말 시트가 같은 조작 8줄로 덮인다. 대표 한 줄만 보이고
`keysDisplay`가 키 묶음을 표기한다 — "실제 등록된 것만 그린다"는 계약은 유지된다.

### 결정 4 — 유령 선반: 하나의 상자가 두 역할을 겸했다

`pick-and-place`의 `drop_zone`은 **도착 감지**와 **선반**을 겸하고 있었다. 감지를 위해
`isSensor: true`(물리 반응 없음 = 관통)였고, `collidesWith: ["OBJECT"]`라 **로봇과는 쌍조차
성립하지 않아** 팔이 지나가도 이벤트가 0건이었다. 화면엔 단단한 상자가 보이는데 물리적으로는
존재하지 않았던 셈이다. 실측: 카고 최종 위치 y=0.024가 상자 내부(y 0–0.12)였다.

역할을 나눴다:

- `drop_zone` — 감지 존 (sensor, SENSOR_ZONE, 바닥 패드, `tags: ['detection-zone']`)
- `drop_shelf` — **실체 있는 선반** (ENV 고체, `collidesWith: [ROBOT, OBJECT]`, emitEvents)

선반 위치는 실측으로 잡았다. 팔의 최심 링크 바디 중심이 z=-0.149(t=6.4s)까지 들어오고
collider가 ~0.03 더 뻗으므로, 선반 전면을 z=-0.21에 두어 3cm 여유를 남겼다. 카고는
z=-0.053에서 멈춰 감지 존 안에 안착한다(센서 start @4.79s).

**샘플 시퀀스는 선반을 스치지 않아야 한다**는 것도 계약으로 넣었다. 예제가 매 실행 충돌을
보고하면 "정상"의 기준선이 무너져 진짜 사고가 소음에 묻힌다(`collision-classify.ts` 헤더의
논지와 같다). 게이트가 양쪽을 다 본다: 정상 실행에서 `arm×drop_shelf` 0건이고, 일부러
과잉 스윙시키면 접촉이 보고된다(실측 start 4건, 배지 "충돌 4", 토스트 노출).

kinematic 링크가 fixed collider를 **시각적으로** 통과하는 것 자체는 Rapier의 성질이라 남는다
(EXPERIMENTS Phase 4 CCD 항목과 같은 계열). 바뀐 것은 그것이 이제 **보고된다**는 점이다.

### 결정 5 — 예약 태그 `'detection-zone'` (테스트가 공허하지 않게)

결정 4를 회귀 테스트로 굳히려다 **테스트가 공허하다**는 것을 뮤테이션으로 확인했다.
"비-sensor 정적 장애물은 ROBOT과 쌍이 성립해야 한다"는 규칙은, 장애물을 sensor로 만들면
검사 자체를 건너뛴다 — 원래 결함의 모양 그대로다. 그룹(SENSOR_ZONE)을 함께 옮기면 어떤
규칙도 걸리지 않았다.

의도를 적을 지점이 필요했다. `EntitySpec.tags`에 예약 태그 `'detection-zone'`
(`DETECTION_ZONE_TAG`)을 정의하고, 규칙을 양방향으로 세웠다:

- 태그가 붙은 엔티티의 collider는 **반드시** sensor + SENSOR_ZONE
- 태그가 **없는** fixed 엔티티의 collider는 sensor이면 안 되고, 로봇이 있는 씬이라면
  `collidesWith`에 ROBOT + `emitEvents: true`

재-뮤테이션으로 검증: `drop_shelf`를 다시 sensor로 되돌리면 3개 테스트가 실패하고
(`sample-scenes` 1 + `phase6-scenes` 2), 원복하면 통과한다.

### 검증

- `npm run verify` — typecheck · ESLint 0 · **테스트 1094개 통과**(46 파일).
  신규: `core/ground-clamp.test.ts` 25 · `core/sample-scenes.test.ts` 52 ·
  `ui/viewport/selection-hud.test.ts` 16.
- `npm run gate` — **12종 ALL PASS**: build · orchestration · planner · flow-graph ·
  scene-builder · scene-switch · **viewport-edit(신규)** · **samples(신규 6종)**.
- 신규 게이트 `--expect=viewport-edit` 9항목: 선택 HUD 표시 · 방향키 이동(0.050m) ·
  Shift 미세(0.010m) · **방향키가 Step을 동시에 일으키지 않음(★회귀)** ·
  **PageDown ×12에도 지하 불가(★회귀, y=0.030 유지)** · End 바닥에 붙이기(0.180→0.030) ·
  HUD ± 치수 조정 후 접지 유지 · 선택 해제 시 HUD 숨김 · 선택 없으면 →는 Step.
- `gate:samples`는 제공 예제 6종을 전부 돈다: falling-boxes · arm-sequence · pick-and-place ·
  obstacle-avoidance · collision-testbed · two-arms.
- `two-arms` 게이트의 기존 방향키 항목 2개는 뷰포트 포커스를 명시하도록 고쳤다 — 실제
  사용자는 3D 화면 클릭으로 얻는 조건을, 파사드 `select()`를 쓰는 게이트가 갖추게 한 것이다.
- 게이트 플레이크 1건 제거: nudge 기준점이 **살아있는 pose**여서(기즈모 드래그와 같은 커밋
  경로) 재생 중이면 표본 사이에 바디가 굴러 이동량이 흔들렸다(0.010 ↔ 0.040 실측).
  측정 전 `engine.pause()`로 고정 — 실제 UI도 편집 시작 시 자동 일시정지하므로 같은 조건이다.

### 영향 파일

**신규**: `src/core/ground-clamp.ts` · `src/core/ground-clamp.test.ts` ·
`src/core/sample-scenes.test.ts` · `src/ui/viewport/selection-hud.ts` ·
`src/ui/viewport/selection-hud.test.ts`.
**수정**: `src/main.ts`(클램프 배선 · 단축키 등록표 · HUD 글루) ·
`src/render/interaction.ts`(window keydown 제거 → `nudgeSelected` 명령) ·
`src/ui/shortcuts.ts`(`hidden`/`keysDisplay`) · `src/ui/help-sheet.ts`(별칭 숨김) ·
`src/ui/workspace.ts`(뷰포트 포커스) · `src/ui/library/templates.ts`(감지 존 태그) ·
`src/schema/types.ts`(`DETECTION_ZONE_TAG`) ·
`src/assets/scenes/pick-and-place.scene.json`(선반/감지 존 분리) ·
`src/assets/scenes/collision-testbed.scene.json`(태그) ·
`src/core/phase6-scenes.test.ts` · `scripts/gate-browser.mjs` · `package.json`(게이트 스크립트) ·
`CLAUDE.md` · `docs/UX_DESIGN.md` · `docs/DATA_MODEL.md`.

**남은 것**: `arm-touch-box`·`two-arms-collision` 시퀀스는 밀기(push) 방식이라 그리퍼가
사물을 실제로 **파지**하지는 않는다 — 접촉 물리 정밀도가 필요한 시점의 MuJoCo 교체
경로(CLAUDE.md §7)와 같은 주제로 남긴다.

### 추가 (같은 날) — 게이트가 잡은 다섯 번째 결함: 편집 기준점이 stale했다

위 `viewport-edit` 게이트를 전체 실행에 넣자 `Shift 병용이 미세 이동` 항목이 **간헐적으로**
실패했다. 실측이 0.010m ↔ 0.040m를 오갔다. 처음엔 재생 중 바디가 굴러서라고 보고
`engine.pause()`를 넣었는데도 재현됐다 — 가설이 틀렸다는 뜻이다.

0.040 = 0.050 − 0.010. 이 숫자가 원인을 가리켰다. **두 번째 이동이 첫 번째 이동 전
pose에서 출발**한 것이다.

`RenderSync.commit()`은 prev 스냅샷만 갱신하고 Object3D는 렌더 루프의 `apply(alpha)`가
쓴다. 그런데 편집 중에는 엔진이 일시정지돼 있어 다음 렌더 프레임이 언제 올지 편집 코드가
알 수 없고, 기즈모 앵커와 방향키 이동은 **시각 노드의 월드 행렬**을 기준점으로 삼는다
(`render/interaction.ts`의 `decomposeSelectedRoot`). 프레임이 사이에 끼면 1cm, 안 끼면
4cm — **같은 조작이 프레임 타이밍에 따라 다른 결과**를 냈다. 플레이크의 정체가 곧 결함이었다.

수정: `SceneEditorImpl`의 teleport 경로를 `syncVisualsNow()`(= `commit()` + `apply(1)`)로
모았다. 물리 → 시각 단방향이므로 §2.1은 그대로다. `RenderSyncLike`에 `apply`를 추가했다.

회귀 테스트(`scene-editor.test.ts`): `apply()`를 **부르지 않고** 연속 두 번 `updateTransform`
하고, 시각 노드가 매번 즉시 새 pose를 갖는지 본다. `apply(1)`을 빼면 실패한다(뮤테이션 확인).

교훈: **간헐적으로 실패하는 게이트를 "플레이크"로 분류하기 전에 숫자를 읽는다.**
0.040이 0.050 − 0.010이라는 사실이 원인을 정확히 지목했고, 관용 오차를 늘렸다면
사용자가 겪는 실제 비결정성을 테스트로 덮을 뻔했다.

---

## 2026-08-03 — 컨베이어 벨트: 새 엔티티 종류 + conveyor-pick-place 예제

요청: "컨베이어 벨트 오브젝트를 하나 만들어주면 좋겠어. 그 위에 물건들이 계속 오고, 로봇이
그걸 픽앤플레이스 하는 예제도 하나 추가."

CLAUDE.md §6의 "새 엔티티 종류를 추가할 때 (예: 컨베이어)" 워크플로가 가리키던 바로 그
케이스라, 이 작업이 그 절차의 레퍼런스가 되도록 기록한다.

### 결정 1 — 벨트는 움직이지 않는다. **표면 속도**만 흐른다

실제 컨베이어는 프레임이 고정이고 벨트 표면만 흐른다. Rapier에는 surface velocity 개념이
없어서 셋 중 하나를 골라야 했다:

| 방법 | 판정 |
|-----|-----|
| 벨트 바디를 `kinematicVelocity`로 굴린다 | **탈락** — 벨트가 씬 밖으로 날아간다 |
| 무한궤도를 여러 조각 바디로 만들어 돌린다 | **탈락** — 접촉 수 폭발 + 조각 경계마다 걸림 |
| 접촉 중인 동적 바디를 직접 구동한다 | **채택** |

채택안은 매 물리 스텝 **직전**(preStep)에 `world.contactPairsWith`로 접촉 목록을 물리
엔진에 묻고(narrow-phase가 진실 — AABB 추정이 아니다) 그 바디들의 속도를 지정한다.

`PhysicsWorld`에 4개를 더했다: `dynamicBodiesTouching` · `dynamicBodies` ·
`getLinearVelocity` · `setLinearVelocity`. **Rapier 타입은 여전히 core 밖으로 나가지
않는다**(§7) — MuJoCo 교체 시 `core/conveyor.ts`는 한 줄도 바뀌지 않는다.

### 결정 2 — ★ 진행축만 지정한다 (벨트가 로봇을 이기면 안 된다)

첫 구현은 수평 속도를 통째로 벨트 속도로 덮어썼다. 결과: **팔이 2초를 밀어도 사물의 z가
5 mm밖에 움직이지 않았다.** 측면 성분이 매 스텝 0으로 지워져 240 Hz로 되돌려진 것이다.
화면상 로봇은 밀고 있는데 물건은 벨트를 따라 그냥 흘러갔다.

`drivenVelocity`가 축을 셋으로 나눈다: **진행축 = 벨트가 지정 / 측면 = 보존 / 수직 = 보존**.
실제 컨베이어도 진행 방향으로만 끌고, 옆으로 밀면 옆으로 간다. 이 분해 없이는 컨베이어
위에서 로봇이 할 수 있는 일이 없다.

### 결정 3 — "물건이 계속 온다"는 **스폰이 아니라 재순환**이다

런타임 엔티티 생성은 §2.5(SceneSpec이 진실)와 정면으로 부딪힌다 — 스펙에 없는 엔티티가
씬에 생기고, 충돌 로그·Undo·저장·인스펙터가 그것을 설명할 수 없다. 대신 씬에 선언된 N개가
벨트 끝에서 시작점으로 돌아간다. 사용자에게 보이는 결과는 같고 데이터 모델은 온전하다.

되돌리기는 세 조건을 **모두** 만족할 때만이다:

1. 진행축으로 끝을 확실히 지났다 (`RECYCLE_OVERSHOOT_M` — 립에서 떨리지 않게)
2. 측면으로 벨트 폭 근처다 (옆으로 굴러 나간 사고품은 되돌리지 않는다 — 사용자가 봐야 한다)
3. 벨트 상면보다 크게 높지 않다 — **3번이 없으면 로봇이 집어 든 물건이 손에서 사라진다**

되돌리기는 preStep에서 일어나므로 뒤따르는 `sync.commit()`이 새 pose를 prev로 잡는다 —
화면에 순간이동 궤적이 그려지지 않는다(engine.ts tick 순서가 이걸 공짜로 만들어 준다).

### 결정 4 — ★ 포토아이 게이트: 로봇을 라인 밖에서 기다리게 한다

첫 예제는 팔이 벨트 위 픽 지점에서 `waitForCollision(arm, item_a)`로 기다렸다. 실측 결과
두 가지가 깨졌다:

- 대기 6초 동안 **다음 물건(item_b)이 팔에 부딪혔다** — 선언하지 않은 접촉이라 `unexpected`
  = 충돌로 집계됐다. 샘플이 매 실행 충돌 2건을 보고하면 "정상"의 기준선이 무너진다.
- 팔이 벨트 높이까지 내려가 있어 **`arm × belt` 접촉**도 났다.

두 번째는 자세 높이를 실측으로 찾아 해결했다(아래 결정 5). 첫 번째는 **설계를 바꿔야**
했다: 벨트 위 픽 지점에 **포토아이(`pick_gate`, sensor)** 를 두고, 시퀀스가 그 감지를
기다린다. 팔은 라인 **위**에서 대기하다가(물건이 그 아래로 자유롭게 통과) 감지 즉시
0.5초 만에 내려온다. 실제 컨베이어 셀의 동작이기도 하고, 결과적으로 배리어 2단이 된다:

```
waitForCollision(item_a × pick_gate)  →  하강  →  waitForCollision(arm × item_a)  →  스윕
```

`happenedSince`가 kind 필터 없이 모든 접촉을 보므로 sensor 이벤트로도 배리어가 풀린다 —
이미 있던 계약이 새 용도를 그대로 지원했다.

### 결정 5 — 자세는 추측하지 말고 실측한다

`arm × belt` 접촉을 없애려고 λ(approach→lowered 보간)를 0.0~1.0으로 훑어 브라우저에서
실제 접촉 쌍을 관측했다:

| λ | 그리퍼 y | item 접촉 | belt 접촉 |
|---|---------|----------|----------|
| 0.0 | 0.180 | ✗ | ✗ |
| 0.4 | 0.142 | ✗ | ✗ |
| **0.6** | 0.123 | **✓** | ✗ |
| **0.8** | 0.104 | **✓** | ✗ |
| 1.0 | 0.085 | ✓ | **✓** |

λ=0.7을 채택(깨끗한 구간 0.6~0.8의 중앙). 벨트 중심 z도 그리퍼 대기 z(0.143)에 **정렬**
했다 — 3.7 cm 어긋나 있을 때는 물건이 그리퍼 옆면을 스치며 벨트 반대편으로 빗겨 나갔다.

### 결정 6 — ★ 빌드 시점에 캐시한 기하는 편집 경로마다 다시 만들어야 한다

벨트 기하(진행축·반길이·상면 높이)는 성능을 위해 `ConveyorBinding` 생성 시점에 고정한다.
그런데 `SceneEditor.updateTransform`은 **재빌드하지 않고 teleport만 한다** —
`updateDimensions`/`updatePhysics`/`renameEntity`가 `rebuildInPlace`로 바인딩을 새로
만드는 것과 달리, 기즈모로 벨트를 옮기면 **화면의 벨트는 옮겨졌는데 사물은 옛 자리의
진행축·끝점으로 실려 가는** 상태가 된다. 눈으로는 원인을 알 수 없는 종류의 결함이다.

`refreshConveyor`(remove → registerConveyor)를 `updateTransform`에 걸어 막았다.
`ConveyorRegistry.add`가 중복 등록을 던지므로 **remove가 먼저**여야 한다 — 테스트가 그
순서를 고정한다.

### 결정 7 — 데이터가 표현할 수 없는 조합은 이유와 함께 거부한다

`validateScene` 교차 규칙 5개를 넣었다. 전부 **조용한 오작동**을 막기 위한 것이다:

| 규칙 | 어기면 |
|-----|-------|
| `type: 'static'` | 벨트 자신이 물리에 밀려 씬을 떠난다 |
| `bodyType: 'fixed'` | 같음 |
| `colliders[0]`가 box | 길이/폭을 몰라 재순환 지점을 계산할 수 없다 |
| `colliders[0].isSensor !== true` | 접촉이 성립하지 않아 실어 나를 대상이 영원히 0개 |
| `direction`의 수평 성분 ≠ 0 | 정규화 불가 — 조용히 "속도 0인 벨트"가 된다 |

조용한 no-op은 사용자가 시행착오로 배우게 만든다. 사람이 읽을 한국어 이유를 준다.

한 번 물린 함정: `superRefine((conveyor, ctx) => { const [x,,z] = ... })`에서 **`z`가 zod의
`z`를 가렸다.** 컴파일러가 `Property 'ZodIssueCode' does not exist on type 'number'`로
잡았다 — 이 모듈에서 좌표를 `z`로 구조분해하면 안 된다.

### 결정 8 — 테스트가 잡아야 할 것을 잡는지 확인했다

`sample-scenes.test.ts`에 **"컨베이어가 있으면 그 위에서 시작하는 동적 사물이 최소 1개"**
계약을 넣었다. 이건 가정이 아니라 실제로 낸 실수다 — 처음에 `item_b`를 벨트 시작점보다
앞(x=-0.02)에 두어 그 물건이 바닥에 떨어진 채 한 번도 실려 가지 않았다. 판정에는
`core/conveyor.ts`의 **검증된 순수 기하를 그대로** 쓴다(테스트가 기하를 재구현하면 구현과
테스트가 같이 틀릴 수 있다).

뮤테이션 검증: 아이템을 벨트 밖으로 옮기면 이 테스트가 실패하고, 원복하면 통과한다.

### 검증

- `npm run verify` — typecheck · ESLint 0 · **테스트 1150개 통과**(47 파일).
  신규 `core/conveyor.test.ts` 31개(순수 기하 + 실 Rapier 이송/재순환/편집 정합).
- `npm run gate` — **13종 ALL PASS** (신규 `gate:conveyor-pick-place` 포함, samples 7종).
- 신규 게이트 13항목 실측:
  - 벨트 이송 0.237 m / 3s (선언 0.1 m/s — 안착 슬립 포함)
  - 재순환 2회 / 12s (런타임 스폰 없이)
  - 포토아이 감지 5회 · `arm × item_a` 접촉 2회 · **`item_a × drop_zone` 감지 1회**
  - **`arm × belt` 0건** (정상 시퀀스는 벨트를 건드리지 않는다)
  - 픽 이후에도 라인 계속 회전 (item_b 0.520 → 0.320)
  - 인스펙터 Conveyor 섹션은 벨트에만 표시
  - **파사드 편집(방향 반전 + 증속)이 실제 거동을 바꾼다** — 역방향 0.228 m / 2s
- 상태줄 **충돌 0건** — 샘플이 정상 기준선을 오염시키지 않는다.

### 영향 파일

**신규**: `src/core/conveyor.ts` · `src/core/conveyor.test.ts` ·
`src/assets/scenes/conveyor-pick-place.scene.json` ·
`src/assets/sequences/conveyor-pick-place.sequence.json`.
**수정**: `src/schema/types.ts`(ConveyorSpec) · `src/schema/validate.ts`(zod 미러 + 교차 규칙) ·
`src/core/types.ts`·`world.ts`(PhysicsWorld 4개 확장) · `scene-loader.ts`(ConveyorRegistry ·
registerConveyor) · `scene-editor.ts`(updateConveyor · refreshConveyor) · `scene-edit-types.ts` ·
`src/main.ts`(preStep tickAll · SCENE_REGISTRY · 인스펙터/파사드 배선) ·
`src/ui/library/templates.ts` · `src/ui/icons.ts`(conveyor 아이콘) ·
`src/ui/inspector/entity-editor.ts`(Conveyor 섹션 · bindCheckbox) ·
`src/core/sample-scenes.test.ts` · `phase6-scenes.test.ts` · `scripts/gate-browser.mjs` ·
`package.json` · `CLAUDE.md` · `docs/DATA_MODEL.md` · `docs/USAGE.md` · `README.md`.

**남은 것**: 그리퍼는 여전히 실제로 **파지하지 않는다**(밀기). 컨베이어 예제의 "픽"도
벨트 밖으로 쓸어내는 동작이다 — 접촉 물리 정밀도가 필요한 시점의 MuJoCo 교체 경로
(CLAUDE.md §7)와 같은 주제로 남긴다.

### 후속 — 적대적 리뷰가 잡은 결함 10건 (같은 날)

구현 후 4개 축(물리 정확성 / 스키마·계약 / UI 통합 / 예제·테스트)으로 결함을 찾고, 각
후보를 **서로 다른 두 렌즈**(실제 재현되는가 / 다른 계층이 이미 막는가)로 반박 시도했다.
28건 중 **12건이 두 렌즈 모두를 통과**(16건 반박). 중복 제거 후 10건을 고쳤다.

| # | 결함 | 왜 위험한가 |
|---|-----|-----------|
| 1 | `sync.bind` 성공 후 `registerConveyor`가 던지면 **바인딩이 남는다** | 제거된 바디를 가리키는 바인딩 → 다음 `commit()`이 죽은 핸들로 `getPose` → 씬 정지. `bind`가 마지막 문장이던 시절엔 없던 경로다 |
| 2 | `physicsSpecSchema`가 `colliders: []`를 허용 → 벨트 교차 규칙이 **조기 반환** | 검증을 통과한 뒤 로더가 던진다. main.ts는 이미 이전 씬을 dispose한 뒤라 **잘 돌던 씬을 잃는다** |
| 3 | `registerConveyor`가 `colliders[0].offset`을 버린다 | 벨트 판이 원점에서 떨어져 있으면 상면 높이·끝점이 어긋나 재순환이 조용히 안 된다 |
| 4 | 재순환이 **월드의 모든 동적 바디**를 후보로 훑는다 | 벨트가 둘이면 다른 벨트 위 사물을 자기 시작점으로 순간이동시킨다(실측: 첫 tick에 1.28 m 끌려옴) |
| 5 | 재순환 높이에 **하한이 없다** | 끝에서 바닥까지 떨어진 사물이 벨트 높이로 되돌려져 판에 박힌다 |
| 6 | `SceneHandle.reset()` 후 접촉 그래프가 **stale** | teleport만 하고 스텝을 안 돌리므로 다음 preStep이 리셋 전 접촉을 본다 → 공중 26 cm 사물에 0.1 m/s 주입. **되감기 재생 ≠ 최초 재생**(결정론 위반) |
| 7 | 겹친 벨트가 속도를 **합성**한다 | 앞 벨트의 구동이 뒤 벨트의 "보존할 측면 성분"으로 오인된다(실측: 0.25 벨트 둘 → 0.336 m/s) |
| 8 | 12 cm "들어 올림" 가드가 낮다 | 10 cm만 들어 옮기는 정상 이송도 가드 안이라 **로봇 손에서 물건을 빼앗는다**(실측: 1.08 m 순간이동) |
| 9 | 인스펙터가 **현재 선택**에 커밋하고, 거부된 값을 폼에 되찍는다 | 편집 중 다른 엔티티를 고르면 그쪽에 컨베이어 블록이 써진다 / 검증 거부 후 폼과 스펙이 어긋난다 |
| 10 | 포토아이 어서션이 **공허**했다 | `item_a`가 `pick_gate`와 겹친 채 스폰돼(x 0.275~0.325 vs 0.255~0.285) t=0에 센서 start가 발생 — **컨베이어를 통째로 지워도 통과**했다 |

주목할 만한 수정 둘:

**8번의 해법을 높이가 아니라 시간으로 바꿨다.** "로봇이 들고 있는가"를 높이 임계값으로
맞히려는 것 자체가 틀렸다 — 정상 이송 높이와 낙하 높이는 겹친다. 대신 **경과 시간**으로
가른다(`RECYCLE_GRACE_TICKS`): 벨트에서 굴러 떨어진 사물은 접촉이 끊긴 직후 수 tick 안에
끝을 지나지만, 로봇이 집어 옮기는 사물은 한참 뒤에 지난다. 두 상황이 실제로 다른 축이다.

**7번은 tick마다 "이미 구동된 바디" 집합을 공유해 중재한다.** 먼저 선언된 벨트가 이긴다 —
삽입 순서 = 씬 선언 순서이므로 결정론적이다.

반박된 것 중 기록할 가치가 있는 것: "픽 배리어 timeout 6초가 짧다"는 주장은 **실측
0.63~0.65초(10회, 9.2배 여유)**로 반박됐고, "item_b 후미 추돌이 위상 의존"이라는 주장은
시퀀스가 포토아이에 **위상 고정**돼 있어(벽시계가 아니라 벨트 위상에 동기) 13회 실행이
모두 동일하다는 실측으로 반박됐다.

### 검증 (최종)

- `npm run verify` — typecheck · ESLint 0 · **테스트 1154개 통과**(47 파일).
  `core/conveyor.test.ts` 35개 — 위 4·6·7·8번 회귀를 **뮤테이션으로 검증**(결함 4종을
  동시에 재주입하면 정확히 그 4개 테스트가 실패하고, 원복하면 통과).
- `npm run gate` — **13종 ALL PASS**.
- 벨트 실제 속도는 선언값의 85~90%다(실측 0.25 → 0.221). 속도 지정이 스텝 **직전**이고
  그 뒤 접촉 솔버가 정지한 벨트 면과의 마찰로 되잡기 때문이다 — 선언값은 **상한**으로
  읽는 것이 정확하다. 테스트도 비율 밴드로 고정했고 `conveyor.ts` 헤더에 명시했다.

**알려진 한계 (문서화하고 남긴다)**: 겹친 벨트는 "먼저 선언된 쪽이 이긴다"로 중재할 뿐
전이(transfer) 물리를 모사하지 않는다. 그리퍼는 여전히 파지하지 않는다(밀기) — 접촉 물리
정밀도가 필요해지면 MuJoCo 교체 경로(CLAUDE.md §7)와 함께 다룬다.

---

## 2026-08-03 — 감지 존을 "통과 가능해 보이게" + 픽앤플레이스를 실제 파지로

사용자 보고 둘:

1. "컨베이어의 칸막이? 문? 과 드랍존이 그냥 녹색 직사각형으로 보여서 관통하는 것처럼 보여.
   좀 더 모양이 있게 해서 사용자가 구분이 되게 해줘."
2. "픽앤플레이스에서 상자를 잡고 조금 위로 올려서 플레이싱 하도록 수정해줘."

### 결정 1 — 통과 가능한 것은 **통과 가능해 보여야** 한다

이건 앞선 "유령 선반" 결함의 **시각 쪽 나머지 절반**이다. 그때는 단단해 보이는 상자가
물리적으로 없었던 것이 문제였고, 지금은 **정말로 통과 가능한 것(sensor)이 단단해 보이는**
것이 문제다. 방향은 반대지만 원인은 같다: 화면이 물리와 다른 말을 한다.

`VisualSpec`에 순수 표현 필드 둘을 추가했다 — 물리에 영향하지 않는다:

| 필드 | 역할 |
|-----|-----|
| `opacity` (0..1) | 반투명 — "여기는 부피지 벽이 아니다" |
| `edges` | 모서리 선 — 반투명 면만으로는 배경에 묻히는 경계를 세운다 |

렌더 구현에서 챙긴 것 셋:

- **`depthWrite: false`** — 켜 두면 반투명 부피의 뒷면이 앞면을 지워 속이 빈 껍데기가
  되고, 존 안에 든 사물이 **사라진 것처럼** 보인다.
- **`side: DoubleSide`** — 안쪽에서 봐도 부피로 읽힌다.
- **`castShadow = false`** — 통과 가능한 부피가 그림자를 드리우면 다시 단단해 보인다.

`disposeMeshResources`도 함께 고쳤다: 모서리 선은 `LineSegments`라 `isMesh`가 false여서
기존 순회가 건너뛰었다 — 씬을 전환할 때마다 감지 존 개수만큼 GPU 자원이 샜을 것이다.

### 결정 2 — 게이트는 "문"이 아니라 **광전 센서**로 보이게

사용자가 "칸막이? 문?"이라고 물은 것 자체가 답이다 — 형태가 역할을 말하지 못했다.
색과 형태를 **역할별로 갈랐다**:

- **포토아이 게이트**: 호박색(#e0a33e) 반투명 빔 + 양옆에 **센서 기둥 2개**.
  기둥은 `physics` 없는 **순수 장식**이다(§2.1 시각 전용 예외) — 물리를 주면 로봇
  작업 반경에 새 장애물이 생겨 정상 예제가 매 실행 충돌을 보고하게 된다.
- **도착 감지 존**: 청록색(#2fbf8f) **바닥 패드**(두께 1.2 cm). 세로 빔과 형태가
  겹치지 않아 한눈에 구분된다.

`collision-testbed`의 `slide_gate`와 라이브러리 `Sensor Zone` 템플릿에도 같은 규약을
적용했다 — 사용자가 새로 놓는 감지 영역도 처음부터 통과 가능해 보인다.

### 결정 3 — 파지는 되지만 **마찰 창(window)이 좁다**

"밀기"를 "집어 올려 옮기기"로 바꿨다: 파지 → **상승** → 이송 → **하강** → 놓기 (9 → 11 step).

놀랍게도 이 프로젝트의 그리퍼로 **실제 파지가 된다**. 손가락은 `finger_*_joint`
0→0.03 프리즘 관절이고, URDF 기하상 손가락 안쪽 간격은 `2×(0.008+v) − 0.012`다.
상자 폭 0.05에 대해 `state 0.6`(v=0.018)이면 간격 0.040 — **1 cm를 무는** 셈이다.

그런데 마찰을 올릴수록 좋아지지 않았다. 실측 스윕:

| gripper state | cargo 마찰 | 스윙 | 상승 | 착지 | 배리어 |
|---|---|---|---|---|---|
| 0.70 | 0.5 | 1.6s | 4.2cm | 미끄러짐 (z=0.079) | 빠름 |
| 0.65 | 1.2 | 2.2s | **0** (안 들림) | — | 빠름 |
| 0.65 | 1.8 | 2.6s | 4.2cm | 존 안 | **timeout 6s** |
| 0.60 | 1.0 | 2.6s | **0** | — | timeout |
| **0.60** | **0.8** | **3.0s** | **4.2cm** | **존 안(z=0.018)** | **빠름** |

마찰 상한이 존재하는 이유가 흥미롭다: 마찰이 너무 높으면 첫 접촉에서 상자가 곧바로
물려 **접촉이 다시 시작되지 않는다.** 그러면 `waitForCollision`이 기다리는 새 start
이벤트가 영영 오지 않아 배리어가 timeout 6초를 그대로 버린다(실측 elapsed 16.0s vs
정상 10.5s). 즉 **파지 강도가 동기화 기구를 굶긴다** — 물리와 제어 흐름이 얽힌 지점이다.

채택: `state 0.60 / 마찰 0.8 / 스윙 3.0s`. 3회 반복 실행이 **밀리미터까지 동일**
(상승 4.2cm, 착지 [0.367, 0.024, 0.018], 경과 10.55~10.63s). 감지 패드는 실측 착지점에
맞춰 재배치했다 — 가장자리에 걸치지 않게.

### 검증

- `npm run verify` — typecheck · ESLint 0 · **테스트 1158개 통과**(47 파일).
- `npm run gate` — **13종 ALL PASS**.
- 신규 게이트 어서션 **`상자를 집어 올려서 옮긴다 ★ (끌기 아님)`**: 재생 내내 cargo의 y를
  표본해 정착 높이 대비 **+4.3 cm** 상승을 확인한다. 최종 pose만 보면 끌고 간 것과 들고 간
  것을 구분할 수 없다(둘 다 바닥에서 끝난다) — 그래서 궤적의 최고점을 잰다.
- 신규 데이터 계약: 파지→상승→이송→하강→놓기 **순서**와 상승/하강 목표의 대소 관계,
  cargo 마찰 하한(배리어를 굶기지 않는 상한의 근거를 주석에 명시).
- 감지 존 시각 계약: 두 존 모두 `opacity < 1` + `edges` + **서로 다른 색**.
  기둥 장식은 `physics`가 없어야 한다(로봇 경로에 장애물을 추가하지 않는다).

**남은 것**: 파지가 성립하는 마찰 창이 좁고 비단조적이다(1.0에서 안 들리고 0.8·1.8에서
들린다). 접촉 물리 정밀도가 필요한 시나리오를 더 얹으려면 MuJoCo 교체 경로
(CLAUDE.md §7)를 먼저 밟는 편이 낫다 — 지금 값은 **이 씬에 맞춘 실측 튜닝**이다.

### 후속 — 드랍존을 "떨어진 자리"가 아니라 "놓는 자리"에 맞춘다

사용자 지적: "실제로 로봇팔이 내려놓는 곳이 아니라 중간에 미끄러져서 내려오게 되어 있고
거기가 드랍존으로 되어 있어서 이상해."

정확한 지적이었다. 앞선 튜닝에서 나는 **순서를 거꾸로 했다** — 상자가 이송 중 미끄러져
떨어진 지점(z≈0.018)에 감지 존을 맞춰 놓고, 게이트가 초록불이니 됐다고 봤다. 게이트가
통과한 이유도 명확하다: `상승 4.2cm`도 `drop_zone 감지 1건`도 **미끄러진 경우에 모두
참**이다. 두 어서션 어디에도 "로봇이 놓았는가"를 묻는 것이 없었다.

### 실측 — 언제 놓쳤나

매 tick 상자와 그리퍼를 추적했다:

```
t=4.775 step=7 cargo=[0.338,0.067,0.128] grip=[0.367,0.127,0.116]   ← 들고 출발
t=5.900 step=7 cargo=[0.360,0.027,0.020] grip=[0.385,0.127,0.009]   ← 놓침 (1.3s 만에)
t=8.442 step=9 cargo=[0.353,0.024,0.000] grip=[0.381,0.085,-0.030]  ← 놓기 동작
```

3.0초 이송의 **1.3초 만에** 손에서 빠졌고, 그리퍼는 z=-0.138까지 계속 갔다. 상자가 멈춘
z=0.02와 로봇이 놓으려던 z=-0.138 사이에 **16cm의 간극**이 있었다.

### 원인은 마찰이 아니라 **호(arc)의 길이**였다

마찰·파지 강도를 여러 조합으로 훑었지만(0.5~1.8 × state 0.55~0.70) 어느 것도 끝까지
버티지 못했다. 정작 효과가 있었던 변수는 **이송 호의 길이**다:

| joint1 목표 | 놓는 순간 그리퍼-상자 거리 |
|---|---|
| 0.3 (원래) | **15.6 cm** — 중간에 빠짐 |
| 0.1 | 4.1 cm |
| **0.0** | **3.8 cm** — 끝까지 물고 감 |

호가 길수록 접선 가속이 커져 손가락 마찰을 이긴다. 짧은 호에서는 gap이 이송 내내
**3.0~3.8cm로 일정**했다 — 미끄러진 것이 아니라 **실제로 물고 이동**한 것이다:

```
t=4.775 gap=0.031   t=5.875 gap=0.031   t=6.975 gap=0.035   t=7.708 gap=0.038
```

### 적용

- 이송 `joint1: 0.3 → 0.0` (마찰 0.8·파지 state 0.60은 유지 — 배리어를 굶기지 않는 값)
- `drop_zone`을 **로봇이 실제로 놓는 자리**로 이동: `[0.354, 0.006, 0.0]`
  (상자 최종 정착 위치 = 그리퍼 놓기 위치의 3.8cm 이내)
- `drop_shelf`도 함께 앞으로: 팔의 최소 도달 z가 -0.038(collider 포함 ≈ -0.07)로 얕아졌으므로
  전면을 -0.12에 두어 5cm 여유를 남겼다 — 패드 바로 뒤에 선반이 서는 그림이 된다

### 이 결함을 잡는 어서션을 추가했다

`놓는 순간 상자가 아직 그리퍼에 있다 ★ (미끄러져 떨어진 것 아님)` —
놓기 step(index 9)의 첫 tick에 상자와 그리퍼의 **수평 거리**를 재서 8cm 이하를 요구한다.
상승량·센서 감지로는 절대 구분되지 않는 것을 이 한 값이 가른다.

뮤테이션 검증: 이송 호를 0.3으로 되돌리면
`거리 15.6cm — 이송 중 손에서 빠졌다`로 실패하고(선반 충돌도 함께 3건 발생), 원복하면 통과한다.

**교훈**: 게이트가 초록이어도 **무엇을 묻지 않았는지**를 봐야 한다. "결과가 존 안에 있다"와
"로봇이 그것을 했다"는 다른 질문이고, 후자를 묻지 않으면 데이터를 실패에 맞추게 된다.

### 검증

- `npm run verify` — **테스트 1158개 통과**, typecheck · ESLint 0
- `npm run gate` — **13종 ALL PASS**
- pick-and-place 게이트 실측: 상승 4.4cm · 놓기 거리 3.8cm · 존 감지 1건 · `arm×drop_shelf` 0건 ·
  배리어 이벤트 해제(10.47s < 13s)

---

## 컨베이어 라인 — 이송 중 낙하와 "세 개 연속 안착" (2026-08-04)

**사용자 보고**: "컨베이 픽앤플레이스는 여전히 가다가 떨어뜨린다. 그리고 컨베이어에서
나오는 상자 한 세 개 정도 연속으로 드랍존에 안착시키는 것까지를 실행 플로우에 담아줘."

### 낙하의 원인은 마찰이 아니라 **이송 호 길이**였다

컨베이어 시퀀스는 pick-and-place와 달리 파지 후 곧바로 크게 회전했다. 마찰을 1.6까지
올려도 결과가 바뀌지 않아 호를 바꿔가며 실측했다:

| 이송 호 (rad) | 결과 |
|---|---|
| 0.21 – 0.30 | 3사이클 모두 파지 유지 (놓기 거리 2.9–9.1 cm) |
| 0.395 이상 | 이송 중 낙하 — 사용자가 본 그 현상 |

평행 손가락은 접촉 마찰만으로 버티는데, 회전이 길수록 원심 성분이 **오래** 걸려 상자가
손가락 사이로 빠져나간다. 마찰을 더 올려도 소용없는 이유다(그리고 마찰 과다는 배리어를
굶긴다 — pick-and-place 항목 참조). 상한을 `CPP_MAX_TRANSPORT_ARC_RAD = 0.3`으로 고정했다.

### 3사이클로 확장하며 나온 것들

- **파지 후 상승 → 회전, 놓기 후 상승 → 복귀.** 낮게 쓸고 지나가면 라인에서 대기 중인
  상자를 쳐낸다. 든 상자 바닥(0.068)이 대기 상자 윗면(0.080)보다 낮은 것은 상승량(4.2cm)이
  상자 높이(5cm)보다 작아 **구조적으로 불가피**하므로, 빠른 이송(ω 0.25)으로 접촉 시간을 줄였다.
- **5 cm 상자.** 4 cm로 줄이면 그리퍼가 상자 위를 스쳐 파지 자체가 실패한다.
- **드랍존은 점이 아니라 레인이다.** 세 개를 같은 지점에 놓으면 나중 상자가 먼저 놓인
  상자를 +x로 민다(놓기 지점 x≈0.335 → 밀려서 0.596). 존을 놓는 한 점에 맞추면 정상
  실행인데 "안착 실패"가 된다. 실측 안착 범위 전체를 덮는 적재 레인으로 만들었다.
- **정지판(bin_stop)은 오히려 나빴다.** 벽을 세우면 상자가 튕겨 라인 뒤(z 0.252)까지
  날아간다. 담아 두려다 흩뜨렸다 — 제거했다.

### 게이트가 초록인데 거짓이었던 두 지점

1. **관측 단계가 씬을 뒤흔든 채로 재생했다.** 벨트 관측(3s) + 재순환 관측(12s) 동안 상자들이
   몇 바퀴를 돌아 선입선출 순서가 깨진 상태에서 시퀀스를 시작하니 step 12에서 45초를
   기다리다 죽었다. 재생 전에 `orchestrator.stop()`으로 **씬·player·충돌 이력을 함께**
   되감는다. 이력을 되감지 않으면 관측 단계에서 생긴 게이트 이벤트가 시퀀스의 증거로 오독된다.
2. **측정 좌석이 방향과 어긋났다.** 역방향 편집 검증에서 상자를 벨트 가운데에 앉히면
   1초 관측 창 안에 반대쪽 끝을 넘어간다(0.34 → 0.09 < 벨트 시작 0.12). 좌석은 측정하려는
   **방향에 맞춰** 잡아야 한다.

### 뮤테이션 검증 (데이터 계약 6종 — 전부 실패 확인 후 원복)

| 되돌린 결함 | 잡은 어서션 |
|---|---|
| 이송 호 0.395 | `이송 호(joint1 변화량)가 파지가 버티는 범위 안에 있다` |
| 파지 직후 상승 제거 | `각 사이클이 파지→상승→이송, 놓기→상승→복귀 순서를 지킨다` |
| 놓기 후 상승 제거 | 같음 |
| 아이템 마찰 0.5 | `item_*: … + 파지 마찰` |
| 드랍존을 한 점으로 축소 | `drop_zone: 세 상자가 밀려 쌓이는 레인 전체를 덮는다` |
| item_c 제거 | `라인 위 아이템 수가 시퀀스 사이클 수와 같다` 외 6건 |

마지막 항목은 처음에 **suite 수집 실패**로 나타나 나머지 28개 테스트를 통째로 침묵시켰다
(모듈 스코프에서 시퀀스를 검증한 탓). 지연 호출로 바꿔 이름 있는 실패 7건으로 보이게 했다 —
원인 한 줄이 28개의 침묵을 만들면 안 된다.

### 검증

- `npm run verify` — **테스트 1163개 통과**, typecheck · ESLint 0
- `npm run gate` — **13종 ALL PASS**
- conveyor 게이트 실측(재현 확인: 2회 연속 동일): 3사이클 완주 24.86s ·
  놓기 거리 2.9 / 9.1 / 8.3 cm · 세 상자 모두 존 진입 + 존 안 정지 · `arm×belt` 0건 · 충돌 0건

---

## 3D 임포트 검증 · 라이브러리 로봇 3종 (2026-08-04)

**사용자 요청**: "3d 파일 임포트가 잘 되는지 확인하자. 다운받을 수 있는 물체들 넣어보고,
로봇 팔도 다양한 종류 로봇 팔(여러 가지 로봇 손 파트)도 넣어볼 수 있게 종류를 2개 더 확장해줘."

### 임포트 경로는 브라우저에서 한 번도 검증된 적이 없었다

`scripts/gate-browser.mjs`에 임포트 관련 어서션이 **0건**이었다. 파싱만 되면 아무도 몰랐다.
`--expect=mesh-import` 게이트(12 어서션)를 신설했다.

**계측 픽스처를 따로 만든 것이 이 작업의 핵심 결정이다** (`scripts/make-import-fixtures.mjs`).
`gate-box.{glb,stl,obj}` — 변 0.30×0.20×0.10, 원점에서 어긋난 직육면체:

| 형상 제약 | 없으면 |
|---|---|
| 세 변이 모두 다르다 | Z-up 회전(y↔z 교환)이 관측되지 않아 upAxis를 통째로 무시해도 초록 |
| bbox가 원점 중심이 아니다 | 피벗 재정렬의 x/z 성분이 항등이라 검증되지 않는다 |
| 3종이 같은 솔리드 | 포맷 간 교차 검증이 성립하지 않는다 |

다운로드 모델(CC0 5종)은 **카탈로그 스모크**로 분리했다 — 치수를 우리가 소유하지 않으므로
수치 어서션을 걸면 에셋 갱신 때 게이트가 통째로 무너진다.

뮤테이션 3종 전부 확인: Up-axis 무시 / 스케일 무시 / 피벗 재정렬 생략 → 각각 해당 어서션이
실패. 특히 피벗 생략 시 시각 중심이 정확히 `[0.25, ·, −0.25]`(픽스처 bbox 중심)만큼 어긋나고
상자가 trimesh를 통과해 바닥에 앉았다 — 원점 밖 픽스처의 설계 의도가 값으로 확인됐다.

### 손 모양이 서로 다른 로봇 3종

| 로봇 | 구조 | 손 | 게이트가 보는 신호 |
|---|---|---|---|
| Arm-6 | 6축 관절팔 | 평행 2지 | 위치 0.030m (prismatic) |
| SCARA-4 | 수평 2축 + 수직 직동 + 손목 | 흡착 패드 | 위치 0.012m (prismatic) |
| Cobot-7 | 7축 여유자유도 | 3지 클로 | **회전 0.75rad** (revolute) |

게이트를 만들며 측정을 두 번 틀렸고, 둘 다 교훈이 남았다:

1. 손 끝을 "월드 원점에서 먼 링크"로 골랐다 → 로봇이 원점 밖에 놓이자 엉뚱한 링크를 집었다.
2. 위치만 쟀다 → **revolute 손가락은 링크 원점이 회전축 위에 있어 위치가 변하지 않는다.**
   정상 동작하는 3지 클로를 "손이 안 움직인다"고 오판했다. 위치와 자세를 **모두** 봐야 한다.

### 게이트가 못 보는 것: 자기 간섭

`robot-library` 게이트는 "관절이 말단을 움직이는가"만 재고, **링크가 서로 파고드는가**는 못 본다.
selfCollision이 기본 off라(§5) 물리 이벤트도 나지 않는다 — 화면에는 팔이 자기 어깨를 관통해
지나가는데 게이트는 전부 초록이다. 적대적 검토가 이것을 지적했고,
`src/render/robot-self-clearance.test.ts`(OBB 분리축)로 영구 가드를 만들었다.

**AABB로 재면 안 된다**: 회전한 긴 링크의 축정렬 상자가 크게 부풀어 **정상인 arm6조차 50mm
관통으로 보고된다**. OBB-SAT로 바꾸자 arm6이 통과하며 기준선이 잡혔고, 그 위에서 신규 2종의
실제 결함이 드러났다:

| 결함 | 실측 | 조치 |
|---|---|---|
| Cobot-7 어깨 관통 | link1↔link3 20.6mm | 어깨 오프셋 0.048 → 0.085 (하한 = 실린더 반지름 0.052 + 박스 반폭 0.028) |
| Cobot-7 손목 | link5↔link7 6.8mm | joint6 ±2.05 → ±1.86 |
| Cobot-7 팔꿈치 | link2↔link5 7.5mm | joint4 상한 2.65 → 2.59 |
| Cobot-7 3지 상호 관통 | 3.3mm @ close | close 0.1 → 0.05, URDF 손가락 상한도 동일하게 |
| SCARA-4 받침 관통 | base↔link4 10.2mm | joint2 ±2.70 → ±2.64 |

3지 상호 관통은 **관절을 하나씩 훑는 것으로는 영원히 안 잡힌다** — 손가락 하나만 접으면 아무것도
만나지 않고, 셋이 함께 접혀야 서로 맞닿는다. 그리퍼는 한 덩어리로 훑어야 한다.

### 이름이 약속하는 것과 동작이 다르면 이름을 고친다

이 엔진에는 부착/용접 제약이 없다 — 파지는 순수 접촉 마찰뿐이다. 따라서 **흡착 패드는 누를 수만
있고 물체를 들어 올리지 못한다**. "흡착 그리퍼"라는 이름은 엔진이 못 하는 것을 약속한다.
라이브러리 카드 문구를 `흡착 패드는 누르기 전용`으로 고쳤다. 같은 이유로 Cobot-7 카드에는
개구(13cm)를 명시했다 — 기본 Box(한 변 10cm, 외접원 14.1cm)는 물 수 없다.

### 그 밖에 적대적 검토가 잡은 것

- `water-bottle.glb`의 `emissiveFactor`가 `[1,1,1]`로 남아 물병 전체가 백색 자체발광이었다 → `[0,0,0]`
- **OBJ만 재질이 달랐다** — STL은 `IMPORTED_MESH_COLOR` MeshStandardMaterial을 받는데 OBJ는
  OBJLoader 기본값(흰색 MeshPhongMaterial)이 남아 PBR 조명 아래 혼자 납작했다.
  `mesh-import.ts`의 obj 분기가 STL과 같은 재질을 씌우도록 고쳤다.
- SCARA `joint3`은 prismatic(미터)인데 플래너 컨텍스트는 revolute(라디안)로 가정한다 →
  `jointLimits: { joint3: [0, 0.18] }`를 RobotSpec에 명시. `home`에 `joint4`도 추가(없으면
  카드가 광고하는 4번째 축을 시퀀스가 주소지정할 수 없다).

### 검증

- `npm run verify` — **테스트 1170개 통과**, typecheck · ESLint 0
- `npm run gate` — **15종 ALL PASS** (mesh-import · robot-library 신규 2종 포함)

---

## ㄱ자 라인 셀 시퀀스 — 로봇 3대 순차 구동 (`l-line-cell.sequence.json`)

씬 `l-line-cell`(직각으로 이어진 벨트 2개 + SCARA-4/Arm-6/Cobot-7)에 시퀀스를 얹으면서
**컨베이어 위 파지의 물리적 한계**가 드러났다. 결정과 근거를 남긴다.

### 벨트는 진행축 속도를 매 스텝 주입한다 — 수직 벽으로는 상자를 못 세운다

`conveyor.drivenVelocity`는 진행축 성분을 벨트 속도로 **덮어쓰고** 측면·수직은 보존한다.
그래서 그리퍼 손가락을 상자 진행 방향 앞에 세워 스토퍼로 쓰려는 시도는 실패한다:
손끝(TCP y=0.083 → 손가락 하단 0.053)이 상자 윗면(0.080)만 눌러 **상자가 손가락을 타고
넘어간다**. TCP를 0.065까지 낮춰도 같았다(3/3 실패, 상자는 벨트 끝까지 가서 떨어졌다).

반면 **press(SCARA-4)의 패드는 상자를 세운다.** 차이는 힘의 방향이다 — 수직 하중은
접촉 법선력을 키워 마찰이 진행축 속도 주입을 이긴다(실측: 0.5초 유지 중 이동 0.0mm).
같은 이유로 **측면(lateral)으로 무는 파지는 성립한다** — 측면축은 벨트가 보존하기 때문이다.

결론: 벨트 위 파지는 `conveyor-pick-place`와 같은 패턴만 쓴다 —
**조(jaw)를 벨트 진행축과 직교로 벌리고 상자가 조 사이로 들어오게 한다** (arm6 `joint6 = 0`).

### 접촉 즉시 닫으면 안 된다 — 상자가 조 안에 다 들어온 뒤에 닫는다

`waitForCollision [picker, item_a]`는 상자 **앞모서리가 조 앞날에 닿는 순간** 발화한다.
그때 바로 닫으면 상자의 앞쪽 8mm만 물어 파지가 풀린다(실측 3/3 실패). 조의 진행축 길이는
30mm, 상자는 50mm이므로 겹침 창은 |Δz| < 40mm — 벨트 실효속도 0.107 m/s에서 0.75초다.
접촉 후 **`wait 0.35`(= 0.040 m / 0.107 m·s⁻¹)** 를 넣어 상자를 조 한가운데로 보낸 뒤
닫자 파지가 유지됐다. 파지 세기는 `state 0.35`(조 침투 12.5mm/측).

### 관절 보간은 카테시안 반경을 부풀린다 — 레일 옆 하강은 경유점을 둔다

`moveJoints`는 **관절 공간** 선형 보간이다. 툴다운 제약(joint2+joint3+joint5=π)은 보간 중에도
유지되지만 반경은 유지되지 않는다: TCP y 0.24 → 0.068 한 번에 내리면 중간에 반경이 **10.6mm
부풀어** 손가락이 `rail_outer`(x=0.72)를 3mm 긁는다(브라우저 실측으로 확인). 중간 경유점
(y=0.15)을 하나 넣어 부풀림을 3.2mm/2.5mm로 나누자 0건이 됐다.

### Cobot-7의 3지 클로는 `joint7`(공구 롤) 위상이 안전 파라미터다

완전 개방(g=-0.65)한 3지 클로의 반경은 83mm다. 인계 패드(`zone_handoff`)는 `belt_out`
안쪽면에서 90mm, 팔레트(`zone_pallet`)는 picker 베이스에서 85mm — 손가락 하나가 그 방향을
향하면 반드시 닿는다. 손가락이 120° 간격이므로 **한 방향을 비우면 반대 방향에 손가락이 온다**:
`joint7 = 0.8568`이 두 제약을 동시에 만족하는 유일한 위상대였다(핸드오프 maxX 0.5325 < 0.54,
팔레트 minX 0.3623 > 0.35). 파지 중에는 `joint7`을 돌릴 수 없으므로(손가락이 상자를 훑는다)
**집기 전에 한 번 맞춰 놓고 끝까지 유지한다.**

### 배리어는 "다음 step이 만들 접촉"을 기다릴 수 없다

`waitForCollision`은 init 시점에 마커를 발급한다. palletizer의 파지 접촉을 만드는 것은
**바로 다음 `gripper` step**이라, 그 사이에 배리어를 두면 영원히 timeout한다(실측 1.58s).
그렇다고 선언을 빼면 `collision-classify`가 정상 파지를 `unexpected`(충돌)로 센다.
`main.ts`의 `awaitedCollisionPairs()`가 **비활성 배리어도 조작 대상으로 포함**하도록
설계돼 있으므로, `enabled: false` 배리어로 쌍만 선언했다.

### 검증 (브라우저 실측, 새 페이지 2회 연속)

| 항목 | 결과 |
|---|---|
| 시퀀스 완주 | `status === 'done'`, 38/38 step, 33.7 s |
| press | `press × item_b` 3건 @5.35 s — 차단 0.5초 이동 0.0mm, 윗면 압박 0.5초 이동 0.0mm |
| picker | `picker × item_a` 5건 @11.7 s → `zone_handoff` 감지 @14.97 s, 정지 좌표 (0.450, 0.024, 0.482) — 패드 내부 |
| palletizer | `palletizer × item_a` 6건 @24.51 s → `zone_pallet` 감지 @28.74 s, 정지 좌표 (0.416, 0.024, 0.509) — 패드 내부 |
| 예기치 않은 충돌 | robot×belt / robot×rail / robot×robot **0건** |
| 결정론 | 새 페이지 2회 연속 최종 좌표·이벤트 시각 완전 일치 |
| page error / 경고 | 0 / 0 |

한계: item_b·item_c는 벨트 B 끝에서 떨어져 배출 구역에 남는다(이 시퀀스는 상자 1개 완주가
목표다). 3개 반복은 픽 스테이션 점유 시간과 상자 간격(2.8초)이 겹쳐 미검증이다.

---

## ㄱ자 라인 셀 — 직각 컨베이어 + 로봇 3종 협업 (2026-08-04)

**사용자 요청**: "새로 넣은 로봇들도 포함된 케이스도 추가해줘. 컨베이어 케이스에서 컨베이어를
더 길게 직각으로 더 이어지게 만들어서 새로 들어간 로봇들도 그 라인에서 일하는 예시로 만들어보자."

### 직각 코너는 된다 — 조건 다섯 개를 전부 지켰을 때만

두 벨트를 겹쳐 놓으면 `ConveyorRegistry.tickAll()`의 `claimed` 중재가 한 tick에 한 벨트만
각 바디를 구동한다. 그 위에서 코너가 성립하는 조건을 실측으로 하나씩 확정했다:

| 조건 | 어기면 (실측) |
|---|---|
| `belt_in`을 `belt_out`보다 **먼저 선언** | 중재 승자가 뒤바뀌어, 앞모서리가 하류에 닿는 순간 꺾여 몸통 대부분이 아직 상류인 채 낙하 |
| `belt_out`이 `belt_in` 끝을 **0.12m 오버런** | 0.06m면 정지하거나 x=0.715까지 오버런해 낙하 |
| `belt_out`의 레인이 `belt_in` 레인을 **완전히** 덮음 | 절반만 덮으면 코너에서 옆으로 빠짐 |
| 가이드 레일 2개(바깥+안쪽) | 레일 없이 4개 큐는 **1/12**만 생존, 바깥만 8/12, 둘 다 12/12 |
| 이어진 벨트는 `recycle: false` | 코너 지점이 이미 상류의 재순환 조건을 만족해 **시작점으로 순간이동** |

왜 오버런이 필요한가: 상류의 접촉이 깔끔하게 끊기지 않는다. 상자가 흔들리며 접촉이
17회 깜빡이고, 그동안 두 벨트가 번갈아 속도를 주입해 대각선으로 0.065m 표류한다.
하류가 그 표류를 받아낼 만큼 넓어야 한다.

### 내가 만든 결함 둘 — 둘 다 "조사 값에서 임의로 벗어난" 것이다

1. **가이드 레일이 라인을 막았다.** `rail_inner`의 z 시작을 벨트 A의 +z 모서리보다 앞에 두었더니
   레일이 레인 한가운데에 서서 지나가는 상자를 z=-0.085로 쳐냈다. 레일은 코너 **바깥**에서
   안내해야지 레인을 침범하면 안 된다.
2. **인계 패드와 팔레트가 간극 0으로 맞닿았다.** x=0.435에서 정확히 접해 5cm 상자가 두 존을
   동시에 발화시켰다 — picker가 인계하는 **그 순간** zone_pallet도 start했고, 실행 종료 시점에
   두 센서 모두 닫히지 않은 start를 보유했다. 즉 "팔레트 감지"가 적재의 증거가 되지 못했다.
   적대적 검토가 잡았다.

두 번째의 해법이 이 작업에서 가장 배운 것이다. 팔레트를 **호로 더 멀리** 보내려 했더니 picker
베이스에 닿았고(마찰 파지의 호 상한 0.30rad과 셀 배치가 정면으로 충돌한다), **반경을 당기니**
10cm가 벌어졌다 — 반경 이동에는 원심 성분이 없어 파지가 풀리지 않는다. 다만 반경을 당기면
x가 벨트 쪽으로 붙어 **완전히 벌린 3지 클로(반경 83mm)** 가 벨트 안쪽면을 스쳤고(실측 1건),
방위를 0.054rad만 틀어 해결했다.

### 각 로봇에게 증명된 일만 시킨다

| 스테이션 | 로봇 | 일 | 왜 이 일인가 |
|---|---|---|---|
| 벨트 A 중간 | SCARA-4 | **검사 프레스** — 눌러 세웠다 놓아준다 | 흡착 패드는 이 엔진에서 물체를 **들지 못한다**(부착 제약 없음). 누르면 벨트가 못 이긴다(유지 중 이동 0.3mm) |
| 벨트 B | Arm-6 | **라인 피킹** — 열고 기다렸다 벨트가 손에 밀어 넣으면 닫는다 | 기존 conveyor-pick-place가 3사이클 증명한 유일한 파지 |
| 인계 패드 → 팔레트 | Cobot-7 | **정지한** 상자를 3지로 집어 적재 | 정지 물체 파지만 실증됐다. 움직이는 상자는 손가락 하나가 진행 방향을 막아 튕긴다 |

프레스는 타이밍이 아니라 **배리어**로 만든다: 패드를 라인 높이로 내려 상자가 **부딪혀 서게 하고**
(`waitForCollision [press, item_b]` — 벨트 실효속도 오차와 무관), 유지 → 살짝 들어 통과 →
윗면 압박. 대상이 item_a가 아니라 item_b인 이유도 실측이다 — item_a를 세우면 뒤 상자와 간격이
0.09m로 줄어 픽 스테이션에서 끼어들고, item_b를 세우면 반대로 **벌어져** picker에게 유리하다.

### 이 씬의 최대 자기기만 벡터: 관절 이름 겹침

세 로봇의 관절 이름이 `joint1`~`joint4`까지 **완전히 겹친다**. step에 `robot`을 빠뜨리면
`validateSequence`를 통과한 뒤 **조용히 엉뚱한 로봇이 움직인다**. 38 step 전부에 `robot`을
명시했고, 독립 검증이 누락 0건을 확인했다.

### 게이트가 물어야 하는 것

"시퀀스가 done으로 끝났다"만으로는 로봇이 허공에서 춤춰도 통과한다. `--expect=l-line-cell`은
스테이션마다 **그 로봇이 그 상자를 만졌다는 접촉**과 **상자가 실제로 옮겨진 좌표**를 함께 본다.

두 가지를 의도적으로 피했다:
- **센서 이벤트로 적재를 판정하지 않는다.** 위 결함 2 때문에 존 이벤트는 "인계 상태"와
  "적재 상태"를 구분하지 못한다. **중심 좌표 포함**만이 판별력을 갖는다.
- **분류(target/unexpected)를 거치지 않고 원시 쌍으로 센다.** `awaitedCollisionPairs()`가
  배리어 선언으로 시각 무관 화이트리스트를 만들기 때문에, 분류를 믿으면 "충돌 0건"이
  물리가 아니라 선언을 재는 값이 된다(검토 지적).
- 재생 전에 **페이지를 새로 연다.** 공통 검사가 흘려보낸 2.3초 동안 벨트가 상자를 실어 날라
  라인 위상이 바뀌고, 그 상태로 재생하면 press 배리어가 이미 지나간 상자를 기다리다 죽는다
  (실측: step 14에서 45초 정지). `stop()`은 좌표만 되돌리고 결정론을 복원하지 못한다.

### 알려진 한계

- **상자 1개만 3스테이션을 통과한다.** item_b·item_c는 코너를 돌아 벨트 B 끝에 남는다.
  픽 스테이션 점유 시간(약 4.5초)이 상자 도착 간격(2.8초)보다 길다 — 3개 반복은 미검증이다.
- **동시 동작은 불가능하다.** 선형 1-step 모델이라 로봇은 차례로만 일한다. 동시성의 착시는
  컨베이어가 만든다(매 물리 tick 계속 돈다).
- palletizer의 파지 접촉은 `enabled: false` 배리어로 쌍만 선언한다 —
  `waitForCollision`이 "바로 다음 step이 만들 접촉"을 기다릴 수 없는 구조적 한계 때문이다.

### 검증

- `npm run verify` — typecheck · ESLint 0
- `npm run gate:l-line-cell` — **ALL PASS** (8 어서션)
- 실측(새 페이지 2회 연속 동일): 완주 32.1s · 38/38 · 놓기 거리 3.8cm / 7.7cm ·
  item_a 최종 [0.451, 0.024, 0.579](팔레트 안, 인계 패드 밖) · robot×정적 0건 · robot×robot 0건
- 씬 계약 8건 + 샘플 공통 그물, 뮤테이션 5종(선언 순서·오버런·재순환·레일 침범·동일 URDF) 전부 확인

---

## ⏹ 되감기의 결정론 — 접촉 매니폴드를 버린다 (2026-08-04)

되감기(⏹) 후 재생이 **같은 설정에서 다른 결과**를 냈다. A/B 실측(각 3회, 컨베이어 씬):

| | 결과 |
|---|---|
| 로드 직후 바로 재생 | 최종 좌표가 소수점 5자리까지 **3회 동일** |
| 15초 돌린 뒤 ⏹ → 재생 | **3회 중 2회가 다름** (한 번은 완주 54.4s, 정상 24.9s) |

되감기 직후·재생 직전의 아이템 좌표는 3회 모두 동일했고 그 사이 물리 진행도 0이었다.
**같은 좌표에서 다른 결과** → 원인은 좌표가 아니라 솔버 내부 상태다. `teleport`는 pose만
되돌리고 Rapier의 **접촉 매니폴드에 누적된 워밍스타트 임펄스**는 남긴다. 그 값이 다음 스텝의
초기값이 되므로 "직전에 무슨 일이 있었는지"가 결과에 남는다. 헤드리스에서 fresh 빌드끼리는
바이트 동일이었다는 사실이 이를 뒷받침한다 — 차이는 되감기냐 새로 짓기냐 하나뿐이었다.

### 고친 방법

`PhysicsWorld.clearContactState()` — collider를 제거하고 같은 바디에 같은 스펙으로 다시 만든다.
Rapier가 collider 제거 시 그 좁은 단계 상태를 함께 버리므로 접촉 이력만 사라지고 기하는 그대로다.
재생성에 필요한 `ColliderSpec`을 `RapierWorld`가 보관하도록 했고, `SceneHandle.reset()`이 호출한다.
제거가 유발하는 'stop' 이벤트는 삼킨다 — 되감기는 충돌 이력도 비우므로 그 이벤트는 이미 지워진
과거에 속하고, 남기면 되감기 직후 로그에 유령 이벤트가 찍힌다.

### 고쳐진 것과 **고쳐지지 않은 것**

- **고쳐졌다**: 되감기 후 재생이 **매번 같은 결과**를 낸다 (수정 후 3/3 동일).
- **고쳐지지 않았다**: 그 결과가 **새로 연 페이지의 재생과 같지는 않다**
  (되감기 54.4s vs 새 월드 24.8s — 컨베이어 씬 세 번째 사이클의 배리어 두 개(step 22·24)에서
  20.1s·12.0s를 소모한다). collider 재생성이 Rapier 내부 핸들·순회 순서를 바꾸므로 되감기
  재생은 자기들끼리만 재현된다. 완전 동등성은 월드 재빌드가 필요하고 그건 별도 작업이다.

계약 문구를 실제 달성 범위에 맞춰 고쳤다 — `ui/orchestrator.ts`·`main.ts`의 "결정론적 재생 준비"를
"재생 준비"로 낮추고, 무엇이 보장되고 무엇이 아닌지를 `core/types.ts`에 적었다.
**자동화·게이트가 새 월드와 같은 결과를 원하면 페이지를 새로 열어야 한다**(게이트들이 그렇게 한다).
