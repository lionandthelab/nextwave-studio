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
