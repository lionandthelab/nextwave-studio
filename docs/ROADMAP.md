# ROADMAP — robot-sim-web

각 Phase는 **검증 게이트(Definition of Done)**를 통과해야 다음으로 넘어간다.
게이트를 통과하면 `EXPERIMENTS.md`에 결과와 결정을 기록한다.

---

## Phase 0 — 스캐폴딩 & 하네스 (기반)

**목표**: 프로젝트가 돌아가는 최소 뼈대와 개발 환경.

- [ ] Vite + TypeScript(strict) 초기화, `vite.config.ts` WASM 대응 반영
- [ ] `package.json` 의존성 설치: `three`, `@dimforge/rapier3d-compat`, `urdf-loader`
- [ ] ESLint/tsc 통과 파이프라인, 기본 테스트 러너(vitest 등)
- [ ] `src/` 디렉터리 골격 생성(CLAUDE.md §3 지도대로 빈 모듈 + 배럴)
- [ ] `RAPIER.init()` 후 빈 world 생성 + three.js 빈 씬 렌더 확인(회색 화면 + 그리드)

**게이트**: 브라우저에서 빈 3D 씬이 뜨고 콘솔에 `Rapier ready`가 찍힌다.
`tsc --noEmit` 통과.

---

## Phase 1 — 렌더 + 물리 최소 결합

**목표**: 물리가 진실, 시각은 거울인 파이프라인 검증.

- [ ] `RapierWorld`(PhysicsWorld 구현): 강체/collider 생성·스텝, 핸들 매핑
- [ ] 고정 timestep 루프(`Engine`) + accumulator + dt clamp
- [ ] `RenderSync`: RigidBody pose → Object3D 단방향, alpha 보간
- [ ] 하드코딩 데모: 바닥(fixed) + 떨어지는 박스(dynamic)가 물리적으로 낙하·정지

**게이트**: 박스가 중력으로 떨어져 바닥에 안정적으로 놓인다. 프레임레이트를
바꿔도(30/60/144) 낙하 궤적이 동일(결정론). 물리-시각 어긋남 없음.

---

## Phase 2 — 스키마 & 씬 로더

**목표**: 씬을 데이터로 선언 → 코드 변경 없이 로드.

- [ ] `src/schema`: `SceneSpec`/`EntitySpec`/`PhysicsSpec` 타입 + zod 런타임 검증
- [ ] `scene-loader`: SceneSpec → world 바디 + render 노드 + 핸들 매핑
- [ ] 프리미티브 사물(box/sphere/capsule) 로드, 그룹/필터 비트마스크 적용
- [ ] 검증 오류를 사람이 읽을 수 있게 UI/콘솔에 표시
- [ ] 샘플 SceneSpec 1종을 파일에서 로드해 렌더

**게이트**: JSON만 바꿔 사물 개수/배치를 바꿔도 코드 수정 없이 반영된다.
잘못된 SceneSpec은 명확한 오류로 로드가 중단된다.

---

## Phase 3 — URDF 로봇 로딩

**목표**: 로봇을 로드하고 관절을 개별 제어.

- [ ] `urdf-loader`로 URDF → three.js, `loader.packages`로 메시 경로 매핑
- [ ] URDF(Z-up) → 내부 Y-up 축 변환을 `scene-loader` 한 곳에서 처리
- [ ] 링크 collider 생성(`fromVisual`/primitive), self-collision 기본 off
- [ ] 관절 setpoint API(`robots.setJoint`)로 슬라이더 수동 제어(임시 UI)
- [ ] `home` 관절값으로 초기 포즈 설정

**게이트**: 로봇이 바르게 서고(축 정상), 슬라이더로 각 관절이 움직인다.
로봇 링크와 사물이 겹칠 때 물리적으로 인지 가능한 상태(다음 Phase에서 이벤트화).

---

## Phase 4 — 충돌 감지 ★ (프로젝트 핵심)

**목표**: 로봇–사물/환경 충돌을 이벤트로 감지·발행.

- [ ] `emitEvents` collider에 `ActiveEvents.COLLISION_EVENTS` 설정
- [ ] `world.step(eventQueue)` + `drainCollisionEvents` → `ContactEvent`
- [ ] `CollisionMonitor`: 핸들→엔티티 변환, `start/stop`·`contact/sensor` 분류, 발행
- [ ] sensor 영역(감지만, 반응 없음) 동작 확인
- [ ] CCD 필요 케이스(빠른 링크 vs 얇은 사물) 검증
- [ ] UI 충돌 로그 패널(시간·엔티티 쌍·phase) + 충돌 오브젝트 하이라이트

**게이트**: 로봇이 박스에 닿는 정확한 시점에 `start`, 떨어질 때 `stop`이 로그에 남는다.
sensor 영역 진입/이탈이 물리 반응 없이 감지된다. 고속 이동에서도 관통 없이 감지.

---

## Phase 5 — 제어 시퀀스 재생

**목표**: 선언적 시퀀스로 로봇을 구동, 재생 제어.

- [ ] `ControlSequence` 스키마 + 검증(robot/joint 참조 무결성)
- [ ] `ControlPlayer`: `moveJoints`(보간/easing), `setJoints`, `wait`, `gripper`
- [ ] 흐름 제어(`label`/`goto`) + 전체 `loop`
- [ ] `waitForCollision` 배리어(충돌 이력 조회로 해제, timeout 처리)
- [ ] 재생 컨트롤 UI: play/pause/reset/step-once/속도
- [ ] 샘플 시퀀스로 픽앤플레이스 흐름 재생

**게이트**: 샘플 시퀀스가 재생/일시정지/리셋된다. `waitForCollision`이 실제 충돌에
동기화되어 다음 step으로 진행. 단일 스텝 디버깅 동작.

---

## Phase 6 — 통합 데모 & 다듬기

**목표**: 배포 가능한 완성형 MVP.

- [ ] 샘플 씬 3종: 픽앤플레이스 / 장애물 회피 / 충돌 테스트베드
- [ ] 씬·시퀀스 파일 업로드 로더, 프리셋 선택 UI
- [ ] frontend-design 스킬 원칙으로 UI 정리(레이아웃·타이포·인터랙션)
- [ ] 인스펙터(엔티티 목록·선택·트랜스폼/관절 상태 표시)
- [ ] 정적 빌드(`vite build`) → 정적 호스팅 배포 확인(GPU 없는 노트북 브라우저)
- [ ] README 실행법·샘플 스크린샷/영상

**게이트**: PRD §7 성공 기준 충족(코어 시뮬 데모). 링크만 열면 3종 데모가 재생·충돌
감지된다. 새 씬/시퀀스를 데이터만으로 추가 가능.

> Phase 0–6은 **코어 시뮬레이터**(물리·충돌·시퀀스 재생) 트랙이다. 아래 **UI/Planner
> 트랙**(Phase 7–10)은 `docs/UX_DESIGN.md`·`docs/PLANNER.md`의 저작 경험을 그 위에 올린다.
> 일부는 코어 트랙과 **인터리브** 가능하다(의존성은 각 Phase에 표기).

---

# UI/Planner 트랙 — 저작 경험 (docs/UX_DESIGN.md · docs/PLANNER.md)

## Phase 7 — Scene Builder UI (드래그앤드롭 씬 구성)

**목표**: 라이브러리에서 물체·로봇을 드래그로 추가하고 치수를 쉽게 조정. (선행: Phase 3)

- [ ] 워크스페이스 셸(커맨드바 + 3존 + 독) 레이아웃, 리사이즈/접기 (UX_DESIGN §2)
- [ ] Library 패널: 프리미티브/로봇 템플릿 카드, 검색, 드래그 프리뷰→바닥 레이캐스트 배치
- [ ] Viewport 상호작용: 선택·아웃라인·Transform 기즈모(이동/회전/스케일)·스냅
- [ ] Inspector(오브젝트): Transform + **치수(W/H/L·반지름·높이)** 숫자입력/스크럽 →
      메시+collider 동시 갱신, Physics(그룹·마찰·bodyType)
- [ ] 3D 파일 임포트: 드래그/파일선택 → 다이얼로그(형식·스케일·up-axis·collider 전략) →
      씬 편입 (glTF/glb·STL·OBJ; 동적 trimesh 금지→convexHull) (UX_DESIGN §4.4)
- [ ] 씬 ⇄ SceneSpec 왕복(저장/불러오기), Undo/Redo

**게이트**: 코드 수정 없이 드래그앤드롭 + 인스펙터만으로 임의 씬을 구성·저장하고,
외부 3D 파일을 임포트해 물리 바디로 쓸 수 있다.

## Phase 8 — Flow Graph 에디터 (n8n형)

**목표**: ControlSequence를 노드 그래프로 시각화·편집. (선행: Phase 5, 스키마의 FlowGraph
변환)

- [ ] `schema`에 `FlowGraph` 뷰모델 + `toSequence`/`fromSequence` 무손실 변환 (UX_DESIGN §6)
- [ ] 노드 캔버스: 팬/줌/미니맵/fit, step 종류별 노드 렌더(아이콘·요약·상태 점)
- [ ] **드래그 재정렬**(체인 내 삽입선 프리뷰 → steps 순서 갱신)
- [ ] 엣지 ＋로 중간 삽입(노드 팔레트 팝오버), 선택 삭제 시 앞뒤 자동 재연결, 복제
- [ ] Inspector(노드): step별 파라미터 폼(관절 슬라이더·duration·easing·엔티티 픽커 등)
- [ ] 활성/비활성 토글, `{} JSON` 원본 뷰어와 실시간 동기(라운드트립 검증)

**게이트**: 노드를 드래그로 재정렬·삽입·삭제·편집하면 즉시 유효한 ControlSequence로
직렬화된다. 직렬화 불가능한 상태를 만들 수 없다(불변식 §2.8).

## Phase 9 — 자연어 Planner (NL → ControlSequence)

**목표**: 문장을 씬에 그라운딩해 액션 플로우 초안 생성. (선행: Phase 8)

- [ ] `scene-context.ts`: SceneSpec+상태 → `SceneContext`(로봇·관절·limits·물체·방향규약)
- [ ] `planner.ts`: 프롬프트(스키마+컨텍스트+규약) → JSON-only 출력, LLM 어댑터로 격리
- [ ] `validate-repair.ts`: zod+참조 무결성 검증, 오류 되먹임 복구 루프(상한 N회)
- [ ] 출력 계약 `PlannerResult`(sequence/clarify/error) 처리, `assumptions` 노출
- [ ] 커맨드바 자연어 입력·생성 상태, 명확화 카드(옵션 버튼), 교체/이어서 모드
- [ ] **미검증 출력은 실행 비노출, 자동 실행 금지**(불변식 §2.9)

**게이트**: 자연어 문장이 씬에 그라운딩된 유효 ControlSequence로 변환되어 그래프에
로드된다. 모호 입력은 clarify로, 실패는 사람이 읽을 수 있는 오류로 처리된다.

## Phase 10 — 실행 오케스트레이션 UI

**목표**: JSON대로 시뮬레이터에 동작을 하나씩 요청하고 상태를 동기 표시. (선행: Phase 8,
코어 Phase 5)

- [ ] `orchestrator.ts`: 그래프 순회 → 노드별 dispatch/완료대기/피드백 수집(UX_DESIGN §5)
- [ ] 재생 컨트롤(Play/Pause/Stop/Step/속도)과 노드 경계 정지
- [ ] **동기 강조**: 활성 노드(그래프) ↔ 로봇 동작(뷰포트) ↔ Timeline 커서 일치
- [ ] Collision Log 연동(행 클릭→오브젝트 포커스+당시 노드 강조), 충돌 인지 자동정지 옵션
- [ ] 노드/타임라인 마커에서 재실행(결정론적 재생)

**게이트**: 플로우가 노드 단위로 실행되며 활성 노드·로봇 동작·타임라인이 항상 동기화되고,
충돌이 로그·뷰포트에 반영된다. Flow 1(자연어→검토→실행) 전체가 매끄럽게 동작.

---

## 백로그 (Non-goal에서 승격 후보)

- **IK 솔버**: `moveToPose`(카테시안) step 구현. 관절 공간 → 작업 공간 제어.
- **MuJoCo 물리 계층**: `MujocoWorld` + SceneSpec→MJCF 변환기. 접촉 파지 사실성.
- **폐루프(agentic) 실행**: 노드 실행 피드백을 planner에 되먹여 다음 행동 적응 (PLANNER §5.1).
- **실기 연동**: websocket/serial 브리지로 digital-twin(양방향 상태 동기화).
- **센서 시뮬**: 오프스크린 카메라 depth/RGB, (선택) LiDAR 근사.
- **그래프 분기/조건**: 단순 goto/label을 넘어선 조건 분기 노드.
- **성능**: collider LOD, instancing, 다수 사물 최적화.
- **xacro 지원**: 로드 시 파싱 또는 사전 변환 파이프라인.

---

## 진행 규칙

1. Phase는 순서대로. 단, 스키마(Phase 2)는 이후 모든 Phase의 선행 조건.
2. 각 Phase 종료 시 `EXPERIMENTS.md`에 결정·트레이드오프·측정치 기록.
3. 게이트 미통과 항목은 다음 Phase로 미루지 말고 명시적으로 백로그에 등록.
4. 불변식(CLAUDE.md §2)과 충돌하는 지름길은 채택하지 않는다.
