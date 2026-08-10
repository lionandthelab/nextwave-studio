# Workcell

> 저장소 이름은 `robot-sim-web`, **제품명은 `Workcell`** 이다 (상위 브랜드 NextWave Studio).
> 제품 정체성 상수의 단일 진실은 `src/ui/brand.ts`.

브라우저에서 완결되는 **로봇 시뮬레이션 스튜디오**. 씬(로봇·사물·환경)을 JSON으로 선언해
로드하고, URDF 로봇 팔을 제어 시퀀스로 재생하며, 로봇–사물/환경 충돌을 물리 엔진 이벤트로
실시간 감지한다. 설치도 계정도 없이 브라우저에서 열리고, 백엔드 없이 정적 호스팅으로 배포된다.

**구현 범위 — ROADMAP Phase 0–10 + Phase 11 완료(모든 게이트 통과)**: 물리·URDF 로봇·충돌
감지·시퀀스 재생·런타임 씬 전환·인스펙터 UI에 더해, **드래그앤드롭 씬 빌더(Phase 7)**·
**n8n형 노드 그래프 에디터(Phase 8)**·**자연어 플래너(Phase 9)**·**노드 단위 실행
오케스트레이션(Phase 10)**·**제품화(Phase 11)** 까지 브라우저에서 완결된다.

**Phase 12 — 협업 콘솔(백엔드 · 다중 사용자)**: 선택적 서버(`server/` — Fastify + SQLite
단일 파일, 정적 번들 동시 서빙)와 **콘솔 평면**(공정 · 작업 · 재사용 블록 · 장비 · 실행
기록 · 설정, `#/tasks` 해시 라우팅)이 추가됐다. 사용자 타일 + PIN 로그인(현장 공유 단말
전제), 낙관적 버전 + 편집 잠금, soft-delete 휴지통(30일), append-only 실행 기록과
"이 노드부터 재현", 오프라인 outbox 동기화까지 — **서버가 없으면 기존처럼 로컬 모드로
동작한다**(정적 호스팅 불변식 유지). 규범 문서: [`docs/BACKEND.md`](docs/BACKEND.md).

**Phase 11 — 제품화(Studio Hardening)**: 5인 디자인 팀 진단(`docs/UX_AUDIT.md`)을 실행해
디자인 시스템(토큰 7축 · SVG 아이콘 · Pretendard 자체 호스팅)을 세우고, 뷰포트를 압살하던
고정 레이아웃을 `clamp()` 정책으로, 흩어진 전역 키 리스너 5개를 단일 라우터로, 씬만 저장하던
경로를 `{scene, sequence, assets}` 문서 모델(+IndexedDB 자동저장)로 바꿨다.
근거와 트레이드오프는 `EXPERIMENTS.md` "Phase 11".

| 지표 | 진단 시점 → 현재 |
|---|---|
| 3D 뷰포트 점유율 (1366×768) | 20.9% → **33.2%** |
| 커맨드바 요소 겹침 (1280×720) | 20건 → **0건** |
| 그래프 fit 줌 (7노드, 836×240) | 47% → **55%** |
| 단위 테스트 | 759 → **978** |

| 역할 | 기술 |
|-----|------|
| 물리 엔진 | Rapier (`@dimforge/rapier3d-compat`, WASM) |
| 렌더링 | three.js |
| 로봇 모델 | `urdf-loader` (URDF → three.js, Z-up→Y-up 변환) |
| 스키마 검증 | zod (`SceneSpec` / `ControlSequence` 런타임 검증) |
| 빌드/테스트 | Vite + TypeScript strict + vitest + Playwright 게이트 |

## 데모 씬 7종

`?scene=<이름>` 쿼리 파라미터 또는 커맨드바의 씬 select로 전환한다.
각 씬은 `scripts/gate-browser.mjs`의 자동 게이트로 검증된다.

| 씬 | 내용 | 게이트 (`--expect=`) |
|----|------|---------------------|
| `falling-boxes` | 박스들이 낙하해 바닥에 정착 — 물리/결정론 스모크 | `falling-boxes`: 전 동적 바디가 바닥 위 정착 |
| `arm-and-boxes` | 6-DOF+그리퍼 URDF 팔이 박스에 접촉 — `waitForCollision` 배리어 + 그리퍼 개폐 시퀀스 | `arm`: 축 변환·home 포즈·관절 구동 / `arm-sequence`: 무자동재생·arm×box_a 충돌 이력·배리어 해제·done |
| `pick-and-place` | 팔이 cargo를 밀어 drop_zone(센서 영역)으로 옮기는 9-step 시퀀스 | `pick-and-place`: arm×cargo 접촉 start + cargo×drop_zone 센서 start + 시간 예산 내 done |
| `obstacle-avoidance` | 기둥을 위로 넘어가는 회피 경로 후 목표 박스 접촉 | `obstacle-avoidance`: arm×pillar 이벤트 **0건** + arm×target_box 접촉 start + done |
| `collision-testbed` | 로봇 없는 물리 쇼케이스 — 반발 공 바운스, 저마찰 미끄럼 + 센서 게이트 통과, 구름 공의 스택 전도 | `collision-testbed`: 접촉 쌍 ≥3종 + 센서 start ≥1 + 전 바디 정착 |
| `two-arms-collision` | 로봇팔 2대가 중앙에서 접촉 — 로봇↔로봇 충돌 감지 + 접촉점 마커 | `two-arms`: arm_left×arm_right 접촉 start + 접촉점 좌표 + 자기 링크 억제 + 방향키 이동 |
| `conveyor-pick-place` | **컨베이어 라인** — 벨트가 물건을 계속 실어 오고(끝에서 시작점으로 순환), 포토아이가 도착을 감지하면 로봇이 집어 적재 레인에 넣는다. **상자 3개를 연속으로** 처리한다 | `conveyor-pick-place`: 벨트 이송 + 재순환 + 포토아이 3회 감지 + 픽 접촉 3회 + **세 상자 모두 존 안착** + 놓는 순간 파지 유지 + **arm×belt 0건** + 편집이 거동을 바꿈 |
| `l-line-cell` | **ㄱ자 라인 셀** — 직각으로 이어진 벨트 2개(1.3m + 0.66m) 위에서 **손이 서로 다른 로봇 3대**가 차례로 일한다: SCARA-4 검사 프레스 → Arm-6 라인 피킹 → Cobot-7 팔레타이징 | `l-line-cell`: 코너 이송 3개 + 스테이션 3종 접촉 + 팔레트 안착(중심 좌표 판정) + 놓기 파지 유지 + **로봇×설비 0건** |

추가로 씬이 아닌 **기능 게이트**가 저작·실행 계층을 검증한다: `--expect=mesh-import`
(3D 파일 임포트 — glb/stl/obj 3종 파싱 · 스케일·Up-axis·피벗이 collider에 반영 · 임포트
사물이 실제로 낙하해 바닥과 충돌 · trimesh가 단단함 · 손상 파일 거부) ·
`--expect=robot-library`(로봇 3종이 서고 관절이 말단을 움직이고 손이 여닫힘) ·
`--expect=scene-switch`
(런타임 씬 전환, URDF 재로드 왕복) · `--expect=scene-builder`(드래그앤드롭 편집·undo) ·
`--expect=flow-graph`(노드 편집 ↔ 유효 시퀀스 무손실 변환) · `--expect=planner`(자연어 →
검증 → 그래프 로드, 무자동재생) · `--expect=orchestration`(노드 단위 실행·트라이페인 동기·
결정론적 재실행) · `--expect=viewport-edit`(바닥 하한·방향키 이동·선택 HUD).
**ROADMAP Phase 0–10의 게이트가 전부 통과한다.**

## 실행법

### Docker (OS 무관 — Windows / Linux / macOS 동일)

호스트에 Node를 설치할 필요가 없다. 이미지 안에서 빌드까지 끝나고 nginx로 정적 서빙한다.

```bash
docker compose up -d          # 빌드 + 실행 → http://localhost:8080
docker compose down           # 정지

# 개발(HMR, 소스 bind mount)
docker compose --profile dev up dev        # → http://localhost:5173
# 검증(컨테이너 안에서 typecheck + lint + 단위 테스트)
docker compose --profile ci run --rm verify
```

포트 변경은 `APP_PORT=9000 docker compose up -d`. 컨테이너 배포 형상을 실브라우저로
검증하려면 `node scripts/verify-container.mjs`(= `npm run docker:check`).
자세한 내용은 [`docs/USAGE.md` §10.2](docs/USAGE.md).

### Node 직접 실행

```bash
npm install
npm run dev        # Vite 개발 서버 (기본 http://localhost:5173)
npm run build      # tsc --noEmit + vite build → dist/
npm run verify     # typecheck(앱+서버) + lint + vitest (단위 테스트)

# 협업 서버 (Phase 12 — 선택. 없으면 앱은 로컬 모드로 동작)
npm run server:dev     # 개발: tsx watch (기본 http://localhost:8787, dev 서버가 /api 프록시)
npm run server:start   # 운영: dist/ 정적 서빙 + API + SQLite 단일 파일 (포트 1개, 프로세스 1개)

# 브라우저 게이트 (Playwright chromium 필요: npx playwright install chromium)
npm run gate               # 전체 13종 (build + 기능 6종 + 제공 예제 7종)
npm run gate:samples       # 제공 예제 7종만 — 각 씬 로드 + 시퀀스 완주 + 충돌 감지
npm run gate:viewport-edit # 뷰포트 편집 UX (바닥 하한 · 방향키 이동 · 선택 HUD)

# 개별 실행 (vite build 선행 필요)
node scripts/gate-browser.mjs --expect=arm-sequence
node scripts/gate-browser.mjs --expect=collision-testbed   # 등 표의 --expect 값
```

게이트는 `dist/`를 프리뷰 서버로 띄워 실제 브라우저에서 물리 어서션을 검증하고
`gate-screenshot.png`를 남긴다.

> **사용법 전체는 [`docs/USAGE.md`](docs/USAGE.md)** — 화면 구성, 단축키, 씬 구성,
> 플로우 그래프 편집, 자연어 플래너, 실행·재실행, 문제 해결까지 작업별 절차를 담았다.

## 조작법

- **카메라**: 마우스 왼쪽 드래그 = orbit, 오른쪽 드래그 = pan, 휠 = zoom.
- **Space**: 재생/일시정지 토글(입력 필드 포커스 중에는 무시).
- **커맨드바(상단)**: 자연어 입력(플래너) · 씬 프리셋 select · 📂 씬 JSON 업로드
  (SceneSpec 단독 또는 `{scene, sequence}` 봉투) · 💾 현재 SceneSpec 다운로드 ·
  ▶ Play / ⏸ / ⏹(리셋) / **⏭ Step(노드 1개)** · 재생 속도 · **충돌 시 자동 정지** 토글 ·
  **플로우**(노드 그래프 페인 열기) · `{} JSON` **시퀀스 JSON 편집기 + 버전 이력**
  (직접 편집 → 검증 통과 시에만 적용 · 임의 버전으로 되돌리기) · ⚙ 플래너 설정.
- **좌측 라이브러리**: 오브젝트/로봇 템플릿 카드 — 뷰포트로 드래그앤드롭 배치, 3D 파일
  임포트(⬆). **뷰포트**: 클릭 선택 · W/E/R 기즈모(이동/회전/스케일) · **방향키로 선택
  오브젝트 이동**(5cm, Shift=1cm, PageUp/Down=수직) · 좌하단 실행 오버레이
  (● 상태 · node k/n · 활성 노드 라벨 · simTime).
- **우측 패널**: 관절 슬라이더(로봇 씬) · 인스펙터(엔티티 목록·pose·관절) · 엔티티 편집 폼
  (이름·Transform·치수·Physics) · 노드 파라미터 폼(플로우 노드 선택 시).
- **하단 독**: Timeline(step 마커·simTime, **마커 클릭 → 그 노드부터 재실행**) ·
  Collision Log(시간·엔티티 쌍·phase·kind, **행 클릭 → 오브젝트 하이라이트 + 당시 활성 노드
  강조**) · Console.
- **플로우 그래프**(Phase 8): 노드 드래그 재정렬 · ＋로 노드 삽입 · Del 삭제 · Ctrl/Cmd+D
  복제 — 모든 편집이 항상 유효한 `ControlSequence`로 무손실 변환된다(불변식 §2.8).
- 시퀀스·플래너 생성물은 **자동 재생되지 않는다** — 검증 통과 후 ▶ Play를 눌러야 시작된다
  (human-in-the-loop, 불변식 §2.9).

## Flow 1: 자연어 → 검토 → 실행 (Phase 9–10)

핵심 사용자 흐름은 "**말로 지시 → 그래프로 검토 → 노드 단위로 실행**"이다
(UX_DESIGN §4.1/§5). 예: `arm-and-boxes` 씬에서

1. **생성** — 커맨드바 자연어 입력에 `box_a를 집어` 입력. 플래너(규칙 기반 기본, 오프라인;
   ⚙에서 Anthropic 선택 가능)가 현재 씬을 그라운딩해 `ControlSequence`를 만든다.
2. **검토** — 생성물은 **스키마 검증을 통과한 뒤에만** 플로우 그래프에 `AI` 배지 노드로
   로드된다(무효 출력은 실행에 노출되지 않는다 — §2.9). 노드를 클릭해 파라미터를 확인/수정
   하거나, 드래그로 순서를 바꾼다. 자동 재생하지 않는다.
3. **실행** — ▶ Play로 노드 단위 재생. 실행 오케스트레이터가 **활성 노드를 세 뷰에 동시에
   강조**한다(§5 동기 강조): 그래프 노드 아웃라인 ↔ 뷰포트 오버레이 `node k/n`+활성 라벨 ↔
   Timeline 커서. ⏭ Step으로 노드 하나씩, ⏸로 정지, 속도 조절.
4. **관찰·재실행** — 충돌은 Collision Log에 기록되고, **접촉점에 빨강 마커**가 뜨며 관련
   오브젝트가 빨강 펄스한다(로봇↔로봇 충돌도 감지된다 — 단 kinematic 로봇은 반력이 없어
   밀려나지는 않는다).
   로그 행을 클릭하면 그 충돌 시점에 활성이던 노드가 강조된다. Timeline 마커/‘충돌 시 자동
   정지’ 토글로 예기치 않은 충돌에서 멈추고, 마커를 눌러 **그 노드부터 결정론적으로 재실행**
   한다(처음부터 되감아 빨리감기 — per-node 스냅샷은 백로그).

`scripts/gate-browser.mjs --expect=orchestration`이 이 실행 계층(초기 pending → Play →
상태 진행 → arm×box_a 충돌 → 트라이페인 일관 → 완주 → stepNode 1노드 → 재실행)을 실브라우저
에서 검증한다.

## 아키텍처 한 장 요약

```
main.ts (조립 글루 — window.__sim 게이트 훅 노출)
  ├─ ui/       커맨드바(자연어/씬/재생/JSON) · 독(타임라인·충돌로그·콘솔) · 인스펙터 ·
  │            라이브러리 · 플로우 그래프(노드 에디터) · 실행 오케스트레이터 ·
  │            뷰포트(선택·기즈모·실행 오버레이)   (디자인 토큰: ui/theme.ts)
  ├─ planner/  자연어 → ControlSequence 생성·검증·복구 (LLM 어댑터로 격리)
  ├─ core/     engine(고정 timestep 루프) · world(Rapier 래퍼) · scene-loader ·
  │            collision(EventQueue 소비) · control/player(시퀀스 해석) · sync
  ├─ render/   three.js 렌더러 · 프리미티브 메시 · URDF 로더(Z-up→Y-up axisFix)
  └─ schema/   SceneSpec/ControlSequence 타입 + zod 런타임 검증

의존 방향: ui → core → {render, schema}. core는 three.js를 모르고(경계는 sync/render),
Rapier 심볼은 core/world·collision 밖으로 새지 않는다(PhysicsWorld 인터페이스 —
MuJoCo WASM으로 물리 계층만 교체 가능한 지점, CLAUDE.md §7).
```

핵심 불변식 3개 (전체는 `CLAUDE.md` §2):

1. **물리가 진실** — 트랜스폼의 단일 원천은 Rapier. three.js는 매 스텝 후 단방향
   동기화되는 거울이다.
2. **고정 timestep** — accumulator 패턴으로 렌더레이트와 물리를 분리(결정론 지향).
3. **충돌은 EventQueue로만** — `world.step(eventQueue)` 유래 이벤트가 유일한 충돌
   진실. 메시 겹침 추정 금지.

문서: **`docs/USAGE.md`(사용법 — 조작·작업 절차·문제 해결)** ·
**`docs/TECH_STACK.md`(구현 기술 스택 한 장 요약)** · `docs/PRD.md`(목표·범위) ·
`docs/ARCHITECTURE.md`(계층·데이터 흐름) · `docs/DATA_MODEL.md`(스키마 규범) ·
`docs/SIMULATION.md`(루프·player·충돌) · `docs/UX_DESIGN.md`(화면 설계) ·
`docs/PLANNER.md`(자연어 플래너 설계) · `docs/ROADMAP.md`(마일스톤·게이트).

## 개발 하네스 (AI-native 방식)

이 저장소는 Claude Code 같은 AI 에이전트가 안전하게 작업하도록 설계된 하네스를
갖는다:

- **`CLAUDE.md`** — 프로젝트 헌법. 절대 불변식(§2)·계층 규칙(§3)·충돌 그룹 규약(§5)·
  Definition of Done(§8)을 정의하고, 개별 작업 지시보다 우선한다.
- **`AGENTS.md`** — 리뷰 역할 5종: Physics Correctness / Layer Boundary /
  Determinism / Schema & Contract, 그리고 완료 보고 전 항상 도는
  **Conservative Self-Critique(거부권)** — 과장·은폐를 막는 정직성 게이트.
- **`EXPERIMENTS.md`** — append-only 결정 로그. 모든 설계 결정의 "왜"와 측정치를
  기록한다.
- **검증 게이트** — `npm run verify`(tsc·eslint·vitest 단위 테스트)와
  `scripts/gate-browser.mjs`(실브라우저 물리 어서션). 게이트를 실제로 통과해야만
  Phase가 "완료"다(ROADMAP).
- **`.claude/commands/`** — `/new-scene`(데이터만으로 씬 추가), `/add-control-step`
  (스키마→핸들러→검증 워크플로), `/verify-gate` 슬래시 커맨드.

## 새 씬 추가법 — 코드 수정 없이

씬은 데이터다(CLAUDE.md §2.5). `/new-scene` 커맨드의 절차:

1. `docs/DATA_MODEL.md`의 `SceneSpec` 스키마를 따라
   `src/assets/scenes/<이름>.scene.json` 작성(Y-up, 미터, 라디안, 쿼터니언).
2. 필요하면 `src/assets/sequences/<이름>.sequence.json`(ControlSequence)도 작성 —
   robot/joint 참조는 씬에 실제 존재해야 검증을 통과한다.
3. `src/main.ts`의 `SCENE_REGISTRY`에 한 줄 추가(프리셋 select에 자동 반영).
   레지스트리 등록 없이도 📂 업로드 버튼으로 `{scene, sequence}` 봉투 JSON을
   바로 로드할 수 있다.
4. 충돌을 감지할 collider에는 `emitEvents: true` + 상대 그룹을 `collidesWith`에
   포함(그룹 규약은 CLAUDE.md §5). 동적 바디에 `trimesh` 금지.

## 한계 (정직한 상태 — PRD §6 Non-goals)

- **파지(grasping) 물리 없음** — 그리퍼는 관절 개폐일 뿐, 마찰 기반 파지·정밀 접촉은
  시뮬레이션되지 않는다(픽앤플레이스 데모는 "밀기"로 구현). 정밀 접촉이 필요해지면
  물리 계층만 MuJoCo WASM으로 교체하는 경로를 설계에 반영해 두었다(CLAUDE.md §7).
- **kinematic 링크 스윕 한계** — 로봇 링크는 kinematicPosition 바디로 구동된다.
  매우 빠른 링크 이동은 CCD를 켜도 kinematic 스윕 특성상 얇은 사물에 대한 충돌
  감지를 놓칠 수 있다. 데모 시퀀스는 이 한계 안의 속도로 작성되어 있다.
- **안전 인증 부적합** — 물리 정확도 인증이 없다. 교육·프로토타입·데모용이며,
  안전 검증 용도로 쓰면 안 된다.
- **IK 없음** — 관절 공간 직접 제어만 지원(카테시안 `moveToPose`는 로드맵).
- **센서 시뮬 없음** — 카메라 이미지·LiDAR·depth는 범위 밖.
- **plan-then-execute MVP** — 플래너가 노드를 미리 다 만든 뒤 실행한다. 각 노드 실행
  피드백으로 다음 행동을 적응시키는 폐루프(agentic) 실행은 백로그(`PLANNER.md`).
- **재실행 결정론 한계** — 노드/마커에서의 재실행은 per-node 씬 스냅샷이 없어 항상 처음부터
  되감아 빨리감기하며, 목표 노드에서 "경계 ±1 tick"에 멈춘다(정밀 재개점은 백로그).
- **규칙 기반 플래너의 범위** — 기본 백엔드는 오프라인 규칙 기반이라 지원 패턴이 제한적이다
  (⚙에서 Anthropic 백엔드 선택 시 이 세션 브라우저에서 직접 호출 — 교육/프로토타입용).
