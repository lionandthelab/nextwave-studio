# CLAUDE.md — Workcell (repo: robot-sim-web)

> 이 파일은 Claude Code가 이 저장소에서 작업할 때 가장 먼저 읽고, 항상 준수해야 하는
> 프로젝트 헌법(harness)이다. 개별 작업 지시보다 이 문서의 규칙이 우선한다.

---

## 1. 프로젝트 한 줄 정의

**제품명은 `Workcell`이다** (상위 브랜드 NextWave Studio). 저장소 이름(`robot-sim-web`)은
제품명이 아니다 — 사용자에게 보이는 모든 곳(탭 제목·워드마크·문서)에서는 `Workcell`을 쓴다.
정체성 상수의 단일 진실은 `src/ui/brand.ts`이며, 제품명 리터럴을 다른 곳에 흩지 않는다.
저장 파일 확장자는 `.workcell.json`이고 `{scene, sequence, assets}` 봉투다.

브라우저에서 완결되는 **간소화된 IsaacSim형 로봇 시뮬레이터**. 로봇/사물 오브젝트로
가상 환경을 구성하고, 로봇 제어 시퀀스를 재생하며, **로봇–사물 충돌을 감지**한다.
백엔드 없이 정적 호스팅으로 배포 가능한 것을 1차 목표로 한다.

**Phase 12+**: 선택적 백엔드(`server/` — Fastify + SQLite 단일 파일)가 추가되어
다중 사용자(관리자/설치기사, PIN 로그인)·공정/작업/블록/장비/실행기록 공유가 켜진다.
**서버는 선택적이다** — 서버가 없으면 앱은 IndexedDB 로컬 모드로 강등되어 그대로
동작한다(정적 호스팅 불변식 유지). 규범 문서는 `docs/BACKEND.md`, 개체 스키마의
단일 진실은 `src/schema/entities.ts`(클라이언트/서버 공유 zod). 1차 사용자는
**로봇 설치기사**다: 단일 프로세스 배포·단일 파일 DB·공유 단말 PIN 로그인·큰 터치
타깃 — IT 담당자 없이 현장에서 켜지는 것이 설계 기준이다.

- **물리 엔진**: Rapier (`@dimforge/rapier3d-compat`, WASM)
- **렌더링**: three.js
- **로봇 모델 로딩**: `urdf-loader` (URDF → three.js)
- **빌드/개발**: Vite + TypeScript
- **업그레이드 경로**: 접촉 물리 정밀도가 필요해지면 물리 계층만 MuJoCo WASM
  (`@mujoco/mujoco` 공식 바인딩)으로 교체 (§7 참조)

전체 배경과 근거는 `docs/PRD.md`, `docs/ARCHITECTURE.md`를 읽는다.

---

## 2. 절대 불변식 (Non-negotiables)

이 규칙을 깨는 변경은 이유 불문 거부하고, 대안을 제시한다.

1. **트랜스폼의 단일 진실 원천은 물리(Rapier)다.**
   매 시뮬레이션 스텝 후 Rapier RigidBody의 pose를 three.js `Object3D`로 **단방향 동기화**한다.
   three.js 쪽에서 직접 위치를 바꿔 물리와 어긋나게 만들지 않는다.
   (예외: 순수 시각 요소 — 카메라, 그리드, 라이트, 디버그 헬퍼.)

2. **물리 바디와 시각 메시는 독립 객체다.**
   시각 메시는 복잡해도 되지만, collider는 가능한 한 단순한 프리미티브(box/sphere/capsule/
   convex hull)로 둔다. 매 프레임 물리 계산이 프레임 예산 안에 끝나야 한다.

3. **시뮬레이션은 고정 timestep으로 돈다.**
   렌더 프레임레이트와 물리 스텝을 분리한다(accumulator 패턴, `docs/SIMULATION.md`).
   `world.step()`에 가변 dt를 그대로 넣지 않는다. 결정론(determinism)이 목표다.

4. **충돌 이벤트는 항상 EventQueue를 통해 읽는다.**
   `world.step(eventQueue)` → `drainCollisionEvents(...)`. 충돌을 물리 스텝 바깥에서
   메시 겹침으로 추정하지 않는다. Collider handle ↔ Entity id 매핑을 유일한 진실로 유지한다.

5. **씬은 데이터로 선언한다.**
   로봇/사물/환경은 `SceneSpec`(JSON) 스키마로 기술한다(`docs/DATA_MODEL.md`).
   씬 구성 로직을 하드코딩하지 않는다. 새 씬 = 새 데이터, 코드 변경 없음이 이상적이다.

6. **제어 시퀀스도 데이터다.**
   로봇 동작은 `ControlSequence`(선언적 step 배열)로 기술하고, 별도 player가 해석한다.
   특정 로봇 동작을 엔진 코드에 박아 넣지 않는다.

7. **WASM 초기화는 비동기다.**
   `await RAPIER.init()`(compat 빌드) 완료 전에는 어떤 물리 API도 호출하지 않는다.
   부트스트랩 순서를 지킨다: WASM init → scene load → build world → start loop.

8. **Flow Graph는 ControlSequence의 뷰다.**
   n8n형 노드 그래프는 선형 `ControlSequence`와 무손실 상호 변환된다(`toSequence`/
   `fromSequence`). 그래프 편집으로 **직렬화 불가능한 상태를 만들 수 없다** — 모든 편집
   후 결과가 스키마 검증을 통과해야 하며, 노드 좌표·실행 상태는 순수 표현으로 실행에
   영향하지 않는다. (`docs/UX_DESIGN.md` §6)

9. **디자인 토큰은 `src/ui/theme.ts`가 단일 진실이다.**
   색·타입·간격·반경·모션·아이콘 치수를 모듈에 하드코딩하지 않는다. 인라인 `fontSize`
   리터럴과 raw px 간격은 금지 — `applyType(el, TYPE.*)`와 `SPACE.*`를 쓴다.
   아이콘은 `src/ui/icons.ts`의 SVG를 쓴다(**이모지·딩벳 금지** — OS마다 다른 그림이 나오고
   `currentColor`를 못 받아 상태 변화가 반쪽이 된다). 액센트는 **역할이 3분할**되어 있다:
   면(`--primary`)=주요 액션 / 보더(`--accent`)=토글 상태 / **선택은 `SELECT`(청색) 별도 축**.
   텍스트 토큰은 **가장 밝은 표면(`SURFACE.modal`) 위에서도** 대비 ≥ 4.5:1이어야 한다.

10. **전역 단축키의 단일 소유자는 `src/ui/shortcuts.ts` 라우터다.**
   모듈이 `window`에 `keydown`을 직접 걸지 않는다. 모듈은 "무엇을 할지"만 제공하고
   통합자가 라우터에 등록한다. 소유자 없는 키맵은 반드시 충돌로 수렴한다(§9 참조).
   도움말 시트는 **실제 등록된 바인딩만** 그린다 — 문서에만 있는 키를 광고하지 않는다.

11. **작업물은 문서(document)로 보존된다.**
   저장은 `SceneSpec` 단독이 아니라 `{scene, sequence, assets}` 봉투다. 시퀀스만 바뀌어도
   미저장(dirty)이며, 편집 커밋은 IndexedDB에 자동저장된다. 파괴적 동작에는 **되돌릴 경로**
   (Undo 또는 실행취소 토스트)가 반드시 있어야 한다 — 되돌릴 수 없으면 사용자는 탐색을 멈춘다.
   같은 이유로 **사람이 유발한 편집은 사물을 바닥 아래에 둘 수 없다**(`core/ground-clamp.ts`):
   static은 물리가 밀어내지 않고, dynamic은 바닥 슬래브 밑으로 빠지면 무한 낙하해 화면에서
   사라지며, 로봇 링크는 kinematic이라 자가 교정이 없다 — 지하 배치는 실수가 아니라 손실이다.
   클램프는 편집 경로(기즈모·방향키·인스펙터·배치·치수)에만 걸고 **씬 JSON 로드는 손대지
   않는다**(수작성 좌표를 조용히 고쳐 쓰면 파일과 화면이 어긋난다 — §2.5).
   복구 조작으로 `End`(바닥에 붙이기)를 제공한다.

12. **생성물은 검증·사람 승인 후에만 실행한다.**
   자연어 플래너 출력은 스키마 검증을 통과한 뒤에만 실행 대상이 되고, 자동 실행하지
   않는다 — 사용자가 검토 후 Play를 눌러야 한다(human-in-the-loop). 미검증/무효 출력은
   시뮬레이터로 보내지 않는다. (`docs/PLANNER.md` §6)

---

## 3. 저장소 지도

```
robot-sim-web/
├── CLAUDE.md              ← 이 파일 (harness)
├── AGENTS.md              ← 리뷰 역할(서브에이전트) 정의
├── EXPERIMENTS.md         ← 설계 결정/실험 로그 (append-only)
├── README.md              ← 사람용 개요 + 실행법
├── package.json           ← 고정 의존성
├── vite.config.ts         ← WASM 대응 설정
├── docs/
│   ├── PRD.md             ← 목표·범위·요구사항·non-goals
│   ├── ARCHITECTURE.md    ← 계층 구조·데이터 흐름·모듈 경계
│   ├── DATA_MODEL.md      ← SceneSpec / EntitySpec / ControlSequence 스키마
│   ├── SIMULATION.md      ← 시뮬 루프·제어 player·충돌 감지(Rapier API)
│   ├── UX_DESIGN.md       ← 화면 UX 설계서(워크스페이스·노드그래프·씬빌더·실행)
│   ├── UX_AUDIT.md        ← 5인 디자인 팀 진단 + Phase 11 실행 결과 (증거: docs/ux-audit/)
│   ├── PLANNER.md         ← 자연어 → ControlSequence 생성·검증·복구
│   └── ROADMAP.md         ← 단계별 마일스톤 + 검증 게이트
├── .claude/commands/      ← Claude Code 커스텀 슬래시 커맨드
└── src/                   ← 구현 (ROADMAP Phase 1부터 생성)
    ├── core/              ← 엔진 오케스트레이션 (프레임워크 비의존)
    │   ├── engine.ts          시뮬 루프, 부트스트랩
    │   ├── world.ts           Rapier world 래퍼
    │   ├── scene-loader.ts    SceneSpec → world + three.js
    │   ├── collision.ts       CollisionMonitor (EventQueue 소비)
    │   ├── collision-classify.ts  의도된 접촉 vs 예기치 않은 충돌 분류 (순수)
    │   ├── ground-clamp.ts    편집 배치의 바닥 하한 · 바닥에 붙이기 (순수, §2.11)
    │   ├── conveyor.ts       컨베이어 표면 구동 + 재순환 (DATA_MODEL §4.2)
    │   ├── control/
    │   │   ├── player.ts       ControlSequence 해석기
    │   │   └── steps.ts        step 종류별 핸들러
    │   └── sync.ts            Rapier pose → three.js Object3D
    ├── planner/           ← 자연어 → ControlSequence (LLM 어댑터로 격리)
    │   ├── planner.ts         생성 파이프라인
    │   ├── scene-context.ts   씬 → LLM 그라운딩 컨텍스트 직렬화
    │   └── validate-repair.ts 스키마 검증 + 복구 루프
    ├── render/            ← three.js 렌더러, 카메라, URDF 로딩
    ├── schema/            ← 타입 정의 + 런타임 검증(zod 등)
    │                         (SceneSpec/ControlSequence + FlowGraph 뷰모델 변환)
    ├── ui/                ← 화면(UX_DESIGN.md 구현)
    │   ├── theme.ts           디자인 토큰 단일 진실 (SURFACE/BORDER/TYPE/SPACE/MOTION/ICON)
    │   ├── icons.ts           SVG 아이콘 세트 (currentColor 상속, IconName 타입)
    │   ├── a11y.ts            trapFocus · rovingTabindex · createAnnouncer
    │   ├── shortcuts.ts       단축키 단일 라우터 (스코프 + 위젯 소유권 규칙)
    │   ├── document.ts        워크셀 문서 (봉투 저장 · IndexedDB 자동저장 · dirty)
    │   ├── brand.ts           제품 정체성 상수
    │   ├── help-sheet.ts      ? 도움말 · 단축키 시트
    │   ├── command-bar/       자연어 입력·생성·재생 컨트롤·씬 저장/로드·JSON 뷰어
    │   ├── library/           오브젝트/로봇 라이브러리 + 3D 임포트 다이얼로그
    │   ├── viewport/          3D 상호작용(선택·기즈모·배치·충돌 시각화)
    │   │                        + selection-hud.ts 선택 조작 HUD(이동 키 안내·치수 ±·바닥)
    │   ├── flow-graph/        n8n형 노드 에디터(드래그 재정렬·삽입·삭제)
    │   ├── inspector/         컨텍스트 인스펙터(노드 파라미터 / 오브젝트 치수·Physics)
    │   ├── dock/              타임라인 · 충돌 로그 · 콘솔
    │   ├── feedback/          명확화 카드 · 토스트 · 오류 표시
    │   └── orchestrator.ts    실행 오케스트레이터(노드 하나씩 요청·상태 이벤트)
    └── assets/            ← urdf, mesh, 라이브러리 템플릿, 샘플 Scene/Sequence
```

**계층 의존 방향 규칙**: `ui → {core, planner, api} → {render, schema}`. `core`/`planner`는
UI 프레임워크에 의존하지 않는다(React든 vanilla든 재사용 가능). `render`는 three.js를,
`core/world`·`core/collision`은 Rapier를 아는 유일한 지점이다. `planner`는 물리/렌더에
의존하지 않고 스키마만 안다.

**Phase 12+ 추가 계층** (규범: `docs/BACKEND.md`):

```
├── server/                ← 백엔드 (Fastify + better-sqlite3, 선택적)
│   ├── app.ts                 buildApp (테스트는 inject — listen 없음)
│   ├── auth.ts                scrypt PIN 해시 · 세션 · 시도 제한
│   ├── db.ts                  단일 파일 SQLite, 전진 전용 마이그레이션
│   └── routes-*.ts            auth / entities(CRUD·잠금) / runs(append-only)
└── src/
    ├── api/               ← 클라이언트 API 계층 (schema만 안다 — planner와 동렬)
    │                         ApiClient(연결 상태 머신) · 리소스 클라이언트 · 오프라인 outbox
    ├── schema/entities.ts ← 개체 5종(Process/Task/Block/Device/Run) — 서버와 공유
    ├── schema/blocks.ts   ← 블록 캡처/전개 순수 함수 (전개 후 재검증 — §2.8 정신)
    └── ui/
        ├── shell/         ← 2평면 셸: 해시 라우터 · 네비 레일 · 로그인(PIN 패드)
        └── console/       ← 콘솔 평면 화면들 (작업/공정/블록/장비/기록/설정) + 프리미티브
```

`server`는 `src/schema`(zod 계약)만 import한다 — Rapier/three/ui를 절대 모른다.
씬/시퀀스 내용은 서버가 해석하지 않는다(봉투 검증까지만 — 도메인 검증은 클라이언트 몫).
**복사본 의미론**: Task.scene은 공정 씬의 전체 사본이다(참조 아님). 공정 갱신은 자동
전파되지 않고 `sceneOrigin` 비교로 "알리고 사람이 적용"한다. 블록도 삽입 시 전개(사본)다.

---

## 4. 코딩 컨벤션

- **언어**: TypeScript strict 모드. `any` 금지(불가피하면 `// @justify:` 주석).
- **좌표계**: 내부 표준은 **Y-up**, 미터, 라디안. URDF는 Z-up이 흔하므로
  로딩 시 축 변환을 render의 URDF 래퍼(`render/urdf.ts`의 axisFix 그룹) 한 곳에서만
  처리하고 이후엔 Y-up으로 통일한다 (`core/robot-types.ts` 헤더 계약).
- **회전 표현**: 쿼터니언 `[x, y, z, w]`. 오일러각은 UI 입력 경계에서만 쓰고 즉시 변환.
- **핸들 매핑**: `Map<ColliderHandle, EntityId>`, `Map<EntityId, RigidBodyHandle>`를
  `world.ts`가 소유한다. 다른 모듈은 이 매핑을 통해서만 물리 객체에 접근한다.
- **단위 접미사**: 시간 변수는 `Sec`/`Ms`, 각도는 `Rad`/`Deg`를 이름에 명시한다.
- **부수효과 격리**: `core`의 순수 로직(스텝 보간, 시퀀스 진행)은 물리/렌더 없이
  단위 테스트 가능해야 한다.
- **매직넘버 금지**: timestepHz, gravity, group 비트마스크 등은 상수/스키마로 노출.

---

## 4-b. UI 언어 정책

한 화면에 영/한이 규칙 없이 섞이면 저자마다 다른 선택을 하게 된다. 축을 고정한다.

1. **UI 크롬(패널 제목·버튼·메뉴·탭·안내 문구)은 한국어로 통일한다.**
   `라이브러리` `인스펙터` `타임라인` `충돌 로그` `콘솔` `열기` `저장` `재생`.
2. **도메인 식별자는 영문 원문을 유지한다.** `MoveJoints` `durationSec` `position (m)`
   엔티티 id, 관절 이름 — 사용자가 JSON에서 보는 것과 일치해야 하므로 번역하지 않는다.
3. **1과 2가 만나는 지점에는 `lang="en"`을 부여한다.** 노드 kind 라벨, 로그의 엔티티 id 등.
   `lang="ko"` 아래 영문을 두면 한국어 TTS가 CamelCase를 철자로 읽는다(WCAG 3.1.2).
   SVG `<text>`에도 `lang`이 동작한다.
4. **이중 표기(`Box · 박스`)는 라이브러리 카드 한정 예외**다(첫 사용자 온보딩 목적).
   다른 곳으로 번지지 않게 한다.
5. **사용자에게 보이는 문자열에 내부 로드맵 어휘(`Phase N`)를 쓰지 않는다.**

---

## 5. 충돌 그룹 규약 (프로젝트 전역)

Rapier interaction group은 0–15의 16개 그룹만 존재한다. 이 프로젝트의 배정을 고정한다:

| 그룹 | 용도 |
|-----|------|
| 0 | `ENV` — 바닥/벽 등 정적 환경 |
| 1 | `ROBOT` — 로봇 링크 |
| 2 | `OBJECT` — 조작 대상 사물 |
| 3 | `SENSOR_ZONE` — 감지 전용 sensor collider (물리 반응 없음) |
| 4–14 | 예약 |
| 15 | `DEBUG` |

- 로봇 링크끼리의 self-collision은 기본 **비활성**(URDF 인접 링크 무시).
- "로봇–사물 충돌 감지"의 핵심 쌍: `ROBOT × OBJECT`, `ROBOT × ENV`.
- 감지만 하고 튕기지 않아야 하는 영역은 `SENSOR_ZONE`(sensor=true)로 만들고,
  **엔티티에 `tags: ['detection-zone']`을 붙여 의도를 선언한다**(`DETECTION_ZONE_TAG`).
  태그 없는 sensor 정적 엔티티는 "단단해 보이는데 모든 것이 통과하는 유령 장애물"이며
  결함으로 간주한다 — 반대로 **단단해야 할 것은 sensor가 아니어야 하고, 로봇이 있는
  씬이라면 `collidesWith`에 `ROBOT` + `emitEvents: true`**여야 접촉이 로그에 남는다.
  (쌍 필터는 양방향이다: 한쪽만 선언하면 이벤트가 0건이 된다.)
  `src/core/sample-scenes.test.ts`가 모든 샘플 씬에 이 규칙을 강제한다.

새 그룹이 필요하면 예약 슬롯에서 할당하고 이 표와 `EXPERIMENTS.md`에 기록한다.

---

## 6. 작업 워크플로 (Claude Code가 따를 절차)

### 새 기능을 추가할 때
1. 관련 설계문서(`docs/*`)를 먼저 읽어 불변식과 충돌하지 않는지 확인한다.
2. 스키마 변경이면 `src/schema`를 먼저 고치고 타입을 통과시킨 뒤 구현한다.
3. 계층 경계(§3)를 지킨다. `core`에 three.js/React를 import하지 않는다.
4. 결정 사항·트레이드오프를 `EXPERIMENTS.md`에 append-only로 남긴다.

### 새 엔티티 종류를 추가할 때 (예: 컨베이어)
`schema` 타입 확장 → `validate`의 zod 미러 + **교차 규칙** → `scene-loader`에 빌더/등록
추가 → collider/그룹 배정 → 샘플 SceneSpec 갱신 → 로드/충돌 동작 검증.

**컨베이어(`conveyor` 블록)가 이 워크플로의 레퍼런스다** (DATA_MODEL §4.2). 그때 배운 것:

- 새 거동이 매 스텝 물리를 만져야 하면 `RobotRegistry` 패턴을 따른다 — 레지스트리를
  `SceneHandle`이 소유하고 통합자가 `preStep`에서 `tickAll()`을 부른다.
- 물리 조작이 필요하면 **`PhysicsWorld` 인터페이스를 넓힌다**. Rapier 타입을 core 밖으로
  내보내지 않는다(§7) — MuJoCo 교체 시 새 구현만 추가하면 되도록.
- 빌드 시점에 캐시한 기하가 있으면 **배치를 바꾸는 편집 경로마다 다시 만들어야 한다.**
  `updateTransform`은 재빌드하지 않는다 — 여기서 조용히 어긋나기 쉽다.
- 데이터가 표현할 수 없는 조합은 `superRefine`으로 **사람이 읽을 이유와 함께** 거부한다.
  조용한 no-op은 사용자가 시행착오로 배우게 만든다.

### 새 제어 step을 추가할 때 (예: `follow-path`)
`schema`의 `ControlStep` 유니온 확장 → `control/steps.ts`에 핸들러 → `player`가 라우팅
→ **`flow-graph`에 노드 타입 + 인스펙터 폼 추가**(UX_DESIGN §3.4/§3.5) →
`fromSequence`/`toSequence` 변환 확인 → 샘플 ControlSequence 갱신 → 재생/보간 검증.
(`/add-control-step` 슬래시 커맨드 사용.)

### 자연어 플래너를 손볼 때
`docs/PLANNER.md`를 먼저 읽는다. 그라운딩 컨텍스트(`scene-context.ts`)·출력 계약
(`PlannerResult`)·검증/복구 루프의 계약을 지킨다. **미검증 출력은 실행에 노출하지 않는다**
(불변식 §2.9). LLM 어댑터로 모델 호출을 격리하고, 순수 로직은 LLM 없이 테스트한다.

### UI/UX를 구현·수정할 때
`docs/UX_DESIGN.md`를 규범으로 삼는다(워크스페이스 레이아웃·패널·플로우·상태). 실제 시각
구현 시 `/mnt/skills/public/frontend-design/SKILL.md` 원칙을 적용한다. 뷰 상태는 `ui`가,
시뮬 진실은 `core`(물리)가 소유한다 — 뷰에서 물리 pose를 역으로 쓰지 않는다.

---

## 7. MuJoCo 업그레이드 경로 (미래)

접촉 물리 사실성(그리퍼 파지, 마찰 접촉)이 필요해지면 **물리 계층만** 교체한다.
`render`(three.js), `schema`, `ui`, `ControlSequence` 계층은 재사용된다.

- 교체 지점: `core/world.ts`(강체/collider 생성·스텝), `core/collision.ts`
  (`MjData.contact` 읽기), `core/sync.ts`(body pose 소스).
- 씬 표현은 SceneSpec → MJCF(XML) 변환기를 추가하는 방식으로 흡수한다.
- 이 경계를 유지하기 위해, `core`는 Rapier 타입을 **직접 노출하지 않고**
  내부 인터페이스(`PhysicsWorld`, `PhysicsBody`, `ContactEvent`)로 감싼다. — 불변식.

---

## 8. Definition of Done (검증 게이트)

작업을 "완료"로 보고하기 전 아래를 만족해야 한다.

- [ ] `tsc --noEmit` 통과, ESLint 경고 0
- [ ] 새 UI는 `theme.ts` 토큰만 소비 (인라인 fontSize/raw px 간격/이모지 아이콘 0건)
- [ ] 새 전역 단축키는 `shortcuts.ts` 라우터에 등록 (window 직접 바인딩 0건)
- [ ] 새 파괴적 동작에 되돌릴 경로(Undo 또는 실행취소 토스트)가 있음
- [ ] 관련 순수 로직 단위 테스트 통과
- [ ] 샘플 씬이 로드되고 지정 프레임레이트에서 물리가 안정적으로 도는지 확인
- [ ] 의도한 충돌 쌍이 EventQueue로 실제 감지되어 로그에 남는지 확인
- [ ] 불변식(§2) 위반 없음, 계층 의존 방향(§3) 준수
- [ ] 결정/변경 사항 `EXPERIMENTS.md` 기록

---

## 9. 흔한 함정 (미리 알고 피할 것)

- **Vite + Rapier WASM**: 번들러 문제를 피하려고 `@dimforge/rapier3d`(top-level await/
  ESM WASM) 대신 **`@dimforge/rapier3d-compat`**(WASM base64 내장)를 쓴다. `RAPIER.init()`
  을 반드시 await한다. `vite.config.ts`의 예외 설정을 확인한다.
- **좌표 축 혼동**: URDF(Z-up)를 로드하고 축 변환을 빼먹으면 로봇이 눕는다.
  변환은 render URDF 래퍼(`render/urdf.ts` axisFix) 한 곳에서만.
- **터널링**: 빠른 링크가 얇은 사물을 통과하면 `setCcdEnabled(true)`로 CCD를 켠다.
- **스텝 중 바디 속성 접근**: (react-three-rapier 사용 시) 물리 스텝 콜백 안에서
  `translation()`/`linvel()` 등 rigidbody 속성을 직접 읽지 않는다(Rust aliasing 오류).
- **spiral of death**: 프레임 dt를 clamp한다(예: 0.1s 상한). 안 그러면 탭 백그라운드
  복귀 시 물리 스텝이 폭주한다.
- **URDF 메시 경로**: `loader.packages`로 ROS 패키지 경로를 매핑하지 않으면 메시 404.

---

## 10. 이 문서의 유지보수

- 아키텍처/불변식이 바뀌면 **먼저 이 문서와 `docs/`를 고치고** 코드를 바꾼다.
- 결정의 "왜"는 `EXPERIMENTS.md`, 결정의 "무엇"은 이 문서와 `docs/`에 반영한다.
