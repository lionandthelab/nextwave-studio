# UX_DESIGN — robot-sim-web

화면 UX 설계서. 자연어 입력 → 제어 플로우 생성 → 노드 그래프 편집 → 시뮬레이터
실행까지의 사용자 경험을 정의한다. 데이터 모델은 `DATA_MODEL.md`, 실행 엔진은
`SIMULATION.md`, 자연어 생성은 `PLANNER.md`를 함께 참조한다.

---

## 1. UX 목표와 원칙

- **한 화면에서 완결**: 씬 구성 · 플로우 편집 · 실행 관찰을 모드 전환 없이 병렬로.
- **직접 조작(direct manipulation)**: 드래그앤드롭으로 물체·로봇·노드를 다룬다.
  숨은 메뉴보다 눈에 보이는 핸들·기즈모·카드 우선.
- **실행 동기화가 핵심 순간**: 로봇이 움직일 때(뷰포트) 어떤 노드가 실행 중인지
  (그래프)가 동시에 강조된다. 이 연결이 이 도구의 "이해 가능성"을 만든다.
- **생성은 초안, 사람은 편집자**: 자연어 플래너는 플로우 초안을 만들 뿐. 사용자가
  노드를 검토·수정·재정렬한 뒤 **Play를 눌러야 실행**된다(human-in-the-loop).
- **점진적 공개(progressive disclosure)**: 기본은 단순하게, 세부 파라미터는 인스펙터에서.

---

## 2. 정보 구조 (전체 워크스페이스)

단일 페이지 스튜디오. 상단 커맨드바 + 3존(좌 라이브러리 / 중앙 뷰포트+그래프 / 우
인스펙터) + 하단 독.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ COMMAND BAR                                                                         │
│ [≡] robot-sim-web   [Scene ▾][💾][📂][＋New]                                        │
│ 🗣  "로봇에게 시킬 일을 자연어로…______________________________"  [✨ Generate]      │
│                                   [▶Play][⏸][⏹][⏭Step]  [속도 1×▾]   [⚙][{} JSON]  │
├────────────┬──────────────────────────────────────────────────┬────────────────────┤
│ LIBRARY    │  VIEWPORT (3D Simulator)                     [⤢][□]│  INSPECTOR         │
│            │  ┌──────────────────────────────────────────────┐ │  (선택 대상에 따라) │
│ ▾ Objects  │  │           ╱│                                   │ │                    │
│  □ Box     │  │          ╱ │   🦾  (robot arm)                 │ │  ▸ 노드 선택 시:    │
│  ○ Sphere  │  │     ▣ box_a      ▣ box_b                       │ │    · 타입 헤더      │
│  ◍ Cylinder│  │  ───────────────── grid ──────────────────    │ │    · 파라미터       │
│  ⬭ Capsule │  │  gizmo: ⤡move ⟳rotate ⤢scale                  │ │    · 대상 로봇      │
│  ▭ Plane   │  └──────────────────────────────────────────────┘ │    · 활성/비활성     │
│            │  simTime 2.34s · ● Running · node 3/7              │ │                    │
│ ▾ Robots   ├──────────────────────────────────────────────────┤ │  ▸ 오브젝트 선택 시:│
│  🦾 Arm-6  │  FLOW GRAPH (n8n형)                     [fit][−][+]│ │    · 이름/ID       │
│  🦾 Arm-7  │  ┌────┐  ┌─────────┐  ┌───────┐  ┌────┐            │ │    · Transform     │
│  ⌸ SCARA   │  │Start│─▶│MoveJoint│─▶│Gripper│─▶│Wait│─▶ …       │ │    · W / H / L     │
│  ✋ Gripper │  └────┘  └─────────┘  └───────┘  └────┘            │ │    · Physics       │
│            │           ▲ 드래그로 재정렬 · 클릭 편집 · ＋삽입     │ │                    │
│ ▾ Import   │                                                    │ │  (로봇/메시는       │
│  ⬆ 3D 파일 │                                                    │ │   전용 필드)        │
├────────────┴──────────────────────────────────────────────────┴─┴────────────────────┤
│ DOCK   [◷ Timeline ▓▓▓▓░░░ 3/7]   [⚠ Collision Log]   [🖥 Console]                    │
│        t=2.31s  arm × box_a  ● start (contact)                                        │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

**레이아웃 규칙**
- 중앙은 **Viewport(상) / Flow Graph(하) 분할 페인**. 리사이즈 가능, 각 페인 최대화 가능.
- 기본 비율 뷰포트 55 / 그래프 45. `[⤢]`로 어느 쪽이든 전체화면 토글.
- 좌/우 패널은 접기 가능(아이콘 레일만 남김). 좁은 화면에서 자동 접힘.
- 실행(Run) 중에는 뷰포트가 우선 확장되지만 그래프는 활성 노드 표시용으로 최소 높이 유지.

---

## 3. 패널별 상세

### 3.1 Command Bar (상단)

| 요소 | 동작 |
|-----|------|
| Scene ▾ / 💾 / 📂 / ＋New | 씬 저장·불러오기·신규(SceneSpec 파일 왕복) |
| 🗣 자연어 입력 | 로봇에게 시킬 일을 문장으로. `PLANNER.md`로 전달 |
| ✨ Generate | 플래너 실행 → 그래프 생성/갱신. 실행 중 스피너 + "생성 중…" |
| ▶ Play / ⏸ / ⏹ | 실행 오케스트레이터 제어(§5) |
| ⏭ Step | 노드 하나만 실행(디버깅) |
| 속도 1×▾ | 0.25× / 0.5× / 1× / 2× / 4× |
| {} JSON | 현재 ControlSequence 원본 JSON 뷰어(읽기 + 복사/내보내기) |

- **생성 모드 선택**: 입력창 우측 토글로 `교체`(새 플로우로 대체) / `이어서`(기존 뒤에
  추가). 기본 `교체`.
- **입력창 상태**: idle / 생성 중(비활성+스피너) / 성공(초록 체크 토스트) /
  명확화 필요(§4.1 Flow 1의 clarify 카드) / 오류(빨강 토스트 + 사유).

### 3.2 Library (좌)

물체·로봇 템플릿을 카드로 제공하고, 뷰포트로 드래그해 추가한다.

- **섹션**: Objects(프리미티브) / Robots(로봇팔 템플릿) / Import(3D 파일).
- **카드**: 썸네일 + 이름. 호버 시 간단 설명. 검색/필터 상단 고정.
- **Objects**: Box, Sphere, Cylinder, Capsule, Plane. 드롭 시 기본 치수로 생성 후
  인스펙터에서 조정.
- **Robots**: Arm-6(6-DOF), Arm-7, SCARA, Gripper 등. 드롭 시 `home` 포즈로 배치.
- **Import ⬆**: 카드 클릭 = 파일 선택, 또는 파일을 뷰포트로 직접 드래그(§4.4).
- 섹션 접기/펼치기, 최근 사용 항목 상단 노출.

### 3.3 Viewport (중앙 상단) — 3D 시뮬레이터

- **카메라**: orbit / pan / zoom. `Home` 키로 프레이밍 리셋, `F`로 선택 대상 포커스.
- **그리드 + 바닥**, 축 기즈모(우하단), 조명.
- **선택**: 클릭 → 아웃라인 하이라이트 + Transform 기즈모. 빈 곳 클릭 = 선택 해제.
  다중 선택(Shift+클릭), 박스 선택.
- **Transform 기즈모**: 이동/회전/스케일 토글(단축키 `W`/`E`/`R`). 스냅(Shift 홀드,
  격자/각도 스냅 값은 설정).
- **라이브러리 드래그 프리뷰**: 드래그 중 반투명 고스트를 바닥 레이캐스트 지점에 표시,
  드롭 시 그 위치에 생성.
- **실행 오버레이**: 좌하단에 재생 상태 · simTime · 활성 노드 배지.
- **충돌 시각화**: 충돌 이벤트 발생 시 관련 오브젝트가 빨강 펄스로 깜빡이고, 접촉점
  마커 표시(옵션). 하단 Collision Log와 연동(로그 클릭 → 해당 오브젝트 포커스).
- **스케일 vs 정밀 치수**: 기즈모 스케일은 빠른 조정용. 정확한 값은 인스펙터에서 입력.

### 3.4 Flow Graph (중앙 하단) — n8n형 노드 에디터

`ControlSequence`를 노드 체인으로 시각화·편집한다. 뷰 모델은 §6.

- **노드**: 좌측 아이콘 + 타입명 + 요약(예: "MoveJoints · 2.0s") + 상태 점.
- **엣지**: 실행 순서(순차). `goto`는 label 노드로 향하는 곡선 백엣지(루프)로 표현.
- **상호작용**
  - **클릭** → 인스펙터에 파라미터 표시.
  - **드래그 재정렬** → 노드를 체인 내 다른 위치로 끌어 두 노드 사이에 드롭하면
    그 지점에 삽입되고 `steps` 배열 순서가 갱신된다. 드롭 지점에 삽입선 프리뷰.
  - **＋ 삽입** → 캔버스의 ＋ 또는 엣지 위 ＋ 클릭 → 노드 팔레트 팝오버(step 종류별
    분류) → 선택 위치에 삽입.
  - **삭제** → 선택 후 `Del` 또는 노드 컨텍스트 메뉴. 삭제 시 앞뒤 엣지 자동 재연결.
  - **활성/비활성** → 노드를 실행에서 임시 제외(회색 처리, 직렬화엔 `enabled:false`).
  - **복제** → `Ctrl/Cmd+D`.
- **캔버스**: 팬/줌, fit-to-view, 미니맵(노드 많을 때).
- **실행 상태 색**: pending(회색) · active(강조 펄스) · done(초록 체크) ·
  error/collision(빨강). 실행 커서가 그래프를 따라 이동.
- **출처 배지**: 플래너 생성 노드엔 `AI` 배지. 사용자가 편집하면 `수정됨` 표시.
- **원본 JSON 토글**: 상단 `{} JSON`과 연동, 그래프 편집이 즉시 JSON에 반영됨을 확인.

### 3.5 Inspector (우) — 컨텍스트 인스펙터

선택 대상에 따라 내용이 바뀐다.

**(A) 오브젝트(씬 엔티티) 선택 시**
- 이름/ID(편집 가능), 태그.
- **Transform**: position X/Y/Z, rotation(오일러 입력→내부 쿼터니언 변환), 스크럽 가능.
- **Dimensions(핵심)**: 프리미티브별 치수를 숫자 입력 + 드래그 스크럽으로 쉽게 조정.
  - Box → 너비/높이/길이(W/H/L)
  - Sphere → 반지름
  - Cylinder/Capsule → 반지름 + 높이
  - 값 변경 시 뷰포트 메시와 collider가 함께 갱신(단일 진실은 물리).
- **Physics**: bodyType(dynamic/fixed/kinematic), density/mass, friction, restitution,
  충돌 그룹(CLAUDE.md §5), `emitEvents`, sensor 여부, CCD.
- **로봇 선택 시**: 관절 목록(현재값 슬라이더 + limits 표시), `home` 포즈, 그리퍼 설정,
  self-collision 토글.
- **임포트 메시 선택 시**: 스케일/단위, up-axis, collider 생성 전략(convexHull/AABB/
  trimesh-static), 피벗, object/static 지정.

**(B) 노드(플로우) 선택 시** — step 종류별 폼
- `moveJoints`: 대상 관절 선택 + 목표값(슬라이더, limits 클램프) · durationSec · easing.
- `setJoints`: 관절 목표값(즉시).
- `gripper`: open/close/0..1 · durationSec.
- `wait`: durationSec.
- `waitForCollision`: 엔티티 쌍 선택(드롭다운, 씬의 엔티티로 채워짐) · timeoutSec.
- `goto`/`label`: label 이름 · 반복 횟수.
- `moveToPose`(로드맵): 목표 Transform(카테시안) — IK 주입.
- 공통: 대상 로봇 선택(다중 로봇 시), 노드 노트, 활성/비활성.

### 3.6 Dock (하단)

탭 전환: Timeline / Collision Log / Console.

- **Timeline**: 시퀀스 진행 바 + 노드 마커. 현재 노드/총 노드, simTime. 마커 클릭 →
  해당 노드로 하이라이트 점프(재생 위치 이동은 §5 참조).
- **Collision Log**: 충돌 이벤트 스트림(시간 · 엔티티 쌍 · phase · kind). 엔티티로 필터.
  행 클릭 → 관련 오브젝트 포커스 + 당시 활성 노드 강조.
- **Console**: 플래너 메시지, 스키마 검증 오류, 경고.

---

## 4. 핵심 UX 플로우

### 4.1 Flow 1 — 자연어 → 실행 가능한 플로우

```
1. (선행) 씬에 로봇 + 물체가 있다.
2. 커맨드바에 문장 입력 → ✨Generate.
3. 플래너가 현재 씬을 그라운딩(로봇 'arm'·관절·물체 'box_a' 위치 등)해 초안 생성.
4. 그래프가 노드로 채워지고, 토스트로 "가정" 요약 노출
   (예: "box_a를 대상으로 가정, 왼쪽=−X로 해석").
5. 모호하면 clarify 카드 노출:
   ┌───────────────────────────────────────────────┐
   │ 어느 박스를 집을까요?                          │
   │  ( box_a )  ( box_b )  ( 직접 지정 )            │
   └───────────────────────────────────────────────┘
   사용자 선택 → 그래프 갱신.
6. 사용자가 노드 검토·파라미터 조정·재정렬.
7. ▶Play → 노드 하나씩 실행, 활성 노드 강조, 로봇 이동, 충돌 로깅.
8. 예기치 않은 충돌 시 해당 노드 빨강 표시 → ⏸ 후 그 노드 편집 → 재개.
```

### 4.2 Flow 2 — 씬 구성(드래그앤드롭)

```
1. Library에서 로봇팔 카드를 뷰포트로 드래그 → home 포즈로 배치.
2. Box 카드 드래그 × N → 바닥 위 커서 지점에 배치.
3. 각 박스 선택 → 인스펙터에서 W/H/L·위치 조정(숫자 입력/스크럽).
4. Physics(그룹·마찰·bodyType) 확인. 감지 대상엔 emitEvents 켜짐 확인.
5. 씬은 SceneSpec으로 자동 반영(💾로 파일 저장).
```

### 4.3 Flow 3 — 노드 편집(n8n형)

```
· 드래그로 순서 변경, 엣지 ＋로 중간 삽입, 인스펙터로 파라미터 수정, Del로 삭제.
· 모든 편집이 즉시 ControlSequence JSON에 반영({} JSON에서 확인·복사·내보내기).
· 비활성 노드는 실행에서 제외되나 순서는 유지.
```

### 4.4 Flow 4 — 3D 파일 임포트

```
1. 파일을 뷰포트로 드래그(또는 Library ⬆ 클릭 → 파일 선택).
2. Import 다이얼로그:
   ┌──────────────────────────────────────────────┐
   │ 형식 감지: glTF (.glb)                          │
   │ 단위/스케일:  [1.0]  (m ▾)                      │
   │ Up-axis:     ( Y-up )  ( Z-up→변환 )            │
   │ Collider:    ( Convex Hull )( AABB )( Trimesh-정적 )│
   │ 유형:        ( Object · 동적 )  ( Static · 고정 ) │
   │ [미리보기]                       [취소] [추가]   │
   └──────────────────────────────────────────────┘
3. 추가 → 씬 엔티티로 편입, 인스펙터에서 재조정 가능.
```

- **지원 형식(MVP)**: glTF/glb, STL, OBJ(+MTL). 로봇은 URDF 전용 임포트 경로(별도).
- **안전 규칙**: 동적 바디에 trimesh 금지 → 기본 Convex Hull 권장(DATA_MODEL §2).
- **큰 파일**: 로딩 진행률 표시, 실패 시 사유 안내(형식 미지원/메시 누락 등).

---

## 5. 실행 오케스트레이션 UX ("하나씩 요청")

"JSON대로 시뮬레이터에 동작을 하나씩 요청"하는 실행 계층. 내부적으로는 `SIMULATION.md`의
`ControlPlayer` + `Engine`을 감싸 **노드 단위 상태와 컨트롤**을 UI에 노출한다.

```
Play 누름
 └▶ Orchestrator가 그래프를 순서대로 순회
      for each node in flow(순서):
        node.status = active         # 그래프 강조 + 뷰포트 배지
        dispatch(node.action)        # player가 setpoint/배리어 적용
        await 완료조건               # duration 경과 / waitForCollision 충족 등
        collect(feedback)            # pose, 충돌 이벤트 → Dock·뷰포트 반영
        node.status = done | error   # 충돌/타임아웃이면 error
      끝 → idle (loop면 처음으로)
```

- **컨트롤**: ▶Play(연속) / ⏸Pause(현재 노드 경계에서 정지) / ⏹Stop(초기화) /
  ⏭Step(노드 1개만) / 속도.
- **동기화**: 활성 노드 강조 ↔ 뷰포트 로봇 동작 ↔ Timeline 커서가 항상 일치.
- **충돌 인지 정지(옵션)**: 설정에서 "예기치 않은 충돌 시 자동 일시정지" 토글. 켜면
  로봇–환경/사물의 비의도 충돌에서 자동 ⏸ + 해당 노드 강조.
- **재실행**: Timeline 마커/노드 클릭 → 그 노드부터 재실행(씬 상태를 노드 시작 시점
  스냅샷으로 되돌리거나, 처음부터 결정론적으로 재생 — 결정론 원칙과 정합).

> 플래너가 노드를 미리 다 만든 뒤 실행하는 것이 MVP(plan-then-execute). 각 노드 실행
> 피드백으로 플래너가 다음 행동을 적응시키는 폐루프(agentic) 실행은 백로그(`PLANNER.md`).

---

## 6. Flow Graph ↔ ControlSequence 뷰 모델

Flow Graph는 선형 `ControlSequence.steps[]`에 대한 **편집 가능한 뷰**다. 항상 유효한
ControlSequence로 상호 변환된다(불변식).

```ts
interface FlowNode {
  id: string;                       // = ControlStep.id
  kind: ControlStep['kind'];        // moveJoints | gripper | wait | ...
  params: Record<string, unknown>;  // 해당 step 필드 값
  enabled: boolean;                 // false면 실행 제외(순서 유지)
  origin: 'generated' | 'manual' | 'modified'; // 출처 배지용
  note?: string;
  ui: { x: number; y: number };     // 캔버스 좌표(레이아웃 전용, 실행 무관)
  status?: 'pending' | 'active' | 'done' | 'error'; // 런타임
}

interface FlowEdge {
  from: string; to: string;
  kind: 'seq' | 'loop';             // 순차 | goto 백엣지
}

interface FlowGraph {
  nodes: FlowNode[];
  edges: FlowEdge[];
  robot: string;                    // 기본 대상 로봇
}

// 변환(무손실, 검증 통과 보장)
function toSequence(graph: FlowGraph): ControlSequence;   // 실행/저장용
function fromSequence(seq: ControlSequence): FlowGraph;   // 플래너 출력 로드/편집용
```

**불변식**
- 그래프 편집으로 **직렬화 불가능한 상태를 만들 수 없다.** 모든 편집 후 `toSequence`가
  스키마 검증(참조 무결성·그룹 유효성)을 통과해야 하며, 실패 시 편집이 거부/보정된다.
- `ui.x/y`와 `status`는 순수 표현 상태로 실행 결과에 영향하지 않는다.
- 노드 순서(체인)는 `steps` 배열 순서와 1:1.

---

## 7. 상태와 피드백(빈/로딩/오류)

| 상태 | 표현 |
|-----|------|
| 빈 씬 | 뷰포트 중앙 안내("Library에서 로봇/물체를 드래그하거나 3D 파일을 놓으세요") |
| 빈 플로우 | 그래프에 Start 노드 + "자연어로 생성하거나 ＋로 노드를 추가" |
| 생성 중 | 커맨드바 스피너, 그래프 위 반투명 로딩, 입력 비활성 |
| 명확화 필요 | clarify 카드(옵션 버튼) — Flow 1 |
| 검증 오류 | Console + 해당 노드/필드 빨강 표시 + 사람이 읽을 수 있는 사유 |
| 실행 중 | 활성 노드 펄스, Timeline 진행, 뷰포트 배지 |
| 충돌 감지 | 오브젝트 빨강 펄스 + Collision Log 행 + (옵션) 자동 ⏸ |
| 임포트 실패 | 다이얼로그 내 사유(형식 미지원/메시 누락) + 재시도 |

- **되돌리기/다시하기**: 씬·그래프 편집 전반에 Undo/Redo(`Ctrl/Cmd+Z` / `Shift+…`).
- **저장 상태 표시**: 변경 시 "● 저장 안 됨", 저장 후 체크.

---

## 8. 레이아웃 모드 / 반응형

- **와이드(기본)**: 3존 + 독 전체 표시.
- **중간 화면**: 좌/우 패널을 아이콘 레일로 접고, 클릭 시 오버레이로 확장.
- **뷰포트/그래프 포커스**: `[⤢]`로 한쪽 전체화면(집중 편집/집중 관찰).
- **좁은 화면(태블릿)**: 상단 세그먼트 컨트롤로 Viewport / Flow / Library를 전환하되,
  실행 중에는 Viewport+활성 노드 스트립을 유지(동기화 원칙 보존).
- 데스크톱 우선. 모바일은 뷰어(읽기) 수준으로 축소.

---

## 9. 키보드 · 접근성

- **단축키**: Space(Play/Pause), `←/→`(Step), `W/E/R`(기즈모 모드), `F`(포커스),
  `Del`(삭제), `Ctrl/Cmd+D`(복제), `Ctrl/Cmd+Z/Shift+Z`(undo/redo), `Home`(카메라 리셋).
- **접근성**: 인스펙터 폼은 라벨·키보드 조작 완전 지원. 노드/카드에 role·aria.
  색만으로 상태를 전달하지 않음(아이콘/텍스트 병행: active=펄스+"실행중", error=빨강+⚠).
- **피드백**: 파괴적 동작(노드/오브젝트 삭제, 씬 교체)엔 확인 또는 즉시 Undo 토스트.

---

## 10. 컴포넌트 인벤토리(구현 지도)

`src/ui` 하위 구성(계층 규칙은 CLAUDE.md §3: `ui → core → {render, schema}`).

| 컴포넌트 | 위치 | 책임 |
|---------|------|------|
| CommandBar | `ui/command-bar` | 자연어 입력·생성 트리거·재생 컨트롤·씬 저장/로드·JSON 뷰어 |
| Library | `ui/library` | 오브젝트/로봇 카드, 검색, 3D 임포트 진입 |
| Viewport | `ui/viewport` | 3D 상호작용(선택·기즈모·배치·충돌 시각화), render 계층 래핑 |
| FlowGraph | `ui/flow-graph` | n8n형 노드 캔버스, 드래그 재정렬·삽입·삭제, 상태 색 |
| Inspector | `ui/inspector` | 컨텍스트 폼(오브젝트 치수/Physics · 노드 파라미터) |
| Dock | `ui/dock` | Timeline · Collision Log · Console |
| ImportDialog | `ui/library` | 3D 파일 형식/스케일/up-axis/collider 설정 |
| Orchestrator | `ui/orchestrator.ts` | 노드 단위 실행·상태 이벤트(§5), core Engine/Player 래핑 |
| clarify/toast | `ui/feedback` | 명확화 카드·토스트·오류 표시 |

- 뷰 상태(선택·하이라이트·레이아웃)는 `ui` 소유. 시뮬 진실은 `core`(물리) 소유.
- 실제 시각 구현 단계에서 `/mnt/skills/public/frontend-design/SKILL.md` 원칙 적용.

---

## 11. 로드맵 연계

이 문서의 기능은 `ROADMAP.md`의 **UI/Planner 트랙**으로 구현한다: Scene Builder UI →
Flow Graph 에디터 → NL Planner → 실행 오케스트레이션 UI. 각 Phase 게이트를 따른다.
