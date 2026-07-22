# PLANNER — robot-sim-web

자연어 문장을 로봇 제어 액션 플로우(`ControlSequence`)로 변환하는 서브시스템 설계.
UX는 `UX_DESIGN.md` Flow 1, 실행은 `SIMULATION.md`, 스키마는 `DATA_MODEL.md` 참조.

---

## 1. 역할과 위치

`src/planner`는 **자연어 + 씬 컨텍스트 → 검증된 ControlSequence**를 생성한다. 계층
규칙상 `ui → planner → schema`이며, planner는 물리/렌더에 의존하지 않는다.

```
[자연어 입력] ─┐
               ├─▶ Planner ──▶ (검증) ──▶ ControlSequence ──▶ FlowGraph(UX) ──▶ Orchestrator(실행)
[Scene 컨텍스트]┘                 │
                                 └─(무효/모호)─▶ 복구 재시도 / 명확화 요청(UX)
```

- **생성은 초안**이며, 사람이 그래프에서 검토·수정 후 Play해야 실행된다(human-in-the-loop).
- planner 출력은 **반드시 스키마 검증을 통과한 뒤에만** 실행 대상이 된다(불변식).

---

## 2. 입력: Scene 컨텍스트 그라운딩

LLM이 존재하지 않는 로봇/관절/물체를 지어내지 않도록, 현재 씬을 구조화해 전달한다.

```ts
interface SceneContext {
  robots: {
    id: string;
    joints: {
      name: string;
      type: 'revolute' | 'prismatic';
      limits: [number, number];      // rad 또는 m
      current: number;
    }[];
    gripper?: { name: string; open: number; close: number };
    homePose: Record<string, number>;
  }[];
  objects: {
    id: string;
    type: 'object' | 'static';
    position: [number, number, number];
    dimensions?: Record<string, number>;  // box: {w,h,l} 등
    tags?: string[];
  }[];
  capabilities: {
    stepKinds: string[];             // 사용 가능한 ControlStep 종류
  };
  frame: 'y-up, meters, radians';    // 좌표 규약 명시
}
```

- `scene-context.ts`가 `SceneSpec` + 현재 상태에서 이 컨텍스트를 직렬화한다.
- **방향어 해석 규약**을 명시적으로 포함/전달한다(예: 왼쪽=−X, 위=+Y, 앞=+Z). 해석은
  결과의 `assumptions`로 사용자에게 다시 노출한다.

---

## 3. 출력 계약(프로토콜)

planner는 세 가지 중 하나를 반환한다. UX는 이에 따라 분기한다.

```ts
type PlannerResult =
  | { type: 'sequence'; sequence: ControlSequence; assumptions?: string[] }
  | { type: 'clarify';  question: string; options?: string[] }   // 모호 → 사용자에게 질문
  | { type: 'error';    message: string };                        // 생성 실패
```

- `sequence`: 검증 통과한 ControlSequence + 어떤 가정을 했는지(방향·대상 물체 등).
- `clarify`: 대상/의도가 모호할 때. UX는 옵션 버튼 카드로 노출(Flow 1의 clarify 카드).
- `error`: 복구 재시도 후에도 실패. 사람이 읽을 수 있는 사유.

---

## 4. 생성 파이프라인

```
1. buildContext(scene)                     # SceneContext 직렬화
2. prompt = system(schema+context+rules) + user(자연어)
3. raw = LLM(prompt)                        # JSON만 출력하도록 강하게 제약
4. parsed = extractJson(raw)                # 코드펜스/서문 제거 후 파싱
5. result = validate(parsed)               # zod 스키마 + 참조 무결성
   ├─ 성공 → { type:'sequence', ... }
   ├─ 스키마 실패 → repairLoop (오류를 다시 LLM에 전달, 최대 N회)
   └─ 모델이 clarify를 요청 → { type:'clarify', ... }
6. N회 초과 → { type:'error', ... }
```

### 4.1 프롬프트 설계 원칙

- **출력 제약**: "오직 유효한 JSON만 출력. 서문/설명/코드펜스 금지." 스키마를 시스템
  프롬프트에 명시(ControlStep 유니온, 필드, 단위).
- **그라운딩 강제**: robots/joints/objects는 **주어진 SceneContext의 id만 사용**.
  없는 대상이 필요하면 값을 지어내지 말고 `clarify`를 반환하도록 지시.
- **안전 규약**: 관절 목표는 limits 내로. 위험/무의미 동작은 거부하고 clarify.
- **가정 노출**: 방향어·기본 duration 등 가정을 `assumptions`에 요약.
- **결정성 힌트**: 동일 입력에 안정적 출력을 위해 온도 낮게, 예시(few-shot) 1~2개.

### 4.2 검증 & 복구 루프

- `validate-repair.ts`가 zod 스키마 + 프로젝트 규칙(참조 무결성: robot/joint/object가
  SceneContext에 존재, 그룹 유효성, trimesh+dynamic 금지 등)을 검사.
- 실패 시 **구체적 오류 메시지**(어느 필드가 왜 틀렸는지)를 모델에 되먹여 재생성.
- 재시도 상한(예: 3회) 초과 → `error`. 부분 성공(일부 노드만 유효)은 채택하지 않는다.

---

## 5. 실행과의 접점

- planner 출력(sequence) → `fromSequence`로 FlowGraph 로드(UX_DESIGN §6) → 사용자 검토.
- ▶Play → Orchestrator가 **노드 하나씩** 시뮬레이터에 요청(UX_DESIGN §5, SIMULATION §3).
- planner는 실행을 직접 하지 않는다. 실행 진실·충돌 감지는 `core`가 소유.

### 5.1 폐루프(agentic) 실행 — 백로그

MVP는 plan-then-execute. 향후 각 노드 실행 후 시뮬레이터 피드백(pose·충돌)을 planner에
되먹여 다음 행동을 적응시키는 폐루프를 옵션으로 지원.

```
loop:
  action = planner.next(goal, worldFeedback)   # 다음 한 스텝 제안
  if action == done: break
  worldFeedback = orchestrator.execute(action) # pose, collisions 반환
```

- 각 스텝도 동일하게 **검증 후 실행**, 예기치 않은 충돌 시 정지·재계획.
- 무한 루프/한계 초과 가드, 사람 개입 지점 유지.

---

## 6. 안전 · 신뢰 경계

- **검증 없인 실행 없음**: 미검증/무효 출력은 절대 시뮬레이터로 보내지 않는다.
- **사람 확인**: 생성 직후 자동 실행하지 않는다. 사용자가 Play를 눌러야 한다.
- **limits 준수**: 관절 목표·속도가 한계를 넘으면 클램프하거나 거부.
- **투명성**: 어떤 가정을 했는지(`assumptions`)를 항상 노출. clarify로 모호성 해소.
- **오류 가시성**: 실패·거부 사유를 Console/토스트로 사람이 읽을 수 있게.
- 이 도구는 교육·프로토타입용이며 안전 인증 용도가 아님을 UX에 명시(PRD §6).

---

## 7. 구현 인터페이스(요약)

```ts
// src/planner
interface Planner {
  generate(nl: string, ctx: SceneContext): Promise<PlannerResult>;
  // 백로그: next(goal, feedback) for closed-loop
}

function buildContext(scene: SceneSpec, live: WorldSnapshot): SceneContext;
function validateSequence(x: unknown, ctx: SceneContext): 
  | { ok: true; value: ControlSequence }
  | { ok: false; errors: string[] };
```

- LLM 호출은 어댑터로 격리(모델·엔드포인트 교체 가능).
- planner 순수 로직(컨텍스트 직렬화·검증·복구 판정)은 LLM 없이 단위 테스트 가능하게
  분리한다.

---

## 8. 로드맵 연계

`ROADMAP.md` UI/Planner 트랙의 **Phase 9(NL Planner)**에서 구현한다. 선행 조건:
Flow Graph 에디터(Phase 8)와 스키마(Phase 2). 게이트: 자연어 문장이 씬에 그라운딩된
유효 ControlSequence로 변환되고, 모호 입력은 clarify로, 실패는 명확한 오류로 처리된다.
