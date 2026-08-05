# plan.md — 지금 무엇을, 왜

> 하네스 4축 중 **의도**. `CLAUDE.md`가 "무엇을 절대 깨면 안 되는가"라면
> 이 파일은 "지금 이 시점에 무엇을 왜 하는가"다.
>
> 외부 루프의 **모든 세션이 이 파일을 첫 번째로 읽는다.**
> 그래서 이 파일은 **사람만 편집한다** — 루프에는 쓰기 권한이 없다
> (`harness/loop-settings.json`의 deny). 루프가 자기 목표를 고칠 수 있으면
> 그건 목표가 아니라 사후 합리화다.

---

## 1. 지금까지 (사실)

`docs/ROADMAP.md`의 Phase 0–10, `docs/UX_AUDIT.md`의 Phase 11(제품화)까지 완료.
`EXPERIMENTS.md`에 각 단계의 결정·트레이드오프·실측이 append-only로 쌓여 있다.

검증 자산:

- `npm run verify` — typecheck · ESLint · vitest (테스트 1158개 / 47 파일)
- `npm run gate` — playwright 브라우저 게이트 13종 (샘플 씬 7종 완주 포함)

루프는 앞의 것을 매 태스크 돌리고, 뒤의 것은 태스크가 `gate`를 선언할 때만 돌린다.
브라우저 게이트는 느리고 출력이 크다 — 상시로 돌리면 캐시 주기가 깨진다.

## 2. 이번 마일스톤

<!-- ↓ 사람이 채운다. 루프를 기동하기 전에 반드시 이 절을 실제 목표로 바꿀 것. -->

**목표**: _(미정 — 채울 것)_

**왜 지금**: _(미정 — 채울 것)_

**끝났다고 말할 수 있는 조건**:

- [ ] _(미정)_

후보(`docs/ROADMAP.md` 백로그에서):

- IK 솔버 — `moveToPose`(카테시안) step. 관절 공간 → 작업 공간 제어
- MuJoCo 물리 계층 — `MujocoWorld` + SceneSpec→MJCF 변환기 (`CLAUDE.md` §7)
- 폐루프 실행 — 노드 실행 피드백을 planner에 되먹임 (`docs/PLANNER.md` §5.1)
- 센서 시뮬 — 오프스크린 카메라 depth/RGB
- 그래프 분기 — goto/label을 넘어선 조건 분기 노드
- 성능 — collider LOD, instancing

## 3. 이번 마일스톤에서 하지 않는 것

- `CLAUDE.md` §2 불변식을 완화하는 변경. 지름길이 보이면 그건 태스크가 잘못 쪼개진 것이다.
- 요청되지 않은 리팩터링·파일 이동·의존성 추가.
- UI 문자열의 언어 정책 변경 (`CLAUDE.md` §4-b).

## 4. 태스크를 쪼개는 규칙

큐(`TASKS.jsonl`)는 **사람이 채운다.** 루프는 태스크를 만들지 않는다 —
스스로 일감을 만드는 루프는 24시간 뒤에 아무도 원하지 않은 코드를 남긴다.

한 태스크는 이래야 한다:

1. **세션 하나에 끝난다.** 컨텍스트가 한 번 죽으면 처음부터인 구조이므로,
   태스크가 세션보다 크면 영원히 완료되지 않는다. 기준: 예산 $2 안쪽.
2. **`done_when`이 기계로 확인된다.** "잘 동작한다"가 아니라
   "`bash harness/verify.sh --gate=X`가 통과한다", "테스트 `Y`가 존재하고 통과한다".
3. **선행 관계는 `deps`로 쓴다.** 큐 순서에 의존하지 않는다.
4. **커밋 메시지를 `commit` 필드에 미리 쓴다.** 저장소 관례(Conventional, 한국어)에 맞춰서.

작성 예:

```bash
node harness/hx.mjs add '{
  "id": "T-012",
  "title": "moveToPose step의 IK 실패를 사람이 읽는 오류로 변환",
  "why": "지금은 조용히 무동작이라 사용자가 시행착오로 배운다 (CLAUDE.md §6)",
  "done_when": [
    "IK 미수렴 시 PlayerError를 던지고 콘솔 독에 이유가 남는다",
    "src/core/control/steps.test.ts에 미수렴 케이스 테스트가 있다",
    "bash harness/verify.sh 통과"
  ],
  "model": "sonnet",
  "effort": "medium",
  "gate": null,
  "commit": "fix(control): moveToPose IK 미수렴을 조용한 무동작에서 명시적 오류로"
}'
```
