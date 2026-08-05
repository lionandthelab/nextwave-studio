# 하네스 × 루프 — 운용 매뉴얼

> 루프는 모델이 아니라 셸이 돌린다.
> 하네스는 컨텍스트가 죽어도 살아남는 상태다.

세션을 짧게 죽이고, 상태를 파일에 남기고, 외부 루프가 재기동한다.
모델이 스스로 "계속 할까 말까"를 판단하지 않는다 — 그 판단은 `while` 문이 한다.

---

## 1. 하네스 4축

| 축 | 파일 | 소유자 | 성질 |
|---|---|---|---|
| **의도** | `CLAUDE.md` · `plan.md` | 사람 | 왜·무엇을. 루프는 읽기만 |
| **상태** | `TASKS.jsonl` · `JOURNAL.md` | 루프 | 남은 것 · 이미 해본 것 |
| **검증** | `harness/verify.sh` | 셸 | 모델의 자기보고를 대체한다 |
| **경계** | `.claude/settings.json` · `harness/loop-settings.json` | 사람 | 권한 + 토큰 절단 훅 |

각 축이 파일이라는 점이 핵심이다. 컨텍스트는 매 이터레이션 폐기되지만 파일은 남는다.

### 왜 루프가 자기 하네스를 못 고치는가

`harness/loop-settings.json`의 deny에 `harness/**` · `.claude/**` · `CLAUDE.md` ·
`plan.md` · `TASKS.jsonl` · `JOURNAL.md` · `package.json`이 들어 있다.

검증에 걸린 모델에게 가장 싼 해법은 언제나 **검증을 고치는 것**이다.
자기 채점표를 고칠 수 있는 루프는 검증이 아니라 자기승인을 한다.
같은 이유로 `git commit`도 deny다 — 커밋은 `verify.sh`가 통과한 뒤 셸이 한다.

---

## 2. 기동

```bash
# 0) 사전: plan.md §2 "이번 마일스톤"을 실제 목표로 채운다. 비어 있으면 채우고 시작.
# 1) 큐를 채운다 (작성 규칙과 예시는 plan.md §4)
node harness/hx.mjs add '{"id":"T-001", ...}'
node harness/hx.mjs list

# 2) 배선 점검 — claude를 부르지 않고 계획만 출력한다
WC_DRY_RUN=1 bash harness/outer-loop.sh

# 3) 방치 기동
nohup bash harness/outer-loop.sh > .harness/loop.out 2>&1 &
tail -f .harness/loop.out
```

워킹트리가 더러우면 **기동을 거부한다.** 루프는 `git add -A`로 커밋하므로,
기존 미커밋 변경이 섞이면 어떤 커밋이 어떤 태스크의 것인지 영영 알 수 없다.

### 멈추기

```bash
kill "$(cat .harness/loop.lock)"     # 현재 세션이 끝나는 대로 멈춘다
node harness/hx.mjs list             # 어디까지 갔는지
column -t -s $'\t' .harness/ledger.tsv   # 태스크별 실제 비용
```

### 설정

| 환경변수 | 기본 | 뜻 |
|---|---|---|
| `WC_BUDGET_USD` | `40.00` | 루프 전체 예산. 넘으면 멈춘다 |
| `WC_TASK_BUDGET_USD` | `2.00` | 세션 하나의 하드 캡 (`--max-budget-usd`) |
| `WC_MODEL` | `sonnet` | 기본 모델 (태스크의 `model`이 이긴다) |
| `WC_EFFORT` | `medium` | 기본 사고 예산 (태스크의 `effort`가 이긴다) |
| `WC_MAX_ATTEMPTS` | `3` | 연속 실패 허용. 넘으면 `blocked` |
| `WC_SESSION_TIMEOUT` | `1800` | 세션 벽시계 상한(초) |
| `WC_KEEP_ON_FAIL` | `0` | `1`이면 실패한 작업물을 트리에 남긴다 |
| `WC_ALLOW_DIRTY` | `0` | `1`이면 더러운 트리에서도 기동 |
| `WC_BARE` | `0` | `1`이면 `--bare` (§4 참조) |
| `WC_DRY_RUN` | `0` | `1`이면 계획만 출력 |

---

## 3. 한 이터레이션에서 일어나는 일

```
hx next          → 다음 todo (deps가 전부 done인 것만)
hx bump/set      → attempts += 1, status = doing
git rev-parse    → 되돌아갈 지점 기록
claude -p "/work <id>"   ← 짧은 세션 하나. 컨텍스트는 여기서 끝난다
verify.sh        ← 셸이 검증한다. 모델의 "완료했습니다"는 증거가 아니다
  통과 → git add -A && git commit   /  status = done  /  저널에 한 줄
  실패 → 저널에 실패 블록(세션 요약 + 검증 요지) / git reset --hard / status = todo
         시도 상한 도달 시 status = blocked, 다음 태스크로
```

실패 시 **되돌리는 것이 기본**이다. 깨진 절반이 트리에 남으면 다음 세션은
자기 태스크가 아니라 남이 만든 잔해를 디버깅한다. 학습은 저널이 나른다.
이어서 쌓아 올려야 하는 태스크라면 `WC_KEEP_ON_FAIL=1`.

### 세션이 남기는 것

`/work`는 마지막 출력을 고정 형식으로 강제한다:

```
RESULT: pass | fail
FILES:  …
WHAT:   …
LEARNED: 다음 시도가 알아야 할 사실
```

`LEARNED`가 실패 시 저널로 들어가 **다음 세션의 컨텍스트가 된다.**
세션 간 유일한 정보 통로다.

---

## 4. 비용 — 효과가 큰 순서대로

### ① 캐시 TTL에 루프 주기를 맞춘다

프롬프트 캐시 수명은 **구독 1시간 / API 키 5분**이다. 이터레이션 사이에 `sleep`을
넣지 않는 이유이고, 브라우저 게이트(13종, 수 분)를 매 태스크가 아니라 태스크의
`gate` 필드가 있을 때만 돌리는 이유다. 긴 검증이 캐시를 식히면 다음 세션은
컨텍스트를 처음부터 다시 처리한다.

같은 이유로 루프는 `--exclude-dynamic-system-prompt-sections`를 켠다.
cwd·git status 같은 매 세션 달라지는 조각이 시스템 프롬프트에 있으면
캐시 프리픽스가 세션마다 어긋난다. 이걸 켜면 그 조각들이 첫 사용자 메시지로 내려가고,
**세션이 바뀌어도 캐시 프리픽스가 동일해진다.** 매 이터레이션이 새 세션인 이 구조에서
가장 직접적인 절감이다.

**`WC_BARE=1`을 기본으로 켜지 않는 이유**: `--bare`는 훅·CLAUDE.md 자동적재를 끄고
인증을 `ANTHROPIC_API_KEY`로 강제한다(OAuth를 읽지 않는다). 즉 하네스의 **경계**와
**의도** 축이 같이 꺼지고, 구독으로 돌던 것이 API 키로 넘어가면서 캐시 TTL이
1시간 → 5분으로 떨어진다. 토큰 단가를 아끼려다 캐시와 가드레일을 잃는다.

### ② 모델 라우팅

태스크마다 `model` · `effort`를 선언한다. 기본은 `sonnet` / `medium`.
`opus`는 아키텍처 결정처럼 다단계 추론이 필요한 태스크에만.
실무 배분: **계획 1회 opus → 구현 루프 전량 sonnet**.

### ③ 훅으로 토큰이 컨텍스트에 들어오기 전에 자른다

`harness/hooks/guard-bash.mjs`(PreToolUse)가 `npm test` · `npx vitest` ·
`npm run gate` · `npm run build`를 **차단하고** `harness/verify.sh`로 보낸다.
`verify.sh`는 통과하면 줄당 `ok` 하나만 찍고, 실패하면 요지만 낸다.
전체 로그는 `.harness/logs/`에 남는다 — 필요하면 `grep`으로 좁혀 읽는다.

이 저장소에서 그냥 `npm run verify`는 테스트 1158개(47 파일) 출력이고
`npm run gate`는 playwright 13종이다. 그걸 통째로 읽으면 세션 예산의 상당 부분이
로그 낭독에 쓰인다.

### ④ 사고 예산 조절

루프가 `effort`에 따라 `MAX_THINKING_TOKENS`를 세팅한다:
`low → 4000`, `medium → 8000`, `high → 모델 기본값`.
사고 토큰은 출력 토큰으로 과금된다 — 구현 태스크에 수만 토큰짜리 기본 예산을 주면
루프 전체가 사고 비용으로 샌다.

### ⑤ compact 대신 clear

이 구조에서는 자동으로 해결된다. **매 이터레이션이 새 세션**이므로 압축할 대화가
애초에 없다. `/compact`는 요약할 대화를 읽는 것 자체가 큰 요청이다.

### ⑥ MCP를 끈다

루프는 `--strict-mcp-config`를 `--mcp-config` 없이 준다 → MCP 서버 0개.
쓰지 않을 도구 스키마를 매 세션 프롬프트에 싣지 않는다.

### ⑦ 에이전트 팀을 붙이지 않는다

에이전트 팀은 표준 세션의 수 배 토큰을 쓴다. 24시간 방치 루프에 팀을 붙이면
예산이 새벽에 증발한다. 이 루프는 세션당 서브에이전트 없이 단일 컨텍스트로 돈다.

---

## 5. 도구

### `harness/hx.mjs` — 상태 조작

Git Bash에 `jq`도 `bc`도 없어서 node로 만들었다. (그리고 `jq -i`는 **존재하지 않는
플래그**다 — 그걸로 상태를 갱신하면 아무 일도 없이 루프가 같은 태스크를 영원히 돈다.
상태 전이는 실패하면 시끄럽게 실패해야 한다.)

```
hx next                    다음 todo id (없으면 exit 3)
hx get <id>                태스크 전문 — /work 세션이 읽는 계약서
hx field <id> <key>        필드 하나 (셸용)
hx set <id> <key> <value>  상태 전이
hx bump <id>               attempts += 1
hx add <json>              태스크 추가
hx list [status]           큐 보기
hx journal <id> [N]        이 태스크의 과거 시도만
hx jget <key> [기본값]     stdin JSON에서 필드 하나
hx stat                    큐 요약
```

### `harness/verify.sh` — 검증

```bash
bash harness/verify.sh                       # typecheck → lint → test (fail-fast)
bash harness/verify.sh --gate=pick-and-place # + 브라우저 게이트 하나
bash harness/verify.sh --gate                # + 전체 13종 (느리다)
```

fail-fast인 이유: 타입이 깨진 채로 돌린 테스트 실패는 전부 파생 소음이다.
첫 실패에서 멈춰야 모델이 한 번에 한 가지를 고친다.

---

## 6. 태스크 스키마 (`TASKS.jsonl`, 1줄 1태스크)

| 필드 | 뜻 |
|---|---|
| `id` | 고유 id (`T-012`). 저널·커밋이 이걸로 엮인다 |
| `title` | 한 줄 요약. 기본 커밋 메시지에도 쓰인다 |
| `status` | `todo` · `doing` · `done` · `blocked` · `backlog` |
| `why` | 왜 하는가. 세션이 판단할 때 쓴다 |
| `done_when` | **기계로 확인 가능한** 완료 조건 배열. 이게 계약이다 |
| `model` | `sonnet`(기본) · `opus` · `haiku` |
| `effort` | `low` · `medium`(기본) · `high` |
| `gate` | 브라우저 게이트 이름 또는 `null` (예: `"pick-and-place"`) |
| `deps` | 선행 태스크 id 배열. 전부 `done`이어야 선택된다 |
| `attempts` | 루프가 관리 |
| `commit` | 커밋 메시지 (저장소 관례: Conventional, 한국어) |

`backlog`는 루프가 절대 고르지 않는다 — 아직 `done_when`이 없는 것들의 대기실이다.

---

## 7. 장애 대응

| 증상 | 원인 / 처치 |
|---|---|
| `이미 도는 루프가 있다` | `kill $(cat .harness/loop.lock)`. 죽은 락은 자동 회수된다 |
| `워킹트리가 더럽다` | 커밋하거나 stash. 정말 무시하려면 `WC_ALLOW_DIRTY=1` |
| 태스크가 계속 `blocked` | `hx journal <id>`를 읽는다. 대개 태스크가 세션 하나보다 크다 → 쪼갠다 |
| 비용이 예상보다 큼 | `.harness/ledger.tsv`를 본다. 특정 태스크가 재시도를 태우는지 확인 |
| 세션이 매번 예산 소진 | `done_when`이 모호하다. 기계로 확인 가능한 문장으로 바꾼다 |
| 훅이 안 걸린다 | 훅은 프로젝트 디렉터리를 cwd로 실행된다. 안 되면 `loop-settings.json`의 command를 절대경로로 바꾼다 |

`.harness/`는 gitignore 대상이다 — 로그·락·원장은 저장소에 들어가지 않는다.
저장소에 남는 상태는 `TASKS.jsonl` · `JOURNAL.md` · 커밋 세 가지뿐이다.

---

## 8. 이 CLI에서 확인한 것 (실측)

- `--max-turns`는 **이 버전에 없다.** 세션 상한은 `--max-budget-usd`가 유일하다.
- `--permission-mode`의 선택지: `acceptEdits` `auto` `bypassPermissions` `manual`
  `dontAsk` `plan`.
- Git Bash에 `jq` · `bc` 없음 → `hx.mjs` + `awk`로 대체.
- `timeout`이 있으면 세션 벽시계 상한이 걸리고, 없으면 경고만 찍고 계속한다.
