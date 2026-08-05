#!/usr/bin/env bash
# harness/outer-loop.sh — 루프는 모델이 아니라 셸이 돌린다.
#
#   nohup bash harness/outer-loop.sh > .harness/loop.out 2>&1 &
#   tail -f .harness/loop.out
#
# 구조: 태스크 하나 = 세션 하나. 세션은 끝나면 죽고 컨텍스트는 버려진다.
# 살아남는 것은 파일뿐이다 — 코드(git), 큐(TASKS.jsonl), 학습(JOURNAL.md).
# 컨텍스트가 죽어도 다음 세션이 그 파일들로부터 다시 시작할 수 있으면 하네스는 성립한다.
#
# 설정(전부 환경변수, 기본값은 보수적):
#   WC_BUDGET_USD=40.00        루프 전체 예산 상한. 넘으면 멈춘다.
#   WC_TASK_BUDGET_USD=2.00    세션 하나의 하드 캡(--max-budget-usd).
#   WC_MODEL=sonnet            기본 모델. 태스크의 model 필드가 이기다.
#   WC_EFFORT=medium           기본 사고 예산. 태스크의 effort 필드가 이긴다.
#   WC_MAX_ITER=200            안전 상한(무한루프 방지).
#   WC_MAX_ATTEMPTS=3          같은 태스크 연속 실패 허용 횟수. 넘으면 blocked.
#   WC_SESSION_TIMEOUT=1800    세션 하나의 벽시계 상한(초). timeout 있을 때만.
#   WC_KEEP_ON_FAIL=0          1이면 실패한 작업물을 트리에 남긴다(기본: 되돌린다).
#   WC_ALLOW_DIRTY=0           1이면 더러운 워킹트리에서도 기동한다(권장하지 않음).
#   WC_BARE=0                  1이면 --bare. §비용 주석 참조 — 기본은 끄는 게 맞다.
#   WC_DRY_RUN=0               1이면 claude를 부르지 않고 계획만 출력한다.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

BUDGET_TOTAL="${WC_BUDGET_USD:-40.00}"
TASK_BUDGET="${WC_TASK_BUDGET_USD:-2.00}"
DEFAULT_MODEL="${WC_MODEL:-sonnet}"
DEFAULT_EFFORT="${WC_EFFORT:-medium}"
MAX_ITER="${WC_MAX_ITER:-200}"
MAX_ATTEMPTS="${WC_MAX_ATTEMPTS:-3}"
SESSION_TIMEOUT="${WC_SESSION_TIMEOUT:-1800}"
KEEP_ON_FAIL="${WC_KEEP_ON_FAIL:-0}"
ALLOW_DIRTY="${WC_ALLOW_DIRTY:-0}"
BARE="${WC_BARE:-0}"
DRY_RUN="${WC_DRY_RUN:-0}"

STATE=".harness"
LOGS="$STATE/logs"
LOCK="$STATE/loop.lock"
LEDGER="$STATE/ledger.tsv"
HX="node harness/hx.mjs"

mkdir -p "$LOGS"

# --- 유틸 --------------------------------------------------------------------

ts()  { date '+%Y-%m-%d %H:%M:%S'; }
say() { printf '[%s] %s\n' "$(ts)" "$*"; }
die() { printf '[%s] 중단: %s\n' "$(ts)" "$*" >&2; exit 1; }

# bc가 없다(Git Bash). awk로 부동소수 비교/누산.
fadd() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%.4f", a+b}'; }
flt()  { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a<b)}'; }

cleanup() { rm -f "$LOCK"; say "루프 종료. 누적 \$$SPENT / \$$BUDGET_TOTAL"; }

# JOURNAL.md는 루프만 쓴다(단일 필자). 세션은 읽기만 한다 —
# 두 필자가 append하면 순서가 섞이고, 섞인 저널은 다음 시도를 잘못 이끈다.
journal_fail() {
  local task="$1" attempt="$2" cost="$3" resfile="$4" verfile="$5"
  {
    echo
    echo "## $(ts) · $task · 시도 $attempt · 실패 (\$$cost)"
    echo
    echo "**세션이 남긴 것**"
    echo '```'
    tail -n 25 "$resfile" 2>/dev/null | sed 's/[[:space:]]*$//'
    echo '```'
    echo
    echo "**검증이 막은 것**"
    echo '```'
    tail -n 25 "$verfile" 2>/dev/null | sed 's/[[:space:]]*$//'
    echo '```'
    echo
    if [ "$KEEP_ON_FAIL" = "1" ]; then
      echo "> 작업물은 트리에 남겼다(WC_KEEP_ON_FAIL=1)."
    else
      echo "> 작업물은 되돌렸다. 다음 시도는 깨끗한 트리에서 시작한다 —"
      echo "> **위 접근을 반복하지 마라.**"
    fi
  } >> JOURNAL.md
}

journal_ok() {
  printf '\n## %s · %s · 완료 (\$%s, 시도 %s) — `%s`\n' \
    "$(ts)" "$1" "$3" "$2" "$4" >> JOURNAL.md
}

# --- 사전 점검 ---------------------------------------------------------------

command -v node   >/dev/null || die "node가 없다"
command -v git    >/dev/null || die "git이 없다"
command -v claude >/dev/null || die "claude CLI가 PATH에 없다"

git rev-parse --git-dir >/dev/null 2>&1 || die "git 저장소가 아니다"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
[ "$BRANCH" = "HEAD" ] && die "detached HEAD다. 브랜치를 먼저 만든다"

# 루프는 `git add -A`로 커밋한다. 기존의 더러운 변경이 섞여 들어가면
# 어떤 커밋이 어떤 태스크의 것인지 영영 알 수 없게 된다.
STARTED_CLEAN=1
if [ -n "$(git status --porcelain)" ]; then
  STARTED_CLEAN=0
  [ "$ALLOW_DIRTY" != "1" ] &&
    die "워킹트리가 더럽다. 커밋하거나 stash한 뒤 기동한다 (무시하려면 WC_ALLOW_DIRTY=1)"
  say "경고: 더러운 트리에서 기동한다. 실패 되돌리기에서 git clean을 쓰지 않는다 —"
  say "       네 미추적 파일을 지우지 않기 위해서다. 실패한 태스크가 만든 새 파일은 남는다."
fi

# 단일 인스턴스. 두 루프가 같은 큐를 물면 상태 전이가 서로를 덮어쓴다.
if [ -f "$LOCK" ]; then
  OLD="$(cat "$LOCK" 2>/dev/null)"
  if kill -0 "$OLD" 2>/dev/null; then
    die "이미 도는 루프가 있다 (pid $OLD). 멈추려면: kill $OLD"
  fi
  say "죽은 락 발견(pid $OLD) — 회수한다"
fi
echo $$ > "$LOCK"
trap cleanup EXIT INT TERM

TIMEOUT_BIN=""
command -v timeout >/dev/null && TIMEOUT_BIN="timeout"
[ -z "$TIMEOUT_BIN" ] && say "경고: timeout이 없다. 세션 벽시계 상한이 걸리지 않는다"

[ -f "$LEDGER" ] || printf 'when\ttask\tattempt\tmodel\tcost_usd\tverdict\n' > "$LEDGER"

SPENT=0
ITER=0

say "──────────────────────────────────────────────"
say "브랜치 $BRANCH · 모델 $DEFAULT_MODEL · 예산 \$$BUDGET_TOTAL (세션당 \$$TASK_BUDGET)"
say "큐: $($HX stat)"
say "──────────────────────────────────────────────"

# --- 루프 --------------------------------------------------------------------

while flt "$SPENT" "$BUDGET_TOTAL"; do
  ITER=$((ITER + 1))
  if [ "$ITER" -gt "$MAX_ITER" ]; then
    say "반복 상한 $MAX_ITER 도달"; break
  fi

  TASK="$($HX next)"
  if [ -z "$TASK" ]; then
    say "todo 없음 — 큐를 비웠다"; break
  fi

  TITLE="$($HX field "$TASK" title)"
  MODEL="$($HX field "$TASK" model)"; MODEL="${MODEL:-$DEFAULT_MODEL}"
  EFFORT="$($HX field "$TASK" effort)"; EFFORT="${EFFORT:-$DEFAULT_EFFORT}"
  GATE="$($HX field "$TASK" gate)"
  COMMITMSG="$($HX field "$TASK" commit)"

  # 드라이런은 상태를 건드리지 않는다. (건드리면 status를 todo로 되돌리는 순간
  # next가 같은 태스크를 다시 집어 MAX_ITER까지 헛돈다.) 배선만 보여주고 끝낸다.
  if [ "$DRY_RUN" = "1" ]; then
    say "▶ (드라이런) $TASK · $MODEL/$EFFORT · gate=${GATE:-없음} · $TITLE"
    say "  claude -p \"/work $TASK\" --permission-mode dontAsk \\"
    say "    --settings harness/loop-settings.json --model $MODEL --effort $EFFORT \\"
    say "    --max-budget-usd $TASK_BUDGET --output-format json \\"
    say "    --exclude-dynamic-system-prompt-sections --strict-mcp-config"
    say "  검증: bash harness/verify.sh ${GATE:+--gate=$GATE}"
    say "  상태 전이 없음. 남은 큐: $($HX stat)"
    break
  fi

  ATTEMPT="$($HX bump "$TASK")"
  $HX set "$TASK" status doing

  STAMP="$(date '+%Y%m%d-%H%M%S')"
  RESFILE="$LOGS/$STAMP-$TASK-a$ATTEMPT.txt"
  VERFILE="$LOGS/$STAMP-$TASK-a$ATTEMPT.verify.txt"
  BASE="$(git rev-parse HEAD)"

  say "▶ $TASK (시도 $ATTEMPT/$MAX_ATTEMPTS · $MODEL/$EFFORT) $TITLE"

  # 사고 예산은 출력 토큰으로 과금된다. 구현 태스크에 수만 토큰짜리 기본
  # 예산을 주면 루프 전체가 사고 비용으로 샌다. effort=high는 계획성 태스크
  # 전용이므로 그때만 모델 기본값에 맡긴다.
  case "$EFFORT" in
    low)    export MAX_THINKING_TOKENS=4000 ;;
    medium) export MAX_THINKING_TOKENS=8000 ;;
    *)      unset MAX_THINKING_TOKENS ;;
  esac

  ARGS=(
    -p "/work $TASK"
    --permission-mode dontAsk
    --settings "harness/loop-settings.json"
    --model "$MODEL"
    --effort "$EFFORT"
    --max-budget-usd "$TASK_BUDGET"
    --output-format json
    # 매 이터레이션이 새 세션이므로, cwd·git status 같은 가변 구간을
    # 시스템 프롬프트에서 빼야 캐시 프리픽스가 세션 간에 동일해진다.
    --exclude-dynamic-system-prompt-sections
    # MCP 서버 도구 스키마는 이 작업에 쓰이지 않는다 — 프롬프트에서 뺀다.
    --strict-mcp-config
  )
  # --bare는 훅과 CLAUDE.md 자동적재를 끄고 인증을 API 키로 강제한다.
  # 즉 하네스의 "경계"와 "의도" 축이 같이 꺼진다. 구독으로 돌면 캐시 TTL도
  # 1시간 → 5분으로 떨어진다. 기본이 0인 이유다.
  [ "$BARE" = "1" ] && ARGS+=( --bare --add-dir "$ROOT" )

  RAW="$LOGS/$STAMP-$TASK-a$ATTEMPT.json"
  if [ -n "$TIMEOUT_BIN" ]; then
    $TIMEOUT_BIN "$SESSION_TIMEOUT" claude "${ARGS[@]}" > "$RAW" 2>"$RESFILE.err"
  else
    claude "${ARGS[@]}" > "$RAW" 2>"$RESFILE.err"
  fi
  CLAUDE_RC=$?

  COST="$($HX jget total_cost_usd 0 < "$RAW")"
  SPENT="$(fadd "$SPENT" "$COST")"
  $HX jget result "(출력 없음)" < "$RAW" > "$RESFILE"
  [ -s "$RESFILE.err" ] && cat "$RESFILE.err" >> "$RESFILE"

  if [ "$CLAUDE_RC" -ne 0 ]; then
    say "  세션 비정상 종료 (rc=$CLAUDE_RC) — 예산 소진이거나 타임아웃"
  fi

  # 검증은 셸이 한다. 모델의 "다 됐습니다"는 증거가 아니다.
  VERIFY_ARGS=()
  [ -n "$GATE" ] && VERIFY_ARGS=( "--gate=$GATE" )
  if bash harness/verify.sh "${VERIFY_ARGS[@]}" > "$VERFILE" 2>&1; then
    if [ -z "$(git status --porcelain)" ]; then
      say "  통과했지만 변경이 없다 — 이미 만족된 태스크로 본다 (\$$COST)"
      $HX set "$TASK" status done
      journal_ok "$TASK" "$ATTEMPT" "$COST" "변경 없음"
      printf '%s\t%s\t%s\t%s\t%s\tnoop\n' "$(ts)" "$TASK" "$ATTEMPT" "$MODEL" "$COST" >> "$LEDGER"
    else
      MSG="${COMMITMSG:-chore($TASK): ${TITLE:-태스크 완료}}"
      git add -A
      git commit -q -m "$MSG" -m "task: $TASK (시도 $ATTEMPT)

Co-Authored-By: Claude ($MODEL) <noreply@anthropic.com>"
      $HX set "$TASK" status done
      journal_ok "$TASK" "$ATTEMPT" "$COST" "$MSG"
      say "  ✔ 통과 · 커밋 $(git rev-parse --short HEAD) (\$$COST · 누적 \$$SPENT)"
      printf '%s\t%s\t%s\t%s\t%s\tpass\n' "$(ts)" "$TASK" "$ATTEMPT" "$MODEL" "$COST" >> "$LEDGER"
    fi
  else
    journal_fail "$TASK" "$ATTEMPT" "$COST" "$RESFILE" "$VERFILE"
    printf '%s\t%s\t%s\t%s\t%s\tfail\n' "$(ts)" "$TASK" "$ATTEMPT" "$MODEL" "$COST" >> "$LEDGER"

    if [ "$KEEP_ON_FAIL" != "1" ]; then
      git reset -q --hard "$BASE"
      # 미추적 파일 정리는 **깨끗한 트리에서 기동했을 때만** 한다.
      # 더러운 트리에서 기동했다면 저 미추적 파일들은 사용자 것이다 — 지우면 안 된다.
      # (-x 없음: .harness/ 등 gitignore 대상은 어느 쪽이든 건드리지 않는다.)
      [ "$STARTED_CLEAN" = "1" ] && git clean -qfd
    fi

    if [ "$ATTEMPT" -ge "$MAX_ATTEMPTS" ]; then
      $HX set "$TASK" status blocked
      say "  ✘ 실패 · 시도 $ATTEMPT회 — blocked로 내리고 다음으로 (\$$COST · 누적 \$$SPENT)"
    else
      $HX set "$TASK" status todo
      say "  ✘ 실패 · 저널에 남기고 재시도 예정 (\$$COST · 누적 \$$SPENT)"
    fi
  fi

  # 이터레이션 사이에 sleep을 넣지 않는다 — 프롬프트 캐시가 식으면
  # 다음 세션이 전체 컨텍스트를 처음부터 다시 처리한다.
done

say "큐: $($HX stat)"
