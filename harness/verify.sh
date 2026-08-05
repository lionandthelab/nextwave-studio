#!/usr/bin/env bash
# harness/verify.sh — 하네스 4축 중 **검증**.
#
# 계약: 통과하면 조용하다(줄당 ok 하나). 실패하면 실패한 것만 짧게 말하고 exit 1.
# 전체 로그는 .harness/logs/ 에 남기고 컨텍스트에는 요지만 흘린다 — 1만 줄 로그가
# 모델 컨텍스트로 들어가면 그 세션의 예산은 거기서 끝난다.
#
# 사용법:
#   bash harness/verify.sh                    typecheck → lint → test (fail-fast)
#   bash harness/verify.sh --gate             + 브라우저 게이트 전체 (느리다)
#   bash harness/verify.sh --gate=pick-and-place   + 게이트 하나
#
# 환경변수:
#   WC_DIGEST_LINES  실패 요지 줄 수 (기본 40)

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

LOGDIR=".harness/logs"
mkdir -p "$LOGDIR"

GATE=""
DIGEST_LINES="${WC_DIGEST_LINES:-40}"

for a in "$@"; do
  case "$a" in
    --gate)     GATE="gate" ;;
    --gate=*)   GATE="gate:${a#--gate=}" ;;
    -q|--quiet) DIGEST_LINES=15 ;;
    -h|--help)  sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *)          echo "verify: 알 수 없는 인자 '$a'" >&2; exit 2 ;;
  esac
done

START=$SECONDS

# step <이름> <명령...>
# 성공: "ok <이름> (Ns)" 한 줄. 실패: 요지 + 로그 경로, 그리고 exit 1.
step() {
  local name="$1"; shift
  local log="$LOGDIR/verify-$name.log"
  local t0=$SECONDS

  if "$@" >"$log" 2>&1; then
    printf 'ok   %-10s %ss\n' "$name" "$((SECONDS - t0))"
    return 0
  fi

  printf 'FAIL %-10s %ss   전체 로그: %s\n' "$name" "$((SECONDS - t0))" "$log"
  echo
  echo "--- 실패 요지 ---"
  grep -aE 'error|Error|ERROR|FAIL|✕|✗|✘|Expected|Received|Cannot find|is not assignable' "$log" \
    | head -n "$DIGEST_LINES"
  echo "--- 로그 마지막 ${DIGEST_LINES}줄 ---"
  tail -n "$DIGEST_LINES" "$log"
  echo
  echo "(요지가 부족하면 grep 으로 좁혀 읽는다. 로그 전문을 통째로 읽지 않는다.)"
  return 1
}

# fail-fast: 타입이 깨진 채로 돌린 테스트 실패는 전부 파생 소음이다.
step typecheck npm run typecheck || exit 1
step lint      npm run lint      || exit 1
step test      npx vitest run --reporter=dot || exit 1

if [ -n "$GATE" ]; then
  step "${GATE//:/-}" npm run "$GATE" || exit 1
fi

echo "검증 게이트 통과 ($((SECONDS - START))s)"
