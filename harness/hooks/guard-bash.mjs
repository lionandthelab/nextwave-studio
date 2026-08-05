#!/usr/bin/env node
// harness/hooks/guard-bash.mjs — PreToolUse(Bash) 훅.
//
// 목적: **토큰이 컨텍스트에 들어오기 전에** 자른다.
// `npm run gate`는 playwright 게이트 13종이고 `npx vitest`는 테스트 1158개다.
// 그 출력을 모델이 통째로 읽으면 세션 예산의 절반이 로그 낭독에 쓰인다.
// 그래서 시끄러운 형태를 막고 조용한 등가물로 보낸다.
//
// 규약: exit 2 = 도구 호출 차단 + stderr가 모델에게 전달된다.
//       exit 0 = 통과. 훅이 깨져도 작업을 막지 않는다(파싱 실패 시 통과).

const NOISY = [
  {
    re: /(^|[;&|]\s*)(npm\s+(run\s+)?(test|verify)\b|npx?\s+vitest\b|\bvitest\s+run\b)/,
    say:
      '테스트를 직접 돌리지 않는다. `bash harness/verify.sh` 를 쓴다 — ' +
      'typecheck → lint → test 를 순서대로 돌리고 **실패한 것만** 출력한다. ' +
      '전체 로그는 .harness/logs/ 에 남으니 필요하면 grep 으로 좁혀 읽는다.',
  },
  {
    re: /(^|[;&|]\s*)npm\s+run\s+gate/,
    say:
      '브라우저 게이트는 13종이고 출력이 매우 크다. ' +
      '`bash harness/verify.sh --gate=<이름>` 으로 필요한 것 하나만 돌린다 ' +
      '(예: --gate=pick-and-place). 전체가 정말 필요하면 --gate.',
  },
  {
    re: /(^|[;&|]\s*)(npm\s+run\s+build\b|npx?\s+vite\s+build\b)/,
    say:
      '빌드 출력은 컨텍스트에 담을 가치가 없다. 타입 확인이 목적이면 ' +
      '`npm run typecheck`, 산출물 확인이 목적이면 `bash harness/verify.sh --gate`.',
  },
  {
    re: /(^|[;&|]\s*)(cat|head)\s+[^|]*\.harness\/logs\/[^\s|]*\.log(\s|$)/,
    say:
      '로그 전문을 읽지 않는다. `grep -aE "error|FAIL" <로그> | head -40` 또는 ' +
      '`tail -40 <로그>` 로 좁혀 읽는다.',
  },
];

let buf = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (d) => (buf += d));
process.stdin.on('end', () => {
  let cmd = '';
  try {
    cmd = (JSON.parse(buf).tool_input || {}).command || '';
  } catch {
    process.exit(0); // 훅이 이해 못 하는 입력이면 방해하지 않는다
  }

  for (const rule of NOISY) {
    if (rule.re.test(cmd)) {
      process.stderr.write(`[하네스 가드] ${rule.say}\n`);
      process.exit(2);
    }
  }
  process.exit(0);
});
