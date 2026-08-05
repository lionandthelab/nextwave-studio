#!/usr/bin/env node
// harness/hx.mjs — 하네스 상태 파일(TASKS.jsonl / JOURNAL.md) 조작기.
//
// 왜 node인가: Git Bash에 jq·bc가 없다(실측). 그리고 `jq -i`는 애초에 존재하지
// 않는 플래그다 — 그걸로 상태를 갱신하면 아무 일도 일어나지 않은 채 루프가
// 같은 태스크를 영원히 돈다. 상태 전이는 실패하면 시끄럽게 실패해야 한다.
//
// 의존성 0. node >= 18.

import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const TASKS = resolve(ROOT, 'TASKS.jsonl');
const JOURNAL = resolve(ROOT, 'JOURNAL.md');

const EXIT_ERROR = 1;
const EXIT_EMPTY = 3; // "찾는 것이 없다" — 오류가 아니라 정상 종료 신호

function die(msg) {
  process.stderr.write(`hx: ${msg}\n`);
  process.exit(EXIT_ERROR);
}

function load() {
  if (!existsSync(TASKS)) return [];
  const out = [];
  const lines = readFileSync(TASKS, 'utf8').split(/\r?\n/);
  lines.forEach((raw, i) => {
    const line = raw.trim();
    if (!line || line.startsWith('//')) return;
    try {
      out.push(JSON.parse(line));
    } catch (e) {
      die(`TASKS.jsonl ${i + 1}행 파싱 실패: ${e.message}\n  ${line.slice(0, 120)}`);
    }
  });
  return out;
}

function save(tasks) {
  writeFileSync(TASKS, tasks.map((t) => JSON.stringify(t)).join('\n') + '\n', 'utf8');
}

function find(tasks, id) {
  const t = tasks.find((x) => x.id === id);
  if (!t) die(`태스크 없음: ${id}`);
  return t;
}

function readStdin() {
  try {
    return readFileSync(0, 'utf8');
  } catch {
    return '';
  }
}

// --- 서브커맨드 --------------------------------------------------------------

const CMDS = {
  // 다음에 처리할 태스크 id 하나. 없으면 EXIT_EMPTY.
  // 순서 = 파일 순서. deps가 전부 done인 todo만 고른다.
  next() {
    const tasks = load();
    const doneIds = new Set(tasks.filter((t) => t.status === 'done').map((t) => t.id));
    const t = tasks.find(
      (x) => x.status === 'todo' && (x.deps || []).every((d) => doneIds.has(d)),
    );
    if (!t) process.exit(EXIT_EMPTY);
    process.stdout.write(t.id + '\n');
  },

  // 태스크 한 줄을 보기 좋은 JSON으로. /work 세션이 읽는 계약서다.
  get(id) {
    if (!id) die('사용법: hx get <id>');
    process.stdout.write(JSON.stringify(find(load(), id), null, 2) + '\n');
  },

  // 필드 하나만. 셸에서 쓰기 위한 것 — 없으면 빈 줄.
  field(id, key) {
    if (!id || !key) die('사용법: hx field <id> <key>');
    const v = find(load(), id)[key];
    process.stdout.write((v === undefined || v === null ? '' : String(v)) + '\n');
  },

  // 상태 전이. 값은 JSON으로 먼저 해석하고, 실패하면 문자열로 둔다.
  set(id, key, ...rest) {
    if (!id || !key) die('사용법: hx set <id> <key> <value>');
    const raw = rest.join(' ');
    let value;
    try {
      value = JSON.parse(raw);
    } catch {
      value = raw;
    }
    const tasks = load();
    find(tasks, id)[key] = value;
    save(tasks);
  },

  bump(id) {
    if (!id) die('사용법: hx bump <id>');
    const tasks = load();
    const t = find(tasks, id);
    t.attempts = (t.attempts || 0) + 1;
    save(tasks);
    process.stdout.write(String(t.attempts) + '\n');
  },

  // 태스크 추가. 사람이 큐를 채우는 경로.
  //   node harness/hx.mjs add '{"id":"T-007","title":"...","done_when":["..."]}'
  add(...rest) {
    const raw = rest.join(' ').trim() || readStdin().trim();
    if (!raw) die('사용법: hx add \'{"id":"T-007","title":"..."}\'');
    let obj;
    try {
      obj = JSON.parse(raw);
    } catch (e) {
      die(`JSON 파싱 실패: ${e.message}`);
    }
    if (!obj.id) die('id는 필수다');
    const tasks = load();
    if (tasks.some((t) => t.id === obj.id)) die(`이미 있는 id: ${obj.id}`);
    tasks.push({
      status: 'todo',
      model: 'sonnet',
      effort: 'medium',
      gate: null,
      attempts: 0,
      deps: [],
      ...obj,
    });
    save(tasks);
    process.stdout.write(`추가: ${obj.id}\n`);
  },

  list(status) {
    const tasks = load().filter((t) => !status || t.status === status);
    if (!tasks.length) {
      process.stdout.write('(비어 있음)\n');
      return;
    }
    const w = Math.max(...tasks.map((t) => t.id.length));
    for (const t of tasks) {
      const att = t.attempts ? ` (시도 ${t.attempts})` : '';
      process.stdout.write(
        `${t.id.padEnd(w)}  ${String(t.status).padEnd(7)}  ${t.title || ''}${att}\n`,
      );
    }
  },

  // 이 태스크의 과거 시도만 뽑는다. 저널 전체를 컨텍스트에 넣지 않기 위한 것.
  journal(id, limitArg) {
    if (!id) die('사용법: hx journal <id> [최근 N건]');
    if (!existsSync(JOURNAL)) return;
    const limit = Number(limitArg) > 0 ? Number(limitArg) : 3;
    const blocks = readFileSync(JOURNAL, 'utf8')
      .split(/\n(?=## )/)
      .filter((b) => b.includes(id));
    if (!blocks.length) {
      process.stdout.write(`(${id}에 대한 과거 기록 없음 — 첫 시도다)\n`);
      return;
    }
    process.stdout.write(blocks.slice(-limit).join('\n').trim() + '\n');
  },

  // stdin의 JSON에서 필드 하나. `claude --output-format json` 결과 파싱용.
  // 숫자 필드가 없으면 0 — 루프의 누적 계산이 NaN으로 죽지 않게.
  jget(key, fallback = '0') {
    const raw = readStdin();
    try {
      const v = JSON.parse(raw)[key];
      process.stdout.write((v === undefined || v === null ? fallback : String(v)) + '\n');
    } catch {
      process.stdout.write(fallback + '\n');
    }
  },

  // 큐 상태 한 줄 요약 — 루프 하트비트용.
  stat() {
    const tasks = load();
    const by = {};
    for (const t of tasks) by[t.status] = (by[t.status] || 0) + 1;
    const parts = ['todo', 'doing', 'done', 'blocked', 'backlog']
      .filter((s) => by[s])
      .map((s) => `${s} ${by[s]}`);
    process.stdout.write((parts.join(' · ') || '비어 있음') + '\n');
  },

  help() {
    process.stdout.write(
      [
        'hx.mjs — 하네스 상태 조작',
        '',
        '  next                    다음 todo id (없으면 exit 3)',
        '  get <id>                태스크 전문(JSON)',
        '  field <id> <key>        필드 하나',
        '  set <id> <key> <value>  상태 전이',
        '  bump <id>               attempts += 1',
        '  add <json>              태스크 추가',
        '  list [status]           큐 보기',
        '  journal <id> [N]        이 태스크의 과거 시도만',
        '  jget <key> [기본값]     stdin JSON에서 필드 하나',
        '  stat                    큐 요약',
        '',
      ].join('\n'),
    );
  },
};

const [cmd, ...args] = process.argv.slice(2);
const fn = CMDS[cmd || 'help'];
if (!fn) die(`알 수 없는 명령: ${cmd}  (hx help)`);
fn(...args);
