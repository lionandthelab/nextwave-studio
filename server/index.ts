// server/index.ts — 실행 진입점: API + 정적 번들(dist/)을 한 프로세스·한 포트로 서빙
//
// 왜 단일 프로세스인가(docs/BACKEND.md §1·§2): 현장 PC에서 IT 담당자 없이 켜지는 것이
// 목표다. 프로세스 1개 · 포트 1개 · DB 파일 1개 — nginx도 리버스 프록시도 없다.
// dist/가 없으면(프런트 미빌드) 경고만 내고 API 전용으로 뜬다 — 서버가 안 뜨는 것보다
// 반쪽이라도 뜨는 쪽이 현장에서 진단하기 쉽다.

import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import fastifyStatic from '@fastify/static';
import { API_PREFIX } from '../src/schema/entities';
import { buildApp, sendError } from './app';
import { loadConfig } from './config';

const config = loadConfig();
const app = buildApp({ dbPath: config.dbPath, serverName: config.serverName });

// 정적 번들은 저장소의 dist/ (server/의 형제) — cwd가 아니라 이 파일 기준으로 찾는다
const distRoot = fileURLToPath(new URL('../dist', import.meta.url));

if (existsSync(join(distRoot, 'index.html'))) {
  await app.register(fastifyStatic, { root: distRoot });
  // SPA 폴백 — 라우팅되지 않은 경로는 index.html. 단 /api는 JSON 404 봉투가 우선한다
  // (API 오타가 HTML 200으로 위장하면 클라이언트 디버깅이 불가능해진다).
  app.setNotFoundHandler((req, reply) => {
    if ((req.url.split('?')[0] ?? '').startsWith(API_PREFIX)) {
      sendError(reply, 404, 'not-found', '요청한 API 경로가 없습니다');
      return;
    }
    void reply.type('text/html').sendFile('index.html');
  });
} else {
  console.warn(
    `[workcell-server] dist/가 없습니다 — 정적 서빙 없이 API만 시작합니다. ` +
      `프런트엔드까지 서빙하려면 먼저 \`npm run build\`를 실행하세요.`,
  );
}

await app.listen({ port: config.port, host: '0.0.0.0' });
console.log(
  `[workcell-server] ${config.serverName} 시작 — http://localhost:${config.port} ` +
    `(API ${API_PREFIX}, DB ${config.dbPath})`,
);
