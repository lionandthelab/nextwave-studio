// scripts/verify-container.mjs — 도커 컨테이너로 서빙되는 앱의 실브라우저 검증
//
// gate-browser.mjs는 자체 vite preview(dist/)를 띄우지만, 이 스크립트는 **이미 떠 있는
// 컨테이너 URL**을 직접 친다. 즉 nginx 설정·MIME·에셋 경로까지 포함한 "배포 형상"을
// 검증한다. Windows/Linux/macOS 어디서 컨테이너를 띄웠든 동일하게 쓸 수 있다.
//
// 사용:
//   docker compose up -d
//   node scripts/verify-container.mjs                 # 기본 http://localhost:8080
//   node scripts/verify-container.mjs http://localhost:9000
//
// 요구: 호스트에 Playwright chromium (npx playwright install chromium)
import { chromium } from 'playwright';

const BASE = process.argv[2] ?? 'http://localhost:8080';
const results = [];
const pass = (n, d) => results.push({ ok: true, n, d });
const fail = (n, d) => results.push({ ok: false, n, d });

const browser = await chromium.launch();
try {
  const page = await browser.newPage();
  const logs = [];
  const errors = [];
  page.on('console', (m) => logs.push(m.text()));
  page.on('pageerror', (e) => errors.push(String(e)));

  await page.goto(`${BASE}/?scene=arm-and-boxes`, { waitUntil: 'load' });

  try {
    await page.waitForFunction(() => window.__sim !== undefined, undefined, { timeout: 20000 });
    pass('window.__sim exposed');
  } catch {
    fail('window.__sim exposed', 'not found in 20s');
  }

  if (logs.some((l) => l.includes('Rapier ready'))) pass('Rapier WASM initialized');
  else fail('Rapier WASM initialized', logs.slice(0, 8).join(' | '));

  await page.waitForTimeout(2500);

  const sim = await page.evaluate(() => {
    const s = window.__sim;
    if (!s) return null;
    return {
      scene: s.spec?.name,
      simTime: s.engine.simTimeSec,
      entities: s.sceneHandle?.entityIds ?? [],
      robots: s.robots?.ids() ?? [],
      jointCount: s.robots?.joints('arm')?.length ?? 0,
    };
  });

  if (sim && sim.simTime > 1) pass('physics advances', `simTime=${sim.simTime.toFixed(2)}s scene=${sim.scene}`);
  else fail('physics advances', JSON.stringify(sim));

  if (sim && sim.robots.includes('arm') && sim.jointCount >= 8) {
    pass('URDF robot loaded from container', `joints=${sim.jointCount} entities=${sim.entities.length}`);
  } else {
    fail('URDF robot loaded from container', JSON.stringify(sim));
  }

  // 시퀀스 실행까지 — 컨테이너 번들이 실제 물리·충돌을 도는지
  await page.evaluate(() => window.__sim.orchestrator.play());
  const deadline = Date.now() + 20000;
  let done = false;
  while (Date.now() < deadline) {
    const st = await page.evaluate(() => ({
      status: window.__sim.player.status,
      hits: window.__sim.collision.recent(50).filter((e) => e.phase === 'start').length,
    }));
    if (st.status === 'done') { done = true; break; }
    await page.waitForTimeout(400);
  }
  const hits = await page.evaluate(() => window.__sim.collision.recent(50).filter((e) => e.phase === 'start').length);
  if (done && hits > 0) pass('sequence runs + collisions detected', `collisionStarts=${hits}`);
  else fail('sequence runs + collisions detected', `done=${done} hits=${hits}`);

  if (errors.length === 0) pass('no page errors');
  else fail('no page errors', errors.join(' | '));

  await page.screenshot({ path: 'docker-verify.png' });
} finally {
  await browser.close();
}

for (const r of results) console.log(`${r.ok ? 'PASS' : 'FAIL'}  ${r.n}${r.d ? '  — ' + r.d : ''}`);
const failed = results.filter((r) => !r.ok).length;
console.log(failed === 0 ? '\nCONTAINER VERIFY: ALL PASS' : `\nCONTAINER VERIFY: ${failed} FAILED`);
process.exit(failed === 0 ? 0 : 1);
