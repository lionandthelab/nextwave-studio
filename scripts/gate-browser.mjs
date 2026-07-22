// scripts/gate-browser.mjs — ROADMAP 검증 게이트의 브라우저 자동 검증 하네스
//
// 사용법:
//   node scripts/gate-browser.mjs                 # dist/ 프리뷰 서버로 검증 (vite build 선행 필요)
//   node scripts/gate-browser.mjs --expect=falling-boxes   # /?scene=falling-boxes 로드
//   node scripts/gate-browser.mjs --expect=arm             # /?scene=arm-and-boxes 로드 (Phase 3)
//
// 검증 항목(공통):
//   1. 콘솔에 'Rapier ready' 출력
//   2. 페이지 에러(uncaught) 0건
//   3. window.__sim 노출 (engine/world/sceneHandle/spec)
//   4. 시뮬 시간이 실제로 전진 (simTimeSec > 1)
// --expect 로 씬별 물리 어서션 추가 (아래 SCENE_BY_EXPECT가 ?scene= 파라미터로 매핑).

import { spawn, execSync } from 'node:child_process';
import { chromium } from 'playwright';

/** Windows에서 shell 경유 spawn은 kill()로 자식 트리가 안 죽는다 — taskkill로 정리 */
function killTree(proc) {
  if (process.platform === 'win32') {
    try { execSync(`taskkill /pid ${proc.pid} /T /F`, { stdio: 'ignore' }); } catch { /* already dead */ }
  } else {
    proc.kill();
  }
}

const PORT = 4173;
const expectArg = process.argv.find((a) => a.startsWith('--expect='))?.split('=')[1] ?? null;

// --expect 값 → ?scene= 파라미터 매핑 (미등록 값은 이름 그대로 씬으로 시도)
const SCENE_BY_EXPECT = {
  'falling-boxes': 'falling-boxes',
  arm: 'arm-and-boxes',
};

// --expect=arm 어서션 상수
const ARM_MIN_JOINT_COUNT = 8;          // 6 revolute + 2 prismatic finger
const ARM_HOME_JOINT2_RAD = -0.6;       // arm-and-boxes.scene.json home
const ARM_HOME_TOLERANCE_RAD = 1e-3;
const ARM_STANDING_MIN_Y_M = 0.35;      // 축 변환 증명: 링크가 이 높이 위에 있어야 "서 있음"
const ARM_LYING_MAX_Y_M = 0.2;          // 전 링크가 이 아래면 로봇이 누움 = 축 변환 실패
const ARM_JOINT1_TARGET_RAD = 1.2;      // 구동 검증용 목표값
const ARM_DRIVE_WAIT_MS = 300;          // 목표 적용 후 물리 반영 대기
const ARM_MIN_LINK_DISPLACEMENT_M = 0.02; // x/z 이동 판정 임계

function startPreview() {
  const proc = spawn('npx', ['vite', 'preview', '--port', String(PORT), '--strictPort'], {
    cwd: process.cwd(),
    shell: true,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('vite preview start timeout')), 20000);
    proc.stdout.on('data', (d) => {
      if (String(d).includes('http://localhost')) {
        clearTimeout(timer);
        resolve(proc);
      }
    });
    proc.stderr.on('data', (d) => process.stderr.write(d));
    proc.on('exit', (code) => reject(new Error(`vite preview exited early (${code})`)));
  });
}

async function main() {
  const results = [];
  const fail = (name, detail) => results.push({ name, ok: false, detail });
  const pass = (name, detail) => results.push({ name, ok: true, detail });

  const server = await startPreview();
  const browser = await chromium.launch();
  try {
    const page = await browser.newPage();
    const consoleLines = [];
    const pageErrors = [];
    page.on('console', (msg) => consoleLines.push(msg.text()));
    page.on('pageerror', (err) => pageErrors.push(String(err)));

    const sceneName = expectArg ? (SCENE_BY_EXPECT[expectArg] ?? expectArg) : null;
    const url = `http://localhost:${PORT}/${sceneName ? `?scene=${encodeURIComponent(sceneName)}` : ''}`;
    await page.goto(url, { waitUntil: 'load' });

    // 1. Rapier ready
    try {
      await page.waitForFunction(
        () => window.__sim !== undefined,
        undefined,
        { timeout: 15000 },
      );
      pass('window.__sim exposed');
    } catch {
      fail('window.__sim exposed', 'window.__sim not found within 15s');
    }

    if (consoleLines.some((l) => l.includes('Rapier ready'))) pass('console: Rapier ready');
    else fail('console: Rapier ready', `console was: ${consoleLines.slice(0, 10).join(' | ')}`);

    // 시뮬 2.5초 진행 대기
    await page.waitForTimeout(2500);

    const sim = await page.evaluate(() => {
      const s = window.__sim;
      if (!s) return null;
      const entities = {};
      for (const id of s.sceneHandle?.entityIds ?? []) {
        const bodies = s.world.bodiesOfEntity(id);
        if (bodies.length > 0) {
          entities[id] = s.world.getPose(bodies[0]);
        }
      }
      return {
        state: s.engine.state,
        simTimeSec: s.engine.simTimeSec,
        sceneName: s.spec?.name,
        entities,
      };
    });

    if (sim && sim.simTimeSec > 1) pass('sim time advances', `simTimeSec=${sim.simTimeSec.toFixed(2)}`);
    else fail('sim time advances', `sim=${JSON.stringify(sim)}`);

    // 씬별 어서션
    if (expectArg === 'falling-boxes' && sim) {
      const dyn = Object.entries(sim.entities).filter(([id]) => !id.startsWith('__') && !id.includes('wall') && id !== 'arm');
      const settled = dyn.filter(([, p]) => p.position[1] > 0 && p.position[1] < 1.0);
      if (dyn.length >= 3 && settled.length === dyn.length) {
        pass('falling-boxes: all dynamic bodies settled above ground', JSON.stringify(Object.fromEntries(dyn.map(([id, p]) => [id, p.position[1].toFixed(3)]))));
      } else {
        fail('falling-boxes: all dynamic bodies settled above ground', JSON.stringify(sim.entities));
      }
    }

    if (expectArg === 'arm') {
      // 1) 로봇 파사드 + 관절/홈/기립(축 변환) 스냅샷
      const arm = await page.evaluate(() => {
        const robots = window.__sim?.robots;
        if (!robots) return null;
        const ids = robots.ids();
        if (!ids.includes('arm')) return { ids, jointCount: 0, joints: {}, linkPoses: [] };
        return {
          ids,
          jointCount: robots.joints('arm').length,
          joints: robots.readJoints('arm'),
          linkPoses: robots.linkPoses('arm'),
        };
      });

      if (arm) pass('arm: window.__sim.robots exposed');
      else fail('arm: window.__sim.robots exposed', 'robots facade not found');

      if (arm?.ids.includes('arm')) pass('arm: robots.ids() contains "arm"', `ids=[${arm.ids.join(', ')}]`);
      else fail('arm: robots.ids() contains "arm"', `ids=${JSON.stringify(arm?.ids)}`);

      if (arm && arm.jointCount >= ARM_MIN_JOINT_COUNT) {
        pass(`arm: joint count >= ${ARM_MIN_JOINT_COUNT}`, `count=${arm.jointCount}`);
      } else {
        fail(`arm: joint count >= ${ARM_MIN_JOINT_COUNT}`, `count=${arm?.jointCount}`);
      }

      const joint2 = arm?.joints?.joint2;
      if (typeof joint2 === 'number' && Math.abs(joint2 - ARM_HOME_JOINT2_RAD) < ARM_HOME_TOLERANCE_RAD) {
        pass('arm: home pose applied (joint2 ≈ -0.6)', `joint2=${joint2}`);
      } else {
        fail('arm: home pose applied (joint2 ≈ -0.6)', `joints=${JSON.stringify(arm?.joints)}`);
      }

      // 축 변환 증명: 로봇이 "서" 있으면 일부 링크가 y > 0.35, "누워" 있으면 전 링크 y < 0.2
      const linkYs = (arm?.linkPoses ?? []).map((p) => p.position[1]);
      const standing = linkYs.some((y) => y > ARM_STANDING_MIN_Y_M);
      const lyingDown = linkYs.length > 0 && linkYs.every((y) => y < ARM_LYING_MAX_Y_M);
      if (standing && !lyingDown) {
        pass('arm: robot standing (some link y > 0.35 — Z-up→Y-up ok)', `linkYs=[${linkYs.map((y) => y.toFixed(3)).join(', ')}]`);
      } else {
        fail('arm: robot standing (some link y > 0.35 — Z-up→Y-up ok)', `linkYs=[${linkYs.map((y) => y.toFixed(3)).join(', ')}] (lyingDown=${lyingDown})`);
      }

      // 2) 관절 구동이 실제로 물리 바디를 움직이는지 (setJoint → 링크 pose 변위)
      const before = arm?.linkPoses ?? [];
      await page.evaluate((target) => {
        window.__sim?.robots.setJoint('arm', 'joint1', target);
      }, ARM_JOINT1_TARGET_RAD);
      await page.waitForTimeout(ARM_DRIVE_WAIT_MS);
      const after = await page.evaluate(() => window.__sim?.robots.linkPoses('arm') ?? []);

      const moved = before.some((p, i) => {
        const q = after[i];
        if (!q) return false;
        return (
          Math.abs(q.position[0] - p.position[0]) > ARM_MIN_LINK_DISPLACEMENT_M ||
          Math.abs(q.position[2] - p.position[2]) > ARM_MIN_LINK_DISPLACEMENT_M
        );
      });
      if (before.length > 0 && moved) {
        pass('arm: setJoint(joint1) moves physics link bodies (x/z > 0.02)');
      } else {
        fail(
          'arm: setJoint(joint1) moves physics link bodies (x/z > 0.02)',
          `before=${JSON.stringify(before.map((p) => p.position))} after=${JSON.stringify(after.map((p) => p.position))}`,
        );
      }
    }

    // 페이지 에러는 씬별 상호작용까지 끝난 뒤 마지막에 판정한다
    if (pageErrors.length === 0) pass('no page errors');
    else fail('no page errors', pageErrors.join(' | '));

    await page.screenshot({ path: 'gate-screenshot.png' });
  } finally {
    await browser.close();
    killTree(server);
  }

  const failed = results.filter((r) => !r.ok);
  for (const r of results) {
    console.log(`${r.ok ? 'PASS' : 'FAIL'}  ${r.name}${r.detail ? '  — ' + r.detail : ''}`);
  }
  console.log(failed.length === 0 ? '\nGATE: ALL PASS' : `\nGATE: ${failed.length} FAILED`);
  process.exit(failed.length === 0 ? 0 : 1);
}

main().catch((err) => {
  console.error('gate-browser harness error:', err);
  process.exit(2);
});
