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
//   --expect=arm-sequence : Phase 4+5 통합 — 시퀀스 무자동재생(human-in-the-loop) 확인 후
//   파사드 play()로 재생 시작 → arm×box_a 충돌 start 이력 + waitForCollision 통과 +
//   그리퍼 닫힘 + status done + 충돌 로그 DOM 행 검증.

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
  'arm-sequence': 'arm-and-boxes',
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

// --expect=arm-sequence 어서션 상수 (arm-touch-box.sequence.json 기준)
const SEQ_WAIT_FOR_COLLISION_INDEX = 3;   // [gripper, moveJoints, setJoints, ★waitForCollision, gripper, wait, moveJoints]
const SEQ_STEP_COUNT = 7;
const SEQ_SIM_TIME_BUDGET_SEC = 12;       // Play 이후 이 sim 시간 안에 done이어야 함
const SEQ_EVENT_DONE_MAX_SIM_SEC = 9;     // Play→done sim 경과: 이벤트 해제 경로 ≈6s, timeout 경로 ≈11.9s — 9s 미만이어야 "실제 충돌" 해제
const SEQ_POLL_INTERVAL_MS = 100;
const SEQ_REALTIME_DEADLINE_MS = 30000;   // 폴링 실시간 상한 (행 방지)
const SEQ_GRIPPER_OPEN_MIN_M = 0.025;     // "열림 관측" 판정 (open=0.03)
const SEQ_GRIPPER_CLOSED_TOL_M = 2e-3;    // "닫힘(≈0) 관측" 판정 (close=0.0)

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

    if (expectArg === 'arm-sequence') {
      // 1) player 파사드 노출 + 무자동재생 (human-in-the-loop — CLAUDE.md §2.9의 원칙)
      const initial = await page.evaluate(() => {
        const p = window.__sim?.player;
        if (!p) return null;
        return { status: p.status, stepCount: p.stepCount, index: p.currentStepIndex };
      });
      if (initial) pass('arm-sequence: __sim.player exposed', JSON.stringify(initial));
      else fail('arm-sequence: __sim.player exposed', 'player facade not found');

      if (initial?.status === 'idle') {
        pass('arm-sequence: no autoplay (status idle before Play)');
      } else {
        fail('arm-sequence: no autoplay (status idle before Play)', `status=${initial?.status}`);
      }
      if (initial?.stepCount === SEQ_STEP_COUNT) {
        pass(`arm-sequence: sequence validated & loaded (${SEQ_STEP_COUNT} steps)`);
      } else {
        fail(`arm-sequence: sequence validated & loaded (${SEQ_STEP_COUNT} steps)`, `stepCount=${initial?.stepCount}`);
      }

      // 파사드가 없으면(시퀀스 검증 회귀 등) 이후 상호작용 evaluate가 TypeError로
      // 거부되어 하네스가 exit 2로 죽고 PASS/FAIL 표를 잃는다 — FAIL로 기록하고
      // 남은 arm-sequence 어서션을 건너뛴다 (종료 코드는 어차피 비-0).
      if (!initial) {
        fail('arm-sequence: interaction checks skipped', 'player facade missing — Play/barrier/gripper/done assertions cannot run');
      } else {
      // 2) ▶ Play — 파사드로 사람 승인 재생 시작 (Play 시점 simTime을 경과 기준으로 기록)
      const started = await page.evaluate(() => {
        window.__sim.player.play();
        return { status: window.__sim.player.status, simTimeSec: window.__sim.engine.simTimeSec };
      });
      const playSimTimeSec = started.simTimeSec;
      if (started.status === 'running') pass('arm-sequence: Play starts sequence (status running)');
      else fail('arm-sequence: Play starts sequence (status running)', `status=${started.status}`);

      // 3) sim 시간 예산 안에서 done까지 폴링 — 그리퍼 열림→닫힘 궤적 관측
      let sawGripperOpen = false;
      let minFingerAfterOpenM = Infinity;
      let sawPastWaitIndex = false;
      let last = null;
      const realDeadline = Date.now() + SEQ_REALTIME_DEADLINE_MS;
      for (;;) {
        last = await page.evaluate(() => {
          const s = window.__sim;
          const joints = s.robots.readJoints('arm');
          return {
            status: s.player.status,
            index: s.player.currentStepIndex,
            simTimeSec: s.engine.simTimeSec,
            fingerMaxM: Math.max(joints.finger_left_joint ?? 0, joints.finger_right_joint ?? 0),
          };
        });
        if (last.fingerMaxM >= SEQ_GRIPPER_OPEN_MIN_M) sawGripperOpen = true;
        if (sawGripperOpen) minFingerAfterOpenM = Math.min(minFingerAfterOpenM, last.fingerMaxM);
        if (last.index > SEQ_WAIT_FOR_COLLISION_INDEX) sawPastWaitIndex = true;
        if (last.status === 'done') break;
        if (last.simTimeSec - playSimTimeSec > SEQ_SIM_TIME_BUDGET_SEC || Date.now() > realDeadline) break;
        await page.waitForTimeout(SEQ_POLL_INTERVAL_MS);
      }

      // 4) 충돌 이력: arm × box_a start 이벤트가 기록되었다 (EventQueue 유래 — CLAUDE.md §2.4)
      const armBoxStarts = await page.evaluate(() => {
        const events = window.__sim.collision.recent(200);
        return events
          .filter((e) => e.phase === 'start'
            && ((e.a === 'arm' && e.b === 'box_a') || (e.a === 'box_a' && e.b === 'arm')))
          .map((e) => e.timeSec);
      });
      if (armBoxStarts.length >= 1) {
        pass('arm-sequence: collision history has arm×box_a start', `timeSec=[${armBoxStarts.map((t) => t.toFixed(3)).join(', ')}]`);
      } else {
        fail('arm-sequence: collision history has arm×box_a start', `history=${JSON.stringify(await page.evaluate(() => window.__sim.collision.recent(50)))}`);
      }

      // 5) waitForCollision 통과 (배리어 해제 — 커서가 인덱스를 지나감)
      if (sawPastWaitIndex) pass('arm-sequence: player advanced past waitForCollision index');
      else fail('arm-sequence: player advanced past waitForCollision index', `last=${JSON.stringify(last)}`);

      // 6) 그리퍼: 열림(≥0.025) 후 닫힘(≈0) 관측 — close step이 실제 구동됨
      if (sawGripperOpen && minFingerAfterOpenM <= SEQ_GRIPPER_CLOSED_TOL_M) {
        pass('arm-sequence: gripper opened then closed (finger ≈ 0)', `minAfterOpen=${minFingerAfterOpenM.toExponential(2)}`);
      } else {
        fail('arm-sequence: gripper opened then closed (finger ≈ 0)', `sawOpen=${sawGripperOpen} minAfterOpen=${minFingerAfterOpenM}`);
      }

      // 7) 최종 done — timeout(6s) 경로가 아니라 실제 충돌 해제 경로의 시간 안에서
      const elapsedSinceplaySec = (last?.simTimeSec ?? Infinity) - playSimTimeSec;
      if (last?.status === 'done') {
        pass('arm-sequence: sequence finished (status done)', `simTime=${last.simTimeSec.toFixed(2)}s (play 후 ${elapsedSinceplaySec.toFixed(2)}s)`);
      } else {
        fail('arm-sequence: sequence finished (status done)', `last=${JSON.stringify(last)}`);
      }
      if (last?.status === 'done' && elapsedSinceplaySec < SEQ_EVENT_DONE_MAX_SIM_SEC) {
        pass(`arm-sequence: done within ${SEQ_EVENT_DONE_MAX_SIM_SEC}s of Play (barrier released by event, not timeout)`);
      } else {
        fail(`arm-sequence: done within ${SEQ_EVENT_DONE_MAX_SIM_SEC}s of Play (barrier released by event, not timeout)`, `elapsed=${elapsedSinceplaySec}`);
      }

      // 8) 충돌 로그 DOM: box_a가 포함된 행 ≥ 1 (UI 배선 — UX_DESIGN §3.6)
      const boxARowCount = await page.$$eval('[data-testid="collision-row"]',
        (rows) => rows.filter((r) => (r.textContent ?? '').includes('box_a')).length);
      if (boxARowCount >= 1) pass('arm-sequence: collision-log DOM has box_a row(s)', `rows=${boxARowCount}`);
      else fail('arm-sequence: collision-log DOM has box_a row(s)', `rows=${boxARowCount}`);
      } // end if (initial) — 파사드 부재 시 상호작용 어서션 건너뜀
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
