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
//   --expect=scene-switch : Phase 6 런타임 씬 전환 스모크 — arm-and-boxes 부트 →
//   UI select(change 이벤트)로 collision-testbed 전환 → spec.name/엔티티 수/sim 전진
//   검증 → arm-and-boxes로 복귀(URDF 재로드) → 동일 검증 + 페이지 에러 0건.
//   --expect=scene-builder : Phase 7 Scene Builder — __sim.editor 파사드로 템플릿 추가
//   (라이브러리 배치 경로) → 바디 존재/인스펙터 목록 → updateTransform teleport →
//   updateDimensions 후 물리 정착 y 관측 → removeEntity 정리 → __sim.history.undo로
//   엔티티 복원(전체 재로드) → 라이브러리 카드 DOM ≥ 6 + 페이지 에러 0건.
//   --expect=flow-graph : Phase 8 Flow Graph — 페인 표시 + DOM 노드 수 → insertWait
//   삽입(직렬화 유효) → JSON 뷰어 동기 → reorder 순서 갱신 → (재로드 후)
//   waitForCollision 비활성 → 배리어 없이 완주(timeout 경고 없음, 스킵 노드 무active)
//   → Stop 런 상태 리셋 + 재-Play 완주 → 노드 삭제 → 로봇 rename 플로우 재동기
//   → 페이지 에러 0건 (불변식 §2.8).
//   --expect=planner : Phase 9 NL Planner — __sim.planner 파사드(규칙 기반 백엔드).
//   (a) generate('box_a를 집어') → sequence + 그래프 로드(AI 배지) + 무자동재생(§2.9)
//   + 라이브 시퀀스 유효, (b) ▶ Play → arm×box_a 충돌 start(reach가 실제 접촉) + done,
//   (c) generate('박스를 집어') → clarify(2+ 옵션) → box_b 선택 → box_b 시퀀스,
//   (d) 없는 대상 → clarify/error, 무의미 입력 → 읽을 수 있는 error → 페이지 에러 0건.

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
  'pick-and-place': 'pick-and-place',
  'obstacle-avoidance': 'obstacle-avoidance',
  'collision-testbed': 'collision-testbed',
  'scene-switch': 'arm-and-boxes', // 런타임 전환 스모크 — arm-and-boxes에서 출발
  'scene-builder': 'arm-and-boxes', // Phase 7 씬 편집 — 로봇+박스 씬 위에서 편집 검증
  'flow-graph': 'arm-and-boxes', // Phase 8 Flow Graph — 시퀀스 있는 씬 위에서 그래프 편집 검증
  planner: 'arm-and-boxes', // Phase 9 NL Planner — 규칙 기반(오프라인) 백엔드로 결정론 검증
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

// ── Phase 6 샘플 씬 게이트 상수 ─────────────────────────────────────
// --expect=pick-and-place (pick-and-place.sequence.json 기준, step 9개)
//   명목 길이 ≈ 0.4+2.0+0.8+0+배리어+0.5+2.0+0.4+2.0 ≈ 8.1s(이벤트 해제) / ≈14.1s(timeout 경로)
const PNP_STEP_COUNT = 9;
const PNP_SIM_TIME_BUDGET_SEC = 16;
const PNP_EVENT_DONE_MAX_SIM_SEC = 10;
// --expect=obstacle-avoidance (obstacle-avoidance.sequence.json 기준, step 8개)
//   명목 길이 ≈ 0.3+2.0+1.5+2.0+1.5+0+배리어+2.0 ≈ 9.3s(이벤트 해제) / ≈15.3s(timeout 경로)
const OA_STEP_COUNT = 8;
const OA_SIM_TIME_BUDGET_SEC = 18;
const OA_EVENT_DONE_MAX_SIM_SEC = 12;
// --expect=collision-testbed (시퀀스 없음 — 씬 자체 물리 쇼케이스)
const TESTBED_EVENT_WINDOW_SEC = 5;    // 이 sim 시간 안에 접촉/센서 이벤트가 나야 함
const TESTBED_MIN_CONTACT_PAIRS = 3;   // 서로 다른 contact start 쌍 최소 개수
const TESTBED_MIN_SENSOR_EVENTS = 1;   // sensor start 최소 개수
const TESTBED_SETTLE_SIM_SEC = 8;      // 이 시점에 전 동적 바디 y > 0 (정착) 판정
// 공용: 충돌 이력 조회 상한(모니터 이력 상한 1000과 일치) · 폴링 실시간 상한
const HISTORY_FETCH_LIMIT = 1000;
const PHASE6_REALTIME_DEADLINE_MS = 45000;
// --expect=scene-switch (런타임 씬 전환 스모크 — UI select 경유 왕복 전환)
const SWITCH_TARGET_SCENE = 'collision-testbed';   // 로봇 없는 씬으로 전환 (이질적 조합)
const SWITCH_BACK_SCENE = 'arm-and-boxes';         // 로봇 씬으로 복귀 (URDF 재로드 경로)
const SWITCH_REALTIME_DEADLINE_MS = 20000;         // 전환(URDF 로드 포함) 실시간 상한
const SWITCH_MIN_SIM_ADVANCE_SEC = 0.5;            // 전환 직후 새 엔진이 이만큼 전진해야 함
// --expect=scene-builder (Phase 7 Scene Builder — __sim.editor/__sim.history 파사드)
const SB_ADD_POSITION = [0.2, 0.05, 0.2];          // box 템플릿 추가 위치 (기존 엔티티와 이격)
const SB_MOVE_POSITION = [0.5, 0.3, 0.35];         // updateTransform 목표 (자유 낙하 공간)
const SB_NEW_HALF_EXTENT_M = 0.1;                  // updateDimensions 목표 halfExtents (0.05→0.1)
const SB_SETTLE_Y_TOLERANCE_M = 0.03;              // 정착 y ≈ halfExtent 판정 허용 오차
const SB_POSE_TOLERANCE_M = 1e-3;                  // teleport 직후 pose 일치 허용 오차
const SB_SETTLE_REALTIME_DEADLINE_MS = 15000;      // 치수 변경 후 정착 폴링 실시간 상한
const SB_MIN_LIBRARY_CARDS = 6;                    // 라이브러리 템플릿 카드 최소 개수
// --expect=flow-graph (Phase 8 — __sim.flowGraph 파사드 + DOM/JSON 뷰어/스킵 재생)
const FG_BOOT_NODE_COUNT = 7;                      // arm-touch-box.sequence.json step 수
const FG_INSERT_AT = 2;                            // insertWait 삽입 위치
const FG_INSERTED_WAIT_SEC = 1;                    // defaultNodeFor('wait') 기본 durationSec
// 배리어(waitForCollision) 비활성 시 명목 길이 ≈ 0.4+2.5+0+0.5+0.5+2.0 = 5.9s.
// 배리어 이벤트 해제 경로 ≈ 6s+, timeout 경로 ≈ 11.9s — 7.5s 미만 done이면 스킵 증명.
const FG_SKIP_DONE_MAX_SIM_SEC = 7.5;
const FG_SKIP_SIM_BUDGET_SEC = 13;                 // 폴링 sim 예산 (timeout 경로도 포착)

// ── Phase 9: planner 게이트 상수 (규칙 기반 백엔드 — 결정론, 네트워크 없음) ──
// 규칙 기반 'box_a를 집어' → open→approach(moveJoints)→nudge(setJoints)→
// waitForCollision→close→wait→home = 7 step (2단 접근으로 배리어 직후 접촉 → 이벤트 해제).
const PLANNER_BOXA_STEP_COUNT = 7;
const PLANNER_SIM_BUDGET_SEC = 12;                 // Play→done sim 예산
const PLANNER_EVENT_DONE_MAX_SIM_SEC = 9;          // 이벤트 해제 경로(≈6.7s) vs timeout(≈12s) 구분

/** 두 엔티티 쌍 일치(순서 무관) */
function isPair(event, idA, idB) {
  return (event.a === idA && event.b === idB) || (event.a === idB && event.b === idA);
}

/**
 * 파사드 play() 호출 후 status 'done'까지 sim 시간 예산 안에서 폴링한다.
 * (pick-and-place / obstacle-avoidance 공용 — arm-sequence 블록은 기존 계약 유지)
 */
async function playAndAwaitDone(page, budgetSimSec) {
  const startSimTimeSec = await page.evaluate(() => {
    window.__sim.player.play();
    return window.__sim.engine.simTimeSec;
  });
  const realDeadline = Date.now() + PHASE6_REALTIME_DEADLINE_MS;
  let last = null;
  for (;;) {
    last = await page.evaluate(() => ({
      status: window.__sim.player.status,
      index: window.__sim.player.currentStepIndex,
      simTimeSec: window.__sim.engine.simTimeSec,
    }));
    if (last.status === 'done') break;
    if (last.simTimeSec - startSimTimeSec > budgetSimSec || Date.now() > realDeadline) break;
    await page.waitForTimeout(SEQ_POLL_INTERVAL_MS);
  }
  return { ...last, elapsedSimSec: last.simTimeSec - startSimTimeSec };
}

/**
 * UI 씬 프리셋 select로 씬을 전환하고(사용자 change 이벤트 경로 — scene-controls.ts),
 * __sim.spec.name이 대상 씬이 될 때까지 폴링한다. 성공/시간초과 모두 마지막 스냅샷을
 * 돌려준다 (FAIL detail에 사용). 전환 중에는 __sim이 잠시 undefined다(이전 씬 해제
 * → 새 빌드) — null 스냅샷은 건너뛰고 계속 폴링한다.
 */
async function switchSceneViaSelect(page, sceneName) {
  await page.evaluate((target) => {
    const select = document.querySelector('[data-testid="scene-select"]');
    select.value = target;
    select.dispatchEvent(new Event('change', { bubbles: true }));
  }, sceneName);
  const deadline = Date.now() + SWITCH_REALTIME_DEADLINE_MS;
  let snap = null;
  for (;;) {
    snap = await page.evaluate(() => {
      const s = window.__sim;
      if (!s) return null;
      return {
        name: s.spec.name,
        simTimeSec: s.engine.simTimeSec,
        entityIdCount: s.sceneHandle.entityIds.length,
        specEntityCount: s.spec.entities.length,
        hasGround: Boolean(s.spec.environment && s.spec.environment.ground),
        robotIds: s.robots.ids(),
        selectValue: document.querySelector('[data-testid="scene-select"]')?.value ?? null,
      };
    });
    if (snap && snap.name === sceneName) return snap;
    if (Date.now() > deadline) return snap;
    await page.waitForTimeout(SEQ_POLL_INTERVAL_MS);
  }
}

/** 현재 활성 엔진의 sim 시간이 minAdvanceSec만큼 실제 전진하는지 폴링 판정 */
async function awaitSimAdvance(page, minAdvanceSec) {
  const fromSec = await page.evaluate(() => window.__sim?.engine.simTimeSec ?? 0);
  const deadline = Date.now() + SWITCH_REALTIME_DEADLINE_MS;
  for (;;) {
    const toSec = await page.evaluate(() => window.__sim?.engine.simTimeSec ?? 0);
    if (toSec - fromSec >= minAdvanceSec) return { advanced: true, fromSec, toSec };
    if (Date.now() > deadline) return { advanced: false, fromSec, toSec };
    await page.waitForTimeout(SEQ_POLL_INTERVAL_MS);
  }
}

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

    // ── Phase 6: pick-and-place — 밀기(push) 픽앤플레이스 + SENSOR_ZONE 감지 ──
    if (expectArg === 'pick-and-place') {
      const initial = await page.evaluate(() => {
        const p = window.__sim?.player;
        return p ? { status: p.status, stepCount: p.stepCount } : null;
      });
      if (initial?.status === 'idle' && initial.stepCount === PNP_STEP_COUNT) {
        pass(`pick-and-place: sequence loaded, no autoplay (idle, ${PNP_STEP_COUNT} steps)`);
      } else {
        fail(
          `pick-and-place: sequence loaded, no autoplay (idle, ${PNP_STEP_COUNT} steps)`,
          `initial=${JSON.stringify(initial)}`,
        );
      }

      if (!initial) {
        fail('pick-and-place: interaction checks skipped', 'player facade missing');
      } else {
        const last = await playAndAwaitDone(page, PNP_SIM_TIME_BUDGET_SEC);
        const history = await page.evaluate(
          (limit) => window.__sim.collision.recent(limit),
          HISTORY_FETCH_LIMIT,
        );

        // 1) arm×cargo 접촉 start (EventQueue 유래 — CLAUDE.md §2.4)
        const armCargoStarts = history.filter(
          (e) => e.phase === 'start' && e.kind === 'contact' && isPair(e, 'arm', 'cargo'),
        );
        if (armCargoStarts.length >= 1) {
          pass('pick-and-place: collision history has arm×cargo contact start',
            `timeSec=[${armCargoStarts.map((e) => e.timeSec.toFixed(3)).join(', ')}]`);
        } else {
          fail('pick-and-place: collision history has arm×cargo contact start',
            `history=${JSON.stringify(history.slice(-30))}`);
        }

        // 2) cargo×drop_zone sensor start — 스윕 밀기가 카고를 영역 안으로 옮겼다
        const sensorStarts = history.filter(
          (e) => e.phase === 'start' && e.kind === 'sensor' && isPair(e, 'cargo', 'drop_zone'),
        );
        if (sensorStarts.length >= 1) {
          pass('pick-and-place: cargo entered drop_zone (sensor start)',
            `timeSec=[${sensorStarts.map((e) => e.timeSec.toFixed(3)).join(', ')}]`);
        } else {
          fail('pick-and-place: cargo entered drop_zone (sensor start)',
            `history=${JSON.stringify(history.slice(-30))}`);
        }

        // 3) done — timeout(6s) 경로가 아니라 실제 충돌 해제 경로의 시간 안에서
        if (last.status === 'done' && last.elapsedSimSec < PNP_EVENT_DONE_MAX_SIM_SEC) {
          pass(`pick-and-place: sequence done within ${PNP_EVENT_DONE_MAX_SIM_SEC}s (event-released barrier)`,
            `elapsed=${last.elapsedSimSec.toFixed(2)}s`);
        } else {
          fail(`pick-and-place: sequence done within ${PNP_EVENT_DONE_MAX_SIM_SEC}s (event-released barrier)`,
            `last=${JSON.stringify(last)}`);
        }
      }
    }

    // ── Phase 6: obstacle-avoidance — 위로 넘어가는 경로, 기둥 무접촉 ──
    if (expectArg === 'obstacle-avoidance') {
      const initial = await page.evaluate(() => {
        const p = window.__sim?.player;
        return p ? { status: p.status, stepCount: p.stepCount } : null;
      });
      if (initial?.status === 'idle' && initial.stepCount === OA_STEP_COUNT) {
        pass(`obstacle-avoidance: sequence loaded, no autoplay (idle, ${OA_STEP_COUNT} steps)`);
      } else {
        fail(
          `obstacle-avoidance: sequence loaded, no autoplay (idle, ${OA_STEP_COUNT} steps)`,
          `initial=${JSON.stringify(initial)}`,
        );
      }

      if (!initial) {
        fail('obstacle-avoidance: interaction checks skipped', 'player facade missing');
      } else {
        const last = await playAndAwaitDone(page, OA_SIM_TIME_BUDGET_SEC);
        const history = await page.evaluate(
          (limit) => window.__sim.collision.recent(limit),
          HISTORY_FETCH_LIMIT,
        );

        // 1) arm×pillar 이벤트 0건 — 회피 경로가 기둥을 건드리지 않았다
        //    (pillar collider는 emitEvents:true — 닿았다면 반드시 이력에 남는다)
        const armPillarEvents = history.filter((e) => isPair(e, 'arm', 'pillar'));
        if (armPillarEvents.length === 0) {
          pass('obstacle-avoidance: NO arm×pillar events (path clears the pillar)');
        } else {
          fail('obstacle-avoidance: NO arm×pillar events (path clears the pillar)',
            `events=${JSON.stringify(armPillarEvents)}`);
        }

        // 2) arm×target_box 접촉 start — 경로의 종착이 실제 접촉으로 감지됐다
        const armTargetStarts = history.filter(
          (e) => e.phase === 'start' && e.kind === 'contact' && isPair(e, 'arm', 'target_box'),
        );
        if (armTargetStarts.length >= 1) {
          pass('obstacle-avoidance: collision history has arm×target_box start',
            `timeSec=[${armTargetStarts.map((e) => e.timeSec.toFixed(3)).join(', ')}]`);
        } else {
          fail('obstacle-avoidance: collision history has arm×target_box start',
            `history=${JSON.stringify(history.slice(-30))}`);
        }

        // 3) done — 이벤트 해제 경로의 시간 안에서
        if (last.status === 'done' && last.elapsedSimSec < OA_EVENT_DONE_MAX_SIM_SEC) {
          pass(`obstacle-avoidance: sequence done within ${OA_EVENT_DONE_MAX_SIM_SEC}s (event-released barrier)`,
            `elapsed=${last.elapsedSimSec.toFixed(2)}s`);
        } else {
          fail(`obstacle-avoidance: sequence done within ${OA_EVENT_DONE_MAX_SIM_SEC}s (event-released barrier)`,
            `last=${JSON.stringify(last)}`);
        }
      }
    }

    // ── Phase 6: collision-testbed — 로봇 없는 낙하/미끄럼/전도 쇼케이스 ──
    if (expectArg === 'collision-testbed') {
      // 물리 루프는 부팅 직후 자동 재생 — TESTBED_SETTLE_SIM_SEC까지 폴링 대기
      const realDeadline = Date.now() + PHASE6_REALTIME_DEADLINE_MS;
      let simTimeSec = 0;
      for (;;) {
        simTimeSec = await page.evaluate(() => window.__sim?.engine.simTimeSec ?? 0);
        if (simTimeSec >= TESTBED_SETTLE_SIM_SEC || Date.now() > realDeadline) break;
        await page.waitForTimeout(SEQ_POLL_INTERVAL_MS);
      }
      if (simTimeSec >= TESTBED_SETTLE_SIM_SEC) {
        pass(`collision-testbed: sim advanced to ${TESTBED_SETTLE_SIM_SEC}s`, `simTimeSec=${simTimeSec.toFixed(2)}`);
      } else {
        fail(`collision-testbed: sim advanced to ${TESTBED_SETTLE_SIM_SEC}s`, `simTimeSec=${simTimeSec}`);
      }

      const snapshot = await page.evaluate((limit) => {
        const s = window.__sim;
        const dynamicYById = {};
        for (const entity of s.spec.entities) {
          if (entity.physics?.bodyType !== 'dynamic') continue;
          const bodies = s.world.bodiesOfEntity(entity.id);
          dynamicYById[entity.id] =
            bodies.length > 0 ? s.world.getPose(bodies[0]).position[1] : null;
        }
        return { history: s.collision.recent(limit), dynamicYById };
      }, HISTORY_FETCH_LIMIT);

      // 1) TESTBED_EVENT_WINDOW_SEC 안에 서로 다른 contact start 쌍 ≥ 3
      const contactPairs = new Set(
        snapshot.history
          .filter((e) => e.phase === 'start' && e.kind === 'contact' && e.timeSec <= TESTBED_EVENT_WINDOW_SEC)
          .map((e) => [e.a, e.b].sort().join('×')),
      );
      if (contactPairs.size >= TESTBED_MIN_CONTACT_PAIRS) {
        pass(`collision-testbed: ≥${TESTBED_MIN_CONTACT_PAIRS} distinct contact start pairs within ${TESTBED_EVENT_WINDOW_SEC}s`,
          `pairs=[${[...contactPairs].join(', ')}]`);
      } else {
        fail(`collision-testbed: ≥${TESTBED_MIN_CONTACT_PAIRS} distinct contact start pairs within ${TESTBED_EVENT_WINDOW_SEC}s`,
          `pairs=[${[...contactPairs].join(', ')}]`);
      }

      // 2) sensor start ≥ 1 (slider가 slide_gate 통과 — 물리 반응 없는 감지)
      const sensorStarts = snapshot.history.filter(
        (e) => e.phase === 'start' && e.kind === 'sensor' && e.timeSec <= TESTBED_EVENT_WINDOW_SEC,
      );
      if (sensorStarts.length >= TESTBED_MIN_SENSOR_EVENTS) {
        pass(`collision-testbed: ≥${TESTBED_MIN_SENSOR_EVENTS} sensor start within ${TESTBED_EVENT_WINDOW_SEC}s`,
          `events=[${sensorStarts.map((e) => `${e.a}×${e.b}@${e.timeSec.toFixed(2)}`).join(', ')}]`);
      } else {
        fail(`collision-testbed: ≥${TESTBED_MIN_SENSOR_EVENTS} sensor start within ${TESTBED_EVENT_WINDOW_SEC}s`,
          `history=${JSON.stringify(snapshot.history.slice(0, 30))}`);
      }

      // 3) 전 동적 바디가 바닥 위(y > 0)에 정착 — 관통/이탈 없음
      const entries = Object.entries(snapshot.dynamicYById);
      const sunk = entries.filter(([, y]) => typeof y !== 'number' || y <= 0);
      if (entries.length > 0 && sunk.length === 0) {
        pass('collision-testbed: all dynamic bodies settled above ground (y > 0)',
          JSON.stringify(Object.fromEntries(entries.map(([id, y]) => [id, y.toFixed(3)]))));
      } else {
        fail('collision-testbed: all dynamic bodies settled above ground (y > 0)',
          JSON.stringify(snapshot.dynamicYById));
      }
    }

    // ── Phase 6: scene-switch — 런타임 씬 전환 스모크 (UI select 경유 왕복) ──
    if (expectArg === 'scene-switch') {
      // 0) 부트 씬 확인 (arm-and-boxes + 로봇) — 전환 전 기준 상태
      const boot = await page.evaluate(() => ({
        name: window.__sim?.spec.name ?? null,
        robotIds: window.__sim?.robots.ids() ?? [],
      }));
      if (boot.name === SWITCH_BACK_SCENE && boot.robotIds.includes('arm')) {
        pass(`scene-switch: boot scene is ${SWITCH_BACK_SCENE} with robot arm`);
      } else {
        fail(`scene-switch: boot scene is ${SWITCH_BACK_SCENE} with robot arm`, JSON.stringify(boot));
      }

      // 1) UI select로 collision-testbed 전환 → __sim.spec.name 갱신 (게이트 계약:
      //    전환 후 __sim은 항상 새 씬의 새 인스턴스들을 가리킨다)
      const toTarget = await switchSceneViaSelect(page, SWITCH_TARGET_SCENE);
      if (toTarget?.name === SWITCH_TARGET_SCENE) {
        pass(`scene-switch: switched to ${SWITCH_TARGET_SCENE} via UI select (spec.name updated)`);
      } else {
        fail(`scene-switch: switched to ${SWITCH_TARGET_SCENE} via UI select (spec.name updated)`, JSON.stringify(toTarget));
      }

      if (toTarget) {
        // 2) 엔티티 수 일치: sceneHandle.entityIds = spec.entities + ground 예약 엔티티
        const expectedCount = toTarget.specEntityCount + (toTarget.hasGround ? 1 : 0);
        if (toTarget.entityIdCount === expectedCount) {
          pass('scene-switch: entity count matches spec after switch',
            `${toTarget.entityIdCount} = ${toTarget.specEntityCount} entities + ${toTarget.hasGround ? 1 : 0} ground`);
        } else {
          fail('scene-switch: entity count matches spec after switch', JSON.stringify(toTarget));
        }
        // 로봇 없는 씬 — robots 파사드가 빈 목록인지 (이전 씬 상태 누수 검출)
        if (toTarget.robotIds.length === 0) {
          pass('scene-switch: robots facade empty in robot-less scene (no leak from previous scene)');
        } else {
          fail('scene-switch: robots facade empty in robot-less scene (no leak from previous scene)',
            `robotIds=${JSON.stringify(toTarget.robotIds)}`);
        }
      }

      // 3) 전환 직후 새 엔진의 sim 시간이 실제 전진 (물리 루프 자동 시작 정책)
      const advanceAfterSwitch = await awaitSimAdvance(page, SWITCH_MIN_SIM_ADVANCE_SEC);
      if (advanceAfterSwitch.advanced) {
        pass('scene-switch: sim advances after switch',
          `${advanceAfterSwitch.fromSec.toFixed(2)}s → ${advanceAfterSwitch.toSec.toFixed(2)}s`);
      } else {
        fail('scene-switch: sim advances after switch', JSON.stringify(advanceAfterSwitch));
      }

      // 4) 되돌아오기 (arm-and-boxes) — URDF 재로드 경로 포함 왕복 전환
      const backHome = await switchSceneViaSelect(page, SWITCH_BACK_SCENE);
      if (backHome?.name === SWITCH_BACK_SCENE) {
        pass(`scene-switch: switched back to ${SWITCH_BACK_SCENE} (spec.name updated)`);
      } else {
        fail(`scene-switch: switched back to ${SWITCH_BACK_SCENE} (spec.name updated)`, JSON.stringify(backHome));
      }
      if (backHome) {
        const expectedBackCount = backHome.specEntityCount + (backHome.hasGround ? 1 : 0);
        if (backHome.entityIdCount === expectedBackCount && backHome.robotIds.includes('arm')) {
          pass('scene-switch: entity count + robot arm restored after switch-back',
            `entities=${backHome.entityIdCount}, robots=[${backHome.robotIds.join(', ')}]`);
        } else {
          fail('scene-switch: entity count + robot arm restored after switch-back', JSON.stringify(backHome));
        }
        if (backHome.selectValue === SWITCH_BACK_SCENE) {
          pass('scene-switch: UI select reflects active scene');
        } else {
          fail('scene-switch: UI select reflects active scene', `selectValue=${backHome.selectValue}`);
        }
      }
      const advanceAfterBack = await awaitSimAdvance(page, SWITCH_MIN_SIM_ADVANCE_SEC);
      if (advanceAfterBack.advanced) {
        pass('scene-switch: sim advances after switch-back',
          `${advanceAfterBack.fromSec.toFixed(2)}s → ${advanceAfterBack.toSec.toFixed(2)}s`);
      } else {
        fail('scene-switch: sim advances after switch-back', JSON.stringify(advanceAfterBack));
      }
    }

    // ── Phase 7: scene-builder — __sim.editor/__sim.history 파사드로 씬 편집 검증 ──
    if (expectArg === 'scene-builder') {
      // (f) 라이브러리 DOM: 템플릿 카드 ≥ SB_MIN_LIBRARY_CARDS (워크스페이스 좌 슬롯)
      const cardCount = await page.$$eval('[data-testid^="library-card-"]', (els) => els.length);
      if (cardCount >= SB_MIN_LIBRARY_CARDS) {
        pass(`scene-builder: library renders >= ${SB_MIN_LIBRARY_CARDS} template cards`, `cards=${cardCount}`);
      } else {
        fail(`scene-builder: library renders >= ${SB_MIN_LIBRARY_CARDS} template cards`, `cards=${cardCount}`);
      }

      // (a) 프로그램적 addEntity — box 템플릿을 지정 위치에 추가 (라이브러리 배치 경로)
      const added = await page.evaluate(async (position) => {
        const s = window.__sim;
        const before = s.editor.entityIds().length;
        const id = await s.editor.placeTemplate('box', position);
        const bodies = s.world.bodiesOfEntity(id);
        return {
          id,
          before,
          after: s.editor.entityIds().length,
          bodyCount: bodies.length,
          pose: bodies.length > 0 ? s.world.getPose(bodies[0]) : null,
          pickables: s.editor.pickableIds(),
          selected: s.editor.selectedId(),
        };
      }, SB_ADD_POSITION);

      if (added.after === added.before + 1 && added.bodyCount === 1) {
        pass('scene-builder: addEntity(+1 entity, body created in world)',
          `id=${added.id}, entities ${added.before}→${added.after}`);
      } else {
        fail('scene-builder: addEntity(+1 entity, body created in world)', JSON.stringify(added));
      }
      const posOk = added.pose
        && Math.abs(added.pose.position[0] - SB_ADD_POSITION[0]) < SB_POSE_TOLERANCE_M
        && Math.abs(added.pose.position[2] - SB_ADD_POSITION[2]) < SB_POSE_TOLERANCE_M;
      if (posOk) {
        pass('scene-builder: added body spawned at requested x/z', JSON.stringify(added.pose.position));
      } else {
        fail('scene-builder: added body spawned at requested x/z', JSON.stringify(added.pose));
      }
      if (added.pickables.includes(added.id) && added.selected === added.id) {
        pass('scene-builder: new entity pickable + auto-selected');
      } else {
        fail('scene-builder: new entity pickable + auto-selected',
          `pickables=${JSON.stringify(added.pickables)} selected=${added.selected}`);
      }
      const inInspector = await page.$$eval(
        '[data-testid="inspector-entity"]',
        (rows, id) => rows.some((r) => (r.textContent ?? '').includes(id)),
        added.id,
      );
      if (inInspector) pass('scene-builder: inspector list shows new entity');
      else fail('scene-builder: inspector list shows new entity', `id=${added.id}`);

      // (b) updateTransform → 물리 바디가 즉시 teleport (getPose가 반영)
      const moved = await page.evaluate(({ id, target }) => {
        const s = window.__sim;
        s.editor.updateTransform(id, { position: target });
        return s.world.getPose(s.world.bodiesOfEntity(id)[0]).position;
      }, { id: added.id, target: SB_MOVE_POSITION });
      const movedOk = Math.abs(moved[0] - SB_MOVE_POSITION[0]) < SB_POSE_TOLERANCE_M
        && Math.abs(moved[1] - SB_MOVE_POSITION[1]) < SB_POSE_TOLERANCE_M
        && Math.abs(moved[2] - SB_MOVE_POSITION[2]) < SB_POSE_TOLERANCE_M;
      if (movedOk) pass('scene-builder: updateTransform teleports body (getPose reflects)', JSON.stringify(moved));
      else fail('scene-builder: updateTransform teleports body (getPose reflects)', JSON.stringify(moved));

      // (c) updateDimensions(halfExtents 0.05→0.1) → collider가 실제로 커졌다는 것을
      //     물리 "행동"으로 관측: 자유 낙하 후 정착 y ≈ 새 halfExtent (0.1)
      await page.evaluate(({ id, half }) => {
        window.__sim.editor.updateDimensions(id, { kind: 'box', halfExtents: [half, half, half] });
      }, { id: added.id, half: SB_NEW_HALF_EXTENT_M });
      const settleDeadline = Date.now() + SB_SETTLE_REALTIME_DEADLINE_MS;
      let settledY = null;
      for (;;) {
        settledY = await page.evaluate((id) => {
          const s = window.__sim;
          const bodies = s.world.bodiesOfEntity(id);
          return bodies.length > 0 ? s.world.getPose(bodies[0]).position[1] : null;
        }, added.id);
        if (settledY !== null && Math.abs(settledY - SB_NEW_HALF_EXTENT_M) < SB_SETTLE_Y_TOLERANCE_M) break;
        if (Date.now() > settleDeadline) break;
        await page.waitForTimeout(SEQ_POLL_INTERVAL_MS);
      }
      if (settledY !== null && Math.abs(settledY - SB_NEW_HALF_EXTENT_M) < SB_SETTLE_Y_TOLERANCE_M) {
        pass(`scene-builder: updateDimensions observable (settled y ≈ ${SB_NEW_HALF_EXTENT_M})`, `y=${settledY.toFixed(4)}`);
      } else {
        fail(`scene-builder: updateDimensions observable (settled y ≈ ${SB_NEW_HALF_EXTENT_M})`, `y=${settledY}`);
      }

      // (d) removeEntity → 바디/픽킹 대상 정리
      const removed = await page.evaluate((id) => {
        const s = window.__sim;
        s.editor.removeEntity(id);
        return {
          bodyCount: s.world.bodiesOfEntity(id).length,
          pickables: s.editor.pickableIds(),
          count: s.editor.entityIds().length,
        };
      }, added.id);
      if (removed.bodyCount === 0 && !removed.pickables.includes(added.id) && removed.count === added.before) {
        pass('scene-builder: removeEntity cleans up (no bodies, pickable gone)', JSON.stringify(removed));
      } else {
        fail('scene-builder: removeEntity cleans up (no bodies, pickable gone)', JSON.stringify(removed));
      }

      // (e) undo → 전체 재로드로 제거 이전 상태 복원 (엔티티 수 +1, 바디 재생성)
      //     undo는 씬을 재빌드하므로 window.__sim이 새 핸들로 교체된 뒤를 읽는다.
      const undone = await page.evaluate(async (id) => {
        const ok = await window.__sim.history.undo();
        const s = window.__sim;
        return {
          ok,
          ids: s ? s.editor.entityIds() : [],
          bodyCount: s ? s.world.bodiesOfEntity(id).length : 0,
        };
      }, added.id);
      if (undone.ok && undone.ids.includes(added.id) && undone.bodyCount >= 1) {
        pass('scene-builder: undo restores removed entity (full reload)',
          `ids=[${undone.ids.join(', ')}]`);
      } else {
        fail('scene-builder: undo restores removed entity (full reload)', JSON.stringify(undone));
      }
      // undo 재로드(URDF 포함) 후 새 엔진이 실제로 전진하는지 확인
      const advanceAfterUndo = await awaitSimAdvance(page, SWITCH_MIN_SIM_ADVANCE_SEC);
      if (advanceAfterUndo.advanced) {
        pass('scene-builder: sim advances after undo reload',
          `${advanceAfterUndo.fromSec.toFixed(2)}s → ${advanceAfterUndo.toSec.toFixed(2)}s`);
      } else {
        fail('scene-builder: sim advances after undo reload', JSON.stringify(advanceAfterUndo));
      }
    }

    // ── Phase 8: flow-graph — 그래프 편집이 항상 유효한 시퀀스로 직렬화 (불변식 §2.8) ──
    if (expectArg === 'flow-graph') {
      // (a) 페인 표시 + 렌더된 DOM 노드 수 = 시퀀스 step 수 (arm-touch-box 7개)
      const paneVisible = await page.$eval(
        '[data-testid="workspace-flow-graph"]',
        (el) => getComputedStyle(el).display !== 'none',
      );
      const domNodes0 = await page.$$eval('[data-testid="flow-node"]', (els) => els.length);
      const facade0 = await page.evaluate(() => ({
        visible: window.__sim.flowGraph.visible(),
        nodeCount: window.__sim.flowGraph.nodeCount(),
        kinds: window.__sim.flowGraph.kinds(),
      }));
      if (
        paneVisible && facade0.visible
        && domNodes0 === FG_BOOT_NODE_COUNT && facade0.nodeCount === FG_BOOT_NODE_COUNT
      ) {
        pass(`flow-graph: pane visible with ${FG_BOOT_NODE_COUNT} rendered nodes`,
          `dom=${domNodes0}, kinds=[${facade0.kinds.join(',')}]`);
      } else {
        fail(`flow-graph: pane visible with ${FG_BOOT_NODE_COUNT} rendered nodes`,
          JSON.stringify({ paneVisible, domNodes0, facade0 }));
      }

      // (b) insertWait(2) → 8노드 + 직렬화 시퀀스가 파싱·검증 통과 (facade lastValidation)
      const inserted = await page.evaluate((at) => {
        const fg = window.__sim.flowGraph;
        const ok = fg.insertWait(at);
        return {
          ok,
          nodeCount: fg.nodeCount(),
          lastValidation: fg.lastValidation(),
          sequenceJson: fg.sequenceJson(),
        };
      }, FG_INSERT_AT);
      let insertedSeq = null;
      try { insertedSeq = JSON.parse(inserted.sequenceJson); } catch { insertedSeq = null; }
      const insertedStep = insertedSeq?.steps?.[FG_INSERT_AT];
      const domNodesAfterInsert = await page.$$eval('[data-testid="flow-node"]', (els) => els.length);
      if (
        inserted.ok && inserted.lastValidation === 'ok'
        && inserted.nodeCount === FG_BOOT_NODE_COUNT + 1
        && domNodesAfterInsert === FG_BOOT_NODE_COUNT + 1
        && insertedSeq && insertedSeq.steps.length === FG_BOOT_NODE_COUNT + 1
        && insertedStep && insertedStep.kind === 'wait'
        && insertedStep.durationSec === FG_INSERTED_WAIT_SEC
      ) {
        pass(`flow-graph: insertWait(${FG_INSERT_AT}) → ${FG_BOOT_NODE_COUNT + 1} nodes, sequence parses + validates`,
          `steps[${FG_INSERT_AT}]=${JSON.stringify(insertedStep)}`);
      } else {
        fail(`flow-graph: insertWait(${FG_INSERT_AT}) → ${FG_BOOT_NODE_COUNT + 1} nodes, sequence parses + validates`,
          JSON.stringify({ inserted, domNodesAfterInsert }));
      }

      // (f) '{} JSON' 뷰어에 삽입된 wait step 반영 (그래프 편집 ↔ JSON 실시간 동기)
      await page.click('[data-testid="json-toggle"]');
      const viewerText = await page.$eval('[data-testid="json-content"]', (el) => el.textContent ?? '');
      let viewerSeq = null;
      try { viewerSeq = JSON.parse(viewerText); } catch { viewerSeq = null; }
      const viewerStep = viewerSeq?.steps?.[FG_INSERT_AT];
      if (viewerStep && viewerStep.kind === 'wait' && viewerStep.durationSec === FG_INSERTED_WAIT_SEC) {
        pass('flow-graph: JSON viewer shows inserted wait step');
      } else {
        fail('flow-graph: JSON viewer shows inserted wait step', viewerText.slice(0, 300));
      }
      await page.click('[data-testid="json-close"]');

      // (c) reorder: 첫 노드를 인덱스 1로 — 시퀀스 순서가 그대로 갱신 (앞 두 kind 스왑)
      const reordered = await page.evaluate(() => {
        const fg = window.__sim.flowGraph;
        const before = fg.kinds();
        const ids = fg.nodeIds();
        const ok = fg.reorder(ids[0], 1);
        return {
          ok,
          before,
          after: fg.kinds(),
          seqKinds: JSON.parse(fg.sequenceJson()).steps.map((s) => s.kind),
          lastValidation: fg.lastValidation(),
        };
      });
      const expectedAfter = [reordered.before[1], reordered.before[0], ...reordered.before.slice(2)];
      const orderOk = JSON.stringify(reordered.after) === JSON.stringify(expectedAfter)
        && JSON.stringify(reordered.seqKinds) === JSON.stringify(expectedAfter);
      if (reordered.ok && orderOk && reordered.lastValidation === 'ok') {
        pass('flow-graph: reorder updates sequence order accordingly',
          `[${reordered.before.slice(0, 2).join(',')}] → [${reordered.after.slice(0, 2).join(',')}]`);
      } else {
        fail('flow-graph: reorder updates sequence order accordingly', JSON.stringify(reordered));
      }

      // 결정론 리셋: (d)는 편집 전 원본 시퀀스에서 시작해야 한다 — 페이지 재로드
      await page.goto(url, { waitUntil: 'load' });
      await page.waitForFunction(() => window.__sim !== undefined, undefined, { timeout: 15000 });

      // (d) waitForCollision 비활성 → 재생: 배리어 없이 완주(빠른 done, timeout 경고
      //     없음) + 스킵 노드는 한 번도 active로 표시되지 않는다
      const disabled = await page.evaluate(() => {
        const fg = window.__sim.flowGraph;
        const kinds = fg.kinds();
        const ids = fg.nodeIds();
        const index = kinds.indexOf('waitForCollision');
        const id = ids[index];
        const ok = fg.setEnabled(id, false);
        return { ok, id, index, lastValidation: fg.lastValidation() };
      });
      if (disabled.ok && disabled.index >= 0 && disabled.lastValidation === 'ok') {
        pass('flow-graph: setEnabled(waitForCollision, false) commits (sequence still valid)',
          `nodeId=${disabled.id} (index ${disabled.index})`);
      } else {
        fail('flow-graph: setEnabled(waitForCollision, false) commits (sequence still valid)',
          JSON.stringify(disabled));
      }
      const run = await playAndAwaitDone(page, FG_SKIP_SIM_BUDGET_SEC);
      if (run.status === 'done' && run.elapsedSimSec < FG_SKIP_DONE_MAX_SIM_SEC) {
        pass(`flow-graph: disabled barrier → done within ${FG_SKIP_DONE_MAX_SIM_SEC}s (no waitForCollision wait)`,
          `elapsed=${run.elapsedSimSec.toFixed(2)}s`);
      } else {
        fail(`flow-graph: disabled barrier → done within ${FG_SKIP_DONE_MAX_SIM_SEC}s (no waitForCollision wait)`,
          JSON.stringify(run));
      }
      const skipObs = await page.evaluate((id) => ({
        everActive: window.__sim.flowGraph.everActiveNodeIds(),
        status: window.__sim.flowGraph.nodeStatuses()[id] ?? '(none)',
        consoleText: document.querySelector('[data-testid="console-panel"]')?.textContent ?? '',
      }), disabled.id);
      const noTimeoutWarn = !skipObs.consoleText.includes('감지되지 않았습니다');
      if (
        !skipObs.everActive.includes(disabled.id)
        && skipObs.status !== 'active' && skipObs.status !== 'error'
        && noTimeoutWarn
      ) {
        pass('flow-graph: skipped node never active, no timeout warn',
          `status=${skipObs.status}, everActive=[${skipObs.everActive.join(',')}]`);
      } else {
        fail('flow-graph: skipped node never active, no timeout warn',
          JSON.stringify({ everActive: skipObs.everActive, status: skipObs.status, noTimeoutWarn }));
      }

      // (d2) Stop → 캔버스 런 상태 완전 리셋 (unarm + statuses/everActive 클리어),
      //      이어서 재-Play가 처음부터 완주한다 — Stop→Play 재실행에 이전 런 잔상 없음
      const stopped = await page.evaluate(() => {
        window.__sim.player.stop();
        const statuses = window.__sim.flowGraph.nodeStatuses();
        return {
          everActive: window.__sim.flowGraph.everActiveNodeIds(),
          nonPending: Object.values(statuses).filter((s) => s !== 'pending').length,
        };
      });
      if (stopped.everActive.length === 0 && stopped.nonPending === 0) {
        pass('flow-graph: Stop clears run state (all statuses pending, everActive empty)');
      } else {
        fail('flow-graph: Stop clears run state (all statuses pending, everActive empty)',
          JSON.stringify(stopped));
      }
      const replay = await playAndAwaitDone(page, FG_SKIP_SIM_BUDGET_SEC);
      if (replay.status === 'done' && replay.elapsedSimSec < FG_SKIP_DONE_MAX_SIM_SEC) {
        pass('flow-graph: Stop → Play replays from start to done (re-arm + revalidation path)',
          `elapsed=${replay.elapsedSimSec.toFixed(2)}s`);
      } else {
        fail('flow-graph: Stop → Play replays from start to done (re-arm + revalidation path)',
          JSON.stringify(replay));
      }

      // (e) 노드 삭제 → 노드 수 감소 + 시퀀스 여전히 유효 (§2.8)
      const removedNode = await page.evaluate(() => {
        const fg = window.__sim.flowGraph;
        const before = fg.nodeCount();
        const kinds = fg.kinds();
        const ids = fg.nodeIds();
        const ok = fg.remove(ids[kinds.indexOf('wait')]);
        return {
          ok,
          before,
          after: fg.nodeCount(),
          lastValidation: fg.lastValidation(),
          stepCount: JSON.parse(fg.sequenceJson()).steps.length,
        };
      });
      const domAfterRemove = await page.$$eval('[data-testid="flow-node"]', (els) => els.length);
      if (
        removedNode.ok && removedNode.after === removedNode.before - 1
        && domAfterRemove === removedNode.after
        && removedNode.lastValidation === 'ok' && removedNode.stepCount === removedNode.after
      ) {
        pass('flow-graph: remove node → count drops, sequence valid',
          `nodes ${removedNode.before}→${removedNode.after} (dom=${domAfterRemove})`);
      } else {
        fail('flow-graph: remove node → count drops, sequence valid',
          JSON.stringify({ removedNode, domAfterRemove }));
      }

      // (g) 로봇 rename → 플로우 참조(기본 robot·between) 자동 재동기 + 편집 계속 가능
      //     (rename 후 모든 플로우 편집이 "씬에 없는 엔티티"로 거부되는 잠김 회귀 방지)
      const renamed = await page.evaluate(() => {
        const s = window.__sim;
        s.editor.renameEntity('arm', 'arm_renamed');
        const fg = s.flowGraph;
        const seq = JSON.parse(fg.sequenceJson());
        const okInsert = fg.insertWait(0); // rename 이후에도 §2.8 파이프라인이 통과해야 함
        return {
          robotIds: s.robots.ids(),
          seqRobot: seq.robot,
          betweens: seq.steps
            .filter((st) => st.kind === 'waitForCollision')
            .map((st) => st.between),
          okInsert,
          lastValidation: fg.lastValidation(),
        };
      });
      const betweenRemapped = renamed.betweens.length >= 1
        && renamed.betweens.every((b) => b.includes('arm_renamed') && !b.includes('arm'));
      if (
        renamed.robotIds.includes('arm_renamed')
        && renamed.seqRobot === 'arm_renamed'
        && betweenRemapped
        && renamed.okInsert
        && renamed.lastValidation === 'ok'
      ) {
        pass('flow-graph: robot rename resyncs flow refs, editing still works',
          `robot=${renamed.seqRobot}, betweens=${JSON.stringify(renamed.betweens)}`);
      } else {
        fail('flow-graph: robot rename resyncs flow refs, editing still works',
          JSON.stringify(renamed));
      }
    }

    // ── Phase 9: planner — 자연어 → 검증된 시퀀스 → 그래프 로드 (§2.9 무자동재생) ──
    if (expectArg === 'planner') {
      // (a) generate('box_a를 집어') → type 'sequence' + 그래프 로드 + AI 배지 + 무자동재생
      const genA = await page.evaluate(async () => {
        const res = await window.__sim.planner.generate('box_a를 집어');
        const fg = window.__sim.flowGraph;
        return {
          type: res.type,
          last: window.__sim.planner.lastResult(),
          nodeCount: fg.nodeCount(),
          kinds: fg.kinds(),
          isLoaded: window.__sim.planner.isLoadedIntoGraph(),
          playerStatus: window.__sim.planner.playerStatus(),
          sequenceJson: fg.sequenceJson(),
        };
      });

      if (genA.type === 'sequence') pass('planner: generate(box_a) returns sequence');
      else fail('planner: generate(box_a) returns sequence', JSON.stringify(genA));

      if (genA.nodeCount === PLANNER_BOXA_STEP_COUNT && genA.last?.stepCount === PLANNER_BOXA_STEP_COUNT) {
        pass(`planner: graph loaded to generated length (${PLANNER_BOXA_STEP_COUNT} nodes)`,
          `kinds=[${genA.kinds.join(',')}]`);
      } else {
        fail(`planner: graph loaded to generated length (${PLANNER_BOXA_STEP_COUNT} nodes)`,
          JSON.stringify(genA));
      }

      // AI 배지: data-origin='generated' 노드 (또는 배지 텍스트 'AI') ≥ 1
      const aiBadges = await page.$$eval('[data-testid="flow-node"]',
        (els) => els.filter((el) => el.dataset.origin === 'generated'
          || (el.textContent ?? '').includes('AI')).length);
      const domNodesA = await page.$$eval('[data-testid="flow-node"]', (els) => els.length);
      if (aiBadges >= 1 && domNodesA === PLANNER_BOXA_STEP_COUNT) {
        pass('planner: generated nodes carry AI badge + DOM count matches', `ai=${aiBadges}, dom=${domNodesA}`);
      } else {
        fail('planner: generated nodes carry AI badge + DOM count matches', `ai=${aiBadges}, dom=${domNodesA}`);
      }

      if (genA.isLoaded) pass('planner: isLoadedIntoGraph() true after sequence generate');
      else fail('planner: isLoadedIntoGraph() true after sequence generate', JSON.stringify(genA));

      // §2.9 증명: 생성만으로 자동 재생하지 않는다 (player not running)
      if (genA.playerStatus !== 'running') {
        pass('planner: NO autoplay after generate (§2.9 — player not running)', `status=${genA.playerStatus}`);
      } else {
        fail('planner: NO autoplay after generate (§2.9 — player not running)', `status=${genA.playerStatus}`);
      }

      // 라이브 시퀀스 JSON 유효 + 그래프 kinds와 일치
      let seqA = null;
      try { seqA = JSON.parse(genA.sequenceJson); } catch { seqA = null; }
      const kindsMatch = seqA && Array.isArray(seqA.steps)
        && seqA.steps.length === PLANNER_BOXA_STEP_COUNT
        && JSON.stringify(seqA.steps.map((s) => s.kind)) === JSON.stringify(genA.kinds);
      if (kindsMatch) pass('planner: live sequence JSON valid and matches graph kinds');
      else fail('planner: live sequence JSON valid and matches graph kinds', (genA.sequenceJson ?? '').slice(0, 300));

      // (b) ▶ Play → 규칙 기반 reach가 실제로 box_a를 만진다 (충돌 start) + done
      const runB = await playAndAwaitDone(page, PLANNER_SIM_BUDGET_SEC);
      const historyB = await page.evaluate((limit) => window.__sim.collision.recent(limit), HISTORY_FETCH_LIMIT);
      const armBoxAStarts = historyB.filter((e) => e.phase === 'start' && isPair(e, 'arm', 'box_a'));
      if (armBoxAStarts.length >= 1) {
        pass('planner: Play → arm×box_a collision start (rule-based reach actually touches)',
          `timeSec=[${armBoxAStarts.map((e) => e.timeSec.toFixed(3)).join(', ')}]`);
      } else {
        fail('planner: Play → arm×box_a collision start (rule-based reach actually touches)',
          `history=${JSON.stringify(historyB.slice(-20))}`);
      }
      if (runB.status === 'done' && runB.elapsedSimSec < PLANNER_EVENT_DONE_MAX_SIM_SEC) {
        pass(`planner: sequence done within ${PLANNER_EVENT_DONE_MAX_SIM_SEC}s (barrier released by event)`,
          `elapsed=${runB.elapsedSimSec.toFixed(2)}s`);
      } else {
        fail(`planner: sequence done within ${PLANNER_EVENT_DONE_MAX_SIM_SEC}s (barrier released by event)`,
          JSON.stringify(runB));
      }
      // 결정론 리셋 — 이후 clarify 생성이 깨끗한 씬 상태에서 시작하도록
      await page.evaluate(() => window.__sim.player.stop());

      // (c) generate('박스를 집어') → clarify(2+ 옵션) → box_b 선택 → box_b 시퀀스
      const clarifyC = await page.evaluate(async () => {
        const res = await window.__sim.planner.generate('박스를 집어');
        return { type: res.type, last: window.__sim.planner.lastResult() };
      });
      if (clarifyC.type === 'clarify' && (clarifyC.last?.options?.length ?? 0) >= 2) {
        pass('planner: ambiguous target → clarify with 2+ options', `options=${JSON.stringify(clarifyC.last?.options)}`);
      } else {
        fail('planner: ambiguous target → clarify with 2+ options', JSON.stringify(clarifyC));
      }
      // P1 규약: 선택을 원문에 되붙여(선택 토큰) 재생성 — box_b 확정
      const pickedC = await page.evaluate(async () => {
        const res = await window.__sim.planner.generate('박스를 집어 [선택: box_b]');
        const seq = JSON.parse(window.__sim.flowGraph.sequenceJson());
        return {
          type: res.type,
          betweens: seq.steps
            .filter((s) => s.kind === 'waitForCollision')
            .map((s) => s.between),
        };
      });
      const boxBTargeted = pickedC.type === 'sequence'
        && pickedC.betweens.some((b) => Array.isArray(b) && b.includes('box_b'));
      if (boxBTargeted) pass('planner: clarify pick box_b → sequence targets box_b', JSON.stringify(pickedC.betweens));
      else fail('planner: clarify pick box_b → sequence targets box_b', JSON.stringify(pickedC));

      // (d) 견고성: 없는 대상 → clarify 또는 error (크래시 없음); 무의미 입력 → 읽을 수 있는 error
      //     (facade 결과는 type만 노출 — 읽을 수 있는 사유는 Console 패널에서 확인)
      const robustD = await page.evaluate(async () => {
        const unknown = await window.__sim.planner.generate('없는거 집어');
        const gibberish = await window.__sim.planner.generate('asdf qwer');
        return {
          unknownType: unknown.type,
          gibberishType: gibberish.type,
          gibberishLast: window.__sim.planner.lastResult(),
          consoleText: document.querySelector('[data-testid="console-panel"]')?.textContent ?? '',
        };
      });
      if (robustD.unknownType === 'clarify' || robustD.unknownType === 'error') {
        pass('planner: unknown target → clarify or error (no crash)', `type=${robustD.unknownType}`);
      } else {
        fail('planner: unknown target → clarify or error (no crash)', JSON.stringify(robustD));
      }
      // 무의미 입력은 error이고, 사람이 읽을 수 있는 사유(지원 패턴 안내)가 Console에 남는다
      const readableError = robustD.gibberishType === 'error'
        && robustD.gibberishLast?.type === 'error'
        && robustD.consoleText.includes('지원하는 명령');
      if (readableError) {
        pass('planner: gibberish → error with readable message (Console)');
      } else {
        fail('planner: gibberish → error with readable message (Console)', JSON.stringify(robustD));
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
