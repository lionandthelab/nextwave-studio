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
  orchestration: 'arm-and-boxes', // Phase 10 실행 오케스트레이션 — arm-touch-box 시퀀스 위에서 트라이페인 동기 검증
  'two-arms': 'two-arms-collision', // 로봇↔로봇 충돌 회귀 — 두 팔이 중앙에서 접촉
  'viewport-edit': 'arm-and-boxes', // 뷰포트 편집 UX — 바닥 하한·방향키 이동·선택 HUD
  'conveyor-pick-place': 'conveyor-pick-place', // 컨베이어 라인 — 이송·재순환·포토아이 픽
  'mesh-import': 'arm-and-boxes', // 3D 파일 임포트 — 바닥+로봇이 있는 씬 위에서 검증
  'l-line-cell': 'l-line-cell', // ㄱ자 라인 — 코너 이송 + 로봇 3종 스테이션
  'robot-library': 'arm-and-boxes', // 라이브러리 로봇 3종 — 로드·관절 구동·그리퍼
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
const PNP_STEP_COUNT = 11;
const PNP_SIM_TIME_BUDGET_SEC = 22;
// 명목 길이 ≈ 0.4+2.0+0.8+0+배리어+0.5+0.7+3.0+0.6+0.4+2.0 ≈ 10.4s(이벤트 해제) /
// ≈16.4s(배리어 timeout 6s 경로). 실측 10.55~10.63s — 13s면 두 경로를 확실히 가른다.
const PNP_EVENT_DONE_MAX_SIM_SEC = 13;
/**
 * cargo가 "들렸다"고 볼 최소 상승량 (m). 바닥 정착 y=0.0237 기준 +2 cm.
 * 밀기(끌기)만 하면 y는 정착 높이에서 거의 변하지 않으므로 이 값이 둘을 가른다.
 * 실측 최고점 0.0658 (= +4.2 cm)로 2배 이상 여유가 있다.
 */
const PNP_MIN_LIFT_M = 0.02;
/**
 * 놓기(gripper open) step의 인덱스 — 시퀀스 JSON 순서와 일치해야 한다.
 * [0]open [1]approach [2]lower [3]setJoints [4]barrier [5]grip [6]lift [7]transport
 * [8]lower [9]★release [10]home
 */
const PNP_RELEASE_STEP_INDEX = 9;
/**
 * 놓는 순간 상자와 그리퍼 사이의 최대 수평 거리 (m).
 *
 * ★ 사용자 보고 회귀: "로봇팔이 내려놓는 곳이 아니라 중간에 미끄러져서 내려오고 거기가
 * 드랍존으로 되어 있어서 이상해." 상자가 이송 중 손에서 빠져 굴러가도, 감지 존이 그
 * 자리에 있으면 sensor·상승 어서션은 **모두 통과한다**. "로봇이 놓았다"와 "떨어진 자리에
 * 존이 있다"를 가르는 유일한 신호가 이 거리다. 실측 0.041 m (손가락 폭 수준).
 */
const PNP_MAX_RELEASE_GAP_M = 0.08;
/**
 * 선반 접촉 프로브 관절값 — 시퀀스가 하지 않는 **과잉 스윙**으로 팔을 drop_shelf
 * (z ≈ -0.25)까지 밀어 넣는다. 시퀀스의 정상 목표(joint1 0.3)보다 크게 돌린다.
 */
const PNP_SHELF_PROBE_JOINTS = { joint1: 0.62, joint2: 0.638, joint3: 1.63, joint5: 0.873 };
const PNP_SHELF_PROBE_WAIT_MS = 1500;
const PNP_SHELF_PROBE_SETTLE_MS = 300;       // stop→play 후 물리가 다시 돌기 시작할 여유
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

// ── conveyor-pick-place 게이트 상수 (컨베이어 라인) ─────────────────
// 시퀀스 9 step: gripper·moveJoints·waitForCollision(포토아이)·moveJoints·
// waitForCollision(픽)·gripper·moveJoints(스윕)·gripper·moveJoints(home).
const CPP_STEP_COUNT = 33;             // phase6-scenes.test.ts CPP_SEQUENCE_STEP_COUNT와 일치
const CPP_CYCLE_ITEMS = ['item_a', 'item_b', 'item_c']; // 사이클 순서 = 라인 선입선출 순서
const CPP_SIM_TIME_BUDGET_SEC = 60;    // 3사이클 실측 ~25s + 포토아이 대기(최대 한 바퀴 4.4s)×3
/**
 * 놓는 순간 그리퍼–상자 수평 거리 상한 (m).
 *
 * ★ 사용자 보고: "컨베이 픽앤플레이스는 여전히 가다가 떨어뜨린다." 최종 위치만 보면
 * 이송 중 떨어뜨린 상자와 제대로 놓은 상자를 구분할 수 없다 — **둘 다 바닥에서 끝난다**.
 * 놓기 step의 첫 tick에 상자가 아직 손 안에 있는지를 사이클마다 재야 회귀가 잡힌다.
 * 실측: 3사이클 0.044 / 0.091 / 0.053 m. 상한은 여유를 두되 "이송 호 절반"보다 작게.
 */
const CPP_MAX_RELEASE_GAP_M = 0.12;
/** 놓기(gripper open) step 인덱스 — 사이클당 1개 (시퀀스 구조와 함께 갱신) */
const CPP_RELEASE_STEP_INDEXES = [9, 19, 29];
const CPP_BELT_OBSERVE_SEC = 3;        // Play 전 벨트만 도는 것을 관측하는 시간
const CPP_MIN_BELT_TRAVEL_M = 0.15;    // 위 시간 동안 최소 이송 거리 (0.1 m/s × 3s = 0.3m)
const CPP_BELT_START_X = 0.12;         // 벨트 진행축 시작 x (center 0.34 − half 0.22)
const CPP_BELT_END_X = 0.56;           // 벨트 진행축 끝 x
const CPP_RECYCLE_OBSERVE_SEC = 12;    // 재순환을 확인하기 위해 관측하는 시간
const CPP_EDITED_SPEED_MPS = 0.25;     // 편집 검증용 증속 (기본 0.1보다 확실히 크게)
const CPP_EDIT_OBSERVE_SEC = 1;        // 편집 후 속도가 벨트 값으로 수렴할 여유
const CPP_SPEED_TOLERANCE_MPS = 0.05;  // 벨트 지정 속도 대비 허용 오차
const CPP_MIN_EVENT_SIM_SEC = 0.5;     // 이 시각 이전 이벤트는 "스폰 겹침"으로 보고 무시
/** 벨트 한가운데 착좌 지점 (belt center [0.34, 0.015, 0.143], 상면 0.03 + item half 0.025) */
const CPP_BELT_SEAT_POSITION = [0.34, 0.055, 0.143];
/** 벨트 밖 대기 자리 — 재사용한 상자를 치워 다음 측정과 부딪히지 않게 한다 */
const CPP_BELT_PARK_POSITION = [0.34, 0.03, -0.35];
/**
 * 역방향 측정용 좌석 — 진행 방향 **반대쪽 끝**에 앉힌다.
 *
 * 런웨이는 0.44 m뿐이고 증속 벨트는 1s에 0.25 m를 간다. 가운데에 앉히면 관측 창 안에
 * 반대쪽 끝을 넘어가 버려(0.34 → 0.09 < 벨트 시작 0.12) 벨트를 벗어난 상태로 측정된다
 * — 실측 onBelt=false. 좌석은 측정하려는 방향에 맞춰 잡아야 한다.
 */
const CPP_BELT_REVERSE_SEAT_POSITION = [0.50, 0.055, 0.143];

// ── viewport-edit 게이트 상수 (바닥 하한 · 방향키 이동 · 선택 HUD) ──
const VE_TARGET_ID = 'box_a';                // arm-and-boxes의 동적 박스
const VE_NUDGE_MIN_DELTA_M = 0.02;           // 기본 nudge 0.05m — 여유를 둔 하한
const VE_NUDGE_FINE_M = 0.01;                // NUDGE_FINE_STEP_M (render/interaction.ts)
const VE_NUDGE_TOLERANCE_M = 0.002;          // 스냅/부동소수 여유
const VE_SINK_PRESS_COUNT = 12;              // PageDown 반복 (클램프 없으면 -0.6m)
const VE_LIFT_PRESS_COUNT = 6;               // End 검증용으로 띄우는 횟수
const VE_GROUNDED_TOLERANCE_M = 0.005;       // 치수 변경 후 접지 판정 여유
const VE_STEP_DEADLINE_MS = 5000;            // Step 1회의 상태 전이 폴링 실시간 상한

// ── mesh-import 게이트 상수 (3D 파일 임포트 — UX_DESIGN §4.4) ──────
//
// 계측 픽스처는 **세 변이 모두 다르고 원점에서 어긋난** 직육면체다(scripts/make-import-fixtures.mjs).
// 정육면체를 쓰면 스케일은 재도 Up-axis 회전(y↔z 교환)을 재지 못해, 임포트가 upAxis를
// 통째로 무시해도 게이트가 초록이 된다. 원점 중심이면 피벗 재정렬의 x/z 성분이 항등이라
// 검증되지 않는다. 다운로드 에셋(avocado 등)은 치수를 우리가 통제하지 않으므로 수치
// 어서션에 쓰지 않고 **카탈로그 스모크**로만 돌린다.
const IMP_MODEL_DIR = '/assets/models/';
const IMP_FIXTURES = [
  { format: 'glb', file: 'gate-box.glb', label: 'glTF (.glb)' },
  { format: 'stl', file: 'gate-box.stl', label: 'STL (.stl)' },
  { format: 'obj', file: 'gate-box.obj', label: 'OBJ (.obj)' },
];
/** 배포용 다운로드 모델 — 파싱만 확인한다(치수는 외부 소유) */
const IMP_CATALOG = ['avocado.glb', 'water-bottle.glb', 'teacup.stl', 'boombox.stl', 'barramundi-fish.obj'];
const IMP_FIXTURE_SIZE_LABEL = '0.300 × 0.200 × 0.100 m'; // formatBboxSizeM(scale=1)
const IMP_FIXTURE_HALF_SIZE_LABEL = '0.150 × 0.100 × 0.050 m'; // scale=0.5
const IMP_FIXTURE_TRIANGLES = 12;            // 박스 = 12 삼각형 (3종 동일해야 한다)
const IMP_FIXTURE_HALF_EXTENTS = [0.15, 0.1, 0.05]; // scale=1, y-up일 때 AABB half
const IMP_SCALE_HALF = 0.5;
const IMP_HALF_TOLERANCE_M = 1e-3;
const IMP_RATIO_TOLERANCE = 1e-3;
const IMP_SETTLE_TOLERANCE_M = 0.01;         // 정착 원점 y ≈ 0 (피벗 = bbox 바닥 중심)
const IMP_MAX_SINK_M = 0.01;                 // 이보다 깊으면 접촉 침투가 아니라 지하
const IMP_VISUAL_TOLERANCE_M = 0.02;         // anchorProbe visualCenter 판정
const IMP_DROP_HEIGHT_M = 0.4;               // 낙하 관측 시작 높이
const IMP_MIN_FALL_M = 0.2;                  // "실제로 떨어졌다" 판정
const IMP_TRIMESH_SUPPORT_MIN_Y_M = 0.15;    // trimesh(윗면 0.20) 위에 얹힌 판정
const IMP_PARSE_DEADLINE_MS = 15000;
const IMP_CONFIRM_DEADLINE_MS = 8000;
const IMP_SETTLE_DEADLINE_MS = 20000;
/** 임포트 엔티티 id — placeEntity는 uniquify하지 않으므로 케이스마다 달라야 한다 */
const IMP_ID = {
  hullY: 'imp-hull-y',
  hullZ: 'imp-hull-z',
  aabb: 'imp-aabb',
  aabbHalf: 'imp-aabb-half',
  aabbZ: 'imp-aabb-z',
  trimesh: 'imp-trimesh',
};
/**
 * 임포트 엔티티 주차 자리 (x, z).
 *
 * 임포트는 전부 "뷰포트 중앙"에 떨어지므로 그대로 두면 여러 개가 한 자리에 쌓여
 * 서로 부딪힌다 — 그러면 각 어서션이 자기 대상이 아니라 **더미**를 재게 된다
 * (실측: 서로 다른 두 대상의 정착 y가 소수점 4자리까지 같았다). 로봇 작업 반경
 * (약 0.5m) 밖에, 서로 0.6m 이상 떨어뜨린다.
 */
const IMP_PARK_Z = -1.2;
const IMP_PARK_X0 = 1.0;
const IMP_PARK_DX = 0.6;

const IMP_BAD_FILE = { name: 'not-a-model.txt', body: 'hello workcell' };
const IMP_CORRUPT_FILE = { name: 'corrupt.glb', body: 'not a glb at all' };

// ── robot-library 게이트 상수 (라이브러리 로봇 3종) ────────────────
//
// arm6(6축 관절팔·평행 2지) 하나뿐이던 것을 손 모양이 서로 다른 3종으로 넓혔다.
// 이 게이트가 묻는 것: 각 로봇이 **실제로 서고**(링크 바디 생성), **관절이 말단을 움직이고**,
// **그리퍼가 손을 실제로 여닫는가**. URDF가 파싱만 되고 구동되지 않는 회귀를 잡는다.
const RL_ROBOTS = [
  {
    key: 'arm-6',
    idBase: 'arm',
    minLinks: 8,                         // 6 revolute + 2 prismatic finger
    driveJoint: 'joint2',
    driveDelta: 0.6,
    gripperJoints: ['finger_left_joint', 'finger_right_joint'],
    gripperOpen: 0.03,
    gripperClose: 0.0,
    handKind: '평행 2지',
  },
  {
    key: 'scara-4',
    idBase: 'scara',
    minLinks: 6,                         // base + 4 + suction pad
    driveJoint: 'joint2',
    driveDelta: 0.7,
    gripperJoints: ['suction_joint'],
    gripperOpen: 0.0,
    gripperClose: 0.012,
    handKind: '흡착 패드',
  },
  {
    key: 'cobot-7',
    idBase: 'cobot',
    minLinks: 11,                        // base + 7 + finger ×3
    driveJoint: 'joint2',
    driveDelta: 0.5,
    gripperJoints: ['finger_a_joint', 'finger_b_joint', 'finger_c_joint'],
    gripperOpen: -0.65,
    gripperClose: 0.05,
    handKind: '3지 클로',
  },
];
/** 로봇을 놓는 자리 — 서로/기존 로봇과 겹치지 않게 (arm-and-boxes 로봇은 원점) */
const RL_SPOT_Z = -1.4;
const RL_SPOT_X0 = 0.8;
const RL_SPOT_DX = 0.9;
/** 관절을 움직였을 때 말단이 이만큼은 움직여야 "구동된다"고 본다 (m) */
const RL_MIN_END_MOVE_M = 0.02;
/**
 * 그리퍼 여닫을 때 손 링크가 이만큼은 움직여야 한다.
 *
 * 위치와 **자세를 모두** 본다: 평행 2지(arm6)·흡착 패드(scara)는 prismatic이라 링크
 * 원점이 이동하지만, 3지 클로(cobot7)는 revolute라 링크 원점이 **회전축 위에 있어
 * 위치가 전혀 변하지 않는다** — 자세만 바뀐다. 위치만 재면 정상 동작하는 클로를
 * "손이 안 움직인다"고 오판한다(실측으로 겪었다).
 */
const RL_MIN_HAND_DELTA_M = 0.005;
const RL_MIN_HAND_ROT_RAD = 0.1;
/** home 포즈에서 링크가 바닥 아래로 내려가도 되는 허용치 (m) — 로봇 링크는 kinematic이라 자가 교정이 없다 */
const RL_MAX_UNDERGROUND_M = 0.005;
const RL_PLACE_DEADLINE_MS = 20000;

// ── l-line-cell 게이트 상수 (ㄱ자 라인 + 로봇 3종) ─────────────────
//
// 이 씬이 묻는 것: **세 로봇이 각자 실제로 일했는가**. "시퀀스가 done으로 끝났다"만으로는
// 로봇이 허공에서 춤춰도 통과한다 — 각 스테이션마다 그 로봇이 그 상자를 만졌다는
// 접촉 이벤트와, 상자가 실제로 옮겨졌다는 좌표를 함께 본다.
const LL_STEP_COUNT = 38;
const LL_SIM_BUDGET_SEC = 120;      // 실측 완주 32.2s + 여유 (벨트 실효속도가 선언값의 69~78%)
/** 스테이션별 (로봇, 대상 상자) — 이 쌍의 접촉이 없으면 그 로봇은 일하지 않은 것이다 */
const LL_STATIONS = [
  { robot: 'press', item: 'item_b', role: '검사 프레스' },
  { robot: 'picker', item: 'item_a', role: '라인 피킹' },
  { robot: 'palletizer', item: 'item_a', role: '팔레타이징' },
];
const LL_ROBOTS = ['press', 'picker', 'palletizer'];
/** 로봇이 절대 닿으면 안 되는 정적물 — 닿으면 셀 배치가 틀린 것이다 */
const LL_STATICS = ['belt_in', 'belt_out', 'rail_outer', 'rail_inner'];
/**
 * 판정은 **중심 좌표 포함**으로 한다 — 센서 이벤트로는 안 된다.
 * 인계 패드와 팔레트는 10cm 거리라 5cm 상자가 두 존을 동시에 발화시킬 수 있고,
 * 실제로 처음 배치(간극 0)에서 그랬다: picker가 인계하는 순간 zone_pallet도 함께 start해
 * "팔레트 감지"가 적재의 증거가 되지 못했다(검토에서 잡힌 결함).
 */
const LL_FINAL_ITEM = 'item_a';
const LL_PLACE_STEP_INDEXES = [21, 34];  // picker 인계 / palletizer 적재
const LL_MAX_RELEASE_GAP_M = 0.08;       // 실측 0.021 / 0.033

// ── Phase 10: orchestration 게이트 상수 (arm-and-boxes + arm-touch-box, 7 step) ──
const ORCH_NODE_COUNT = 7;                         // arm-touch-box.sequence.json step 수
const ORCH_SIM_BUDGET_SEC = 12;                    // Play→done sim 예산 (배리어 이벤트 해제 경로)
const ORCH_REALTIME_DEADLINE_MS = 45000;           // 폴링 실시간 상한 (fast-forward/행 방지)
const ORCH_STEP_MAX_SIM_SEC = 4;                   // stepNode 1개 노드의 sim 상한 (전체 ~8s의 일부)

/**
 * 3D 뷰포트 슬롯에 포커스를 준다.
 *
 * 뷰포트 편집 단축키(W/E/R · 방향키 · End)는 `scope: 'viewport'` 바인딩이라, 라우터가
 * **활성 요소**에서 스코프를 거슬러 찾을 때 뷰포트 안에 포커스가 있어야 선택된다
 * (불변식 §2.10). 실제 사용자는 3D 화면을 클릭하며 자연히 포커스를 얻지만
 * (workspace.ts의 pointerdown 훅), 파사드 select()에는 그 부수효과가 없다.
 */
async function focusViewportSlot(page) {
  await page.evaluate(() => {
    document.querySelector('[data-testid="workspace-viewport"]')?.focus();
  });
}

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

// ── mesh-import 헬퍼 ────────────────────────────────────────────────

/**
 * 픽스처를 fetch해 File로 만들고 임포트 다이얼로그를 연다 — 라이브러리 ⬆ / 뷰포트 드롭과
 * **동일 진입점**(__sim.meshImport.open). 파사드가 폼을 우회해 EntitySpec을 직접 조립하면
 * "다이얼로그는 망가졌는데 게이트는 초록"이 되므로, 여는 것만 파사드로 하고 나머지는
 * 사람과 같이 DOM을 조작한다.
 */
async function openImportFixture(page, fileName) {
  await page.evaluate(
    async ({ dir, name }) => {
      const res = await fetch(dir + name);
      if (!res.ok) throw new Error(`fixture fetch failed: ${dir}${name} → ${res.status}`);
      window.__sim.meshImport.open(new File([await res.arrayBuffer()], name));
    },
    { dir: IMP_MODEL_DIR, name: fileName },
  );
  return awaitImportPhase(page);
}

/** 임의 텍스트를 파일로 만들어 다이얼로그를 연다 (실패 경로 검증용) */
async function openImportText(page, name, body) {
  await page.evaluate(({ n, b }) => {
    window.__sim.meshImport.open(new File([b], n));
  }, { n: name, b: body });
  return awaitImportPhase(page);
}

/** '분석 중…'을 벗어날 때까지 폴링 → 리드아웃/폼 상태 스냅샷 */
async function awaitImportPhase(page) {
  await page.waitForFunction(
    () => {
      const dlg = document.querySelector('[data-testid="import-dialog"]');
      if (!dlg || getComputedStyle(dlg).display === 'none') return false;
      const fmt = document.querySelector('[data-testid="import-format"]')?.textContent ?? '';
      return fmt !== '' && !fmt.includes('분석 중');
    },
    undefined,
    { timeout: IMP_PARSE_DEADLINE_MS },
  );
  return page.evaluate(() => {
    const visible = (el) => el !== null && getComputedStyle(el).display !== 'none';
    const err = document.querySelector('[data-testid="import-error"]');
    return {
      format: document.querySelector('[data-testid="import-format"]')?.textContent ?? null,
      triangles: document.querySelector('[data-testid="import-triangles"]')?.textContent ?? null,
      size: document.querySelector('[data-testid="import-size"]')?.textContent ?? null,
      errorVisible: visible(err),
      errorText: err?.textContent ?? null,
      confirmDisabled: document.querySelector('[data-testid="import-confirm"]')?.disabled ?? null,
      objectKindDisabled: document.querySelector('[data-testid="import-kind-object"]')?.disabled ?? null,
      trimeshNoteVisible: visible(document.querySelector('[data-testid="import-trimesh-note"]')),
    };
  });
}

/**
 * 폼을 채운다. trimesh 전략은 유형을 Object로 되돌릴 수 없으므로(강제 Static — 버튼
 * disabled) kind를 넘기지 않는다. disabled 버튼에 click하면 enabled를 기다리다 타임아웃한다.
 */
async function fillImportForm(page, { id, scale, upAxis, collider, kind }) {
  await page.fill('[data-testid="import-id"]', id);
  await page.fill('[data-testid="import-scale"]', String(scale));
  await page.click(`[data-testid="import-upaxis-${upAxis}"]`);
  await page.click(`[data-testid="import-collider-${collider}"]`);
  if (kind !== undefined) await page.click(`[data-testid="import-kind-${kind}"]`);
}

/** [추가] 확정 — 다이얼로그가 닫히고 엔티티가 1개 늘 때까지 기다린다 */
async function confirmImport(page) {
  const before = await page.evaluate(() => window.__sim.editor.entityIds().length);
  await page.click('[data-testid="import-confirm"]');
  await page.waitForFunction(
    (n) => {
      const dlg = document.querySelector('[data-testid="import-dialog"]');
      const closed = dlg === null || getComputedStyle(dlg).display === 'none';
      return closed && window.__sim.editor.entityIds().length === n + 1;
    },
    before,
    { timeout: IMP_CONFIRM_DEADLINE_MS },
  );
}

async function closeImportDialog(page) {
  await page.click('[data-testid="import-cancel"]');
  await page.waitForFunction(() => {
    const dlg = document.querySelector('[data-testid="import-dialog"]');
    return dlg === null || getComputedStyle(dlg).display === 'none';
  }, undefined, { timeout: IMP_CONFIRM_DEADLINE_MS });
}

/** 편집 스펙의 엔티티 1건 (없으면 null) */
function importedSpec(page, id) {
  return page.evaluate(
    (i) => window.__sim.editor.serialize().entities.find((e) => e.id === i) ?? null,
    id,
  );
}

/** 임포트 엔티티를 제 자리로 옮긴다 (다른 임포트와 겹치지 않게 — IMP_PARK_* 주석 참조) */
function parkImported(page, id, slot) {
  return page.evaluate(
    ({ i, x, z }) => window.__sim.editor.updateTransform(i, { position: [x, 0, z] }),
    { i: id, x: IMP_PARK_X0 + slot * IMP_PARK_DX, z: IMP_PARK_Z },
  );
}

/** 임포트 → 폼 → 확정 한 번에 (계측 픽스처 전용) */
async function importFixture(page, file, form) {
  const phase = await openImportFixture(page, file);
  await fillImportForm(page, form);
  await confirmImport(page);
  return phase;
}

/** 엔티티를 들어올려 낙하시키고 정착 y를 잰다 (물리 바디가 실재하는지의 증거) */
async function dropAndSettle(page, id, dropY) {
  await page.evaluate(
    ({ i, y }) => {
      const p = window.__sim.editor.serialize().entities.find((e) => e.id === i).transform.position;
      window.__sim.editor.updateTransform(i, { position: [p[0], y, p[2]] });
      window.__sim.engine.play();
    },
    { i: id, y: dropY },
  );
  const readY = () =>
    page.evaluate((i) => {
      const b = window.__sim.world.bodiesOfEntity(i)[0];
      return b === undefined ? null : window.__sim.world.getPose(b).position[1];
    }, id);
  const deadline = Date.now() + IMP_SETTLE_DEADLINE_MS;
  let previous = await readY();
  let stable = 0;
  for (;;) {
    await page.waitForTimeout(200);
    const y = await readY();
    if (y === null) return { settledY: null, fell: 0 };
    if (Math.abs(y - previous) < 1e-4) stable += 1;
    else stable = 0;
    previous = y;
    if (stable >= 3 || Date.now() > deadline) break;
  }
  return { settledY: previous, fell: dropY - previous };
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

    // 이 게이트는 **스튜디오(로컬 모드)** 를 검증한다 — 협업 서버는 대상이 아니다.
    // vite preview는 vite.config.ts의 프록시를 그대로 쓰므로, 개발자가 백엔드를 띄워 둔
    // 상태면 앱이 서버 모드로 부팅해 **로그인 화면이 캔버스를 덮어** 모든 클릭이 막힌다.
    // 게이트 결과가 "주변에 서버가 떠 있었는가"에 좌우되면 안 되므로, 여기서 API를 명시적으로
    // 끊어 로컬 모드를 강제한다(BACKEND §1의 서버 없는 경로 = 이 게이트의 검증 대상).
    await page.route('**/api/**', (route) => route.abort());

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
        // 재생 내내 cargo의 y를 표본해 **실제로 떠올랐는지** 잰다. 최종 pose만 보면
        // 끌고 간 것과 들고 간 것을 구분할 수 없다(둘 다 바닥에서 끝난다).
        await page.evaluate((releaseStep) => {
          window.__cargoMaxY = -Infinity;
          window.__releaseGap = null;
          window.__sim.engine.onTick(() => {
            const w = window.__sim.world;
            const body = w.bodiesOfEntity('cargo')[0];
            if (body === undefined) return;
            const cargo = w.getPose(body).position;
            if (cargo[1] > window.__cargoMaxY) window.__cargoMaxY = cargo[1];
            // 놓기 step의 **첫 tick**에 상자가 아직 그리퍼에 있는지 (베이스에서 가장 먼
            // 링크를 그리퍼로 본다 — anchorProbe와 같은 관례)
            if (window.__releaseGap !== null) return;
            if (window.__sim.player?.currentStepIndex !== releaseStep) return;
            let far = null;
            let maxR = -1;
            for (const bid of w.bodiesOfEntity('arm')) {
              const q = w.getPose(bid).position;
              const r = Math.hypot(q[0], q[2]);
              if (r > maxR) { maxR = r; far = q; }
            }
            if (far === null) return;
            window.__releaseGap = Math.hypot(cargo[0] - far[0], cargo[2] - far[2]);
          });
        }, PNP_RELEASE_STEP_INDEX);
        const restingY = await page.evaluate(() => {
          const w = window.__sim.world;
          const body = w.bodiesOfEntity('cargo')[0];
          return body === undefined ? null : w.getPose(body).position[1];
        });
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

        // 3-b) ★ 사용자 요청 회귀: 상자를 **집어 올려서** 옮긴다 (끌고 가지 않는다).
        //      바닥 정착 높이 대비 최고점이 유의미하게 높아야 한다.
        const cargoMaxY = await page.evaluate(() => window.__cargoMaxY);
        const lift = restingY === null || cargoMaxY === null ? null : cargoMaxY - restingY;
        if (lift !== null && lift >= PNP_MIN_LIFT_M) {
          pass('pick-and-place: 상자를 집어 올려서 옮긴다 ★ (끌기 아님)',
            `상승 ${(lift * 100).toFixed(1)}cm (정착 y=${restingY?.toFixed(4)} → 최고 ${cargoMaxY?.toFixed(4)})`);
        } else {
          fail('pick-and-place: 상자를 집어 올려서 옮긴다 ★ (끌기 아님)',
            `상승 ${lift === null ? 'n/a' : (lift * 100).toFixed(1) + 'cm'} < ${PNP_MIN_LIFT_M * 100}cm — 그리퍼가 놓쳤을 수 있다`);
        }

        // 3-c) ★ 사용자 보고 회귀: **로봇이 놓은 것**이지, 미끄러져 떨어진 자리에 존이
        //      있는 것이 아니다. 놓는 순간 상자가 아직 그리퍼에 물려 있어야 한다.
        const releaseGap = await page.evaluate(() => window.__releaseGap);
        if (releaseGap !== null && releaseGap <= PNP_MAX_RELEASE_GAP_M) {
          pass('pick-and-place: 놓는 순간 상자가 아직 그리퍼에 있다 ★ (미끄러져 떨어진 것 아님)',
            `그리퍼-상자 거리 ${(releaseGap * 100).toFixed(1)}cm ≤ ${PNP_MAX_RELEASE_GAP_M * 100}cm`);
        } else {
          fail('pick-and-place: 놓는 순간 상자가 아직 그리퍼에 있다 ★ (미끄러져 떨어진 것 아님)',
            releaseGap === null
              ? '놓기 step을 관측하지 못했다 (PNP_RELEASE_STEP_INDEX가 시퀀스와 어긋났을 수 있다)'
              : `거리 ${(releaseGap * 100).toFixed(1)}cm — 이송 중 손에서 빠졌다`);
        }

        // 4) 예제는 선반을 스치지 않는다 — 샘플이 매 실행 충돌을 보고하면 "정상"의
        //    기준선이 무너져 진짜 사고가 소음에 묻힌다
        const shelfHitsInSample = history.filter(
          (e) => e.phase === 'start' && isPair(e, 'arm', 'drop_shelf'),
        );
        if (shelfHitsInSample.length === 0) {
          pass('pick-and-place: 정상 시퀀스는 선반을 건드리지 않는다 (arm×drop_shelf 0건)');
        } else {
          fail('pick-and-place: 정상 시퀀스는 선반을 건드리지 않는다 (arm×drop_shelf 0건)',
            `hits=${JSON.stringify(shelfHitsInSample.map((e) => e.timeSec.toFixed(3)))}`);
        }

        // 5) ★ 사용자 보고 회귀: 선반은 **실체가 있고 로봇과 쌍이 성립**해야 한다.
        //    구 구현은 선반이 sensor + collidesWith [OBJECT]뿐이라, 팔이 선반을 지나가도
        //    이벤트가 0건이었다(= "충돌 표시 안 뜨고 관통"). 일부러 선반으로 팔을 돌려
        //    접촉이 실제로 보고되는지 확인한다.
        //    시퀀스 완주 후 엔진은 일시정지 상태다 — 접촉은 world.step()에서만 생기므로
        //    stop()으로 씬(과 충돌 이력)을 리셋한 뒤 **물리를 재개**하고 관절을 밀어 넣는다.
        await page.evaluate(() => {
          window.__sim.player?.stop();
          window.__sim.engine.play();
        });
        await page.waitForTimeout(PNP_SHELF_PROBE_SETTLE_MS);
        await page.evaluate((joints) => {
          for (const [name, value] of Object.entries(joints)) {
            window.__sim.robots.setJoint('arm', name, value);
          }
        }, PNP_SHELF_PROBE_JOINTS);
        await page.waitForTimeout(PNP_SHELF_PROBE_WAIT_MS);
        const afterProbe = await page.evaluate(
          (limit) => window.__sim.collision.recent(limit),
          HISTORY_FETCH_LIMIT,
        );
        const shelfHits = afterProbe.filter(
          (e) => e.phase === 'start' && e.kind === 'contact' && isPair(e, 'arm', 'drop_shelf'),
        );
        if (shelfHits.length >= 1) {
          pass('pick-and-place: 선반에 닿으면 충돌로 보고된다 ★ 회귀 (구: sensor라 이벤트 0건)',
            `starts=${shelfHits.length}`);
        } else {
          fail('pick-and-place: 선반에 닿으면 충돌로 보고된다 ★ 회귀 (구: sensor라 이벤트 0건)',
            `pairs=${JSON.stringify([...new Set(afterProbe.map((e) => `${e.a}×${e.b}`))])}`);
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

      // (f) '{} JSON' 패널에 삽입된 wait step 반영 (그래프 편집 → JSON 실시간 동기).
      // 패널이 편집 가능해진 뒤로 진실은 textarea의 **value**다(구 읽기 전용 <pre>는
      // 편집 모드에서 숨겨진다) — 편집 불가 폴백을 위해 <pre>도 함께 본다.
      await page.click('[data-testid="json-toggle"]');
      await page.click('[data-testid="json-tab-json"]').catch(() => {}); // 탭은 사용자 상태로 유지된다
      const viewerText = await page.evaluate(() => {
        const editor = document.querySelector('[data-testid="json-editor"]');
        if (editor instanceof HTMLTextAreaElement && editor.value !== '') return editor.value;
        return document.querySelector('[data-testid="json-content"]')?.textContent ?? '';
      });
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

    // ── Phase 10: orchestration — 노드 단위 실행 · 트라이페인 동기 · 재실행 (UX_DESIGN §5) ──
    if (expectArg === 'orchestration') {
      const orchStop = () => page.evaluate(() => window.__sim.orchestrator.stop());

      // (a) 초기: 전 노드 pending · 활성 노드 없음 · player idle(무자동재생) · 오버레이 Idle
      const init = await page.evaluate(() => {
        const o = window.__sim.orchestrator;
        const statuses = o.statuses();
        return {
          vals: Object.values(statuses),
          count: Object.keys(statuses).length,
          activeNodeId: o.activeNodeId(),
          overlayText: o.overlayText(),
          playerStatus: window.__sim.player.status,
        };
      });
      const allPending = init.count === ORCH_NODE_COUNT && init.vals.every((s) => s === 'pending');
      if (allPending && init.activeNodeId === null && init.playerStatus === 'idle'
          && init.overlayText.includes('Idle')) {
        pass('orchestration: initial all-pending, no active node, player idle, overlay Idle',
          `overlay="${init.overlayText}"`);
      } else {
        fail('orchestration: initial all-pending, no active node, player idle, overlay Idle',
          JSON.stringify(init));
      }

      // (b/c) Play → 상태 진행 + 충돌 + activeNodeId 추적 + 오버레이 running + 트라이페인 일관
      const startSim = await page.evaluate(() => {
        window.__sim.orchestrator.play();
        return window.__sim.engine.simTimeSec;
      });
      const realDeadline = Date.now() + ORCH_REALTIME_DEADLINE_MS;
      let sawActive = false, sawDone = false, runningOverlay = null, triPane = null, last = null;
      for (;;) {
        last = await page.evaluate(() => {
          const o = window.__sim.orchestrator;
          const vals = Object.values(o.statuses());
          return {
            status: window.__sim.player.status,
            simTimeSec: window.__sim.engine.simTimeSec,
            anyActive: vals.includes('active'),
            anyDone: vals.includes('done'),
            overlayText: o.overlayText(),
            activeNodeId: o.activeNodeId(),
          };
        });
        if (last.anyActive) sawActive = true;
        if (last.anyDone) sawDone = true;
        // 재생 중 관측되는 running 오버레이를 매 폴에서 최신값으로 잡는다 — 한 프레임 지연에
        // 영구 실패하지 않도록(제품 수정으로 Play 순간부터 이미 'Running'이지만 방어적으로 갱신).
        if (last.status === 'running' && last.anyActive) {
          runningOverlay = last.overlayText;
        }
        // 트라이페인 스냅샷 — 활성 노드가 있는 한 순간에 그래프/오버레이/타임라인/facade 정합 확인
        if (triPane === null && last.activeNodeId !== null && last.anyActive) {
          triPane = await page.evaluate(() => {
            const o = window.__sim.orchestrator;
            const activeNodeId = o.activeNodeId();
            const statuses = o.statuses();
            const statusActiveId = Object.keys(statuses).find((k) => statuses[k] === 'active') ?? null;
            const nodeIds = window.__sim.flowGraph.nodeIds();
            const markers = [...document.querySelectorAll('[data-testid="timeline-marker"]')];
            const tlIdx = markers.findIndex((m) => m.getAttribute('aria-current') === 'step');
            const el = document.querySelector(`[data-fg-node="${activeNodeId}"]`);
            // 뷰포트 오버레이 'node k/n'을 파싱해 노드 id로 되돌린다 — 오버레이를 트라이페인
            // 등식의 독립 항으로 넣어 "graph == overlay == timeline == facade"를 실제로 검증한다.
            const om = o.overlayText().match(/node (\d+)\/(\d+)/);
            const overlayNodeId = om ? (nodeIds[Number(om[1]) - 1] ?? null) : null;
            return {
              activeNodeId,
              statusActiveId,
              tlNodeId: tlIdx >= 0 ? (nodeIds[tlIdx] ?? null) : null,
              overlayNodeId,
              hasSelected: el ? el.classList.contains('rsw-fg-node--selected') : false,
              hasActiveDot: el ? el.querySelector('.rsw-fg-dot--active') !== null : false,
            };
          });
        }
        if (last.status === 'done') break;
        if (last.simTimeSec - startSim > ORCH_SIM_BUDGET_SEC || Date.now() > realDeadline) break;
        await page.waitForTimeout(SEQ_POLL_INTERVAL_MS);
      }

      // (b) 상태 진행 (active → done 관측)
      if (sawActive && sawDone) pass('orchestration: node statuses progress (active then done)');
      else fail('orchestration: node statuses progress (active then done)', `sawActive=${sawActive} sawDone=${sawDone}`);

      // (b) 충돌 arm×box_a start 이력 (배리어 대상 접촉 — 실제 접근)
      const history = await page.evaluate((limit) => window.__sim.collision.recent(limit), HISTORY_FETCH_LIMIT);
      const armBoxA = history.filter((e) => e.phase === 'start' && isPair(e, 'arm', 'box_a'));
      if (armBoxA.length >= 1) {
        pass('orchestration: collision history has arm×box_a start', `timeSec=[${armBoxA.map((e) => e.timeSec.toFixed(3)).join(', ')}]`);
      } else {
        fail('orchestration: collision history has arm×box_a start', `history=${JSON.stringify(history.slice(-20))}`);
      }

      // (b) activeNodeId 추적 + 오버레이 running 지시자 + 캔버스 활성 클래스(DOM)
      if (triPane !== null && (triPane.hasSelected || triPane.hasActiveDot)) {
        pass('orchestration: activeNodeId tracked + canvas active class present',
          `active=${triPane.activeNodeId} selected=${triPane.hasSelected} dot=${triPane.hasActiveDot}`);
      } else {
        fail('orchestration: activeNodeId tracked + canvas active class present', JSON.stringify(triPane));
      }
      if (runningOverlay && runningOverlay.includes('node') && runningOverlay.includes('Running')) {
        pass('orchestration: overlay shows running + node progress', `overlay="${runningOverlay}"`);
      } else {
        fail('orchestration: overlay shows running + node progress', `overlay="${runningOverlay}"`);
      }

      // (c) 트라이페인 일관: facade 활성 == 상태맵 active == 뷰포트 오버레이 node == Timeline
      // 활성 마커가 모두 같은 노드 (네 항 모두 등식에 포함 — 오버레이는 독립 파싱 항이다)
      if (triPane && triPane.activeNodeId !== null
          && triPane.activeNodeId === triPane.statusActiveId
          && triPane.activeNodeId === triPane.overlayNodeId
          && triPane.activeNodeId === triPane.tlNodeId) {
        pass('orchestration: tri-pane consistent (facade == statusmap == overlay node == timeline marker)',
          `node=${triPane.activeNodeId}`);
      } else {
        fail('orchestration: tri-pane consistent (facade == statusmap == overlay node == timeline marker)', JSON.stringify(triPane));
      }

      // (d) 완주 → 전 노드 done · 오버레이 Idle (시퀀스 미실행 상태로 접힘)
      const final = await page.evaluate(() => {
        const o = window.__sim.orchestrator;
        const statuses = o.statuses();
        return {
          vals: Object.values(statuses),
          count: Object.keys(statuses).length,
          overlayText: o.overlayText(),
          playerStatus: window.__sim.player.status,
        };
      });
      const allDone = final.count === ORCH_NODE_COUNT && final.vals.every((s) => s === 'done');
      if (last?.status === 'done' && allDone && final.overlayText.includes('Idle')) {
        pass('orchestration: sequence completes → all done, overlay Idle',
          `elapsed=${(last.simTimeSec - startSim).toFixed(2)}s overlay="${final.overlayText}"`);
      } else {
        fail('orchestration: sequence completes → all done, overlay Idle',
          JSON.stringify({ lastStatus: last?.status, final }));
      }

      // (d-2) 완주 = 실행 종료: 엔진이 서고 simTime이 더 가지 않는다.
      // 회귀 배경: 구 동작은 물리 루프가 계속 돌아 simTime이 영원히 올라갔고, 사용자에게는
      // "끝났는데 안 멈춘다"로 보였다. 씬 상태는 보존한다(⏹ 정지와 달리 리셋하지 않는다).
      const doneT1 = await page.evaluate(() => ({
        state: window.__sim.engine.state,
        simTimeSec: window.__sim.engine.simTimeSec,
      }));
      await page.waitForTimeout(1200);
      const doneT2 = await page.evaluate(() => window.__sim.engine.simTimeSec);
      if (doneT1.state === 'paused' && Math.abs(doneT2 - doneT1.simTimeSec) < 1e-6) {
        pass('orchestration: 완주 후 실행 종료 (엔진 정지 · simTime 고정)',
          `state=${doneT1.state} simTime=${doneT1.simTimeSec.toFixed(3)} (1.2s 뒤 동일)`);
      } else {
        fail('orchestration: 완주 후 실행 종료 (엔진 정지 · simTime 고정)',
          JSON.stringify({ state: doneT1.state, t1: doneT1.simTimeSec, t2: doneT2 }));
      }

      // (d-3) 완주 후 ▶ 재생 = 처음부터 다시 실행.
      // 회귀 배경: armSequenceIfAvailable이 sequenceArmed=true면 조기 반환해서, 완주 상태의
      // ▶는 완전한 no-op이었다 — ⏹ 정지를 눌러야만 다시 재생할 수 있었다.
      await page.evaluate(() => {
        document.querySelector('[data-testid="playback-play"]').click();
      });
      const replayDeadline = Date.now() + ORCH_REALTIME_DEADLINE_MS;
      let replay = null;
      for (;;) {
        replay = await page.evaluate(() => ({
          status: window.__sim.player.status,
          index: window.__sim.player.currentStepIndex,
          state: window.__sim.engine.state,
        }));
        if (replay.status === 'running' && replay.state === 'playing') break;
        if (Date.now() > replayDeadline) break;
        await page.waitForTimeout(SEQ_POLL_INTERVAL_MS);
      }
      if (replay.status === 'running' && replay.state === 'playing') {
        pass('orchestration: 완주 후 ▶ 재생이 처음부터 다시 실행한다',
          `status=${replay.status} index=${replay.index} engine=${replay.state}`);
      } else {
        fail('orchestration: 완주 후 ▶ 재생이 처음부터 다시 실행한다', JSON.stringify(replay));
      }

      // (e) stepNode: 신선한 리셋에서 정확히 노드 1개 전진 (커서 +1, sim은 일부만 전진)
      await orchStop();
      const beforeStep = await page.evaluate(() => ({
        index: window.__sim.player.currentStepIndex,
        simTimeSec: window.__sim.engine.simTimeSec,
      }));
      await page.evaluate(() => window.__sim.orchestrator.stepNode());
      const stepDeadline = Date.now() + ORCH_REALTIME_DEADLINE_MS;
      let stepped = null;
      for (;;) {
        stepped = await page.evaluate(() => ({
          index: window.__sim.player.currentStepIndex,
          simTimeSec: window.__sim.engine.simTimeSec,
          engineState: window.__sim.engine.state,
        }));
        if (stepped.engineState === 'paused' && stepped.index > beforeStep.index) break;
        if (Date.now() > stepDeadline) break;
        await page.waitForTimeout(SEQ_POLL_INTERVAL_MS);
      }
      const stepAdvance = stepped.simTimeSec - beforeStep.simTimeSec;
      if (stepped.index === beforeStep.index + 1 && stepped.engineState === 'paused'
          && stepAdvance > 0 && stepAdvance < ORCH_STEP_MAX_SIM_SEC) {
        pass('orchestration: stepNode advances exactly one node then pauses',
          `index ${beforeStep.index}→${stepped.index}, +${stepAdvance.toFixed(2)}s sim`);
      } else {
        fail('orchestration: stepNode advances exactly one node then pauses',
          JSON.stringify({ beforeStep, stepped, stepAdvance }));
      }

      // (f) setAutoPause 토글이 facade + UI 체크박스에 반영 (unexpected 충돌 강제는 비결정적 — 배선만 검증)
      const autoPause = await page.evaluate(() => {
        const o = window.__sim.orchestrator;
        o.setAutoPause(true);
        const onFlag = o.autoPause();
        const checkbox = document.querySelector('[data-testid="autopause-toggle"]');
        const checked = checkbox ? checkbox.checked : null;
        o.setAutoPause(false);
        return { onFlag, checked, offFlag: o.autoPause() };
      });
      if (autoPause.onFlag === true && autoPause.checked === true && autoPause.offFlag === false) {
        pass('orchestration: setAutoPause(true) reflected in facade + UI checkbox, toggles off');
      } else {
        fail('orchestration: setAutoPause(true) reflected in facade + UI checkbox, toggles off', JSON.stringify(autoPause));
      }

      // (g) runFromNode(secondNode) → 되감고 재생 (sim 리셋 후 전진, 크래시 없음, 유효 종료)
      await orchStop();
      const nodeIds = await page.evaluate(() => window.__sim.flowGraph.nodeIds());
      const secondNodeId = nodeIds[1];
      await page.evaluate((id) => window.__sim.orchestrator.runFromNode(id), secondNodeId);
      const rerunDeadline = Date.now() + ORCH_REALTIME_DEADLINE_MS;
      let rerun = null;
      for (;;) {
        rerun = await page.evaluate(() => ({
          index: window.__sim.player.currentStepIndex,
          simTimeSec: window.__sim.engine.simTimeSec,
          engineState: window.__sim.engine.state,
          activeNodeId: window.__sim.orchestrator.activeNodeId(),
        }));
        if (rerun.engineState === 'paused' && rerun.index >= 1) break;
        if (Date.now() > rerunDeadline) break;
        await page.waitForTimeout(SEQ_POLL_INTERVAL_MS);
      }
      if (rerun.index >= 1 && rerun.simTimeSec > 0 && rerun.activeNodeId === secondNodeId) {
        pass('orchestration: runFromNode(second) rewinds then fast-forwards to that node',
          `index=${rerun.index}, sim=${rerun.simTimeSec.toFixed(2)}s, active=${rerun.activeNodeId}`);
      } else {
        fail('orchestration: runFromNode(second) rewinds then fast-forwards to that node', JSON.stringify(rerun));
      }

      // 다음 검증(공통 페이지 에러)을 위해 정지로 리셋
      await orchStop();
    }

    // ── 로봇↔로봇 충돌 회귀 (--expect=two-arms) ─────────────────────
    // 결함: ROBOT 그룹이 링크 필터에서 빠져 있어 두 로봇이 서로 통과했다.
    // 여기서는 (a) 실제 충돌 이벤트 발행, (b) 접촉점 좌표 보강, (c) 자기 링크 노이즈
    // 억제, (d) 방향키 오브젝트 이동을 실브라우저에서 확인한다.
    if (expectArg === 'two-arms') {
      const TWO_ARMS_BUDGET_MS = 30000;
      await page.evaluate(() => window.__sim.orchestrator.play());

      const deadline = Date.now() + TWO_ARMS_BUDGET_MS;
      let snapshot = null;
      while (Date.now() < deadline) {
        snapshot = await page.evaluate(() => {
          const all = window.__sim.collision.recent(100);
          const isPairOf = (e, x, y) => (e.a === x && e.b === y) || (e.a === y && e.b === x);
          return {
            robotPairStarts: all.filter(
              (e) => e.phase === 'start' && isPairOf(e, 'arm_left', 'arm_right'),
            ),
            selfContacts: all.filter((e) => e.a === e.b),
            status: window.__sim.player.status,
          };
        });
        if (snapshot.robotPairStarts.length > 0) break;
        await page.waitForTimeout(SEQ_POLL_INTERVAL_MS);
      }

      const starts = snapshot?.robotPairStarts ?? [];
      if (starts.length > 0) {
        pass('two-arms: 로봇↔로봇 충돌이 감지된다 ★ 회귀',
          `starts=${starts.length} t=${starts[0].timeSec.toFixed(3)}s`);
      } else {
        fail('two-arms: 로봇↔로봇 충돌이 감지된다 ★ 회귀', JSON.stringify(snapshot));
      }

      // 접촉점 보강 — 마커를 띄울 월드 좌표가 실려야 한다
      const withPoint = starts.find((e) => Array.isArray(e.point) && e.point.length === 3);
      if (withPoint) {
        pass('two-arms: 접촉 이벤트에 월드 접촉점이 실린다 (접촉 마커 입력)',
          `point=[${withPoint.point.map((n) => n.toFixed(3)).join(', ')}]`);
      } else {
        fail('two-arms: 접촉 이벤트에 월드 접촉점이 실린다 (접촉 마커 입력)',
          JSON.stringify(starts.slice(0, 3)));
      }

      // 자기 링크(같은 EntityId) 접촉은 selfCollision=false이므로 로그에 없어야 한다
      if ((snapshot?.selfContacts ?? []).length === 0) {
        pass('two-arms: 자기 링크 접촉은 억제된다 (selfCollision=false)');
      } else {
        fail('two-arms: 자기 링크 접촉은 억제된다 (selfCollision=false)',
          JSON.stringify(snapshot.selfContacts.slice(0, 3)));
      }

      // 방향키 오브젝트 이동 — 선택 후 ArrowRight 1회 = NUDGE_STEP_M(0.05m)
      await page.evaluate(() => window.__sim.orchestrator.stop());
      await page.waitForTimeout(300);
      const NUDGE_EXPECTED_M = 0.05;
      const NUDGE_TOLERANCE_M = 0.005;
      const posOf = () =>
        page.evaluate(() => {
          const s = window.__sim;
          return s.world.getPose(s.world.bodiesOfEntity('witness_box')[0]).position;
        });
      await page.evaluate(() => window.__sim.editor.select('witness_box'));
      // 방향키는 **뷰포트 스코프** 바인딩이다(불변식 §2.10 — 라우터가 소유권을 가른다).
      // 실제 사용자는 3D 화면을 클릭하면서 포커스를 얻지만, 파사드 select()에는 그
      // 부수효과가 없으므로 게이트가 같은 조건을 명시적으로 만든다.
      await focusViewportSlot(page);
      const beforeNudge = await posOf();
      await page.keyboard.press('ArrowRight');
      await page.waitForTimeout(300);
      const afterNudge = await posOf();
      const horizontalDelta = Math.hypot(
        afterNudge[0] - beforeNudge[0],
        afterNudge[2] - beforeNudge[2],
      );
      if (Math.abs(horizontalDelta - NUDGE_EXPECTED_M) < NUDGE_TOLERANCE_M) {
        pass('two-arms: 방향키로 선택 오브젝트가 이동한다 (물리 반영)',
          `delta=${horizontalDelta.toFixed(4)}m`);
      } else {
        fail('two-arms: 방향키로 선택 오브젝트가 이동한다 (물리 반영)',
          `delta=${horizontalDelta.toFixed(4)}m before=${JSON.stringify(beforeNudge)} after=${JSON.stringify(afterNudge)}`);
      }

      // ── 로봇도 오브젝트와 똑같이 이동한다 ★ 회귀 ───────────────────
      // 사용자 보고: "오브젝트는 이동하는데 로봇이 이동을 안한다".
      // 로봇은 물리 pose(kinematic 링크 바디)와 시각 pose(FK 그래프 루트)가 서로 다른
      // 주체가 소유하므로 **둘이 함께** 움직였는지, 그리고 되감기(⏹ Stop → reset) 후에도
      // 편집된 배치가 유지되는지까지 확인한다.
      const ROBOT_ID = 'arm_left';
      /** 엔티티의 물리(첫 바디)·시각(spec 위치/회전)·현재 선택 스냅샷 */
      const robotSnapshotOf = (entityId) =>
        page.evaluate((id) => {
          const s = window.__sim;
          const body = s.world.bodiesOfEntity(id)[0];
          const entity = s.editor.serialize().entities.find((e) => e.id === id);
          return {
            physics: s.world.getPose(body).position,
            spec: entity ? entity.transform.position : null,
            rotation: entity ? (entity.transform.rotation ?? [0, 0, 0, 1]) : null,
            selected: s.editor.selectedId(),
          };
        }, entityId);
      const robotSnapshot = () => robotSnapshotOf(ROBOT_ID);

      await page.evaluate((robotId) => window.__sim.editor.select(robotId), ROBOT_ID);
      await focusViewportSlot(page); // 방향키는 뷰포트 스코프 (위 witness_box 주석 참조)
      const robotBefore = await robotSnapshot();
      await page.keyboard.press('ArrowRight');
      await page.waitForTimeout(300);
      const robotAfter = await robotSnapshot();

      const robotPhysDelta = Math.hypot(
        robotAfter.physics[0] - robotBefore.physics[0],
        robotAfter.physics[2] - robotBefore.physics[2],
      );
      const robotSpecDelta =
        robotBefore.spec && robotAfter.spec
          ? Math.hypot(
              robotAfter.spec[0] - robotBefore.spec[0],
              robotAfter.spec[2] - robotBefore.spec[2],
            )
          : Number.NaN;
      const robotMoved =
        robotAfter.selected === ROBOT_ID &&
        Math.abs(robotPhysDelta - NUDGE_EXPECTED_M) < NUDGE_TOLERANCE_M &&
        Math.abs(robotSpecDelta - NUDGE_EXPECTED_M) < NUDGE_TOLERANCE_M;
      if (robotMoved) {
        pass('two-arms: 방향키로 로봇이 이동한다 — 물리+시각(spec) 동시 ★ 회귀',
          `phys=${robotPhysDelta.toFixed(4)}m spec=${robotSpecDelta.toFixed(4)}m`);
      } else {
        fail('two-arms: 방향키로 로봇이 이동한다 — 물리+시각(spec) 동시 ★ 회귀',
          `phys=${robotPhysDelta.toFixed(4)}m spec=${robotSpecDelta.toFixed(4)}m ` +
          `before=${JSON.stringify(robotBefore)} after=${JSON.stringify(robotAfter)}`);
      }

      // 되감기(⏹ Stop)가 로봇 루트를 spec 배치로 되돌린다 ★ 회귀.
      // 판별력을 위해 **시각 루트만** 어긋뜨린 뒤 reset을 부른다: 로봇 루트는 물리가
      // 아니라 렌더 핸들이 소유하므로, 레코드에 initialPose가 없고 reset이 루트를
      // 복구하지 않으면 커밋되지 않은 드래그 프리뷰가 영구히 남는다(원 결함).
      const RESET_PERTURB_M = 0.4;
      await page.evaluate(([robotId, offset]) => {
        const record = window.__sim.sceneHandle.builtEntities.get(robotId);
        const p = record.robot.handle; // 렌더 핸들 — spec/물리를 건드리지 않는 시각 경로
        const entity = window.__sim.editor.serialize().entities.find((e) => e.id === robotId);
        const base = entity.transform.position;
        p.setRootTransform({
          position: [base[0] + offset, base[1], base[2] + offset],
          rotation: entity.transform.rotation ?? [0, 0, 0, 1],
        });
      }, [ROBOT_ID, RESET_PERTURB_M]);
      await page.evaluate(() => window.__sim.orchestrator.stop());
      await page.waitForTimeout(400);
      const robotAfterReset = await robotSnapshot();
      const resetDrift = Math.hypot(
        robotAfterReset.physics[0] - robotAfter.physics[0],
        robotAfterReset.physics[2] - robotAfter.physics[2],
      );
      if (resetDrift < NUDGE_TOLERANCE_M) {
        pass('two-arms: 되감기(⏹ Stop)가 로봇 루트를 spec 배치로 복원한다 ★ 회귀',
          `perturb=${RESET_PERTURB_M}m drift=${resetDrift.toFixed(4)}m pos=[${robotAfterReset.physics.map((n) => n.toFixed(3)).join(', ')}]`);
      } else {
        fail('two-arms: 되감기(⏹ Stop)가 로봇 루트를 spec 배치로 복원한다 ★ 회귀',
          `drift=${resetDrift.toFixed(4)}m after=${JSON.stringify(robotAfter)} reset=${JSON.stringify(robotAfterReset)}`);
      }

      // 기즈모 앵커가 로봇의 **보이는 몸통**(시각 AABB 중심)에 붙는지 — 핸들을 못 잡아
      // 카메라만 돌던 근본 원인의 회귀 가드. 좌표만 보면 attach 대상이 루트로 되돌아간
      // 회귀를 못 잡으므로(앵커 좌표는 그대로다) attachedToAnchor를 함께 본다.
      const ANCHOR_TOLERANCE_M = 0.01;
      const dist3 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
      const probes = await page.evaluate((robotId) => {
        const s = window.__sim;
        s.editor.select(robotId);
        const robot = s.editor.anchorProbe();
        s.editor.select('witness_box');
        const box = s.editor.anchorProbe();
        s.editor.select(robotId);
        return { robot, box };
      }, ROBOT_ID);

      const robotProbe = probes?.robot ?? null;
      const boxProbe = probes?.box ?? null;
      if (robotProbe && boxProbe) {
        const robotAnchorGap = dist3(robotProbe.anchor, robotProbe.visualCenter);
        const robotRootGap = dist3(robotProbe.rootOrigin, robotProbe.visualCenter);
        const boxAnchorGap = dist3(boxProbe.anchor, boxProbe.visualCenter);
        // 로봇: 앵커는 보이는 중심에 붙고(≈0), 루트 원점은 여전히 발밑이라 떨어져 있다.
        // 오브젝트: 원래부터 0 — 두 경우 모두 "핸들이 보이는 몸통 위"라는 같은 계약.
        if (
          robotAnchorGap < ANCHOR_TOLERANCE_M &&
          boxAnchorGap < ANCHOR_TOLERANCE_M &&
          robotRootGap > ANCHOR_TOLERANCE_M &&
          robotProbe.attachedToAnchor === true &&
          boxProbe.attachedToAnchor === true
        ) {
          pass('two-arms: 기즈모가 로봇의 보이는 몸통 중심(앵커)에 붙는다 ★ 회귀',
            `robotAnchorGap=${robotAnchorGap.toFixed(4)}m robotRootGap=${robotRootGap.toFixed(4)}m boxAnchorGap=${boxAnchorGap.toFixed(4)}m attached=${robotProbe.attachedToAnchor}`);
        } else {
          fail('two-arms: 기즈모가 로봇의 보이는 몸통 중심(앵커)에 붙는다 ★ 회귀',
            JSON.stringify({ robotProbe, boxProbe }));
        }
      } else {
        fail('two-arms: 기즈모가 로봇의 보이는 몸통 중심(앵커)에 붙는다 ★ 회귀',
          `anchorProbe 미반환 robot=${JSON.stringify(robotProbe)} box=${JSON.stringify(boxProbe)}`);
      }

      // ── 사용자의 실제 제스처: 보이는 몸통을 마우스로 끌기 ★ 회귀 ────
      // 원 결함("로봇이 이동을 안한다")은 이 경로에서만 재현됐다 — 파사드·방향키는
      // 전부 정상이었다. 핸들이 몸통에서 벗어나면(attach(root) 회귀) 드래그가 기즈모에
      // 잡히지 않고 OrbitControls로 흘러 **이동량 0**이 되므로 이 어서션이 적발한다.
      const GIZMO_DRAG_MIN_M = 0.05; // 이보다 적게 움직이면 핸들을 못 잡은 것
      const GIZMO_DRAG_PX = { dx: 90, dy: 30 };
      /** 앵커 화면 좌표에서 (dx, dy)만큼 드래그하고 spec/물리 이동량을 잰다 */
      const dragFromAnchor = async (entityId, dx, dy) => {
        await page.evaluate((id) => window.__sim.editor.select(id), entityId);
        await page.waitForTimeout(150);
        const start = await page.evaluate(() => window.__sim.editor.anchorScreenPoint());
        if (!start) return { moved: Number.NaN, spec: Number.NaN };
        const before = await robotSnapshotOf(entityId);
        await page.mouse.move(start[0], start[1]);
        await page.mouse.down();
        await page.mouse.move(start[0] + dx, start[1] + dy, { steps: 14 });
        await page.mouse.up();
        await page.waitForTimeout(350);
        const after = await robotSnapshotOf(entityId);
        return {
          moved: Math.hypot(
            after.physics[0] - before.physics[0],
            after.physics[2] - before.physics[2],
          ),
          spec: Math.hypot(after.spec[0] - before.spec[0], after.spec[2] - before.spec[2]),
        };
      };

      const robotDrag = await dragFromAnchor(ROBOT_ID, GIZMO_DRAG_PX.dx, GIZMO_DRAG_PX.dy);
      const boxDrag = await dragFromAnchor('witness_box', GIZMO_DRAG_PX.dx, GIZMO_DRAG_PX.dy);
      if (
        robotDrag.moved > GIZMO_DRAG_MIN_M &&
        robotDrag.spec > GIZMO_DRAG_MIN_M &&
        boxDrag.moved > GIZMO_DRAG_MIN_M
      ) {
        pass('two-arms: 보이는 로봇 몸통을 마우스로 끌면 로봇이 이동한다 ★ 회귀',
          `robot phys=${robotDrag.moved.toFixed(4)}m spec=${robotDrag.spec.toFixed(4)}m / box phys=${boxDrag.moved.toFixed(4)}m`);
      } else {
        fail('two-arms: 보이는 로봇 몸통을 마우스로 끌면 로봇이 이동한다 ★ 회귀',
          `robot=${JSON.stringify(robotDrag)} box=${JSON.stringify(boxDrag)} (핸들을 못 잡으면 0m)`);
      }

      // ── 회전 기즈모는 로봇을 **제자리에서** 돌린다 ★ 회귀 ───────────
      // 앵커를 피벗으로 루트를 역산하면 회전만 해도 베이스가 앵커를 중심으로 공전해
      // 바닥에 서 있던 로봇이 최대 0.7 m 떠오른다. 피벗은 루트 원점(베이스)이어야 한다.
      await page.evaluate((robotId) => window.__sim.editor.select(robotId), ROBOT_ID);
      await page.waitForTimeout(150);
      await page.keyboard.press('e'); // 회전 모드 (W/E/R — UX §3.3)
      const canvasBox = await page.locator('canvas').boundingBox();
      // TransformControls는 기즈모를 화면 크기 일정하게 그린다 — 회전 링의 화면 반경은
      // 캔버스 높이에 비례한다(three r169: 0.5 * factor/4, factor = dist * 1.9 * tan(fov/2)).
      const ringRadiusPx = 0.11875 * canvasBox.height;
      const ROTATE_DRAG_PX = 140;
      const ROTATE_MAX_POSITION_SHIFT_M = 0.01;
      /**
       * 채택 최소 회전량 (쿼터니언 성분 거리) — 판별력의 핵심.
       * 0.1 ≈ 11.5°이고, 앵커 공전 결함이 남아 있으면 이때 베이스가
       * 2·0.376·sin(θ/2) ≈ 0.075 m 밀린다(허용치 0.01 m의 7배). 회전이 이보다 작은
       * 제스처는 "핸들을 스쳤다"로 보고 다음 후보를 계속 시도한다.
       */
      const ROTATE_MIN_DELTA = 0.1;
      let rotateResult = null;
      let rotateBest = null;
      // 어느 링(X/Y/Z/E)을 잡게 될지는 카메라 각도에 달렸다 — 반경·방위를 훑어
      // "충분히 큰 회전이 일어난" 첫 제스처를 채택한다(못 잡으면 FAIL — 조용한 통과 금지).
      for (const scale of [0.8, 1.0, 1.2]) {
        for (const angleDeg of [0, 90, 180, 270]) {
          const anchor = await page.evaluate(() => window.__sim.editor.anchorScreenPoint());
          const before = await robotSnapshotOf(ROBOT_ID);
          const angleRad = (angleDeg * Math.PI) / 180;
          const gx = anchor[0] + Math.cos(angleRad) * ringRadiusPx * scale;
          const gy = anchor[1] + Math.sin(angleRad) * ringRadiusPx * scale;
          // 접선 방향으로 끈다 (반경 방향 드래그는 회전각이 잘 안 생긴다)
          const tx = -Math.sin(angleRad) * ROTATE_DRAG_PX;
          const ty = Math.cos(angleRad) * ROTATE_DRAG_PX;
          await page.mouse.move(gx, gy);
          await page.mouse.down();
          await page.mouse.move(gx + tx, gy + ty, { steps: 14 });
          await page.mouse.up();
          await page.waitForTimeout(250);
          const after = await robotSnapshotOf(ROBOT_ID);
          const rotationDelta = Math.hypot(
            ...after.rotation.map((v, i) => v - before.rotation[i]),
          );
          const positionShift = Math.hypot(
            after.spec[0] - before.spec[0],
            after.spec[1] - before.spec[1],
            after.spec[2] - before.spec[2],
          );
          if (rotateBest === null || rotationDelta > rotateBest.rotationDelta) {
            rotateBest = { rotationDelta, positionShift, scale, angleDeg };
          }
          if (rotationDelta > ROTATE_MIN_DELTA) {
            rotateResult = { rotationDelta, positionShift, scale, angleDeg };
            break;
          }
        }
        if (rotateResult !== null) break;
      }
      if (rotateResult === null) {
        fail('two-arms: 회전 기즈모가 로봇을 제자리에서 돌린다 (베이스가 뜨지 않는다) ★ 회귀',
          `충분한 회전(${ROTATE_MIN_DELTA})을 만드는 링 파지에 실패 — best=${JSON.stringify(rotateBest)} ringRadiusPx=${ringRadiusPx.toFixed(1)} canvas=${JSON.stringify(canvasBox)}`);
      } else if (rotateResult.positionShift < ROTATE_MAX_POSITION_SHIFT_M) {
        pass('two-arms: 회전 기즈모가 로봇을 제자리에서 돌린다 (베이스가 뜨지 않는다) ★ 회귀',
          `dRot=${rotateResult.rotationDelta.toFixed(4)} posShift=${rotateResult.positionShift.toFixed(4)}m (ring×${rotateResult.scale} @${rotateResult.angleDeg}°)`);
      } else {
        fail('two-arms: 회전 기즈모가 로봇을 제자리에서 돌린다 (베이스가 뜨지 않는다) ★ 회귀',
          `posShift=${rotateResult.positionShift.toFixed(4)}m dRot=${rotateResult.rotationDelta.toFixed(4)} — 회전이 베이스를 옮겼다(앵커 공전 회귀)`);
      }
      await page.keyboard.press('w'); // 이동 모드로 복귀 (다음 검증에 영향 없게)
    }

    // ── conveyor-pick-place — 컨베이어 이송 · 재순환 · 포토아이 픽앤플레이스 ──
    //
    // 이 게이트가 지키는 것은 "컨베이어가 데이터로 선언되고 물리로 동작한다"이다.
    // 벨트는 fixed 바디라 **자신은 움직이지 않으므로**, 동작 증거는 오직 그 위 사물의
    // 이동뿐이다 — 그래서 재생 전에 벨트만 돌려 사물이 실제로 실려 가는지부터 본다.
    if (expectArg === 'conveyor-pick-place') {
      const itemX = async (id) =>
        page.evaluate((entityId) => {
          const w = window.__sim.world;
          const b = w.bodiesOfEntity(entityId)[0];
          return b === undefined ? null : w.getPose(b).position[0];
        }, id);

      // (a) ★ 벨트가 사물을 실어 나른다 — 시퀀스 재생 전, 벨트만으로
      const beforeCarry = await itemX('item_a');
      await page.waitForTimeout(CPP_BELT_OBSERVE_SEC * 1000);
      const afterCarry = await itemX('item_a');
      const carried = beforeCarry !== null && afterCarry !== null ? afterCarry - beforeCarry : null;
      // 한 바퀴 돌았으면 x가 되감기므로 벨트 길이를 더해 진행량을 복원한다
      const travelled =
        carried === null ? null : carried >= 0 ? carried : carried + (CPP_BELT_END_X - CPP_BELT_START_X);
      if (travelled !== null && travelled >= CPP_MIN_BELT_TRAVEL_M) {
        pass('conveyor: 벨트가 사물을 진행 방향으로 실어 나른다 (재생 전, 벨트 단독)',
          `${beforeCarry?.toFixed(3)} → ${afterCarry?.toFixed(3)} (이송 ${travelled.toFixed(3)}m / ${CPP_BELT_OBSERVE_SEC}s)`);
      } else {
        fail('conveyor: 벨트가 사물을 진행 방향으로 실어 나른다 (재생 전, 벨트 단독)',
          `before=${beforeCarry} after=${afterCarry} travelled=${travelled}`);
      }

      // (b) ★ 재순환 — 끝에 도달한 사물이 시작점으로 돌아온다 ("물건이 계속 온다")
      let recycles = 0;
      let previousX = await itemX('item_a');
      const recycleDeadline = Date.now() + CPP_RECYCLE_OBSERVE_SEC * 1000;
      while (Date.now() < recycleDeadline) {
        await page.waitForTimeout(SEQ_POLL_INTERVAL_MS * 2);
        const x = await itemX('item_a');
        if (x !== null && previousX !== null && x < previousX - CPP_MIN_BELT_TRAVEL_M) recycles += 1;
        previousX = x;
      }
      if (recycles >= 1) {
        pass('conveyor: 끝에 도달한 사물이 시작점으로 재순환한다 ★ (런타임 스폰 없이)',
          `재순환 ${recycles}회 / ${CPP_RECYCLE_OBSERVE_SEC}s`);
      } else {
        fail('conveyor: 끝에 도달한 사물이 시작점으로 재순환한다 ★ (런타임 스폰 없이)',
          `재순환 0회 — 마지막 x=${previousX}`);
      }

      // (c) 시퀀스는 Play 전까지 멈춰 있다 (human-in-the-loop §2.12)
      const initial = await page.evaluate(() => {
        const p = window.__sim?.player;
        return p ? { status: p.status, stepCount: p.stepCount } : null;
      });
      if (initial?.status === 'idle' && initial.stepCount === CPP_STEP_COUNT) {
        pass(`conveyor: sequence loaded, no autoplay (idle, ${CPP_STEP_COUNT} steps)`);
      } else {
        fail(`conveyor: sequence loaded, no autoplay (idle, ${CPP_STEP_COUNT} steps)`,
          `initial=${JSON.stringify(initial)}`);
      }

      if (!initial) {
        fail('conveyor: interaction checks skipped', 'player facade missing');
      } else {
        // 앞의 관측 단계(벨트 15s + 재순환)가 라인의 위상을 바꿔 놓았다 — 상자들이
        // 재순환을 몇 바퀴 돌아 씬 선언과 다른 순서/위치에 있다. 시퀀스는 선입선출
        // (item_a → b → c)을 전제하므로 그대로 재생하면 오지 않을 상자를 기다리다 죽는다
        // (실측: step 12에서 45s 정지). 그래서 재생 전에 초기 상태로 돌아가야 한다.
        //
        // ★ ⏹(orchestrator.stop)로는 부족하다 — **결정론이 돌아오지 않는다.**
        // stop()은 바디를 스펙 좌표로 텔레포트하지만 Rapier 솔버의 warm-start 임펄스와
        // 접촉 매니폴드는 지우지 못한다. 관측 15초 동안 쌓인 그 내부 상태는 머신 부하에
        // 따라 tick 수가 달라져 매번 다르고, 상자끼리 밀치는 사슬에서 증폭된다.
        // A/B 실측(각 3회): 로드 직후 바로 재생 → 최종 좌표가 소수점 5자리까지 3회 동일.
        // 15초 관측 → stop() → 재생 → 3회 중 2회가 다른 결과(한 번은 완주 54.4s, 정상 24.9s).
        // 이것이 이 게이트가 유휴에서 6회 중 1회, 전체 스위트 13번째 위치에서 2/2 실패하던
        // 원인이다. **페이지를 새로 열어 새 월드에서 재생한다** — 되감기가 아니라 재빌드다.
        await page.goto(url, { waitUntil: 'load' });
        await page.waitForFunction(() => window.__sim !== undefined, undefined, { timeout: 15000 });

        // 사이클마다 **놓는 순간**의 그리퍼–상자 거리를 표본한다. 최종 위치만 보면
        // 이송 중 떨어뜨린 것과 제대로 놓은 것이 구분되지 않는다(둘 다 바닥에서 끝난다).
        //
        // 표본은 반드시 **물리 tick** 위에서 뜬다. engine.onTick은 rAF당 1회라(240Hz 물리에
        // 60Hz 표본) "놓기 step의 첫 tick"을 최대 24 tick 늦게 잡고, 그 지연이 프레임
        // 타이밍에 좌우된다 — 같은 초기 상태에서 놓기 거리가 0.019 / 0.021 / 0.044로
        // 갈리는 것을 실측했다. 그건 시뮬 차이가 아니라 측정 아티팩트다.
        await page.evaluate(
          ({ releaseSteps, items }) => {
            window.__releaseGaps = {};
            window.__sim.engine.onPhysicsTick(() => {
              const player = window.__sim.player;
              if (!player) return;
              const cycle = releaseSteps.indexOf(player.currentStepIndex);
              if (cycle < 0) return;
              const item = items[cycle];
              if (window.__releaseGaps[item] !== undefined) return;
              const w = window.__sim.world;
              const body = w.bodiesOfEntity(item)[0];
              if (body === undefined) return;
              const p = w.getPose(body).position;
              // 베이스에서 가장 먼 링크를 그리퍼로 본다 (anchorProbe와 같은 관례)
              let far = null;
              let maxR = -1;
              for (const bid of w.bodiesOfEntity('arm')) {
                const q = w.getPose(bid).position;
                const r = Math.hypot(q[0], q[2]);
                if (r > maxR) { maxR = r; far = q; }
              }
              if (far === null) return;
              window.__releaseGaps[item] = Math.hypot(p[0] - far[0], p[2] - far[2]);
            });
          },
          { releaseSteps: CPP_RELEASE_STEP_INDEXES, items: CPP_CYCLE_ITEMS },
        );

        const last = await playAndAwaitDone(page, CPP_SIM_TIME_BUDGET_SEC);
        const history = await page.evaluate(
          (limit) => window.__sim.collision.recent(limit),
          HISTORY_FETCH_LIMIT,
        );

        // (d) 포토아이 게이트가 **세 상자의 도착을 각각** 감지한다 (sensor start)
        // timeSec > 0 조건이 핵심이다: 사물이 게이트와 **겹친 채로 스폰**되면 t=0에
        // 센서 start가 한 건 생겨, 벨트가 전혀 돌지 않아도 이 어서션이 통과한다.
        // (실제로 첫 배치가 그 상태였다 — 씬을 고치고 어서션도 함께 조인다.)
        const gateMisses = CPP_CYCLE_ITEMS.filter(
          (item) =>
            !history.some(
              (e) =>
                e.phase === 'start' &&
                e.kind === 'sensor' &&
                isPair(e, item, 'pick_gate') &&
                e.timeSec > CPP_MIN_EVENT_SIM_SEC,
            ),
        );
        if (gateMisses.length === 0) {
          pass('conveyor: 포토아이(pick_gate)가 세 상자의 도착을 각각 감지한다 (스폰 겹침 아님)');
        } else {
          fail('conveyor: 포토아이(pick_gate)가 세 상자의 도착을 각각 감지한다 (스폰 겹침 아님)',
            `미감지=${JSON.stringify(gateMisses)} pairs=${JSON.stringify([...new Set(history.map((e) => `${e.a}×${e.b}`))])}`);
        }

        // (e) 로봇이 세 상자를 각각 잡는다 (arm×item_* 접촉 — 선언된 타겟)
        const pickMisses = CPP_CYCLE_ITEMS.filter(
          (item) =>
            !history.some(
              (e) => e.phase === 'start' && e.kind === 'contact' && isPair(e, 'arm', item),
            ),
        );
        if (pickMisses.length === 0) {
          pass('conveyor: 로봇이 세 상자에 각각 접촉한다 (선언된 타겟)');
        } else {
          fail('conveyor: 로봇이 세 상자에 각각 접촉한다 (선언된 타겟)',
            `미접촉=${JSON.stringify(pickMisses)}`);
        }

        // (f) ★ 사용자 요청의 본체 — "상자 세 개를 연속으로 드랍존에 안착"시킨다.
        //     이벤트(존 진입)와 최종 정지 위치를 **둘 다** 본다: 진입만 보면 존 위를
        //     스쳐 지나간 상자도 통과하고, 위치만 보면 센서 쌍이 끊겨도 통과한다.
        const zoneMisses = CPP_CYCLE_ITEMS.filter(
          (item) =>
            !history.some(
              (e) => e.phase === 'start' && e.kind === 'sensor' && isPair(e, item, 'drop_zone'),
            ),
        );
        const resting = await page.evaluate((items) => {
          const w = window.__sim.world;
          // 존 기하는 씬 데이터에서 읽는다 — 게이트에 좌표를 복사해 두면 씬을 옮길 때
          // 게이트가 조용히 거짓말을 한다(존은 옮겼는데 어서션은 옛 자리를 본다).
          const zone = window.__sim.editor.serialize().entities.find((e) => e.id === 'drop_zone');
          const half = zone?.physics?.colliders?.[0]?.shape?.halfExtents ?? null;
          const center = zone?.transform?.position ?? null;
          if (!half || !center) return null;
          const out = {};
          for (const id of items) {
            const body = w.bodiesOfEntity(id)[0];
            if (body === undefined) { out[id] = null; continue; }
            const p = w.getPose(body).position;
            out[id] = {
              p: [p[0], p[1], p[2]],
              inside:
                Math.abs(p[0] - center[0]) <= half[0] && Math.abs(p[2] - center[2]) <= half[2],
            };
          }
          return out;
        }, CPP_CYCLE_ITEMS);
        const outside = resting
          ? CPP_CYCLE_ITEMS.filter((id) => !resting[id]?.inside)
          : CPP_CYCLE_ITEMS;
        if (zoneMisses.length === 0 && outside.length === 0) {
          pass('conveyor: 세 상자가 연속으로 감지 존에 안착한다 ★ (라인 픽앤플레이스 성립)',
            CPP_CYCLE_ITEMS.map((id) => `${id}=[${resting[id].p.map((v) => v.toFixed(2)).join(',')}]`).join(' '));
        } else {
          fail('conveyor: 세 상자가 연속으로 감지 존에 안착한다 ★ (라인 픽앤플레이스 성립)',
            `존 진입 없음=${JSON.stringify(zoneMisses)} / 존 밖 정지=${JSON.stringify(outside)} / ${JSON.stringify(resting)}`);
        }

        // (f-2) ★ "가다가 떨어뜨린다"의 직접 회귀 — 놓는 순간 상자가 아직 손에 있는가.
        const gaps = await page.evaluate(() => window.__releaseGaps);
        const dropped = CPP_CYCLE_ITEMS.filter(
          (id) => !(gaps?.[id] <= CPP_MAX_RELEASE_GAP_M),
        );
        if (dropped.length === 0) {
          pass('conveyor: 놓는 순간 세 상자가 모두 아직 그리퍼에 있다 (이송 중 낙하 없음)',
            CPP_CYCLE_ITEMS.map((id) => `${id}=${(gaps[id] * 100).toFixed(1)}cm`).join(' '));
        } else {
          fail('conveyor: 놓는 순간 세 상자가 모두 아직 그리퍼에 있다 (이송 중 낙하 없음)',
            `${JSON.stringify(gaps)} — 손에서 빠졌거나 CPP_RELEASE_STEP_INDEXES가 시퀀스와 어긋났다`);
        }

        // (g) 정상 실행은 벨트를 건드리지 않는다 — 샘플이 매번 충돌을 보고하면
        //     "정상"의 기준선이 무너진다 (pick-and-place 선반 계약과 같은 이유).
        //     팔이 라인 밖에서 대기하도록 포토아이로 트리거하는 설계의 근거이기도 하다.
        const beltHits = history.filter((e) => e.phase === 'start' && isPair(e, 'arm', 'belt'));
        if (beltHits.length === 0) {
          pass('conveyor: 정상 시퀀스는 벨트를 건드리지 않는다 (arm×belt 0건)');
        } else {
          fail('conveyor: 정상 시퀀스는 벨트를 건드리지 않는다 (arm×belt 0건)',
            `hits=${JSON.stringify(beltHits.map((e) => e.timeSec.toFixed(2)))}`);
        }

        // (h) 라인은 세 개를 다 처리한 뒤에도 계속 돈다 — 다음 물건이 오면 실어 간다.
        //     세 상자가 모두 적재 레인으로 빠졌으므로 벨트 위에 관측 대상이 없다.
        //     상자 하나를 라인에 **다시 올려** 실려 가는지로 판정한다(위상 무관).
        //     시퀀스 완주 후 엔진은 일시정지 상태이고 벨트는 물리 스텝에서만 구동하므로,
        //     관측 전에 물리를 재개한다 (멈춘 시뮬에서 벨트가 도는 것이 오히려 결함이다).
        await page.evaluate((seat) => {
          window.__sim.editor.updateTransform('item_a', { position: seat });
          window.__sim.engine.play();
        }, CPP_BELT_SEAT_POSITION);
        const bBefore = await itemX('item_a');
        await page.waitForTimeout(CPP_BELT_OBSERVE_SEC * 1000);
        const bAfter = await itemX('item_a');
        const bMoved = bBefore !== null && bAfter !== null && Math.abs(bAfter - bBefore) > 0.05;
        if (bMoved) {
          pass('conveyor: 세 개를 처리한 뒤에도 라인이 계속 돈다 (다음 물건이 오면 실어 간다)',
            `item_a ${bBefore?.toFixed(3)} → ${bAfter?.toFixed(3)}`);
        } else {
          fail('conveyor: 세 개를 처리한 뒤에도 라인이 계속 돈다 (다음 물건이 오면 실어 간다)',
            `item_a ${bBefore} → ${bAfter}`);
        }

        // (i) 시퀀스 완주
        if (last.status === 'done') {
          pass('conveyor: sequence done', `elapsed=${last.elapsedSimSec.toFixed(2)}s`);
        } else {
          fail('conveyor: sequence done', `last=${JSON.stringify(last)}`);
        }

        // (j) 인스펙터에 Conveyor 섹션이 나타난다 (벨트를 선택했을 때만)
        const sections = await page.evaluate(() => {
          const sel = window.__sim.editor;
          sel.select('belt');
          const beltHas = document.querySelector('[data-testid="ee-sec-conveyor"]') !== null;
          sel.select('item_a');
          const itemHas = document.querySelector('[data-testid="ee-sec-conveyor"]') !== null;
          sel.select(null);
          return { beltHas, itemHas };
        });
        if (sections.beltHas && !sections.itemHas) {
          pass('conveyor: 인스펙터 Conveyor 섹션은 벨트에만 나타난다');
        } else {
          fail('conveyor: 인스펙터 Conveyor 섹션은 벨트에만 나타난다', JSON.stringify(sections));
        }

        // (k) ★ 벨트 편집이 실제 물리 거동을 바꾼다 — 방향을 뒤집으면 흐름이 뒤집힌다.
        //     벨트 기하는 바인딩 생성 시점에 고정되므로, updateConveyor가 엔티티를
        //     재빌드해 새 바인딩을 만들지 않으면 이 어서션이 실패한다.
        // 측정을 벨트 위상에서 완전히 떼어낸다: 편집 후 item_b를 벨트 한가운데에
        // **명시적으로** 올려놓고 잰다. 그러지 않으면 이 시점의 item_b가 이미 라인 끝
        // 근처일 수 있어(앞 어서션들이 시간을 소비한다) 관측 창 안에 벨트를 벗어난다
        // — 실측으로 재현된 레이스다(vx≈0, onBelt=false).
        await page.evaluate(
          ({ speed, seat, park }) => {
            window.__sim.editor.updateConveyor('belt', {
              direction: [-1, 0, 0],
              speedMps: speed,
              recycle: true,
            });
            // 앞 어서션이 item_a를 라인에 다시 올려 두었다 — 치우지 않으면 두 상자가
            // 같은 자리를 두고 부딪혀 item_b가 벨트 밖으로 밀려난다(실측 onBelt=false).
            window.__sim.editor.updateTransform('item_a', { position: park });
            window.__sim.editor.updateTransform('item_b', { position: seat });
            window.__sim.engine.play();
          },
          { speed: CPP_EDITED_SPEED_MPS, seat: CPP_BELT_REVERSE_SEAT_POSITION, park: CPP_BELT_PARK_POSITION },
        );
        // 판정은 **변위가 아니라 속도**로 한다. 벨트 런웨이는 0.44 m뿐이라, 관측
        // 시작 시점의 위상에 따라 사물이 반대쪽 끝으로 떨어져 변위 임계값을 못 채울 수
        // 있다(= 위상 레이스). 벨트가 지정하는 것은 속도이므로 속도를 보는 것이 정확하다.
        await page.waitForTimeout(CPP_EDIT_OBSERVE_SEC * 1000);
        const reversed = await page.evaluate((expected) => {
          const w = window.__sim.world;
          const b = w.bodiesOfEntity('item_b')[0];
          if (b === undefined) return null;
          const v = w.getLinearVelocity(b);
          const p = w.getPose(b).position;
          return { vx: v[0], expected, y: p[1], x: p[0], onBelt: p[1] > 0.03 };
        }, CPP_EDITED_SPEED_MPS);
        const reversedOk =
          reversed !== null &&
          reversed.onBelt &&
          reversed.vx < 0 &&
          Math.abs(Math.abs(reversed.vx) - CPP_EDITED_SPEED_MPS) < CPP_SPEED_TOLERANCE_MPS;
        if (reversedOk) {
          pass('conveyor: 인스펙터/파사드 편집이 벨트 거동을 바꾼다 ★ (방향 반전 + 증속)',
            `item_b vx=${reversed.vx.toFixed(3)} m/s (기대 −${CPP_EDITED_SPEED_MPS})`);
        } else {
          fail('conveyor: 인스펙터/파사드 편집이 벨트 거동을 바꾼다 ★ (방향 반전 + 증속)',
            `${JSON.stringify(reversed)} — 바인딩이 재생성되지 않았을 수 있다`);
        }
      }
    }

    // ── ㄱ자 라인 셀 — 코너 이송 + 로봇 3종 스테이션 ────────────────
    //
    // 직각으로 이어진 벨트 2개 위에서 손이 서로 다른 로봇 3대가 차례로 일한다.
    // 동시 동작은 이 엔진이 표현할 수 없다(선형 1-step 모델) — 대신 컨베이어가 매 물리
    // tick 계속 돌아 "라인은 흐르고 로봇은 순서대로 일한다"가 된다.
    if (expectArg === 'l-line-cell') {
      const initial = await page.evaluate(() => {
        const p = window.__sim?.player;
        return p ? { status: p.status, stepCount: p.stepCount } : null;
      });
      if (initial?.status === 'idle' && initial.stepCount === LL_STEP_COUNT) {
        pass(`l-line: sequence loaded, no autoplay (idle, ${LL_STEP_COUNT} steps)`);
      } else {
        fail(`l-line: sequence loaded, no autoplay (idle, ${LL_STEP_COUNT} steps)`,
          `initial=${JSON.stringify(initial)}`);
      }

      if (!initial) {
        fail('l-line: interaction checks skipped', 'player facade missing');
      } else {
        // ★ 공통 검사(sim time advances)가 재생 전에 이미 물리를 흘려보냈다 — 그동안 벨트가
        // 상자를 앞으로 실어 날라 라인 위상이 바뀐다. 그 상태로 재생하면 press 배리어가
        // 이미 지나간 상자를 기다리다 죽는다(실측: step 14에서 45초 정지).
        // 되감기(orchestrator.stop)로는 부족하다 — 좌표는 돌아와도 Rapier 솔버의 warm-start
        // 상태가 남아 결정론이 복원되지 않는다(A/B 실측). **페이지를 새로 열어 새 월드에서** 한다.
        await page.goto(url, { waitUntil: 'load' });
        await page.waitForFunction(() => window.__sim !== undefined, undefined, { timeout: 15000 });

        // 놓는 순간 상자가 아직 손에 있는지 — 물리 tick 위에서 표본한다(rAF는 최대 24 tick 늦다)
        await page.evaluate(
          ({ steps, item }) => {
            window.__llGaps = {};
            window.__sim.engine.onPhysicsTick(() => {
              const p = window.__sim.player;
              if (!p) return;
              const k = steps.indexOf(p.currentStepIndex);
              if (k < 0 || window.__llGaps[String(k)] !== undefined) return;
              const w = window.__sim.world;
              const body = w.bodiesOfEntity(item)[0];
              if (body === undefined) return;
              const q = w.getPose(body).position;
              // 그 순간 움직이던 로봇의 손 = 상자에 가장 가까운 로봇 링크
              let best = Infinity;
              for (const r of ['picker', 'palletizer']) {
                for (const bid of w.bodiesOfEntity(r)) {
                  const a = w.getPose(bid).position;
                  const d = Math.hypot(a[0] - q[0], a[1] - q[1], a[2] - q[2]);
                  if (d < best) best = d;
                }
              }
              window.__llGaps[String(k)] = best;
            });
          },
          { steps: LL_PLACE_STEP_INDEXES, item: LL_FINAL_ITEM },
        );

        const last = await playAndAwaitDone(page, LL_SIM_BUDGET_SEC);
        const history = await page.evaluate(
          (limit) => window.__sim.collision.recent(limit),
          HISTORY_FETCH_LIMIT,
        );

        // (a) 완주
        if (last.status === 'done') {
          pass('l-line: sequence done', `elapsed=${last.elapsedSimSec.toFixed(2)}s step=${last.index}`);
        } else {
          fail('l-line: sequence done', `last=${JSON.stringify(last)}`);
        }

        // (b) ★ 세 로봇이 각자 자기 상자를 실제로 만졌다 — 이게 "일했다"의 최소 증거다
        const idle = LL_STATIONS.filter(
          (st) => !history.some((e) => e.phase === 'start' && isPair(e, st.robot, st.item)),
        );
        if (idle.length === 0) {
          pass('l-line: 로봇 3종이 각자 스테이션에서 자기 상자를 만진다 ★',
            LL_STATIONS.map((st) => `${st.robot}(${st.role})×${st.item}`).join(' '));
        } else {
          fail('l-line: 로봇 3종이 각자 스테이션에서 자기 상자를 만진다 ★',
            `무접촉=${JSON.stringify(idle.map((s2) => `${s2.robot}×${s2.item}`))} ` +
            `pairs=${JSON.stringify([...new Set(history.map((e) => `${e.a}×${e.b}`))])}`);
        }

        // (c) ★ 상자가 인계 패드를 거쳐 팔레트에 **중심 좌표로** 안착한다.
        //     센서 이벤트를 쓰지 않는 이유는 위 상수 주석 참조.
        const placement = await page.evaluate((item) => {
          const w = window.__sim.world;
          const spec = window.__sim.editor.serialize();
          const zoneOf = (id) => {
            const e = spec.entities.find((x) => x.id === id);
            const h = e?.physics?.colliders?.[0]?.shape?.halfExtents ?? null;
            return h === null ? null : { p: e.transform.position, h };
          };
          const inside = (pos, z) =>
            z !== null && Math.abs(pos[0] - z.p[0]) <= z.h[0] && Math.abs(pos[2] - z.p[2]) <= z.h[2];
          const body = w.bodiesOfEntity(item)[0];
          if (body === undefined) return null;
          const pos = w.getPose(body).position;
          return {
            pos: [pos[0], pos[1], pos[2]],
            inPallet: inside(pos, zoneOf('zone_pallet')),
            inHandoff: inside(pos, zoneOf('zone_handoff')),
          };
        }, LL_FINAL_ITEM);
        // 팔레트 안 + 인계 패드 **밖** 둘 다 요구한다 — 인계 자리에 그대로 있으면
        // palletizer가 일하지 않은 것이고, 존이 겹쳐 있으면 둘 다 참이 된다.
        if (placement?.inPallet === true && placement.inHandoff === false) {
          pass('l-line: 상자가 인계 패드를 거쳐 팔레트에 안착한다 ★ (중심 좌표 판정)',
            `${LL_FINAL_ITEM}=[${placement.pos.map((v) => v.toFixed(3)).join(', ')}]`);
        } else {
          fail('l-line: 상자가 인계 패드를 거쳐 팔레트에 안착한다 ★ (중심 좌표 판정)',
            JSON.stringify(placement));
        }

        // (d) 놓는 순간 상자가 아직 손에 있었다 — 옮긴 것과 벨트가 밀어 놓은 것을 가른다
        const gaps = await page.evaluate(() => window.__llGaps);
        const dropped = LL_PLACE_STEP_INDEXES.map((_, k) => String(k)).filter(
          (k) => !(gaps?.[k] <= LL_MAX_RELEASE_GAP_M),
        );
        if (dropped.length === 0) {
          pass('l-line: 두 번의 놓기에서 상자가 아직 손 안에 있다 (이송 중 낙하 없음)',
            LL_PLACE_STEP_INDEXES.map((_, k) => `${(gaps[String(k)] * 100).toFixed(1)}cm`).join(' / '));
        } else {
          fail('l-line: 두 번의 놓기에서 상자가 아직 손 안에 있다 (이송 중 낙하 없음)',
            `${JSON.stringify(gaps)} — 손에서 빠졌거나 LL_PLACE_STEP_INDEXES가 시퀀스와 어긋났다`);
        }

        // (e) ★ 로봇이 라인 설비를 건드리지 않는다. 분류(target/unexpected)를 거치지 않고
        //     **원시 쌍**으로 센다 — 배리어 선언이 화이트리스트를 만들어 숫자를 가릴 수 있다.
        const hits = [];
        for (const robot of LL_ROBOTS) {
          for (const other of [...LL_STATICS, ...LL_ROBOTS.filter((r) => r !== robot)]) {
            const n = history.filter((e) => e.phase === 'start' && isPair(e, robot, other)).length;
            if (n > 0) hits.push(`${robot}×${other}:${n}`);
          }
        }
        if (hits.length === 0) {
          pass('l-line: 로봇이 벨트·가이드·다른 로봇을 건드리지 않는다 ★ (원시 쌍 집계)');
        } else {
          fail('l-line: 로봇이 벨트·가이드·다른 로봇을 건드리지 않는다 ★ (원시 쌍 집계)',
            hits.join(' '));
        }

        // (f) 코너가 실제로 물건을 돌렸다 — 벨트 A에서 출발한 상자가 벨트 B 레인에 있다.
        //     라인이 죽어 있으면 상자들이 출발 x 근처에 남는다.
        const turned = await page.evaluate(() => {
          const w = window.__sim.world;
          return ['item_a', 'item_b', 'item_c'].filter((id) => {
            const b = w.bodiesOfEntity(id)[0];
            if (b === undefined) return false;
            const p = w.getPose(b).position;
            return p[2] > 0.2; // 벨트 A는 z≈0 — z가 커졌다면 코너를 돈 것이다
          }).length;
        });
        if (turned === 3) {
          pass('l-line: 상자 3개가 모두 직각 코너를 돌아 벨트 B 구간으로 넘어갔다 ★',
            `z>0.2인 상자 ${turned}개`);
        } else {
          fail('l-line: 상자 3개가 모두 직각 코너를 돌아 벨트 B 구간으로 넘어갔다 ★',
            `z>0.2인 상자 ${turned}개 (기대 3)`);
        }
      }
    }

    // ── 라이브러리 로봇 3종 (arm-6 / SCARA-4 / Cobot-7) ─────────────
    //
    // 손 모양이 서로 다르다: 평행 2지 · 흡착 패드 · 3지 클로. URDF가 파싱만 되고
    // 실제로 구동되지 않는 회귀를 잡는다 — 파싱은 라이브러리 카드를 띄우기에 충분하지만
    // 시퀀스는 관절이 움직여야 성립한다.
    if (expectArg === 'robot-library') {
      const rows = [];
      for (let i = 0; i < RL_ROBOTS.length; i += 1) {
        const r = RL_ROBOTS[i];
        const x = RL_SPOT_X0 + i * RL_SPOT_DX;

        // 라이브러리 카드 드롭과 동일 경로 (placeTemplate) — 로봇은 URDF 로드라 async
        const id = await page.evaluate(
          async ({ key, px, pz }) => window.__sim.editor.placeTemplate(key, [px, 0, pz]),
          { key: r.key, px: x, pz: RL_SPOT_Z },
        );
        await page.waitForFunction(
          (i2) => window.__sim.robots.ids().includes(i2),
          id,
          { timeout: RL_PLACE_DEADLINE_MS },
        );

        const before = await page.evaluate((i2) => ({
          joints: window.__sim.robots.joints(i2).map((j) => j.name),
          links: window.__sim.robots.linkPoses(i2).map((p) => [...p.position]),
          values: window.__sim.robots.readJoints(i2),
        }), id);

        // 관절을 움직이면 말단(베이스에서 가장 먼 링크)이 실제로 이동하는가
        await page.evaluate(
          ({ i2, j, v }) => window.__sim.robots.setJoint(i2, j, v),
          { i2: id, j: r.driveJoint, v: (before.values[r.driveJoint] ?? 0) + r.driveDelta },
        );
        await awaitSimAdvance(page, 0.1);
        const afterLinks = await page.evaluate(
          (i2) => window.__sim.robots.linkPoses(i2).map((p) => [...p.position]),
          id,
        );
        const endMove = Math.max(
          ...before.links.map((p, k) => {
            const q = afterLinks[k];
            return q === undefined ? 0 : Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
          }),
        );

        // 그리퍼가 손을 실제로 여닫는가 — **링크 변위**로 잰다(관절 구동 판정과 같은 방식).
        // 손 끝을 "원점에서 먼 링크"로 고르면 로봇이 원점에서 떨어져 놓인 순간 엉뚱한
        // 링크를 집는다(실측: 로봇을 x=2.6에 놓자 handDelta가 0으로 나왔다). 어느 링크가
        // 손인지 게이트가 알 필요 없다 — 그리퍼 관절을 바꿨을 때 **가장 많이 움직인 링크**가
        // 손이고, 그 변위가 0이면 손이 안 움직인 것이다.
        const setGripper = async (value) => {
          await page.evaluate(
            ({ i2, joints, v }) => {
              for (const j of joints) window.__sim.robots.setJoint(i2, j, v);
            },
            { i2: id, joints: r.gripperJoints, v: value },
          );
          await awaitSimAdvance(page, 0.1);
          return page.evaluate(
            (i2) =>
              window.__sim.robots
                .linkPoses(i2)
                .map((p) => ({ position: [...p.position], rotation: [...p.rotation] })),
            id,
          );
        };
        const opened = await setGripper(r.gripperOpen);
        const closed = await setGripper(r.gripperClose);
        const handDelta = Math.max(
          ...opened.map((p, k) => {
            const q = closed[k];
            return q === undefined
              ? 0
              : Math.hypot(p.position[0] - q.position[0], p.position[1] - q.position[1], p.position[2] - q.position[2]);
          }),
        );
        // 쿼터니언 사이각 = 2·acos(|dot|) — revolute 손가락의 유일한 신호다
        const handRot = Math.max(
          ...opened.map((p, k) => {
            const q = closed[k];
            if (q === undefined) return 0;
            const dot = Math.abs(
              p.rotation[0] * q.rotation[0] + p.rotation[1] * q.rotation[1] +
              p.rotation[2] * q.rotation[2] + p.rotation[3] * q.rotation[3],
            );
            return 2 * Math.acos(Math.min(1, dot));
          }),
        );

        // home 포즈에서 바닥 아래로 내려간 링크가 없는가 (§2.11 — kinematic은 자가 교정이 없다)
        await page.evaluate((i2) => window.__sim.robots.ids().includes(i2), id);
        const minY = Math.min(...before.links.map((p) => p[1]));

        rows.push({
          key: r.key,
          id,
          hand: r.handKind,
          links: before.links.length,
          joints: before.joints.length,
          endMove: +endMove.toFixed(4),
          handDelta: +handDelta.toFixed(4),
          handRot: +handRot.toFixed(4),
          minY: +minY.toFixed(4),
          ok:
            before.links.length >= r.minLinks &&
            before.joints.includes(r.driveJoint) &&
            r.gripperJoints.every((j) => before.joints.includes(j)) &&
            endMove >= RL_MIN_END_MOVE_M &&
            (handDelta >= RL_MIN_HAND_DELTA_M || handRot >= RL_MIN_HAND_ROT_RAD) &&
            minY >= -RL_MAX_UNDERGROUND_M,
        });
      }

      const summary = rows.map((r) => `${r.key}(${r.hand}) 링크${r.links} 구동${r.endMove}m 손 ${r.handDelta}m/${r.handRot}rad`).join(' / ');
      if (rows.every((r) => r.ok)) {
        pass('robot-library: 로봇 3종이 서고 · 관절이 말단을 움직이고 · 손이 여닫힌다 ★', summary);
      } else {
        fail('robot-library: 로봇 3종이 서고 · 관절이 말단을 움직이고 · 손이 여닫힌다 ★',
          JSON.stringify(rows.filter((r) => !r.ok)));
      }

      // 세 로봇이 한 씬에 공존한다 (id 충돌·URDF 캐시 오염 없음)
      const coexist = await page.evaluate(() => window.__sim.robots.ids().length);
      if (coexist >= RL_ROBOTS.length + 1) {
        pass('robot-library: 세 로봇이 기존 로봇과 한 씬에 공존한다', `robots=${coexist}`);
      } else {
        fail('robot-library: 세 로봇이 기존 로봇과 한 씬에 공존한다', `robots=${coexist}`);
      }

      // 편집 스펙이 저장 가능한 상태로 남는가 (§2.11 — 문서로 보존된다)
      const serialized = await page.evaluate(() => {
        const spec = window.__sim.editor.serialize();
        return spec.entities.filter((e) => e.type === 'robot').map((e) => ({ id: e.id, urdf: e.urdf }));
      });
      const urdfOk = serialized.length >= RL_ROBOTS.length + 1 && serialized.every((e) => typeof e.urdf === 'string' && e.urdf.length > 0);
      if (urdfOk) {
        pass('robot-library: 로봇이 urdf 경로를 가진 채 직렬화된다 (저장/재로드 가능)',
          serialized.map((e) => e.urdf.split('/').pop()).join(' '));
      } else {
        fail('robot-library: 로봇이 urdf 경로를 가진 채 직렬화된다 (저장/재로드 가능)',
          JSON.stringify(serialized));
      }
    }

    // ── 3D 파일 임포트 (UX_DESIGN §4.4) ──────────────────────────────
    //
    // 이 경로는 브라우저에서 한 번도 검증된 적이 없었다. 계측은 손으로 만든 픽스처
    // gate-box.{glb,stl,obj}로 한다 — 세 변이 모두 다르고 원점에서 어긋나 있어야
    // 스케일·Up-axis·피벗 재정렬이 **각각** 관측된다(정육면체·원점중심이면 셋 다 무시해도 초록).
    if (expectArg === 'mesh-import') {
      // (a) 포맷 3종이 같은 솔리드로 파싱된다 — 로더별 회귀 가드
      const parseRows = [];
      for (const fx of IMP_FIXTURES) {
        const phase = await openImportFixture(page, fx.file);
        parseRows.push({ file: fx.file, ...phase });
        await closeImportDialog(page);
      }
      const parseOk = parseRows.every(
        (r, i) =>
          r.format === IMP_FIXTURES[i].label &&
          Number(String(r.triangles).replace(/,/g, '')) === IMP_FIXTURE_TRIANGLES &&
          r.size === IMP_FIXTURE_SIZE_LABEL &&
          r.errorVisible === false &&
          r.confirmDisabled === false,
      );
      if (parseOk) {
        pass('import: glb/stl/obj 3종이 같은 삼각형 수·치수로 파싱된다 ★',
          `triangles=${IMP_FIXTURE_TRIANGLES} size=${IMP_FIXTURE_SIZE_LABEL}`);
      } else {
        fail('import: glb/stl/obj 3종이 같은 삼각형 수·치수로 파싱된다 ★', JSON.stringify(parseRows));
      }

      // (b) 스케일이 리드아웃과 collider **양쪽**에 반영된다.
      //     둘은 독립 경로다(readout=formatBboxSizeM, collider=prepareForScene) —
      //     한쪽만 보면 다른 쪽이 스케일을 무시해도 통과한다.
      await importFixture(page, 'gate-box.glb',
        { id: IMP_ID.aabb, scale: 1, upAxis: 'y', collider: 'aabb', kind: 'object' });
      await parkImported(page, IMP_ID.aabb, 0);
      const halfPhase = await openImportFixture(page, 'gate-box.glb');
      await fillImportForm(page,
        { id: IMP_ID.aabbHalf, scale: IMP_SCALE_HALF, upAxis: 'y', collider: 'aabb', kind: 'object' });
      const halfSizeLabel = await page.evaluate(
        () => document.querySelector('[data-testid="import-size"]')?.textContent ?? null,
      );
      await confirmImport(page);
      await parkImported(page, IMP_ID.aabbHalf, 1);

      const fullSpec = await importedSpec(page, IMP_ID.aabb);
      const halfSpec = await importedSpec(page, IMP_ID.aabbHalf);
      const halfOf = (spec) => spec?.physics?.colliders?.[0]?.shape?.halfExtents ?? null;
      const fullHalf = halfOf(fullSpec);
      const halfHalf = halfOf(halfSpec);
      const absOk =
        fullHalf !== null &&
        IMP_FIXTURE_HALF_EXTENTS.every((v, i) => Math.abs(fullHalf[i] - v) < IMP_HALF_TOLERANCE_M);
      const ratioOk =
        fullHalf !== null && halfHalf !== null &&
        fullHalf.every((v, i) => Math.abs(halfHalf[i] / v - IMP_SCALE_HALF) < IMP_RATIO_TOLERANCE);
      if (absOk && ratioOk && halfSizeLabel === IMP_FIXTURE_HALF_SIZE_LABEL) {
        pass('import: 스케일이 리드아웃과 collider 양쪽에 반영된다 (독립 경로 교차 검증)',
          `half=${JSON.stringify(fullHalf)} 비율=${IMP_SCALE_HALF} readout=${halfSizeLabel}`);
      } else {
        fail('import: 스케일이 리드아웃과 collider 양쪽에 반영된다 (독립 경로 교차 검증)',
          `full=${JSON.stringify(fullHalf)} half=${JSON.stringify(halfHalf)} readout=${halfSizeLabel} (기대 ${IMP_FIXTURE_HALF_SIZE_LABEL}) phase=${JSON.stringify(halfPhase)}`);
      }

      // (c) Z-up → Y-up 변환이 halfExtents y/z를 **교환**한다.
      //     "그냥 작아지는" 회귀와 구분하려면 값이 바뀌는 게 아니라 자리가 바뀌어야 한다.
      await importFixture(page, 'gate-box.glb',
        { id: IMP_ID.aabbZ, scale: 1, upAxis: 'z', collider: 'aabb', kind: 'object' });
      await parkImported(page, IMP_ID.aabbZ, 2);
      const zHalf = halfOf(await importedSpec(page, IMP_ID.aabbZ));
      const [ex, ey, ez] = IMP_FIXTURE_HALF_EXTENTS;
      const swapOk =
        zHalf !== null &&
        Math.abs(zHalf[0] - ex) < IMP_HALF_TOLERANCE_M &&
        Math.abs(zHalf[1] - ez) < IMP_HALF_TOLERANCE_M &&
        Math.abs(zHalf[2] - ey) < IMP_HALF_TOLERANCE_M;
      if (swapOk) {
        pass('import: Z-up 선택이 축 변환을 실제로 적용한다 ★ (halfExtents y↔z 교환)',
          `y-up=${JSON.stringify(IMP_FIXTURE_HALF_EXTENTS)} → z-up=${JSON.stringify(zHalf)}`);
      } else {
        fail('import: Z-up 선택이 축 변환을 실제로 적용한다 ★ (halfExtents y↔z 교환)',
          `z-up half=${JSON.stringify(zHalf)} (기대 [${ex}, ${ez}, ${ey}])`);
      }

      // (d) 피벗이 bbox **바닥 중심**으로 재정렬된다 → y=0에서 지면 안착 (§2.11).
      //     픽스처가 원점 밖이라 x/z 성분도 항등이 아니다 — 재정렬이 없으면 시각 중심이
      //     bbox 중심 [0.25, 0.15, −0.25]만큼 어긋난다.
      const probe = await page.evaluate((i) => {
        window.__sim.editor.select(i);
        return window.__sim.editor.anchorProbe();
      }, IMP_ID.aabb);
      // 판정은 **스펙**에서 한다 — 동적 바디는 임포트 직후부터 낙하·정착하므로 물리
      // pose로 재면 관측 시점에 좌우된다(경주). collider offset.y = halfY 이면 바디
      // 원점이 bbox **바닥**이라는 뜻이고, 피벗이 bbox 중심이면 이 값이 0이 된다.
      const offsetY = fullSpec?.physics?.colliders?.[0]?.offset?.position?.[1] ?? null;
      const pivotOk =
        fullSpec !== null &&
        Math.abs(fullSpec.transform.position[1]) < IMP_SETTLE_TOLERANCE_M &&
        offsetY !== null &&
        Math.abs(offsetY - IMP_FIXTURE_HALF_EXTENTS[1]) < IMP_HALF_TOLERANCE_M &&
        probe !== null &&
        Math.abs(probe.visualCenter[0] - probe.rootOrigin[0]) < IMP_VISUAL_TOLERANCE_M &&
        Math.abs(probe.visualCenter[2] - probe.rootOrigin[2]) < IMP_VISUAL_TOLERANCE_M;
      if (pivotOk) {
        pass('import: 피벗이 bbox 바닥 중심으로 재정렬된다 ★ (배치 y=0 · collider offset=halfY)',
          `배치y=${fullSpec.transform.position[1]} offsetY=${offsetY.toFixed(4)} (bbox 중심이면 0)`);
      } else {
        fail('import: 피벗이 bbox 바닥 중심으로 재정렬된다 ★ (배치 y=0 · collider offset=halfY)',
          `spec.y=${fullSpec?.transform?.position?.[1]} offsetY=${offsetY} probe=${JSON.stringify(probe)}`);
      }

      // (e) 쌍 필터가 양방향으로 선언됐는가 (CLAUDE.md §5) — 안 그러면 로봇이 임포트
      //     사물을 건드려도 충돌 로그가 0건이 되는 조용한 회귀가 된다.
      const impCollider = fullSpec?.physics?.colliders?.[0] ?? null;
      const pairOk =
        impCollider !== null &&
        impCollider.collidesWith?.includes('ROBOT') === true &&
        impCollider.collidesWith?.includes('ENV') === true &&
        impCollider.emitEvents === true;
      if (pairOk) {
        pass('import: 임포트 사물이 ROBOT·ENV와 쌍이 성립하고 이벤트를 낸다 (§5 양쪽 규칙)',
          `collidesWith=${JSON.stringify(impCollider.collidesWith)}`);
      } else {
        fail('import: 임포트 사물이 ROBOT·ENV와 쌍이 성립하고 이벤트를 낸다 (§5 양쪽 규칙)',
          JSON.stringify(impCollider));
      }

      // (f) ★ 유령이 아니다 — convexHull 엔티티를 들어올려 떨어뜨리면 실제로 낙하하고
      //     바닥과 contact를 낸다. 정착 y ≈ 0 vs ≈ halfY(0.10)가 "피벗이 바닥 중심"과
      //     "피벗이 bbox 중심"을 정확히 가른다.
      await importFixture(page, 'gate-box.glb',
        { id: IMP_ID.hullY, scale: 1, upAxis: 'y', collider: 'hull', kind: 'object' });
      await parkImported(page, IMP_ID.hullY, 3);
      const bodyCount = await page.evaluate(
        (i) => window.__sim.world.bodiesOfEntity(i).length, IMP_ID.hullY);
      const { settledY, fell } = await dropAndSettle(page, IMP_ID.hullY, IMP_DROP_HEIGHT_M);
      const groundHits = await page.evaluate(
        ({ i, limit }) =>
          window.__sim.collision
            .recent(limit)
            .filter((e) => e.phase === 'start' && e.kind === 'contact' &&
              ((e.a === i && e.b === '__ground') || (e.b === i && e.a === '__ground'))).length,
        { i: IMP_ID.hullY, limit: HISTORY_FETCH_LIMIT },
      );
      // 정착 **높이**는 보지 않는다 — 0.4m에서 떨어뜨린 직육면체는 기울어 눕는 것이
      // 정상이고, 그건 임포트 결함이 아니다. 피벗 판정은 (d)가 스펙에서 이미 했다.
      // 여기서 묻는 것은 셋뿐이다: 물리 바디가 있는가 / 실제로 떨어졌는가 /
      // 바닥과 접촉 이벤트를 냈는가. 그리고 바닥을 뚫고 지하로 가지 않았는가(§2.11).
      const ghostOk =
        bodyCount === 1 &&
        fell >= IMP_MIN_FALL_M &&
        settledY !== null &&
        settledY > -IMP_MAX_SINK_M &&
        groundHits >= 1;
      if (ghostOk) {
        pass('import: 임포트 사물이 실제로 낙하하고 바닥과 충돌한다 ★ (그림만 있는 유령 아님)',
          `bodies=${bodyCount} 낙하=${fell.toFixed(3)}m 정착y=${settledY.toFixed(4)} ground접촉=${groundHits}건`);
      } else {
        fail('import: 임포트 사물이 실제로 낙하하고 바닥과 충돌한다 ★ (그림만 있는 유령 아님)',
          `bodies=${bodyCount} 낙하=${fell} 정착y=${settledY} ground접촉=${groundHits}`);
      }

      // (g) trimesh 전략은 Static을 강제하고 실제로 단단하다 — 그 위의 상자가 지지된다.
      const trimeshPhase = await openImportFixture(page, 'gate-box.glb');
      await fillImportForm(page,
        { id: IMP_ID.trimesh, scale: 1, upAxis: 'y', collider: 'trimesh' });
      const trimeshForm = await page.evaluate(() => ({
        objectKindDisabled: document.querySelector('[data-testid="import-kind-object"]')?.disabled ?? null,
        noteVisible: (() => {
          const n = document.querySelector('[data-testid="import-trimesh-note"]');
          return n !== null && getComputedStyle(n).display !== 'none';
        })(),
      }));
      await confirmImport(page);
      await parkImported(page, IMP_ID.trimesh, 4);
      const trimeshSpec = await importedSpec(page, IMP_ID.trimesh);
      const boxId = await page.evaluate(
        async ({ i, y }) => {
          const t = window.__sim.editor.serialize().entities.find((e) => e.id === i).transform.position;
          return window.__sim.editor.placeTemplate('box', [t[0], y, t[2]]);
        },
        { i: IMP_ID.trimesh, y: IMP_DROP_HEIGHT_M },
      );
      const supported = await dropAndSettle(page, boxId, IMP_DROP_HEIGHT_M);
      const trimeshOk =
        trimeshForm.objectKindDisabled === true &&
        trimeshForm.noteVisible === true &&
        trimeshSpec?.type === 'static' &&
        trimeshSpec?.physics?.bodyType === 'fixed' &&
        trimeshSpec?.physics?.colliders?.[0]?.shape?.kind === 'trimesh' &&
        supported.settledY !== null &&
        supported.settledY > IMP_TRIMESH_SUPPORT_MIN_Y_M;
      if (trimeshOk) {
        pass('import: trimesh는 Static 강제 + 실제로 단단하다 ★ (위의 상자가 지지된다)',
          `상자 정착y=${supported.settledY.toFixed(3)} > ${IMP_TRIMESH_SUPPORT_MIN_Y_M}`);
      } else {
        fail('import: trimesh는 Static 강제 + 실제로 단단하다 ★ (위의 상자가 지지된다)',
          `form=${JSON.stringify(trimeshForm)} spec=${JSON.stringify(trimeshSpec?.physics)} 상자=${JSON.stringify(supported)} phase=${JSON.stringify(trimeshPhase)}`);
      }

      // (h) 실패는 시끄럽게 — 미지원 확장자/손상 파일은 한국어 사유로 거부되고
      //     씬에 아무것도 들어가지 않는다. 조용한 no-op은 사용자가 원인을 알 수 없다.
      const beforeBad = await page.evaluate(() => window.__sim.editor.entityIds().length);
      const badPhase = await openImportText(page, IMP_BAD_FILE.name, IMP_BAD_FILE.body);
      await closeImportDialog(page);
      const corruptPhase = await openImportText(page, IMP_CORRUPT_FILE.name, IMP_CORRUPT_FILE.body);
      await closeImportDialog(page);
      const afterBad = await page.evaluate(() => window.__sim.editor.entityIds().length);
      const failLoudOk =
        badPhase.errorVisible === true && badPhase.confirmDisabled === true &&
        corruptPhase.errorVisible === true && corruptPhase.confirmDisabled === true &&
        beforeBad === afterBad;
      if (failLoudOk) {
        pass('import: 미지원·손상 파일은 사유와 함께 거부되고 씬은 그대로다',
          `"${String(badPhase.errorText).slice(0, 40)}" / "${String(corruptPhase.errorText).slice(0, 40)}"`);
      } else {
        fail('import: 미지원·손상 파일은 사유와 함께 거부되고 씬은 그대로다',
          `bad=${JSON.stringify(badPhase)} corrupt=${JSON.stringify(corruptPhase)} 엔티티 ${beforeBad}→${afterBad}`);
      }

      // (i) 세션 한정 에셋이 씬 재빌드(undo/redo)를 넘어 살아남는다.
      //     MeshAssetStore가 앱 수명인 이유가 정확히 이것이다 — 씬 수명으로 강등되면
      //     재로드 시 "메시 에셋을 해석할 수 없습니다"로 씬 빌드가 통째로 실패한다.
      const refsBefore = await page.evaluate(() => window.__sim.meshImport.assetRefs().length);
      await page.evaluate(() => window.__sim.history.undo());
      await page.evaluate(() => window.__sim.history.redo());
      const survived = await page.evaluate(
        (i) => window.__sim.editor.entityIds().includes(i), IMP_ID.hullY);
      const bodiesAfter = await page.evaluate(
        (i) => window.__sim.world.bodiesOfEntity(i).length, IMP_ID.hullY);
      if (survived && bodiesAfter === 1 && refsBefore > 0) {
        pass('import: 임포트 에셋이 undo/redo 씬 재빌드를 넘어 살아남는다',
          `asset refs=${refsBefore} bodies=${bodiesAfter}`);
      } else {
        fail('import: 임포트 에셋이 undo/redo 씬 재빌드를 넘어 살아남는다',
          `survived=${survived} bodies=${bodiesAfter} refs=${refsBefore}`);
      }

      // (j) 카탈로그 스모크 — 배포한 다운로드 모델이 전부 파싱된다.
      //     수치는 재지 않는다(치수를 우리가 소유하지 않는다). 깨진 에셋 배포 방지용.
      const catalogRows = [];
      for (const name of IMP_CATALOG) {
        const phase = await openImportFixture(page, name);
        catalogRows.push({
          name,
          ok: phase.errorVisible === false && Number(String(phase.triangles).replace(/,/g, '')) > 0,
          triangles: phase.triangles,
          error: phase.errorText,
        });
        await closeImportDialog(page);
      }
      if (catalogRows.every((r) => r.ok)) {
        pass(`import: 동봉한 다운로드 모델 ${IMP_CATALOG.length}종이 전부 파싱된다 (카탈로그 스모크)`,
          catalogRows.map((r) => `${r.name}=${r.triangles}`).join(' '));
      } else {
        fail(`import: 동봉한 다운로드 모델 ${IMP_CATALOG.length}종이 전부 파싱된다 (카탈로그 스모크)`,
          JSON.stringify(catalogRows.filter((r) => !r.ok)));
      }
    }

    // ── 뷰포트 편집 UX — 바닥 하한 · 방향키 이동 · 선택 HUD · 키 소유권 ──
    //
    // 사용자 보고 3건이 여기 모인다:
    //   (1) "물체를 넣을 때 바닥 아래 지하로 위치돼 사라진다" → 하한 클램프
    //   (2) "방향키로 어떻게 움직이는지, 너비 조정을 어떻게 하는지 알기 어렵다" → 선택 HUD
    //   (3) `→`가 재생 Step과 오브젝트 이동을 **동시에** 일으키던 이중 소유 (§2.10 위반)
    if (expectArg === 'viewport-edit') {
      const specPosition = (id) =>
        page.evaluate(
          (i) =>
            window.__sim.editor.serialize().entities.find((e) => e.id === i)?.transform.position ??
            null,
          id,
        );
      const focusViewport = () => focusViewportSlot(page);

      // 물리를 멈추고 잰다: nudge의 기준점은 **현재 살아있는 pose**여서(기즈모 드래그와
      // 같은 커밋 경로) 재생 중이면 표본 사이에 바디가 굴러 이동량 측정이 흔들린다.
      // 실제 UI도 편집 시작 시 자동 일시정지하므로(pauseForEditIfPlaying) 같은 조건이다.
      await page.evaluate(() => window.__sim.engine.pause());
      await page.evaluate((id) => window.__sim.editor.select(id), VE_TARGET_ID);
      await focusViewport();

      // (a) 선택 HUD가 나타나고 대상 id·치수 스테퍼를 보여준다
      const hud = await page.evaluate(() => {
        const el = document.querySelector('[data-testid="selection-hud"]');
        if (!el) return null;
        return {
          visible: getComputedStyle(el).display !== 'none',
          name: el.querySelector('[data-testid="selection-hud-name"]')?.textContent ?? null,
          hasHeightStepper:
            el.querySelector('[data-testid="selection-hud-height-inc"]') !== null,
          hint: el.querySelector('[data-testid="selection-hud-hint"]')?.textContent ?? null,
        };
      });
      if (hud?.visible === true && hud.name === VE_TARGET_ID && hud.hasHeightStepper) {
        pass('viewport-edit: 선택 시 조작 HUD 표시 (대상 id + 치수 스테퍼 + 키 안내)',
          `hint=${JSON.stringify(hud.hint)}`);
      } else {
        fail('viewport-edit: 선택 시 조작 HUD 표시 (대상 id + 치수 스테퍼 + 키 안내)',
          JSON.stringify(hud));
      }

      // (b) 방향키 이동 — 뷰포트 스코프에서 선택 오브젝트가 실제로 움직인다
      const beforeNudge = await specPosition(VE_TARGET_ID);
      await page.keyboard.press('ArrowRight');
      const afterNudge = await specPosition(VE_TARGET_ID);
      const nudgeDelta = Math.hypot(
        afterNudge[0] - beforeNudge[0],
        afterNudge[2] - beforeNudge[2],
      );
      if (nudgeDelta > VE_NUDGE_MIN_DELTA_M) {
        pass('viewport-edit: 방향키가 선택 오브젝트를 이동시킨다', `delta=${nudgeDelta.toFixed(4)}m`);
      } else {
        fail('viewport-edit: 방향키가 선택 오브젝트를 이동시킨다',
          `before=${JSON.stringify(beforeNudge)} after=${JSON.stringify(afterNudge)}`);
      }

      // (c) Shift 미세 이동이 기본 이동보다 작다
      const beforeFine = await specPosition(VE_TARGET_ID);
      await page.keyboard.down('Shift');
      await page.keyboard.press('ArrowRight');
      await page.keyboard.up('Shift');
      const afterFine = await specPosition(VE_TARGET_ID);
      const fineDelta = Math.hypot(afterFine[0] - beforeFine[0], afterFine[2] - beforeFine[2]);
      if (Math.abs(fineDelta - VE_NUDGE_FINE_M) <= VE_NUDGE_TOLERANCE_M) {
        pass('viewport-edit: Shift 병용이 미세 이동',
          `fine=${fineDelta.toFixed(4)}m (기대 ${VE_NUDGE_FINE_M}m) < coarse=${nudgeDelta.toFixed(4)}m`);
      } else {
        fail('viewport-edit: Shift 병용이 미세 이동',
          `fine=${fineDelta.toFixed(4)}m, 기대 ${VE_NUDGE_FINE_M}±${VE_NUDGE_TOLERANCE_M}m (coarse=${nudgeDelta.toFixed(4)}m)`);
      }

      // (d) ★ 이중 소유 회귀: 방향키가 재생 시퀀스 step까지 진행시키면 안 된다
      const stepIndexAfterArrows = await page.evaluate(
        () => window.__sim.player?.currentStepIndex ?? null,
      );
      if (stepIndexAfterArrows === 0 || stepIndexAfterArrows === null) {
        pass('viewport-edit: 방향키가 재생 Step을 동시에 일으키지 않는다 ★ 회귀 (§2.10)',
          `stepIndex=${stepIndexAfterArrows}`);
      } else {
        fail('viewport-edit: 방향키가 재생 Step을 동시에 일으키지 않는다 ★ 회귀 (§2.10)',
          `stepIndex=${stepIndexAfterArrows} — 라우터 밖의 두 번째 키맵이 살아 있다`);
      }

      // (e) ★ 바닥 하한: PageDown을 아무리 눌러도 지하로 내려가지 않는다
      for (let i = 0; i < VE_SINK_PRESS_COUNT; i += 1) await page.keyboard.press('PageDown');
      const sunk = await specPosition(VE_TARGET_ID);
      if (sunk[1] > 0) {
        pass('viewport-edit: 방향키로 바닥 아래에 놓을 수 없다 ★ 회귀 (지하 = 작업물 손실)',
          `y=${sunk[1].toFixed(4)}m (PageDown ×${VE_SINK_PRESS_COUNT})`);
      } else {
        fail('viewport-edit: 방향키로 바닥 아래에 놓을 수 없다 ★ 회귀 (지하 = 작업물 손실)',
          `y=${sunk[1]}`);
      }

      // (f) End = 바닥에 붙이기 — 띄운 사물을 정확히 바닥에 앉힌다
      for (let i = 0; i < VE_LIFT_PRESS_COUNT; i += 1) await page.keyboard.press('PageUp');
      const lifted = await specPosition(VE_TARGET_ID);
      await page.keyboard.press('End');
      const snapped = await specPosition(VE_TARGET_ID);
      if (lifted[1] > snapped[1] && snapped[1] > 0) {
        pass('viewport-edit: End로 바닥에 붙이기', `${lifted[1].toFixed(3)}m → ${snapped[1].toFixed(3)}m`);
      } else {
        fail('viewport-edit: End로 바닥에 붙이기',
          `lifted=${lifted[1]} snapped=${snapped[1]}`);
      }

      // (g) HUD 치수 + 버튼 → 실제로 커지고, 커진 뒤에도 바닥 위에 남는다
      const dimText = () =>
        page.evaluate(
          () =>
            document.querySelector('[data-testid="selection-hud-dim-height"]')?.textContent ?? null,
        );
      const dimBefore = Number(await dimText());
      await page.click('[data-testid="selection-hud-height-inc"]');
      await page.click('[data-testid="selection-hud-height-inc"]');
      const dimAfter = Number(await dimText());
      const posAfterResize = await specPosition(VE_TARGET_ID);
      const grew = dimAfter > dimBefore;
      const stillGrounded = posAfterResize[1] >= dimAfter / 2 - VE_GROUNDED_TOLERANCE_M;
      if (grew && stillGrounded) {
        pass('viewport-edit: HUD ± 버튼으로 치수 조정 — 커져도 바닥에 남는다',
          `높이 ${dimBefore}→${dimAfter}m, y=${posAfterResize[1].toFixed(4)}m`);
      } else {
        fail('viewport-edit: HUD ± 버튼으로 치수 조정 — 커져도 바닥에 남는다',
          `dim ${dimBefore}→${dimAfter}, y=${posAfterResize[1]}`);
      }

      // (h) 선택 해제 → HUD 숨김 (빈 상태 카드가 3D 화면을 상시 잠식하지 않는다)
      await page.evaluate(() => window.__sim.editor.select(null));
      const hudHidden = await page.evaluate(() => {
        const el = document.querySelector('[data-testid="selection-hud"]');
        return el === null || getComputedStyle(el).display === 'none';
      });
      if (hudHidden) pass('viewport-edit: 선택 해제 시 HUD 숨김');
      else fail('viewport-edit: 선택 해제 시 HUD 숨김', 'HUD가 남아 있다');

      // (i) 선택이 없으면 `→`는 규정대로 재생 Step이다 (스코프 분기 계약).
      //     Step은 노드 1개를 sim 시간만큼 재생하고 경계에서 멈추므로, 즉시 관측되는
      //     신호는 **arm 전이**(idle → running)다. 인덱스 전진은 그 뒤에 따라온다.
      await focusViewport();
      const stepBefore = await page.evaluate(() => ({
        status: window.__sim.player?.status ?? null,
        index: window.__sim.player?.currentStepIndex ?? null,
      }));
      await page.keyboard.press('ArrowRight');
      const stepDeadline = Date.now() + VE_STEP_DEADLINE_MS;
      let stepAfter = stepBefore;
      for (;;) {
        stepAfter = await page.evaluate(() => ({
          status: window.__sim.player?.status ?? null,
          index: window.__sim.player?.currentStepIndex ?? null,
        }));
        if (stepAfter.status !== stepBefore.status || stepAfter.index > stepBefore.index) break;
        if (Date.now() > stepDeadline) break;
        await page.waitForTimeout(SEQ_POLL_INTERVAL_MS);
      }
      const stepped =
        stepBefore.status === 'idle' &&
        (stepAfter.status !== 'idle' || stepAfter.index > stepBefore.index);
      if (stepped) {
        pass('viewport-edit: 선택이 없으면 →는 규정대로 Step (UX_DESIGN §9)',
          `${JSON.stringify(stepBefore)} → ${JSON.stringify(stepAfter)}`);
      } else {
        fail('viewport-edit: 선택이 없으면 →는 규정대로 Step (UX_DESIGN §9)',
          `before=${JSON.stringify(stepBefore)} after=${JSON.stringify(stepAfter)}`);
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
