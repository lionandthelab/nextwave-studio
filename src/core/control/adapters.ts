// core/control/adapters.ts — RobotRegistry/CollisionMonitor → player 인터페이스 어댑터
//
// steps.ts의 좁은 인터페이스(RobotApi/CollisionQuery)와 core 구현체(RobotRegistry·
// RobotBinding, CollisionMonitor) 사이의 갭을 여기서 1회 닫는다 — 글루(main.ts)와
// 통합 테스트(e2e-sequence.test.ts)가 동일 어댑터를 공유한다(중복 배선 방지).
//
// 갭 내용:
// - RobotApi.readJoints(robot, names?)는 robot 인자를 받지만 RobotBinding.readJoints는
//   (names?)만 받는다 → registry.get(robot) 조회로 연결.
// - RobotApi.gripperConfig는 RobotSpec.gripper(스키마 데이터)에서 온다 — core는 스키마
//   "해석"을 소유하지 않으므로 조회 함수를 호출자가 주입한다. gripper.joints의 논리명
//   해석(jointMap)은 RobotBinding.setJoint/readJoints의 resolveJoint가 이미 수행한다.
// - CollisionQuery.mark()는 opaque(unknown) 계약이고 CollisionMonitor.mark()는 branded
//   CollisionMark를 반환한다 → 어댑터가 브랜드를 복원해 happenedSince로 넘긴다.

import type { CollisionMark, CollisionMonitor } from '../collision';
import type { RobotRegistry } from '../robots';
import type { CollisionQuery, RobotApi } from './steps';

/** RobotSpec.gripper와 동형 (schema 값을 그대로 통과시킨다 — DATA_MODEL §4.1) */
export interface GripperConfig {
  joints: string[];
  open: number;
  close: number;
}

/**
 * RobotRegistry를 player의 RobotApi로 어댑트한다.
 * @param gripperConfigOf 로봇 엔티티 id → gripper 설정 (RobotSpec.gripper — 없으면
 *        undefined를 반환하면 gripper step이 경고 후 건너뛴다). 글루가 SceneSpec에서
 *        구현해 주입한다 — core는 스키마 엔티티 조회를 소유하지 않는다.
 */
export function robotApiFromRegistry(
  registry: RobotRegistry,
  gripperConfigOf: (robotId: string) => GripperConfig | undefined,
): RobotApi {
  return {
    readJoints: (robot, names) => registry.get(robot).readJoints(names),
    setJoints: (robot, values) => {
      registry.get(robot).setJoints(values);
    },
    gripperConfig: (robot) => gripperConfigOf(robot),
  };
}

/** CollisionMonitor를 player의 CollisionQuery로 어댑트한다. */
export function collisionQueryFromMonitor(monitor: CollisionMonitor): CollisionQuery {
  return {
    mark: () => monitor.mark(),
    // CollisionQuery.mark()가 발급한 opaque 값만 되돌아온다는 계약이므로 브랜드 복원이
    // 안전하다 (CollisionMark는 branded number — steps.ts는 unknown으로만 들고 다닌다).
    happenedSince: (mark, between) => monitor.happenedSince(mark as CollisionMark, between),
  };
}
