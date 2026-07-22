// core/robot-types.ts — 로봇 관절 구동 계약 (Phase 3)
//
// 설계 (EXPERIMENTS.md 기록 참조):
// - 관절 상태(joint values)의 진실은 core(RobotBinding)가 소유한다.
// - FK(관절값 → 링크 월드 pose) 계산은 render 계층의 urdf-loader 씬 그래프가 수행하고,
//   core는 이 RobotFkView 인터페이스로만 접근한다(three.js 비노출 — CLAUDE.md §3).
// - 매 물리 tick 전(preStep) FK 결과를 kinematicPosition 바디에 밀어 넣는다.
//   시각 로봇(urdf Object3D)과 물리 바디가 같은 관절 상태에서 유도되므로 단일 진실 유지.
// - URDF는 Z-up → 내부 Y-up 변환은 render의 URDF 로딩 지점 한 곳에서만 수행한다(§4).

import type { Pose } from './types';
import type { ColliderShape, Transform } from '../schema/types';

export type JointType = 'revolute' | 'prismatic' | 'continuous' | 'fixed';

export interface JointInfo {
  name: string;
  type: JointType;
  /** [lower, upper] — continuous/fixed는 없음. 단위: rad(revolute) / m(prismatic) */
  limits?: [number, number];
  /** URDF 로드 직후 초기값 (보통 0) */
  initial: number;
}

/** URDF <collision> 태그에서 유도한 링크-로컬 collider 정의 */
export interface LinkColliderDef {
  shape: ColliderShape;
  /** 링크 프레임 기준 로컬 오프셋 (Y-up 변환 완료 상태) */
  offset?: Transform;
}

/**
 * render 계층(urdf-loader)이 구현하는 로봇 FK 뷰.
 * core는 이 인터페이스만 참조한다 — three.js 심볼이 경계를 넘지 않는다.
 */
export interface RobotFkView {
  /** 구동 가능한 관절 목록 (fixed 제외) */
  readonly joints: readonly JointInfo[];
  /** 관절값 적용 (시각 로봇 즉시 갱신 + 이후 readLinkPoses에 반영) */
  setJointValues(values: Readonly<Record<string, number>>): void;
  /** 링크명 → 월드 공간 pose (엔티티 배치·Y-up 변환 포함). setJointValues 후 호출 */
  readLinkPoses(): ReadonlyMap<string, Pose>;
  /** 링크명 → collider 정의 (URDF collision 태그 유래, 링크 로컬·Y-up) */
  readonly linkColliders: ReadonlyMap<string, readonly LinkColliderDef[]>;
}
