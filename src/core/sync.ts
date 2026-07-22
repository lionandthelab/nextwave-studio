// core/sync.ts — RenderSync: 물리 pose → three.js Object3D 단방향 동기화 (보간)
//
// ARCHITECTURE.md §2/§3의 "sync (pose → Object3D)" 구현. core에서 three.js를 알아도
// 되는 유일한 파일이며, 그마저도 타입 전용 import만 사용한다(런타임 three 심볼 없음).
// 물리 접근은 PhysicsWorld 인터페이스(core/types)로만 한다 — Rapier 심볼 비노출 불변식.
//
// ── 불변식 (CLAUDE.md §2.1) ──────────────────────────────────────────
// 트랜스폼의 단일 진실 원천은 물리다. 이 모듈은 물리 → 시각 "단방향"으로만 쓴다.
// Object3D 쪽 값을 읽어 물리로 되돌리는 코드는 어떤 경우에도 추가하지 않는다.
//
// ── 바인딩 계약 ──────────────────────────────────────────────────────
// bind()에 넘기는 Object3D는 반드시 씬 루트(THREE.Scene)의 **직접 자식**이어야 한다.
// PhysicsWorld.getPose()는 월드 좌표 pose를 주고, 이 모듈은 그 값을 obj.position /
// obj.quaternion에 그대로 쓴다. 변환을 가진 중간 그룹 아래에 바인딩하면 부모 변환이
// 중복 적용되어(이중 변환) 시각 pose가 물리와 어긋난다.
//
// ── 호출 계약 (Engine 고정 timestep 루프, SIMULATION.md §2/§5) ───────
//   매 물리 tick:    sync.commit();      // world.step() "직전" — prev ← 현재 pose
//                    world.step();
//   매 렌더 프레임:  sync.apply(alpha);  // alpha = accumulator / fixedDt ∈ [0, 1)
//
// commit()이 스텝 직전 pose를 prev로 스냅샷하므로, apply(alpha)는 "직전 스텝 pose"와
// "최신 스텝 pose" 사이를 보간한다. 보간은 시각적 매끄러움 전용이며(alpha는 렌더에만
// 영향) 물리 상태는 항상 정확히 fixedDt 단위로만 전진한다.
//
// ── 성능 계약 ────────────────────────────────────────────────────────
// commit()/apply()는 핫 패스다. per-binding prev 버퍼는 bind() 시 1회 할당하고,
// 보간 결과는 클래스 소유 스크래치 튜플에 재사용해 프레임당 신규 할당이 없다.
// 바인딩 순회는 Map의 삽입 순서를 따른다(결정론 체크리스트, SIMULATION.md §6).

import type { Object3D } from 'three';
import type { BodyId, PhysicsWorld } from './types';
import type { Quat, Vec3 } from '../schema/types';
import {
  clamp01,
  copyQuatInto,
  copyVec3Into,
  lerpVec3Into,
  slerpQuatInto,
} from '../render/sync-math';

interface Binding {
  readonly obj: Object3D;
  /** 직전 물리 스텝의 pose 스냅샷 (bind 시 1회 할당, commit이 덮어씀) */
  readonly prevPosition: Vec3;
  readonly prevRotation: Quat;
}

export class RenderSync {
  /** bodyId → 바인딩. 삽입 순서 순회(Map 계약) — 이벤트/렌더 순서 결정론 유지. */
  private readonly bindings = new Map<BodyId, Binding>();

  // 보간 결과용 스크래치 버퍼 — 매 프레임 재사용, 신규 할당 없음
  private readonly scratchPosition: Vec3 = [0, 0, 0];
  private readonly scratchRotation: Quat = [0, 0, 0, 1];

  constructor(private readonly world: PhysicsWorld) {}

  /**
   * 물리 바디 ↔ 시각 노드 바인딩 등록.
   * obj는 씬 루트의 직접 자식이어야 한다(파일 상단 "바인딩 계약" 참조).
   * 현재 물리 pose를 즉시 prev로 스냅샷하므로, 첫 commit() 전에 apply()가 불려도
   * 유효한 pose가 쓰인다.
   */
  bind(bodyId: BodyId, obj: Object3D): void {
    const pose = this.world.getPose(bodyId);
    this.bindings.set(bodyId, {
      obj,
      prevPosition: [pose.position[0], pose.position[1], pose.position[2]],
      prevRotation: [pose.rotation[0], pose.rotation[1], pose.rotation[2], pose.rotation[3]],
    });
  }

  /** 바인딩 해제. 엔티티 제거/씬 편집 시 scene-loader가 호출. */
  unbind(bodyId: BodyId): void {
    this.bindings.delete(bodyId);
  }

  /** 모든 바인딩 해제 (씬 재로드/리셋). */
  clear(): void {
    this.bindings.clear();
  }

  /**
   * 모든 바인딩의 현재 물리 pose를 prev로 스냅샷한다.
   * Engine이 매 물리 tick에서 world.step() "직전"에 호출한다.
   */
  commit(): void {
    for (const [bodyId, binding] of this.bindings) {
      const pose = this.world.getPose(bodyId);
      copyVec3Into(binding.prevPosition, pose.position);
      copyQuatInto(binding.prevRotation, pose.rotation);
    }
  }

  /**
   * prev(직전 스텝)와 cur(최신 물리 pose) 사이를 보간해 Object3D에 쓴다.
   * 위치는 선형보간, 회전은 최단 경로 slerp. 물리 → 시각 단방향(불변식 §2.1).
   *
   * @param alpha 보간 계수 = accumulator / fixedDt. [0,1] 밖 값은 방어적으로 클램프.
   */
  apply(alpha: number): void {
    const t = clamp01(alpha);
    for (const [bodyId, binding] of this.bindings) {
      const pose = this.world.getPose(bodyId);
      lerpVec3Into(this.scratchPosition, binding.prevPosition, pose.position, t);
      slerpQuatInto(this.scratchRotation, binding.prevRotation, pose.rotation, t);
      binding.obj.position.set(
        this.scratchPosition[0], this.scratchPosition[1], this.scratchPosition[2],
      );
      binding.obj.quaternion.set(
        this.scratchRotation[0], this.scratchRotation[1],
        this.scratchRotation[2], this.scratchRotation[3],
      );
    }
  }
}
