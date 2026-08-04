// core/scene-edit-types.ts — 씬 편집 계약 (Phase 7 Scene Builder)
//
// 설계 원칙:
// - 편집의 단일 진실은 SceneSpec(데이터)이다 — 불변식 §2.5. 모든 편집 연산은
//   spec을 먼저 갱신하고, 물리/시각을 spec에 맞게 재생성(또는 teleport)한다.
// - 시뮬 트랜스폼의 진실은 여전히 물리(§2.1) — 편집 연산의 teleport는 "씬 리셋/편집
//   전용" API(PhysicsWorld.teleport)만 사용한다.
// - Undo/Redo는 ui 계층이 serialize() 스냅샷 스택으로 구현한다(전체 재로드 방식 —
//   정확성 우선, EXPERIMENTS 기록).

import type {
  ColliderShape,
  ConveyorSpec,
  EntitySpec,
  PhysicsSpec,
  SceneSpec,
  Transform,
} from '../schema/types';

/** 편집 변경 종류 (onChange 통지용) */
export type SceneEditKind =
  | 'add' | 'remove' | 'transform' | 'dimensions' | 'physics' | 'conveyor' | 'rename' | 'reload';

export interface SceneEditEvent {
  kind: SceneEditKind;
  entityId: string;
}

/**
 * 씬 편집기 — SceneSpec과 살아있는 월드(물리+시각)를 동기 상태로 유지한다.
 * 구현은 core(scene-editor.ts). ui는 이 인터페이스만 사용한다.
 */
export interface SceneEditor {
  /** 현재 편집 상태의 spec (읽기 전용 — 변경은 반드시 아래 연산으로) */
  readonly spec: Readonly<SceneSpec>;

  /** 엔티티 추가 (robot은 URDF 로드로 async). id 중복 시 한국어 오류 throw */
  addEntity(spec: EntitySpec): Promise<void>;
  removeEntity(id: string): void;

  /** 배치 편집: spec.transform 갱신 + 물리 teleport + 시각 반영 */
  updateTransform(id: string, transform: Transform): void;

  /**
   * spec을 바꾸지 않고 물리/시각만 현재 spec.transform으로 재수렴시킨다 (통지 없음).
   * 커밋되지 않은 기즈모 드래그 프리뷰를 버리는 용도 — 특히 로봇의 시각 루트는
   * RenderSync가 되돌려주지 않으므로 이 연산이 유일한 복원 경로다(scene-editor.ts 주석).
   */
  resyncTransform(id: string): void;

  /** 치수 편집: 프리미티브 shape 교체 — 시각 메시 + collider 동시 재생성 (UX §3.5) */
  updateDimensions(id: string, shape: ColliderShape): void;

  /** 물리 속성 편집: bodyType/그룹/마찰 등 — 바디+collider 재생성 */
  updatePhysics(id: string, physics: PhysicsSpec): void;

  /**
   * 컨베이어 표면 구동 편집 (속도·방향·재순환) — DATA_MODEL §4.2.
   * 벨트 기하는 바인딩 생성 시점에 고정되므로 엔티티를 재빌드해 바인딩을 새로 만든다.
   * conveyor 블록이 없던 엔티티에 붙이는 것도 허용한다(라이브러리 Plane → 벨트 승격).
   */
  updateConveyor(id: string, conveyor: ConveyorSpec): void;

  renameEntity(id: string, newId: string): void;

  /** 저장/Undo 스냅샷용 깊은 복사본 (검증 통과 보장) */
  serialize(): SceneSpec;

  /** 편집 통지 구독 (UI 목록/인스펙터 갱신) — 해제 함수 반환 */
  onChange(fn: (e: SceneEditEvent) => void): () => void;
}

/**
 * 임포트된 3D 에셋 저장소 — collider convexHull/trimesh의 ref 해석 지점.
 * 구현은 render/main 글루. core(world)는 resolver 함수만 주입받는다.
 */
export interface MeshAssetResolver {
  /** ref(예: "asset://uuid" 또는 data: URI) → 정점 배열 [x,y,z,...]. 미해석 시 undefined */
  getPoints(ref: string): Float32Array | undefined;
  /** trimesh용 인덱스 (없으면 undefined → convex hull 전용) */
  getIndices(ref: string): Uint32Array | undefined;
}
