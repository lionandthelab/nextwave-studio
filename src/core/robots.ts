// core/robots.ts — 로봇 관절 상태 소유 + kinematicPosition 구동 (Phase 3)
//
// 설계 (core/robot-types.ts 헤더, docs/ARCHITECTURE.md §5, docs/SIMULATION.md §3):
// - 관절 상태(joint values)의 유일한 진실은 이 모듈의 RobotBinding이다.
// - FK(관절값 → 링크 월드 pose)는 render 계층(urdf-loader 씬 그래프)이 계산하고,
//   core는 RobotFkView 인터페이스(POJO만)로 결과를 읽는다 — three.js 심볼 비노출.
// - 매 물리 tick, world.step() "직전"(Engine preStep 훅)에 RobotRegistry.tickAll()이
//   FK 링크 pose를 kinematicPosition 바디에 world.setKinematicPose로 밀어 넣는다.
// - 시각 로봇(urdf Object3D)은 같은 관절 상태에서 직접 갱신된다(그 자체가 FK 그래프).
//   따라서 로봇 링크는 RenderSync에 바인딩하지 않는다 — 물리·시각이 단일 관절 상태에서
//   유도되므로 단일 진실이 유지된다 (CLAUDE.md §2.1의 정신).
// - URDF Z-up → 내부 Y-up 변환은 render의 URDF 래퍼 한 곳에서만 수행한다 (CLAUDE.md §4).
//   이 모듈에 도달하는 pose는 이미 Y-up·월드 좌표다.
//
// 계층 규칙: core/types + core/robot-types만 안다. three.js/Rapier/스키마 검증 비의존.

import type { BodyId, EntityId, PhysicsWorld, Pose } from './types';
import type { JointInfo, RobotFkView } from './robot-types';

// ── 옵션 ────────────────────────────────────────────────────────────

export interface RobotBindingOptions {
  /** SceneSpec 엔티티 id (충돌 로그·제어 시퀀스 참조 키) */
  entityId: EntityId;
  /** render 계층이 구현한 FK 뷰 (urdf-loader 씬 그래프) */
  fk: RobotFkView;
  /** kinematicPosition 링크 바디를 소유한 물리 월드 */
  world: PhysicsWorld;
  /** URDF 링크명 → 물리 바디 핸들. 순회는 이 Map의 삽입 순서를 따른다(결정론). */
  linkBodies: ReadonlyMap<string, BodyId>;
  /** 논리 관절명 → URDF joint명 (RobotSpec.jointMap) */
  jointMap?: Record<string, string>;
  /**
   * 안전 클램프 override (RobotSpec.jointLimits). URDF limits와 "교집합"을 유효
   * limits로 쓴다 — override가 더 좁으면 override가, URDF가 더 좁으면 URDF가 이긴다.
   * 키는 논리명/URDF명 모두 허용(resolveJoint로 해석).
   */
  jointLimitOverrides?: Record<string, [number, number]>;
  /** 초기 관절값 (RobotSpec.home). 키는 논리명/URDF명 모두 허용. */
  home?: Record<string, number>;
}

// ── 내부 헬퍼 (순수) ─────────────────────────────────────────────────

/**
 * URDF limits와 override의 교집합. 둘 다 없으면 undefined(무제한 — continuous 등).
 * 교집합이 비면(lower > upper) 스펙 오류이므로 즉시 던진다 — 조용히 이상한
 * 클램프가 일어나는 것보다 로드 시점 실패가 낫다.
 */
function intersectLimits(
  entityId: EntityId,
  jointName: string,
  urdfLimits: readonly [number, number] | undefined,
  overrideLimits: readonly [number, number] | undefined,
): [number, number] | undefined {
  if (!urdfLimits && !overrideLimits) return undefined;
  const lower = Math.max(
    urdfLimits?.[0] ?? Number.NEGATIVE_INFINITY,
    overrideLimits?.[0] ?? Number.NEGATIVE_INFINITY,
  );
  const upper = Math.min(
    urdfLimits?.[1] ?? Number.POSITIVE_INFINITY,
    overrideLimits?.[1] ?? Number.POSITIVE_INFINITY,
  );
  if (lower > upper) {
    throw new Error(
      `robots: 로봇 '${entityId}' 관절 '${jointName}'의 limit 교집합이 비어 있습니다 — ` +
        `URDF [${urdfLimits?.join(', ') ?? '없음'}] ∩ override [${overrideLimits?.join(', ') ?? '없음'}]. ` +
        'jointLimits 설정을 확인하세요.',
    );
  }
  return [lower, upper];
}

// ── RobotBinding ────────────────────────────────────────────────────

/**
 * 로봇 1대의 관절 상태 소유자 + kinematic 구동자.
 *
 * 생성 시: fk.joints의 initial 값으로 상태를 초기화한 뒤 home을 적용한다
 * (home 키는 resolveJoint로 해석 — 논리명 허용, limits 클램프 적용).
 *
 * 매 물리 tick(preStep): tick()이 관절 상태 → FK → 링크 pose를
 * kinematicPosition 바디에 밀어 넣는다.
 */
export class RobotBinding {
  readonly entityId: EntityId;

  private readonly fk: RobotFkView;
  private readonly world: PhysicsWorld;
  private readonly linkBodies: ReadonlyMap<string, BodyId>;
  /** 논리명 → URDF joint명. Map 사용 — Record의 프로토타입 키 오염 회피. */
  private readonly jointMap: ReadonlyMap<string, string>;
  /** URDF joint명 → JointInfo (존재 확인·오류 메시지용) */
  private readonly jointInfoByRawName: ReadonlyMap<string, JointInfo>;
  /** URDF joint명 → 유효 limits (URDF ∩ override). 무제한 관절은 엔트리 없음. */
  private readonly effectiveLimits: ReadonlyMap<string, readonly [number, number]>;
  /** fk.joints 순서 그대로, limits만 유효 limits로 치환한 사본 */
  private readonly effectiveJoints: readonly JointInfo[];
  /** 관절 상태의 유일한 진실. 키는 URDF joint명, 삽입 순서 = fk.joints 순서. */
  private readonly state: Map<string, number>;
  /** 생성 시 받은 home 사본 — applyHome()이 재사용 */
  private readonly home: ReadonlyMap<string, number>;

  constructor(opts: RobotBindingOptions) {
    this.entityId = opts.entityId;
    this.fk = opts.fk;
    this.world = opts.world;
    this.linkBodies = opts.linkBodies;
    this.jointMap = new Map(Object.entries(opts.jointMap ?? {}));
    this.home = new Map(Object.entries(opts.home ?? {}));

    // 1) 관절 목록·초기 상태 (fk.joints의 initial 값)
    const infoByName = new Map<string, JointInfo>();
    this.state = new Map();
    for (const joint of opts.fk.joints) {
      infoByName.set(joint.name, joint);
      this.state.set(joint.name, joint.initial);
    }
    this.jointInfoByRawName = infoByName;

    // 2) 유효 limits = URDF ∩ override (override 키는 resolveJoint로 해석)
    const overrideByRawName = new Map<string, readonly [number, number]>();
    for (const [name, range] of Object.entries(opts.jointLimitOverrides ?? {})) {
      overrideByRawName.set(this.resolveJoint(name), [range[0], range[1]]);
    }
    const limits = new Map<string, readonly [number, number]>();
    const joints: JointInfo[] = [];
    for (const joint of opts.fk.joints) {
      const effective = intersectLimits(
        this.entityId, joint.name, joint.limits, overrideByRawName.get(joint.name),
      );
      if (effective) limits.set(joint.name, effective);
      joints.push({
        name: joint.name,
        type: joint.type,
        initial: joint.initial,
        limits: effective ? [effective[0], effective[1]] : undefined,
      });
    }
    this.effectiveLimits = limits;
    this.effectiveJoints = joints;

    // 3) home 적용 (논리명 해석 + limits 클램프는 setJoint 경로가 담당)
    this.applyHome();
  }

  /** fk.joints와 같은 순서·구성, limits만 유효 limits(URDF ∩ override)로 치환. */
  get joints(): readonly JointInfo[] {
    return this.effectiveJoints;
  }

  /**
   * 논리 관절명 → URDF joint명. jointMap에 없으면 이름 자체를 URDF명으로 간주한다.
   * 존재하지 않는 관절이면 사용 가능한 관절 목록과 함께 던진다.
   */
  resolveJoint(name: string): string {
    const rawName = this.jointMap.get(name) ?? name;
    if (!this.jointInfoByRawName.has(rawName)) {
      const available = [...this.jointInfoByRawName.keys()].join(', ');
      const logicalNames = [...this.jointMap.keys()];
      const logicalNote = logicalNames.length > 0 ? ` (논리명: ${logicalNames.join(', ')})` : '';
      throw new Error(
        `robots: 로봇 '${this.entityId}'에 관절 '${name}'이(가) 없습니다. ` +
          `사용 가능한 관절: ${available}${logicalNote}`,
      );
    }
    return rawName;
  }

  /**
   * 관절 목표값 설정. 유효 limits(URDF ∩ override)를 벗어나면 경고 후 클램프한다.
   * 실제 물리/시각 반영은 다음 tick()에서 일어난다.
   */
  setJoint(name: string, value: number): void {
    const rawName = this.resolveJoint(name);
    if (!Number.isFinite(value)) {
      throw new Error(
        `robots: 로봇 '${this.entityId}' 관절 '${rawName}' 값이 유한한 숫자가 아닙니다: ${value}`,
      );
    }
    const limits = this.effectiveLimits.get(rawName);
    let applied = value;
    if (limits && (value < limits[0] || value > limits[1])) {
      applied = Math.min(Math.max(value, limits[0]), limits[1]);
      console.warn(
        `robots: 로봇 '${this.entityId}' 관절 '${rawName}' 목표값 ${value}이(가) ` +
          `허용 범위 [${limits[0]}, ${limits[1]}]을 벗어나 ${applied}(으)로 클램프되었습니다`,
      );
    }
    this.state.set(rawName, applied);
  }

  /** 여러 관절 일괄 설정. 키는 논리명/URDF명 모두 허용. */
  setJoints(targets: Readonly<Record<string, number>>): void {
    for (const [name, value] of Object.entries(targets)) {
      this.setJoint(name, value);
    }
  }

  /**
   * 관절값 조회 (항상 복사본 — 내부 상태의 라이브 참조를 노출하지 않는다).
   * - names 생략: 모든 관절을 "URDF 원명" 키로 반환.
   * - names 지정: 각 이름(논리명/원명)을 해석해 "요청한 이름 그대로" 키로 반환
   *   — player가 targets와 같은 키로 시작값을 스냅샷할 수 있게 (SIMULATION.md §3).
   */
  readJoints(names?: readonly string[]): Record<string, number> {
    if (!names) return Object.fromEntries(this.state);
    const out: Record<string, number> = {};
    for (const name of names) {
      const rawName = this.resolveJoint(name);
      const value = this.state.get(rawName);
      if (value === undefined) {
        // resolveJoint 통과 관절은 state에 반드시 있다 — 불변식 방어
        throw new Error(`robots: 로봇 '${this.entityId}' 관절 '${rawName}' 상태가 없습니다 (내부 오류)`);
      }
      out[name] = value;
    }
    return out;
  }

  /**
   * 생성 시점 초기 포즈 복원: 모든 관절을 fk.joints의 initial로 되돌린 뒤 home을
   * 다시 적용한다. home에 없는 관절도 initial로 돌아가므로 씬 리셋이 결정론적이다
   * (SIMULATION.md §6 "초기 상태(home 관절값) 스키마로 고정").
   */
  applyHome(): void {
    for (const joint of this.fk.joints) {
      this.state.set(joint.name, joint.initial);
    }
    for (const [name, value] of this.home) {
      this.setJoint(name, value);
    }
  }

  /**
   * 매 물리 tick(Engine preStep, world.step() 직전) 호출:
   * 관절 상태 → FK 반영 → 링크 월드 pose → kinematicPosition 바디 목표 지정.
   *
   * 변경 여부와 무관하게 매 tick 전체를 push한다 — kinematicPosition 바디는
   * "다음 스텝 목표"를 매 스텝 요구하고(목표 미지정 스텝은 정지로 해석되어 접촉
   * 응답용 속도 추정이 끊긴다), MVP 규모(링크 수십 개)에서 비용은 무시 가능하다.
   * dirty-check 최적화는 병목이 측정된 뒤에만 도입한다.
   */
  tick(): void {
    this.forEachLinkFkPose((bodyId, pose) => {
      this.world.setKinematicPose(bodyId, pose);
    });
  }

  /**
   * 씬 리셋/빌드 전용: 링크 바디를 현재 관절 상태의 FK pose로 "즉시" 텔레포트한다.
   *
   * tick()은 다음 스텝 목표만 지정하므로 스텝 전까지 물리 pose가 이전 상태로 남고,
   * 다음 스텝에서 이전 pose → 목표 pose 스윕이 |Δpose|/dt의 가짜 kinematic 속도를
   * 만든다(접촉 중인 동적 사물에 비현실적 임펄스). 리셋 직후 물리 조회(getPose)가
   * 관절 상태와 즉시 일치하고 첫 스텝 스윕이 0이 되도록 teleport로 정렬한다 —
   * fresh load와 reset()의 스텝 전 상태가 동일해진다(결정론적 재생, SIMULATION.md §6).
   * 시뮬 재생 중에는 호출 금지 (world.teleport 계약 — core/types.ts).
   */
  teleportLinksToFk(): void {
    this.forEachLinkFkPose((bodyId, pose) => {
      this.world.teleport(bodyId, pose);
    });
  }

  /** 관절 상태를 FK 뷰에 반영한 뒤 linkBodies 순서대로 링크 FK pose에 fn을 적용한다. */
  private forEachLinkFkPose(fn: (bodyId: BodyId, pose: Pose) => void): void {
    this.fk.setJointValues(Object.fromEntries(this.state));
    const poses = this.fk.readLinkPoses();
    for (const [linkName, bodyId] of this.linkBodies) {
      const pose = poses.get(linkName);
      if (pose) fn(bodyId, pose);
    }
  }
}

// ── RobotRegistry ───────────────────────────────────────────────────

/**
 * 씬의 모든 RobotBinding 보관소. Engine preStep 훅이 tickAll()을 호출한다.
 * 순회는 등록(삽입) 순서를 따른다 — 결정론 (SIMULATION.md §6).
 */
export class RobotRegistry {
  private readonly bindings = new Map<EntityId, RobotBinding>();

  add(binding: RobotBinding): void {
    if (this.bindings.has(binding.entityId)) {
      throw new Error(`robots: 로봇 '${binding.entityId}'이(가) 이미 등록되어 있습니다`);
    }
    this.bindings.set(binding.entityId, binding);
  }

  get(entityId: EntityId): RobotBinding {
    const binding = this.bindings.get(entityId);
    if (!binding) {
      const ids = [...this.bindings.keys()];
      throw new Error(
        `robots: 로봇 '${entityId}'을(를) 찾을 수 없습니다. ` +
          `등록된 로봇: ${ids.length > 0 ? ids.join(', ') : '(없음)'}`,
      );
    }
    return binding;
  }

  has(entityId: EntityId): boolean {
    return this.bindings.has(entityId);
  }

  /**
   * 등록 해제 (Phase 7 씬 편집 — SceneEditor의 로봇 removeEntity/renameEntity 전용).
   * 해제하지 않으면 Engine preStep의 tickAll()이 제거된 링크 바디에
   * setKinematicPose를 시도해 던진다. 미등록 id는 no-op (dispose 경로 멱등성).
   */
  remove(entityId: EntityId): void {
    this.bindings.delete(entityId);
  }

  /** 등록된 로봇 엔티티 id 목록 (삽입 순서) */
  ids(): string[] {
    return [...this.bindings.keys()];
  }

  /** Engine preStep 훅 대상 — 매 물리 tick, world.step() 전에 1회 호출한다. */
  tickAll(): void {
    for (const binding of this.bindings.values()) {
      binding.tick();
    }
  }
}
