// render/highlight.ts — 충돌/포커스 시각 펄스 (UX_DESIGN §3.3 "충돌 시각화")
//
// pulseEntity(node): 대상 노드 서브트리(Mesh 단일·로봇 Group 모두)의 재질 emissive를
// 잠깐 붉게 올렸다가(~600ms) 원래 값으로 복원한다. 순수 시각 효과다 — 물리 상태를
// 건드리지 않는다(불변식 §2.1의 "순수 시각 요소" 예외). 진행 중인 펄스가 있는 노드에
// 다시 요청하면 이전 펄스를 원복 후 새로 시작한다(중첩 오염 방지).
//
// 성능: 펄스 시작 시 재질 목록·원본 값을 1회 수집하고, 프레임마다 재사용 Color에만
// 쓴다 — rAF 프레임당 신규 할당 없음.

import * as THREE from 'three';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4) ─────────────────────────────

/** 펄스 지속 시간 (ms) */
const PULSE_DURATION_MS = 600;
/** 펄스 색 (UX_DESIGN "빨강 펄스") */
const PULSE_COLOR_HEX = 0xe74c3c;

// ── 내부 타입 ───────────────────────────────────────────────────────

/** emissive를 가진 재질 (MeshStandardMaterial/MeshPhongMaterial/MeshLambertMaterial) */
interface EmissiveMaterial extends THREE.Material {
  emissive: THREE.Color;
}

interface PulsedMaterial {
  readonly material: EmissiveMaterial;
  /** 원본 emissive 값 (복원용 스냅샷) */
  readonly originalEmissiveHex: number;
}

interface ActivePulse {
  readonly materials: readonly PulsedMaterial[];
  rafId: number;
  startMs: number;
}

// ── 내부 상태/헬퍼 ──────────────────────────────────────────────────

/** 노드별 진행 중 펄스 — 재요청 시 이전 펄스를 원복하고 교체한다 */
const activePulses = new WeakMap<THREE.Object3D, ActivePulse>();

/** 프레임 계산 재사용 버퍼 (할당 없는 핫 패스) */
const scratchColor = new THREE.Color();
const pulseColor = new THREE.Color(PULSE_COLOR_HEX);

function hasEmissive(material: THREE.Material): material is EmissiveMaterial {
  return (material as Partial<EmissiveMaterial>).emissive instanceof THREE.Color;
}

/** 서브트리의 emissive 재질을 중복 없이 수집 (공유 재질은 1회만 — 이중 복원 방지) */
function collectEmissiveMaterials(node: THREE.Object3D): PulsedMaterial[] {
  const seen = new Set<THREE.Material>();
  const out: PulsedMaterial[] = [];
  node.traverse((obj) => {
    const mesh = obj as Partial<THREE.Mesh>;
    if (mesh.isMesh !== true || mesh.material === undefined) return;
    const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    for (const material of materials) {
      if (seen.has(material) || !hasEmissive(material)) continue;
      seen.add(material);
      out.push({ material, originalEmissiveHex: material.emissive.getHex() });
    }
  });
  return out;
}

function restore(pulse: ActivePulse): void {
  cancelAnimationFrame(pulse.rafId);
  for (const entry of pulse.materials) {
    entry.material.emissive.setHex(entry.originalEmissiveHex);
  }
}

// ── 공개 API ────────────────────────────────────────────────────────

/**
 * 노드(Mesh 또는 로봇 Group 서브트리)에 ~600ms 붉은 emissive 펄스를 준다.
 * 펄스가 끝나면(또는 같은 노드에 새 펄스가 오면) 원본 재질 값으로 복원한다.
 */
export function pulseEntity(visualNode: THREE.Object3D): void {
  const previous = activePulses.get(visualNode);
  if (previous) {
    restore(previous);
    activePulses.delete(visualNode);
  }

  const materials = collectEmissiveMaterials(visualNode);
  if (materials.length === 0) return; // emissive 없는 재질뿐이면 조용히 무시

  const pulse: ActivePulse = { materials, rafId: 0, startMs: performance.now() };
  activePulses.set(visualNode, pulse);

  const frame = (nowMs: number): void => {
    const t = (nowMs - pulse.startMs) / PULSE_DURATION_MS;
    if (t >= 1) {
      restore(pulse);
      activePulses.delete(visualNode);
      return;
    }
    // 펄스 강도: 시작 최대 → 선형 감쇠. 원본 emissive → 펄스색 사이 보간.
    const intensity = 1 - t;
    for (const entry of pulse.materials) {
      scratchColor.setHex(entry.originalEmissiveHex).lerp(pulseColor, intensity);
      entry.material.emissive.copy(scratchColor);
    }
    pulse.rafId = requestAnimationFrame(frame);
  };
  pulse.rafId = requestAnimationFrame(frame);
}
