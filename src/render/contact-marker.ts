// render/contact-marker.ts — 충돌 접촉점 3D 마커 (UX_DESIGN §3.3 "접촉점 마커 표시")
//
// 충돌 순간 접촉점에 "빨강 구 + 퍼지는 링"을 띄워 어디서 부딪혔는지 즉시 보이게 한다.
// 엔티티 전체 펄스(highlight.ts)가 "무엇이" 부딪혔는지를 알려준다면, 이 마커는
// "어디서"를 알려준다 — 로봇 링크처럼 큰 물체끼리의 접촉에서 특히 중요하다.
//
// 설계:
// - depthTest=false + 높은 renderOrder → 로봇/물체 내부에서 생긴 접촉도 가려지지 않는다.
// - 고정 크기 풀을 재사용한다 — 충돌이 몰려도 프레임당 신규 할당이 없다.
// - 순수 시각 효과다. 물리 상태를 읽지도 쓰지도 않는다(불변식 §2.1 "순수 시각 요소").
// - update(nowMs)는 렌더 프레임마다 호출한다. 일시정지 중에도 잔여 마커가 자연스럽게
//   사라지도록 물리 시간이 아니라 벽시계(rAF) 시간으로 감쇠한다.

import * as THREE from 'three';
import type { Vec3 } from '../schema/types';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4) ─────────────────────────────

/** 동시 표시 가능한 최대 마커 수 — 초과 요청은 가장 오래된 것을 재사용한다 */
const MARKER_POOL_SIZE = 24;
/** 마커 수명 (ms) */
const MARKER_LIFETIME_MS = 1100;
/** 접촉점 구 반지름 (m) */
const DOT_RADIUS_M = 0.016;
/** 링 최종 반지름 (m) — 수명 동안 0에서 이 값까지 퍼진다 */
const RING_MAX_RADIUS_M = 0.16;
/** 링 두께 비율 (반지름 대비) */
const RING_THICKNESS_RATIO = 0.22;
/** 마커 색 (충돌 = 빨강 계열, UX_DESIGN §3.3) */
const MARKER_COLOR_HEX = 0xff3b30;
/** 구 시작 불투명도 */
const DOT_OPACITY = 0.95;
/** 링 시작 불투명도 */
const RING_OPACITY = 0.8;
/** 다른 3D 요소보다 나중에 그려 항상 위에 보이게 한다 */
const MARKER_RENDER_ORDER = 999;
/** 링 분할 수 — 매끄러운 원 */
const RING_SEGMENTS = 32;

// ── 내부 타입 ───────────────────────────────────────────────────────

interface PooledMarker {
  readonly group: THREE.Group;
  readonly dot: THREE.Mesh;
  readonly ring: THREE.Mesh;
  readonly dotMaterial: THREE.MeshBasicMaterial;
  readonly ringMaterial: THREE.MeshBasicMaterial;
  /** 생성 시각(ms). null이면 유휴 슬롯 */
  startMs: number | null;
}

export interface ContactMarkers {
  /** 접촉점에 마커를 띄운다. normal이 있으면 링을 접촉면에 정렬한다. */
  spawn(point: Vec3, normal?: Vec3): void;
  /** 렌더 프레임마다 호출 — 수명이 지난 마커를 숨기고 애니메이션을 진행한다 */
  update(nowMs: number): void;
  /** 표시 중인 마커를 모두 즉시 제거 (씬 전환·리셋) */
  clear(): void;
  dispose(): void;
}

// ── 구현 ────────────────────────────────────────────────────────────

/**
 * 접촉점 마커 풀을 씬에 부착한다.
 * @param parent 마커를 담을 부모 (보통 renderer.scene 루트 — 월드 좌표를 그대로 쓴다)
 */
export function mountContactMarkers(parent: THREE.Object3D): ContactMarkers {
  // 지오메트리는 풀 전체가 공유한다 (마커마다 크기는 group.scale로 조절)
  const dotGeometry = new THREE.SphereGeometry(DOT_RADIUS_M, 12, 8);
  const ringGeometry = new THREE.RingGeometry(
    1 - RING_THICKNESS_RATIO,
    1,
    RING_SEGMENTS,
  );

  const markers: PooledMarker[] = [];
  for (let i = 0; i < MARKER_POOL_SIZE; i += 1) {
    const dotMaterial = new THREE.MeshBasicMaterial({
      color: MARKER_COLOR_HEX,
      transparent: true,
      opacity: DOT_OPACITY,
      depthTest: false, // 물체 내부 접촉도 보이게
    });
    const ringMaterial = new THREE.MeshBasicMaterial({
      color: MARKER_COLOR_HEX,
      transparent: true,
      opacity: RING_OPACITY,
      depthTest: false,
      side: THREE.DoubleSide,
    });

    const dot = new THREE.Mesh(dotGeometry, dotMaterial);
    const ring = new THREE.Mesh(ringGeometry, ringMaterial);
    dot.renderOrder = MARKER_RENDER_ORDER;
    ring.renderOrder = MARKER_RENDER_ORDER;

    const group = new THREE.Group();
    group.add(dot);
    group.add(ring);
    group.visible = false;
    group.renderOrder = MARKER_RENDER_ORDER;
    parent.add(group);

    markers.push({ group, dot, ring, dotMaterial, ringMaterial, startMs: null });
  }

  /** 링 정렬용 재사용 버퍼 (프레임당 할당 없음) */
  const normalVec = new THREE.Vector3();
  let nextSlot = 0;

  /** 유휴 슬롯 우선, 없으면 라운드로빈으로 가장 오래된 슬롯을 재사용 */
  function acquire(): PooledMarker {
    for (const marker of markers) {
      if (marker.startMs === null) return marker;
    }
    const marker = markers[nextSlot]!;
    nextSlot = (nextSlot + 1) % markers.length;
    return marker;
  }

  return {
    spawn(point: Vec3, normal?: Vec3): void {
      const marker = acquire();
      marker.group.position.set(point[0], point[1], point[2]);

      // 링을 접촉면(법선에 수직)에 눕힌다. 법선이 없으면 카메라 무관 기본 자세.
      if (normal) {
        normalVec.set(normal[0], normal[1], normal[2]);
        if (normalVec.lengthSq() > 0) {
          // RingGeometry는 +Z를 향한다 — 법선 방향으로 회전
          marker.ring.quaternion.setFromUnitVectors(
            new THREE.Vector3(0, 0, 1),
            normalVec.normalize(),
          );
        }
      } else {
        marker.ring.quaternion.identity();
      }

      marker.startMs = null; // update가 첫 프레임에 시작 시각을 채운다
      marker.group.visible = true;
      marker.group.scale.setScalar(1);
      marker.dotMaterial.opacity = DOT_OPACITY;
      marker.ringMaterial.opacity = RING_OPACITY;
      marker.ring.scale.setScalar(0.001); // 0에서 시작해 퍼진다
      // 실제 시작 시각은 첫 update에서 확정한다(spawn 시각과 프레임 시각 통일)
      marker.startMs = -1;
    },

    update(nowMs: number): void {
      for (const marker of markers) {
        if (marker.startMs === null) continue;
        if (marker.startMs === -1) marker.startMs = nowMs; // 첫 프레임에 시각 확정

        const elapsed = nowMs - marker.startMs;
        if (elapsed >= MARKER_LIFETIME_MS) {
          marker.group.visible = false;
          marker.startMs = null;
          continue;
        }

        const t = elapsed / MARKER_LIFETIME_MS; // 0..1
        const fade = 1 - t;
        // 링: 반지름 확장 + 페이드아웃 (충돌이 "터지는" 느낌)
        marker.ring.scale.setScalar(Math.max(RING_MAX_RADIUS_M * t, 0.001));
        marker.ringMaterial.opacity = RING_OPACITY * fade;
        // 구: 제자리에서 페이드아웃 (접촉 지점을 계속 지시)
        marker.dotMaterial.opacity = DOT_OPACITY * fade;
      }
    },

    clear(): void {
      for (const marker of markers) {
        marker.group.visible = false;
        marker.startMs = null;
      }
    },

    dispose(): void {
      for (const marker of markers) {
        marker.group.removeFromParent();
        marker.dotMaterial.dispose();
        marker.ringMaterial.dispose();
      }
      markers.length = 0;
      dotGeometry.dispose();
      ringGeometry.dispose();
    },
  };
}
