// render/sync-math.ts — pose 보간 순수 수학 (three.js 비의존)
//
// core/sync.ts(RenderSync)가 사용하는 lerp/slerp/clamp 헬퍼.
// three.js 없이 튜플(Vec3/Quat)만 다루므로 node 환경 vitest로 단위 테스트 가능
// (vitest.config.ts의 원칙: three/Rapier 의존 테스트를 만들지 않는다).
//
// 모든 함수는 out-파라미터에 결과를 쓰는 할당-프리(allocation-free) 스타일이다 —
// 핫 패스(매 렌더 프레임 × 바인딩 수)에서 GC 압력을 만들지 않기 위함.
// slerpQuatInto/lerpVec3Into는 입력을 먼저 모두 읽은 뒤 out에 쓰므로
// out이 a 또는 b와 같은 배열이어도 안전하다(aliasing-safe).
//
// 회전 표현은 프로젝트 표준 쿼터니언 [x, y, z, w] (CLAUDE.md §4).

import type { Quat, Vec3 } from '../schema/types';

/**
 * slerp에서 두 쿼터니언이 거의 평행할 때(1 - |cosθ| < EPS) sin θ ≈ 0 나눗셈의
 * 수치 불안정을 피해 nlerp(정규화 선형보간)로 폴백하는 임계값.
 */
const SLERP_NLERP_FALLBACK_EPS = 1e-9;

/** t를 [0, 1]로 클램프. Engine의 alpha 방어용. */
export function clamp01(t: number): number {
  return t < 0 ? 0 : t > 1 ? 1 : t;
}

/** out ← src (성분 복사, 할당 없음) */
export function copyVec3Into(out: Vec3, src: Readonly<Vec3>): void {
  out[0] = src[0];
  out[1] = src[1];
  out[2] = src[2];
}

/** out ← src (성분 복사, 할당 없음) */
export function copyQuatInto(out: Quat, src: Readonly<Quat>): void {
  out[0] = src[0];
  out[1] = src[1];
  out[2] = src[2];
  out[3] = src[3];
}

/** out ← a + (b − a)·t  (성분별 선형보간) */
export function lerpVec3Into(out: Vec3, a: Readonly<Vec3>, b: Readonly<Vec3>, t: number): void {
  const ax = a[0], ay = a[1], az = a[2];
  out[0] = ax + (b[0] - ax) * t;
  out[1] = ay + (b[1] - ay) * t;
  out[2] = az + (b[2] - az) * t;
}

/**
 * out ← slerp(a, b, t) — 구면 선형보간, 최단 경로(shortest-path).
 *
 * - dot(a, b) < 0이면 b를 부호 반전해 항상 최단 호를 따라 보간한다
 *   (q와 −q는 같은 회전이므로 결과 회전은 동일, 경로만 짧아짐).
 * - 두 회전이 거의 같으면 nlerp + 정규화로 폴백(수치 안정성).
 * - 입력이 단위 쿼터니언이면 출력도 단위 쿼터니언이다.
 */
export function slerpQuatInto(out: Quat, a: Readonly<Quat>, b: Readonly<Quat>, t: number): void {
  const ax = a[0], ay = a[1], az = a[2], aw = a[3];
  let bx = b[0], by = b[1], bz = b[2], bw = b[3];

  let cosHalfTheta = ax * bx + ay * by + az * bz + aw * bw;
  if (cosHalfTheta < 0) {
    bx = -bx; by = -by; bz = -bz; bw = -bw;
    cosHalfTheta = -cosHalfTheta;
  }

  if (1 - cosHalfTheta < SLERP_NLERP_FALLBACK_EPS) {
    // 거의 같은 회전: nlerp 후 정규화
    const x = ax + (bx - ax) * t;
    const y = ay + (by - ay) * t;
    const z = az + (bz - az) * t;
    const w = aw + (bw - aw) * t;
    const n = Math.sqrt(x * x + y * y + z * z + w * w);
    const inv = n > 0 ? 1 / n : 0;
    out[0] = x * inv;
    out[1] = y * inv;
    out[2] = z * inv;
    out[3] = w * inv;
    return;
  }

  const halfTheta = Math.acos(cosHalfTheta);
  const sinHalfTheta = Math.sqrt(1 - cosHalfTheta * cosHalfTheta);
  const ra = Math.sin((1 - t) * halfTheta) / sinHalfTheta;
  const rb = Math.sin(t * halfTheta) / sinHalfTheta;
  out[0] = ax * ra + bx * rb;
  out[1] = ay * ra + by * rb;
  out[2] = az * ra + bz * rb;
  out[3] = aw * ra + bw * rb;
}
