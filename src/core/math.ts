// core/math.ts — 순수 수학 헬퍼 (규범: docs/SIMULATION.md §3 보간 계약)
//
// 이 모듈은 어떤 것도 import하지 않는다 — 물리/렌더/DOM 비의존 순수 함수만 둔다.
// (쿼터니언 slerp는 여기 두지 않는다: 시각 보간은 three.js가 core/sync에서 처리하고,
//  물리 상태는 보간하지 않는다. math.ts는 물리-순수(physics-pure)로 유지한다.)

/** Easing 종류. schema/types.ts의 `Easing`과 구조적으로 동일(모듈 순수성 위해 로컬 정의). */
export type EasingKind = 'linear' | 'easeInOut' | 'step';

/** x를 [0, 1] 구간으로 클램프한다. (NaN은 NaN 그대로 전파) */
export function clamp01(x: number): number {
  return Math.min(Math.max(x, 0), 1);
}

/**
 * 선형 보간: a + (b - a) * t.
 * t는 클램프하지 않는다(외삽 허용) — 호출자가 필요 시 clamp01을 적용한다.
 */
export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/**
 * 정규화 진행도 t(0..1)를 easing 곡선 값으로 변환한다. t는 내부에서 [0,1]로 클램프.
 *
 * - 'linear'    : 항등. e(0)=0, e(1)=1.
 * - 'easeInOut' : 코사인 smoothstep 스타일 — 0.5 - 0.5*cos(π·t).
 *                 (선택 근거: 단조 증가, 양 끝 속도 0, 분기 없는 단일 식이라
 *                  cubic smoothstep과 동등한 성질을 더 단순하게 얻는다.)
 * - 'step'      : t >= 1 ? 1 : 0 — duration이 끝나는 순간에만 목표값으로 점프.
 *                 (관절 setpoint가 마지막 tick 전까지 시작값을 유지하는 의미로 정의)
 */
export function ease(easing: EasingKind, t: number): number {
  const u = clamp01(t);
  switch (easing) {
    case 'linear':
      return u;
    case 'easeInOut':
      return 0.5 - 0.5 * Math.cos(Math.PI * u);
    case 'step':
      return u >= 1 ? 1 : 0;
  }
}
