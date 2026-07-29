// core/collision-classify.ts — 접촉 이벤트 분류 (의도된 접촉 vs 예기치 않은 충돌)
//
// ── 왜 분류가 필요한가 ──────────────────────────────────────────────
//
// 물리 엔진은 "무엇이 무엇에 닿았다"만 알려준다. 그런데 사용자에게 **모든 접촉이 충돌은
// 아니다.** 로봇이 집으려는 박스에 손을 대는 것은 시퀀스가 의도한 성공이고, 로봇이 옆의
// 기둥에 부딪히는 것은 사고다. 둘을 같은 "충돌"로 세면 로그와 카운터가 성공을 사고로
// 물들이고, 정작 진짜 사고는 소음에 묻힌다.
//
// ── 무엇이 "타겟"인가 ───────────────────────────────────────────────
//
// 이 데이터 모델에서 **의도를 선언하는 유일한 지점은 `waitForCollision.between`** 이다
// (DATA_MODEL §6). "팔을 내리다가 box_a에 닿으면 다음으로" 라고 시퀀스가 적어 둔 쌍이
// 곧 조작 대상이다. 그 외의 접촉은 시퀀스가 예고하지 않은 것이므로 충돌로 본다.
//
// 이 규칙의 좋은 성질: 사용자가 "이건 의도한 접촉이다"라고 말하는 방법이 이미 존재하고
// (그래프에 접촉 대기 노드를 놓는 것), 그 선언이 실행 제어와 분류에 **같이** 쓰인다 —
// 분류만을 위한 별도 설정이 필요 없다.
//
// ── 계층 ────────────────────────────────────────────────────────────
// 순수 함수다. schema 타입만 알고 물리/렌더/UI를 모른다. 컨텍스트(로봇 id 집합·타겟 쌍·
// 바닥 id)는 호출자가 평범한 데이터로 넘긴다.

import type { CollisionEvent } from '../schema/types';

/**
 * 접촉의 성격.
 *
 * - `target`     — 시퀀스가 선언한 조작 대상과의 접촉. **성공이지 사고가 아니다.**
 * - `ground`     — 바닥과의 접촉. 로봇이 서 있고 물건이 놓여 있는 정상 상태.
 * - `sensor`     — 감지 영역(sensor collider) 통과. 물리 반응 없는 이벤트.
 * - `incidental` — 로봇이 관여하지 않은 사물끼리의 접촉(쌓기·굴러가기 등).
 * - `unexpected` — **로봇이 타겟도 바닥도 아닌 것에 부딪혔다.** 이것만 "충돌"이다.
 */
export type ContactClass = 'target' | 'ground' | 'sensor' | 'incidental' | 'unexpected';

export interface ClassifyContext {
  /** 씬의 로봇 엔티티 id */
  readonly robotIds: ReadonlySet<string>;
  /** 시퀀스가 선언한 조작 대상 쌍 (waitForCollision.between) — 순서 무관 */
  readonly targetPairs: readonly (readonly [string, string])[];
  /** 바닥 엔티티 id (환경 ground) */
  readonly groundId: string;
}

/** 쌍 일치 (순서 무관) */
export function pairMatches(e: CollisionEvent, x: string, y: string): boolean {
  return (e.a === x && e.b === y) || (e.a === y && e.b === x);
}

/**
 * 접촉 이벤트를 분류한다.
 *
 * 판정 순서가 곧 우선순위다.
 *
 * `sensor`가 `target`보다 먼저인 이유: sensor collider는 물리 반응이 없다 —
 * "부딪혔다"가 아니라 "지나갔다"이다. 배리어가 그 이벤트를 기다리는지는 오케스트레이션
 * 사정이고, 어느 쪽이든 충돌이 아니므로 로그에는 물리적 성격을 보이는 편이 정보가 많다.
 *
 * 그다음이 `target`인 이유: 사용자가 "이 쌍의 접촉을 기다린다"고 명시적으로 선언했다면,
 * 그 접촉은 바닥·로봇 관여 여부 같은 일반 규칙보다 우선해 의도된 것으로 취급되어야 한다.
 */
export function classifyContact(e: CollisionEvent, ctx: ClassifyContext): ContactClass {
  // 같은 엔티티 내부 접촉(링크끼리 등) — 사용자에게 의미 없는 이벤트
  if (e.a === e.b) return 'incidental';

  // 감지 영역은 물리 반응이 없다. "부딪혔다"가 아니라 "지나갔다"이다.
  if (e.kind === 'sensor') return 'sensor';

  // 시퀀스가 선언한 조작 대상 — 성공이다
  for (const [x, y] of ctx.targetPairs) {
    if (pairMatches(e, x, y)) return 'target';
  }

  if (e.a === ctx.groundId || e.b === ctx.groundId) return 'ground';

  const robotCount = (ctx.robotIds.has(e.a) ? 1 : 0) + (ctx.robotIds.has(e.b) ? 1 : 0);
  // 로봇이 관여하지 않으면 이 제품의 관심사(로봇–사물 충돌)가 아니다.
  // 여전히 로그에는 남는다 — 씬에서 무슨 일이 있었는지는 기록할 가치가 있다.
  if (robotCount === 0) return 'incidental';

  // 로봇이 타겟도 바닥도 아닌 것과 부딪혔다. 로봇×로봇도 여기 해당한다
  // (두 팔이 서로 부딪히는 것은 선언하지 않았다면 사고다).
  return 'unexpected';
}

/** "충돌"로 사용자에게 보고할 이벤트인가 — 카운터·배지·토스트·펄스의 기준 */
export function isCollision(cls: ContactClass): boolean {
  return cls === 'unexpected';
}

/** 분류 → 한국어 라벨 (색 없이 의미를 전달 — UX_DESIGN §9) */
export const CONTACT_CLASS_LABEL_KO: Readonly<Record<ContactClass, string>> = {
  target: '대상 접촉',
  ground: '바닥',
  sensor: '감지',
  incidental: '사물 접촉',
  unexpected: '충돌',
};
