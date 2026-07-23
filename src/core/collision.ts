// core/collision.ts — CollisionMonitor: 접촉 이벤트 발행·이력·배리어 조회 (Phase 4)
// (규범: docs/SIMULATION.md §4.3, docs/DATA_MODEL.md §7, docs/ROADMAP.md Phase 4)
//
// ── 역할 ─────────────────────────────────────────────────────────────
// Engine의 onContacts 훅(ARCHITECTURE §5 ③)에서 매 물리 tick 호출된다:
//   world.step()이 반환한 ContactEvent[] → CollisionEvent(timeSec 포함)로 변환
//   → 이력(bounded history)에 기록 → 구독자(UI 충돌 로그·하이라이트)에 통지.
// player의 waitForCollision 배리어는 mark()/happenedSince()로 이력을 조회해 해제된다
// (SIMULATION §3). 이벤트별로 "기록 후 통지" 순서를 지킨다 — 구독자 콜백 안에서
// history()를 읽으면 방금 통지받은 이벤트가 이미 포함되어 있다 (§4.3 record → notify).
//
// ── 이력 경계(bounded history)와 mark 커서 ───────────────────────────
// 이력은 최대 MAX_HISTORY개만 유지하고 넘치면 가장 오래된 것부터 버린다(드롭-oldest).
// mark()는 배열 인덱스가 아니라 "전역 단조 증가 이벤트 시퀀스 번호"를 커서로 쓰므로,
// 이력이 밀려나가도(eviction) 커서가 다른 이벤트를 가리키게 되는 일이 없다.
// 한계: mark 이후의 이벤트가 이미 MAX_HISTORY개 초과로 밀려났다면 그 이벤트는 조회
// 불가다 — happenedSince는 "보존된 이력" 범위에서만 판정한다(거짓 양성 없음, 아주
// 오래된 mark에 한해 거짓 음성 가능). waitForCollision은 mark 직후부터 매 tick 폴링
// 하므로 실사용에서 이 한계에 닿지 않는다(1 tick에 1000+ 이벤트가 나야 함).
//
// ── 계층 규칙 ────────────────────────────────────────────────────────
// 순수 모듈: core/types(ContactEvent) + schema/types(CollisionEvent)만 import한다.
// Rapier/three.js/DOM 비의존 — 물리 없이 단위 테스트 가능 (CLAUDE.md §4 부수효과 격리).
// 핸들 → 엔티티 변환은 world.step()이 이미 끝냈다(handle 매핑은 world 소유 — §2.4).

import type { ContactEvent } from './types';
import type { CollisionEvent } from '../schema/types';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4) ─────────────────────────────

/**
 * 이력 최대 보관 개수. 초과 시 가장 오래된 이벤트부터 버린다.
 * UI 충돌 로그·waitForCollision 배리어가 실사용에서 필요로 하는 범위를 넉넉히 덮으면서
 * 장시간 재생에서도 메모리가 유계(bounded)로 유지되는 값.
 */
export const MAX_HISTORY = 1000;

/** happenedSince의 기본 phase 필터 — 배리어는 "접촉 시작"을 기다리는 것이 기본이다. */
const DEFAULT_SINCE_PHASE: CollisionEvent['phase'] = 'start';

// ── 공개 타입 ───────────────────────────────────────────────────────

declare const collisionMarkBrand: unique symbol;

/**
 * 이력 커서(opaque). mark()가 발급하고 happenedSince()만 소비한다.
 * 내부적으로는 전역 단조 증가 이벤트 시퀀스 번호(number)다 — 배열 인덱스가 아니므로
 * 이력 eviction과 무관하게 항상 같은 시점("이 커서 발급 이후")을 가리킨다.
 * clear() 이후에는 그 이전에 발급된 mark가 모두 무효화된다(클래스 주석 참조).
 */
export type CollisionMark = number & { readonly [collisionMarkBrand]: 'CollisionMark' };

/** 충돌 이벤트 구독 콜백 (UI 로그 패널·뷰포트 하이라이트·오케스트레이터 등) */
export type CollisionListener = (e: CollisionEvent) => void;

/** happenedSince 필터. phase 생략 시 'start', kind 생략 시 모든 kind 허용. */
export interface CollisionSinceFilter {
  phase?: CollisionEvent['phase'];
  kind?: CollisionEvent['kind'];
}

/** history 필터. entity: 해당 엔티티가 a/b 어느 쪽이든 포함된 이벤트만. limit: 최근 N개. */
export interface CollisionHistoryFilter {
  entity?: string;
  limit?: number;
}

// ── CollisionMonitor ────────────────────────────────────────────────

/**
 * 충돌 이벤트의 발행자이자 이력 소유자.
 *
 * - dispatch(): Engine onContacts 훅 전용 — ContactEvent를 CollisionEvent로 변환해
 *   이벤트별로 [이력 기록 → 구독자 통지(구독 순서)]를 수행한다.
 * - subscribe(): UI 등 소비자 등록. 반환된 함수로 해제한다.
 * - mark()/happenedSince(): waitForCollision 배리어용 커서 조회.
 * - history(): UI 충돌 로그 패널용 스냅샷 조회 (최신이 마지막).
 * - clear(): 씬 리셋 지원 — 이력·시퀀스 카운터 초기화. 구독자는 유지된다.
 *
 * clear() 계약: 이력과 시퀀스 카운터가 모두 0으로 리셋되므로 **clear() 이전에 발급된
 * CollisionMark는 전부 무효**다. 무효 mark를 happenedSince에 넘겨도 던지지는 않지만
 * 결과는 의미가 없다 — 씬 리셋 시 배리어 소비자(player)도 함께 리셋되어 mark를 새로
 * 발급받아야 한다(결정론적 재생 — SIMULATION §6).
 */
export class CollisionMonitor {
  /** 구독자 집합 — Set 삽입 순서 = 구독 순서 = 통지 순서 (결정론, SIMULATION §6).
   *  동일 함수의 중복 구독은 병합된다(1회만 통지). */
  private readonly subscribers = new Set<CollisionListener>();

  /** 보존 중인 이력 (오래된 것 → 최신 순). 길이 ≤ MAX_HISTORY 불변식. */
  private readonly events: CollisionEvent[] = [];

  /** events[0]의 전역 시퀀스 번호. eviction 때마다 전진한다. */
  private firstSeq = 0;

  /** 다음 이벤트가 받을 전역 시퀀스 번호 (= 생성/clear 이후 총 발행 수). */
  private nextSeq = 0;

  /**
   * 접촉 이벤트 일괄 발행 — Engine onContacts 훅이 매 물리 tick 호출한다.
   *
   * 이벤트별 처리 순서(§4.3): CollisionEvent 변환 → 이력 기록 → 구독자 통지.
   * 통지는 이벤트마다 구독자 스냅샷을 순회한다 — 콜백 안에서 구독/해제가 일어나도
   * 진행 중인 이벤트의 통지 순서는 흔들리지 않는다(추가된 구독자는 다음 이벤트부터,
   * 해제된 구독자는 현재 이벤트까지 받는다).
   */
  dispatch(contacts: readonly ContactEvent[], simTimeSec: number): void {
    for (const contact of contacts) {
      const evt: CollisionEvent = {
        timeSec: simTimeSec,
        a: contact.a,
        b: contact.b,
        phase: contact.phase,
        kind: contact.kind,
      };
      // 접촉점/법선은 있을 때만 복사(방어적 사본 — 발행 후 원본 변경에 오염되지 않게)
      if (contact.point) evt.point = [contact.point[0], contact.point[1], contact.point[2]];
      if (contact.normal) evt.normal = [contact.normal[0], contact.normal[1], contact.normal[2]];

      this.record(evt);
      for (const listener of [...this.subscribers]) {
        listener(evt);
      }
    }
  }

  /** 구독 등록. 반환된 함수를 호출하면 해제된다(중복 호출은 no-op). */
  subscribe(fn: CollisionListener): () => void {
    this.subscribers.add(fn);
    return () => {
      this.subscribers.delete(fn);
    };
  }

  /**
   * "지금"을 가리키는 이력 커서 발급 — 이후 happenedSince(mark, …)로
   * "이 시점 이후 해당 충돌이 있었는가"를 조회한다 (waitForCollision 배리어).
   */
  mark(): CollisionMark {
    return this.nextSeq as CollisionMark;
  }

  /**
   * mark 이후에 between 쌍(순서 무관 — (a,b)와 (b,a) 동일)의 충돌 이벤트가 있었는지.
   * phase 기본값 'start', kind 기본값은 모든 kind(contact/sensor) 허용.
   *
   * 보존된 이력 범위에서만 판정한다 — mark보다 오래된 이벤트를 잘못 매칭하는
   * 거짓 양성은 없고, mark 이후 이벤트가 이미 밀려난 극단적 경우에만 거짓 음성이
   * 가능하다(파일 헤더의 이력 경계 설명 참조).
   */
  happenedSince(
    mark: CollisionMark,
    between: readonly [string, string],
    opts?: CollisionSinceFilter,
  ): boolean {
    const phase = opts?.phase ?? DEFAULT_SINCE_PHASE;
    const kind = opts?.kind;
    const [x, y] = between;

    // mark(시퀀스 번호) → 보존 이력의 시작 인덱스. mark 이전 구간이 이미 밀려났으면
    // 보존된 처음(0)부터 본다 — mark보다 오래된 이벤트는 남아 있을 수 없으므로 안전.
    const startIndex = Math.max(0, mark - this.firstSeq);
    for (let i = startIndex; i < this.events.length; i += 1) {
      const evt = this.events[i];
      if (!evt) continue; // noUncheckedIndexedAccess 방어 — 루프 범위상 도달 불가
      if (evt.phase !== phase) continue;
      if (kind !== undefined && evt.kind !== kind) continue;
      if ((evt.a === x && evt.b === y) || (evt.a === y && evt.b === x)) return true;
    }
    return false;
  }

  /**
   * 이력 스냅샷 조회 (UI 충돌 로그 패널용). 오래된 것 → 최신 순(최신이 마지막).
   * - entity: a/b 어느 쪽이든 해당 엔티티가 포함된 이벤트만.
   * - limit: (필터 적용 후) 최근 N개만. 0 이상의 정수여야 한다.
   * 반환 배열은 사본이다 — 이후 dispatch가 일어나도 변하지 않는다.
   */
  history(filter?: CollisionHistoryFilter): readonly CollisionEvent[] {
    const limit = filter?.limit;
    if (limit !== undefined && (!Number.isInteger(limit) || limit < 0)) {
      throw new RangeError(`collision: history limit은 0 이상의 정수여야 합니다: ${limit}`);
    }

    const entity = filter?.entity;
    const matched =
      entity !== undefined
        ? this.events.filter((evt) => evt.a === entity || evt.b === entity)
        : [...this.events];

    if (limit !== undefined && matched.length > limit) {
      return matched.slice(matched.length - limit);
    }
    return matched;
  }

  /**
   * 씬 리셋 지원: 이력과 시퀀스 카운터를 초기화한다.
   * 이전에 발급된 모든 CollisionMark는 무효가 된다(클래스 주석의 clear 계약 참조).
   * 구독자는 유지된다 — UI 로그 패널 등은 리셋 후에도 새 이벤트를 계속 받는다.
   */
  clear(): void {
    this.events.length = 0;
    this.firstSeq = 0;
    this.nextSeq = 0;
  }

  // ── 내부 ──────────────────────────────────────────────────────────

  /**
   * 이력 기록 + 시퀀스 전진. MAX_HISTORY 초과 시 가장 오래된 이벤트를 버린다.
   * shift()는 O(길이)지만 넘칠 때만, 상한 1000에서 실행된다 — MVP 규모에서 무시 가능.
   * 링 버퍼 최적화는 병목이 측정된 뒤에만 도입한다 (robots.ts tick()과 같은 원칙).
   */
  private record(evt: CollisionEvent): void {
    this.events.push(evt);
    this.nextSeq += 1;
    if (this.events.length > MAX_HISTORY) {
      this.events.shift();
      this.firstSeq += 1;
    }
  }
}
