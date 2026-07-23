// core/collision.test.ts — CollisionMonitor 단위 테스트 (Phase 4 핵심)
// (규범: docs/SIMULATION.md §4.3, docs/DATA_MODEL.md §7)
//
// 순수 로직만 검증한다 — Rapier/three/DOM 비의존 (CLAUDE.md §4 부수효과 격리).
// ContactEvent는 world.step()이 반환하는 형태를 손으로 구성해 주입한다.

import { describe, expect, it } from 'vitest';
import { CollisionMonitor, MAX_HISTORY } from './collision';
import type { CollisionMark } from './collision';
import type { ContactEvent } from './types';
import type { CollisionEvent, Vec3 } from '../schema/types';

// ── 테스트 헬퍼 ─────────────────────────────────────────────────────

function contact(
  a: string,
  b: string,
  phase: ContactEvent['phase'] = 'start',
  kind: ContactEvent['kind'] = 'contact',
): ContactEvent {
  return { a, b, phase, kind };
}

/** 테스트용 기준 시각 (임의의 simTime) */
const T0_SEC = 1.25;

// ── 1. dispatch → 구독자 통지 순서·페이로드 ─────────────────────────

describe('CollisionMonitor — dispatch/subscribe', () => {
  it('notifies subscribers in subscription order, per event in batch order', () => {
    const monitor = new CollisionMonitor();
    const calls: string[] = [];
    monitor.subscribe((e) => calls.push(`first:${e.a}-${e.b}`));
    monitor.subscribe((e) => calls.push(`second:${e.a}-${e.b}`));
    monitor.subscribe((e) => calls.push(`third:${e.a}-${e.b}`));

    monitor.dispatch([contact('arm', 'box'), contact('box', 'ground')], T0_SEC);

    expect(calls).toEqual([
      'first:arm-box', 'second:arm-box', 'third:arm-box',
      'first:box-ground', 'second:box-ground', 'third:box-ground',
    ]);
  });

  it('converts ContactEvent to CollisionEvent with timeSec and all fields', () => {
    const monitor = new CollisionMonitor();
    const received: CollisionEvent[] = [];
    monitor.subscribe((e) => received.push(e));

    const point: Vec3 = [0.1, 0.2, 0.3];
    const normal: Vec3 = [0, 1, 0];
    monitor.dispatch(
      [{ a: 'arm', b: 'box', phase: 'start', kind: 'sensor', point, normal }],
      T0_SEC,
    );

    expect(received).toHaveLength(1);
    const evt = received[0];
    expect(evt).toEqual({
      timeSec: T0_SEC,
      a: 'arm',
      b: 'box',
      phase: 'start',
      kind: 'sensor',
      point: [0.1, 0.2, 0.3],
      normal: [0, 1, 0],
    });
    // 방어적 사본 — 원본 튜플을 변경해도 발행된 이벤트는 오염되지 않는다
    expect(evt?.point).not.toBe(point);
    expect(evt?.normal).not.toBe(normal);
    point[0] = 999;
    expect(evt?.point?.[0]).toBe(0.1);
  });

  it('omits point/normal when the contact has none', () => {
    const monitor = new CollisionMonitor();
    const received: CollisionEvent[] = [];
    monitor.subscribe((e) => received.push(e));

    monitor.dispatch([contact('arm', 'box')], T0_SEC);

    expect(received[0]).toEqual({
      timeSec: T0_SEC, a: 'arm', b: 'box', phase: 'start', kind: 'contact',
    });
    expect(received[0]).not.toHaveProperty('point');
    expect(received[0]).not.toHaveProperty('normal');
  });

  it('records before notifying — subscriber sees the current event in history (§4.3)', () => {
    const monitor = new CollisionMonitor();
    const historyLastSeen: Array<CollisionEvent | undefined> = [];
    monitor.subscribe((e) => {
      const h = monitor.history();
      historyLastSeen.push(h[h.length - 1]);
      expect(h[h.length - 1]).toEqual(e);
    });

    monitor.dispatch([contact('arm', 'box'), contact('box', 'ground')], T0_SEC);
    expect(historyLastSeen).toHaveLength(2);
  });

  it('does nothing for an empty contact batch', () => {
    const monitor = new CollisionMonitor();
    let called = 0;
    monitor.subscribe(() => { called += 1; });

    monitor.dispatch([], T0_SEC);

    expect(called).toBe(0);
    expect(monitor.history()).toHaveLength(0);
  });
});

// ── 2. 순서 무관 쌍 매칭 ────────────────────────────────────────────

describe('CollisionMonitor — happenedSince unordered pair matching', () => {
  it('matches (a,b) and (b,a) identically', () => {
    const monitor = new CollisionMonitor();
    const mark = monitor.mark();
    monitor.dispatch([contact('arm', 'box')], T0_SEC);

    expect(monitor.happenedSince(mark, ['arm', 'box'])).toBe(true);
    expect(monitor.happenedSince(mark, ['box', 'arm'])).toBe(true);
  });

  it('does not match a different pair', () => {
    const monitor = new CollisionMonitor();
    const mark = monitor.mark();
    monitor.dispatch([contact('arm', 'box')], T0_SEC);

    expect(monitor.happenedSince(mark, ['arm', 'ground'])).toBe(false);
    expect(monitor.happenedSince(mark, ['box', 'ground'])).toBe(false);
  });

  it('does not treat a pair sharing one entity as a match', () => {
    const monitor = new CollisionMonitor();
    const mark = monitor.mark();
    monitor.dispatch([contact('arm', 'box')], T0_SEC);

    // 'arm'이 겹치더라도 상대가 다르면 매칭 아님
    expect(monitor.happenedSince(mark, ['arm', 'arm'])).toBe(false);
  });
});

// ── 3. phase / kind 필터 ────────────────────────────────────────────

describe('CollisionMonitor — happenedSince phase/kind filters', () => {
  it("defaults to phase 'start' — a stop-only event does not release the barrier", () => {
    const monitor = new CollisionMonitor();
    const mark = monitor.mark();
    monitor.dispatch([contact('arm', 'box', 'stop')], T0_SEC);

    expect(monitor.happenedSince(mark, ['arm', 'box'])).toBe(false);
    expect(monitor.happenedSince(mark, ['arm', 'box'], { phase: 'stop' })).toBe(true);
  });

  it("phase 'start' filter ignores stop events and vice versa", () => {
    const monitor = new CollisionMonitor();
    const mark = monitor.mark();
    monitor.dispatch([contact('arm', 'box', 'start')], T0_SEC);

    expect(monitor.happenedSince(mark, ['arm', 'box'], { phase: 'start' })).toBe(true);
    expect(monitor.happenedSince(mark, ['arm', 'box'], { phase: 'stop' })).toBe(false);
  });

  it('defaults to any kind; explicit kind filter narrows to contact/sensor', () => {
    const monitor = new CollisionMonitor();
    const mark = monitor.mark();
    monitor.dispatch([contact('arm', 'zone', 'start', 'sensor')], T0_SEC);

    expect(monitor.happenedSince(mark, ['arm', 'zone'])).toBe(true); // kind 무관
    expect(monitor.happenedSince(mark, ['arm', 'zone'], { kind: 'sensor' })).toBe(true);
    expect(monitor.happenedSince(mark, ['arm', 'zone'], { kind: 'contact' })).toBe(false);
  });

  it('combines phase and kind filters', () => {
    const monitor = new CollisionMonitor();
    const mark = monitor.mark();
    monitor.dispatch(
      [contact('arm', 'zone', 'stop', 'sensor'), contact('arm', 'box', 'start', 'contact')],
      T0_SEC,
    );

    expect(monitor.happenedSince(mark, ['arm', 'zone'], { phase: 'stop', kind: 'sensor' })).toBe(true);
    expect(monitor.happenedSince(mark, ['arm', 'zone'], { phase: 'stop', kind: 'contact' })).toBe(false);
    expect(monitor.happenedSince(mark, ['arm', 'box'], { phase: 'start', kind: 'sensor' })).toBe(false);
  });
});

// ── 4. mark 커서와 이력 eviction ────────────────────────────────────

describe('CollisionMonitor — mark/happenedSince across eviction', () => {
  /** filler 이벤트 n개를 순차 dispatch (다른 쌍 — 매칭 대상 아님) */
  function fill(monitor: CollisionMonitor, n: number): void {
    for (let i = 0; i < n; i += 1) {
      monitor.dispatch([contact('filler_a', 'filler_b')], T0_SEC + i);
    }
  }

  it('only sees events after the mark', () => {
    const monitor = new CollisionMonitor();
    monitor.dispatch([contact('arm', 'box')], T0_SEC);

    const markAfter = monitor.mark();
    expect(monitor.happenedSince(markAfter, ['arm', 'box'])).toBe(false); // mark 이전 이벤트

    monitor.dispatch([contact('arm', 'box')], T0_SEC + 1);
    expect(monitor.happenedSince(markAfter, ['arm', 'box'])).toBe(true);
  });

  it('history is bounded at MAX_HISTORY, dropping the oldest first', () => {
    const monitor = new CollisionMonitor();
    const OVERFLOW = 7;
    for (let i = 0; i < MAX_HISTORY + OVERFLOW; i += 1) {
      monitor.dispatch([contact('filler_a', 'filler_b')], i); // timeSec = 발행 순번
    }

    const history = monitor.history();
    expect(history).toHaveLength(MAX_HISTORY);
    // 가장 오래된 OVERFLOW개가 버려졌다 — 남은 첫 이벤트는 OVERFLOW번째 발행분
    expect(history[0]?.timeSec).toBe(OVERFLOW);
    expect(history[history.length - 1]?.timeSec).toBe(MAX_HISTORY + OVERFLOW - 1);
  });

  it('an old mark still finds a match that arrives after >MAX_HISTORY evictions', () => {
    const monitor = new CollisionMonitor();
    const oldMark = monitor.mark();

    fill(monitor, MAX_HISTORY + 200); // oldMark 직후 구간은 전부 밀려났다
    expect(monitor.happenedSince(oldMark, ['arm', 'box'])).toBe(false);

    monitor.dispatch([contact('arm', 'box')], T0_SEC);
    // 시퀀스 번호 기반 커서 — eviction으로 인덱스가 밀려도 오배열 없이 매칭된다
    expect(monitor.happenedSince(oldMark, ['arm', 'box'])).toBe(true);
    // filler 쌍이 매칭 대상으로 오인되는 거짓 양성도 없다
    expect(monitor.happenedSince(oldMark, ['arm', 'ground'])).toBe(false);
  });

  it('a fresh mark taken after eviction ignores all earlier events', () => {
    const monitor = new CollisionMonitor();
    monitor.dispatch([contact('arm', 'box')], T0_SEC);
    fill(monitor, MAX_HISTORY + 50);

    const freshMark = monitor.mark();
    expect(monitor.happenedSince(freshMark, ['arm', 'box'])).toBe(false);
    expect(monitor.happenedSince(freshMark, ['filler_a', 'filler_b'])).toBe(false);

    monitor.dispatch([contact('arm', 'box')], T0_SEC + 1);
    expect(monitor.happenedSince(freshMark, ['arm', 'box'])).toBe(true);
  });

  it('documented limitation: a match already evicted past the mark is no longer visible (no crash)', () => {
    const monitor = new CollisionMonitor();
    const mark = monitor.mark();
    monitor.dispatch([contact('arm', 'box')], T0_SEC); // 매칭 이벤트
    fill(monitor, MAX_HISTORY + 1); // 매칭 이벤트가 이력 밖으로 밀려난다

    // 보존된 이력에서만 판정 — 거짓 음성(문서화된 한계)이지 예외/오배열이 아니다
    expect(monitor.happenedSince(mark, ['arm', 'box'])).toBe(false);
  });
});

// ── 5. unsubscribe ──────────────────────────────────────────────────

describe('CollisionMonitor — unsubscribe', () => {
  it('stops notifications after unsubscribe; other subscribers unaffected', () => {
    const monitor = new CollisionMonitor();
    const calls: string[] = [];
    const unsubFirst = monitor.subscribe(() => calls.push('first'));
    monitor.subscribe(() => calls.push('second'));

    monitor.dispatch([contact('arm', 'box')], T0_SEC);
    expect(calls).toEqual(['first', 'second']);

    unsubFirst();
    monitor.dispatch([contact('arm', 'box')], T0_SEC + 1);
    expect(calls).toEqual(['first', 'second', 'second']);
  });

  it('unsubscribe is idempotent (double call is a no-op)', () => {
    const monitor = new CollisionMonitor();
    const calls: string[] = [];
    const unsubFirst = monitor.subscribe(() => calls.push('first'));
    monitor.subscribe(() => calls.push('second'));

    unsubFirst();
    expect(() => unsubFirst()).not.toThrow();

    monitor.dispatch([contact('arm', 'box')], T0_SEC);
    expect(calls).toEqual(['second']);
  });

  it('history keeps recording regardless of subscriber presence', () => {
    const monitor = new CollisionMonitor();
    monitor.dispatch([contact('arm', 'box')], T0_SEC); // 구독자 0명
    expect(monitor.history()).toHaveLength(1);
  });
});

// ── 6. history 필터 ─────────────────────────────────────────────────

describe('CollisionMonitor — history filters', () => {
  function seed(monitor: CollisionMonitor): void {
    monitor.dispatch([contact('arm', 'box')], 1);
    monitor.dispatch([contact('box', 'ground')], 2);
    monitor.dispatch([contact('arm', 'ground')], 3);
    monitor.dispatch([contact('arm', 'box', 'stop')], 4);
  }

  it('returns all events, most recent last', () => {
    const monitor = new CollisionMonitor();
    seed(monitor);
    expect(monitor.history().map((e) => e.timeSec)).toEqual([1, 2, 3, 4]);
  });

  it('filters by entity on either side of the pair', () => {
    const monitor = new CollisionMonitor();
    seed(monitor);

    expect(monitor.history({ entity: 'box' }).map((e) => e.timeSec)).toEqual([1, 2, 4]);
    expect(monitor.history({ entity: 'ground' }).map((e) => e.timeSec)).toEqual([2, 3]);
    expect(monitor.history({ entity: 'nope' })).toEqual([]);
  });

  it('limit keeps only the most recent N (after entity filtering)', () => {
    const monitor = new CollisionMonitor();
    seed(monitor);

    expect(monitor.history({ limit: 2 }).map((e) => e.timeSec)).toEqual([3, 4]);
    expect(monitor.history({ limit: 0 })).toEqual([]);
    expect(monitor.history({ limit: 99 })).toHaveLength(4); // 초과 limit은 전체 반환
    expect(monitor.history({ entity: 'box', limit: 2 }).map((e) => e.timeSec)).toEqual([2, 4]);
  });

  it('rejects a negative or non-integer limit', () => {
    const monitor = new CollisionMonitor();
    seed(monitor);
    expect(() => monitor.history({ limit: -1 })).toThrow(RangeError);
    expect(() => monitor.history({ limit: 1.5 })).toThrow(RangeError);
    expect(() => monitor.history({ limit: NaN })).toThrow(RangeError);
  });

  it('returns a snapshot copy — later dispatches do not mutate it', () => {
    const monitor = new CollisionMonitor();
    seed(monitor);
    const snapshot = monitor.history();
    monitor.dispatch([contact('arm', 'box')], 5);

    expect(snapshot).toHaveLength(4);
    expect(monitor.history()).toHaveLength(5);
  });
});

// ── 7. clear ────────────────────────────────────────────────────────

describe('CollisionMonitor — clear', () => {
  it('empties history and resets the sequence counter', () => {
    const monitor = new CollisionMonitor();
    monitor.dispatch([contact('arm', 'box')], T0_SEC);
    monitor.clear();

    expect(monitor.history()).toEqual([]);
    // 시퀀스 리셋 — 새 mark는 fresh 모니터의 첫 mark와 동일하게 동작한다
    const mark = monitor.mark();
    expect(monitor.happenedSince(mark, ['arm', 'box'])).toBe(false);
    monitor.dispatch([contact('arm', 'box')], T0_SEC + 1);
    expect(monitor.happenedSince(mark, ['arm', 'box'])).toBe(true);
  });

  it('invalidates marks issued before clear (no throw, no stale positives while empty)', () => {
    const monitor = new CollisionMonitor();
    const preClearMark = monitor.mark();
    monitor.dispatch([contact('arm', 'box')], T0_SEC);
    const lateMark = monitor.mark(); // 시퀀스 1 시점

    monitor.clear();

    // clear 직후 이력이 비어 있으므로 어떤 mark로도 매칭 없음 — 예외도 없다
    expect(monitor.happenedSince(preClearMark, ['arm', 'box'])).toBe(false);
    expect(monitor.happenedSince(lateMark, ['arm', 'box'])).toBe(false);

    // 문서화된 무효화: clear 이전 mark는 새 시퀀스 기준으로 재해석되므로 의미가 없다.
    // (lateMark=1은 clear 후 첫 이벤트(seq 0)를 건너뛴다 — 새로 mark를 발급해야 한다.)
    monitor.dispatch([contact('arm', 'box')], T0_SEC + 1);
    expect(monitor.happenedSince(lateMark, ['arm', 'box'])).toBe(false);
    expect(monitor.happenedSince(preClearMark, ['arm', 'box'])).toBe(true); // 우연의 일치일 뿐
  });

  it('keeps subscribers across clear (UI log panel stays attached on scene reset)', () => {
    const monitor = new CollisionMonitor();
    const calls: number[] = [];
    monitor.subscribe((e) => calls.push(e.timeSec));

    monitor.dispatch([contact('arm', 'box')], 1);
    monitor.clear();
    monitor.dispatch([contact('arm', 'box')], 2);

    expect(calls).toEqual([1, 2]);
  });
});

// ── 타입 계약 (컴파일 타임) ──────────────────────────────────────────
// CollisionMark는 opaque 브랜드 타입이다 — 임의 number를 넘기면 컴파일 오류가 나야 한다.
// (런타임 검증이 아니라 타입 경계 확인용 — 실수로 배열 인덱스를 넘기는 오용 방지.)

it('CollisionMark is branded — plain numbers are rejected at compile time', () => {
  const monitor = new CollisionMonitor();
  // @ts-expect-error — 임의 number는 CollisionMark가 아니다 (mark()로만 발급)
  monitor.happenedSince(0, ['a', 'b']);
  // 발급된 mark는 그대로 사용 가능
  const mark: CollisionMark = monitor.mark();
  expect(monitor.happenedSince(mark, ['a', 'b'])).toBe(false);
});
