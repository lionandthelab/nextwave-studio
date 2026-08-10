// ui/console/runs-screen.test.ts — 실행 기록 화면 순수 로직 단위 테스트 (DOM 비의존, node)
//
// mountRunsScreen의 DOM 조립·배선은 브라우저 게이트 몫이다(primitives.test.ts와 같은
// 관례). 여기서는 이 화면의 순수 계약만 검증한다:
// - 실행 결과 → 배지 매핑 (autoPaused는 COLLISION 램프 축 — warn 금지, 라벨로 구분)
// - 소요 시간 표기 포맷 (0.1초 단위 → 초 → 분·초 → 시간·분)
// - 통계 카드 숨김 임계값 (작은 표본의 성공률은 노이즈)
// - 개입 정렬 (시각 오름차순 · 안정 · 원본 비변경)
// - 충돌 분류 배지 · 재현 대상 판정 (예기치 않은 충돌 + 노드 있음일 때만)

import { describe, expect, it } from 'vitest';
import {
  STATS_MIN_RUN_COUNT,
  collisionBadgeSpec,
  formatDateTimeKo,
  formatDurationKo,
  formatSimClock,
  interventionKindKo,
  replayTarget,
  resultBadgeSpec,
  shouldShowStats,
  sortInterventionsBySimTime,
  successRatePercent,
} from './runs-screen';
import type { RunCollision, RunIntervention, RunResult } from '../../schema/entities';

// ── 결과 → 배지 매핑 ────────────────────────────────────────────────

describe('resultBadgeSpec', () => {
  it('4종 결과가 전부 매핑된다', () => {
    const results: RunResult[] = ['completed', 'stopped', 'error', 'autoPaused'];
    for (const r of results) {
      const spec = resultBadgeSpec(r);
      expect(spec.labelKo.length).toBeGreaterThan(0);
      expect(spec.status.length).toBeGreaterThan(0);
    }
  });

  it('completed→success, stopped→idle, error→error', () => {
    expect(resultBadgeSpec('completed').status).toBe('success');
    expect(resultBadgeSpec('stopped').status).toBe('idle');
    expect(resultBadgeSpec('error').status).toBe('error');
  });

  it('autoPaused는 COLLISION 램프 축(error)이다 — warn은 "충돌 아님" 축이라 금지', () => {
    expect(resultBadgeSpec('autoPaused').status).toBe('error');
    expect(resultBadgeSpec('autoPaused').status).not.toBe('warn');
  });

  it('error와 autoPaused는 같은 색이라도 라벨이 구분한다 (색만으로 전달 금지 — UX §9)', () => {
    expect(resultBadgeSpec('error').labelKo).not.toBe(resultBadgeSpec('autoPaused').labelKo);
  });

  it('라벨 4종은 전부 서로 다르다', () => {
    const labels = (['completed', 'stopped', 'error', 'autoPaused'] as const).map(
      (r) => resultBadgeSpec(r).labelKo,
    );
    expect(new Set(labels).size).toBe(4);
  });
});

// ── 소요 표기 포맷 ──────────────────────────────────────────────────

describe('formatDurationKo', () => {
  it('10초 미만은 0.1초 단위 (후행 .0은 표기하지 않는다)', () => {
    expect(formatDurationKo(0)).toBe('0초');
    expect(formatDurationKo(0.4)).toBe('0.4초');
    expect(formatDurationKo(4.06)).toBe('4.1초');
    expect(formatDurationKo(4)).toBe('4초');
  });

  it('반올림이 10초에 닿으면 정수 경로로 넘어간다', () => {
    expect(formatDurationKo(9.97)).toBe('10초');
  });

  it('60초 미만은 초 단위 정수', () => {
    expect(formatDurationKo(45)).toBe('45초');
    expect(formatDurationKo(59.4)).toBe('59초');
  });

  it('분·초 표기 — 0초인 하위 단위는 생략', () => {
    expect(formatDurationKo(60)).toBe('1분');
    expect(formatDurationKo(59.6)).toBe('1분'); // 반올림 60초 → 1분
    expect(formatDurationKo(75)).toBe('1분 15초');
    expect(formatDurationKo(150)).toBe('2분 30초');
  });

  it('시간·분 표기 — 초는 버리고 0분인 하위 단위는 생략', () => {
    expect(formatDurationKo(3600)).toBe('1시간');
    expect(formatDurationKo(3700)).toBe('1시간 1분'); // 61분 40초 → 초 버림
    expect(formatDurationKo(7200)).toBe('2시간');
  });

  it('음수/비수치는 0초로 방어한다', () => {
    expect(formatDurationKo(-5)).toBe('0초');
    expect(formatDurationKo(Number.NaN)).toBe('0초');
    expect(formatDurationKo(Number.POSITIVE_INFINITY)).toBe('0초');
  });
});

describe('formatSimClock', () => {
  it('0.1초 단위 s 접미사 리드아웃', () => {
    expect(formatSimClock(0)).toBe('0s');
    expect(formatSimClock(12.34)).toBe('12.3s');
    expect(formatSimClock(2)).toBe('2s');
  });

  it('음수/비수치는 0s로 방어한다', () => {
    expect(formatSimClock(-1)).toBe('0s');
    expect(formatSimClock(Number.NaN)).toBe('0s');
  });
});

describe('formatDateTimeKo', () => {
  it('유효한 ISO는 YYYY-MM-DD HH:mm 로컬 표기다', () => {
    // 시간대 무관 검증 — 포맷 모양만 고정한다
    expect(formatDateTimeKo('2026-08-07T04:00:00.000Z')).toMatch(
      /^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$/,
    );
  });

  it('파싱 불가면 원문을 그대로 돌려준다 (정보 손실 금지)', () => {
    expect(formatDateTimeKo('not-a-date')).toBe('not-a-date');
  });
});

// ── 통계 숨김 임계값 ────────────────────────────────────────────────

describe('shouldShowStats', () => {
  it(`runCount < ${STATS_MIN_RUN_COUNT}이면 숨긴다 — 작은 표본의 성공률은 노이즈다`, () => {
    expect(shouldShowStats(0)).toBe(false);
    expect(shouldShowStats(STATS_MIN_RUN_COUNT - 1)).toBe(false);
  });

  it(`runCount ≥ ${STATS_MIN_RUN_COUNT}이면 보인다`, () => {
    expect(shouldShowStats(STATS_MIN_RUN_COUNT)).toBe(true);
    expect(shouldShowStats(100)).toBe(true);
  });

  it('임계값은 20이다 (임무 명세 고정값)', () => {
    expect(STATS_MIN_RUN_COUNT).toBe(20);
  });
});

describe('successRatePercent', () => {
  it('표본 0이면 null — 0%로 오독시키지 않는다', () => {
    expect(successRatePercent(0, 0)).toBeNull();
  });

  it('반올림 정수 퍼센트', () => {
    expect(successRatePercent(20, 13)).toBe(65);
    expect(successRatePercent(3, 2)).toBe(67);
    expect(successRatePercent(4, 4)).toBe(100);
  });
});

// ── 개입 정렬 · 라벨 ────────────────────────────────────────────────

describe('sortInterventionsBySimTime', () => {
  const iv = (atSimSec: number, kind: RunIntervention['kind'], nodeId: string | null = null): RunIntervention => ({
    atSimSec,
    kind,
    nodeId,
  });

  it('시각 오름차순으로 정렬한다', () => {
    const input = [iv(5, 'stop'), iv(0, 'play'), iv(2.5, 'pause')];
    expect(sortInterventionsBySimTime(input).map((x) => x.kind)).toEqual([
      'play',
      'pause',
      'stop',
    ]);
  });

  it('동시각은 기록 순서를 유지한다 (안정 정렬 — autoPause 직후 stop 등)', () => {
    const input = [iv(0, 'play'), iv(3, 'autoPause', 'node-0001'), iv(3, 'stop')];
    expect(sortInterventionsBySimTime(input).map((x) => x.kind)).toEqual([
      'play',
      'autoPause',
      'stop',
    ]);
  });

  it('원본 배열을 변경하지 않는다 (사본 반환)', () => {
    const input = [iv(5, 'stop'), iv(0, 'play')];
    const out = sortInterventionsBySimTime(input);
    expect(out).not.toBe(input);
    expect(input.map((x) => x.kind)).toEqual(['stop', 'play']);
  });

  it('빈 목록은 빈 배열', () => {
    expect(sortInterventionsBySimTime([])).toEqual([]);
  });
});

describe('interventionKindKo', () => {
  it('6종 전부 한국어 라벨이 있고 서로 다르다', () => {
    const kinds: RunIntervention['kind'][] = [
      'play',
      'pause',
      'stop',
      'stepNode',
      'runFromNode',
      'autoPause',
    ];
    const labels = kinds.map((k) => interventionKindKo(k));
    for (const label of labels) expect(label.length).toBeGreaterThan(0);
    expect(new Set(labels).size).toBe(kinds.length);
  });
});

// ── 충돌 분류 배지 · 재현 대상 ──────────────────────────────────────

const collision = (
  classification: RunCollision['classification'],
  nodeId: string | null,
): RunCollision => ({
  atSimSec: 2.5,
  entityA: 'arm-6',
  entityB: 'box-1',
  phase: 'start',
  nodeId,
  classification,
});

describe('collisionBadgeSpec', () => {
  it('unexpected → error(COLLISION 램프), intended → neutral(상태 축 바깥)', () => {
    expect(collisionBadgeSpec('unexpected').status).toBe('error');
    expect(collisionBadgeSpec('intended').status).toBe('neutral');
  });

  it('라벨이 분류를 말한다 — 색만으로 전달하지 않는다', () => {
    expect(collisionBadgeSpec('unexpected').labelKo).toBe('예기치 않은 충돌');
    expect(collisionBadgeSpec('intended').labelKo).toBe('의도된 접촉');
  });
});

describe('replayTarget', () => {
  it('예기치 않은 충돌 + 발생 노드 있음 → 그 노드 (킬러 기능의 대상 판정)', () => {
    expect(replayTarget(collision('unexpected', 'node-0003'))).toBe('node-0003');
  });

  it('발생 노드가 없으면 null — 재현 지점이 없다', () => {
    expect(replayTarget(collision('unexpected', null))).toBeNull();
  });

  it('의도된 접촉은 노드가 있어도 null — 실패가 아니다', () => {
    expect(replayTarget(collision('intended', 'node-0003'))).toBeNull();
  });
});
