// ui/viewport/run-overlay.test.ts — 실행 오버레이 순수 헬퍼 단위 테스트 (DOM 비의존, node)
//
// mountRunOverlay의 DOM 조립은 얇은 글루라 여기서 다루지 않는다(브라우저 게이트 몫).
// 여기서는 통합자(main.ts)가 매 tick 밀어넣는 상태 → 텍스트/색/노드-인덱스 매핑 계약을
// 검증한다 (UX_DESIGN §3.3 배지 형식 · §3.6 Collision Log→노드 강조 연동).

import { describe, expect, it } from 'vitest';
import { COLOR } from '../theme';
import {
  announceSignature,
  announceText,
  formatNodeProgress,
  formatSimTime,
  normalizeEngineState,
  overlaySummary,
  stateColor,
  stateDotClass,
  stateLabel,
  timeSecToNodeIndex,
} from './run-overlay';
import type { RunOverlayState } from './run-overlay';

// ── 상태 정규화 / 라벨 / 색 (색만으로 전달 금지 — UX §9) ─────────────

describe('normalizeEngineState', () => {
  it('알려진 3상태는 그대로, 미지 문자열은 idle로 방어', () => {
    expect(normalizeEngineState('playing')).toBe('playing');
    expect(normalizeEngineState('paused')).toBe('paused');
    expect(normalizeEngineState('idle')).toBe('idle');
    expect(normalizeEngineState('')).toBe('idle');
    expect(normalizeEngineState('running')).toBe('idle'); // Engine 상태명이 아님
  });
});

describe('stateLabel', () => {
  it('playing→Running · paused→Paused · idle→Idle', () => {
    expect(stateLabel('playing')).toBe('Running');
    expect(stateLabel('paused')).toBe('Paused');
    expect(stateLabel('idle')).toBe('Idle');
    expect(stateLabel('???')).toBe('Idle');
  });
});

describe('stateColor', () => {
  it('상태별 텍스트 색 토큰 (playing=액센트텍스트 · paused=warn · idle=muted)', () => {
    expect(stateColor('playing')).toBe(COLOR.accentText);
    expect(stateColor('paused')).toBe(COLOR.warn);
    expect(stateColor('idle')).toBe(COLOR.muted);
    expect(stateColor('unknown')).toBe(COLOR.muted);
  });
});

describe('stateDotClass', () => {
  it('theme .ui-dot 계열 클래스 (playing만 펄스)', () => {
    expect(stateDotClass('playing')).toBe('ui-dot ui-dot--playing');
    expect(stateDotClass('paused')).toBe('ui-dot ui-dot--paused');
    expect(stateDotClass('idle')).toBe('ui-dot ui-dot--idle');
    expect(stateDotClass('bogus')).toBe('ui-dot ui-dot--idle');
  });
});

// ── 노드 진행 'node N/M' ────────────────────────────────────────────

describe('formatNodeProgress', () => {
  it('0-기반 인덱스를 1-기반 표시로 (node 3/7)', () => {
    expect(formatNodeProgress(2, 7)).toBe('node 3/7');
    expect(formatNodeProgress(0, 7)).toBe('node 1/7');
  });

  it('시퀀스 끝 인덱스(== count)는 node M/M로 클램프', () => {
    expect(formatNodeProgress(7, 7)).toBe('node 7/7');
    expect(formatNodeProgress(99, 7)).toBe('node 7/7');
  });

  it('음수 인덱스는 최소 1로 클램프', () => {
    expect(formatNodeProgress(-1, 5)).toBe('node 1/5');
  });

  it('시퀀스 없음(null) 또는 count ≤ 0 또는 비유한 → null (세그먼트 미표시)', () => {
    expect(formatNodeProgress(null, 7)).toBeNull();
    expect(formatNodeProgress(2, null)).toBeNull();
    expect(formatNodeProgress(null, null)).toBeNull();
    expect(formatNodeProgress(0, 0)).toBeNull();
    expect(formatNodeProgress(Number.NaN, 7)).toBeNull();
    expect(formatNodeProgress(2, Number.POSITIVE_INFINITY)).toBeNull();
  });

  it('소수 인덱스/카운트는 floor 적용', () => {
    expect(formatNodeProgress(2.9, 7.9)).toBe('node 3/7');
  });
});

// ── simTime 리드아웃 ────────────────────────────────────────────────

describe('formatSimTime', () => {
  it('소수 2자리 고정 + 단위', () => {
    expect(formatSimTime(2.345)).toBe('simTime 2.35s');
    expect(formatSimTime(0)).toBe('simTime 0.00s');
    expect(formatSimTime(12)).toBe('simTime 12.00s');
  });

  it('비유한 입력은 0으로 방어', () => {
    expect(formatSimTime(Number.NaN)).toBe('simTime 0.00s');
    expect(formatSimTime(Number.POSITIVE_INFINITY)).toBe('simTime 0.00s');
  });
});

// ── 충돌 시각 → 활성 노드 인덱스 (Collision Log 행 → 노드 강조) ──────

describe('timeSecToNodeIndex', () => {
  // 노드 0..3이 각각 0.0s / 1.0s / 2.5s / 4.0s에 active가 됐다고 가정 (오름차순 경계)
  const boundaries = [0, 1, 2.5, 4];

  it('경계 이후 시각은 그 이하 마지막 경계의 인덱스', () => {
    expect(timeSecToNodeIndex(0, boundaries)).toBe(0);
    expect(timeSecToNodeIndex(0.9, boundaries)).toBe(0);
    expect(timeSecToNodeIndex(1, boundaries)).toBe(1);
    expect(timeSecToNodeIndex(2.4, boundaries)).toBe(1);
    expect(timeSecToNodeIndex(2.5, boundaries)).toBe(2);
    expect(timeSecToNodeIndex(3.9, boundaries)).toBe(2);
    expect(timeSecToNodeIndex(4, boundaries)).toBe(3);
    expect(timeSecToNodeIndex(100, boundaries)).toBe(3);
  });

  it('첫 경계보다 앞선 시각은 0 (재생 시작 전 — 가장 가까운 노드)', () => {
    expect(timeSecToNodeIndex(-5, boundaries)).toBe(0);
  });

  it('경계가 비어 있으면 -1 (노드 없음)', () => {
    expect(timeSecToNodeIndex(1.5, [])).toBe(-1);
  });

  it('단일 경계', () => {
    expect(timeSecToNodeIndex(5, [0])).toBe(0);
    expect(timeSecToNodeIndex(-1, [0])).toBe(0);
  });
});

// ── 요약 텍스트 (시각 라인/aria 공용) ────────────────────────────────

const baseState: RunOverlayState = {
  engineState: 'playing',
  simTimeSec: 2.34,
  activeNodeLabel: 'MoveJoints',
  nodeIndex: 2,
  nodeCount: 7,
  sceneName: 'arm-and-boxes',
};

describe('overlaySummary', () => {
  it('UX §3.3 형식: 상태 · node N/M · 라벨 · simTime · scene', () => {
    expect(overlaySummary(baseState)).toBe(
      'Running · node 3/7 · MoveJoints · simTime 2.34s · arm-and-boxes',
    );
  });

  it('시퀀스 없음: 노드/라벨 세그먼트 생략', () => {
    expect(
      overlaySummary({
        engineState: 'idle',
        simTimeSec: 0,
        activeNodeLabel: null,
        nodeIndex: null,
        nodeCount: null,
        sceneName: 'falling-boxes',
      }),
    ).toBe('Idle · simTime 0.00s · falling-boxes');
  });

  it('활성 노드 라벨 없음(진행만): 라벨 세그먼트 생략', () => {
    expect(overlaySummary({ ...baseState, activeNodeLabel: null })).toBe(
      'Running · node 3/7 · simTime 2.34s · arm-and-boxes',
    );
  });

  it('빈 씬 이름: scene 세그먼트 생략', () => {
    expect(overlaySummary({ ...baseState, sceneName: '' })).toBe(
      'Running · node 3/7 · MoveJoints · simTime 2.34s',
    );
  });
});

// ── announce (스크린리더 범람 방지 — simTime 제외 시그니처) ──────────

describe('announceSignature / announceText', () => {
  it('시그니처는 simTime을 제외한다 — simTime만 바뀌면 동일', () => {
    const a = announceSignature(baseState);
    const b = announceSignature({ ...baseState, simTimeSec: 99.9 });
    expect(a).toBe(b); // rAF마다 갱신되지 않음 (aria-live 범람 차단)
  });

  it('상태/노드/라벨 전이는 시그니처를 바꾼다', () => {
    const base = announceSignature(baseState);
    expect(announceSignature({ ...baseState, engineState: 'paused' })).not.toBe(base);
    expect(announceSignature({ ...baseState, nodeIndex: 3 })).not.toBe(base);
    expect(announceSignature({ ...baseState, activeNodeLabel: 'Gripper' })).not.toBe(base);
  });

  it('announceText는 simTime 없이 concise (상태 · node · 라벨)', () => {
    expect(announceText(baseState)).toBe('Running · node 3/7 · MoveJoints');
    expect(
      announceText({
        engineState: 'idle',
        simTimeSec: 5,
        activeNodeLabel: null,
        nodeIndex: null,
        nodeCount: null,
        sceneName: 's',
      }),
    ).toBe('Idle');
  });
});
