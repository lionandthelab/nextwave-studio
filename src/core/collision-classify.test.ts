// core/collision-classify.test.ts — 접촉 분류 계약 (순수, DOM/물리 비의존)
//
// 회귀 배경: 모든 접촉이 "충돌"로 집계되어, 로봇이 집으려는 박스에 손을 대는 **성공**이
// 사고로 보고됐다. 사용자 요구: "타겟 목표 외의 다른 것에 부딪히는 경우에만 충돌".

import { describe, expect, it } from 'vitest';
import {
  CONTACT_CLASS_LABEL_KO,
  classifyContact,
  isCollision,
  pairMatches,
} from './collision-classify';
import type { ClassifyContext } from './collision-classify';
import type { CollisionEvent } from '../schema/types';

const GROUND = '__ground__';

function ctx(over: Partial<ClassifyContext> = {}): ClassifyContext {
  return {
    robotIds: new Set(['arm']),
    targetPairs: [['arm', 'box_a']],
    groundId: GROUND,
    ...over,
  };
}

function ev(a: string, b: string, over: Partial<CollisionEvent> = {}): CollisionEvent {
  return { timeSec: 1, a, b, phase: 'start', kind: 'contact', ...over };
}

describe('classifyContact', () => {
  it('시퀀스가 선언한 조작 대상과의 접촉은 성공이다 (충돌 아님)', () => {
    expect(classifyContact(ev('arm', 'box_a'), ctx())).toBe('target');
    // 순서가 뒤집혀도 같은 쌍이다
    expect(classifyContact(ev('box_a', 'arm'), ctx())).toBe('target');
  });

  it('로봇이 타겟 아닌 사물에 부딪히면 충돌이다', () => {
    // 이것이 이번 변경의 핵심: 구 판정은 "동적 사물과의 접촉은 정상 조작"이라며
    // box_b 같은 옆 물건과의 충돌을 통째로 면제했다.
    expect(classifyContact(ev('arm', 'box_b'), ctx())).toBe('unexpected');
  });

  it('로봇이 정적 환경에 부딪히면 충돌이다', () => {
    expect(classifyContact(ev('arm', 'wall'), ctx())).toBe('unexpected');
  });

  it('바닥 접촉은 정상이다 (로봇이 서 있고 물건이 놓여 있다)', () => {
    expect(classifyContact(ev('arm', GROUND), ctx())).toBe('ground');
    expect(classifyContact(ev('box_a', GROUND), ctx())).toBe('ground');
  });

  it('감지 영역 통과는 "부딪힘"이 아니다', () => {
    expect(classifyContact(ev('arm', 'zone', { kind: 'sensor' }), ctx())).toBe('sensor');
    // 물리적 성격이 우선한다: 배리어가 기다리는 쌍이어도 sensor는 sensor다.
    // 어느 쪽이든 충돌이 아니므로 카운터에는 영향이 없고, 로그에는 정보가 더 남는다.
    expect(classifyContact(ev('arm', 'box_a', { kind: 'sensor' }), ctx())).toBe('sensor');
    expect(isCollision('sensor')).toBe(false);
  });

  it('로봇이 관여하지 않은 사물끼리의 접촉은 이 제품의 관심사가 아니다', () => {
    expect(classifyContact(ev('box_a', 'box_b'), ctx())).toBe('incidental');
    expect(classifyContact(ev('box_b', 'box_c'), ctx())).toBe('incidental');
  });

  it('로봇↔로봇은 선언하지 않았다면 충돌이다', () => {
    const two = ctx({ robotIds: new Set(['arm_l', 'arm_r']), targetPairs: [] });
    expect(classifyContact(ev('arm_l', 'arm_r'), two)).toBe('unexpected');
  });

  it('로봇↔로봇도 타겟으로 선언하면 의도된 접촉이다', () => {
    const two = ctx({
      robotIds: new Set(['arm_l', 'arm_r']),
      targetPairs: [['arm_l', 'arm_r']],
    });
    expect(classifyContact(ev('arm_l', 'arm_r'), two)).toBe('target');
  });

  it('같은 엔티티 내부 접촉은 사용자에게 의미가 없다', () => {
    expect(classifyContact(ev('arm', 'arm'), ctx())).toBe('incidental');
  });

  it('타겟 쌍이 없는 씬(접촉 대기 노드 없음)에서는 로봇 접촉이 전부 충돌이다', () => {
    const noTarget = ctx({ targetPairs: [] });
    expect(classifyContact(ev('arm', 'box_a'), noTarget)).toBe('unexpected');
    expect(classifyContact(ev('arm', GROUND), noTarget)).toBe('ground');
  });

  it('타겟 쌍이 여러 개면 전부 인정된다 (시퀀스에 접촉 대기가 여러 번)', () => {
    const multi = ctx({
      targetPairs: [
        ['arm', 'box_a'],
        ['arm', 'box_b'],
      ],
    });
    expect(classifyContact(ev('arm', 'box_a'), multi)).toBe('target');
    expect(classifyContact(ev('arm', 'box_b'), multi)).toBe('target');
    expect(classifyContact(ev('arm', 'box_c'), multi)).toBe('unexpected');
  });

  it('stop phase도 같은 규칙으로 분류된다 (표시는 phase 배지가 담당)', () => {
    expect(classifyContact(ev('arm', 'box_a', { phase: 'stop' }), ctx())).toBe('target');
    expect(classifyContact(ev('arm', 'box_b', { phase: 'stop' }), ctx())).toBe('unexpected');
  });
});

describe('isCollision', () => {
  it('unexpected만 "충돌"로 보고한다', () => {
    expect(isCollision('unexpected')).toBe(true);
    for (const c of ['target', 'ground', 'sensor', 'incidental'] as const) {
      expect(isCollision(c), c).toBe(false);
    }
  });
});

describe('pairMatches', () => {
  it('순서 무관 일치', () => {
    expect(pairMatches(ev('a', 'b'), 'a', 'b')).toBe(true);
    expect(pairMatches(ev('a', 'b'), 'b', 'a')).toBe(true);
    expect(pairMatches(ev('a', 'b'), 'a', 'c')).toBe(false);
  });
});

describe('CONTACT_CLASS_LABEL_KO', () => {
  it('모든 분류에 한국어 라벨이 있고 서로 구별된다 (색 없이 의미 전달)', () => {
    const labels = Object.values(CONTACT_CLASS_LABEL_KO);
    expect(labels).toHaveLength(5);
    expect(new Set(labels).size).toBe(5);
    expect(CONTACT_CLASS_LABEL_KO.unexpected).toBe('충돌');
  });
});
