// ui/command-bar/playback.test.ts — 트랜스포트 토글 판정 (DOM 비의존, node)
//
// 회귀 배경: 시퀀스가 완주(done)해도 UI가 영원히 "실행 중"으로 남고, Space가
// "다시 실행"이 아니라 "물리 일시정지"로 동작했다. 원인은 토글이 **물리 엔진 상태**를
// 기준으로 판정한 것 — 물리는 시퀀스가 끝나도 계속 도는 것이 정상이다(오브젝트가
// 계속 정착해야 한다). 시퀀스가 있으면 시퀀스 진행 여부가 기준이어야 한다.

import { describe, expect, it } from 'vitest';
import { nextTransportAction, physicsStateLabel } from './playback';

describe('nextTransportAction', () => {
  it('시퀀스가 돌면 일시정지한다', () => {
    expect(nextTransportAction(true, 'playing')).toBe('pause');
  });

  it('시퀀스가 멈춰 있으면 재생한다 — 물리가 돌고 있어도', () => {
    // 이것이 이번 회귀의 핵심이다: 완주 직후 physics는 여전히 'playing'이지만
    // 사용자가 기대하는 Space의 의미는 "다시 실행"이다.
    expect(nextTransportAction(false, 'playing')).toBe('play');
  });

  it('시퀀스가 멈춰 있고 물리도 멈춰 있으면 재생한다', () => {
    expect(nextTransportAction(false, 'paused')).toBe('play');
    expect(nextTransportAction(false, 'idle')).toBe('play');
  });

  it('시퀀스가 없는 씬(null)에서는 물리 상태가 기준이다', () => {
    expect(nextTransportAction(null, 'playing')).toBe('pause');
    expect(nextTransportAction(null, 'paused')).toBe('play');
    expect(nextTransportAction(null, 'idle')).toBe('play');
  });

  it('시퀀스 축이 물리 축을 항상 이긴다 (두 축이 어긋나도 결정론적)', () => {
    for (const engine of ['idle', 'playing', 'paused'] as const) {
      expect(nextTransportAction(true, engine)).toBe('pause');
      expect(nextTransportAction(false, engine)).toBe('play');
    }
  });
});

describe('physicsStateLabel', () => {
  it('물리 상태에 "물리" 스코프를 붙여 시퀀스 어휘와 분리한다', () => {
    // 커맨드바가 "Running", 뷰포트 오버레이가 "Idle"을 동시에 말하던 모순의 해소
    for (const state of ['idle', 'playing', 'paused'] as const) {
      expect(physicsStateLabel(state)).toMatch(/^물리 /);
    }
    expect(new Set((['idle', 'playing', 'paused'] as const).map(physicsStateLabel)).size).toBe(3);
  });
});
