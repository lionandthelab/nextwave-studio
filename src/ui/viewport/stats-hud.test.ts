// ui/viewport/stats-hud.test.ts — 실행 계측 HUD 순수 헬퍼 단위 테스트 (DOM 비의존, node)
//
// mountStatsHud의 DOM 조립은 얇은 글루라 여기서 다루지 않는다(브라우저 게이트 몫).
// 여기서 지키는 계약은 둘이다 (UX_AUDIT C-15):
//   (1) RTF 계산 — 0으로 나누기/비유한 입력이 리드아웃으로 새지 않는다
//   (2) 자릿수 폭 고정 — 값이 바뀌어도 문자열 길이가 변하지 않는다.
//       매 프레임 갱신되는 수치가 좌우로 떨리면 그 자체가 신뢰도 문제다.

import { describe, expect, it } from 'vitest';
import { COLLISION, COLOR } from '../theme';
import {
  computeRtf,
  formatCount,
  formatFps,
  formatPhysicsMs,
  formatRtf,
  formatStatsLine,
  rtfColor,
} from './stats-hud';
import type { StatsSample } from './stats-hud';

// ── RTF 계산 ────────────────────────────────────────────────────────

describe('computeRtf', () => {
  it('실시간이면 1.0 — simTime 진행 / 벽시계 경과', () => {
    expect(computeRtf(1, 1)).toBe(1);
    expect(computeRtf(0.5, 0.5)).toBe(1);
  });

  it('물리가 벽시계를 못 따라가면 1 미만', () => {
    expect(computeRtf(0.5, 1)).toBe(0.5);
    expect(computeRtf(0.25, 1)).toBe(0.25);
  });

  it('물리가 앞서면 1 초과 (배속 재생/가벼운 씬)', () => {
    expect(computeRtf(2, 1)).toBe(2);
  });

  it('벽시계 경과 0 이하는 0 — Infinity가 리드아웃으로 새지 않는다', () => {
    expect(computeRtf(1, 0)).toBe(0);
    expect(computeRtf(1, -1)).toBe(0);
  });

  it('비유한 입력은 0으로 방어', () => {
    expect(computeRtf(Number.NaN, 1)).toBe(0);
    expect(computeRtf(1, Number.NaN)).toBe(0);
    expect(computeRtf(Number.POSITIVE_INFINITY, 1)).toBe(0);
  });

  it('음수 sim 진행은 0 (시간 역행은 계측 대상이 아니다)', () => {
    expect(computeRtf(-1, 1)).toBe(0);
  });
});

// ── 자릿수 폭 고정 (떨림 방지 — 파일 헤더 (2)) ──────────────────────

describe('formatFps', () => {
  it('소수 1자리 + 폭 5 고정', () => {
    expect(formatFps(59.83)).toBe(' 59.8');
    expect(formatFps(120)).toBe('120.0');
    expect(formatFps(9.8)).toBe('  9.8');
    expect(formatFps(0)).toBe('  0.0');
  });

  it('값이 자릿수를 넘나들어도 문자열 길이가 같다 (리드아웃이 밀리지 않는다)', () => {
    const widths = [0, 9.8, 59.8, 120, 144.4].map((v) => formatFps(v).length);
    expect(new Set(widths).size).toBe(1);
  });

  it('미측정/비유한은 대시로 폭을 유지한다', () => {
    expect(formatFps(Number.NaN)).toBe('    —');
    expect(formatFps(-1)).toBe('    —');
    expect(formatFps(Number.NaN).length).toBe(formatFps(59.8).length);
  });
});

describe('formatRtf', () => {
  it('소수 2자리 + × 단위, 폭 고정', () => {
    expect(formatRtf(1)).toBe(' 1.00×');
    expect(formatRtf(0.42)).toBe(' 0.42×');
    expect(formatRtf(12.5)).toBe('12.50×');
  });

  it('실시간 이하/이상 전이에서 폭이 유지된다', () => {
    const widths = [0, 0.5, 1, 9.99].map((v) => formatRtf(v).length);
    expect(new Set(widths).size).toBe(1);
  });

  it('비유한/음수는 대시', () => {
    expect(formatRtf(Number.NaN)).toBe('    —×');
    expect(formatRtf(Number.POSITIVE_INFINITY)).toBe('    —×');
  });
});

describe('formatPhysicsMs', () => {
  it('소수 1자리 + ms/f 단위, 폭 고정', () => {
    expect(formatPhysicsMs(2)).toBe('  2.0ms/f');
    expect(formatPhysicsMs(16.67)).toBe(' 16.7ms/f');
    expect(formatPhysicsMs(100)).toBe('100.0ms/f');
  });

  it('비유한은 대시', () => {
    expect(formatPhysicsMs(Number.NaN)).toBe('    —ms/f');
  });
});

describe('formatCount', () => {
  it('정수 + 폭 4 고정', () => {
    expect(formatCount(63)).toBe('  63');
    expect(formatCount(7)).toBe('   7');
    expect(formatCount(1024)).toBe('1024');
  });

  it('소수는 반올림한다 (개수는 정수다)', () => {
    expect(formatCount(62.6)).toBe('  63');
  });

  it('엔티티가 늘어도 폭이 유지된다', () => {
    const widths = [0, 9, 63, 999].map((v) => formatCount(v).length);
    expect(new Set(widths).size).toBe(1);
  });

  it('비유한/음수는 대시', () => {
    expect(formatCount(Number.NaN)).toBe('   —');
    expect(formatCount(-3)).toBe('   —');
  });
});

// ── RTF 건전성 색 (색은 보조 채널 — 숫자가 1차 채널) ────────────────

describe('rtfColor', () => {
  it('실시간(≥0.9)은 기본 텍스트 색', () => {
    expect(rtfColor(1)).toBe(COLOR.text);
    expect(rtfColor(0.9)).toBe(COLOR.text);
    expect(rtfColor(2)).toBe(COLOR.text);
  });

  it('0.9 미만은 경고', () => {
    expect(rtfColor(0.89)).toBe(COLOR.warnText);
    expect(rtfColor(0.5)).toBe(COLOR.warnText);
  });

  it('0.5 미만은 충돌 램프 (가장 강한 경보 색)', () => {
    expect(rtfColor(0.49)).toBe(COLLISION.text);
    expect(rtfColor(0)).toBe(COLLISION.text);
  });

  it('미측정은 muted', () => {
    expect(rtfColor(Number.NaN)).toBe(COLOR.muted);
  });
});

// ── 한 줄 요약 (접힘 툴팁) ──────────────────────────────────────────

describe('formatStatsLine', () => {
  const sample: StatsSample = {
    fps: 59.83,
    rtf: 1,
    physicsMsPerFrame: 2,
    entityCount: 63,
    colliderCount: 71,
  };

  it('UX_AUDIT C-15 제안 형식 (패딩 없는 조밀 표기)', () => {
    expect(formatStatsLine(sample)).toBe(
      'FPS 59.8 · RTF 1.00× · 물리 2.0ms/f · 엔티티 63 · 콜라이더 71',
    );
  });

  it('미측정 필드는 대시로 나타난다 (0으로 위장하지 않는다)', () => {
    expect(formatStatsLine({ ...sample, fps: Number.NaN })).toBe(
      'FPS — · RTF 1.00× · 물리 2.0ms/f · 엔티티 63 · 콜라이더 71',
    );
  });
});
