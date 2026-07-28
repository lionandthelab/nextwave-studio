// ui/inspector/inspector.test.ts — 인스펙터 순수 헬퍼 단위 테스트 (DOM 비의존, node)
//
// mountInspector의 DOM 조립은 얇은 글루라 여기서 다루지 않는다(브라우저 게이트 몫).
// 여기서는 포맷팅(quat→euler deg 표시 변환·고정 소수점·리드아웃 문자열)과
// 선택 로직(next/resolve/목록 변경 키)의 계약을 검증한다.

import { describe, expect, it } from 'vitest';
import {
  entityListKey,
  entityTypeMeta,
  formatEulerDeg,
  formatFixed,
  formatLimits,
  formatPosition,
  isBipolarRange,
  matchesEntityFilter,
  nextSelection,
  quatReadout,
  quatToEulerDegXYZ,
  rangeTrackGradient,
  resolveSelection,
} from './inspector';
import type { Quat } from '../../schema/types';

// ── 테스트 전용 쿼터니언 헬퍼 (Hamilton 곱, [x,y,z,w]) ───────────────

function axisQuat(axis: 'x' | 'y' | 'z', deg: number): Quat {
  const halfRad = (deg * Math.PI) / 180 / 2;
  const s = Math.sin(halfRad);
  const c = Math.cos(halfRad);
  if (axis === 'x') return [s, 0, 0, c];
  if (axis === 'y') return [0, s, 0, c];
  return [0, 0, s, c];
}

function mulQuat(a: Quat, b: Quat): Quat {
  const [ax, ay, az, aw] = a;
  const [bx, by, bz, bw] = b;
  return [
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  ];
}

// ── quatToEulerDegXYZ — 표시 전용 변환 (내부 진실은 쿼터니언, CLAUDE.md §4) ─

describe('quatToEulerDegXYZ', () => {
  it('identity → [0, 0, 0]', () => {
    const [x, y, z] = quatToEulerDegXYZ([0, 0, 0, 1]);
    expect(x).toBeCloseTo(0, 9);
    expect(y).toBeCloseTo(0, 9);
    expect(z).toBeCloseTo(0, 9);
  });

  it('단일 축 회전을 복원한다 (X/Y/Z 각 90°)', () => {
    const [rx] = quatToEulerDegXYZ(axisQuat('x', 90));
    expect(rx).toBeCloseTo(90, 6);

    const [, ry] = quatToEulerDegXYZ(axisQuat('y', 90));
    expect(ry).toBeCloseTo(90, 6);

    const [, , rz] = quatToEulerDegXYZ(axisQuat('z', 90));
    expect(rz).toBeCloseTo(90, 6);
  });

  it('합성 회전(XYZ 순서 30°/40°/20°)을 각도별로 복원한다', () => {
    // XYZ(Tait-Bryan) 순서: q = qx ⊗ qy ⊗ qz — 추출식과 동일 규약
    const q = mulQuat(mulQuat(axisQuat('x', 30), axisQuat('y', 40)), axisQuat('z', 20));
    const [x, y, z] = quatToEulerDegXYZ(q);
    expect(x).toBeCloseTo(30, 6);
    expect(y).toBeCloseTo(40, 6);
    expect(z).toBeCloseTo(20, 6);
  });

  it('짐벌락(y=90°)에서 z=0으로 고정하고 x를 복원한다', () => {
    const q = mulQuat(axisQuat('x', 30), axisQuat('y', 90));
    const [x, y, z] = quatToEulerDegXYZ(q);
    expect(x).toBeCloseTo(30, 5);
    expect(y).toBeCloseTo(90, 5);
    expect(z).toBe(0);
  });

  it('비정규 쿼터니언도 내부 정규화로 같은 결과를 낸다', () => {
    const [x, y, z, w] = axisQuat('y', 90);
    const SCALE = 7.3;
    const scaled: Quat = [x * SCALE, y * SCALE, z * SCALE, w * SCALE];
    const [, ry] = quatToEulerDegXYZ(scaled);
    expect(ry).toBeCloseTo(90, 6);
  });

  it('노름 0(무효) 쿼터니언은 [0,0,0]으로 방어한다 (표시 전용 계약)', () => {
    expect(quatToEulerDegXYZ([0, 0, 0, 0])).toEqual([0, 0, 0]);
  });
});

// ── 숫자/리드아웃 포맷팅 ────────────────────────────────────────────

describe('formatFixed', () => {
  it('지정 자릿수로 고정 소수점 포맷한다', () => {
    expect(formatFixed(1.23456, 3)).toBe('1.235');
    expect(formatFixed(0, 3)).toBe('0.000');
    expect(formatFixed(-1.5, 2)).toBe('-1.50');
  });

  it('반올림 결과가 0이면 음의 부호를 지운다 ("-0.000" 방지)', () => {
    expect(formatFixed(-0.0004, 3)).toBe('0.000');
    expect(formatFixed(-0, 3)).toBe('0.000');
    expect(formatFixed(-0.4, 0)).toBe('0');
  });

  it('0이 아닌 음수의 부호는 유지한다', () => {
    expect(formatFixed(-0.001, 3)).toBe('-0.001');
  });
});

describe('formatPosition / formatEulerDeg / quatReadout', () => {
  it('position 리드아웃: 미터 소수 3자리', () => {
    expect(formatPosition([0.4, 0.05, 0])).toBe('X 0.400 · Y 0.050 · Z 0.000');
  });

  it('오일러 리드아웃: deg 소수 1자리 + ° 접미', () => {
    expect(formatEulerDeg([0, 90, -0.04])).toBe('RX 0.0° · RY 90.0° · RZ 0.0°');
  });

  it('쿼터니언 툴팁: 원본 [x,y,z,w] 소수 4자리', () => {
    expect(quatReadout([0, 0, 0, 1])).toBe('quat [0.0000, 0.0000, 0.0000, 1.0000]');
  });
});

describe('formatLimits', () => {
  it('limits가 있으면 [lower, upper] 소수 3자리로 표시한다', () => {
    expect(formatLimits([-1.5708, 1.5708])).toBe('[-1.571, 1.571]');
  });

  it('limits가 없으면 — 를 표시한다 (continuous 관절 등)', () => {
    expect(formatLimits(undefined)).toBe('—');
  });
});

// ── 선택 로직 ───────────────────────────────────────────────────────

describe('nextSelection — 목록 행 클릭 의미론', () => {
  it('다른 행 클릭 → 그 행 선택', () => {
    expect(nextSelection(null, 'box_a')).toBe('box_a');
    expect(nextSelection('arm', 'box_a')).toBe('box_a');
  });

  it('선택된 행 재클릭 → 해제(null)', () => {
    expect(nextSelection('box_a', 'box_a')).toBeNull();
  });
});

describe('resolveSelection — 목록 대비 선택 정규화', () => {
  const IDS = ['arm', 'box_a', 'box_b'] as const;

  it('목록에 있는 id는 유지한다', () => {
    expect(resolveSelection(IDS, 'box_b')).toBe('box_b');
  });

  it('목록에 없는 id는 해제(null)로 정규화한다 (씬 리로드 등)', () => {
    expect(resolveSelection(IDS, 'ghost')).toBeNull();
    expect(resolveSelection([], 'arm')).toBeNull();
  });

  it('null 요청은 null 그대로', () => {
    expect(resolveSelection(IDS, null)).toBeNull();
  });
});

describe('entityListKey — 목록 변경 감지', () => {
  it('id/type/순서가 같으면 키가 같다 (목록 DOM 재구축 생략 조건)', () => {
    const a = entityListKey([
      { id: 'arm', type: 'robot' },
      { id: 'box_a', type: 'object' },
    ]);
    const b = entityListKey([
      { id: 'arm', type: 'robot' },
      { id: 'box_a', type: 'object' },
    ]);
    expect(a).toBe(b);
  });

  it('type 변경·순서 변경·항목 증감을 각각 구별한다', () => {
    const base = entityListKey([
      { id: 'arm', type: 'robot' },
      { id: 'box_a', type: 'object' },
    ]);
    expect(
      entityListKey([
        { id: 'arm', type: 'static' },
        { id: 'box_a', type: 'object' },
      ]),
    ).not.toBe(base);
    expect(
      entityListKey([
        { id: 'box_a', type: 'object' },
        { id: 'arm', type: 'robot' },
      ]),
    ).not.toBe(base);
    expect(entityListKey([{ id: 'arm', type: 'robot' }])).not.toBe(base);
  });
});

describe('entityTypeMeta', () => {
  it('알려진 type은 SVG 아이콘 이름 + 한국어 라벨', () => {
    expect(entityTypeMeta('robot')).toEqual({ icon: 'robotArm', label: '로봇' });
    expect(entityTypeMeta('object')).toEqual({ icon: 'shapeBox', label: '사물' });
    expect(entityTypeMeta('static')).toEqual({ icon: 'shapePlane', label: '정적' });
  });

  it('미지 type은 원문 라벨로 폴백한다', () => {
    expect(entityTypeMeta('conveyor')).toEqual({ icon: 'layers', label: 'conveyor' });
  });

  it('아이콘은 이모지가 아니라 icons.ts의 이름이다 (UX_AUDIT C-13)', () => {
    // 이모지는 currentColor를 못 받아 hover/선택에서 텍스트만 밝아지고 OS마다 그림이 다르다
    for (const type of ['robot', 'object', 'static', 'conveyor']) {
      expect(entityTypeMeta(type).icon).toMatch(/^[a-zA-Z]+$/);
    }
  });
});

// ── 아웃라이너 검색/필터 (UX_AUDIT C-10) ────────────────────────────

describe('matchesEntityFilter', () => {
  const ARM = { id: 'arm', type: 'robot' };
  const BOX = { id: 'box_a', type: 'object' };

  it('검색어가 비면 전부 통과한다', () => {
    expect(matchesEntityFilter(ARM, '', new Set())).toBe(true);
    expect(matchesEntityFilter(ARM, '   ', new Set())).toBe(true);
  });

  it('id 부분 일치 (대소문자 무시)', () => {
    expect(matchesEntityFilter(BOX, 'box', new Set())).toBe(true);
    expect(matchesEntityFilter(BOX, 'BOX_A', new Set())).toBe(true);
    expect(matchesEntityFilter(BOX, 'arm', new Set())).toBe(false);
  });

  it('type으로도 검색된다', () => {
    expect(matchesEntityFilter(ARM, 'robot', new Set())).toBe(true);
  });

  it('타입 칩이 켜져 있으면 그 타입만 통과한다 (칩이 없으면 제한 없음)', () => {
    expect(matchesEntityFilter(ARM, '', new Set(['robot']))).toBe(true);
    expect(matchesEntityFilter(BOX, '', new Set(['robot']))).toBe(false);
    expect(matchesEntityFilter(BOX, '', new Set(['robot', 'object']))).toBe(true);
  });

  it('타입 필터와 검색어는 AND로 결합한다', () => {
    expect(matchesEntityFilter(BOX, 'box', new Set(['object']))).toBe(true);
    expect(matchesEntityFilter(BOX, 'arm', new Set(['object']))).toBe(false);
    expect(matchesEntityFilter(BOX, 'box', new Set(['robot']))).toBe(false);
  });
});

// ── 슬라이더 트랙 채움 (UX_AUDIT C-18) ──────────────────────────────

describe('isBipolarRange', () => {
  it('min < 0 < max일 때만 부호 있는 범위다', () => {
    expect(isBipolarRange(-2, 2)).toBe(true);
    expect(isBipolarRange(-1.5, 0.5)).toBe(true);
    expect(isBipolarRange(0, 1)).toBe(false); // 그리퍼 — min→thumb 채움이 옳다
    expect(isBipolarRange(-2, 0)).toBe(false);
    expect(isBipolarRange(0.5, 2)).toBe(false);
  });
});

describe('rangeTrackGradient', () => {
  it('단극(0..1): 0%에서 값까지 채운다', () => {
    const grad = rangeTrackGradient(0.25, 0, 1);
    expect(grad).toContain('linear-gradient');
    expect(grad).toContain('0 25%'); // 채움이 하한에서 시작한다
    expect(grad).not.toContain('calc('); // 틱 레이어 없음
  });

  it('부호 있는 범위: 채움이 0에서 시작한다 (min이 아니라)', () => {
    // limits ±2.0에서 -0.6 → 0 지점은 50%, 값 지점은 35% — 채움은 [35%, 50%]다.
    // min→thumb였다면 "35% 진행"으로 읽혀 부호와 0 기준점이 시각적으로 틀린다.
    const grad = rangeTrackGradient(-0.6, -2, 2);
    expect(grad).toContain('35% 50%');
  });

  it('부호 있는 범위: 0 지점에 1px 틱 레이어를 얹는다', () => {
    const grad = rangeTrackGradient(1, -2, 2);
    expect(grad).toContain('calc(50% - 0.5px)');
    expect(grad).toContain('calc(50% + 0.5px)');
    // 틱 레이어가 채움 레이어보다 앞(위)에 온다
    expect(grad.indexOf('transparent')).toBeLessThan(grad.lastIndexOf('linear-gradient'));
  });

  it('값이 0이면 채움 폭이 0이다 ("절반 열림"으로 보이지 않는다)', () => {
    expect(rangeTrackGradient(0, -2, 2)).toContain('50% 50%');
  });

  it('범위 밖 값은 클램프하고, 폭 0/비유한 범위는 단색으로 방어한다', () => {
    expect(rangeTrackGradient(9, -2, 2)).toContain('50% 100%');
    expect(rangeTrackGradient(0, 1, 1)).not.toContain('linear-gradient');
    expect(rangeTrackGradient(0, Number.NaN, 1)).not.toContain('linear-gradient');
  });
});
