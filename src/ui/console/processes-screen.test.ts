// ui/console/processes-screen.test.ts — 공정 화면 순수 로직 단위 테스트 (DOM 비의존, node)
//
// mountProcessesScreen의 DOM 조립·배선은 브라우저 게이트 몫이다(toast.test.ts 관례).
// 여기서는 이 화면의 순수 계약만 검증한다:
// - 규칙 폼 정규화 (speedLimit select 문자열 ↔ ProcessRules.speedLimitMult 왕복,
//   미지 값은 **보수적 1× 폴백** — 속도 상한에서 관대한 폴백은 위험하다)
// - 소속 작업 카운트 (processId null = 무소속 제외)
// - 장비 체크리스트 토글 (중복 없음 · 순서 보존)
// - 문서 초안 정규화 (trim · deviceIds 중복 제거)
// - 카드 보조행 텍스트 (장비 수 미상은 '?')
// - 서버 쓰기 차단 사유 (local/offline은 사유 문자열, online은 null)

import { describe, expect, it } from 'vitest';
import type { ConnectionState } from '../../api';
import {
  DEFAULT_PROCESS_RULES,
  SPEED_LIMIT_OPTIONS,
  buildProcessDoc,
  countTasksByProcess,
  normalizeProcessRules,
  parseSpeedLimitValue,
  processCardSublines,
  serverBlockReasonKo,
  speedLimitOptionValue,
  toggleDeviceId,
} from './processes-screen';

// ── 규칙 기본값 ─────────────────────────────────────────────────────

describe('DEFAULT_PROCESS_RULES', () => {
  it('충돌 자동 정지는 켜짐이 안전 기본이다', () => {
    expect(DEFAULT_PROCESS_RULES.autoPauseOnCollision).toBe(true);
  });

  it('속도 상한 기본은 제한 없음(null)', () => {
    expect(DEFAULT_PROCESS_RULES.speedLimitMult).toBeNull();
  });
});

// ── 속도 상한 select ↔ 스키마 값 ────────────────────────────────────

describe('speedLimit 옵션 매핑', () => {
  it('선택지는 스키마 주석(1|2|4|null)과 1:1이다', () => {
    expect(SPEED_LIMIT_OPTIONS.map((o) => o.mult)).toEqual([1, 2, 4, null]);
  });

  it('value → mult → value 왕복이 무손실이다', () => {
    for (const opt of SPEED_LIMIT_OPTIONS) {
      expect(parseSpeedLimitValue(opt.value)).toBe(opt.mult);
      expect(speedLimitOptionValue(opt.mult)).toBe(opt.value);
    }
  });

  it('미지 select 값은 가장 보수적인 1×로 접는다', () => {
    expect(parseSpeedLimitValue('')).toBe(1);
    expect(parseSpeedLimitValue('8')).toBe(1);
    expect(parseSpeedLimitValue('unlimited')).toBe(1);
  });

  it('선택지에 없는 수치(예: 3)도 1× 옵션으로 접는다 — 제한 없음으로 승격하지 않는다', () => {
    expect(speedLimitOptionValue(3)).toBe('1');
    expect(speedLimitOptionValue(8)).toBe('1');
  });
});

describe('normalizeProcessRules', () => {
  it('체크박스 + select 문자열을 ProcessRules로 정규화한다', () => {
    expect(
      normalizeProcessRules({ autoPauseOnCollision: true, speedLimitValue: 'none' }),
    ).toEqual({ autoPauseOnCollision: true, speedLimitMult: null });
    expect(
      normalizeProcessRules({ autoPauseOnCollision: false, speedLimitValue: '4' }),
    ).toEqual({ autoPauseOnCollision: false, speedLimitMult: 4 });
  });
});

// ── 소속 작업 카운트 ────────────────────────────────────────────────

describe('countTasksByProcess', () => {
  it('processId별로 집계하고 무소속(null)은 제외한다', () => {
    const counts = countTasksByProcess([
      { processId: 'proc_a' },
      { processId: 'proc_a' },
      { processId: 'proc_b' },
      { processId: null },
    ]);
    expect(counts.get('proc_a')).toBe(2);
    expect(counts.get('proc_b')).toBe(1);
    expect(counts.size).toBe(2);
  });

  it('빈 목록은 빈 맵', () => {
    expect(countTasksByProcess([]).size).toBe(0);
  });
});

// ── 장비 체크리스트 토글 ────────────────────────────────────────────

describe('toggleDeviceId', () => {
  it('체크하면 뒤에 추가된다 (기존 순서 보존)', () => {
    expect(toggleDeviceId(['a', 'b'], 'c', true)).toEqual(['a', 'b', 'c']);
  });

  it('이미 있는 id를 다시 체크해도 중복이 생기지 않는다', () => {
    expect(toggleDeviceId(['a', 'b'], 'a', true)).toEqual(['b', 'a']);
  });

  it('해제하면 제거된다', () => {
    expect(toggleDeviceId(['a', 'b', 'c'], 'b', false)).toEqual(['a', 'c']);
  });

  it('없는 id 해제는 no-op', () => {
    expect(toggleDeviceId(['a'], 'x', false)).toEqual(['a']);
  });
});

// ── 문서 초안 정규화 ────────────────────────────────────────────────

describe('buildProcessDoc', () => {
  it('이름/설명을 trim하고 deviceIds 중복을 제거한다', () => {
    const scene = { entities: [] };
    const doc = buildProcessDoc({
      id: 'proc_12345678',
      name: '  1라인 포장  ',
      descriptionKo: ' 설명 ',
      scene,
      deviceIds: ['dev_a', 'dev_a', 'dev_b'],
      rules: { autoPauseOnCollision: true, speedLimitMult: 2 },
    });
    expect(doc.name).toBe('1라인 포장');
    expect(doc.descriptionKo).toBe('설명');
    expect(doc.deviceIds).toEqual(['dev_a', 'dev_b']);
    expect(doc.scene).toBe(scene); // 씬 봉투는 해석하지 않고 그대로 (BACKEND §1.3)
    expect(doc.rules).toEqual({ autoPauseOnCollision: true, speedLimitMult: 2 });
  });

  it('rules는 복사된다 — 폼 상태와 문서가 참조를 공유하지 않는다', () => {
    const rules = { autoPauseOnCollision: false, speedLimitMult: null };
    const doc = buildProcessDoc({
      id: 'proc_12345678',
      name: 'x',
      descriptionKo: '',
      scene: null,
      deviceIds: [],
      rules,
    });
    expect(doc.rules).not.toBe(rules);
    expect(doc.rules).toEqual(rules);
  });
});

// ── 카드 보조행 ─────────────────────────────────────────────────────

describe('processCardSublines', () => {
  const nowMs = Date.parse('2026-08-07T12:05:00Z');
  const base = {
    descriptionKo: '',
    deviceCount: 2,
    taskCount: 3,
    updatedAtIso: '2026-08-07T12:00:00Z',
    updatedByName: '김기사',
  };

  it('장비/작업 수와 수정 시각·수정자를 표기한다', () => {
    expect(processCardSublines(base, nowMs)).toEqual([
      '장비 2대 · 작업 3개',
      '수정 5분 전 · 김기사',
    ]);
  });

  it('설명이 있으면 첫 행에 온다', () => {
    const lines = processCardSublines({ ...base, descriptionKo: '포장 라인' }, nowMs);
    expect(lines[0]).toBe('포장 라인');
    expect(lines).toHaveLength(3);
  });

  it('문서 로드 실패(deviceCount null)는 ?로 정직하게 표기한다', () => {
    const lines = processCardSublines({ ...base, deviceCount: null }, nowMs);
    expect(lines[0]).toBe('장비 ?대 · 작업 3개');
  });
});

// ── 서버 쓰기 차단 사유 ─────────────────────────────────────────────

describe('serverBlockReasonKo', () => {
  it('server + online이면 null (사용 가능)', () => {
    const state: ConnectionState = { mode: 'server', online: true };
    expect(serverBlockReasonKo(state)).toBeNull();
  });

  it('local 모드는 서버 미설정 사유를 준다', () => {
    const state: ConnectionState = { mode: 'local', online: false };
    expect(serverBlockReasonKo(state)).toContain('로컬 모드');
  });

  it('offline은 재연결 안내 사유를 준다', () => {
    const state: ConnectionState = { mode: 'server', online: false };
    expect(serverBlockReasonKo(state)).toContain('오프라인');
  });
});
