// ui/console/settings-screen.test.ts — 설정 화면 순수 로직 단위 테스트 (DOM 비의존, node)
//
// mountSettingsScreen의 DOM 조립·배선은 브라우저 게이트 몫이다(toast.test.ts ·
// primitives.test.ts와 같은 관례). 여기서는 이 화면의 순수 계약만 검증한다:
// - PIN 변경 검증 (역할별 자릿수 · 2회 불일치 · 현재와 동일 금지)
// - 사용자 추가 검증 (빈 이름 · PIN 형식)
// - 자기 자신 비활성화 금지 (canDeactivateUser)
// - 용량/동기화/연결 상태 포맷터 (formatBytesKo · syncReportKo · connectionStatusName)
// - lang="en" 판정 (도메인 식별자에만 — WCAG 3.1.2)

import { describe, expect, it } from 'vitest';
import {
  SPEED_MULT_OPTIONS,
  canDeactivateUser,
  checkPinFormat,
  connectionStatusName,
  formatBytesKo,
  formatStorageKo,
  lastSyncTextKo,
  normalizeSpeedMult,
  pinRuleForRole,
  roleLabelKo,
  serverBlockReasonKo,
  speedMultLabel,
  syncReportKo,
  textLang,
  validateNewUser,
  validatePinChange,
} from './settings-screen';

// ── PIN 규칙 (BACKEND §3: tech 4–8 · admin 6–8) ─────────────────────

describe('pinRuleForRole', () => {
  it('admin은 6~8자리, tech는 4~8자리다', () => {
    expect(pinRuleForRole('admin')).toEqual({ minDigits: 6, maxDigits: 8 });
    expect(pinRuleForRole('tech')).toEqual({ minDigits: 4, maxDigits: 8 });
  });
});

describe('checkPinFormat', () => {
  it('빈 PIN을 거부한다', () => {
    const r = checkPinFormat('', 'tech');
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.messageKo).toContain('입력');
  });

  it('숫자 외 문자를 거부한다', () => {
    for (const bad of ['12a4', '12 34', '１２３４', '-1234']) {
      expect(checkPinFormat(bad, 'tech').ok).toBe(false);
    }
  });

  it('역할별 자릿수를 강제한다 — tech 4자리 허용, admin 4자리 거부', () => {
    expect(checkPinFormat('1234', 'tech').ok).toBe(true);
    expect(checkPinFormat('1234', 'admin').ok).toBe(false);
    expect(checkPinFormat('123456', 'admin').ok).toBe(true);
  });

  it('상한(8자리)을 넘으면 거부한다', () => {
    expect(checkPinFormat('123456789', 'tech').ok).toBe(false);
    expect(checkPinFormat('12345678', 'tech').ok).toBe(true);
  });
});

// ── PIN 변경 폼 ─────────────────────────────────────────────────────

describe('validatePinChange', () => {
  const base = { currentPin: '1234', newPin: '567890', newPinRepeat: '567890' } as const;

  it('정상 입력을 통과시킨다', () => {
    expect(validatePinChange({ ...base, role: 'admin' }).ok).toBe(true);
    expect(validatePinChange({ ...base, role: 'tech' }).ok).toBe(true);
  });

  it('현재 PIN이 비어 있으면 거부한다', () => {
    const r = validatePinChange({ ...base, currentPin: '', role: 'tech' });
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.messageKo).toContain('현재 PIN');
  });

  it('새 PIN 2회 입력이 다르면 거부한다', () => {
    const r = validatePinChange({ ...base, newPinRepeat: '567891', role: 'tech' });
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.messageKo).toContain('서로 다릅니다');
  });

  it('새 PIN 자릿수가 역할 규칙에 어긋나면 거부한다 (admin은 6자리 미만 불가)', () => {
    const r = validatePinChange({
      currentPin: '123456',
      newPin: '9876',
      newPinRepeat: '9876',
      role: 'admin',
    });
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.messageKo).toContain('6~8자리');
  });

  it('새 PIN이 현재 PIN과 같으면 거부한다', () => {
    const r = validatePinChange({
      currentPin: '567890',
      newPin: '567890',
      newPinRepeat: '567890',
      role: 'tech',
    });
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.messageKo).toContain('현재 PIN과 같습니다');
  });

  it('자릿수 검증이 2회 일치 검증보다 먼저다 — 형식부터 고치게 안내한다', () => {
    const r = validatePinChange({
      currentPin: '1234',
      newPin: '12',
      newPinRepeat: '34',
      role: 'tech',
    });
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.messageKo).toContain('4~8자리');
  });
});

// ── 사용자 추가 폼 ──────────────────────────────────────────────────

describe('validateNewUser', () => {
  it('정상 입력을 통과시킨다', () => {
    expect(validateNewUser({ name: '김설치', pin: '1234', role: 'tech' }).ok).toBe(true);
    expect(validateNewUser({ name: '관리자', pin: '123456', role: 'admin' }).ok).toBe(true);
  });

  it('빈 이름·공백만 이름을 거부한다 (displayNameSchema 정합)', () => {
    expect(validateNewUser({ name: '', pin: '1234', role: 'tech' }).ok).toBe(false);
    expect(validateNewUser({ name: '   ', pin: '1234', role: 'tech' }).ok).toBe(false);
  });

  it('80자 초과 이름을 거부한다', () => {
    const long = '가'.repeat(81);
    expect(validateNewUser({ name: long, pin: '1234', role: 'tech' }).ok).toBe(false);
    expect(validateNewUser({ name: '가'.repeat(80), pin: '1234', role: 'tech' }).ok).toBe(true);
  });

  it('역할별 PIN 규칙을 적용한다 — admin 4자리 거부', () => {
    expect(validateNewUser({ name: '관리자', pin: '1234', role: 'admin' }).ok).toBe(false);
  });
});

// ── 자기 자신 비활성화 금지 ─────────────────────────────────────────

describe('canDeactivateUser', () => {
  it('타인은 비활성화할 수 있다', () => {
    expect(canDeactivateUser('me-1', 'other-2')).toBe(true);
  });

  it('자기 자신은 비활성화할 수 없다 (마지막 관리자 자기 잠금 방지)', () => {
    expect(canDeactivateUser('me-1', 'me-1')).toBe(false);
  });

  it('로그인 정보가 없으면(null) 비활성화할 수 없다', () => {
    expect(canDeactivateUser(null, 'other-2')).toBe(false);
  });
});

describe('roleLabelKo', () => {
  it('admin=관리자, tech=설치기사 (BACKEND §3)', () => {
    expect(roleLabelKo('admin')).toBe('관리자');
    expect(roleLabelKo('tech')).toBe('설치기사');
  });
});

// ── 용량 포맷터 ─────────────────────────────────────────────────────

describe('formatBytesKo', () => {
  it('1024 미만은 B 단위', () => {
    expect(formatBytesKo(0)).toBe('0 B');
    expect(formatBytesKo(512)).toBe('512 B');
  });

  it('KB/MB/GB로 승급한다', () => {
    expect(formatBytesKo(1024)).toBe('1.0 KB');
    expect(formatBytesKo(1536)).toBe('1.5 KB');
    expect(formatBytesKo(5 * 1024 * 1024)).toBe('5.0 MB');
    expect(formatBytesKo(2.5 * 1024 * 1024 * 1024)).toBe('2.5 GB');
  });

  it('100 이상은 정수로 반올림한다 (소수점 노이즈 제거)', () => {
    expect(formatBytesKo(150 * 1024)).toBe('150 KB');
  });

  it('음수/비정상 값은 알 수 없음', () => {
    expect(formatBytesKo(-1)).toBe('알 수 없음');
    expect(formatBytesKo(Number.NaN)).toBe('알 수 없음');
    expect(formatBytesKo(Number.POSITIVE_INFINITY)).toBe('알 수 없음');
  });
});

describe('formatStorageKo', () => {
  it('추정 불가(null)를 정직하게 말한다', () => {
    expect(formatStorageKo(null)).toBe('확인할 수 없음');
  });

  it('사용량과 전체를 함께 말한다', () => {
    expect(formatStorageKo({ usageBytes: 1536, quotaBytes: 5 * 1024 * 1024 })).toBe(
      '1.5 KB 사용 (전체 5.0 MB)',
    );
  });
});

// ── 연결 상태 매핑 ──────────────────────────────────────────────────

describe('connectionStatusName', () => {
  it('online=success · offline=warn · local=idle', () => {
    expect(connectionStatusName({ mode: 'server', online: true })).toBe('success');
    expect(connectionStatusName({ mode: 'server', online: false })).toBe('warn');
    expect(connectionStatusName({ mode: 'local', online: false })).toBe('idle');
  });
});

describe('serverBlockReasonKo', () => {
  it('온라인이면 차단 사유가 없다(null)', () => {
    expect(serverBlockReasonKo({ mode: 'server', online: true })).toBeNull();
  });

  it('로컬 모드·오프라인은 각각 사람이 읽는 사유를 준다 (회색 버튼 title용)', () => {
    const local = serverBlockReasonKo({ mode: 'local', online: false });
    const offline = serverBlockReasonKo({ mode: 'server', online: false });
    expect(local).toContain('로컬 모드');
    expect(offline).toContain('오프라인');
    expect(local).not.toBe(offline);
  });
});

// ── lang="en" 판정 (WCAG 3.1.2) ─────────────────────────────────────

describe('textLang', () => {
  it('한글이 없는 도메인 식별자에만 en을 준다', () => {
    expect(textLang('Anthropic · claude-sonnet-4-5')).toBe('en');
    expect(textLang('v1.2.3')).toBe('en');
  });

  it('한글이 섞이면 lang을 주지 않는다 (한국어 TTS 유지)', () => {
    expect(textLang('규칙 기반')).toBeUndefined();
    expect(textLang('규칙 기반 (rule-based)')).toBeUndefined();
  });

  it('빈 문자열은 판정하지 않는다', () => {
    expect(textLang('')).toBeUndefined();
  });
});

// ── 재생 속도 기본값 ────────────────────────────────────────────────

describe('normalizeSpeedMult / speedMultLabel', () => {
  it('선택지(1·2·4)는 그대로 통과한다', () => {
    for (const mult of SPEED_MULT_OPTIONS) {
      expect(normalizeSpeedMult(mult)).toBe(mult);
    }
  });

  it('알 수 없는 값은 1×로 정규화한다 (localStorage 손상 방어)', () => {
    expect(normalizeSpeedMult(0)).toBe(1);
    expect(normalizeSpeedMult(3)).toBe(1);
    expect(normalizeSpeedMult(Number.NaN)).toBe(1);
  });

  it('라벨은 × 표기다', () => {
    expect(speedMultLabel(1)).toBe('1×');
    expect(speedMultLabel(4)).toBe('4×');
  });
});

// ── 동기화 문구 ─────────────────────────────────────────────────────

describe('syncReportKo', () => {
  it('보낼 것이 없으면 없다고 말한다', () => {
    expect(syncReportKo({ sentCount: 0, conflictCount: 0, remainingCount: 0 })).toBe(
      '보낼 변경이 없습니다',
    );
  });

  it('전송 건수만 있으면 충돌/대기를 말하지 않는다', () => {
    expect(syncReportKo({ sentCount: 3, conflictCount: 0, remainingCount: 0 })).toBe('3건 전송됨');
  });

  it('충돌·대기가 있으면 함께 말한다', () => {
    expect(syncReportKo({ sentCount: 2, conflictCount: 1, remainingCount: 4 })).toBe(
      '2건 전송됨 · 충돌 1건 · 대기 4건',
    );
  });
});

describe('lastSyncTextKo', () => {
  const nowMs = Date.parse('2026-08-07T12:00:00Z');

  it('기록이 없으면 없다고 말한다', () => {
    expect(lastSyncTextKo(null, nowMs)).toBe('동기화 기록 없음');
  });

  it('기록이 있으면 상대 시각으로 말한다 (lastSyncAgeKo 위임)', () => {
    expect(lastSyncTextKo('2026-08-07T11:55:00Z', nowMs)).toBe('5분 전');
    expect(lastSyncTextKo('2026-08-07T11:59:40Z', nowMs)).toBe('방금 전');
  });
});
