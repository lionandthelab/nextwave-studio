// ui/shell/shell.test.ts — 셸/로그인/연결 배지의 순수 로직 단위 테스트 (node, DOM 비의존)
//
// DOM 조립(mountShell/mountLogin)은 브라우저 게이트 몫이다(primitives.test.ts와 같은
// 관례). 여기서는 이 슬라이스의 순수 계약만 검증한다:
// - PIN 입력 (자릿수 제한 · 백스페이스 · 숫자 외 무시 · 패드 배열 3×4)
// - 잠금 카운트다운 계산 (올림 · 하한 0)
// - 아바타 이니셜 · 역할 라벨
// - 연결 배지 3상태 매핑 (라벨 · 상태 축 · 아이콘 · 대기 건수 병기)
// - 레일 모드 (콘솔=full · 스튜디오=thin · 인증=hidden) · 네비 항목 완전성
// - 설치기사 치수 계약 (패드 키 ≥56 · 레일 폭 72)

import { describe, expect, it } from 'vitest';
import {
  AVATAR_SIZE_PX,
  LOGIN_PIN_MIN_LEN,
  PIN_MAX_LEN,
  PIN_PAD_KEY_MIN_PX,
  PIN_PAD_LAYOUT,
  SETUP_PIN_MIN_LEN,
  initialOf,
  lockoutMessageKo,
  lockoutRemainingSec,
  pinAppend,
  pinBackspace,
  roleLabelKo,
} from './login';
import {
  CONNECTION_LABEL_KO,
  connectionBadgeIcon,
  connectionBadgeLabelKo,
  connectionBadgeShortKo,
  connectionBadgeStatus,
} from './connection-badge';
import {
  NAV_ITEMS,
  RAIL_THIN_WIDTH_PX,
  RAIL_WIDTH_PX,
  SCREEN_TITLE_KO,
  railModeForRoute,
} from './shell';
import { CONSOLE_SCREEN_NAMES, ROUTE_NAMES } from './router';
import type { ConnectionState } from '../../api';

const ONLINE: ConnectionState = { mode: 'server', online: true };
const OFFLINE: ConnectionState = { mode: 'server', online: false };
const LOCAL: ConnectionState = { mode: 'local', online: false };

// ── PIN 입력 순수 로직 ──────────────────────────────────────────────

describe('pinAppend', () => {
  it('숫자 한 자리를 덧붙인다', () => {
    expect(pinAppend('', '1')).toBe('1');
    expect(pinAppend('123', '4')).toBe('1234');
  });

  it('최대 자릿수(기본 8)를 넘는 입력은 조용히 무시한다', () => {
    expect(pinAppend('12345678', '9')).toBe('12345678');
    expect(pinAppend('123456', '7', 6)).toBe('123456');
  });

  it('숫자 1글자가 아니면 무시한다', () => {
    expect(pinAppend('12', 'a')).toBe('12');
    expect(pinAppend('12', '')).toBe('12');
    expect(pinAppend('12', '34')).toBe('12');
    expect(pinAppend('12', ' ')).toBe('12');
  });
});

describe('pinBackspace', () => {
  it('마지막 한 자리를 지운다', () => {
    expect(pinBackspace('1234')).toBe('123');
    expect(pinBackspace('1')).toBe('');
  });

  it('빈 문자열은 그대로다', () => {
    expect(pinBackspace('')).toBe('');
  });
});

describe('PIN_PAD_LAYOUT', () => {
  it('3×4 = 12키: 1-9 · 지우기 · 0 · 백스페이스 순서다 (임무 명세 고정)', () => {
    expect(PIN_PAD_LAYOUT).toHaveLength(12);
    const kinds = PIN_PAD_LAYOUT.map((k) => (k.kind === 'digit' ? k.value : k.kind));
    expect(kinds).toEqual(['1', '2', '3', '4', '5', '6', '7', '8', '9', 'clear', '0', 'backspace']);
  });
});

describe('PIN 자릿수 계약 (schema pinSchema \\d{4,8}와 짝)', () => {
  it('로그인 4자리+ · 셋업(관리자) 6자리+ · 최대 8자리', () => {
    expect(LOGIN_PIN_MIN_LEN).toBe(4);
    expect(SETUP_PIN_MIN_LEN).toBe(6);
    expect(PIN_MAX_LEN).toBe(8);
  });
});

// ── 잠금 카운트다운 ─────────────────────────────────────────────────

describe('lockoutRemainingSec', () => {
  it('남은 ms를 초 단위 올림으로 환산한다', () => {
    expect(lockoutRemainingSec(60_000, 0)).toBe(60);
    expect(lockoutRemainingSec(60_000, 500)).toBe(60); // 59.5s → 올림 60
    expect(lockoutRemainingSec(60_000, 59_001)).toBe(1);
    expect(lockoutRemainingSec(60_000, 60_000)).toBe(0);
  });

  it('만료 이후는 음수가 아니라 0이다', () => {
    expect(lockoutRemainingSec(1000, 5000)).toBe(0);
    expect(lockoutRemainingSec(0, 0)).toBe(0);
  });

  it('카운트다운 문구에 남은 초가 들어간다', () => {
    expect(lockoutMessageKo(42)).toContain('42초');
  });
});

// ── 아바타 이니셜 · 역할 라벨 ───────────────────────────────────────

describe('initialOf', () => {
  it('첫 코드포인트를 대문자로 돌려준다 (한글·라틴)', () => {
    expect(initialOf('홍길동')).toBe('홍');
    expect(initialOf('kim')).toBe('K');
    expect(initialOf('  lee ')).toBe('L');
  });

  it('빈 이름은 ?', () => {
    expect(initialOf('')).toBe('?');
    expect(initialOf('   ')).toBe('?');
  });

  it('서로게이트 쌍(이모지)을 절반으로 쪼개지 않는다', () => {
    expect(initialOf('👷반장')).toBe('👷');
  });
});

describe('roleLabelKo', () => {
  it('admin=관리자, tech=설치기사', () => {
    expect(roleLabelKo('admin')).toBe('관리자');
    expect(roleLabelKo('tech')).toBe('설치기사');
  });
});

// ── 연결 배지 매핑 ──────────────────────────────────────────────────

describe('connection badge 매핑', () => {
  it('좁은 레일용 짧은 라벨 — 아이콘만 남기지 않는다 (연결 여부가 현장의 핵심 정보)', () => {
    expect(connectionBadgeShortKo(ONLINE)).toBe('연결됨');
    expect(connectionBadgeShortKo(OFFLINE)).toBe('오프라인');
    expect(connectionBadgeShortKo(LOCAL)).toBe('로컬');
  });

  it('대기 건수가 있으면 상태보다 건수를 앞세운다 (아직 안 올라간 작업이 있다)', () => {
    expect(connectionBadgeShortKo(OFFLINE, 3)).toBe('대기 3');
    expect(connectionBadgeShortKo(ONLINE, 1)).toBe('대기 1');
    expect(connectionBadgeShortKo(ONLINE, 0)).toBe('연결됨');
  });

  it('3상태 라벨이 계약과 일치한다', () => {
    expect(connectionBadgeLabelKo(ONLINE)).toBe('서버 연결됨');
    expect(connectionBadgeLabelKo(OFFLINE)).toBe('오프라인 — 로컬 저장 중');
    expect(connectionBadgeLabelKo(LOCAL)).toBe('로컬 모드');
    expect(Object.keys(CONNECTION_LABEL_KO).sort()).toEqual(['local-only', 'offline', 'online']);
  });

  it('상태 축: online=success · offline=warn · local-only=neutral(의도된 상태)', () => {
    expect(connectionBadgeStatus(ONLINE)).toBe('success');
    expect(connectionBadgeStatus(OFFLINE)).toBe('warn');
    expect(connectionBadgeStatus(LOCAL)).toBe('neutral');
  });

  it('동기화 대기 건수가 있으면 라벨에 병기된다', () => {
    expect(connectionBadgeLabelKo(OFFLINE, 3)).toBe('오프라인 — 로컬 저장 중 · 대기 3건');
    expect(connectionBadgeLabelKo(ONLINE, 1)).toBe('서버 연결됨 · 대기 1건');
    expect(connectionBadgeLabelKo(ONLINE, 0)).toBe('서버 연결됨');
  });

  it('아이콘: 대기 있으면 sync, online=check, 그 외 cloudOff', () => {
    expect(connectionBadgeIcon(ONLINE)).toBe('check');
    expect(connectionBadgeIcon(OFFLINE)).toBe('cloudOff');
    expect(connectionBadgeIcon(LOCAL)).toBe('cloudOff');
    expect(connectionBadgeIcon(OFFLINE, 2)).toBe('sync');
    expect(connectionBadgeIcon(ONLINE, 2)).toBe('sync');
  });
});

// ── 셸 레일 ─────────────────────────────────────────────────────────

describe('railModeForRoute', () => {
  it('콘솔 화면 6종은 full', () => {
    for (const name of CONSOLE_SCREEN_NAMES) {
      expect(railModeForRoute(name)).toBe('full');
    }
  });

  it('studio는 thin(얇은 오버레이 레일), login/setup은 hidden', () => {
    expect(railModeForRoute('studio')).toBe('thin');
    expect(railModeForRoute('login')).toBe('hidden');
    expect(railModeForRoute('setup')).toBe('hidden');
  });

  it('로컬 모드의 스튜디오에서는 레일을 감춘다 (정적 배포 화면 무변경 — BACKEND §1)', () => {
    expect(railModeForRoute('studio', 'local')).toBe('hidden');
    // 서버 모드에서는 그대로 얇은 레일
    expect(railModeForRoute('studio', 'server')).toBe('thin');
    // 콘솔 라우트를 해시로 직접 열었다면 로컬에서도 돌아갈 길을 남긴다 (막다른 길 금지)
    expect(railModeForRoute('tasks', 'local')).toBe('full');
    // 인증 화면은 모드와 무관하게 숨김
    expect(railModeForRoute('login', 'local')).toBe('hidden');
  });

  it('전 라우트가 셋 중 하나로 정해진다 (누락 없음)', () => {
    for (const name of ROUTE_NAMES) {
      expect(['full', 'thin', 'hidden']).toContain(railModeForRoute(name));
    }
  });
});

describe('NAV_ITEMS / SCREEN_TITLE_KO', () => {
  it('네비 레일은 콘솔 화면 6종을 모두, 정확히 한 번씩 담는다', () => {
    const names = NAV_ITEMS.map((i) => i.name);
    expect(new Set(names).size).toBe(names.length);
    expect([...names].sort()).toEqual([...CONSOLE_SCREEN_NAMES].sort());
  });

  it('임무 명세의 순서·아이콘 배정과 일치한다', () => {
    expect(NAV_ITEMS.map((i) => `${i.name}:${i.iconName}`)).toEqual([
      'processes:factory',
      'tasks:clipboard',
      'blocks:puzzle',
      'devices:plug',
      'runs:history',
      'settings:settings',
    ]);
  });

  it('제목은 전부 비어 있지 않은 한국어 라벨이다', () => {
    for (const name of CONSOLE_SCREEN_NAMES) {
      expect(SCREEN_TITLE_KO[name].length).toBeGreaterThan(0);
    }
  });
});

// ── 설치기사 치수 계약 ──────────────────────────────────────────────

describe('치수 계약 (BACKEND §1 — 공유 단말·장갑)', () => {
  it('PIN 패드 키 ≥ 56px, 아바타 ≥ 48px', () => {
    expect(PIN_PAD_KEY_MIN_PX).toBeGreaterThanOrEqual(56);
    expect(AVATAR_SIZE_PX).toBeGreaterThanOrEqual(48);
  });

  it('레일 폭 72px, 얇은 레일도 터치 타깃 44px 이상', () => {
    expect(RAIL_WIDTH_PX).toBe(72);
    expect(RAIL_THIN_WIDTH_PX).toBeGreaterThanOrEqual(44);
  });
});
