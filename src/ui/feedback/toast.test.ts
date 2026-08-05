// ui/feedback/toast.test.ts — 플래너 UX 순수 헬퍼 단위 테스트 (DOM 비의존, node 환경)
//
// mountToasts/mountNlInput/mountPlannerSettings의 DOM 조립·배선은 브라우저 게이트
// (gate-browser.mjs) 몫이다. 여기서는 이 슬라이스의 순수 계약만 검증한다:
// - 토스트 자동 사라짐 시간 규약 (기본 4s / 오류 8s / 명시 override / 고정)
// - kind → 한국어 라벨 (색 없이 의미 전달 — UX_DESIGN §9)
// - 자연어 생성 게이팅 (빈 입력·busy 시 생성 불가)
// - 플래너 설정 저장 가능성 + 정규화 (Anthropic 키 필수, 모델 기본값)

import { describe, expect, it } from 'vitest';
import {
  TOAST_ACTION_DURATION_MULTIPLIER,
  TOAST_DURATION_DEFAULT_MS,
  TOAST_DURATION_ERROR_MS,
  kindLabelKo,
  resolveToastDurationMs,
  toastAriaLabel,
} from './toast';
import { canGenerate, modeLabelKo } from '../command-bar/nl-input';
import {
  DEFAULT_PLANNER_MODEL,
  canSavePlannerConfig,
  normalizePlannerConfig,
} from '../command-bar/planner-settings';

// ── 토스트: 자동 사라짐 시간 ────────────────────────────────────────

describe('resolveToastDurationMs', () => {
  it('기본은 4s, 오류만 8s (읽을 시간을 더 준다)', () => {
    expect(resolveToastDurationMs('success')).toBe(TOAST_DURATION_DEFAULT_MS);
    expect(resolveToastDurationMs('info')).toBe(TOAST_DURATION_DEFAULT_MS);
    expect(resolveToastDurationMs('warn')).toBe(TOAST_DURATION_DEFAULT_MS);
    expect(resolveToastDurationMs('error')).toBe(TOAST_DURATION_ERROR_MS);
    expect(TOAST_DURATION_DEFAULT_MS).toBe(4000);
    expect(TOAST_DURATION_ERROR_MS).toBe(8000);
  });

  it('opts.durationMs가 숫자면 kind 기본값을 덮는다', () => {
    expect(resolveToastDurationMs('error', { durationMs: 2000 })).toBe(2000);
    expect(resolveToastDurationMs('success', { durationMs: 10000 })).toBe(10000);
  });

  it('durationMs 0/음수는 그대로 반환한다 (고정 토스트 = 자동 사라짐 없음)', () => {
    expect(resolveToastDurationMs('info', { durationMs: 0 })).toBe(0);
    expect(resolveToastDurationMs('error', { durationMs: -1 })).toBe(-1);
  });

  it('detail만 있고 durationMs가 없으면 kind 기본값을 쓴다', () => {
    expect(resolveToastDurationMs('error', { detail: '사유' })).toBe(TOAST_DURATION_ERROR_MS);
  });

  // 액션 토스트: 읽고 → 되돌릴지 판단하고 → 조준하는 3단계가 4초 안에 끝나지 않는다.
  // 사라지면 "복구 경로가 있었다"는 사실 자체가 전달되지 않는다 (UX_AUDIT C-4).
  it('액션이 있으면 자동 사라짐이 2배로 늘어난다', () => {
    const action = { label: '실행 취소', onClick: (): void => {} };
    expect(resolveToastDurationMs('success', { action })).toBe(
      TOAST_DURATION_DEFAULT_MS * TOAST_ACTION_DURATION_MULTIPLIER,
    );
    expect(resolveToastDurationMs('error', { action })).toBe(
      TOAST_DURATION_ERROR_MS * TOAST_ACTION_DURATION_MULTIPLIER,
    );
    expect(TOAST_ACTION_DURATION_MULTIPLIER).toBeGreaterThanOrEqual(2);
  });

  it('명시 durationMs는 액션 배수보다 우선한다 (호출자가 최종 결정권)', () => {
    const action = { label: '실행 취소', onClick: (): void => {} };
    expect(resolveToastDurationMs('success', { action, durationMs: 1000 })).toBe(1000);
    expect(resolveToastDurationMs('success', { action, durationMs: 0 })).toBe(0);
  });
});

describe('toastAriaLabel', () => {
  it('kind 접두 + 메시지 (색 없이 의미 전달 — UX_DESIGN §9)', () => {
    expect(toastAriaLabel('error', '씬을 불러오지 못했습니다')).toBe(
      '오류: 씬을 불러오지 못했습니다',
    );
  });

  it('액션이 있으면 이름에 존재를 명시한다 — role=status의 aria-label이 내용을 대체하므로', () => {
    expect(toastAriaLabel('success', '노드 삭제됨', '실행 취소')).toBe(
      "성공: 노드 삭제됨 — '실행 취소' 버튼 있음",
    );
  });

  it('빈 액션 라벨은 접미를 붙이지 않는다', () => {
    expect(toastAriaLabel('info', '저장됨', '')).toBe('정보: 저장됨');
  });
});

describe('kindLabelKo', () => {
  it('kind마다 색 없이 의미를 전달하는 한국어 라벨을 준다', () => {
    expect(kindLabelKo('success')).toBe('성공');
    expect(kindLabelKo('error')).toBe('오류');
    expect(kindLabelKo('warn')).toBe('경고');
    expect(kindLabelKo('info')).toBe('정보');
  });
});

// ── 자연어 입력: 생성 게이팅 ────────────────────────────────────────

describe('canGenerate', () => {
  it('공백만/빈 입력으로는 생성하지 않는다 (불필요한 플래너 호출 방지)', () => {
    expect(canGenerate('', false)).toBe(false);
    expect(canGenerate('   ', false)).toBe(false);
    expect(canGenerate('\n\t', false)).toBe(false);
  });

  it('내용이 있으면 생성 가능', () => {
    expect(canGenerate('박스를 집어라', false)).toBe(true);
    expect(canGenerate('  box  ', false)).toBe(true);
  });

  it('busy 중이면 내용이 있어도 생성 불가 (중복 트리거 가드)', () => {
    expect(canGenerate('박스를 집어라', true)).toBe(false);
  });
});

describe('modeLabelKo', () => {
  it('생성 모드 한국어 라벨', () => {
    expect(modeLabelKo('replace')).toBe('교체');
    expect(modeLabelKo('append')).toBe('이어서');
  });
});

// ── 플래너 설정: 저장 가능성 + 정규화 ───────────────────────────────

describe('canSavePlannerConfig', () => {
  it('규칙 기반은 키 없이도 항상 저장 가능', () => {
    expect(canSavePlannerConfig({ backend: 'rule-based', apiKey: '', model: '' })).toBe(true);
  });

  it('Anthropic은 키가 비어 있으면(공백 포함) 저장 불가', () => {
    expect(canSavePlannerConfig({ backend: 'anthropic', apiKey: '', model: 'm' })).toBe(false);
    expect(canSavePlannerConfig({ backend: 'anthropic', apiKey: '   ', model: 'm' })).toBe(false);
  });

  it('Anthropic + 키 있으면 저장 가능', () => {
    expect(canSavePlannerConfig({ backend: 'anthropic', apiKey: 'sk-ant-x', model: '' })).toBe(true);
  });
});

describe('normalizePlannerConfig', () => {
  it('모델명 공백은 트림하고, 비면 기본값으로 채운다', () => {
    expect(normalizePlannerConfig({ backend: 'anthropic', apiKey: 'k', model: '  ' }).model).toBe(
      DEFAULT_PLANNER_MODEL,
    );
    expect(normalizePlannerConfig({ backend: 'anthropic', apiKey: 'k', model: '' }).model).toBe(
      DEFAULT_PLANNER_MODEL,
    );
  });

  it('모델명이 있으면 트림만 하고 보존한다', () => {
    expect(
      normalizePlannerConfig({ backend: 'anthropic', apiKey: 'k', model: '  claude-x  ' }).model,
    ).toBe('claude-x');
  });

  it('API 키도 트림한다', () => {
    expect(
      normalizePlannerConfig({ backend: 'anthropic', apiKey: '  sk-ant-x  ', model: 'm' }).apiKey,
    ).toBe('sk-ant-x');
  });

  it('backend는 그대로 보존한다', () => {
    expect(normalizePlannerConfig({ backend: 'rule-based', apiKey: '', model: '' }).backend).toBe(
      'rule-based',
    );
    expect(DEFAULT_PLANNER_MODEL).toBe('claude-opus-5');
  });
});
