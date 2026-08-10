// ui/command-bar/json-viewer.test.ts — JSON 편집기 순수 헬퍼 단위 테스트 (DOM 비의존, node)
//
// 관례: playback.test.ts / layout.test.ts와 동일 — mountJsonViewer의 DOM 조립·배선은
// 브라우저 게이트 몫이고, 여기서는 순수 계약만 검증한다:
// - 시퀀스 → 텍스트 (들여쓰기 2 · null은 빈 문자열)
// - JSON 파싱 오류 → **줄 번호로 환산한 한국어 한 줄** (off-by-one이 나기 쉬운 지점)
// - 버전 행 요약 문구

import { describe, expect, it } from 'vitest';
import { jsonErrorKo, sequenceToText, versionSummaryKo } from './json-viewer';
import type { ControlSequence } from '../../schema';
import type { SequenceVersion } from '../sequence-versions';

// ── sequenceToText ──────────────────────────────────────────────────

describe('sequenceToText', () => {
  const seq: ControlSequence = {
    id: 'flow-graph',
    robot: 'arm',
    steps: [{ kind: 'wait', durationSec: 1 }],
  };

  it('null은 빈 문자열 — 편집기 기준선과 짝이 맞아야 한다', () => {
    expect(sequenceToText(null)).toBe('');
  });

  it('들여쓰기 2칸으로 pretty-print한다', () => {
    const text = sequenceToText(seq);
    expect(text.split('\n')[1]).toBe('  "id": "flow-graph",');
    expect(JSON.parse(text)).toEqual(seq); // 왕복 무손실
  });
});

// ── jsonErrorKo ─────────────────────────────────────────────────────

describe('jsonErrorKo', () => {
  /** 실제 엔진이 던지는 오류를 쓴다 — 메시지 형식을 추측하지 않는다 */
  function parseError(text: string): unknown {
    try {
      JSON.parse(text);
    } catch (err) {
      return err;
    }
    throw new Error('테스트 픽스처가 유효한 JSON이다 — 오류를 만들지 못했다');
  }

  it('입력이 끝까지 닫히지 않으면 마지막 줄을 짚는다 (실제 V8 오류)', () => {
    const text = '{ "id": ';
    const msg = jsonErrorKo(parseError(text), text);
    expect(msg).toContain('JSON 형식 오류');
    expect(msg).toContain('닫히지 않았습니다');
    expect(msg).toContain('1번째 줄');
  });

  it('여러 줄에서 오류 줄을 짚는다 (실제 V8 오류 — 발췌 기반)', () => {
    // 4번째 줄에서 값이 빠져 5번째 줄의 '}'에서 파싱이 실패한다
    const text = '{\n  "a": 1,\n  "b": 2,\n  "c":\n}';
    const msg = jsonErrorKo(parseError(text), text);
    expect(msg).toContain('JSON 형식 오류');
    // 엔진이 오류 지점 또는 그 직전을 가리키므로 4~5줄 범위로 고정한다
    expect(/[45]번째 줄/.test(msg)).toBe(true);
  });

  it('구 V8 형식(position N)도 계속 지원한다', () => {
    const text = '{\n  "a": 1,\n  "b":\n}';
    // 오프셋 17 = 3번째 줄의 ':' / 19 = 4번째 줄의 '}'
    expect(jsonErrorKo(new Error('Unexpected token } in JSON at position 17'), text)).toBe(
      'JSON 형식 오류 — 3번째 줄 근처를 확인하세요',
    );
    expect(jsonErrorKo(new Error('Unexpected token } in JSON at position 19'), text)).toBe(
      'JSON 형식 오류 — 4번째 줄 근처를 확인하세요',
    );
  });

  it('Firefox 형식(line N column M)도 지원한다', () => {
    expect(
      jsonErrorKo(new Error('JSON.parse: expected property name at line 4 column 3'), 'a\nb\nc\nd'),
    ).toBe('JSON 형식 오류 — 4번째 줄 근처를 확인하세요');
  });

  it('position이 텍스트 길이를 넘어도 마지막 줄로 클램프된다', () => {
    expect(jsonErrorKo(new Error('bad at position 999'), 'a\nb')).toBe(
      'JSON 형식 오류 — 2번째 줄 근처를 확인하세요',
    );
  });

  it('위치를 알 수 없어도 영문 원문을 노출하지 않는다 (§4-b)', () => {
    const msg = jsonErrorKo(new Error('Something went terribly wrong'), '{}');
    expect(msg).toBe('JSON 형식 오류 — 괄호·쉼표·따옴표를 확인하세요');
    expect(msg).not.toContain('Something');
  });

  it('Error가 아닌 값도 받아낸다', () => {
    expect(jsonErrorKo('문자열 오류', '{}')).toBe('JSON 형식 오류 — 괄호·쉼표·따옴표를 확인하세요');
  });
});

// ── versionSummaryKo ────────────────────────────────────────────────

describe('versionSummaryKo', () => {
  it('버전·시각·노드 수를 한 줄로 묶는다', () => {
    const entry: SequenceVersion = {
      version: 3,
      atIso: '2026-08-10T09:00:00.000Z',
      labelKo: 'JSON 직접 편집',
      stepCount: 12,
      sequence: { id: 'flow-graph', robot: 'arm', steps: [] },
    };
    expect(versionSummaryKo(entry, '5분 전')).toBe('v3 · 5분 전 · 노드 12개');
  });
});
