// ui/document.test.ts — 문서 모델 순수 로직 단위 테스트 (DOM/IndexedDB 비의존)
//
// UX_AUDIT C-3: 저장이 시퀀스를 버리고, 새로고침이 전부 지우고, 경고가 없던 3중 결함.
// 여기서는 봉투 해석(하위 호환 포함)·파일명·dirty 판정·상대 시간을 고정한다.

import { describe, expect, it, vi } from 'vitest';
import {
  DOCUMENT_VERSION,
  createDirtyTracker,
  createDocument,
  describeAge,
  documentFileName,
  parseDocument,
  sanitizeDocumentName,
} from './document';
import { DOCUMENT_EXTENSION } from './brand';

// ── 봉투 해석 (하위 호환이 핵심) ────────────────────────────────────

describe('parseDocument', () => {
  it('WorkcellDocument를 해석한다', () => {
    const parsed = parseDocument({
      version: 1,
      name: '픽앤플레이스',
      savedAtIso: '2026-07-28T00:00:00.000Z',
      scene: { name: 'sc' },
      sequence: { steps: [] },
      assets: { 'mesh-1': 'data:model/gltf+json;base64,AAA' },
    });
    expect(parsed.kind).toBe('document');
    expect(parsed.name).toBe('픽앤플레이스');
    expect(parsed.scene).toEqual({ name: 'sc' });
    expect(parsed.sequence).toEqual({ steps: [] });
    expect(parsed.assets['mesh-1']).toContain('data:');
  });

  it('구 봉투 { scene, sequence }를 해석한다 (하위 호환)', () => {
    const parsed = parseDocument({ scene: { name: 'sc' }, sequence: { steps: [1] } });
    expect(parsed.kind).toBe('envelope');
    expect(parsed.scene).toEqual({ name: 'sc' });
    expect(parsed.sequence).toEqual({ steps: [1] });
    expect(parsed.name).toBeNull();
  });

  it('SceneSpec 단독을 해석한다 (구 저장 파일 — 하위 호환)', () => {
    // SceneSpec에는 'scene' 필드가 없으므로 세 형식은 모호하지 않다
    const spec = { name: 'arm-and-boxes', entities: [] };
    const parsed = parseDocument(spec);
    expect(parsed.kind).toBe('bare-scene');
    expect(parsed.scene).toEqual(spec);
    expect(parsed.sequence).toBeNull();
    expect(parsed.assets).toEqual({});
  });

  it('시퀀스 없는 봉투는 sequence가 null이다', () => {
    expect(parseDocument({ scene: {} }).sequence).toBeNull();
  });

  it('null/원시값도 죽지 않는다', () => {
    expect(parseDocument(null).kind).toBe('bare-scene');
    expect(parseDocument(42).scene).toBe(42);
    expect(parseDocument(undefined).sequence).toBeNull();
  });

  it('assets가 객체가 아니면 빈 객체로 방어한다', () => {
    expect(parseDocument({ scene: {}, assets: 'nope' }).assets).toEqual({});
  });

  it('빈 문자열 name은 null로 정규화된다', () => {
    expect(parseDocument({ scene: {}, name: '' }).name).toBeNull();
  });
});

// ── 문서 생성 — 시퀀스가 반드시 함께 저장된다 ───────────────────────

describe('createDocument', () => {
  it('시퀀스를 봉투에 포함한다 (구 저장이 버리던 것)', () => {
    const doc = createDocument({
      name: '테스트',
      scene: { name: 'sc' },
      sequence: { steps: [{ kind: 'wait' }] },
      nowIso: '2026-07-28T12:00:00.000Z',
    });
    expect(doc.version).toBe(DOCUMENT_VERSION);
    expect(doc.sequence).toEqual({ steps: [{ kind: 'wait' }] });
    expect(doc.savedAtIso).toBe('2026-07-28T12:00:00.000Z');
  });

  it('시퀀스가 없으면 null로 명시한다 (undefined로 흘리지 않는다)', () => {
    const doc = createDocument({ name: 'x', scene: {}, sequence: undefined, nowIso: 'i' });
    expect(doc.sequence).toBeNull();
  });

  it('빈 assets는 필드 자체를 만들지 않는다 (파일 크기 절약)', () => {
    const doc = createDocument({ name: 'x', scene: {}, sequence: null, assets: {}, nowIso: 'i' });
    expect('assets' in doc).toBe(false);
  });

  it('assets가 있으면 보존한다 — 다른 세션에서 메시가 사라지지 않는다', () => {
    const doc = createDocument({
      name: 'x',
      scene: {},
      sequence: null,
      assets: { m: 'data:x' },
      nowIso: 'i',
    });
    expect(doc.assets).toEqual({ m: 'data:x' });
  });

  it('왕복(create → JSON → parse)에서 시퀀스가 보존된다', () => {
    const doc = createDocument({
      name: '왕복',
      scene: { name: 'sc' },
      sequence: { steps: [1, 2, 3] },
      nowIso: 'i',
    });
    const parsed = parseDocument(JSON.parse(JSON.stringify(doc)));
    expect(parsed.sequence).toEqual({ steps: [1, 2, 3] });
    expect(parsed.name).toBe('왕복');
  });
});

// ── 파일명 ──────────────────────────────────────────────────────────

describe('sanitizeDocumentName / documentFileName', () => {
  it('파일 시스템 금지 문자를 치환한다', () => {
    expect(sanitizeDocumentName('a/b:c*d?e"f<g>h|i')).toBe('a-b-c-d-e-f-g-h-i');
  });

  it('공백은 하이픈으로, 연속 하이픈은 하나로', () => {
    expect(sanitizeDocumentName('픽 앤   플레이스')).toBe('픽-앤-플레이스');
    expect(sanitizeDocumentName('a---b')).toBe('a-b');
  });

  it('앞뒤 하이픈을 다듬고, 비면 기본값을 준다', () => {
    expect(sanitizeDocumentName('  -x-  ')).toBe('x');
    expect(sanitizeDocumentName('   ')).toBe('workcell');
    expect(sanitizeDocumentName('///')).toBe('workcell');
  });

  it('확장자는 .workcell.json — "이 파일이 전부다"를 이름이 약속한다', () => {
    expect(documentFileName('픽 앤 플레이스')).toBe(`픽-앤-플레이스${DOCUMENT_EXTENSION}`);
    expect(DOCUMENT_EXTENSION).toBe('.workcell.json');
  });
});

// ── dirty 추적 ──────────────────────────────────────────────────────

describe('createDirtyTracker', () => {
  it('markSaved 직후는 깨끗하다', () => {
    const t = createDirtyTracker();
    t.markSaved({ a: 1 }, null);
    expect(t.check({ a: 1 }, null)).toBe(false);
    expect(t.isDirty()).toBe(false);
  });

  it('씬이 바뀌면 dirty', () => {
    const t = createDirtyTracker();
    t.markSaved({ a: 1 }, null);
    expect(t.check({ a: 2 }, null)).toBe(true);
  });

  it('**시퀀스만** 바뀌어도 dirty — 구 저장이 놓치던 축', () => {
    const t = createDirtyTracker();
    t.markSaved({ a: 1 }, { steps: [] });
    expect(t.check({ a: 1 }, { steps: [{ kind: 'wait' }] })).toBe(true);
  });

  it('되돌리면 다시 깨끗해진다', () => {
    const t = createDirtyTracker();
    t.markSaved({ a: 1 }, null);
    expect(t.check({ a: 2 }, null)).toBe(true);
    expect(t.check({ a: 1 }, null)).toBe(false);
  });

  it('onChange는 상태가 바뀔 때만 호출된다 (매 check마다가 아니다)', () => {
    const t = createDirtyTracker();
    const spy = vi.fn();
    t.onChange(spy);
    t.markSaved({ a: 1 }, null);
    t.check({ a: 1 }, null);
    t.check({ a: 1 }, null);
    expect(spy).not.toHaveBeenCalled();
    t.check({ a: 2 }, null);
    expect(spy).toHaveBeenCalledTimes(1);
    expect(spy).toHaveBeenLastCalledWith(true);
    t.check({ a: 3 }, null); // 여전히 dirty — 재호출 없음
    expect(spy).toHaveBeenCalledTimes(1);
    t.check({ a: 1 }, null);
    expect(spy).toHaveBeenCalledTimes(2);
    expect(spy).toHaveBeenLastCalledWith(false);
  });

  it('baseline 없이 check하면 현재 상태를 유지한다 (부팅 중 오탐 방지)', () => {
    const t = createDirtyTracker();
    expect(t.check({ a: 1 }, null)).toBe(false);
  });

  it('직렬화 불가 입력은 보수적으로 dirty 처리한다', () => {
    const t = createDirtyTracker();
    const cyclic: Record<string, unknown> = {};
    cyclic['self'] = cyclic;
    t.markSaved({ a: 1 }, null);
    expect(t.check(cyclic, null)).toBe(true);
  });

  it('reset은 baseline과 dirty를 함께 지운다', () => {
    const t = createDirtyTracker();
    t.markSaved({ a: 1 }, null);
    t.check({ a: 2 }, null);
    expect(t.isDirty()).toBe(true);
    t.reset();
    expect(t.isDirty()).toBe(false);
  });
});

// ── 복원 배너 상대 시간 ─────────────────────────────────────────────

describe('describeAge', () => {
  const now = Date.parse('2026-07-28T12:00:00.000Z');

  it('1분 미만은 "방금 전"', () => {
    expect(describeAge('2026-07-28T11:59:30.000Z', now)).toBe('방금 전');
  });

  it('분/시간/일 단위로 올라간다', () => {
    expect(describeAge('2026-07-28T11:57:00.000Z', now)).toBe('3분 전');
    expect(describeAge('2026-07-28T09:00:00.000Z', now)).toBe('3시간 전');
    expect(describeAge('2026-07-26T12:00:00.000Z', now)).toBe('2일 전');
  });

  it('미래 시각은 "방금 전"으로 클램프한다 (시계 어긋남 방어)', () => {
    expect(describeAge('2026-07-28T13:00:00.000Z', now)).toBe('방금 전');
  });

  it('파싱 불가는 안전한 문구를 준다', () => {
    expect(describeAge('not-a-date', now)).toBe('알 수 없는 시각');
  });
});
