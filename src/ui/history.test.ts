// ui/history.test.ts — SceneHistory 스냅샷 스택 단위 테스트 (DOM/물리 비의존)
//
// SceneSpec은 형태만 맞으면 되므로 최소 유효 스펙을 팩토리로 만든다. restore는
// 기록형 fake — 호출 스펙과 성공/실패를 제어한다. 디바운스는 vitest 페이크 타이머.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { HISTORY_CAP_DEFAULT, HISTORY_DEBOUNCE_MS_DEFAULT, SceneHistory } from './history';
import type { HistorySnapshot } from './history';
import type { ControlSequence, SceneSpec } from '../schema/types';

function spec(name: string): SceneSpec {
  return { name, version: 1, gravity: [0, -9.81, 0], timestepHz: 240, entities: [] };
}

function sequence(robot: string): ControlSequence {
  return { name: 'seq', version: 1, robot, steps: [] } as unknown as ControlSequence;
}

/** 스냅샷은 **씬 + 시퀀스** 한 덩어리다 (UX_AUDIT C-4) */
function snap(name: string, seq: ControlSequence | null = null): HistorySnapshot {
  return { scene: spec(name), sequence: seq };
}

interface FakeRestore {
  restore: (s: HistorySnapshot) => Promise<boolean>;
  calls: HistorySnapshot[];
  result: boolean;
}

function makeRestore(): FakeRestore {
  const state: FakeRestore = {
    calls: [],
    result: true,
    restore: (s: HistorySnapshot): Promise<boolean> => {
      state.calls.push(s);
      return Promise.resolve(state.result);
    },
  };
  return state;
}

describe('SceneHistory', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it('스냅샷은 시퀀스를 함께 담는다 — 플로우 그래프 편집도 Undo 대상이다 (UX_AUDIT C-4)', async () => {
    const restore = makeRestore();
    const history = new SceneHistory(restore);
    history.reset(snap('base', sequence('arm')));
    // 씬은 그대로, **시퀀스만** 바뀐 편집 (노드 삭제·재정렬 등)
    history.noteChange(() => snap('base', sequence('arm2')));
    vi.advanceTimersByTime(HISTORY_DEBOUNCE_MS_DEFAULT);
    expect(history.canUndo).toBe(true);
    await history.undo();
    const restored = restore.calls[0];
    expect(restored?.scene.name).toBe('base');
    expect((restored?.sequence as { robot: string } | null)?.robot).toBe('arm');
  });

  it('시퀀스 없는 스냅샷도 왕복한다 (sequence: null 보존)', async () => {
    const restore = makeRestore();
    const history = new SceneHistory(restore);
    history.reset(snap('base', sequence('arm')));
    history.noteChange(() => snap('edit', null));
    vi.advanceTimersByTime(HISTORY_DEBOUNCE_MS_DEFAULT);
    await history.undo();
    expect(restore.calls[0]?.sequence).not.toBeNull();
    await history.redo();
    expect(restore.calls[1]?.sequence).toBeNull();
  });

  it('reset 직후에는 undo/redo 불가 (기준 스냅샷 1장뿐)', () => {
    const history = new SceneHistory(makeRestore());
    history.reset(snap('base'));
    expect(history.canUndo).toBe(false);
    expect(history.canRedo).toBe(false);
  });

  it('noteChange는 디바운스 후 스냅샷을 쌓고 undo 가능해진다', () => {
    const history = new SceneHistory(makeRestore());
    history.reset(snap('base'));
    history.noteChange(() => snap('edit1'));
    expect(history.canUndo).toBe(false); // 아직 디바운스 대기
    vi.advanceTimersByTime(HISTORY_DEBOUNCE_MS_DEFAULT);
    expect(history.canUndo).toBe(true);
  });

  it('burst(연속 noteChange)는 최종 스냅샷 1장으로 합쳐진다 (capture는 매번 즉시 실행)', async () => {
    const restore = makeRestore();
    const history = new SceneHistory(restore);
    history.reset(snap('base'));
    const firstCapture = vi.fn(() => snap('a'));
    history.noteChange(firstCapture);
    history.noteChange(() => snap('b'));
    vi.advanceTimersByTime(HISTORY_DEBOUNCE_MS_DEFAULT);
    expect(firstCapture).toHaveBeenCalledTimes(1); // 즉시 캡처 — 상태 시점 고정
    expect(history.canUndo).toBe(true);
    // undo 1회로 base까지 돌아간다 = burst가 1장(최종 'b' 상태)이었다
    await history.undo();
    expect(restore.calls.map((s) => s.scene.name)).toEqual(['base']);
    expect(history.canUndo).toBe(false);
    // redo 대상은 burst의 최종 상태 'b'다 ('a'는 합쳐져 사라짐)
    await history.redo();
    expect(restore.calls.map((s) => s.scene.name)).toEqual(['base', 'b']);
  });

  it('flushPending으로 구조 변경 경계를 세우면 직전 burst가 별도 스냅샷으로 남는다', async () => {
    const restore = makeRestore();
    const history = new SceneHistory(restore, { debounceMs: 300 });
    history.reset(snap('base'));
    // 연속 조정 burst (디바운스 미발화 상태)
    history.noteChange(() => snap('resize'));
    // 구조 변경(remove) — 통합자 규약: flush → note → flush
    history.flushPending();
    history.noteChange(() => snap('removed'));
    history.flushPending();
    // undo 1회 = remove 취소 → 'resize' 상태 복원 (burst가 유실되지 않았다)
    await history.undo();
    expect(restore.calls.map((s) => s.scene.name)).toEqual(['resize']);
  });

  it('undo는 이전 스냅샷으로 restore하고 redo 가능해진다', async () => {
    const restore = makeRestore();
    const history = new SceneHistory(restore, { debounceMs: 10 });
    history.reset(snap('base'));
    history.noteChange(() => snap('edit1'));
    vi.advanceTimersByTime(10);

    const ok = await history.undo();
    expect(ok).toBe(true);
    expect(restore.calls.map((s) => s.scene.name)).toEqual(['base']);
    expect(history.canUndo).toBe(false);
    expect(history.canRedo).toBe(true);
  });

  it('redo는 되돌린 스냅샷을 다시 restore한다', async () => {
    const restore = makeRestore();
    const history = new SceneHistory(restore, { debounceMs: 10 });
    history.reset(snap('base'));
    history.noteChange(() => snap('edit1'));
    vi.advanceTimersByTime(10);
    await history.undo();

    const ok = await history.redo();
    expect(ok).toBe(true);
    expect(restore.calls.map((s) => s.scene.name)).toEqual(['base', 'edit1']);
    expect(history.canUndo).toBe(true);
    expect(history.canRedo).toBe(false);
  });

  it('undo 진입 시 pending burst를 flush한다 — 마지막 편집도 되돌릴 수 있다', async () => {
    const restore = makeRestore();
    const history = new SceneHistory(restore, { debounceMs: 1000 });
    history.reset(snap('base'));
    history.noteChange(() => snap('edit1'));
    // 디바운스가 아직 발화하지 않은 상태에서 바로 undo
    const ok = await history.undo();
    expect(ok).toBe(true);
    expect(restore.calls.map((s) => s.scene.name)).toEqual(['base']);
    expect(history.canRedo).toBe(true); // flush된 edit1이 redo 대상
  });

  it('restore 실패(false) 시 스택이 원위치로 돌아간다', async () => {
    const restore = makeRestore();
    restore.result = false;
    const history = new SceneHistory(restore, { debounceMs: 10 });
    history.reset(snap('base'));
    history.noteChange(() => snap('edit1'));
    vi.advanceTimersByTime(10);

    const ok = await history.undo();
    expect(ok).toBe(false);
    expect(history.canUndo).toBe(true); // 이동 취소 — 여전히 undo 가능
    expect(history.canRedo).toBe(false);
  });

  it('새 편집은 redo 가지를 무효화한다 (선형 히스토리)', async () => {
    const history = new SceneHistory(makeRestore(), { debounceMs: 10 });
    history.reset(snap('base'));
    history.noteChange(() => snap('edit1'));
    vi.advanceTimersByTime(10);
    await history.undo();
    expect(history.canRedo).toBe(true);

    history.noteChange(() => snap('edit2'));
    vi.advanceTimersByTime(10);
    expect(history.canRedo).toBe(false);
  });

  it('cancelPending은 대기 중 스냅샷을 폐기한다 (씬 teardown 시 stale 스냅샷 방지)', () => {
    const history = new SceneHistory(makeRestore(), { debounceMs: 10 });
    history.reset(snap('base'));
    const capture = vi.fn(() => snap('stale'));
    history.noteChange(capture); // 캡처는 즉시 1회 실행되지만 커밋은 디바운스 대기
    history.cancelPending();
    vi.advanceTimersByTime(50);
    expect(capture).toHaveBeenCalledTimes(1);
    expect(history.canUndo).toBe(false); // 폐기됨 — 스택에 쌓이지 않았다
  });

  it('cap 초과 시 가장 오래된 스냅샷부터 버린다', () => {
    const history = new SceneHistory(makeRestore(), { debounceMs: 10, cap: 3 });
    history.reset(snap('base'));
    for (let i = 0; i < 5; i += 1) {
      history.noteChange(() => snap(`edit${i}`));
      vi.advanceTimersByTime(10);
    }
    // cap=3 → 스택 [edit2, edit3, edit4] — undo 2회만 가능
    expect(history.canUndo).toBe(true);
    expect(HISTORY_CAP_DEFAULT).toBeGreaterThan(3); // 기본 상한은 더 크다 (계약 문서화)
  });

  it('복원 스냅샷은 deep clone이다 — 반환 후 변형이 스택을 오염시키지 않는다', async () => {
    const restore = makeRestore();
    const history = new SceneHistory(restore, { debounceMs: 10 });
    history.reset(snap('base'));
    history.noteChange(() => snap('edit1'));
    vi.advanceTimersByTime(10);
    await history.undo();
    const delivered = restore.calls[0];
    expect(delivered).toBeDefined();
    if (delivered) delivered.scene.name = 'mutated';

    restore.calls.length = 0;
    await history.redo(); // edit1 복원
    await history.undo(); // base 복원 — 오염되지 않은 이름이어야 한다
    expect(restore.calls.map((s) => s.scene.name)).toEqual(['edit1', 'base']);
  });
});
