// ui/run-recorder.test.ts — RunRecorder 수명 주기·검증·방어 동작 (순수, node 환경)
//
// 관례: toast.test.ts와 동일 — DOM 없이 순수 계약만 검증한다. 이 레코더의 계약:
// - begin → record* → finish 수명 주기가 runRecordSchema를 통과하는 RunRecord를 만든다
// - begin 없이 들어온 이벤트는 조용히 무시된다 (기록 오염 방어)
// - finish는 한 번만 유효하다 (한 실행 = 한 기록, runs는 append-only)
// - begin 재호출은 진행 중 기록을 폐기한다 (재생 재시작 의미론)

import { describe, expect, it } from 'vitest';
import { RunRecorder } from './run-recorder';
import type { RunBeginInfo, RunFinishInfo } from './run-recorder';
import { runRecordSchema } from '../schema/entities';
import type { RunCollision } from '../schema/entities';

// entityIdSchema는 min(8)이다 — 픽스처 id는 8자 이상으로 만든다
const BEGIN: RunBeginInfo = {
  taskId: 'task-00000001',
  taskName: '픽앤플레이스',
  taskVersion: 3,
  processId: 'proc-00000001',
  operatorId: 'user-00000001',
  operatorName: '김설치',
  stepsTotal: 5,
  startedAtIso: '2026-08-07T04:00:00.000Z',
};

const FINISH: RunFinishInfo = {
  endedAtIso: '2026-08-07T04:01:30.000Z',
  simTimeSec: 42.5,
  wallTimeSec: 90,
};

const COLLISION: RunCollision = {
  atSimSec: 2.5,
  entityA: 'arm-6',
  entityB: 'box-1',
  phase: 'start',
  nodeId: 'node-0003',
  classification: 'unexpected',
};

describe('RunRecorder — 전체 수명 주기', () => {
  it('begin → record* → finish가 스키마를 통과하는 RunRecord를 만든다', () => {
    const recorder = new RunRecorder();
    expect(recorder.isActive()).toBe(false);

    recorder.begin(BEGIN);
    expect(recorder.isActive()).toBe(true);

    recorder.recordIntervention('play', null, 0);
    recorder.recordCollision(COLLISION);
    recorder.recordIntervention('autoPause', 'node-0003', 2.5);
    recorder.noteStepDone(3);

    const record = recorder.finish('autoPaused', FINISH);
    expect(record).not.toBeNull();
    expect(recorder.isActive()).toBe(false);

    // 자체 검증 계약 — 산출물은 이미 스키마를 통과한 값이다
    expect(() => runRecordSchema.parse(record)).not.toThrow();

    expect(record?.taskId).toBe(BEGIN.taskId);
    expect(record?.taskName).toBe(BEGIN.taskName);
    expect(record?.taskVersion).toBe(3);
    expect(record?.processId).toBe(BEGIN.processId);
    expect(record?.operatorId).toBe(BEGIN.operatorId);
    expect(record?.operatorName).toBe(BEGIN.operatorName);
    expect(record?.startedAtIso).toBe(BEGIN.startedAtIso);
    expect(record?.endedAtIso).toBe(FINISH.endedAtIso);
    expect(record?.result).toBe('autoPaused');
    expect(record?.stepsTotal).toBe(5);
    expect(record?.stepsDone).toBe(3);
    expect(record?.simTimeSec).toBe(42.5);
    expect(record?.wallTimeSec).toBe(90);
    expect(record?.collisions).toEqual([COLLISION]);
    // 개입은 삽입 순서 그대로 보존된다 (정렬은 화면 몫)
    expect(record?.interventions).toEqual([
      { atSimSec: 0, kind: 'play', nodeId: null },
      { atSimSec: 2.5, kind: 'autoPause', nodeId: 'node-0003' },
    ]);
  });

  it('id는 begin 시점에 발급된다 — 기본은 crypto.randomUUID(36자), 주입 가능', () => {
    const recorder = new RunRecorder();
    recorder.begin(BEGIN);
    const record = recorder.finish('completed', FINISH);
    expect(record?.id).toHaveLength(36); // uuid v4 형식

    const fixed = new RunRecorder(() => 'fixed-id-0001');
    fixed.begin(BEGIN);
    expect(fixed.finish('completed', FINISH)?.id).toBe('fixed-id-0001');
  });

  it('processId가 null인 자유 작업도 기록된다', () => {
    const recorder = new RunRecorder(() => 'fixed-id-0002');
    recorder.begin({ ...BEGIN, processId: null });
    const record = recorder.finish('completed', FINISH);
    expect(record?.processId).toBeNull();
  });
});

describe('RunRecorder — 방어 동작', () => {
  it('begin 없이 들어온 이벤트는 무시된다 (기록 오염 방어)', () => {
    const recorder = new RunRecorder();
    // 재생이 아닌 경로에서 새어 들어온 이벤트들 — 전부 무시
    recorder.recordIntervention('play', null, 0);
    recorder.recordCollision(COLLISION);
    recorder.noteStepDone(4);

    recorder.begin(BEGIN);
    const record = recorder.finish('completed', FINISH);
    expect(record?.collisions).toEqual([]);
    expect(record?.interventions).toEqual([]);
    expect(record?.stepsDone).toBe(0);
  });

  it('begin 없이 finish하면 null이다', () => {
    const recorder = new RunRecorder();
    expect(recorder.finish('completed', FINISH)).toBeNull();
  });

  it('이중 finish — 두 번째는 null이다 (한 실행 = 한 기록)', () => {
    const recorder = new RunRecorder();
    recorder.begin(BEGIN);
    expect(recorder.finish('stopped', FINISH)).not.toBeNull();
    expect(recorder.finish('stopped', FINISH)).toBeNull();
  });

  it('begin 재호출은 진행 중 기록을 폐기하고 새로 시작한다 (재생 재시작)', () => {
    const recorder = new RunRecorder();
    recorder.begin(BEGIN);
    recorder.recordCollision(COLLISION);
    recorder.noteStepDone(2);

    recorder.begin({ ...BEGIN, taskName: '두 번째 실행' });
    const record = recorder.finish('completed', FINISH);
    expect(record?.taskName).toBe('두 번째 실행');
    expect(record?.collisions).toEqual([]); // 첫 실행의 충돌이 새어 들어오지 않는다
    expect(record?.stepsDone).toBe(0);
  });

  it('noteStepDone은 [0, stepsTotal]로 클램프하고 정수로 내린다', () => {
    const recorder = new RunRecorder();

    recorder.begin(BEGIN); // stepsTotal 5
    recorder.noteStepDone(-3);
    expect(recorder.finish('stopped', FINISH)?.stepsDone).toBe(0);

    recorder.begin(BEGIN);
    recorder.noteStepDone(7.9); // floor(7.9)=7 → cap 5
    expect(recorder.finish('completed', FINISH)?.stepsDone).toBe(5);

    recorder.begin(BEGIN);
    recorder.noteStepDone(2.9);
    expect(recorder.finish('stopped', FINISH)?.stepsDone).toBe(2);
  });

  it('무효 입력은 finish의 스키마 검증에서 throw로 드러난다 (조용한 오염 금지)', () => {
    const recorder = new RunRecorder();
    recorder.begin({ ...BEGIN, taskId: 'x' }); // entityIdSchema min(8) 위반
    expect(() => recorder.finish('completed', FINISH)).toThrow();
    // throw 후에도 재-finish는 불가 — 오염된 기록을 재사용하지 않는다
    expect(recorder.finish('completed', FINISH)).toBeNull();
    expect(recorder.isActive()).toBe(false);
  });
});
