// schema/blocks.test.ts — 재사용 블록 캡처·전개 단위 테스트 (순수 — node 환경)
//
// 계약 (임무 명세 + blocks.ts 헤더):
// - capture→expand 왕복: 깊은 복사(원본 비변형), 전개 결과 = 캡처 원본과 동등
// - 로봇 재매핑: targetRobotId가 robot 필드를 갖는 모든 step에 일괄 기록
// - 파라미터 대입: 수치·문자·불리언 + 'params.' 별칭 + 2단계 경로(targets.joint)
// - 잘못된 경로/타입/범위/stepIndex/잉여 key 거부 (한국어 오류, 부분 결과 없음)
// - 빈 블록 거부 (capture·expand 양쪽)
// - 전개 결과는 controlStepSchema 재검증을 통과해야 한다 (§2.8 정신)

import { describe, expect, it } from 'vitest';
import {
  BINDING_PATH_MAX_SEGMENTS,
  ROBOT_STEP_KINDS,
  blockRefLabel,
  captureBlock,
  checkParamValue,
  collectRobotIds,
  expandBlock,
  parseBindingPath,
} from './blocks';
import type { BlockDoc, BlockParam } from './entities';
import type { ControlStep } from './types';

// ── 픽스처 ──────────────────────────────────────────────────────────

const BLOCK_ID = 'block-test-0001';

function sampleSteps(): ControlStep[] {
  return [
    {
      kind: 'moveJoints',
      robot: 'arm_a',
      targets: { joint1: 0.5, joint2: -0.25 },
      durationSec: 2,
      easing: 'easeInOut',
    },
    { kind: 'wait', durationSec: 1 },
    { kind: 'gripper', state: 'close', durationSec: 0.5 },
  ];
}

function capturedBlock(): BlockDoc {
  const result = captureBlock(sampleSteps(), { name: '집기 동작', id: BLOCK_ID });
  if (!result.ok) throw new Error(`fixture 캡처 실패: ${result.errors.join(' / ')}`);
  return result.block;
}

function withParams(block: BlockDoc, params: BlockParam[]): BlockDoc {
  return { ...block, params };
}

// ── captureBlock ────────────────────────────────────────────────────

describe('captureBlock', () => {
  it('capture → expand 왕복: 전개 결과가 캡처한 step과 동등하다 (robot 유지 시)', () => {
    const block = capturedBlock();
    const expanded = expandBlock(block, { targetRobotId: null, paramValues: {} });
    expect(expanded.ok).toBe(true);
    if (!expanded.ok) return;
    expect(expanded.steps).toEqual(sampleSteps());
    // 깊은 복사 — 블록 내부 step과 같은 참조가 아니다
    expect(expanded.steps[0]).not.toBe(block.steps[0]);
  });

  it('입력 steps를 깊은 복사한다 — 캡처 후 원본을 바꿔도 블록은 불변', () => {
    const steps = sampleSteps();
    const result = captureBlock(steps, { name: '복사 검증', id: BLOCK_ID });
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    const first = steps[0];
    if (first?.kind === 'moveJoints') first.targets['joint1'] = 999;
    const blockFirst = result.block.steps[0];
    expect(blockFirst?.kind).toBe('moveJoints');
    if (blockFirst?.kind === 'moveJoints') {
      expect(blockFirst.targets['joint1']).toBe(0.5);
    }
  });

  it('robot 참조를 수집해 첫 로봇을 robotHint로 기록한다', () => {
    const steps: ControlStep[] = [
      { kind: 'wait', durationSec: 1 },
      { kind: 'setJoints', robot: 'arm_b', targets: { j1: 0 } },
      { kind: 'gripper', robot: 'arm_c', state: 'open' },
    ];
    const result = captureBlock(steps, { name: '두 로봇', id: BLOCK_ID });
    expect(result.ok).toBe(true);
    if (result.ok) expect(result.block.robotHint).toBe('arm_b');
  });

  it('opts.robotHint 명시가 자동 수집보다 우선한다 (null 포함)', () => {
    const explicit = captureBlock(sampleSteps(), {
      name: 'x힌트',
      id: BLOCK_ID,
      robotHint: 'Arm-6',
    });
    expect(explicit.ok && explicit.block.robotHint).toBe('Arm-6');
    const cleared = captureBlock(sampleSteps(), { name: 'x힌트', id: BLOCK_ID, robotHint: null });
    expect(cleared.ok && cleared.block.robotHint).toBe(null);
  });

  it('robot 참조가 없으면 robotHint는 null이다', () => {
    const result = captureBlock([{ kind: 'wait', durationSec: 1 }], { name: '대기', id: BLOCK_ID });
    expect(result.ok && result.block.robotHint).toBe(null);
  });

  it('빈 블록을 거부한다', () => {
    const result = captureBlock([], { name: '빈 블록', id: BLOCK_ID });
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.errors.join(' ')).toContain('최소 1개');
  });

  it('무효 step(스키마 위반)을 한국어 경로와 함께 거부한다', () => {
    const bad = [{ kind: 'wait', durationSec: -1 }] as ControlStep[];
    const result = captureBlock(bad, { name: '음수 대기', id: BLOCK_ID });
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.errors.some((e) => e.includes('steps[0]'))).toBe(true);
      expect(result.errors.some((e) => e.includes('durationSec'))).toBe(true);
    }
  });

  it('빈 이름(공백만)을 거부한다', () => {
    const result = captureBlock(sampleSteps(), { name: '   ', id: BLOCK_ID });
    expect(result.ok).toBe(false);
  });
});

// ── 로봇 재매핑 ─────────────────────────────────────────────────────

describe('expandBlock — 로봇 재매핑', () => {
  it('targetRobotId를 robot 필드를 갖는 모든 step에 일괄 기록한다', () => {
    const block = capturedBlock();
    const expanded = expandBlock(block, { targetRobotId: 'arm_new', paramValues: {} });
    expect(expanded.ok).toBe(true);
    if (!expanded.ok) return;
    for (const step of expanded.steps) {
      if ((ROBOT_STEP_KINDS as readonly string[]).includes(step.kind)) {
        expect('robot' in step && step.robot).toBe('arm_new');
      } else {
        expect('robot' in step).toBe(false); // wait 등은 손대지 않는다
      }
    }
  });

  it('targetRobotId가 null이면 캡처 당시 robot 참조가 유지된다', () => {
    const block = capturedBlock();
    const expanded = expandBlock(block, { targetRobotId: null, paramValues: {} });
    expect(expanded.ok).toBe(true);
    if (!expanded.ok) return;
    const first = expanded.steps[0];
    expect(first?.kind === 'moveJoints' && first.robot).toBe('arm_a');
    const third = expanded.steps[2];
    // 캡처 시 robot이 없던 step에는 새로 생기지 않는다
    expect(third !== undefined && 'robot' in third).toBe(false);
  });
});

// ── 파라미터 대입 ───────────────────────────────────────────────────

describe('expandBlock — 파라미터 대입', () => {
  const durationParam: BlockParam = {
    key: 'moveSec',
    labelKo: '이동 시간',
    kind: 'number',
    defaultValue: 2,
    min: 0.1,
    max: 10,
    bindings: [{ stepIndex: 0, path: 'params.durationSec' }],
  };

  it("수치 대입 — 'params.' 별칭 경로가 step 루트 필드를 가리킨다", () => {
    const block = withParams(capturedBlock(), [durationParam]);
    const expanded = expandBlock(block, { targetRobotId: null, paramValues: { moveSec: 3.5 } });
    expect(expanded.ok).toBe(true);
    if (!expanded.ok) return;
    const first = expanded.steps[0];
    expect(first?.kind === 'moveJoints' && first.durationSec).toBe(3.5);
  });

  it('값 미지정 시 defaultValue를 대입한다', () => {
    const block = withParams(capturedBlock(), [
      { ...durationParam, defaultValue: 7, bindings: [{ stepIndex: 1, path: 'durationSec' }] },
    ]);
    const expanded = expandBlock(block, { targetRobotId: null, paramValues: {} });
    expect(expanded.ok).toBe(true);
    if (!expanded.ok) return;
    const second = expanded.steps[1];
    expect(second?.kind === 'wait' && second.durationSec).toBe(7);
  });

  it('2단계 경로(targets.joint)에 수치를 대입한다', () => {
    const block = withParams(capturedBlock(), [
      {
        key: 'lift',
        labelKo: '들어올림 각',
        kind: 'number',
        defaultValue: 0,
        bindings: [{ stepIndex: 0, path: 'targets.joint2' }],
      },
    ]);
    const expanded = expandBlock(block, { targetRobotId: null, paramValues: { lift: -1.2 } });
    expect(expanded.ok).toBe(true);
    if (!expanded.ok) return;
    const first = expanded.steps[0];
    expect(first?.kind === 'moveJoints' && first.targets['joint2']).toBe(-1.2);
  });

  it('문자 대입 — note(공통 필드)에 문자열을 넣는다', () => {
    const block = withParams(capturedBlock(), [
      {
        key: 'memo',
        labelKo: '메모',
        kind: 'string',
        defaultValue: '',
        bindings: [{ stepIndex: 2, path: 'note' }],
      },
    ]);
    const expanded = expandBlock(block, {
      targetRobotId: null,
      paramValues: { memo: '파지 지점' },
    });
    expect(expanded.ok).toBe(true);
    if (!expanded.ok) return;
    expect(expanded.steps[2]?.note).toBe('파지 지점');
  });

  it('불리언 대입 — enabled(공통 필드)를 끈다', () => {
    const block = withParams(capturedBlock(), [
      {
        key: 'doGrip',
        labelKo: '그리퍼 실행',
        kind: 'boolean',
        defaultValue: true,
        bindings: [{ stepIndex: 2, path: 'enabled' }],
      },
    ]);
    const expanded = expandBlock(block, { targetRobotId: null, paramValues: { doGrip: false } });
    expect(expanded.ok).toBe(true);
    if (!expanded.ok) return;
    expect(expanded.steps[2]?.enabled).toBe(false);
  });

  it("'robot' 경로 binding은 targetRobotId 재매핑보다 우선한다", () => {
    const block = withParams(capturedBlock(), [
      {
        key: 'who',
        labelKo: '대상 로봇',
        kind: 'string',
        defaultValue: 'arm_a',
        bindings: [{ stepIndex: 0, path: 'robot' }],
      },
    ]);
    const expanded = expandBlock(block, {
      targetRobotId: 'arm_new',
      paramValues: { who: 'arm_special' },
    });
    expect(expanded.ok).toBe(true);
    if (!expanded.ok) return;
    const first = expanded.steps[0];
    expect(first?.kind === 'moveJoints' && first.robot).toBe('arm_special');
  });
});

// ── 거부 경로 ───────────────────────────────────────────────────────

describe('expandBlock — 잘못된 경로/타입 거부', () => {
  const numberParam = (bindings: BlockParam['bindings']): BlockParam => ({
    key: 'v',
    labelKo: '값',
    kind: 'number',
    defaultValue: 1,
    bindings,
  });

  it('타입 불일치를 거부한다 (number 파라미터에 문자열)', () => {
    const block = withParams(capturedBlock(), [
      numberParam([{ stepIndex: 1, path: 'durationSec' }]),
    ]);
    const result = expandBlock(block, { targetRobotId: null, paramValues: { v: 'abc' } });
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.errors.some((e) => e.includes("'v'") && e.includes('number'))).toBe(true);
    }
  });

  it('NaN/Infinity를 거부한다', () => {
    const block = withParams(capturedBlock(), [
      numberParam([{ stepIndex: 1, path: 'durationSec' }]),
    ]);
    expect(expandBlock(block, { targetRobotId: null, paramValues: { v: NaN } }).ok).toBe(false);
    expect(
      expandBlock(block, { targetRobotId: null, paramValues: { v: Infinity } }).ok,
    ).toBe(false);
  });

  it('min/max 범위 밖 값을 거부한다', () => {
    const block = withParams(capturedBlock(), [
      { ...numberParam([{ stepIndex: 1, path: 'durationSec' }]), min: 0.5, max: 5 },
    ]);
    expect(expandBlock(block, { targetRobotId: null, paramValues: { v: 0.1 } }).ok).toBe(false);
    expect(expandBlock(block, { targetRobotId: null, paramValues: { v: 6 } }).ok).toBe(false);
  });

  it('해당 step 종류에 없는 필드 경로를 거부한다 (무음 no-op 금지)', () => {
    const block = withParams(capturedBlock(), [numberParam([{ stepIndex: 1, path: 'timeout' }])]);
    const result = expandBlock(block, { targetRobotId: null, paramValues: {} });
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.errors.some((e) => e.includes("'timeout'"))).toBe(true);
  });

  it('kind 치환 경로를 거부한다', () => {
    const block = withParams(capturedBlock(), [
      {
        key: 'k',
        labelKo: '종류',
        kind: 'string',
        defaultValue: 'wait',
        bindings: [{ stepIndex: 0, path: 'kind' }],
      },
    ]);
    const result = expandBlock(block, { targetRobotId: null, paramValues: {} });
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.errors.some((e) => e.includes('kind'))).toBe(true);
  });

  it('최대 깊이(2단계) 초과 경로를 거부한다', () => {
    const block = withParams(capturedBlock(), [
      numberParam([{ stepIndex: 0, path: 'a.b.c' }]),
    ]);
    const result = expandBlock(block, { targetRobotId: null, paramValues: {} });
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.errors.some((e) => e.includes(`${BINDING_PATH_MAX_SEGMENTS}단계`))).toBe(true);
    }
  });

  it('stepIndex 범위 밖을 거부한다', () => {
    const block = withParams(capturedBlock(), [
      numberParam([{ stepIndex: 99, path: 'durationSec' }]),
    ]);
    const result = expandBlock(block, { targetRobotId: null, paramValues: {} });
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.errors.some((e) => e.includes('stepIndex 99'))).toBe(true);
  });

  it('블록에 정의되지 않은 잉여 파라미터 key를 거부한다', () => {
    const block = capturedBlock();
    const result = expandBlock(block, { targetRobotId: null, paramValues: { ghost: 1 } });
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.errors.some((e) => e.includes("'ghost'"))).toBe(true);
  });

  it('대입 결과가 스키마 위반이면 재검증이 거부한다 (§2.8 — 검증 없이 실행 노출 금지)', () => {
    // min/max 없는 파라미터로 음수 duration을 밀어 넣는다 → controlStepSchema가 잡는다
    const block = withParams(capturedBlock(), [
      numberParam([{ stepIndex: 1, path: 'durationSec' }]),
    ]);
    const result = expandBlock(block, { targetRobotId: null, paramValues: { v: -1 } });
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.errors.some((e) => e.includes('steps[1]'))).toBe(true);
    }
  });

  it('빈 블록(steps 없음)을 거부한다', () => {
    const block = { ...capturedBlock(), steps: [] as ControlStep[] };
    const result = expandBlock(block, { targetRobotId: null, paramValues: {} });
    expect(result.ok).toBe(false);
  });

  it('오류가 하나라도 있으면 부분 결과 없이 전체가 거부된다', () => {
    const block = withParams(capturedBlock(), [
      numberParam([
        { stepIndex: 0, path: 'params.durationSec' }, // 유효
        { stepIndex: 99, path: 'durationSec' }, // 무효
      ]),
    ]);
    const result = expandBlock(block, { targetRobotId: null, paramValues: { v: 3 } });
    expect(result.ok).toBe(false);
  });
});

// ── 순수 헬퍼 ───────────────────────────────────────────────────────

describe('parseBindingPath', () => {
  it("'params.' 별칭을 벗기고 head/tail로 나눈다", () => {
    expect(parseBindingPath('params.durationSec')).toEqual({
      ok: true,
      head: 'durationSec',
      tail: null,
    });
    expect(parseBindingPath('targets.joint1')).toEqual({
      ok: true,
      head: 'targets',
      tail: 'joint1',
    });
    expect(parseBindingPath('params.targets.joint1')).toEqual({
      ok: true,
      head: 'targets',
      tail: 'joint1',
    });
  });

  it('빈 구획·빈 경로·params 단독을 거부한다', () => {
    expect(parseBindingPath('').ok).toBe(false);
    expect(parseBindingPath('a..b').ok).toBe(false);
    expect(parseBindingPath('params').ok).toBe(false);
    expect(parseBindingPath('.durationSec').ok).toBe(false);
  });
});

describe('checkParamValue', () => {
  const base: BlockParam = { key: 'p', labelKo: '값', kind: 'number', defaultValue: 1, bindings: [] };

  it('종류별 타입을 강제한다', () => {
    expect(checkParamValue(base, 2).ok).toBe(true);
    expect(checkParamValue(base, '2').ok).toBe(false);
    expect(checkParamValue({ ...base, kind: 'string', defaultValue: '' }, 'x').ok).toBe(true);
    expect(checkParamValue({ ...base, kind: 'string', defaultValue: '' }, 3).ok).toBe(false);
    expect(checkParamValue({ ...base, kind: 'boolean', defaultValue: true }, false).ok).toBe(true);
    expect(checkParamValue({ ...base, kind: 'boolean', defaultValue: true }, 0).ok).toBe(false);
  });
});

describe('collectRobotIds / blockRefLabel', () => {
  it('등장 순서대로 중복 없이 수집한다', () => {
    const steps: ControlStep[] = [
      { kind: 'gripper', robot: 'b', state: 'open' },
      { kind: 'setJoints', robot: 'a', targets: {} },
      { kind: 'gripper', robot: 'b', state: 'close' },
    ];
    expect(collectRobotIds(steps)).toEqual(['b', 'a']);
  });

  it('blockRefLabel은 이름과 id를 담는다 (전개 흔적 표시용)', () => {
    const label = blockRefLabel({ id: BLOCK_ID, name: '집기 동작' });
    expect(label).toContain('집기 동작');
    expect(label).toContain(BLOCK_ID);
  });
});
