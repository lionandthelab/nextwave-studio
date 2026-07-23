// ui/inspector/node-editor.test.ts — 노드 파라미터 폼 순수 헬퍼 단위 테스트 (DOM 비의존, node)
//
// mountNodeEditor의 DOM 조립은 얇은 글루라 여기서 다루지 않는다(브라우저 게이트 몫).
// 여기서는 params ↔ 폼 상태 매핑 계약을 검증한다:
// - 관절 행 모델(targets ↔ rows, limits 부착/클램프, ±π 폴백, 미사용 관절 후보)
// - gripper 상태(open/close/0..1) 매핑과 durationSec 생략 규약
// - wait/waitForCollision/label/goto의 정규화(음수·비유한 방어, 빈값=무한/무제한)
// - form→params 빌더의 원본 키 보존(무손실 왕복 — UX_DESIGN §6) + 관리 키 제거 규약

import { describe, expect, it } from 'vitest';
import {
  GRIPPER_CLOSE_RATIO,
  GRIPPER_OPEN_RATIO,
  JOINT_RANGE_FALLBACK_RAD,
  applyCollisionFormToParams,
  applyGotoFormToParams,
  applyGripperFormToParams,
  applyJointsFormToParams,
  applyLabelFormToParams,
  applyWaitFormToParams,
  availableJointNames,
  clamp01,
  clampDurationSec,
  clampJointValueRad,
  collisionFormStateOf,
  easingOf,
  gotoFormStateOf,
  gripperFormStateOf,
  jointLimitsByName,
  jointSliderRange,
  jointsFormStateOf,
  labelNameOf,
  nodeKindMeta,
  normalizeGotoTimes,
  normalizeTimeoutSec,
  robotDisplayOf,
  targetsFromJointRows,
  toFiniteNumber,
  validJointLimits,
  waitDurationOf,
} from './node-editor';

// ── 테스트 픽스처 ───────────────────────────────────────────────────

const ROBOT_JOINTS: Array<{ name: string; limits?: [number, number] }> = [
  { name: 'j1', limits: [-1.5, 1.5] },
  { name: 'j2', limits: [0, 2] },
  { name: 'j3' }, // limits 없음 → ±π 폴백
];

// ── 공통 파싱/정규화 ────────────────────────────────────────────────

describe('toFiniteNumber', () => {
  it('유한 수는 그대로, 그 외(NaN/∞/문자열/null)는 null', () => {
    expect(toFiniteNumber(1.25)).toBe(1.25);
    expect(toFiniteNumber(0)).toBe(0);
    expect(toFiniteNumber(Number.NaN)).toBeNull();
    expect(toFiniteNumber(Number.POSITIVE_INFINITY)).toBeNull();
    expect(toFiniteNumber('1')).toBeNull();
    expect(toFiniteNumber(null)).toBeNull();
    expect(toFiniteNumber(undefined)).toBeNull();
  });
});

describe('clampDurationSec / clamp01', () => {
  it('durationSec: null/비유한/음수 → 0, 양수는 그대로', () => {
    expect(clampDurationSec(null)).toBe(0);
    expect(clampDurationSec(Number.NaN)).toBe(0);
    expect(clampDurationSec(-1)).toBe(0);
    expect(clampDurationSec(2.5)).toBe(2.5);
    expect(clampDurationSec(0)).toBe(0);
  });

  it('clamp01: 0..1 범위로 클램프, 비유한은 0(닫힘)', () => {
    expect(clamp01(-0.5)).toBe(0);
    expect(clamp01(1.5)).toBe(1);
    expect(clamp01(0.3)).toBe(0.3);
    expect(clamp01(Number.NaN)).toBe(0);
  });
});

describe('easingOf', () => {
  it('허용 easing 3종만 통과, 그 외는 undefined', () => {
    expect(easingOf('linear')).toBe('linear');
    expect(easingOf('easeInOut')).toBe('easeInOut');
    expect(easingOf('step')).toBe('step');
    expect(easingOf('')).toBeUndefined();
    expect(easingOf('bounce')).toBeUndefined();
    expect(easingOf(3)).toBeUndefined();
  });
});

describe('robotDisplayOf', () => {
  it('params.robot(비어 있지 않은 문자열) 우선, 없으면 기본 로봇', () => {
    expect(robotDisplayOf({ robot: 'arm2' }, 'arm')).toBe('arm2');
    expect(robotDisplayOf({}, 'arm')).toBe('arm');
    expect(robotDisplayOf({ robot: '' }, 'arm')).toBe('arm');
    expect(robotDisplayOf({ robot: 42 }, 'arm')).toBe('arm');
  });
});

describe('nodeKindMeta', () => {
  it('알려진 8종 step kind에 아이콘+라벨을 준다', () => {
    const kinds = [
      'moveJoints',
      'setJoints',
      'gripper',
      'wait',
      'waitForCollision',
      'label',
      'goto',
      'moveToPose',
    ];
    for (const kind of kinds) {
      const meta = nodeKindMeta(kind);
      expect(meta.icon.length).toBeGreaterThan(0);
      expect(meta.label.toLowerCase()).toContain(kind.toLowerCase());
    }
  });

  it('미지 kind는 원문 라벨 + 폴백 아이콘', () => {
    expect(nodeKindMeta('customStep')).toEqual({ icon: '◌', label: 'customStep' });
  });
});

// ── 관절 행 모델 ────────────────────────────────────────────────────

describe('validJointLimits / jointSliderRange', () => {
  it('유효 limits([lo, hi], lo ≤ hi 유한)만 통과한다', () => {
    expect(validJointLimits([-1, 1])).toEqual([-1, 1]);
    expect(validJointLimits([0, 0])).toEqual([0, 0]);
    expect(validJointLimits([1, -1])).toBeUndefined(); // 역전
    expect(validJointLimits([Number.NaN, 1])).toBeUndefined();
    expect(validJointLimits([0, Number.POSITIVE_INFINITY])).toBeUndefined();
    expect(validJointLimits([0])).toBeUndefined();
    expect(validJointLimits(undefined)).toBeUndefined();
  });

  it(`limits 없으면 ±π(${JOINT_RANGE_FALLBACK_RAD.toFixed(3)}) 폴백 (요구 §2)`, () => {
    expect(jointSliderRange([-1.5, 1.5])).toEqual({ minRad: -1.5, maxRad: 1.5 });
    expect(jointSliderRange(undefined)).toEqual({
      minRad: -JOINT_RANGE_FALLBACK_RAD,
      maxRad: JOINT_RANGE_FALLBACK_RAD,
    });
  });

  it('역전 limits는 무효 → ±π 폴백', () => {
    expect(jointSliderRange([2, -2])).toEqual({
      minRad: -JOINT_RANGE_FALLBACK_RAD,
      maxRad: JOINT_RANGE_FALLBACK_RAD,
    });
  });
});

describe('clampJointValueRad', () => {
  it('limits 범위로 클램프한다', () => {
    expect(clampJointValueRad(2, [-1.5, 1.5])).toBe(1.5);
    expect(clampJointValueRad(-2, [-1.5, 1.5])).toBe(-1.5);
    expect(clampJointValueRad(0.5, [-1.5, 1.5])).toBe(0.5);
  });

  it('limits 없으면 ±π로 클램프', () => {
    expect(clampJointValueRad(10)).toBe(JOINT_RANGE_FALLBACK_RAD);
    expect(clampJointValueRad(-10)).toBe(-JOINT_RANGE_FALLBACK_RAD);
  });

  it('비유한 입력은 0을 기준으로 클램프한다 (0이 범위 밖이면 경계값)', () => {
    expect(clampJointValueRad(Number.NaN, [-1, 1])).toBe(0);
    expect(clampJointValueRad(Number.NaN, [0.5, 1])).toBe(0.5);
  });
});

describe('jointLimitsByName', () => {
  it('유효 limits만 맵에 담고, 없거나 무효인 관절은 제외한다', () => {
    const map = jointLimitsByName([
      { name: 'a', limits: [-1, 1] },
      { name: 'b' },
      { name: 'c', limits: [3, -3] }, // 역전 → 무효
    ]);
    expect(map.get('a')).toEqual([-1, 1]);
    expect(map.has('b')).toBe(false);
    expect(map.has('c')).toBe(false);
  });
});

describe('jointsFormStateOf', () => {
  it('targets를 삽입 순서대로 행으로 만들고, robotJoints의 limits를 부착한다', () => {
    const form = jointsFormStateOf(
      { targets: { j2: 1.0, j1: -0.5, j3: 3.0 }, durationSec: 2, easing: 'easeInOut' },
      ROBOT_JOINTS,
    );
    expect(form.rows).toEqual([
      { name: 'j2', valueRad: 1.0, limits: [0, 2] },
      { name: 'j1', valueRad: -0.5, limits: [-1.5, 1.5] },
      { name: 'j3', valueRad: 3.0 }, // limits 없음 (±π 폴백은 슬라이더 몫)
    ]);
    expect(form.durationSec).toBe(2);
    expect(form.easing).toBe('easeInOut');
  });

  it('숫자가 아닌 목표값은 건너뛴다 (방어적 파싱)', () => {
    const form = jointsFormStateOf({ targets: { j1: 'x', j2: 0.5, j3: Number.NaN } }, ROBOT_JOINTS);
    expect(form.rows).toEqual([{ name: 'j2', valueRad: 0.5, limits: [0, 2] }]);
  });

  it('targets 누락/무효, durationSec/easing 무효는 생략된다', () => {
    expect(jointsFormStateOf({}, ROBOT_JOINTS).rows).toEqual([]);
    expect(jointsFormStateOf({ targets: [1, 2] }, ROBOT_JOINTS).rows).toEqual([]);
    const form = jointsFormStateOf({ targets: {}, durationSec: 'x', easing: 'zig' }, ROBOT_JOINTS);
    expect(form.durationSec).toBeUndefined();
    expect(form.easing).toBeUndefined();
  });
});

describe('availableJointNames', () => {
  it('이미 행으로 쓰인 관절을 제외한 후보를 준다 (추가 드롭다운 — 요구 §2)', () => {
    const rows = [{ name: 'j1', valueRad: 0 }];
    expect(availableJointNames(ROBOT_JOINTS, rows)).toEqual(['j2', 'j3']);
    expect(availableJointNames(ROBOT_JOINTS, [])).toEqual(['j1', 'j2', 'j3']);
    expect(
      availableJointNames(ROBOT_JOINTS, [
        { name: 'j1', valueRad: 0 },
        { name: 'j2', valueRad: 0 },
        { name: 'j3', valueRad: 0 },
      ]),
    ).toEqual([]);
  });
});

describe('targetsFromJointRows', () => {
  it('행 목록 → targets 레코드, 값은 행별 limits로 클램프', () => {
    expect(
      targetsFromJointRows([
        { name: 'j1', valueRad: 9, limits: [-1.5, 1.5] },
        { name: 'j3', valueRad: -0.25 },
      ]),
    ).toEqual({ j1: 1.5, j3: -0.25 });
  });
});

describe('applyJointsFormToParams', () => {
  it('moveJoints: targets/durationSec/easing을 관리하고 나머지 키는 보존한다', () => {
    const original = { robot: 'arm', targets: { j1: 0 }, durationSec: 1, extra: 'x' };
    const next = applyJointsFormToParams(
      original,
      { rows: [{ name: 'j2', valueRad: 0.5, limits: [0, 2] }], durationSec: 3, easing: 'step' },
      'moveJoints',
    );
    expect(next).toEqual({
      robot: 'arm',
      targets: { j2: 0.5 },
      durationSec: 3,
      easing: 'step',
      extra: 'x',
    });
    // 원본 불변 (커밋 실패 시 마지막 유효 값 복원의 전제)
    expect(original.targets).toEqual({ j1: 0 });
  });

  it('moveJoints: easing undefined면 키를 제거한다 (기본값 위임)', () => {
    const next = applyJointsFormToParams(
      { targets: {}, durationSec: 1, easing: 'linear' },
      { rows: [], durationSec: 1 },
      'moveJoints',
    );
    expect('easing' in next).toBe(false);
  });

  it('moveJoints: durationSec 음수/누락은 0으로 방어한다 (스키마 nonnegative 정합)', () => {
    expect(
      applyJointsFormToParams({}, { rows: [], durationSec: -2 }, 'moveJoints')['durationSec'],
    ).toBe(0);
    expect(applyJointsFormToParams({}, { rows: [] }, 'moveJoints')['durationSec']).toBe(0);
  });

  it('setJoints: targets만 관리하고 durationSec를 추가하지 않는다', () => {
    const next = applyJointsFormToParams(
      { robot: 'arm' },
      { rows: [{ name: 'j1', valueRad: 1, limits: [-1.5, 1.5] }] },
      'setJoints',
    );
    expect(next).toEqual({ robot: 'arm', targets: { j1: 1 } });
    expect('durationSec' in next).toBe(false);
  });
});

// ── gripper ─────────────────────────────────────────────────────────

describe('gripperFormStateOf', () => {
  it("state 'open'/'close' → 모드 + 극값 비율 (0=닫힘, 1=열림)", () => {
    expect(gripperFormStateOf({ state: 'open' })).toEqual({
      mode: 'open',
      valueRatio: GRIPPER_OPEN_RATIO,
    });
    expect(gripperFormStateOf({ state: 'close' })).toEqual({
      mode: 'close',
      valueRatio: GRIPPER_CLOSE_RATIO,
    });
  });

  it('숫자 state → 값 모드 (0..1 클램프)', () => {
    expect(gripperFormStateOf({ state: 0.3 })).toEqual({ mode: 'value', valueRatio: 0.3 });
    expect(gripperFormStateOf({ state: 1.5 })).toEqual({ mode: 'value', valueRatio: 1 });
    expect(gripperFormStateOf({ state: -1 })).toEqual({ mode: 'value', valueRatio: 0 });
  });

  it('미지/누락 state는 open으로 폴백하고, durationSec은 클램프해 담는다', () => {
    expect(gripperFormStateOf({})).toEqual({ mode: 'open', valueRatio: GRIPPER_OPEN_RATIO });
    expect(gripperFormStateOf({ state: 'open', durationSec: 0.8 }).durationSec).toBe(0.8);
    expect(gripperFormStateOf({ state: 'open', durationSec: -1 }).durationSec).toBe(0);
    expect(gripperFormStateOf({ state: 'open', durationSec: 'x' }).durationSec).toBeUndefined();
  });
});

describe('applyGripperFormToParams', () => {
  it('값 모드 → 숫자 state(클램프), open/close 모드 → 문자열 state', () => {
    expect(applyGripperFormToParams({}, { mode: 'value', valueRatio: 1.2 })['state']).toBe(1);
    expect(applyGripperFormToParams({}, { mode: 'open', valueRatio: 0 })['state']).toBe('open');
    expect(applyGripperFormToParams({}, { mode: 'close', valueRatio: 1 })['state']).toBe('close');
  });

  it('durationSec undefined면 키를 제거하고(즉시 동작 위임), robot 등은 보존한다', () => {
    const next = applyGripperFormToParams(
      { robot: 'arm', state: 'open', durationSec: 2 },
      { mode: 'close', valueRatio: 0 },
    );
    expect(next).toEqual({ robot: 'arm', state: 'close' });
    expect('durationSec' in next).toBe(false);
    const withDuration = applyGripperFormToParams(
      { state: 'open' },
      { mode: 'open', valueRatio: 1, durationSec: -3 },
    );
    expect(withDuration['durationSec']).toBe(0); // 음수 방어
  });
});

// ── wait / waitForCollision ─────────────────────────────────────────

describe('waitDurationOf / applyWaitFormToParams', () => {
  it('durationSec 누락/무효/음수는 0으로 표시·커밋한다', () => {
    expect(waitDurationOf({ durationSec: 1.5 })).toBe(1.5);
    expect(waitDurationOf({})).toBe(0);
    expect(waitDurationOf({ durationSec: -2 })).toBe(0);
    expect(applyWaitFormToParams({ note: 'k' }, 2)).toEqual({ note: 'k', durationSec: 2 });
    expect(applyWaitFormToParams({}, null)).toEqual({ durationSec: 0 });
  });
});

describe('normalizeTimeoutSec / collisionFormStateOf', () => {
  it('timeoutSec은 양수만 유효 — 빈값/0/음수/비유한은 무제한(undefined)', () => {
    expect(normalizeTimeoutSec(5)).toBe(5);
    expect(normalizeTimeoutSec(0)).toBeUndefined();
    expect(normalizeTimeoutSec(-1)).toBeUndefined();
    expect(normalizeTimeoutSec(null)).toBeUndefined();
    expect(normalizeTimeoutSec(Number.NaN)).toBeUndefined();
  });

  it('between 배열에서 엔티티 쌍을 읽고, 무효 항목은 빈 문자열로 방어한다', () => {
    expect(collisionFormStateOf({ between: ['arm', 'box_a'], timeoutSec: 3 })).toEqual({
      a: 'arm',
      b: 'box_a',
      timeoutSec: 3,
    });
    expect(collisionFormStateOf({ between: ['arm'] })).toEqual({ a: 'arm', b: '' });
    expect(collisionFormStateOf({})).toEqual({ a: '', b: '' });
    expect(collisionFormStateOf({ between: [1, 2] })).toEqual({ a: '', b: '' });
  });
});

describe('applyCollisionFormToParams', () => {
  it('between 튜플을 재구성하고 timeoutSec 무효 시 키를 제거한다', () => {
    const next = applyCollisionFormToParams(
      { between: ['x', 'y'], timeoutSec: 9, note: 'n' },
      { a: 'arm', b: 'box_a' },
    );
    expect(next).toEqual({ between: ['arm', 'box_a'], note: 'n' });
    expect('timeoutSec' in next).toBe(false);
    expect(
      applyCollisionFormToParams({}, { a: 'arm', b: 'box_a', timeoutSec: 2 })['timeoutSec'],
    ).toBe(2);
  });
});

// ── label / goto ────────────────────────────────────────────────────

describe('labelNameOf / applyLabelFormToParams', () => {
  it('name 문자열만 읽고, 커밋 시 trim만 한다 (중복/빈 이름 검증은 다운스트림)', () => {
    expect(labelNameOf({ name: 'loop' })).toBe('loop');
    expect(labelNameOf({})).toBe('');
    expect(labelNameOf({ name: 3 })).toBe('');
    expect(applyLabelFormToParams({ name: 'old' }, '  loop ')).toEqual({ name: 'loop' });
    // 빈 이름도 그대로 커밋 → commitParams(스키마 검증)가 한국어 오류로 거부한다
    expect(applyLabelFormToParams({}, '   ')).toEqual({ name: '' });
  });
});

describe('normalizeGotoTimes / gotoFormStateOf / applyGotoFormToParams', () => {
  it('times: 빈값/비유한 → 무한(undefined), 소수는 반올림, 음수는 0', () => {
    expect(normalizeGotoTimes(null)).toBeUndefined();
    expect(normalizeGotoTimes(Number.NaN)).toBeUndefined();
    expect(normalizeGotoTimes(2.4)).toBe(2);
    expect(normalizeGotoTimes(2.6)).toBe(3);
    expect(normalizeGotoTimes(-3)).toBe(0);
    expect(normalizeGotoTimes(0)).toBe(0);
  });

  it('폼 상태를 읽고 무효 times는 무한으로 정규화한다', () => {
    expect(gotoFormStateOf({ label: 'loop', times: 3 })).toEqual({ label: 'loop', times: 3 });
    expect(gotoFormStateOf({ label: 'loop' })).toEqual({ label: 'loop' });
    expect(gotoFormStateOf({})).toEqual({ label: '' });
    expect(gotoFormStateOf({ label: 'loop', times: 'x' })).toEqual({ label: 'loop' });
  });

  it('times undefined면 키를 제거한다 (미지정=무한 — DATA_MODEL §6)', () => {
    const infinite = applyGotoFormToParams({ label: 'a', times: 5 }, { label: 'loop' });
    expect(infinite).toEqual({ label: 'loop' });
    expect('times' in infinite).toBe(false);
    expect(applyGotoFormToParams({}, { label: 'loop', times: 2.6 })).toEqual({
      label: 'loop',
      times: 3,
    });
  });
});
