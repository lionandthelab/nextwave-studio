// ui/sequence-versions.test.ts — 버전 스택·변경 라벨 도출 단위 테스트 (순수, node 환경)
//
// 관례: history.test.ts / toast.test.ts와 동일 — DOM 없이 순수 계약만 검증한다.
// 이 모듈의 계약:
// - 직전 버전과 내용이 같으면 기록하지 않는다 (no-op 커밋이 이력을 밀어내지 않게)
// - 되돌리기로 과거와 같은 내용이 되어도 새 버전이 쌓인다 (append-only — 되돌리기 취소)
// - 라벨은 이전/이후 비교로 자동 도출된다 (편집 호출부가 라벨을 넘기지 않아도 된다)
// - 상한 초과 시 가장 오래된 것부터 버리고, 버전 번호는 재사용하지 않는다
// - 저장된 시퀀스는 깊은 복사본이다 (외부 변형이 이력을 오염시키지 않는다)

import { describe, expect, it } from 'vitest';
import {
  SEQUENCE_VERSION_CAP_DEFAULT,
  SequenceVersions,
  describeSequenceChange,
  sequencesEqual,
} from './sequence-versions';
import type { ControlSequence, ControlStep } from '../schema/types';

// ── 픽스처 ──────────────────────────────────────────────────────────

function seq(steps: ControlStep[], over: Partial<ControlSequence> = {}): ControlSequence {
  return { id: 'flow-graph', robot: 'arm', steps, ...over };
}

const WAIT = (durationSec: number): ControlStep => ({ kind: 'wait', durationSec });
const GRIP = (state: 'open' | 'close'): ControlStep => ({ kind: 'gripper', robot: 'arm', state });

// ── sequencesEqual ──────────────────────────────────────────────────

describe('sequencesEqual', () => {
  it('내용이 같으면 true (다른 객체여도)', () => {
    expect(sequencesEqual(seq([WAIT(1)]), seq([WAIT(1)]))).toBe(true);
  });

  it('내용이 다르면 false', () => {
    expect(sequencesEqual(seq([WAIT(1)]), seq([WAIT(2)]))).toBe(false);
  });

  it('null 취급 — 둘 다 null만 true', () => {
    expect(sequencesEqual(null, null)).toBe(true);
    expect(sequencesEqual(null, seq([]))).toBe(false);
    expect(sequencesEqual(seq([]), null)).toBe(false);
  });
});

// ── 변경 라벨 도출 ──────────────────────────────────────────────────

describe('describeSequenceChange', () => {
  it('이전이 없으면 생성', () => {
    expect(describeSequenceChange(null, seq([WAIT(1)]))).toBe('시퀀스 생성');
  });

  it('노드 수 증감 — 개수까지 말한다', () => {
    expect(describeSequenceChange(seq([WAIT(1)]), seq([WAIT(1), WAIT(2)]))).toBe('노드 1개 추가');
    expect(describeSequenceChange(seq([WAIT(1), WAIT(2), WAIT(3)]), seq([WAIT(1)]))).toBe(
      '노드 2개 삭제',
    );
  });

  it('같은 kind 묶음의 순서만 다르면 재정렬', () => {
    const before = seq([WAIT(1), GRIP('open')]);
    const after = seq([GRIP('open'), WAIT(1)]);
    expect(describeSequenceChange(before, after)).toBe('노드 재정렬');
  });

  it('길이는 같지만 구성이 다르면 교체', () => {
    expect(describeSequenceChange(seq([WAIT(1)]), seq([GRIP('open')]))).toBe('노드 교체');
  });

  it('활성 상태 변경은 파라미터 변경보다 우선한다', () => {
    const before = seq([WAIT(1)]);
    const after = seq([{ ...WAIT(1), enabled: false }]);
    expect(describeSequenceChange(before, after)).toBe('노드 활성 상태 변경');
  });

  it('대상 로봇 · 반복 · 노트 변경을 각각 구분한다', () => {
    expect(describeSequenceChange(seq([WAIT(1)]), seq([WAIT(1)], { robot: 'arm_b' }))).toBe(
      '대상 로봇 변경',
    );
    expect(describeSequenceChange(seq([WAIT(1)]), seq([WAIT(1)], { loop: true }))).toBe(
      '반복 설정 변경',
    );
    expect(
      describeSequenceChange(seq([WAIT(1)]), seq([{ ...WAIT(1), note: '검사 대기' }])),
    ).toBe('노트 변경');
  });

  it('그 외 내용 차이는 파라미터 변경', () => {
    expect(describeSequenceChange(seq([WAIT(1)]), seq([WAIT(5)]))).toBe('파라미터 변경');
  });
});

// ── 스택 ────────────────────────────────────────────────────────────

describe('SequenceVersions', () => {
  const clock = (): string => '2026-08-10T09:00:00.000Z';

  it('첫 기록은 v1이고 라벨이 자동으로 붙는다', () => {
    const versions = new SequenceVersions({ nowIso: clock });
    const v = versions.record(seq([WAIT(1)]));
    expect(v).toMatchObject({ version: 1, labelKo: '시퀀스 생성', stepCount: 1 });
    expect(versions.currentVersion()).toBe(1);
  });

  it('직전과 내용이 같으면 기록하지 않는다 (no-op 커밋 억제)', () => {
    const versions = new SequenceVersions({ nowIso: clock });
    versions.record(seq([WAIT(1)]));
    expect(versions.record(seq([WAIT(1)]))).toBeNull();
    expect(versions.size()).toBe(1);
  });

  it('라벨을 지정하면 자동 도출을 덮어쓴다 (JSON 직접 편집 등)', () => {
    const versions = new SequenceVersions({ nowIso: clock });
    versions.record(seq([WAIT(1)]));
    const v = versions.record(seq([WAIT(2)]), { labelKo: 'JSON 직접 편집' });
    expect(v?.labelKo).toBe('JSON 직접 편집');
  });

  it('되돌리기로 과거와 같은 내용이 되어도 새 버전이 쌓인다 (되돌리기 취소 가능)', () => {
    const versions = new SequenceVersions({ nowIso: clock });
    versions.record(seq([WAIT(1)])); // v1
    versions.record(seq([WAIT(2)])); // v2
    const restored = versions.record(seq([WAIT(1)]), { labelKo: 'v1으로 되돌림' }); // v3
    expect(restored?.version).toBe(3);
    expect(versions.size()).toBe(3);
    // v3의 내용은 v1과 같다 — 그래서 v2로 다시 갈 수도, v3를 되돌릴 수도 있다
    expect(sequencesEqual(versions.get(3)?.sequence ?? null, versions.get(1)?.sequence ?? null)).toBe(
      true,
    );
  });

  it('list는 최신 우선이다', () => {
    const versions = new SequenceVersions({ nowIso: clock });
    versions.record(seq([WAIT(1)]));
    versions.record(seq([WAIT(2)]));
    expect(versions.list().map((v) => v.version)).toEqual([2, 1]);
  });

  it('상한 초과 — 가장 오래된 것부터 버리고 버전 번호는 재사용하지 않는다', () => {
    const versions = new SequenceVersions({ cap: 3, nowIso: clock });
    for (let i = 1; i <= 5; i += 1) versions.record(seq([WAIT(i)]));
    expect(versions.size()).toBe(3);
    expect(versions.list().map((v) => v.version)).toEqual([5, 4, 3]);
    expect(versions.get(1)).toBeNull(); // 폐기됨 — 그 이전으로는 되돌릴 수 없다
    expect(versions.get(5)?.stepCount).toBe(1);
  });

  it('저장본은 깊은 복사다 — 원본을 나중에 변형해도 이력이 오염되지 않는다', () => {
    const versions = new SequenceVersions({ nowIso: clock });
    const live = seq([WAIT(1)]);
    versions.record(live);
    live.steps.push(WAIT(9));
    live.robot = '변조됨';
    expect(versions.get(1)?.sequence.steps).toHaveLength(1);
    expect(versions.get(1)?.sequence.robot).toBe('arm');
  });

  it('clear는 이력과 버전 번호를 함께 되돌린다 (씬 전환)', () => {
    const versions = new SequenceVersions({ nowIso: clock });
    versions.record(seq([WAIT(1)]));
    versions.record(seq([WAIT(2)]));
    versions.clear();
    expect(versions.size()).toBe(0);
    expect(versions.currentVersion()).toBeNull();
    expect(versions.record(seq([WAIT(3)]))?.version).toBe(1);
  });

  it('기본 상한은 50이다', () => {
    expect(SEQUENCE_VERSION_CAP_DEFAULT).toBe(50);
  });
});
