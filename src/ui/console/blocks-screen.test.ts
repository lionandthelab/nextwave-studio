// ui/console/blocks-screen.test.ts — 블록 화면 순수 로직 단위 테스트 (DOM 비의존, node)
//
// mountBlocksScreen의 DOM 조립·모달 배선은 브라우저 게이트 몫이다 (primitives.test.ts와
// 같은 관례). 여기서는 화면의 순수 계약만 검증한다:
// - 카드 모델 매핑 (상세 봉투 → 모델, get 실패 시 강등 모델)
// - 카드 보조행/검색 텍스트 조합
// - 연결 상태 → 저장/삭제 불가 사유 (회색 버튼의 title 계약) · 배지 상태
// - 미리보기 체인: flow-graph kindMeta의 색·라벨 재사용 (색 발명 금지)

import { describe, expect, it } from 'vitest';
import {
  LOCAL_MODE_BLOCKS_HINT_KO,
  blockCardModel,
  blockSearchText,
  blockSubline,
  connectionBadgeStatus,
  degradedCardModel,
  previewChain,
  writeDisabledReasonKo,
} from './blocks-screen';
import { kindMeta } from '../flow-graph/node-render';
import { CATEGORY } from '../theme';
import type { BlockDoc, RecordEnvelope } from '../../schema/entities';

// ── 픽스처 ──────────────────────────────────────────────────────────

function sampleRecord(overrides: Partial<BlockDoc> = {}): RecordEnvelope<BlockDoc> {
  const doc: BlockDoc = {
    id: 'block-pick-0001',
    name: '집기 동작',
    descriptionKo: '접근 → 파지 → 들어올림',
    steps: [
      { kind: 'moveJoints', targets: { joint1: 0.5 }, durationSec: 2 },
      { kind: 'gripper', state: 'close' },
      { kind: 'wait', durationSec: 1 },
    ],
    params: [
      {
        key: 'moveSec',
        labelKo: '이동 시간',
        kind: 'number',
        defaultValue: 2,
        bindings: [{ stepIndex: 0, path: 'params.durationSec' }],
      },
    ],
    robotHint: 'Arm-6',
    ...overrides,
  };
  return {
    doc,
    meta: {
      version: 3,
      createdAtIso: '2026-08-01T00:00:00Z',
      createdBy: 'user-000001',
      createdByName: '김설치',
      updatedAtIso: '2026-08-05T00:00:00Z',
      updatedBy: 'user-000001',
      updatedByName: '김설치',
      deletedAtIso: null,
      deletedByName: null,
    },
  };
}

// ── 카드 모델 ───────────────────────────────────────────────────────

describe('blockCardModel', () => {
  it('상세 봉투에서 이름·설명·step 수·robotHint·파라미터 수·kind 체인을 뽑는다', () => {
    const model = blockCardModel(sampleRecord());
    expect(model).toEqual({
      id: 'block-pick-0001',
      name: '집기 동작',
      descriptionKo: '접근 → 파지 → 들어올림',
      stepCount: 3,
      robotHint: 'Arm-6',
      paramCount: 1,
      kinds: ['moveJoints', 'gripper', 'wait'],
    });
  });

  it('get 실패 시 강등 모델 — stepCount null이 "상세 없음" 표시의 신호다', () => {
    const model = degradedCardModel('block-x-000001', '이름만 아는 블록');
    expect(model.stepCount).toBe(null);
    expect(model.kinds).toEqual([]);
    expect(model.paramCount).toBe(0);
  });
});

describe('blockSubline / blockSearchText', () => {
  it('step 수 + robotHint를 " · "로 잇는다 (도메인 식별자 — lang="en" 표기 대상)', () => {
    expect(blockSubline(3, 'Arm-6')).toBe('3 steps · Arm-6');
    expect(blockSubline(3, null)).toBe('3 steps');
    expect(blockSubline(3, '')).toBe('3 steps');
    expect(blockSubline(null, 'Arm-6')).toBe('Arm-6');
    expect(blockSubline(null, null)).toBe('');
  });

  it('검색 텍스트는 이름·설명·robotHint를 모두 포함한다', () => {
    const text = blockSearchText(blockCardModel(sampleRecord()));
    expect(text).toContain('집기 동작');
    expect(text).toContain('파지');
    expect(text).toContain('Arm-6');
  });
});

// ── 연결 상태 → 사유/배지 ───────────────────────────────────────────

describe('writeDisabledReasonKo', () => {
  it('온라인이면 null (저장/삭제 가능)', () => {
    expect(writeDisabledReasonKo({ mode: 'server', online: true })).toBe(null);
  });

  it('로컬 모드·오프라인에는 각각 사람이 읽을 사유가 있다 (이유 없는 회색 버튼 금지)', () => {
    const local = writeDisabledReasonKo({ mode: 'local', online: false });
    expect(local).toContain('로컬 모드');
    const offline = writeDisabledReasonKo({ mode: 'server', online: false });
    expect(offline).toContain('오프라인');
    expect(local).not.toBe(offline); // 사유가 상태를 구분해 준다
  });

  it('로컬 모드 빈 상태 힌트는 다음 행동(서버 연결)을 안내한다', () => {
    expect(LOCAL_MODE_BLOCKS_HINT_KO).toContain('서버');
  });
});

describe('connectionBadgeStatus', () => {
  it('server+online만 success, 그 외는 warn (색+텍스트 병행 전제)', () => {
    expect(connectionBadgeStatus({ mode: 'server', online: true })).toBe('success');
    expect(connectionBadgeStatus({ mode: 'server', online: false })).toBe('warn');
    expect(connectionBadgeStatus({ mode: 'local', online: false })).toBe('warn');
  });
});

// ── 미리보기 체인 (kindMeta 재사용 — 색 발명 금지) ──────────────────

describe('previewChain', () => {
  it('kindMeta의 라벨과 CATEGORY 색을 그대로 쓴다', () => {
    const chain = previewChain(['moveJoints', 'wait', 'goto']);
    expect(chain).toEqual([
      { kind: 'moveJoints', label: 'MoveJoints', color: CATEGORY.motion },
      { kind: 'wait', label: 'Wait', color: CATEGORY.time },
      { kind: 'goto', label: 'Goto', color: CATEGORY.flow },
    ]);
    // node-render의 kindMeta와 항목별 동일 (독자 매핑을 두지 않는다)
    for (const item of chain) {
      const meta = kindMeta(item.kind);
      expect(item.label).toBe(meta.label);
      expect(item.color).toBe(meta.color);
    }
  });

  it('알 수 없는 kind도 안전하게 표시한다 (throw 없음 — kindMeta 폴백)', () => {
    const chain = previewChain(['teleport']);
    expect(chain[0]?.label).toBe('teleport');
    expect(chain[0]?.color.length).toBeGreaterThan(0);
  });
});
