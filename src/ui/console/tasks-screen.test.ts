// ui/console/tasks-screen.test.ts — 작업 목록 화면 순수 로직 단위 테스트 (DOM 비의존, node)
//
// mountTasksScreen의 DOM 조립·배선은 브라우저 게이트 몫이다(primitives.test.ts와 같은
// 관례 — jsdom 금지). 여기서는 임무 명세가 요구한 순수 계약만 검증한다:
// - 밀도 결정 (사용자 선택 우선, 없으면 20건 초과 → 표)
// - 공정 필터 매처 (전체 / 무소속 / 공정 id)
// - 잠금 → "편집 중" 표시 (내 잠금 제외 · TTL 만료 제외 · fail-open)
// - 마지막 실행 결과 → STATUS 배지 매핑 (runResultSchema 4종 전부)
// - 목록 뷰모델 조립(buildTaskRows) · 가시 행(휴지통/필터/검색/수정 내림차순)
// - 복제 이름(80자 상한) · 오프라인 사유 · id 발급 · 상태 라인 문구

import { describe, expect, it } from 'vitest';
import {
  DENSITY_TABLE_THRESHOLD,
  PROCESS_FILTER_ALL,
  PROCESS_FILTER_NONE,
  RUN_RESULT_BADGE,
  TASK_NAME_MAX_LEN,
  activeLockHolder,
  buildTaskRows,
  decideDensity,
  duplicateName,
  makeEntityId,
  matchesProcessFilter,
  taskCountLabelKo,
  taskSearchText,
  unavailableReasonKo,
  updatedAtSortKey,
  visibleTaskRows,
} from './tasks-screen';
import type { TaskRowVM } from './tasks-screen';
import { STATUS } from '../theme';
import type { EntityMeta, LockInfo, RecordMeta, RunResult } from '../../schema/entities';

// ── 픽스처 ──────────────────────────────────────────────────────────

const NOW_MS = Date.parse('2026-08-07T12:00:00Z');

function recordMeta(over: Partial<RecordMeta> = {}): RecordMeta {
  return {
    version: 1,
    createdAtIso: '2026-08-01T00:00:00Z',
    createdBy: 'user-0001',
    createdByName: '김설치',
    updatedAtIso: '2026-08-05T00:00:00Z',
    updatedBy: 'user-0001',
    updatedByName: '김설치',
    deletedAtIso: null,
    deletedByName: null,
    ...over,
  };
}

function entityMeta(over: Partial<EntityMeta> = {}): EntityMeta {
  return {
    id: 'task-0001',
    name: '픽앤플레이스',
    meta: recordMeta(),
    taskSummary: { stepCount: 5, hasThumbnail: false, lastRun: null },
    processId: null,
    ...over,
  };
}

function lockInfo(over: Partial<LockInfo> = {}): LockInfo {
  return {
    entityKind: 'task',
    entityId: 'task-0001',
    userId: 'user-0002',
    userName: '박기사',
    acquiredAtIso: '2026-08-07T11:59:00Z',
    expiresAtIso: '2026-08-07T12:01:30Z', // NOW_MS + 90초
    ...over,
  };
}

function vm(over: Partial<TaskRowVM> = {}): TaskRowVM {
  return {
    id: 'task-0001',
    name: '픽앤플레이스',
    processId: null,
    processName: null,
    stepCount: 3,
    hasThumbnail: false,
    updatedAtIso: '2026-08-05T00:00:00Z',
    updatedByName: '김설치',
    lastRun: null,
    deletedAtIso: null,
    lockHolder: null,
    conflict: false,
    ...over,
  };
}

// ── 밀도 결정 ───────────────────────────────────────────────────────

describe('decideDensity', () => {
  it('사용자가 고른 값(localStorage)이 건수보다 우선한다', () => {
    expect(decideDensity('grid', 100)).toBe('grid');
    expect(decideDensity('list', 1)).toBe('list');
  });

  it('저장값이 없으면 20건 초과 → 표, 이하 → 카드', () => {
    expect(decideDensity(null, DENSITY_TABLE_THRESHOLD + 1)).toBe('list');
    expect(decideDensity(null, DENSITY_TABLE_THRESHOLD)).toBe('grid');
    expect(decideDensity(null, 0)).toBe('grid');
  });

  it('알 수 없는 저장값(손상)은 무시하고 건수 규칙으로 돌아간다', () => {
    expect(decideDensity('weird', 25)).toBe('list');
    expect(decideDensity('', 3)).toBe('grid');
  });

  it('임계값은 20이다 (임무 명세 고정값)', () => {
    expect(DENSITY_TABLE_THRESHOLD).toBe(20);
  });
});

// ── 공정 필터 ───────────────────────────────────────────────────────

describe('matchesProcessFilter', () => {
  it("'all'은 전부 매치", () => {
    expect(matchesProcessFilter({ processId: null }, PROCESS_FILTER_ALL)).toBe(true);
    expect(matchesProcessFilter({ processId: 'proc-0001' }, PROCESS_FILTER_ALL)).toBe(true);
  });

  it("'none'은 무소속(processId=null)만", () => {
    expect(matchesProcessFilter({ processId: null }, PROCESS_FILTER_NONE)).toBe(true);
    expect(matchesProcessFilter({ processId: 'proc-0001' }, PROCESS_FILTER_NONE)).toBe(false);
  });

  it('공정 id는 정확히 그 공정 소속만', () => {
    expect(matchesProcessFilter({ processId: 'proc-0001' }, 'proc-0001')).toBe(true);
    expect(matchesProcessFilter({ processId: 'proc-0002' }, 'proc-0001')).toBe(false);
    expect(matchesProcessFilter({ processId: null }, 'proc-0001')).toBe(false);
  });
});

// ── 잠금 표시 ───────────────────────────────────────────────────────

describe('activeLockHolder', () => {
  it('잠금 없음 → null', () => {
    expect(activeLockHolder(null, 'user-0001', NOW_MS)).toBeNull();
    expect(activeLockHolder(undefined, 'user-0001', NOW_MS)).toBeNull();
  });

  it('다른 사용자의 유효한 잠금 → 그 이름', () => {
    expect(activeLockHolder(lockInfo(), 'user-0001', NOW_MS)).toBe('박기사');
  });

  it('내 잠금(다른 탭)은 "편집 중"으로 표시하지 않는다', () => {
    expect(activeLockHolder(lockInfo({ userId: 'user-0001' }), 'user-0001', NOW_MS)).toBeNull();
  });

  it('미로그인(currentUserId=null)이면 타인 잠금은 그대로 보인다', () => {
    expect(activeLockHolder(lockInfo(), null, NOW_MS)).toBe('박기사');
  });

  it('TTL이 지난 잠금은 무시한다 — "아무도 없는데 잠김"을 만들지 않는다', () => {
    const expired = lockInfo({ expiresAtIso: '2026-08-07T11:59:59Z' });
    expect(activeLockHolder(expired, 'user-0001', NOW_MS)).toBeNull();
    // 경계: 정확히 지금 만료 = 만료됨
    const atNow = lockInfo({ expiresAtIso: '2026-08-07T12:00:00Z' });
    expect(activeLockHolder(atNow, 'user-0001', NOW_MS)).toBeNull();
  });

  it('만료 시각 파싱 불가면 표시하는 쪽으로(fail-open)', () => {
    const broken = lockInfo({ expiresAtIso: 'not-a-date' });
    expect(activeLockHolder(broken, 'user-0001', NOW_MS)).toBe('박기사');
  });
});

// ── 마지막 실행 배지 ────────────────────────────────────────────────

describe('RUN_RESULT_BADGE', () => {
  it('runResultSchema 4종 전부 커버하고 STATUS 축 이름만 쓴다', () => {
    const results: RunResult[] = ['completed', 'stopped', 'error', 'autoPaused'];
    for (const r of results) {
      const info = RUN_RESULT_BADGE[r];
      expect(info.labelKo.length).toBeGreaterThan(0);
      expect(Object.keys(STATUS)).toContain(info.status);
    }
  });

  it('의미 매핑: 완주=success · 사람 정지=idle · 오류=error · 자동 정지=warn', () => {
    expect(RUN_RESULT_BADGE.completed.status).toBe('success');
    expect(RUN_RESULT_BADGE.stopped.status).toBe('idle');
    expect(RUN_RESULT_BADGE.error.status).toBe('error');
    expect(RUN_RESULT_BADGE.autoPaused.status).toBe('warn');
  });
});

// ── 뷰모델 조립 ─────────────────────────────────────────────────────

describe('buildTaskRows', () => {
  const ctx = {
    processNames: new Map([['proc-0001', '조립 라인 A']]),
    locks: new Map<string, LockInfo | null>([['task-0001', lockInfo()]]),
    conflictIds: new Set(['task-0002']),
    currentUserId: 'user-0001',
    nowMs: NOW_MS,
  };

  it('공정 이름을 매핑하고, 목록에 없는 공정은 null(화면이 id로 폴백)', () => {
    const rows = buildTaskRows(
      [
        entityMeta({ id: 'task-0001', processId: 'proc-0001' }),
        entityMeta({ id: 'task-0002', processId: 'proc-gone' }),
        entityMeta({ id: 'task-0003', processId: null }),
      ],
      ctx,
    );
    expect(rows[0]?.processName).toBe('조립 라인 A');
    expect(rows[1]?.processName).toBeNull();
    expect(rows[2]?.processName).toBeNull();
    expect(rows[2]?.processId).toBeNull();
  });

  it('taskSummary가 null이면(다른 kind 안전망) stepCount/lastRun도 null', () => {
    const rows = buildTaskRows([entityMeta({ taskSummary: null })], ctx);
    expect(rows[0]?.stepCount).toBeNull();
    expect(rows[0]?.lastRun).toBeNull();
    expect(rows[0]?.hasThumbnail).toBe(false);
  });

  it('잠금·충돌·삭제 메타가 행에 접힌다', () => {
    const rows = buildTaskRows(
      [
        entityMeta({ id: 'task-0001' }),
        entityMeta({ id: 'task-0002' }),
        entityMeta({ id: 'task-0003', meta: recordMeta({ deletedAtIso: '2026-08-06T00:00:00Z' }) }),
      ],
      ctx,
    );
    expect(rows[0]?.lockHolder).toBe('박기사');
    expect(rows[1]?.lockHolder).toBeNull();
    expect(rows[1]?.conflict).toBe(true);
    expect(rows[2]?.deletedAtIso).toBe('2026-08-06T00:00:00Z');
  });

  it('lastRun 요약이 그대로 실린다', () => {
    const rows = buildTaskRows(
      [
        entityMeta({
          taskSummary: {
            stepCount: 8,
            hasThumbnail: true,
            lastRun: { atIso: '2026-08-06T09:00:00Z', result: 'error' },
          },
        }),
      ],
      ctx,
    );
    expect(rows[0]?.stepCount).toBe(8);
    expect(rows[0]?.hasThumbnail).toBe(true);
    expect(rows[0]?.lastRun?.result).toBe('error');
  });
});

// ── 정렬 키 · 가시 행 ───────────────────────────────────────────────

describe('updatedAtSortKey', () => {
  it('유효한 ISO는 ms, 파싱 불가는 null', () => {
    expect(updatedAtSortKey('2026-08-05T00:00:00Z')).toBe(Date.parse('2026-08-05T00:00:00Z'));
    expect(updatedAtSortKey('garbage')).toBeNull();
  });
});

describe('visibleTaskRows', () => {
  const rows: TaskRowVM[] = [
    vm({ id: 'task-old', name: '오래된 작업', updatedAtIso: '2026-08-01T00:00:00Z' }),
    vm({
      id: 'task-new',
      name: '방금 작업',
      updatedAtIso: '2026-08-07T00:00:00Z',
      processId: 'proc-0001',
      processName: '조립 라인 A',
    }),
    vm({ id: 'task-mid', name: '중간 작업', updatedAtIso: '2026-08-04T00:00:00Z' }),
    vm({
      id: 'task-del',
      name: '삭제된 작업',
      updatedAtIso: '2026-08-06T00:00:00Z',
      deletedAtIso: '2026-08-06T12:00:00Z',
    }),
  ];

  it('일반 보기: 삭제 안 된 행만, 수정 내림차순(마지막 작업이 맨 위)', () => {
    const out = visibleTaskRows(rows, { query: '', processFilter: PROCESS_FILTER_ALL, trash: false });
    expect(out.map((r) => r.id)).toEqual(['task-new', 'task-mid', 'task-old']);
  });

  it('휴지통 보기: 삭제된 행만', () => {
    const out = visibleTaskRows(rows, { query: '', processFilter: PROCESS_FILTER_ALL, trash: true });
    expect(out.map((r) => r.id)).toEqual(['task-del']);
  });

  it('공정 필터와 검색이 함께 적용된다', () => {
    const byProc = visibleTaskRows(rows, { query: '', processFilter: 'proc-0001', trash: false });
    expect(byProc.map((r) => r.id)).toEqual(['task-new']);
    const byNone = visibleTaskRows(rows, { query: '', processFilter: PROCESS_FILTER_NONE, trash: false });
    expect(byNone.map((r) => r.id)).toEqual(['task-mid', 'task-old']);
    const byQuery = visibleTaskRows(rows, { query: '중간', processFilter: PROCESS_FILTER_ALL, trash: false });
    expect(byQuery.map((r) => r.id)).toEqual(['task-mid']);
  });

  it('검색은 공정 이름으로도 걸린다 (taskSearchText)', () => {
    const out = visibleTaskRows(rows, { query: '조립 라인', processFilter: PROCESS_FILTER_ALL, trash: false });
    expect(out.map((r) => r.id)).toEqual(['task-new']);
    expect(taskSearchText(rows[1] as TaskRowVM)).toContain('조립 라인 A');
  });

  it('수정 시각 파싱 불가 행은 방향과 무관하게 마지막', () => {
    const withBroken = [...rows, vm({ id: 'task-broken', updatedAtIso: 'garbage' })];
    const out = visibleTaskRows(withBroken, {
      query: '',
      processFilter: PROCESS_FILTER_ALL,
      trash: false,
    });
    expect(out[out.length - 1]?.id).toBe('task-broken');
  });
});

// ── 복제 이름 ───────────────────────────────────────────────────────

describe('duplicateName', () => {
  it("' (사본)' 접미를 붙인다", () => {
    expect(duplicateName('픽앤플레이스')).toBe('픽앤플레이스 (사본)');
  });

  it('이름 상한(80자)을 넘지 않는다 — displayNameSchema와 짝', () => {
    const long = 'a'.repeat(TASK_NAME_MAX_LEN);
    const out = duplicateName(long);
    expect(out.length).toBeLessThanOrEqual(TASK_NAME_MAX_LEN);
    expect(out.endsWith(' (사본)')).toBe(true);
  });

  it('사본의 사본도 상한 안에서 안전하다', () => {
    let name = 'b'.repeat(70);
    for (let i = 0; i < 5; i += 1) name = duplicateName(name);
    expect(name.length).toBeLessThanOrEqual(TASK_NAME_MAX_LEN);
  });
});

// ── 오프라인 사유 ───────────────────────────────────────────────────

describe('unavailableReasonKo', () => {
  it('서버 온라인이면 null (전 기능 사용 가능)', () => {
    expect(unavailableReasonKo({ mode: 'server', online: true })).toBeNull();
  });

  it('로컬 모드·오프라인이면 사람이 읽는 사유를 준다 (회색 버튼의 title)', () => {
    const local = unavailableReasonKo({ mode: 'local', online: false });
    expect(local).toContain('로컬 모드');
    const offline = unavailableReasonKo({ mode: 'server', online: false });
    expect(offline).toContain('오프라인');
  });
});

// ── id 발급 · 상태 라인 ─────────────────────────────────────────────

describe('makeEntityId', () => {
  it('entityIdSchema(8~64자)를 만족하고 호출마다 다르다', () => {
    const a = makeEntityId();
    const b = makeEntityId();
    expect(a.length).toBeGreaterThanOrEqual(8);
    expect(a.length).toBeLessThanOrEqual(64);
    expect(a).not.toBe(b);
  });
});

describe('taskCountLabelKo', () => {
  it('필터 없음: 전체 개수만', () => {
    expect(taskCountLabelKo(7, 7, false)).toBe('작업 7개');
  });

  it('필터로 줄었으면 "N개 중 M개 표시"', () => {
    expect(taskCountLabelKo(2, 7, false)).toBe('작업 7개 중 2개 표시');
  });

  it('휴지통 보기는 명사가 바뀐다', () => {
    expect(taskCountLabelKo(3, 3, true)).toBe('휴지통 3개');
  });
});
