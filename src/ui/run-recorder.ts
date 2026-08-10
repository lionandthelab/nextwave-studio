// ui/run-recorder.ts — 실행 기록 레코더 (RunRecord 축적 — 순수, DOM/core 비의존)
//
// 오케스트레이터가 **이미 방출하는** 사건(노드 상태·개입·충돌)을 RunRecord 하나로
// 축적한다. 이 클래스는 아무것도 새로 계측하지 않는다 — 통합자(main)가 orchestrator의
// 상태 이벤트·충돌 로그·재생 컨트롤 콜백을 record* 메서드에 배선하고, 실행이 끝나면
// finish()가 스키마 검증(runRecordSchema.parse)을 통과한 RunRecord를 돌려준다.
// 서버 전송(runs.create)·오프라인 outbox 적재는 호출자 몫이다 (BACKEND §4·§6).
//
// ── 계층 규칙 (CLAUDE.md §3) ────────────────────────────────────────
// schema만 안다(ui → schema). DOM·core·render·api를 import하지 않는다 —
// node 환경 단위 테스트 대상이다(run-recorder.test.ts).
//
// ── 방어 계약 (임무 명세) ───────────────────────────────────────────
// - begin 없이 record*/noteStepDone → 조용히 무시, finish → null. 재생이 아닌 경로
//   (되감기·씬 편집 중 접촉 등)에서 이벤트가 새어 들어와도 기록이 오염되지 않는다.
// - 이중 finish → 두 번째는 null. 한 실행 = 한 RunRecord (runs는 append-only).
// - begin 재호출 → 진행 중이던 기록은 폐기하고 새로 시작(재생 재시작 의미론).
// - finish는 runRecordSchema.parse로 **자체 검증**한다 — 미검증 데이터를 서버/화면에
//   노출하지 않는다(불변식 §2.9와 같은 원칙). 무효 입력(빈 id 등)은 여기서 throw로
//   드러난다 — 조용한 오염보다 이른 실패가 낫다.

import { runRecordSchema } from '../schema/entities';
import type { RunCollision, RunIntervention, RunRecord, RunResult } from '../schema/entities';

/** begin() 입력 — RunRecord에서 실행 시작 시점에 확정되는 필드들의 스냅샷 */
export interface RunBeginInfo {
  readonly taskId: string;
  /** 실행 시점의 작업 이름 스냅샷 — 작업이 개명/삭제돼도 기록은 읽힌다 (entities.ts) */
  readonly taskName: string;
  readonly taskVersion: number;
  readonly processId: string | null;
  readonly operatorId: string;
  readonly operatorName: string;
  readonly stepsTotal: number;
  readonly startedAtIso: string;
}

/** finish() 입력 — 실행 종료 시점에 확정되는 필드들 */
export interface RunFinishInfo {
  readonly endedAtIso: string;
  readonly simTimeSec: number;
  readonly wallTimeSec: number;
}

/** 진행 중 실행의 내부 상태 (finish 전까지 가변 축적) */
interface ActiveRun {
  readonly id: string;
  readonly info: RunBeginInfo;
  stepsDone: number;
  readonly collisions: RunCollision[];
  readonly interventions: RunIntervention[];
}

/**
 * 실행 1회 = RunRecord 1건을 축적하는 레코더.
 *
 * 수명 주기: `begin(info)` → (`recordIntervention` | `recordCollision` |
 * `noteStepDone`)* → `finish(result, info)` → RunRecord.
 *
 * id는 begin 시점에 발급한다(기본 `crypto.randomUUID` — 클라이언트 발급 uuid가
 * 오프라인 생성을 지원하는 BACKEND §4 규약과 동일). 테스트는 생성자로 고정
 * 팩토리를 주입한다.
 */
export class RunRecorder {
  private active: ActiveRun | null = null;

  constructor(private readonly makeId: () => string = (): string => crypto.randomUUID()) {}

  /** 진행 중인 기록이 있는가 (begin 후 finish 전) */
  isActive(): boolean {
    return this.active !== null;
  }

  /**
   * 새 실행 기록을 시작한다. 진행 중이던 기록이 있으면 **폐기**하고 새로 시작한다
   * — 재생 재시작(⏹ 후 ▶)은 새 실행이며, 반쯤 쌓인 기록을 이어 붙이지 않는다.
   */
  begin(info: RunBeginInfo): void {
    this.active = {
      id: this.makeId(),
      info,
      stepsDone: 0,
      collisions: [],
      interventions: [],
    };
  }

  /** 사람/자동 개입 기록 (재생·일시정지·정지·노드 단위 실행·자동 정지). begin 전엔 무시. */
  recordIntervention(kind: RunIntervention['kind'], nodeId: string | null, atSimSec: number): void {
    if (this.active === null) return;
    this.active.interventions.push({ atSimSec, kind, nodeId });
  }

  /** 충돌 기록 (분류·발생 노드 포함 — collision-classify의 산출을 그대로 받는다). begin 전엔 무시. */
  recordCollision(collision: RunCollision): void {
    if (this.active === null) return;
    this.active.collisions.push(collision);
  }

  /**
   * 완료된 step 수를 갱신한다(누적 카운트 — 마지막 값이 남는다).
   * [0, stepsTotal]로 클램프하고 정수로 내린다 — k/n 표기가 n을 넘지 않게.
   */
  noteStepDone(count: number): void {
    if (this.active === null) return;
    const floored = Math.floor(count);
    this.active.stepsDone = Math.min(Math.max(0, floored), this.active.info.stepsTotal);
  }

  /**
   * 기록을 닫고 검증된 RunRecord를 돌려준다. begin 없이(또는 두 번째로) 부르면
   * null — 한 실행은 한 번만 기록된다. 스키마 위반은 ZodError로 throw된다(자체 검증).
   */
  finish(result: RunResult, info: RunFinishInfo): RunRecord | null {
    const active = this.active;
    if (active === null) return null;
    // 이중 finish 방지 — parse가 throw해도 다시 finish할 수 없다(오염된 기록 재사용 금지)
    this.active = null;

    const record: RunRecord = {
      id: active.id,
      taskId: active.info.taskId,
      taskName: active.info.taskName,
      taskVersion: active.info.taskVersion,
      processId: active.info.processId,
      operatorId: active.info.operatorId,
      operatorName: active.info.operatorName,
      startedAtIso: active.info.startedAtIso,
      endedAtIso: info.endedAtIso,
      result,
      stepsTotal: active.info.stepsTotal,
      stepsDone: active.stepsDone,
      simTimeSec: info.simTimeSec,
      wallTimeSec: info.wallTimeSec,
      collisions: active.collisions,
      interventions: active.interventions,
    };
    return runRecordSchema.parse(record);
  }
}
