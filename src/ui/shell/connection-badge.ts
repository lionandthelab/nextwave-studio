// ui/shell/connection-badge.ts — 연결 상태 배지 (docs/BACKEND.md §6)
//
// ApiClient의 ConnectionState 3상태를 상시 노출한다 — 상태는 항상 배지로 보인다
// (공통 규약). 라벨이 의미를 전달하고 색은 보조다:
//   online     → '서버 연결됨'            (STATUS.success)
//   offline    → '오프라인 — 로컬 저장 중' (STATUS.warn — 작업물은 outbox에 쌓인다)
//   local-only → '로컬 모드'              (neutral — 의도된 무서버 상태, 경고가 아니다)
// 동기화 대기(outbox) 건수는 통합자가 setPendingCount로 주입한다 — 이 모듈은
// api/offline을 폴링하지 않는다(배선은 통합자 몫).
//
// 순수 헬퍼(connectionBadgeStatus/LabelKo/Icon)는 DOM 없이 node 테스트된다(shell.test.ts).

import type { ConnectionState } from '../../api';
import { connectionLabel } from '../../api';
import type { ConnectionLabel } from '../../api';
import type { IconName } from '../icons';
import type { BadgeStatus } from '../console/primitives';
import { ensureConsoleStyles, makeBadge } from '../console/primitives';
import { ensureThemeStyles } from '../theme';

// ── 순수 헬퍼 (DOM 비의존 — node 테스트 대상) ───────────────────────

/** 3상태 → 한국어 기본 라벨 (BACKEND §6 — 오프라인은 "저장은 계속된다"를 함께 말한다) */
export const CONNECTION_LABEL_KO: Readonly<Record<ConnectionLabel, string>> = {
  online: '서버 연결됨',
  offline: '오프라인 — 로컬 저장 중',
  'local-only': '로컬 모드',
};

/** 3상태 → 배지 상태 축. local-only는 의도된 상태라 경고색을 쓰지 않는다. */
export function connectionBadgeStatus(state: ConnectionState): BadgeStatus {
  switch (connectionLabel(state)) {
    case 'online':
      return 'success';
    case 'offline':
      return 'warn';
    case 'local-only':
      return 'neutral';
  }
}

/** 배지 전체 라벨 — 동기화 대기 건수가 있으면 병기한다 */
export function connectionBadgeLabelKo(state: ConnectionState, pendingCount = 0): string {
  const base = CONNECTION_LABEL_KO[connectionLabel(state)];
  return pendingCount > 0 ? `${base} · 대기 ${pendingCount}건` : base;
}

/**
 * 좁은 레일(72px)에 들어가는 짧은 라벨. 아이콘만 두면 초록 체크 하나가 남아
 * **"연결됐다는 건지 저장됐다는 건지" 알 수 없다** — 작업이 서버에 올라갔는지가
 * 현장에서 가장 중요한 정보라 두세 글자라도 말로 남긴다(전체 문구는 title에 유지).
 */
export function connectionBadgeShortKo(state: ConnectionState, pendingCount = 0): string {
  if (pendingCount > 0) return `대기 ${pendingCount}`;
  switch (connectionLabel(state)) {
    case 'online':
      return '연결됨';
    case 'offline':
      return '오프라인';
    default:
      return '로컬';
  }
}

/** 대기 건수가 있으면 sync(재전송 예정), 아니면 상태 아이콘 */
export function connectionBadgeIcon(state: ConnectionState, pendingCount = 0): IconName {
  if (pendingCount > 0) return 'sync';
  return connectionLabel(state) === 'online' ? 'check' : 'cloudOff';
}

// ── 배지 위젯 ───────────────────────────────────────────────────────

export interface ConnectionBadgeHandle {
  readonly el: HTMLElement;
  setState(state: ConnectionState): void;
  /** outbox 대기 건수 주입 (통합자가 flush 전후로 갱신한다) */
  setPendingCount(count: number): void;
  /** true면 아이콘만 보인다(얇은 레일) — 전체 라벨은 title/aria-label로 유지된다 */
  setCompact(compact: boolean): void;
  dispose(): void;
}

export interface ConnectionBadgeOptions {
  readonly testid?: string;
}

/**
 * 연결 상태 배지를 만든다. 래퍼가 role="status"라 상태 전환이 정중히 발화된다 —
 * 대기 건수 증가는 라벨에 포함되므로 별도 알림이 필요 없다.
 */
export function makeConnectionBadge(
  initial: ConnectionState,
  opts: ConnectionBadgeOptions = {},
): ConnectionBadgeHandle {
  ensureThemeStyles();
  ensureConsoleStyles();
  const testid = opts.testid ?? 'connection-badge';

  const el = document.createElement('span');
  el.setAttribute('role', 'status');
  el.dataset.testid = testid;

  let state = initial;
  let pendingCount = 0;
  let compact = false;

  const render = (): void => {
    const labelKo = connectionBadgeLabelKo(state, pendingCount);
    // compact(좁은 레일)에서는 아이콘만 — 단, 동기화 대기가 있으면 **건수 숫자는 남긴다**.
    // "몇 건이 아직 서버에 없다"는 현장에서 가장 놓치면 안 되는 정보라, 폭이 없다고
    // 통째로 지우면 사용자가 알 방법이 사라진다(전체 문구는 title/aria-label에 유지).
    const compactText = pendingCount > 0 ? String(pendingCount) : '';
    const badge = makeBadge(compact ? compactText : labelKo, connectionBadgeStatus(state), {
      iconName: connectionBadgeIcon(state, pendingCount),
      testid: `${testid}-chip`,
    });
    el.textContent = '';
    el.appendChild(badge);
    el.title = labelKo;
    el.setAttribute('aria-label', labelKo);
  };
  render();

  return {
    el,
    setState: (next: ConnectionState): void => {
      if (next.mode === state.mode && next.online === state.online) return;
      state = next;
      render();
    },
    setPendingCount: (count: number): void => {
      const normalized = Number.isFinite(count) && count > 0 ? Math.floor(count) : 0;
      if (normalized === pendingCount) return;
      pendingCount = normalized;
      render();
    },
    setCompact: (next: boolean): void => {
      if (next === compact) return;
      compact = next;
      render();
    },
    dispose: (): void => {
      el.remove();
    },
  };
}
