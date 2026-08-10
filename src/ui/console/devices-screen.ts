// ui/console/devices-screen.ts — 콘솔 평면 ⑥ 장비 관리 화면 (docs/BACKEND.md Phase 12+)
//
// 장비(Device) = 씬의 로봇/카메라/PLC를 대표하는 개체. v1에서는 **가상 연결만 실동작**
// 한다(씬 로봇 엔티티 대응) — 실제 연결(real)은 어댑터 경계 예약이며, UI는 미구현 사유를
// 정직하게 표기한다(entities.ts deviceConnectionSchema 주석). 로봇 템플릿은
// src/ui/library/templates.ts의 것을 **templatesProvider 주입으로** 재사용한다 —
// 이 모듈이 라이브러리를 직접 import하면 콘솔 평면이 스튜디오 라이브러리에 결합된다.
//
// 계층/치수/언어/삭제 규약은 processes-screen.ts 헤더와 동일하다(같은 콘솔 평면 슬라이스).
// 순수 로직(kind 매핑·연결 배지·템플릿 필터·문서 정규화·공정 칩)은 DOM 없이 export되어
// node 환경 vitest가 검증한다(devices-screen.test.ts).

import type { ConnectionState, GetResult, ListResult, RemoveResult, SaveResult } from '../../api';
import type {
  DeviceConnection,
  DeviceDoc,
  DeviceKind,
  EntityMeta,
  ProcessDoc,
  RecordEnvelope,
} from '../../schema/entities';
import { createAnnouncer } from '../a11y';
import { makeIconButton } from '../icons';
import type { IconName } from '../icons';
import { COLOR, SPACE, TYPE, applyType, makeButton, styled } from '../theme';
import type { ToastHandle } from '../feedback/toast';
import {
  applyTouchTarget,
  filterRowsByQuery,
  makeBadge,
  makeCard,
  makeCardGrid,
  makeEmptyState,
  makeModalShell,
  makeSearchField,
} from './primitives';
import type { ModalShellHandle } from './primitives';
import { formRow, makeTextInput, sectionTitle, serverBlockReasonKo } from './processes-screen';

// ── 상수 (매직넘버 금지 — CLAUDE.md §4) ─────────────────────────────

/** 헤더 검색 필드 폭 — processes-screen과 동일 (콘솔 평면 화면 간 시각 정합) */
const SEARCH_FIELD_WIDTH_PX = 240;

// ── 순수 상수/헬퍼 (DOM 비의존 — node 테스트 대상) ──────────────────

/** real 모드 미구현 사유 — 비활성 토글의 title에 그대로 실린다 (정직한 경계 표기) */
export const MSG_REAL_MODE_PENDING_KO = '실제 장비 브리지는 추후 제공됩니다';
/** 카메라 장비의 동작 방식 안내 — v1 카메라는 물리 개체가 아니라 시점 프리셋이다 */
export const CAMERA_PRESET_HINT_KO = '카메라는 뷰포트 시점 프리셋으로 동작합니다';

export interface DeviceKindMeta {
  readonly icon: IconName;
  readonly labelKo: string;
}

/** kind → 아이콘/라벨 (임무 명세 고정: robot=robotArm · camera=camera · plc=plug) */
export const DEVICE_KIND_META: Readonly<Record<DeviceKind, DeviceKindMeta>> = {
  robot: { icon: 'robotArm', labelKo: '로봇' },
  camera: { icon: 'camera', labelKo: '카메라' },
  plc: { icon: 'plug', labelKo: 'PLC' },
};

/** 추가 다이얼로그의 kind 표시 순서 (가장 흔한 것부터) */
export const DEVICE_KIND_ORDER: readonly DeviceKind[] = ['robot', 'camera', 'plc'];

export function deviceKindIcon(kind: DeviceKind): IconName {
  return DEVICE_KIND_META[kind].icon;
}

export function deviceKindLabelKo(kind: DeviceKind): string {
  return DEVICE_KIND_META[kind].labelKo;
}

export interface ConnectionBadgeSpec {
  readonly labelKo: string;
  readonly status: 'success' | 'neutral';
  /** 있으면 배지 title로 노출 (real 모드의 미구현 사유) */
  readonly titleKo?: string;
}

/** 연결 형상 → 배지 — 가상=success '가상 연결됨' / real=neutral '실제 연결 — 준비 중' */
export function deviceConnectionBadge(connection: DeviceConnection): ConnectionBadgeSpec {
  if (connection.mode === 'virtual') {
    return { labelKo: '가상 연결됨', status: 'success' };
  }
  return { labelKo: '실제 연결 — 준비 중', status: 'neutral', titleKo: MSG_REAL_MODE_PENDING_KO };
}

/** templatesProvider가 돌려주는 항목의 최소 구조 — LibraryTemplate이 구조적으로 만족 */
export interface TemplateOptionSource {
  readonly key: string;
  readonly labelKo: string;
  readonly section: string;
}

export type TemplatesProvider = () => readonly TemplateOptionSource[];

export interface RobotTemplateOption {
  readonly key: string;
  readonly labelKo: string;
}

/** 라이브러리 템플릿에서 로봇(section 'robots')만 — 장비 추가 다이얼로그 select 소스 */
export function robotTemplateOptions(
  templates: readonly TemplateOptionSource[],
): RobotTemplateOption[] {
  return templates
    .filter((t) => t.section === 'robots')
    .map((t) => ({ key: t.key, labelKo: t.labelKo }));
}

export interface DeviceDraft {
  readonly id: string;
  readonly name: string;
  readonly kind: DeviceKind;
  readonly templateKey: string | null;
}

/**
 * 폼 초안 → DeviceDoc — 이름 trim, **로봇이 아니면 templateKey를 null로 정규화**
 * (카메라/PLC에 로봇 템플릿 키가 남으면 데이터가 거짓말을 한다). 연결은 v1 규약대로
 * virtual 고정(endpoint null) — real 전환 UI는 어댑터가 생기면 열린다.
 */
export function buildDeviceDoc(draft: DeviceDraft): DeviceDoc {
  return {
    id: draft.id,
    name: draft.name.trim(),
    kind: draft.kind,
    templateKey: draft.kind === 'robot' ? draft.templateKey : null,
    connection: { mode: 'virtual', endpoint: null },
    notes: '',
  };
}

/** 공정 문서들 → deviceId → 소속 공정 이름 목록 (카드의 소속 공정 칩 소스) */
export function processNamesByDevice(
  processes: readonly { readonly name: string; readonly deviceIds: readonly string[] }[],
): Map<string, string[]> {
  const map = new Map<string, string[]>();
  for (const p of processes) {
    for (const deviceId of p.deviceIds) {
      const names = map.get(deviceId);
      if (names !== undefined) names.push(p.name);
      else map.set(deviceId, [p.name]);
    }
  }
  return map;
}

// ── deps 계약 (좁은 구조적 인터페이스) ──────────────────────────────

export interface DevicesResource {
  list(opts?: { readonly q?: string }): Promise<ListResult>;
  get(id: string): Promise<GetResult<DeviceDoc>>;
  create(doc: DeviceDoc): Promise<SaveResult<DeviceDoc>>;
  update(id: string, doc: DeviceDoc, baseVersion: number): Promise<SaveResult<DeviceDoc>>;
  remove(id: string): Promise<RemoveResult>;
  restore(id: string): Promise<GetResult<DeviceDoc>>;
}

/** 소속 공정 칩용 읽기 전용 표면 — 목록 메타에는 deviceIds가 없어 문서를 당긴다 */
export interface ProcessesReadResource {
  list(): Promise<ListResult>;
  get(id: string): Promise<GetResult<ProcessDoc>>;
}

export interface DevicesScreenDeps {
  readonly devices: DevicesResource;
  readonly processes: ProcessesReadResource;
  /** 라이브러리 로봇 템플릿 주입 (`() => LIBRARY_TEMPLATES` — 통합자가 배선) */
  readonly templatesProvider: TemplatesProvider;
  /** 연결 상태 getter — refresh 시점마다 재평가한다 */
  connection(): ConnectionState;
  readonly toast: ToastHandle;
  /** 카메라 장비의 [시점 적용] — 없으면 버튼을 그리지 않는다 */
  onApplyCameraPreset?(deviceId: string): void;
}

export interface DevicesScreenHandle {
  refresh(): void;
  dispose(): void;
}

// ── 마운트 ──────────────────────────────────────────────────────────

export function mountDevicesScreen(
  host: HTMLElement,
  deps: DevicesScreenDeps,
): DevicesScreenHandle {
  const root = styled(document.createElement('div'), {
    display: 'flex',
    flexDirection: 'column',
    gap: SPACE.xl,
    padding: SPACE.xl,
    boxSizing: 'border-box',
    height: '100%',
    minHeight: '0',
    overflow: 'hidden',
  });
  root.dataset.testid = 'devices-screen';
  host.appendChild(root);

  const announcer = createAnnouncer(root);

  // ── 헤더 ──────────────────────────────────────────────────────────
  const header = styled(document.createElement('div'), {
    display: 'flex',
    alignItems: 'center',
    gap: SPACE.lg,
    flexWrap: 'wrap',
  });
  const heading = applyType(document.createElement('h2'), TYPE.title);
  styled(heading, { margin: '0', color: COLOR.textStrong });
  heading.textContent = '장비';
  header.appendChild(heading);

  const connHost = styled(document.createElement('span'), { display: 'inline-flex' });
  header.appendChild(connHost);
  header.appendChild(styled(document.createElement('span'), { flex: '1 1 auto' }));

  let queryText = '';
  const search = makeSearchField({
    placeholderKo: '장비 검색',
    testid: 'devices-search',
    onInput: (q) => {
      queryText = q;
      renderList();
    },
  });
  styled(search.el, { width: `${SEARCH_FIELD_WIDTH_PX}px`, maxWidth: '100%' });
  header.appendChild(search.el);

  const addButton = makeIconButton('plus', '장비 추가', '장비 추가', 'devices-add', 'primary');
  applyTouchTarget(addButton);
  addButton.addEventListener('click', () => {
    openAddDialog();
  });
  header.appendChild(addButton);
  root.appendChild(header);

  // ── 목록 영역 ─────────────────────────────────────────────────────
  const listHost = styled(document.createElement('div'), {
    flex: '1 1 auto',
    minHeight: '0',
    overflowY: 'auto',
  });
  listHost.className = 'ui-scroll';
  root.appendChild(listHost);

  const grid = makeCardGrid({ testid: 'devices-grid' });

  // ── 상태 ──────────────────────────────────────────────────────────
  let disposed = false;
  let loadSeq = 0;
  let items: EntityMeta[] = [];
  const docs = new Map<string, RecordEnvelope<DeviceDoc>>();
  let processNames: Map<string, string[]> = new Map();
  let activeModal: ModalShellHandle | null = null;

  const closeActiveModal = (): void => {
    if (activeModal !== null) {
      activeModal.dispose();
      activeModal = null;
    }
  };

  const adoptModal = (modal: ModalShellHandle): void => {
    closeActiveModal();
    activeModal = modal;
    root.appendChild(modal.root);
    modal.open();
  };

  const finishModal = (modal: ModalShellHandle): void => {
    modal.close();
    if (activeModal === modal) activeModal = null;
    modal.dispose();
  };

  const blockReason = (): string | null => serverBlockReasonKo(deps.connection());

  const paintConnection = (): void => {
    connHost.textContent = '';
    const state = deps.connection();
    const reason = serverBlockReasonKo(state);
    if (reason !== null) {
      const badge = makeBadge(
        state.mode === 'local' ? '로컬 모드' : '오프라인',
        state.mode === 'local' ? 'neutral' : 'warn',
        { testid: 'devices-conn-badge' },
      );
      badge.title = reason;
      connHost.appendChild(badge);
    }
    addButton.disabled = reason !== null;
    addButton.title = reason !== null ? `장비 추가 — ${reason}` : '장비 추가';
  };

  // ── 목록 렌더 ─────────────────────────────────────────────────────

  const showOnly = (el: HTMLElement): void => {
    listHost.textContent = '';
    listHost.appendChild(el);
  };

  const showLoading = (): void => {
    const el = applyType(document.createElement('p'), TYPE.body);
    styled(el, { margin: '0', color: COLOR.muted });
    el.dataset.testid = 'devices-loading';
    el.textContent = '불러오는 중…';
    showOnly(el);
  };

  const showError = (messageKo: string): void => {
    const retry = makeIconButton('refresh', '다시 시도', '목록 다시 불러오기', 'devices-retry');
    retry.addEventListener('click', () => {
      refresh();
    });
    showOnly(
      makeEmptyState({
        iconName: 'alert',
        titleKo: '목록을 불러오지 못했습니다',
        hintKo: messageKo,
        actions: [retry],
        testid: 'devices-error',
      }),
    );
  };

  const makeAddActionButton = (testid: string): HTMLButtonElement => {
    const btn = makeIconButton('plus', '장비 추가', '장비 추가', testid, 'primary');
    const reason = blockReason();
    if (reason !== null) {
      btn.disabled = true;
      btn.title = `장비 추가 — ${reason}`;
    }
    btn.addEventListener('click', () => {
      openAddDialog();
    });
    return btn;
  };

  const buildDeviceCard = (m: EntityMeta, reason: string | null): HTMLElement => {
    const env = docs.get(m.id);
    const doc = env?.doc;

    // 보조행: kind 한국어 라벨 / templateKey(도메인 식별자 — lang en) / 카메라 안내
    const sublines: ({ text: string; lang?: string } | string)[] = [];
    if (doc !== undefined) {
      sublines.push(deviceKindLabelKo(doc.kind));
      if (doc.templateKey !== null) sublines.push({ text: doc.templateKey, lang: 'en' });
      if (doc.kind === 'camera') sublines.push(CAMERA_PRESET_HINT_KO);
    } else {
      sublines.push('정보를 불러오지 못했습니다');
    }

    // 연결 배지 (제목행 우측) — 색만이 아니라 라벨이 상태를 말한다
    let badge: HTMLElement | undefined;
    if (doc !== undefined) {
      const spec = deviceConnectionBadge(doc.connection);
      badge = makeBadge(spec.labelKo, spec.status, { testid: `device-conn-${m.id}` });
      if (spec.titleKo !== undefined) badge.title = spec.titleKo;
    }

    // 하단: kind 칩 + 소속 공정 칩(좌) · 액션 버튼(우)
    const chipsWrap = styled(document.createElement('span'), {
      display: 'flex',
      alignItems: 'center',
      flexWrap: 'wrap',
      gap: SPACE.xs,
      flex: '1 1 auto',
      minWidth: '0',
    });
    if (doc !== undefined) {
      chipsWrap.appendChild(
        makeBadge(deviceKindLabelKo(doc.kind), 'neutral', { iconName: deviceKindIcon(doc.kind) }),
      );
    }
    for (const name of processNames.get(m.id) ?? []) {
      chipsWrap.appendChild(makeBadge(name, 'neutral'));
    }

    const buttons = styled(document.createElement('span'), {
      display: 'flex',
      alignItems: 'center',
      gap: SPACE.sm,
    });
    const onApplyCameraPreset = deps.onApplyCameraPreset;
    if (doc !== undefined && doc.kind === 'camera' && onApplyCameraPreset !== undefined) {
      const preset = makeIconButton(
        'camera',
        '시점 적용',
        `'${m.name}' 시점 프리셋 적용`,
        `device-camera-preset-${m.id}`,
        'ghost',
      );
      applyTouchTarget(preset);
      preset.addEventListener('click', () => {
        onApplyCameraPreset(m.id);
      });
      buttons.appendChild(preset);
    }
    // real 전환 토글 — 어댑터 경계가 아직 없으므로 정직하게 비활성 + 사유 title
    const realToggle = makeIconButton(
      'plug',
      '실제 연결',
      MSG_REAL_MODE_PENDING_KO,
      `device-real-${m.id}`,
      'ghost',
    );
    applyTouchTarget(realToggle);
    realToggle.disabled = true;
    realToggle.title = MSG_REAL_MODE_PENDING_KO;
    buttons.appendChild(realToggle);

    const del = makeIconButton('trash', '', `장비 '${m.name}' 삭제`, `device-delete-${m.id}`, 'ghost');
    applyTouchTarget(del, { square: true });
    if (reason !== null) {
      del.disabled = true;
      del.title = `삭제 — ${reason}`;
    }
    del.addEventListener('click', () => {
      void removeDevice(m);
    });
    buttons.appendChild(del);

    return makeCard({
      title: m.name,
      sublines,
      badge,
      onClick: () => {
        openEditDialog(m);
      },
      actions: [chipsWrap, buttons],
      testid: `device-card-${m.id}`,
    });
  };

  const renderList = (): void => {
    if (items.length === 0) {
      showOnly(
        makeEmptyState({
          iconName: 'plug',
          titleKo: '등록된 장비가 없습니다',
          hintKo:
            '장비는 라인의 로봇·카메라·PLC를 대표합니다. 지금은 가상 연결(씬의 로봇 대응)만 실동작합니다.',
          actions: [makeAddActionButton('devices-add-empty')],
          testid: 'devices-empty',
        }),
      );
      return;
    }
    const filtered = filterRowsByQuery(items, queryText, (m) => {
      const doc = docs.get(m.id)?.doc;
      const kindLabel = doc !== undefined ? deviceKindLabelKo(doc.kind) : '';
      return `${m.name} ${kindLabel} ${doc?.templateKey ?? ''}`;
    });
    if (filtered.length === 0) {
      const clear = makeButton('검색어 지우기', '검색어 지우기', 'devices-search-clear-empty', 'ghost');
      applyTouchTarget(clear);
      clear.addEventListener('click', () => {
        search.setValue('');
      });
      showOnly(
        makeEmptyState({
          iconName: 'search',
          titleKo: '검색 결과가 없습니다',
          actions: [clear],
          testid: 'devices-search-empty',
        }),
      );
      return;
    }
    const reason = blockReason();
    grid.setCards(filtered.map((m) => buildDeviceCard(m, reason)));
    showOnly(grid.el);
  };

  // ── 데이터 로드 ───────────────────────────────────────────────────

  const load = async (): Promise<void> => {
    const seq = ++loadSeq;
    paintConnection();
    // 로컬 모드 — 서버 개체가 없다. 요청을 보내지 않는다 (tasks/processes 화면과 동일 계약)
    if (deps.connection().mode === 'local') {
      items = [];
      docs.clear();
      renderList();
      return;
    }
    showLoading();
    const [devRes, procRes] = await Promise.all([deps.devices.list(), deps.processes.list()]);
    if (disposed || seq !== loadSeq) return;
    if (devRes.kind !== 'ok') {
      showError(devRes.messageKo);
      return;
    }
    items = devRes.items;

    // 카드에 kind/연결/템플릿이 필요하다 — 목록 메타에는 payload가 없어 문서를 당긴다.
    docs.clear();
    const procDocs: ProcessDoc[] = [];
    await Promise.all([
      ...devRes.items.map(async (m) => {
        const r = await deps.devices.get(m.id);
        if (r.kind === 'ok') docs.set(m.id, r.record);
      }),
      // 소속 공정 칩 — 공정 문서의 deviceIds가 진실(장비는 공정을 모른다)
      ...(procRes.kind === 'ok'
        ? procRes.items.map(async (p) => {
            const r = await deps.processes.get(p.id);
            if (r.kind === 'ok') procDocs.push(r.record.doc);
          })
        : []),
    ]);
    if (disposed || seq !== loadSeq) return;
    processNames = processNamesByDevice(procDocs);
    renderList();
    announcer.announce(`장비 ${items.length}대`);
  };

  const refresh = (): void => {
    void load();
  };

  // ── 삭제 (soft + 실행취소 토스트 — CLAUDE.md §2.11) ───────────────

  const removeDevice = async (m: EntityMeta): Promise<void> => {
    const r = await deps.devices.remove(m.id);
    if (disposed) return;
    if (r.kind !== 'ok') {
      deps.toast.show('error', '장비를 삭제하지 못했습니다', { detail: r.messageKo });
      return;
    }
    deps.toast.show('success', `장비 '${m.name}' 삭제됨`, {
      detail: '30일 안에 복원할 수 있습니다',
      action: {
        label: '실행 취소',
        onClick: () => {
          void restoreDevice(m);
        },
      },
    });
    announcer.announceNow(`장비 '${m.name}' 삭제됨`);
    refresh();
  };

  const restoreDevice = async (m: EntityMeta): Promise<void> => {
    const r = await deps.devices.restore(m.id);
    if (disposed) return;
    if (r.kind !== 'ok') {
      deps.toast.show('error', '장비를 복원하지 못했습니다', { detail: r.messageKo });
      return;
    }
    deps.toast.show('success', `장비 '${m.name}' 복원됨`);
    refresh();
  };

  // ── 추가 다이얼로그 ───────────────────────────────────────────────

  const openAddDialog = (): void => {
    const reason = blockReason();
    if (reason !== null) return; // 버튼이 비활성이라 정상 경로로는 도달하지 않는다 (방어)

    const modal = makeModalShell({
      titleKo: '장비 추가',
      testid: 'device-add-modal',
      onClose: () => {
        if (activeModal === modal) activeModal = null;
        modal.dispose();
      },
    });
    const markDirty = (): void => {
      modal.setDirty(true);
    };

    // 종류 select
    const kindSelect = document.createElement('select');
    kindSelect.className = 'ui-select';
    kindSelect.dataset.testid = 'device-kind-select';
    applyTouchTarget(kindSelect);
    for (const kind of DEVICE_KIND_ORDER) {
      const option = document.createElement('option');
      option.value = kind;
      option.textContent = deviceKindLabelKo(kind);
      if (kind === 'plc') option.setAttribute('lang', 'en'); // 'PLC'는 영문 두문자어
      kindSelect.appendChild(option);
    }

    // 로봇 템플릿 select (templatesProvider 주입 — 라이브러리 3종 재사용)
    const templateSelect = document.createElement('select');
    templateSelect.className = 'ui-select';
    templateSelect.dataset.testid = 'device-template-select';
    // 옵션 라벨은 'Arm-6 · 6축 로봇팔'처럼 영문 키가 앞선다 — 도메인 식별자 표기
    templateSelect.setAttribute('lang', 'en');
    applyTouchTarget(templateSelect);
    const templateOptions = robotTemplateOptions(deps.templatesProvider());
    if (templateOptions.length === 0) {
      const option = document.createElement('option');
      option.value = '';
      option.textContent = '(템플릿 없음)';
      option.setAttribute('lang', 'ko');
      templateSelect.appendChild(option);
      templateSelect.disabled = true;
      templateSelect.title = '로봇 템플릿을 불러올 수 없습니다';
    } else {
      for (const t of templateOptions) {
        const option = document.createElement('option');
        option.value = t.key;
        option.textContent = t.labelKo;
        templateSelect.appendChild(option);
      }
    }
    templateSelect.addEventListener('change', markDirty);

    const templateRow = formRow('로봇 템플릿', templateSelect);
    const syncTemplateVisibility = (): void => {
      templateRow.style.display = kindSelect.value === 'robot' ? '' : 'none';
    };
    kindSelect.addEventListener('change', () => {
      syncTemplateVisibility();
      markDirty();
    });
    syncTemplateVisibility();

    const nameInput = makeTextInput('device-name-input', '', '예: 1라인 로봇');
    nameInput.addEventListener('input', markDirty);

    // 연결 안내 — v1은 virtual 고정, real은 정직하게 예약 표기
    const connNote = applyType(document.createElement('p'), TYPE.caption);
    styled(connNote, { margin: '0', color: COLOR.muted });
    connNote.textContent = `연결: 가상(씬의 로봇에 대응) — ${MSG_REAL_MODE_PENDING_KO}`;

    modal.body.appendChild(formRow('종류', kindSelect));
    modal.body.appendChild(templateRow);
    modal.body.appendChild(formRow('이름', nameInput));
    modal.body.appendChild(connNote);

    const submit = makeButton('추가', '장비 추가', 'device-add-submit', 'primary');
    applyTouchTarget(submit);
    submit.addEventListener('click', () => {
      void submitAdd();
    });
    modal.footer.appendChild(submit);

    const submitAdd = async (): Promise<void> => {
      if (nameInput.value.trim() === '') {
        deps.toast.show('warn', '이름을 입력하세요');
        nameInput.focus();
        return;
      }
      submit.disabled = true;
      // select 옵션은 DEVICE_KIND_ORDER에서만 나온다 — 단언 대신 좁히기로 방어
      const kindValue = kindSelect.value;
      const kind: DeviceKind =
        kindValue === 'camera' || kindValue === 'plc' ? kindValue : 'robot';
      const doc = buildDeviceDoc({
        id: crypto.randomUUID(),
        name: nameInput.value,
        kind,
        templateKey: templateSelect.value === '' ? null : templateSelect.value,
      });
      const r = await deps.devices.create(doc);
      if (disposed) return;
      submit.disabled = false;
      if (r.kind === 'ok') {
        deps.toast.show('success', `장비 '${doc.name}' 추가됨`);
        finishModal(modal);
        refresh();
      } else if (r.kind === 'conflict') {
        deps.toast.show('error', r.messageKo);
      } else {
        deps.toast.show('error', '장비를 추가하지 못했습니다', { detail: r.messageKo });
      }
    };

    adoptModal(modal);
    nameInput.focus();
  };

  // ── 이름 편집 다이얼로그 (카드 클릭) ──────────────────────────────

  const openEditDialog = (m: EntityMeta): void => {
    const env = docs.get(m.id);
    if (env === undefined) {
      deps.toast.show('error', '장비 정보를 불러오지 못했습니다', {
        detail: '새로고침 후 다시 시도하세요',
      });
      return;
    }
    const doc = env.doc;
    const reason = blockReason();

    const modal = makeModalShell({
      titleKo: `장비 · ${doc.name}`,
      testid: 'device-edit-modal',
      onClose: () => {
        if (activeModal === modal) activeModal = null;
        modal.dispose();
      },
    });

    const nameInput = makeTextInput('device-name-edit', doc.name, '장비 이름');
    nameInput.addEventListener('input', () => {
      modal.setDirty(true);
    });
    modal.body.appendChild(formRow('이름', nameInput));

    // 읽기 전용 정보 — 종류/템플릿은 만들 때 정한다 (씬 대응이 바뀌면 새 장비가 맞다)
    modal.body.appendChild(sectionTitle('정보'));
    const infoList = styled(document.createElement('div'), {
      display: 'flex',
      flexDirection: 'column',
      gap: SPACE.xs,
    });
    const infoRow = (labelKo: string, value: string, lang?: string): HTMLElement => {
      const row = styled(document.createElement('div'), {
        display: 'flex',
        gap: SPACE.md,
        alignItems: 'baseline',
      });
      const label = applyType(document.createElement('span'), TYPE.caption);
      styled(label, { color: COLOR.label, flex: 'none' });
      label.textContent = labelKo;
      const val = applyType(document.createElement('span'), TYPE.body);
      styled(val, { color: COLOR.text });
      if (lang !== undefined) val.setAttribute('lang', lang);
      val.textContent = value;
      row.appendChild(label);
      row.appendChild(val);
      return row;
    };
    infoList.appendChild(infoRow('종류', deviceKindLabelKo(doc.kind)));
    if (doc.templateKey !== null) {
      infoList.appendChild(infoRow('템플릿', doc.templateKey, 'en'));
    }
    const connSpec = deviceConnectionBadge(doc.connection);
    infoList.appendChild(
      infoRow('연결', `${connSpec.labelKo} — ${MSG_REAL_MODE_PENDING_KO}`),
    );
    if (doc.kind === 'camera') {
      infoList.appendChild(infoRow('동작', CAMERA_PRESET_HINT_KO));
    }
    modal.body.appendChild(infoList);

    const saveBtn = makeButton('저장', '장비 저장', 'device-save', 'primary');
    applyTouchTarget(saveBtn);
    if (reason !== null) {
      saveBtn.disabled = true;
      saveBtn.title = `저장 — ${reason}`;
    }
    saveBtn.addEventListener('click', () => {
      void save();
    });
    modal.footer.appendChild(saveBtn);

    const save = async (): Promise<void> => {
      if (nameInput.value.trim() === '') {
        deps.toast.show('warn', '이름을 입력하세요');
        nameInput.focus();
        return;
      }
      saveBtn.disabled = true;
      const next: DeviceDoc = { ...doc, name: nameInput.value.trim() };
      const r = await deps.devices.update(doc.id, next, env.meta.version);
      if (disposed) return;
      saveBtn.disabled = reason !== null;
      if (r.kind === 'ok') {
        deps.toast.show('success', `장비 '${next.name}' 저장됨`);
        finishModal(modal);
        refresh();
      } else if (r.kind === 'conflict') {
        deps.toast.show('error', r.messageKo, {
          detail: '모달을 닫고 다시 열면 최신 내용을 볼 수 있습니다',
        });
      } else {
        deps.toast.show('error', '장비를 저장하지 못했습니다', { detail: r.messageKo });
      }
    };

    adoptModal(modal);
  };

  // 초기 로드
  refresh();

  return {
    refresh,
    dispose: (): void => {
      disposed = true;
      loadSeq += 1;
      closeActiveModal();
      search.dispose();
      announcer.dispose();
      root.remove();
    },
  };
}
