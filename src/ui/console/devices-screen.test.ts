// ui/console/devices-screen.test.ts — 장비 화면 순수 로직 단위 테스트 (DOM 비의존, node)
//
// mountDevicesScreen의 DOM 조립·배선은 브라우저 게이트 몫이다(toast.test.ts 관례).
// 여기서는 이 화면의 순수 계약만 검증한다:
// - kind → 아이콘/라벨 매핑 (임무 명세 고정: robot=robotArm · camera=camera · plc=plug)
// - 연결 배지 (가상=success / real=neutral + 미구현 사유 title)
// - 로봇 템플릿 필터 (실제 LIBRARY_TEMPLATES의 로봇 3종이 그대로 나온다 — 재사용 계약)
// - 문서 초안 정규화 (로봇 아니면 templateKey null · 연결은 virtual 고정)
// - 소속 공정 칩 소스 (공정 문서 deviceIds → deviceId별 공정 이름)

import { describe, expect, it } from 'vitest';
import { LIBRARY_TEMPLATES } from '../library/templates';
import {
  CAMERA_PRESET_HINT_KO,
  DEVICE_KIND_META,
  DEVICE_KIND_ORDER,
  MSG_REAL_MODE_PENDING_KO,
  buildDeviceDoc,
  deviceConnectionBadge,
  deviceKindIcon,
  deviceKindLabelKo,
  processNamesByDevice,
  robotTemplateOptions,
} from './devices-screen';

// ── kind 매핑 ───────────────────────────────────────────────────────

describe('deviceKindIcon / deviceKindLabelKo', () => {
  it('임무 명세 고정 매핑: robot=robotArm · camera=camera · plc=plug', () => {
    expect(deviceKindIcon('robot')).toBe('robotArm');
    expect(deviceKindIcon('camera')).toBe('camera');
    expect(deviceKindIcon('plc')).toBe('plug');
  });

  it('한국어 라벨 (PLC는 영문 두문자어 유지 — UI 언어 정책 §4-b)', () => {
    expect(deviceKindLabelKo('robot')).toBe('로봇');
    expect(deviceKindLabelKo('camera')).toBe('카메라');
    expect(deviceKindLabelKo('plc')).toBe('PLC');
  });

  it('표시 순서는 3종 전부를 정확히 한 번씩 담는다', () => {
    expect([...DEVICE_KIND_ORDER].sort()).toEqual(Object.keys(DEVICE_KIND_META).sort());
  });
});

// ── 연결 배지 ───────────────────────────────────────────────────────

describe('deviceConnectionBadge', () => {
  it("virtual → success '가상 연결됨'", () => {
    const spec = deviceConnectionBadge({ mode: 'virtual', endpoint: null });
    expect(spec).toEqual({ labelKo: '가상 연결됨', status: 'success' });
  });

  it("real → neutral '실제 연결 — 준비 중' + 미구현 사유 title", () => {
    const spec = deviceConnectionBadge({ mode: 'real', endpoint: 'http://bridge.local' });
    expect(spec.labelKo).toBe('실제 연결 — 준비 중');
    expect(spec.status).toBe('neutral');
    expect(spec.titleKo).toBe(MSG_REAL_MODE_PENDING_KO);
  });
});

// ── 로봇 템플릿 필터 (라이브러리 재사용 계약) ───────────────────────

describe('robotTemplateOptions', () => {
  it('실제 LIBRARY_TEMPLATES에서 로봇 3종(arm-6/scara-4/cobot-7)이 나온다', () => {
    const options = robotTemplateOptions(LIBRARY_TEMPLATES);
    expect(options.map((o) => o.key)).toEqual(['arm-6', 'scara-4', 'cobot-7']);
    for (const o of options) expect(o.labelKo.length).toBeGreaterThan(0);
  });

  it("section 'robots'만 통과한다 (objects는 장비 템플릿이 아니다)", () => {
    const options = robotTemplateOptions([
      { key: 'box', labelKo: 'Box · 박스', section: 'objects' },
      { key: 'arm-6', labelKo: 'Arm-6', section: 'robots' },
    ]);
    expect(options).toEqual([{ key: 'arm-6', labelKo: 'Arm-6' }]);
  });

  it('빈 입력은 빈 목록', () => {
    expect(robotTemplateOptions([])).toEqual([]);
  });
});

// ── 문서 초안 정규화 ────────────────────────────────────────────────

describe('buildDeviceDoc', () => {
  it('로봇은 templateKey를 유지하고 연결은 virtual 고정이다', () => {
    const doc = buildDeviceDoc({
      id: 'dev_12345678',
      name: '  1라인 로봇  ',
      kind: 'robot',
      templateKey: 'scara-4',
    });
    expect(doc.name).toBe('1라인 로봇');
    expect(doc.kind).toBe('robot');
    expect(doc.templateKey).toBe('scara-4');
    expect(doc.connection).toEqual({ mode: 'virtual', endpoint: null });
    expect(doc.notes).toBe('');
  });

  it('로봇이 아니면 templateKey를 null로 정규화한다 (데이터 거짓말 방지)', () => {
    expect(
      buildDeviceDoc({ id: 'dev_12345678', name: '검사 카메라', kind: 'camera', templateKey: 'arm-6' })
        .templateKey,
    ).toBeNull();
    expect(
      buildDeviceDoc({ id: 'dev_12345678', name: '라인 PLC', kind: 'plc', templateKey: 'cobot-7' })
        .templateKey,
    ).toBeNull();
  });
});

// ── 소속 공정 칩 소스 ───────────────────────────────────────────────

describe('processNamesByDevice', () => {
  it('공정 문서의 deviceIds에서 deviceId → 공정 이름 목록을 만든다', () => {
    const map = processNamesByDevice([
      { name: '포장 라인', deviceIds: ['dev_a', 'dev_b'] },
      { name: '검사 라인', deviceIds: ['dev_a'] },
      { name: '무장비 공정', deviceIds: [] },
    ]);
    expect(map.get('dev_a')).toEqual(['포장 라인', '검사 라인']);
    expect(map.get('dev_b')).toEqual(['포장 라인']);
    expect(map.has('dev_c')).toBe(false);
  });

  it('빈 입력은 빈 맵', () => {
    expect(processNamesByDevice([]).size).toBe(0);
  });
});

// ── 문구 상수 (카드 안내가 조용히 바뀌지 않게) ──────────────────────

describe('안내 문구', () => {
  it('카메라 동작 안내는 시점 프리셋을 말한다', () => {
    expect(CAMERA_PRESET_HINT_KO).toBe('카메라는 뷰포트 시점 프리셋으로 동작합니다');
  });

  it('real 미구현 사유는 브리지 추후 제공을 말한다', () => {
    expect(MSG_REAL_MODE_PENDING_KO).toBe('실제 장비 브리지는 추후 제공됩니다');
  });
});
