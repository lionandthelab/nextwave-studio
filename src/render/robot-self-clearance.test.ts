// render/robot-self-clearance.test.ts — 라이브러리 로봇 URDF의 **자기 간섭** 회귀 가드
//
// 왜 이 테스트가 필요한가: `scripts/gate-browser.mjs --expect=robot-library`는 "관절이
// 말단을 움직이는가 / 손이 여닫히는가"를 재지만, **링크가 서로 파고드는가**는 못 본다.
// selfCollision이 기본 off라(CLAUDE.md §5) 물리 이벤트도 나지 않는다 — 화면에는 팔이
// 자기 어깨를 관통해 지나가는데 게이트는 전부 초록이다. 실제로 그 상태의 URDF가
// 검토에서 발견됐다(Cobot-7 어깨 14.7mm 관통, 3지 클로 패드 3.3mm 상호 관통).
//
// 판정은 순수 기하다: URDF <collision> 박스/실린더를 **방향 있는 박스(OBB)** 로 감싸고,
// 관절 범위를 훑으며 **인접하지 않은** 링크 쌍이 겹치는지 분리축 정리(SAT)로 본다.
// 인접 쌍(부모-자식)은 관절에서 맞닿는 것이 정상이므로 제외한다.
//
// AABB로 재면 안 된다 — 회전한 긴 링크의 축정렬 상자가 크게 부풀어 **정상인 arm6조차
// 50mm 관통으로 보고된다**(실측). 링크가 전부 박스/실린더라 OBB-SAT가 정확한 도구다.
// 실린더는 외접 박스로 감싸므로 여전히 보수적이다 — 즉 이 테스트가 통과하면 진짜 안전하고,
// 실패하면 실린더 모서리 때문일 수 있어 수치를 보고 판단해야 한다.

// @vitest-environment jsdom
//
// jsdom 환경을 파일 단위로 켠다 — urdf-loader가 DOMParser/Document를 요구하는데
// 저장소 기본 vitest 환경은 node다. `jsdom` 패키지를 직접 import하지 않는 이유:
// 그러려면 @types/jsdom과 @types/node가 필요하고, tsconfig의 types에 'node'를 열면
// **core를 포함한 src 전체**가 node 전역을 보게 되어 계층 규율(§3)이 헐거워진다.
// URDF도 node:fs가 아니라 Vite의 ?raw 임포트로 읽어 node 의존을 0으로 유지한다.

import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import arm6Urdf from '../../public/assets/robots/arm6/arm6.urdf?raw';
import scara4Urdf from '../../public/assets/robots/scara4/scara4.urdf?raw';
import cobot7Urdf from '../../public/assets/robots/cobot7/cobot7.urdf?raw';
import type { RobotFkView } from '../core/robot-types';

/** 링크가 서로 파고들어도 되는 최대 깊이 (m). 접촉면 오차만 허용한다. */
const MAX_PENETRATION_M = 0.001;
/** 관절 범위를 훑는 표본 수 (양 끝 포함) */
const SWEEP_SAMPLES = 9;

/**
 * gripper: templates.ts의 RobotSpec.gripper와 같은 값. 손가락은 **동시에** 구동되므로
 * 관절을 하나씩 훑는 것만으로는 상호 관통이 드러나지 않는다 — 3지 클로가 정확히 그렇다
 * (하나만 접으면 아무것도 안 만나고, 셋이 함께 접혀야 서로 맞닿는다).
 */
const ROBOTS = [
  {
    id: 'arm6',
    path: 'assets/robots/arm6/arm6.urdf',
    text: arm6Urdf,
    gripper: { joints: ['finger_left_joint', 'finger_right_joint'], open: 0.03, close: 0.0 },
  },
  {
    id: 'scara4',
    path: 'assets/robots/scara4/scara4.urdf',
    text: scara4Urdf,
    gripper: { joints: ['suction_joint'], open: 0.0, close: 0.012 },
  },
  {
    id: 'cobot7',
    path: 'assets/robots/cobot7/cobot7.urdf',
    text: cobot7Urdf,
    gripper: { joints: ['finger_a_joint', 'finger_b_joint', 'finger_c_joint'], open: -0.65, close: 0.05 },
  },
];

/** urdf-loader가 부르는 fetch를 ?raw로 읽어 둔 텍스트로 응답한다 */
function installUrdfFetch(): void {
  (globalThis as unknown as { fetch: unknown }).fetch = async (url: string) => {
    const path = String(url).replace(/^\//, '');
    const robot = ROBOTS.find((r) => path.endsWith(r.path));
    if (!robot) throw new Error(`알 수 없는 URDF 요청: ${url}`);
    return { ok: true, status: 200, text: async () => robot.text } as unknown as Response;
  };
}

/** collider 정의 → 링크 로컬 반폭(보수적 AABB). 실린더/캡슐은 반지름으로 감싼다. */
function halfExtentsOf(shape: { kind: string } & Record<string, unknown>): [number, number, number] | null {
  switch (shape.kind) {
    case 'box':
      return [...(shape.halfExtents as [number, number, number])];
    case 'sphere': {
      const r = shape.radius as number;
      return [r, r, r];
    }
    case 'cylinder':
    case 'capsule': {
      const r = shape.radius as number;
      const h = shape.halfHeight as number;
      return [r, h + (shape.kind === 'capsule' ? r : 0), r];
    }
    default:
      return null; // trimesh/convexHull/fromVisual — 라이브러리 로봇은 쓰지 않는다
  }
}

interface Obb {
  link: string;
  center: THREE.Vector3;
  axes: [THREE.Vector3, THREE.Vector3, THREE.Vector3];
  half: [number, number, number];
}

/** 링크 pose + 로컬 collider → 월드 OBB */
function worldObbs(fk: RobotFkView): Obb[] {
  const poses = fk.readLinkPoses();
  const out: Obb[] = [];
  for (const [link, defs] of fk.linkColliders) {
    const pose = poses.get(link);
    if (!pose) continue;
    for (const def of defs) {
      const half = halfExtentsOf(def.shape as { kind: string } & Record<string, unknown>);
      if (half === null) continue;
      const q = new THREE.Quaternion(...pose.rotation);
      const localOffset = def.offset?.position ?? [0, 0, 0];
      const offsetRot = def.offset?.rotation ?? [0, 0, 0, 1];
      const center = new THREE.Vector3(...localOffset)
        .applyQuaternion(q)
        .add(new THREE.Vector3(...pose.position));
      const rot = q.clone().multiply(new THREE.Quaternion(...offsetRot));
      const basis = new THREE.Matrix4().makeRotationFromQuaternion(rot);
      const axes: [THREE.Vector3, THREE.Vector3, THREE.Vector3] = [
        new THREE.Vector3().setFromMatrixColumn(basis, 0),
        new THREE.Vector3().setFromMatrixColumn(basis, 1),
        new THREE.Vector3().setFromMatrixColumn(basis, 2),
      ];
      out.push({ link, center, axes, half });
    }
  }
  return out;
}

/**
 * 두 OBB의 침투 깊이 (겹치지 않으면 0) — 분리축 정리.
 * 검사 축은 15개: 각 상자의 면 법선 3개씩 + 두 축의 외적 9개.
 */
function penetrationDepth(a: Obb, b: Obb): number {
  const delta = new THREE.Vector3().subVectors(b.center, a.center);
  const project = (box: Obb, axis: THREE.Vector3): number =>
    box.half[0] * Math.abs(axis.dot(box.axes[0])) +
    box.half[1] * Math.abs(axis.dot(box.axes[1])) +
    box.half[2] * Math.abs(axis.dot(box.axes[2]));

  const candidates: THREE.Vector3[] = [...a.axes, ...b.axes];
  for (const u of a.axes) {
    for (const v of b.axes) {
      const cross = new THREE.Vector3().crossVectors(u, v);
      if (cross.lengthSq() > 1e-12) candidates.push(cross.normalize());
    }
  }

  let minOverlap = Infinity;
  for (const axis of candidates) {
    const overlap = project(a, axis) + project(b, axis) - Math.abs(delta.dot(axis));
    if (overlap <= 0) return 0; // 분리축 발견 → 겹치지 않는다
    if (overlap < minOverlap) minOverlap = overlap;
  }
  return minOverlap;
}

describe('라이브러리 로봇 URDF: 자기 간섭', () => {
  installUrdfFetch();

  for (const robot of ROBOTS) {
    it(`${robot.id}: 관절 범위 전체에서 인접하지 않은 링크가 서로 파고들지 않는다`, async () => {
      const { loadUrdfRobot } = await import('./urdf');
      const handle = await loadUrdfRobot(new THREE.Object3D(), { urdfPath: robot.path });
      const fk = handle as unknown as RobotFkView;

      // 부모-자식(인접) 쌍은 관절에서 맞닿는 것이 정상 — URDF 트리에서 추출한다
      const urdfText = robot.text;
      const adjacent = new Set<string>();
      const jointRe = /<joint[^>]*>[\s\S]*?<\/joint>/g;
      for (const block of urdfText.match(jointRe) ?? []) {
        const parent = /<parent\s+link="([^"]+)"/.exec(block)?.[1];
        const child = /<child\s+link="([^"]+)"/.exec(block)?.[1];
        if (parent && child) adjacent.add([parent, child].sort().join('|'));
      }

      const worst: { pair: string; depth: number; at: string }[] = [];
      const check = (label: string): void => {
        const boxes = worldObbs(fk);
        for (let i = 0; i < boxes.length; i += 1) {
          for (let j = i + 1; j < boxes.length; j += 1) {
            const a = boxes[i];
            const b = boxes[j];
            if (a === undefined || b === undefined) continue;
            if (a.link === b.link) continue;
            const key = [a.link, b.link].sort().join('|');
            if (adjacent.has(key)) continue;
            const depth = penetrationDepth(a, b);
            if (depth > MAX_PENETRATION_M) worst.push({ pair: key, depth, at: label });
          }
        }
      };

      // 초기 자세 + 각 관절을 자기 범위 전체로 훑는다(다른 관절은 0)
      check('all-zero');
      for (const joint of fk.joints) {
        if (!joint.limits) continue;
        const [lo, hi] = joint.limits;
        for (let k = 0; k < SWEEP_SAMPLES; k += 1) {
          const value = lo + ((hi - lo) * k) / (SWEEP_SAMPLES - 1);
          fk.setJointValues({ [joint.name]: value });
          check(`${joint.name}=${value.toFixed(3)}`);
        }
        fk.setJointValues({ [joint.name]: 0 });
      }

      // 그리퍼는 **함께** 구동된다 — open↔close 전 구간을 한 덩어리로 훑는다.
      // 손가락을 하나씩만 훑으면 서로 만나지 않아 상호 관통이 영원히 드러나지 않는다.
      const { joints: handJoints, open, close } = robot.gripper;
      for (let k = 0; k < SWEEP_SAMPLES; k += 1) {
        const value = open + ((close - open) * k) / (SWEEP_SAMPLES - 1);
        const values: Record<string, number> = {};
        for (const name of handJoints) values[name] = value;
        fk.setJointValues(values);
        check(`gripper=${value.toFixed(3)}`);
      }
      for (const name of handJoints) fk.setJointValues({ [name]: 0 });

      const summary = worst
        .sort((x, y) => y.depth - x.depth)
        .slice(0, 5)
        .map((w) => `${w.pair} ${(w.depth * 1000).toFixed(1)}mm @ ${w.at}`);
      expect(summary, `자기 관통:\n  ${summary.join('\n  ')}`).toEqual([]);
    });
  }
});
