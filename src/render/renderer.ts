// render/renderer.ts — three.js 렌더러 (씬/카메라/조명/그리드/orbit)
//
// three.js를 아는 유일한 계층은 render(+core/sync)다 (CLAUDE.md §3).
// 렌더는 물리 결과를 비추는 거울이며, 물리 pose를 역으로 쓰지 않는다.

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

export interface RendererOptions {
  skyColor?: string;
  cameraPosition?: [number, number, number];
  cameraTarget?: [number, number, number];
  cameraFov?: number;
}

// ── 씬별 옵션 기본값 (생성자·applySceneOptions 공용 — 매직넘버 금지, CLAUDE.md §4) ──

const DEFAULT_SKY_COLOR = '#1b1e23';
const DEFAULT_CAMERA_POSITION: [number, number, number] = [1.8, 1.4, 1.8];
const DEFAULT_CAMERA_TARGET: [number, number, number] = [0, 0.3, 0];
const DEFAULT_CAMERA_FOV = 50;

export class Renderer {
  readonly scene: THREE.Scene;
  readonly camera: THREE.PerspectiveCamera;
  private readonly webgl: THREE.WebGLRenderer;
  private readonly controls: OrbitControls;
  private readonly host: HTMLElement;
  private readonly onResize: () => void;
  /** 현재 씬의 기본 카메라 (Home 키가 되돌릴 지점) */
  private sceneCamPosition: [number, number, number] = DEFAULT_CAMERA_POSITION;
  private sceneCamTarget: [number, number, number] = DEFAULT_CAMERA_TARGET;

  constructor(host: HTMLElement, opts: RendererOptions = {}) {
    this.host = host;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(opts.skyColor ?? DEFAULT_SKY_COLOR);

    this.camera = new THREE.PerspectiveCamera(
      opts.cameraFov ?? DEFAULT_CAMERA_FOV,
      host.clientWidth / Math.max(host.clientHeight, 1),
      0.01,
      100,
    );
    const camPos = opts.cameraPosition ?? DEFAULT_CAMERA_POSITION;
    this.camera.position.set(...camPos);
    this.sceneCamPosition = camPos;

    this.webgl = new THREE.WebGLRenderer({ antialias: true });
    this.webgl.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.webgl.setSize(host.clientWidth, host.clientHeight);
    this.webgl.shadowMap.enabled = true;
    this.webgl.shadowMap.type = THREE.PCFSoftShadowMap;
    host.appendChild(this.webgl.domElement);

    this.controls = new OrbitControls(this.camera, this.webgl.domElement);
    const target = opts.cameraTarget ?? DEFAULT_CAMERA_TARGET;
    this.controls.target.set(...target);
    this.sceneCamTarget = target;
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.update();

    // 조명
    const hemi = new THREE.HemisphereLight(0xdfe8ff, 0x2a2d33, 0.9);
    this.scene.add(hemi);
    const dir = new THREE.DirectionalLight(0xffffff, 1.6);
    dir.position.set(3, 6, 3);
    dir.castShadow = true;
    dir.shadow.mapSize.set(2048, 2048);
    dir.shadow.camera.near = 0.1;
    dir.shadow.camera.far = 20;
    this.scene.add(dir);

    // 그리드 + 축 헬퍼 (순수 시각 요소 — 물리와 무관, 불변식 §2.1 예외 대상)
    const grid = new THREE.GridHelper(10, 100, 0x3a3f47, 0x2a2d33);
    this.scene.add(grid);
    const axes = new THREE.AxesHelper(0.5);
    axes.position.y = 0.001;
    this.scene.add(axes);

    this.onResize = () => {
      const w = this.host.clientWidth;
      const h = Math.max(this.host.clientHeight, 1);
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
      this.webgl.setSize(w, h);
    };
    window.addEventListener('resize', this.onResize);
  }

  /**
   * 씬 전환 시 새 SceneSpec의 카메라/환경 설정을 기존 렌더러에 재적용한다
   * (Phase 6 씬 전환 — 렌더러/캔버스는 씬을 가로질러 유지되고, 씬별 옵션만 갱신).
   * 미지정 옵션은 생성자와 동일한 기본값으로 되돌린다 — 이전 씬 설정이 새지 않는다.
   */
  applySceneOptions(opts: RendererOptions = {}): void {
    this.scene.background = new THREE.Color(opts.skyColor ?? DEFAULT_SKY_COLOR);
    const camPos = opts.cameraPosition ?? DEFAULT_CAMERA_POSITION;
    this.camera.position.set(...camPos);
    this.camera.fov = opts.cameraFov ?? DEFAULT_CAMERA_FOV;
    this.camera.updateProjectionMatrix();
    const target = opts.cameraTarget ?? DEFAULT_CAMERA_TARGET;
    this.controls.target.set(...target);
    this.controls.update();
    this.sceneCamPosition = camPos;
    this.sceneCamTarget = target;
  }

  /**
   * 대상 오브젝트를 화면에 담도록 카메라를 옮긴다 (F 키 · 충돌 로그 행 클릭).
   *
   * 구 구현에는 카메라 프레이밍이 아예 없어서, 충돌 로그 행의 "오브젝트 포커스"
   * 안내가 실제로는 펄스만 시켰다 — 대상이 화면 밖이거나 로봇 뒤에 있으면 사용자는
   * 아무것도 보지 못했다(UX_AUDIT: 레이블과 실제 동작 불일치).
   *
   * 시선 방향은 유지하고 거리만 바운딩 스피어에 맞춘다 — 사용자가 잡아 둔 각도를
   * 빼앗지 않는다(Blender/Onshape의 frame-selected 관행).
   */
  frameObject(object: THREE.Object3D, paddingRatio = 1.9): void {
    const box = new THREE.Box3().setFromObject(object);
    if (box.isEmpty()) return;
    const sphere = box.getBoundingSphere(new THREE.Sphere());
    const radius = Math.max(sphere.radius, 0.05);
    const halfFovRad = (this.camera.fov * Math.PI) / 360;
    const distance = Math.max((radius * paddingRatio) / Math.tan(halfFovRad), 0.3);
    const dir = new THREE.Vector3().subVectors(this.camera.position, this.controls.target);
    if (dir.lengthSq() < 1e-8) dir.set(1, 1, 1);
    dir.normalize();
    this.controls.target.copy(sphere.center);
    this.camera.position.copy(sphere.center).addScaledVector(dir, distance);
    this.controls.update();
  }

  /** 씬의 기본 카메라로 되돌린다 (Home 키 — UX_DESIGN §9) */
  resetCamera(): void {
    this.camera.position.set(...this.sceneCamPosition);
    this.controls.target.set(...this.sceneCamTarget);
    this.controls.update();
  }

  /**
   * WebGL 캔버스 요소 — ViewportInteraction이 포인터/TransformControls를 붙일 대상
   * (render/interaction.ts 헤더의 "통합 방법" 계약. 통합자 querySelector 의존 제거).
   */
  get domElement(): HTMLCanvasElement {
    return this.webgl.domElement;
  }

  /**
   * OrbitControls — 기즈모 드래그 중 잠금(enabled=false) 용도로만 노출한다
   * (ViewportInteractionDeps.orbitControls의 OrbitControlsLike 표면).
   * 카메라 상태 변경은 여전히 Renderer가 소유한다.
   */
  get orbitControls(): OrbitControls {
    return this.controls;
  }

  /** 매 프레임 호출 (Engine 루프의 마지막 단계) */
  draw(): void {
    this.controls.update();
    this.webgl.render(this.scene, this.camera);
  }

  dispose(): void {
    window.removeEventListener('resize', this.onResize);
    this.controls.dispose();
    this.webgl.dispose();
    this.webgl.domElement.remove();
  }
}
