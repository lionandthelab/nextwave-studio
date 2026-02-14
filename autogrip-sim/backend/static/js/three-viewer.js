/* ========================================
   STL Viewer - Three.js WebGL Renderer
   Real 3D viewer with STLLoader, OrbitControls,
   lighting, bounding box, center of mass, and grid.
   ======================================== */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';

class STLViewer {
  constructor(container) {
    this.container = container;
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.controls = null;
    this.mesh = null;
    this.wireframeMesh = null;
    this.bboxHelper = null;
    this.comMarker = null;
    this.gridHelper = null;
    this.animationId = null;
    this.showWireframe = false;
    this.showBbox = true;
    this.meshInfo = { vertices: 0, triangles: 0 };

    this._initScene();

    // Store globally for cleanup
    window._stlViewerInstance = this;
  }

  _initScene() {
    const width = this.container.clientWidth || 800;
    const height = this.container.clientHeight || 600;

    // Scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0d1117);

    // Camera
    this.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 10000);
    this.camera.position.set(0, 100, 200);

    // Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.2;
    this.container.appendChild(this.renderer.domElement);

    // Controls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.rotateSpeed = 0.8;
    this.controls.zoomSpeed = 1.2;
    this.controls.panSpeed = 0.8;
    this.controls.minDistance = 1;
    this.controls.maxDistance = 5000;

    // Lighting
    this._setupLighting();

    // Grid
    this._setupGrid();

    // Handle resize
    this._resizeObserver = new ResizeObserver(() => this._onResize());
    this._resizeObserver.observe(this.container);
  }

  _setupLighting() {
    // Ambient light - soft overall illumination
    const ambient = new THREE.AmbientLight(0x404060, 0.6);
    this.scene.add(ambient);

    // Hemisphere light - natural sky/ground gradient
    const hemiLight = new THREE.HemisphereLight(0x58a6ff, 0x1a1a2e, 0.4);
    this.scene.add(hemiLight);

    // Main directional light
    const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
    dirLight.position.set(50, 100, 80);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.width = 1024;
    dirLight.shadow.mapSize.height = 1024;
    dirLight.shadow.camera.near = 0.5;
    dirLight.shadow.camera.far = 500;
    dirLight.shadow.camera.left = -100;
    dirLight.shadow.camera.right = 100;
    dirLight.shadow.camera.top = 100;
    dirLight.shadow.camera.bottom = -100;
    this.scene.add(dirLight);

    // Fill light from the opposite side
    const fillLight = new THREE.DirectionalLight(0x58a6ff, 0.3);
    fillLight.position.set(-30, 40, -60);
    this.scene.add(fillLight);

    // Rim light from behind
    const rimLight = new THREE.DirectionalLight(0xbc8cff, 0.2);
    rimLight.position.set(0, -20, -80);
    this.scene.add(rimLight);
  }

  _setupGrid() {
    // Ground grid
    this.gridHelper = new THREE.GridHelper(200, 20, 0x30363d, 0x21262d);
    this.gridHelper.position.y = 0;
    this.gridHelper.material.opacity = 0.4;
    this.gridHelper.material.transparent = true;
    this.scene.add(this.gridHelper);
  }

  loadFromBuffer(buffer) {
    // Remove existing mesh
    this._clearModel();

    const loader = new STLLoader();
    const geometry = loader.parse(buffer);

    if (!geometry || geometry.attributes.position.count === 0) return;

    // Compute normals for proper lighting
    geometry.computeVertexNormals();

    // Store mesh info
    this.meshInfo = {
      vertices: geometry.attributes.position.count,
      triangles: geometry.attributes.position.count / 3,
    };

    // Material - metallic blue with slight transparency
    const material = new THREE.MeshPhysicalMaterial({
      color: 0x4a90d9,
      metalness: 0.3,
      roughness: 0.4,
      clearcoat: 0.2,
      clearcoatRoughness: 0.4,
      side: THREE.DoubleSide,
    });

    this.mesh = new THREE.Mesh(geometry, material);
    this.mesh.castShadow = true;
    this.mesh.receiveShadow = true;
    this.scene.add(this.mesh);

    // Center model on geometry
    geometry.computeBoundingBox();
    const bbox = geometry.boundingBox;
    const center = new THREE.Vector3();
    bbox.getCenter(center);
    geometry.translate(-center.x, -center.y, -center.z);

    // Position mesh so bottom sits on grid
    const size = new THREE.Vector3();
    geometry.boundingBox.getSize(size);
    // Recompute after centering
    geometry.computeBoundingBox();
    const newBbox = geometry.boundingBox;
    this.mesh.position.y = -newBbox.min.y;

    // Wireframe overlay
    const wireGeo = new THREE.WireframeGeometry(geometry);
    const wireMat = new THREE.LineBasicMaterial({
      color: 0x58a6ff,
      opacity: 0.08,
      transparent: true,
    });
    this.wireframeMesh = new THREE.LineSegments(wireGeo, wireMat);
    this.wireframeMesh.position.copy(this.mesh.position);
    this.wireframeMesh.visible = this.showWireframe;
    this.scene.add(this.wireframeMesh);

    // Bounding box helper
    this.bboxHelper = new THREE.Box3Helper(
      new THREE.Box3().setFromObject(this.mesh),
      0x58a6ff
    );
    this.bboxHelper.material.opacity = 0.3;
    this.bboxHelper.material.transparent = true;
    this.bboxHelper.visible = this.showBbox;
    this.scene.add(this.bboxHelper);

    // Center of mass indicator (approximate: centroid of bounding box)
    const comPos = new THREE.Vector3();
    new THREE.Box3().setFromObject(this.mesh).getCenter(comPos);
    const comGeo = new THREE.SphereGeometry(
      Math.max(size.x, size.y, size.z) * 0.015,
      16,
      16
    );
    const comMat = new THREE.MeshBasicMaterial({
      color: 0x3fb950,
      opacity: 0.8,
      transparent: true,
    });
    this.comMarker = new THREE.Mesh(comGeo, comMat);
    this.comMarker.position.copy(comPos);
    this.scene.add(this.comMarker);

    // Add glow ring around CoM
    const ringGeo = new THREE.RingGeometry(
      Math.max(size.x, size.y, size.z) * 0.02,
      Math.max(size.x, size.y, size.z) * 0.03,
      32
    );
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0x3fb950,
      opacity: 0.3,
      transparent: true,
      side: THREE.DoubleSide,
    });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.rotation.x = -Math.PI / 2;
    ring.position.copy(comPos);
    this.comMarker.add(ring);

    // Adjust grid to model size
    const maxDim = Math.max(size.x, size.y, size.z);
    const gridSize = Math.ceil(maxDim * 2 / 10) * 10;
    this.scene.remove(this.gridHelper);
    this.gridHelper = new THREE.GridHelper(gridSize, 20, 0x30363d, 0x21262d);
    this.gridHelper.material.opacity = 0.4;
    this.gridHelper.material.transparent = true;
    this.scene.add(this.gridHelper);

    // Fit camera to model
    this._fitCameraToModel(size, maxDim);

    // Dispatch info event
    this._dispatchMeshInfo();
  }

  _clearModel() {
    if (this.mesh) {
      this.scene.remove(this.mesh);
      this.mesh.geometry.dispose();
      this.mesh.material.dispose();
      this.mesh = null;
    }
    if (this.wireframeMesh) {
      this.scene.remove(this.wireframeMesh);
      this.wireframeMesh.geometry.dispose();
      this.wireframeMesh.material.dispose();
      this.wireframeMesh = null;
    }
    if (this.bboxHelper) {
      this.scene.remove(this.bboxHelper);
      this.bboxHelper = null;
    }
    if (this.comMarker) {
      this.scene.remove(this.comMarker);
      this.comMarker = null;
    }
  }

  _fitCameraToModel(size, maxDim) {
    const distance = maxDim * 2.5;
    this.camera.position.set(
      distance * 0.6,
      distance * 0.5,
      distance * 0.8
    );
    this.camera.lookAt(0, size.y * 0.3, 0);
    this.controls.target.set(0, size.y * 0.3, 0);
    this.controls.update();

    // Update near/far planes
    this.camera.near = maxDim * 0.01;
    this.camera.far = maxDim * 20;
    this.camera.updateProjectionMatrix();
  }

  resetCamera() {
    if (!this.mesh) return;
    const bbox = new THREE.Box3().setFromObject(this.mesh);
    const size = new THREE.Vector3();
    bbox.getSize(size);
    const maxDim = Math.max(size.x, size.y, size.z);
    this._fitCameraToModel(size, maxDim);
  }

  // ==========================================================
  // Unitree Inspire dexterous hand model & grasp animation
  // ==========================================================

  addGripper() {
    if (this.gripperGroup) this.removeGripper();
    if (!this.mesh) return;

    const bbox = new THREE.Box3().setFromObject(this.mesh);
    const objSize = new THREE.Vector3();
    bbox.getSize(objSize);
    const maxDim = Math.max(objSize.x, objSize.y, objSize.z);
    const s = maxDim * 0.3; // scale unit

    // Store metrics for animation
    this._gs = {
      scale: s,
      objBbox: bbox,
      objCenter: new THREE.Vector3(),
      maxDim,
      startY: bbox.max.y + maxDim * 0.85,
      graspY: bbox.max.y + s * 0.15,
      liftY: bbox.max.y + maxDim * 0.65,
      // Curl angles (radians) for closed grasp
      curlAngles: {
        proximal: Math.PI * 0.38,
        middle: Math.PI * 0.33,
        distal: Math.PI * 0.22,
      },
      thumbCurl: {
        proximal: Math.PI * 0.30,
        middle: Math.PI * 0.25,
        distal: Math.PI * 0.18,
      },
    };
    bbox.getCenter(this._gs.objCenter);

    // Materials - Unitree dark robotic aesthetic
    const bodyMat = new THREE.MeshPhysicalMaterial({
      color: 0x2a2a2e, metalness: 0.7, roughness: 0.25,
      clearcoat: 0.3, clearcoatRoughness: 0.3,
    });
    const jointMat = new THREE.MeshPhysicalMaterial({
      color: 0x1a1a1e, metalness: 0.5, roughness: 0.4,
    });
    const padMat = new THREE.MeshPhysicalMaterial({
      color: 0xe67e22, metalness: 0.1, roughness: 0.85,
    });
    const accentMat = new THREE.MeshPhysicalMaterial({
      color: 0x58a6ff, metalness: 0.6, roughness: 0.3,
      emissive: 0x1a3a5c, emissiveIntensity: 0.3,
    });

    this.gripperGroup = new THREE.Group();
    this._fingerJoints = [];

    // --- Forearm / Wrist ---
    const forearmGeo = new THREE.CylinderGeometry(s * 0.24, s * 0.22, s * 1.0, 20);
    const forearm = new THREE.Mesh(forearmGeo, bodyMat);
    forearm.position.y = s * 0.9;
    forearm.castShadow = true;
    this.gripperGroup.add(forearm);

    // Wrist rotation ring
    const wristRingGeo = new THREE.TorusGeometry(s * 0.25, s * 0.025, 8, 32);
    const wristRing = new THREE.Mesh(wristRingGeo, accentMat);
    wristRing.position.y = s * 0.42;
    wristRing.rotation.x = Math.PI / 2;
    this.gripperGroup.add(wristRing);

    // Second accent ring
    const wristRing2Geo = new THREE.TorusGeometry(s * 0.23, s * 0.015, 8, 32);
    const wristRing2 = new THREE.Mesh(wristRing2Geo, jointMat);
    wristRing2.position.y = s * 0.34;
    wristRing2.rotation.x = Math.PI / 2;
    this.gripperGroup.add(wristRing2);

    // --- Palm ---
    const palmW = s * 1.1;
    const palmH = s * 0.22;
    const palmD = s * 0.9;

    // Main palm body
    const palmGeo = new THREE.BoxGeometry(palmW, palmH, palmD);
    const palm = new THREE.Mesh(palmGeo, bodyMat);
    palm.castShadow = true;
    this.gripperGroup.add(palm);

    // Palm bottom accent plate
    const plateGeo = new THREE.BoxGeometry(palmW * 1.02, s * 0.025, palmD * 1.02);
    const plate = new THREE.Mesh(plateGeo, accentMat);
    plate.position.y = -palmH * 0.5;
    this.gripperGroup.add(plate);

    // Knuckle bar (visible mechanism across front of palm)
    const knuckleGeo = new THREE.CylinderGeometry(s * 0.04, s * 0.04, palmW * 0.85, 12);
    knuckleGeo.rotateZ(Math.PI / 2);
    const knuckle = new THREE.Mesh(knuckleGeo, jointMat);
    knuckle.position.set(0, -palmH * 0.3, -palmD * 0.44);
    this.gripperGroup.add(knuckle);

    // Palm detail lines (surface grooves)
    for (let i = -1; i <= 1; i += 2) {
      const grooveGeo = new THREE.BoxGeometry(palmW * 0.8, s * 0.005, s * 0.015);
      const groove = new THREE.Mesh(grooveGeo, jointMat);
      groove.position.set(0, -palmH * 0.51, i * palmD * 0.15);
      this.gripperGroup.add(groove);
    }

    // --- Fingers (index, middle, ring, pinky) ---
    const fingerSpecs = [
      { name: 'index',  x: -s * 0.34, z: -palmD * 0.5, len: s * 0.88, w: s * 0.1 },
      { name: 'middle', x: -s * 0.115, z: -palmD * 0.52, len: s * 0.98, w: s * 0.105 },
      { name: 'ring',   x:  s * 0.115, z: -palmD * 0.5, len: s * 0.92, w: s * 0.1 },
      { name: 'pinky',  x:  s * 0.34, z: -palmD * 0.46, len: s * 0.72, w: s * 0.085 },
    ];

    fingerSpecs.forEach(spec => {
      const finger = this._createFinger(spec, bodyMat, jointMat, padMat, s);
      finger.group.position.set(spec.x, -palmH * 0.4, spec.z);
      this.gripperGroup.add(finger.group);
      this._fingerJoints.push({ ...finger, isThumb: false });
    });

    // --- Thumb (opposable, offset to side) ---
    const thumbSpec = {
      name: 'thumb', x: -palmW * 0.56, z: -palmD * 0.12,
      len: s * 0.68, w: s * 0.12,
    };
    const thumb = this._createFinger(thumbSpec, bodyMat, jointMat, padMat, s);
    thumb.group.position.set(thumbSpec.x, -palmH * 0.3, thumbSpec.z);
    thumb.group.rotation.z = Math.PI * 0.35;  // splay outward
    thumb.group.rotation.y = Math.PI * 0.15;  // angle toward fingers
    this.gripperGroup.add(thumb.group);
    this._fingerJoints.push({ ...thumb, isThumb: true });

    // Thumb base joint (visible ball mechanism)
    const tbGeo = new THREE.SphereGeometry(s * 0.075, 12, 12);
    const thumbBase = new THREE.Mesh(tbGeo, jointMat);
    thumbBase.position.set(thumbSpec.x, -palmH * 0.3, thumbSpec.z);
    this.gripperGroup.add(thumbBase);

    // --- Position hand above object ---
    this.gripperGroup.position.set(
      this._gs.objCenter.x,
      this._gs.startY,
      this._gs.objCenter.z
    );
    this.scene.add(this.gripperGroup);

    this._contactIndicators = [];
    this._fitCameraForGrasp();
  }

  /** Create a single articulated finger with 3 phalanx segments and joint pivots. */
  _createFinger(spec, bodyMat, jointMat, padMat, s) {
    const { len, w } = spec;
    const segLens = [len * 0.42, len * 0.32, len * 0.26];
    const joints = [];

    const group = new THREE.Group();
    let parent = group;

    segLens.forEach((segLen, i) => {
      // Joint pivot group (rotation origin)
      const joint = new THREE.Group();

      // Visible joint sphere (knuckle mechanism)
      const jRadius = w * (i === 0 ? 0.55 : 0.45);
      const jGeo = new THREE.SphereGeometry(jRadius, 10, 10);
      const jMesh = new THREE.Mesh(jGeo, jointMat);
      jMesh.castShadow = true;
      joint.add(jMesh);

      // Phalanx segment (extends downward from joint)
      const segW = w * (1 - i * 0.1);
      const segD = segW * 0.9;
      const segGeo = new THREE.BoxGeometry(segW, segLen * 0.85, segD);
      const seg = new THREE.Mesh(segGeo, bodyMat);
      seg.position.y = -segLen * 0.5;
      seg.castShadow = true;
      joint.add(seg);

      // Side rails on proximal segment for mechanical look
      if (i === 0) {
        for (const side of [-1, 1]) {
          const railGeo = new THREE.BoxGeometry(s * 0.01, segLen * 0.6, segD * 1.1);
          const rail = new THREE.Mesh(railGeo, jointMat);
          rail.position.set(side * segW * 0.5, -segLen * 0.4, 0);
          joint.add(rail);
        }
      }

      // Fingertip pad and rounded cap on distal segment
      if (i === segLens.length - 1) {
        // Orange rubber pad
        const padGeo = new THREE.BoxGeometry(segW * 0.85, segLen * 0.5, segD * 0.3);
        const pad = new THREE.Mesh(padGeo, padMat);
        pad.position.set(0, -segLen * 0.65, segD * 0.35);
        joint.add(pad);

        // Rounded fingertip cap
        const tipGeo = new THREE.SphereGeometry(segW * 0.45, 10, 10);
        tipGeo.scale(1, 0.5, 0.9);
        const tip = new THREE.Mesh(tipGeo, padMat);
        tip.position.y = -segLen * 0.95;
        joint.add(tip);
      }

      // Position: first joint at group origin, subsequent at end of prev segment
      if (i > 0) {
        joint.position.y = -segLens[i - 1];
      }

      parent.add(joint);
      parent = joint;
      joints.push(joint);
    });

    // Invisible marker at the very tip (used for contact point positions)
    const tipMarker = new THREE.Object3D();
    tipMarker.position.y = -segLens[segLens.length - 1];
    joints[joints.length - 1].add(tipMarker);

    return { group, joints, tipMarker };
  }

  removeGripper() {
    if (this.gripperGroup) {
      this.scene.remove(this.gripperGroup);
      this.gripperGroup.traverse(child => {
        if (child.geometry) child.geometry.dispose();
        if (child.material) child.material.dispose();
      });
      this.gripperGroup = null;
    }
    this._clearContactIndicators();
    this._fingerJoints = null;
    this._gs = null;
  }

  _fitCameraForGrasp() {
    if (!this._gs) return;
    const g = this._gs;
    const sceneHeight = g.startY + g.scale;
    const lookY = sceneHeight * 0.38;
    const dist = g.maxDim * 2.8;
    this.camera.position.set(
      dist * 0.8,
      lookY + dist * 0.35,
      dist * 1.0
    );
    this.camera.lookAt(g.objCenter.x, lookY, g.objCenter.z);
    this.controls.target.set(g.objCenter.x, lookY, g.objCenter.z);
    this.controls.update();
  }

  /** Run one full grasp cycle animation with finger curling. Returns a Promise. */
  async animateGraspAttempt(success, iterLabel) {
    if (!this.gripperGroup || !this._gs || !this._fingerJoints) return;
    const g = this._gs;

    this._clearContactIndicators();
    this._clearResultLabel();

    // 1. Reset position & all finger joints to open
    this.gripperGroup.position.y = g.startY;
    this._fingerJoints.forEach(f => {
      f.joints.forEach(j => { j.rotation.x = 0; });
    });

    // 2. Descend to grasp position
    await this._tween(this.gripperGroup.position, 'y', g.graspY, 700);

    // 3. Curl all fingers simultaneously (thumb uses different angles)
    const curlPromises = [];
    this._fingerJoints.forEach(f => {
      const angles = f.isThumb ? g.thumbCurl : g.curlAngles;
      const targets = [angles.proximal, angles.middle, angles.distal];
      f.joints.forEach((joint, i) => {
        curlPromises.push(this._tween(joint.rotation, 'x', targets[i], 500));
      });
    });
    await Promise.all(curlPromises);

    // 4. Show contact points at fingertips
    this._showContactPoints(success);

    // 5. Lift
    await this._tween(this.gripperGroup.position, 'y', g.liftY, 550);

    // 6. Result feedback
    if (success) {
      await this._flashGripper(0x3fb950, 3);
      this._showResultLabel('HOLD', 0x3fb950);
    } else {
      await this._flashGripper(0xf85149, 3);
      this._showResultLabel('DROPPED', 0xf85149);
      if (this.mesh) {
        const origY = this.mesh.position.y;
        await this._tween(this.mesh.position, 'y', origY - g.maxDim * 0.15, 300);
        await this._delay(200);
        await this._tween(this.mesh.position, 'y', origY, 250);
      }
    }

    await this._delay(600);

    // 7. Uncurl fingers & return to start
    this._clearContactIndicators();
    this._clearResultLabel();
    const uncurlPromises = [];
    this._fingerJoints.forEach(f => {
      f.joints.forEach(joint => {
        uncurlPromises.push(this._tween(joint.rotation, 'x', 0, 350));
      });
    });
    await Promise.all(uncurlPromises);
    await this._tween(this.gripperGroup.position, 'y', g.startY, 500);
  }

  _showContactPoints(success) {
    if (!this._fingerJoints || !this._gs) return;
    const g = this._gs;
    const color = success ? 0x3fb950 : 0xf85149;

    // Place contact indicators at each fingertip (world position)
    this._fingerJoints.forEach(f => {
      const worldPos = new THREE.Vector3();
      f.tipMarker.getWorldPosition(worldPos);

      // Sphere
      const sGeo = new THREE.SphereGeometry(g.scale * 0.04, 10, 10);
      const sMat = new THREE.MeshBasicMaterial({
        color, transparent: true, opacity: 0.9,
      });
      const sphere = new THREE.Mesh(sGeo, sMat);
      sphere.position.copy(worldPos);
      this.scene.add(sphere);
      this._contactIndicators.push(sphere);

      // Glow ring
      const rGeo = new THREE.RingGeometry(g.scale * 0.06, g.scale * 0.1, 20);
      const rMat = new THREE.MeshBasicMaterial({
        color, transparent: true, opacity: 0.35, side: THREE.DoubleSide,
      });
      const ring = new THREE.Mesh(rGeo, rMat);
      ring.position.copy(worldPos);
      this.scene.add(ring);
      this._contactIndicators.push(ring);
    });
  }

  _clearContactIndicators() {
    if (!this._contactIndicators) return;
    this._contactIndicators.forEach(obj => {
      this.scene.remove(obj);
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) obj.material.dispose();
    });
    this._contactIndicators = [];
  }

  _showResultLabel(text, color) {
    this._clearResultLabel();
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = `#${color.toString(16).padStart(6, '0')}`;
    ctx.font = 'bold 36px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, 128, 32);

    const texture = new THREE.CanvasTexture(canvas);
    const mat = new THREE.SpriteMaterial({ map: texture, transparent: true, opacity: 0.9 });
    this._resultSprite = new THREE.Sprite(mat);
    const g = this._gs;
    this._resultSprite.position.set(
      g.objCenter.x,
      this.gripperGroup.position.y + g.scale * 1.5,
      g.objCenter.z
    );
    this._resultSprite.scale.set(g.maxDim * 1.2, g.maxDim * 0.3, 1);
    this.scene.add(this._resultSprite);
  }

  _clearResultLabel() {
    if (this._resultSprite) {
      this.scene.remove(this._resultSprite);
      this._resultSprite.material.map.dispose();
      this._resultSprite.material.dispose();
      this._resultSprite = null;
    }
  }

  async _flashGripper(color, times) {
    if (!this.gripperGroup) return;
    const originals = [];
    this.gripperGroup.traverse(child => {
      if (child.isMesh && child.material.color) {
        originals.push({ mesh: child, color: child.material.color.getHex() });
      }
    });
    for (let i = 0; i < times; i++) {
      originals.forEach(o => o.mesh.material.color.setHex(color));
      await this._delay(120);
      originals.forEach(o => o.mesh.material.color.setHex(o.color));
      await this._delay(120);
    }
  }

  /** Simple linear tween. Returns a Promise that resolves when done. */
  _tween(obj, prop, target, duration) {
    return new Promise(resolve => {
      const start = obj[prop];
      const delta = target - start;
      if (Math.abs(delta) < 0.001) { resolve(); return; }
      const t0 = performance.now();
      const step = (now) => {
        const elapsed = now - t0;
        const progress = Math.min(elapsed / duration, 1);
        // ease-in-out cubic
        const ease = progress < 0.5
          ? 4 * progress * progress * progress
          : 1 - Math.pow(-2 * progress + 2, 3) / 2;
        obj[prop] = start + delta * ease;
        if (progress < 1) {
          requestAnimationFrame(step);
        } else {
          obj[prop] = target;
          resolve();
        }
      };
      requestAnimationFrame(step);
    });
  }

  _delay(ms) {
    return new Promise(r => setTimeout(r, ms));
  }

  toggleWireframe() {
    this.showWireframe = !this.showWireframe;
    if (this.wireframeMesh) {
      this.wireframeMesh.visible = this.showWireframe;
      if (this.showWireframe) {
        this.wireframeMesh.material.opacity = 0.15;
      }
    }
    return this.showWireframe;
  }

  toggleBbox() {
    this.showBbox = !this.showBbox;
    if (this.bboxHelper) {
      this.bboxHelper.visible = this.showBbox;
    }
    return this.showBbox;
  }

  _dispatchMeshInfo() {
    const event = new CustomEvent('meshloaded', {
      detail: this.meshInfo,
    });
    this.container.dispatchEvent(event);
  }

  startAnimation() {
    const animate = () => {
      this.animationId = requestAnimationFrame(animate);
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    };
    animate();
  }

  stopAnimation() {
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }
  }

  _onResize() {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    if (width === 0 || height === 0) return;

    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  dispose() {
    this.stopAnimation();
    this.removeGripper();
    this._clearResultLabel();
    this._clearModel();
    if (this._resizeObserver) {
      this._resizeObserver.disconnect();
    }
    if (this.renderer) {
      this.renderer.dispose();
      if (this.renderer.domElement.parentNode) {
        this.renderer.domElement.parentNode.removeChild(this.renderer.domElement);
      }
    }
    if (this.controls) {
      this.controls.dispose();
    }
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    window._stlViewerInstance = null;
  }
}

// Expose globally
window.STLViewer = STLViewer;
