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
    this.controls.minDistance = 0.05;
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

    // Update near/far planes and zoom limits for close inspection
    this.camera.near = maxDim * 0.005;
    this.camera.far = maxDim * 20;
    this.camera.updateProjectionMatrix();
    this.controls.minDistance = maxDim * 0.03;
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

    // Scale hand so palm is wider than the object cross-section and
    // fingers are long enough to reach alongside the object height.
    // This ensures fingers hang OUTSIDE the object bounding box.
    const crossSection = Math.max(objSize.x, objSize.z);
    const s = Math.max(crossSection * 1.35, objSize.y * 0.8, maxDim * 0.6);

    // Store metrics for animation
    this._gs = {
      scale: s,
      objBbox: bbox,
      objCenter: new THREE.Vector3(),
      maxDim,
      startY: bbox.max.y + s * 1.6,
      graspY: bbox.max.y + s * 0.22 * 0.5 + s * 0.08,  // palm just above object top
      liftY: bbox.max.y + s * 1.0,
      // Curl angles: NEGATIVE = inward curl for front fingers (toward +Z / object center)
      curlAngles: {
        proximal: -Math.PI * 0.38,
        middle: -Math.PI * 0.33,
        distal: -Math.PI * 0.22,
      },
      // Thumb curls in opposite direction (positive) to oppose the fingers
      thumbCurl: {
        proximal: Math.PI * 0.30,
        middle: Math.PI * 0.25,
        distal: Math.PI * 0.18,
      },
      // Splay angles: fingers start slightly outward before closing
      splayOpen: 0.25,       // radians outward for 4 fingers (positive = away from object)
      thumbSplayOpen: -0.20, // thumb splays the other way
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
    const segWidths = [];

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
      segWidths.push(segW);
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

    return { group, joints, segWidths, tipMarker };
  }

  /** Add a translucent tray marker at the pick-and-place target position. */
  addTrayMarker(position) {
    this.removeTrayMarker();
    if (!this.mesh) return;

    const bbox = new THREE.Box3().setFromObject(this.mesh);
    const objSize = new THREE.Vector3();
    bbox.getSize(objSize);
    const maxDim = Math.max(objSize.x, objSize.y, objSize.z);

    // Scale from sim meters to viewer units (rough mapping)
    const scale = maxDim / 0.15; // assume object ~15cm in sim
    const trayX = (position[0] - 0.5) * scale; // offset from object center at (0.5,0,0.05)
    const trayZ = -position[1] * scale;
    const trayY = 0;

    this._trayGroup = new THREE.Group();

    // Tray base
    const baseW = maxDim * 0.8;
    const baseH = maxDim * 0.04;
    const baseD = maxDim * 0.8;
    const baseMat = new THREE.MeshPhysicalMaterial({
      color: 0x3fb950, metalness: 0.2, roughness: 0.6,
      transparent: true, opacity: 0.35,
    });
    const baseGeo = new THREE.BoxGeometry(baseW, baseH, baseD);
    const base = new THREE.Mesh(baseGeo, baseMat);
    base.position.y = baseH / 2;
    this._trayGroup.add(base);

    // Tray walls (4 sides)
    const wallH = maxDim * 0.12;
    const wallThick = maxDim * 0.02;
    const wallMat = new THREE.MeshPhysicalMaterial({
      color: 0x3fb950, metalness: 0.2, roughness: 0.6,
      transparent: true, opacity: 0.25,
    });
    const wallConfigs = [
      { w: baseW, d: wallThick, x: 0, z: -baseD / 2 },
      { w: baseW, d: wallThick, x: 0, z: baseD / 2 },
      { w: wallThick, d: baseD, x: -baseW / 2, z: 0 },
      { w: wallThick, d: baseD, x: baseW / 2, z: 0 },
    ];
    wallConfigs.forEach(cfg => {
      const wGeo = new THREE.BoxGeometry(cfg.w, wallH, cfg.d);
      const wall = new THREE.Mesh(wGeo, wallMat);
      wall.position.set(cfg.x, baseH + wallH / 2, cfg.z);
      this._trayGroup.add(wall);
    });

    // Target crosshair on the tray
    const ringGeo = new THREE.RingGeometry(maxDim * 0.08, maxDim * 0.12, 32);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0x3fb950, transparent: true, opacity: 0.5, side: THREE.DoubleSide,
    });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = baseH + 0.5;
    this._trayGroup.add(ring);

    this._trayGroup.position.set(trayX, trayY, trayZ);
    this.scene.add(this._trayGroup);
  }

  removeTrayMarker() {
    if (this._trayGroup) {
      this._trayGroup.traverse(child => {
        if (child.geometry) child.geometry.dispose();
        if (child.material) child.material.dispose();
      });
      this.scene.remove(this._trayGroup);
      this._trayGroup = null;
    }
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

  /**
   * Pre-compute mesh triangles in world space for exact collision detection.
   * Returns array of THREE.Triangle objects.
   */
  _buildMeshTriangles() {
    if (!this.mesh) return [];
    const geometry = this.mesh.geometry;
    const positions = geometry.attributes.position;
    const worldMatrix = this.mesh.matrixWorld;
    const triangles = [];

    for (let i = 0; i < positions.count; i += 3) {
      const v0 = new THREE.Vector3().fromBufferAttribute(positions, i).applyMatrix4(worldMatrix);
      const v1 = new THREE.Vector3().fromBufferAttribute(positions, i + 1).applyMatrix4(worldMatrix);
      const v2 = new THREE.Vector3().fromBufferAttribute(positions, i + 2).applyMatrix4(worldMatrix);
      triangles.push(new THREE.Triangle(v0, v1, v2));
    }
    return triangles;
  }

  /**
   * Compute the minimum unsigned distance from a point to any mesh triangle.
   * This is mathematically exact — no raycasting heuristics.
   */
  _minDistToTriangles(point, triangles) {
    let minDist = Infinity;
    const closestPt = new THREE.Vector3();

    for (const tri of triangles) {
      tri.closestPointToPoint(point, closestPt);
      const dist = point.distanceTo(closestPt);
      if (dist < minDist) {
        minDist = dist;
        if (minDist < 1e-6) return 0; // touching — early exit
      }
    }
    return minDist;
  }

  /**
   * Pre-compute per-finger max curl factor using exact point-to-triangle
   * distance collision detection. Samples points along each phalanx segment's
   * centerline AND at its edges (±width, ±depth offsets). Stops curling when
   * any sample point's distance to the nearest mesh triangle is below threshold.
   * Returns an array of factors [0..1] per finger, or null on error.
   */
  _precomputeFingerLimits() {
    if (!this.mesh || !this._fingerJoints || !this._gs) return null;

    const g = this._gs;
    const limits = [];
    const STEPS = 30;

    // Build world-space triangles once (exact geometry, no raycasting)
    this.mesh.updateMatrixWorld(true);
    const triangles = this._buildMeshTriangles();
    if (!triangles.length) return null;

    // Object AABB for quick rejection
    const objBox = new THREE.Box3().setFromObject(this.mesh);
    const objBoxExpanded = objBox.clone().expandByScalar(g.maxDim * 0.5);

    // Move gripper to grasp height temporarily
    const savedY = this.gripperGroup.position.y;
    this.gripperGroup.position.y = g.graspY;

    for (const finger of this._fingerJoints) {
      const baseAngles = finger.isThumb ? g.thumbCurl : g.curlAngles;
      const savedRots = finger.joints.map(j => j.rotation.x);
      let limitFactor = 1.0;

      for (let s = 1; s <= STEPS; s++) {
        const f = s / STEPS;

        // Set joints to this curl level
        finger.joints[0].rotation.x = baseAngles.proximal * f;
        finger.joints[1].rotation.x = baseAngles.middle * f;
        finger.joints[2].rotation.x = baseAngles.distal * f;
        this.gripperGroup.updateMatrixWorld(true);

        let penetrating = false;

        // Check each of the 3 phalanx segments
        for (let ji = 0; ji < 3 && !penetrating; ji++) {
          // Segment endpoints in world space
          const topPos = new THREE.Vector3();
          finger.joints[ji].getWorldPosition(topPos);
          const bottomPos = new THREE.Vector3();
          if (ji < 2) {
            finger.joints[ji + 1].getWorldPosition(bottomPos);
          } else {
            finger.tipMarker.getWorldPosition(bottomPos);
          }

          const halfW = (finger.segWidths[ji] || g.scale * 0.1) * 0.55;

          // Compute perpendicular axes for edge sampling
          const segDir = bottomPos.clone().sub(topPos);
          const segLen = segDir.length();
          if (segLen < 1e-6) continue;
          segDir.normalize();
          const refUp = Math.abs(segDir.y) < 0.9
            ? new THREE.Vector3(0, 1, 0)
            : new THREE.Vector3(1, 0, 0);
          const perp1 = new THREE.Vector3().crossVectors(segDir, refUp).normalize();
          const perp2 = new THREE.Vector3().crossVectors(segDir, perp1).normalize();

          // Sample 5 positions along segment, each with center + 4 edges = 25 pts
          for (let t = 0; t <= 1.0 && !penetrating; t += 0.25) {
            const center = topPos.clone().lerp(bottomPos, t);

            // Quick AABB rejection
            if (!objBoxExpanded.containsPoint(center)) continue;

            // Center point + 4 edge points (at finger half-width from center)
            const samplePoints = [
              center,
              center.clone().addScaledVector(perp1, halfW),
              center.clone().addScaledVector(perp1, -halfW),
              center.clone().addScaledVector(perp2, halfW),
              center.clone().addScaledVector(perp2, -halfW),
            ];

            for (const pt of samplePoints) {
              const dist = this._minDistToTriangles(pt, triangles);
              // If center point is within halfW of surface, finger edge touches
              // If edge point is within thin margin, finger edge is on surface
              const threshold = (pt === center) ? halfW : halfW * 0.15;
              if (dist < threshold) {
                penetrating = true;
                break;
              }
            }
          }
        }

        if (penetrating) {
          limitFactor = Math.max(0.05, (s - 1) / STEPS);
          break;
        }
      }

      // Restore joint rotations
      finger.joints.forEach((j, i) => { j.rotation.x = savedRots[i]; });
      limits.push(limitFactor);
    }

    // Restore gripper position
    this.gripperGroup.position.y = savedY;
    this.gripperGroup.updateMatrixWorld(true);

    return limits;
  }

  /**
   * Fallback curl factor when triangle distance is unavailable.
   */
  _calculateCurlFactor() {
    if (!this.mesh || !this._gs) return 0.65;
    const g = this._gs;
    const bbox = new THREE.Box3().setFromObject(this.mesh);
    const objSize = new THREE.Vector3();
    bbox.getSize(objSize);
    const objRadius = Math.max(objSize.x, objSize.z) / 2;
    const fingerReach = g.scale * 0.88 * 0.42;
    const ratio = objRadius / Math.max(fingerReach, 0.01);
    return Math.max(0.2, Math.min(0.95, 1.0 - ratio * 0.7));
  }

  _fitCameraForGrasp() {
    if (!this._gs) return;
    const g = this._gs;
    const sceneHeight = g.startY + g.scale;
    const lookY = sceneHeight * 0.35;
    const dist = Math.max(g.maxDim, g.scale) * 3.5;
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

    // 1. Reset position & splay fingers outward (pre-approach posture)
    this.gripperGroup.position.y = g.startY;
    this._fingerJoints.forEach(f => {
      const splay = f.isThumb ? g.thumbSplayOpen : g.splayOpen;
      f.joints[0].rotation.x = splay;
      f.joints[1].rotation.x = splay * 0.5;
      f.joints[2].rotation.x = 0;
    });

    // 2. Descend to grasp position with fingers splayed open
    await this._tween(this.gripperGroup.position, 'y', g.graspY, 800);

    // 3. Curl fingers inward — collision limits prevent mesh penetration
    const fingerLimits = this._precomputeFingerLimits();
    const fallbackFactor = this._calculateCurlFactor();
    const curlPromises = [];
    this._fingerJoints.forEach((f, idx) => {
      const angles = f.isThumb ? g.thumbCurl : g.curlAngles;
      const limit = fingerLimits ? fingerLimits[idx] : fallbackFactor;
      const targets = [
        angles.proximal * limit,
        angles.middle * limit,
        angles.distal * limit,
      ];
      f.joints.forEach((joint, i) => {
        curlPromises.push(this._tween(joint.rotation, 'x', targets[i], 600));
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

  // ==========================================================
  // Simulation Replay — data-driven animation from Isaac Sim
  // ==========================================================

  /**
   * Replay a simulation iteration using real data from the backend.
   * @param {Object} replayData — replay_data from SimulationResult SSE event
   *   phases: [{name, timestamp}], object_trajectory: [{position, timestamp}],
   *   contact_forces: [{finger, force_n}], mode, success, place_target, duration
   */
  async replaySimulation(replayData) {
    if (!this.gripperGroup || !this._gs || !this._fingerJoints) return;
    const g = this._gs;

    this._clearContactIndicators();
    this._clearResultLabel();
    this._clearPhaseLabel();

    // Reset gripper position & all finger joints to open
    this.gripperGroup.position.y = g.startY;
    this._fingerJoints.forEach(f => {
      f.joints.forEach(j => { j.rotation.x = 0; });
    });

    const phases = replayData.phases || [];
    const mode = replayData.mode || 'grasp_only';
    const success = replayData.success;
    const placeTarget = replayData.place_target;
    const trajectory = replayData.object_trajectory || [];
    const contactForces = replayData.contact_forces || [];

    // Compute animation speed factor based on duration (cap replay to ~6-10s)
    const simDuration = replayData.duration || 9.0;
    const targetReplayTime = Math.min(simDuration, 8.0);
    const speedFactor = simDuration / targetReplayTime;

    // Calculate phase-to-phase intervals (ms) scaled to replay time
    const phaseDurations = [];
    for (let i = 0; i < phases.length; i++) {
      const nextT = i + 1 < phases.length ? phases[i + 1].timestamp : phases[i].timestamp + 1.0;
      phaseDurations.push(Math.max(200, ((nextT - phases[i].timestamp) / speedFactor) * 1000));
    }

    // Transport target in viewer coordinates (for pick-and-place lateral move)
    let transportX = 0;
    let transportZ = 0;
    if (placeTarget && this.mesh) {
      const bbox = new THREE.Box3().setFromObject(this.mesh);
      const objSize = new THREE.Vector3();
      bbox.getSize(objSize);
      const maxDim = Math.max(objSize.x, objSize.y, objSize.z);
      const scale = maxDim / 0.15;
      transportX = (placeTarget[0] - 0.5) * scale;
      transportZ = -placeTarget[1] * scale;
    }

    // Execute each phase
    for (let i = 0; i < phases.length; i++) {
      const phase = phases[i];
      const dur = phaseDurations[i];
      this._showPhaseLabel(phase.name, i + 1, phases.length);

      switch (phase.name) {
        case 'APPROACH': {
          // Splay fingers outward while approaching
          this._fingerJoints.forEach(f => {
            const splay = f.isThumb ? g.thumbSplayOpen : g.splayOpen;
            f.joints[0].rotation.x = splay;
            f.joints[1].rotation.x = splay * 0.5;
            f.joints[2].rotation.x = 0;
          });
          await this._tween(this.gripperGroup.position, 'y', g.startY - g.scale * 0.2, dur);
          break;
        }

        case 'DESCEND':
          await this._tween(this.gripperGroup.position, 'y', g.graspY, dur);
          break;

        case 'GRASP': {
          // Curl fingers inward with per-finger collision limits
          const fingerLimits = this._precomputeFingerLimits();
          const fallbackFactor = this._calculateCurlFactor();
          const curlPromises = [];
          this._fingerJoints.forEach((f, idx) => {
            const baseAngles = f.isThumb ? g.thumbCurl : g.curlAngles;
            const limit = fingerLimits ? fingerLimits[idx] : fallbackFactor;
            const targets = [
              baseAngles.proximal * limit,
              baseAngles.middle * limit,
              baseAngles.distal * limit,
            ];
            f.joints.forEach((joint, j) => {
              curlPromises.push(this._tween(joint.rotation, 'x', targets[j], dur));
            });
          });
          await Promise.all(curlPromises);
          this._showContactPointsWithForce(contactForces, success);
          break;
        }

        case 'LIFT':
          // Lift gripper + object
          if (this.mesh && success) {
            const origObjY = this.mesh.position.y;
            this._replayObjOrigY = origObjY;
            await Promise.all([
              this._tween(this.gripperGroup.position, 'y', g.liftY, dur),
              this._tween(this.mesh.position, 'y', origObjY + g.maxDim * 0.5, dur),
            ]);
          } else {
            await this._tween(this.gripperGroup.position, 'y', g.liftY, dur);
          }
          break;

        case 'HOLD':
          // Grasp-only: hold and show result
          await this._delay(dur);
          break;

        case 'TRANSPORT':
          // Move laterally toward place target
          if (this.mesh && success) {
            await Promise.all([
              this._tween(this.gripperGroup.position, 'x', g.objCenter.x + transportX, dur),
              this._tween(this.gripperGroup.position, 'z', g.objCenter.z + transportZ, dur),
              this._tween(this.mesh.position, 'x', this.mesh.position.x + transportX, dur),
              this._tween(this.mesh.position, 'z', this.mesh.position.z + transportZ, dur),
            ]);
          } else {
            await Promise.all([
              this._tween(this.gripperGroup.position, 'x', g.objCenter.x + transportX, dur),
              this._tween(this.gripperGroup.position, 'z', g.objCenter.z + transportZ, dur),
            ]);
          }
          break;

        case 'PLACE':
          // Lower into tray
          if (this.mesh && success) {
            await Promise.all([
              this._tween(this.gripperGroup.position, 'y', g.graspY, dur),
              this._tween(this.mesh.position, 'y', this._replayObjOrigY || 0, dur),
            ]);
          } else {
            await this._tween(this.gripperGroup.position, 'y', g.graspY, dur);
          }
          break;

        case 'RELEASE': {
          // Uncurl fingers
          this._clearContactIndicators();
          const releasePromises = [];
          this._fingerJoints.forEach(f => {
            f.joints.forEach(joint => {
              releasePromises.push(this._tween(joint.rotation, 'x', 0, dur));
            });
          });
          await Promise.all(releasePromises);
          break;
        }

        case 'RETRACT':
          // Raise arm back up
          await this._tween(this.gripperGroup.position, 'y', g.startY, dur);
          break;

        default:
          await this._delay(dur);
          break;
      }
    }

    // Final result feedback
    if (success) {
      await this._flashGripper(0x3fb950, 3);
      const resultText = mode === 'pick_and_place' ? 'PLACED' : 'HOLD';
      this._showResultLabel(resultText, 0x3fb950);
    } else {
      await this._flashGripper(0xf85149, 3);
      this._showResultLabel('FAILED', 0xf85149);
      // Drop animation on failure
      if (this.mesh && mode === 'grasp_only') {
        const origY = this.mesh.position.y;
        await this._tween(this.mesh.position, 'y', origY - g.maxDim * 0.15, 300);
        await this._delay(200);
        await this._tween(this.mesh.position, 'y', origY, 250);
      }
    }

    await this._delay(800);

    // Reset to start position
    this._clearContactIndicators();
    this._clearResultLabel();
    this._clearPhaseLabel();

    // Uncurl any remaining finger curl
    const resetPromises = [];
    this._fingerJoints.forEach(f => {
      f.joints.forEach(joint => {
        resetPromises.push(this._tween(joint.rotation, 'x', 0, 300));
      });
    });
    await Promise.all(resetPromises);

    // Reset object position if moved during PnP
    if (this.mesh && mode === 'pick_and_place') {
      const origCenter = g.objCenter.clone();
      origCenter.y = this._replayObjOrigY || this.mesh.position.y;
      await Promise.all([
        this._tween(this.mesh.position, 'x', 0, 400),
        this._tween(this.mesh.position, 'z', 0, 400),
        this._tween(this.mesh.position, 'y', this._replayObjOrigY || this.mesh.position.y, 400),
      ]);
    }

    // Return gripper to start
    await Promise.all([
      this._tween(this.gripperGroup.position, 'x', g.objCenter.x, 400),
      this._tween(this.gripperGroup.position, 'z', g.objCenter.z, 400),
      this._tween(this.gripperGroup.position, 'y', g.startY, 400),
    ]);

    this._replayObjOrigY = null;
  }

  /** Show contact indicators sized by force magnitude. */
  _showContactPointsWithForce(contactForces, success) {
    if (!this._fingerJoints || !this._gs) return;
    const g = this._gs;
    const color = success ? 0x3fb950 : 0xf85149;

    // Map force data by finger name
    const forceMap = {};
    contactForces.forEach(c => { forceMap[c.finger] = c.force_n || 0; });

    const fingerNames = ['index', 'middle', 'ring', 'pinky', 'thumb'];
    this._fingerJoints.forEach((f, idx) => {
      const worldPos = new THREE.Vector3();
      f.tipMarker.getWorldPosition(worldPos);

      const fName = fingerNames[idx] || `finger_${idx}`;
      const force = forceMap[fName] || 0;
      // Scale indicator by force (min 0.03, max 0.08 of scale)
      const radius = g.scale * (0.03 + Math.min(force / 20, 0.05));

      // Sphere
      const sGeo = new THREE.SphereGeometry(radius, 10, 10);
      const sMat = new THREE.MeshBasicMaterial({
        color, transparent: true, opacity: 0.9,
      });
      const sphere = new THREE.Mesh(sGeo, sMat);
      sphere.position.copy(worldPos);
      this.scene.add(sphere);
      this._contactIndicators.push(sphere);

      // Force label (only if force > 0)
      if (force > 0.1) {
        const canvas = document.createElement('canvas');
        canvas.width = 128;
        canvas.height = 32;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = `#${color.toString(16).padStart(6, '0')}`;
        ctx.font = 'bold 18px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(`${force.toFixed(1)}N`, 64, 16);

        const texture = new THREE.CanvasTexture(canvas);
        const spMat = new THREE.SpriteMaterial({ map: texture, transparent: true, opacity: 0.85 });
        const sprite = new THREE.Sprite(spMat);
        sprite.position.copy(worldPos);
        sprite.position.y += g.scale * 0.15;
        sprite.scale.set(g.scale * 0.5, g.scale * 0.12, 1);
        this.scene.add(sprite);
        this._contactIndicators.push(sprite);
      }

      // Glow ring
      const rGeo = new THREE.RingGeometry(radius * 1.5, radius * 2.2, 20);
      const rMat = new THREE.MeshBasicMaterial({
        color, transparent: true, opacity: 0.3, side: THREE.DoubleSide,
      });
      const ring = new THREE.Mesh(rGeo, rMat);
      ring.position.copy(worldPos);
      this.scene.add(ring);
      this._contactIndicators.push(ring);
    });
  }

  /** Show floating phase label above the gripper. */
  _showPhaseLabel(phaseName, phaseNum, totalPhases) {
    this._clearPhaseLabel();
    if (!this._gs) return;

    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');

    // Phase badge background
    ctx.fillStyle = 'rgba(88, 166, 255, 0.85)';
    ctx.beginPath();
    ctx.roundRect(20, 6, 472, 52, 8);
    ctx.fill();

    // Phase text
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 28px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(`${phaseNum}/${totalPhases}  ${phaseName}`, 256, 32);

    const texture = new THREE.CanvasTexture(canvas);
    const mat = new THREE.SpriteMaterial({ map: texture, transparent: true });
    this._phaseSprite = new THREE.Sprite(mat);

    const g = this._gs;
    this._phaseSprite.position.set(
      this.gripperGroup.position.x,
      this.gripperGroup.position.y + g.scale * 2.0,
      this.gripperGroup.position.z
    );
    this._phaseSprite.scale.set(g.maxDim * 2.0, g.maxDim * 0.25, 1);
    this.scene.add(this._phaseSprite);

    // Auto-update position to follow gripper
    this._phaseFollowId = setInterval(() => {
      if (this._phaseSprite && this.gripperGroup) {
        this._phaseSprite.position.set(
          this.gripperGroup.position.x,
          this.gripperGroup.position.y + g.scale * 2.0,
          this.gripperGroup.position.z
        );
      }
    }, 50);
  }

  _clearPhaseLabel() {
    if (this._phaseFollowId) {
      clearInterval(this._phaseFollowId);
      this._phaseFollowId = null;
    }
    if (this._phaseSprite) {
      this.scene.remove(this._phaseSprite);
      this._phaseSprite.material.map.dispose();
      this._phaseSprite.material.dispose();
      this._phaseSprite = null;
    }
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
    this.removeTrayMarker();
    this._clearResultLabel();
    this._clearPhaseLabel();
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
