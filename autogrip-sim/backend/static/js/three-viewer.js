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
