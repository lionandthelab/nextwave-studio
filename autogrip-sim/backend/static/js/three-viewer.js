/* ========================================
   STL Viewer - Canvas 2D Wireframe Renderer
   Parses binary/ASCII STL and renders a
   rotating wireframe preview on a <canvas>.
   ======================================== */

class STLViewer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.triangles = [];
    this.center = { x: 0, y: 0, z: 0 };
    this.scale = 1;
    this.rotation = { x: -0.5, y: 0, z: 0 };
    this.animationId = null;
    this.autoRotateSpeed = 0.008;

    // Store globally for cleanup
    window._stlViewerInstance = this;
  }

  loadFromBuffer(buffer) {
    this.triangles = this.parseSTL(buffer);
    if (this.triangles.length === 0) return;
    this.computeBounds();
    this.render();
  }

  parseSTL(buffer) {
    const view = new DataView(buffer);

    // Try binary first: 80 byte header + 4 byte triangle count
    if (buffer.byteLength > 84) {
      const triCount = view.getUint32(80, true);
      const expectedSize = 84 + triCount * 50;

      if (Math.abs(buffer.byteLength - expectedSize) < 10) {
        return this.parseBinarySTL(view, triCount);
      }
    }

    // Fall back to ASCII
    const text = new TextDecoder().decode(buffer);
    if (text.trimStart().startsWith('solid')) {
      return this.parseASCIISTL(text);
    }

    // Force binary parse
    const triCount = view.getUint32(80, true);
    return this.parseBinarySTL(view, triCount);
  }

  parseBinarySTL(view, triCount) {
    const triangles = [];
    let offset = 84;

    for (let i = 0; i < triCount && offset + 50 <= view.byteLength; i++) {
      // Skip normal (12 bytes)
      offset += 12;

      const verts = [];
      for (let v = 0; v < 3; v++) {
        verts.push({
          x: view.getFloat32(offset, true),
          y: view.getFloat32(offset + 4, true),
          z: view.getFloat32(offset + 8, true),
        });
        offset += 12;
      }

      // Skip attribute byte count
      offset += 2;

      triangles.push(verts);
    }

    return triangles;
  }

  parseASCIISTL(text) {
    const triangles = [];
    const facetRegex = /facet\s+normal\s+[\s\S]*?outer\s+loop\s+([\s\S]*?)endloop/gi;
    const vertexRegex = /vertex\s+([-\d.e+]+)\s+([-\d.e+]+)\s+([-\d.e+]+)/gi;
    let facetMatch;

    while ((facetMatch = facetRegex.exec(text)) !== null) {
      const verts = [];
      let vertMatch;
      const loopText = facetMatch[1];
      const localVertexRegex = /vertex\s+([-\d.e+]+)\s+([-\d.e+]+)\s+([-\d.e+]+)/gi;

      while ((vertMatch = localVertexRegex.exec(loopText)) !== null) {
        verts.push({
          x: parseFloat(vertMatch[1]),
          y: parseFloat(vertMatch[2]),
          z: parseFloat(vertMatch[3]),
        });
      }

      if (verts.length === 3) {
        triangles.push(verts);
      }
    }

    return triangles;
  }

  computeBounds() {
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

    for (const tri of this.triangles) {
      for (const v of tri) {
        minX = Math.min(minX, v.x);
        minY = Math.min(minY, v.y);
        minZ = Math.min(minZ, v.z);
        maxX = Math.max(maxX, v.x);
        maxY = Math.max(maxY, v.y);
        maxZ = Math.max(maxZ, v.z);
      }
    }

    this.center = {
      x: (minX + maxX) / 2,
      y: (minY + maxY) / 2,
      z: (minZ + maxZ) / 2,
    };

    const dx = maxX - minX;
    const dy = maxY - minY;
    const dz = maxZ - minZ;
    const maxDim = Math.max(dx, dy, dz);

    this.scale = Math.min(this.canvas.width, this.canvas.height) * 0.4 / (maxDim || 1);
  }

  project(point) {
    // Center the point
    let x = point.x - this.center.x;
    let y = point.y - this.center.y;
    let z = point.z - this.center.z;

    // Rotate X
    const cosX = Math.cos(this.rotation.x);
    const sinX = Math.sin(this.rotation.x);
    const y1 = y * cosX - z * sinX;
    const z1 = y * sinX + z * cosX;
    y = y1;
    z = z1;

    // Rotate Y
    const cosY = Math.cos(this.rotation.y);
    const sinY = Math.sin(this.rotation.y);
    const x1 = x * cosY + z * sinY;
    const z2 = -x * sinY + z * cosY;
    x = x1;
    z = z2;

    // Scale and project to 2D (orthographic)
    const sx = x * this.scale + this.canvas.width / 2;
    const sy = -y * this.scale + this.canvas.height / 2;

    return { x: sx, y: sy, z: z };
  }

  render() {
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;

    ctx.clearRect(0, 0, w, h);

    // Background
    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0, 0, w, h);

    if (this.triangles.length === 0) return;

    // Calculate projected triangles with depth for sorting
    const projected = [];
    for (const tri of this.triangles) {
      const pts = tri.map(v => this.project(v));
      const avgZ = (pts[0].z + pts[1].z + pts[2].z) / 3;
      projected.push({ pts, avgZ });
    }

    // Sort back-to-front (painter's algorithm)
    projected.sort((a, b) => a.avgZ - b.avgZ);

    // Draw triangles
    for (const { pts, avgZ } of projected) {
      // Simple depth-based shading
      const maxZ = this.triangles.length > 0 ? this.scale * 100 : 1;
      const brightness = 0.3 + 0.5 * ((avgZ + maxZ) / (2 * maxZ));
      const r = Math.round(88 * brightness);
      const g = Math.round(166 * brightness);
      const b = Math.round(255 * brightness);

      ctx.beginPath();
      ctx.moveTo(pts[0].x, pts[0].y);
      ctx.lineTo(pts[1].x, pts[1].y);
      ctx.lineTo(pts[2].x, pts[2].y);
      ctx.closePath();

      // Fill with semi-transparent color
      ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.15)`;
      ctx.fill();

      // Wireframe edges
      ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, 0.4)`;
      ctx.lineWidth = 0.5;
      ctx.stroke();
    }

    // Draw subtle grid on bottom
    this.drawGrid(ctx, w, h);
  }

  drawGrid(ctx, w, h) {
    ctx.save();
    ctx.strokeStyle = 'rgba(48, 54, 61, 0.3)';
    ctx.lineWidth = 0.5;

    const gridSize = 20;
    const cx = w / 2;
    const cy = h - 40;

    for (let i = -10; i <= 10; i++) {
      const x = cx + i * gridSize;
      ctx.beginPath();
      ctx.moveTo(x, cy - 5);
      ctx.lineTo(x, cy + 5);
      ctx.stroke();
    }

    ctx.beginPath();
    ctx.moveTo(cx - 10 * gridSize, cy);
    ctx.lineTo(cx + 10 * gridSize, cy);
    ctx.stroke();

    ctx.restore();
  }

  startAnimation() {
    const animate = () => {
      this.rotation.y += this.autoRotateSpeed;
      this.render();
      this.animationId = requestAnimationFrame(animate);
    };
    animate();
  }

  stopAnimation() {
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }
  }
}

// Expose globally
window.STLViewer = STLViewer;
