/* ========================================
   AutoGrip-Sim Engine - Application Logic
   ======================================== */

class AutoGripApp {
  constructor() {
    // State
    this.sessionId = null;
    this.cadFileId = null;
    this.manualFileId = null;
    this.eventSource = null;
    this.isRunning = false;
    this.currentIteration = 0;
    this.maxIter = 20;
    this.previousCode = '';
    this.currentCode = '';
    this.activeLogFilter = 'all';

    // Bind DOM elements
    this.bindElements();
    this.attachEventListeners();
    this.init();
  }

  bindElements() {
    // Upload zones
    this.cadDropZone = document.getElementById('cadDropZone');
    this.manualDropZone = document.getElementById('manualDropZone');
    this.cadFileInput = document.getElementById('cadFileInput');
    this.manualFileInput = document.getElementById('manualFileInput');

    // CAD info
    this.cadUploadContent = document.getElementById('cadUploadContent');
    this.cadFileInfo = document.getElementById('cadFileInfo');
    this.cadFileName = document.getElementById('cadFileName');
    this.cadFileSize = document.getElementById('cadFileSize');
    this.cadFileType = document.getElementById('cadFileType');
    this.cadDimensions = document.getElementById('cadDimensions');
    this.cadDimValues = document.getElementById('cadDimValues');
    this.cadProgress = document.getElementById('cadProgress');
    this.cadProgressFill = document.getElementById('cadProgressFill');
    this.cadRemoveBtn = document.getElementById('cadRemoveBtn');

    // Manual info
    this.manualUploadContent = document.getElementById('manualUploadContent');
    this.manualFileInfo = document.getElementById('manualFileInfo');
    this.manualFileName = document.getElementById('manualFileName');
    this.manualFileSize = document.getElementById('manualFileSize');
    this.manualProgress = document.getElementById('manualProgress');
    this.manualProgressFill = document.getElementById('manualProgressFill');
    this.manualPages = document.getElementById('manualPages');
    this.manualPageCount = document.getElementById('manualPageCount');
    this.manualRemoveBtn = document.getElementById('manualRemoveBtn');

    // Controls
    this.robotModel = document.getElementById('robotModel');
    this.maxIterations = document.getElementById('maxIterations');
    this.maxIterValue = document.getElementById('maxIterValue');
    this.successThreshold = document.getElementById('successThreshold');
    this.successThreshValue = document.getElementById('successThreshValue');
    this.startBtn = document.getElementById('startBtn');
    this.stopBtn = document.getElementById('stopBtn');

    // Progress
    this.progressSection = document.getElementById('progressSection');
    this.progressCounter = document.getElementById('progressCounter');
    this.mainProgressFill = document.getElementById('mainProgressFill');
    this.progressStatus = document.getElementById('progressStatus');

    // Viewer
    this.viewerPlaceholder = document.getElementById('viewerPlaceholder');
    this.cadPreviewCanvas = document.getElementById('cadPreviewCanvas');
    this.simulationFrame = document.getElementById('simulationFrame');
    this.resultGif = document.getElementById('resultGif');
    this.iterationBadge = document.getElementById('iterationBadge');
    this.statusBadge = document.getElementById('statusBadge');

    // Code
    this.codePlaceholder = document.getElementById('codePlaceholder');
    this.codeContainer = document.getElementById('codeContainer');
    this.codeOutput = document.getElementById('codeOutput');
    this.codeIterLabel = document.getElementById('codeIterLabel');
    this.copyCodeBtn = document.getElementById('copyCodeBtn');
    this.downloadCodeBtn = document.getElementById('downloadCodeBtn');

    // Logs
    this.logContainer = document.getElementById('logContainer');
    this.clearLogsBtn = document.getElementById('clearLogsBtn');
    this.filterBtns = document.querySelectorAll('.filter-btn');

    // Session badge
    this.sessionBadge = document.getElementById('sessionBadge');

    // Toast
    this.toastContainer = document.getElementById('toastContainer');
  }

  attachEventListeners() {
    // Upload zones - click
    this.cadDropZone.addEventListener('click', (e) => {
      if (!this.cadFileId && !e.target.closest('.file-remove')) {
        this.cadFileInput.click();
      }
    });
    this.manualDropZone.addEventListener('click', (e) => {
      if (!this.manualFileId && !e.target.closest('.file-remove')) {
        this.manualFileInput.click();
      }
    });

    // File inputs
    this.cadFileInput.addEventListener('change', (e) => {
      if (e.target.files[0]) this.handleFileUpload(e.target.files[0], 'cad');
    });
    this.manualFileInput.addEventListener('change', (e) => {
      if (e.target.files[0]) this.handleFileUpload(e.target.files[0], 'manual');
    });

    // Drag & drop
    this.setupDragDrop(this.cadDropZone, 'cad');
    this.setupDragDrop(this.manualDropZone, 'manual');

    // Remove buttons
    this.cadRemoveBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      this.removeFile('cad');
    });
    this.manualRemoveBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      this.removeFile('manual');
    });

    // Sliders
    this.maxIterations.addEventListener('input', () => {
      this.maxIterValue.textContent = this.maxIterations.value;
      this.maxIter = parseInt(this.maxIterations.value);
    });
    this.successThreshold.addEventListener('input', () => {
      this.successThreshValue.textContent = this.successThreshold.value;
    });

    // Buttons
    this.startBtn.addEventListener('click', () => this.startGeneration());
    this.stopBtn.addEventListener('click', () => this.stopGeneration());
    this.copyCodeBtn.addEventListener('click', () => this.copyToClipboard(this.currentCode));
    this.downloadCodeBtn.addEventListener('click', () => this.downloadCode(this.currentCode, 'grasp_code.py'));
    this.clearLogsBtn.addEventListener('click', () => this.clearLogs());

    // Log filters
    this.filterBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        this.filterBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.activeLogFilter = btn.dataset.filter;
        this.applyLogFilter();
      });
    });
  }

  init() {
    // Restore session from sessionStorage
    const saved = sessionStorage.getItem('autogrip_session');
    if (saved) {
      try {
        const data = JSON.parse(saved);
        if (data.sessionId) {
          this.sessionId = data.sessionId;
          this.sessionBadge.textContent = `Session: ${this.sessionId.slice(0, 8)}`;
          this.sessionBadge.classList.add('active');
        }
      } catch (e) {
        // Ignore
      }
    }

    this.addLog('info', 'AutoGrip-Sim Engine ready. Upload a CAD file to begin.');
    this.updateStartButton();
  }

  // --- Drag & Drop ---
  setupDragDrop(element, type) {
    const events = ['dragenter', 'dragover', 'dragleave', 'drop'];
    events.forEach(evt => {
      element.addEventListener(evt, (e) => {
        e.preventDefault();
        e.stopPropagation();
      });
    });

    element.addEventListener('dragenter', () => element.classList.add('drag-over'));
    element.addEventListener('dragover', () => element.classList.add('drag-over'));
    element.addEventListener('dragleave', () => element.classList.remove('drag-over'));
    element.addEventListener('drop', (e) => {
      element.classList.remove('drag-over');
      const files = e.dataTransfer.files;
      if (files.length > 0) {
        this.handleFileUpload(files[0], type);
      }
    });
  }

  // --- File Upload ---
  async handleFileUpload(file, type) {
    // Validate file type
    const validTypes = type === 'cad'
      ? ['.stl', '.obj', '.step', '.stp']
      : ['.pdf'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();

    if (!validTypes.includes(ext)) {
      this.showNotification(
        `Invalid file type "${ext}". Expected: ${validTypes.join(', ')}`,
        'error'
      );
      return;
    }

    const endpoint = type === 'cad' ? '/api/v1/upload/cad' : '/api/v1/upload/manual';
    const progressEl = type === 'cad' ? this.cadProgress : this.manualProgress;
    const progressFill = type === 'cad' ? this.cadProgressFill : this.manualProgressFill;
    const contentEl = type === 'cad' ? this.cadUploadContent : this.manualUploadContent;
    const infoEl = type === 'cad' ? this.cadFileInfo : this.manualFileInfo;
    const dropZone = type === 'cad' ? this.cadDropZone : this.manualDropZone;

    // Show progress
    contentEl.classList.add('hidden');
    infoEl.classList.add('hidden');
    progressEl.classList.remove('hidden');
    progressFill.style.width = '0%';

    // Simulate progress (real upload progress would use XMLHttpRequest)
    let progress = 0;
    const progressInterval = setInterval(() => {
      progress = Math.min(progress + Math.random() * 30, 90);
      progressFill.style.width = progress + '%';
    }, 200);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const res = await fetch(endpoint, { method: 'POST', body: formData });

      clearInterval(progressInterval);
      progressFill.style.width = '100%';

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Upload failed (${res.status})`);
      }

      const data = await res.json();

      // Short delay so user sees 100%
      await new Promise(r => setTimeout(r, 300));

      progressEl.classList.add('hidden');
      infoEl.classList.remove('hidden');
      dropZone.classList.add('uploaded');

      if (type === 'cad') {
        this.cadFileId = data.id;
        this.cadFileName.textContent = data.filename;
        this.cadFileSize.textContent = this.formatBytes(data.size_bytes);
        this.cadFileType.textContent = data.file_type?.toUpperCase() || ext.slice(1).toUpperCase();

        if (data.metadata) {
          this.displayCADInfo(data.metadata);
        }

        // Try to show STL preview
        if (ext === '.stl' && window.STLViewer) {
          this.showSTLPreview(file);
        }

        this.addLog('success', `CAD file uploaded: ${data.filename} (${this.formatBytes(data.size_bytes)})`);
      } else {
        this.manualFileId = data.id;
        this.manualFileName.textContent = data.filename;
        this.manualFileSize.textContent = this.formatBytes(data.size_bytes);

        if (data.metadata && data.metadata.page_count) {
          this.manualPages.classList.remove('hidden');
          this.manualPageCount.textContent = data.metadata.page_count;
        }

        this.addLog('success', `Robot manual uploaded: ${data.filename}`);
      }

      this.saveSession();
      this.updateStartButton();
      this.showNotification(`${type === 'cad' ? 'CAD file' : 'Manual'} uploaded successfully`, 'success');

    } catch (err) {
      clearInterval(progressInterval);
      progressEl.classList.add('hidden');
      contentEl.classList.remove('hidden');
      this.addLog('error', `Upload failed: ${err.message}`);
      this.showNotification(`Upload failed: ${err.message}`, 'error');
    }
  }

  displayCADInfo(metadata) {
    if (metadata.dimensions) {
      const d = metadata.dimensions;
      this.cadDimensions.classList.remove('hidden');
      this.cadDimValues.textContent = `${d.x?.toFixed(1) || '?'} x ${d.y?.toFixed(1) || '?'} x ${d.z?.toFixed(1) || '?'} mm`;
    }
  }

  removeFile(type) {
    if (this.isRunning) {
      this.showNotification('Cannot remove files while generation is running', 'warn');
      return;
    }

    if (type === 'cad') {
      this.cadFileId = null;
      this.cadUploadContent.classList.remove('hidden');
      this.cadFileInfo.classList.add('hidden');
      this.cadDropZone.classList.remove('uploaded');
      this.cadDimensions.classList.add('hidden');
      this.cadFileInput.value = '';
      this.hideCADPreview();
      this.addLog('info', 'CAD file removed');
    } else {
      this.manualFileId = null;
      this.manualUploadContent.classList.remove('hidden');
      this.manualFileInfo.classList.add('hidden');
      this.manualDropZone.classList.remove('uploaded');
      this.manualPages.classList.add('hidden');
      this.manualFileInput.value = '';
      this.addLog('info', 'Robot manual removed');
    }

    this.saveSession();
    this.updateStartButton();
  }

  showSTLPreview(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const viewer = new window.STLViewer(this.cadPreviewCanvas);
        viewer.loadFromBuffer(e.target.result);
        viewer.startAnimation();

        this.viewerPlaceholder.classList.add('hidden');
        this.cadPreviewCanvas.classList.remove('hidden');
        this.simulationFrame.classList.add('hidden');
        this.resultGif.classList.add('hidden');
      } catch (err) {
        // Silently fail - preview is optional
      }
    };
    reader.readAsArrayBuffer(file);
  }

  hideCADPreview() {
    this.cadPreviewCanvas.classList.add('hidden');
    this.viewerPlaceholder.classList.remove('hidden');
    if (window._stlViewerInstance) {
      window._stlViewerInstance.stopAnimation();
    }
  }

  // --- Generation Control ---
  async startGeneration() {
    if (!this.cadFileId) {
      this.showNotification('Please upload a CAD file first', 'warn');
      return;
    }

    this.maxIter = parseInt(this.maxIterations.value);

    const body = {
      cad_file_id: this.cadFileId,
      robot_model: this.robotModel.value,
      max_iterations: this.maxIter,
      success_threshold: parseInt(this.successThreshold.value),
    };

    // Reuse existing server-assigned session if available
    if (this.sessionId) {
      body.session_id = this.sessionId;
    }

    if (this.manualFileId) {
      body.manual_file_id = this.manualFileId;
    }

    try {
      this.addLog('info', 'Starting generation...');
      this.setRunningState(true);

      const res = await fetch('/api/v1/generate/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Start failed (${res.status})`);
      }

      const data = await res.json();
      this.sessionId = data.session_id || this.sessionId;
      this.sessionBadge.textContent = `Session: ${this.sessionId.slice(0, 8)}`;
      this.sessionBadge.classList.add('active');

      this.addLog('success', 'Generation started successfully');
      this.connectSSE(this.sessionId);
      this.saveSession();

    } catch (err) {
      this.setRunningState(false);
      this.addLog('error', `Failed to start: ${err.message}`);
      this.showNotification(`Failed to start: ${err.message}`, 'error');
    }
  }

  async stopGeneration() {
    if (!this.sessionId) return;

    try {
      this.addLog('warn', 'Stopping generation...');

      const res = await fetch(`/api/v1/generate/stop/${this.sessionId}`, {
        method: 'POST',
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Stop failed (${res.status})`);
      }

      this.disconnectSSE();
      this.setRunningState(false);
      this.addLog('warn', 'Generation stopped by user');
      this.showNotification('Generation stopped', 'info');

    } catch (err) {
      this.addLog('error', `Failed to stop: ${err.message}`);
      this.showNotification(`Failed to stop: ${err.message}`, 'error');
    }
  }

  // --- SSE Connection ---
  connectSSE(sessionId) {
    this.disconnectSSE();

    const url = `/api/v1/monitor/stream/${sessionId}`;
    this.eventSource = new EventSource(url);

    this.eventSource.addEventListener('log', (e) => {
      try {
        const data = JSON.parse(e.data);
        const level = (data.level || 'info').toLowerCase().replace('warning', 'warn');
        this.addLog(level, data.message || e.data);
      } catch {
        this.addLog('info', e.data);
      }
    });

    this.eventSource.addEventListener('iteration_start', (e) => {
      try {
        const data = JSON.parse(e.data);
        this.currentIteration = data.iteration || this.currentIteration + 1;
        this.updateProgress(this.currentIteration, this.maxIter, 'running');
        this.addIterationSeparator(this.currentIteration);
        this.addLog('info', `Iteration ${this.currentIteration} started`);
      } catch {
        this.currentIteration++;
        this.updateProgress(this.currentIteration, this.maxIter, 'running');
      }
    });

    this.eventSource.addEventListener('iteration_result', (e) => {
      try {
        const data = JSON.parse(e.data);
        const success = data.success;
        const score = data.score;
        if (success) {
          this.addLog('success', `Iteration ${this.currentIteration} succeeded (score: ${score})`);
        } else {
          this.addLog('warn', `Iteration ${this.currentIteration} failed - ${data.error || 'retrying...'}`);
        }
      } catch {
        // ignore
      }
    });

    this.eventSource.addEventListener('code_update', (e) => {
      try {
        const data = JSON.parse(e.data);
        this.updateCode(data.code || e.data);
      } catch {
        this.updateCode(e.data);
      }
    });

    this.eventSource.addEventListener('frame', (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.image) {
          this.updateViewer(data.image);
        } else if (data.url) {
          this.updateViewerUrl(data.url);
        }
      } catch {
        // Try treating data as base64 image directly
        if (e.data && e.data.length > 100) {
          this.updateViewer(e.data);
        }
      }
    });

    this.eventSource.addEventListener('complete', (e) => {
      try {
        const data = JSON.parse(e.data);
        const status = data.status || 'success';
        if (status === 'success') {
          this.addLog('success', 'Generation completed successfully!');
          this.updateProgress(this.currentIteration, this.maxIter, 'complete');
        } else if (status === 'stopped') {
          this.addLog('warn', 'Generation stopped by user');
          this.updateProgress(this.currentIteration, this.maxIter, 'error');
        } else {
          this.addLog('error', `Generation ended with status: ${status}`);
          this.updateProgress(this.currentIteration, this.maxIter, 'error');
        }
        this.showResult(data.gif_url, data.code);
      } catch {
        this.addLog('info', 'Generation completed');
        this.fetchResults();
        this.updateProgress(this.currentIteration, this.maxIter, 'complete');
      }
      this.disconnectSSE();
      this.setRunningState(false);
    });

    this.eventSource.addEventListener('error', (e) => {
      if (this.eventSource.readyState === EventSource.CLOSED) {
        this.addLog('error', 'Connection to server lost');
        this.disconnectSSE();
        this.setRunningState(false);
        this.updateProgress(this.currentIteration, this.maxIter, 'error');
      }
    });

    // Generic message handler
    this.eventSource.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.type) {
          // Dispatch based on type if not handled by named events
          switch (data.type) {
            case 'log':
              this.addLog(data.level || 'info', data.message);
              break;
            case 'code_update':
              this.updateCode(data.code);
              break;
            case 'frame':
              if (data.image) this.updateViewer(data.image);
              break;
            case 'complete':
              this.addLog('success', 'Generation completed!');
              this.showResult(data.gif_url, data.code);
              this.disconnectSSE();
              this.setRunningState(false);
              break;
          }
        }
      } catch {
        // Plain text message
        this.addLog('info', e.data);
      }
    };
  }

  disconnectSSE() {
    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
    }
  }

  async fetchResults() {
    if (!this.sessionId) return;
    try {
      // Fetch final code
      const codeRes = await fetch(`/api/v1/generate/code/${this.sessionId}`);
      if (codeRes.ok) {
        const data = await codeRes.json();
        if (data.code) this.updateCode(data.code);
      }
      // Fetch GIF
      const gifUrl = `/api/v1/monitor/result/${this.sessionId}/gif`;
      this.showResult(gifUrl, this.currentCode);
    } catch {
      // Ignore
    }
  }

  // --- UI State ---
  setRunningState(running) {
    this.isRunning = running;

    this.startBtn.classList.toggle('hidden', running);
    this.stopBtn.classList.toggle('hidden', !running);
    this.progressSection.classList.toggle('hidden', !running);

    // Disable controls while running
    this.cadDropZone.style.pointerEvents = running ? 'none' : '';
    this.manualDropZone.style.pointerEvents = running ? 'none' : '';
    this.robotModel.disabled = running;
    this.maxIterations.disabled = running;
    this.successThreshold.disabled = running;

    // Status badge
    if (running) {
      this.statusBadge.textContent = 'Running';
      this.statusBadge.className = 'status-badge running';
    } else {
      this.statusBadge.textContent = 'Idle';
      this.statusBadge.className = 'status-badge';
    }

    if (running) {
      this.currentIteration = 0;
      this.updateProgress(0, this.maxIter, 'initializing');
    }
  }

  updateStartButton() {
    this.startBtn.disabled = !this.cadFileId;
  }

  updateProgress(iteration, maxIter, status) {
    this.progressSection.classList.remove('hidden');
    this.currentIteration = iteration;
    const pct = maxIter > 0 ? Math.round((iteration / maxIter) * 100) : 0;

    this.progressCounter.textContent = `${iteration} / ${maxIter}`;
    this.mainProgressFill.style.width = pct + '%';

    this.iterationBadge.classList.remove('hidden');
    this.iterationBadge.textContent = `Iteration ${iteration}`;

    const statusMap = {
      initializing: 'Initializing pipeline...',
      running: `Running iteration ${iteration}...`,
      complete: 'Generation complete!',
      error: 'Error occurred',
    };
    this.progressStatus.textContent = statusMap[status] || status;

    if (status === 'complete') {
      this.statusBadge.textContent = 'Complete';
      this.statusBadge.className = 'status-badge success';
      this.mainProgressFill.style.width = '100%';
    } else if (status === 'error') {
      this.statusBadge.textContent = 'Error';
      this.statusBadge.className = 'status-badge error';
    }
  }

  // --- Code Display ---
  updateCode(code) {
    if (!code) return;

    this.previousCode = this.currentCode;
    this.currentCode = code;

    this.codePlaceholder.classList.add('hidden');
    this.codeContainer.classList.remove('hidden');
    this.codeIterLabel.textContent = `Iteration ${this.currentIteration || 1}`;

    // Enable buttons
    this.copyCodeBtn.disabled = false;
    this.downloadCodeBtn.disabled = false;

    // Render with syntax highlighting and diff
    this.renderCode(code);
  }

  renderCode(code) {
    const lines = code.split('\n');
    const prevLines = this.previousCode ? this.previousCode.split('\n') : [];

    const html = lines.map((line, i) => {
      const lineNum = i + 1;
      const highlighted = this.highlightSyntax(this.escapeHtml(line));
      let lineClass = 'code-line';

      if (prevLines.length > 0) {
        if (i >= prevLines.length) {
          lineClass += ' added';
        } else if (prevLines[i] !== line) {
          lineClass += ' changed';
        }
      }

      return `<span class="${lineClass}" data-ln="${lineNum}">${highlighted}</span>`;
    }).join('\n');

    this.codeOutput.innerHTML = html;
  }

  highlightSyntax(code) {
    // Order matters: comments first to avoid highlighting inside them

    // Comments
    code = code.replace(/(#.*)$/gm, '<span class="cmt">$1</span>');

    // Triple-quoted strings
    code = code.replace(/("""[\s\S]*?"""|'''[\s\S]*?''')/g, '<span class="str">$1</span>');

    // Strings (double and single quoted)
    code = code.replace(/(f?"(?:[^"\\]|\\.)*"|f?'(?:[^'\\]|\\.)*')/g, '<span class="str">$1</span>');

    // Decorators
    code = code.replace(/(@\w+)/g, '<span class="dec">$1</span>');

    // Keywords
    const kwPattern = /\b(def|class|return|if|elif|else|for|while|try|except|finally|with|as|import|from|raise|yield|pass|break|continue|and|or|not|in|is|lambda|async|await|global|nonlocal|assert|del)\b/g;
    code = code.replace(kwPattern, '<span class="kw">$1</span>');

    // Built-in constants and values
    code = code.replace(/\b(True|False|None|self)\b/g, '<span class="self">$1</span>');

    // Built-in functions
    const builtinPattern = /\b(print|len|range|int|float|str|list|dict|tuple|set|type|isinstance|hasattr|getattr|setattr|enumerate|zip|map|filter|sorted|reversed|abs|min|max|sum|round|open|super|property|staticmethod|classmethod)\b/g;
    code = code.replace(builtinPattern, '<span class="builtin">$1</span>');

    // Numbers
    code = code.replace(/\b(\d+\.?\d*(?:e[+-]?\d+)?)\b/g, '<span class="num">$1</span>');

    // Function definitions
    code = code.replace(/\b(def)\b(\s+)(\w+)/g, '<span class="kw">$1</span>$2<span class="fn">$3</span>');

    // Class definitions
    code = code.replace(/\b(class)\b(\s+)(\w+)/g, '<span class="kw">$1</span>$2<span class="cls">$3</span>');

    return code;
  }

  escapeHtml(text) {
    const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' };
    return text.replace(/[&<>"']/g, c => map[c]);
  }

  // --- Viewer ---
  updateViewer(base64Image) {
    this.viewerPlaceholder.classList.add('hidden');
    this.cadPreviewCanvas.classList.add('hidden');
    this.resultGif.classList.add('hidden');
    this.simulationFrame.classList.remove('hidden');

    // Handle data URL or raw base64
    if (base64Image.startsWith('data:')) {
      this.simulationFrame.src = base64Image;
    } else {
      this.simulationFrame.src = `data:image/jpeg;base64,${base64Image}`;
    }
  }

  updateViewerUrl(url) {
    this.viewerPlaceholder.classList.add('hidden');
    this.cadPreviewCanvas.classList.add('hidden');
    this.resultGif.classList.add('hidden');
    this.simulationFrame.classList.remove('hidden');
    this.simulationFrame.src = url;
  }

  showResult(gifUrl, code) {
    if (gifUrl) {
      this.viewerPlaceholder.classList.add('hidden');
      this.cadPreviewCanvas.classList.add('hidden');
      this.simulationFrame.classList.add('hidden');
      this.resultGif.classList.remove('hidden');
      this.resultGif.src = gifUrl;
    }

    if (code) {
      this.updateCode(code);
    }

    this.showNotification('Generation completed successfully!', 'success');
  }

  // --- Logs ---
  addLog(level, message) {
    const entry = document.createElement('div');
    entry.className = `log-entry log-${level}`;
    entry.dataset.level = level;

    const time = this.formatTimestamp();
    const levelLabel = `[${level.toUpperCase()}]`;

    entry.innerHTML = `
      <span class="log-time">${time}</span>
      <span class="log-level">${levelLabel}</span>
      <span class="log-message">${this.escapeHtml(message)}</span>
    `;

    // Apply current filter
    if (this.activeLogFilter !== 'all' && level !== this.activeLogFilter) {
      entry.classList.add('hidden');
    }

    this.logContainer.appendChild(entry);
    this.scrollLogsToBottom();
  }

  addIterationSeparator(iteration) {
    const sep = document.createElement('div');
    sep.className = 'log-separator';
    sep.dataset.level = 'all';
    sep.textContent = `Iteration ${iteration}`;
    this.logContainer.appendChild(sep);
    this.scrollLogsToBottom();
  }

  clearLogs() {
    this.logContainer.innerHTML = '';
    this.addLog('info', 'Logs cleared');
  }

  applyLogFilter() {
    const entries = this.logContainer.querySelectorAll('.log-entry, .log-separator');
    entries.forEach(entry => {
      if (this.activeLogFilter === 'all') {
        entry.classList.remove('hidden');
      } else {
        const level = entry.dataset.level;
        entry.classList.toggle('hidden', level !== this.activeLogFilter && level !== 'all');
      }
    });
  }

  scrollLogsToBottom() {
    requestAnimationFrame(() => {
      this.logContainer.scrollTop = this.logContainer.scrollHeight;
    });
  }

  // --- Notifications ---
  showNotification(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    this.toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.classList.add('toast-out');
      toast.addEventListener('animationend', () => toast.remove());
    }, 3500);
  }

  // --- Utility ---
  formatBytes(bytes) {
    if (!bytes || bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }

  formatTimestamp() {
    const now = new Date();
    return [now.getHours(), now.getMinutes(), now.getSeconds()]
      .map(n => String(n).padStart(2, '0'))
      .join(':');
  }

  generateSessionId() {
    return 'ses_' + Date.now().toString(36) + '_' + Math.random().toString(36).slice(2, 8);
  }

  async copyToClipboard(text) {
    try {
      await navigator.clipboard.writeText(text);
      this.showNotification('Code copied to clipboard', 'success');
    } catch {
      // Fallback
      const textarea = document.createElement('textarea');
      textarea.value = text;
      textarea.style.position = 'fixed';
      textarea.style.opacity = '0';
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      document.body.removeChild(textarea);
      this.showNotification('Code copied to clipboard', 'success');
    }
  }

  downloadCode(code, filename) {
    if (!code) return;
    const blob = new Blob([code], { type: 'text/x-python' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    this.showNotification(`Downloaded ${filename}`, 'success');
  }

  saveSession() {
    sessionStorage.setItem('autogrip_session', JSON.stringify({
      sessionId: this.sessionId,
      cadFileId: this.cadFileId,
      manualFileId: this.manualFileId,
    }));
  }
}

// --- Initialize ---
document.addEventListener('DOMContentLoaded', () => {
  window.app = new AutoGripApp();
});
