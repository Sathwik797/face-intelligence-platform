/**
 * Main Dashboard Application Controller.
 * Orchestrates camera streaming, frame capture loop, API communication, and UI rendering.
 */
document.addEventListener('DOMContentLoaded', () => {
  const api = window.apiClient;
  const state = window.dashboardState;
  const camera = new window.CameraManager(640, 480);

  // DOM Elements
  const videoElem = document.getElementById('camera-video');
  const overlayCanvas = document.getElementById('camera-overlay');
  const cameraStage = document.getElementById('camera-stage');
  const placeholder = document.getElementById('camera-placeholder');

  const btnStart = document.getElementById('btn-start');
  const btnStop = document.getElementById('btn-stop');
  const btnReset = document.getElementById('btn-reset');

  const bannerError = document.getElementById('banner-error');
  const bannerErrorMsg = document.getElementById('banner-error-msg');

  const overlay = new window.OverlayRenderer(overlayCanvas);

  // Loop & FPS tracking
  let loopIntervalId = null;
  let presenceActiveIntervalId = null;
  let presenceHistoryIntervalId = null;

  let lastFrameTime = performance.now();
  let frameCountForFps = 0;
  let fpsTimer = performance.now();

  // 1. Initialize Dashboard
  async function init() {
    bindEvents();
    state.subscribe(renderUI);

    try {
      const health = await api.health();
      const status = await api.runtimeStatus();
      state.update({
        health,
        runtimeStatus: status.status,
        frameCounter: status.frame_counter
      });
      state.clearError();
    } catch (err) {
      state.setError(`Backend unavailable: ${err.message}`);
    }

    // Start background presence synchronization
    startPresencePolling();
  }

  // 2. Event Listeners
  function bindEvents() {
    btnStart.addEventListener('click', startStreaming);
    btnStop.addEventListener('click', stopStreaming);
    btnReset.addEventListener('click', resetState);

    window.addEventListener('resize', () => {
      if (camera.isActive) {
        overlay.resize(cameraStage.getBoundingClientRect());
      }
    });
  }

  // 3. Streaming Controls
  async function startStreaming() {
    try {
      state.clearError();
      btnStart.disabled = true;

      await api.runtimeStart();
      await camera.start(videoElem);

      state.update({
        isStreaming: true,
        runtimeStatus: 'RUNNING'
      });

      placeholder.style.display = 'none';
      videoElem.style.display = 'block';

      // Start frame capture loop (~10 FPS / 100ms interval)
      startFrameLoop(100);
    } catch (err) {
      camera.stop();
      state.setError(`Camera or runtime start failed: ${err.message}`);
      btnStart.disabled = false;
    }
  }

  async function stopStreaming() {
    stopFrameLoop();
    camera.stop();
    overlay.clear();

    try {
      await api.runtimeStop('user_stopped');
    } catch (err) {
      console.warn('Error during runtime stop:', err);
    }

    state.update({
      isStreaming: false,
      runtimeStatus: 'STOPPED'
    });

    placeholder.style.display = 'flex';
    videoElem.style.display = 'none';
  }

  async function resetState() {
    try {
      await api.runtimeReset();
      overlay.clear();
      state.update({
        latestResult: null,
        activeSessions: [],
        eventsLog: [],
        frameCounter: 0,
        runtimeStatus: 'STOPPED'
      });
    } catch (err) {
      state.setError(`Reset failed: ${err.message}`);
    }
  }

  // 4. Flight-Controlled Frame Processing Loop
  function startFrameLoop(intervalMs = 100) {
    if (loopIntervalId) clearInterval(loopIntervalId);

    loopIntervalId = setInterval(async () => {
      const currentState = state.getState();
      if (!currentState.isStreaming || currentState.isProcessing) {
        // Skip frame tick if a previous request is still in flight
        return;
      }

      state.update({ isProcessing: true });
      const b64 = camera.captureFrame(0.85);

      if (!b64) {
        state.update({ isProcessing: false });
        return;
      }

      const t0 = performance.now();
      try {
        const result = await api.processFrame(b64);
        const rtt = Math.round(performance.now() - t0);

        // Update FPS calculations
        frameCountForFps++;
        const now = performance.now();
        let currentFps = currentState.telemetry.fps;
        if (now - fpsTimer >= 1000) {
          currentFps = Math.round((frameCountForFps * 1000) / (now - fpsTimer));
          frameCountForFps = 0;
          fpsTimer = now;
        }

        // Extract latencies
        const lats = result.latencies || {};

        state.update({
          latestResult: result,
          frameCounter: result.frame_index || currentState.frameCounter + 1,
          activeSessions: result.active_sessions || [],
          telemetry: {
            fps: currentFps,
            clientRttMs: rtt,
            recognitionMs: lats.recognition_ms || 0,
            temporalMs: lats.temporal_ms || 0,
            presenceMs: lats.presence_ms || 0,
            totalMs: lats.total_ms || 0
          }
        });

        // Append any new presence events to chronological stream
        if (result.presence_events && result.presence_events.length > 0) {
          result.presence_events.forEach(ev => state.appendEvent(ev));
        }

        // Render face bounding box overlay
        const displayRect = cameraStage.getBoundingClientRect();
        const srcDims = camera.getDimensions();
        overlay.render(result.recognition, result.temporal, srcDims, displayRect);

      } catch (err) {
        console.warn('Frame processing dropped:', err.message);
      } finally {
        state.update({ isProcessing: false });
      }
    }, intervalMs);
  }

  function stopFrameLoop() {
    if (loopIntervalId) {
      clearInterval(loopIntervalId);
      loopIntervalId = null;
    }
  }

  // 5. Background Presence Synchronization
  function startPresencePolling() {
    presenceActiveIntervalId = setInterval(async () => {
      try {
        const data = await api.presenceActive();
        if (data && data.sessions) {
          state.update({ activeSessions: data.sessions });
        }
      } catch (err) {
        // Silent failure on background polling
      }
    }, 3000);

    presenceHistoryIntervalId = setInterval(async () => {
      try {
        const data = await api.presenceHistory();
        if (data && data.sessions) {
          state.update({ historySessions: data.sessions });
        }
      } catch (err) {
        // Silent failure on background polling
      }
    }, 5000);
  }

  // 6. UI Render Function
  function renderUI(s) {
    // Controls State
    btnStart.disabled = s.isStreaming;
    btnStop.disabled = !s.isStreaming;

    // Status Banner
    if (s.error) {
      bannerError.style.display = 'flex';
      bannerErrorMsg.textContent = s.error;
    } else {
      bannerError.style.display = 'none';
    }

    // Header Badges
    const statusPill = document.getElementById('badge-runtime-status');
    statusPill.textContent = s.runtimeStatus;
    statusPill.className = `badge ${s.runtimeStatus === 'RUNNING' ? 'badge-success' : 'badge-neutral'}`;

    document.getElementById('stat-fps').textContent = `${s.telemetry.fps} FPS`;
    document.getElementById('stat-rtt').textContent = `${s.telemetry.clientRttMs} ms`;
    document.getElementById('stat-frames').textContent = s.frameCounter;

    // Recognition Intelligence
    const rec = s.latestResult ? s.latestResult.recognition : null;
    if (rec) {
      document.getElementById('rec-identity').textContent = rec.identity || 'Unknown';
      document.getElementById('rec-decision').textContent = rec.recognized ? 'RECOGNIZED' : (rec.reason || 'UNCONFIRMED');
      document.getElementById('rec-decision').className = rec.recognized ? 'badge badge-success' : 'badge badge-danger';

      const simVal = rec.similarity !== null ? rec.similarity : 0;
      document.getElementById('rec-similarity').textContent = rec.similarity >= 0 ? rec.similarity.toFixed(4) : '--';
      document.getElementById('rec-threshold').textContent = rec.threshold ? rec.threshold.toFixed(4) : '0.2400';

      const pct = Math.min(100, Math.max(0, Math.round(simVal * 100)));
      const fillElem = document.getElementById('rec-similarity-fill');
      fillElem.style.width = `${pct}%`;
      fillElem.style.background = rec.recognized ? 'var(--success)' : 'var(--danger)';

      document.getElementById('rec-quality-status').textContent = (rec.quality_status || 'NONE').toUpperCase();
    } else {
      document.getElementById('rec-identity').textContent = '--';
      document.getElementById('rec-decision').textContent = 'STANDBY';
      document.getElementById('rec-decision').className = 'badge badge-neutral';
      document.getElementById('rec-similarity').textContent = '--';
      document.getElementById('rec-similarity-fill').style.width = '0%';
      document.getElementById('rec-quality-status').textContent = '--';
    }

    // Temporal Stability
    const temp = s.latestResult ? s.latestResult.temporal : null;
    if (temp) {
      document.getElementById('temp-identity').textContent = temp.stable_identity || '--';
      document.getElementById('temp-state').textContent = (temp.state || 'UNKNOWN').toUpperCase();
      document.getElementById('temp-state').className = temp.is_stable ? 'badge badge-success' : 'badge badge-warning';
      document.getElementById('temp-confidence').textContent = `${Math.round(temp.confidence_score * 100)}%`;
      document.getElementById('temp-obs-count').textContent = `${temp.observations_count} in window`;
    } else {
      document.getElementById('temp-identity').textContent = '--';
      document.getElementById('temp-state').textContent = 'STANDBY';
      document.getElementById('temp-state').className = 'badge badge-neutral';
      document.getElementById('temp-confidence').textContent = '--';
      document.getElementById('temp-obs-count').textContent = '--';
    }

    // Presence & Session
    const activeSessions = s.activeSessions || [];
    if (activeSessions.length > 0) {
      const mainSession = activeSessions[0];
      document.getElementById('pres-state').textContent = (mainSession.state || 'PRESENT').toUpperCase();
      document.getElementById('pres-state').className = mainSession.state === 'PRESENT' ? 'badge badge-success' : 'badge badge-warning';
      document.getElementById('pres-identity').textContent = mainSession.identity;
      document.getElementById('pres-session-id').textContent = `#${mainSession.session_id.substring(0, 8)}`;
      document.getElementById('pres-duration').textContent = `${mainSession.duration_seconds.toFixed(1)}s`;
      document.getElementById('pres-obs').textContent = mainSession.observation_count;
      document.getElementById('pres-interruptions').textContent = mainSession.interruption_count;
    } else {
      document.getElementById('pres-state').textContent = 'NOT_PRESENT';
      document.getElementById('pres-state').className = 'badge badge-neutral';
      document.getElementById('pres-identity').textContent = '--';
      document.getElementById('pres-session-id').textContent = '--';
      document.getElementById('pres-duration').textContent = '--';
      document.getElementById('pres-obs').textContent = '--';
      document.getElementById('pres-interruptions').textContent = '--';
    }

    // Stage Latencies Telemetry
    const t = s.telemetry;
    document.getElementById('lat-rec').textContent = `${t.recognitionMs.toFixed(1)} ms`;
    document.getElementById('lat-temp').textContent = `${t.temporalMs.toFixed(1)} ms`;
    document.getElementById('lat-pres').textContent = `${t.presenceMs.toFixed(1)} ms`;
    document.getElementById('lat-total').textContent = `${t.totalMs.toFixed(1)} ms`;

    // Activity Stream Table
    const tbody = document.getElementById('activity-table-body');
    if (s.eventsLog && s.eventsLog.length > 0) {
      tbody.innerHTML = s.eventsLog.slice(0, 15).map(ev => {
        const timeStr = (ev.timestamp || '').split('T')[1]?.split('.')[0] || ev.timestamp;
        let badgeClass = 'badge-neutral';
        if (ev.event_type.includes('CONFIRMED')) badgeClass = 'badge-success';
        if (ev.event_type.includes('GRACE')) badgeClass = 'badge-warning';
        if (ev.event_type.includes('ENDED')) badgeClass = 'badge-danger';

        return `
          <tr>
            <td>${timeStr}</td>
            <td><strong>${ev.identity || 'Unknown'}</strong></td>
            <td><span class="badge ${badgeClass}">${ev.event_type}</span></td>
            <td>${ev.previous_state} &rarr; ${ev.new_state}</td>
          </tr>
        `;
      }).join('');
    } else {
      tbody.innerHTML = `<tr><td colspan="4" style="text-align: center; color: var(--text-muted);">No presence events recorded yet.</td></tr>`;
    }
  }

  // Run initialization
  init();
});
