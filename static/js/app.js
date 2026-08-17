/**
 * Main Dashboard Application Controller.
 * Orchestrates camera streaming, frame capture loop, API communication, tab switching, and UI rendering.
 */

document.addEventListener('DOMContentLoaded', () => {
  const api = window.apiClient;
  const state = window.dashboardState;
  const camera = new window.CameraManager(640, 480);
  window.cameraManager = camera;

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

  // Tab Navigation Elements
  const tabBtnLive = document.getElementById('tab-btn-live');
  const tabBtnAttendance = document.getElementById('tab-btn-attendance');
  const tabBtnIdentities = document.getElementById('tab-btn-identities');

  const viewLive = document.getElementById('view-live');
  const viewAttendance = document.getElementById('view-attendance');
  const viewIdentities = document.getElementById('view-identities');

  const overlay = new window.OverlayRenderer(overlayCanvas);

  // Loop & FPS tracking
  let loopIntervalId = null;
  let presenceActiveIntervalId = null;
  let presenceHistoryIntervalId = null;

  let frameCountForFps = 0;
  let fpsTimer = performance.now();

  // 1. Initialize Dashboard
  async function init() {
    bindEvents();
    state.subscribe(renderUI);

    // Initialize submodule controllers
    if (window.attendanceView) window.attendanceView.init();
    if (window.identitiesView) window.identitiesView.init();

    try {
      const health = await api.checkHealth();
      const status = await api.getRuntimeStatus();
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

  // 2. Event Listeners & Tab Switching
  function switchTab(activeTab) {
    [tabBtnLive, tabBtnAttendance, tabBtnIdentities].forEach(b => b?.classList.remove('active'));
    [viewLive, viewAttendance, viewIdentities].forEach(v => { if (v) v.style.display = 'none'; });

    if (activeTab === 'live') {
      tabBtnLive?.classList.add('active');
      if (viewLive) viewLive.style.display = 'block';
    } else if (activeTab === 'attendance') {
      tabBtnAttendance?.classList.add('active');
      if (viewAttendance) viewAttendance.style.display = 'block';
      if (window.attendanceView) window.attendanceView.loadData();
    } else if (activeTab === 'identities') {
      tabBtnIdentities?.classList.add('active');
      if (viewIdentities) viewIdentities.style.display = 'block';
      if (window.identitiesView) window.identitiesView.loadData();
    }
  }

  function bindEvents() {
    if (tabBtnLive) tabBtnLive.addEventListener('click', () => switchTab('live'));
    if (tabBtnAttendance) tabBtnAttendance.addEventListener('click', () => switchTab('attendance'));
    if (tabBtnIdentities) tabBtnIdentities.addEventListener('click', () => switchTab('identities'));

    if (btnStart) btnStart.addEventListener('click', startStreaming);
    if (btnStop) btnStop.addEventListener('click', stopStreaming);
    if (btnReset) btnReset.addEventListener('click', resetState);

    window.addEventListener('resize', () => {
      if (camera.isActive && cameraStage) {
        overlay.resize(cameraStage.getBoundingClientRect());
      }
    });
  }

  // 3. Streaming Controls
  async function startStreaming() {
    try {
      state.clearError();
      if (btnStart) btnStart.disabled = true;

      await api.startRuntime();
      await camera.start(videoElem);

      state.update({
        isStreaming: true,
        runtimeStatus: 'RUNNING'
      });

      if (placeholder) placeholder.style.display = 'none';
      if (videoElem) videoElem.style.display = 'block';

      // Start frame capture loop (~10 FPS / 100ms interval)
      startFrameLoop(100);
    } catch (err) {
      camera.stop();
      state.setError(`Camera or runtime start failed: ${err.message}`);
      if (btnStart) btnStart.disabled = false;
    }
  }

  async function stopStreaming() {
    stopFrameLoop();
    camera.stop();
    overlay.clear();

    try {
      await api.stopRuntime();
    } catch (err) {
      console.warn('Error during runtime stop:', err);
    }

    state.update({
      isStreaming: false,
      runtimeStatus: 'STOPPED'
    });

    if (placeholder) placeholder.style.display = 'flex';
    if (videoElem) videoElem.style.display = 'none';
  }

  async function resetState() {
    try {
      await api.resetRuntime();
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
        if (cameraStage) {
          const displayRect = cameraStage.getBoundingClientRect();
          const srcDims = camera.getDimensions();
          overlay.render(result.recognition, result.temporal, srcDims, displayRect);
        }

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
        const data = await api.getActivePresence();
        if (data && data.sessions) {
          state.update({ activeSessions: data.sessions });
        }
      } catch (err) {
        // Silent failure on background polling
      }
    }, 3000);

    presenceHistoryIntervalId = setInterval(async () => {
      try {
        const data = await api.getPresenceHistory();
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
    if (btnStart) btnStart.disabled = s.isStreaming;
    if (btnStop) btnStop.disabled = !s.isStreaming;

    // Status Banner
    if (bannerError) {
      if (s.error) {
        bannerError.style.display = 'flex';
        if (bannerErrorMsg) bannerErrorMsg.textContent = s.error;
      } else {
        bannerError.style.display = 'none';
      }
    }

    // Header Badges
    const statusPill = document.getElementById('badge-runtime-status');
    if (statusPill) {
      statusPill.textContent = s.runtimeStatus;
      statusPill.className = `badge ${s.runtimeStatus === 'RUNNING' ? 'badge-success' : 'badge-neutral'}`;
    }

    const statFps = document.getElementById('stat-fps');
    if (statFps) statFps.textContent = `${s.telemetry.fps} FPS`;

    const statRtt = document.getElementById('stat-rtt');
    if (statRtt) statRtt.textContent = `${s.telemetry.clientRttMs} ms`;

    const statFrames = document.getElementById('stat-frames');
    if (statFrames) statFrames.textContent = s.frameCounter;

    // Recognition Intelligence
    const rec = s.latestResult ? s.latestResult.recognition : null;
    const recIdElem = document.getElementById('rec-identity');
    const recDecElem = document.getElementById('rec-decision');
    const recSimElem = document.getElementById('rec-similarity');
    const recFillElem = document.getElementById('rec-similarity-fill');
    const recQualElem = document.getElementById('rec-quality-status');

    if (rec) {
      if (recIdElem) recIdElem.textContent = rec.identity || 'Unknown';
      if (recDecElem) {
        recDecElem.textContent = rec.recognized ? 'RECOGNIZED' : (rec.reason || 'UNCONFIRMED');
        recDecElem.className = rec.recognized ? 'badge badge-success' : 'badge badge-danger';
      }

      const simVal = rec.similarity !== null ? rec.similarity : 0;
      if (recSimElem) recSimElem.textContent = rec.similarity >= 0 ? rec.similarity.toFixed(4) : '--';

      const pct = Math.min(100, Math.max(0, Math.round(simVal * 100)));
      if (recFillElem) {
        recFillElem.style.width = `${pct}%`;
        recFillElem.style.background = rec.recognized ? 'var(--success)' : 'var(--danger)';
      }

      if (recQualElem) recQualElem.textContent = (rec.quality_status || 'NONE').toUpperCase();
    } else {
      if (recIdElem) recIdElem.textContent = '--';
      if (recDecElem) {
        recDecElem.textContent = 'STANDBY';
        recDecElem.className = 'badge badge-neutral';
      }
      if (recSimElem) recSimElem.textContent = '--';
      if (recFillElem) recFillElem.style.width = '0%';
      if (recQualElem) recQualElem.textContent = '--';
    }

    // Temporal Stability
    const temp = s.latestResult ? s.latestResult.temporal : null;
    const tempIdElem = document.getElementById('temp-identity');
    const tempStateElem = document.getElementById('temp-state');
    const tempConfElem = document.getElementById('temp-confidence');
    const tempObsElem = document.getElementById('temp-obs-count');

    if (temp) {
      if (tempIdElem) tempIdElem.textContent = temp.stable_identity || '--';
      if (tempStateElem) {
        tempStateElem.textContent = (temp.state || 'UNKNOWN').toUpperCase();
        tempStateElem.className = temp.is_stable ? 'badge badge-success' : 'badge badge-warning';
      }
      if (tempConfElem) tempConfElem.textContent = `${Math.round(temp.confidence_score * 100)}%`;
      if (tempObsElem) tempObsElem.textContent = `${temp.observations_count} in window`;
    } else {
      if (tempIdElem) tempIdElem.textContent = '--';
      if (tempStateElem) {
        tempStateElem.textContent = 'STANDBY';
        tempStateElem.className = 'badge badge-neutral';
      }
      if (tempConfElem) tempConfElem.textContent = '--';
      if (tempObsElem) tempObsElem.textContent = '--';
    }

    // Presence & Session
    const activeSessions = s.activeSessions || [];
    const presStateElem = document.getElementById('pres-state');
    const presIdElem = document.getElementById('pres-identity');
    const presSessElem = document.getElementById('pres-session-id');
    const presDurElem = document.getElementById('pres-duration');
    const presInterElem = document.getElementById('pres-interruptions');

    if (activeSessions.length > 0) {
      const mainSession = activeSessions[0];
      if (presStateElem) {
        presStateElem.textContent = (mainSession.state || 'PRESENT').toUpperCase();
        presStateElem.className = mainSession.state === 'PRESENT' ? 'badge badge-success' : 'badge badge-warning';
      }
      if (presIdElem) presIdElem.textContent = mainSession.identity;
      if (presSessElem) presSessElem.textContent = `#${mainSession.session_id.substring(0, 8)}`;
      if (presDurElem) presDurElem.textContent = `${mainSession.duration_seconds.toFixed(1)}s`;
      if (presInterElem) presInterElem.textContent = mainSession.interruption_count;
    } else {
      if (presStateElem) {
        presStateElem.textContent = 'NOT_PRESENT';
        presStateElem.className = 'badge badge-neutral';
      }
      if (presIdElem) presIdElem.textContent = '--';
      if (presSessElem) presSessElem.textContent = '--';
      if (presDurElem) presDurElem.textContent = '--';
      if (presInterElem) presInterElem.textContent = '--';
    }

    // Stage Latencies Telemetry
    const t = s.telemetry;
    const latRec = document.getElementById('lat-rec');
    const latTemp = document.getElementById('lat-temp');
    const latPres = document.getElementById('lat-pres');
    const latTotal = document.getElementById('lat-total');

    if (latRec) latRec.textContent = `${t.recognitionMs.toFixed(1)} ms`;
    if (latTemp) latTemp.textContent = `${t.temporalMs.toFixed(1)} ms`;
    if (latPres) latPres.textContent = `${t.presenceMs.toFixed(1)} ms`;
    if (latTotal) latTotal.textContent = `${t.totalMs.toFixed(1)} ms`;

    // Activity Stream Table
    const tbody = document.getElementById('activity-table-body');
    if (tbody) {
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
  }

  // Run initialization
  init();
});
