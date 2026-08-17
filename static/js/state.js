/**
 * Centralized Reactive State Store for the Face Intelligence Dashboard.
 */
class DashboardState {
  constructor() {
    this.state = {
      isStreaming: false,
      isProcessing: false,
      health: { status: 'unknown', models_ready: false },
      runtimeStatus: 'STOPPED',
      frameCounter: 0,
      latestResult: null,
      activeSessions: [],
      historySessions: [],
      eventsLog: [],
      telemetry: {
        fps: 0,
        clientRttMs: 0,
        recognitionMs: 0,
        temporalMs: 0,
        presenceMs: 0,
        totalMs: 0
      },
      error: null
    };

    this.listeners = [];
  }

  getState() {
    return this.state;
  }

  subscribe(listener) {
    this.listeners.push(listener);
    // Initial call
    listener(this.state);
    return () => {
      this.listeners = this.listeners.filter(l => l !== listener);
    };
  }

  update(partialState) {
    this.state = { ...this.state, ...partialState };
    this.listeners.forEach(l => l(this.state));
  }

  appendEvent(event) {
    const eventsLog = [event, ...this.state.eventsLog].slice(0, 50); // Keep latest 50 events
    this.update({ eventsLog });
  }

  setError(errorMessage) {
    this.update({ error: errorMessage });
  }

  clearError() {
    this.update({ error: null });
  }
}

window.dashboardState = new DashboardState();
