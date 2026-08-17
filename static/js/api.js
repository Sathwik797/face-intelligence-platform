/**
 * API Client module encapsulating all REST communications with the Phase 12 Flask backend.
 */
class ApiClient {
  constructor(baseUrl = '/api/v1', timeoutMs = 8000) {
    this.baseUrl = baseUrl;
    this.timeoutMs = timeoutMs;
  }

  async _request(endpoint, options = {}) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeoutMs);

    const url = `${this.baseUrl}${endpoint}`;
    const headers = {
      'Content-Type': 'application/json',
      ...(options.headers || {})
    };

    try {
      const response = await fetch(url, {
        ...options,
        headers,
        signal: controller.signal
      });

      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        const error = new Error(data.message || `HTTP ${response.status} on ${endpoint}`);
        error.status = response.status;
        error.data = data;
        throw error;
      }

      return data;
    } catch (err) {
      if (err.name === 'AbortError') {
        throw new Error(`Request timeout (${this.timeoutMs}ms) to ${endpoint}`);
      }
      throw err;
    } finally {
      clearTimeout(timeoutId);
    }
  }

  async health() {
    return this._request('/health', { method: 'GET' });
  }

  async runtimeStatus() {
    return this._request('/runtime/status', { method: 'GET' });
  }

  async runtimeStart() {
    return this._request('/runtime/start', { method: 'POST' });
  }

  async runtimeStop(reason = 'runtime_shutdown') {
    return this._request('/runtime/stop', {
      method: 'POST',
      body: JSON.stringify({ reason })
    });
  }

  async runtimeReset() {
    return this._request('/runtime/reset', { method: 'POST' });
  }

  async processFrame(base64Image, timestamp = null) {
    const payload = { image: base64Image };
    if (timestamp) {
      payload.timestamp = timestamp;
    }
    return this._request('/runtime/process-frame', {
      method: 'POST',
      body: JSON.stringify(payload)
    });
  }

  async presenceActive() {
    return this._request('/presence/active', { method: 'GET' });
  }

  async presenceHistory() {
    return this._request('/presence/history', { method: 'GET' });
  }

  async presenceIdentity(name) {
    return this._request(`/presence/identity/${encodeURIComponent(name)}`, { method: 'GET' });
  }
}

// Export singleton instance
window.apiClient = new ApiClient();
