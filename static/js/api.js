/**
 * REST API Client for Face Intelligence Platform.
 * Connects frontend dashboard components to /api/v1/ endpoints.
 */

class ApiClient {
    constructor(baseUrl = "/api/v1", timeoutMs = 8000) {
        this.baseUrl = baseUrl;
        this.timeoutMs = timeoutMs;
    }

    async _fetchWithTimeout(url, options = {}) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.timeoutMs);
        try {
            const response = await fetch(url, {
                ...options,
                signal: controller.signal
            });
            clearTimeout(timeoutId);
            return response;
        } catch (err) {
            clearTimeout(timeoutId);
            if (err.name === "AbortError") {
                throw new Error(`Request timed out after ${this.timeoutMs}ms`);
            }
            throw err;
        }
    }

    // Health & Status
    async checkHealth() {
        const res = await this._fetchWithTimeout(`${this.baseUrl}/health`);
        return await res.json();
    }

    async getRuntimeStatus() {
        const res = await this._fetchWithTimeout(`${this.baseUrl}/runtime/status`);
        return await res.json();
    }

    // Runtime Lifecycle
    async startRuntime() {
        const res = await this._fetchWithTimeout(`${this.baseUrl}/runtime/start`, {
            method: "POST"
        });
        return await res.json();
    }

    async stopRuntime() {
        const res = await this._fetchWithTimeout(`${this.baseUrl}/runtime/stop`, {
            method: "POST"
        });
        return await res.json();
    }

    async resetRuntime() {
        const res = await this._fetchWithTimeout(`${this.baseUrl}/runtime/reset`, {
            method: "POST"
        });
        return await res.json();
    }

    // Frame Processing
    async processFrame(base64Image, clientTimestamp = null) {
        const payload = {
            image: base64Image,
            timestamp: clientTimestamp || new Date().toISOString()
        };
        const res = await this._fetchWithTimeout(`${this.baseUrl}/runtime/process-frame`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.message || `Server error: ${res.status}`);
        }
        return await res.json();
    }

    // Presence Queries
    async getActivePresence() {
        const res = await this._fetchWithTimeout(`${this.baseUrl}/presence/active`);
        return await res.json();
    }

    async getPresenceHistory() {
        const res = await this._fetchWithTimeout(`${this.baseUrl}/presence/history`);
        return await res.json();
    }

    // Attendance Queries & Export
    async getAttendanceRecords(dateStr = null, identity = null) {
        let url = `${this.baseUrl}/attendance/records`;
        const params = new URLSearchParams();
        if (dateStr) params.append("date", dateStr);
        if (identity) params.append("identity", identity);
        if (params.toString()) url += `?${params.toString()}`;

        const res = await this._fetchWithTimeout(url);
        return await res.json();
    }

    async getAttendanceSummary(dateStr = null) {
        let url = `${this.baseUrl}/attendance/summary`;
        if (dateStr) url += `?date=${encodeURIComponent(dateStr)}`;
        const res = await this._fetchWithTimeout(url);
        return await res.json();
    }

    getAttendanceExportUrl(dateStr = null, format = "csv") {
        let url = `${this.baseUrl}/attendance/export?format=${format}`;
        if (dateStr) url += `&date=${encodeURIComponent(dateStr)}`;
        return url;
    }

    // Identity Management & Dynamic Enrollment
    async listIdentities() {
        const res = await this._fetchWithTimeout(`${this.baseUrl}/identities`);
        return await res.json();
    }

    async getIdentityDetails(name) {
        const res = await this._fetchWithTimeout(`${this.baseUrl}/identities/${encodeURIComponent(name)}`);
        return await res.json();
    }

    async enrollIdentity(identity, base64Image, qualityMode = "balanced", notes = null) {
        const payload = {
            identity,
            image: base64Image,
            quality_mode: qualityMode,
            notes
        };
        const res = await this._fetchWithTimeout(`${this.baseUrl}/identities/enroll`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (!res.ok) {
            const err = new Error(data.message || `Enrollment failed: ${res.status}`);
            err.status = res.status;
            err.data = data;
            throw err;
        }
        return data;
    }

    async deleteIdentity(name) {
        const res = await this._fetchWithTimeout(`${this.baseUrl}/identities/${encodeURIComponent(name)}`, {
            method: "DELETE"
        });
        return await res.json();
    }
}

window.apiClient = new ApiClient();
