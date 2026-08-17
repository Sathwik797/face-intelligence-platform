/**
 * Attendance Journal View Controller.
 * Handles date filtering, statistics summaries, and record table rendering.
 */

class AttendanceViewController {
    constructor() {
        this.dateInput = null;
        this.btnRefresh = null;
        this.btnExportCsv = null;
        this.btnExportJson = null;
        this.tableBody = null;
        this.statPresent = null;
        this.statInProgress = null;
        this.statAvgDwell = null;
    }

    init() {
        this.dateInput = document.getElementById("att-date-input");
        this.btnRefresh = document.getElementById("att-btn-refresh");
        this.btnExportCsv = document.getElementById("att-btn-export-csv");
        this.btnExportJson = document.getElementById("att-btn-export-json");
        this.tableBody = document.getElementById("att-table-body");
        this.statPresent = document.getElementById("att-stat-present");
        this.statInProgress = document.getElementById("att-stat-inprogress");
        this.statAvgDwell = document.getElementById("att-stat-avgdwell");

        if (this.dateInput) {
            // Default to today in YYYY-MM-DD
            const today = new Date().toISOString().split("T")[0];
            this.dateInput.value = today;
            this.dateInput.addEventListener("change", () => this.loadData());
        }

        if (this.btnRefresh) {
            this.btnRefresh.addEventListener("click", () => this.loadData());
        }

        if (this.btnExportCsv) {
            this.btnExportCsv.addEventListener("click", () => this.exportRecords("csv"));
        }

        if (this.btnExportJson) {
            this.btnExportJson.addEventListener("click", () => this.exportRecords("json"));
        }
    }

    async loadData() {
        const dateStr = this.dateInput ? this.dateInput.value : "";
        if (!dateStr) return;

        try {
            // 1. Fetch summary
            const summary = await window.apiClient.getAttendanceSummary(dateStr);
            if (this.statPresent) this.statPresent.textContent = summary.total_present || 0;
            if (this.statInProgress) this.statInProgress.textContent = summary.total_in_progress || 0;
            if (this.statAvgDwell) {
                const mins = Math.round((summary.average_dwell_seconds || 0) / 60);
                this.statAvgDwell.textContent = `${mins} min`;
            }

            // 2. Fetch records
            const recordsData = await window.apiClient.getAttendanceRecords(dateStr);
            this.renderRecords(recordsData.records || []);
        } catch (err) {
            console.error("Failed to load attendance data:", err);
            if (this.tableBody) {
                this.tableBody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--danger);">Failed to load attendance records: ${err.message}</td></tr>`;
            }
        }
    }

    renderRecords(records) {
        if (!this.tableBody) return;
        if (records.length === 0) {
            this.tableBody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--text-muted);">No attendance records found for this date.</td></tr>`;
            return;
        }

        this.tableBody.innerHTML = records.map(r => {
            const firstIn = r.first_check_in ? new Date(r.first_check_in).toLocaleTimeString() : "--";
            const lastOut = r.last_check_out ? new Date(r.last_check_out).toLocaleTimeString() : "In Progress";
            const dwellMins = (r.total_dwell_seconds / 60).toFixed(1);
            
            let statusBadge = "badge-neutral";
            if (r.status === "PRESENT") statusBadge = "badge-success";
            else if (r.status === "IN_PROGRESS") statusBadge = "badge-warning";
            else if (r.status === "PARTIAL") statusBadge = "badge-danger";

            return `
                <tr>
                    <td style="font-weight: 500;">${r.identity}</td>
                    <td>${firstIn}</td>
                    <td>${lastOut}</td>
                    <td>${dwellMins} min</td>
                    <td>${r.session_count}</td>
                    <td><span class="badge ${statusBadge}">${r.status}</span></td>
                </tr>
            `;
        }).join("");
    }

    exportRecords(format = "csv") {
        const dateStr = this.dateInput ? this.dateInput.value : "";
        const url = window.apiClient.getAttendanceExportUrl(dateStr, format);
        window.open(url, "_blank");
    }
}

window.attendanceView = new AttendanceViewController();
