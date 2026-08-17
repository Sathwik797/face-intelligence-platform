/**
 * Identity Directory & Dynamic Enrollment View Controller.
 * Manages identity listing, webcam snapshot capture, file uploads, FQA feedback, and deletion.
 */

class IdentitiesViewController {
    constructor() {
        this.tableBody = null;
        this.statCount = null;
        this.btnOpenModal = null;
        this.btnCloseModal = null;
        this.modal = null;
        
        // Modal elements
        this.inputName = null;
        this.inputNotes = null;
        this.selectQuality = null;
        this.btnSnapWebcam = null;
        this.inputFile = null;
        this.previewImg = null;
        this.btnSubmitEnroll = null;
        this.enrollFeedback = null;

        this.capturedBase64 = null;
    }

    init() {
        this.tableBody = document.getElementById("id-table-body");
        this.statCount = document.getElementById("id-stat-total");
        this.btnOpenModal = document.getElementById("id-btn-open-enroll");
        this.btnCloseModal = document.getElementById("modal-enroll-close");
        this.modal = document.getElementById("modal-enroll");

        this.inputName = document.getElementById("enroll-name");
        this.inputNotes = document.getElementById("enroll-notes");
        this.selectQuality = document.getElementById("enroll-quality-mode");
        this.btnSnapWebcam = document.getElementById("enroll-btn-snap");
        this.inputFile = document.getElementById("enroll-file-input");
        this.previewImg = document.getElementById("enroll-preview-img");
        this.btnSubmitEnroll = document.getElementById("enroll-btn-submit");
        this.enrollFeedback = document.getElementById("enroll-feedback");

        if (this.btnOpenModal) {
            this.btnOpenModal.addEventListener("click", () => this.openEnrollModal());
        }

        if (this.btnCloseModal) {
            this.btnCloseModal.addEventListener("click", () => this.closeEnrollModal());
        }

        if (this.btnSnapWebcam) {
            this.btnSnapWebcam.addEventListener("click", () => this.captureFromCamera());
        }

        if (this.inputFile) {
            this.inputFile.addEventListener("change", (e) => this.handleFileUpload(e));
        }

        if (this.btnSubmitEnroll) {
            this.btnSubmitEnroll.addEventListener("click", () => this.submitEnrollment());
        }
    }

    async loadData() {
        try {
            const data = await window.apiClient.listIdentities();
            const identities = data.identities || [];
            if (this.statCount) this.statCount.textContent = identities.length;
            this.renderIdentities(identities);
        } catch (err) {
            console.error("Failed to load identities:", err);
            if (this.tableBody) {
                this.tableBody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--danger);">Failed to load identities: ${err.message}</td></tr>`;
            }
        }
    }

    renderIdentities(identities) {
        if (!this.tableBody) return;
        if (identities.length === 0) {
            this.tableBody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--text-muted);">No enrolled identities found. Click "Enroll New Person" to onboard.</td></tr>`;
            return;
        }

        this.tableBody.innerHTML = identities.map(i => {
            const enrolledAt = i.created_at ? new Date(i.created_at).toLocaleDateString() : "--";
            const notes = i.notes || "--";

            return `
                <tr>
                    <td style="font-weight: 500; font-size: 14px;">${i.identity}</td>
                    <td><span class="badge badge-primary">${i.template_count} template(s)</span></td>
                    <td>${enrolledAt}</td>
                    <td style="color: var(--text-muted); font-size: 12px;">${notes}</td>
                    <td>
                        <button class="btn btn-secondary btn-sm" style="color: var(--danger); border-color: var(--border);" onclick="window.identitiesView.deleteIdentity('${encodeURIComponent(i.identity)}')">Delete</button>
                    </td>
                </tr>
            `;
        }).join("");
    }

    openEnrollModal() {
        this.capturedBase64 = null;
        if (this.inputName) this.inputName.value = "";
        if (this.inputNotes) this.inputNotes.value = "";
        if (this.previewImg) {
            this.previewImg.src = "";
            this.previewImg.style.display = "none";
        }
        if (this.enrollFeedback) {
            this.enrollFeedback.style.display = "none";
            this.enrollFeedback.textContent = "";
        }
        if (this.modal) this.modal.style.display = "flex";
    }

    closeEnrollModal() {
        if (this.modal) this.modal.style.display = "none";
    }

    captureFromCamera() {
        if (!window.cameraManager || !window.cameraManager.isStreaming) {
            this.showFeedback("Please start the live camera on the Live Stream tab first.", "error");
            return;
        }

        const frame = window.cameraManager.captureFrame(0.92);
        if (!frame) {
            this.showFeedback("Could not capture frame from webcam stream.", "error");
            return;
        }

        this.capturedBase64 = frame;
        if (this.previewImg) {
            this.previewImg.src = frame;
            this.previewImg.style.display = "block";
        }
        this.showFeedback("Snapshot captured from live video stream.", "success");
    }

    handleFileUpload(e) {
        const file = e.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (event) => {
            this.capturedBase64 = event.target.result;
            if (this.previewImg) {
                this.previewImg.src = this.capturedBase64;
                this.previewImg.style.display = "block";
            }
            this.showFeedback("Image file selected.", "success");
        };
        reader.readAsDataURL(file);
    }

    async submitEnrollment() {
        const name = this.inputName ? this.inputName.value.trim() : "";
        const notes = this.inputNotes ? this.inputNotes.value.trim() : "";
        const qualityMode = this.selectQuality ? this.selectQuality.value : "balanced";

        if (!name) {
            this.showFeedback("Please enter a person name.", "error");
            return;
        }

        if (!this.capturedBase64) {
            this.showFeedback("Please capture a webcam snapshot or select an image file.", "error");
            return;
        }

        if (this.btnSubmitEnroll) this.btnSubmitEnroll.disabled = true;
        this.showFeedback("Evaluating face quality and extracting ArcFace embedding...", "info");

        try {
            const res = await window.apiClient.enrollIdentity(name, this.capturedBase64, qualityMode, notes);
            this.showFeedback(`✓ ${res.message} (Quality Score: ${res.quality_score || 'Passed'})`, "success");
            setTimeout(() => {
                this.closeEnrollModal();
                this.loadData();
            }, 1200);
        } catch (err) {
            console.error("Enrollment failed:", err);
            const msg = err.data && err.data.message ? err.data.message : err.message;
            this.showFeedback(`✗ ${msg}`, "error");
        } finally {
            if (this.btnSubmitEnroll) this.btnSubmitEnroll.disabled = false;
        }
    }

    async deleteIdentity(encodedName) {
        const name = decodeURIComponent(encodedName);
        if (!confirm(`Are you sure you want to remove '${name}' from the biometric gallery and database?`)) {
            return;
        }

        try {
            await window.apiClient.deleteIdentity(name);
            this.loadData();
        } catch (err) {
            alert(`Failed to delete identity: ${err.message}`);
        }
    }

    showFeedback(message, type = "info") {
        if (!this.enrollFeedback) return;
        this.enrollFeedback.style.display = "block";
        this.enrollFeedback.textContent = message;
        this.enrollFeedback.className = `status-banner ${type}`;
    }
}

window.identitiesView = new IdentitiesViewController();
