/**
 * Camera Module managing HTML5 MediaDevices webcam access and off-screen canvas frame extraction.
 */
class CameraManager {
  constructor(targetWidth = 640, targetHeight = 480) {
    this.targetWidth = targetWidth;
    this.targetHeight = targetHeight;
    this.stream = null;
    this.videoElement = null;
    this.offscreenCanvas = document.createElement('canvas');
    this.offscreenCanvas.width = targetWidth;
    this.offscreenCanvas.height = targetHeight;
    this.offscreenCtx = this.offscreenCanvas.getContext('2d', { willReadFrequently: true });
    this.isActive = false;
  }

  async start(videoElement) {
    if (this.isActive) return;
    this.videoElement = videoElement;

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('Browser does not support getUserMedia API.');
    }

    const constraints = {
      video: {
        width: { ideal: this.targetWidth },
        height: { ideal: this.targetHeight },
        facingMode: 'user'
      },
      audio: false
    };

    try {
      this.stream = await navigator.mediaDevices.getUserMedia(constraints);
      this.videoElement.srcObject = this.stream;
      await this.videoElement.play();
      this.isActive = true;
    } catch (err) {
      this.isActive = false;
      throw err;
    }
  }

  stop() {
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }
    if (this.videoElement) {
      this.videoElement.srcObject = null;
    }
    this.isActive = false;
  }

  captureFrame(quality = 0.85) {
    if (!this.isActive || !this.videoElement || this.videoElement.readyState < 2) {
      return null;
    }

    const vw = this.videoElement.videoWidth || this.targetWidth;
    const vh = this.videoElement.videoHeight || this.targetHeight;

    if (this.offscreenCanvas.width !== vw || this.offscreenCanvas.height !== vh) {
      this.offscreenCanvas.width = vw;
      this.offscreenCanvas.height = vh;
    }

    this.offscreenCtx.drawImage(this.videoElement, 0, 0, vw, vh);
    // Export JPEG format with configurable compression
    return this.offscreenCanvas.toDataURL('image/jpeg', quality);
  }

  getDimensions() {
    if (!this.videoElement) return { width: this.targetWidth, height: this.targetHeight };
    return {
      width: this.videoElement.videoWidth || this.targetWidth,
      height: this.videoElement.videoHeight || this.targetHeight
    };
  }
}

window.CameraManager = CameraManager;
