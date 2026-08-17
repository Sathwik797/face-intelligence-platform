/**
 * Overlay Renderer module for drawing real-time face bounding boxes and status labels over the video feed.
 */
class OverlayRenderer {
  constructor(canvasElement) {
    this.canvas = canvasElement;
    this.ctx = this.canvas.getContext('2d');
  }

  resize(displayRect) {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.round(displayRect.width);
    const h = Math.round(displayRect.height);

    if (this.canvas.width !== w * dpr || this.canvas.height !== h * dpr) {
      this.canvas.width = w * dpr;
      this.canvas.height = h * dpr;
      this.canvas.style.width = `${w}px`;
      this.canvas.style.height = `${h}px`;
      this.ctx.scale(dpr, dpr);
    }
  }

  clear() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  render(recognitionResult, temporalResult, sourceDimensions, displayRect) {
    this.clear();
    this.resize(displayRect);

    if (!recognitionResult || !recognitionResult.bbox) {
      return;
    }

    const [top, right, bottom, left] = recognitionResult.bbox;
    const srcW = sourceDimensions.width || 640;
    const srcH = sourceDimensions.height || 480;

    const scaleX = displayRect.width / srcW;
    const scaleY = displayRect.height / srcH;

    const boxX = left * scaleX;
    const boxY = top * scaleY;
    const boxW = (right - left) * scaleX;
    const boxH = (bottom - top) * scaleY;

    // Determine status color
    let color = '#ef4444'; // Red for Unknown
    let label = 'Unknown';

    if (recognitionResult.recognized && recognitionResult.identity) {
      color = '#10b981'; // Green for Recognized
      label = `${recognitionResult.identity} (${Math.round(recognitionResult.similarity * 100)}%)`;
    } else if (recognitionResult.quality_status && recognitionResult.quality_status !== 'good' && recognitionResult.quality_status !== 'none') {
      color = '#06b6d4'; // Cyan for Poor Quality
      label = `Quality Rejected (${recognitionResult.quality_status})`;
    } else if (temporalResult && temporalResult.state === 'candidate') {
      color = '#f59e0b'; // Yellow for Candidate
      label = 'Evaluating...';
    }

    // Draw bounding box
    this.ctx.save();
    this.ctx.lineWidth = 3;
    this.ctx.strokeStyle = color;
    this.ctx.strokeRect(boxX, boxY, boxW, boxH);

    // Draw label background pill
    this.ctx.font = 'bold 12px monospace';
    const textMetrics = this.ctx.measureText(label);
    const padding = 6;
    const pillW = textMetrics.width + padding * 2;
    const pillH = 20;
    const pillX = boxX;
    const pillY = Math.max(0, boxY - pillH - 4);

    this.ctx.fillStyle = color;
    this.ctx.fillRect(pillX, pillY, pillW, pillH);

    // Draw label text
    this.ctx.fillStyle = '#ffffff';
    this.ctx.fillText(label, pillX + padding, pillY + 14);

    this.ctx.restore();
  }
}

window.OverlayRenderer = OverlayRenderer;
