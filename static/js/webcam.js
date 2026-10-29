/**
 * FaceSense AI – Real-Time Live Webcam Controller
 * Advanced Computer Vision & Deep Learning Expression Analysis
 * 
 * High-Performance Jitter-Free Features:
 * - WebRTC getUserMedia video capture & synchronized canvas overlay
 * - Hardware-accelerated 60 FPS bounding box smoothing via Linear Interpolation (Lerp)
 * - Deadband filtering to eliminate Haar-cascade micro-tremors & vibration
 * - Emotion label hysteresis stabilization to prevent rapid label fluttering
 * - Zero layout-shift DOM updates (fixed container geometry, smooth opacity transitions)
 * - Widescreen Theater Mode / Expand View toggle
 * - Instantaneous FPS and backend inference latency tracking
 * - Pre-inference face quality diagnostics (lighting & blur detection)
 * - Rolling prediction history timeline
 */

let webcamStream = null;
let isWebcamActive = false;
let autoDetectTimer = null;
let isAnalyzingFrame = false;

// Temporal and smoothing state for CNN probabilities
let lastProbabilities = null;
let predictionHistory = [];
const MAX_HISTORY_ITEMS = 12;

// Processing metrics
let currentFps = 0;

// 60 FPS Smooth Canvas Animation State (Lerp + Deadband)
let currentBbox = null;     // { x, y, w, h }
let targetBbox = null;      // { x, y, w, h }
let targetSecondaryBboxes = [];
let currentSecondaryBboxes = [];
let overlayAnimId = null;
let boxAlpha = 0;
let targetBoxAlpha = 0;

let currentTag = { label: "Neutral", conf: 0, color: "#10b981", isUncertain: false };
let targetTag = { label: "Neutral", conf: 0, color: "#10b981", isUncertain: false };

// Phase 3: Multi-Mode Analysis State ('cnn', 'lstm', 'compare')
let currentAnalysisMode = "cnn";
let clientSessionId = "session_" + Date.now() + "_" + Math.random().toString(36).substring(2, 8);

// Emotion Label Hysteresis (prevents rapid 1-frame label flipping)
let stableLabel = null;
let stableConf = 0;
let candidateLabel = null;
let candidateStreak = 0;

// Widescreen Theater Mode State
let isTheaterMode = false;

document.addEventListener("DOMContentLoaded", () => {
    initWebcamControls();
});

function initWebcamControls() {
    const threshSlider = document.getElementById("confidence-threshold-slider");
    const threshValue = document.getElementById("confidence-threshold-value");
    if (threshSlider && threshValue) {
        threshSlider.addEventListener("input", (e) => {
            threshValue.textContent = `${e.target.value}%`;
        });
    }
}

/**
 * Switch facial expression analysis mode:
 * 'cnn'     : Single-frame instantaneous spatial classification
 * 'lstm'    : 16-frame spatiotemporal sequence classification
 * 'compare' : Dual inference displaying both predictions side-by-side
 */
function setAnalysisMode(mode) {
    if (!["cnn", "lstm", "compare"].includes(mode)) return;
    currentAnalysisMode = mode;

    // Toggle button styling
    const btnCnn = document.getElementById("mode-cnn-btn");
    const btnLstm = document.getElementById("mode-lstm-btn");
    const btnCmp = document.getElementById("mode-compare-btn");

    if (btnCnn) btnCnn.classList.toggle("active", mode === "cnn");
    if (btnLstm) btnLstm.classList.toggle("active", mode === "lstm");
    if (btnCmp) btnCmp.classList.toggle("active", mode === "compare");

    // Dynamic headers and HUD tags
    const titleEl = document.getElementById("viewport-mode-title");
    const subEl = document.getElementById("viewport-mode-sub");
    const modeBadge = document.getElementById("active-mode-badge-label");

    if (mode === "cnn") {
        if (titleEl) titleEl.textContent = "Live Webcam Stream (CNN Mode)";
        if (subEl) subEl.textContent = "Instantaneous single-frame spatial analysis";
        if (modeBadge) modeBadge.textContent = "CNN";
    } else if (mode === "lstm") {
        if (titleEl) titleEl.textContent = "Live Webcam Stream (CNN + LSTM Mode)";
        if (subEl) subEl.textContent = "16-frame spatiotemporal sequence sliding window";
        if (modeBadge) modeBadge.textContent = "CNN+LSTM";
    } else {
        if (titleEl) titleEl.textContent = "Live Webcam Stream (Dual Comparison Mode)";
        if (subEl) subEl.textContent = "Side-by-side spatial vs temporal prediction analysis";
        if (modeBadge) modeBadge.textContent = "Dual Comparison";
    }

    // Toggle temporal buffer bar
    const bufferBox = document.getElementById("sequence-buffer-box");
    if (bufferBox) {
        if (mode === "lstm" || mode === "compare") {
            bufferBox.classList.add("is-active");
        } else {
            bufferBox.classList.remove("is-active");
        }
    }

    // Toggle side-by-side compare panel
    const compareBox = document.getElementById("webcam-compare-container");
    if (compareBox) {
        if (mode === "compare") {
            compareBox.classList.add("is-active");
        } else {
            compareBox.classList.remove("is-active");
        }
    }

    // Reset buffer on backend for clean mode transition
    fetch("/api/sequence_buffer_reset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: clientSessionId })
    }).catch(err => console.debug("Buffer reset ping:", err));

    updateTemporalBufferUI({ current_frames: 0, sequence_length: 16, progress_percent: 0, is_ready: false });
}

function updateTemporalBufferUI(bufStatus) {
    if (!bufStatus) return;
    const countVal = document.getElementById("seq-count-val");
    const pctVal = document.getElementById("seq-pct-val");
    const fillEl = document.getElementById("seq-progress-fill");
    const badgeEl = document.getElementById("seq-status-badge");

    const reqFrames = bufStatus.sequence_length || 16;
    if (countVal) countVal.textContent = `${bufStatus.current_frames} / ${reqFrames} Frames`;
    if (pctVal) pctVal.textContent = `${bufStatus.progress_percent || 0}%`;
    if (fillEl) {
        fillEl.style.width = `${bufStatus.progress_percent || 0}%`;
        if (bufStatus.is_ready) {
            fillEl.classList.add("ready");
        } else {
            fillEl.classList.remove("ready");
        }
    }
    if (badgeEl) {
        if (bufStatus.missing_streak > 0) {
            badgeEl.textContent = "Tracking lost: Buffer reset";
        } else if (bufStatus.is_ready) {
            badgeEl.textContent = "Sliding Window Active";
        } else {
            badgeEl.textContent = `Collecting Frames (${bufStatus.current_frames}/${reqFrames})...`;
        }
    }
}

function updateCompareUI(cmp) {
    if (!cmp) return;
    const cmpCnnLabel = document.getElementById("cmp-cnn-label");
    const cmpCnnConf = document.getElementById("cmp-cnn-conf");
    const cmpCnnLatency = document.getElementById("cmp-cnn-latency");
    const cmpLstmLabel = document.getElementById("cmp-lstm-label");
    const cmpLstmConf = document.getElementById("cmp-lstm-conf");
    const cmpLstmLatency = document.getElementById("cmp-lstm-latency");
    const pill = document.getElementById("cmp-agreement-pill");
    const desc = document.getElementById("cmp-agreement-desc");

    if (cmp.cnn) {
        if (cmpCnnLabel) {
            cmpCnnLabel.textContent = cmp.cnn.predicted_label;
            cmpCnnLabel.style.color = cmp.cnn.color || "#3b82f6";
        }
        if (cmpCnnConf) cmpCnnConf.textContent = `${cmp.cnn.confidence}% Confidence`;
        if (cmpCnnLatency) cmpCnnLatency.textContent = `${cmp.cnn.latency_ms || 0} ms`;
    }

    if (cmp.lstm) {
        if (cmp.lstm.status === "success") {
            if (cmpLstmLabel) {
                cmpLstmLabel.textContent = cmp.lstm.predicted_label;
                cmpLstmLabel.style.color = cmp.lstm.color || "#8b5cf6";
            }
            if (cmpLstmConf) cmpLstmConf.textContent = `${cmp.lstm.confidence}% Confidence`;
            if (cmpLstmLatency) cmpLstmLatency.textContent = `${cmp.lstm.latency_ms || 0} ms`;
        } else if (cmp.lstm.status === "not_trained") {
            if (cmpLstmLabel) {
                cmpLstmLabel.textContent = "Awaiting Video Training";
                cmpLstmLabel.style.color = "#fbbf24";
            }
            if (cmpLstmConf) cmpLstmConf.textContent = "Pipeline Ready (dataset/videos/)";
            if (cmpLstmLatency) cmpLstmLatency.textContent = "-- ms";
        } else {
            if (cmpLstmLabel) {
                cmpLstmLabel.textContent = "Buffering...";
                cmpLstmLabel.style.color = "#c4b5fd";
            }
            if (cmpLstmConf) cmpLstmConf.textContent = cmp.lstm.message || "Collecting frames";
            if (cmpLstmLatency) cmpLstmLatency.textContent = "-- ms";
        }
    }

    if (pill && desc) {
        if (cmp.agreement === true) {
            pill.className = "agreement-pill agree";
            pill.innerHTML = '<span class="pill-dot green"></span> <span>Models in Consensus</span>';
            desc.textContent = `Both models agree on expression '${cmp.cnn.predicted_label}'.`;
        } else if (cmp.agreement === false) {
            pill.className = "agreement-pill diverge";
            pill.innerHTML = '<span class="pill-dot amber"></span> <span>Models Diverge</span>';
            desc.textContent = `CNN predicts '${cmp.cnn.predicted_label}' vs LSTM predicts '${cmp.lstm.predicted_label}'.`;
        } else {
            pill.className = "agreement-pill buffering";
            pill.innerHTML = '<span class="pill-dot purple"></span> <span>Buffering Sequence</span>';
            desc.textContent = (cmp.lstm && cmp.lstm.message) ? cmp.lstm.message : "Accumulating consecutive video frames...";
        }
    }
}

/**
 * Toggle Widescreen Theater Mode to optimize viewport fit on large screens.
 */
function toggleTheaterMode() {
    isTheaterMode = !isTheaterMode;
    const grid = document.getElementById("webcam-main-grid");
    const btnText = document.getElementById("theater-btn-text");

    if (grid) {
        if (isTheaterMode) {
            grid.classList.add("theater-mode");
            if (btnText) btnText.textContent = "Normal View";
        } else {
            grid.classList.remove("theater-mode");
            if (btnText) btnText.textContent = "Wide View";
        }
    }
    // Re-synchronize overlay canvas size after transition
    setTimeout(initCanvasSize, 320);
}

/**
 * Request hardware camera access and initiate live processing.
 */
async function startWebcam() {
    const video = document.getElementById("webcam-video");
    const placeholder = document.getElementById("webcam-placeholder");
    const startBtn = document.getElementById("start-cam-btn");
    const stopBtn = document.getElementById("stop-cam-btn");
    const autoToggle = document.getElementById("auto-detect-toggle");
    const statusText = document.getElementById("cam-status-text");
    const indicator = document.getElementById("cam-indicator");

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert("Your web browser does not support the Camera MediaDevices API.");
        return;
    }

    try {
        statusText.textContent = "Connecting to webcam hardware...";
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: "user"
            },
            audio: false
        });

        video.srcObject = webcamStream;
        await video.play();

        isWebcamActive = true;
        placeholder.style.display = "none";
        startBtn.disabled = true;
        stopBtn.disabled = false;
        if (autoToggle) autoToggle.disabled = false;

        indicator.classList.add("active");
        statusText.textContent = "Live camera stream active (640x480)";

        video.addEventListener("loadedmetadata", initCanvasSize);
        initCanvasSize();

        // Start 60 FPS hardware-interpolated canvas render loop
        startOverlayRenderLoop();

        // Start continuous inference analysis
        const autoCheck = document.getElementById("auto-detect-toggle");
        if (autoCheck && autoCheck.checked) {
            toggleAutoDetection(true);
        } else {
            captureAndPredict();
        }

    } catch (err) {
        console.error("Camera Hardware Error:", err);
        indicator.classList.remove("active");
        if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
            statusText.textContent = "Camera permission denied in browser settings.";
            alert("Camera Permission Denied: Please enable camera access in your browser address bar.");
        } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
            statusText.textContent = "No camera hardware detected on device.";
            alert("No webcam found on this device.");
        } else {
            statusText.textContent = `Camera error: ${err.message}`;
        }
    }
}

/**
 * Safely terminate camera stream and reset rendering pipelines.
 */
function stopWebcam() {
    const video = document.getElementById("webcam-video");
    const placeholder = document.getElementById("webcam-placeholder");
    const startBtn = document.getElementById("start-cam-btn");
    const stopBtn = document.getElementById("stop-cam-btn");
    const statusText = document.getElementById("cam-status-text");
    const indicator = document.getElementById("cam-indicator");

    if (autoDetectTimer) {
        clearInterval(autoDetectTimer);
        autoDetectTimer = null;
    }

    if (overlayAnimId) {
        cancelAnimationFrame(overlayAnimId);
        overlayAnimId = null;
    }

    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }

    if (video) video.srcObject = null;
    isWebcamActive = false;
    lastProbabilities = null;

    // Reset bounding box smoothing targets
    targetBbox = null;
    currentBbox = null;
    boxAlpha = 0;
    targetBoxAlpha = 0;
    stableLabel = null;
    candidateStreak = 0;

    clearOverlayCanvas();

    placeholder.style.display = "flex";
    startBtn.disabled = false;
    stopBtn.disabled = true;
    indicator.classList.remove("active");
    statusText.textContent = "Camera offline";

    const emptyState = document.getElementById("webcam-empty-state");
    const liveData = document.getElementById("webcam-live-data");
    if (emptyState) emptyState.style.display = "flex";
    if (liveData) liveData.style.display = "none";

    // Reset telemetry badges
    updateQualityBadgeUI({ status: "standby", message: "Standby" });
    const facesBadge = document.getElementById("webcam-faces-badge");
    if (facesBadge) facesBadge.classList.remove("is-visible");
}

function initCanvasSize() {
    const video = document.getElementById("webcam-video");
    const canvas = document.getElementById("webcam-canvas");
    if (video && canvas && video.videoWidth > 0) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
    }
}

function toggleAutoDetection(enabled) {
    if (enabled) {
        if (!autoDetectTimer && isWebcamActive) {
            // 450ms cycle provides responsive real-time feedback with zero CPU overload
            autoDetectTimer = setInterval(() => {
                if (isWebcamActive && !isAnalyzingFrame) {
                    captureAndPredict();
                }
            }, 450);
        }
    } else {
        if (autoDetectTimer) {
            clearInterval(autoDetectTimer);
            autoDetectTimer = null;
        }
    }
}

/**
 * Capture frame from live video and dispatch to backend CNN inference API.
 */
async function captureAndPredict() {
    if (!isWebcamActive || isAnalyzingFrame) return;

    const video = document.getElementById("webcam-video");
    const statusText = document.getElementById("cam-status-text");

    if (!video || video.readyState !== 4) return;

    isAnalyzingFrame = true;
    const clientReqTime = performance.now();

    try {
        const offCanvas = document.createElement("canvas");
        offCanvas.width = video.videoWidth || 640;
        offCanvas.height = video.videoHeight || 480;
        const offCtx = offCanvas.getContext("2d");
        offCtx.drawImage(video, 0, 0, offCanvas.width, offCanvas.height);

        const frameData = offCanvas.toDataURL("image/jpeg", 0.85);

        // Read UI settings
        const threshInput = document.getElementById("confidence-threshold-slider");
        const confThresh = threshInput ? parseFloat(threshInput.value) : 35.0;

        const smoothingToggle = document.getElementById("smoothing-toggle");
        const enableSmoothing = smoothingToggle ? smoothingToggle.checked : true;

        statusText.textContent = "Analyzing facial expression...";

        const response = await fetch("/predict_webcam", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-Requested-With": "XMLHttpRequest"
            },
            body: JSON.stringify({
                frame: frameData,
                confidence_threshold: confThresh,
                enable_smoothing: enableSmoothing,
                previous_probabilities: lastProbabilities,
                mode: currentAnalysisMode,
                session_id: clientSessionId
            })
        });

        const data = await response.json();

        // Calculate client-side latency & instantaneous FPS
        const elapsedMs = performance.now() - clientReqTime;
        currentFps = Math.round(1000 / Math.max(elapsedMs, 30));
        updateFpsAndLatencyUI(data.latency_ms || Math.round(elapsedMs), currentFps);

        // Update temporal sequence buffer status
        if (data.temporal_buffer) {
            updateTemporalBufferUI(data.temporal_buffer);
        }

        // Mode C Comparison handling
        if (data.mode === "compare" && data.comparison) {
            updateCompareUI(data.comparison);
        }

        if (response.status === 503) {
            statusText.textContent = "Model not found: Run 'python training/train.py' first.";
            alert("Model file not found. Please train the model using 'python training/train.py'");
            stopWebcam();
            return;
        }

        if (data.status === "success" && data.face_detected) {
            // Save probabilities for next frame EMA smoothing
            lastProbabilities = data.probabilities;

            // Apply Hysteresis stabilization on dominant expression label
            const stabilized = getStabilizedLabel(data.predicted_label, data.confidence, data.probabilities);
            const displayLabel = stabilized.label;
            const displayConf = stabilized.conf;

            // Update Target Bounding Box for 60 FPS Lerp animation
            targetBbox = data.bbox;
            targetTag = {
                label: displayLabel,
                conf: displayConf,
                color: data.color || "#10b981",
                isUncertain: data.is_uncertain
            };

            // Secondary bounding boxes
            targetSecondaryBboxes = (data.all_bboxes && data.all_bboxes.length > 1) 
                ? data.all_bboxes.slice(1) 
                : [];

            // Update metric cards and probability bars without layout shift
            updateWebcamResultsUI({
                ...data,
                predicted_label: displayLabel,
                confidence: displayConf
            });

            // Update rolling timeline
            recordPredictionHistory({
                ...data,
                predicted_label: displayLabel,
                confidence: displayConf
            });

            // Status message
            if (data.is_uncertain) {
                statusText.textContent = `Uncertain expression (${displayConf}%): ${displayLabel}`;
            } else {
                statusText.textContent = `Predicted: ${displayLabel} (${displayConf}%)`;
            }

        } else if (data.status === "no_face") {
            targetBbox = null;
            targetSecondaryBboxes = [];
            lastProbabilities = null;
            statusText.textContent = "Looking for a face... Center your face under good lighting.";
            updateQualityBadgeUI({ status: "searching", message: "Center face in view" });
            const facesBadge = document.getElementById("webcam-faces-badge");
            if (facesBadge) facesBadge.classList.remove("is-visible");
        } else {
            statusText.textContent = data.message || "Detection error";
        }

    } catch (err) {
        console.error("Webcam Inference Error:", err);
        statusText.textContent = "Inference request interrupted";
    } finally {
        isAnalyzingFrame = false;
    }
}

/**
 * Emotion Label Hysteresis Stabilizer.
 * Prevents rapid 1-frame label fluttering between two close probability classes.
 */
function getStabilizedLabel(newLabel, newConf, probabilities) {
    if (!stableLabel) {
        stableLabel = newLabel;
        stableConf = newConf;
        return { label: newLabel, conf: newConf };
    }

    if (newLabel === stableLabel) {
        candidateStreak = 0;
        candidateLabel = null;
        stableConf = newConf;
        return { label: stableLabel, conf: stableConf };
    }

    // Candidate label wants to displace current dominant label
    if (newLabel === candidateLabel) {
        candidateStreak++;
    } else {
        candidateLabel = newLabel;
        candidateStreak = 1;
    }

    // Switch if confidence gap >= 5.0% or candidate sustained for at least 2 frames
    const currentDominantProb = (probabilities && probabilities[stableLabel]) || 0;
    const gap = newConf - currentDominantProb;

    if (gap >= 5.0 || candidateStreak >= 2) {
        stableLabel = newLabel;
        stableConf = newConf;
        candidateStreak = 0;
        candidateLabel = null;
    }

    return { label: stableLabel, conf: stableConf };
}

/**
 * 60 FPS Hardware-Interpolated Canvas Render Loop.
 * Uses Linear Interpolation (Lerp) and a deadband filter to eliminate box vibration completely.
 */
function startOverlayRenderLoop() {
    if (overlayAnimId) cancelAnimationFrame(overlayAnimId);

    function loop() {
        if (!isWebcamActive) {
            clearOverlayCanvas();
            return;
        }

        const canvas = document.getElementById("webcam-canvas");
        if (canvas) {
            const ctx = canvas.getContext("2d");
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            if (targetBbox) {
                targetBoxAlpha = 1.0;

                if (!currentBbox) {
                    currentBbox = { ...targetBbox };
                    currentTag = { ...targetTag };
                } else {
                    const lerpFactor = 0.22;
                    const dx = targetBbox.x - currentBbox.x;
                    const dy = targetBbox.y - currentBbox.y;
                    const dw = targetBbox.w - currentBbox.w;
                    const dh = targetBbox.h - currentBbox.h;

                    // Deadband threshold: Ignore micro-jitter (< 2.5px) to keep box completely motionless
                    if (Math.abs(dx) > 2.5) currentBbox.x += dx * lerpFactor;
                    if (Math.abs(dy) > 2.5) currentBbox.y += dy * lerpFactor;
                    if (Math.abs(dw) > 2.5) currentBbox.w += dw * lerpFactor;
                    if (Math.abs(dh) > 2.5) currentBbox.h += dh * lerpFactor;

                    currentTag = { ...targetTag };
                }
            } else {
                targetBoxAlpha = 0.0;
            }

            // Smooth opacity fade in / fade out
            boxAlpha += (targetBoxAlpha - boxAlpha) * 0.18;

            if (boxAlpha > 0.04 && currentBbox) {
                drawSmoothFaceHUD(ctx, currentBbox, currentTag, boxAlpha);
            } else if (boxAlpha <= 0.04) {
                currentBbox = null;
            }
        }

        overlayAnimId = requestAnimationFrame(loop);
    }

    overlayAnimId = requestAnimationFrame(loop);
}

/**
 * Draw decorative HUD brackets, subtle glow, and expression badge on the canvas.
 */
function drawSmoothFaceHUD(ctx, bbox, tag, alpha) {
    ctx.save();
    ctx.globalAlpha = alpha;

    const { x, y, w, h } = bbox;
    const color = tag.color || "#10b981";

    // 1. Primary bounding box with tech glow
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.strokeRect(x, y, w, h);

    // Inner subtle glow
    ctx.shadowColor = color;
    ctx.shadowBlur = 10;
    ctx.strokeRect(x, y, w, h);
    ctx.shadowBlur = 0; // Reset blur for clean corners

    // 2. High-Tech Corner Brackets
    const cLen = Math.min(18, Math.floor(w / 4), Math.floor(h / 4));
    ctx.lineWidth = 4;
    ctx.strokeStyle = color;

    // Top-left
    ctx.beginPath(); ctx.moveTo(x, y + cLen); ctx.lineTo(x, y); ctx.lineTo(x + cLen, y); ctx.stroke();
    // Top-right
    ctx.beginPath(); ctx.moveTo(x + w - cLen, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + cLen); ctx.stroke();
    // Bottom-left
    ctx.beginPath(); ctx.moveTo(x, y + h - cLen); ctx.lineTo(x, y + h); ctx.lineTo(x + cLen, y + h); ctx.stroke();
    // Bottom-right
    ctx.beginPath(); ctx.moveTo(x + w - cLen, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - cLen); ctx.stroke();

    // 3. Expression Label Tag
    const tagLabel = tag.isUncertain
        ? `${tag.label}? (${tag.conf.toFixed(1)}%)`
        : `${tag.label} (${tag.conf.toFixed(1)}%)`;

    ctx.font = "600 14px 'Plus Jakarta Sans', sans-serif";
    const textWidth = ctx.measureText(tagLabel).width;

    const tagH = 26;
    const tagW = textWidth + 18;
    const tagY = Math.max(0, y - tagH - 4);

    // Pill background
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.roundRect(x, tagY, tagW, tagH, [4, 4, 4, 4]);
    ctx.fill();

    // Tag text
    ctx.fillStyle = "#ffffff";
    ctx.fillText(tagLabel, x + 9, tagY + 18);

    ctx.restore();
}

function clearOverlayCanvas() {
    const canvas = document.getElementById("webcam-canvas");
    if (canvas) {
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
}

/**
 * Update UI cards, probabilities, and telemetry with ZERO layout shift.
 */
function updateWebcamResultsUI(data) {
    const emptyState = document.getElementById("webcam-empty-state");
    const liveData = document.getElementById("webcam-live-data");
    const predLabel = document.getElementById("webcam-predicted-label");
    const confidenceVal = document.getElementById("webcam-confidence");
    const descText = document.getElementById("webcam-description");
    const heroCard = document.getElementById("webcam-label-card");
    const uncertainNotice = document.getElementById("webcam-uncertain-notice");

    if (emptyState) emptyState.style.display = "none";
    if (liveData) liveData.style.display = "block";

    if (predLabel) predLabel.textContent = data.predicted_label;
    if (confidenceVal) confidenceVal.textContent = `${data.confidence}%`;
    if (descText) descText.textContent = data.description || "";

    if (heroCard) {
        heroCard.style.borderColor = data.color || "#4f46e5";
        heroCard.style.boxShadow = `0 4px 20px -2px ${data.color}33`;
    }

    // Uncertain banner (Toggle CSS class without changing container height)
    if (uncertainNotice) {
        if (data.is_uncertain) {
            uncertainNotice.classList.add("is-active");
            const noteText = document.getElementById("webcam-uncertain-text");
            if (noteText) noteText.textContent = data.uncertainty_note || "Low confidence prediction.";
        } else {
            uncertainNotice.classList.remove("is-active");
        }
    }

    // Update 7-class horizontal probability bars smoothly
    if (data.probabilities) {
        for (const [className, prob] of Object.entries(data.probabilities)) {
            const bar = document.getElementById(`bar-${className}`);
            const val = document.getElementById(`val-${className}`);
            const row = document.getElementById(`row-${className}`);

            if (bar) bar.style.width = `${prob}%`;
            if (val) val.textContent = `${prob.toFixed(1)}%`;

            if (row) {
                if (className === data.predicted_label) {
                    row.classList.add("active-emotion-row");
                } else {
                    row.classList.remove("active-emotion-row");
                }
            }
        }

        // Phase 4: Update interactive Chart.js probability visualization
        if (typeof updateWebcamProbChart === "function") {
            updateWebcamProbChart(data.probabilities, data.predicted_label);
        }
    }

    // Update quality diagnostics
    if (data.quality) {
        updateQualityBadgeUI(data.quality);
    }

    // Multi-face indicator
    const facesBadge = document.getElementById("webcam-faces-badge");
    const facesVal = document.getElementById("webcam-faces-val");
    if (facesBadge && facesVal) {
        if (data.faces_count > 1) {
            facesBadge.classList.add("is-visible");
            facesVal.textContent = `${data.faces_count} Faces`;
        } else if (data.faces_count === 1) {
            facesBadge.classList.add("is-visible");
            facesVal.textContent = "1 Face";
        } else {
            facesBadge.classList.remove("is-visible");
        }
    }
}

function updateFpsAndLatencyUI(latencyMs, fps) {
    const latencyEl = document.getElementById("webcam-latency-val");
    const fpsEl = document.getElementById("webcam-fps-val");

    if (latencyEl) latencyEl.textContent = `${latencyMs} ms`;
    if (fpsEl) fpsEl.textContent = `${fps} FPS`;
}

function updateQualityBadgeUI(quality) {
    const badge = document.getElementById("webcam-quality-badge");
    const dot = document.getElementById("quality-dot");
    const text = document.getElementById("quality-text");
    if (!badge || !dot || !text) return;

    badge.className = "metric-pill";
    dot.className = "pill-dot";

    if (quality.status === "optimal") {
        badge.classList.add("pill-success");
        dot.classList.add("green");
        text.textContent = "Lighting Optimal";
    } else if (quality.status === "low_light") {
        badge.classList.add("pill-warning");
        dot.classList.add("amber");
        text.textContent = "Low Light";
    } else if (quality.status === "overexposed") {
        badge.classList.add("pill-warning");
        dot.classList.add("amber");
        text.textContent = "Overexposed";
    } else if (quality.status === "blurry") {
        badge.classList.add("pill-warning");
        dot.classList.add("amber");
        text.textContent = "Motion Blur";
    } else {
        badge.classList.add("pill-neutral");
        dot.classList.add("gray");
        text.textContent = quality.message || "Standby";
    }
}

function recordPredictionHistory(data) {
    const timelineContainer = document.getElementById("webcam-timeline-track");
    if (!timelineContainer) return;

    const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const modeName = (data.mode || currentAnalysisMode || "cnn").toLowerCase();

    predictionHistory.push({
        label: data.predicted_label,
        confidence: data.confidence,
        color: data.color || "#64748b",
        is_uncertain: data.is_uncertain,
        mode: modeName,
        time: timeStr
    });

    if (predictionHistory.length > MAX_HISTORY_ITEMS) {
        predictionHistory.shift();
    }

    timelineContainer.innerHTML = "";
    predictionHistory.forEach((item) => {
        const chip = document.createElement("div");
        chip.className = `timeline-chip ${item.is_uncertain ? 'uncertain' : ''}`;
        chip.style.borderLeftColor = item.color;
        const modeTag = `<span class="chip-mode-tag tag-${item.mode}">${item.mode.toUpperCase()}</span>`;
        chip.innerHTML = `
            ${modeTag}
            <span class="chip-label" style="color: ${item.color}">${item.label}</span>
            <span class="chip-conf">${item.confidence}%</span>
            <span class="chip-time">${item.time}</span>
        `;
        timelineContainer.appendChild(chip);
    });

    // Auto scroll smoothly to end
    timelineContainer.scrollLeft = timelineContainer.scrollWidth;

    // Phase 4: Push to interactive Chart.js line timeline
    if (typeof recordTimelineChartPoint === "function") {
        recordTimelineChartPoint(data.predicted_label, data.confidence, modeName, timeStr);
    }
}

window.addEventListener("beforeunload", () => {
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
    }
});
