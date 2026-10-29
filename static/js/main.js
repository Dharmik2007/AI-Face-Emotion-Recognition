/**
 * FaceSense AI – Main Client-side Controller
 * Handles Multi-tab Navigation, Image Upload (JPG, PNG, JPEG, WEBP),
 * Drag-and-Drop, Client Validation, In-Page AJAX Prediction, and Metrics Visualization.
 */

document.addEventListener("DOMContentLoaded", () => {
    initDropZone();
    initUploadForm();
    checkUrlHash();
    loadModelPerformanceMetrics();
});

// Check URL Hash and query param for direct tab activation
function checkUrlHash() {
    const params = new URLSearchParams(window.location.search);
    const tabParam = params.get("tab");
    const validTabs = ["dashboard", "webcam", "upload", "compare", "performance"];

    if (tabParam && validTabs.includes(tabParam)) {
        switchTab(tabParam, false);
        return;
    }

    const hash = window.location.hash;
    if (hash === "#webcam-section" || hash === "#pane-webcam") {
        switchTab("webcam", false);
    } else if (hash === "#upload-section" || hash === "#pane-upload") {
        switchTab("upload", false);
    } else if (hash === "#performance-section" || hash === "#pane-performance") {
        switchTab("performance", false);
    } else if (hash === "#compare-section" || hash === "#pane-compare") {
        switchTab("compare", false);
    } else if (hash === "#dashboard-section" || hash === "#pane-dashboard") {
        switchTab("dashboard", false);
    }
}

// Multi-Tab Switching across 5 application sections
function switchTab(mode, updateUrl = true) {
    const tabs = ["dashboard", "webcam", "upload", "compare", "performance"];
    if (!tabs.includes(mode)) mode = "dashboard";

    tabs.forEach(t => {
        const btn = document.getElementById(`tab-${t}-btn`);
        const pane = document.getElementById(`pane-${t}`);
        if (btn) btn.classList.toggle("active", t === mode);
        if (pane) pane.classList.toggle("active", t === mode);
    });

    // Sync navbar links if present
    const navMapping = {
        "dashboard": "/",
        "webcam": "/live",
        "upload": "/upload",
        "compare": "/compare",
        "performance": "/performance"
    };

    const navLinks = document.querySelectorAll(".nav-links .nav-link");
    navLinks.forEach(link => {
        const href = link.getAttribute("href") || "";
        const expected = navMapping[mode];
        const isMatch = (mode === "dashboard" && (href === "/" || href.endsWith("/"))) ||
                        (expected && href.includes(expected));
        if (expected && !href.includes("about") && !href.includes("demo")) {
            link.classList.toggle("active", isMatch);
        }
    });

    if (updateUrl && window.history && window.history.replaceState) {
        const newUrl = mode === "dashboard" ? "/" : `?tab=${mode}`;
        window.history.replaceState(null, "", newUrl);
    }

    // Force Chart.js resize on tab switch to avoid zero-size rendering
    setTimeout(() => {
        if (typeof Chart !== "undefined") {
            try {
                if (typeof webcamProbChart !== "undefined" && webcamProbChart) webcamProbChart.resize();
                if (typeof webcamTimelineChart !== "undefined" && webcamTimelineChart) webcamTimelineChart.resize();
                if (typeof uploadProbChart !== "undefined" && uploadProbChart) uploadProbChart.resize();
                if (typeof latencyComparisonChart !== "undefined" && latencyComparisonChart) latencyComparisonChart.resize();
                if (typeof perfMetricsChart !== "undefined" && perfMetricsChart) perfMetricsChart.resize();
            } catch (e) {
                console.debug("Chart resize handled", e);
            }
        }
    }, 60);
}

// Drag & Drop Upload Initializer
function initDropZone() {
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    if (!dropZone || !fileInput) return;

    ["dragenter", "dragover", "dragleave", "drop"].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    ["dragenter", "dragover"].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add("dragover"), false);
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove("dragover"), false);
    });

    dropZone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files && files.length > 0) {
            handleFile(files[0]);
        }
    }, false);

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

// File Validation & Preview (Supporting JPG, JPEG, PNG, WEBP)
function handleFile(file) {
    const errorDiv = document.getElementById("upload-error");
    const submitBtn = document.getElementById("submit-btn");
    const promptDiv = document.getElementById("drop-zone-prompt");
    const previewContainer = document.getElementById("preview-container");
    const previewImg = document.getElementById("image-preview");
    const filenameEl = document.getElementById("preview-filename");
    const filesizeEl = document.getElementById("preview-filesize");
    const fileInput = document.getElementById("file-input");

    if (errorDiv) {
        errorDiv.style.display = "none";
        errorDiv.innerText = "";
    }

    // 1. Validate File Type (including WEBP)
    const validTypes = ["image/jpeg", "image/png", "image/jpg", "image/webp"];
    const extension = file.name.split(".").pop().toLowerCase();
    const validExtensions = ["jpg", "jpeg", "png", "webp"];

    if (!validTypes.includes(file.type) && !validExtensions.includes(extension)) {
        showError("Invalid file type. Supported formats: JPG, JPEG, PNG, and WEBP.");
        clearSelectedImage();
        return;
    }

    // 2. Validate File Size (Max 5MB)
    const maxSize = 5 * 1024 * 1024;
    if (file.size > maxSize) {
        showError("File size exceeds 5 MB limit. Please select a smaller image.");
        clearSelectedImage();
        return;
    }

    // Assign file to input
    if (fileInput && fileInput.files[0] !== file) {
        try {
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            fileInput.files = dataTransfer.files;
        } catch (err) {
            console.warn("DataTransfer not supported in this context:", err);
        }
    }

    // 3. Render Preview
    const reader = new FileReader();
    reader.onload = (e) => {
        if (previewImg) previewImg.src = e.target.result;
        if (filenameEl) filenameEl.textContent = file.name;
        if (filesizeEl) filesizeEl.textContent = formatBytes(file.size);

        if (promptDiv) promptDiv.style.display = "none";
        if (previewContainer) previewContainer.style.display = "flex";
        if (submitBtn) submitBtn.disabled = false;
    };
    reader.onerror = () => {
        showError("Failed to read the selected file. Please choose another image.");
        clearSelectedImage();
    };
    reader.readAsDataURL(file);
}

function clearSelectedImage(e) {
    if (e) {
        e.stopPropagation();
        e.preventDefault();
    }
    const fileInput = document.getElementById("file-input");
    const promptDiv = document.getElementById("drop-zone-prompt");
    const previewContainer = document.getElementById("preview-container");
    const previewImg = document.getElementById("image-preview");
    const submitBtn = document.getElementById("submit-btn");
    const errorDiv = document.getElementById("upload-error");
    const inlineResult = document.getElementById("upload-inline-result");

    if (fileInput) fileInput.value = "";
    if (previewImg) previewImg.src = "#";
    if (previewContainer) previewContainer.style.display = "none";
    if (promptDiv) promptDiv.style.display = "block";
    if (submitBtn) submitBtn.disabled = true;
    if (errorDiv) {
        errorDiv.style.display = "none";
        errorDiv.innerText = "";
    }
    if (inlineResult) inlineResult.style.display = "none";
}

function showError(msg) {
    const errorDiv = document.getElementById("upload-error");
    if (errorDiv) {
        errorDiv.innerText = msg;
        errorDiv.style.display = "block";
    }
}

function formatBytes(bytes, decimals = 1) {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ["Bytes", "KB", "MB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + " " + sizes[i];
}

// Handle Form Submission with AJAX option for immediate in-page display
function initUploadForm() {
    const form = document.getElementById("upload-form");
    const submitBtn = document.getElementById("submit-btn");
    if (!form || !submitBtn) return;

    form.addEventListener("submit", async (e) => {
        // Use AJAX for smooth in-page display
        e.preventDefault();

        const fileInput = document.getElementById("file-input");
        if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
            showError("Please select a valid image file before analyzing.");
            return;
        }

        submitBtn.disabled = true;
        const btnText = submitBtn.querySelector(".btn-text");
        const btnSpinner = submitBtn.querySelector(".btn-spinner");
        if (btnText) btnText.style.display = "none";
        if (btnSpinner) btnSpinner.style.display = "inline-flex";

        const formData = new FormData(form);

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: {
                    "X-Requested-With": "XMLHttpRequest"
                },
                body: formData
            });

            const data = await response.json();

            if (response.status === 200 && data.status === "success") {
                renderUploadInlineResult(data);
            } else {
                showError(data.message || "Prediction failed. Please try another image.");
            }
        } catch (err) {
            console.error("Upload error:", err);
            // Fallback to normal form submit if AJAX fails
            form.submit();
        } finally {
            submitBtn.disabled = false;
            if (btnText) btnText.style.display = "inline-flex";
            if (btnSpinner) btnSpinner.style.display = "none";
        }
    });
}

function renderUploadInlineResult(data) {
    const inlineResult = document.getElementById("upload-inline-result");
    if (!inlineResult) return;

    inlineResult.style.display = "block";
    inlineResult.scrollIntoView({ behavior: "smooth", block: "start" });

    // Set images
    const annotatedImg = document.getElementById("inline-annotated-img");
    const cropImg = document.getElementById("inline-crop-img");
    if (annotatedImg) annotatedImg.src = data.annotated_image;
    if (cropImg) cropImg.src = data.crop_image;

    // Top prediction
    const predLabel = document.getElementById("inline-pred-label");
    const confVal = document.getElementById("inline-conf-val");
    const descText = document.getElementById("inline-desc-text");
    const heroCard = document.getElementById("inline-hero-card");
    const qualityPill = document.getElementById("inline-quality-pill");
    const multiFaceBadge = document.getElementById("inline-multiface-badge");

    if (predLabel) predLabel.textContent = data.predicted_label;
    if (confVal) confVal.textContent = `${data.confidence}% Confidence`;
    if (descText) descText.textContent = data.description || "";

    if (heroCard) {
        heroCard.style.borderColor = data.color || "#4f46e5";
        heroCard.style.boxShadow = `0 8px 30px -4px ${data.color}33`;
    }

    if (multiFaceBadge) {
        if (data.faces_count > 1) {
            multiFaceBadge.style.display = "inline-flex";
            multiFaceBadge.textContent = `${data.faces_count} Faces Detected in Photo`;
        } else {
            multiFaceBadge.style.display = "none";
        }
    }

    if (qualityPill && data.quality) {
        qualityPill.innerHTML = data.quality.status === "optimal"
            ? `<span class="pill-dot green"></span> Illumination Optimal`
            : `<span class="pill-dot amber"></span> ${data.quality.message}`;
    }

    // Probability Bars
    if (data.probabilities) {
        for (const [name, prob] of Object.entries(data.probabilities)) {
            const bar = document.getElementById(`inline-bar-${name}`);
            const val = document.getElementById(`inline-val-${name}`);
            const row = document.getElementById(`inline-row-${name}`);

            if (bar) bar.style.width = `${prob}%`;
            if (val) val.textContent = `${prob.toFixed(1)}%`;
            if (row) {
                if (name === data.predicted_label) {
                    row.classList.add("active-emotion-row");
                } else {
                    row.classList.remove("active-emotion-row");
                }
            }
        }

        // Phase 4: Update interactive Chart.js probability visualization
        if (typeof updateUploadProbChart === "function") {
            updateUploadProbChart(data.probabilities, data.predicted_label);
        }
    }
}

// Load pre-generated sample face image directly into drop zone
async function loadSampleFace(url, filename) {
    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error("Failed to load sample face.");
        const blob = await response.blob();
        const file = new File([blob], filename, { type: "image/png" });
        handleFile(file);
    } catch (err) {
        console.error("Error loading sample image:", err);
        showError("Could not load sample face. Please upload a file manually.");
    }
}

// Load real model performance metrics for the Performance tab
async function loadModelPerformanceMetrics() {
    try {
        const response = await fetch("/api/model_info");
        if (!response.ok) return;
        const data = await response.json();

        if (data.evaluation_metrics) {
            renderPerformanceMetricsUI(data.evaluation_metrics);
        }
    } catch (err) {
        console.warn("Could not load model info:", err);
    }
}

function renderPerformanceMetricsUI(metrics) {
    const testAccEl = document.getElementById("metric-test-accuracy");
    const testLossEl = document.getElementById("metric-test-loss");
    const totalSamplesEl = document.getElementById("metric-test-samples");
    const datasetNameEl = document.getElementById("metric-dataset-name");

    if (testAccEl) testAccEl.textContent = `${metrics.test_accuracy}%`;
    if (testLossEl) testLossEl.textContent = metrics.test_loss;
    if (totalSamplesEl) totalSamplesEl.textContent = metrics.total_test_samples.toLocaleString();
    if (datasetNameEl) datasetNameEl.textContent = metrics.dataset_name;

    // Render Per-Class Precision / Recall / F1 table
    const tableBody = document.getElementById("metrics-table-body");
    if (tableBody && metrics.per_class) {
        tableBody.innerHTML = "";
        for (const [name, stats] of Object.entries(metrics.per_class)) {
            const row = document.createElement("tr");
            row.innerHTML = `
                <td><strong>${name}</strong></td>
                <td>${stats.precision}%</td>
                <td>${stats.recall}%</td>
                <td><span class="badge badge-primary">${stats.f1_score}%</span></td>
                <td>${stats.support}</td>
            `;
            tableBody.appendChild(row);
        }
    }
}
