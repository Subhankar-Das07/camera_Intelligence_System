document.addEventListener("DOMContentLoaded", () => {
    // ── DOM refs ──────────────────────────────────────────────────────────────
    const videoUpload       = document.getElementById("video-upload");
    const uploadBtn         = document.getElementById("upload-btn");
    const uploadStatus      = document.getElementById("upload-status");

    const placeholderMsg    = document.getElementById("placeholder-msg");
    const videoContainer    = document.getElementById("video-container");
    const mainVideoImg      = document.getElementById("main-video-img");
    const roiCanvas         = document.getElementById("roi-canvas");
    const ctx               = roiCanvas.getContext("2d");

    const clearRoiBtn       = document.getElementById("clear-roi-btn");
    const pipelineSelect    = document.getElementById("pipeline-select");
    const runBtn            = document.getElementById("run-btn");
    const stopBtn           = document.getElementById("stop-btn");
    const alertsContainer   = document.getElementById("alerts-container");

    const rtspUrlInput      = document.getElementById("rtsp-url");
    const connectRtspBtn    = document.getElementById("connect-rtsp-btn");
    const disconnectRtspBtn = document.getElementById("disconnect-rtsp-btn");

    // ── State ─────────────────────────────────────────────────────────────────
    let currentVideoData  = null;   // {video_id, filename, width, height, preview_url...}
    let currentStreamId   = null;   // RTSP stream ID (raw stream)
    let currentSessionId  = null;   // Analysis session ID
    let isRtspMode        = false;  // true when using live RTSP

    let roiPoints    = [];
    let imageWidth   = 0;
    let imageHeight  = 0;

    let pollingInterval = null;
    let knownAlerts = new Set();

    // ── Fetch Pipelines ───────────────────────────────────────────────────────
    fetch("/api/pipelines")
        .then(res => res.json())
        .then(data => {
            pipelineSelect.innerHTML = "";
            data.pipelines
                .filter(p => p !== "face_recognition")   // FR has its own dedicated page
                .forEach(p => {
                    const opt = document.createElement("option");
                    opt.value = p; opt.textContent = p;
                    pipelineSelect.appendChild(opt);
                });
            checkRunReady();
        })
        .catch(err => console.error("Error fetching pipelines:", err));

    // ── File Upload ───────────────────────────────────────────────────────────
    uploadBtn.addEventListener("click", () => videoUpload.click());

    videoUpload.addEventListener("change", (e) => {
        if (e.target.files.length) handleFileUpload(e.target.files[0]);
    });

    function handleFileUpload(file) {
        const formData = new FormData();
        formData.append("file", file);
        uploadStatus.textContent = "Uploading...";
        uploadStatus.style.color = "#94a3b8";
        uploadBtn.disabled = true;

        fetch("/api/upload", { method: "POST", body: formData })
            .then(res => { if (!res.ok) throw new Error("Upload failed."); return res.json(); })
            .then(data => {
                disconnectCurrentRtsp();
                currentVideoData = data;
                isRtspMode = false;
                uploadStatus.textContent = file.name;
                uploadStatus.style.color = "#10b981";
                uploadBtn.disabled = false;
                roiPoints = [];
                setupPreview(data.preview_url, data.width, data.height);
            })
            .catch(err => {
                console.error(err);
                uploadStatus.textContent = "Error";
                uploadStatus.style.color = "#ef4444";
                uploadBtn.disabled = false;
            });
    }

    // ── RTSP Connect ──────────────────────────────────────────────────────────
    connectRtspBtn.addEventListener("click", () => {
        const url = rtspUrlInput.value.trim();
        if (!url) { rtspUrlInput.style.borderColor = "#ef4444"; return; }
        rtspUrlInput.style.borderColor = "#475569";
        connectRtspBtn.disabled = true;
        connectRtspBtn.textContent = "Connecting...";

        fetch("/api/connect_stream", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url })
        })
        .then(res => {
            if (!res.ok) return res.json().then(e => { throw new Error(e.detail || "Failed."); });
            return res.json();
        })
        .then(data => {
            currentStreamId  = data.stream_id;
            currentVideoData = { video_id: data.stream_id, filename: url,
                                 width: data.width, height: data.height,
                                 preview_url: data.preview_url };
            isRtspMode = true;

            connectRtspBtn.textContent = "✓ Connected";
            uploadStatus.textContent = "Using RTSP stream";
            uploadStatus.style.color = "#10b981";
            disconnectRtspBtn.classList.remove("hidden");

            roiPoints = [];
            
            // Prefer the static preview frame if available, otherwise fallback to the live raw stream.
            // Both display exactly the same way in the new unified container!
            const srcUrl = data.preview_url ? data.preview_url : `/api/raw_stream/${data.stream_id}`;
            setupPreview(srcUrl, data.width, data.height);
        })
        .catch(err => {
            connectRtspBtn.textContent = "Connect Stream";
            connectRtspBtn.disabled = false;
            alert("❌ Stream Error: " + err.message);
        });
    });

    disconnectRtspBtn.addEventListener("click", () => {
        disconnectCurrentRtsp();
        resetToPlaceholder();
    });

    function disconnectCurrentRtsp() {
        if (!currentStreamId) return;
        fetch(`/api/stream/${currentStreamId}`, { method: "DELETE" }).catch(() => {});
        currentStreamId = null;
        isRtspMode = false;
        connectRtspBtn.textContent = "Connect Stream";
        connectRtspBtn.disabled = false;
        disconnectRtspBtn.classList.add("hidden");
    }

    // ── Canvas / Preview setup ────────────────────────────────────────────────
    function setupPreview(srcUrl, width, height) {
        stopAnalysisMode(false);
        placeholderMsg.classList.add("hidden");
        
        // Ensure cache bust for static previews
        const finalUrl = srcUrl.includes('/api/raw_stream') ? srcUrl : srcUrl + "?t=" + Date.now();
        mainVideoImg.src = finalUrl;
        
        // Set internal canvas resolution to match actual video pixels exactly
        roiCanvas.width = width;
        roiCanvas.height = height;
        imageWidth = width;
        imageHeight = height;
        
        videoContainer.classList.remove("hidden");
        
        roiPoints = [];
        
        // Wait for image to load before positioning canvas
        mainVideoImg.onload = alignCanvas;
        
        // If it's an MJPEG stream, onload might not fire reliably, so we force alignment after a short delay
        setTimeout(alignCanvas, 100);
        setTimeout(alignCanvas, 500);

        drawPoly();
        checkRunReady();
    }

    // Bulletproof alignment: mathematically calculate where the image is rendered inside the container
    // due to object-fit: contain, and position the canvas exactly over it.
    function alignCanvas() {
        if (!imageWidth || !imageHeight) return;
        
        const rect = mainVideoImg.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return;

        const imgRatio = imageWidth / imageHeight;
        const boxRatio = rect.width / rect.height;
        
        let renderWidth, renderHeight, offsetX, offsetY;

        if (imgRatio > boxRatio) {
            // Image is wider than the box, so letterboxed top/bottom
            renderWidth = rect.width;
            renderHeight = rect.width / imgRatio;
            offsetX = 0;
            offsetY = (rect.height - renderHeight) / 2;
        } else {
            // Image is taller than the box, so letterboxed left/right
            renderHeight = rect.height;
            renderWidth = rect.height * imgRatio;
            offsetX = (rect.width - renderWidth) / 2;
            offsetY = 0;
        }

        roiCanvas.style.width = renderWidth + 'px';
        roiCanvas.style.height = renderHeight + 'px';
        roiCanvas.style.left = offsetX + 'px';
        roiCanvas.style.top = offsetY + 'px';
    }

    window.addEventListener('resize', alignCanvas);

    // ── ROI Drawing ───────────────────────────────────────────────────────────

    roiCanvas.addEventListener("click", (e) => {
        const rect = roiCanvas.getBoundingClientRect();
        roiPoints.push({
            x: (e.clientX - rect.left) * (roiCanvas.width / rect.width),
            y: (e.clientY - rect.top)  * (roiCanvas.height / rect.height)
        });
        drawPoly();
        checkRunReady();
    });

    clearRoiBtn.addEventListener("click", () => {
        roiPoints = [];
        drawPoly();
        checkRunReady();
    });

    function drawPoly() {
        ctx.clearRect(0, 0, roiCanvas.width, roiCanvas.height);
        if (roiPoints.length === 0) return;
        
        ctx.beginPath();
        ctx.moveTo(roiPoints[0].x, roiPoints[0].y);
        for (let i = 1; i < roiPoints.length; i++) ctx.lineTo(roiPoints[i].x, roiPoints[i].y);
        
        if (roiPoints.length > 2) {
            ctx.lineTo(roiPoints[0].x, roiPoints[0].y);
            ctx.fillStyle = "rgba(0, 255, 255, 0.15)";
            ctx.fill();
        }
        
        ctx.strokeStyle = "#00ffff";
        ctx.lineWidth = 2;
        ctx.stroke();
        
        roiPoints.forEach(p => {
            ctx.beginPath();
            ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
            ctx.fillStyle = "#ff0044";
            ctx.fill();
        });
    }

    function checkRunReady() {
        const needsRoi = pipelineSelect.value !== "fall_detection";
        const hasRoi = roiPoints.length > 2;
        runBtn.disabled = !( (needsRoi ? hasRoi : true) && pipelineSelect.value && currentVideoData);
    }

    pipelineSelect.addEventListener("change", (e) => {
        const isFallDetection = e.target.value === "fall_detection";
        const machineSettings = document.getElementById("machine-settings");
        const roiControls = document.getElementById("roi-controls");
        
        if (machineSettings) machineSettings.style.display = isFallDetection ? "none" : "block";
        if (roiControls) {
            roiControls.style.display = isFallDetection ? "none" : "block";
            roiCanvas.style.display = isFallDetection ? "none" : "block";
        }
        checkRunReady();
    });

    // ── Analysis ──────────────────────────────────────────────────────────────

    runBtn.addEventListener("click", () => {
        const roiNormalized = roiPoints.map(p => [p.x / imageWidth, p.y / imageHeight]);
        const machineActive = document.getElementById("machine-active-toggle")?.checked || false;

        const payload = {
            video_id:      currentVideoData.video_id,
            filename:      currentVideoData.filename,
            pipeline_name: pipelineSelect.value,
            roi_normalized: roiNormalized,
            config: { machine_active: machineActive },
            stream_id:     currentStreamId || null
        };

        runBtn.disabled = true;

        fetch("/api/start_analysis", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        })
        .then(res => { if (!res.ok) throw new Error("Failed to start analysis"); return res.json(); })
        .then(data => {
            currentSessionId = data.session_id;
            startAnalysisMode();
        })
        .catch(err => {
            console.error(err);
            alert("Error starting analysis.");
            runBtn.disabled = false;
        });
    });

    stopBtn.addEventListener("click", () => stopAnalysisMode(true));

    function startAnalysisMode() {
        runBtn.classList.add("hidden");
        stopBtn.classList.remove("hidden");

        // Switch the main image source to the annotated stream!
        mainVideoImg.src = `/api/stream/${currentSessionId}`;
        
        // Hide ROI canvas so it doesn't duplicate what the AI is drawing on the frame
        roiCanvas.classList.add("hidden");

        alertsContainer.innerHTML = '<p class="placeholder">No alerts detected yet.</p>';
        knownAlerts.clear();

        pollingInterval = setInterval(pollAlerts, 1500);
    }

    function stopAnalysisMode(callApi = true) {
        if (pollingInterval) { clearInterval(pollingInterval); pollingInterval = null; }

        if (callApi && currentSessionId) {
            fetch(`/api/stop_analysis/${currentSessionId}`, { method: "POST" }).catch(() => {});
        }

        runBtn.classList.remove("hidden");
        stopBtn.classList.add("hidden");

        // Unhide our drawing canvas
        roiCanvas.classList.remove("hidden");

        // Restore the preview source
        if (currentVideoData) {
            if (isRtspMode && currentStreamId) {
                // If it's RTSP, switch back to raw stream or preview
                const srcUrl = currentVideoData.preview_url ? currentVideoData.preview_url : `/api/raw_stream/${currentStreamId}`;
                mainVideoImg.src = srcUrl.includes('/api/raw_stream') ? srcUrl : srcUrl + "?t=" + Date.now();
            } else if (currentVideoData.preview_url) {
                // File upload preview
                mainVideoImg.src = currentVideoData.preview_url + "?t=" + Date.now();
            }
        }

        checkRunReady();
    }

    function resetToPlaceholder() {
        stopAnalysisMode(false);
        videoContainer.classList.add("hidden");
        mainVideoImg.src = "";
        placeholderMsg.classList.remove("hidden");
        currentVideoData = null;
        roiPoints = [];
        checkRunReady();
    }

    // ── Alert Polling ─────────────────────────────────────────────────────────
    function pollAlerts() {
        if (!currentSessionId) return;
        fetch(`/api/alerts/${currentSessionId}`)
            .then(res => res.json())
            .then(data => {
                if (data.alerts && data.alerts.length > 0) {
                    const placeholder = alertsContainer.querySelector('.placeholder');
                    if (placeholder) placeholder.remove();
                    data.alerts.forEach(alert => {
                        if (!knownAlerts.has(alert.id)) {
                            knownAlerts.add(alert.id);
                            createAlertCard(alert);
                            if (alert.immediate_buzzer_trigger) playBuzzer(3);
                        }
                    });
                }
            })
            .catch(err => console.error("Error polling alerts:", err));
    }

    function playBuzzer(times) {
        if (times <= 0) return;
        const audio = new Audio('/assets/buzzer.mp3');
        audio.play().catch(e => console.error("Audio play failed:", e));
        audio.onended = () => playBuzzer(times - 1);
    }

    function createAlertCard(alert) {
        const card = document.createElement("div");
        card.className = "alert-card";
        let label = "⚠️ Intrusion";
        let color = "var(--text-primary)";
        if (alert.severity) {
            label = alert.severity === "SEVERE" ? "🚨 SEVERE DANGER" : "⚠️ WARNING (NEAR)";
            color = alert.severity === "SEVERE" ? "#ff4444" : "#ffa500";
        }
        card.innerHTML = `
            <div class="alert-time" style="color:${color};font-weight:600;">${label} @ ${alert.formatted_time}</div>
            <video class="alert-inline-video" src="${alert.clip_url}" controls autoplay loop muted></video>
        `;
        alertsContainer.prepend(card);
    }
});

// ═══════════════════════════════════════════════════════════════════════════
//  Tab switching + Vehicle Recognition pipeline controller
// ═══════════════════════════════════════════════════════════════════════════
document.addEventListener("DOMContentLoaded", () => {

    // ── Tab switcher ──────────────────────────────────────────────────────
    const tabBtns   = document.querySelectorAll(".tab-btn");
    const tabPanels = document.querySelectorAll(".tab-panel");

    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const target = btn.dataset.tab;

            tabBtns.forEach(b => b.classList.toggle("active", b.dataset.tab === target));
            tabPanels.forEach(p => {
                const isTarget = p.id === `panel-${target}`;
                p.classList.toggle("hidden", !isTarget);
            });
        });
    });

    // ── Vehicle Recognition DOM refs ──────────────────────────────────────
    const vrConnectBtn = document.getElementById("vr-connect-btn");
    const vrRunBtn     = document.getElementById("vr-run-btn");
    const vrStopBtn    = document.getElementById("vr-stop-btn");
    const vrStatusMsg  = document.getElementById("vr-status-msg");
    const vrStreamImg  = document.getElementById("vr-stream-img");
    const vrPlaceholder = document.getElementById("vr-placeholder");
    const vrLogContainer = document.getElementById("vr-log-container");

    // ── VR State ──────────────────────────────────────────────────────────
    let vrVideoData    = null;  // {video_id, filename, width, height}
    let vrStreamId     = null;  // RTSP stream_id (if using live camera)
    let vrSessionId    = null;  // analysis session_id
    let vrIsRtsp       = false;
    let vrPollInterval = null;
    const vrSeenPlates = new Set();

    // ── Connect Source (re-uses main upload flow via a hidden file input) ─
    // We create a one-off file input so the VR tab has its own upload button
    const vrFileInput = document.createElement("input");
    vrFileInput.type  = "file";
    vrFileInput.accept = "video/mp4,video/avi,video/quicktime";
    vrFileInput.style.display = "none";
    document.body.appendChild(vrFileInput);

    vrConnectBtn.addEventListener("click", () => {
        // Offer: file OR rtsp prompt
        const choice = confirm(
            "Click OK to upload a video file.\nClick Cancel to enter an RTSP URL."
        );
        if (choice) {
            vrFileInput.click();
        } else {
            const url = prompt("Enter RTSP / webcam URL (or 0 for local webcam):");
            if (url !== null && url.trim() !== "") {
                vrConnectRtsp(url.trim());
            }
        }
    });

    vrFileInput.addEventListener("change", (e) => {
        if (!e.target.files.length) return;
        vrUploadFile(e.target.files[0]);
        vrFileInput.value = "";   // reset so re-selecting same file triggers change
    });

    function vrUploadFile(file) {
        vrStatusMsg.textContent = "Uploading...";
        vrStatusMsg.style.color = "#94a3b8";
        vrConnectBtn.disabled = true;

        const fd = new FormData();
        fd.append("file", file);

        fetch("/api/upload", { method: "POST", body: fd })
            .then(r => { if (!r.ok) throw new Error("Upload failed."); return r.json(); })
            .then(data => {
                vrVideoData = data;
                vrIsRtsp    = false;
                vrStreamId  = null;
                vrStatusMsg.textContent = `✔ ${file.name}`;
                vrStatusMsg.style.color = "#10b981";
                vrConnectBtn.disabled   = false;
                vrRunBtn.disabled       = false;

                // Show static preview
                vrPlaceholder.classList.add("hidden");
                vrStreamImg.src = data.preview_url + "?t=" + Date.now();
                vrStreamImg.classList.remove("hidden");
            })
            .catch(err => {
                vrStatusMsg.textContent = "Upload error";
                vrStatusMsg.style.color = "#ef4444";
                vrConnectBtn.disabled   = false;
                console.error(err);
            });
    }

    function vrConnectRtsp(url) {
        vrStatusMsg.textContent = "Connecting...";
        vrStatusMsg.style.color = "#94a3b8";
        vrConnectBtn.disabled = true;

        fetch("/api/connect_stream", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url })
        })
        .then(r => { if (!r.ok) return r.json().then(e => { throw new Error(e.detail); }); return r.json(); })
        .then(data => {
            vrStreamId  = data.stream_id;
            vrVideoData = { video_id: data.stream_id, filename: url,
                            width: data.width, height: data.height };
            vrIsRtsp    = true;
            vrStatusMsg.textContent = "✔ RTSP connected";
            vrStatusMsg.style.color = "#10b981";
            vrConnectBtn.disabled   = false;
            vrRunBtn.disabled       = false;

            // Show raw preview stream
            vrPlaceholder.classList.add("hidden");
            vrStreamImg.src = `/api/raw_stream/${data.stream_id}`;
            vrStreamImg.classList.remove("hidden");
        })
        .catch(err => {
            vrStatusMsg.textContent = "Connection failed";
            vrStatusMsg.style.color = "#ef4444";
            vrConnectBtn.disabled   = false;
            alert("❌ " + err.message);
        });
    }

    // ── Run ───────────────────────────────────────────────────────────────
    vrRunBtn.addEventListener("click", () => {
        if (!vrVideoData) return;

        const payload = {
            video_id:       vrVideoData.video_id,
            filename:       vrVideoData.filename,
            pipeline_name:  "vehicle_recognition",
            roi_normalized: [],   // vehicle recognition needs no ROI polygon
            config:         {},
            stream_id:      vrStreamId || null
        };

        vrRunBtn.disabled = true;

        fetch("/api/start_analysis", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        })
        .then(r => { if (!r.ok) throw new Error("Failed to start."); return r.json(); })
        .then(data => {
            vrSessionId = data.session_id;
            vrStartMode();
        })
        .catch(err => {
            console.error(err);
            alert("Error starting vehicle recognition: " + err.message);
            vrRunBtn.disabled = false;
        });
    });

    // ── Stop ──────────────────────────────────────────────────────────────
    vrStopBtn.addEventListener("click", () => vrStopMode(true));

    function vrStartMode() {
        vrRunBtn.classList.add("hidden");
        vrStopBtn.classList.remove("hidden");
        vrStatusMsg.textContent = "🔴 Live";
        vrStatusMsg.style.color = "#ef4444";

        // Switch img to the annotated AI stream
        vrStreamImg.src = `/api/stream/${vrSessionId}`;
        vrStreamImg.classList.remove("hidden");
        vrPlaceholder.classList.add("hidden");

        vrLogContainer.innerHTML = '<p class="placeholder">Scanning…</p>';
        vrSeenPlates.clear();

        // Poll the session metadata for new detections
        vrPollInterval = setInterval(vrPollDetections, 2000);
    }

    function vrStopMode(callApi = true) {
        if (vrPollInterval) { clearInterval(vrPollInterval); vrPollInterval = null; }

        if (callApi && vrSessionId) {
            fetch(`/api/stop_analysis/${vrSessionId}`, { method: "POST" }).catch(() => {});
        }
        vrSessionId = null;

        vrRunBtn.classList.remove("hidden");
        vrRunBtn.disabled = false;
        vrStopBtn.classList.add("hidden");
        vrStatusMsg.textContent = vrVideoData ? "⏹ Stopped" : "No source selected";
        vrStatusMsg.style.color = "#94a3b8";
    }

    // ── Detection polling (reads frame_metadata via /api/alerts endpoint) ─
    // VehicleRecognitionPipeline emits no "alert" events; instead we parse
    // the detections out of a lightweight dedicated endpoint below.
    function vrPollDetections() {
        if (!vrSessionId) return;

        // We re-use the existing /api/alerts endpoint which returns []
        // for vehicle_recognition (no alert events). For actual detection
        // cards we call a thin metadata endpoint we add to main.py.
        fetch(`/api/vr_detections/${vrSessionId}`)
            .then(r => r.ok ? r.json() : null)
            .then(data => {
                if (!data || !data.detections) return;
                const placeholder = vrLogContainer.querySelector(".placeholder");
                data.detections.forEach(det => {
                    if (!det.plate || vrSeenPlates.has(det.plate)) return;
                    vrSeenPlates.add(det.plate);
                    if (placeholder) placeholder.remove();
                    vrLogContainer.prepend(createVrCard(det));
                });
            })
            .catch(() => {});
    }

    function createVrCard(det) {
        const card = document.createElement("div");
        card.className = "vr-card";
        card.innerHTML = `
            <div class="vr-plate">🚗 ${det.plate}</div>
            <div class="vr-visits">Total Visits: <strong>${det.total_visits}</strong></div>
            <div class="vr-time">${new Date().toLocaleTimeString()}</div>
        `;
        return card;
    }
});
