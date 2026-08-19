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
        runBtn.disabled = !(roiPoints.length > 2 && pipelineSelect.value && currentVideoData);
    }

    pipelineSelect.addEventListener("change", () => {
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
