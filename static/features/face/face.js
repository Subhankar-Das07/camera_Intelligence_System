/**
 * fr_app.js — Face Recognition Dashboard JavaScript
 *
 * Fix log:
 *   - Webcam: now calls /api/connect_webcam to get a stream_id (integer device index)
 *     before calling start_analysis. Webcam is released on stop.
 *   - All source tabs wired correctly (webcam / RTSP / upload)
 *   - Stop cleans up webcam stream
 *   - Snapshot registration gated on any active stream_id
 *   - Identity filter, rename modal, stats, events all verified
 */

"use strict";

document.addEventListener("DOMContentLoaded", () => {

    // ── DOM refs ───────────────────────────────────────────────────────────────
    const statusDot      = document.getElementById("status-dot");
    const statusText     = document.getElementById("status-text");

    const statTotal      = document.getElementById("stat-total");
    const statKnown      = document.getElementById("stat-known");
    const statUnknown    = document.getElementById("stat-unknown"); // optional
    const statVectors    = document.getElementById("stat-vectors");

    const tabs           = document.querySelectorAll(".fr-tab");
    const tabContents    = document.querySelectorAll(".fr-tab-content");

    const webcamIndex    = document.getElementById("webcam-index");
    const frRtspUrl      = document.getElementById("fr-rtsp-url");
    const frConnectBtn   = document.getElementById("fr-connect-rtsp-btn");
    const frDisconnBtn   = document.getElementById("fr-disconnect-rtsp-btn");
    const frVideoUpload  = document.getElementById("fr-video-upload");
    const frUploadBtn    = document.getElementById("fr-upload-btn");
    const frUploadStatus = document.getElementById("fr-upload-status");

    const frStartBtn     = document.getElementById("fr-start-btn");
    const frStopBtn      = document.getElementById("fr-stop-btn");

    const videoWrapper   = document.getElementById("fr-video-wrapper");
    const placeholder    = document.getElementById("fr-placeholder");
    const streamImg      = document.getElementById("fr-stream-img");

    const eventsList     = document.getElementById("fr-events-list");
    const eventsPlaceholder = document.getElementById("fr-events-placeholder");

    const regName        = document.getElementById("reg-name");
    const regPhoto       = document.getElementById("reg-photo");
    const regDropZone    = document.getElementById("reg-drop-zone");
    const regPreview     = document.getElementById("reg-preview");
    const regPreviewImg  = document.getElementById("reg-preview-img");
    const regSnapshotBtn = document.getElementById("reg-snapshot-btn");
    const regSubmitBtn   = document.getElementById("reg-submit-btn");
    const regFeedback    = document.getElementById("reg-feedback");

    const identityList   = document.getElementById("identity-list");
    const pendingList    = document.getElementById("pending-list");
    const refreshBtn     = document.getElementById("refresh-identities-btn");
    const filterTabs     = document.querySelectorAll(".fr-filter-tab");
    const filterKnownBtn = document.getElementById("filter-known-btn");
    const filterPendingBtn = document.getElementById("filter-pending-btn");

    const frAdminSwitchBtn = document.getElementById("fr-admin-switch-btn");
    const frReportsSwitchBtn = document.getElementById("fr-reports-switch-btn");
    const frAdminBackBtn   = document.getElementById("fr-admin-back-btn");
    const frReportsBackBtn = document.getElementById("fr-reports-back-btn");
    const frAdminEntryHeader = document.getElementById("fr-admin-entry-header");
    const frAdminContent   = document.getElementById("fr-admin-content");
    const frReportsContent = document.getElementById("fr-reports-content");
    
    const reportsList      = document.getElementById("reports-list");
    const refreshReportsBtn= document.getElementById("refresh-reports-btn");
    
    const reportModal      = document.getElementById("report-modal");
    const closeReportModal = document.getElementById("close-report-modal");
    const reportModalTitle = document.getElementById("report-modal-title");
    const reportModalMeta  = document.getElementById("report-modal-meta");
    const reportModalBody  = document.getElementById("report-modal-body");

    const renameModal    = document.getElementById("rename-modal");
    const renameInput    = document.getElementById("rename-input");
    const renameConfirm  = document.getElementById("rename-confirm-btn");
    const renameCancel   = document.getElementById("rename-cancel-btn");
    const renameModalPid = document.getElementById("rename-modal-pid");

    const modeSelector       = document.getElementById("mode-selector");
    const visitorControls    = document.getElementById("fr-visitor-controls");
    const attendanceControls = document.getElementById("fr-attendance-controls");
    const visionWatchControls = document.getElementById("fr-vision-watch-controls");
    const rightPanelTitle    = document.querySelector(".fr-identity-panel .fr-panel-title");
    
    // Vision Watch elements
    const vwWatchmanName = document.getElementById("vw-watchman-name");
    const vwClearRoiBtn = document.getElementById("vw-clear-roi-btn");
    const frRoiCanvas = document.getElementById("fr-roi-canvas");
    const vwCtx = frRoiCanvas.getContext("2d");
    const buzzerAudio = new Audio("/assets/buzzer.mp3");

    // Admin toggling
    const adminSwitchBtn     = document.getElementById("fr-admin-switch-btn");
    const adminBackBtn       = document.getElementById("fr-admin-back-btn");
    const adminEntryHeader   = document.getElementById("fr-admin-entry-header");
    const adminContent       = document.getElementById("fr-admin-content");

    if (adminSwitchBtn && adminBackBtn && adminEntryHeader && adminContent) {
        frAdminSwitchBtn.addEventListener("click", () => {
            frAdminEntryHeader.classList.add("hidden");
            frAdminContent.classList.remove("hidden");
            frReportsContent.classList.add("hidden");
            loadIdentities();
        });
        
        frReportsSwitchBtn.addEventListener("click", () => {
            frAdminEntryHeader.classList.add("hidden");
            frAdminContent.classList.add("hidden");
            frReportsContent.classList.remove("hidden");
            loadReports();
        });

        frAdminBackBtn.addEventListener("click", () => {
            frAdminContent.classList.add("hidden");
            frAdminEntryHeader.classList.remove("hidden");
        });
        
        frReportsBackBtn.addEventListener("click", () => {
            frReportsContent.classList.add("hidden");
            frAdminEntryHeader.classList.remove("hidden");
        });
        
        refreshReportsBtn.addEventListener("click", () => {
            loadReports();
        });
        
        closeReportModal.addEventListener("click", () => {
            reportModal.classList.add("hidden");
        });
    }

    const attRegName        = document.getElementById("att-reg-name");
    const attRegSnapshotBtn = document.getElementById("att-reg-snapshot-btn");
    const attRegSubmitBtn   = document.getElementById("att-reg-submit-btn");
    const attRegFeedback    = document.getElementById("att-reg-feedback");

    // ── State ──────────────────────────────────────────────────────────────────
    let activeTab        = "webcam";
    let currentStreamId  = null;   // stream_id for webcam/RTSP (used by pipeline)
    let currentVideoData = null;   // {video_id, filename, width, height}
    let currentSessionId = null;   // analysis session
    let pollingInterval  = null;
    let knownEventIds    = new Set();

    // Vision Watch ROI logic — single door zone polygon
    let vwDoorZones     = [];  // completed door polygons
    let currentDoorZone = [];  // in-progress door polygon
    let doorPollInterval = null;

    function alignCanvas() {
        if (!frRoiCanvas || frRoiCanvas.classList.contains("hidden")) return;
        const rect = streamImg.getBoundingClientRect();
        if (rect.width === 0) return;
        frRoiCanvas.style.left = streamImg.offsetLeft + "px";
        frRoiCanvas.style.top  = streamImg.offsetTop  + "px";
        frRoiCanvas.width  = rect.width;
        frRoiCanvas.height = rect.height;
        drawVwPolygons();
    }
    window.addEventListener("resize", alignCanvas);
    streamImg.onload = alignCanvas;

    frRoiCanvas.addEventListener("click", (e) => {
        if (getMode() !== "vision_watch") return;
        const rect = frRoiCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Close polygon if clicking near first point
        if (currentDoorZone.length > 2) {
            const first = currentDoorZone[0];
            if (Math.hypot(first.x - x, first.y - y) < 15) {
                vwDoorZones = [currentDoorZone]; // Only keep ONE door zone
                currentDoorZone = [];
                drawVwPolygons();
                _updateDoorHint("drawing");
                return;
            }
        }
        currentDoorZone.push({ x, y });
        drawVwPolygons();
    });

    vwClearRoiBtn.addEventListener("click", () => {
        vwDoorZones     = [];
        currentDoorZone = [];
        drawVwPolygons();
        _updateDoorHint("idle");
    });

    function _updateDoorHint(status, data) {
        const hint = document.getElementById("vw-door-status-hint");
        if (!hint) return;
        if (status === "idle")      hint.innerHTML = "🟡 Draw a zone, then start stream. Auto-detects in ~3 seconds.";
        else if (status === "drawing") hint.innerHTML = "✅ Door zone set! Start the stream to begin monitoring.";
        else if (status === "learning") hint.innerHTML = "🟡 Learning background... (~3 seconds remaining)";
        else if (status === "OPEN")  hint.innerHTML = `🔴 <b>Door: OPEN</b> &nbsp;|&nbsp; Opens: <b>${data.open_count}</b> &nbsp; Closes: <b>${data.close_count}</b> &nbsp; Last: ${data.last_event || '—'}`;
        else if (status === "CLOSED") hint.innerHTML = `🟢 <b>Door: CLOSED</b> &nbsp;|&nbsp; Opens: <b>${data.open_count}</b> &nbsp; Closes: <b>${data.close_count}</b> &nbsp; Last: ${data.last_event || '—'}`;
    }

    function _pollDoorState() {
        if (!currentSessionId || getMode() !== "vision_watch") return;
        fetch(`/api/vision_watch/door_state/${currentSessionId}`)
            .then(r => r.json())
            .then(data => {
                const state = data.state;
                if (data.learning) {
                    _updateDoorHint("learning");
                } else {
                    _updateDoorHint(state, data);
                }
            })
            .catch(() => {});
    }

    function drawVwPolygons() {
        if (!vwCtx) return;
        vwCtx.clearRect(0, 0, frRoiCanvas.width, frRoiCanvas.height);

        const drawPoly = (zones, current, color, fill) => {
            vwCtx.strokeStyle = color;
            vwCtx.lineWidth   = 2;
            vwCtx.fillStyle   = fill;
            zones.forEach(zone => {
                if (zone.length < 2) return;
                vwCtx.beginPath();
                vwCtx.moveTo(zone[0].x, zone[0].y);
                zone.slice(1).forEach(p => vwCtx.lineTo(p.x, p.y));
                vwCtx.closePath();
                vwCtx.fill();
                vwCtx.stroke();
            });
            if (current.length > 0) {
                vwCtx.beginPath();
                vwCtx.moveTo(current[0].x, current[0].y);
                current.slice(1).forEach(p => vwCtx.lineTo(p.x, p.y));
                vwCtx.stroke();
                vwCtx.fillStyle = "#fff";
                current.forEach(p => {
                    vwCtx.beginPath();
                    vwCtx.arc(p.x, p.y, 4, 0, Math.PI * 2);
                    vwCtx.fill();
                });
            }
        };

        // Door zone = blue
        drawPoly(vwDoorZones, currentDoorZone, "#3b82f6", "rgba(59,130,246,0.15)");
    }
    let identityFilter   = "all";
    let renamingPersonId = null;
    let regPhotoFile     = null;
    let identitiesCache  = [];
    let isWebcamStream   = false;  // track if webcam opened (needs explicit release on stop)
    let attendanceStatus = { active: false, present_ids: [] };

    function getMode() {
        return modeSelector ? modeSelector.value : "visitor";
    }

    modeSelector.addEventListener("change", () => {
        const mode = getMode();
        if (mode === "attendance") {
            visitorControls.classList.add("hidden");
            attendanceControls.classList.remove("hidden");
            visionWatchControls.classList.add("hidden");
            rightPanelTitle.textContent = "🗂️ Attendance List";
            frStartBtn.textContent = "▶ Start Attendance Tracker";
            if (filterKnownBtn) filterKnownBtn.textContent = "Registered";
            frRoiCanvas.classList.add("hidden");
            if (doorPollInterval) { clearInterval(doorPollInterval); doorPollInterval = null; }
        } else if (mode === "vision_watch") {
            visitorControls.classList.add("hidden");
            attendanceControls.classList.add("hidden");
            visionWatchControls.classList.remove("hidden");
            rightPanelTitle.textContent = "🛡️ Vision Watch";
            frStartBtn.textContent = "▶ Start Vision Watch";
            if (filterKnownBtn) filterKnownBtn.textContent = "Known";
            frRoiCanvas.classList.remove("hidden");
            alignCanvas();
        } else {
            visitorControls.classList.remove("hidden");
            attendanceControls.classList.add("hidden");
            visionWatchControls.classList.add("hidden");
            rightPanelTitle.textContent = "👤 Identity Manager";
            frStartBtn.textContent = "▶ Start Recognition";
            if (filterKnownBtn) filterKnownBtn.textContent = "Known";
            frRoiCanvas.classList.add("hidden");
            if (doorPollInterval) { clearInterval(doorPollInterval); doorPollInterval = null; }
        }
        loadStats();
        loadIdentities();
        if (!frReportsContent.classList.contains("hidden")) {
            loadReports();
        }
        if (currentSessionId) {
            _stopStream(); // Restart stream if mode is changed while active
        }
    });

    // ── Init ───────────────────────────────────────────────────────────────────
    setStatus("loading", "Connecting...");
    loadStats();
    loadIdentities();
    frStartBtn.disabled = false;   // webcam tab is default → always ready

    // ── Source tab switching ───────────────────────────────────────────────────
    tabs.forEach(tab => {
        tab.addEventListener("click", () => {
            const target = tab.dataset.tab;
            tabs.forEach(t => t.classList.remove("active"));
            tabContents.forEach(c => c.classList.remove("active"));
            tab.classList.add("active");
            document.getElementById(`content-${target}`).classList.add("active");
            activeTab = target;
            updateStartBtn();
        });
    });

    function updateStartBtn() {
        if (activeTab === "webcam") {
            frStartBtn.disabled = false;
        } else if (activeTab === "rtsp") {
            frStartBtn.disabled = (currentStreamId === null);
        } else if (activeTab === "upload") {
            frStartBtn.disabled = (currentVideoData === null);
        }
    }

    // ── File Upload ────────────────────────────────────────────────────────────
    frUploadBtn.addEventListener("click", () => frVideoUpload.click());

    frVideoUpload.addEventListener("change", (e) => {
        if (!e.target.files.length) return;
        const file = e.target.files[0];
        frUploadStatus.textContent = "Uploading...";
        frUploadStatus.style.color = "";

        const formData = new FormData();
        formData.append("file", file);

        fetch("/api/upload", { method: "POST", body: formData })
            .then(r => {
                if (!r.ok) return r.json().then(e => { throw new Error(e.detail || "Upload failed"); });
                return r.json();
            })
            .then(data => {
                currentVideoData = data;
                currentStreamId  = null;
                frUploadStatus.textContent = file.name;
                frUploadStatus.style.color = "var(--known)";
                updateStartBtn();
            })
            .catch(err => {
                frUploadStatus.textContent = "Upload failed: " + err.message;
                frUploadStatus.style.color = "var(--danger)";
            });
    });

    // ── RTSP Connect / Disconnect ──────────────────────────────────────────────
    frConnectBtn.addEventListener("click", () => {
        const url = frRtspUrl.value.trim();
        if (!url) {
            frRtspUrl.style.borderColor = "var(--danger)";
            return;
        }
        frRtspUrl.style.borderColor = "";
        frConnectBtn.disabled = true;
        frConnectBtn.textContent = "Connecting...";

        fetch("/api/connect_stream", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url })
        })
        .then(r => {
            if (!r.ok) return r.json().then(e => { throw new Error(e.detail); });
            return r.json();
        })
        .then(data => {
            currentStreamId  = data.stream_id;
            currentVideoData = {
                video_id: data.stream_id,
                filename: url,
                width:    data.width,
                height:   data.height,
            };
            isWebcamStream = false;
            frConnectBtn.textContent = "✓ Connected";
            frDisconnBtn.classList.remove("hidden");
            regSnapshotBtn.disabled = false;
            attRegSnapshotBtn.disabled = false;
            updateStartBtn();

            // ── Show live preview so user can draw ROI on real video ──
            placeholder.classList.add("hidden");
            streamImg.src = `/api/raw_stream/${data.stream_id}`;
            streamImg.classList.remove("hidden");
            videoWrapper.classList.add("active");
            // Re-align the canvas once the first frame loads
            streamImg.onload = alignCanvas;
        })
        .catch(err => {
            frConnectBtn.textContent = "Connect RTSP";
            frConnectBtn.disabled = false;
            alert("RTSP error: " + err.message);
        });
    });

    frDisconnBtn.addEventListener("click", () => {
        _releaseStream();
        frConnectBtn.textContent = "Connect RTSP";
        frConnectBtn.disabled = false;
        frDisconnBtn.classList.add("hidden");
        regSnapshotBtn.disabled = true;
        attRegSnapshotBtn.disabled = true;
        updateStartBtn();
        // Hide stream preview on explicit disconnect
        streamImg.src = "";
        streamImg.classList.add("hidden");
        placeholder.classList.remove("hidden");
        videoWrapper.classList.remove("active");
        vwDoorZones = [];
        currentDoorZone = [];
        drawVwPolygons();
    });

    // ── Start Analysis ─────────────────────────────────────────────────────────
    frStartBtn.addEventListener("click", async () => {
        frStartBtn.disabled = true;

        // ── Webcam: connect first via dedicated endpoint ──────────────────────
        if (activeTab === "webcam") {
            const idx = parseInt(webcamIndex.value.trim() || "0", 10);
            frStartBtn.textContent = "Opening webcam...";
            try {
                const resp = await fetch(`/api/connect_webcam?index=${idx}`, { method: "POST" });
                if (!resp.ok) {
                    const err = await resp.json();
                    throw new Error(err.detail || "Failed to open webcam");
                }
                const data = await resp.json();
                currentStreamId  = data.stream_id;
                currentVideoData = {
                    video_id: data.stream_id,
                    filename: `webcam_${idx}`,
                    width:    data.width,
                    height:   data.height,
                };
                isWebcamStream = true;
                regSnapshotBtn.disabled = false;
                attRegSnapshotBtn.disabled = false;
            } catch (e) {
                alert("Webcam error: " + e.message);
                frStartBtn.textContent = getMode() === "attendance" ? "▶ Start Attendance Tracker" : "▶ Start Recognition";
                frStartBtn.disabled = false;
                return;
            }
        }

        frStartBtn.textContent = "Starting...";

        // Process ROI coordinates
        const mode = getMode();
        let doorRoiList = [];
        if (mode === "vision_watch" && frRoiCanvas.width > 0) {
            vwDoorZones.forEach(zone => {
                const normZone = zone.map(p => [p.x / frRoiCanvas.width, p.y / frRoiCanvas.height]);
                doorRoiList.push(normZone);
            });
        }
        
        const payload = {
            video_id:       currentVideoData.video_id,
            filename:       currentVideoData.filename,
            pipeline_name:  "face_recognition",
            roi_normalized: [[0, 0], [1, 0], [1, 1], [0, 1]],
            config: {
                mode: mode,
                watchman_name: vwWatchmanName ? vwWatchmanName.value.trim() : "",
                multiple_rois: [],      // watchman zones reserved for future
                door_rois: doorRoiList
            },
            stream_id: currentStreamId || null,
        };

        try {
            const resp = await fetch("/api/start_analysis", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            if (!resp.ok) throw new Error("Failed to start analysis");
            const data = await resp.json();
            currentSessionId = data.session_id;

            // Start door state poller for Vision Watch
            if (mode === "vision_watch") {
                if (doorPollInterval) clearInterval(doorPollInterval);
                doorPollInterval = setInterval(_pollDoorState, 2000);
                _updateDoorHint("learning");
            }

            _startStream();
        } catch (e) {
            alert("Failed to start recognition: " + e.message);
            // If we opened a webcam for this attempt, release it
            if (isWebcamStream && currentStreamId) {
                fetch(`/api/stream/${currentStreamId}`, { method: "DELETE" }).catch(() => {});
                currentStreamId = null;
                isWebcamStream  = false;
            }
            frStartBtn.textContent = getMode() === "attendance" ? "▶ Start Attendance Tracker" : "▶ Start Recognition";
            frStartBtn.disabled = false;
        }
    });

    // ── Stop Analysis ──────────────────────────────────────────────────────────
    frStopBtn.addEventListener("click", _stopStream);

    function _startStream() {
        placeholder.classList.add("hidden");
        streamImg.src = `/api/stream/${currentSessionId}`;
        streamImg.classList.remove("hidden");
        videoWrapper.classList.add("active");

        if (getMode() === "attendance") {
            fetch("/api/attendance/start", { method: "POST" });
        }

        frStartBtn.classList.add("hidden");
        frStartBtn.textContent = getMode() === "attendance" ? "▶ Start Attendance Tracker" : "▶ Start Recognition";
        frStopBtn.classList.remove("hidden");
        setStatus("active", "Recognition Active");

        // Start door poller if in vision_watch mode
        if (getMode() === "vision_watch") {
            if (doorPollInterval) clearInterval(doorPollInterval);
            doorPollInterval = setInterval(_pollDoorState, 2000);
        }

        knownEventIds.clear();
        if (eventsPlaceholder) eventsPlaceholder.style.display = "none";
        pollingInterval = setInterval(_pollEvents, 1500);
    }

    function _stopStream() {
        if (pollingInterval)  { clearInterval(pollingInterval);  pollingInterval  = null; }
        if (doorPollInterval) { clearInterval(doorPollInterval); doorPollInterval = null; }
        _updateDoorHint("idle");

        if (currentSessionId) {
            if (getMode() === "attendance") {
                fetch("/api/attendance/stop", { method: "POST" }).then(() => {
                    loadIdentities(); // Refresh identities to show absent status correctly
                });
            }
            fetch(`/api/stop_analysis/${currentSessionId}`, { method: "POST" }).catch(() => {});
            currentSessionId = null;
        }

        // Release webcam when stopping (RTSP stays alive for re-use)
        if (isWebcamStream && currentStreamId) {
            fetch(`/api/stream/${currentStreamId}`, { method: "DELETE" }).catch(() => {});
            currentStreamId = null;
            isWebcamStream  = false;
            regSnapshotBtn.disabled = true;
            attRegSnapshotBtn.disabled = true;
        }

        // If RTSP is still connected, restore the raw preview for ROI editing
        if (currentStreamId && !isWebcamStream) {
            streamImg.src = `/api/raw_stream/${currentStreamId}`;
            streamImg.classList.remove("hidden");
            placeholder.classList.add("hidden");
            videoWrapper.classList.add("active");
            streamImg.onload = alignCanvas;
        } else {
            streamImg.src = "";
            streamImg.classList.add("hidden");
            placeholder.classList.remove("hidden");
            videoWrapper.classList.remove("active");
        }

        frStartBtn.classList.remove("hidden");
        frStopBtn.classList.add("hidden");
        setStatus("idle", "Idle");

        updateStartBtn();
        loadStats();
        loadIdentities();
    }

    function _releaseStream() {
        if (currentStreamId && !isWebcamStream) {
            fetch(`/api/stream/${currentStreamId}`, { method: "DELETE" }).catch(() => {});
        }
        currentStreamId  = null;
        currentVideoData = null;
        isWebcamStream   = false;
    }

    // ── Event Polling ──────────────────────────────────────────────────────────
    function _pollEvents() {
        if (!currentSessionId) return;
        fetch(`/api/alerts/${currentSessionId}`)
            .then(r => r.json())
            .then(data => {
                if (!data.alerts || data.alerts.length === 0) return;
                data.alerts.forEach(alert => {
                    if (knownEventIds.has(alert.id)) return;
                    knownEventIds.add(alert.id);
                    _addEventCard(alert);
                });
                loadStats();
                loadIdentities();
            })
            .catch(err => console.warn("Poll error:", err));
    }

    function _addEventCard(alert) {
        if (alert.type === "vision_watch_alert") {
            const card = document.createElement("div");
            card.className = "fr-event-card unknown";
            card.style.borderColor = "var(--danger)";
            card.style.backgroundColor = "rgba(255, 68, 68, 0.1)";
            card.innerHTML = `
                <span class="fr-event-icon">🚨</span>
                <span class="fr-event-label" style="color: var(--danger); font-weight: bold;">${escHtml(alert.message || "Vision Watch Alert")}</span>
                <span class="fr-event-conf"></span>
                <span class="fr-event-time">${alert.timestamp || ""}</span>
            `;
            eventsList.prepend(card);
            buzzerAudio.currentTime = 0;
            buzzerAudio.play().catch(e => console.warn("Audio play prevented:", e));
            return;
        }

        if (alert.type === "door_open") {
            const card = document.createElement("div");
            card.className = "fr-event-card";
            card.style.borderColor = "#ef4444";
            card.style.backgroundColor = "rgba(239,68,68,0.08)";
            card.innerHTML = `
                <span class="fr-event-icon">🔴</span>
                <span class="fr-event-label" style="color:#ef4444; font-weight:bold;">${escHtml(alert.message || "Door Opened")}</span>
                <span class="fr-event-conf"></span>
                <span class="fr-event-time">${alert.timestamp || ""}</span>
            `;
            eventsList.prepend(card);
            return;
        }

        if (alert.type === "door_close") {
            const card = document.createElement("div");
            card.className = "fr-event-card";
            card.style.borderColor = "#22c55e";
            card.style.backgroundColor = "rgba(34,197,94,0.08)";
            card.innerHTML = `
                <span class="fr-event-icon">🟢</span>
                <span class="fr-event-label" style="color:#22c55e; font-weight:bold;">${escHtml(alert.message || "Door Closed")}</span>
                <span class="fr-event-conf"></span>
                <span class="fr-event-time">${alert.timestamp || ""}</span>
            `;
            eventsList.prepend(card);
            return;
        }

        const isKnown = alert.status === "known" || alert.status === "recognised" || alert.confidence > 0;
        const card = document.createElement("div");
        card.className = `fr-event-card ${isKnown ? "known" : "unknown"}`;
        card.innerHTML = `
            <span class="fr-event-icon">${isKnown ? "✅" : "👤"}</span>
            <span class="fr-event-label">${escHtml(alert.label || "Face detected")}</span>
            <span class="fr-event-conf">${alert.confidence ? Math.round(alert.confidence * 100) + "%" : ""}</span>
            <span class="fr-event-time">${alert.timestamp || ""}</span>
        `;
        eventsList.prepend(card);
    }

    // ── Stats ──────────────────────────────────────────────────────────────────
    function deleteIdentity(pid) {
        if (!confirm(`Are you sure you want to completely delete person ${pid}?`)) return;
        const mode = getMode();
        fetch(`/api/faces/identity/${pid}?mode=${mode}`, { method: "DELETE" })
            .then(r => r.json())
            .then(() => {
                loadStats();
                loadIdentities();
            });
    }

    // ── Reports Manager ────────────────────────────────────────────────────────
    
    function loadReports() {
        const mode = getMode();
        reportsList.innerHTML = `<div class="fr-identity-loading">Loading reports...</div>`;
        fetch(`/api/reports?mode=${mode}`)
            .then(r => r.json())
            .then(data => {
                const list = data.reports || [];
                if (list.length === 0) {
                    reportsList.innerHTML = `<div class="fr-identity-empty">No reports found.</div>`;
                    return;
                }
                
                reportsList.innerHTML = "";
                list.forEach(report => {
                    const card = document.createElement("div");
                    card.className = "report-card";
                    
                    const startStr = new Date(report.start_time * 1000).toLocaleString();
                    const durationStr = Math.round(report.duration) + "s";
                    
                    let statsHtml = "";
                    if (mode === "attendance") {
                        statsHtml = `Present: <b>${report.present_count}</b> | Absent: <b>${report.absent_count}</b>`;
                    } else if (mode === "vision_watch") {
                        statsHtml = `🔴 Opens: <b>${report.door_open_count ?? 0}</b> &nbsp;|&nbsp; 🟢 Closes: <b>${report.door_close_count ?? 0}</b>`;
                    } else {
                        statsHtml = `Known: <b>${report.known_count}</b> | Unknown: <b>${report.unknown_count}</b>`;
                    }

                    const displayId = String(report.session_id || "").replace(/[^0-9]/g, "").slice(-3) || report.session_id;
                    card.innerHTML = `
                        <div class="report-title">
                            ${mode === "vision_watch" ? "🚪" : mode === "attendance" ? "📚" : "👤"}
                            Session #${displayId}
                        </div>
                        <div class="report-meta">${startStr} &nbsp;·&nbsp; ${durationStr}</div>
                        <div class="report-meta" style="margin-top: 4px; color: #cbd5e1;">${statsHtml}</div>
                    `;

                    card.addEventListener("click", () => showReportDetails(report.session_id, mode));
                    reportsList.appendChild(card);
                });
            })
            .catch(() => {
                reportsList.innerHTML = `<div class="fr-identity-empty">Error loading reports.</div>`;
            });
    }
    
    function showReportDetails(sessionId, mode) {
        reportModalBody.innerHTML = `<div class="fr-identity-loading">Fetching details...</div>`;
        reportModalMeta.textContent = `Session: ${sessionId}`;
        reportModalTitle.textContent = mode === "attendance" ? "📚 Attendance Report"
            : mode === "vision_watch" ? "🚪 Vision Watch Report"
            : "👤 Visitor Report";
        reportModal.classList.remove("hidden");
        
        fetch(`/api/reports/${sessionId}?mode=${mode}`)
            .then(r => r.json())
            .then(report => {
                let html = "";
                
                if (mode === "attendance") {
                    html += `
                        <div class="report-section">
                            <div class="report-section-title">Present (${report.present.length})</div>
                            ${report.present.map(p => `<div class="report-item"><span>${escHtml(p.label)}</span><span style="color:#4ade80;">✓ Present</span></div>`).join("")}
                            ${report.present.length === 0 ? '<div class="report-meta">No one was present.</div>' : ''}
                        </div>
                        <div class="report-section">
                            <div class="report-section-title">Absent (${report.absent.length})</div>
                            ${report.absent.map(p => `<div class="report-item"><span>${escHtml(p.label)}</span><span style="color:#f87171;">✗ Absent</span></div>`).join("")}
                            ${report.absent.length === 0 ? '<div class="report-meta">Everyone was present.</div>' : ''}
                        </div>
                    `;
                } else if (mode === "vision_watch") {
                    html += `
                        <div class="report-section" style="margin-bottom:12px;">
                            <div class="report-section-title" style="font-size:1rem;">🚪 Door Activity Summary</div>
                            <div class="report-item" style="background:rgba(239,68,68,0.1); border-radius:6px; padding:10px 14px; margin-top:8px; display:flex; justify-content:space-between; align-items:center;">
                                <span>🔴 Total Opens</span>
                                <span style="font-weight:bold; font-size:1.4rem; color:#ef4444;">${report.door_open_count ?? 0}</span>
                            </div>
                            <div class="report-item" style="background:rgba(34,197,94,0.1); border-radius:6px; padding:10px 14px; margin-top:6px; display:flex; justify-content:space-between; align-items:center;">
                                <span>🟢 Total Closes</span>
                                <span style="font-weight:bold; font-size:1.4rem; color:#22c55e;">${report.door_close_count ?? 0}</span>
                            </div>
                        </div>
                    `;
                } else {

                    html += `
                        <div class="report-section">
                            <div class="report-section-title">Known Visitors (${report.known_visitors.length})</div>
                            ${report.known_visitors.map(p => {
                                const seen = new Date(p.first_seen * 1000).toLocaleTimeString();
                                return `<div class="report-item"><span>${escHtml(p.label)}</span><span>Seen at ${seen} (${p.count}x)</span></div>`;
                            }).join("")}
                            ${report.known_visitors.length === 0 ? '<div class="report-meta">No known visitors seen.</div>' : ''}
                        </div>
                        <div class="report-section">
                            <div class="report-section-title">Unknown Visitors (${report.unknown_visitors.length})</div>
                            ${report.unknown_visitors.map(p => {
                                const seen = new Date(p.first_seen * 1000).toLocaleTimeString();
                                return `<div class="report-item"><span>Track ID ${p.id}</span><span>Seen at ${seen} (${p.count}x)</span></div>`;
                            }).join("")}
                            ${report.unknown_visitors.length === 0 ? '<div class="report-meta">No unknown visitors seen.</div>' : ''}
                        </div>
                    `;
                }
                
                reportModalBody.innerHTML = html;
            })
            .catch(() => {
                reportModalBody.innerHTML = `<div class="fr-identity-empty">Failed to load details.</div>`;
            });
    }

    function loadStats() {
        const mode = getMode();
        fetch(`/api/faces/status?mode=${mode}`)
            .then(r => r.json())
            .then(data => {
                statTotal.textContent   = data.total_identities ?? "\u2014";
                statKnown.textContent   = data.total_identities ?? "\u2014";  // everyone is known
                // statUnknown provision is hidden in UI now
                statVectors.textContent = data.faiss_vectors    ?? "\u2014";
                setStatus("idle", "System Ready");
            })
            .catch(() => setStatus("error", "Backend unreachable"));
    }

    // ── Identity Manager ───────────────────────────────────────────────────────
    function loadIdentities() {
        const mode = getMode();
        let identitiesPromise = fetch(`/api/faces/identities?mode=${mode}`).then(r => r.json());
        let attendancePromise = mode === "attendance" 
            ? fetch("/api/attendance/status").then(r => r.json())
            : Promise.resolve(null);
            
        Promise.all([identitiesPromise, attendancePromise])
            .then(([idData, attData]) => {
                identitiesCache = idData.identities || [];
                if (attData) attendanceStatus = attData;
                if (identityFilter !== "pending") renderIdentities();
            })
            .catch(() => {
                identityList.innerHTML = `<div class="fr-identity-empty">Error loading identities.</div>`;
            });
    }

    function loadPending() {
        const mode = getMode();
        pendingList.innerHTML = `<div class="fr-identity-loading">Loading pending requests...</div>`;
        fetch(`/api/faces/pending?mode=${mode}`)
            .then(r => r.json())
            .then(data => {
                const list = data.pending || [];
                if (list.length === 0) {
                    pendingList.innerHTML = `<div class="fr-identity-empty">No pending approval requests.</div>`;
                    return;
                }
                
                pendingList.innerHTML = "";
                list.forEach(pending => {
                    const card = document.createElement("div");
                    card.className = "fr-identity-card pending-card";
                    
                    const thumbHtml = pending.thumbnail_url
                        ? `<img src="${pending.thumbnail_url}" alt="face" class="fr-identity-thumb unknown">`
                        : `<div class="fr-identity-thumb-placeholder">?</div>`;
                        
                    const created = new Date(pending.timestamp * 1000).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit', second:'2-digit'});
                    
                    card.innerHTML = `
                        ${thumbHtml}
                        <div class="fr-identity-info" style="gap: 4px;">
                            <div class="fr-identity-meta" style="margin-bottom: 4px;">Req: ${pending.req_id} • ${created}</div>
                            <input type="text" class="fr-input fr-pending-name" placeholder="Enter Name/Roll No..." style="padding: 4px; font-size: 0.85rem;">
                        </div>
                        <button class="fr-btn fr-btn-icon approve-btn" title="Approve" style="color: #4ade80;" data-req="${pending.req_id}">✓</button>
                        <button class="fr-btn fr-btn-icon reject-btn" title="Reject" style="color: #f87171;" data-req="${pending.req_id}">✗</button>
                    `;
                    
                    const inputField = card.querySelector(".fr-pending-name");
                    const approveBtn = card.querySelector(".approve-btn");
                    const rejectBtn = card.querySelector(".reject-btn");
                    
                    approveBtn.addEventListener("click", () => {
                        const label = inputField.value.trim() || `Unknown_${pending.req_id.slice(-4)}`;
                        approveBtn.disabled = true;
                        rejectBtn.disabled = true;
                        fetch(`/api/faces/pending/${pending.req_id}/approve?mode=${mode}`, {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ label })
                        }).then(r => r.json()).then(() => {
                            loadPending();
                            loadIdentities();
                        });
                    });
                    
                    rejectBtn.addEventListener("click", () => {
                        approveBtn.disabled = true;
                        rejectBtn.disabled = true;
                        fetch(`/api/faces/pending/${pending.req_id}/reject?mode=${mode}`, {
                            method: "POST"
                        }).then(r => r.json()).then(() => {
                            loadPending();
                        });
                    });
                    
                    pendingList.appendChild(card);
                });
            })
            .catch(() => {
                pendingList.innerHTML = `<div class="fr-identity-empty">Error loading pending requests.</div>`;
            });
    }

    function renderIdentities() {
        // The user's philosophy: all saved faces are "Known"
        let list = identitiesCache;
        if (identityFilter === "known")   list = identitiesCache;
        if (identityFilter === "blank")   list = [];

        if (list.length === 0) {
            identityList.innerHTML = `<div class="fr-identity-empty">
                No persons${identityFilter !== "all" ? ` (${identityFilter})` : ""} yet.
                Start recognition — faces are auto-saved as soon as the system sees them.
            </div>`;
            return;
        }

        identityList.innerHTML = "";
        [...list].reverse().forEach(identity => {
            const card = document.createElement("div");
            const isNamed = identity.named === true;
            let statusText = identity.person_id;
            let statusClass = isNamed ? "known" : "unknown";
            let typeBadgeHtml = `<div class="fr-identity-type ${statusClass}">${statusText}</div>`;

            if (getMode() === "attendance") {
                const isPresent = attendanceStatus && attendanceStatus.present_ids.includes(identity.person_id);
                
                if (isPresent) {
                    statusClass = "present";
                    statusText = "Present";
                } else {
                    statusClass = "absent";
                    statusText = "Absent";
                }
                card.className = `fr-identity-card ${statusClass}`;
                typeBadgeHtml = `<div class="fr-identity-type ${statusClass}">${statusText}</div>`;
            } else {
                card.className = `fr-identity-card ${statusClass}`;
            }

            const icon = isNamed ? "✅" : "👤";
            const thumbHtml = identity.thumbnail_url
                ? `<img src="${identity.thumbnail_url}" alt="face"
                        class="fr-identity-thumb ${statusClass}"
                        onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
                   <div class="fr-identity-thumb-placeholder" style="display:none">${icon}</div>`
                : `<div class="fr-identity-thumb-placeholder">${icon}</div>`;

            const created = identity.created_at
                ? new Date(identity.created_at * 1000).toLocaleDateString()
                : "";

            card.innerHTML = `
                ${thumbHtml}
                <div class="fr-identity-info">
                    <div class="fr-identity-name" title="${escHtml(identity.label)}">${escHtml(identity.label)}</div>
                    <div class="fr-identity-meta">${typeBadgeHtml} · ${identity.occurrences ?? 0} sessions · ${created}</div>
                </div>
                <button class="fr-identity-rename-btn" title="Rename"
                    data-pid="${identity.person_id}"
                    data-label="${escHtml(identity.label)}">Edit</button>
                <button class="fr-identity-delete-btn" title="Delete"
                    data-pid="${identity.person_id}">Del</button>
            `;

            card.querySelector(".fr-identity-rename-btn").addEventListener("click", () => {
                openRenameModal(identity.person_id, identity.label);
            });

            card.querySelector(".fr-identity-delete-btn").addEventListener("click", () => {
                deleteIdentity(identity.person_id);
            });

            identityList.appendChild(card);
        });
    }

    refreshBtn.addEventListener("click", () => {
        loadStats();
        loadIdentities();
    });

    filterTabs.forEach(tab => {
        tab.addEventListener("click", () => {
            filterTabs.forEach(t => t.classList.remove("active"));
            tab.classList.add("active");
            identityFilter = tab.dataset.filter;
            if (identityFilter === "pending") {
                identityList.classList.add("hidden");
                pendingList.classList.remove("hidden");
                loadPending();
            } else {
                identityList.classList.remove("hidden");
                pendingList.classList.add("hidden");
                renderIdentities();
            }
        });
    });

    refreshBtn.addEventListener("click", () => {
        if (identityFilter === "pending") {
            loadPending();
        } else {
            loadStats();
            loadIdentities();
        }
    });

    // ── Rename Modal ───────────────────────────────────────────────────────────
    function openRenameModal(personId, currentLabel) {
        renamingPersonId = personId;
        renameInput.value = currentLabel || "";
        renameModalPid.textContent = `ID: ${personId}`;
        renameModal.classList.remove("hidden");
        renameInput.focus();
        renameInput.select();
    }

    renameConfirm.addEventListener("click", () => {
        const newLabel = renameInput.value.trim();
        if (!newLabel || !renamingPersonId) return;

        fetch(`/api/faces/identity/${renamingPersonId}?mode=${getMode()}`, {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ new_label: newLabel })
        })
        .then(r => { if (!r.ok) return r.json().then(e => { throw new Error(e.detail); }); return r.json(); })
        .then(() => {
            renameModal.classList.add("hidden");
            renamingPersonId = null;
            loadIdentities();
            loadStats();
        })
        .catch(err => alert("Rename error: " + err.message));
    });

    renameCancel.addEventListener("click", () => {
        renameModal.classList.add("hidden");
        renamingPersonId = null;
    });

    renameInput.addEventListener("keydown", e => {
        if (e.key === "Enter")  renameConfirm.click();
        if (e.key === "Escape") renameCancel.click();
    });

    // ── Registration — Photo Upload ────────────────────────────────────────────
    regDropZone.addEventListener("click", () => regPhoto.click());

    regPhoto.addEventListener("change", e => {
        if (!e.target.files.length) return;
        regPhotoFile = e.target.files[0];
        regPreviewImg.src = URL.createObjectURL(regPhotoFile);
        regPreview.classList.remove("hidden");
        regSubmitBtn.disabled = false;
    });

    // Drag-and-drop
    regDropZone.addEventListener("dragover", e => {
        e.preventDefault();
        regDropZone.style.borderColor = "var(--fr-accent-2)";
    });
    regDropZone.addEventListener("dragleave", () => regDropZone.style.borderColor = "");
    regDropZone.addEventListener("drop", e => {
        e.preventDefault();
        regDropZone.style.borderColor = "";
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith("image/")) {
            regPhotoFile = file;
            regPreviewImg.src = URL.createObjectURL(file);
            regPreview.classList.remove("hidden");
            regSubmitBtn.disabled = false;
        }
    });

    regSubmitBtn.addEventListener("click", async () => {
        if (!regPhotoFile) { alert("Select a photo first."); return; }
        const label = regName.value.trim() || null;
        const url   = label
            ? `/api/faces/register?label=${encodeURIComponent(label)}&mode=${getMode()}`
            : `/api/faces/register?mode=${getMode()}`;

        regSubmitBtn.disabled = true;
        regSubmitBtn.textContent = "Registering...";

        const formData = new FormData();
        formData.append("file", regPhotoFile);

        try {
            const resp = await fetch(url, { method: "POST", body: formData });
            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || "Registration failed");
            }
            const data = await resp.json();
            showRegFeedback(`Registered: ${data.label} (${data.person_id})`, "success");
            regName.value   = "";
            regPhotoFile    = null;
            regPreviewImg.src = "";
            regPreview.classList.add("hidden");
            regSubmitBtn.disabled = true;
            loadIdentities();
            loadStats();
        } catch (e) {
            showRegFeedback(e.message, "error");
        } finally {
            regSubmitBtn.textContent = "✓ Register Identity";
            if (!regPhotoFile) regSubmitBtn.disabled = true;
        }
    });

    // ── Registration — Live Snapshot ───────────────────────────────────────────
    regSnapshotBtn.addEventListener("click", async () => {
        if (!currentStreamId) {
            alert("Start a live source first (webcam or RTSP), then click snapshot.");
            return;
        }
        const label = regName.value.trim() || null;
        const url   = label
            ? `/api/faces/snapshot/${currentStreamId}?label=${encodeURIComponent(label)}&mode=${getMode()}`
            : `/api/faces/snapshot/${currentStreamId}?mode=${getMode()}`;

        regSnapshotBtn.disabled = true;
        regSnapshotBtn.textContent = "Capturing...";

        try {
            const resp = await fetch(url, { method: "POST" });
            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || "Snapshot failed");
            }
            const data = await resp.json();
            showRegFeedback(`Registered: ${data.label}`, "success");
            regName.value = "";
            loadIdentities();
            loadStats();
        } catch (e) {
            showRegFeedback(e.message, "error");
        } finally {
            regSnapshotBtn.disabled = false;
            regSnapshotBtn.textContent = "📸 Snapshot from Live Feed";
        }
    });

    // ── Attendance Registration — Live Snapshot ────────────────────────────────
    attRegSnapshotBtn.addEventListener("click", async () => {
        if (!currentStreamId) {
            alert("Start a live source first (webcam or RTSP), then click snapshot.");
            return;
        }
        const label = attRegName.value.trim() || null;
        if (!label) {
            alert("Please enter a full name for attendance tracking.");
            return;
        }
        const url = `/api/faces/snapshot/${currentStreamId}?label=${encodeURIComponent(label)}&mode=attendance`;

        attRegSnapshotBtn.disabled = true;
        attRegSnapshotBtn.textContent = "Capturing...";

        try {
            const resp = await fetch(url, { method: "POST" });
            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || "Snapshot failed");
            }
            const data = await resp.json();
            
            attRegFeedback.textContent = "✓ Registered: " + data.label;
            attRegFeedback.className = "fr-reg-feedback success";
            attRegFeedback.classList.remove("hidden");
            setTimeout(() => attRegFeedback.classList.add("hidden"), 5000);
            
            attRegName.value = "";
            loadIdentities();
            loadStats();
        } catch (e) {
            attRegFeedback.textContent = "✗ " + e.message;
            attRegFeedback.className = "fr-reg-feedback error";
            attRegFeedback.classList.remove("hidden");
            setTimeout(() => attRegFeedback.classList.add("hidden"), 5000);
        } finally {
            attRegSnapshotBtn.disabled = false;
            attRegSnapshotBtn.textContent = "📸 Snapshot from Live Feed";
        }
    });

    // ── Helpers ────────────────────────────────────────────────────────────────
    function showRegFeedback(msg, type) {
        regFeedback.textContent = (type === "success" ? "✓ " : "✗ ") + msg;
        regFeedback.className = `fr-reg-feedback ${type}`;
        regFeedback.classList.remove("hidden");
        setTimeout(() => regFeedback.classList.add("hidden"), 5000);
    }

    function setStatus(state, text) {
        statusDot.className = "fr-status-dot"
            + (state === "active"  ? " active"  : "")
            + (state === "loading" ? " loading" : "");
        statusText.textContent = text;
    }

    function escHtml(str) {
        if (!str) return "";
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;");
    }

    // Webcam index change keeps button enabled
    webcamIndex.addEventListener("input", updateStartBtn);

});
