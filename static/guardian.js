/**
 * guardian.js - Room Object Guardian frontend module
 * ====================================================
 * Fully self-contained. Owns all DOM elements under id="guardian-*".
 * Never modifies or imports from app.js.
 *
 * Activation: Becomes active when the user selects "room_guardian"
 * from the existing #pipeline-select dropdown.
 *
 * Two-phase workflow:
 *   Phase 1 - Scan & Select:
 *     - "Scan Object" button calls POST /api/guardian/scan
 *     - YOLO boxes drawn on canvas (orange). Click to select (green).
 *     - "Draw Custom Object" toggle: clean canvas, rect-draw mode.
 *       Each rectangle auto-labeled Unknown-1, Unknown-2, etc.
 *     - Toggle back to YOLO mode: both YOLO boxes and custom rects visible.
 *     - Selection list panel shows all enrolled objects.
 *   Phase 2 - Guardian mode:
 *     - "Start Guarding" calls POST /api/guardian/start
 *     - Live annotated stream loaded via /api/stream/{session_id}
 *     - Alerts polled from /api/alerts/{session_id}
 *     - Stop button returns to Phase 1.
 */

(function () {
    "use strict";

    // -- Wait for DOM ----------------------------------------------------------
    document.addEventListener("DOMContentLoaded", initGuardian);

    function initGuardian() {

        // -- DOM refs (guardian-namespaced elements) ---------------------------
        const guardianPanel     = document.getElementById("guardian-controls");
        const scanBtn           = document.getElementById("guardian-scan-btn");

        const startGuardBtn     = document.getElementById("guardian-start-btn");
        const stopGuardBtn      = document.getElementById("guardian-stop-btn");
        const selectionList     = document.getElementById("guardian-selection-list");
        const guardianStatus    = document.getElementById("guardian-status");

        // Shared canvas & image already in the DOM (owned by app.js visually, but we draw on top)
        const mainVideoImg  = document.getElementById("main-video-img");
        const roiCanvas     = document.getElementById("roi-canvas");
        const ctx           = roiCanvas.getContext("2d");

        // Pipeline select (owned by app.js, we only listen)
        const pipelineSelect = document.getElementById("pipeline-select");

        // -- Guardian-local state ----------------------------------------------
        const state = {
            active:          false,          // true when guardian mode is showing
            phase:           "idle",         // "idle" | "scan" | "guarding"


            // Scan data from /api/guardian/scan
            scanPreviewUrl:  null,
            scanDetections:  [],             // [{id, label, class_id, confidence, bbox_normalized}]
            imageWidth:      0,
            imageHeight:     0,

            // Enrollment
            selectedYoloIds: new Set(),      // ids of clicked YOLO boxes


            // Guardian session
            sessionId:       null,
            pollingInterval: null,
            knownAlerts:     new Set(),

            // Stream / video source (read from window - set by app.js when user uploads/connects)
            // We access these via getter functions to avoid coupling
        };

        // -- Helper: read current source from app.js window state -------------
        // app.js sets currentVideoData and currentStreamId on window implicitly
        // via its own closure - we can't access them directly.
        // Instead we read from the DOM: the img src tells us what's loaded,
        // and we grab stream/filename info from the guardian-start request
        // by calling /api/guardian/scan which accepts stream_id OR filename.
        //
        // We piggyback on app.js's exposed data by reading the hidden data attrs
        // we'll write onto the img element when a source loads (see hookSourceChange).

        function getSourcePayload() {
            const streamId = mainVideoImg.dataset.guardianStreamId || null;
            const filename  = mainVideoImg.dataset.guardianFilename  || null;
            const videoId   = mainVideoImg.dataset.guardianVideoId   || null;
            return { stream_id: streamId, filename, video_id: videoId };
        }

        // -- Hook into app.js source events via MutationObserver --------------
        // app.js sets mainVideoImg.src when a source loads. We watch for this
        // and update our data attrs from the upload/connect response.
        // We also hook the pipeline-select change to show/hide our panel.

        pipelineSelect.addEventListener("change", onPipelineChange);
        onPipelineChange(); // run once on load in case room_guardian is pre-selected

        function onPipelineChange() {
            const isGuardian = pipelineSelect.value === "room_guardian";
            if (guardianPanel) {
                guardianPanel.classList.toggle("hidden", !isGuardian);
            }
            state.active = isGuardian;

            if (!isGuardian) {
                // Reset our canvas state so we don't interfere with other pipelines
                resetPhase1(false);
            }
        }

        // -- Intercept app.js upload/connect responses -------------------------
        // We monkey-patch the img's src setter to detect source changes.
        // This is the safest zero-coupling approach: no app.js modification needed.
        const _origSrcDescriptor = Object.getOwnPropertyDescriptor(HTMLImageElement.prototype, "src");
        const guardianSelf = { state, mainVideoImg };

        // Instead of monkey-patching (fragile), we use a MutationObserver on data attrs.
        // app.js sets currentVideoData in its closure. We expose a tiny bridge:
        // guardian.js reads window.__guardianBridge which app.js populates via
        // the data-guardian-* attributes we ask the HTML to set.
        // The cleanest zero-coupling: watch the img src change and re-parse what we know.

        // Actually the simplest approach: listen to our own scan button which fires
        // the scan request. The user always clicks "Scan Object" after connecting a source.
        // At that point we read the data attributes we instructed index.html to set.
        // (See index.html guardian-source-bridge hidden inputs.)

        // -- Scan button -------------------------------------------------------
        if (scanBtn) {
            scanBtn.addEventListener("click", handleScan);
        }

        async function handleScan() {
            const src = getSourcePayload();
            if (!src.stream_id && !src.filename) {
                setStatus("⚠️ No video source loaded. Upload a file or connect RTSP first.", "warn");
                return;
            }

            setStatus("Scanning scene...", "info");
            scanBtn.disabled = true;

            try {
                const res = await fetch("/api/guardian/scan", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(src),
                });
                if (!res.ok) {
                    const err = await res.json();
                    throw new Error(err.detail || "Scan failed.");
                }
                const data = await res.json();

                state.scanPreviewUrl = data.preview_url;
                // Sort detections by area (smallest first) so overlapping nested boxes can be clicked
                state.scanDetections  = (data.detections || []).sort((a, b) => {
                    const areaA = a.bbox_normalized[2] * a.bbox_normalized[3];
                    const areaB = b.bbox_normalized[2] * b.bbox_normalized[3];
                    return areaA - areaB;
                });
                state.imageWidth      = data.width;
                state.imageHeight     = data.height;
                state.phase           = "scan";

                // Show the scan preview still on the main image
                mainVideoImg.src = data.preview_url + "?t=" + Date.now();

                // Sync canvas internal resolution to match actual video pixels
                roiCanvas.width  = data.width;
                roiCanvas.height = data.height;

                // Force canvas visible - use inline style to beat any competing display:none
                roiCanvas.style.display = "block";
                roiCanvas.classList.remove("hidden");
                roiCanvas.style.cursor = "pointer";  // YOLO mode: click to select

                // Re-draw after the preview image renders. Use onload + staggered timeouts
                // because browsers may not re-fire onload for cached images.
                const doAlignAndDraw = () => { alignCanvas(); redrawCanvas(); };
                mainVideoImg.onload = doAlignAndDraw;
                setTimeout(doAlignAndDraw, 50);
                setTimeout(doAlignAndDraw, 250);
                setTimeout(doAlignAndDraw, 700);

                setStatus(`Found ${data.detections.length} object(s). Click a box to select.`, "ok");
            } catch (err) {
                setStatus("❌ " + err.message, "error");
            } finally {
                scanBtn.disabled = false;
            }
        }

        // -- Canvas interaction ------------------------------------------------
        roiCanvas.addEventListener("click",      onCanvasClick);

        function canvasXY(e) {
            const rect = roiCanvas.getBoundingClientRect();
            return {
                x: (e.clientX - rect.left) * (roiCanvas.width  / rect.width),
                y: (e.clientY - rect.top)  * (roiCanvas.height / rect.height),
            };
        }

        // --- Click (YOLO box selection) ---
        function onCanvasClick(e) {
            if (!state.active || state.phase !== "scan") return;

            const { x, y } = canvasXY(e);
            const W = roiCanvas.width, H = roiCanvas.height;

            for (const det of state.scanDetections) {
                const [nx, ny, nw, nh] = det.bbox_normalized;
                const bx = nx * W, by = ny * H, bw = nw * W, bh = nh * H;

                if (x >= bx && x <= bx + bw && y >= by && y <= by + bh) {
                    if (state.selectedYoloIds.has(det.id)) {
                        state.selectedYoloIds.delete(det.id);
                    } else {
                        state.selectedYoloIds.add(det.id);
                    }
                    redrawCanvas();
                    updateSelectionList();
                    checkStartReady();
                    return;
                }
            }
        }



        // -- Canvas drawing ----------------------------------------------------
        function redrawCanvas() {
            ctx.clearRect(0, 0, roiCanvas.width, roiCanvas.height);
            if (state.phase !== "scan") return;

            const W = roiCanvas.width, H = roiCanvas.height;

            // Draw YOLO detection boxes
            for (const det of state.scanDetections) {
                const [nx, ny, nw, nh] = det.bbox_normalized;
                const bx = nx * W, by = ny * H, bw = nw * W, bh = nh * H;
                const selected = state.selectedYoloIds.has(det.id);

                ctx.globalAlpha = 1.0;

                ctx.strokeStyle = selected ? "#22c55e" : "#f97316";
                ctx.lineWidth   = selected ? 2.5 : 1.5;
                ctx.setLineDash(selected ? [] : [5, 3]);
                ctx.strokeRect(bx, by, bw, bh);

                if (selected) {
                    ctx.fillStyle = "rgba(34,197,94,0.15)";
                    ctx.fillRect(bx, by, bw, bh);
                }

                // Label
                ctx.setLineDash([]);
                const labelText = `${det.label} (${Math.round(det.confidence * 100)}%)`;
                ctx.font        = "bold 12px Inter, sans-serif";
                const tw        = ctx.measureText(labelText).width;
                ctx.fillStyle   = selected ? "#22c55e" : "#f97316";
                ctx.fillRect(bx, by - 18, tw + 8, 18);
                ctx.fillStyle   = "#000";
                ctx.fillText(labelText, bx + 4, by - 4);

                ctx.globalAlpha = 1.0;
            }
        }

        // -- Canvas alignment (mirrors app.js alignCanvas logic) ---------------
        function alignCanvas() {
            if (!state.imageWidth || !state.imageHeight) return;
            const rect = mainVideoImg.getBoundingClientRect();
            if (rect.width === 0 || rect.height === 0) return;

            const containerRect = mainVideoImg.parentElement.getBoundingClientRect();

            const imgRatio = state.imageWidth / state.imageHeight;
            const boxRatio = rect.width / rect.height;
            let renderW, renderH, offsetX, offsetY;

            if (imgRatio > boxRatio) {
                renderW = rect.width; renderH = rect.width / imgRatio;
                offsetX = 0; offsetY = (rect.height - renderH) / 2;
            } else {
                renderH = rect.height; renderW = rect.height * imgRatio;
                offsetX = (rect.width - renderW) / 2; offsetY = 0;
            }

            const finalLeft = (rect.left - containerRect.left) + offsetX;
            const finalTop  = (rect.top - containerRect.top) + offsetY;

            roiCanvas.style.width  = renderW + "px";
            roiCanvas.style.height = renderH + "px";
            roiCanvas.style.left   = finalLeft + "px";
            roiCanvas.style.top    = finalTop + "px";
        }

        window.addEventListener("resize", () => { if (state.active) alignCanvas(); });

        // -- Selection list ----------------------------------------------------
        function updateSelectionList() {
            if (!selectionList) return;
            selectionList.innerHTML = "";

            const allSelected = getEnrolledObjects();
            if (allSelected.length === 0) {
                selectionList.innerHTML = '<p class="guardian-empty-list">No objects selected yet.</p>';
                return;
            }

            for (const obj of allSelected) {
                const item = document.createElement("div");
                item.className = "guardian-selection-item";

                const dot = document.createElement("span");
                dot.className = "guardian-dot";
                dot.style.background = "#22c55e";

                const lbl = document.createElement("span");
                lbl.textContent = obj.label;
                lbl.style.flex = "1";

                const rmBtn = document.createElement("button");
                rmBtn.textContent = "✕";
                rmBtn.className   = "guardian-remove-btn";
                rmBtn.addEventListener("click", () => {
                    state.selectedYoloIds.delete(obj.id);
                    redrawCanvas();
                    updateSelectionList();
                    checkStartReady();
                });

                item.append(dot, lbl, rmBtn);
                selectionList.appendChild(item);
            }
        }

        function getEnrolledObjects() {
            const result = [];

            for (const det of state.scanDetections) {
                if (state.selectedYoloIds.has(det.id)) {
                    result.push({
                        id:              det.id,
                        type:            "yolo",
                        label:           det.label,
                        class_id:        det.class_id,
                        bbox_normalized: det.bbox_normalized,
                    });
                }
            }

            return result;
        }

        function checkStartReady() {
            if (!startGuardBtn) return;
            const hasObjects = state.selectedYoloIds.size > 0;
            startGuardBtn.disabled = !(hasObjects && state.phase === "scan");
        }

        // -- Start Guarding ----------------------------------------------------
        if (startGuardBtn) {
            startGuardBtn.addEventListener("click", handleStartGuarding);
        }

        async function handleStartGuarding() {
            const enrolled = getEnrolledObjects();
            if (enrolled.length === 0) {
                setStatus("Select at least one object first.", "warn");
                return;
            }

            const src = getSourcePayload();

            setStatus("Starting tracking session...", "info");
            startGuardBtn.disabled = true;

            try {
                const res = await fetch("/api/guardian/start", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        stream_id:       src.stream_id,
                        video_id:        src.video_id,
                        filename:        src.filename,
                        watched_objects: enrolled,
                    }),
                });
                if (!res.ok) {
                    const err = await res.json();
                    throw new Error(err.detail || "Failed to start guardian.");
                }
                const data = await res.json();
                state.sessionId = data.session_id;
                enterGuardingPhase();
            } catch (err) {
                setStatus("❌ " + err.message, "error");
                startGuardBtn.disabled = false;
            }
        }

        function enterGuardingPhase() {
            state.phase = "guarding";
            state.knownAlerts.clear();

            // Hide canvas (AI draws boxes on the annotated stream directly)
            roiCanvas.classList.add("hidden");

            // Switch image to the live annotated guardian stream
            mainVideoImg.src = `/api/stream/${state.sessionId}`;

            // Show stop button
            if (stopGuardBtn)  stopGuardBtn.classList.remove("hidden");
            if (startGuardBtn) startGuardBtn.classList.add("hidden");
            if (scanBtn)       scanBtn.disabled = true;

            // Clear alert area
            const alertsContainer = document.getElementById("alerts-container");
            if (alertsContainer) {
                alertsContainer.innerHTML = '<p class="placeholder">Tracking active - no alerts yet.</p>';
            }

            setStatus(`🛡️ Guarding ${getEnrolledObjects().length} object(s)...`, "ok");

            // Start polling
            state.pollingInterval = setInterval(pollGuardianAlerts, 1500);
        }

        // -- Stop Guarding -----------------------------------------------------
        if (stopGuardBtn) {
            stopGuardBtn.addEventListener("click", handleStopGuarding);
        }

        function handleStopGuarding() {
            if (state.sessionId) {
                fetch(`/api/stop_analysis/${state.sessionId}`, { method: "POST" }).catch(() => {});
            }
            resetPhase1(true);
        }

        function resetPhase1(keepScan) {
            if (state.pollingInterval) { clearInterval(state.pollingInterval); state.pollingInterval = null; }

            state.phase     = keepScan && state.scanPreviewUrl ? "scan" : "idle";
            state.sessionId = null;

            if (stopGuardBtn)  stopGuardBtn.classList.add("hidden");
            if (startGuardBtn) { startGuardBtn.classList.remove("hidden"); startGuardBtn.disabled = true; }
            if (scanBtn)       scanBtn.disabled = false;

            roiCanvas.classList.remove("hidden");

            if (keepScan && state.scanPreviewUrl) {
                mainVideoImg.src = state.scanPreviewUrl + "?t=" + Date.now();
                redrawCanvas();
                setStatus("Tracking stopped. Re-scan or adjust selection.", "info");
            } else {
                // Full reset
                state.scanDetections  = [];
                state.selectedYoloIds = new Set();
                ctx.clearRect(0, 0, roiCanvas.width, roiCanvas.height);
                setStatus("Upload a video or connect RTSP, then click Scan Object.", "info");
            }
            updateSelectionList();
        }

        // -- Alert polling -----------------------------------------------------
        function pollGuardianAlerts() {
            if (!state.sessionId) return;
            fetch(`/api/alerts/${state.sessionId}`)
                .then(r => r.json())
                .then(data => {
                    const alerts = data.alerts || [];
                    if (alerts.length === 0) return;

                    const container = document.getElementById("alerts-container");
                    if (!container) return;

                    const placeholder = container.querySelector(".placeholder");
                    if (placeholder) placeholder.remove();

                    for (const alert of alerts) {
                        if (state.knownAlerts.has(alert.id)) continue;
                        state.knownAlerts.add(alert.id);
                        createGuardianAlertCard(alert, container);
                        playBuzzer(2);
                    }
                })
                .catch(() => {});
        }

        function createGuardianAlertCard(alert, container) {
            const card = document.createElement("div");
            card.className = "alert-card";
            card.innerHTML = `
                <div class="alert-time" style="color:#ff4444;font-weight:600;">
                    🚨 OBJECT MISSING: <strong>${alert.object_label || "Unknown"}</strong> @ ${alert.formatted_time}
                </div>
                <video class="alert-inline-video" src="${alert.clip_url}" controls autoplay loop muted></video>
            `;
            container.prepend(card);
        }

        function playBuzzer(times) {
            if (times <= 0) return;
            const audio = new Audio("/assets/buzzer.mp3");
            audio.play().catch(() => {});
            audio.onended = () => playBuzzer(times - 1);
        }

        // -- Status display ----------------------------------------------------
        function setStatus(msg, level = "info") {
            if (!guardianStatus) return;
            const colors = { info: "#94a3b8", ok: "#22c55e", warn: "#f97316", error: "#ef4444" };
            guardianStatus.textContent = msg;
            guardianStatus.style.color = colors[level] || "#94a3b8";
        }

        // -- Source bridge: expose data-attrs for guardian.js to read ----------
        // app.js calls these by setting data-guardian-* on mainVideoImg.
        // We expose a global function that app.js can call without coupling:
        window.__guardianSetSource = function({ stream_id, filename, video_id, width, height }) {
            mainVideoImg.dataset.guardianStreamId = stream_id  || "";
            mainVideoImg.dataset.guardianFilename  = filename   || "";
            mainVideoImg.dataset.guardianVideoId   = video_id   || "";
            if (width)  state.imageWidth  = width;
            if (height) state.imageHeight = height;

            // Auto-reset scan state when source changes
            if (state.phase !== "guarding") {
                state.scanDetections   = [];
                state.selectedYoloIds  = new Set();
                state.phase            = "idle";
                ctx.clearRect(0, 0, roiCanvas.width, roiCanvas.height);
                updateSelectionList();
                checkStartReady();
                if (state.active) setStatus("Source loaded. Click Scan Object to begin.", "ok");
            }
        };

        // -- Init --------------------------------------------------------------
        setStatus("Upload a video or connect RTSP, then click Scan Object.", "info");
        updateSelectionList();

        // Trigger pipeline change check in case room_guardian is already selected
        onPipelineChange();
    }

})();
