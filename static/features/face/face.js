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
    const refreshBtn     = document.getElementById("refresh-identities-btn");
    const filterTabs     = document.querySelectorAll(".fr-filter-tab");
    const filterKnownBtn = document.getElementById("filter-known-btn");

    const renameModal    = document.getElementById("rename-modal");
    const renameInput    = document.getElementById("rename-input");
    const renameConfirm  = document.getElementById("rename-confirm-btn");
    const renameCancel   = document.getElementById("rename-cancel-btn");
    const renameModalPid = document.getElementById("rename-modal-pid");

    const modeSelector       = document.getElementById("mode-selector");
    const visitorControls    = document.getElementById("fr-visitor-controls");
    const attendanceControls = document.getElementById("fr-attendance-controls");
    const rightPanelTitle    = document.getElementById("right-panel-title");
    
    // Admin toggling
    const adminSwitchBtn     = document.getElementById("fr-admin-switch-btn");
    const adminBackBtn       = document.getElementById("fr-admin-back-btn");
    const adminEntryHeader   = document.getElementById("fr-admin-entry-header");
    const adminContent       = document.getElementById("fr-admin-content");

    if (adminSwitchBtn && adminBackBtn && adminEntryHeader && adminContent) {
        adminSwitchBtn.addEventListener("click", () => {
            adminEntryHeader.classList.add("hidden");
            adminContent.classList.remove("hidden");
        });
        adminBackBtn.addEventListener("click", () => {
            adminContent.classList.add("hidden");
            adminEntryHeader.classList.remove("hidden");
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
            rightPanelTitle.textContent = "🗂️ Attendance List";
            frStartBtn.textContent = "▶ Start Attendance Tracker";
            if (filterKnownBtn) filterKnownBtn.textContent = "Registered";
        } else {
            visitorControls.classList.remove("hidden");
            attendanceControls.classList.add("hidden");
            rightPanelTitle.textContent = "🗂️ Identity Manager";
            frStartBtn.textContent = "▶ Start Recognition";
            if (filterKnownBtn) filterKnownBtn.textContent = "Known";
        }
        loadStats();
        loadIdentities();
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

        // Build payload — always uses stream_id when available
        const payload = {
            video_id:       currentVideoData.video_id,
            filename:       currentVideoData.filename,
            pipeline_name:  "face_recognition",
            roi_normalized: [[0, 0], [1, 0], [1, 1], [0, 1]],
            config:         { mode: getMode() },
            stream_id:      currentStreamId || null,
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

        knownEventIds.clear();
        if (eventsPlaceholder) eventsPlaceholder.style.display = "none";
        pollingInterval = setInterval(_pollEvents, 1500);
    }

    function _stopStream() {
        if (pollingInterval) { clearInterval(pollingInterval); pollingInterval = null; }

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

        streamImg.src = "";
        streamImg.classList.add("hidden");
        placeholder.classList.remove("hidden");
        videoWrapper.classList.remove("active");

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
        const isKnown = alert.status === "known";
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
                renderIdentities();
            })
            .catch(() => {
                identityList.innerHTML = `<div class="fr-identity-empty">Error loading identities.</div>`;
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
                    <div class="fr-identity-meta">${typeBadgeHtml} · ${identity.face_count ?? 1} frames · ${created}</div>
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
                if (!confirm(`Delete ${identity.label}? This cannot be undone.`)) return;
                fetch(`/api/faces/identity/${identity.person_id}?mode=${getMode()}`, { method: "DELETE" })
                    .then(r => r.json())
                    .then(() => { loadStats(); loadIdentities(); })
                    .catch(err => alert("Delete failed: " + err.message));
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
            renderIdentities();
        });
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
