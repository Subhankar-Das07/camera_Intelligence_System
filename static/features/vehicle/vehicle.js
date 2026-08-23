document.addEventListener("DOMContentLoaded", () => {
  const uploadBtn = document.getElementById("vr-upload-btn");
  const fileInput = document.getElementById("vr-video-upload");
  const uploadStatus = document.getElementById("vr-upload-status");
  const rtspInput = document.getElementById("vr-rtsp-url");
  const connectBtn = document.getElementById("vr-connect-rtsp-btn");
  const disconnectBtn = document.getElementById("vr-disconnect-rtsp-btn");
  const runBtn = document.getElementById("vr-run-btn");
  const stopBtn = document.getElementById("vr-stop-btn");
  const statusMsg = document.getElementById("vr-status-msg");
  const streamImg = document.getElementById("vr-stream-img");
  const placeholder = document.getElementById("vr-placeholder");
  const logContainer = document.getElementById("vr-log-container");

  let videoData = null;
  let streamId = null;
  let sessionId = null;
  let isRtsp = false;
  let pollInterval = null;
  const seenPlates = new Set();

  const roiCanvas = document.getElementById("vr-roi-canvas");
  const ctx = roiCanvas ? roiCanvas.getContext("2d") : null;
  const clearRoiBtn = document.getElementById("vr-clear-roi-btn");
  const videoContainer = document.getElementById("vr-video-container");

  let roiPoints = [];
  let imageWidth = 1280;
  let imageHeight = 720;

  uploadBtn.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", async (e) => {
    if (!e.target.files.length) return;
    const file = e.target.files[0];
    fileInput.value = "";
    uploadStatus.textContent = "Uploading...";
    uploadStatus.style.color = "#94a3b8";
    uploadBtn.disabled = true;
    try {
      await CISSource.disconnectStream(streamId);
      streamId = null;
      videoData = await CISSource.uploadVideo(file);
      isRtsp = false;
      uploadStatus.textContent = file.name;
      uploadStatus.style.color = "#10b981";
      statusMsg.textContent = "File ready";
      statusMsg.style.color = "#10b981";
      runBtn.disabled = false;
      placeholder.classList.add("hidden");
      streamImg.src = videoData.preview_url + "?t=" + Date.now();
      streamImg.classList.remove("hidden");
      if (videoContainer) {
          videoContainer.classList.remove("hidden");
          const img = new Image();
          img.onload = function() {
              imageWidth = this.width;
              imageHeight = this.height;
              roiCanvas.width = imageWidth;
              roiCanvas.height = imageHeight;
              roiPoints = [];
              drawPoly();
              setTimeout(alignCanvas, 100);
          };
          img.src = videoData.preview_url;
      }
    } catch (err) {
      console.error(err);
      uploadStatus.textContent = "Upload error";
      uploadStatus.style.color = "#ef4444";
    } finally {
      uploadBtn.disabled = false;
    }
  });

  connectBtn.addEventListener("click", async () => {
    const url = rtspInput.value.trim();
    if (!url) {
      rtspInput.style.borderColor = "#ef4444";
      return;
    }
    rtspInput.style.borderColor = "#475569";
    connectBtn.disabled = true;
    statusMsg.textContent = "Connecting...";
    statusMsg.style.color = "#94a3b8";
    try {
      const data = await CISSource.connectStream(url);
      streamId = data.stream_id;
      videoData = {
        video_id: data.stream_id,
        filename: url,
        width: data.width,
        height: data.height,
      };
      isRtsp = true;
      statusMsg.textContent = "Live connected";
      statusMsg.style.color = "#10b981";
      disconnectBtn.classList.remove("hidden");
      runBtn.disabled = false;
      placeholder.classList.add("hidden");
      streamImg.src = `/api/raw_stream/${data.stream_id}`;
      streamImg.classList.remove("hidden");
      if (videoContainer) {
          videoContainer.classList.remove("hidden");
          imageWidth = data.width || 1280;
          imageHeight = data.height || 720;
          roiCanvas.width = imageWidth;
          roiCanvas.height = imageHeight;
          roiPoints = [];
          drawPoly();
          setTimeout(alignCanvas, 500);
      }
    } catch (err) {
      statusMsg.textContent = "Connection failed";
      statusMsg.style.color = "#ef4444";
      alert(err.message);
    } finally {
      connectBtn.disabled = false;
    }
  });

  disconnectBtn.addEventListener("click", async () => {
    await stopMode(true);
    await CISSource.disconnectStream(streamId);
    streamId = null;
    isRtsp = false;
    videoData = null;
    disconnectBtn.classList.add("hidden");
    runBtn.disabled = true;
    streamImg.classList.add("hidden");
    if (videoContainer) videoContainer.classList.add("hidden");
    streamImg.src = "";
    placeholder.classList.remove("hidden");
    statusMsg.textContent = "No source selected";
    statusMsg.style.color = "#94a3b8";
    roiPoints = [];
    if (ctx) drawPoly();
  });

  runBtn.addEventListener("click", async () => {
    if (!videoData) return;
    const roiNormalized = roiPoints.map((p) => [p.x / imageWidth, p.y / imageHeight]);
    const payload = {
      video_id: videoData.video_id,
      filename: videoData.filename,
      pipeline_name: "vehicle_recognition",
      roi_normalized: roiNormalized,
      config: {},
      stream_id: streamId || null,
    };
    runBtn.disabled = true;
    try {
      const data = await CISSource.startAnalysis(payload);
      sessionId = data.session_id;
      startMode();
    } catch (err) {
      console.error(err);
      alert("Error starting vehicle recognition: " + err.message);
      runBtn.disabled = false;
    }
  });

  stopBtn.addEventListener("click", () => stopMode(true));

  // --- ROI CANVAS LOGIC ---
  function alignCanvas() {
    if (!imageWidth || !imageHeight || !roiCanvas) return;
    const rect = streamImg.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;
    const imgRatio = imageWidth / imageHeight;
    const boxRatio = rect.width / rect.height;
    let renderWidth, renderHeight, offsetX, offsetY;
    if (imgRatio > boxRatio) {
      renderWidth = rect.width;
      renderHeight = rect.width / imgRatio;
      offsetX = 0;
      offsetY = (rect.height - renderHeight) / 2;
    } else {
      renderHeight = rect.height;
      renderWidth = rect.height * imgRatio;
      offsetX = (rect.width - renderWidth) / 2;
      offsetY = 0;
    }
    roiCanvas.style.width = renderWidth + "px";
    roiCanvas.style.height = renderHeight + "px";
    roiCanvas.style.left = offsetX + "px";
    roiCanvas.style.top = offsetY + "px";
  }

  window.addEventListener("resize", alignCanvas);
  
  if (streamImg) {
      streamImg.addEventListener("load", () => {
          setTimeout(alignCanvas, 100);
      });
  }

  if (roiCanvas) {
    roiCanvas.addEventListener("click", (e) => {
      // Only accept clicks when the canvas is visible (i.e. not hidden during live stream)
      if (roiCanvas.classList.contains("hidden")) return;
      const rect = roiCanvas.getBoundingClientRect();
      roiPoints.push({
        x: (e.clientX - rect.left) * (roiCanvas.width / rect.width),
        y: (e.clientY - rect.top) * (roiCanvas.height / rect.height),
      });
      drawPoly();
    });
  }

  if (clearRoiBtn) {
    clearRoiBtn.addEventListener("click", () => {
      roiPoints = [];
      drawPoly();
    });
  }

  function drawPoly() {
    if (!ctx) return;
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
    roiPoints.forEach((p) => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
      ctx.fillStyle = "#ff0044";
      ctx.fill();
    });
  }

  function startMode() {
    runBtn.classList.add("hidden");
    stopBtn.classList.remove("hidden");
    statusMsg.textContent = "Live";
    statusMsg.style.color = "#ef4444";
    streamImg.src = `/api/stream/${sessionId}`;
    streamImg.classList.remove("hidden");
    if (roiCanvas) roiCanvas.classList.add("hidden");
    placeholder.classList.add("hidden");
    logContainer.innerHTML = '<p class="placeholder">Scanning…</p>';
    seenPlates.clear();
    pollInterval = setInterval(pollDetections, 2000);
  }

  async function stopMode(callApi = true) {
    if (pollInterval) {
      clearInterval(pollInterval);
      pollInterval = null;
    }
    if (callApi) await CISSource.stopAnalysis(sessionId);
    sessionId = null;
    runBtn.classList.remove("hidden");
    runBtn.disabled = !videoData;
    stopBtn.classList.add("hidden");
    if (roiCanvas) roiCanvas.classList.remove("hidden");
    statusMsg.textContent = videoData ? "Stopped" : "No source selected";
    statusMsg.style.color = "#94a3b8";
  }

  function pollDetections() {
    if (!sessionId) return;
    fetch(`/api/vr_detections/${sessionId}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (!data || !data.detections) return;
        const ph = logContainer.querySelector(".placeholder");
        data.detections.forEach((det) => {
          if (!det.plate || seenPlates.has(det.plate)) return;
          seenPlates.add(det.plate);
          if (ph) ph.remove();
          logContainer.prepend(createCard(det));
        });

        // Trigger Suspicious Vehicle Alerts
        if (data.alerts && data.alerts.length > 0) {
          data.alerts.forEach((alertData) => triggerSuspiciousAlert(alertData));
        }
      })
      .catch(() => {});
  }

  function createCard(det) {
      const card = document.createElement("div");
      card.className = "vr-card";

      const plateText = det.plate || 'N/A';
      const visits = det.total_visits || 1;
      // Use the robust API endpoint — same as admin table, always works once a snap is stored
      const imgPath = `/api/vr/image/snap/${plateText}/${visits}?t=${Date.now()}`;
      const vehType = det.vehicle_type || 'Car';
      const status = det.status || 'Unknown';
      const statusClass = status.toLowerCase() === 'known' ? 'badge-known' : 'badge-unknown';

      card.innerHTML = `
        <img src="${imgPath}" class="vr-card-img" alt="Vehicle" onerror="this.style.display='none'">
        <div class="vr-card-info">
            <div class="vr-plate">${plateText}</div>
            <div class="vr-badges">
                <span class="vr-badge">${vehType}</span>
                <span class="vr-badge ${statusClass}">${status}</span>
            </div>
            <div class="vr-visits">Total Visits: <strong>${visits}</strong></div>
            <div class="vr-time">${new Date().toLocaleTimeString()}</div>
        </div>
      `;
      return card;
  }

  // --- ADMIN PORTAL LOGIC ---
  let currentAdminVehicles = [];
  let currentAdminTab = "pending";

  const adminBtn = document.getElementById("vr-admin-btn");
  const adminModal = document.getElementById("admin-modal");
  const adminCloseBtn = document.getElementById("admin-close-btn");
  const adminTableBody = document.getElementById("admin-table-body");

  if (adminBtn) adminBtn.addEventListener("click", openAdminPortal);
  if (adminCloseBtn) adminCloseBtn.addEventListener("click", () => adminModal.classList.add("hidden"));

  // Bulletproof Event Delegation for Tabs
  document.addEventListener("click", (e) => {
      if (e.target.id === "tab-pending") {
          currentAdminTab = "pending";
          document.getElementById("tab-pending").classList.add("active");
          document.getElementById("tab-known").classList.remove("active");
          populateAdminTable();
      } else if (e.target.id === "tab-known") {
          currentAdminTab = "known";
          document.getElementById("tab-known").classList.add("active");
          document.getElementById("tab-pending").classList.remove("active");
          populateAdminTable();
      }
  });

  async function openAdminPortal() {
      if (!adminModal) return;
      adminModal.classList.remove("hidden");
      adminTableBody.innerHTML = '<tr><td colspan="6" style="text-align:center;">Loading vehicles...</td></tr>';
      try {
          const res = await fetch("/api/vehicles");
          if (res.ok) {
              const data = await res.json();
              currentAdminVehicles = data.vehicles || data;
              populateAdminTable();
          } else {
              adminTableBody.innerHTML = '<tr><td colspan="6" style="text-align:center;color:#ef4444;">Failed to load vehicles.</td></tr>';
          }
      } catch (error) {
          adminTableBody.innerHTML = '<tr><td colspan="6" style="text-align:center;color:#ef4444;">Network error.</td></tr>';
      }
  }

  function populateAdminTable() {
      adminTableBody.innerHTML = "";

      const filteredVehicles = currentAdminVehicles.filter(veh => {
          const isKnown = veh.status && veh.status.toLowerCase() === "known";
          return currentAdminTab === "known" ? isKnown : !isKnown;
      });

      if (!filteredVehicles || filteredVehicles.length === 0) {
          adminTableBody.innerHTML = '<tr><td colspan="6" style="text-align:center;">No vehicles found in this category.</td></tr>';
          return;
      }

      filteredVehicles.forEach(veh => {
          const tr = document.createElement("tr");
          const plateText = veh.plate_number || veh.plate || 'N/A';
          const visits = veh.total_visits || 1;
          const imgPath = `/api/vr/image/snap/${plateText}/${visits}`;
          const isKnown = veh.status && veh.status.toLowerCase() === "known";

          let actionButton = currentAdminTab === "pending"
              ? `<button class="btn secondary" onclick="registerSocietyVehicle('${plateText}')">Approve</button>`
              : `<button class="btn danger" onclick="unregisterSocietyVehicle('${plateText}')">Revoke</button>`;

          tr.innerHTML = `
              <td><img src="${imgPath}" alt="crop" style="width: 80px; height: 50px; object-fit: cover; border-radius: 4px; border: 1px solid #334155;"></td>
              <td><strong>${plateText}</strong></td>
              <td>${veh.vehicle_type || 'Unknown'}</td>
              <td><span class="vr-badge ${isKnown ? 'badge-known' : 'badge-unknown'}">${veh.status || 'Unknown'}</span></td>
              <td>${visits}</td>
              <td>${actionButton}</td>
          `;
          adminTableBody.appendChild(tr);
      });
  }

  window.registerSocietyVehicle = async function(plate) {
      try {
          const res = await fetch(`/api/register_vehicle/${plate}`, { method: "POST" });
          if (res.ok) {
              openAdminPortal();
          } else {
              alert("Failed to register vehicle. Check backend logs.");
          }
      } catch (err) {
          console.error("Error registering vehicle:", err);
      }
  };

  window.unregisterSocietyVehicle = async function(plate) {
      try {
          const res = await fetch(`/api/unregister_vehicle/${plate}`, { method: "POST" });
          if (res.ok) {
              openAdminPortal();
          } else {
              alert("Failed to unregister vehicle. Check backend logs.");
          }
      } catch (err) {
          console.error("Error unregistering vehicle:", err);
      }
  };


  function triggerSuspiciousAlert(alertData) {
      if (document.getElementById(`alert-${alertData.plate}`)) return;

      const alertDiv = document.createElement("div");
      alertDiv.id = `alert-${alertData.plate}`;
      alertDiv.className = "suspicious-alert";

      const imgPath = alertData.image_path || alertData.snapshot_path || '/shell/assets/placeholder.jpg';
      const vidId = `vid-${alertData.plate}-${Date.now()}`;

      alertDiv.innerHTML = `
          <div class="alert-header">⚠️ SUSPICIOUS VEHICLE DETECTED ⚠️</div>
          <div class="alert-body">
              <img src="${imgPath}" onerror="this.style.display='none'">
              <div class="alert-info">
                  <p><strong>Plate:</strong> ${alertData.plate}</p>
                  <p><strong>Type:</strong> ${alertData.type}</p>
                  <p><strong>Loitering:</strong> ${alertData.time_spent}s</p>
                  <p><strong>Status:</strong> <span style="color:#ef4444;font-weight:bold;">UNKNOWN</span></p>
              </div>
          </div>
          <p style="font-size: 0.8rem; margin-bottom: 4px; color: #94a3b8;">Incident Playback (6s):</p>
          <video id="${vidId}" controls autoplay loop muted class="alert-video" style="background: black;"></video>
          <button class="btn danger" style="width:100%; margin-top: 8px;" onclick="this.parentElement.remove()">Dismiss Alert</button>
      `;
      document.body.appendChild(alertDiv);

      // Delay video source assignment to let backend finish writing the webm file.
      // Cache-bust with ?t= so browser never plays a stale clip from a previous session.
      setTimeout(() => {
          const vid = document.getElementById(vidId);
          if (vid) vid.src = `${alertData.clip_path}?t=${Date.now()}`;
      }, 1500);
  }
});
