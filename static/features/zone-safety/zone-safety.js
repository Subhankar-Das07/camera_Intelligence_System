document.addEventListener("DOMContentLoaded", () => {
  const ZONE_PIPELINES = new Set([
    "intrusion_detection",
    "new_intrusion",
    "danger_zone",
    "fall_detection",
    "room_guardian",
  ]);

  const videoUpload = document.getElementById("video-upload");
  const uploadBtn = document.getElementById("upload-btn");
  const uploadStatus = document.getElementById("upload-status");
  const placeholderMsg = document.getElementById("placeholder-msg");
  const videoContainer = document.getElementById("video-container");
  const mainVideoImg = document.getElementById("main-video-img");
  const roiCanvas = document.getElementById("roi-canvas");
  const ctx = roiCanvas.getContext("2d");
  const clearRoiBtn = document.getElementById("clear-roi-btn");
  const pipelineSelect = document.getElementById("pipeline-select");
  const runBtn = document.getElementById("run-btn");
  const stopBtn = document.getElementById("stop-btn");
  const alertsContainer = document.getElementById("alerts-container");
  const rtspUrlInput = document.getElementById("rtsp-url");
  const connectRtspBtn = document.getElementById("connect-rtsp-btn");
  const disconnectRtspBtn = document.getElementById("disconnect-rtsp-btn");

  let currentVideoData = null;
  let currentStreamId = null;
  let currentSessionId = null;
  let isRtspMode = false;
  let roiPoints = [];
  let imageWidth = 0;
  let imageHeight = 0;
  let pollingInterval = null;
  const knownAlerts = new Set();

  fetch("/api/pipelines")
    .then((res) => res.json())
    .then((data) => {
      pipelineSelect.innerHTML = "";
      (data.pipelines || [])
        .filter((p) => ZONE_PIPELINES.has(p))
                .forEach((p) => {
          const opt = document.createElement("option");
          opt.value = p;
          let displayName = p;
          if (p === "intrusion_detection") displayName = "Intrusion Detection";
          if (p === "new_intrusion") displayName = "Intrusion Detection (OpenVINO)";
          if (p === "danger_zone") displayName = "Danger Zone";
          if (p === "fall_detection") displayName = "Fall Detection";
          if (p === "room_guardian") displayName = "Object Tracking";
          opt.textContent = displayName;
          pipelineSelect.appendChild(opt);
        });
      if (!pipelineSelect.options.length) {
        pipelineSelect.innerHTML = '<option value="" disabled selected>No zone pipelines</option>';
      }
      checkRunReady();
      updatePipelineUi();
    })
    .catch((err) => console.error("Error fetching pipelines:", err));

  uploadBtn.addEventListener("click", () => videoUpload.click());
  videoUpload.addEventListener("change", (e) => {
    if (e.target.files.length) handleFileUpload(e.target.files[0]);
  });

  async function handleFileUpload(file) {
    uploadStatus.textContent = "Uploading...";
    uploadStatus.style.color = "#94a3b8";
    uploadBtn.disabled = true;
    try {
      await CISSource.disconnectStream(currentStreamId);
      currentStreamId = null;
      const data = await CISSource.uploadVideo(file);
      if (window.__guardianSetSource) window.__guardianSetSource({
          stream_id: null, filename: data.filename, video_id: data.video_id,
          width: data.width, height: data.height,
      });
      currentVideoData = data;
      isRtspMode = false;
      uploadStatus.textContent = file.name;
      uploadStatus.style.color = "#10b981";
      roiPoints = [];
      setupPreview(data.preview_url, data.width, data.height);
    } catch (err) {
      console.error(err);
      uploadStatus.textContent = "Error";
      uploadStatus.style.color = "#ef4444";
    } finally {
      uploadBtn.disabled = false;
    }
  }

  connectRtspBtn.addEventListener("click", async () => {
    const url = rtspUrlInput.value.trim();
    if (!url) {
      rtspUrlInput.style.borderColor = "#ef4444";
      return;
    }
    rtspUrlInput.style.borderColor = "#475569";
    connectRtspBtn.disabled = true;
    connectRtspBtn.textContent = "Connecting...";
    try {
      const data = await CISSource.connectStream(url);
      if (window.__guardianSetSource) window.__guardianSetSource({
          stream_id: data.stream_id, filename: null, video_id: data.stream_id,
          width: data.width, height: data.height,
      });
      currentStreamId = data.stream_id;
      currentVideoData = {
        video_id: data.stream_id,
        filename: url,
        width: data.width,
        height: data.height,
        preview_url: data.preview_url,
      };
      isRtspMode = true;
      connectRtspBtn.textContent = "Connected";
      uploadStatus.textContent = "Using RTSP stream";
      uploadStatus.style.color = "#10b981";
      disconnectRtspBtn.classList.remove("hidden");
      roiPoints = [];
      const srcUrl = data.preview_url || `/api/raw_stream/${data.stream_id}`;
      setupPreview(srcUrl, data.width, data.height);
    } catch (err) {
      connectRtspBtn.textContent = "Connect Stream";
      connectRtspBtn.disabled = false;
      alert("Stream Error: " + err.message);
    }
  });

  disconnectRtspBtn.addEventListener("click", async () => {
    await disconnectCurrentRtsp();
    resetToPlaceholder();
  });

  async function disconnectCurrentRtsp() {
    await CISSource.disconnectStream(currentStreamId);
    currentStreamId = null;
    isRtspMode = false;
    connectRtspBtn.textContent = "Connect Stream";
    connectRtspBtn.disabled = false;
    disconnectRtspBtn.classList.add("hidden");
  }

  function setupPreview(srcUrl, width, height) {
    stopAnalysisMode(false);
    placeholderMsg.classList.add("hidden");
    const finalUrl = srcUrl.includes("/api/raw_stream") ? srcUrl : srcUrl + "?t=" + Date.now();
    mainVideoImg.src = finalUrl;
    roiCanvas.width = width;
    roiCanvas.height = height;
    imageWidth = width;
    imageHeight = height;
    videoContainer.classList.remove("hidden");
    roiPoints = [];
    mainVideoImg.onload = alignCanvas;
    setTimeout(alignCanvas, 100);
    setTimeout(alignCanvas, 500);
    drawPoly();
    checkRunReady();
  }

  function alignCanvas() {
    if (!imageWidth || !imageHeight) return;
    const rect = mainVideoImg.getBoundingClientRect();
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

  roiCanvas.addEventListener("click", (e) => {
    const rect = roiCanvas.getBoundingClientRect();
    roiPoints.push({
      x: (e.clientX - rect.left) * (roiCanvas.width / rect.width),
      y: (e.clientY - rect.top) * (roiCanvas.height / rect.height),
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
  }

  function checkRunReady() {
    const pipeline = pipelineSelect.value;
    const needsRoi = pipeline === "danger_zone" || pipeline === "intrusion_detection" || pipeline === "new_intrusion";
    const hasRoi = roiPoints.length > 2;
    runBtn.disabled = !((needsRoi ? hasRoi : true) && pipeline && currentVideoData);
  }

  function updatePipelineUi() {
    const pipeline = pipelineSelect.value;
    const isDangerZone = pipeline === "danger_zone";
    const needsRoi = pipeline === "danger_zone" || pipeline === "intrusion_detection" || pipeline === "new_intrusion";
    const isGuardian = pipeline === "room_guardian";
    
    const machineSettings = document.getElementById("machine-settings");
    const roiControls = document.getElementById("roi-controls");
    const controlActions = document.querySelector(".control-actions");
    const guardianControls = document.getElementById("guardian-controls");
    
    if (machineSettings) machineSettings.style.display = isDangerZone ? "block" : "none";
    if (roiControls) roiControls.style.display = needsRoi ? "block" : "none";
    if (controlActions) controlActions.style.display = isGuardian ? "none" : "block";
    
    if (guardianControls) {
      if (isGuardian) {
        guardianControls.classList.remove("hidden");
        guardianControls.style.display = "block";
      } else {
        guardianControls.classList.add("hidden");
        guardianControls.style.display = "none";
      }
    }
    
    if (!isGuardian) {
      roiCanvas.style.display = needsRoi ? "block" : "none";
    }
    
    checkRunReady();
  }

  pipelineSelect.addEventListener("change", updatePipelineUi);

  runBtn.addEventListener("click", async () => {
    const roiNormalized = roiPoints.map((p) => [p.x / imageWidth, p.y / imageHeight]);
    const machineActive = document.getElementById("machine-active-toggle")?.checked || false;
    const payload = {
      video_id: currentVideoData.video_id,
      filename: currentVideoData.filename,
      pipeline_name: pipelineSelect.value,
      roi_normalized: roiNormalized,
      config: { machine_active: machineActive },
      stream_id: currentStreamId || null,
    };
    runBtn.disabled = true;
    try {
      const data = await CISSource.startAnalysis(payload);
      currentSessionId = data.session_id;
      startAnalysisMode();
    } catch (err) {
      console.error(err);
      alert("Error starting analysis.");
      runBtn.disabled = false;
    }
  });

  stopBtn.addEventListener("click", () => stopAnalysisMode(true));

  function startAnalysisMode() {
    runBtn.classList.add("hidden");
    stopBtn.classList.remove("hidden");
    mainVideoImg.src = `/api/stream/${currentSessionId}`;
    roiCanvas.classList.add("hidden");
    alertsContainer.innerHTML = '<p class="placeholder">No alerts detected yet.</p>';
    knownAlerts.clear();
    pollingInterval = setInterval(pollAlerts, 1500);
  }

  function stopAnalysisMode(callApi = true) {
    if (pollingInterval) {
      clearInterval(pollingInterval);
      pollingInterval = null;
    }
    if (callApi) CISSource.stopAnalysis(currentSessionId);
    runBtn.classList.remove("hidden");
    stopBtn.classList.add("hidden");
    roiCanvas.classList.remove("hidden");
    if (currentVideoData) {
      if (isRtspMode && currentStreamId) {
        const srcUrl = currentVideoData.preview_url || `/api/raw_stream/${currentStreamId}`;
        mainVideoImg.src = srcUrl.includes("/api/raw_stream") ? srcUrl : srcUrl + "?t=" + Date.now();
      } else if (currentVideoData.preview_url) {
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

  function pollAlerts() {
    if (!currentSessionId) return;
    fetch(`/api/alerts/${currentSessionId}`)
      .then((res) => res.json())
      .then((data) => {
        if (!data.alerts || !data.alerts.length) return;
        const placeholder = alertsContainer.querySelector(".placeholder");
        if (placeholder) placeholder.remove();
        data.alerts.forEach((alert) => {
          if (!knownAlerts.has(alert.id)) {
            knownAlerts.add(alert.id);
            createAlertCard(alert);
          }
        });
      })
      .catch((err) => console.error("Error polling alerts:", err));
  }

  function createAlertCard(alert) {
    const card = document.createElement("div");
    card.className = "alert-card";
    let label = "Intrusion";
    let color = "var(--text-primary)";
    if (alert.severity) {
      label = alert.severity === "SEVERE" ? "SEVERE DANGER" : "WARNING (NEAR)";
      color = alert.severity === "SEVERE" ? "#ff4444" : "#ffa500";
    }
    card.innerHTML = `
      <div class="alert-time" style="color:${color};font-weight:600;">${label} @ ${alert.formatted_time}</div>
      <video class="alert-inline-video" src="${alert.clip_url}" controls autoplay loop muted></video>
    `;
    alertsContainer.prepend(card);
  }
});


