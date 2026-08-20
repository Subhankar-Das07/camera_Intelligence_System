document.addEventListener("DOMContentLoaded", () => {
  const uploadBtn = document.getElementById("vr-upload-btn");
  const fileInput = document.getElementById("vr-video-upload");
  const uploadStatus = document.getElementById("vr-upload-status");
  const rtspInput = document.getElementById("vr-rtsp-url");
  const connectBtn = document.getElementById("vr-connect-rtsp-btn");
  const disconnectBtn = document.getElementById("vr-disconnect-btn");
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
      statusMsg.textContent = "RTSP connected";
      statusMsg.style.color = "#10b981";
      disconnectBtn.classList.remove("hidden");
      runBtn.disabled = false;
      placeholder.classList.add("hidden");
      streamImg.src = `/api/raw_stream/${data.stream_id}`;
      streamImg.classList.remove("hidden");
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
    streamImg.src = "";
    placeholder.classList.remove("hidden");
    statusMsg.textContent = "No source selected";
    statusMsg.style.color = "#94a3b8";
  });

  runBtn.addEventListener("click", async () => {
    if (!videoData) return;
    const payload = {
      video_id: videoData.video_id,
      filename: videoData.filename,
      pipeline_name: "vehicle_recognition",
      roi_normalized: [],
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

  function startMode() {
    runBtn.classList.add("hidden");
    stopBtn.classList.remove("hidden");
    statusMsg.textContent = "Live";
    statusMsg.style.color = "#ef4444";
    streamImg.src = `/api/stream/${sessionId}`;
    streamImg.classList.remove("hidden");
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
      })
      .catch(() => {});
  }

  function createCard(det) {
    const card = document.createElement("div");
    card.className = "vr-card";
    card.innerHTML = `
      <div class="vr-plate">${det.plate}</div>
      <div class="vr-visits">Total Visits: <strong>${det.total_visits}</strong></div>
      <div class="vr-time">${new Date().toLocaleTimeString()}</div>
    `;
    return card;
  }
});
