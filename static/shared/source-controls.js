/**
 * Shared video source helpers (upload / RTSP connect / disconnect).
 */
window.CISSource = {
  async uploadVideo(file) {
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch("/api/upload", { method: "POST", body: formData });
    if (!res.ok) throw new Error("Upload failed.");
    return res.json();
  },

  async connectStream(url) {
    const res = await fetch("/api/connect_stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Failed to connect stream.");
    }
    return res.json();
  },

  async disconnectStream(streamId) {
    if (!streamId) return;
    await fetch(`/api/stream/${streamId}`, { method: "DELETE" }).catch(() => {});
  },

  async startAnalysis(payload) {
    const res = await fetch("/api/start_analysis", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const text = await res.text().catch(() => "No response body");
      throw new Error(`HTTP ${res.status}: ${text}`);
    }
    return res.json();
  },

  async stopAnalysis(sessionId) {
    if (!sessionId) return;
    await fetch(`/api/stop_analysis/${sessionId}`, { method: "POST" }).catch(() => {});
  },
};
