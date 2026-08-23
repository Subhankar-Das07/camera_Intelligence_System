(() => {
  const API = "/api/site-admin";
  const main = document.getElementById("sa-main");
  const roleSwitch = document.getElementById("role-switch");

  const state = {
    view: "wizard",
    role: localStorage.getItem("cis_role") || "admin",
    status: null,
    cameras: [],
    rules: [],
    alerts: [],
    roi: [],
    previewUrl: "",
    monitorSessionId: null,
    monitorCameraPick: "",
    monitorPoll: null,
    monitorFramePoll: null,
    monitorEventCount: 0,
    monitorSeenEventIds: null,
    monitorEvents: [],
    seekDragging: false,
    wasPlayingBeforeDrag: false,
    seekDebounceTimer: null,
    monitorPaused: false,
    monitorTempId: null,
    monitorTempName: "",
    monitorTempPreview: "",
    monitorSourceMode: "rtsp",
    gateDrawMode: "count_line",
    gateConfig: {
      count_line: [],
      gate_roi: [],
      distance_zones: { near: [], medium: [], far: [] },
      direction_in: "left",
    },
    roiSuggestions: [],
    roiPickMode: false,
    roiSuggestBusy: false,
    scanPreviewWizard: false,
    scanPreviewSidebar: false,
    scanPreviewTimer: null,
    scanPreviewGen: 0,
    scanPreviewApplyRules: false,
    scanPreviewFocusCamId: "",
    scanPreviewLastHeroAt: 0,
  };

  function emptyGateConfig() {
    return {
      count_line: [],
      gate_roi: [],
      distance_zones: { near: [], medium: [], far: [] },
      direction_in: "left",
    };
  }

  const GATE_ZONE_COLORS = {
    count_line: "#ffff00",
    gate_roi: "#c878ff",
    near: "#ff4444",
    medium: "#ffaa00",
    far: "#44cc66",
  };

  function gatePointsForMode(mode) {
    if (mode === "count_line") return state.gateConfig.count_line;
    if (mode === "gate_roi") return state.gateConfig.gate_roi;
    return state.gateConfig.distance_zones[mode] || [];
  }

  function setGatePointsForMode(mode, pts) {
    if (mode === "count_line") state.gateConfig.count_line = pts;
    else if (mode === "gate_roi") state.gateConfig.gate_roi = pts;
    else if (state.gateConfig.distance_zones[mode]) state.gateConfig.distance_zones[mode] = pts;
  }

  function pointInPoly(pt, poly) {
    const [x, y] = pt;
    let inside = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const xi = poly[i][0];
      const yi = poly[i][1];
      const xj = poly[j][0];
      const yj = poly[j][1];
      if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi + 1e-12) + xi) inside = !inside;
    }
    return inside;
  }

  function findSuggestionAt(nx, ny) {
    const hit = [];
    for (const r of state.roiSuggestions) {
      const poly = r.polygon || [];
      if (poly.length >= 3 && pointInPoly([nx, ny], poly)) hit.push(r);
    }
    if (!hit.length) return null;
    hit.sort((a, b) => (a.area || 1) - (b.area || 1));
    return hit[0];
  }

  function drawSuggestions(ctx, canvas) {
    if (!state.roiPickMode || !state.roiSuggestions.length) return;
    state.roiSuggestions.forEach((r, idx) => {
      const pts = r.polygon || [];
      if (pts.length < 3) return;
      const hue = (idx * 47) % 360;
      ctx.strokeStyle = `hsla(${hue}, 80%, 60%, 0.95)`;
      ctx.fillStyle = `hsla(${hue}, 70%, 50%, 0.18)`;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      pts.forEach((p, i) => {
        const x = p[0] * canvas.width;
        const y = p[1] * canvas.height;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      const cx = pts.reduce((s, p) => s + p[0], 0) / pts.length;
      const cy = pts.reduce((s, p) => s + p[1], 0) / pts.length;
      ctx.fillStyle = `hsla(${hue}, 90%, 70%, 0.95)`;
      ctx.font = "11px sans-serif";
      ctx.fillText(String(idx + 1), cx * canvas.width - 4, cy * canvas.height + 4);
    });
  }

  function applySuggestion(region, isGate) {
    const pts = (region.polygon || []).map((p) => [p[0], p[1]]);
    if (pts.length < 3) return false;
    if (isGate) {
      if (state.gateDrawMode === "count_line") {
        alert("Switch to Gate ROI or a distance zone, then click a suggested region.");
        return false;
      }
      setGatePointsForMode(state.gateDrawMode, pts);
    } else {
      state.roi = pts;
    }
    return true;
  }

  function exitRoiPickMode() {
    state.roiPickMode = false;
  }

  function clearGateMode(mode) {
    setGatePointsForMode(mode, []);
  }

  function drawGateCanvas(canvas, img) {
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const drawPoly = (pts, color, closed) => {
      if (!pts.length) return;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      if (color.startsWith("#")) {
        const r = parseInt(color.slice(1, 3), 16);
        const g = parseInt(color.slice(3, 5), 16);
        const b = parseInt(color.slice(5, 7), 16);
        if (closed && pts.length >= 3) {
          ctx.fillStyle = `rgba(${r},${g},${b},0.22)`;
        }
      }
      ctx.beginPath();
      pts.forEach((p, i) => {
        const x = p[0] * canvas.width;
        const y = p[1] * canvas.height;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      if (closed && pts.length >= 3) {
        ctx.closePath();
        ctx.fill();
      }
      ctx.stroke();
      pts.forEach((p) => {
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(p[0] * canvas.width, p[1] * canvas.height, 4, 0, Math.PI * 2);
        ctx.fill();
      });
    };
    drawPoly(state.gateConfig.count_line, GATE_ZONE_COLORS.count_line, false);
    drawPoly(state.gateConfig.gate_roi, GATE_ZONE_COLORS.gate_roi, true);
    ["near", "medium", "far"].forEach((z) => {
      drawPoly(state.gateConfig.distance_zones[z], GATE_ZONE_COLORS[z], true);
    });
    drawSuggestions(ctx, canvas);
  }

  function drawRoiOrGate(canvas, img, isGate) {
    if (isGate) drawGateCanvas(canvas, img);
    else drawRoi(canvas, img);
  }

  function bindGateCanvas(img, canvas) {
    if (!img || !canvas) return;
    canvas.width = img.clientWidth;
    canvas.height = img.clientHeight;
    canvas.classList.toggle("sa-pick-mode", !!state.roiPickMode);
    canvas.onclick = (ev) => {
      const rect = canvas.getBoundingClientRect();
      const nx = (ev.clientX - rect.left) / rect.width;
      const ny = (ev.clientY - rect.top) / rect.height;
      if (state.roiPickMode) {
        const hit = findSuggestionAt(nx, ny);
        if (hit && applySuggestion(hit, true)) {
          exitRoiPickMode();
          canvas.classList.remove("sa-pick-mode");
          drawGateCanvas(canvas, img);
          syncRoiPickUi();
        }
        return;
      }
      const pt = [nx, ny];
      const mode = state.gateDrawMode;
      let pts = gatePointsForMode(mode).slice();
      if (mode === "count_line") {
        if (pts.length >= 2) pts = [];
        pts.push(pt);
      } else {
        pts.push(pt);
      }
      setGatePointsForMode(mode, pts);
      drawGateCanvas(canvas, img);
    };
    drawGateCanvas(canvas, img);
  }

  function bindCanvas(img, canvas, isGate) {
    if (!img || !canvas) return;
    canvas.width = img.clientWidth;
    canvas.height = img.clientHeight;
    if (isGate) {
      bindGateCanvas(img, canvas);
      return;
    }
    canvas.classList.toggle("sa-pick-mode", !!state.roiPickMode);
    canvas.onclick = (ev) => {
      const rect = canvas.getBoundingClientRect();
      const nx = (ev.clientX - rect.left) / rect.width;
      const ny = (ev.clientY - rect.top) / rect.height;
      if (state.roiPickMode) {
        const hit = findSuggestionAt(nx, ny);
        if (hit && applySuggestion(hit, false)) {
          exitRoiPickMode();
          canvas.classList.remove("sa-pick-mode");
          drawRoi(canvas, img);
          syncRoiPickUi();
        }
        return;
      }
      state.roi.push([nx, ny]);
      drawRoi(canvas, img);
    };
    drawRoi(canvas, img);
  }

  function syncRoiPickUi() {
    const hint = document.getElementById("roi-hint");
    const btnSuggest = document.getElementById("r-suggest");
    const btnDone = document.getElementById("r-suggest-done");
    const wrap = document.getElementById("roi-wrap");
    if (btnSuggest) {
      btnSuggest.disabled = state.roiSuggestBusy;
      btnSuggest.textContent = state.roiSuggestBusy ? "Suggesting…" : "Suggest regions";
    }
    if (btnDone) btnDone.style.display = state.roiPickMode ? "inline-block" : "none";
    if (wrap) wrap.classList.toggle("sa-pick-active", !!state.roiPickMode);
    if (hint && state.roiPickMode) {
      hint.textContent =
        "Click a highlighted region to use it as the polygon. Or draw manually after Done.";
    }
  }

  function ensureLightbox() {
    let el = document.getElementById("sa-lightbox");
    if (el) return el;
    el = document.createElement("div");
    el.id = "sa-lightbox";
    el.className = "sa-lightbox hidden";
    el.innerHTML = `
      <div class="sa-lightbox-backdrop" data-close="1"></div>
      <div class="sa-lightbox-panel" role="dialog" aria-modal="true">
        <div class="sa-lightbox-head">
          <h3 id="sa-lb-title"></h3>
          <button type="button" class="btn secondary" id="sa-lb-close">Close</button>
        </div>
        <div class="sa-lightbox-body" id="sa-lb-body"></div>
      </div>`;
    document.body.appendChild(el);
    const close = () => el.classList.add("hidden");
    el.querySelector("[data-close]")?.addEventListener("click", close);
    el.querySelector("#sa-lb-close")?.addEventListener("click", close);
    document.addEventListener("keydown", (ev) => {
      if (ev.key === "Escape" && !el.classList.contains("hidden")) close();
    });
    return el;
  }

  function openLightbox(opts) {
    const el = ensureLightbox();
    const title = document.getElementById("sa-lb-title");
    const body = document.getElementById("sa-lb-body");
    if (title) title.textContent = opts.title || "Preview";
    if (!body) return;
    body.innerHTML = "";
    if (opts.body) {
      if (typeof opts.body === "string") body.innerHTML = opts.body;
      else body.appendChild(opts.body);
    } else if (opts.videoUrl) {
      const v = document.createElement("video");
      v.className = "sa-lightbox-media";
      v.src = opts.videoUrl;
      v.controls = true;
      v.autoplay = true;
      v.playsInline = true;
      body.appendChild(v);
    } else if (opts.canvas) {
      body.appendChild(opts.canvas);
    } else if (opts.imageUrl) {
      const img = document.createElement("img");
      img.className = "sa-lightbox-media";
      img.src = opts.imageUrl;
      img.alt = opts.title || "Preview";
      body.appendChild(img);
    } else {
      body.innerHTML = '<p class="sa-muted">No snapshot available</p>';
    }
    el.classList.remove("hidden");
  }

  function strokePoly(ctx, pts, w, h, color, fill) {
    if (!pts || pts.length < 2) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = fill ? 3 : 2;
    if (fill && pts.length >= 3) {
      ctx.fillStyle = colorWithAlpha(color, 0.22);
    }
    ctx.beginPath();
    pts.forEach((p, i) => {
      const x = p[0] * w;
      const y = p[1] * h;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    if (fill && pts.length >= 3) {
      ctx.closePath();
      ctx.fill();
    }
    ctx.stroke();
  }

  function colorWithAlpha(color, alpha) {
    if (!color) return `rgba(34,211,238,${alpha})`;
    if (color.startsWith("rgb(")) {
      const m = color.match(/rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)/);
      if (m) return `rgba(${m[1]},${m[2]},${m[3]},${alpha})`;
    }
    if (color.startsWith("#") && color.length >= 7) {
      const r = parseInt(color.slice(1, 3), 16);
      const g = parseInt(color.slice(3, 5), 16);
      const b = parseInt(color.slice(5, 7), 16);
      return `rgba(${r},${g},${b},${alpha})`;
    }
    return color;
  }

  function paintRuleOverlays(ctx, w, h, rule) {
    if (!rule) return;
    const accent = rule.css_color || "#22d3ee";
    if (rule.scan_type === "gate_analytics") {
      const gc = rule.gate_config || {};
      strokePoly(ctx, gc.count_line, w, h, "#ffff00", false);
      strokePoly(ctx, gc.gate_roi, w, h, "#c878ff", true);
      const zones = gc.distance_zones || {};
      strokePoly(ctx, zones.near, w, h, "#ff4444", true);
      strokePoly(ctx, zones.medium, w, h, "#ffaa00", true);
      strokePoly(ctx, zones.far, w, h, "#44cc66", true);
      return;
    }
    strokePoly(ctx, rule.roi_normalized || [], w, h, accent, true);
  }

  function drawRuleOntoCanvas(canvas, img, rule) {
    if (!canvas || !img) return;
    const w = canvas.width;
    const h = canvas.height;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, w, h);
    try {
      ctx.drawImage(img, 0, 0, w, h);
    } catch (_) {
      /* cross-origin or not loaded */
    }
    paintRuleOverlays(ctx, w, h, rule);
  }

  function renderRuleThumb(canvas, previewUrl, rule, maxW) {
    if (!canvas || !previewUrl) return Promise.resolve(false);
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        const tw = maxW || 140;
        const scale = tw / img.naturalWidth;
        canvas.width = tw;
        canvas.height = Math.max(40, Math.round(img.naturalHeight * scale));
        drawRuleOntoCanvas(canvas, img, rule);
        resolve(true);
      };
      img.onerror = () => resolve(false);
      img.src = previewUrl;
    });
  }

  function openRuleLightbox(rule, cam) {
    const previewUrl = cam?.preview_url;
    if (!previewUrl) {
      openLightbox({ title: (rule?.name || "Rule") + " — no preview" });
      return;
    }
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.className = "sa-lightbox-media";
      const maxW = Math.min(720, img.naturalWidth);
      const scale = maxW / img.naturalWidth;
      canvas.width = maxW;
      canvas.height = Math.round(img.naturalHeight * scale);
      drawRuleOntoCanvas(canvas, img, rule);
      openLightbox({
        title: `${rule.name || "Rule"} · ${(rule.scan_type || "").replace(/_/g, " ")}`,
        canvas,
      });
    };
    img.onerror = () => openLightbox({ title: rule?.name || "Rule", imageUrl: previewUrl });
    img.src = previewUrl;
  }

  function alertMediaHtml(a) {
    const thumb = a.thumb_url
      ? `<button type="button" class="sa-thumb-btn" data-lb-thumb="${escapeHtml(a.thumb_url)}" data-lb-clip="${escapeHtml(a.clip_url || "")}" data-lb-title="${escapeHtml(a.camera_name || "Alert")}">
          <img class="sa-alert-thumb" src="${escapeHtml(a.thumb_url)}" alt="Event thumbnail">
        </button>`
      : a.clip_url
        ? `<button type="button" class="sa-thumb-btn" data-lb-clip="${escapeHtml(a.clip_url)}" data-lb-title="${escapeHtml(a.camera_name || "Alert")}">
            <video class="sa-alert-thumb" src="${escapeHtml(a.clip_url)}" muted playsinline preload="metadata"></video>
          </button>`
        : `<div class="sa-alert-thumb sa-alert-thumb--empty">No snapshot</div>`;
    const clip = a.clip_url
      ? `<video class="sa-alert-clip" src="${escapeHtml(a.clip_url)}" controls muted playsinline preload="metadata"></video>`
      : "";
    return `<div class="sa-alert-media">${thumb}${clip}</div>`;
  }

  function formatMonitorTime(sec) {
    const s = Math.max(0, Math.floor(sec || 0));
    const m = Math.floor(s / 60);
    const r = s % 60;
    return String(m).padStart(2, "0") + ":" + String(r).padStart(2, "0");
  }

  function monitorControlsHtml(st) {
    if (!st || !st.active) return "";
    if (st.seekable) {
      const cur = formatMonitorTime(st.current_time_sec);
      const dur = formatMonitorTime(st.duration_sec);
      const pos = Math.round((st.position || 0) * 1000);
      const paused = !!st.paused;
      return `<div class="sa-monitor-controls">
          <div class="sa-monitor-controls-row">
            <button class="btn" type="button" id="mon-play" ${paused && isAdmin() ? "" : "disabled"}>Play</button>
            <button class="btn" type="button" id="mon-pause" ${!paused && isAdmin() ? "" : "disabled"}>Pause</button>
            <span id="mon-time" class="sa-monitor-time">${cur} / ${dur}</span>
          </div>
          <input type="range" id="mon-seek" class="sa-monitor-seek" min="0" max="1000" value="${pos}" ${isAdmin() ? "" : "disabled"}>
        </div>`;
    }
    return `<div class="sa-monitor-controls sa-monitor-controls--disabled">
        <p class="sa-muted">Live stream — upload a test video under Cameras to scrub</p>
      </div>`;
  }

  function syncMonitorPlayback(st) {
    const timeEl = document.getElementById("mon-time");
    const seekEl = document.getElementById("mon-seek");
    const playBtn = document.getElementById("mon-play");
    const pauseBtn = document.getElementById("mon-pause");
    if (!st || !st.seekable) return;
    state.monitorPaused = !!st.paused;
    if (timeEl) {
      timeEl.textContent =
        formatMonitorTime(st.current_time_sec) + " / " + formatMonitorTime(st.duration_sec);
    }
    if (seekEl && !state.seekDragging) {
      seekEl.value = String(Math.round((st.position || 0) * 1000));
    }
    if (playBtn) playBtn.disabled = !st.paused || !isAdmin();
    if (pauseBtn) pauseBtn.disabled = st.paused || !isAdmin();
  }

  async function postMonitorSeek(position) {
    if (!state.monitorSessionId || !isAdmin()) return;
    try {
      const r = await api("/monitor/seek/" + state.monitorSessionId, {
        method: "POST",
        json: { position },
      });
      syncMonitorPlayback(r);
    } catch (e) {
      console.warn("seek failed", e);
    }
  }

  async function postMonitorPause(paused) {
    if (!state.monitorSessionId || !isAdmin()) return;
    try {
      const r = await api("/monitor/pause/" + state.monitorSessionId, {
        method: "POST",
        json: { paused },
      });
      syncMonitorPlayback(r);
    } catch (e) {
      alert(e.message);
    }
  }

  function bindMonitorControls(st) {
    if (!st || !st.seekable || !isAdmin()) return;
    const seekEl = document.getElementById("mon-seek");
    const playBtn = document.getElementById("mon-play");
    const pauseBtn = document.getElementById("mon-pause");
    if (!seekEl) return;

    seekEl.addEventListener("mousedown", () => {
      state.seekDragging = true;
      state.wasPlayingBeforeDrag = !state.monitorPaused;
      if (state.wasPlayingBeforeDrag) postMonitorPause(true);
    });
    seekEl.addEventListener("touchstart", () => {
      state.seekDragging = true;
      state.wasPlayingBeforeDrag = !state.monitorPaused;
      if (state.wasPlayingBeforeDrag) postMonitorPause(true);
    }, { passive: true });

    const endDrag = () => {
      if (!state.seekDragging) return;
      state.seekDragging = false;
      postMonitorSeek(Number(seekEl.value) / 1000);
      if (state.wasPlayingBeforeDrag) postMonitorPause(false);
      state.wasPlayingBeforeDrag = false;
    };
    seekEl.addEventListener("mouseup", endDrag);
    seekEl.addEventListener("touchend", endDrag);

    seekEl.addEventListener("input", () => {
      const timeEl = document.getElementById("mon-time");
      if (timeEl && st.duration_sec) {
        const t = (Number(seekEl.value) / 1000) * st.duration_sec;
        timeEl.textContent = formatMonitorTime(t) + " / " + formatMonitorTime(st.duration_sec);
      }
      if (state.seekDebounceTimer) clearTimeout(state.seekDebounceTimer);
      state.seekDebounceTimer = setTimeout(() => {
        postMonitorSeek(Number(seekEl.value) / 1000);
      }, 150);
    });

    playBtn?.addEventListener("click", () => postMonitorPause(false));
    pauseBtn?.addEventListener("click", () => postMonitorPause(true));
  }

  function headers(extra) {
    return Object.assign(
      { "Content-Type": "application/json", "X-CIS-Role": state.role },
      extra || {}
    );
  }

  async function api(path, opts = {}) {
    const method = opts.method || (opts.json || opts.body ? "POST" : "GET");
    const res = await fetch(API + path, {
      method,
      headers: headers(opts.json ? { "Content-Type": "application/json" } : opts.headers),
      body: opts.json ? JSON.stringify(opts.json) : opts.body,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const d = data.detail;
      const msg = typeof d === "string" ? d : d ? JSON.stringify(d) : res.statusText;
      throw new Error(msg);
    }
    return data;
  }

  const ADMIN_VIEWS = ["wizard", "cameras", "rules", "staff", "vehicles", "settings"];

  function isAdmin() {
    return state.role !== "viewer";
  }

  function setRole(role) {
    state.role = role === "viewer" ? "viewer" : "admin";
    localStorage.setItem("cis_role", state.role);
    roleSwitch?.querySelectorAll(".sa-role-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.getAttribute("data-role") === state.role);
    });
    const hideAdmin = !isAdmin();
    document.querySelectorAll(".sa-tabs [data-view]").forEach((b) => {
      const v = b.getAttribute("data-view");
      const locked = hideAdmin && ADMIN_VIEWS.includes(v);
      b.style.display = locked ? "none" : "";
    });
    document.querySelectorAll(".sa-nav-group[data-nav-group='rules']").forEach((g) => {
      g.style.display = hideAdmin ? "none" : "";
    });
  }

  async function switchToRole(role) {
    if (role === state.role) return;
    let pin = "";
    if (role === "admin") {
      const site = state.status?.site || (await api("/site"));
      if ((site.admin_pin || "").trim()) {
        pin = window.prompt("Enter Admin PIN") || "";
        if (!pin) return;
      }
    }
    try {
      await api("/login", { json: { role, pin } });
      setRole(role);
      if (role === "viewer" && ADMIN_VIEWS.includes(state.view)) {
        go("alerts");
      } else if (role === "admin" && state.view === "alerts") {
        render();
      } else {
        render();
      }
    } catch (err) {
      alert(err.message || "Could not switch role");
    }
  }

  function escapeHtml(s) {
    return String(s ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function healthBadge(h) {
    const ok = h === "online";
    const cls = ok ? "ok" : h === "offline" ? "bad" : "";
    return `<span class="sa-badge ${cls}">${escapeHtml(h || "unknown")}</span>`;
  }

  function scanningCameras() {
    return state.cameras.filter(
      (c) =>
        c.enabled !== false &&
        state.rules.some((r) => r.camera_id === c.id && r.enabled !== false)
    );
  }

  function previewCameras() {
    return state.cameras.filter((c) => c.enabled !== false);
  }

  function cameraHasEnabledRules(camId) {
    return state.rules.some((r) => r.camera_id === camId && r.enabled !== false);
  }

  function previewCamerasWithRules() {
    return previewCameras().filter((c) => cameraHasEnabledRules(c.id));
  }

  function previewCamerasWithoutRules() {
    return previewCameras().filter((c) => !cameraHasEnabledRules(c.id));
  }

  function previewHeroApplyRules(camId) {
    return !!(state.scanPreviewApplyRules && camId && cameraHasEnabledRules(camId));
  }

  function activeScanCameraIds() {
    const rt = state.status?.runtime || {};
    if (Array.isArray(rt.active_camera_ids) && rt.active_camera_ids.length) {
      return rt.active_camera_ids;
    }
    const single = rt.active_camera_id || "";
    return single ? [single] : [];
  }

  function resolvePreviewFocusCamId() {
    const cams = previewCameras();
    if (!cams.length) return "";
    const activeIds = activeScanCameraIds();
    if (state.scanPreviewFocusCamId && cams.some((c) => c.id === state.scanPreviewFocusCamId)) {
      return state.scanPreviewFocusCamId;
    }
    const activeId = activeIds[0] || "";
    if (activeId && cams.some((c) => c.id === activeId)) return activeId;
    return cams[0].id;
  }

  function previewSnapshotUrl(camId) {
    return `${API}/cameras/${encodeURIComponent(camId)}/snapshot?t=${Date.now()}`;
  }

  function previewHeroUrl(camId) {
    const apply = previewHeroApplyRules(camId) ? "1" : "0";
    return `${API}/cameras/${encodeURIComponent(camId)}/preview-frame?apply_rules=${apply}&t=${Date.now()}`;
  }

  function scanPreviewWanted() {
    return !!(state.status?.go_live && (state.scanPreviewWizard || state.scanPreviewSidebar));
  }

  function clearScanPreviewTimer() {
    if (state.scanPreviewTimer) {
      clearInterval(state.scanPreviewTimer);
      state.scanPreviewTimer = null;
    }
  }

  function stopScanPreview() {
    state.scanPreviewWizard = false;
    state.scanPreviewSidebar = false;
    state.scanPreviewApplyRules = false;
    state.scanPreviewFocusCamId = "";
    state.scanPreviewLastHeroAt = 0;
    state.scanPreviewGen += 1;
    clearScanPreviewTimer();
    syncScanPreviewUI();
  }

  function updatePreviewHighlights() {
    const activeIds = new Set(activeScanCameraIds());
    const focusId = resolvePreviewFocusCamId();
    document.querySelectorAll(".sa-preview-card:not(.sa-preview-card-compact)").forEach((card) => {
      const camId = card.getAttribute("data-cam-id") || "";
      card.classList.toggle("is-active", activeIds.has(camId));
      card.classList.toggle("is-selected", camId === focusId);
    });
    document.querySelectorAll(".sa-preview-card:not(.sa-preview-card-compact) .sa-preview-badge").forEach((badge) => {
      const card = badge.closest(".sa-preview-card");
      if (card) badge.hidden = !activeIds.has(card.getAttribute("data-cam-id") || "");
    });
    document.querySelectorAll(".sa-preview-card-compact").forEach((card) => {
      const camId = card.getAttribute("data-cam-id") || "";
      const isActive = activeIds.has(camId);
      card.classList.toggle("is-active", isActive);
      const badge = card.querySelector(".sa-preview-badge");
      if (badge) badge.hidden = !isActive;
    });
    const heroLabel = document.getElementById("w-preview-hero-label");
    if (heroLabel) {
      const cam = previewCameras().find((c) => c.id === focusId);
      heroLabel.textContent = cam ? cam.name : "";
    }
    syncPreviewRulesCheckbox();
  }

  function syncPreviewRulesCheckbox() {
    const focusId = resolvePreviewFocusCamId();
    const hasRules = !!(focusId && cameraHasEnabledRules(focusId));
    if (!hasRules && state.scanPreviewApplyRules) {
      state.scanPreviewApplyRules = false;
    }
    const rulesInner = document.getElementById("w-preview-rules-inner");
    const rulesLabel = document.querySelector("#w-preview-grid .sa-preview-rules-check");
    const hint = document.getElementById("w-preview-rules-hint");
    if (rulesInner) {
      rulesInner.disabled = !hasRules;
      rulesInner.checked = hasRules && state.scanPreviewApplyRules;
    }
    if (rulesLabel) {
      rulesLabel.classList.toggle("is-disabled", !hasRules);
    }
    if (hint) {
      hint.hidden = hasRules;
    }
  }

  function buildPreviewCardHtml(cam, compact, noRules) {
    const activeIds = new Set(activeScanCameraIds());
    const focusId = resolvePreviewFocusCamId();
    const isActive = activeIds.has(cam.id);
    const isSelected = !compact && cam.id === focusId;
    const badge = '<span class="sa-preview-badge"' + (isActive ? "" : " hidden") + ">Scanning</span>";
    if (compact) {
      return `<div class="sa-preview-card sa-preview-card-compact${isActive ? " is-active" : ""}" data-cam-id="${escapeHtml(cam.id)}" title="${escapeHtml(cam.name)}">
        <img alt="" loading="lazy" />
        ${badge}
      </div>`;
    }
    const noRulesCls = noRules ? " sa-preview-card-no-rules" : "";
    return `<div class="sa-preview-card${noRulesCls}${isActive ? " is-active" : ""}${isSelected ? " is-selected" : ""}" data-cam-id="${escapeHtml(cam.id)}" role="button" tabindex="0" title="Show in large preview">
      <div class="sa-preview-thumb"><img alt="" loading="lazy" /></div>
      <div class="sa-preview-label">${escapeHtml(cam.name)}${badge}</div>
    </div>`;
  }

  function buildPreviewGridSectionsHtml() {
    const withRules = previewCamerasWithRules();
    const withoutRules = previewCamerasWithoutRules();
    const withSection = `<section class="sa-preview-group">
      <h4 class="sa-preview-group-title">With rules</h4>
      ${
        withRules.length
          ? `<div class="sa-preview-grid" data-group="with-rules">${withRules.map((c) => buildPreviewCardHtml(c, false, false)).join("")}</div>`
          : `<p class="sa-preview-group-empty">No cameras have rules yet.</p>`
      }
    </section>`;
    const noSection = `<section class="sa-preview-group">
      <h4 class="sa-preview-group-title">No rules yet</h4>
      ${
        withoutRules.length
          ? `<div class="sa-preview-grid" data-group="no-rules">${withoutRules.map((c) => buildPreviewCardHtml(c, false, true)).join("")}</div>`
          : `<p class="sa-preview-group-empty">All cameras have rules.</p>`
      }
    </section>`;
    return withSection + noSection;
  }

  function bindPreviewGridClicks() {
    document.querySelectorAll("#w-preview-grid .sa-preview-card[data-cam-id]").forEach((card) => {
      const camId = card.getAttribute("data-cam-id");
      if (!camId) return;
      card.onclick = () => {
        state.scanPreviewFocusCamId = camId;
        updatePreviewHighlights();
        state.scanPreviewGen += 1;
        refreshHeroPreview(state.scanPreviewGen);
      };
    });
  }

  function refreshHeroPreview(gen) {
    if (gen !== state.scanPreviewGen || !scanPreviewWanted() || !state.scanPreviewWizard) return Promise.resolve();
    const camId = resolvePreviewFocusCamId();
    const img = document.getElementById("w-preview-hero-img");
    if (!img || !camId) return Promise.resolve();
    const url = previewHeroUrl(camId);
    return new Promise((resolve) => {
      img.onerror = () => {
        img.classList.add("sa-preview-error");
        resolve();
      };
      img.onload = () => {
        img.classList.remove("sa-preview-error");
        resolve();
      };
      img.src = url;
      state.scanPreviewLastHeroAt = Date.now();
      updatePreviewHighlights();
    });
  }

  async function refreshPreviewFrames(gen) {
    const cams = previewCameras();
    for (let i = 0; i < cams.length; i += 1) {
      if (gen !== state.scanPreviewGen || !scanPreviewWanted()) return;
      const cam = cams[i];
      const url = previewSnapshotUrl(cam.id);
      document.querySelectorAll(`.sa-preview-card[data-cam-id="${CSS.escape(cam.id)}"] img`).forEach((img) => {
        img.onerror = () => img.classList.add("sa-preview-error");
        img.onload = () => img.classList.remove("sa-preview-error");
        img.src = url;
      });
      if (i < cams.length - 1) {
        await new Promise((resolve) => window.setTimeout(resolve, 300));
      }
    }
    const heroInterval = previewHeroApplyRules(resolvePreviewFocusCamId()) ? 6000 : 4000;
    const due = Date.now() - (state.scanPreviewLastHeroAt || 0) >= heroInterval;
    if (state.scanPreviewWizard && due) {
      await refreshHeroPreview(gen);
    }
    updatePreviewHighlights();
  }

  function ensureScanPreviewLoop() {
    if (!scanPreviewWanted() || document.hidden) {
      clearScanPreviewTimer();
      return;
    }
    if (state.scanPreviewTimer) return;
    const tick = async () => {
      if (!scanPreviewWanted() || document.hidden) return;
      try {
        state.status = await api("/status");
        if (!state.status?.go_live) {
          stopScanPreview();
          return;
        }
        const live = !!state.status.go_live;
        document.getElementById("live-dot").className = "sa-dot " + (live ? "on" : "off");
        document.getElementById("live-label").textContent = live ? "Scanning" : "Idle";
      } catch (_) {
        /* ignore */
      }
      await refreshPreviewFrames(state.scanPreviewGen);
      updatePreviewHighlights();
    };
    tick();
    state.scanPreviewTimer = setInterval(tick, 4000);
  }

  function syncScanPreviewUI() {
    const live = !!state.status?.go_live;
    const toggleBtn = document.getElementById("live-preview-toggle");
    const strip = document.getElementById("live-preview-strip");

    if (toggleBtn) {
      toggleBtn.hidden = !live;
      toggleBtn.textContent = state.scanPreviewSidebar ? "Hide preview" : "Show preview";
    }
    if (strip) {
      strip.hidden = !live || !state.scanPreviewSidebar;
      if (!strip.hidden) {
        const cams = previewCameras();
        const stripKey = cams.map((c) => c.id).join("|");
        if (strip.dataset.stripKey !== stripKey) {
          strip.dataset.stripKey = stripKey;
          strip.innerHTML = cams.length
            ? cams.map((c) => buildPreviewCardHtml(c, true)).join("")
            : '<p class="sa-preview-placeholder">No enabled cameras</p>';
        }
      }
    }

    const wToggle = document.getElementById("w-preview-toggle");
    const wGrid = document.getElementById("w-preview-grid");
    if (wToggle) {
      wToggle.textContent = state.scanPreviewWizard ? "Hide preview" : "Show preview";
    }
    if (wGrid) {
      wGrid.hidden = !live || !state.scanPreviewWizard;
      if (!wGrid.hidden) {
        const withRules = previewCamerasWithRules();
        const withoutRules = previewCamerasWithoutRules();
        const gridKey = `${withRules.map((c) => c.id).join("|")}||${withoutRules.map((c) => c.id).join("|")}`;
        if (wGrid.dataset.gridKey !== gridKey || !wGrid.querySelector(".sa-preview-group")) {
          wGrid.dataset.gridKey = gridKey;
          const total = withRules.length + withoutRules.length;
          wGrid.innerHTML = total
            ? `<label class="sa-preview-rules-check"><input type="checkbox" id="w-preview-rules-inner"${state.scanPreviewApplyRules ? " checked" : ""}> Apply rules on preview</label>
               <p class="sa-preview-rules-hint" id="w-preview-rules-hint" hidden>Select a camera with rules to preview overlays.</p>
               <div class="sa-preview-hero">
                 <img id="w-preview-hero-img" alt="" />
                 <span class="sa-preview-hero-label" id="w-preview-hero-label"></span>
               </div>
               ${buildPreviewGridSectionsHtml()}`
            : '<p class="sa-preview-placeholder">Add enabled cameras to preview.</p>';
          const rulesInner = document.getElementById("w-preview-rules-inner");
          if (rulesInner) {
            rulesInner.onchange = () => {
              if (rulesInner.disabled) return;
              state.scanPreviewApplyRules = rulesInner.checked;
              state.scanPreviewLastHeroAt = 0;
              state.scanPreviewGen += 1;
              refreshHeroPreview(state.scanPreviewGen);
            };
          }
          bindPreviewGridClicks();
          syncPreviewRulesCheckbox();
        } else {
          syncPreviewRulesCheckbox();
        }
        updatePreviewHighlights();
      }
    }

    if (scanPreviewWanted()) {
      ensureScanPreviewLoop();
    } else {
      clearScanPreviewTimer();
    }
  }

  function toggleScanPreviewSidebar() {
    state.scanPreviewSidebar = !state.scanPreviewSidebar;
    syncScanPreviewUI();
    if (scanPreviewWanted()) {
      state.scanPreviewGen += 1;
      refreshPreviewFrames(state.scanPreviewGen);
    }
  }

  function toggleScanPreviewWizard() {
    state.scanPreviewWizard = !state.scanPreviewWizard;
    if (state.scanPreviewWizard && !state.scanPreviewFocusCamId) {
      state.scanPreviewFocusCamId = resolvePreviewFocusCamId();
    }
    syncScanPreviewUI();
    if (scanPreviewWanted()) {
      state.scanPreviewLastHeroAt = 0;
      state.scanPreviewGen += 1;
      refreshPreviewFrames(state.scanPreviewGen);
    }
  }

  async function refresh() {
    state.status = await api("/status");
    const live = !!state.status.go_live;
    if (!live && (state.scanPreviewWizard || state.scanPreviewSidebar)) {
      state.scanPreviewWizard = false;
      state.scanPreviewSidebar = false;
      state.scanPreviewApplyRules = false;
      state.scanPreviewFocusCamId = "";
      state.scanPreviewLastHeroAt = 0;
      state.scanPreviewGen += 1;
      clearScanPreviewTimer();
    }
    document.getElementById("live-dot").className = "sa-dot " + (live ? "on" : "off");
    document.getElementById("live-label").textContent = live ? "Scanning" : "Idle";
    syncScanPreviewUI();
    if (scanPreviewWanted()) {
      updatePreviewHighlights();
    }
  }

  async function loadLists() {
    const [c, r] = await Promise.all([api("/cameras"), api("/rules")]);
    state.cameras = c.cameras || [];
    state.rules = r.rules || [];
  }

  function syncNavActive(view) {
    const rulesFamily = view === "rules" || view === "staff" || view === "vehicles";
    document.querySelectorAll(".sa-tabs [data-view]").forEach((b) => {
      const v = b.getAttribute("data-view");
      if (b.classList.contains("sa-nav-parent") && v === "rules") {
        b.classList.toggle("active", rulesFamily);
      } else {
        b.classList.toggle("active", v === view);
      }
    });
    document.querySelectorAll(".sa-nav-group[data-nav-group='rules']").forEach((g) => {
      g.classList.toggle("is-open", rulesFamily);
    });
  }

  function go(view) {
    if (state.view === "monitor" && view !== "monitor") {
      stopMonitorIfAny();
    }
    if (state.view === "wizard" && view !== "wizard") {
      state.scanPreviewWizard = false;
      if (!state.scanPreviewSidebar) {
        state.scanPreviewGen += 1;
        clearScanPreviewTimer();
      }
    }
    if (!isAdmin() && ADMIN_VIEWS.includes(view)) {
      view = "alerts";
    }
    state.view = view;
    syncNavActive(view);
    render();
    syncScanPreviewUI();
  }

  function render() {
    const views = { wizard, cameras, rules, staff, vehicles, monitor, alerts, reports, settings };
    (views[state.view] || wizard)();
  }

  function scanCatalog() {
    return state.status?.scan_catalog || [];
  }

  function catalogById(id) {
    return scanCatalog().find((e) => e.id === id) || null;
  }

  function buildScanTypeSelectHtml(selectedId) {
    const catalog = scanCatalog();
    const fallback = (state.status?.scan_types || ["intrusion"]).map((id) => ({
      id,
      label: id.replace(/_/g, " "),
      category: "Other",
      status: "available",
    }));
    const entries = catalog.length ? catalog : fallback;
    const groups = {};
    entries.forEach((e) => {
      const cat = e.category || "Other";
      if (!groups[cat]) groups[cat] = [];
      groups[cat].push(e);
    });
    const order = ["People & safety", "Entrance", "Vehicles", "Staff", "Property", "Safety gear", "Other"];
    const cats = order.filter((c) => groups[c]).concat(Object.keys(groups).filter((c) => !order.includes(c)));
    return cats
      .map((cat) => {
        const opts = groups[cat]
          .map((e) => {
            const soon = e.status !== "available";
            const label = soon ? `${e.label} (soon)` : e.label;
            const sel = e.id === selectedId ? " selected" : "";
            return `<option value="${escapeHtml(e.id)}"${sel}>${escapeHtml(label)}</option>`;
          })
          .join("");
        return `<optgroup label="${escapeHtml(cat)}">${opts}</optgroup>`;
      })
      .join("");
  }

  function openScanCatalogHelp() {
    const rows = scanCatalog()
      .map((e) => {
        const st = e.status === "available" ? "Available" : "Coming soon";
        return `<tr>
          <td>${escapeHtml(e.label)}</td>
          <td>${escapeHtml(e.typical_need)}</td>
          <td>${escapeHtml(e.fit)}</td>
          <td><span class="sa-badge ${e.status === "available" ? "ok" : ""}">${st}</span></td>
        </tr>`;
      })
      .join("");
    const table = document.createElement("div");
    table.className = "sa-catalog-help";
    table.innerHTML = `
      <p class="sa-muted">Rules are the heart of the product. Pick an Available type to run today. Coming soon types are shown for planning only.</p>
      <div class="sa-catalog-scroll">
        <table class="sa-table sa-catalog-table">
          <thead><tr><th>Label</th><th>Typical need</th><th>Fit for you</th><th>Status</th></tr></thead>
          <tbody>${rows || '<tr><td colspan="4" class="sa-muted">Catalog not loaded. Refresh the page.</td></tr>'}</tbody>
        </table>
      </div>`;
    openLightbox({ title: "Scan types", body: table });
  }

  async function stopMonitorIfAny() {
    clearMonitorFramePoll();
    resetMonitorEventTracking();
    if (!state.monitorSessionId) return;
    const sid = state.monitorSessionId;
    state.monitorSessionId = null;
    try {
      await api("/monitor/stop/" + sid, { method: "POST" });
    } catch (_) {
      /* ignore */
    }
  }

  function wizard() {
    const s = state.status?.site || {};
    main.innerHTML = `
      <div class="sa-h">
        <div>
          <h2>Setup wizard</h2>
          <p>Admin configures the site once. After go-live, feeds scan by rules.</p>
        </div>
      </div>
      <div class="sa-grid">
        <div class="sa-card"><h3>Cameras</h3><div class="sa-stat">${state.status?.cameras ?? 0}</div></div>
        <div class="sa-card"><h3>Rules</h3><div class="sa-stat">${state.status?.rules ?? 0}</div></div>
        <div class="sa-card"><h3>Recent alerts</h3><div class="sa-stat">${state.status?.alerts ?? 0}</div></div>
        <div class="sa-card"><h3>WhatsApp</h3><p>${state.status?.whatsapp_configured ? "API configured" : "Not configured (web alerts still work)"}</p></div>
      </div>
      <div class="sa-steps">
        <section class="sa-step">
          <h3>Site profile</h3>
          <div class="sa-form">
            <div class="sa-field"><label>Site name</label><input class="text-input" id="w-name" value="${escapeHtml(s.name)}"></div>
            <div class="sa-field"><label>Type</label>
              <select id="w-type" class="text-input">
                <option value="shop"${s.type === "shop" ? " selected" : ""}>Shop</option>
                <option value="home"${s.type === "home" ? " selected" : ""}>Home</option>
                <option value="factory"${s.type === "factory" ? " selected" : ""}>Factory</option>
              </select>
            </div>
            <div class="sa-field"><label>Timezone</label><input class="text-input" id="w-tz" value="${escapeHtml(s.timezone || "Asia/Kolkata")}"></div>
            <button class="btn primary" id="w-save-site" type="button"${isAdmin() ? "" : " disabled"}>Save site</button>
          </div>
        </section>
        <section class="sa-step">
          <h3>Add a feed</h3>
          <p class="sa-muted">Use Cameras to add live RTSP or upload a test video, then come back.</p>
          <button class="btn secondary" type="button" id="w-to-cam">Open Cameras</button>
        </section>
        <section class="sa-step">
          <h3>Draw a rule</h3>
          <p class="sa-muted">Pick a camera, choose what to scan, draw an area if needed.</p>
          <button class="btn secondary" type="button" id="w-to-rules">Open Rules</button>
        </section>
        <section class="sa-step">
          <h3>Verify live</h3>
          <p class="sa-muted">Open Monitor to see all rules applied on the video before go-live.</p>
          <button class="btn secondary" type="button" id="w-to-mon">Open Monitor</button>
        </section>
        <section class="sa-step">
          <h3>Go live</h3>
          <p class="sa-muted">Starts continuous scanning. Survives app restart while go-live stays on.</p>
          <div class="sa-row">
            <button class="btn primary" type="button" id="w-live"${isAdmin() ? "" : " disabled"}>${s.go_live ? "Stop scanning" : "Start scanning"}</button>
            <span class="sa-muted" id="w-msg"></span>
          </div>
          ${
            s.go_live
              ? `<div class="sa-golive-preview">
            <p class="sa-muted">On-demand preview — verify all cameras while scanning. Stops when hidden or you leave this page.</p>
            <button class="btn secondary" type="button" id="w-preview-toggle">${state.scanPreviewWizard ? "Hide preview" : "Show preview"}</button>
            <div id="w-preview-grid" class="sa-golive-preview-grid"${state.scanPreviewWizard ? "" : " hidden"}></div>
          </div>`
              : ""
          }
        </section>
      </div>`;
    document.getElementById("w-to-cam").onclick = () => go("cameras");
    document.getElementById("w-to-rules").onclick = () => go("rules");
    document.getElementById("w-to-mon").onclick = () => go("monitor");
    document.getElementById("w-save-site").onclick = async () => {
      try {
        await api("/site", {
          json: {
            name: document.getElementById("w-name").value,
            type: document.getElementById("w-type").value,
            timezone: document.getElementById("w-tz").value,
          },
        });
        await refresh();
        document.getElementById("w-msg") && (document.getElementById("w-msg").textContent = "Saved");
      } catch (e) {
        alert(e.message);
      }
    };
    document.getElementById("w-live").onclick = async () => {
      try {
        const turningOff = !!s.go_live;
        await api("/site", {
          json: { go_live: !s.go_live, setup_complete: true },
        });
        if (turningOff) {
          stopScanPreview();
        }
        await refresh();
        await loadLists();
        wizard();
      } catch (e) {
        alert(e.message);
      }
    };
    document.getElementById("w-preview-toggle") &&
      (document.getElementById("w-preview-toggle").onclick = toggleScanPreviewWizard);
    syncScanPreviewUI();
    if (state.scanPreviewWizard && scanPreviewWanted()) {
      state.scanPreviewGen += 1;
      refreshPreviewFrames(state.scanPreviewGen);
    }
  }

  function cameras() {
    const rows = state.cameras
      .map(
        (c) => `
        <tr>
          <td>${escapeHtml(c.name)}</td>
          <td><span class="sa-badge ${c.type}">${escapeHtml(c.type)}</span></td>
          <td>${healthBadge(c.health)}</td>
          <td>${c.enabled === false ? "Off" : "On"}</td>
          <td>
            ${
              isAdmin()
                ? `<button class="btn secondary" data-toggle="${c.id}" type="button">Toggle</button>
                   <button class="btn danger" data-del="${c.id}" type="button">Remove</button>`
                : ""
            }
          </td>
        </tr>`
      )
      .join("");
    main.innerHTML = `
      <div class="sa-h"><div><h2>Cameras</h2><p>Add DVR (HTTP snapshot) or NVR (RTSP). Upload a video to test without CCTV.</p></div></div>
      ${
        isAdmin()
          ? `<div class="sa-card" style="margin-bottom:1rem;">
        <h3>Add live CCTV</h3>
        <div class="sa-form" style="margin-top:0.7rem;">
          <div class="sa-field"><label>Name</label><input class="text-input" id="c-name" placeholder="Kitchen"></div>
          <div class="sa-field">
            <label>Device type</label>
            <select class="text-input" id="c-device-type">
              <option value="dvr">DVR (HTTP snapshot)</option>
              <option value="rtsp">NVR (RTSP)</option>
            </select>
          </div>
          <div id="c-dvr-fields">
            <div class="sa-field">
              <label>HTTP snapshot URL</label>
              <input class="text-input" id="c-snapshot-url" placeholder="http://192.168.1.100/ISAPI/Streaming/channels/101/picture">
              <p class="sa-muted sa-field-hint">Hikvision channels: 101, 201, 301 … 801 (cam 1–8 main stream)</p>
            </div>
            <div class="sa-field"><label>Username</label><input class="text-input" id="c-http-user" placeholder="admin" autocomplete="username"></div>
            <div class="sa-field"><label>Password</label><input class="text-input" id="c-http-pass" type="password" placeholder="••••••" autocomplete="current-password"></div>
          </div>
          <div id="c-rtsp-fields" hidden>
            <div class="sa-field"><label>RTSP URL</label><input class="text-input" id="c-rtsp-url" placeholder="rtsp://user:pass@nvr:554/Streaming/Channels/101"></div>
          </div>
          <div class="sa-row">
            <button class="btn secondary" type="button" id="c-test">Test connection</button>
            <button class="btn primary" type="button" id="c-save" style="width:auto;">Save camera</button>
            <span id="c-msg" class="sa-muted"></span>
          </div>
          <div id="c-preview-wrap" class="sa-camera-preview" hidden>
            <img id="c-preview-img" alt="Camera test preview">
            <span id="c-preview-meta" class="sa-muted"></span>
          </div>
        </div>
        <div class="divider-or">— OR —</div>
        <h3>Upload test video</h3>
        <div class="sa-row" style="margin-top:0.6rem;">
          <input type="file" id="c-file" accept="video/mp4,video/avi,video/quicktime">
          <button class="btn secondary" type="button" id="c-upload">Upload to library</button>
        </div>
      </div>`
          : ""
      }
      <table class="sa-table">
        <thead><tr><th>Name</th><th>Type</th><th>Health</th><th>Enabled</th><th></th></tr></thead>
        <tbody>${rows || '<tr><td colspan="5" class="sa-muted">No cameras yet</td></tr>'}</tbody>
      </table>`;

    const msg = document.getElementById("c-msg");
    const deviceType = document.getElementById("c-device-type");
    const dvrFields = document.getElementById("c-dvr-fields");
    const rtspFields = document.getElementById("c-rtsp-fields");
    const previewWrap = document.getElementById("c-preview-wrap");
    const previewImg = document.getElementById("c-preview-img");
    const previewMeta = document.getElementById("c-preview-meta");

    function syncCameraDeviceFields() {
      const isDvr = deviceType?.value === "dvr";
      if (dvrFields) dvrFields.hidden = !isDvr;
      if (rtspFields) rtspFields.hidden = isDvr;
      state.previewUrl = "";
      if (previewWrap) previewWrap.hidden = true;
      if (previewImg) previewImg.removeAttribute("src");
      if (previewMeta) previewMeta.textContent = "";
      if (msg) {
        msg.textContent = "";
        msg.className = "sa-muted";
      }
    }

    function cameraTestPayload() {
      const isDvr = deviceType?.value === "dvr";
      const base = { name: document.getElementById("c-name")?.value || "Camera" };
      if (isDvr) {
        return {
          ...base,
          type: "dvr",
          snapshot_url: document.getElementById("c-snapshot-url")?.value || "",
          http_user: document.getElementById("c-http-user")?.value || "",
          http_password: document.getElementById("c-http-pass")?.value || "",
        };
      }
      return {
        ...base,
        type: "rtsp",
        rtsp_url: document.getElementById("c-rtsp-url")?.value || "",
      };
    }

    function showCameraPreview(r) {
      state.previewUrl = r.preview_url || "";
      if (!previewWrap || !previewImg) return;
      if (state.previewUrl) {
        previewWrap.hidden = false;
        previewImg.src = state.previewUrl + "?t=" + Date.now();
        if (previewMeta) {
          previewMeta.textContent = r.width && r.height ? `${r.width}×${r.height}` : "";
        }
      } else {
        previewWrap.hidden = true;
      }
    }

    if (deviceType) {
      deviceType.onchange = syncCameraDeviceFields;
      syncCameraDeviceFields();
    }

    const testBtn = document.getElementById("c-test");
    if (testBtn) {
      testBtn.onclick = async () => {
        msg.textContent = "Testing…";
        msg.className = "sa-muted";
        try {
          const r = await api("/cameras/test-connection", { json: cameraTestPayload() });
          showCameraPreview(r);
          msg.textContent = deviceType?.value === "dvr" ? "Online — snapshot received" : "Online — frames received";
          msg.className = "sa-ok";
        } catch (e) {
          state.previewUrl = "";
          if (previewWrap) previewWrap.hidden = true;
          if (previewImg) previewImg.removeAttribute("src");
          msg.textContent = e.message;
          msg.className = "sa-error";
        }
      };
    }
    const saveBtn = document.getElementById("c-save");
    if (saveBtn) {
      saveBtn.onclick = async () => {
        try {
          const payload = {
            ...cameraTestPayload(),
            preview_url: state.previewUrl || "",
            enabled: true,
            health: state.previewUrl ? "online" : "unknown",
          };
          await api("/cameras", { json: payload });
          await loadLists();
          cameras();
        } catch (e) {
          alert(e.message);
        }
      };
    }
    const up = document.getElementById("c-upload");
    if (up) {
      up.onclick = async () => {
        const f = document.getElementById("c-file").files[0];
        if (!f) return alert("Choose a video file");
        const fd = new FormData();
        fd.append("file", f);
        const res = await fetch(API + "/cameras/upload", {
          method: "POST",
          headers: { "X-CIS-Role": state.role },
          body: fd,
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) return alert(data.detail || "Upload failed");
        await loadLists();
        cameras();
      };
    }
    main.querySelectorAll("[data-del]").forEach((b) => {
      b.onclick = async () => {
        if (!confirm("Remove this camera?")) return;
        await api("/cameras/" + b.getAttribute("data-del"), { method: "DELETE" });
        await loadLists();
        cameras();
      };
    });
    main.querySelectorAll("[data-toggle]").forEach((b) => {
      b.onclick = async () => {
        const id = b.getAttribute("data-toggle");
        const cam = state.cameras.find((c) => c.id === id);
        if (!cam) return;
        await api("/cameras/" + id, {
          method: "PUT",
          json: { ...cam, enabled: cam.enabled === false },
        });
        await loadLists();
        cameras();
      };
    });
  }

  function rules() {
    const camOpts = state.cameras
      .map((c) => `<option value="${c.id}">${escapeHtml(c.name)} (${c.type})</option>`)
      .join("");
    const typeOpts = buildScanTypeSelectHtml("intrusion");
    const rows = state.rules
      .map((r) => {
        const cam = state.cameras.find((c) => c.id === r.camera_id);
        const entry = catalogById(r.scan_type);
        const scanLabel = entry?.label || (r.scan_type || "").replace(/_/g, " ");
        const roiNote =
          r.scan_type === "gate_analytics"
            ? `${(r.gate_config?.count_line || []).length}/2 line`
            : `${(r.roi_normalized || []).length} pts`;
        const hasPreview = !!cam?.preview_url;
        const thumbCell = hasPreview
          ? `<td class="sa-rule-thumb-cell">
              <button type="button" class="sa-thumb-btn" data-rule-preview="${escapeHtml(r.id)}" title="View rule on footage">
                <canvas class="sa-rule-thumb" data-rule-cv="${escapeHtml(r.id)}" width="140" height="80"></canvas>
              </button>
            </td>`
          : `<td class="sa-rule-thumb-cell"><span class="sa-alert-thumb sa-alert-thumb--empty">No preview</span></td>`;
        return `<tr>
          ${thumbCell}
          <td>${escapeHtml(r.name)}</td>
          <td>${escapeHtml(cam?.name || r.camera_id)}</td>
          <td>${escapeHtml(scanLabel)}</td>
          <td>${roiNote}</td>
          <td>${r.enabled === false ? "Off" : "On"}</td>
          <td>${isAdmin() ? `<button class="btn danger" data-delr="${r.id}" type="button">Remove</button>` : ""}</td>
        </tr>`;
      })
      .join("");
    main.innerHTML = `
      <div class="sa-h">
        <div>
          <h2>Rules</h2>
          <p>What to scan on each feed. Rules drive Monitor, Alerts, and Reports. Click ⓘ for the full catalog.</p>
        </div>
        <div class="sa-row">
          <button class="btn secondary" type="button" id="r-goto-staff">Staff</button>
          <button class="btn secondary" type="button" id="r-goto-vehicles">Vehicles</button>
        </div>
      </div>
      ${
        isAdmin()
          ? `<div class="sa-card" style="margin-bottom:1rem;">
        <div class="sa-form">
          <div class="sa-field"><label>Rule name</label><input class="text-input" id="r-name" placeholder="Main gate"></div>
          <div class="sa-field"><label>Camera</label><select class="text-input" id="r-cam">${camOpts}</select></div>
          <p class="sa-muted" id="r-cam-limit-hint"></p>
          <div class="sa-field">
            <label class="sa-label-row">Scan type
              <button type="button" class="sa-info-btn" id="r-type-info" title="What each scan type means">i</button>
            </label>
            <select class="text-input" id="r-type">${typeOpts}</select>
            <p class="sa-muted sa-type-hint" id="r-type-hint"></p>
          </div>
          <div class="sa-field"><label>Channels</label>
            <label style="display:inline;margin-right:1rem;"><input type="checkbox" id="ch-web" checked> Web</label>
            <label style="display:inline;"><input type="checkbox" id="ch-wa"> WhatsApp</label>
          </div>
          <div id="gate-setup" class="sa-gate-setup" style="display:none;">
            <p class="sa-muted">Draw each shape on the preview. Click a step, then click the image.</p>
            <div class="sa-row sa-gate-steps">
              <button class="btn secondary sa-gate-step active" type="button" data-gmode="count_line">Count line (2)</button>
              <button class="btn secondary sa-gate-step" type="button" data-gmode="gate_roi">Gate ROI</button>
              <button class="btn secondary sa-gate-step" type="button" data-gmode="near">Near zone</button>
              <button class="btn secondary sa-gate-step" type="button" data-gmode="medium">Medium zone</button>
              <button class="btn secondary sa-gate-step" type="button" data-gmode="far">Far zone</button>
            </div>
            <div class="sa-field">
              <label>Direction counted as IN</label>
              <label style="display:inline;margin-right:1rem;"><input type="radio" name="g-dir" value="left" checked> Left of line</label>
              <label style="display:inline;"><input type="radio" name="g-dir" value="right"> Right of line</label>
            </div>
            <button class="btn secondary" type="button" id="g-clear-mode">Clear current step</button>
          </div>
          <p class="sa-muted" id="roi-hint">Click the preview to draw ROI, or use Suggest regions (FastSAM) to click a detected area. Fall / face / vehicle can run without a polygon.</p>
          <div class="sa-canvas-wrap" id="roi-wrap" style="display:none;">
            <img id="roi-img" alt="Preview">
            <canvas id="roi-cv"></canvas>
          </div>
          <div class="sa-row">
            <button class="btn secondary" type="button" id="r-suggest">Suggest regions</button>
            <button class="btn secondary" type="button" id="r-suggest-done" style="display:none;">Done picking</button>
            <button class="btn secondary" type="button" id="r-clear">Clear ROI</button>
            <button class="btn primary" type="button" id="r-save" style="width:auto;">Save rule</button>
          </div>
        </div>
      </div>`
          : ""
      }
      <table class="sa-table">
        <thead><tr><th>Preview</th><th>Name</th><th>Camera</th><th>Scan</th><th>Geometry</th><th>On</th><th></th></tr></thead>
        <tbody>${rows || '<tr><td colspan="7" class="sa-muted">No rules yet</td></tr>'}</tbody>
      </table>`;

    document.getElementById("r-goto-staff")?.addEventListener("click", () => go("staff"));
    document.getElementById("r-goto-vehicles")?.addEventListener("click", () => go("vehicles"));

    state.rules.forEach((r) => {
      const cam = state.cameras.find((c) => c.id === r.camera_id);
      const cv = main.querySelector(`[data-rule-cv="${r.id}"]`);
      if (cv && cam?.preview_url) renderRuleThumb(cv, cam.preview_url, r, 140);
    });
    main.querySelectorAll("[data-rule-preview]").forEach((b) => {
      b.onclick = () => {
        const rid = b.getAttribute("data-rule-preview");
        const rule = state.rules.find((x) => x.id === rid);
        if (!rule) return;
        const cam = state.cameras.find((c) => c.id === rule.camera_id);
        openRuleLightbox(rule, cam);
      };
    });

    const typeSel = document.getElementById("r-type");
    const gateSetup = document.getElementById("gate-setup");
    const roiHint = document.getElementById("roi-hint");
    const typeHint = document.getElementById("r-type-hint");

    function isGateType() {
      return typeSel?.value === "gate_analytics";
    }

    function syncTypeHint() {
      const entry = catalogById(typeSel?.value || "");
      if (!typeHint) return;
      if (!entry) {
        typeHint.textContent = "";
        return;
      }
      if (entry.status !== "available") {
        typeHint.textContent =
          "Coming soon — you can browse this type, but Save is blocked until the pipeline ships.";
        return;
      }
      if (entry.kind === "identity") {
        typeHint.textContent =
          entry.id === "vehicle"
            ? "After save, run for a few days, then approve plates in the Vehicles tab. Unknowns can alert later."
            : "After save, run for a few days, then approve faces in the Staff tab. Unknowns can alert later.";
        return;
      }
      typeHint.textContent = entry.typical_need || "";
    }

    function syncGateUi() {
      const gate = isGateType();
      if (gateSetup) gateSetup.style.display = gate ? "block" : "none";
      if (roiHint && !state.roiPickMode) {
        roiHint.textContent = gate
          ? "Gate rule: draw count line, gate ROI, and three distance zones — or Suggest regions and click a shape for ROI/zones."
          : "Click the preview to draw ROI, or Suggest regions (FastSAM) to click a detected area. Fall / face / vehicle can run without a polygon.";
      }
      document.querySelectorAll(".sa-gate-step").forEach((b) => {
        b.classList.toggle("active", b.getAttribute("data-gmode") === state.gateDrawMode);
      });
      syncRoiPickUi();
      syncTypeHint();
    }

    document.getElementById("r-type-info")?.addEventListener("click", () => openScanCatalogHelp());

    typeSel?.addEventListener("change", () => {
      syncGateUi();
      showPreview();
    });

    document.querySelectorAll(".sa-gate-step").forEach((b) => {
      b.addEventListener("click", () => {
        state.gateDrawMode = b.getAttribute("data-gmode");
        syncGateUi();
        const img = document.getElementById("roi-img");
        const cv = document.getElementById("roi-cv");
        if (img && cv) drawGateCanvas(cv, img);
      });
    });

    document.getElementById("g-clear-mode")?.addEventListener("click", () => {
      clearGateMode(state.gateDrawMode);
      const img = document.getElementById("roi-img");
      const cv = document.getElementById("roi-cv");
      if (img && cv) drawGateCanvas(cv, img);
    });

    const camSel = document.getElementById("r-cam");
    const MAX_RULES_PER_CAMERA = 3;

    function enabledRulesOnCamera(camId) {
      return state.rules.filter((r) => r.camera_id === camId && r.enabled !== false);
    }

    function syncCameraRuleLimitHint() {
      const hint = document.getElementById("r-cam-limit-hint");
      if (!hint || !camSel) return;
      const camId = camSel.value;
      const n = enabledRulesOnCamera(camId).length;
      if (!camId) {
        hint.textContent = "";
        return;
      }
      hint.textContent =
        n >= MAX_RULES_PER_CAMERA
          ? `This camera already has ${MAX_RULES_PER_CAMERA} enabled rules (max). Disable or remove one to add another.`
          : `${n} of ${MAX_RULES_PER_CAMERA} enabled rules on this camera. Monitor runs all rules in parallel.`;
    }

    function showPreview() {
      const cam = state.cameras.find((c) => c.id === camSel?.value);
      const wrap = document.getElementById("roi-wrap");
      const img = document.getElementById("roi-img");
      state.roiSuggestions = [];
      state.roiPickMode = false;
      if (!wrap || !img || !cam?.preview_url) {
        if (wrap) wrap.style.display = "none";
        syncRoiPickUi();
        return;
      }
      wrap.style.display = "inline-block";
      img.onload = () => bindCanvas(img, document.getElementById("roi-cv"), isGateType());
      img.src = cam.preview_url;
      syncRoiPickUi();
      syncCameraRuleLimitHint();
    }
    if (camSel) {
      camSel.onchange = showPreview;
      showPreview();
    }
    syncGateUi();

    document.getElementById("r-suggest")?.addEventListener("click", async () => {
      const camId = camSel?.value;
      if (!camId) return alert("Pick a camera first.");
      const img = document.getElementById("roi-img");
      const cv = document.getElementById("roi-cv");
      if (!img || !cv || !document.getElementById("roi-wrap")?.style.display || document.getElementById("roi-wrap").style.display === "none") {
        return alert("Camera needs a preview still. Re-test the connection or re-upload the file.");
      }
      state.roiSuggestBusy = true;
      syncRoiPickUi();
      try {
        const data = await api("/cameras/" + encodeURIComponent(camId) + "/suggest-roi", {
          method: "POST",
          json: {},
        });
        state.roiSuggestions = data.regions || [];
        if (!state.roiSuggestions.length) {
          state.roiPickMode = false;
          alert("No regions found. Try again or draw manually.");
        } else {
          state.roiPickMode = true;
        }
        bindCanvas(img, cv, isGateType());
        syncRoiPickUi();
      } catch (e) {
        alert(e.message || "Suggest failed");
      } finally {
        state.roiSuggestBusy = false;
        syncRoiPickUi();
      }
    });

    document.getElementById("r-suggest-done")?.addEventListener("click", () => {
      exitRoiPickMode();
      const img = document.getElementById("roi-img");
      const cv = document.getElementById("roi-cv");
      if (img && cv) bindCanvas(img, cv, isGateType());
      syncGateUi();
    });

    document.getElementById("r-clear")?.addEventListener("click", () => {
      if (isGateType()) {
        state.gateConfig = emptyGateConfig();
        state.gateDrawMode = "count_line";
        syncGateUi();
      } else {
        state.roi = [];
      }
      state.roiSuggestions = [];
      state.roiPickMode = false;
      const img = document.getElementById("roi-img");
      const cv = document.getElementById("roi-cv");
      if (cv && img) drawRoiOrGate(cv, img, isGateType());
      syncRoiPickUi();
    });
    document.getElementById("r-save")?.addEventListener("click", async () => {
      const scanType = document.getElementById("r-type").value;
      const entry = catalogById(scanType);
      if (entry && entry.status !== "available") {
        return alert("This scan type is Coming soon. Pick an Available type (see ⓘ).");
      }
      const channels = [];
      if (document.getElementById("ch-web").checked) channels.push("web");
      if (document.getElementById("ch-wa").checked) channels.push("whatsapp");
      const camId = document.getElementById("r-cam").value;
      if (enabledRulesOnCamera(camId).length >= MAX_RULES_PER_CAMERA) {
        return alert(`Max ${MAX_RULES_PER_CAMERA} enabled rules per camera. Disable or remove one first.`);
      }
      const payload = {
        name: document.getElementById("r-name").value || "Rule",
        camera_id: camId,
        scan_type: scanType,
        roi_normalized: state.roi,
        channels: channels.length ? channels : ["web"],
        enabled: true,
      };
      if (scanType === "gate_analytics") {
        const dirEl = document.querySelector('input[name="g-dir"]:checked');
        state.gateConfig.direction_in = dirEl ? dirEl.value : "left";
        payload.gate_config = JSON.parse(JSON.stringify(state.gateConfig));
        payload.roi_normalized = [];
        if ((payload.gate_config.count_line || []).length !== 2) {
          return alert("Draw the count line (2 clicks).");
        }
        if ((payload.gate_config.gate_roi || []).length < 3) {
          return alert("Draw the gate ROI (3+ clicks).");
        }
        for (const z of ["near", "medium", "far"]) {
          if ((payload.gate_config.distance_zones[z] || []).length < 3) {
            return alert("Draw near, medium, and far zones (3+ clicks each).");
          }
        }
      }
      try {
        const saved = await api("/rules", { json: payload });
        state.roi = [];
        state.gateConfig = emptyGateConfig();
        state.gateDrawMode = "count_line";
        if (scanType === "gate_analytics" && saved.id) {
          try {
            await api("/rules/" + saved.id + "/gate-calibrate", { method: "POST" });
          } catch (_) {
            /* calibrate optional if camera offline */
          }
        }
        if (entry?.kind === "identity") {
          alert(
            scanType === "vehicle"
              ? "Rule saved. After plates appear over a few days, approve yours in Vehicles."
              : "Rule saved. After faces appear over a few days, approve staff in Staff."
          );
        }
        await loadLists();
        rules();
      } catch (e) {
        alert(e.message);
      }
    });
    main.querySelectorAll("[data-delr]").forEach((b) => {
      b.onclick = async () => {
        await api("/rules/" + b.getAttribute("data-delr"), { method: "DELETE" });
        await loadLists();
        rules();
      };
    });
  }

  async function staff() {
    let faces = [];
    try {
      const data = await api("/known-faces");
      faces = data.faces || [];
    } catch (_) {
      faces = [];
    }
    const filter = state._staffFilter || "all";
    const shown =
      filter === "all" ? faces : faces.filter((f) => (f.status || "candidate") === filter);
    const cards = shown
      .map((f) => {
        const st = f.status || "candidate";
        const thumb = f.thumb_url
          ? `<img src="${escapeHtml(f.thumb_url)}" alt="">`
          : `<span class="sa-id-empty">No thumb</span>`;
        return `<article class="sa-id-card" data-id="${escapeHtml(f.id)}">
          <div class="sa-id-thumb">${thumb}</div>
          <div class="sa-id-body">
            <strong>${escapeHtml(f.label || "Unknown")}</strong>
            <span class="sa-badge">${escapeHtml(st)}</span>
            <p class="sa-muted">${escapeHtml(f.note || f.person_id || "")}</p>
            <div class="sa-row">
              ${st !== "approved" ? `<button class="btn primary" type="button" data-face-approve="${escapeHtml(f.id)}">Approve</button>` : ""}
              ${st !== "ignored" ? `<button class="btn secondary" type="button" data-face-ignore="${escapeHtml(f.id)}">Ignore</button>` : ""}
            </div>
          </div>
        </article>`;
      })
      .join("");
    main.innerHTML = `
      <div class="sa-h">
        <div>
          <h2>Staff</h2>
          <p>Approve faces collected while a Staff / face rule runs. Known staff stay quiet; unknowns can alert later.</p>
        </div>
        <div class="sa-row">
          <button class="btn secondary${filter === "all" ? " active" : ""}" type="button" data-staff-f="all">All</button>
          <button class="btn secondary${filter === "candidate" ? " active" : ""}" type="button" data-staff-f="candidate">Candidates</button>
          <button class="btn secondary${filter === "approved" ? " active" : ""}" type="button" data-staff-f="approved">Approved</button>
        </div>
      </div>
      <div class="sa-id-grid">${cards || '<p class="sa-muted">No faces yet. Save a Face / staff attendance rule, run for a few days, then candidates appear here.</p>'}</div>`;
    main.querySelectorAll("[data-staff-f]").forEach((b) => {
      b.onclick = () => {
        state._staffFilter = b.getAttribute("data-staff-f");
        staff();
      };
    });
    main.querySelectorAll("[data-face-approve]").forEach((b) => {
      b.onclick = async () => {
        try {
          await api("/known-faces/" + b.getAttribute("data-face-approve"), {
            method: "PATCH",
            json: { status: "approved" },
          });
          staff();
        } catch (e) {
          alert(e.message);
        }
      };
    });
    main.querySelectorAll("[data-face-ignore]").forEach((b) => {
      b.onclick = async () => {
        try {
          await api("/known-faces/" + b.getAttribute("data-face-ignore"), {
            method: "PATCH",
            json: { status: "ignored" },
          });
          staff();
        } catch (e) {
          alert(e.message);
        }
      };
    });
  }

  async function vehicles() {
    let vehiclesList = [];
    try {
      const data = await api("/known-vehicles");
      vehiclesList = data.vehicles || [];
    } catch (_) {
      vehiclesList = [];
    }
    const filter = state._vehicleFilter || "all";
    const shown =
      filter === "all"
        ? vehiclesList
        : filter === "risk"
          ? vehiclesList.filter((v) => ["risk", "danger"].includes(v.status || ""))
          : vehiclesList.filter((v) => (v.status || "candidate") === filter);
    const cards = shown
      .map((v) => {
        const st = v.status || "candidate";
        const thumb = v.thumb_url
          ? `<img src="${escapeHtml(v.thumb_url)}" alt="">`
          : `<span class="sa-id-empty">No thumb</span>`;
        const seen = v.last_seen_at
          ? new Date(v.last_seen_at * 1000).toLocaleString()
          : "";
        const count = v.seen_count != null ? ` · seen ${v.seen_count}×` : "";
        const badgeCls =
          st === "approved" ? "ok" : st === "risk" || st === "danger" ? "bad" : "";
        return `<article class="sa-id-card" data-id="${escapeHtml(v.id)}">
          <div class="sa-id-thumb">${thumb}</div>
          <div class="sa-id-body">
            <strong>${escapeHtml(v.plate || v.label || "UNKNOWN")}</strong>
            <span class="sa-badge ${badgeCls}">${escapeHtml(st === "candidate" ? "unknown" : st)}</span>
            <p class="sa-muted">${escapeHtml(seen + count)}</p>
            <div class="sa-row">
              ${st !== "approved" ? `<button class="btn primary" type="button" data-veh-approve="${escapeHtml(v.id)}">Approve (ours)</button>` : ""}
              ${st !== "ignored" ? `<button class="btn secondary" type="button" data-veh-ignore="${escapeHtml(v.id)}">Ignore</button>` : ""}
              ${st !== "risk" && st !== "danger" ? `<button class="btn danger" type="button" data-veh-risk="${escapeHtml(v.id)}">Mark risk</button>` : ""}
              ${st === "risk" || st === "danger" ? `<button class="btn secondary" type="button" data-veh-approve="${escapeHtml(v.id)}">Clear risk → ours</button>` : ""}
              <button class="btn danger" type="button" data-veh-del="${escapeHtml(v.id)}" data-veh-plate="${escapeHtml(v.plate || v.label || "")}">Remove</button>
            </div>
          </div>
        </article>`;
      })
      .join("");
    main.innerHTML = `
      <div class="sa-h">
        <div>
          <h2>Vehicles</h2>
          <p>Each plate is stored once with a thumb. Approve society cars, leave unknowns for review, or Mark risk for immediate alerts on return.</p>
        </div>
        <div class="sa-row">
          <button class="btn secondary${filter === "all" ? " active" : ""}" type="button" data-veh-f="all">All</button>
          <button class="btn secondary${filter === "candidate" ? " active" : ""}" type="button" data-veh-f="candidate">Unknown</button>
          <button class="btn secondary${filter === "approved" ? " active" : ""}" type="button" data-veh-f="approved">Ours</button>
          <button class="btn secondary${filter === "risk" ? " active" : ""}" type="button" data-veh-f="risk">Risk</button>
          <button class="btn secondary${filter === "ignored" ? " active" : ""}" type="button" data-veh-f="ignored">Ignored</button>
        </div>
      </div>
      <div class="sa-id-grid">${cards || '<p class="sa-muted">No plates yet. Save a Vehicle rule and run Monitor or Go live — first OCR of each plate appears here once (no spam).</p>'}</div>`;
    main.querySelectorAll("[data-veh-f]").forEach((b) => {
      b.onclick = () => {
        state._vehicleFilter = b.getAttribute("data-veh-f");
        vehicles();
      };
    });
    main.querySelectorAll("[data-veh-approve]").forEach((b) => {
      b.onclick = async () => {
        try {
          await api("/known-vehicles/" + b.getAttribute("data-veh-approve"), {
            method: "PATCH",
            json: { status: "approved" },
          });
          vehicles();
        } catch (e) {
          alert(e.message);
        }
      };
    });
    main.querySelectorAll("[data-veh-ignore]").forEach((b) => {
      b.onclick = async () => {
        try {
          await api("/known-vehicles/" + b.getAttribute("data-veh-ignore"), {
            method: "PATCH",
            json: { status: "ignored" },
          });
          vehicles();
        } catch (e) {
          alert(e.message);
        }
      };
    });
    main.querySelectorAll("[data-veh-risk]").forEach((b) => {
      b.onclick = async () => {
        try {
          await api("/known-vehicles/" + b.getAttribute("data-veh-risk"), {
            method: "PATCH",
            json: { status: "risk" },
          });
          vehicles();
        } catch (e) {
          alert(e.message);
        }
      };
    });
    main.querySelectorAll("[data-veh-del]").forEach((b) => {
      b.onclick = async () => {
        const plate = b.getAttribute("data-veh-plate") || "this vehicle";
        if (!confirm("Remove plate " + plate + " from Vehicles?")) return;
        try {
          await api("/known-vehicles/" + b.getAttribute("data-veh-del"), { method: "DELETE" });
          vehicles();
        } catch (e) {
          alert(e.message);
        }
      };
    });
  }

  function bindRoi(img, canvas) {
    if (!img || !canvas) return;
    canvas.width = img.clientWidth;
    canvas.height = img.clientHeight;
    canvas.onclick = (ev) => {
      const rect = canvas.getBoundingClientRect();
      state.roi.push([(ev.clientX - rect.left) / rect.width, (ev.clientY - rect.top) / rect.height]);
      drawRoi(canvas, img);
    };
    drawRoi(canvas, img);
  }

  function drawRoi(canvas, img) {
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawSuggestions(ctx, canvas);
    if (state.roi.length === 0) return;
    ctx.strokeStyle = "#22d3ee";
    ctx.fillStyle = "rgba(34, 211, 238, 0.2)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    state.roi.forEach((p, i) => {
      const x = p[0] * canvas.width;
      const y = p[1] * canvas.height;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    if (state.roi.length >= 3) ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }

  let monitorAudioCtx = null;

  function playMonitorBipTone(ctx) {
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "sine";
    osc.frequency.value = 880;
    const now = ctx.currentTime;
    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.exponentialRampToValueAtTime(0.22, now + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.15);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start(now);
    osc.stop(now + 0.16);
  }

  function playMonitorBeepsForNewEvents(events) {
    if (!state.monitorSeenEventIds) state.monitorSeenEventIds = new Set();
    const seen = state.monitorSeenEventIds;
    const fresh = (events || []).filter((ev) => {
      const eid = ev.event_id || ev.id;
      return eid && !seen.has(eid);
    });
    fresh.forEach((ev, idx) => {
      const eid = ev.event_id || ev.id;
      seen.add(eid);
      setTimeout(() => playMonitorBip(), idx * 120);
    });
  }

  function seedMonitorSeenEvents(events) {
    state.monitorSeenEventIds = new Set();
    (events || []).forEach((ev) => {
      const eid = ev.event_id || ev.id;
      if (eid) state.monitorSeenEventIds.add(eid);
    });
  }

  function resetMonitorEventTracking() {
    state.monitorEventCount = 0;
    state.monitorSeenEventIds = new Set();
  }

  function playMonitorBip() {
    try {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return;
      if (!monitorAudioCtx) monitorAudioCtx = new AC();
      const ctx = monitorAudioCtx;
      if (ctx.state === "suspended") {
        ctx.resume().then(() => playMonitorBipTone(ctx)).catch(() => {});
        return;
      }
      playMonitorBipTone(ctx);
    } catch (_) {
      /* ignore autoplay / AudioContext errors */
    }
  }

  function eventAsRule(ev) {
    return {
      scan_type: ev.scan_type,
      roi_normalized: ev.roi_normalized || [],
      gate_config: ev.gate_config || {},
      css_color: ev.css_color,
    };
  }

  function openEventLightbox(ev) {
    const title = `${ev.rule_name || "Event"} · ${(ev.scan_type || "").replace(/_/g, " ")}`;
    if (ev.clip_url && !ev.thumb_url) {
      openLightbox({ title, videoUrl: ev.clip_url });
      return;
    }
    const thumb = ev.thumb_url;
    if (!thumb) {
      openLightbox({ title, body: '<p class="sa-muted">No snapshot for this event.</p>' });
      return;
    }
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.className = "sa-lightbox-media";
      const maxW = Math.min(720, img.naturalWidth);
      const scale = maxW / img.naturalWidth;
      canvas.width = maxW;
      canvas.height = Math.round(img.naturalHeight * scale);
      drawRuleOntoCanvas(canvas, img, eventAsRule(ev));
      openLightbox({ title, canvas });
    };
    img.onerror = () => openLightbox({ title, imageUrl: thumb });
    img.src = thumb;
  }

  function eventTrackerHtml(events) {
    const list = events || [];
    if (!list.length) {
      return `<div class="sa-event-tracker sa-card">
        <h3>Event tracker</h3>
        <p class="sa-muted">Events from this monitor session appear here as thumbnails.</p>
      </div>`;
    }
    const chips = list
      .slice()
      .reverse()
      .map((ev) => {
        const t = ev.ts ? new Date(ev.ts * 1000).toLocaleTimeString() : "";
        const eid = ev.event_id || ev.id || "";
        const swatch = ev.css_color
          ? `<span class="sa-event-chip-swatch" style="background:${escapeHtml(ev.css_color)}"></span>`
          : "";
        const borderStyle = ev.css_color ? ` style="border-color:${escapeHtml(ev.css_color)}"` : "";
        const media = ev.thumb_url
          ? `<img src="${escapeHtml(ev.thumb_url)}" alt="">`
          : ev.clip_url
            ? `<video src="${escapeHtml(ev.clip_url)}" muted playsinline preload="metadata"></video>`
            : `<span class="sa-event-chip-empty">No snap</span>`;
        return `<button type="button" class="sa-event-chip" data-event-id="${escapeHtml(eid)}"${borderStyle} data-ev-thumb="${escapeHtml(ev.thumb_url || "")}" data-ev-clip="${escapeHtml(ev.clip_url || "")}" data-ev-title="${escapeHtml(ev.rule_name || ev.scan_type || "Event")}">
          <div class="sa-event-chip-media">${media}</div>
          <div class="sa-event-chip-meta">
            ${swatch}<strong>${escapeHtml((ev.scan_type || "").replace(/_/g, " "))}</strong>
            <span>${escapeHtml(ev.rule_name || "")}</span>
            <span class="sa-muted">${escapeHtml(t)}</span>
          </div>
        </button>`;
      })
      .join("");
    return `<div class="sa-event-tracker sa-card">
      <h3>Event tracker <span class="sa-muted">(${list.length})</span></h3>
      <div class="sa-event-tracker-row" id="mon-event-row">${chips}</div>
    </div>`;
  }

  function bindEventTrackerClicks(root) {
    (root || document).querySelectorAll(".sa-event-chip").forEach((b) => {
      b.onclick = () => {
        const eid = b.getAttribute("data-event-id");
        const ev = (state.monitorEvents || []).find((e) => (e.event_id || e.id) === eid);
        if (ev) {
          openEventLightbox(ev);
          return;
        }
        const title = b.getAttribute("data-ev-title") || "Event";
        const clip = b.getAttribute("data-ev-clip") || "";
        const thumb = b.getAttribute("data-ev-thumb") || "";
        if (clip) openLightbox({ title, videoUrl: clip });
        else if (thumb) openLightbox({ title, imageUrl: thumb });
      };
    });
  }

  function renderEventTrackerStrip(events) {
    const wrap = document.querySelector(".sa-event-tracker");
    if (!wrap) return;
    const html = eventTrackerHtml(events);
    const tmp = document.createElement("div");
    tmp.innerHTML = html;
    const next = tmp.firstElementChild;
    if (next) {
      wrap.replaceWith(next);
      bindEventTrackerClicks(next);
    }
  }

  function clearMonitorFramePoll() {
    if (state.monitorFramePoll) {
      clearInterval(state.monitorFramePoll);
      state.monitorFramePoll = null;
    }
  }

  function hideMonitorOverlays() {
    const loading = document.getElementById("sa-monitor-loading");
    const errEl = document.getElementById("sa-monitor-error");
    if (loading) loading.hidden = true;
    if (errEl) errEl.hidden = true;
  }

  function monitorCameraType(monStatus) {
    if (monStatus?.camera_type) return monStatus.camera_type;
    const cam = state.cameras.find((c) => c.id === monStatus?.camera_id);
    return cam?.type || "rtsp";
  }

  function bindMonitorStreamImg(useDvrPoll, sessionId) {
    const img = document.getElementById("sa-monitor-video");
    const loading = document.getElementById("sa-monitor-loading");
    const errEl = document.getElementById("sa-monitor-error");
    if (!img) return;

    clearMonitorFramePoll();
    if (errEl) errEl.hidden = true;

    img.onload = () => hideMonitorOverlays();
    img.onerror = () => {
      /* Keep overlay hidden when polling — a failed refresh must not block the last good frame */
      if (useDvrPoll) return;
    };

    if (img.complete && img.naturalWidth > 0) {
      hideMonitorOverlays();
    }

    if (useDvrPoll && sessionId) {
      const refresh = () => {
        if (state.view !== "monitor" || !state.monitorSessionId) return;
        img.src = `/api/site-admin/monitor/frame/${encodeURIComponent(sessionId)}?t=${Date.now()}`;
      };
      refresh();
      state.monitorFramePoll = setInterval(refresh, 1000);
      window.setTimeout(hideMonitorOverlays, 4000);
      return;
    }

    window.setTimeout(hideMonitorOverlays, 2500);
  }

  async function monitor() {
    if (state.monitorPoll) {
      clearInterval(state.monitorPoll);
      state.monitorPoll = null;
    }
    if (!state.monitorSessionId) {
      clearMonitorFramePoll();
    }
    await loadLists();
    const monStatus = await api("/monitor/status");
    if (monStatus.active && monStatus.session_id) {
      state.monitorSessionId = monStatus.session_id;
      if (monStatus.camera_id) {
        state.monitorCameraPick = monStatus.camera_id;
      }
    } else if (state.monitorSessionId && !monStatus.active) {
      state.monitorSessionId = null;
    }

    const camerasWithRules = state.cameras.filter((c) =>
      state.rules.some((r) => r.camera_id === c.id && r.enabled !== false)
    );
    const activeCamId = monStatus.active ? monStatus.camera_id || state.monitorCameraPick : state.monitorCameraPick;
    const camOpts =
      `<option value=""${!activeCamId ? " selected" : ""}>Select camera…</option>` +
      camerasWithRules
        .map((c) => {
          const n = state.rules.filter((r) => r.camera_id === c.id && r.enabled !== false).length;
          const sel = c.id === activeCamId ? " selected" : "";
          return `<option value="${c.id}"${sel}>${escapeHtml(c.name)} (${n} rules)</option>`;
        })
        .join("");

    const activeRules = monStatus.active ? monStatus.rules || [] : [];
    const legend = activeRules
      .map(
        (r) => `<div class="sa-legend-item">
          <span class="sa-legend-swatch" style="background:${escapeHtml(r.css_color || "#666")}"></span>
          <span>${escapeHtml(r.name)} — ${escapeHtml(r.scan_type)}</span>
        </div>`
      )
      .join("");

    const heavyNote =
      (monStatus.scan_types || []).length > 2
        ? `<p class="sa-muted">Multiple scan types may reduce frame rate on a mini-PC.</p>`
        : "";

    const hasGate = (monStatus.scan_types || []).includes("gate_analytics");
    const gl = monStatus.gate_live || {};
    const gt = gl.session_totals || {};
    const gatePanel =
      hasGate && state.monitorSessionId
        ? `<div class="sa-gate-live sa-card">
            <span class="sa-gate-badge sa-gate-badge--${escapeHtml(gl.gate_state || "unknown")}">Gate: ${escapeHtml((gl.gate_state || "unknown").toUpperCase())}</span>
            <div class="sa-gate-stats">
              <span>Near/Med/Far: ${gl.near_count || 0} / ${gl.medium_count || 0} / ${gl.far_count || 0}</span>
              <span>People in/out: ${gt.persons_in || 0} / ${gt.persons_out || 0}</span>
              <span>Cars in/out: ${gt.cars_in || 0} / ${gt.cars_out || 0}</span>
              <span>Bikes in/out: ${gt.bikes_in || 0} / ${gt.bikes_out || 0}</span>
              <span>Opens/closes: ${gt.gate_opens || 0} / ${gt.gate_closes || 0}</span>
            </div>
          </div>`
        : "";

    const streamLive = !!(monStatus.active && state.monitorSessionId);
    const useDvrPoll = streamLive && monitorCameraType(monStatus) === "dvr";
    const videoSrc = streamLive
      ? useDvrPoll
        ? `/api/site-admin/monitor/frame/${escapeHtml(state.monitorSessionId)}?t=${Date.now()}`
        : `/api/site-admin/monitor/stream/${escapeHtml(state.monitorSessionId)}?t=${Date.now()}`
      : "";
    const streaming = streamLive
      ? `<div class="sa-monitor-wrap">
          <div id="sa-monitor-loading" class="sa-monitor-loading">Connecting…</div>
          <div id="sa-monitor-error" class="sa-monitor-error" hidden>
            Stream unavailable — check camera or Docker network to DVR
          </div>
          <img id="sa-monitor-video" class="sa-monitor-video" alt="Live monitor"
            src="${videoSrc}">
        </div>
        ${monitorControlsHtml(monStatus.active ? monStatus : null)}`
      : `<div class="sa-monitor-wrap"><p class="placeholder-msg" style="padding:2rem;">Select a camera and click Watch live</p></div>`;

    main.innerHTML = `
      <div class="sa-h">
        <div>
          <h2>Live monitor</h2>
          <p>Select a camera that already has rules, then Watch live. Add cameras under Cameras and draw rules under Rules.</p>
        </div>
      </div>
      ${heavyNote}
      <div class="sa-row" style="margin-bottom:1rem;">
        <select class="text-input" id="mon-cam" style="max-width:320px;" ${streamLive ? "disabled" : ""}>
          ${camOpts || '<option value="">No cameras with rules</option>'}
        </select>
        ${
          isAdmin()
            ? streamLive
              ? `<button class="btn danger" type="button" id="mon-stop">Stop</button>`
              : `<button class="btn primary" type="button" id="mon-start" style="width:auto;">Watch live</button>`
            : `<span class="sa-muted">Viewer — watch only when Admin starts a session</span>`
        }
        <button class="btn" type="button" id="mon-test-beep">Test beep</button>
      </div>
      ${streaming}
      ${gatePanel}
      ${eventTrackerHtml(monStatus.events || [])}
      <div class="sa-card">
        <h3>Rules on this stream</h3>
        <div class="sa-legend">${legend || '<span class="sa-muted">Start monitoring to see rule legend</span>'}</div>
      </div>`;

    document.getElementById("mon-cam")?.addEventListener("change", (e) => {
      state.monitorCameraPick = e.target.value || "";
    });

    document.getElementById("mon-test-beep")?.addEventListener("click", () => {
      playMonitorBip();
    });

    document.getElementById("mon-start")?.addEventListener("click", async () => {
      const camSel = document.getElementById("mon-cam");
      const camId = camSel?.value;
      if (!camId) return alert("Select a camera with rules");
      const startBtn = document.getElementById("mon-start");
      if (startBtn) {
        startBtn.disabled = true;
        startBtn.textContent = "Starting…";
      }
      try {
        const r = await api("/monitor/start", { method: "POST", json: { camera_id: camId } });
        state.monitorSessionId = r.session_id;
        state.monitorCameraPick = camId;
        state.monitorEventCount = 0;
        resetMonitorEventTracking();
        monitor();
      } catch (e) {
        alert(e.message);
        if (startBtn) {
          startBtn.disabled = false;
          startBtn.textContent = "Watch live";
        }
      }
    });

    document.getElementById("mon-stop")?.addEventListener("click", async () => {
      await stopMonitorIfAny();
      state.monitorCameraPick = "";
      monitor();
    });

    bindMonitorStreamImg(useDvrPoll, state.monitorSessionId);
    if (streamLive) hideMonitorOverlays();

    bindEventTrackerClicks(main);

    state.monitorEvents = monStatus.events || [];

    if (state.monitorSessionId && monStatus.active) {
      seedMonitorSeenEvents(monStatus.events || []);
      state.monitorEventCount = (monStatus.events || []).length;
      state.monitorPaused = !!monStatus.paused;
      syncMonitorPlayback(monStatus);
      bindMonitorControls(monStatus);
    }

    if (state.monitorSessionId && monStatus.active) {
      state.monitorPoll = setInterval(async () => {
        if (state.view !== "monitor") {
          clearInterval(state.monitorPoll);
          return;
        }
        try {
          const st = await api("/monitor/status");
          if (!st.active && state.monitorSessionId) {
            state.monitorSessionId = null;
            resetMonitorEventTracking();
            clearInterval(state.monitorPoll);
            state.monitorPoll = null;
            monitor();
            return;
          }
          if (st.active) {
            hideMonitorOverlays();
          }
          syncMonitorPlayback(st);
          const events = st.events || [];
          state.monitorEvents = events;
          playMonitorBeepsForNewEvents(events);
          state.monitorEventCount = events.length;
          renderEventTrackerStrip(events);
          const gateEl = document.querySelector(".sa-gate-live");
          if (st.gate_live && gateEl) {
            const gl2 = st.gate_live;
            const gt2 = gl2.session_totals || {};
            const badge = gateEl.querySelector(".sa-gate-badge");
            if (badge) {
              badge.textContent = "Gate: " + (gl2.gate_state || "unknown").toUpperCase();
              badge.className = "sa-gate-badge sa-gate-badge--" + (gl2.gate_state || "unknown");
            }
            const stats = gateEl.querySelector(".sa-gate-stats");
            if (stats) {
              stats.innerHTML = `
                <span>Near/Med/Far: ${gl2.near_count || 0} / ${gl2.medium_count || 0} / ${gl2.far_count || 0}</span>
                <span>People in/out: ${gt2.persons_in || 0} / ${gt2.persons_out || 0}</span>
                <span>Cars in/out: ${gt2.cars_in || 0} / ${gt2.cars_out || 0}</span>
                <span>Bikes in/out: ${gt2.bikes_in || 0} / ${gt2.bikes_out || 0}</span>
                <span>Opens/closes: ${gt2.gate_opens || 0} / ${gt2.gate_closes || 0}</span>`;
            }
          }
        } catch (_) {
          /* ignore */
        }
      }, 2000);
    }
  }

  async function alerts() {
    const data = await api("/alerts?limit=80");
    state.alerts = data.alerts || [];
    const items = state.alerts
      .map((a) => {
        const t = a.created_at ? new Date(a.created_at * 1000).toLocaleString() : "";
        return `<article class="sa-alert${a.acked ? " acked" : ""}">
          <label class="sa-alert-check">
            <input type="checkbox" class="sa-alert-cb" value="${escapeHtml(a.id)}">
          </label>
          ${alertMediaHtml(a)}
          <div class="sa-alert-body">
            <strong>${escapeHtml(a.camera_name || a.camera_id)}</strong>
            <span class="sa-badge">${escapeHtml(a.scan_type)}</span>
            <p class="sa-muted">${escapeHtml(a.message)} · ${escapeHtml(t)}</p>
            <div class="sa-row">
              ${!a.acked ? `<button class="btn secondary" data-ack="${a.id}" type="button">Ack</button>` : ""}
              ${isAdmin() ? `<button class="btn secondary" data-mute="${a.id}" type="button">Mute 1h</button>` : ""}
              <button class="btn secondary" data-wa="${a.id}" type="button">WhatsApp</button>
              <button class="btn danger" data-del-alert="${a.id}" type="button">Remove</button>
            </div>
          </div>
        </article>`;
      })
      .join("");
    main.innerHTML = `
      <div class="sa-h">
        <div>
          <h2>Alerts</h2>
          <p>Inbox for rule matches. Select several and Remove selected, or remove one at a time.</p>
        </div>
        <div class="sa-alert-actions sa-row">
          <button class="btn secondary" type="button" id="a-refresh">Refresh</button>
          ${
            state.alerts.length
              ? `<button class="btn secondary" type="button" id="a-select-all">Select all</button>
                 <button class="btn secondary" type="button" id="a-clear-sel">Clear</button>
                 <button class="btn danger" type="button" id="a-remove-sel" disabled>Remove selected</button>`
              : ""
          }
        </div>
      </div>
      ${items || '<p class="sa-muted">No alerts yet. Start scanning from the Wizard after cameras and rules are saved.</p>'}`;

    function selectedIds() {
      return Array.from(main.querySelectorAll(".sa-alert-cb:checked")).map((cb) => cb.value);
    }
    function syncBulkBtn() {
      const btn = document.getElementById("a-remove-sel");
      if (btn) btn.disabled = selectedIds().length === 0;
    }

    document.getElementById("a-refresh").onclick = () => alerts();
    document.getElementById("a-select-all")?.addEventListener("click", () => {
      main.querySelectorAll(".sa-alert-cb").forEach((cb) => {
        cb.checked = true;
      });
      syncBulkBtn();
    });
    document.getElementById("a-clear-sel")?.addEventListener("click", () => {
      main.querySelectorAll(".sa-alert-cb").forEach((cb) => {
        cb.checked = false;
      });
      syncBulkBtn();
    });
    document.getElementById("a-remove-sel")?.addEventListener("click", async () => {
      const ids = selectedIds();
      if (!ids.length) return;
      if (!confirm("Remove " + ids.length + " alert" + (ids.length === 1 ? "" : "s") + "?")) return;
      try {
        await api("/alerts/delete", { method: "POST", json: { ids } });
        alerts();
      } catch (e) {
        alert(e.message);
      }
    });
    main.querySelectorAll(".sa-alert-cb").forEach((cb) => {
      cb.addEventListener("change", syncBulkBtn);
    });

    main.querySelectorAll("[data-lb-thumb], [data-lb-clip]").forEach((b) => {
      if (!b.classList.contains("sa-thumb-btn")) return;
      b.onclick = () => {
        const title = b.getAttribute("data-lb-title") || "Alert";
        const clip = b.getAttribute("data-lb-clip") || "";
        const thumb = b.getAttribute("data-lb-thumb") || "";
        if (clip) openLightbox({ title, videoUrl: clip });
        else if (thumb) openLightbox({ title, imageUrl: thumb });
      };
    });
    main.querySelectorAll("[data-ack]").forEach((b) => {
      b.onclick = async () => {
        await api("/alerts/" + b.getAttribute("data-ack"), { method: "PATCH", json: { acked: true } });
        alerts();
      };
    });
    main.querySelectorAll("[data-mute]").forEach((b) => {
      b.onclick = async () => {
        await api("/alerts/" + b.getAttribute("data-mute") + "/mute-rule", { method: "POST", json: { seconds: 3600 } });
        alerts();
      };
    });
    main.querySelectorAll("[data-wa]").forEach((b) => {
      b.onclick = async () => {
        const r = await api("/alerts/" + b.getAttribute("data-wa") + "/share-whatsapp", { method: "POST", json: {} });
        alert(r.configured ? JSON.stringify(r.results) : "WhatsApp API not configured. Set WHATSAPP_TOKEN and WHATSAPP_PHONE_NUMBER_ID.");
      };
    });
    main.querySelectorAll("[data-del-alert]").forEach((b) => {
      b.onclick = async () => {
        if (!confirm("Remove this alert?")) return;
        try {
          await api("/alerts/" + b.getAttribute("data-del-alert"), { method: "DELETE" });
          alerts();
        } catch (e) {
          alert(e.message);
        }
      };
    });
  }

  async function reports() {
    await loadLists();
    const hours = document.getElementById("rep-hours")?.value || "24";
    const ruleFilter = document.getElementById("rep-gate-rule")?.value || "";
    const data = await api("/reports?hours=" + hours);
    let gateData = { totals: {}, rules: [] };
    try {
      const q = "/reports/gate?hours=" + hours + (ruleFilter ? "&rule_id=" + encodeURIComponent(ruleFilter) : "");
      gateData = await api(q);
    } catch (_) {
      /* ignore */
    }
    const types = Object.entries(data.by_type || {})
      .map(([k, v]) => `<li>${escapeHtml(k)}: ${v}</li>`)
      .join("");
    const cams = Object.entries(data.by_camera || {})
      .map(([k, v]) => `<li>${escapeHtml(k)}: ${v}</li>`)
      .join("");
    const gt = gateData.totals || {};
    const gateRules = state.rules.filter((r) => r.scan_type === "gate_analytics");
    const gateRuleOpts = gateRules
      .map((r) => `<option value="${r.id}"${ruleFilter === r.id ? " selected" : ""}>${escapeHtml(r.name)}</option>`)
      .join("");
    main.innerHTML = `
      <div class="sa-h"><div><h2>Reports</h2><p>Alert summaries and gate activity counters. Export CSV when you need a file.</p></div></div>
      <div class="sa-row" style="margin-bottom:1rem;">
        <label class="sa-muted">Hours <input class="text-input" id="rep-hours" type="number" min="1" value="${escapeHtml(hours)}" style="width:6rem;display:inline-block;"></label>
        <button class="btn secondary" type="button" id="rep-go">Generate</button>
        <a class="btn secondary" href="${API}/reports/csv?hours=${escapeHtml(hours)}" style="display:inline-block;text-decoration:none;">Alerts CSV</a>
      </div>
      <div class="sa-grid">
        <div class="sa-card"><h3>Alerts</h3><div class="sa-stat">${data.alert_count || 0}</div></div>
        <div class="sa-card"><h3>Attendance events</h3><div class="sa-stat">${data.attendance_events || 0}</div></div>
        <div class="sa-card"><h3>People seen</h3><p>${(data.attendance_people || []).map(escapeHtml).join(", ") || "—"}</p></div>
      </div>
      <div class="sa-card" style="margin-top:1rem;">
        <h3>Gate activity</h3>
        <div class="sa-row" style="margin-bottom:0.75rem;">
          <label class="sa-muted">Gate rule
            <select class="text-input" id="rep-gate-rule" style="max-width:220px;display:inline-block;">
              <option value="">All gate rules</option>
              ${gateRuleOpts}
            </select>
          </label>
          <a class="btn secondary" href="${API}/reports/gate/csv?hours=${escapeHtml(hours)}${ruleFilter ? "&rule_id=" + encodeURIComponent(ruleFilter) : ""}" style="display:inline-block;text-decoration:none;">Gate CSV</a>
        </div>
        <div class="sa-gate-report-grid">
          <div class="sa-card sa-gate-stat"><h4>Gate opens</h4><div class="sa-stat">${gt.gate_opens || 0}</div></div>
          <div class="sa-card sa-gate-stat"><h4>Gate closes</h4><div class="sa-stat">${gt.gate_closes || 0}</div></div>
          <div class="sa-card sa-gate-stat"><h4>People in</h4><div class="sa-stat">${gt.persons_in || 0}</div></div>
          <div class="sa-card sa-gate-stat"><h4>People out</h4><div class="sa-stat">${gt.persons_out || 0}</div></div>
          <div class="sa-card sa-gate-stat"><h4>Cars in</h4><div class="sa-stat">${gt.cars_in || 0}</div></div>
          <div class="sa-card sa-gate-stat"><h4>Cars out</h4><div class="sa-stat">${gt.cars_out || 0}</div></div>
          <div class="sa-card sa-gate-stat"><h4>Bikes in</h4><div class="sa-stat">${gt.bikes_in || 0}</div></div>
          <div class="sa-card sa-gate-stat"><h4>Bikes out</h4><div class="sa-stat">${gt.bikes_out || 0}</div></div>
          <div class="sa-card sa-gate-stat"><h4>Near zone</h4><div class="sa-stat">${gt.near_events || 0}</div></div>
          <div class="sa-card sa-gate-stat"><h4>Medium zone</h4><div class="sa-stat">${gt.medium_events || 0}</div></div>
          <div class="sa-card sa-gate-stat"><h4>Far zone</h4><div class="sa-stat">${gt.far_events || 0}</div></div>
        </div>
      </div>
      <div class="sa-card"><h3>By type</h3><ul class="sa-muted">${types || "<li>None</li>"}</ul></div>
      <div class="sa-card" style="margin-top:0.8rem;"><h3>By camera</h3><ul class="sa-muted">${cams || "<li>None</li>"}</ul></div>`;
    document.getElementById("rep-go").onclick = () => reports();
    document.getElementById("rep-gate-rule")?.addEventListener("change", () => reports());
  }

  function settings() {
    const s = state.status?.site || {};
    const nums = (s.whatsapp_numbers || []).join(", ");
    main.innerHTML = `
      <div class="sa-h"><div><h2>Settings</h2><p>Roles, quiet hours, and recipients. Viewer can only open Alerts and Reports.</p></div></div>
      <div class="sa-form">
        <div class="sa-field"><label>This browser role</label>
          <select class="text-input" id="s-role">
            <option value="admin"${state.role === "admin" ? " selected" : ""}>Admin</option>
            <option value="viewer"${state.role === "viewer" ? " selected" : ""}>Viewer</option>
          </select>
        </div>
        <div class="sa-field"><label>Admin PIN (optional)</label><input class="text-input" id="s-pin" type="password" placeholder="Leave blank for open admin" value="${escapeHtml(s.admin_pin)}"></div>
        <div class="sa-field"><label>Quiet hours start</label><input class="text-input" id="s-qh1" placeholder="22:00" value="${escapeHtml(s.quiet_hours_start)}"></div>
        <div class="sa-field"><label>Quiet hours end</label><input class="text-input" id="s-qh2" placeholder="07:00" value="${escapeHtml(s.quiet_hours_end)}"></div>
        <div class="sa-field"><label>WhatsApp numbers (comma separated, country code)</label><input class="text-input" id="s-wa" value="${escapeHtml(nums)}"></div>
        <button class="btn primary" type="button" id="s-save"${isAdmin() ? "" : " disabled"}>Save settings</button>
        <button class="btn secondary" type="button" id="s-test"${isAdmin() ? "" : " disabled"}>Send WhatsApp test</button>
        <span id="s-msg" class="sa-muted"></span>
      </div>`;
    document.getElementById("s-role").onchange = async (e) => {
      const role = e.target.value;
      e.target.value = state.role;
      await switchToRole(role);
      e.target.value = state.role;
    };
    document.getElementById("s-save").onclick = async () => {
      const numbers = document
        .getElementById("s-wa")
        .value.split(",")
        .map((x) => x.trim())
        .filter(Boolean);
      await api("/site", {
        json: {
          admin_pin: document.getElementById("s-pin").value,
          quiet_hours_start: document.getElementById("s-qh1").value,
          quiet_hours_end: document.getElementById("s-qh2").value,
          whatsapp_numbers: numbers,
        },
      });
      await refresh();
      document.getElementById("s-msg").textContent = "Saved";
    };
    document.getElementById("s-test").onclick = async () => {
      const r = await api("/whatsapp/test", { method: "POST", json: { message: "Site Admin test" } });
      document.getElementById("s-msg").textContent = r.configured
        ? "Sent (check phone)"
        : "API not configured — set env vars";
    };
  }

  document.querySelectorAll(".sa-tabs [data-view]").forEach((b) => {
    b.onclick = () => go(b.getAttribute("data-view"));
  });

  document.getElementById("live-preview-toggle")?.addEventListener("click", toggleScanPreviewSidebar);

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      clearScanPreviewTimer();
      return;
    }
    if (scanPreviewWanted()) {
      ensureScanPreviewLoop();
      state.scanPreviewGen += 1;
      refreshPreviewFrames(state.scanPreviewGen);
    }
  });

  window.addEventListener("beforeunload", () => {
    clearScanPreviewTimer();
  });

  roleSwitch?.querySelectorAll(".sa-role-btn").forEach((btn) => {
    btn.addEventListener("click", () => switchToRole(btn.getAttribute("data-role")));
  });

  setRole(state.role);
  refresh()
    .then(loadLists)
    .then(() => go(isAdmin() ? "wizard" : "alerts"))
    .catch((e) => {
      main.innerHTML = `<p class="sa-error">Could not reach Site Admin API. Is Redis running? ${escapeHtml(e.message)}</p>`;
    });

  setInterval(() => {
    refresh().catch(() => {});
  }, 8000);
})();
