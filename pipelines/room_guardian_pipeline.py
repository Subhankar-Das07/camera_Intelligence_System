"""
Room Guardian Pipeline  v3.0 - ByteTrack Edition
================================================
Core tracking engine using Ultralytics native ByteTrack (via FastSAM).
Substitutes CSRT entirely for robust, multi-object tracking.

Architecture
------------
- Runs `model.track(..., tracker="bytetrack.yaml")` on every frame.
- ByteTrack natively handles short-term occlusions and track associations.
- `WatchedObject` instances are anchored to specific ByteTrack `track_id`s.
- If a `track_id` is lost by ByteTrack, a fallback appearance matching 
  (histogram) is used to re-associate to a new `track_id` if FastSAM fragmented it.
- strict loss detection: if the track ID is missing and appearance doesn't match,
  the missing_frames counter ticks up immediately.
"""

import cv2
import numpy as np
import uuid
import os
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from collections import deque

from ultralytics import FastSAM

from core.base_pipeline import BaseVideoPipeline
from core.video_source import get_video_source

log = logging.getLogger(__name__)

# ── Tuning constants ──────────────────────────────────────────────────────────
CONF_THRESHOLD   = 0.60       # FastSAM confidence gate
ANCHOR_IOU       = 0.40       # Min IoU to anchor user's scan box to a ByteTrack box
REINIT_IOU       = 0.30       # Re-init track if IoU drops but appearance matches

# ── Colours ───────────────────────────────────────────────────────────────────
COLOR_PRESENT  = (0, 255, 100)   # green  - tracking OK
COLOR_MISSING  = (0, 165, 255)   # orange - counting down
COLOR_ALERT    = (0, 0, 255)     # red    - alert fired, still absent
COLOR_LABEL_BG = (20, 20, 20)


# ── WatchedObject ─────────────────────────────────────────────────────────────

@dataclass
class WatchedObject:
    """All state needed to track one user-selected object via ByteTrack."""
    id:             str                  # unique id from frontend scan
    label:          str                  # class name or "Object"
    last_bbox:      List[int]            # [x, y, w, h] pixels (absolute)

    # ByteTrack state
    track_id:       Optional[int] = None # The assigned global track ID
    track_ok:       bool = False         # Was it found in the latest frame?

    # Appearance cache for re-association if ByteTrack drops the ID
    hist_bank:      list  = field(default_factory=list, repr=False)   # HSV hists

    # Tracking counters
    missing_frames: int = 0             # frames since object is truly absent
    frames_since_update: int = 0

    # Alert state
    alert_fired:    bool = False
    center_history: deque = field(default_factory=lambda: deque(maxlen=30), repr=False)


# ── Utility functions ─────────────────────────────────────────────────────────

def _iou(boxA: List[int], boxB: List[int]) -> float:
    """Compute IoU between two [x, y, w, h] boxes (absolute pixels)."""
    ax1, ay1 = boxA[0], boxA[1]
    ax2, ay2 = ax1 + boxA[2], ay1 + boxA[3]
    bx1, by1 = boxB[0], boxB[1]
    bx2, by2 = bx1 + boxB[2], by1 + boxB[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = boxA[2] * boxA[3] + boxB[2] * boxB[3] - inter
    return inter / union if union > 0 else 0.0


def _compute_hist(patch: np.ndarray) -> Optional[np.ndarray]:
    """Compute a normalized HSV histogram from an BGR patch."""
    if patch is None or patch.size == 0:
        return None
    try:
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist
    except Exception:
        return None


def _hist_match_precomputed(query_hist: np.ndarray, hist_bank: list, threshold: float = 0.42) -> bool:
    """Compare a query histogram against a bank of histograms."""
    if not hist_bank or query_hist is None:
        return False
    for tmpl_hist in reversed(hist_bank):
        if tmpl_hist is None:
            continue
        dist = cv2.compareHist(tmpl_hist, query_hist, cv2.HISTCMP_BHATTACHARYYA)
        if dist < threshold:
            return True
    return False


def _safe_patch(frame: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
    """Safely extract a patch from frame given [x, y, w, h] bbox."""
    fh, fw = frame.shape[:2]
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    x = max(0, min(x, fw - 1))
    y = max(0, min(y, fh - 1))
    w = max(1, min(w, fw - x))
    h = max(1, min(h, fh - y))
    patch = frame[y:y + h, x:x + w]
    return patch if patch.size > 0 else None


def _draw_label(frame: np.ndarray, text: str, x: int, y: int, color: tuple) -> None:
    font      = cv2.FONT_HERSHEY_SIMPLEX
    scale     = 0.55
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    pad = 4
    cv2.rectangle(frame, (x, y - th - pad * 2), (x + tw + pad * 2, y), COLOR_LABEL_BG, -1)
    cv2.putText(frame, text, (x + pad, y - pad), font, scale, color, thickness, cv2.LINE_AA)


def _write_clip(frames: list, output_path: str, fps: float, size: tuple) -> bool:
    """Write a list/deque of frames to a .webm file. Returns True on success."""
    if not frames:
        return False
    fourcc = cv2.VideoWriter_fourcc(*"vp80")
    writer = cv2.VideoWriter(output_path, fourcc, fps, size)
    if not writer.isOpened():
        log.error("[Guardian] Could not open VideoWriter for %s", output_path)
        return False
    for f in frames:
        writer.write(f)
    writer.release()
    return True


# ── Environment profiles ──────────────────────────────────────────────────────
ENV_PROFILES = {
    "home":    {"absence_seconds": 5,  "conf_threshold": CONF_THRESHOLD},
    "shop":    {"absence_seconds": 3,  "conf_threshold": CONF_THRESHOLD},
    "factory": {"absence_seconds": 8,  "conf_threshold": CONF_THRESHOLD},
}


# ── Pipeline ──────────────────────────────────────────────────────────────────

class RoomGuardianPipeline(BaseVideoPipeline):
    """
    Room Object Guardian pipeline — v3.0 (ByteTrack).
    """

    def initialize(self, model_weight: str = "FastSAM-s.pt", **kwargs) -> None:
        self.model      = FastSAM(model_weight)
        self.model_name = model_weight
        log.info("[Guardian] v3.0 (ByteTrack) initialized with model: %s", model_weight)

    def process_frame(self, frame, frame_idx, roi_polygon, config):
        return frame, {}

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run_on_video(self, input_path, output_dir, roi_normalized, config):
        """
        Generator: yields (annotated_frame, alert_event_or_None).
        """
        cap    = get_video_source(input_path)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Profile settings
        env_profile_name  = config.get("environment_profile", "home")
        profile           = ENV_PROFILES.get(env_profile_name, ENV_PROFILES["home"])
        absence_threshold = max(1, int(fps * profile["absence_seconds"]))
        ring_buf_size     = max(1, int(fps * 4.5))   # 4.5s ring buffer
        conf_threshold    = profile["conf_threshold"]

        # Reload FastSAM if model weight changed
        req_weight = config.get("model_weight", "FastSAM-s.pt")
        if not hasattr(self, "model_name") or self.model_name != req_weight:
            self.model      = FastSAM(req_weight)
            self.model_name = req_weight
            log.info("[Guardian] Reloaded model: %s", req_weight)

        # ── Build WatchedObject list ──────────────────────────────────────────
        raw_objects: List[Dict] = config.get("watched_objects", [])
        if not raw_objects:
            log.warning("[Guardian] No watched_objects in config — nothing to guard.")

        watched: List[WatchedObject] = []
        for obj in raw_objects:
            bn = obj.get("bbox_normalized", [0, 0, 0.1, 0.1])
            px = [
                int(bn[0] * width),
                int(bn[1] * height),
                int(bn[2] * width),
                int(bn[3] * height),
            ]
            wo = WatchedObject(
                id       = obj.get("id", str(uuid.uuid4())),
                label    = obj.get("label", "Object"),
                last_bbox= px,
            )
            watched.append(wo)

        # Rolling ring buffer (raw un-annotated frames)
        ring_buffer: deque  = deque(maxlen=ring_buf_size)
        pending_alerts: list = []
        frame_idx   = 0

        # Reset ultralytics tracker state
        # The first track() call internally initializes it, but we can pass persist=True

        # ── Frame loop ────────────────────────────────────────────────────────
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            raw_frame = frame.copy()   # for ring buffer — un-annotated

            # 1. Run ByteTrack via FastSAM natively
            try:
                results = self.model.track(
                    frame, 
                    persist=True, 
                    tracker="bytetrack.yaml", 
                    conf=conf_threshold, 
                    verbose=False
                )[0]
            except Exception as e:
                log.warning("[Guardian] FastSAM ByteTrack error frame %d: %s", frame_idx, e)
                results = None

            # Extract current frame tracked boxes
            tracked_boxes = {}  # dict of track_id -> [x,y,w,h]
            if results and results.boxes is not None and results.boxes.id is not None:
                for box, t_id in zip(results.boxes, results.boxes.id):
                    xywh = box.xywh[0].cpu().numpy()
                    cx, cy, bw, bh = float(xywh[0]), float(xywh[1]), float(xywh[2]), float(xywh[3])
                    bx = int(cx - bw / 2)
                    by = int(cy - bh / 2)
                    tracked_boxes[int(t_id)] = [bx, by, int(bw), int(bh)]

            assigned_track_ids = set()

            # ── 2. Object Association & Tracking ──────────────────────────────
            for wo in watched:
                wo.track_ok = False
                
                # A. Frame 0 Initialization (anchor user's box to a track_id)
                if frame_idx == 0 or wo.track_id is None:
                    best_iou = 0.0
                    best_tid = None
                    for tid, tbox in tracked_boxes.items():
                        if tid in assigned_track_ids: continue
                        iou = _iou(wo.last_bbox, tbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_tid = tid
                    
                    if best_iou > ANCHOR_IOU and best_tid is not None:
                        wo.track_id = best_tid
                        wo.last_bbox = tracked_boxes[best_tid]
                        wo.track_ok = True
                        log.info("[Guardian] Anchored %s to ByteTrack ID %d (IoU %.2f)", wo.id, best_tid, best_iou)

                        # Build initial histogram
                        patch = _safe_patch(frame, wo.last_bbox)
                        h_val = _compute_hist(patch)
                        if h_val is not None:
                            wo.hist_bank.append(h_val)

                # B. Normal Tracking
                elif wo.track_id in tracked_boxes:
                    wo.last_bbox = tracked_boxes[wo.track_id]
                    wo.track_ok = True

                # C. Track Recovery (ByteTrack lost the ID, or ID changed)
                else:
                    # Look through unassigned boxes for an appearance + position match
                    best_app_iou = 0.0
                    recovery_tid = None
                    for tid, tbox in tracked_boxes.items():
                        if tid in assigned_track_ids: continue
                        patch_fb = _safe_patch(frame, tbox)
                        fb_hist  = _compute_hist(patch_fb)
                        if _hist_match_precomputed(fb_hist, wo.hist_bank):
                            iou = _iou(wo.last_bbox, tbox)
                            if iou > best_app_iou:
                                best_app_iou = iou
                                recovery_tid = tid

                    # Re-anchor if we have a solid match
                    if recovery_tid is not None and best_app_iou > REINIT_IOU:
                        old_id = wo.track_id
                        wo.track_id = recovery_tid
                        wo.last_bbox = tracked_boxes[recovery_tid]
                        wo.track_ok = True
                        log.debug("[Guardian] Recovered %s! Changed track_id %s -> %s (IoU %.2f)", 
                                  wo.id, old_id, recovery_tid, best_app_iou)

                # D. Update state
                if wo.track_ok:
                    assigned_track_ids.add(wo.track_id)
                    x, y, w, h = wo.last_bbox
                    wo.center_history.append((x + w / 2, y + h / 2))
                    
                    # Update histogram bank every 2 seconds
                    wo.frames_since_update += 1
                    if wo.frames_since_update >= int(fps * 2):
                        patch = _safe_patch(frame, wo.last_bbox)
                        h_val = _compute_hist(patch)
                        if h_val is not None:
                            wo.hist_bank = wo.hist_bank[-5:]
                            wo.hist_bank.append(h_val)
                        wo.frames_since_update = 0

            # ── 3. Missing Counter & Alerts ───────────────────────────────────
            alert_event = None

            for wo in watched:
                if wo.track_ok:
                    was_missing       = wo.missing_frames > 0
                    wo.missing_frames = 0
                    if wo.alert_fired and was_missing:
                        wo.alert_fired = False
                        log.info("[Guardian] %s re-appeared — alert cleared.", wo.id)
                else:
                    # Clean increment! No grace period needed because ByteTrack 
                    # internally handles short-term occlusion. If it's gone here, it's GONE.
                    wo.missing_frames += 1

                # Alert scheduling
                if wo.missing_frames == absence_threshold and not wo.alert_fired:
                    wo.alert_fired = True

                    exit_type = "occluded_or_removed"
                    if len(wo.center_history) > 0:
                        last_cx, last_cy = wo.center_history[-1]
                        margin_x = width  * 0.05
                        margin_y = height * 0.05
                        if (last_cx <= margin_x or last_cx >= width  - margin_x or
                                last_cy <= margin_y or last_cy >= height - margin_y):
                            exit_type = "left_frame"

                    disappearance_frame = max(0, frame_idx - absence_threshold)
                    pending_alerts.append({
                        "wo":                 wo,
                        "target_frame":       frame_idx + int(fps * 2),
                        "disappearance_frame": disappearance_frame,
                        "exit_type":          exit_type,
                    })
                    log.info("[Guardian] Scheduled alert for %s (disappearance ~frame %d).",
                             wo.id, disappearance_frame)

                # Draw tracking annotation
                x, y, w, h = wo.last_bbox
                x2, y2 = x + w, y + h
                if wo.alert_fired:
                    color = COLOR_ALERT
                    status_text = f"{wo.label} [MISSING!]"
                elif wo.missing_frames > 0:
                    pct = min(100, int(wo.missing_frames / absence_threshold * 100))
                    color = COLOR_MISSING
                    status_text = f"{wo.label} [Lost {pct}%]"
                else:
                    color = COLOR_PRESENT
                    status_text = f"{wo.label} [ID:{wo.track_id}]"

                cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
                _draw_label(frame, status_text, x, y, color)

            # ── Ring buffer update ────────────────────────────────────────────
            ring_buffer.append(raw_frame)

            # ── Process pending alerts ────────────────────────────────────────
            for pa in pending_alerts[:]:
                if frame_idx >= pa["target_frame"]:
                    wo = pa["wo"]
                    alert_id  = str(uuid.uuid4())
                    clip_name = f"guardian_{alert_id}.webm"
                    clip_path = os.path.join(output_dir, clip_name)
                    clip_url  = f"/storage/alerts/{clip_name}"

                    frames_needed   = int(fps * 4)
                    frames_to_write = list(ring_buffer)[-frames_needed:] \
                                      if len(ring_buffer) > frames_needed else list(ring_buffer)

                    success = _write_clip(frames_to_write, clip_path, fps, (width, height))
                    if success:
                        ts_sec = pa["disappearance_frame"] / fps
                        alert_event = {
                            "id":                    alert_id,
                            "object_id":             wo.id,
                            "object_label":          wo.label,
                            "timestamp_sec":         ts_sec,
                            "formatted_time":        f"{int(ts_sec // 60):02d}:{int(ts_sec % 60):02d}",
                            "clip_url":              clip_url,
                            "severity":              "MISSING",
                            "immediate_buzzer_trigger": True,
                            "exit_type":             pa["exit_type"],
                        }
                        log.info("[Guardian] ALERT fired for %s. Clip: %s", wo.label, clip_path)

                    pending_alerts.remove(pa)
                    break   # one alert per frame

            # ── Guardian HUD ──────────────────────────────────────────────────
            guarded_count = len(watched)
            missing_count = sum(1 for wo in watched if wo.missing_frames > 0)
            alerted_count = sum(1 for wo in watched if wo.alert_fired)
            hud_color     = (0, 0, 255) if alerted_count > 0 else (0, 255, 100)
            cv2.putText(
                frame,
                f"GUARDIAN | ByteTrack | Watching: {guarded_count} | Missing: {missing_count} | Alerts: {alerted_count}",
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud_color, 2, cv2.LINE_AA,
            )

            yield frame, alert_event
            frame_idx += 1

        cap.release()
        log.info("[Guardian] Stream ended after %d frames.", frame_idx)
