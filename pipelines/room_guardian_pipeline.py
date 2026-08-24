"""
Room Guardian Pipeline  v4.0 - YOLOv8 + ByteTrack Edition
============================================================================
Refactored to use ultralytics, supervision, and trackers libraries.
- Leverages state-of-the-art YOLOv8 for extremely fast CPU detection.
- Uses ByteTrackTracker for robust object tracking.
- Removed legacy custom Kalman Filter and appearance matching in favor of 
  efficient built-in tracking mechanisms.
- Maintains missing object alerts (3 seconds out of frame).
"""

import cv2
import numpy as np
import uuid
import os
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from collections import deque

from ultralytics import YOLO
import supervision as sv
from trackers import ByteTrackTracker

from core.base_pipeline import BaseVideoPipeline
from core.video_source import get_video_source

log = logging.getLogger(__name__)

# ── Tuning constants ──────────────────────────────────────────────────────────
CONF_THRESHOLD   = 0.20       # Detection confidence threshold
ANCHOR_IOU       = 0.40       # Min IoU to anchor user's scan box to a ByteTrack box

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

    # Tracking counters
    missing_frames: int = 0             # frames since object is truly absent

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

def _draw_label(frame: np.ndarray, text: str, x: int, y: int, color: tuple) -> None:
    font      = cv2.FONT_HERSHEY_SIMPLEX
    scale     = 0.55
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    pad = 4
    cv2.rectangle(frame, (x, y - th - pad * 2), (x + tw + pad * 2, y), COLOR_LABEL_BG, -1)
    cv2.putText(frame, text, (x + pad, y - pad), font, scale, color, thickness, cv2.LINE_AA)


def _write_clip(frames: list, output_path: str, fps: float, size: tuple) -> bool:
    """Write a list/deque of frames to a .mp4 file. Returns True on success."""
    if not frames:
        return False
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, size)
    if not writer.isOpened():
        log.error("[Guardian] Could not open VideoWriter for %s", output_path)
        return False
    for f in frames:
        writer.write(f)
    writer.release()
    return True


# ── Robust Exit Detection ────────────────────────────────────────────────────

def _detect_exit(center_history: deque, width: int, height: int, margin: float = 0.05) -> str:
    """
    Determine exit type based on trajectory.
    Returns 'left_frame', 'occluded_or_removed', or 'unknown'.
    """
    if len(center_history) < 3:
        return "unknown"
    # Use last 3 points for velocity
    pts = list(center_history)[-3:]
    dx = pts[-1][0] - pts[0][0]
    dy = pts[-1][1] - pts[0][1]
    last_x, last_y = pts[-1]

    near_right = last_x > width * (1 - margin)
    near_left  = last_x < width * margin
    near_bottom = last_y > height * (1 - margin)
    near_top   = last_y < height * margin

    # Motion outward if near edge and moving further outward
    if (near_right and dx > 0) or (near_left and dx < 0) or (near_bottom and dy > 0) or (near_top and dy < 0):
        return "left_frame"
    else:
        return "occluded_or_removed"


# ── Environment profiles ──────────────────────────────────────────────────────
ENV_PROFILES = {
    "home":    {"absence_seconds": 5,  "conf_threshold": CONF_THRESHOLD},
    "shop":    {"absence_seconds": 3,  "conf_threshold": CONF_THRESHOLD},
    "factory": {"absence_seconds": 8,  "conf_threshold": CONF_THRESHOLD},
}


# ── Pipeline ──────────────────────────────────────────────────────────────────

class RoomGuardianPipeline(BaseVideoPipeline):
    """
    Room Object Guardian pipeline — v4.0 (ultralytics + trackers Edition).
    """

    def initialize(self, model_weight: str = "yolov8s-640", **kwargs) -> None:
        if not model_weight.endswith(".pt"):
            model_weight = "yolov8s.pt"
        self.model      = YOLO(model_weight)
        self.model_name = model_weight
        self.tracker    = ByteTrackTracker()
        log.info("[Guardian] v4.0 initialized with model: %s", model_weight)

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

        # Reload Model if model weight changed
        req_weight = config.get("model_weight", "yolov8s-640")
        if not hasattr(self, "model_name") or self.model_name != req_weight:
            if not req_weight.endswith(".pt"):
                req_weight = "yolov8s.pt"
            self.model      = YOLO(req_weight)
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

        # Reset tracker state
        self.tracker.reset()

        # ── Frame loop ────────────────────────────────────────────────────────
        while cap.isOpened():
            alert_event = None
            ret, frame = cap.read()
            if not ret:
                break

            raw_frame = frame.copy()   # for ring buffer — un-annotated

            # 1. Run inference and tracking
            try:
                results = self.model(frame, conf=conf_threshold, verbose=False)[0]
                detections = sv.Detections.from_ultralytics(results)
                detections = self.tracker.update(detections)
            except Exception as e:
                log.warning("[Guardian] Inference/Tracking error frame %d: %s", frame_idx, e)
                detections = sv.Detections.empty()

            # Extract current frame tracked boxes
            tracked_boxes = {}  # dict of track_id -> [x,y,w,h]
            if detections is not None and len(detections) > 0:
                for xyxy, mask, confidence, class_id, tracker_id, data in detections:
                    if tracker_id is not None:
                        bx = int(xyxy[0])
                        by = int(xyxy[1])
                        bw = int(xyxy[2] - xyxy[0])
                        bh = int(xyxy[3] - xyxy[1])
                        tracked_boxes[int(tracker_id)] = [bx, by, bw, bh]

            assigned_track_ids = set()

            # ── 2. Object Association & Tracking ──
            for wo in watched:
                wo.track_ok = False

                # Case A: We don't have a track_id yet, anchor to the best overlapping box
                if wo.track_id is None:
                    best_iou = 0
                    best_tid = None
                    for tid, tbox in tracked_boxes.items():
                        if tid in assigned_track_ids:
                            continue
                        iou = _iou(wo.last_bbox, tbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_tid = tid
                    
                    if best_tid is not None and best_iou > ANCHOR_IOU:
                        wo.track_id = best_tid
                        wo.last_bbox = tracked_boxes[best_tid]
                        wo.track_ok = True
                        assigned_track_ids.add(best_tid)
                else:
                    # Case B: We have a track_id, check if it's still alive
                    if wo.track_id in tracked_boxes:
                        wo.last_bbox = tracked_boxes[wo.track_id]
                        wo.track_ok = True
                        assigned_track_ids.add(wo.track_id)
                    else:
                        # Tracker lost this ID.
                        wo.track_ok = False

                if wo.track_ok:
                    cx = wo.last_bbox[0] + wo.last_bbox[2]/2
                    cy = wo.last_bbox[1] + wo.last_bbox[3]/2
                    wo.center_history.append((cx, cy))

                # ── 3. Update missing counters & alert logic ──
                if wo.track_ok:
                    was_missing       = wo.missing_frames > 0
                    wo.missing_frames = 0
                    if wo.alert_fired and was_missing:
                        wo.alert_fired = False
                        log.info("[Guardian] %s re-appeared — alert cleared.", wo.id)
                else:
                    wo.missing_frames += 1

                # Alert scheduling
                if wo.missing_frames >= absence_threshold and not wo.alert_fired:
                    wo.alert_fired = True

                    # Use robust exit detection
                    exit_type = _detect_exit(wo.center_history, width, height)
                    if exit_type == "unknown":
                        exit_type = "occluded_or_removed"  # fallback

                    disappearance_frame = max(0, frame_idx - absence_threshold)
                    pending_alerts.append({
                        "wo":                 wo,
                        "target_frame":       frame_idx + int(fps * 2),
                        "disappearance_frame": disappearance_frame,
                        "exit_type":          exit_type,
                    })
                    log.info("[Guardian] Scheduled alert for %s (disappearance ~frame %d, exit_type=%s).",
                             wo.id, disappearance_frame, exit_type)

                # ── Draw tracking annotation ──
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
                    clip_name = f"guardian_{alert_id}.mp4"
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
                f"GUARDIAN v4.0 | YOLOv8+ByteTrack | Watching: {guarded_count} | Missing: {missing_count} | Alerts: {alerted_count}",
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud_color, 2, cv2.LINE_AA,
            )

            yield frame, alert_event
            frame_idx += 1

        cap.release()
        log.info("[Guardian] Stream ended after %d frames.", frame_idx)