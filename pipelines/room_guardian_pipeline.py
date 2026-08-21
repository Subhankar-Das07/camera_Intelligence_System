"""
Room Guardian Pipeline
======================
Tracks a static camera's view and alerts when user-selected objects
disappear or go missing for more than 5 seconds.

Workflow
--------
Phase 1 (frontend): User scans the room via /api/guardian/scan → selects
  YOLO-detected objects (click) or draws custom ROI rectangles (drag).

Phase 2 (this file): /api/guardian/start passes the enrolled watched_objects
  list into config. run_on_video() tracks only those objects and yields
  (annotated_frame, alert_event_or_None) matching the BaseVideoPipeline contract.

Tracking strategy
-----------------
- YOLO-enrolled objects : IoU-matching against the correct class detections each frame.
- Custom-ROI objects    : OpenCV CSRT tracker (appearance-based, CPU-only, ~1ms/tracker/frame).

Alert logic
-----------
- A 2-second rolling ring buffer of raw frames is maintained at all times.
- If an object is not found for >= fps * 5 consecutive frames (5 seconds):
    → ring buffer is flushed to a .webm clip (the last 2 s before disappearance).
    → alert_event is yielded to the MJPEG pipeline and collected by the session.
- If the object re-appears BEFORE the 5-second threshold: missing_frames resets silently.
- After an alert fires: alert_fired flag prevents duplicate alerts for the same
  continuous absence. If the object later re-appears, the flag clears so a future
  disappearance can trigger a new alert.
"""

import cv2
import numpy as np
import uuid
import os
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from collections import deque

from ultralytics import FastSAM

from core.base_pipeline import BaseVideoPipeline
from core.video_source import get_video_source

log = logging.getLogger(__name__)

# ── Colours for annotation ────────────────────────────────────────────────────
COLOR_PRESENT   = (0, 255, 100)    # green  — object currently visible
COLOR_MISSING   = (0, 165, 255)    # orange — counting down
COLOR_ALERT     = (0, 0, 255)      # red    — alert fired, still absent
COLOR_LABEL_BG  = (20, 20, 20)


# ── WatchedObject ─────────────────────────────────────────────────────────────

@dataclass
class WatchedObject:
    """All state needed to track one user-selected object."""
    id: str                             # "yolo-<uuid>" | "Unknown-1", ...
    obj_type: str                       # "yolo" | "custom"
    label: str                          # class name or "Unknown-N"
    yolo_class_id: Optional[int]        # None for custom objects
    last_bbox: List[int]                # [x, y, w, h] pixels (absolute)
    template_bank: deque = field(default_factory=lambda: deque(maxlen=5), repr=False)
    frames_since_bank_update: int = 0
    missing_frames: int = 0
    alert_fired: bool = False
    track_id: Optional[int] = None      # YOLO BoT-SORT track ID
    center_history: deque = field(default_factory=lambda: deque(maxlen=30), repr=False)


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
    """Draw a filled background label above a bounding box."""
    font      = cv2.FONT_HERSHEY_SIMPLEX
    scale     = 0.55
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 4
    rx1, ry1 = x, y - th - pad * 2
    rx2, ry2 = x + tw + pad * 2, y
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), COLOR_LABEL_BG, -1)
    cv2.putText(frame, text, (x + pad, y - pad), font, scale, color, thickness, cv2.LINE_AA)


def _write_clip(frames: deque, output_path: str, fps: float, size: tuple) -> bool:
    """Write a deque of frames to a .webm file. Returns True on success."""
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


def _color_hist_match(frame, box, template_bank, threshold=0.45):
    """Compare HSV histogram of a bounding box patch against the template bank."""
    bx, by, bw, bh = box
    fh, fw = frame.shape[:2]
    
    # Safe extract patch
    bx = max(0, min(bx, fw - 1))
    by = max(0, min(by, fh - 1))
    bw = max(1, min(bw, fw - bx))
    bh = max(1, min(bh, fh - by))
    patch = frame[by:by+bh, bx:bx+bw]
    if patch.size == 0:
        return False
        
    try:
        hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        hist_patch = cv2.calcHist([hsv_patch], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist_patch, hist_patch, 0, 1, cv2.NORM_MINMAX)
        
        for template in reversed(template_bank):
            hsv_tmpl = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
            hist_tmpl = cv2.calcHist([hsv_tmpl], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(hist_tmpl, hist_tmpl, 0, 1, cv2.NORM_MINMAX)
            
            dist = cv2.compareHist(hist_tmpl, hist_patch, cv2.HISTCMP_BHATTACHARYYA)
            if dist < threshold:
                return True
    except Exception as e:
        log.debug(f"[Guardian] Hist match err: {e}")
            
    return False


# Environment Profiles configuration
ENV_PROFILES = {
    "home": {
        "absence_seconds": 5,
        "conf_threshold": 0.25,
        "yolo_tracker": "botsort.yaml",
        "custom_tracker": "csrt"
    },
    "shop": {
        "absence_seconds": 3,
        "conf_threshold": 0.20,
        "yolo_tracker": "botsort.yaml",
        "custom_tracker": "csrt"
    },
    "factory": {
        "absence_seconds": 8,
        "conf_threshold": 0.25,
        "yolo_tracker": "botsort.yaml",
        "custom_tracker": "csrt"
    }
}


# ── Pipeline ──────────────────────────────────────────────────────────────────

class RoomGuardianPipeline(BaseVideoPipeline):
    """
    Room Object Guardian pipeline.
    Inherits BaseVideoPipeline; tracked via the standard registry + session system.
    """

    def initialize(self, model_weight: str = "FastSAM-s.pt", **kwargs) -> None:
        """Load the FastSAM detection model."""
        self.model = FastSAM(model_weight)
        self.model_name = model_weight
        log.info("[Guardian] Initialized with model: %s", model_weight)

    # process_frame is not used — all logic is in run_on_video
    def process_frame(self, frame, frame_idx, roi_polygon, config):
        return frame, {}

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run_on_video(self, input_path, output_dir, roi_normalized, config):
        """
        Generator: yields (annotated_frame, alert_event_or_None).

        config["watched_objects"] must be a list of dicts:
        [
          {
            "id":           str,          # "yolo-<uuid>" | "Unknown-N"
            "type":         "yolo"|"custom",
            "label":        str,
            "class_id":     int|null,     # for yolo type only
            "bbox_normalized": [x, y, w, h]  # 0-1 range, relative to frame
          },
          ...
        ]
        """
        cap = get_video_source(input_path)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Parse environment profile
        env_profile_name = config.get("environment_profile", "home")
        profile = ENV_PROFILES.get(env_profile_name, ENV_PROFILES["home"])
        absence_threshold = 20  # Hardcoded to 20 frames per user request
        ring_buf_size     = max(1, int(fps * 2))  # 2 second ring buffer
        yolo_tracker_cfg  = config.get("tracker", profile["yolo_tracker"])
        custom_tracker_cfg= config.get("custom_tracker", profile["custom_tracker"])
        conf_threshold    = profile["conf_threshold"]

        # FastSAM is class-agnostic, always use FastSAM-s.pt
        req_model_weight = config.get("model_weight", "FastSAM-s.pt")
        if not hasattr(self, "model_name") or self.model_name != req_model_weight:
            self.model = FastSAM(req_model_weight)
            self.model_name = req_model_weight
            log.info("[Guardian] Loaded model specific to session: %s", req_model_weight)

        # Parse enrolled objects from config
        raw_objects: List[Dict] = config.get("watched_objects", [])
        if not raw_objects:
            log.warning("[Guardian] No watched_objects in config — nothing to guard.")

        # Denormalise bboxes to pixel space
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
                id            = obj.get("id", str(uuid.uuid4())),
                obj_type      = obj.get("type", "custom"),
                label         = obj.get("label", "Unknown"),
                yolo_class_id = obj.get("class_id", None),
                last_bbox     = px,
            )
            watched.append(wo)

        # Rolling ring buffer of raw (un-annotated) frames
        ring_buffer: deque = deque(maxlen=ring_buf_size)

        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            raw_frame = frame.copy()   # un-annotated copy for ring buffer

            # Extract initial templates on Frame 0
            if frame_idx == 0:
                for wo in watched:
                    x, y, w, h = wo.last_bbox
                    x = max(0, min(x, width - 1))
                    y = max(0, min(y, height - 1))
                    w = max(1, min(w, width - x))
                    h = max(1, min(h, height - y))
                    wo.last_bbox = [x, y, w, h]
                    patch = frame[y:y + h, x:x + w]
                    if patch.size > 0:
                        wo.template_bank.append(patch.copy())

            # ── FastSAM inference (tracked) ──────────────────────────────────────
            # FastSAM tracks all distinct objects it segments
            yolo_results = self.model.track(frame, persist=True, tracker=yolo_tracker_cfg, conf=conf_threshold, verbose=False)[0]

            # Collect all tracked bounding boxes
            tracked_boxes = []
            if yolo_results.boxes is not None:
                for box in yolo_results.boxes:
                    if box.id is None:
                        continue
                    xywh    = box.xywh[0].cpu().numpy()
                    cx, cy, bw, bh = float(xywh[0]), float(xywh[1]), float(xywh[2]), float(xywh[3])
                    bx = int(cx - bw / 2)
                    by = int(cy - bh / 2)
                    track_id = int(box.id[0])
                    det_box = [bx, by, int(bw), int(bh)]
                    tracked_boxes.append({
                        "bbox": det_box,
                        "track_id": track_id
                    })

            # Track which IDs have been assigned to watched objects in this frame
            claimed_track_ids = set()

            # ── 1. Update watched objects by their existing track_id ────────
            for wo in watched:
                wo._found_this_frame = False
                if wo.track_id is not None:
                    for det in tracked_boxes:
                        if det["track_id"] == wo.track_id:
                            wo.last_bbox = det["bbox"]
                            wo._found_this_frame = True
                            claimed_track_ids.add(wo.track_id)
                            break

            # ── 2. Spatial Re-association & Frame 0 Enrollment ──────────────
            for wo in watched:
                if wo._found_this_frame:
                    continue
                
                # If the object lost its track ID (occlusion) OR it's Frame 0 (needs initial assignment),
                # we do a spatial association: find the unassigned box with the highest IoU to last_bbox.
                best_iou = 0.0
                best_match = None
                
                for det in tracked_boxes:
                    # Allow multiple watched objects to claim the same track_id if they merge
                    iou = _iou(wo.last_bbox, det["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_match = det
                
                # For initial assignment (Frame 0), threshold is permissive (0.10).
                # For occlusion recovery (Frame N), threshold is strict (0.65) to avoid swapping IDs with walking people.
                req_iou = 0.10 if wo.track_id is None else 0.65
                
                if best_iou >= req_iou and best_match is not None:
                    wo.track_id = best_match["track_id"]
                    wo.last_bbox = best_match["bbox"]
                    wo._found_this_frame = True
                    claimed_track_ids.add(wo.track_id)
                    log.debug(f"[Guardian] Re-associated {wo.id} to track_id {wo.track_id} (IoU {best_iou:.2f})")

                # 2. Appearance Re-association (for moving objects)
                if not wo._found_this_frame and len(wo.template_bank) > 0:
                    best_match_tmpl = None
                    for det in tracked_boxes:
                        # We specifically do NOT check claimed_track_ids here.
                        # If two objects are moved together (e.g. on a tray), FastSAM might merge them 
                        # into a single track_id. Both WatchedObjects should be allowed to attach to it!
                        
                        # Compare color histograms to see if this FastSAM box is our object
                        if _color_hist_match(frame, det["bbox"], wo.template_bank):
                            best_match_tmpl = det
                            break
                            
                    if best_match_tmpl is not None:
                        wo.track_id = best_match_tmpl["track_id"]
                        wo.last_bbox = best_match_tmpl["bbox"]
                        wo._found_this_frame = True
                        claimed_track_ids.add(wo.track_id)
                        log.debug(f"[Guardian] Appearance re-associated {wo.id} to track_id {wo.track_id}")

            # ── Countdown logic ───────────────────────────────────────────
            alert_event = None

            for wo in watched:
                found = wo._found_this_frame
                
                if found:
                    x, y, w, h = wo.last_bbox
                    wo.center_history.append((x + w/2, y + h/2))
                    
                    wo.frames_since_bank_update += 1
                    if wo.frames_since_bank_update >= int(fps * 1.5):  # refresh ~every 1.5s
                        patch = frame[y:y+h, x:x+w]
                        if patch.size > 0:
                            wo.template_bank.append(patch.copy())
                        wo.frames_since_bank_update = 0
                    
                    was_missing = wo.missing_frames > 0
                    wo.missing_frames = 0
                    # If object re-appears after a previous alert, allow re-alert
                    if wo.alert_fired and was_missing:
                        wo.alert_fired = False
                        log.info("[Guardian] %s re-appeared — alert cleared.", wo.id)
                else:
                    wo.missing_frames += 1

                # ── Alert firing ──────────────────────────────────────────────
                if wo.missing_frames >= absence_threshold and not wo.alert_fired:
                    wo.alert_fired = True
                    alert_id  = str(uuid.uuid4())
                    clip_name = f"guardian_{alert_id}.webm"
                    clip_path = os.path.join(output_dir, clip_name)
                    clip_url  = f"/storage/guardian_alerts/{clip_name}"

                    # Dump ring buffer (last 2 seconds before disappearance)
                    success = _write_clip(ring_buffer, clip_path, fps, (width, height))
                    if success:
                        ts_sec = frame_idx / fps
                        
                        exit_type = "occluded_or_removed"
                        if len(wo.center_history) > 0:
                            last_cx, last_cy = wo.center_history[-1]
                            margin_x = width * 0.05
                            margin_y = height * 0.05
                            if last_cx <= margin_x or last_cx >= width - margin_x or \
                               last_cy <= margin_y or last_cy >= height - margin_y:
                                exit_type = "left_frame"

                        alert_event = {
                            "id":                    alert_id,
                            "object_id":             wo.id,
                            "object_label":          wo.label,
                            "timestamp_sec":         ts_sec,
                            "formatted_time":        f"{int(ts_sec // 60):02d}:{int(ts_sec % 60):02d}",
                            "clip_url":              clip_url,
                            "severity":              "MISSING",
                            "immediate_buzzer_trigger": True,
                            "exit_type":             exit_type
                        }
                        log.info("[Guardian] ALERT — %s (%s) missing > 5s. Clip: %s",
                                 wo.label, wo.id, clip_path)

                # ── Draw tracking box on frame ────────────────────────────────
                x, y, w, h = wo.last_bbox
                x2, y2     = x + w, y + h

                if wo.alert_fired:
                    color       = COLOR_ALERT
                    status_text = f"{wo.label} [MISSING!]"
                elif wo.missing_frames > 0:
                    pct         = min(100, int(wo.missing_frames / absence_threshold * 100))
                    color       = COLOR_MISSING
                    status_text = f"{wo.label} [Lost {pct}%]"
                else:
                    color       = COLOR_PRESENT
                    status_text = f"{wo.label} [OK]"

                cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
                _draw_label(frame, status_text, x, y, color)

            # ── Guardian HUD ─────────────────────────────────────────────────
            guarded_count  = len(watched)
                
            missing_count  = sum(1 for wo in watched if wo.missing_frames > 0)
            alerted_count  = sum(1 for wo in watched if wo.alert_fired)
            hud_color      = (0, 0, 255) if alerted_count > 0 else (0, 255, 100)
            cv2.putText(frame,
                        f"GUARDIAN | Watching: {guarded_count} | Missing: {missing_count} | Alerts: {alerted_count}",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud_color, 2, cv2.LINE_AA)

            # Update ring buffer with the un-annotated raw frame
            ring_buffer.append(raw_frame)

            yield frame, alert_event
            frame_idx += 1

        cap.release()
        log.info("[Guardian] Stream ended after %d frames.", frame_idx)
