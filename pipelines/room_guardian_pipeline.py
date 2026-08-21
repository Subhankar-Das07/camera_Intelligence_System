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

from ultralytics import YOLO

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
    csrt_tracker: Any = field(default=None, repr=False)   # cv2 tracker or None
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


def _template_match(
    frame: np.ndarray,
    template: np.ndarray,
    threshold: float = 0.50
) -> tuple:
    """
    Try to re-locate a custom object by template matching.
    Returns (found: bool, bbox: [x, y, w, h] | None).

    Uses TM_CCOEFF_NORMED (normalised cross-correlation) which is robust to
    lighting changes and works well for small-to-medium appearance patches.
    Threshold 0.50 is intentionally permissive — CSRT takes over after
    re-detection so precision is handled there.
    """
    if template is None or template.size == 0:
        return False, None
    fh, fw = frame.shape[:2]
    th, tw = template.shape[:2]
    # Template must be smaller than the frame
    if tw >= fw or th >= fh:
        return False, None
    try:
        gray_frame = cv2.cvtColor(frame,    cv2.COLOR_BGR2GRAY)
        gray_tmpl  = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(gray_frame, gray_tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val >= threshold:
            tx, ty = max_loc
            return True, [tx, ty, tw, th]
    except Exception as exc:
        log.debug("[Guardian] Template match error: %s", exc)
    return False, None


def _feature_match_scaled(frame, template, scales=(1.0, 0.8, 1.25), min_matches=8):
    """ORB match against one template at a few scales, to catch size change."""
    orb = cv2.ORB_create(nfeatures=500)
    kp2, des2 = orb.detectAndCompute(frame, None)
    if des2 is None:
        return False, None
    best = None
    for s in scales:
        th, tw = template.shape[:2]
        resized = cv2.resize(template, (max(1, int(tw * s)), max(1, int(th * s))))
        kp1, des1 = orb.detectAndCompute(resized, None)
        if des1 is None:
            continue
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(bf.match(des1, des2), key=lambda m: m.distance)
        good = matches[:min_matches]
        if len(good) < min_matches:
            continue
        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            continue
        h, w = resized.shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(corners, H)
        x, y, w2, h2 = cv2.boundingRect(proj)
        score = len(good)
        if best is None or score > best[0]:
            best = (score, [max(0, int(x)), max(0, int(y)), max(1, int(w2)), max(1, int(h2))])
    return (True, best[1]) if best else (False, None)

def _multi_template_match(frame, template_bank, **kwargs):
    """Try the most recent templates first — recent appearance is most likely to still match."""
    for template in reversed(template_bank):
        found, bbox = _feature_match_scaled(frame, template, **kwargs)
        if found:
            return True, bbox
    return False, None

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

    def initialize(self, model_weight: str = "yolov8n.pt", **kwargs) -> None:
        """Load the YOLO detection model."""
        self.model = YOLO(model_weight)
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
        absence_threshold = int(fps * profile["absence_seconds"])
        ring_buf_size     = max(1, int(fps * 2))  # 2 second ring buffer
        yolo_tracker_cfg  = config.get("tracker", profile["yolo_tracker"])
        custom_tracker_cfg= config.get("custom_tracker", profile["custom_tracker"])
        conf_threshold    = profile["conf_threshold"]

        # Ensure correct model is loaded for this session
        req_model_weight = config.get("model_weight", "yolo11n.pt")
        if not hasattr(self, "model_name") or self.model_name != req_model_weight:
            self.model = YOLO(req_model_weight)
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

        # Initialise CSRT trackers on the first valid frame
        trackers_initialised = False

        # Rolling ring buffer of raw (un-annotated) frames
        ring_buffer: deque = deque(maxlen=ring_buf_size)

        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            raw_frame = frame.copy()   # un-annotated copy for ring buffer

            # ── Initialise CSRT trackers + extract appearance templates ────────
            if not trackers_initialised:
                for wo in watched:
                    if wo.obj_type == "custom":
                        x, y, w, h = wo.last_bbox
                        # Clamp to frame bounds
                        x = max(0, min(x, width  - 1))
                        y = max(0, min(y, height - 1))
                        w = max(1, min(w, width  - x))
                        h = max(1, min(h, height - y))
                        wo.last_bbox = [x, y, w, h]

                        # Extract appearance template for re-detection fallback
                        patch = frame[y:y + h, x:x + w]
                        if patch.size > 0:
                            wo.template_bank.append(patch.copy())
                            log.debug("[Guardian] Template extracted for %s — size %dx%d",
                                      wo.id, w, h)

                        # Initialise configurable custom tracker
                        if custom_tracker_cfg == "nano" and hasattr(cv2, "TrackerNano_create"):
                            tracker = cv2.TrackerNano_create()
                        elif custom_tracker_cfg == "vit" and hasattr(cv2, "TrackerVit_create"):
                            tracker = cv2.TrackerVit_create()
                        else:
                            tracker = cv2.TrackerCSRT_create()
                        tracker.init(frame, (x, y, w, h))
                        wo.csrt_tracker = tracker
                        log.debug("[Guardian] %s init for %s @ %s", custom_tracker_cfg.upper(), wo.id, (x, y, w, h))
                trackers_initialised = True

            # ── YOLO inference (tracked) ──────────────────────────────────────
            yolo_results = None
            detections_by_class: Dict[int, List[dict]] = {}
            has_yolo_objects = any(wo.obj_type == "yolo" for wo in watched)

            if has_yolo_objects:
                yolo_results = self.model.track(frame, persist=True, tracker=yolo_tracker_cfg, conf=conf_threshold, verbose=False)[0]

                if yolo_results.boxes is not None:
                    for box in yolo_results.boxes:
                        cls_id  = int(box.cls[0])
                        xywh    = box.xywh[0].cpu().numpy()
                        cx, cy, bw, bh = float(xywh[0]), float(xywh[1]), float(xywh[2]), float(xywh[3])
                        bx = int(cx - bw / 2)
                        by = int(cy - bh / 2)
                        track_id = int(box.id[0]) if box.id is not None else None
                        det_box = [bx, by, int(bw), int(bh)]
                        detections_by_class.setdefault(cls_id, []).append({
                            "bbox": det_box,
                            "track_id": track_id
                        })

            # ── Per-object presence check ─────────────────────────────────────
            alert_event = None

            for wo in watched:
                found = False

                if wo.obj_type == "yolo" and wo.yolo_class_id is not None:
                    candidates = detections_by_class.get(wo.yolo_class_id, [])
                    best_match = None

                    if wo.track_id is None:
                        # First frame: Assign track ID based on highest IoU
                        best_iou = 0.0
                        for det in candidates:
                            iou = _iou(wo.last_bbox, det["bbox"])
                            if iou > best_iou:
                                best_iou = iou
                                best_match = det
                        if best_iou >= 0.10 and best_match is not None:
                            wo.track_id = best_match["track_id"]
                            wo.last_bbox = best_match["bbox"]
                            found = True
                    else:
                        # Subsequent frames: Find by track_id
                        for det in candidates:
                            if det["track_id"] == wo.track_id:
                                best_match = det
                                break
                        if best_match is not None:
                            wo.last_bbox = best_match["bbox"]
                            found = True
                        else:
                            # Fallback: tracker lost ID (fast motion / occlusion). 
                            # Re-acquire closest detection of same class that isn't claimed by another watched object.
                            claimed_track_ids = {other_wo.track_id for other_wo in watched if other_wo != wo and other_wo.track_id is not None}
                            
                            best_iou = 0.0
                            for det in candidates:
                                if det["track_id"] in claimed_track_ids:
                                    continue
                                iou = _iou(wo.last_bbox, det["bbox"])
                                if iou > best_iou:
                                    best_iou = iou
                                    best_match = det
                            
                            if best_iou >= 0.10 and best_match is not None:
                                wo.track_id = best_match["track_id"]
                                wo.last_bbox = best_match["bbox"]
                                found = True

                elif wo.obj_type == "custom" and wo.csrt_tracker is not None:
                    # Primary: CSRT tracker update
                    success, bbox = wo.csrt_tracker.update(frame)
                    if success:
                        x, y, w, h   = [int(v) for v in bbox]
                        wo.last_bbox = [x, y, w, h]
                        found        = True
                    else:
                        # Fallback: ORB feature matching — actively re-locate the object
                        tm_found, new_bbox = _multi_template_match(frame, wo.template_bank)
                        if tm_found and new_bbox is not None:
                            nx, ny, nw, nh = new_bbox
                            wo.last_bbox   = [nx, ny, nw, nh]
                            # Re-init tracker at the re-detected location
                            if custom_tracker_cfg == "nano" and hasattr(cv2, "TrackerNano_create"):
                                new_tracker = cv2.TrackerNano_create()
                            elif custom_tracker_cfg == "vit" and hasattr(cv2, "TrackerVit_create"):
                                new_tracker = cv2.TrackerVit_create()
                            else:
                                new_tracker = cv2.TrackerCSRT_create()
                            new_tracker.init(frame, (nx, ny, nw, nh))
                            wo.csrt_tracker = new_tracker
                            found = True
                            log.debug("[Guardian] %s re-detected via ORB match @ %s",
                                      wo.id, new_bbox)

                # ── Countdown logic ───────────────────────────────────────────
                if found:
                    x, y, w, h = wo.last_bbox
                    wo.center_history.append((x + w/2, y + h/2))
                    
                    if wo.obj_type == "custom":
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
