"""
Fall Detection Pipeline (YOLO Object Detection + ByteTrack)

Architecture: Simple state machine per tracked person.
  NORMAL -> CONFIRMED (if class is 'falling' for N consecutive frames) -> alert
  CONFIRMED -> NORMAL (if class is 'standing' or lost)
"""

import cv2
import time
import numpy as np
from ultralytics import YOLO
import uuid
import os
import logging
from collections import deque
from core.base_pipeline import BaseVideoPipeline
from core.video_source import get_video_source

log = logging.getLogger(__name__)

class _PersonState:
    """Lightweight per-track state container."""
    __slots__ = (
        "state",             # "NORMAL" | "CONFIRMED"
        "down_frame_count",  # consecutive frames person looks "down"
        "last_frame",
    )

    def __init__(self):
        self.state          = "NORMAL"
        self.down_frame_count = 0
        self.last_frame     = 0


class FallDetectionPipeline(BaseVideoPipeline):

    DEFAULT_CONFIG = {
        # Model
        "person_conf_threshold":  0.45,
        "fall_class_index":       1,      # Class index for 'falling'
        
        # Temporal buffer for confirmation
        "confirm_frames":         10,     # consecutive frames in "falling" class to confirm

        # Alert / recording
        "alert_cooldown_sec":     12.0,
        "post_alert_patience_sec": 3.0,
        "pre_event_buffer_sec":   3.0,
        "track_expiry_sec":       15.0,
    }

    def initialize(self, model_weight: str = "best.pt"):
        # Resolve the path relative to the project root (one level up from the pipelines folder)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, model_weight)
        
        if not os.path.exists(model_path):
            log.warning(f"Model not found at {model_path}. Trying current directory.")
            model_path = model_weight
            
        self.model = YOLO(model_path)
        self._states: dict[int, _PersonState] = {}

    def process_frame(self, frame, frame_idx, roi_polygon, config):
        pass  # streaming is handled inside run_on_video

    def run_on_video(self, input_path, output_dir, roi_normalized, config):
        cfg = {**self.DEFAULT_CONFIG, **(config or {})}
        cap = get_video_source(input_path)

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0

        expiry_frames   = int(fps * cfg["track_expiry_sec"])
        cooldown_max    = int(fps * cfg["alert_cooldown_sec"])
        patience_frames = max(1, int(fps * cfg["post_alert_patience_sec"]))
        pre_buf_frames  = max(1, int(fps * cfg["pre_event_buffer_sec"]))
        confirm_frames  = cfg["confirm_frames"]
        fall_class_idx  = cfg["fall_class_index"]

        conf_thresh = cfg["person_conf_threshold"]

        frame_idx         = 0
        cooldown          = 0
        alert_active      = False
        alert_writer      = None
        alert_id          = None
        alert_start_frame = 0
        no_detect_frames  = 0
        frame_buffer      = deque(maxlen=pre_buf_frames)
        start_time        = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if cooldown > 0:
                cooldown -= 1

            # track all classes, we'll filter below
            # Note: Do not pass classes=[0] here if we need multiple classes from custom model
            results = self.model.track(
                frame, conf=conf_thresh, imgsz=640,
                persist=True, verbose=False,
            )[0]
            frame = results.plot()

            any_flagged    = False
            confirmed_now  = False

            if (results.boxes is not None and results.boxes.id is not None):
                track_ids  = results.boxes.id.int().cpu().tolist()
                boxes_xywh = results.boxes.xywh.cpu().numpy()
                classes    = results.boxes.cls.int().cpu().tolist()

                for i, tid in enumerate(track_ids):
                    bx, by, bw, bh = boxes_xywh[i]
                    cls_id = classes[i]
                    if bw <= 0 or bh <= 0:
                        continue

                    # ---------- per-track state ----------
                    if tid not in self._states:
                        self._states[tid] = _PersonState()
                    ps = self._states[tid]
                    ps.last_frame = frame_idx

                    # check if class is falling
                    is_falling = (cls_id == fall_class_idx)
                    
                    if is_falling:
                        ps.down_frame_count += 1
                    else:
                        # recovered or standing
                        ps.down_frame_count = 0
                        if ps.state == "CONFIRMED":
                            log.debug(f"[Fall] Track {tid} recovered → NORMAL")
                        ps.state = "NORMAL"

                    if ps.down_frame_count >= confirm_frames and ps.state != "CONFIRMED":
                        ps.state = "CONFIRMED"
                        confirmed_now = True
                        log.debug(f"[Fall] Track {tid} CONFIRMED")

                    flagged = (ps.state == "CONFIRMED") or (ps.down_frame_count > 0)
                    
                    if flagged:
                        any_flagged = True
                        x1, y1 = int(bx - bw / 2), int(by - bh / 2)
                        x2, y2 = int(bx + bw / 2), int(by + bh / 2)
                        if ps.state == "CONFIRMED":
                            col, lbl = (0, 0, 255), "FALL CONFIRMED"
                        else:
                            col, lbl = (0, 165, 255), f"CHECKING... ({ps.down_frame_count}/{confirm_frames})"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 4)
                        cv2.putText(frame, lbl, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 3)

            # Evict stale tracks
            stale = [tid for tid, ps in self._states.items()
                     if frame_idx - ps.last_frame > expiry_frames]
            for tid in stale:
                del self._states[tid]

            alert_event = None

            # ---------- Alert recording ----------
            if confirmed_now and cooldown == 0 and not alert_active:
                alert_active      = True
                cooldown          = cooldown_max
                no_detect_frames  = 0
                alert_id          = str(uuid.uuid4())
                alert_start_frame = frame_idx
                alert_path        = os.path.join(output_dir, f"alert_{alert_id}.mp4")
                
                # Dynamically calculate actual processing FPS to prevent fast-forwarding
                elapsed_time = time.time() - start_time
                actual_fps = max(5.0, frame_idx / elapsed_time) if elapsed_time > 0 and frame_idx > 0 else fps
                alert_writer = cv2.VideoWriter(alert_path, cv2.VideoWriter_fourcc(*"mp4v"), actual_fps, (width, height))
                for buf_frame in frame_buffer:
                    alert_writer.write(buf_frame)
                    
                # Yield the alert JSON to the frontend IMMEDIATELY so the UI updates instantly
                ts_sec = alert_start_frame / fps
                alert_event = {
                    "id": alert_id,
                    "timestamp_sec": ts_sec,
                    "formatted_time": f"{int(ts_sec // 60):02d}:{int(ts_sec % 60):02d}",
                    "clip_url": f"/storage/alerts/alert_{alert_id}.mp4",
                    "severity": "SEVERE",
                    "immediate_buzzer_trigger": True,
                }

            elif alert_active:
                if any_flagged:
                    no_detect_frames = 0
                else:
                    no_detect_frames += 1

                if no_detect_frames >= patience_frames:
                    alert_active = False
                    if alert_writer:
                        alert_writer.release()
                        alert_writer = None

            if alert_active and alert_writer:
                cv2.putText(frame, "CRITICAL: FALL DETECTED", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                alert_writer.write(frame)

            frame_buffer.append(frame.copy())
            yield frame, alert_event
            frame_idx += 1

        # Flush any open writer at end of video file
        if alert_writer:
            alert_writer.release()
            ts_sec = alert_start_frame / fps
            yield None, {
                "id": alert_id,
                "timestamp_sec": ts_sec,
                "formatted_time": f"{int(ts_sec // 60):02d}:{int(ts_sec % 60):02d}",
                "clip_url": f"/storage/alerts/alert_{alert_id}.mp4",
                "severity": "SEVERE",
                "immediate_buzzer_trigger": True,
            }

        cap.release()
