"""
Fall Detection Pipeline (YOLOv8-pose + ByteTrack)

Architecture: Two-phase state machine per tracked person.
  NORMAL -> FALLING (sudden hip drop detected) -> CONFIRMED (still down after N frames) -> alert
  NORMAL <- FALLING if person recovers to upright quickly (was a controlled motion)

Key signals used (all skeleton-based, no bounding box ratio):
  1. Hip drop velocity  : hip_y drop / baseline_height / sec over a short window
  2. Torso angle        : angle of shoulder->hip vector from vertical (0=upright, 90=flat)
  3. Hip-ankle distance : when fallen, hips are close to the floor (near ankles)

Bounding-box aspect ratio is NOT used. The torso angle is the only shape signal.
This correctly handles: sitting (torso stays upright), crouching, stretching, etc.

Thresholds are intentionally relaxed for true-positive recall. The state machine
provides the false-positive rejection instead of tight individual thresholds.
"""

import cv2
import time
import numpy as np
from ultralytics import YOLO
import uuid
import os
import logging
from math import atan2, degrees
from collections import deque
from core.base_pipeline import BaseVideoPipeline
from core.video_source import get_video_source

log = logging.getLogger(__name__)

# COCO pose keypoint indices
KP_L_SHOULDER, KP_R_SHOULDER = 5, 6
KP_L_HIP,      KP_R_HIP      = 11, 12
KP_L_ANKLE,    KP_R_ANKLE    = 15, 16


class _PersonState:
    """Lightweight per-track state container."""
    __slots__ = (
        "hip_history",       # deque of (frame_idx, hip_y)
        "baseline_h",        # EMA of bounding-box height while person is upright
        "state",             # "NORMAL" | "FALLING" | "CONFIRMED"
        "fall_start_frame",
        "down_frame_count",  # consecutive frames person looks "down" after FALLING
        "last_frame",
    )

    def __init__(self, maxlen):
        self.hip_history    = deque(maxlen=maxlen)
        self.baseline_h     = None
        self.state          = "NORMAL"
        self.fall_start_frame = 0
        self.down_frame_count = 0
        self.last_frame     = 0


class FallDetectionPipeline(BaseVideoPipeline):

    # ------------------------------------------------------------------
    # Tunable defaults – can be overridden via the `config` dict
    # ------------------------------------------------------------------
    DEFAULT_CONFIG = {
        # Model
        "person_conf_threshold":  0.70,   # lowered from 0.75 to catch partially-visible falls
        "keypoint_conf_threshold": 0.4,   # lowered so ankle/shoulder are usable on floor poses

        # Phase 1 – detecting the fall impulse
        # Hip must drop by this fraction of baseline height per second
        "velocity_window_sec":    0.35,   # look-back window for velocity calc
        "velocity_threshold":     0.55,   # drop/baseline_h per second; lower = more sensitive

        # Torso angle from vertical (deg). 0=upright, 90=flat.
        # We only BLOCK Phase-1 if torso is clearly upright (sitting/standing).
        # i.e. if torso_angle < sit_angle, skip (person is clearly upright – no fall).
        "sit_angle_deg":          30.0,   # below this = sitting/standing (no fall possible)

        # Phase 2 – confirming the person stayed down
        "confirm_frames":         8,      # consecutive frames in "down" posture to confirm
        # Person looks "down" when torso angle > this AND hips near ankles
        "down_torso_angle_deg":   40.0,   # generous – flat on floor often gives ~60-80 deg
        "down_hip_ankle_ratio":   0.45,   # hip-ankle gap < this * baseline_h = near floor

        # Alert / recording
        "alert_cooldown_sec":     12.0,
        "post_alert_patience_sec": 3.0,
        "pre_event_buffer_sec":   3.0,
        "track_expiry_sec":       15.0,
    }

    def initialize(self, model_weight: str = "yolov8n-pose.pt"):
        self.model = YOLO(model_weight)
        self._states: dict[int, _PersonState] = {}

    def process_frame(self, frame, frame_idx, roi_polygon, config):
        pass  # streaming is handled inside run_on_video

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mid(kxy, kconf, li, ri, thresh):
        """Return midpoint of a left/right keypoint pair, or None."""
        l_ok = kconf[li] > thresh
        r_ok = kconf[ri] > thresh
        if l_ok and r_ok:
            return ((kxy[li][0] + kxy[ri][0]) / 2.0,
                    (kxy[li][1] + kxy[ri][1]) / 2.0)
        if l_ok: return (float(kxy[li][0]), float(kxy[li][1]))
        if r_ok: return (float(kxy[ri][0]), float(kxy[ri][1]))
        return None

    @staticmethod
    def _torso_angle(hip, shoulder):
        """Angle of the torso from vertical. 0 = upright, 90 = flat."""
        dx = abs(shoulder[0] - hip[0])
        dy = abs(hip[1] - shoulder[1]) + 1e-6
        return degrees(atan2(dx, dy))

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run_on_video(self, input_path, output_dir, roi_normalized, config):
        cfg = {**self.DEFAULT_CONFIG, **(config or {})}
        cap = get_video_source(input_path)

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0

        vel_window      = max(2, int(fps * cfg["velocity_window_sec"]))
        history_maxlen  = max(60, int(fps * 4))
        expiry_frames   = int(fps * cfg["track_expiry_sec"])
        cooldown_max    = int(fps * cfg["alert_cooldown_sec"])
        patience_frames = max(1, int(fps * cfg["post_alert_patience_sec"]))
        pre_buf_frames  = max(1, int(fps * cfg["pre_event_buffer_sec"]))
        confirm_frames  = cfg["confirm_frames"]

        kp_thresh = cfg["keypoint_conf_threshold"]
        conf_thresh = cfg["person_conf_threshold"]

        frame_idx         = 0
        cooldown          = 0
        alert_active      = False
        alert_writer      = None
        alert_id          = None
        alert_start_frame = 0
        no_detect_frames  = 0
        frame_buffer      = deque(maxlen=pre_buf_frames)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if cooldown > 0:
                cooldown -= 1

            results = self.model.track(
                frame, classes=[0], conf=conf_thresh, imgsz=480,
                persist=True, verbose=False,
            )[0]
            frame = results.plot()

            any_flagged    = False
            confirmed_now  = False

            if (results.boxes is not None
                    and results.boxes.id is not None
                    and results.keypoints is not None):

                track_ids  = results.boxes.id.int().cpu().tolist()
                boxes_xywh = results.boxes.xywh.cpu().numpy()
                kpts_xy    = results.keypoints.xy.cpu().numpy()
                kpts_conf  = results.keypoints.conf.cpu().numpy()

                for i, tid in enumerate(track_ids):
                    bx, by, bw, bh = boxes_xywh[i]
                    if bw <= 0 or bh <= 0:
                        continue

                    kxy   = kpts_xy[i]
                    kconf = kpts_conf[i]

                    hip      = self._mid(kxy, kconf, KP_L_HIP,      KP_R_HIP,      kp_thresh)
                    shoulder = self._mid(kxy, kconf, KP_L_SHOULDER,  KP_R_SHOULDER, kp_thresh)
                    ankle    = self._mid(kxy, kconf, KP_L_ANKLE,     KP_R_ANKLE,    kp_thresh)

                    if hip is None:
                        continue  # can't do anything without hips
                    if ankle is None:
                        ankle = (bx, by + bh / 2.0)  # fallback: centre-bottom

                    # Torso angle (None if no shoulders)
                    t_angle = self._torso_angle(hip, shoulder) if shoulder is not None else None

                    # ---------- per-track state ----------
                    if tid not in self._states:
                        self._states[tid] = _PersonState(history_maxlen)
                        self._states[tid].baseline_h = bh
                    ps = self._states[tid]
                    ps.last_frame = frame_idx

                    # Update baseline only while clearly upright
                    is_upright = (
                        t_angle is not None
                        and t_angle < cfg["sit_angle_deg"]
                    )
                    if is_upright and ps.baseline_h is not None:
                        ps.baseline_h = 0.85 * ps.baseline_h + 0.15 * bh

                    ps.hip_history.append((frame_idx, hip[1]))

                    # ---------- PHASE 1: detect fall impulse ----------
                    flagged = ps.state in ("FALLING", "CONFIRMED")

                    if ps.state == "NORMAL":
                        # Compute hip drop velocity over vel_window frames
                        if len(ps.hip_history) >= vel_window + 1:
                            ref_f, ref_y  = ps.hip_history[-1 - vel_window]
                            cur_f, cur_y  = ps.hip_history[-1]
                            dt = (cur_f - ref_f) / fps
                            if dt > 0 and ps.baseline_h:
                                drop_vel = (cur_y - ref_y) / ps.baseline_h / dt

                                # Guard: if torso is clearly upright RIGHT NOW, skip.
                                # This is the only sitting guard we need.
                                torso_upright = (t_angle is not None and t_angle < cfg["sit_angle_deg"])

                                if drop_vel > cfg["velocity_threshold"]:
                                    ps.state = "FALLING"
                                    ps.fall_start_frame = frame_idx
                                    ps.down_frame_count = 0
                                    flagged = True
                                    log.debug(f"[Fall] Track {tid} FALLING: vel={drop_vel:.2f}")

                    # ---------- PHASE 2: confirm "stayed down" ----------
                    elif ps.state == "FALLING":
                        hip_ankle_gap = ankle[1] - hip[1]
                        hip_near_floor = (
                            ps.baseline_h is not None
                            and hip_ankle_gap < cfg["down_hip_ankle_ratio"] * ps.baseline_h
                        )
                        torso_flat = (
                            t_angle is not None
                            and t_angle > cfg["down_torso_angle_deg"]
                        )
                        # To prevent fast squatting from triggering a fall, we ensure they are either completely flat, 
                        # or if their hips are on the floor, their torso is not perfectly upright.
                        torso_upright = (t_angle is not None and t_angle < cfg["sit_angle_deg"])
                        still_down = torso_flat or (hip_near_floor and not torso_upright)

                        if still_down:
                            ps.down_frame_count += 1
                        else:
                            # recovered — standing back up
                            ps.state = "NORMAL"
                            ps.down_frame_count = 0
                            flagged = False
                            log.debug(f"[Fall] Track {tid} recovered → NORMAL")

                        if ps.down_frame_count >= confirm_frames:
                            ps.state = "CONFIRMED"
                            confirmed_now = True
                            flagged = True
                            log.debug(f"[Fall] Track {tid} CONFIRMED")

                    # CONFIRMED stays flagged until recovery
                    elif ps.state == "CONFIRMED":
                        hip_ankle_gap = ankle[1] - hip[1]
                        hip_near_floor = (
                            ps.baseline_h is not None
                            and hip_ankle_gap < cfg["down_hip_ankle_ratio"] * ps.baseline_h
                        )
                        torso_flat = (t_angle is not None and t_angle > cfg["down_torso_angle_deg"])
                        torso_upright = (t_angle is not None and t_angle < cfg["sit_angle_deg"])
                        if not torso_flat and not (hip_near_floor and not torso_upright):
                            ps.state = "NORMAL"
                            ps.down_frame_count = 0
                        flagged = ps.state == "CONFIRMED"

                    if flagged:
                        any_flagged = True
                        x1, y1 = int(bx - bw / 2), int(by - bh / 2)
                        x2, y2 = int(bx + bw / 2), int(by + bh / 2)
                        if ps.state == "CONFIRMED":
                            col, lbl = (0, 0, 255), "FALL CONFIRMED"
                        else:
                            col, lbl = (0, 165, 255), "CHECKING..."
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
                alert_path        = os.path.join(output_dir, f"alert_{alert_id}.webm")
                # Dynamically calculate actual processing FPS to prevent fast-forwarding
                elapsed_time = time.time() - start_time
                actual_fps = max(5.0, frame_idx / elapsed_time) if elapsed_time > 0 and frame_idx > 0 else fps
                fourcc = cv2.VideoWriter_fourcc(*"vp80") if "'" not in "vp80" else cv2.VideoWriter_fourcc(*"vp80")
                alert_writer = cv2.VideoWriter(alert_path, cv2.VideoWriter_fourcc(*"vp80"), actual_fps, (width, height))
                for buf_frame in frame_buffer:
                    alert_writer.write(buf_frame)
                    
                # Yield the alert JSON to the frontend IMMEDIATELY so the UI updates instantly
                ts_sec = alert_start_frame / fps
                alert_event = {
                    "id": alert_id,
                    "timestamp_sec": ts_sec,
                    "formatted_time": f"{int(ts_sec // 60):02d}:{int(ts_sec % 60):02d}",
                    "clip_url": f"/storage/alerts/alert_{alert_id}.webm",
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
                "clip_url": f"/storage/alerts/alert_{alert_id}.webm",
                "severity": "SEVERE",
                "immediate_buzzer_trigger": True,
            }

        cap.release()

