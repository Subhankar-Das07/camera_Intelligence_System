"""
Fall Detection Pipeline v2 (YOLOv8-pose + ByteTrack)
======================================================

Rework goals, in priority order:
  1. Reject false positives: sitting, crouching, bending, kneeling to garden,
     floor exercises/push-ups, someone lying down on purpose (outdoor blanket,
     stretching), ID-swap glitches right after a track appears.
  2. Keep true-positive recall high, including for people far from the
     camera where pose keypoints are small and noisy.
  3. Be debuggable in production: every alert carries the signal values that
     fired it, not just a boolean.

What changed vs. v1
--------------------
v1 triggered "FALLING" off a single signal (hip-drop velocity computed by
raw frame differencing) gated only by a static torso-angle check. That is
exactly the setup that fires on: pose jitter (a person standing still with
a few noisy keypoints can produce a spurious velocity spike), and slow
controlled sit-downs that happen to cross the angle threshold.

v2 requires FOUR independent signals to agree before it will even enter the
"FALLING" candidate state:
  - hip_drop_velocity   : Kalman-filtered (not raw-diff) vertical hip
                           velocity, normalized by the person's own
                           standing height (perspective-invariant).
  - angular_velocity     : rate of change of torso angle, not its static
                           value. A real fall snaps from upright to flat in
                           a few hundred ms; a sit-down does not. This is
                           the single strongest sit-vs-fall discriminator.
  - motion_energy_spike  : local frame-difference energy in the person's
                           bbox must spike above its own recent baseline at
                           the moment of the drop (rules out a stationary
                           but jittery skeleton triggering on nothing).
  - track_age / conf     : the track must be old enough and confident enough
                           that this isn't an ID-swap artifact.

Confirmation (Phase 2) additionally requires a stillness window after the
posture looks "down" before the alert actually fires, which is what
separates a genuine collapse from someone doing floor push-ups or a plank
that happens to hold a flat-torso posture for a few seconds.

Far-range detection
--------------------
YOLO pose keypoints degrade fast once a person's bounding box drops below
roughly 100-120px tall (typical for outdoor IP cameras covering a yard or
parking area). Instead of lowering confidence thresholds globally (which
just re-admits noise), v2 detects when a person's box is small, crops a
padded region around them from the *original* full-res frame, upscales it,
and re-runs pose estimation on just that crop. The refined keypoints are
mapped back into full-frame coordinates and replace the low-fidelity ones.
This is applied selectively (small boxes only, rate-limited) so cost stays
bounded even with several distant people in frame.

Threading
---------
`ThreadedFrameReader` decouples camera I/O from inference so a slow
network link (very common on outdoor IP cameras) can't cause inference to
fall behind and produce fast-forwarded alert clips or stale detections. It
keeps only the newest frame (bounded queue, drop-oldest) and auto-reconnects
on read failure. See the accompanying C++ files for how the same
producer/consumer + state-machine split maps onto a DeepStream/TensorRT
pipeline for multi-stream, GPU-batched production deployment.
"""

import cv2
import time
import uuid
import os
import logging
import threading
import queue
import numpy as np
from math import atan2, degrees
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Callable

from ultralytics import YOLO
from core.base_pipeline import BaseVideoPipeline
from core.video_source import get_video_source

log = logging.getLogger(__name__)

# COCO-17 pose keypoint indices
KP_L_SHOULDER, KP_R_SHOULDER = 5, 6
KP_L_HIP, KP_R_HIP = 11, 12
KP_L_ANKLE, KP_R_ANKLE = 15, 16


# =============================================================================
# Threaded, reconnecting frame source — production I/O for outdoor IP cameras
# =============================================================================
class ThreadedFrameReader:
    """
    Runs cap.read() on its own thread and exposes only the newest frame.

    Why this matters for IP cameras specifically: cv2.VideoCapture buffers
    frames internally, so if your processing loop (detection + pose + state
    machine) takes longer than the camera's frame interval, you silently
    fall behind and start processing stale frames — alert clips end up
    fast-forwarded relative to wall-clock, and "confirm_frames" no longer
    corresponds to real elapsed time. A bounded drop-oldest queue plus a
    reconnect loop for transient network drops fixes both problems.
    """

    def __init__(self, cap_factory: Callable[[], "cv2.VideoCapture"],
                 max_reconnect_attempts: int = 20, reconnect_backoff_sec: float = 2.0):
        self._cap_factory = cap_factory
        self._cap = cap_factory()
        self._max_reconnect_attempts = max_reconnect_attempts
        self._reconnect_backoff_sec = reconnect_backoff_sec
        self._q: "queue.Queue" = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="frame-reader")
        self._dropped = 0

    def start(self) -> "ThreadedFrameReader":
        self._thread.start()
        return self

    def _loop(self):
        fail_count = 0
        while not self._stop_event.is_set():
            ok, frame = self._cap.read()
            if not ok:
                fail_count += 1
                log.warning("frame read failed (%d/%d)", fail_count, self._max_reconnect_attempts)
                if fail_count > self._max_reconnect_attempts:
                    log.error("exceeded max reconnect attempts, stopping reader")
                    break
                time.sleep(min(self._reconnect_backoff_sec * fail_count, 10.0))
                try:
                    self._cap.release()
                    self._cap = self._cap_factory()
                except Exception:
                    pass
                continue
            fail_count = 0
            if self._q.full():
                try:
                    self._q.get_nowait()
                    self._dropped += 1
                except queue.Empty:
                    pass
            self._q.put(frame)
        self._stop_event.set()

    def read(self, timeout: float = 2.0):
        try:
            return True, self._q.get(timeout=timeout)
        except queue.Empty:
            return False, None

    def isOpened(self) -> bool:  # noqa: N802 - mirrors cv2.VideoCapture API
        return not self._stop_event.is_set() or not self._q.empty()

    def get(self, prop):
        return self._cap.get(prop)

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        try:
            self._cap.release()
        except Exception:
            pass

    @property
    def dropped_frame_count(self) -> int:
        return self._dropped


# =============================================================================
# Minimal constant-velocity Kalman filter for a noisy scalar (hip_y)
# =============================================================================
class Kalman1D:
    """
    Raw frame-to-frame differencing of a pose keypoint is one of the biggest
    sources of false "fast drop" triggers: keypoints jitter by a handful of
    pixels between frames even for a person standing still, and that jitter
    alone can exceed a naive velocity threshold. A constant-velocity Kalman
    filter gives a much smoother, more physically meaningful velocity
    estimate without adding meaningful lag.
    """
    __slots__ = ("x", "v", "P", "q", "r", "initialized")

    def __init__(self, process_var: float = 4.0, meas_var: float = 6.0):
        self.x = 0.0
        self.v = 0.0
        self.P = np.eye(2) * 100.0
        self.q = process_var
        self.r = meas_var
        self.initialized = False

    def reset(self, x0: float):
        self.x = x0
        self.v = 0.0
        self.P = np.eye(2) * 100.0
        self.initialized = True

    def update(self, z: float, dt: float):
        if not self.initialized or dt <= 0:
            self.reset(z)
            return self.x, self.v
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]]) * self.q
        state = F @ np.array([self.x, self.v])
        P = F @ self.P @ F.T + Q
        H = np.array([[1.0, 0.0]])
        y = z - (H @ state)[0]
        S = (H @ P @ H.T)[0, 0] + self.r
        K = (P @ H.T) / S
        state = state + (K.flatten() * y)
        self.P = (np.eye(2) - K @ H) @ P
        self.x, self.v = float(state[0]), float(state[1])
        return self.x, self.v


# =============================================================================
# Per-track state
# =============================================================================
@dataclass
class _PersonState:
    baseline_h: Optional[float] = None
    state: str = "NORMAL"                # NORMAL | FALLING | CONFIRMED
    down_frame_count: int = 0
    still_frame_count: int = 0
    fall_start_frame: int = 0
    last_frame: int = 0
    track_age: int = 0

    hip_kf: Kalman1D = field(default_factory=Kalman1D)
    angle_history: deque = field(default_factory=lambda: deque(maxlen=15))
    motion_history: deque = field(default_factory=lambda: deque(maxlen=15))
    # last computed signals, kept for alert metadata / debugging
    last_signals: dict = field(default_factory=dict)


class FallDetectionPipelineV2(BaseVideoPipeline):

    DEFAULT_CONFIG = {
        # Detection
        "person_conf_threshold": 0.45,
        "keypoint_conf_threshold": 0.4,
        "base_imgsz": 640,

        # Far-range refinement (crop + upscale + re-infer for small boxes)
        "refine_small_persons": True,
        "small_person_px_height": 130,     # bbox height below this triggers refinement
        "refine_upscale_factor": 2.5,
        "refine_pad_frac": 0.35,           # extra context padded around the crop
        "max_refinements_per_frame": 4,    # cost control when many distant people

        # Phase 1 trigger — ALL of these must agree (AND-gate, not fuzzy OR)
        "velocity_threshold": 0.55,        # normalized hip-drop vel (Kalman)
        "angular_velocity_threshold_deg_s": 90.0,   # torso-angle change rate
        "motion_spike_ratio": 1.8,         # spike must exceed local baseline * this
        "motion_spike_floor": 0.01,
        "sit_angle_deg": 30.0,
        "min_track_age_frames": 5,
        "min_keypoint_conf": 0.35,

        # Phase 2 — posture-down confirmation
        "confirm_frames": 8,
        "down_torso_angle_deg": 40.0,
        "down_hip_ankle_ratio": 0.45,

        # Phase 3 — stillness confirmation (filters push-ups / floor exercise)
        "require_stillness_confirmation": True,
        "stillness_frames": 10,
        "stillness_motion_threshold": 0.02,

        # Alerting / recording
        "alert_cooldown_sec": 12.0,
        "post_alert_patience_sec": 3.0,
        "pre_event_buffer_sec": 3.0,
        "track_expiry_sec": 15.0,

        # I/O
        "use_threaded_capture": True,
    }

    def initialize(self, model_weight: str = "yolov8n-pose.pt"):
        self.model = YOLO(model_weight)
        self._states: dict[int, _PersonState] = {}

    def process_frame(self, frame, frame_idx, roi_polygon, config):
        pass  # streaming handled inside run_on_video

    # -------------------------------------------------------------------
    # Geometry helpers
    # -------------------------------------------------------------------
    @staticmethod
    def _mid(kxy, kconf, li, ri, thresh):
        l_ok, r_ok = kconf[li] > thresh, kconf[ri] > thresh
        if l_ok and r_ok:
            return ((kxy[li][0] + kxy[ri][0]) / 2.0,
                     (kxy[li][1] + kxy[ri][1]) / 2.0), (kconf[li] + kconf[ri]) / 2.0
        if l_ok:
            return (float(kxy[li][0]), float(kxy[li][1])), float(kconf[li])
        if r_ok:
            return (float(kxy[ri][0]), float(kxy[ri][1])), float(kconf[ri])
        return None, 0.0

    @staticmethod
    def _torso_angle(hip, shoulder):
        dx = abs(shoulder[0] - hip[0])
        dy = abs(hip[1] - shoulder[1]) + 1e-6
        return degrees(atan2(dx, dy))

    # -------------------------------------------------------------------
    # Far-range refinement: crop around a small person, upscale, re-infer
    # -------------------------------------------------------------------
    def _refine_keypoints(self, frame, box_xywh, cfg):
        """Runs single-person pose inference on an upscaled crop.

        Returns (kxy, kconf) in ORIGINAL frame coordinates, or None if the
        refinement pass found nothing usable (falls back to the original
        low-fidelity detection upstream).
        """
        bx, by, bw, bh = box_xywh
        h, w = frame.shape[:2]
        pad = cfg["refine_pad_frac"]
        x1 = max(0, int(bx - bw * (0.5 + pad)))
        y1 = max(0, int(by - bh * (0.5 + pad)))
        x2 = min(w, int(bx + bw * (0.5 + pad)))
        y2 = min(h, int(by + bh * (0.5 + pad)))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None

        crop = frame[y1:y2, x1:x2]
        scale = cfg["refine_upscale_factor"]
        crop_up = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        result = self.model.predict(crop_up, conf=cfg["person_conf_threshold"] * 0.8,
                                      imgsz=max(crop_up.shape[:2]), verbose=False)[0]
        if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
            return None

        # Pick the detection with the largest box (should be our person, now
        # dominating the crop) rather than any incidental neighbor.
        areas = (result.boxes.xywh[:, 2] * result.boxes.xywh[:, 3]).cpu().numpy()
        best = int(np.argmax(areas))

        kxy_crop = result.keypoints.xy.cpu().numpy()[best]
        kconf = result.keypoints.conf.cpu().numpy()[best]

        # Map crop-space keypoints back to full-frame coordinates
        kxy_full = kxy_crop.copy()
        kxy_full[:, 0] = kxy_full[:, 0] / scale + x1
        kxy_full[:, 1] = kxy_full[:, 1] / scale + y1
        return kxy_full, kconf

    # -------------------------------------------------------------------
    # Motion energy: mean abs frame-diff inside a bbox, normalized 0..~1
    # -------------------------------------------------------------------
    @staticmethod
    def _motion_energy(prev_gray_full, gray_full, box_xywh):
        if prev_gray_full is None:
            return 0.0
        bx, by, bw, bh = box_xywh
        h, w = gray_full.shape
        x1, y1 = max(0, int(bx - bw / 2)), max(0, int(by - bh / 2))
        x2, y2 = min(w, int(bx + bw / 2)), min(h, int(by + bh / 2))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return 0.0
        patch_a = prev_gray_full[y1:y2, x1:x2]
        patch_b = gray_full[y1:y2, x1:x2]
        if patch_a.shape != patch_b.shape:
            return 0.0
        diff = cv2.absdiff(patch_a, patch_b)
        return float(np.mean(diff)) / 255.0

    # -------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------
    def run_on_video(self, input_path, output_dir, roi_normalized, config):
        cfg = {**self.DEFAULT_CONFIG, **(config or {})}

        is_stream = isinstance(input_path, str) and input_path.startswith(("rtsp://", "http://", "https://"))
        if cfg["use_threaded_capture"] and is_stream:
            cap = ThreadedFrameReader(lambda: get_video_source(input_path)).start()
        else:
            cap = get_video_source(input_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        expiry_frames = int(fps * cfg["track_expiry_sec"])
        cooldown_max = int(fps * cfg["alert_cooldown_sec"])
        patience_frames = max(1, int(fps * cfg["post_alert_patience_sec"]))
        pre_buf_frames = max(1, int(fps * cfg["pre_event_buffer_sec"]))
        confirm_frames = cfg["confirm_frames"]

        kp_thresh = cfg["keypoint_conf_threshold"]
        conf_thresh = cfg["person_conf_threshold"]

        frame_idx = 0
        cooldown = 0
        alert_active = False
        alert_writer = None
        alert_id = None
        alert_start_frame = 0
        no_detect_frames = 0
        frame_buffer = deque(maxlen=pre_buf_frames)
        prev_gray = None
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if cooldown > 0:
                cooldown -= 1

            results = self.model.track(
                frame, classes=[0], conf=conf_thresh, imgsz=cfg["base_imgsz"],
                persist=True, verbose=False,
            )[0]
            annotated = results.plot()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            any_flagged = False
            confirmed_now = False
            refinements_used = 0
            confirmed_track_id = None
            confirmed_signals = {}

            if (results.boxes is not None and results.boxes.id is not None
                    and results.keypoints is not None):

                track_ids = results.boxes.id.int().cpu().tolist()
                boxes_xywh = results.boxes.xywh.cpu().numpy()
                kpts_xy = results.keypoints.xy.cpu().numpy()
                kpts_conf = results.keypoints.conf.cpu().numpy()

                for i, tid in enumerate(track_ids):
                    box = boxes_xywh[i]
                    bx, by, bw, bh = box
                    if bw <= 0 or bh <= 0:
                        continue

                    kxy, kconf = kpts_xy[i], kpts_conf[i]

                    # ---- far-range refinement for small/distant people ----
                    if (cfg["refine_small_persons"] and bh < cfg["small_person_px_height"]
                            and refinements_used < cfg["max_refinements_per_frame"]):
                        refined = self._refine_keypoints(frame, box, cfg)
                        if refined is not None:
                            kxy, kconf = refined
                            refinements_used += 1

                    hip, hip_conf = self._mid(kxy, kconf, KP_L_HIP, KP_R_HIP, kp_thresh)
                    shoulder, _ = self._mid(kxy, kconf, KP_L_SHOULDER, KP_R_SHOULDER, kp_thresh)
                    ankle, _ = self._mid(kxy, kconf, KP_L_ANKLE, KP_R_ANKLE, kp_thresh)
                    if hip is None:
                        continue
                    if ankle is None:
                        ankle = (bx, by + bh / 2.0)

                    t_angle = self._torso_angle(hip, shoulder) if shoulder is not None else None

                    if tid not in self._states:
                        self._states[tid] = _PersonState(baseline_h=bh)
                    ps = self._states[tid]
                    ps.last_frame = frame_idx
                    ps.track_age += 1

                    is_upright = t_angle is not None and t_angle < cfg["sit_angle_deg"]
                    if is_upright and ps.baseline_h:
                        ps.baseline_h = 0.85 * ps.baseline_h + 0.15 * bh
                    if ps.baseline_h is None:
                        ps.baseline_h = bh

                    dt = 1.0 / fps
                    _, hip_vel = ps.hip_kf.update(hip[1], dt)
                    drop_vel_norm = (hip_vel / ps.baseline_h) if ps.baseline_h else 0.0

                    ps.angle_history.append(t_angle if t_angle is not None else ps.angle_history[-1] if ps.angle_history else 0.0)
                    angular_vel = 0.0
                    if len(ps.angle_history) >= 2:
                        span_sec = len(ps.angle_history) / fps
                        angular_vel = (ps.angle_history[-1] - ps.angle_history[0]) / span_sec

                    motion_e = self._motion_energy(prev_gray, gray, box)
                    ps.motion_history.append(motion_e)
                    motion_baseline = float(np.mean(ps.motion_history)) if ps.motion_history else 0.0

                    ps.last_signals = {
                        "drop_vel_norm": round(drop_vel_norm, 3),
                        "angular_vel_deg_s": round(angular_vel, 1),
                        "motion_energy": round(motion_e, 4),
                        "motion_baseline": round(motion_baseline, 4),
                        "torso_angle_deg": None if t_angle is None else round(t_angle, 1),
                        "hip_conf": round(hip_conf, 2),
                    }

                    flagged = ps.state in ("FALLING", "CONFIRMED")

                    # ---------------- PHASE 1: candidate fall impulse ----------------
                    if ps.state == "NORMAL":
                        velocity_ok = drop_vel_norm > cfg["velocity_threshold"]
                        angular_ok = abs(angular_vel) > cfg["angular_velocity_threshold_deg_s"]
                        motion_ok = motion_e > (motion_baseline * cfg["motion_spike_ratio"] + cfg["motion_spike_floor"])
                        age_ok = ps.track_age > cfg["min_track_age_frames"]
                        conf_ok = hip_conf > cfg["min_keypoint_conf"]

                        if velocity_ok and angular_ok and motion_ok and age_ok and conf_ok:
                            ps.state = "FALLING"
                            ps.fall_start_frame = frame_idx
                            ps.down_frame_count = 0
                            ps.still_frame_count = 0
                            flagged = True
                            log.debug("[Fall] track %s -> FALLING signals=%s", tid, ps.last_signals)

                    # ---------------- PHASE 2: posture-down confirmation ----------------
                    elif ps.state == "FALLING":
                        hip_ankle_gap = ankle[1] - hip[1]
                        hip_near_floor = ps.baseline_h is not None and hip_ankle_gap < cfg["down_hip_ankle_ratio"] * ps.baseline_h
                        torso_flat = t_angle is not None and t_angle > cfg["down_torso_angle_deg"]
                        torso_upright = t_angle is not None and t_angle < cfg["sit_angle_deg"]
                        still_down = torso_flat or (hip_near_floor and not torso_upright)

                        if still_down:
                            ps.down_frame_count += 1
                        else:
                            ps.state = "NORMAL"
                            ps.down_frame_count = 0
                            ps.still_frame_count = 0
                            flagged = False
                            log.debug("[Fall] track %s recovered -> NORMAL", tid)

                        if ps.down_frame_count >= confirm_frames:
                            flagged = True
                            if not cfg["require_stillness_confirmation"]:
                                ps.state = "CONFIRMED"
                                confirmed_now = True
                                log.debug("[Fall] track %s -> CONFIRMED (no stillness check)", tid)
                            else:
                                # PHASE 3: require a low-motion window before firing.
                                # This is what rejects push-ups / floor exercise that
                                # hold a flat-torso posture but keep moving.
                                if motion_e < cfg["stillness_motion_threshold"]:
                                    ps.still_frame_count += 1
                                else:
                                    ps.still_frame_count = 0
                                if ps.still_frame_count >= cfg["stillness_frames"]:
                                    ps.state = "CONFIRMED"
                                    confirmed_now = True
                                    log.debug("[Fall] track %s -> CONFIRMED after stillness", tid)

                    # ---------------- stay CONFIRMED until real recovery ----------------
                    elif ps.state == "CONFIRMED":
                        hip_ankle_gap = ankle[1] - hip[1]
                        hip_near_floor = ps.baseline_h is not None and hip_ankle_gap < cfg["down_hip_ankle_ratio"] * ps.baseline_h
                        torso_flat = t_angle is not None and t_angle > cfg["down_torso_angle_deg"]
                        torso_upright = t_angle is not None and t_angle < cfg["sit_angle_deg"]
                        if not torso_flat and not (hip_near_floor and not torso_upright):
                            ps.state = "NORMAL"
                            ps.down_frame_count = 0
                            ps.still_frame_count = 0
                        flagged = ps.state == "CONFIRMED"

                    if flagged:
                        any_flagged = True
                        x1, y1 = int(bx - bw / 2), int(by - bh / 2)
                        x2, y2 = int(bx + bw / 2), int(by + bh / 2)
                        col, lbl = ((0, 0, 255), "FALL CONFIRMED") if ps.state == "CONFIRMED" else ((0, 165, 255), "CHECKING...")
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), col, 4)
                        cv2.putText(annotated, lbl, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 3)
                        if confirmed_now and confirmed_track_id is None:
                            confirmed_track_id = tid
                            confirmed_signals = dict(ps.last_signals)

            prev_gray = gray

            # Evict stale tracks
            stale = [tid for tid, ps in self._states.items() if frame_idx - ps.last_frame > expiry_frames]
            for tid in stale:
                del self._states[tid]

            alert_event = None

            # ---------------- Alert recording ----------------
            if confirmed_now and cooldown == 0 and not alert_active:
                alert_active = True
                cooldown = cooldown_max
                no_detect_frames = 0
                alert_id = str(uuid.uuid4())
                alert_start_frame = frame_idx
                alert_path = os.path.join(output_dir, f"alert_{alert_id}.webm")

                elapsed_time = time.time() - start_time
                actual_fps = max(5.0, frame_idx / elapsed_time) if elapsed_time > 0 and frame_idx > 0 else fps
                alert_writer = cv2.VideoWriter(alert_path, cv2.VideoWriter_fourcc(*"vp80"), actual_fps, (width, height))
                for buf_frame in frame_buffer:
                    alert_writer.write(buf_frame)

                ts_sec = alert_start_frame / fps
                alert_event = {
                    "id": alert_id,
                    "timestamp_sec": ts_sec,
                    "formatted_time": f"{int(ts_sec // 60):02d}:{int(ts_sec % 60):02d}",
                    "clip_url": f"/storage/alerts/alert_{alert_id}.webm",
                    "severity": "SEVERE",
                    "immediate_buzzer_trigger": True,
                    "track_id": confirmed_track_id,
                    "signals": confirmed_signals,
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
                cv2.putText(annotated, "CRITICAL: FALL DETECTED", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                alert_writer.write(annotated)

            frame_buffer.append(annotated.copy())
            yield annotated, alert_event
            frame_idx += 1

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

        if isinstance(cap, ThreadedFrameReader):
            cap.stop()
        else:
            cap.release()
