"""
pipelines/new_fall_detection_pipeline.py
=========================================
FallDetectionPipelineV2 — Enhanced Fall Detection (Drop-in V2 upgrade)
Registered as "fall_detection_v2" in core/registry.py.

WHAT IS THIS FILE?
------------------
This is a completely self-contained, decoupled upgrade to the original
`fall_detection_pipeline.py`. It has the identical generator interface
(yields (annotated_frame, alert_event_or_None)) so it plugs into
main.py / site_admin_monitor.py without any changes to those files.

WHY IS IT BETTER THAN V1?
--------------------------
V1 triggered "FALLING" off a single signal: a raw hip-drop velocity computed
by differencing entries in a rolling deque. That produces false alarms when:
  - A person sits down quickly (hip drops fast but not a fall)
  - Pose keypoints jitter (2-3 noisy pixels per frame creates a fake spike)
  - Someone does floor exercises (lying flat looks "fallen" forever)

V2 requires FOUR independent signals to ALL agree before even entering
the "FALLING" candidate state (AND-gate, not OR-gate):
  1. hip_drop_velocity   -- Kalman-filtered (not raw-diff) hip Y velocity,
                            normalized by the person's own standing height.
  2. angular_velocity    -- Rate of change of the torso angle (deg/sec).
                            A real fall snaps from upright to flat in <500ms.
                            A sit-down takes 1-2 seconds. This is the strongest
                            sit-vs-fall discriminator.
  3. motion_energy_spike -- Local frame-difference energy in the person's bbox
                            must spike above its own recent baseline. Rejects
                            a person standing still with a jittery skeleton.
  4. track_age / conf    -- Track must be old enough and keypoint confidence
                            high enough to rule out ID-swap artifacts.

V2 also adds a Phase 3 "stillness window" before firing the alert, which
rejects floor push-ups and planks that hold a flat-torso posture but keep
moving.

FILE STRUCTURE
--------------
  ThreadedFrameReader       -- Background I/O thread that keeps camera reads
                               decoupled from slow inference (critical for RTSP).
  Kalman1D                  -- Constant-velocity Kalman filter for a single scalar
                               (used to smooth the noisy hip Y-coordinate).
  _PersonState              -- Lightweight dataclass holding all per-tracked-person
                               state (filter, angle history, state-machine phase).
  FallDetectionPipelineV2
    ._refine_keypoints()    -- Crops & upscales small/distant person bboxes for
                               better pose keypoints on far-away people.
    ._motion_energy()       -- Per-person frame-diff energy (normalized 0..1).
    .run_on_video()         -- Main generator loop: inference, state machine,
                               annotation, alert recording, yield.

ALERT EVENT SCHEMA (same as V1 + extra `signals` field)
--------------------------------------------------------
{
  "id": "<uuid>",
  "timestamp_sec": 12.4,
  "formatted_time": "00:12",
  "clip_url": "/storage/alerts/alert_<uuid>.mp4",
  "severity": "SEVERE",
  "immediate_buzzer_trigger": true,
  "track_id": 3,           # which ByteTrack ID fell (new in V2)
  "signals": {             # the four signal values that triggered (new in V2)
    "drop_vel_norm":      0.72,
    "angular_vel_deg_s":  115.3,
    "motion_energy":      0.038,
    "motion_baseline":    0.009,
    "torso_angle_deg":    68.1,
    "hip_conf":           0.81
  }
}

HOW TO SELECT THIS PIPELINE (API)
----------------------------------
When starting an analysis session, set:
  "pipeline_type": "fall_detection_v2"

The original "fall_detection" pipeline is completely unmodified.
"""

# ---------------------------------------------------------------------------
# Standard-library imports
# ---------------------------------------------------------------------------
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
from typing import Optional, Dict, Callable

# ---------------------------------------------------------------------------
# Project imports -- same as every other pipeline in this repo
# ---------------------------------------------------------------------------
from ultralytics import YOLO
from core.base_pipeline import BaseVideoPipeline
from core.video_source import get_video_source

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# COCO-17 pose keypoint indices.
# These are fixed by the COCO standard; do not change unless you switch to
# a different skeleton format (e.g. MPII, OpenPose).
# ---------------------------------------------------------------------------
KP_L_SHOULDER, KP_R_SHOULDER = 5, 6
KP_L_HIP,      KP_R_HIP      = 11, 12
KP_L_ANKLE,    KP_R_ANKLE    = 15, 16


# ===========================================================================
# COMPONENT 1: ThreadedFrameReader
# ===========================================================================
class ThreadedFrameReader:
    """
    Decouples camera I/O from model inference using a background thread.

    WHY THIS EXISTS
    ---------------
    For RTSP / IP cameras, cv2.VideoCapture.read() can block for tens of
    milliseconds waiting for the network, and the internal OpenCV buffer
    silently accumulates stale frames if your processing loop is slower than
    the camera's FPS. The result: alert clips are "fast-forwarded" relative
    to real time, and confirm_frames no longer maps to real elapsed seconds.

    This class runs cap.read() on its own daemon thread. Only the single
    *newest* frame is kept in a queue of size 1 (drop-oldest policy). The
    inference loop always gets the freshest frame, never a stale one.

    RECONNECT LOGIC
    ---------------
    If the camera drops (network hiccup), the reader thread sleeps with
    exponential back-off and reopens the source. This is invisible to the
    inference loop.

    NOTE: For local file-based video (e.g. test .mp4), use_threaded_capture
    should be False (see DEFAULT_CONFIG in FallDetectionPipelineV2). Threading
    adds no benefit and adds a tiny overhead on files.
    """

    def __init__(
        self,
        cap_factory: Callable[[], "cv2.VideoCapture"],
        max_reconnect_attempts: int = 20,
        reconnect_backoff_sec: float = 2.0,
    ):
        # cap_factory is a zero-argument callable that returns a new cap object.
        # Storing the factory (not the cap itself) lets us re-create it after
        # a disconnect without capturing a closed object in the closure.
        self._cap_factory = cap_factory
        self._cap = cap_factory()
        self._max_reconnect_attempts = max_reconnect_attempts
        self._reconnect_backoff_sec = reconnect_backoff_sec

        # Queue of size 1: producer drops old frame if consumer hasn't read yet.
        self._q: "queue.Queue" = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="fall-v2-frame-reader"
        )
        self._dropped = 0  # total frames dropped (metric for monitoring)

    def start(self) -> "ThreadedFrameReader":
        """Start the background read thread. Returns self for fluent chaining."""
        self._thread.start()
        return self

    def _loop(self):
        """Background thread body: read frames endlessly, reconnect on failure."""
        fail_count = 0
        while not self._stop_event.is_set():
            ok, frame = self._cap.read()
            if not ok:
                fail_count += 1
                log.warning(
                    "[FallV2] Frame read failed (%d/%d), will retry.",
                    fail_count, self._max_reconnect_attempts,
                )
                if fail_count > self._max_reconnect_attempts:
                    log.error("[FallV2] Too many failures -- stopping reader thread.")
                    break
                # Exponential back-off capped at 10 s
                time.sleep(min(self._reconnect_backoff_sec * fail_count, 10.0))
                try:
                    self._cap.release()
                    self._cap = self._cap_factory()
                except Exception:
                    pass
                continue

            fail_count = 0
            # Drop the old frame if inference hasn't consumed it yet
            if self._q.full():
                try:
                    self._q.get_nowait()
                    self._dropped += 1
                except queue.Empty:
                    pass
            self._q.put(frame)

        self._stop_event.set()

    def read(self, timeout: float = 2.0):
        """Blocking read of the newest frame. Returns (ok, frame)."""
        try:
            return True, self._q.get(timeout=timeout)
        except queue.Empty:
            return False, None

    # Mirror cv2.VideoCapture API so this class is interchangeable with it
    def isOpened(self) -> bool:  # noqa: N802
        return not self._stop_event.is_set() or not self._q.empty()

    def get(self, prop):
        return self._cap.get(prop)

    def stop(self):
        """Signal the background thread to stop and wait for cleanup."""
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        try:
            self._cap.release()
        except Exception:
            pass

    @property
    def dropped_frame_count(self) -> int:
        """How many frames were dropped because inference was too slow."""
        return self._dropped


# ===========================================================================
# COMPONENT 2: Kalman1D -- scalar constant-velocity Kalman filter
# ===========================================================================
class Kalman1D:
    """
    A minimal Kalman filter for tracking a single noisy scalar (hip Y).

    WHY THIS EXISTS
    ---------------
    Raw frame-to-frame differencing of a pose keypoint is one of the biggest
    sources of false-positive fall triggers. A person standing perfectly still
    can have their hip keypoint jitter +/-3 pixels per frame due to model noise.
    Over a 0.35-second window at 30 fps (~10 frames), that jitter can sum to
    30+ pixels -- which easily exceeds a naive velocity threshold.

    A constant-velocity Kalman filter gives a much smoother, physically
    meaningful velocity estimate. The filter state is:
        [position (hip_y), velocity (hip_y_dot)]
    and the measurement is just the current hip_y from the pose model.

    PARAMETERS
    ----------
    process_var : How much we trust the constant-velocity motion model.
                  Higher = more responsive to real motion, but also more
                  affected by noise.
    meas_var    : How noisy we expect the keypoint measurement to be.
                  Higher = more smoothing, but more lag on fast motion.

    Defaults (4.0, 6.0) are tuned for 30fps with COCO pose keypoint noise.
    """

    __slots__ = ("x", "v", "P", "q", "r", "initialized")

    def __init__(self, process_var: float = 4.0, meas_var: float = 6.0):
        self.x = 0.0                    # filtered position (hip Y)
        self.v = 0.0                    # filtered velocity (positive = moving DOWN the image)
        self.P = np.eye(2) * 100.0     # error covariance (high = lots of uncertainty at start)
        self.q = process_var
        self.r = meas_var
        self.initialized = False

    def reset(self, x0: float):
        """Initialize the filter at a given starting position."""
        self.x = x0
        self.v = 0.0
        self.P = np.eye(2) * 100.0
        self.initialized = True

    def update(self, z: float, dt: float):
        """
        One Kalman predict + update step.

        Parameters
        ----------
        z  : new measurement (current hip Y in pixels)
        dt : time elapsed since the last call (seconds)

        Returns
        -------
        (filtered_position, filtered_velocity)
        """
        if not self.initialized or dt <= 0:
            self.reset(z)
            return self.x, self.v

        # Predict: state transitions with constant-velocity model x(t+dt) = x(t) + v*dt
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = np.array([[dt**4 / 4, dt**3 / 2],
                      [dt**3 / 2, dt**2]]) * self.q
        state = F @ np.array([self.x, self.v])
        P_pred = F @ self.P @ F.T + Q

        # Update: observe position only (H picks out the first state element)
        H = np.array([[1.0, 0.0]])
        y_res = z - (H @ state)[0]
        S = (H @ P_pred @ H.T)[0, 0] + self.r
        K = (P_pred @ H.T) / S
        state = state + (K.flatten() * y_res)
        self.P = (np.eye(2) - K @ H) @ P_pred
        self.x, self.v = float(state[0]), float(state[1])
        return self.x, self.v


# ===========================================================================
# COMPONENT 3: Per-track state container
# ===========================================================================
@dataclass
class _PersonState:
    """
    All per-tracked-person state needed by the fall-detection state machine.

    One instance exists per active ByteTrack track ID. They are created on
    first detection and evicted after `track_expiry_sec` of no detections.

    Fields
    ------
    baseline_h        : EMA of bounding-box height while the person is upright.
                        Used to normalize distances so the logic is perspective-
                        invariant (a person 10m away has a smaller bbox but the
                        same physiology as a person 2m away).
    state             : Current state-machine phase: "NORMAL" -> "FALLING" -> "CONFIRMED"
    down_frame_count  : Consecutive frames in which the posture looks "down" (Phase 2).
    still_frame_count : Consecutive frames the person is still while down (Phase 3).
    fall_start_frame  : Frame index when Phase 1 was triggered.
    last_frame        : Frame index of the last time this track was seen.
    track_age         : Frames since first detection (guards against ID-swap artifacts).
    hip_kf            : Kalman1D instance for this person's hip Y coordinate.
    angle_history     : Rolling window of torso angles (to compute angular velocity).
    motion_history    : Rolling window of motion energy values (to compute spike ratio).
    last_signals      : Snapshot of the last computed signal values for alert metadata.
    """
    baseline_h: Optional[float] = None
    state: str = "NORMAL"        # "NORMAL" | "FALLING" | "CONFIRMED"
    down_frame_count: int = 0
    still_frame_count: int = 0
    fall_start_frame: int = 0
    last_frame: int = 0
    # last_frame_idx mirrors the C++ PersonState.last_frame_idx.
    # Stores the actual frame counter so dt can be computed accurately
    # even when frames are dropped (RTSP jitter, network hiccups).
    # -1 = "not yet seen" (same sentinel as C++).
    last_frame_idx: int = -1
    track_age: int = 0

    hip_kf: Kalman1D = field(default_factory=Kalman1D)

    # maxlen=15 gives ~0.5s history at 30fps; long enough for angular velocity
    # but short enough to react quickly to a sudden fall
    angle_history: deque = field(default_factory=lambda: deque(maxlen=15))
    motion_history: deque = field(default_factory=lambda: deque(maxlen=15))

    # Stored for alert payload and debug logs -- not used by the state machine itself
    last_signals: dict = field(default_factory=dict)


# ===========================================================================
# MAIN PIPELINE CLASS
# ===========================================================================
class FallDetectionPipelineV2(BaseVideoPipeline):
    """
    Enhanced fall detection pipeline (V2).
    Registered as "fall_detection_v2" in core/registry.py.

    Implements BaseVideoPipeline with the identical generator interface as V1,
    so it is a true drop-in replacement from the perspective of main.py and
    site_admin_monitor.py.

    See the module docstring at the top of this file for a full comparison
    of what is different vs. the original FallDetectionPipeline (V1).

    To use this pipeline, set "pipeline_type": "fall_detection_v2" when
    starting an analysis session. The original "fall_detection" is unchanged.
    """

    # -----------------------------------------------------------------------
    # Configuration defaults.
    # All values can be overridden at runtime via the `config` dict passed to
    # run_on_video(), following the same pattern as all other pipelines.
    # -----------------------------------------------------------------------
    DEFAULT_CONFIG = {
        # ── Detection ────────────────────────────────────────────────────────
        # Lower conf than V1 (was 0.70). We rely on the 4-signal AND-gate to
        # reject false positives instead of a strict detector threshold.
        "person_conf_threshold":   0.45,
        "keypoint_conf_threshold": 0.40,
        "base_imgsz":              640,  # inference image size (larger = more accurate, slower)

        # ── Far-range refinement ─────────────────────────────────────────────
        # When a person's bbox height is below small_person_px_height pixels,
        # we crop + upscale the region and re-run pose inference for better
        # keypoints. Typical for outdoor cameras covering large areas.
        "refine_small_persons":      True,
        "small_person_px_height":    130,  # pixel height threshold below which refinement triggers
        "refine_upscale_factor":     2.5,  # how much to upscale the crop
        "refine_pad_frac":           0.35, # extra context padding (fraction of bbox size)
        "max_refinements_per_frame": 4,    # cost limit when many distant people are in frame

        # ── Phase 1: fall impulse detection (AND-gate of 4 signals) ──────────
        # All of these must be True simultaneously for NORMAL -> FALLING.
        # See docstring above for explanation of each signal.
        "velocity_threshold":               0.55,   # Kalman hip-drop normalized by baseline_h
        "angular_velocity_threshold_deg_s": 90.0,   # torso-angle change rate in deg/sec
        "motion_spike_ratio":               1.8,    # motion_e must exceed baseline * this
        "motion_spike_floor":               0.01,   # minimum absolute floor for the spike check
        "sit_angle_deg":                    30.0,   # torso angle below this = upright (no fall)
        "min_track_age_frames":             5,      # ignore brand-new tracks (ID-swap guard)
        "min_keypoint_conf":                0.35,   # hip keypoint must be at least this confident

        # ── Phase 2: posture-down confirmation ───────────────────────────────
        "confirm_frames":      8,    # consecutive "down posture" frames needed to advance
        "down_torso_angle_deg": 40.0, # torso angle above this = "looks flat on the ground"
        "down_hip_ankle_ratio": 0.45, # hip-ankle gap < this * baseline_h = "hips near floor"

        # ── Phase 3: stillness confirmation ──────────────────────────────────
        # After confirm_frames of flat posture, additionally require the person
        # to stop moving before the alert fires. Rejects push-ups, planks, etc.
        "require_stillness_confirmation": True,
        "stillness_frames":              10,    # consecutive low-motion frames needed
        "stillness_motion_threshold":    0.02,  # motion_energy below this = "considered still"

        # ── Alerting / recording ─────────────────────────────────────────────
        "alert_cooldown_sec":      12.0,  # no new alert from same person within this window
        "post_alert_patience_sec":  3.0,  # how long to keep recording after last detection
        "pre_event_buffer_sec":     3.0,  # pre-fall footage seconds to include in clip
        "track_expiry_sec":        15.0,  # evict a track from _states after this many seconds

        # ── I/O ──────────────────────────────────────────────────────────────
        # Use threaded capture only for live streams (RTSP/HTTP).
        # For local test .mp4 files, threading adds overhead with no benefit.
        "use_threaded_capture": True,
    }

    # -----------------------------------------------------------------------
    # BaseVideoPipeline: initialize
    # -----------------------------------------------------------------------
    def initialize(self, model_weight: str = "yolov8n-pose.pt", **kwargs) -> None:
        """
        Load the YOLO pose model. Called once by the PipelineRegistry when
        this pipeline is first requested. The model is kept in memory and
        reused across all subsequent video sessions.

        Parameters
        ----------
        model_weight : Filename or path of the YOLO pose weights file.
                       "yolov8n-pose.pt" (nano) is the default. For better
                       accuracy at the cost of speed, use "yolov8s-pose.pt"
                       (small) or "yolov8m-pose.pt" (medium).
        """
        log.info("[FallV2] Loading model: %s", model_weight)
        self.model = YOLO(model_weight)
        # Dict of {track_id (int) -> _PersonState} for all currently tracked people
        self._states: Dict[int, _PersonState] = {}

    # -----------------------------------------------------------------------
    # BaseVideoPipeline: process_frame (stub -- not used in streaming mode)
    # -----------------------------------------------------------------------
    def process_frame(self, frame, frame_idx, roi_polygon, config):
        """
        Not used -- all processing is handled inside run_on_video().
        This stub satisfies the BaseVideoPipeline abstract method requirement.
        """
        return frame, {}

    # -----------------------------------------------------------------------
    # GEOMETRY HELPERS
    # -----------------------------------------------------------------------
    @staticmethod
    def _mid(kxy, kconf, li: int, ri: int, thresh: float):
        """
        Compute the midpoint of a left/right keypoint pair.
        Falls back gracefully to whichever side is visible above threshold.

        Parameters
        ----------
        kxy   : (17, 2) array of (x, y) coordinates for all 17 COCO keypoints
        kconf : (17,) array of confidence scores (0..1)
        li    : left keypoint index (e.g. KP_L_HIP = 11)
        ri    : right keypoint index (e.g. KP_R_HIP = 12)
        thresh: minimum confidence to treat a keypoint as "visible"

        Returns
        -------
        ((x, y), confidence) -- or (None, 0.0) if both sides are below threshold.
        """
        l_ok = kconf[li] > thresh
        r_ok = kconf[ri] > thresh
        if l_ok and r_ok:
            return (
                ((kxy[li][0] + kxy[ri][0]) / 2.0,
                 (kxy[li][1] + kxy[ri][1]) / 2.0),
                (kconf[li] + kconf[ri]) / 2.0,
            )
        if l_ok:
            return (float(kxy[li][0]), float(kxy[li][1])), float(kconf[li])
        if r_ok:
            return (float(kxy[ri][0]), float(kxy[ri][1])), float(kconf[ri])
        return None, 0.0

    @staticmethod
    def _torso_angle(hip, shoulder) -> float:
        """
        Compute the torso angle from vertical, in degrees.
          0 deg = perfectly upright (shoulder directly above hip)
          90 deg = lying perfectly flat (shoulder level with hip)

        Using atan2(horizontal_offset, vertical_offset) gives a value in
        [0, 90] that is intuitive and perspective-independent.
        """
        dx = abs(shoulder[0] - hip[0])
        dy = abs(hip[1] - shoulder[1]) + 1e-6  # +epsilon avoids divide-by-zero
        return degrees(atan2(dx, dy))

    # -----------------------------------------------------------------------
    # FAR-RANGE REFINEMENT
    # -----------------------------------------------------------------------
    def _refine_keypoints(self, frame: np.ndarray, box_xywh: np.ndarray, cfg: dict):
        """
        Re-run pose inference on an upscaled crop around a small/distant person.

        WHEN THIS IS USEFUL
        -------------------
        For outdoor IP cameras covering large areas, a person 10+ meters away
        may have a bounding box only 60-80 pixels tall. YOLO-pose keypoints
        at this scale are very noisy -- small errors in hip/shoulder position
        translate into large errors in the velocity and angle signals.

        This method crops a padded region around the person from the
        full-resolution frame (not the downscaled inference frame), upscales
        it, and re-runs pose inference on just that crop. The refined keypoints
        are then mapped back to full-frame coordinates.

        COST CONTROL
        ------------
        This is rate-limited by max_refinements_per_frame in the caller so
        that having many distant people in the scene doesn't compound the cost.

        Parameters
        ----------
        frame    : The full-resolution original frame.
        box_xywh : Person bounding box from YOLO in (cx, cy, w, h) format.
        cfg      : The merged config dict.

        Returns
        -------
        (kxy_full, kconf) with coordinates in the full-frame space,
        or None if the refinement found nothing usable.
        """
        bx, by, bw, bh = box_xywh
        h_frame, w_frame = frame.shape[:2]
        pad = cfg["refine_pad_frac"]

        # Padded crop region, clamped to frame boundaries
        x1 = max(0, int(bx - bw * (0.5 + pad)))
        y1 = max(0, int(by - bh * (0.5 + pad)))
        x2 = min(w_frame, int(bx + bw * (0.5 + pad)))
        y2 = min(h_frame, int(by + bh * (0.5 + pad)))

        if x2 - x1 < 8 or y2 - y1 < 8:
            return None  # crop too small to be useful

        crop = frame[y1:y2, x1:x2]
        scale = cfg["refine_upscale_factor"]
        crop_up = cv2.resize(crop, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_CUBIC)

        # Run single-person pose inference on the upscaled crop
        result = self.model.predict(
            crop_up,
            # Slightly looser threshold on the crop -- the person fills the frame
            conf=cfg["person_conf_threshold"] * 0.8,
            imgsz=max(crop_up.shape[:2]),
            verbose=False,
        )[0]

        if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
            return None

        # Pick the largest detection in the crop (our person, now dominating the frame)
        areas = (result.boxes.xywh[:, 2] * result.boxes.xywh[:, 3]).cpu().numpy()
        best_idx = int(np.argmax(areas))

        kxy_crop = result.keypoints.xy.cpu().numpy()[best_idx]
        kconf    = result.keypoints.conf.cpu().numpy()[best_idx]

        # Map crop-space keypoint coordinates back to full-frame coordinates
        kxy_full = kxy_crop.copy()
        kxy_full[:, 0] = kxy_full[:, 0] / scale + x1
        kxy_full[:, 1] = kxy_full[:, 1] / scale + y1

        return kxy_full, kconf

    # -----------------------------------------------------------------------
    # MOTION ENERGY
    # -----------------------------------------------------------------------
    @staticmethod
    def _motion_energy(
        prev_gray: Optional[np.ndarray],
        gray: np.ndarray,
        box_xywh: np.ndarray,
    ) -> float:
        """
        Compute normalized mean absolute frame-difference inside a person's bbox.

        Returns a value in [0.0, ~0.1+]:
          ~0.00  = completely still
          ~0.02  = very slight movement (breathing, small shifts)
          ~0.05+ = noticeable movement (walking, arms moving)
          ~0.10+ = fast motion (running, falling)

        Used for two purposes in the state machine:
          1. Phase 1: a motion energy spike (vs. recent baseline) is one of the
             four signals required to enter FALLING state.
          2. Phase 3: low motion energy after falling confirms the person is
             still on the ground (not doing push-ups or floor exercises).

        Returns 0.0 on the first frame (no prev_gray) or if the crop is too small.
        """
        if prev_gray is None:
            return 0.0

        bx, by, bw, bh = box_xywh
        h_f, w_f = gray.shape

        x1 = max(0, int(bx - bw / 2))
        y1 = max(0, int(by - bh / 2))
        x2 = min(w_f, int(bx + bw / 2))
        y2 = min(h_f, int(by + bh / 2))

        if x2 - x1 < 4 or y2 - y1 < 4:
            return 0.0

        patch_a = prev_gray[y1:y2, x1:x2]
        patch_b = gray[y1:y2, x1:x2]

        if patch_a.shape != patch_b.shape:
            return 0.0  # shape mismatch (can happen if video resolution changes mid-stream)

        diff = cv2.absdiff(patch_a, patch_b)
        return float(np.mean(diff)) / 255.0  # normalize to [0, 1]

    # -----------------------------------------------------------------------
    # MAIN LOOP
    # -----------------------------------------------------------------------
    def run_on_video(self, input_path, output_dir, roi_normalized, config):
        """
        Generator that processes a video and yields (annotated_frame, alert_event_or_None).

        This method is called by main.py / site_admin_monitor.py via the
        pipeline registry. It has the identical signature and yield contract as
        the original V1 fall_detection_pipeline.py, so it is a true drop-in.

        Parameters
        ----------
        input_path    : str -- path to a video file, or an RTSP/HTTP stream URL.
        output_dir    : str -- directory to write alert clip .mp4 files.
        roi_normalized: list -- not used for fall detection (no zone needed).
                                kept for API compatibility with other pipelines.
        config        : dict -- runtime overrides for DEFAULT_CONFIG values.

        Yields
        ------
        (annotated_frame, alert_event_or_None) on every processed frame.
        On the final yield after the stream ends, annotated_frame may be None
        if a clip was still being written (trailing alert_event is yielded).
        """
        cfg = {**self.DEFAULT_CONFIG, **(config or {})}

        # ── Set up video source ─────────────────────────────────────────────
        # ThreadedFrameReader is only beneficial for live IP camera streams.
        # For local .mp4 files, we use the plain cv2.VideoCapture path.
        is_stream = isinstance(input_path, str) and input_path.startswith(
            ("rtsp://", "http://", "https://")
        )
        if cfg["use_threaded_capture"] and is_stream:
            log.info("[FallV2] Using ThreadedFrameReader for stream: %s", input_path)
            cap = ThreadedFrameReader(lambda: get_video_source(input_path)).start()
        else:
            cap = get_video_source(input_path)

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Pre-compute frame count thresholds from the time-based config values
        expiry_frames   = int(fps * cfg["track_expiry_sec"])
        cooldown_max    = int(fps * cfg["alert_cooldown_sec"])
        patience_frames = max(1, int(fps * cfg["post_alert_patience_sec"]))
        pre_buf_frames  = max(1, int(fps * cfg["pre_event_buffer_sec"]))
        confirm_frames  = cfg["confirm_frames"]
        kp_thresh       = cfg["keypoint_conf_threshold"]
        conf_thresh     = cfg["person_conf_threshold"]

        # ── Per-session state ───────────────────────────────────────────────
        frame_idx         = 0
        cooldown          = 0           # frames remaining in alert cooldown
        alert_active      = False       # currently recording a clip?
        alert_writer      = None        # cv2.VideoWriter or None
        alert_id          = None
        alert_start_frame = 0
        no_detect_frames  = 0           # consecutive frames without a flagged person
        frame_buffer      = deque(maxlen=pre_buf_frames)  # rolling pre-event buffer
        prev_gray         = None        # previous grayscale frame (for motion energy)
        start_time        = time.time()

        # ── Main frame loop ─────────────────────────────────────────────────
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if cooldown > 0:
                cooldown -= 1

            # ── Run YOLOv8-pose with ByteTrack ───────────────────────────────
            # persist=True keeps ByteTrack IDs consistent across frames.
            # classes=[0] restricts detection to the "person" class only.
            results = self.model.track(
                frame,
                classes=[0],
                conf=conf_thresh,
                imgsz=cfg["base_imgsz"],
                persist=True,
                verbose=False,
            )[0]

            # results.plot() draws skeleton + bounding boxes (YOLO built-in)
            annotated = results.plot()
            # Grayscale frame needed for per-person motion energy computation
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ── Per-frame aggregate flags ─────────────────────────────────────
            any_flagged        = False   # any tracked person is in FALLING or CONFIRMED state
            confirmed_now      = False   # a new CONFIRMED happened this frame
            refinements_used   = 0       # counter for far-range refinement cost limit
            confirmed_track_id = None    # track ID of the person who just confirmed (for alert)
            confirmed_signals  = {}      # signal snapshot at confirmation (for alert payload)

            # ── Process each tracked person ──────────────────────────────────
            if (results.boxes is not None
                    and results.boxes.id is not None
                    and results.keypoints is not None):

                track_ids  = results.boxes.id.int().cpu().tolist()
                boxes_xywh = results.boxes.xywh.cpu().numpy()
                kpts_xy    = results.keypoints.xy.cpu().numpy()
                kpts_conf  = results.keypoints.conf.cpu().numpy()

                for i, tid in enumerate(track_ids):
                    box = boxes_xywh[i]
                    bx, by, bw, bh = box
                    if bw <= 0 or bh <= 0:
                        continue  # degenerate/invalid bbox -- skip

                    kxy   = kpts_xy[i]
                    kconf = kpts_conf[i]

                    # ── Far-range keypoint refinement ─────────────────────────
                    # For small/distant people, crop + upscale + re-infer to get
                    # better keypoints for the signal computation below.
                    if (cfg["refine_small_persons"]
                            and bh < cfg["small_person_px_height"]
                            and refinements_used < cfg["max_refinements_per_frame"]):
                        refined = self._refine_keypoints(frame, box, cfg)
                        if refined is not None:
                            kxy, kconf = refined
                            refinements_used += 1

                    # ── Extract key body landmarks ────────────────────────────
                    hip,      hip_conf = self._mid(kxy, kconf, KP_L_HIP,      KP_R_HIP,      kp_thresh)
                    shoulder, _        = self._mid(kxy, kconf, KP_L_SHOULDER,  KP_R_SHOULDER, kp_thresh)
                    ankle,    _        = self._mid(kxy, kconf, KP_L_ANKLE,     KP_R_ANKLE,    kp_thresh)

                    if hip is None:
                        # Without hip keypoints we cannot compute any meaningful signal
                        continue
                    if ankle is None:
                        # Fallback: use bottom-center of the bbox as an ankle proxy
                        ankle = (bx, by + bh / 2.0)

                    # Torso angle (None if shoulders are not visible this frame)
                    t_angle = self._torso_angle(hip, shoulder) if shoulder is not None else None

                    # ── Initialize or retrieve per-track state ────────────────
                    if tid not in self._states:
                        self._states[tid] = _PersonState(baseline_h=bh)
                    ps = self._states[tid]
                    ps.last_frame = frame_idx
                    ps.track_age += 1

                    # Update standing-height baseline (only while clearly upright).
                    # EMA with alpha=0.15 so a single crouch doesn't corrupt the baseline.
                    is_upright = t_angle is not None and t_angle < cfg["sit_angle_deg"]
                    if is_upright and ps.baseline_h:
                        ps.baseline_h = 0.85 * ps.baseline_h + 0.15 * bh
                    if ps.baseline_h is None:
                        ps.baseline_h = bh

                    # ── SIGNAL 1: Kalman-filtered hip drop velocity ───────────
                    # Compute dt from ACTUAL frame gap, not assumed 1/fps.
                    # This matches the C++ FallStateMachine::update() logic exactly:
                    #   dt = (frame_idx - ps.last_frame_idx) / fps  if seen before
                    #   dt = 1.0 / fps                              on first frame
                    # Without this, dropped RTSP frames cause the Kalman velocity
                    # to be underestimated, making the velocity gate fail to open.
                    if ps.last_frame_idx >= 0:
                        dt = (frame_idx - ps.last_frame_idx) / fps
                    else:
                        dt = 1.0 / fps
                    ps.last_frame_idx = frame_idx

                    _, hip_vel = ps.hip_kf.update(hip[1], dt)
                    # Normalize by standing height to be perspective-invariant
                    drop_vel_norm = (hip_vel / ps.baseline_h) if ps.baseline_h else 0.0

                    # ── SIGNAL 2: Torso angular velocity (deg/sec) ────────────
                    # Rate of change of torso angle over the angle_history window.
                    # A real fall snaps 60-80 degrees in <0.5s => ~120-160 deg/s.
                    # A sit-down rotates ~20-30 degrees over 1-2s => ~15-30 deg/s.
                    if t_angle is not None:
                        ps.angle_history.append(t_angle)
                    elif ps.angle_history:
                        ps.angle_history.append(ps.angle_history[-1])  # carry last known
                    else:
                        ps.angle_history.append(0.0)

                    angular_vel = 0.0
                    if len(ps.angle_history) >= 2:
                        span_sec = len(ps.angle_history) / fps
                        angular_vel = (ps.angle_history[-1] - ps.angle_history[0]) / span_sec

                    # ── SIGNAL 3: Per-person motion energy spike ──────────────
                    motion_e = self._motion_energy(prev_gray, gray, box)
                    ps.motion_history.append(motion_e)
                    motion_baseline = float(np.mean(ps.motion_history)) if ps.motion_history else 0.0

                    # ── Store signals snapshot (for alert payload + debug logs) ─
                    ps.last_signals = {
                        "drop_vel_norm":     round(drop_vel_norm, 3),
                        "angular_vel_deg_s": round(angular_vel, 1),
                        "motion_energy":     round(motion_e, 4),
                        "motion_baseline":   round(motion_baseline, 4),
                        "torso_angle_deg":   round(t_angle, 1) if t_angle is not None else None,
                        "hip_conf":          round(hip_conf, 2),
                    }

                    # ── STATE MACHINE ─────────────────────────────────────────
                    flagged = ps.state in ("FALLING", "CONFIRMED")

                    # ─────────────── PHASE 1: detect fall impulse ─────────────
                    if ps.state == "NORMAL":
                        # All four gates must open simultaneously (AND, not OR).
                        # This is the core anti-false-positive mechanism of V2.
                        velocity_ok = drop_vel_norm > cfg["velocity_threshold"]
                        angular_ok  = abs(angular_vel) > cfg["angular_velocity_threshold_deg_s"]
                        motion_ok   = motion_e > (motion_baseline * cfg["motion_spike_ratio"]
                                                   + cfg["motion_spike_floor"])
                        age_ok      = ps.track_age > cfg["min_track_age_frames"]
                        conf_ok     = hip_conf > cfg["min_keypoint_conf"]

                        if velocity_ok and angular_ok and motion_ok and age_ok and conf_ok:
                            ps.state = "FALLING"
                            ps.fall_start_frame  = frame_idx
                            ps.down_frame_count  = 0
                            ps.still_frame_count = 0
                            flagged = True
                            log.debug(
                                "[FallV2] Track %s -> FALLING  signals=%s",
                                tid, ps.last_signals,
                            )

                    # ─────────────── PHASE 2: posture-down confirmation ───────
                    elif ps.state == "FALLING":
                        # Person must maintain a "down" posture for confirm_frames
                        # consecutive frames before we advance to the next phase.
                        hip_ankle_gap  = ankle[1] - hip[1]
                        hip_near_floor = (
                            ps.baseline_h is not None
                            and hip_ankle_gap < cfg["down_hip_ankle_ratio"] * ps.baseline_h
                        )
                        torso_flat    = t_angle is not None and t_angle > cfg["down_torso_angle_deg"]
                        torso_upright = t_angle is not None and t_angle < cfg["sit_angle_deg"]

                        # "Down" = torso is flat, OR hips are near the floor but
                        # torso is not clearly upright (rules out deep squats).
                        still_down = torso_flat or (hip_near_floor and not torso_upright)

                        if still_down:
                            ps.down_frame_count += 1
                        else:
                            # Person stood back up -- likely a false alarm or a near-miss
                            ps.state = "NORMAL"
                            ps.down_frame_count  = 0
                            ps.still_frame_count = 0
                            flagged = False
                            log.debug("[FallV2] Track %s recovered -> NORMAL", tid)

                        if ps.down_frame_count >= confirm_frames:
                            flagged = True
                            if not cfg["require_stillness_confirmation"]:
                                # Phase 3 disabled: confirm immediately after posture check
                                ps.state = "CONFIRMED"
                                confirmed_now = True
                                log.debug(
                                    "[FallV2] Track %s -> CONFIRMED (stillness check skipped)",
                                    tid,
                                )
                            else:
                                # ── PHASE 3: stillness gate ───────────────────────
                                # The person is in a flat/down posture AND has held it
                                # for confirm_frames. Now we additionally require them to
                                # be STILL. This is what rejects push-ups, planks, and
                                # any deliberate floor exercise: they hold the posture but
                                # motion_energy stays elevated, so still_frame_count
                                # never accumulates.
                                if motion_e < cfg["stillness_motion_threshold"]:
                                    ps.still_frame_count += 1
                                else:
                                    # Still moving -- reset the stillness counter
                                    ps.still_frame_count = 0

                                if ps.still_frame_count >= cfg["stillness_frames"]:
                                    ps.state = "CONFIRMED"
                                    confirmed_now = True
                                    log.debug(
                                        "[FallV2] Track %s -> CONFIRMED after stillness check",
                                        tid,
                                    )

                    # ─────────────── CONFIRMED: stay until recovery ───────────
                    elif ps.state == "CONFIRMED":
                        # Keep drawing the CONFIRMED box until the person clearly
                        # recovers to an upright posture.
                        hip_ankle_gap  = ankle[1] - hip[1]
                        hip_near_floor = (
                            ps.baseline_h is not None
                            and hip_ankle_gap < cfg["down_hip_ankle_ratio"] * ps.baseline_h
                        )
                        torso_flat    = t_angle is not None and t_angle > cfg["down_torso_angle_deg"]
                        torso_upright = t_angle is not None and t_angle < cfg["sit_angle_deg"]

                        if not torso_flat and not (hip_near_floor and not torso_upright):
                            # Person has stood up -- alert state ends
                            ps.state = "NORMAL"
                            ps.down_frame_count  = 0
                            ps.still_frame_count = 0

                        flagged = ps.state == "CONFIRMED"

                    # ── Draw annotation on the frame ──────────────────────────
                    if flagged:
                        any_flagged = True
                        x1_b = int(bx - bw / 2)
                        y1_b = int(by - bh / 2)
                        x2_b = int(bx + bw / 2)
                        y2_b = int(by + bh / 2)

                        if ps.state == "CONFIRMED":
                            col, lbl = (0, 0, 255), "FALL CONFIRMED"
                        else:
                            col, lbl = (0, 165, 255), "CHECKING..."

                        cv2.rectangle(annotated, (x1_b, y1_b), (x2_b, y2_b), col, 4)
                        cv2.putText(
                            annotated, lbl, (x1_b, y1_b - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 3,
                        )

                        # Save the signal snapshot from the first confirmed track
                        if confirmed_now and confirmed_track_id is None:
                            confirmed_track_id = tid
                            confirmed_signals  = dict(ps.last_signals)

            # Save current frame for next iteration's motion energy computation
            prev_gray = gray

            # ── Evict stale tracks ────────────────────────────────────────────
            # Prevents _states from growing unboundedly as people enter/leave.
            stale = [
                tid for tid, ps in self._states.items()
                if frame_idx - ps.last_frame > expiry_frames
            ]
            for tid in stale:
                del self._states[tid]

            alert_event = None

            # ── Alert recording ───────────────────────────────────────────────
            if confirmed_now and cooldown == 0 and not alert_active:
                # New confirmed fall, not in cooldown window -- start a clip
                alert_active      = True
                cooldown          = cooldown_max
                no_detect_frames  = 0
                alert_id          = str(uuid.uuid4())
                alert_start_frame = frame_idx

                alert_path = os.path.join(output_dir, f"alert_{alert_id}.mp4")

                # Use actual processing FPS (not camera FPS) to avoid fast-forwarded
                # clips when inference is slower than the camera framerate.
                elapsed_time = time.time() - start_time
                actual_fps = (
                    max(5.0, frame_idx / elapsed_time)
                    if elapsed_time > 0 and frame_idx > 0
                    else fps
                )
                alert_writer = cv2.VideoWriter(
                    alert_path, cv2.VideoWriter_fourcc(*"mp4v"), actual_fps, (width, height)
                )

                # Include pre-event buffer (footage from the 3s before the fall)
                for buf_frame in frame_buffer:
                    alert_writer.write(buf_frame)

                # Build the alert event dict (identical to V1 schema + track_id/signals)
                ts_sec = alert_start_frame / fps
                alert_event = {
                    "id":                       alert_id,
                    "timestamp_sec":            ts_sec,
                    "formatted_time":           f"{int(ts_sec // 60):02d}:{int(ts_sec % 60):02d}",
                    "clip_url":                 f"/storage/alerts/alert_{alert_id}.mp4",
                    "severity":                 "SEVERE",
                    "immediate_buzzer_trigger": True,
                    # V2 additions: which person fell + the signal values that triggered
                    "track_id": confirmed_track_id,
                    "signals":  confirmed_signals,
                }
                log.info(
                    "[FallV2] ALERT fired (track=%s). Signals: %s",
                    confirmed_track_id, confirmed_signals,
                )

            elif alert_active:
                # While active, count frames without a flagged person.
                # Once the patience window expires, stop recording.
                if any_flagged:
                    no_detect_frames = 0
                else:
                    no_detect_frames += 1

                if no_detect_frames >= patience_frames:
                    alert_active = False
                    if alert_writer:
                        alert_writer.release()
                        alert_writer = None

            # Write annotated frame to clip while alert is active
            if alert_active and alert_writer:
                cv2.putText(
                    annotated, "CRITICAL: FALL DETECTED", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3,
                )
                alert_writer.write(annotated)

            # Roll the pre-event buffer (always maintained so it's ready for
            # the next fall event, even mid-stream)
            frame_buffer.append(annotated.copy())

            # ── Yield to caller ───────────────────────────────────────────────
            yield annotated, alert_event
            frame_idx += 1

        # ── End of stream ─────────────────────────────────────────────────────
        if alert_writer:
            alert_writer.release()
            ts_sec = alert_start_frame / fps
            # Final yield: alert_event even though frame is None (clip is done)
            yield None, {
                "id":                       alert_id,
                "timestamp_sec":            ts_sec,
                "formatted_time":           f"{int(ts_sec // 60):02d}:{int(ts_sec % 60):02d}",
                "clip_url":                 f"/storage/alerts/alert_{alert_id}.mp4",
                "severity":                 "SEVERE",
                "immediate_buzzer_trigger": True,
            }

        # Release capture (type-aware: ThreadedFrameReader vs. plain cv2.VideoCapture)
        if isinstance(cap, ThreadedFrameReader):
            log.info(
                "[FallV2] Stream ended after %d frames. Total dropped by reader: %d",
                frame_idx, cap.dropped_frame_count,
            )
            cap.stop()
        else:
            log.info("[FallV2] Video ended after %d frames.", frame_idx)
            cap.release()
