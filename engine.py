"""
engine.py
=========
Standalone CV inference bridge for the Edge Computing Server.

Exposes a single public function:
    process_frame(frame: np.ndarray) -> tuple[np.ndarray, dict]

Internally wraps the existing smart_vision_assistant modules:
    - ObjectDetector  (YOLO11 + ByteTrack)
    - SceneMemory     (cross-frame object tracking)
    - ReportEngine    (structured report dict builder)

Design decisions:
    - sys.path is patched at import time so the original smart_vision_assistant/
      directory is never modified.
    - All state is held in module-level singletons (detector, memory, reporter).
      This is intentional: the server runs one inference loop, not per-request.
    - Gemini verification is DISABLED for this PoC — enable via config.ENABLE_GEMINI.
    - process_frame() is NOT thread-safe by itself; the server's InferenceWorker
      thread is the only caller.

RTSP upgrade path:
    Swap the WebSocket frame input in server.py for an RTSP OpenCV capture;
    this file requires zero changes.
"""

import sys
import time
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Path bootstrap — add smart_vision_assistant/ to Python search path.
# This lets us import detector, config, scene_memory, etc. without modifying
# the original project at all.
# ──────────────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_SVA_DIR = _HERE / "smart_vision_assistant"

if not _SVA_DIR.exists():
    raise RuntimeError(
        f"smart_vision_assistant/ not found at: {_SVA_DIR}\n"
        "Ensure the folder is in the same directory as engine.py"
    )

if str(_SVA_DIR) not in sys.path:
    sys.path.insert(0, str(_SVA_DIR))

# ──────────────────────────────────────────────────────────────────────────────
# Now safe to import from the original project
# ──────────────────────────────────────────────────────────────────────────────
import config                               # smart_vision_assistant/config.py
from detector import ObjectDetector, Detection  # YOLO11 + ByteTrack
from scene_memory import SceneMemory        # Cross-frame object tracking
from report_engine import ReportEngine      # Structured report builder

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
log = logging.getLogger("engine")

# ──────────────────────────────────────────────────────────────────────────────
# Module-level singletons  (initialised once on first import)
# ──────────────────────────────────────────────────────────────────────────────
log.info("[Engine] Initialising ObjectDetector …")
_detector = ObjectDetector()

log.info("[Engine] Initialising SceneMemory …")
_memory = SceneMemory()

log.info("[Engine] Initialising ReportEngine …")
_reporter = ReportEngine(scene_memory=_memory, db=None, session_id=None)

# FPS tracking (rolling average over 30 frames for the reporter)
import collections as _collections
_fps_times: _collections.deque = _collections.deque(maxlen=30)


def _rolling_fps() -> float:
    """Compute rolling FPS from the last N frame timestamps."""
    if len(_fps_times) < 2:
        return 0.0
    elapsed = _fps_times[-1] - _fps_times[0]
    return (len(_fps_times) - 1) / elapsed if elapsed > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def process_frame(frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Run full detection + tracking + report pipeline on one frame.

    Parameters
    ----------
    frame : np.ndarray
        Raw BGR frame decoded from the WebSocket client (H x W x 3).

    Returns
    -------
    annotated_frame : np.ndarray
        A copy of `frame` with coloured bounding boxes drawn on it.
        Intended for developer-only cv2.imshow() — never sent to clients.

    report_dict : dict
        Safe, client-facing report containing only detection results:
        {
            "timestamp":     "HH:MM:SS",
            "total_objects": int,
            "new_objects":   int,
            "removed_objects": int,
            "scene_stability": float,   # 0–100
            "objects": [
                {
                    "track_id":         int,
                    "label":            str,
                    "category":         str,
                    "confidence":       float,   # 0–1
                    "duration_seconds": float,
                    "is_new":           bool,
                }
                ...
            ]
        }

        NOTE: System stats (FPS, CPU, RAM) and model info are intentionally
        EXCLUDED from this dict. They are available in the developer window only.
    """
    if frame is None or frame.size == 0:
        return frame, _empty_report()

    # ── Timestamp FPS tick ────────────────────────────────────────────────────
    _fps_times.append(time.monotonic())
    fps = _rolling_fps()

    # ── 1. Object Detection ───────────────────────────────────────────────────
    detections: list[Detection] = _detector.detect(frame, use_accurate=False)

    # ── 2. Update Scene Memory ────────────────────────────────────────────────
    # verification_cache={} because Gemini is disabled for this PoC.
    _memory.update(detections, verification_cache={})

    # ── 3. Draw bounding boxes onto a copy (DEVELOPER VIEW ONLY) ─────────────
    annotated = frame.copy()
    ObjectDetector.draw_detections(
        annotated,
        detections,
        scene_memory=_memory,
        verification_cache={},   # Gemini cache empty (Gemini disabled for PoC)
    )

    # ── 4. Build report dict via ReportEngine ─────────────────────────────────
    # _build_report_data() returns (lines, report_data_dict).
    # We call it directly to get the structured dict without printing.
    _reporter._current_fps = fps          # inject current fps
    _, full_report = _reporter._build_report_data()

    # ── 5. Strip system stats — produce client-safe report ────────────────────
    from datetime import datetime
    client_report = _build_client_report(full_report, datetime.now())

    return annotated, client_report


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_client_report(full_report: dict, now) -> dict:
    """
    Filter the full internal report to only include safe, client-facing fields.
    Deliberately omits: fps, cpu, ram_mb, gemini details, and bounding box coords.
    """
    client_objects = []
    for obj in full_report.get("objects", []):
        client_objects.append({
            "track_id":         obj.get("track_id"),
            "label":            obj.get("label", "unknown").title(),
            "category":         obj.get("category", "Other"),
            "confidence":       round(obj.get("confidence", 0.0), 2),
            "duration_seconds": round(obj.get("duration_seconds", 0.0), 1),
            "is_new":           bool(obj.get("is_new", False)),
        })

    return {
        "timestamp":       now.strftime("%H:%M:%S"),
        "total_objects":   full_report.get("total_objects", 0),
        "new_objects":     full_report.get("new_objects", 0),
        "removed_objects": full_report.get("removed_objects", 0),
        "scene_stability": round(full_report.get("scene_stability", 0.0), 1),
        "objects":         client_objects,
    }


def _empty_report() -> dict:
    """Return a valid but empty report dict when no frame is available."""
    from datetime import datetime
    return {
        "timestamp":       datetime.now().strftime("%H:%M:%S"),
        "total_objects":   0,
        "new_objects":     0,
        "removed_objects": 0,
        "scene_stability": 0.0,
        "objects":         [],
    }
