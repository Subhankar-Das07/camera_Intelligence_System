"""
main.py
=======
Smart Vision Assistant — Visual Scene Intelligence Engine.

Architecture:
  - CameraThread   : Background frame grabber (lock-protected)
  - ObjectDetector : YOLO11m + BoT-SORT inference
  - SceneMemory    : Persistent object tracking with categories & lifetimes
  - ReportEngine   : Periodic structured console reports
  - GeminiVerifier : Async background label refinement (budget-controlled)
  - Adaptive frame skipping: protects FPS under CPU load

Terminal output is the primary interface. Voice/TTS has been removed.

Controls:
  Q / ESC   — Quit
  R         — Force immediate scene report to console
  D         — Toggle bounding box overlays on/off
  V         — Force Gemini verification of largest visible object
"""

import collections
import logging
import sys
import threading
import time
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

import config
from database_manager import DatabaseManager
from detector import Detection, ObjectDetector
from scene_memory import SceneMemory
from report_engine import ReportEngine
from gemini_verifier import GeminiVerifier
from event_engine import EventEngine
from relationship_engine import RelationshipEngine

# ──────────────────────────────────────────────────────────────────────────────
# Logging — console output is the primary interface
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")


# ──────────────────────────────────────────────────────────────────────────────
# FPS Tracker
# ──────────────────────────────────────────────────────────────────────────────

class FPSTracker:
    """Rolling-average FPS calculator using a circular time window (deque O(1))."""

    def __init__(self, window: int = 30):
        # deque with maxlen auto-evicts the oldest entry — no pop(0) O(N) shift
        self._times: collections.deque = collections.deque(maxlen=window)

    def tick(self) -> float:
        self._times.append(time.monotonic())
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Camera Thread — lock-protected frame grab
# ──────────────────────────────────────────────────────────────────────────────

class CameraThread:
    """
    Continuously grabs frames in a background thread.
    Uses a double-buffer approach: the camera thread writes to _write_frame;
    read() atomically swaps the pointer — no full memcpy under the lock.
    """

    def __init__(self, src: int = config.CAMERA_INDEX):
        log.info(
            "Opening webcam %d at %dx%d (threaded grab)", src,
            config.FRAME_WIDTH, config.FRAME_HEIGHT,
        )
        # Try DirectShow backend first (Windows, lower latency)
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Camera {src} is unavailable.\n"
                "Diagnosis: Another app (Teams/Zoom) is using it, or OS permissions block Python.\n"
                "Fix: Close all video apps, or change CAMERA_INDEX in config.py."
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimise capture buffer lag

        self._lock = threading.Lock()
        # Double-buffer: camera thread writes _write_frame, read() swaps pointer
        self._write_frame: Optional[np.ndarray] = None
        self._read_frame: Optional[np.ndarray] = None
        self._new_frame_available: bool = False
        self._ret: bool = False
        self._running: bool = True

        # Prime the first frame before starting the thread
        self._ret, self._write_frame = self.cap.read()
        self._read_frame = self._write_frame

        self._thread = threading.Thread(target=self._update, daemon=True, name="CameraGrab")
        self._thread.start()

    def _update(self) -> None:
        while self._running:
            ret, frame = self.cap.read()
            with self._lock:
                self._ret = ret
                if ret:
                    # Swap write buffer — main thread sees the new pointer on next read()
                    self._write_frame = frame
                    self._new_frame_available = True

    def read(self) -> tuple:
        """Return (ret, frame) — pointer swap under lock, no memcpy."""
        with self._lock:
            if self._write_frame is None:
                return False, None
            if self._new_frame_available:
                # Swap buffers: detach current read buffer, point to latest write
                self._read_frame = self._write_frame
                self._new_frame_available = False
            return self._ret, self._read_frame

    def release(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)
        self.cap.release()


# ──────────────────────────────────────────────────────────────────────────────
# HUD Rendering
# ──────────────────────────────────────────────────────────────────────────────

def _shadow_text(frame, text, org, scale, colour, thickness=1):
    """Draw text with a 1px black shadow for readability on any background."""
    x, y = org
    cv2.putText(frame, text, (x + 1, y + 1),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, org,
                cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness, cv2.LINE_AA)


def draw_hud(frame, fps, det_count, overlays_on, gemini_budget, h, w):
    """Render the heads-up display overlay onto the frame."""
    fps_colour = config.COLOUR_WARNING if fps < config.FPS_WARN_THRESHOLD else config.COLOUR_FPS
    _shadow_text(frame, f"FPS: {fps:5.1f}", (10, 28), 0.65, fps_colour)
    _shadow_text(frame, f"Objects: {det_count}", (10, 52), 0.60, config.COLOUR_TEXT)
    _shadow_text(frame, f"Gemini budget: {gemini_budget}", (10, 72), 0.50, config.COLOUR_TEXT)

    if not overlays_on:
        _shadow_text(frame, "BOXES OFF", (w - 125, 28), 0.55, config.COLOUR_WARNING)

    legend = "Q:Quit  R:Report  D:Boxes  V:Verify"
    _shadow_text(frame, legend, (10, h - 12), 0.45, (160, 160, 160))


# ──────────────────────────────────────────────────────────────────────────────
# Main Application
# ──────────────────────────────────────────────────────────────────────────────

class SmartVisionAssistant:
    """
    Visual Scene Intelligence Engine.

    Orchestrates detection, tracking, scene memory, reporting, and
    optional Gemini verification. No voice/TTS features.
    """

    def __init__(self):
        log.info("=" * 60)
        log.info("  Smart Vision Assistant — Visual Scene Intelligence Engine")
        log.info("=" * 60)

        self._show_overlays: bool = config.SHOW_OVERLAYS_DEFAULT
        self._running: bool = False
        self._frame_counter: int = 0
        self._last_detections: List[Detection] = []
        self._force_verify_next: bool = False
        self._total_objects_seen: int = 0   # Distinct track IDs seen this session
        self._seen_track_ids: set = set()   # Track IDs already counted

        # Core subsystems
        self._detector = ObjectDetector()
        self._scene_memory = SceneMemory()
        self._event_engine = EventEngine(stationary_time=30.0, abandoned_time=30.0)
        self._relationship_engine = RelationshipEngine()
        # Database
        self._db = DatabaseManager()
        self._session_id = self._db.start_session()
        self._total_events = 0

        self._verification_cache: dict = {}
        self._gemini = GeminiVerifier(
            self._verification_cache,
            db_manager=self._db,
            session_id=self._session_id,
        )

        # Reporter (pass DB so reports are persisted)
        self._reporter = ReportEngine(
            self._scene_memory,
            db=self._db,
            session_id=self._session_id,
        )
        self._fps = FPSTracker()
        self._camera: Optional[CameraThread] = None

    # ──────────────────────────────────────────────────────────────────────────
    # Keyboard Handling
    # ──────────────────────────────────────────────────────────────────────────

    def _handle_key(self, key: int) -> bool:
        """Process a keypress. Returns False to signal quit."""
        ch = chr(key & 0xFF).upper() if 0 <= key < 256 else ""

        if ch == "Q" or key == 27:   # Q or ESC
            return False

        elif ch == "R":
            # Force an immediate scene report
            log.info("[Key R] Forcing immediate scene report.")
            self._reporter.tick.__func__  # ensure tick is available
            self._reporter._print_report()  # force print

        elif ch == "D":
            self._show_overlays = not self._show_overlays
            log.info("[Key D] Bounding boxes: %s", "ON" if self._show_overlays else "OFF")

        elif ch == "V":
            self._force_verify_next = True
            log.info("[Key V] Manual Gemini verification requested.")

        return True

    # ──────────────────────────────────────────────────────────────────────────
    # Gemini Verification Logic
    # ──────────────────────────────────────────────────────────────────────────

    def _handle_verification(
        self, detections: List[Detection], frame: np.ndarray, fps: float
    ) -> None:
        """
        Decide whether to send any detections to Gemini.

        Two modes run simultaneously:
          1. Label verification: refine low-confidence YOLO labels (short prompt)
          2. Deep analysis (DEEP_ANALYSIS_ENABLED): request 3-bullet description
             for every tracked object with sufficient age
        """
        h, w = frame.shape[:2]
        auto_verify_ok = getattr(config, 'DEBUG_BYPASS_FPS_GATING', False) or (fps >= config.FPS_CRITICAL_THRESHOLD)

        largest_det = None
        largest_area = 0

        for det in detections:
            tid = det.track_id
            if tid is None:
                continue

            area = (det.bbox[2] - det.bbox[0]) * (det.bbox[3] - det.bbox[1])

            # Track largest object for manual verify (V key)
            if area > largest_area:
                largest_area = area
                largest_det = det

            # ── Check Eligibility ────────────────────────────────────────────────
            rec = self._scene_memory.get_record(tid)
            if not rec:
                continue

            eligible_for_label = (
                not self._force_verify_next
                and det.confidence < config.GEMINI_VERIFY_THRESHOLD
                and self._scene_memory.should_verify(tid)
                and tid not in self._verification_cache
                and getattr(rec, 'is_stationary', False)
            )

            cached = self._verification_cache.get(tid)
            has_description = cached and cached.get("description")
            has_error = cached and cached.get("skipped_reason")
            eligible_for_desc = (
                config.DEEP_ANALYSIS_ENABLED
                and self._scene_memory.should_verify(tid)
                and not has_description
                and not has_error
            )

            if eligible_for_label or eligible_for_desc:
                if not auto_verify_ok:
                    rec.gemini_skipped_reason = "LOW_FPS"
                elif self._gemini.budget_remaining <= 0:
                    rec.gemini_skipped_reason = "BUDGET_EXCEEDED"
                else:
                    rec.gemini_skipped_reason = None
                    crop = self._crop_detection(frame, det, w, h)
                    if crop is not None:
                        success = True
                        if eligible_for_label:
                            age = self._scene_memory.get_object_duration(tid)
                            success = self._gemini.enqueue_verification(tid, det.label, crop, age)
                        elif eligible_for_desc:
                            display = cached["label"] if cached and cached.get("label") else det.label
                            success = self._gemini.request_object_description(tid, display, crop)
                        
                        if not success and self._gemini.is_available:
                            rec.gemini_skipped_reason = "QUEUE_FULL"
                        elif success:
                            log.info("[Gemini] Description queued for track #%d", tid)

        # ── Manual verify: send largest visible object (V key) ───────────────
        if self._force_verify_next and largest_det is not None:
            crop = self._crop_detection(frame, largest_det, w, h)
            if crop is not None:
                age = self._scene_memory.get_object_duration(largest_det.track_id)
                if self._gemini.enqueue_verification(
                    largest_det.track_id, largest_det.label, crop, age
                ):
                    log.info(
                        "[Verify] Queued manual verification for '%s' #%d",
                        largest_det.label, largest_det.track_id,
                    )
        self._force_verify_next = False

    def _crop_detection(
        self, frame: np.ndarray, det: Detection, w: int, h: int
    ) -> Optional[Image.Image]:
        """Extract and convert a detection crop to PIL Image for Gemini."""
        x1, y1, x2, y2 = det.bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop_bgr = frame[y1:y2, x1:x2]
        return Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))

    # ──────────────────────────────────────────────────────────────────────────
    # Main Loop
    # ──────────────────────────────────────────────────────────────────────────

    def run(self):
        self._camera = CameraThread(config.CAMERA_INDEX)
        self._running = True

        log.info("System ready. Press R for report, D for boxes, V for verify, Q to quit.")
        log.info("First scene report will print in %.0f seconds.", config.REPORT_INTERVAL)

        try:
            while self._running:
                ret, frame = self._camera.read()
                if not ret or frame is None:
                    log.warning("Invalid frame — retrying...")
                    time.sleep(0.02)
                    continue

                h, w = frame.shape[:2]
                self._frame_counter += 1

                # ── FPS Tracking ───────────────────────────────────────────────
                current_fps = self._fps.tick()

                # ── Adaptive Frame Skipping ────────────────────────────────────
                # When FPS is low, only run YOLO on every Nth frame.
                # Use the last known detections for skipped frames.
                run_inference = True
                if (
                    current_fps > 0
                    and current_fps < config.FRAME_SKIP_FPS_THRESHOLD
                    and self._frame_counter % config.FRAME_SKIP_N != 0
                ):
                    run_inference = False

                # ── Detection ─────────────────────────────────────────────────
                if run_inference:
                    detections = self._detector.detect(frame)
                    self._last_detections = detections
                else:
                    detections = self._last_detections  # Reuse previous frame's results

                # ── Scene Memory Update ────────────────────────────────────────
                new_ids, removed_ids = self._scene_memory.update(
                    detections, self._verification_cache
                )

                # Log new arrivals to console AND database immediately
                for tid in new_ids:
                    rec = self._scene_memory.get_record(tid)
                    if rec:
                        log.info(
                            "[Scene] ▶ New: #%d '%s' (%s)", tid, rec.display_label, rec.category
                        )
                        # Log to DB
                        self._db.on_object_new(
                            session_id=self._session_id,
                            track_id=tid,
                            yolo_label=rec.yolo_label,
                            display_label=rec.display_label,
                            category=rec.category,
                            confidence=rec.confidence,
                            bbox=rec.bbox,
                        )
                        self._total_events += 1
                        # Count unique objects seen this session
                        if tid not in self._seen_track_ids:
                            self._seen_track_ids.add(tid)
                            self._total_objects_seen += 1

                # Log removals to console AND database immediately
                # Use direct dict lookup (O(1)) instead of linear list scan
                removed_dict = self._scene_memory.get_recently_removed_dict()
                for tid in removed_ids:
                    rec = removed_dict.get(tid)
                    if rec:
                        log.info(
                            "[Scene] ◀ Removed: #%d '%s' — was in scene for %s",
                            tid, rec.display_label, rec.duration_str(),
                        )
                        # Log to DB
                        self._db.on_object_removed(
                            session_id=self._session_id,
                            track_id=tid,
                            label=rec.display_label,
                            category=rec.category,
                            confidence=rec.confidence,
                            duration_seconds=rec.duration,
                        )
                        self._total_events += 1

                # ── Event Engine Update ────────────────────────────────────────
                active_records = self._scene_memory.get_active_records()
                events = self._event_engine.process(active_records, time.monotonic())
                for tid, event_type, desc in events:
                    rec = self._scene_memory.get_record(tid)
                    if rec:
                        log.info(
                            "[Event] ⚡ #%d '%s': %s", tid, rec.display_label, desc
                        )
                        self._db.on_object_event(
                            session_id=self._session_id,
                            track_id=tid,
                            event_type=event_type,
                            label=rec.display_label,
                            category=rec.category,
                            confidence=rec.confidence,
                            detail=desc,
                        )
                        self._total_events += 1

                # ── Relationship Engine Update ─────────────────────────────────
                rel_events = self._relationship_engine.process(active_records)
                for tid_a, tid_b, event_type, desc in rel_events:
                    log.info("[Rel] 🔗 %s", desc)
                    rec_a = self._scene_memory.get_record(tid_a)
                    if rec_a:
                        self._db.on_object_event(
                            session_id=self._session_id,
                            track_id=tid_a,
                            event_type=event_type,
                            label=rec_a.display_label,
                            category=rec_a.category,
                            confidence=rec_a.confidence,
                            detail=desc,
                        )
                        self._total_events += 1

                # ── Gemini Verification ────────────────────────────────────────
                if run_inference:  # Only consider verification on real inference frames
                    self._handle_verification(detections, frame, current_fps)

                # ── Periodic Scene Report ──────────────────────────────────────
                self._reporter.tick(current_fps)

                # ── Rendering (bounding boxes + HUD) ──────────────────────
                # Draw on a copy only for display — inference already used the raw frame
                display_frame = frame.copy() if True else frame  # safe to draw on
                if self._show_overlays:
                    self._detector.draw_detections(
                        display_frame, detections,
                        scene_memory=self._scene_memory,
                        verification_cache=self._verification_cache,
                    )

                draw_hud(
                    display_frame, current_fps,
                    det_count=len(detections),
                    overlays_on=self._show_overlays,
                    gemini_budget=self._gemini.budget_remaining,
                    h=h, w=w,
                )

                # ── Display ────────────────────────────────────────────────
                cv2.imshow("Smart Vision Assistant", display_frame)

                # ── Key Handling ───────────────────────────────────────────────
                key = cv2.waitKey(1)
                if key != -1 and not self._handle_key(key):
                    self._running = False

        except KeyboardInterrupt:
            log.info("KeyboardInterrupt received — shutting down.")
        finally:
            self._cleanup()

    # ──────────────────────────────────────────────────────────────────────────
    # Shutdown
    # ──────────────────────────────────────────────────────────────────────────

    def _cleanup(self):
        log.info("Shutting down...")
        if self._camera:
            self._camera.release()
        cv2.destroyAllWindows()
        self._gemini.shutdown()
        self._reporter.shutdown()
        # Close DB session with final stats, then shut down writer thread
        self._db.end_session(
            session_id=self._session_id,
            total_reports=self._reporter._report_count,
            total_unique_objects=self._total_objects_seen,
            total_events=self._total_events,
        )
        self._db.shutdown()
        log.info("Shutdown complete.")


# ──────────────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    try:
        app = SmartVisionAssistant()
        app.run()
    except Exception as exc:
        log.error("Fatal error: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
