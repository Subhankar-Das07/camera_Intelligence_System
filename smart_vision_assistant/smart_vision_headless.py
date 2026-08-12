"""
smart_vision_headless.py
========================
Headless wrapper around the Smart Vision Assistant AI pipeline.

This is the ONLY new file that touches the core AI logic, and it does so
purely by composition — all existing modules are instantiated and called
exactly as main.py does, but WITHOUT:
  - cv2.imshow()       (no GUI window)
  - cv2.waitKey()      (no keyboard polling)
  - draw_hud()         (no HUD overlay on display frame)

Instead, this class:
  - Exposes get_latest_frame() → bytes (JPEG for MJPEG stream)
  - Exposes get_state()        → dict  (for WebSocket payloads)
  - Writes to EntityRegistry every frame
  - Runs in a background thread launched by web_server.py

All existing modules (SceneMemory, Detector, ReportEngine, etc.) are UNCHANGED.
"""

import logging
import os
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
from entity_registry import EntityRegistry
from ocr_processor import OCRProcessor
from snapshot_engine import SnapshotEngine

log = logging.getLogger("headless")

# ── Optional psutil ────────────────────────────────────────────────────────────
try:
    import psutil as _psutil_mod
    _psutil_proc = _psutil_mod.Process(os.getpid())
    _PSUTIL_OK = True
except ImportError:
    _psutil_mod = None
    _psutil_proc = None
    _PSUTIL_OK = False


# ──────────────────────────────────────────────────────────────────────────────
# FPS Tracker (copy of main.py's FPSTracker — avoids import cycle)
# ──────────────────────────────────────────────────────────────────────────────

class _FPSTracker:
    def __init__(self, window: int = 30):
        self._times: List[float] = []
        self._window = window

    def tick(self) -> float:
        now = time.monotonic()
        self._times.append(now)
        if len(self._times) > self._window:
            self._times.pop(0)
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Camera Thread (identical logic to main.py — kept here to be self-contained)
# ──────────────────────────────────────────────────────────────────────────────

class _CameraThread:
    def __init__(self, src: int = config.CAMERA_INDEX):
        log.info("Opening webcam %d at %dx%d (headless mode)", src,
                 config.FRAME_WIDTH, config.FRAME_HEIGHT)
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Camera {src} is unavailable. Close Teams/Zoom or change CAMERA_INDEX in config.py."
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._ret: bool = False
        self._running: bool = True

        self._ret, self._frame = self.cap.read()
        self._thread = threading.Thread(target=self._update, daemon=True, name="CameraGrab")
        self._thread.start()

    def _update(self) -> None:
        while self._running:
            ret, frame = self.cap.read()
            with self._lock:
                self._ret = ret
                if ret:
                    self._frame = frame

    def read(self) -> tuple:
        with self._lock:
            if self._frame is None:
                return False, None
            return self._ret, self._frame.copy()

    def release(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)
        self.cap.release()


# ──────────────────────────────────────────────────────────────────────────────
# SmartVisionHeadless
# ──────────────────────────────────────────────────────────────────────────────

class SmartVisionHeadless:
    """
    Headless AI pipeline runner.

    Instantiate once. Call start() to launch the background thread.
    Query get_latest_frame() and get_state() from the FastAPI thread.
    Call stop() on application shutdown.
    """

    # Confidence thresholds for JPEG-frame overlay (what gets drawn on stream)
    _STREAM_CONF_THRESHOLD = 0.45  # Raw YOLO threshold (same as config)

    def __init__(self):
        log.info("Initialising SmartVisionHeadless…")

        self._running = False
        self._frame_counter = 0
        self._last_detections: List[Detection] = []
        self._force_verify_next = False
        self._seen_track_ids: set = set()
        self._total_objects_seen = 0
        self._total_events = 0
        self._start_time = time.monotonic()

        # ── Core subsystems (identical to main.py) ─────────────────────────
        self._detector = ObjectDetector()
        self._scene_memory = SceneMemory()
        self._event_engine = EventEngine(stationary_time=30.0, abandoned_time=30.0)
        self._relationship_engine = RelationshipEngine()
        self._db = DatabaseManager()
        self._session_id = self._db.start_session()
        self._verification_cache: dict = {}
        self._gemini = None
        if config.ENABLE_GEMINI:
            self._gemini = GeminiVerifier(
                self._verification_cache,
                db_manager=self._db,
                session_id=self._session_id,
            )
        self._reporter = ReportEngine(
            self._scene_memory,
            db=self._db,
            session_id=self._session_id,
        )

        # ── Entity Registry — NEW layer ────────────────────────────────────
        self._entity_registry = EntityRegistry()

        # ── OCR Processor — NEW layer ──────────────────────────────────────
        self._ocr_processor = OCRProcessor(self._entity_registry, self._db, self._session_id)

        # ── Snapshot Engine — NEW layer ────────────────────────────────────
        self._snapshot_engine = SnapshotEngine(self._session_id)

        # ── State shared with API thread (lock-protected) ──────────────────
        self._state_lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._cached_state: dict = {
            "fps": 0.0, "cpu": 0.0, "ram_mb": 0.0,
            "active_objects": 0, "session_time": 0.0,
            "objects": [], "events": [], "relationships": [],
            "scene_stability": 0.0, "report": {},
        }

        # ── Report build throttle ──────────────────────────────────────────
        # build_report() serialises all entity dicts — expensive at 30 FPS.
        # WebSocket reads every 1s, so we rebuild every 1s to match.
        self._last_report_build: float = 0.0
        self._last_built_report: Optional[dict] = None
        self._report_build_interval: float = 1.0   # seconds between full report builds

        self._fps_tracker = _FPSTracker()
        self._camera: Optional[_CameraThread] = None
        self._thread: Optional[threading.Thread] = None

    # ──────────────────────────────────────────────────────────────────────────
    # Public API (called from FastAPI / WebSocket threads)
    # ──────────────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Launch the headless AI loop in a background daemon thread."""
        self._camera = _CameraThread(config.CAMERA_INDEX)
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="VisionLoop"
        )
        if config.ENABLE_OCR:
            self._ocr_processor.start()
        self._snapshot_engine.start()
        self._thread.start()
        log.info("SmartVisionHeadless started (background thread)")

    def stop(self) -> None:
        """Graceful shutdown — safe to call multiple times."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        if self._camera:
            self._camera.release()
        if config.ENABLE_OCR:
            self._ocr_processor.stop()
        self._snapshot_engine.stop()
        if config.ENABLE_GEMINI and self._gemini:
            self._gemini.shutdown()
        self._reporter.shutdown()
        self._db.end_session(
            session_id=self._session_id,
            total_reports=self._reporter._report_count,
            total_unique_objects=self._total_objects_seen,
            total_events=self._total_events,
        )
        self._db.shutdown()
        log.info("SmartVisionHeadless stopped")

    def get_latest_frame(self) -> Optional[bytes]:
        """Return the latest annotated JPEG frame bytes (for MJPEG stream)."""
        with self._state_lock:
            return self._latest_jpeg

    def get_state(self) -> dict:
        """Return the latest cached state snapshot (for WebSocket/REST)."""
        with self._state_lock:
            return dict(self._cached_state)

    def get_entity_registry(self) -> EntityRegistry:
        """Direct access to EntityRegistry (for /objects endpoint etc.)."""
        return self._entity_registry

    @property
    def is_running(self) -> bool:
        return self._running

    # ──────────────────────────────────────────────────────────────────────────
    # Internal AI Loop
    # ──────────────────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        log.info("Vision loop running…")
        try:
            while self._running:
                ret, frame = self._camera.read()
                if not ret or frame is None:
                    time.sleep(0.02)
                    continue

                h, w = frame.shape[:2]
                self._frame_counter += 1
                current_fps = self._fps_tracker.tick()

                # ── Adaptive frame skipping ────────────────────────────────
                run_inference = True
                if (
                    current_fps > 0
                    and current_fps < config.FRAME_SKIP_FPS_THRESHOLD
                    and self._frame_counter % config.FRAME_SKIP_N != 0
                ):
                    run_inference = False

                # ── Detection ─────────────────────────────────────────────
                if run_inference:
                    detections = self._detector.detect(frame)
                    self._last_detections = detections
                else:
                    detections = self._last_detections

                # ── Scene Memory Update ────────────────────────────────────
                new_ids, removed_ids = self._scene_memory.update(
                    detections, self._verification_cache
                )

                # ── Entity Registry Update ─────────────────────────────────
                relinked_tids = self._entity_registry.sync_from_scene_memory(
                    self._scene_memory, self._verification_cache
                )

                # ── DB Logging ─────────────────────────────────────────────
                for tid in new_ids:
                    rec = self._scene_memory.get_record(tid)
                    if rec:
                        self._db.on_object_new(
                            session_id=self._session_id,
                            track_id=tid,
                            yolo_label=rec.yolo_label,
                            display_label=rec.display_label,
                            category=rec.category,
                            confidence=rec.confidence,
                            bbox=rec.bbox,
                            is_returned=(tid in relinked_tids)
                        )
                        self._total_events += 1
                        if tid not in self._seen_track_ids:
                            self._seen_track_ids.add(tid)
                            self._total_objects_seen += 1

                for tid in removed_ids:
                    self._snapshot_engine.process_removal(str(tid))
                    removed = [r for r in self._scene_memory.get_recently_removed()
                               if r.track_id == tid]
                    if removed:
                        rec = removed[0]
                        self._db.on_object_removed(
                            session_id=self._session_id,
                            track_id=tid,
                            label=rec.display_label,
                            category=rec.category,
                            confidence=rec.confidence,
                            duration_seconds=rec.duration,
                        )
                        self._total_events += 1

                # ── Event Engine ───────────────────────────────────────────
                active_records = self._scene_memory.get_active_records()
                events = self._event_engine.process(active_records, time.monotonic())
                for tid, event_type, desc in events:
                    rec = self._scene_memory.get_record(tid)
                    if rec:
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

                # ── Relationship Engine ────────────────────────────────────
                rel_events = self._relationship_engine.process(active_records)
                for tid_a, tid_b, event_type, desc in rel_events:
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

                # ── Entity Registry Events Update ──────────────────────────
                self._entity_registry.update_events(events)
                self._entity_registry.update_relationships(rel_events)

                # ── Gemini Verification ────────────────────────────────────
                if run_inference and config.ENABLE_GEMINI:
                    self._handle_verification(detections, frame, current_fps)
                    
                    # ── Feed Snapshot Engine ───────────────────────────────────
                    for det in detections:
                        if det.track_id is None:
                            continue
                        is_new = det.track_id in new_ids
                        x1, y1, x2, y2 = det.bbox
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            self._snapshot_engine.process_detection(
                                str(det.track_id), crop, det.confidence, is_new
                            )

                # ── Periodic Scene Report (console) ────────────────────────
                self._reporter.tick(current_fps)



                # ── Trigger OCR (Async) ────────────────────────────────────
                if run_inference and config.ENABLE_OCR:
                    for det in detections:
                        if det.track_id is not None:
                            x1, y1, x2, y2 = det.bbox
                            crop = frame[y1:y2, x1:x2]
                            if crop.size > 0:
                                self._ocr_processor.submit_crop(str(det.track_id), crop, det.label)

                # ── Render bounding boxes onto frame ───────────────────────
                annotated = frame.copy()
                self._detector.draw_detections(
                    annotated, detections,
                    scene_memory=self._scene_memory,
                    verification_cache=self._verification_cache,
                )

                # ── Encode to JPEG ─────────────────────────────────────────
                ok, jpeg_buf = cv2.imencode(
                    ".jpg", annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, 75]
                )
                jpeg_bytes = jpeg_buf.tobytes() if ok else None

                # ── System stats (cheap — no dict allocation) ──────────────────
                cpu = ram_mb = 0.0
                if _PSUTIL_OK:
                    try:
                        cpu = _psutil_mod.cpu_percent(interval=None)
                        ram_mb = _psutil_proc.memory_info().rss / 1024 / 1024
                    except Exception:
                        pass

                session_time = time.monotonic() - self._start_time
                scene_stability = self._scene_memory.get_stability_pct()

                # ── Build structured report — throttled to 1s ─────────────────
                # build_report() serialises all entities to dicts (expensive at 30 FPS).
                # The WebSocket reads state every 1s, so building every frame wastes ~29
                # serialisation passes per second for no UI benefit.
                _now = time.monotonic()
                if (_now - self._last_report_build) >= self._report_build_interval:
                    self._last_report_build = _now
                    report = self._entity_registry.build_report(
                        fps=current_fps,
                        cpu=cpu,
                        ram_mb=ram_mb,
                        session_time=session_time,
                        scene_stability=scene_stability,
                    )
                    # Cache the built report for reuse between build intervals
                    self._last_built_report = report
                else:
                    # Reuse previous report — only update the fast-changing scalars
                    report = self._last_built_report
                    if report:
                        report["fps"] = round(current_fps, 1)
                        report["cpu"] = round(cpu, 1)
                        report["session_time"] = round(session_time, 1)

                # ── Update shared state (one lock acquisition) ─────────────────
                with self._state_lock:
                    self._latest_jpeg = jpeg_bytes
                    if report:
                        _active_count = report.get("active_objects", 0)
                        self._cached_state = {
                            "fps": round(current_fps, 1),
                            "cpu": round(cpu, 1),
                            "ram_mb": round(ram_mb, 1),
                            "active_objects": _active_count,  # reuse from report — no extra get_active()
                            "session_time": round(session_time, 1),
                            "objects": report["objects"],                    # active only
                            "inactive_objects": report["inactive_objects"],  # session history
                            "events": report["events"],
                            "relationships": report["relationships"],
                            "scene_stability": round(scene_stability, 1),
                            "report": report,
                            "status": {
                                "session_id": self._session_id,
                                "yolo": "running",
                                "tracker": "running",
                                "database": "connected",
                                "gemini": "ready" if (config.ENABLE_GEMINI and self._gemini and self._gemini.is_available) else "unavailable",
                                "fps": round(current_fps, 1),
                                "cpu": round(cpu, 1),
                                "ram_mb": round(ram_mb, 1),
                            },
                        }
                    else:
                        # First frame — no report yet, just update JPEG and fps
                        self._cached_state["fps"] = round(current_fps, 1)
                        self._cached_state["cpu"] = round(cpu, 1)
                        self._latest_jpeg = jpeg_bytes

        except Exception as exc:
            log.error("Vision loop crashed: %s", exc, exc_info=True)
        finally:
            log.info("Vision loop exited")

    # ──────────────────────────────────────────────────────────────────────────
    # Gemini Verification (identical logic to main.py._handle_verification)
    # ──────────────────────────────────────────────────────────────────────────

    def _handle_verification(
        self, detections: List[Detection], frame: np.ndarray, fps: float
    ) -> None:
        h, w = frame.shape[:2]
        auto_verify_ok = (
            config.ENABLE_AUTO_VERIFY and
            (getattr(config, 'DEBUG_BYPASS_FPS_GATING', False)
            or fps >= config.FPS_CRITICAL_THRESHOLD)
        )
        largest_det = None
        largest_area = 0

        for det in detections:
            tid = det.track_id
            if tid is None:
                continue
            area = (det.bbox[2] - det.bbox[0]) * (det.bbox[3] - det.bbox[1])
            if area > largest_area:
                largest_area = area
                largest_det = det

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
                    crop = self._crop(frame, det, w, h)
                    if crop is not None:
                        if eligible_for_label:
                            age = self._scene_memory.get_object_duration(tid)
                            self._gemini.enqueue_verification(tid, det.label, crop, age)
                        elif eligible_for_desc:
                            display = cached["label"] if cached and cached.get("label") else det.label
                            self._gemini.request_object_description(tid, display, crop)

        if self._force_verify_next and largest_det is not None:
            crop = self._crop(frame, largest_det, w, h)
            if crop is not None:
                age = self._scene_memory.get_object_duration(largest_det.track_id)
                self._gemini.enqueue_verification(
                    largest_det.track_id, largest_det.label, crop, age
                )
        self._force_verify_next = False

    def _crop(self, frame, det, w, h):
        x1, y1, x2, y2 = det.bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop_bgr = frame[y1:y2, x1:x2]
        return Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
