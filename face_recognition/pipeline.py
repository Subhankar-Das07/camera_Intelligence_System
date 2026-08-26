"""
face_recognition_pipeline.py — Face Recognition pipeline for Camera Intelligence System.

Integrates with the existing BaseVideoPipeline contract:
    initialize()     → load models (lazy, on first frame)
    process_frame()  → single-frame processing (used when called standalone)
    run_on_video()   → main streaming loop (used by the FastAPI MJPEG endpoint)

Architecture inside this pipeline:
    Frame → FaceEmbedder (SCRFD detect + ArcFace embed)
          → FaceTracker  (ByteTrack, every frame)
          → FaceRecognizer (every N-th frame, temporal consensus)
          → Annotator (draw bounding boxes + labels)
          → yield (annotated_frame, optional_alert_event)

Loose coupling:
    All face-recognition logic lives in face_recognition/; this file only
    orchestrates the sub-modules and adapts their output to the pipeline
    contract expected by main.py.
"""

import logging
import os
import time
import uuid
from typing import Any, Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np

from core.base_pipeline import BaseVideoPipeline
from core.video_source import get_video_source

from face_recognition.config import (
    COLOR_KNOWN, COLOR_UNKNOWN, COLOR_LOW_QUALITY, COLOR_CANDIDATE,
    RECOGNITION_EVERY_N_FRAMES,
    AUTO_LABEL_PREFIX,
    DOOR_LEARNING_FRAMES, DOOR_PIXEL_DIFF_THRESH,
    DOOR_OPEN_THRESHOLD, DOOR_CLOSE_THRESHOLD, DOOR_DEBOUNCE,
)
from face_recognition.embedder import FaceEmbedder
from face_recognition.tracker import FaceTracker, TrackedFace
from face_recognition.recognizer import FaceRecognizer, RecognitionResult
from face_recognition.identity_manager import IdentityManager

log = logging.getLogger(__name__)


class FaceRecognitionPipeline(BaseVideoPipeline):
    """
    Production-grade face recognition pipeline.

    Annotation colours:
        GREEN  → face is recognised (in persons DB) — shows label + confidence
        BLUE   → scanning (collecting quality frames, N/3 shown)
        GRAY   → too blurry / too small to process
    """

    def __init__(self):
        self._embedder: Optional[FaceEmbedder] = None
        self._tracker: Optional[FaceTracker] = None
        self._visitor_recognizer: Optional[FaceRecognizer] = None
        self._attendance_recognizer: Optional[FaceRecognizer] = None
        self._visitor_manager: Optional[IdentityManager] = None
        self._attendance_manager: Optional[IdentityManager] = None
        self._door_state: Dict[str, Dict[str, Any]] = {}
        self._initialized = False

    # ── BaseVideoPipeline interface ────────────────────────────────────────────

    def initialize(self, **kwargs) -> None:
        """
        Load all sub-modules. Called once by the registry on first use.
        InsightFace models are downloaded on first call (cached after).
        """
        log.info("[FaceRecognitionPipeline] Initializing sub-modules…")
        self._visitor_manager   = IdentityManager("fr:visitor")
        self._attendance_manager = IdentityManager("fr:attendance")
        self._embedder         = FaceEmbedder()
        self._tracker          = FaceTracker()
        self._visitor_recognizer = FaceRecognizer(self._visitor_manager)
        self._attendance_recognizer = FaceRecognizer(self._attendance_manager)
        self._initialized = True
        log.info("[FaceRecognitionPipeline] Ready. Visitor IDs: %d | Attendance IDs: %d",
                 len(self._visitor_manager.get_all_identities()),
                 len(self._attendance_manager.get_all_identities()))

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        roi_polygon: np.ndarray,
        config: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Single-frame processing (standalone usage)."""
        if not self._initialized:
            self.initialize()
        annotated, event = self._process(frame, frame_idx, config)
        return annotated, event or {}

    def run_on_video(
        self,
        input_path,
        output_dir: str,
        roi_normalized: List[Tuple[float, float]],
        config: Dict[str, Any],
    ) -> Generator[Tuple[np.ndarray, Optional[Dict[str, Any]]], None, None]:
        """
        Main streaming loop. Yields (annotated_frame, optional_alert_event) per frame.

        ROI is intentionally unused in face recognition — faces are detected
        across the whole frame regardless of the drawn region of interest.
        (The ROI concept is relevant for intrusion/danger-zone but not for face ID.)
        """
        if not self._initialized:
            self.initialize()

        cap = get_video_source(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_idx = 0
        vw_states = {} # For Vision Watch tracking

        log.info("[FaceRecognitionPipeline] Stream started (recognition every %d frames).",
                 RECOGNITION_EVERY_N_FRAMES)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame, alert_event = self._process(frame, frame_idx, config, vw_states)

            yield annotated_frame, alert_event
            frame_idx += 1

        cap.release()
        log.info("[FaceRecognitionPipeline] Stream ended at frame %d.", frame_idx)

    # ── Door State Tracker (BGS-based) ────────────────────────────────────────

    def get_door_state(self, session_id: str) -> Dict[str, Any]:
        """Return the current live door tracking state for a session."""
        ds = self._door_state.get(session_id, {})
        learning_frames = ds.get("learning_frames", 0)
        is_learning = learning_frames < DOOR_LEARNING_FRAMES
        return {
            "state":       ds.get("state", "INITIALIZING"),
            "open_count":  ds.get("open_count", 0),
            "close_count": ds.get("close_count", 0),
            "last_event":  ds.get("last_event", None),
            "learning":    is_learning,
        }

    # ── Core per-frame logic ───────────────────────────────────────────────────

    def _process(
        self,
        frame: np.ndarray,
        frame_idx: int,
        config: Dict[str, Any],
        vw_states: Optional[Dict[int, Any]] = None,
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Full processing pipeline for one frame.

        Returns:
            (annotated_frame, alert_event_or_None)
        """
        mode = config.get("mode", "visitor")
        do_recognition = (frame_idx % RECOGNITION_EVERY_N_FRAMES == 0)
        self._last_config = config

        h, w = frame.shape[:2]  # define h, w early for door tracking

        # Step 1: Detect (+ embed only on recognition frames)
        face_results = self._embedder.get_faces(frame, extract_embeddings=do_recognition)

        # Step 2: Track (every frame — ByteTrack keeps IDs stable)
        tracked_faces = self._tracker.update(face_results, frame)

        # Step 3: Recognise (every N-th frame OR if recognition is forced)
        results_map: Dict[int, RecognitionResult] = {}
        alert_event = None

        for tf in tracked_faces:
            if do_recognition and tf.is_quality and tf.embedding is not None:
                if mode == "attendance":
                    rec = self._attendance_recognizer.classify(tf.track_id, tf.embedding, self._tracker, auto_register=False)
                    active_recognizer = self._attendance_recognizer
                else:
                    rec = self._visitor_recognizer.classify(tf.track_id, tf.embedding, self._tracker, auto_register=True)
                    active_recognizer = self._visitor_recognizer
                
                results_map[tf.track_id] = rec

                # Fire alert event on fresh commitment (within 0.5s of commit)
                if rec.status == "recognised" and rec.person_id:
                    state = active_recognizer._track_states.get(tf.track_id)
                    if state and state.committed_at and (time.time() - state.committed_at) < 0.5:
                        alert_event = {
                            "id":        str(uuid.uuid4()),
                            "person_id": rec.person_id,
                            "label":     rec.label,
                            "status":    "recognised",
                            "confidence": rec.confidence,
                            "timestamp": time.strftime("%H:%M:%S"),
                            "type":      "face_recognised",
                            "track_id":  tf.track_id,
                            "bbox":      list(tf.bbox) if tf.bbox is not None else [],
                        }
            else:
                active_recognizer = self._attendance_recognizer if mode == "attendance" else self._visitor_recognizer
                if tf.track_id in active_recognizer._track_states:
                    # Reuse last result for non-recognition frames
                    state = active_recognizer._track_states[tf.track_id]
                    if state.last_result:
                        results_map[tf.track_id] = state.last_result

        # Vision Watch Logic
        if mode == "vision_watch" and vw_states is not None:
            watchman_name = config.get("watchman_name", "").lower()
            multiple_rois = config.get("multiple_rois", [])

            # Convert normalized rois to absolute polygons
            abs_polys = []
            for roi in multiple_rois:
                poly = np.array([[int(p[0] * w), int(p[1] * h)] for p in roi], dtype=np.int32)
                abs_polys.append(poly)
            
            for tf in tracked_faces:
                rec = results_map.get(tf.track_id)
                if not rec or rec.status != "recognised":
                    continue
                
                # Check if face center is in any polygon
                x1, y1, x2, y2 = tf.bbox
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                
                in_zone = False
                for poly in abs_polys:
                    if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                        in_zone = True
                        break
                
                prev_in_zone = vw_states.get(tf.track_id, False)
                vw_states[tf.track_id] = in_zone
                
                # Gate Open Count logic: Fire alert when entering zone (if previously outside)
                if in_zone and not prev_in_zone:
                    if not alert_event: # Only fire if we haven't already fired an alert this frame
                        alert_event = {
                            "id": str(uuid.uuid4()),
                            "person_id": rec.person_id,
                            "label": f"Gate Access: {rec.label}",
                            "status": "known",
                            "timestamp": time.strftime("%H:%M:%S"),
                            "type": "gate_access",
                        }
                
                # Watchman Alert Logic: Fire alert if watchman is outside all zones
                if watchman_name and rec.label.lower() == watchman_name:
                    if not in_zone:
                        # Throttle watchman alerts to max 1 per 2 seconds
                        last_alert_time = vw_states.get("last_alert_time", 0)
                        if time.time() - last_alert_time > 2.0:
                            vw_states["last_alert_time"] = time.time()
                            alert_event = {
                                "id": str(uuid.uuid4()),
                                "person_id": rec.person_id,
                                "label": f"Watchman Alert",
                                "message": f"{rec.label} has left the zone!",
                                "status": "unknown",
                                "timestamp": time.strftime("%H:%M:%S"),
                                "type": "vision_watch_alert",
                            }

            # Door Tracking — Structural reference comparator
            session_id = config.get("session_id", "default")
            door_rois = config.get("door_rois", [])

            if door_rois:
                # Initialize fresh state for each new session
                if session_id not in self._door_state:
                    self._door_state[session_id] = {
                        "closed_reference":  None,
                        "learning_buffer":   [],
                        "state":             "INITIALIZING",
                        "open_count":        0,
                        "close_count":       0,
                        "transition_frames": 0,
                        "last_event":        None,
                        "learning_frames":   0,
                    }

                ds = self._door_state[session_id]

                # Build absolute door polygon from the FIRST drawn zone
                door_poly = np.array(
                    [[int(p[0] * w), int(p[1] * h)] for p in door_rois[0]], dtype=np.int32
                )

                # Mask restricted to door zone
                zone_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(zone_mask, [door_poly], 255)
                zone_pixels = cv2.countNonZero(zone_mask)

                if zone_pixels > 0:
                    # Structural comparison logic
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    roi_gray = cv2.bitwise_and(gray, gray, mask=zone_mask)
                    # Blur slightly to reduce sensor noise
                    roi_gray = cv2.GaussianBlur(roi_gray, (5, 5), 0)

                    ds["learning_frames"] += 1

                    if ds["learning_frames"] <= DOOR_LEARNING_FRAMES:
                        ds["learning_buffer"].append(roi_gray)
                        ds["state"] = "INITIALIZING"
                        
                        if ds["learning_frames"] == DOOR_LEARNING_FRAMES:
                            # Compute the median closed reference image from the buffer
                            stacked = np.stack(ds["learning_buffer"], axis=0)
                            ds["closed_reference"] = np.median(stacked, axis=0).astype(np.uint8)
                            ds["learning_buffer"].clear()  # Free memory
                            ds["state"] = "CLOSED"
                    else:
                        # Compare current frame against the ROCK SOLID closed reference
                        diff = cv2.absdiff(roi_gray, ds["closed_reference"])
                        _, thresh = cv2.threshold(diff, DOOR_PIXEL_DIFF_THRESH, 255, cv2.THRESH_BINARY)

                        # Morphological clean-up — remove speckle noise
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                        fg_clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                        fg_clean = cv2.dilate(fg_clean, kernel, iterations=2)

                        active_pixels  = cv2.countNonZero(fg_clean)
                        activity_score = active_pixels / zone_pixels

                        current_state = ds["state"]

                        if current_state in ("CLOSED", "INITIALIZING"):
                            if activity_score > DOOR_OPEN_THRESHOLD:
                                ds["transition_frames"] += 1
                                if ds["transition_frames"] >= DOOR_DEBOUNCE:
                                    ds["state"]             = "OPEN"
                                    ds["open_count"]        += 1
                                    ds["transition_frames"] = 0
                                    ds["last_event"]        = time.strftime("%H:%M:%S")
                                    if not alert_event:
                                        alert_event = {
                                            "id":        str(uuid.uuid4()),
                                            "person_id": None,
                                            "label":     "Door Opened",
                                            "message":   f"Door opened! (Open #{ds['open_count']})",
                                            "status":    "info",
                                            "timestamp": ds["last_event"],
                                            "type":      "door_open",
                                        }
                            else:
                                ds["transition_frames"] = 0

                        elif current_state == "OPEN":
                            if activity_score < DOOR_CLOSE_THRESHOLD:
                                ds["transition_frames"] += 1
                                if ds["transition_frames"] >= DOOR_DEBOUNCE:
                                    ds["state"]             = "CLOSED"
                                    ds["close_count"]       += 1
                                    ds["transition_frames"] = 0
                                    ds["last_event"]        = time.strftime("%H:%M:%S")
                                    if not alert_event:
                                        alert_event = {
                                            "id":        str(uuid.uuid4()),
                                            "person_id": None,
                                            "label":     "Door Closed",
                                            "message":   f"Door closed! (Close #{ds['close_count']})",
                                            "status":    "info",
                                            "timestamp": ds["last_event"],
                                            "type":      "door_close",
                                        }
                            else:
                                ds["transition_frames"] = 0


        # Step 4: Annotate frame
        annotated = self._annotate(frame, tracked_faces, results_map, mode)

        # Step 5: Draw HUD stats
        annotated = self._draw_hud(annotated)

        return annotated, alert_event

    # ── Annotation ─────────────────────────────────────────────────────────────

    def _annotate(
        self,
        frame: np.ndarray,
        tracked_faces: List[TrackedFace],
        results_map: Dict[int, RecognitionResult],
        mode: str = "visitor",
    ) -> np.ndarray:
        """Draw bounding boxes and labels on frame. Returns new annotated copy."""
        out = frame.copy()
        
        # In Vision Watch mode, draw the zones on the backend output
        if mode == "vision_watch" and hasattr(self, "_last_config"):
            multiple_rois = self._last_config.get("multiple_rois", [])
            h, w = out.shape[:2]
            for roi in multiple_rois:
                poly = np.array([[int(p[0] * w), int(p[1] * h)] for p in roi], dtype=np.int32)
                cv2.polylines(out, [poly], isClosed=True, color=(0, 0, 255), thickness=2)

            door_rois = self._last_config.get("door_rois", [])
            session_id = self._last_config.get("session_id", "default")
            ds = self._door_state.get(session_id, {})
            door_state = ds.get("state", "INITIALIZING")
            open_count  = ds.get("open_count", 0)
            close_count = ds.get("close_count", 0)
            is_learning = ds.get("learning_frames", 0) <= DOOR_LEARNING_FRAMES

            if door_state == "OPEN":
                door_color = (0, 0, 255)    # Red = open
            elif door_state == "CLOSED":
                door_color = (0, 200, 0)    # Green = closed
            else:
                door_color = (128, 128, 0)  # Yellow = initializing

            for roi in door_rois:
                poly = np.array([[int(p[0] * w), int(p[1] * h)] for p in roi], dtype=np.int32)
                cv2.polylines(out, [poly], isClosed=True, color=door_color, thickness=3)
                label = (
                    f"Door: LEARNING..." if is_learning else
                    f"Door: {door_state} | Opens:{open_count} Closes:{close_count}"
                )
                cv2.putText(
                    out, label,
                    (poly[0][0], max(14, poly[0][1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, door_color, 2, cv2.LINE_AA
                )

        for tf in tracked_faces:
            x1, y1, x2, y2 = tf.bbox
            rec = results_map.get(tf.track_id)

            # ── Colour + label ─────────────────────────────────────────────────
            if not tf.is_quality:
                color = COLOR_LOW_QUALITY
                label = f"ID:{tf.track_id} (low quality)"

            elif rec is None:
                color = COLOR_LOW_QUALITY
                label = f"ID:{tf.track_id} ..."

            elif rec.status == "recognised":
                # If confidence > 0, they matched an existing embedding in the DB (previously known)
                # If confidence == 0, they were just auto-saved for the first time in this session (unknown)
                is_known_to_system = (rec.confidence > 0.0)

                if mode == "attendance":
                    color = COLOR_KNOWN      # Green
                    prefix = "Present: "
                else:
                    if is_known_to_system:
                        color = COLOR_KNOWN      # Green
                        prefix = "Known: "
                    else:
                        color = COLOR_UNKNOWN    # Orange
                        prefix = "Unknown: "
                
                pct   = int(rec.confidence * 100)
                base_label = f"{rec.label} ({pct}%)" if pct > 0 else rec.label
                label = f"{prefix}{base_label}"

            elif rec.status == "scanning":
                color = COLOR_CANDIDATE  # Blue
                label = rec.label        # "Scanning... (N/3)"

            elif rec.status == "unknown":
                color = (0, 0, 255)      # Red for unregistered in attendance
                label = "Unregistered"

            else:
                color = COLOR_LOW_QUALITY
                label = f"ID:{tf.track_id}"

            # Bounding box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            # Label background pill
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            lx1 = x1
            ly1 = max(0, y1 - th - baseline - 6)
            lx2 = x1 + tw + 8
            ly2 = y1

            cv2.rectangle(out, (lx1, ly1), (lx2, ly2), color, -1)
            cv2.putText(
                out, label,
                (lx1 + 4, ly2 - baseline - 2),
                font, font_scale,
                (255, 255, 255),
                thickness, cv2.LINE_AA,
            )

        return out

    def _draw_hud(self, frame: np.ndarray) -> np.ndarray:
        """Draw top-left status overlay on the frame."""
        stats   = self.get_visitor_manager().get_stats()
        total   = stats["total_identities"]

        pad = 10
        cv2.putText(frame, f"FR SYSTEM ACTIVE", (pad, pad + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Total Persons: {total}", (pad, pad + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"Known: {total} | Other: 0", (pad, pad + 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return frame

    # ── Utility: expose identity_managers for API endpoints ────────────────────

    def get_visitor_manager(self) -> IdentityManager:
        """Allow main.py API endpoints to access the visitor identity store."""
        if not self._initialized:
            self.initialize()
        return self._visitor_manager

    def get_attendance_manager(self) -> IdentityManager:
        """Allow main.py API endpoints to access the attendance identity store."""
        if not self._initialized:
            self.initialize()
        return self._attendance_manager
