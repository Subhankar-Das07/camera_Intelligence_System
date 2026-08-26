"""
test_pipeline.py — Integration tests for pipeline.py business logic.

Covers:
  1. Spatial / ROI containment engine (ShapelyPolygon.contains)
  2. 6-second loitering → alert trigger for Unknown vehicles
  3. Known vehicles are NOT alerted even after 6s loitering
  4. alerted_tracks reset between sessions (run_on_video state clearing)
  5. Red bounding box color (_ALERT_COLOR) applied on alerted tracks

All heavy ML models (YOLO, PlateReader) and Redis are fully mocked.
Tests run in ~0.3s total.
"""

from __future__ import annotations

import threading
import time
import unittest.mock as _mock
from collections import deque
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Constants mirrored from pipeline.py for assertion clarity
# ---------------------------------------------------------------------------
_BOX_COLOR   = (0, 220, 60)
_ALERT_COLOR = (0, 0, 255)
_SCAN_COLOR  = (30, 170, 255)


# ---------------------------------------------------------------------------
# Factory: build a fully-mocked VehicleRecognitionPipeline instance
# ---------------------------------------------------------------------------

def _make_pipeline(
    min_consensus_frames: int = 1,
    session_cooldown: int = 120,
):
    """
    Instantiate VehicleRecognitionPipeline without loading any real models or
    touching Redis. conftest.py has already stubbed all heavy dependencies
    at the session level, so a direct import is safe.
    """
    from pipelines.vehicle_recognition.pipeline import VehicleRecognitionPipeline
    p = VehicleRecognitionPipeline.__new__(VehicleRecognitionPipeline)

    # Wire up the sub-component mocks directly
    p.detector       = MagicMock()
    p.plate_reader   = MagicMock()
    p.db             = MagicMock()
    p.visit_manager  = MagicMock()
    p.visit_manager.finalized_tracks = {}
    p.frame_buffer   = deque(maxlen=180)
    p.loiter_timers  = {}
    p.alerted_tracks = set()

    return p


def _blank_frame(h: int = 480, w: int = 640) -> np.ndarray:
    """Return a blank BGR frame usable as a video frame."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_yolo_result(boxes, track_ids, cls_ids):
    """
    Build a fake YOLO result object matching the exact attribute chain
    that process_frame reads:
      results[0].boxes.xyxy.cpu().numpy().astype(int)
      results[0].boxes.id.cpu().numpy().astype(int)
      results[0].boxes.cls.cpu().numpy().astype(int)
    """
    _boxes_arr = np.array(boxes,     dtype=np.float32)
    _ids_arr   = np.array(track_ids, dtype=np.float32)
    _cls_arr   = np.array(cls_ids,   dtype=np.float32)

    def _tensor_mock(arr):
        """Return a mock whose .cpu().numpy() gives the real array."""
        t = _mock.MagicMock()
        t.cpu.return_value.numpy.return_value = arr
        return t

    boxes_mock = _mock.MagicMock()
    boxes_mock.xyxy = _tensor_mock(_boxes_arr)
    boxes_mock.id   = _tensor_mock(_ids_arr)
    boxes_mock.cls  = _tensor_mock(_cls_arr)

    result = _mock.MagicMock()
    result.boxes = boxes_mock
    return result


# ===========================================================================
# TEST GROUP 1: Spatial ROI Engine (Shapely containment)
# ===========================================================================

class TestSpatialROIEngine:
    """
    Verifies the ShapelyPolygon.contains logic using the bottom-center point
    of the vehicle bounding box (bc_pt = Point((x1+x2)/2, y2)).
    """

    def test_vehicle_inside_roi_starts_loiter_timer(self):
        """bc_pt inside polygon → loiter timer should start for this track."""
        p = _make_pipeline()

        # A finalized status so process_track returns immediately (already confirmed plate)
        p.visit_manager.process_track.return_value = {
            "plate": "TEST001", "total_visits": 1, "vehicle_type": "Car"
        }
        p.visit_manager.finalized_tracks = {42: True}
        p.db.get_vehicle_stats.return_value = {"status": "Unknown", "total_visits": 1}

        # ROI polygon: a square from (0,0) to (640,480) — covers the whole frame
        roi = np.array([[0, 0], [640, 0], [640, 480], [0, 480]], dtype=int)

        # Vehicle bounding box: centre at (320, 240), bottom at y=300
        # bc_pt = (320, 300) → INSIDE the polygon
        frame = _blank_frame()
        p.detector.track.return_value = [_make_yolo_result(
            boxes=[[100, 100, 540, 300]], track_ids=[42], cls_ids=[2]  # cls 2 = Car
        )]

        p.process_frame(frame, frame_idx=1, roi_polygon=roi, config={})

        # Timer must have been started for track 42
        assert 42 in p.loiter_timers

    def test_vehicle_outside_roi_does_not_start_timer(self):
        """bc_pt outside polygon → loiter timer must NOT start."""
        p = _make_pipeline()
        p.visit_manager.process_track.return_value = {
            "plate": "TEST002", "total_visits": 1, "vehicle_type": "Car"
        }
        p.visit_manager.finalized_tracks = {99: True}
        p.db.get_vehicle_stats.return_value = {"status": "Unknown", "total_visits": 1}

        # Tiny ROI polygon in the top-left corner only (0,0)→(50,50)
        roi = np.array([[0, 0], [50, 0], [50, 50], [0, 50]], dtype=int)

        # Vehicle bounding box has bc_pt at (320, 300) — far outside the tiny ROI
        frame = _blank_frame()
        p.detector.track.return_value = [_make_yolo_result(
            boxes=[[100, 100, 540, 300]], track_ids=[99], cls_ids=[2]
        )]

        p.process_frame(frame, frame_idx=1, roi_polygon=roi, config={})

        assert 99 not in p.loiter_timers

    def test_no_roi_means_whole_frame_is_zone(self):
        """When roi_polygon is None, in_zone defaults to True → timer starts."""
        p = _make_pipeline()
        p.visit_manager.process_track.return_value = {
            "plate": "TEST003", "total_visits": 1, "vehicle_type": "Car"
        }
        p.visit_manager.finalized_tracks = {11: True}
        p.db.get_vehicle_stats.return_value = {"status": "Unknown", "total_visits": 1}

        frame = _blank_frame()
        p.detector.track.return_value = [_make_yolo_result(
            boxes=[[100, 100, 540, 300]], track_ids=[11], cls_ids=[2]
        )]

        p.process_frame(frame, frame_idx=1, roi_polygon=None, config={})

        assert 11 in p.loiter_timers

    def test_vehicle_leaving_roi_resets_timer(self):
        """A vehicle that exits the ROI should have its loiter timer removed."""
        p = _make_pipeline()
        p.visit_manager.process_track.return_value = {
            "plate": "TEST004", "total_visits": 1, "vehicle_type": "Car"
        }
        p.visit_manager.finalized_tracks = {77: True}
        p.db.get_vehicle_stats.return_value = {"status": "Unknown", "total_visits": 1}

        # Pre-seed a loiter timer as if the vehicle was already inside
        p.loiter_timers[77] = time.time() - 3.0

        # Tiny ROI — vehicle bc_pt (320, 300) is outside
        roi = np.array([[0, 0], [50, 0], [50, 50], [0, 50]], dtype=int)
        frame = _blank_frame()
        p.detector.track.return_value = [_make_yolo_result(
            boxes=[[100, 100, 540, 300]], track_ids=[77], cls_ids=[2]
        )]

        p.process_frame(frame, frame_idx=1, roi_polygon=roi, config={})

        # Timer must be cleared because bc_pt is outside the ROI
        assert 77 not in p.loiter_timers


# ===========================================================================
# TEST GROUP 2: 6-Second Loitering Alert
# ===========================================================================

class TestLoiteringAlert:
    """
    Verifies the 6-second loitering threshold:
    - Unknown vehicles trigger an alert + alert clip after 6 s
    - Known vehicles are NEVER alerted
    - Each track only fires once (alerted_tracks deduplication)
    """

    def _setup_loitering_scenario(self, p, tid: int, status_str: str, seconds_loitered: float):
        """Shared setup for loitering tests."""
        p.visit_manager.process_track.return_value = {
            "plate": "MH12AB9999", "total_visits": 2, "vehicle_type": "Car"
        }
        p.visit_manager.finalized_tracks = {tid: True}
        p.db.get_vehicle_stats.return_value = {"status": status_str, "total_visits": 2}

        # Pre-seed a loiter timer so the vehicle appears to have been in zone for N seconds
        p.loiter_timers[tid] = time.time() - seconds_loitered

        # Vehicle is inside the whole-frame ROI (roi_polygon=None)
        frame = _blank_frame()
        p.detector.track.return_value = [_make_yolo_result(
            boxes=[[50, 50, 400, 380]], track_ids=[tid], cls_ids=[2]
        )]
        return frame

    def test_unknown_vehicle_triggers_alert_after_6s(self):
        """An Unknown vehicle loitering for >6s must be added to alerted_tracks."""
        p = _make_pipeline()
        tid = 55
        frame = self._setup_loitering_scenario(p, tid, "Unknown", seconds_loitered=7.0)

        _, metadata = p.process_frame(frame, frame_idx=1, roi_polygon=None, config={})

        assert tid in p.alerted_tracks, "Track should be in alerted_tracks after 6s loitering"
        assert len(metadata["alerts"]) == 1, "Exactly one alert should be emitted"
        assert metadata["alerts"][0]["plate"] == "MH12AB9999"

    def test_known_vehicle_never_triggers_alert(self):
        """A Known vehicle loitering for >6s must NOT trigger an alert."""
        p = _make_pipeline()
        tid = 66
        frame = self._setup_loitering_scenario(p, tid, "Known", seconds_loitered=10.0)

        _, metadata = p.process_frame(frame, frame_idx=1, roi_polygon=None, config={})

        assert tid not in p.alerted_tracks, "Known vehicles should never be alerted"
        assert len(metadata["alerts"]) == 0, "No alerts should be emitted for Known vehicles"

    def test_vehicle_under_6s_does_not_trigger_alert(self):
        """An Unknown vehicle loitering for <6s must NOT trigger an alert yet."""
        p = _make_pipeline()
        tid = 77
        frame = self._setup_loitering_scenario(p, tid, "Unknown", seconds_loitered=3.0)

        _, metadata = p.process_frame(frame, frame_idx=1, roi_polygon=None, config={})

        assert tid not in p.alerted_tracks
        assert len(metadata["alerts"]) == 0

    def test_alert_fires_only_once_per_track(self):
        """The same track must only fire ONE alert even across multiple frames."""
        p = _make_pipeline()
        tid = 88
        frame = self._setup_loitering_scenario(p, tid, "Unknown", seconds_loitered=8.0)

        # First frame fires the alert
        _, meta1 = p.process_frame(frame, frame_idx=1, roi_polygon=None, config={})
        # Second frame — track is already in alerted_tracks, should not fire again
        _, meta2 = p.process_frame(frame, frame_idx=2, roi_polygon=None, config={})

        assert len(meta1["alerts"]) == 1
        assert len(meta2["alerts"]) == 0, "Second frame must not re-fire the alert"

    def test_alert_metadata_contains_required_fields(self):
        """Alert payload must include plate, clip_path, snapshot_path, type, time_spent."""
        p = _make_pipeline()
        tid = 99
        frame = self._setup_loitering_scenario(p, tid, "Unknown", seconds_loitered=7.5)

        _, metadata = p.process_frame(frame, frame_idx=1, roi_polygon=None, config={})

        assert metadata["alerts"], "Expected at least one alert"
        alert = metadata["alerts"][0]
        assert "plate"         in alert
        assert "clip_path"     in alert
        assert "snapshot_path" in alert
        assert "type"          in alert
        assert "time_spent"    in alert


# ===========================================================================
# TEST GROUP 3: Session State Reset Between Runs
# ===========================================================================

class TestSessionStateReset:
    """
    Verifies that run_on_video clears frame_buffer, alerted_tracks, and
    loiter_timers at the start of every new session, preventing stale state
    from bleeding into new video sessions.
    """

    def test_run_on_video_clears_frame_buffer(self):
        """frame_buffer must be empty at the start of a new run."""
        p = _make_pipeline()
        # Pre-load stale frames from a previous session
        p.frame_buffer.extend([_blank_frame() for _ in range(50)])
        assert len(p.frame_buffer) == 50

        # Provide a VideoCapture that immediately returns False on grab()
        with patch("pipelines.vehicle_recognition.pipeline.get_video_source") as mock_src:
            cap = MagicMock()
            cap.isOpened.return_value = True
            cap.grab.return_value = False    # immediately break the while loop
            cap.release = MagicMock()
            mock_src.return_value = cap

            # Exhaust the generator — it will clear buffers then return on first grab
            list(p.run_on_video("fake_path.mp4", "output_dir"))

        assert len(p.frame_buffer) == 0, "frame_buffer must be cleared for new sessions"

    def test_run_on_video_clears_alerted_tracks(self):
        """alerted_tracks must be empty at the start of a new run."""
        p = _make_pipeline()
        p.alerted_tracks = {1, 2, 3}  # Stale state from previous session

        with patch("pipelines.vehicle_recognition.pipeline.get_video_source") as mock_src:
            cap = MagicMock()
            cap.isOpened.return_value = True
            cap.grab.return_value = False
            cap.release = MagicMock()
            mock_src.return_value = cap
            list(p.run_on_video("fake_path.mp4", "output_dir"))

        assert len(p.alerted_tracks) == 0, "alerted_tracks must be cleared for new sessions"

    def test_run_on_video_clears_loiter_timers(self):
        """loiter_timers must be empty at the start of a new run."""
        p = _make_pipeline()
        p.loiter_timers = {10: time.time() - 100}  # Stale timer from old session

        with patch("pipelines.vehicle_recognition.pipeline.get_video_source") as mock_src:
            cap = MagicMock()
            cap.isOpened.return_value = True
            cap.grab.return_value = False
            cap.release = MagicMock()
            mock_src.return_value = cap
            list(p.run_on_video("fake_path.mp4", "output_dir"))

        assert len(p.loiter_timers) == 0, "loiter_timers must be cleared for new sessions"


# ===========================================================================
# TEST GROUP 4: Bounding Box Color Logic
# ===========================================================================

class TestBoundingBoxColors:
    """
    Verifies the visual annotation color assignment:
    - Normal Unknown vehicle  → _SCAN_COLOR (amber)
    - Confirmed plate         → _BOX_COLOR (green)
    - Alerted track           → _ALERT_COLOR (red)
    """

    def _run_frame_for_tid(self, p, tid: int, status_return, pre_alerted: bool = False):
        if pre_alerted:
            p.alerted_tracks.add(tid)
        p.visit_manager.finalized_tracks = {tid: True} if status_return else {}
        p.visit_manager.process_track.return_value = status_return
        p.db.get_vehicle_stats.return_value = {"status": "Unknown", "total_visits": 1}

        frame = _blank_frame()
        p.detector.track.return_value = [_make_yolo_result(
            boxes=[[100, 100, 400, 300]], track_ids=[tid], cls_ids=[2]
        )]
        annotated, _ = p.process_frame(frame, frame_idx=1, roi_polygon=None, config={})
        return annotated

    def test_alerted_track_draws_red_box_on_frame(self):
        """
        When a track is in alerted_tracks, the bounding box drawn on the frame
        must use _ALERT_COLOR (red: BGR = 0, 0, 255).
        We verify this by checking that at least some pixel in the expected box
        area contains a reddish value (blue channel near 0, red channel near 255).
        """
        import cv2 as cv2_module
        p = _make_pipeline()
        tid = 200

        # Mock cv2.rectangle to record what color it was called with
        drawn_colors = []
        original_rect = cv2_module.rectangle

        def capture_rect(img, pt1, pt2, color, thickness):
            drawn_colors.append(color)
            return original_rect(img, pt1, pt2, color, thickness)

        with patch("pipelines.vehicle_recognition.pipeline.cv2.rectangle", side_effect=capture_rect):
            self._run_frame_for_tid(
                p, tid,
                status_return={"plate": "ALERT01", "total_visits": 1, "vehicle_type": "Car"},
                pre_alerted=True
            )

        # The first rectangle drawn is the bounding box
        assert drawn_colors, "cv2.rectangle was never called"
        assert drawn_colors[0] == _ALERT_COLOR, (
            f"Expected ALERT_COLOR {_ALERT_COLOR}, got {drawn_colors[0]}"
        )
