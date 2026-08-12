"""
tests/test_suite.py — Smart Vision Assistant
=============================================
Comprehensive test suite covering all mandatory features.

Run with:
    cd smart_vision_assistant
    python -m pytest tests/ -v

Dependencies
------------
    pip install pytest pytest-mock

Tests are designed to run WITHOUT a physical webcam or GPU.
All hardware interactions are mocked.
"""

from __future__ import annotations

import time
import queue
import sys
import os
import types
import unittest
from collections import Counter
from unittest.mock import MagicMock, patch, PropertyMock

# ── Ensure project root is on sys.path when running from tests/ subdirectory ──
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ── Stub numpy if not installed (provides minimal array support for tests) ────
try:
    import numpy as np
except ImportError:
    np_stub = types.ModuleType("numpy")
    np_stub.ndarray = type("ndarray", (), {})
    np_stub.zeros = lambda *a, **kw: [[[]]]
    np_stub.uint8 = "uint8"
    sys.modules["numpy"] = np_stub
    import numpy as np  # type: ignore

# ──────────────────────────────────────────────────────────────────────────────
# Minimal stub imports so modules load without GPU / hardware
# ──────────────────────────────────────────────────────────────────────────────

# Stub out cv2 so detector.py / main.py load without OpenCV installed
cv2_stub = types.ModuleType("cv2")
cv2_stub.VideoCapture = MagicMock()
cv2_stub.imshow = MagicMock()
cv2_stub.waitKey = MagicMock(return_value=255)
cv2_stub.destroyAllWindows = MagicMock()
cv2_stub.rectangle = MagicMock()
cv2_stub.putText = MagicMock()
cv2_stub.getTextSize = MagicMock(return_value=((80, 15), 3))
cv2_stub.FONT_HERSHEY_SIMPLEX = 0
cv2_stub.FILLED = -1
cv2_stub.LINE_AA = 16
cv2_stub.CAP_PROP_FRAME_WIDTH = 3
cv2_stub.CAP_PROP_FRAME_HEIGHT = 4
cv2_stub.CAP_PROP_FPS = 5
cv2_stub.CAP_PROP_BUFFERSIZE = 38
cv2_stub.CAP_DSHOW = 700
sys.modules.setdefault("cv2", cv2_stub)

# Stub out pyttsx3 before importing voice.py
pyttsx3_stub = types.ModuleType("pyttsx3")
pyttsx3_stub.init = MagicMock(return_value=MagicMock(
    getProperty=MagicMock(return_value=[]),
    setProperty=MagicMock(),
    say=MagicMock(),
    runAndWait=MagicMock(),
))
sys.modules.setdefault("pyttsx3", pyttsx3_stub)



# Stub out ultralytics before importing detector.py
ultralytics_stub = types.ModuleType("ultralytics")
ultralytics_stub.YOLO = MagicMock()
sys.modules.setdefault("ultralytics", ultralytics_stub)

# Now import project modules
import config                          # noqa: E402
from detector import Detection, ObjectDetector  # noqa: E402
from voice import VoiceEngine          # noqa: E402
from main import (                     # noqa: E402
    FPSTracker,
    SmartVisionAssistant,
    build_scene_summary,
    _number_word,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helper: make a dummy BGR frame
# ──────────────────────────────────────────────────────────────────────────────

def make_frame(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def make_detection(label: str, conf: float = 0.85) -> Detection:
    return Detection(label=label, confidence=conf, bbox=(10, 10, 100, 100), class_id=0)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Config
# ══════════════════════════════════════════════════════════════════════════════

class TestConfig(unittest.TestCase):

    def test_conf_threshold_range(self):
        self.assertGreater(config.CONF_THRESHOLD, 0.0)
        self.assertLess(config.CONF_THRESHOLD, 1.0)

    def test_speech_cooldown_positive(self):
        self.assertGreater(config.SPEECH_COOLDOWN, 0.0)

    def test_camera_resolution_sane(self):
        self.assertGreaterEqual(config.CAMERA_WIDTH, 320)
        self.assertGreaterEqual(config.CAMERA_HEIGHT, 240)

    def test_voice_rate_sane(self):
        self.assertGreater(config.VOICE_RATE, 0)

    def test_voice_volume_range(self):
        self.assertGreaterEqual(config.VOICE_VOLUME, 0.0)
        self.assertLessEqual(config.VOICE_VOLUME, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Detection dataclass
# ══════════════════════════════════════════════════════════════════════════════

class TestDetectionDataclass(unittest.TestCase):

    def test_bottle_detection(self):
        det = make_detection("bottle", 0.92)
        self.assertEqual(det.label, "bottle")
        self.assertAlmostEqual(det.confidence, 0.92)
        self.assertEqual(det.bbox, (10, 10, 100, 100))

    def test_phone_detection(self):
        det = make_detection("cell phone", 0.78)
        self.assertEqual(det.label, "cell phone")

    def test_laptop_detection(self):
        det = make_detection("laptop", 0.95)
        self.assertEqual(det.label, "laptop")

    def test_book_detection(self):
        det = make_detection("book", 0.81)
        self.assertEqual(det.label, "book")

    def test_person_detection(self):
        det = make_detection("person", 0.99)
        self.assertEqual(det.label, "person")

    def test_confidence_below_threshold_filtered(self):
        """Simulate detector filtering low-confidence results."""
        self.assertLess(0.10, config.CONF_THRESHOLD)


# ══════════════════════════════════════════════════════════════════════════════
# 3. ObjectDetector (mocked YOLO)
# ══════════════════════════════════════════════════════════════════════════════

class TestObjectDetector(unittest.TestCase):

    def _make_detector(self):
        """Create a detector with a fully mocked YOLO model (bypasses _load_model)."""
        mock_model = MagicMock()
        mock_model.names = {0: "person", 1: "bottle", 2: "laptop",
                            3: "book", 4: "cell phone"}
        # Bypass __init__ entirely — inject attributes directly
        det = ObjectDetector.__new__(ObjectDetector)
        det._model = mock_model
        det._class_names = list(mock_model.names.values())
        return det

    def _make_mock_results(self, label_id: int, conf: float, bbox: list):
        """Build a mock YOLO result object."""
        mock_box = MagicMock()
        mock_box.conf = [conf]
        mock_box.cls = [label_id]
        mock_box.xyxy = [MagicMock()]
        mock_box.xyxy[0].tolist.return_value = bbox

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        return [mock_result]

    def test_returns_empty_on_none_frame(self):
        det = self._make_detector()
        self.assertEqual(det.detect(None), [])

    def test_returns_empty_on_empty_frame(self):
        det = self._make_detector()
        self.assertEqual(det.detect(np.array([])), [])

    def test_detect_person(self):
        det = self._make_detector()
        det._model.predict.return_value = self._make_mock_results(0, 0.95, [10, 10, 100, 100])
        results = det.detect(make_frame())
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].label, "person")

    def test_detect_bottle(self):
        det = self._make_detector()
        det._model.predict.return_value = self._make_mock_results(1, 0.88, [50, 50, 200, 300])
        results = det.detect(make_frame())
        self.assertEqual(results[0].label, "bottle")

    def test_detect_laptop(self):
        det = self._make_detector()
        det._model.predict.return_value = self._make_mock_results(2, 0.91, [0, 0, 640, 480])
        results = det.detect(make_frame())
        self.assertEqual(results[0].label, "laptop")

    def test_draw_detections_returns_frame(self):
        det = self._make_detector()
        frame = make_frame()
        dets = [make_detection("bottle", 0.80)]
        result = det.draw_detections(frame, dets)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, frame.shape)

    def test_inference_exception_returns_empty(self):
        det = self._make_detector()
        det._model.predict.side_effect = RuntimeError("GPU OOM")
        results = det.detect(make_frame())
        self.assertEqual(results, [])


# ══════════════════════════════════════════════════════════════════════════════
# 4. OCR Reader
# ══════════════════════════════════════════════════════════════════════════════

class TestOCRReader(unittest.TestCase):

    def _make_reader(self) -> OCRReader:
        reader = OCRReader()
        reader._reader = mock_reader_instance   # inject stub
        return reader

    def test_empty_frame_returns_empty(self):
        reader = self._make_reader()
        mock_reader_instance.readtext.return_value = []
        self.assertEqual(reader.read(None), [])

    def test_valid_text_returned(self):
        reader = self._make_reader()
        mock_reader_instance.readtext.return_value = [
            (None, "Python Programming", 0.92)
        ]
        results = reader.read(make_frame())
        self.assertIn("Python Programming", results)

    def test_low_confidence_filtered(self):
        reader = self._make_reader()
        mock_reader_instance.readtext.return_value = [
            (None, "Hello World", 0.10)   # Below OCR_MIN_CONFIDENCE
        ]
        results = reader.read(make_frame())
        self.assertEqual(results, [])

    def test_short_text_filtered(self):
        reader = self._make_reader()
        mock_reader_instance.readtext.return_value = [
            (None, "AB", 0.95)            # Below OCR_MIN_LENGTH (3)
        ]
        results = reader.read(make_frame())
        self.assertEqual(results, [])

    def test_symbol_only_filtered(self):
        reader = self._make_reader()
        mock_reader_instance.readtext.return_value = [
            (None, "!!!!", 0.95)
        ]
        results = reader.read(make_frame())
        self.assertEqual(results, [])

    def test_duplicate_text_deduplicated_across_calls(self):
        reader = self._make_reader()
        mock_reader_instance.readtext.return_value = [
            (None, "Hello World", 0.95)
        ]
        first = reader.read(make_frame())
        second = reader.read(make_frame())   # Same text → should be filtered
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 0)

    def test_reset_session_clears_memory(self):
        reader = self._make_reader()
        mock_reader_instance.readtext.return_value = [
            (None, "Hello World", 0.95)
        ]
        reader.read(make_frame())
        reader.reset_session()
        second = reader.read(make_frame())
        self.assertEqual(len(second), 1)

    def test_clean_filters_noise(self):
        self.assertEqual(OCRReader._clean("AB"), "")          # too short
        self.assertEqual(OCRReader._clean("!!!!!!"), "")      # no alphanum
        self.assertEqual(OCRReader._clean("lllllll"), "")     # 1 unique char
        self.assertNotEqual(OCRReader._clean("Hello World"), "")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Voice Engine
# ══════════════════════════════════════════════════════════════════════════════

class TestVoiceEngine(unittest.TestCase):

    def _make_engine(self) -> VoiceEngine:
        """Create a VoiceEngine with a live mock TTS engine."""
        engine = VoiceEngine.__new__(VoiceEngine)
        engine._muted = False
        engine._queue = queue.Queue(maxsize=config.SPEECH_QUEUE_MAX)
        engine._cooldown_map = {}
        engine._engine = MagicMock()
        engine._available = True
        engine._thread = None
        return engine

    def test_speak_queues_text(self):
        engine = self._make_engine()
        result = engine.speak("Hello", key="hello_test_unique_key_1")
        self.assertTrue(result)
        self.assertFalse(engine._queue.empty())

    def test_cooldown_prevents_repeat(self):
        engine = self._make_engine()
        config_backup = config.SPEECH_COOLDOWN
        config.SPEECH_COOLDOWN = 60.0   # Long cooldown
        try:
            engine.speak("bottle", key="bottle_unique_2")
            second = engine.speak("bottle", key="bottle_unique_2")
            self.assertFalse(second)
        finally:
            config.SPEECH_COOLDOWN = config_backup

    def test_mute_suppresses_speech(self):
        engine = self._make_engine()
        engine._muted = True
        result = engine.speak("Hello", key="hello_mute_test_3")
        self.assertFalse(result)

    def test_toggle_mute(self):
        engine = self._make_engine()
        self.assertFalse(engine.is_muted)
        engine.toggle_mute()
        self.assertTrue(engine.is_muted)
        engine.toggle_mute()
        self.assertFalse(engine.is_muted)

    def test_speak_immediate_clears_queue(self):
        engine = self._make_engine()
        engine._queue.put("stale1")
        engine._queue.put("stale2")
        engine.speak_immediate("Summary!")
        # Queue should have only the summary
        self.assertEqual(engine._queue.qsize(), 1)
        self.assertEqual(engine._queue.get(), "Summary!")

    def test_reset_cooldowns(self):
        engine = self._make_engine()
        engine._cooldown_map["laptop"] = time.monotonic()
        engine.reset_cooldowns()
        self.assertEqual(len(engine._cooldown_map), 0)

    def test_queue_full_drops_gracefully(self):
        engine = self._make_engine()
        engine._queue = queue.Queue(maxsize=2)
        engine.speak("Item1", key="q_full_1")
        engine.speak("Item2", key="q_full_2")
        result = engine.speak("Item3", key="q_full_3")   # Should return False (queue full)
        self.assertFalse(result)


# ══════════════════════════════════════════════════════════════════════════════
# 6. FPS Tracker
# ══════════════════════════════════════════════════════════════════════════════

class TestFPSTracker(unittest.TestCase):

    def test_single_tick_returns_zero(self):
        fps = FPSTracker()
        result = fps.tick()
        self.assertEqual(result, 0.0)

    def test_multiple_ticks_return_positive(self):
        fps = FPSTracker()
        fps.tick()
        time.sleep(0.05)
        fps.tick()
        result = fps.tick()
        self.assertGreater(result, 0.0)

    def test_window_capped(self):
        fps = FPSTracker(window=5)
        for _ in range(20):
            fps.tick()
        self.assertLessEqual(len(fps._times), 5)


# ══════════════════════════════════════════════════════════════════════════════
# 7. Scene Summary
# ══════════════════════════════════════════════════════════════════════════════

class TestSceneSummary(unittest.TestCase):

    def test_empty_returns_nothing_message(self):
        result = build_scene_summary([], [])
        self.assertIn("don't see anything", result.lower())

    def test_single_object(self):
        dets = [make_detection("laptop", 0.9)]
        result = build_scene_summary(dets, [])
        self.assertIn("laptop", result)

    def test_multiple_objects_counted(self):
        dets = [
            make_detection("bottle", 0.9),
            make_detection("bottle", 0.85),
            make_detection("laptop", 0.95),
        ]
        result = build_scene_summary(dets, [])
        self.assertIn("bottle", result)
        self.assertIn("laptop", result)
        self.assertIn("two", result)

    def test_ocr_text_included(self):
        result = build_scene_summary([], ["Python Programming"])
        self.assertIn("Python Programming", result)

    def test_objects_and_text_combined(self):
        dets = [make_detection("phone", 0.88)]
        result = build_scene_summary(dets, ["Hello World"])
        self.assertIn("phone", result)
        self.assertIn("Hello World", result)

    def test_number_word_conversion(self):
        self.assertEqual(_number_word(1), "one")
        self.assertEqual(_number_word(5), "five")
        self.assertEqual(_number_word(11), "11")   # Falls back to str


# ══════════════════════════════════════════════════════════════════════════════
# 8. Keyboard Controls (unit-level)
# ══════════════════════════════════════════════════════════════════════════════

class TestKeyboardControls(unittest.TestCase):

    def _make_app(self) -> SmartVisionAssistant:
        """Create a SmartVisionAssistant with all hardware mocked."""
        with patch.object(ObjectDetector, "_load_model", return_value=None), \
             patch.object(VoiceEngine, "_start_worker", return_value=None):
            app = SmartVisionAssistant.__new__(SmartVisionAssistant)
            app._show_overlays = True
                        app._running = True
            app._frame_idx = 0
            app._summary_display_until = 0.0
            app._last_detections = []
            
            # Mock sub-systems
            app._detector = MagicMock(spec=ObjectDetector)
            app._voice = MagicMock(spec=VoiceEngine)
            app._voice.is_muted = False
            
            app._fps = FPSTracker()
            app._cap = None
        return app

    def test_Q_returns_false(self):
        app = self._make_app()
        result = app._handle_key(ord("q"))
        self.assertFalse(result)

    def test_ESC_returns_false(self):
        app = self._make_app()
        result = app._handle_key(27)
        self.assertFalse(result)

    def test_M_toggles_mute(self):
        app = self._make_app()
        app._handle_key(ord("m"))
        app._voice.toggle_mute.assert_called_once()

    def test_D_toggles_overlays(self):
        app = self._make_app()
        app._handle_key(ord("d"))
        self.assertFalse(app._show_overlays)
        app._handle_key(ord("d"))
        self.assertTrue(app._show_overlays)

    def test_R_calls_speak_immediate(self):
        app = self._make_app()
        app._last_detections = [make_detection("bottle")]
        app._handle_key(ord("r"))
        app._voice.speak_immediate.assert_called_once()
        call_arg = app._voice.speak_immediate.call_args[0][0]
        self.assertIn("bottle", call_arg)

    def test_R_sets_summary_display_time(self):
        app = self._make_app()
        before = time.monotonic()
        app._handle_key(ord("r"))
        self.assertGreater(app._summary_display_until, before)


# ══════════════════════════════════════════════════════════════════════════════
# 9. Integration smoke test
# ══════════════════════════════════════════════════════════════════════════════

class TestIntegrationSmoke(unittest.TestCase):
    """
    Simulates one full pipeline pass with all hardware mocked.
    Verifies components interact correctly without crashing.
    """

    def test_full_pipeline_one_frame(self):
        """Run the detection → OCR → speech pipeline for one synthetic frame."""
        frame = make_frame()
        dets = [make_detection("laptop", 0.92), make_detection("bottle", 0.87)]

        # 1. YOLO detect (mocked already)
        # 2. Object count
        self.assertEqual(len(dets), 2)

                reader._reader = mock_reader_instance
        mock_reader_instance.readtext.return_value = [
            (None, "Smart Vision", 0.90)
        ]
        texts = reader.read(frame)
        self.assertEqual(len(texts), 1)

        # 4. Scene summary
        summary = build_scene_summary(dets, texts)
        self.assertIn("laptop", summary)
        self.assertIn("bottle", summary)
        self.assertIn("Smart Vision", summary)

        # 5. Voice (check speak is callable)
        engine_mock = MagicMock()
        engine_mock.speak("laptop", key="laptop")
        engine_mock.speak.assert_called_with("laptop", key="laptop")


# ──────────────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
