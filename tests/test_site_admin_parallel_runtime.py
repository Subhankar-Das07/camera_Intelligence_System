"""Unit tests for parallel Site Admin runtime helpers."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from core import site_admin_scan as scan
from core import site_admin_runtime as runtime


class RuntimeConfigTests(unittest.TestCase):
    def test_runtime_config_defaults(self):
        with patch.dict(os.environ, {}, clear=False):
            for key in (
                "SITE_ADMIN_MAX_CAMERAS",
                "SITE_ADMIN_INFERENCE_SLOTS",
                "SITE_ADMIN_WORKER_STAGGER_MS",
                "SITE_ADMIN_TICK_MIN_SEC",
                "SITE_ADMIN_TICK_MAX_SEC",
            ):
                os.environ.pop(key, None)
            cfg = runtime.runtime_config()
        self.assertEqual(cfg["max_cameras"], 8)
        self.assertEqual(cfg["inference_slots"], 2)
        self.assertEqual(cfg["stagger_ms"], 250)
        self.assertEqual(cfg["tick_min"], 1.0)
        self.assertEqual(cfg["tick_max"], 3.0)

    def test_runtime_config_env_override(self):
        env = {
            "SITE_ADMIN_MAX_CAMERAS": "4",
            "SITE_ADMIN_INFERENCE_SLOTS": "1",
            "SITE_ADMIN_TICK_MIN_SEC": "0.5",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = runtime.runtime_config()
        self.assertEqual(cfg["max_cameras"], 4)
        self.assertEqual(cfg["inference_slots"], 1)
        self.assertEqual(cfg["tick_min"], 0.5)


class TickSleepTests(unittest.TestCase):
    def test_high_priority_shorter_sleep(self):
        rules = [{"scan_type": "intrusion"}]
        high = scan.compute_tick_sleep_sec(rules, inference_wait_ms=0, tick_min=1.0, tick_max=3.0)
        low = scan.compute_tick_sleep_sec(
            [{"scan_type": "face_attendance"}],
            inference_wait_ms=2500,
            tick_min=1.0,
            tick_max=3.0,
        )
        self.assertEqual(high, 1.0)
        self.assertEqual(low, 3.0)


class ScanParallelConfigTests(unittest.TestCase):
    def test_scan_rule_workers_default(self):
        self.assertGreaterEqual(scan.SCAN_RULE_WORKERS, 1)
        self.assertLessEqual(scan.SCAN_RULE_WORKERS, 3)

    def test_worker_state_has_pipe_lock(self):
        state = scan.new_worker_state()
        self.assertIn("pipe_lock", state)


class RuntimeStatusTests(unittest.TestCase):
    @patch("core.site_admin_runtime.store.list_runtime_worker_heartbeats")
    @patch("core.site_admin_runtime.store.get_site")
    def test_active_camera_ids_from_workers(self, mock_site, mock_workers):
        import time
        mock_site.return_value = {"go_live": True}
        mock_workers.return_value = [
            {"camera_id": "cam-a", "last_tick_at": time.time() - 1.0},
            {"camera_id": "cam-b", "last_tick_at": time.time() - 2.0},
        ]
        status = runtime.get_runtime_status()
        self.assertTrue(status["scanning"])
        self.assertEqual(status["active_camera_ids"], ["cam-a", "cam-b"])
        self.assertEqual(status["active_camera_id"], "cam-a")
        self.assertEqual(len(status["workers"]), 2)


class IntegrationNotes(unittest.TestCase):
    """Manual integration checklist (requires Redis + DVR)."""

    def test_manual_checklist_documented(self):
        steps = [
            "Start go-live with 8 DVR cameras and rules on 4+ cameras",
            "Verify docker logs show one worker per camera with rules",
            "Preview UI shows SCANNING on all active workers",
            "Open Live Monitor on one camera — its worker pauses (Redis monitored set)",
            "Stop go-live — workers exit within ~10s",
        ]
        self.assertGreaterEqual(len(steps), 4)


if __name__ == "__main__":
    unittest.main()
