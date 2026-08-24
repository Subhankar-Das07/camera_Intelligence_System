"""Tests for parallel Live Monitor rule evaluation and limits."""

from __future__ import annotations

import time
import unittest
from unittest.mock import patch

from core import site_admin_common as common
from core import site_admin_monitor as monitor


class MaxRulesPerCameraTests(unittest.TestCase):
    @patch("core.site_admin_common.store.list_rules")
    def test_camera_rule_limit_reached(self, mock_rules):
        mock_rules.return_value = [
            {"id": "r1", "camera_id": "cam-a", "enabled": True},
            {"id": "r2", "camera_id": "cam-a", "enabled": True},
            {"id": "r3", "camera_id": "cam-a", "enabled": True},
        ]
        self.assertTrue(common.camera_rule_limit_reached("cam-a"))
        self.assertFalse(common.camera_rule_limit_reached("cam-a", exclude_rule_id="r1"))

    @patch("core.site_admin_common.store.list_rules")
    def test_enabled_rules_for_camera(self, mock_rules):
        mock_rules.return_value = [
            {"id": "r1", "camera_id": "cam-a", "enabled": True},
            {"id": "r2", "camera_id": "cam-b", "enabled": True},
            {"id": "r3", "camera_id": "cam-a", "enabled": False},
        ]
        enabled = common.enabled_rules_for_camera("cam-a")
        self.assertEqual(len(enabled), 1)
        self.assertEqual(enabled[0]["id"], "r1")


class MonitorDetectionDebounceTests(unittest.TestCase):
    def _session(self):
        return {
            "events": [],
            "monitor_debounce": {},
            "event_lock": __import__("threading").Lock(),
            "last_raw_frame": None,
        }

    def test_records_first_detection(self):
        session = self._session()
        rule = {
            "id": "rule-1",
            "name": "Intrusion",
            "scan_type": "intrusion",
            "severity": "high",
            "roi_normalized": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9]],
        }
        event = {"type": "intrusion", "severity": "high"}
        entry = monitor._record_monitor_detection(session, rule, event, frame=None)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.get("event_id"), entry.get("id"))
        self.assertIn("css_color", entry)
        self.assertEqual(len(entry.get("roi_normalized") or []), 3)
        self.assertEqual(len(session["events"]), 1)

    def test_debounce_blocks_repeat_within_window(self):
        session = self._session()
        rule = {"id": "rule-1", "name": "Intrusion", "scan_type": "intrusion"}
        event = {"type": "intrusion"}
        monitor._record_monitor_detection(session, rule, event, frame=None)
        again = monitor._record_monitor_detection(session, rule, event, frame=None)
        self.assertIsNone(again)
        self.assertEqual(len(session["events"]), 1)

    def test_allows_after_debounce_window(self):
        session = self._session()
        session["monitor_debounce"]["rule-1"] = time.time() - common.MONITOR_DEBOUNCE_SEC - 1
        rule = {"id": "rule-1", "name": "Intrusion", "scan_type": "intrusion"}
        event = {"type": "intrusion"}
        entry = monitor._record_monitor_detection(session, rule, event, frame=None)
        self.assertIsNotNone(entry)
        self.assertEqual(len(session["events"]), 1)


class ParallelEvalConfigTests(unittest.TestCase):
    def test_monitor_rule_workers_default(self):
        self.assertGreaterEqual(monitor.MONITOR_RULE_WORKERS, 1)
        self.assertLessEqual(monitor.MONITOR_RULE_WORKERS, 3)


class PreviewPersistStatsTests(unittest.TestCase):
    def test_skip_gate_store_when_preview_without_persist(self):
        session = {"preview_only": True, "persist_stats": False}
        state = monitor._monitor_eval_state(session, 0, None)
        self.assertTrue(state["skip_gate_store"])

    def test_persist_gate_store_when_preview_with_persist(self):
        session = {"preview_only": True, "persist_stats": True}
        state = monitor._monitor_eval_state(session, 0, None)
        self.assertFalse(state["skip_gate_store"])

    def test_preview_only_skips_monitor_detection(self):
        session = {"preview_only": True, "events": [], "monitor_debounce": {}}
        rule = {"id": "r1", "name": "Intrusion", "scan_type": "intrusion"}
        entry = monitor._record_monitor_detection(session, rule, {"type": "intrusion"}, frame=None)
        self.assertIsNone(entry)

    def test_record_preview_event(self):
        session = {
            "preview_only": True,
            "preview_events": [],
            "event_lock": __import__("threading").Lock(),
        }
        rule = {"id": "r1", "name": "Gate", "scan_type": "gate_analytics"}
        monitor._record_preview_event(session, rule, {"type": "gate_near"})
        self.assertEqual(len(session["preview_events"]), 1)
        self.assertIn("css_color", session["preview_events"][0])

    def test_get_preview_status_active_and_inactive(self):
        cam_id = "cam-preview-test"
        monitor._preview_cache[cam_id] = {
            "camera": {"name": "Front"},
            "rules": [{"id": "r1", "name": "Gate", "scan_type": "gate_analytics"}],
            "scan_types": ["gate_analytics"],
            "rule_status": [{"id": "r1", "name": "Gate", "scan_type": "gate_analytics", "state": "running"}],
            "preview_events": [],
            "last_frame_at": time.time(),
            "last_access": time.time(),
            "persist_stats": True,
        }
        try:
            st = monitor.get_preview_status(cam_id)
            self.assertTrue(st["active"])
            self.assertEqual(st["rule_count"], 1)
            self.assertEqual(len(st.get("rule_status") or []), 1)
        finally:
            monitor._preview_cache.pop(cam_id, None)

        missing = monitor.get_preview_status("no-such-camera")
        self.assertFalse(missing["active"])


if __name__ == "__main__":
    unittest.main()
