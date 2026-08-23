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


if __name__ == "__main__":
    unittest.main()
