"""Tests for gate analytics reporting — window totals, daily buckets, footfall."""

from __future__ import annotations

import time
import unittest
from typing import Any, Dict
from unittest.mock import patch

from core import site_admin_store as store


class _FakePipeline:
    def __init__(self, redis: "_FakeRedis") -> None:
        self._redis = redis
        self._ops: list = []

    def hincrby(self, key: str, field: str, amount: int) -> None:
        self._ops.append(("hincrby", key, field, amount))

    def execute(self) -> None:
        for op in self._ops:
            _, key, field, amount = op
            self._redis.hincrby(key, field, amount)


class _FakeRedis:
    def __init__(self) -> None:
        self.hashes: Dict[str, Dict[Any, Any]] = {}

    def pipeline(self) -> _FakePipeline:
        return _FakePipeline(self)

    def hincrby(self, key: str, field: str, amount: int) -> None:
        bucket = self.hashes.setdefault(key, {})
        raw = bucket.get(field.encode()) or bucket.get(field) or 0
        bucket[field] = str(int(raw) + int(amount)).encode()

    def hgetall(self, key: str) -> Dict[Any, Any]:
        return dict(self.hashes.get(key, {}))


class GateReportStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = _FakeRedis()
        self.redis_patcher = patch("core.site_admin_store.get_redis", return_value=self.fake)
        self.redis_patcher.start()

    def tearDown(self) -> None:
        self.redis_patcher.stop()

    def test_increment_writes_daily_and_hourly(self) -> None:
        rule_id = "rule-gate-1"
        store.increment_gate_counters(rule_id, {"persons_in": 2, "cars_out": 1})
        day = time.strftime("%Y%m%d", time.localtime())
        hour = time.strftime("%Y%m%d%H", time.localtime())
        self.assertIn(f"gate:daily:{rule_id}:{day}", self.fake.hashes)
        self.assertIn(f"gate:hourly:{rule_id}:{hour}", self.fake.hashes)
        self.assertIn(f"gate:totals:{rule_id}", self.fake.hashes)

    def test_window_totals_not_lifetime(self) -> None:
        rule_id = "rule-gate-2"
        now = time.time()
        hour_bucket = time.strftime("%Y%m%d%H", time.localtime(now))
        day_bucket = time.strftime("%Y%m%d", time.localtime(now))
        self.fake.hashes[f"gate:hourly:{rule_id}:{hour_bucket}"] = {
            b"persons_in": b"3",
            b"persons_out": b"1",
        }
        self.fake.hashes[f"gate:daily:{rule_id}:{day_bucket}"] = {
            b"persons_in": b"3",
            b"persons_out": b"1",
        }
        self.fake.hashes[f"gate:totals:{rule_id}"] = {
            b"persons_in": b"999",
            b"persons_out": b"888",
        }

        since = now - 3600
        with patch("core.site_admin_store.list_rules") as mock_rules:
            mock_rules.return_value = [
                {
                    "id": rule_id,
                    "name": "Main gate",
                    "camera_id": "cam-1",
                    "scan_type": "gate_analytics",
                }
            ]
            report = store.gate_report_summary(since, now, rule_id)

        totals = report["totals"]
        self.assertEqual(totals["persons_in"], 3)
        self.assertEqual(totals["persons_out"], 1)
        self.assertEqual(totals["footfall"], 4)
        self.assertNotEqual(totals["persons_in"], 999)

    def test_gate_report_by_period_weekly_includes_daily(self) -> None:
        rule_id = "rule-gate-3"
        now = time.time()
        day_bucket = time.strftime("%Y%m%d", time.localtime(now))
        hour_bucket = time.strftime("%Y%m%d%H", time.localtime(now))
        self.fake.hashes[f"gate:daily:{rule_id}:{day_bucket}"] = {
            b"persons_in": b"5",
            b"persons_out": b"2",
        }
        self.fake.hashes[f"gate:hourly:{rule_id}:{hour_bucket}"] = {
            b"persons_in": b"5",
            b"persons_out": b"2",
        }

        with patch("core.site_admin_store.list_rules") as mock_rules:
            mock_rules.return_value = [
                {
                    "id": rule_id,
                    "name": "Side gate",
                    "camera_id": "cam-2",
                    "scan_type": "gate_analytics",
                }
            ]
            report = store.gate_report_by_period("weekly", rule_id=rule_id)

        self.assertEqual(report["period"], "weekly")
        daily = report.get("daily") or []
        self.assertTrue(any(row.get("date") == day_bucket for row in daily))
        self.assertEqual(report["totals"]["footfall"], 7)


if __name__ == "__main__":
    unittest.main()
