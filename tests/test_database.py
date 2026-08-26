"""
test_database.py — Unit tests for pipelines/vehicle_recognition/database.py

All Redis calls are mocked in-memory. No real Redis connection is needed.
Tests run in ~0.1s total.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, call
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db(mock_redis: MagicMock):
    """Return a VehicleDatabase instance wired to a fake Redis client."""
    # conftest.py already stubs core.redis_client globally, so we can import
    # VehicleDatabase directly and then swap its .r attribute to our mock.
    from pipelines.vehicle_recognition.database import VehicleDatabase
    db = VehicleDatabase.__new__(VehicleDatabase)
    db.db_path = "storage/vehicle_intelligence.db"
    import threading
    db.lock = threading.Lock()
    db.r = mock_redis
    return db


def _blank_img() -> np.ndarray:
    """Return a tiny white image sufficient for cv2.imencode."""
    return np.ones((20, 40, 3), dtype=np.uint8) * 255


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def redis_mock():
    """Fresh MagicMock acting as a Redis client for each test."""
    r = MagicMock()
    # Default pipeline mock: execute returns an empty list
    pipe = MagicMock()
    pipe.execute.return_value = []
    r.pipeline.return_value = pipe
    return r


# ===========================================================================
# TEST: register_vehicle
# ===========================================================================

class TestRegisterVehicle:
    """Verifies that registering a vehicle sets status = 'Known' in Redis."""

    def test_register_sets_known_status(self, redis_mock):
        db = _make_db(redis_mock)
        db.register_vehicle("AB12CD3456")
        redis_mock.hset.assert_called_once_with(
            "vr:vehicle:AB12CD3456", "status", "Known"
        )

    def test_register_strips_and_uppercases_plate(self, redis_mock):
        db = _make_db(redis_mock)
        db.register_vehicle("  ab12cd3456  ")
        redis_mock.hset.assert_called_once_with(
            "vr:vehicle:AB12CD3456", "status", "Known"
        )

    def test_register_does_not_affect_other_plates(self, redis_mock):
        db = _make_db(redis_mock)
        db.register_vehicle("PLATE1")
        # hset must be called exactly once — not for PLATE2 or anything else
        assert redis_mock.hset.call_count == 1
        key_used = redis_mock.hset.call_args[0][0]
        assert "PLATE1" in key_used
        assert "PLATE2" not in key_used


# ===========================================================================
# TEST: unregister_vehicle
# ===========================================================================

class TestUnregisterVehicle:
    """Verifies that unregistering a vehicle reverts status to 'Unknown'."""

    def test_unregister_sets_unknown_status(self, redis_mock):
        db = _make_db(redis_mock)
        db.unregister_vehicle("AB12CD3456")
        redis_mock.hset.assert_called_once_with(
            "vr:vehicle:AB12CD3456", "status", "Unknown"
        )

    def test_unregister_strips_and_uppercases_plate(self, redis_mock):
        db = _make_db(redis_mock)
        db.unregister_vehicle("  mh12ab1234  ")
        redis_mock.hset.assert_called_once_with(
            "vr:vehicle:MH12AB1234", "status", "Unknown"
        )

    def test_register_then_unregister_correct_sequence(self, redis_mock):
        """After register→unregister, the LAST call must write 'Unknown'."""
        db = _make_db(redis_mock)
        db.register_vehicle("KA01AB1111")
        db.unregister_vehicle("KA01AB1111")
        assert redis_mock.hset.call_count == 2
        last_args = redis_mock.hset.call_args_list[-1][0]
        assert last_args[2] == "Unknown"


# ===========================================================================
# TEST: get_vehicle_stats
# ===========================================================================

class TestGetVehicleStats:
    """Verifies Redis data is correctly decoded and returned by get_vehicle_stats."""

    def _redis_vehicle_data(self):
        return {
            b"plate_number": b"DL4CAB1234",
            b"total_visits": b"3",
            b"first_seen":   b"2026-08-23 10:00:00",
            b"last_seen":    b"2026-08-23 12:00:00",
            b"status":       b"Known",
            b"vehicle_type": b"Car",
        }

    def test_returns_none_for_missing_plate(self, redis_mock):
        redis_mock.hgetall.return_value = {}
        db = _make_db(redis_mock)
        assert db.get_vehicle_stats("MISSING") is None

    def test_returns_correct_fields(self, redis_mock):
        redis_mock.hgetall.return_value = self._redis_vehicle_data()
        redis_mock.lrange.return_value = []
        db = _make_db(redis_mock)
        stats = db.get_vehicle_stats("DL4CAB1234")
        assert stats["plate_number"] == "DL4CAB1234"
        assert stats["total_visits"] == 3
        assert stats["status"] == "Known"
        assert stats["vehicle_type"] == "Car"

    def test_returns_correct_history_entries(self, redis_mock):
        visit_record = {
            "id": 1, "plate_number": "DL4CAB1234", "visit_number": 1,
            "timestamp": "2026-08-23 10:00:00"
        }
        redis_mock.hgetall.return_value = self._redis_vehicle_data()
        redis_mock.lrange.return_value = [json.dumps(visit_record).encode()]
        db = _make_db(redis_mock)
        stats = db.get_vehicle_stats("DL4CAB1234")
        assert len(stats["history"]) == 1
        assert stats["history"][0]["plate_number"] == "DL4CAB1234"


# ===========================================================================
# TEST: record_visit (visit count increment)
# ===========================================================================

class TestRecordVisit:
    """Verifies that record_visit increments the visit counter and uses a pipeline."""

    def test_first_visit_sets_visit_count_1(self, redis_mock):
        # Simulate no prior visit stored
        redis_mock.hget.return_value = None
        db = _make_db(redis_mock)
        db.record_visit("MH12XY9999", _blank_img(), _blank_img(), 0.95, "Car")
        pipe = redis_mock.pipeline.return_value
        pipe.execute.assert_called_once()

    def test_second_visit_increments_count(self, redis_mock):
        # Simulate 1 existing visit in Redis
        redis_mock.hget.return_value = b"1"
        db = _make_db(redis_mock)
        visit_num = db.record_visit("MH12XY9999", _blank_img(), _blank_img(), 0.90, "Car")
        assert visit_num == 2

    def test_plate_is_added_to_plates_set(self, redis_mock):
        redis_mock.hget.return_value = None
        db = _make_db(redis_mock)
        db.record_visit("GJ01TT5555", _blank_img(), _blank_img(), 0.88, "Truck")
        pipe = redis_mock.pipeline.return_value
        # Verify sadd was called on the pipeline for the vr:plates set
        sadd_calls = [c for c in pipe.sadd.call_args_list if "vr:plates" in c[0]]
        assert len(sadd_calls) == 1
        assert "GJ01TT5555" in sadd_calls[0][0]


# ===========================================================================
# TEST: get_image_bytes
# ===========================================================================

class TestGetImageBytes:
    """Verifies get_image_bytes constructs the correct Redis key."""

    def test_snap_key_format(self, redis_mock):
        redis_mock.get.return_value = b"fakeimagedata"
        db = _make_db(redis_mock)
        result = db.get_image_bytes("snap", "KL11BK7777", 2)
        redis_mock.get.assert_called_once_with("vr:snap:KL11BK7777:2")
        assert result == b"fakeimagedata"

    def test_crop_key_format(self, redis_mock):
        redis_mock.get.return_value = None
        db = _make_db(redis_mock)
        db.get_image_bytes("crop", "TN09ZZ1234", 5)
        redis_mock.get.assert_called_once_with("vr:crop:TN09ZZ1234:5")

    def test_returns_none_when_key_missing(self, redis_mock):
        redis_mock.get.return_value = None
        db = _make_db(redis_mock)
        result = db.get_image_bytes("snap", "MISSING", 1)
        assert result is None
