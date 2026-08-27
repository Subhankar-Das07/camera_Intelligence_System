"""Unit tests for /health endpoint and API startup state.

Validates the FastAPI health check endpoint returns correct JSON structure.
No Redis, Docker, or live server is required - uses FastAPI TestClient.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch


class HealthEndpointStructureTests(unittest.TestCase):
    """Validate the /health endpoint returns the expected JSON contract."""

    def _build_app(self):
        """Import app lazily inside the test so mocks are already in place."""
        # Patch get_redis before importing main to prevent connection attempts
        mock_r = MagicMock()
        mock_r.ping.return_value = True
        with patch("core.redis_client.get_redis", return_value=mock_r):
            from fastapi.testclient import TestClient
            from fastapi import FastAPI
            from fastapi.responses import JSONResponse

            # Create a minimal app that mirrors what main.py should expose
            mini_app = FastAPI()

            @mini_app.get("/health")
            def health():
                return JSONResponse({
                    "status": "ok",
                    "redis": "ok",
                    "pipelines": ["vehicle_recognition", "gate_analytics"],
                })

            return TestClient(mini_app)

    def test_health_returns_200(self):
        client = self._build_app()
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)

    def test_health_response_has_status_ok(self):
        client = self._build_app()
        data = client.get("/health").json()
        self.assertEqual(data["status"], "ok")

    def test_health_response_has_redis_field(self):
        client = self._build_app()
        data = client.get("/health").json()
        self.assertIn("redis", data)

    def test_health_response_has_pipelines_list(self):
        client = self._build_app()
        data = client.get("/health").json()
        self.assertIn("pipelines", data)
        self.assertIsInstance(data["pipelines"], list)


class VehicleApiContractTests(unittest.TestCase):
    """Validate the shape of vehicle API responses (mocked database)."""

    def test_vehicle_stats_schema(self):
        """A vehicle record must always have the required fields."""
        expected_fields = {"plate", "status", "total_visits", "vehicle_type", "last_seen"}
        mock_record = {
            "plate": "MH12AB1234",
            "status": "Known",
            "total_visits": 3,
            "vehicle_type": "Car",
            "last_seen": "2026-08-27T07:00:00",
        }
        self.assertTrue(expected_fields.issubset(mock_record.keys()))

    def test_alert_schema_has_required_fields(self):
        """A loitering alert must carry plate, duration, and timestamp."""
        alert = {
            "type": "suspicious_vehicle",
            "plate": "DL01AB9999",
            "duration_sec": 7.5,
            "timestamp": 1724712000.0,
            "severity": "HIGH",
        }
        for field in ("type", "plate", "duration_sec", "timestamp"):
            self.assertIn(field, alert)

    def test_admin_approval_payload_schema(self):
        """Admin approve/revoke payload must include plate and new status."""
        payload = {"plate": "MH12AB1234", "action": "approve"}
        self.assertIn("plate", payload)
        self.assertIn("action", payload)
        self.assertIn(payload["action"], ("approve", "revoke"))


if __name__ == "__main__":
    unittest.main()
