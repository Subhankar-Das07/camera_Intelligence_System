"""Tests for cross-camera journey store — match, merge, handoff window."""

from __future__ import annotations

import time
import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import numpy as np

from core import journey_store as js


class _FakeRedis:
    def __init__(self) -> None:
        self.kv: Dict[Any, Any] = {}
        self.lists: Dict[str, List[Any]] = {}
        self.zsets: Dict[str, Dict[str, float]] = {}
        self.sets: Dict[str, set] = {}
        self.ttls: Dict[str, int] = {}

    def incr(self, key: str) -> int:
        k = key if isinstance(key, str) else key.decode()
        n = int(self.kv.get(k, 0) or 0) + 1
        self.kv[k] = n
        return n

    def get(self, key: str) -> Optional[bytes]:
        k = key if isinstance(key, str) else key.decode()
        v = self.kv.get(k)
        if v is None:
            return None
        if isinstance(v, bytes):
            return v
        if isinstance(v, str):
            return v.encode()
        if isinstance(v, (int, float)):
            return str(v).encode()
        return v

    def set(self, key: str, value: Any) -> None:
        k = key if isinstance(key, str) else key.decode()
        self.kv[k] = value if isinstance(value, bytes) else (
            value.encode() if isinstance(value, str) else value
        )

    def setex(self, key: str, ttl: int, value: Any) -> None:
        self.set(key, value)
        self.ttls[key if isinstance(key, str) else key.decode()] = ttl

    def exists(self, key: str) -> int:
        k = key if isinstance(key, str) else key.decode()
        return 1 if k in self.kv else 0

    def delete(self, *keys: str) -> None:
        for key in keys:
            k = key if isinstance(key, str) else key.decode()
            self.kv.pop(k, None)

    def lpush(self, key: str, value: Any) -> None:
        k = key if isinstance(key, str) else key.decode()
        self.lists.setdefault(k, []).insert(0, value)

    def ltrim(self, key: str, start: int, end: int) -> None:
        k = key if isinstance(key, str) else key.decode()
        lst = self.lists.get(k, [])
        self.lists[k] = lst[start : end + 1]

    def lrange(self, key: str, start: int, end: int) -> List[Any]:
        k = key if isinstance(key, str) else key.decode()
        lst = self.lists.get(k, [])
        if end == -1:
            return list(lst[start:])
        return list(lst[start : end + 1])

    def zadd(self, key: str, mapping: Dict[str, float]) -> None:
        k = key if isinstance(key, str) else key.decode()
        bucket = self.zsets.setdefault(k, {})
        for member, score in mapping.items():
            m = member if isinstance(member, str) else member.decode()
            bucket[m] = float(score)

    def zrem(self, key: str, member: str) -> None:
        k = key if isinstance(key, str) else key.decode()
        m = member if isinstance(member, str) else member.decode()
        self.zsets.get(k, {}).pop(m, None)

    def zrevrangebyscore(self, key: str, max_score: float, min_score: float) -> List[bytes]:
        k = key if isinstance(key, str) else key.decode()
        items = [
            (m, s)
            for m, s in self.zsets.get(k, {}).items()
            if min_score <= s <= max_score
        ]
        items.sort(key=lambda x: x[1], reverse=True)
        return [m.encode() for m, _ in items]

    def sismember(self, key: str, member: str) -> bool:
        k = key if isinstance(key, str) else key.decode()
        m = member if isinstance(member, str) else member.decode()
        return m in self.sets.get(k, set())

    def sadd(self, key: str, *members: str) -> int:
        k = key if isinstance(key, str) else key.decode()
        bucket = self.sets.setdefault(k, set())
        n = 0
        for member in members:
            m = member if isinstance(member, str) else member.decode()
            if m not in bucket:
                bucket.add(m)
                n += 1
        return n

    def srem(self, key: str, *members: str) -> int:
        k = key if isinstance(key, str) else key.decode()
        bucket = self.sets.get(k, set())
        n = 0
        for member in members:
            m = member if isinstance(member, str) else member.decode()
            if m in bucket:
                bucket.remove(m)
                n += 1
        return n

    def smembers(self, key: str):
        k = key if isinstance(key, str) else key.decode()
        return {m.encode() for m in self.sets.get(k, set())}


class JourneyStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = _FakeRedis()
        self.patcher = patch("core.journey_store.get_redis", return_value=self.fake)
        self.patcher.start()
        # Disable cooldown so successive sightings append
        self.cd = patch("core.journey_store.SIGHTING_COOLDOWN_SEC", 0)
        self.cd.start()
        # Clear cooldown exists always false when ttl 0 — force exists to 0 for cd keys
        orig_exists = self.fake.exists

        def _exists(key: str) -> int:
            k = key if isinstance(key, str) else key.decode()
            if k.startswith("journey:cd:"):
                return 0
            return orig_exists(key)

        self.fake.exists = _exists  # type: ignore

    def tearDown(self) -> None:
        self.cd.stop()
        self.patcher.stop()

    def test_face_upsert_stable_gid(self) -> None:
        a = js.upsert_face_sighting(
            person_id="P0001",
            label="Ayush",
            camera_id="cam-a",
            camera_name="Cam A",
        )
        b = js.upsert_face_sighting(
            person_id="P0001",
            label="Ayush",
            camera_id="cam-b",
            camera_name="Cam B",
        )
        self.assertEqual(a["gid"], b["gid"])
        detail = js.get_journey(a["gid"])
        self.assertIsNotNone(detail)
        self.assertIn("Cam A", detail["hop_summary"])
        self.assertIn("Cam B", detail["hop_summary"])

    def test_plate_upsert(self) -> None:
        with patch("core.site_admin_store.normalize_plate", side_effect=lambda p: str(p).upper()):
            m = js.upsert_plate_sighting(plate="mh12ab1234", camera_id="cam-1", camera_name="Gate")
        self.assertEqual(m["kind"], "vehicle")
        self.assertTrue(m["gid"].startswith("G"))

    def test_reid_match_within_window(self) -> None:
        emb = np.ones(512, dtype=np.float32)
        emb = emb / np.linalg.norm(emb)
        first = js.match_or_create_reid(emb, camera_id="cam-1", camera_name="A")
        second = js.match_or_create_reid(emb, camera_id="cam-2", camera_name="B")
        self.assertEqual(first["gid"], second["gid"])
        self.assertEqual(second.get("match_method"), "reid")

    def test_reid_rejects_outside_handoff(self) -> None:
        emb = np.random.randn(512).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        first = js.match_or_create_reid(emb, camera_id="cam-1", handoff_sec=1.0)
        # Age the active score beyond window
        self.fake.zsets[js.ACTIVE_KEY][first["gid"]] = time.time() - 120
        other = np.random.randn(512).astype(np.float32)
        other = other / np.linalg.norm(other)
        # Same embedding but outside window should mint new if only old candidate
        third = js.match_or_create_reid(emb, camera_id="cam-2", handoff_sec=1.0)
        self.assertNotEqual(first["gid"], third["gid"])

    def test_merge_anonymous_into_face(self) -> None:
        emb = np.ones(512, dtype=np.float32)
        emb /= np.linalg.norm(emb)
        anon = js.match_or_create_reid(emb, camera_id="cam-1", camera_name="A")
        face = js.upsert_face_sighting(
            person_id="P9",
            label="Staff",
            camera_id="cam-2",
            camera_name="B",
            merge_from_gid=anon["gid"],
        )
        self.assertEqual(face["gid"], anon["gid"])
        self.assertEqual(face.get("person_id"), "P9")
        self.assertEqual(face.get("label"), "Staff")

    def test_list_journeys(self) -> None:
        js.upsert_face_sighting(person_id="P2", label="X", camera_id="c1", camera_name="C1")
        items = js.list_journeys(limit=10, active_minutes=60)
        self.assertGreaterEqual(len(items), 1)

    def test_watch_emits_alert_on_hop(self) -> None:
        alerts: List[Dict[str, Any]] = []

        def _append(alert: Dict[str, Any]) -> Dict[str, Any]:
            alerts.append(alert)
            return alert

        with patch("core.site_admin_store.append_alert", side_effect=_append):
            meta = js.upsert_face_sighting(
                person_id="PW",
                label="WatchMe",
                camera_id="cam-a",
                camera_name="Cam A",
            )
            gid = meta["gid"]
            js.watch_journey(gid)
            self.assertTrue(js.is_watched(gid))
            self.assertGreaterEqual(len(alerts), 1)
            js.upsert_face_sighting(
                person_id="PW",
                label="WatchMe",
                camera_id="cam-b",
                camera_name="Cam B",
            )
            hop_msgs = [a for a in alerts if a.get("type") == "journey_hop"]
            self.assertGreaterEqual(len(hop_msgs), 1)
            self.assertIn("Cam B", hop_msgs[-1]["message"])


if __name__ == "__main__":
    unittest.main()
