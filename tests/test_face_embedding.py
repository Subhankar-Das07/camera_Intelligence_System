"""Unit tests for Face ReID embedding math and matching threshold logic.

These tests exercise the pure math in core.journey_store (_cosine, threshold)
without any Redis, FAISS, or InsightFace dependencies.
"""

from __future__ import annotations

import unittest
import numpy as np
from unittest.mock import patch
from core.journey_store import _cosine, match_or_create_reid


def _unit(dim: int = 512) -> np.ndarray:
    v = np.random.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class CosineSimilarityTests(unittest.TestCase):

    def test_identical_vectors_score_one(self):
        v = _unit()
        score = _cosine(v, v)
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_orthogonal_vectors_score_zero(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        self.assertAlmostEqual(_cosine(a, b), 0.0, places=5)

    def test_opposite_vectors_score_minus_one(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0], dtype=np.float32)
        self.assertAlmostEqual(_cosine(a, b), -1.0, places=5)

    def test_zero_vector_returns_negative_one(self):
        a = np.zeros(512, dtype=np.float32)
        self.assertEqual(_cosine(a, _unit()), -1.0)

    def test_none_vector_returns_negative_one(self):
        self.assertEqual(_cosine(None, _unit()), -1.0)

    def test_high_similarity_above_threshold(self):
        v = _unit()
        noise = np.random.randn(512).astype(np.float32) * 0.01
        v2 = v + noise
        v2 = v2 / np.linalg.norm(v2)
        self.assertGreater(_cosine(v, v2), 0.62)

    def test_random_unrelated_vectors_below_threshold(self):
        np.random.seed(42)
        scores = [_cosine(_unit(), _unit()) for _ in range(20)]
        below = sum(1 for s in scores if s < 0.62)
        self.assertGreaterEqual(below, 18)

    def test_symmetric_property(self):
        a, b = _unit(), _unit()
        self.assertAlmostEqual(_cosine(a, b), _cosine(b, a), places=6)

    def test_dimension_mismatch_handled(self):
        a = np.ones(256, dtype=np.float32) / np.sqrt(256)
        b = np.ones(512, dtype=np.float32) / np.sqrt(512)
        score = _cosine(a, b)
        self.assertIsInstance(score, float)


class MatchOrCreateReidTests(unittest.TestCase):

    def setUp(self):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from tests.test_journeys import _FakeRedis
        self.fake = _FakeRedis()
        self.r_patch = patch("core.journey_store.get_redis", return_value=self.fake)
        self.r_patch.start()

    def tearDown(self):
        self.r_patch.stop()

    def test_first_embedding_mints_new_gid(self):
        result = match_or_create_reid(_unit(), camera_id="cam-1")
        self.assertTrue(result["gid"].startswith("G"))
        self.assertEqual(result.get("match_method"), "new")

    def test_same_embedding_matches_existing_journey(self):
        emb = _unit()
        first = match_or_create_reid(emb, camera_id="cam-1")
        second = match_or_create_reid(emb, camera_id="cam-2")
        self.assertEqual(first["gid"], second["gid"])
        self.assertEqual(second.get("match_method"), "reid")

    def test_empty_embedding_mints_new_gid(self):
        emb = np.array([], dtype=np.float32)
        result = match_or_create_reid(emb, camera_id="cam-1")
        self.assertTrue(result["gid"].startswith("G"))


if __name__ == "__main__":
    unittest.main()
