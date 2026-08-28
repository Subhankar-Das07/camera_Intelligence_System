import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from core.site_admin_verify import (
    _denorm_roi,
    point_in_roi,
    box_overlaps_roi,
    Vote,
    VerifyResult
)

class TestSiteAdminVerify(unittest.TestCase):
    def test_denorm_roi(self):
        # Normalized ROI
        roi = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]
        # 1000x1000 resolution
        px = _denorm_roi(roi, 1000, 1000)
        self.assertEqual(px.shape, (4, 2))
        self.assertEqual(px[0][0], 100.0)
        self.assertEqual(px[2][0], 900.0)

    def test_point_in_roi(self):
        roi_px = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        # Inside
        self.assertTrue(point_in_roi((50, 50), roi_px))
        # Outside
        self.assertFalse(point_in_roi((150, 50), roi_px))

    def test_box_overlaps_roi(self):
        roi_px = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        # Fully inside
        self.assertTrue(box_overlaps_roi([10, 10, 90, 90], roi_px))
        # Fully outside
        self.assertFalse(box_overlaps_roi([150, 150, 200, 200], roi_px))
        # Partially overlapping
        self.assertTrue(box_overlaps_roi([50, 50, 150, 150], roi_px))

    def test_verify_result_dataclass(self):
        res = VerifyResult(ok=True)
        self.assertTrue(res.ok)
        self.assertEqual(res.streak, 0)
        
        v = Vote(name="YOLO", status="pass", score=0.99)
        res.votes.append(v)
        self.assertEqual(len(res.votes), 1)
        self.assertEqual(res.votes[0].name, "YOLO")

    def test_run_verify_graph_no_hints(self):
        from core.site_admin_verify import run_verify_graph, Vote
        payload = {
            "votes": [Vote(name="PersonVerify", status="pass")],
            "streak": 5,
            "rag_hints": {}
        }
        res = run_verify_graph(payload)
        # Should pass because streak (5) >= MIN_FRAMES (default 2), and PersonVerify passed
        self.assertTrue(res["ok"])
        
        # Verify RAGPolicy chip was appended as skip
        rag_vote = next(v for v in res["votes"]["chips"] if v["name"] == "RAGPolicy")
        self.assertEqual(rag_vote["status"], "skip")

    def test_rag_hints_for_camera(self):
        from core.site_admin_verify import rag_hints_for_camera
        hints = rag_hints_for_camera("cam-1")
        self.assertIn("reject_patterns", hints)
        self.assertEqual(hints["camera"], "cam-1")

