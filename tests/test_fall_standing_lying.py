"""Unit tests for fall_standing_lying snapshot fall detection."""

from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from core import site_admin_scan as scan
from pipelines.fall_standing_lying_pipeline import (
    FallStandingLyingPipeline,
    _default_transition_state,
    _nearest_upright,
    classify_posture,
    torso_angle,
)


class PostureHelperTests(unittest.TestCase):
    def test_torso_upright(self):
        hip = (100.0, 200.0)
        shoulder = (100.0, 100.0)
        angle = torso_angle(hip, shoulder)
        self.assertLess(angle, 15.0)

    def test_torso_lying(self):
        hip = (100.0, 200.0)
        shoulder = (200.0, 200.0)
        angle = torso_angle(hip, shoulder)
        self.assertGreater(angle, 80.0)

    def test_classify_upright(self):
        cfg = {"sit_angle_deg": 30.0, "down_torso_angle_deg": 40.0, "down_hip_ankle_ratio": 0.45}
        result = classify_posture(10.0, (100, 200), (100, 350), 150.0, cfg)
        self.assertEqual(result, "upright")

    def test_classify_lying_flat_torso(self):
        cfg = {"sit_angle_deg": 30.0, "down_torso_angle_deg": 40.0, "down_hip_ankle_ratio": 0.45}
        result = classify_posture(70.0, (100, 200), (100, 350), 150.0, cfg)
        self.assertEqual(result, "lying")

    def test_classify_lying_hip_near_floor(self):
        cfg = {"sit_angle_deg": 30.0, "down_torso_angle_deg": 40.0, "down_hip_ankle_ratio": 0.45}
        result = classify_posture(35.0, (100, 300), (100, 320), 150.0, cfg)
        self.assertEqual(result, "lying")


class NearestUprightTests(unittest.TestCase):
    def test_finds_nearby_entry(self):
        now = 1000.0
        history = [{"cx": 50.0, "cy": 50.0, "t": now - 5, "bh": 120.0}]
        match = _nearest_upright(history, 55.0, 52.0, max_dist=50.0, now=now, memory_sec=45.0)
        self.assertIsNotNone(match)
        self.assertEqual(match["cx"], 50.0)

    def test_rejects_distant_entry(self):
        now = 1000.0
        history = [{"cx": 50.0, "cy": 50.0, "t": now - 5, "bh": 120.0}]
        match = _nearest_upright(history, 300.0, 300.0, max_dist=50.0, now=now, memory_sec=45.0)
        self.assertIsNone(match)

    def test_rejects_expired_entry(self):
        now = 1000.0
        history = [{"cx": 50.0, "cy": 50.0, "t": now - 60, "bh": 120.0}]
        match = _nearest_upright(history, 50.0, 50.0, max_dist=50.0, now=now, memory_sec=45.0)
        self.assertIsNone(match)


def _mock_pose_result(persons):
    """Build a mock ultralytics result with boxes + keypoints."""
    result = MagicMock()
    if not persons:
        result.keypoints = None
        result.boxes = None
        return result

    boxes = MagicMock()
    boxes.xywh.cpu.return_value.numpy.return_value = np.array(
        [[p["cx"], p["cy"], p["bw"], p["bh"]] for p in persons],
        dtype=np.float32,
    )
    kpts = MagicMock()
    kxy = []
    kconf = []
    for p in persons:
        xy = np.zeros((17, 2), dtype=np.float32)
        conf = np.zeros(17, dtype=np.float32)
        hip_y = p.get("hip_y", p["cy"])
        shoulder_y = p.get("shoulder_y", p["cy"] - p["bh"] * 0.4)
        ankle_y = p.get("ankle_y", p["cy"] + p["bh"] * 0.4)
        for idx in (5, 6, 11, 12, 15, 16):
            conf[idx] = 0.9
        # If bw > bh (horizontal posture), spread keypoints in X axis
        if p["bw"] > p["bh"]:
            dx_shoulder = p["bw"] * 0.3
            dx_hip = -p["bw"] * 0.3
        else:
            dx_shoulder = 0
            dx_hip = 0

        xy[5] = [p["cx"] + dx_shoulder - 10, shoulder_y]
        xy[6] = [p["cx"] + dx_shoulder + 10, shoulder_y]
        xy[11] = [p["cx"] + dx_hip - 10, hip_y]
        xy[12] = [p["cx"] + dx_hip + 10, hip_y]
        xy[15] = [p["cx"] + dx_hip - 10, ankle_y]
        xy[16] = [p["cx"] + dx_hip + 10, ankle_y]
        kxy.append(xy)
        kconf.append(conf)
    kpts.xy.cpu.return_value.numpy.return_value = np.array(kxy)
    kpts.conf.cpu.return_value.numpy.return_value = np.array(kconf)
    result.boxes = boxes
    result.keypoints = kpts
    return result


class TransitionStateMachineTests(unittest.TestCase):
    def setUp(self):
        self.pipe = FallStandingLyingPipeline()
        self.pipe.model = MagicMock()
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.cfg_base = {
            **FallStandingLyingPipeline.DEFAULT_CONFIG,
            "confirm_snapshots": 2,
            "transition_state": _default_transition_state(),
        }

    def test_upright_then_lying_fires_after_confirm(self):
        upright = {
            "cx": 320.0, "cy": 240.0, "bw": 80.0, "bh": 160.0,
            "hip_y": 280.0, "shoulder_y": 180.0, "ankle_y": 360.0,
        }
        lying = {
            "cx": 320.0, "cy": 380.0, "bw": 160.0, "bh": 80.0,
            "hip_y": 400.0, "shoulder_y": 400.0, "ankle_y": 420.0,
        }
        self.pipe.model.return_value = [_mock_pose_result([upright])]
        cfg = {**self.cfg_base, "timestamp": 1000.0}
        _, meta1 = self.pipe.process_frame(self.frame, 0, np.array([]), cfg)
        self.assertEqual(meta1.get("type"), None)

        self.pipe.model.return_value = [_mock_pose_result([lying])]
        cfg["timestamp"] = 1002.0
        _, meta2 = self.pipe.process_frame(self.frame, 1, np.array([]), cfg)
        self.assertEqual(meta2.get("type"), None)
        
        cfg["timestamp"] = 1004.0
        _, meta3 = self.pipe.process_frame(self.frame, 2, np.array([]), cfg)
        self.assertEqual(meta3.get("type"), "fall_standing_lying")

    def test_lying_without_prior_upright_no_alert(self):
        lying = {
            "cx": 320.0, "cy": 380.0, "bw": 160.0, "bh": 80.0,
            "hip_y": 400.0, "shoulder_y": 400.0, "ankle_y": 420.0,
        }
        self.pipe.model.return_value = [_mock_pose_result([lying])]
        cfg = {**self.cfg_base, "timestamp": 1000.0}
        for i in range(3):
            cfg["timestamp"] = 1000.0 + i
            _, meta = self.pipe.process_frame(self.frame, i, np.array([]), cfg)
            self.assertNotEqual(meta.get("type"), "fall_standing_lying")

    def test_spatial_mismatch_no_alert(self):
        upright = {
            "cx": 100.0, "cy": 240.0, "bw": 80.0, "bh": 160.0,
            "hip_y": 280.0, "shoulder_y": 180.0, "ankle_y": 360.0,
        }
        lying_far = {
            "cx": 500.0, "cy": 380.0, "bw": 160.0, "bh": 80.0,
            "hip_y": 400.0, "shoulder_y": 400.0, "ankle_y": 420.0,
        }
        self.pipe.model.return_value = [_mock_pose_result([upright])]
        cfg = {**self.cfg_base, "timestamp": 1000.0}
        self.pipe.process_frame(self.frame, 0, np.array([]), cfg)

        self.pipe.model.return_value = [_mock_pose_result([lying_far])]
        for i in range(3):
            cfg["timestamp"] = 1002.0 + i
            _, meta = self.pipe.process_frame(self.frame, i + 1, np.array([]), cfg)
            self.assertNotEqual(meta.get("type"), "fall_standing_lying")


class PriorityWeightTests(unittest.TestCase):
    def test_fall_standing_lying_high_priority_band(self):
        rules = [{"scan_type": "fall_standing_lying"}]
        sleep_sec = scan.compute_tick_sleep_sec(rules, inference_wait_ms=0, tick_min=1.0, tick_max=3.0)
        self.assertEqual(sleep_sec, 2.0)
        self.assertEqual(scan.priority_weight("fall_standing_lying"), 24)


if __name__ == "__main__":
    unittest.main()
