import unittest
from core.loitering_trajectory import (
    path_length,
    displacement,
    tortuosity,
    mean_speed_px_s,
    normalize_loiter_config,
    loitering_decision,
)

class TestLoiteringTrajectoryMetrics(unittest.TestCase):
    def test_path_length_empty_or_single(self):
        self.assertEqual(path_length([]), 0.0)
        self.assertEqual(path_length([(10.0, 10.0)]), 0.0)

    def test_path_length_multiple_points(self):
        # 3,4,5 triangle logic: (0,0) -> (3,4) -> (6,8)
        points = [(0.0, 0.0), (3.0, 4.0), (6.0, 8.0)]
        self.assertAlmostEqual(path_length(points), 10.0)

    def test_displacement_empty_or_single(self):
        self.assertEqual(displacement([]), 0.0)
        self.assertEqual(displacement([(10.0, 10.0)]), 0.0)

    def test_displacement_multiple_points(self):
        # Starts at (0,0), ends at (10, 0) despite moving around
        points = [(0.0, 0.0), (5.0, 5.0), (10.0, 0.0)]
        self.assertAlmostEqual(displacement(points), 10.0)

    def test_tortuosity(self):
        # Straight line: length = displacement, tortuosity = 1.0
        straight = [(0.0, 0.0), (10.0, 0.0)]
        self.assertAlmostEqual(tortuosity(straight), 1.0)
        
        # Wandering path: displacement = 0 (ends where it starts), tortuosity > 1
        wandering = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)]
        # Length is 40. Displacement is 0. Falls back to length / 1.0 = 40.0
        self.assertAlmostEqual(tortuosity(wandering), 40.0)

    def test_mean_speed(self):
        points = [(0.0, 0.0), (10.0, 0.0)]
        times = [100.0, 102.0]
        # Length = 10. Time = 2s. Speed = 5 px/s
        self.assertAlmostEqual(mean_speed_px_s(points, times), 5.0)
        
        # Empty inputs
        self.assertEqual(mean_speed_px_s([], []), 0.0)

class TestLoiteringConfig(unittest.TestCase):
    def test_normalize_loiter_config_defaults(self):
        cfg = normalize_loiter_config({})
        self.assertEqual(cfg["dwell_sec"], 20.0)
        self.assertEqual(cfg["min_tortuosity"], 1.8)
        self.assertEqual(cfg["max_speed"], 45.0)

    def test_normalize_loiter_config_overrides(self):
        cfg = normalize_loiter_config({
            "loiter_config": {
                "dwell_sec": 15.0,
                "min_tortuosity": 2.5,
                "max_speed": 30.0
            }
        })
        self.assertEqual(cfg["dwell_sec"], 15.0)
        self.assertEqual(cfg["min_tortuosity"], 2.5)
        self.assertEqual(cfg["max_speed"], 30.0)

    def test_normalize_loiter_config_clamps(self):
        cfg = normalize_loiter_config({
            "dwell_sec": 1000.0, # Clamped to 600
            "min_tortuosity": 0.1, # Clamped to 1.0
            "max_speed": 1000.0 # Clamped to 500
        })
        self.assertEqual(cfg["dwell_sec"], 600.0)
        self.assertEqual(cfg["min_tortuosity"], 1.0)
        self.assertEqual(cfg["max_speed"], 500.0)

class TestLoiteringDecision(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "dwell_sec": 10.0,
            "min_tortuosity": 2.0,
            "max_speed": 50.0
        }

    def test_straight_walk_through_no_alert(self):
        # Walks straight through quickly (length 100, disp 100, tort 1.0, time 1s -> speed 100)
        points = [(0.0, 0.0), (100.0, 0.0)]
        times = [1.0, 2.0]
        # Dwell is long enough
        decision = loitering_decision(11.0, points, times, self.cfg)
        self.assertFalse(decision["triggered"])

    def test_wandering_triggers_alert(self):
        # Tortuous path, length 40, disp 0, tort 40, time 11s -> speed 3.6
        points = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)]
        times = [1.0, 3.0, 5.0, 7.0, 12.0]
        decision = loitering_decision(11.0, points, times, self.cfg)
        self.assertTrue(decision["triggered"])
        self.assertTrue(decision["dwell_ok"])
        self.assertTrue(decision["tort_ok"])
        self.assertTrue(decision["slow_ok"])

    def test_slow_straight_walk_triggers_alert(self):
        # Length 10, time 11s -> speed < 1.0
        points = [(0.0, 0.0), (10.0, 0.0)]
        times = [1.0, 12.0]
        decision = loitering_decision(11.0, points, times, self.cfg)
        self.assertTrue(decision["triggered"])
        self.assertTrue(decision["dwell_ok"])
        self.assertFalse(decision["tort_ok"]) # Tortuosity is 1.0
        self.assertTrue(decision["slow_ok"]) # But it's very slow

    def test_insufficient_dwell_no_alert(self):
        points = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)]
        times = [1.0, 3.0, 5.0, 7.0, 9.0]
        decision = loitering_decision(8.0, points, times, self.cfg)
        self.assertFalse(decision["triggered"])
        self.assertFalse(decision["dwell_ok"])
