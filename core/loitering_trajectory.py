"""Loitering via trajectory analysis (RS-WACV24-inspired metrics).

Based on: Núñez et al., "Identifying Loitering Behavior With Trajectory Analysis",
WACV Workshops 2024 — dwell time, pace, and path directionality/tortuosity.
https://github.com/johnnynunez/RS-WACV24_Loitering

We adopt the metric definitions for RGB CCTV; we do not vendor the thermal dataset.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_DWELL_SEC = float(os.environ.get("SITE_ADMIN_LOITER_DWELL_SEC", "20"))
DEFAULT_MIN_TORTUOSITY = float(os.environ.get("SITE_ADMIN_LOITER_MIN_TORTUOSITY", "1.8"))
DEFAULT_MAX_SPEED = float(os.environ.get("SITE_ADMIN_LOITER_MAX_SPEED", "45"))  # px/s


def path_length(points: Sequence[Tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(points)):
        dx = points[i][0] - points[i - 1][0]
        dy = points[i][1] - points[i - 1][1]
        total += math.hypot(dx, dy)
    return total


def displacement(points: Sequence[Tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    dx = points[-1][0] - points[0][0]
    dy = points[-1][1] - points[0][1]
    return math.hypot(dx, dy)


def tortuosity(points: Sequence[Tuple[float, float]]) -> float:
    """Path length / net displacement. High values ≈ wandering / loitering."""
    disp = displacement(points)
    length = path_length(points)
    if disp < 1.0:
        return length / 1.0 if length > 0 else 0.0
    return length / disp


def mean_speed_px_s(points: Sequence[Tuple[float, float]], times: Sequence[float]) -> float:
    if len(points) < 2 or len(times) < 2:
        return 0.0
    length = path_length(points)
    dt = max(1e-3, float(times[-1]) - float(times[0]))
    return length / dt


def normalize_loiter_config(rule: Dict[str, Any]) -> Dict[str, float]:
    raw = rule.get("loiter_config") if isinstance(rule.get("loiter_config"), dict) else {}
    try:
        dwell = float(raw.get("dwell_sec", rule.get("dwell_sec", DEFAULT_DWELL_SEC)))
    except (TypeError, ValueError):
        dwell = DEFAULT_DWELL_SEC
    try:
        tort = float(raw.get("min_tortuosity", rule.get("min_tortuosity", DEFAULT_MIN_TORTUOSITY)))
    except (TypeError, ValueError):
        tort = DEFAULT_MIN_TORTUOSITY
    try:
        max_spd = float(raw.get("max_speed", rule.get("max_speed", DEFAULT_MAX_SPEED)))
    except (TypeError, ValueError):
        max_spd = DEFAULT_MAX_SPEED
    return {
        "dwell_sec": max(3.0, min(600.0, dwell)),
        "min_tortuosity": max(1.0, min(50.0, tort)),
        "max_speed": max(5.0, min(500.0, max_spd)),
    }


def loitering_decision(
    dwell_sec: float,
    points: Sequence[Tuple[float, float]],
    times: Sequence[float],
    cfg: Dict[str, float],
) -> Dict[str, Any]:
    """
    Alert when person remains in ROI long enough AND
    (path is tortuous OR pace is low) — WACV-style trajectory cues.
    """
    tort = tortuosity(points)
    speed = mean_speed_px_s(points, times)
    dwell_ok = dwell_sec >= float(cfg["dwell_sec"])
    tort_ok = tort >= float(cfg["min_tortuosity"])
    slow_ok = speed <= float(cfg["max_speed"])
    # Walking straight through: high speed + low tortuosity → no alert even if briefly over dwell
    triggered = bool(dwell_ok and (tort_ok or slow_ok))
    return {
        "triggered": triggered,
        "dwell_sec": round(dwell_sec, 2),
        "tortuosity": round(tort, 3),
        "speed_px_s": round(speed, 2),
        "dwell_ok": dwell_ok,
        "tort_ok": tort_ok,
        "slow_ok": slow_ok,
        "path_points": len(points),
    }
