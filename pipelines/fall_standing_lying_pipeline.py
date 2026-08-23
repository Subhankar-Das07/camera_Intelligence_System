"""
Fall detection — standing & lying (snapshot-friendly).

Alerts when a person was upright on a recent snapshot and is now lying on the floor
in the same area. Designed for HTTP/DVR picture polling (1–3 s ticks), not live video.
"""

from __future__ import annotations

import logging
import time
from math import atan2, degrees
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from shapely.geometry import Point, Polygon
from ultralytics import YOLO

from core.base_pipeline import BaseVideoPipeline
from core.video_source import get_video_source

log = logging.getLogger(__name__)

KP_L_SHOULDER, KP_R_SHOULDER = 5, 6
KP_L_HIP, KP_R_HIP = 11, 12
KP_L_ANKLE, KP_R_ANKLE = 15, 16


def _mid(kxy, kconf, li, ri, thresh):
    l_ok = kconf[li] > thresh
    r_ok = kconf[ri] > thresh
    if l_ok and r_ok:
        return (
            (kxy[li][0] + kxy[ri][0]) / 2.0,
            (kxy[li][1] + kxy[ri][1]) / 2.0,
        )
    if l_ok:
        return (float(kxy[li][0]), float(kxy[li][1]))
    if r_ok:
        return (float(kxy[ri][0]), float(kxy[ri][1]))
    return None


def torso_angle(hip: Tuple[float, float], shoulder: Tuple[float, float]) -> float:
    dx = abs(shoulder[0] - hip[0])
    dy = abs(hip[1] - shoulder[1]) + 1e-6
    return degrees(atan2(dx, dy))


def classify_posture(
    t_angle: Optional[float],
    hip: Tuple[float, float],
    ankle: Tuple[float, float],
    baseline_h: float,
    cfg: Dict[str, Any],
) -> str:
    """Return 'upright', 'lying', or 'unknown'."""
    sit_angle = float(cfg.get("sit_angle_deg", 30.0))
    down_angle = float(cfg.get("down_torso_angle_deg", 40.0))
    down_ratio = float(cfg.get("down_hip_ankle_ratio", 0.45))

    if t_angle is not None and t_angle < sit_angle:
        return "upright"

    hip_ankle_gap = ankle[1] - hip[1]
    hip_near_floor = baseline_h > 0 and hip_ankle_gap < down_ratio * baseline_h
    torso_flat = t_angle is not None and t_angle > down_angle
    torso_upright = t_angle is not None and t_angle < sit_angle

    if torso_flat or (hip_near_floor and not torso_upright):
        return "lying"
    return "unknown"


def _prune_upright_history(history: List[Dict[str, Any]], now: float, memory_sec: float) -> None:
    cutoff = now - memory_sec
    while history and history[0].get("t", 0) < cutoff:
        history.pop(0)


def _nearest_upright(
    history: List[Dict[str, Any]],
    cx: float,
    cy: float,
    max_dist: float,
    now: float,
    memory_sec: float,
) -> Optional[Dict[str, Any]]:
    best = None
    best_d = max_dist
    cutoff = now - memory_sec
    for entry in history:
        t = entry.get("t", 0)
        if t < cutoff:
            continue
        dx = entry["cx"] - cx
        dy = entry["cy"] - cy
        d = (dx * dx + dy * dy) ** 0.5
        if d < best_d:
            best_d = d
            best = entry
    return best


def _default_transition_state() -> Dict[str, Any]:
    return {
        "upright_history": [],
        "lying_streak": 0,
        "pending_match": None,
    }


class FallStandingLyingPipeline(BaseVideoPipeline):
    DEFAULT_CONFIG = {
        "person_conf_threshold": 0.70,
        "keypoint_conf_threshold": 0.4,
        "sit_angle_deg": 30.0,
        "down_torso_angle_deg": 40.0,
        "down_hip_ankle_ratio": 0.45,
        "upright_memory_sec": 45.0,
        "match_dist_ratio": 0.25,
        "confirm_snapshots": 2,
    }

    def initialize(self, model_weight: str = "yolov8n-pose.pt"):
        self.model = YOLO(model_weight)

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        roi_polygon: np.ndarray,
        config: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        cfg = {**self.DEFAULT_CONFIG, **(config or {})}
        ts = config.get("timestamp") or time.time()
        rule_state = config.get("transition_state")
        if rule_state is None:
            rule_state = _default_transition_state()
        if "upright_history" not in rule_state:
            rule_state.update(_default_transition_state())

        h, w = frame.shape[:2]
        out = frame.copy()
        meta: Dict[str, Any] = {}

        roi_poly = None
        if roi_polygon is not None and len(roi_polygon) >= 3:
            roi_poly = Polygon(roi_polygon)

        kp_thresh = float(cfg["keypoint_conf_threshold"])
        conf_thresh = float(cfg["person_conf_threshold"])
        memory_sec = float(cfg["upright_memory_sec"])
        match_dist = float(cfg["match_dist_ratio"]) * w
        confirm = int(cfg["confirm_snapshots"])

        results = self.model(out, classes=[0], conf=conf_thresh, verbose=False)[0]
        if results.keypoints is None or results.boxes is None:
            rule_state["lying_streak"] = 0
            rule_state["pending_match"] = None
            return out, meta

        boxes_xywh = results.boxes.xywh.cpu().numpy()
        kpts_xy = results.keypoints.xy.cpu().numpy()
        kpts_conf = results.keypoints.conf.cpu().numpy()

        upright_now: List[Dict[str, Any]] = []
        lying_now: List[Dict[str, Any]] = []

        for i, (bx, by, bw, bh) in enumerate(boxes_xywh):
            if bw <= 0 or bh <= 0:
                continue
            kxy = kpts_xy[i]
            kconf = kpts_conf[i]
            hip = _mid(kxy, kconf, KP_L_HIP, KP_R_HIP, kp_thresh)
            if hip is None:
                continue
            shoulder = _mid(kxy, kconf, KP_L_SHOULDER, KP_R_SHOULDER, kp_thresh)
            ankle = _mid(kxy, kconf, KP_L_ANKLE, KP_R_ANKLE, kp_thresh)
            if ankle is None:
                ankle = (float(bx), float(by + bh / 2.0))

            cx, cy = float(bx), float(by)
            if roi_poly is not None and not roi_poly.contains(Point(cx, cy)):
                continue

            t_angle = torso_angle(hip, shoulder) if shoulder is not None else None
            posture = classify_posture(t_angle, hip, ankle, float(bh), cfg)
            x1, y1 = int(bx - bw / 2), int(by - bh / 2)
            x2, y2 = int(bx + bw / 2), int(by + bh / 2)

            if posture == "upright":
                upright_now.append({"cx": cx, "cy": cy, "t": ts, "bh": float(bh)})
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(
                    out, "UPRIGHT", (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2,
                )
            elif posture == "lying":
                lying_now.append({"cx": cx, "cy": cy, "bh": float(bh)})
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 140, 255), 2)
                cv2.putText(
                    out, "LYING", (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2,
                )

        history: List[Dict[str, Any]] = rule_state["upright_history"]
        _prune_upright_history(history, ts, memory_sec)
        history.extend(upright_now)

        matched = False
        for person in lying_now:
            entry = _nearest_upright(history, person["cx"], person["cy"], match_dist, ts, memory_sec)
            if entry is not None:
                matched = True
                rule_state["pending_match"] = {"cx": entry["cx"], "cy": entry["cy"]}
                break

        if matched:
            rule_state["lying_streak"] = int(rule_state.get("lying_streak") or 0) + 1
        else:
            rule_state["lying_streak"] = 0
            rule_state["pending_match"] = None

        if rule_state["lying_streak"] >= confirm:
            pm = rule_state.get("pending_match") or (lying_now[0] if lying_now else None)
            meta = {
                "type": "fall_standing_lying",
                "severity": "SEVERE",
                "message": "Person was standing, now lying",
            }
            rule_state["lying_streak"] = 0
            rule_state["pending_match"] = None
            if pm:
                cx, cy = int(pm.get("cx", w // 2)), int(pm.get("cy", h // 2))
            else:
                cx, cy = w // 2, h // 2
            cv2.putText(
                out, "FALL: STANDING->LYING", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3,
            )
            cv2.circle(out, (cx, cy), 12, (0, 0, 255), -1)

        if roi_poly is not None and len(roi_polygon) >= 3:
            cv2.polylines(out, [np.asarray(roi_polygon, dtype=np.int32)], True, (0, 255, 255), 2)

        config["transition_state"] = rule_state
        return out, meta

    def run_on_video(self, input_path, output_dir, roi_normalized, config):
        cfg = {**self.DEFAULT_CONFIG, **(config or {})}
        transition_state = config.get("transition_state") if config else None
        if transition_state is None:
            transition_state = _default_transition_state()

        cap = get_video_source(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

        roi_np = np.array([], dtype=np.int32)
        if roi_normalized and len(roi_normalized) >= 3:
            roi_np = np.array(
                [[int(x * width), int(y * height)] for x, y in roi_normalized],
                dtype=np.int32,
            )

        frame_idx = 0
        frame_cfg = {**cfg, "transition_state": transition_state}
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            annotated, event = self.process_frame(frame, frame_idx, roi_np, frame_cfg)
            yield annotated, event if event else None
            frame_idx += 1
        cap.release()
