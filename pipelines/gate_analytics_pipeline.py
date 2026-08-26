"""Gate analytics: proximity zones, line crossing counts, gate open/close detection."""

from __future__ import annotations

import logging
from typing import Any, Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np
from shapely.geometry import Point, Polygon
from ultralytics import YOLO

from core.base_pipeline import BaseVideoPipeline
from core.video_source import get_video_source

logger = logging.getLogger(__name__)

_DETECT_CLASSES = [0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorcycle, bus, truck
_CAR_CLASSES = {2, 5, 7}
_BIKE_CLASSES = {1, 3}
_PERSON_CLASS = 0

_ZONE_COLORS = {
    "near": (0, 0, 255),
    "medium": (0, 165, 255),
    "far": (0, 200, 80),
}

_COUNTER_KEYS = (
    "gate_opens",
    "gate_closes",
    "persons_in",
    "persons_out",
    "cars_in",
    "cars_out",
    "bikes_in",
    "bikes_out",
    "near_events",
    "medium_events",
    "far_events",
)


def _empty_counters() -> Dict[str, int]:
    return {k: 0 for k in _COUNTER_KEYS}


def _denorm_poly(pts: List, w: int, h: int) -> np.ndarray:
    return np.array([[int(x * w), int(y * h)] for x, y in pts], dtype=np.int32)


def _denorm_line(line: List, w: int, h: int) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    p0 = (float(line[0][0]) * w, float(line[0][1]) * h)
    p1 = (float(line[1][0]) * w, float(line[1][1]) * h)
    return p0, p1


def _side_of_line(pt: Tuple[float, float], p0: Tuple[float, float], p1: Tuple[float, float]) -> str:
    cross = (p1[0] - p0[0]) * (pt[1] - p0[1]) - (p1[1] - p0[1]) * (pt[0] - p0[0])
    return "left" if cross > 0 else "right"


def _foot_point(x1: int, y1: int, x2: int, y2: int) -> Tuple[float, float]:
    return ((x1 + x2) / 2.0, float(y2))


def _classify(cls_id: int) -> str:
    if cls_id == _PERSON_CLASS:
        return "person"
    if cls_id in _BIKE_CLASSES:
        return "bike"
    if cls_id in _CAR_CLASSES:
        return "car"
    return "other"


def _roi_edge_score(frame: np.ndarray, roi_pts: np.ndarray) -> float:
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [roi_pts], 255)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    region = edges[mask > 0]
    if region.size == 0:
        return 0.0
    return float(np.mean(region))


def _band_for_point(
    pt: Tuple[float, float],
    zones: Dict[str, List],
    w: int,
    h: int,
) -> Optional[str]:
    for band in ("near", "medium", "far"):
        poly_pts = zones.get(band) or []
        if len(poly_pts) < 3:
            continue
        poly = Polygon(_denorm_poly(poly_pts, w, h))
        if poly.contains(Point(pt)):
            return band
    return None


class GateAnalyticsPipeline(BaseVideoPipeline):
    """YOLO + ByteTrack gate counting and zone analytics."""

    def __init__(self) -> None:
        self.detector: Optional[YOLO] = None
        self._rule_state: Dict[str, Dict[str, Any]] = {}

    def initialize(self, vehicle_weights: str = "yolov8n.pt", **kwargs) -> None:
        self.detector = YOLO(vehicle_weights)
        logger.info("GateAnalyticsPipeline initialized with %s", vehicle_weights)

    def _state(self, rule_id: str) -> Dict[str, Any]:
        if rule_id not in self._rule_state:
            self._rule_state[rule_id] = {
                "track_side": {},
                "crossed": set(),
                "track_band": {},
                "journey_labels": {},
                "gate_state": "unknown",
                "baseline_score": None,
                "baseline_samples": [],
                "session_totals": _empty_counters(),
                "live": {"gate_state": "unknown", "near_count": 0, "medium_count": 0, "far_count": 0},
            }
        return self._rule_state[rule_id]

    def reset_baseline(self, rule_id: str) -> None:
        st = self._state(rule_id)
        st["baseline_score"] = None
        st["baseline_samples"] = []
        st["gate_state"] = "unknown"

    def set_baseline_score(self, rule_id: str, score: float) -> None:
        st = self._state(rule_id)
        st["baseline_score"] = score
        st["baseline_samples"] = [score]

    def get_live(self, rule_id: str) -> Dict[str, Any]:
        st = self._state(rule_id)
        live = dict(st.get("live") or {})
        live["session_totals"] = dict(st.get("session_totals") or _empty_counters())
        return live

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
        roi_polygon: Optional[np.ndarray] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        config = config or {}
        gate_config = config.get("gate_config") or {}
        rule_id = config.get("rule_id") or "default"
        direction_in = gate_config.get("direction_in") or "left"

        meta: Dict[str, Any] = {"type": "gate_tick", "counters": _empty_counters(), "live": {}}
        if frame is None or frame.size == 0 or self.detector is None:
            return frame, meta

        h, w = frame.shape[:2]
        out = frame.copy()
        st = self._state(rule_id)
        deltas = _empty_counters()
        count_line = gate_config.get("count_line") or []
        gate_roi = gate_config.get("gate_roi") or []
        zones = gate_config.get("distance_zones") or {}

        if len(count_line) == 2:
            p0, p1 = _denorm_line(count_line, w, h)
            cv2.line(out, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), (255, 255, 0), 2)
            cv2.putText(out, "COUNT LINE", (int(p0[0]), int(p0[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

        for band, color in _ZONE_COLORS.items():
            pts_list = zones.get(band) or []
            if len(pts_list) >= 3:
                pts = _denorm_poly(pts_list, w, h)
                overlay = out.copy()
                cv2.fillPoly(overlay, [pts], color)
                cv2.addWeighted(overlay, 0.25, out, 0.75, 0, out)
                cv2.polylines(out, [pts], True, color, 2)
                cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
                cv2.putText(out, band.upper(), (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if len(gate_roi) >= 3:
            gpts = _denorm_poly(gate_roi, w, h)
            cv2.polylines(out, [gpts], True, (200, 100, 255), 2)
            score = _roi_edge_score(frame, gpts)
            baseline = st.get("baseline_score")
            if baseline is None:
                samples: List[float] = st["baseline_samples"]
                samples.append(score)
                if len(samples) >= 30:
                    st["baseline_score"] = float(np.mean(samples))
                    st["gate_state"] = "closed"
                elif config.get("force_baseline") is not None:
                    st["baseline_score"] = float(config["force_baseline"])
                    st["gate_state"] = "closed"
            else:
                prev = st.get("gate_state") or "closed"
                open_thresh = baseline * 1.25
                close_thresh = baseline * 1.08
                if prev in ("closed", "unknown") and score > open_thresh:
                    st["gate_state"] = "open"
                    deltas["gate_opens"] += 1
                elif prev == "open" and score < close_thresh:
                    st["gate_state"] = "closed"
                    deltas["gate_closes"] += 1

        band_counts = {"near": 0, "medium": 0, "far": 0}
        person_tracks: List[Dict[str, Any]] = []
        journey_labels: Dict[int, str] = dict(st.get("journey_labels") or {})

        try:
            results = self.detector.track(
                frame,
                persist=True,
                classes=_DETECT_CLASSES,
                tracker="bytetrack.yaml",
                verbose=False,
            )
            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)

                active = set(int(t) for t in track_ids)
                for tid in list(st["track_side"].keys()):
                    if tid not in active:
                        st["track_side"].pop(tid, None)
                        st["track_band"].pop(tid, None)
                        journey_labels.pop(tid, None)

                for box, tid, cls_id in zip(boxes, track_ids, cls_ids):
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                    tid = int(tid)
                    kind = _classify(int(cls_id))
                    if kind == "other":
                        continue

                    foot = _foot_point(x1, y1, x2, y2)
                    cx, cy = int(foot[0]), int(foot[1])

                    if kind == "person":
                        person_tracks.append(
                            {
                                "track_id": tid,
                                "xyxy": [x1, y1, x2, y2],
                                "kind": kind,
                            }
                        )

                    if kind == "person" and zones:
                        band = _band_for_point(foot, zones, w, h)
                        if band:
                            band_counts[band] += 1
                            prev_band = st["track_band"].get(tid)
                            if prev_band != band:
                                st["track_band"][tid] = band
                                deltas[f"{band}_events"] += 1
                                if band == "near":
                                    meta["band"] = "near"
                                    meta["alert_near"] = st.get("gate_state") == "closed"
                            cv2.putText(out, band.upper(), (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _ZONE_COLORS[band], 2)

                    if len(count_line) == 2:
                        p0, p1 = _denorm_line(count_line, w, h)
                        side = _side_of_line(foot, p0, p1)
                        prev_side = st["track_side"].get(tid)
                        if prev_side and prev_side != side:
                            cross_key = (tid, side)
                            if cross_key not in st["crossed"]:
                                st["crossed"].add(cross_key)
                                is_in = side == direction_in
                                suffix = "in" if is_in else "out"
                                if kind == "person":
                                    deltas[f"persons_{suffix}"] += 1
                                elif kind == "car":
                                    deltas[f"cars_{suffix}"] += 1
                                elif kind == "bike":
                                    deltas[f"bikes_{suffix}"] += 1
                        st["track_side"][tid] = side

                    color = (0, 220, 60) if kind == "person" else (255, 180, 0)
                    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                    cv2.circle(out, (cx, cy), 4, color, -1)
                    jlabel = journey_labels.get(tid) or ""
                    tag = f"{kind}:{tid}"
                    if jlabel:
                        tag = f"{jlabel} {tag}"
                    cv2.putText(out, tag, (x1, y2 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                st["journey_labels"] = journey_labels
        except Exception as e:
            logger.warning("gate analytics frame error: %s", e)

        meta["person_tracks"] = person_tracks

        for k, v in deltas.items():
            st["session_totals"][k] = st["session_totals"].get(k, 0) + v

        gate_state = st.get("gate_state") or "unknown"
        st["live"] = {
            "gate_state": gate_state,
            "near_count": band_counts["near"],
            "medium_count": band_counts["medium"],
            "far_count": band_counts["far"],
        }
        meta["counters"] = deltas
        meta["live"] = dict(st["live"])
        meta["live"]["session_totals"] = dict(st["session_totals"])

        badge = f"Gate: {gate_state.upper()}"
        cv2.rectangle(out, (8, 8), (220, 36), (0, 0, 0), -1)
        cv2.putText(out, badge, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_off = 58
        tot = st["session_totals"]
        for label, a, b in [
            ("P in/out", "persons_in", "persons_out"),
            ("Cars in/out", "cars_in", "cars_out"),
            ("Bikes in/out", "bikes_in", "bikes_out"),
        ]:
            txt = f"{label}: {tot.get(a, 0)}/{tot.get(b, 0)}"
            cv2.putText(out, txt, (12, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1)
            y_off += 18

        return out, meta

    def run_on_video(
        self,
        input_path: str,
        output_dir: str,
        roi_normalized=None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Generator[Tuple[np.ndarray, Optional[Dict[str, Any]]], None, None]:
        config = config or {}
        cap = get_video_source(input_path)
        if not cap.isOpened():
            logger.error("Cannot open video source: %s", input_path)
            return
        frame_idx = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                annotated, meta = self.process_frame(frame, frame_idx, None, config)
                yield annotated, meta
                frame_idx += 1
        finally:
            cap.release()
