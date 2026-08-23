"""Single-frame rule evaluation for parallel go-live camera workers."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from shapely.geometry import Point, Polygon

from core import site_admin_common as common
from core import site_admin_store as store
from core.registry import registry
from core.video_source import get_video_source

log = logging.getLogger("site_admin.scan")

_pose_model = None


class FrameFeed:
    """Capture-like source for fall-detection generator ticks."""

    def __init__(self, width: int = 640, height: int = 480, fps: float = 30.0):
        self.frame: Optional[np.ndarray] = None
        self._w = width
        self._h = height
        self._fps = fps
        self._opened = True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.frame is None:
            return False, None
        return True, self.frame

    def isOpened(self) -> bool:
        return self._opened and self.frame is not None

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._w)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._h)
        if prop_id == cv2.CAP_PROP_FPS:
            return float(self._fps)
        return 0.0

    def release(self) -> None:
        self._opened = False


def new_worker_state() -> Dict[str, Any]:
    return {
        "frame_idx": 0,
        "capture": None,
        "source": None,
        "pipelines": {},
        "generators": {},
        "frame_feeds": {},
        "fall_sl": {},
        "pipelines_initialized": False,
        "width": 640,
        "height": 480,
        "fps": 30.0,
    }


def _denorm_roi(roi_normalized: List, width: int, height: int) -> np.ndarray:
    pts = [[int(x * width), int(y * height)] for x, y in roi_normalized]
    return np.array(pts, dtype=np.int32)


def _get_pose_model():
    global _pose_model
    if _pose_model is None:
        from ultralytics import YOLO

        _pose_model = YOLO("yolov8n-pose.pt")
    return _pose_model


def _vehicle_event(meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for det in meta.get("detections") or []:
        plate = det.get("plate")
        if plate:
            return {"plate": plate, "type": "vehicle", "severity": "medium"}
    return None


def _ensure_capture(cam: Dict[str, Any], state: Dict[str, Any]):
    source = common.input_for_camera(cam)
    if not source:
        return None
    if state.get("source") != source:
        cap = state.get("capture")
        if cap is not None and hasattr(cap, "release"):
            try:
                cap.release()
            except Exception:
                pass
        state["source"] = source
        state["capture"] = get_video_source(source)
        state["pipelines_initialized"] = False
    cap = state.get("capture")
    if cap is None or not cap.isOpened():
        return None
    return cap


def read_camera_frame(cam: Dict[str, Any], state: Dict[str, Any]) -> Optional[np.ndarray]:
    cap = _ensure_capture(cam, state)
    if cap is None:
        return None
    ret, frame = cap.read()
    if not ret or frame is None:
        return None
    h, w = frame.shape[:2]
    state["width"] = w
    state["height"] = h
    if not state.get("pipelines_initialized"):
        _init_pipelines(state, cam, frame)
    return frame


def _init_pipelines(state: Dict[str, Any], cam: Dict[str, Any], frame: np.ndarray) -> None:
    h, w = frame.shape[:2]
    state["width"] = w
    state["height"] = h
    fps = float(state.get("fps") or 30.0)
    rules = [
        r
        for r in store.list_rules()
        if r.get("camera_id") == cam.get("id") and r.get("enabled", True)
    ]
    scan_types = {r.get("scan_type") for r in rules if r.get("scan_type")}
    for scan_type in scan_types:
        pipeline_name = common.SCAN_TO_PIPELINE.get(scan_type or "")
        if not pipeline_name:
            continue
        if scan_type in ("face_attendance", "vehicle", "gate_analytics", "fall_standing_lying"):
            if scan_type not in state["pipelines"]:
                try:
                    state["pipelines"][scan_type] = registry.get_pipeline(pipeline_name)
                except Exception as e:
                    log.warning("pipeline %s: %s", pipeline_name, e)
        elif scan_type == "fall":
            first = next((r for r in rules if r.get("scan_type") == "fall"), None)
            roi = (first or {}).get("roi_normalized") or []
            try:
                feed = FrameFeed(w, h, fps)
                pipe = registry.get_pipeline(pipeline_name)
                gen = pipe.run_on_video(
                    feed,
                    common.ALERTS_DIR,
                    roi,
                    common.pipeline_config("fall"),
                )
                state["frame_feeds"]["fall"] = feed
                state["generators"]["fall"] = gen
            except Exception as e:
                log.warning("fall pipeline init: %s", e)
    state["pipelines_initialized"] = True


def build_pose_cache(frame: np.ndarray) -> Dict[str, Any]:
    """Single YOLO pose pass; share across intrusion/danger_zone rules."""
    model = _get_pose_model()
    results = model(frame, classes=[0], verbose=False)[0]
    cache: Dict[str, Any] = {"results": results, "keypoints_xy": None, "keypoints_conf": None}
    if results.keypoints is not None:
        cache["keypoints_xy"] = results.keypoints.xy.cpu().numpy()
        cache["keypoints_conf"] = results.keypoints.conf.cpu().numpy()
    return cache


def _evaluate_pose_from_keypoints(
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray,
    rule: Dict[str, Any],
    scan_type: str,
    width: int,
    height: int,
) -> Optional[Dict[str, Any]]:
    roi = rule.get("roi_normalized") or []
    if len(roi) < 3:
        return None
    roi_poly = Polygon(_denorm_roi(roi, width, height))
    machine_active = scan_type == "danger_zone"
    buffered = roi_poly.buffer(20) if machine_active else roi_poly

    for person_kpts, conf in zip(keypoints_xy, keypoints_conf):
        indices = [15, 16] if scan_type == "intrusion" else range(len(person_kpts))
        for kpt_idx in indices:
            if conf[kpt_idx] <= 0.5:
                continue
            x, y = person_kpts[kpt_idx]
            pt = Point(x, y)
            inside = buffered.contains(pt) if machine_active else roi_poly.contains(pt)
            if inside:
                return {"type": scan_type, "severity": rule.get("severity") or "high"}
    return None


def _evaluate_pose_rule(
    frame: np.ndarray,
    rule: Dict[str, Any],
    scan_type: str,
    pose_cache: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    roi = rule.get("roi_normalized") or []
    if len(roi) < 3:
        return None
    h, w = frame.shape[:2]

    if pose_cache and pose_cache.get("keypoints_xy") is not None:
        return _evaluate_pose_from_keypoints(
            pose_cache["keypoints_xy"],
            pose_cache["keypoints_conf"],
            rule,
            scan_type,
            w,
            h,
        )

    model = _get_pose_model()
    results = model(frame, classes=[0], verbose=False)[0]
    if results.keypoints is None:
        return None

    keypoints_xy = results.keypoints.xy.cpu().numpy()
    keypoints_conf = results.keypoints.conf.cpu().numpy()

    for person_kpts, conf in zip(keypoints_xy, keypoints_conf):
        indices = [15, 16] if scan_type == "intrusion" else range(len(person_kpts))
        for kpt_idx in indices:
            if conf[kpt_idx] <= 0.5:
                continue
            x, y = person_kpts[kpt_idx]
            pt = Point(x, y)
            roi_poly = Polygon(_denorm_roi(roi, w, h))
            machine_active = scan_type == "danger_zone"
            buffered = roi_poly.buffer(20) if machine_active else roi_poly
            inside = buffered.contains(pt) if machine_active else roi_poly.contains(pt)
            if inside:
                return {"type": scan_type, "severity": rule.get("severity") or "high"}
    return None


def _with_pipe_lock(state: Dict[str, Any], fn):
    lock = state.get("pipe_lock")
    if lock is not None:
        with lock:
            return fn()
    return fn()


def evaluate_rule_on_frame(
    cam: Dict[str, Any],
    rule: Dict[str, Any],
    frame: np.ndarray,
    state: Dict[str, Any],
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any], np.ndarray]]:
    """Evaluate one rule on a frame. Returns (rule, event, frame) or None."""
    scan_type = rule.get("scan_type") or ""
    pipeline_name = common.SCAN_TO_PIPELINE.get(scan_type)
    if not pipeline_name:
        return None
    if common.needs_roi(scan_type) and len(rule.get("roi_normalized") or []) < 3:
        return None

    frame_idx = int(state.get("frame_idx") or 0)
    out = frame

    try:
        if scan_type in ("intrusion", "danger_zone"):
            pose_cache = state.get("pose_cache")
            event = _evaluate_pose_rule(frame, rule, scan_type, pose_cache=pose_cache)
            if event:
                return rule, event, out

        elif scan_type == "face_attendance":
            pipe = (state.get("pipelines") or {}).get("face_attendance")
            if pipe:
                def _run_face():
                    return pipe.process_frame(
                        out,
                        frame_idx,
                        np.array([]),
                        common.pipeline_config("face_attendance"),
                    )

                annotated, meta = _with_pipe_lock(state, _run_face)
                if annotated is not None:
                    out = annotated
                if isinstance(meta, dict) and (
                    meta.get("person_id") or meta.get("type") == "face_recognised"
                ):
                    return rule, meta, out

        elif scan_type == "vehicle":
            pipe = (state.get("pipelines") or {}).get("vehicle")
            if pipe:
                def _run_vehicle():
                    return pipe.process_frame(out, frame_idx, np.array([]), {})

                annotated, meta = _with_pipe_lock(state, _run_vehicle)
                if annotated is not None:
                    out = annotated
                event = _vehicle_event(meta or {})
                if event:
                    return rule, event, out

        elif scan_type == "gate_analytics":
            pipe = (state.get("pipelines") or {}).get("gate_analytics")
            if pipe:
                gate_config = rule.get("gate_config") or {}
                if not common.gate_config_valid(gate_config):
                    return None
                rule_id = rule.get("id") or ""
                baseline = store.get_gate_baseline(rule_id)

                def _run_gate():
                    config: Dict[str, Any] = {
                        "gate_config": gate_config,
                        "rule_id": rule_id,
                    }
                    if baseline is not None:
                        config["force_baseline"] = baseline
                    return pipe.process_frame(out, frame_idx, None, config)

                annotated, meta = _with_pipe_lock(state, _run_gate)
                if annotated is not None:
                    out = annotated
                if isinstance(meta, dict) and meta.get("type") == "gate_tick":
                    counters = meta.get("counters") or {}
                    if counters and not state.get("skip_gate_store"):
                        store.increment_gate_counters(rule_id, counters)
                    monitor_session = state.get("monitor_session")
                    session_lock = state.get("session_lock")
                    if monitor_session is not None and session_lock is not None:
                        with session_lock:
                            monitor_session["gate_live"] = meta.get("live") or {}
                    if meta.get("band") == "near" and meta.get("alert_near"):
                        if not store.on_cooldown(rule_id):
                            return (
                                rule,
                                {
                                    "type": "gate_near",
                                    "severity": rule.get("severity") or "medium",
                                },
                                out,
                            )

        elif scan_type == "fall_standing_lying":
            pipe = (state.get("pipelines") or {}).get("fall_standing_lying")
            if pipe:
                rule_id = rule.get("id") or ""
                roi = rule.get("roi_normalized") or []
                roi_np = _denorm_roi(roi, out.shape[1], out.shape[0]) if len(roi) >= 3 else np.array([])
                fall_sl = state.setdefault("fall_sl", {})
                rule_state = fall_sl.setdefault(rule_id, {})
                cfg = {
                    **common.pipeline_config("fall_standing_lying"),
                    "transition_state": rule_state,
                    "rule_id": rule_id,
                }

                def _run_fsl():
                    return pipe.process_frame(out, frame_idx, roi_np, cfg)

                annotated, meta = _with_pipe_lock(state, _run_fsl)
                if annotated is not None:
                    out = annotated
                if isinstance(meta, dict) and meta.get("type") == "fall_standing_lying":
                    return rule, meta, out

        elif scan_type == "fall":
            feed: FrameFeed = (state.get("frame_feeds") or {}).get("fall")
            gen = (state.get("generators") or {}).get("fall")
            if feed and gen:
                def _run_fall():
                    feed.frame = out
                    return next(gen)

                annotated, event = _with_pipe_lock(state, _run_fall)
                if annotated is not None:
                    out = annotated
                if common.is_event_dict(event):
                    return rule, event, out

    except StopIteration:
        pass
    except Exception as e:
        log.warning("scan rule %s/%s: %s", cam.get("id"), scan_type, e)

    return None


def evaluate_camera_frame(
    cam: Dict[str, Any],
    rules: List[Dict[str, Any]],
    frame: np.ndarray,
    state: Dict[str, Any],
) -> List[Tuple[Dict[str, Any], Dict[str, Any], np.ndarray]]:
    """Evaluate all rules on one frame; returns alert candidates."""
    hits: List[Tuple[Dict[str, Any], Dict[str, Any], np.ndarray]] = []
    for rule in rules:
        hit = evaluate_rule_on_frame(cam, rule, frame, state)
        if hit:
            hits.append(hit)
    state["frame_idx"] = int(state.get("frame_idx") or 0) + 1
    return hits


def priority_weight(scan_type: str) -> int:
    weights = {
        "intrusion": 30,
        "danger_zone": 28,
        "fall": 25,
        "fall_standing_lying": 24,
        "gate_analytics": 15,
        "vehicle": 12,
        "face_attendance": 10,
    }
    return weights.get(scan_type or "", 5)


def compute_tick_sleep_sec(
    rules: List[Dict[str, Any]],
    inference_wait_ms: float,
    tick_min: float,
    tick_max: float,
) -> float:
    if not rules:
        return tick_max
    score = max(priority_weight(r.get("scan_type") or "") for r in rules)
    if inference_wait_ms > 2000:
        score -= 10
    elif inference_wait_ms > 800:
        score -= 5
    if score >= 25:
        return tick_min
    if score >= 15:
        return (tick_min + tick_max) / 2
    return tick_max
