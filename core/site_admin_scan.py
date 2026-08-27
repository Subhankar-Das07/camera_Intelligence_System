"""Single-frame rule evaluation for parallel go-live camera workers."""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from shapely.geometry import Point, Polygon

from core import site_admin_common as common
from core import site_admin_store as store
from core.registry import registry
from core.video_source import get_video_source

log = logging.getLogger("site_admin.scan")

SCAN_RULE_WORKERS = max(1, min(3, int(os.environ.get("SITE_ADMIN_SCAN_RULE_WORKERS", "3"))))

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
        "pipe_lock": threading.Lock(),
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


POSE_DETECT_CONF = float(os.environ.get("SITE_ADMIN_POSE_DETECT_CONF", "0.55"))
PERSON_BOX_MIN_CONF = float(os.environ.get("SITE_ADMIN_PERSON_BOX_MIN_CONF", "0.65"))
PERSON_MIN_VISIBLE_KPTS = int(os.environ.get("SITE_ADMIN_PERSON_MIN_KPTS", "4"))
PERSON_MIN_BOX_OVERLAP = float(os.environ.get("SITE_ADMIN_PERSON_MIN_BOX_OVERLAP", "0.12"))
PERSON_MIN_BOX_AREA_FRAC = float(os.environ.get("SITE_ADMIN_PERSON_MIN_BOX_AREA_FRAC", "0.008"))
# Shoulders / hips / knees / ankles — reject head-only reflection ghosts
_BODY_SUPPORT_KPT_INDICES = (5, 6, 11, 12, 13, 14, 15, 16)


def build_pose_cache(frame: np.ndarray) -> Dict[str, Any]:
    """Single YOLO pose pass; share across intrusion/danger_zone rules."""
    model = _get_pose_model()
    results = model(frame, classes=[0], conf=POSE_DETECT_CONF, verbose=False)[0]
    cache: Dict[str, Any] = {
        "results": results,
        "keypoints_xy": None,
        "keypoints_conf": None,
        "boxes_xyxy": None,
        "boxes_conf": None,
    }
    if results.keypoints is not None:
        cache["keypoints_xy"] = results.keypoints.xy.cpu().numpy()
        cache["keypoints_conf"] = results.keypoints.conf.cpu().numpy()
    if results.boxes is not None and len(results.boxes) > 0:
        try:
            cache["boxes_xyxy"] = results.boxes.xyxy.cpu().numpy()
            cache["boxes_conf"] = results.boxes.conf.cpu().numpy()
        except Exception:
            pass
    # #region agent log
    try:
        import json as _json
        _n = 0 if cache["keypoints_xy"] is None else int(len(cache["keypoints_xy"]))
        _bc = []
        if cache.get("boxes_conf") is not None:
            _bc = [round(float(x), 3) for x in list(cache["boxes_conf"][:8])]
        _kc_max = []
        if cache.get("keypoints_conf") is not None:
            for row in list(cache["keypoints_conf"][:4]):
                _kc_max.append(round(float(max(row)), 3) if len(row) else 0.0)
        _payload = _json.dumps({
            "sessionId": "46c418",
            "hypothesisId": "A,C",
            "location": "site_admin_scan.py:build_pose_cache",
            "message": "pose pass summary",
            "data": {
                "n_pose_persons": _n,
                "n_boxes": len(_bc),
                "boxes_conf": _bc,
                "kpt_max_conf_per_person": _kc_max,
                "pose_detect_conf": POSE_DETECT_CONF,
                "h": int(frame.shape[0]),
                "w": int(frame.shape[1]),
            },
            "timestamp": int(time.time() * 1000),
            "runId": "post-fix-v3",
        }) + "\n"
        for _p in (
            "debug-46c418.log",
            os.path.join("static", "debug-46c418.log"),
            os.path.join(os.path.dirname(__file__), "debug-46c418.log"),
        ):
            try:
                with open(_p, "a", encoding="utf-8") as _f:
                    _f.write(_payload)
            except Exception:
                pass
    except Exception:
        pass
    # #endregion
    return cache


def _part_for_kpt(kpt_idx: int, parts: List[str]) -> str:
    for part in parts:
        if kpt_idx in (common.POSE_PART_INDICES.get(part) or []):
            return part
    return "whole"


def _person_box_passes_gate(
    p_i: int,
    boxes_xyxy: Optional[np.ndarray],
    boxes_conf: Optional[np.ndarray],
    roi_poly: Polygon,
    *,
    min_box_conf: float,
    frame_w: int,
    frame_h: int,
    min_overlap: float = PERSON_MIN_BOX_OVERLAP,
    min_area_frac: float = PERSON_MIN_BOX_AREA_FRAC,
) -> Tuple[bool, Optional[float], bool, float]:
    """
    Require a confident person detection with meaningful ROI overlap.
    Returns (ok, box_conf, bbox_intersects_roi, overlap_ratio).
    """
    if boxes_conf is None or boxes_xyxy is None or p_i >= len(boxes_conf) or p_i >= len(boxes_xyxy):
        return False, None, False, 0.0
    box_conf = float(boxes_conf[p_i])
    if box_conf < min_box_conf:
        return False, box_conf, False, 0.0
    x1, y1, x2, y2 = [float(v) for v in boxes_xyxy[p_i][:4]]
    box_w = max(0.0, x2 - x1)
    box_h = max(0.0, y2 - y1)
    box_area = box_w * box_h
    frame_area = float(max(1, frame_w) * max(1, frame_h))
    if box_area < min_area_frac * frame_area:
        return False, box_conf, False, 0.0
    overlap_ratio = 0.0
    intersects = False
    try:
        from shapely.geometry import box as shapely_box

        person_box = shapely_box(x1, y1, x2, y2)
        if person_box.intersects(roi_poly):
            intersects = True
            inter = person_box.intersection(roi_poly).area
            overlap_ratio = float(inter / box_area) if box_area > 1e-6 else 0.0
    except Exception:
        intersects = False
        overlap_ratio = 0.0
    if not intersects:
        return False, box_conf, False, overlap_ratio
    # Center-in-ROI also counts as a strong presence signal
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    center_in = False
    try:
        center_in = bool(roi_poly.contains(Point(cx, cy)))
    except Exception:
        center_in = False
    if not center_in and overlap_ratio < min_overlap:
        return False, box_conf, True, overlap_ratio
    return True, box_conf, True, overlap_ratio


def _person_skeleton_passes(
    conf: np.ndarray,
    *,
    min_conf: float,
    min_visible: int = PERSON_MIN_VISIBLE_KPTS,
) -> Tuple[bool, int, int]:
    """
    Reject sparse / head-only pose ghosts (wet pavement reflections, etc.).
    Returns (ok, visible_count, body_support_count).
    """
    visible = 0
    body = 0
    for i, c in enumerate(conf):
        if float(c) < min_conf:
            continue
        visible += 1
        if i in _BODY_SUPPORT_KPT_INDICES:
            body += 1
    ok = visible >= min_visible and body >= 1
    return ok, visible, body


def _evaluate_pose_from_keypoints(
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray,
    rule: Dict[str, Any],
    scan_type: str,
    width: int,
    height: int,
    boxes_xyxy: Optional[np.ndarray] = None,
    boxes_conf: Optional[np.ndarray] = None,
) -> Optional[Dict[str, Any]]:
    roi = rule.get("roi_normalized") or []
    if len(roi) < 3:
        return None
    roi_poly = Polygon(_denorm_roi(roi, width, height))
    # Danger zones previously used a 20px buffer; that caused plant/edge FPs
    # when keypoints landed just outside the drawn ROI (buffer_only_hit).
    machine_active = scan_type == "danger_zone"
    danger_buffer_px = int(os.environ.get("SITE_ADMIN_DANGER_BUFFER_PX", "0"))
    buffered = roi_poly.buffer(danger_buffer_px) if machine_active and danger_buffer_px > 0 else roi_poly

    trigger = common.normalize_pose_trigger(scan_type, rule.get("pose_trigger"))
    indices = common.pose_trigger_indices(trigger)
    min_conf = float(trigger.get("min_conf") or 0.5)
    parts = list(trigger.get("parts") or [])
    require_person = bool(trigger.get("require_person", True))
    rule_name = rule.get("name") or scan_type.replace("_", " ")
    min_box_conf = PERSON_BOX_MIN_CONF

    # #region agent log
    try:
        import json as _json
        _payload = _json.dumps({
            "sessionId": "46c418",
            "hypothesisId": "B,D",
            "location": "site_admin_scan.py:_evaluate_pose_from_keypoints:entry",
            "message": "pose eval entry",
            "data": {
                "scan_type": scan_type,
                "rule_name": rule_name,
                "require_person": require_person,
                "require_person_enforced": True,
                "min_conf": min_conf,
                "min_box_conf": min_box_conf,
                "min_visible_kpts": PERSON_MIN_VISIBLE_KPTS,
                "min_box_overlap": PERSON_MIN_BOX_OVERLAP,
                "n_persons": int(len(keypoints_xy)),
                "n_boxes": 0 if boxes_conf is None else int(len(boxes_conf)),
                "danger_buffer_px": danger_buffer_px if machine_active else 0,
                "parts": parts,
            },
            "timestamp": int(time.time() * 1000),
            "runId": "post-fix-v3",
        }) + "\n"
        for _p in (
            "debug-46c418.log",
            os.path.join("static", "debug-46c418.log"),
            os.path.join(os.path.dirname(__file__), "debug-46c418.log"),
        ):
            try:
                with open(_p, "a", encoding="utf-8") as _f:
                    _f.write(_payload)
            except Exception:
                pass
    except Exception:
        pass
    # #endregion

    for p_i, (person_kpts, conf) in enumerate(zip(keypoints_xy, keypoints_conf)):
        box_ok, box_conf, bbox_intersects, overlap_ratio = True, None, True, 1.0
        if require_person:
            box_ok, box_conf, bbox_intersects, overlap_ratio = _person_box_passes_gate(
                p_i,
                boxes_xyxy,
                boxes_conf,
                roi_poly,
                min_box_conf=min_box_conf,
                frame_w=width,
                frame_h=height,
            )
            if not box_ok:
                # #region agent log
                try:
                    import json as _json
                    _payload = _json.dumps({
                        "sessionId": "46c418",
                        "hypothesisId": "C",
                        "location": "site_admin_scan.py:_evaluate_pose_from_keypoints:person_gate_reject",
                        "message": "skipped weak/non-overlapping person",
                        "data": {
                            "rule_name": rule_name,
                            "person_idx": int(p_i),
                            "box_conf": None if box_conf is None else round(float(box_conf), 4),
                            "bbox_intersects_roi": bool(bbox_intersects),
                            "overlap_ratio": round(float(overlap_ratio), 4),
                            "min_box_conf": min_box_conf,
                            "min_box_overlap": PERSON_MIN_BOX_OVERLAP,
                        },
                        "timestamp": int(time.time() * 1000),
                        "runId": "post-fix-v3",
                    }) + "\n"
                    for _p in (
                        "debug-46c418.log",
                        os.path.join("static", "debug-46c418.log"),
                        os.path.join(os.path.dirname(__file__), "debug-46c418.log"),
                    ):
                        try:
                            with open(_p, "a", encoding="utf-8") as _f:
                                _f.write(_payload)
                        except Exception:
                            pass
                except Exception:
                    pass
                # #endregion
                continue

            sk_ok, vis_n, body_n = _person_skeleton_passes(conf, min_conf=min_conf)
            if not sk_ok:
                # #region agent log
                try:
                    import json as _json
                    _payload = _json.dumps({
                        "sessionId": "46c418",
                        "hypothesisId": "E",
                        "location": "site_admin_scan.py:_evaluate_pose_from_keypoints:skeleton_reject",
                        "message": "skipped incoherent/sparse pose",
                        "data": {
                            "rule_name": rule_name,
                            "person_idx": int(p_i),
                            "box_conf": None if box_conf is None else round(float(box_conf), 4),
                            "visible_kpts": int(vis_n),
                            "body_support_kpts": int(body_n),
                            "min_visible_kpts": PERSON_MIN_VISIBLE_KPTS,
                            "overlap_ratio": round(float(overlap_ratio), 4),
                        },
                        "timestamp": int(time.time() * 1000),
                        "runId": "post-fix-v3",
                    }) + "\n"
                    for _p in (
                        "debug-46c418.log",
                        os.path.join("static", "debug-46c418.log"),
                        os.path.join(os.path.dirname(__file__), "debug-46c418.log"),
                    ):
                        try:
                            with open(_p, "a", encoding="utf-8") as _f:
                                _f.write(_payload)
                        except Exception:
                            pass
                except Exception:
                    pass
                # #endregion
                continue

        for kpt_idx in indices:
            if kpt_idx >= len(person_kpts) or kpt_idx >= len(conf):
                continue
            kpt_c = float(conf[kpt_idx])
            if kpt_c <= min_conf:
                continue
            x, y = person_kpts[kpt_idx]
            pt = Point(float(x), float(y))
            inside_roi = roi_poly.contains(pt)
            inside_buffered = buffered.contains(pt) if buffered is not roi_poly else inside_roi
            # Always require the keypoint inside the drawn ROI (no soft buffer hits)
            if inside_buffered and not inside_roi:
                # #region agent log
                try:
                    import json as _json
                    _payload = _json.dumps({
                        "sessionId": "46c418",
                        "hypothesisId": "B",
                        "location": "site_admin_scan.py:_evaluate_pose_from_keypoints:buffer_reject",
                        "message": "skipped buffer-only keypoint",
                        "data": {
                            "rule_name": rule_name,
                            "person_idx": int(p_i),
                            "keypoint": common.POSE_KPT_NAMES.get(int(kpt_idx), f"kpt_{kpt_idx}"),
                            "kpt_conf": round(kpt_c, 4),
                            "box_conf": None if box_conf is None else round(float(box_conf), 4),
                            "xy": [round(float(x), 1), round(float(y), 1)],
                            "danger_buffer_px": danger_buffer_px if machine_active else 0,
                        },
                        "timestamp": int(time.time() * 1000),
                        "runId": "post-fix-v3",
                    }) + "\n"
                    for _p in (
                        "debug-46c418.log",
                        os.path.join("static", "debug-46c418.log"),
                        os.path.join(os.path.dirname(__file__), "debug-46c418.log"),
                    ):
                        try:
                            with open(_p, "a", encoding="utf-8") as _f:
                                _f.write(_payload)
                        except Exception:
                            pass
                except Exception:
                    pass
                # #endregion
                continue
            if inside_roi:
                kpt_name = common.POSE_KPT_NAMES.get(int(kpt_idx), f"kpt_{kpt_idx}")
                body_part = _part_for_kpt(int(kpt_idx), parts)
                part_label = common.POSE_PART_LABELS.get(body_part, body_part)
                # #region agent log
                try:
                    import json as _json
                    _payload = _json.dumps({
                        "sessionId": "46c418",
                        "hypothesisId": "A,C,D,E",
                        "location": "site_admin_scan.py:_evaluate_pose_from_keypoints:hit",
                        "message": "pose ROI hit",
                        "data": {
                            "scan_type": scan_type,
                            "rule_name": rule_name,
                            "person_idx": int(p_i),
                            "keypoint": kpt_name,
                            "kpt_conf": round(kpt_c, 4),
                            "box_conf": None if box_conf is None else round(float(box_conf), 4),
                            "has_box": box_conf is not None,
                            "bbox_intersects_roi": bool(bbox_intersects),
                            "overlap_ratio": round(float(overlap_ratio), 4),
                            "inside_raw_roi": bool(inside_roi),
                            "inside_buffered": bool(inside_buffered),
                            "buffer_only_hit": False,
                            "xy": [round(float(x), 1), round(float(y), 1)],
                            "require_person": require_person,
                            "require_person_enforced": True,
                        },
                        "timestamp": int(time.time() * 1000),
                        "runId": "post-fix-v3",
                    }) + "\n"
                    for _p in (
                        "debug-46c418.log",
                        os.path.join("static", "debug-46c418.log"),
                        os.path.join(os.path.dirname(__file__), "debug-46c418.log"),
                    ):
                        try:
                            with open(_p, "a", encoding="utf-8") as _f:
                                _f.write(_payload)
                        except Exception:
                            pass
                except Exception:
                    pass
                # #endregion
                return {
                    "type": scan_type,
                    "severity": rule.get("severity") or "high",
                    "body_part": body_part,
                    "keypoint": kpt_name,
                    "keypoint_idx": int(kpt_idx),
                    "person_idx": int(p_i),
                    "person_xyxy": (
                        [float(v) for v in boxes_xyxy[p_i][:4]]
                        if boxes_xyxy is not None and p_i < len(boxes_xyxy)
                        else None
                    ),
                    "message": f"{rule_name} — {kpt_name.replace('_', ' ')} ({part_label}) in area",
                }
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
            boxes_xyxy=pose_cache.get("boxes_xyxy"),
            boxes_conf=pose_cache.get("boxes_conf"),
        )

    model = _get_pose_model()
    results = model(frame, classes=[0], conf=POSE_DETECT_CONF, verbose=False)[0]
    if results.keypoints is None:
        return None

    keypoints_xy = results.keypoints.xy.cpu().numpy()
    keypoints_conf = results.keypoints.conf.cpu().numpy()
    boxes_xyxy = None
    boxes_conf = None
    if results.boxes is not None and len(results.boxes) > 0:
        try:
            boxes_xyxy = results.boxes.xyxy.cpu().numpy()
            boxes_conf = results.boxes.conf.cpu().numpy()
        except Exception:
            pass
    return _evaluate_pose_from_keypoints(
        keypoints_xy,
        keypoints_conf,
        rule,
        scan_type,
        w,
        h,
        boxes_xyxy=boxes_xyxy,
        boxes_conf=boxes_conf,
    )


def _apply_person_journeys(
    cam: Dict[str, Any],
    rule: Dict[str, Any],
    frame: np.ndarray,
    meta: Dict[str, Any],
    state: Dict[str, Any],
    *,
    pipe: Any = None,
    rule_id: str = "",
) -> np.ndarray:
    """Match person crops to cross-camera journeys and annotate global IDs."""
    from core import journey_store
    from core.person_reid import crop_from_xyxy, embed_person_crop, reid_enabled

    if not reid_enabled():
        return frame
    tracks = meta.get("person_tracks") or []
    if not tracks or frame is None or frame.size == 0:
        return frame

    camera_id = cam.get("id") or ""
    camera_name = cam.get("name") or camera_id
    frame_idx = int(state.get("frame_idx") or 0)
    # Re-ID every 3rd tick per worker to limit load
    if frame_idx % 3 != 0:
        # Still refresh journey_live from cached labels
        labels = {}
        if pipe is not None and hasattr(pipe, "_state"):
            labels = dict((pipe._state(rule_id) or {}).get("journey_labels") or {})
        live = []
        for tid, lab in labels.items():
            live.append({"gid": lab.split()[0] if lab else "", "local_track_id": tid, "label": lab})
        meta["journey_live"] = live
        return frame

    journey_live: List[Dict[str, Any]] = []
    labels: Dict[int, str] = {}
    if pipe is not None and hasattr(pipe, "_state") and rule_id:
        labels = dict((pipe._state(rule_id) or {}).get("journey_labels") or {})

    for tr in tracks[:6]:
        tid = int(tr.get("track_id") or 0)
        xyxy = tr.get("xyxy") or []
        if len(xyxy) != 4:
            continue
        crop = crop_from_xyxy(frame, (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])))
        emb = embed_person_crop(crop) if crop is not None else None
        if emb is None:
            continue
        try:
            jmeta = journey_store.match_or_create_reid(
                emb,
                camera_id=camera_id,
                camera_name=camera_name,
                rule_id=rule.get("id") or rule_id,
                scan_type=rule.get("scan_type") or "gate_analytics",
                local_track_id=tid,
                bbox=[float(x) for x in xyxy],
            )
        except Exception as e:
            log.warning("journey reid: %s", e)
            continue
        gid = jmeta.get("gid") or ""
        label = jmeta.get("label") or "unknown"
        display = f"{gid}" if label in ("", "unknown") else f"{gid}·{label}"
        labels[tid] = display
        journey_store.attach_local_track(camera_id, tid, gid)
        journey_live.append(
            {
                "gid": gid,
                "label": label,
                "local_track_id": tid,
                "match_method": jmeta.get("match_method"),
                "hop_summary": "",
                "bbox": [float(x) for x in xyxy],
                "bbox_norm": [
                    float(xyxy[0]) / max(1, frame.shape[1]),
                    float(xyxy[1]) / max(1, frame.shape[0]),
                    float(xyxy[2]) / max(1, frame.shape[1]),
                    float(xyxy[3]) / max(1, frame.shape[0]),
                ],
                "watched": journey_store.is_watched(gid),
            }
        )
        # Annotate once more with gid
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        cv2.putText(
            frame,
            display,
            (x1, max(14, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
        )

    if pipe is not None and hasattr(pipe, "_state") and rule_id:
        st = pipe._state(rule_id)
        st["journey_labels"] = labels

    # Enrich hop summaries
    for item in journey_live:
        gid = item.get("gid") or ""
        if gid and not item.get("hop_summary"):
            detail = journey_store.get_journey(gid, timeline_limit=10)
            if detail:
                item["hop_summary"] = detail.get("hop_summary") or ""
                item["label"] = detail.get("label") or item.get("label")

    meta["journey_live"] = journey_live
    return frame


def _with_pipe_lock(state: Dict[str, Any], fn):
    lock = state.get("pipe_lock")
    if lock is not None:
        with lock:
            return fn()
    return fn()


def _store_verifier_votes(state: Dict[str, Any], rule: Dict[str, Any], payload: Dict[str, Any]) -> None:
    monitor_session = state.get("monitor_session")
    session_lock = state.get("session_lock")
    if monitor_session is None:
        # Preview / go-live may stash on state directly
        votes = state.setdefault("verifier_votes_by_rule", {})
        votes[rule.get("id") or ""] = payload
        state["verifier_votes"] = payload
        return
    if session_lock is not None:
        with session_lock:
            votes = monitor_session.setdefault("verifier_votes_by_rule", {})
            votes[rule.get("id") or ""] = payload
            monitor_session["verifier_votes"] = payload
    else:
        votes = monitor_session.setdefault("verifier_votes_by_rule", {})
        votes[rule.get("id") or ""] = payload
        monitor_session["verifier_votes"] = payload


def _detect_persons_xyxy(frame: np.ndarray, pose_cache: Optional[Dict[str, Any]] = None):
    if pose_cache and pose_cache.get("boxes_xyxy") is not None and len(pose_cache["boxes_xyxy"]):
        return pose_cache["boxes_xyxy"], pose_cache.get("boxes_conf")
    model = _get_pose_model()
    results = model(frame, classes=[0], conf=POSE_DETECT_CONF, verbose=False)[0]
    if results.boxes is None or len(results.boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), None
    return results.boxes.xyxy.cpu().numpy(), results.boxes.conf.cpu().numpy()


def _detect_persons_for_loitering(
    frame: np.ndarray,
    pose_cache: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Person boxes for loitering — prefer detect model (seated/far) at low conf."""
    conf = float(os.environ.get("SITE_ADMIN_LOITER_DETECT_CONF", "0.22"))
    try:
        from core import site_admin_verify as verify

        model = verify._get_person_model()
        results = model(frame, classes=[0], conf=conf, verbose=False)[0]
        if results.boxes is not None and len(results.boxes):
            return results.boxes.xyxy.cpu().numpy(), results.boxes.conf.cpu().numpy()
    except Exception as e:
        log.debug("loiter detect model failed: %s", e)
    # Fallbacks
    model = _get_pose_model()
    results = model(frame, classes=[0], conf=conf, verbose=False)[0]
    if results.boxes is not None and len(results.boxes):
        return results.boxes.xyxy.cpu().numpy(), results.boxes.conf.cpu().numpy()
    return _detect_persons_xyxy(frame, pose_cache)


def _evaluate_loitering(
    frame: np.ndarray,
    rule: Dict[str, Any],
    cam: Dict[str, Any],
    state: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """RS-WACV24-inspired loitering: dwell + tortuosity/pace in ROI."""
    from core import site_admin_verify as verify
    from core.loitering_trajectory import loitering_decision, normalize_loiter_config

    roi = rule.get("roi_normalized") or []
    if len(roi) < 3:
        return None
    h, w = frame.shape[:2]
    roi_px = verify.roi_polygon_px(roi, w, h)
    boxes, confs = _detect_persons_for_loitering(frame, state.get("pose_cache"))
    if boxes is None or len(boxes) == 0:
        votes = {
            "final": "fail",
            "chips": [
                {"name": "PrimaryYOLO", "status": "fail", "detail": "no_person"},
                {"name": "LoiterMetrics", "status": "skip", "detail": "no_track"},
                {"name": "Final", "status": "fail", "detail": "no_person"},
            ],
        }
        _store_verifier_votes(state, rule, votes)
        return None

    tid, dwell, path = verify.track_and_dwell(
        state,
        cam.get("id") or "",
        rule.get("id") or "",
        boxes,
        confs,
        roi_px,
    )
    # Recover timestamps from dwell store
    key = f"{cam.get('id') or ''}:{rule.get('id') or ''}"
    rec = (state.get("_dwell") or {}).get(key, {}).get(tid or "", {})
    timed_path = rec.get("path") or []
    points = [(float(p[0]), float(p[1])) for p in timed_path] if timed_path else list(path)
    times = [float(p[2]) for p in timed_path if len(p) > 2] if timed_path else []
    if len(times) < len(points):
        import time as _time

        now = _time.time()
        times = [now - (len(points) - i) * 0.5 for i in range(len(points))]

    cfg = normalize_loiter_config(rule)
    decision = loitering_decision(dwell, points, times, cfg)

    # Person verify on best box overlapping ROI
    best_box = None
    best_conf = 0.0
    best_overlap = 0.0
    box_summaries = []
    for i, box in enumerate(boxes):
        inside, ov = verify.box_in_loiter_roi(box, roi_px)
        c = float(confs[i]) if confs is not None and i < len(confs) else 0.5
        box_summaries.append({
            "i": i,
            "conf": round(c, 3),
            "overlap": round(ov, 3),
            "inside": bool(inside),
            "xyxy": [round(float(v), 1) for v in box[:4]],
        })
        if inside and c >= best_conf:
            best_conf = c
            best_box = box
            best_overlap = ov
    if best_box is None and box_summaries:
        top = max(box_summaries, key=lambda s: s["overlap"])
        if top["overlap"] > 0 or top.get("inside"):
            best_overlap = top["overlap"]
            best_conf = top["conf"]
            best_box = boxes[top["i"]]
    crop = verify.crop_person(frame, best_box) if best_box is not None else None
    # Primary detection already class=0; accept when box is in/near ROI
    if best_box is not None and best_conf >= 0.22 and (
        best_overlap >= 0.02 or any(s.get("inside") for s in box_summaries)
    ):
        person_vote = verify.Vote(
            "PersonVerify", "pass", f"primary_conf={best_conf:.2f}", score=best_conf
        )
    elif crop is not None:
        person_vote = verify.verify_person_on_crop(crop)
    else:
        person_vote = verify.Vote("PersonVerify", "fail", "no_box")
    # Zone: prefer track dwell / loiter membership over strict PolygonZone alone
    zone_vote = verify.supervision_zone_vote(boxes, confs, roi_px, frame_shape=(h, w))
    in_roi = dwell > 0 or any(s.get("inside") for s in box_summaries) or best_overlap >= 0.02
    if zone_vote.status == "fail" and in_roi:
        zone_vote = verify.Vote(
            "ZoneDwell",
            "pass",
            f"overlap={best_overlap:.2f};dwell={dwell:.1f}",
            score=best_overlap or dwell,
        )
    loiter_vote = verify.Vote(
        "LoiterMetrics",
        "pass" if decision["triggered"] else "fail",
        f"dwell={decision['dwell_sec']}s tort={decision['tortuosity']} spd={decision['speed_px_s']}",
        score=decision["dwell_sec"],
    )
    votes_list = [
        verify.Vote("PrimaryYOLO", "pass", f"n={len(boxes)}"),
        person_vote,
        zone_vote,
        loiter_vote,
    ]
    soft = (
        person_vote.status != "fail"
        and zone_vote.status != "fail"
        and decision["triggered"]
    )
    streak = verify.bump_confirm_streak(
        state, cam.get("id") or "", rule.get("id") or "", tid or "loiter", soft
    )
    ok = soft and streak >= verify.VERIFY_MIN_FRAMES
    votes_list.append(
        verify.Vote("Final", "pass" if ok else "fail", f"streak={streak}/{verify.VERIFY_MIN_FRAMES}")
    )
    payload = verify.votes_to_payload(votes_list, final_ok=ok, streak=streak)
    # Optional agentic graph + RAG hints
    graph_out = verify.run_verify_graph(
        {
            "votes": votes_list,
            "streak": streak,
            "rag_hints": verify.rag_hints_for_camera(cam.get("name") or ""),
        }
    )
    payload["graph"] = graph_out.get("votes")
    _store_verifier_votes(state, rule, payload)
    if not ok:
        return None
    rule_name = rule.get("name") or "Loitering"
    return {
        "type": "loitering",
        "severity": rule.get("severity") or "medium",
        "message": (
            f"{rule_name} — loitering "
            f"{decision['dwell_sec']:.0f}s (tort={decision['tortuosity']:.1f})"
        ),
        "loitering": decision,
        "verifier_votes": payload,
        "track_id": tid,
    }


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
                from core import site_admin_verify as verify

                person_idx = event.get("person_idx")
                boxes = (pose_cache or {}).get("boxes_xyxy")
                confs = (pose_cache or {}).get("boxes_conf")
                vr = verify.verify_pose_candidate(
                    frame,
                    rule,
                    event,
                    state,
                    camera_id=cam.get("id") or "",
                    boxes_xyxy=boxes,
                    boxes_conf=confs,
                    person_idx=person_idx if isinstance(person_idx, int) else None,
                )
                event["verifier_votes"] = verify.votes_to_payload(
                    vr.votes, final_ok=vr.ok, streak=vr.streak
                )
                _store_verifier_votes(state, rule, event["verifier_votes"])
                if not vr.ok:
                    return None
                return rule, event, out

        elif scan_type == "loitering":
            event = _evaluate_loitering(frame, rule, cam, state)
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
                    if not state.get("skip_gate_store"):
                        out = _apply_person_journeys(
                            cam, rule, out, meta, state, pipe=pipe, rule_id=rule_id
                        )
                    monitor_session = state.get("monitor_session")
                    session_lock = state.get("session_lock")
                    if monitor_session is not None and session_lock is not None:
                        with session_lock:
                            monitor_session["gate_live"] = meta.get("live") or {}
                            if meta.get("journey_live"):
                                monitor_session["journey_live"] = meta.get("journey_live")
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
    """Evaluate all rules on one frame in parallel; returns alert candidates."""
    hits: List[Tuple[Dict[str, Any], Dict[str, Any], np.ndarray]] = []
    needs_pose = any(
        (r.get("scan_type") or "") in ("intrusion", "danger_zone", "loitering") for r in rules
    )
    state["pose_cache"] = build_pose_cache(frame) if needs_pose else None
    workers = min(SCAN_RULE_WORKERS, max(1, len(rules)))

    def _eval_one(rule: Dict[str, Any]):
        fr = frame.copy()
        return evaluate_rule_on_frame(cam, rule, fr, state)

    if workers <= 1 or len(rules) <= 1:
        for rule in rules:
            hit = _eval_one(rule)
            if hit:
                hits.append(hit)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_eval_one, rule): rule for rule in rules}
            for fut in as_completed(futures):
                try:
                    hit = fut.result()
                    if hit:
                        hits.append(hit)
                except Exception as e:
                    log.warning("parallel scan rule: %s", e)

    state["frame_idx"] = int(state.get("frame_idx") or 0) + 1
    return hits


def priority_weight(scan_type: str) -> int:
    weights = {
        "intrusion": 30,
        "danger_zone": 28,
        "loitering": 22,
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
