"""Candidate-only multi-verifier for Site Admin rules (verify-before-alert).

Phase 1: Ultralytics person re-check + supervision zone/track dwell.
Phase 2+: optional RTMDet/RTMPose, RF-DETR, SAHI via env flags.
Phase 4: thin rule-based aggregator (LangGraph-shaped state machine).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger("site_admin.verify")

VERIFY_PERSON = os.environ.get("SITE_ADMIN_VERIFY_PERSON", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
VERIFY_MIN_FRAMES = max(1, int(os.environ.get("SITE_ADMIN_VERIFY_MIN_FRAMES", "2")))
VERIFY_PERSON_CONF = float(os.environ.get("SITE_ADMIN_VERIFY_PERSON_CONF", "0.55"))
VERIFY_CROP_PAD = float(os.environ.get("SITE_ADMIN_VERIFY_CROP_PAD", "0.15"))
ENABLE_RTM = os.environ.get("SITE_ADMIN_ENABLE_RTM", "0").strip().lower() in ("1", "true", "yes")
ENABLE_RFDETR = os.environ.get("SITE_ADMIN_ENABLE_RFDETR", "0").strip().lower() in (
    "1",
    "true",
    "yes",
)
ENABLE_SAHI = os.environ.get("SITE_ADMIN_ENABLE_SAHI", "0").strip().lower() in ("1", "true", "yes")
COMPARE_VERIFIERS = os.environ.get("SITE_ADMIN_COMPARE_VERIFIERS", "0").strip().lower() in (
    "1",
    "true",
    "yes",
)

_person_model = None
_rtm_bundle = None
_rfdetr_model = None


@dataclass
class Vote:
    name: str
    status: str  # pass | fail | skip
    detail: str = ""
    score: Optional[float] = None


@dataclass
class VerifyResult:
    ok: bool
    votes: List[Vote] = field(default_factory=list)
    streak: int = 0
    track_key: str = ""


def _denorm_roi(roi: Sequence[Sequence[float]], w: int, h: int) -> np.ndarray:
    pts = []
    for p in roi:
        if len(p) < 2:
            continue
        pts.append([float(p[0]) * w, float(p[1]) * h])
    return np.array(pts, dtype=np.float32) if pts else np.zeros((0, 2), dtype=np.float32)


def roi_polygon_px(roi_normalized: Sequence[Sequence[float]], w: int, h: int) -> np.ndarray:
    return _denorm_roi(roi_normalized, w, h)


def point_in_roi(xy: Tuple[float, float], roi_px: np.ndarray) -> bool:
    if roi_px is None or len(roi_px) < 3:
        return False
    try:
        from shapely.geometry import Point, Polygon

        return bool(Polygon(roi_px).contains(Point(float(xy[0]), float(xy[1]))))
    except Exception:
        return False


def box_overlaps_roi(xyxy: Sequence[float], roi_px: np.ndarray, min_overlap: float = 0.08) -> bool:
    if roi_px is None or len(roi_px) < 3 or len(xyxy) < 4:
        return False
    try:
        from shapely.geometry import Polygon, box as shapely_box

        x1, y1, x2, y2 = [float(v) for v in xyxy[:4]]
        person = shapely_box(x1, y1, x2, y2)
        poly = Polygon(roi_px)
        if not person.intersects(poly):
            return False
        area = person.area
        if area <= 1e-6:
            return False
        return float(person.intersection(poly).area / area) >= min_overlap
    except Exception:
        return False


def box_in_loiter_roi(xyxy: Sequence[float], roi_px: np.ndarray) -> Tuple[bool, float]:
    """
    Loitering membership: any meaningful intersection OR center/foot inside ROI.
    Ratio-only checks fail for tall people barely overlapping a small polygon.
    """
    if roi_px is None or len(roi_px) < 3 or len(xyxy) < 4:
        return False, 0.0
    try:
        from shapely.geometry import Polygon, box as shapely_box

        x1, y1, x2, y2 = [float(v) for v in xyxy[:4]]
        person = shapely_box(x1, y1, x2, y2)
        poly = Polygon(roi_px)
        if not person.intersects(poly):
            return False, 0.0
        inter = float(person.intersection(poly).area)
        ratio = inter / max(person.area, 1e-6)
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        foot = ((x1 + x2) * 0.5, y2)
        inside = (
            inter >= 40.0
            or ratio >= 0.02
            or point_in_roi((cx, cy), roi_px)
            or point_in_roi(foot, roi_px)
        )
        return inside, ratio
    except Exception:
        ok = box_overlaps_roi(xyxy, roi_px, min_overlap=0.02)
        return ok, 0.02 if ok else 0.0


def crop_person(
    frame: np.ndarray,
    xyxy: Sequence[float],
    pad: float = VERIFY_CROP_PAD,
) -> Optional[np.ndarray]:
    if frame is None or frame.size == 0 or len(xyxy) < 4:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in xyxy[:4]]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    x1 = max(0, int(x1 - bw * pad))
    y1 = max(0, int(y1 - bh * pad))
    x2 = min(w, int(x2 + bw * pad))
    y2 = min(h, int(y2 + bh * pad))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def _get_person_model():
    global _person_model
    if _person_model is None:
        from ultralytics import YOLO

        _person_model = YOLO("yolov8n.pt")
    return _person_model


def verify_person_on_crop(
    crop: np.ndarray,
    *,
    conf: float = VERIFY_PERSON_CONF,
) -> Vote:
    if crop is None or crop.size == 0:
        return Vote("PersonVerify", "fail", "empty_crop")
    try:
        model = _get_person_model()
        results = model(crop, classes=[0], conf=conf, verbose=False)[0]
        n = 0 if results.boxes is None else len(results.boxes)
        best = 0.0
        if results.boxes is not None and len(results.boxes):
            best = float(results.boxes.conf.cpu().numpy().max())
        if n > 0 and best >= conf:
            return Vote("PersonVerify", "pass", f"n={n}", score=best)
        return Vote("PersonVerify", "fail", f"n={n}", score=best)
    except Exception as e:
        log.warning("person verify failed: %s", e)
        return Vote("PersonVerify", "skip", str(e)[:80])


def supervision_zone_vote(
    detections_xyxy: np.ndarray,
    detections_conf: Optional[np.ndarray],
    roi_px: np.ndarray,
    *,
    frame_shape: Tuple[int, int],
) -> Vote:
    """Use supervision PolygonZone when available; shapely fallback otherwise."""
    if detections_xyxy is None or len(detections_xyxy) == 0:
        return Vote("ZoneDwell", "fail", "no_detections")
    if roi_px is None or len(roi_px) < 3:
        return Vote("ZoneDwell", "skip", "no_roi")

    try:
        import supervision as sv

        h, w = frame_shape[:2]
        confs = (
            detections_conf
            if detections_conf is not None
            else np.ones(len(detections_xyxy), dtype=np.float32)
        )
        dets = sv.Detections(
            xyxy=np.asarray(detections_xyxy, dtype=np.float32),
            confidence=np.asarray(confs, dtype=np.float32),
            class_id=np.zeros(len(detections_xyxy), dtype=int),
        )
        zone = sv.PolygonZone(
            polygon=np.asarray(roi_px, dtype=np.int32),
            triggering_anchors=(sv.Position.BOTTOM_CENTER,),
        )
        mask = zone.trigger(detections=dets)
        count = int(np.sum(mask)) if mask is not None else 0
        if count > 0:
            return Vote("ZoneDwell", "pass", f"in_zone={count}", score=float(count))
        return Vote("ZoneDwell", "fail", "in_zone=0", score=0.0)
    except Exception as e:
        # Fallback: any box overlapping ROI
        for box in detections_xyxy:
            if box_overlaps_roi(box, roi_px):
                return Vote("ZoneDwell", "pass", f"fallback_overlap:{e.__class__.__name__}")
        return Vote("ZoneDwell", "fail", f"fallback_none:{e.__class__.__name__}")


def track_and_dwell(
    state: Dict[str, Any],
    camera_id: str,
    rule_id: str,
    detections_xyxy: np.ndarray,
    detections_conf: Optional[np.ndarray],
    roi_px: np.ndarray,
    *,
    now: Optional[float] = None,
) -> Tuple[Optional[str], float, List[Tuple[float, float]]]:
    """
    Update ByteTrack (via supervision) and return (track_id, dwell_sec, path_points)
    for the best track currently inside the ROI.
    """
    now = now if now is not None else time.time()
    trackers: Dict[str, Any] = state.setdefault("_sv_trackers", {})
    dwell_map: Dict[str, Dict[str, Any]] = state.setdefault("_dwell", {})
    key = f"{camera_id}:{rule_id}"

    try:
        import supervision as sv

        tracker = trackers.get(key)
        if tracker is None:
            tracker = sv.ByteTrack()
            trackers[key] = tracker
        confs = (
            detections_conf
            if detections_conf is not None
            else np.ones(len(detections_xyxy), dtype=np.float32)
        )
        dets = sv.Detections(
            xyxy=np.asarray(detections_xyxy, dtype=np.float32),
            confidence=np.asarray(confs, dtype=np.float32),
            class_id=np.zeros(len(detections_xyxy), dtype=int),
        )
        tracked = tracker.update_with_detections(dets)
    except Exception as e:
        log.debug("ByteTrack unavailable (%s); centroid fallback", e)
        tracked = None

    best_tid = None
    best_dwell = 0.0
    best_path: List[Tuple[float, float]] = []

    rule_dwell = dwell_map.setdefault(key, {})
    active_ids = set()

    def _update_one(tid: str, cx: float, cy: float, inside: bool) -> None:
        nonlocal best_tid, best_dwell, best_path
        active_ids.add(tid)
        rec = rule_dwell.get(tid) or {
            "first_in": None,
            "last_seen": now,
            "path": [],
            "inside": False,
        }
        if inside:
            if rec.get("first_in") is None:
                rec["first_in"] = now
            rec["inside"] = True
            path = list(rec.get("path") or [])
            path.append((cx, cy, now))
            if len(path) > 240:
                path = path[-240:]
            rec["path"] = path
            dwell = max(0.0, now - float(rec["first_in"]))
            if dwell >= best_dwell:
                best_dwell = dwell
                best_tid = tid
                best_path = [(p[0], p[1]) for p in path]
        else:
            rec["first_in"] = None
            rec["inside"] = False
            rec["path"] = []
        rec["last_seen"] = now
        rule_dwell[tid] = rec

    if tracked is not None and len(tracked) > 0:
        ids = tracked.tracker_id
        for i, box in enumerate(tracked.xyxy):
            tid = str(int(ids[i])) if ids is not None else f"i{i}"
            x1, y1, x2, y2 = [float(v) for v in box[:4]]
            cx, cy = (x1 + x2) * 0.5, y2  # foot-ish
            cx_c, cy_c = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            inside, _ov = box_in_loiter_roi(box, roi_px)
            if not inside:
                inside = point_in_roi((cx, cy), roi_px) or point_in_roi((cx_c, cy_c), roi_px)
            _update_one(tid, cx_c, cy_c, inside)
    else:
        for i, box in enumerate(detections_xyxy):
            tid = f"fb{i}"
            x1, y1, x2, y2 = [float(v) for v in box[:4]]
            cx, cy = (x1 + x2) * 0.5, y2
            cx_c, cy_c = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            inside, _ov = box_in_loiter_roi(box, roi_px)
            if not inside:
                inside = point_in_roi((cx, cy), roi_px) or point_in_roi((cx_c, cy_c), roi_px)
            _update_one(tid, cx_c, cy_c, inside)

    # Evict stale tracks
    for tid in list(rule_dwell.keys()):
        if tid not in active_ids and now - float(rule_dwell[tid].get("last_seen") or 0) > 8.0:
            rule_dwell.pop(tid, None)

    return best_tid, best_dwell, best_path


def bump_confirm_streak(
    state: Dict[str, Any],
    camera_id: str,
    rule_id: str,
    track_key: str,
    success: bool,
) -> int:
    streaks: Dict[str, Dict[str, Any]] = state.setdefault("_verify_streaks", {})
    key = f"{camera_id}:{rule_id}"
    rec = streaks.get(key) or {"track": "", "n": 0}
    if not success:
        rec = {"track": track_key or "", "n": 0}
    elif rec.get("track") == track_key:
        rec["n"] = int(rec.get("n") or 0) + 1
    else:
        rec = {"track": track_key or "", "n": 1}
    streaks[key] = rec
    return int(rec["n"])


def optional_rtm_vote(crop: np.ndarray) -> Vote:
    """Phase 2: RTMDet/RTMPose if SITE_ADMIN_ENABLE_RTM=1 and packages installed.

    Prefer ONNX Runtime export when available to avoid full OpenMMLab in Docker.
    Until configs/weights are provisioned, this node stays skip (compare-safe).
    """
    if not ENABLE_RTM and not COMPARE_VERIFIERS:
        return Vote("RTMPose", "skip", "disabled")
    if crop is None or crop.size == 0:
        return Vote("RTMPose", "skip", "no_crop")
    try:
        global _rtm_bundle
        if _rtm_bundle is None:
            # Lazy import — OpenMMLab / ONNX optional (not in default image)
            try:
                import onnxruntime as ort  # type: ignore

                model_path = os.environ.get("SITE_ADMIN_RTMPOSE_ONNX", "").strip()
                if model_path and os.path.isfile(model_path):
                    _rtm_bundle = ort.InferenceSession(model_path)
                else:
                    from mmpose.apis import inference_topdown, init_model  # type: ignore

                    _ = inference_topdown, init_model
                    _rtm_bundle = "unavailable"
            except Exception:
                _rtm_bundle = "unavailable"
        if _rtm_bundle == "unavailable":
            return Vote("RTMPose", "skip", "mmpose_not_configured")
        # ONNX session present — soft pass without full postprocess in Phase 2 scaffold
        return Vote("RTMPose", "pass", "onnx_session", score=1.0)
    except Exception as e:
        return Vote("RTMPose", "skip", f"unavailable:{e.__class__.__name__}")


def optional_rfdetr_vote(crop: np.ndarray) -> Vote:
    """Phase 3: RF-DETR person confirm when enabled."""
    if not ENABLE_RFDETR and not COMPARE_VERIFIERS:
        return Vote("RFDETR", "skip", "disabled")
    if crop is None or crop.size == 0:
        return Vote("RFDETR", "skip", "no_crop")
    try:
        global _rfdetr_model
        if _rfdetr_model is None:
            from rfdetr import RFDETRBase  # type: ignore

            _rfdetr_model = RFDETRBase()
        # Minimal API varies by package version — treat as soft compare
        preds = _rfdetr_model.predict(crop, threshold=VERIFY_PERSON_CONF)
        # Best-effort: if predict returns anything truthy, pass
        ok = bool(preds)
        return Vote("RFDETR", "pass" if ok else "fail", "predict")
    except Exception as e:
        return Vote("RFDETR", "skip", f"unavailable:{e.__class__.__name__}")


def optional_sahi_hint(
    frame: np.ndarray,
    *,
    camera: Optional[Dict[str, Any]] = None,
    rule: Optional[Dict[str, Any]] = None,
) -> Vote:
    """Phase 3: SAHI slicer only for far/small cameras when SITE_ADMIN_ENABLE_SAHI=1."""
    if not ENABLE_SAHI:
        return Vote("SAHI", "skip", "disabled")
    cam = camera or {}
    rl = rule or {}
    far = bool(
        cam.get("far_camera")
        or cam.get("small_objects")
        or rl.get("use_sahi")
        or (cam.get("flags") or {}).get("far")
    )
    if not far:
        return Vote("SAHI", "skip", "not_far_camera")
    try:
        from sahi import AutoDetectionModel  # type: ignore
        from sahi.predict import get_sliced_prediction  # type: ignore

        _ = AutoDetectionModel, get_sliced_prediction, frame
        # Lazy path: package present; full slice runs in a later hardening pass
        return Vote("SAHI", "skip", "configured_far_cam")
    except Exception as e:
        return Vote("SAHI", "skip", f"unavailable:{e.__class__.__name__}")


def aggregate_votes(votes: List[Vote], *, require_person: bool = True) -> bool:
    """
    Thin LangGraph-shaped aggregator (Phase 4 rule-based):
    - PersonVerify must pass when require_person
    - ZoneDwell must pass or skip
    - Optional RTM/RFDETR: if they fail (not skip), veto; skip is OK
    """
    by = {v.name: v for v in votes}
    person = by.get("PersonVerify")
    if require_person and VERIFY_PERSON:
        if person is None or person.status == "fail":
            return False
        if person.status == "skip":
            # Soft: allow if zone passed (model missing)
            zone = by.get("ZoneDwell")
            if zone is None or zone.status == "fail":
                return False
    zone = by.get("ZoneDwell")
    if zone is not None and zone.status == "fail":
        return False
    for name in ("RTMPose", "RFDETR"):
        v = by.get(name)
        if v is not None and v.status == "fail":
            return False
    return True


def votes_to_payload(votes: List[Vote], *, final_ok: bool, streak: int = 0) -> Dict[str, Any]:
    return {
        "final": "pass" if final_ok else "fail",
        "streak": streak,
        "min_frames": VERIFY_MIN_FRAMES,
        "chips": [
            {
                "name": v.name,
                "status": v.status,
                "detail": v.detail,
                "score": v.score,
            }
            for v in votes
        ],
    }


def verify_pose_candidate(
    frame: np.ndarray,
    rule: Dict[str, Any],
    event: Dict[str, Any],
    state: Dict[str, Any],
    *,
    camera_id: str,
    boxes_xyxy: Optional[np.ndarray] = None,
    boxes_conf: Optional[np.ndarray] = None,
    person_idx: Optional[int] = None,
) -> VerifyResult:
    """Run Phase-1(+optional) verifiers for an intrusion/danger pose hit."""
    h, w = frame.shape[:2]
    roi = rule.get("roi_normalized") or []
    roi_px = roi_polygon_px(roi, w, h)
    votes: List[Vote] = [Vote("PrimaryYOLO", "pass", event.get("keypoint") or "hit")]

    xyxy = None
    if (
        boxes_xyxy is not None
        and person_idx is not None
        and 0 <= person_idx < len(boxes_xyxy)
    ):
        xyxy = [float(v) for v in boxes_xyxy[person_idx][:4]]
        event["person_xyxy"] = xyxy

    crop = crop_person(frame, xyxy) if xyxy else None
    if VERIFY_PERSON:
        votes.append(verify_person_on_crop(crop) if crop is not None else Vote("PersonVerify", "fail", "no_box"))
    else:
        votes.append(Vote("PersonVerify", "skip", "disabled"))

    dets = boxes_xyxy if boxes_xyxy is not None and len(boxes_xyxy) else (
        np.array([xyxy], dtype=np.float32) if xyxy else np.zeros((0, 4), dtype=np.float32)
    )
    confs = None
    if boxes_conf is not None and len(boxes_conf) == len(dets):
        confs = boxes_conf
    votes.append(supervision_zone_vote(dets, confs, roi_px, frame_shape=(h, w)))

    # Optional compare / phase 2-3
    if crop is not None:
        votes.append(optional_rtm_vote(crop))
        votes.append(optional_rfdetr_vote(crop))
    else:
        votes.append(Vote("RTMPose", "skip", "no_crop"))
        votes.append(Vote("RFDETR", "skip", "no_crop"))
    votes.append(
        optional_sahi_hint(
            frame,
            camera={"id": camera_id, **(rule.get("_camera") or {})},
            rule=rule,
        )
    )

    tid, dwell, _path = track_and_dwell(
        state,
        camera_id,
        rule.get("id") or "",
        dets,
        confs,
        roi_px,
    )
    track_key = tid or (f"p{person_idx}" if person_idx is not None else "unknown")
    votes.append(Vote("TrackDwell", "pass" if dwell > 0 else "skip", f"dwell={dwell:.1f}s", score=dwell))

    soft_ok = aggregate_votes(votes, require_person=True)
    streak = bump_confirm_streak(
        state, camera_id, rule.get("id") or "", track_key, soft_ok
    )
    ok = soft_ok and streak >= VERIFY_MIN_FRAMES
    # Phase 4: LangGraph-shaped aggregate + optional FP-policy RAG hints
    graph_out = run_verify_graph(
        {
            "votes": list(votes),
            "streak": streak,
            "rag_hints": rag_hints_for_camera(str(rule.get("camera_name") or camera_id)),
        }
    )
    graph_chips = (graph_out.get("votes") or {}).get("chips") or []
    for chip in graph_chips:
        if chip.get("name") == "RAGPolicy":
            votes.append(
                Vote(
                    "RAGPolicy",
                    chip.get("status") or "skip",
                    chip.get("detail") or "",
                    score=chip.get("score"),
                )
            )
            break
    if graph_out.get("ok") is False:
        ok = False
    votes.append(Vote("Final", "pass" if ok else "fail", f"streak={streak}/{VERIFY_MIN_FRAMES}"))
    return VerifyResult(ok=ok, votes=votes, streak=streak, track_key=track_key)


# ── Phase 4: thin agentic state machine (LangGraph-shaped, no LLM required) ──


def run_verify_graph(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic graph: Primary -> PersonVerify -> ZoneDwell -> Optional -> Aggregate.
    Optional RAG policy can bias thresholds via payload['rag_hints'].
    """
    votes = list(payload.get("votes") or [])
    hints = payload.get("rag_hints") or {}
    min_frames = int(hints.get("min_frames") or VERIFY_MIN_FRAMES)
    # Apply RAG hint: raise person conf requirement symbolically via detail
    if hints.get("reject_patterns"):
        votes.append(Vote("RAGPolicy", "pass", f"patterns={len(hints['reject_patterns'])}"))
    else:
        votes.append(Vote("RAGPolicy", "skip", "no_hints"))
    ok = aggregate_votes([v if isinstance(v, Vote) else Vote(**v) for v in votes])
    streak = int(payload.get("streak") or 0)
    final = ok and streak >= min_frames
    return {"ok": final, "votes": votes_to_payload(
        [v if isinstance(v, Vote) else Vote(**v) for v in votes],
        final_ok=final,
        streak=streak,
    )}


_RAG_HINTS: Dict[str, Any] = {
    "reject_patterns": [
        "wet alley plants as nose/ankle",
        "puddle reflection person ghost",
    ],
}


def rag_hints_for_camera(camera_name: str = "") -> Dict[str, Any]:
    """Phase 4 lightweight policy memory (in-process; replace with vector store later)."""
    hints = dict(_RAG_HINTS)
    if camera_name:
        hints["camera"] = camera_name
    return hints
