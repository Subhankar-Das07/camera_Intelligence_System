"""Person detect + ByteTrack for Live Monitor select-and-follow."""

from __future__ import annotations

import logging
import os
import threading
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("site_admin.person_track")

_model = None
_model_lock = threading.Lock()
_WEIGHTS = os.environ.get("SITE_ADMIN_PERSON_WEIGHTS", "yolov8n.pt")
_TRACKER = os.environ.get("SITE_ADMIN_PERSON_TRACKER", "bytetrack.yaml")
_CONF = float(os.environ.get("SITE_ADMIN_PERSON_CONF", "0.35"))


def get_person_model():
    """Lazy YOLO detector used only for Live Monitor person tracking."""
    global _model
    with _model_lock:
        if _model is None:
            from ultralytics import YOLO

            log.info("Loading person track model: %s", _WEIGHTS)
            _model = YOLO(_WEIGHTS)
        return _model


def reset_tracker() -> None:
    """Drop tracker state (e.g. when monitor session stops)."""
    global _model
    with _model_lock:
        _model = None


def _iou_xyxy(a: List[float], b: List[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def _box_to_region(xyxy: List[float], w: int, h: int, conf: float, track_id: Optional[int]) -> Dict[str, Any]:
    x0, y0, x1, y1 = [float(v) for v in xyxy]
    x0 = max(0.0, min(float(w - 1), x0))
    y0 = max(0.0, min(float(h - 1), y0))
    x1 = max(0.0, min(float(w), x1))
    y1 = max(0.0, min(float(h), y1))
    bw = max(1.0, x1 - x0)
    bh = max(1.0, y1 - y0)
    area = (bw * bh) / float(max(1, w * h))
    polygon = [
        [round(x0 / w, 5), round(y0 / h, 5)],
        [round(x1 / w, 5), round(y0 / h, 5)],
        [round(x1 / w, 5), round(y1 / h, 5)],
        [round(x0 / w, 5), round(y1 / h, 5)],
    ]
    return {
        "id": f"person-{uuid.uuid4().hex[:10]}",
        "polygon": polygon,
        "bbox": [round(x0 / w, 5), round(y0 / h, 5), round(bw / w, 5), round(bh / h, 5)],
        "bbox_norm": [
            round(x0 / w, 5),
            round(y0 / h, 5),
            round(x1 / w, 5),
            round(y1 / h, 5),
        ],
        "xyxy": [int(x0), int(y0), int(x1), int(y1)],
        "area": round(area, 5),
        "conf": round(float(conf), 4),
        "track_id": int(track_id) if track_id is not None else None,
        "label": "person",
        "kind": "person",
    }


def track_persons(frame: np.ndarray, *, persist: bool = True) -> List[Dict[str, Any]]:
    """Run YOLO person ByteTrack; return regions with track_id when available."""
    if frame is None or getattr(frame, "size", 0) == 0:
        return []
    h, w = frame.shape[:2]
    model = get_person_model()
    with _model_lock:
        try:
            results = model.track(
                frame,
                persist=persist,
                classes=[0],
                tracker=_TRACKER,
                conf=_CONF,
                verbose=False,
            )
        except Exception as e:
            log.warning("person track failed, falling back to detect: %s", e)
            results = model.predict(frame, classes=[0], conf=_CONF, verbose=False)

    if not results:
        return []
    r0 = results[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return []
    xyxy = r0.boxes.xyxy.cpu().numpy()
    confs = r0.boxes.conf.cpu().numpy() if r0.boxes.conf is not None else np.ones(len(xyxy))
    ids = None
    if getattr(r0.boxes, "id", None) is not None:
        try:
            ids = r0.boxes.id.cpu().numpy().astype(int)
        except Exception:
            ids = None

    out: List[Dict[str, Any]] = []
    for i in range(len(xyxy)):
        tid = int(ids[i]) if ids is not None and i < len(ids) else None
        out.append(_box_to_region(xyxy[i].tolist(), w, h, float(confs[i]), tid))
    out.sort(key=lambda x: x.get("area") or 0, reverse=True)
    return out


def suggest_person_regions(
    frame: np.ndarray,
    *,
    max_regions: int = 36,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Person boxes for Monitor Suggest (Rules-parity region list)."""
    if frame is None or getattr(frame, "size", 0) == 0:
        return [], 0, 0
    h, w = frame.shape[:2]
    regions = track_persons(frame, persist=True)
    return regions[: max(1, max_regions)], w, h


def person_at_point(frame: np.ndarray, nx: float, ny: float) -> Optional[Dict[str, Any]]:
    """Pick smallest person box containing the normalized click."""
    regions = track_persons(frame, persist=True)
    if not regions:
        return None
    hits = []
    for r in regions:
        bn = r.get("bbox_norm") or []
        if len(bn) != 4:
            continue
        if bn[0] <= nx <= bn[2] and bn[1] <= ny <= bn[3]:
            hits.append(r)
    if hits:
        hits.sort(key=lambda x: x.get("area") or 1)
        return hits[0]
    # Nearest center fallback
    best = None
    best_d = 1e9
    for r in regions:
        bn = r.get("bbox_norm") or []
        if len(bn) != 4:
            continue
        cx = (bn[0] + bn[2]) * 0.5
        cy = (bn[1] + bn[3]) * 0.5
        d = (cx - nx) ** 2 + (cy - ny) ** 2
        if d < best_d:
            best_d = d
            best = r
    return best


def match_track_to_xyxy(frame: np.ndarray, xyxy: List[float], *, min_iou: float = 0.25) -> Optional[Dict[str, Any]]:
    """After user picks a region, bind the best overlapping ByteTrack person."""
    regions = track_persons(frame, persist=True)
    if not regions:
        return None
    best = None
    best_iou = 0.0
    for r in regions:
        iou = _iou_xyxy([float(x) for x in xyxy], [float(x) for x in (r.get("xyxy") or [])])
        if iou > best_iou:
            best_iou = iou
            best = r
    if best is None or best_iou < min_iou:
        return None
    return best


def find_track_by_id(frame: np.ndarray, track_id: int) -> Optional[Dict[str, Any]]:
    regions = track_persons(frame, persist=True)
    for r in regions:
        if r.get("track_id") is not None and int(r["track_id"]) == int(track_id):
            return r
    return None
