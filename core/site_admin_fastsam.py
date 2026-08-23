"""FastSAM helpers for Site Admin ROI suggestion (setup assist only)."""

from __future__ import annotations

import logging
import threading
import uuid
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("site_admin.fastsam")

_model = None
_model_lock = threading.Lock()
_model_name = "FastSAM-s.pt"


def get_fastsam(weights: str = "FastSAM-s.pt"):
    """Lazy-load and cache FastSAM (do not reload per request)."""
    global _model, _model_name
    with _model_lock:
        if _model is None or _model_name != weights:
            from ultralytics import FastSAM

            log.info("Loading FastSAM weights: %s", weights)
            _model = FastSAM(weights)
            _model_name = weights
        return _model


def _simplify_poly(pts: np.ndarray, max_pts: int = 24) -> np.ndarray:
    """Approx contour to a manageable polygon."""
    if pts is None or len(pts) < 3:
        return pts
    contour = pts.astype(np.float32).reshape(-1, 1, 2)
    peri = cv2.arcLength(contour, True)
    eps = max(1.0, 0.008 * peri)
    approx = cv2.approxPolyDP(contour, eps, True).reshape(-1, 2)
    while len(approx) > max_pts and eps < peri:
        eps *= 1.4
        approx = cv2.approxPolyDP(contour, eps, True).reshape(-1, 2)
    if len(approx) < 3:
        return pts.reshape(-1, 2)
    return approx


def _mask_to_polygon(mask_hw: np.ndarray) -> Optional[np.ndarray]:
    """Largest external contour of a binary mask → Nx2 float polygon."""
    m = (mask_hw > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 16:
        return None
    return _simplify_poly(c.reshape(-1, 2).astype(np.float32))


def _poly_from_xy(xy: np.ndarray) -> Optional[np.ndarray]:
    if xy is None or len(xy) < 3:
        return None
    return _simplify_poly(np.asarray(xy, dtype=np.float32))


def _normalize_poly(pts: np.ndarray, w: int, h: int) -> List[List[float]]:
    out = []
    for x, y in pts:
        out.append([round(float(x) / w, 5), round(float(y) / h, 5)])
    return out


def _bbox_norm(pts: np.ndarray, w: int, h: int) -> List[float]:
    xs = pts[:, 0]
    ys = pts[:, 1]
    x0, y0 = float(xs.min()), float(ys.min())
    x1, y1 = float(xs.max()), float(ys.max())
    return [
        round(x0 / w, 5),
        round(y0 / h, 5),
        round((x1 - x0) / w, 5),
        round((y1 - y0) / h, 5),
    ]


def suggest_regions(
    frame: np.ndarray,
    *,
    conf: float = 0.25,
    imgsz: int = 640,
    max_regions: int = 36,
    min_area: float = 0.0015,
    max_area: float = 0.85,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Run FastSAM once and return normalized polygon suggestions.
    Sorted largest-first. Setup assist only — not for live monitoring.
    """
    if frame is None or frame.size == 0:
        return [], 0, 0
    h, w = frame.shape[:2]
    model = get_fastsam()
    results = model(frame, conf=conf, imgsz=imgsz, verbose=False, retina_masks=True)
    if not results:
        return [], w, h
    r0 = results[0]
    regions: List[Dict[str, Any]] = []

    polys_px: List[np.ndarray] = []
    if getattr(r0, "masks", None) is not None and r0.masks is not None:
        xy_list = getattr(r0.masks, "xy", None)
        if xy_list is not None and len(xy_list):
            for xy in xy_list:
                poly = _poly_from_xy(xy)
                if poly is not None:
                    polys_px.append(poly)
        else:
            data = r0.masks.data
            if data is not None:
                for i in range(int(data.shape[0])):
                    m = data[i].cpu().numpy()
                    if m.shape[0] != h or m.shape[1] != w:
                        m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
                    poly = _mask_to_polygon(m)
                    if poly is not None:
                        polys_px.append(poly)

    # Fallback: boxes only → rectangle polygons
    if not polys_px and getattr(r0, "boxes", None) is not None and r0.boxes is not None:
        for box in r0.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            x0, y0, x1, y1 = map(float, xyxy)
            polys_px.append(
                np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
            )

    frame_area = float(w * h)
    for poly in polys_px:
        area = abs(cv2.contourArea(poly.reshape(-1, 1, 2))) / frame_area
        if area < min_area or area > max_area:
            continue
        regions.append(
            {
                "id": f"seg-{uuid.uuid4().hex[:10]}",
                "polygon": _normalize_poly(poly, w, h),
                "bbox": _bbox_norm(poly, w, h),
                "area": round(area, 5),
            }
        )

    regions.sort(key=lambda x: x["area"], reverse=True)
    return regions[: max(1, max_regions)], w, h
