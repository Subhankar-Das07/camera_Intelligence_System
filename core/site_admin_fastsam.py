"""FastSAM helpers for Site Admin ROI suggestion (setup assist only)."""

from __future__ import annotations

import logging
import os
import threading
import uuid
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("site_admin.fastsam")

_model = None
_model_lock = threading.Lock()
_model_name = os.environ.get("SITE_ADMIN_FASTSAM_WEIGHTS", "FastSAM-s.pt")


def get_fastsam(weights: Optional[str] = None):
    """Lazy-load and cache FastSAM (do not reload per request)."""
    global _model, _model_name
    weights = (weights or _model_name or "FastSAM-s.pt").strip()
    with _model_lock:
        if _model is None or _model_name != weights:
            from ultralytics import FastSAM

            log.info("Loading FastSAM weights: %s", weights)
            try:
                _model = FastSAM(weights)
                _model_name = weights
            except Exception as e:
                raise RuntimeError(
                    f"FastSAM weights unavailable ({weights}). "
                    "Run download-weights.bat and rebuild the Docker image, "
                    "or set SITE_ADMIN_FASTSAM_WEIGHTS to a local .pt file."
                ) from e
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


def _point_in_poly(nx: float, ny: float, poly: List[List[float]]) -> bool:
    """Ray casting; poly is [[x,y],...] normalized or pixel — same units as nx,ny."""
    if not poly or len(poly) < 3:
        return False
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = float(poly[i][0]), float(poly[i][1])
        xj, yj = float(poly[j][0]), float(poly[j][1])
        if ((yi > ny) != (yj > ny)) and (nx < (xj - xi) * (ny - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _xyxy_from_poly(poly: np.ndarray) -> List[int]:
    xs = poly[:, 0]
    ys = poly[:, 1]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def segment_at_point(
    frame: np.ndarray,
    nx: float,
    ny: float,
    *,
    conf: float = 0.25,
    imgsz: int = 640,
) -> Optional[Dict[str, Any]]:
    """
    FastSAM select object under a normalized click (nx, ny in 0..1).
    Returns polygon (norm), bbox_norm [x1,y1,x2,y2], xyxy pixels, area.
    Prefers point prompt; falls back to everything-mode region containing the click.
    """
    if frame is None or frame.size == 0:
        return None
    h, w = frame.shape[:2]
    px = int(max(0, min(w - 1, round(float(nx) * w))))
    py = int(max(0, min(h - 1, round(float(ny) * h))))
    model = get_fastsam()

    # 1) Try Ultralytics FastSAMPrompt point prompt
    try:
        everything = model(
            frame,
            conf=conf,
            imgsz=imgsz,
            verbose=False,
            retina_masks=True,
        )
        from ultralytics.models.fastsam import FastSAMPrompt

        prompt = FastSAMPrompt(frame, everything, device="cpu")
        ann = prompt.point_prompt(points=[[px, py]], pointlabel=[1])
        # ann is typically a plotted image; masks live on everything after prompt
        # Prefer masks from the prompt process results
        masks = None
        if everything and getattr(everything[0], "masks", None) is not None:
            # Filter: find mask containing point among all, prefer smallest
            data = everything[0].masks.data
            if data is not None:
                candidates = []
                for i in range(int(data.shape[0])):
                    m = data[i].cpu().numpy()
                    if m.shape[0] != h or m.shape[1] != w:
                        m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
                    if m[py, px] > 0.5:
                        area = float((m > 0.5).sum()) / float(w * h)
                        candidates.append((area, m))
                if candidates:
                    candidates.sort(key=lambda t: t[0])
                    masks = candidates[0][1]
        if masks is None and ann is not None:
            # Some versions return annotation with masks attribute
            pass
        if masks is not None:
            poly = _mask_to_polygon(masks)
            if poly is not None and len(poly) >= 3:
                xyxy = _xyxy_from_poly(poly)
                area = abs(cv2.contourArea(poly.reshape(-1, 1, 2))) / float(w * h)
                return {
                    "polygon": _normalize_poly(poly, w, h),
                    "bbox_norm": [
                        round(xyxy[0] / w, 5),
                        round(xyxy[1] / h, 5),
                        round(xyxy[2] / w, 5),
                        round(xyxy[3] / h, 5),
                    ],
                    "xyxy": xyxy,
                    "area": round(area, 5),
                    "method": "point_prompt",
                    "click": [px, py],
                }
    except Exception as e:
        log.warning("FastSAM point_prompt failed, falling back: %s", e)

    # 2) Fallback: everything regions — smallest polygon containing click
    regions, _, _ = suggest_regions(frame, conf=conf, imgsz=imgsz, max_regions=48, min_area=0.0004)
    hits = []
    for r in regions:
        poly = r.get("polygon") or []
        if _point_in_poly(nx, ny, poly):
            hits.append(r)
    if not hits:
        # Expand: nearest region centroid
        best = None
        best_d = 1e9
        for r in regions:
            poly = r.get("polygon") or []
            if len(poly) < 3:
                continue
            cx = sum(p[0] for p in poly) / len(poly)
            cy = sum(p[1] for p in poly) / len(poly)
            d = (cx - nx) ** 2 + (cy - ny) ** 2
            if d < best_d:
                best_d = d
                best = r
        if best is None or best_d > 0.04:  # ~0.2 normalized radius
            return None
        hits = [best]
    hits.sort(key=lambda r: float(r.get("area") or 1))
    chosen = hits[0]
    poly_n = chosen["polygon"]
    pts = np.array([[p[0] * w, p[1] * h] for p in poly_n], dtype=np.float32)
    xyxy = _xyxy_from_poly(pts)
    return {
        "polygon": poly_n,
        "bbox_norm": [
            round(xyxy[0] / w, 5),
            round(xyxy[1] / h, 5),
            round(xyxy[2] / w, 5),
            round(xyxy[3] / h, 5),
        ],
        "xyxy": xyxy,
        "area": chosen.get("area"),
        "method": "contain_or_nearest",
        "click": [px, py],
    }
