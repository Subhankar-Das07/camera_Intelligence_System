"""Person appearance Re-ID embeddings (OSNet ONNX when available, OpenCV fallback)."""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)

REID_DIM = 512
_DEFAULT_WEIGHTS = (
    os.environ.get("SITE_ADMIN_REID_WEIGHTS")
    or "models/osnet_x0_25.onnx"
)


def reid_enabled() -> bool:
    return os.environ.get("JOURNEY_REID_ENABLED", "1").lower() not in ("0", "false", "no", "off")


class PersonReID:
    """Embed person crops for cross-camera matching."""

    def __init__(self, weights_path: Optional[str] = None) -> None:
        self._lock = threading.Lock()
        self._session = None
        self._input_name = None
        self._backend = "none"
        self._weights_path = weights_path or _DEFAULT_WEIGHTS
        self._try_load()

    def _try_load(self) -> None:
        path = Path(self._weights_path)
        candidates = [path, Path("models") / path.name, Path("/app") / path.name, Path("/app/models") / path.name]
        onnx_path = next((p for p in candidates if p.is_file() and p.stat().st_size > 10_000), None)
        if onnx_path is None:
            self._backend = "opencv"
            log.info("PersonReID: OSNet ONNX not found — using OpenCV appearance fallback")
            return
        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            self._session = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
            self._input_name = self._session.get_inputs()[0].name
            self._backend = "osnet"
            log.info("PersonReID: loaded OSNet from %s", onnx_path)
        except Exception as e:
            log.warning("PersonReID: failed to load %s (%s) — OpenCV fallback", onnx_path, e)
            self._session = None
            self._backend = "opencv"

    @property
    def backend(self) -> str:
        return self._backend

    def embed(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        if crop_bgr is None or not isinstance(crop_bgr, np.ndarray) or crop_bgr.size == 0:
            return None
        h, w = crop_bgr.shape[:2]
        if h < 16 or w < 8:
            return None
        with self._lock:
            if self._backend == "osnet" and self._session is not None:
                return self._embed_osnet(crop_bgr)
            return self._embed_opencv(crop_bgr)

    def _embed_osnet(self, crop_bgr: np.ndarray) -> np.ndarray:
        # OSNet typically expects 256x128 RGB, ImageNet-normalized NCHW
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (128, 256), interpolation=cv2.INTER_LINEAR)
        x = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = (x - mean) / std
        x = np.transpose(x, (2, 0, 1))[None, ...]
        outs = self._session.run(None, {self._input_name: x})
        vec = np.asarray(outs[0], dtype=np.float32).reshape(-1)
        return _l2_pad(vec, REID_DIM)

    def _embed_opencv(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Compact appearance vector: HSV hist + vertical color strips + aspect."""
        resized = cv2.resize(crop_bgr, (64, 128), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        parts: List[np.ndarray] = []
        for i, ch in enumerate(cv2.split(hsv)):
            bins = 32 if i == 0 else 16
            hist = cv2.calcHist([ch], [0], None, [bins], [0, 180 if i == 0 else 256])
            hist = hist.flatten().astype(np.float32)
            s = float(hist.sum()) + 1e-6
            parts.append(hist / s)
        # 4 vertical bands × mean BGR
        bands = np.array_split(resized, 4, axis=0)
        for band in bands:
            mean = band.reshape(-1, 3).mean(axis=0).astype(np.float32) / 255.0
            parts.append(mean)
        h, w = crop_bgr.shape[:2]
        parts.append(np.array([h / max(w, 1), w / max(h, 1)], dtype=np.float32))
        vec = np.concatenate(parts)
        return _l2_pad(vec, REID_DIM)


def _l2_pad(vec: np.ndarray, dim: int) -> np.ndarray:
    out = np.zeros(dim, dtype=np.float32)
    n = min(dim, int(vec.size))
    out[:n] = vec.reshape(-1)[:n]
    norm = float(np.linalg.norm(out))
    if norm > 1e-8:
        out /= norm
    return out


_reid: Optional[PersonReID] = None
_reid_lock = threading.Lock()


def get_person_reid() -> PersonReID:
    global _reid
    if _reid is not None:
        return _reid
    with _reid_lock:
        if _reid is None:
            _reid = PersonReID()
        return _reid


def embed_person_crop(crop_bgr: np.ndarray) -> Optional[np.ndarray]:
    if not reid_enabled():
        return None
    return get_person_reid().embed(crop_bgr)


def crop_from_xyxy(frame: np.ndarray, xyxy: Tuple[int, int, int, int], pad: float = 0.05) -> Optional[np.ndarray]:
    if frame is None or frame.size == 0:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    px, py = int(bw * pad), int(bh * pad)
    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(w, x2 + px)
    y2 = min(h, y2 + py)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()
