"""
embedder.py — InsightFace buffalo_l wrapper for face detection + embedding.

Responsibilities:
    - Lazy-load InsightFace FaceAnalysis model (auto-download on first run)
    - Detect all faces in a frame using SCRFD
    - Extract ArcFace 512-d embeddings for each face
    - Apply quality filtering (face size + sharpness)
    - Return structured FaceResult objects

Hardware: CPU-only via onnxruntime. GPU upgrade = swap onnxruntime package,
no code changes required here (InsightFace picks up CUDA providers automatically).
"""

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from face_recognition.config import (
    INSIGHTFACE_MODEL_PACK, DETECTION_THRESHOLD, MIN_FACE_SIZE_PX,
    SHARPNESS_THRESHOLD, MODELS_CACHE_DIR,
)

log = logging.getLogger(__name__)


@dataclass
class FaceResult:
    """
    Holds all data extracted from a single detected face in one frame.

    Attributes:
        bbox        : [x1, y1, x2, y2] in pixel coordinates
        embedding   : 512-d L2-normalised ArcFace embedding (None if quality failed)
        score       : SCRFD detection confidence (0–1)
        crop        : BGR face-region image crop
        is_quality  : True if face passed size + sharpness quality gates
        kps         : 5-point facial landmarks [[x,y], ...]
    """
    bbox: np.ndarray                      # shape (4,)
    score: float
    crop: np.ndarray                      # BGR image
    is_quality: bool
    kps: Optional[np.ndarray] = None      # shape (5, 2)
    embedding: Optional[np.ndarray] = None  # shape (512,)


class FaceEmbedder:
    """
    Wrapper around InsightFace buffalo_l providing detection and embedding.
    Singleton-safe — create once per pipeline instance.
    """

    def __init__(self):
        self._app = None   # Lazy-loaded on first call
        self._loaded = False

    def _load(self):
        """Load InsightFace model pack (downloads on first run, cached after)."""
        if self._loaded:
            return
        try:
            import insightface
            from insightface.app import FaceAnalysis

            os.makedirs(MODELS_CACHE_DIR, exist_ok=True)

            # providers: CPU-only. InsightFace will pick up CUDAExecutionProvider
            # automatically if onnxruntime-gpu is installed — zero code changes.
            self._app = FaceAnalysis(
                name=INSIGHTFACE_MODEL_PACK,
                root=MODELS_CACHE_DIR,
                providers=["CPUExecutionProvider"],
            )
            # ctx_id=0 → GPU if available; -1 → CPU. We force CPU here.
            # det_size: (640, 640) gives best accuracy; (320, 320) for speed trade-off.
            self._app.prepare(ctx_id=-1, det_size=(640, 640))
            self._loaded = True
            log.info("[FaceEmbedder] InsightFace %s loaded successfully (CPU).", INSIGHTFACE_MODEL_PACK)

        except ImportError as e:
            raise RuntimeError(
                "InsightFace is not installed. Run: pip install insightface onnxruntime"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load InsightFace model: {e}") from e

    def get_faces(self, frame: np.ndarray, extract_embeddings: bool = True) -> List[FaceResult]:
        """
        Run SCRFD detection on a BGR frame, optionally followed by ArcFace embedding.

        Args:
            frame: BGR numpy array from OpenCV
            extract_embeddings: If True, runs the heavy ArcFace model on detected quality faces.
                                If False, only runs SCRFD detection.

        Returns:
            List of FaceResult, one per detected face.
            Faces that fail quality checks have is_quality=False and embedding=None.
        """
        self._load()

        if frame is None or frame.size == 0:
            return []

        try:
            # Bypass self._app.get() to manually decouple detection and recognition
            bboxes, kpss = self._app.det_model.detect(frame)
        except Exception as e:
            log.warning("[FaceEmbedder] Detection failed: %s", e)
            return []
            
        if bboxes.shape[0] == 0:
            return []

        results: List[FaceResult] = []
        from insightface.app.common import Face

        for i in range(bboxes.shape[0]):
            # SCRFD confidence gate
            score = float(bboxes[i, 4])
            if score < DETECTION_THRESHOLD:
                continue

            bbox = bboxes[i, 0:4].astype(int)   # [x1, y1, x2, y2]
            kps = kpss[i] if kpss is not None else None
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1

            # Size gate
            if w < MIN_FACE_SIZE_PX or h < MIN_FACE_SIZE_PX:
                # Still report the face (for tracking) but mark low quality
                crop = _safe_crop(frame, x1, y1, x2, y2)
                results.append(FaceResult(
                    bbox=bbox, score=score, crop=crop,
                    is_quality=False, kps=kps,
                    embedding=None,
                ))
                continue

            crop = _safe_crop(frame, x1, y1, x2, y2)

            # Sharpness gate (Laplacian variance)
            if not _is_sharp(crop):
                results.append(FaceResult(
                    bbox=bbox, score=score, crop=crop,
                    is_quality=False, kps=kps,
                    embedding=None,
                ))
                continue

            # All quality gates passed — extract embedding if requested
            embedding = None
            if extract_embeddings:
                face_obj = Face(bbox=bboxes[i, 0:4], kps=kps, det_score=bboxes[i, 4])
                self._app.models['recognition'].get(frame, face_obj)
                if face_obj.embedding is not None:
                    embedding = face_obj.embedding.astype(np.float32)

            results.append(FaceResult(
                bbox=bbox, score=score, crop=crop,
                is_quality=True, kps=kps,
                embedding=embedding,
            ))

        return results


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_crop(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Clip bbox to frame bounds and return crop. Returns empty array on failure."""
    h, w = frame.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    return frame[y1:y2, x1:x2].copy()


def _is_sharp(crop: np.ndarray) -> bool:
    """
    Estimate sharpness using Laplacian variance.
    Fast, single-call, works on any resolution crop.
    """
    if crop is None or crop.size == 0:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance >= SHARPNESS_THRESHOLD
