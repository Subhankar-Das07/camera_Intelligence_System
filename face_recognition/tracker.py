"""
tracker.py — ByteTrack face tracker wrapper using the Supervision library.

Responsibilities:
    - Accept a list of FaceResult objects per frame
    - Convert to supervision.Detections format
    - Feed into ByteTrack to get stable track IDs
    - Return annotated FaceResult list with .track_id added
    - Maintain per-track candidate embedding buffers
    - Manage track lifecycle (active, lost, removed)

Why ByteTrack:
    Tracker runs every frame; recognition runs every N-th frame.
    ByteTrack keeps a stable integer ID for each face across frames,
    so we only run the expensive ArcFace inference once we have high confidence.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from face_recognition.config import CANDIDATE_MIN_EMBEDDINGS

log = logging.getLogger(__name__)


@dataclass
class TrackedFace:
    """
    Combines FaceResult data with a ByteTrack-assigned track ID.
    """
    track_id: int
    bbox: np.ndarray          # [x1, y1, x2, y2]
    score: float
    crop: np.ndarray          # BGR crop
    is_quality: bool
    embedding: Optional[np.ndarray] = None   # 512-d, None if low-quality
    kps: Optional[np.ndarray] = None


class FaceTracker:
    """
    Wraps supervision.ByteTrack with face-specific bookkeeping.

    Candidate buffer: per-track, we accumulate quality embeddings before
    committing to a recognition query. This prevents spurious identities
    from partial / angled views.
    """

    def __init__(self):
        self._tracker = None      # Lazy-loaded
        self._loaded = False

        # track_id → list of (embedding, crop) tuples (quality frames only)
        self._candidate_buffers: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}

        # track_id → True if this track has been fully identified
        self._identified_tracks: Dict[int, bool] = {}

    def _load(self):
        if self._loaded:
            return
        try:
            import supervision as sv
            # ByteTrack default parameters are well-tuned for face tracking
            self._tracker = sv.ByteTrack(
                track_activation_threshold=0.4,   # min det score to start a track
                lost_track_buffer=30,              # frames to keep a lost track alive
                minimum_matching_threshold=0.8,   # IOU threshold for association
                frame_rate=30,
            )
            self._sv = sv
            self._loaded = True
            log.info("[FaceTracker] ByteTrack loaded via supervision.")
        except ImportError as e:
            raise RuntimeError(
                "Supervision library not installed. Run: pip install supervision"
            ) from e

    def update(self, face_results, frame: np.ndarray) -> List[TrackedFace]:
        """
        Update tracker with detections from the current frame.

        Args:
            face_results: List[FaceResult] from FaceEmbedder.detect_and_embed()
            frame:        current BGR frame (needed for supervision Detections)

        Returns:
            List[TrackedFace] with assigned track_ids.
        """
        self._load()

        if not face_results:
            # Let ByteTrack know there are no detections this frame
            empty = self._sv.Detections.empty()
            self._tracker.update_with_detections(empty)
            return []

        # Build supervision Detections from FaceResults
        xyxy   = np.array([fr.bbox for fr in face_results], dtype=np.float32)
        confs  = np.array([fr.score for fr in face_results], dtype=np.float32)

        detections = self._sv.Detections(
            xyxy=xyxy,
            confidence=confs,
        )

        tracked = self._tracker.update_with_detections(detections)

        # Map updated detections back to FaceResults by index
        # supervision returns tracked detections in the same order as input detections
        # but may drop some (lost tracks) or return fewer.
        # We match by bounding-box overlap.
        tracked_faces: List[TrackedFace] = []

        if tracked.tracker_id is None:
            return []

        for i, tid in enumerate(tracked.tracker_id):
            if tid is None:
                continue
            # Find the closest matching FaceResult by IoU
            tracked_bbox = tracked.xyxy[i]
            best_fr = _match_bbox(tracked_bbox, face_results)
            if best_fr is None:
                continue

            tracked_face = TrackedFace(
                track_id=int(tid),
                bbox=best_fr.bbox,
                score=best_fr.score,
                crop=best_fr.crop,
                is_quality=best_fr.is_quality,
                embedding=best_fr.embedding,
                kps=best_fr.kps,
            )
            tracked_faces.append(tracked_face)

            # Update candidate buffer for quality frames
            if best_fr.is_quality and best_fr.embedding is not None:
                if int(tid) not in self._candidate_buffers:
                    self._candidate_buffers[int(tid)] = []
                buf = self._candidate_buffers[int(tid)]
                # Keep buffer capped at 10 (more than enough for 4-embedding threshold)
                if len(buf) < 10:
                    buf.append((best_fr.embedding.copy(), best_fr.crop.copy()))

        return tracked_faces

    def get_candidate_embeddings(self, track_id: int) -> Optional[Tuple[np.ndarray, List[np.ndarray]]]:
        """
        If this track has accumulated enough quality embeddings, return them.

        Returns:
            (embedding_matrix, crop_list) where embedding_matrix is (N, 512),
            or None if not enough embeddings yet.
        """
        buf = self._candidate_buffers.get(track_id)
        if not buf or len(buf) < CANDIDATE_MIN_EMBEDDINGS:
            return None
        embeddings = np.stack([e for e, _ in buf], axis=0)
        crops = [c for _, c in buf]
        return embeddings, crops

    def clear_candidate_buffer(self, track_id: int):
        """Clear the candidate buffer after identity has been resolved."""
        self._candidate_buffers.pop(track_id, None)

    def mark_identified(self, track_id: int):
        """Mark a track as having a resolved identity."""
        self._identified_tracks[track_id] = True

    def is_identified(self, track_id: int) -> bool:
        """Check if track already has a resolved identity."""
        return self._identified_tracks.get(track_id, False)

    def reset_track(self, track_id: int):
        """Remove all state for a track (called when track is lost)."""
        self._candidate_buffers.pop(track_id, None)
        self._identified_tracks.pop(track_id, None)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def _match_bbox(tracked_bbox: np.ndarray, face_results) -> Optional[object]:
    """Find the FaceResult with highest IoU overlap to a tracked bbox."""
    best_fr = None
    best_iou = 0.2   # Minimum overlap threshold
    for fr in face_results:
        iou_val = _iou(tracked_bbox, fr.bbox.astype(float))
        if iou_val > best_iou:
            best_iou = iou_val
            best_fr = fr
    return best_fr
