"""
scene_memory.py
===============
Persistent object memory for the Smart Vision Assistant.

Tracks every detected object across frames, maintaining:
  - First/last seen timestamps
  - Object category (human, vehicle, animal, etc.)
  - New/active/removed status
  - Gemini-verified label vs YOLO label
  - Duration in scene

The 80 COCO classes are mapped to 10 human-readable categories
for structured scene reporting.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import config

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# COCO → Category Mapping
# 80 COCO classes → 10 scene categories
# ──────────────────────────────────────────────────────────────────────────────

CATEGORY_MAP: Dict[str, str] = {
    # Human
    "person":           "Humans",

    # Vehicles
    "bicycle":          "Vehicles",
    "car":              "Vehicles",
    "motorcycle":       "Vehicles",
    "airplane":         "Vehicles",
    "bus":              "Vehicles",
    "train":            "Vehicles",
    "truck":            "Vehicles",
    "boat":             "Vehicles",

    # Animals
    "bird":             "Animals",
    "cat":              "Animals",
    "dog":              "Animals",
    "horse":            "Animals",
    "sheep":            "Animals",
    "cow":              "Animals",
    "elephant":         "Animals",
    "bear":             "Animals",
    "zebra":            "Animals",
    "giraffe":          "Animals",

    # Electronics
    "tv":               "Electronics",
    "laptop":           "Electronics",
    "mouse":            "Electronics",
    "remote":           "Electronics",
    "keyboard":         "Electronics",
    "cell phone":       "Electronics",

    # Furniture
    "chair":            "Furniture",
    "couch":            "Furniture",
    "bed":              "Furniture",
    "dining table":     "Furniture",
    "toilet":           "Furniture",

    # Kitchen & Food
    "bottle":           "Kitchen",
    "wine glass":       "Kitchen",
    "cup":              "Kitchen",
    "fork":             "Kitchen",
    "knife":            "Kitchen",
    "spoon":            "Kitchen",
    "bowl":             "Kitchen",
    "banana":           "Kitchen",
    "apple":            "Kitchen",
    "sandwich":         "Kitchen",
    "orange":           "Kitchen",
    "broccoli":         "Kitchen",
    "carrot":           "Kitchen",
    "hot dog":          "Kitchen",
    "pizza":            "Kitchen",
    "donut":            "Kitchen",
    "cake":             "Kitchen",
    "microwave":        "Kitchen",
    "oven":             "Kitchen",
    "toaster":          "Kitchen",
    "sink":             "Kitchen",
    "refrigerator":     "Kitchen",

    # Sports
    "frisbee":          "Sports",
    "skis":             "Sports",
    "snowboard":        "Sports",
    "sports ball":      "Sports",
    "kite":             "Sports",
    "baseball bat":     "Sports",
    "baseball glove":   "Sports",
    "skateboard":       "Sports",
    "surfboard":        "Sports",
    "tennis racket":    "Sports",

    # Tools & Personal
    "scissors":         "Tools",
    "hair drier":       "Tools",
    "toothbrush":       "Tools",

    # Containers & Accessories
    "backpack":         "Containers",
    "umbrella":         "Containers",
    "handbag":          "Containers",
    "tie":              "Containers",
    "suitcase":         "Containers",

    # Household
    "potted plant":     "Household",
    "clock":            "Household",
    "vase":             "Household",
    "book":             "Household",
    "traffic light":    "Household",
    "fire hydrant":     "Household",
    "stop sign":        "Household",
    "parking meter":    "Household",
    "bench":            "Household",
}

# All known categories in display order
CATEGORY_ORDER = [
    "Humans", "Vehicles", "Animals", "Electronics",
    "Furniture", "Kitchen", "Sports", "Tools", "Containers", "Household",
]


def get_category(label: str) -> str:
    """Map a YOLO/Gemini label to a scene category. Returns 'Other' if unknown."""
    return CATEGORY_MAP.get(label.lower(), "Other")


# ──────────────────────────────────────────────────────────────────────────────
# ObjectRecord
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ObjectRecord:
    """Represents one tracked object across its entire lifetime in the scene."""
    track_id: int
    yolo_label: str                   # Original YOLO class name
    display_label: str                # Current best label (YOLO or Gemini-refined)
    category: str                     # Scene category
    first_seen: float                 # time.monotonic() at first detection
    last_seen: float                  # time.monotonic() at most recent detection
    bbox: Tuple[int, int, int, int]  # Most recent bounding box
    confidence: float                 # Most recent confidence score
    gemini_verified: bool = False     # Has Gemini refined this label?
    is_new: bool = True               # True until OBJECT_NEW_GRACE seconds have elapsed
    gemini_description: Optional[str] = None  # Rich Gemini per-object description (bullet points)
    
    raw_bbox: Optional[Tuple[int, int, int, int]] = None
    ema_alpha: float = 0.2
    
    # Event tracking states
    last_centroid: Optional[Tuple[float, float]] = None
    anchor_centroid: Optional[Tuple[float, float]] = None
    anchor_time: float = 0.0
    is_stationary: bool = False
    last_event: str = 'new'
    was_near_person: bool = False
    unattended_since: float = 0.0
    gemini_skipped_reason: Optional[str] = None

    @property
    def duration(self) -> float:
        """Seconds this object has been in the scene."""
        return self.last_seen - self.first_seen

    @property
    def age_since_last_seen(self) -> float:
        """Seconds since the object was last detected."""
        return time.monotonic() - self.last_seen

    def duration_str(self) -> str:
        """Human-readable duration string."""
        d = self.duration
        if d < 60:
            return f"{d:.0f}s"
        return f"{d/60:.1f}m"


# ──────────────────────────────────────────────────────────────────────────────
# SceneMemory
# ──────────────────────────────────────────────────────────────────────────────

class SceneMemory:
    """
    Maintains persistent awareness of all objects seen in the webcam feed.

    Usage in main loop:
        new_ids, removed_ids = memory.update(detections, verification_cache)
        snapshot = memory.get_scene_snapshot()
    """

    def __init__(self) -> None:
        # Active objects: track_id → ObjectRecord
        self._active: Dict[int, ObjectRecord] = {}
        # Recently removed: track_id → ObjectRecord (held for reporting)
        self._recently_removed: Dict[int, ObjectRecord] = {}
        self._removed_timeout: float = 15.0    # How long to remember removed objects
        # Stability tracking: set of label-sets seen per report interval
        self._stability_history: List[float] = []
        self._last_label_set: frozenset = frozenset()
        self._stable_since: float = time.monotonic()

    # ──────────────────────────────────────────────────────────────────────────
    # Core Update
    # ──────────────────────────────────────────────────────────────────────────

    def update(
        self,
        detections,                       # List[Detection]
        verification_cache: Dict,
    ) -> Tuple[List[int], List[int]]:
        """
        Process one frame's detections.

        Returns:
            new_ids     — track IDs that first appeared this frame
            removed_ids — track IDs evicted due to OBJECT_STALE_TIMEOUT
        """
        now = time.monotonic()
        seen_ids: Set[int] = set()

        new_ids: List[int] = []

        for det in detections:
            tid = det.track_id
            if tid is None:
                continue

            # Determine best label (Gemini-refined takes priority)
            gemini_data = verification_cache.get(tid)
            display_label = gemini_data.get("label", det.label) if gemini_data else det.label
            gemini_verified = bool(gemini_data and gemini_data.get("label"))
            category = get_category(display_label)

            seen_ids.add(tid)

            if tid in self._active:
                # Update existing record
                rec = self._active[tid]
                rec.last_seen = now
                rec.raw_bbox = det.bbox
                
                # Apply EMA Smoothing
                alpha = rec.ema_alpha
                x1, y1, x2, y2 = rec.bbox
                nx1, ny1, nx2, ny2 = det.bbox
                
                # Hard reset EMA if tracker jumped aggressively
                width = max(1, x2 - x1)
                if abs(nx1 - x1) > width * 3.0:
                    rec.bbox = det.bbox
                else:
                    rec.bbox = (
                        int(alpha * nx1 + (1 - alpha) * x1),
                        int(alpha * ny1 + (1 - alpha) * y1),
                        int(alpha * nx2 + (1 - alpha) * x2),
                        int(alpha * ny2 + (1 - alpha) * y2)
                    )
                
                if getattr(config, 'DEBUG_SPATIAL_JITTER', False):
                    log.info("[Jitter Debug] #%d Raw: %s EMA: %s", tid, rec.raw_bbox, rec.bbox)
                    
                rec.confidence = det.confidence
                rec.display_label = display_label
                rec.category = category
                rec.gemini_verified = gemini_verified
                # Pull in Gemini description if newly available
                if gemini_data:
                    if gemini_data.get("description") and not rec.gemini_description:
                        rec.gemini_description = gemini_data["description"]
                    if gemini_data.get("skipped_reason"):
                        rec.gemini_skipped_reason = gemini_data["skipped_reason"]

                # Age out the "new" flag
                if rec.is_new and (now - rec.first_seen) > config.OBJECT_NEW_GRACE:
                    rec.is_new = False
            else:
                # New object entering scene
                rec = ObjectRecord(
                    track_id=tid,
                    yolo_label=det.label,
                    display_label=display_label,
                    category=category,
                    first_seen=now,
                    last_seen=now,
                    bbox=det.bbox,
                    raw_bbox=det.bbox,
                    confidence=det.confidence,
                    gemini_verified=gemini_verified,
                    is_new=True,
                )
                self._active[tid] = rec
                new_ids.append(tid)
                # If it was previously removed, pull it out of removed dict
                self._recently_removed.pop(tid, None)
                log.info(
                    "[Memory] New object: #%d '%s' (%s)",
                    tid, display_label, category,
                )

        # Evict stale objects
        removed_ids = self._evict_stale(seen_ids, now)

        # Clean old removed records
        self._cleanup_removed(now)

        # Update stability tracking
        self._update_stability(seen_ids)

        return new_ids, removed_ids

    # ──────────────────────────────────────────────────────────────────────────
    # Eviction & Cleanup
    # ──────────────────────────────────────────────────────────────────────────

    def _evict_stale(self, seen_ids: Set[int], now: float) -> List[int]:
        """Remove objects not seen recently. Returns list of evicted track IDs."""
        to_remove = [
            tid for tid, rec in self._active.items()
            if tid not in seen_ids
            and (now - rec.last_seen) > config.OBJECT_STALE_TIMEOUT
        ]
        removed_ids = []
        for tid in to_remove:
            rec = self._active.pop(tid)
            rec.is_new = False
            self._recently_removed[tid] = rec
            removed_ids.append(tid)
            log.info(
                "[Memory] Removed: #%d '%s' (was in scene for %s)",
                tid, rec.display_label, rec.duration_str(),
            )
        return removed_ids

    def _cleanup_removed(self, now: float) -> None:
        """Evict from recently_removed after they're too old to report."""
        expired = [
            tid for tid, rec in self._recently_removed.items()
            if (now - rec.last_seen) > self._removed_timeout
        ]
        for tid in expired:
            del self._recently_removed[tid]

    # ──────────────────────────────────────────────────────────────────────────
    # Stability Tracking
    # ──────────────────────────────────────────────────────────────────────────

    def _update_stability(self, seen_ids: Set[int]) -> None:
        current_set = frozenset(seen_ids)
        if current_set != self._last_label_set:
            self._stable_since = time.monotonic()
            self._last_label_set = current_set

    def get_stability_pct(self) -> float:
        """
        Percentage of STABILITY_WINDOW seconds during which the scene was unchanged.
        100% = nothing moved in/out for the whole window.
        """
        stable_duration = time.monotonic() - self._stable_since
        pct = min(100.0, (stable_duration / config.STABILITY_WINDOW) * 100.0)
        return round(pct, 1)

    # ──────────────────────────────────────────────────────────────────────────
    # Query API
    # ──────────────────────────────────────────────────────────────────────────

    def get_record(self, track_id: int) -> Optional[ObjectRecord]:
        """Return the ObjectRecord for a given track ID, or None."""
        return self._active.get(track_id)

    def get_active_records(self) -> List[ObjectRecord]:
        """All currently visible ObjectRecords."""
        return list(self._active.values())

    def get_recently_removed(self) -> List[ObjectRecord]:
        """Objects that left the scene recently (within _removed_timeout)."""
        return list(self._recently_removed.values())

    def get_recently_removed_dict(self) -> Dict[int, ObjectRecord]:
        """Direct dict access for O(1) track-id lookup (no list copy)."""
        return self._recently_removed

    def get_new_objects(self) -> List[ObjectRecord]:
        """Active objects still within their new-object grace period."""
        return [r for r in self._active.values() if r.is_new]

    def get_scene_snapshot(self) -> Dict[str, List[ObjectRecord]]:
        """
        Returns active objects grouped by category.
        Example: {"Humans": [rec1], "Electronics": [rec2, rec3], ...}
        """
        snapshot: Dict[str, List[ObjectRecord]] = {}
        for rec in self._active.values():
            snapshot.setdefault(rec.category, []).append(rec)
        return snapshot

    def get_object_duration(self, track_id: int) -> float:
        """Seconds an active object has been in the scene. Returns 0 if not found."""
        rec = self._active.get(track_id)
        return rec.duration if rec else 0.0

    def active_count(self) -> int:
        return len(self._active)

    def should_verify(self, track_id: int) -> bool:
        """True if this object has been in scene long enough to be worth verifying."""
        rec = self._active.get(track_id)
        if rec is None:
            return False
        return rec.duration >= config.GEMINI_MIN_OBJECT_AGE
