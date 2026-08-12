"""
services/watchlist_manager.py
==============================
Object Watchlist Manager — Real-Time Object Monitoring.

Design Principles:
  - Completely isolated from the core vision pipeline.
  - Consumes ONLY EntityRegistry state (already computed every frame).
  - Zero additional inference, zero detector interaction, zero FPS cost.
  - In-memory only (v1). No database writes.

State Machine:
  IDLE
    ↓  (user selects object + starts tracking)
  VISIBLE
    ↓  (entity.is_active == False for >= WATCHLIST_MISSING_THRESHOLD seconds)
  LOST  (alert_active = True)
    ↓  (entity.is_active == True again)
  RETURNED  (alert_active = False, returned banner shown for 5s)
    ↓  (5 seconds elapsed)
  VISIBLE

Watch Status Strings (used by frontend):
  'idle'      — nothing being tracked
  'visible'   — tracked object is currently in camera view
  'lost'      — tracked object has been missing > threshold seconds
  'returned'  — tracked object just reappeared (5-second grace display)
"""

import time
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
WATCHLIST_MISSING_THRESHOLD: float = 3.0   # Seconds before LOST alert fires
RETURNED_DISPLAY_DURATION:   float = 5.0   # Seconds to show RETURNED banner

# Grace window: if entity was seen within this many seconds, treat as still present.
# This absorbs brief tracker drops during object movement (BoT-SORT can drop 1-3 frames).
# Value must be > OBJECT_STALE_TIMEOUT (2.0s in config) to bridge the gap.
WATCHLIST_SEEN_GRACE_WINDOW: float = 4.0


# ── Position Grid Helper ───────────────────────────────────────────────────────

def _compute_position_label(bbox, frame_w: int = 640, frame_h: int = 480) -> str:
    """
    Convert a bounding box (x1, y1, x2, y2) into a human-readable
    3×3 grid position: Top Left, Center, Bottom Right, etc.

    Uses the centroid of the bounding box relative to the frame dimensions.
    Falls back gracefully if bbox is None or malformed.
    """
    try:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # Thirds
        col = cx / frame_w
        row = cy / frame_h

        if col < 0.333:
            h = "Left"
        elif col < 0.667:
            h = "Center"
        else:
            h = "Right"

        if row < 0.333:
            v = "Top"
        elif row < 0.667:
            v = "Center"
        else:
            v = "Bottom"

        # Compact: "Center Center" → "Center", otherwise "Top Left" etc.
        if v == "Center" and h == "Center":
            return "Center"
        return f"{v} {h}"
    except Exception:
        return "Unknown"


# ── WatchItem ─────────────────────────────────────────────────────────────────

@dataclass
class WatchItem:
    """
    Single tracked object.

    Fields
    ------
    entity_uuid      : Stable UUID from EntityRegistry (never changes).
    display_label    : Human-readable label (e.g. "Laptop", "Person").
    category         : Entity category (e.g. "Electronics", "Humans").
    watch_started_at : monotonic() timestamp when tracking began.

    active           : True if the entity is currently in the scene.
    alert_active     : True when missing > WATCHLIST_MISSING_THRESHOLD.
    returned_at      : monotonic() when entity reappeared (for RETURNED banner).

    last_seen        : wall-clock datetime string of last confirmed visibility.
    last_seen_ts     : monotonic() timestamp for elapsed calculations.
    last_confidence  : Most recent confidence score (0-1).
    last_position    : Grid position string at last confirmed sighting.
    seconds_missing  : Accumulated seconds since entity went inactive.
    """
    entity_uuid:      str
    display_label:    str
    category:         str
    watch_started_at: float = field(default_factory=time.monotonic)

    active:           bool  = True
    alert_active:     bool  = False
    returned_at:      Optional[float] = None

    last_seen:        str   = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))
    last_seen_ts:     float = field(default_factory=time.monotonic)
    last_confidence:  float = 0.0
    last_position:    str   = "Unknown"
    seconds_missing:  float = 0.0

    # In-memory event log (last 50 entries)
    events: List[Dict[str, str]] = field(default_factory=list)

    # ── Derived ───────────────────────────────────────────────────────────────

    @property
    def watch_status(self) -> str:
        """UI-facing status string."""
        if self.alert_active:
            return "lost"
        if self.returned_at is not None:
            elapsed = time.monotonic() - self.returned_at
            if elapsed < RETURNED_DISPLAY_DURATION:
                return "returned"
            # Grace window expired — back to visible
        return "visible"

    @property
    def observed_for(self) -> float:
        """Total seconds since tracking began."""
        return time.monotonic() - self.watch_started_at

    @property
    def observed_for_str(self) -> str:
        """Human-readable observed duration."""
        s = int(self.observed_for)
        if s < 60:
            return f"{s}s"
        return f"{s // 60}m {s % 60:02d}s"

    @property
    def seconds_missing_str(self) -> str:
        """Human-readable missing duration."""
        s = int(self.seconds_missing)
        if s < 60:
            return f"{s}s"
        return f"{s // 60}m {s % 60:02d}s"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_uuid":      self.entity_uuid,
            "display_label":    self.display_label,
            "category":         self.category,
            "watch_status":     self.watch_status,
            "active":           self.active,
            "alert_active":     self.alert_active,
            "last_seen":        self.last_seen,
            "last_confidence":  round(self.last_confidence * 100),   # percent int
            "last_position":    self.last_position,
            "seconds_missing":  round(self.seconds_missing, 1),
            "seconds_missing_str": self.seconds_missing_str,
            "observed_for":     round(self.observed_for, 1),
            "observed_for_str": self.observed_for_str,
            "recent_events":    self.events[-10:],
        }

    def _log_event(self, event_type: str, detail: str = "") -> None:
        entry = {
            "time":   datetime.now().strftime("%H:%M:%S"),
            "type":   event_type,
            "detail": detail,
        }
        self.events.append(entry)
        if len(self.events) > 50:
            self.events = self.events[-50:]


# ── WatchlistManager ──────────────────────────────────────────────────────────

class WatchlistManager:
    """
    Manages a single active WatchItem (v1 — one object at a time).

    Thread-safety note:
      tick() is called from the WebSocket coroutine (asyncio thread).
      watch() / stop() are called from FastAPI request handlers (asyncio thread).
      Both run in the same event loop → no lock needed for asyncio.
      If ever called from a background thread, wrap with asyncio.Lock.

    Usage
    -----
      manager = WatchlistManager()

      # Called once on user request
      manager.watch(entity_uuid, label, category)

      # Called once per WebSocket push cycle (1s)
      manager.tick(entity_registry)

      # API read
      status = manager.get_status()          # → dict (always safe to call)
      available = manager.get_available(reg) # → list[dict]
    """

    def __init__(self) -> None:
        self._item: Optional[WatchItem] = None
        log.info("[WatchlistManager] Initialised (in-memory, v1)")

    # ── Control ───────────────────────────────────────────────────────────────

    def watch(self, entity_uuid: str, display_label: str, category: str) -> None:
        """
        Begin watching the specified entity.
        Replaces any existing watch item (one at a time in v1).
        """
        self._item = WatchItem(
            entity_uuid=entity_uuid,
            display_label=display_label,
            category=category,
        )
        self._item._log_event("WATCH_STARTED", f"Started monitoring '{display_label}'")
        log.info("[WatchlistManager] Watching entity %s ('%s')", entity_uuid[:8], display_label)

    def stop(self) -> None:
        """Stop watching and clear the current item."""
        if self._item:
            self._item._log_event("WATCH_STOPPED", f"Stopped monitoring '{self._item.display_label}'")
            log.info("[WatchlistManager] Stopped watching '%s'", self._item.display_label)
        self._item = None

    # ── Tick (called every WebSocket push — 1s cadence) ───────────────────────

    def tick(self, entity_registry, elapsed: float = 1.0) -> None:
        """
        Update the watch item against current entity registry state.

        Presence Logic (key design decision):
          We do NOT use entity.is_active as the sole signal.
          Reason: BoT-SORT can briefly lose a track during fast movement (1-3 frames).
          SceneMemory evicts the record after OBJECT_STALE_TIMEOUT (2s), which makes
          EntityRegistry mark the entity INACTIVE — even though the object is still
          physically in the frame and will be re-detected next cycle.

          Instead we use entity.last_seen_ago:
            - If the entity was seen within WATCHLIST_SEEN_GRACE_WINDOW seconds → PRESENT
            - Only if unseen for longer → start accumulating missing time

          This means: object movement that causes brief track loss never triggers an alert.
          Only a genuine exit from the camera field of view (unseen > threshold) triggers.

        Parameters
        ----------
        entity_registry : EntityRegistry instance
        elapsed         : Seconds since last tick (caller-provided for accuracy).
        """
        if self._item is None:
            return

        item = self._item
        now = time.monotonic()

        # Find the tracked entity in the registry
        entity = entity_registry._session.get(item.entity_uuid)

        if entity is None:
            # Entity not in registry (should not happen after valid watch() call)
            if not item.alert_active:
                item.seconds_missing += elapsed
                if item.seconds_missing >= WATCHLIST_MISSING_THRESHOLD:
                    self._trigger_lost(item)
            else:
                item.seconds_missing += elapsed
            return

        last_seen_ago = now - entity.last_seen   # seconds since last confirmed detection

        # ── UUID self-healing fallback (secondary defense) ────────────────────
        # Primary fix: entity_registry deactivates before relinking, so the same
        # UUID normally survives track_id changes. This fallback handles edge cases
        # where relinking still didn't happen (label mismatch, Gemini rename, etc.):
        # If the original entity is permanently absent (beyond grace window) but an
        # ACTIVE entity with the same label exists, silently re-anchor to it.
        if not entity.is_active and last_seen_ago > WATCHLIST_SEEN_GRACE_WINDOW:
            label_lc = item.display_label.lower()
            replacement = None
            replacement_last_seen = -1.0
            for e in entity_registry._session.values():
                if (
                    e.is_active
                    and e.label.lower() == label_lc
                    and e.category == item.category
                    and e.last_seen > replacement_last_seen
                ):
                    replacement = e
                    replacement_last_seen = e.last_seen
            if replacement is not None:
                log.info(
                    "[WatchlistManager] UUID re-anchor %s -> %s ('%s')",
                    item.entity_uuid[:8], replacement.entity_id[:8], item.display_label,
                )
                item.entity_uuid = replacement.entity_id
                entity = replacement
                last_seen_ago = now - entity.last_seen

        # ── True presence check: use last_seen_ago, NOT just is_active ────────
        # An entity seen within WATCHLIST_SEEN_GRACE_WINDOW seconds is "still present"
        # even if the tracker briefly dropped it (movement, partial occlusion, etc.)
        currently_present = (
            entity.is_active
            or last_seen_ago < WATCHLIST_SEEN_GRACE_WINDOW
        )

        if currently_present:
            # ── Object is in scene ────────────────────────────────────────────
            was_lost = item.alert_active
            item.active = True
            item.seconds_missing = 0.0  # reset immediately

            # Update live metadata only when actively tracked (is_active True)
            if entity.is_active:
                item.last_seen = datetime.now().strftime("%H:%M:%S")
                item.last_seen_ts = now
                item.last_confidence = entity.confidence
                item.last_position = _compute_position_label(entity.bbox)

            if was_lost:
                # Transition: LOST → RETURNED
                self._trigger_returned(item)

        else:
            # ── Object is genuinely absent ────────────────────────────────────
            item.active = False
            item.seconds_missing += elapsed

            if not item.alert_active and item.seconds_missing >= WATCHLIST_MISSING_THRESHOLD:
                self._trigger_lost(item)

    def _trigger_lost(self, item: WatchItem) -> None:
        """Transition item to LOST state and log event."""
        item.alert_active = True
        item.returned_at = None
        msg = (
            f"'{item.display_label}' left camera view — "
            f"last seen {item.last_seen} at {item.last_position}"
        )
        item._log_event("OBJECT_LOST", msg)
        log.info("[WatchlistManager] ALERT: %s", msg)

    def _trigger_returned(self, item: WatchItem) -> None:
        """Transition item from LOST back to RETURNED/VISIBLE."""
        item.alert_active = False
        item.seconds_missing = 0.0
        item.returned_at = time.monotonic()
        msg = (
            f"'{item.display_label}' returned to camera view — "
            f"now at {item.last_position}"
        )
        item._log_event("OBJECT_RETURNED", msg)
        log.info("[WatchlistManager] RETURNED: %s", msg)

    # ── Query API ─────────────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """
        Return the current watchlist state as a JSON-safe dict.
        Always returns a valid dict even when idle.
        """
        if self._item is None:
            return {"watching": False, "item": None}
        return {"watching": True, "item": self._item.to_dict()}

    def get_available(self, entity_registry) -> List[Dict[str, Any]]:
        """
        Return currently ACTIVE entities suitable for the dropdown.

        Source: EntityRegistry.get_active() — already stabilised entities.
        Filters: conf >= 80%, age >= 2s (same as dashboard).
        """
        active = entity_registry.get_active(min_confidence=0.80, min_age_seconds=2.0)
        result = []
        for e in active:
            result.append({
                "entity_uuid":   e.entity_id,
                "display_label": e.label,
                "category":      e.category,
                "confidence":    round(e.confidence * 100),
                "position":      _compute_position_label(e.bbox),
                "duration_str":  e.duration_str,
            })
        return result

    @property
    def is_watching(self) -> bool:
        return self._item is not None

    @property
    def current_item(self) -> Optional[WatchItem]:
        return self._item
