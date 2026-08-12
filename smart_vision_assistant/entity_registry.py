"""
entity_registry.py
==================
Entity Registry — Persistent Object Identity Layer (v2: Session Memory).

Session Memory Enhancement:
  Objects that were ever detected with confidence >= 80% are NEVER evicted
  during the session. They transition from ACTIVE → INACTIVE when they leave
  the camera view. Only on application shutdown is memory cleared.

  This makes the registry a cumulative session-level memory — a complete
  record of everything observed since the app started.

Object States:
  ACTIVE   — Currently visible in camera
  INACTIVE — Previously detected, no longer visible

Design rules:
  - Confidence gate: only objects that ENTERED at >= 80% conf are stored
  - Objects that were low-conf when they first entered are tracked but never
    promoted to session memory
  - INACTIVE objects retain ALL their data: duration, events, relationships,
    Gemini description — stored once, never reprocessed
  - get_active()   → currently visible, conf >= 0.80, age >= 2s
  - get_inactive() → previously seen, conf >= 0.80 (sorted by last_seen desc)
  - get_session()  → active + inactive combined (full session picture)
"""

import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ── State constants ────────────────────────────────────────────────────────────
STATE_ACTIVE   = "active"
STATE_INACTIVE = "inactive"

# Minimum confidence to enter session memory (objects below this are ignored)
SESSION_MIN_CONFIDENCE: float = 0.80


# ──────────────────────────────────────────────────────────────────────────────
# Entity — persistent object identity
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Entity:
    """
    Stable representation of one detected real-world object.

    entity_id  — UUID, never changes for the lifetime of this session.
    track_id   — Current (or last known) YOLO BoT-SORT tracker ID.
    state      — STATE_ACTIVE or STATE_INACTIVE
    """
    entity_id: str                           # Stable UUID
    track_id: int                            # Current / last known tracker ID
    label: str                               # Best display label (Gemini or YOLO)
    yolo_label: str                          # Raw YOLO class name
    category: str                            # Scene category (Humans, Electronics, …)
    confidence: float                        # Highest recorded confidence score
    bbox: Tuple[int, int, int, int]          # Most recent bounding box (x1,y1,x2,y2)
    first_seen: float = field(default_factory=time.monotonic)
    last_seen: float  = field(default_factory=time.monotonic)
    state: str = STATE_ACTIVE                # STATE_ACTIVE or STATE_INACTIVE
    is_stationary: bool = False
    is_new: bool = True
    gemini_verified: bool = False
    gemini_description: Optional[str] = None

    # OCR / Text Intelligence (added in Phase 1)
    detected_texts: List[str] = field(default_factory=list)
    best_text: str = ""
    brand: str = ""
    product_type: str = ""
    inferred_display_label: str = ""
    ocr_last_run: float = 0.0

    # Cumulative session data — append-only, never cleared
    events: List[str] = field(default_factory=list)
    relationships: List[str] = field(default_factory=list)
    relationship_counts: Dict[str, int] = field(default_factory=dict)

    # Full detection history snapshot (capped at 500 entries per entity)
    detection_history: List[dict] = field(default_factory=list)

    # Accumulated visible time (seconds)
    # Updated when entity transitions ACTIVE → INACTIVE
    total_visible_duration: float = 0.0
    _active_since: float = field(default_factory=time.monotonic)  # when last activated

    # ── Derived Properties ─────────────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        return self.state == STATE_ACTIVE

    @property
    def duration(self) -> float:
        """Total visible time including current active stint if applicable."""
        base = self.total_visible_duration
        if self.state == STATE_ACTIVE:
            # Add current active stint
            base += (time.monotonic() - self._active_since)
        return base

    @property
    def duration_str(self) -> str:
        """Human-readable total visible duration."""
        d = self.duration
        if d < 60:
            return f"{int(d)}s"
        mins = int(d // 60)
        secs = int(d % 60)
        return f"{mins}m {secs:02d}s"

    @property
    def last_seen_ago(self) -> float:
        """Seconds since last seen (0 if currently active)."""
        if self.state == STATE_ACTIVE:
            return 0.0
        return time.monotonic() - self.last_seen

    @property
    def last_seen_ago_str(self) -> str:
        """Human-readable 'last seen X ago' string."""
        if self.state == STATE_ACTIVE:
            return "now"
        ago = self.last_seen_ago
        if ago < 60:
            return f"{int(ago)}s ago"
        mins = int(ago // 60)
        secs = int(ago % 60)
        return f"{mins}m {secs:02d}s ago"

    @property
    def status_label(self) -> str:
        """Display status for the UI."""
        if self.state == STATE_INACTIVE:
            return f"last seen {self.last_seen_ago_str}"
        if self.is_new:
            return "new"
        if self.is_stationary:
            return "stationary"
        return "tracked"

    # ── Serialization ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """
        Serialize to structured JSON-safe dict for the web API.
        Canonical object representation sent to the frontend.
        """
        # Build relationship list with counts
        rel_list = []
        for rel, count in sorted(
            self.relationship_counts.items(), key=lambda x: -x[1]
        ):
            rel_list.append({"description": rel, "count": count})

        return {
            "entity_id": self.entity_id,
            "track_id": self.track_id,
            "label": self.label,
            "yolo_label": self.yolo_label,
            "category": self.category,
            "brand": self.brand,
            "product_type": self.product_type,
            "best_text": self.best_text,
            "detected_texts": self.detected_texts,
            "inferred_display_label": self.inferred_display_label,
            "confidence": round(self.confidence, 4),
            "confidence_pct": f"{self.confidence:.0%}",
            "bbox": list(self.bbox),
            # Duration
            "duration": round(self.duration, 2),
            "duration_str": self.duration_str,
            "total_visible_duration": round(self.total_visible_duration, 2),
            # State
            "state": self.state,
            "is_active": self.is_active,
            "is_new": self.is_new,
            "is_stationary": self.is_stationary,
            # Timing
            "first_seen": round(self.first_seen, 2),
            "last_seen": round(self.last_seen, 2),
            "last_seen_ago": round(self.last_seen_ago, 1),
            "last_seen_ago_str": self.last_seen_ago_str,
            # Status
            "status": self.status_label,
            # Gemini
            "gemini_verified": self.gemini_verified,
            "gemini_description": self.gemini_description,
            # Session data
            "events": list(self.events[-20:]),
            "relationships": list(self.relationships[-20:]),
            "relationship_counts": rel_list,
        }


# ──────────────────────────────────────────────────────────────────────────────
# EntityRegistry
# ──────────────────────────────────────────────────────────────────────────────

class EntityRegistry:
    """
    Session-persistent identity registry for all detected objects.

    Objects that meet the confidence gate (>= 80%) are stored for the ENTIRE
    session duration. They transition ACTIVE ↔ INACTIVE as they appear/disappear
    from the camera. They are NEVER deleted until application shutdown.

    Usage in the headless AI loop:
        registry.sync_from_scene_memory(scene_memory, verification_cache)

    Usage in the API layer:
        active   = registry.get_active(min_age_seconds=2.0)
        inactive = registry.get_inactive()
        session  = registry.get_session()
    """

    def __init__(self) -> None:
        # Session memory: entity_id → Entity (NEVER deleted during session)
        self._session: Dict[str, Entity] = {}
        # Fast lookup: current YOLO track_id → entity_id
        self._track_to_entity: Dict[int, str] = {}
        # Total ever registered
        self._total_registered: int = 0
        # Session start time
        self._session_start: float = time.monotonic()

    # ──────────────────────────────────────────────────────────────────────────
    # Sync from SceneMemory (called every frame)
    # ──────────────────────────────────────────────────────────────────────────

    def sync_from_scene_memory(self, scene_memory, verification_cache: dict) -> None:
        """
        Pull the latest SceneMemory state into session memory.

        ── CRITICAL ORDERING FIX ────────────────────────────────────────────────
        The method must run in THREE phases, not two:

          Phase 1 — Compute which track_ids are present this frame.
          Phase 2 — Deactivate ACTIVE entities whose track_id disappeared  ← FIRST
          Phase 3 — Process new/existing active records                    ← SECOND

        WHY ORDER MATTERS:
          When an object moves fast, BoT-SORT may drop track_id=5 and assign
          track_id=6 to the same object IN THE SAME FRAME. _find_relinkable
          searches for INACTIVE entities by label. If we process new records
          BEFORE deactivating, entity UUID=X (track_id=5) is still STATE_ACTIVE
          when _find_relinkable runs → it finds nothing → creates a NEW entity
          UUID=Y for track_id=6 → UUID=X is permanently stuck INACTIVE → watchlist
          alert fires and never clears.

          With Phase 2 before Phase 3: entity UUID=X is INACTIVE before
          _find_relinkable runs → found and relinked to track_id=6 → UUID=X
          stays active → watchlist sees it as present. Correct.
        ─────────────────────────────────────────────────────────────────────────
        """
        now = time.monotonic()
        active_records = scene_memory.get_active_records()

        # ── Phase 1: Compute active track_id set for this frame ──────────────
        active_track_ids: set = set()
        for rec in active_records:
            if rec.track_id is not None:
                active_track_ids.add(rec.track_id)

        # ── Phase 2: Deactivate entities whose track vanished — BEFORE relinking
        # This must run first so _find_relinkable sees them as INACTIVE.
        for eid, entity in self._session.items():
            if entity.state == STATE_ACTIVE and entity.track_id not in active_track_ids:
                entity.total_visible_duration += (now - entity._active_since)
                entity.state = STATE_INACTIVE
                entity.last_seen = now
                log.debug(
                    "[EntityRegistry] Deactivated entity %s ('%s') — visible: %.1fs",
                    eid[:8], entity.label, entity.total_visible_duration,
                )
                self._track_to_entity.pop(entity.track_id, None)

        # ── Phase 3: Process active records — relink or create ───────────────
        relinked_tids = set()

        for rec in active_records:
            tid = rec.track_id
            if tid is None:
                continue

            # Only admit objects meeting the confidence gate
            if rec.confidence < SESSION_MIN_CONFIDENCE:
                continue

            eid = self._track_to_entity.get(tid)
            if eid and eid in self._session:
                entity = self._session[eid]
            else:
                # Unknown track_id — try to relink to a recently deactivated entity.
                # Because Phase 2 already ran, same-label INACTIVE entities are
                # now correctly available for relinking.
                entity = self._find_relinkable(rec.display_label, rec.category)
                if entity is None:
                    entity = Entity(
                        entity_id=str(uuid.uuid4()),
                        track_id=tid,
                        label=rec.display_label,
                        yolo_label=rec.yolo_label,
                        category=rec.category,
                        confidence=rec.confidence,
                        bbox=rec.bbox,
                        first_seen=rec.first_seen,
                        last_seen=rec.last_seen,
                        is_new=rec.is_new,
                        state=STATE_ACTIVE,
                        _active_since=now,
                    )
                    self._session[entity.entity_id] = entity
                    self._total_registered += 1
                    log.debug(
                        "[EntityRegistry] New session entity %s → track #%d '%s'",
                        entity.entity_id[:8], tid, rec.display_label,
                    )
                else:
                    # Re-link: same object, new track_id — identity preserved
                    log.debug(
                        "[EntityRegistry] Re-linked entity %s ('%s') → track #%d",
                        entity.entity_id[:8], entity.label, tid,
                    )
                    relinked_tids.add(tid)
                    # Shed 'new' flag so UI renders immediately without grace delay
                    rec.is_new = False

                self._track_to_entity[tid] = entity.entity_id

            # ── Reactivate if it was INACTIVE (relink path or race condition) ──
            if entity.state == STATE_INACTIVE:
                entity._active_since = now
                entity.state = STATE_ACTIVE
                log.debug(
                    "[EntityRegistry] Reactivated entity %s ('%s')",
                    entity.entity_id[:8], entity.label,
                )

            # ── Update mutable fields from latest detection ───────────────────
            entity.track_id = tid
            entity.label = rec.display_label
            entity.yolo_label = rec.yolo_label
            entity.category = rec.category
            if rec.confidence > entity.confidence:
                entity.confidence = rec.confidence
            entity.bbox = rec.bbox
            entity.last_seen = rec.last_seen
            entity.is_new = rec.is_new
            entity.is_stationary = getattr(rec, 'is_stationary', False)
            entity.gemini_verified = rec.gemini_verified
            if rec.gemini_description and not entity.gemini_description:
                entity.gemini_description = rec.gemini_description

            # Detection history snapshot (capped at 500)
            entity.detection_history.append({
                "t":    round(now, 3),
                "conf": round(rec.confidence, 4),
                "bbox": list(rec.bbox),
            })
            if len(entity.detection_history) > 500:
                entity.detection_history = entity.detection_history[-500:]

        # ── Mark inactive: entities currently ACTIVE but not in SceneMemory ──
        for eid, entity in self._session.items():
            if entity.state == STATE_ACTIVE and entity.track_id not in active_track_ids:
                # Freeze visible duration
                entity.total_visible_duration += (now - entity._active_since)
                entity.state = STATE_INACTIVE
                entity.last_seen = now
                log.debug("[EntityRegistry] Deactivated entity %s ('%s') — total visible: %.1fs",
                          eid[:8], entity.label, entity.total_visible_duration)
                # Remove stale track mapping
                self._track_to_entity.pop(entity.track_id, None)
                
        return relinked_tids

    def _find_relinkable(self, label: str, category: str) -> Optional[Entity]:
        """
        Try to find an INACTIVE entity with the same label that disappeared
        recently to re-link rather than creating a duplicate.
        This handles the case where a person leaves and returns.
        The timeout is intentionally massive (12 hours) to ensure identity persistence.
        """
        now = time.monotonic()
        candidates = [
            e for e in self._session.values()
            if e.state == STATE_INACTIVE
            and e.label.lower() == label.lower()
            and e.category == category
            and (now - e.last_seen) < 43200.0
        ]
        if not candidates:
            return None
        # Pick the most recently seen candidate
        return max(candidates, key=lambda e: e.last_seen)

    # ──────────────────────────────────────────────────────────────────────────
    # Event & Relationship Updates
    # ──────────────────────────────────────────────────────────────────────────

    def update_events(self, events: list) -> None:
        """
        Merge event engine output into entity event lists.
        events: list of (track_id, event_type, description)
        Stored permanently — never cleared for INACTIVE entities.
        """
        for tid, event_type, desc in events:
            eid = self._track_to_entity.get(tid)
            if eid and eid in self._session:
                entry = f"{event_type}: {desc}"
                if entry not in self._session[eid].events:
                    self._session[eid].events.append(entry)

    def update_relationships(self, rel_events: list) -> None:
        """
        Merge relationship engine output into entity relationship lists.
        rel_events: list of (track_id_a, track_id_b, event_type, description)
        Tracks observation count per relationship string.
        """
        for tid_a, tid_b, event_type, desc in rel_events:
            for tid in (tid_a, tid_b):
                eid = self._track_to_entity.get(tid)
                if eid and eid in self._session:
                    entity = self._session[eid]
                    if desc not in entity.relationships:
                        entity.relationships.append(desc)
                    # Increment observation count
                    entity.relationship_counts[desc] = (
                        entity.relationship_counts.get(desc, 0) + 1
                    )

    # ──────────────────────────────────────────────────────────────────────────
    # OCR Updates
    # ──────────────────────────────────────────────────────────────────────────

    def update_ocr_result(
        self,
        entity_id: str,
        detected_texts: List[str],
        best_text: str,
        brand: str,
        product_type: str,
        inferred_label: str
    ) -> None:
        """
        Merge async OCR results into the session entity.
        Called by the OCRProcessor background thread.
        """
        entity = self._session.get(entity_id)
        if not entity:
            return

        # We append texts that we haven't seen in this entity's history yet
        for txt in detected_texts:
            if txt not in entity.detected_texts:
                entity.detected_texts.append(txt)

        # We only update if the new best text is longer or same length (avoids flickering to shorter string)
        if len(best_text) >= len(entity.best_text):
            entity.best_text = best_text
            
        if brand and not entity.brand:
            entity.brand = brand
            
        if product_type and not entity.product_type:
            entity.product_type = product_type
            
        # If infer_entity_identity changed the label, store it
        if inferred_label != entity.yolo_label:
            entity.inferred_display_label = inferred_label
        else:
            entity.inferred_display_label = entity.label # fallback to what it was


    # ──────────────────────────────────────────────────────────────────────────
    # Query API
    # ──────────────────────────────────────────────────────────────────────────

    def get_active(
        self,
        min_confidence: float = SESSION_MIN_CONFIDENCE,
        min_age_seconds: float = 2.0,
    ) -> List[Entity]:
        """
        Return currently ACTIVE entities meeting both filters.
        Sorted by duration descending (most stable first).
        """
        results = [
            e for e in self._session.values()
            if e.state == STATE_ACTIVE
            and e.confidence >= min_confidence
            and e.duration >= min_age_seconds
        ]
        results.sort(key=lambda e: e.duration, reverse=True)
        return results

    def get_inactive(
        self,
        min_confidence: float = SESSION_MIN_CONFIDENCE,
    ) -> List[Entity]:
        """
        Return all INACTIVE session entities sorted by last_seen descending
        (most recently departed first).
        """
        results = [
            e for e in self._session.values()
            if e.state == STATE_INACTIVE
            and e.confidence >= min_confidence
        ]
        results.sort(key=lambda e: e.last_seen, reverse=True)
        return results

    def get_session(
        self,
        min_confidence: float = SESSION_MIN_CONFIDENCE,
        min_age_seconds: float = 2.0,
    ) -> dict:
        """
        Return the full session picture: active + inactive objects.
        This is what the web UI uses to render the complete session report.
        """
        active = self.get_active(min_confidence, min_age_seconds)
        inactive = self.get_inactive(min_confidence)
        return {
            "active": [e.to_dict() for e in active],
            "inactive": [e.to_dict() for e in inactive],
            "active_count": len(active),
            "inactive_count": len(inactive),
            "total_count": len(active) + len(inactive),
        }

    def get_entity_by_track(self, track_id: int) -> Optional[Entity]:
        eid = self._track_to_entity.get(track_id)
        return self._session.get(eid) if eid else None

    def active_count_filtered(
        self,
        min_confidence: float = SESSION_MIN_CONFIDENCE,
        min_age_seconds: float = 2.0,
    ) -> int:
        return len(self.get_active(min_confidence, min_age_seconds))

    @property
    def total_registered(self) -> int:
        return self._total_registered

    @property
    def session_object_count(self) -> int:
        """Total objects ever admitted to session memory."""
        return len(self._session)

    # ──────────────────────────────────────────────────────────────────────────
    # Structured Report Builder
    # ──────────────────────────────────────────────────────────────────────────

    def build_report(
        self,
        fps: float = 0.0,
        cpu: float = 0.0,
        ram_mb: float = 0.0,
        session_time: float = 0.0,
        scene_stability: float = 0.0,
        min_confidence: float = SESSION_MIN_CONFIDENCE,
        min_age_seconds: float = 2.0,
    ) -> dict:
        """
        Build a fully structured JSON session intelligence report.

        Includes BOTH active and inactive objects — the complete session picture.
        """
        from datetime import datetime

        active   = self.get_active(min_confidence, min_age_seconds)
        inactive = self.get_inactive(min_confidence)
        all_objects = active + inactive

        # Category breakdown for active objects only
        active_by_category: Dict[str, int] = {}
        for e in active:
            active_by_category[e.category] = active_by_category.get(e.category, 0) + 1

        # Category breakdown for all session objects
        session_by_category: Dict[str, int] = {}
        for e in all_objects:
            session_by_category[e.category] = session_by_category.get(e.category, 0) + 1

        # Scene summary (active only for current-state sentence)
        summary_sentences = []
        for cat, count in sorted(active_by_category.items(), key=lambda x: -x[1]):
            noun = cat.rstrip("s") if count == 1 else cat
            summary_sentences.append(f"{count} {noun} detected.")

        if not summary_sentences and inactive:
            summary_sentences.append("No objects currently in view.")

        # All session events (active + inactive)
        all_events = []
        seen_ev: set = set()
        for e in all_objects:
            for ev in e.events[-5:]:
                key = f"{e.entity_id}:{ev}"
                if key not in seen_ev:
                    all_events.append({
                        "entity_id": e.entity_id,
                        "label": e.label,
                        "event": ev,
                        "state": e.state,
                    })
                    seen_ev.add(key)

        # All session relationships with counts
        all_relationships = []
        seen_rels: set = set()
        for e in all_objects:
            for rel, count in e.relationship_counts.items():
                key = f"{e.label}:{rel}"
                if key not in seen_rels:
                    all_relationships.append({
                        "label": e.label,
                        "relationship": rel,
                        "count": count,
                        "entity_id": e.entity_id,
                        "state": e.state,
                    })
                    seen_rels.add(key)
        # Sort by observation count descending
        all_relationships.sort(key=lambda r: -r["count"])

        return {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "session_time": round(session_time, 1),
            "session_time_str": _format_duration(session_time),
            "fps": round(fps, 1),
            "cpu": round(cpu, 1),
            "ram_mb": round(ram_mb, 1),
            "scene_stability": round(scene_stability, 1),
            # Active now
            "active_objects": len(active),
            "active_by_category": active_by_category,
            # Full session
            "session_objects": len(all_objects),
            "session_by_category": session_by_category,
            "total_registered": self._total_registered,
            # Summary text
            "summary": " ".join(summary_sentences) if summary_sentences else "Initialising…",
            # Object lists
            "objects": [e.to_dict() for e in active],          # active only (for ObjectCards live section)
            "inactive_objects": [e.to_dict() for e in inactive], # session history
            # Session-wide events and relationships
            "events": all_events,
            "relationships": all_relationships,
            "filters": {
                "min_confidence": min_confidence,
                "min_age_seconds": min_age_seconds,
            },
        }


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    if mins < 60:
        return f"{mins}m {secs:02d}s"
    hours = mins // 60
    mins = mins % 60
    return f"{hours}h {mins:02d}m"
