"""
relationship_engine.py
======================
State-based Relationship Engine. Evaluates purely geometric relationships
between stable objects and tracks state transitions to emit events.

Performance: throttled to RELATIONSHIP_EVAL_HZ (default 5 Hz) to avoid
paying O(N²) spatial check cost on every 30 FPS frame.
"""

import time
from typing import List, Tuple, Set, Dict, Any, Optional
import spatial_engine

# Max frequency at which spatial pair-checks are evaluated (Hz)
_RELATIONSHIP_EVAL_HZ: float = 5.0
_RELATIONSHIP_EVAL_INTERVAL: float = 1.0 / _RELATIONSHIP_EVAL_HZ

class RelationshipEngine:
    def __init__(self):
        # Maps (track_id_A, track_id_B) -> set of active relationships (e.g., {'near', 'A_inside_B'})
        self._active_relationships: Dict[Tuple[int, int], Set[str]] = {}
        # Throttle: track last time we ran the full O(N²) evaluation
        self._last_eval: float = 0.0

    def _is_inside_valid(self, cat_inner: str, cat_outer: str) -> bool:
        if cat_inner == 'Humans' and cat_outer in ('Electronics', 'Furniture'):
            return False
        if cat_inner == 'Furniture' and cat_outer == 'Humans':
            return False
        return True

    def _is_on_top_of_valid(self, cat_top: str, cat_bottom: str) -> bool:
        if cat_top == 'Humans' and cat_bottom == 'Electronics':
            return False
        return True

    def process(self, records: List[Any]) -> List[Tuple[int, int, str, str]]:
        """
        Evaluate relationships among stable objects.
        Returns a list of transition events: (track_id_A, track_id_B, event_type, description)
        where event_type is 'RELATIONSHIP_BEGIN' or 'RELATIONSHIP_END'.
        Throttled to _RELATIONSHIP_EVAL_HZ to avoid O(N²) cost every frame.
        """
        # Throttle: skip evaluation if called too soon
        now = time.monotonic()
        if now - self._last_eval < _RELATIONSHIP_EVAL_INTERVAL:
            return []   # Return empty — no change possible in <200ms anyway
        self._last_eval = now

        # 1. Stability Gating: Filter for stable objects
        stable_records = [r for r in records if not r.is_new]

        # Build O(1) lookup dict: track_id -> record
        record_map: Dict[int, Any] = {r.track_id: r for r in stable_records}

        # 2. Compute current frame relationships
        current_relationships: Dict[Tuple[int, int], Set[str]] = {}
        
        n = len(stable_records)
        for i in range(n):
            for j in range(i + 1, n):
                r1 = stable_records[i]
                r2 = stable_records[j]
                
                # Order by track_id for consistent dictionary keys
                rec_a, rec_b = (r1, r2) if r1.track_id < r2.track_id else (r2, r1)
                pair = (rec_a.track_id, rec_b.track_id)
                
                rels = set()
                
                if spatial_engine.is_near_adaptive(rec_a.bbox, rec_b.bbox):
                    rels.add('near')
                    
                if spatial_engine.is_overlapping(rec_a.bbox, rec_b.bbox):
                    rels.add('overlapping')
                    
                if spatial_engine.is_inside(rec_a.bbox, rec_b.bbox) and self._is_inside_valid(rec_a.category, rec_b.category):
                    rels.add('A_inside_B')
                elif spatial_engine.is_inside(rec_b.bbox, rec_a.bbox) and self._is_inside_valid(rec_b.category, rec_a.category):
                    rels.add('B_inside_A')
                    
                if spatial_engine.is_on_top_of(rec_a.bbox, rec_b.bbox) and self._is_on_top_of_valid(rec_a.category, rec_b.category):
                    rels.add('A_on_top_of_B')
                elif spatial_engine.is_on_top_of(rec_b.bbox, rec_a.bbox) and self._is_on_top_of_valid(rec_b.category, rec_a.category):
                    rels.add('B_on_top_of_A')
                    
                if rels:
                    current_relationships[pair] = rels
                    
        # 3. Detect transitions (State-Based Tracking)
        events = []
        
        # Check for BEGIN events
        for pair, current_rels in current_relationships.items():
            previous_rels = self._active_relationships.get(pair, set())
            new_rels = current_rels - previous_rels
            for rel in new_rels:
                # O(1) dict lookup instead of O(N) linear search
                rec_a = record_map.get(pair[0])
                rec_b = record_map.get(pair[1])
                if rec_a and rec_b:
                    desc = self._format_desc(rec_a, rec_b, rel, "began")
                    events.append((pair[0], pair[1], "RELATIONSHIP_BEGIN", desc))

        # Check for END events
        for pair, previous_rels in self._active_relationships.items():
            current_rels = current_relationships.get(pair, set())
            ended_rels = previous_rels - current_rels
            for rel in ended_rels:
                # O(1) dict lookup; objects may have left the scene
                rec_a = record_map.get(pair[0])
                rec_b = record_map.get(pair[1])

                label_a = rec_a.display_label if rec_a else f"Object #{pair[0]}"
                label_b = rec_b.display_label if rec_b else f"Object #{pair[1]}"

                desc = self._format_desc_labels(label_a, label_b, rel, "ended")
                events.append((pair[0], pair[1], "RELATIONSHIP_END", desc))
                
        # 4. Update state
        self._active_relationships = current_relationships
        
        return events

    def _format_desc(self, rec_a: Any, rec_b: Any, rel: str, action: str) -> str:
        # Check if they have inferred display labels or regular labels
        label_a = getattr(rec_a, 'inferred_display_label', getattr(rec_a, 'display_label', f"#{rec_a.track_id}"))
        label_b = getattr(rec_b, 'inferred_display_label', getattr(rec_b, 'display_label', f"#{rec_b.track_id}"))
        return self._format_desc_labels(label_a, label_b, rel, action)

    def _format_desc_labels(self, label_a: str, label_b: str, rel: str, action: str) -> str:
        if rel == 'near':
            return f"{label_a} and {label_b} {action} being near each other"
        elif rel == 'overlapping':
            return f"{label_a} and {label_b} {action} overlapping"
        elif rel == 'A_inside_B':
            return f"{label_a} {action} being inside {label_b}"
        elif rel == 'B_inside_A':
            return f"{label_b} {action} being inside {label_a}"
        elif rel == 'A_on_top_of_B':
            return f"{label_a} {action} being on top of {label_b}"
        elif rel == 'B_on_top_of_A':
            return f"{label_b} {action} being on top of {label_a}"
        return f"Unknown relationship {rel} {action}"
