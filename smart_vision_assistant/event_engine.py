"""
event_engine.py
===============
Rules engine for determining high-level events (Stationary, Moved, Abandoned).
Hardened for stability using adaptive thresholds and state debouncing.
"""

import logging
from typing import List, Tuple, Any
from spatial_engine import is_near_adaptive
import math

log = logging.getLogger(__name__)

class EventEngine:
    def __init__(self, stationary_time: float = 30.0, abandoned_time: float = 30.0):
        self.stationary_time = stationary_time
        self.abandoned_time = abandoned_time
        self.excluded_categories = {'Furniture', 'Household'}

    def process(self, records: List[Any], now: float) -> List[Tuple[int, str, str]]:
        """
        Process active object records and return a list of events.
        Events are tuples: (track_id, event_type, description)
        """
        events = []
        
        # Pass 1: Update positions and determine Stationary/Moved
        for rec in records:
            x1, y1, x2, y2 = rec.bbox
            current_centroid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            
            # Adaptive drift tolerance based on object width/height
            width = max(1, x2 - x1)
            height = max(1, y2 - y1)
            drift_tolerance = max(20.0, max(width, height) * 0.10)
            
            if getattr(rec, 'anchor_centroid', None) is None:
                rec.anchor_centroid = current_centroid
                rec.anchor_time = now
                rec.is_stationary = False
                rec.was_near_person = False
                rec.unattended_since = 0.0
                rec.last_event = 'new'
            else:
                dist = math.hypot(
                    rec.anchor_centroid[0] - current_centroid[0],
                    rec.anchor_centroid[1] - current_centroid[1]
                )
                if dist > drift_tolerance:
                    # Object has moved significantly from anchor
                    rec.anchor_centroid = current_centroid
                    rec.anchor_time = now
                    if rec.is_stationary:
                        rec.is_stationary = False
                        rec.unattended_since = 0.0 # Reset abandoned state
                        if rec.last_event != 'moved':
                            events.append((rec.track_id, 'moved', f"Moved {dist:.0f}px"))
                            rec.last_event = 'moved'
                else:
                    # Object is staying near anchor
                    if not rec.is_stationary and (now - rec.anchor_time) >= self.stationary_time:
                        rec.is_stationary = True
                        if rec.last_event != 'stationary':
                            events.append((rec.track_id, 'stationary', f"Stationary for >{self.stationary_time}s"))
                            rec.last_event = 'stationary'
            
            rec.last_centroid = current_centroid

        # Pass 2: Determine Abandoned Objects (ATTENDED -> UNATTENDED -> ABANDONED)
        humans = [r for r in records if r.category == 'Humans']
        
        for rec in records:
            if rec.category == 'Humans' or rec.category in self.excluded_categories:
                continue
                
            # Check if attended
            is_near_any_human = False
            for h in humans:
                if is_near_adaptive(rec.bbox, h.bbox):
                    is_near_any_human = True
                    break
                    
            if is_near_any_human:
                rec.was_near_person = True # It has been attended at least once
                rec.unattended_since = 0.0 # Reset timer (Debounce)
                
            # Abandoned Logic: Must be stationary, must have been attended previously, and currently unattended
            if rec.is_stationary and rec.was_near_person and not is_near_any_human:
                if rec.unattended_since == 0.0:
                    rec.unattended_since = now
                    
                if (now - rec.unattended_since) >= self.abandoned_time:
                    if rec.last_event != 'abandoned':
                        events.append((rec.track_id, 'abandoned', "Abandoned object (Person left scene)"))
                        rec.last_event = 'abandoned'
            else:
                # If it starts moving or a person comes back, pause/reset the abandoned timer
                pass

        return events
