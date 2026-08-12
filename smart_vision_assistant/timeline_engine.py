"""
timeline_engine.py
==================
Generates chronological event histories and object lifecycles.
"""

import sqlite3
import logging
from typing import List, Dict, Any

log = logging.getLogger("timeline_engine")

class TimelineEngine:
    def __init__(self, db_path: str = "data/smart_vision.db"):
        self.db_path = db_path

    def get_session_timeline(self, session_id: int, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Returns a chronological list of events for a given session.
        """
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        
        try:
            # Join with tracked_objects to get the best label
            query = """
                SELECT 
                    e.event_at, e.event_type, e.track_id, e.category, e.extra_data,
                    COALESCE(NULLIF(t.inferred_display_label, ''), t.display_label, e.label) as label,
                    t.brand
                FROM object_events e
                LEFT JOIN tracked_objects t ON e.track_id = t.track_id AND e.session_id = t.session_id
                WHERE e.session_id = ?
                ORDER BY e.event_at DESC
                LIMIT ?
            """
            cursor = conn.execute(query, (session_id, limit))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "time": row["event_at"],
                    "type": row["event_type"],
                    "track_id": row["track_id"],
                    "label": row["label"],
                    "category": row["category"],
                    "brand": row["brand"],
                    "extra": row["extra_data"]
                })
            return results
            
        except Exception as e:
            log.error("Timeline query failed: %s", e)
            return []
        finally:
            conn.close()

    def get_all_timeline(self, limit: int = 500) -> List[Dict[str, Any]]:
        """
        Returns a chronological list of events across ALL sessions.
        """
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        
        try:
            query = """
                SELECT 
                    e.event_at, e.event_type, e.track_id, e.category, e.extra_data,
                    COALESCE(NULLIF(t.inferred_display_label, ''), t.display_label, e.label) as label,
                    t.brand
                FROM object_events e
                LEFT JOIN tracked_objects t ON e.track_id = t.track_id AND e.session_id = t.session_id
                ORDER BY e.event_at DESC
                LIMIT ?
            """
            cursor = conn.execute(query, (limit,))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "time": row["event_at"],
                    "type": row["event_type"],
                    "track_id": row["track_id"],
                    "label": row["label"],
                    "category": row["category"],
                    "brand": row["brand"],
                    "extra": row["extra_data"]
                })
            return results
            
        except Exception as e:
            log.error("All timeline query failed: %s", e)
            return []
        finally:
            conn.close()
