"""
search_engine.py
================
Visual Memory Search Engine.
Queries the SQLite database for historical observations based on structured filters.
"""

import sqlite3
import logging
from typing import List, Dict, Any

log = logging.getLogger("search_engine")

class SearchEngine:
    def __init__(self, db_path: str = "data/smart_vision.db"):
        self.db_path = db_path

    def search(self, session_id: int, filters: dict) -> List[Dict[str, Any]]:
        """
        Executes a search against tracked_objects for a given session.
        Filters can include: brand, product_type, category, text, event.
        """
        # Connect in read-only mode for concurrency
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        
        try:
            query = """
                SELECT 
                    t.track_id, t.display_label, t.inferred_display_label, 
                    t.category, t.brand, t.product_type, t.best_text, 
                    t.first_seen, t.last_seen, t.total_duration_sec,
                    t.highest_confidence
                FROM tracked_objects t
                WHERE t.session_id = ?
            """
            params = [session_id]
            
            # Apply filters
            if "brand" in filters:
                query += " AND t.brand LIKE ?"
                params.append(f"%{filters['brand']}%")
                
            if "product_type" in filters:
                query += " AND t.product_type LIKE ?"
                params.append(f"%{filters['product_type']}%")
                
            if "category" in filters:
                query += " AND t.category = ?"
                params.append(filters["category"])
                
            if "text" in filters:
                query += " AND t.best_text LIKE ?"
                params.append(f"%{filters['text']}%")
                
            if "event" in filters:
                query += """ AND t.track_id IN (
                    SELECT track_id FROM object_events 
                    WHERE session_id = ? AND event_type = ?
                )"""
                params.extend([session_id, filters["event"]])
                
            query += " ORDER BY t.first_seen DESC LIMIT 100"
            
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                track_id = row["track_id"]
                # Fetch recent events for this track
                ev_cursor = conn.execute(
                    "SELECT event_type, event_at FROM object_events WHERE session_id=? AND track_id=? ORDER BY id DESC LIMIT 5",
                    (session_id, track_id)
                )
                events = [{"type": e["event_type"], "at": e["event_at"]} for e in ev_cursor.fetchall()]
                
                results.append({
                    "track_id": track_id,
                    "inferred_label": row["inferred_display_label"] or row["display_label"],
                    "category": row["category"],
                    "brand": row["brand"],
                    "product_type": row["product_type"],
                    "ocr_text": row["best_text"],
                    "first_seen": row["first_seen"],
                    "last_seen": row["last_seen"],
                    "duration": row["total_duration_sec"],
                    "confidence": row["highest_confidence"],
                    "events": events
                })
                
            return results
            
        except Exception as e:
            log.error("Search failed: %s", e)
            return []
        finally:
            conn.close()
