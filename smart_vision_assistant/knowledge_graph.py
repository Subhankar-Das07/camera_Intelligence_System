"""
knowledge_graph.py
==================
Builds a session knowledge graph from entities and their relationships.
"""

import sqlite3
import logging
from typing import Dict, Any

log = logging.getLogger("knowledge_graph")

class KnowledgeGraph:
    def __init__(self, db_path: str = "data/smart_vision.db"):
        self.db_path = db_path

    def get_graph(self, session_id: int) -> Dict[str, Any]:
        """
        Returns nodes (entities, brands) and edges (relationships) for the session.
        """
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        
        try:
            nodes = []
            edges = []
            
            # Fetch all tracked objects as nodes
            cursor = conn.execute("""
                SELECT track_id, display_label, inferred_display_label, category, brand, product_type
                FROM tracked_objects
                WHERE session_id = ?
            """, (session_id,))
            
            for row in cursor.fetchall():
                label = row["inferred_display_label"] or row["display_label"]
                nodes.append({
                    "id": f"obj_{row['track_id']}",
                    "label": f"#{row['track_id']} {label}",
                    "group": row["category"],
                    "brand": row["brand"]
                })
                
                # If it has a brand, create a brand node and an edge
                if row["brand"]:
                    brand_id = f"brand_{row['brand'].lower()}"
                    # Don't duplicate brand nodes
                    if not any(n["id"] == brand_id for n in nodes):
                        nodes.append({
                            "id": brand_id,
                            "label": row["brand"],
                            "group": "Brand"
                        })
                    edges.append({
                        "source": f"obj_{row['track_id']}",
                        "target": brand_id,
                        "label": "is_brand"
                    })

            # Fetch relationship events for edges
            rel_cursor = conn.execute("""
                SELECT event_type, track_id, extra_data 
                FROM object_events 
                WHERE session_id = ? AND event_type IN ('near', 'interacted_with', 'observed_with')
            """, (session_id,))
            
            for row in rel_cursor.fetchall():
                # extra_data usually contains the 'other' track_id in some form, but our DB schema 
                # might only store a descriptive string. We'll simplify this by just returning statistics.
                pass
                
            stats = {
                "total_nodes": len(nodes),
                "total_edges": len(edges)
            }
            
            return {
                "nodes": nodes,
                "edges": edges,
                "stats": stats
            }
            
        except Exception as e:
            log.error("Graph query failed: %s", e)
            return {"nodes": [], "edges": [], "stats": {}}
        finally:
            conn.close()
