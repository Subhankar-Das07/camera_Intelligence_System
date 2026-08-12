"""
database_manager.py
===================
Intelligent Visual Memory Engine — SQLite persistence layer.

Architecture: producer/consumer with a dedicated background writer thread.
  - Main thread submits write tasks to a queue (never blocks detection loop)
  - Background writer thread owns the single SQLite connection
  - Synchronous operations (needing return IDs) use threading.Event handshake
  - All other operations are fire-and-forget async writes

Schema (5 tables):
  sessions        — one row per application run
  reports         — one row per scene report generated
  report_objects  — one row per detected object within each report
  object_events   — full lifecycle event log (new/removed/verified/description_updated)
  tracked_objects — complete lifetime record for every unique tracked object

DB file: data/smart_vision.db
Future migration: swap sqlite3 for psycopg2 and update DB_PATH → connection string.
"""

import json
import logging
import queue
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

DB_PATH = Path("data") / "smart_vision.db"

# ──────────────────────────────────────────────────────────────────────────────
# Operation type constants (internal queue protocol)
# ──────────────────────────────────────────────────────────────────────────────
_OP_INIT             = "init"
_OP_START_SESSION    = "start_session"
_OP_END_SESSION      = "end_session"
_OP_SAVE_REPORT      = "save_report"
_OP_SAVE_OBJ_ROWS    = "save_report_objects"
_OP_OBJECT_NEW       = "object_new"
_OP_OBJECT_REMOVED   = "object_removed"
_OP_OBJECT_VERIFIED  = "object_verified"
_OP_OBJECT_DESCRIBED = "object_described"
_OP_CUSTOM_EVENT     = "custom_event"
_OP_OCR_RESULT       = "ocr_result"
_OP_STOP             = "stop"

# ──────────────────────────────────────────────────────────────────────────────
# Schema DDL
# ──────────────────────────────────────────────────────────────────────────────

_SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ────────────────────────────────────────────────────────────────────────────
-- sessions — one row per application run
-- ────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sessions (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at           TEXT    NOT NULL,   -- ISO-8601 local datetime
    ended_at             TEXT,               -- NULL until clean shutdown
    total_reports        INTEGER DEFAULT 0,
    total_unique_objects INTEGER DEFAULT 0,  -- distinct track IDs seen
    total_events         INTEGER DEFAULT 0   -- total object lifecycle events
);

-- ────────────────────────────────────────────────────────────────────────────
-- reports — one row per scene report printed to console
-- ────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS reports (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    reported_at     TEXT    NOT NULL,
    report_number   INTEGER NOT NULL,
    total_objects   INTEGER NOT NULL DEFAULT 0,
    new_objects     INTEGER NOT NULL DEFAULT 0,
    removed_objects INTEGER NOT NULL DEFAULT 0,
    scene_stability REAL    DEFAULT 0,
    fps             REAL,
    cpu_percent     REAL,
    ram_mb          REAL
);

-- ────────────────────────────────────────────────────────────────────────────
-- report_objects — every OBJECT N entry inside a scene report
-- Answers: "Which objects were in report #5?" / "All reports containing a laptop?"
-- ────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS report_objects (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id           INTEGER NOT NULL REFERENCES reports(id)  ON DELETE CASCADE,
    session_id          INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    track_id            INTEGER NOT NULL,
    object_number       INTEGER NOT NULL,   -- 1-based index in the report
    label               TEXT    NOT NULL,   -- best label (Gemini-refined if available)
    yolo_label          TEXT    NOT NULL,   -- raw YOLO class name
    category            TEXT    NOT NULL,   -- Humans / Electronics / etc.
    confidence          REAL    NOT NULL,
    duration_seconds    REAL    NOT NULL,   -- seconds in scene at report time
    is_new              INTEGER NOT NULL DEFAULT 0,
    is_gemini_verified  INTEGER NOT NULL DEFAULT 0,
    gemini_description  TEXT,              -- bullet-point analysis text
    bbox_x1             INTEGER,
    bbox_y1             INTEGER,
    bbox_x2             INTEGER,
    bbox_y2             INTEGER
);

-- ────────────────────────────────────────────────────────────────────────────
-- object_events — full lifecycle event log
-- Event types: new | removed | verified | description_updated
-- Answers: "How many people entered today?" / "When was track #12 verified?"
-- ────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS object_events (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    event_at         TEXT    NOT NULL,
    event_type       TEXT    NOT NULL,
    track_id         INTEGER NOT NULL,
    label            TEXT    NOT NULL,
    category         TEXT    NOT NULL,
    confidence       REAL,
    duration_seconds REAL,              -- seconds in scene (meaningful for 'removed')
    extra_data       TEXT               -- JSON blob for event-specific extras
);

-- ────────────────────────────────────────────────────────────────────────────
-- tracked_objects — complete lifetime record per unique tracked object per session
-- This is the long-term memory core.
-- Answers: "What did Gemini say about each object?"
--          "Which objects appeared most frequently?"
--          "How long was each object visible?"
-- ────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS tracked_objects (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id           INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    track_id             INTEGER NOT NULL,
    yolo_label           TEXT    NOT NULL,
    display_label        TEXT    NOT NULL,
    category             TEXT    NOT NULL,
    first_seen           TEXT    NOT NULL,   -- ISO-8601 datetime
    last_seen            TEXT    NOT NULL,   -- updated on removal
    total_duration_sec   REAL    DEFAULT 0,
    highest_confidence   REAL    DEFAULT 0,
    latest_confidence    REAL    DEFAULT 0,
    confidence_sum       REAL    DEFAULT 0,  -- for computing running average
    confidence_count     INTEGER DEFAULT 0,  -- for computing running average
    gemini_description   TEXT,              -- latest bullet-point description
    is_gemini_verified   INTEGER DEFAULT 0,
    report_appearances   INTEGER DEFAULT 0, -- how many reports contained this object
    status               TEXT    DEFAULT 'active' CHECK(status IN ('active','removed')),
    UNIQUE(session_id, track_id)
);

-- ────────────────────────────────────────────────────────────────────────────
-- Indexes — optimised for the most common queries
-- ────────────────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_reports_session         ON reports(session_id);
CREATE INDEX IF NOT EXISTS idx_reports_at              ON reports(reported_at);
CREATE INDEX IF NOT EXISTS idx_ro_report               ON report_objects(report_id);
CREATE INDEX IF NOT EXISTS idx_ro_session_track        ON report_objects(session_id, track_id);
CREATE INDEX IF NOT EXISTS idx_ro_label                ON report_objects(label);
CREATE INDEX IF NOT EXISTS idx_ro_category             ON report_objects(category);
CREATE INDEX IF NOT EXISTS idx_events_session          ON object_events(session_id);
CREATE INDEX IF NOT EXISTS idx_events_type             ON object_events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_track            ON object_events(track_id);
CREATE INDEX IF NOT EXISTS idx_events_label            ON object_events(label);
CREATE INDEX IF NOT EXISTS idx_events_at               ON object_events(event_at);
CREATE INDEX IF NOT EXISTS idx_tracked_session_track   ON tracked_objects(session_id, track_id);
CREATE INDEX IF NOT EXISTS idx_tracked_label           ON tracked_objects(display_label);
CREATE INDEX IF NOT EXISTS idx_tracked_category        ON tracked_objects(category);
CREATE INDEX IF NOT EXISTS idx_tracked_status          ON tracked_objects(status);

-- ────────────────────────────────────────────────────────────────────────────
-- ocr_events — OCR specific event log for future search functionality
-- ────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ocr_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    entity_id   TEXT    NOT NULL,
    track_id    INTEGER NOT NULL,
    event_at    TEXT    NOT NULL,
    raw_texts   TEXT    NOT NULL,    -- JSON string array
    best_text   TEXT    NOT NULL,
    brand       TEXT    DEFAULT '',
    inferred_label TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_ocr_entity   ON ocr_events(entity_id);
CREATE INDEX IF NOT EXISTS idx_ocr_brand    ON ocr_events(brand);
CREATE INDEX IF NOT EXISTS idx_ocr_session  ON ocr_events(session_id);

"""


# ──────────────────────────────────────────────────────────────────────────────
# DatabaseManager
# ──────────────────────────────────────────────────────────────────────────────

class DatabaseManager:
    """
    Background-threaded SQLite persistence layer.

    Thread model:
      - All SQLite I/O runs in a single background daemon thread.
      - The main (detection) thread submits work via _submit_sync() or _submit_async().
      - _submit_sync() blocks the caller until the write completes and returns
        the row ID; use only for session/report creation.
      - _submit_async() is fire-and-forget; detection loop never stalls.

    PostgreSQL migration path:
      - Replace sqlite3.connect() with psycopg2.connect().
      - Change ? placeholders to %s.
      - Replace AUTOINCREMENT with SERIAL.
      - All business logic stays identical.
    """

    def __init__(self) -> None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._write_queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(
            target=self._writer_loop, name="DBWriter", daemon=True
        )
        self._thread.start()
        # Initialise schema synchronously before any other calls
        self._submit_sync(_OP_INIT, None)
        log.info("[DB] DatabaseManager ready — %s", DB_PATH.resolve())

    # ──────────────────────────────────────────────────────────────────────────
    # Background Writer Thread
    # ──────────────────────────────────────────────────────────────────────────

    def _writer_loop(self) -> None:
        """Single background thread — owns the SQLite connection exclusively."""
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA synchronous = NORMAL")  # Fast yet safe with WAL

        while True:
            try:
                item = self._write_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            op, args, evt, result_box = item

            if op == _OP_STOP:
                try:
                    conn.commit()
                    conn.close()
                except Exception:
                    pass
                if evt:
                    evt.set()
                break

            try:
                result = self._dispatch(conn, op, args)
                conn.commit()
                if result_box is not None:
                    result_box[0] = result
            except Exception as exc:
                log.error("[DB] Write error (op=%s): %s", op, exc, exc_info=False)
                if result_box is not None:
                    result_box[1] = exc
            finally:
                if evt is not None:
                    evt.set()
                self._write_queue.task_done()

    def _dispatch(self, conn: sqlite3.Connection, op: str, args: Any) -> Any:
        """Route an operation to the correct SQL handler. Runs in writer thread."""
        if op == _OP_INIT:
            conn.executescript(_SCHEMA)
            try:
                # Add OCR columns backward-compatibly (Phase 1)
                alter_statements = [
                    "ALTER TABLE tracked_objects ADD COLUMN brand TEXT DEFAULT ''",
                    "ALTER TABLE tracked_objects ADD COLUMN product_type TEXT DEFAULT ''",
                    "ALTER TABLE tracked_objects ADD COLUMN detected_texts TEXT DEFAULT '[]'",
                    "ALTER TABLE tracked_objects ADD COLUMN best_text TEXT DEFAULT ''",
                    "ALTER TABLE tracked_objects ADD COLUMN inferred_display_label TEXT DEFAULT ''",
                    "ALTER TABLE report_objects ADD COLUMN brand TEXT DEFAULT ''",
                    "ALTER TABLE report_objects ADD COLUMN product_type TEXT DEFAULT ''",
                    "ALTER TABLE report_objects ADD COLUMN best_text TEXT DEFAULT ''",
                    "ALTER TABLE report_objects ADD COLUMN inferred_display_label TEXT DEFAULT ''"
                ]
                for stmt in alter_statements:
                    try:
                        conn.execute(stmt)
                    except sqlite3.OperationalError:
                        pass # already exists

                # Phase 2: Search Indexes for OCR columns and timestamps
                search_indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_tracked_brand ON tracked_objects(brand)",
                    "CREATE INDEX IF NOT EXISTS idx_tracked_product ON tracked_objects(product_type)",
                    "CREATE INDEX IF NOT EXISTS idx_tracked_first_seen ON tracked_objects(first_seen)",
                    "CREATE INDEX IF NOT EXISTS idx_tracked_last_seen ON tracked_objects(last_seen)"
                ]
                for idx in search_indexes:
                    conn.execute(idx)

            except Exception as e:
                log.warning("[DB] Migration warning: %s", e)
            return None

        if op == _OP_START_SESSION:
            cur = conn.execute(
                "INSERT INTO sessions (started_at) VALUES (?)", (_iso_now(),)
            )
            log.info("[DB] Session #%d started.", cur.lastrowid)
            return cur.lastrowid

        if op == _OP_END_SESSION:
            session_id, total_reports, total_objects, total_events = args
            conn.execute(
                """UPDATE sessions
                      SET ended_at=?, total_reports=?, total_unique_objects=?, total_events=?
                    WHERE id=?""",
                (_iso_now(), total_reports, total_objects, total_events, session_id),
            )
            log.info(
                "[DB] Session #%d closed. Reports=%d Objects=%d Events=%d",
                session_id, total_reports, total_objects, total_events,
            )
            return None

        if op == _OP_SAVE_REPORT:
            (sid, report_number, total_objects, new_objects,
             removed_objects, stability, fps, cpu, ram) = args
            cur = conn.execute(
                """INSERT INTO reports
                   (session_id, reported_at, report_number, total_objects,
                    new_objects, removed_objects, scene_stability, fps, cpu_percent, ram_mb)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (sid, _iso_now(), report_number, total_objects,
                 new_objects, removed_objects, stability, fps, cpu, ram),
            )
            return cur.lastrowid

        if op == _OP_SAVE_OBJ_ROWS:
            report_id, session_id, objects = args
            rows = [
                (
                    report_id, session_id,
                    o["track_id"], o["object_number"],
                    o["label"], o["yolo_label"], o["category"],
                    o["confidence"], o["duration_seconds"],
                    int(o["is_new"]), int(o["is_gemini_verified"]),
                    o.get("gemini_description"),
                    o.get("bbox_x1"), o.get("bbox_y1"),
                    o.get("bbox_x2"), o.get("bbox_y2"),
                )
                for o in objects
            ]
            conn.executemany(
                """INSERT INTO report_objects
                   (report_id, session_id, track_id, object_number,
                    label, yolo_label, category, confidence, duration_seconds,
                    is_new, is_gemini_verified, gemini_description,
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )
            # Bump report_appearances in tracked_objects for each track
            for o in objects:
                conn.execute(
                    """UPDATE tracked_objects
                          SET report_appearances = report_appearances + 1,
                              last_seen = ?
                        WHERE session_id=? AND track_id=?""",
                    (_iso_now(), session_id, o["track_id"]),
                )
            return None

        if op == _OP_OBJECT_NEW:
            (session_id, track_id, yolo_label, display_label,
             category, confidence, bbox, is_returned) = args
            now = _iso_now()
            conn.execute(
                """INSERT INTO tracked_objects
                   (session_id, track_id, yolo_label, display_label, category,
                    first_seen, last_seen, highest_confidence, latest_confidence,
                    confidence_sum, confidence_count, status)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,'active')
                   ON CONFLICT(session_id, track_id) DO UPDATE SET
                       last_seen        = excluded.last_seen,
                       latest_confidence= excluded.latest_confidence,
                       status           = 'active'""",
                (session_id, track_id, yolo_label, display_label, category,
                 now, now, confidence, confidence, confidence, 1),
            )
            extra = json.dumps({"bbox": list(bbox)}) if bbox else None
            event_type = "returned" if is_returned else "new"
            conn.execute(
                """INSERT INTO object_events
                   (session_id, event_at, event_type, track_id, label, category, confidence, extra_data)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (session_id, now, event_type, track_id, display_label, category, confidence, extra),
            )
            return None

        if op == _OP_OBJECT_REMOVED:
            (session_id, track_id, label, category, confidence, duration) = args
            now = _iso_now()
            conn.execute(
                """UPDATE tracked_objects
                      SET last_seen          = ?,
                          status             = 'removed',
                          total_duration_sec = total_duration_sec + ?,
                          latest_confidence  = ?,
                          confidence_sum     = confidence_sum + ?,
                          confidence_count   = confidence_count + 1,
                          highest_confidence = MAX(highest_confidence, ?)
                    WHERE session_id=? AND track_id=?""",
                (now, duration or 0, confidence or 0, confidence or 0,
                 confidence or 0, session_id, track_id),
            )
            conn.execute(
                """INSERT INTO object_events
                   (session_id, event_at, event_type, track_id, label, category,
                    confidence, duration_seconds)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (session_id, now, "removed", track_id, label, category,
                 confidence, duration),
            )
            return None

        if op == _OP_OBJECT_VERIFIED:
            (session_id, track_id, refined_label, yolo_label) = args
            now = _iso_now()
            conn.execute(
                """UPDATE tracked_objects
                      SET display_label      = ?,
                          is_gemini_verified = 1,
                          last_seen          = ?
                    WHERE session_id=? AND track_id=?""",
                (refined_label, now, session_id, track_id),
            )
            extra = json.dumps({"yolo_label": yolo_label, "refined_label": refined_label})
            conn.execute(
                """INSERT INTO object_events
                   (session_id, event_at, event_type, track_id, label, category, extra_data)
                   SELECT ?, ?, 'verified', track_id, ?, category, ?
                     FROM tracked_objects WHERE session_id=? AND track_id=?""",
                (session_id, now, refined_label, extra, session_id, track_id),
            )
            return None

        if op == _OP_OBJECT_DESCRIBED:
            (session_id, track_id, description) = args
            now = _iso_now()
            conn.execute(
                """UPDATE tracked_objects
                      SET gemini_description = ?,
                          last_seen          = ?
                    WHERE session_id=? AND track_id=?""",
                (description, now, session_id, track_id),
            )
            extra = json.dumps({"preview": description[:80] if description else ""})
            conn.execute(
                """INSERT INTO object_events
                   (session_id, event_at, event_type, track_id, label, category, extra_data)
                   SELECT ?, ?, 'description_updated', track_id, display_label, category, ?
                     FROM tracked_objects WHERE session_id=? AND track_id=?""",
                (session_id, now, extra, session_id, track_id),
            )
            return None

        if op == _OP_CUSTOM_EVENT:
            (session_id, track_id, event_type, label, category, confidence, detail) = args
            now = _iso_now()
            extra = json.dumps({"detail": detail}) if detail else None
            conn.execute(
                """INSERT INTO object_events
                   (session_id, event_at, event_type, track_id, label, category, confidence, extra_data)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (session_id, now, event_type, track_id, label, category, confidence, extra),
            )
            return None

        if op == _OP_OCR_RESULT:
            (session_id, entity_id, track_id, raw_texts, best_text, brand, inferred_label) = args
            now = _iso_now()
            raw_texts_json = json.dumps(raw_texts)
            
            # 1. Update tracked_objects with OCR data
            conn.execute(
                """UPDATE tracked_objects
                      SET brand = ?,
                          best_text = ?,
                          detected_texts = ?,
                          inferred_display_label = ?
                    WHERE session_id=? AND track_id=?""",
                (brand, best_text, raw_texts_json, inferred_label, session_id, track_id),
            )
            
            # 2. Add to ocr_events log
            conn.execute(
                """INSERT INTO ocr_events
                   (session_id, entity_id, track_id, event_at, raw_texts, best_text, brand, inferred_label)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (session_id, entity_id, track_id, now, raw_texts_json, best_text, brand, inferred_label),
            )
            return None

        if op == "get_session_summary":
            session_id = args
            summary = {
                "most_common": [],
                "longest_dwell": [],
                "avg_dwell_by_category": [],
                "total_events": 0,
            }
            try:
                cur = conn.execute(
                    """SELECT display_label, COUNT(*) as c 
                       FROM tracked_objects 
                       WHERE session_id=? 
                       GROUP BY display_label 
                       ORDER BY c DESC LIMIT 3""", (session_id,)
                )
                summary["most_common"] = [{"label": row[0], "count": row[1]} for row in cur.fetchall()]
                
                cur = conn.execute(
                    """SELECT display_label, total_duration_sec 
                       FROM tracked_objects 
                       WHERE session_id=? 
                       ORDER BY total_duration_sec DESC LIMIT 3""", (session_id,)
                )
                summary["longest_dwell"] = [{"label": row[0], "duration": row[1]} for row in cur.fetchall()]
                
                cur = conn.execute(
                    """SELECT category, AVG(total_duration_sec) as avg_d
                       FROM tracked_objects 
                       WHERE session_id=? AND status='removed'
                       GROUP BY category 
                       ORDER BY avg_d DESC""", (session_id,)
                )
                summary["avg_dwell_by_category"] = [{"category": row[0], "duration": row[1]} for row in cur.fetchall()]
                
                cur = conn.execute(
                    "SELECT COUNT(*) FROM object_events WHERE session_id=?", (session_id,)
                )
                summary["total_events"] = cur.fetchone()[0]
            except Exception as e:
                log.error("[DB] Summary query failed: %s", e)
            return summary

        log.warning("[DB] Unknown operation: %s", op)
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Queue Submission Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _submit_sync(self, op: str, args: Any) -> Any:
        """Submit operation and BLOCK until writer thread completes it. Returns result."""
        evt = threading.Event()
        result_box: list = [None, None]   # [result, exception]
        self._write_queue.put((op, args, evt, result_box))
        completed = evt.wait(timeout=10.0)
        if not completed:
            log.error("[DB] Sync operation '%s' timed out after 10s.", op)
            return None
        if result_box[1] is not None:
            raise result_box[1]
        return result_box[0]

    def _submit_async(self, op: str, args: Any) -> None:
        """Submit operation to writer thread. Returns immediately (fire-and-forget)."""
        try:
            self._write_queue.put_nowait((op, args, None, None))
        except queue.Full:
            log.warning("[DB] Write queue full — dropped async op: %s", op)

    # ──────────────────────────────────────────────────────────────────────────
    # Public Sync API  (return IDs — called from main thread at startup/report time)
    # ──────────────────────────────────────────────────────────────────────────

    def start_session(self) -> int:
        """Create a new session row and return its ID. Synchronous."""
        return self._submit_sync(_OP_START_SESSION, None)

    def get_session_summary(self, session_id: int) -> dict:
        """Returns analytics summary for the session. Synchronous."""
        return self._submit_sync("get_session_summary", session_id)

    def save_report(
        self,
        session_id: int,
        report_number: int,
        total_objects: int,
        new_objects: int,
        removed_objects: int,
        scene_stability: float,
        fps: Optional[float],
        cpu_percent: Optional[float],
        ram_mb: Optional[float],
    ) -> int:
        """Insert a report row and return its ID. Synchronous (report_id needed for objects)."""
        return self._submit_sync(
            _OP_SAVE_REPORT,
            (session_id, report_number, total_objects, new_objects,
             removed_objects, scene_stability, fps, cpu_percent, ram_mb),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Public Async API  (fire-and-forget — detection loop never stalls)
    # ──────────────────────────────────────────────────────────────────────────

    def on_ocr_result(
        self,
        session_id: int,
        entity_id: str,
        track_id: int,
        raw_texts: List[str],
        best_text: str,
        brand: str,
        inferred_label: str
    ) -> None:
        """Asynchronously log OCR result and update tracked_objects."""
        self._submit_async(
            _OP_OCR_RESULT,
            (session_id, entity_id, track_id, raw_texts, best_text, brand, inferred_label)
        )

    def save_report_objects(
        self, report_id: int, session_id: int, objects: List[dict]
    ) -> None:
        """Persist all OBJECT N rows for a report and bump report_appearances. Async."""
        self._submit_async(_OP_SAVE_OBJ_ROWS, (report_id, session_id, objects))

    def on_object_new(
        self,
        session_id: int,
        track_id: int,
        yolo_label: str,
        display_label: str,
        category: str,
        confidence: float,
        bbox: Optional[tuple] = None,
        is_returned: bool = False,
    ) -> None:
        """Called when a new tracked object enters the scene. Async."""
        self._submit_async(
            _OP_OBJECT_NEW,
            (session_id, track_id, yolo_label, display_label, category, confidence, bbox, is_returned),
        )

    def on_object_removed(
        self,
        session_id: int,
        track_id: int,
        label: str,
        category: str,
        confidence: Optional[float],
        duration_seconds: Optional[float],
    ) -> None:
        """Called when a tracked object leaves the scene. Async."""
        self._submit_async(
            _OP_OBJECT_REMOVED,
            (session_id, track_id, label, category, confidence, duration_seconds),
        )

    def on_object_verified(
        self,
        session_id: int,
        track_id: int,
        refined_label: str,
        yolo_label: str,
    ) -> None:
        """Called when Gemini refines an object's label. Logs 'verified' event. Async."""
        self._submit_async(
            _OP_OBJECT_VERIFIED,
            (session_id, track_id, refined_label, yolo_label),
        )

    def on_description_updated(
        self,
        session_id: int,
        track_id: int,
        description: str,
    ) -> None:
        """Called when Gemini generates a description. Logs 'description_updated' event. Async."""
        self._submit_async(
            _OP_OBJECT_DESCRIBED,
            (session_id, track_id, description),
        )

    def on_object_event(
        self,
        session_id: int,
        track_id: int,
        event_type: str,
        label: str,
        category: str,
        confidence: float,
        detail: str,
    ) -> None:
        """Called for spatial and state events (stationary, moved, abandoned). Async."""
        self._submit_async(
            _OP_CUSTOM_EVENT,
            (session_id, track_id, event_type, label, category, confidence, detail),
        )

    def on_ocr_result(
        self, session_id: int, entity_id: str, track_id: int, 
        raw_texts: List[str], best_text: str, brand: str, inferred_label: str
    ) -> None:
        """Called when async OCR thread completes processing a crop."""
        self._submit_async(
            _OP_OCR_RESULT,
            (session_id, entity_id, track_id, raw_texts, best_text, brand, inferred_label)
        )

    def end_session(
        self,
        session_id: int,
        total_reports: int,
        total_unique_objects: int,
        total_events: int,
    ) -> None:
        """Close session with final stats. Synchronous to ensure it commits before process exits."""
        self._submit_sync(
            _OP_END_SESSION,
            (session_id, total_reports, total_unique_objects, total_events),
        )

    def shutdown(self) -> None:
        """Drain the write queue and close the DB connection cleanly."""
        evt = threading.Event()
        self._write_queue.put((_OP_STOP, None, evt, None))
        closed = evt.wait(timeout=8.0)
        if not closed:
            log.warning("[DB] Writer thread did not stop cleanly within 8s.")
        log.info("[DB] Shutdown complete.")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _iso_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
