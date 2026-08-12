"""
web_server.py
=============
FastAPI web server for the Smart Vision Assistant.

Endpoints:
  GET  /api/status         → Live system metrics JSON
  GET  /scene/report       → Full session intelligence report (active + inactive)
  GET  /objects            → Active objects (conf ≥ 80%, age ≥ 2s)
  GET  /session            → Full session memory (active + inactive)
  WS   /ws/live            → WebSocket push (1-second interval)
  GET  /video/stream       → MJPEG camera stream
  GET  /health             → Server health check

Start with:
  uvicorn web_server:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import logging
import time as _time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from smart_vision_headless import SmartVisionHeadless
from search_engine import SearchEngine
from timeline_engine import TimelineEngine
from knowledge_graph import KnowledgeGraph
from query_interpreter import QueryInterpreter
from services.watchlist_manager import WatchlistManager

log = logging.getLogger("web_server")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

# ──────────────────────────────────────────────────────────────────────────────
# Global vision instance
# ──────────────────────────────────────────────────────────────────────────────

_vision: SmartVisionHeadless = None  # type: ignore

# ──────────────────────────────────────────────────────────────────────────────
# Lifespan
# ──────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _vision
    log.info("Starting SmartVisionHeadless…")
    _vision = SmartVisionHeadless()
    _vision.start()
    log.info("Vision system started. Server ready.")
    yield
    log.info("Shutting down vision system…")
    _vision.stop()
    log.info("Shutdown complete.")


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Smart Vision Assistant API",
    description="Visual Scene Intelligence Engine — Session Memory Web Interface",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve snapshots folder
snapshots_dir = Path("data/entity_snapshots")
snapshots_dir.mkdir(parents=True, exist_ok=True)
app.mount("/snapshots", StaticFiles(directory="data/entity_snapshots"), name="snapshots")

# Instantiate stateless engines
search_engine = SearchEngine()
timeline_engine = TimelineEngine()
knowledge_graph = KnowledgeGraph()
query_interpreter = QueryInterpreter()

# Object Watchlist — isolated feature, zero inference cost
_watchlist = WatchlistManager()

# ──────────────────────────────────────────────────────────────────────────────
# REST Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/api/status")
async def get_status() -> JSONResponse:
    """Live system metrics + session object counts."""
    state = _vision.get_state()
    registry = _vision.get_entity_registry()
    return JSONResponse({
        "fps":              state.get("fps", 0.0),
        "cpu":              state.get("cpu", 0.0),
        "ram_mb":           state.get("ram_mb", 0.0),
        "active_objects":   state.get("active_objects", 0),
        "inactive_objects": registry.session_object_count - state.get("active_objects", 0),
        "session_objects":  registry.session_object_count,
        "session_time":     state.get("session_time", 0.0),
        "status":           state.get("status", {}),
    })


@app.get("/scene/report")
async def get_scene_report() -> JSONResponse:
    """
    Full session intelligence report.
    Contains BOTH active (currently visible) and inactive (previously seen) objects.
    Confidence gate: >= 80%. Age gate for active: >= 2s.
    Returns structured JSON — never text blobs.
    """
    state = _vision.get_state()
    report = state.get("report", {})
    if not report:
        report = {
            "summary": "Initialising…",
            "active_objects": 0,
            "session_objects": 0,
            "objects": [],
            "inactive_objects": [],
            "events": [],
            "relationships": [],
        }
    return JSONResponse(report)


@app.get("/objects")
async def get_objects() -> JSONResponse:
    """
    Currently active objects: conf >= 80%, age >= 2s.
    For the live camera overlay section.
    """
    registry = _vision.get_entity_registry()
    active = registry.get_active(min_confidence=0.80, min_age_seconds=2.0)
    return JSONResponse({
        "count":   len(active),
        "objects": [e.to_dict() for e in active],
        "filters": {"min_confidence": 0.80, "min_age_seconds": 2.0},
    })


@app.get("/session")
async def get_session() -> JSONResponse:
    """
    Full session memory: all objects ever seen with conf >= 80%.
    Includes both active and inactive objects.
    This never shrinks — it is the complete session history.
    """
    registry = _vision.get_entity_registry()
    session = registry.get_session(min_confidence=0.80, min_age_seconds=2.0)
    return JSONResponse(session)


@app.get("/search")
async def search_memory(query: str = Query(None), brand: str = None, category: str = None, text: str = None, event: str = None) -> JSONResponse:
    """Search visual memory using natural language query or explicit filters."""
    filters = {}
    if query:
        filters = query_interpreter.parse(query)
    
    # Explicit filters override natural language parsed filters
    if brand: filters["brand"] = brand
    if category: filters["category"] = category
    if text: filters["text"] = text
    if event: filters["event"] = event
    
    session_id = _vision._session_id
    results = search_engine.search(session_id, filters)
    return JSONResponse({"results": results, "session_id": session_id, "filters_applied": filters})


@app.get("/timeline")
async def get_timeline(limit: int = Query(200), all_sessions: bool = Query(False)) -> JSONResponse:
    """Fetch chronological event timeline for the session(s)."""
    session_id = _vision._session_id
    if all_sessions:
        timeline = timeline_engine.get_all_timeline(limit=limit)
        session_id = "ALL"
    else:
        timeline = timeline_engine.get_session_timeline(session_id, limit=limit)
    return JSONResponse({"timeline": timeline, "session_id": session_id, "count": len(timeline)})


@app.get("/api/db-stats")
async def get_db_stats() -> JSONResponse:
    """Verify DB is storing reports — returns counts of all tables for the current session."""
    import sqlite3
    from pathlib import Path
    db_path = Path("data/smart_vision.db")
    if not db_path.exists():
        return JSONResponse({"error": "Database file not found", "path": str(db_path.resolve())})
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        session_id = _vision._session_id if _vision else None
        stats = {"db_path": str(db_path.resolve()), "session_id": session_id}
        tables = ["sessions", "reports", "report_objects", "object_events", "tracked_objects", "ocr_events"]
        for t in tables:
            try:
                if session_id:
                    row = conn.execute(f"SELECT COUNT(*) as c FROM {t} WHERE session_id=?", (session_id,)).fetchone()
                else:
                    row = conn.execute(f"SELECT COUNT(*) as c FROM {t}").fetchone()
                stats[t] = row["c"] if row else 0
            except Exception:
                stats[t] = "n/a"
        # Also get all-time totals
        totals = {}
        for t in tables:
            try:
                row = conn.execute(f"SELECT COUNT(*) as c FROM {t}").fetchone()
                totals[t] = row["c"] if row else 0
            except Exception:
                totals[t] = "n/a"
        stats["all_sessions_totals"] = totals
        conn.close()
        return JSONResponse(stats)
    except Exception as e:
        return JSONResponse({"error": str(e)})


@app.get("/graph")
async def get_graph() -> JSONResponse:
    """Fetch session knowledge graph (nodes & edges)."""
    session_id = _vision._session_id
    graph = knowledge_graph.get_graph(session_id)
    return JSONResponse(graph)


@app.get("/api/diagnostics")
async def get_diagnostics() -> JSONResponse:
    """Comprehensive diagnostic endpoint."""
    state = _vision.get_state() if _vision else {}
    status = state.get("status", {})
    
    # Get OCR telemetry safely
    ocr_telemetry = {}
    if _vision and hasattr(_vision, '_ocr_processor') and _vision._ocr_processor:
        if hasattr(_vision._ocr_processor, 'get_telemetry'):
            ocr_telemetry = _vision._ocr_processor.get_telemetry()
    
    return JSONResponse({
        "status": "ok" if _vision and getattr(_vision, 'is_running', True) else "stopped",
        "ocr_metrics": {
            "queue_size": ocr_telemetry.get("queue_size", 0),
            "submitted": ocr_telemetry.get("crops_submitted", 0),
            "completed": ocr_telemetry.get("tasks_completed", 0),
            "text_found": ocr_telemetry.get("text_found", 0),
            "entities_enriched": ocr_telemetry.get("entities_enriched", 0),
            "dropped": 0,
        },
        "total_events": _vision._total_events if _vision and hasattr(_vision, '_total_events') else 0,
        "system": {
            "fps": status.get("fps", 0.0),
            "cpu": status.get("cpu", 0.0),
            "ram_mb": status.get("ram_mb", 0.0),
        }
    })


# ──────────────────────────────────────────────────────────────────────────────
# WebSocket — Live Push
# ──────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    """
    WebSocket — pushes full session state every second.

    Payload includes:
      objects          — active objects (conf >= 80%, age >= 2s)
      inactive_objects — previously seen objects (full session history)
      events           — all session events (active + inactive)
      relationships    — all session relationships with counts
      status           — component health + FPS/CPU/RAM
      report           — summary + category counts + session stats
      watchlist        — current watchlist status (idle / visible / lost / returned)
    """
    await ws.accept()
    log.info("WebSocket client connected: %s", ws.client)
    _last_tick = _time.monotonic()
    try:
        while True:
            now = _time.monotonic()
            elapsed = now - _last_tick
            _last_tick = now

            state = _vision.get_state()
            report = state.get("report", {})

            # ── Tick watchlist (consumes only EntityRegistry state) ──────────
            try:
                _watchlist.tick(_vision.get_entity_registry(), elapsed=max(elapsed, 0.1))
            except Exception as _we:
                log.debug("[Watchlist] tick error: %s", _we)

            payload = {
                "objects":          state.get("objects", []),
                "inactive_objects": state.get("inactive_objects", []),
                "events":           state.get("events", []),
                "relationships":    state.get("relationships", []),
                "status":           state.get("status", {}),
                "report": {
                    "summary":            report.get("summary", ""),
                    "active_objects":     report.get("active_objects", 0),
                    "session_objects":    report.get("session_objects", 0),
                    "inactive_count":     len(state.get("inactive_objects", [])),
                    "fps":                report.get("fps", 0.0),
                    "cpu":                report.get("cpu", 0.0),
                    "ram_mb":             report.get("ram_mb", 0.0),
                    "session_time":       report.get("session_time", 0.0),
                    "session_time_str":   report.get("session_time_str", "0s"),
                    "scene_stability":    report.get("scene_stability", 0.0),
                    "active_by_category": report.get("active_by_category", {}),
                    "session_by_category":report.get("session_by_category", {}),
                    "total_registered":   report.get("total_registered", 0),
                },
                # ── Watchlist state pushed every second ───────────────────────
                "watchlist": _watchlist.get_status(),
            }

            await ws.send_json(payload)
            await asyncio.sleep(1.0)

    except WebSocketDisconnect:
        log.info("WebSocket client disconnected: %s", ws.client)
    except Exception as exc:
        log.warning("WebSocket error: %s", exc)


# ──────────────────────────────────────────────────────────────────────────────
# MJPEG Camera Stream
# ──────────────────────────────────────────────────────────────────────────────

async def _mjpeg_generator() -> AsyncGenerator[bytes, None]:
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    tail = b"\r\n"
    while True:
        jpeg = _vision.get_latest_frame()
        if jpeg:
            yield boundary + jpeg + tail
        await asyncio.sleep(1 / 30)


@app.get("/video/stream")
async def video_stream():
    """MJPEG stream — React CameraView uses <img src='/video/stream' />."""
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace;boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Access-Control-Allow-Origin": "*",
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Object Watchlist Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/watchlist/available")
async def watchlist_available() -> JSONResponse:
    """
    Return currently ACTIVE entities suitable for the watchlist dropdown.
    Source: EntityRegistry.get_active() — conf >= 80%, age >= 2s.
    These are already stabilised entities; no detector query needed.
    """
    registry = _vision.get_entity_registry()
    available = _watchlist.get_available(registry)
    return JSONResponse({"available": available, "count": len(available)})


@app.post("/watchlist/watch")
async def watchlist_start(body: dict) -> JSONResponse:
    """
    Start monitoring an entity.
    Body: {"entity_uuid": str, "display_label": str, "category": str}
    """
    entity_uuid   = body.get("entity_uuid", "").strip()
    display_label = body.get("display_label", "").strip()
    category      = body.get("category", "").strip()

    if not entity_uuid or not display_label:
        return JSONResponse({"error": "entity_uuid and display_label are required"}, status_code=400)

    # Verify entity actually exists and is active right now
    registry = _vision.get_entity_registry()
    entity = registry._session.get(entity_uuid)
    if entity is None:
        return JSONResponse({"error": f"Entity '{entity_uuid[:8]}' not found in registry"}, status_code=404)

    _watchlist.watch(entity_uuid, display_label, category)
    return JSONResponse({"ok": True, "watching": display_label})


@app.post("/watchlist/stop")
async def watchlist_stop() -> JSONResponse:
    """Stop monitoring the current entity."""
    label = _watchlist.current_item.display_label if _watchlist.current_item else None
    _watchlist.stop()
    return JSONResponse({"ok": True, "stopped": label})


@app.get("/watchlist/status")
async def watchlist_status() -> JSONResponse:
    """
    Current watchlist state — polled by UI as fallback if WebSocket unavailable.
    """
    return JSONResponse(_watchlist.get_status())


# ──────────────────────────────────────────────────────────────────────────────
# Health Check
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> JSONResponse:
    registry = _vision.get_entity_registry() if _vision else None
    return JSONResponse({
        "status":          "ok",
        "vision_running":  _vision.is_running if _vision else False,
        "session_objects": registry.session_object_count if registry else 0,
    })
