"""
server.py
=========
FastAPI WebSocket server for the Edge Computing CV pipeline.

Endpoints
---------
GET  /               → serves templates/index.html (the mobile PWA)
WS   /ws/capture     → receives raw JPEG frames from the mobile camera client
WS   /ws/monitor     → broadcasts clean detection JSON to monitor clients

Architecture
------------
    ┌─────────────┐   binary JPEG   ┌──────────────────┐
    │ Mobile Phone│ ──────────────► │  /ws/capture     │
    │ (camera)    │                 │  FrameBuffer      │
    └─────────────┘                 │  InferenceWorker  │
                                    │  cv2.imshow()     │ ← developer only
    ┌─────────────┐   JSON report   │  /ws/monitor ─────┼──► all monitors
    │ Monitor Tab │ ◄────────────── └──────────────────┘
    └─────────────┘

Security
--------
    - cv2.imshow() is called on the server machine ONLY — never streamed back.
    - Report JSON sent to /ws/monitor contains ONLY:
        timestamp, total_objects, new_objects, removed_objects,
        scene_stability, and per-object: label, category, confidence,
        duration_seconds, is_new.
    - System stats (FPS, CPU, RAM), model names, server paths, and
      error stack traces are NEVER sent to any client.

RTSP upgrade path
-----------------
    To use IP cameras instead of the mobile WebSocket stream:
    1. Replace the `on_capture_frame()` WebSocket handler with a background
       thread that reads from `cv2.VideoCapture("rtsp://...")`.
    2. FrameBuffer and InferenceWorker stay exactly the same.
"""

import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional, Set

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import engine  # Our CV bridge

# ──────────────────────────────────────────────────────────────────────────────
# Logging — server-side only, never forwarded to clients
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("server")

# ──────────────────────────────────────────────────────────────────────────────
# FastAPI App + Static Files
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Edge CV Server",
    docs_url=None,      # Disable Swagger UI (no internal API exposure)
    redoc_url=None,     # Disable ReDoc
    openapi_url=None,   # Disable OpenAPI schema endpoint
)

_BASE = Path(__file__).resolve().parent

# Mount static assets (PWA manifest, service worker, icons)
_STATIC = _BASE / "static"
_STATIC.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")

# Load the HTML template once at startup
_TEMPLATE_PATH = _BASE / "templates" / "index.html"


# ──────────────────────────────────────────────────────────────────────────────
# FrameBuffer — thread-safe single-slot latest-frame store
# ──────────────────────────────────────────────────────────────────────────────

class FrameBuffer:
    """
    Holds only the most recently received frame.
    Older frames are discarded automatically — zero queue buildup, zero lag.
    """

    def __init__(self) -> None:
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._event = threading.Event()    # Signals when a new frame arrives

    def put(self, frame: np.ndarray) -> None:
        """Called from the asyncio WebSocket handler (via run_in_executor)."""
        with self._lock:
            self._frame = frame
        self._event.set()

    def get(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Block until a new frame arrives or timeout.
        Returns the latest frame and clears the event.
        """
        got = self._event.wait(timeout=timeout)
        if not got:
            return None
        self._event.clear()
        with self._lock:
            return self._frame


_frame_buffer = FrameBuffer()


# ──────────────────────────────────────────────────────────────────────────────
# Monitor Registry — thread-safe set of connected WebSocket clients
# ──────────────────────────────────────────────────────────────────────────────

class MonitorRegistry:
    """Manages the set of currently connected /ws/monitor WebSocket clients."""

    def __init__(self) -> None:
        self._clients: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def add(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.add(ws)
        log.info("[Monitor] Client connected. Total: %d", len(self._clients))

    async def remove(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)
        log.info("[Monitor] Client disconnected. Total: %d", len(self._clients))

    async def broadcast(self, payload: str) -> None:
        """Send JSON string to all connected monitor clients."""
        async with self._lock:
            dead = set()
            for ws in self._clients:
                try:
                    await ws.send_text(payload)
                except Exception:
                    dead.add(ws)
            self._clients -= dead


_monitor_registry = MonitorRegistry()

# asyncio event loop reference (set at startup)
_loop: Optional[asyncio.AbstractEventLoop] = None


# ──────────────────────────────────────────────────────────────────────────────
# Inference Worker — runs in a dedicated background daemon thread
# ──────────────────────────────────────────────────────────────────────────────

def _inference_worker() -> None:
    """
    Background thread: pull frames → run CV engine → show dev window → broadcast.

    This thread owns cv2.imshow (developer-only window).
    It schedules asyncio coroutines via run_coroutine_threadsafe to broadcast
    the cleaned report JSON to monitor clients.
    """
    log.info("[Worker] Inference worker started.")
    window_name = "Developer View — Smart Vision"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    while True:
        frame = _frame_buffer.get(timeout=1.0)

        if frame is None:
            # No frame received yet — keep the window alive so it doesn't freeze
            cv2.waitKey(1)
            continue

        # ── Run CV inference pipeline ──────────────────────────────────────
        try:
            annotated_frame, report_dict = engine.process_frame(frame)
        except Exception as exc:
            log.error("[Worker] Engine error: %s", exc)
            cv2.waitKey(1)
            continue

        # ── Developer-only display (NEVER sent to clients) ─────────────────
        try:
            cv2.imshow(window_name, annotated_frame)
        except Exception:
            pass  # Headless fallback — imshow silently fails without display

        # ── Broadcast clean report to all monitor clients ──────────────────
        if _loop is not None and not _loop.is_closed():
            payload = json.dumps(report_dict)
            asyncio.run_coroutine_threadsafe(
                _monitor_registry.broadcast(payload), _loop
            )

        # ── Handle 'Q' keypress to quit the dev window ────────────────────
        if cv2.waitKey(1) & 0xFF == ord("q"):
            log.info("[Worker] Developer quit the imshow window.")
            cv2.destroyAllWindows()
            break

    log.info("[Worker] Inference worker exiting.")


# ──────────────────────────────────────────────────────────────────────────────
# Startup / Shutdown lifecycle
# ──────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def _on_startup() -> None:
    global _loop
    _loop = asyncio.get_running_loop()

    # Start inference worker as a daemon thread
    worker = threading.Thread(target=_inference_worker, daemon=True, name="InferenceWorker")
    worker.start()
    log.info("[Server] Startup complete. Inference worker running.")


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_root() -> HTMLResponse:
    """Serve the mobile PWA shell."""
    html = _TEMPLATE_PATH.read_text(encoding="utf-8")
    return HTMLResponse(content=html)


@app.websocket("/ws/capture")
async def on_capture_frame(ws: WebSocket) -> None:
    """
    Receive raw JPEG frames from the mobile camera client.

    Protocol:
        Client sends binary WebSocket messages containing a JPEG image.
        Server decodes, puts into FrameBuffer, sends no reply.
        On error, connection is cleanly closed.
    """
    await ws.accept()
    log.info("[Capture] Camera client connected.")

    loop = asyncio.get_running_loop()

    try:
        while True:
            data: bytes = await ws.receive_bytes()

            # Decode JPEG → OpenCV BGR frame (off the event loop thread)
            frame = await loop.run_in_executor(
                None, _decode_jpeg, data
            )

            if frame is not None:
                # Put frame into buffer (also off event loop)
                await loop.run_in_executor(None, _frame_buffer.put, frame)

    except WebSocketDisconnect:
        log.info("[Capture] Camera client disconnected.")
    except Exception as exc:
        log.warning("[Capture] Unexpected error: %s", exc)


@app.websocket("/ws/monitor")
async def on_monitor_connect(ws: WebSocket) -> None:
    """
    Broadcast clean detection JSON reports to monitoring clients.

    Protocol:
        Server pushes JSON text messages whenever new reports are available.
        Client sends no messages (receive is not expected).
    """
    await ws.accept()
    await _monitor_registry.add(ws)

    try:
        # Keep connection alive — server pushes data, client just listens.
        while True:
            await asyncio.sleep(30)   # Heartbeat: keep loop alive
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.warning("[Monitor] Client error: %s", exc)
    finally:
        await _monitor_registry.remove(ws)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _decode_jpeg(data: bytes) -> Optional[np.ndarray]:
    """Decode a JPEG byte payload into an OpenCV BGR ndarray."""
    try:
        arr = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame if frame is not None else None
    except Exception as exc:
        log.debug("[Decode] JPEG decode error: %s", exc)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,          # Disable reload when running directly
        log_level="info",
    )
