import asyncio
import json
import logging
import threading
import time
from typing import Optional, Set
from datetime import datetime

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ultralytics import YOLO

log = logging.getLogger("mobile_ws")

router = APIRouter()

# ──────────────────────────────────────────────────────────────────────────────
# FrameBuffer — thread-safe single-slot latest-frame store
# ──────────────────────────────────────────────────────────────────────────────

class FrameBuffer:
    def __init__(self) -> None:
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._event = threading.Event()

    def put(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame
        self._event.set()

    def get(self, timeout: float = 1.0) -> Optional[np.ndarray]:
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
    def __init__(self) -> None:
        self._clients: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def add(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.add(ws)

    async def remove(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)

    async def broadcast(self, payload: str) -> None:
        async with self._lock:
            dead = set()
            for ws in self._clients:
                try:
                    await ws.send_text(payload)
                except Exception:
                    dead.add(ws)
            self._clients -= dead

_monitor_registry = MonitorRegistry()
_loop: Optional[asyncio.AbstractEventLoop] = None

# ──────────────────────────────────────────────────────────────────────────────
# Independent Inference Worker for Flutter App
# ──────────────────────────────────────────────────────────────────────────────

def _mobile_inference_worker() -> None:
    log.info("[MobileWorker] Started.")
    # Initialize our own lightweight YOLO model
    model = YOLO("yolov8n.pt")
    
    while True:
        frame = _frame_buffer.get(timeout=1.0)
        if frame is None:
            continue

        try:
            results = model(frame, verbose=False)[0]
            
            # Format report exactly as the Flutter app expects
            client_objects = []
            for i, b in enumerate(results.boxes):
                cls_id = int(b.cls[0])
                conf = float(b.conf[0])
                label = model.names[cls_id]
                
                client_objects.append({
                    "track_id": i,
                    "label": label.title(),
                    "category": "Detection",
                    "confidence": round(conf, 2),
                    "duration_seconds": 0.0,
                    "is_new": True,
                })

            report_dict = {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "total_objects": len(client_objects),
                "new_objects": len(client_objects),
                "removed_objects": 0,
                "scene_stability": 100.0,
                "objects": client_objects,
            }

            if _loop is not None and not _loop.is_closed():
                payload = json.dumps(report_dict)
                asyncio.run_coroutine_threadsafe(
                    _monitor_registry.broadcast(payload), _loop
                )
        except Exception as exc:
            log.error("[MobileWorker] Inference error: %s", exc)

# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

def _decode_jpeg(data: bytes) -> Optional[np.ndarray]:
    try:
        arr = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame
    except Exception:
        return None

@router.websocket("/ws/capture")
async def on_capture_frame(ws: WebSocket) -> None:
    await ws.accept()
    log.info("[Capture] Flutter client connected.")
    loop = asyncio.get_running_loop()
    try:
        while True:
            data: bytes = await ws.receive_bytes()
            frame = await loop.run_in_executor(None, _decode_jpeg, data)
            if frame is not None:
                await loop.run_in_executor(None, _frame_buffer.put, frame)
    except WebSocketDisconnect:
        log.info("[Capture] Flutter client disconnected.")

@router.websocket("/ws/monitor")
async def on_monitor_connect(ws: WebSocket) -> None:
    await ws.accept()
    await _monitor_registry.add(ws)
    try:
        while True:
            await asyncio.sleep(30)
    except WebSocketDisconnect:
        pass
    finally:
        await _monitor_registry.remove(ws)

def start_mobile_worker(loop: asyncio.AbstractEventLoop):
    global _loop
    _loop = loop
    worker = threading.Thread(target=_mobile_inference_worker, daemon=True, name="MobileInferenceWorker")
    worker.start()
