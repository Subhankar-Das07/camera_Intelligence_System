import os
import uuid
import socket
import urllib.parse
import cv2
import threading
import time
import asyncio
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Tuple, Dict, Any, Optional

from core.registry import registry
from core.video_source import get_video_source, ThreadedCamera
from core.mobile_ws import router as mobile_router, start_mobile_worker

app = FastAPI(title="Video Analytics Testing Platform")

# ── Storage directories ──────────────────────────────────────────────────────
STORAGE_DIR          = "storage"
UPLOAD_DIR           = os.path.join(STORAGE_DIR, "uploads")
PREVIEW_DIR          = os.path.join(STORAGE_DIR, "previews")
OUTPUT_DIR           = os.path.join(STORAGE_DIR, "outputs")
ALERTS_DIR           = os.path.join(STORAGE_DIR, "alerts")
GUARDIAN_ALERTS_DIR  = os.path.join(STORAGE_DIR, "guardian_alerts")  # room_guardian pipeline clips

for d in [UPLOAD_DIR, PREVIEW_DIR, OUTPUT_DIR, ALERTS_DIR, GUARDIAN_ALERTS_DIR]:
    os.makedirs(d, exist_ok=True)

app.mount("/storage", StaticFiles(directory=STORAGE_DIR), name="storage")
app.include_router(mobile_router)

@app.on_event("startup")
async def _on_startup():
    loop = asyncio.get_running_loop()
    start_mobile_worker(loop)

# ── In-memory session state ──────────────────────────────────────────────────
# RTSP live streams: stream_id -> ThreadedCamera (raw, no AI)
rtsp_streams: Dict[str, ThreadedCamera] = {}
rtsp_dims:    Dict[str, Tuple[int, int]] = {}  # stream_id -> (width, height)

# Analysis sessions
active_sessions: Dict[str, Any] = {}
session_alerts:  Dict[str, List[Dict[str, Any]]] = {}
stop_events:     Dict[str, threading.Event] = {}   # session_id -> Event

# ── Pydantic models ──────────────────────────────────────────────────────────
class RtspConnectRequest(BaseModel):
    url: str

class ProcessRequest(BaseModel):
    video_id: str
    filename: str
    pipeline_name: str
    roi_normalized: List[Tuple[float, float]]
    config: Dict[str, Any] = {}
    stream_id: Optional[str] = None   # set when sourcing from a live RTSP session

# ── Guardian Pydantic models (decoupled — room_guardian pipeline only) ────────
class GuardianScanRequest(BaseModel):
    """Request a one-shot YOLO scan of a single frame to populate the selection UI."""
    stream_id: Optional[str] = None
    video_id:  Optional[str] = None
    filename:  Optional[str] = None

class GuardianWatchedObject(BaseModel):
    """One user-selected object to be guarded (either YOLO-enrolled or custom-ROI)."""
    id:               str
    type:             str                # "yolo" | "custom"
    label:            str
    class_id:         Optional[int] = None
    bbox_normalized:  List[float]        # [x, y, w, h] in 0-1 range

class GuardianStartRequest(BaseModel):
    """Start a guardian session for the given watched objects."""
    stream_id:      Optional[str] = None
    video_id:       Optional[str] = None
    filename:       Optional[str] = None
    watched_objects: List[GuardianWatchedObject] = []

# ── Helpers ──────────────────────────────────────────────────────────────────
def _socket_check(url: str, timeout: float = 3.0) -> Optional[str]:
    """Return None if host:port is reachable, or an error string if not.
    Uses socket.create_connection() which is cross-platform and handles
    Windows WSAEWOULDBLOCK (10035) correctly.
    """
    try:
        parsed = urllib.parse.urlparse(url)
        host = parsed.hostname
        port = parsed.port or (554 if parsed.scheme in ("rtsp", "rtsps") else 80)
        
        # create_connection handles all platform-specific quirks
        conn = socket.create_connection((host, port), timeout=timeout)
        conn.close()
        return None  # Reachable ✓
    except socket.timeout:
        return f"Host {parsed.hostname}:{port} did not respond in {timeout}s. Check NVR IP/port."
    except ConnectionRefusedError:
        return f"Host {parsed.hostname}:{port} refused the connection. Check if RTSP is enabled on the NVR."
    except Exception as e:
        return f"Cannot reach stream host: {e}"


def _mjpeg_generator_raw(stream_id: str):
    """Yield raw (un-annotated) MJPEG frames from an open ThreadedCamera."""
    cam = rtsp_streams.get(stream_id)
    if cam is None:
        return
    while stream_id in rtsp_streams:
        ret, frame = cam.read()
        if not ret or frame is None:
            time.sleep(0.05)
            continue
        ok, buf = cv2.imencode('.jpg', frame)
        if not ok:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + buf.tobytes() + b'\r\n')
        time.sleep(0.033)  # ~30fps

def _mjpeg_generator_analysis(session_id: str, pipeline, input_path, roi_normalized, config,
                               output_dir: str = None):
    """Yield AI-annotated MJPEG frames and collect alert events."""
    if output_dir is None:
        output_dir = ALERTS_DIR
    stop_ev = stop_events.get(session_id)
    generator = pipeline.run_on_video(
        input_path=input_path,
        output_dir=output_dir,
        roi_normalized=roi_normalized,
        config=config
    )
    for frame, alert_event in generator:
        if stop_ev and stop_ev.is_set():
            break
        if alert_event:
            session_alerts[session_id].append(alert_event)
        if frame is not None:
            ok, buf = cv2.imencode('.jpg', frame)
            if not ok:
                continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                   + buf.tobytes() + b'\r\n')
            # Removed time.sleep(0.033) here so processing goes as fast as possible

# ── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
        raise HTTPException(status_code=400, detail="Unsupported file format.")
    video_id  = str(uuid.uuid4())
    ext       = os.path.splitext(file.filename)[1]
    filename  = f"{video_id}{ext}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    cap = cv2.VideoCapture(file_path)
    ret, frame = cap.read()
    if not ret:
        cap.release(); os.remove(file_path)
        raise HTTPException(status_code=400, detail="Could not read video file.")
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    preview_fn   = f"{video_id}.jpg"
    preview_path = os.path.join(PREVIEW_DIR, preview_fn)
    cv2.imwrite(preview_path, frame)

    return {"video_id": video_id, "width": width, "height": height,
            "preview_url": f"/storage/previews/{preview_fn}", "filename": filename}


@app.post("/api/connect_stream")
async def connect_stream(body: RtspConnectRequest):
    """
    Open a live RTSP stream, grab a real preview frame for ROI drawing,
    and keep the stream running in the background.
    Returns stream_id, dims AND a preview_url (a real still frame from the camera).
    The preview_url is used by the frontend exactly like file upload — user draws ROI
    on a pixel-accurate still frame, so there is no letterbox mismatch.
    """
    url = body.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="No stream URL provided.")
        
    # If the user typed "0" or "1", convert to integer for local webcam support
    if isinstance(url, str) and url.isdigit():
        url = int(url)
    elif isinstance(url, str) and url.startswith(('rtsp://', 'rtsps://', 'http://', 'https://')):
        err = _socket_check(url)
        if err:
            raise HTTPException(status_code=400, detail=err)

    def _open():
        cam = ThreadedCamera(url)
        if not cam.isOpened():
            return None, None, None, None
        cam.start()
        # Try up to 5 seconds to read the first real frame (RTSP handshake)
        first_frame = None
        for _ in range(50):
            ret, frame = cam.read()
            if ret and frame is not None:
                first_frame = frame
                break
            time.sleep(0.1)
        if first_frame is None:
            w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return cam, w or 640, h or 480, None
        h, w = first_frame.shape[:2]
        return cam, w, h, first_frame

    loop = asyncio.get_running_loop()
    try:
        cam, width, height, first_frame = await asyncio.wait_for(
            loop.run_in_executor(None, _open), timeout=15.0
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=400,
            detail="Stream timed out (15s). Ensure the RTSP URL is correct and the NVR is reachable.")

    if cam is None:
        raise HTTPException(status_code=400,
            detail="Could not open stream. Check credentials and stream path in your RTSP URL.")

    stream_id = str(uuid.uuid4())
    rtsp_streams[stream_id] = cam
    rtsp_dims[stream_id]    = (width, height)

    return {"stream_id": stream_id, "width": width, "height": height}



@app.get("/api/raw_stream/{stream_id}")
def raw_stream(stream_id: str):
    """Raw un-annotated MJPEG stream for live preview & ROI drawing."""
    if stream_id not in rtsp_streams:
        raise HTTPException(status_code=404, detail="Stream not found.")
    return StreamingResponse(
        _mjpeg_generator_raw(stream_id),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )


@app.get("/api/pipelines")
async def get_pipelines():
    return {"pipelines": registry.get_available_pipelines()}


@app.post("/api/start_analysis")
async def start_analysis(request: ProcessRequest):
    session_id = str(uuid.uuid4())
    active_sessions[session_id] = request
    session_alerts[session_id]  = []
    stop_events[session_id]     = threading.Event()
    return {"session_id": session_id}


@app.get("/api/stream/{session_id}")
def stream_video(session_id: str):
    """AI-annotated MJPEG stream. Stops when stop_event is set."""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    req = active_sessions[session_id]

    try:
        pipeline = registry.get_pipeline(req.pipeline_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not pipeline:
        raise HTTPException(status_code=400, detail="Pipeline not found.")

    if req.stream_id and req.stream_id in rtsp_streams:
        # Pass the existing live camera object instead of reopening a second connection
        # This prevents NVRs from throttling/rejecting the duplicate connection.
        input_path = rtsp_streams[req.stream_id]
    else:
        input_path = os.path.join(UPLOAD_DIR, req.filename)

    # Route guardian sessions to their own storage folder
    out_dir = GUARDIAN_ALERTS_DIR if req.pipeline_name == "room_guardian" else ALERTS_DIR

    return StreamingResponse(
        _mjpeg_generator_analysis(session_id, pipeline, input_path,
                                  req.roi_normalized, req.config, output_dir=out_dir),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )


@app.post("/api/stop_analysis/{session_id}")
def stop_analysis(session_id: str):
    """Signal the analysis pipeline to stop, without closing the raw stream."""
    ev = stop_events.get(session_id)
    if ev:
        ev.set()
    return {"status": "stopped"}


@app.delete("/api/stream/{stream_id}")
def close_stream(stream_id: str):
    """Tear down a raw RTSP stream when the user navigates away."""
    cam = rtsp_streams.pop(stream_id, None)
    rtsp_dims.pop(stream_id, None)
    if cam:
        cam.release()
    return {"status": "closed"}


@app.get("/api/alerts/{session_id}")
def get_alerts(session_id: str):
    return {"alerts": session_alerts.get(session_id, [])}


# ── Guardian endpoints ────────────────────────────────────────────────────────
# These are fully decoupled from the existing pipeline endpoints.
# They use the same session/alert infrastructure but are namespaced under /api/guardian/.

@app.post("/api/guardian/scan")
async def guardian_scan(request: GuardianScanRequest):
    """
    Run a one-shot YOLO detection on a single frame and return bounding boxes.
    The frontend uses these to draw the Phase-1 selection overlay.
    """
    import numpy as _np
    from ultralytics import YOLO as _YOLO

    # Resolve the video source
    if request.stream_id and request.stream_id in rtsp_streams:
        cam = rtsp_streams[request.stream_id]
        ret, frame = cam.read()
        if not ret or frame is None:
            raise HTTPException(status_code=503, detail="Could not read frame from stream.")
        width, height = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    elif request.filename:
        file_path = os.path.join(UPLOAD_DIR, request.filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Video file not found.")
        cap = cv2.VideoCapture(file_path)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            raise HTTPException(status_code=400, detail="Could not read video frame.")
        height, width = frame.shape[:2]
    else:
        raise HTTPException(status_code=400, detail="Provide stream_id or filename.")

    # Save a preview still for the frontend to display
    scan_preview_id = str(uuid.uuid4())
    preview_fn      = f"guardian_scan_{scan_preview_id}.jpg"
    preview_path    = os.path.join(PREVIEW_DIR, preview_fn)
    cv2.imwrite(preview_path, frame)

    # Run YOLO inference in a thread pool to avoid blocking the event loop
    loop = asyncio.get_running_loop()
    _model = _YOLO("yolov8n.pt")

    def _infer():
        return _model(frame, verbose=False)[0]

    results = await loop.run_in_executor(None, _infer)

    detections = []
    if results.boxes is not None:
        for i, box in enumerate(results.boxes):
            cls_id  = int(box.cls[0])
            conf    = float(box.conf[0])
            label   = _model.names[cls_id]
            xywh    = box.xywh[0].cpu().numpy()
            cx, cy, bw, bh = float(xywh[0]), float(xywh[1]), float(xywh[2]), float(xywh[3])
            # Normalise to [0,1] range, xywh format (top-left x,y + w,h)
            nx = (cx - bw / 2) / width
            ny = (cy - bh / 2) / height
            nw = bw / width
            nh = bh / height
            detections.append({
                "id":             f"yolo-{i}-{str(uuid.uuid4())[:8]}",
                "label":          label,
                "class_id":       cls_id,
                "confidence":     round(conf, 2),
                "bbox_normalized": [round(nx, 4), round(ny, 4), round(nw, 4), round(nh, 4)],
            })

    return {
        "preview_url": f"/storage/previews/{preview_fn}",
        "width":       width,
        "height":      height,
        "detections":  detections,
    }


@app.post("/api/guardian/start")
async def guardian_start(request: GuardianStartRequest):
    """
    Start a guardian analysis session.
    Returns a session_id compatible with /api/stream/{session_id} and /api/alerts/{session_id}.
    """
    if not request.watched_objects:
        raise HTTPException(status_code=400, detail="No watched_objects provided.")

    # Serialise watched objects for the pipeline config
    watched_list = [
        {
            "id":              wo.id,
            "type":            wo.type,
            "label":           wo.label,
            "class_id":        wo.class_id,
            "bbox_normalized": wo.bbox_normalized,
        }
        for wo in request.watched_objects
    ]

    # Build a ProcessRequest-compatible record so the existing /api/stream endpoint works
    session_id = str(uuid.uuid4())

    # Determine source
    filename  = request.filename or ""
    stream_id = request.stream_id or None

    # We store a ProcessRequest-like object (dict is fine — stream_video reads .pipeline_name etc.)
    from types import SimpleNamespace
    fake_req = SimpleNamespace(
        video_id      = request.video_id or session_id,
        filename      = filename,
        pipeline_name = "room_guardian",
        roi_normalized= [],
        config        = {"watched_objects": watched_list},
        stream_id     = stream_id,
    )

    active_sessions[session_id] = fake_req
    session_alerts[session_id]  = []
    stop_events[session_id]     = threading.Event()

    return {"session_id": session_id}


# Mount static root last
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
