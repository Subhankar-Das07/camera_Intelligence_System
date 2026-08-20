import os
import uuid
import socket
import urllib.parse
import cv2
import threading
import time
import asyncio

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
from typing import List, Tuple, Dict, Any, Optional
import json

from core.registry import registry
from core.redis_client import get_redis, redis_str
from core.video_source import get_video_source, ThreadedCamera
from core.mobile_ws import router as mobile_router, start_mobile_worker
from pipelines.vehicle_recognition import VehicleRecognitionPipeline
from pipelines.vehicle_recognition.database import VehicleDatabase

# Register the vehicle recognition pipeline into the shared singleton registry
registry.register("vehicle_recognition", VehicleRecognitionPipeline)

app = FastAPI(title="Video Analytics Testing Platform")

# ── Storage directories (uploads / alerts only — DBs are Redis) ───────────────
STORAGE_DIR = "storage"
UPLOAD_DIR  = os.path.join(STORAGE_DIR, "uploads")
PREVIEW_DIR = os.path.join(STORAGE_DIR, "previews")
OUTPUT_DIR  = os.path.join(STORAGE_DIR, "outputs")
ALERTS_DIR  = os.path.join(STORAGE_DIR, "alerts")

for d in [UPLOAD_DIR, PREVIEW_DIR, OUTPUT_DIR, ALERTS_DIR]:
    os.makedirs(d, exist_ok=True)

app.mount("/storage", StaticFiles(directory=STORAGE_DIR), name="storage")
app.include_router(mobile_router)

@app.on_event("startup")
async def _on_startup():
    # Fail fast if Redis is unreachable
    get_redis().ping()
    loop = asyncio.get_running_loop()
    start_mobile_worker(loop)

# ── Live stream handles stay in-process; DB/session data is Redis ─────────────
rtsp_streams: Dict[str, ThreadedCamera] = {}
rtsp_dims:    Dict[str, Tuple[int, int]] = {}
active_sessions: Dict[str, Any] = {}
stop_events:     Dict[str, threading.Event] = {}


def _alerts_key(session_id: str) -> str:
    return f"sess:alerts:{session_id}"


def _vr_det_key(session_id: str) -> str:
    return f"vr:det:{session_id}"


def _vr_plates_key(session_id: str) -> str:
    return f"vr:detplates:{session_id}"


def _append_session_alert(session_id: str, alert_event: Dict[str, Any]) -> None:
    get_redis().rpush(_alerts_key(session_id), json.dumps(alert_event).encode())


def _get_session_alerts(session_id: str) -> List[Dict[str, Any]]:
    items = get_redis().lrange(_alerts_key(session_id), 0, -1)
    out: List[Dict[str, Any]] = []
    for raw in items:
        try:
            out.append(json.loads(redis_str(raw)))
        except json.JSONDecodeError:
            continue
    return out


def _clear_session_alerts(session_id: str) -> None:
    get_redis().delete(_alerts_key(session_id))


def _reset_vr_detections(session_id: str) -> None:
    r = get_redis()
    r.delete(_vr_det_key(session_id), _vr_plates_key(session_id))


def _vr_add_detection(session_id: str, plate: str, total_visits: int) -> bool:
    """Return True if this plate is newly recorded for the session."""
    r = get_redis()
    added = r.sadd(_vr_plates_key(session_id), plate)
    if not added:
        return False
    r.rpush(
        _vr_det_key(session_id),
        json.dumps({"plate": plate, "total_visits": total_visits}).encode(),
    )
    return True


def _get_vr_detections(session_id: str) -> List[Dict[str, Any]]:
    items = get_redis().lrange(_vr_det_key(session_id), 0, -1)
    out: List[Dict[str, Any]] = []
    for raw in items:
        try:
            out.append(json.loads(redis_str(raw)))
        except json.JSONDecodeError:
            continue
    return out

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

def _mjpeg_generator_analysis(session_id: str, pipeline, input_path, roi_normalized, config):
    """Yield AI-annotated MJPEG frames and collect alert events."""
    stop_ev = stop_events.get(session_id)
    generator = pipeline.run_on_video(
        input_path=input_path,
        output_dir=ALERTS_DIR,
        roi_normalized=roi_normalized,
        config=config
    )
    for frame, alert_event in generator:
        if stop_ev and stop_ev.is_set():
            break
        if alert_event:
            _append_session_alert(session_id, alert_event)
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
    if not file.filename.endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(status_code=400, detail="Unsupported file format.")
    video_id  = str(uuid.uuid4())
    ext       = os.path.splitext(file.filename)[1]
    filename  = f"{video_id}{ext}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

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
    _clear_session_alerts(session_id)
    stop_events[session_id] = threading.Event()
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
        input_path = rtsp_streams[req.stream_id]
    else:
        input_path = os.path.join(UPLOAD_DIR, req.filename)

    # Vehicle recognition uses a specialised generator that harvests plate metadata
    if req.pipeline_name == "vehicle_recognition":
        generator_fn = _vr_mjpeg_generator(
            session_id, pipeline, input_path, req.roi_normalized, req.config
        )
    else:
        generator_fn = _mjpeg_generator_analysis(
            session_id, pipeline, input_path, req.roi_normalized, req.config
        )

    return StreamingResponse(
        generator_fn,
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


@app.post("/api/connect_webcam")
async def connect_webcam(index: int = 0):
    """
    Open a local webcam by device index and register it as a live stream.
    Returns stream_id that can be passed to start_analysis just like an RTSP stream.
    Used by the Face Recognition dashboard webcam tab.
    """
    def _open():
        cam = ThreadedCamera(index)   # integer index → cv2.VideoCapture(0/1/2...)
        if not cam.isOpened():
            return None, None, None
        cam.start()
        # Wait for first real frame
        for _ in range(30):
            ret, frame = cam.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                return cam, w, h
            time.sleep(0.1)
        # Fallback to reported dimensions
        w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        return cam, w, h

    loop = asyncio.get_running_loop()
    try:
        cam, width, height = await asyncio.wait_for(
            loop.run_in_executor(None, _open), timeout=10.0
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=400, detail="Webcam timed out. Check that no other app is using it.")

    if cam is None:
        raise HTTPException(status_code=400,
            detail=f"Could not open webcam index {index}. Check the device is connected and not in use.")

    stream_id = str(uuid.uuid4())
    rtsp_streams[stream_id] = cam
    rtsp_dims[stream_id] = (width, height)
    return {"stream_id": stream_id, "width": width, "height": height}


@app.get("/api/alerts/{session_id}")
def get_alerts(session_id: str):
    return {"alerts": _get_session_alerts(session_id)}


# ══════════════════════════════════════════════════════════════════════════════
# ── Face Recognition Management APIs (Ayush module — additive) ───────────────
# ══════════════════════════════════════════════════════════════════════════════

from fastapi.responses import JSONResponse
from pydantic import BaseModel as _BaseModel

class FaceRenameRequest(_BaseModel):
    new_label: str


def _get_fr_pipeline():
    """Helper to retrieve the face_recognition pipeline instance."""
    try:
        return registry.get_pipeline("face_recognition")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Face recognition pipeline unavailable: {e}")


@app.get("/api/faces/status")
async def face_status():
    """Return stats about the face identity database."""
    pipeline = _get_fr_pipeline()
    return pipeline.get_identity_manager().get_stats()


@app.get("/api/faces/identities")
async def list_identities():
    """Return all registered identities with metadata and thumbnail URLs."""
    pipeline = _get_fr_pipeline()
    im = pipeline.get_identity_manager()
    identities = im.get_all_identities()
    result = []
    for pid, meta in identities.items():
        entry = {"person_id": pid, **meta}
        entry["thumbnail_url"] = im.get_face_thumbnail_url(pid)
        result.append(entry)
    result.sort(key=lambda x: x.get("created_at", 0))
    return {"identities": result}


@app.post("/api/faces/register")
async def register_face(
    label: Optional[str] = None,
    file: UploadFile = File(...),
):
    """Register a new known person from an uploaded face image."""
    import numpy as np
    pipeline = _get_fr_pipeline()
    im = pipeline.get_identity_manager()
    embedder = pipeline._embedder

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    faces = embedder.detect_and_embed(img)
    quality_faces = [f for f in faces if f.is_quality and f.embedding is not None]
    if not quality_faces:
        raise HTTPException(
            status_code=422,
            detail="No clear face detected. Ensure face is well-lit, front-facing, at least 50x50px."
        )

    best = max(quality_faces, key=lambda f: f.score)
    pid = im.add_identity(
        embeddings=best.embedding.reshape(1, -1),
        label=label or None,
        face_crops=[best.crop],
    )
    meta = im.get_identity(pid)
    return {"person_id": pid, "label": meta["label"], "status": "registered"}


@app.post("/api/faces/snapshot/{stream_id}")
async def snapshot_and_register(stream_id: str, label: Optional[str] = None):
    """Grab current frame from live RTSP stream, detect face, register it."""
    import numpy as np
    cam = rtsp_streams.get(stream_id)
    if cam is None:
        raise HTTPException(status_code=404, detail="Stream not found.")

    ret, frame = cam.read()
    if not ret or frame is None:
        raise HTTPException(status_code=503, detail="Could not read frame from stream.")

    pipeline = _get_fr_pipeline()
    im = pipeline.get_identity_manager()
    embedder = pipeline._embedder

    faces = embedder.detect_and_embed(frame)
    quality_faces = [f for f in faces if f.is_quality and f.embedding is not None]
    if not quality_faces:
        raise HTTPException(status_code=422, detail="No clear face in current frame.")

    best = max(quality_faces, key=lambda f: f.score)
    pid = im.add_identity(
        embeddings=best.embedding.reshape(1, -1),
        label=label or None,
        face_crops=[best.crop],
    )
    meta = im.get_identity(pid)
    return {"person_id": pid, "label": meta["label"], "status": "registered"}


@app.patch("/api/faces/identity/{person_id}")
async def rename_identity(person_id: str, body: FaceRenameRequest):
    """Rename an existing identity."""
    pipeline = _get_fr_pipeline()
    im = pipeline.get_identity_manager()
    success = im.rename_identity(person_id, body.new_label)
    if not success:
        raise HTTPException(status_code=404, detail=f"Person '{person_id}' not found.")
    return {"person_id": person_id, "new_label": body.new_label, "status": "renamed"}


@app.delete("/api/faces/identity/{person_id}")
async def delete_identity(person_id: str):
    """Delete a person from the database entirely."""
    pipeline = _get_fr_pipeline()
    im = pipeline.get_identity_manager()
    success = im.delete_identity(person_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Person '{person_id}' not found.")
    return {"person_id": person_id, "status": "deleted"}


@app.get("/api/faces/image/{person_id}/{index}")
def face_image(person_id: str, index: int = 1):
    """Serve a face crop JPEG stored in Redis."""
    pipeline = _get_fr_pipeline()
    im = pipeline.get_identity_manager()
    data = im.get_face_bytes(person_id, index)
    if not data:
        raise HTTPException(status_code=404, detail="Face image not found.")
    return Response(content=data, media_type="image/jpeg")


_vr_db = VehicleDatabase()


@app.get("/api/vr/image/{kind}/{plate}/{visit}")
def vr_image(kind: str, plate: str, visit: int):
    """Serve vehicle snapshot or plate-crop JPEG stored in Redis."""
    if kind not in ("snap", "crop"):
        raise HTTPException(status_code=400, detail="kind must be snap or crop")
    data = _vr_db.get_image_bytes(kind, plate, visit)
    if not data:
        raise HTTPException(status_code=404, detail="Image not found.")
    return Response(content=data, media_type="image/jpeg")


# Serve face recognition dashboard (separate sub-page)
_FR_STATIC_DIR = os.path.join("static", "face_recognition")
os.makedirs(_FR_STATIC_DIR, exist_ok=True)


def _vr_mjpeg_generator(session_id: str, pipeline, input_path, roi_normalized, config):
    """Annotated MJPEG generator that also captures plate detections for the log."""
    stop_ev = stop_events.get(session_id)
    _reset_vr_detections(session_id)

    generator = pipeline.run_on_video(
        input_path=input_path,
        output_dir=ALERTS_DIR,
        roi_normalized=roi_normalized,
        config=config,
    )
    for frame, metadata in generator:
        if stop_ev and stop_ev.is_set():
            break
        if metadata and metadata.get("detections"):
            for det in metadata["detections"]:
                plate = det.get("plate")
                if plate:
                    _vr_add_detection(session_id, plate, det.get("total_visits", 1))
        if frame is not None:
            ok, buf = cv2.imencode('.jpg', frame)
            if ok:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                       + buf.tobytes() + b'\r\n')


@app.get("/api/vr_detections/{session_id}")
def get_vr_detections(session_id: str):
    """Return the list of confirmed unique plates detected in a VR session."""
    return {"detections": _get_vr_detections(session_id)}


# Mount static root last
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
