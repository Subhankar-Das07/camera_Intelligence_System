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
from core.site_admin_api import router as site_admin_router
from core.site_admin_runtime import start_runtime
from pipelines.vehicle_recognition import VehicleRecognitionPipeline
from pipelines.vehicle_recognition.database import VehicleDatabase
from ultralytics import FastSAM as _FastSAM
from ultralytics import YOLO as _YOLO
from pipelines.gate_analytics_pipeline import GateAnalyticsPipeline

# Register the vehicle recognition pipeline into the shared singleton registry
registry.register("vehicle_recognition", VehicleRecognitionPipeline)
registry.register("gate_analytics", GateAnalyticsPipeline)

# ── Guardian scan models — loaded ONCE at startup, never reloaded per request ──
# This eliminates the 2-5s cold-load that was happening on every scan click.
_guardian_fastsam: Optional[_FastSAM] = None
_guardian_yolo:    Optional[_YOLO]    = None

def _get_guardian_models():
    """Lazy-load guardian scan models as singletons."""
    global _guardian_fastsam, _guardian_yolo
    if _guardian_fastsam is None:
        _guardian_fastsam = _FastSAM("FastSAM-s.pt")
    if _guardian_yolo is None:
        # Use the existing YOLO model already present in the image for class labeling
        _guardian_yolo = _YOLO("yolov8n-pose.pt")
    return _guardian_fastsam, _guardian_yolo


def _guardian_iou(boxA, boxB) -> float:
    """IoU between two [x,y,w,h] boxes."""
    ax1, ay1, ax2, ay2 = boxA[0], boxA[1], boxA[0]+boxA[2], boxA[1]+boxA[3]
    bx1, by1, bx2, by2 = boxB[0], boxB[1], boxB[0]+boxB[2], boxB[1]+boxB[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0: return 0.0
    union = boxA[2]*boxA[3] + boxB[2]*boxB[3] - inter
    return inter / union if union > 0 else 0.0

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
app.mount("/vehicle_images", StaticFiles(directory="storage/vehicle_images"), name="vehicle_images")
app.include_router(mobile_router)
app.include_router(site_admin_router)

@app.on_event("startup")
async def _on_startup():
    # Fail fast if Redis is unreachable
    get_redis().ping()
    loop = asyncio.get_running_loop()
    start_mobile_worker(loop)
    try:
        registry.get_pipeline("vehicle_recognition").initialize()
    except Exception:
        pass
    try:
        registry.get_pipeline("gate_analytics").initialize()
    except Exception:
        pass
    start_runtime()

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


def _vr_alerts_key(session_id: str) -> str:
    return f"vr:alerts:{session_id}"


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
    r.delete(_vr_det_key(session_id), _vr_plates_key(session_id), _vr_alerts_key(session_id))
    r.delete(f"vr:alerted_plates:{session_id}")


def _vr_add_detection(session_id: str, plate: str, total_visits: int, status: str = "Unknown", vehicle_type: str = "Car", image_path: str = None) -> bool:
    """Return True if this plate is newly recorded for the session."""
    r = get_redis()
    added = r.sadd(_vr_plates_key(session_id), plate)
    if not added:
        return False
    r.rpush(
        _vr_det_key(session_id),
        json.dumps({
            "plate": plate,
            "total_visits": total_visits,
            "status": status,
            "vehicle_type": vehicle_type,
            "image_path": image_path
        }).encode(),
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

# Attendance Session state
attendance_session = {
    "active": False,
    "start_time": None,
    "present_ids": set()
}

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

# ── Guardian Models ──────────────────────────────────────────────────────────
class GuardianScanRequest(BaseModel):
    stream_id: Optional[str] = None
    filename: Optional[str] = None

class WatchedObject(BaseModel):
    id: str
    type: str
    label: str
    class_id: Optional[int] = None
    bbox_normalized: List[float]

class GuardianStartRequest(BaseModel):
    stream_id: Optional[str] = None
    video_id: Optional[str] = None
    filename: Optional[str] = None
    watched_objects: List[WatchedObject]

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
            if alert_event.get("type") == "face_recognised" and config.get("mode") == "attendance":
                if attendance_session["active"]:
                    attendance_session["present_ids"].add(alert_event["person_id"])
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
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Unsupported file format.")
    video_id  = str(uuid.uuid4())
    ext       = os.path.splitext(file.filename)[1].lower()
    filename  = f"{video_id}{ext}"
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    cap = cv2.VideoCapture(file_path)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=400, detail="Could not read video file due to codec failure or corrupted stream.")
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

    try:
        # Call directly on the main event loop thread to avoid COM/threading issues with DSHOW
        cam, width, height = _open()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

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

def _get_identity_manager(mode: str = "visitor"):
    pipeline = _get_fr_pipeline()
    if mode == "attendance":
        return pipeline.get_attendance_manager()
    return pipeline.get_visitor_manager()


@app.get("/api/faces/status")
async def face_status(mode: str = "visitor"):
    """Return stats about the face identity database."""
    im = _get_identity_manager(mode)
    return im.get_stats()


@app.get("/api/faces/identities")
async def list_identities(mode: str = "visitor"):
    """Return all registered identities with metadata and thumbnail URLs."""
    im = _get_identity_manager(mode)
    identities = im.get_all_identities()
    result = []
    for pid, meta in identities.items():
        entry = {"person_id": pid, **meta}
        entry["thumbnail_url"] = im.get_face_thumbnail_url(pid, mode)
        result.append(entry)
    result.sort(key=lambda x: x.get("created_at", 0))
    return {"identities": result}


@app.post("/api/faces/register")
async def register_face(
    label: Optional[str] = None,
    mode: str = "visitor",
    file: UploadFile = File(...),
):
    """Register a new known person from an uploaded face image."""
    import numpy as np
    pipeline = _get_fr_pipeline()
    im = _get_identity_manager(mode)
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
async def snapshot_and_register(stream_id: str, label: Optional[str] = None, mode: str = "visitor"):
    """Grab current frame from live RTSP stream, detect face, register it."""
    import numpy as np
    cam = rtsp_streams.get(stream_id)
    if cam is None:
        raise HTTPException(status_code=404, detail="Stream not found.")

    ret, frame = cam.read()
    if not ret or frame is None:
        raise HTTPException(status_code=503, detail="Could not read frame from stream.")

    pipeline = _get_fr_pipeline()
    im = _get_identity_manager(mode)
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
async def rename_identity(person_id: str, body: FaceRenameRequest, mode: str = "visitor"):
    """Rename an existing identity."""
    im = _get_identity_manager(mode)
    success = im.rename_identity(person_id, body.new_label)
    if not success:
        raise HTTPException(status_code=404, detail=f"Person '{person_id}' not found.")
    return {"person_id": person_id, "new_label": body.new_label, "status": "renamed"}


@app.delete("/api/faces/identity/{person_id}")
async def delete_identity(person_id: str, mode: str = "visitor"):
    """Delete a person from the database entirely."""
    im = _get_identity_manager(mode)
    success = im.delete_identity(person_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Person '{person_id}' not found.")
    return {"person_id": person_id, "status": "deleted"}

# ── Attendance Session Management ────────────────────────────────────────────

@app.post("/api/attendance/start")
def start_attendance_session():
    attendance_session["active"] = True
    attendance_session["start_time"] = time.time()
    attendance_session["present_ids"] = set()
    return {"status": "started"}

@app.post("/api/attendance/stop")
def stop_attendance_session():
    attendance_session["active"] = False
    return {"status": "stopped", "present_ids": list(attendance_session["present_ids"])}

@app.get("/api/attendance/status")
def get_attendance_status():
    return {
        "active": attendance_session["active"],
        "present_ids": list(attendance_session["present_ids"])
    }


@app.get("/api/faces/image/{person_id}/{index}")
async def face_image(person_id: str, index: int, mode: str = "visitor"):
    """Serve a face crop JPEG stored in Redis."""
    im = _get_identity_manager(mode)
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


_ATT_DATA_DIR = os.path.join("face_recognition", "attendance_data")
os.makedirs(_ATT_DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(_ATT_DATA_DIR, "persons"), exist_ok=True)
app.mount("/attendance_data", StaticFiles(directory=_ATT_DATA_DIR), name="attendance_data")

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

                    # Fetch latest status and vehicle_type from DB for this plate
                    db_status       = "Unknown"
                    db_vehicle_type = det.get("vehicle_type", "Car")
                    image_path      = det.get("image_path")
                    try:
                        vehicle_row = pipeline.db.get_vehicle_stats(plate)
                        if vehicle_row:
                            db_status       = vehicle_row.get("status", "Unknown")
                            db_vehicle_type = vehicle_row.get("vehicle_type", db_vehicle_type)
                    except Exception:
                        pass
                    _vr_add_detection(session_id, plate, det.get("total_visits", 1), db_status, db_vehicle_type, image_path)

        if metadata and metadata.get("alerts"):
            r = get_redis()
            for alert in metadata["alerts"]:
                # Use Redis Sets to avoid duplicate alerts for the same plate
                added = r.sadd(f"vr:alerted_plates:{session_id}", alert["plate"])
                if added:
                    r.rpush(_vr_alerts_key(session_id), json.dumps(alert).encode())

        if frame is not None:
            ok, buf = cv2.imencode('.jpg', frame)
            if ok:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                       + buf.tobytes() + b'\r\n')


@app.get("/api/vr_detections/{session_id}")
def get_vr_detections(session_id: str):
    """Return the list of confirmed unique plates and active loitering alerts."""
    r = get_redis()

    # Get detections
    det_items = r.lrange(_vr_det_key(session_id), 0, -1)
    detections = [json.loads(redis_str(x)) for x in det_items if x]

    # Get VR loitering alerts
    alert_items = r.lrange(_vr_alerts_key(session_id), 0, -1)
    alerts = [json.loads(redis_str(x)) for x in alert_items if x]

    return {"detections": detections, "alerts": alerts}

@app.get("/api/vehicles")
def get_all_vehicles():
    """Return all vehicles from the database for the Admin Portal."""
    try:
        pipeline = registry.get_pipeline("vehicle_recognition")
        vehicles = pipeline.db.get_all_vehicles()
        return {"vehicles": vehicles}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/register_vehicle/{plate}")
def register_vehicle(plate: str):
    """Mark a vehicle plate as Known in the database."""
    try:
        pipeline = registry.get_pipeline("vehicle_recognition")
        pipeline.db.register_vehicle(plate)
        return {"status": "ok", "plate": plate}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/unregister_vehicle/{plate}")
def unregister_vehicle(plate: str):
    """Revert a vehicle plate back to Unknown in the database."""
    try:
        pipeline = registry.get_pipeline("vehicle_recognition")
        pipeline.db.unregister_vehicle(plate)
        return {"status": "ok", "plate": plate}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Guardian endpoints ────────────────────────────────────────────────────────
# These are fully decoupled from the existing pipeline endpoints.
# They use the same session/alert infrastructure but are namespaced under /api/guardian/.

@app.post("/api/guardian/scan")
async def guardian_scan(request: GuardianScanRequest):
    """
    Run a one-shot scan on a single frame and return bounding boxes.
    - FastSAM detects all objects (class-agnostic segments).
    - YOLOv8 provides class labels (COCO names) matched by IoU.
    - Only detections >= 60% confidence are returned.
    - Models are singletons — no cold-load on each click.
    """
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

    # Save a preview still for the frontend
    scan_preview_id = str(uuid.uuid4())
    preview_fn      = f"guardian_scan_{scan_preview_id}.jpg"
    preview_path    = os.path.join(PREVIEW_DIR, preview_fn)
    cv2.imwrite(preview_path, frame)

    # Run both models in thread pool (non-blocking) using singletons
    loop = asyncio.get_running_loop()
    fastsam_model, yolo_model = _get_guardian_models()
    CONF_GATE = 0.60

    def _infer():
        # 1. FastSAM: class-agnostic segments
        sam_results  = fastsam_model(frame, conf=CONF_GATE, verbose=False)[0]
        # 2. YOLOv8: class-aware detections for labeling
        yolo_results = yolo_model(frame, conf=0.35, verbose=False)[0]
        return sam_results, yolo_results

    sam_results, yolo_results = await loop.run_in_executor(None, _infer)

    # Build YOLO label lookup: list of (bbox_xywh_abs, class_name)
    yolo_boxes = []
    if yolo_results.boxes is not None:
        names = yolo_results.names or {}
        for box in yolo_results.boxes:
            cls_id = int(box.cls[0])
            xywh   = box.xywh[0].cpu().numpy()
            cx, cy, bw, bh = float(xywh[0]), float(xywh[1]), float(xywh[2]), float(xywh[3])
            yolo_boxes.append({
                "bbox":  [int(cx - bw/2), int(cy - bh/2), int(bw), int(bh)],
                "label": names.get(cls_id, f"cls_{cls_id}"),
            })

    def _match_label(sam_bbox_abs):
        """Find best-matching YOLO class label for a FastSAM box."""
        best_iou, best_label = 0.0, "Object"
        for yb in yolo_boxes:
            iou = _guardian_iou(sam_bbox_abs, yb["bbox"])
            if iou > best_iou:
                best_iou  = iou
                best_label = yb["label"]
        # Only use label if IoU is convincing (>=0.3)
        return best_label if best_iou >= 0.30 else "Object"

    detections = []
    if sam_results.boxes is not None:
        for i, box in enumerate(sam_results.boxes):
            conf = float(box.conf[0])
            if conf < CONF_GATE:
                continue   # strict confidence gate
            xywh = box.xywh[0].cpu().numpy()
            cx, cy, bw, bh = float(xywh[0]), float(xywh[1]), float(xywh[2]), float(xywh[3])
            bx_abs = int(cx - bw/2)
            by_abs = int(cy - bh/2)
            nx  = bx_abs / width
            ny  = by_abs / height
            nw  = bw / width
            nh  = bh / height
            label = _match_label([bx_abs, by_abs, int(bw), int(bh)])
            detections.append({
                "id":              f"sam-{i}-{str(uuid.uuid4())[:8]}",
                "label":           label,
                "class_id":        None,
                "confidence":      round(conf, 2),
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
    _clear_session_alerts(session_id)
    stop_events[session_id]     = threading.Event()

    return {"session_id": session_id}


# Mount static root last
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

