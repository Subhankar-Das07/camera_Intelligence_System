"""FastAPI routes for Site Admin. Mounted under /api/site-admin."""

from __future__ import annotations

import csv
import io
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import cv2
from fastapi import APIRouter, File, Header, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from core import site_admin_common as common
from core import site_admin_monitor
from core import site_admin_runtime
from core import site_admin_store as store
from core import whatsapp_adapter
from core.registry import registry
from core.snapshot_camera import fetch_snapshot_jpeg
from core.video_source import ThreadedCamera, get_video_source

router = APIRouter(prefix="/api/site-admin", tags=["site-admin"])

UPLOAD_DIR = os.path.join("storage", "uploads")
PREVIEW_DIR = os.path.join("storage", "previews")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PREVIEW_DIR, exist_ok=True)


class SitePatch(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None
    timezone: Optional[str] = None
    quiet_hours_start: Optional[str] = None
    quiet_hours_end: Optional[str] = None
    setup_complete: Optional[bool] = None
    wizard_step: Optional[int] = None
    admin_pin: Optional[str] = None
    whatsapp_numbers: Optional[List[str]] = None
    go_live: Optional[bool] = None


class CameraIn(BaseModel):
    id: Optional[str] = None
    name: str = "Camera"
    type: str = "rtsp"
    rtsp_url: str = ""
    snapshot_url: str = ""
    http_user: str = ""
    http_password: str = ""
    filename: str = ""
    video_id: str = ""
    preview_url: str = ""
    scenario_tag: str = ""
    enabled: bool = True
    location: str = ""


class RuleIn(BaseModel):
    id: Optional[str] = None
    name: str = "Rule"
    camera_id: str
    scan_type: str = "intrusion"
    roi_normalized: List[List[float]] = Field(default_factory=list)
    gate_config: Dict[str, Any] = Field(default_factory=dict)
    schedule: str = "always"
    severity: str = "high"
    channels: List[str] = Field(default_factory=lambda: ["web"])
    enabled: bool = True
    cooldown_sec: int = 60


class LoginIn(BaseModel):
    pin: str = ""
    role: str = "admin"


class AlertPatch(BaseModel):
    acked: Optional[bool] = None


class AlertDeleteIn(BaseModel):
    ids: List[str] = Field(default_factory=list)


class MuteIn(BaseModel):
    seconds: int = 3600


class MonitorStartIn(BaseModel):
    camera_id: Optional[str] = None
    temp_id: Optional[str] = None
    rule_ids: List[str] = Field(default_factory=list)


class MonitorSeekIn(BaseModel):
    position: Optional[float] = None
    frame: Optional[int] = None


class MonitorPauseIn(BaseModel):
    paused: bool = True


class WhatsAppTestIn(BaseModel):
    number: str = ""
    message: str = "Camera Intelligence test alert"


class SuggestRoiIn(BaseModel):
    max_regions: int = 36
    conf: float = 0.25
    min_area: float = 0.0015


class KnownFaceIn(BaseModel):
    id: Optional[str] = None
    label: str = "Unknown"
    thumb_url: str = ""
    status: str = "candidate"
    person_id: str = ""
    camera_id: str = ""
    note: str = ""


class KnownFacePatch(BaseModel):
    status: Optional[str] = None
    label: Optional[str] = None
    note: Optional[str] = None


class KnownVehicleIn(BaseModel):
    id: Optional[str] = None
    plate: str = "UNKNOWN"
    label: str = ""
    thumb_url: str = ""
    status: str = "candidate"
    camera_id: str = ""
    note: str = ""


class KnownVehiclePatch(BaseModel):
    status: Optional[str] = None
    plate: Optional[str] = None
    label: Optional[str] = None
    note: Optional[str] = None


VEHICLE_STATUS_OK = ("candidate", "approved", "ignored", "risk", "danger")


def _save_preview_frame(frame) -> Dict[str, Any]:
    h, w = frame.shape[:2]
    vid = str(uuid.uuid4())
    preview_fn = f"{vid}.jpg"
    cv2.imwrite(os.path.join(PREVIEW_DIR, preview_fn), frame)
    return {
        "ok": True,
        "width": w,
        "height": h,
        "preview_url": f"/storage/previews/{preview_fn}",
        "health": "online",
    }


def _grab_camera_frame(cam: Dict[str, Any]):
    """Read one frame from a saved camera (file, RTSP, or DVR snapshot)."""
    kind = cam.get("type") or "rtsp"
    if kind == "dvr":
        snapshot_url = (cam.get("snapshot_url") or "").strip()
        if not snapshot_url:
            return None
        frame = fetch_snapshot_jpeg(
            snapshot_url,
            user=(cam.get("http_user") or "").strip(),
            password=(cam.get("http_password") or ""),
        )
        return frame

    source = common.input_for_camera(cam)
    if not source:
        return None
    if kind == "file":
        cap = get_video_source(source)
        if not cap.isOpened():
            return None
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None

    if hasattr(source, "read") and hasattr(source, "isOpened"):
        if not source.isOpened():
            return None
        ret, frame = source.read()
        if hasattr(source, "release"):
            source.release()
        return frame if ret else None

    cam_src = ThreadedCamera(source)
    if not cam_src.isOpened():
        return None
    cam_src.start()
    frame = None
    for _ in range(40):
        ret, fr = cam_src.read()
        if ret and fr is not None:
            frame = fr
            break
        time.sleep(0.1)
    cam_src.release()
    return frame


def _role(x_cis_role: Optional[str]) -> str:
    r = (x_cis_role or "admin").strip().lower()
    return "viewer" if r == "viewer" else "admin"


def _require_admin(x_cis_role: Optional[str]) -> None:
    if _role(x_cis_role) != "admin":
        raise HTTPException(status_code=403, detail="Admin only")


@router.get("/status")
def status():
    site = store.get_site()
    cams = store.list_cameras()
    rules = store.list_rules()
    return {
        "site": site,
        "cameras": len(cams),
        "rules": len(rules),
        "alerts": len(store.list_alerts(50)),
        "go_live": bool(site.get("go_live")),
        "whatsapp_configured": whatsapp_adapter.configured(),
        "scan_types": common.available_scan_types(),
        "scan_catalog": common.SCAN_CATALOG,
        "monitor": site_admin_monitor.get_monitor_status(),
        "runtime": site_admin_runtime.get_runtime_status(),
    }


@router.get("/site")
def get_site():
    return store.get_site()


@router.put("/site")
@router.post("/site")
def put_site(body: SitePatch, x_cis_role: Optional[str] = Header(default="admin")):
    _require_admin(x_cis_role)
    current = store.get_site()
    patch = body.model_dump(exclude_none=True)
    current.update(patch)
    saved = store.save_site(current)
    if saved.get("go_live"):
        site_admin_runtime.start_runtime()
    else:
        site_admin_monitor.clear_preview_cache()
    return saved


@router.post("/login")
def login(body: LoginIn):
    site = store.get_site()
    pin = (site.get("admin_pin") or "").strip()
    role = "admin" if body.role != "viewer" else "viewer"
    if role == "admin" and pin and body.pin.strip() != pin:
        raise HTTPException(status_code=401, detail="Invalid admin PIN")
    return {"ok": True, "role": role}


@router.get("/cameras")
def cameras():
    return {"cameras": store.list_cameras()}


@router.post("/cameras")
def create_camera(body: CameraIn, x_cis_role: Optional[str] = Header(default="admin")):
    _require_admin(x_cis_role)
    cam = body.model_dump()
    cam["health"] = cam.get("health") or "unknown"
    return store.save_camera(cam)


@router.put("/cameras/{cam_id}")
def update_camera(cam_id: str, body: CameraIn, x_cis_role: Optional[str] = Header(default="admin")):
    _require_admin(x_cis_role)
    existing = store.get_camera(cam_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Camera not found")
    data = body.model_dump()
    data["id"] = cam_id
    data["created_at"] = existing.get("created_at")
    data["health"] = existing.get("health", "unknown")
    return store.save_camera(data)


@router.delete("/cameras/{cam_id}")
def remove_camera(cam_id: str, x_cis_role: Optional[str] = Header(default="admin")):
    _require_admin(x_cis_role)
    store.delete_camera(cam_id)
    return {"ok": True}


@router.get("/cameras/{cam_id}/snapshot")
def camera_snapshot(cam_id: str):
    """Lightweight JPEG for go-live preview — no rule pipelines."""
    if not store.get_site().get("go_live"):
        raise HTTPException(status_code=409, detail="Scanning is not active")
    cam = store.get_camera(cam_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    frame = _grab_camera_frame(cam)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not read a frame from this camera")
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode frame")
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@router.get("/cameras/{cam_id}/fresh-preview")
def camera_fresh_preview(cam_id: str):
    """Grab a live frame for Rules ROI setup — updates saved preview_url (no go_live required)."""
    cam = store.get_camera(cam_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    frame = _grab_camera_frame(cam)
    if frame is None:
        raise HTTPException(
            status_code=400,
            detail="Could not read a frame from this camera — check URL, credentials, and network",
        )
    saved = _save_preview_frame(frame)
    preview_url = saved.get("preview_url") or ""
    store.save_camera({
        **cam,
        "preview_url": preview_url,
        "health": "online",
        "last_error": "",
        "last_frame_at": time.time(),
    })
    return {
        "ok": True,
        "preview_url": preview_url,
        "width": saved.get("width"),
        "height": saved.get("height"),
    }


@router.get("/cameras/{cam_id}/preview-frame")
def camera_preview_frame(cam_id: str, apply_rules: bool = False):
    """Go-live hero preview — raw or with rule overlays (no alerts)."""
    if not store.get_site().get("go_live"):
        raise HTTPException(status_code=409, detail="Scanning is not active")
    cam = store.get_camera(cam_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    if apply_rules:
        data = site_admin_monitor.render_preview_frame(cam_id, True)
    else:
        frame = _grab_camera_frame(cam)
        if frame is None:
            raise HTTPException(status_code=400, detail="Could not read a frame from this camera")
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
        data = buf.tobytes() if ok else None
    if not data:
        raise HTTPException(status_code=400, detail="Could not render preview frame")
    return Response(content=data, media_type="image/jpeg")


@router.get("/cameras/{cam_id}/preview-status")
def camera_preview_status(cam_id: str):
    """Go-live hero preview — rule legend, live hits, gate counters (JSON)."""
    if not store.get_site().get("go_live"):
        raise HTTPException(status_code=409, detail="Scanning is not active")
    cam = store.get_camera(cam_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    return site_admin_monitor.get_preview_status(cam_id)


@router.post("/cameras/{cam_id}/suggest-roi")
def suggest_roi(
    cam_id: str,
    body: Optional[SuggestRoiIn] = None,
    x_cis_role: Optional[str] = Header(default="admin"),
):
    """FastSAM one-shot region suggestions for Rules ROI / gate polygon setup."""
    _require_admin(x_cis_role)
    opts = body or SuggestRoiIn()
    cam = store.get_camera(cam_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    frame = _grab_camera_frame(cam)
    if frame is None:
        preview = (cam.get("preview_url") or "").strip()
        if preview.startswith("/storage/previews/"):
            path = os.path.join("storage", "previews", os.path.basename(preview))
            if os.path.isfile(path):
                frame = cv2.imread(path)
        if frame is None:
            raise HTTPException(status_code=400, detail="Could not read a frame from this camera")
    try:
        from core import site_admin_fastsam as fastsam

        regions, w, h = fastsam.suggest_regions(
            frame,
            conf=max(0.05, min(0.9, float(opts.conf))),
            max_regions=max(1, min(80, int(opts.max_regions))),
            min_area=max(0.0002, float(opts.min_area)),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FastSAM suggest failed: {e}") from e
    return {
        "ok": True,
        "camera_id": cam_id,
        "width": w,
        "height": h,
        "regions": regions,
        "count": len(regions),
    }


@router.post("/cameras/test-connection")
def test_connection(body: CameraIn, x_cis_role: Optional[str] = Header(default="admin")):
    _require_admin(x_cis_role)
    kind = (body.type or "rtsp").strip().lower()
    if kind == "dvr":
        snapshot_url = (body.snapshot_url or "").strip()
        if not snapshot_url:
            raise HTTPException(status_code=400, detail="HTTP snapshot URL required")
        frame = fetch_snapshot_jpeg(
            snapshot_url,
            user=(body.http_user or "").strip(),
            password=body.http_password or "",
        )
        if frame is None:
            raise HTTPException(status_code=400, detail="Could not fetch snapshot — check URL and credentials")
        return _save_preview_frame(frame)

    url = (body.rtsp_url or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="RTSP URL required")
    cam = ThreadedCamera(url)
    if not cam.isOpened():
        raise HTTPException(status_code=400, detail="Could not open RTSP URL")
    cam.start()
    frame = None
    for _ in range(40):
        ret, fr = cam.read()
        if ret and fr is not None:
            frame = fr
            break
        time.sleep(0.1)
    cam.release()
    if frame is None:
        raise HTTPException(status_code=400, detail="Opened but no frames received")
    return _save_preview_frame(frame)


@router.post("/cameras/test-rtsp")
def test_rtsp(body: CameraIn, x_cis_role: Optional[str] = Header(default="admin")):
    """Backward-compatible alias for camera connection test."""
    if not (body.type or "").strip():
        body = body.model_copy(update={"type": "rtsp"})
    return test_connection(body, x_cis_role)


@router.post("/cameras/upload")
async def upload_source(
    file: UploadFile = File(...),
    x_cis_role: Optional[str] = Header(default="admin"),
):
    _require_admin(x_cis_role)
    name = file.filename or "clip.mp4"
    if not name.lower().endswith((".mp4", ".avi", ".mov")):
        raise HTTPException(status_code=400, detail="Use mp4, avi, or mov")
    video_id = str(uuid.uuid4())
    ext = os.path.splitext(name)[1]
    filename = f"{video_id}{ext}"
    path = os.path.join(UPLOAD_DIR, filename)
    with open(path, "wb") as f:
        f.write(await file.read())
    cap = cv2.VideoCapture(path)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        os.remove(path)
        raise HTTPException(status_code=400, detail="Could not read video")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    preview_fn = f"{video_id}.jpg"
    cv2.imwrite(os.path.join(PREVIEW_DIR, preview_fn), frame)
    cam = store.save_camera({
        "name": os.path.splitext(name)[0],
        "type": "file",
        "filename": filename,
        "video_id": video_id,
        "preview_url": f"/storage/previews/{preview_fn}",
        "enabled": True,
        "health": "online",
        "width": w,
        "height": h,
    })
    return cam


@router.get("/rules")
def rules():
    return {"rules": store.list_rules()}


@router.post("/rules")
def create_rule(body: RuleIn, x_cis_role: Optional[str] = Header(default="admin")):
    _require_admin(x_cis_role)
    if not common.is_scan_available(body.scan_type):
        raise HTTPException(
            status_code=400,
            detail="This scan type is not available yet. Pick an Available type from the catalog.",
        )
    if not store.get_camera(body.camera_id):
        raise HTTPException(status_code=400, detail="Unknown camera")
    if body.scan_type == "gate_analytics" and not common.gate_config_valid(body.gate_config):
        raise HTTPException(status_code=400, detail="Gate rule needs count line, gate ROI, and near/medium/far zones")
    if body.enabled and common.camera_rule_limit_reached(body.camera_id):
        raise HTTPException(
            status_code=400,
            detail=f"Max {common.MAX_RULES_PER_CAMERA} enabled rules per camera. Disable or remove one first.",
        )
    return store.save_rule(body.model_dump())


@router.put("/rules/{rule_id}")
def update_rule(rule_id: str, body: RuleIn, x_cis_role: Optional[str] = Header(default="admin")):
    _require_admin(x_cis_role)
    existing = store.get_rule(rule_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Rule not found")
    if not common.is_scan_available(body.scan_type):
        raise HTTPException(
            status_code=400,
            detail="This scan type is not available yet. Pick an Available type from the catalog.",
        )
    if body.scan_type == "gate_analytics" and not common.gate_config_valid(body.gate_config):
        raise HTTPException(status_code=400, detail="Gate rule needs count line, gate ROI, and near/medium/far zones")
    if body.enabled and common.camera_rule_limit_reached(body.camera_id, exclude_rule_id=rule_id):
        raise HTTPException(
            status_code=400,
            detail=f"Max {common.MAX_RULES_PER_CAMERA} enabled rules per camera. Disable or remove one first.",
        )
    data = body.model_dump()
    data["id"] = rule_id
    data["created_at"] = existing.get("created_at")
    return store.save_rule(data)


@router.delete("/rules/{rule_id}")
def remove_rule(rule_id: str, x_cis_role: Optional[str] = Header(default="admin")):
    _require_admin(x_cis_role)
    store.delete_rule(rule_id)
    return {"ok": True}


@router.post("/rules/{rule_id}/gate-calibrate")
def gate_calibrate(rule_id: str, x_cis_role: Optional[str] = Header(default="admin")):
    _require_admin(x_cis_role)
    rule = store.get_rule(rule_id)
    if not rule or rule.get("scan_type") != "gate_analytics":
        raise HTTPException(status_code=404, detail="Gate rule not found")
    gate_config = rule.get("gate_config") or {}
    gate_roi = gate_config.get("gate_roi") or []
    if len(gate_roi) < 3:
        raise HTTPException(status_code=400, detail="Draw gate ROI first")
    cam = store.get_camera(rule.get("camera_id") or "")
    if not cam:
        raise HTTPException(status_code=400, detail="Camera not found")
    source = common.input_for_camera(cam)
    if not source:
        raise HTTPException(status_code=400, detail="Camera source unavailable")
    cap = get_video_source(source)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Could not open camera source")
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise HTTPException(status_code=400, detail="No frame from camera")
    h, w = frame.shape[:2]
    from pipelines.gate_analytics_pipeline import _denorm_poly, _roi_edge_score

    pts = _denorm_poly(gate_roi, w, h)
    score = _roi_edge_score(frame, pts)
    store.set_gate_baseline(rule_id, score)
    try:
        pipe = registry.get_pipeline("gate_analytics")
        pipe.reset_baseline(rule_id)
        pipe.set_baseline_score(rule_id, score)
    except Exception:
        pass
    return {"ok": True, "baseline_score": score, "rule_id": rule_id}


@router.get("/alerts")
def alerts(limit: int = 100):
    return {"alerts": store.list_alerts(limit)}


@router.patch("/alerts/{alert_id}")
def patch_alert(alert_id: str, body: AlertPatch):
    item = store.update_alert(alert_id, body.model_dump(exclude_none=True))
    if not item:
        raise HTTPException(status_code=404, detail="Alert not found")
    return item


@router.delete("/alerts/{alert_id}")
def remove_alert(alert_id: str):
    if not store.delete_alert(alert_id):
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"ok": True}


@router.post("/alerts/delete")
def remove_alerts_batch(body: AlertDeleteIn):
    ids = [i for i in (body.ids or []) if i]
    if not ids:
        raise HTTPException(status_code=400, detail="No alert ids provided")
    deleted = 0
    for aid in ids:
        if store.delete_alert(aid):
            deleted += 1
    return {"ok": True, "deleted": deleted}


@router.post("/alerts/{alert_id}/mute-rule")
def mute_from_alert(alert_id: str, body: MuteIn, x_cis_role: Optional[str] = Header(default="admin")):
    _require_admin(x_cis_role)
    alerts = store.list_alerts(200)
    found = next((a for a in alerts if a.get("id") == alert_id), None)
    if not found or not found.get("rule_id"):
        raise HTTPException(status_code=404, detail="Alert or rule not found")
    store.mute_rule(found["rule_id"], body.seconds)
    return {"ok": True, "muted_seconds": body.seconds}


@router.post("/alerts/{alert_id}/share-whatsapp")
def share_whatsapp(alert_id: str):
    alerts = store.list_alerts(200)
    found = next((a for a in alerts if a.get("id") == alert_id), None)
    if not found:
        raise HTTPException(status_code=404, detail="Alert not found")
    site = store.get_site()
    body = found.get("message") or "Camera Intelligence alert"
    if found.get("clip_url"):
        body += f"\n{found['clip_url']}"
    results = whatsapp_adapter.notify_numbers(site.get("whatsapp_numbers") or [], body)
    return {"results": results, "configured": whatsapp_adapter.configured()}


@router.get("/reports")
def reports(hours: float = 24.0):
    until = time.time()
    since = until - max(0.1, hours) * 3600
    return store.report_summary(since, until)


@router.get("/reports/csv")
def reports_csv(hours: float = 24.0):
    until = time.time()
    since = until - max(0.1, hours) * 3600
    data = store.report_summary(since, until)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["time", "camera", "scan_type", "severity", "message", "clip_url", "acked"])
    for a in data.get("alerts") or []:
        w.writerow([
            a.get("created_at"),
            a.get("camera_name"),
            a.get("scan_type"),
            a.get("severity"),
            a.get("message"),
            a.get("clip_url"),
            a.get("acked"),
        ])
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=site-admin-report.csv"},
    )


@router.get("/reports/gate")
def reports_gate(
    hours: float = 24.0,
    rule_id: Optional[str] = None,
    period: str = "hours",
    since: Optional[float] = None,
    until: Optional[float] = None,
):
    period = (period or "hours").lower()
    if period not in ("hours", "daily", "weekly", "monthly"):
        period = "hours"
    if since is not None and until is not None:
        return store.gate_report_by_period(
            period=period,
            rule_id=rule_id,
            since_ts=since,
            until_ts=until,
        )
    return store.gate_report_by_period(period=period, rule_id=rule_id, hours=hours)


@router.get("/reports/gate/csv")
def reports_gate_csv(
    hours: float = 24.0,
    rule_id: Optional[str] = None,
    period: str = "hours",
    since: Optional[float] = None,
    until: Optional[float] = None,
):
    period = (period or "hours").lower()
    if period not in ("hours", "daily", "weekly", "monthly"):
        period = "hours"
    if since is not None and until is not None:
        data = store.gate_report_by_period(
            period=period,
            rule_id=rule_id,
            since_ts=since,
            until_ts=until,
        )
    else:
        data = store.gate_report_by_period(period=period, rule_id=rule_id, hours=hours)
    buf = io.StringIO()
    w = csv.writer(buf)
    if period in ("daily", "weekly", "monthly") and data.get("daily"):
        w.writerow([
            "date",
            "rule_id",
            "footfall",
            "persons_in",
            "persons_out",
            "cars_in",
            "cars_out",
            "vehicle_crossings",
            "gate_opens",
            "gate_closes",
            "bikes_in",
            "bikes_out",
            "near_events",
            "medium_events",
            "far_events",
        ])
        for row in data.get("daily") or []:
            t = row.get("counters") or {}
            w.writerow([
                row.get("date"),
                row.get("rule_id"),
                t.get("footfall", 0),
                t.get("persons_in", 0),
                t.get("persons_out", 0),
                t.get("cars_in", 0),
                t.get("cars_out", 0),
                t.get("vehicle_crossings", 0),
                t.get("gate_opens", 0),
                t.get("gate_closes", 0),
                t.get("bikes_in", 0),
                t.get("bikes_out", 0),
                t.get("near_events", 0),
                t.get("medium_events", 0),
                t.get("far_events", 0),
            ])
    else:
        w.writerow([
            "rule_id",
            "rule_name",
            "footfall",
            "gate_opens",
            "gate_closes",
            "persons_in",
            "persons_out",
            "cars_in",
            "cars_out",
            "vehicle_crossings",
            "bikes_in",
            "bikes_out",
            "near_events",
            "medium_events",
            "far_events",
        ])
        for row in data.get("rules") or []:
            t = row.get("totals") or {}
            w.writerow([
                row.get("rule_id"),
                row.get("rule_name"),
                t.get("footfall", 0),
                t.get("gate_opens", 0),
                t.get("gate_closes", 0),
                t.get("persons_in", 0),
                t.get("persons_out", 0),
                t.get("cars_in", 0),
                t.get("cars_out", 0),
                t.get("vehicle_crossings", 0),
                t.get("bikes_in", 0),
                t.get("bikes_out", 0),
                t.get("near_events", 0),
                t.get("medium_events", 0),
                t.get("far_events", 0),
            ])
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=gate-report.csv"},
    )


@router.get("/known-faces")
def known_faces(status: Optional[str] = None):
    return {"faces": store.list_known_faces(200, status=status)}


@router.post("/known-faces")
def create_known_face(body: KnownFaceIn, x_cis_role: Optional[str] = Header(default="admin")):
    _require_admin(x_cis_role)
    data = body.model_dump(exclude_none=True)
    if data.get("status") not in ("candidate", "approved", "ignored"):
        data["status"] = "candidate"
    return store.save_known_face(data)


@router.patch("/known-faces/{face_id}")
def patch_known_face(face_id: str, body: KnownFacePatch, x_cis_role: Optional[str] = Header(default="admin")):
    _require_admin(x_cis_role)
    patch = body.model_dump(exclude_none=True)
    if "status" in patch and patch["status"] not in ("candidate", "approved", "ignored"):
        raise HTTPException(status_code=400, detail="status must be candidate, approved, or ignored")
    item = store.patch_known_face(face_id, patch)
    if not item:
        raise HTTPException(status_code=404, detail="Face not found")
    return item


@router.get("/known-vehicles")
def known_vehicles(status: Optional[str] = None):
    return {"vehicles": store.list_known_vehicles(200, status=status)}


@router.post("/known-vehicles")
def create_known_vehicle(body: KnownVehicleIn, x_cis_role: Optional[str] = Header(default="admin")):
    _require_admin(x_cis_role)
    data = body.model_dump(exclude_none=True)
    if data.get("status") not in VEHICLE_STATUS_OK:
        data["status"] = "candidate"
    return store.save_known_vehicle(data)


@router.patch("/known-vehicles/{vehicle_id}")
def patch_known_vehicle(
    vehicle_id: str,
    body: KnownVehiclePatch,
    x_cis_role: Optional[str] = Header(default="admin"),
):
    _require_admin(x_cis_role)
    patch = body.model_dump(exclude_none=True)
    if "status" in patch and patch["status"] not in VEHICLE_STATUS_OK:
        raise HTTPException(
            status_code=400,
            detail="status must be candidate, approved, ignored, risk, or danger",
        )
    item = store.patch_known_vehicle(vehicle_id, patch)
    if not item:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    return item


@router.delete("/known-vehicles/{vehicle_id}")
def remove_known_vehicle(vehicle_id: str, x_cis_role: Optional[str] = Header(default="admin")):
    _require_admin(x_cis_role)
    if not store.delete_known_vehicle(vehicle_id):
        raise HTTPException(status_code=404, detail="Vehicle not found")
    return {"ok": True}


@router.post("/whatsapp/test")
def whatsapp_test(body: WhatsAppTestIn, x_cis_role: Optional[str] = Header(default="admin")):
    _require_admin(x_cis_role)
    numbers = [body.number] if body.number.strip() else store.get_site().get("whatsapp_numbers") or []
    return {
        "configured": whatsapp_adapter.configured(),
        "results": whatsapp_adapter.notify_numbers(numbers, body.message),
    }


@router.get("/monitor/status")
def monitor_status():
    return site_admin_monitor.get_monitor_status()


@router.post("/monitor/upload-temp")
async def monitor_upload_temp(
    file: UploadFile = File(...),
    x_cis_role: Optional[str] = Header(default="admin"),
):
    _require_admin(x_cis_role)
    name = file.filename or "clip.mp4"
    try:
        return site_admin_monitor.save_temp_upload(name, await file.read())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/monitor/start")
def monitor_start(body: MonitorStartIn, x_cis_role: Optional[str] = Header(default="admin")):
    _require_admin(x_cis_role)
    try:
        return site_admin_monitor.start_monitor(
            camera_id=body.camera_id,
            temp_id=body.temp_id,
            rule_ids=body.rule_ids,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/monitor/stream/{session_id}")
def monitor_stream(session_id: str):
    if not site_admin_monitor.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Monitor session not found")
    return StreamingResponse(
        site_admin_monitor.mjpeg_generator(session_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/monitor/frame/{session_id}")
def monitor_frame(session_id: str):
    """Single JPEG frame — used for DVR HTTP snapshot poll preview."""
    if not site_admin_monitor.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Monitor session not found")
    data = site_admin_monitor.capture_frame_jpeg(session_id)
    if not data:
        raise HTTPException(status_code=400, detail="No frame available")
    return Response(content=data, media_type="image/jpeg")


@router.post("/monitor/stop/{session_id}")
def monitor_stop(session_id: str, x_cis_role: Optional[str] = Header(default="admin")):
    _require_admin(x_cis_role)
    site_admin_monitor.stop_monitor(session_id)
    return {"ok": True}


@router.post("/monitor/seek/{session_id}")
def monitor_seek(session_id: str, body: MonitorSeekIn, x_cis_role: Optional[str] = Header(default="admin")):
    _require_admin(x_cis_role)
    try:
        status = site_admin_monitor.seek_monitor(session_id, position=body.position, frame=body.frame)
        return {"ok": True, "session_id": session_id, **status}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/monitor/pause/{session_id}")
def monitor_pause(session_id: str, body: MonitorPauseIn, x_cis_role: Optional[str] = Header(default="admin")):
    _require_admin(x_cis_role)
    try:
        status = site_admin_monitor.set_monitor_paused(session_id, body.paused)
        return {"ok": True, "session_id": session_id, **status}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
