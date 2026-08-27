"""Live Site Admin monitor — one camera, all rules, MJPEG stream."""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np
from shapely.geometry import Point, Polygon

from core import site_admin_common as common
from core import site_admin_scan as scan
from core import site_admin_store as store
from core.registry import registry
from core.video_source import ThreadedCamera, get_video_source

log = logging.getLogger("site_admin.monitor")

MONITOR_TEMP_DIR = os.path.join("storage", "monitor_temp")
MONITOR_TEMP_PREVIEW_DIR = os.path.join("storage", "monitor_temp", "previews")
os.makedirs(MONITOR_TEMP_DIR, exist_ok=True)
os.makedirs(MONITOR_TEMP_PREVIEW_DIR, exist_ok=True)

MAX_SESSION_EVENTS = 50
MAX_PREVIEW_EVENTS = 20
# Heavy pipelines block the go-live preview lock for minutes; run them in workers only.
PREVIEW_LIGHT_SCAN_TYPES = frozenset({"intrusion", "danger_zone", "gate_analytics", "loitering"})
MONITOR_RULE_WORKERS = max(1, min(3, int(os.environ.get("SITE_ADMIN_MONITOR_RULE_WORKERS", "3"))))

RULE_COLORS = [
    (34, 211, 238),
    (251, 191, 36),
    (74, 222, 128),
    (244, 114, 182),
    (129, 140, 248),
    (248, 113, 113),
]

_sessions: Dict[str, Dict[str, Any]] = {}
_stop_events: Dict[str, threading.Event] = {}
_lock = threading.Lock()
_preview_cache: Dict[str, Dict[str, Any]] = {}
_preview_lock = threading.Lock()
PREVIEW_CACHE_TTL = 90.0
_pose_model = None


class FrameFeed:
    """Capture-like source that serves the frame set by the monitor loop."""

    def __init__(self, width: int = 640, height: int = 480, fps: float = 30.0):
        self.frame: Optional[np.ndarray] = None
        self._w = width
        self._h = height
        self._fps = fps
        self._opened = True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.frame is None:
            return False, None
        return True, self.frame

    def isOpened(self) -> bool:
        return self._opened and self.frame is not None

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._w)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._h)
        if prop_id == cv2.CAP_PROP_FPS:
            return float(self._fps)
        return 0.0

    def release(self) -> None:
        self._opened = False


def _rule_color(rule_id: str) -> Tuple[int, int, int]:
    return common.rule_overlay_color(rule_id)  # type: ignore[return-value]


def _denorm_roi(roi_normalized: List, width: int, height: int) -> np.ndarray:
    return common.denorm_roi_points(roi_normalized, width, height)


def _draw_all_rois(frame: np.ndarray, rules: List[Dict[str, Any]]) -> None:
    for rule in rules:
        common.draw_rule_geometry(frame, rule, highlight=False)


def _draw_rule_geometry(frame: np.ndarray, rule: Dict[str, Any], highlight: bool = False) -> None:
    """Draw one rule's ROI or gate geometry; optional highlight for breach."""
    common.draw_rule_geometry(frame, rule, highlight=highlight)


def _frame_with_rule_emphasis(frame: np.ndarray, rule: Dict[str, Any]) -> np.ndarray:
    """Copy frame with this rule's zone highlighted for event thumbs."""
    out = common.frame_with_rule_emphasis(frame, rule)
    return out if out is not None else frame.copy()


def _get_pose_model():
    global _pose_model
    if _pose_model is None:
        from ultralytics import YOLO
        _pose_model = YOLO("yolov8n-pose.pt")
    return _pose_model


def _apply_pose_rules(
    frame: np.ndarray,
    rules: List[Dict[str, Any]],
    scan_type: str,
) -> Tuple[np.ndarray, List[Tuple[Dict[str, Any], Dict[str, Any]]]]:
    """Run one pose pass; evaluate all ROIs for intrusion or danger_zone rules."""
    h, w = frame.shape[:2]
    out = frame.copy()
    hits: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    model = _get_pose_model()
    results = model(out, classes=[0], verbose=False)[0]
    out = results.plot()

    if results.keypoints is None:
        return out, hits

    keypoints_xy = results.keypoints.xy.cpu().numpy()
    keypoints_conf = results.keypoints.conf.cpu().numpy()
    machine_active = scan_type == "danger_zone"

    for rule in rules:
        if rule.get("scan_type") != scan_type:
            continue
        roi = rule.get("roi_normalized") or []
        if len(roi) < 3:
            continue
        roi_poly = Polygon(_denorm_roi(roi, w, h))
        buffered = roi_poly.buffer(20) if machine_active else roi_poly
        breached = False

        for person_kpts, conf in zip(keypoints_xy, keypoints_conf):
            if scan_type == "intrusion":
                indices = [15, 16]
            else:
                indices = range(len(person_kpts))

            for kpt_idx in indices:
                if conf[kpt_idx] <= 0.5:
                    continue
                x, y = person_kpts[kpt_idx]
                pt = Point(x, y)
                inside = buffered.contains(pt) if machine_active else roi_poly.contains(pt)
                if inside:
                    breached = True
                    cv2.circle(out, (int(x), int(y)), 8, (0, 0, 255), -1)
                    cv2.putText(
                        out,
                        scan_type.upper().replace("_", " "),
                        (int(x) - 20, int(y) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 255),
                        2,
                    )

        if breached:
            hits.append((rule, {"type": scan_type, "severity": rule.get("severity") or "high"}))

    return out, hits


def _vehicle_event(meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for det in meta.get("detections") or []:
        plate = det.get("plate")
        if plate:
            return {"plate": plate, "type": "vehicle", "severity": "medium"}
    return None


def _rule_css_color(rule_id: str) -> str:
    b, g, r = _rule_color(rule_id)
    return f"rgb({r},{g},{b})"


def _rule_payload(rule: Dict[str, Any]) -> Dict[str, Any]:
    rid = rule.get("id") or ""
    return {
        "id": rid,
        "name": rule.get("name"),
        "scan_type": rule.get("scan_type"),
        "color": _rule_color(rid),
        "css_color": _rule_css_color(rid),
    }


def _playback_status(session: Dict[str, Any]) -> Dict[str, Any]:
    total = int(session.get("total_frames") or 0)
    cur = int(session.get("current_frame") or 0)
    dur = float(session.get("duration_sec") or 0)
    pos = (cur / max(1, total - 1)) if total > 1 else 0.0
    return {
        "seekable": bool(session.get("seekable")),
        "paused": bool(session.get("paused")),
        "current_frame": cur,
        "total_frames": total,
        "current_time_sec": float(session.get("current_time_sec") or 0),
        "duration_sec": dur,
        "position": min(1.0, max(0.0, pos)),
    }


def _emit_if_allowed(
    session: Dict[str, Any],
    cam: Dict[str, Any],
    rule: Dict[str, Any],
    event: Dict[str, Any],
    frame: Optional[np.ndarray] = None,
) -> Optional[Dict[str, Any]]:
    if session.get("preview_only") or session.get("scrub_mode"):
        return None
    snap = frame if frame is not None else session.get("last_raw_frame")
    if (rule.get("scan_type") or "") == "vehicle":
        return common.handle_vehicle_sighting(cam, rule, event, frame=snap)
    return common.emit_site_alert(cam, rule, event, frame=snap)


def _record_monitor_detection(
    session: Dict[str, Any],
    rule: Dict[str, Any],
    event: Dict[str, Any],
    frame: Optional[np.ndarray] = None,
) -> Optional[Dict[str, Any]]:
    """Append monitor event (short debounce); independent of alert cooldown."""
    if session.get("preview_only") or session.get("scrub_mode"):
        return None
    rule_id = rule.get("id") or ""
    now = time.time()
    debounce = session.setdefault("monitor_debounce", {})
    last_ts = float(debounce.get(rule_id) or 0)
    if now - last_ts < common.MONITOR_DEBOUNCE_SEC:
        return None
    debounce[rule_id] = now

    event_id = str(uuid.uuid4())
    scan_type = rule.get("scan_type") or event.get("type") or ""
    msg = event.get("message") or (
        f"{rule.get('name') or scan_type.replace('_', ' ')} — {scan_type.replace('_', ' ')}"
    )
    thumb_url = event.get("thumb_url") or ""
    snap = frame if frame is not None else session.get("last_raw_frame")
    if snap is not None:
        thumb_src = _frame_with_rule_emphasis(snap, rule)
        thumb_url = common._save_frame_thumb(thumb_src, event_id, common.ALERTS_DIR) or thumb_url

    entry = {
        "event_id": event_id,
        "id": event.get("id") or event_id,
        "ts": now,
        "rule_id": rule_id,
        "rule_name": rule.get("name"),
        "scan_type": scan_type,
        "message": msg,
        "thumb_url": thumb_url,
        "clip_url": event.get("clip_url") or "",
        "severity": event.get("severity") or rule.get("severity") or "high",
        "css_color": _rule_css_color(rule_id),
        "roi_normalized": rule.get("roi_normalized") or [],
        "gate_config": rule.get("gate_config") or {},
    }

    lock = session.get("event_lock")
    if lock:
        with lock:
            events = session.setdefault("events", [])
            events.append(entry)
            if len(events) > MAX_SESSION_EVENTS:
                session["events"] = events[-MAX_SESSION_EVENTS:]
            session["last_event"] = msg
            session["last_event_at"] = now
    else:
        events = session.setdefault("events", [])
        events.append(entry)
        if len(events) > MAX_SESSION_EVENTS:
            session["events"] = events[-MAX_SESSION_EVENTS:]
        session["last_event"] = msg
        session["last_event_at"] = now
    return entry


def _record_preview_event(
    session: Dict[str, Any],
    rule: Dict[str, Any],
    event: Dict[str, Any],
) -> None:
    if not session.get("preview_only"):
        return
    rule_id = rule.get("id") or ""
    scan_type = rule.get("scan_type") or event.get("type") or ""
    msg = event.get("message") or (
        f"{rule.get('name') or scan_type.replace('_', ' ')} — {scan_type.replace('_', ' ')}"
    )
    entry = {
        "ts": time.time(),
        "rule_id": rule_id,
        "rule_name": rule.get("name"),
        "scan_type": scan_type,
        "message": msg,
        "css_color": _rule_css_color(rule_id),
    }

    def _append() -> None:
        events = session.setdefault("preview_events", [])
        events.append(entry)
        if len(events) > MAX_PREVIEW_EVENTS:
            session["preview_events"] = events[-MAX_PREVIEW_EVENTS:]

    lock = session.get("event_lock")
    if lock:
        with lock:
            _append()
    else:
        _append()


def _update_session_rule_status(
    session: Dict[str, Any],
    rules: List[Dict[str, Any]],
    hit_rule_ids: set,
) -> None:
    rule_status = []
    for rule in rules:
        rid = rule.get("id") or ""
        payload = _rule_payload(rule)
        payload["state"] = "triggered" if rid in hit_rule_ids else "running"
        rule_status.append(payload)
    session["rule_status"] = rule_status
    session["last_hit_rule_ids"] = hit_rule_ids
    session["last_frame_at"] = time.time()


def _handle_rule_hit(
    session: Dict[str, Any],
    cam: Dict[str, Any],
    rule: Dict[str, Any],
    event: Dict[str, Any],
    frame: Optional[np.ndarray] = None,
) -> None:
    stored_monitor = _record_monitor_detection(session, rule, event, frame)
    stored_alert = _emit_if_allowed(session, cam, rule, event, frame)
    if stored_monitor and stored_alert and stored_alert.get("thumb_url") and not stored_monitor.get("thumb_url"):
        stored_monitor["thumb_url"] = stored_alert.get("thumb_url") or ""


def _blend_overlay(base: np.ndarray, overlay: np.ndarray) -> None:
    if base.shape != overlay.shape:
        return
    cv2.addWeighted(overlay, 0.55, base, 0.45, 0, base)


def _draw_pose_cache_markers(
    out: np.ndarray,
    rules: List[Dict[str, Any]],
    pose_cache: Optional[Dict[str, Any]],
    hit_rule_ids: Optional[set] = None,
) -> None:
    if not pose_cache or pose_cache.get("keypoints_xy") is None:
        return
    h, w = out.shape[:2]
    keypoints_xy = pose_cache["keypoints_xy"]
    keypoints_conf = pose_cache["keypoints_conf"]
    hit_rule_ids = hit_rule_ids or set()
    for rule in rules:
        scan_type = rule.get("scan_type") or ""
        if scan_type not in ("intrusion", "danger_zone"):
            continue
        rule_id = rule.get("id") or ""
        color = _rule_color(rule_id)
        marker_color = (0, 0, 255) if rule_id in hit_rule_ids else color
        roi = rule.get("roi_normalized") or []
        if len(roi) < 3:
            continue
        roi_poly = Polygon(_denorm_roi(roi, w, h))
        buffered = roi_poly.buffer(20) if scan_type == "danger_zone" else roi_poly
        for person_kpts, conf in zip(keypoints_xy, keypoints_conf):
            indices = [15, 16] if scan_type == "intrusion" else range(len(person_kpts))
            for kpt_idx in indices:
                if conf[kpt_idx] <= 0.5:
                    continue
                x, y = person_kpts[kpt_idx]
                pt = Point(x, y)
                inside = buffered.contains(pt) if scan_type == "danger_zone" else roi_poly.contains(pt)
                if inside:
                    cv2.circle(out, (int(x), int(y)), 10 if rule_id in hit_rule_ids else 8, marker_color, -1)
                    lbl = (rule.get("name") or scan_type).upper().replace("_", " ")[:18]
                    cv2.putText(
                        out,
                        lbl,
                        (int(x) - 20, int(y) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        marker_color,
                        2,
                    )


def _monitor_eval_state(session: Dict[str, Any], frame_idx: int, pose_cache: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    # Persist track/dwell/streak across frames (required for loitering + N-frame confirm).
    dwell = session.setdefault("_dwell", {})
    trackers = session.setdefault("_sv_trackers", {})
    streaks = session.setdefault("_verify_streaks", {})
    return {
        "frame_idx": frame_idx,
        "pose_cache": pose_cache,
        "pipelines": session.get("pipelines") or {},
        "generators": session.get("generators") or {},
        "frame_feeds": session.get("frame_feeds") or {},
        "fall_sl": session.setdefault("fall_sl", {}),
        "pipe_lock": session.get("pipe_lock"),
        "monitor_session": session,
        "session_lock": session.get("session_lock"),
        "skip_gate_store": bool(session.get("preview_only")) and not session.get("persist_stats"),
        "_dwell": dwell,
        "_sv_trackers": trackers,
        "_verify_streaks": streaks,
    }


def _evaluate_rules_parallel(
    session: Dict[str, Any],
    cam: Dict[str, Any],
    frame: np.ndarray,
    rules: List[Dict[str, Any]],
    eval_state: Dict[str, Any],
) -> List[Tuple[Dict[str, Any], Dict[str, Any], np.ndarray]]:
    hits: List[Tuple[Dict[str, Any], Dict[str, Any], np.ndarray]] = []
    workers = min(MONITOR_RULE_WORKERS, max(1, len(rules)))

    def _eval_one(rule: Dict[str, Any]):
        fr = frame.copy()
        return scan.evaluate_rule_on_frame(cam, rule, fr, eval_state)

    if workers <= 1 or len(rules) <= 1:
        for rule in rules:
            hit = _eval_one(rule)
            if hit:
                hits.append(hit)
        return hits

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_eval_one, rule): rule for rule in rules}
        for fut in as_completed(futures):
            try:
                hit = fut.result()
                if hit:
                    hits.append(hit)
            except Exception as e:
                log.warning("parallel monitor rule: %s", e)
    return hits


def temp_video_path(temp_id: str) -> Optional[str]:
    if not temp_id or ".." in temp_id or "/" in temp_id or "\\" in temp_id:
        return None
    for ext in (".mp4", ".avi", ".mov", ".webm"):
        path = os.path.join(MONITOR_TEMP_DIR, f"{temp_id}{ext}")
        if os.path.isfile(path):
            return path
    return None


def temp_preview_path(temp_id: str) -> str:
    return os.path.join(MONITOR_TEMP_PREVIEW_DIR, f"{temp_id}.jpg")


def save_temp_upload(filename: str, data: bytes) -> Dict[str, Any]:
    name = filename or "clip.mp4"
    if not name.lower().endswith((".mp4", ".avi", ".mov")):
        raise ValueError("Use mp4, avi, or mov")
    temp_id = str(uuid.uuid4())
    ext = os.path.splitext(name)[1].lower() or ".mp4"
    video_path = os.path.join(MONITOR_TEMP_DIR, f"{temp_id}{ext}")
    with open(video_path, "wb") as f:
        f.write(data)
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        try:
            os.remove(video_path)
        except OSError:
            pass
        raise ValueError("Could not read video")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    cap.release()
    preview = temp_preview_path(temp_id)
    cv2.imwrite(preview, frame)
    return {
        "temp_id": temp_id,
        "filename": name,
        "width": w,
        "height": h,
        "preview_url": f"/storage/monitor_temp/previews/{temp_id}.jpg",
    }


def _cleanup_temp(temp_id: Optional[str]) -> None:
    if not temp_id:
        return
    path = temp_video_path(temp_id)
    if path and os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            pass
    prev = temp_preview_path(temp_id)
    if os.path.isfile(prev):
        try:
            os.remove(prev)
        except OSError:
            pass


def start_monitor(
    camera_id: Optional[str] = None,
    temp_id: Optional[str] = None,
    rule_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    with _lock:
        if _sessions:
            raise ValueError("Only one live monitor session allowed at a time")

        temp_id = (temp_id or "").strip() or None
        camera_id = (camera_id or "").strip() or None
        ids = [rid for rid in (rule_ids or []) if rid]

        if temp_id:
            source = temp_video_path(temp_id)
            if not source:
                raise ValueError("Temp video not found — upload again")
            cam = {
                "id": f"temp:{temp_id}",
                "name": "Test video",
                "type": "file",
                "filename": os.path.basename(source),
            }
            is_file = True
            monitored_id = ""
            if not ids:
                raise ValueError("Select at least one rule")
            rules = []
            for rid in ids:
                rule = store.get_rule(rid)
                if not rule:
                    raise ValueError(f"Rule not found: {rid}")
                if rule.get("enabled") is False:
                    raise ValueError(f"Rule is disabled: {rule.get('name') or rid}")
                rules.append(rule)
        elif camera_id:
            cam = store.get_camera(camera_id)
            if not cam:
                raise ValueError("Camera not found")
            source = common.input_for_camera(cam)
            if not source:
                raise ValueError("Camera source unavailable")
            is_file = (cam.get("type") or "") == "file"
            monitored_id = camera_id
            if ids:
                rules = []
                for rid in ids:
                    rule = store.get_rule(rid)
                    if not rule:
                        raise ValueError(f"Rule not found: {rid}")
                    if rule.get("enabled") is False:
                        raise ValueError(f"Rule is disabled: {rule.get('name') or rid}")
                    rules.append(rule)
            else:
                rules = [
                    r
                    for r in store.list_rules()
                    if r.get("camera_id") == camera_id and r.get("enabled", True)
                ]
            if not rules:
                raise ValueError("No enabled rules for this camera")
            ids = [r.get("id") for r in rules if r.get("id")]
        else:
            raise ValueError("Provide a camera")

        session_id = str(uuid.uuid4())
        stop_ev = threading.Event()
        scan_types = sorted({r.get("scan_type") for r in rules if r.get("scan_type")})

        session: Dict[str, Any] = {
            "id": session_id,
            "camera_id": camera_id or "",
            "camera": cam,
            "rules": rules,
            "selected_rule_ids": ids,
            "scan_types": scan_types,
            "frame_idx": 0,
            "last_event": "",
            "last_event_at": 0.0,
            "events": [],
            "source": source,
            "temp_id": temp_id,
            "is_file": is_file,
            "seekable": is_file,
            "paused": False,
            "scrub_mode": False,
            "seek_to_frame": None,
            "total_frames": 0,
            "fps": 30.0,
            "duration_sec": 0.0,
            "current_frame": 0,
            "current_time_sec": 0.0,
            "last_raw_frame": None,
            "last_jpeg": None,
            "pipelines": {},
            "generators": {},
            "frame_feeds": {},
            "fall_sl": {},
            "monitor_debounce": {},
            "pipe_lock": threading.Lock(),
            "event_lock": threading.Lock(),
            "session_lock": threading.Lock(),
        }

        _sessions[session_id] = session
        _stop_events[session_id] = stop_ev
        if monitored_id:
            common.mark_camera_monitored(monitored_id)

        result = {
            "session_id": session_id,
            "camera_id": camera_id or "",
            "temp_id": temp_id,
            "camera_name": cam.get("name"),
            "rule_count": len(rules),
            "scan_types": scan_types,
            "rules": [_rule_payload(r) for r in rules],
            "events": [],
        }
        result.update(_playback_status(session))

    # Warmup outside lock — prove the source yields at least one frame
    try:
        ret, frame = _read_frame(session)
        if not ret or frame is None:
            stop_monitor(session_id)
            raise ValueError(
                "Could not read from camera — check URL, credentials, and Docker network to DVR"
            )
        h, w = frame.shape[:2]
        session["width"] = w
        session["height"] = h
        session["last_raw_frame"] = frame
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if ok:
            session["last_jpeg"] = (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            )
    except ValueError:
        raise
    except Exception as e:
        stop_monitor(session_id)
        raise ValueError(
            "Could not read from camera — check URL, credentials, and Docker network to DVR"
        ) from e

    return result


def stop_monitor(session_id: str) -> None:
    ev = _stop_events.get(session_id)
    if ev:
        ev.set()
    with _lock:
        session = _sessions.pop(session_id, None)
        _stop_events.pop(session_id, None)
        if session:
            cam_id = session.get("camera_id") or ""
            if cam_id and not session.get("temp_id"):
                common.unmark_camera_monitored(cam_id)
            cap = session.get("capture")
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            for gen in (session.get("generators") or {}).values():
                try:
                    gen.close()
                except Exception:
                    pass
            _cleanup_temp(session.get("temp_id"))
    try:
        from core import site_admin_person_track as ptrack

        ptrack.reset_tracker()
    except Exception:
        pass


def get_monitor_status() -> Dict[str, Any]:
    with _lock:
        if not _sessions:
            return {"active": False}
        sid, session = next(iter(_sessions.items()))
        out = {
            "active": True,
            "session_id": sid,
            "camera_id": session.get("camera_id") or "",
            "temp_id": session.get("temp_id"),
            "camera_name": (session.get("camera") or {}).get("name"),
            "camera_type": (session.get("camera") or {}).get("type") or "rtsp",
            "rule_count": len(session.get("rules") or []),
            "scan_types": session.get("scan_types") or [],
            "last_event": session.get("last_event") or "",
            "last_event_at": session.get("last_event_at") or 0,
            "rules": [_rule_payload(r) for r in (session.get("rules") or [])],
            "events": list(session.get("events") or []),
        }
        out.update(_playback_status(session))
        if session.get("gate_live"):
            out["gate_live"] = session.get("gate_live")
        journey_live = session.get("journey_live")
        if not journey_live:
            try:
                from core import journey_store

                cam_id = session.get("camera_id") or ""
                journey_live = journey_store.journeys_for_camera(cam_id, within_sec=90.0, limit=12)
            except Exception:
                journey_live = []
        if journey_live:
            out["journey_live"] = journey_live
        if session.get("sam_selection"):
            out["sam_selection"] = session.get("sam_selection")
        return out


def _reinit_fall_pipeline(session: Dict[str, Any]) -> None:
    for gen in (session.get("generators") or {}).values():
        try:
            gen.close()
        except Exception:
            pass
    session["generators"] = {}
    session["frame_feeds"] = {}
    w = int(session.get("width") or 640)
    h = int(session.get("height") or 480)
    fps = float(session.get("fps") or 30.0)
    if "fall" not in (session.get("scan_types") or []):
        return
    pipeline_name = common.SCAN_TO_PIPELINE.get("fall")
    if not pipeline_name:
        return
    first = next((r for r in session["rules"] if r.get("scan_type") == "fall"), None)
    roi = (first or {}).get("roi_normalized") or []
    feed = FrameFeed(w, h, fps)
    pipe = registry.get_pipeline(pipeline_name)
    gen = pipe.run_on_video(feed, common.ALERTS_DIR, roi, common.pipeline_config("fall"))
    session["frame_feeds"]["fall"] = feed
    session["generators"]["fall"] = gen


def seek_monitor(
    session_id: str,
    position: Optional[float] = None,
    frame: Optional[int] = None,
) -> Dict[str, Any]:
    session = _sessions.get(session_id)
    if not session:
        raise ValueError("Monitor session not found")
    if not session.get("seekable"):
        raise ValueError("Seek is only available for uploaded test videos")
    total = int(session.get("total_frames") or 1)
    if frame is not None:
        target = max(0, min(int(frame), total - 1))
    elif position is not None:
        target = max(0, min(int(float(position) * (total - 1)), total - 1))
    else:
        raise ValueError("Provide position (0..1) or frame index")
    session["seek_to_frame"] = target
    session["scrub_mode"] = True
    session["frame_idx"] = target
    session["last_jpeg"] = None
    _reinit_fall_pipeline(session)
    return _playback_status(session)


def set_monitor_paused(session_id: str, paused: bool) -> Dict[str, Any]:
    session = _sessions.get(session_id)
    if not session:
        raise ValueError("Monitor session not found")
    session["paused"] = paused
    session["scrub_mode"] = paused
    if not paused:
        session["scrub_mode"] = False
    if paused:
        session["last_jpeg"] = None
    return _playback_status(session)


def _ensure_capture(session: Dict[str, Any]):
    if session.get("capture") is not None:
        return session["capture"]
    source = session["source"]
    cap = get_video_source(source)
    session["capture"] = cap
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    if w <= 1 or h <= 1:
        ret, frame = cap.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            session["last_raw_frame"] = frame
    w = w or 640
    h = h or 480
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    session["width"] = w
    session["height"] = h
    session["fps"] = fps
    if session.get("is_file"):
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        session["total_frames"] = max(1, total)
        session["duration_sec"] = total / fps if fps > 0 else 0.0
        session["seekable"] = True
    else:
        session["total_frames"] = 0
        session["duration_sec"] = 0.0
        session["seekable"] = False

    for scan_type in session.get("scan_types") or []:
        pipeline_name = common.SCAN_TO_PIPELINE.get(scan_type)
        if not pipeline_name:
            continue
        if scan_type in ("face_attendance", "vehicle", "gate_analytics", "fall_standing_lying"):
            session["pipelines"][scan_type] = registry.get_pipeline(pipeline_name)
        elif scan_type == "fall":
            first = next((r for r in session["rules"] if r.get("scan_type") == "fall"), None)
            roi = (first or {}).get("roi_normalized") or []
            feed = FrameFeed(w, h, fps)
            pipe = registry.get_pipeline(pipeline_name)
            gen = pipe.run_on_video(
                feed,
                common.ALERTS_DIR,
                roi,
                common.pipeline_config("fall"),
            )
            session["frame_feeds"]["fall"] = feed
            session["generators"]["fall"] = gen
    return cap


def _read_frame(session: Dict[str, Any]) -> Tuple[bool, Optional[np.ndarray]]:
    cap = _ensure_capture(session)
    fps = float(session.get("fps") or 30.0)

    if session.get("seek_to_frame") is not None:
        target = int(session["seek_to_frame"])
        session["seek_to_frame"] = None
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = cap.read()
        if ret and frame is not None:
            session["current_frame"] = target
            session["current_time_sec"] = target / fps if fps > 0 else 0.0
            session["last_raw_frame"] = frame
            session["last_jpeg"] = None
            return True, frame

    if session.get("paused") and session.get("last_raw_frame") is not None:
        return True, session["last_raw_frame"]

    ret, frame = cap.read()
    if ret and frame is not None:
        pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if pos < 0:
            pos = int(session.get("current_frame") or 0)
        session["current_frame"] = pos
        session["current_time_sec"] = pos / fps if fps > 0 else 0.0
        session["last_raw_frame"] = frame
        return True, frame

    if session.get("is_file"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        session["current_frame"] = 0
        session["current_time_sec"] = 0.0
        ret, frame = cap.read()
        if ret and frame is not None:
            session["last_raw_frame"] = frame
            session["last_jpeg"] = None
        return ret, frame
    return False, None


def _process_frame(session: Dict[str, Any], frame: np.ndarray) -> np.ndarray:
    rules: List[Dict[str, Any]] = session["rules"]
    cam: Dict[str, Any] = session["camera"]
    frame_idx = session["frame_idx"]
    out = frame.copy()

    needs_pose = any(
        (r.get("scan_type") or "") in ("intrusion", "danger_zone", "loitering") for r in rules
    )
    pose_cache = scan.build_pose_cache(frame) if needs_pose else None
    eval_state = _monitor_eval_state(session, frame_idx, pose_cache)

    hits = _evaluate_rules_parallel(session, cam, frame, rules, eval_state)
    hit_rule_ids = {rule.get("id") or "" for rule, _, _ in hits}

    # Surface latest verifier votes for preview legend chips
    votes_by_rule = eval_state.get("verifier_votes_by_rule") or {}
    if votes_by_rule:
        session["verifier_votes_by_rule"] = dict(votes_by_rule)
        # Prefer a vote from a hit; else any recent
        session["verifier_votes"] = None
        for rule, event, _ in hits:
            vv = (event or {}).get("verifier_votes")
            if vv:
                session["verifier_votes"] = vv
                break
        if session.get("verifier_votes") is None:
            session["verifier_votes"] = next(iter(votes_by_rule.values()), None)
    elif eval_state.get("verifier_votes"):
        session["verifier_votes"] = eval_state.get("verifier_votes")

    for rule, event, annotated in hits:
        snap = annotated if annotated is not None else out
        _handle_rule_hit(session, cam, rule, event, snap)
        if session.get("preview_only"):
            _record_preview_event(session, rule, event)
        _draw_rule_geometry(out, rule, highlight=True)
        if annotated is not None and annotated is not out:
            _blend_overlay(out, annotated)

    _update_session_rule_status(session, rules, hit_rule_ids)

    # YOLO results.plot() paints green skeletons and washes the preview feed.
    # Keep markers + ROI outlines on preview; full plot only for Live Monitor.
    if pose_cache and pose_cache.get("results") is not None:
        if not session.get("preview_only"):
            try:
                plotted = pose_cache["results"].plot()
                if plotted is not None and plotted.shape == out.shape:
                    cv2.addWeighted(plotted, 0.3, out, 0.7, 0, out)
            except Exception:
                pass
        _draw_pose_cache_markers(out, rules, pose_cache, hit_rule_ids)

    for rule in rules:
        rid = rule.get("id") or ""
        if rid not in hit_rule_ids:
            _draw_rule_geometry(out, rule, highlight=False)
    session["frame_idx"] = frame_idx + 1
    return out


def _refresh_watched_person_tracks(session: Dict[str, Any], frame: np.ndarray) -> None:
    """Update SAM/journey overlay bboxes from YOLO ByteTrack for pinned people."""
    sel = session.get("sam_selection") or {}
    tid = sel.get("local_track_id")
    live = list(session.get("journey_live") or [])
    need = tid is not None or any(j.get("local_track_id") is not None for j in live)
    if not need or frame is None or getattr(frame, "size", 0) == 0:
        return
    now = time.time()
    if now - float(session.get("_person_track_at") or 0) < 0.2:
        return
    session["_person_track_at"] = now
    try:
        from core import site_admin_person_track as ptrack
        from core import journey_store

        tracks = ptrack.track_persons(frame, persist=True)
    except Exception:
        return
    by_id = {int(t["track_id"]): t for t in tracks if t.get("track_id") is not None}
    cam_id = session.get("camera_id") or ""
    last_meta_at = float(session.get("_person_track_meta_at") or 0)
    write_meta = (now - last_meta_at) >= 1.0

    def _apply(track: Dict[str, Any], item: Dict[str, Any]) -> None:
        bn = track.get("bbox_norm") or []
        if len(bn) == 4:
            item["bbox_norm"] = bn
        poly = track.get("polygon") or []
        if len(poly) >= 3:
            item["polygon"] = poly
        xyxy = track.get("xyxy") or []
        gid = item.get("gid") or ""
        if write_meta and gid and len(xyxy) == 4:
            try:
                meta = journey_store.get_meta(gid)
                if meta:
                    meta["last_bbox"] = [float(x) for x in xyxy]
                    meta["last_seen_at"] = now
                    if cam_id:
                        meta["last_camera_id"] = cam_id
                        meta["last_camera_name"] = (session.get("camera") or {}).get("name") or cam_id
                    journey_store._save_meta(meta)
            except Exception:
                pass

    if tid is not None and int(tid) in by_id:
        track = by_id[int(tid)]
        sel = dict(sel)
        _apply(track, sel)
        sel["local_track_id"] = int(tid)
        session["sam_selection"] = sel

    new_live = []
    for j in live:
        jtid = j.get("local_track_id")
        item = dict(j)
        if jtid is not None and int(jtid) in by_id:
            _apply(by_id[int(jtid)], item)
            item["local_track_id"] = int(jtid)
        if sel and (item.get("gid") or "") == (sel.get("gid") or "") and sel.get("bbox_norm"):
            item["bbox_norm"] = sel["bbox_norm"]
            if sel.get("local_track_id") is not None:
                item["local_track_id"] = sel["local_track_id"]
        new_live.append(item)
    session["journey_live"] = new_live[:20]
    if write_meta:
        session["_person_track_meta_at"] = now


def _process_frame_for_stream(session: Dict[str, Any], frame: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Live Monitor / DVR poll path: never block the HTTP response on face/vehicle.
    Those share a singleton InsightFace/YOLO with go-live workers and can hang
    forever under concurrent process_frame — matching go-live hero preview, we
    only run light rules here (pose / gate / loitering) and draw geometry for heavy ones.
    """
    _refresh_watched_person_tracks(session, frame)

    rules: List[Dict[str, Any]] = list(session.get("rules") or [])
    light_rules = [r for r in rules if (r.get("scan_type") or "") in PREVIEW_LIGHT_SCAN_TYPES]
    heavy_rules = [r for r in rules if r not in light_rules]

    if not light_rules:
        out = frame.copy()
        for rule in rules:
            _draw_rule_geometry(out, rule, highlight=False)
        session["frame_idx"] = int(session.get("frame_idx") or 0) + 1
        return out, "geometry_only"

    saved = session["rules"]
    session["rules"] = light_rules
    try:
        out = _process_frame(session, frame)
        for rule in heavy_rules:
            _draw_rule_geometry(out, rule, highlight=False)
        return out, "light_rules"
    finally:
        session["rules"] = saved


def session_exists(session_id: str) -> bool:
    return session_id in _sessions


def get_session_raw_frame(session_id: str) -> Optional[np.ndarray]:
    """Latest raw (or freshly read) BGR frame for a live monitor session."""
    session = _sessions.get(session_id)
    if not session:
        return None
    frame = session.get("last_raw_frame")
    if frame is not None and getattr(frame, "size", 0) > 0:
        return frame.copy()
    ret, frame = _read_frame(session)
    if ret and frame is not None:
        session["last_raw_frame"] = frame
        return frame.copy()
    return None


def set_sam_selection(session_id: str, selection: Dict[str, Any], journey_item: Dict[str, Any]) -> None:
    """Store FastSAM click selection + enrich journey_live for Monitor overlay."""
    with _lock:
        sess = _sessions.get(session_id)
        if sess is None:
            return
        sess["sam_selection"] = selection
        live = list(sess.get("journey_live") or [])
        gid = journey_item.get("gid") or ""
        live = [x for x in live if (x.get("gid") or "") != gid]
        live.insert(0, journey_item)
        sess["journey_live"] = live[:20]


def clear_sam_selection(session_id: str) -> bool:
    """Clear FastSAM selection overlay on a live monitor session (does not unwatch)."""
    with _lock:
        sess = _sessions.get(session_id)
        if sess is None:
            return False
        sess["sam_selection"] = None
        return True


def clear_selection_for_gid(gid: str) -> bool:
    """If any live monitor session is highlighting this journey, clear it."""
    if not gid:
        return False
    cleared = False
    with _lock:
        for sess in _sessions.values():
            sel = sess.get("sam_selection") or {}
            if (sel.get("gid") or "") == gid:
                sess["sam_selection"] = None
                cleared = True
            live = list(sess.get("journey_live") or [])
            filtered = [x for x in live if (x.get("gid") or "") != gid]
            if len(filtered) != len(live):
                sess["journey_live"] = filtered
                cleared = True
    return cleared


def capture_frame_jpeg(session_id: str) -> Optional[bytes]:
    """Single processed JPEG for DVR poll preview (and MJPEG fallback)."""
    # #region agent log
    def _agent_log(hyp, msg, data=None):
        try:
            import json as _json
            import time as _time
            line = _json.dumps({"sessionId": "46c418", "hypothesisId": hyp, "location": "site_admin_monitor.py:capture_frame_jpeg", "message": msg, "data": data or {}, "timestamp": int(_time.time() * 1000)}) + "\n"
            for p in ("/app/static/debug-46c418.log", "static/debug-46c418.log", "debug-46c418.log"):
                try:
                    with open(p, "a", encoding="utf-8") as f:
                        f.write(line)
                    break
                except Exception:
                    continue
        except Exception:
            pass
    # #endregion
    session = _sessions.get(session_id)
    if not session:
        # #region agent log
        _agent_log("B", "capture_frame no session", {"session_id": session_id[:8] if session_id else ""})
        # #endregion
        return None
    t0 = time.time()
    # #region agent log
    cam = session.get("camera") or {}
    _agent_log("B", "capture_frame start", {"cam_type": cam.get("type"), "scan_types": session.get("scan_types"), "has_raw": session.get("last_raw_frame") is not None})
    # #endregion
    ret, frame = _read_frame(session)
    if not ret or frame is None:
        # #region agent log
        _agent_log("B", "capture_frame read failed", {"elapsed_ms": int((time.time() - t0) * 1000)})
        # #endregion
        return None
    t1 = time.time()
    # #region agent log
    _agent_log("B", "capture_frame after read", {"read_ms": int((t1 - t0) * 1000), "shape": list(frame.shape) if frame is not None else None, "runId": "post-fix"})
    # #endregion
    out, mode = _process_frame_for_stream(session, frame)
    t2 = time.time()
    ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    # #region agent log
    _agent_log("B", "capture_frame done", {"read_ms": int((t1 - t0) * 1000), "process_ms": int((t2 - t1) * 1000), "ok": bool(ok), "jpeg_bytes": int(len(buf.tobytes()) if ok else 0), "shape": list(frame.shape) if frame is not None else None, "mode": mode, "runId": "post-fix"})
    # #endregion
    return buf.tobytes() if ok else None


def mjpeg_generator(session_id: str) -> Generator[bytes, None, None]:
    session = _sessions.get(session_id)
    if not session:
        return
    stop_ev = _stop_events.get(session_id)
    os.makedirs(common.ALERTS_DIR, exist_ok=True)

    # #region agent log
    def _agent_log(hyp, msg, data=None):
        try:
            import json as _json
            line = _json.dumps({"sessionId": "46c418", "hypothesisId": hyp, "location": "site_admin_monitor.py:mjpeg_generator", "message": msg, "data": data or {}, "timestamp": int(time.time() * 1000)}) + "\n"
            for p in ("/app/static/debug-46c418.log", "static/debug-46c418.log", "debug-46c418.log"):
                try:
                    with open(p, "a", encoding="utf-8") as f:
                        f.write(line)
                    break
                except Exception:
                    continue
        except Exception:
            pass
    cam = session.get("camera") or {}
    _agent_log("A", "mjpeg start", {"cam_type": cam.get("type"), "has_last_jpeg": bool(session.get("last_jpeg")), "scan_types": session.get("scan_types")})
    # #endregion

    try:
        if session.get("last_jpeg"):
            # #region agent log
            _agent_log("A", "mjpeg yield warmup", {"chunk_len": len(session.get("last_jpeg") or b"")})
            # #endregion
            yield session["last_jpeg"]
        _yield_n = 0
        while stop_ev and not stop_ev.is_set():
            pending_seek = session.get("seek_to_frame") is not None
            if session.get("paused") and not pending_seek and session.get("last_jpeg"):
                yield session["last_jpeg"]
                time.sleep(0.2)
                continue

            ret, frame = _read_frame(session)
            if not ret or frame is None:
                time.sleep(0.1)
                continue
            t0 = time.time()
            out, mode = _process_frame_for_stream(session, frame)
            ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if ok:
                chunk = (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + buf.tobytes()
                    + b"\r\n"
                )
                session["last_jpeg"] = chunk
                _yield_n += 1
                # #region agent log
                if _yield_n <= 3:
                    _agent_log("A", "mjpeg yield frame", {"n": _yield_n, "process_ms": int((time.time() - t0) * 1000), "chunk_len": len(chunk), "mode": mode, "runId": "post-fix"})
                # #endregion
                yield chunk
            if session.get("paused"):
                time.sleep(0.15)
            else:
                time.sleep(0.066)
    except GeneratorExit:
        pass
    except Exception as e:
        # #region agent log
        _agent_log("A", "mjpeg exception", {"err": str(e)[:200]})
        # #endregion
        log.warning("monitor stream %s ended: %s", session_id, e)


def _grab_preview_frame(cam: Dict[str, Any]) -> Optional[np.ndarray]:
    """One-shot frame for go-live preview (no persistent capture)."""
    kind = cam.get("type") or "rtsp"
    if kind == "dvr":
        from core.snapshot_camera import fetch_snapshot_jpeg

        snapshot_url = (cam.get("snapshot_url") or "").strip()
        if not snapshot_url:
            return None
        return fetch_snapshot_jpeg(
            snapshot_url,
            user=(cam.get("http_user") or "").strip(),
            password=(cam.get("http_password") or ""),
        )

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


def _dispose_preview_session(cam_id: str) -> None:
    session = _preview_cache.pop(cam_id, None)
    if not session:
        return
    for gen in (session.get("generators") or {}).values():
        try:
            gen.close()
        except Exception:
            pass


def clear_preview_cache() -> None:
    with _preview_lock:
        for cam_id in list(_preview_cache.keys()):
            _dispose_preview_session(cam_id)


def _init_preview_pipelines(session: Dict[str, Any], frame: np.ndarray) -> None:
    h, w = frame.shape[:2]
    session["width"] = w
    session["height"] = h
    fps = float(session.get("fps") or 30.0)
    if session.get("pipelines_initialized"):
        return
    for scan_type in session.get("scan_types") or []:
        pipeline_name = common.SCAN_TO_PIPELINE.get(scan_type)
        if not pipeline_name:
            continue
        if scan_type in ("face_attendance", "vehicle", "gate_analytics", "fall_standing_lying"):
            session["pipelines"][scan_type] = registry.get_pipeline(pipeline_name)
        elif scan_type == "fall":
            first = next((r for r in session["rules"] if r.get("scan_type") == "fall"), None)
            roi = (first or {}).get("roi_normalized") or []
            feed = FrameFeed(w, h, fps)
            pipe = registry.get_pipeline(pipeline_name)
            gen = pipe.run_on_video(
                feed,
                common.ALERTS_DIR,
                roi,
                common.pipeline_config("fall"),
            )
            session["frame_feeds"]["fall"] = feed
            session["generators"]["fall"] = gen
    session["pipelines_initialized"] = True


def _ensure_preview_session(cam_id: str) -> Optional[Dict[str, Any]]:
    now = time.time()
    with _preview_lock:
        stale = [cid for cid, s in _preview_cache.items() if now - float(s.get("last_access") or 0) > PREVIEW_CACHE_TTL]
        for cid in stale:
            _dispose_preview_session(cid)

        session = _preview_cache.get(cam_id)
        if session:
            session["last_access"] = now
            # Refresh rules each hit so newly saved ROIs appear on preview
            cam = store.get_camera(cam_id) or session.get("camera") or {}
            rules = [
                r
                for r in store.list_rules()
                if r.get("camera_id") == cam_id and r.get("enabled", True)
            ]
            scan_types = sorted({r.get("scan_type") for r in rules if r.get("scan_type")})
            old_types = set(session.get("scan_types") or [])
            session["camera"] = cam
            session["rules"] = rules
            session["scan_types"] = scan_types
            if set(scan_types) != old_types:
                session["pipelines_initialized"] = False
            return session

        cam = store.get_camera(cam_id)
        if not cam:
            return None
        rules = [
            r
            for r in store.list_rules()
            if r.get("camera_id") == cam_id and r.get("enabled", True)
        ]
        scan_types = sorted({r.get("scan_type") for r in rules if r.get("scan_type")})
        session = {
            "camera_id": cam_id,
            "camera": cam,
            "rules": rules,
            "scan_types": scan_types,
            "frame_idx": 0,
            "pipelines": {},
            "generators": {},
            "frame_feeds": {},
            "preview_only": True,
            "persist_stats": False,
            "preview_events": [],
            "rule_status": [{**_rule_payload(r), "state": "idle"} for r in rules],
            "last_hit_rule_ids": set(),
            "last_frame_at": 0.0,
            "pipe_lock": threading.Lock(),
            "session_lock": threading.Lock(),
            "event_lock": threading.Lock(),
            "render_lock": threading.Lock(),
            "render_busy": False,
            "last_overlay_jpeg": None,
            "last_access": now,
            "width": 640,
            "height": 480,
            "fps": 30.0,
            "pipelines_initialized": False,
        }
        _preview_cache[cam_id] = session
        return session


def render_preview_frame(cam_id: str, apply_rules: bool) -> Optional[bytes]:
    """JPEG for go-live hero preview; optional rule overlays without alerts."""
    cam = store.get_camera(cam_id)
    if not cam:
        return None
    frame = _grab_preview_frame(cam)
    if frame is None:
        return None
    if not apply_rules:
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
        return buf.tobytes() if ok else None

    session = _ensure_preview_session(cam_id)
    if not session:
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
        return buf.tobytes() if ok else None

    session["last_access"] = time.time()
    session["persist_stats"] = True
    if "render_lock" not in session:
        session["render_lock"] = threading.Lock()
    session["last_raw_frame"] = frame
    rules = session.get("rules") or []
    if not rules:
        out = frame.copy()
        ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        return buf.tobytes() if ok else None

    # Fast path: always draw rule outlines so preview stays responsive
    geometry_out = frame.copy()
    for rule in rules:
        _draw_rule_geometry(geometry_out, rule, highlight=False)

    light_rules = [r for r in rules if (r.get("scan_type") or "") in PREVIEW_LIGHT_SCAN_TYPES]
    # Face/vehicle-only cameras: outlines only (workers do detection). No lock needed.
    if not light_rules:
        ok, buf = cv2.imencode(".jpg", geometry_out, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        data = buf.tobytes() if ok else None
        if data:
            session["last_overlay_jpeg"] = data
            session["last_frame_at"] = time.time()
        return data

    lock = session["render_lock"]
    # Drop orphaned lock from a previous hung face/vehicle process (pre-fix)
    if session.get("render_busy") and float(session.get("busy_since") or 0) > 0:
        if time.time() - float(session["busy_since"]) > 8.0:
            session["render_busy"] = False
            try:
                lock.release()
            except RuntimeError:
                pass

    got_lock = lock.acquire(blocking=False)
    if not got_lock:
        ok, buf = cv2.imencode(".jpg", geometry_out, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        data = buf.tobytes() if ok else None
        if data:
            session["last_overlay_jpeg"] = data
        return data

    session["render_busy"] = True
    session["busy_since"] = time.time()
    try:
        # Preview: evaluate only light rules so face/vehicle cannot hold the lock forever.
        # Go-live workers still run ALL rules on every camera in parallel processes.
        saved_rules = session["rules"]
        session["rules"] = light_rules
        _init_preview_pipelines(session, frame)
        out = _process_frame(session, frame)
        # Ensure every rule outline is present (including heavy types)
        for rule in saved_rules:
            if (rule.get("scan_type") or "") not in PREVIEW_LIGHT_SCAN_TYPES:
                _draw_rule_geometry(out, rule, highlight=False)
        session["rules"] = saved_rules
    except Exception:
        session["rules"] = rules
        out = geometry_out
    finally:
        session["render_busy"] = False
        session["busy_since"] = 0.0
        lock.release()

    ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    data = buf.tobytes() if ok else None
    if data:
        session["last_overlay_jpeg"] = data
    return data


def get_preview_status(cam_id: str) -> Dict[str, Any]:
    """JSON status for go-live hero preview — rules, hits, gate live, preview events."""
    now = time.time()
    with _preview_lock:
        session = _preview_cache.get(cam_id)
        if not session:
            return {"active": False, "camera_id": cam_id}
        last = float(session.get("last_frame_at") or session.get("last_access") or 0)
        if now - last > PREVIEW_CACHE_TTL:
            return {"active": False, "camera_id": cam_id}
        cam = session.get("camera") or {}
        rules = session.get("rules") or []
        out: Dict[str, Any] = {
            "active": True,
            "camera_id": cam_id,
            "camera_name": cam.get("name"),
            "persist_stats": bool(session.get("persist_stats")),
            "rule_count": len(rules),
            "scan_types": session.get("scan_types") or [],
            "rules": [_rule_payload(r) for r in rules],
            "rule_status": list(session.get("rule_status") or []),
            "events": list(session.get("preview_events") or []),
            "parallel_workers": MONITOR_RULE_WORKERS,
            "last_frame_at": session.get("last_frame_at") or 0,
        }
        if session.get("gate_live"):
            out["gate_live"] = session.get("gate_live")
        if session.get("verifier_votes"):
            out["verifier_votes"] = session.get("verifier_votes")
        if session.get("verifier_votes_by_rule"):
            out["verifier_votes_by_rule"] = session.get("verifier_votes_by_rule")
        journey_live = session.get("journey_live")
        if not journey_live:
            try:
                from core import journey_store

                journey_live = journey_store.journeys_for_camera(cam_id, within_sec=90.0, limit=12)
            except Exception:
                journey_live = []
        if journey_live:
            out["journey_live"] = journey_live
        return out
