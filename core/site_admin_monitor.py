"""Live Site Admin monitor — one camera, all rules, MJPEG stream."""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from typing import Any, Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np
from shapely.geometry import Point, Polygon

from core import site_admin_common as common
from core import site_admin_store as store
from core.registry import registry
from core.video_source import ThreadedCamera, get_video_source

log = logging.getLogger("site_admin.monitor")

MONITOR_TEMP_DIR = os.path.join("storage", "monitor_temp")
MONITOR_TEMP_PREVIEW_DIR = os.path.join("storage", "monitor_temp", "previews")
os.makedirs(MONITOR_TEMP_DIR, exist_ok=True)
os.makedirs(MONITOR_TEMP_PREVIEW_DIR, exist_ok=True)

MAX_SESSION_EVENTS = 50

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
    idx = abs(hash(rule_id)) % len(RULE_COLORS)
    return RULE_COLORS[idx]


def _denorm_roi(roi_normalized: List, width: int, height: int) -> np.ndarray:
    pts = [[int(x * width), int(y * height)] for x, y in roi_normalized]
    return np.array(pts, dtype=np.int32)


def _draw_all_rois(frame: np.ndarray, rules: List[Dict[str, Any]]) -> None:
    h, w = frame.shape[:2]
    for rule in rules:
        roi = rule.get("roi_normalized") or []
        if len(roi) < 3:
            continue
        color = _rule_color(rule.get("id") or "")
        pts = _denorm_roi(roi, w, h)
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
        label = f"{rule.get('name', 'Rule')} ({rule.get('scan_type', '')})"
        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))
        cv2.putText(frame, label, (cx, max(20, cy - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


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
    if session.get("scrub_mode"):
        return None
    snap = frame if frame is not None else session.get("last_raw_frame")
    if (rule.get("scan_type") or "") == "vehicle":
        stored = common.handle_vehicle_sighting(cam, rule, event, frame=snap)
    else:
        stored = common.emit_site_alert(cam, rule, event, frame=snap)
    if stored:
        session["last_event"] = stored.get("message") or rule.get("name") or ""
        session["last_event_at"] = time.time()
        events = session.setdefault("events", [])
        events.append({
            "id": stored.get("id"),
            "ts": time.time(),
            "rule_id": rule.get("id"),
            "rule_name": rule.get("name"),
            "scan_type": rule.get("scan_type") or stored.get("scan_type"),
            "message": stored.get("message") or "",
            "thumb_url": stored.get("thumb_url") or "",
            "clip_url": stored.get("clip_url") or "",
        })
        if len(events) > MAX_SESSION_EVENTS:
            session["events"] = events[-MAX_SESSION_EVENTS:]
    return stored


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
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
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
        if scan_type in ("face_attendance", "vehicle", "gate_analytics"):
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
    _draw_all_rois(out, rules)

    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for rule in rules:
        st = rule.get("scan_type") or ""
        by_type.setdefault(st, []).append(rule)

    if "intrusion" in by_type:
        out, hits = _apply_pose_rules(out, by_type["intrusion"], "intrusion")
        for rule, event in hits:
            _emit_if_allowed(session, cam, rule, event, out)

    if "danger_zone" in by_type:
        out, hits = _apply_pose_rules(out, by_type["danger_zone"], "danger_zone")
        for rule, event in hits:
            _emit_if_allowed(session, cam, rule, event, out)

    if "face_attendance" in by_type:
        pipe = session["pipelines"].get("face_attendance")
        if pipe:
            try:
                annotated, meta = pipe.process_frame(out, frame_idx, np.array([]), common.pipeline_config("face_attendance"))
                if annotated is not None:
                    out = annotated
                if isinstance(meta, dict) and (meta.get("person_id") or meta.get("type") == "face_recognised"):
                    for rule in by_type["face_attendance"]:
                        _emit_if_allowed(session, cam, rule, meta, out)
            except Exception as e:
                log.warning("face monitor frame: %s", e)

    if "vehicle" in by_type:
        pipe = session["pipelines"].get("vehicle")
        if pipe:
            try:
                annotated, meta = pipe.process_frame(out, frame_idx, np.array([]), {})
                if annotated is not None:
                    out = annotated
                event = _vehicle_event(meta or {})
                if event:
                    for rule in by_type["vehicle"]:
                        _emit_if_allowed(session, cam, rule, event, out)
            except Exception as e:
                log.warning("vehicle monitor frame: %s", e)

    if "gate_analytics" in by_type:
        pipe = session["pipelines"].get("gate_analytics")
        if pipe:
            for rule in by_type["gate_analytics"]:
                try:
                    gate_config = rule.get("gate_config") or {}
                    if not common.gate_config_valid(gate_config):
                        continue
                    rule_id = rule.get("id") or ""
                    baseline = store.get_gate_baseline(rule_id)
                    config: Dict[str, Any] = {
                        "gate_config": gate_config,
                        "rule_id": rule_id,
                    }
                    if baseline is not None:
                        config["force_baseline"] = baseline
                    annotated, meta = pipe.process_frame(out, frame_idx, None, config)
                    if annotated is not None:
                        out = annotated
                    if isinstance(meta, dict) and meta.get("type") == "gate_tick":
                        counters = meta.get("counters") or {}
                        if any(counters.values()):
                            store.increment_gate_counters(rule_id, counters)
                        session["gate_live"] = meta.get("live") or {}
                        if meta.get("band") == "near" and meta.get("alert_near"):
                            _emit_if_allowed(
                                session,
                                cam,
                                rule,
                                {"type": "gate_near", "severity": rule.get("severity") or "medium"},
                                out,
                            )
                except Exception as e:
                    log.warning("gate monitor frame: %s", e)

    if "fall" in by_type:
        feed: FrameFeed = session["frame_feeds"].get("fall")
        gen = session["generators"].get("fall")
        if feed and gen:
            try:
                feed.frame = out
                annotated, event = next(gen)
                if annotated is not None:
                    out = annotated
                if common.is_event_dict(event):
                    for rule in by_type["fall"]:
                        _emit_if_allowed(session, cam, rule, event, out)
            except StopIteration:
                pass
            except Exception as e:
                log.warning("fall monitor frame: %s", e)

    _draw_all_rois(out, rules)
    session["frame_idx"] = frame_idx + 1
    return out


def session_exists(session_id: str) -> bool:
    return session_id in _sessions


def mjpeg_generator(session_id: str) -> Generator[bytes, None, None]:
    session = _sessions.get(session_id)
    if not session:
        return
    stop_ev = _stop_events.get(session_id)
    os.makedirs(common.ALERTS_DIR, exist_ok=True)

    try:
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
            out = _process_frame(session, frame)
            ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if ok:
                chunk = (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + buf.tobytes()
                    + b"\r\n"
                )
                session["last_jpeg"] = chunk
                yield chunk
            if session.get("paused"):
                time.sleep(0.15)
            else:
                time.sleep(0.066)
    finally:
        stop_monitor(session_id)
