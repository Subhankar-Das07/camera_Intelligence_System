"""Shared Site Admin helpers for runtime and live monitor."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from core import site_admin_store as store
from core import whatsapp_adapter
from core import email_adapter

UPLOAD_DIR = os.path.join("storage", "uploads")
ALERTS_DIR = os.path.join("storage", "alerts")
VEHICLES_DIR = os.path.join("storage", "vehicles")

VEHICLE_STATUSES = ("candidate", "approved", "ignored", "risk", "danger")

MAX_RULES_PER_CAMERA = 3
MONITOR_DEBOUNCE_SEC = float(os.environ.get("SITE_ADMIN_MONITOR_DEBOUNCE_SEC", "3"))

SCAN_TO_PIPELINE = {
    "intrusion": "intrusion_detection",
    "danger_zone": "danger_zone",
    "fall": "fall_detection",
    "fall_standing_lying": "fall_standing_lying",
    "face_attendance": "face_recognition",
    "vehicle": "vehicle_recognition",
    "gate_analytics": "gate_analytics",
}

# Full product catalog for Rules UI. Only ids in SCAN_TO_PIPELINE can be saved/run.
SCAN_CATALOG: List[Dict[str, Any]] = [
    {
        "id": "intrusion",
        "label": "Intrusion / restricted area",
        "typical_need": "Person enters a drawn area",
        "fit": "High — shops, homes, factories",
        "category": "People & safety",
        "status": "available",
        "kind": "area",
    },
    {
        "id": "danger_zone",
        "label": "Danger zone",
        "typical_need": "Person near machines or hazards",
        "fit": "High — factories, workshops",
        "category": "People & safety",
        "status": "available",
        "kind": "area",
    },
    {
        "id": "fall",
        "label": "Fall detection",
        "typical_need": "Detect a person falling (live video / RTSP)",
        "fit": "High — RTSP, NVR, Live Monitor",
        "category": "People & safety",
        "status": "available",
        "kind": "area",
    },
    {
        "id": "fall_standing_lying",
        "label": "Fall detection — standing & lying",
        "typical_need": "Person was standing, then found lying (snapshot cameras)",
        "fit": "High — DVR / HTTP picture polling",
        "category": "People & safety",
        "status": "available",
        "kind": "area",
    },
    {
        "id": "loitering",
        "label": "Loitering",
        "typical_need": "Person stays in an area too long",
        "fit": "High — shops, alleys, lobbies",
        "category": "People & safety",
        "status": "coming_soon",
        "kind": "area",
    },
    {
        "id": "crowding",
        "label": "Crowding / occupancy",
        "typical_need": "Too many people in a zone",
        "fit": "High — queues, small shops",
        "category": "People & safety",
        "status": "coming_soon",
        "kind": "count",
    },
    {
        "id": "line_crossing",
        "label": "Line crossing / tripwire",
        "typical_need": "Someone crosses a line either way",
        "fit": "High — simpler than full gate setup",
        "category": "Entrance",
        "status": "coming_soon",
        "kind": "area",
    },
    {
        "id": "gate_analytics",
        "label": "Gate / entrance analytics",
        "typical_need": "Count line, distance zones, open/close",
        "fit": "High — main gates and driveways",
        "category": "Entrance",
        "status": "available",
        "kind": "count",
    },
    {
        "id": "vehicle",
        "label": "Vehicle / number plate",
        "typical_need": "Read plates; approve ours in Vehicles tab",
        "fit": "High — parking, gates; learn then alert unknowns",
        "category": "Vehicles",
        "status": "available",
        "kind": "identity",
    },
    {
        "id": "wrong_way",
        "label": "Wrong-way / vehicle direction",
        "typical_need": "Vehicle moving the wrong way",
        "fit": "Medium — one-way drives, exits",
        "category": "Vehicles",
        "status": "coming_soon",
        "kind": "area",
    },
    {
        "id": "face_attendance",
        "label": "Face / staff attendance",
        "typical_need": "Staff presence; approve faces in Staff tab",
        "fit": "High — staff check-in; unknowns alert later",
        "category": "Staff",
        "status": "available",
        "kind": "identity",
    },
    {
        "id": "object_left",
        "label": "Object left / abandoned",
        "typical_need": "Bag or object left behind",
        "fit": "Medium — shops, lobbies",
        "category": "Property",
        "status": "coming_soon",
        "kind": "area",
    },
    {
        "id": "object_removed",
        "label": "Object removed",
        "typical_need": "Watched item goes missing",
        "fit": "Medium — displays, desks",
        "category": "Property",
        "status": "coming_soon",
        "kind": "area",
    },
    {
        "id": "animal_pet",
        "label": "Animal / pet",
        "typical_need": "Pet or animal in an area (counts in Reports later)",
        "fit": "Medium — homes; light rule now",
        "category": "Property",
        "status": "coming_soon",
        "kind": "count",
    },
    {
        "id": "fire_smoke",
        "label": "Fire / smoke",
        "typical_need": "Smoke or fire in view",
        "fit": "High for safety — later pipeline",
        "category": "Safety gear",
        "status": "coming_soon",
        "kind": "area",
    },
    {
        "id": "shoplifting",
        "label": "Shoplifting / unusual motion",
        "typical_need": "Suspicious retail motion",
        "fit": "Medium — retail; later pipeline",
        "category": "Safety gear",
        "status": "coming_soon",
        "kind": "area",
    },
    {
        "id": "ppe",
        "label": "PPE (helmet / vest)",
        "typical_need": "Missing helmet or high-vis vest",
        "fit": "High — factories and sites",
        "category": "Safety gear",
        "status": "coming_soon",
        "kind": "area",
    },
]

_CATALOG_BY_ID = {e["id"]: e for e in SCAN_CATALOG}

_monitored_camera_ids: set[str] = set()


def catalog_entry(scan_type: str) -> Optional[Dict[str, Any]]:
    return _CATALOG_BY_ID.get(scan_type)


def is_scan_available(scan_type: str) -> bool:
    entry = catalog_entry(scan_type)
    if entry:
        return entry.get("status") == "available" and scan_type in SCAN_TO_PIPELINE
    return scan_type in SCAN_TO_PIPELINE


def available_scan_types() -> List[str]:
    return [e["id"] for e in SCAN_CATALOG if e.get("status") == "available" and e["id"] in SCAN_TO_PIPELINE]


def enabled_rules_for_camera(camera_id: str, exclude_rule_id: str = "") -> List[Dict[str, Any]]:
    return [
        r
        for r in store.list_rules()
        if r.get("camera_id") == camera_id
        and r.get("enabled", True)
        and (r.get("id") or "") != exclude_rule_id
    ]


def camera_rule_limit_reached(camera_id: str, exclude_rule_id: str = "") -> bool:
    return len(enabled_rules_for_camera(camera_id, exclude_rule_id)) >= MAX_RULES_PER_CAMERA


def input_for_camera(cam: Dict[str, Any]) -> Optional[Any]:
    kind = cam.get("type") or "rtsp"
    if kind == "file":
        filename = cam.get("filename") or ""
        path = os.path.join(UPLOAD_DIR, filename)
        if os.path.isfile(path):
            return path
        return None
    if kind == "dvr":
        from core.snapshot_camera import SnapshotCamera

        snapshot_url = (cam.get("snapshot_url") or "").strip()
        if not snapshot_url:
            return None
        return SnapshotCamera(
            snapshot_url,
            user=(cam.get("http_user") or "").strip(),
            password=(cam.get("http_password") or ""),
        )
    url = (cam.get("rtsp_url") or "").strip()
    return url or None


def needs_roi(scan_type: str) -> bool:
    return scan_type in ("intrusion", "danger_zone")


# COCO-17 keypoint groups for YOLOv8-pose (client-selectable rule triggers)
POSE_PART_INDICES: Dict[str, List[int]] = {
    "head": [0, 1, 2, 3, 4],
    "hands": [7, 8, 9, 10],  # elbows + wrists
    "legs": [13, 14, 15, 16],  # knees + ankles
    "torso": [5, 6, 11, 12],  # shoulders + hips
    "whole": list(range(17)),
}

POSE_PART_LABELS: Dict[str, str] = {
    "head": "Head",
    "hands": "Hands",
    "legs": "Legs",
    "torso": "Torso",
    "whole": "Whole person",
}

POSE_KPT_NAMES: Dict[int, str] = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}


def default_pose_parts(scan_type: str) -> List[str]:
    """Backward-compatible defaults matching previous hard-coded behavior."""
    if scan_type == "intrusion":
        return ["legs"]
    if scan_type == "danger_zone":
        return ["whole"]
    return ["whole"]


def normalize_pose_trigger(scan_type: str, pose_trigger: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Normalize / default pose_trigger for intrusion and danger_zone rules."""
    raw = pose_trigger if isinstance(pose_trigger, dict) else {}
    parts_in = raw.get("parts")
    if not isinstance(parts_in, list) or not parts_in:
        parts = default_pose_parts(scan_type)
    else:
        parts = []
        for p in parts_in:
            key = str(p or "").strip().lower()
            if key in POSE_PART_INDICES and key not in parts:
                parts.append(key)
        if not parts:
            parts = default_pose_parts(scan_type)
    try:
        min_conf = float(raw.get("min_conf", 0.5))
    except (TypeError, ValueError):
        min_conf = 0.5
    min_conf = max(0.15, min(0.95, min_conf))
    return {
        "parts": parts,
        "min_conf": min_conf,
        "require_person": True,
    }


def pose_trigger_indices(pose_trigger: Dict[str, Any]) -> List[int]:
    """Unique sorted keypoint indices for the selected body-part groups."""
    seen = set()
    out: List[int] = []
    for part in pose_trigger.get("parts") or []:
        for idx in POSE_PART_INDICES.get(part) or []:
            if idx not in seen:
                seen.add(idx)
                out.append(idx)
    return out


def is_counter_scan(scan_type: str) -> bool:
    return scan_type == "gate_analytics"


def needs_gate_config(scan_type: str) -> bool:
    return scan_type == "gate_analytics"


def gate_config_valid(gate_config: Dict[str, Any]) -> bool:
    if not gate_config:
        return False
    count_line = gate_config.get("count_line") or []
    gate_roi = gate_config.get("gate_roi") or []
    zones = gate_config.get("distance_zones") or {}
    if len(count_line) != 2:
        return False
    if len(gate_roi) < 3:
        return False
    for band in ("near", "medium", "far"):
        if len(zones.get(band) or []) < 3:
            return False
    return True


def pipeline_config(scan_type: str) -> Dict[str, Any]:
    if scan_type == "face_attendance":
        return {"mode": "attendance"}
    if scan_type == "danger_zone":
        return {"machine_active": True}
    if scan_type == "fall_standing_lying":
        return {
            "person_conf_threshold": 0.70,
            "keypoint_conf_threshold": 0.4,
            "sit_angle_deg": 30.0,
            "down_torso_angle_deg": 40.0,
            "down_hip_ankle_ratio": 0.45,
            "upright_memory_sec": 45.0,
            "match_dist_ratio": 0.25,
            "confirm_snapshots": 2,
        }
    return {}


def in_quiet_hours(site: Dict[str, Any]) -> bool:
    start = (site.get("quiet_hours_start") or "").strip()
    end = (site.get("quiet_hours_end") or "").strip()
    if not start or not end:
        return False
    try:
        now = time.strftime("%H:%M")
        if start <= end:
            return start <= now < end
        return now >= start or now < end
    except Exception:
        return False


def is_event_dict(event: Any) -> bool:
    if not isinstance(event, dict) or not event:
        return False
    return bool(
        event.get("clip_url")
        or event.get("type")
        or event.get("id")
        or event.get("person_id")
        or event.get("plate")
        or (event.get("detections") and len(event.get("detections", [])) > 0)
    )


def _save_frame_thumb(frame: Any, prefix: str, folder: str) -> str:
    """Write a JPEG thumb; return public /storage/... URL or empty string."""
    if frame is None:
        return ""
    try:
        import cv2
        import numpy as np

        if not isinstance(frame, np.ndarray) or frame.size == 0:
            return ""
        os.makedirs(folder, exist_ok=True)
        tid = prefix or str(int(time.time() * 1000))
        thumb_fn = f"thumb_{tid}.jpg"
        thumb_path = os.path.join(folder, thumb_fn)
        snap = frame.copy()
        h, w = snap.shape[:2]
        max_w = 480
        if w > max_w:
            scale = max_w / float(w)
            snap = cv2.resize(snap, (int(w * scale), int(h * scale)))
        cv2.imwrite(thumb_path, snap, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        rel = folder.replace("\\", "/").lstrip("./")
        return f"/{rel}/{thumb_fn}".replace("//", "/")
    except Exception:
        return ""


def record_journey_from_hit(
    cam: Dict[str, Any],
    rule: Dict[str, Any],
    event: Dict[str, Any],
    frame: Any = None,
) -> Optional[Dict[str, Any]]:
    """Upsert cross-camera journey from a face or vehicle hit (alerts optional)."""
    try:
        from core import journey_store

        scan_type = rule.get("scan_type") or event.get("type") or ""
        camera_id = cam.get("id") or ""
        camera_name = cam.get("name") or camera_id
        rule_id = rule.get("id") or ""
        thumb = event.get("thumb_url") or ""
        track_id = event.get("track_id") or event.get("local_track_id")
        bbox = event.get("bbox") or event.get("xyxy") or []

        person_id = event.get("person_id") or ""
        plate = event.get("plate") or ""
        if not plate and scan_type == "vehicle":
            plate = event.get("label") or ""

        merge_gid = ""
        if track_id is not None and camera_id:
            merge_gid = journey_store.lookup_local_track(camera_id, track_id) or ""

        if person_id or scan_type in ("face_attendance",) or event.get("type") == "face_recognised":
            if not person_id:
                return None
            label = event.get("label") or person_id
            # Prefer Staff gallery display name when approved
            try:
                for face in store.list_known_faces(200):
                    if (face.get("person_id") or face.get("id")) == person_id:
                        if (face.get("status") or "") == "approved" and face.get("label"):
                            label = face["label"]
                        break
            except Exception:
                pass
            meta = journey_store.upsert_face_sighting(
                person_id=person_id,
                label=label,
                camera_id=camera_id,
                camera_name=camera_name,
                rule_id=rule_id,
                scan_type=scan_type or "face_attendance",
                local_track_id=track_id,
                bbox=bbox if isinstance(bbox, list) else [],
                thumb_url=thumb,
                merge_from_gid=merge_gid,
            )
            if meta and track_id is not None:
                journey_store.attach_local_track(camera_id, track_id, meta.get("gid") or "")
            return meta

        if plate or scan_type == "vehicle":
            key = store.normalize_plate(plate)
            if not key:
                return None
            label = key
            try:
                veh = store.get_known_vehicle_by_plate(key)
                if veh and (veh.get("status") or "") == "approved" and veh.get("label"):
                    label = veh["label"]
            except Exception:
                pass
            meta = journey_store.upsert_plate_sighting(
                plate=key,
                label=label,
                camera_id=camera_id,
                camera_name=camera_name,
                rule_id=rule_id,
                scan_type=scan_type or "vehicle",
                local_track_id=track_id,
                bbox=bbox if isinstance(bbox, list) else [],
                thumb_url=thumb,
                merge_from_gid=merge_gid,
            )
            if meta and track_id is not None:
                journey_store.attach_local_track(camera_id, track_id, meta.get("gid") or "")
            return meta
    except Exception:
        return None
    return None


def emit_site_alert(
    cam: Dict[str, Any],
    rule: Dict[str, Any],
    event: Dict[str, Any],
    frame: Any = None,
    *,
    skip_cooldown: bool = False,
) -> Optional[Dict[str, Any]]:
    record_journey_from_hit(cam, rule, event, frame=frame)
    rule_id = rule.get("id") or ""
    if store.is_muted(rule_id):
        return None
    if not skip_cooldown and store.on_cooldown(rule_id):
        return None
    scan_type = rule.get("scan_type") or "unknown"
    label = event.get("type") or scan_type
    alert_id = event.get("id") or ""
    thumb_url = event.get("thumb_url") or ""
    msg = event.get("message") or (
        f"{cam.get('name') or 'Camera'}: {rule.get('name') or scan_type.replace('_', ' ')} "
        f"— {scan_type.replace('_', ' ')}"
    )

    alert = {
        "camera_id": cam.get("id"),
        "camera_name": cam.get("name") or cam.get("id"),
        "rule_id": rule_id,
        "scan_type": scan_type,
        "type": label,
        "severity": event.get("severity") or rule.get("severity") or "high",
        "clip_url": event.get("clip_url"),
        "thumb_url": thumb_url,
        "person_id": event.get("person_id"),
        "label": event.get("label") or event.get("plate"),
        "channels": rule.get("channels") or ["web"],
        "message": msg,
    }
    stored = store.append_alert(alert)

    if not stored.get("thumb_url") and frame is not None:
        thumb_url = _save_frame_thumb(frame, stored.get("id") or alert_id or "snap", ALERTS_DIR)
        if thumb_url:
            stored = store.update_alert(stored["id"], {"thumb_url": thumb_url}) or stored

    if not skip_cooldown:
        store.set_cooldown(rule_id, int(rule.get("cooldown_sec") or 60))
    channels = stored.get("channels") or []
    if not in_quiet_hours(store.get_site()):
        site = store.get_site()
        body = stored["message"]
        if stored.get("clip_url"):
            body += f"\nClip: {stored['clip_url']}"
        if "whatsapp" in channels:
            whatsapp_adapter.notify_numbers(site.get("whatsapp_numbers") or [], body)
        if "email" in channels:
            email_adapter.notify_alert_emails(
                site.get("alert_emails") or [],
                stored,
                site_name=str(site.get("name") or "").strip(),
            )
    return stored


def handle_vehicle_sighting(
    cam: Dict[str, Any],
    rule: Dict[str, Any],
    event: Dict[str, Any],
    frame: Any = None,
) -> Optional[Dict[str, Any]]:
    """
    Upsert plate into Vehicles gallery and alert by status:
    - new candidate → one unknown alert
    - existing candidate → silent (refresh thumb/seen)
    - approved / ignored → silent
    - risk / danger → immediate high alert (bypass cooldown)
    """
    plate = event.get("plate") or event.get("label") or ""
    key = store.normalize_plate(plate)
    if not key:
        return None

    thumb_url = event.get("thumb_url") or ""
    if not thumb_url and frame is not None:
        thumb_url = _save_frame_thumb(frame, f"veh_{key}", VEHICLES_DIR)

    # Journey trail updates even when alerts are suppressed (approved / cooldown).
    record_journey_from_hit(
        cam,
        rule,
        {**event, "plate": key, "thumb_url": thumb_url or event.get("thumb_url") or ""},
        frame=frame,
    )

    record = store.upsert_known_vehicle_by_plate(
        {
            "plate": key,
            "thumb_url": thumb_url,
            "clip_url": event.get("clip_url") or "",
            "camera_id": cam.get("id") or "",
            "label": event.get("label") or "",
            "status": "candidate",
        }
    )
    status = (record.get("status") or "candidate").lower()
    created = bool(record.get("_created"))
    seen = int(record.get("seen_count") or 0)

    if status in ("approved", "ignored"):
        return None

    if status in ("risk", "danger"):
        # Bypass rule cooldown, but avoid alerting every video frame.
        plate_cd = f"plate_risk:{key}"
        if store.on_cooldown(plate_cd):
            return None
        sev = "critical" if status == "danger" else "high"
        stored = emit_site_alert(
            cam,
            rule,
            {
                "type": "vehicle_risk",
                "plate": key,
                "label": key,
                "severity": sev,
                "thumb_url": record.get("thumb_url") or thumb_url,
                "clip_url": record.get("clip_url") or event.get("clip_url"),
                "message": (
                    f"{cam.get('name') or 'Camera'}: RISK vehicle {key} "
                    f"({status}) — immediate alert"
                ),
            },
            frame=frame,
            skip_cooldown=True,
        )
        if stored:
            store.set_cooldown(plate_cd, 45)
        return stored

    # candidate / unknown: alert only on first sighting
    if created or seen <= 1:
        return emit_site_alert(
            cam,
            rule,
            {
                "type": "vehicle_unknown",
                "plate": key,
                "label": key,
                "severity": event.get("severity") or rule.get("severity") or "medium",
                "thumb_url": record.get("thumb_url") or thumb_url,
                "clip_url": record.get("clip_url") or event.get("clip_url"),
                "message": (
                    f"{cam.get('name') or 'Camera'}: Unknown plate {key} — "
                    f"review in Vehicles"
                ),
            },
            frame=frame,
            skip_cooldown=False,
        )
    return None


def mark_camera_monitored(camera_id: str) -> None:
    _monitored_camera_ids.add(camera_id)
    store.mark_camera_monitored_redis(camera_id)


def unmark_camera_monitored(camera_id: str) -> None:
    _monitored_camera_ids.discard(camera_id)
    store.unmark_camera_monitored_redis(camera_id)


def is_camera_monitored(camera_id: str) -> bool:
    return camera_id in _monitored_camera_ids or store.is_camera_monitored_redis(camera_id)


def monitored_camera_ids() -> set[str]:
    return set(_monitored_camera_ids)
