"""Redis-backed Site Admin config, alerts, and reports. Isolated key prefixes."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, List, Optional

from core.redis_client import get_redis, redis_str

SITE_KEY = "site:profile"
CAM_INDEX = "cam:index"
RULE_INDEX = "rule:index"
ALERT_LIST = "alert:inbox"
MUTE_PREFIX = "alert:mute:"
COOLDOWN_PREFIX = "alert:cd:"
LIVE_KEY = "site:live"
KNOWN_FACES_LIST = "site:known_faces"
KNOWN_VEHICLES_LIST = "site:known_vehicles"


def _dumps(obj: Any) -> bytes:
    return json.dumps(obj, separators=(",", ":")).encode()


def _loads(raw: Any) -> Optional[Dict[str, Any]]:
    if raw is None:
        return None
    try:
        return json.loads(redis_str(raw))
    except json.JSONDecodeError:
        return None


def default_site() -> Dict[str, Any]:
    return {
        "name": "",
        "type": "shop",
        "timezone": "Asia/Kolkata",
        "quiet_hours_start": "",
        "quiet_hours_end": "",
        "setup_complete": False,
        "wizard_step": 0,
        "admin_pin": "",
        "whatsapp_numbers": [],
        "go_live": False,
        "updated_at": time.time(),
    }


def get_site() -> Dict[str, Any]:
    data = _loads(get_redis().get(SITE_KEY))
    if not data:
        data = default_site()
        save_site(data)
    return data


def save_site(site: Dict[str, Any]) -> Dict[str, Any]:
    merged = default_site()
    merged.update(site)
    merged["updated_at"] = time.time()
    get_redis().set(SITE_KEY, _dumps(merged))
    return merged


def _cam_key(cam_id: str) -> str:
    return f"cam:{cam_id}"


def list_cameras() -> List[Dict[str, Any]]:
    r = get_redis()
    ids = [redis_str(x) for x in r.smembers(CAM_INDEX)]
    out: List[Dict[str, Any]] = []
    for cid in ids:
        item = _loads(r.get(_cam_key(cid)))
        if item:
            out.append(item)
    out.sort(key=lambda c: c.get("created_at", 0))
    return out


def get_camera(cam_id: str) -> Optional[Dict[str, Any]]:
    return _loads(get_redis().get(_cam_key(cam_id)))


def save_camera(cam: Dict[str, Any]) -> Dict[str, Any]:
    if not cam.get("id"):
        cam["id"] = str(uuid.uuid4())
        cam["created_at"] = time.time()
    cam["updated_at"] = time.time()
    r = get_redis()
    r.set(_cam_key(cam["id"]), _dumps(cam))
    r.sadd(CAM_INDEX, cam["id"])
    return cam


def delete_camera(cam_id: str) -> None:
    r = get_redis()
    r.delete(_cam_key(cam_id))
    r.srem(CAM_INDEX, cam_id)
    for rule in list_rules():
        if rule.get("camera_id") == cam_id:
            delete_rule(rule["id"])


def _rule_key(rule_id: str) -> str:
    return f"rule:{rule_id}"


def list_rules() -> List[Dict[str, Any]]:
    r = get_redis()
    ids = [redis_str(x) for x in r.smembers(RULE_INDEX)]
    out: List[Dict[str, Any]] = []
    for rid in ids:
        item = _loads(r.get(_rule_key(rid)))
        if item:
            out.append(item)
    out.sort(key=lambda c: c.get("created_at", 0))
    return out


def get_rule(rule_id: str) -> Optional[Dict[str, Any]]:
    return _loads(get_redis().get(_rule_key(rule_id)))


def save_rule(rule: Dict[str, Any]) -> Dict[str, Any]:
    if not rule.get("id"):
        rule["id"] = str(uuid.uuid4())
        rule["created_at"] = time.time()
    rule["updated_at"] = time.time()
    r = get_redis()
    r.set(_rule_key(rule["id"]), _dumps(rule))
    r.sadd(RULE_INDEX, rule["id"])
    return rule


def delete_rule(rule_id: str) -> None:
    r = get_redis()
    r.delete(_rule_key(rule_id))
    r.srem(RULE_INDEX, rule_id)


def append_alert(alert: Dict[str, Any]) -> Dict[str, Any]:
    if not alert.get("id"):
        alert["id"] = str(uuid.uuid4())
    alert.setdefault("created_at", time.time())
    alert.setdefault("acked", False)
    get_redis().lpush(ALERT_LIST, _dumps(alert))
    get_redis().ltrim(ALERT_LIST, 0, 499)
    return alert


def list_alerts(limit: int = 100) -> List[Dict[str, Any]]:
    items = get_redis().lrange(ALERT_LIST, 0, max(0, limit - 1))
    out: List[Dict[str, Any]] = []
    for raw in items:
        item = _loads(raw)
        if item:
            out.append(item)
    return out


def update_alert(alert_id: str, patch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    r = get_redis()
    items = r.lrange(ALERT_LIST, 0, 499)
    for i, raw in enumerate(items):
        item = _loads(raw)
        if item and item.get("id") == alert_id:
            item.update(patch)
            r.lset(ALERT_LIST, i, _dumps(item))
            return item
    return None


def delete_alert(alert_id: str) -> bool:
    r = get_redis()
    items = r.lrange(ALERT_LIST, 0, 499)
    for raw in items:
        item = _loads(raw)
        if item and item.get("id") == alert_id:
            r.lrem(ALERT_LIST, 1, raw)
            return True
    return False


def mute_rule(rule_id: str, seconds: int = 3600) -> None:
    get_redis().setex(f"{MUTE_PREFIX}{rule_id}", seconds, b"1")


def is_muted(rule_id: str) -> bool:
    return bool(get_redis().exists(f"{MUTE_PREFIX}{rule_id}"))


def set_cooldown(rule_id: str, seconds: int = 60) -> None:
    get_redis().setex(f"{COOLDOWN_PREFIX}{rule_id}", seconds, b"1")


def on_cooldown(rule_id: str) -> bool:
    return bool(get_redis().exists(f"{COOLDOWN_PREFIX}{rule_id}"))


GATE_COUNTER_FIELDS = (
    "gate_opens",
    "gate_closes",
    "persons_in",
    "persons_out",
    "cars_in",
    "cars_out",
    "bikes_in",
    "bikes_out",
    "near_events",
    "medium_events",
    "far_events",
)


def _gate_totals_key(rule_id: str) -> str:
    return f"gate:totals:{rule_id}"


def _gate_hourly_key(rule_id: str, hour_bucket: str) -> str:
    return f"gate:hourly:{rule_id}:{hour_bucket}"


def _gate_events_key(rule_id: str) -> str:
    return f"gate:events:{rule_id}"


def _gate_baseline_key(rule_id: str) -> str:
    return f"gate:baseline:{rule_id}"


def _empty_gate_counters() -> Dict[str, int]:
    return {f: 0 for f in GATE_COUNTER_FIELDS}


def increment_gate_counters(rule_id: str, deltas: Dict[str, int]) -> None:
    if not rule_id or not deltas:
        return
    r = get_redis()
    hour_bucket = time.strftime("%Y%m%d%H", time.localtime())
    pipe = r.pipeline()
    for field, val in deltas.items():
        if field not in GATE_COUNTER_FIELDS:
            continue
        n = int(val or 0)
        if n == 0:
            continue
        pipe.hincrby(_gate_totals_key(rule_id), field, n)
        pipe.hincrby(_gate_hourly_key(rule_id, hour_bucket), field, n)
    pipe.execute()


def append_gate_event(rule_id: str, kind: str, detail: str = "") -> None:
    r = get_redis()
    evt = {"ts": time.time(), "kind": kind, "detail": detail}
    r.lpush(_gate_events_key(rule_id), _dumps(evt))
    r.ltrim(_gate_events_key(rule_id), 0, 499)


def get_gate_totals(rule_id: str) -> Dict[str, int]:
    raw = get_redis().hgetall(_gate_totals_key(rule_id))
    out = _empty_gate_counters()
    for field in GATE_COUNTER_FIELDS:
        val = raw.get(field.encode()) or raw.get(field)
        if val is not None:
            try:
                out[field] = int(redis_str(val))
            except (TypeError, ValueError):
                out[field] = 0
    return out


def set_gate_baseline(rule_id: str, score: float) -> None:
    get_redis().set(_gate_baseline_key(rule_id), str(float(score)).encode())


def get_gate_baseline(rule_id: str) -> Optional[float]:
    raw = get_redis().get(_gate_baseline_key(rule_id))
    if raw is None:
        return None
    try:
        return float(redis_str(raw))
    except (TypeError, ValueError):
        return None


def _sum_hourly_hash(data: Dict[Any, Any]) -> Dict[str, int]:
    out = _empty_gate_counters()
    for field in GATE_COUNTER_FIELDS:
        val = data.get(field.encode()) or data.get(field)
        if val is not None:
            try:
                out[field] = int(redis_str(val))
            except (TypeError, ValueError):
                pass
    return out


def gate_report_summary(
    since_ts: float,
    until_ts: float,
    rule_id: Optional[str] = None,
) -> Dict[str, Any]:
    rules = list_rules()
    gate_rules = [r for r in rules if r.get("scan_type") == "gate_analytics"]
    if rule_id:
        gate_rules = [r for r in gate_rules if r.get("id") == rule_id]

    per_rule: List[Dict[str, Any]] = []
    totals = _empty_gate_counters()
    hourly: List[Dict[str, Any]] = []

    r = get_redis()
    for rule in gate_rules:
        rid = rule.get("id") or ""
        if not rid:
            continue
        rule_totals = get_gate_totals(rid)
        per_rule.append({
            "rule_id": rid,
            "rule_name": rule.get("name"),
            "camera_id": rule.get("camera_id"),
            "totals": rule_totals,
        })
        for field in GATE_COUNTER_FIELDS:
            totals[field] += rule_totals.get(field, 0)

        t = since_ts
        while t <= until_ts:
            bucket = time.strftime("%Y%m%d%H", time.localtime(t))
            hdata = r.hgetall(_gate_hourly_key(rid, bucket))
            if hdata:
                summed = _sum_hourly_hash(hdata)
                if any(summed.values()):
                    hourly.append({
                        "rule_id": rid,
                        "hour": bucket,
                        "counters": summed,
                    })
            t += 3600

    return {
        "from": since_ts,
        "to": until_ts,
        "rules": per_rule,
        "totals": totals,
        "hourly": hourly,
    }


def report_summary(since_ts: float, until_ts: float) -> Dict[str, Any]:
    alerts = [a for a in list_alerts(500) if since_ts <= float(a.get("created_at", 0)) <= until_ts]
    by_type: Dict[str, int] = {}
    by_camera: Dict[str, int] = {}
    for a in alerts:
        t = a.get("scan_type") or a.get("type") or "unknown"
        by_type[t] = by_type.get(t, 0) + 1
        cid = a.get("camera_name") or a.get("camera_id") or "unknown"
        by_camera[cid] = by_camera.get(cid, 0) + 1
    attendance = [a for a in alerts if a.get("scan_type") == "face_attendance"]
    people = sorted({a.get("person_id") or a.get("label") or "unknown" for a in attendance})
    return {
        "from": since_ts,
        "to": until_ts,
        "alert_count": len(alerts),
        "by_type": by_type,
        "by_camera": by_camera,
        "attendance_events": len(attendance),
        "attendance_people": people,
        "alerts": alerts[:200],
    }


def _list_identity(key: str, limit: int = 200) -> List[Dict[str, Any]]:
    items = get_redis().lrange(key, 0, max(0, limit - 1))
    out: List[Dict[str, Any]] = []
    for raw in items:
        item = _loads(raw)
        if item:
            out.append(item)
    return out


def _upsert_identity(key: str, item: Dict[str, Any], max_keep: int = 500) -> Dict[str, Any]:
    if not item.get("id"):
        item["id"] = str(uuid.uuid4())
    item.setdefault("created_at", time.time())
    item["updated_at"] = time.time()
    r = get_redis()
    items = r.lrange(key, 0, max_keep - 1)
    for i, raw in enumerate(items):
        cur = _loads(raw)
        if cur and cur.get("id") == item["id"]:
            cur.update(item)
            cur["updated_at"] = time.time()
            r.lset(key, i, _dumps(cur))
            return cur
    r.lpush(key, _dumps(item))
    r.ltrim(key, 0, max_keep - 1)
    return item


def _patch_identity(key: str, item_id: str, patch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    r = get_redis()
    items = r.lrange(key, 0, 499)
    for i, raw in enumerate(items):
        item = _loads(raw)
        if item and item.get("id") == item_id:
            item.update(patch)
            item["updated_at"] = time.time()
            r.lset(key, i, _dumps(item))
            return item
    return None


def list_known_faces(limit: int = 200, status: Optional[str] = None) -> List[Dict[str, Any]]:
    items = _list_identity(KNOWN_FACES_LIST, limit)
    if status:
        items = [x for x in items if (x.get("status") or "candidate") == status]
    return items


def save_known_face(item: Dict[str, Any]) -> Dict[str, Any]:
    item.setdefault("status", "candidate")
    item.setdefault("label", item.get("label") or "Unknown")
    return _upsert_identity(KNOWN_FACES_LIST, item)


def patch_known_face(item_id: str, patch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    return _patch_identity(KNOWN_FACES_LIST, item_id, patch)


def list_known_vehicles(limit: int = 200, status: Optional[str] = None) -> List[Dict[str, Any]]:
    items = _list_identity(KNOWN_VEHICLES_LIST, limit)
    if status:
        items = [x for x in items if (x.get("status") or "candidate") == status]
    return items


def normalize_plate(plate: Any) -> str:
    """Uppercase plate key without spaces/dashes for stable identity."""
    s = str(plate or "").upper()
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
    return "".join(out)


def get_known_vehicle_by_plate(plate: str) -> Optional[Dict[str, Any]]:
    key = normalize_plate(plate)
    if not key:
        return None
    for item in list_known_vehicles(500):
        if normalize_plate(item.get("plate") or "") == key:
            return item
    return None


def upsert_known_vehicle_by_plate(fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    Find-or-create by normalized plate. Returns the stored record.
    Sets `_created` True on the returned dict when a new row was inserted.
    """
    plate_raw = fields.get("plate") or fields.get("label") or ""
    key = normalize_plate(plate_raw)
    if not key:
        raise ValueError("Empty plate")
    now = time.time()
    existing = get_known_vehicle_by_plate(key)
    if existing:
        patch: Dict[str, Any] = {
            "plate": key,
            "last_seen_at": now,
            "seen_count": int(existing.get("seen_count") or 0) + 1,
            "updated_at": now,
        }
        if fields.get("thumb_url"):
            patch["thumb_url"] = fields["thumb_url"]
        if fields.get("clip_url"):
            patch["clip_url"] = fields["clip_url"]
        if fields.get("camera_id"):
            patch["camera_id"] = fields["camera_id"]
        if fields.get("label") and not existing.get("label"):
            patch["label"] = fields["label"]
        updated = patch_known_vehicle(existing["id"], patch) or existing
        updated["_created"] = False
        return updated

    item = {
        "id": str(uuid.uuid4()),
        "plate": key,
        "label": fields.get("label") or "",
        "thumb_url": fields.get("thumb_url") or "",
        "clip_url": fields.get("clip_url") or "",
        "camera_id": fields.get("camera_id") or "",
        "status": fields.get("status") or "candidate",
        "note": fields.get("note") or "",
        "first_seen_at": now,
        "last_seen_at": now,
        "seen_count": 1,
        "created_at": now,
        "updated_at": now,
    }
    saved = _upsert_identity(KNOWN_VEHICLES_LIST, item)
    saved["_created"] = True
    return saved


def save_known_vehicle(item: Dict[str, Any]) -> Dict[str, Any]:
    item.setdefault("status", "candidate")
    plate = normalize_plate(item.get("plate") or item.get("label") or "UNKNOWN")
    item["plate"] = plate or "UNKNOWN"
    if plate and plate != "UNKNOWN":
        existing = get_known_vehicle_by_plate(plate)
        if existing:
            item["id"] = existing["id"]
            item.setdefault("first_seen_at", existing.get("first_seen_at"))
            item["seen_count"] = int(existing.get("seen_count") or 0) + 1
            item["last_seen_at"] = time.time()
    return _upsert_identity(KNOWN_VEHICLES_LIST, item)


def patch_known_vehicle(item_id: str, patch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if "plate" in patch and patch["plate"] is not None:
        patch = {**patch, "plate": normalize_plate(patch["plate"]) or patch["plate"]}
    return _patch_identity(KNOWN_VEHICLES_LIST, item_id, patch)
