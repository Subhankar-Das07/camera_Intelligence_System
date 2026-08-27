"""Cross-camera journey store: global IDs linking face, plate, and anonymous Re-ID sightings."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.redis_client import get_redis, redis_str

GID_PREFIX = "G"
META_PREFIX = "journey:meta:"
TIMELINE_PREFIX = "journey:timeline:"
ACTIVE_KEY = "journey:active"
BY_FACE_PREFIX = "journey:by_face:"
BY_PLATE_PREFIX = "journey:by_plate:"
REID_EMB_PREFIX = "journey:reid:emb:"
COOLDOWN_PREFIX = "journey:cd:"
WATCH_SET = "journey:watched"
WATCH_ALERT_CD_PREFIX = "journey:watch_alert_cd:"
SEQ_KEY = "journey:seq"

HANDOFF_SEC = float(os.environ.get("JOURNEY_HANDOFF_SEC", "120"))
SIGHTING_COOLDOWN_SEC = float(os.environ.get("JOURNEY_SIGHTING_COOLDOWN_SEC", "8"))
WATCH_ALERT_COOLDOWN_SEC = float(os.environ.get("JOURNEY_WATCH_ALERT_COOLDOWN_SEC", "15"))
TIMELINE_TRIM = 200
REID_DIM = 512

_lock = threading.RLock()


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)


def _dumps(obj: Any) -> bytes:
    return json.dumps(obj, separators=(",", ":"), cls=_NumpyEncoder).encode()


def _loads(raw: Any) -> Optional[Dict[str, Any]]:
    if raw is None:
        return None
    try:
        return json.loads(redis_str(raw))
    except (json.JSONDecodeError, TypeError):
        return None


def _mint_gid() -> str:
    r = get_redis()
    n = int(r.incr(SEQ_KEY))
    return f"{GID_PREFIX}{n:08d}"


def _meta_key(gid: str) -> str:
    return f"{META_PREFIX}{gid}"


def _timeline_key(gid: str) -> str:
    return f"{TIMELINE_PREFIX}{gid}"


def get_meta(gid: str) -> Optional[Dict[str, Any]]:
    return _loads(get_redis().get(_meta_key(gid)))


def _save_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    r = get_redis()
    gid = meta["gid"]
    r.set(_meta_key(gid), _dumps(meta))
    r.zadd(ACTIVE_KEY, {gid: float(meta.get("last_seen_at") or time.time())})
    return meta


def _default_meta(
    *,
    kind: str,
    label: str = "unknown",
    person_id: str = "",
    plate: str = "",
    match_method: str = "new",
) -> Dict[str, Any]:
    now = time.time()
    return {
        "gid": _mint_gid(),
        "kind": kind,
        "label": label or "unknown",
        "person_id": person_id or "",
        "plate": plate or "",
        "created_at": now,
        "last_seen_at": now,
        "last_camera_id": "",
        "last_camera_name": "",
        "match_method": match_method,
        "merged_into": "",
    }


def _on_cooldown(gid: str, camera_id: str) -> bool:
    key = f"{COOLDOWN_PREFIX}{gid}:{camera_id or '_'}"
    return bool(get_redis().exists(key))


def _set_cooldown(gid: str, camera_id: str) -> None:
    key = f"{COOLDOWN_PREFIX}{gid}:{camera_id or '_'}"
    get_redis().setex(key, max(1, int(SIGHTING_COOLDOWN_SEC)), b"1")


def _latest_thumb_url(gid: str, meta: Optional[Dict[str, Any]] = None) -> str:
    """Prefer meta last_thumb_url; else newest timeline event with a thumb."""
    m = meta if meta is not None else (get_meta(gid) or {})
    thumb = (m.get("last_thumb_url") or m.get("thumb_url") or "").strip()
    if thumb:
        return thumb
    try:
        for raw in get_redis().lrange(_timeline_key(gid), 0, 24):
            ev = _loads(raw)
            if not isinstance(ev, dict):
                continue
            t = (ev.get("thumb_url") or "").strip()
            if t:
                return t
    except Exception:
        pass
    return ""


def _append_sighting(gid: str, event: Dict[str, Any], *, force: bool = False) -> bool:
    """Append timeline event; return False if cooldown skipped the write."""
    camera_id = event.get("camera_id") or ""
    meta_before = get_meta(gid) or {}
    prev_cam = meta_before.get("last_camera_id") or ""
    event_thumb = (event.get("thumb_url") or "").strip()

    if not force and _on_cooldown(gid, camera_id):
        # Still refresh last_seen lightly
        meta = get_meta(gid)
        if meta:
            meta["last_seen_at"] = float(event.get("ts") or time.time())
            if camera_id:
                meta["last_camera_id"] = camera_id
                meta["last_camera_name"] = event.get("camera_name") or meta.get("last_camera_name") or ""
            if event.get("bbox"):
                meta["last_bbox"] = event.get("bbox")
            if event_thumb:
                meta["last_thumb_url"] = event_thumb
                meta["thumb_url"] = event_thumb
            _save_meta(meta)
        # Still alert watchers on camera hop even if timeline deduped
        if camera_id and prev_cam and camera_id != prev_cam and is_watched(gid):
            if event_thumb and not event.get("thumb_url"):
                event = dict(event)
                event["thumb_url"] = event_thumb
            elif not (event.get("thumb_url") or "").strip():
                event = dict(event)
                event["thumb_url"] = _latest_thumb_url(gid, meta or meta_before)
            _emit_watch_alert(gid, event, hop=True, prev_camera_id=prev_cam)
        return False

    r = get_redis()
    event = dict(event)
    event.setdefault("ts", time.time())
    event["gid"] = gid
    r.lpush(_timeline_key(gid), _dumps(event))
    r.ltrim(_timeline_key(gid), 0, TIMELINE_TRIM - 1)
    _set_cooldown(gid, camera_id)

    meta = get_meta(gid) or {}
    meta["last_seen_at"] = float(event["ts"])
    if camera_id:
        meta["last_camera_id"] = camera_id
        meta["last_camera_name"] = event.get("camera_name") or ""
    if event.get("match_method"):
        meta["match_method"] = event["match_method"]
    if event.get("bbox"):
        meta["last_bbox"] = event.get("bbox")
    if event_thumb:
        meta["last_thumb_url"] = event_thumb
        meta["thumb_url"] = event_thumb
    _save_meta(meta)

    if is_watched(gid):
        hop = bool(camera_id and prev_cam and camera_id != prev_cam)
        if not (event.get("thumb_url") or "").strip():
            event["thumb_url"] = _latest_thumb_url(gid, meta)
        _emit_watch_alert(gid, event, hop=hop, prev_camera_id=prev_cam)
    return True


def is_watched(gid: str) -> bool:
    if not gid:
        return False
    return bool(get_redis().sismember(WATCH_SET, gid))


def watch_journey(gid: str) -> Optional[Dict[str, Any]]:
    """Pin a journey for Alerts on every new sighting / camera hop."""
    meta = get_meta(gid)
    if not meta or meta.get("merged_into"):
        if meta and meta.get("merged_into"):
            gid = meta["merged_into"]
            meta = get_meta(gid)
        if not meta:
            return None
    get_redis().sadd(WATCH_SET, gid)
    meta = dict(meta)
    meta["watched"] = True
    thumb = _latest_thumb_url(gid, meta)
    if thumb and not meta.get("last_thumb_url"):
        meta["last_thumb_url"] = thumb
        meta["thumb_url"] = thumb
        _save_meta(meta)
    # Immediate alert so it shows in Alerts
    _emit_watch_alert(
        gid,
        {
            "ts": time.time(),
            "camera_id": meta.get("last_camera_id") or "",
            "camera_name": meta.get("last_camera_name") or "",
            "match_method": "watch",
            "label": meta.get("label") or "unknown",
            "thumb_url": thumb,
        },
        hop=False,
        prev_camera_id="",
        force=True,
        message_override=(
            f"Watching {meta.get('gid')} · {meta.get('label') or 'unknown'} — "
            f"future sightings and camera hops go to Alerts"
        ),
    )
    return meta


def unwatch_journey(gid: str) -> bool:
    if not gid:
        return False
    removed = get_redis().srem(WATCH_SET, gid)
    return bool(removed)


def delete_journey(gid: str) -> bool:
    """Permanently remove a journey: unwatch + meta + timeline + Re-ID embedding."""
    if not gid:
        return False
    meta = get_meta(gid)
    if meta and meta.get("merged_into"):
        gid = str(meta["merged_into"])
        meta = get_meta(gid)
    r = get_redis()
    existed = bool(meta) or bool(r.exists(_meta_key(gid))) or bool(r.exists(_timeline_key(gid))) or bool(
        r.sismember(WATCH_SET, gid)
    )
    unwatch_journey(gid)
    # Best-effort: drop local track maps from recent timeline
    try:
        raws = r.lrange(_timeline_key(gid), 0, 40)
        for raw in raws:
            ev = _loads(raw)
            if not ev:
                continue
            cam = ev.get("camera_id") or ""
            tid = ev.get("local_track_id")
            if cam and tid is not None:
                r.delete(f"journey:track:{cam}:{tid}")
    except Exception:
        pass
    r.delete(_meta_key(gid))
    r.delete(_timeline_key(gid))
    r.delete(f"{REID_EMB_PREFIX}{gid}")
    return existed


def list_watched(*, limit: int = 50) -> List[Dict[str, Any]]:
    r = get_redis()
    gids = [redis_str(x) for x in r.smembers(WATCH_SET)]
    out: List[Dict[str, Any]] = []
    for gid in gids:
        meta = get_meta(gid)
        if not meta or meta.get("merged_into"):
            if meta and meta.get("merged_into"):
                r.srem(WATCH_SET, gid)
                r.sadd(WATCH_SET, meta["merged_into"])
            continue
        item = dict(meta)
        item["watched"] = True
        hops = _recent_hops(gid, limit=4)
        item["hops"] = hops
        item["hop_summary"] = _format_hops(hops)
        thumb = _latest_thumb_url(gid, item)
        if thumb:
            item["thumb_url"] = thumb
            item["last_thumb_url"] = item.get("last_thumb_url") or thumb
            if not meta.get("last_thumb_url"):
                meta["last_thumb_url"] = thumb
                meta["thumb_url"] = thumb
                _save_meta(meta)
        # #region agent log
        try:
            import json as _json
            _line = _json.dumps({"sessionId": "46c418", "hypothesisId": "B", "location": "journey_store.py:list_watched", "message": "watched item thumb fields", "data": {"gid": gid, "has_thumb_url": bool(item.get("thumb_url")), "has_last_thumb_url": bool(item.get("last_thumb_url")), "thumb_len": len(thumb or ""), "meta_keys": sorted([k for k in item.keys() if "thumb" in k.lower() or k in ("gid", "label", "last_camera_name")])}, "timestamp": int(time.time() * 1000), "runId": "post-fix"}) + "\n"
            for _p in ("/app/static/debug-46c418.log", "static/debug-46c418.log", "debug-46c418.log"):
                try:
                    with open(_p, "a", encoding="utf-8") as _f:
                        _f.write(_line)
                    break
                except Exception:
                    continue
        except Exception:
            pass
        # #endregion
        out.append(item)
    out.sort(key=lambda x: float(x.get("last_seen_at") or 0), reverse=True)
    return out[:limit]


def _watch_alert_on_cooldown(gid: str, camera_id: str) -> bool:
    key = f"{WATCH_ALERT_CD_PREFIX}{gid}:{camera_id or '_'}"
    return bool(get_redis().exists(key))


def _set_watch_alert_cooldown(gid: str, camera_id: str) -> None:
    key = f"{WATCH_ALERT_CD_PREFIX}{gid}:{camera_id or '_'}"
    get_redis().setex(key, max(1, int(WATCH_ALERT_COOLDOWN_SEC)), b"1")


def _emit_watch_alert(
    gid: str,
    event: Dict[str, Any],
    *,
    hop: bool,
    prev_camera_id: str = "",
    force: bool = False,
    message_override: str = "",
) -> None:
    """Push a Site Admin alert for a watched journey sighting."""
    camera_id = event.get("camera_id") or ""
    if not force and _watch_alert_on_cooldown(gid, camera_id):
        return
    meta = get_meta(gid) or {}
    label = event.get("label") or meta.get("label") or "unknown"
    cam_name = event.get("camera_name") or meta.get("last_camera_name") or camera_id or "Camera"
    thumb = (event.get("thumb_url") or "").strip() or _latest_thumb_url(gid, meta)
    if message_override:
        msg = message_override
    elif hop and prev_camera_id:
        prev_name = ""
        try:
            from core import site_admin_store as store

            prev_cam = store.get_camera(prev_camera_id)
            prev_name = (prev_cam or {}).get("name") or prev_camera_id
        except Exception:
            prev_name = prev_camera_id
        msg = (
            f"Watched {gid} · {label} moved {prev_name} → {cam_name}"
        )
    else:
        msg = f"Watched {gid} · {label} seen on {cam_name}"

    try:
        from core import site_admin_store as store

        # #region agent log
        try:
            import json as _json
            _line = _json.dumps({"sessionId": "46c418", "hypothesisId": "C", "location": "journey_store.py:_emit_watch_alert", "message": "emit watch alert thumb", "data": {"gid": gid, "hop": hop, "thumb_len": len(thumb or ""), "thumb_prefix": (thumb or "")[:60]}, "timestamp": int(time.time() * 1000), "runId": "post-fix"}) + "\n"
            for _p in ("/app/static/debug-46c418.log", "static/debug-46c418.log", "debug-46c418.log"):
                try:
                    with open(_p, "a", encoding="utf-8") as _f:
                        _f.write(_line)
                    break
                except Exception:
                    continue
        except Exception:
            pass
        # #endregion

        store.append_alert(
            {
                "camera_id": camera_id,
                "camera_name": cam_name,
                "rule_id": f"watch:{gid}",
                "scan_type": "journey_watch",
                "type": "journey_hop" if hop else "journey_watch",
                "severity": "medium",
                "thumb_url": thumb,
                "person_id": meta.get("person_id") or "",
                "label": f"{gid} · {label}",
                "gid": gid,
                "channels": ["web"],
                "message": msg,
            }
        )
        if not force:
            _set_watch_alert_cooldown(gid, camera_id)
    except Exception:
        pass


def resolve_gid_by_face(person_id: str) -> Optional[str]:
    if not person_id:
        return None
    raw = get_redis().get(f"{BY_FACE_PREFIX}{person_id}")
    return redis_str(raw) or None


def resolve_gid_by_plate(plate: str) -> Optional[str]:
    if not plate:
        return None
    raw = get_redis().get(f"{BY_PLATE_PREFIX}{plate}")
    return redis_str(raw) or None


def _bind_face(gid: str, person_id: str) -> None:
    if person_id:
        get_redis().set(f"{BY_FACE_PREFIX}{person_id}", gid.encode())


def _bind_plate(gid: str, plate: str) -> None:
    if plate:
        get_redis().set(f"{BY_PLATE_PREFIX}{plate}", gid.encode())


def upsert_face_sighting(
    *,
    person_id: str,
    label: str,
    camera_id: str,
    camera_name: str = "",
    rule_id: str = "",
    scan_type: str = "face_attendance",
    local_track_id: Any = None,
    bbox: Optional[List[float]] = None,
    thumb_url: str = "",
    merge_from_gid: str = "",
) -> Optional[Dict[str, Any]]:
    """Create or update a person journey from a face recognition hit."""
    if not person_id:
        return None
    with _lock:
        gid = resolve_gid_by_face(person_id)
        if merge_from_gid and merge_from_gid != gid:
            gid = merge_journeys(merge_from_gid, gid or merge_from_gid, person_id=person_id, label=label)
        if not gid:
            meta = _default_meta(
                kind="person",
                label=label or person_id,
                person_id=person_id,
                match_method="face",
            )
            gid = meta["gid"]
            _save_meta(meta)
            _bind_face(gid, person_id)
        else:
            meta = get_meta(gid) or {}
            if meta.get("merged_into"):
                gid = meta["merged_into"]
                meta = get_meta(gid) or meta
            meta["person_id"] = person_id
            if label:
                meta["label"] = label
            meta["kind"] = "person"
            _save_meta(meta)
            _bind_face(gid, person_id)

        _append_sighting(
            gid,
            {
                "ts": time.time(),
                "camera_id": camera_id,
                "camera_name": camera_name,
                "rule_id": rule_id,
                "scan_type": scan_type,
                "local_track_id": local_track_id,
                "bbox": bbox or [],
                "thumb_url": thumb_url,
                "match_method": "face",
                "person_id": person_id,
                "label": label or person_id,
            },
        )
        return get_meta(gid)


def upsert_plate_sighting(
    *,
    plate: str,
    label: str = "",
    camera_id: str,
    camera_name: str = "",
    rule_id: str = "",
    scan_type: str = "vehicle",
    local_track_id: Any = None,
    bbox: Optional[List[float]] = None,
    thumb_url: str = "",
    merge_from_gid: str = "",
) -> Optional[Dict[str, Any]]:
    """Create or update a vehicle journey from a plate hit."""
    from core import site_admin_store as store

    key = store.normalize_plate(plate)
    if not key:
        return None
    display = label or key
    with _lock:
        gid = resolve_gid_by_plate(key)
        if merge_from_gid and merge_from_gid != gid:
            gid = merge_journeys(merge_from_gid, gid or merge_from_gid, plate=key, label=display)
        if not gid:
            meta = _default_meta(
                kind="vehicle",
                label=display,
                plate=key,
                match_method="plate",
            )
            gid = meta["gid"]
            _save_meta(meta)
            _bind_plate(gid, key)
        else:
            meta = get_meta(gid) or {}
            if meta.get("merged_into"):
                gid = meta["merged_into"]
                meta = get_meta(gid) or meta
            meta["plate"] = key
            meta["label"] = display
            meta["kind"] = "vehicle"
            _save_meta(meta)
            _bind_plate(gid, key)

        _append_sighting(
            gid,
            {
                "ts": time.time(),
                "camera_id": camera_id,
                "camera_name": camera_name,
                "rule_id": rule_id,
                "scan_type": scan_type,
                "local_track_id": local_track_id,
                "bbox": bbox or [],
                "thumb_url": thumb_url,
                "match_method": "plate",
                "plate": key,
                "label": display,
            },
        )
        return get_meta(gid)


def _load_reid_embedding(gid: str) -> Optional[np.ndarray]:
    raw = get_redis().get(f"{REID_EMB_PREFIX}{gid}")
    if not raw:
        return None
    try:
        arr = np.frombuffer(raw, dtype=np.float32)
        if arr.size == 0:
            return None
        return arr
    except Exception:
        return None


def _save_reid_embedding(gid: str, emb: np.ndarray) -> None:
    vec = np.asarray(emb, dtype=np.float32).reshape(-1)
    get_redis().set(f"{REID_EMB_PREFIX}{gid}", vec.tobytes())


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or a.size == 0 or b.size == 0:
        return -1.0
    n = min(a.size, b.size)
    aa = a[:n].astype(np.float32)
    bb = b[:n].astype(np.float32)
    na = float(np.linalg.norm(aa))
    nb = float(np.linalg.norm(bb))
    if na < 1e-8 or nb < 1e-8:
        return -1.0
    return float(np.dot(aa, bb) / (na * nb))


def match_or_create_reid(
    embedding: np.ndarray,
    *,
    camera_id: str,
    camera_name: str = "",
    rule_id: str = "",
    scan_type: str = "gate_analytics",
    local_track_id: Any = None,
    bbox: Optional[List[float]] = None,
    thumb_url: str = "",
    threshold: Optional[float] = None,
    handoff_sec: Optional[float] = None,
) -> Dict[str, Any]:
    """Match anonymous person embedding to an active journey or mint a new gid."""
    thr = float(threshold if threshold is not None else os.environ.get("JOURNEY_REID_THRESHOLD", "0.62"))
    window = float(handoff_sec if handoff_sec is not None else HANDOFF_SEC)
    now = time.time()
    emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if emb.size == 0:
        meta = _default_meta(kind="person", label="unknown", match_method="new")
        _save_meta(meta)
        return meta

    with _lock:
        r = get_redis()
        candidates = r.zrevrangebyscore(ACTIVE_KEY, now, now - window)
        best_gid = None
        best_score = -1.0
        for raw_gid in candidates:
            gid = redis_str(raw_gid)
            meta = get_meta(gid)
            if not meta or meta.get("merged_into"):
                continue
            if meta.get("kind") == "vehicle":
                continue
            # Prefer handoff across cameras; allow same-camera reappear within window
            other_emb = _load_reid_embedding(gid)
            if other_emb is None:
                continue
            score = _cosine(emb, other_emb)
            if score >= thr and score > best_score:
                # Prefer different camera when scores are close
                if best_gid is None or camera_id != (meta.get("last_camera_id") or ""):
                    best_score = score
                    best_gid = gid
                elif score > best_score + 0.02:
                    best_score = score
                    best_gid = gid

        if best_gid:
            meta = get_meta(best_gid) or {}
            method = "reid"
            _save_reid_embedding(best_gid, emb)  # refresh appearance
            _append_sighting(
                best_gid,
                {
                    "ts": now,
                    "camera_id": camera_id,
                    "camera_name": camera_name,
                    "rule_id": rule_id,
                    "scan_type": scan_type,
                    "local_track_id": local_track_id,
                    "bbox": bbox or [],
                    "thumb_url": thumb_url,
                    "match_method": method,
                    "reid_score": round(best_score, 4),
                    "label": meta.get("label") or "unknown",
                },
            )
            out = get_meta(best_gid) or meta
            out = dict(out)
            out["reid_score"] = round(best_score, 4)
            out["match_method"] = method
            return out

        meta = _default_meta(kind="person", label="unknown", match_method="new")
        gid = meta["gid"]
        _save_meta(meta)
        _save_reid_embedding(gid, emb)
        _append_sighting(
            gid,
            {
                "ts": now,
                "camera_id": camera_id,
                "camera_name": camera_name,
                "rule_id": rule_id,
                "scan_type": scan_type,
                "local_track_id": local_track_id,
                "bbox": bbox or [],
                "thumb_url": thumb_url,
                "match_method": "new",
                "label": "unknown",
            },
            force=True,
        )
        return get_meta(gid) or meta


def merge_journeys(
    source_gid: str,
    target_gid: str,
    *,
    person_id: str = "",
    plate: str = "",
    label: str = "",
) -> str:
    """Merge source into target (or create target). Returns surviving gid."""
    if not source_gid:
        return target_gid
    if not target_gid or target_gid == source_gid:
        src = get_meta(source_gid)
        if src and person_id:
            src["person_id"] = person_id
            if label:
                src["label"] = label
            _save_meta(src)
            _bind_face(source_gid, person_id)
        if src and plate:
            src["plate"] = plate
            if label:
                src["label"] = label
            src["kind"] = "vehicle"
            _save_meta(src)
            _bind_plate(source_gid, plate)
        return source_gid

    src = get_meta(source_gid)
    tgt = get_meta(target_gid)
    if not tgt:
        return source_gid
    if not src:
        return target_gid

    # Move timeline entries
    r = get_redis()
    events = r.lrange(_timeline_key(source_gid), 0, -1)
    for raw in reversed(events):
        r.lpush(_timeline_key(target_gid), raw)
    r.ltrim(_timeline_key(target_gid), 0, TIMELINE_TRIM - 1)

    tgt["last_seen_at"] = max(float(src.get("last_seen_at") or 0), float(tgt.get("last_seen_at") or 0))
    if person_id or src.get("person_id"):
        tgt["person_id"] = person_id or src.get("person_id") or tgt.get("person_id") or ""
        _bind_face(target_gid, tgt["person_id"])
    if plate or src.get("plate"):
        tgt["plate"] = plate or src.get("plate") or tgt.get("plate") or ""
        _bind_plate(target_gid, tgt["plate"])
        tgt["kind"] = "vehicle"
    if label:
        tgt["label"] = label
    elif src.get("label") and (tgt.get("label") in ("", "unknown") or not tgt.get("person_id")):
        tgt["label"] = src["label"]

    # Transfer ReID embedding if target lacks one
    if _load_reid_embedding(target_gid) is None:
        emb = _load_reid_embedding(source_gid)
        if emb is not None:
            _save_reid_embedding(target_gid, emb)

    src["merged_into"] = target_gid
    _save_meta(src)
    _save_meta(tgt)
    r.zrem(ACTIVE_KEY, source_gid)
    if is_watched(source_gid):
        r.srem(WATCH_SET, source_gid)
        r.sadd(WATCH_SET, target_gid)
    return target_gid


def list_journeys(
    *,
    limit: int = 50,
    active_minutes: float = 60.0,
    kind: str = "",
    known_only: bool = False,
    anonymous_only: bool = False,
) -> List[Dict[str, Any]]:
    now = time.time()
    since = now - max(1.0, active_minutes) * 60.0
    r = get_redis()
    gids = r.zrevrangebyscore(ACTIVE_KEY, now + 1, since)
    out: List[Dict[str, Any]] = []
    for raw in gids:
        gid = redis_str(raw)
        meta = get_meta(gid)
        if not meta or meta.get("merged_into"):
            continue
        if kind and meta.get("kind") != kind:
            continue
        has_known = bool(meta.get("person_id") or meta.get("plate"))
        if known_only and not has_known:
            continue
        if anonymous_only and has_known:
            continue
        hops = _recent_hops(gid, limit=4)
        item = dict(meta)
        item["hops"] = hops
        item["hop_summary"] = _format_hops(hops)
        item["watched"] = is_watched(gid)
        out.append(item)
        if len(out) >= limit:
            break
    return out


def get_journey(gid: str, *, timeline_limit: int = 100) -> Optional[Dict[str, Any]]:
    meta = get_meta(gid)
    if not meta:
        return None
    if meta.get("merged_into"):
        survivor = get_meta(meta["merged_into"])
        if survivor:
            meta = dict(survivor)
            gid = meta["gid"]
    events = []
    for raw in get_redis().lrange(_timeline_key(gid), 0, timeline_limit - 1):
        ev = _loads(raw)
        if ev:
            events.append(ev)
    hops = _recent_hops(gid, limit=20)
    return {
        **meta,
        "timeline": events,
        "hops": hops,
        "hop_summary": _format_hops(hops),
        "watched": is_watched(gid),
    }


def _recent_hops(gid: str, limit: int = 8) -> List[Dict[str, Any]]:
    """Collapse consecutive same-camera sightings into hop list (oldest→newest)."""
    raws = get_redis().lrange(_timeline_key(gid), 0, 80)
    events = []
    for raw in raws:
        ev = _loads(raw)
        if ev:
            events.append(ev)
    events.reverse()  # chronological
    hops: List[Dict[str, Any]] = []
    for ev in events:
        cam = ev.get("camera_id") or ""
        if hops and hops[-1].get("camera_id") == cam:
            hops[-1]["last_ts"] = ev.get("ts")
            hops[-1]["count"] = int(hops[-1].get("count") or 1) + 1
            continue
        hops.append(
            {
                "camera_id": cam,
                "camera_name": ev.get("camera_name") or cam,
                "first_ts": ev.get("ts"),
                "last_ts": ev.get("ts"),
                "match_method": ev.get("match_method"),
                "count": 1,
            }
        )
    return hops[-limit:]


def _format_hops(hops: List[Dict[str, Any]]) -> str:
    if not hops:
        return ""
    names = [h.get("camera_name") or h.get("camera_id") or "?" for h in hops]
    return " → ".join(names)


def journeys_for_camera(camera_id: str, *, within_sec: float = 90.0, limit: int = 20) -> List[Dict[str, Any]]:
    """Active journeys last seen on this camera (for Monitor badges)."""
    if not camera_id:
        return []
    now = time.time()
    items = list_journeys(limit=80, active_minutes=max(1.0, within_sec / 60.0))
    out = []
    for item in items:
        if item.get("last_camera_id") != camera_id:
            continue
        if now - float(item.get("last_seen_at") or 0) > within_sec:
            continue
        if not item.get("hop_summary"):
            hops = _recent_hops(item.get("gid") or "", limit=6)
            item = dict(item)
            item["hops"] = hops
            item["hop_summary"] = _format_hops(hops)
        item = dict(item)
        item["watched"] = is_watched(item.get("gid") or "")
        bbox = item.get("last_bbox") or []
        if isinstance(bbox, list) and len(bbox) == 4:
            # Store as pixel bbox on meta; UI also gets bbox_norm if frame size unknown — leave as last_bbox
            item["bbox"] = bbox
        out.append(item)
        if len(out) >= limit:
            break
    return out


def attach_local_track(camera_id: str, local_track_id: Any, gid: str) -> None:
    """Map local ByteTrack id → gid for merge when face/plate arrives."""
    if not camera_id or local_track_id is None or not gid:
        return
    key = f"journey:track:{camera_id}:{local_track_id}"
    get_redis().setex(key, max(30, int(HANDOFF_SEC)), str(gid).encode())


def lookup_local_track(camera_id: str, local_track_id: Any) -> Optional[str]:
    if not camera_id or local_track_id is None:
        return None
    key = f"journey:track:{camera_id}:{local_track_id}"
    raw = get_redis().get(key)
    return redis_str(raw) or None


def upsert_sam_selection(
    *,
    embedding: Optional[np.ndarray],
    camera_id: str,
    camera_name: str = "",
    bbox: Optional[List[float]] = None,
    thumb_url: str = "",
    label: str = "person",
    local_track_id: Any = None,
) -> Dict[str, Any]:
    """Create or match a journey from a Monitor person / SAM selection."""
    if embedding is not None and getattr(embedding, "size", 0) > 0:
        meta = match_or_create_reid(
            embedding,
            camera_id=camera_id,
            camera_name=camera_name,
            rule_id="person_track",
            scan_type="person_track",
            local_track_id=local_track_id,
            bbox=bbox or [],
            thumb_url=thumb_url,
        )
        gid = (meta or {}).get("gid") or ""
        if gid and local_track_id is not None:
            attach_local_track(camera_id, local_track_id, gid)
        return meta
    with _lock:
        meta = _default_meta(kind="person", label=label or "person", match_method="person_track")
        gid = meta["gid"]
        _save_meta(meta)
        _append_sighting(
            gid,
            {
                "ts": time.time(),
                "camera_id": camera_id,
                "camera_name": camera_name,
                "rule_id": "person_track",
                "scan_type": "person_track",
                "local_track_id": local_track_id,
                "bbox": bbox or [],
                "thumb_url": thumb_url,
                "match_method": "person_track",
                "label": label or "person",
            },
            force=True,
        )
        if local_track_id is not None:
            attach_local_track(camera_id, local_track_id, gid)
        return get_meta(gid) or meta
