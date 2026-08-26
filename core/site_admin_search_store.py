"""Redis-backed search replay jobs (file cameras + existing rules)."""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import numpy as np

from core import site_admin_store as store
from core.redis_client import get_redis, redis_str

SEARCH_JOB_PREFIX = "search:job:"
SEARCH_INDEX = "search:job:index"
SEARCH_THUMB_DIR = os.path.join("storage", "search")
UPLOAD_DIR = os.path.join("storage", "uploads")
MAX_RULES = 3
MAX_CLIP_SEC = 180
MIN_CLIP_SEC = 5
JOB_TTL_SEC = 86400 * 7


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _dumps(obj: Any) -> bytes:
    return json.dumps(obj, separators=(",", ":"), cls=_NumpyEncoder).encode()


def _loads(raw: Any) -> Any:
    if raw is None:
        return None
    try:
        return json.loads(redis_str(raw))
    except json.JSONDecodeError:
        return None


def _job_key(job_id: str) -> str:
    return f"{SEARCH_JOB_PREFIX}{job_id}"


def list_file_cameras() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for cam in store.list_cameras():
        if cam.get("type") != "file":
            continue
        fn = cam.get("filename") or ""
        path = os.path.join(UPLOAD_DIR, fn) if fn else ""
        out.append(
            {
                "id": cam.get("id"),
                "name": cam.get("name") or cam.get("id"),
                "preview_url": cam.get("preview_url") or "",
                "filename": fn,
                "has_video": bool(fn and os.path.isfile(path)),
            }
        )
    out.sort(key=lambda c: c.get("name") or "")
    return out


def list_camera_rules(camera_id: str) -> List[Dict[str, Any]]:
    out = []
    for rule in store.list_rules():
        if rule.get("camera_id") != camera_id:
            continue
        if rule.get("enabled") is False:
            continue
        out.append(
            {
                "id": rule.get("id"),
                "name": rule.get("name") or rule.get("id"),
                "scan_type": rule.get("scan_type") or "",
                "severity": rule.get("severity") or "high",
            }
        )
    out.sort(key=lambda r: r.get("name") or "")
    return out


def resolve_camera_video(camera_id: str) -> tuple[str, str, Dict[str, Any]]:
    cam = store.get_camera(camera_id)
    if not cam or cam.get("type") != "file":
        raise ValueError("Camera must be a file-type upload from Cameras")
    fn = cam.get("filename") or ""
    if not fn:
        raise ValueError("Camera has no video file")
    path = os.path.join(UPLOAD_DIR, fn)
    if not os.path.isfile(path):
        raise ValueError("Video file not found on disk")
    return path, cam.get("name") or camera_id, cam


def probe_video_file(path: str) -> Dict[str, Any]:
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        if fps <= 0:
            fps = 25.0
        duration_sec = round(frame_count / fps if frame_count > 0 else 0.0, 2)
        if duration_sec <= 0:
            duration_sec = 0.01
        return {
            "duration_sec": duration_sec,
            "fps": round(fps, 3),
            "frame_count": max(0, frame_count),
        }
    finally:
        cap.release()


def get_video_meta(camera_id: str) -> Dict[str, Any]:
    path, _name, cam = resolve_camera_video(camera_id)
    meta = probe_video_file(path)
    fn = cam.get("filename") or ""
    return {
        **meta,
        "video_url": f"/storage/uploads/{fn}" if fn else "",
        "max_clip_sec": MAX_CLIP_SEC,
        "camera_id": camera_id,
        "camera_name": cam.get("name") or camera_id,
    }


def validate_clip(start_sec: float, end_sec: Optional[float], duration_sec: float) -> tuple[float, float]:
    if duration_sec <= 0:
        raise ValueError("Video has no readable duration")
    start = float(start_sec or 0)
    if start < 0:
        raise ValueError("Clip start must be zero or greater")
    if start >= duration_sec:
        raise ValueError("Clip start is beyond video duration")

    max_window = min(MAX_CLIP_SEC, duration_sec)
    end = float(end_sec) if end_sec is not None else min(start + max_window, duration_sec)
    if end <= start:
        raise ValueError("Clip end must be after start")
    if end > duration_sec + 0.05:
        raise ValueError("Clip end is beyond video duration")

    clip_len = end - start
    if clip_len < MIN_CLIP_SEC:
        raise ValueError(f"Clip must be at least {MIN_CLIP_SEC} seconds")
    if clip_len > MAX_CLIP_SEC + 0.05:
        raise ValueError(f"Clip cannot exceed {MAX_CLIP_SEC // 60} minutes")

    end = min(end, duration_sec)
    start = max(0.0, start)
    return round(start, 2), round(end, 2)


def validate_rule_ids(camera_id: str, rule_ids: List[str]) -> List[Dict[str, Any]]:
    if not rule_ids:
        raise ValueError("Select at least one rule")
    if len(rule_ids) > MAX_RULES:
        raise ValueError(f"Maximum {MAX_RULES} rules per search")
    rules: List[Dict[str, Any]] = []
    seen = set()
    for rid in rule_ids:
        if rid in seen:
            continue
        seen.add(rid)
        rule = store.get_rule(rid)
        if not rule:
            raise ValueError(f"Unknown rule: {rid}")
        if rule.get("camera_id") != camera_id:
            raise ValueError(f"Rule {rid} does not belong to this camera")
        if rule.get("enabled") is False:
            raise ValueError(f"Rule {rule.get('name') or rid} is disabled")
        rules.append(rule)
    if not rules:
        raise ValueError("No valid rules selected")
    return rules


def create_job(
    camera_id: str,
    rule_ids: List[str],
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
) -> Dict[str, Any]:
    video_path, source_name, cam = resolve_camera_video(camera_id)
    rules = validate_rule_ids(camera_id, rule_ids)
    meta = probe_video_file(video_path)
    clip_start, clip_end = validate_clip(start_sec, end_sec, meta["duration_sec"])
    job_id = str(uuid.uuid4())
    job = {
        "id": job_id,
        "status": "pending",
        "camera_id": camera_id,
        "rule_ids": [r["id"] for r in rules],
        "video_path": video_path,
        "source_name": source_name,
        "camera_name": cam.get("name") or camera_id,
        "start_sec": clip_start,
        "end_sec": clip_end,
        "clip_duration_sec": round(clip_end - clip_start, 2),
        "video_duration_sec": meta["duration_sec"],
        "fps": meta["fps"],
        "error": "",
        "created_at": time.time(),
    }
    r = get_redis()
    r.set(_job_key(job_id), _dumps(job), ex=JOB_TTL_SEC)
    r.sadd(SEARCH_INDEX, job_id)
    r.set(
        f"{SEARCH_JOB_PREFIX}{job_id}:progress",
        _dumps({"processed": 0, "total": 0, "percent": 0, "phase": "queued", "message": "Queued"}),
        ex=JOB_TTL_SEC,
    )
    r.set(f"{SEARCH_JOB_PREFIX}{job_id}:results", _dumps([]), ex=JOB_TTL_SEC)
    os.makedirs(os.path.join(SEARCH_THUMB_DIR, job_id), exist_ok=True)
    return job


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    return _loads(get_redis().get(_job_key(job_id)))


def save_job(job: Dict[str, Any]) -> Dict[str, Any]:
    get_redis().set(_job_key(job["id"]), _dumps(job), ex=JOB_TTL_SEC)
    return job


def get_progress(job_id: str) -> Dict[str, Any]:
    data = _loads(get_redis().get(f"{SEARCH_JOB_PREFIX}{job_id}:progress"))
    return data or {"processed": 0, "total": 0, "percent": 0, "phase": "unknown", "message": ""}


def set_progress(job_id: str, progress: Dict[str, Any]) -> None:
    get_redis().set(f"{SEARCH_JOB_PREFIX}{job_id}:progress", _dumps(progress), ex=JOB_TTL_SEC)


def get_results(job_id: str) -> List[Dict[str, Any]]:
    data = _loads(get_redis().get(f"{SEARCH_JOB_PREFIX}{job_id}:results"))
    return data if isinstance(data, list) else []


def append_result(job_id: str, event: Dict[str, Any]) -> None:
    r = get_redis()
    key = f"{SEARCH_JOB_PREFIX}{job_id}:results"
    results = get_results(job_id)
    results.append(event)
    r.set(key, _dumps(results), ex=JOB_TTL_SEC)


def thumb_path(job_id: str, name: str) -> str:
    return os.path.join(SEARCH_THUMB_DIR, job_id, name)


def thumb_url(job_id: str, name: str) -> str:
    return f"/storage/search/{job_id}/{name}"
