"""Search replay worker — run existing rules on uploaded file camera footage."""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from typing import Any, Dict, List

import cv2

from core import site_admin_common as common
from core import site_admin_search_store as sstore
from core import site_admin_store as store
from core.site_admin_scan import _init_pipelines, evaluate_camera_frame, new_worker_state

log = logging.getLogger("site_admin.search")

FRAME_SKIP = max(1, int(os.environ.get("SEARCH_FRAME_SKIP", "3")))
MAX_PROCESSED_SAMPLES = max(100, int(os.environ.get("SEARCH_MAX_FRAMES", "5000")))


def _format_mmss(sec: float) -> str:
    s = max(0, int(sec or 0))
    return f"{s // 60:02d}:{s % 60:02d}"


def _event_message(cam: Dict[str, Any], rule: Dict[str, Any], event: Dict[str, Any]) -> str:
    scan_type = rule.get("scan_type") or "unknown"
    return common._alert_message(cam, rule, event, scan_type)


def _save_event_thumb(job_id: str, frame, rule: Dict[str, Any]) -> str:
    stamped = common.frame_with_rule_emphasis(frame, rule)
    name = f"evt_{uuid.uuid4().hex[:10]}.jpg"
    path = sstore.thumb_path(job_id, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, stamped if stamped is not None else frame)
    return sstore.thumb_url(job_id, name)


def _cooldown_active(state: Dict[str, Any], rule_id: str, frame_idx: int) -> bool:
    until = (state.get("search_cooldown_until") or {}).get(rule_id, -1)
    return frame_idx < until


def _set_cooldown(state: Dict[str, Any], rule: Dict[str, Any], frame_idx: int, fps: float) -> None:
    sec = int(rule.get("cooldown_sec") or 60)
    frames = max(1, int(sec * max(fps, 1.0)))
    cd = state.setdefault("search_cooldown_until", {})
    cd[rule.get("id") or ""] = frame_idx + frames


def run_search_job(job_id: str) -> None:
    job = sstore.get_job(job_id)
    if not job:
        return
    job["status"] = "running"
    sstore.save_job(job)

    try:
        cam = store.get_camera(job["camera_id"])
        if not cam:
            raise RuntimeError("Camera not found")
        rules = sstore.validate_rule_ids(job["camera_id"], job.get("rule_ids") or [])

        cap = cv2.VideoCapture(job.get("video_path") or "")
        if not cap.isOpened():
            raise RuntimeError("Could not open video")

        fps = float(job.get("fps") or cap.get(cv2.CAP_PROP_FPS) or 25.0)
        if fps <= 0:
            fps = 25.0

        start_sec = float(job.get("start_sec") or 0.0)
        end_sec = float(job.get("end_sec") or start_sec + sstore.MAX_CLIP_SEC)
        video_duration_sec = float(job.get("video_duration_sec") or 0.0)
        if video_duration_sec <= 0:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            video_duration_sec = frame_count / fps if frame_count > 0 else end_sec

        start_frame = max(0, int(start_sec * fps))
        end_frame = max(start_frame + 1, int(end_sec * fps))
        clip_frames = max(1, end_frame - start_frame)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        state = new_worker_state()
        state["skip_gate_store"] = True
        state["search_replay"] = True
        state["fps"] = fps
        state["frame_idx"] = start_frame
        state["search_cooldown_until"] = {}

        clip_label = f"{_format_mmss(start_sec)}–{_format_mmss(end_sec)} of {_format_mmss(video_duration_sec)}"
        sstore.set_progress(
            job_id,
            {
                "processed": 0,
                "total": clip_frames,
                "percent": 0,
                "phase": "scanning",
                "message": f"Scanning {clip_label}",
            },
        )

        frame_idx = start_frame
        processed_samples = 0
        pipelines_ready = False

        while frame_idx < end_frame and processed_samples < MAX_PROCESSED_SAMPLES:
            ret, frame = cap.read()
            if not ret:
                break

            if not pipelines_ready:
                _init_pipelines(state, cam, frame)
                pipelines_ready = True

            if (frame_idx - start_frame) % FRAME_SKIP == 0:
                state["frame_idx"] = frame_idx
                hits = evaluate_camera_frame(cam, rules, frame.copy(), state)
                ts_sec = round(frame_idx / fps if fps > 0 else frame_idx, 2)

                for hit_rule, event, snap in hits:
                    if not isinstance(event, dict):
                        continue
                    rule_id = hit_rule.get("id") or ""
                    if _cooldown_active(state, rule_id, frame_idx):
                        continue
                    thumb = _save_event_thumb(job_id, snap if snap is not None else frame, hit_rule)
                    sstore.append_result(
                        job_id,
                        {
                            "id": str(uuid.uuid4()),
                            "ts_sec": ts_sec,
                            "frame": frame_idx,
                            "rule_id": rule_id,
                            "rule_name": hit_rule.get("name") or rule_id,
                            "scan_type": hit_rule.get("scan_type") or "",
                            "message": _event_message(cam, hit_rule, event),
                            "thumb_url": thumb,
                            "event_type": event.get("type") or hit_rule.get("scan_type") or "",
                        },
                    )
                    _set_cooldown(state, hit_rule, frame_idx, fps)

                processed_samples += 1
                if processed_samples % 10 == 0:
                    done_in_clip = frame_idx - start_frame
                    pct = min(99, int(100 * done_in_clip / clip_frames))
                    sstore.set_progress(
                        job_id,
                        {
                            "processed": done_in_clip,
                            "total": clip_frames,
                            "percent": pct,
                            "phase": "scanning",
                            "message": f"Scanning {clip_label} — frame {done_in_clip} of {clip_frames}",
                        },
                    )

            frame_idx += 1

        cap.release()
        results = sstore.get_results(job_id)
        job = sstore.get_job(job_id) or job
        job["status"] = "done"
        job["error"] = ""
        sstore.save_job(job)
        sstore.set_progress(
            job_id,
            {
                "processed": clip_frames,
                "total": clip_frames,
                "percent": 100,
                "phase": "done",
                "message": f"Found {len(results)} rule break(s) in {clip_label}",
            },
        )
    except Exception as e:
        log.exception("Search job %s failed", job_id)
        job = sstore.get_job(job_id) or {"id": job_id}
        job["status"] = "error"
        job["error"] = str(e)
        sstore.save_job(job)
        sstore.set_progress(
            job_id,
            {"processed": 0, "total": 0, "percent": 0, "phase": "error", "message": str(e)},
        )


def spawn_search_job(job_id: str) -> None:
    threading.Thread(target=run_search_job, args=(job_id,), daemon=True).start()
