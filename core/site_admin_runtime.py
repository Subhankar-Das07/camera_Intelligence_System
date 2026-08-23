"""Background scanner: enabled cameras + rules → Site Admin alerts."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import defaultdict
from typing import Any, Dict, Optional

from core import site_admin_common as common
from core import site_admin_store as store
from core.registry import registry
from core.video_source import get_video_source

log = logging.getLogger("site_admin.runtime")

_stop = threading.Event()
_thread: Optional[threading.Thread] = None
_lock = threading.Lock()


def _run_rule_burst(cam: Dict[str, Any], rule: Dict[str, Any], budget_sec: float = 12.0) -> None:
    scan_type = rule.get("scan_type") or ""
    pipeline_name = common.SCAN_TO_PIPELINE.get(scan_type)
    if not pipeline_name:
        return
    roi = rule.get("roi_normalized") or []
    if common.needs_roi(scan_type) and len(roi) < 3:
        return
    source = common.input_for_camera(cam)
    if not source:
        store.save_camera({**cam, "health": "offline", "last_error": "Source unavailable"})
        return
    try:
        pipeline = registry.get_pipeline(pipeline_name)
    except Exception as e:
        log.warning("pipeline %s: %s", pipeline_name, e)
        return

    os.makedirs(common.ALERTS_DIR, exist_ok=True)
    started = time.time()
    try:
        gen = pipeline.run_on_video(
            source,
            common.ALERTS_DIR,
            roi,
            common.pipeline_config(scan_type),
        )
        store.save_camera({**cam, "health": "online", "last_error": "", "last_frame_at": time.time()})
        for _frame, alert_event in gen:
            if _stop.is_set() or not store.get_site().get("go_live"):
                break
            if scan_type == "vehicle":
                plate = None
                if isinstance(alert_event, dict):
                    plate = alert_event.get("plate") or alert_event.get("label")
                    if not plate:
                        for det in alert_event.get("detections") or []:
                            if det.get("plate"):
                                plate = det.get("plate")
                                break
                if not plate:
                    if time.time() - started > budget_sec:
                        break
                    continue
                common.handle_vehicle_sighting(
                    cam,
                    rule,
                    {"plate": plate, "type": "vehicle", "severity": "medium"},
                    frame=_frame,
                )
                break
            if common.is_event_dict(alert_event):
                common.emit_site_alert(cam, rule, alert_event, frame=_frame)
                break
            if time.time() - started > budget_sec:
                break
    except Exception as e:
        log.warning("scan %s/%s failed: %s", cam.get("id"), scan_type, e)
        store.save_camera({**cam, "health": "offline", "last_error": str(e)[:200]})


def _run_gate_tick(cam: Dict[str, Any], rule: Dict[str, Any], budget_sec: float = 2.5) -> None:
    gate_config = rule.get("gate_config") or {}
    if not common.gate_config_valid(gate_config):
        return
    source = common.input_for_camera(cam)
    if not source:
        store.save_camera({**cam, "health": "offline", "last_error": "Source unavailable"})
        return
    try:
        pipeline = registry.get_pipeline("gate_analytics")
    except Exception as e:
        log.warning("gate pipeline: %s", e)
        return

    rule_id = rule.get("id") or ""
    baseline = store.get_gate_baseline(rule_id)
    config: Dict[str, Any] = {
        "gate_config": gate_config,
        "rule_id": rule_id,
    }
    if baseline is not None:
        config["force_baseline"] = baseline

    cap = get_video_source(source)
    if not cap.isOpened():
        store.save_camera({**cam, "health": "offline", "last_error": "Source unavailable"})
        return

    batch: Dict[str, int] = defaultdict(int)
    started = time.time()
    frame_idx = 0
    try:
        while time.time() - started < budget_sec:
            if _stop.is_set() or not store.get_site().get("go_live"):
                break
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            _, meta = pipeline.process_frame(frame, frame_idx, None, config)
            if isinstance(meta, dict) and meta.get("type") == "gate_tick":
                for k, v in (meta.get("counters") or {}).items():
                    batch[k] += int(v or 0)
                if meta.get("band") == "near" and meta.get("alert_near"):
                    if not store.on_cooldown(rule_id):
                        common.emit_site_alert(
                            cam,
                            rule,
                            {"type": "gate_near", "severity": rule.get("severity") or "medium"},
                            frame=frame,
                        )
            frame_idx += 1
        store.save_camera({**cam, "health": "online", "last_error": "", "last_frame_at": time.time()})
    except Exception as e:
        log.warning("gate tick %s failed: %s", rule_id, e)
        store.save_camera({**cam, "health": "offline", "last_error": str(e)[:200]})
    finally:
        try:
            cap.release()
        except Exception:
            pass

    if batch:
        store.increment_gate_counters(rule_id, dict(batch))


def _loop() -> None:
    log.info("Site Admin runtime started")
    while not _stop.is_set():
        site = store.get_site()
        if not site.get("go_live"):
            time.sleep(1.5)
            continue
        cameras = [c for c in store.list_cameras() if c.get("enabled", True)]
        rules = [r for r in store.list_rules() if r.get("enabled", True)]
        if not cameras or not rules:
            time.sleep(2)
            continue
        for cam in cameras:
            if _stop.is_set():
                break
            cam_id = cam.get("id") or ""
            if common.is_camera_monitored(cam_id):
                continue
            cam_rules = [r for r in rules if r.get("camera_id") == cam_id]
            for rule in cam_rules:
                if _stop.is_set() or not store.get_site().get("go_live"):
                    break
                if common.is_camera_monitored(cam_id):
                    break
                scan_type = rule.get("scan_type") or ""
                if common.is_counter_scan(scan_type):
                    _run_gate_tick(cam, rule)
                else:
                    _run_rule_burst(cam, rule)
        time.sleep(1.0)
    log.info("Site Admin runtime stopped")


def start_runtime() -> None:
    global _thread
    with _lock:
        if _thread and _thread.is_alive():
            return
        _stop.clear()
        _thread = threading.Thread(target=_loop, name="site-admin-runtime", daemon=True)
        _thread.start()


def stop_runtime() -> None:
    _stop.set()
