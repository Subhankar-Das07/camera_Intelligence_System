"""Parallel go-live scanner: one process per camera (max 8), shared inference capacity."""

from __future__ import annotations

import logging
import multiprocessing
import os
import threading
import time
from typing import Any, Dict, List, Optional

from core import site_admin_common as common
from core import site_admin_scan as scan
from core import site_admin_store as store

log = logging.getLogger("site_admin.runtime")

_lock = threading.Lock()
_supervisor_thread: Optional[threading.Thread] = None
_stop_supervisor = threading.Event()
_ctx: Optional[multiprocessing.context.BaseContext] = None
_worker_stop: Optional[Any] = None
_inference_sem: Optional[Any] = None
_workers: Dict[str, multiprocessing.Process] = {}
_runtime_config: Dict[str, Any] = {}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def runtime_config() -> Dict[str, Any]:
    return {
        "max_cameras": _env_int("SITE_ADMIN_MAX_CAMERAS", 8),
        "inference_slots": max(1, _env_int("SITE_ADMIN_INFERENCE_SLOTS", 2)),
        "stagger_ms": max(0, _env_int("SITE_ADMIN_WORKER_STAGGER_MS", 250)),
        "tick_min": max(0.2, _env_float("SITE_ADMIN_TICK_MIN_SEC", 1.0)),
        "tick_max": max(0.5, _env_float("SITE_ADMIN_TICK_MAX_SEC", 3.0)),
    }


def _bootstrap_worker_registry() -> None:
    """Spawned workers need full pipeline registry (not loaded via main.py)."""
    from core.registry import registry

    names = set(registry.get_available_pipelines())
    if "vehicle_recognition" not in names:
        from pipelines.vehicle_recognition.pipeline import VehicleRecognitionPipeline

        registry.register("vehicle_recognition", VehicleRecognitionPipeline)
    if "gate_analytics" not in names:
        from pipelines.gate_analytics_pipeline import GateAnalyticsPipeline

        registry.register("gate_analytics", GateAnalyticsPipeline)


def _cameras_with_rules() -> List[str]:
    rules = [r for r in store.list_rules() if r.get("enabled", True)]
    cams = [c for c in store.list_cameras() if c.get("enabled", True)]
    cam_ids = []
    for cam in cams:
        cam_id = cam.get("id") or ""
        if not cam_id:
            continue
        if any(r.get("camera_id") == cam_id for r in rules):
            cam_ids.append(cam_id)
    return cam_ids


def get_runtime_status() -> Dict[str, Any]:
    workers = store.list_runtime_worker_heartbeats(max_age_sec=5.0)
    active_ids = [w.get("camera_id") for w in workers if w.get("camera_id")]
    live = bool(store.get_site().get("go_live"))
    scanning = live and (bool(active_ids) or bool(_workers))
    first = active_ids[0] if active_ids else ""
    first_worker = workers[0] if workers else {}
    return {
        "scanning": scanning,
        "active_camera_ids": active_ids,
        "active_camera_id": first,
        "active_rule_id": "",
        "active_scan_type": "",
        "workers": workers,
        "worker_count": len(_workers),
        "inference_slots": _runtime_config.get("inference_slots", 2),
    }


def camera_worker_main(
    camera_id: str,
    stop_flag: Any,
    inference_sem: Any,
    config: Dict[str, Any],
) -> None:
    """Entry point for each camera scan process (spawn context)."""
    _bootstrap_worker_registry()
    os.makedirs(common.ALERTS_DIR, exist_ok=True)
    state = scan.new_worker_state()
    tick_min = float(config.get("tick_min", 1.0))
    tick_max = float(config.get("tick_max", 3.0))
    log.info("camera worker started: %s", camera_id)

    try:
        while not stop_flag.is_set():
            if not store.get_site().get("go_live"):
                break

            if store.is_camera_monitored_redis(camera_id):
                time.sleep(1.0)
                continue

            cam = store.get_camera(camera_id)
            if not cam or cam.get("enabled") is False:
                time.sleep(2.0)
                continue

            rules = [
                r
                for r in store.list_rules()
                if r.get("camera_id") == camera_id and r.get("enabled", True)
            ]
            if not rules:
                time.sleep(2.0)
                continue

            frame = scan.read_camera_frame(cam, state)
            if frame is None:
                store.save_camera(
                    {**cam, "health": "offline", "last_error": "Source unavailable"}
                )
                time.sleep(tick_max)
                continue

            store.save_camera(
                {
                    **cam,
                    "health": "online",
                    "last_error": "",
                    "last_frame_at": time.time(),
                }
            )

            last_wait_ms = 0.0
            t0 = time.time()
            inference_sem.acquire()
            last_wait_ms = (time.time() - t0) * 1000.0
            try:
                hits = scan.evaluate_camera_frame(cam, rules, frame, state)
            finally:
                inference_sem.release()

            for hit_rule, event, snap in hits:
                if stop_flag.is_set():
                    break
                rule_id = hit_rule.get("id") or ""
                scan_type = hit_rule.get("scan_type") or ""
                if scan_type == "vehicle":
                    common.handle_vehicle_sighting(cam, hit_rule, event, frame=snap)
                elif not store.on_cooldown(rule_id):
                    if common.is_event_dict(event) or (isinstance(event, dict) and event.get("type")):
                        common.emit_site_alert(cam, hit_rule, event, frame=snap)
                elif scan_type == "face_attendance" and isinstance(event, dict) and event.get("person_id"):
                    # Still update journeys when alert cooldown suppresses inbox spam
                    common.record_journey_from_hit(cam, hit_rule, event, frame=snap)

            now = time.time()
            store.set_runtime_worker_heartbeat(
                camera_id,
                {
                    "last_tick_at": now,
                    "inference_wait_ms": round(last_wait_ms, 1),
                    "rule_count": len(rules),
                },
            )

            sleep_sec = scan.compute_tick_sleep_sec(rules, last_wait_ms, tick_min, tick_max)
            deadline = now + sleep_sec
            while time.time() < deadline:
                if stop_flag.is_set() or not store.get_site().get("go_live"):
                    break
                time.sleep(min(0.25, deadline - time.time()))
    except Exception as e:
        log.warning("camera worker %s crashed: %s", camera_id, e)
    finally:
        cap = state.get("capture")
        if cap is not None and hasattr(cap, "release"):
            try:
                cap.release()
            except Exception:
                pass
        for gen in (state.get("generators") or {}).values():
            try:
                gen.close()
            except Exception:
                pass
        log.info("camera worker stopped: %s", camera_id)


def _stop_all_workers() -> None:
    global _worker_stop, _workers
    if _worker_stop is not None:
        _worker_stop.set()
    for proc in list(_workers.values()):
        if proc.is_alive():
            proc.join(timeout=8.0)
            if proc.is_alive():
                log.warning("terminating worker %s", proc.name)
                proc.terminate()
                proc.join(timeout=2.0)
    _workers.clear()
    store.clear_runtime_worker_heartbeats()


def _spawn_workers(camera_ids: List[str]) -> None:
    global _ctx, _worker_stop, _inference_sem, _workers, _runtime_config

    if _ctx is None:
        _ctx = multiprocessing.get_context("spawn")
    _runtime_config = runtime_config()
    max_cameras = int(_runtime_config["max_cameras"])
    if len(camera_ids) > max_cameras:
        log.warning(
            "runtime: %s cameras with rules; max %s — extra cameras skipped",
            len(camera_ids),
            max_cameras,
        )
        camera_ids = camera_ids[:max_cameras]

    _stop_all_workers()
    _worker_stop = _ctx.Event()
    _worker_stop.clear()
    slots = int(_runtime_config["inference_slots"])
    _inference_sem = _ctx.Semaphore(slots)
    stagger_ms = int(_runtime_config["stagger_ms"])
    worker_cfg = {
        "tick_min": _runtime_config["tick_min"],
        "tick_max": _runtime_config["tick_max"],
    }

    for idx, cam_id in enumerate(camera_ids):
        if idx > 0 and stagger_ms > 0:
            time.sleep(stagger_ms / 1000.0)
        proc = _ctx.Process(
            target=camera_worker_main,
            args=(cam_id, _worker_stop, _inference_sem, worker_cfg),
            name=f"site-admin-scan-{cam_id[:8]}",
            daemon=True,
        )
        proc.start()
        _workers[cam_id] = proc
        log.info("spawned scan worker for camera %s (pid=%s)", cam_id, proc.pid)


def _reconcile_workers() -> None:
    if not store.get_site().get("go_live"):
        if _workers:
            _stop_all_workers()
        return

    target = _cameras_with_rules()
    current = set(_workers.keys())
    desired = set(target[: runtime_config()["max_cameras"]])

    dead = [cid for cid, proc in _workers.items() if not proc.is_alive()]
    if dead:
        log.warning("workers dead: %s — restarting pool", dead)

    if current != desired or dead:
        _spawn_workers(target)


def _supervisor_loop() -> None:
    log.info("runtime supervisor started")
    while not _stop_supervisor.is_set():
        try:
            if store.get_site().get("go_live"):
                _reconcile_workers()
            else:
                _stop_all_workers()
        except Exception as e:
            log.warning("runtime supervisor error: %s", e)
        time.sleep(2.0)
    _stop_all_workers()
    log.info("runtime supervisor stopped")


def start_runtime() -> None:
    global _supervisor_thread
    with _lock:
        if _supervisor_thread and _supervisor_thread.is_alive():
            return
        _stop_supervisor.clear()
        _supervisor_thread = threading.Thread(
            target=_supervisor_loop,
            name="site-admin-runtime-supervisor",
            daemon=True,
        )
        _supervisor_thread.start()
        log.info("runtime start requested")


def stop_runtime() -> None:
    global _supervisor_thread
    _stop_supervisor.set()
    _stop_all_workers()
    with _lock:
        if _supervisor_thread and _supervisor_thread.is_alive():
            _supervisor_thread.join(timeout=3.0)
        _supervisor_thread = None
    log.info("runtime stop requested")
