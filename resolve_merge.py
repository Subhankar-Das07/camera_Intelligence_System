import os
import re

# 1. database.py
with open("pipelines/vehicle_recognition/database.py", "r", encoding="utf-8") as f:
    db_content = f.read()

db_clean = """\"\"\"Vehicle recognition database backed entirely by Redis.\"\"\"

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from core.redis_client import get_redis, redis_str

logger = logging.getLogger(__name__)


class VehicleDatabase:
    \"\"\"
    Redis-backed vehicle / visit store.
    Snapshot and plate-crop images are stored as Redis binary blobs.
    \"\"\"

    def __init__(self, db_path: str = "storage/vehicle_intelligence.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self.r = get_redis()
        logger.info("VehicleDatabase using Redis at %s", self.r.connection_pool.connection_kwargs)

    def _encode_jpeg(self, img: Optional[np.ndarray]) -> Optional[bytes]:
        if img is None or getattr(img, "size", 0) == 0:
            return None
        ok, buf = cv2.imencode(".jpg", img)
        return buf.tobytes() if ok else None

    def record_visit(
        self,
        plate_number: str,
        snapshot_img: np.ndarray,
        plate_crop_img: np.ndarray,
        confidence: float,
        vehicle_type: str = "Car",
    ) -> int:
        now = datetime.now()
        now_iso = now.isoformat(sep=" ", timespec="seconds")
        plate = plate_number.strip().upper()
        veh_key = f"vr:vehicle:{plate}"
        visits_key = f"vr:visits:{plate}"

        with self.lock:
            raw_visits = self.r.hget(veh_key, "total_visits")
            visit_num = int(redis_str(raw_visits, "0") or "0") + 1

            pipe = self.r.pipeline()
            if visit_num == 1:
                pipe.hset(
                    veh_key,
                    mapping={
                        b"plate_number": plate.encode(),
                        b"total_visits": str(visit_num).encode(),
                        b"first_seen": now_iso.encode(),
                        b"last_seen": now_iso.encode(),
                        b"vehicle_type": vehicle_type.encode(),
                        b"status": b"Unknown",
                    },
                )
            else:
                pipe.hset(
                    veh_key,
                    mapping={
                        b"total_visits": str(visit_num).encode(),
                        b"last_seen": now_iso.encode(),
                        b"vehicle_type": vehicle_type.encode(),
                    },
                )
            pipe.sadd("vr:plates", plate)

            snap_key = f"vr:snap:{plate}:{visit_num}"
            crop_key = f"vr:crop:{plate}:{visit_num}"
            snap_bytes = self._encode_jpeg(snapshot_img)
            crop_bytes = self._encode_jpeg(plate_crop_img)
            if snap_bytes:
                pipe.set(snap_key, snap_bytes)
            if crop_bytes:
                pipe.set(crop_key, crop_bytes)

            visit_rec = {
                "id": visit_num,
                "plate_number": plate,
                "visit_number": visit_num,
                "timestamp": now_iso,
                "snapshot_key": snap_key if snap_bytes else None,
                "plate_crop_key": crop_key if crop_bytes else None,
                "snapshot_path": f"/api/vr/image/snap/{plate}/{visit_num}" if snap_bytes else None,
                "plate_crop_path": f"/api/vr/image/crop/{plate}/{visit_num}" if crop_bytes else None,
                "ocr_confidence": float(confidence),
                "vehicle_type": vehicle_type,
            }
            pipe.rpush(visits_key, json.dumps(visit_rec).encode())
            pipe.execute()

            logger.info("Recorded visit %s for plate %s (Redis)", visit_num, plate)
            return visit_num

    def get_vehicle_stats(self, plate_number: str) -> Optional[Dict[str, Any]]:
        plate = plate_number.strip().upper()
        veh_key = f"vr:vehicle:{plate}"
        with self.lock:
            data = self.r.hgetall(veh_key)
            if not data:
                return None
            history_raw: List[bytes] = self.r.lrange(f"vr:visits:{plate}", 0, -1)
            history = []
            for item in reversed(history_raw):
                try:
                    history.append(json.loads(redis_str(item)))
                except json.JSONDecodeError:
                    continue
            return {
                "plate_number": redis_str(data.get(b"plate_number"), plate),
                "total_visits": int(redis_str(data.get(b"total_visits"), "0")),
                "first_seen": redis_str(data.get(b"first_seen")),
                "last_seen": redis_str(data.get(b"last_seen")),
                "status": redis_str(data.get(b"status"), "Unknown"),
                "vehicle_type": redis_str(data.get(b"vehicle_type"), "Car"),
                "history": history,
            }

    def get_all_vehicles(self) -> list:
        with self.lock:
            plates = self.r.smembers("vr:plates")
            results = []
            for p in plates:
                plate_str = redis_str(p)
                data = self.r.hgetall(f"vr:vehicle:{plate_str}")
                if data:
                    results.append({
                        "plate_number": redis_str(data.get(b"plate_number"), plate_str),
                        "total_visits": int(redis_str(data.get(b"total_visits"), "0")),
                        "first_seen": redis_str(data.get(b"first_seen")),
                        "last_seen": redis_str(data.get(b"last_seen")),
                        "status": redis_str(data.get(b"status"), "Unknown"),
                        "vehicle_type": redis_str(data.get(b"vehicle_type"), "Car"),
                        "image_path": f"/api/vr/image/snap/{plate_str}/{int(redis_str(data.get(b'total_visits'), '1'))}"
                    })
            results.sort(key=lambda x: x.get("last_seen", ""), reverse=True)
            return results

    def register_vehicle(self, plate_number: str) -> None:
        plate = plate_number.strip().upper()
        self.r.hset(f"vr:vehicle:{plate}", "status", "Known")
        logger.info(f"Vehicle %s registered as Known.", plate)

    def get_image_bytes(self, kind: str, plate_number: str, visit_number: int) -> Optional[bytes]:
        plate = plate_number.strip().upper()
        prefix = "vr:snap" if kind == "snap" else "vr:crop"
        return self.r.get(f"{prefix}:{plate}:{visit_number}")
"""
with open("pipelines/vehicle_recognition/database.py", "w", encoding="utf-8") as f:
    f.write(db_clean)

# 2. index.html
with open("static/index.html", "r", encoding="utf-8") as f:
    index_content = f.read()

# The conflict is resolved by taking the `=======` to `>>>>>>>` section.
index_clean = re.sub(r'<<<<<<< HEAD.*?=======\n(.*?)\n>>>>>>> [a-f0-9]+', r'\1', index_content, flags=re.DOTALL)
with open("static/index.html", "w", encoding="utf-8") as f:
    f.write(index_clean)

# 3. main.py
with open("main.py", "r", encoding="utf-8") as f:
    main_content = f.read()

# Fix _vr_add_detection
main_clean = main_content.replace(
'''def _vr_add_detection(session_id: str, plate: str, total_visits: int) -> bool:
    """Return True if this plate is newly recorded for the session."""
    r = get_redis()
    added = r.sadd(_vr_plates_key(session_id), plate)
    if not added:
        return False
    r.rpush(
        _vr_det_key(session_id),
        json.dumps({"plate": plate, "total_visits": total_visits}).encode(),
    )
    return True''',
'''def _vr_add_detection(session_id: str, plate: str, total_visits: int, status: str = "Unknown", vehicle_type: str = "Car", image_path: str = None) -> bool:
    """Return True if this plate is newly recorded for the session."""
    r = get_redis()
    added = r.sadd(_vr_plates_key(session_id), plate)
    if not added:
        return False
    r.rpush(
        _vr_det_key(session_id),
        json.dumps({
            "plate": plate,
            "total_visits": total_visits,
            "status": status,
            "vehicle_type": vehicle_type,
            "image_path": image_path
        }).encode(),
    )
    return True'''
)

# Resolve main.py conflict block
main_conflict = re.search(r'<<<<<<< HEAD(.*?)=======.*?_vr_add_detection.*?>>>>>>> [a-f0-9]+', main_clean, re.DOTALL)
if main_conflict:
    merged_block = """
                    # Fetch latest status and vehicle_type from DB for this plate
                    db_status       = "Unknown"
                    db_vehicle_type = det.get("vehicle_type", "Car")
                    image_path      = det.get("image_path")
                    try:
                        vehicle_row = pipeline.db.get_vehicle_stats(plate)
                        if vehicle_row:
                            db_status       = vehicle_row.get("status", "Unknown")
                            db_vehicle_type = vehicle_row.get("vehicle_type", db_vehicle_type)
                    except Exception:
                        pass
                    _vr_add_detection(session_id, plate, det.get("total_visits", 1), db_status, db_vehicle_type, image_path)
"""
    main_clean = main_clean[:main_conflict.start()] + merged_block + main_clean[main_conflict.end():]

with open("main.py", "w", encoding="utf-8") as f:
    f.write(main_clean)
