"""
identity_manager.py — Face identities persisted entirely in Redis.

Redis keys:
  fr:identities              → JSON object {person_id: metadata}
  fr:emb:{person_id}         → numpy float32 matrix bytes + shape in fr:embmeta:{pid}
  fr:face:{person_id}:{n}    → JPEG face crop bytes
  fr:face_count:{person_id}  → integer count of crops

FAISS IndexFlatIP is rebuilt in-memory from Redis embeddings on load/mutation.
No identities.json / embeddings.npz / faces.faiss disk persistence.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Dict, List, Optional, Tuple

import cv2
import faiss
import numpy as np

from core.redis_client import get_redis, redis_str
from face_recognition.config import (
    PERSON_ID_PREFIX,
    AUTO_LABEL_PREFIX,
)

log = logging.getLogger(__name__)

EMBEDDING_DIM = 512
FR_IDENTITIES_KEY = b"fr:identities"


class IdentityManager:
    """Thread-safe face identity manager backed only by Redis + in-memory FAISS."""

    def __init__(self, base_dir: str = BASE_DIR):
        self._lock = threading.RLock()
        self._redis_prefix = base_dir  # We repurpose base_dir parameter as redis_prefix
        if self._redis_prefix.endswith('/'):
            self._redis_prefix = self._redis_prefix[:-1]

        self._identities: Dict[str, dict] = {}
        self._embeddings: Dict[str, np.ndarray] = {}
        self._faiss_index: Optional[faiss.IndexFlatIP] = None
        self._row_to_person: List[str] = []
        self.r = get_redis()
        self._load_from_redis()

    # ── Redis I/O ───────────────────────────────────────────────────────────────

    def _load_from_redis(self):
        with self._lock:
            raw = self.r.get(f"{self._redis_prefix}:identities")
            if raw:
                try:
                    self._identities = json.loads(redis_str(raw))
                except json.JSONDecodeError as e:
                    log.error("[IdentityManager] Bad %s:identities JSON: %s", self._redis_prefix, e)
                    self._identities = {}
            else:
                self._identities = {}

            self._embeddings = {}
            for pid in list(self._identities.keys()):
                emb = self._load_embedding(pid)
                if emb is not None:
                    self._embeddings[pid] = emb

            self._rebuild_faiss_index()
            log.info(
                "[IdentityManager] Loaded %d identities from Redis.",
                len(self._identities),
            )

    def _load_embedding(self, person_id: str) -> Optional[np.ndarray]:
        meta_raw = self.r.get(f"{self._redis_prefix}:embmeta:{person_id}")
        data = self.r.get(f"{self._redis_prefix}:emb:{person_id}")
        if not meta_raw or not data:
            return None
        try:
            meta = json.loads(redis_str(meta_raw))
            shape = tuple(meta["shape"])
            dtype = np.dtype(meta.get("dtype", "float32"))
            arr = np.frombuffer(data, dtype=dtype).reshape(shape).copy()
            return arr.astype(np.float32)
        except Exception as e:
            log.error("[IdentityManager] Failed to load embedding %s: %s", person_id, e)
            return None

    def _save_identities(self):
        try:
            self.r.set(f"{self._redis_prefix}:identities", json.dumps(self._identities).encode())
        except Exception as e:
            log.error("[IdentityManager] Failed to save identities to Redis: %s", e)
    def _save_embedding(self, person_id: str):
        arr = self._embeddings.get(person_id)
        if arr is None:
            self.r.delete(f"{self._redis_prefix}:emb:{person_id}", f"{self._redis_prefix}:embmeta:{person_id}")
            return
        meta = {"shape": list(arr.shape), "dtype": "float32"}
        pipe = self.r.pipeline()
        pipe.set(f"{self._redis_prefix}:emb:{person_id}", np.ascontiguousarray(arr, dtype=np.float32).tobytes())
        pipe.set(f"{self._redis_prefix}:embmeta:{person_id}", json.dumps(meta).encode())
        pipe.execute()

    def _save_face_crops(self, person_id: str, face_crops: List[np.ndarray]):
        if not face_crops:
            return
        existing = int(redis_str(self.r.get(f"{self._redis_prefix}:face_count:{person_id}"), "0") or "0")
        pipe = self.r.pipeline()
        for i, crop in enumerate(face_crops):
            idx = existing + i + 1
            ok, buf = cv2.imencode(".jpg", crop)
            if ok:
                pipe.set(f"{self._redis_prefix}:face:{person_id}:{idx}", buf.tobytes())
        pipe.set(f"{self._redis_prefix}:face_count:{person_id}", str(existing + len(face_crops)).encode())
        pipe.execute()

    def _delete_face_crops(self, person_id: str):
        count = int(redis_str(self.r.get(f"{self._redis_prefix}:face_count:{person_id}"), "0") or "0")
        pipe = self.r.pipeline()
        for i in range(1, count + 1):
            pipe.delete(f"{self._redis_prefix}:face:{person_id}:{i}")
        pipe.delete(f"{self._redis_prefix}:face_count:{person_id}")
        pipe.execute()

    def get_face_bytes(self, person_id: str, index: int = 1) -> Optional[bytes]:
        return self.r.get(f"{self._redis_prefix}:face:{person_id}:{index}")

    def _rebuild_faiss_index(self):
        self._faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self._row_to_person = []
        all_vecs = []
        for pid, emb_array in self._embeddings.items():
            for _ in range(emb_array.shape[0]):
                self._row_to_person.append(pid)
            all_vecs.append(emb_array)
        if all_vecs:
            matrix = np.vstack(all_vecs).astype(np.float32)
            self._faiss_index.add(matrix)
        log.info(
            "[IdentityManager] FAISS (memory): %d vectors across %d persons.",
            self._faiss_index.ntotal,
            len(self._embeddings),
        )

    def _next_person_id(self) -> str:
        existing = [k for k in self._identities if k.startswith(PERSON_ID_PREFIX)]
        if not existing:
            return f"{PERSON_ID_PREFIX}000001"
        nums = [
            int(k[len(PERSON_ID_PREFIX):])
            for k in existing
            if k[len(PERSON_ID_PREFIX):].isdigit()
        ]
        return f"{PERSON_ID_PREFIX}{max(nums) + 1:06d}"

    def _next_auto_label(self) -> str:
        prefix = AUTO_LABEL_PREFIX + "_"
        existing = [
            v["label"] for v in self._identities.values() if v["label"].startswith(prefix)
        ]
        nums = []
        for lbl in existing:
            suffix = lbl.split("_")[-1]
            if suffix.isdigit():
                nums.append(int(suffix))
        next_num = max(nums) + 1 if nums else 1
        return f"{AUTO_LABEL_PREFIX}_{next_num:03d}"

    def add_identity(
        self,
        embeddings: np.ndarray,
        label: Optional[str] = None,
        face_crops: Optional[List[np.ndarray]] = None,
        person_id: Optional[str] = None,
        identity_type: str = "known",
    ) -> str:
        with self._lock:
            pid = person_id if person_id else self._next_person_id()
            auto_label = label if label else self._next_auto_label()
            named = bool(label)

            self._identities[pid] = {
                "label": auto_label,
                "named": named,
                "created_at": time.time(),
                "face_count": int(embeddings.shape[0]),
            }
            normed = self._l2_normalize(embeddings)
            self._embeddings[pid] = normed

            self._save_identities()
            self._save_embedding(pid)
            self._rebuild_faiss_index()
            if face_crops:
                self._save_face_crops(pid, face_crops)

            log.info(
                "[IdentityManager] Saved person %s ('%s') with %d embeddings (Redis).",
                pid,
                auto_label,
                embeddings.shape[0],
            )
            return pid

    def append_embeddings(self, person_id: str, new_embeddings: np.ndarray) -> None:
        with self._lock:
            if person_id not in self._embeddings:
                log.warning("[IdentityManager] append_embeddings: %s not found.", person_id)
                return
            normed = self._l2_normalize(new_embeddings)
            self._embeddings[person_id] = np.vstack([self._embeddings[person_id], normed])
            self._identities[person_id]["face_count"] = self._embeddings[person_id].shape[0]
            self._save_identities()
            self._save_embedding(person_id)
            self._rebuild_faiss_index()

    def rename_identity(self, person_id: str, new_label: str) -> bool:
        with self._lock:
            if person_id not in self._identities:
                return False
            self._identities[person_id]["label"] = new_label
            self._identities[person_id]["named"] = True
            self._save_identities()
            log.info("[IdentityManager] Renamed %s to '%s'.", person_id, new_label)
            return True

    def delete_identity(self, person_id: str) -> bool:
        with self._lock:
            if person_id not in self._identities:
                return False
            self._identities.pop(person_id, None)
            self._embeddings.pop(person_id, None)
            self._save_identities()
            self._save_embedding(person_id)
            self._delete_face_crops(person_id)
            self._rebuild_faiss_index()
            log.info("[IdentityManager] Deleted person %s.", person_id)
            return True

    def get_identity(self, person_id: str) -> Optional[dict]:
        with self._lock:
            return self._identities.get(person_id)

    def get_all_identities(self) -> dict:
        with self._lock:
            return dict(self._identities)

    def search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> List[Tuple[str, str, float]]:
        with self._lock:
            if self._faiss_index is None or self._faiss_index.ntotal == 0:
                return []
            q = self._l2_normalize(query_embedding.reshape(1, -1)).astype(np.float32)
            k = min(top_k, self._faiss_index.ntotal)
            scores, indices = self._faiss_index.search(q, k)
            person_best: Dict[str, float] = {}
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                pid = self._row_to_person[idx]
                if pid not in person_best or score > person_best[pid]:
                    person_best[pid] = float(score)
            results = []
            for pid, score in sorted(person_best.items(), key=lambda x: x[1], reverse=True):
                label = self._identities.get(pid, {}).get("label", pid)
                results.append((pid, label, score))
            return results

    def get_face_thumbnail_url(self, person_id: str, mode: str = "visitor") -> Optional[str]:
        count = int(redis_str(self.r.get(f"{self._redis_prefix}:face_count:{person_id}"), "0") or "0")
        if count < 1:
            return None
        return f"/api/faces/image/{person_id}/1?mode={mode}"

    def get_stats(self) -> dict:
        with self._lock:
            named = sum(1 for v in self._identities.values() if v.get("named"))
            return {
                "total_identities": len(self._identities),
                "named_count": named,
                "unnamed_count": len(self._identities) - named,
                "faiss_vectors": self._faiss_index.ntotal if self._faiss_index else 0,
            }

    @staticmethod
    def _l2_normalize(x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (x / norms).astype(np.float32)
