"""
identity_manager.py — Persistent local storage for face identities.

Storage model (v3 — clean):
    face_recognition/data/
    ├── identities.json       → {pid: {label, created_at, named, face_count}}
    ├── embeddings.npz        → {pid: ndarray(N, 512)}
    ├── faces.faiss           → IndexFlatIP cosine-similarity index
    └── persons/
        ├── P000001/          → face crops for person 1
        │   ├── face_001.jpg
        │   └── face_002.jpg
        ├── P000002/
        └── ...

No "known" / "unknown" split. Every unique face seen by the camera gets
its own folder. Auto-label = "Person_001", "Person_002", ...
User can rename via the UI ("Person_001" → "Ayush").

On startup: all embeddings are loaded from disk → FAISS index rebuilt.
On match:   cosine_sim >= SIMILARITY_THRESHOLD → person recognised (green).
On no match: new person created, saved to disk, shown as green immediately.
"""

import os
import json
import logging
import threading
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import faiss

from face_recognition.config import (
    BASE_DIR, PERSONS_DIR,
    IDENTITIES_FILE, EMBEDDINGS_FILE, FAISS_INDEX_FILE,
    PERSON_ID_PREFIX, AUTO_LABEL_PREFIX,
)

log = logging.getLogger(__name__)

EMBEDDING_DIM = 512   # ArcFace output dimension


class IdentityManager:
    """
    Thread-safe manager for face identities.
    Backed by JSON (metadata), NPZ (embeddings), and FAISS (vector index).
    All persons are treated the same — no known/unknown split.
    """

    def __init__(self):
        self._lock = threading.RLock()

        # {person_id: {"label": str, "created_at": float, "named": bool, "face_count": int}}
        self._identities: Dict[str, dict] = {}

        # {person_id: np.ndarray of shape (N, 512)}
        self._embeddings: Dict[str, np.ndarray] = {}

        # FAISS IndexFlatIP — cosine similarity on L2-normalised embeddings
        self._faiss_index: Optional[faiss.IndexFlatIP] = None

        # Lookup: FAISS row index → person_id
        self._row_to_person: List[str] = []

        self._ensure_dirs()
        self._load_from_disk()

    # ── Directory setup ────────────────────────────────────────────────────────

    def _ensure_dirs(self):
        """Create storage directories if they don't exist."""
        os.makedirs(BASE_DIR, exist_ok=True)
        os.makedirs(PERSONS_DIR, exist_ok=True)

    # ── Disk I/O ───────────────────────────────────────────────────────────────

    def _load_from_disk(self):
        """Load all persisted data into memory on startup."""
        with self._lock:
            # Load identities metadata
            if os.path.exists(IDENTITIES_FILE):
                try:
                    with open(IDENTITIES_FILE, "r", encoding="utf-8") as f:
                        self._identities = json.load(f)
                    log.info("[IdentityManager] Loaded %d identities from disk.", len(self._identities))
                except Exception as e:
                    log.error("[IdentityManager] Failed to load identities.json: %s", e)
                    self._identities = {}
            else:
                self._identities = {}

            # Load embeddings
            if os.path.exists(EMBEDDINGS_FILE):
                try:
                    data = np.load(EMBEDDINGS_FILE, allow_pickle=False)
                    self._embeddings = {k: data[k] for k in data.files}
                    log.info("[IdentityManager] Loaded embeddings for %d persons.", len(self._embeddings))
                except Exception as e:
                    log.error("[IdentityManager] Failed to load embeddings.npz: %s", e)
                    self._embeddings = {}
            else:
                self._embeddings = {}

            # Rebuild FAISS index from loaded embeddings
            self._rebuild_faiss_index()

    def _save_identities(self):
        """Flush identity metadata to disk."""
        try:
            with open(IDENTITIES_FILE, "w", encoding="utf-8") as f:
                json.dump(self._identities, f, indent=2)
        except Exception as e:
            log.error("[IdentityManager] Failed to save identities.json: %s", e)

    def _save_embeddings(self):
        """Flush all embeddings to disk."""
        try:
            np.savez(EMBEDDINGS_FILE, **self._embeddings)
        except Exception as e:
            log.error("[IdentityManager] Failed to save embeddings.npz: %s", e)

    def _save_faiss_index(self):
        """Write FAISS index to disk."""
        if self._faiss_index is not None:
            try:
                faiss.write_index(self._faiss_index, FAISS_INDEX_FILE)
            except Exception as e:
                log.error("[IdentityManager] Failed to save FAISS index: %s", e)

    def _rebuild_faiss_index(self):
        """
        Rebuild cosine-similarity FAISS index from all in-memory embeddings.
        IndexFlatIP on L2-normalised vectors → cosine similarity in [-1, 1].
        Higher = more similar. Identical = 1.0.
        """
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

        log.info("[IdentityManager] FAISS index: %d vectors across %d persons.",
                 self._faiss_index.ntotal, len(self._embeddings))

    # ── ID / Label Generation ──────────────────────────────────────────────────

    def _next_person_id(self) -> str:
        """Generate next sequential person ID: P000001, P000002, ..."""
        existing = [k for k in self._identities if k.startswith(PERSON_ID_PREFIX)]
        if not existing:
            return f"{PERSON_ID_PREFIX}000001"
        nums = [int(k[len(PERSON_ID_PREFIX):]) for k in existing
                if k[len(PERSON_ID_PREFIX):].isdigit()]
        return f"{PERSON_ID_PREFIX}{max(nums) + 1:06d}"

    def _next_auto_label(self) -> str:
        """Generate next auto-label: Person_001, Person_002, ..."""
        prefix = AUTO_LABEL_PREFIX + "_"
        existing = [v["label"] for v in self._identities.values()
                    if v["label"].startswith(prefix)]
        nums = []
        for lbl in existing:
            suffix = lbl.split("_")[-1]
            if suffix.isdigit():
                nums.append(int(suffix))
        next_num = max(nums) + 1 if nums else 1
        return f"{AUTO_LABEL_PREFIX}_{next_num:03d}"

    # ── Public API ─────────────────────────────────────────────────────────────

    def add_identity(
        self,
        embeddings: np.ndarray,
        label: Optional[str] = None,
        face_crops: Optional[List[np.ndarray]] = None,
        person_id: Optional[str] = None,
        # identity_type kept for backward-compat with existing call sites; ignored
        identity_type: str = "known",
    ) -> str:
        """
        Save a new person to the database.

        Args:
            embeddings:  shape (N, 512) — quality embeddings for this person
            label:       human name; auto-generated as "Person_NNN" if None
            face_crops:  BGR face-crop images saved as JPEGs under persons/PID/
            person_id:   force a specific ID (used when merging or registering via UI)

        Returns:
            The assigned person_id string.
        """
        with self._lock:
            pid        = person_id if person_id else self._next_person_id()
            auto_label = label if label else self._next_auto_label()
            named      = bool(label)   # True if caller gave a real human name

            self._identities[pid] = {
                "label":      auto_label,
                "named":      named,
                "created_at": time.time(),
                "face_count": embeddings.shape[0],
            }

            # Normalise + store embeddings
            normed = self._l2_normalize(embeddings)
            self._embeddings[pid] = normed

            # Persist everything
            self._save_identities()
            self._save_embeddings()
            self._rebuild_faiss_index()
            self._save_faiss_index()

            # Save face crops to persons/PID/
            if face_crops:
                person_dir = os.path.join(PERSONS_DIR, pid)
                os.makedirs(person_dir, exist_ok=True)
                for i, crop in enumerate(face_crops):
                    crop_path = os.path.join(person_dir, f"face_{i + 1:03d}.jpg")
                    cv2.imwrite(crop_path, crop)

            log.info("[IdentityManager] Saved person %s ('%s') with %d embeddings.",
                     pid, auto_label, embeddings.shape[0])
            return pid

    def append_embeddings(self, person_id: str, new_embeddings: np.ndarray) -> None:
        """Add more embeddings to an existing person (improves robustness)."""
        with self._lock:
            if person_id not in self._embeddings:
                log.warning("[IdentityManager] append_embeddings: %s not found.", person_id)
                return
            normed = self._l2_normalize(new_embeddings)
            self._embeddings[person_id] = np.vstack([self._embeddings[person_id], normed])
            self._identities[person_id]["face_count"] = self._embeddings[person_id].shape[0]
            self._save_identities()
            self._save_embeddings()
            self._rebuild_faiss_index()
            self._save_faiss_index()

    def rename_identity(self, person_id: str, new_label: str) -> bool:
        """Rename a person. Sets named=True so UI knows it has a real name."""
        with self._lock:
            if person_id not in self._identities:
                return False
            self._identities[person_id]["label"] = new_label
            self._identities[person_id]["named"]  = True
            self._save_identities()
            log.info("[IdentityManager] Renamed %s to '%s'.", person_id, new_label)
            return True

    def delete_identity(self, person_id: str) -> bool:
        """Remove a person completely from DB and FAISS."""
        with self._lock:
            if person_id not in self._identities:
                return False
            self._identities.pop(person_id, None)
            self._embeddings.pop(person_id, None)
            self._save_identities()
            self._save_embeddings()
            self._rebuild_faiss_index()
            self._save_faiss_index()
            log.info("[IdentityManager] Deleted person %s.", person_id)
            return True

    def get_identity(self, person_id: str) -> Optional[dict]:
        """Return identity metadata dict or None."""
        with self._lock:
            return self._identities.get(person_id)

    def get_all_identities(self) -> dict:
        """Return shallow copy of all identity metadata."""
        with self._lock:
            return dict(self._identities)

    def search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> List[Tuple[str, str, float]]:
        """
        Search FAISS for closest persons.

        Returns:
            List of (person_id, label, cosine_score) sorted best-first.
            cosine_score in [-1, 1]; >= SIMILARITY_THRESHOLD → same person.
        """
        with self._lock:
            if self._faiss_index is None or self._faiss_index.ntotal == 0:
                return []

            q = self._l2_normalize(query_embedding.reshape(1, -1)).astype(np.float32)
            k = min(top_k, self._faiss_index.ntotal)
            scores, indices = self._faiss_index.search(q, k)

            # Aggregate: best cosine score per person
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

    def get_face_thumbnail_url(self, person_id: str) -> Optional[str]:
        """Return URL to first saved face crop for this person."""
        person_dir = os.path.join(PERSONS_DIR, person_id)
        if not os.path.isdir(person_dir):
            return None
        crops = sorted(os.listdir(person_dir))
        if not crops:
            return None
        rel = os.path.join("face_data", "persons", person_id, crops[0])
        return "/" + rel.replace("\\", "/")

    def get_stats(self) -> dict:
        """Summary dict for the dashboard stats bar."""
        with self._lock:
            named = sum(1 for v in self._identities.values() if v.get("named"))
            return {
                "total_identities": len(self._identities),
                "named_count":   named,
                "unnamed_count": len(self._identities) - named,
                "faiss_vectors": self._faiss_index.ntotal if self._faiss_index else 0,
            }

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _l2_normalize(x: np.ndarray) -> np.ndarray:
        """L2-normalise so IndexFlatIP gives cosine similarity directly."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (x / norms).astype(np.float32)
