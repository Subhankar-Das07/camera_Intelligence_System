"""
recognizer.py — Face recognition: FAISS cosine search + temporal consensus.

Flow per face track:
    1. Frames 1-N: collect quality embeddings in a candidate buffer.
    2. After CANDIDATE_MIN_EMBEDDINGS good frames:
       a. Search FAISS. If score >= SIMILARITY_THRESHOLD → RECOGNISED (green).
       b. If no match → SAVE as new Person_NNN → show as RECOGNISED (green) immediately.
    3. On subsequent frames for same track: return cached committed result (stable label).
    4. On track re-appear (new ByteTrack ID): repeat from step 1; FAISS now finds them.

Status values returned:
    "recognised"   — face is known to the system (matched or just saved). Show GREEN.
    "scanning"     — collecting frames. Show BLUE.
    "low_quality"  — never got a usable embedding. Show GRAY (handled by pipeline).
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Deque, List, Optional, Tuple

import numpy as np

from face_recognition.config import (
    SIMILARITY_THRESHOLD, MARGIN_THRESHOLD,
    TEMPORAL_WINDOW_SIZE, CONSENSUS_MIN_VOTES,
    CANDIDATE_MIN_EMBEDDINGS,
)

log = logging.getLogger(__name__)


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class RecognitionResult:
    """
    Result of classifying a single face track.

    status:
        "recognised" — face is in the persons DB (matched or just auto-saved). Show green.
        "scanning"   — still collecting quality frames. Show blue.
    label:       display name — "Person_001" (auto) or "Ayush" (renamed by user)
    person_id:   internal DB id (P000001, P000002, ...)
    confidence:  cosine similarity [0-1] — 0.0 if just saved (no prior FAISS match)
    """
    status: str
    label: str
    person_id: Optional[str] = None
    confidence: float = 0.0


@dataclass
class TrackState:
    """Per-ByteTrack-track state maintained by the Recognizer."""
    track_id: int

    # Sliding window of recent recognition votes (person_id or None per frame)
    vote_window: Deque = field(default_factory=lambda: deque(maxlen=TEMPORAL_WINDOW_SIZE))

    # Committed identity — set once and never changed for this track
    committed_status:     Optional[str] = None
    committed_person_id:  Optional[str] = None
    committed_label:      Optional[str] = None
    committed_confidence: float = 0.0
    committed_at:         Optional[float] = None

    # Last returned result (reused on non-recognition frames)
    last_result: Optional[RecognitionResult] = None


class FaceRecognizer:
    """
    Stateful per-pipeline recognizer. One instance per pipeline session.
    """

    def __init__(self, identity_manager):
        self._identity_manager = identity_manager
        self._track_states: Dict[int, TrackState] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def classify(
        self,
        track_id: int,
        embedding: Optional[np.ndarray],
        tracker,
        auto_register: bool = True,
    ) -> RecognitionResult:
        """
        Classify a face given its current embedding and track history.
        Returns a RecognitionResult with status, label and confidence.
        If auto_register is False, unknown faces will not be added to the IdentityManager.
        """
        state = self._get_or_create_state(track_id)

        # ── Already committed this track → return stable cached result ─────────
        if state.committed_person_id:
            return RecognitionResult(
                status=state.committed_status or "recognised",
                label=state.committed_label,
                person_id=state.committed_person_id,
                confidence=state.committed_confidence,
            )

        # ── No usable embedding this frame ─────────────────────────────────────
        if embedding is None:
            if state.last_result:
                return state.last_result
            buf = tracker._candidate_buffers.get(track_id, [])
            return RecognitionResult(
                status="scanning",
                label=f"Scanning... ({len(buf)}/{CANDIDATE_MIN_EMBEDDINGS})",
            )

        # ── FAISS cosine search ────────────────────────────────────────────────
        # hits = [(pid, label, cosine_score), ...] sorted best-first
        hits = self._identity_manager.search(embedding, top_k=5)
        vote = None
        best_confidence = 0.0

        if hits:
            best_pid, best_label, best_score = hits[0]
            if best_score >= SIMILARITY_THRESHOLD:
                # Margin check: must clearly beat 2nd-best to avoid ambiguous matches
                if len(hits) == 1 or (best_score - hits[1][2]) >= MARGIN_THRESHOLD:
                    vote = best_pid
                    best_confidence = round(float(best_score), 3)

        # ── Temporal voting ────────────────────────────────────────────────────
        state.vote_window.append(vote)
        consensus_pid = self._evaluate_consensus(state)

        if consensus_pid is not None:
            id_meta = self._identity_manager.get_identity(consensus_pid)
            label   = id_meta["label"] if id_meta else consensus_pid
            self._commit(state, "recognised", consensus_pid, label, best_confidence, tracker, track_id)
            log.info("[Recognizer] Track %d → RECOGNISED '%s' (%s) conf=%.2f",
                     track_id, label, consensus_pid, best_confidence)
            result = RecognitionResult("recognised", label, consensus_pid, best_confidence)
            state.last_result = result
            return result

        # ── Candidate buffer check ─────────────────────────────────────────────
        candidate_data = tracker.get_candidate_embeddings(track_id)
        if candidate_data is not None and not tracker.is_identified(track_id):
            emb_matrix, crops = candidate_data
            mean_emb = emb_matrix.mean(axis=0)
            hits2 = self._identity_manager.search(mean_emb, top_k=3)

            if hits2 and hits2[0][2] >= SIMILARITY_THRESHOLD:
                # Matches existing person via averaged embedding
                best_pid2, best_label2, best_score2 = hits2[0]
                id_meta2 = self._identity_manager.get_identity(best_pid2)
                label2   = id_meta2["label"] if id_meta2 else best_pid2
                conf2    = round(float(best_score2), 3)
                self._commit(state, "recognised", best_pid2, label2, conf2, tracker, track_id)
                log.info("[Recognizer] Track %d → RECOGNISED '%s' via mean embedding. conf=%.2f",
                         track_id, label2, conf2)
                result = RecognitionResult("recognised", label2, best_pid2, conf2)
                state.last_result = result
                return result

            else:
                if auto_register:
                    # New person — never seen before. Save immediately to persons DB.
                    new_pid = self._identity_manager.add_identity(
                        embeddings=emb_matrix,
                        face_crops=crops[:3],
                    )
                    id_meta_new = self._identity_manager.get_identity(new_pid)
                    new_label   = id_meta_new["label"] if id_meta_new else new_pid
                    # confidence=0.0 because this is the first save (no prior FAISS match)
                    self._commit(state, "recognised", new_pid, new_label, 0.0, tracker, track_id)
                    log.info("[Recognizer] Track %d → NEW person saved: '%s' (%s).",
                             track_id, new_label, new_pid)
                    result = RecognitionResult("recognised", new_label, new_pid, 0.0)
                    state.last_result = result
                    return result
                else:
                    # Auto register is false (e.g. Attendance Mode). Return unknown.
                    self._commit(state, "unknown", "unknown", "Unregistered", 0.0, tracker, track_id)
                    result = RecognitionResult("unknown", "Unregistered", None, 0.0)
                    state.last_result = result
                    return result

        # ── Still collecting frames ────────────────────────────────────────────
        buf = tracker._candidate_buffers.get(track_id, [])
        result = RecognitionResult(
            status="scanning",
            label=f"Scanning... ({len(buf)}/{CANDIDATE_MIN_EMBEDDINGS})",
        )
        state.last_result = result
        return result

    def forget_track(self, track_id: int):
        """Remove all state for a lost/ended ByteTrack track."""
        self._track_states.pop(track_id, None)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _get_or_create_state(self, track_id: int) -> TrackState:
        if track_id not in self._track_states:
            self._track_states[track_id] = TrackState(track_id=track_id)
        return self._track_states[track_id]

    def _commit(self, state: TrackState, status: str, pid: str, label: str,
                confidence: float, tracker, track_id: int):
        """Lock in a committed identity for this track."""
        state.committed_status     = status
        state.committed_person_id  = pid
        state.committed_label      = label
        state.committed_confidence = confidence
        state.committed_at         = time.time()
        tracker.clear_candidate_buffer(track_id)
        tracker.mark_identified(track_id)

    def _evaluate_consensus(self, state: TrackState) -> Optional[str]:
        """
        Return the winning person_id if it has >= CONSENSUS_MIN_VOTES
        in the current window, else None.
        """
        window = list(state.vote_window)
        if len(window) < CONSENSUS_MIN_VOTES:
            return None
        vote_counts: Dict[str, int] = {}
        for v in window:
            if v is not None:
                vote_counts[v] = vote_counts.get(v, 0) + 1
        if not vote_counts:
            return None
        best_pid = max(vote_counts, key=vote_counts.__getitem__)
        return best_pid if vote_counts[best_pid] >= CONSENSUS_MIN_VOTES else None
