import time
import logging
import threading
from collections import Counter
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class VisitManager:
    """
    Multi-frame consensus engine and session manager for vehicle visit tracking.

    Buffers per-track OCR readings across multiple frames, applies majority
    voting to determine a confident plate number, enforces a cooldown window
    to prevent duplicate database entries for the same visit, and caches
    finalized results for continuous bounding-box overlay without re-processing.
    """

    def __init__(
        self,
        db_manager,
        min_consensus_frames: int = 3,
        session_cooldown_seconds: int = 120,
    ):
        """
        Initialize the VisitManager.

        Args:
            db_manager: An instance of VehicleDatabase used to persist visit records.
            min_consensus_frames (int): Minimum number of agreeing frames required before
                                        a plate reading is considered reliable.
            session_cooldown_seconds (int): Time (in seconds) within which repeated
                                            detections of the same plate are treated as
                                            the same visit (not re-recorded in DB).
        """
        self.db = db_manager
        self.min_consensus_frames = min_consensus_frames
        self.session_cooldown_seconds = session_cooldown_seconds

        # Per-track rolling buffer of OCR readings.
        # Structure: { track_id: [{'plate': str, 'conf': float,
        #                           'car_crop': np.ndarray, 'plate_crop': np.ndarray}, ...] }
        self.track_buffers: Dict[int, List[dict]] = {}

        # Maps plate_number -> UNIX timestamp of when that plate was last committed to DB.
        self.active_sessions: Dict[str, float] = {}

        # Cache of tracks that have been fully resolved (consensus reached).
        # Structure: { track_id: {'plate': str, 'total_visits': int} }
        self.finalized_tracks: Dict[int, dict] = {}

        # Thread safety: a single lock guards all shared state.
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_frame_reading(
        self,
        track_id: int,
        plate_text: str,
        confidence: float,
        car_crop: np.ndarray,
        plate_crop: np.ndarray,
    ) -> None:
        """
        Append a single-frame OCR reading to the track's rolling buffer.

        Readings for already-finalized tracks are silently discarded to
        avoid wasting CPU on tracks whose plate has already been confirmed.

        Args:
            track_id (int): Unique tracker ID for this vehicle.
            plate_text (str): OCR-extracted plate string (may be empty / invalid).
            confidence (float): OCR confidence score [0.0 – 1.0].
            car_crop (np.ndarray): Full vehicle snapshot image.
            plate_crop (np.ndarray): Cropped license plate image.
        """
        if not plate_text or not plate_text.strip():
            return

        with self._lock:
            # If this track is already resolved, skip further processing.
            if track_id in self.finalized_tracks:
                return

            if track_id not in self.track_buffers:
                self.track_buffers[track_id] = []

            self.track_buffers[track_id].append(
                {
                    "plate": plate_text,
                    "conf": confidence,
                    "car_crop": car_crop,
                    "plate_crop": plate_crop,
                }
            )
            logger.debug(
                f"Track {track_id}: buffered reading '{plate_text}' "
                f"(conf={confidence:.2f}, total frames={len(self.track_buffers[track_id])})"
            )

    def process_track(self, track_id: int) -> Optional[dict]:
        """
        Attempt to finalize a track by running majority-vote consensus across
        all buffered readings.

        If the track is already finalized, the cached result is returned
        immediately (useful for continuous overlay rendering).

        If fewer than `min_consensus_frames` readings have been collected,
        returns None and waits for more frames.

        Args:
            track_id (int): Unique tracker ID to process.

        Returns:
            Optional[dict]: {'plate': str, 'total_visits': int} when finalized,
                            or None if not enough frames are available yet.
        """
        with self._lock:
            # ── Already finalized → return cache immediately ──────────────
            if track_id in self.finalized_tracks:
                return self.finalized_tracks[track_id]

            buffer = self.track_buffers.get(track_id, [])
            if len(buffer) < self.min_consensus_frames:
                return None

            # ── Majority Voting ───────────────────────────────────────────
            plate_counts = Counter(entry["plate"] for entry in buffer)
            winning_plate, vote_count = plate_counts.most_common(1)[0]

            logger.info(
                f"Track {track_id}: consensus reached → '{winning_plate}' "
                f"({vote_count}/{len(buffer)} votes)"
            )

            # ── Best Crop Selection (highest confidence for winning plate) ─
            candidates = [e for e in buffer if e["plate"] == winning_plate]
            best_entry = max(candidates, key=lambda e: e["conf"])
            best_car_crop: np.ndarray = best_entry["car_crop"]
            best_plate_crop: np.ndarray = best_entry["plate_crop"]
            best_confidence: float = best_entry["conf"]

            # ── Cooldown & Session Check ──────────────────────────────────
            current_time = time.time()
            total_visits = self._resolve_session(
                winning_plate, best_car_crop, best_plate_crop, best_confidence, current_time
            )

            # ── Finalize and free buffer memory ──────────────────────────
            result = {"plate": winning_plate, "total_visits": total_visits}
            self.finalized_tracks[track_id] = result
            self.track_buffers.pop(track_id, None)

            return result

    def cleanup_stale_tracks(self, active_track_ids: List[int]) -> None:
        """
        Remove internal state for tracks that are no longer being tracked.

        Should be called once per frame with the current set of live tracker IDs
        so memory doesn't accumulate for vehicles that have left the scene.

        Args:
            active_track_ids (List[int]): Track IDs that are still active in the current frame.
        """
        with self._lock:
            active_set = set(active_track_ids)

            stale_buffer_ids = [tid for tid in self.track_buffers if tid not in active_set]
            for tid in stale_buffer_ids:
                logger.debug(f"Track {tid}: removing stale buffer entry.")
                del self.track_buffers[tid]

            stale_final_ids = [tid for tid in self.finalized_tracks if tid not in active_set]
            for tid in stale_final_ids:
                logger.debug(f"Track {tid}: removing stale finalized entry.")
                del self.finalized_tracks[tid]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_session(
        self,
        plate: str,
        car_crop: np.ndarray,
        plate_crop: np.ndarray,
        confidence: float,
        current_time: float,
    ) -> int:
        """
        Determine whether this detection constitutes a new visit or a continuation
        of an ongoing session, and act accordingly.

        NOTE: Must be called while `self._lock` is already held.

        Args:
            plate (str): Winning plate number from majority vote.
            car_crop (np.ndarray): Best vehicle snapshot crop.
            plate_crop (np.ndarray): Best plate crop.
            confidence (float): OCR confidence of the best crop.
            current_time (float): Current UNIX timestamp.

        Returns:
            int: The total visit count for this plate.
        """
        last_seen = self.active_sessions.get(plate)
        within_cooldown = (
            last_seen is not None
            and (current_time - last_seen) < self.session_cooldown_seconds
        )

        if within_cooldown:
            # Same visit session → do NOT record again; just refresh the timestamp.
            self.active_sessions[plate] = current_time
            try:
                stats = self.db.get_vehicle_stats(plate)
                total_visits = stats["total_visits"] if stats else 1
            except Exception as e:
                logger.warning(f"Could not fetch stats for '{plate}' during cooldown: {e}")
                total_visits = 1
            logger.info(
                f"Plate '{plate}': within cooldown window "
                f"({current_time - last_seen:.1f}s / {self.session_cooldown_seconds}s). "
                f"Visit count = {total_visits}"
            )
        else:
            # New vehicle or returning after cooldown → record in DB.
            try:
                total_visits = self.db.record_visit(plate, car_crop, plate_crop, confidence)
                self.active_sessions[plate] = current_time
                logger.info(
                    f"Plate '{plate}': new visit recorded. Total visits = {total_visits}"
                )
            except Exception as e:
                logger.error(f"Failed to record visit for '{plate}': {e}")
                # Graceful degradation: fall back to a stats look-up.
                try:
                    stats = self.db.get_vehicle_stats(plate)
                    total_visits = stats["total_visits"] if stats else 1
                except Exception:
                    total_visits = 1

        return total_visits
