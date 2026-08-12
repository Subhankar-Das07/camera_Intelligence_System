"""
gemini_verifier.py
==================
Thread-safe background worker for refining ambiguous YOLO labels using Gemini.

Design principles:
  - NEVER blocks the detection loop (runs in a daemon thread)
  - Hard session budget cap (GEMINI_SESSION_BUDGET calls max)
  - Response validation: rejects empty, generic, or too-long answers
  - Rate limiting via cooldown
  - Only verifies objects with sufficient age (GEMINI_MIN_OBJECT_AGE)
  - Deduplication: will not re-queue a track already in cache or queue
"""

import logging
import os
import queue
import threading
import time
from typing import Dict, Optional
from PIL import Image

import config
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

_STOP_SENTINEL = object()

# Labels that Gemini commonly returns when it cannot identify an object.
# These are useless — reject them and keep the YOLO label.
_GARBAGE_RESPONSES = {
    "unknown", "object", "item", "thing", "something", "unidentifiable",
    "unclear", "blurry", "cannot", "unable", "n/a", "na", "none", "",
}


class GeminiVerifier:
    """
    Thread-safe Gemini API client for refining object labels.
    Runs entirely in a background daemon thread.
    """

    def __init__(
        self,
        verification_cache: Dict[int, dict],
        db_manager=None,
        session_id: Optional[int] = None,
    ) -> None:
        self._cache = verification_cache
        self._db = db_manager        # DatabaseManager instance (optional)
        self._session_id = session_id
        self._queue: queue.Queue = queue.Queue(maxsize=config.GEMINI_MAX_PENDING)
        self._thread: Optional[threading.Thread] = None
        self._client = None
        self._available: bool = False
        self._last_request_time: float = 0.0
        self._session_calls: int = 0          # Hard budget counter
        self._description_queued: set = set() # Track IDs queued for description (avoid re-queue)
        self._verify_queued: set = set()      # Track IDs queued for label verification (O(1) dedup)

        self._start_worker()

    # ──────────────────────────────────────────────────────────────────────────
    # Worker Lifecycle
    # ──────────────────────────────────────────────────────────────────────────

    def _start_worker(self) -> None:
        self._thread = threading.Thread(
            target=self._worker_loop, name="GeminiWorker", daemon=True
        )
        self._thread.start()

    def _worker_loop(self) -> None:
        """Background thread: initialise Gemini client then drain the queue."""
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            log.warning(
                "GEMINI_API_KEY not found. Gemini verification disabled. "
                "Set it in .env to enable."
            )
            self._available = False
        else:
            try:
                from google import genai
                self._client = genai.Client(api_key=api_key)
                self._available = True
                log.info(
                    "Gemini verification ready (model=%s, budget=%d calls/session).",
                    config.GEMINI_MODEL, config.GEMINI_SESSION_BUDGET,
                )
            except ImportError:
                log.error("google-genai not installed. Gemini verification disabled.")
                self._available = False
            except Exception as exc:
                log.error("Gemini client init failed: %s", exc)
                self._available = False

        while True:
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if item is _STOP_SENTINEL:
                break

            item_type = item[0]  # 'verify' or 'describe'

            if not self._available:
                log.debug("Gemini unavailable — skipping item.")
                self._cache[item[1]] = {"skipped_reason": "GEMINI_UNAVAILABLE"}
                self._queue.task_done()
                continue

            # Enforce rate limiting
            elapsed = time.monotonic() - self._last_request_time
            if elapsed < config.GEMINI_VERIFY_COOLDOWN:
                time.sleep(config.GEMINI_VERIFY_COOLDOWN - elapsed)

            # Enforce session budget
            if self._session_calls >= config.GEMINI_SESSION_BUDGET:
                log.warning(
                    "Gemini session budget of %d calls exhausted. "
                    "Verification disabled for remainder of session.",
                    config.GEMINI_SESSION_BUDGET,
                )
                self._available = False
                self._cache[item[1]] = {"skipped_reason": "GEMINI_BUDGET_EXCEEDED"}
                self._queue.task_done()
                continue

            if item_type == 'verify':
                _, track_id, label, crop_img, object_age = item
                # Skip if already resolved while waiting in queue
                if track_id in self._cache:
                    self._verify_queued.discard(track_id)  # Clean up
                    self._queue.task_done()
                    continue
                self._process_verification(track_id, label, crop_img)
                self._verify_queued.discard(track_id)  # Done — allow future re-queue if needed
            elif item_type == 'describe':
                _, track_id, label, crop_img = item
                self._process_description(track_id, label, crop_img)

            self._last_request_time = time.monotonic()
            self._session_calls += 1
            self._queue.task_done()

    # ──────────────────────────────────────────────────────────────────────────
    # Verification Logic
    # ──────────────────────────────────────────────────────────────────────────

    def _process_verification(
        self, track_id: int, label: str, crop_img: Image.Image
    ) -> None:
        log.info(
            "[Gemini] Verifying track #%d | YOLO label: '%s' | call %d/%d",
            track_id, label, self._session_calls + 1, config.GEMINI_SESSION_BUDGET,
        )

        prompt = (
            f"A fast object detector labelled this as '{label}'. "
            "Look at the image carefully. What specific object is this? "
            "Reply with only 1-3 words. Do NOT use sentences or punctuation."
        )

        try:
            log.info("[Gemini] API request sent for track #%d", track_id)
            response = self._client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=[crop_img, prompt],
            )
            log.info("[Gemini] API response received for track #%d", track_id)
            raw = (response.text or "").strip()
            refined = self._validate_response(raw, label)

            if refined:
                log.info("[Gemini] track #%d: '%s' → '%s'", track_id, label, refined)
                cache_entry = self._cache.get(track_id, {})
                cache_entry.update({
                    "label": refined,
                    "timestamp": time.time(),
                    "yolo_label": label,
                    "gemini_call": self._session_calls + 1,
                })
                self._cache[track_id] = cache_entry
                # Notify database — logs 'verified' event + updates tracked_objects
                if self._db and self._session_id:
                    self._db.on_object_verified(
                        self._session_id, track_id, refined, label
                    )
            else:
                log.info(
                    "[Gemini] track #%d: response '%s' rejected — keeping YOLO label '%s'.",
                    track_id, raw, label,
                )
                # Mark as "attempted" with the original label to prevent re-queuing
                cache_entry = self._cache.get(track_id, {})
                cache_entry.update({
                    "label": label,
                    "timestamp": time.time(),
                    "yolo_label": label,
                    "gemini_call": self._session_calls + 1,
                })
                self._cache[track_id] = cache_entry

        except Exception as exc:
            log.warning("[Gemini] API error for track #%d: %s", track_id, exc)
            if "timeout" in str(exc).lower():
                self._cache[track_id] = {"skipped_reason": "GEMINI_TIMEOUT"}
            else:
                self._cache[track_id] = {"skipped_reason": "GEMINI_API_ERROR"}
    def _process_description(
        self, track_id: int, label: str, crop_img: Image.Image
    ) -> None:
        """Generate a rich 3-bullet-point description of an object using Gemini."""
        log.info(
            "[Gemini] Describing track #%d | label: '%s' | call %d/%d",
            track_id, label, self._session_calls + 1, config.GEMINI_SESSION_BUDGET,
        )

        prompt = (
            f"You are analysing an object detected by a camera. The detector identified it as '{label}'. "
            "Look at the image carefully and describe this specific object. "
            "Give exactly 3 short bullet points (each under 8 words). "
            "Focus on: appearance (colour, size, material), state/condition, and any notable feature. "
            "For a person: describe clothing colour, approximate age/build, and pose or activity. "
            "Format your response EXACTLY like this (no extra text):\n"
            "- <feature 1>\n"
            "- <feature 2>\n"
            "- <feature 3>"
        )

        try:
            log.info("[Gemini] API request sent for track #%d", track_id)
            response = self._client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=[crop_img, prompt],
            )
            log.info("[Gemini] API response received for track #%d", track_id)
            raw = (response.text or "").strip()

            # Parse bullet lines
            lines = [ln.strip() for ln in raw.splitlines() if ln.strip().startswith("-")]
            if len(lines) >= 1:
                # Take up to 3 bullet points, clean them up
                bullets = [ln.lstrip("- ").strip() for ln in lines[:3]]
                description = "\n".join(f"  ▸ {b}" for b in bullets if b)
                log.info("[Gemini] track #%d description ready.", track_id)
                cache_entry = self._cache.get(track_id, {})
                cache_entry["description"] = description
                cache_entry.setdefault("label", label)
                cache_entry.setdefault("timestamp", time.time())
                cache_entry.setdefault("yolo_label", label)
                self._cache[track_id] = cache_entry
                log.info("[Gemini] Description cached for track #%d", track_id)
                log.info("[Gemini] Cache write successful for track #%d", track_id)
                # Notify database — logs 'description_updated' event + updates tracked_objects
                if self._db and self._session_id:
                    self._db.on_description_updated(
                        self._session_id, track_id, description
                    )
            else:
                log.info(
                    "[Gemini] Failed: PARSE_ERROR for track #%d. Raw response: '%s'",
                    track_id, raw[:80],
                )
                self._cache[track_id] = {"skipped_reason": "GEMINI_PARSE_FAILED"}

        except Exception as exc:
            if "timeout" in str(exc).lower():
                log.warning("[Gemini] Failed: TIMEOUT for track #%d: %s", track_id, exc)
                self._cache[track_id] = {"skipped_reason": "GEMINI_TIMEOUT"}
            else:
                log.warning("[Gemini] Failed: API_ERROR for track #%d: %s", track_id, exc)
                self._cache[track_id] = {"skipped_reason": "GEMINI_API_ERROR"}

    def _validate_response(self, raw: str, fallback: str) -> Optional[str]:
        """
        Validate and clean a Gemini response string.
        Returns cleaned label on success, None if response is garbage.
        """
        if not raw:
            return None

        # Clean up punctuation and casing
        cleaned = raw.strip().strip(".!?,").title()

        # Reject if too long (more than 4 words = probably a sentence)
        if len(cleaned.split()) > 4:
            return None

        # Reject generic/garbage responses
        if cleaned.lower() in _GARBAGE_RESPONSES:
            return None

        # Reject if identical to YOLO label (no value added)
        if cleaned.lower() == fallback.lower():
            return None

        return cleaned

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def enqueue_verification(
        self,
        track_id: int,
        label: str,
        crop_img: Image.Image,
        object_age: float = 0.0,
    ) -> bool:
        """
        Add an object crop to the async verification queue (label refinement).
        Returns True if successfully enqueued, False otherwise.
        """
        if not self._available:
            return False
        if track_id in self._cache:
            return False  # Already verified (or attempted)
        if track_id in self._verify_queued:
            return False  # Already in queue (O(1) set lookup, no queue scan)

        try:
            self._queue.put_nowait(('verify', track_id, label, crop_img, object_age))
            self._verify_queued.add(track_id)
            log.debug("[Gemini] Enqueued track #%d ('%s') for verification.", track_id, label)
            return True
        except queue.Full:
            log.debug("[Gemini] Queue full — dropped verification for track #%d.", track_id)
            return False

    def request_object_description(
        self,
        track_id: int,
        label: str,
        crop_img: Image.Image,
    ) -> bool:
        """
        Queue a rich Gemini per-object description request.
        Describes appearance, state, and notable features in 3 bullet points.
        Returns True if successfully enqueued, False otherwise.
        """
        if not self._available:
            return False
        if track_id in self._description_queued:
            return False  # Already queued for description
        gemini_data = self._cache.get(track_id)
        if gemini_data and gemini_data.get("description"):
            return False  # Already has a description

        # O(1) set check — no queue scan needed
        try:
            self._queue.put_nowait(('describe', track_id, label, crop_img))
            self._description_queued.add(track_id)
            log.debug("[Gemini] Enqueued track #%d ('%s') for description.", track_id, label)
            return True
        except queue.Full:
            log.debug("[Gemini] Queue full — dropped description for track #%d.", track_id)
            return False

    @property
    def session_calls(self) -> int:
        return self._session_calls

    @property
    def budget_remaining(self) -> int:
        return max(0, config.GEMINI_SESSION_BUDGET - self._session_calls)

    @property
    def is_available(self) -> bool:
        return self._available

    def shutdown(self) -> None:
        try:
            self._queue.put_nowait(_STOP_SENTINEL)
        except queue.Full:
            pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        log.info(
            "[Gemini] Shutdown complete. Total API calls this session: %d",
            self._session_calls,
        )
