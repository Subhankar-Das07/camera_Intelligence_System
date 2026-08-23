"""HTTP ISAPI snapshot source for Hikvision DVR cameras (digest auth)."""

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import requests
from requests.auth import HTTPDigestAuth

log = logging.getLogger(__name__)


def fetch_snapshot_jpeg(
    url: str,
    user: str = "",
    password: str = "",
    timeout: float = 10.0,
) -> Optional[np.ndarray]:
    """Fetch one JPEG frame from an ISAPI /picture URL."""
    url = (url or "").strip()
    if not url:
        return None
    auth = HTTPDigestAuth(user, password) if user else None
    try:
        resp = requests.get(url, auth=auth, timeout=timeout)
        if resp.status_code != 200 or len(resp.content) < 500:
            log.warning("snapshot fetch failed: status=%s bytes=%s", resp.status_code, len(resp.content))
            return None
        arr = np.frombuffer(resp.content, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        log.warning("snapshot fetch error: %s", e)
        return None


class SnapshotCamera:
    """
    VideoCapture-like source that polls HTTP snapshot URLs.
    Returns cached frame between polls to avoid hammering the DVR.
    """

    def __init__(
        self,
        url: str,
        user: str = "",
        password: str = "",
        poll_interval: float = 1.5,
        timeout: float = 10.0,
    ):
        self.url = (url or "").strip()
        self.user = user or ""
        self.password = password or ""
        self.poll_interval = max(0.5, float(poll_interval))
        self.timeout = timeout
        self._frame: Optional[np.ndarray] = None
        self._last_fetch = 0.0
        self._opened = bool(self.url)
        self._fail_count = 0
        self._w = 0
        self._h = 0
        self._fps = 1.0 / self.poll_interval

    def _refresh(self, force: bool = False) -> bool:
        now = time.time()
        if not force and self._frame is not None and (now - self._last_fetch) < self.poll_interval:
            return True
        frame = fetch_snapshot_jpeg(self.url, self.user, self.password, self.timeout)
        if frame is None:
            self._fail_count += 1
            if self._fail_count >= 5:
                self._opened = False
            return self._frame is not None
        self._frame = frame
        self._h, self._w = frame.shape[:2]
        self._last_fetch = now
        self._fail_count = 0
        self._opened = True
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._opened and self._frame is None:
            return False, None
        ok = self._refresh(force=self._frame is None)
        if not ok or self._frame is None:
            return False, None
        return True, self._frame.copy()

    def isOpened(self) -> bool:
        if not self.url:
            return False
        if self._frame is not None:
            return True
        return self._refresh(force=True)

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._w)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._h)
        if prop_id == cv2.CAP_PROP_FPS:
            return float(self._fps)
        return 0.0

    def release(self) -> None:
        self._opened = False
        self._frame = None
