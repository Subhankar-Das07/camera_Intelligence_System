"""Shared Redis connection for all application databases."""

from __future__ import annotations

import os
import threading
from typing import Optional

import redis

_lock = threading.Lock()
_client: Optional[redis.Redis] = None


def get_redis_url() -> str:
    return os.environ.get("REDIS_URL", "redis://localhost:6379/0")


def get_redis() -> redis.Redis:
    """Return a process-wide Redis client (decode_responses=False for binary blobs)."""
    global _client
    if _client is not None:
        return _client
    with _lock:
        if _client is None:
            _client = redis.Redis.from_url(
                get_redis_url(),
                decode_responses=False,
                socket_connect_timeout=5,
                socket_timeout=30,
                protocol=2,
            )
            _client.ping()
        return _client


def redis_str(value: bytes | str | None, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
