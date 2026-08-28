"""Shared Redis connection for all application databases."""

from __future__ import annotations

import os
import threading
from typing import Optional

import redis

_lock = threading.Lock()
_pool: Optional[redis.ConnectionPool] = None
_client: Optional[redis.Redis] = None


def get_redis_url() -> str:
    return os.environ.get("REDIS_URL", "redis://localhost:6379/0")


def get_redis() -> redis.Redis:
    """Return a process-wide Redis client backed by a ConnectionPool."""
    global _pool, _client
    if _client is not None:
        return _client
    with _lock:
        if _client is None:
            _pool = redis.ConnectionPool.from_url(
                get_redis_url(),
                decode_responses=False,
                socket_connect_timeout=5,
                socket_timeout=30,
                max_connections=50,
            )
            _client = redis.Redis(connection_pool=_pool)
            _client.ping()
        return _client

def disconnect_redis() -> None:
    """Disconnect pool to prevent zombie connections in child processes."""
    global _pool, _client
    with _lock:
        if _pool is not None:
            _pool.disconnect()
            _pool = None
        if _client is not None:
            _client.close()
            _client = None

def redis_str(value: bytes | str | None, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)

