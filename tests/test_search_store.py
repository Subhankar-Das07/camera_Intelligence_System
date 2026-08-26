"""Tests for search replay store validation."""

from __future__ import annotations

import pytest

from core import site_admin_search_store as sstore


def test_rule_ids_max_3():
    with pytest.raises(ValueError, match="3"):
        sstore.validate_rule_ids("cam-1", ["a", "b", "c", "d"])


def test_rule_ids_empty():
    with pytest.raises(ValueError, match="at least one"):
        sstore.validate_rule_ids("cam-1", [])


def test_search_worker_skips_alert_inbox():
    import inspect

    from core import site_admin_search

    src = inspect.getsource(site_admin_search.run_search_job)
    assert "emit_site_alert" not in src
    assert "append_alert" not in src
    assert "skip_gate_store" in src or "search_replay" in src


def test_validate_clip_max_180():
    with pytest.raises(ValueError, match="3 minutes"):
        sstore.validate_clip(0, 200, 300)


def test_validate_clip_negative_start():
    with pytest.raises(ValueError, match="start"):
        sstore.validate_clip(-1, 60, 120)


def test_validate_clip_beyond_duration():
    with pytest.raises(ValueError, match="beyond"):
        sstore.validate_clip(100, 150, 120)


def test_validate_clip_short_video_full_length():
    start, end = sstore.validate_clip(0, None, 90)
    assert start == 0
    assert end == 90


def test_validate_clip_three_minute_window():
    start, end = sstore.validate_clip(60, None, 600)
    assert start == 60
    assert end == 240
