"""Session-level lifecycle of the compression-cache registry.

Covers the two eviction paths on ``HeadroomProxy._get_compression_cache``:

* capacity eviction must be LRU by *access* (a busy long-lived session
  survives; the idlest session goes), not FIFO by creation, and
* the lazy idle-TTL sweep must reclaim sessions whose provider prompt
  cache has lapsed, while an access refreshes the clock.

Entry-level LRU/limits inside a single ``CompressionCache`` live in
``test_compression_cache.py``.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("fastapi")


def _make_proxy():
    from headroom.proxy.server import ProxyConfig, create_app

    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
    )
    app = create_app(config)
    return app.state.proxy


def test_capacity_eviction_is_lru_not_fifo(monkeypatch) -> None:
    """At capacity, the idlest session is evicted — not the oldest-created."""
    import headroom.proxy.server as server_mod

    monkeypatch.setattr(server_mod, "MAX_COMPRESSION_CACHE_SESSIONS", 4)
    proxy = _make_proxy()

    for sid in ("a", "b", "c", "d"):
        proxy._get_compression_cache(sid)
    # "a" is the oldest-created; touch it so "b" becomes the LRU.
    cache_a = proxy._get_compression_cache("a")

    proxy._get_compression_cache("e")

    assert "b" not in proxy._compression_caches
    assert proxy._get_compression_cache("a") is cache_a
    assert "b" not in proxy._compression_cache_last_seen


def test_capacity_eviction_count_respects_small_caps(monkeypatch) -> None:
    """A cap below 4 still evicts at least one session instead of looping."""
    import headroom.proxy.server as server_mod

    monkeypatch.setattr(server_mod, "MAX_COMPRESSION_CACHE_SESSIONS", 2)
    proxy = _make_proxy()

    proxy._get_compression_cache("a")
    proxy._get_compression_cache("b")
    proxy._get_compression_cache("c")

    assert len(proxy._compression_caches) == 2
    assert "a" not in proxy._compression_caches


def test_idle_ttl_sweep_evicts_expired_sessions(monkeypatch) -> None:
    """A session idle past the TTL is reclaimed by the lazy sweep."""
    import headroom.proxy.server as server_mod

    monkeypatch.setattr(server_mod, "COMPRESSION_CACHE_TTL_SECONDS", 100.0)
    proxy = _make_proxy()

    proxy._get_compression_cache("stale")
    proxy._get_compression_cache("fresh")

    now = time.time()
    # Backdate "stale" past the TTL and allow the sweep to run again.
    proxy._compression_cache_last_seen["stale"] = now - 101.0
    proxy._compression_caches_last_cleanup = (
        now - proxy._COMPRESSION_CACHE_CLEANUP_INTERVAL_SECONDS - 1.0
    )

    proxy._get_compression_cache("trigger")

    assert "stale" not in proxy._compression_caches
    assert "stale" not in proxy._compression_cache_last_seen
    assert "fresh" in proxy._compression_caches


def test_access_refreshes_ttl_clock(monkeypatch) -> None:
    """Accessing a session resets its idle clock, so it survives the sweep."""
    import headroom.proxy.server as server_mod

    monkeypatch.setattr(server_mod, "COMPRESSION_CACHE_TTL_SECONDS", 100.0)
    proxy = _make_proxy()

    proxy._get_compression_cache("busy")
    now = time.time()
    proxy._compression_cache_last_seen["busy"] = now - 101.0

    # Access refreshes last_seen before any sweep can see it as expired.
    cache = proxy._get_compression_cache("busy")

    proxy._compression_caches_last_cleanup = (
        now - proxy._COMPRESSION_CACHE_CLEANUP_INTERVAL_SECONDS - 1.0
    )
    proxy._get_compression_cache("trigger")

    assert proxy._get_compression_cache("busy") is cache


def test_sweep_is_rate_limited(monkeypatch) -> None:
    """Within the cleanup interval, even an expired session is not swept."""
    import headroom.proxy.server as server_mod

    monkeypatch.setattr(server_mod, "COMPRESSION_CACHE_TTL_SECONDS", 100.0)
    proxy = _make_proxy()

    proxy._get_compression_cache("stale")
    proxy._compression_cache_last_seen["stale"] = time.time() - 101.0
    # _compression_caches_last_cleanup is recent (set in __init__), so the
    # sweep must not run yet.
    proxy._get_compression_cache("trigger")

    assert "stale" in proxy._compression_caches
