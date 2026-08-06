"""Coalescing must fold concurrent calls without changing what any caller gets.

A fake endpoint stands in for Modal: it records how many HTTP requests happened,
so "did N calls become one request" is directly observable.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from headroom.transforms.kompress_compressor import KompressConfig
from headroom.transforms.kompress_remote import RemoteKompressCompressor

BODY = " ".join(f"word{i}" for i in range(40))


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self._payload


def _payload_for(content: str) -> dict[str, Any]:
    return {
        "compressed": content[: len(content) // 2],
        "original_tokens": len(content.split()),
        "compressed_tokens": len(content.split()) // 2,
        "compression_ratio": 0.5,
    }


class _FakeClient:
    """Counts requests per route and answers both of them."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.batch_sizes: list[int] = []
        self._lock = threading.Lock()

    def post(self, url: str, headers: Any = None, json: Any = None) -> _FakeResponse:
        with self._lock:
            self.calls.append(url)
        if url.endswith("/compress_batch"):
            contents = json["contents"]
            with self._lock:
                self.batch_sizes.append(len(contents))
            return _FakeResponse({"results": [_payload_for(c) for c in contents]})
        return _FakeResponse(_payload_for(json["content"]))

    def close(self) -> None:
        pass


def _compressor(**kwargs: Any) -> tuple[RemoteKompressCompressor, _FakeClient]:
    c = RemoteKompressCompressor(
        endpoint="https://fake.invalid",
        config=KompressConfig(enable_ccr=False),
        # Wide enough that all threads land in the same window on a loaded CI box.
        coalesce_window_s=kwargs.pop("coalesce_window_s", 0.2),
        **kwargs,
    )
    client = _FakeClient()
    c._client = client  # type: ignore[assignment]
    return c, client


def test_concurrent_calls_fold_into_one_batch() -> None:
    c, client = _compressor()
    contents = [f"{BODY} {i}" for i in range(8)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda x: c.compress(x, target_ratio=0.6), contents))

    assert client.batch_sizes == [8], f"expected one fold of 8, got {client.batch_sizes}"
    assert not any(u.endswith("/compress") for u in client.calls)
    # Each caller must get ITS OWN slice, not the leader's.
    for content, result in zip(contents, results):
        assert result.original == content
        assert result.compressed == _payload_for(content)["compressed"]


def test_different_ratios_do_not_ride_together() -> None:
    """The batch route applies one ratio to the whole request."""
    c, client = _compressor()
    with ThreadPoolExecutor(max_workers=4) as pool:
        pool.map(lambda r: c.compress(f"{BODY} {r}", target_ratio=r), [0.3, 0.3, 0.7, 0.7])
    assert sorted(client.batch_sizes) == [2, 2]


def test_max_batch_caps_the_fold() -> None:
    c, client = _compressor(max_batch=3)
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(lambda i: c.compress(f"{BODY} {i}", target_ratio=0.6), range(6)))
    assert client.batch_sizes, "nothing was batched"
    assert max(client.batch_sizes) <= 3


def test_broken_batch_route_falls_back_to_singles() -> None:
    c, client = _compressor()

    real_post = client.post

    def post(url: str, headers: Any = None, json: Any = None) -> _FakeResponse:
        if url.endswith("/compress_batch"):
            raise RuntimeError("no batch route here")
        return real_post(url, headers=headers, json=json)

    client.post = post  # type: ignore[method-assign]
    result = c.compress(BODY, target_ratio=0.6)
    # Compression still happened, via the single route.
    assert result.compressed == _payload_for(BODY)["compressed"]
    assert client.calls[-1].endswith("/compress")


def test_empty_batch_path_disables_coalescing() -> None:
    c, client = _compressor(batch_path="")
    c.compress(BODY, target_ratio=0.6)
    assert client.calls == ["https://fake.invalid/compress"]


@pytest.mark.parametrize("short", ["", "one two three"])
def test_short_content_never_reaches_the_endpoint(short: str) -> None:
    c, client = _compressor()
    assert c.compress(short).compressed == short
    assert client.calls == []
