"""Remote Kompress: offload ML compression to a hosted ``/compress`` endpoint.

Lets a sandboxed proxy — installed WITHOUT the ``[ml]`` extra (no torch/onnx) —
still run Kompress by calling a remote endpoint over HTTP. The class mirrors
:class:`~headroom.transforms.kompress_compressor.KompressCompressor`'s public
surface (``is_ready`` / ``preload`` / ``ensure_background_load`` / ``compress``),
so it is a drop-in at the ContentRouter seam.

Only the model inference is remote. The CCR store + retrieval marker stay
proxy-local (the endpoint is stateless, ``enable_ccr=False``), so
``headroom_retrieve`` keeps working and original content never persists off-box.

Enabled by ``HEADROOM_KOMPRESS_ENDPOINT`` — see ``ContentRouter._get_kompress``.

# Bring-your-own deployment

The endpoint does not have to be Headroom Labs'. An org can pull the Kompress
weights from HuggingFace, serve them on its own stack (vLLM, TorchServe,
SageMaker, KServe, a bare FastAPI box) and point Headroom at it. Nothing about
this class is Modal-specific, and no credential is required — auth is whatever
the operator's own infrastructure expects, including none at all:

    HEADROOM_KOMPRESS_ENDPOINT          https://ml.internal.acme.com
    HEADROOM_KOMPRESS_ENDPOINT_PATH     /compress   (default; set empty to use
                                        the endpoint URL verbatim)
    HEADROOM_KOMPRESS_ENDPOINT_TOKEN    optional; sent as `Authorization: Bearer`
    HEADROOM_KOMPRESS_ENDPOINT_HEADERS  optional; `k=v,k2=v2`, applied last so it
                                        can replace the Authorization header for
                                        stacks that want `x-api-key` or similar

Both new knobs default to today's behaviour: with only
``HEADROOM_KOMPRESS_ENDPOINT`` set, the request is byte-identical to before —
``POST <endpoint>/compress`` with an optional Bearer token. Existing Modal
deployments need no change.

# The HTTP contract

Deliberately small, so a shim in front of an existing inference server is a few
lines. ``POST <endpoint><path>``:

    request   {"content": "<text>", "target_ratio": 0.5 | null}
    response  {"compressed": "<text>",          # REQUIRED, must be a string
               "original_tokens": int,          # optional, defaults to word count
               "compressed_tokens": int,        # optional, defaults to word count
               "compression_ratio": float,      # optional, defaults to 1.0
               "model_used": str}               # optional

``compressed`` is the only required field; every other value is derived if
absent. Any non-2xx, timeout, malformed field, or missing ``compressed`` makes
this pass the content through verbatim — a broken endpoint costs compression,
never correctness.

# Request coalescing, and the one rule that governs it

A hosted endpoint's cost is dominated by the round trip, not the model: a call
takes ~470ms whether the payload is 10 characters or 8,000. So a caller that fans
out — ``ContentRouter._compress_mixed`` splitting a README into 35 sections —
pays 35 × RTT, ~14s of near-pure network wait for ~7s of actual inference.

**The rule: never let concurrent requests reach the model.** A Kompress server
admits ONE inference at a time (``_default_max_concurrent`` returns 1 for the
onnx backend), and a caller that misses that slot before its deadline gets its
chunk back UNCOMPRESSED — ``compress`` logs "execution saturated … skipping
chunk". That is load-shedding, not a race, so overlapping requests do not queue,
they silently lose compression. Measured with 24 cache-cold variants of identical
content: sequential singles 89 tokens, 16 concurrent singles 143 tokens (+61%).

Coalescing is how we get the round trips back without breaking that rule. When
several threads call :meth:`compress` at once, the first becomes a leader, gathers
its siblings for a few milliseconds, and issues **one** ``/compress_batch``
request. N concurrent callers become one request, which the server compresses in
a single thread — so the fan-out never turns into concurrent inference. Verified
cache-cold on 35 sections: 14.0s → 6.9s (2.0x) with output identical to
sequential (0 of 35 sections differing by more than one token).

Two traps when changing any of this. The server memoizes by
``(model, ratio, content)``, so an A/B that reuses content replays earlier answers
and looks clean — always verify with a unique nonce per call and assert the
response is not ``cached``. And this client fails open, so a regression surfaces
as "less compression", never as an error.

Requires a server whose batch route compresses in one thread (fixed 2026-08).
``HEADROOM_KOMPRESS_ENDPOINT_BATCH_PATH=""`` disables coalescing for an endpoint
that has no batch route, or one whose batch route still fans out internally.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .kompress_compressor import KompressConfig, KompressResult, store_kompress_in_ccr

logger = logging.getLogger(__name__)

# Appended to the configured endpoint unless overridden. An operator whose stack
# already serves a full path (``/v1/models/kompress:predict``) sets
# HEADROOM_KOMPRESS_ENDPOINT_PATH="" and gives the complete URL instead.
DEFAULT_ENDPOINT_PATH = "/compress"

# Batch sibling of DEFAULT_ENDPOINT_PATH. Takes ``{"contents": [...]}`` and
# returns ``{"results": [...]}`` whose entries have the same shape as a single
# /compress response.
DEFAULT_BATCH_PATH = "/compress_batch"

# How long a leader holds the queue open collecting siblings. Only ever costs
# latency when a single call is in flight; with a fan-out in progress the siblings
# are already queued and the batch closes on the size cap instead. Against a
# ~470ms round trip, a few ms of gathering is noise.
DEFAULT_COALESCE_WINDOW_S = 0.005

# Upper bound on one batch request, mirrored by the server's own cap. Exists to
# bound request size and blast radius; the endpoint stays flat well past it.
DEFAULT_MAX_BATCH = 64


def parse_endpoint_headers(raw: str | None) -> dict[str, str]:
    """Parse ``k=v,k2=v2`` into a header dict.

    Same format as ``HEADROOM_OTEL_METRICS_HEADERS`` so operators meet one
    convention. Reimplemented rather than imported from
    :mod:`headroom.observability.metrics`, which pulls in opentelemetry at module
    scope — remote Kompress exists precisely so a proxy can run without heavy
    optional deps, so it must not drag one in through a parsing helper.
    """
    pairs: dict[str, str] = {}
    for item in (raw or "").split(","):
        part = item.strip()
        if not part or "=" not in part:
            continue
        key, _, value = part.partition("=")
        key, value = key.strip(), value.strip()
        if key and value:
            pairs[key] = value
    return pairs


# Below this word count local Kompress passes through verbatim (KompressCompressor
# .compress); mirror it so we never pay a round-trip on a trivially small block.
_MIN_WORDS = 10

# Accept-any-shrink CCR gate, identical to KompressCompressor.compress: only
# store + mark when the shrink is worth the retrieval marker's own cost.
_CCR_RATIO_GATE = 0.8


@dataclass
class _Waiter:
    """One caller parked on a batch that a leader thread will fire."""

    content: str
    done: threading.Event = field(default_factory=threading.Event)
    payload: dict[str, Any] | None = None


class RemoteKompressCompressor:
    """Drop-in for KompressCompressor that POSTs to a hosted ``/compress`` endpoint.

    Fails OPEN: any network/HTTP error returns the content verbatim so a flaky
    endpoint degrades compression rather than breaking the proxy.
    """

    name = "kompress_compressor"

    def __init__(
        self,
        endpoint: str,
        token: str | None = None,
        config: KompressConfig | None = None,
        timeout: float = 20.0,
        path: str | None = DEFAULT_ENDPOINT_PATH,
        headers: dict[str, str] | None = None,
        batch_path: str | None = DEFAULT_BATCH_PATH,
        coalesce_window_s: float = DEFAULT_COALESCE_WINDOW_S,
        max_batch: int = DEFAULT_MAX_BATCH,
    ) -> None:
        self.config = config or KompressConfig()
        # Default keeps the pre-existing behaviour exactly: <endpoint>/compress.
        # An empty (or None) path means the caller has supplied a complete URL —
        # needed because real inference servers do not serve at /compress
        # (TorchServe /predictions/<model>, KServe /v1/models/<name>:predict,
        # SageMaker /invocations), and appending to those yields a 404.
        if path:
            suffix = path if path.startswith("/") else "/" + path
            self._url = endpoint.rstrip("/") + suffix
        else:
            self._url = endpoint
        self._headers = {"content-type": "application/json"}
        if token:
            self._headers["authorization"] = f"Bearer {token}"
        # Applied last on purpose: lets an operator replace `authorization` with
        # whatever their gateway wants (x-api-key, a signed header, a tenant id)
        # without needing a separate auth-scheme setting.
        if headers:
            self._headers.update(headers)
        # The batch route is a sibling of the single one, so it hangs off the same
        # base. When the caller supplied a complete URL (path=""), there is nothing
        # to hang it from and coalescing stays off rather than guessing.
        if batch_path and path:
            suffix = batch_path if batch_path.startswith("/") else "/" + batch_path
            self._batch_url: str | None = endpoint.rstrip("/") + suffix
        else:
            self._batch_url = None
        self._coalesce_window_s = max(0.0, coalesce_window_s)
        self._max_batch = max(1, max_batch)
        # Waiters are keyed by target_ratio: the batch route applies one ratio to
        # the whole request, so calls wanting different ratios must not ride
        # together.
        self._pending: dict[float | None, list[_Waiter]] = {}
        self._pending_lock = threading.Lock()
        self._timeout = timeout
        # httpx.Client is safe to share across the proxy's worker threads.
        self._client = httpx.Client(timeout=timeout)

    @property
    def url(self) -> str:
        """The resolved POST target.

        Logged when the router builds this, so a wrong ``_PATH`` shows up as a
        visibly odd URL at startup rather than as silent pass-through later —
        the fail-open contract means a 404 never surfaces as an error.
        """
        return self._url

    # Nothing to load locally; short-circuit the router straight to compress().
    def is_ready(self) -> bool:
        return True

    def ready_backend(self) -> str | None:
        return "remote"

    def preload(self, *, allow_download: bool = True) -> str:
        return "remote"

    def ensure_background_load(self) -> None:
        return None

    def _passthrough(self, content: str, n_words: int) -> KompressResult:
        return KompressResult(
            compressed=content,
            original=content,
            original_tokens=n_words,
            compressed_tokens=n_words,
            compression_ratio=1.0,
            model_used=self.config.model_id,
        )

    def _result_from_payload(self, content: str, data: dict[str, Any]) -> KompressResult:
        """Build a result from one endpoint payload.

        Raises on a malformed payload; every caller runs inside a fail-open guard.
        Coerce the numeric/string metadata here rather than trusting it: a 200 with
        a non-numeric string or an explicit JSON null (``data.get`` returns None for
        a present key, and float(None)/int(None) raise) would otherwise escape and
        break the proxy request, defeating the fail-open contract.
        """
        compressed = data["compressed"]
        if not isinstance(compressed, str):
            raise TypeError("remote Kompress response field 'compressed' must be a string")
        n_words = len(content.split())
        return KompressResult(
            compressed=compressed,
            original=content,
            original_tokens=int(data.get("original_tokens", n_words)),
            compressed_tokens=int(data.get("compressed_tokens", len(compressed.split()))),
            compression_ratio=float(data.get("compression_ratio", 1.0)),
            model_used=str(data.get("model_used", self.config.model_id)),
        )

    def _store_ccr(self, content: str, result: KompressResult) -> KompressResult:
        """Register the original in CCR and append the retrieval marker.

        CCR stays PROXY-LOCAL: the endpoint is stateless (enable_ccr=False), so the
        mapping and marker live here — same policy and marker format as
        KompressCompressor.compress.
        """
        if not (self.config.enable_ccr and result.compression_ratio < _CCR_RATIO_GATE):
            return result
        cache_key = store_kompress_in_ccr(content, result.compressed, result.original_tokens)
        if cache_key:
            result.cache_key = cache_key
            # Report the source line span so a reader can tell content was
            # compressed away rather than absent (#2586).
            source_lines = content.count("\n") + 1
            line_word = "line" if source_lines == 1 else "lines"
            result.compressed += (
                f"\n[{result.original_tokens} items compressed to "
                f"{result.compressed_tokens} (from {source_lines} source {line_word})."
                f" Retrieve more: hash={cache_key}]"
            )
        return result

    def _fetch_batch(self, contents: list[str], target_ratio: float | None) -> list[dict[str, Any]]:
        """One round-trip for N blocks. Raises unless every block came back."""
        assert self._batch_url is not None
        resp = self._client.post(
            self._batch_url,
            headers=self._headers,
            json={"contents": contents, "target_ratio": target_ratio},
        )
        resp.raise_for_status()
        results = resp.json()["results"]
        if not isinstance(results, list) or len(results) < len(contents):
            raise ValueError(
                f"remote Kompress batch returned {len(results)} results for {len(contents)} inputs"
            )
        return [dict(r) for r in results[: len(contents)]]

    def _run_batch(self, batch: list[_Waiter], target_ratio: float | None) -> None:
        """Fire one batch and hand each waiter its slice. Never raises."""
        try:
            payloads = self._fetch_batch([w.content for w in batch], target_ratio)
        except Exception as e:
            # Leave every payload None: each waiter falls back to a single call,
            # so a broken batch route costs latency, not compression.
            logger.warning("Remote Kompress batch failed (%s); falling back to singles", e)
        else:
            for waiter, payload in zip(batch, payloads):
                waiter.payload = payload
        finally:
            for waiter in batch:
                waiter.done.set()

    def _coalesced_payload(self, content: str, target_ratio: float | None) -> dict[str, Any] | None:
        """Ride a batch with whatever else arrives inside the window.

        The first caller for a ratio becomes the leader: it waits out the window,
        claims the queue, and fires one request for everyone. Followers just park.
        Returns None when the batch could not answer — the caller then does its own
        single call, which is why this never needs to raise.
        """
        me = _Waiter(content=content)
        with self._pending_lock:
            queue = self._pending.setdefault(target_ratio, [])
            queue.append(me)
            leader = len(queue) == 1
            full = len(queue) >= self._max_batch

        if not leader:
            if not full:
                # The leader fires within the window; the batch itself then takes as
                # long as the request does, hence the client timeout as the bound.
                me.done.wait(timeout=self._coalesce_window_s + self._timeout)
                return me.payload
            # Queue is full and nobody is coming: fire it now rather than making
            # everyone wait out the leader's window.

        if leader and self._coalesce_window_s:
            time.sleep(self._coalesce_window_s)

        with self._pending_lock:
            batch = self._pending.pop(target_ratio, [])
        if not batch:
            # Another thread already claimed this queue (a full-queue follower
            # racing the leader). Wait for it like any other follower.
            me.done.wait(timeout=self._timeout)
            return me.payload

        self._run_batch(batch, target_ratio)
        return me.payload

    def compress(
        self,
        content: str,
        context: str = "",
        content_type: str | None = None,
        question: str | None = None,
        target_ratio: float | None = None,
        *,
        allow_download: bool = True,
    ) -> KompressResult:
        n_words = len(content.split())
        if n_words < _MIN_WORDS:
            return self._passthrough(content, n_words)

        if self._batch_url is not None:
            try:
                payload = self._coalesced_payload(content, target_ratio)
                if payload is not None:
                    return self._store_ccr(content, self._result_from_payload(content, payload))
            except Exception as e:  # fail OPEN, then still try the single route
                logger.warning("Remote Kompress coalesced call failed (%s)", e)

        try:
            resp = self._client.post(
                self._url,
                headers=self._headers,
                json={"content": content, "target_ratio": target_ratio},
            )
            resp.raise_for_status()
            result = self._result_from_payload(content, resp.json())
        except Exception as e:  # fail OPEN — never break the proxy on a bad endpoint
            logger.warning("Remote Kompress failed (%s); passing through", e)
            return self._passthrough(content, n_words)

        return self._store_ccr(content, result)

    def close(self) -> None:
        self._client.close()
