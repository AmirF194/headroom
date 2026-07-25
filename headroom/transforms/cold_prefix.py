"""Cold-prefix cache-miss hook: decide what to rewrite when the prompt cache is dead.

When the prefix cache has lapsed (idle since the last turn exceeded the provider
TTL), forwarding the byte-identical prefix buys nothing — the cache is gone — so
this is the safe moment for rewrites that would otherwise bust a warm cache. What
we rewrite depends on the model's reasoning shape:

* **Plain-text reasoning (Kimi / GLM / DeepSeek-R1)** — reasoning is resent as
  billable text (`reasoning_content` field or inline ``<think>``). On a cold turn
  we can DROP the old reasoning outright (full block), not just Kompress it.
* **Encrypted reasoning (Claude / OpenAI Codex)** — the reasoning is an opaque
  server-side handle billed free/light; touching it saves nothing. Instead, on a
  cold turn, dedupe + drop superseded reads across the (now-unfreezable) prefix.

This module is the *decision* surface (is-it-cold + which-shape); the handlers
apply the chosen rewrite. Never raises.

Cache note: dropping/deduping the prefix is cache-safe here **because the cache is
already dead** — nothing to bust. The one cost is the cold turn re-caches a
smaller prefix, which then benefits every subsequent warm turn until the next
lapse. (For plain-text reasoning, deterministic Kompress-every-turn — see
``thinking_compactor`` — stays cache-stable on warm turns; the cold DROP is the
extra, aggressive step reserved for when the cache is confirmed gone.)
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

_DEFAULT_MARGIN_SECONDS = 60.0


def is_cold_prefix(prefix_tracker: Any, *, margin_seconds: float = _DEFAULT_MARGIN_SECONDS) -> bool:
    """True when the prompt-cache prefix has (confidently) lapsed.

    Uses the idle gap captured at fetch (``_idle_seconds_at_fetch``) vs the
    provider cache TTL (``resolved_cache_ttl_seconds``). The margin makes us
    *confident* it's past TTL before we treat it as cold — we'd rather miss a
    just-barely-expired cache than mistakenly rewrite a still-warm one. Returns
    False on any missing attribute (conservative: never assume cold).
    """
    try:
        idle = float(getattr(prefix_tracker, "_idle_seconds_at_fetch", 0.0) or 0.0)
        ttl_fn = getattr(prefix_tracker, "resolved_cache_ttl_seconds", None)
        if ttl_fn is None:
            return False
        ttl = float(ttl_fn())
    except Exception:
        return False
    return idle > ttl + margin_seconds


def has_plaintext_reasoning(messages: list[dict[str, Any]]) -> bool:
    """True if any assistant turn carries reasoning as PLAIN TEXT we can drop/compress.

    Two shapes: a Kimi-style ``reasoning_content`` field, or an inline
    ``<think>…</think>`` span in string content (GLM / DeepSeek-R1). Encrypted
    reasoning (Claude signature / OpenAI ``encrypted_content``) never appears in
    these forms, so this is False for those — which routes them to the dedupe /
    superseded-read branch instead.
    """
    for m in messages:
        if m.get("role") != "assistant":
            continue
        rc = m.get("reasoning_content")
        if isinstance(rc, str) and rc.strip():
            return True
        c = m.get("content")
        if isinstance(c, str) and "<think>" in c and "</think>" in c:
            return True
    return False


def cold_recompact_messages(
    messages: list[dict[str, Any]], *, tokenizer: Any, context: str = ""
) -> tuple[list[dict[str, Any]], list[str]]:
    """Lossless whole-prefix recompaction for a confirmed-cold turn.

    Runs a fresh lossless + cross-turn-dedup ContentRouter over the *entire*
    conversation (``frozen_message_count=0``): superseded/stale-read drop +
    verbatim dedupe + lossless folds — the safe, information-preserving rewrites
    (never lossy, so old context the model relies on is not mangled). Used when
    the prompt cache is dead (idle past TTL) and the byte-identical splice would
    preserve nothing. Lossless + prefix-monotonic ⇒ deterministic per content ⇒
    the recompacted prefix re-caches and stays byte-stable on later warm turns.

    Returns (new_messages, transforms_applied). Fail-open: returns the input
    unchanged on any error (never breaks the request).
    """
    try:
        from headroom.transforms.content_router import (
            ContentRouter,
            ContentRouterConfig,
        )

        router = ContentRouter(ContentRouterConfig(lossless=True, enable_cross_turn_dedup=True))
        res = router.apply(list(messages), tokenizer, frozen_message_count=0, context=context)
        return res.messages, list(res.transforms_applied)
    except Exception as e:  # never break the request
        log.warning("cold-prefix recompaction failed (%s); leaving prefix unchanged", e)
        return list(messages), []


def _demo() -> None:
    class _T:
        def __init__(self, idle: float, ttl: float) -> None:
            self._idle_seconds_at_fetch = idle
            self._ttl = ttl

        def resolved_cache_ttl_seconds(self) -> float:
            return self._ttl

    assert is_cold_prefix(_T(400, 300))  # 400 idle > 300 ttl + 60 margin? 400 > 360 ✓
    assert not is_cold_prefix(_T(350, 300))  # 350 < 360 → warm
    assert not is_cold_prefix(_T(10, 300))  # back-to-back → warm
    assert not is_cold_prefix(object())  # missing attrs → conservative False

    assert has_plaintext_reasoning([{"role": "assistant", "reasoning_content": "abc"}])
    assert has_plaintext_reasoning([{"role": "assistant", "content": "<think>x</think> y"}])
    assert not has_plaintext_reasoning([{"role": "assistant", "content": "plain"}])
    assert not has_plaintext_reasoning([{"role": "user", "reasoning_content": "x"}])
    print("cold_prefix self-check OK")


if __name__ == "__main__":
    _demo()
