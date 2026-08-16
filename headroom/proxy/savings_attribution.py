"""Bounded attribution for savings that do not have a built-in metric."""

from __future__ import annotations

import base64
import json
import re
from collections.abc import MutableMapping
from typing import Any

SAVINGS_ATTRIBUTION_TAG = "_headroom_savings_attribution"
_NAME_RE = re.compile(r"[^a-z0-9_.-]+")
MAX_SOURCES = 32
_SCOPE_KEY = "headroom_savings_attribution"

# Per-request stage timings contributed from outside the handler, merged into
# ``RequestOutcome.pipeline_timing`` at the outcome funnel.
#
# An ASGI middleware wraps the handler, so every millisecond it spends lands in
# the client's latency while ``overhead_ms`` -- measured inside the handler --
# stays flat. An extension that halves the bill and adds 200ms per request is a
# trade the operator has to be able to see both halves of, and until now only
# one half reached the dashboard.
STAGE_TIMING_TAG = "_headroom_stage_timing"
_TIMING_SCOPE_KEY = "headroom_stage_timing"

# Stage names are extension-supplied, so they are capped like every other
# client-influenced label in this proxy (see MAX_DISTINCT_MODELS).
MAX_STAGES = 16

# Namespace, so an extension can never shadow a built-in transform's timing --
# ``deep_copy`` reported by a plugin and ``deep_copy`` reported by the pipeline
# must not accumulate into the same series.
STAGE_PREFIX = "ext:"


def _source_name(value: object) -> str:
    name = _NAME_RE.sub("_", str(value or "other").strip().lower()).strip("_.-")
    return (name or "other")[:64]


def _ledger(tags: MutableMapping[str, Any]) -> list[dict[str, Any]]:
    current = tags.get(SAVINGS_ATTRIBUTION_TAG)
    if isinstance(current, list):
        return current
    current = []
    tags[SAVINGS_ATTRIBUTION_TAG] = current
    return current


def bind_scope(tags: MutableMapping[str, Any], scope: MutableMapping[str, Any]) -> None:
    """Share the savings and timing ledgers between ASGI middleware and the handler.

    Both are bound together because an extension that reports one usually
    reports the other, and a handler that binds only savings would drop the
    timings silently -- which is the failure this call is here to prevent.
    """
    state = scope.setdefault("state", {})
    ledger = state.get(_SCOPE_KEY)
    if not isinstance(ledger, list):
        ledger = []
        state[_SCOPE_KEY] = ledger
    tags[SAVINGS_ATTRIBUTION_TAG] = ledger

    timings = state.get(_TIMING_SCOPE_KEY)
    if not isinstance(timings, dict):
        timings = {}
        state[_TIMING_SCOPE_KEY] = timings
    tags[STAGE_TIMING_TAG] = timings


def record_scope_savings(scope: MutableMapping[str, Any], source: object, **values: Any) -> None:
    state = scope.setdefault("state", {})
    ledger = state.get(_SCOPE_KEY)
    if not isinstance(ledger, list):
        ledger = []
        state[_SCOPE_KEY] = ledger
    record_savings({SAVINGS_ATTRIBUTION_TAG: ledger}, source, **values)


def record_scope_timing(scope: MutableMapping[str, Any], stage: object, ms: float) -> None:
    """Attribute milliseconds spent outside the handler to a named stage.

    Additive within one request, so a middleware that works in two passes
    (before and after ``call_next``) reports each and gets their sum. Never
    raises and never changes a response: a plugin's telemetry must not be able
    to break the request it is describing.
    """
    try:
        elapsed = float(ms)
    except (TypeError, ValueError):
        return
    if not elapsed > 0.0:
        # Non-positive is either a clock artifact or nothing happening. Either
        # way it is not a measurement, and averaging it in would drag the mean
        # toward zero exactly where the stage is cheapest to ignore.
        return

    state = scope.setdefault("state", {})
    timings = state.get(_TIMING_SCOPE_KEY)
    if not isinstance(timings, dict):
        timings = {}
        state[_TIMING_SCOPE_KEY] = timings

    name = STAGE_PREFIX + _source_name(stage)
    if name not in timings and len(timings) >= MAX_STAGES:
        return
    timings[name] = round(float(timings.get(name, 0.0)) + elapsed, 4)


def timings_from_tags(tags: MutableMapping[str, Any] | None) -> dict[str, float]:
    """Extension stage timings carried on the request's tags, if any."""
    raw = (tags or {}).get(STAGE_TIMING_TAG)
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for name, value in list(raw.items())[:MAX_STAGES]:
        try:
            elapsed = float(value)
        except (TypeError, ValueError):
            continue
        if elapsed > 0.0:
            out[str(name)] = elapsed
    return out


def record_savings(
    tags: MutableMapping[str, Any],
    source: object,
    *,
    tokens: int = 0,
    usd: float = 0.0,
    realized: bool = True,
    estimated: bool = False,
    details: dict[str, Any] | None = None,
) -> None:
    """Attribute savings to a source; this never changes headline totals."""
    ledger = _ledger(tags)
    if len(ledger) >= MAX_SOURCES:
        return
    item: dict[str, Any] = {
        "source": _source_name(source),
        "realized": bool(realized),
        "estimated": bool(estimated),
        "tokens": max(0, int(tokens or 0)),
        "usd": round(float(usd or 0.0), 12),
    }
    if details:
        item["details"] = {
            _source_name(key): value
            for key, value in list(details.items())[:12]
            if isinstance(value, (str, int, float, bool)) or value is None
        }
    ledger.append(item)


def from_tags(tags: MutableMapping[str, Any] | None) -> list[dict[str, Any]]:
    raw = (tags or {}).get(SAVINGS_ATTRIBUTION_TAG)
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw[:MAX_SOURCES] if isinstance(item, dict)]


_INTERNAL_TAGS = frozenset({SAVINGS_ATTRIBUTION_TAG, STAGE_TIMING_TAG})


def public_tags(tags: MutableMapping[str, Any] | None) -> dict[str, Any]:
    """Tags minus the internal ledgers, which are structures rather than labels.

    They are carried on ``tags`` because that is the one dict that reaches the
    outcome funnel from every handler; letting them through to ``RequestLog``
    would put a list and a dict into a string-keyed label store.
    """
    return {key: value for key, value in (tags or {}).items() if key not in _INTERNAL_TAGS}


def encode(items: list[dict[str, Any]]) -> str:
    if not items:
        return "none"
    payload = json.dumps(items, separators=(",", ":"), sort_keys=True).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")


def decode(value: str) -> list[dict[str, Any]]:
    if not value or value == "none":
        return []
    try:
        padded = value + "=" * (-len(value) % 4)
        decoded = json.loads(base64.urlsafe_b64decode(padded).decode())
    except Exception:
        return []
    return (
        [dict(item) for item in decoded if isinstance(item, dict)]
        if isinstance(decoded, list)
        else []
    )
