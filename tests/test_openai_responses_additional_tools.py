"""Codex >= 0.149.0 ``additional_tools`` normalization (#3185).

Codex CLI 0.149.0 sends tool definitions as ``input`` items of type
``additional_tools`` instead of a top-level ``tools`` array for models its
capability cache flags (``gpt-5.6-sol``). Without the lift, every tools
consumer (schema compaction, output-shaper stratum, tools token accounting)
sees a tool-less request and records zero tool-schema savings.
"""

from __future__ import annotations

import copy
from typing import Any

from headroom.proxy.handlers.openai import (
    _compact_openai_responses_tools,
    _lift_codex_additional_tools,
)


def _verbose_tool(name: str) -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": " ".join(["Runs a shell command in the workspace."] * 30),
        "parameters": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "title": name,
            "properties": {
                "command": {
                    "type": "array",
                    "title": "command",
                    "items": {"type": "string"},
                }
            },
            "required": ["command"],
        },
    }


def _codex_0149_payload() -> dict[str, Any]:
    return {
        "model": "gpt-5.6-sol",
        "include": ["reasoning.encrypted_content"],
        "reasoning": {"effort": "low", "context": "all_turns"},
        "tool_choice": "auto",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "do the thing"}],
            },
            {
                "type": "additional_tools",
                "tools": [_verbose_tool("shell"), _verbose_tool("update_plan")],
            },
        ],
    }


def test_lift_moves_additional_tools_to_top_level() -> None:
    payload = _codex_0149_payload()

    lifted = _lift_codex_additional_tools(payload)

    assert lifted == 2
    assert [t["name"] for t in payload["tools"]] == ["shell", "update_plan"]
    # The carrier item is dropped; every other input item survives in order.
    assert [item["type"] for item in payload["input"]] == ["message"]


def test_lift_concatenates_multiple_carrier_items() -> None:
    payload = _codex_0149_payload()
    payload["input"].append({"type": "additional_tools", "tools": [_verbose_tool("view_image")]})

    lifted = _lift_codex_additional_tools(payload)

    assert lifted == 3
    assert [t["name"] for t in payload["tools"]] == ["shell", "update_plan", "view_image"]


def test_lift_is_noop_when_top_level_tools_present() -> None:
    payload = _codex_0149_payload()
    payload["tools"] = [_verbose_tool("shell")]
    before = copy.deepcopy(payload)

    assert _lift_codex_additional_tools(payload) == 0
    assert payload == before


def test_lift_is_noop_without_carrier_items() -> None:
    payload = _codex_0149_payload()
    payload["input"] = [item for item in payload["input"] if item["type"] != "additional_tools"]
    before = copy.deepcopy(payload)

    assert _lift_codex_additional_tools(payload) == 0
    assert payload == before

    assert _lift_codex_additional_tools({"model": "gpt-5.6-sol", "input": "not-a-list"}) == 0
    assert _lift_codex_additional_tools("not-a-dict") == 0  # type: ignore[arg-type]


def test_lift_disabled_by_kill_switch(monkeypatch) -> None:
    monkeypatch.setenv("HEADROOM_CODEX_ADDITIONAL_TOOLS_LIFT", "0")
    payload = _codex_0149_payload()
    before = copy.deepcopy(payload)

    assert _lift_codex_additional_tools(payload) == 0
    assert payload == before


def test_lift_logs_with_request_id(caplog) -> None:
    payload = _codex_0149_payload()

    with caplog.at_level("INFO", logger="headroom.proxy"):
        assert _lift_codex_additional_tools(payload, request_id="req_test") == 2

    assert any(
        "req_test" in message and "additional_tools" in message for message in caplog.messages
    )


def test_lift_preserves_empty_carrier_items() -> None:
    payload = _codex_0149_payload()
    payload["input"].append({"type": "additional_tools", "tools": []})

    lifted = _lift_codex_additional_tools(payload)

    # The empty carrier holds no definitions to lift; it is preserved rather
    # than invented into an empty top-level array.
    assert lifted == 2
    assert [item["type"] for item in payload["input"]] == ["message", "additional_tools"]


def test_lifted_tools_reach_schema_compaction() -> None:
    payload = _codex_0149_payload()

    # Without the lift: compaction sees no tools and returns unmodified —
    # the exact production failure.
    _, modified, _, _ = _compact_openai_responses_tools(copy.deepcopy(payload))
    assert modified is False

    _lift_codex_additional_tools(payload)
    compacted, modified, before_bytes, after_bytes = _compact_openai_responses_tools(payload)

    assert modified is True
    assert after_bytes < before_bytes
    # Compaction preserves the invocation shape the model needs.
    assert [t["name"] for t in compacted["tools"]] == ["shell", "update_plan"]
