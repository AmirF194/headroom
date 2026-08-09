"""A turn hook's re-drives are billed calls and must reach token accounting.

A hook that resolves an injected tool call re-drives the model. Both OpenAI
handlers read usage from the FINAL upstream response, so before this every
intermediate call was free as far as Headroom was concerned — a token-saving
feature could spend a whole extra model call hiding a tool belt and report only
the saving, never the overhead.
"""

from __future__ import annotations

from headroom.proxy.handlers.openai import TurnHookUsage

CHAT = {
    "input_key": "prompt_tokens",
    "output_key": "completion_tokens",
    "details_key": "prompt_tokens_details",
}
RESPONSES = {
    "input_key": "input_tokens",
    "output_key": "output_tokens",
    "details_key": "input_tokens_details",
}


def test_starts_empty_so_a_turn_with_no_redrive_is_unchanged() -> None:
    u = TurnHookUsage()
    assert (u.calls, u.input_tokens, u.output_tokens, u.cache_read_tokens) == (0, 0, 0, 0)


def test_chat_shape_accumulates_across_rounds() -> None:
    u = TurnHookUsage()
    u.add(
        {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "prompt_tokens_details": {"cached_tokens": 60},
            }
        },
        **CHAT,
    )
    u.add(
        {
            "usage": {
                "prompt_tokens": 150,
                "completion_tokens": 30,
                "prompt_tokens_details": {"cached_tokens": 90},
            }
        },
        **CHAT,
    )
    assert u.calls == 2
    assert u.input_tokens == 250
    assert u.output_tokens == 50
    assert u.cache_read_tokens == 150


def test_responses_shape_uses_its_own_key_names() -> None:
    u = TurnHookUsage()
    u.add(
        {
            "usage": {
                "input_tokens": 400,
                "output_tokens": 40,
                "input_tokens_details": {"cached_tokens": 300},
            }
        },
        **RESPONSES,
    )
    assert (u.calls, u.input_tokens, u.output_tokens, u.cache_read_tokens) == (1, 400, 40, 300)
    # The chat keys must not read a Responses payload — a silent 0 here would
    # look exactly like "the hook cost nothing".
    v = TurnHookUsage()
    v.add({"usage": {"input_tokens": 400}}, **CHAT)
    assert v.input_tokens == 0
    assert v.calls == 1, "the call still happened even if its shape was unreadable"


def test_never_raises_on_a_shape_it_does_not_recognise() -> None:
    """A hook must not be able to 500 a request by returning something odd."""
    u = TurnHookUsage()
    for payload in (
        None,
        {},
        [],
        "not a dict",
        {"usage": None},
        {"usage": "nope"},
        {"usage": {"prompt_tokens": None, "completion_tokens": "x"}},
        {"usage": {"prompt_tokens": -5, "prompt_tokens_details": "nope"}},
    ):
        u.add(payload, **CHAT)
    assert u.calls == 8, "every attempt counts as a call"
    assert (u.input_tokens, u.output_tokens, u.cache_read_tokens) == (0, 0, 0)


def test_negative_counts_are_floored_not_subtracted() -> None:
    # A provider returning a negative must never reduce the bill.
    u = TurnHookUsage()
    u.add({"usage": {"prompt_tokens": -100, "completion_tokens": 10}}, **CHAT)
    assert u.input_tokens == 0
    assert u.output_tokens == 10
