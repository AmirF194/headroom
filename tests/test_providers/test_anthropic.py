"""Tests for Anthropic provider."""

import pytest


class TestAnthropicModelSanitization:
    def test_sanitize_model_id_removes_ansi_escape_sequences(self):
        from headroom.providers.anthropic import sanitize_anthropic_model_id

        assert sanitize_anthropic_model_id("claude-opus-4-8\x1b[1m") == "claude-opus-4-8"

    def test_sanitize_model_id_removes_displayed_style_suffix(self):
        from headroom.providers.anthropic import sanitize_anthropic_model_id

        assert sanitize_anthropic_model_id("claude-opus-4-8[1m]") == "claude-opus-4-8"
        assert sanitize_anthropic_model_id("glm-5.2[1m]") == "glm-5.2"

    def test_sanitize_model_metadata_cleans_nested_model_ids(self):
        from headroom.providers.anthropic import sanitize_anthropic_model_metadata

        payload = {
            "data": [
                {"id": "claude-opus-4-8\x1b[1m", "display_name": "Claude Opus 4.8"},
                {"id": "claude-sonnet-4-5[1m]"},
            ],
            "model": "claude-opus-4-8[1m]",
        }

        assert sanitize_anthropic_model_metadata(payload) == {
            "data": [
                {"id": "claude-opus-4-8", "display_name": "Claude Opus 4.8"},
                {"id": "claude-sonnet-4-5"},
            ],
            "model": "claude-opus-4-8",
        }


class TestContext1MSuffix:
    """`[1m]` is a 1M-context tier request, not just an ANSI artifact (#1158).

    Claude Code appends `[1m]` to a model id and only then sends the
    `context-1m` beta header, so the real upstream window is 1M even when the
    base model defaults to 200K. The suffix must still be stripped off the wire
    (upstream rejects it, #2027) but must not be lost before we size the budget.
    """

    @pytest.fixture
    def provider(self):
        from headroom.providers.anthropic import AnthropicProvider

        return AnthropicProvider()

    def test_1m_suffix_is_detected(self):
        from headroom.providers.anthropic import has_context_1m_suffix

        assert has_context_1m_suffix("claude-sonnet-4-5[1m]")
        assert has_context_1m_suffix("claude-sonnet-4-5[1m][1m]")
        assert not has_context_1m_suffix("claude-sonnet-4-5")

    def test_ansi_artifacts_are_not_mistaken_for_a_tier_request(self):
        from headroom.providers.anthropic import has_context_1m_suffix

        # A dangling reset, a compound style, and a real escape sequence are
        # terminal noise -- none of them means "give me 1M".
        assert not has_context_1m_suffix("claude-sonnet-4-5[0m]")
        assert not has_context_1m_suffix("claude-sonnet-4-5[1;32m]")
        assert not has_context_1m_suffix("\x1b[1mclaude-sonnet-4-5\x1b[0m")

    def test_1m_suffix_raises_a_200k_model_to_1m(self, provider):
        # The regression: sanitizing before the lookup resolved this to the
        # base model's 200K window, so a 1M request was budgeted at 1/5 size.
        assert provider.get_context_limit("claude-sonnet-4-5") == 200_000
        assert provider.get_context_limit("claude-sonnet-4-5[1m]") == 1_000_000

    def test_1m_suffix_never_lowers_an_already_larger_window(self, provider):
        # max(), not a flat assignment: a base model wider than 1M keeps its own.
        assert provider.get_context_limit("claude-opus-5[1m]") >= 1_000_000

    def test_ansi_artifact_does_not_inflate_the_window(self, provider):
        assert provider.get_context_limit("claude-sonnet-4-5[0m]") == 200_000
        assert provider.get_context_limit("\x1b[1mclaude-sonnet-4-5\x1b[0m") == 200_000

    def test_wire_model_id_still_drops_the_suffix(self):
        # Upstream rejects `[1m]`; the tier fix must not regress #2027.
        from headroom.providers.anthropic import sanitize_anthropic_model_id

        assert sanitize_anthropic_model_id("claude-sonnet-4-5[1m]") == "claude-sonnet-4-5"


class TestAnthropicTokenCounting:
    @pytest.fixture
    def anthropic_provider(self):
        from headroom.providers.anthropic import AnthropicProvider

        return AnthropicProvider()

    def test_count_text_fallback(self, anthropic_provider):
        # Without API client, should use tiktoken fallback
        counter = anthropic_provider.get_token_counter("claude-3-5-sonnet-20241022")
        count = counter.count_text("Hello world")
        assert count > 0

    def test_count_messages_basic(self, anthropic_provider):
        counter = anthropic_provider.get_token_counter("claude-3-5-sonnet-20241022")
        messages = [{"role": "user", "content": "Hello"}]
        count = counter.count_messages(messages)
        assert count > 0

    def test_count_messages_tolerates_null_tool_calls(self, anthropic_provider):
        # OpenAI-format assistant messages routinely carry `tool_calls: null`
        # (and occasionally `function: null`) on a no-tool turn. The estimated
        # counter iterated the value after only a key-presence check, so it
        # raised `TypeError: 'NoneType' object is not iterable`.
        counter = anthropic_provider.get_token_counter("claude-3-5-sonnet-20241022")
        messages = [
            {"role": "assistant", "content": "hi", "tool_calls": None},
            {"role": "assistant", "content": "x", "tool_calls": [{"id": "a", "function": None}]},
        ]
        assert counter.count_messages(messages) > 0

    def test_count_text_allows_literal_special_tokens(self, anthropic_provider):
        counter = anthropic_provider.get_token_counter("claude-3-5-sonnet-20241022")
        count = counter.count_text("prefix <|fim_suffix|> suffix")
        assert count > 0


class TestAnthropicModelLimits:
    @pytest.fixture
    def anthropic_provider(self):
        from headroom.providers.anthropic import AnthropicProvider

        return AnthropicProvider()

    def test_get_context_limit_claude_sonnet(self, anthropic_provider):
        limit = anthropic_provider.get_context_limit("claude-3-5-sonnet-20241022")
        assert limit == 200000

    def test_get_context_limit_claude_opus(self, anthropic_provider):
        limit = anthropic_provider.get_context_limit("claude-3-opus-20240229")
        assert limit == 200000

    def test_get_context_limit_strips_ansi_model_suffix(self, anthropic_provider):
        assert anthropic_provider.get_context_limit("claude-opus-4-7[1m]") == 1000000

    def test_get_context_limit_claude_5_family(self, anthropic_provider):
        assert anthropic_provider.get_context_limit("claude-fable-5") == 1000000
        assert anthropic_provider.get_context_limit("claude-opus-4-8") == 1000000
        assert anthropic_provider.get_context_limit("claude-sonnet-5") == 1000000

    def test_supports_model_known(self, anthropic_provider):
        assert anthropic_provider.supports_model("claude-3-5-sonnet-20241022")

    def test_supports_model_prefix(self, anthropic_provider):
        assert anthropic_provider.supports_model("claude-3-5-sonnet-latest")

    def test_token_counter_cache_uses_sanitized_model_id(self, anthropic_provider):
        plain = anthropic_provider.get_token_counter("claude-opus-4-7")
        styled = anthropic_provider.get_token_counter("claude-opus-4-7\x1b[1m")

        assert styled is plain


class TestAnthropicCostEstimation:
    @pytest.fixture
    def anthropic_provider(self):
        from headroom.providers.anthropic import AnthropicProvider

        return AnthropicProvider()

    def test_estimate_cost_basic(self, anthropic_provider):
        cost = anthropic_provider.estimate_cost(
            input_tokens=1000000,
            output_tokens=0,
            model="claude-3-5-sonnet-20241022",
        )
        # $3.00 per 1M input
        assert cost == pytest.approx(3.00, rel=0.1)

    def test_pricing_lookup_strips_ansi_model_suffix(self, anthropic_provider):
        assert anthropic_provider._get_pricing("claude-opus-4-7[1m]") == (
            anthropic_provider._get_pricing("claude-opus-4-7")
        )

    def test_pricing_claude_5_family(self, anthropic_provider):
        fable = anthropic_provider._get_pricing("claude-fable-5")
        assert fable == {"input": 10.00, "output": 50.00, "cached_input": 1.00}

        opus = anthropic_provider._get_pricing("claude-opus-4-8")
        assert opus == {"input": 5.00, "output": 25.00, "cached_input": 0.50}

        sonnet = anthropic_provider._get_pricing("claude-sonnet-5")
        assert sonnet == {"input": 3.00, "output": 15.00, "cached_input": 0.30}
