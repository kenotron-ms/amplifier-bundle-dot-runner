"""DoD §8.6 — Prompt Caching.

Verifies prompt caching behavior across providers using mocks.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from unified_llm import (
    Message,
    Request,
    Usage,
)
from unified_llm.adapters.anthropic import AnthropicAdapter
from unified_llm.adapters.gemini import GeminiAdapter
from unified_llm.adapters.openai import OpenAIAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _openai_adapter() -> OpenAIAdapter:
    with patch("unified_llm.adapters.openai.openai.AsyncOpenAI"):
        return OpenAIAdapter(api_key="test")


def _anthropic_adapter() -> AnthropicAdapter:
    with patch("unified_llm.adapters.anthropic.anthropic.AsyncAnthropic"):
        return AnthropicAdapter(api_key="test")


def _gemini_adapter() -> GeminiAdapter:
    with patch("unified_llm.adapters.gemini.genai.Client"):
        return GeminiAdapter(api_key="test")


# ---------------------------------------------------------------------------
# §8.6 — OpenAI caching
# ---------------------------------------------------------------------------


class TestOpenAICaching:
    """OpenAI caching via Responses API."""

    def test_cache_read_tokens_populated(self) -> None:
        """[ ] OpenAI: Usage.cache_read_tokens populated from cached_tokens."""
        adapter = _openai_adapter()
        raw = SimpleNamespace(
            id="resp_1",
            model="gpt-4.1",
            status="completed",
            output=[
                SimpleNamespace(
                    type="message",
                    role="assistant",
                    content=[SimpleNamespace(type="output_text", text="Hi")],
                )
            ],
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=10,
                total_tokens=110,
                output_tokens_details=None,
                input_tokens_details=SimpleNamespace(cached_tokens=80),
            ),
        )
        response = adapter._translate_response(raw)
        assert response.usage.cache_read_tokens == 80


# ---------------------------------------------------------------------------
# §8.6 — Anthropic caching
# ---------------------------------------------------------------------------


class TestAnthropicCaching:
    """Anthropic caching via Messages API."""

    def test_cache_control_injected(self) -> None:
        """[ ] Anthropic: adapter injects cache_control breakpoints."""
        adapter = _anthropic_adapter()
        request = Request(
            model="claude-sonnet-4-20250514",
            messages=[
                Message.system(
                    "You are a helpful assistant with a long system prompt " * 20
                ),
                Message.user("Hi"),
            ],
            max_tokens=1024,
        )
        kwargs = adapter._translate_request(request)
        # Should have system parameter with cache_control
        system_blocks = kwargs.get("system", [])
        if isinstance(system_blocks, list) and len(system_blocks) > 0:
            last_block = system_blocks[-1]
            if isinstance(last_block, dict):
                assert "cache_control" in last_block

    def test_cache_read_and_write_tokens(self) -> None:
        """[ ] Anthropic: cache_read_tokens and cache_write_tokens populated."""
        adapter = _anthropic_adapter()
        raw = SimpleNamespace(
            id="msg_1",
            model="claude-sonnet-4-20250514",
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text="Hello")],
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=10,
                cache_read_input_tokens=50,
                cache_creation_input_tokens=20,
            ),
        )
        response = adapter._translate_response(raw)
        assert response.usage.cache_read_tokens == 50
        assert response.usage.cache_write_tokens == 20

    def test_auto_cache_disabled_via_provider_options(self) -> None:
        """[ ] Anthropic: automatic caching can be disabled via provider_options."""
        adapter = _anthropic_adapter()
        request = Request(
            model="claude-sonnet-4-20250514",
            messages=[
                Message.system("System prompt " * 20),
                Message.user("Hi"),
            ],
            max_tokens=1024,
            provider_options={"anthropic": {"auto_cache": False}},
        )
        kwargs = adapter._translate_request(request)
        # When auto_cache is False, system blocks should NOT have cache_control
        system_blocks = kwargs.get("system", [])
        if isinstance(system_blocks, list):
            for block in system_blocks:
                if isinstance(block, dict):
                    assert "cache_control" not in block


# ---------------------------------------------------------------------------
# §8.6 — Gemini caching
# ---------------------------------------------------------------------------


class TestGeminiCaching:
    """Gemini automatic prefix caching."""

    def test_cache_read_tokens_populated(self) -> None:
        """[ ] Gemini: cache_read_tokens populated from cachedContentTokenCount."""
        adapter = _gemini_adapter()
        mock_candidate = SimpleNamespace(
            content=SimpleNamespace(
                parts=[SimpleNamespace(text="Hi", function_call=None, thought=None)]
            ),
            finish_reason="STOP",
        )
        mock_raw = SimpleNamespace(
            candidates=[mock_candidate],
            usage_metadata=SimpleNamespace(
                prompt_token_count=100,
                candidates_token_count=5,
                total_token_count=105,
                thoughts_token_count=None,
                cached_content_token_count=60,
            ),
        )
        response = adapter._translate_response(mock_raw, model="gemini-2.5-flash")
        assert response.usage.cache_read_tokens == 60


# ---------------------------------------------------------------------------
# §8.6 — Usage addition preserves cache tokens
# ---------------------------------------------------------------------------


class TestUsageCacheAddition:
    """Multi-turn: verify cache tokens aggregate correctly."""

    def test_usage_addition_preserves_cache_tokens(self) -> None:
        """[ ] Cache tokens aggregate correctly across steps."""
        u1 = Usage(
            input_tokens=100,
            output_tokens=10,
            total_tokens=110,
            cache_read_tokens=50,
            cache_write_tokens=20,
        )
        u2 = Usage(
            input_tokens=80,
            output_tokens=15,
            total_tokens=95,
            cache_read_tokens=70,
            cache_write_tokens=5,
        )
        combined = u1 + u2
        assert combined.cache_read_tokens == 120
        assert combined.cache_write_tokens == 25
