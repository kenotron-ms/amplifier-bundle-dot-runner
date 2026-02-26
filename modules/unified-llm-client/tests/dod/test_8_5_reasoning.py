"""DoD §8.5 — Reasoning Tokens.

Verifies reasoning/thinking token support across providers using mocks.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch


from unified_llm import (
    ContentKind,
    Message,
    Request,
    Usage,
)
from unified_llm.adapters.openai import OpenAIAdapter
from unified_llm.adapters.anthropic import AnthropicAdapter
from unified_llm.adapters.gemini import GeminiAdapter


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
# §8.5 — OpenAI reasoning tokens
# ---------------------------------------------------------------------------


class TestOpenAIReasoningTokens:
    """OpenAI reasoning models return reasoning_tokens in Usage."""

    def test_reasoning_tokens_in_usage(self) -> None:
        """[ ] OpenAI reasoning models return reasoning_tokens in Usage via Responses API."""
        adapter = _openai_adapter()
        raw = SimpleNamespace(
            id="resp_1",
            model="o3",
            status="completed",
            output=[
                SimpleNamespace(
                    type="message",
                    role="assistant",
                    content=[SimpleNamespace(type="output_text", text="42")],
                )
            ],
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=500,
                total_tokens=600,
                output_tokens_details=SimpleNamespace(reasoning_tokens=400),
                input_tokens_details=None,
            ),
        )
        response = adapter._translate_response(raw)
        assert response.usage.reasoning_tokens == 400
        assert response.usage.output_tokens == 500

    def test_reasoning_effort_passed_through(self) -> None:
        """[ ] reasoning_effort parameter is passed through correctly."""
        adapter = _openai_adapter()
        request = Request(
            model="o3",
            messages=[Message.user("Think hard")],
            reasoning_effort="high",
        )
        kwargs = adapter._translate_request(request)
        assert kwargs["reasoning"] == {"effort": "high"}


# ---------------------------------------------------------------------------
# §8.5 — Anthropic thinking blocks
# ---------------------------------------------------------------------------


class TestAnthropicThinkingBlocks:
    """Anthropic extended thinking blocks returned as THINKING content parts."""

    def test_thinking_blocks_returned(self) -> None:
        """[ ] Anthropic thinking blocks returned as THINKING content parts."""
        adapter = _anthropic_adapter()
        raw = SimpleNamespace(
            id="msg_1",
            model="claude-sonnet-4-20250514",
            stop_reason="end_turn",
            content=[
                SimpleNamespace(
                    type="thinking",
                    thinking="Let me think...",
                    signature="sig_abc",
                ),
                SimpleNamespace(type="text", text="The answer is 42."),
            ],
            usage=SimpleNamespace(input_tokens=50, output_tokens=30),
        )
        response = adapter._translate_response(raw)
        # Should have thinking + text content parts
        thinking_parts = [
            p for p in response.message.content if p.kind == ContentKind.THINKING
        ]
        text_parts = [p for p in response.message.content if p.kind == ContentKind.TEXT]
        assert len(thinking_parts) == 1
        assert thinking_parts[0].thinking is not None
        assert thinking_parts[0].thinking.text == "Let me think..."
        assert len(text_parts) == 1
        assert text_parts[0].text == "The answer is 42."

    def test_thinking_signature_preserved(self) -> None:
        """[ ] Thinking block signature field is preserved for round-tripping."""
        adapter = _anthropic_adapter()
        raw = SimpleNamespace(
            id="msg_1",
            model="claude-sonnet-4-20250514",
            stop_reason="end_turn",
            content=[
                SimpleNamespace(
                    type="thinking",
                    thinking="reasoning...",
                    signature="sig_xyz789",
                ),
                SimpleNamespace(type="text", text="Answer"),
            ],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )
        response = adapter._translate_response(raw)
        thinking_parts = [
            p for p in response.message.content if p.kind == ContentKind.THINKING
        ]
        assert len(thinking_parts) == 1
        assert thinking_parts[0].thinking is not None
        assert thinking_parts[0].thinking.signature == "sig_xyz789"


# ---------------------------------------------------------------------------
# §8.5 — Gemini thinking tokens
# ---------------------------------------------------------------------------


class TestGeminiThinkingTokens:
    """Gemini thinking tokens mapped to reasoning_tokens in Usage."""

    def test_thoughts_token_count_mapped(self) -> None:
        """[ ] Gemini thinking tokens (thoughtsTokenCount) mapped to reasoning_tokens."""
        adapter = _gemini_adapter()
        mock_candidate = SimpleNamespace(
            content=SimpleNamespace(
                parts=[SimpleNamespace(text="42", function_call=None, thought=None)]
            ),
            finish_reason="STOP",
        )
        mock_raw = SimpleNamespace(
            candidates=[mock_candidate],
            usage_metadata=SimpleNamespace(
                prompt_token_count=50,
                candidates_token_count=10,
                total_token_count=60,
                thoughts_token_count=200,
                cached_content_token_count=None,
            ),
        )
        response = adapter._translate_response(mock_raw, model="gemini-2.5-flash")
        assert response.usage.reasoning_tokens == 200


# ---------------------------------------------------------------------------
# §8.5 — Usage correctly distinguishes reasoning from output
# ---------------------------------------------------------------------------


class TestUsageDistinction:
    """Usage correctly reports reasoning_tokens as distinct from output_tokens."""

    def test_reasoning_tokens_distinct(self) -> None:
        """[ ] Usage correctly reports reasoning_tokens as distinct from output_tokens."""
        usage = Usage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=550,
            reasoning_tokens=400,
        )
        assert usage.reasoning_tokens == 400
        assert usage.output_tokens == 50
        # They are separate fields
        assert usage.reasoning_tokens != usage.output_tokens
