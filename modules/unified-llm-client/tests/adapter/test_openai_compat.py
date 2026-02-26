"""Tests for unified_llm.adapters.openai_compat — OpenAI-Compatible adapter.

Uses Chat Completions API (/v1/chat/completions) for third-party services
(vLLM, Ollama, Together AI, Groq). Distinct from the main OpenAI adapter
which uses the Responses API.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest

import unified_llm.errors as E
from unified_llm.types import (
    ContentKind,
    ContentPart,
    ImageData,
    Message,
    Request,
    Role,
    StreamEvent,
    StreamEventType,
    Tool,
    ToolCallData,
    ToolChoice,
)


def _make_adapter() -> Any:
    """Create adapter with mocked AsyncOpenAI client."""
    from unified_llm.adapters.openai_compat import OpenAICompatAdapter

    with patch("unified_llm.adapters.openai_compat.openai.AsyncOpenAI"):
        return OpenAICompatAdapter(
            api_key="test-key",
            base_url="https://my-vllm.example.com/v1",
        )


# ---------------------------------------------------------------------------
# Construction and Properties
# ---------------------------------------------------------------------------


class TestConstruction:
    """OpenAICompatAdapter construction and basic properties."""

    def test_name_is_openai_compat(self) -> None:
        """name property returns 'openai_compat'."""
        adapter = _make_adapter()
        assert adapter.name == "openai_compat"

    def test_accepts_base_url_and_api_key(self) -> None:
        """Constructor accepts base_url for custom endpoints."""
        from unified_llm.adapters.openai_compat import OpenAICompatAdapter

        with patch("unified_llm.adapters.openai_compat.openai.AsyncOpenAI") as mock_cls:
            OpenAICompatAdapter(
                api_key="my-key",
                base_url="https://ollama.local/v1",
            )
            mock_cls.assert_called_once()
            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["api_key"] == "my-key"
            assert call_kwargs["base_url"] == "https://ollama.local/v1"


# ---------------------------------------------------------------------------
# Request Translation — Chat Completions Format
# ---------------------------------------------------------------------------


class TestRequestTranslation:
    """Verify unified Request → Chat Completions API format."""

    def test_basic_messages(self) -> None:
        """Messages translated to Chat Completions format."""
        adapter = _make_adapter()
        request = Request(
            model="llama-3.1-8b",
            messages=[
                Message.system("You are helpful"),
                Message.user("Hello"),
            ],
        )
        kwargs = adapter._translate_request(request)
        assert kwargs["model"] == "llama-3.1-8b"
        assert kwargs["messages"] == [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]

    def test_assistant_messages(self) -> None:
        """Assistant messages translated correctly."""
        adapter = _make_adapter()
        request = Request(
            model="model",
            messages=[
                Message.user("Hello"),
                Message.assistant("Hi there"),
            ],
        )
        kwargs = adapter._translate_request(request)
        assert kwargs["messages"][1] == {"role": "assistant", "content": "Hi there"}

    def test_developer_role_becomes_system(self) -> None:
        """DEVELOPER role mapped to 'system' in Chat Completions."""
        adapter = _make_adapter()
        request = Request(
            model="model",
            messages=[
                Message(
                    role=Role.DEVELOPER,
                    content=[ContentPart(kind=ContentKind.TEXT, text="Dev msg")],
                ),
                Message.user("Hello"),
            ],
        )
        kwargs = adapter._translate_request(request)
        assert kwargs["messages"][0]["role"] == "system"
        assert kwargs["messages"][0]["content"] == "Dev msg"

    def test_image_url_content(self) -> None:
        """Image URL content translated to multipart format."""
        adapter = _make_adapter()
        request = Request(
            model="model",
            messages=[
                Message(
                    role=Role.USER,
                    content=[
                        ContentPart(kind=ContentKind.TEXT, text="What is this?"),
                        ContentPart(
                            kind=ContentKind.IMAGE,
                            image=ImageData(url="https://example.com/img.png"),
                        ),
                    ],
                ),
            ],
        )
        kwargs = adapter._translate_request(request)
        msg = kwargs["messages"][0]
        assert msg["role"] == "user"
        assert isinstance(msg["content"], list)
        assert msg["content"][0] == {"type": "text", "text": "What is this?"}
        assert msg["content"][1]["type"] == "image_url"
        assert msg["content"][1]["image_url"]["url"] == "https://example.com/img.png"

    def test_image_base64_content(self) -> None:
        """Image base64 content translated correctly."""
        adapter = _make_adapter()
        request = Request(
            model="model",
            messages=[
                Message(
                    role=Role.USER,
                    content=[
                        ContentPart(
                            kind=ContentKind.IMAGE,
                            image=ImageData(
                                data=b"\x89PNG",
                                media_type="image/png",
                            ),
                        ),
                    ],
                ),
            ],
        )
        kwargs = adapter._translate_request(request)
        msg = kwargs["messages"][0]
        assert isinstance(msg["content"], list)
        img_part = msg["content"][0]
        assert img_part["type"] == "image_url"
        assert img_part["image_url"]["url"].startswith("data:image/png;base64,")

    def test_generation_params(self) -> None:
        """Temperature, top_p, max_tokens, stop mapped correctly."""
        adapter = _make_adapter()
        request = Request(
            model="model",
            messages=[Message.user("Hi")],
            temperature=0.5,
            top_p=0.9,
            max_tokens=200,
            stop_sequences=["END"],
        )
        kwargs = adapter._translate_request(request)
        assert kwargs["temperature"] == 0.5
        assert kwargs["top_p"] == 0.9
        assert kwargs["max_tokens"] == 200
        assert kwargs["stop"] == ["END"]

    def test_tools_translation(self) -> None:
        """Tools translated to Chat Completions format."""
        adapter = _make_adapter()
        request = Request(
            model="model",
            messages=[Message.user("Hi")],
            tools=[
                Tool(
                    name="get_weather",
                    description="Get weather",
                    parameters={"type": "object", "properties": {}},
                ),
            ],
        )
        kwargs = adapter._translate_request(request)
        assert len(kwargs["tools"]) == 1
        assert kwargs["tools"][0] == {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {}},
            },
        }

    def test_tool_choice_translation(self) -> None:
        """ToolChoice modes mapped correctly."""
        adapter = _make_adapter()
        for mode, expected in [
            ("auto", "auto"),
            ("none", "none"),
            ("required", "required"),
        ]:
            request = Request(
                model="model",
                messages=[Message.user("Hi")],
                tool_choice=ToolChoice(mode=mode),
            )
            kwargs = adapter._translate_request(request)
            assert kwargs["tool_choice"] == expected

    def test_tool_choice_named(self) -> None:
        """Named tool choice maps to Chat Completions format."""
        adapter = _make_adapter()
        request = Request(
            model="model",
            messages=[Message.user("Hi")],
            tool_choice=ToolChoice(mode="named", tool_name="my_func"),
        )
        kwargs = adapter._translate_request(request)
        assert kwargs["tool_choice"] == {
            "type": "function",
            "function": {"name": "my_func"},
        }

    def test_response_format_json_schema(self) -> None:
        """Response format for structured output translated correctly."""
        from unified_llm.types import ResponseFormat

        adapter = _make_adapter()
        request = Request(
            model="model",
            messages=[Message.user("Hi")],
            response_format=ResponseFormat(
                type="json_schema",
                json_schema={"type": "object", "properties": {"x": {"type": "int"}}},
            ),
        )
        kwargs = adapter._translate_request(request)
        assert kwargs["response_format"]["type"] == "json_schema"

    def test_provider_options_passthrough(self) -> None:
        """Provider options for 'openai_compat' pass through."""
        adapter = _make_adapter()
        request = Request(
            model="model",
            messages=[Message.user("Hi")],
            provider_options={"openai_compat": {"seed": 42, "logprobs": True}},
        )
        kwargs = adapter._translate_request(request)
        assert kwargs["seed"] == 42
        assert kwargs["logprobs"] is True

    def test_tool_result_messages(self) -> None:
        """Tool result messages translated to Chat Completions format."""
        adapter = _make_adapter()
        request = Request(
            model="model",
            messages=[
                Message.user("Weather?"),
                Message(
                    role=Role.ASSISTANT,
                    content=[
                        ContentPart(
                            kind=ContentKind.TOOL_CALL,
                            tool_call=ToolCallData(
                                id="call_1",
                                name="get_weather",
                                arguments={"city": "SF"},
                            ),
                        ),
                    ],
                ),
                Message.tool_result(
                    tool_call_id="call_1",
                    content="72F sunny",
                ),
            ],
        )
        kwargs = adapter._translate_request(request)
        # Assistant message should have tool_calls
        asst_msg = kwargs["messages"][1]
        assert asst_msg["role"] == "assistant"
        assert len(asst_msg["tool_calls"]) == 1
        assert asst_msg["tool_calls"][0]["id"] == "call_1"
        # Tool result message
        tool_msg = kwargs["messages"][2]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "call_1"
        assert tool_msg["content"] == "72F sunny"


# ---------------------------------------------------------------------------
# Response Translation
# ---------------------------------------------------------------------------


class TestResponseTranslation:
    """Verify Chat Completions response → unified Response."""

    def test_basic_text_response(self) -> None:
        """Basic text response correctly translated."""
        adapter = _make_adapter()
        raw = SimpleNamespace(
            id="chatcmpl-123",
            model="llama-3.1-8b",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content="Hello!",
                        tool_calls=None,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
        )
        response = adapter._translate_response(raw)
        assert response.id == "chatcmpl-123"
        assert response.model == "llama-3.1-8b"
        assert response.provider == "openai_compat"
        assert response.text == "Hello!"
        assert response.finish_reason.reason == "stop"
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 5
        assert response.usage.total_tokens == 15

    def test_tool_call_response(self) -> None:
        """Tool calls in response correctly translated."""
        adapter = _make_adapter()
        raw = SimpleNamespace(
            id="chatcmpl-456",
            model="model",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                id="call_abc",
                                type="function",
                                function=SimpleNamespace(
                                    name="get_weather",
                                    arguments='{"city": "SF"}',
                                ),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=20,
                completion_tokens=10,
                total_tokens=30,
            ),
        )
        response = adapter._translate_response(raw)
        assert response.finish_reason.reason == "tool_calls"
        assert len(response.tool_calls) == 1
        tc = response.tool_calls[0]
        assert tc.id == "call_abc"
        assert tc.name == "get_weather"
        assert tc.arguments == {"city": "SF"}

    def test_finish_reason_mapping(self) -> None:
        """Chat Completions finish reasons map to unified values."""
        adapter = _make_adapter()
        for raw_reason, expected in [
            ("stop", "stop"),
            ("length", "length"),
            ("tool_calls", "tool_calls"),
            ("content_filter", "content_filter"),
        ]:
            raw = SimpleNamespace(
                id="id",
                model="model",
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            role="assistant",
                            content="text",
                            tool_calls=None,
                        ),
                        finish_reason=raw_reason,
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=0, completion_tokens=0, total_tokens=0
                ),
            )
            response = adapter._translate_response(raw)
            assert response.finish_reason.reason == expected


# ---------------------------------------------------------------------------
# complete() Integration
# ---------------------------------------------------------------------------


class TestComplete:
    """complete() sends request and returns Response."""

    @pytest.mark.asyncio(loop_scope="function")
    async def test_complete_basic(self) -> None:
        """complete() calls chat.completions.create and returns Response."""
        adapter = _make_adapter()
        mock_raw = SimpleNamespace(
            id="chatcmpl-789",
            model="llama-3.1-8b",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content="Hello from vLLM!",
                        tool_calls=None,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=5,
                completion_tokens=3,
                total_tokens=8,
            ),
        )
        adapter._client.chat.completions.create = AsyncMock(return_value=mock_raw)

        request = Request(model="llama-3.1-8b", messages=[Message.user("Hello")])
        response = await adapter.complete(request)

        assert response.text == "Hello from vLLM!"
        assert response.provider == "openai_compat"
        adapter._client.chat.completions.create.assert_called_once()


# ---------------------------------------------------------------------------
# Error Translation
# ---------------------------------------------------------------------------


class TestErrorTranslation:
    """Verify OpenAI SDK errors → unified error hierarchy."""

    @pytest.mark.asyncio(loop_scope="function")
    async def test_auth_error_401(self) -> None:
        """401 → AuthenticationError."""
        adapter = _make_adapter()
        api_error = openai.AuthenticationError(
            message="Invalid API key",
            response=MagicMock(status_code=401, headers={}),
            body={"error": {"message": "Invalid API key"}},
        )
        adapter._client.chat.completions.create = AsyncMock(side_effect=api_error)
        request = Request(model="model", messages=[Message.user("Hi")])

        with pytest.raises(E.AuthenticationError):
            await adapter.complete(request)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_rate_limit_429(self) -> None:
        """429 → RateLimitError."""
        adapter = _make_adapter()
        api_error = openai.RateLimitError(
            message="Rate limited",
            response=MagicMock(status_code=429, headers={}),
            body={"error": {"message": "Rate limited"}},
        )
        adapter._client.chat.completions.create = AsyncMock(side_effect=api_error)
        request = Request(model="model", messages=[Message.user("Hi")])

        with pytest.raises(E.RateLimitError):
            await adapter.complete(request)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_not_found_404(self) -> None:
        """404 → NotFoundError."""
        adapter = _make_adapter()
        api_error = openai.NotFoundError(
            message="Model not found",
            response=MagicMock(status_code=404, headers={}),
            body={"error": {"message": "Model not found"}},
        )
        adapter._client.chat.completions.create = AsyncMock(side_effect=api_error)
        request = Request(model="nonexistent", messages=[Message.user("Hi")])

        with pytest.raises(E.NotFoundError):
            await adapter.complete(request)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_timeout_error(self) -> None:
        """Timeout → RequestTimeoutError."""
        adapter = _make_adapter()
        timeout_error = openai.APITimeoutError(request=MagicMock())
        adapter._client.chat.completions.create = AsyncMock(side_effect=timeout_error)
        request = Request(model="model", messages=[Message.user("Hi")])

        with pytest.raises(E.RequestTimeoutError):
            await adapter.complete(request)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_connection_error(self) -> None:
        """Connection error → NetworkError."""
        adapter = _make_adapter()
        conn_error = openai.APIConnectionError(request=MagicMock())
        adapter._client.chat.completions.create = AsyncMock(side_effect=conn_error)
        request = Request(model="model", messages=[Message.user("Hi")])

        with pytest.raises(E.NetworkError):
            await adapter.complete(request)


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class TestStreaming:
    """Verify Chat Completions streaming → unified StreamEvent."""

    @pytest.mark.asyncio(loop_scope="function")
    async def test_stream_text_events(self) -> None:
        """Streaming text deltas produce correct events."""
        adapter = _make_adapter()

        # Simulate Chat Completions streaming chunks
        chunks = [
            SimpleNamespace(
                id="chatcmpl-stream",
                model="llama-3.1-8b",
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            role="assistant", content=None, tool_calls=None
                        ),
                        finish_reason=None,
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(
                id="chatcmpl-stream",
                model="llama-3.1-8b",
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            role=None, content="Hello", tool_calls=None
                        ),
                        finish_reason=None,
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(
                id="chatcmpl-stream",
                model="llama-3.1-8b",
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            role=None, content=" world", tool_calls=None
                        ),
                        finish_reason=None,
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(
                id="chatcmpl-stream",
                model="llama-3.1-8b",
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(role=None, content=None, tool_calls=None),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=5, completion_tokens=2, total_tokens=7
                ),
            ),
        ]

        class MockAsyncStream:
            """Simulates openai.AsyncStream (awaitable + async iterable)."""

            def __init__(self, items: list[Any]) -> None:
                self._items = items

            def __aiter__(self) -> MockAsyncStream:
                self._index = 0
                return self

            async def __anext__(self) -> Any:
                if self._index >= len(self._items):
                    raise StopAsyncIteration
                item = self._items[self._index]
                self._index += 1
                return item

        adapter._client.chat.completions.create = AsyncMock(
            return_value=MockAsyncStream(chunks)
        )

        request = Request(model="llama-3.1-8b", messages=[Message.user("Hello")])
        events: list[StreamEvent] = []
        async for event in adapter.stream(request):
            events.append(event)

        # Should have STREAM_START, TEXT_START, TEXT_DELTA(s), TEXT_END, FINISH
        event_types = [e.type for e in events]
        assert StreamEventType.STREAM_START in event_types
        assert StreamEventType.TEXT_DELTA in event_types
        assert StreamEventType.FINISH in event_types

        # Verify text deltas
        deltas = [e.delta for e in events if e.type == StreamEventType.TEXT_DELTA]
        assert "".join(d for d in deltas if d) == "Hello world"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_stream_tool_call_events(self) -> None:
        """Streaming tool calls produce correct events."""
        adapter = _make_adapter()

        chunks = [
            # First chunk: role
            SimpleNamespace(
                id="chatcmpl-tc",
                model="model",
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            role="assistant",
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id="call_1",
                                    type="function",
                                    function=SimpleNamespace(
                                        name="get_weather",
                                        arguments="",
                                    ),
                                )
                            ],
                        ),
                        finish_reason=None,
                    )
                ],
                usage=None,
            ),
            # Second chunk: arguments delta
            SimpleNamespace(
                id="chatcmpl-tc",
                model="model",
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            role=None,
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id=None,
                                    type=None,
                                    function=SimpleNamespace(
                                        name=None,
                                        arguments='{"city":',
                                    ),
                                )
                            ],
                        ),
                        finish_reason=None,
                    )
                ],
                usage=None,
            ),
            # Third chunk: more arguments
            SimpleNamespace(
                id="chatcmpl-tc",
                model="model",
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            role=None,
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id=None,
                                    type=None,
                                    function=SimpleNamespace(
                                        name=None,
                                        arguments='"SF"}',
                                    ),
                                )
                            ],
                        ),
                        finish_reason=None,
                    )
                ],
                usage=None,
            ),
            # Final: finish
            SimpleNamespace(
                id="chatcmpl-tc",
                model="model",
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(role=None, content=None, tool_calls=None),
                        finish_reason="tool_calls",
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=10, completion_tokens=15, total_tokens=25
                ),
            ),
        ]

        class MockAsyncStream:
            """Simulates openai.AsyncStream (awaitable + async iterable)."""

            def __init__(self, items: list[Any]) -> None:
                self._items = items

            def __aiter__(self) -> MockAsyncStream:
                self._index = 0
                return self

            async def __anext__(self) -> Any:
                if self._index >= len(self._items):
                    raise StopAsyncIteration
                item = self._items[self._index]
                self._index += 1
                return item

        adapter._client.chat.completions.create = AsyncMock(
            return_value=MockAsyncStream(chunks)
        )

        request = Request(model="model", messages=[Message.user("Weather?")])
        events: list[StreamEvent] = []
        async for event in adapter.stream(request):
            events.append(event)

        event_types = [e.type for e in events]
        assert StreamEventType.TOOL_CALL_START in event_types
        assert StreamEventType.TOOL_CALL_END in event_types
        assert StreamEventType.FINISH in event_types

        # Verify tool call end has parsed arguments
        end_events = [e for e in events if e.type == StreamEventType.TOOL_CALL_END]
        assert len(end_events) == 1
        assert end_events[0].tool_call is not None
        assert end_events[0].tool_call.name == "get_weather"
        assert end_events[0].tool_call.arguments == {"city": "SF"}


# ---------------------------------------------------------------------------
# No Reasoning Tokens (Chat Completions limitation)
# ---------------------------------------------------------------------------


class TestNoReasoningTokens:
    """OpenAI-compatible adapter does NOT support reasoning tokens."""

    def test_reasoning_effort_ignored(self) -> None:
        """reasoning_effort is NOT passed through (Chat Completions limitation)."""
        adapter = _make_adapter()
        request = Request(
            model="model",
            messages=[Message.user("Hi")],
            reasoning_effort="high",
        )
        kwargs = adapter._translate_request(request)
        assert "reasoning" not in kwargs
        assert "reasoning_effort" not in kwargs


# ---------------------------------------------------------------------------
# Supports tool_choice
# ---------------------------------------------------------------------------


class TestSupportsToolChoice:
    """supports_tool_choice method works."""

    def test_supports_standard_modes(self) -> None:
        adapter = _make_adapter()
        assert adapter.supports_tool_choice("auto") is True
        assert adapter.supports_tool_choice("none") is True
        assert adapter.supports_tool_choice("required") is True
        assert adapter.supports_tool_choice("named") is True
