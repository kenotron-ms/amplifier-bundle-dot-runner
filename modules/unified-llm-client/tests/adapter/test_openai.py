"""Tests for unified_llm.adapters.openai — OpenAI Responses API adapter."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest

import unified_llm.errors as E
from unified_llm.adapters.openai import OpenAIAdapter
from unified_llm.types import (
    ContentKind,
    ContentPart,
    ImageData,
    Message,
    Request,
    Role,
    Tool,
    ToolCallData,
    ToolChoice,
)


def _make_adapter() -> OpenAIAdapter:
    """Create adapter with mocked AsyncOpenAI client."""
    with patch("unified_llm.adapters.openai.openai.AsyncOpenAI"):
        return OpenAIAdapter(api_key="test-key")


# ---------------------------------------------------------------------------
# Task 28: OpenAI Request Translation (Responses API)
# ---------------------------------------------------------------------------


class TestRequestTranslation:
    """Task 28: Verify unified Request → OpenAI Responses API format."""

    def test_system_message_to_instructions(self) -> None:
        """System messages extracted to 'instructions' parameter."""
        adapter = _make_adapter()
        request = Request(
            model="gpt-4.1",
            messages=[
                Message.system("You are helpful"),
                Message.user("Hello"),
            ],
        )
        kwargs = adapter._translate_request(request)
        assert kwargs["instructions"] == "You are helpful"
        # System message should NOT appear in the input array
        input_items = kwargs["input"]
        roles = [
            item.get("role")
            for item in input_items
            if isinstance(item, dict) and "role" in item
        ]
        assert "system" not in roles

    def test_developer_role_to_instructions(self) -> None:
        """DEVELOPER role messages merge into instructions parameter."""
        adapter = _make_adapter()
        request = Request(
            model="gpt-4.1",
            messages=[
                Message.system("System instructions"),
                Message(
                    role=Role.DEVELOPER,
                    content=[
                        ContentPart(kind=ContentKind.TEXT, text="Dev instructions")
                    ],
                ),
                Message.user("Hello"),
            ],
        )
        kwargs = adapter._translate_request(request)
        assert "System instructions" in kwargs["instructions"]
        assert "Dev instructions" in kwargs["instructions"]

    def test_user_message_translation(self) -> None:
        """User messages → input items with type='message', role='user'."""
        adapter = _make_adapter()
        request = Request(
            model="gpt-4.1",
            messages=[Message.user("Hello world")],
        )
        kwargs = adapter._translate_request(request)
        msg = kwargs["input"][0]
        assert msg["type"] == "message"
        assert msg["role"] == "user"
        assert msg["content"][0]["type"] == "input_text"
        assert msg["content"][0]["text"] == "Hello world"

    def test_assistant_message_translation(self) -> None:
        """Assistant messages → input items with type='message', role='assistant'."""
        adapter = _make_adapter()
        request = Request(
            model="gpt-4.1",
            messages=[
                Message.user("Hi"),
                Message.assistant("Hello!"),
            ],
        )
        kwargs = adapter._translate_request(request)
        assistant_msg = kwargs["input"][1]
        assert assistant_msg["type"] == "message"
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg["content"][0]["type"] == "output_text"
        assert assistant_msg["content"][0]["text"] == "Hello!"

    def test_tool_call_as_function_call_input(self) -> None:
        """TOOL_CALL parts → top-level function_call input items."""
        adapter = _make_adapter()
        request = Request(
            model="gpt-4.1",
            messages=[
                Message.user("What's the weather?"),
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
                        )
                    ],
                ),
            ],
        )
        kwargs = adapter._translate_request(request)
        # The function_call should be a top-level input item
        fc_items = [
            item for item in kwargs["input"] if item.get("type") == "function_call"
        ]
        assert len(fc_items) == 1
        assert fc_items[0]["name"] == "get_weather"
        assert fc_items[0]["call_id"] == "call_1"

    def test_tool_result_as_function_call_output(self) -> None:
        """TOOL_RESULT → top-level function_call_output input items."""
        adapter = _make_adapter()
        request = Request(
            model="gpt-4.1",
            messages=[
                Message.user("What's the weather?"),
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
                        )
                    ],
                ),
                Message.tool_result(tool_call_id="call_1", content="72F sunny"),
            ],
        )
        kwargs = adapter._translate_request(request)
        output_items = [
            item
            for item in kwargs["input"]
            if item.get("type") == "function_call_output"
        ]
        assert len(output_items) == 1
        assert output_items[0]["call_id"] == "call_1"
        assert output_items[0]["output"] == "72F sunny"

    def test_tool_definitions_translated(self) -> None:
        """Tool definitions → {type: 'function', name, description, parameters}."""
        adapter = _make_adapter()
        request = Request(
            model="gpt-4.1",
            messages=[Message.user("Hi")],
            tools=[
                Tool(
                    name="get_weather",
                    description="Get the weather",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                ),
            ],
        )
        kwargs = adapter._translate_request(request)
        assert len(kwargs["tools"]) == 1
        tool = kwargs["tools"][0]
        assert tool["type"] == "function"
        assert tool["name"] == "get_weather"
        assert tool["description"] == "Get the weather"
        assert tool["parameters"]["type"] == "object"

    def test_tool_choice_auto(self) -> None:
        adapter = _make_adapter()
        request = Request(
            model="gpt-4.1",
            messages=[Message.user("Hi")],
            tools=[Tool(name="t", description="d", parameters={})],
            tool_choice=ToolChoice(mode="auto"),
        )
        kwargs = adapter._translate_request(request)
        assert kwargs["tool_choice"] == "auto"

    def test_tool_choice_none(self) -> None:
        adapter = _make_adapter()
        request = Request(
            model="gpt-4.1",
            messages=[Message.user("Hi")],
            tools=[Tool(name="t", description="d", parameters={})],
            tool_choice=ToolChoice(mode="none"),
        )
        kwargs = adapter._translate_request(request)
        assert kwargs["tool_choice"] == "none"

    def test_tool_choice_required(self) -> None:
        adapter = _make_adapter()
        request = Request(
            model="gpt-4.1",
            messages=[Message.user("Hi")],
            tools=[Tool(name="t", description="d", parameters={})],
            tool_choice=ToolChoice(mode="required"),
        )
        kwargs = adapter._translate_request(request)
        assert kwargs["tool_choice"] == "required"

    def test_tool_choice_named(self) -> None:
        adapter = _make_adapter()
        request = Request(
            model="gpt-4.1",
            messages=[Message.user("Hi")],
            tools=[Tool(name="get_weather", description="d", parameters={})],
            tool_choice=ToolChoice(mode="named", tool_name="get_weather"),
        )
        kwargs = adapter._translate_request(request)
        assert kwargs["tool_choice"] == {
            "type": "function",
            "name": "get_weather",
        }

    def test_generation_params_passed(self) -> None:
        """temperature, top_p, stop_sequences, max_tokens forwarded."""
        adapter = _make_adapter()
        request = Request(
            model="gpt-4.1",
            messages=[Message.user("Hi")],
            temperature=0.7,
            top_p=0.9,
            max_tokens=1024,
            stop_sequences=["END"],
        )
        kwargs = adapter._translate_request(request)
        assert kwargs["temperature"] == 0.7
        assert kwargs["top_p"] == 0.9
        assert kwargs["max_output_tokens"] == 1024
        assert kwargs["stop"] == ["END"]

    def test_provider_options_passthrough(self) -> None:
        """provider_options['openai'] passes through extra parameters."""
        adapter = _make_adapter()
        request = Request(
            model="gpt-4.1",
            messages=[Message.user("Hi")],
            provider_options={
                "openai": {
                    "store": True,
                    "metadata": {"session": "abc"},
                }
            },
        )
        kwargs = adapter._translate_request(request)
        assert kwargs["store"] is True
        assert kwargs["metadata"] == {"session": "abc"}

    def test_image_url_translation(self) -> None:
        """IMAGE with URL → input_image with image_url."""
        adapter = _make_adapter()
        request = Request(
            model="gpt-4.1",
            messages=[
                Message(
                    role=Role.USER,
                    content=[
                        ContentPart(kind=ContentKind.TEXT, text="What's this?"),
                        ContentPart(
                            kind=ContentKind.IMAGE,
                            image=ImageData(url="https://example.com/img.png"),
                        ),
                    ],
                ),
            ],
        )
        kwargs = adapter._translate_request(request)
        content = kwargs["input"][0]["content"]
        assert content[1]["type"] == "input_image"
        assert content[1]["image_url"] == "https://example.com/img.png"

    def test_image_base64_translation(self) -> None:
        """IMAGE with data → input_image with data URI."""
        adapter = _make_adapter()
        request = Request(
            model="gpt-4.1",
            messages=[
                Message(
                    role=Role.USER,
                    content=[
                        ContentPart(
                            kind=ContentKind.IMAGE,
                            image=ImageData(data=b"\x89PNG", media_type="image/png"),
                        ),
                    ],
                ),
            ],
        )
        kwargs = adapter._translate_request(request)
        content = kwargs["input"][0]["content"]
        assert content[0]["type"] == "input_image"
        assert content[0]["image_url"].startswith("data:image/png;base64,")

    def test_reasoning_effort_param(self) -> None:
        """reasoning_effort → reasoning.effort parameter."""
        adapter = _make_adapter()
        request = Request(
            model="o4-mini",
            messages=[Message.user("Solve this")],
            reasoning_effort="high",
        )
        kwargs = adapter._translate_request(request)
        assert kwargs["reasoning"] == {"effort": "high"}

    def test_name_property(self) -> None:
        """Adapter name is 'openai'."""
        adapter = _make_adapter()
        assert adapter.name == "openai"

    def test_model_passed_through(self) -> None:
        """Model string passed as 'model' kwarg."""
        adapter = _make_adapter()
        request = Request(
            model="gpt-4.1",
            messages=[Message.user("Hi")],
        )
        kwargs = adapter._translate_request(request)
        assert kwargs["model"] == "gpt-4.1"

    def test_no_max_tokens_when_unset(self) -> None:
        """max_tokens not sent when not specified (OpenAI doesn't require it)."""
        adapter = _make_adapter()
        request = Request(
            model="gpt-4.1",
            messages=[Message.user("Hi")],
        )
        kwargs = adapter._translate_request(request)
        assert "max_output_tokens" not in kwargs

    # ------------------------------------------------------------------
    # Structured output (Spec §4.5 / capability matrix :987)
    # ------------------------------------------------------------------

    def test_json_schema_response_format_sets_text_format(self) -> None:
        """json_schema response_format → text.format with json_schema type (Responses API).

        Asserts the outgoing request carries json_schema in text.format so the
        OpenAI Responses API enforces the schema server-side.
        """
        from unified_llm.types import ResponseFormat

        adapter = _make_adapter()
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
        }
        request = Request(
            model="gpt-4.1",
            messages=[Message.user("Extract info")],
            response_format=ResponseFormat(
                type="json_schema",
                json_schema=schema,
                strict=True,
            ),
        )
        kwargs = adapter._translate_request(request)

        # Responses API structured output: text.format.type == "json_schema"
        assert "text" in kwargs, "OpenAI Responses API expects 'text' kwarg for format"
        text_fmt = kwargs["text"]["format"]
        assert text_fmt["type"] == "json_schema"
        assert text_fmt["strict"] is True
        # The schema must be present under "schema" key
        assert "schema" in text_fmt
        assert text_fmt["schema"]["type"] == "object"
        assert "name" in text_fmt["schema"]["properties"]


# ---------------------------------------------------------------------------
# Task 29: OpenAI Response Translation
# ---------------------------------------------------------------------------


def _mock_openai_response(
    *,
    id: str = "resp_test123",
    model: str = "gpt-4.1",
    output: list[SimpleNamespace] | None = None,
    status: str = "completed",
    input_tokens: int = 10,
    output_tokens: int = 20,
    total_tokens: int | None = None,
    reasoning_tokens: int | None = None,
    cached_tokens: int | None = None,
) -> SimpleNamespace:
    """Create a mock OpenAI Responses API response object."""
    output_details = None
    if reasoning_tokens is not None:
        output_details = SimpleNamespace(reasoning_tokens=reasoning_tokens)

    input_details = None
    if cached_tokens is not None:
        input_details = SimpleNamespace(cached_tokens=cached_tokens)

    usage_kwargs: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens
        if total_tokens is not None
        else input_tokens + output_tokens,
    }
    if output_details is not None:
        usage_kwargs["output_tokens_details"] = output_details
    else:
        usage_kwargs["output_tokens_details"] = None
    if input_details is not None:
        usage_kwargs["input_tokens_details"] = input_details
    else:
        usage_kwargs["input_tokens_details"] = None

    default_output = [
        SimpleNamespace(
            type="message",
            role="assistant",
            content=[SimpleNamespace(type="output_text", text="Hello!")],
        )
    ]

    return SimpleNamespace(
        id=id,
        model=model,
        status=status,
        output=output if output is not None else default_output,
        usage=SimpleNamespace(**usage_kwargs),
    )


class TestResponseTranslation:
    """Task 29: Verify OpenAI Responses API response → unified Response."""

    def test_text_output_item(self) -> None:
        """Message output items with output_text → TEXT ContentParts."""
        adapter = _make_adapter()
        raw = _mock_openai_response(
            output=[
                SimpleNamespace(
                    type="message",
                    role="assistant",
                    content=[SimpleNamespace(type="output_text", text="Hello world")],
                )
            ],
        )
        response = adapter._translate_response(raw)
        assert len(response.message.content) == 1
        part = response.message.content[0]
        assert part.kind == ContentKind.TEXT
        assert part.text == "Hello world"

    def test_function_call_output_item(self) -> None:
        """function_call output items → TOOL_CALL ContentParts."""
        adapter = _make_adapter()
        raw = _mock_openai_response(
            output=[
                SimpleNamespace(
                    type="function_call",
                    call_id="call_123",
                    name="get_weather",
                    arguments='{"city": "SF"}',
                ),
            ],
            status="completed",
        )
        response = adapter._translate_response(raw)
        part = response.message.content[0]
        assert part.kind == ContentKind.TOOL_CALL
        assert part.tool_call is not None
        assert part.tool_call.id == "call_123"
        assert part.tool_call.name == "get_weather"
        assert part.tool_call.arguments == {"city": "SF"}

    def test_finish_reason_completed(self) -> None:
        """status='completed' → FinishReason(reason='stop')."""
        adapter = _make_adapter()
        raw = _mock_openai_response(status="completed")
        response = adapter._translate_response(raw)
        assert response.finish_reason.reason == "stop"

    def test_finish_reason_incomplete(self) -> None:
        """status='incomplete' → FinishReason(reason='length')."""
        adapter = _make_adapter()
        raw = _mock_openai_response(status="incomplete")
        response = adapter._translate_response(raw)
        assert response.finish_reason.reason == "length"

    def test_finish_reason_tool_calls(self) -> None:
        """Presence of function_call items → FinishReason(reason='tool_calls')."""
        adapter = _make_adapter()
        raw = _mock_openai_response(
            output=[
                SimpleNamespace(
                    type="function_call",
                    call_id="call_1",
                    name="get_weather",
                    arguments='{"city": "SF"}',
                ),
            ],
            status="completed",
        )
        response = adapter._translate_response(raw)
        assert response.finish_reason.reason == "tool_calls"

    def test_usage_extraction(self) -> None:
        """Usage fields mapped correctly."""
        adapter = _make_adapter()
        raw = _mock_openai_response(input_tokens=100, output_tokens=50)
        response = adapter._translate_response(raw)
        assert response.usage.input_tokens == 100
        assert response.usage.output_tokens == 50
        assert response.usage.total_tokens == 150

    def test_usage_reasoning_tokens(self) -> None:
        """reasoning_tokens from output_tokens_details."""
        adapter = _make_adapter()
        raw = _mock_openai_response(
            input_tokens=100, output_tokens=80, reasoning_tokens=60
        )
        response = adapter._translate_response(raw)
        assert response.usage.reasoning_tokens == 60

    def test_usage_cache_read_tokens(self) -> None:
        """cached_tokens from input_tokens_details → cache_read_tokens."""
        adapter = _make_adapter()
        raw = _mock_openai_response(
            input_tokens=100, output_tokens=50, cached_tokens=80
        )
        response = adapter._translate_response(raw)
        assert response.usage.cache_read_tokens == 80

    def test_response_metadata(self) -> None:
        """Response id, model, provider populated correctly."""
        adapter = _make_adapter()
        raw = _mock_openai_response(id="resp_abc", model="gpt-4.1")
        response = adapter._translate_response(raw)
        assert response.id == "resp_abc"
        assert response.model == "gpt-4.1"
        assert response.provider == "openai"

    def test_mixed_output_items(self) -> None:
        """Multiple output item types in a single response."""
        adapter = _make_adapter()
        raw = _mock_openai_response(
            output=[
                SimpleNamespace(
                    type="message",
                    role="assistant",
                    content=[
                        SimpleNamespace(type="output_text", text="Here's my answer")
                    ],
                ),
                SimpleNamespace(
                    type="function_call",
                    call_id="call_1",
                    name="search",
                    arguments='{"q": "test"}',
                ),
            ],
            status="completed",
        )
        response = adapter._translate_response(raw)
        assert len(response.message.content) == 2
        assert response.message.content[0].kind == ContentKind.TEXT
        assert response.message.content[1].kind == ContentKind.TOOL_CALL


# ---------------------------------------------------------------------------
# Task 30: OpenAI Error Translation
# ---------------------------------------------------------------------------


def _make_api_status_error(
    status_code: int,
    message: str = "error",
    *,
    body: dict[str, Any] | None = None,
    retry_after: str | None = None,
) -> openai.APIStatusError:
    """Create a mock OpenAI APIStatusError."""
    import httpx

    headers: dict[str, str] = {}
    if retry_after is not None:
        headers["retry-after"] = retry_after
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(
        status_code=status_code,
        headers=headers,
        json=body or {"error": {"type": "api_error", "message": message}},
        request=request,
    )
    return openai.APIStatusError(
        message=message,
        response=response,
        body=body or {"error": {"type": "api_error", "message": message}},
    )


class TestErrorTranslation:
    """Task 30: Verify OpenAI SDK exceptions → unified error hierarchy."""

    def test_authentication_error(self) -> None:
        """401 → errors.AuthenticationError."""
        adapter = _make_adapter()
        exc = _make_api_status_error(401, "Invalid API key")
        result = adapter._translate_error(exc)
        assert isinstance(result, E.AuthenticationError)
        assert result.retryable is False

    def test_permission_denied_error(self) -> None:
        """403 → errors.AccessDeniedError."""
        adapter = _make_adapter()
        exc = _make_api_status_error(403, "Forbidden")
        result = adapter._translate_error(exc)
        assert isinstance(result, E.AccessDeniedError)
        assert result.retryable is False

    def test_not_found_error(self) -> None:
        """404 → errors.NotFoundError."""
        adapter = _make_adapter()
        exc = _make_api_status_error(404, "Model not found")
        result = adapter._translate_error(exc)
        assert isinstance(result, E.NotFoundError)

    def test_bad_request_error(self) -> None:
        """400 → errors.InvalidRequestError."""
        adapter = _make_adapter()
        exc = _make_api_status_error(400, "Bad request")
        result = adapter._translate_error(exc)
        assert isinstance(result, E.InvalidRequestError)
        assert result.retryable is False

    def test_rate_limit_error(self) -> None:
        """429 → errors.RateLimitError with retryable=True."""
        adapter = _make_adapter()
        exc = _make_api_status_error(429, "Rate limited")
        result = adapter._translate_error(exc)
        assert isinstance(result, E.RateLimitError)
        assert result.retryable is True

    def test_rate_limit_with_retry_after(self) -> None:
        """429 with Retry-After header → retry_after parsed."""
        adapter = _make_adapter()
        exc = _make_api_status_error(429, "Rate limited", retry_after="30")
        result = adapter._translate_error(exc)
        assert isinstance(result, E.RateLimitError)
        assert isinstance(result, E.ProviderError)
        assert result.retry_after == 30.0

    def test_server_error(self) -> None:
        """500 → errors.ServerError with retryable=True."""
        adapter = _make_adapter()
        exc = _make_api_status_error(500, "Internal server error")
        result = adapter._translate_error(exc)
        assert isinstance(result, E.ServerError)
        assert result.retryable is True

    def test_connection_error(self) -> None:
        """openai.APIConnectionError → errors.NetworkError."""
        adapter = _make_adapter()
        exc = openai.APIConnectionError(request=SimpleNamespace(url="test"))  # type: ignore[arg-type]
        result = adapter._translate_error(exc)
        assert isinstance(result, E.NetworkError)
        assert result.retryable is True

    def test_timeout_error(self) -> None:
        """openai.APITimeoutError → errors.RequestTimeoutError."""
        adapter = _make_adapter()
        exc = openai.APITimeoutError(request=SimpleNamespace(url="test"))  # type: ignore[arg-type]
        result = adapter._translate_error(exc)
        assert isinstance(result, E.RequestTimeoutError)
        assert result.retryable is True

    def test_error_preserves_cause(self) -> None:
        """Translated errors preserve the original exception as cause."""
        adapter = _make_adapter()
        exc = _make_api_status_error(500, "Server error")
        result = adapter._translate_error(exc)
        assert result.cause is exc

    def test_context_length_error(self) -> None:
        """413 → errors.ContextLengthError."""
        adapter = _make_adapter()
        exc = _make_api_status_error(413, "Context length exceeded")
        result = adapter._translate_error(exc)
        assert isinstance(result, E.ContextLengthError)
        assert result.retryable is False


# ---------------------------------------------------------------------------
# Task 31: OpenAI complete() Integration
# ---------------------------------------------------------------------------


class TestCompleteIntegration:
    """Task 31: Wire up request/response/error into complete()."""

    def test_complete_round_trip(self) -> None:
        """Full round-trip: unified Request → SDK call → unified Response."""
        with patch("unified_llm.adapters.openai.openai.AsyncOpenAI") as mock_cls:
            adapter = OpenAIAdapter(api_key="test-key")
            mock_client = mock_cls.return_value
            _mock_raw = MagicMock()
            _mock_raw.parse.return_value = _mock_openai_response(
                output=[
                    SimpleNamespace(
                        type="message",
                        role="assistant",
                        content=[
                            SimpleNamespace(type="output_text", text="Hello from GPT!")
                        ],
                    )
                ],
            )
            _mock_raw.headers = {}
            mock_client.responses.with_raw_response.create = AsyncMock(
                return_value=_mock_raw
            )

            request = Request(
                model="gpt-4.1",
                messages=[Message.user("Hi")],
            )
            response = asyncio.run(adapter.complete(request))

            assert response.text == "Hello from GPT!"
            assert response.provider == "openai"
            assert response.finish_reason.reason == "stop"
            mock_client.responses.with_raw_response.create.assert_called_once()

    def test_complete_passes_translated_kwargs(self) -> None:
        """complete() passes correctly translated kwargs to SDK."""
        with patch("unified_llm.adapters.openai.openai.AsyncOpenAI") as mock_cls:
            adapter = OpenAIAdapter(api_key="test-key")
            mock_client = mock_cls.return_value
            _mock_raw = MagicMock()
            _mock_raw.parse.return_value = _mock_openai_response()
            _mock_raw.headers = {}
            mock_client.responses.with_raw_response.create = AsyncMock(
                return_value=_mock_raw
            )

            request = Request(
                model="gpt-4.1",
                messages=[
                    Message.system("Be helpful"),
                    Message.user("Hello"),
                ],
                temperature=0.5,
                max_tokens=1024,
            )
            asyncio.run(adapter.complete(request))

            call_kwargs = mock_client.responses.with_raw_response.create.call_args[1]
            assert call_kwargs["model"] == "gpt-4.1"
            assert call_kwargs["max_output_tokens"] == 1024
            assert call_kwargs["temperature"] == 0.5
            assert "instructions" in call_kwargs

    def test_complete_translates_api_errors(self) -> None:
        """complete() catches SDK exceptions and raises unified errors."""
        with patch("unified_llm.adapters.openai.openai.AsyncOpenAI") as mock_cls:
            adapter = OpenAIAdapter(api_key="test-key")
            mock_client = mock_cls.return_value
            mock_client.responses.with_raw_response.create = AsyncMock(
                side_effect=_make_api_status_error(429, "Rate limited")
            )

            request = Request(
                model="gpt-4.1",
                messages=[Message.user("Hi")],
            )
            with pytest.raises(E.RateLimitError):
                asyncio.run(adapter.complete(request))

    def test_complete_translates_connection_errors(self) -> None:
        """complete() catches connection errors and raises NetworkError."""
        with patch("unified_llm.adapters.openai.openai.AsyncOpenAI") as mock_cls:
            adapter = OpenAIAdapter(api_key="test-key")
            mock_client = mock_cls.return_value
            mock_client.responses.with_raw_response.create = AsyncMock(
                side_effect=openai.APIConnectionError(
                    request=SimpleNamespace(url="test")  # type: ignore[arg-type]
                )
            )

            request = Request(
                model="gpt-4.1",
                messages=[Message.user("Hi")],
            )
            with pytest.raises(E.NetworkError):
                asyncio.run(adapter.complete(request))

    def test_complete_with_tool_response(self) -> None:
        """complete() handles function_call responses correctly."""
        with patch("unified_llm.adapters.openai.openai.AsyncOpenAI") as mock_cls:
            adapter = OpenAIAdapter(api_key="test-key")
            mock_client = mock_cls.return_value
            _mock_raw = MagicMock()
            _mock_raw.parse.return_value = _mock_openai_response(
                output=[
                    SimpleNamespace(
                        type="function_call",
                        call_id="call_1",
                        name="get_weather",
                        arguments='{"city": "SF"}',
                    ),
                ],
                status="completed",
            )
            _mock_raw.headers = {}
            mock_client.responses.with_raw_response.create = AsyncMock(
                return_value=_mock_raw
            )

            request = Request(
                model="gpt-4.1",
                messages=[Message.user("What's the weather?")],
                tools=[
                    Tool(
                        name="get_weather",
                        description="Get weather",
                        parameters={},
                    )
                ],
            )
            response = asyncio.run(adapter.complete(request))

            assert response.finish_reason.reason == "tool_calls"
            assert len(response.tool_calls) == 1
            assert response.tool_calls[0].name == "get_weather"


# ---------------------------------------------------------------------------
# Task 32: OpenAI Streaming Translation
# ---------------------------------------------------------------------------


def _make_stream_events_text() -> list[SimpleNamespace]:
    """Mock OpenAI Responses API stream events for a simple text response."""
    return [
        SimpleNamespace(
            type="response.created",
            response=SimpleNamespace(id="resp_stream1"),
        ),
        SimpleNamespace(
            type="response.output_text.delta",
            delta="Hello",
        ),
        SimpleNamespace(
            type="response.output_text.delta",
            delta=" world",
        ),
        SimpleNamespace(
            type="response.output_text.done",
            text="Hello world",
        ),
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(
                id="resp_stream1",
                model="gpt-4.1",
                status="completed",
                output=[],
                usage=SimpleNamespace(
                    input_tokens=10,
                    output_tokens=5,
                    total_tokens=15,
                    output_tokens_details=None,
                    input_tokens_details=None,
                ),
            ),
        ),
    ]


def _make_stream_events_tool_use() -> list[SimpleNamespace]:
    """Mock stream events for a function_call response."""
    return [
        SimpleNamespace(
            type="response.created",
            response=SimpleNamespace(id="resp_tool1"),
        ),
        SimpleNamespace(
            type="response.function_call_arguments.delta",
            call_id="call_1",
            name="get_weather",
            item_id="call_1",
            delta='{"city"',
        ),
        SimpleNamespace(
            type="response.function_call_arguments.delta",
            call_id="call_1",
            name="get_weather",
            item_id="call_1",
            delta=': "SF"}',
        ),
        SimpleNamespace(
            type="response.function_call_arguments.done",
            call_id="call_1",
            name="get_weather",
            item_id="call_1",
            arguments='{"city": "SF"}',
        ),
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(
                id="resp_tool1",
                model="gpt-4.1",
                status="completed",
                output=[
                    SimpleNamespace(
                        type="function_call",
                        call_id="call_1",
                        name="get_weather",
                        arguments='{"city": "SF"}',
                    )
                ],
                usage=SimpleNamespace(
                    input_tokens=15,
                    output_tokens=10,
                    total_tokens=25,
                    output_tokens_details=None,
                    input_tokens_details=None,
                ),
            ),
        ),
    ]


class _MockAsyncStream:
    """Mock async iterator for OpenAI streaming."""

    def __init__(self, events: list[SimpleNamespace]) -> None:
        self._events = events
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._events):
            raise StopAsyncIteration
        event = self._events[self._index]
        self._index += 1
        return event


from unified_llm.types import StreamEventType  # noqa: E402


class TestStreamingTranslation:
    """Task 32: Verify Responses API streaming events → unified StreamEvent sequence."""

    def _collect_stream(
        self,
        adapter: OpenAIAdapter,
        request: Request,
        events: list[SimpleNamespace],
    ) -> list[Any]:
        """Run streaming and collect all events."""
        from unified_llm.types import StreamEvent as SE

        mock_create = AsyncMock(return_value=_MockAsyncStream(events))
        with patch.object(adapter._client.responses, "create", mock_create):
            result: list[SE] = []

            async def run():
                async for evt in adapter.stream(request):
                    result.append(evt)

            asyncio.run(run())
        return result

    def test_text_stream_event_sequence(self) -> None:
        """Text stream: STREAM_START → TEXT_START → TEXT_DELTA*2 → TEXT_END → FINISH."""
        adapter = _make_adapter()
        request = Request(model="gpt-4.1", messages=[Message.user("Hi")])
        events = self._collect_stream(adapter, request, _make_stream_events_text())

        types = [e.type for e in events]
        assert StreamEventType.STREAM_START in types
        assert StreamEventType.TEXT_START in types
        assert types.count(StreamEventType.TEXT_DELTA) == 2
        assert StreamEventType.TEXT_END in types
        assert StreamEventType.FINISH in types

    def test_text_deltas_contain_text(self) -> None:
        """TEXT_DELTA events carry the delta text."""
        adapter = _make_adapter()
        request = Request(model="gpt-4.1", messages=[Message.user("Hi")])
        events = self._collect_stream(adapter, request, _make_stream_events_text())

        deltas = [e for e in events if e.type == StreamEventType.TEXT_DELTA]
        assert deltas[0].delta == "Hello"
        assert deltas[1].delta == " world"

    def test_finish_event_has_usage(self) -> None:
        """FINISH event carries usage and finish_reason."""
        adapter = _make_adapter()
        request = Request(model="gpt-4.1", messages=[Message.user("Hi")])
        events = self._collect_stream(adapter, request, _make_stream_events_text())

        finish = [e for e in events if e.type == StreamEventType.FINISH][0]
        assert finish.finish_reason is not None
        assert finish.finish_reason.reason == "stop"
        assert finish.usage is not None
        assert finish.usage.input_tokens == 10
        assert finish.usage.output_tokens == 5

    def test_tool_call_stream_events(self) -> None:
        """Tool use stream: TOOL_CALL_START → TOOL_CALL_DELTA → TOOL_CALL_END."""
        adapter = _make_adapter()
        request = Request(model="gpt-4.1", messages=[Message.user("Hi")])
        events = self._collect_stream(adapter, request, _make_stream_events_tool_use())

        types = [e.type for e in events]
        assert StreamEventType.TOOL_CALL_START in types
        assert StreamEventType.TOOL_CALL_DELTA in types
        assert StreamEventType.TOOL_CALL_END in types

    def test_tool_call_end_has_parsed_args(self) -> None:
        """TOOL_CALL_END carries the complete parsed tool call."""
        adapter = _make_adapter()
        request = Request(model="gpt-4.1", messages=[Message.user("Hi")])
        events = self._collect_stream(adapter, request, _make_stream_events_tool_use())

        end_evt = [e for e in events if e.type == StreamEventType.TOOL_CALL_END][0]
        assert end_evt.tool_call is not None
        assert end_evt.tool_call.id == "call_1"
        assert end_evt.tool_call.name == "get_weather"
        assert end_evt.tool_call.arguments == {"city": "SF"}

    def test_tool_use_finish_reason(self) -> None:
        """Tool use stream: FINISH has reason='tool_calls'."""
        adapter = _make_adapter()
        request = Request(model="gpt-4.1", messages=[Message.user("Hi")])
        events = self._collect_stream(adapter, request, _make_stream_events_tool_use())

        finish = [e for e in events if e.type == StreamEventType.FINISH][0]
        assert finish.finish_reason is not None
        assert finish.finish_reason.reason == "tool_calls"

    def test_stream_error_translated(self) -> None:
        """Stream errors are caught and translated to unified errors."""
        adapter = _make_adapter()
        request = Request(model="gpt-4.1", messages=[Message.user("Hi")])

        exc = _make_api_status_error(500, "Server error")
        with patch.object(adapter._client.responses, "create", side_effect=exc):
            with pytest.raises(E.ServerError):

                async def run():
                    async for _ in adapter.stream(request):
                        pass

                asyncio.run(run())


# ---------------------------------------------------------------------------
# Task 33: OpenAI Reasoning Tokens
# ---------------------------------------------------------------------------


class TestReasoningTokens:
    """Task 33: Verify reasoning_tokens and reasoning_effort passthrough."""

    def test_reasoning_effort_in_request(self) -> None:
        """reasoning_effort → reasoning.effort in Responses API request."""
        adapter = _make_adapter()
        request = Request(
            model="o4-mini",
            messages=[Message.user("Solve this complex problem")],
            reasoning_effort="high",
        )
        kwargs = adapter._translate_request(request)
        assert kwargs["reasoning"] == {"effort": "high"}

    def test_reasoning_effort_low(self) -> None:
        """reasoning_effort='low' passes through correctly."""
        adapter = _make_adapter()
        request = Request(
            model="o4-mini",
            messages=[Message.user("Quick question")],
            reasoning_effort="low",
        )
        kwargs = adapter._translate_request(request)
        assert kwargs["reasoning"] == {"effort": "low"}

    def test_reasoning_effort_medium(self) -> None:
        """reasoning_effort='medium' passes through correctly."""
        adapter = _make_adapter()
        request = Request(
            model="o4-mini",
            messages=[Message.user("Question")],
            reasoning_effort="medium",
        )
        kwargs = adapter._translate_request(request)
        assert kwargs["reasoning"] == {"effort": "medium"}

    def test_no_reasoning_when_not_set(self) -> None:
        """No reasoning param when reasoning_effort is not set."""
        adapter = _make_adapter()
        request = Request(
            model="gpt-4.1",
            messages=[Message.user("Hi")],
        )
        kwargs = adapter._translate_request(request)
        assert "reasoning" not in kwargs

    def test_reasoning_tokens_in_response(self) -> None:
        """reasoning_tokens populated from output_tokens_details."""
        adapter = _make_adapter()
        raw = _mock_openai_response(
            model="o4-mini",
            input_tokens=100,
            output_tokens=80,
            reasoning_tokens=60,
        )
        response = adapter._translate_response(raw)
        assert response.usage.reasoning_tokens == 60
        assert response.usage.output_tokens == 80

    def test_reasoning_tokens_none_for_non_reasoning_model(self) -> None:
        """reasoning_tokens is None when model doesn't use reasoning."""
        adapter = _make_adapter()
        raw = _mock_openai_response(
            model="gpt-4.1",
            input_tokens=100,
            output_tokens=50,
        )
        response = adapter._translate_response(raw)
        assert response.usage.reasoning_tokens is None

    def test_reasoning_tokens_in_stream_finish(self) -> None:
        """Streaming FINISH event carries reasoning_tokens in usage."""
        adapter = _make_adapter()
        request = Request(model="o4-mini", messages=[Message.user("Solve")])

        stream_events = [
            SimpleNamespace(
                type="response.created",
                response=SimpleNamespace(id="resp_reason1"),
            ),
            SimpleNamespace(
                type="response.output_text.delta",
                delta="Answer",
            ),
            SimpleNamespace(
                type="response.output_text.done",
                text="Answer",
            ),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(
                    id="resp_reason1",
                    model="o4-mini",
                    status="completed",
                    output=[],
                    usage=SimpleNamespace(
                        input_tokens=100,
                        output_tokens=80,
                        total_tokens=180,
                        output_tokens_details=SimpleNamespace(reasoning_tokens=60),
                        input_tokens_details=None,
                    ),
                ),
            ),
        ]

        mock_create = AsyncMock(return_value=_MockAsyncStream(stream_events))
        with patch.object(adapter._client.responses, "create", mock_create):
            result: list[Any] = []

            async def run():
                async for evt in adapter.stream(request):
                    result.append(evt)

            asyncio.run(run())

        finish = [e for e in result if e.type == StreamEventType.FINISH][0]
        assert finish.usage is not None
        assert finish.usage.reasoning_tokens == 60

    def test_cache_read_tokens_in_response(self) -> None:
        """cache_read_tokens populated from input_tokens_details.cached_tokens."""
        adapter = _make_adapter()
        raw = _mock_openai_response(
            input_tokens=100,
            output_tokens=50,
            cached_tokens=80,
        )
        response = adapter._translate_response(raw)
        assert response.usage.cache_read_tokens == 80
