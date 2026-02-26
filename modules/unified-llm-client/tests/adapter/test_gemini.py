"""Tests for unified_llm.adapters.gemini — Gemini API adapter."""

from __future__ import annotations

import base64
from unittest.mock import patch

from unified_llm.adapters.gemini import GeminiAdapter
from unified_llm.types import (
    ContentKind,
    ContentPart,
    ImageData,
    Message,
    Request,
    Role,
    StreamEventType,
    Tool,
    ToolCallData,
    ToolChoice,
    ToolResultData,
)


def _make_adapter() -> GeminiAdapter:
    """Create adapter with mocked google.genai.Client."""
    with patch("unified_llm.adapters.gemini.genai.Client"):
        return GeminiAdapter(api_key="test-key")


# ---------------------------------------------------------------------------
# Task 34: Gemini Request Translation
# ---------------------------------------------------------------------------


class TestRequestTranslation:
    """Task 34: Verify unified Request → Gemini API format."""

    def test_system_message_to_system_instruction(self) -> None:
        """System messages → systemInstruction parameter."""
        adapter = _make_adapter()
        request = Request(
            model="gemini-2.0-flash",
            messages=[
                Message.system("You are helpful"),
                Message.user("Hello"),
            ],
        )
        kwargs = adapter._translate_request(request)
        assert "system_instruction" in kwargs
        assert kwargs["system_instruction"] == "You are helpful"
        # System should NOT appear in contents
        for content in kwargs["contents"]:
            assert content["role"] != "system"

    def test_developer_role_merged_with_system(self) -> None:
        """DEVELOPER role messages merge into systemInstruction."""
        adapter = _make_adapter()
        request = Request(
            model="gemini-2.0-flash",
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
        assert "System instructions" in kwargs["system_instruction"]
        assert "Dev instructions" in kwargs["system_instruction"]

    def test_user_message_translated(self) -> None:
        """User messages → 'user' role with text parts."""
        adapter = _make_adapter()
        request = Request(
            model="gemini-2.0-flash",
            messages=[Message.user("Hello world")],
        )
        kwargs = adapter._translate_request(request)
        assert len(kwargs["contents"]) == 1
        msg = kwargs["contents"][0]
        assert msg["role"] == "user"
        assert msg["parts"][0] == {"text": "Hello world"}

    def test_assistant_message_to_model_role(self) -> None:
        """Assistant messages → 'model' role."""
        adapter = _make_adapter()
        request = Request(
            model="gemini-2.0-flash",
            messages=[
                Message.user("Hi"),
                Message.assistant("Hello!"),
            ],
        )
        kwargs = adapter._translate_request(request)
        assert kwargs["contents"][1]["role"] == "model"
        assert kwargs["contents"][1]["parts"][0] == {"text": "Hello!"}

    def test_text_content_part(self) -> None:
        """TEXT → {"text": "..."} parts."""
        adapter = _make_adapter()
        request = Request(
            model="gemini-2.0-flash",
            messages=[Message.user("Hello")],
        )
        kwargs = adapter._translate_request(request)
        assert kwargs["contents"][0]["parts"] == [{"text": "Hello"}]

    def test_image_url_to_file_data(self) -> None:
        """IMAGE with URL → fileData part."""
        adapter = _make_adapter()
        request = Request(
            model="gemini-2.0-flash",
            messages=[
                Message(
                    role=Role.USER,
                    content=[
                        ContentPart(kind=ContentKind.TEXT, text="What's this?"),
                        ContentPart(
                            kind=ContentKind.IMAGE,
                            image=ImageData(
                                url="https://example.com/img.png",
                                media_type="image/png",
                            ),
                        ),
                    ],
                ),
            ],
        )
        kwargs = adapter._translate_request(request)
        parts = kwargs["contents"][0]["parts"]
        assert parts[0] == {"text": "What's this?"}
        assert parts[1] == {
            "file_data": {
                "mime_type": "image/png",
                "file_uri": "https://example.com/img.png",
            }
        }

    def test_image_base64_to_inline_data(self) -> None:
        """IMAGE with data → inlineData part."""
        adapter = _make_adapter()
        raw_bytes = b"\x89PNG"
        request = Request(
            model="gemini-2.0-flash",
            messages=[
                Message(
                    role=Role.USER,
                    content=[
                        ContentPart(
                            kind=ContentKind.IMAGE,
                            image=ImageData(data=raw_bytes, media_type="image/png"),
                        ),
                    ],
                ),
            ],
        )
        kwargs = adapter._translate_request(request)
        part = kwargs["contents"][0]["parts"][0]
        assert "inline_data" in part
        assert part["inline_data"]["mime_type"] == "image/png"
        assert part["inline_data"]["data"] == base64.b64encode(raw_bytes).decode()

    def test_tool_call_to_function_call(self) -> None:
        """TOOL_CALL → functionCall parts."""
        adapter = _make_adapter()
        request = Request(
            model="gemini-2.0-flash",
            messages=[
                Message.user("What's the weather?"),
                Message(
                    role=Role.ASSISTANT,
                    content=[
                        ContentPart(
                            kind=ContentKind.TOOL_CALL,
                            tool_call=ToolCallData(
                                id="call_123",
                                name="get_weather",
                                arguments={"city": "SF"},
                            ),
                        ),
                    ],
                ),
            ],
        )
        kwargs = adapter._translate_request(request)
        model_msg = kwargs["contents"][1]
        assert model_msg["role"] == "model"
        fc_part = model_msg["parts"][0]
        assert "function_call" in fc_part
        assert fc_part["function_call"]["name"] == "get_weather"
        assert fc_part["function_call"]["args"] == {"city": "SF"}

    def test_tool_result_to_function_response(self) -> None:
        """TOOL_RESULT → functionResponse parts using function NAME (not ID)."""
        adapter = _make_adapter()
        request = Request(
            model="gemini-2.0-flash",
            messages=[
                Message.user("What's the weather?"),
                Message(
                    role=Role.ASSISTANT,
                    content=[
                        ContentPart(
                            kind=ContentKind.TOOL_CALL,
                            tool_call=ToolCallData(
                                id="call_123",
                                name="get_weather",
                                arguments={"city": "SF"},
                            ),
                        ),
                    ],
                ),
                Message(
                    role=Role.TOOL,
                    content=[
                        ContentPart(
                            kind=ContentKind.TOOL_RESULT,
                            tool_result=ToolResultData(
                                tool_call_id="call_123",
                                content="72F sunny",
                            ),
                        ),
                    ],
                ),
            ],
        )
        kwargs = adapter._translate_request(request)
        tool_msg = kwargs["contents"][2]
        assert tool_msg["role"] == "user"
        fr_part = tool_msg["parts"][0]
        assert "function_response" in fr_part
        # Uses function NAME, not the call ID
        assert fr_part["function_response"]["name"] == "get_weather"
        assert fr_part["function_response"]["response"] == {"result": "72F sunny"}

    def test_tool_result_dict_content_passed_directly(self) -> None:
        """Dict tool result content passed directly, not wrapped."""
        adapter = _make_adapter()
        request = Request(
            model="gemini-2.0-flash",
            messages=[
                Message.user("Query"),
                Message(
                    role=Role.ASSISTANT,
                    content=[
                        ContentPart(
                            kind=ContentKind.TOOL_CALL,
                            tool_call=ToolCallData(
                                id="call_1",
                                name="search",
                                arguments={"q": "test"},
                            ),
                        ),
                    ],
                ),
                Message(
                    role=Role.TOOL,
                    content=[
                        ContentPart(
                            kind=ContentKind.TOOL_RESULT,
                            tool_result=ToolResultData(
                                tool_call_id="call_1",
                                content={"results": ["a", "b"]},
                            ),
                        ),
                    ],
                ),
            ],
        )
        kwargs = adapter._translate_request(request)
        tool_msg = kwargs["contents"][2]
        fr_part = tool_msg["parts"][0]
        assert fr_part["function_response"]["response"] == {"results": ["a", "b"]}

    def test_tool_definitions_to_function_declarations(self) -> None:
        """Tool definitions → functionDeclarations format."""
        adapter = _make_adapter()
        request = Request(
            model="gemini-2.0-flash",
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
        assert "tools" in kwargs
        assert len(kwargs["tools"]) == 1
        tool = kwargs["tools"][0]
        assert "function_declarations" in tool
        fd = tool["function_declarations"][0]
        assert fd["name"] == "get_weather"
        assert fd["description"] == "Get the weather"
        assert fd["parameters"]["type"] == "object"

    def test_generation_params_in_config(self) -> None:
        """temperature, top_p, max_tokens, stop_sequences → generation config."""
        adapter = _make_adapter()
        request = Request(
            model="gemini-2.0-flash",
            messages=[Message.user("Hi")],
            temperature=0.7,
            top_p=0.9,
            max_tokens=1024,
            stop_sequences=["END"],
        )
        kwargs = adapter._translate_request(request)
        config = kwargs.get("config", {})
        assert config["temperature"] == 0.7
        assert config["top_p"] == 0.9
        assert config["max_output_tokens"] == 1024
        assert config["stop_sequences"] == ["END"]

    def test_provider_options_passthrough(self) -> None:
        """provider_options['gemini'] passes through extra parameters."""
        adapter = _make_adapter()
        request = Request(
            model="gemini-2.0-flash",
            messages=[Message.user("Hi")],
            provider_options={
                "gemini": {
                    "safety_settings": [
                        {"category": "HARM", "threshold": "NONE"}
                    ],
                }
            },
        )
        kwargs = adapter._translate_request(request)
        assert kwargs["safety_settings"] == [
            {"category": "HARM", "threshold": "NONE"}
        ]

    def test_tool_choice_auto(self) -> None:
        adapter = _make_adapter()
        request = Request(
            model="gemini-2.0-flash",
            messages=[Message.user("Hi")],
            tools=[Tool(name="t", description="d", parameters={})],
            tool_choice=ToolChoice(mode="auto"),
        )
        kwargs = adapter._translate_request(request)
        config = kwargs.get("config", {})
        fc_config = config.get("tool_config", {}).get(
            "function_calling_config", {}
        )
        assert fc_config.get("mode") == "AUTO"

    def test_tool_choice_none(self) -> None:
        adapter = _make_adapter()
        request = Request(
            model="gemini-2.0-flash",
            messages=[Message.user("Hi")],
            tools=[Tool(name="t", description="d", parameters={})],
            tool_choice=ToolChoice(mode="none"),
        )
        kwargs = adapter._translate_request(request)
        config = kwargs.get("config", {})
        fc_config = config.get("tool_config", {}).get(
            "function_calling_config", {}
        )
        assert fc_config.get("mode") == "NONE"

    def test_tool_choice_required(self) -> None:
        adapter = _make_adapter()
        request = Request(
            model="gemini-2.0-flash",
            messages=[Message.user("Hi")],
            tools=[Tool(name="t", description="d", parameters={})],
            tool_choice=ToolChoice(mode="required"),
        )
        kwargs = adapter._translate_request(request)
        config = kwargs.get("config", {})
        fc_config = config.get("tool_config", {}).get(
            "function_calling_config", {}
        )
        assert fc_config.get("mode") == "ANY"

    def test_name_property(self) -> None:
        """Adapter name is 'gemini'."""
        adapter = _make_adapter()
        assert adapter.name == "gemini"

    def test_model_passed_through(self) -> None:
        """Contents are correctly built for model."""
        adapter = _make_adapter()
        request = Request(
            model="gemini-2.0-flash",
            messages=[Message.user("Hi")],
        )
        kwargs = adapter._translate_request(request)
        assert len(kwargs["contents"]) == 1


# ---------------------------------------------------------------------------
# Task 35: Gemini Response Translation
# ---------------------------------------------------------------------------

from types import SimpleNamespace  # noqa: E402


def _mock_gemini_response(
    *,
    text: str | None = "Hello!",
    parts: list[SimpleNamespace] | None = None,
    finish_reason: str = "STOP",
    prompt_token_count: int = 10,
    candidates_token_count: int = 20,
    total_token_count: int = 30,
    thoughts_token_count: int | None = None,
    cached_content_token_count: int | None = None,
    model_version: str = "gemini-2.0-flash",
) -> SimpleNamespace:
    """Create a mock Gemini generateContent response object."""
    if parts is None:
        parts = [SimpleNamespace(text=text, function_call=None)]

    usage_kwargs: dict = {
        "prompt_token_count": prompt_token_count,
        "candidates_token_count": candidates_token_count,
        "total_token_count": total_token_count,
    }
    if thoughts_token_count is not None:
        usage_kwargs["thoughts_token_count"] = thoughts_token_count
    if cached_content_token_count is not None:
        usage_kwargs["cached_content_token_count"] = cached_content_token_count

    return SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=parts,
                    role="model",
                ),
                finish_reason=finish_reason,
            ),
        ],
        usage_metadata=SimpleNamespace(**usage_kwargs),
        model_version=model_version,
    )


class TestResponseTranslation:
    """Task 35: Verify Gemini response → unified Response."""

    def test_text_content_part(self) -> None:
        """Text parts → TEXT ContentParts."""
        adapter = _make_adapter()
        raw = _mock_gemini_response(text="Hello world")
        response = adapter._translate_response(raw, model="gemini-2.0-flash")
        assert len(response.message.content) == 1
        part = response.message.content[0]
        assert part.kind == ContentKind.TEXT
        assert part.text == "Hello world"

    def test_function_call_content_part(self) -> None:
        """functionCall parts → TOOL_CALL ContentParts with synthetic ID."""
        adapter = _make_adapter()
        raw = _mock_gemini_response(
            parts=[
                SimpleNamespace(
                    text=None,
                    function_call=SimpleNamespace(
                        name="get_weather",
                        args={"city": "SF"},
                    ),
                ),
            ],
            finish_reason="STOP",
        )
        response = adapter._translate_response(raw, model="gemini-2.0-flash")
        part = response.message.content[0]
        assert part.kind == ContentKind.TOOL_CALL
        assert part.tool_call is not None
        assert part.tool_call.name == "get_weather"
        assert part.tool_call.arguments == {"city": "SF"}
        # Synthetic ID should start with "call_"
        assert part.tool_call.id.startswith("call_")

    def test_finish_reason_stop(self) -> None:
        """STOP → FinishReason(reason='stop')."""
        adapter = _make_adapter()
        raw = _mock_gemini_response(finish_reason="STOP")
        response = adapter._translate_response(raw, model="gemini-2.0-flash")
        assert response.finish_reason.reason == "stop"
        assert response.finish_reason.raw == "STOP"

    def test_finish_reason_max_tokens(self) -> None:
        """MAX_TOKENS → FinishReason(reason='length')."""
        adapter = _make_adapter()
        raw = _mock_gemini_response(finish_reason="MAX_TOKENS")
        response = adapter._translate_response(raw, model="gemini-2.0-flash")
        assert response.finish_reason.reason == "length"

    def test_finish_reason_safety(self) -> None:
        """SAFETY → FinishReason(reason='content_filter')."""
        adapter = _make_adapter()
        raw = _mock_gemini_response(finish_reason="SAFETY")
        response = adapter._translate_response(raw, model="gemini-2.0-flash")
        assert response.finish_reason.reason == "content_filter"

    def test_finish_reason_recitation(self) -> None:
        """RECITATION → FinishReason(reason='content_filter')."""
        adapter = _make_adapter()
        raw = _mock_gemini_response(finish_reason="RECITATION")
        response = adapter._translate_response(raw, model="gemini-2.0-flash")
        assert response.finish_reason.reason == "content_filter"

    def test_finish_reason_inferred_tool_calls(self) -> None:
        """Presence of functionCall parts → FinishReason(reason='tool_calls')."""
        adapter = _make_adapter()
        raw = _mock_gemini_response(
            parts=[
                SimpleNamespace(
                    text=None,
                    function_call=SimpleNamespace(
                        name="search", args={"q": "test"}
                    ),
                ),
            ],
            finish_reason="STOP",
        )
        response = adapter._translate_response(raw, model="gemini-2.0-flash")
        assert response.finish_reason.reason == "tool_calls"

    def test_usage_extraction(self) -> None:
        """usageMetadata → Usage fields mapped correctly."""
        adapter = _make_adapter()
        raw = _mock_gemini_response(
            prompt_token_count=100,
            candidates_token_count=50,
            total_token_count=150,
        )
        response = adapter._translate_response(raw, model="gemini-2.0-flash")
        assert response.usage.input_tokens == 100
        assert response.usage.output_tokens == 50
        assert response.usage.total_tokens == 150

    def test_usage_reasoning_tokens(self) -> None:
        """thoughtsTokenCount → reasoning_tokens."""
        adapter = _make_adapter()
        raw = _mock_gemini_response(thoughts_token_count=200)
        response = adapter._translate_response(raw, model="gemini-2.0-flash")
        assert response.usage.reasoning_tokens == 200

    def test_usage_cache_tokens(self) -> None:
        """cachedContentTokenCount → cache_read_tokens."""
        adapter = _make_adapter()
        raw = _mock_gemini_response(cached_content_token_count=500)
        response = adapter._translate_response(raw, model="gemini-2.0-flash")
        assert response.usage.cache_read_tokens == 500

    def test_response_metadata(self) -> None:
        """Response provider set to 'gemini'."""
        adapter = _make_adapter()
        raw = _mock_gemini_response()
        response = adapter._translate_response(raw, model="gemini-2.0-flash")
        assert response.provider == "gemini"
        assert response.model == "gemini-2.0-flash"

    def test_mixed_content_parts(self) -> None:
        """Multiple content part types in a single response."""
        adapter = _make_adapter()
        raw = _mock_gemini_response(
            parts=[
                SimpleNamespace(text="Let me search", function_call=None),
                SimpleNamespace(
                    text=None,
                    function_call=SimpleNamespace(
                        name="search", args={"q": "test"}
                    ),
                ),
            ],
            finish_reason="STOP",
        )
        response = adapter._translate_response(raw, model="gemini-2.0-flash")
        assert len(response.message.content) == 2
        assert response.message.content[0].kind == ContentKind.TEXT
        assert response.message.content[1].kind == ContentKind.TOOL_CALL


# ---------------------------------------------------------------------------
# Task 36: Gemini Error Translation (including gRPC)
# ---------------------------------------------------------------------------

import unified_llm.errors as E  # noqa: E402
from google.genai import errors as genai_errors  # noqa: E402


def _make_genai_client_error(
    code: int,
    *,
    status: str | None = None,
    message: str = "error",
) -> genai_errors.ClientError:
    """Create a mock google-genai ClientError (4xx)."""
    response_json = {
        "error": {
            "code": code,
            "message": message,
            "status": status,
        }
    }
    return genai_errors.ClientError(code, response_json, response=None)


def _make_genai_server_error(
    code: int,
    *,
    status: str | None = None,
    message: str = "error",
) -> genai_errors.ServerError:
    """Create a mock google-genai ServerError (5xx)."""
    response_json = {
        "error": {
            "code": code,
            "message": message,
            "status": status,
        }
    }
    return genai_errors.ServerError(code, response_json, response=None)


class TestErrorTranslation:
    """Task 36: Verify google-genai SDK exceptions → unified error hierarchy."""

    # -- HTTP status code mapping --

    def test_400_invalid_request(self) -> None:
        """400 → InvalidRequestError."""
        adapter = _make_adapter()
        exc = _make_genai_client_error(400, status="INVALID_ARGUMENT", message="Bad request")
        result = adapter._translate_error(exc)
        assert isinstance(result, E.InvalidRequestError)
        assert result.retryable is False

    def test_401_authentication(self) -> None:
        """401 → AuthenticationError."""
        adapter = _make_adapter()
        exc = _make_genai_client_error(401, status="UNAUTHENTICATED", message="Invalid API key")
        result = adapter._translate_error(exc)
        assert isinstance(result, E.AuthenticationError)
        assert result.retryable is False

    def test_403_access_denied(self) -> None:
        """403 → AccessDeniedError."""
        adapter = _make_adapter()
        exc = _make_genai_client_error(403, status="PERMISSION_DENIED", message="Forbidden")
        result = adapter._translate_error(exc)
        assert isinstance(result, E.AccessDeniedError)
        assert result.retryable is False

    def test_404_not_found(self) -> None:
        """404 → NotFoundError."""
        adapter = _make_adapter()
        exc = _make_genai_client_error(404, status="NOT_FOUND", message="Model not found")
        result = adapter._translate_error(exc)
        assert isinstance(result, E.NotFoundError)

    def test_429_rate_limit(self) -> None:
        """429 → RateLimitError."""
        adapter = _make_adapter()
        exc = _make_genai_client_error(429, status="RESOURCE_EXHAUSTED", message="Rate limited")
        result = adapter._translate_error(exc)
        assert isinstance(result, E.RateLimitError)
        assert result.retryable is True

    def test_500_server_error(self) -> None:
        """500 → ServerError."""
        adapter = _make_adapter()
        exc = _make_genai_server_error(500, status="INTERNAL", message="Internal error")
        result = adapter._translate_error(exc)
        assert isinstance(result, E.ServerError)
        assert result.retryable is True

    def test_503_server_error(self) -> None:
        """503 → ServerError."""
        adapter = _make_adapter()
        exc = _make_genai_server_error(503, status="UNAVAILABLE", message="Service unavailable")
        result = adapter._translate_error(exc)
        assert isinstance(result, E.ServerError)
        assert result.retryable is True

    # -- gRPC status code mapping (spec §6.4) --

    def test_grpc_not_found(self) -> None:
        """gRPC NOT_FOUND → NotFoundError."""
        adapter = _make_adapter()
        exc = _make_genai_client_error(404, status="NOT_FOUND")
        result = adapter._translate_error(exc)
        assert isinstance(result, E.NotFoundError)

    def test_grpc_invalid_argument(self) -> None:
        """gRPC INVALID_ARGUMENT → InvalidRequestError."""
        adapter = _make_adapter()
        exc = _make_genai_client_error(400, status="INVALID_ARGUMENT")
        result = adapter._translate_error(exc)
        assert isinstance(result, E.InvalidRequestError)

    def test_grpc_unauthenticated(self) -> None:
        """gRPC UNAUTHENTICATED → AuthenticationError."""
        adapter = _make_adapter()
        exc = _make_genai_client_error(401, status="UNAUTHENTICATED")
        result = adapter._translate_error(exc)
        assert isinstance(result, E.AuthenticationError)

    def test_grpc_permission_denied(self) -> None:
        """gRPC PERMISSION_DENIED → AccessDeniedError."""
        adapter = _make_adapter()
        exc = _make_genai_client_error(403, status="PERMISSION_DENIED")
        result = adapter._translate_error(exc)
        assert isinstance(result, E.AccessDeniedError)

    def test_grpc_resource_exhausted(self) -> None:
        """gRPC RESOURCE_EXHAUSTED → RateLimitError."""
        adapter = _make_adapter()
        exc = _make_genai_client_error(429, status="RESOURCE_EXHAUSTED")
        result = adapter._translate_error(exc)
        assert isinstance(result, E.RateLimitError)

    def test_grpc_unavailable(self) -> None:
        """gRPC UNAVAILABLE → ServerError."""
        adapter = _make_adapter()
        exc = _make_genai_server_error(503, status="UNAVAILABLE")
        result = adapter._translate_error(exc)
        assert isinstance(result, E.ServerError)

    def test_grpc_deadline_exceeded(self) -> None:
        """gRPC DEADLINE_EXCEEDED → RequestTimeoutError."""
        adapter = _make_adapter()
        exc = _make_genai_server_error(504, status="DEADLINE_EXCEEDED")
        result = adapter._translate_error(exc)
        assert isinstance(result, E.RequestTimeoutError)
        assert result.retryable is True

    def test_grpc_internal(self) -> None:
        """gRPC INTERNAL → ServerError."""
        adapter = _make_adapter()
        exc = _make_genai_server_error(500, status="INTERNAL")
        result = adapter._translate_error(exc)
        assert isinstance(result, E.ServerError)

    # -- Edge cases --

    def test_unknown_status_retryable(self) -> None:
        """Unknown status codes default to retryable (Spec §6.3)."""
        adapter = _make_adapter()
        exc = _make_genai_server_error(599, status="UNKNOWN")
        result = adapter._translate_error(exc)
        assert isinstance(result, E.ProviderError)
        assert result.retryable is True

    def test_generic_exception_fallback(self) -> None:
        """Non-genai exceptions → ProviderError fallback."""
        adapter = _make_adapter()
        exc = RuntimeError("Something broke")
        result = adapter._translate_error(exc)
        assert isinstance(result, E.ProviderError)
        assert result.retryable is True

    def test_error_preserves_cause(self) -> None:
        """Translated errors preserve the original exception as cause."""
        adapter = _make_adapter()
        exc = _make_genai_server_error(500, message="Server error")
        result = adapter._translate_error(exc)
        assert result.cause is exc


# ---------------------------------------------------------------------------
# Task 37: Gemini complete() Integration
# ---------------------------------------------------------------------------

import asyncio  # noqa: E402
from unittest.mock import AsyncMock, MagicMock  # noqa: E402

import pytest  # noqa: E402


class TestCompleteIntegration:
    """Task 37: Wire up request/response/error into complete()."""

    def test_complete_round_trip(self) -> None:
        """Full round-trip: unified Request → SDK call → unified Response."""
        with patch("unified_llm.adapters.gemini.genai.Client") as mock_cls:
            adapter = GeminiAdapter(api_key="test-key")
            mock_client = mock_cls.return_value

            raw_response = _mock_gemini_response(
                text="Hello from Gemini!",
            )
            # google-genai uses client.aio.models.generate_content for async
            mock_aio = MagicMock()
            mock_models = MagicMock()
            mock_models.generate_content = AsyncMock(return_value=raw_response)
            mock_aio.models = mock_models
            mock_client.aio = mock_aio

            request = Request(
                model="gemini-2.0-flash",
                messages=[Message.user("Hi")],
            )
            response = asyncio.run(adapter.complete(request))

            assert response.text == "Hello from Gemini!"
            assert response.provider == "gemini"
            assert response.finish_reason.reason == "stop"
            mock_models.generate_content.assert_called_once()

    def test_complete_passes_translated_kwargs(self) -> None:
        """complete() passes correctly translated kwargs to SDK."""
        with patch("unified_llm.adapters.gemini.genai.Client") as mock_cls:
            adapter = GeminiAdapter(api_key="test-key")
            mock_client = mock_cls.return_value

            raw_response = _mock_gemini_response()
            mock_aio = MagicMock()
            mock_models = MagicMock()
            mock_models.generate_content = AsyncMock(return_value=raw_response)
            mock_aio.models = mock_models
            mock_client.aio = mock_aio

            request = Request(
                model="gemini-2.0-flash",
                messages=[
                    Message.system("Be helpful"),
                    Message.user("Hello"),
                ],
                temperature=0.5,
                max_tokens=1024,
            )
            asyncio.run(adapter.complete(request))

            call_kwargs = mock_models.generate_content.call_args[1]
            assert call_kwargs["model"] == "gemini-2.0-flash"
            assert "system_instruction" in call_kwargs
            assert "contents" in call_kwargs
            config = call_kwargs.get("config", {})
            assert config.get("temperature") == 0.5
            assert config.get("max_output_tokens") == 1024

    def test_complete_translates_api_errors(self) -> None:
        """complete() catches SDK exceptions and raises unified errors."""
        with patch("unified_llm.adapters.gemini.genai.Client") as mock_cls:
            adapter = GeminiAdapter(api_key="test-key")
            mock_client = mock_cls.return_value

            mock_aio = MagicMock()
            mock_models = MagicMock()
            mock_models.generate_content = AsyncMock(
                side_effect=_make_genai_client_error(
                    429, status="RESOURCE_EXHAUSTED", message="Rate limited"
                )
            )
            mock_aio.models = mock_models
            mock_client.aio = mock_aio

            request = Request(
                model="gemini-2.0-flash",
                messages=[Message.user("Hi")],
            )
            with pytest.raises(E.RateLimitError):
                asyncio.run(adapter.complete(request))

    def test_complete_with_tool_response(self) -> None:
        """complete() handles tool call responses correctly."""
        with patch("unified_llm.adapters.gemini.genai.Client") as mock_cls:
            adapter = GeminiAdapter(api_key="test-key")
            mock_client = mock_cls.return_value

            raw_response = _mock_gemini_response(
                parts=[
                    SimpleNamespace(
                        text=None,
                        function_call=SimpleNamespace(
                            name="get_weather", args={"city": "SF"}
                        ),
                    ),
                ],
                finish_reason="STOP",
            )
            mock_aio = MagicMock()
            mock_models = MagicMock()
            mock_models.generate_content = AsyncMock(return_value=raw_response)
            mock_aio.models = mock_models
            mock_client.aio = mock_aio

            request = Request(
                model="gemini-2.0-flash",
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

    def test_complete_generic_exception(self) -> None:
        """complete() handles unexpected exceptions."""
        with patch("unified_llm.adapters.gemini.genai.Client") as mock_cls:
            adapter = GeminiAdapter(api_key="test-key")
            mock_client = mock_cls.return_value

            mock_aio = MagicMock()
            mock_models = MagicMock()
            mock_models.generate_content = AsyncMock(
                side_effect=RuntimeError("Unexpected")
            )
            mock_aio.models = mock_models
            mock_client.aio = mock_aio

            request = Request(
                model="gemini-2.0-flash",
                messages=[Message.user("Hi")],
            )
            with pytest.raises(E.ProviderError):
                asyncio.run(adapter.complete(request))


# ---------------------------------------------------------------------------
# Task 38: Gemini Streaming Translation
# ---------------------------------------------------------------------------


def _make_stream_chunk_text(
    text: str,
    *,
    finish_reason: str | None = None,
    prompt_token_count: int = 0,
    candidates_token_count: int = 0,
    total_token_count: int = 0,
    thoughts_token_count: int | None = None,
) -> SimpleNamespace:
    """Create a mock Gemini streaming chunk with text."""
    usage_kwargs: dict = {
        "prompt_token_count": prompt_token_count,
        "candidates_token_count": candidates_token_count,
        "total_token_count": total_token_count,
    }
    if thoughts_token_count is not None:
        usage_kwargs["thoughts_token_count"] = thoughts_token_count

    candidate_kwargs: dict = {
        "content": SimpleNamespace(
            parts=[SimpleNamespace(text=text, function_call=None)],
            role="model",
        ),
    }
    if finish_reason is not None:
        candidate_kwargs["finish_reason"] = finish_reason
    else:
        candidate_kwargs["finish_reason"] = None

    return SimpleNamespace(
        candidates=[SimpleNamespace(**candidate_kwargs)],
        usage_metadata=SimpleNamespace(**usage_kwargs),
    )


def _make_stream_chunk_function_call(
    name: str,
    args: dict,
    *,
    finish_reason: str | None = None,
) -> SimpleNamespace:
    """Create a mock Gemini streaming chunk with a function call."""
    candidate_kwargs: dict = {
        "content": SimpleNamespace(
            parts=[
                SimpleNamespace(
                    text=None,
                    function_call=SimpleNamespace(name=name, args=args),
                )
            ],
            role="model",
        ),
    }
    if finish_reason is not None:
        candidate_kwargs["finish_reason"] = finish_reason
    else:
        candidate_kwargs["finish_reason"] = None

    return SimpleNamespace(
        candidates=[SimpleNamespace(**candidate_kwargs)],
        usage_metadata=SimpleNamespace(
            prompt_token_count=0,
            candidates_token_count=0,
            total_token_count=0,
        ),
    )


class _MockAsyncStream:
    """Mock async iterator for Gemini streaming."""

    def __init__(self, chunks: list[SimpleNamespace]) -> None:
        self._chunks = chunks
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk


class TestStreamingTranslation:
    """Task 38: Verify Gemini streaming chunks → unified StreamEvent sequence."""

    def _collect_stream(
        self,
        adapter: GeminiAdapter,
        request: Request,
        chunks: list[SimpleNamespace],
    ) -> list:
        """Run streaming and collect all events."""
        from unified_llm.types import StreamEvent as SE

        mock_generate = AsyncMock(return_value=_MockAsyncStream(chunks))
        adapter._client.aio.models.generate_content_stream = mock_generate

        result: list[SE] = []

        async def run():
            async for evt in adapter.stream(request):
                result.append(evt)

        asyncio.run(run())
        return result

    def test_text_stream_event_sequence(self) -> None:
        """Text stream: TEXT_START → TEXT_DELTA*2 → FINISH."""
        adapter = _make_adapter()
        request = Request(
            model="gemini-2.0-flash", messages=[Message.user("Hi")]
        )
        chunks = [
            _make_stream_chunk_text("Hello"),
            _make_stream_chunk_text(
                " world",
                finish_reason="STOP",
                prompt_token_count=10,
                candidates_token_count=5,
                total_token_count=15,
            ),
        ]
        events = self._collect_stream(adapter, request, chunks)

        types = [e.type for e in events]
        assert StreamEventType.TEXT_START in types
        assert types.count(StreamEventType.TEXT_DELTA) == 2
        assert StreamEventType.FINISH in types

    def test_text_deltas_contain_text(self) -> None:
        """TEXT_DELTA events carry the delta text."""
        adapter = _make_adapter()
        request = Request(
            model="gemini-2.0-flash", messages=[Message.user("Hi")]
        )
        chunks = [
            _make_stream_chunk_text("Hello"),
            _make_stream_chunk_text(" world", finish_reason="STOP"),
        ]
        events = self._collect_stream(adapter, request, chunks)

        deltas = [e for e in events if e.type == StreamEventType.TEXT_DELTA]
        assert deltas[0].delta == "Hello"
        assert deltas[1].delta == " world"

    def test_finish_event_has_usage(self) -> None:
        """FINISH event carries usage and finish_reason."""
        adapter = _make_adapter()
        request = Request(
            model="gemini-2.0-flash", messages=[Message.user("Hi")]
        )
        chunks = [
            _make_stream_chunk_text(
                "Hello",
                finish_reason="STOP",
                prompt_token_count=10,
                candidates_token_count=5,
                total_token_count=15,
            ),
        ]
        events = self._collect_stream(adapter, request, chunks)

        finish = [e for e in events if e.type == StreamEventType.FINISH][0]
        assert finish.finish_reason is not None
        assert finish.finish_reason.reason == "stop"
        assert finish.usage is not None
        assert finish.usage.input_tokens == 10
        assert finish.usage.output_tokens == 5

    def test_function_call_stream_events(self) -> None:
        """Function call: TOOL_CALL_START → TOOL_CALL_END (complete in one chunk)."""
        adapter = _make_adapter()
        request = Request(
            model="gemini-2.0-flash", messages=[Message.user("Hi")]
        )
        chunks = [
            _make_stream_chunk_function_call(
                "get_weather", {"city": "SF"}, finish_reason="STOP"
            ),
        ]
        events = self._collect_stream(adapter, request, chunks)

        types = [e.type for e in events]
        assert StreamEventType.TOOL_CALL_START in types
        assert StreamEventType.TOOL_CALL_END in types
        assert StreamEventType.FINISH in types

    def test_tool_call_end_has_parsed_args(self) -> None:
        """TOOL_CALL_END carries the complete parsed tool call."""
        adapter = _make_adapter()
        request = Request(
            model="gemini-2.0-flash", messages=[Message.user("Hi")]
        )
        chunks = [
            _make_stream_chunk_function_call(
                "get_weather", {"city": "SF"}, finish_reason="STOP"
            ),
        ]
        events = self._collect_stream(adapter, request, chunks)

        end_evt = [e for e in events if e.type == StreamEventType.TOOL_CALL_END][0]
        assert end_evt.tool_call is not None
        assert end_evt.tool_call.name == "get_weather"
        assert end_evt.tool_call.arguments == {"city": "SF"}
        assert end_evt.tool_call.id.startswith("call_")

    def test_tool_call_finish_reason(self) -> None:
        """Tool call stream: FINISH has reason='tool_calls'."""
        adapter = _make_adapter()
        request = Request(
            model="gemini-2.0-flash", messages=[Message.user("Hi")]
        )
        chunks = [
            _make_stream_chunk_function_call(
                "search", {"q": "test"}, finish_reason="STOP"
            ),
        ]
        events = self._collect_stream(adapter, request, chunks)

        finish = [e for e in events if e.type == StreamEventType.FINISH][0]
        assert finish.finish_reason is not None
        assert finish.finish_reason.reason == "tool_calls"

    def test_stream_error_translated(self) -> None:
        """Stream errors are caught and translated to unified errors."""
        adapter = _make_adapter()
        request = Request(
            model="gemini-2.0-flash", messages=[Message.user("Hi")]
        )

        exc = _make_genai_server_error(500, message="Server error")
        adapter._client.aio.models.generate_content_stream = AsyncMock(
            side_effect=exc
        )

        with pytest.raises(E.ServerError):

            async def run():
                async for _ in adapter.stream(request):
                    pass

            asyncio.run(run())

    def test_text_end_emitted(self) -> None:
        """TEXT_END is emitted when finish_reason is present after text."""
        adapter = _make_adapter()
        request = Request(
            model="gemini-2.0-flash", messages=[Message.user("Hi")]
        )
        chunks = [
            _make_stream_chunk_text("Hello"),
            _make_stream_chunk_text(" world", finish_reason="STOP"),
        ]
        events = self._collect_stream(adapter, request, chunks)

        types = [e.type for e in events]
        assert StreamEventType.TEXT_END in types

    def test_finish_has_response_object(self) -> None:
        """FINISH event carries a Response object."""
        adapter = _make_adapter()
        request = Request(
            model="gemini-2.0-flash", messages=[Message.user("Hi")]
        )
        chunks = [
            _make_stream_chunk_text("Hi", finish_reason="STOP"),
        ]
        events = self._collect_stream(adapter, request, chunks)

        finish = [e for e in events if e.type == StreamEventType.FINISH][0]
        assert finish.response is not None
        assert finish.response.provider == "gemini"
        assert finish.response.model == "gemini-2.0-flash"
