"""Tests for unified_llm public API exports (Task 47).

Verifies that all public symbols are importable from the top-level package.
Users should be able to: from unified_llm import Client, generate, stream, Message, ...
"""

from __future__ import annotations


class TestCoreClientExports:
    """Core client symbols importable from unified_llm."""

    def test_client_importable(self) -> None:
        from unified_llm import Client

        assert Client is not None

    def test_set_default_client_importable(self) -> None:
        from unified_llm import set_default_client

        assert callable(set_default_client)

    def test_get_default_client_importable(self) -> None:
        from unified_llm import get_default_client

        assert callable(get_default_client)


class TestHighLevelAPIExports:
    """High-level API functions importable from unified_llm."""

    def test_generate_importable(self) -> None:
        from unified_llm import generate

        assert callable(generate)

    def test_stream_importable(self) -> None:
        from unified_llm import stream

        assert callable(stream)

    def test_generate_object_importable(self) -> None:
        from unified_llm import generate_object

        assert callable(generate_object)

    def test_stream_object_importable(self) -> None:
        from unified_llm import stream_object

        assert callable(stream_object)


class TestTypeExports:
    """All types importable from unified_llm."""

    def test_role(self) -> None:
        from unified_llm import Role

        assert Role.USER.value == "user"

    def test_content_kind(self) -> None:
        from unified_llm import ContentKind

        assert ContentKind.TEXT.value == "text"

    def test_content_part(self) -> None:
        from unified_llm import ContentPart

        assert ContentPart is not None

    def test_message(self) -> None:
        from unified_llm import Message

        assert Message is not None

    def test_image_data(self) -> None:
        from unified_llm import ImageData

        assert ImageData is not None

    def test_audio_data(self) -> None:
        from unified_llm import AudioData

        assert AudioData is not None

    def test_document_data(self) -> None:
        from unified_llm import DocumentData

        assert DocumentData is not None

    def test_tool_call_data(self) -> None:
        from unified_llm import ToolCallData

        assert ToolCallData is not None

    def test_tool_result_data(self) -> None:
        from unified_llm import ToolResultData

        assert ToolResultData is not None

    def test_thinking_data(self) -> None:
        from unified_llm import ThinkingData

        assert ThinkingData is not None

    def test_request(self) -> None:
        from unified_llm import Request

        assert Request is not None

    def test_response(self) -> None:
        from unified_llm import Response

        assert Response is not None

    def test_response_format(self) -> None:
        from unified_llm import ResponseFormat

        assert ResponseFormat is not None

    def test_finish_reason(self) -> None:
        from unified_llm import FinishReason

        assert FinishReason is not None

    def test_usage(self) -> None:
        from unified_llm import Usage

        assert Usage is not None

    def test_warning(self) -> None:
        from unified_llm import Warning

        assert Warning is not None

    def test_rate_limit_info(self) -> None:
        from unified_llm import RateLimitInfo

        assert RateLimitInfo is not None

    def test_generate_result(self) -> None:
        from unified_llm import GenerateResult

        assert GenerateResult is not None

    def test_step_result(self) -> None:
        from unified_llm import StepResult

        assert StepResult is not None

    def test_stream_event(self) -> None:
        from unified_llm import StreamEvent

        assert StreamEvent is not None

    def test_stream_event_type(self) -> None:
        from unified_llm import StreamEventType

        assert StreamEventType is not None

    def test_stream_accumulator(self) -> None:
        from unified_llm import StreamAccumulator

        assert StreamAccumulator is not None

    def test_tool(self) -> None:
        from unified_llm import Tool

        assert Tool is not None

    def test_tool_choice(self) -> None:
        from unified_llm import ToolChoice

        assert ToolChoice is not None

    def test_tool_call(self) -> None:
        from unified_llm import ToolCall

        assert ToolCall is not None

    def test_tool_result(self) -> None:
        from unified_llm import ToolResult

        assert ToolResult is not None

    def test_timeout_config(self) -> None:
        from unified_llm import TimeoutConfig

        assert TimeoutConfig is not None

    def test_adapter_timeout(self) -> None:
        from unified_llm import AdapterTimeout

        assert AdapterTimeout is not None

    def test_model_info(self) -> None:
        from unified_llm import ModelInfo

        assert ModelInfo is not None


class TestErrorExports:
    """All error types importable from unified_llm."""

    def test_sdk_error(self) -> None:
        from unified_llm import SDKError

        assert issubclass(SDKError, Exception)

    def test_provider_error(self) -> None:
        from unified_llm import ProviderError

        assert issubclass(ProviderError, Exception)

    def test_authentication_error(self) -> None:
        from unified_llm import AuthenticationError

        assert AuthenticationError is not None

    def test_access_denied_error(self) -> None:
        from unified_llm import AccessDeniedError

        assert AccessDeniedError is not None

    def test_not_found_error(self) -> None:
        from unified_llm import NotFoundError

        assert NotFoundError is not None

    def test_invalid_request_error(self) -> None:
        from unified_llm import InvalidRequestError

        assert InvalidRequestError is not None

    def test_rate_limit_error(self) -> None:
        from unified_llm import RateLimitError

        assert RateLimitError is not None

    def test_server_error(self) -> None:
        from unified_llm import ServerError

        assert ServerError is not None

    def test_content_filter_error(self) -> None:
        from unified_llm import ContentFilterError

        assert ContentFilterError is not None

    def test_context_length_error(self) -> None:
        from unified_llm import ContextLengthError

        assert ContextLengthError is not None

    def test_quota_exceeded_error(self) -> None:
        from unified_llm import QuotaExceededError

        assert QuotaExceededError is not None

    def test_request_timeout_error(self) -> None:
        from unified_llm import RequestTimeoutError

        assert RequestTimeoutError is not None

    def test_abort_error(self) -> None:
        from unified_llm import AbortError

        assert AbortError is not None

    def test_network_error(self) -> None:
        from unified_llm import NetworkError

        assert NetworkError is not None

    def test_stream_error(self) -> None:
        from unified_llm import StreamError

        assert StreamError is not None

    def test_invalid_tool_call_error(self) -> None:
        from unified_llm import InvalidToolCallError

        assert InvalidToolCallError is not None

    def test_no_object_generated_error(self) -> None:
        from unified_llm import NoObjectGeneratedError

        assert NoObjectGeneratedError is not None

    def test_configuration_error(self) -> None:
        from unified_llm import ConfigurationError

        assert ConfigurationError is not None


class TestRetryExports:
    """Retry policy importable from unified_llm."""

    def test_retry_policy(self) -> None:
        from unified_llm import RetryPolicy

        assert RetryPolicy is not None


class TestCatalogExports:
    """Catalog functions importable from unified_llm."""

    def test_get_model_info(self) -> None:
        from unified_llm import get_model_info

        assert callable(get_model_info)

    def test_list_models(self) -> None:
        from unified_llm import list_models

        assert callable(list_models)

    def test_get_latest_model(self) -> None:
        from unified_llm import get_latest_model

        assert callable(get_latest_model)


class TestAdapterExports:
    """Adapter interface importable from unified_llm."""

    def test_provider_adapter(self) -> None:
        from unified_llm import ProviderAdapter

        assert ProviderAdapter is not None


class TestAllList:
    """__all__ is defined and complete."""

    def test_all_defined(self) -> None:
        import unified_llm

        assert hasattr(unified_llm, "__all__")
        assert isinstance(unified_llm.__all__, list)
        assert len(unified_llm.__all__) > 40  # Should have 40+ public symbols

    def test_all_symbols_importable(self) -> None:
        """Every symbol in __all__ is actually importable."""
        import unified_llm

        for name in unified_llm.__all__:
            assert hasattr(unified_llm, name), f"__all__ lists '{name}' but it's not importable"
