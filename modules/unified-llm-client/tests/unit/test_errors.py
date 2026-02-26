"""Tests for unified_llm.errors — full 13-type error hierarchy."""

import unified_llm.errors as E


class TestErrorHierarchy:
    """Spec §6.1 — All errors inherit from SDKError."""

    def test_sdk_error_is_base(self) -> None:
        assert issubclass(E.ProviderError, E.SDKError)
        assert issubclass(E.NetworkError, E.SDKError)
        assert issubclass(E.StreamError, E.SDKError)
        assert issubclass(E.ConfigurationError, E.SDKError)

    def test_provider_error_subtypes(self) -> None:
        subtypes = [
            E.AuthenticationError, E.AccessDeniedError, E.NotFoundError,
            E.InvalidRequestError, E.RateLimitError, E.ServerError,
            E.ContentFilterError, E.ContextLengthError, E.QuotaExceededError,
        ]
        for cls in subtypes:
            assert issubclass(cls, E.ProviderError), f"{cls.__name__} not subclass of ProviderError"

    def test_non_provider_errors(self) -> None:
        non_provider = [
            E.RequestTimeoutError, E.AbortError, E.NetworkError,
            E.StreamError, E.InvalidToolCallError, E.NoObjectGeneratedError,
            E.ConfigurationError,
        ]
        for cls in non_provider:
            assert issubclass(cls, E.SDKError)
            assert not issubclass(cls, E.ProviderError), f"{cls.__name__} should NOT be ProviderError"


class TestSDKError:
    """SDKError base has message and cause."""

    def test_message(self) -> None:
        err = E.SDKError("something broke")
        assert str(err) == "something broke"
        assert err.message == "something broke"

    def test_cause(self) -> None:
        original = ValueError("bad value")
        err = E.SDKError("wrapped", cause=original)
        assert err.cause is original


class TestProviderError:
    """Spec §6.2 — ProviderError has extra fields."""

    def test_fields(self) -> None:
        err = E.RateLimitError(
            message="Rate limited",
            provider="anthropic",
            status_code=429,
            error_code="rate_limit_exceeded",
            retryable=True,
            retry_after=30.0,
            raw={"type": "error", "message": "Rate limited"},
        )
        assert err.provider == "anthropic"
        assert err.status_code == 429
        assert err.error_code == "rate_limit_exceeded"
        assert err.retryable is True
        assert err.retry_after == 30.0
        assert err.raw is not None


class TestRetryability:
    """Spec §6.3 — Retryability classification."""

    def test_non_retryable_errors(self) -> None:
        non_retryable = [
            E.AuthenticationError(message="bad key", provider="openai"),
            E.AccessDeniedError(message="forbidden", provider="openai"),
            E.NotFoundError(message="no model", provider="openai"),
            E.InvalidRequestError(message="bad params", provider="openai"),
            E.ContextLengthError(message="too long", provider="openai"),
            E.QuotaExceededError(message="over quota", provider="openai"),
            E.ContentFilterError(message="blocked", provider="openai"),
            E.ConfigurationError("no provider"),
        ]
        for err in non_retryable:
            assert not err.retryable, f"{type(err).__name__} should not be retryable"

    def test_retryable_errors(self) -> None:
        retryable = [
            E.RateLimitError(message="429", provider="openai", status_code=429),
            E.ServerError(message="500", provider="openai", status_code=500),
            E.RequestTimeoutError("timed out"),
            E.NetworkError("connection refused"),
            E.StreamError("stream broke"),
        ]
        for err in retryable:
            assert err.retryable, f"{type(err).__name__} should be retryable"


class TestHTTPStatusMapping:
    """Spec §6.4 — HTTP status code to error type mapping."""

    def test_status_code_mapping(self) -> None:
        mapping = {
            400: E.InvalidRequestError,
            401: E.AuthenticationError,
            403: E.AccessDeniedError,
            404: E.NotFoundError,
            408: E.RequestTimeoutError,
            413: E.ContextLengthError,
            422: E.InvalidRequestError,
            429: E.RateLimitError,
            500: E.ServerError,
            502: E.ServerError,
            503: E.ServerError,
            504: E.ServerError,
        }
        for status, expected_cls in mapping.items():
            err = E.error_from_status_code(
                status_code=status, message="test", provider="test",
            )
            assert isinstance(err, expected_cls), (
                f"Status {status} should produce {expected_cls.__name__}, "
                f"got {type(err).__name__}"
            )

    def test_unknown_status_defaults_retryable(self) -> None:
        """Spec §6.3: Unknown errors default to retryable."""
        err = E.error_from_status_code(
            status_code=999, message="unknown", provider="test",
        )
        assert isinstance(err, E.ProviderError)
        assert err.retryable is True
