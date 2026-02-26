"""DoD §8.8 — Error Handling & Retry.

Verifies error hierarchy, retryable flags, exponential backoff, Retry-After,
and retry behavior for different error types.
Uses mocked adapters — no real API keys needed.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from unified_llm import (
    AbortError,
    AccessDeniedError,
    AuthenticationError,
    Client,
    ConfigurationError,
    ContentFilterError,
    ContextLengthError,
    FinishReason,
    InvalidRequestError,
    Message,
    NetworkError,
    NotFoundError,
    ProviderError,
    QuotaExceededError,
    RateLimitError,
    Request,
    RequestTimeoutError,
    Response,
    ServerError,
    StreamError,
    StreamEvent,
    StreamEventType,
    Usage,
)
from unified_llm.errors import error_from_status_code
from unified_llm.retry import RetryPolicy, retry


# ---------------------------------------------------------------------------
# §8.8 — Error hierarchy: correct HTTP status codes
# ---------------------------------------------------------------------------


class TestErrorHierarchy:
    """All errors raised for correct HTTP status codes (Section 6.4 table)."""

    def test_400_invalid_request(self) -> None:
        err = error_from_status_code(
            status_code=400, message="Bad request", provider="test"
        )
        assert isinstance(err, InvalidRequestError)

    def test_401_authentication(self) -> None:
        err = error_from_status_code(
            status_code=401, message="Unauthorized", provider="test"
        )
        assert isinstance(err, AuthenticationError)

    def test_403_access_denied(self) -> None:
        err = error_from_status_code(
            status_code=403, message="Forbidden", provider="test"
        )
        assert isinstance(err, AccessDeniedError)

    def test_404_not_found(self) -> None:
        err = error_from_status_code(
            status_code=404, message="Not found", provider="test"
        )
        assert isinstance(err, NotFoundError)

    def test_408_timeout(self) -> None:
        err = error_from_status_code(
            status_code=408, message="Timeout", provider="test"
        )
        assert isinstance(err, RequestTimeoutError)

    def test_413_context_length(self) -> None:
        err = error_from_status_code(
            status_code=413, message="Too large", provider="test"
        )
        assert isinstance(err, ContextLengthError)

    def test_422_invalid_request(self) -> None:
        err = error_from_status_code(
            status_code=422, message="Unprocessable", provider="test"
        )
        assert isinstance(err, InvalidRequestError)

    def test_429_rate_limit(self) -> None:
        err = error_from_status_code(
            status_code=429, message="Rate limited", provider="test"
        )
        assert isinstance(err, RateLimitError)

    def test_500_server_error(self) -> None:
        err = error_from_status_code(
            status_code=500, message="Internal error", provider="test"
        )
        assert isinstance(err, ServerError)

    def test_502_server_error(self) -> None:
        err = error_from_status_code(
            status_code=502, message="Bad gateway", provider="test"
        )
        assert isinstance(err, ServerError)

    def test_503_server_error(self) -> None:
        err = error_from_status_code(
            status_code=503, message="Unavailable", provider="test"
        )
        assert isinstance(err, ServerError)

    def test_504_server_error(self) -> None:
        err = error_from_status_code(
            status_code=504, message="Gateway timeout", provider="test"
        )
        assert isinstance(err, ServerError)

    def test_unknown_status_retryable(self) -> None:
        """Unknown status codes default to retryable (Spec §6.3)."""
        err = error_from_status_code(
            status_code=599, message="Unknown", provider="test"
        )
        assert isinstance(err, ProviderError)
        assert err.retryable is True


# ---------------------------------------------------------------------------
# §8.8 — Retryable flag
# ---------------------------------------------------------------------------


class TestRetryableFlag:
    """retryable flag is set correctly on each error type."""

    def test_non_retryable_errors(self) -> None:
        """401, 403, 404, 400/422 are NOT retryable."""
        for cls in [
            AuthenticationError,
            AccessDeniedError,
            NotFoundError,
            InvalidRequestError,
            ContentFilterError,
            ContextLengthError,
            QuotaExceededError,
        ]:
            err = cls(message="test", provider="test")
            assert err.retryable is False, f"{cls.__name__} should not be retryable"

    def test_retryable_errors(self) -> None:
        """429, 500-504 ARE retryable."""
        for cls in [RateLimitError, ServerError]:
            err = cls(message="test", provider="test")
            assert err.retryable is True, f"{cls.__name__} should be retryable"

    def test_non_provider_retryable(self) -> None:
        """NetworkError, StreamError, RequestTimeoutError are retryable."""
        assert NetworkError("test").retryable is True
        assert StreamError("test").retryable is True
        assert RequestTimeoutError("test").retryable is True

    def test_non_provider_non_retryable(self) -> None:
        """AbortError, ConfigurationError are NOT retryable."""
        assert AbortError("test").retryable is False
        assert ConfigurationError("test").retryable is False


# ---------------------------------------------------------------------------
# §8.8 — Exponential backoff with jitter
# ---------------------------------------------------------------------------


class TestExponentialBackoff:
    """Exponential backoff with jitter works."""

    def test_delays_increase(self) -> None:
        """[ ] Delays increase correctly per attempt."""
        policy = RetryPolicy(
            base_delay=1.0, backoff_multiplier=2.0, max_delay=60.0, jitter=False
        )
        d0 = policy.calculate_delay(0)
        d1 = policy.calculate_delay(1)
        d2 = policy.calculate_delay(2)
        assert d0 == 1.0
        assert d1 == 2.0
        assert d2 == 4.0

    def test_max_delay_capped(self) -> None:
        """Delays are capped at max_delay."""
        policy = RetryPolicy(
            base_delay=1.0, backoff_multiplier=2.0, max_delay=5.0, jitter=False
        )
        d10 = policy.calculate_delay(10)
        assert d10 == 5.0

    def test_jitter_applied(self) -> None:
        """With jitter, delays vary."""
        policy = RetryPolicy(
            base_delay=1.0, backoff_multiplier=2.0, max_delay=60.0, jitter=True
        )
        delays = {policy.calculate_delay(0) for _ in range(100)}
        # Should have variation (extremely unlikely all 100 are identical)
        assert len(delays) > 1


# ---------------------------------------------------------------------------
# §8.8 — Retry-After header
# ---------------------------------------------------------------------------


class TestRetryAfterHeader:
    """Retry-After header overrides calculated backoff."""

    @pytest.mark.asyncio(loop_scope="function")
    async def test_retry_after_respected(self) -> None:
        """[ ] Retry-After header overrides calculated backoff when present."""
        call_count = 0

        async def failing_fn(request: Request) -> Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RateLimitError(
                    message="Rate limited",
                    provider="test",
                    retry_after=0.01,  # Very short for testing
                )
            return Response(
                id="r1",
                model="test",
                provider="test",
                message=Message.assistant("ok"),
                finish_reason=FinishReason(reason="stop"),
                usage=Usage(input_tokens=5, output_tokens=2, total_tokens=7),
            )

        policy = RetryPolicy(max_retries=2, base_delay=0.01)
        request = Request(model="test", messages=[Message.user("hi")])
        result = await retry(failing_fn, policy, request)
        assert result.text == "ok"
        assert call_count == 2

    @pytest.mark.asyncio(loop_scope="function")
    async def test_retry_after_exceeds_max_delay_raises(self) -> None:
        """[ ] Retry-After exceeding max_delay raises immediately."""

        async def failing_fn(request: Request) -> Response:
            raise RateLimitError(
                message="Rate limited",
                provider="test",
                retry_after=999.0,  # Way above max_delay
            )

        policy = RetryPolicy(max_retries=3, max_delay=60.0)
        request = Request(model="test", messages=[Message.user("hi")])
        with pytest.raises(RateLimitError):
            await retry(failing_fn, policy, request)


# ---------------------------------------------------------------------------
# §8.8 — max_retries = 0 disables retries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="function")
async def test_max_retries_zero() -> None:
    """[ ] max_retries = 0 disables automatic retries."""
    call_count = 0

    async def always_fail(request: Request) -> Response:
        nonlocal call_count
        call_count += 1
        raise ServerError(message="Server error", provider="test")

    policy = RetryPolicy(max_retries=0)
    request = Request(model="test", messages=[Message.user("hi")])
    with pytest.raises(ServerError):
        await retry(always_fail, policy, request)
    assert call_count == 1


# ---------------------------------------------------------------------------
# §8.8 — Rate limit errors retried transparently
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="function")
async def test_rate_limit_retried() -> None:
    """[ ] Rate limit errors (429) are retried transparently."""
    call_count = 0

    async def fail_then_succeed(request: Request) -> Response:
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise RateLimitError(message="Rate limited", provider="test")
        return Response(
            id="r1",
            model="test",
            provider="test",
            message=Message.assistant("ok"),
            finish_reason=FinishReason(reason="stop"),
            usage=Usage(input_tokens=5, output_tokens=2, total_tokens=7),
        )

    policy = RetryPolicy(max_retries=3, base_delay=0.01)
    request = Request(model="test", messages=[Message.user("hi")])
    result = await retry(fail_then_succeed, policy, request)
    assert result.text == "ok"
    assert call_count == 3


# ---------------------------------------------------------------------------
# §8.8 — Non-retryable errors raised immediately
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="function")
async def test_non_retryable_raised_immediately() -> None:
    """[ ] Non-retryable errors (401, 403, 404) are raised immediately without retry."""
    call_count = 0

    async def auth_fail(request: Request) -> Response:
        nonlocal call_count
        call_count += 1
        raise AuthenticationError(message="Bad key", provider="test")

    policy = RetryPolicy(max_retries=3, base_delay=0.01)
    request = Request(model="test", messages=[Message.user("hi")])
    with pytest.raises(AuthenticationError):
        await retry(auth_fail, policy, request)
    assert call_count == 1  # Only called once — no retry


# ---------------------------------------------------------------------------
# §8.8 — Retries apply per-step
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="function")
async def test_retries_per_step() -> None:
    """[ ] Retries apply per-step, not to the entire multi-step operation."""
    from unified_llm.generate import generate

    step_call_count = 0

    class _RetryAdapter:
        @property
        def name(self) -> str:
            return "mock"

        async def complete(self, request: Request) -> Response:
            nonlocal step_call_count
            step_call_count += 1
            # Fail on first call of each step, succeed on retry
            if step_call_count % 2 == 1:
                raise ServerError(message="Transient", provider="mock")
            return Response(
                id="r1",
                model="test",
                provider="mock",
                message=Message.assistant("ok"),
                finish_reason=FinishReason(reason="stop"),
                usage=Usage(input_tokens=5, output_tokens=2, total_tokens=7),
            )

        async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
            yield StreamEvent(type=StreamEventType.FINISH)

        async def close(self) -> None:
            pass

        async def initialize(self) -> None:
            pass

        def supports_tool_choice(self, mode: str) -> bool:
            return True

    client = Client(providers={"mock": _RetryAdapter()}, default_provider="mock")
    result = await generate(
        model="test",
        prompt="Hi",
        client=client,
        provider="mock",
        max_retries=2,
    )
    assert result.text == "ok"
    # Should have needed retries per step
    assert step_call_count >= 2
