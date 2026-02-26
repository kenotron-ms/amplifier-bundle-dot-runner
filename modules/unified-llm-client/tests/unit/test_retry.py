"""Tests for unified_llm.retry — exponential backoff with jitter."""

import asyncio
from unittest.mock import AsyncMock, patch

from unified_llm.errors import (
    AuthenticationError,
    RateLimitError,
    ServerError,
)
from unified_llm.retry import RetryPolicy, retry


class TestRetryPolicy:
    """Spec §6.6 — RetryPolicy record with defaults."""

    def test_defaults(self) -> None:
        policy = RetryPolicy()
        assert policy.max_retries == 2
        assert policy.base_delay == 1.0
        assert policy.max_delay == 60.0
        assert policy.backoff_multiplier == 2.0
        assert policy.jitter is True
        assert policy.on_retry is None

    def test_delay_calculation_no_jitter(self) -> None:
        policy = RetryPolicy(base_delay=1.0, backoff_multiplier=2.0, jitter=False)
        assert policy.calculate_delay(0) == 1.0
        assert policy.calculate_delay(1) == 2.0
        assert policy.calculate_delay(2) == 4.0
        assert policy.calculate_delay(3) == 8.0

    def test_delay_capped_at_max(self) -> None:
        policy = RetryPolicy(
            base_delay=1.0, backoff_multiplier=2.0, max_delay=5.0, jitter=False,
        )
        assert policy.calculate_delay(10) == 5.0

    def test_delay_with_jitter_in_range(self) -> None:
        policy = RetryPolicy(base_delay=1.0, jitter=True)
        delays = [policy.calculate_delay(0) for _ in range(100)]
        assert all(0.5 <= d <= 1.5 for d in delays), f"Delays out of range: {min(delays)}-{max(delays)}"


class TestRetryFunction:
    """Spec §6.6 — retry() utility wrapping async callables."""

    def test_success_no_retry(self) -> None:
        mock_fn = AsyncMock(return_value="ok")
        result = asyncio.run(retry(mock_fn, RetryPolicy()))
        assert result == "ok"
        assert mock_fn.call_count == 1

    def test_retries_on_retryable_error(self) -> None:
        mock_fn = AsyncMock(side_effect=[
            ServerError(message="500", provider="test", status_code=500),
            "ok",
        ])
        with patch("unified_llm.retry.asyncio.sleep", new_callable=AsyncMock):
            result = asyncio.run(retry(mock_fn, RetryPolicy(max_retries=2)))
        assert result == "ok"
        assert mock_fn.call_count == 2

    def test_no_retry_on_non_retryable(self) -> None:
        mock_fn = AsyncMock(side_effect=AuthenticationError(
            message="bad key", provider="test", status_code=401,
        ))
        try:
            asyncio.run(retry(mock_fn, RetryPolicy(max_retries=3)))
            assert False, "Should have raised"
        except AuthenticationError:
            pass
        assert mock_fn.call_count == 1

    def test_max_retries_exhausted(self) -> None:
        mock_fn = AsyncMock(side_effect=ServerError(
            message="500", provider="test", status_code=500,
        ))
        with patch("unified_llm.retry.asyncio.sleep", new_callable=AsyncMock):
            try:
                asyncio.run(retry(mock_fn, RetryPolicy(max_retries=2)))
                assert False, "Should have raised"
            except ServerError:
                pass
        assert mock_fn.call_count == 3  # 1 initial + 2 retries

    def test_max_retries_zero_disables(self) -> None:
        mock_fn = AsyncMock(side_effect=ServerError(
            message="500", provider="test", status_code=500,
        ))
        try:
            asyncio.run(retry(mock_fn, RetryPolicy(max_retries=0)))
            assert False, "Should have raised"
        except ServerError:
            pass
        assert mock_fn.call_count == 1

    def test_retry_after_within_max_delay(self) -> None:
        """Spec: If Retry-After < max_delay, use provider's delay."""
        err = RateLimitError(
            message="429", provider="test", status_code=429, retry_after=5.0,
        )
        mock_fn = AsyncMock(side_effect=[err, "ok"])
        sleep_mock = AsyncMock()
        with patch("unified_llm.retry.asyncio.sleep", sleep_mock):
            asyncio.run(retry(mock_fn, RetryPolicy(max_delay=60.0)))
        sleep_mock.assert_called_once()
        actual_delay = sleep_mock.call_args[0][0]
        assert actual_delay == 5.0

    def test_retry_after_exceeds_max_delay_raises(self) -> None:
        """Spec: If Retry-After > max_delay, raise immediately."""
        err = RateLimitError(
            message="429", provider="test", status_code=429, retry_after=120.0,
        )
        mock_fn = AsyncMock(side_effect=err)
        try:
            asyncio.run(retry(mock_fn, RetryPolicy(max_delay=60.0)))
            assert False, "Should have raised"
        except RateLimitError:
            pass
        assert mock_fn.call_count == 1

    def test_on_retry_callback(self) -> None:
        callback = AsyncMock()
        mock_fn = AsyncMock(side_effect=[
            ServerError(message="500", provider="test", status_code=500),
            "ok",
        ])
        with patch("unified_llm.retry.asyncio.sleep", new_callable=AsyncMock):
            asyncio.run(retry(mock_fn, RetryPolicy(on_retry=callback)))
        callback.assert_called_once()
