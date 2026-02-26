"""Retry system with exponential backoff and jitter (Spec §6.6)."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from unified_llm.errors import SDKError

T = TypeVar("T")


@dataclass
class RetryPolicy:
    """Retry configuration (Spec §6.6)."""

    max_retries: int = 2
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    on_retry: Callable[..., Any] | None = None

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for attempt n (0-indexed). Spec §6.6 formula."""
        delay = min(
            self.base_delay * (self.backoff_multiplier**attempt), self.max_delay
        )
        if self.jitter:
            delay *= random.uniform(0.5, 1.5)
        return delay


async def retry(
    fn: Callable[..., Awaitable[T]],
    policy: RetryPolicy,
    *args: Any,
    **kwargs: Any,
) -> T:
    """Execute fn with automatic retry on retryable errors.

    Spec §6.6: Retries apply to individual calls. Only retryable errors trigger retry.
    Retry-After header is respected: if within max_delay, use it; if exceeds, raise.
    """
    last_error: SDKError | None = None

    for attempt in range(policy.max_retries + 1):
        try:
            return await fn(*args, **kwargs)
        except SDKError as err:
            last_error = err

            # Non-retryable: raise immediately
            if not err.retryable:
                raise

            # Last attempt: raise
            if attempt >= policy.max_retries:
                raise

            # Check Retry-After (on ProviderError)
            retry_after = getattr(err, "retry_after", None)
            if retry_after is not None and retry_after > policy.max_delay:
                raise

            # Calculate delay
            if retry_after is not None:
                delay = retry_after
            else:
                delay = policy.calculate_delay(attempt)

            # on_retry callback
            if policy.on_retry is not None:
                callback_result = policy.on_retry(err, attempt, delay)
                if asyncio.iscoroutine(callback_result):
                    await callback_result

            await asyncio.sleep(delay)

    # Should not reach here, but satisfy type checker
    assert last_error is not None
    raise last_error
