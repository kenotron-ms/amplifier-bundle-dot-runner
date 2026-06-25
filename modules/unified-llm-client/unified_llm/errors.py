"""Error hierarchy for the unified LLM client (Spec §6.1-6.6).

13 error types: SDKError base, ProviderError with 9 subtypes,
plus RequestTimeoutError, AbortError, NetworkError, StreamError,
InvalidToolCallError, NoObjectGeneratedError, ConfigurationError.

Names deliberately avoid shadowing Python built-ins.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class SDKError(Exception):
    """Base exception for all unified-llm-client errors."""

    def __init__(self, message: str, *, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.cause = cause
        if cause:
            self.__cause__ = cause

    @property
    def retryable(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Provider errors (Spec §6.2)
# ---------------------------------------------------------------------------


class ProviderError(SDKError):
    """Error from an LLM provider."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        status_code: int | None = None,
        error_code: str | None = None,
        retryable: bool = False,
        retry_after: float | None = None,
        raw: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message, cause=cause)
        self.provider = provider
        self.status_code = status_code
        self.error_code = error_code
        self._retryable = retryable
        self.retry_after = retry_after
        self.raw = raw

    @property
    def retryable(self) -> bool:
        return self._retryable


# -- Non-retryable provider errors --


class AuthenticationError(ProviderError):
    """401 — Invalid API key or expired token."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("retryable", False)
        super().__init__(**kwargs)


class AccessDeniedError(ProviderError):
    """403 — Insufficient permissions."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("retryable", False)
        super().__init__(**kwargs)


class NotFoundError(ProviderError):
    """404 — Model or endpoint not found."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("retryable", False)
        super().__init__(**kwargs)


class InvalidRequestError(ProviderError):
    """400/422 — Malformed request or invalid parameters."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("retryable", False)
        super().__init__(**kwargs)


class ContentFilterError(ProviderError):
    """Response blocked by safety/content filter."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("retryable", False)
        super().__init__(**kwargs)


class ContextLengthError(ProviderError):
    """413 — Input + output exceeds context window."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("retryable", False)
        super().__init__(**kwargs)


class QuotaExceededError(ProviderError):
    """Billing/usage quota exhausted."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("retryable", False)
        super().__init__(**kwargs)


# -- Retryable provider errors --


class RateLimitError(ProviderError):
    """429 — Rate limit exceeded."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("retryable", True)
        super().__init__(**kwargs)


class ServerError(ProviderError):
    """500-504 — Provider internal error."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("retryable", True)
        super().__init__(**kwargs)


# ---------------------------------------------------------------------------
# Non-provider errors
# ---------------------------------------------------------------------------


class RequestTimeoutError(SDKError):
    """Request or stream timed out. Retryable."""

    @property
    def retryable(self) -> bool:
        return True


class AbortError(SDKError):
    """Request cancelled via abort signal. Not retryable."""

    @property
    def retryable(self) -> bool:
        return False


class NetworkError(SDKError):
    """Network-level failure. Retryable."""

    @property
    def retryable(self) -> bool:
        return True


class StreamError(SDKError):
    """Error during stream consumption. Retryable."""

    @property
    def retryable(self) -> bool:
        return True


class StreamProtocolError(StreamError):
    """Stream protocol violation during consumption."""


class InvalidToolCallError(SDKError):
    """Tool call arguments failed validation. Not retryable."""

    @property
    def retryable(self) -> bool:
        return False


class NoObjectGeneratedError(SDKError):
    """Structured output parsing/validation failed. Not retryable."""

    @property
    def retryable(self) -> bool:
        return False


class ConfigurationError(SDKError):
    """SDK misconfiguration (missing provider, etc.). Not retryable."""

    @property
    def retryable(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# HTTP status → error type factory (Spec §6.4)
# ---------------------------------------------------------------------------

_STATUS_MAP: dict[int, type[ProviderError]] = {
    400: InvalidRequestError,
    401: AuthenticationError,
    403: AccessDeniedError,
    404: NotFoundError,
    408: RequestTimeoutError,  # type: ignore[dict-item]  # handled specially below
    413: ContextLengthError,
    422: InvalidRequestError,
    429: RateLimitError,
    500: ServerError,
    502: ServerError,
    503: ServerError,
    504: ServerError,
}

# ---------------------------------------------------------------------------
# ULM-9: Error message-body classification (Spec §6.5)
# ---------------------------------------------------------------------------

# Conservative phrase sets — ordered most-specific first within each group.
# Keep these narrow: false positives are worse than false negatives.
_QUOTA_PHRASES: tuple[str, ...] = (
    "insufficient_quota",
    "quota exceeded",
    "billing",
    "rate limit",
    "quota",
)
_CONTEXT_PHRASES: tuple[str, ...] = (
    "context length",
    "maximum context",
    "too many tokens",
    "prompt is too long",
)
_CONTENT_FILTER_PHRASES: tuple[str, ...] = (
    "content filter",
    "content_policy",
    "safety",
    "blocked",
)


def _classify_by_message(message: str) -> type[ProviderError] | None:
    """Return a more-specific error class when the message body contains
    known classification phrases (Spec §6.5).

    Applied as a secondary signal after status-code mapping.  Only promotes
    from generic types; never demotes a specific type already determined by
    the status code.
    """
    msg_lower = message.lower()
    for phrase in _QUOTA_PHRASES:
        if phrase in msg_lower:
            return QuotaExceededError
    for phrase in _CONTEXT_PHRASES:
        if phrase in msg_lower:
            return ContextLengthError
    for phrase in _CONTENT_FILTER_PHRASES:
        if phrase in msg_lower:
            return ContentFilterError
    return None


def error_from_status_code(
    *,
    status_code: int,
    message: str,
    provider: str,
    error_code: str | None = None,
    raw: dict[str, Any] | None = None,
    retry_after: float | None = None,
    cause: Exception | None = None,
) -> SDKError:
    """Map an HTTP status code to the appropriate error type (Spec §6.4).

    After status-code mapping, applies message-body classification (Spec §6.5)
    to promote generic errors (InvalidRequestError, unknown status) to more
    specific types when phrase signals are present.
    """
    if status_code == 408:
        return RequestTimeoutError(message, cause=cause)

    # Some providers (e.g. OpenAI Responses API) return model-not-found
    # as a 400 with error_code="model_not_found". Promote to NotFoundError.
    if error_code == "model_not_found":
        return NotFoundError(
            message=message,
            provider=provider,
            status_code=status_code,
            error_code=error_code,
            raw=raw,
            retry_after=retry_after,
            cause=cause,
        )

    cls: type[ProviderError] | None = _STATUS_MAP.get(status_code)

    # ULM-9: message-body classification — promote when status code gives a
    # generic type (InvalidRequestError) or is unknown (cls is None).
    # Do NOT override specific types already determined by the status code
    # (e.g. AuthenticationError, RateLimitError, ContextLengthError).
    _GENERIC_TYPES: tuple[type[ProviderError], ...] = (InvalidRequestError,)
    if cls is None or cls in _GENERIC_TYPES:
        msg_cls = _classify_by_message(message)
        if msg_cls is not None:
            cls = msg_cls

    if cls is None:
        # Unknown status codes default to retryable (Spec §6.3)
        return ProviderError(
            message=message,
            provider=provider,
            status_code=status_code,
            error_code=error_code,
            retryable=True,
            raw=raw,
            retry_after=retry_after,
            cause=cause,
        )
    return cls(
        message=message,
        provider=provider,
        status_code=status_code,
        error_code=error_code,
        raw=raw,
        retry_after=retry_after,
        cause=cause,
    )
