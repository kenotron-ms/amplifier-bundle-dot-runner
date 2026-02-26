"""DoD §8.4 — Generation.

Verifies generate(), stream(), generate_object() and cancellation/timeout behavior.
Uses mocked adapters — no real API keys needed.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from unified_llm import (
    AbortError,
    Client,
    ConfigurationError,
    FinishReason,
    GenerateResult,
    Message,
    NoObjectGeneratedError,
    Request,
    RequestTimeoutError,
    Response,
    StreamEvent,
    StreamEventType,
    Usage,
)
from unified_llm.generate import (
    AbortController,
    generate,
    generate_object,
    stream,
)

# ---------------------------------------------------------------------------
# Shared mock adapter
# ---------------------------------------------------------------------------


class _MockAdapter:
    def __init__(
        self,
        text: str = "Hello!",
        provider: str = "mock",
    ) -> None:
        self._text = text
        self._provider = provider

    @property
    def name(self) -> str:
        return self._provider

    async def complete(self, request: Request) -> Response:
        return Response(
            id="r1",
            model=request.model,
            provider=self._provider,
            message=Message.assistant(self._text),
            finish_reason=FinishReason(reason="stop"),
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
        )

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(type=StreamEventType.STREAM_START)
        yield StreamEvent(type=StreamEventType.TEXT_START)
        for word in self._text.split():
            yield StreamEvent(type=StreamEventType.TEXT_DELTA, delta=word + " ")
        yield StreamEvent(type=StreamEventType.TEXT_END)
        yield StreamEvent(
            type=StreamEventType.FINISH,
            finish_reason=FinishReason(reason="stop"),
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            response=Response(
                id="r1",
                model=request.model,
                provider=self._provider,
                message=Message.assistant(self._text),
                finish_reason=FinishReason(reason="stop"),
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
        )

    async def close(self) -> None:
        pass

    async def initialize(self) -> None:
        pass

    def supports_tool_choice(self, mode: str) -> bool:
        return True


def _client(text: str = "Hello!") -> Client:
    return Client(
        providers={"mock": _MockAdapter(text=text)},
        default_provider="mock",
    )


# ---------------------------------------------------------------------------
# §8.4 — generate() with prompt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="function")
async def test_generate_with_prompt() -> None:
    """[ ] generate() works with a simple text prompt."""
    result = await generate(
        model="test", prompt="Hi", client=_client(), provider="mock"
    )
    assert isinstance(result, GenerateResult)
    assert result.text == "Hello!"
    assert result.usage.total_tokens == 15


# ---------------------------------------------------------------------------
# §8.4 — generate() with messages
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="function")
async def test_generate_with_messages() -> None:
    """[ ] generate() works with a full messages list."""
    result = await generate(
        model="test",
        messages=[Message.system("Be helpful"), Message.user("Hi")],
        client=_client(),
        provider="mock",
    )
    assert result.text == "Hello!"


# ---------------------------------------------------------------------------
# §8.4 — generate() rejects both prompt and messages
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="function")
async def test_generate_rejects_both_prompt_and_messages() -> None:
    """[ ] generate() rejects when both prompt and messages are provided."""
    with pytest.raises(ConfigurationError, match="Cannot specify both"):
        await generate(
            model="test",
            prompt="Hi",
            messages=[Message.user("Hi")],
            client=_client(),
            provider="mock",
        )


# ---------------------------------------------------------------------------
# §8.4 — stream() TEXT_DELTA events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="function")
async def test_stream_text_deltas() -> None:
    """[ ] stream() yields TEXT_DELTA events that concatenate to the full response text."""
    result = stream(model="test", prompt="Hi", client=_client(), provider="mock")
    deltas: list[str] = []
    async for event in result:
        if event.type == StreamEventType.TEXT_DELTA and event.delta:
            deltas.append(event.delta)
    assert len(deltas) > 0
    assert "Hello!" in "".join(deltas)


# ---------------------------------------------------------------------------
# §8.4 — stream() STREAM_START and FINISH
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="function")
async def test_stream_start_and_finish() -> None:
    """[ ] stream() yields STREAM_START and FINISH events with correct metadata."""
    result = stream(model="test", prompt="Hi", client=_client(), provider="mock")
    event_types: list[StreamEventType | str] = []
    async for event in result:
        event_types.append(event.type)
    assert StreamEventType.STREAM_START in event_types
    assert StreamEventType.FINISH in event_types


# ---------------------------------------------------------------------------
# §8.4 — Streaming start/delta/end pattern
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="function")
async def test_stream_start_delta_end_pattern() -> None:
    """[ ] Streaming follows the start/delta/end pattern for text segments."""
    result = stream(model="test", prompt="Hi", client=_client(), provider="mock")
    event_types: list[StreamEventType | str] = []
    async for event in result:
        event_types.append(event.type)

    # Should see text_start before text_delta before text_end
    text_events = [
        t
        for t in event_types
        if t
        in (
            StreamEventType.TEXT_START,
            StreamEventType.TEXT_DELTA,
            StreamEventType.TEXT_END,
        )
    ]
    assert text_events[0] == StreamEventType.TEXT_START
    assert text_events[-1] == StreamEventType.TEXT_END
    assert any(t == StreamEventType.TEXT_DELTA for t in text_events)


# ---------------------------------------------------------------------------
# §8.4 — generate_object()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="function")
async def test_generate_object_returns_parsed() -> None:
    """[ ] generate_object() returns parsed, validated structured output."""
    json_client = _client(text='{"name": "Alice", "age": 30}')
    result = await generate_object(
        model="test",
        prompt="Extract",
        schema={
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
        },
        client=json_client,
        provider="mock",
    )
    assert result.output is not None
    assert result.output["name"] == "Alice"
    assert result.output["age"] == 30


# ---------------------------------------------------------------------------
# §8.4 — generate_object() raises NoObjectGeneratedError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="function")
async def test_generate_object_raises_on_invalid_json() -> None:
    """[ ] generate_object() raises NoObjectGeneratedError on parse/validation failure."""
    bad_client = _client(text="This is not JSON")
    with pytest.raises(NoObjectGeneratedError):
        await generate_object(
            model="test",
            prompt="Extract",
            schema={"type": "object", "properties": {}, "required": []},
            client=bad_client,
            provider="mock",
        )


# ---------------------------------------------------------------------------
# §8.4 — Cancellation via abort signal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="function")
async def test_abort_signal_cancellation() -> None:
    """[ ] Cancellation via abort signal works for generate()."""
    controller = AbortController()
    controller.abort()  # Pre-abort

    with pytest.raises(AbortError):
        await generate(
            model="test",
            prompt="Hi",
            client=_client(),
            provider="mock",
            abort_signal=controller.signal,
        )


# ---------------------------------------------------------------------------
# §8.4 — Timeouts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="function")
async def test_timeout_works() -> None:
    """[ ] Timeouts work (total timeout)."""

    class _SlowAdapter(_MockAdapter):
        async def complete(self, request: Request) -> Response:
            await asyncio.sleep(10)  # Simulate slow response
            return await super().complete(request)

    slow_client = Client(
        providers={"mock": _SlowAdapter()},
        default_provider="mock",
    )

    with pytest.raises(RequestTimeoutError):
        await generate(
            model="test",
            prompt="Hi",
            client=slow_client,
            provider="mock",
            timeout=0.05,
            max_retries=0,
        )
