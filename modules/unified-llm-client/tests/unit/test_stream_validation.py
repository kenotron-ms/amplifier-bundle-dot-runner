"""TDD tests for stream validation behavior."""

import asyncio
import logging
from collections.abc import AsyncIterator

import pytest

from unified_llm import StreamProtocolError, validate_stream
from unified_llm.types import StreamEvent, StreamEventType, ToolCall


async def _source(events: list[StreamEvent]) -> AsyncIterator[StreamEvent]:
    for event in events:
        yield event


def _collect(
    events: list[StreamEvent],
    *,
    mode: str = "strict",
    logger: logging.Logger | None = None,
) -> list[StreamEvent]:
    async def _run() -> list[StreamEvent]:
        output: list[StreamEvent] = []
        async for event in validate_stream(_source(events), mode=mode, logger=logger):
            output.append(event)
        return output

    return asyncio.run(_run())


def _consume(events: list[StreamEvent], *, mode: str = "strict") -> None:
    async def _run() -> None:
        async for _ in validate_stream(_source(events), mode=mode):
            pass

    asyncio.run(_run())


def test_validate_stream_happy_path_text() -> None:
    events = [
        StreamEvent(type=StreamEventType.STREAM_START),
        StreamEvent(type=StreamEventType.TEXT_START, text_id="t1"),
        StreamEvent(type=StreamEventType.TEXT_DELTA, delta="hello", text_id="t1"),
        StreamEvent(type=StreamEventType.TEXT_END, text_id="t1"),
        StreamEvent(type=StreamEventType.FINISH),
    ]

    assert _collect(events) == events


def test_validate_stream_happy_path_tool_call() -> None:
    events = [
        StreamEvent(type=StreamEventType.STREAM_START),
        StreamEvent(
            type=StreamEventType.TOOL_CALL_START,
            tool_call=ToolCall(id="call_1", name="search", arguments={}),
        ),
        StreamEvent(
            type=StreamEventType.TOOL_CALL_DELTA,
            tool_call=ToolCall(
                id="call_1",
                name="search",
                arguments={},
                raw_arguments='{"q":',
            ),
        ),
        StreamEvent(
            type=StreamEventType.TOOL_CALL_END,
            tool_call=ToolCall(id="call_1", name="search", arguments={"q": "test"}),
        ),
        StreamEvent(type=StreamEventType.FINISH),
    ]

    assert _collect(events) == events


def test_validate_stream_missing_text_start_raises() -> None:
    events = [
        StreamEvent(type=StreamEventType.STREAM_START),
        StreamEvent(type=StreamEventType.TEXT_DELTA, delta="oops"),
        StreamEvent(type=StreamEventType.TEXT_END),
        StreamEvent(type=StreamEventType.FINISH),
    ]

    with pytest.raises(StreamProtocolError):
        _consume(events)


def test_validate_stream_missing_text_end_raises() -> None:
    events = [
        StreamEvent(type=StreamEventType.STREAM_START),
        StreamEvent(type=StreamEventType.TEXT_START),
        StreamEvent(type=StreamEventType.TEXT_DELTA, delta="oops"),
        StreamEvent(type=StreamEventType.FINISH),
    ]

    with pytest.raises(StreamProtocolError):
        _consume(events)


def test_validate_stream_missing_finish_raises() -> None:
    events = [
        StreamEvent(type=StreamEventType.STREAM_START),
        StreamEvent(type=StreamEventType.TEXT_START),
        StreamEvent(type=StreamEventType.TEXT_DELTA, delta="oops"),
        StreamEvent(type=StreamEventType.TEXT_END),
    ]

    with pytest.raises(StreamProtocolError):
        _consume(events)


def test_validate_stream_ordering_text_delta_before_start() -> None:
    events = [
        StreamEvent(type=StreamEventType.STREAM_START),
        StreamEvent(type=StreamEventType.TEXT_DELTA, delta="oops"),
        StreamEvent(type=StreamEventType.TEXT_START),
        StreamEvent(type=StreamEventType.TEXT_END),
        StreamEvent(type=StreamEventType.FINISH),
    ]

    with pytest.raises(StreamProtocolError):
        _consume(events)


def test_validate_stream_ordering_tool_call_end_without_start() -> None:
    events = [
        StreamEvent(type=StreamEventType.STREAM_START),
        StreamEvent(
            type=StreamEventType.TOOL_CALL_END,
            tool_call=ToolCall(id="call_1", name="search", arguments={}),
        ),
        StreamEvent(type=StreamEventType.FINISH),
    ]

    with pytest.raises(StreamProtocolError):
        _consume(events)


def test_validate_stream_missing_tool_call_end_raises() -> None:
    events = [
        StreamEvent(type=StreamEventType.STREAM_START),
        StreamEvent(
            type=StreamEventType.TOOL_CALL_START,
            tool_call=ToolCall(id="call_1", name="search", arguments={}),
        ),
        StreamEvent(
            type=StreamEventType.TOOL_CALL_DELTA,
            tool_call=ToolCall(
                id="call_1",
                name="search",
                arguments={},
                raw_arguments='{"q":',
            ),
        ),
        StreamEvent(type=StreamEventType.FINISH),
    ]

    with pytest.raises(StreamProtocolError):
        _consume(events)


def test_validate_stream_warn_mode_logs_violation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    events = [
        StreamEvent(type=StreamEventType.STREAM_START),
        StreamEvent(type=StreamEventType.TEXT_DELTA, delta="oops"),
        StreamEvent(type=StreamEventType.TEXT_END),
        StreamEvent(type=StreamEventType.FINISH),
    ]

    logger = logging.getLogger("stream_validation_test")
    caplog.set_level(logging.WARNING, logger="stream_validation_test")

    output = _collect(events, mode="warn", logger=logger)

    assert output == events
    assert any(record.levelno >= logging.WARNING for record in caplog.records)


def test_validate_stream_normalize_inserts_text_start() -> None:
    events = [
        StreamEvent(type=StreamEventType.STREAM_START),
        StreamEvent(type=StreamEventType.TEXT_DELTA, delta="hi", text_id="t1"),
        StreamEvent(type=StreamEventType.TEXT_END, text_id="t1"),
        StreamEvent(type=StreamEventType.FINISH),
    ]

    expected = [
        StreamEvent(type=StreamEventType.STREAM_START),
        StreamEvent(type=StreamEventType.TEXT_START, text_id="t1"),
        StreamEvent(type=StreamEventType.TEXT_DELTA, delta="hi", text_id="t1"),
        StreamEvent(type=StreamEventType.TEXT_END, text_id="t1"),
        StreamEvent(type=StreamEventType.FINISH),
    ]

    assert _collect(events, mode="normalize") == expected


def test_validate_stream_normalize_ambiguous_raises() -> None:
    events = [
        StreamEvent(type=StreamEventType.STREAM_START),
        StreamEvent(type=StreamEventType.TEXT_DELTA, delta="one", text_id="t1"),
        StreamEvent(type=StreamEventType.TEXT_DELTA, delta="two", text_id="t2"),
        StreamEvent(type=StreamEventType.TEXT_END, text_id="t1"),
        StreamEvent(type=StreamEventType.TEXT_END, text_id="t2"),
        StreamEvent(type=StreamEventType.FINISH),
    ]

    with pytest.raises(StreamProtocolError):
        _consume(events, mode="normalize")
