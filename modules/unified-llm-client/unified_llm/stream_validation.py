"""Stream validation helpers.

Scaffolded validation for streaming events. Currently passes through events
unchanged while providing the interface for future protocol checks.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from unified_llm.errors import StreamProtocolError
from unified_llm.types import StreamEvent, StreamEventType


async def validate_stream(
    events: AsyncIterator[StreamEvent],
    mode: str = "strict",
    logger: logging.Logger | None = None,
) -> AsyncIterator[StreamEvent]:
    """Validate streaming events before exposing them to callers.

    Args:
        events: Async stream of StreamEvent values.
        mode: Validation mode ("strict", "warn", or "normalize").
        logger: Optional logger for validation diagnostics.
    """
    logger = logger or logging.getLogger(__name__)
    open_text = False
    open_text_id: str | None = None
    open_tool = False
    open_tool_id: str | None = None
    saw_finish = False

    def violation(message: str) -> None:
        if mode == "warn":
            logger.warning(message)
            return
        raise StreamProtocolError(message)

    async for event in events:
        event_type = event.type

        if event_type in {
            StreamEventType.STREAM_START,
            StreamEventType.REASONING_START,
            StreamEventType.REASONING_DELTA,
            StreamEventType.REASONING_END,
            StreamEventType.PROVIDER_EVENT,
            StreamEventType.ERROR,
        }:
            yield event
            continue

        if event_type == StreamEventType.TEXT_START:
            if open_text:
                violation("TEXT_START while text open")
            else:
                open_text = True
                open_text_id = event.text_id
            yield event
            continue

        if event_type == StreamEventType.TEXT_DELTA:
            if not open_text:
                if mode == "normalize" and event.text_id is not None:
                    open_text = True
                    open_text_id = event.text_id
                    yield StreamEvent(
                        type=StreamEventType.TEXT_START, text_id=event.text_id
                    )
                    yield event
                    continue
                violation("TEXT_DELTA without TEXT_START")
                yield event
                continue
            if event.text_id is not None:
                if open_text_id is None:
                    open_text_id = event.text_id
                elif event.text_id != open_text_id:
                    violation("TEXT_DELTA id mismatch")
            yield event
            continue

        if event_type == StreamEventType.TEXT_END:
            if not open_text:
                violation("TEXT_END without TEXT_START")
                yield event
                continue
            if event.text_id is not None and open_text_id is not None:
                if event.text_id != open_text_id:
                    violation("TEXT_END id mismatch")
                    yield event
                    continue
            open_text = False
            open_text_id = None
            yield event
            continue

        if event_type == StreamEventType.TOOL_CALL_START:
            if open_tool:
                violation("TOOL_CALL_START while tool call open")
            if event.tool_call is None:
                violation("TOOL_CALL_START missing tool_call")
            else:
                open_tool = True
                open_tool_id = event.tool_call.id
            yield event
            continue

        if event_type == StreamEventType.TOOL_CALL_DELTA:
            if not open_tool:
                violation("TOOL_CALL_DELTA without TOOL_CALL_START")
            if event.tool_call is None:
                violation("TOOL_CALL_DELTA id mismatch")
            else:
                if event.tool_call.id is not None:
                    if open_tool_id is None:
                        open_tool_id = event.tool_call.id
                    elif event.tool_call.id != open_tool_id:
                        violation("TOOL_CALL_DELTA id mismatch")
            yield event
            continue

        if event_type == StreamEventType.TOOL_CALL_END:
            if not open_tool:
                violation("TOOL_CALL_END without TOOL_CALL_START")
            if event.tool_call is None:
                violation("TOOL_CALL_END id mismatch")
            elif event.tool_call.id is not None and open_tool_id is not None:
                if event.tool_call.id != open_tool_id:
                    violation("TOOL_CALL_END id mismatch")
                    yield event
                    continue
            open_tool = False
            open_tool_id = None
            yield event
            continue

        if event_type == StreamEventType.FINISH:
            if open_text or open_tool:
                violation("FINISH before closing text/tool call")
            saw_finish = True
            yield event
            continue

        yield event

    if not saw_finish:
        violation("Missing FINISH event")
