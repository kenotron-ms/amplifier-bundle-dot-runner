"""Tests for unified_llm.generate — high-level API functions."""

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

from unified_llm.client import Client, set_default_client
from unified_llm.errors import ConfigurationError
from unified_llm.generate import generate
from unified_llm.types import (
    ContentKind,
    ContentPart,
    FinishReason,
    Message,
    Request,
    Response,
    Role,
    StreamEvent,
    StreamEventType,
    Tool,
    ToolCallData,
    Usage,
)

# AsyncIterator, Request, StreamEvent, StreamEventType used in later tasks (41-42)


def _make_response(text: str = "Hello world") -> Response:
    return Response(
        id="r1",
        model="test",
        provider="mock",
        message=Message.assistant(text),
        finish_reason=FinishReason(reason="stop"),
        usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
    )


class _MockAdapter:
    def __init__(self, responses: list[Response] | None = None) -> None:
        if responses is None:
            responses = [_make_response()]
        self._responses = list(responses)
        self._call_index = 0
        self.complete_mock = AsyncMock(side_effect=self._next_response)

    async def _next_response(self, request: object) -> Response:
        idx = min(self._call_index, len(self._responses) - 1)
        self._call_index += 1
        return self._responses[idx]

    @property
    def name(self) -> str:
        return "mock"

    async def complete(self, request: Request) -> Response:
        return await self.complete_mock(request)

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        raise NotImplementedError
        yield  # type: ignore[misc]  # make this an async generator

    async def close(self) -> None:
        pass

    async def initialize(self) -> None:
        pass

    def supports_tool_choice(self, mode: str) -> bool:
        return True


def _make_client(adapter: _MockAdapter | None = None) -> tuple[_MockAdapter, Client]:
    adapter = adapter or _MockAdapter()
    client = Client(providers={"mock": adapter}, default_provider="mock")
    return adapter, client


class TestGenerateBasic:
    """Spec §4.3 — generate() with simple prompt."""

    def test_simple_prompt(self) -> None:
        adapter, client = _make_client()
        result = asyncio.run(
            generate(model="test", prompt="Hello", provider="mock", client=client)
        )
        assert result.text == "Hello world"
        assert result.finish_reason.reason == "stop"
        assert result.usage.total_tokens == 15
        assert result.total_usage.total_tokens == 15
        assert len(result.steps) == 1

    def test_messages_input(self) -> None:
        _, client = _make_client()
        result = asyncio.run(
            generate(
                model="test",
                messages=[Message.user("Hello")],
                provider="mock",
                client=client,
            )
        )
        assert result.text == "Hello world"

    def test_prompt_and_messages_raises(self) -> None:
        """Spec: Using both prompt and messages is an error."""
        _, client = _make_client()
        try:
            asyncio.run(
                generate(
                    model="test",
                    prompt="Hello",
                    messages=[Message.user("Hello")],
                    provider="mock",
                    client=client,
                )
            )
            assert False, "Should raise"
        except ConfigurationError:
            pass

    def test_neither_prompt_nor_messages_raises(self) -> None:
        """Must provide at least one of prompt or messages."""
        _, client = _make_client()
        try:
            asyncio.run(generate(model="test", provider="mock", client=client))
            assert False, "Should raise"
        except ConfigurationError:
            pass

    def test_system_param_prepended(self) -> None:
        adapter, client = _make_client()
        asyncio.run(
            generate(
                model="test",
                prompt="Hi",
                system="Be helpful",
                provider="mock",
                client=client,
            )
        )
        call_args = adapter.complete_mock.call_args[0][0]
        assert call_args.messages[0].role.value == "system"
        assert call_args.messages[0].text == "Be helpful"
        assert call_args.messages[1].role.value == "user"
        assert call_args.messages[1].text == "Hi"

    def test_uses_default_client(self) -> None:
        _, client = _make_client()
        set_default_client(client)
        try:
            result = asyncio.run(
                generate(model="test", prompt="Hello", provider="mock")
            )
            assert result.text == "Hello world"
        finally:
            set_default_client(None)  # type: ignore[arg-type]

    def test_step_result_populated(self) -> None:
        """StepResult for a basic call should have correct fields."""
        _, client = _make_client()
        result = asyncio.run(
            generate(model="test", prompt="Hello", provider="mock", client=client)
        )
        step = result.steps[0]
        assert step.text == "Hello world"
        assert step.finish_reason.reason == "stop"
        assert step.usage.total_tokens == 15
        assert step.tool_calls == []
        assert step.tool_results == []
        assert step.response.id == "r1"

    def test_retry_on_transient_error(self) -> None:
        """Retry applies per-step for transient errors."""
        from unified_llm.errors import ServerError

        call_count = 0

        async def failing_then_ok(request: object) -> Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ServerError(
                    message="Internal error", provider="mock", status_code=500
                )
            return _make_response()

        adapter = _MockAdapter()
        adapter.complete_mock = AsyncMock(side_effect=failing_then_ok)
        _, client = _make_client(adapter)

        result = asyncio.run(
            generate(
                model="test",
                prompt="Hello",
                provider="mock",
                client=client,
                max_retries=2,
            )
        )
        assert result.text == "Hello world"
        assert call_count == 2


# ---------------------------------------------------------------------------
# Helpers for tool loop tests
# ---------------------------------------------------------------------------


def _make_tool_call_response(
    tool_name: str = "get_weather",
    tool_call_id: str = "call_1",
    arguments: dict | None = None,
) -> Response:
    """Create a response that contains a tool call."""
    args = arguments or {"location": "SF"}
    return Response(
        id="r_tc",
        model="test",
        provider="mock",
        message=Message(
            role=Role.ASSISTANT,
            content=[
                ContentPart(
                    kind=ContentKind.TOOL_CALL,
                    tool_call=ToolCallData(
                        id=tool_call_id, name=tool_name, arguments=args
                    ),
                )
            ],
        ),
        finish_reason=FinishReason(reason="tool_calls"),
        usage=Usage(input_tokens=20, output_tokens=10, total_tokens=30),
    )


def _make_multi_tool_call_response() -> Response:
    """Create a response with two parallel tool calls."""
    return Response(
        id="r_mtc",
        model="test",
        provider="mock",
        message=Message(
            role=Role.ASSISTANT,
            content=[
                ContentPart(
                    kind=ContentKind.TOOL_CALL,
                    tool_call=ToolCallData(
                        id="call_a",
                        name="get_weather",
                        arguments={"location": "SF"},
                    ),
                ),
                ContentPart(
                    kind=ContentKind.TOOL_CALL,
                    tool_call=ToolCallData(
                        id="call_b",
                        name="get_weather",
                        arguments={"location": "NYC"},
                    ),
                ),
            ],
        ),
        finish_reason=FinishReason(reason="tool_calls"),
        usage=Usage(input_tokens=20, output_tokens=15, total_tokens=35),
    )


def _weather_tool(with_execute: bool = True) -> Tool:
    """Create a weather tool, optionally with an execute handler."""

    def execute(location: str = "unknown") -> str:
        return f"72F and sunny in {location}"

    return Tool(
        name="get_weather",
        description="Get weather",
        parameters={
            "type": "object",
            "properties": {"location": {"type": "string"}},
        },
        execute=execute if with_execute else None,
    )


# ---------------------------------------------------------------------------
# Task 40: generate() — Tool Loop
# ---------------------------------------------------------------------------


class TestGenerateToolLoop:
    """Spec §5.6 — tool execution loop in generate()."""

    def test_active_tool_triggers_loop(self) -> None:
        """Active tools (with execute) trigger auto-execution and loop."""
        adapter = _MockAdapter(
            responses=[_make_tool_call_response(), _make_response("Weather is 72F")]
        )
        _, client = _make_client(adapter)
        tool = _weather_tool(with_execute=True)

        result = asyncio.run(
            generate(
                model="test",
                prompt="Weather?",
                tools=[tool],
                provider="mock",
                client=client,
            )
        )
        # 2 steps: tool call + final response
        assert len(result.steps) == 2
        assert result.text == "Weather is 72F"
        # Step 0 should have tool calls and results
        assert len(result.steps[0].tool_calls) == 1
        assert result.steps[0].tool_calls[0].name == "get_weather"
        assert len(result.steps[0].tool_results) == 1
        assert result.steps[0].tool_results[0].is_error is False
        assert "72F" in result.steps[0].tool_results[0].content

    def test_passive_tool_no_loop(self) -> None:
        """Passive tools (no execute handler) return tool_calls without looping."""
        adapter = _MockAdapter(responses=[_make_tool_call_response()])
        _, client = _make_client(adapter)
        tool = _weather_tool(with_execute=False)

        result = asyncio.run(
            generate(
                model="test",
                prompt="Weather?",
                tools=[tool],
                provider="mock",
                client=client,
            )
        )
        # Only 1 step — no loop
        assert len(result.steps) == 1
        assert len(result.steps[0].tool_calls) == 1
        assert result.steps[0].tool_results == []
        assert result.finish_reason.reason == "tool_calls"

    def test_max_tool_rounds_zero_disables_execution(self) -> None:
        """max_tool_rounds=0 means no auto-execution (at most 1 LLM call)."""
        adapter = _MockAdapter(responses=[_make_tool_call_response()])
        _, client = _make_client(adapter)
        tool = _weather_tool(with_execute=True)

        result = asyncio.run(
            generate(
                model="test",
                prompt="Weather?",
                tools=[tool],
                max_tool_rounds=0,
                provider="mock",
                client=client,
            )
        )
        # Only 1 step, no tool execution
        assert len(result.steps) == 1
        assert adapter.complete_mock.call_count == 1

    def test_max_tool_rounds_limits_iterations(self) -> None:
        """max_tool_rounds=N means at most N+1 LLM calls."""
        # Model always returns tool calls — should stop after max_tool_rounds
        adapter = _MockAdapter(responses=[_make_tool_call_response()])
        _, client = _make_client(adapter)
        tool = _weather_tool(with_execute=True)

        result = asyncio.run(
            generate(
                model="test",
                prompt="Weather?",
                tools=[tool],
                max_tool_rounds=2,
                provider="mock",
                client=client,
            )
        )
        # max_tool_rounds=2: initial call + 2 rounds = 3 LLM calls
        assert len(result.steps) == 3
        assert adapter.complete_mock.call_count == 3

    def test_parallel_tool_calls_executed_concurrently(self) -> None:
        """When model returns N tool calls, all N execute concurrently."""
        adapter = _MockAdapter(
            responses=[_make_multi_tool_call_response(), _make_response("Both done")]
        )
        _, client = _make_client(adapter)
        tool = _weather_tool(with_execute=True)

        result = asyncio.run(
            generate(
                model="test",
                prompt="Weather in SF and NYC?",
                tools=[tool],
                provider="mock",
                client=client,
            )
        )
        assert len(result.steps) == 2
        # Step 0 should have 2 tool calls and 2 results
        assert len(result.steps[0].tool_calls) == 2
        assert len(result.steps[0].tool_results) == 2
        assert result.steps[0].tool_results[0].is_error is False
        assert result.steps[0].tool_results[1].is_error is False
        assert result.text == "Both done"

    def test_tool_execution_error_produces_error_result(self) -> None:
        """Tool execution errors → error result to model (is_error=True), NOT exception."""

        def failing_tool(location: str = "") -> str:
            raise ValueError("API down")

        tool = Tool(
            name="get_weather",
            description="Get weather",
            parameters={"type": "object", "properties": {}},
            execute=failing_tool,
        )
        adapter = _MockAdapter(
            responses=[_make_tool_call_response(), _make_response("Sorry, tool failed")]
        )
        _, client = _make_client(adapter)

        result = asyncio.run(
            generate(
                model="test",
                prompt="Weather?",
                tools=[tool],
                provider="mock",
                client=client,
            )
        )
        # Should complete without raising
        assert len(result.steps) == 2
        assert result.steps[0].tool_results[0].is_error is True
        assert "API down" in result.steps[0].tool_results[0].content

    def test_unknown_tool_call_produces_error_result(self) -> None:
        """Unknown tool calls → error result, not exception."""
        # Response calls "unknown_tool" but we only define "get_weather"
        adapter = _MockAdapter(
            responses=[
                _make_tool_call_response(tool_name="unknown_tool"),
                _make_response("Recovered"),
            ]
        )
        _, client = _make_client(adapter)
        tool = _weather_tool(with_execute=True)

        result = asyncio.run(
            generate(
                model="test",
                prompt="Do something",
                tools=[tool],
                provider="mock",
                client=client,
            )
        )
        assert len(result.steps) == 2
        assert result.steps[0].tool_results[0].is_error is True
        assert "Unknown tool" in result.steps[0].tool_results[0].content

    def test_total_usage_aggregated_across_steps(self) -> None:
        """total_usage sums usage across ALL steps."""
        adapter = _MockAdapter(
            responses=[_make_tool_call_response(), _make_response("Done")]
        )
        _, client = _make_client(adapter)
        tool = _weather_tool(with_execute=True)

        result = asyncio.run(
            generate(
                model="test",
                prompt="Weather?",
                tools=[tool],
                provider="mock",
                client=client,
            )
        )
        # Step 0: 30 total, Step 1: 15 total
        assert result.total_usage.total_tokens == 45
        assert result.total_usage.input_tokens == 30
        assert result.total_usage.output_tokens == 15
        # result.usage is just the final step's usage
        assert result.usage.total_tokens == 15

    def test_conversation_includes_tool_results(self) -> None:
        """After tool execution, conversation includes assistant msg + tool results."""
        adapter = _MockAdapter(
            responses=[_make_tool_call_response(), _make_response("Done")]
        )
        _, client = _make_client(adapter)
        tool = _weather_tool(with_execute=True)

        asyncio.run(
            generate(
                model="test",
                prompt="Weather?",
                tools=[tool],
                provider="mock",
                client=client,
            )
        )
        # Check second call's messages include tool result
        second_call = adapter.complete_mock.call_args_list[1][0][0]
        messages = second_call.messages
        # Should be: user msg, assistant (tool call), tool result
        assert messages[-1].role == Role.TOOL
        assert messages[-2].role == Role.ASSISTANT

    def test_async_tool_execute(self) -> None:
        """Tool execute handlers can be async."""

        async def async_weather(location: str = "unknown") -> str:
            return f"Async: 72F in {location}"

        tool = Tool(
            name="get_weather",
            description="Get weather",
            parameters={"type": "object", "properties": {}},
            execute=async_weather,
        )
        adapter = _MockAdapter(
            responses=[_make_tool_call_response(), _make_response("Done")]
        )
        _, client = _make_client(adapter)

        result = asyncio.run(
            generate(
                model="test",
                prompt="Weather?",
                tools=[tool],
                provider="mock",
                client=client,
            )
        )
        assert len(result.steps) == 2
        assert "Async: 72F" in result.steps[0].tool_results[0].content


# ---------------------------------------------------------------------------
# Helpers for stream tests
# ---------------------------------------------------------------------------


def _mock_stream_events() -> list[StreamEvent]:
    """Basic stream events: start, text deltas, finish."""
    return [
        StreamEvent(type=StreamEventType.STREAM_START),
        StreamEvent(type=StreamEventType.TEXT_START, text_id="t1"),
        StreamEvent(type=StreamEventType.TEXT_DELTA, delta="Hello ", text_id="t1"),
        StreamEvent(type=StreamEventType.TEXT_DELTA, delta="world", text_id="t1"),
        StreamEvent(type=StreamEventType.TEXT_END, text_id="t1"),
        StreamEvent(
            type=StreamEventType.FINISH,
            finish_reason=FinishReason(reason="stop"),
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            response=Response(
                id="r_s1",
                model="test",
                provider="mock",
                message=Message.assistant("Hello world"),
                finish_reason=FinishReason(reason="stop"),
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
        ),
    ]


class _StreamingMockAdapter:
    """Mock adapter that supports both complete() and stream()."""

    def __init__(
        self,
        stream_events_sequence: list[list[StreamEvent]] | None = None,
        responses: list[Response] | None = None,
    ) -> None:
        self._stream_events_sequence = stream_events_sequence or [_mock_stream_events()]
        self._stream_call_index = 0
        self._responses = responses or [_make_response()]
        self._complete_call_index = 0
        self.complete_mock = AsyncMock(side_effect=self._next_response)

    async def _next_response(self, request: object) -> Response:
        idx = min(self._complete_call_index, len(self._responses) - 1)
        self._complete_call_index += 1
        return self._responses[idx]

    @property
    def name(self) -> str:
        return "mock"

    async def complete(self, request: Request) -> Response:
        return await self.complete_mock(request)

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        idx = min(self._stream_call_index, len(self._stream_events_sequence) - 1)
        self._stream_call_index += 1
        events = self._stream_events_sequence[idx]
        for event in events:
            yield event

    async def close(self) -> None:
        pass

    async def initialize(self) -> None:
        pass

    def supports_tool_choice(self, mode: str) -> bool:
        return True


# ---------------------------------------------------------------------------
# Task 41: stream() — Basic
# ---------------------------------------------------------------------------


class TestStreamBasic:
    """Spec §4.4 — stream() basic without tools."""

    def test_yields_stream_events(self) -> None:
        """stream() returns a StreamResult that yields StreamEvent objects."""
        from unified_llm.generate import stream

        adapter = _StreamingMockAdapter()
        client = Client(providers={"mock": adapter}, default_provider="mock")

        async def run() -> list[StreamEvent]:
            result = stream(
                model="test", prompt="Hello", provider="mock", client=client
            )
            events: list[StreamEvent] = []
            async for event in result:
                events.append(event)
            return events

        events = asyncio.run(run())
        # Should have all the mock events
        assert len(events) >= 3
        types = [e.type for e in events]
        assert StreamEventType.TEXT_DELTA in types
        assert StreamEventType.FINISH in types

    def test_response_available_after_iteration(self) -> None:
        """StreamResult.response() returns accumulated Response after stream ends."""
        from unified_llm.generate import stream

        adapter = _StreamingMockAdapter()
        client = Client(providers={"mock": adapter}, default_provider="mock")

        async def run() -> Response:
            result = stream(
                model="test", prompt="Hello", provider="mock", client=client
            )
            async for _ in result:
                pass
            return result.response()

        response = asyncio.run(run())
        assert response.text == "Hello world"
        assert response.finish_reason.reason == "stop"

    def test_text_stream_yields_only_text_deltas(self) -> None:
        """StreamResult.text_stream yields only text delta strings."""
        from unified_llm.generate import stream

        adapter = _StreamingMockAdapter()
        client = Client(providers={"mock": adapter}, default_provider="mock")

        async def run() -> list[str]:
            result = stream(
                model="test", prompt="Hello", provider="mock", client=client
            )
            chunks: list[str] = []
            async for chunk in result.text_stream:
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(run())
        assert chunks == ["Hello ", "world"]

    def test_prompt_and_messages_raises(self) -> None:
        """stream() raises ConfigurationError if both prompt and messages given."""
        from unified_llm.generate import stream

        adapter = _StreamingMockAdapter()
        client = Client(providers={"mock": adapter}, default_provider="mock")

        async def run() -> None:
            result = stream(
                model="test",
                prompt="Hello",
                messages=[Message.user("Hello")],
                provider="mock",
                client=client,
            )
            async for _ in result:
                pass

        try:
            asyncio.run(run())
            assert False, "Should raise"
        except ConfigurationError:
            pass

    def test_system_param_prepended_in_stream(self) -> None:
        """stream() prepends system message like generate()."""
        from unified_llm.generate import stream

        captured_requests: list[Request] = []

        class _CapturingAdapter(_StreamingMockAdapter):
            async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
                captured_requests.append(request)
                for event in _mock_stream_events():
                    yield event

        adapter = _CapturingAdapter()
        client = Client(providers={"mock": adapter}, default_provider="mock")

        async def run() -> None:
            result = stream(
                model="test",
                prompt="Hi",
                system="Be helpful",
                provider="mock",
                client=client,
            )
            async for _ in result:
                pass

        asyncio.run(run())
        assert len(captured_requests) == 1
        msgs = captured_requests[0].messages
        assert msgs[0].role == Role.SYSTEM
        assert msgs[0].text == "Be helpful"
        assert msgs[1].role == Role.USER

    def test_stream_retry_on_initial_connection(self) -> None:
        """Spec: stream() retries on initial connection failure, not after partial data."""
        from unified_llm.errors import ServerError
        from unified_llm.generate import stream

        call_count = 0

        class _RetryAdapter(_StreamingMockAdapter):
            async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise ServerError(
                        message="Connection failed",
                        provider="mock",
                        status_code=500,
                    )
                for event in _mock_stream_events():
                    yield event

        adapter = _RetryAdapter()
        client = Client(providers={"mock": adapter}, default_provider="mock")

        async def run() -> list[StreamEvent]:
            result = stream(
                model="test",
                prompt="Hello",
                provider="mock",
                client=client,
                max_retries=2,
            )
            events: list[StreamEvent] = []
            async for event in result:
                events.append(event)
            return events

        events = asyncio.run(run())
        assert call_count == 2
        assert any(e.type == StreamEventType.FINISH for e in events)

    def test_stream_validation_warn_allows_invalid_sequence(self) -> None:
        """stream_validation_mode='warn' allows invalid event ordering."""
        from unified_llm.generate import stream

        invalid_events = [
            StreamEvent(type=StreamEventType.STREAM_START),
            StreamEvent(type=StreamEventType.TEXT_DELTA, delta="Hello", text_id="t1"),
            StreamEvent(
                type=StreamEventType.FINISH,
                finish_reason=FinishReason(reason="stop"),
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
                response=_make_response("Hello"),
            ),
        ]

        adapter = _StreamingMockAdapter(stream_events_sequence=[invalid_events])
        client = Client(providers={"mock": adapter}, default_provider="mock")

        async def run() -> list[StreamEvent]:
            result = stream(
                model="test",
                prompt="Hello",
                provider="mock",
                client=client,
                stream_validation_mode="warn",
            )
            events: list[StreamEvent] = []
            async for event in result:
                events.append(event)
            return events

        events = asyncio.run(run())
        assert any(e.type == StreamEventType.TEXT_DELTA for e in events)


# ---------------------------------------------------------------------------
# Helpers for stream tool loop tests
# ---------------------------------------------------------------------------


def _mock_tool_call_stream_events() -> list[StreamEvent]:
    """Stream events that include a tool call (like _make_tool_call_response but streamed)."""
    from unified_llm.types import ToolCall as TC

    return [
        StreamEvent(type=StreamEventType.STREAM_START),
        StreamEvent(
            type=StreamEventType.TOOL_CALL_START,
            tool_call=TC(id="call_s1", name="get_weather", arguments={}),
        ),
        StreamEvent(
            type=StreamEventType.TOOL_CALL_END,
            tool_call=TC(
                id="call_s1",
                name="get_weather",
                arguments={"location": "SF"},
            ),
        ),
        StreamEvent(
            type=StreamEventType.FINISH,
            finish_reason=FinishReason(reason="tool_calls"),
            usage=Usage(input_tokens=20, output_tokens=10, total_tokens=30),
            response=Response(
                id="r_stc",
                model="test",
                provider="mock",
                message=Message(
                    role=Role.ASSISTANT,
                    content=[
                        ContentPart(
                            kind=ContentKind.TOOL_CALL,
                            tool_call=ToolCallData(
                                id="call_s1",
                                name="get_weather",
                                arguments={"location": "SF"},
                            ),
                        )
                    ],
                ),
                finish_reason=FinishReason(reason="tool_calls"),
                usage=Usage(input_tokens=20, output_tokens=10, total_tokens=30),
            ),
        ),
    ]


def _mock_final_stream_events(text: str = "Weather is 72F") -> list[StreamEvent]:
    """Final stream events (text response after tool execution)."""
    return [
        StreamEvent(type=StreamEventType.STREAM_START),
        StreamEvent(type=StreamEventType.TEXT_START, text_id="t1"),
        StreamEvent(type=StreamEventType.TEXT_DELTA, delta=text, text_id="t1"),
        StreamEvent(type=StreamEventType.TEXT_END, text_id="t1"),
        StreamEvent(
            type=StreamEventType.FINISH,
            finish_reason=FinishReason(reason="stop"),
            usage=Usage(input_tokens=30, output_tokens=8, total_tokens=38),
            response=Response(
                id="r_sf",
                model="test",
                provider="mock",
                message=Message.assistant(text),
                finish_reason=FinishReason(reason="stop"),
                usage=Usage(input_tokens=30, output_tokens=8, total_tokens=38),
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Task 42: stream() — Tool Loop
# ---------------------------------------------------------------------------


class TestStreamToolLoop:
    """Spec §5.9 — streaming with active tools and tool execution loop."""

    def test_stream_tool_loop_executes_tools_and_resumes(self) -> None:
        """Stream pauses during tool execution, emits step_finish, then resumes."""
        from unified_llm.generate import stream

        adapter = _StreamingMockAdapter(
            stream_events_sequence=[
                _mock_tool_call_stream_events(),
                _mock_final_stream_events(),
            ],
        )
        client = Client(providers={"mock": adapter}, default_provider="mock")
        tool = _weather_tool(with_execute=True)

        async def run() -> list[StreamEvent]:
            result = stream(
                model="test",
                prompt="Weather?",
                tools=[tool],
                provider="mock",
                client=client,
            )
            events: list[StreamEvent] = []
            async for event in result:
                events.append(event)
            return events

        events = asyncio.run(run())
        types = [e.type for e in events]

        # Should have events from both steps plus a step_finish between them
        assert "step_finish" in [str(t) for t in types]
        assert StreamEventType.TOOL_CALL_END in types
        assert StreamEventType.TEXT_DELTA in types

        # Find the step_finish event
        step_finish_events = [e for e in events if str(e.type) == "step_finish"]
        assert len(step_finish_events) == 1
        assert step_finish_events[0].response is not None

    def test_stream_tool_loop_text_stream_spans_steps(self) -> None:
        """text_stream yields text from all steps (including after tool execution)."""
        from unified_llm.generate import stream

        adapter = _StreamingMockAdapter(
            stream_events_sequence=[
                _mock_tool_call_stream_events(),
                _mock_final_stream_events("Done!"),
            ],
        )
        client = Client(providers={"mock": adapter}, default_provider="mock")
        tool = _weather_tool(with_execute=True)

        async def run() -> list[str]:
            result = stream(
                model="test",
                prompt="Weather?",
                tools=[tool],
                provider="mock",
                client=client,
            )
            chunks: list[str] = []
            async for chunk in result.text_stream:
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(run())
        # Only the final step has text deltas
        assert "Done!" in chunks

    def test_stream_passive_tools_no_loop(self) -> None:
        """Passive tools (no execute) return tool calls without looping."""
        from unified_llm.generate import stream

        adapter = _StreamingMockAdapter(
            stream_events_sequence=[_mock_tool_call_stream_events()],
        )
        client = Client(providers={"mock": adapter}, default_provider="mock")
        tool = _weather_tool(with_execute=False)

        async def run() -> list[StreamEvent]:
            result = stream(
                model="test",
                prompt="Weather?",
                tools=[tool],
                provider="mock",
                client=client,
            )
            events: list[StreamEvent] = []
            async for event in result:
                events.append(event)
            return events

        events = asyncio.run(run())
        types = [str(e.type) for e in events]
        # No step_finish — should not loop
        assert "step_finish" not in types
        # Should have tool call events
        assert str(StreamEventType.TOOL_CALL_END) in types

    def test_stream_response_after_tool_loop(self) -> None:
        """response() after consuming stream with tools returns final accumulated response."""
        from unified_llm.generate import stream

        adapter = _StreamingMockAdapter(
            stream_events_sequence=[
                _mock_tool_call_stream_events(),
                _mock_final_stream_events("Final answer"),
            ],
        )
        client = Client(providers={"mock": adapter}, default_provider="mock")
        tool = _weather_tool(with_execute=True)

        async def run() -> Response:
            result = stream(
                model="test",
                prompt="Weather?",
                tools=[tool],
                provider="mock",
                client=client,
            )
            async for _ in result:
                pass
            return result.response()

        response = asyncio.run(run())
        # The response() accumulator captures events from ALL steps
        # The final step's finish event has the response
        assert response.finish_reason.reason == "stop"


# ---------------------------------------------------------------------------
# Task 43: generate_object() — Structured Output
# ---------------------------------------------------------------------------


class TestGenerateObject:
    """Spec §4.5 — generate_object() with JSON schema validation."""

    def test_basic_structured_output(self) -> None:
        """generate_object() parses JSON response and sets result.output."""
        from unified_llm.generate import generate_object

        json_response = _make_response('{"name": "Alice", "age": 30}')
        adapter = _MockAdapter(responses=[json_response])
        _, client = _make_client(adapter)

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }

        result = asyncio.run(
            generate_object(
                model="test",
                prompt="Extract info",
                schema=schema,
                provider="mock",
                client=client,
            )
        )
        assert result.output == {"name": "Alice", "age": 30}
        assert result.text == '{"name": "Alice", "age": 30}'

    def test_sets_response_format_json_schema(self) -> None:
        """generate_object() sets response_format to json_schema with the schema."""
        from unified_llm.generate import generate_object

        json_response = _make_response('{"x": 1}')
        adapter = _MockAdapter(responses=[json_response])
        _, client = _make_client(adapter)

        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}

        asyncio.run(
            generate_object(
                model="test",
                prompt="Extract",
                schema=schema,
                provider="mock",
                client=client,
            )
        )
        call_args = adapter.complete_mock.call_args[0][0]
        assert call_args.response_format is not None
        assert call_args.response_format.type == "json_schema"
        assert call_args.response_format.json_schema == schema

    def test_invalid_json_raises_no_object_generated(self) -> None:
        """generate_object() raises NoObjectGeneratedError on invalid JSON."""
        from unified_llm.errors import NoObjectGeneratedError
        from unified_llm.generate import generate_object

        bad_response = _make_response("This is not JSON at all")
        adapter = _MockAdapter(responses=[bad_response])
        _, client = _make_client(adapter)

        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}

        try:
            asyncio.run(
                generate_object(
                    model="test",
                    prompt="Extract",
                    schema=schema,
                    provider="mock",
                    client=client,
                )
            )
            assert False, "Should raise"
        except NoObjectGeneratedError:
            pass

    def test_schema_validation_failure_raises_no_object_generated(self) -> None:
        """generate_object() raises NoObjectGeneratedError on schema validation failure."""
        from unified_llm.errors import NoObjectGeneratedError
        from unified_llm.generate import generate_object

        # Valid JSON but doesn't match schema (missing required field)
        bad_response = _make_response('{"name": "Alice"}')
        adapter = _MockAdapter(responses=[bad_response])
        _, client = _make_client(adapter)

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }

        try:
            asyncio.run(
                generate_object(
                    model="test",
                    prompt="Extract",
                    schema=schema,
                    provider="mock",
                    client=client,
                )
            )
            assert False, "Should raise"
        except NoObjectGeneratedError:
            pass

    def test_schema_validation_not_retried(self) -> None:
        """Spec: schema validation failures are NOT retried."""
        from unified_llm.errors import NoObjectGeneratedError
        from unified_llm.generate import generate_object

        bad_response = _make_response("not json")
        adapter = _MockAdapter(responses=[bad_response])
        _, client = _make_client(adapter)

        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}

        try:
            asyncio.run(
                generate_object(
                    model="test",
                    prompt="Extract",
                    schema=schema,
                    provider="mock",
                    client=client,
                    max_retries=3,
                )
            )
            assert False, "Should raise"
        except NoObjectGeneratedError:
            pass
        # Should have only called the adapter once (no retry on schema failure)
        assert adapter.complete_mock.call_count == 1

    def test_json_with_markdown_fences_parsed(self) -> None:
        """generate_object() strips markdown code fences around JSON."""
        from unified_llm.generate import generate_object

        fenced = _make_response('```json\n{"x": 42}\n```')
        adapter = _MockAdapter(responses=[fenced])
        _, client = _make_client(adapter)

        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}

        result = asyncio.run(
            generate_object(
                model="test",
                prompt="Extract",
                schema=schema,
                provider="mock",
                client=client,
            )
        )
        assert result.output == {"x": 42}

    def test_prompt_and_messages_raises(self) -> None:
        """generate_object() raises ConfigurationError if both prompt and messages given."""
        from unified_llm.generate import generate_object

        _, client = _make_client()
        schema = {"type": "object", "properties": {}}

        try:
            asyncio.run(
                generate_object(
                    model="test",
                    prompt="Hello",
                    messages=[Message.user("Hello")],
                    schema=schema,
                    provider="mock",
                    client=client,
                )
            )
            assert False, "Should raise"
        except ConfigurationError:
            pass

    # ------------------------------------------------------------------
    # Anthropic tool-extraction path (Spec §4.5 / capability matrix :989)
    # ------------------------------------------------------------------

    def test_tool_extraction_path_returns_tool_arguments_as_output(self) -> None:
        """generate_object() extracts output from __structured_output__ tool call.

        When the Anthropic adapter uses tool-based extraction, the provider
        returns a tool_call instead of JSON text.  generate_object() must
        detect the extraction tool by name and use its .arguments dict
        directly rather than attempting to parse an empty text body.
        """
        from unified_llm.generate import generate_object

        # Convention: the extraction tool name is "__structured_output__"
        # (matches STRUCTURED_OUTPUT_TOOL_NAME in adapters/anthropic.py and
        #  _ANTHROPIC_STRUCTURED_OUTPUT_TOOL in generate.py).
        _TOOL_NAME = "__structured_output__"

        structured_data = {"name": "Alice", "age": 30}
        tool_call_response = Response(
            id="r1",
            model="test",
            provider="mock",
            message=Message(
                role=Role.ASSISTANT,
                content=[
                    ContentPart(
                        kind=ContentKind.TOOL_CALL,
                        tool_call=ToolCallData(
                            id="tc_1",
                            name=_TOOL_NAME,
                            arguments=structured_data,
                        ),
                    )
                ],
            ),
            finish_reason=FinishReason(reason="tool_calls"),
            usage=Usage(input_tokens=10, output_tokens=20, total_tokens=30),
        )

        adapter = _MockAdapter(responses=[tool_call_response])
        _, client = _make_client(adapter)

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }

        result = asyncio.run(
            generate_object(
                model="test",
                prompt="Extract info",
                schema=schema,
                provider="mock",
                client=client,
            )
        )
        assert result.output == structured_data, (
            f"Expected {structured_data}, got {result.output}"
        )

    def test_empty_text_and_no_extraction_tool_raises(self) -> None:
        """generate_object() raises NoObjectGeneratedError when text is empty and no extraction tool.

        Ensures the fail-loud requirement: a provider that returns no text and no
        structured-output tool call triggers an error rather than silent success.
        """
        from unified_llm.errors import NoObjectGeneratedError
        from unified_llm.generate import generate_object

        # Response with an unrelated tool call (not the extraction tool) and no text
        unrelated_response = Response(
            id="r2",
            model="test",
            provider="mock",
            message=Message(
                role=Role.ASSISTANT,
                content=[
                    ContentPart(
                        kind=ContentKind.TOOL_CALL,
                        tool_call=ToolCallData(
                            id="tc_2",
                            name="some_other_tool",
                            arguments={"x": 1},
                        ),
                    )
                ],
            ),
            finish_reason=FinishReason(reason="tool_calls"),
            usage=Usage(input_tokens=5, output_tokens=5, total_tokens=10),
        )

        adapter = _MockAdapter(responses=[unrelated_response])
        _, client = _make_client(adapter)

        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}

        try:
            asyncio.run(
                generate_object(
                    model="test",
                    prompt="Extract",
                    schema=schema,
                    provider="mock",
                    client=client,
                )
            )
            assert False, "Should raise NoObjectGeneratedError"
        except NoObjectGeneratedError:
            pass


# ---------------------------------------------------------------------------
# Task 44: stream_object() — Streaming Structured Output
# ---------------------------------------------------------------------------


def _json_stream_events(
    json_text: str = '{"name": "Alice", "age": 30}',
) -> list[StreamEvent]:
    """Stream events that incrementally produce JSON text."""
    # Split JSON into chunks for realistic streaming
    mid = len(json_text) // 2
    chunk1 = json_text[:mid]
    chunk2 = json_text[mid:]
    return [
        StreamEvent(type=StreamEventType.STREAM_START),
        StreamEvent(type=StreamEventType.TEXT_START, text_id="t1"),
        StreamEvent(type=StreamEventType.TEXT_DELTA, delta=chunk1, text_id="t1"),
        StreamEvent(type=StreamEventType.TEXT_DELTA, delta=chunk2, text_id="t1"),
        StreamEvent(type=StreamEventType.TEXT_END, text_id="t1"),
        StreamEvent(
            type=StreamEventType.FINISH,
            finish_reason=FinishReason(reason="stop"),
            usage=Usage(input_tokens=10, output_tokens=20, total_tokens=30),
            response=Response(
                id="r_so",
                model="test",
                provider="mock",
                message=Message.assistant(json_text),
                finish_reason=FinishReason(reason="stop"),
                usage=Usage(input_tokens=10, output_tokens=20, total_tokens=30),
            ),
        ),
    ]


class TestStreamObject:
    """Spec §4.6 — stream_object() with incremental JSON parsing."""

    def test_yields_partial_objects(self) -> None:
        """stream_object() yields partial objects as JSON tokens arrive."""
        from unified_llm.generate import stream_object

        adapter = _StreamingMockAdapter(
            stream_events_sequence=[
                _json_stream_events('{"name": "Alice", "age": 30}')
            ],
        )
        client = Client(providers={"mock": adapter}, default_provider="mock")
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        }

        async def run() -> list[dict | None]:
            result = stream_object(
                model="test",
                prompt="Extract info",
                schema=schema,
                provider="mock",
                client=client,
            )
            partials: list[dict | None] = []
            async for partial in result:
                partials.append(partial)
            return partials

        partials = asyncio.run(run())
        # Should have at least one partial and a final complete object
        assert len(partials) >= 1
        # The last partial should be the complete object
        assert partials[-1] == {"name": "Alice", "age": 30}

    def test_final_object_available(self) -> None:
        """stream_object().object() returns the final validated object."""
        from unified_llm.generate import stream_object

        adapter = _StreamingMockAdapter(
            stream_events_sequence=[_json_stream_events('{"x": 42}')],
        )
        client = Client(providers={"mock": adapter}, default_provider="mock")
        schema = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
        }

        async def run() -> object:
            result = stream_object(
                model="test",
                prompt="Extract",
                schema=schema,
                provider="mock",
                client=client,
            )
            async for _ in result:
                pass
            return result.object()

        obj = asyncio.run(run())
        assert obj == {"x": 42}

    def test_invalid_json_raises_no_object_generated(self) -> None:
        """stream_object().object() raises NoObjectGeneratedError on invalid JSON."""
        from unified_llm.errors import NoObjectGeneratedError
        from unified_llm.generate import stream_object

        adapter = _StreamingMockAdapter(
            stream_events_sequence=[_json_stream_events("not valid json at all")],
        )
        client = Client(providers={"mock": adapter}, default_provider="mock")
        schema = {"type": "object", "properties": {}}

        async def run() -> object:
            result = stream_object(
                model="test",
                prompt="Extract",
                schema=schema,
                provider="mock",
                client=client,
            )
            async for _ in result:
                pass
            return result.object()

        try:
            asyncio.run(run())
            assert False, "Should raise"
        except NoObjectGeneratedError:
            pass

    def test_sets_response_format(self) -> None:
        """stream_object() sets response_format to json_schema."""
        from unified_llm.generate import stream_object

        captured_requests: list[Request] = []

        class _CapturingAdapter(_StreamingMockAdapter):
            async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
                captured_requests.append(request)
                for event in _json_stream_events('{"x": 1}'):
                    yield event

        adapter = _CapturingAdapter()
        client = Client(providers={"mock": adapter}, default_provider="mock")
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}

        async def run() -> None:
            result = stream_object(
                model="test",
                prompt="Extract",
                schema=schema,
                provider="mock",
                client=client,
            )
            async for _ in result:
                pass

        asyncio.run(run())
        assert len(captured_requests) == 1
        assert captured_requests[0].response_format is not None
        assert captured_requests[0].response_format.type == "json_schema"


# ---------------------------------------------------------------------------
# Task 45: Abort + Timeout
# ---------------------------------------------------------------------------


class TestAbortSignal:
    """Spec §4.7 — AbortSignal/AbortController for cancellation."""

    def test_abort_controller_cancels_generate(self) -> None:
        """AbortController.abort() cancels a pending generate() with AbortError."""
        from unified_llm.errors import AbortError
        from unified_llm.generate import AbortController

        controller = AbortController()

        async def slow_complete(request: object) -> Response:
            # Simulate a slow operation — abort before it completes
            await asyncio.sleep(10)
            return _make_response()

        adapter = _MockAdapter()
        adapter.complete_mock = AsyncMock(side_effect=slow_complete)
        _, client = _make_client(adapter)

        async def run() -> None:
            # Schedule abort after a short delay
            async def abort_soon() -> None:
                await asyncio.sleep(0.05)
                controller.abort()

            task = asyncio.create_task(abort_soon())
            try:
                await generate(
                    model="test",
                    prompt="Hello",
                    provider="mock",
                    client=client,
                    abort_signal=controller.signal,
                )
            finally:
                await task

        try:
            asyncio.run(run())
            assert False, "Should raise AbortError"
        except AbortError:
            pass

    def test_abort_signal_already_aborted(self) -> None:
        """If signal is already aborted, generate() raises immediately."""
        from unified_llm.errors import AbortError
        from unified_llm.generate import AbortController

        controller = AbortController()
        controller.abort()  # Abort before calling generate

        _, client = _make_client()

        try:
            asyncio.run(
                generate(
                    model="test",
                    prompt="Hello",
                    provider="mock",
                    client=client,
                    abort_signal=controller.signal,
                )
            )
            assert False, "Should raise AbortError"
        except AbortError:
            pass

    def test_abort_cancels_stream(self) -> None:
        """AbortController.abort() cancels a pending stream() with AbortError."""
        from unified_llm.errors import AbortError
        from unified_llm.generate import AbortController, stream

        controller = AbortController()

        class _SlowStreamAdapter(_StreamingMockAdapter):
            async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
                yield StreamEvent(type=StreamEventType.STREAM_START)
                yield StreamEvent(type=StreamEventType.TEXT_START, text_id="t1")
                await asyncio.sleep(10)  # Will be cancelled
                yield StreamEvent(
                    type=StreamEventType.TEXT_DELTA,
                    delta="never",
                    text_id="t1",
                )

        adapter = _SlowStreamAdapter()
        client = Client(providers={"mock": adapter}, default_provider="mock")

        async def run() -> None:
            async def abort_soon() -> None:
                await asyncio.sleep(0.05)
                controller.abort()

            task = asyncio.create_task(abort_soon())
            try:
                result = stream(
                    model="test",
                    prompt="Hello",
                    provider="mock",
                    client=client,
                    abort_signal=controller.signal,
                )
                async for _ in result:
                    pass
            finally:
                await task

        try:
            asyncio.run(run())
            assert False, "Should raise AbortError"
        except AbortError:
            pass


class TestTimeout:
    """Spec §4.7 — TimeoutConfig for total and per-step timeouts."""

    def test_total_timeout_on_generate(self) -> None:
        """Total timeout cancels entire multi-step generate()."""
        from unified_llm.errors import RequestTimeoutError
        from unified_llm.types import TimeoutConfig

        async def slow_complete(request: object) -> Response:
            await asyncio.sleep(10)
            return _make_response()

        adapter = _MockAdapter()
        adapter.complete_mock = AsyncMock(side_effect=slow_complete)
        _, client = _make_client(adapter)

        try:
            asyncio.run(
                generate(
                    model="test",
                    prompt="Hello",
                    provider="mock",
                    client=client,
                    timeout=TimeoutConfig(total=0.1),
                )
            )
            assert False, "Should raise RequestTimeoutError"
        except RequestTimeoutError:
            pass

    def test_per_step_timeout(self) -> None:
        """Per-step timeout applies to each individual LLM call."""
        from unified_llm.errors import RequestTimeoutError
        from unified_llm.types import TimeoutConfig

        async def slow_complete(request: object) -> Response:
            await asyncio.sleep(10)
            return _make_response()

        adapter = _MockAdapter()
        adapter.complete_mock = AsyncMock(side_effect=slow_complete)
        _, client = _make_client(adapter)

        try:
            asyncio.run(
                generate(
                    model="test",
                    prompt="Hello",
                    provider="mock",
                    client=client,
                    timeout=TimeoutConfig(per_step=0.1),
                )
            )
            assert False, "Should raise RequestTimeoutError"
        except RequestTimeoutError:
            pass

    def test_float_timeout_treated_as_total(self) -> None:
        """A plain float timeout is treated as total timeout."""
        from unified_llm.errors import RequestTimeoutError

        async def slow_complete(request: object) -> Response:
            await asyncio.sleep(10)
            return _make_response()

        adapter = _MockAdapter()
        adapter.complete_mock = AsyncMock(side_effect=slow_complete)
        _, client = _make_client(adapter)

        try:
            asyncio.run(
                generate(
                    model="test",
                    prompt="Hello",
                    provider="mock",
                    client=client,
                    timeout=0.1,
                )
            )
            assert False, "Should raise RequestTimeoutError"
        except RequestTimeoutError:
            pass

    def test_total_timeout_on_stream(self) -> None:
        """Total timeout cancels streaming."""
        from unified_llm.errors import RequestTimeoutError
        from unified_llm.generate import stream
        from unified_llm.types import TimeoutConfig

        class _SlowStreamAdapter(_StreamingMockAdapter):
            async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
                yield StreamEvent(type=StreamEventType.STREAM_START)
                yield StreamEvent(type=StreamEventType.TEXT_START, text_id="t1")
                await asyncio.sleep(10)
                yield StreamEvent(
                    type=StreamEventType.TEXT_DELTA,
                    delta="never",
                    text_id="t1",
                )

        adapter = _SlowStreamAdapter()
        client = Client(providers={"mock": adapter}, default_provider="mock")

        async def run() -> None:
            result = stream(
                model="test",
                prompt="Hello",
                provider="mock",
                client=client,
                timeout=TimeoutConfig(total=0.1),
            )
            async for _ in result:
                pass

        try:
            asyncio.run(run())
            assert False, "Should raise RequestTimeoutError"
        except RequestTimeoutError:
            pass

    def test_no_timeout_completes_normally(self) -> None:
        """When no timeout is set, operations complete normally."""
        _, client = _make_client()
        result = asyncio.run(
            generate(model="test", prompt="Hello", provider="mock", client=client)
        )
        assert result.text == "Hello world"
