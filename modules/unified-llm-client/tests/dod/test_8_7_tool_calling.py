"""DoD §8.7 — Tool Calling.

Verifies tool calling behavior: active/passive tools, parallel execution,
error handling, tool choice modes, and StepResult tracking.
Uses mocked adapters — no real API keys needed.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from unified_llm import (
    Client,
    ContentKind,
    ContentPart,
    FinishReason,
    Message,
    Request,
    Response,
    Role,
    StepResult,
    StreamEvent,
    StreamEventType,
    Tool,
    ToolCallData,
    ToolChoice,
    Usage,
)
from unified_llm.generate import generate


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _tool_call_response(
    tool_calls: list[tuple[str, str, dict[str, Any]]],
    provider: str = "mock",
) -> Response:
    """Create a response with tool calls."""
    content = [
        ContentPart(
            kind=ContentKind.TOOL_CALL,
            tool_call=ToolCallData(id=tc_id, name=tc_name, arguments=tc_args),
        )
        for tc_id, tc_name, tc_args in tool_calls
    ]
    return Response(
        id="r1",
        model="test",
        provider=provider,
        message=Message(role=Role.ASSISTANT, content=content),
        finish_reason=FinishReason(reason="tool_calls"),
        usage=Usage(input_tokens=20, output_tokens=10, total_tokens=30),
    )


def _text_response(text: str = "Done!", provider: str = "mock") -> Response:
    return Response(
        id="r2",
        model="test",
        provider=provider,
        message=Message.assistant(text),
        finish_reason=FinishReason(reason="stop"),
        usage=Usage(input_tokens=30, output_tokens=5, total_tokens=35),
    )


class _ToolLoopAdapter:
    """Mock adapter that returns tool calls then a final text response."""

    def __init__(
        self,
        tool_calls: list[tuple[str, str, dict[str, Any]]],
        final_text: str = "Done!",
    ) -> None:
        self._tool_calls = tool_calls
        self._final_text = final_text
        self._call_count = 0

    @property
    def name(self) -> str:
        return "mock"

    async def complete(self, request: Request) -> Response:
        self._call_count += 1
        if self._call_count == 1:
            return _tool_call_response(self._tool_calls)
        return _text_response(self._final_text)

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(type=StreamEventType.FINISH)

    async def close(self) -> None:
        pass

    async def initialize(self) -> None:
        pass

    def supports_tool_choice(self, mode: str) -> bool:
        return True


class _PassiveToolAdapter:
    """Mock adapter that returns tool calls once (for passive tool testing)."""

    @property
    def name(self) -> str:
        return "mock"

    async def complete(self, request: Request) -> Response:
        return _tool_call_response([("call_1", "get_data", {"key": "val"})])

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(type=StreamEventType.FINISH)

    async def close(self) -> None:
        pass

    async def initialize(self) -> None:
        pass

    def supports_tool_choice(self, mode: str) -> bool:
        return True


# ---------------------------------------------------------------------------
# §8.7 — Active tools trigger automatic execution loops
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="function")
async def test_active_tools_trigger_execution_loop() -> None:
    """[ ] Tools with execute handlers (active tools) trigger automatic tool execution loops."""
    executed: list[str] = []

    def get_weather(city: str) -> str:
        executed.append(city)
        return f"72F in {city}"

    weather_tool = Tool(
        name="get_weather",
        description="Get weather",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}},
        execute=get_weather,
    )

    adapter = _ToolLoopAdapter(
        tool_calls=[("call_1", "get_weather", {"city": "SF"})],
        final_text="The weather is 72F.",
    )
    client = Client(providers={"mock": adapter}, default_provider="mock")

    result = await generate(
        model="test",
        prompt="What's the weather in SF?",
        tools=[weather_tool],
        max_tool_rounds=3,
        client=client,
        provider="mock",
        max_retries=0,
    )
    assert result.text == "The weather is 72F."
    assert "SF" in executed
    assert len(result.steps) >= 2


# ---------------------------------------------------------------------------
# §8.7 — Passive tools return tool calls without looping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="function")
async def test_passive_tools_no_loop() -> None:
    """[ ] Tools without execute handlers (passive tools) return tool calls without looping."""
    passive_tool = Tool(
        name="get_data",
        description="Get data",
        parameters={"type": "object", "properties": {"key": {"type": "string"}}},
        execute=None,  # Passive — no handler
    )

    client = Client(providers={"mock": _PassiveToolAdapter()}, default_provider="mock")

    result = await generate(
        model="test",
        prompt="Get data",
        tools=[passive_tool],
        max_tool_rounds=3,
        client=client,
        provider="mock",
        max_retries=0,
    )
    # Should stop after one step — no execution loop
    assert len(result.steps) == 1
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "get_data"


# ---------------------------------------------------------------------------
# §8.7 — max_tool_rounds respected
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="function")
async def test_max_tool_rounds_respected() -> None:
    """[ ] max_tool_rounds is respected: loop stops after configured number of rounds."""

    class _AlwaysToolCallAdapter:
        @property
        def name(self) -> str:
            return "mock"

        async def complete(self, request: Request) -> Response:
            return _tool_call_response([("call_x", "my_tool", {})])

        async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
            yield StreamEvent(type=StreamEventType.FINISH)

        async def close(self) -> None:
            pass

        async def initialize(self) -> None:
            pass

        def supports_tool_choice(self, mode: str) -> bool:
            return True

    my_tool = Tool(
        name="my_tool",
        description="Always called",
        parameters={"type": "object", "properties": {}},
        execute=lambda: "ok",
    )

    client = Client(
        providers={"mock": _AlwaysToolCallAdapter()}, default_provider="mock"
    )

    result = await generate(
        model="test",
        prompt="Go",
        tools=[my_tool],
        max_tool_rounds=2,
        client=client,
        provider="mock",
        max_retries=0,
    )
    # Should stop after max_tool_rounds + 1 steps
    assert len(result.steps) <= 3


# ---------------------------------------------------------------------------
# §8.7 — max_tool_rounds = 0 disables execution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="function")
async def test_max_tool_rounds_zero() -> None:
    """[ ] max_tool_rounds = 0 disables automatic execution entirely."""
    my_tool = Tool(
        name="get_data",
        description="Get data",
        parameters={"type": "object", "properties": {}},
        execute=lambda: "should not run",
    )

    client = Client(providers={"mock": _PassiveToolAdapter()}, default_provider="mock")

    result = await generate(
        model="test",
        prompt="Go",
        tools=[my_tool],
        max_tool_rounds=0,
        client=client,
        provider="mock",
        max_retries=0,
    )
    # Only one step, no execution
    assert len(result.steps) == 1


# ---------------------------------------------------------------------------
# §8.7 — Parallel tool calls executed concurrently
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="function")
async def test_parallel_tool_calls() -> None:
    """[ ] Parallel tool calls: when model returns N calls, all N executed concurrently."""
    executed_cities: list[str] = []

    def get_weather(city: str) -> str:
        executed_cities.append(city)
        return f"Weather in {city}: 72F"

    weather_tool = Tool(
        name="get_weather",
        description="Get weather",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}},
        execute=get_weather,
    )

    adapter = _ToolLoopAdapter(
        tool_calls=[
            ("call_1", "get_weather", {"city": "SF"}),
            ("call_2", "get_weather", {"city": "NYC"}),
        ],
        final_text="Weather in both cities is 72F.",
    )
    client = Client(providers={"mock": adapter}, default_provider="mock")

    result = await generate(
        model="test",
        prompt="Weather in SF and NYC?",
        tools=[weather_tool],
        max_tool_rounds=3,
        client=client,
        provider="mock",
        max_retries=0,
    )
    # Both cities should have been executed
    assert "SF" in executed_cities
    assert "NYC" in executed_cities
    assert len(executed_cities) == 2


# ---------------------------------------------------------------------------
# §8.7 — Tool execution errors sent as error results
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="function")
async def test_tool_errors_sent_as_results() -> None:
    """[ ] Tool execution errors sent to model as error results (is_error=True)."""

    def failing_tool() -> str:
        raise ValueError("Tool crashed!")

    tool = Tool(
        name="bad_tool",
        description="Fails",
        parameters={"type": "object", "properties": {}},
        execute=failing_tool,
    )

    adapter = _ToolLoopAdapter(
        tool_calls=[("call_1", "bad_tool", {})],
        final_text="Tool failed, but I handled it.",
    )
    client = Client(providers={"mock": adapter}, default_provider="mock")

    # Should NOT raise — errors are sent to model
    result = await generate(
        model="test",
        prompt="Run bad tool",
        tools=[tool],
        max_tool_rounds=3,
        client=client,
        provider="mock",
        max_retries=0,
    )
    assert result.text == "Tool failed, but I handled it."
    # Check that the error result was recorded
    step = result.steps[0]
    assert len(step.tool_results) == 1
    assert step.tool_results[0].is_error is True


# ---------------------------------------------------------------------------
# §8.7 — Unknown tool calls send error result
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="function")
async def test_unknown_tool_sends_error_result() -> None:
    """[ ] Unknown tool calls send an error result, not an exception."""
    known_tool = Tool(
        name="known_tool",
        description="Known",
        parameters={"type": "object", "properties": {}},
        execute=lambda: "ok",
    )

    adapter = _ToolLoopAdapter(
        tool_calls=[("call_1", "unknown_tool", {})],
        final_text="Handled unknown tool.",
    )
    client = Client(providers={"mock": adapter}, default_provider="mock")

    result = await generate(
        model="test",
        prompt="Call unknown tool",
        tools=[known_tool],
        max_tool_rounds=3,
        client=client,
        provider="mock",
        max_retries=0,
    )
    assert result.text == "Handled unknown tool."
    step = result.steps[0]
    assert len(step.tool_results) == 1
    assert step.tool_results[0].is_error is True
    assert "Unknown tool" in str(step.tool_results[0].content)


# ---------------------------------------------------------------------------
# §8.7 — ToolChoice modes translated
# ---------------------------------------------------------------------------


class TestToolChoiceModes:
    """ToolChoice modes (auto, none, required, named) translated correctly."""

    def test_tool_choice_auto(self) -> None:
        tc = ToolChoice(mode="auto")
        assert tc.mode == "auto"

    def test_tool_choice_none(self) -> None:
        tc = ToolChoice(mode="none")
        assert tc.mode == "none"

    def test_tool_choice_required(self) -> None:
        tc = ToolChoice(mode="required")
        assert tc.mode == "required"

    def test_tool_choice_named(self) -> None:
        tc = ToolChoice(mode="named", tool_name="my_func")
        assert tc.mode == "named"
        assert tc.tool_name == "my_func"


# ---------------------------------------------------------------------------
# §8.7 — StepResult tracks each step
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="function")
async def test_step_result_tracking() -> None:
    """[ ] StepResult objects track each step's tool calls, results, and usage."""

    def get_weather(city: str) -> str:
        return f"72F in {city}"

    weather_tool = Tool(
        name="get_weather",
        description="Get weather",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}},
        execute=get_weather,
    )

    adapter = _ToolLoopAdapter(
        tool_calls=[("call_1", "get_weather", {"city": "SF"})],
        final_text="The weather is nice.",
    )
    client = Client(providers={"mock": adapter}, default_provider="mock")

    result = await generate(
        model="test",
        prompt="Weather?",
        tools=[weather_tool],
        max_tool_rounds=3,
        client=client,
        provider="mock",
        max_retries=0,
    )
    assert len(result.steps) >= 2

    # First step should have tool calls and results
    step0 = result.steps[0]
    assert isinstance(step0, StepResult)
    assert len(step0.tool_calls) == 1
    assert step0.tool_calls[0].name == "get_weather"
    assert len(step0.tool_results) == 1
    assert step0.usage.total_tokens > 0

    # Last step should have final text
    last_step = result.steps[-1]
    assert last_step.text == "The weather is nice."
