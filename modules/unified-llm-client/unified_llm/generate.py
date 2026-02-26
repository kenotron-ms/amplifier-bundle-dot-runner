"""High-level API functions (Spec §4.3-4.7).

generate(), stream(), generate_object(), stream_object().
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from unified_llm.client import Client, get_default_client
from unified_llm.errors import (
    AbortError,
    ConfigurationError,
    NoObjectGeneratedError,
    RequestTimeoutError,
)
from unified_llm.retry import RetryPolicy, retry
from unified_llm.types import (
    GenerateResult,
    Message,
    Request,
    Response,
    ResponseFormat,
    StepResult,
    StreamAccumulator,
    StreamEvent,
    StreamEventType,
    TimeoutConfig,
    Tool,
    ToolCall,
    ToolChoice,
    ToolResult,
)


# ---------------------------------------------------------------------------
# Abort + Timeout — Spec §4.7
# ---------------------------------------------------------------------------


class AbortSignal:
    """Cooperative cancellation signal (Spec §4.7).

    Check .aborted to see if cancellation was requested.
    """

    def __init__(self) -> None:
        self._aborted = False
        self._event = asyncio.Event()

    @property
    def aborted(self) -> bool:
        return self._aborted

    def _set_aborted(self) -> None:
        self._aborted = True
        self._event.set()

    async def wait(self) -> None:
        """Wait until abort is signaled."""
        await self._event.wait()


class AbortController:
    """Controls an AbortSignal (Spec §4.7).

    Usage:
        controller = AbortController()
        result = await generate(..., abort_signal=controller.signal)
        # In another coroutine:
        controller.abort()
    """

    def __init__(self) -> None:
        self._signal = AbortSignal()

    @property
    def signal(self) -> AbortSignal:
        return self._signal

    def abort(self) -> None:
        """Signal cancellation."""
        self._signal._set_aborted()


def _resolve_timeout(timeout: float | TimeoutConfig | None) -> TimeoutConfig | None:
    """Normalize timeout parameter to TimeoutConfig."""
    if timeout is None:
        return None
    if isinstance(timeout, (int, float)):
        return TimeoutConfig(total=float(timeout))
    if isinstance(timeout, TimeoutConfig):
        return timeout
    return None


async def _with_abort_and_timeout(
    coro: Any,
    abort_signal: AbortSignal | None,
    timeout_config: TimeoutConfig | None,
    is_per_step: bool = False,
) -> Any:
    """Wrap a coroutine with abort signal and timeout support."""
    # Check if already aborted
    if abort_signal and abort_signal.aborted:
        raise AbortError("Operation aborted")

    # Determine timeout value
    timeout_val: float | None = None
    if timeout_config:
        if is_per_step and timeout_config.per_step is not None:
            timeout_val = timeout_config.per_step
        elif not is_per_step and timeout_config.total is not None:
            timeout_val = timeout_config.total
        elif timeout_config.total is not None:
            timeout_val = timeout_config.total

    tasks: list[asyncio.Task[Any]] = []
    main_task = asyncio.ensure_future(coro)
    tasks.append(main_task)

    if abort_signal:
        abort_task = asyncio.ensure_future(abort_signal.wait())
        tasks.append(abort_task)

    try:
        done, pending = await asyncio.wait(
            tasks,
            timeout=timeout_val,
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cancel pending tasks
        for t in pending:
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

        # Check what completed
        if main_task in done:
            return main_task.result()

        # Abort signal fired
        if abort_signal and any(t is not main_task for t in done):
            raise AbortError("Operation aborted")

        # Timeout
        raise RequestTimeoutError("Operation timed out")

    except asyncio.CancelledError:
        main_task.cancel()
        raise AbortError("Operation cancelled")


async def generate(
    model: str,
    *,
    prompt: str | None = None,
    messages: list[Message] | None = None,
    system: str | None = None,
    tools: list[Tool] | None = None,
    tool_choice: ToolChoice | None = None,
    max_tool_rounds: int = 1,
    stop_when: Any | None = None,
    response_format: Any | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    stop_sequences: list[str] | None = None,
    reasoning_effort: str | None = None,
    provider: str | None = None,
    provider_options: dict[str, Any] | None = None,
    max_retries: int = 2,
    timeout: float | Any | None = None,
    abort_signal: Any | None = None,
    client: Client | None = None,
) -> GenerateResult:
    """Primary blocking generation function (Spec §4.3).

    Wraps Client.complete() with tool execution loops, retry, timeout.
    """
    # Validate prompt/messages
    if prompt is not None and messages is not None:
        raise ConfigurationError(
            "Cannot specify both 'prompt' and 'messages'. Use one or the other."
        )

    # Resolve client
    resolved_client = client or get_default_client()

    # Build message list
    msg_list: list[Message] = []
    if system:
        msg_list.append(Message.system(system))
    if prompt is not None:
        msg_list.append(Message.user(prompt))
    elif messages is not None:
        msg_list.extend(messages)
    else:
        raise ConfigurationError("Either 'prompt' or 'messages' must be provided.")

    # Build base request template (immutable params across steps)
    base_kwargs: dict[str, Any] = dict(
        model=model,
        provider=provider,
        tools=tools,
        tool_choice=tool_choice,
        response_format=response_format,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop_sequences=stop_sequences,
        reasoning_effort=reasoning_effort,
        provider_options=provider_options,
    )

    # Resolve abort/timeout
    timeout_config = _resolve_timeout(timeout)
    signal: AbortSignal | None = (
        abort_signal if isinstance(abort_signal, AbortSignal) else None
    )

    # Check pre-aborted
    if signal and signal.aborted:
        raise AbortError("Operation aborted")

    # Retry policy
    policy = RetryPolicy(max_retries=max_retries)

    async def _run_generate() -> GenerateResult:
        # Tool loop
        steps: list[StepResult] = []
        conversation = list(msg_list)

        for round_num in range(max_tool_rounds + 1):
            # Check abort between steps
            if signal and signal.aborted:
                raise AbortError("Operation aborted")

            # Each step's LLM call is retried independently
            step_request = Request(messages=conversation, **base_kwargs)

            step_coro = retry(
                resolved_client.complete,
                policy,
                step_request,
            )

            # Apply per-step timeout if configured
            per_step = (
                timeout_config.per_step
                if timeout_config and timeout_config.per_step
                else None
            )
            if per_step is not None:
                try:
                    response: Response = await asyncio.wait_for(
                        step_coro, timeout=per_step
                    )
                except asyncio.TimeoutError:
                    raise RequestTimeoutError("Per-step timeout exceeded") from None
            else:
                response = await step_coro

            # Extract tool calls from response
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.name,
                    arguments=tc.arguments if isinstance(tc.arguments, dict) else {},
                )
                for tc in response.tool_calls
            ]

            # Execute active tools if model wants to call them
            tool_results: list[ToolResult] = []
            has_active_tools = tools and any(t.execute is not None for t in tools)
            if (
                tool_calls
                and response.finish_reason.reason == "tool_calls"
                and has_active_tools
            ):
                tool_results = await _execute_tools(tools or [], tool_calls)

            step = StepResult(
                text=response.text,
                reasoning=response.reasoning,
                tool_calls=tool_calls,
                tool_results=tool_results,
                finish_reason=response.finish_reason,
                usage=response.usage,
                response=response,
                warnings=response.warnings,
            )
            steps.append(step)

            # Check stop conditions
            if not tool_calls or response.finish_reason.reason != "tool_calls":
                break
            if not has_active_tools:
                break  # Passive tools — return tool_calls without looping
            if round_num >= max_tool_rounds:
                break
            if stop_when is not None and stop_when(steps):
                break

            # Continue conversation with tool results
            conversation.append(response.message)
            for tr in tool_results:
                conversation.append(
                    Message.tool_result(
                        tool_call_id=tr.tool_call_id,
                        content=tr.content
                        if isinstance(tr.content, str)
                        else str(tr.content),
                        is_error=tr.is_error,
                    )
                )

        # Aggregate results
        final = steps[-1]
        total_usage = steps[0].usage
        for s in steps[1:]:
            total_usage = total_usage + s.usage

        return GenerateResult(
            text=final.text,
            reasoning=final.reasoning,
            tool_calls=final.tool_calls,
            tool_results=final.tool_results,
            finish_reason=final.finish_reason,
            usage=final.usage,
            total_usage=total_usage,
            steps=steps,
            response=final.response,
        )

    # Apply abort + timeout wrapper if needed
    if signal or timeout_config:
        return await _with_abort_and_timeout(_run_generate(), signal, timeout_config)
    return await _run_generate()


# ---------------------------------------------------------------------------
# stream() — Spec §4.4
# ---------------------------------------------------------------------------


def _build_messages(
    *,
    prompt: str | None,
    messages: list[Message] | None,
    system: str | None,
) -> list[Message]:
    """Shared prompt/messages/system standardization for generate() and stream()."""
    if prompt is not None and messages is not None:
        raise ConfigurationError(
            "Cannot specify both 'prompt' and 'messages'. Use one or the other."
        )

    msg_list: list[Message] = []
    if system:
        msg_list.append(Message.system(system))
    if prompt is not None:
        msg_list.append(Message.user(prompt))
    elif messages is not None:
        msg_list.extend(messages)
    else:
        raise ConfigurationError("Either 'prompt' or 'messages' must be provided.")
    return msg_list


def _build_base_kwargs(
    *,
    model: str,
    provider: str | None,
    tools: list[Tool] | None,
    tool_choice: ToolChoice | None,
    response_format: Any | None,
    temperature: float | None,
    top_p: float | None,
    max_tokens: int | None,
    stop_sequences: list[str] | None,
    reasoning_effort: str | None,
    stream_validation_mode: str | None,
    provider_options: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build shared request kwargs for generate()/stream()."""
    return dict(
        model=model,
        provider=provider,
        tools=tools,
        tool_choice=tool_choice,
        response_format=response_format,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop_sequences=stop_sequences,
        reasoning_effort=reasoning_effort,
        stream_validation_mode=stream_validation_mode,
        provider_options=provider_options,
    )


class StreamResult:
    """Async iterable wrapper over streaming events (Spec §4.4).

    Provides:
    - Async iteration over StreamEvent objects
    - .response() — accumulated Response after stream ends
    - .text_stream — async iterable yielding only text delta strings
    """

    def __init__(
        self,
        event_source: AsyncIterator[StreamEvent],
    ) -> None:
        self._source = event_source
        self._accumulator = StreamAccumulator()
        self._consumed = False

    def __aiter__(self) -> StreamResult:
        return self

    async def __anext__(self) -> StreamEvent:
        try:
            event = await self._source.__anext__()
            self._accumulator.process(event)
            return event
        except StopAsyncIteration:
            self._consumed = True
            raise

    def response(self) -> Response:
        """Return accumulated Response. Call after stream is fully consumed."""
        return self._accumulator.response()

    @property
    def text_stream(self) -> _TextStream:
        """Async iterable that yields only text delta strings."""
        return _TextStream(self)


class _TextStream:
    """Async iterable that yields only text deltas from a StreamResult."""

    def __init__(self, stream_result: StreamResult) -> None:
        self._stream = stream_result

    def __aiter__(self) -> _TextStream:
        return self

    async def __anext__(self) -> str:
        while True:
            event = await self._stream.__anext__()
            if event.type == StreamEventType.TEXT_DELTA and event.delta:
                return event.delta


def stream(
    model: str,
    *,
    prompt: str | None = None,
    messages: list[Message] | None = None,
    system: str | None = None,
    tools: list[Tool] | None = None,
    tool_choice: ToolChoice | None = None,
    max_tool_rounds: int = 1,
    stop_when: Any | None = None,
    response_format: Any | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    stop_sequences: list[str] | None = None,
    reasoning_effort: str | None = None,
    stream_validation_mode: str | None = None,
    provider: str | None = None,
    provider_options: dict[str, Any] | None = None,
    max_retries: int = 2,
    timeout: float | Any | None = None,
    abort_signal: Any | None = None,
    client: Client | None = None,
) -> StreamResult:
    """Primary streaming generation function (Spec §4.4).

    Returns a StreamResult — an async iterable over StreamEvent objects.
    Supports tool execution loops between streaming steps.
    """
    msg_list = _build_messages(prompt=prompt, messages=messages, system=system)
    resolved_client = client or get_default_client()

    base_kwargs = _build_base_kwargs(
        model=model,
        provider=provider,
        tools=tools,
        tool_choice=tool_choice,
        response_format=response_format,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop_sequences=stop_sequences,
        reasoning_effort=reasoning_effort,
        stream_validation_mode=stream_validation_mode,
        provider_options=provider_options,
    )

    policy = RetryPolicy(max_retries=max_retries)
    has_active_tools = tools and any(t.execute is not None for t in tools)
    timeout_config = _resolve_timeout(timeout)
    signal: AbortSignal | None = (
        abort_signal if isinstance(abort_signal, AbortSignal) else None
    )

    async def _stream_with_tool_loop() -> AsyncIterator[StreamEvent]:
        conversation = list(msg_list)

        for round_num in range(max_tool_rounds + 1):
            # Check abort between steps
            if signal and signal.aborted:
                raise AbortError("Operation aborted")

            step_request = Request(messages=conversation, **base_kwargs)

            # Retry the initial stream connection (Spec: retry initial, not partial)
            async def _start_and_get_first(
                req: Request,
            ) -> tuple[StreamEvent, AsyncIterator[StreamEvent]]:
                it = aiter(resolved_client.stream(req))
                first = await anext(it)
                return first, it

            first_event, event_iter = await retry(
                _start_and_get_first, policy, step_request
            )

            # Accumulate this step's events to detect tool calls
            step_accumulator = StreamAccumulator()

            # Yield first event
            step_accumulator.process(first_event)
            yield first_event

            # Yield remaining events (no retry after partial data)
            async for event in event_iter:
                # Check abort during streaming
                if signal and signal.aborted:
                    raise AbortError("Operation aborted")
                step_accumulator.process(event)
                yield event

            # Check if we need to execute tools and loop
            step_response = step_accumulator.response()
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.name,
                    arguments=tc.arguments if isinstance(tc.arguments, dict) else {},
                )
                for tc in step_response.tool_calls
            ]

            if (
                not tool_calls
                or step_response.finish_reason.reason != "tool_calls"
                or not has_active_tools
            ):
                break
            if round_num >= max_tool_rounds:
                break

            # Execute tools
            tool_results = await _execute_tools(tools or [], tool_calls)

            # Emit step_finish event between steps
            yield StreamEvent(
                type="step_finish",
                response=step_response,
            )

            # Continue conversation
            conversation.append(step_response.message)
            for tr in tool_results:
                conversation.append(
                    Message.tool_result(
                        tool_call_id=tr.tool_call_id,
                        content=tr.content
                        if isinstance(tr.content, str)
                        else str(tr.content),
                        is_error=tr.is_error,
                    )
                )

    # Wrap with timeout if configured
    if timeout_config and timeout_config.total is not None:
        return StreamResult(
            _stream_with_timeout(_stream_with_tool_loop(), timeout_config.total, signal)
        )
    return StreamResult(_stream_with_tool_loop())


async def _stream_with_timeout(
    source: AsyncIterator[StreamEvent],
    total_timeout: float,
    signal: AbortSignal | None,
) -> AsyncIterator[StreamEvent]:
    """Wrap a stream with total timeout support."""
    import time

    deadline = time.monotonic() + total_timeout
    async for event in source:
        if signal and signal.aborted:
            raise AbortError("Operation aborted")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RequestTimeoutError("Stream total timeout exceeded")
        yield event


# ---------------------------------------------------------------------------
# generate_object() — Spec §4.5
# ---------------------------------------------------------------------------


async def generate_object(
    model: str,
    *,
    schema: dict[str, Any],
    prompt: str | None = None,
    messages: list[Message] | None = None,
    system: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    stop_sequences: list[str] | None = None,
    reasoning_effort: str | None = None,
    provider: str | None = None,
    provider_options: dict[str, Any] | None = None,
    max_retries: int = 2,
    timeout: float | Any | None = None,
    abort_signal: Any | None = None,
    client: Client | None = None,
) -> GenerateResult:
    """Structured output generation with JSON schema validation (Spec §4.5).

    Sets response_format to json_schema, parses response JSON, validates
    against schema, and sets result.output. Raises NoObjectGeneratedError
    on parse/validation failure. Schema validation failures are NOT retried.
    """
    response_format = ResponseFormat(
        type="json_schema",
        json_schema=schema,
        strict=True,
    )

    result = await generate(
        model=model,
        prompt=prompt,
        messages=messages,
        system=system,
        response_format=response_format,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop_sequences=stop_sequences,
        reasoning_effort=reasoning_effort,
        provider=provider,
        provider_options=provider_options,
        max_retries=max_retries,
        timeout=timeout,
        abort_signal=abort_signal,
        client=client,
        max_tool_rounds=0,  # No tool loop for structured output
    )

    # Parse JSON from response text
    raw_text = result.text.strip()
    parsed = _parse_json_response(raw_text)

    # Validate against schema (basic required-field check)
    _validate_against_schema(parsed, schema)

    # Return result with .output set
    return GenerateResult(
        text=result.text,
        reasoning=result.reasoning,
        tool_calls=result.tool_calls,
        tool_results=result.tool_results,
        finish_reason=result.finish_reason,
        usage=result.usage,
        total_usage=result.total_usage,
        steps=result.steps,
        response=result.response,
        output=parsed,
    )


def _parse_json_response(text: str) -> Any:
    """Parse JSON from response text, stripping markdown fences if present."""
    import json
    import re

    # Strip markdown code fences
    stripped = text.strip()
    fence_match = re.match(r"^```(?:json)?\s*\n(.*?)\n```$", stripped, re.DOTALL)
    if fence_match:
        stripped = fence_match.group(1).strip()

    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError) as exc:
        raise NoObjectGeneratedError(
            f"Failed to parse JSON from response: {exc}",
        ) from exc


def _validate_against_schema(obj: Any, schema: dict[str, Any]) -> None:
    """Basic schema validation: check required fields and root type."""
    if schema.get("type") == "object" and isinstance(obj, dict):
        required = schema.get("required", [])
        missing = [r for r in required if r not in obj]
        if missing:
            raise NoObjectGeneratedError(
                f"Response missing required fields: {missing}",
            )
    elif schema.get("type") == "object" and not isinstance(obj, dict):
        raise NoObjectGeneratedError(
            f"Expected object, got {type(obj).__name__}",
        )
    elif schema.get("type") == "array" and not isinstance(obj, list):
        raise NoObjectGeneratedError(
            f"Expected array, got {type(obj).__name__}",
        )


# ---------------------------------------------------------------------------
# stream_object() — Spec §4.6
# ---------------------------------------------------------------------------


class StreamObjectResult:
    """Async iterable that yields partial JSON objects as tokens arrive (Spec §4.6).

    Uses incremental JSON parsing to yield progressively more complete objects.
    After iteration, .object() returns the final validated object.
    """

    def __init__(
        self,
        stream_result: StreamResult,
        schema: dict[str, Any],
    ) -> None:
        self._stream = stream_result
        self._schema = schema
        self._accumulated_text = ""
        self._last_partial: Any | None = None
        self._final_object: Any | None = None
        self._consumed = False

    def __aiter__(self) -> StreamObjectResult:
        return self

    async def __anext__(self) -> Any:
        import json

        while True:
            try:
                event = await self._stream.__anext__()
            except StopAsyncIteration:
                self._consumed = True
                # Try final parse
                if self._accumulated_text.strip():
                    try:
                        self._final_object = _parse_json_response(
                            self._accumulated_text
                        )
                        _validate_against_schema(self._final_object, self._schema)
                    except NoObjectGeneratedError:
                        pass  # Will be raised by .object()
                raise

            if event.type == StreamEventType.TEXT_DELTA and event.delta:
                self._accumulated_text += event.delta
                # Try incremental JSON parse
                try:
                    partial = json.loads(self._accumulated_text)
                    self._last_partial = partial
                    return partial
                except (json.JSONDecodeError, ValueError):
                    # Not valid JSON yet — continue accumulating
                    continue

    def object(self) -> Any:
        """Return the final validated object. Call after stream is consumed."""
        if self._final_object is not None:
            return self._final_object
        # Try parsing the accumulated text
        if self._accumulated_text.strip():
            self._final_object = _parse_json_response(self._accumulated_text)
            _validate_against_schema(self._final_object, self._schema)
            return self._final_object
        raise NoObjectGeneratedError("No JSON content received in stream")

    def response(self) -> Response:
        """Return accumulated Response from underlying stream."""
        return self._stream.response()


def stream_object(
    model: str,
    *,
    schema: dict[str, Any],
    prompt: str | None = None,
    messages: list[Message] | None = None,
    system: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    stop_sequences: list[str] | None = None,
    reasoning_effort: str | None = None,
    provider: str | None = None,
    provider_options: dict[str, Any] | None = None,
    max_retries: int = 2,
    timeout: float | Any | None = None,
    abort_signal: Any | None = None,
    client: Client | None = None,
) -> StreamObjectResult:
    """Streaming structured output with incremental JSON parsing (Spec §4.6).

    Returns an async iterable of partial objects that grow as tokens arrive.
    After iteration, .object() returns the final validated object.
    """
    response_fmt = ResponseFormat(
        type="json_schema",
        json_schema=schema,
        strict=True,
    )

    result = stream(
        model=model,
        prompt=prompt,
        messages=messages,
        system=system,
        response_format=response_fmt,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop_sequences=stop_sequences,
        reasoning_effort=reasoning_effort,
        provider=provider,
        provider_options=provider_options,
        max_retries=max_retries,
        timeout=timeout,
        abort_signal=abort_signal,
        client=client,
        max_tool_rounds=0,  # No tool loop for structured output
    )

    return StreamObjectResult(result, schema)


# ---------------------------------------------------------------------------
# Tool execution — shared by generate() and stream()
# ---------------------------------------------------------------------------


async def _execute_tools(
    tools: list[Tool],
    tool_calls: list[ToolCall],
) -> list[ToolResult]:
    """Execute tool calls concurrently (Spec §5.7).

    All calls are launched concurrently. All results returned, even on partial failure.
    """
    tool_map = {t.name: t for t in tools if t.execute is not None}

    async def execute_one(call: ToolCall) -> ToolResult:
        tool = tool_map.get(call.name)
        if tool is None or tool.execute is None:
            return ToolResult(
                tool_call_id=call.id,
                content=f"Unknown tool: {call.name}",
                is_error=True,
            )
        try:
            result = tool.execute(**call.arguments)
            if asyncio.iscoroutine(result):
                result = await result
            content: str | dict[str, Any] | list[Any]
            if isinstance(result, (str, dict, list)):
                content = result
            else:
                content = str(result)
            return ToolResult(tool_call_id=call.id, content=content, is_error=False)
        except Exception as exc:
            return ToolResult(
                tool_call_id=call.id,
                content=str(exc),
                is_error=True,
            )

    tasks = [execute_one(call) for call in tool_calls]
    results = await asyncio.gather(*tasks)
    return list(results)
