"""Tests for unified_llm.middleware — onion pattern middleware chain."""

import asyncio
from collections.abc import AsyncIterator

from unified_llm.middleware import apply_middleware, apply_streaming_middleware
from unified_llm.types import (
    FinishReason,
    Message,
    Request,
    Response,
    StreamEvent,
    StreamEventType,
    Usage,
)


def _make_request() -> Request:
    return Request(model="test", messages=[Message.user("hi")])


def _make_response() -> Response:
    return Response(
        id="r1", model="test", provider="test",
        message=Message.assistant("hello"),
        finish_reason=FinishReason(reason="stop"),
        usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
    )


class TestApplyMiddleware:
    """Spec §2.3 — Onion/chain-of-responsibility middleware."""

    def test_no_middleware(self) -> None:
        """Base handler called directly when no middleware."""
        async def handler(req: Request) -> Response:
            return _make_response()

        result = asyncio.run(apply_middleware([], handler, _make_request()))
        assert result.text == "hello"

    def test_single_middleware(self) -> None:
        order: list[str] = []

        async def mw(request: Request, next_fn):
            order.append("mw_request")
            response = await next_fn(request)
            order.append("mw_response")
            return response

        async def handler(req: Request) -> Response:
            order.append("handler")
            return _make_response()

        asyncio.run(apply_middleware([mw], handler, _make_request()))
        assert order == ["mw_request", "handler", "mw_response"]

    def test_execution_order(self) -> None:
        """Spec: Registration order for request, reverse for response."""
        order: list[str] = []

        async def mw_a(request, next_fn):
            order.append("A_req")
            response = await next_fn(request)
            order.append("A_resp")
            return response

        async def mw_b(request, next_fn):
            order.append("B_req")
            response = await next_fn(request)
            order.append("B_resp")
            return response

        async def handler(req):
            order.append("handler")
            return _make_response()

        asyncio.run(apply_middleware([mw_a, mw_b], handler, _make_request()))
        assert order == ["A_req", "B_req", "handler", "B_resp", "A_resp"]

    def test_middleware_can_modify_request(self) -> None:
        async def add_temp(request, next_fn):
            request.temperature = 0.5
            return await next_fn(request)

        captured_temp = None

        async def handler(req):
            nonlocal captured_temp
            captured_temp = req.temperature
            return _make_response()

        asyncio.run(apply_middleware([add_temp], handler, _make_request()))
        assert captured_temp == 0.5


class TestApplyStreamingMiddleware:
    """Spec §2.3 — Streaming middleware wraps the event iterator."""

    def test_streaming_passthrough(self) -> None:
        events = [
            StreamEvent(type=StreamEventType.TEXT_DELTA, delta="hi"),
            StreamEvent(type=StreamEventType.FINISH),
        ]

        async def handler(req: Request) -> AsyncIterator[StreamEvent]:
            for e in events:
                yield e

        async def run() -> list[StreamEvent]:
            result = []
            async for evt in apply_streaming_middleware([], handler, _make_request()):
                result.append(evt)
            return result

        result = asyncio.run(run())
        assert len(result) == 2

    def test_streaming_middleware_observes_events(self) -> None:
        seen: list[str] = []

        async def logger_mw(request, next_fn):
            async for event in next_fn(request):
                if event.delta:
                    seen.append(event.delta)
                yield event

        async def handler(req):
            yield StreamEvent(type=StreamEventType.TEXT_DELTA, delta="hello")
            yield StreamEvent(type=StreamEventType.FINISH)

        async def run() -> list[StreamEvent]:
            result = []
            async for evt in apply_streaming_middleware([logger_mw], handler, _make_request()):
                result.append(evt)
            return result

        asyncio.run(run())
        assert seen == ["hello"]
