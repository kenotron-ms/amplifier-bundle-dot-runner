"""Middleware chain with onion/chain-of-responsibility pattern (Spec §2.3).

Request phase: registration order (first registered = first to execute).
Response phase: reverse order.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from unified_llm.types import Request, Response, StreamEvent

# Middleware type covers both blocking (-> Awaitable[Response]) and
# streaming (-> AsyncIterator[StreamEvent]) paths, so we use Any return.
Middleware = Callable[..., Any]


async def apply_middleware(
    middleware: list[Middleware],
    handler: Callable[[Request], Awaitable[Response]],
    request: Request,
) -> Response:
    """Apply middleware chain to a blocking complete() call."""
    if not middleware:
        return await handler(request)

    async def build_chain(index: int, req: Request) -> Response:
        if index >= len(middleware):
            return await handler(req)
        return await middleware[index](req, lambda r: build_chain(index + 1, r))

    return await build_chain(0, request)


async def apply_streaming_middleware(
    middleware: list[Middleware],
    handler: Callable[[Request], AsyncIterator[StreamEvent]],
    request: Request,
) -> AsyncIterator[StreamEvent]:
    """Apply middleware chain to a streaming call."""
    if not middleware:
        async for event in handler(request):
            yield event
        return

    async def build_chain(
        index: int,
        req: Request,
    ) -> AsyncIterator[StreamEvent]:
        if index >= len(middleware):
            async for event in handler(req):
                yield event
        else:
            async for event in middleware[index](
                req, lambda r: build_chain(index + 1, r)
            ):
                yield event

    async for event in build_chain(0, request):
        yield event
