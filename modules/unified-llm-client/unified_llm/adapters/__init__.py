"""Provider adapter interface (Spec §2.4, §7.1).

Every provider adapter must implement complete() and stream().
Optional: close(), initialize(), supports_tool_choice().
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from unified_llm.types import Request, Response, StreamEvent


@runtime_checkable
class ProviderAdapter(Protocol):
    """Interface that every provider adapter must implement."""

    @property
    def name(self) -> str:
        """Provider name, e.g. 'openai', 'anthropic', 'gemini'."""
        ...

    async def complete(self, request: Request) -> Response:
        """Send a request, block until done, return full Response. No retry."""
        ...

    def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        """Send a request, return async iterator of StreamEvent. No retry."""
        ...

    async def close(self) -> None:
        """Release resources. Called by Client.close()."""
        ...

    async def initialize(self) -> None:
        """Validate configuration on startup. Called on registration."""
        ...

    def supports_tool_choice(self, mode: str) -> bool:
        """Check if a particular tool choice mode is supported."""
        ...
