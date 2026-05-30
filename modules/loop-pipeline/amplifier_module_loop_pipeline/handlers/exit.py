"""Exit node handler.

Returns SUCCESS immediately. The exit node (shape=Msquare) is the
terminal node of the pipeline.

Spec coverage: HEXIT-001–003, Section 4.4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..context import PipelineContext

if TYPE_CHECKING:
    from ..engine import PipelineEngine
from ..graph import Graph, Node
from ..outcome import Outcome, StageStatus


class ExitHandler:
    """Handler for exit nodes (shape=Msquare)."""

    async def execute(
        self,
        node: Node,
        context: PipelineContext,
        graph: Graph,
        logs_root: str,
        *,
        engine: "PipelineEngine | None" = None,
    ) -> Outcome:
        """Return SUCCESS immediately."""
        return Outcome(status=StageStatus.SUCCESS, notes=f"Exit node: {node.id}")
