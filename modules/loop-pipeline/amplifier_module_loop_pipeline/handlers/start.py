"""Start node handler.

Returns SUCCESS immediately. The start node (shape=Mdiamond) is the
entry point of the pipeline and performs no work.

Spec coverage: HSTART-001–002, Section 4.3.
"""

from __future__ import annotations

from ..context import PipelineContext
from ..graph import Graph, Node
from ..outcome import Outcome, StageStatus


class StartHandler:
    """Handler for start nodes (shape=Mdiamond)."""

    async def execute(
        self,
        node: Node,
        context: PipelineContext,
        graph: Graph,
        logs_root: str,
    ) -> Outcome:
        """Return SUCCESS immediately."""
        return Outcome(status=StageStatus.SUCCESS, notes=f"Start node: {node.id}")
