"""Conditional node handler.

Returns SUCCESS immediately. Conditional nodes (shape=diamond) perform
no work — routing is handled entirely by edge selection based on the
previous node's outcome and context.

Spec coverage: COND-001, Section 4.7.
"""

from __future__ import annotations

from ..context import PipelineContext
from ..graph import Graph, Node
from ..outcome import Outcome, StageStatus


class ConditionalHandler:
    """Handler for conditional nodes (shape=diamond)."""

    async def execute(
        self,
        node: Node,
        context: PipelineContext,
        graph: Graph,
        logs_root: str,
    ) -> Outcome:
        """Return SUCCESS immediately — routing is via edge conditions."""
        return Outcome(status=StageStatus.SUCCESS, notes=f"Conditional node: {node.id}")
