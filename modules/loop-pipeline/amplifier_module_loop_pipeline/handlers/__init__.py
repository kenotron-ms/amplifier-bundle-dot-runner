"""Handler registry and base protocol for pipeline node handlers.

Each node type (start, exit, codergen, conditional, tool, etc.) has a
handler that implements the NodeHandler protocol. The HandlerRegistry
maps nodes to their handlers based on the node's type attribute or
shape-to-handler-type mapping.

Spec coverage: HAND-001–007, Section 4.1–4.2.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..context import PipelineContext
from ..graph import Graph, Node
from ..outcome import Outcome
from ..validation import SHAPE_TO_HANDLER


@runtime_checkable
class NodeHandler(Protocol):
    """Protocol for pipeline node handlers.

    Spec Section 4.1: Handler Interface.
    """

    async def execute(
        self,
        node: Node,
        context: PipelineContext,
        graph: Graph,
        logs_root: str,
    ) -> Outcome: ...


class HandlerRegistry:
    """Maps nodes to their handlers.

    Resolution order:
    1. Node's explicit ``type`` attribute (e.g. type="conditional")
    2. Shape-to-handler-type mapping (spec Section 2.8)
    3. Default: codergen

    Spec Section 4.2: Handler Registry.
    """

    def __init__(self, **kwargs: Any) -> None:
        from .codergen import CodergenHandler
        from .conditional import ConditionalHandler
        from .exit import ExitHandler
        from .fan_in import FanInHandler
        from .human import HumanGateHandler
        from .manager_loop import ManagerLoopHandler
        from .parallel import ParallelHandler
        from .start import StartHandler
        from .tool import ToolHandler

        self._hooks = kwargs.get("hooks")

        self._handlers: dict[str, NodeHandler] = {
            "start": StartHandler(),
            "exit": ExitHandler(),
            "codergen": CodergenHandler(backend=kwargs.get("backend")),
            "conditional": ConditionalHandler(),
            "tool": ToolHandler(),
            "wait.human": HumanGateHandler(
                interviewer=kwargs.get("interviewer"),
                hooks=self._hooks,
            ),
            "stack.manager_loop": ManagerLoopHandler(
                subgraph_runner=kwargs.get("subgraph_runner"),
            ),
            "parallel": ParallelHandler(
                subgraph_runner=kwargs.get("subgraph_runner"),
                hooks=self._hooks,
            ),
            "parallel.fan_in": FanInHandler(),
        }

    def get(self, node: Node) -> NodeHandler:
        """Resolve the handler for a node.

        Uses the node's explicit type first, then shape mapping,
        falling back to codergen.
        """
        handler_type = node.type or SHAPE_TO_HANDLER.get(node.shape, "codergen")
        return self._handlers.get(handler_type, self._handlers["codergen"])

    def register(self, handler_type: str, handler: NodeHandler) -> None:
        """Register a custom handler for a handler type."""
        self._handlers[handler_type] = handler
