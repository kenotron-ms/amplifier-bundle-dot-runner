"""Attractor pipeline orchestrator module.

A DOT graph-driven multi-stage AI workflow engine. Parses directed graphs
(defined in Graphviz DOT syntax) to orchestrate multi-stage AI pipelines
where each node is an AI task and edges define the flow between them.

Implements the Attractor specification (attractor-spec.md).
"""

from __future__ import annotations

# Amplifier module metadata
__amplifier_module_type__ = "orchestrator"

import json
import logging
import os
import tempfile
from typing import Any

from .context import PipelineContext
from .dot_parser import parse_dot
from .engine import PipelineEngine
from .handlers import HandlerRegistry
from .validation import validate_or_raise

logger = logging.getLogger(__name__)


async def mount(coordinator: Any, config: dict[str, Any] | None = None) -> None:
    """Mount the loop-pipeline orchestrator.

    Config options:
        dot_source: Inline DOT digraph string.
        dot_file: Path to a .dot file.
    """
    cfg = config or {}
    orchestrator = PipelineOrchestrator(cfg)
    await coordinator.mount("orchestrator", orchestrator)
    logger.info("loop-pipeline orchestrator mounted")


class PipelineOrchestrator:
    """DOT graph-driven pipeline orchestrator.

    Parses a DOT digraph and walks it node-by-node, executing handlers
    for each node type and selecting edges based on outcomes.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    async def execute(
        self,
        prompt: str,
        context: Any,
        providers: dict[str, Any],
        tools: dict[str, Any],
        hooks: Any,
        **kwargs: Any,
    ) -> str:
        """Execute the pipeline.

        Parses the DOT graph, validates it, and walks from start to exit.

        Returns a JSON string with the pipeline outcome.
        """
        # 1. Get DOT source
        dot_source = self._resolve_dot_source()

        # 2. Parse the DOT graph
        graph = parse_dot(dot_source)

        # 3. Validate the graph
        validate_or_raise(graph)

        # 4. Create pipeline context with goal from the prompt
        pipeline_context = PipelineContext()
        if prompt:
            pipeline_context.set("graph.goal", prompt)

        # 5. Set up logs directory
        logs_root = self.config.get(
            "logs_root", os.path.join(tempfile.gettempdir(), "attractor-pipeline")
        )
        os.makedirs(logs_root, exist_ok=True)

        # 6. Register handlers
        backend = kwargs.get("backend")
        registry = HandlerRegistry(backend=backend)

        # 7. Run the engine
        engine = PipelineEngine(
            graph=graph,
            context=pipeline_context,
            handler_registry=registry,
            logs_root=logs_root,
        )
        outcome = await engine.run(goal=prompt or None)

        # 8. Return the final outcome as JSON
        result = {
            "status": outcome.status.value,
            "notes": outcome.notes,
            "failure_reason": outcome.failure_reason,
        }
        return json.dumps(result)

    def _resolve_dot_source(self) -> str:
        """Resolve DOT source from config (inline or file)."""
        dot_source = self.config.get("dot_source")
        if dot_source:
            return dot_source

        dot_file = self.config.get("dot_file")
        if dot_file:
            with open(dot_file) as f:
                return f.read()

        raise ValueError(
            "No DOT source configured. Set 'dot_source' or 'dot_file' in config."
        )
