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
from .outcome import Outcome, StageStatus
from .validation import validate_or_raise

logger = logging.getLogger(__name__)


class DirectProviderBackend:
    """Backend that calls a provider directly for each codergen node.

    This is the default backend when no session.spawn capability is
    available.  It builds a single-turn ChatRequest from the node's
    prompt and calls ``provider.complete()``, returning an Outcome.

    This is intentionally simple — one LLM call per node, no tool
    execution.  It proves the pipeline-to-real-API chain works and
    can be replaced with a full spawn-based backend later.
    """

    def __init__(
        self,
        provider: Any,
        tools: dict[str, Any] | None = None,
        hooks: Any = None,
        coordinator: Any = None,
    ) -> None:
        self._provider = provider
        self._tools = tools or {}
        self._hooks = hooks
        self._coordinator = coordinator

    async def run(
        self,
        node: Any,
        prompt: str,
        context: PipelineContext,
        **kwargs: Any,
    ) -> Outcome:
        """Call the provider with a single-turn request for *node*.

        Builds a ChatRequest from *prompt*, calls the provider, and
        extracts text from the response content blocks.
        """
        from amplifier_core import ChatRequest, Message

        messages = [Message(role="user", content=prompt)]

        request = ChatRequest(
            messages=messages,
            reasoning_effort=node.attrs.get("reasoning_effort", "high"),
        )

        try:
            response = await self._provider.complete(request)
        except Exception as exc:
            logger.warning("Provider call failed for node %s: %s", node.id, exc)
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason=str(exc),
            )

        # Extract text from response content blocks
        text = ""
        content = getattr(response, "content", None)
        if content:
            for block in content:
                if hasattr(block, "text"):
                    text += block.text

        return Outcome(
            status=StageStatus.SUCCESS,
            notes=f"Stage completed: {node.id}",
            context_updates={
                "last_stage": node.id,
                "last_response": text[:200] if text else "",
            },
        )


def _build_backend(
    providers: dict[str, Any],
    tools: dict[str, Any],
    hooks: Any,
    coordinator: Any | None,
) -> Any | None:
    """Auto-construct a backend from the available providers.

    Resolution order:
    1. If coordinator exposes ``session.spawn`` → use AmplifierBackend
       (full "sessions all the way down").
    2. Else if at least one provider is available → use
       DirectProviderBackend (single-turn LLM call per node).
    3. Otherwise → return None (codergen handler falls through to
       simulation mode).
    """
    # Try the full spawn-based backend first
    if coordinator is not None:
        spawn_fn = None
        if hasattr(coordinator, "get_capability"):
            try:
                spawn_fn = coordinator.get_capability("session.spawn")
            except Exception:
                pass
        if spawn_fn is not None:
            from .backend import AmplifierBackend

            logger.info("Using AmplifierBackend (session.spawn available)")
            return AmplifierBackend(coordinator, profiles={})

    # Fall back to direct provider calls
    if providers:
        provider = next(iter(providers.values()))
        logger.info("Using DirectProviderBackend (calling provider directly)")
        return DirectProviderBackend(provider, tools, hooks, coordinator)

    logger.warning(
        "No providers available — codergen nodes will run in simulation mode"
    )
    return None


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

        # 6. Resolve backend: explicit kwarg → auto-construct from providers
        coordinator = kwargs.get("coordinator")
        backend = kwargs.get("backend")
        if backend is None:
            backend = _build_backend(providers, tools, hooks, coordinator)

        # 7. Register handlers
        registry = HandlerRegistry(backend=backend)

        # 8. Run the engine
        engine = PipelineEngine(
            graph=graph,
            context=pipeline_context,
            handler_registry=registry,
            logs_root=logs_root,
            hooks=hooks,
        )
        outcome = await engine.run(goal=prompt or None)

        # 9. Return the final outcome as JSON
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
