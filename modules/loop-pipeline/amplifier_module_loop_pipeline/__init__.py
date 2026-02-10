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
    """Backend that calls a provider directly with a mini tool loop.

    This is the default backend when no session.spawn capability is
    available.  It runs an agentic loop — call LLM, execute any tool
    calls, feed results back, repeat — until the model returns a
    text-only response or the max round limit is reached.
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
        """Run a mini agentic tool loop for *node*.

        Builds a ChatRequest from *prompt*, calls the provider, and if
        the response contains tool calls, executes them and feeds the
        results back.  Repeats until the model returns text only or the
        round limit is hit.
        """
        from amplifier_core import ChatRequest, Message

        from .backend import (
            _build_tool_specs,
            _extract_text,
            _extract_tool_calls,
            _build_assistant_message,
            _parse_outcome,
            _MAX_TOOL_LOOP_ROUNDS,
        )

        messages: list[Message] = [Message(role="user", content=prompt)]
        reasoning_effort = node.attrs.get("reasoning_effort")
        tool_specs = _build_tool_specs(self._tools)

        for _round in range(_MAX_TOOL_LOOP_ROUNDS):
            request = ChatRequest(
                messages=messages,
                tools=tool_specs or None,
                tool_choice="auto" if tool_specs else None,
                reasoning_effort=reasoning_effort,
            )

            try:
                response = await self._provider.complete(request)
            except Exception as exc:
                logger.warning(
                    "Provider call failed for node %s (round %d): %s",
                    node.id,
                    _round,
                    exc,
                )
                return Outcome(
                    status=StageStatus.FAIL,
                    failure_reason=str(exc),
                )

            text = _extract_text(response)
            tool_calls = _extract_tool_calls(response, self._provider)

            if not tool_calls:
                # Model is done — parse the final text as an outcome
                if text:
                    outcome = _parse_outcome(text)
                else:
                    outcome = Outcome(
                        status=StageStatus.SUCCESS,
                        notes=f"Stage completed: {node.id}",
                    )
                outcome.context_updates = {
                    "last_stage": node.id,
                    "last_response": text[:200] if text else "",
                }
                return outcome

            # Append assistant message and execute tools
            messages.append(_build_assistant_message(response))

            for tc in tool_calls:
                tool = self._tools.get(tc.name)
                if tool is not None:
                    try:
                        result = await tool.execute(tc.arguments)
                        output = (
                            result.output if hasattr(result, "output") else str(result)
                        )
                    except Exception as exc:
                        output = f"Tool error: {exc}"
                else:
                    output = f"Unknown tool: {tc.name}"

                messages.append(
                    Message(
                        role="tool",
                        tool_call_id=tc.id,
                        content=str(output) if not isinstance(output, str) else output,
                    )
                )

        return Outcome(
            status=StageStatus.PARTIAL_SUCCESS,
            notes=f"Max tool loop rounds ({_MAX_TOOL_LOOP_ROUNDS}) reached",
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
       (full "sessions all the way down").  A direct provider and tools
       are also passed so the backend can fall back to a mini tool loop
       if spawn becomes unavailable for a particular call.
    2. Else if at least one provider is available → use
       DirectProviderBackend (mini agentic tool loop per node).
    3. Otherwise → return None (codergen handler falls through to
       simulation mode).
    """
    first_provider = next(iter(providers.values()), None) if providers else None

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
            return AmplifierBackend(
                coordinator,
                profiles={},
                provider=first_provider,
                tools=tools,
            )

    # Fall back to direct provider tool loop
    if first_provider is not None:
        logger.info("Using DirectProviderBackend (direct provider tool loop)")
        return DirectProviderBackend(first_provider, tools, hooks, coordinator)

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
