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
from .transforms import apply_transforms
from .validation import validate_or_raise

logger = logging.getLogger(__name__)


class DirectProviderBackend:
    """Backend that calls a provider directly with a mini tool loop.

    This is the default backend when no session.spawn capability is
    available.  It runs an agentic loop \u2014 call LLM, execute any tool
    calls, feed results back, repeat \u2014 until the model returns a
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
        # Fidelity state (H-9): track completed nodes and message history
        self._completed_nodes: dict[str, Any] = {}
        self._message_pools: dict[str, list] = {}  # thread_key -> message history
        self._last_node_id: str | None = None

    async def run(
        self,
        node: Any,
        prompt: str,
        context: PipelineContext,
        *,
        incoming_edge: Any | None = None,
        graph: Any | None = None,
        **kwargs: Any,
    ) -> Outcome:
        """Run a mini agentic tool loop for *node*.

        Supports fidelity-aware context carryover (spec Section 5.4):
        - full: reuse message history from previous calls with same thread key
        - compact/truncate/summary: prepend preamble from completed node history
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

        # Resolve fidelity mode (spec FID-001)
        from .fidelity import build_preamble, resolve_fidelity, resolve_thread_key

        fidelity = "compact"  # default
        thread_key = node.id
        if graph is not None:
            fidelity = resolve_fidelity(node, incoming_edge, graph)
            thread_key = resolve_thread_key(
                node, incoming_edge, graph, self._last_node_id
            )

        # Build messages based on fidelity mode
        if fidelity == "full":
            # Reuse accumulated message history for this thread key
            messages: list[Any] = list(self._message_pools.get(thread_key, []))
            messages.append(Message(role="user", content=prompt))
        else:
            # Fresh session with preamble
            if graph is not None and self._completed_nodes:
                preamble = build_preamble(fidelity, context, self._completed_nodes)
                effective_prompt = (
                    f"{preamble}\n\n---\n\n{prompt}" if preamble else prompt
                )
            else:
                effective_prompt = prompt
            messages = [Message(role="user", content=effective_prompt)]

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
                # Model is done \u2014 parse the final text as an outcome
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

                # Record fidelity state for future calls
                self._completed_nodes[node.id] = outcome
                self._last_node_id = node.id

                # For full fidelity: save message history including response
                if fidelity == "full":
                    messages.append(Message(role="assistant", content=text or ""))
                    self._message_pools[thread_key] = messages

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

        outcome = Outcome(
            status=StageStatus.PARTIAL_SUCCESS,
            notes=f"Max tool loop rounds ({_MAX_TOOL_LOOP_ROUNDS}) reached",
        )
        self._completed_nodes[node.id] = outcome
        self._last_node_id = node.id
        return outcome


def _build_backend(
    providers: dict[str, Any],
    tools: dict[str, Any],
    hooks: Any,
    coordinator: Any | None,
    orchestrator_config: dict[str, Any] | None = None,
) -> Any | None:
    """Auto-construct a backend from the available providers.

    Resolution order:
    1. If coordinator exposes ``session.spawn`` \u2192 use AmplifierBackend
       (full "sessions all the way down").  Profiles are resolved from
       ``orchestrator_config["profiles"]`` or auto-discovered from
       ``coordinator.config["agents"]``.
    2. Else if at least one provider is available \u2192 use
       DirectProviderBackend (mini agentic tool loop per node).
    3. Otherwise \u2192 return None (codergen handler falls through to
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

            # Resolve profiles: explicit config > auto-discovery from agents
            cfg = orchestrator_config or {}
            profiles: dict[str, str] = {}

            # Source 1: Explicit profiles mapping in orchestrator config
            # e.g. config.profiles = {"anthropic": "attractor-anthropic"}
            explicit_profiles = cfg.get("profiles")
            if isinstance(explicit_profiles, dict):
                profiles.update(explicit_profiles)

            # Source 2: Auto-discover from coordinator.config["agents"]
            # Each agent entry is mapped as agent_name -> agent_name.
            if not profiles:
                coordinator_config = getattr(coordinator, "config", None) or {}
                agents = coordinator_config.get("agents", {})
                for agent_name, agent_cfg in agents.items():
                    if isinstance(agent_cfg, dict):
                        profiles[agent_name] = agent_name

            if profiles:
                logger.info(
                    "Using AmplifierBackend (session.spawn available, profiles=%s)",
                    list(profiles.keys()),
                )
            else:
                logger.warning(
                    "Using AmplifierBackend but profiles dict is empty. "
                    "Pipeline nodes may fail to resolve agent profiles. "
                    "Add 'profiles' to orchestrator config or 'agents' "
                    "to the bundle."
                )

            return AmplifierBackend(
                coordinator,
                profiles=profiles,
                provider=first_provider,
                tools=tools,
            )

    # Fall back to direct provider tool loop
    if first_provider is not None:
        logger.info("Using DirectProviderBackend (direct provider tool loop)")
        return DirectProviderBackend(first_provider, tools, hooks, coordinator)

    logger.warning(
        "No providers available \u2014 codergen nodes will run in simulation mode"
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

        # 3. Create pipeline context with goal from the prompt
        pipeline_context = PipelineContext()
        if prompt:
            pipeline_context.set("graph.goal", prompt)

        # 4. Apply transforms (variable expansion, stylesheet) before validation
        apply_transforms(graph, pipeline_context)

        # 5. Validate the (transformed) graph
        validate_or_raise(graph)

        # 6. Set up logs directory
        logs_root = self.config.get(
            "logs_root", os.path.join(tempfile.gettempdir(), "attractor-pipeline")
        )
        os.makedirs(logs_root, exist_ok=True)

        # 7. Resolve backend: explicit kwarg \u2192 auto-construct from providers
        coordinator = kwargs.get("coordinator")
        backend = kwargs.get("backend")
        if backend is None:
            backend = _build_backend(providers, tools, hooks, coordinator, self.config)

        # 8. Create engine first (handlers need its _run_from method)
        # Use a placeholder registry, then replace after wiring
        engine = PipelineEngine(
            graph=graph,
            context=pipeline_context,
            handler_registry=HandlerRegistry(backend=backend),  # temp
            logs_root=logs_root,
            hooks=hooks,
        )

        # 9. Create subgraph runner closure that delegates to engine._run_from
        async def subgraph_runner(
            node_id: str,
            branch_context: PipelineContext,
            _graph: Any,
            _logs_root: str,
        ) -> Outcome:
            """Execute a subgraph branch via the engine."""
            return await engine._run_from(node_id, context=branch_context)

        # 10. Register handlers with the subgraph runner wired in
        registry = HandlerRegistry(
            backend=backend,
            subgraph_runner=subgraph_runner,
            hooks=hooks,
        )
        engine.handler_registry = registry

        # 11. Run the engine
        outcome = await engine.run(goal=prompt or None)

        # 12. Build a meaningful summary from all completed nodes
        summary = self._build_pipeline_summary(engine, outcome)

        # 13. Return the final outcome as JSON
        result = {
            "status": outcome.status.value,
            "notes": summary,
            "failure_reason": outcome.failure_reason,
            "nodes_completed": len(engine.completed_nodes),
            "node_statuses": {
                nid: engine.node_outcomes[nid].status.value
                for nid in engine.completed_nodes
                if nid in engine.node_outcomes
            },
        }
        return json.dumps(result)

    def _build_pipeline_summary(self, engine: PipelineEngine, outcome: Outcome) -> str:
        """Build a human-readable pipeline summary.

        If the final outcome has meaningful notes, use them.
        Otherwise, synthesize a summary from all completed nodes.
        """
        # Use the outcome's notes if they exist and are meaningful
        if outcome.notes and len(outcome.notes) > 20:
            return outcome.notes

        # Synthesize from all node outcomes
        parts: list[str] = []
        total = len(engine.completed_nodes)
        succeeded = sum(
            1
            for nid in engine.completed_nodes
            if nid in engine.node_outcomes and engine.node_outcomes[nid].is_success
        )
        failed = total - succeeded

        parts.append(f"Pipeline completed: {succeeded}/{total} nodes succeeded.")

        if failed:
            failed_nodes = [
                nid
                for nid in engine.completed_nodes
                if nid in engine.node_outcomes
                and not engine.node_outcomes[nid].is_success
            ]
            parts.append(f"Failed nodes: {', '.join(failed_nodes)}.")

        # Include the last node's notes if available
        if engine.completed_nodes:
            last_id = engine.completed_nodes[-1]
            last_out = engine.node_outcomes.get(last_id)
            if last_out and last_out.notes:
                # Truncate to avoid bloating the summary
                snippet = last_out.notes[:300]
                parts.append(f"Last node ({last_id}): {snippet}")

        return " ".join(parts)

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
