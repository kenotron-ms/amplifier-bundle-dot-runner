"""AmplifierBackend — CodergenBackend adapter using session spawning.

This is the "sessions all the way down" integration point. When the
pipeline engine hits a codergen node, the CodergenHandler calls this
backend, which spawns a coding agent sub-session via the Amplifier
``session.spawn`` capability.

Spec coverage: Section 4.5 (CodergenBackend Interface), Section 1.4,
               FID-001–010, Section 5.4.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .context import PipelineContext
from .fidelity import build_preamble, resolve_fidelity, resolve_thread_key
from .graph import Edge, Graph, Node
from .outcome import Outcome, StageStatus

logger = logging.getLogger(__name__)

# Map StageStatus value strings to enum members for parsing
_STATUS_MAP: dict[str, StageStatus] = {s.value: s for s in StageStatus}


class AmplifierBackend:
    """CodergenBackend implementation using Amplifier session spawning.

    Resolves the provider profile from node attributes, spawns a child
    coding agent session, and parses the outcome from the response.

    Supports fidelity-based context control:
    - ``full``: Reuses sessions via a thread-keyed session pool.
    - ``compact``/``truncate``/``summary:*``: Fresh session with preamble.
    """

    def __init__(
        self,
        coordinator: Any,
        profiles: dict[str, str],
    ) -> None:
        """Initialize the backend.

        Args:
            coordinator: Amplifier coordinator with session.spawn capability.
            profiles: Map of provider name to profile/bundle name.
                      e.g. {"anthropic": "attractor-anthropic", ...}
        """
        self._coordinator = coordinator
        self._profiles = profiles
        self._spawn_fn: Any | None = None
        self._session_pool: dict[str, str] = {}
        self._completed_nodes: dict[str, Outcome] = {}
        self._last_node_id: str | None = None

    async def run(
        self,
        node: Node,
        prompt: str,
        context: PipelineContext,
        incoming_edge: Edge | None = None,
        graph: Graph | None = None,
    ) -> Outcome:
        """Execute a coding task by spawning a child session.

        Args:
            node: The pipeline node being executed.
            prompt: The expanded prompt string.
            context: The current pipeline context.
            incoming_edge: The edge leading to this node (for fidelity resolution).
            graph: The pipeline graph (for fidelity resolution).

        Returns:
            Outcome parsed from the child session's response.
        """
        # 1. Get spawn capability (lazy resolution)
        if self._spawn_fn is None:
            self._spawn_fn = self._coordinator.get_capability("session.spawn")
        if self._spawn_fn is None:
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason="Session spawning not available",
            )

        # 2. Resolve provider and profile from node attributes
        provider = node.attrs.get("llm_provider", "anthropic")
        model = node.attrs.get("llm_model")
        reasoning_effort = node.attrs.get("reasoning_effort", "high")
        profile_name = self._profiles.get(
            provider, next(iter(self._profiles.values()), "")
        )

        # 3. Resolve fidelity mode (spec FID-001–010)
        if graph is not None:
            fidelity = resolve_fidelity(node, incoming_edge, graph)
        else:
            # Fallback when graph not provided (backward compat)
            fidelity = node.attrs.get("fidelity", "compact")

        # 4. Build the instruction with preamble for non-full modes
        if fidelity == "full":
            instruction = prompt
        else:
            preamble = build_preamble(fidelity, context, self._completed_nodes)
            instruction = f"{preamble}\n\n---\n\n{prompt}" if preamble else prompt

        # 5. Build spawn kwargs
        spawn_kwargs: dict[str, Any] = {
            "agent_name": profile_name,
            "instruction": instruction,
            "orchestrator_config": {
                "reasoning_effort": reasoning_effort,
            },
        }
        if model:
            spawn_kwargs["provider_preferences"] = [
                {"provider": provider, "model": model}
            ]

        # 6. Session pool for full fidelity (spec FID-001: thread reuse)
        if fidelity == "full" and graph is not None:
            thread_key = resolve_thread_key(
                node, incoming_edge, graph, self._last_node_id
            )
            existing_session = self._session_pool.get(thread_key)
            if existing_session is not None:
                spawn_kwargs["session_id"] = existing_session

        # 7. Spawn the child session
        try:
            result = await self._spawn_fn(**spawn_kwargs)
        except Exception as e:
            logger.warning("Spawn failed for node %s: %s", node.id, e)
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason=str(e),
            )

        # 8. Parse outcome from result
        output = result.get("output", "") if isinstance(result, dict) else str(result)
        outcome = _parse_outcome(output)

        # 9. Record session_id in pool for full fidelity reuse
        if fidelity == "full" and graph is not None:
            session_id = result.get("session_id") if isinstance(result, dict) else None
            if session_id:
                thread_key = resolve_thread_key(
                    node, incoming_edge, graph, self._last_node_id
                )
                self._session_pool[thread_key] = session_id

        # 10. Record completed node outcome for future preambles
        self._completed_nodes[node.id] = outcome
        self._last_node_id = node.id

        return outcome


def _parse_outcome(output: str) -> Outcome:
    """Parse an outcome from child session output.

    Tries JSON first (from tool-report-outcome), falls back to
    wrapping plain text as SUCCESS.
    """
    # Try to parse JSON outcome
    stripped = output.strip()
    if stripped.startswith("{"):
        try:
            data = json.loads(stripped)
            if "status" in data:
                status = _STATUS_MAP.get(data["status"])
                if status is not None:
                    return Outcome(
                        status=status,
                        failure_reason=data.get("failure_reason"),
                        notes=data.get("notes"),
                        preferred_label=data.get("preferred_label"),
                        suggested_next_ids=data.get("suggested_next_ids"),
                        context_updates=data.get("context_updates"),
                    )
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    # Fall back to plain text → SUCCESS
    return Outcome(
        status=StageStatus.SUCCESS,
        notes=output[:200] if output else "No output",
    )
