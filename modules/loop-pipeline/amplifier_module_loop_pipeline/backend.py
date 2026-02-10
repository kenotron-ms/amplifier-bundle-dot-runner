"""AmplifierBackend — CodergenBackend adapter using session spawning.

This is the "sessions all the way down" integration point. When the
pipeline engine hits a codergen node, the CodergenHandler calls this
backend, which spawns a coding agent sub-session via the Amplifier
``session.spawn`` capability.

Spec coverage: Section 4.5 (CodergenBackend Interface), Section 1.4.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .context import PipelineContext
from .graph import Node
from .outcome import Outcome, StageStatus

logger = logging.getLogger(__name__)

# Map StageStatus value strings to enum members for parsing
_STATUS_MAP: dict[str, StageStatus] = {s.value: s for s in StageStatus}


class AmplifierBackend:
    """CodergenBackend implementation using Amplifier session spawning.

    Resolves the provider profile from node attributes, spawns a child
    coding agent session, and parses the outcome from the response.
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

    async def run(self, node: Node, prompt: str, context: PipelineContext) -> Outcome:
        """Execute a coding task by spawning a child session.

        Args:
            node: The pipeline node being executed.
            prompt: The expanded prompt string.
            context: The current pipeline context.

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

        # 3. Build spawn kwargs
        spawn_kwargs: dict[str, Any] = {
            "agent_name": profile_name,
            "instruction": prompt,
            "orchestrator_config": {
                "reasoning_effort": reasoning_effort,
            },
        }
        if model:
            spawn_kwargs["provider_preferences"] = [
                {"provider": provider, "model": model}
            ]

        # 4. Spawn the child session
        try:
            result = await self._spawn_fn(**spawn_kwargs)
        except Exception as e:
            logger.warning("Spawn failed for node %s: %s", node.id, e)
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason=str(e),
            )

        # 5. Parse outcome from result
        output = result.get("output", "") if isinstance(result, dict) else str(result)
        return _parse_outcome(output)


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
