"""Manager loop handler — supervisor pattern over a child subgraph.

The manager loop (shape=house) orchestrates sprint-based iteration by
supervising a child subgraph. Each cycle the manager:

1. **Observes** — runs the child subgraph via a subgraph_runner callback.
2. **Evaluates** — checks a guard condition (manager.stop_condition) to
   decide whether to stop or continue.
3. **Acts** — optionally injects steering context, then waits before the
   next cycle.

The loop terminates when the guard is satisfied, the child succeeds
(default guard), or max_cycles is exhausted.

Node attributes:
    manager.max_cycles      — Maximum observation cycles (default 10).
    manager.poll_interval   — Delay between cycles, e.g. "45s" (default "0s").
    manager.stop_condition  — Condition expression for early exit (default "").
    manager.actions         — Comma-separated: observe, steer, wait (default
                              "observe,wait").

Spec coverage: MGR-001-010, COMP-001-002, Section 4.11.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Callable, Coroutine

from ..conditions import evaluate_condition
from ..context import PipelineContext
from ..graph import Graph, Node
from ..outcome import Outcome, StageStatus

logger = logging.getLogger(__name__)

# Type alias matching the ParallelHandler's SubgraphRunner convention.
SubgraphRunner = Callable[
    [str, PipelineContext, Graph, str],
    Coroutine[Any, Any, Outcome],
]

# Pattern for parsing duration strings like "45s", "2m", "500ms".
_DURATION_RE = re.compile(
    r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>ms|s|m)?\s*$",
    re.IGNORECASE,
)


def _parse_duration(raw: str) -> float:
    """Parse a duration string to seconds.

    Supports: "45s", "2m", "500ms", or plain number (treated as seconds).
    Returns 0.0 on parse failure.
    """
    m = _DURATION_RE.match(raw)
    if not m:
        return 0.0
    value = float(m.group("value"))
    unit = (m.group("unit") or "s").lower()
    if unit == "ms":
        return value / 1000.0
    if unit == "m":
        return value * 60.0
    return value  # seconds


class ManagerLoopHandler:
    """Handler for manager loop nodes (shape=house).

    Runs an observe/evaluate/act cycle over a child subgraph,
    delegating child execution to the provided subgraph_runner.
    """

    def __init__(self, subgraph_runner: SubgraphRunner | None = None) -> None:
        self._runner = subgraph_runner

    async def execute(
        self,
        node: Node,
        context: PipelineContext,
        graph: Graph,
        logs_root: str,
    ) -> Outcome:
        """Execute a manager loop node.

        Reads configuration from node attributes, then enters the
        observe/evaluate/act cycle until a stop condition is met or
        max_cycles is exhausted.
        """
        # -- Parse configuration ------------------------------------------------
        max_cycles = int(node.attrs.get("manager.max_cycles", 1000))  # spec default
        poll_interval_s = _parse_duration(
            str(node.attrs.get("manager.poll_interval", "0s"))
        )
        stop_condition = str(node.attrs.get("manager.stop_condition", ""))
        actions_raw = str(node.attrs.get("manager.actions", "observe,wait"))
        actions = [a.strip() for a in actions_raw.split(",")]

        # -- Validate prerequisites ---------------------------------------------
        child_edges = graph.outgoing_edges(node.id)
        if not child_edges:
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason="Manager loop has no child to supervise",
            )

        if self._runner is None:
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason="Manager loop requires a subgraph_runner",
            )

        child_start_id = child_edges[0].to_node

        logger.info(
            "Manager '%s': max_cycles=%d, poll=%.1fs, actions=%s, child=%s",
            node.id,
            max_cycles,
            poll_interval_s,
            actions,
            child_start_id,
        )

        # -- Observation loop ---------------------------------------------------
        last_outcome: Outcome | None = None
        for cycle in range(1, max_cycles + 1):
            # 1. OBSERVE — build child context and run child subgraph
            child_context = context.clone()
            if "steer" in actions and last_outcome is not None:
                # M-15: Include actual failure details in steering message
                parts = [
                    f"Cycle {cycle - 1} of {max_cycles} resulted in"
                    f" {last_outcome.status.value}.",
                ]
                if last_outcome.failure_reason:
                    parts.append(f"Failure reason: {last_outcome.failure_reason}")
                if last_outcome.notes:
                    parts.append(f"Notes: {last_outcome.notes}")
                parts.append("Adjust your approach based on these details.")
                child_context.set("manager.steering", " ".join(parts))

            try:
                child_outcome = await self._runner(
                    child_start_id, child_context, graph, logs_root
                )
            except Exception as exc:
                logger.warning(
                    "Manager '%s' cycle %d: child raised %s",
                    node.id,
                    cycle,
                    exc,
                )
                child_outcome = Outcome(
                    status=StageStatus.FAIL,
                    failure_reason=str(exc),
                )

            last_outcome = child_outcome

            # Record cycle telemetry in parent context
            context.set(f"manager.cycle_{cycle}.status", child_outcome.status.value)
            context.set("manager.last_child_status", child_outcome.status.value)
            context.set("manager.cycles_completed", cycle)

            # 2. EVALUATE — check stop / guard condition
            if stop_condition:
                if evaluate_condition(stop_condition, child_outcome, context):
                    return Outcome(
                        status=child_outcome.status,
                        notes=f"Manager completed in {cycle} cycle(s) — stop condition satisfied",
                        context_updates={
                            "last_stage": node.id,
                            "manager.cycles": cycle,
                        },
                    )
            else:
                # Default guard: stop on success or partial_success
                if child_outcome.is_success:
                    return Outcome(
                        status=child_outcome.status,
                        notes=f"Manager completed in {cycle} cycle(s)",
                        context_updates={
                            "last_stage": node.id,
                            "manager.cycles": cycle,
                        },
                    )

            # 3. ACT — wait before next cycle
            if "wait" in actions and cycle < max_cycles and poll_interval_s > 0:
                await asyncio.sleep(poll_interval_s)

        # -- Max cycles exhausted -----------------------------------------------
        return Outcome(
            status=StageStatus.FAIL,
            failure_reason=f"Manager exhausted {max_cycles} cycle(s)",
            notes=f"Last child status: {last_outcome.status.value if last_outcome else 'none'}",
            context_updates={
                "last_stage": node.id,
                "manager.cycles": max_cycles,
            },
        )
