"""Manager loop handler (stub) — supervisor pattern over a child pipeline.

The manager loop (shape=house) orchestrates sprint-based iteration by
supervising a child pipeline. The manager observes the child's telemetry,
evaluates progress via a guard function, and optionally steers the child
through intervention.

This is a **stub implementation** that allows pipelines containing manager
nodes to parse and execute without crashing. The stub runs a single cycle
and returns SUCCESS.

**Full implementation (future):**
The complete manager loop would:
1. Auto-start a child pipeline from ``stack.child_dotfile``.
2. Enter an observation loop with ``manager.max_cycles`` iterations.
3. Each cycle:
   a. **Observe** — ingest child telemetry (active stage, outcomes,
      retry counts, artifacts) into context.
   b. **Guard** — evaluate ``manager.stop_condition`` to decide whether
      to continue, intervene, or escalate.
   c. **Steer** — if cooldown has elapsed, write intervention instructions
      to the child's active stage directory.
   d. **Wait** — sleep for ``manager.poll_interval``.
4. Return SUCCESS if child completes, FAIL if max cycles exceeded.

Spec coverage: MGR-001–010, COMP-001–002, Section 4.11.
"""

from __future__ import annotations

import logging

from ..context import PipelineContext
from ..graph import Graph, Node
from ..outcome import Outcome, StageStatus

logger = logging.getLogger(__name__)


class ManagerLoopHandler:
    """Stub handler for manager loop nodes (shape=house).

    Allows pipelines with manager nodes to execute. Returns SUCCESS
    after a single simulated cycle. See module docstring for the
    full implementation design.
    """

    async def execute(
        self,
        node: Node,
        context: PipelineContext,
        graph: Graph,
        logs_root: str,
    ) -> Outcome:
        """Execute a manager loop node (stub).

        Reads configuration from node attributes but does not actually
        spawn or supervise a child pipeline.
        """
        # Read configuration (parsed but not used in stub)
        max_cycles = int(node.attrs.get("manager.max_cycles", 10))
        poll_interval = node.attrs.get("manager.poll_interval", "45s")
        _stop_condition = node.attrs.get("manager.stop_condition", "")
        _actions = node.attrs.get("manager.actions", "observe,wait")
        child_dotfile = node.attrs.get("stack.child_dotfile", "")

        logger.info(
            "Manager loop stub '%s': max_cycles=%d, poll=%s, child=%s",
            node.id,
            max_cycles,
            poll_interval,
            child_dotfile,
        )

        return Outcome(
            status=StageStatus.SUCCESS,
            notes=(
                f"Manager loop stub — ran 1 simulated cycle "
                f"(max_cycles={max_cycles}, child={child_dotfile!r})"
            ),
            context_updates={
                "last_stage": node.id,
            },
        )
