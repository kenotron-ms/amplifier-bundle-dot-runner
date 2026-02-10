"""Fan-in handler — consolidates results from a parallel node.

Reads ``parallel.results`` from context, ranks candidates by outcome
status, selects the best one, and records the winner in context for
downstream nodes.

Spec coverage: FANIN-001–005, Section 4.9.

Heuristic ranking (best first):
    SUCCESS > PARTIAL_SUCCESS > RETRY > FAIL

Ties are broken by node ID (lexicographic ascending).
"""

from __future__ import annotations

import logging
from typing import Any

from ..context import PipelineContext
from ..graph import Graph, Node
from ..outcome import Outcome, StageStatus

logger = logging.getLogger(__name__)

# Ranking for heuristic selection: lower number = better
_STATUS_RANK: dict[str, int] = {
    "success": 0,
    "partial_success": 1,
    "retry": 2,
    "skipped": 3,
    "fail": 4,
}


class FanInHandler:
    """Handler for fan-in nodes (shape=tripleoctagon).

    Evaluates parallel results and selects the best candidate.
    """

    async def execute(
        self,
        node: Node,
        context: PipelineContext,
        graph: Graph,
        logs_root: str,
    ) -> Outcome:
        """Evaluate parallel results and select the best candidate.

        1. Read parallel.results from context.
        2. Rank candidates by status (heuristic).
        3. Record winner in context.
        4. Return SUCCESS if at least one candidate succeeded.
        """
        results: list[dict[str, Any]] | None = context.get("parallel.results")

        if not results:
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason="No parallel results to evaluate",
            )

        # Heuristic selection: rank by status, then node_id for tiebreak
        best = _heuristic_select(results)

        if best is None:
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason="No parallel results to evaluate",
            )

        best_status = best.get("status", "fail")
        best_id = best.get("node_id", "unknown")

        # Record winner in context
        context.set("parallel.fan_in.best_id", best_id)
        context.set("parallel.fan_in.best_status", best_status)

        # If best candidate failed, fan-in fails
        if best_status == "fail":
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason=f"All candidates failed. Best: {best_id}",
                notes=best.get("notes"),
            )

        return Outcome(
            status=StageStatus.SUCCESS,
            notes=f"Selected best candidate: {best_id} ({best_status})",
            context_updates={
                "parallel.fan_in.best_id": best_id,
                "parallel.fan_in.best_status": best_status,
            },
        )


def _heuristic_select(
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Select the best candidate by status ranking, then node ID.

    Spec Section 4.9: heuristic_select algorithm.
    """
    if not candidates:
        return None

    return sorted(
        candidates,
        key=lambda c: (
            _STATUS_RANK.get(c.get("status", "fail"), 99),
            c.get("node_id", ""),
        ),
    )[0]
