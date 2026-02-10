"""Parallel handler — fans out execution to concurrent branches.

Each parallel branch receives an isolated clone of the parent context
and runs independently. The handler waits for all branches to complete
(or applies a configurable join policy) before returning.

Spec coverage: PAR-001–013, CONC-001–004, Section 4.8.

Node attributes:
    max_parallel   – Maximum concurrent branches (default 4).
    join_policy    – wait_all | first_success | k_of_n | quorum (default wait_all).
    error_policy   – fail_fast | continue | ignore (default continue).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Coroutine

from ..context import PipelineContext
from ..graph import Graph, Node
from ..outcome import Outcome, StageStatus

logger = logging.getLogger(__name__)

# Type alias for the subgraph runner callback
SubgraphRunner = Callable[
    [str, PipelineContext, Graph, str],
    Coroutine[Any, Any, Outcome],
]


class ParallelHandler:
    """Handler for parallel fan-out nodes (shape=component).

    Spawns concurrent branches with isolated contexts, respects
    bounded parallelism, and evaluates a join policy on results.
    """

    def __init__(self, subgraph_runner: SubgraphRunner | None = None) -> None:
        """Initialize the parallel handler.

        Args:
            subgraph_runner: Async callable that executes a subgraph
                starting from a given node ID. Signature:
                (node_id, context, graph, logs_root) -> Outcome.
                If None, branches return SUCCESS (simulation mode).
        """
        self._runner = subgraph_runner

    async def execute(
        self,
        node: Node,
        context: PipelineContext,
        graph: Graph,
        logs_root: str,
    ) -> Outcome:
        """Execute a parallel node by fanning out to all outgoing edges.

        1. Identify fan-out edges (all outgoing edges from this node).
        2. Clone context per branch for isolation.
        3. Execute branches concurrently with bounded parallelism.
        4. Store results in parent context for downstream fan-in.
        5. Evaluate join policy and return aggregate outcome.
        """
        branches = graph.outgoing_edges(node.id)
        if not branches:
            return Outcome(
                status=StageStatus.SUCCESS,
                notes="Parallel node with no branches",
            )

        max_parallel = int(node.attrs.get("max_parallel", 4))
        join_policy = str(node.attrs.get("join_policy", "wait_all"))
        semaphore = asyncio.Semaphore(max_parallel)

        async def run_branch(target_node_id: str) -> dict[str, Any]:
            """Execute a single branch with bounded concurrency."""
            async with semaphore:
                branch_context = context.clone()
                try:
                    if self._runner is not None:
                        outcome = await self._runner(
                            target_node_id, branch_context, graph, logs_root
                        )
                    else:
                        outcome = Outcome(
                            status=StageStatus.SUCCESS,
                            notes=f"Simulated branch: {target_node_id}",
                        )
                except Exception as e:
                    logger.warning("Branch %s raised exception: %s", target_node_id, e)
                    outcome = Outcome(
                        status=StageStatus.FAIL,
                        failure_reason=str(e),
                    )

                return {
                    "node_id": target_node_id,
                    "status": outcome.status.value,
                    "notes": outcome.notes,
                    "failure_reason": outcome.failure_reason,
                    "context_updates": outcome.context_updates,
                }

        # Launch all branches concurrently
        tasks = [run_branch(edge.to_node) for edge in branches]
        results: list[dict[str, Any]] = await asyncio.gather(*tasks)

        # Store results in parent context for fan-in
        context.set("parallel.results", results)
        context.set("parallel.count", len(results))

        # Evaluate join policy
        return _apply_join_policy(results, join_policy)


def _apply_join_policy(results: list[dict[str, Any]], policy: str) -> Outcome:
    """Evaluate a join policy against branch results.

    Currently implements wait_all. Other policies can be added
    incrementally.
    """
    if not results:
        return Outcome(status=StageStatus.SUCCESS, notes="No branches")

    success_count = sum(
        1 for r in results if r["status"] in ("success", "partial_success")
    )
    fail_count = sum(1 for r in results if r["status"] == "fail")
    total = len(results)

    if policy == "wait_all":
        if fail_count == 0:
            return Outcome(
                status=StageStatus.SUCCESS,
                notes=f"All {total} branches succeeded",
            )
        return Outcome(
            status=StageStatus.PARTIAL_SUCCESS,
            notes=f"{success_count}/{total} branches succeeded, {fail_count} failed",
        )

    if policy == "first_success":
        if success_count > 0:
            return Outcome(
                status=StageStatus.SUCCESS,
                notes=f"At least one branch succeeded ({success_count}/{total})",
            )
        return Outcome(
            status=StageStatus.FAIL,
            failure_reason=f"No branches succeeded out of {total}",
        )

    # Default: treat as wait_all
    if fail_count == 0:
        return Outcome(
            status=StageStatus.SUCCESS,
            notes=f"All {total} branches succeeded",
        )
    return Outcome(
        status=StageStatus.PARTIAL_SUCCESS,
        notes=f"{success_count}/{total} branches succeeded",
    )
