"""Parallel handler — fans out execution to concurrent branches.

Each parallel branch receives an isolated clone of the parent context
and runs independently. The handler waits for all branches to complete
(or applies a configurable join policy) before returning.

Spec coverage: PAR-001–013, CONC-001–004, Section 4.8.

Node attributes:
    max_parallel   – Maximum concurrent branches (default 4).
    join_policy    – wait_all | first_success | k_of_n | quorum (default wait_all).
    error_policy   – fail_fast | continue | ignore (default continue).
    min_success    – Required successes for k_of_n (default 1).
    quorum_fraction – Required fraction for quorum (default 0.5).
"""

from __future__ import annotations

import asyncio
import logging
import math
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

    def __init__(
        self,
        subgraph_runner: SubgraphRunner | None = None,
        hooks: Any = None,
    ) -> None:
        """Initialize the parallel handler.

        Args:
            subgraph_runner: Async callable that executes a subgraph
                starting from a given node ID. Signature:
                (node_id, context, graph, logs_root) -> Outcome.
                If None, branches return SUCCESS (simulation mode).
            hooks: Optional hooks object for event emission.
        """
        self._runner = subgraph_runner
        self._hooks = hooks

    async def _emit(self, event_name: str, data: dict[str, Any]) -> None:
        """Emit an event via hooks, if provided."""
        if self._hooks is not None:
            await self._hooks.emit(event_name, data)

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
        from ..pipeline_events import (
            PIPELINE_PARALLEL_BRANCH_COMPLETED,
            PIPELINE_PARALLEL_BRANCH_STARTED,
            PIPELINE_PARALLEL_COMPLETED,
            PIPELINE_PARALLEL_STARTED,
        )

        branches = graph.outgoing_edges(node.id)
        if not branches:
            return Outcome(
                status=StageStatus.SUCCESS,
                notes="Parallel node with no branches",
            )

        max_parallel = int(node.attrs.get("max_parallel", 4))
        join_policy = str(node.attrs.get("join_policy", "wait_all"))
        error_policy = str(node.attrs.get("error_policy", "continue"))
        semaphore = asyncio.Semaphore(max_parallel)

        await self._emit(
            PIPELINE_PARALLEL_STARTED,
            {"node_id": node.id, "branch_count": len(branches)},
        )

        async def run_branch(target_node_id: str) -> dict[str, Any]:
            """Execute a single branch with bounded concurrency."""
            async with semaphore:
                await self._emit(
                    PIPELINE_PARALLEL_BRANCH_STARTED,
                    {"node_id": node.id, "branch_node_id": target_node_id},
                )
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

                await self._emit(
                    PIPELINE_PARALLEL_BRANCH_COMPLETED,
                    {
                        "node_id": node.id,
                        "branch_node_id": target_node_id,
                        "status": outcome.status.value,
                    },
                )

                return {
                    "node_id": target_node_id,
                    "status": outcome.status.value,
                    "notes": outcome.notes,
                    "failure_reason": outcome.failure_reason,
                    "context_updates": outcome.context_updates,
                }

        # Dispatch based on error policy
        if error_policy == "fail_fast":
            results = await _run_fail_fast(branches, run_branch, semaphore)
        else:
            # Default (continue) and ignore both run all branches
            tasks = [run_branch(edge.to_node) for edge in branches]
            results = list(await asyncio.gather(*tasks))

        # Apply ignore error policy: filter out failures before storing
        if error_policy == "ignore":
            results = [
                r for r in results if r["status"] in ("success", "partial_success")
            ]

        # Store results in parent context for fan-in
        context.set("parallel.results", results)
        context.set("parallel.count", len(results))

        await self._emit(
            PIPELINE_PARALLEL_COMPLETED,
            {
                "node_id": node.id,
                "branch_count": len(branches),
                "result_count": len(results),
            },
        )

        # Evaluate join policy
        return _apply_join_policy(results, join_policy, node_attrs=node.attrs)


async def _run_fail_fast(
    branches: list,
    run_branch: Callable,
    semaphore: asyncio.Semaphore,
) -> list[dict[str, Any]]:
    """Execute branches with fail_fast: cancel remaining on first failure.

    Uses asyncio tasks with a shared cancellation event. When any branch
    completes with a failure status, remaining branches are cancelled.
    """
    results: list[dict[str, Any]] = []
    failure_event = asyncio.Event()

    async def guarded_branch(edge) -> dict[str, Any] | None:
        """Run a branch but bail early if failure_event is set."""
        if failure_event.is_set():
            return None
        result = await run_branch(edge.to_node)
        if result["status"] == "fail":
            failure_event.set()
        return result

    tasks = [asyncio.create_task(guarded_branch(edge)) for edge in branches]

    # Wait with FIRST_EXCEPTION so we can cancel promptly
    done: set[asyncio.Task] = set()
    pending: set[asyncio.Task] = set(tasks)

    while pending:
        newly_done, pending = await asyncio.wait(
            pending, return_when=asyncio.FIRST_COMPLETED
        )
        done.update(newly_done)

        # Check if any completed task indicates failure
        for task in newly_done:
            result = task.result()
            if result is not None:
                results.append(result)
                if result["status"] == "fail":
                    # Cancel remaining pending tasks
                    for p in pending:
                        p.cancel()
                    # Collect any already-done results from pending
                    if pending:
                        cancelled_done, _ = await asyncio.wait(pending)
                        for ct in cancelled_done:
                            try:
                                cr = ct.result()
                                if cr is not None:
                                    results.append(cr)
                            except asyncio.CancelledError:
                                pass
                    return results

    return results


def _apply_join_policy(
    results: list[dict[str, Any]],
    policy: str,
    node_attrs: dict[str, Any] | None = None,
) -> Outcome:
    """Evaluate a join policy against branch results.

    Supports: wait_all, first_success, k_of_n, quorum.
    Unknown policies fall back to wait_all behaviour.
    """
    if not results:
        return Outcome(status=StageStatus.SUCCESS, notes="No branches")

    attrs = node_attrs or {}

    success_count = sum(
        1 for r in results if r["status"] in ("success", "partial_success")
    )
    fail_count = sum(1 for r in results if r["status"] == "fail")
    total = len(results)

    # -- wait_all --------------------------------------------------------
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

    # -- first_success ---------------------------------------------------
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

    # -- k_of_n ----------------------------------------------------------
    if policy == "k_of_n":
        k = int(attrs.get("min_success", 1))
        if success_count >= k:
            return Outcome(
                status=StageStatus.SUCCESS,
                notes=f"{success_count}/{total} branches succeeded (needed {k})",
            )
        return Outcome(
            status=StageStatus.FAIL,
            failure_reason=(
                f"Only {success_count}/{k} required branches succeeded "
                f"(out of {total} total)"
            ),
        )

    # -- quorum ----------------------------------------------------------
    if policy == "quorum":
        fraction = float(attrs.get("quorum_fraction", 0.5))
        needed = math.ceil(total * fraction)
        if success_count >= needed:
            return Outcome(
                status=StageStatus.SUCCESS,
                notes=(
                    f"{success_count}/{total} branches succeeded "
                    f"(needed {needed}, fraction={fraction})"
                ),
            )
        return Outcome(
            status=StageStatus.FAIL,
            failure_reason=(
                f"Only {success_count}/{needed} required branches succeeded "
                f"(fraction={fraction}, total={total})"
            ),
        )

    # -- Unknown policy: fall back to wait_all ---------------------------
    if fail_count == 0:
        return Outcome(
            status=StageStatus.SUCCESS,
            notes=f"All {total} branches succeeded",
        )
    return Outcome(
        status=StageStatus.PARTIAL_SUCCESS,
        notes=f"{success_count}/{total} branches succeeded",
    )
