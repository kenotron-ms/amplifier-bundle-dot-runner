"""Pipeline execution engine — the graph-walking core.

Traverses a parsed DOT graph from the start node to an exit node,
executing handlers for each node and selecting edges based on outcomes.
This is the heart of the Attractor pipeline orchestrator.

Spec coverage: EXEC-001–018, CHKP-004–006, EVT-001–008, DIR-001, STAT-001–004,
               Sections 3.2, 5.6, 9.6.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any

from .artifacts import ArtifactStore
from .checkpoint import Checkpoint, load_checkpoint, save_checkpoint
from .context import PipelineContext
from .edge_selection import select_all_matching_edges, select_edge
from .graph import Graph, Node
from .handlers import HandlerRegistry
from .outcome import Outcome, StageStatus
from .pipeline_events import (
    PIPELINE_CHECKPOINT,
    PIPELINE_COMPLETE,
    PIPELINE_EDGE_SELECTED,
    PIPELINE_ERROR,
    PIPELINE_GOAL_GATE_CHECK,
    PIPELINE_NODE_COMPLETE,
    PIPELINE_NODE_START,
    PIPELINE_START,
)
from .retry import RetryPolicy, execute_with_retry

logger = logging.getLogger(__name__)


class PipelineEngine:
    """Graph-walking execution engine.

    Walks the graph from start to exit, executing node handlers and
    selecting edges deterministically based on outcomes and context.

    Saves a checkpoint after each node execution so the pipeline can
    resume after crashes.
    """

    # Maximum number of goal-gate-driven retries before giving up.
    # Prevents infinite loops when a gate's retry_target never satisfies.
    _MAX_GOAL_GATE_RETRIES: int = 50

    def __init__(
        self,
        graph: Graph,
        context: PipelineContext,
        handler_registry: HandlerRegistry,
        logs_root: str,
        hooks: Any | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        self.graph = graph
        self.context = context
        self.handler_registry = handler_registry
        self.logs_root = logs_root
        self.hooks = hooks
        self._cancel_event = cancel_event
        self.node_outcomes: dict[str, Outcome] = {}
        self.completed_nodes: list[str] = []
        self.iteration_count: int = 0
        self._checkpoint_path = os.path.join(logs_root, "checkpoint.json")
        self.artifact_store = ArtifactStore(base_dir=logs_root)

    def _check_cancelled(self) -> bool:
        """Check if cancellation has been requested via the cancel event."""
        return self._cancel_event is not None and self._cancel_event.is_set()

    async def run(self, goal: str | None = None) -> Outcome:
        """Execute the pipeline from start to exit.

        If a checkpoint exists at ``{logs_root}/checkpoint.json``, the
        engine resumes from it, skipping already-completed nodes and
        restoring context state.

        Args:
            goal: Optional goal string to set in context. If not provided,
                uses the graph-level goal attribute.

        Returns:
            The final Outcome of the pipeline run.
        """
        pipeline_start_time = time.monotonic()

        # Initialize context with graph attributes
        self._initialize_context(goal)

        # Note: transforms (variable expansion, stylesheet) are applied by
        # PipelineOrchestrator.execute() between parse and validate, before
        # the engine is constructed.  Do NOT re-apply here.

        # Create run directory structure (manifest, artifacts/)
        self._write_manifest(goal)

        # Emit pipeline:start
        await self._emit(
            PIPELINE_START,
            {
                "graph_name": self.graph.name,
                "node_count": len(self.graph.nodes),
                "edge_count": len(self.graph.edges),
                "goal": self.graph.goal or goal or "",
                "dot_source": self.graph.dot_source,
            },
        )

        # Check for existing checkpoint and restore state
        resumed = self._try_resume_from_checkpoint()

        # Find the start node
        current_node = self._find_start_node()

        # Bound goal-gate-driven retries to prevent infinite loops
        goal_gate_retries = 0
        # Bound failure-routing retries (no-matching-edge fallback chain)
        failure_routing_retries = 0

        while True:
            # Step 0: Enforce max_pipeline_duration if set on the graph.
            # The DOT parser stores durations as milliseconds.
            if self.graph.max_pipeline_duration:
                elapsed_ms = (time.monotonic() - pipeline_start_time) * 1000
                if elapsed_ms > self.graph.max_pipeline_duration:
                    duration_outcome = Outcome(
                        status=StageStatus.FAIL,
                        notes=(
                            f"Pipeline exceeded max duration of "
                            f"{self.graph.max_pipeline_duration}ms"
                        ),
                        failure_reason="max_pipeline_duration_exceeded",
                    )
                    await self._emit_complete(duration_outcome, pipeline_start_time)
                    return duration_outcome

            # Step 0.5: Check for cancellation (cooperative cross-thread signal)
            if self._check_cancelled():
                cancelled_outcome = Outcome(
                    status=StageStatus.FAIL,
                    notes="Pipeline cancelled by user request",
                    failure_reason="cancelled",
                )
                await self._emit(
                    PIPELINE_COMPLETE,
                    {
                        "status": "cancelled",
                        "total_nodes_executed": len(self.completed_nodes),
                        "duration_ms": (time.monotonic() - pipeline_start_time) * 1000,
                    },
                )
                return cancelled_outcome

            # Step 1: Check for terminal node (exit)
            if current_node.is_exit_node():
                self._save_checkpoint(current_node.id)
                await self._emit(
                    PIPELINE_CHECKPOINT,
                    {
                        "node_id": current_node.id,
                        "checkpoint_path": self._checkpoint_path,
                    },
                )
                gate_result = await self._check_goal_gates()

                # All gates satisfied — return final outcome
                if gate_result.status != StageStatus.FAIL:
                    await self._emit_complete(gate_result, pipeline_start_time)
                    return gate_result

                # Unsatisfied gate with retry target — jump there
                if (
                    gate_result.suggested_next_ids
                    and goal_gate_retries < self._MAX_GOAL_GATE_RETRIES
                ):
                    retry_node_id = gate_result.suggested_next_ids[0]
                    current_node = self.graph.nodes[retry_node_id]
                    goal_gate_retries += 1
                    logger.info(
                        "Goal gate unsatisfied, retrying from '%s' (attempt %d)",
                        retry_node_id,
                        goal_gate_retries,
                    )
                    continue

                # No retry target or retries exhausted — fail
                await self._emit_complete(gate_result, pipeline_start_time)
                return gate_result

            # Step 1b: Skip already-completed nodes (resume path)
            if resumed and current_node.id in {nid for nid in self.completed_nodes}:
                # Re-select the edge this node used last time
                edge = select_edge(
                    current_node.id,
                    self.node_outcomes.get(
                        current_node.id, Outcome(status=StageStatus.SUCCESS)
                    ),
                    self.context,
                    self.graph,
                )
                if edge is None:
                    fail_outcome = Outcome(
                        status=StageStatus.FAIL,
                        failure_reason=f"No matching edge from resumed node '{current_node.id}'",
                    )
                    await self._emit(
                        PIPELINE_ERROR,
                        {
                            "node_id": current_node.id,
                            "error_type": "no_matching_edge",
                            "message": fail_outcome.failure_reason or "",
                        },
                    )
                    await self._emit_complete(fail_outcome, pipeline_start_time)
                    return fail_outcome
                current_node = self.graph.nodes[edge.to_node]
                continue

            # Step 2: Execute node handler with retry policy
            handler = self.handler_registry.get(current_node)
            handler_type = current_node.type or current_node.shape
            await self._emit(
                PIPELINE_NODE_START,
                {
                    "node_id": current_node.id,
                    "handler_type": handler_type,
                    "attempt": 1,
                },
            )

            node_start_time = time.monotonic()
            retry_policy = RetryPolicy.from_node(current_node, self.graph)

            # Per-node timeout enforcement: wrap handler execution with
            # asyncio.timeout when the node declares a timeout attribute.
            # DOT timeout values are in seconds (per NLSpec timeout_seconds).
            node_timeout_raw = current_node.timeout
            if node_timeout_raw:
                timeout_s = float(node_timeout_raw)
                try:
                    async with asyncio.timeout(timeout_s):
                        outcome = await execute_with_retry(
                            handler,
                            current_node,
                            self.context,
                            self.graph,
                            self.logs_root,
                            retry_policy,
                            hooks=self.hooks,
                        )
                except asyncio.TimeoutError:
                    node_duration_ms = (time.monotonic() - node_start_time) * 1000
                    outcome = Outcome(
                        status=StageStatus.FAIL,
                        notes=f"Node '{current_node.id}' timed out after {timeout_s}s",
                        failure_reason="timeout",
                    )
                    await self._emit(
                        PIPELINE_NODE_COMPLETE,
                        {
                            "node_id": current_node.id,
                            "status": "timeout",
                            "duration_ms": node_duration_ms,
                            "notes": outcome.notes,
                            "failure_reason": outcome.failure_reason,
                        },
                    )
            else:
                outcome = await execute_with_retry(
                    handler,
                    current_node,
                    self.context,
                    self.graph,
                    self.logs_root,
                    retry_policy,
                    hooks=self.hooks,
                )
            node_duration_ms = (time.monotonic() - node_start_time) * 1000

            # Step 2.5: Check for cancellation after node execution
            if self._check_cancelled():
                cancelled_outcome = Outcome(
                    status=StageStatus.FAIL,
                    notes=f"Pipeline cancelled after node '{current_node.id}' completed",
                    failure_reason="cancelled",
                )
                await self._emit(
                    PIPELINE_COMPLETE,
                    {
                        "status": "cancelled",
                        "total_nodes_executed": len(self.completed_nodes),
                        "duration_ms": (time.monotonic() - pipeline_start_time) * 1000,
                    },
                )
                return cancelled_outcome

            # L-9: auto_status — override non-success to SUCCESS when enabled
            if current_node.auto_status is True and not outcome.is_success:
                logger.debug(
                    "Node '%s' has auto_status=true; overriding %s to SUCCESS",
                    current_node.id,
                    outcome.status.value,
                )
                outcome = Outcome(
                    status=StageStatus.SUCCESS,
                    notes=f"auto_status override (was {outcome.status.value})",
                    context_updates=outcome.context_updates,
                    preferred_label=outcome.preferred_label,
                    suggested_next_ids=outcome.suggested_next_ids,
                )

            # Step 3: Record completion
            self.completed_nodes.append(current_node.id)
            self.node_outcomes[current_node.id] = outcome
            logger.debug("Node %s completed: %s", current_node.id, outcome.status.value)

            await self._emit(
                PIPELINE_NODE_COMPLETE,
                {
                    "node_id": current_node.id,
                    "status": outcome.status.value,
                    "duration_ms": node_duration_ms,
                    "notes": outcome.notes,
                    "failure_reason": outcome.failure_reason,
                },
            )

            # Step 3b: Write per-node status.json
            self._write_node_status(current_node.id, outcome, node_duration_ms)

            # Step 4: Apply context updates from outcome
            if outcome.context_updates:
                self.context.update(outcome.context_updates)
            self.context.set("outcome", outcome.status.value)
            if outcome.preferred_label:
                self.context.set("preferred_label", outcome.preferred_label)

            # Step 4b: Save checkpoint after each node
            self._save_checkpoint(current_node.id)
            await self._emit(
                PIPELINE_CHECKPOINT,
                {
                    "node_id": current_node.id,
                    "checkpoint_path": self._checkpoint_path,
                },
            )

            # Step 5: Select next edge(s) — detect multi-edge fan-out
            all_matching = select_all_matching_edges(
                current_node.id, outcome, self.context, self.graph
            )

            if len(all_matching) > 1:
                # Multi-edge fan-out: execute all targets in parallel
                logger.info(
                    "Multi-edge fan-out from '%s': %d parallel targets",
                    current_node.id,
                    len(all_matching),
                )

                await self._execute_parallel_fan_out(all_matching, pipeline_start_time)

                # Find convergence node: the first node that all parallel
                # targets share as a common outgoing edge target
                fan_in_node_id = self._find_fan_in_node(
                    [e.to_node for e in all_matching]
                )
                if fan_in_node_id is None:
                    fail_outcome = Outcome(
                        status=StageStatus.FAIL,
                        failure_reason=(
                            f"Multi-edge fan-out from '{current_node.id}' "
                            f"has no convergence (fan-in) node"
                        ),
                    )
                    await self._emit_complete(fail_outcome, pipeline_start_time)
                    return fail_outcome

                # Store parallel results in context for the fan-in node
                current_node = self.graph.nodes[fan_in_node_id]
                continue

            # Single-edge selection (normal path)
            edge = select_edge(current_node.id, outcome, self.context, self.graph)
            if edge is None:
                # Try failure routing: node/graph retry targets
                retry_node = self._resolve_failure_retry_target(current_node)
                if (
                    retry_node is not None
                    and failure_routing_retries < self._MAX_GOAL_GATE_RETRIES
                ):
                    failure_routing_retries += 1
                    logger.info(
                        "No matching edge from '%s', failure-routing to '%s' "
                        "(attempt %d)",
                        current_node.id,
                        retry_node.id,
                        failure_routing_retries,
                    )
                    current_node = retry_node
                    continue

                fail_outcome = Outcome(
                    status=StageStatus.FAIL,
                    failure_reason=f"No matching edge from node '{current_node.id}'",
                )
                await self._emit(
                    PIPELINE_ERROR,
                    {
                        "node_id": current_node.id,
                        "error_type": "no_matching_edge",
                        "message": fail_outcome.failure_reason or "",
                    },
                )
                await self._emit_complete(fail_outcome, pipeline_start_time)
                return fail_outcome

            await self._emit(
                PIPELINE_EDGE_SELECTED,
                {
                    "from_node": edge.from_node,
                    "to_node": edge.to_node,
                    "edge_label": edge.label,
                },
            )

            # Step 6: Handle loop_restart edge attribute (NLSpec Section 174)
            if edge.loop_restart is True:
                self.iteration_count += 1
                iteration_dir = os.path.join(
                    self.logs_root, f"iteration_{self.iteration_count}"
                )
                os.makedirs(iteration_dir, exist_ok=True)
                logger.info(
                    "loop_restart: iteration %d, fresh log dir '%s', "
                    "continuing from '%s'",
                    self.iteration_count,
                    iteration_dir,
                    edge.to_node,
                )
                # Reset engine state for clean re-execution
                self.completed_nodes.clear()
                self.node_outcomes.clear()
                goal_gate_retries = 0
                failure_routing_retries = 0

            # Step 7: Advance to next node
            current_node = self.graph.nodes[edge.to_node]

    async def _run_from(
        self,
        start_node_id: str,
        *,
        context: PipelineContext | None = None,
    ) -> Outcome:
        """Execute a subgraph starting from the given node.

        Walks from *start_node_id* until an exit node is reached, no
        outgoing edges exist, or the node is not in the graph.

        This is the subgraph runner used by ParallelHandler and
        ManagerLoopHandler to execute branches and child subgraphs.

        Args:
            start_node_id: Node ID to begin execution from.
            context: Optional isolated context for this subgraph run.
                     If None, uses the engine's main context.

        Returns:
            The final Outcome of the subgraph execution.
        """
        ctx = context if context is not None else self.context

        if start_node_id not in self.graph.nodes:
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason=f"Subgraph start node '{start_node_id}' not found in graph",
            )

        current_node = self.graph.nodes[start_node_id]
        last_outcome: Outcome | None = None

        # Safety bound to prevent infinite loops
        max_steps = len(self.graph.nodes) * self._MAX_GOAL_GATE_RETRIES

        for _step in range(max_steps):
            # Check cancellation in subgraph runner too
            if self._check_cancelled():
                return Outcome(
                    status=StageStatus.FAIL,
                    notes="Pipeline cancelled during subgraph execution",
                    failure_reason="cancelled",
                )

            # Check for terminal node (exit or fan_in)
            if current_node.is_exit_node() or current_node.shape == "tripleoctagon":
                return last_outcome or Outcome(
                    status=StageStatus.SUCCESS,
                    notes="Subgraph reached terminal node",
                )

            # Execute node handler (no retry policy in subgraph -- parent manages retries)
            handler = self.handler_registry.get(current_node)

            # Skip start nodes (no-op)
            if current_node.is_start_node():
                outcome = Outcome(status=StageStatus.SUCCESS)
            else:
                try:
                    outcome = await handler.execute(
                        current_node, ctx, self.graph, self.logs_root
                    )
                except Exception as exc:
                    return Outcome(
                        status=StageStatus.FAIL,
                        failure_reason=f"Subgraph node '{current_node.id}' raised: {exc}",
                    )

            last_outcome = outcome

            # Apply context updates
            if outcome.context_updates:
                ctx.update(outcome.context_updates)
            ctx.set("outcome", outcome.status.value)
            if outcome.preferred_label:
                ctx.set("preferred_label", outcome.preferred_label)

            # Select next edge
            edge = select_edge(current_node.id, outcome, ctx, self.graph)
            if edge is None:
                # No outgoing edge -- subgraph is complete
                return outcome

            current_node = self.graph.nodes[edge.to_node]

        # Safety bound exceeded
        return Outcome(
            status=StageStatus.FAIL,
            failure_reason=f"Subgraph exceeded {max_steps} steps (safety bound)",
        )

    def _initialize_context(self, goal: str | None) -> None:
        """Mirror graph attributes into context.

        Spec Section 3.1: Initialize phase.
        """
        # Set goal from argument or graph attribute
        effective_goal = goal or self.graph.goal
        if effective_goal:
            self.context.set("graph.goal", effective_goal)

        # Mirror graph-level attributes
        for key, value in self.graph.graph_attrs.items():
            self.context.set(f"graph.{key}", value)

    def _find_start_node(self) -> Node:
        """Find the start node.

        Resolution order (L-21, Spec Section 3.2, NLSpec line 344):
          1. shape=Mdiamond
          2. node_type="start" attribute
          3. id="start" (case-insensitive)

        Raises ValueError if no start node can be resolved.
        """
        # Priority 1: shape=Mdiamond
        for node in self.graph.nodes.values():
            if node.shape == "Mdiamond":
                return node

        # Priority 2: node_type="start" attribute
        for node in self.graph.nodes.values():
            if node.attrs.get("node_type") == "start":
                logger.debug(
                    "No Mdiamond node found; using node_type='start' node '%s'",
                    node.id,
                )
                return node

        # Priority 3: id="start" (case-insensitive, L-21)
        for node in self.graph.nodes.values():
            if node.id.lower() == "start":
                logger.debug(
                    "No Mdiamond/node_type node found; using id='%s' as start node",
                    node.id,
                )
                return node

        raise ValueError(
            "No start node found (no shape=Mdiamond, no node_type='start', "
            "and no id='start'/'Start')"
        )

    async def _check_goal_gates(self) -> Outcome:
        """Check goal gate satisfaction at exit.

        Spec Section 3.4: Goal Gate Enforcement.

        Returns:
            SUCCESS/PARTIAL_SUCCESS if all goal gates passed.
            FAIL with suggested_next_ids=[retry_target] if a gate is
            unsatisfied and a retry target exists.
            FAIL without suggested_next_ids if no retry target.
        """
        unsatisfied: list[tuple[str, Outcome]] = []
        satisfied: list[str] = []
        for node_id, outcome in self.node_outcomes.items():
            node = self.graph.nodes.get(node_id)
            if node is None:
                continue
            if node.attrs.get("goal_gate") is True:
                if outcome.is_success:
                    satisfied.append(node_id)
                else:
                    unsatisfied.append((node_id, outcome))

        unsatisfied_ids = [nid for nid, _ in unsatisfied]
        await self._emit(
            PIPELINE_GOAL_GATE_CHECK,
            {
                "satisfied": satisfied,
                "unsatisfied": unsatisfied_ids,
            },
        )

        if not unsatisfied:
            # All goal gates satisfied (or none exist)
            if self.completed_nodes:
                last_id = self.completed_nodes[-1]
                last_outcome = self.node_outcomes.get(last_id)
                if last_outcome:
                    return last_outcome
            return Outcome(status=StageStatus.SUCCESS, notes="Pipeline completed")

        # Find the first unsatisfied gate and its retry target
        gate_node_id, gate_outcome = unsatisfied[0]
        gate_node = self.graph.nodes[gate_node_id]

        # Retry target resolution: node > node fallback > graph > graph fallback
        retry_target = (
            gate_node.attrs.get("retry_target")
            or gate_node.attrs.get("fallback_retry_target")
            or self.graph.graph_attrs.get("retry_target")
            or self.graph.graph_attrs.get("fallback_retry_target")
        )

        failure_reason = f"Unsatisfied goal gates: {unsatisfied_ids}"

        if retry_target and retry_target in self.graph.nodes:
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason=failure_reason,
                suggested_next_ids=[retry_target],
            )

        return Outcome(
            status=StageStatus.FAIL,
            failure_reason=failure_reason,
        )

    def _try_resume_from_checkpoint(self) -> bool:
        """Try to load and restore state from an existing checkpoint.

        Returns True if a checkpoint was loaded (resume mode), False otherwise.

        Spec Section 5.3: Resume behavior.
        """
        if not os.path.exists(self._checkpoint_path):
            return False

        try:
            cp = load_checkpoint(self._checkpoint_path)
        except (FileNotFoundError, KeyError, ValueError):
            logger.warning("Failed to load checkpoint, starting fresh")
            return False

        # Restore context from checkpoint
        for key, value in cp.context_snapshot.items():
            self.context.set(key, value)

        # M-23: Degrade fidelity from "full" to "summary:high" on resume.
        # The full session context is lost after a crash, so full fidelity
        # cannot be honoured.  Other modes pass through unchanged.
        restored_fidelity = self.context.get("graph.default_fidelity")
        if restored_fidelity == "full":
            self.context.set("graph.default_fidelity", "summary:high")
            logger.info(
                "Checkpoint resume: degraded fidelity from 'full' to "
                "'summary:high' (full session context unavailable)"
            )

        # Restore completed nodes and outcomes
        for node_id, status_str in cp.completed_nodes.items():
            if node_id not in self.completed_nodes:
                self.completed_nodes.append(node_id)
            # Reconstruct a minimal Outcome from saved status
            self.node_outcomes[node_id] = Outcome(
                status=StageStatus(status_str),
                notes=cp.node_outcomes.get(node_id, {}).get("notes"),
                failure_reason=cp.node_outcomes.get(node_id, {}).get("failure_reason"),
                preferred_label=cp.node_outcomes.get(node_id, {}).get(
                    "preferred_label"
                ),
            )

        logger.info(
            "Resumed from checkpoint: %d nodes completed, current=%s",
            len(self.completed_nodes),
            cp.current_node,
        )
        return True

    def _save_checkpoint(self, current_node_id: str) -> None:
        """Save a checkpoint after a node execution.

        Spec Section 5.3: Checkpoint.save.
        """
        os.makedirs(self.logs_root, exist_ok=True)

        # Serialize node outcomes
        serialized_outcomes: dict[str, dict[str, str | None]] = {}
        for node_id, outcome in self.node_outcomes.items():
            serialized_outcomes[node_id] = {
                "status": outcome.status.value,
                "notes": outcome.notes,
                "failure_reason": outcome.failure_reason,
                "preferred_label": outcome.preferred_label,
            }

        cp = Checkpoint(
            current_node=current_node_id,
            completed_nodes={
                nid: self.node_outcomes[nid].status.value
                for nid in self.completed_nodes
                if nid in self.node_outcomes
            },
            context_snapshot=self.context.snapshot(),
            node_outcomes=serialized_outcomes,
            timestamp=datetime.now(timezone.utc).isoformat(),
            logs=self.context.get_logs(),  # L-7: include logs in checkpoint
        )
        save_checkpoint(cp, self._checkpoint_path)

    # -- Run directory helpers -----------------------------------------------

    def _write_manifest(self, goal: str | None) -> None:
        """Write manifest.json and create the artifacts/ directory.

        Spec Section 5.6: Run Directory Structure.
        """
        os.makedirs(self.logs_root, exist_ok=True)
        os.makedirs(os.path.join(self.logs_root, "artifacts"), exist_ok=True)

        manifest = {
            "graph_name": self.graph.name,
            "goal": self.graph.goal or goal or "",
            "start_time": datetime.now(timezone.utc).isoformat(),
            "node_count": len(self.graph.nodes),
            "edge_count": len(self.graph.edges),
        }
        manifest_path = os.path.join(self.logs_root, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    def _write_node_status(
        self, node_id: str, outcome: Outcome, duration_ms: float
    ) -> None:
        """Write status.json for a node after execution.

        Spec Section 5.6: Per-node status.json.
        """
        node_dir = os.path.join(self.logs_root, node_id)
        os.makedirs(node_dir, exist_ok=True)
        status = {
            "node_id": node_id,
            "outcome": outcome.status.value,
            "status": outcome.status.value,  # backward compat (M-19)
            "preferred_next_label": outcome.preferred_label,
            "suggested_next_ids": outcome.suggested_next_ids,
            "context_updates": outcome.context_updates,
            "duration_ms": duration_ms,
            "notes": outcome.notes,
            "failure_reason": outcome.failure_reason,
        }
        status_path = os.path.join(node_dir, "status.json")
        with open(status_path, "w") as f:
            json.dump(status, f, indent=2)

    # -- Multi-edge parallel fan-out helpers -----------------------------------

    async def _execute_parallel_fan_out(
        self,
        edges: list,
        pipeline_start_time: float,
    ) -> list[dict[str, Any]]:
        """Execute multiple branch targets in parallel with isolated contexts.

        Each branch gets a clone of the current context for isolation
        (NLSpec Section 798). Results are collected and context_updates
        from all branches are merged back into the main context.
        """

        async def run_branch(target_node_id: str) -> dict[str, Any]:
            branch_context = self.context.clone()
            node = self.graph.nodes[target_node_id]
            handler = self.handler_registry.get(node)
            handler_type = node.type or node.shape

            await self._emit(
                PIPELINE_NODE_START,
                {"node_id": node.id, "handler_type": handler_type, "attempt": 1},
            )

            node_start = time.monotonic()
            retry_policy = RetryPolicy.from_node(node, self.graph)

            try:
                outcome = await execute_with_retry(
                    handler,
                    node,
                    branch_context,
                    self.graph,
                    self.logs_root,
                    retry_policy,
                    hooks=self.hooks,
                )
            except Exception as exc:
                outcome = Outcome(
                    status=StageStatus.FAIL,
                    failure_reason=f"Parallel branch '{target_node_id}' raised: {exc}",
                )

            node_duration = (time.monotonic() - node_start) * 1000

            # Record completion in the main engine state
            self.completed_nodes.append(target_node_id)
            self.node_outcomes[target_node_id] = outcome

            await self._emit(
                PIPELINE_NODE_COMPLETE,
                {
                    "node_id": target_node_id,
                    "status": outcome.status.value,
                    "duration_ms": node_duration,
                    "notes": outcome.notes,
                    "failure_reason": outcome.failure_reason,
                },
            )
            self._write_node_status(target_node_id, outcome, node_duration)

            return {
                "node_id": target_node_id,
                "status": outcome.status.value,
                "notes": outcome.notes,
                "failure_reason": outcome.failure_reason,
                "context_updates": outcome.context_updates,
            }

        # Execute all branches concurrently
        tasks = [run_branch(edge.to_node) for edge in edges]
        results = list(await asyncio.gather(*tasks))

        # Apply context_updates from all branches
        for result in results:
            updates = result.get("context_updates")
            if updates:
                self.context.update(updates)

        return results

    def _find_fan_in_node(self, parallel_target_ids: list[str]) -> str | None:
        """Find the convergence node where all parallel branches meet.

        Looks for a node that is an outgoing edge target from ALL
        parallel target nodes (intersection of downstream neighbors).
        """
        if not parallel_target_ids:
            return None

        # Collect all outgoing edge targets from each parallel node
        target_sets: list[set[str]] = []
        for node_id in parallel_target_ids:
            targets = {e.to_node for e in self.graph.outgoing_edges(node_id)}
            target_sets.append(targets)

        if not target_sets:
            return None

        # Fan-in node = intersection of all target sets
        common = target_sets[0]
        for ts in target_sets[1:]:
            common = common & ts

        if not common:
            return None

        # If multiple common targets, pick the first alphabetically
        return sorted(common)[0]

    # -- Failure routing helpers ----------------------------------------------

    def _resolve_failure_retry_target(self, node: Node) -> Node | None:
        """Resolve a retry target when no edge matches after node execution.

        Fallback chain (first match wins):
        1. node.retry_target
        2. node.fallback_retry_target
        3. graph.retry_target
        4. graph.fallback_retry_target

        Returns the target Node or None if no valid target exists.
        """
        target_id = (
            node.attrs.get("retry_target")
            or node.attrs.get("fallback_retry_target")
            or self.graph.graph_attrs.get("retry_target")
            or self.graph.graph_attrs.get("fallback_retry_target")
        )
        if target_id and target_id in self.graph.nodes:
            return self.graph.nodes[target_id]
        return None

    # -- Event helpers -------------------------------------------------------

    async def _emit(self, event_name: str, data: dict[str, Any]) -> None:
        """Emit an event via hooks, if provided."""
        if self.hooks is not None:
            await self.hooks.emit(event_name, data)

    async def _emit_complete(self, outcome: Outcome, start_time: float) -> None:
        """Emit the pipeline:complete event."""
        duration_ms = (time.monotonic() - start_time) * 1000
        await self._emit(
            PIPELINE_COMPLETE,
            {
                "status": outcome.status.value,
                "total_nodes_executed": len(self.completed_nodes),
                "duration_ms": duration_ms,
            },
        )
