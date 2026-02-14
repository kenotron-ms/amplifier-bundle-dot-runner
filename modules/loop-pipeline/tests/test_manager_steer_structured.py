"""Tests for structured steering messages in the manager loop (Fix 2.8).

Spec coverage: MGR-005 — the manager loop handler should follow a full
observe/guard/steer cycle. The steering step should format a structured
message that includes:
  - Previous cycle's status and failure details
  - Current cycle number and max_cycles for context
  - Remaining cycles budget
  - Clear actionable instruction

This improves on the previous flat-text format by giving child agents
structured context they can parse and act on.
"""

from __future__ import annotations

import pytest

from amplifier_module_loop_pipeline.context import PipelineContext
from amplifier_module_loop_pipeline.graph import Edge, Graph, Node
from amplifier_module_loop_pipeline.handlers.manager_loop import ManagerLoopHandler
from amplifier_module_loop_pipeline.outcome import Outcome, StageStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_graph(
    *,
    manager_attrs: dict | None = None,
    has_child_edge: bool = True,
) -> Graph:
    nodes = {
        "start": Node(id="start", shape="Mdiamond"),
        "manager": Node(
            id="manager",
            shape="house",
            label="Sprint Manager",
            attrs=manager_attrs or {},
        ),
        "child_task": Node(id="child_task", shape="box", label="Do work"),
        "exit": Node(id="exit", shape="Msquare"),
    }
    edges = [
        Edge(from_node="start", to_node="manager"),
    ]
    if has_child_edge:
        edges.append(Edge(from_node="manager", to_node="child_task"))
    edges.append(Edge(from_node="child_task", to_node="exit"))
    return Graph(name="test_manager", nodes=nodes, edges=edges)


# =====================================================================
# Structured steering message tests
# =====================================================================


class TestStructuredSteeringMessage:
    """Steering message should be structured with clear sections."""

    @pytest.mark.asyncio
    async def test_steering_includes_cycle_count(self):
        """Steering message includes 'Cycle X of Y' context."""
        captured_contexts: list[PipelineContext] = []

        async def capturing_runner(node_id, context, graph, logs_root):
            captured_contexts.append(context)
            if len(captured_contexts) < 2:
                return Outcome(status=StageStatus.FAIL, failure_reason="broken")
            return Outcome(status=StageStatus.SUCCESS)

        handler = ManagerLoopHandler(subgraph_runner=capturing_runner)
        graph = _make_graph(
            manager_attrs={
                "manager.max_cycles": "5",
                "manager.actions": "observe,steer,wait",
                "manager.poll_interval": "0s",
            }
        )
        ctx = PipelineContext()
        await handler.execute(graph.nodes["manager"], ctx, graph, "/tmp")

        steering = captured_contexts[1].get("manager.steering")
        assert steering is not None
        # Should mention cycle 1 (the previous cycle that failed)
        assert "1" in steering
        # Should mention max_cycles for budget context
        assert "5" in steering

    @pytest.mark.asyncio
    async def test_steering_includes_remaining_cycles(self):
        """Steering message includes remaining cycles budget."""
        captured_contexts: list[PipelineContext] = []

        async def capturing_runner(node_id, context, graph, logs_root):
            captured_contexts.append(context)
            if len(captured_contexts) < 3:
                return Outcome(status=StageStatus.FAIL, failure_reason="still broken")
            return Outcome(status=StageStatus.SUCCESS)

        handler = ManagerLoopHandler(subgraph_runner=capturing_runner)
        graph = _make_graph(
            manager_attrs={
                "manager.max_cycles": "5",
                "manager.actions": "observe,steer,wait",
                "manager.poll_interval": "0s",
            }
        )
        ctx = PipelineContext()
        await handler.execute(graph.nodes["manager"], ctx, graph, "/tmp")

        # Third cycle's steering (after 2 failures)
        steering = captured_contexts[2].get("manager.steering")
        assert steering is not None
        # Should mention remaining cycles (5 - 2 = 3 remaining)
        assert "3" in steering
        assert "remaining" in steering.lower()

    @pytest.mark.asyncio
    async def test_steering_is_multiline_structured(self):
        """Steering message uses multi-line structured format, not flat text."""
        captured_contexts: list[PipelineContext] = []

        async def capturing_runner(node_id, context, graph, logs_root):
            captured_contexts.append(context)
            if len(captured_contexts) < 2:
                return Outcome(
                    status=StageStatus.FAIL,
                    failure_reason="compilation error in main.py line 42",
                    notes="Build step failed",
                )
            return Outcome(status=StageStatus.SUCCESS)

        handler = ManagerLoopHandler(subgraph_runner=capturing_runner)
        graph = _make_graph(
            manager_attrs={
                "manager.max_cycles": "5",
                "manager.actions": "observe,steer,wait",
                "manager.poll_interval": "0s",
            }
        )
        ctx = PipelineContext()
        await handler.execute(graph.nodes["manager"], ctx, graph, "/tmp")

        steering = captured_contexts[1].get("manager.steering")
        assert steering is not None
        # Should be multi-line (structured), not a single line
        assert "\n" in steering

    @pytest.mark.asyncio
    async def test_steering_preserves_failure_reason(self):
        """Structured format still includes the full failure reason."""
        captured_contexts: list[PipelineContext] = []

        async def capturing_runner(node_id, context, graph, logs_root):
            captured_contexts.append(context)
            if len(captured_contexts) < 2:
                return Outcome(
                    status=StageStatus.FAIL,
                    failure_reason="tests failing: 3 of 10 assertions broken",
                    notes="Unit tests did not pass",
                )
            return Outcome(status=StageStatus.SUCCESS)

        handler = ManagerLoopHandler(subgraph_runner=capturing_runner)
        graph = _make_graph(
            manager_attrs={
                "manager.max_cycles": "5",
                "manager.actions": "observe,steer,wait",
                "manager.poll_interval": "0s",
            }
        )
        ctx = PipelineContext()
        await handler.execute(graph.nodes["manager"], ctx, graph, "/tmp")

        steering = captured_contexts[1].get("manager.steering")
        assert steering is not None
        # Must still include the actual failure details (existing test contract)
        assert "tests failing" in steering
        assert "3 of 10 assertions broken" in steering

    @pytest.mark.asyncio
    async def test_steering_preserves_notes(self):
        """Structured format includes notes from the previous cycle."""
        captured_contexts: list[PipelineContext] = []

        async def capturing_runner(node_id, context, graph, logs_root):
            captured_contexts.append(context)
            if len(captured_contexts) < 2:
                return Outcome(
                    status=StageStatus.FAIL,
                    failure_reason="build failed",
                    notes="3 warnings, 1 error in output",
                )
            return Outcome(status=StageStatus.SUCCESS)

        handler = ManagerLoopHandler(subgraph_runner=capturing_runner)
        graph = _make_graph(
            manager_attrs={
                "manager.max_cycles": "5",
                "manager.actions": "observe,steer,wait",
                "manager.poll_interval": "0s",
            }
        )
        ctx = PipelineContext()
        await handler.execute(graph.nodes["manager"], ctx, graph, "/tmp")

        steering = captured_contexts[1].get("manager.steering")
        assert steering is not None
        assert "3 warnings, 1 error in output" in steering

    @pytest.mark.asyncio
    async def test_steering_status_included(self):
        """Steering includes the previous cycle's status value."""
        captured_contexts: list[PipelineContext] = []

        async def capturing_runner(node_id, context, graph, logs_root):
            captured_contexts.append(context)
            if len(captured_contexts) < 2:
                return Outcome(status=StageStatus.FAIL, failure_reason="broke")
            return Outcome(status=StageStatus.SUCCESS)

        handler = ManagerLoopHandler(subgraph_runner=capturing_runner)
        graph = _make_graph(
            manager_attrs={
                "manager.max_cycles": "5",
                "manager.actions": "observe,steer,wait",
                "manager.poll_interval": "0s",
            }
        )
        ctx = PipelineContext()
        await handler.execute(graph.nodes["manager"], ctx, graph, "/tmp")

        steering = captured_contexts[1].get("manager.steering")
        assert steering is not None
        assert "fail" in steering.lower()

    @pytest.mark.asyncio
    async def test_no_steering_on_first_cycle(self):
        """First cycle has no steering (no prior outcome)."""
        captured_contexts: list[PipelineContext] = []

        async def capturing_runner(node_id, context, graph, logs_root):
            captured_contexts.append(context)
            return Outcome(status=StageStatus.SUCCESS)

        handler = ManagerLoopHandler(subgraph_runner=capturing_runner)
        graph = _make_graph(
            manager_attrs={
                "manager.max_cycles": "5",
                "manager.actions": "observe,steer,wait",
                "manager.poll_interval": "0s",
            }
        )
        ctx = PipelineContext()
        await handler.execute(graph.nodes["manager"], ctx, graph, "/tmp")

        assert captured_contexts[0].get("manager.steering") is None

    @pytest.mark.asyncio
    async def test_steering_with_no_failure_reason(self):
        """Steering handles outcomes that have no failure_reason gracefully."""
        captured_contexts: list[PipelineContext] = []

        async def capturing_runner(node_id, context, graph, logs_root):
            captured_contexts.append(context)
            if len(captured_contexts) < 2:
                return Outcome(status=StageStatus.FAIL)  # no failure_reason
            return Outcome(status=StageStatus.SUCCESS)

        handler = ManagerLoopHandler(subgraph_runner=capturing_runner)
        graph = _make_graph(
            manager_attrs={
                "manager.max_cycles": "5",
                "manager.actions": "observe,steer,wait",
                "manager.poll_interval": "0s",
            }
        )
        ctx = PipelineContext()
        await handler.execute(graph.nodes["manager"], ctx, graph, "/tmp")

        steering = captured_contexts[1].get("manager.steering")
        assert steering is not None
        # Should still be valid even without failure_reason
        assert "fail" in steering.lower()
