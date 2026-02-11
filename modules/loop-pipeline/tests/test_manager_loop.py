"""Tests for the manager loop handler — supervisor pattern over a child subgraph.

The manager loop (shape=house) orchestrates observe/evaluate/act cycles
over a child subgraph. It runs the child, checks a guard condition, and
loops until the guard is satisfied or max cycles are exhausted.

Spec coverage: MGR-001-010, COMP-001-002, Section 4.11.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

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
    """Build a minimal graph with a manager node and optional child target."""
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


def _make_runner(outcomes: list[Outcome]) -> AsyncMock:
    """Create a mock subgraph_runner that returns outcomes in sequence."""
    runner = AsyncMock()
    runner.side_effect = list(outcomes)
    return runner


# ---------------------------------------------------------------------------
# Core behavior tests
# ---------------------------------------------------------------------------


class TestManagerLoopExecution:
    """Manager loop runs child subgraph and evaluates outcomes."""

    @pytest.mark.asyncio
    async def test_child_success_stops_loop(self):
        """When the child succeeds, the manager returns SUCCESS after 1 cycle."""
        runner = _make_runner(
            [
                Outcome(status=StageStatus.SUCCESS, notes="child ok"),
            ]
        )
        handler = ManagerLoopHandler(subgraph_runner=runner)
        graph = _make_graph(manager_attrs={"manager.max_cycles": "5"})
        ctx = PipelineContext()

        result = await handler.execute(graph.nodes["manager"], ctx, graph, "/tmp")

        assert result.status == StageStatus.SUCCESS
        assert runner.call_count == 1

    @pytest.mark.asyncio
    async def test_child_fail_retries_then_succeeds(self):
        """Manager retries when child fails, stops when child succeeds."""
        runner = _make_runner(
            [
                Outcome(status=StageStatus.FAIL, failure_reason="broken"),
                Outcome(status=StageStatus.FAIL, failure_reason="still broken"),
                Outcome(status=StageStatus.SUCCESS, notes="fixed"),
            ]
        )
        handler = ManagerLoopHandler(subgraph_runner=runner)
        graph = _make_graph(
            manager_attrs={
                "manager.max_cycles": "5",
                "manager.poll_interval": "0s",
            }
        )
        ctx = PipelineContext()

        result = await handler.execute(graph.nodes["manager"], ctx, graph, "/tmp")

        assert result.status == StageStatus.SUCCESS
        assert runner.call_count == 3

    @pytest.mark.asyncio
    async def test_max_cycles_exhausted(self):
        """When all cycles are used without success, returns FAIL."""
        runner = _make_runner(
            [
                Outcome(status=StageStatus.FAIL, failure_reason="nope"),
                Outcome(status=StageStatus.FAIL, failure_reason="nope"),
                Outcome(status=StageStatus.FAIL, failure_reason="nope"),
            ]
        )
        handler = ManagerLoopHandler(subgraph_runner=runner)
        graph = _make_graph(
            manager_attrs={
                "manager.max_cycles": "3",
                "manager.poll_interval": "0s",
            }
        )
        ctx = PipelineContext()

        result = await handler.execute(graph.nodes["manager"], ctx, graph, "/tmp")

        assert result.status == StageStatus.FAIL
        assert runner.call_count == 3
        assert "3" in (result.failure_reason or "")

    @pytest.mark.asyncio
    async def test_partial_success_stops_loop(self):
        """PARTIAL_SUCCESS from child also satisfies the default guard."""
        runner = _make_runner(
            [
                Outcome(status=StageStatus.PARTIAL_SUCCESS, notes="close enough"),
            ]
        )
        handler = ManagerLoopHandler(subgraph_runner=runner)
        graph = _make_graph(manager_attrs={"manager.max_cycles": "5"})
        ctx = PipelineContext()

        result = await handler.execute(graph.nodes["manager"], ctx, graph, "/tmp")

        assert result.is_success
        assert runner.call_count == 1


# ---------------------------------------------------------------------------
# Guard / stop condition tests
# ---------------------------------------------------------------------------


class TestManagerStopCondition:
    """Stop condition (guard) controls when the manager exits the loop."""

    @pytest.mark.asyncio
    async def test_stop_condition_evaluated(self):
        """Guard condition 'outcome=success' stops on first success."""
        runner = _make_runner(
            [
                Outcome(status=StageStatus.FAIL),
                Outcome(status=StageStatus.SUCCESS),
            ]
        )
        handler = ManagerLoopHandler(subgraph_runner=runner)
        graph = _make_graph(
            manager_attrs={
                "manager.max_cycles": "5",
                "manager.stop_condition": "outcome=success",
                "manager.poll_interval": "0s",
            }
        )
        ctx = PipelineContext()

        result = await handler.execute(graph.nodes["manager"], ctx, graph, "/tmp")

        assert result.status == StageStatus.SUCCESS
        assert runner.call_count == 2

    @pytest.mark.asyncio
    async def test_stop_condition_not_met_keeps_looping(self):
        """If guard never satisfied, loop exhausts max_cycles."""
        runner = _make_runner(
            [
                Outcome(status=StageStatus.PARTIAL_SUCCESS),
                Outcome(status=StageStatus.PARTIAL_SUCCESS),
            ]
        )
        handler = ManagerLoopHandler(subgraph_runner=runner)
        graph = _make_graph(
            manager_attrs={
                "manager.max_cycles": "2",
                "manager.stop_condition": "outcome=success",
                "manager.poll_interval": "0s",
            }
        )
        ctx = PipelineContext()

        result = await handler.execute(graph.nodes["manager"], ctx, graph, "/tmp")

        # Guard requires "success" exactly, partial_success doesn't match
        assert result.status == StageStatus.FAIL
        assert runner.call_count == 2


# ---------------------------------------------------------------------------
# Context recording tests
# ---------------------------------------------------------------------------


class TestManagerContextRecording:
    """Manager records cycle telemetry in the pipeline context."""

    @pytest.mark.asyncio
    async def test_records_cycle_status(self):
        """Each cycle's child status is recorded in context."""
        runner = _make_runner(
            [
                Outcome(status=StageStatus.FAIL),
                Outcome(status=StageStatus.SUCCESS),
            ]
        )
        handler = ManagerLoopHandler(subgraph_runner=runner)
        graph = _make_graph(
            manager_attrs={
                "manager.max_cycles": "5",
                "manager.poll_interval": "0s",
            }
        )
        ctx = PipelineContext()

        await handler.execute(graph.nodes["manager"], ctx, graph, "/tmp")

        assert ctx.get("manager.cycle_1.status") == "fail"
        assert ctx.get("manager.cycle_2.status") == "success"
        assert ctx.get("manager.last_child_status") == "success"

    @pytest.mark.asyncio
    async def test_outcome_contains_context_updates(self):
        """The returned outcome includes last_stage and cycle count."""
        runner = _make_runner(
            [
                Outcome(status=StageStatus.SUCCESS),
            ]
        )
        handler = ManagerLoopHandler(subgraph_runner=runner)
        graph = _make_graph(manager_attrs={"manager.max_cycles": "3"})
        ctx = PipelineContext()

        result = await handler.execute(graph.nodes["manager"], ctx, graph, "/tmp")

        assert result.context_updates is not None
        assert result.context_updates["last_stage"] == "manager"
        assert result.context_updates["manager.cycles"] == 1


# ---------------------------------------------------------------------------
# Steer action tests
# ---------------------------------------------------------------------------


class TestManagerSteer:
    """'steer' action injects steering context for child retries."""

    @pytest.mark.asyncio
    async def test_steer_injects_context(self):
        """When 'steer' in actions, child context gets steering message."""
        captured_contexts: list[PipelineContext] = []

        async def capturing_runner(
            node_id: str,
            context: PipelineContext,
            graph: Graph,
            logs_root: str,
        ) -> Outcome:
            captured_contexts.append(context)
            if len(captured_contexts) < 2:
                return Outcome(status=StageStatus.FAIL)
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

        # First cycle: no steering (no prior failure)
        assert captured_contexts[0].get("manager.steering") is None
        # Second cycle: steering injected from first failure
        steering = captured_contexts[1].get("manager.steering")
        assert steering is not None
        assert "fail" in steering.lower()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestManagerEdgeCases:
    """Edge cases for the manager loop handler."""

    @pytest.mark.asyncio
    async def test_no_child_edges_returns_fail(self):
        """Manager with no outgoing edges returns FAIL."""
        handler = ManagerLoopHandler(subgraph_runner=AsyncMock())
        graph = _make_graph(has_child_edge=False)
        ctx = PipelineContext()

        result = await handler.execute(graph.nodes["manager"], ctx, graph, "/tmp")

        assert result.status == StageStatus.FAIL
        assert "no child" in (result.failure_reason or "").lower()

    @pytest.mark.asyncio
    async def test_no_runner_returns_fail(self):
        """Manager without a subgraph_runner returns FAIL."""
        handler = ManagerLoopHandler()  # no runner
        graph = _make_graph(manager_attrs={"manager.max_cycles": "3"})
        ctx = PipelineContext()

        result = await handler.execute(graph.nodes["manager"], ctx, graph, "/tmp")

        assert result.status == StageStatus.FAIL

    @pytest.mark.asyncio
    async def test_default_max_cycles_is_1000(self):
        """M-14: Default max_cycles is 1000 per spec when not specified."""
        # Instead of running 1000 cycles, succeed on cycle 11 to prove
        # the default is > 10 (the old wrong default).
        call_count = 0

        async def runner_succeeds_on_11(node_id, context, graph, logs_root):
            nonlocal call_count
            call_count += 1
            if call_count >= 11:
                return Outcome(status=StageStatus.SUCCESS)
            return Outcome(status=StageStatus.FAIL)

        handler = ManagerLoopHandler(subgraph_runner=runner_succeeds_on_11)
        graph = _make_graph(
            manager_attrs={
                "manager.poll_interval": "0s",
                # No max_cycles — should default to 1000
            }
        )
        ctx = PipelineContext()

        result = await handler.execute(graph.nodes["manager"], ctx, graph, "/tmp")

        # With old default of 10, this would FAIL. With 1000, it succeeds.
        assert result.status == StageStatus.SUCCESS
        assert call_count == 11

    @pytest.mark.asyncio
    async def test_default_max_cycles_exhaustion_message(self):
        """M-14: When default max_cycles exhausted, failure mentions 1000."""
        call_count = 0

        async def always_fails(node_id, context, graph, logs_root):
            nonlocal call_count
            call_count += 1
            # Succeed on 1001 (never reached if default is 1000)
            if call_count > 1000:
                return Outcome(status=StageStatus.SUCCESS)
            return Outcome(status=StageStatus.FAIL)

        handler = ManagerLoopHandler(subgraph_runner=always_fails)
        graph = _make_graph(
            manager_attrs={
                "manager.poll_interval": "0s",
            }
        )
        ctx = PipelineContext()

        result = await handler.execute(graph.nodes["manager"], ctx, graph, "/tmp")

        assert result.status == StageStatus.FAIL
        assert "1000" in (result.failure_reason or "")
        assert call_count == 1000

    @pytest.mark.asyncio
    async def test_child_exception_treated_as_fail(self):
        """If the child runner raises, treat it as a FAIL outcome."""
        runner = AsyncMock(
            side_effect=[
                RuntimeError("boom"),
                Outcome(status=StageStatus.SUCCESS),
            ]
        )
        handler = ManagerLoopHandler(subgraph_runner=runner)
        graph = _make_graph(
            manager_attrs={
                "manager.max_cycles": "5",
                "manager.poll_interval": "0s",
            }
        )
        ctx = PipelineContext()

        result = await handler.execute(graph.nodes["manager"], ctx, graph, "/tmp")

        # First call raises, second succeeds — manager should recover
        assert result.status == StageStatus.SUCCESS
        assert runner.call_count == 2

    @pytest.mark.asyncio
    async def test_runner_receives_child_node_id(self):
        """The subgraph_runner is called with the correct child start node ID."""
        runner = _make_runner(
            [
                Outcome(status=StageStatus.SUCCESS),
            ]
        )
        handler = ManagerLoopHandler(subgraph_runner=runner)
        graph = _make_graph(manager_attrs={"manager.max_cycles": "1"})
        ctx = PipelineContext()

        await handler.execute(graph.nodes["manager"], ctx, graph, "/tmp")

        call_args = runner.call_args
        assert call_args[0][0] == "child_task"  # first positional arg


# ---------------------------------------------------------------------------
# Handler registration
# ---------------------------------------------------------------------------


class TestManagerHandlerRegistration:
    """Handler registry resolves house shape to ManagerLoopHandler."""

    def test_registry_resolves_manager_handler(self):
        from amplifier_module_loop_pipeline.handlers import HandlerRegistry

        registry = HandlerRegistry()
        node = Node(id="mgr", shape="house")
        handler = registry.get(node)
        assert isinstance(handler, ManagerLoopHandler)
