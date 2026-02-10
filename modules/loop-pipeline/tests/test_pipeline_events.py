"""Tests for pipeline event emission.

Spec coverage: EVT-001–008, Section 9.6.
"""

from __future__ import annotations

from typing import Any

import pytest

from amplifier_module_loop_pipeline.context import PipelineContext
from amplifier_module_loop_pipeline.dot_parser import parse_dot
from amplifier_module_loop_pipeline.engine import PipelineEngine
from amplifier_module_loop_pipeline.graph import Edge, Graph, Node
from amplifier_module_loop_pipeline.handlers import HandlerRegistry
from amplifier_module_loop_pipeline.outcome import Outcome, StageStatus
from amplifier_module_loop_pipeline.pipeline_events import (
    PIPELINE_CHECKPOINT,
    PIPELINE_COMPLETE,
    PIPELINE_EDGE_SELECTED,
    PIPELINE_ERROR,
    PIPELINE_GOAL_GATE_CHECK,
    PIPELINE_NODE_COMPLETE,
    PIPELINE_NODE_START,
    PIPELINE_START,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

class MockHooks:
    """Captures all emitted events for assertion."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def emit(self, event_name: str, data: dict[str, Any]) -> None:
        self.events.append((event_name, data))

    def names(self) -> list[str]:
        return [name for name, _ in self.events]

    def get(self, event_name: str) -> list[dict[str, Any]]:
        return [data for name, data in self.events if name == event_name]


class MockBackend:
    """Backend that returns a fixed string."""

    def __init__(self, return_value: str = "done") -> None:
        self._return_value = return_value

    async def run(self, node: Node, prompt: str, context: PipelineContext) -> str:
        return self._return_value


class FailingBackend:
    """Backend that returns FAIL for a specific node."""

    def __init__(self, fail_node: str = "bad") -> None:
        self._fail_node = fail_node

    async def run(
        self, node: Node, prompt: str, context: PipelineContext
    ) -> str | Outcome:
        if node.id == self._fail_node:
            return Outcome(status=StageStatus.FAIL, failure_reason="intentional")
        return "ok"


def _make_engine(
    dot_source: str,
    backend: object | None = None,
    logs_root: str = "/tmp/test-events",
    hooks: object | None = None,
) -> PipelineEngine:
    """Parse DOT, validate, and build an engine with optional hooks."""
    from amplifier_module_loop_pipeline.validation import validate_or_raise

    graph = parse_dot(dot_source)
    validate_or_raise(graph)
    context = PipelineContext()
    registry = HandlerRegistry(backend=backend)
    return PipelineEngine(
        graph=graph,
        context=context,
        handler_registry=registry,
        logs_root=logs_root,
        hooks=hooks,
    )


# ---------------------------------------------------------------------------
# Event constant tests
# ---------------------------------------------------------------------------

class TestEventConstants:
    """Event name constants are defined correctly."""

    def test_pipeline_start_constant(self):
        assert PIPELINE_START == "pipeline:start"

    def test_pipeline_complete_constant(self):
        assert PIPELINE_COMPLETE == "pipeline:complete"

    def test_pipeline_node_start_constant(self):
        assert PIPELINE_NODE_START == "pipeline:node_start"

    def test_pipeline_node_complete_constant(self):
        assert PIPELINE_NODE_COMPLETE == "pipeline:node_complete"

    def test_pipeline_edge_selected_constant(self):
        assert PIPELINE_EDGE_SELECTED == "pipeline:edge_selected"

    def test_pipeline_checkpoint_constant(self):
        assert PIPELINE_CHECKPOINT == "pipeline:checkpoint"

    def test_pipeline_goal_gate_check_constant(self):
        assert PIPELINE_GOAL_GATE_CHECK == "pipeline:goal_gate_check"

    def test_pipeline_error_constant(self):
        assert PIPELINE_ERROR == "pipeline:error"


# ---------------------------------------------------------------------------
# Engine emits pipeline:start and pipeline:complete
# ---------------------------------------------------------------------------

class TestPipelineLifecycleEvents:
    """Engine emits start and complete events."""

    @pytest.mark.asyncio
    async def test_emits_pipeline_start(self, tmp_path):
        """pipeline:start is emitted at the beginning of run()."""
        hooks = MockHooks()
        engine = _make_engine(
            dot_source="""
            digraph {
                start [shape=Mdiamond]
                exit [shape=Msquare]
                start -> exit
            }
            """,
            backend=MockBackend(),
            logs_root=str(tmp_path),
            hooks=hooks,
        )
        await engine.run()
        start_events = hooks.get(PIPELINE_START)
        assert len(start_events) == 1
        assert "graph_name" in start_events[0]
        assert "node_count" in start_events[0]
        assert "edge_count" in start_events[0]

    @pytest.mark.asyncio
    async def test_start_event_has_goal(self, tmp_path):
        """pipeline:start includes the goal when set."""
        hooks = MockHooks()
        engine = _make_engine(
            dot_source="""
            digraph {
                goal = "build auth"
                start [shape=Mdiamond]
                exit [shape=Msquare]
                start -> exit
            }
            """,
            backend=MockBackend(),
            logs_root=str(tmp_path),
            hooks=hooks,
        )
        await engine.run()
        start_events = hooks.get(PIPELINE_START)
        assert start_events[0]["goal"] == "build auth"

    @pytest.mark.asyncio
    async def test_emits_pipeline_complete(self, tmp_path):
        """pipeline:complete is emitted when the engine finishes successfully."""
        hooks = MockHooks()
        engine = _make_engine(
            dot_source="""
            digraph {
                start [shape=Mdiamond]
                work [prompt="Do work"]
                exit [shape=Msquare]
                start -> work -> exit
            }
            """,
            backend=MockBackend(),
            logs_root=str(tmp_path),
            hooks=hooks,
        )
        await engine.run()
        complete_events = hooks.get(PIPELINE_COMPLETE)
        assert len(complete_events) == 1
        assert "status" in complete_events[0]
        assert "total_nodes_executed" in complete_events[0]
        assert "duration_ms" in complete_events[0]

    @pytest.mark.asyncio
    async def test_complete_event_counts_nodes(self, tmp_path):
        """pipeline:complete has the correct number of nodes executed."""
        hooks = MockHooks()
        engine = _make_engine(
            dot_source="""
            digraph {
                start [shape=Mdiamond]
                a [prompt="A"]
                b [prompt="B"]
                exit [shape=Msquare]
                start -> a -> b -> exit
            }
            """,
            backend=MockBackend(),
            logs_root=str(tmp_path),
            hooks=hooks,
        )
        await engine.run()
        complete_events = hooks.get(PIPELINE_COMPLETE)
        # start + a + b = 3 nodes executed
        assert complete_events[0]["total_nodes_executed"] == 3

    @pytest.mark.asyncio
    async def test_start_is_first_event(self, tmp_path):
        """pipeline:start is the very first event emitted."""
        hooks = MockHooks()
        engine = _make_engine(
            dot_source="""
            digraph {
                start [shape=Mdiamond]
                exit [shape=Msquare]
                start -> exit
            }
            """,
            backend=MockBackend(),
            logs_root=str(tmp_path),
            hooks=hooks,
        )
        await engine.run()
        assert hooks.names()[0] == PIPELINE_START

    @pytest.mark.asyncio
    async def test_complete_is_last_event(self, tmp_path):
        """pipeline:complete is the very last event emitted."""
        hooks = MockHooks()
        engine = _make_engine(
            dot_source="""
            digraph {
                start [shape=Mdiamond]
                exit [shape=Msquare]
                start -> exit
            }
            """,
            backend=MockBackend(),
            logs_root=str(tmp_path),
            hooks=hooks,
        )
        await engine.run()
        assert hooks.names()[-1] == PIPELINE_COMPLETE


# ---------------------------------------------------------------------------
# Node events
# ---------------------------------------------------------------------------

class TestNodeEvents:
    """Engine emits node_start and node_complete for each node."""

    @pytest.mark.asyncio
    async def test_emits_node_start(self, tmp_path):
        """pipeline:node_start is emitted before each node execution."""
        hooks = MockHooks()
        engine = _make_engine(
            dot_source="""
            digraph {
                start [shape=Mdiamond]
                work [prompt="Do work"]
                exit [shape=Msquare]
                start -> work -> exit
            }
            """,
            backend=MockBackend(),
            logs_root=str(tmp_path),
            hooks=hooks,
        )
        await engine.run()
        node_starts = hooks.get(PIPELINE_NODE_START)
        node_ids = [e["node_id"] for e in node_starts]
        assert "start" in node_ids
        assert "work" in node_ids

    @pytest.mark.asyncio
    async def test_node_start_has_handler_type(self, tmp_path):
        """pipeline:node_start includes handler_type."""
        hooks = MockHooks()
        engine = _make_engine(
            dot_source="""
            digraph {
                start [shape=Mdiamond]
                work [prompt="Do work"]
                exit [shape=Msquare]
                start -> work -> exit
            }
            """,
            backend=MockBackend(),
            logs_root=str(tmp_path),
            hooks=hooks,
        )
        await engine.run()
        node_starts = hooks.get(PIPELINE_NODE_START)
        for event in node_starts:
            assert "handler_type" in event
            assert "attempt" in event

    @pytest.mark.asyncio
    async def test_emits_node_complete(self, tmp_path):
        """pipeline:node_complete is emitted after each node execution."""
        hooks = MockHooks()
        engine = _make_engine(
            dot_source="""
            digraph {
                start [shape=Mdiamond]
                work [prompt="Do work"]
                exit [shape=Msquare]
                start -> work -> exit
            }
            """,
            backend=MockBackend(),
            logs_root=str(tmp_path),
            hooks=hooks,
        )
        await engine.run()
        node_completes = hooks.get(PIPELINE_NODE_COMPLETE)
        node_ids = [e["node_id"] for e in node_completes]
        assert "start" in node_ids
        assert "work" in node_ids

    @pytest.mark.asyncio
    async def test_node_complete_has_status_and_duration(self, tmp_path):
        """pipeline:node_complete includes status and duration_ms."""
        hooks = MockHooks()
        engine = _make_engine(
            dot_source="""
            digraph {
                start [shape=Mdiamond]
                work [prompt="Do work"]
                exit [shape=Msquare]
                start -> work -> exit
            }
            """,
            backend=MockBackend(),
            logs_root=str(tmp_path),
            hooks=hooks,
        )
        await engine.run()
        node_completes = hooks.get(PIPELINE_NODE_COMPLETE)
        for event in node_completes:
            assert "status" in event
            assert "duration_ms" in event
            assert isinstance(event["duration_ms"], (int, float))


# ---------------------------------------------------------------------------
# Edge selection events
# ---------------------------------------------------------------------------

class TestEdgeEvents:
    """Engine emits edge_selected after each edge selection."""

    @pytest.mark.asyncio
    async def test_emits_edge_selected(self, tmp_path):
        """pipeline:edge_selected is emitted after edge selection."""
        hooks = MockHooks()
        engine = _make_engine(
            dot_source="""
            digraph {
                start [shape=Mdiamond]
                work [prompt="Do work"]
                exit [shape=Msquare]
                start -> work -> exit
            }
            """,
            backend=MockBackend(),
            logs_root=str(tmp_path),
            hooks=hooks,
        )
        await engine.run()
        edge_events = hooks.get(PIPELINE_EDGE_SELECTED)
        assert len(edge_events) >= 1
        for event in edge_events:
            assert "from_node" in event
            assert "to_node" in event


# ---------------------------------------------------------------------------
# Checkpoint events
# ---------------------------------------------------------------------------

class TestCheckpointEvents:
    """Engine emits checkpoint events after saving."""

    @pytest.mark.asyncio
    async def test_emits_checkpoint(self, tmp_path):
        """pipeline:checkpoint is emitted after each checkpoint save."""
        hooks = MockHooks()
        engine = _make_engine(
            dot_source="""
            digraph {
                start [shape=Mdiamond]
                work [prompt="Do work"]
                exit [shape=Msquare]
                start -> work -> exit
            }
            """,
            backend=MockBackend(),
            logs_root=str(tmp_path),
            hooks=hooks,
        )
        await engine.run()
        cp_events = hooks.get(PIPELINE_CHECKPOINT)
        assert len(cp_events) >= 1
        for event in cp_events:
            assert "node_id" in event
            assert "checkpoint_path" in event


# ---------------------------------------------------------------------------
# Goal gate events
# ---------------------------------------------------------------------------

class TestGoalGateEvents:
    """Engine emits goal_gate_check at exit."""

    @pytest.mark.asyncio
    async def test_emits_goal_gate_check(self, tmp_path):
        """pipeline:goal_gate_check is emitted when checking gates."""
        hooks = MockHooks()
        engine = _make_engine(
            dot_source="""
            digraph {
                start [shape=Mdiamond]
                exit [shape=Msquare]
                start -> exit
            }
            """,
            backend=MockBackend(),
            logs_root=str(tmp_path),
            hooks=hooks,
        )
        await engine.run()
        gate_events = hooks.get(PIPELINE_GOAL_GATE_CHECK)
        assert len(gate_events) >= 1
        for event in gate_events:
            assert "satisfied" in event
            assert "unsatisfied" in event


# ---------------------------------------------------------------------------
# Error events
# ---------------------------------------------------------------------------

class TestErrorEvents:
    """Engine emits error events on failures."""

    @pytest.mark.asyncio
    async def test_emits_error_on_no_edge(self, tmp_path):
        """pipeline:error is emitted when no matching edge exists."""
        hooks = MockHooks()
        # Build a graph with a dead end manually
        graph = Graph(
            name="test",
            nodes={
                "start": Node(id="start", shape="Mdiamond"),
                "dead_end": Node(id="dead_end", prompt="work"),
                "exit": Node(id="exit", shape="Msquare"),
            },
            edges=[
                Edge(from_node="start", to_node="dead_end"),
                # dead_end has NO outgoing edges
            ],
        )
        context = PipelineContext()
        registry = HandlerRegistry(backend=MockBackend("ok"))
        engine = PipelineEngine(
            graph=graph,
            context=context,
            handler_registry=registry,
            logs_root=str(tmp_path),
            hooks=hooks,
        )
        outcome = await engine.run()
        assert outcome.status == StageStatus.FAIL
        error_events = hooks.get(PIPELINE_ERROR)
        assert len(error_events) >= 1
        assert "node_id" in error_events[0]
        assert "error_type" in error_events[0]
        assert "message" in error_events[0]


# ---------------------------------------------------------------------------
# No hooks (backward compatibility)
# ---------------------------------------------------------------------------

class TestNoHooksBackwardCompat:
    """Engine works fine without hooks (existing tests pass)."""

    @pytest.mark.asyncio
    async def test_engine_works_without_hooks(self, tmp_path):
        """Engine runs successfully with hooks=None (default)."""
        graph = parse_dot("""
        digraph {
            start [shape=Mdiamond]
            work [prompt="Do work"]
            exit [shape=Msquare]
            start -> work -> exit
        }
        """)
        from amplifier_module_loop_pipeline.validation import validate_or_raise

        validate_or_raise(graph)
        context = PipelineContext()
        registry = HandlerRegistry(backend=MockBackend())
        engine = PipelineEngine(
            graph=graph,
            context=context,
            handler_registry=registry,
            logs_root=str(tmp_path),
        )
        outcome = await engine.run()
        assert outcome.status in (StageStatus.SUCCESS, StageStatus.PARTIAL_SUCCESS)
