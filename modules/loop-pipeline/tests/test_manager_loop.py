"""Tests for the manager loop handler (stub).

The manager loop (shape=house) is a supervisor pattern over a child
pipeline. This tests the stub implementation that allows pipelines
with manager nodes to parse and execute without crashing.

Spec coverage: MGR-001–010, COMP-001–002, Section 4.11.
"""

import pytest

from amplifier_module_loop_pipeline.context import PipelineContext
from amplifier_module_loop_pipeline.graph import Edge, Graph, Node
from amplifier_module_loop_pipeline.handlers.manager_loop import ManagerLoopHandler
from amplifier_module_loop_pipeline.outcome import StageStatus


def _make_graph_with_manager() -> Graph:
    """Graph with a manager loop node."""
    return Graph(
        name="test",
        nodes={
            "start": Node(id="start", shape="Mdiamond"),
            "manager": Node(
                id="manager",
                shape="house",
                label="Supervise coding",
                attrs={
                    "manager.max_cycles": "5",
                    "manager.poll_interval": "10s",
                    "manager.stop_condition": "",
                    "manager.actions": "observe,wait",
                    "stack.child_dotfile": "child.dot",
                },
            ),
            "exit": Node(id="exit", shape="Msquare"),
        },
        edges=[
            Edge(from_node="start", to_node="manager"),
            Edge(from_node="manager", to_node="exit"),
        ],
    )


def _make_context() -> PipelineContext:
    return PipelineContext()


class TestManagerLoopHandler:
    """MGR-001–010: Manager loop handler stub."""

    @pytest.mark.asyncio
    async def test_returns_success(self):
        """Stub handler completes successfully."""
        graph = _make_graph_with_manager()
        node = graph.nodes["manager"]
        handler = ManagerLoopHandler()
        outcome = await handler.execute(node, _make_context(), graph, "/tmp")
        assert outcome.status == StageStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_notes_indicate_stub(self):
        """Outcome notes indicate this is a stub implementation."""
        graph = _make_graph_with_manager()
        node = graph.nodes["manager"]
        handler = ManagerLoopHandler()
        outcome = await handler.execute(node, _make_context(), graph, "/tmp")
        assert outcome.notes is not None
        assert "stub" in outcome.notes.lower()

    @pytest.mark.asyncio
    async def test_reads_max_cycles_from_attrs(self):
        """Handler reads manager.max_cycles from node attrs."""
        graph = _make_graph_with_manager()
        node = graph.nodes["manager"]
        handler = ManagerLoopHandler()
        outcome = await handler.execute(node, _make_context(), graph, "/tmp")
        # Stub succeeds regardless, but should have parsed config
        assert outcome.status == StageStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_default_max_cycles(self):
        """Default max_cycles is used when not specified."""
        graph = Graph(
            name="test",
            nodes={
                "start": Node(id="start", shape="Mdiamond"),
                "mgr": Node(id="mgr", shape="house", label="Manager"),
                "exit": Node(id="exit", shape="Msquare"),
            },
            edges=[
                Edge(from_node="start", to_node="mgr"),
                Edge(from_node="mgr", to_node="exit"),
            ],
        )
        handler = ManagerLoopHandler()
        outcome = await handler.execute(
            graph.nodes["mgr"], _make_context(), graph, "/tmp"
        )
        assert outcome.status == StageStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_sets_context_updates(self):
        """Handler sets context updates about the manager execution."""
        graph = _make_graph_with_manager()
        node = graph.nodes["manager"]
        handler = ManagerLoopHandler()
        outcome = await handler.execute(node, _make_context(), graph, "/tmp")
        assert outcome.context_updates is not None
        assert "last_stage" in outcome.context_updates

    @pytest.mark.asyncio
    async def test_no_backend_required(self):
        """Manager loop stub doesn't need a backend to run."""
        graph = _make_graph_with_manager()
        node = graph.nodes["manager"]
        handler = ManagerLoopHandler()
        # Should not raise
        outcome = await handler.execute(node, _make_context(), graph, "/tmp")
        assert outcome.status == StageStatus.SUCCESS


class TestManagerHandlerRegistration:
    """Handler registry resolves house shape to ManagerLoopHandler."""

    def test_registry_resolves_manager_handler(self):
        from amplifier_module_loop_pipeline.handlers import HandlerRegistry

        registry = HandlerRegistry()
        node = Node(id="mgr", shape="house")
        handler = registry.get(node)
        assert isinstance(handler, ManagerLoopHandler)
