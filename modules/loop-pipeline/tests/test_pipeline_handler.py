"""Tests for pipeline handler — DOT file path resolution and PipelineHandler.execute().

Spec coverage: resolve_dot_path, _expand_path_variables, PipelineHandler.execute().
"""

import json
import os
from unittest.mock import AsyncMock

import pytest

from amplifier_module_loop_pipeline.context import PipelineContext
from amplifier_module_loop_pipeline.graph import Edge, Graph, Node
from amplifier_module_loop_pipeline.handlers.pipeline import (
    PipelineHandler,
    resolve_dot_path,
)
from amplifier_module_loop_pipeline.outcome import StageStatus


class TestResolveDotPath:
    """Tests for resolve_dot_path() and _expand_path_variables()."""

    def test_absolute_path_unchanged(self) -> None:
        """Absolute paths are returned as-is."""
        ctx = PipelineContext()
        result = resolve_dot_path("/abs/child.dot", source_dir="/parent", context=ctx)
        assert result == "/abs/child.dot"

    def test_relative_to_source_dir(self) -> None:
        """Relative paths resolve against source_dir."""
        ctx = PipelineContext()
        result = resolve_dot_path(
            "child.dot", source_dir="/parent/pipelines", context=ctx
        )
        assert result == "/parent/pipelines/child.dot"

    def test_relative_subdirectory(self) -> None:
        """Subdirectory paths resolve correctly."""
        ctx = PipelineContext()
        result = resolve_dot_path("sub/child.dot", source_dir="/parent", context=ctx)
        assert result == "/parent/sub/child.dot"

    def test_variable_expansion(self) -> None:
        """$language token is expanded from context."""
        ctx = PipelineContext()
        ctx.set("language", "python")
        result = resolve_dot_path(
            "$language/tasks.dot", source_dir="/pipelines", context=ctx
        )
        assert result == "/pipelines/python/tasks.dot"

    def test_variable_expansion_then_absolute(self) -> None:
        """If expansion produces an absolute path, use it."""
        ctx = PipelineContext()
        ctx.set("base", "/absolute/root")
        result = resolve_dot_path("$base/child.dot", source_dir="/parent", context=ctx)
        assert result == "/absolute/root/child.dot"

    def test_empty_source_dir_uses_cwd(self) -> None:
        """Empty source_dir falls back to os.getcwd()."""
        ctx = PipelineContext()
        result = resolve_dot_path("child.dot", source_dir="", context=ctx)
        assert result == os.path.join(os.getcwd(), "child.dot")

    def test_no_variable_in_path(self) -> None:
        """Paths without $ are not modified by context."""
        ctx = PipelineContext()
        ctx.set("language", "python")
        result = resolve_dot_path("plain/child.dot", source_dir="/parent", context=ctx)
        assert result == "/parent/plain/child.dot"

    def test_unknown_variable_left_unchanged(self) -> None:
        """Unknown $tokens survive expansion unchanged."""
        ctx = PipelineContext()
        ctx.set("language", "python")
        result = resolve_dot_path(
            "$unknown/$language/child.dot", source_dir="/parent", context=ctx
        )
        assert result == "/parent/$unknown/python/child.dot"


# ---------------------------------------------------------------------------
# PipelineHandler.execute() tests
# ---------------------------------------------------------------------------

CHILD_DOT = """\
digraph child {
    start [shape=Mdiamond]
    work [prompt="Do child work"]
    done [shape=Msquare]
    start -> work -> done
}
"""


def _write_child_dot(tmp_path):
    """Write CHILD_DOT to a file and return the path."""
    dot_file = tmp_path / "child.dot"
    dot_file.write_text(CHILD_DOT)
    return str(dot_file)


def _make_parent_graph(tmp_path):
    """Build a minimal parent graph with a pipeline node pointing to child.dot."""
    dot_path = _write_child_dot(tmp_path)
    return Graph(
        name="parent",
        nodes={
            "start": Node(id="start", shape="Mdiamond"),
            "sub": Node(
                id="sub",
                shape="component",
                type="pipeline",
                attrs={"dot_file": str(dot_path)},
            ),
            "done": Node(id="done", shape="Msquare"),
        },
        edges=[
            Edge(from_node="start", to_node="sub"),
            Edge(from_node="sub", to_node="done"),
        ],
        source_dir=str(tmp_path),
    )


class TestPipelineHandlerExecute:
    """Tests for PipelineHandler.execute()."""

    @pytest.mark.asyncio
    async def test_executes_child_pipeline_and_returns_success(self, tmp_path):
        """Valid child DOT executes and returns SUCCESS."""
        graph = _make_parent_graph(tmp_path)
        node = graph.nodes["sub"]
        context = PipelineContext()
        logs_root = str(tmp_path / "logs")

        handler = PipelineHandler()
        outcome = await handler.execute(node, context, graph, logs_root)

        assert outcome.status == StageStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_missing_dot_file_returns_fail(self, tmp_path):
        """Missing DOT file on disk returns FAIL with 'not found'."""
        graph = _make_parent_graph(tmp_path)
        node = graph.nodes["sub"]
        # Point to a non-existent file
        node.attrs["dot_file"] = str(tmp_path / "nonexistent.dot")
        context = PipelineContext()
        logs_root = str(tmp_path / "logs")

        handler = PipelineHandler()
        outcome = await handler.execute(node, context, graph, logs_root)

        assert outcome.status == StageStatus.FAIL
        assert outcome.failure_reason is not None
        assert "not found" in outcome.failure_reason.lower()

    @pytest.mark.asyncio
    async def test_invalid_dot_source_returns_fail(self, tmp_path):
        """Invalid DOT syntax returns FAIL with 'parse' in failure_reason."""
        bad_dot = tmp_path / "bad.dot"
        bad_dot.write_text("this is not valid DOT syntax at all!!!")
        graph = _make_parent_graph(tmp_path)
        node = graph.nodes["sub"]
        node.attrs["dot_file"] = str(bad_dot)
        context = PipelineContext()
        logs_root = str(tmp_path / "logs")

        handler = PipelineHandler()
        outcome = await handler.execute(node, context, graph, logs_root)

        assert outcome.status == StageStatus.FAIL
        assert outcome.failure_reason is not None
        assert "parse" in outcome.failure_reason.lower()

    @pytest.mark.asyncio
    async def test_missing_dot_file_attr_returns_fail(self, tmp_path):
        """Missing dot_file attribute returns FAIL with 'dot_file'."""
        graph = _make_parent_graph(tmp_path)
        node = graph.nodes["sub"]
        # Remove the dot_file attribute
        if "dot_file" in node.attrs:
            del node.attrs["dot_file"]
        context = PipelineContext()
        logs_root = str(tmp_path / "logs")

        handler = PipelineHandler()
        outcome = await handler.execute(node, context, graph, logs_root)

        assert outcome.status == StageStatus.FAIL
        assert outcome.failure_reason is not None
        assert "dot_file" in outcome.failure_reason.lower()

    @pytest.mark.asyncio
    async def test_child_logs_written_to_subdirectory(self, tmp_path):
        """Child logs are written to {logs_root}/subgraph_{node_id}/."""
        graph = _make_parent_graph(tmp_path)
        node = graph.nodes["sub"]
        context = PipelineContext()
        logs_root = str(tmp_path / "logs")

        handler = PipelineHandler()
        await handler.execute(node, context, graph, logs_root)

        manifest_path = os.path.join(logs_root, "subgraph_sub", "manifest.json")
        assert os.path.exists(manifest_path)
        with open(manifest_path) as f:
            manifest = json.load(f)
        assert "graph_name" in manifest

    @pytest.mark.asyncio
    async def test_child_context_is_cloned(self, tmp_path):
        """Parent context is not polluted by child execution."""
        graph = _make_parent_graph(tmp_path)
        node = graph.nodes["sub"]
        context = PipelineContext()
        context.set("parent_key", "parent_value")
        logs_root = str(tmp_path / "logs")

        handler = PipelineHandler()
        await handler.execute(node, context, graph, logs_root)

        # Parent context should still have parent_key
        assert context.get("parent_key") == "parent_value"
        # Parent context should NOT have child engine's internal keys
        # (like graph.goal which is set by engine._initialize_context)
        assert context.get("outcome") is None


# ---------------------------------------------------------------------------
# PipelineHandler observability tests
# ---------------------------------------------------------------------------


class TestPipelineHandlerObservability:
    """Tests for subgraph observability — _subgraph_runs capture and event emission."""

    @pytest.mark.asyncio
    async def test_populates_subgraph_runs(self, tmp_path):
        """After execution, handler._subgraph_runs['sub'] contains all expected keys."""
        graph = _make_parent_graph(tmp_path)
        node = graph.nodes["sub"]
        context = PipelineContext()
        logs_root = str(tmp_path / "logs")

        handler = PipelineHandler()
        await handler.execute(node, context, graph, logs_root)

        assert "sub" in handler._subgraph_runs
        run = handler._subgraph_runs["sub"]
        expected_keys = {
            "dot_file",
            "dot_source",
            "pipeline_id",
            "goal",
            "status",
            "execution_path",
            "node_outcomes",
            "total_elapsed_ms",
            "nodes_completed",
            "nodes_total",
        }
        assert expected_keys.issubset(run.keys())
        assert run["status"] == "success"
        assert isinstance(run["total_elapsed_ms"], float)
        assert isinstance(run["nodes_completed"], int)
        assert isinstance(run["nodes_total"], int)
        assert run["nodes_completed"] > 0

    @pytest.mark.asyncio
    async def test_emits_subgraph_start_event(self, tmp_path):
        """hooks.emit is called with 'pipeline:subgraph_start' including node_id."""
        hooks = AsyncMock()
        graph = _make_parent_graph(tmp_path)
        node = graph.nodes["sub"]
        context = PipelineContext()
        logs_root = str(tmp_path / "logs")

        handler = PipelineHandler(hooks=hooks)
        await handler.execute(node, context, graph, logs_root)

        # Find the pipeline:subgraph_start call
        start_calls = [
            c for c in hooks.emit.call_args_list if c[0][0] == "pipeline:subgraph_start"
        ]
        assert len(start_calls) == 1
        data = start_calls[0][0][1]
        assert data["node_id"] == "sub"
        assert "dot_file" in data
        assert "pipeline_id" in data
        assert "goal" in data

    @pytest.mark.asyncio
    async def test_emits_subgraph_complete_event(self, tmp_path):
        """hooks.emit is called with 'pipeline:subgraph_complete' including node_id, status, duration_ms."""
        hooks = AsyncMock()
        graph = _make_parent_graph(tmp_path)
        node = graph.nodes["sub"]
        context = PipelineContext()
        logs_root = str(tmp_path / "logs")

        handler = PipelineHandler(hooks=hooks)
        await handler.execute(node, context, graph, logs_root)

        # Find the pipeline:subgraph_complete call
        complete_calls = [
            c
            for c in hooks.emit.call_args_list
            if c[0][0] == "pipeline:subgraph_complete"
        ]
        assert len(complete_calls) == 1
        data = complete_calls[0][0][1]
        assert data["node_id"] == "sub"
        assert data["status"] == "success"
        assert "duration_ms" in data
        assert isinstance(data["duration_ms"], float)
        assert "pipeline_id" in data
        assert "nodes_completed" in data
        assert "nodes_total" in data
