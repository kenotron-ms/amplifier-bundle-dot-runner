"""Tests for Phase 3: health-check.dot and fix-iteration.dot -- Systematic Fix Loop.

Verifies that:
- examples/dev-machine/runtime/health-check.dot is correct per spec:
  - 5 nodes with correct shapes (start, initial_check, clean_gate, fix_loop, done)
  - 5 edges
  - initial_check is parallelogram with parse_json='true'
  - clean_gate is diamond
  - fix_loop is house with manager.max_cycles, manager.stop_condition='outcome=success'
  - fix_loop has manager.poll_interval='0s'
  - fix_loop has stack.child_dotfile referencing fix-iteration.dot
  - clean_gate routes clean->done and failed->fix_loop

- examples/dev-machine/runtime/fix-iteration.dot is correct per spec:
  - 5 nodes with correct shapes (start, read_errors, fix_session, verify, done)
  - 4 edges (linear chain)
  - read_errors is parallelogram with parse_json='true'
  - fix_session is box with context_fidelity='truncate' and VERBATIM prompt
  - verify is parallelogram with parse_json='true'
  - Chain order: start->read_errors->fix_session->verify->done
  - fix_session prompt contains required safety/strategy phrases

Spec coverage: health-check.dot + fix-iteration.dot Phase 3 requirements.
"""

from __future__ import annotations

import os

import pytest

from amplifier_module_loop_pipeline.dot_parser import parse_dot

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_TESTS_DIR = os.path.dirname(__file__)
# From modules/loop-pipeline/tests/ -> up 3 levels -> amplifier-bundle-attractor/ -> examples/
_EXAMPLES_DIR = os.path.abspath(os.path.join(_TESTS_DIR, "..", "..", "..", "examples"))
_HEALTH_CHECK_DOT = os.path.join(
    _EXAMPLES_DIR, "dev-machine", "runtime", "health-check.dot"
)
_FIX_ITERATION_DOT = os.path.join(
    _EXAMPLES_DIR, "dev-machine", "runtime", "fix-iteration.dot"
)


def _load_health_check() -> str:
    with open(_HEALTH_CHECK_DOT) as f:
        return f.read()


def _graph_health_check():
    return parse_dot(_load_health_check())


def _load_fix_iteration() -> str:
    with open(_FIX_ITERATION_DOT) as f:
        return f.read()


def _graph_fix_iteration():
    return parse_dot(_load_fix_iteration())


# ===========================================================================
# TestHealthCheckParse -- health-check.dot structural tests
# ===========================================================================


class TestHealthCheckParse:
    """Tests for health-check.dot parse correctness and structural requirements."""

    # -----------------------------------------------------------------------
    # AC-1: File exists
    # -----------------------------------------------------------------------

    def test_file_exists(self):
        """health-check.dot exists at examples/dev-machine/runtime/health-check.dot."""
        assert os.path.isfile(_HEALTH_CHECK_DOT), (
            f"health-check.dot not found at {_HEALTH_CHECK_DOT}"
        )

    # -----------------------------------------------------------------------
    # AC-2: Parses without error
    # -----------------------------------------------------------------------

    def test_parses_without_error(self):
        """health-check.dot parses without raising exceptions."""
        graph = _graph_health_check()
        assert graph is not None

    # -----------------------------------------------------------------------
    # AC-3: Exactly 5 nodes
    # -----------------------------------------------------------------------

    def test_node_count(self):
        """Exactly 5 nodes."""
        graph = _graph_health_check()
        assert len(graph.nodes) == 5, (
            f"Expected 5 nodes, got {len(graph.nodes)}: {list(graph.nodes.keys())}"
        )

    # -----------------------------------------------------------------------
    # AC-4: All 5 required node IDs present
    # -----------------------------------------------------------------------

    REQUIRED_NODE_IDS = [
        "start",
        "initial_check",
        "clean_gate",
        "fix_loop",
        "done",
    ]

    def test_all_required_node_ids_present(self):
        """All 5 required node IDs are present.

        Intentionally overlaps with test_each_required_node_id[*] below:
        this bulk test catches the complete set in one assertion, while the
        parametrized variant gives per-node visibility in CI output so a
        failure names exactly which node is missing.
        """
        graph = _graph_health_check()
        missing = [nid for nid in self.REQUIRED_NODE_IDS if nid not in graph.nodes]
        assert not missing, (
            f"Missing node IDs: {missing}. Present: {list(graph.nodes.keys())}"
        )

    @pytest.mark.parametrize("node_id", REQUIRED_NODE_IDS)
    def test_each_required_node_id(self, node_id):
        """Each required node ID is individually present (per-node CI visibility)."""
        graph = _graph_health_check()
        assert node_id in graph.nodes, (
            f"Node '{node_id}' not found. Present: {list(graph.nodes.keys())}"
        )

    # -----------------------------------------------------------------------
    # AC-5: start is Mdiamond, done is Msquare
    # -----------------------------------------------------------------------

    def test_start_shape(self):
        """start node has shape=Mdiamond."""
        graph = _graph_health_check()
        assert graph.nodes["start"].shape == "Mdiamond", (
            f"Expected start shape=Mdiamond, got {graph.nodes['start'].shape!r}"
        )

    def test_done_shape(self):
        """done node has shape=Msquare."""
        graph = _graph_health_check()
        assert graph.nodes["done"].shape == "Msquare", (
            f"Expected done shape=Msquare, got {graph.nodes['done'].shape!r}"
        )

    # -----------------------------------------------------------------------
    # AC-6: initial_check is parallelogram with parse_json='true'
    # -----------------------------------------------------------------------

    def test_initial_check_shape(self):
        """initial_check node has shape=parallelogram."""
        graph = _graph_health_check()
        assert graph.nodes["initial_check"].shape == "parallelogram", (
            f"Expected initial_check shape=parallelogram, "
            f"got {graph.nodes['initial_check'].shape!r}"
        )

    def test_initial_check_parse_json(self):
        """initial_check node has parse_json='true'."""
        graph = _graph_health_check()
        val = graph.nodes["initial_check"].attrs.get("parse_json")
        assert val == "true", f"Expected initial_check parse_json='true', got {val!r}"

    # -----------------------------------------------------------------------
    # AC-7: clean_gate is diamond
    # -----------------------------------------------------------------------

    def test_clean_gate_shape(self):
        """clean_gate node has shape=diamond."""
        graph = _graph_health_check()
        assert graph.nodes["clean_gate"].shape == "diamond", (
            f"Expected clean_gate shape=diamond, got {graph.nodes['clean_gate'].shape!r}"
        )

    # -----------------------------------------------------------------------
    # AC-8: fix_loop is house with manager attributes
    # -----------------------------------------------------------------------

    def test_fix_loop_shape(self):
        """fix_loop node has shape=house."""
        graph = _graph_health_check()
        assert graph.nodes["fix_loop"].shape == "house", (
            f"Expected fix_loop shape=house, got {graph.nodes['fix_loop'].shape!r}"
        )

    def test_fix_loop_max_cycles(self):
        """fix_loop node has manager.max_cycles attribute."""
        graph = _graph_health_check()
        val = graph.nodes["fix_loop"].attrs.get("manager.max_cycles")
        assert val is not None, "Expected fix_loop to have manager.max_cycles attribute"

    def test_fix_loop_stop_condition(self):
        """fix_loop node has manager.stop_condition='outcome=success'."""
        graph = _graph_health_check()
        val = graph.nodes["fix_loop"].attrs.get("manager.stop_condition")
        assert val == "outcome=success", (
            f"Expected fix_loop manager.stop_condition='outcome=success', got {val!r}"
        )

    def test_fix_loop_poll_interval(self):
        """fix_loop node has manager.poll_interval='0s' (parsed to 0ms by DOT parser)."""
        graph = _graph_health_check()
        val = graph.nodes["fix_loop"].attrs.get("manager.poll_interval")
        # DOT parser converts duration strings to milliseconds: '0s' -> 0
        assert val == 0, (
            f"Expected fix_loop manager.poll_interval=0 (parsed from '0s'), got {val!r}"
        )

    def test_fix_loop_child_dotfile(self):
        """fix_loop node has stack.child_dotfile referencing fix-iteration.dot."""
        graph = _graph_health_check()
        val = graph.nodes["fix_loop"].attrs.get("stack.child_dotfile")
        assert val is not None, (
            "Expected fix_loop to have stack.child_dotfile attribute"
        )
        assert "fix-iteration.dot" in val, (
            f"Expected fix_loop stack.child_dotfile to reference fix-iteration.dot, "
            f"got {val!r}"
        )

    # -----------------------------------------------------------------------
    # AC-9: Exactly 5 edges
    # -----------------------------------------------------------------------

    def test_edge_count(self):
        """Exactly 5 edges."""
        graph = _graph_health_check()
        assert len(graph.edges) == 5, (
            f"Expected 5 edges, got {len(graph.edges)}: "
            + "\n".join(f"  {e.from_node} -> {e.to_node}" for e in graph.edges)
        )

    # -----------------------------------------------------------------------
    # AC-10: clean_gate routes clean->done and failed->fix_loop
    # -----------------------------------------------------------------------

    def test_clean_gate_clean_to_done(self):
        """clean_gate has a 'clean' edge routing to done."""
        graph = _graph_health_check()
        gate_edges = [e for e in graph.edges if e.from_node == "clean_gate"]
        done_edges = [e for e in gate_edges if e.to_node == "done"]
        assert done_edges, "Expected clean_gate -> done edge. Edges: " + str(
            [(e.to_node, e.label, e.condition) for e in gate_edges]
        )
        done_edge = done_edges[0]
        has_clean = (done_edge.label and "clean" in done_edge.label.lower()) or (
            done_edge.condition and "clean" in done_edge.condition.lower()
        )
        assert has_clean, (
            f"Expected clean_gate -> done edge to have 'clean' label/condition, "
            f"got label={done_edge.label!r}, condition={done_edge.condition!r}"
        )

    def test_clean_gate_failed_to_fix_loop(self):
        """clean_gate has a 'failed' edge routing to fix_loop."""
        graph = _graph_health_check()
        gate_edges = [e for e in graph.edges if e.from_node == "clean_gate"]
        fix_edges = [e for e in gate_edges if e.to_node == "fix_loop"]
        assert fix_edges, "Expected clean_gate -> fix_loop edge. Edges: " + str(
            [(e.to_node, e.label, e.condition) for e in gate_edges]
        )
        fix_edge = fix_edges[0]
        has_failed = (fix_edge.label and "failed" in fix_edge.label.lower()) or (
            fix_edge.condition and "failed" in fix_edge.condition.lower()
        )
        assert has_failed, (
            f"Expected clean_gate -> fix_loop edge to have 'failed' label/condition, "
            f"got label={fix_edge.label!r}, condition={fix_edge.condition!r}"
        )

    # -----------------------------------------------------------------------
    # AC-11: Required edges exist
    # -----------------------------------------------------------------------

    REQUIRED_EDGES = [
        ("start", "initial_check"),
        ("initial_check", "clean_gate"),
        ("fix_loop", "done"),
    ]

    @pytest.mark.parametrize("from_node,to_node", REQUIRED_EDGES)
    def test_required_edges_present(self, from_node, to_node):
        """Required edges are present."""
        graph = _graph_health_check()
        edge_pairs = {(e.from_node, e.to_node) for e in graph.edges}
        assert (from_node, to_node) in edge_pairs, (
            f"Edge {from_node} -> {to_node} not found. "
            f"Present edges: {sorted(edge_pairs)}"
        )


# ===========================================================================
# TestFixIterationParse -- fix-iteration.dot structural tests
# ===========================================================================


class TestFixIterationParse:
    """Tests for fix-iteration.dot parse correctness and structural requirements."""

    # -----------------------------------------------------------------------
    # AC-1: File exists
    # -----------------------------------------------------------------------

    def test_file_exists(self):
        """fix-iteration.dot exists at examples/dev-machine/runtime/fix-iteration.dot."""
        assert os.path.isfile(_FIX_ITERATION_DOT), (
            f"fix-iteration.dot not found at {_FIX_ITERATION_DOT}"
        )

    # -----------------------------------------------------------------------
    # AC-2: Parses without error
    # -----------------------------------------------------------------------

    def test_parses_without_error(self):
        """fix-iteration.dot parses without raising exceptions."""
        graph = _graph_fix_iteration()
        assert graph is not None

    # -----------------------------------------------------------------------
    # AC-3: Exactly 5 nodes
    # -----------------------------------------------------------------------

    def test_node_count(self):
        """Exactly 5 nodes."""
        graph = _graph_fix_iteration()
        assert len(graph.nodes) == 5, (
            f"Expected 5 nodes, got {len(graph.nodes)}: {list(graph.nodes.keys())}"
        )

    # -----------------------------------------------------------------------
    # AC-4: All 5 required node IDs present
    # -----------------------------------------------------------------------

    REQUIRED_NODE_IDS = [
        "start",
        "read_errors",
        "fix_session",
        "verify",
        "done",
    ]

    def test_all_required_node_ids_present(self):
        """All 5 required node IDs are present.

        Intentionally overlaps with test_each_required_node_id[*] below:
        this bulk test catches the complete set in one assertion, while the
        parametrized variant gives per-node visibility in CI output so a
        failure names exactly which node is missing.
        """
        graph = _graph_fix_iteration()
        missing = [nid for nid in self.REQUIRED_NODE_IDS if nid not in graph.nodes]
        assert not missing, (
            f"Missing node IDs: {missing}. Present: {list(graph.nodes.keys())}"
        )

    @pytest.mark.parametrize("node_id", REQUIRED_NODE_IDS)
    def test_each_required_node_id(self, node_id):
        """Each required node ID is individually present (per-node CI visibility)."""
        graph = _graph_fix_iteration()
        assert node_id in graph.nodes, (
            f"Node '{node_id}' not found. Present: {list(graph.nodes.keys())}"
        )

    # -----------------------------------------------------------------------
    # AC-5: start is Mdiamond, done is Msquare
    # -----------------------------------------------------------------------

    def test_start_shape(self):
        """start node has shape=Mdiamond."""
        graph = _graph_fix_iteration()
        assert graph.nodes["start"].shape == "Mdiamond", (
            f"Expected start shape=Mdiamond, got {graph.nodes['start'].shape!r}"
        )

    def test_done_shape(self):
        """done node has shape=Msquare."""
        graph = _graph_fix_iteration()
        assert graph.nodes["done"].shape == "Msquare", (
            f"Expected done shape=Msquare, got {graph.nodes['done'].shape!r}"
        )

    # -----------------------------------------------------------------------
    # AC-6: read_errors is parallelogram with parse_json='true'
    # -----------------------------------------------------------------------

    def test_read_errors_shape(self):
        """read_errors node has shape=parallelogram."""
        graph = _graph_fix_iteration()
        assert graph.nodes["read_errors"].shape == "parallelogram", (
            f"Expected read_errors shape=parallelogram, "
            f"got {graph.nodes['read_errors'].shape!r}"
        )

    def test_read_errors_parse_json(self):
        """read_errors node has parse_json='true'."""
        graph = _graph_fix_iteration()
        val = graph.nodes["read_errors"].attrs.get("parse_json")
        assert val == "true", f"Expected read_errors parse_json='true', got {val!r}"

    def test_read_errors_continue_on_fail(self):
        """read_errors node has continue_on_fail='true'."""
        graph = _graph_fix_iteration()
        val = graph.nodes["read_errors"].attrs.get("continue_on_fail")
        assert val == "true", (
            f"Expected read_errors continue_on_fail='true', got {val!r}"
        )

    # -----------------------------------------------------------------------
    # AC-7: fix_session is box with context_fidelity='truncate'
    # -----------------------------------------------------------------------

    def test_fix_session_shape(self):
        """fix_session node has shape=box."""
        graph = _graph_fix_iteration()
        assert graph.nodes["fix_session"].shape == "box", (
            f"Expected fix_session shape=box, got {graph.nodes['fix_session'].shape!r}"
        )

    def test_fix_session_context_fidelity(self):
        """fix_session node has context_fidelity='truncate'."""
        graph = _graph_fix_iteration()
        val = graph.nodes["fix_session"].attrs.get("context_fidelity")
        assert val == "truncate", (
            f"Expected fix_session context_fidelity='truncate', got {val!r}"
        )

    # -----------------------------------------------------------------------
    # AC-8: verify is parallelogram with parse_json='true'
    # -----------------------------------------------------------------------

    def test_verify_shape(self):
        """verify node has shape=parallelogram."""
        graph = _graph_fix_iteration()
        assert graph.nodes["verify"].shape == "parallelogram", (
            f"Expected verify shape=parallelogram, got {graph.nodes['verify'].shape!r}"
        )

    def test_verify_parse_json(self):
        """verify node has parse_json='true'."""
        graph = _graph_fix_iteration()
        val = graph.nodes["verify"].attrs.get("parse_json")
        assert val == "true", f"Expected verify parse_json='true', got {val!r}"

    def test_verify_continue_on_fail(self):
        """verify node has continue_on_fail='true'."""
        graph = _graph_fix_iteration()
        val = graph.nodes["verify"].attrs.get("continue_on_fail")
        assert val == "true", f"Expected verify continue_on_fail='true', got {val!r}"

    # -----------------------------------------------------------------------
    # AC-9: fix_session prompt contains required content
    # -----------------------------------------------------------------------

    def test_fix_session_prompt_mission(self):
        """fix_session prompt contains 'YOUR MISSION: Fix all build errors and test failures.'"""
        graph = _graph_fix_iteration()
        prompt = graph.nodes["fix_session"].prompt
        assert "YOUR MISSION: Fix all build errors and test failures." in prompt, (
            "Expected 'YOUR MISSION: Fix all build errors and test failures.' "
            "in fix_session prompt"
        )

    def test_fix_session_prompt_surgical(self):
        """fix_session prompt contains 'Be surgical -- minimal changes to resolve each error.'"""
        graph = _graph_fix_iteration()
        prompt = graph.nodes["fix_session"].prompt
        assert "Be surgical -- minimal changes to resolve each error." in prompt, (
            "Expected 'Be surgical -- minimal changes to resolve each error.' "
            "in fix_session prompt"
        )

    def test_fix_session_prompt_fixing_strategy(self):
        """fix_session prompt contains 'FIXING STRATEGY'."""
        graph = _graph_fix_iteration()
        prompt = graph.nodes["fix_session"].prompt
        assert "FIXING STRATEGY" in prompt, (
            "Expected 'FIXING STRATEGY' in fix_session prompt"
        )

    def test_fix_session_prompt_group_errors_by_file(self):
        """fix_session prompt contains 'Group errors by file'."""
        graph = _graph_fix_iteration()
        prompt = graph.nodes["fix_session"].prompt
        assert "Group errors by file" in prompt, (
            "Expected 'Group errors by file' in fix_session prompt"
        )

    def test_fix_session_prompt_safety_constraints(self):
        """fix_session prompt contains 'SAFETY CONSTRAINTS'."""
        graph = _graph_fix_iteration()
        prompt = graph.nodes["fix_session"].prompt
        assert "SAFETY CONSTRAINTS" in prompt, (
            "Expected 'SAFETY CONSTRAINTS' in fix_session prompt"
        )

    def test_fix_session_prompt_forbidden_commands(self):
        """fix_session prompt contains 'FORBIDDEN commands'."""
        graph = _graph_fix_iteration()
        prompt = graph.nodes["fix_session"].prompt
        assert "FORBIDDEN commands" in prompt, (
            "Expected 'FORBIDDEN commands' in fix_session prompt"
        )

    def test_fix_session_prompt_data_loss_warning(self):
        """fix_session prompt contains data loss violation warning."""
        graph = _graph_fix_iteration()
        prompt = graph.nodes["fix_session"].prompt
        assert "VIOLATION OF THESE CONSTRAINTS MAY CAUSE DATA LOSS" in prompt, (
            "Expected 'VIOLATION OF THESE CONSTRAINTS MAY CAUSE DATA LOSS' "
            "in fix_session prompt"
        )

    def test_fix_session_prompt_project_dir(self):
        """fix_session prompt uses $project_dir (not Jinja2 {{project_dir}})."""
        graph = _graph_fix_iteration()
        prompt = graph.nodes["fix_session"].prompt
        assert "$project_dir" in prompt, "Expected '$project_dir' in fix_session prompt"

    def test_fix_session_prompt_iteration(self):
        """fix_session prompt uses $iteration reference."""
        graph = _graph_fix_iteration()
        prompt = graph.nodes["fix_session"].prompt
        assert "$iteration" in prompt, "Expected '$iteration' in fix_session prompt"

    def test_fix_session_prompt_no_jinja2(self):
        """fix_session prompt does NOT use Jinja2 {{...}} syntax."""
        graph = _graph_fix_iteration()
        prompt = graph.nodes["fix_session"].prompt
        assert "{{" not in prompt and "}}" not in prompt, (
            "fix_session prompt should NOT contain Jinja2 {{...}} syntax"
        )

    # -----------------------------------------------------------------------
    # AC-10: Exactly 4 edges (linear chain)
    # -----------------------------------------------------------------------

    def test_edge_count(self):
        """Exactly 4 edges."""
        graph = _graph_fix_iteration()
        assert len(graph.edges) == 4, (
            f"Expected 4 edges, got {len(graph.edges)}: "
            + "\n".join(f"  {e.from_node} -> {e.to_node}" for e in graph.edges)
        )

    # -----------------------------------------------------------------------
    # AC-11: Chain order: start->read_errors->fix_session->verify->done
    # -----------------------------------------------------------------------

    EXPECTED_CHAIN = [
        ("start", "read_errors"),
        ("read_errors", "fix_session"),
        ("fix_session", "verify"),
        ("verify", "done"),
    ]

    def test_linear_chain_edges(self):
        """All 4 edges form the expected linear chain."""
        graph = _graph_fix_iteration()
        edge_pairs = {(e.from_node, e.to_node) for e in graph.edges}
        missing = [pair for pair in self.EXPECTED_CHAIN if pair not in edge_pairs]
        assert not missing, (
            f"Missing edges: {missing}. Present edges: {sorted(edge_pairs)}"
        )

    @pytest.mark.parametrize("from_node,to_node", EXPECTED_CHAIN)
    def test_each_chain_edge(self, from_node, to_node):
        """Each link in the chain is individually present (per-edge CI visibility)."""
        graph = _graph_fix_iteration()
        edge_pairs = {(e.from_node, e.to_node) for e in graph.edges}
        assert (from_node, to_node) in edge_pairs, (
            f"Edge {from_node} -> {to_node} not found. "
            f"Present edges: {sorted(edge_pairs)}"
        )


# ===========================================================================
# TestFixIterationExecution -- mock-backend execution tests for fix-iteration.dot
# ===========================================================================


class _MockToolHandler:
    """Configurable tool handler for fix-iteration execution tests.

    Returns configurable context_updates per node without executing any shell
    command. Defaults to SUCCESS with empty context_updates.
    """

    def __init__(
        self,
        context_updates_by_node: dict | None = None,
    ) -> None:
        self._context_updates = context_updates_by_node or {}
        self.called: list[str] = []

    async def execute(self, node, context, graph, logs_root: str):  # type: ignore[override]
        import os

        from amplifier_module_loop_pipeline.outcome import Outcome, StageStatus

        self.called.append(node.id)
        os.makedirs(os.path.join(logs_root, node.id), exist_ok=True)
        updates = self._context_updates.get(node.id, {})
        return Outcome(status=StageStatus.SUCCESS, context_updates=updates)


class _CapturingBackend:
    """Codergen backend that captures every (node_id, prompt) pair it receives."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def run(self, node, prompt: str, context):  # type: ignore[override]
        from amplifier_module_loop_pipeline.outcome import Outcome, StageStatus

        self.calls.append((node.id, prompt))
        return Outcome(status=StageStatus.SUCCESS, notes=f"Captured: {node.id}")


class TestFixIterationExecution:
    """Mock-backend execution tests for fix-iteration.dot.

    Verifies that the linear chain executes correctly and that context
    variables are expanded in agent prompts.
    """

    def _make_engine(
        self,
        tmp_path,
        tool_handler=None,
        backend=None,
    ):
        """Build a PipelineEngine over fix-iteration.dot."""
        from amplifier_module_loop_pipeline.context import PipelineContext
        from amplifier_module_loop_pipeline.dot_parser import parse_dot
        from amplifier_module_loop_pipeline.engine import PipelineEngine
        from amplifier_module_loop_pipeline.handlers import HandlerRegistry
        from amplifier_module_loop_pipeline.validation import validate_or_raise

        with open(_FIX_ITERATION_DOT) as f:
            dot_source = f.read()

        graph = parse_dot(dot_source)
        validate_or_raise(graph)

        context = PipelineContext()
        context.set("project_dir", str(tmp_path))
        context.set("project_name", "test-project")
        context.set("build_command", "echo build")
        context.set("test_command", "echo test")
        context.set("build_timeout", "120")
        context.set("max_fix_iterations", "5")
        context.set("iteration", "1")

        # Default no-op backend for codergen nodes
        class _NoOpBackend:
            def __init__(self) -> None:
                self.called: list[str] = []

            async def run(self, node, prompt: str, ctx):  # type: ignore[override]
                from amplifier_module_loop_pipeline.outcome import (
                    Outcome,
                    StageStatus,
                )

                self.called.append(node.id)
                return Outcome(status=StageStatus.SUCCESS, notes=f"NoOp: {node.id}")

        effective_backend = backend if backend is not None else _NoOpBackend()
        registry = HandlerRegistry(backend=effective_backend)
        if tool_handler is not None:
            registry.register("tool", tool_handler)

        return PipelineEngine(
            graph=graph,
            context=context,
            handler_registry=registry,
            logs_root=str(tmp_path / "logs"),
        )

    # -----------------------------------------------------------------------
    # Test 1: All nodes in the linear chain run to completion
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_linear_path_succeeds(self, tmp_path):
        """read_errors -> fix_session -> verify all run successfully.

        fix-iteration.dot is a simple linear chain with no branching.
        All non-structural nodes must appear in completed_nodes after a run.
        """
        mock_tool = _MockToolHandler()
        engine = self._make_engine(tmp_path, tool_handler=mock_tool)

        from amplifier_module_loop_pipeline.outcome import StageStatus

        outcome = await engine.run()

        assert outcome.status == StageStatus.SUCCESS, (
            f"Expected SUCCESS, got {outcome.status!r}"
        )
        for node_id in ("read_errors", "fix_session", "verify"):
            assert node_id in engine.completed_nodes, (
                f"Node '{node_id}' must have run in the linear chain"
            )

    # -----------------------------------------------------------------------
    # Test 2: $iteration is expanded in fix_session prompt
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_fix_session_receives_iteration_from_context(self, tmp_path):
        """CapturingBackend verifies $iteration is expanded in fix_session prompt.

        The fix_session prompt contains '$iteration'. Before dispatching the
        codergen backend, the CodergenHandler expands context variables.
        With iteration=1 in context, the prompt must contain '1' in place of
        '$iteration'.
        """
        capturing = _CapturingBackend()
        mock_tool = _MockToolHandler()
        engine = self._make_engine(tmp_path, tool_handler=mock_tool, backend=capturing)

        await engine.run()

        # Find the fix_session call
        fix_calls = [(nid, p) for nid, p in capturing.calls if nid == "fix_session"]
        assert fix_calls, "fix_session must have been dispatched to the backend"

        _fix_node_id, prompt = fix_calls[0]

        # The prompt should have $iteration expanded to '1'
        # (context was seeded with iteration=1)
        assert "$iteration" not in prompt or "1" in prompt, (
            "fix_session prompt must expand $iteration from context; "
            f"got: {prompt[:200]!r}"
        )
        # More specifically: the literal string '$iteration' should be gone
        # and the iteration number should appear
        assert "1" in prompt, (
            "fix_session prompt must contain the expanded iteration value '1'"
        )
