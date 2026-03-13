"""Tests for Phase 3: iteration.dot -- The Core 8-Step Pipeline.

Verifies that examples/dev-machine/runtime/iteration.dot is correct per spec:
- 13 nodes with correct shapes
- 15 edges
- All required node IDs present
- orient has parse_json='true' and tool_command references $state_file
- spec_drift, api_inventory, module_health have continue_on_fail='true'
- test_preflight does NOT have continue_on_fail
- working_session has context_fidelity='truncate', VERBATIM safety prompt with
  required phrases, $state_file references (not {{state_file}})
- post_session references post-session.dot
- orient_gate routes blocked->done and healthy->spec_drift
- test_preflight_gate routes broken->done and ok->module_health
- build_check has parse_json='true'
- build_gate routes both clean and failed to post_session

Spec coverage: iteration.dot Phase 3 requirements.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from amplifier_module_loop_pipeline.dot_parser import parse_dot

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_TESTS_DIR = os.path.dirname(__file__)
# From modules/loop-pipeline/tests/ -> up 3 levels -> amplifier-bundle-attractor/ -> examples/
_EXAMPLES_DIR = os.path.abspath(os.path.join(_TESTS_DIR, "..", "..", "..", "examples"))
_ITERATION_DOT = os.path.join(_EXAMPLES_DIR, "dev-machine", "runtime", "iteration.dot")


def _load() -> str:
    with open(_ITERATION_DOT) as f:
        return f.read()


def _graph():
    return parse_dot(_load())


# ===========================================================================
# TestIterationParse -- top-level structural tests
# ===========================================================================


class TestIterationParse:
    """Tests for iteration.dot parse correctness and structural requirements."""

    # -----------------------------------------------------------------------
    # AC-1: File exists
    # -----------------------------------------------------------------------

    def test_file_exists(self):
        """iteration.dot exists at examples/dev-machine/runtime/iteration.dot."""
        assert os.path.isfile(_ITERATION_DOT), (
            f"iteration.dot not found at {_ITERATION_DOT}"
        )

    # -----------------------------------------------------------------------
    # AC-2: Parses without error
    # -----------------------------------------------------------------------

    def test_parses_without_error(self):
        """iteration.dot parses without raising exceptions."""
        graph = _graph()
        assert graph is not None

    # -----------------------------------------------------------------------
    # AC-3: Exactly 13 nodes with correct shapes
    # -----------------------------------------------------------------------

    def test_node_count(self):
        """Exactly 13 nodes."""
        graph = _graph()
        assert len(graph.nodes) == 13, (
            f"Expected 13 nodes, got {len(graph.nodes)}: {list(graph.nodes.keys())}"
        )

    def test_parallelogram_count(self):
        """Exactly 6 parallelogram (tool) nodes."""
        graph = _graph()
        count = sum(1 for n in graph.nodes.values() if n.shape == "parallelogram")
        assert count == 6, f"Expected 6 parallelogram nodes, got {count}"

    def test_diamond_count(self):
        """Exactly 3 diamond (conditional gate) nodes."""
        graph = _graph()
        count = sum(1 for n in graph.nodes.values() if n.shape == "diamond")
        assert count == 3, f"Expected 3 diamond nodes, got {count}"

    def test_box_count(self):
        """Exactly 1 box (codegen/LLM) node."""
        graph = _graph()
        count = sum(1 for n in graph.nodes.values() if n.shape == "box")
        assert count == 1, f"Expected 1 box node, got {count}"

    def test_folder_count(self):
        """Exactly 1 folder (sub-pipeline) node."""
        graph = _graph()
        count = sum(1 for n in graph.nodes.values() if n.shape == "folder")
        assert count == 1, f"Expected 1 folder node, got {count}"

    def test_mdiamond_count(self):
        """Exactly 1 Mdiamond (start) node."""
        graph = _graph()
        count = sum(1 for n in graph.nodes.values() if n.shape == "Mdiamond")
        assert count == 1, f"Expected 1 Mdiamond node, got {count}"

    def test_msquare_count(self):
        """Exactly 1 Msquare (done/exit) node."""
        graph = _graph()
        count = sum(1 for n in graph.nodes.values() if n.shape == "Msquare")
        assert count == 1, f"Expected 1 Msquare node, got {count}"

    # -----------------------------------------------------------------------
    # AC-4: All 13 required node IDs present
    # -----------------------------------------------------------------------

    REQUIRED_NODE_IDS = [
        "start",
        "orient",
        "orient_gate",
        "spec_drift",
        "api_inventory",
        "test_preflight",
        "test_preflight_gate",
        "module_health",
        "working_session",
        "build_check",
        "build_gate",
        "post_session",
        "done",
    ]

    def test_all_required_node_ids_present(self):
        """All 13 required node IDs are present.

        Intentionally overlaps with test_each_required_node_id[*] below:
        this bulk test catches the complete set in one assertion, while the
        parametrized variant gives per-node visibility in CI output so a
        failure names exactly which node is missing.
        """
        graph = _graph()
        missing = [nid for nid in self.REQUIRED_NODE_IDS if nid not in graph.nodes]
        assert not missing, (
            f"Missing node IDs: {missing}. Present: {list(graph.nodes.keys())}"
        )

    @pytest.mark.parametrize("node_id", REQUIRED_NODE_IDS)
    def test_each_required_node_id(self, node_id):
        """Each required node ID is individually present (per-node CI visibility)."""
        graph = _graph()
        assert node_id in graph.nodes, (
            f"Node '{node_id}' not found. Present: {list(graph.nodes.keys())}"
        )

    # -----------------------------------------------------------------------
    # AC-5: orient has parse_json='true' and tool_command references $state_file
    # -----------------------------------------------------------------------

    def test_orient_shape(self):
        """orient node has shape=parallelogram."""
        graph = _graph()
        assert graph.nodes["orient"].shape == "parallelogram", (
            f"Expected orient shape=parallelogram, got {graph.nodes['orient'].shape!r}"
        )

    def test_orient_parse_json(self):
        """orient node has parse_json='true'."""
        graph = _graph()
        val = graph.nodes["orient"].attrs.get("parse_json")
        assert val == "true", f"Expected orient parse_json='true', got {val!r}"

    def test_orient_tool_command_references_state_file(self):
        """orient node's tool_command contains $state_file."""
        graph = _graph()
        cmd = graph.nodes["orient"].attrs.get("tool_command", "")
        assert "$state_file" in cmd, (
            f"Expected orient tool_command to reference $state_file, got: {cmd!r}"
        )

    # -----------------------------------------------------------------------
    # AC-6: spec_drift, api_inventory, module_health have continue_on_fail='true'
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize(
        "node_id", ["spec_drift", "api_inventory", "module_health"]
    )
    def test_continue_on_fail_nodes(self, node_id):
        """spec_drift, api_inventory, module_health have continue_on_fail='true'."""
        graph = _graph()
        val = graph.nodes[node_id].attrs.get("continue_on_fail")
        assert val == "true", f"Expected {node_id} continue_on_fail='true', got {val!r}"

    @pytest.mark.parametrize(
        "node_id", ["spec_drift", "api_inventory", "module_health"]
    )
    def test_continue_on_fail_nodes_are_parallelogram(self, node_id):
        """spec_drift, api_inventory, module_health have shape=parallelogram."""
        graph = _graph()
        assert graph.nodes[node_id].shape == "parallelogram", (
            f"Expected {node_id} shape=parallelogram, got {graph.nodes[node_id].shape!r}"
        )

    # -----------------------------------------------------------------------
    # AC-7: test_preflight does NOT have continue_on_fail
    # -----------------------------------------------------------------------

    def test_test_preflight_no_continue_on_fail(self):
        """test_preflight does NOT have continue_on_fail (it is a hard gate)."""
        graph = _graph()
        val = graph.nodes["test_preflight"].attrs.get("continue_on_fail")
        assert val != "true", (
            f"test_preflight should NOT have continue_on_fail='true', got {val!r}"
        )

    def test_test_preflight_shape(self):
        """test_preflight has shape=parallelogram."""
        graph = _graph()
        assert graph.nodes["test_preflight"].shape == "parallelogram", (
            f"Expected test_preflight shape=parallelogram, "
            f"got {graph.nodes['test_preflight'].shape!r}"
        )

    # -----------------------------------------------------------------------
    # AC-8: working_session attributes and prompt content
    # -----------------------------------------------------------------------

    def test_working_session_shape(self):
        """working_session has shape=box."""
        graph = _graph()
        assert graph.nodes["working_session"].shape == "box", (
            f"Expected working_session shape=box, "
            f"got {graph.nodes['working_session'].shape!r}"
        )

    def test_working_session_context_fidelity(self):
        """working_session has context_fidelity='truncate'."""
        graph = _graph()
        val = graph.nodes["working_session"].attrs.get("context_fidelity")
        assert val == "truncate", (
            f"Expected working_session context_fidelity='truncate', got {val!r}"
        )

    def test_working_session_prompt_safety_constraints(self):
        """working_session prompt contains 'SAFETY CONSTRAINTS'."""
        graph = _graph()
        prompt = graph.nodes["working_session"].prompt
        assert "SAFETY CONSTRAINTS" in prompt, (
            "Expected 'SAFETY CONSTRAINTS' in working_session prompt"
        )

    def test_working_session_prompt_forbidden_commands(self):
        """working_session prompt contains 'FORBIDDEN commands'."""
        graph = _graph()
        prompt = graph.nodes["working_session"].prompt
        assert "FORBIDDEN commands" in prompt, (
            "Expected 'FORBIDDEN commands' in working_session prompt"
        )

    def test_working_session_prompt_data_loss_warning(self):
        """working_session prompt contains data loss violation warning."""
        graph = _graph()
        prompt = graph.nodes["working_session"].prompt
        assert "VIOLATION OF THESE CONSTRAINTS MAY CAUSE DATA LOSS" in prompt, (
            "Expected 'VIOLATION OF THESE CONSTRAINTS MAY CAUSE DATA LOSS' "
            "in working_session prompt"
        )

    def test_working_session_prompt_platform_grounding(self):
        """working_session prompt contains 'Platform Grounding'."""
        graph = _graph()
        prompt = graph.nodes["working_session"].prompt
        assert "Platform Grounding" in prompt, (
            "Expected 'Platform Grounding' in working_session prompt"
        )

    def test_working_session_prompt_hallucinating_apis(self):
        """working_session prompt contains 'Hallucinating APIs'."""
        graph = _graph()
        prompt = graph.nodes["working_session"].prompt
        assert "Hallucinating APIs" in prompt, (
            "Expected 'Hallucinating APIs' in working_session prompt"
        )

    def test_working_session_prompt_state_file_dollar_sign(self):
        """working_session prompt uses $state_file (not Jinja2 {{state_file}})."""
        graph = _graph()
        prompt = graph.nodes["working_session"].prompt
        assert "$state_file" in prompt, (
            "Expected '$state_file' in working_session prompt"
        )

    def test_working_session_prompt_no_jinja2(self):
        """working_session prompt does NOT use Jinja2 {{...}} syntax."""
        graph = _graph()
        prompt = graph.nodes["working_session"].prompt
        assert "{{" not in prompt and "}}" not in prompt, (
            "working_session prompt should NOT contain Jinja2 {{...}} syntax"
        )

    # -----------------------------------------------------------------------
    # AC-9: post_session references post-session.dot
    # -----------------------------------------------------------------------

    def test_post_session_shape(self):
        """post_session has shape=folder."""
        graph = _graph()
        assert graph.nodes["post_session"].shape == "folder", (
            f"Expected post_session shape=folder, "
            f"got {graph.nodes['post_session'].shape!r}"
        )

    def test_post_session_dot_file(self):
        """post_session references post-session.dot via dot_file attribute."""
        graph = _graph()
        dot_file = graph.nodes["post_session"].attrs.get("dot_file", "")
        assert "post-session.dot" in dot_file, (
            f"Expected post_session dot_file to contain 'post-session.dot', "
            f"got {dot_file!r}"
        )

    # -----------------------------------------------------------------------
    # AC-10: Exactly 15 edges
    # -----------------------------------------------------------------------

    def test_edge_count(self):
        """Exactly 15 edges."""
        graph = _graph()
        assert len(graph.edges) == 15, (
            f"Expected 15 edges, got {len(graph.edges)}: "
            + "\n".join(f"  {e.from_node} -> {e.to_node}" for e in graph.edges)
        )

    # -----------------------------------------------------------------------
    # AC-11: orient_gate edges: blocked->done, healthy->spec_drift
    # -----------------------------------------------------------------------

    def test_orient_gate_shape(self):
        """orient_gate has shape=diamond."""
        graph = _graph()
        assert graph.nodes["orient_gate"].shape == "diamond", (
            f"Expected orient_gate shape=diamond, "
            f"got {graph.nodes['orient_gate'].shape!r}"
        )

    def test_orient_gate_blocked_to_done(self):
        """orient_gate has a 'blocked' edge routing to done."""
        graph = _graph()
        gate_edges = [e for e in graph.edges if e.from_node == "orient_gate"]
        done_edges = [e for e in gate_edges if e.to_node == "done"]
        assert done_edges, (
            "Expected orient_gate -> done edge. orient_gate edges: "
            + str([(e.to_node, e.label, e.condition) for e in gate_edges])
        )
        # Verify it has a 'blocked' label or condition
        done_edge = done_edges[0]
        has_blocked = (done_edge.label and "blocked" in done_edge.label.lower()) or (
            done_edge.condition and "blocked" in done_edge.condition.lower()
        )
        assert has_blocked, (
            f"Expected orient_gate -> done edge to have 'blocked' label/condition, "
            f"got label={done_edge.label!r}, condition={done_edge.condition!r}"
        )

    def test_orient_gate_healthy_to_spec_drift(self):
        """orient_gate has a 'healthy' edge routing to spec_drift."""
        graph = _graph()
        gate_edges = [e for e in graph.edges if e.from_node == "orient_gate"]
        drift_edges = [e for e in gate_edges if e.to_node == "spec_drift"]
        assert drift_edges, (
            "Expected orient_gate -> spec_drift edge. orient_gate edges: "
            + str([(e.to_node, e.label, e.condition) for e in gate_edges])
        )
        drift_edge = drift_edges[0]
        has_healthy = (drift_edge.label and "healthy" in drift_edge.label.lower()) or (
            drift_edge.condition and "healthy" in drift_edge.condition.lower()
        )
        assert has_healthy, (
            f"Expected orient_gate -> spec_drift edge to have 'healthy' label/condition, "
            f"got label={drift_edge.label!r}, condition={drift_edge.condition!r}"
        )

    # -----------------------------------------------------------------------
    # AC-12: test_preflight_gate has broken->done edge
    # -----------------------------------------------------------------------

    def test_test_preflight_gate_shape(self):
        """test_preflight_gate has shape=diamond."""
        graph = _graph()
        assert graph.nodes["test_preflight_gate"].shape == "diamond", (
            f"Expected test_preflight_gate shape=diamond, "
            f"got {graph.nodes['test_preflight_gate'].shape!r}"
        )

    def test_test_preflight_gate_broken_to_done(self):
        """test_preflight_gate has a 'broken' edge routing to done."""
        graph = _graph()
        gate_edges = [e for e in graph.edges if e.from_node == "test_preflight_gate"]
        done_edges = [e for e in gate_edges if e.to_node == "done"]
        assert done_edges, "Expected test_preflight_gate -> done edge. Edges: " + str(
            [(e.to_node, e.label, e.condition) for e in gate_edges]
        )
        done_edge = done_edges[0]
        has_broken = (done_edge.label and "broken" in done_edge.label.lower()) or (
            done_edge.condition and "broken" in done_edge.condition.lower()
        )
        assert has_broken, (
            f"Expected test_preflight_gate -> done edge to have 'broken' label/condition, "
            f"got label={done_edge.label!r}, condition={done_edge.condition!r}"
        )

    def test_test_preflight_gate_ok_to_module_health(self):
        """test_preflight_gate has an 'ok' edge routing to module_health."""
        graph = _graph()
        gate_edges = [e for e in graph.edges if e.from_node == "test_preflight_gate"]
        health_edges = [e for e in gate_edges if e.to_node == "module_health"]
        assert health_edges, (
            "Expected test_preflight_gate -> module_health edge. Edges: "
            + str([(e.to_node, e.label, e.condition) for e in gate_edges])
        )

    # -----------------------------------------------------------------------
    # AC-13: build_check has parse_json='true'
    # -----------------------------------------------------------------------

    def test_build_check_shape(self):
        """build_check has shape=parallelogram."""
        graph = _graph()
        assert graph.nodes["build_check"].shape == "parallelogram", (
            f"Expected build_check shape=parallelogram, "
            f"got {graph.nodes['build_check'].shape!r}"
        )

    def test_build_check_parse_json(self):
        """build_check has parse_json='true'."""
        graph = _graph()
        val = graph.nodes["build_check"].attrs.get("parse_json")
        assert val == "true", f"Expected build_check parse_json='true', got {val!r}"

    # -----------------------------------------------------------------------
    # AC-14: build_gate routes both clean and failed to post_session
    # -----------------------------------------------------------------------

    def test_build_gate_shape(self):
        """build_gate has shape=diamond."""
        graph = _graph()
        assert graph.nodes["build_gate"].shape == "diamond", (
            f"Expected build_gate shape=diamond, "
            f"got {graph.nodes['build_gate'].shape!r}"
        )

    def test_build_gate_clean_to_post_session(self):
        """build_gate has a 'clean' edge routing to post_session."""
        graph = _graph()
        gate_edges = [e for e in graph.edges if e.from_node == "build_gate"]
        post_edges = [e for e in gate_edges if e.to_node == "post_session"]
        assert len(post_edges) >= 1, (
            "Expected at least one build_gate -> post_session edge. Edges: "
            + str([(e.to_node, e.label, e.condition) for e in gate_edges])
        )
        # At least one must mention 'clean'
        clean_edges = [
            e
            for e in post_edges
            if (e.label and "clean" in e.label.lower())
            or (e.condition and "clean" in e.condition.lower())
        ]
        assert clean_edges, (
            "Expected a build_gate -> post_session 'clean' edge. "
            "post_session edges: " + str([(e.label, e.condition) for e in post_edges])
        )

    def test_build_gate_failed_to_post_session(self):
        """build_gate has a 'failed' edge routing to post_session."""
        graph = _graph()
        gate_edges = [e for e in graph.edges if e.from_node == "build_gate"]
        post_edges = [e for e in gate_edges if e.to_node == "post_session"]
        assert len(post_edges) >= 1, (
            "Expected at least one build_gate -> post_session edge."
        )
        # At least one must mention 'failed'
        failed_edges = [
            e
            for e in post_edges
            if (e.label and "failed" in e.label.lower())
            or (e.condition and "failed" in e.condition.lower())
        ]
        assert failed_edges, (
            "Expected a build_gate -> post_session 'failed' edge. "
            "post_session edges: " + str([(e.label, e.condition) for e in post_edges])
        )

    def test_build_gate_both_route_to_post_session(self):
        """Both build_gate outgoing edges route to post_session."""
        graph = _graph()
        gate_edges = [e for e in graph.edges if e.from_node == "build_gate"]
        assert len(gate_edges) >= 2, (
            f"Expected at least 2 edges from build_gate, got {len(gate_edges)}"
        )
        for e in gate_edges:
            assert e.to_node == "post_session", (
                f"Expected all build_gate edges to route to post_session, "
                f"but {e.from_node} -> {e.to_node} (label={e.label!r})"
            )


# ===========================================================================
# TestPostSessionParse -- post-session.dot structural tests
# ===========================================================================

_POST_SESSION_DOT = os.path.join(
    _EXAMPLES_DIR, "dev-machine", "runtime", "post-session.dot"
)


def _load_post_session() -> str:
    with open(_POST_SESSION_DOT) as f:
        return f.read()


def _graph_post_session():
    return parse_dot(_load_post_session())


class TestPostSessionParse:
    """Tests for post-session.dot parse correctness and structural requirements."""

    # -----------------------------------------------------------------------
    # AC-1: File exists
    # -----------------------------------------------------------------------

    def test_file_exists(self):
        """post-session.dot exists at examples/dev-machine/runtime/post-session.dot."""
        assert os.path.isfile(_POST_SESSION_DOT), (
            f"post-session.dot not found at {_POST_SESSION_DOT}"
        )

    # -----------------------------------------------------------------------
    # AC-2: Parses without error
    # -----------------------------------------------------------------------

    def test_parses_without_error(self):
        """post-session.dot parses without raising exceptions."""
        graph = _graph_post_session()
        assert graph is not None

    # -----------------------------------------------------------------------
    # AC-3: Exactly 7 nodes
    # -----------------------------------------------------------------------

    def test_node_count(self):
        """Exactly 7 nodes."""
        graph = _graph_post_session()
        assert len(graph.nodes) == 7, (
            f"Expected 7 nodes, got {len(graph.nodes)}: {list(graph.nodes.keys())}"
        )

    # -----------------------------------------------------------------------
    # AC-4: Exactly 5 parallelogram nodes
    # -----------------------------------------------------------------------

    def test_parallelogram_count(self):
        """Exactly 5 parallelogram (tool) nodes."""
        graph = _graph_post_session()
        count = sum(1 for n in graph.nodes.values() if n.shape == "parallelogram")
        assert count == 5, f"Expected 5 parallelogram nodes, got {count}"

    # -----------------------------------------------------------------------
    # AC-5: All 7 required node IDs present
    # -----------------------------------------------------------------------

    REQUIRED_NODE_IDS = [
        "start",
        "archive_features",
        "session_accounting",
        "reconcile",
        "periodic_check",
        "status_output",
        "done",
    ]

    def test_all_required_node_ids_present(self):
        """All 7 required node IDs are present.

        Intentionally overlaps with test_each_required_node_id[*] below:
        this bulk test catches the complete set in one assertion, while the
        parametrized variant gives per-node visibility in CI output so a
        failure names exactly which node is missing.
        """
        graph = _graph_post_session()
        missing = [nid for nid in self.REQUIRED_NODE_IDS if nid not in graph.nodes]
        assert not missing, (
            f"Missing node IDs: {missing}. Present: {list(graph.nodes.keys())}"
        )

    @pytest.mark.parametrize("node_id", REQUIRED_NODE_IDS)
    def test_each_required_node_id(self, node_id):
        """Each required node ID is individually present (per-node CI visibility)."""
        graph = _graph_post_session()
        assert node_id in graph.nodes, (
            f"Node '{node_id}' not found. Present: {list(graph.nodes.keys())}"
        )

    # -----------------------------------------------------------------------
    # AC-6: reconcile and periodic_check have continue_on_fail='true'
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("node_id", ["reconcile", "periodic_check"])
    def test_continue_on_fail_nodes(self, node_id):
        """reconcile and periodic_check have continue_on_fail='true'."""
        graph = _graph_post_session()
        val = graph.nodes[node_id].attrs.get("continue_on_fail")
        assert val == "true", f"Expected {node_id} continue_on_fail='true', got {val!r}"

    @pytest.mark.parametrize("node_id", ["reconcile", "periodic_check"])
    def test_continue_on_fail_nodes_are_parallelogram(self, node_id):
        """reconcile and periodic_check have shape=parallelogram."""
        graph = _graph_post_session()
        assert graph.nodes[node_id].shape == "parallelogram", (
            f"Expected {node_id} shape=parallelogram, "
            f"got {graph.nodes[node_id].shape!r}"
        )

    # -----------------------------------------------------------------------
    # AC-7: archive_features and session_accounting do NOT have continue_on_fail
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("node_id", ["archive_features", "session_accounting"])
    def test_critical_nodes_no_continue_on_fail(self, node_id):
        """archive_features and session_accounting do NOT have continue_on_fail."""
        graph = _graph_post_session()
        val = graph.nodes[node_id].attrs.get("continue_on_fail")
        assert val != "true", (
            f"{node_id} should NOT have continue_on_fail='true', got {val!r}"
        )

    @pytest.mark.parametrize("node_id", ["archive_features", "session_accounting"])
    def test_critical_nodes_are_parallelogram(self, node_id):
        """archive_features and session_accounting have shape=parallelogram."""
        graph = _graph_post_session()
        assert graph.nodes[node_id].shape == "parallelogram", (
            f"Expected {node_id} shape=parallelogram, "
            f"got {graph.nodes[node_id].shape!r}"
        )

    # -----------------------------------------------------------------------
    # AC-8: status_output has parse_json='true'
    # -----------------------------------------------------------------------

    def test_status_output_parse_json(self):
        """status_output has parse_json='true'."""
        graph = _graph_post_session()
        val = graph.nodes["status_output"].attrs.get("parse_json")
        assert val == "true", f"Expected status_output parse_json='true', got {val!r}"

    def test_status_output_shape(self):
        """status_output has shape=parallelogram."""
        graph = _graph_post_session()
        assert graph.nodes["status_output"].shape == "parallelogram", (
            f"Expected status_output shape=parallelogram, "
            f"got {graph.nodes['status_output'].shape!r}"
        )

    def test_status_output_no_continue_on_fail(self):
        """status_output does NOT have continue_on_fail."""
        graph = _graph_post_session()
        val = graph.nodes["status_output"].attrs.get("continue_on_fail")
        assert val != "true", (
            f"status_output should NOT have continue_on_fail='true', got {val!r}"
        )

    # -----------------------------------------------------------------------
    # AC-9: Exactly 6 edges (linear chain)
    # -----------------------------------------------------------------------

    def test_edge_count(self):
        """Exactly 6 edges."""
        graph = _graph_post_session()
        assert len(graph.edges) == 6, (
            f"Expected 6 edges, got {len(graph.edges)}: "
            + "\n".join(f"  {e.from_node} -> {e.to_node}" for e in graph.edges)
        )

    # -----------------------------------------------------------------------
    # AC-10: Chain order verified
    #   start -> archive_features -> session_accounting -> reconcile
    #   -> periodic_check -> status_output -> done
    # -----------------------------------------------------------------------

    EXPECTED_CHAIN = [
        ("start", "archive_features"),
        ("archive_features", "session_accounting"),
        ("session_accounting", "reconcile"),
        ("reconcile", "periodic_check"),
        ("periodic_check", "status_output"),
        ("status_output", "done"),
    ]

    def test_linear_chain_edges(self):
        """All 6 edges form the expected linear chain."""
        graph = _graph_post_session()
        edge_pairs = {(e.from_node, e.to_node) for e in graph.edges}
        missing = [pair for pair in self.EXPECTED_CHAIN if pair not in edge_pairs]
        assert not missing, (
            f"Missing edges: {missing}. Present edges: {sorted(edge_pairs)}"
        )

    @pytest.mark.parametrize("from_node,to_node", EXPECTED_CHAIN)
    def test_each_chain_edge(self, from_node, to_node):
        """Each link in the chain is individually present (per-edge CI visibility)."""
        graph = _graph_post_session()
        edge_pairs = {(e.from_node, e.to_node) for e in graph.edges}
        assert (from_node, to_node) in edge_pairs, (
            f"Edge {from_node} -> {to_node} not found. "
            f"Present edges: {sorted(edge_pairs)}"
        )


# ===========================================================================
# TestIterationExecution -- mock-backend execution tests for iteration.dot
# ===========================================================================

_STUB_POST_SESSION_DOT = """\
digraph PostSession {
    start [shape=Mdiamond]
    done  [shape=Msquare]
    start -> done
}
"""


class MockToolHandler:
    """Configurable tool handler that returns specified context_updates per node.

    Simulates parse_json by returning context_updates directly without executing
    any shell command. Used to control conditional routing in execution tests.
    """

    def __init__(
        self,
        context_updates_by_node: dict[str, dict] | None = None,
        failing_nodes: set[str] | None = None,
    ) -> None:
        self._context_updates = context_updates_by_node or {}
        self._failing_nodes = failing_nodes or set()
        self.called: list[str] = []

    async def execute(  # type: ignore[override]
        self,
        node: "Any",
        context: "Any",
        graph: "Any",
        logs_root: str,
    ) -> "Any":
        import os

        from amplifier_module_loop_pipeline.outcome import Outcome, StageStatus

        self.called.append(node.id)
        os.makedirs(os.path.join(logs_root, node.id), exist_ok=True)

        if node.id in self._failing_nodes:
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason=f"Simulated tool failure for node {node.id!r}",
            )
        updates = self._context_updates.get(node.id, {})
        return Outcome(status=StageStatus.SUCCESS, context_updates=updates)


class FailingToolHandler:
    """Tool handler that always returns FAIL for every node."""

    def __init__(self) -> None:
        self.called: list[str] = []

    async def execute(  # type: ignore[override]
        self,
        node: "Any",
        context: "Any",
        graph: "Any",
        logs_root: str,
    ) -> "Any":
        import os

        from amplifier_module_loop_pipeline.outcome import Outcome, StageStatus

        self.called.append(node.id)
        os.makedirs(os.path.join(logs_root, node.id), exist_ok=True)
        return Outcome(status=StageStatus.FAIL, failure_reason="tool always fails")


class NoOpBackend:
    """Codergen backend that returns SUCCESS for every node (no LLM calls)."""

    def __init__(self) -> None:
        self.called: list[str] = []

    async def run(self, node: "Any", prompt: str, context: "Any") -> "Any":
        from amplifier_module_loop_pipeline.outcome import Outcome, StageStatus

        self.called.append(node.id)
        return Outcome(status=StageStatus.SUCCESS, notes=f"NoOp: {node.id}")


class TestIterationExecution:
    """Mock-backend execution tests for iteration.dot.

    These tests verify routing behavior without executing real shell commands
    or spawning LLM sessions. The MockToolHandler controls what context_updates
    each tool node produces, which drives the conditional gates.
    """

    def _make_engine(
        self,
        tmp_path,
        tool_handler: object | None = None,
        backend: object | None = None,
    ):
        """Build a PipelineEngine over a patched iteration.dot.

        Patches applied:
        1. dot_file for post_session -> absolute path of stub post-session.dot
        2. post_session -> start edge -> post_session -> done (avoid infinite loop)
        """
        from amplifier_module_loop_pipeline.context import PipelineContext
        from amplifier_module_loop_pipeline.dot_parser import parse_dot
        from amplifier_module_loop_pipeline.engine import PipelineEngine
        from amplifier_module_loop_pipeline.handlers import HandlerRegistry
        from amplifier_module_loop_pipeline.validation import validate_or_raise

        # 1. Write stub post-session.dot
        stub_path = tmp_path / "post-session.dot"
        stub_path.write_text(_STUB_POST_SESSION_DOT)

        # 2. Load and patch iteration.dot
        with open(_ITERATION_DOT) as f:
            dot_source = f.read()

        # Patch dot_file to absolute path so PipelineHandler can find it
        dot_source = dot_source.replace(
            'dot_file="post-session.dot"',
            f'dot_file="{stub_path}"',
        )
        # Break the post_session -> start loop to prevent infinite iteration
        dot_source = dot_source.replace(
            "post_session -> start",
            "post_session -> done",
        )

        # 3. Parse and validate
        graph = parse_dot(dot_source)
        validate_or_raise(graph)

        # 4. Set required context variables
        context = PipelineContext()
        context.set("state_file", str(tmp_path / "STATE.yaml"))
        context.set("context_file", str(tmp_path / "CONTEXT.md"))
        context.set("specs_dir", str(tmp_path / "specs"))
        context.set("project_dir", str(tmp_path))
        context.set("architecture_spec", str(tmp_path / "ARCHITECTURE.md"))
        context.set("test_command", "echo test")
        context.set("build_command", "echo build")
        context.set("commit_prefix", "test")
        context.set("max_features_per_session", "3")
        context.set("module_size_threshold", "500")
        context.set("session_timeout", "3600")
        context.set("build_timeout", "120")

        # 5. Build HandlerRegistry with optional mock overrides
        registry = HandlerRegistry(backend=backend or NoOpBackend())
        if tool_handler is not None:
            registry.register("tool", tool_handler)

        # 6. Create engine
        logs_root = str(tmp_path / "logs")
        return PipelineEngine(
            graph=graph,
            context=context,
            handler_registry=registry,
            logs_root=logs_root,
        )

    # -----------------------------------------------------------------------
    # Test 1: orient returns status=blocked → exits at orient_gate → done
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_blocked_status_exits_early(self, tmp_path):
        """orient returns status=blocked: spec_drift and working_session must NOT run.

        When orient produces context.status=blocked, orient_gate routes directly
        to done. The pipeline should exit without running any downstream nodes.
        """
        mock_tool = MockToolHandler(
            context_updates_by_node={"orient": {"status": "blocked"}}
        )
        engine = self._make_engine(tmp_path, tool_handler=mock_tool)

        await engine.run()

        assert "orient" in engine.completed_nodes, "orient must have run"
        assert "orient_gate" in engine.completed_nodes, "orient_gate must have run"
        assert "spec_drift" not in engine.completed_nodes, (
            "spec_drift must NOT run when orient returns status=blocked"
        )
        assert "working_session" not in engine.completed_nodes, (
            "working_session must NOT run when orient returns status=blocked"
        )

    # -----------------------------------------------------------------------
    # Test 2: orient returns status=healthy → proceeds to spec_drift
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_healthy_status_proceeds_to_preflight(self, tmp_path):
        """orient returns status=healthy: spec_drift must run.

        After orient produces context.status=healthy, orient_gate routes to
        spec_drift. Exit early by having test_preflight produce test_env=broken
        so test_preflight_gate routes to done without wasting test time.
        """
        mock_tool = MockToolHandler(
            context_updates_by_node={
                "orient": {"status": "healthy"},
                "test_preflight": {"test_env": "broken"},
            }
        )
        engine = self._make_engine(tmp_path, tool_handler=mock_tool)

        await engine.run()

        assert "spec_drift" in engine.completed_nodes, (
            "spec_drift must run when orient returns status=healthy"
        )
        assert "working_session" not in engine.completed_nodes, (
            "working_session must NOT run when test_preflight returns test_env=broken"
        )

    # -----------------------------------------------------------------------
    # Test 3: test_preflight returns test_env=broken → exits early
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_test_env_broken_exits_early(self, tmp_path):
        """test_preflight returns test_env=broken: module_health and working_session must NOT run.

        test_preflight_gate has a hard-gate 'broken' edge to done.
        When the test environment is broken, neither module_health nor working_session
        should execute (no point in running code when the test runner is broken).
        """
        mock_tool = MockToolHandler(
            context_updates_by_node={
                "orient": {"status": "healthy"},
                "test_preflight": {"test_env": "broken"},
            }
        )
        engine = self._make_engine(tmp_path, tool_handler=mock_tool)

        await engine.run()

        assert "test_preflight" in engine.completed_nodes, (
            "test_preflight must have run"
        )
        assert "module_health" not in engine.completed_nodes, (
            "module_health must NOT run when test_env=broken"
        )
        assert "working_session" not in engine.completed_nodes, (
            "working_session must NOT run when test_env=broken"
        )

    # -----------------------------------------------------------------------
    # Test 4: continue_on_fail nodes fail but pipeline reaches working_session
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_continue_on_fail_preflight_nodes_dont_halt(self, tmp_path):
        """spec_drift/api_inventory/module_health fail → pipeline still reaches working_session.

        All three nodes have continue_on_fail='true', so FAIL outcomes are
        overridden to SUCCESS for routing purposes. The pipeline must proceed
        all the way to working_session despite these failures.
        """
        mock_tool = MockToolHandler(
            context_updates_by_node={
                "orient": {"status": "healthy"},
                "test_preflight": {"test_env": "ok"},
                "build_check": {"build_status": "clean"},
            },
            failing_nodes={"spec_drift", "api_inventory", "module_health"},
        )
        backend = NoOpBackend()
        engine = self._make_engine(tmp_path, tool_handler=mock_tool, backend=backend)

        await engine.run()

        assert "working_session" in engine.completed_nodes, (
            "working_session must run even when spec_drift/api_inventory/module_health fail"
        )
        assert "spec_drift" in engine.completed_nodes, (
            "spec_drift must have been attempted (then continue_on_fail overrides)"
        )
        assert "api_inventory" in engine.completed_nodes, (
            "api_inventory must have been attempted"
        )
        assert "module_health" in engine.completed_nodes, (
            "module_health must have been attempted"
        )
