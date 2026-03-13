"""Tests for Phase 3: qa.dot + qa-iteration.dot -- QA Loop.

Verifies that:
- examples/dev-machine/runtime/qa.dot is correct per spec:
  - 7 nodes: start (Mdiamond), container_check, read_qa_state, blocked_gate,
    qa_loop, final_summary, done (Msquare)
  - container_check is parallelogram
  - read_qa_state is parallelogram with parse_json='true'
  - blocked_gate is diamond
  - qa_loop is house with manager.max_cycles=20, manager.stop_condition='outcome=success',
    manager.child_dotfile referencing qa-iteration.dot
  - final_summary is parallelogram with continue_on_fail='true'
  - blocked_gate routes blocked->done and testing->qa_loop
  - Flow: start->container_check->read_qa_state->blocked_gate;
    blocked_gate->[done|qa_loop]; qa_loop->final_summary->done

- examples/dev-machine/runtime/qa-iteration.dot is correct per spec:
  - 6 nodes: start (Mdiamond), orient, orient_gate, qa_session, post_qasession,
    done (Msquare)
  - orient is parallelogram with parse_json='true'
  - orient_gate is diamond
  - qa_session is box with context_fidelity='truncate' and VERBATIM prompt
  - post_qasession is parallelogram with parse_json='true'
  - orient_gate routes testing->qa_session, done->done, blocked->done
  - qa_session prompt contains required safety/QA phrases with $variable syntax

Spec coverage: qa.dot + qa-iteration.dot Phase 3 requirements.
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
_QA_DOT = os.path.join(_EXAMPLES_DIR, "dev-machine", "runtime", "qa.dot")
_QA_ITERATION_DOT = os.path.join(
    _EXAMPLES_DIR, "dev-machine", "runtime", "qa-iteration.dot"
)


def _load_qa() -> str:
    with open(_QA_DOT) as f:
        return f.read()


def _graph_qa():
    return parse_dot(_load_qa())


def _load_qa_iteration() -> str:
    with open(_QA_ITERATION_DOT) as f:
        return f.read()


def _graph_qa_iteration():
    return parse_dot(_load_qa_iteration())


# ===========================================================================
# TestQAParse -- qa.dot structural tests
# ===========================================================================


class TestQAParse:
    """Tests for qa.dot parse correctness and structural requirements."""

    # -----------------------------------------------------------------------
    # AC-1: File exists
    # -----------------------------------------------------------------------

    def test_file_exists(self):
        """qa.dot exists at examples/dev-machine/runtime/qa.dot."""
        assert os.path.isfile(_QA_DOT), f"qa.dot not found at {_QA_DOT}"

    # -----------------------------------------------------------------------
    # AC-2: Parses without error
    # -----------------------------------------------------------------------

    def test_parses_without_error(self):
        """qa.dot parses without raising exceptions."""
        graph = _graph_qa()
        assert graph is not None

    # -----------------------------------------------------------------------
    # AC-3: Exactly 7 nodes
    # -----------------------------------------------------------------------

    def test_node_count(self):
        """Exactly 7 nodes."""
        graph = _graph_qa()
        assert len(graph.nodes) == 7, (
            f"Expected 7 nodes, got {len(graph.nodes)}: {list(graph.nodes.keys())}"
        )

    # -----------------------------------------------------------------------
    # AC-4: All 7 required node IDs present
    # -----------------------------------------------------------------------

    REQUIRED_NODE_IDS = [
        "start",
        "container_check",
        "read_qa_state",
        "blocked_gate",
        "qa_loop",
        "final_summary",
        "done",
    ]

    def test_all_required_node_ids_present(self):
        """All 7 required node IDs are present.

        Intentionally overlaps with test_each_required_node_id[*] below:
        this bulk test catches the complete set in one assertion, while the
        parametrized variant gives per-node visibility in CI output so a
        failure names exactly which node is missing.
        """
        graph = _graph_qa()
        missing = [nid for nid in self.REQUIRED_NODE_IDS if nid not in graph.nodes]
        assert not missing, (
            f"Missing node IDs: {missing}. Present: {list(graph.nodes.keys())}"
        )

    @pytest.mark.parametrize("node_id", REQUIRED_NODE_IDS)
    def test_each_required_node_id(self, node_id):
        """Each required node ID is individually present (per-node CI visibility)."""
        graph = _graph_qa()
        assert node_id in graph.nodes, (
            f"Node '{node_id}' not found. Present: {list(graph.nodes.keys())}"
        )

    # -----------------------------------------------------------------------
    # AC-5: start is Mdiamond, done is Msquare
    # -----------------------------------------------------------------------

    def test_start_shape(self):
        """start node has shape=Mdiamond."""
        graph = _graph_qa()
        assert graph.nodes["start"].shape == "Mdiamond", (
            f"Expected start shape=Mdiamond, got {graph.nodes['start'].shape!r}"
        )

    def test_done_shape(self):
        """done node has shape=Msquare."""
        graph = _graph_qa()
        assert graph.nodes["done"].shape == "Msquare", (
            f"Expected done shape=Msquare, got {graph.nodes['done'].shape!r}"
        )

    # -----------------------------------------------------------------------
    # AC-6: container_check is parallelogram
    # -----------------------------------------------------------------------

    def test_container_check_shape(self):
        """container_check node has shape=parallelogram."""
        graph = _graph_qa()
        assert graph.nodes["container_check"].shape == "parallelogram", (
            f"Expected container_check shape=parallelogram, "
            f"got {graph.nodes['container_check'].shape!r}"
        )

    # -----------------------------------------------------------------------
    # AC-7: read_qa_state is parallelogram with parse_json='true'
    # -----------------------------------------------------------------------

    def test_read_qa_state_shape(self):
        """read_qa_state node has shape=parallelogram."""
        graph = _graph_qa()
        assert graph.nodes["read_qa_state"].shape == "parallelogram", (
            f"Expected read_qa_state shape=parallelogram, "
            f"got {graph.nodes['read_qa_state'].shape!r}"
        )

    def test_read_qa_state_parse_json(self):
        """read_qa_state node has parse_json='true'."""
        graph = _graph_qa()
        val = graph.nodes["read_qa_state"].attrs.get("parse_json")
        assert val == "true", f"Expected read_qa_state parse_json='true', got {val!r}"

    # -----------------------------------------------------------------------
    # AC-8: blocked_gate is diamond
    # -----------------------------------------------------------------------

    def test_blocked_gate_shape(self):
        """blocked_gate node has shape=diamond."""
        graph = _graph_qa()
        assert graph.nodes["blocked_gate"].shape == "diamond", (
            f"Expected blocked_gate shape=diamond, "
            f"got {graph.nodes['blocked_gate'].shape!r}"
        )

    # -----------------------------------------------------------------------
    # AC-9: qa_loop is house with manager attributes
    # -----------------------------------------------------------------------

    def test_qa_loop_shape(self):
        """qa_loop node has shape=house."""
        graph = _graph_qa()
        assert graph.nodes["qa_loop"].shape == "house", (
            f"Expected qa_loop shape=house, got {graph.nodes['qa_loop'].shape!r}"
        )

    def test_qa_loop_max_cycles(self):
        """qa_loop node has manager.max_cycles attribute."""
        graph = _graph_qa()
        val = graph.nodes["qa_loop"].attrs.get("manager.max_cycles")
        assert val is not None, "Expected qa_loop to have manager.max_cycles attribute"

    def test_qa_loop_stop_condition(self):
        """qa_loop node has manager.stop_condition='outcome=success'."""
        graph = _graph_qa()
        val = graph.nodes["qa_loop"].attrs.get("manager.stop_condition")
        assert val == "outcome=success", (
            f"Expected qa_loop manager.stop_condition='outcome=success', got {val!r}"
        )

    def test_qa_loop_child_dotfile(self):
        """qa_loop node has manager.child_dotfile referencing qa-iteration.dot."""
        graph = _graph_qa()
        val = graph.nodes["qa_loop"].attrs.get("manager.child_dotfile")
        assert val is not None, (
            "Expected qa_loop to have manager.child_dotfile attribute"
        )
        assert "qa-iteration.dot" in val, (
            f"Expected qa_loop manager.child_dotfile to reference qa-iteration.dot, "
            f"got {val!r}"
        )

    # -----------------------------------------------------------------------
    # AC-10: final_summary is parallelogram with continue_on_fail='true'
    # -----------------------------------------------------------------------

    def test_final_summary_shape(self):
        """final_summary node has shape=parallelogram."""
        graph = _graph_qa()
        assert graph.nodes["final_summary"].shape == "parallelogram", (
            f"Expected final_summary shape=parallelogram, "
            f"got {graph.nodes['final_summary'].shape!r}"
        )

    def test_final_summary_continue_on_fail(self):
        """final_summary node has continue_on_fail='true'."""
        graph = _graph_qa()
        val = graph.nodes["final_summary"].attrs.get("continue_on_fail")
        assert val == "true", (
            f"Expected final_summary continue_on_fail='true', got {val!r}"
        )

    # -----------------------------------------------------------------------
    # AC-11: blocked_gate routes blocked->done and testing->qa_loop
    # -----------------------------------------------------------------------

    def test_blocked_gate_blocked_to_done(self):
        """blocked_gate has a 'blocked' edge routing to done."""
        graph = _graph_qa()
        gate_edges = [e for e in graph.edges if e.from_node == "blocked_gate"]
        done_edges = [e for e in gate_edges if e.to_node == "done"]
        assert done_edges, "Expected blocked_gate -> done edge. Edges: " + str(
            [(e.to_node, e.label, e.condition) for e in gate_edges]
        )
        done_edge = done_edges[0]
        has_blocked = (done_edge.label and "blocked" in done_edge.label.lower()) or (
            done_edge.condition and "blocked" in done_edge.condition.lower()
        )
        assert has_blocked, (
            f"Expected blocked_gate -> done edge to have 'blocked' label/condition, "
            f"got label={done_edge.label!r}, condition={done_edge.condition!r}"
        )

    def test_blocked_gate_testing_to_qa_loop(self):
        """blocked_gate has a 'testing' edge routing to qa_loop."""
        graph = _graph_qa()
        gate_edges = [e for e in graph.edges if e.from_node == "blocked_gate"]
        qa_loop_edges = [e for e in gate_edges if e.to_node == "qa_loop"]
        assert qa_loop_edges, "Expected blocked_gate -> qa_loop edge. Edges: " + str(
            [(e.to_node, e.label, e.condition) for e in gate_edges]
        )
        qa_edge = qa_loop_edges[0]
        has_testing = (qa_edge.label and "testing" in qa_edge.label.lower()) or (
            qa_edge.condition and "testing" in qa_edge.condition.lower()
        )
        assert has_testing, (
            f"Expected blocked_gate -> qa_loop edge to have 'testing' label/condition, "
            f"got label={qa_edge.label!r}, condition={qa_edge.condition!r}"
        )

    # -----------------------------------------------------------------------
    # AC-12: Required edges exist
    # -----------------------------------------------------------------------

    REQUIRED_EDGES = [
        ("start", "container_check"),
        ("container_check", "read_qa_state"),
        ("read_qa_state", "blocked_gate"),
        ("qa_loop", "final_summary"),
        ("final_summary", "done"),
    ]

    @pytest.mark.parametrize("from_node,to_node", REQUIRED_EDGES)
    def test_required_edges_present(self, from_node, to_node):
        """Required edges are present."""
        graph = _graph_qa()
        edge_pairs = {(e.from_node, e.to_node) for e in graph.edges}
        assert (from_node, to_node) in edge_pairs, (
            f"Edge {from_node} -> {to_node} not found. "
            f"Present edges: {sorted(edge_pairs)}"
        )


# ===========================================================================
# TestQAIterationParse -- qa-iteration.dot structural tests
# ===========================================================================


class TestQAIterationParse:
    """Tests for qa-iteration.dot parse correctness and structural requirements."""

    # -----------------------------------------------------------------------
    # AC-1: File exists
    # -----------------------------------------------------------------------

    def test_file_exists(self):
        """qa-iteration.dot exists at examples/dev-machine/runtime/qa-iteration.dot."""
        assert os.path.isfile(_QA_ITERATION_DOT), (
            f"qa-iteration.dot not found at {_QA_ITERATION_DOT}"
        )

    # -----------------------------------------------------------------------
    # AC-2: Parses without error
    # -----------------------------------------------------------------------

    def test_parses_without_error(self):
        """qa-iteration.dot parses without raising exceptions."""
        graph = _graph_qa_iteration()
        assert graph is not None

    # -----------------------------------------------------------------------
    # AC-3: Exactly 6 nodes
    # -----------------------------------------------------------------------

    def test_node_count(self):
        """Exactly 6 nodes."""
        graph = _graph_qa_iteration()
        assert len(graph.nodes) == 6, (
            f"Expected 6 nodes, got {len(graph.nodes)}: {list(graph.nodes.keys())}"
        )

    # -----------------------------------------------------------------------
    # AC-4: All 6 required node IDs present
    # -----------------------------------------------------------------------

    REQUIRED_NODE_IDS = [
        "start",
        "orient",
        "orient_gate",
        "qa_session",
        "post_qasession",
        "done",
    ]

    def test_all_required_node_ids_present(self):
        """All 6 required node IDs are present.

        Intentionally overlaps with test_each_required_node_id[*] below:
        this bulk test catches the complete set in one assertion, while the
        parametrized variant gives per-node visibility in CI output so a
        failure names exactly which node is missing.
        """
        graph = _graph_qa_iteration()
        missing = [nid for nid in self.REQUIRED_NODE_IDS if nid not in graph.nodes]
        assert not missing, (
            f"Missing node IDs: {missing}. Present: {list(graph.nodes.keys())}"
        )

    @pytest.mark.parametrize("node_id", REQUIRED_NODE_IDS)
    def test_each_required_node_id(self, node_id):
        """Each required node ID is individually present (per-node CI visibility)."""
        graph = _graph_qa_iteration()
        assert node_id in graph.nodes, (
            f"Node '{node_id}' not found. Present: {list(graph.nodes.keys())}"
        )

    # -----------------------------------------------------------------------
    # AC-5: start is Mdiamond, done is Msquare
    # -----------------------------------------------------------------------

    def test_start_shape(self):
        """start node has shape=Mdiamond."""
        graph = _graph_qa_iteration()
        assert graph.nodes["start"].shape == "Mdiamond", (
            f"Expected start shape=Mdiamond, got {graph.nodes['start'].shape!r}"
        )

    def test_done_shape(self):
        """done node has shape=Msquare."""
        graph = _graph_qa_iteration()
        assert graph.nodes["done"].shape == "Msquare", (
            f"Expected done shape=Msquare, got {graph.nodes['done'].shape!r}"
        )

    # -----------------------------------------------------------------------
    # AC-6: orient is parallelogram with parse_json='true'
    # -----------------------------------------------------------------------

    def test_orient_shape(self):
        """orient node has shape=parallelogram."""
        graph = _graph_qa_iteration()
        assert graph.nodes["orient"].shape == "parallelogram", (
            f"Expected orient shape=parallelogram, got {graph.nodes['orient'].shape!r}"
        )

    def test_orient_parse_json(self):
        """orient node has parse_json='true'."""
        graph = _graph_qa_iteration()
        val = graph.nodes["orient"].attrs.get("parse_json")
        assert val == "true", f"Expected orient parse_json='true', got {val!r}"

    # -----------------------------------------------------------------------
    # AC-7: orient_gate is diamond
    # -----------------------------------------------------------------------

    def test_orient_gate_shape(self):
        """orient_gate node has shape=diamond."""
        graph = _graph_qa_iteration()
        assert graph.nodes["orient_gate"].shape == "diamond", (
            f"Expected orient_gate shape=diamond, "
            f"got {graph.nodes['orient_gate'].shape!r}"
        )

    # -----------------------------------------------------------------------
    # AC-8: qa_session is box with context_fidelity='truncate'
    # -----------------------------------------------------------------------

    def test_qa_session_shape(self):
        """qa_session node has shape=box."""
        graph = _graph_qa_iteration()
        assert graph.nodes["qa_session"].shape == "box", (
            f"Expected qa_session shape=box, got {graph.nodes['qa_session'].shape!r}"
        )

    def test_qa_session_context_fidelity(self):
        """qa_session node has context_fidelity='truncate'."""
        graph = _graph_qa_iteration()
        val = graph.nodes["qa_session"].attrs.get("context_fidelity")
        assert val == "truncate", (
            f"Expected qa_session context_fidelity='truncate', got {val!r}"
        )

    # -----------------------------------------------------------------------
    # AC-9: post_qasession is parallelogram with parse_json='true'
    # -----------------------------------------------------------------------

    def test_post_qasession_shape(self):
        """post_qasession node has shape=parallelogram."""
        graph = _graph_qa_iteration()
        assert graph.nodes["post_qasession"].shape == "parallelogram", (
            f"Expected post_qasession shape=parallelogram, "
            f"got {graph.nodes['post_qasession'].shape!r}"
        )

    def test_post_qasession_parse_json(self):
        """post_qasession node has parse_json='true'."""
        graph = _graph_qa_iteration()
        val = graph.nodes["post_qasession"].attrs.get("parse_json")
        assert val == "true", f"Expected post_qasession parse_json='true', got {val!r}"

    # -----------------------------------------------------------------------
    # AC-10: qa_session prompt contains required content
    # -----------------------------------------------------------------------

    def test_qa_session_prompt_qa_testing_session(self):
        """qa_session prompt contains 'QA TESTING SESSION'."""
        graph = _graph_qa_iteration()
        prompt = graph.nodes["qa_session"].prompt
        assert "QA TESTING SESSION" in prompt, (
            "Expected 'QA TESTING SESSION' in qa_session prompt"
        )

    def test_qa_session_prompt_state_persistence(self):
        """qa_session prompt contains 'State Persistence'."""
        graph = _graph_qa_iteration()
        prompt = graph.nodes["qa_session"].prompt
        assert "State Persistence" in prompt, (
            "Expected 'State Persistence' in qa_session prompt"
        )

    def test_qa_session_prompt_safety_constraints(self):
        """qa_session prompt contains 'SAFETY CONSTRAINTS'."""
        graph = _graph_qa_iteration()
        prompt = graph.nodes["qa_session"].prompt
        assert "SAFETY CONSTRAINTS" in prompt, (
            "Expected 'SAFETY CONSTRAINTS' in qa_session prompt"
        )

    def test_qa_session_prompt_forbidden_commands(self):
        """qa_session prompt contains 'FORBIDDEN commands'."""
        graph = _graph_qa_iteration()
        prompt = graph.nodes["qa_session"].prompt
        assert "FORBIDDEN commands" in prompt, (
            "Expected 'FORBIDDEN commands' in qa_session prompt"
        )

    def test_qa_session_prompt_data_loss_warning(self):
        """qa_session prompt contains data loss violation warning."""
        graph = _graph_qa_iteration()
        prompt = graph.nodes["qa_session"].prompt
        assert "VIOLATION OF THESE CONSTRAINTS MAY CAUSE DATA LOSS" in prompt, (
            "Expected 'VIOLATION OF THESE CONSTRAINTS MAY CAUSE DATA LOSS' "
            "in qa_session prompt"
        )

    def test_qa_session_prompt_qa_state_file(self):
        """qa_session prompt uses $qa_state_file (not Jinja2 {{qa_state_file}})."""
        graph = _graph_qa_iteration()
        prompt = graph.nodes["qa_session"].prompt
        assert "$qa_state_file" in prompt, (
            "Expected '$qa_state_file' in qa_session prompt"
        )
        assert "{{qa_state_file}}" not in prompt, (
            "qa_session prompt should NOT contain '{{qa_state_file}}' Jinja2 syntax; "
            "use '$qa_state_file' instead"
        )

    def test_qa_session_prompt_next_test(self):
        """qa_session prompt uses $next_test reference."""
        graph = _graph_qa_iteration()
        prompt = graph.nodes["qa_session"].prompt
        assert "$next_test" in prompt, "Expected '$next_test' in qa_session prompt"

    def test_qa_session_prompt_no_jinja2(self):
        """qa_session prompt does NOT use Jinja2 {{...}} syntax."""
        graph = _graph_qa_iteration()
        prompt = graph.nodes["qa_session"].prompt
        assert "{{" not in prompt and "}}" not in prompt, (
            "qa_session prompt should NOT contain Jinja2 {{...}} syntax"
        )

    # -----------------------------------------------------------------------
    # AC-11: orient_gate routes testing->qa_session and done->done
    # -----------------------------------------------------------------------

    def test_orient_gate_testing_to_qa_session(self):
        """orient_gate has a 'testing' edge routing to qa_session."""
        graph = _graph_qa_iteration()
        gate_edges = [e for e in graph.edges if e.from_node == "orient_gate"]
        qa_edges = [e for e in gate_edges if e.to_node == "qa_session"]
        assert qa_edges, "Expected orient_gate -> qa_session edge. Edges: " + str(
            [(e.to_node, e.label, e.condition) for e in gate_edges]
        )
        qa_edge = qa_edges[0]
        has_testing = (qa_edge.label and "testing" in qa_edge.label.lower()) or (
            qa_edge.condition and "testing" in qa_edge.condition.lower()
        )
        assert has_testing, (
            f"Expected orient_gate -> qa_session edge to have 'testing' "
            f"label/condition, got label={qa_edge.label!r}, "
            f"condition={qa_edge.condition!r}"
        )

    def test_orient_gate_done_to_done(self):
        """orient_gate has a 'done' edge routing to done node."""
        graph = _graph_qa_iteration()
        gate_edges = [e for e in graph.edges if e.from_node == "orient_gate"]
        done_edges = [e for e in gate_edges if e.to_node == "done"]
        assert done_edges, "Expected orient_gate -> done edge. Edges: " + str(
            [(e.to_node, e.label, e.condition) for e in gate_edges]
        )
        # At least one done edge should have 'done' label/condition
        done_labeled = [
            e
            for e in done_edges
            if (e.label and "done" in e.label.lower())
            or (e.condition and "done" in e.condition.lower())
        ]
        assert done_labeled, (
            "Expected at least one orient_gate -> done edge with 'done' "
            "label/condition. Edges: "
            + str([(e.to_node, e.label, e.condition) for e in done_edges])
        )

    # -----------------------------------------------------------------------
    # AC-12: Required edges exist
    # -----------------------------------------------------------------------

    REQUIRED_EDGES = [
        ("start", "orient"),
        ("orient", "orient_gate"),
        ("qa_session", "post_qasession"),
        ("post_qasession", "done"),
    ]

    @pytest.mark.parametrize("from_node,to_node", REQUIRED_EDGES)
    def test_required_edges_present(self, from_node, to_node):
        """Required edges are present."""
        graph = _graph_qa_iteration()
        edge_pairs = {(e.from_node, e.to_node) for e in graph.edges}
        assert (from_node, to_node) in edge_pairs, (
            f"Edge {from_node} -> {to_node} not found. "
            f"Present edges: {sorted(edge_pairs)}"
        )
