"""Tests for Phase 3: smoke-test.dot -- Pre-Flight Validator.

Verifies that examples/dev-machine/runtime/smoke-test.dot is correct per spec:
- 9 nodes: start (Mdiamond), preflight, check_files, check_dot_validity,
  check_docker, check_state, check_robustness, smoke_summary, done (Msquare)
- 7 parallelogram nodes (all except start and done)
- 8 edges (linear chain)
- Chain order: start->preflight->check_files->check_dot_validity->check_docker
              ->check_state->check_robustness->smoke_summary->done
- Five check nodes have continue_on_fail='true':
  check_files, check_dot_validity, check_docker, check_state, check_robustness
- preflight does NOT have continue_on_fail (hard-fail)
- smoke_summary does NOT have continue_on_fail (hard-fail)
- check_dot_validity tool_command references .dot files

Spec coverage: smoke-test.dot Phase 3 requirements.
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
_SMOKE_TEST_DOT = os.path.join(
    _EXAMPLES_DIR, "dev-machine", "runtime", "smoke-test.dot"
)


def _load() -> str:
    with open(_SMOKE_TEST_DOT) as f:
        return f.read()


def _graph():
    return parse_dot(_load())


# ===========================================================================
# TestSmokeTestParse -- smoke-test.dot structural tests
# ===========================================================================


class TestSmokeTestParse:
    """Tests for smoke-test.dot parse correctness and structural requirements."""

    # -----------------------------------------------------------------------
    # AC-1: File exists
    # -----------------------------------------------------------------------

    def test_file_exists(self):
        """smoke-test.dot exists at examples/dev-machine/runtime/smoke-test.dot."""
        assert os.path.isfile(_SMOKE_TEST_DOT), (
            f"smoke-test.dot not found at {_SMOKE_TEST_DOT}"
        )

    # -----------------------------------------------------------------------
    # AC-2: Parses without error
    # -----------------------------------------------------------------------

    def test_parses_without_error(self):
        """smoke-test.dot parses without raising exceptions."""
        graph = _graph()
        assert graph is not None

    # -----------------------------------------------------------------------
    # AC-3: Exactly 9 nodes
    # -----------------------------------------------------------------------

    def test_node_count(self):
        """Exactly 9 nodes."""
        graph = _graph()
        assert len(graph.nodes) == 9, (
            f"Expected 9 nodes, got {len(graph.nodes)}: {list(graph.nodes.keys())}"
        )

    # -----------------------------------------------------------------------
    # AC-4: Exactly 7 parallelogram nodes
    # -----------------------------------------------------------------------

    def test_parallelogram_count(self):
        """Exactly 7 parallelogram (tool) nodes."""
        graph = _graph()
        count = sum(1 for n in graph.nodes.values() if n.shape == "parallelogram")
        assert count == 7, f"Expected 7 parallelogram nodes, got {count}"

    # -----------------------------------------------------------------------
    # AC-5: All 9 required node IDs present
    # -----------------------------------------------------------------------

    REQUIRED_NODE_IDS = [
        "start",
        "preflight",
        "check_files",
        "check_dot_validity",
        "check_docker",
        "check_state",
        "check_robustness",
        "smoke_summary",
        "done",
    ]

    def test_all_required_node_ids_present(self):
        """All 9 required node IDs are present.

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
    # AC-5 (shapes): start is Mdiamond, done is Msquare
    # -----------------------------------------------------------------------

    def test_start_shape(self):
        """start node has shape=Mdiamond."""
        graph = _graph()
        assert graph.nodes["start"].shape == "Mdiamond", (
            f"Expected start shape=Mdiamond, got {graph.nodes['start'].shape!r}"
        )

    def test_done_shape(self):
        """done node has shape=Msquare."""
        graph = _graph()
        assert graph.nodes["done"].shape == "Msquare", (
            f"Expected done shape=Msquare, got {graph.nodes['done'].shape!r}"
        )

    # -----------------------------------------------------------------------
    # AC-6: Five check nodes have continue_on_fail='true'
    # -----------------------------------------------------------------------

    CHECK_NODES = [
        "check_files",
        "check_dot_validity",
        "check_docker",
        "check_state",
        "check_robustness",
    ]

    @pytest.mark.parametrize("node_id", CHECK_NODES)
    def test_check_nodes_continue_on_fail(self, node_id):
        """Each of the 5 check nodes has continue_on_fail='true'."""
        graph = _graph()
        val = graph.nodes[node_id].attrs.get("continue_on_fail")
        assert val == "true", f"Expected {node_id} continue_on_fail='true', got {val!r}"

    @pytest.mark.parametrize("node_id", CHECK_NODES)
    def test_check_nodes_are_parallelogram(self, node_id):
        """Each of the 5 check nodes has shape=parallelogram."""
        graph = _graph()
        assert graph.nodes[node_id].shape == "parallelogram", (
            f"Expected {node_id} shape=parallelogram, got {graph.nodes[node_id].shape!r}"
        )

    # -----------------------------------------------------------------------
    # AC-7: preflight does NOT have continue_on_fail (hard-fail)
    # -----------------------------------------------------------------------

    def test_preflight_no_continue_on_fail(self):
        """preflight does NOT have continue_on_fail (it is a hard-fail step)."""
        graph = _graph()
        val = graph.nodes["preflight"].attrs.get("continue_on_fail")
        assert val != "true", (
            f"preflight should NOT have continue_on_fail='true', got {val!r}"
        )

    def test_preflight_shape(self):
        """preflight has shape=parallelogram."""
        graph = _graph()
        assert graph.nodes["preflight"].shape == "parallelogram", (
            f"Expected preflight shape=parallelogram, "
            f"got {graph.nodes['preflight'].shape!r}"
        )

    # -----------------------------------------------------------------------
    # AC-8: smoke_summary does NOT have continue_on_fail (hard-fail)
    # -----------------------------------------------------------------------

    def test_smoke_summary_no_continue_on_fail(self):
        """smoke_summary does NOT have continue_on_fail (it is a hard-fail step)."""
        graph = _graph()
        val = graph.nodes["smoke_summary"].attrs.get("continue_on_fail")
        assert val != "true", (
            f"smoke_summary should NOT have continue_on_fail='true', got {val!r}"
        )

    def test_smoke_summary_shape(self):
        """smoke_summary has shape=parallelogram."""
        graph = _graph()
        assert graph.nodes["smoke_summary"].shape == "parallelogram", (
            f"Expected smoke_summary shape=parallelogram, "
            f"got {graph.nodes['smoke_summary'].shape!r}"
        )

    # -----------------------------------------------------------------------
    # AC-9: check_dot_validity tool_command references .dot files
    # -----------------------------------------------------------------------

    def test_check_dot_validity_tool_command_references_dot_files(self):
        """check_dot_validity tool_command references .dot files."""
        graph = _graph()
        cmd = graph.nodes["check_dot_validity"].attrs.get("tool_command", "")
        assert ".dot" in cmd, (
            f"Expected check_dot_validity tool_command to reference .dot files, "
            f"got: {cmd!r}"
        )

    def test_check_dot_validity_shape(self):
        """check_dot_validity has shape=parallelogram."""
        graph = _graph()
        assert graph.nodes["check_dot_validity"].shape == "parallelogram", (
            f"Expected check_dot_validity shape=parallelogram, "
            f"got {graph.nodes['check_dot_validity'].shape!r}"
        )

    # -----------------------------------------------------------------------
    # AC-10: Exactly 8 edges (linear chain)
    # -----------------------------------------------------------------------

    def test_edge_count(self):
        """Exactly 8 edges."""
        graph = _graph()
        assert len(graph.edges) == 8, (
            f"Expected 8 edges, got {len(graph.edges)}: "
            + "\n".join(f"  {e.from_node} -> {e.to_node}" for e in graph.edges)
        )

    # -----------------------------------------------------------------------
    # AC-11: Chain order verified
    # -----------------------------------------------------------------------

    EXPECTED_CHAIN = [
        ("start", "preflight"),
        ("preflight", "check_files"),
        ("check_files", "check_dot_validity"),
        ("check_dot_validity", "check_docker"),
        ("check_docker", "check_state"),
        ("check_state", "check_robustness"),
        ("check_robustness", "smoke_summary"),
        ("smoke_summary", "done"),
    ]

    def test_linear_chain_edges(self):
        """All 8 edges form the expected linear chain."""
        graph = _graph()
        edge_pairs = {(e.from_node, e.to_node) for e in graph.edges}
        missing = [pair for pair in self.EXPECTED_CHAIN if pair not in edge_pairs]
        assert not missing, (
            f"Missing edges: {missing}. Present edges: {sorted(edge_pairs)}"
        )

    @pytest.mark.parametrize("from_node,to_node", EXPECTED_CHAIN)
    def test_each_chain_edge(self, from_node, to_node):
        """Each link in the chain is individually present (per-edge CI visibility)."""
        graph = _graph()
        edge_pairs = {(e.from_node, e.to_node) for e in graph.edges}
        assert (from_node, to_node) in edge_pairs, (
            f"Edge {from_node} -> {to_node} not found. "
            f"Present edges: {sorted(edge_pairs)}"
        )
