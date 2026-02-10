"""DOT file interop validation tests (Phase 7, Task 7.4).

Verifies that DOT files from the Attractor spec parse correctly and
produce valid graphs. Each example DOT graph from the spec is copied
verbatim into a test fixture and tested for:
- Successful parsing
- Correct node and edge counts
- Correct graph-level attributes
- Validation passes (no ERROR diagnostics)
- Node attributes are preserved
- Stylesheet application (where applicable)

Spec DOT examples tested:
1. spec_simple_linear.dot  — Section 2.13 "Simple linear workflow"
2. spec_branching.dot      — Section 2.13 "Branching workflow with conditions"
3. spec_human_gate.dot     — Section 2.13 "Human gate"
4. spec_stylesheet.dot     — Section 8.6 "Stylesheet example"
5. spec_smoke_test.dot     — Section 11.13 "Integration Smoke Test"
"""

import os

import pytest

from amplifier_module_loop_pipeline.dot_parser import parse_dot
from amplifier_module_loop_pipeline.validation import validate, validate_or_raise


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def _load_fixture(name: str) -> str:
    """Load a DOT fixture file by name."""
    path = os.path.join(FIXTURES_DIR, name)
    with open(path) as f:
        return f.read()


# ===========================================================================
# Spec Example 1: Simple Linear Workflow (Section 2.13)
# ===========================================================================


class TestSpecSimpleLinear:
    """Interop tests for the spec's simple linear workflow example."""

    def _load(self) -> str:
        return _load_fixture("spec_simple_linear.dot")

    def test_parses_without_error(self):
        """The spec example parses without raising exceptions."""
        graph = parse_dot(self._load())
        assert graph is not None

    def test_graph_name(self):
        """Graph name matches the DOT digraph name."""
        graph = parse_dot(self._load())
        assert graph.name == "Simple"

    def test_node_count(self):
        """Correct number of nodes: start, run_tests, report, exit."""
        graph = parse_dot(self._load())
        assert len(graph.nodes) == 4

    def test_edge_count(self):
        """Correct number of edges: start->run_tests, run_tests->report, report->exit."""
        graph = parse_dot(self._load())
        assert len(graph.edges) == 3

    def test_goal_attribute(self):
        """Graph-level goal attribute is extracted."""
        graph = parse_dot(self._load())
        assert graph.goal == "Run tests and report"

    def test_node_shapes(self):
        """Start and exit nodes have correct shapes."""
        graph = parse_dot(self._load())
        assert graph.nodes["start"].shape == "Mdiamond"
        assert graph.nodes["exit"].shape == "Msquare"

    def test_node_labels(self):
        """Nodes with explicit labels have them set."""
        graph = parse_dot(self._load())
        assert graph.nodes["start"].label == "Start"
        assert graph.nodes["exit"].label == "Exit"
        assert graph.nodes["run_tests"].label == "Run Tests"
        assert graph.nodes["report"].label == "Report"

    def test_node_prompts(self):
        """Codergen nodes have their prompt attributes."""
        graph = parse_dot(self._load())
        assert "test suite" in graph.nodes["run_tests"].prompt
        assert "Summarize" in graph.nodes["report"].prompt

    def test_chained_edges(self):
        """Chained edge (start -> run_tests -> report -> exit) produces 3 edges."""
        graph = parse_dot(self._load())
        edge_pairs = [(e.from_node, e.to_node) for e in graph.edges]
        assert ("start", "run_tests") in edge_pairs
        assert ("run_tests", "report") in edge_pairs
        assert ("report", "exit") in edge_pairs

    def test_validates_no_errors(self):
        """Validation produces no ERROR diagnostics."""
        graph = parse_dot(self._load())
        diags = validate(graph)
        errors = [d for d in diags if d.severity == "ERROR"]
        assert len(errors) == 0

    def test_validate_or_raise_succeeds(self):
        """validate_or_raise does not raise."""
        graph = parse_dot(self._load())
        validate_or_raise(graph)  # Should not raise


# ===========================================================================
# Spec Example 2: Branching Workflow (Section 2.13)
# ===========================================================================


class TestSpecBranching:
    """Interop tests for the spec's branching workflow example."""

    def _load(self) -> str:
        return _load_fixture("spec_branching.dot")

    def test_parses_without_error(self):
        graph = parse_dot(self._load())
        assert graph is not None

    def test_graph_name(self):
        graph = parse_dot(self._load())
        assert graph.name == "Branch"

    def test_node_count(self):
        """6 nodes: start, exit, plan, implement, validate, gate."""
        graph = parse_dot(self._load())
        assert len(graph.nodes) == 6

    def test_edge_count(self):
        """6 edges: chain of 5 + gate->exit + gate->implement."""
        graph = parse_dot(self._load())
        assert len(graph.edges) == 6

    def test_goal_attribute(self):
        graph = parse_dot(self._load())
        assert graph.goal == "Implement and validate a feature"

    def test_node_defaults_applied(self):
        """node [shape=box] default applies to non-special nodes."""
        graph = parse_dot(self._load())
        assert graph.nodes["plan"].shape == "box"
        assert graph.nodes["implement"].shape == "box"
        assert graph.nodes["validate"].shape == "box"

    def test_node_default_timeout(self):
        """node [timeout=\"900s\"] default applies — parsed as 900000ms."""
        graph = parse_dot(self._load())
        # Timeout is stored as ms in attrs (via duration parsing)
        plan_timeout = graph.nodes["plan"].attrs.get("timeout")
        assert plan_timeout == 900_000  # 900s = 900000ms

    def test_diamond_shape(self):
        """Gate node has shape=diamond."""
        graph = parse_dot(self._load())
        assert graph.nodes["gate"].shape == "diamond"

    def test_conditional_edges(self):
        """Gate outgoing edges have conditions."""
        graph = parse_dot(self._load())
        gate_edges = [e for e in graph.edges if e.from_node == "gate"]
        assert len(gate_edges) == 2

        success_edge = next(e for e in gate_edges if e.to_node == "exit")
        assert success_edge.condition == "outcome=success"
        assert success_edge.label == "Yes"

        fail_edge = next(e for e in gate_edges if e.to_node == "implement")
        assert fail_edge.condition == "outcome!=success"
        assert fail_edge.label == "No"

    def test_validates_no_errors(self):
        graph = parse_dot(self._load())
        diags = validate(graph)
        errors = [d for d in diags if d.severity == "ERROR"]
        assert len(errors) == 0


# ===========================================================================
# Spec Example 3: Human Gate (Section 2.13)
# ===========================================================================


class TestSpecHumanGate:
    """Interop tests for the spec's human gate example."""

    def _load(self) -> str:
        return _load_fixture("spec_human_gate.dot")

    def test_parses_without_error(self):
        graph = parse_dot(self._load())
        assert graph is not None

    def test_graph_name(self):
        graph = parse_dot(self._load())
        assert graph.name == "Review"

    def test_node_count(self):
        """5 nodes: start, exit, review_gate, ship_it, fixes."""
        graph = parse_dot(self._load())
        assert len(graph.nodes) == 5

    def test_edge_count(self):
        """5 edges: start->review_gate, review_gate->ship_it,
        review_gate->fixes, ship_it->exit, fixes->review_gate."""
        graph = parse_dot(self._load())
        assert len(graph.edges) == 5

    def test_human_gate_shape(self):
        """review_gate has shape=hexagon."""
        graph = parse_dot(self._load())
        assert graph.nodes["review_gate"].shape == "hexagon"

    def test_human_gate_type(self):
        """review_gate has explicit type='wait.human'."""
        graph = parse_dot(self._load())
        assert graph.nodes["review_gate"].type == "wait.human"

    def test_human_gate_label(self):
        """review_gate has label='Review Changes'."""
        graph = parse_dot(self._load())
        assert graph.nodes["review_gate"].label == "Review Changes"

    def test_accelerator_labels_on_edges(self):
        """Human gate edges have accelerator key labels."""
        graph = parse_dot(self._load())
        gate_edges = [e for e in graph.edges if e.from_node == "review_gate"]
        assert len(gate_edges) == 2

        labels = sorted(e.label for e in gate_edges)
        assert "[A] Approve" in labels
        assert "[F] Fix" in labels

    def test_cycle_back_to_gate(self):
        """fixes -> review_gate creates a cycle (valid for human gates)."""
        graph = parse_dot(self._load())
        cycle_edge = [
            e
            for e in graph.edges
            if e.from_node == "fixes" and e.to_node == "review_gate"
        ]
        assert len(cycle_edge) == 1

    def test_validates_no_errors(self):
        graph = parse_dot(self._load())
        diags = validate(graph)
        errors = [d for d in diags if d.severity == "ERROR"]
        assert len(errors) == 0


# ===========================================================================
# Spec Example 4: Stylesheet (Section 8.6)
# ===========================================================================


class TestSpecStylesheet:
    """Interop tests for the spec's stylesheet example."""

    def _load(self) -> str:
        return _load_fixture("spec_stylesheet.dot")

    def test_parses_without_error(self):
        graph = parse_dot(self._load())
        assert graph is not None

    def test_graph_name(self):
        graph = parse_dot(self._load())
        assert graph.name == "Pipeline"

    def test_node_count(self):
        """5 nodes: start, exit, plan, implement, critical_review."""
        graph = parse_dot(self._load())
        assert len(graph.nodes) == 5

    def test_edge_count(self):
        """4 edges: start->plan->implement->critical_review->exit."""
        graph = parse_dot(self._load())
        assert len(graph.edges) == 4

    def test_goal_attribute(self):
        graph = parse_dot(self._load())
        assert graph.goal == "Implement feature X"

    def test_model_stylesheet_extracted(self):
        """model_stylesheet attribute is extracted from graph."""
        graph = parse_dot(self._load())
        assert graph.model_stylesheet != ""
        assert "claude-sonnet" in graph.model_stylesheet
        assert "claude-opus" in graph.model_stylesheet
        assert "gpt-5.2" in graph.model_stylesheet

    def test_class_attribute_on_nodes(self):
        """Nodes have class attributes from the DOT declaration."""
        graph = parse_dot(self._load())
        assert graph.nodes["plan"].attrs.get("class") == "planning"
        assert graph.nodes["implement"].attrs.get("class") == "code"
        assert graph.nodes["critical_review"].attrs.get("class") == "code"

    def test_validates_no_errors(self):
        graph = parse_dot(self._load())
        diags = validate(graph)
        errors = [d for d in diags if d.severity == "ERROR"]
        assert len(errors) == 0

    def test_stylesheet_contains_selectors(self):
        """Stylesheet includes universal, class, and ID selectors."""
        graph = parse_dot(self._load())
        ss = graph.model_stylesheet
        assert "*" in ss  # Universal selector
        assert ".code" in ss  # Class selector
        assert "#critical_review" in ss  # ID selector


# ===========================================================================
# Spec Example 5: Integration Smoke Test (Section 11.13)
# ===========================================================================


class TestSpecSmokeTestDOT:
    """Interop tests for the spec's Section 11.13 smoke test DOT."""

    def _load(self) -> str:
        return _load_fixture("spec_smoke_test.dot")

    def test_parses_without_error(self):
        graph = parse_dot(self._load())
        assert graph is not None

    def test_graph_name(self):
        graph = parse_dot(self._load())
        assert graph.name == "test_pipeline"

    def test_node_count(self):
        """5 nodes: start, plan, implement, review, done."""
        graph = parse_dot(self._load())
        assert len(graph.nodes) == 5

    def test_edge_count(self):
        """6 edges: start->plan, plan->implement,
        implement->review (success), implement->plan (fail),
        review->done (success), review->implement (fail)."""
        graph = parse_dot(self._load())
        assert len(graph.edges) == 6

    def test_goal_attribute(self):
        graph = parse_dot(self._load())
        assert graph.goal == "Create a hello world Python script"

    def test_goal_gate_on_implement(self):
        """implement node has goal_gate=true."""
        graph = parse_dot(self._load())
        assert graph.nodes["implement"].attrs.get("goal_gate") is True

    def test_prompts_with_goal_variable(self):
        """plan node's prompt contains $goal template variable."""
        graph = parse_dot(self._load())
        assert "$goal" in graph.nodes["plan"].prompt

    def test_conditional_edges(self):
        """Conditional edges have correct conditions."""
        graph = parse_dot(self._load())

        impl_edges = [e for e in graph.edges if e.from_node == "implement"]
        assert len(impl_edges) == 2
        success_edge = next(e for e in impl_edges if e.to_node == "review")
        assert success_edge.condition == "outcome=success"
        fail_edge = next(e for e in impl_edges if e.to_node == "plan")
        assert fail_edge.condition == "outcome=fail"
        assert fail_edge.label == "Retry"

        review_edges = [e for e in graph.edges if e.from_node == "review"]
        assert len(review_edges) == 2
        done_edge = next(e for e in review_edges if e.to_node == "done")
        assert done_edge.condition == "outcome=success"
        fix_edge = next(e for e in review_edges if e.to_node == "implement")
        assert fix_edge.condition == "outcome=fail"
        assert fix_edge.label == "Fix"

    def test_validates_no_errors(self):
        graph = parse_dot(self._load())
        diags = validate(graph)
        errors = [d for d in diags if d.severity == "ERROR"]
        assert len(errors) == 0

    def test_validate_or_raise_succeeds(self):
        graph = parse_dot(self._load())
        validate_or_raise(graph)  # Should not raise


# ===========================================================================
# Cross-fixture validation: all spec examples parse and validate
# ===========================================================================


SPEC_FIXTURES = [
    "spec_simple_linear.dot",
    "spec_branching.dot",
    "spec_human_gate.dot",
    "spec_stylesheet.dot",
    "spec_smoke_test.dot",
]


@pytest.mark.parametrize("fixture_name", SPEC_FIXTURES)
def test_all_spec_fixtures_parse(fixture_name):
    """Every spec fixture parses without error."""
    dot = _load_fixture(fixture_name)
    graph = parse_dot(dot)
    assert graph is not None
    assert len(graph.nodes) > 0
    assert len(graph.edges) > 0


@pytest.mark.parametrize("fixture_name", SPEC_FIXTURES)
def test_all_spec_fixtures_validate(fixture_name):
    """Every spec fixture validates without ERROR diagnostics."""
    dot = _load_fixture(fixture_name)
    graph = parse_dot(dot)
    diags = validate(graph)
    errors = [d for d in diags if d.severity == "ERROR"]
    assert len(errors) == 0, f"{fixture_name} has validation errors: " + "; ".join(
        d.message for d in errors
    )


@pytest.mark.parametrize("fixture_name", SPEC_FIXTURES)
def test_all_spec_fixtures_have_start_and_exit(fixture_name):
    """Every spec fixture has exactly one start node and at least one exit node."""
    dot = _load_fixture(fixture_name)
    graph = parse_dot(dot)

    start_nodes = [n for n in graph.nodes.values() if n.shape == "Mdiamond"]
    exit_nodes = [n for n in graph.nodes.values() if n.shape == "Msquare"]

    assert len(start_nodes) == 1, (
        f"{fixture_name}: expected 1 start node, got {len(start_nodes)}"
    )
    assert len(exit_nodes) >= 1, (
        f"{fixture_name}: expected ≥1 exit node, got {len(exit_nodes)}"
    )
