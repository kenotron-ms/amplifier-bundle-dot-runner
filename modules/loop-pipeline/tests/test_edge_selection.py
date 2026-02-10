"""Tests for the 5-step edge selection algorithm.

Spec coverage: ESEL-001–010, Section 3.3.
"""

from amplifier_module_loop_pipeline.context import PipelineContext
from amplifier_module_loop_pipeline.edge_selection import select_edge
from amplifier_module_loop_pipeline.graph import Edge, Graph, Node
from amplifier_module_loop_pipeline.outcome import Outcome, StageStatus


def _make_graph(edges: list[Edge]) -> Graph:
    """Build a minimal graph with the given edges."""
    nodes: dict[str, Node] = {}
    for e in edges:
        if e.from_node not in nodes:
            nodes[e.from_node] = Node(id=e.from_node)
        if e.to_node not in nodes:
            nodes[e.to_node] = Node(id=e.to_node)
    return Graph(name="test", nodes=nodes, edges=edges)


def test_condition_matching_takes_priority():
    """Step 1: condition-matching edges first (ESEL-002)."""
    edges = [
        Edge("A", "B", condition="outcome=fail"),
        Edge("A", "C", label="success"),
    ]
    graph = _make_graph(edges)
    outcome = Outcome(status=StageStatus.FAIL)
    ctx = PipelineContext()
    selected = select_edge("A", outcome, ctx, graph)
    assert selected is not None
    assert selected.to_node == "B"


def test_condition_no_match_falls_through():
    """Condition edges that don't match are skipped."""
    edges = [
        Edge("A", "B", condition="outcome=fail"),
        Edge("A", "C"),
    ]
    graph = _make_graph(edges)
    outcome = Outcome(status=StageStatus.SUCCESS)
    ctx = PipelineContext()
    selected = select_edge("A", outcome, ctx, graph)
    assert selected is not None
    assert selected.to_node == "C"


def test_preferred_label_match():
    """Step 2: preferred_label match (ESEL-003)."""
    edges = [Edge("A", "B", label="tests_pass"), Edge("A", "C", label="tests_fail")]
    graph = _make_graph(edges)
    outcome = Outcome(status=StageStatus.SUCCESS, preferred_label="tests_pass")
    selected = select_edge("A", outcome, PipelineContext(), graph)
    assert selected is not None
    assert selected.to_node == "B"


def test_label_normalization():
    """Labels normalized: lowercase, strip accelerators (ESEL-004)."""
    edges = [Edge("A", "B", label="[Y] Tests Pass")]
    graph = _make_graph(edges)
    outcome = Outcome(status=StageStatus.SUCCESS, preferred_label="tests pass")
    selected = select_edge("A", outcome, PipelineContext(), graph)
    assert selected is not None
    assert selected.to_node == "B"


def test_suggested_next_ids():
    """Step 3: suggested_next_ids match (ESEL-005)."""
    edges = [Edge("A", "B"), Edge("A", "C")]
    graph = _make_graph(edges)
    outcome = Outcome(status=StageStatus.SUCCESS, suggested_next_ids=["C"])
    selected = select_edge("A", outcome, PipelineContext(), graph)
    assert selected is not None
    assert selected.to_node == "C"


def test_weight_tiebreak():
    """Step 4: higher weight wins (ESEL-006)."""
    edges = [Edge("A", "B", weight=1), Edge("A", "C", weight=5)]
    graph = _make_graph(edges)
    outcome = Outcome(status=StageStatus.SUCCESS)
    selected = select_edge("A", outcome, PipelineContext(), graph)
    assert selected is not None
    assert selected.to_node == "C"


def test_lexical_tiebreak():
    """Step 5: equal weight -> lexical order (ESEL-007)."""
    edges = [Edge("A", "zebra"), Edge("A", "alpha")]
    graph = _make_graph(edges)
    outcome = Outcome(status=StageStatus.SUCCESS)
    selected = select_edge("A", outcome, PipelineContext(), graph)
    assert selected is not None
    assert selected.to_node == "alpha"


def test_no_edges_returns_none():
    """No outgoing edges returns None."""
    graph = Graph(name="test", nodes={"A": Node(id="A")}, edges=[])
    selected = select_edge(
        "A", Outcome(status=StageStatus.SUCCESS), PipelineContext(), graph
    )
    assert selected is None


def test_condition_edges_sorted_by_weight():
    """Multiple matching conditions use weight tiebreak."""
    edges = [
        Edge("A", "B", condition="outcome=success", weight=1),
        Edge("A", "C", condition="outcome=success", weight=10),
    ]
    graph = _make_graph(edges)
    outcome = Outcome(status=StageStatus.SUCCESS)
    selected = select_edge("A", outcome, PipelineContext(), graph)
    assert selected is not None
    assert selected.to_node == "C"


# --- Additional accelerator normalization patterns ---


def test_label_normalization_strip_accelerator_paren():
    """Strip accelerator prefix like 'A) ' from labels (ESEL-004)."""
    edges = [Edge("A", "B", label="A) Fix Code")]
    graph = _make_graph(edges)
    outcome = Outcome(status=StageStatus.SUCCESS, preferred_label="fix code")
    selected = select_edge("A", outcome, PipelineContext(), graph)
    assert selected is not None
    assert selected.to_node == "B"


def test_label_normalization_strip_accelerator_dash():
    """Strip accelerator prefix like 'Y - ' from labels (ESEL-004)."""
    edges = [Edge("A", "B", label="Y - Accept")]
    graph = _make_graph(edges)
    outcome = Outcome(status=StageStatus.SUCCESS, preferred_label="accept")
    selected = select_edge("A", outcome, PipelineContext(), graph)
    assert selected is not None
    assert selected.to_node == "B"


# --- Suggested next IDs edge cases ---


def test_suggested_next_ids_first_match_wins():
    """First match in suggested_next_ids is selected."""
    edges = [Edge("A", "B"), Edge("A", "C")]
    graph = _make_graph(edges)
    outcome = Outcome(status=StageStatus.SUCCESS, suggested_next_ids=["C", "B"])
    selected = select_edge("A", outcome, PipelineContext(), graph)
    assert selected is not None
    assert selected.to_node == "C"  # C listed first in suggested_next_ids


def test_suggested_next_ids_no_match_falls_through():
    """If suggested IDs don't match any edge target, fall through to weight."""
    edges = [Edge("A", "B")]
    graph = _make_graph(edges)
    outcome = Outcome(status=StageStatus.SUCCESS, suggested_next_ids=["X", "Y"])
    selected = select_edge("A", outcome, PipelineContext(), graph)
    assert selected is not None
    assert selected.to_node == "B"


# --- Determinism ---


def test_deterministic_with_same_inputs():
    """Same inputs always produce same output (ESEL-001)."""
    edges = [
        Edge("A", "B", weight=3),
        Edge("A", "C", weight=3),
        Edge("A", "D", weight=1),
    ]
    graph = _make_graph(edges)
    outcome = Outcome(status=StageStatus.SUCCESS)
    ctx = PipelineContext()
    results = [select_edge("A", outcome, ctx, graph) for _ in range(20)]
    first = results[0]
    assert first is not None
    assert all(r is not None and r.to_node == first.to_node for r in results)


# --- Fallback when all conditions fail ---


def test_all_conditions_false_fallback_to_weight():
    """If all edges have conditions and none match, fallback picks by weight."""
    edges = [
        Edge("A", "B", condition="outcome=fail", weight=1),
        Edge("A", "C", condition="outcome=fail", weight=5),
    ]
    graph = _make_graph(edges)
    outcome = Outcome(status=StageStatus.SUCCESS)
    selected = select_edge("A", outcome, PipelineContext(), graph)
    assert selected is not None
    assert selected.to_node == "C"


# --- Priority order tests ---


def test_condition_beats_preferred_label():
    """Condition match (step 1) beats preferred label (step 2)."""
    edges = [
        Edge("A", "B", condition="outcome=success"),
        Edge("A", "C", label="go_here"),
    ]
    graph = _make_graph(edges)
    outcome = Outcome(status=StageStatus.SUCCESS, preferred_label="go_here")
    selected = select_edge("A", outcome, PipelineContext(), graph)
    assert selected is not None
    assert selected.to_node == "B"


def test_preferred_label_beats_suggested_ids():
    """Preferred label (step 2) beats suggested IDs (step 3)."""
    edges = [Edge("A", "B", label="go_here"), Edge("A", "C")]
    graph = _make_graph(edges)
    outcome = Outcome(
        status=StageStatus.SUCCESS,
        preferred_label="go_here",
        suggested_next_ids=["C"],
    )
    selected = select_edge("A", outcome, PipelineContext(), graph)
    assert selected is not None
    assert selected.to_node == "B"
