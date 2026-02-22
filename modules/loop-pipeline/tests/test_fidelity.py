"""Tests for context fidelity modes.

Spec coverage: FID-001–010, Section 5.4.
"""

from amplifier_module_loop_pipeline.context import PipelineContext
from amplifier_module_loop_pipeline.fidelity import (
    VALID_FIDELITY_MODES,
    build_preamble,
    resolve_fidelity,
    resolve_thread_key,
)
from amplifier_module_loop_pipeline.graph import Edge, Graph, Node
from amplifier_module_loop_pipeline.outcome import Outcome, StageStatus


def _make_graph(**kwargs) -> Graph:
    return Graph(
        name="test",
        nodes={"start": Node(id="start", shape="Mdiamond")},
        edges=[],
        **kwargs,
    )


# --- resolve_fidelity ---


def test_resolve_fidelity_default_is_compact():
    """Default fidelity is 'compact' when nothing is set."""
    node = Node(id="plan", prompt="Plan")
    graph = _make_graph()
    assert resolve_fidelity(node, None, graph) == "compact"


def test_resolve_fidelity_graph_default():
    """Graph default_fidelity overrides the system default."""
    node = Node(id="plan", prompt="Plan")
    graph = _make_graph(graph_attrs={"default_fidelity": "full"})
    assert resolve_fidelity(node, None, graph) == "full"


def test_resolve_fidelity_node_overrides_graph():
    """Node fidelity overrides graph default_fidelity."""
    node = Node(id="plan", prompt="Plan", attrs={"fidelity": "truncate"})
    graph = _make_graph(graph_attrs={"default_fidelity": "full"})
    assert resolve_fidelity(node, None, graph) == "truncate"


def test_resolve_fidelity_edge_overrides_node():
    """Edge fidelity overrides node fidelity."""
    node = Node(id="plan", prompt="Plan", attrs={"fidelity": "truncate"})
    edge = Edge(from_node="start", to_node="plan", attrs={"fidelity": "summary:high"})
    graph = _make_graph()
    assert resolve_fidelity(node, edge, graph) == "summary:high"


def test_resolve_fidelity_edge_overrides_graph():
    """Edge fidelity overrides graph default."""
    node = Node(id="plan", prompt="Plan")
    edge = Edge(from_node="start", to_node="plan", attrs={"fidelity": "full"})
    graph = _make_graph(graph_attrs={"default_fidelity": "compact"})
    assert resolve_fidelity(node, edge, graph) == "full"


def test_resolve_fidelity_all_summary_modes():
    """All summary:* modes are valid."""
    for mode in ("summary:low", "summary:medium", "summary:high"):
        node = Node(id="n", prompt="X", attrs={"fidelity": mode})
        assert resolve_fidelity(node, None, _make_graph()) == mode


def test_valid_fidelity_modes_constant():
    """VALID_FIDELITY_MODES contains all 6 modes."""
    assert VALID_FIDELITY_MODES == {
        "full",
        "truncate",
        "compact",
        "summary:low",
        "summary:medium",
        "summary:high",
    }


# --- resolve_thread_key ---


def test_resolve_thread_key_fallback_is_previous_node():
    """Default thread key falls back to previous node ID."""
    node = Node(id="impl", prompt="Build")
    graph = _make_graph()
    assert resolve_thread_key(node, None, graph, previous_node_id="plan") == "plan"


def test_resolve_thread_key_node_thread_id():
    """Node thread_id attribute is highest priority."""
    node = Node(id="impl", prompt="Build", attrs={"thread_id": "shared-thread"})
    edge = Edge(from_node="plan", to_node="impl", attrs={"thread_id": "edge-thread"})
    graph = _make_graph(graph_attrs={"default_thread_id": "graph-thread"})
    assert (
        resolve_thread_key(node, edge, graph, previous_node_id="plan")
        == "shared-thread"
    )


def test_resolve_thread_key_edge_thread_id():
    """Edge thread_id is used when node doesn't have one."""
    node = Node(id="impl", prompt="Build")
    edge = Edge(from_node="plan", to_node="impl", attrs={"thread_id": "edge-thread"})
    graph = _make_graph()
    assert (
        resolve_thread_key(node, edge, graph, previous_node_id="plan") == "edge-thread"
    )


def test_resolve_thread_key_graph_default():
    """Graph-level default_thread_id is used as fallback."""
    node = Node(id="impl", prompt="Build")
    graph = _make_graph(graph_attrs={"default_thread_id": "graph-thread"})
    assert (
        resolve_thread_key(node, None, graph, previous_node_id="plan") == "graph-thread"
    )


def test_resolve_thread_key_no_previous_node():
    """When no previous node and no thread_id set, falls back to node ID."""
    node = Node(id="impl", prompt="Build")
    graph = _make_graph()
    assert resolve_thread_key(node, None, graph, previous_node_id=None) == "impl"


# --- build_preamble ---


def test_build_preamble_truncate():
    """Truncate mode: minimal preamble with goal and run ID only."""
    ctx = PipelineContext()
    ctx.set("graph.goal", "Build auth feature")
    ctx.set("internal.run_id", "run-42")
    preamble = build_preamble("truncate", ctx, {})
    assert "Build auth feature" in preamble
    assert "run-42" in preamble


def test_build_preamble_compact():
    """Compact mode: structured bullet-point summary."""
    ctx = PipelineContext()
    ctx.set("graph.goal", "Add caching")
    completed = {
        "plan": Outcome(status=StageStatus.SUCCESS, notes="Plan completed"),
        "implement": Outcome(status=StageStatus.PARTIAL_SUCCESS, notes="Mostly done"),
    }
    preamble = build_preamble("compact", ctx, completed)
    assert "Add caching" in preamble
    assert "plan" in preamble
    assert "implement" in preamble
    assert "success" in preamble.lower()


def test_build_preamble_summary_low():
    """summary:low produces a brief summary within ~600 token budget."""
    ctx = PipelineContext()
    ctx.set("graph.goal", "Fix bugs")
    completed = {
        "plan": Outcome(status=StageStatus.SUCCESS, notes="Done"),
    }
    preamble = build_preamble("summary:low", ctx, completed)
    assert "Fix bugs" in preamble
    # ~600 tokens ≈ ~2400 chars max; should be well within
    assert len(preamble) < 3000


def test_build_preamble_summary_medium():
    """summary:medium includes more detail than low."""
    ctx = PipelineContext()
    ctx.set("graph.goal", "Refactor DB")
    completed = {
        "plan": Outcome(status=StageStatus.SUCCESS, notes="Plan done"),
        "implement": Outcome(status=StageStatus.SUCCESS, notes="Code written"),
    }
    preamble = build_preamble("summary:medium", ctx, completed)
    assert "Refactor DB" in preamble
    assert "plan" in preamble
    assert "implement" in preamble


def test_build_preamble_summary_high():
    """summary:high includes comprehensive detail."""
    ctx = PipelineContext()
    ctx.set("graph.goal", "Deploy service")
    completed = {
        "plan": Outcome(status=StageStatus.SUCCESS, notes="Plan ready"),
        "impl": Outcome(status=StageStatus.SUCCESS, notes="Code done"),
        "test": Outcome(
            status=StageStatus.PARTIAL_SUCCESS,
            notes="3/4 pass",
            failure_reason="Test X failed",
        ),
    }
    preamble = build_preamble("summary:high", ctx, completed)
    assert "Deploy service" in preamble
    assert "test" in preamble
    # High should include failure details
    assert "Test X failed" in preamble


def test_build_preamble_full_returns_empty():
    """Full fidelity doesn't need a preamble (session reused)."""
    ctx = PipelineContext()
    preamble = build_preamble("full", ctx, {})
    assert preamble == ""


def test_build_preamble_empty_completed_nodes():
    """Preamble works with no completed nodes."""
    ctx = PipelineContext()
    ctx.set("graph.goal", "Start fresh")
    preamble = build_preamble("compact", ctx, {})
    assert "Start fresh" in preamble


def test_build_preamble_includes_context_values():
    """Compact preamble includes key context values."""
    ctx = PipelineContext()
    ctx.set("graph.goal", "My goal")
    ctx.set("context.architecture", "microservices")
    completed = {}
    preamble = build_preamble("compact", ctx, completed)
    assert "My goal" in preamble
    # context.* values should be included
    assert "architecture" in preamble


# --- last_response in preamble (built-in context keys) ---


def test_compact_preamble_includes_last_response():
    """Compact preamble includes last_response from context when present."""
    ctx = PipelineContext()
    ctx.set("graph.goal", "Explain monads")
    ctx.set("last_response", "A monad is a design pattern in functional programming")
    ctx.set("last_stage", "draft")
    completed = {
        "draft": Outcome(status=StageStatus.SUCCESS, notes="Stage completed: draft"),
    }
    preamble = build_preamble("compact", ctx, completed)
    assert "Explain monads" in preamble
    assert "draft" in preamble
    # The actual LLM response must appear in the preamble
    assert "A monad is a design pattern" in preamble


def test_compact_preamble_without_last_response():
    """Compact preamble still works when last_response is absent."""
    ctx = PipelineContext()
    ctx.set("graph.goal", "Some goal")
    completed = {
        "plan": Outcome(status=StageStatus.SUCCESS, notes="Done"),
    }
    preamble = build_preamble("compact", ctx, completed)
    assert "Some goal" in preamble
    # No crash, no "Last response" section
    assert "Last response" not in preamble


def test_summary_medium_includes_last_response():
    """summary:medium preamble includes last_response from context."""
    ctx = PipelineContext()
    ctx.set("graph.goal", "Build API")
    ctx.set("last_response", "Here is the API design with three endpoints")
    completed = {
        "design": Outcome(status=StageStatus.SUCCESS, notes="Design done"),
    }
    preamble = build_preamble("summary:medium", ctx, completed)
    assert "Here is the API design" in preamble


def test_summary_high_includes_last_response():
    """summary:high preamble includes last_response from context."""
    ctx = PipelineContext()
    ctx.set("graph.goal", "Deploy service")
    ctx.set("last_response", "Deployment plan: step 1 provision, step 2 deploy")
    completed = {
        "plan": Outcome(status=StageStatus.SUCCESS, notes="Plan ready"),
    }
    preamble = build_preamble("summary:high", ctx, completed)
    assert "Deployment plan: step 1 provision" in preamble


def test_compact_preamble_last_response_after_stages_before_context():
    """last_response appears after completed stages and before context.* values."""
    ctx = PipelineContext()
    ctx.set("graph.goal", "Test ordering")
    ctx.set("last_response", "The draft output text here")
    ctx.set("context.style", "formal")
    completed = {
        "draft": Outcome(status=StageStatus.SUCCESS, notes="Done"),
    }
    preamble = build_preamble("compact", ctx, completed)
    # All three sections should be present
    stages_pos = preamble.index("Completed stages")
    response_pos = preamble.index("Last response")
    context_pos = preamble.index("Context values")
    # Order: stages < last_response < context values
    assert stages_pos < response_pos < context_pos
