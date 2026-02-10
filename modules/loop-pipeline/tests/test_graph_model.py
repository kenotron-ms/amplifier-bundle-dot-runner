"""Tests for the graph data model."""

from amplifier_module_loop_pipeline.graph import Edge, Graph, Node


def test_node_defaults():
    """Node should have sensible defaults per spec Section 2.6."""
    node = Node(id="step1")
    assert node.id == "step1"
    assert node.label == "step1"  # label defaults to node ID
    assert node.shape == "box"
    assert node.type == ""
    assert node.prompt == ""
    assert node.attrs == {}
    assert node.handler_type == ""


def test_node_with_attributes():
    """Node should accept all spec attributes."""
    node = Node(
        id="implement",
        label="Implement Feature",
        shape="box",
        prompt="Build the feature for $goal",
        attrs={"max_retries": 3, "goal_gate": True, "timeout": 900000},
    )
    assert node.label == "Implement Feature"
    assert node.prompt == "Build the feature for $goal"
    assert node.attrs["max_retries"] == 3
    assert node.attrs["goal_gate"] is True


def test_edge_defaults():
    """Edge should have sensible defaults per spec Section 2.7."""
    edge = Edge(from_node="A", to_node="B")
    assert edge.from_node == "A"
    assert edge.to_node == "B"
    assert edge.label == ""
    assert edge.condition == ""
    assert edge.weight == 0
    assert edge.attrs == {}


def test_edge_with_attributes():
    """Edge should accept condition and weight."""
    edge = Edge(
        from_node="gate",
        to_node="exit",
        label="Yes",
        condition="outcome=success",
        weight=5,
    )
    assert edge.label == "Yes"
    assert edge.condition == "outcome=success"
    assert edge.weight == 5


def test_graph_construction():
    """Graph should hold nodes, edges, and graph-level attributes."""
    nodes = {
        "start": Node(id="start", shape="Mdiamond"),
        "work": Node(id="work", label="Do Work", prompt="Do the work"),
        "exit": Node(id="exit", shape="Msquare"),
    }
    edges = [
        Edge(from_node="start", to_node="work"),
        Edge(from_node="work", to_node="exit"),
    ]
    graph = Graph(name="test_pipeline", nodes=nodes, edges=edges)
    assert graph.name == "test_pipeline"
    assert len(graph.nodes) == 3
    assert len(graph.edges) == 2
    assert graph.goal == ""
    assert graph.default_max_retry == 50
    assert graph.graph_attrs == {}


def test_graph_with_goal():
    """Graph should store the goal attribute."""
    graph = Graph(
        name="pipeline",
        nodes={},
        edges=[],
        goal="Build the feature",
        default_max_retry=5,
    )
    assert graph.goal == "Build the feature"
    assert graph.default_max_retry == 5


def test_graph_outgoing_edges():
    """Graph.outgoing_edges should return edges from a given node."""
    edges = [
        Edge(from_node="A", to_node="B"),
        Edge(from_node="A", to_node="C"),
        Edge(from_node="B", to_node="C"),
    ]
    graph = Graph(name="test", nodes={}, edges=edges)
    outgoing = graph.outgoing_edges("A")
    assert len(outgoing) == 2
    assert {e.to_node for e in outgoing} == {"B", "C"}


def test_graph_incoming_edges():
    """Graph.incoming_edges should return edges targeting a given node."""
    edges = [
        Edge(from_node="A", to_node="C"),
        Edge(from_node="B", to_node="C"),
        Edge(from_node="C", to_node="D"),
    ]
    graph = Graph(name="test", nodes={}, edges=edges)
    incoming = graph.incoming_edges("C")
    assert len(incoming) == 2
    assert {e.from_node for e in incoming} == {"A", "B"}


def test_node_label_defaults_to_id():
    """When label is not provided, it should default to the node ID."""
    node = Node(id="my_step")
    assert node.label == "my_step"


def test_node_explicit_label_overrides_default():
    """An explicit label should be used instead of the ID."""
    node = Node(id="my_step", label="My Step")
    assert node.label == "My Step"
