"""Graph data model for Attractor pipelines.

Defines Node, Edge, and Graph dataclasses that represent a parsed DOT
digraph. These are the core data structures used throughout the pipeline
engine.

Spec coverage: DOT-001–017, NATTR-001–017, EDGE-001–006
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Node:
    """A node in the pipeline graph.

    Attributes map to spec Section 2.6 (Node Attributes).
    The shape determines the default handler type via the
    shape-to-handler-type mapping (spec Section 2.8).
    """

    id: str
    label: str = ""
    shape: str = "box"
    type: str = ""
    prompt: str = ""
    attrs: dict[str, Any] = field(default_factory=dict)
    handler_type: str = ""  # Resolved from type or shape

    def __post_init__(self) -> None:
        if not self.label:
            self.label = self.id


@dataclass
class Edge:
    """A directed edge in the pipeline graph.

    Attributes map to spec Section 2.7 (Edge Attributes).
    """

    from_node: str
    to_node: str
    label: str = ""
    condition: str = ""
    weight: int = 0
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Graph:
    """A parsed pipeline graph.

    Contains nodes, edges, and graph-level attributes from
    spec Section 2.5 (Graph-Level Attributes).
    """

    name: str
    nodes: dict[str, Node]
    edges: list[Edge]
    goal: str = ""
    default_max_retry: int = 50
    model_stylesheet: str = ""
    graph_attrs: dict[str, str] = field(default_factory=dict)

    def outgoing_edges(self, node_id: str) -> list[Edge]:
        """Return all edges originating from the given node."""
        return [e for e in self.edges if e.from_node == node_id]

    def incoming_edges(self, node_id: str) -> list[Edge]:
        """Return all edges targeting the given node."""
        return [e for e in self.edges if e.to_node == node_id]
