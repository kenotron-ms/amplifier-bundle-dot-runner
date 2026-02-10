"""Graph validation and lint rules for Attractor pipelines.

Validates parsed Graph models against the rules defined in
spec Section 7 (Validation and Linting). Produces Diagnostic objects
with severity ERROR (blocks execution) or WARNING (informational).

Spec coverage: LINT-001–018
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .graph import Graph

# Shape-to-handler-type mapping (spec Section 2.8)
SHAPE_TO_HANDLER: dict[str, str] = {
    "Mdiamond": "start",
    "Msquare": "exit",
    "box": "codergen",
    "hexagon": "wait.human",
    "diamond": "conditional",
    "component": "parallel",
    "tripleoctagon": "parallel.fan_in",
    "parallelogram": "tool",
    "house": "stack.manager_loop",
}

# Shapes that map to LLM/codergen handler
_LLM_SHAPES = {"box"}


@dataclass
class Diagnostic:
    """A single validation diagnostic.

    Spec Section 7.1: rule, severity, message, optional node_id/edge/fix.
    """

    rule: str
    severity: str  # "ERROR", "WARNING", "INFO"
    message: str
    node_id: str = ""
    edge: tuple[str, str] | None = None
    fix: str = ""


class ValidationError(Exception):
    """Raised by validate_or_raise when ERROR diagnostics are found."""

    def __init__(self, diagnostics: list[Diagnostic]) -> None:
        self.diagnostics = diagnostics
        messages = [d.message for d in diagnostics if d.severity == "ERROR"]
        super().__init__(f"Validation failed: {'; '.join(messages)}")


def validate(graph: Graph) -> list[Diagnostic]:
    """Run all built-in lint rules against a graph.

    Returns a list of Diagnostic objects. ERROR-severity diagnostics
    indicate the pipeline will not execute.

    Spec Section 7.3: validate API.
    """
    diags: list[Diagnostic] = []
    _check_start_node(graph, diags)
    _check_terminal_node(graph, diags)
    _check_edge_targets(graph, diags)
    _check_start_no_incoming(graph, diags)
    _check_exit_no_outgoing(graph, diags)
    _check_reachability(graph, diags)
    _check_goal_gate_has_retry(graph, diags)
    _check_prompt_on_llm_nodes(graph, diags)
    return diags


def validate_or_raise(graph: Graph) -> list[Diagnostic]:
    """Validate and raise ValidationError if any ERROR diagnostics found.

    Returns non-error diagnostics (warnings/info) on success.

    Spec Section 7.3: validate_or_raise API.
    """
    diags = validate(graph)
    errors = [d for d in diags if d.severity == "ERROR"]
    if errors:
        raise ValidationError(errors)
    return diags


# --- Individual lint rules ---


def _check_start_node(graph: Graph, diags: list[Diagnostic]) -> None:
    """LINT: start_node — exactly one start node (shape=Mdiamond)."""
    start_nodes = [n for n in graph.nodes.values() if n.shape == "Mdiamond"]
    if len(start_nodes) == 0:
        diags.append(
            Diagnostic(
                rule="start_node",
                severity="ERROR",
                message="Pipeline must have exactly one start node (shape=Mdiamond)",
                fix="Add a node with shape=Mdiamond",
            )
        )
    elif len(start_nodes) > 1:
        ids = ", ".join(n.id for n in start_nodes)
        diags.append(
            Diagnostic(
                rule="start_node",
                severity="ERROR",
                message=f"Pipeline has {len(start_nodes)} start nodes ({ids}); exactly one is required",
                fix="Remove extra start nodes so only one has shape=Mdiamond",
            )
        )


def _check_terminal_node(graph: Graph, diags: list[Diagnostic]) -> None:
    """LINT: terminal_node — at least one exit node (shape=Msquare)."""
    exit_nodes = [n for n in graph.nodes.values() if n.shape == "Msquare"]
    if len(exit_nodes) == 0:
        diags.append(
            Diagnostic(
                rule="terminal_node",
                severity="ERROR",
                message="Pipeline must have at least one exit node (shape=Msquare)",
                fix="Add a node with shape=Msquare",
            )
        )


def _check_edge_targets(graph: Graph, diags: list[Diagnostic]) -> None:
    """LINT: edge_target_exists — all edge endpoints must reference existing nodes."""
    node_ids = set(graph.nodes.keys())
    for edge in graph.edges:
        if edge.from_node not in node_ids:
            diags.append(
                Diagnostic(
                    rule="edge_target_exists",
                    severity="ERROR",
                    message=f"Edge source '{edge.from_node}' does not reference an existing node",
                    edge=(edge.from_node, edge.to_node),
                    fix=f"Add a node declaration for '{edge.from_node}'",
                )
            )
        if edge.to_node not in node_ids:
            diags.append(
                Diagnostic(
                    rule="edge_target_exists",
                    severity="ERROR",
                    message=f"Edge target '{edge.to_node}' does not reference an existing node",
                    edge=(edge.from_node, edge.to_node),
                    fix=f"Add a node declaration for '{edge.to_node}'",
                )
            )


def _check_start_no_incoming(graph: Graph, diags: list[Diagnostic]) -> None:
    """LINT: start_no_incoming — start node must have no incoming edges."""
    start_nodes = [n for n in graph.nodes.values() if n.shape == "Mdiamond"]
    for start in start_nodes:
        incoming = graph.incoming_edges(start.id)
        if incoming:
            sources = ", ".join(e.from_node for e in incoming)
            diags.append(
                Diagnostic(
                    rule="start_no_incoming",
                    severity="ERROR",
                    message=f"Start node '{start.id}' has incoming edges from: {sources}",
                    node_id=start.id,
                    fix="Remove edges targeting the start node",
                )
            )


def _check_exit_no_outgoing(graph: Graph, diags: list[Diagnostic]) -> None:
    """LINT: exit_no_outgoing — exit node must have no outgoing edges."""
    exit_nodes = [n for n in graph.nodes.values() if n.shape == "Msquare"]
    for exit_node in exit_nodes:
        outgoing = graph.outgoing_edges(exit_node.id)
        if outgoing:
            targets = ", ".join(e.to_node for e in outgoing)
            diags.append(
                Diagnostic(
                    rule="exit_no_outgoing",
                    severity="ERROR",
                    message=f"Exit node '{exit_node.id}' has outgoing edges to: {targets}",
                    node_id=exit_node.id,
                    fix="Remove edges originating from the exit node",
                )
            )


def _check_reachability(graph: Graph, diags: list[Diagnostic]) -> None:
    """LINT: reachability — all nodes reachable from start via BFS."""
    start_nodes = [n for n in graph.nodes.values() if n.shape == "Mdiamond"]
    if not start_nodes:
        return  # start_node rule already flagged

    start = start_nodes[0]
    visited: set[str] = set()
    queue: deque[str] = deque([start.id])

    while queue:
        node_id = queue.popleft()
        if node_id in visited:
            continue
        visited.add(node_id)
        for edge in graph.outgoing_edges(node_id):
            if edge.to_node in graph.nodes:
                queue.append(edge.to_node)

    unreachable = set(graph.nodes.keys()) - visited
    for node_id in sorted(unreachable):
        diags.append(
            Diagnostic(
                rule="reachability",
                severity="ERROR",
                message=f"Node '{node_id}' is not reachable from the start node",
                node_id=node_id,
                fix=f"Add an edge path from start to '{node_id}'",
            )
        )


def _check_goal_gate_has_retry(graph: Graph, diags: list[Diagnostic]) -> None:
    """LINT: goal_gate_has_retry — goal gates should have retry targets."""
    for node in graph.nodes.values():
        if node.attrs.get("goal_gate") is True:
            has_retry = bool(
                node.attrs.get("retry_target")
                or node.attrs.get("fallback_retry_target")
                or graph.graph_attrs.get("retry_target")
            )
            if not has_retry:
                diags.append(
                    Diagnostic(
                        rule="goal_gate_has_retry",
                        severity="WARNING",
                        message=f"Node '{node.id}' has goal_gate=true but no retry_target",
                        node_id=node.id,
                        fix="Add retry_target or fallback_retry_target attribute",
                    )
                )


def _check_prompt_on_llm_nodes(graph: Graph, diags: list[Diagnostic]) -> None:
    """LINT: prompt_on_llm_nodes — codergen nodes should have prompt or meaningful label."""
    for node in graph.nodes.values():
        # Determine if this is an LLM/codergen node
        handler = node.type or SHAPE_TO_HANDLER.get(node.shape, "codergen")
        if handler != "codergen":
            continue

        has_prompt = bool(node.prompt)
        # label == id means no explicit label was set
        has_explicit_label = node.label != node.id

        if not has_prompt and not has_explicit_label:
            diags.append(
                Diagnostic(
                    rule="prompt_on_llm_nodes",
                    severity="WARNING",
                    message=f"LLM node '{node.id}' has no prompt and no explicit label",
                    node_id=node.id,
                    fix="Add a prompt attribute or a descriptive label",
                )
            )
