"""Built-in transforms for pipeline graph preprocessing.

Transforms modify the pipeline graph after parsing and before execution.
They run in a defined order: variable expansion first, then stylesheet
application, then any custom transforms.

Spec coverage: XFORM-001-006, Section 9.2

Built-in transforms:
    expand_variables  — Replace $goal in node prompts with the graph goal.
    apply_transforms  — Run all built-in transforms in order.

M-20: Formal Transform protocol for custom transforms.
L-17: Shared expand_goal_variable utility (single source of truth).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .context import PipelineContext
from .graph import Graph
from .stylesheet import apply_stylesheet, parse_stylesheet


# ---------------------------------------------------------------------------
# L-17: Shared variable expansion utility
# ---------------------------------------------------------------------------


def expand_goal_variable(text: str, graph_goal: str, context_goal: str | Any) -> str:
    """Replace ``$goal`` in *text* with the goal value.

    Resolution order:
    1. *context_goal* (from ``context.get("graph.goal")``).
    2. *graph_goal* (the graph-level goal attribute).

    If neither is truthy, *text* is returned unchanged.

    This is the **single** location for ``$goal`` expansion (L-17).
    """
    goal_value = context_goal or graph_goal
    if not goal_value:
        return text
    return text.replace("$goal", str(goal_value))


# ---------------------------------------------------------------------------
# M-20: Transform protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Transform(Protocol):
    """Interface for graph transforms.

    Spec Section 9.2: Transform protocol.

    Implementors must provide an ``apply`` method that takes a Graph
    and returns the (possibly modified) Graph.
    """

    def apply(self, graph: Graph) -> Graph: ...


# ---------------------------------------------------------------------------
# Built-in transforms
# ---------------------------------------------------------------------------


def expand_variables(graph: Graph, context: PipelineContext) -> Graph:
    """Replace ``$goal`` in node prompts with the goal value.

    Resolution order for the goal value:
    1. ``context.get("graph.goal")`` — set during engine initialization.
    2. ``graph.goal`` — the graph-level goal attribute (fallback).

    Only ``$goal`` is expanded. Other ``$``-prefixed tokens are left
    unchanged (spec Section 9.2: no arbitrary expression expansion).

    Args:
        graph: The pipeline graph to transform (modified in place).
        context: The pipeline context with runtime values.

    Returns:
        The same graph, with ``$goal`` replaced in node prompts.
    """
    context_goal = context.get("graph.goal") or ""
    graph_goal = graph.goal or ""

    for node in graph.nodes.values():
        if node.prompt and "$goal" in node.prompt:
            node.prompt = expand_goal_variable(node.prompt, graph_goal, context_goal)

    return graph


def apply_transforms(
    graph: Graph,
    context: PipelineContext,
    *,
    extra_transforms: list[Transform] | None = None,
) -> Graph:
    """Run all built-in transforms on the graph, then any custom transforms.

    Order:
    1. Variable expansion (``$goal`` → goal value).
    2. Stylesheet application (CSS-like model config rules).
    3. Custom transforms (in order provided).

    Args:
        graph: The pipeline graph to transform (modified in place).
        context: The pipeline context with runtime values.
        extra_transforms: Optional list of additional Transform objects
            to run after the built-in transforms (M-20).

    Returns:
        The same graph, fully transformed.
    """
    # 1. Variable expansion
    expand_variables(graph, context)

    # 2. Stylesheet application
    if graph.model_stylesheet:
        rules = parse_stylesheet(graph.model_stylesheet)
        apply_stylesheet(graph, rules)

    # 3. Custom transforms (M-20)
    if extra_transforms:
        for transform in extra_transforms:
            graph = transform.apply(graph)

    return graph
