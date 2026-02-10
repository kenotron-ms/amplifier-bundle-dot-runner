"""Built-in transforms for pipeline graph preprocessing.

Transforms modify the pipeline graph after parsing and before execution.
They run in a defined order: variable expansion first, then stylesheet
application.

Spec coverage: XFORM-001–006, Section 9.

Built-in transforms:
    expand_variables  — Replace $goal in node prompts with the graph goal.
    apply_transforms  — Run all built-in transforms in order.
"""

from __future__ import annotations

from .context import PipelineContext
from .graph import Graph
from .stylesheet import apply_stylesheet, parse_stylesheet


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
    goal_value = context.get("graph.goal") or graph.goal
    if not goal_value:
        return graph

    for node in graph.nodes.values():
        if node.prompt and "$goal" in node.prompt:
            node.prompt = node.prompt.replace("$goal", str(goal_value))

    return graph


def apply_transforms(graph: Graph, context: PipelineContext) -> Graph:
    """Run all built-in transforms on the graph.

    Order:
    1. Variable expansion (``$goal`` → goal value).
    2. Stylesheet application (CSS-like model config rules).

    Args:
        graph: The pipeline graph to transform (modified in place).
        context: The pipeline context with runtime values.

    Returns:
        The same graph, fully transformed.
    """
    # 1. Variable expansion
    expand_variables(graph, context)

    # 2. Stylesheet application
    if graph.model_stylesheet:
        rules = parse_stylesheet(graph.model_stylesheet)
        apply_stylesheet(graph, rules)

    return graph
