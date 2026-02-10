"""Tests for transforms (variable expansion, stylesheet application).

Transforms modify the pipeline graph after parsing and before execution.
Built-in transforms: variable expansion ($goal), stylesheet application.

Spec coverage: XFORM-001–006, Section 9.
"""

import pytest

from amplifier_module_loop_pipeline.context import PipelineContext
from amplifier_module_loop_pipeline.dot_parser import parse_dot
from amplifier_module_loop_pipeline.engine import PipelineEngine
from amplifier_module_loop_pipeline.graph import Edge, Graph, Node
from amplifier_module_loop_pipeline.handlers import HandlerRegistry
from amplifier_module_loop_pipeline.outcome import StageStatus
from amplifier_module_loop_pipeline.transforms import (
    apply_transforms,
    expand_variables,
)
from amplifier_module_loop_pipeline.validation import validate_or_raise


def _make_graph(**overrides) -> Graph:
    """Helper to build a simple graph with defaults."""
    defaults = {
        "name": "test",
        "nodes": {
            "start": Node(id="start", shape="Mdiamond"),
            "plan": Node(id="plan", prompt="Plan the work for $goal"),
            "implement": Node(id="implement", prompt="Build $goal"),
            "exit": Node(id="exit", shape="Msquare"),
        },
        "edges": [
            Edge(from_node="start", to_node="plan"),
            Edge(from_node="plan", to_node="implement"),
            Edge(from_node="implement", to_node="exit"),
        ],
        "goal": "authentication system",
    }
    defaults.update(overrides)
    return Graph(**defaults)


# --- expand_variables ---


class TestExpandVariables:
    """XFORM-001–002: Variable expansion replaces $goal in prompts."""

    def test_replaces_goal_in_prompt(self):
        """$goal in node prompt is replaced with graph goal value."""
        graph = _make_graph()
        context = PipelineContext()
        context.set("graph.goal", "authentication system")
        result = expand_variables(graph, context)
        assert result.nodes["plan"].prompt == "Plan the work for authentication system"
        assert result.nodes["implement"].prompt == "Build authentication system"

    def test_no_goal_anywhere_leaves_placeholder(self):
        """When neither context nor graph has a goal, $goal is unchanged."""
        graph = _make_graph(goal="")
        context = PipelineContext()
        result = expand_variables(graph, context)
        assert "$goal" in result.nodes["plan"].prompt

    def test_no_dollar_goal_unchanged(self):
        """Prompts without $goal are not modified."""
        graph = Graph(
            name="test",
            nodes={
                "start": Node(id="start", shape="Mdiamond"),
                "step": Node(id="step", prompt="Do some work"),
                "exit": Node(id="exit", shape="Msquare"),
            },
            edges=[
                Edge(from_node="start", to_node="step"),
                Edge(from_node="step", to_node="exit"),
            ],
        )
        context = PipelineContext()
        context.set("graph.goal", "anything")
        result = expand_variables(graph, context)
        assert result.nodes["step"].prompt == "Do some work"

    def test_empty_prompt_unchanged(self):
        """Nodes with empty prompts are not affected."""
        graph = Graph(
            name="test",
            nodes={
                "start": Node(id="start", shape="Mdiamond"),
                "step": Node(id="step", prompt=""),
                "exit": Node(id="exit", shape="Msquare"),
            },
            edges=[
                Edge(from_node="start", to_node="step"),
                Edge(from_node="step", to_node="exit"),
            ],
        )
        context = PipelineContext()
        context.set("graph.goal", "test")
        result = expand_variables(graph, context)
        assert result.nodes["step"].prompt == ""

    def test_multiple_goal_occurrences(self):
        """Multiple $goal in one prompt are all replaced."""
        graph = Graph(
            name="test",
            nodes={
                "start": Node(id="start", shape="Mdiamond"),
                "step": Node(
                    id="step",
                    prompt="First do $goal, then verify $goal is done",
                ),
                "exit": Node(id="exit", shape="Msquare"),
            },
            edges=[
                Edge(from_node="start", to_node="step"),
                Edge(from_node="step", to_node="exit"),
            ],
        )
        context = PipelineContext()
        context.set("graph.goal", "auth")
        result = expand_variables(graph, context)
        assert result.nodes["step"].prompt == "First do auth, then verify auth is done"

    def test_only_goal_variable_expanded(self):
        """Only $goal is expanded; other $-prefixed words are not."""
        graph = Graph(
            name="test",
            nodes={
                "start": Node(id="start", shape="Mdiamond"),
                "step": Node(id="step", prompt="Work on $goal using $other"),
                "exit": Node(id="exit", shape="Msquare"),
            },
            edges=[
                Edge(from_node="start", to_node="step"),
                Edge(from_node="step", to_node="exit"),
            ],
        )
        context = PipelineContext()
        context.set("graph.goal", "auth")
        result = expand_variables(graph, context)
        assert result.nodes["step"].prompt == "Work on auth using $other"

    def test_uses_graph_goal_when_context_empty(self):
        """Falls back to graph.goal attribute when context has no graph.goal."""
        graph = _make_graph(goal="fallback goal")
        context = PipelineContext()
        # Don't set graph.goal in context — should use graph.goal attribute
        result = expand_variables(graph, context)
        assert result.nodes["plan"].prompt == "Plan the work for fallback goal"


# --- apply_transforms ---


class TestApplyTransforms:
    """XFORM-003–006: apply_transforms runs expansion + stylesheet."""

    def test_runs_variable_expansion(self):
        """apply_transforms expands $goal in prompts."""
        graph = _make_graph()
        context = PipelineContext()
        context.set("graph.goal", "auth system")
        result = apply_transforms(graph, context)
        assert "$goal" not in result.nodes["plan"].prompt
        assert "auth system" in result.nodes["plan"].prompt

    def test_applies_stylesheet(self):
        """apply_transforms applies stylesheet rules to nodes."""
        graph = Graph(
            name="test",
            nodes={
                "start": Node(id="start", shape="Mdiamond"),
                "step": Node(id="step", prompt="Work"),
                "exit": Node(id="exit", shape="Msquare"),
            },
            edges=[
                Edge(from_node="start", to_node="step"),
                Edge(from_node="step", to_node="exit"),
            ],
            model_stylesheet="* { llm_model: gpt-4o; }",
        )
        context = PipelineContext()
        result = apply_transforms(graph, context)
        # Stylesheet should have set llm_model on the step node
        assert result.nodes["step"].attrs.get("llm_model") == "gpt-4o"

    def test_expansion_before_stylesheet(self):
        """Variable expansion runs before stylesheet application."""
        graph = Graph(
            name="test",
            nodes={
                "start": Node(id="start", shape="Mdiamond"),
                "step": Node(id="step", prompt="Build $goal"),
                "exit": Node(id="exit", shape="Msquare"),
            },
            edges=[
                Edge(from_node="start", to_node="step"),
                Edge(from_node="step", to_node="exit"),
            ],
            model_stylesheet="* { llm_model: gpt-4o; }",
        )
        context = PipelineContext()
        context.set("graph.goal", "auth")
        result = apply_transforms(graph, context)
        # Both transforms should have been applied
        assert result.nodes["step"].prompt == "Build auth"
        assert result.nodes["step"].attrs.get("llm_model") == "gpt-4o"

    def test_no_stylesheet_still_expands_variables(self):
        """When no model_stylesheet, variable expansion still runs."""
        graph = _make_graph(model_stylesheet="")
        context = PipelineContext()
        context.set("graph.goal", "auth")
        result = apply_transforms(graph, context)
        assert result.nodes["plan"].prompt == "Plan the work for auth"

    def test_returns_same_graph_object(self):
        """Transforms modify the graph in place and return it."""
        graph = _make_graph()
        context = PipelineContext()
        context.set("graph.goal", "auth")
        result = apply_transforms(graph, context)
        assert result is graph


# --- Engine integration ---


class MockBackend:
    """Backend that captures prompts seen by each node."""

    def __init__(self) -> None:
        self.seen_prompts: dict[str, str] = {}

    async def run(self, node: Node, prompt: str, context: PipelineContext) -> str:
        self.seen_prompts[node.id] = node.prompt
        return "done"


def _make_engine(
    dot_source: str,
    backend: object | None = None,
    logs_root: str = "/tmp/test-transforms",
) -> PipelineEngine:
    graph = parse_dot(dot_source)
    validate_or_raise(graph)
    context = PipelineContext()
    registry = HandlerRegistry(backend=backend)
    return PipelineEngine(
        graph=graph,
        context=context,
        handler_registry=registry,
        logs_root=logs_root,
    )


class TestEngineTransformIntegration:
    """Engine applies transforms before execution loop."""

    @pytest.mark.asyncio
    async def test_engine_expands_goal_in_prompts(self, tmp_path):
        """$goal in node prompts is expanded before handlers see them."""
        backend = MockBackend()
        engine = _make_engine(
            dot_source="""
            digraph {
                goal = "build auth"
                start [shape=Mdiamond]
                plan [prompt="Plan $goal"]
                exit [shape=Msquare]
                start -> plan -> exit
            }
            """,
            backend=backend,
            logs_root=str(tmp_path),
        )
        outcome = await engine.run()
        assert outcome.status in (StageStatus.SUCCESS, StageStatus.PARTIAL_SUCCESS)
        # The handler should have seen the expanded prompt
        assert backend.seen_prompts.get("plan") == "Plan build auth"

    @pytest.mark.asyncio
    async def test_engine_applies_stylesheet(self, tmp_path):
        """Stylesheet rules are applied to nodes before execution."""
        backend = MockBackend()
        engine = _make_engine(
            dot_source="""
            digraph {
                model_stylesheet = "* { llm_model: gpt-4o; }"
                start [shape=Mdiamond]
                step [prompt="Work"]
                exit [shape=Msquare]
                start -> step -> exit
            }
            """,
            backend=backend,
            logs_root=str(tmp_path),
        )
        await engine.run()
        # After transforms, node should have llm_model set
        assert engine.graph.nodes["step"].attrs.get("llm_model") == "gpt-4o"
