"""Tests for the pipeline execution engine.

Spec coverage: EXEC-001–018, Section 3.2.
"""

import pytest

from amplifier_module_loop_pipeline.context import PipelineContext
from amplifier_module_loop_pipeline.dot_parser import parse_dot
from amplifier_module_loop_pipeline.engine import PipelineEngine
from amplifier_module_loop_pipeline.graph import Edge, Graph, Node
from amplifier_module_loop_pipeline.handlers import HandlerRegistry
from amplifier_module_loop_pipeline.outcome import Outcome, StageStatus
from amplifier_module_loop_pipeline.validation import validate_or_raise


class MockBackend:
    """Backend that returns a fixed string for every call."""

    def __init__(self, return_value: str = "done"):
        self._return_value = return_value
        self.calls: list[str] = []

    async def run(self, node: Node, prompt: str, context: PipelineContext) -> str:
        self.calls.append(node.id)
        return self._return_value


class SequenceBackend:
    """Backend that returns different outcomes per node id."""

    def __init__(self, outcomes: dict[str, str | Outcome]):
        self._outcomes = outcomes
        self.calls: list[str] = []

    async def run(
        self, node: Node, prompt: str, context: PipelineContext
    ) -> str | Outcome:
        self.calls.append(node.id)
        return self._outcomes.get(node.id, "ok")


def _make_engine(
    dot_source: str,
    backend: object | None = None,
    logs_root: str = "/tmp/test-pipeline",
) -> PipelineEngine:
    """Parse DOT, validate, and build an engine."""
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


@pytest.mark.asyncio
async def test_simple_linear_pipeline(tmp_path):
    """start -> plan -> implement -> exit completes successfully."""
    engine = _make_engine(
        dot_source="""
        digraph {
            start [shape=Mdiamond]
            plan [prompt="Plan the work"]
            implement [prompt="Build it"]
            exit [shape=Msquare]
            start -> plan -> implement -> exit
        }
        """,
        backend=MockBackend("done"),
        logs_root=str(tmp_path),
    )
    outcome = await engine.run()
    assert outcome.status in (StageStatus.SUCCESS, StageStatus.PARTIAL_SUCCESS)


@pytest.mark.asyncio
async def test_engine_visits_all_nodes(tmp_path):
    """Engine visits start, plan, implement, exit in order."""
    backend = MockBackend("done")
    engine = _make_engine(
        dot_source="""
        digraph {
            start [shape=Mdiamond]
            plan [prompt="Plan"]
            implement [prompt="Build"]
            exit [shape=Msquare]
            start -> plan -> implement -> exit
        }
        """,
        backend=backend,
        logs_root=str(tmp_path),
    )
    await engine.run()
    # Backend is only called for codergen nodes (plan, implement)
    assert backend.calls == ["plan", "implement"]


@pytest.mark.asyncio
async def test_conditional_branching(tmp_path):
    """Condition-based routing follows matching edges."""
    backend = SequenceBackend(
        outcomes={
            "check": Outcome(status=StageStatus.SUCCESS),
        }
    )
    engine = _make_engine(
        dot_source="""
        digraph {
            start [shape=Mdiamond]
            check [shape=diamond]
            pass_path [prompt="Tests pass"]
            fail_path [prompt="Tests fail"]
            exit [shape=Msquare]
            start -> check
            check -> pass_path [condition="outcome=success"]
            check -> fail_path [condition="outcome=fail"]
            pass_path -> exit
            fail_path -> exit
        }
        """,
        backend=backend,
        logs_root=str(tmp_path),
    )
    outcome = await engine.run()
    assert outcome.status == StageStatus.SUCCESS
    # Should have taken the pass_path since check returned SUCCESS
    assert "pass_path" in backend.calls
    assert "fail_path" not in backend.calls


@pytest.mark.asyncio
async def test_context_updates_propagate(tmp_path):
    """Context updates from outcomes are visible to subsequent nodes."""

    class ContextCheckBackend:
        def __init__(self):
            self.seen_values: dict[str, str | None] = {}

        async def run(self, node, prompt, context):
            if node.id == "step1":
                return Outcome(
                    status=StageStatus.SUCCESS,
                    context_updates={"my_key": "my_value"},
                )
            if node.id == "step2":
                self.seen_values["my_key"] = context.get("my_key")
                return "done"
            return "ok"

    backend = ContextCheckBackend()
    engine = _make_engine(
        dot_source="""
        digraph {
            start [shape=Mdiamond]
            step1 [prompt="Step 1"]
            step2 [prompt="Step 2"]
            exit [shape=Msquare]
            start -> step1 -> step2 -> exit
        }
        """,
        backend=backend,
        logs_root=str(tmp_path),
    )
    await engine.run()
    assert backend.seen_values.get("my_key") == "my_value"


@pytest.mark.asyncio
async def test_goal_set_in_context(tmp_path):
    """Graph goal is mirrored into context."""
    engine = _make_engine(
        dot_source="""
        digraph {
            goal = "build auth"
            start [shape=Mdiamond]
            exit [shape=Msquare]
            start -> exit
        }
        """,
        backend=MockBackend("ok"),
        logs_root=str(tmp_path),
    )
    await engine.run()
    assert engine.context.get("graph.goal") == "build auth"


@pytest.mark.asyncio
async def test_no_matching_edge_returns_fail(tmp_path):
    """No outgoing edges from a non-terminal node returns fail."""
    # Build a graph manually where a codergen node has no outgoing edges.
    # (Can't use the parser helper because validation would reject it,
    # so we build the engine directly.)
    from amplifier_module_loop_pipeline.graph import Edge

    graph = Graph(
        name="test",
        nodes={
            "start": Node(id="start", shape="Mdiamond"),
            "dead_end": Node(id="dead_end", prompt="work"),
            "exit": Node(id="exit", shape="Msquare"),
        },
        edges=[
            Edge(from_node="start", to_node="dead_end"),
            # dead_end has NO outgoing edges
        ],
    )
    context = PipelineContext()
    registry = HandlerRegistry(backend=MockBackend("ok"))
    engine = PipelineEngine(
        graph=graph, context=context, handler_registry=registry,
        logs_root=str(tmp_path),
    )
    outcome = await engine.run()
    assert outcome.status == StageStatus.FAIL
    assert "No matching edge" in (outcome.failure_reason or "")


@pytest.mark.asyncio
async def test_goal_gate_unsatisfied_returns_fail(tmp_path):
    """Goal gate with non-success outcome fails the pipeline at exit."""

    class FailingBackend:
        async def run(self, node, prompt, context):
            if node.id == "critical":
                return Outcome(status=StageStatus.FAIL, failure_reason="broken")
            return "ok"

    engine = _make_engine(
        dot_source="""
        digraph {
            start [shape=Mdiamond]
            critical [prompt="Critical step", goal_gate=true]
            exit [shape=Msquare]
            start -> critical
            critical -> exit [condition="outcome=fail"]
        }
        """,
        backend=FailingBackend(),
        logs_root=str(tmp_path),
    )
    outcome = await engine.run()
    assert outcome.status == StageStatus.FAIL


@pytest.mark.asyncio
async def test_deterministic_execution(tmp_path):
    """Same graph + same context = same path."""
    backend1 = MockBackend("done")
    backend2 = MockBackend("done")

    dot_source = """
    digraph {
        start [shape=Mdiamond]
        a [prompt="A"]
        b [prompt="B"]
        exit [shape=Msquare]
        start -> a -> b -> exit
    }
    """
    engine1 = _make_engine(dot_source, backend=backend1, logs_root=str(tmp_path / "r1"))
    engine2 = _make_engine(dot_source, backend=backend2, logs_root=str(tmp_path / "r2"))

    await engine1.run()
    await engine2.run()
    assert backend1.calls == backend2.calls


@pytest.mark.asyncio
async def test_engine_records_node_outcomes(tmp_path):
    """Engine tracks outcomes for every visited node."""
    engine = _make_engine(
        dot_source="""
        digraph {
            start [shape=Mdiamond]
            step [prompt="Do work"]
            exit [shape=Msquare]
            start -> step -> exit
        }
        """,
        backend=MockBackend("done"),
        logs_root=str(tmp_path),
    )
    await engine.run()
    assert "start" in engine.node_outcomes
    assert "step" in engine.node_outcomes
    assert engine.node_outcomes["step"].status == StageStatus.SUCCESS
