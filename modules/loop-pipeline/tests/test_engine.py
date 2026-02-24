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
        graph=graph,
        context=context,
        handler_registry=registry,
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
async def test_start_node_fallback_to_id_start(tmp_path):
    """Engine falls back to id='start' when no Mdiamond node exists (L-21)."""
    # Build graph manually — no Mdiamond, but a node with id="start"
    graph = Graph(
        name="test",
        nodes={
            "start": Node(id="start", shape="box", prompt="Begin"),
            "work": Node(id="work", shape="box", prompt="Do work"),
            "exit": Node(id="exit", shape="Msquare"),
        },
        edges=[
            Edge(from_node="start", to_node="work"),
            Edge(from_node="work", to_node="exit"),
        ],
    )
    context = PipelineContext()
    registry = HandlerRegistry(backend=MockBackend("ok"))
    engine = PipelineEngine(
        graph=graph,
        context=context,
        handler_registry=registry,
        logs_root=str(tmp_path),
    )
    outcome = await engine.run()
    # Should succeed — engine found the start node via id fallback
    assert outcome.status in (StageStatus.SUCCESS, StageStatus.PARTIAL_SUCCESS)
    assert "start" in engine.completed_nodes


@pytest.mark.asyncio
async def test_start_node_fallback_to_id_Start(tmp_path):
    """Engine falls back to id='Start' (capitalized) when no Mdiamond (L-21)."""
    graph = Graph(
        name="test",
        nodes={
            "Start": Node(id="Start", shape="box", prompt="Begin"),
            "exit": Node(id="exit", shape="Msquare"),
        },
        edges=[
            Edge(from_node="Start", to_node="exit"),
        ],
    )
    context = PipelineContext()
    registry = HandlerRegistry(backend=MockBackend("ok"))
    engine = PipelineEngine(
        graph=graph,
        context=context,
        handler_registry=registry,
        logs_root=str(tmp_path),
    )
    outcome = await engine.run()
    assert outcome.status in (StageStatus.SUCCESS, StageStatus.PARTIAL_SUCCESS)


@pytest.mark.asyncio
async def test_start_node_shape_takes_priority(tmp_path):
    """Mdiamond shape is preferred over id='start' fallback (L-21)."""
    graph = Graph(
        name="test",
        nodes={
            "begin": Node(id="begin", shape="Mdiamond"),
            "start": Node(id="start", shape="box", prompt="Not the start"),
            "exit": Node(id="exit", shape="Msquare"),
        },
        edges=[
            Edge(from_node="begin", to_node="exit"),
            Edge(from_node="start", to_node="exit"),
        ],
    )
    context = PipelineContext()
    registry = HandlerRegistry(backend=MockBackend("ok"))
    engine = PipelineEngine(
        graph=graph,
        context=context,
        handler_registry=registry,
        logs_root=str(tmp_path),
    )
    outcome = await engine.run()
    assert outcome.status in (StageStatus.SUCCESS, StageStatus.PARTIAL_SUCCESS)
    # "start" (the box node) should NOT have been visited as the entry point
    assert "start" not in engine.completed_nodes


@pytest.mark.asyncio
async def test_auto_status_overrides_fail_to_success(tmp_path):
    """auto_status=true overrides non-success outcome to SUCCESS (L-9)."""

    class FailingBackend:
        async def run(self, node, prompt, context):
            if node.id == "auto_node":
                return Outcome(status=StageStatus.FAIL, failure_reason="oops")
            return "ok"

    graph = Graph(
        name="test",
        nodes={
            "start": Node(id="start", shape="Mdiamond"),
            "auto_node": Node(
                id="auto_node",
                shape="box",
                prompt="work",
                auto_status=True,
            ),
            "exit": Node(id="exit", shape="Msquare"),
        },
        edges=[
            Edge(from_node="start", to_node="auto_node"),
            Edge(from_node="auto_node", to_node="exit"),
        ],
    )
    context = PipelineContext()
    registry = HandlerRegistry(backend=FailingBackend())
    engine = PipelineEngine(
        graph=graph,
        context=context,
        handler_registry=registry,
        logs_root=str(tmp_path),
    )
    outcome = await engine.run()
    # auto_status=true should have overridden the FAIL to SUCCESS
    assert engine.node_outcomes["auto_node"].status == StageStatus.SUCCESS
    assert outcome.status in (StageStatus.SUCCESS, StageStatus.PARTIAL_SUCCESS)


@pytest.mark.asyncio
async def test_auto_status_false_preserves_fail(tmp_path):
    """Without auto_status, FAIL outcome is preserved (L-9)."""

    class FailingBackend:
        async def run(self, node, prompt, context):
            if node.id == "fail_node":
                return Outcome(status=StageStatus.FAIL, failure_reason="oops")
            return "ok"

    graph = Graph(
        name="test",
        nodes={
            "start": Node(id="start", shape="Mdiamond"),
            "fail_node": Node(id="fail_node", shape="box", prompt="work"),
            "exit": Node(id="exit", shape="Msquare"),
        },
        edges=[
            Edge(from_node="start", to_node="fail_node"),
            Edge(from_node="fail_node", to_node="exit", condition="outcome=fail"),
        ],
    )
    context = PipelineContext()
    registry = HandlerRegistry(backend=FailingBackend())
    engine = PipelineEngine(
        graph=graph,
        context=context,
        handler_registry=registry,
        logs_root=str(tmp_path),
    )
    await engine.run()
    # Without auto_status, FAIL is preserved
    assert engine.node_outcomes["fail_node"].status == StageStatus.FAIL


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


# --- Alternative start/exit node conventions ---


@pytest.mark.asyncio
async def test_engine_finds_start_by_node_type_attr(tmp_path):
    """Engine finds start node via node_type='start' attribute."""
    graph = Graph(
        name="test",
        nodes={
            "Start": Node(
                id="Start",
                shape="circle",
                label="Start",
                attrs={"node_type": "start"},
            ),
            "work": Node(id="work", shape="box", prompt="Do work"),
            "exit": Node(id="exit", shape="Msquare", label="Exit"),
        },
        edges=[
            Edge(from_node="Start", to_node="work"),
            Edge(from_node="work", to_node="exit"),
        ],
    )
    context = PipelineContext()
    registry = HandlerRegistry(backend=MockBackend("done"))
    engine = PipelineEngine(
        graph=graph,
        context=context,
        handler_registry=registry,
        logs_root=str(tmp_path),
    )
    start = engine._find_start_node()
    assert start.id == "Start"


@pytest.mark.asyncio
async def test_engine_runs_alternative_start_exit(tmp_path):
    """Engine executes pipeline with circle/doublecircle + node_type conventions."""
    graph = Graph(
        name="test",
        nodes={
            "Start": Node(
                id="Start",
                shape="circle",
                label="Start",
                attrs={"node_type": "start"},
            ),
            "work": Node(id="work", shape="box", prompt="Do work"),
            "Exit": Node(
                id="Exit",
                shape="doublecircle",
                label="Exit",
                attrs={"node_type": "exit"},
            ),
        },
        edges=[
            Edge(from_node="Start", to_node="work"),
            Edge(from_node="work", to_node="Exit"),
        ],
    )
    context = PipelineContext()
    registry = HandlerRegistry(backend=MockBackend("done"))
    engine = PipelineEngine(
        graph=graph,
        context=context,
        handler_registry=registry,
        logs_root=str(tmp_path),
    )
    outcome = await engine.run()
    assert outcome.status in (StageStatus.SUCCESS, StageStatus.PARTIAL_SUCCESS)


@pytest.mark.asyncio
async def test_engine_mdiamond_takes_priority_over_node_type(tmp_path):
    """shape=Mdiamond has higher priority than node_type='start'."""
    graph = Graph(
        name="test",
        nodes={
            "real_start": Node(id="real_start", shape="Mdiamond", label="RealStart"),
            "alt_start": Node(
                id="alt_start",
                shape="circle",
                attrs={"node_type": "start"},
            ),
            "work": Node(id="work", shape="box", prompt="Do work"),
            "exit": Node(id="exit", shape="Msquare", label="Exit"),
        },
        edges=[
            Edge(from_node="real_start", to_node="work"),
            Edge(from_node="work", to_node="exit"),
        ],
    )
    context = PipelineContext()
    registry = HandlerRegistry(backend=MockBackend("done"))
    engine = PipelineEngine(
        graph=graph,
        context=context,
        handler_registry=registry,
        logs_root=str(tmp_path),
    )
    start = engine._find_start_node()
    assert start.id == "real_start", "Mdiamond should take priority over node_type"
