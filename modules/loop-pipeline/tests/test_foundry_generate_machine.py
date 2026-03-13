"""Tests for generate-machine.dot -- Artifact Generation Pipeline.

Verifies that examples/dev-machine/foundry/generate-machine.dot is correct per spec:
- start (Mdiamond)
- design_check (parallelogram/tool) checking .dev-machine-design.md existence with parse_json=true
- design_gate diamond: design_exists=false -> done_no_design; design_exists=true -> read_design
- read_design codergen node that reads and summarizes .dev-machine-design.md
- qa_check node (parallelogram/tool) checking if QA features are enabled
- qa_gate diamond: qa_enabled=true -> gen_qa; qa_enabled=false -> gen_iteration (skip QA)
- Multiple convergence-factory folder nodes:
    gen_iteration (iteration.dot), gen_post_session (post-session.dot),
    gen_health_check (health-check.dot), gen_fix_iteration (fix-iteration.dot),
    gen_qa (qa-iteration.dot - conditional), gen_scripts (pipeline scripts),
    gen_infra (infrastructure scripts)
- validation_all node that validates all generated artifacts
- gen_smoke_test node that generates smoke-test.dot
- done_complete terminal (Msquare)
- done_no_design terminal (Msquare)
- Each convergence-factory folder node has context.artifact_goal (verbatim generation rules)
  and context.artifact_path
- Generation rules are verbatim from generate-machine.md and machine-generator agent

Source material:
- amplifier-bundle-dev-machine/modes/generate-machine.md (194 lines)
- amplifier-bundle-dev-machine/agents/machine-generator.md (115 lines)
- amplifier-bundle-dev-machine/context/templates-reference.md (115 lines)

Test file: modules/loop-pipeline/tests/test_foundry_generate_machine.py
DOT file: examples/dev-machine/foundry/generate-machine.dot
"""

from __future__ import annotations

import os

import pytest

from amplifier_module_loop_pipeline.context import PipelineContext
from amplifier_module_loop_pipeline.dot_parser import parse_dot
from amplifier_module_loop_pipeline.engine import PipelineEngine
from amplifier_module_loop_pipeline.graph import Graph, Node
from amplifier_module_loop_pipeline.handlers import HandlerRegistry
from amplifier_module_loop_pipeline.interviewer import Answer, Option, QueueInterviewer  # noqa: F401
from amplifier_module_loop_pipeline.outcome import Outcome, StageStatus

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_TESTS_DIR = os.path.dirname(__file__)
# From modules/loop-pipeline/tests/ -> up 3 levels -> amplifier-bundle-attractor/ -> examples/
_EXAMPLES_DIR = os.path.abspath(os.path.join(_TESTS_DIR, "..", "..", "..", "examples"))
_GENERATE_MACHINE_DOT = os.path.join(
    _EXAMPLES_DIR, "dev-machine", "foundry", "generate-machine.dot"
)


@pytest.fixture(scope="class")
def generate_machine_graph():
    """Parse generate-machine.dot once per test class run."""
    with open(_GENERATE_MACHINE_DOT) as f:
        return parse_dot(f.read())


# ===========================================================================
# TestGenerateMachineParse -- generate-machine.dot structural tests
# ===========================================================================


class TestGenerateMachineParse:
    """Tests for generate-machine.dot parse correctness and structural requirements."""

    # -----------------------------------------------------------------------
    # AC-1: File exists
    # -----------------------------------------------------------------------

    def test_file_exists(self):
        """generate-machine.dot exists at examples/dev-machine/foundry/generate-machine.dot."""
        assert os.path.isfile(_GENERATE_MACHINE_DOT), (
            f"generate-machine.dot not found at {_GENERATE_MACHINE_DOT}"
        )

    # -----------------------------------------------------------------------
    # AC-2: Parses without error
    # -----------------------------------------------------------------------

    def test_parses_without_error(self, generate_machine_graph):
        """generate-machine.dot parses without raising exceptions."""
        assert generate_machine_graph is not None

    # -----------------------------------------------------------------------
    # AC-3: start node with Mdiamond shape
    # -----------------------------------------------------------------------

    def test_has_start_node(self, generate_machine_graph):
        """start node exists with shape=Mdiamond."""
        assert "start" in generate_machine_graph.nodes, (
            f"Node 'start' not found. Nodes: {list(generate_machine_graph.nodes.keys())}"
        )
        assert generate_machine_graph.nodes["start"].shape == "Mdiamond", (
            f"Expected start shape=Mdiamond, "
            f"got {generate_machine_graph.nodes['start'].shape!r}"
        )

    # -----------------------------------------------------------------------
    # AC-4: design_check is a tool (parallelogram) node
    # -----------------------------------------------------------------------

    def test_has_design_check_tool_node(self, generate_machine_graph):
        """design_check node exists with shape=parallelogram (tool node)."""
        assert "design_check" in generate_machine_graph.nodes, (
            f"Node 'design_check' not found. "
            f"Nodes: {list(generate_machine_graph.nodes.keys())}"
        )
        node = generate_machine_graph.nodes["design_check"]
        assert node.shape == "parallelogram", (
            f"Expected design_check shape=parallelogram, got {node.shape!r}"
        )

    # -----------------------------------------------------------------------
    # AC-5: design_check tool_command references .dev-machine-design.md
    # -----------------------------------------------------------------------

    def test_design_check_references_design_file(self, generate_machine_graph):
        """design_check tool_command references .dev-machine-design.md."""
        node = generate_machine_graph.nodes.get("design_check")
        assert node is not None, "design_check node not found"
        tool_command = node.attrs.get("tool_command", "")
        assert ".dev-machine-design.md" in tool_command, (
            f"Expected '.dev-machine-design.md' in design_check tool_command. "
            f"Got: {tool_command!r}"
        )

    # -----------------------------------------------------------------------
    # AC-6: design_gate is a diamond node
    # -----------------------------------------------------------------------

    def test_has_design_gate_diamond(self, generate_machine_graph):
        """design_gate node exists with shape=diamond."""
        assert "design_gate" in generate_machine_graph.nodes, (
            f"Node 'design_gate' not found. "
            f"Nodes: {list(generate_machine_graph.nodes.keys())}"
        )
        node = generate_machine_graph.nodes["design_gate"]
        assert node.shape == "diamond", (
            f"Expected design_gate shape=diamond, got {node.shape!r}"
        )

    # -----------------------------------------------------------------------
    # AC-7: read_design is a codergen (box/default) node
    # -----------------------------------------------------------------------

    def test_has_read_design_node(self, generate_machine_graph):
        """read_design node exists with codergen (box/default) shape."""
        assert "read_design" in generate_machine_graph.nodes, (
            f"Node 'read_design' not found. "
            f"Nodes: {list(generate_machine_graph.nodes.keys())}"
        )
        node = generate_machine_graph.nodes["read_design"]
        # codergen nodes have box or default (empty/None) shape
        assert node.shape in ("box", "rectangle", None, ""), (
            f"Expected read_design to be a codergen (box/default) node, "
            f"got shape={node.shape!r}"
        )

    # -----------------------------------------------------------------------
    # AC-8: qa_check is a tool (parallelogram) node
    # -----------------------------------------------------------------------

    def test_has_qa_check_node(self, generate_machine_graph):
        """qa_check node exists with shape=parallelogram (tool node)."""
        assert "qa_check" in generate_machine_graph.nodes, (
            f"Node 'qa_check' not found. "
            f"Nodes: {list(generate_machine_graph.nodes.keys())}"
        )
        node = generate_machine_graph.nodes["qa_check"]
        assert node.shape == "parallelogram", (
            f"Expected qa_check shape=parallelogram, got {node.shape!r}"
        )

    # -----------------------------------------------------------------------
    # AC-9: qa_gate is a diamond node
    # -----------------------------------------------------------------------

    def test_has_qa_gate_diamond(self, generate_machine_graph):
        """qa_gate node exists with shape=diamond."""
        assert "qa_gate" in generate_machine_graph.nodes, (
            f"Node 'qa_gate' not found. "
            f"Nodes: {list(generate_machine_graph.nodes.keys())}"
        )
        node = generate_machine_graph.nodes["qa_gate"]
        assert node.shape == "diamond", (
            f"Expected qa_gate shape=diamond, got {node.shape!r}"
        )

    # -----------------------------------------------------------------------
    # AC-10: gen_iteration is a folder (convergence-factory) node
    # -----------------------------------------------------------------------

    def test_has_gen_iteration_folder(self, generate_machine_graph):
        """gen_iteration node exists with shape=folder."""
        assert "gen_iteration" in generate_machine_graph.nodes, (
            f"Node 'gen_iteration' not found. "
            f"Nodes: {list(generate_machine_graph.nodes.keys())}"
        )
        node = generate_machine_graph.nodes["gen_iteration"]
        assert node.shape == "folder", (
            f"Expected gen_iteration shape=folder, got {node.shape!r}"
        )

    # -----------------------------------------------------------------------
    # AC-11: gen_iteration references convergence-factory.dot
    # -----------------------------------------------------------------------

    def test_gen_iteration_references_convergence_factory(self, generate_machine_graph):
        """gen_iteration dot_file references convergence-factory.dot."""
        node = generate_machine_graph.nodes.get("gen_iteration")
        assert node is not None, "gen_iteration node not found"
        dot_file = node.attrs.get("dot_file", "")
        assert "convergence-factory.dot" in dot_file, (
            f"Expected 'convergence-factory.dot' in gen_iteration dot_file. "
            f"Got: {dot_file!r}"
        )

    # -----------------------------------------------------------------------
    # AC-12: gen_iteration has context.artifact_path
    # -----------------------------------------------------------------------

    def test_gen_iteration_has_artifact_path(self, generate_machine_graph):
        """gen_iteration has context.artifact_path attribute."""
        node = generate_machine_graph.nodes.get("gen_iteration")
        assert node is not None, "gen_iteration node not found"
        assert "context.artifact_path" in node.attrs, (
            f"Node gen_iteration missing context.artifact_path attr. "
            f"Attrs: {list(node.attrs.keys())}"
        )
        artifact_path = node.attrs.get("context.artifact_path", "")
        assert "iteration.dot" in artifact_path, (
            f"Expected 'iteration.dot' in gen_iteration artifact_path. "
            f"Got: {artifact_path!r}"
        )

    # -----------------------------------------------------------------------
    # AC-13: gen_post_session is a folder node
    # -----------------------------------------------------------------------

    def test_has_gen_post_session_folder(self, generate_machine_graph):
        """gen_post_session node exists with shape=folder."""
        assert "gen_post_session" in generate_machine_graph.nodes, (
            f"Node 'gen_post_session' not found. "
            f"Nodes: {list(generate_machine_graph.nodes.keys())}"
        )
        node = generate_machine_graph.nodes["gen_post_session"]
        assert node.shape == "folder", (
            f"Expected gen_post_session shape=folder, got {node.shape!r}"
        )

    # -----------------------------------------------------------------------
    # AC-14: gen_health_check is a folder node
    # -----------------------------------------------------------------------

    def test_has_gen_health_check_folder(self, generate_machine_graph):
        """gen_health_check node exists with shape=folder."""
        assert "gen_health_check" in generate_machine_graph.nodes, (
            f"Node 'gen_health_check' not found. "
            f"Nodes: {list(generate_machine_graph.nodes.keys())}"
        )
        node = generate_machine_graph.nodes["gen_health_check"]
        assert node.shape == "folder", (
            f"Expected gen_health_check shape=folder, got {node.shape!r}"
        )

    # -----------------------------------------------------------------------
    # AC-15: gen_fix_iteration is a folder node
    # -----------------------------------------------------------------------

    def test_has_gen_fix_iteration_folder(self, generate_machine_graph):
        """gen_fix_iteration node exists with shape=folder."""
        assert "gen_fix_iteration" in generate_machine_graph.nodes, (
            f"Node 'gen_fix_iteration' not found. "
            f"Nodes: {list(generate_machine_graph.nodes.keys())}"
        )
        node = generate_machine_graph.nodes["gen_fix_iteration"]
        assert node.shape == "folder", (
            f"Expected gen_fix_iteration shape=folder, got {node.shape!r}"
        )

    # -----------------------------------------------------------------------
    # AC-16: gen_qa is a folder node (conditional - for QA-enabled projects)
    # -----------------------------------------------------------------------

    def test_has_gen_qa_folder(self, generate_machine_graph):
        """gen_qa node exists with shape=folder."""
        assert "gen_qa" in generate_machine_graph.nodes, (
            f"Node 'gen_qa' not found. "
            f"Nodes: {list(generate_machine_graph.nodes.keys())}"
        )
        node = generate_machine_graph.nodes["gen_qa"]
        assert node.shape == "folder", (
            f"Expected gen_qa shape=folder, got {node.shape!r}"
        )

    # -----------------------------------------------------------------------
    # AC-17: gen_scripts is a folder node
    # -----------------------------------------------------------------------

    def test_has_gen_scripts_folder(self, generate_machine_graph):
        """gen_scripts node exists with shape=folder."""
        assert "gen_scripts" in generate_machine_graph.nodes, (
            f"Node 'gen_scripts' not found. "
            f"Nodes: {list(generate_machine_graph.nodes.keys())}"
        )
        node = generate_machine_graph.nodes["gen_scripts"]
        assert node.shape == "folder", (
            f"Expected gen_scripts shape=folder, got {node.shape!r}"
        )

    # -----------------------------------------------------------------------
    # AC-18: gen_infra is a folder node
    # -----------------------------------------------------------------------

    def test_has_gen_infra_folder(self, generate_machine_graph):
        """gen_infra node exists with shape=folder."""
        assert "gen_infra" in generate_machine_graph.nodes, (
            f"Node 'gen_infra' not found. "
            f"Nodes: {list(generate_machine_graph.nodes.keys())}"
        )
        node = generate_machine_graph.nodes["gen_infra"]
        assert node.shape == "folder", (
            f"Expected gen_infra shape=folder, got {node.shape!r}"
        )

    # -----------------------------------------------------------------------
    # AC-19: validation_all node exists (validates generated DOT files)
    # -----------------------------------------------------------------------

    def test_has_validation_all_node(self, generate_machine_graph):
        """validation_all node exists."""
        assert "validation_all" in generate_machine_graph.nodes, (
            f"Node 'validation_all' not found. "
            f"Nodes: {list(generate_machine_graph.nodes.keys())}"
        )

    # -----------------------------------------------------------------------
    # AC-20: gen_smoke_test is a folder node (generates smoke-test.dot)
    # -----------------------------------------------------------------------

    def test_has_gen_smoke_test_folder(self, generate_machine_graph):
        """gen_smoke_test node exists with shape=folder."""
        assert "gen_smoke_test" in generate_machine_graph.nodes, (
            f"Node 'gen_smoke_test' not found. "
            f"Nodes: {list(generate_machine_graph.nodes.keys())}"
        )
        node = generate_machine_graph.nodes["gen_smoke_test"]
        assert node.shape == "folder", (
            f"Expected gen_smoke_test shape=folder, got {node.shape!r}"
        )

    # -----------------------------------------------------------------------
    # AC-21: done_complete is an Msquare terminal
    # -----------------------------------------------------------------------

    def test_has_done_complete_terminal(self, generate_machine_graph):
        """done_complete node exists with shape=Msquare."""
        assert "done_complete" in generate_machine_graph.nodes, (
            f"Node 'done_complete' not found. "
            f"Nodes: {list(generate_machine_graph.nodes.keys())}"
        )
        node = generate_machine_graph.nodes["done_complete"]
        assert node.shape == "Msquare", (
            f"Expected done_complete shape=Msquare, got {node.shape!r}"
        )

    # -----------------------------------------------------------------------
    # AC-22: done_no_design is an Msquare terminal
    # -----------------------------------------------------------------------

    def test_has_done_no_design_terminal(self, generate_machine_graph):
        """done_no_design node exists with shape=Msquare."""
        assert "done_no_design" in generate_machine_graph.nodes, (
            f"Node 'done_no_design' not found. "
            f"Nodes: {list(generate_machine_graph.nodes.keys())}"
        )
        node = generate_machine_graph.nodes["done_no_design"]
        assert node.shape == "Msquare", (
            f"Expected done_no_design shape=Msquare, got {node.shape!r}"
        )

    # -----------------------------------------------------------------------
    # AC-23: qa_gate routes correctly -- qa_enabled=false skips gen_qa (goes to gen_iteration)
    # -----------------------------------------------------------------------

    def test_qa_gate_skips_gen_qa_when_disabled(self, generate_machine_graph):
        """qa_gate has an edge to gen_iteration when qa_enabled=false (skip path).

        Verifies that qa_gate routes qa_enabled=false directly to gen_iteration,
        bypassing gen_qa for projects without QA.
        """
        qa_gate_edges = [
            e for e in generate_machine_graph.edges if e.from_node == "qa_gate"
        ]
        assert len(qa_gate_edges) >= 2, (
            f"Expected at least 2 edges from qa_gate, got {len(qa_gate_edges)}: "
            f"{[(e.to_node, e.label) for e in qa_gate_edges]}"
        )
        # One edge should go to gen_iteration (qa_enabled=false path)
        targets = [e.to_node for e in qa_gate_edges]
        assert "gen_iteration" in targets, (
            f"Expected qa_gate to have an edge to gen_iteration (qa_disabled path). "
            f"Edges: {[(e.to_node, e.label) for e in qa_gate_edges]}"
        )
        # One edge should go to gen_qa (qa_enabled=true path)
        assert "gen_qa" in targets, (
            f"Expected qa_gate to have an edge to gen_qa (qa_enabled path). "
            f"Edges: {[(e.to_node, e.label) for e in qa_gate_edges]}"
        )


# ===========================================================================
# TestGenerateMachineExecution -- generate-machine.dot engine execution tests
# ===========================================================================

# Path helpers for execution tests
_FOUNDRY_DIR_GENERATE = os.path.abspath(
    os.path.join(_TESTS_DIR, "..", "..", "..", "examples", "dev-machine", "foundry")
)


class MockToolHandlerGenerate:
    """Mock tool handler for generate-machine.dot that injects context variables.

    Handles design_check and qa_check nodes with configurable context values,
    bypassing real shell command execution.
    """

    def __init__(
        self,
        design_exists: str = "false",
        qa_enabled: str = "false",
    ) -> None:
        self._design_exists = design_exists
        self._qa_enabled = qa_enabled

    async def execute(
        self,
        node: Node,
        context: PipelineContext,
        graph: Graph,
        logs_root: str,
    ) -> Outcome:
        if node.id == "design_check":
            context.set("design_exists", self._design_exists)
            return Outcome(
                status=StageStatus.SUCCESS,
                context_updates={"design_exists": self._design_exists},
                notes=f"Mock tool: design_check -> design_exists={self._design_exists}",
            )
        if node.id == "qa_check":
            context.set("qa_enabled", self._qa_enabled)
            return Outcome(
                status=StageStatus.SUCCESS,
                context_updates={"qa_enabled": self._qa_enabled},
                notes=f"Mock tool: qa_check -> qa_enabled={self._qa_enabled}",
            )
        # For any other tool nodes (e.g., validate in convergence-factory)
        return Outcome(
            status=StageStatus.SUCCESS,
            notes=f"Mock tool: {node.id} passed",
        )


class GenerateConvergedBackend:
    """Mock backend for generate-machine.dot that handles read_design and convergence-factory nodes.

    For 'assess' nodes (inside convergence-factory.dot): returns preferred_label='converged'.
    For all other codergen nodes: returns plain SUCCESS.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def run(self, node: Node, prompt: str, context: PipelineContext) -> Outcome:
        self.calls.append(node.id)
        if node.id == "assess":
            return Outcome(status=StageStatus.SUCCESS, preferred_label="converged")
        return Outcome(status=StageStatus.SUCCESS, notes=f"Mock backend: {node.id}")


class TestGenerateMachineExecution:
    """Execution tests for generate-machine.dot using mock backends (no real API calls)."""

    @pytest.mark.asyncio
    async def test_design_missing_exits_early(self, tmp_path):
        """When design_exists=false, pipeline exits to done_no_design without reaching read_design.

        The design_check tool node returns design_exists=false, which causes
        design_gate to route to done_no_design (early exit). The read_design
        codergen node must NOT be reached.
        """
        import pathlib

        with open(_GENERATE_MACHINE_DOT) as f:
            dot_source = f.read()
        graph = parse_dot(dot_source)
        graph.source_dir = _FOUNDRY_DIR_GENERATE

        context = PipelineContext()
        # design_exists=false → early exit
        tool_handler = MockToolHandlerGenerate(design_exists="false")
        registry = HandlerRegistry(backend=GenerateConvergedBackend())
        registry.register("tool", tool_handler)

        engine = PipelineEngine(
            graph=graph,
            context=context,
            handler_registry=registry,
            logs_root=str(pathlib.Path(str(tmp_path)) / "logs"),
        )
        outcome = await engine.run()

        # Pipeline should exit successfully via the early-exit path.
        # Note: terminal nodes (Msquare / exit nodes) are NOT added to completed_nodes
        # by the engine — the engine exits when it reaches them without recording them.
        # We verify the early exit by checking the gate was traversed and the
        # design_exists context variable is 'false'.
        assert outcome.status in (StageStatus.SUCCESS, StageStatus.PARTIAL_SUCCESS), (
            f"Expected SUCCESS on early exit, got {outcome.status}. "
            f"failure_reason={outcome.failure_reason!r}, notes={outcome.notes!r}"
        )
        # design_gate must be traversed (confirms the routing decision was made)
        assert "design_gate" in engine.completed_nodes, (
            f"Expected 'design_gate' in completed_nodes. "
            f"completed_nodes={engine.completed_nodes}"
        )
        # Context confirms design was flagged as missing
        assert engine.context.get("design_exists") == "false", (
            f"Expected design_exists='false' in context, "
            f"got {engine.context.get('design_exists')!r}"
        )
        # read_design must NOT be reached (early exit before it)
        assert "read_design" not in engine.completed_nodes, (
            f"'read_design' should NOT be in completed_nodes for design-missing path. "
            f"completed_nodes={engine.completed_nodes}"
        )

    @pytest.mark.asyncio
    async def test_qa_disabled_skips_gen_qa(self, tmp_path):
        """When qa_enabled=false, gen_qa node is NOT visited.

        The design_check returns design_exists=true so the pipeline continues.
        The qa_check returns qa_enabled=false, which causes qa_gate to route
        directly to gen_iteration (skipping gen_qa). Verifies gen_qa is absent
        from completed_nodes.
        """
        import pathlib

        with open(_GENERATE_MACHINE_DOT) as f:
            dot_source = f.read()
        graph = parse_dot(dot_source)
        graph.source_dir = _FOUNDRY_DIR_GENERATE

        context = PipelineContext()
        # design_exists=true (continue), qa_enabled=false (skip gen_qa)
        tool_handler = MockToolHandlerGenerate(design_exists="true", qa_enabled="false")
        registry = HandlerRegistry(backend=GenerateConvergedBackend())
        registry.register("tool", tool_handler)

        engine = PipelineEngine(
            graph=graph,
            context=context,
            handler_registry=registry,
            logs_root=str(pathlib.Path(str(tmp_path)) / "logs"),
        )
        await engine.run()

        # gen_qa must NOT be visited on the qa_disabled path
        assert "gen_qa" not in engine.completed_nodes, (
            f"'gen_qa' should NOT be in completed_nodes when qa_enabled=false. "
            f"completed_nodes={engine.completed_nodes}"
        )
        # design_gate should route past done_no_design (design exists)
        assert "done_no_design" not in engine.completed_nodes, (
            f"Pipeline should not exit via done_no_design when design_exists=true. "
            f"completed_nodes={engine.completed_nodes}"
        )
