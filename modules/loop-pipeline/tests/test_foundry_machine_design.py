"""Tests for machine-design.dot -- Founding Session Pipeline.

Verifies that examples/dev-machine/foundry/machine-design.dot is correct per spec:
- start (Mdiamond)
- assessment_check (parallelogram/tool) checks .dev-machine-assessment.md existence
- assessment_gate (diamond) with 2 conditional edges:
    assessment_exists=false -> done_no_assessment
    assessment_exists=true  -> phase1_config
- 6 convergence-factory folder nodes (phase1_config through phase6_manifest),
  each referencing ../../patterns/convergence-factory.dot
- Each folder node has context.artifact_goal and context.artifact_path
- Phase topics: Configuration, Feature Decomposition, Script Design,
  DOT Pipeline Design, Design Document Assembly, Manifest Generation
- Phase 5 artifact_goal references .dev-machine-design.md
- done_complete (Msquare) and done_no_assessment (Msquare) terminals
- Sequential flow: start -> assessment_check -> assessment_gate ->
  phase1 -> phase2 -> phase3 -> phase4 -> phase5 -> phase6 -> done_complete

Source material:
- amplifier-bundle-dev-machine/modes/machine-design.md (143 lines)
- amplifier-bundle-dev-machine/agents/machine-designer.md (63 lines)

Test file: modules/loop-pipeline/tests/test_foundry_machine_design.py
DOT file: examples/dev-machine/foundry/machine-design.dot
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
_MACHINE_DESIGN_DOT = os.path.join(
    _EXAMPLES_DIR, "dev-machine", "foundry", "machine-design.dot"
)


@pytest.fixture(scope="class")
def machine_design_graph():
    """Parse machine-design.dot once per test class run."""
    with open(_MACHINE_DESIGN_DOT) as f:
        return parse_dot(f.read())


# ===========================================================================
# TestMachineDesignParse -- machine-design.dot structural tests
# ===========================================================================


class TestMachineDesignParse:
    """Tests for machine-design.dot parse correctness and structural requirements."""

    # -----------------------------------------------------------------------
    # AC-1: File exists
    # -----------------------------------------------------------------------

    def test_file_exists(self):
        """machine-design.dot exists at examples/dev-machine/foundry/machine-design.dot."""
        assert os.path.isfile(_MACHINE_DESIGN_DOT), (
            f"machine-design.dot not found at {_MACHINE_DESIGN_DOT}"
        )

    # -----------------------------------------------------------------------
    # AC-2: Parses without error
    # -----------------------------------------------------------------------

    def test_parses_without_error(self, machine_design_graph):
        """machine-design.dot parses without raising exceptions."""
        assert machine_design_graph is not None

    # -----------------------------------------------------------------------
    # AC-3: start node with Mdiamond shape
    # -----------------------------------------------------------------------

    def test_has_start_node(self, machine_design_graph):
        """start node exists with shape=Mdiamond."""
        assert "start" in machine_design_graph.nodes, (
            f"Node 'start' not found. Nodes: {list(machine_design_graph.nodes.keys())}"
        )
        assert machine_design_graph.nodes["start"].shape == "Mdiamond", (
            f"Expected start shape=Mdiamond, "
            f"got {machine_design_graph.nodes['start'].shape!r}"
        )

    # -----------------------------------------------------------------------
    # AC-4: assessment_check is a tool (parallelogram) node
    # -----------------------------------------------------------------------

    def test_has_assessment_check_tool_node(self, machine_design_graph):
        """assessment_check node exists with shape=parallelogram (tool node)."""
        assert "assessment_check" in machine_design_graph.nodes, (
            f"Node 'assessment_check' not found. "
            f"Nodes: {list(machine_design_graph.nodes.keys())}"
        )
        node = machine_design_graph.nodes["assessment_check"]
        assert node.shape == "parallelogram", (
            f"Expected assessment_check shape=parallelogram, got {node.shape!r}"
        )

    # -----------------------------------------------------------------------
    # AC-5: assessment_check tool_command references .dev-machine-assessment.md
    # -----------------------------------------------------------------------

    def test_assessment_check_references_assessment_file(self, machine_design_graph):
        """assessment_check tool_command references .dev-machine-assessment.md."""
        node = machine_design_graph.nodes.get("assessment_check")
        assert node is not None, "assessment_check node not found"
        tool_command = node.attrs.get("tool_command", "")
        assert ".dev-machine-assessment.md" in tool_command, (
            f"Expected '.dev-machine-assessment.md' in assessment_check tool_command. "
            f"Got: {tool_command!r}"
        )

    # -----------------------------------------------------------------------
    # AC-6: assessment_gate is a diamond node
    # -----------------------------------------------------------------------

    def test_has_assessment_gate_diamond(self, machine_design_graph):
        """assessment_gate node exists with shape=diamond."""
        assert "assessment_gate" in machine_design_graph.nodes, (
            f"Node 'assessment_gate' not found. "
            f"Nodes: {list(machine_design_graph.nodes.keys())}"
        )
        node = machine_design_graph.nodes["assessment_gate"]
        assert node.shape == "diamond", (
            f"Expected assessment_gate shape=diamond, got {node.shape!r}"
        )

    # -----------------------------------------------------------------------
    # AC-7: assessment_gate has exactly 2 conditional edges
    # -----------------------------------------------------------------------

    def test_assessment_gate_has_two_conditional_edges(self, machine_design_graph):
        """assessment_gate has exactly 2 outgoing conditional edges."""
        edges = [
            e for e in machine_design_graph.edges if e.from_node == "assessment_gate"
        ]
        assert len(edges) == 2, (
            f"Expected 2 edges from assessment_gate, got {len(edges)}: "
            f"{[(e.to_node, e.label) for e in edges]}"
        )
        for e in edges:
            assert e.condition, (
                f"Edge assessment_gate->{e.to_node} should have a condition, "
                f"got condition={e.condition!r}"
            )

    # -----------------------------------------------------------------------
    # AC-8: Exactly 6 folder nodes
    # -----------------------------------------------------------------------

    def test_has_six_folder_nodes(self, machine_design_graph):
        """Exactly 6 folder nodes (phase1_config through phase6_manifest)."""
        folder_nodes = [
            n for n in machine_design_graph.nodes.values() if n.shape == "folder"
        ]
        assert len(folder_nodes) == 6, (
            f"Expected 6 folder nodes, got {len(folder_nodes)}: "
            f"{[n.id for n in folder_nodes]}"
        )

    # -----------------------------------------------------------------------
    # AC-9: All folder nodes reference convergence-factory.dot
    # -----------------------------------------------------------------------

    def test_folder_nodes_reference_convergence_factory(self, machine_design_graph):
        """All folder nodes reference ../../patterns/convergence-factory.dot."""
        folder_nodes = [
            n for n in machine_design_graph.nodes.values() if n.shape == "folder"
        ]
        for node in folder_nodes:
            dot_file = node.attrs.get("dot_file", "")
            assert "convergence-factory.dot" in dot_file, (
                f"Node {node.id!r} dot_file should reference convergence-factory.dot, "
                f"got {dot_file!r}"
            )

    # -----------------------------------------------------------------------
    # AC-10: Each folder node has context.artifact_goal and context.artifact_path
    # -----------------------------------------------------------------------

    def test_folder_nodes_have_artifact_goal_and_path(self, machine_design_graph):
        """Each folder node has context.artifact_goal and context.artifact_path."""
        folder_nodes = [
            n for n in machine_design_graph.nodes.values() if n.shape == "folder"
        ]
        for node in folder_nodes:
            assert "context.artifact_goal" in node.attrs, (
                f"Node {node.id!r} missing context.artifact_goal attr"
            )
            assert "context.artifact_path" in node.attrs, (
                f"Node {node.id!r} missing context.artifact_path attr"
            )

    # -----------------------------------------------------------------------
    # AC-11: Phase 1 mentions Configuration
    # -----------------------------------------------------------------------

    def test_phase1_mentions_configuration(self, machine_design_graph):
        """Phase 1 folder node artifact_goal mentions 'Configuration'."""
        folder_nodes = [
            n for n in machine_design_graph.nodes.values() if n.shape == "folder"
        ]
        phase1_nodes = [
            n
            for n in folder_nodes
            if "phase1" in n.id.lower()
            or ("config" in n.id.lower() and "phase" in n.id.lower())
        ]
        assert len(phase1_nodes) > 0, (
            f"Expected a phase1 folder node. Node IDs: {[n.id for n in folder_nodes]}"
        )
        assert any(
            "configuration" in n.attrs.get("context.artifact_goal", "").lower()
            for n in phase1_nodes
        ), (
            f"Expected 'configuration' in phase1 artifact_goal. "
            f"Goals: {[n.attrs.get('context.artifact_goal', '')[:100] for n in phase1_nodes]}"
        )

    # -----------------------------------------------------------------------
    # AC-12: Phase 2 mentions Feature
    # -----------------------------------------------------------------------

    def test_phase2_mentions_feature(self, machine_design_graph):
        """Phase 2 folder node artifact_goal mentions 'feature'."""
        folder_nodes = [
            n for n in machine_design_graph.nodes.values() if n.shape == "folder"
        ]
        phase2_nodes = [n for n in folder_nodes if "phase2" in n.id.lower()]
        assert len(phase2_nodes) > 0, (
            f"Expected a phase2 folder node. Node IDs: {[n.id for n in folder_nodes]}"
        )
        assert any(
            "feature" in n.attrs.get("context.artifact_goal", "").lower()
            for n in phase2_nodes
        ), (
            f"Expected 'feature' in phase2 artifact_goal. "
            f"Goals: {[n.attrs.get('context.artifact_goal', '')[:100] for n in phase2_nodes]}"
        )

    # -----------------------------------------------------------------------
    # AC-13: Phase 3 mentions Script
    # -----------------------------------------------------------------------

    def test_phase3_mentions_script(self, machine_design_graph):
        """Phase 3 folder node artifact_goal mentions 'script'."""
        folder_nodes = [
            n for n in machine_design_graph.nodes.values() if n.shape == "folder"
        ]
        phase3_nodes = [n for n in folder_nodes if "phase3" in n.id.lower()]
        assert len(phase3_nodes) > 0, (
            f"Expected a phase3 folder node. Node IDs: {[n.id for n in folder_nodes]}"
        )
        assert any(
            "script" in n.attrs.get("context.artifact_goal", "").lower()
            for n in phase3_nodes
        ), (
            f"Expected 'script' in phase3 artifact_goal. "
            f"Goals: {[n.attrs.get('context.artifact_goal', '')[:100] for n in phase3_nodes]}"
        )

    # -----------------------------------------------------------------------
    # AC-14: Phase 4 mentions DOT or pipeline
    # -----------------------------------------------------------------------

    def test_phase4_mentions_dot_pipeline(self, machine_design_graph):
        """Phase 4 folder node artifact_goal mentions 'DOT' or 'pipeline'."""
        folder_nodes = [
            n for n in machine_design_graph.nodes.values() if n.shape == "folder"
        ]
        phase4_nodes = [n for n in folder_nodes if "phase4" in n.id.lower()]
        assert len(phase4_nodes) > 0, (
            f"Expected a phase4 folder node. Node IDs: {[n.id for n in folder_nodes]}"
        )
        assert any(
            "dot" in n.attrs.get("context.artifact_goal", "").lower()
            or "pipeline" in n.attrs.get("context.artifact_goal", "").lower()
            for n in phase4_nodes
        ), (
            f"Expected 'dot' or 'pipeline' in phase4 artifact_goal. "
            f"Goals: {[n.attrs.get('context.artifact_goal', '')[:100] for n in phase4_nodes]}"
        )

    # -----------------------------------------------------------------------
    # AC-15: Phase 5 mentions .dev-machine-design.md
    # -----------------------------------------------------------------------

    def test_phase5_mentions_design_document(self, machine_design_graph):
        """Phase 5 folder node artifact_goal mentions .dev-machine-design.md."""
        folder_nodes = [
            n for n in machine_design_graph.nodes.values() if n.shape == "folder"
        ]
        phase5_nodes = [n for n in folder_nodes if "phase5" in n.id.lower()]
        assert len(phase5_nodes) > 0, (
            f"Expected a phase5 folder node. Node IDs: {[n.id for n in folder_nodes]}"
        )
        assert any(
            ".dev-machine-design.md" in n.attrs.get("context.artifact_goal", "")
            for n in phase5_nodes
        ), (
            f"Expected '.dev-machine-design.md' in phase5 artifact_goal. "
            f"Goals: {[n.attrs.get('context.artifact_goal', '')[:100] for n in phase5_nodes]}"
        )

    # -----------------------------------------------------------------------
    # AC-16: Phase 6 mentions Manifest
    # -----------------------------------------------------------------------

    def test_phase6_mentions_manifest(self, machine_design_graph):
        """Phase 6 folder node artifact_goal mentions 'manifest'."""
        folder_nodes = [
            n for n in machine_design_graph.nodes.values() if n.shape == "folder"
        ]
        phase6_nodes = [n for n in folder_nodes if "phase6" in n.id.lower()]
        assert len(phase6_nodes) > 0, (
            f"Expected a phase6 folder node. Node IDs: {[n.id for n in folder_nodes]}"
        )
        assert any(
            "manifest" in n.attrs.get("context.artifact_goal", "").lower()
            for n in phase6_nodes
        ), (
            f"Expected 'manifest' in phase6 artifact_goal. "
            f"Goals: {[n.attrs.get('context.artifact_goal', '')[:100] for n in phase6_nodes]}"
        )

    # -----------------------------------------------------------------------
    # AC-17: done_complete is an Msquare terminal
    # -----------------------------------------------------------------------

    def test_has_done_complete_terminal(self, machine_design_graph):
        """done_complete node exists with shape=Msquare."""
        assert "done_complete" in machine_design_graph.nodes, (
            f"Node 'done_complete' not found. "
            f"Nodes: {list(machine_design_graph.nodes.keys())}"
        )
        node = machine_design_graph.nodes["done_complete"]
        assert node.shape == "Msquare", (
            f"Expected done_complete shape=Msquare, got {node.shape!r}"
        )

    # -----------------------------------------------------------------------
    # AC-18: done_no_assessment is an Msquare terminal
    # -----------------------------------------------------------------------

    def test_has_done_no_assessment_terminal(self, machine_design_graph):
        """done_no_assessment node exists with shape=Msquare."""
        assert "done_no_assessment" in machine_design_graph.nodes, (
            f"Node 'done_no_assessment' not found. "
            f"Nodes: {list(machine_design_graph.nodes.keys())}"
        )
        node = machine_design_graph.nodes["done_no_assessment"]
        assert node.shape == "Msquare", (
            f"Expected done_no_assessment shape=Msquare, got {node.shape!r}"
        )

    # -----------------------------------------------------------------------
    # AC-19: Sequential flow from assessment through phases to done_complete
    # -----------------------------------------------------------------------

    def test_sequential_flow_assessment_to_done(self, machine_design_graph):
        """Sequential flow: start->assessment_check->assessment_gate->phase1->...->phase6->done_complete.

        Verifies:
        - start has exactly 1 outgoing edge to assessment_check
        - assessment_check has exactly 1 outgoing edge to assessment_gate
        - assessment_gate has an edge to a phase node (via conditional)
        - phase1 through phase6 form a sequential chain to done_complete (7 nodes)
        """
        # Build edge map: from_node -> list of to_nodes
        edge_map: dict[str, list[str]] = {}
        for e in machine_design_graph.edges:
            edge_map.setdefault(e.from_node, []).append(e.to_node)

        # Verify start -> assessment_check (single outgoing edge)
        start_targets = edge_map.get("start", [])
        assert "assessment_check" in start_targets, (
            f"Expected start -> assessment_check. start edges: {start_targets}"
        )
        assert len(start_targets) == 1, (
            f"Expected start to have exactly 1 outgoing edge, got {len(start_targets)}: "
            f"{start_targets}"
        )

        # Verify assessment_check -> assessment_gate (single outgoing edge)
        check_targets = edge_map.get("assessment_check", [])
        assert "assessment_gate" in check_targets, (
            f"Expected assessment_check -> assessment_gate. "
            f"assessment_check edges: {check_targets}"
        )
        assert len(check_targets) == 1, (
            f"Expected assessment_check to have exactly 1 outgoing edge, "
            f"got {len(check_targets)}: {check_targets}"
        )

        # Find the edge from assessment_gate to a phase node
        gate_targets = edge_map.get("assessment_gate", [])
        phase_targets = [t for t in gate_targets if "phase" in t.lower()]
        assert len(phase_targets) == 1, (
            f"Expected exactly 1 phase target from assessment_gate, "
            f"got {phase_targets}. All gate targets: {gate_targets}"
        )
        first_phase = phase_targets[0]

        # Walk the sequential chain: phase1 -> phase2 -> ... -> phase6 -> done_complete
        # Expected: 7 nodes (6 phases + done_complete)
        current = first_phase
        chain = [current]
        for _ in range(7):
            next_nodes = edge_map.get(current, [])
            if not next_nodes:
                break
            assert len(next_nodes) == 1, (
                f"Sequential node '{current}' should have exactly 1 outgoing edge, "
                f"got {len(next_nodes)}: {next_nodes}"
            )
            current = next_nodes[0]
            chain.append(current)

        assert chain[-1] == "done_complete", (
            f"Chain should end with 'done_complete', got {chain[-1]}. "
            f"Full chain: {chain}"
        )
        assert len(chain) == 7, (
            f"Expected 7-node chain (phase1 through phase6 + done_complete), "
            f"got {len(chain)}: {chain}"
        )


# ===========================================================================
# TestMachineDesignExecution -- machine-design.dot engine execution tests
# ===========================================================================

# Path helpers for execution tests
_FOUNDRY_DIR_DESIGN = os.path.abspath(
    os.path.join(_TESTS_DIR, "..", "..", "..", "examples", "dev-machine", "foundry")
)


class MockToolHandlerDesign:
    """Mock tool handler that returns SUCCESS with assessment_exists='true' JSON.

    Registered as the 'tool' handler in HandlerRegistry to replace the real
    ToolHandler (which runs shell commands) during tests. Returns assessment_exists='true'
    so the pipeline continues to phase1_config rather than exiting early.
    """

    async def execute(
        self,
        node: Node,
        context: PipelineContext,
        graph: Graph,
        logs_root: str,
    ) -> Outcome:
        # Simulate assessment file exists by injecting assessment_exists=true
        context.set("assessment_exists", "true")
        return Outcome(
            status=StageStatus.SUCCESS,
            context_updates={"assessment_exists": "true"},
            notes=f"Mock tool: {node.id} -> assessment_exists=true",
        )


class DesignConvergedBackend:
    """Simulates all gate scoring and all convergence-factory phases converging.

    For 'assess' nodes (inside convergence-factory.dot subgraph): returns preferred_label='converged'.
    For all other codergen nodes: returns plain SUCCESS.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def run(self, node: Node, prompt: str, context: PipelineContext) -> Outcome:
        self.calls.append(node.id)
        if node.id == "assess":
            return Outcome(status=StageStatus.SUCCESS, preferred_label="converged")
        return Outcome(status=StageStatus.SUCCESS, notes=f"Mock backend: {node.id}")


class MockToolMissingAssessment:
    """Mock tool handler that returns assessment_exists='false'.

    Used for test_assessment_missing_exits_early to simulate the assessment file
    being absent, which triggers the early exit to done_no_assessment.
    """

    async def execute(
        self,
        node: Node,
        context: PipelineContext,
        graph: Graph,
        logs_root: str,
    ) -> Outcome:
        # Simulate assessment file does NOT exist
        context.set("assessment_exists", "false")
        return Outcome(
            status=StageStatus.SUCCESS,
            context_updates={"assessment_exists": "false"},
            notes=f"Mock tool: {node.id} -> assessment_exists=false",
        )


class TestMachineDesignExecution:
    """Execution tests for machine-design.dot using mock backends (no real API calls)."""

    @pytest.mark.asyncio
    async def test_assessment_missing_exits_early(self, tmp_path):
        """When assessment_exists=false, pipeline exits to done_no_assessment without reaching phase1_config.

        The assessment_check tool node returns assessment_exists=false, which causes
        assessment_gate to route to done_no_assessment (early exit). The phase1_config
        folder node (and any subsequent phases) must NOT be reached.
        """
        import pathlib

        with open(_MACHINE_DESIGN_DOT) as f:
            dot_source = f.read()
        graph = parse_dot(dot_source)
        graph.source_dir = _FOUNDRY_DIR_DESIGN

        context = PipelineContext()
        registry = HandlerRegistry(backend=DesignConvergedBackend())
        # Override the 'tool' handler to simulate assessment missing
        registry.register("tool", MockToolMissingAssessment())

        engine = PipelineEngine(
            graph=graph,
            context=context,
            handler_registry=registry,
            logs_root=str(pathlib.Path(str(tmp_path)) / "logs"),
        )
        outcome = await engine.run()

        # Pipeline should exit successfully via the early-exit path
        # Note: terminal nodes (Msquare / exit nodes) are NOT added to completed_nodes
        # by the engine — the engine exits when it reaches them without recording them.
        # We verify the early exit by checking the gate was traversed and the
        # assessment_exists context variable is 'false'.
        assert outcome.status in (StageStatus.SUCCESS, StageStatus.PARTIAL_SUCCESS), (
            f"Expected SUCCESS on early exit, got {outcome.status}. "
            f"failure_reason={outcome.failure_reason!r}, notes={outcome.notes!r}"
        )
        # assessment_gate must be traversed (confirms the routing decision was made)
        assert "assessment_gate" in engine.completed_nodes, (
            f"Expected 'assessment_gate' in completed_nodes. "
            f"completed_nodes={engine.completed_nodes}"
        )
        # Context confirms assessment was flagged as missing
        assert engine.context.get("assessment_exists") == "false", (
            f"Expected assessment_exists='false' in context, "
            f"got {engine.context.get('assessment_exists')!r}"
        )
        # phase1_config must NOT be reached (early exit before it)
        assert "phase1_config" not in engine.completed_nodes, (
            f"'phase1_config' should NOT be in completed_nodes for assessment-missing path. "
            f"completed_nodes={engine.completed_nodes}"
        )
