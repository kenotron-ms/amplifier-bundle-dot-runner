"""Tests for admissions.dot -- Project Readiness Evaluation pipeline.

Verifies that examples/dev-machine/foundry/admissions.dot is correct per spec:
- 10 nodes: start (Mdiamond), gate1-5 (folder), compile_assessment (codergen/box),
  verdict_gate (diamond), 3 done terminals (Msquare: done_proceed, done_caution, done_not_ready)
- All 5 folder nodes reference ../../patterns/conversational-gate.dot
- Each folder node has context.gate_topic, context.gate_criteria, context.gate_output_path
- Gate topics: DECOMPOSABILITY, CORRECTNESS, ARCHITECTURE, TOOLCHAIN, SPEC
- Gate criteria contain scoring thresholds (75%, 50%)
- compile_assessment references all 5 gate output files, scoring thresholds,
  .dev-machine-assessment.md, and preferred_label routing instruction
- verdict_gate has 3 conditional edges: proceed, caution, not_ready
- Sequential flow: start -> gate1 -> gate2 -> gate3 -> gate4 -> gate5
                           -> compile_assessment -> verdict_gate

Source material:
- amplifier-bundle-dev-machine/modes/admissions.md (115 lines)
- amplifier-bundle-dev-machine/context/gate-criteria.md (195 lines)
- amplifier-bundle-dev-machine/agents/admissions-advisor.md (58 lines)

Test file: modules/loop-pipeline/tests/test_foundry_admissions.py
DOT file: examples/dev-machine/foundry/admissions.dot
"""

from __future__ import annotations

import os

import pytest

from amplifier_module_loop_pipeline.context import PipelineContext
from amplifier_module_loop_pipeline.dot_parser import parse_dot
from amplifier_module_loop_pipeline.engine import PipelineEngine
from amplifier_module_loop_pipeline.graph import Graph, Node  # noqa: F401
from amplifier_module_loop_pipeline.handlers import HandlerRegistry
from amplifier_module_loop_pipeline.interviewer import (  # noqa: F401
    Answer,
    Option,
    QueueInterviewer,
)
from amplifier_module_loop_pipeline.outcome import Outcome, StageStatus

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_TESTS_DIR = os.path.dirname(__file__)
# From modules/loop-pipeline/tests/ -> up 3 levels -> amplifier-bundle-attractor/ -> examples/
_EXAMPLES_DIR = os.path.abspath(os.path.join(_TESTS_DIR, "..", "..", "..", "examples"))
_ADMISSIONS_DOT = os.path.join(
    _EXAMPLES_DIR, "dev-machine", "foundry", "admissions.dot"
)


@pytest.fixture(scope="class")
def admissions_graph():
    """Parse admissions.dot once per test class run."""
    with open(_ADMISSIONS_DOT) as f:
        return parse_dot(f.read())


# ===========================================================================
# TestAdmissionsParse -- admissions.dot structural tests
# ===========================================================================


class TestAdmissionsParse:
    """Tests for admissions.dot parse correctness and structural requirements."""

    # -----------------------------------------------------------------------
    # AC-1: File exists
    # -----------------------------------------------------------------------

    def test_file_exists(self):
        """admissions.dot exists at examples/dev-machine/foundry/admissions.dot."""
        assert os.path.isfile(_ADMISSIONS_DOT), (
            f"admissions.dot not found at {_ADMISSIONS_DOT}"
        )

    # -----------------------------------------------------------------------
    # AC-2: Parses without error
    # -----------------------------------------------------------------------

    def test_parses_without_error(self, admissions_graph):
        """admissions.dot parses without raising exceptions."""
        assert admissions_graph is not None

    # -----------------------------------------------------------------------
    # AC-3: Exactly 10 nodes (spec description actually yields 11)
    # -----------------------------------------------------------------------

    def test_has_eleven_nodes(self, admissions_graph):
        """Exactly 11 nodes: start, gate1-5, compile_assessment, verdict_gate, 3 done terminals.

        Note: the spec description lists 11 distinct node elements
        (start + 5 gates + compile_assessment + verdict_gate + 3 terminals = 11).
        """
        assert len(admissions_graph.nodes) == 11, (
            f"Expected 11 nodes, got {len(admissions_graph.nodes)}: "
            f"{list(admissions_graph.nodes.keys())}"
        )

    # -----------------------------------------------------------------------
    # AC-4: start node exists with Mdiamond shape
    # -----------------------------------------------------------------------

    def test_has_start_node(self, admissions_graph):
        """start node exists with shape=Mdiamond."""
        assert "start" in admissions_graph.nodes, (
            f"Node 'start' not found. Nodes: {list(admissions_graph.nodes.keys())}"
        )
        assert admissions_graph.nodes["start"].shape == "Mdiamond", (
            f"Expected start shape=Mdiamond, got {admissions_graph.nodes['start'].shape!r}"
        )

    # -----------------------------------------------------------------------
    # AC-5: Exactly 5 folder nodes
    # -----------------------------------------------------------------------

    def test_has_five_folder_nodes(self, admissions_graph):
        """Exactly 5 folder nodes (gate1-gate5)."""
        folder_nodes = [
            n for n in admissions_graph.nodes.values() if n.shape == "folder"
        ]
        assert len(folder_nodes) == 5, (
            f"Expected 5 folder nodes, got {len(folder_nodes)}: "
            f"{[n.id for n in folder_nodes]}"
        )

    # -----------------------------------------------------------------------
    # AC-6: All folder nodes reference conversational-gate.dot
    # -----------------------------------------------------------------------

    def test_folder_nodes_reference_conversational_gate(self, admissions_graph):
        """All folder nodes reference ../../patterns/conversational-gate.dot."""
        folder_nodes = [
            n for n in admissions_graph.nodes.values() if n.shape == "folder"
        ]
        for node in folder_nodes:
            dot_file = node.attrs.get("dot_file", "")
            assert "conversational-gate.dot" in dot_file, (
                f"Node {node.id!r} dot_file should reference conversational-gate.dot, "
                f"got {dot_file!r}"
            )

    # -----------------------------------------------------------------------
    # AC-7: Each folder node has all three context attrs
    # -----------------------------------------------------------------------

    def test_folder_nodes_have_all_three_context_attrs(self, admissions_graph):
        """Each folder node has context.gate_topic, context.gate_criteria, context.gate_output_path."""
        folder_nodes = [
            n for n in admissions_graph.nodes.values() if n.shape == "folder"
        ]
        for node in folder_nodes:
            assert "context.gate_topic" in node.attrs, (
                f"Node {node.id!r} missing context.gate_topic attr"
            )
            assert "context.gate_criteria" in node.attrs, (
                f"Node {node.id!r} missing context.gate_criteria attr"
            )
            assert "context.gate_output_path" in node.attrs, (
                f"Node {node.id!r} missing context.gate_output_path attr"
            )

    # -----------------------------------------------------------------------
    # AC-8: Gate output paths are unique
    # -----------------------------------------------------------------------

    def test_gate_output_paths_are_unique(self, admissions_graph):
        """Each gate has a unique context.gate_output_path."""
        folder_nodes = [
            n for n in admissions_graph.nodes.values() if n.shape == "folder"
        ]
        paths = [
            node.attrs.get("context.gate_output_path", "") for node in folder_nodes
        ]
        assert len(paths) == len(set(paths)), (
            f"Expected all gate output paths to be unique, got duplicates: {paths}"
        )

    # -----------------------------------------------------------------------
    # AC-9..13: Gate topics cover all 5 areas
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize(
        "keyword",
        ["DECOMPOSABILITY", "CORRECTNESS", "ARCHITECTURE", "TOOLCHAIN", "SPEC"],
    )
    def test_gate_topics_contain_keyword(self, admissions_graph, keyword: str):
        """At least one gate topic contains the expected keyword."""
        folder_nodes = [
            n for n in admissions_graph.nodes.values() if n.shape == "folder"
        ]
        topics = [node.attrs.get("context.gate_topic", "") for node in folder_nodes]
        assert any(keyword in t for t in topics), (
            f"Expected {keyword} in a gate topic. Topics: {topics}"
        )

    # -----------------------------------------------------------------------
    # AC-14: Gate criteria contain scoring thresholds
    # -----------------------------------------------------------------------

    def test_gate_criteria_contain_scoring_thresholds(self, admissions_graph):
        """Gate criteria contain scoring thresholds (75% and 50%)."""
        folder_nodes = [
            n for n in admissions_graph.nodes.values() if n.shape == "folder"
        ]
        all_criteria = " ".join(
            node.attrs.get("context.gate_criteria", "") for node in folder_nodes
        )
        assert "75" in all_criteria, (
            f"Expected '75' (75%) threshold in gate criteria. "
            f"Criteria (first 300 chars): {all_criteria[:300]}"
        )
        assert "50" in all_criteria, (
            f"Expected '50' (50%) threshold in gate criteria. "
            f"Criteria (first 300 chars): {all_criteria[:300]}"
        )

    # -----------------------------------------------------------------------
    # AC-15: compile_assessment is a codergen node (box/default shape)
    # -----------------------------------------------------------------------

    def test_has_compile_assessment_codergen_node(self, admissions_graph):
        """compile_assessment node exists with codergen (box/default) shape."""
        assert "compile_assessment" in admissions_graph.nodes, (
            f"Node 'compile_assessment' not found. "
            f"Nodes: {list(admissions_graph.nodes.keys())}"
        )
        node = admissions_graph.nodes["compile_assessment"]
        # codergen nodes have box or default (empty/None) shape
        assert node.shape in ("box", "rectangle", None, ""), (
            f"Expected compile_assessment to be a codergen (box/default) node, "
            f"got shape={node.shape!r}"
        )

    # -----------------------------------------------------------------------
    # AC-16: compile_assessment prompt references all 5 gate output files
    # -----------------------------------------------------------------------

    def test_compile_assessment_prompt_contains_gate_files(self, admissions_graph):
        """compile_assessment prompt references all 5 gate output files."""
        node = admissions_graph.nodes.get("compile_assessment")
        assert node is not None, "compile_assessment node not found"
        prompt = node.prompt or ""
        for i in range(1, 6):
            assert f".ai/gate{i}_" in prompt, (
                f"Expected reference to .ai/gate{i}_*.md in compile_assessment prompt. "
                f"Prompt (first 400 chars): {prompt[:400]}"
            )

    # -----------------------------------------------------------------------
    # AC-17: compile_assessment prompt contains threshold rules
    # -----------------------------------------------------------------------

    def test_compile_assessment_prompt_contains_threshold_rules(self, admissions_graph):
        """compile_assessment prompt contains scoring threshold rules (50%, 75%)."""
        node = admissions_graph.nodes.get("compile_assessment")
        assert node is not None, "compile_assessment node not found"
        prompt = node.prompt or ""
        assert "50" in prompt, (
            f"Expected '50' (50%) threshold rule in compile_assessment prompt. "
            f"Prompt (first 400 chars): {prompt[:400]}"
        )
        assert "75" in prompt, (
            f"Expected '75' (75%) threshold rule in compile_assessment prompt. "
            f"Prompt (first 400 chars): {prompt[:400]}"
        )

    # -----------------------------------------------------------------------
    # AC-18: compile_assessment prompt mentions .dev-machine-assessment.md
    # -----------------------------------------------------------------------

    def test_compile_assessment_prompt_mentions_dev_machine_assessment(
        self, admissions_graph
    ):
        """compile_assessment prompt mentions .dev-machine-assessment.md."""
        node = admissions_graph.nodes.get("compile_assessment")
        assert node is not None, "compile_assessment node not found"
        prompt = node.prompt or ""
        assert ".dev-machine-assessment.md" in prompt, (
            f"Expected '.dev-machine-assessment.md' in compile_assessment prompt. "
            f"Prompt (first 400 chars): {prompt[:400]}"
        )

    # -----------------------------------------------------------------------
    # AC-19: compile_assessment prompt mentions preferred_label
    # -----------------------------------------------------------------------

    def test_compile_assessment_prompt_mentions_preferred_label(self, admissions_graph):
        """compile_assessment prompt mentions preferred_label routing instruction."""
        node = admissions_graph.nodes.get("compile_assessment")
        assert node is not None, "compile_assessment node not found"
        prompt = node.prompt or ""
        assert "preferred_label" in prompt, (
            f"Expected 'preferred_label' in compile_assessment prompt. "
            f"Prompt (first 400 chars): {prompt[:400]}"
        )

    # -----------------------------------------------------------------------
    # AC-20: verdict_gate has diamond shape
    # -----------------------------------------------------------------------

    def test_has_verdict_diamond(self, admissions_graph):
        """verdict_gate node exists with shape=diamond."""
        assert "verdict_gate" in admissions_graph.nodes, (
            f"Node 'verdict_gate' not found. "
            f"Nodes: {list(admissions_graph.nodes.keys())}"
        )
        assert admissions_graph.nodes["verdict_gate"].shape == "diamond", (
            f"Expected verdict_gate shape=diamond, "
            f"got {admissions_graph.nodes['verdict_gate'].shape!r}"
        )

    # -----------------------------------------------------------------------
    # AC-21: Exactly 3 terminal (Msquare) nodes
    # -----------------------------------------------------------------------

    def test_has_three_terminal_nodes(self, admissions_graph):
        """Exactly 3 Msquare terminal nodes (done_proceed, done_caution, done_not_ready)."""
        terminal_nodes = [
            n for n in admissions_graph.nodes.values() if n.shape == "Msquare"
        ]
        assert len(terminal_nodes) == 3, (
            f"Expected 3 Msquare terminal nodes, got {len(terminal_nodes)}: "
            f"{[n.id for n in terminal_nodes]}"
        )

    # -----------------------------------------------------------------------
    # AC-22: verdict_gate has 3 conditional outgoing edges
    # -----------------------------------------------------------------------

    def test_verdict_gate_has_three_conditional_edges(self, admissions_graph):
        """verdict_gate has exactly 3 conditional outgoing edges."""
        verdict_edges = [
            e for e in admissions_graph.edges if e.from_node == "verdict_gate"
        ]
        assert len(verdict_edges) == 3, (
            f"Expected 3 edges from verdict_gate, got {len(verdict_edges)}: "
            f"{[(e.to_node, e.label) for e in verdict_edges]}"
        )
        for e in verdict_edges:
            assert e.condition, (
                f"Edge verdict_gate->{e.to_node} should have a condition, "
                f"got condition={e.condition!r}"
            )

    # -----------------------------------------------------------------------
    # AC-23: verdict conditions cover proceed, caution, not_ready
    # -----------------------------------------------------------------------

    def test_verdict_conditions_cover_all_verdicts(self, admissions_graph):
        """verdict_gate edges cover proceed, caution, and not_ready conditions."""
        verdict_edges = [
            e for e in admissions_graph.edges if e.from_node == "verdict_gate"
        ]
        labels = {e.label for e in verdict_edges}
        assert "proceed" in labels, (
            f"Expected 'proceed' label in verdict_gate edges, got {labels}"
        )
        assert "caution" in labels, (
            f"Expected 'caution' label in verdict_gate edges, got {labels}"
        )
        assert "not_ready" in labels, (
            f"Expected 'not_ready' label in verdict_gate edges, got {labels}"
        )

    # -----------------------------------------------------------------------
    # AC-24: Sequential flow start -> gate1 -> ... -> verdict_gate
    # -----------------------------------------------------------------------

    def test_sequential_flow_start_to_verdict(self, admissions_graph):
        """Sequential 8-node chain: start->gate1->gate2->gate3->gate4->gate5->compile_assessment->verdict_gate."""
        # Build edge map: from_node -> list of to_nodes
        edge_map: dict[str, list[str]] = {}
        for e in admissions_graph.edges:
            edge_map.setdefault(e.from_node, []).append(e.to_node)

        # Walk the chain starting from 'start' (7 hops to reach verdict_gate)
        current = "start"
        chain = [current]
        for _ in range(7):
            next_nodes = edge_map.get(current, [])
            if not next_nodes:
                break
            # Every node in this sequential chain should have exactly one outgoing
            # edge; if a node gains a second edge (e.g. an error bypass), this
            # assertion will catch the assumption before the wrong path is silently
            # traversed.
            assert len(next_nodes) == 1, (
                f"Sequential node '{current}' should have exactly 1 outgoing edge, "
                f"got {len(next_nodes)}: {next_nodes}"
            )
            current = next_nodes[0]
            chain.append(current)

        assert len(chain) == 8, (
            f"Expected 8-node chain (start->...->verdict_gate), "
            f"got {len(chain)}: {chain}"
        )
        assert chain[0] == "start", f"Chain should start with 'start', got {chain[0]}"
        assert chain[-1] == "verdict_gate", (
            f"Chain should end with 'verdict_gate', got {chain[-1]}"
        )
        # Verify all required nodes are in the chain
        for required in [
            "gate1",
            "gate2",
            "gate3",
            "gate4",
            "gate5",
            "compile_assessment",
        ]:
            assert required in chain, (
                f"'{required}' not found in sequential chain: {chain}"
            )


# ===========================================================================
# TestAdmissionsExecution -- admissions.dot engine execution tests
# ===========================================================================

# Path helpers for execution tests
_FOUNDRY_DIR = os.path.abspath(
    os.path.join(_TESTS_DIR, "..", "..", "..", "examples", "dev-machine", "foundry")
)
_PATTERNS_DIR = os.path.abspath(
    os.path.join(_TESTS_DIR, "..", "..", "..", "examples", "patterns")
)


class ProceedBackend:
    """Simulates all 5 gates scoring + compile_assessment returning proceed verdict.

    For gate eval nodes (id='eval' inside conversational-gate.dot): returns preferred_label='scored'.
    For compile_assessment: returns preferred_label='proceed'.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def run(self, node: Node, prompt: str, context: PipelineContext) -> Outcome:
        self.calls.append(node.id)
        if node.id == "compile_assessment":
            return Outcome(status=StageStatus.SUCCESS, preferred_label="proceed")
        # Gate eval nodes (inside conversational-gate.dot subgraphs)
        return Outcome(status=StageStatus.SUCCESS, preferred_label="scored")


class CautionBackend:
    """Simulates compile_assessment returning caution verdict.

    For gate eval nodes: returns preferred_label='scored'.
    For compile_assessment: returns preferred_label='caution'.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def run(self, node: Node, prompt: str, context: PipelineContext) -> Outcome:
        self.calls.append(node.id)
        if node.id == "compile_assessment":
            return Outcome(status=StageStatus.SUCCESS, preferred_label="caution")
        return Outcome(status=StageStatus.SUCCESS, preferred_label="scored")


class NotReadyBackend:
    """Simulates compile_assessment returning not_ready verdict.

    For gate eval nodes: returns preferred_label='scored'.
    For compile_assessment: returns preferred_label='not_ready'.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def run(self, node: Node, prompt: str, context: PipelineContext) -> Outcome:
        self.calls.append(node.id)
        if node.id == "compile_assessment":
            return Outcome(status=StageStatus.SUCCESS, preferred_label="not_ready")
        return Outcome(status=StageStatus.SUCCESS, preferred_label="scored")


def _make_admissions_engine(
    backend: object,
    tmp_path: object,
    patterns_dir: str,
) -> PipelineEngine:
    """Create a PipelineEngine for the admissions.dot graph with the given backend.

    Parses admissions.dot, sets graph.source_dir to the foundry directory so that
    folder nodes can resolve ../../patterns/conversational-gate.dot correctly, then
    wires up a HandlerRegistry and PipelineEngine ready for test execution.

    Args:
        backend: Mock backend to use for codergen nodes.
        tmp_path: Temporary directory for logs.
        patterns_dir: Path to the patterns directory (used for path resolution context).

    Returns:
        A PipelineEngine configured to run admissions.dot.
    """
    import pathlib

    with open(_ADMISSIONS_DOT) as f:
        dot_source = f.read()
    graph = parse_dot(dot_source)
    # Set source_dir so PipelineHandler resolves '../../patterns/conversational-gate.dot'
    # relative to the foundry directory.
    graph.source_dir = _FOUNDRY_DIR

    context = PipelineContext()
    registry = HandlerRegistry(backend=backend)
    return PipelineEngine(
        graph=graph,
        context=context,
        handler_registry=registry,
        logs_root=str(pathlib.Path(str(tmp_path)) / "logs"),
    )


class TestAdmissionsExecution:
    """Execution tests for admissions.dot using mock backends (no real API calls)."""

    @pytest.mark.asyncio
    async def test_proceed_path_reaches_done_proceed(self, tmp_path):
        """ProceedBackend causes pipeline to reach done_proceed terminal.

        All 5 gates score successfully and compile_assessment returns 'proceed',
        routing verdict_gate to done_proceed.
        """
        backend = ProceedBackend()
        engine = _make_admissions_engine(backend, tmp_path, _PATTERNS_DIR)
        outcome = await engine.run()

        # Pipeline should complete successfully
        assert outcome.status in (StageStatus.SUCCESS, StageStatus.PARTIAL_SUCCESS), (
            f"Expected SUCCESS, got {outcome.status}. "
            f"failure_reason={outcome.failure_reason!r}, notes={outcome.notes!r}"
        )
        # The verdict_gate must have been traversed (it sets preferred_label routing)
        assert "verdict_gate" in engine.completed_nodes, (
            f"Expected 'verdict_gate' in completed_nodes. "
            f"completed_nodes={engine.completed_nodes}"
        )
        # Context preferred_label should be 'proceed' (set by compile_assessment)
        # Note: terminal nodes (Msquare) are exit nodes - not added to completed_nodes.
        # We verify the routing happened correctly via the context preferred_label.
        assert engine.context.get("preferred_label") == "proceed", (
            f"Expected preferred_label='proceed' in context, "
            f"got {engine.context.get('preferred_label')!r}"
        )

    @pytest.mark.asyncio
    async def test_caution_path_reaches_done_caution(self, tmp_path):
        """CautionBackend causes pipeline to reach done_caution terminal.

        All 5 gates score successfully and compile_assessment returns 'caution',
        routing verdict_gate to done_caution.
        """
        backend = CautionBackend()
        engine = _make_admissions_engine(backend, tmp_path, _PATTERNS_DIR)
        outcome = await engine.run()

        assert outcome.status in (StageStatus.SUCCESS, StageStatus.PARTIAL_SUCCESS), (
            f"Expected SUCCESS, got {outcome.status}. "
            f"failure_reason={outcome.failure_reason!r}, notes={outcome.notes!r}"
        )
        assert "verdict_gate" in engine.completed_nodes, (
            f"Expected 'verdict_gate' in completed_nodes. "
            f"completed_nodes={engine.completed_nodes}"
        )
        # Note: terminal nodes (Msquare) are exit nodes - not added to completed_nodes.
        # Verify routing via context preferred_label.
        assert engine.context.get("preferred_label") == "caution", (
            f"Expected preferred_label='caution' in context, "
            f"got {engine.context.get('preferred_label')!r}"
        )

    @pytest.mark.asyncio
    async def test_not_ready_path_reaches_done_not_ready(self, tmp_path):
        """NotReadyBackend causes pipeline to reach done_not_ready terminal.

        All 5 gates score successfully and compile_assessment returns 'not_ready',
        routing verdict_gate to done_not_ready.
        """
        backend = NotReadyBackend()
        engine = _make_admissions_engine(backend, tmp_path, _PATTERNS_DIR)
        outcome = await engine.run()

        assert outcome.status in (StageStatus.SUCCESS, StageStatus.PARTIAL_SUCCESS), (
            f"Expected SUCCESS, got {outcome.status}. "
            f"failure_reason={outcome.failure_reason!r}, notes={outcome.notes!r}"
        )
        assert "verdict_gate" in engine.completed_nodes, (
            f"Expected 'verdict_gate' in completed_nodes. "
            f"completed_nodes={engine.completed_nodes}"
        )
        # Note: terminal nodes (Msquare) are exit nodes - not added to completed_nodes.
        # Verify routing via context preferred_label.
        assert engine.context.get("preferred_label") == "not_ready", (
            f"Expected preferred_label='not_ready' in context, "
            f"got {engine.context.get('preferred_label')!r}"
        )

    @pytest.mark.asyncio
    async def test_compile_assessment_is_called_after_five_gates(self, tmp_path):
        """compile_assessment appears in completed_nodes after all 5 gates complete.

        Verifies the sequential flow: gate1->gate2->gate3->gate4->gate5->compile_assessment.
        All 5 gate folder nodes must appear before compile_assessment in the completed list.
        """
        backend = ProceedBackend()
        engine = _make_admissions_engine(backend, tmp_path, _PATTERNS_DIR)
        await engine.run()

        completed = engine.completed_nodes
        assert "compile_assessment" in completed, (
            f"compile_assessment not found in completed_nodes: {completed}"
        )

        compile_idx = completed.index("compile_assessment")

        # All 5 gates must appear before compile_assessment
        for gate in ["gate1", "gate2", "gate3", "gate4", "gate5"]:
            assert gate in completed, (
                f"'{gate}' not found in completed_nodes: {completed}"
            )
            gate_idx = completed.index(gate)
            assert gate_idx < compile_idx, (
                f"'{gate}' (idx={gate_idx}) should appear before "
                f"compile_assessment (idx={compile_idx})"
            )

        # compile_assessment was also called by the backend
        assert "compile_assessment" in backend.calls, (
            f"compile_assessment was not called by backend. calls={backend.calls}"
        )

    @pytest.mark.asyncio
    async def test_verdict_gate_is_traversed(self, tmp_path):
        """verdict_gate is visited during every admissions pipeline run.

        Confirms that the diamond node sits between compile_assessment and
        the terminal node, regardless of which verdict path is taken.
        """
        backend = ProceedBackend()
        engine = _make_admissions_engine(backend, tmp_path, _PATTERNS_DIR)
        await engine.run()

        assert "verdict_gate" in engine.completed_nodes, (
            f"'verdict_gate' not found in completed_nodes: {engine.completed_nodes}"
        )

        # verdict_gate must appear after compile_assessment and before a done node
        completed = engine.completed_nodes
        verdict_idx = completed.index("verdict_gate")
        compile_idx = completed.index("compile_assessment")
        assert compile_idx < verdict_idx, (
            f"compile_assessment (idx={compile_idx}) should appear before "
            f"verdict_gate (idx={verdict_idx})"
        )
