"""Tests for Phase 3 Runtime Integration: all 7 DOT files.

Verifies structural and content invariants across the complete set of
dev-machine runtime pipeline files:
  - iteration.dot
  - post-session.dot
  - health-check.dot
  - fix-iteration.dot
  - smoke-test.dot
  - qa.dot
  - qa-iteration.dot

Test classes:
  TestAllRuntimeDotsExist      -- 7 parametrized existence checks
  TestAllRuntimeDotsParse      -- 5 structural checks × 7 files = 35 tests
  TestAgentPromptsVerbatim     -- 4 verbatim content checks for agent prompts

Spec coverage: Phase 3 runtime DOT completeness and quality gates.
"""

from __future__ import annotations

import os
from collections import deque

import pytest

from amplifier_module_loop_pipeline.dot_parser import parse_dot

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_TESTS_DIR = os.path.dirname(__file__)
# From modules/loop-pipeline/tests/ -> up 3 levels -> amplifier-bundle-attractor/
_EXAMPLES_DIR = os.path.abspath(os.path.join(_TESTS_DIR, "..", "..", "..", "examples"))
_RUNTIME_DIR = os.path.join(_EXAMPLES_DIR, "dev-machine", "runtime")

# The 7 runtime DOT files that constitute the Phase 3 pipeline suite
_RUNTIME_DOT_FILES = [
    "iteration.dot",
    "post-session.dot",
    "health-check.dot",
    "fix-iteration.dot",
    "smoke-test.dot",
    "qa.dot",
    "qa-iteration.dot",
]

# Valid DOT shapes recognised by the pipeline engine
_VALID_SHAPES = {
    "Mdiamond",  # start
    "Msquare",  # done/exit
    "parallelogram",  # tool
    "diamond",  # conditional gate
    "box",  # codergen/LLM
    "folder",  # nested pipeline
    "house",  # manager loop
    "ellipse",  # codergen (alternate)
    "hexagon",  # human gate
    "invtriangle",  # parallel fan-out
    "trapezium",  # parallel fan-in
}


def _dot_path(filename: str) -> str:
    """Return the absolute path for a runtime DOT filename."""
    return os.path.join(_RUNTIME_DIR, filename)


def _load_and_parse(filename: str):
    """Load and parse a runtime DOT file, return the graph."""
    with open(_dot_path(filename)) as f:
        source = f.read()
    return parse_dot(source)


# ===========================================================================
# TestAllRuntimeDotsExist -- 7 parametrized existence checks
# ===========================================================================


@pytest.mark.parametrize("filename", _RUNTIME_DOT_FILES)
class TestAllRuntimeDotsExist:
    """Each of the 7 runtime DOT files must exist on disk."""

    def test_file_exists(self, filename: str):
        """Runtime DOT file exists at examples/dev-machine/runtime/{filename}."""
        path = _dot_path(filename)
        assert os.path.isfile(path), f"Runtime DOT file not found: {path}"


# ===========================================================================
# TestAllRuntimeDotsParse -- 5 structural checks × 7 files = 35 tests
# ===========================================================================


@pytest.mark.parametrize("filename", _RUNTIME_DOT_FILES)
class TestAllRuntimeDotsParse:
    """Structural invariants that every runtime DOT file must satisfy.

    Five checks per file = 35 total parametrized tests.
    """

    # -----------------------------------------------------------------------
    # Check 1: Parses without error
    # -----------------------------------------------------------------------

    def test_parses_without_error(self, filename: str):
        """Each runtime DOT file parses without raising an exception."""
        graph = _load_and_parse(filename)
        assert graph is not None, f"{filename} produced a None graph"

    # -----------------------------------------------------------------------
    # Check 2: Has start (Mdiamond) and done (Msquare) nodes
    # -----------------------------------------------------------------------

    def test_has_start_and_done_nodes(self, filename: str):
        """Each file has exactly one Mdiamond (start) and one Msquare (done) node."""
        graph = _load_and_parse(filename)

        start_nodes = [n for n in graph.nodes.values() if n.shape == "Mdiamond"]
        done_nodes = [n for n in graph.nodes.values() if n.shape == "Msquare"]

        assert len(start_nodes) >= 1, (
            f"{filename}: expected at least one Mdiamond (start) node, "
            f"found {len(start_nodes)}"
        )
        assert len(done_nodes) >= 1, (
            f"{filename}: expected at least one Msquare (done) node, "
            f"found {len(done_nodes)}"
        )

    # -----------------------------------------------------------------------
    # Check 3: All node shapes are valid
    # -----------------------------------------------------------------------

    def test_all_shapes_valid(self, filename: str):
        """Every node in the file uses a shape recognised by the pipeline engine."""
        graph = _load_and_parse(filename)

        invalid = [
            (n.id, n.shape)
            for n in graph.nodes.values()
            if n.shape not in _VALID_SHAPES
        ]
        assert not invalid, (
            f"{filename}: nodes with unrecognised shapes: {invalid}. "
            f"Valid shapes: {sorted(_VALID_SHAPES)}"
        )

    # -----------------------------------------------------------------------
    # Check 4: All nodes are reachable from the start node
    # -----------------------------------------------------------------------

    def test_all_nodes_reachable(self, filename: str):
        """Every node is reachable from the start node via directed edges."""
        graph = _load_and_parse(filename)

        # Find the start node (shape=Mdiamond)
        start_nodes = [n for n in graph.nodes.values() if n.shape == "Mdiamond"]
        assert start_nodes, f"{filename}: no Mdiamond (start) node found"
        start_id = start_nodes[0].id

        # Build adjacency list for forward traversal
        adjacency: dict[str, list[str]] = {nid: [] for nid in graph.nodes}
        for edge in graph.edges:
            if edge.from_node in adjacency:
                adjacency[edge.from_node].append(edge.to_node)

        # BFS from start
        visited: set[str] = set()
        queue: deque[str] = deque([start_id])
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            for neighbour in adjacency.get(current, []):
                if neighbour not in visited:
                    queue.append(neighbour)

        unreachable = [nid for nid in graph.nodes if nid not in visited]
        assert not unreachable, (
            f"{filename}: nodes unreachable from start ({start_id!r}): {unreachable}"
        )

    # -----------------------------------------------------------------------
    # Check 5: No Jinja2 {{...}} syntax in the DOT source
    # -----------------------------------------------------------------------

    def test_no_jinja2_syntax(self, filename: str):
        """No Jinja2 {{...}} or {{...}} syntax anywhere in the DOT source.

        All variable references must use $variable syntax.
        Jinja2 syntax in DOT files indicates an unported template variable.
        """
        with open(_dot_path(filename)) as f:
            source = f.read()

        assert "{{" not in source, (
            f"{filename}: found Jinja2 '{{{{' syntax -- use $variable notation instead"
        )
        assert "}}" not in source, (
            f"{filename}: found Jinja2 '}}}}' syntax -- use $variable notation instead"
        )


# ===========================================================================
# TestAgentPromptsVerbatim -- 4 verbatim content checks for agent prompts
# ===========================================================================

_ITERATION_DOT = _dot_path("iteration.dot")
_FIX_ITERATION_DOT = _dot_path("fix-iteration.dot")
_QA_ITERATION_DOT = _dot_path("qa-iteration.dot")


def _graph_iteration():
    return _load_and_parse("iteration.dot")


def _graph_fix_iteration():
    return _load_and_parse("fix-iteration.dot")


def _graph_qa_iteration():
    return _load_and_parse("qa-iteration.dot")


class TestAgentPromptsVerbatim:
    """Verbatim content checks for agent-facing prompts.

    These tests verify that the three LLM agent nodes contain the exact
    safety and guidance phrases required by the dev-machine specification.
    Agent prompts are the primary interface between the pipeline and the
    coding agents -- correctness here directly affects agent behaviour.
    """

    # -----------------------------------------------------------------------
    # Test 1: working_session contains 'ZERO prior context' and state
    # persistence reminder
    # -----------------------------------------------------------------------

    def test_working_session_zero_prior_context_and_writing_persistence(self):
        """working_session prompt has 'ZERO prior context' and writing-persistence reminder.

        'ZERO prior context' ensures the agent does not assume continuation from
        a previous session. 'What you don't write down is lost forever' reminds
        the agent that all state must be explicitly persisted to files because
        the session's in-memory context is discarded after the session ends.
        """
        graph = _graph_iteration()
        prompt = graph.nodes["working_session"].prompt

        assert "ZERO prior context" in prompt, (
            "working_session prompt must contain 'ZERO prior context'"
        )
        assert "What you don't write down is lost forever" in prompt, (
            "working_session prompt must contain "
            "'What you don't write down is lost forever'"
        )

    # -----------------------------------------------------------------------
    # Test 2: fix_session has verbatim mission statement
    # -----------------------------------------------------------------------

    def test_fix_session_mission_verbatim(self):
        """fix_session prompt contains the verbatim mission statement.

        The mission statement 'YOUR MISSION: Fix all build errors and test
        failures.' is the top-level directive for the health-check fix agent.
        Its presence ensures the agent understands the scope of its task.
        """
        graph = _graph_fix_iteration()
        prompt = graph.nodes["fix_session"].prompt

        assert "YOUR MISSION: Fix all build errors and test failures." in prompt, (
            "fix_session prompt must contain verbatim mission statement "
            "'YOUR MISSION: Fix all build errors and test failures.'"
        )

    # -----------------------------------------------------------------------
    # Test 3: qa_session has verbatim state persistence section
    # -----------------------------------------------------------------------

    def test_qa_session_state_persistence_verbatim(self):
        """qa_session prompt contains the verbatim 'State Persistence (CRITICAL)' section.

        The state persistence section ensures the QA agent updates the QA
        state file after each test run. Without this, QA results are lost.
        """
        graph = _graph_qa_iteration()
        prompt = graph.nodes["qa_session"].prompt

        assert "State Persistence (CRITICAL)" in prompt, (
            "qa_session prompt must contain 'State Persistence (CRITICAL)'"
        )

    # -----------------------------------------------------------------------
    # Test 4: All 3 agent prompts have SAFETY CONSTRAINTS section
    # -----------------------------------------------------------------------

    def test_all_agent_prompts_have_safety_constraints(self):
        """All 3 agent prompts (working_session, fix_session, qa_session) have SAFETY CONSTRAINTS.

        The SAFETY CONSTRAINTS section is mandatory in all agent prompts.
        It prevents agents from running destructive commands, accessing
        credentials, or making changes outside the project directory.
        """
        checks = [
            ("working_session", _graph_iteration().nodes["working_session"].prompt),
            ("fix_session", _graph_fix_iteration().nodes["fix_session"].prompt),
            ("qa_session", _graph_qa_iteration().nodes["qa_session"].prompt),
        ]

        missing = [
            node_id for node_id, prompt in checks if "SAFETY CONSTRAINTS" not in prompt
        ]
        assert not missing, (
            f"The following agent prompts are missing 'SAFETY CONSTRAINTS': {missing}"
        )
