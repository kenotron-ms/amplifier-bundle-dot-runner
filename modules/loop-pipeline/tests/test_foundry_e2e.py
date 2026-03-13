"""End-to-End Tests for the Foundry -> Runtime Chain.

Proves that the foundry->runtime chain works:
- All 3 foundry DOT files exist and parse
- Chain continuity: admissions -> machine-design -> generate-machine via file checks
- Minimal runtime DOT content is valid DOT syntax
- Structural validation of generated runtime DOTs

Test file: modules/loop-pipeline/tests/test_foundry_e2e.py

Chain continuity:
  admissions.compile_assessment  writes  .dev-machine-assessment.md
  machine-design.assessment_check checks  .dev-machine-assessment.md

  machine-design.phase5           writes  .dev-machine-design.md
  generate-machine.design_check  checks  .dev-machine-design.md
"""

from __future__ import annotations

import os

from amplifier_module_loop_pipeline.dot_parser import parse_dot

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_TESTS_DIR = os.path.dirname(__file__)
# From modules/loop-pipeline/tests/ -> up 3 levels -> amplifier-bundle-attractor/ -> examples/
_EXAMPLES_DIR = os.path.abspath(os.path.join(_TESTS_DIR, "..", "..", "..", "examples"))
_FOUNDRY_DIR = os.path.join(_EXAMPLES_DIR, "dev-machine", "foundry")
_RUNTIME_DIR = os.path.join(_EXAMPLES_DIR, "dev-machine", "runtime")

_ADMISSIONS_DOT = os.path.join(_FOUNDRY_DIR, "admissions.dot")
_MACHINE_DESIGN_DOT = os.path.join(_FOUNDRY_DIR, "machine-design.dot")
_GENERATE_MACHINE_DOT = os.path.join(_FOUNDRY_DIR, "generate-machine.dot")

# ---------------------------------------------------------------------------
# Minimal DOT content fixtures
# These are minimal valid DOT files matching the structure the foundry
# would generate, used for structural validation tests.
# ---------------------------------------------------------------------------

MINIMAL_ITERATION_DOT = """\
digraph Iteration {
    graph [label="Dev Machine Iteration Pipeline"]
    rankdir=TB

    start [shape=Mdiamond, label="Start"]
    done  [shape=Msquare,  label="Done"]

    orient [
        shape=parallelogram,
        label="Orient",
        tool_command="python3 scripts/pipeline/orient.py $state_file",
        parse_json="true"
    ]

    orient_gate [
        shape=diamond,
        label="Blocked?"
    ]

    working_session [
        shape=box,
        context_fidelity="truncate",
        label="Working Session",
        prompt="You are a WORKING SESSION of the development machine."
    ]

    build_check [
        shape=parallelogram,
        label="Build Check",
        tool_command="python3 scripts/pipeline/build-check.py $project_dir $build_command $build_timeout",
        parse_json="true"
    ]

    build_gate [
        shape=diamond,
        label="Build OK?"
    ]

    post_session [
        shape=folder,
        label="Post Session",
        dot_file="post-session.dot"
    ]

    start           -> orient
    orient          -> orient_gate
    orient_gate     -> done          [label="blocked", condition="context.status=blocked"]
    orient_gate     -> working_session [label="healthy", condition="context.status=healthy"]
    working_session -> build_check
    build_check     -> build_gate
    build_gate      -> post_session  [label="clean",  condition="context.build_status=clean"]
    build_gate      -> post_session  [label="failed", condition="context.build_status=failed"]
    post_session    -> start
}
"""

MINIMAL_POST_SESSION_DOT = """\
digraph PostSession {
    graph [label="Dev Machine Post-Session Sub-Pipeline"]
    rankdir=TB

    start [shape=Mdiamond, label="Start"]
    done  [shape=Msquare,  label="Done"]

    archive [
        shape=parallelogram,
        label="Archive Features",
        tool_command="python3 scripts/pipeline/post-session-archive.py $state_file $context_file"
    ]

    reconcile [
        shape=parallelogram,
        label="Reconcile",
        tool_command="python3 scripts/pipeline/post-session-reconcile.py $state_file $context_file",
        continue_on_fail="true"
    ]

    start   -> archive
    archive -> reconcile
    reconcile -> done
}
"""

MINIMAL_HEALTH_CHECK_DOT = """\
digraph HealthCheck {
    graph [label="Dev Machine Health Check Pipeline"]
    rankdir=TB

    start [shape=Mdiamond, label="Start"]
    done  [shape=Msquare,  label="Done"]

    initial_check [
        shape=parallelogram,
        label="Initial Check",
        tool_command="bash -c 'echo \"{\\\"status\\\": \\\"clean\\\"}\"'",
        parse_json="true",
        continue_on_fail="true"
    ]

    clean_gate [
        shape=diamond,
        label="Clean?"
    ]

    fix_loop [
        shape=house,
        label="Fix Loop",
        manager.max_cycles="$max_fix_iterations",
        manager.stop_condition="outcome=success",
        stack.child_dotfile="fix-iteration.dot"
    ]

    start         -> initial_check
    initial_check -> clean_gate
    clean_gate    -> done     [label="clean",  condition="context.status=clean"]
    clean_gate    -> fix_loop [label="failed", condition="context.status=needs-fixing"]
    fix_loop      -> done
}
"""

MINIMAL_FIX_ITERATION_DOT = """\
digraph FixIteration {
    graph [label="Dev Machine Fix Iteration Sub-Pipeline"]
    rankdir=TB

    start [shape=Mdiamond, label="Start"]
    done  [shape=Msquare,  label="Done"]

    read_errors [
        shape=parallelogram,
        label="Read Errors",
        tool_command="bash -c 'echo \"{\\\"iteration\\\": \\\"1\\\"}\"'",
        parse_json="true",
        continue_on_fail="true"
    ]

    fix_session [
        shape=box,
        context_fidelity="truncate",
        label="Fix Session",
        prompt="You are a HEALTH CHECK FIX SESSION. Fix all build errors and test failures."
    ]

    verify [
        shape=parallelogram,
        label="Verify",
        tool_command="bash -c 'echo \"{\\\"status\\\": \\\"done\\\"}\"'",
        parse_json="true",
        continue_on_fail="true"
    ]

    start       -> read_errors
    read_errors -> fix_session
    fix_session -> verify
    verify      -> done
}
"""

MINIMAL_SMOKE_TEST_DOT = """\
digraph SmokeTest {
    graph [label="Dev Machine Pre-Flight Smoke Test"]
    rankdir=TB

    start [shape=Mdiamond, label="Start"]
    done  [shape=Msquare,  label="Done"]

    check_files [
        shape=parallelogram,
        label="Check Files",
        tool_command="python3 scripts/pipeline/smoke-check-files.py",
        continue_on_fail="true"
    ]

    check_dots [
        shape=parallelogram,
        label="Check DOTs",
        tool_command="python3 scripts/pipeline/smoke-check-dot-validity.py",
        continue_on_fail="true"
    ]

    check_scripts [
        shape=parallelogram,
        label="Check Scripts",
        tool_command="python3 scripts/pipeline/smoke-check-scripts.py",
        continue_on_fail="true"
    ]

    check_infra [
        shape=parallelogram,
        label="Check Infra",
        tool_command="python3 scripts/pipeline/smoke-check-infra.py",
        continue_on_fail="true"
    ]

    check_state [
        shape=parallelogram,
        label="Check State",
        tool_command="python3 scripts/pipeline/smoke-check-state.py",
        continue_on_fail="true"
    ]

    summary [
        shape=parallelogram,
        label="Smoke Summary",
        tool_command="python3 scripts/pipeline/smoke-summary.py"
    ]

    start        -> check_files
    check_files  -> check_dots
    check_dots   -> check_scripts
    check_scripts -> check_infra
    check_infra  -> check_state
    check_state  -> summary
    summary      -> done
}
"""

# ---------------------------------------------------------------------------
# ARTIFACT_MAP: maps artifact names to their minimal DOT content
# ---------------------------------------------------------------------------

ARTIFACT_MAP: dict[str, str] = {
    "iteration.dot": MINIMAL_ITERATION_DOT,
    "post-session.dot": MINIMAL_POST_SESSION_DOT,
    "health-check.dot": MINIMAL_HEALTH_CHECK_DOT,
    "fix-iteration.dot": MINIMAL_FIX_ITERATION_DOT,
    "smoke-test.dot": MINIMAL_SMOKE_TEST_DOT,
}

# Known pattern file names that foundry folder nodes may reference
_KNOWN_PATTERNS = {
    "conversational-gate.dot",
    "convergence-factory.dot",
    "iteration.dot",
    "post-session.dot",
    "health-check.dot",
    "fix-iteration.dot",
    "smoke-test.dot",
    "qa-iteration.dot",
    "qa.dot",
}


# ===========================================================================
# TestFoundryChainStructure -- E2E tests for the foundry -> runtime chain
# ===========================================================================


class TestFoundryChainStructure:
    """End-to-end structural tests for the Foundry -> Runtime chain.

    Tests verify that:
    1. All 3 foundry DOT files exist and parse
    2. The chain continuity is correct (each step feeds the next)
    3. All minimal runtime DOT content is valid
    4. Required nodes are present in minimal DOT fixtures
    5. All foundry folder dot_file references point to known patterns
    """

    # -----------------------------------------------------------------------
    # AC-1: All 3 foundry DOT files exist
    # -----------------------------------------------------------------------

    def test_all_three_foundry_dots_exist(self):
        """All 3 foundry DOT files exist at their expected paths."""
        for dot_path, name in [
            (_ADMISSIONS_DOT, "admissions.dot"),
            (_MACHINE_DESIGN_DOT, "machine-design.dot"),
            (_GENERATE_MACHINE_DOT, "generate-machine.dot"),
        ]:
            assert os.path.isfile(dot_path), f"{name} not found at {dot_path}"

    # -----------------------------------------------------------------------
    # AC-2: All 3 foundry DOT files parse without error
    # -----------------------------------------------------------------------

    def test_all_three_foundry_dots_parse(self):
        """All 3 foundry DOT files parse without raising exceptions."""
        for dot_path, name in [
            (_ADMISSIONS_DOT, "admissions.dot"),
            (_MACHINE_DESIGN_DOT, "machine-design.dot"),
            (_GENERATE_MACHINE_DOT, "generate-machine.dot"),
        ]:
            with open(dot_path) as f:
                content = f.read()
            graph = parse_dot(content)
            assert graph is not None, f"{name} parsed to None"
            assert len(graph.nodes) > 0, f"{name} parsed with 0 nodes"

    # -----------------------------------------------------------------------
    # AC-3: Chain continuity -- admissions -> machine-design
    # compile_assessment writes .dev-machine-assessment.md;
    # assessment_check in machine-design checks for it.
    # -----------------------------------------------------------------------

    def test_foundry_chain_output_of_admissions_feeds_machine_design(self):
        """admissions.compile_assessment writes .dev-machine-assessment.md;
        machine-design.assessment_check checks for .dev-machine-assessment.md.

        Verifies the chain continuity: the file that admissions produces
        is exactly the file that machine-design checks for.
        """
        # -- Admissions side: compile_assessment must mention .dev-machine-assessment.md --
        with open(_ADMISSIONS_DOT) as f:
            admissions_graph = parse_dot(f.read())

        compile_node = admissions_graph.nodes.get("compile_assessment")
        assert compile_node is not None, (
            "compile_assessment node not found in admissions.dot"
        )
        prompt = compile_node.prompt or ""
        assert ".dev-machine-assessment.md" in prompt, (
            f"compile_assessment prompt should reference .dev-machine-assessment.md. "
            f"Prompt (first 400 chars): {prompt[:400]}"
        )

        # -- Machine-design side: assessment_check must reference the same file --
        with open(_MACHINE_DESIGN_DOT) as f:
            machine_design_graph = parse_dot(f.read())

        check_node = machine_design_graph.nodes.get("assessment_check")
        assert check_node is not None, (
            "assessment_check node not found in machine-design.dot"
        )
        tool_command = check_node.attrs.get("tool_command", "")
        assert ".dev-machine-assessment.md" in tool_command, (
            f"assessment_check tool_command should check for .dev-machine-assessment.md. "
            f"Got: {tool_command!r}"
        )

    # -----------------------------------------------------------------------
    # AC-4: Chain continuity -- machine-design -> generate-machine
    # machine-design phase5 writes .dev-machine-design.md;
    # design_check in generate-machine checks for it.
    # -----------------------------------------------------------------------

    def test_foundry_chain_output_of_machine_design_feeds_generate_machine(self):
        """machine-design writes .dev-machine-design.md (via phase5);
        generate-machine.design_check checks for .dev-machine-design.md.

        Verifies the chain continuity: the file that machine-design produces
        is exactly the file that generate-machine checks for.
        """
        # -- Machine-design side: phase5 folder must mention .dev-machine-design.md --
        with open(_MACHINE_DESIGN_DOT) as f:
            machine_design_graph = parse_dot(f.read())

        folder_nodes = [
            n for n in machine_design_graph.nodes.values() if n.shape == "folder"
        ]
        phase5_nodes = [n for n in folder_nodes if "phase5" in n.id.lower()]
        assert len(phase5_nodes) > 0, (
            f"Expected a phase5 folder node in machine-design.dot. "
            f"Node IDs: {[n.id for n in folder_nodes]}"
        )
        assert any(
            ".dev-machine-design.md" in n.attrs.get("context.artifact_goal", "")
            for n in phase5_nodes
        ), (
            f"Expected phase5 artifact_goal to mention .dev-machine-design.md. "
            f"Goals: {[n.attrs.get('context.artifact_goal', '')[:200] for n in phase5_nodes]}"
        )

        # -- Generate-machine side: design_check must reference the same file --
        with open(_GENERATE_MACHINE_DOT) as f:
            generate_machine_graph = parse_dot(f.read())

        design_check_node = generate_machine_graph.nodes.get("design_check")
        assert design_check_node is not None, (
            "design_check node not found in generate-machine.dot"
        )
        tool_command = design_check_node.attrs.get("tool_command", "")
        assert ".dev-machine-design.md" in tool_command, (
            f"design_check tool_command should check for .dev-machine-design.md. "
            f"Got: {tool_command!r}"
        )

    # -----------------------------------------------------------------------
    # AC-5: All minimal DOT content is valid DOT syntax
    # -----------------------------------------------------------------------

    def test_generate_machine_artifact_paths_produce_valid_dot(self):
        """All minimal DOT content in ARTIFACT_MAP is valid DOT syntax (parses cleanly)."""
        for artifact_name, dot_content in ARTIFACT_MAP.items():
            graph = parse_dot(dot_content)
            assert graph is not None, f"ARTIFACT_MAP[{artifact_name!r}] parsed to None"
            assert len(graph.nodes) > 0, (
                f"ARTIFACT_MAP[{artifact_name!r}] parsed with 0 nodes"
            )

    # -----------------------------------------------------------------------
    # AC-6: Minimal iteration.dot has required nodes
    # -----------------------------------------------------------------------

    def test_minimal_iteration_dot_has_required_nodes(self):
        """MINIMAL_ITERATION_DOT has required nodes: start, orient, working_session, build_check, done."""
        graph = parse_dot(MINIMAL_ITERATION_DOT)
        required_nodes = ["start", "orient", "working_session", "build_check", "done"]
        node_ids = set(graph.nodes.keys())
        for node_id in required_nodes:
            assert node_id in node_ids, (
                f"Required node {node_id!r} not found in MINIMAL_ITERATION_DOT. "
                f"Nodes: {sorted(node_ids)}"
            )

    # -----------------------------------------------------------------------
    # AC-7: Minimal smoke-test.dot has 5+ tool check nodes (parallelogram)
    # -----------------------------------------------------------------------

    def test_minimal_smoke_test_dot_has_six_check_nodes(self):
        """MINIMAL_SMOKE_TEST_DOT has 5+ parallelogram check nodes."""
        graph = parse_dot(MINIMAL_SMOKE_TEST_DOT)
        parallelogram_nodes = [
            n for n in graph.nodes.values() if n.shape == "parallelogram"
        ]
        assert len(parallelogram_nodes) >= 5, (
            f"Expected 5+ parallelogram check nodes in MINIMAL_SMOKE_TEST_DOT, "
            f"got {len(parallelogram_nodes)}: "
            f"{[n.id for n in parallelogram_nodes]}"
        )

    # -----------------------------------------------------------------------
    # AC-8: generate-machine's gen_iteration artifact_goal mentions orient node
    # -----------------------------------------------------------------------

    def test_generate_machine_iteration_artifact_goal_mentions_orient_node(self):
        """generate-machine.gen_iteration context.artifact_goal describes the orient node."""
        with open(_GENERATE_MACHINE_DOT) as f:
            generate_machine_graph = parse_dot(f.read())

        gen_iteration = generate_machine_graph.nodes.get("gen_iteration")
        assert gen_iteration is not None, (
            "gen_iteration node not found in generate-machine.dot"
        )
        artifact_goal = gen_iteration.attrs.get("context.artifact_goal", "")
        assert "orient" in artifact_goal.lower(), (
            f"Expected gen_iteration artifact_goal to mention 'orient' node. "
            f"artifact_goal (first 400 chars): {artifact_goal[:400]}"
        )

    # -----------------------------------------------------------------------
    # AC-9: Valid iteration.dot passes smoke-test structural validation
    # -----------------------------------------------------------------------

    def test_smoke_test_dot_validation_of_generated_iteration(self):
        """Valid iteration.dot would pass smoke-test DOT structural validation.

        Verifies that:
        1. The actual examples/dev-machine/runtime/iteration.dot parses cleanly
        2. The actual examples/dev-machine/runtime/smoke-test.dot parses cleanly
        Both are required for the smoke-test to validate generated iteration files.
        """
        # Verify actual iteration.dot parses cleanly
        iteration_dot = os.path.join(_RUNTIME_DIR, "iteration.dot")
        assert os.path.isfile(iteration_dot), (
            f"runtime/iteration.dot not found at {iteration_dot}"
        )
        with open(iteration_dot) as f:
            iteration_graph = parse_dot(f.read())
        assert iteration_graph is not None, "runtime/iteration.dot parsed to None"
        assert len(iteration_graph.nodes) > 0, (
            "runtime/iteration.dot parsed with 0 nodes"
        )

        # Verify actual smoke-test.dot parses cleanly
        smoke_test_dot = os.path.join(_RUNTIME_DIR, "smoke-test.dot")
        assert os.path.isfile(smoke_test_dot), (
            f"runtime/smoke-test.dot not found at {smoke_test_dot}"
        )
        with open(smoke_test_dot) as f:
            smoke_graph = parse_dot(f.read())
        assert smoke_graph is not None, "runtime/smoke-test.dot parsed to None"
        assert len(smoke_graph.nodes) > 0, "runtime/smoke-test.dot parsed with 0 nodes"

        # Verify smoke-test has parallelogram check nodes that would validate DOTs
        parallelogram_nodes = [
            n for n in smoke_graph.nodes.values() if n.shape == "parallelogram"
        ]
        assert len(parallelogram_nodes) >= 5, (
            f"smoke-test.dot should have 5+ parallelogram check nodes for validation. "
            f"Got {len(parallelogram_nodes)}: "
            f"{[n.id for n in parallelogram_nodes]}"
        )

        # Verify iteration.dot has all required nodes for a valid iteration pipeline
        required_iteration_nodes = [
            "start",
            "orient",
            "working_session",
            "build_check",
            "done",
        ]
        iteration_node_ids = set(iteration_graph.nodes.keys())
        for node_id in required_iteration_nodes:
            assert node_id in iteration_node_ids, (
                f"Required node {node_id!r} not found in runtime/iteration.dot. "
                f"Nodes: {sorted(iteration_node_ids)}"
            )

    # -----------------------------------------------------------------------
    # AC-10: All foundry folder dot_file refs point to known patterns
    # -----------------------------------------------------------------------

    def test_all_foundry_patterns_are_relative_to_patterns_dir(self):
        """All folder dot_file attributes in foundry DOTs reference known patterns.

        Known patterns: conversational-gate.dot, convergence-factory.dot,
        or runtime DOT files (iteration.dot, post-session.dot, etc.).
        """
        for dot_path, name in [
            (_ADMISSIONS_DOT, "admissions.dot"),
            (_MACHINE_DESIGN_DOT, "machine-design.dot"),
            (_GENERATE_MACHINE_DOT, "generate-machine.dot"),
        ]:
            with open(dot_path) as f:
                graph = parse_dot(f.read())

            folder_nodes = [n for n in graph.nodes.values() if n.shape == "folder"]
            for node in folder_nodes:
                dot_file = node.attrs.get("dot_file", "")
                assert dot_file, (
                    f"Node {node.id!r} in {name} has shape=folder but missing dot_file attr"
                )
                # Verify the dot_file references a known pattern
                # (the filename portion must match a known pattern name)
                dot_filename = os.path.basename(dot_file)
                assert dot_filename in _KNOWN_PATTERNS, (
                    f"Node {node.id!r} in {name} dot_file={dot_file!r} "
                    f"references unknown pattern {dot_filename!r}. "
                    f"Known patterns: {sorted(_KNOWN_PATTERNS)}"
                )
