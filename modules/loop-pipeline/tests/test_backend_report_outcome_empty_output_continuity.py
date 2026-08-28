"""Regression test for support#498: empty-output explicit report_outcome turns
must not silently vanish from the fidelity=full transcript.

Prior to this fix, a `fidelity="full"` node whose spawn result carried an
explicit `metadata.report_outcome` verdict AND an empty final `output` (the
child did its work via tool calls and ended its turn on a terminal
report_outcome, with no closing prose) was SKIPPED entirely at the issue-#287
guard (`backend.py`, the `if spawn_outcome is not None and
spawn_outcome.is_explicit:` branch's `output.strip()` condition). The turn
never entered `_thread_transcripts`, so a later same-thread spawn's
`parent_messages` silently omitted the actor's own prior attempt.

The fix synthesizes a compact, deterministic, non-empty assistant-content
string from the structured outcome (status / preferred_label / notes) instead
of dropping the exchange -- preserving continuity WITHOUT reintroducing the
empty assistant content that issue #287 guarded against (some providers
reject `{"role": "assistant", "content": ""}`).

This test is RED against the unfixed code (proven by stashing the production
edit and re-running) and GREEN with the fix applied.

Spec coverage: EXTENSIONS.md #12 (parent_messages node-exchange granularity),
#25 (is_explicit / report_outcome verdict transport), #35 (report_outcome
precedence policy).
"""

from typing import ClassVar

import pytest

from amplifier_module_loop_pipeline.backend import AmplifierBackend
from amplifier_module_loop_pipeline.context import PipelineContext
from amplifier_module_loop_pipeline.graph import Edge, Graph, Node

# ---------------------------------------------------------------------------
# Test helpers (self-contained per this repo's test-file convention -- no
# test file here imports from another test file).
# ---------------------------------------------------------------------------


class _MockSession:
    config: ClassVar[dict] = {}


class _SpawnCapture:
    """Coordinator mock that captures every spawn call's kwargs and returns a
    queued sequence of spawn results (one per call), so a test can drive a
    child that reports an explicit verdict with empty output on turn 1 and a
    normal output on turn 2.
    """

    def __init__(self, results: list[dict]):
        self._results = list(results)
        self.calls: list[dict] = []
        self.session = _MockSession()
        # 'attractor-anthropic' with a non-pipeline session.orchestrator so the
        # identity recursion guard in _run_with_spawn does not fire.
        self.config: dict = {
            "agents": {
                "attractor-anthropic": {
                    "session": {"orchestrator": {"module": "loop-agent"}},
                },
            }
        }

    def get_capability(self, name: str):
        if name == "session.spawn":
            return self._spawn_fn
        return None

    async def _spawn_fn(self, **kwargs):
        self.calls.append(dict(kwargs))
        idx = len(self.calls) - 1
        return self._results[idx]


def _make_full_node(node_id: str, thread_id: str = "main") -> Node:
    return Node(
        id=node_id,
        prompt="Do work",
        attrs={"llm_provider": "anthropic", "fidelity": "full", "thread_id": thread_id},
    )


def _make_graph_with_full_nodes(*node_ids: str, thread_id: str = "main") -> Graph:
    nodes: dict[str, Node] = {"start": Node(id="start", shape="Mdiamond")}
    for nid in node_ids:
        nodes[nid] = _make_full_node(nid, thread_id=thread_id)
    nodes["exit"] = Node(id="exit", shape="Msquare")

    edges: list[Edge] = [Edge(from_node="start", to_node=node_ids[0])]
    for i in range(len(node_ids) - 1):
        edges.append(Edge(from_node=node_ids[i], to_node=node_ids[i + 1]))
    edges.append(Edge(from_node=node_ids[-1], to_node="exit"))

    return Graph(name="test", nodes=nodes, edges=edges, graph_attrs={})


def _make_context() -> PipelineContext:
    ctx = PipelineContext()
    ctx.set("graph.goal", "Test report_outcome continuity")
    return ctx


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_explicit_report_outcome_with_empty_output_preserves_transcript_turn():
    """A full-fidelity node that reports an explicit verdict with NO closing
    prose must still leave a non-empty (instruction, output) turn in the
    thread transcript -- and a later same-thread node must see it via
    parent_messages with non-empty assistant content.

    Turn 1 (node_a): spawn result carries metadata.report_outcome (explicit
    verdict) AND output="" (empty closing prose -- the pre-#287-guard-fix
    failure mode). Before the fix: _append_to_transcript is never called for
    this node, so _thread_transcripts has NO entry for node_a. After the fix:
    an entry exists with synthesized non-empty assistant content.

    Turn 2 (node_b, same thread): parent_messages must contain exactly one
    prior exchange (2 messages: user + assistant), and the assistant message's
    content must be non-empty -- proving continuity was preserved without
    reintroducing empty assistant content (the exact hazard issue #287
    guarded against).
    """
    coordinator = _SpawnCapture(
        results=[
            {
                "output": "",
                "session_id": "sess-A",
                "metadata": {
                    "report_outcome": {
                        "status": "success",
                        "preferred_label": "converged",
                        "notes": "wrote the files via tool calls",
                    }
                },
            },
            {"output": "Second node output", "session_id": "sess-B"},
        ]
    )
    backend = AmplifierBackend(
        coordinator=coordinator,
        profiles={"anthropic": "attractor-anthropic"},
    )

    graph = _make_graph_with_full_nodes("node_a", "node_b", thread_id="T")
    context = _make_context()
    edge_to_a = Edge(from_node="start", to_node="node_a")
    edge_to_b = Edge(from_node="node_a", to_node="node_b")

    # --- Turn 1: node_a reports an explicit verdict with empty output ---
    outcome_a = await backend.run(
        graph.nodes["node_a"],
        "First instruction",
        context,
        incoming_edge=edge_to_a,
        graph=graph,
    )

    # The verdict itself must be untouched by the fix: still the explicit
    # SUCCESS/converged verdict from metadata.report_outcome.
    assert outcome_a.is_explicit is True
    assert outcome_a.preferred_label == "converged"

    # --- The core assertion: node_a's turn must be IN the transcript ---
    turns = backend._thread_transcripts.get("T", [])
    assert len(turns) == 1, (
        f"Expected exactly one transcript turn for node_a's empty-output "
        f"explicit-verdict exchange (support#498). Got {len(turns)} turns: "
        f"{turns!r}. On unfixed code this is 0 -- the issue-#287 guard's "
        f"`output.strip()` condition skips the append entirely when output "
        f"is empty, even though the verdict was explicit."
    )
    node_id, stored_instruction, stored_output = turns[0]
    assert node_id == "node_a"
    assert stored_instruction == "First instruction"
    assert stored_output.strip() != "", (
        "The synthesized transcript output must be non-empty -- an empty "
        "string here would reintroduce the exact hazard issue #287 guarded "
        "against ({'role': 'assistant', 'content': ''}, which some "
        "providers reject)."
    )

    # --- Turn 2: node_b runs on the same thread ---
    outcome_b = await backend.run(
        graph.nodes["node_b"],
        "Second instruction",
        context,
        incoming_edge=edge_to_b,
        graph=graph,
    )
    assert outcome_b is not None  # sanity: node_b actually ran

    assert len(coordinator.calls) == 2
    second_call = coordinator.calls[1]
    assert "parent_messages" in second_call, (
        "node_b must receive parent_messages carrying node_a's exchange -- "
        "support#498's continuity hole."
    )
    pm = second_call["parent_messages"]
    assert len(pm) == 2, (
        f"Exactly one prior exchange (node_a) = 2 messages (user + "
        f"assistant). Got {len(pm)}: {pm}"
    )
    assert pm[0] == {"role": "user", "content": "First instruction"}
    assert pm[1]["role"] == "assistant"
    assert pm[1]["content"].strip() != "", (
        "node_b's parent_messages must carry a non-empty assistant message "
        "for node_a's turn -- an empty string is the exact provider-rejected "
        "shape issue #287 guarded against, and support#498's whole point is "
        "that continuity must be preserved WITHOUT reintroducing it."
    )


@pytest.mark.asyncio
async def test_status_only_empty_output_completion_preserves_transcript_turn():
    """Same continuity hole, reached via the OTHER empty-output path: the
    outcome is recovered from the orchestrator's lifecycle completion status
    alone (no `report_outcome` tool call at all -- e.g. a clean completion
    with empty closing text), not from an explicit verdict.

    Before the fix this path (the `if spawn_outcome is not None:` branch
    inside `if not output.strip():`) returned the recovered Outcome WITHOUT
    ever calling `_append_to_transcript` -- unconditionally, regardless of
    fidelity/thread_key -- so a full-fidelity thread lost this node's turn
    exactly as in the explicit-verdict case. The fix applies the same
    `_synthesize_outcome_output` treatment here, gated on
    fidelity=="full"/graph/thread_key like every other append site.
    """
    coordinator = _SpawnCapture(
        results=[
            {
                "output": "",
                "session_id": "sess-A",
                "status": "success",
                # No metadata.report_outcome -- lifecycle status is the ONLY
                # outcome signal, so _outcome_from_spawn_result recovers an
                # is_explicit=False Outcome via _SPAWN_SUCCESS_STATUSES.
            },
            {"output": "Second node output", "session_id": "sess-B"},
        ]
    )
    backend = AmplifierBackend(
        coordinator=coordinator,
        profiles={"anthropic": "attractor-anthropic"},
    )

    graph = _make_graph_with_full_nodes("node_a", "node_b", thread_id="T")
    context = _make_context()
    edge_to_a = Edge(from_node="start", to_node="node_a")
    edge_to_b = Edge(from_node="node_a", to_node="node_b")

    outcome_a = await backend.run(
        graph.nodes["node_a"],
        "First instruction",
        context,
        incoming_edge=edge_to_a,
        graph=graph,
    )

    # Recovered via lifecycle status only -- NOT an explicit verdict. The
    # fix must not change this classification.
    assert outcome_a.is_explicit is False
    assert outcome_a.is_success is True

    turns = backend._thread_transcripts.get("T", [])
    assert len(turns) == 1, (
        f"Expected exactly one transcript turn for node_a's status-only "
        f"empty-output completion (support#498). Got {len(turns)} turns: "
        f"{turns!r}. On unfixed code this is 0 -- this branch never calls "
        f"_append_to_transcript at all."
    )
    node_id, stored_instruction, stored_output = turns[0]
    assert node_id == "node_a"
    assert stored_instruction == "First instruction"
    assert stored_output.strip() != "", (
        "The synthesized transcript output must be non-empty for the same "
        "provider-compatibility reason as the explicit-verdict case."
    )

    await backend.run(
        graph.nodes["node_b"],
        "Second instruction",
        context,
        incoming_edge=edge_to_b,
        graph=graph,
    )
    second_call = coordinator.calls[1]
    pm = second_call.get("parent_messages", [])
    assert len(pm) == 2, (
        f"node_b must see node_a's exchange via parent_messages. Got {len(pm)}: {pm}"
    )
    assert pm[1]["role"] == "assistant"
    assert pm[1]["content"].strip() != ""
