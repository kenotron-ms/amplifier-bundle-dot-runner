"""Tests for hooks wiring from orchestrator to pipeline engine (Task 1.6).

Verifies that PipelineOrchestrator.execute() passes the hooks parameter
through to PipelineEngine so pipeline events are actually emitted.

Spec coverage: EVT-001-008, Section 9.6.
"""

import json

import pytest
from unittest.mock import AsyncMock, MagicMock

from amplifier_module_loop_pipeline import PipelineOrchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SIMPLE_DOT = """
digraph {
    start [shape=Mdiamond]
    step [prompt="Do something"]
    exit [shape=Msquare]
    start -> step -> exit
}
"""


class MockBackend:
    """Backend that returns a fixed outcome for every call."""

    async def run(self, node, prompt, context, **kwargs):
        return "done"


def _make_hooks():
    hooks = MagicMock()
    hooks._emitted = []  # list of (event_name, data) tuples

    async def _recording_emit(event: str, data: dict):
        hooks._emitted.append((event, data))
        return MagicMock(action="continue")

    hooks.emit = AsyncMock(side_effect=_recording_emit)
    return hooks


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_emits_start_event_via_hooks(tmp_path):
    """Pipeline orchestrator must pass hooks to engine so pipeline:start fires."""
    orchestrator = PipelineOrchestrator(
        config={"dot_source": SIMPLE_DOT, "logs_root": str(tmp_path)}
    )
    hooks = _make_hooks()

    await orchestrator.execute(
        prompt="test goal",
        context=MagicMock(),
        providers={"mock": MagicMock()},
        tools={},
        hooks=hooks,
        backend=MockBackend(),
    )

    emit_events = [e[0] for e in hooks._emitted]
    assert "pipeline:start" in emit_events


@pytest.mark.asyncio
async def test_pipeline_emits_complete_event_via_hooks(tmp_path):
    """Pipeline orchestrator must pass hooks to engine so pipeline:complete fires."""
    orchestrator = PipelineOrchestrator(
        config={"dot_source": SIMPLE_DOT, "logs_root": str(tmp_path)}
    )
    hooks = _make_hooks()

    await orchestrator.execute(
        prompt="test goal",
        context=MagicMock(),
        providers={"mock": MagicMock()},
        tools={},
        hooks=hooks,
        backend=MockBackend(),
    )

    emit_events = [e[0] for e in hooks._emitted]
    assert "pipeline:complete" in emit_events


@pytest.mark.asyncio
async def test_pipeline_emits_node_events_via_hooks(tmp_path):
    """Pipeline must emit node_start and node_complete for each executed node."""
    orchestrator = PipelineOrchestrator(
        config={"dot_source": SIMPLE_DOT, "logs_root": str(tmp_path)}
    )
    hooks = _make_hooks()

    await orchestrator.execute(
        prompt="test goal",
        context=MagicMock(),
        providers={"mock": MagicMock()},
        tools={},
        hooks=hooks,
        backend=MockBackend(),
    )

    emit_events = [e[0] for e in hooks._emitted]
    assert "pipeline:node_start" in emit_events
    assert "pipeline:node_complete" in emit_events


@pytest.mark.asyncio
async def test_pipeline_emits_edge_selected_via_hooks(tmp_path):
    """Pipeline must emit pipeline:edge_selected when traversing edges."""
    orchestrator = PipelineOrchestrator(
        config={"dot_source": SIMPLE_DOT, "logs_root": str(tmp_path)}
    )
    hooks = _make_hooks()

    await orchestrator.execute(
        prompt="test goal",
        context=MagicMock(),
        providers={"mock": MagicMock()},
        tools={},
        hooks=hooks,
        backend=MockBackend(),
    )

    emit_events = [e[0] for e in hooks._emitted]
    assert "pipeline:edge_selected" in emit_events


@pytest.mark.asyncio
async def test_pipeline_without_hooks_still_works(tmp_path):
    """Pipeline still works when hooks is None (backward compatibility)."""
    orchestrator = PipelineOrchestrator(
        config={"dot_source": SIMPLE_DOT, "logs_root": str(tmp_path)}
    )

    result = await orchestrator.execute(
        prompt="test goal",
        context=MagicMock(),
        providers={"mock": MagicMock()},
        tools={},
        hooks=None,
        backend=MockBackend(),
    )

    data = json.loads(result)
    assert data["status"] == "success"
