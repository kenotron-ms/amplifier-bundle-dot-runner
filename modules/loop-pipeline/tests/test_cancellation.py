"""Tests for engine-side cooperative cancellation.

Spec coverage: EXEC-019 (cooperative cancellation via threading.Event).
"""

import threading

import pytest

from amplifier_module_loop_pipeline.context import PipelineContext
from amplifier_module_loop_pipeline.dot_parser import parse_dot
from amplifier_module_loop_pipeline.engine import PipelineEngine
from amplifier_module_loop_pipeline.handlers import HandlerRegistry
from amplifier_module_loop_pipeline.outcome import StageStatus
from amplifier_module_loop_pipeline.validation import validate_or_raise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LINEAR_DOT = """
digraph {
    start [shape=Mdiamond]
    step1 [prompt="Do step 1"]
    step2 [prompt="Do step 2"]
    exit [shape=Msquare]
    start -> step1 -> step2 -> exit
}
"""


class RecordingBackend:
    """Records which nodes were executed and returns a fixed outcome."""

    def __init__(self, return_value: str = "done"):
        self._return_value = return_value
        self.calls: list[str] = []

    async def run(self, node, prompt, context):
        self.calls.append(node.id)
        return self._return_value


class BlockingBackend:
    """Records calls; sets a threading.Event when the first node runs."""

    def __init__(self, signal_after: str, signal_event: threading.Event):
        self._signal_after = signal_after
        self._signal_event = signal_event
        self.calls: list[str] = []

    async def run(self, node, prompt, context):
        self.calls.append(node.id)
        if node.id == self._signal_after:
            self._signal_event.set()
        return "done"


def _make_engine(
    dot_source: str,
    backend=None,
    logs_root: str = "/tmp/test-pipeline-cancel",
    cancel_event: threading.Event | None = None,
) -> PipelineEngine:
    """Parse DOT, validate, and build an engine with optional cancel_event."""
    graph = parse_dot(dot_source)
    validate_or_raise(graph)
    context = PipelineContext()
    registry = HandlerRegistry(backend=backend)
    return PipelineEngine(
        graph=graph,
        context=context,
        handler_registry=registry,
        logs_root=logs_root,
        cancel_event=cancel_event,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_engine_accepts_cancel_event_parameter(tmp_path):
    """PipelineEngine can be constructed with a threading.Event."""
    cancel_event = threading.Event()
    engine = _make_engine(
        dot_source=_LINEAR_DOT,
        backend=RecordingBackend(),
        logs_root=str(tmp_path),
        cancel_event=cancel_event,
    )
    assert engine is not None


@pytest.mark.asyncio
async def test_engine_stops_on_cancel_before_first_node(tmp_path):
    """Set cancel event before run(); engine returns cancelled outcome immediately."""
    cancel_event = threading.Event()
    cancel_event.set()  # Already cancelled before run()

    backend = RecordingBackend()
    engine = _make_engine(
        dot_source=_LINEAR_DOT,
        backend=backend,
        logs_root=str(tmp_path),
        cancel_event=cancel_event,
    )

    outcome = await engine.run()

    # Should have stopped without executing any work nodes
    assert outcome.status == StageStatus.FAIL
    assert outcome.failure_reason == "cancelled"
    # step1 and step2 should NOT have been called
    assert "step1" not in backend.calls
    assert "step2" not in backend.calls


@pytest.mark.asyncio
async def test_engine_stops_on_cancel_between_nodes(tmp_path):
    """Set cancel event after first node completes; engine stops before second node."""
    cancel_event = threading.Event()

    class SetCancelAfterFirstBackend:
        """Sets cancel_event after step1 executes."""

        def __init__(self):
            self.calls: list[str] = []

        async def run(self, node, prompt, context):
            self.calls.append(node.id)
            if node.id == "step1":
                cancel_event.set()
            return "done"

    backend = SetCancelAfterFirstBackend()
    engine = _make_engine(
        dot_source=_LINEAR_DOT,
        backend=backend,
        logs_root=str(tmp_path),
        cancel_event=cancel_event,
    )

    outcome = await engine.run()

    assert outcome.status == StageStatus.FAIL
    assert outcome.failure_reason == "cancelled"
    # step1 ran, but step2 should NOT have been called
    assert "step1" in backend.calls
    assert "step2" not in backend.calls


@pytest.mark.asyncio
async def test_cancel_outcome_has_correct_status(tmp_path):
    """Cancelled outcome has status=FAIL and failure_reason='cancelled'."""
    cancel_event = threading.Event()
    cancel_event.set()

    engine = _make_engine(
        dot_source=_LINEAR_DOT,
        backend=RecordingBackend(),
        logs_root=str(tmp_path),
        cancel_event=cancel_event,
    )

    outcome = await engine.run()

    assert outcome.status == StageStatus.FAIL
    assert outcome.failure_reason == "cancelled"
    assert "cancel" in (outcome.notes or "").lower()


@pytest.mark.asyncio
async def test_engine_without_cancel_event_runs_normally(tmp_path):
    """When cancel_event=None, engine runs to completion (regression test)."""
    backend = RecordingBackend("done")
    engine = _make_engine(
        dot_source=_LINEAR_DOT,
        backend=backend,
        logs_root=str(tmp_path),
        cancel_event=None,  # explicit None
    )

    outcome = await engine.run()

    assert outcome.status in (StageStatus.SUCCESS, StageStatus.PARTIAL_SUCCESS)
    assert "step1" in backend.calls
    assert "step2" in backend.calls


@pytest.mark.asyncio
async def test_cancel_emits_pipeline_complete_event(tmp_path):
    """When cancelled, engine emits pipeline:complete with status 'cancelled'."""
    cancel_event = threading.Event()
    cancel_event.set()

    emitted_events: list[dict] = []

    class CapturingHooks:
        async def emit(self, event_name: str, data: dict) -> None:
            emitted_events.append({"event": event_name, "data": data})

    graph = parse_dot(_LINEAR_DOT)
    validate_or_raise(graph)
    context = PipelineContext()
    registry = HandlerRegistry(backend=RecordingBackend())
    engine = PipelineEngine(
        graph=graph,
        context=context,
        handler_registry=registry,
        logs_root=str(tmp_path),
        hooks=CapturingHooks(),
        cancel_event=cancel_event,
    )

    await engine.run()

    # Find pipeline:complete event
    complete_events = [e for e in emitted_events if e["event"] == "pipeline:complete"]
    assert len(complete_events) >= 1
    complete_data = complete_events[-1]["data"]
    assert complete_data["status"] == "cancelled"
