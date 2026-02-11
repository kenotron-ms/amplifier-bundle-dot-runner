"""Tests for retry logic with configurable policy.

Spec coverage: RETRY-001–011, FAIL-001, Section 3.5–3.6.
"""

import pytest

from amplifier_module_loop_pipeline.context import PipelineContext
from amplifier_module_loop_pipeline.graph import Graph, Node
from amplifier_module_loop_pipeline.outcome import Outcome, StageStatus
from amplifier_module_loop_pipeline.retry import (
    BackoffConfig,
    RetryPolicy,
    execute_with_retry,
)


class MockHandler:
    """Handler that returns outcomes from a pre-set sequence."""

    def __init__(self, outcomes: list[Outcome]):
        self._outcomes = list(outcomes)
        self.call_count = 0

    async def execute(
        self, node: Node, context: PipelineContext, graph: Graph, logs_root: str
    ) -> Outcome:
        self.call_count += 1
        if self._outcomes:
            return self._outcomes.pop(0)
        return Outcome(status=StageStatus.FAIL, failure_reason="no more outcomes")


class RaisingHandler:
    """Handler that raises an exception on the first N calls."""

    def __init__(self, fail_count: int, then: Outcome):
        self._fail_count = fail_count
        self._then = then
        self.call_count = 0

    async def execute(
        self, node: Node, context: PipelineContext, graph: Graph, logs_root: str
    ) -> Outcome:
        self.call_count += 1
        if self.call_count <= self._fail_count:
            raise RuntimeError(f"Transient error #{self.call_count}")
        return self._then


def _make_node(**kwargs) -> Node:
    defaults = {"id": "step", "prompt": "Do work"}
    defaults.update(kwargs)
    return Node(**defaults)


def _make_graph() -> Graph:
    return Graph(
        name="test",
        nodes={"start": Node(id="start", shape="Mdiamond")},
        edges=[],
    )


# --- Basic retry behavior ---


@pytest.mark.asyncio
async def test_success_on_first_try():
    """No retries needed when first attempt succeeds."""
    handler = MockHandler([Outcome(status=StageStatus.SUCCESS)])
    policy = RetryPolicy(max_attempts=3)
    result = await execute_with_retry(
        handler, _make_node(), PipelineContext(), _make_graph(), "/tmp", policy
    )
    assert result.status == StageStatus.SUCCESS
    assert handler.call_count == 1


@pytest.mark.asyncio
async def test_retry_on_retry_outcome():
    """RETRY outcome triggers additional attempts."""
    handler = MockHandler([
        Outcome(status=StageStatus.RETRY),
        Outcome(status=StageStatus.RETRY),
        Outcome(status=StageStatus.SUCCESS),
    ])
    policy = RetryPolicy(max_attempts=3, backoff=BackoffConfig(initial_delay_ms=0))
    result = await execute_with_retry(
        handler, _make_node(), PipelineContext(), _make_graph(), "/tmp", policy
    )
    assert result.status == StageStatus.SUCCESS
    assert handler.call_count == 3


@pytest.mark.asyncio
async def test_fail_not_retried():
    """FAIL outcome returns immediately — no retries (RETRY-006)."""
    handler = MockHandler([
        Outcome(status=StageStatus.FAIL, failure_reason="bad input")
    ])
    policy = RetryPolicy(max_attempts=3, backoff=BackoffConfig(initial_delay_ms=0))
    result = await execute_with_retry(
        handler, _make_node(), PipelineContext(), _make_graph(), "/tmp", policy
    )
    assert result.status == StageStatus.FAIL
    assert handler.call_count == 1
    assert result.failure_reason == "bad input"


@pytest.mark.asyncio
async def test_success_not_retried():
    """SUCCESS outcome returns immediately."""
    handler = MockHandler([Outcome(status=StageStatus.SUCCESS)])
    policy = RetryPolicy(max_attempts=5, backoff=BackoffConfig(initial_delay_ms=0))
    result = await execute_with_retry(
        handler, _make_node(), PipelineContext(), _make_graph(), "/tmp", policy
    )
    assert result.status == StageStatus.SUCCESS
    assert handler.call_count == 1


@pytest.mark.asyncio
async def test_partial_success_not_retried():
    """PARTIAL_SUCCESS returns immediately."""
    handler = MockHandler([Outcome(status=StageStatus.PARTIAL_SUCCESS)])
    policy = RetryPolicy(max_attempts=3, backoff=BackoffConfig(initial_delay_ms=0))
    result = await execute_with_retry(
        handler, _make_node(), PipelineContext(), _make_graph(), "/tmp", policy
    )
    assert result.status == StageStatus.PARTIAL_SUCCESS
    assert handler.call_count == 1


# --- Retry exhaustion ---


@pytest.mark.asyncio
async def test_retries_exhausted_returns_fail():
    """All retries exhausted returns FAIL (RETRY-005)."""
    handler = MockHandler([Outcome(status=StageStatus.RETRY)] * 3)
    policy = RetryPolicy(max_attempts=3, backoff=BackoffConfig(initial_delay_ms=0))
    result = await execute_with_retry(
        handler, _make_node(), PipelineContext(), _make_graph(), "/tmp", policy
    )
    assert result.status == StageStatus.FAIL
    assert "max retries exceeded" in (result.failure_reason or "").lower()
    assert handler.call_count == 3


@pytest.mark.asyncio
async def test_allow_partial_on_exhaustion():
    """allow_partial=true -> PARTIAL_SUCCESS after retries exhausted (RETRY-005)."""
    handler = MockHandler([Outcome(status=StageStatus.RETRY)] * 3)
    policy = RetryPolicy(max_attempts=3, backoff=BackoffConfig(initial_delay_ms=0))
    node = _make_node(attrs={"allow_partial": True})
    result = await execute_with_retry(
        handler, node, PipelineContext(), _make_graph(), "/tmp", policy
    )
    assert result.status == StageStatus.PARTIAL_SUCCESS
    assert handler.call_count == 3


# --- Default behavior (max_retries=0) ---


@pytest.mark.asyncio
async def test_default_no_retries():
    """Default max_retries=0 means max_attempts=1 — no retries."""
    handler = MockHandler([Outcome(status=StageStatus.RETRY)])
    policy = RetryPolicy(max_attempts=1, backoff=BackoffConfig(initial_delay_ms=0))
    result = await execute_with_retry(
        handler, _make_node(), PipelineContext(), _make_graph(), "/tmp", policy
    )
    assert result.status == StageStatus.FAIL
    assert handler.call_count == 1


# --- Exception handling ---


@pytest.mark.asyncio
async def test_exception_retried():
    """Exceptions are caught and retried (transient failures)."""
    handler = RaisingHandler(
        fail_count=2,
        then=Outcome(status=StageStatus.SUCCESS),
    )
    policy = RetryPolicy(max_attempts=3, backoff=BackoffConfig(initial_delay_ms=0))
    result = await execute_with_retry(
        handler, _make_node(), PipelineContext(), _make_graph(), "/tmp", policy
    )
    assert result.status == StageStatus.SUCCESS
    assert handler.call_count == 3


@pytest.mark.asyncio
async def test_exception_exhausted_returns_fail():
    """All retries exhausted by exceptions returns FAIL."""
    handler = RaisingHandler(
        fail_count=5,
        then=Outcome(status=StageStatus.SUCCESS),
    )
    policy = RetryPolicy(max_attempts=3, backoff=BackoffConfig(initial_delay_ms=0))
    result = await execute_with_retry(
        handler, _make_node(), PipelineContext(), _make_graph(), "/tmp", policy
    )
    assert result.status == StageStatus.FAIL
    assert "Transient error" in (result.failure_reason or "")
    assert handler.call_count == 3


# --- RetryPolicy construction ---


def test_retry_policy_from_node_max_retries():
    """Build policy from node's max_retries attribute."""
    node = _make_node(attrs={"max_retries": 3})
    graph = _make_graph()
    policy = RetryPolicy.from_node(node, graph)
    # max_retries=3 means max_attempts=4 (1 initial + 3 retries)
    assert policy.max_attempts == 4


def test_retry_policy_from_graph_default():
    """Build policy from graph default_max_retry when node has none."""
    node = _make_node(attrs={})
    graph = Graph(
        name="test",
        nodes={"start": Node(id="start", shape="Mdiamond")},
        edges=[],
        default_max_retry=2,
    )
    policy = RetryPolicy.from_node(node, graph)
    # graph default_max_retry=2 -> max_attempts=3
    assert policy.max_attempts == 3


def test_retry_policy_default_zero():
    """When no max_retries anywhere, default is 0 retries (1 attempt)."""
    node = _make_node(attrs={})
    graph = Graph(
        name="test",
        nodes={"start": Node(id="start", shape="Mdiamond")},
        edges=[],
        default_max_retry=0,
    )
    policy = RetryPolicy.from_node(node, graph)
    assert policy.max_attempts == 1


# --- BackoffConfig ---


def test_backoff_delay_exponential():
    """Exponential backoff: delay doubles each attempt."""
    config = BackoffConfig(initial_delay_ms=200, backoff_factor=2.0, jitter=False)
    assert config.delay_for_attempt(1) == 200.0
    assert config.delay_for_attempt(2) == 400.0
    assert config.delay_for_attempt(3) == 800.0


def test_backoff_delay_capped():
    """Delay is capped at max_delay_ms."""
    config = BackoffConfig(
        initial_delay_ms=1000, backoff_factor=10.0, max_delay_ms=5000, jitter=False
    )
    assert config.delay_for_attempt(1) == 1000.0
    assert config.delay_for_attempt(2) == 5000.0  # capped
    assert config.delay_for_attempt(3) == 5000.0  # still capped


def test_backoff_linear():
    """Linear backoff (factor=1.0) has fixed delay."""
    config = BackoffConfig(
        initial_delay_ms=500, backoff_factor=1.0, jitter=False
    )
    assert config.delay_for_attempt(1) == 500.0
    assert config.delay_for_attempt(2) == 500.0
    assert config.delay_for_attempt(3) == 500.0


def test_backoff_with_jitter():
    """Jitter adds randomness within [0.5, 1.5] of the delay."""
    config = BackoffConfig(initial_delay_ms=1000, backoff_factor=1.0, jitter=True)
    delays = [config.delay_for_attempt(1) for _ in range(100)]
    # All delays should be between 500 and 1500
    assert all(500.0 <= d <= 1500.0 for d in delays)
    # Not all the same (very unlikely with 100 samples)
    assert len(set(delays)) > 1


@pytest.mark.asyncio
async def test_retry_emits_retrying_event():
    """execute_with_retry must emit StageRetrying events on retry."""
    from amplifier_module_loop_pipeline.pipeline_events import (
        PIPELINE_STAGE_RETRYING,
    )

    emitted: list[tuple[str, dict]] = []

    class MockHooks:
        async def emit(self, event_name, data):
            emitted.append((event_name, data))

    call_count = 0

    class RetryThenSucceedHandler:
        async def execute(self, node, context, graph, logs_root):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return Outcome(status=StageStatus.RETRY, failure_reason="not yet")
            return Outcome(status=StageStatus.SUCCESS)

    node = Node(id="work", shape="box", prompt="do work")
    graph = Graph(
        name="test",
        nodes={"work": node},
        edges=[],
    )
    ctx = PipelineContext()
    policy = RetryPolicy(max_attempts=3, backoff=BackoffConfig(initial_delay_ms=0))

    await execute_with_retry(
        RetryThenSucceedHandler(),
        node,
        ctx,
        graph,
        "/tmp/test",
        policy,
        hooks=MockHooks(),
    )

    event_names = [e[0] for e in emitted]
    assert PIPELINE_STAGE_RETRYING in event_names


@pytest.mark.asyncio
async def test_retry_emits_stage_failed_on_exhaustion():
    """execute_with_retry must emit StageFailed when retries exhausted."""
    from amplifier_module_loop_pipeline.pipeline_events import (
        PIPELINE_STAGE_FAILED,
    )

    emitted: list[tuple[str, dict]] = []

    class MockHooks:
        async def emit(self, event_name, data):
            emitted.append((event_name, data))

    handler = MockHandler([Outcome(status=StageStatus.RETRY)] * 3)
    node = Node(id="work", shape="box", prompt="do work")
    graph = Graph(
        name="test",
        nodes={"work": node},
        edges=[],
    )
    ctx = PipelineContext()
    policy = RetryPolicy(max_attempts=3, backoff=BackoffConfig(initial_delay_ms=0))

    await execute_with_retry(
        handler,
        node,
        ctx,
        graph,
        "/tmp/test",
        policy,
        hooks=MockHooks(),
    )

    event_names = [e[0] for e in emitted]
    assert PIPELINE_STAGE_FAILED in event_names
