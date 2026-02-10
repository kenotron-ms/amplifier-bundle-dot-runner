"""Tests for the AmplifierBackend (CodergenBackend adapter).

This adapter spawns coding agent sub-sessions via the Amplifier
session.spawn capability. Tests mock the spawn function since it's
an app-layer capability.

Spec coverage: Section 4.5 (CodergenBackend Interface), Section 1.4.
"""

import json

import pytest

from amplifier_module_loop_pipeline.backend import AmplifierBackend
from amplifier_module_loop_pipeline.context import PipelineContext
from amplifier_module_loop_pipeline.graph import Node
from amplifier_module_loop_pipeline.outcome import Outcome, StageStatus


class MockCoordinator:
    """Mock coordinator that tracks spawn calls."""

    def __init__(self, spawn_result: dict | None = None):
        self._spawn_result = spawn_result or {"output": "done", "session_id": "child-1"}
        self.spawn_called = False
        self.spawn_call_count = 0
        self.last_spawn_kwargs: dict = {}
        self._capabilities: dict = {}

    def get_capability(self, name: str):
        if name == "session.spawn":
            return self._spawn_fn
        return self._capabilities.get(name)

    async def _spawn_fn(self, **kwargs):
        self.spawn_called = True
        self.spawn_call_count += 1
        self.last_spawn_kwargs = kwargs
        return self._spawn_result


class FailingCoordinator:
    """Coordinator whose spawn raises an exception."""

    def get_capability(self, name: str):
        if name == "session.spawn":
            return self._spawn_fn
        return None

    async def _spawn_fn(self, **kwargs):
        raise RuntimeError("Spawn failed: connection refused")


class NoSpawnCoordinator:
    """Coordinator that does not have session.spawn capability."""

    def get_capability(self, name: str):
        return None


def _make_node(**kwargs) -> Node:
    defaults = {"id": "implement", "prompt": "Build it"}
    defaults.update(kwargs)
    return Node(**defaults)


def _make_context() -> PipelineContext:
    return PipelineContext()


# --- Core spawn tests ---


@pytest.mark.asyncio
async def test_backend_spawns_session():
    """Backend uses coordinator session.spawn to create child session."""
    coordinator = MockCoordinator(spawn_result={"output": "done", "session_id": "child-1"})
    backend = AmplifierBackend(
        coordinator=coordinator,
        profiles={"anthropic": "attractor-anthropic"},
    )
    node = _make_node(attrs={"llm_provider": "anthropic"})
    result = await backend.run(node, "Build the feature", _make_context())
    assert coordinator.spawn_called
    assert isinstance(result, Outcome)
    assert result.status == StageStatus.SUCCESS


@pytest.mark.asyncio
async def test_backend_selects_profile_by_provider():
    """Different providers select different profile bundles."""
    coordinator = MockCoordinator(spawn_result={"output": "ok", "session_id": "c-1"})
    backend = AmplifierBackend(
        coordinator=coordinator,
        profiles={
            "anthropic": "attractor-anthropic",
            "openai": "attractor-openai",
        },
    )
    node_anthropic = _make_node(id="n1", attrs={"llm_provider": "anthropic"})
    node_openai = _make_node(id="n2", attrs={"llm_provider": "openai"})

    await backend.run(node_anthropic, "task", _make_context())
    first_profile = coordinator.last_spawn_kwargs.get("agent_name")

    await backend.run(node_openai, "task", _make_context())
    second_profile = coordinator.last_spawn_kwargs.get("agent_name")

    assert first_profile == "attractor-anthropic"
    assert second_profile == "attractor-openai"


@pytest.mark.asyncio
async def test_backend_default_provider_is_anthropic():
    """If node has no llm_provider, defaults to anthropic."""
    coordinator = MockCoordinator(spawn_result={"output": "ok", "session_id": "c-1"})
    backend = AmplifierBackend(
        coordinator=coordinator,
        profiles={"anthropic": "attractor-anthropic"},
    )
    node = _make_node(attrs={})  # No llm_provider
    await backend.run(node, "task", _make_context())
    assert coordinator.last_spawn_kwargs.get("agent_name") == "attractor-anthropic"


# --- Outcome parsing tests ---


@pytest.mark.asyncio
async def test_backend_parses_json_outcome():
    """If child returns JSON with status field, parse it as Outcome."""
    json_output = json.dumps({"status": "fail", "failure_reason": "3 tests failing"})
    coordinator = MockCoordinator(
        spawn_result={"output": json_output, "session_id": "c-1"}
    )
    backend = AmplifierBackend(
        coordinator=coordinator,
        profiles={"anthropic": "attractor-anthropic"},
    )
    node = _make_node(attrs={"llm_provider": "anthropic"})
    result = await backend.run(node, "task", _make_context())
    assert isinstance(result, Outcome)
    assert result.status == StageStatus.FAIL
    assert result.failure_reason == "3 tests failing"


@pytest.mark.asyncio
async def test_backend_wraps_plain_text_as_success():
    """If child returns plain text, wrap it in a SUCCESS outcome."""
    coordinator = MockCoordinator(
        spawn_result={"output": "Implementation complete", "session_id": "c-1"}
    )
    backend = AmplifierBackend(
        coordinator=coordinator,
        profiles={"anthropic": "attractor-anthropic"},
    )
    node = _make_node(attrs={"llm_provider": "anthropic"})
    result = await backend.run(node, "task", _make_context())
    assert isinstance(result, Outcome)
    assert result.status == StageStatus.SUCCESS


@pytest.mark.asyncio
async def test_backend_parses_partial_success():
    """JSON outcome with partial_success status is parsed correctly."""
    json_output = json.dumps({"status": "partial_success", "notes": "some tests pass"})
    coordinator = MockCoordinator(
        spawn_result={"output": json_output, "session_id": "c-1"}
    )
    backend = AmplifierBackend(
        coordinator=coordinator,
        profiles={"anthropic": "attractor-anthropic"},
    )
    result = await backend.run(
        _make_node(attrs={"llm_provider": "anthropic"}), "task", _make_context()
    )
    assert result.status == StageStatus.PARTIAL_SUCCESS


# --- Error handling tests ---


@pytest.mark.asyncio
async def test_backend_handles_spawn_failure():
    """Spawn failure returns Outcome(status=FAIL) instead of raising."""
    coordinator = FailingCoordinator()
    backend = AmplifierBackend(
        coordinator=coordinator,
        profiles={"anthropic": "attractor-anthropic"},
    )
    node = _make_node(attrs={"llm_provider": "anthropic"})
    result = await backend.run(node, "task", _make_context())
    assert isinstance(result, Outcome)
    assert result.status == StageStatus.FAIL
    assert "connection refused" in (result.failure_reason or "").lower()


@pytest.mark.asyncio
async def test_backend_handles_no_spawn_capability():
    """No session.spawn capability returns Outcome(status=FAIL)."""
    coordinator = NoSpawnCoordinator()
    backend = AmplifierBackend(
        coordinator=coordinator,
        profiles={"anthropic": "attractor-anthropic"},
    )
    node = _make_node(attrs={"llm_provider": "anthropic"})
    result = await backend.run(node, "task", _make_context())
    assert isinstance(result, Outcome)
    assert result.status == StageStatus.FAIL
    assert "not available" in (result.failure_reason or "").lower()


# --- Config forwarding tests ---


@pytest.mark.asyncio
async def test_backend_forwards_reasoning_effort():
    """reasoning_effort from node attrs is forwarded to spawn call."""
    coordinator = MockCoordinator(spawn_result={"output": "ok", "session_id": "c-1"})
    backend = AmplifierBackend(
        coordinator=coordinator,
        profiles={"anthropic": "attractor-anthropic"},
    )
    node = _make_node(attrs={"llm_provider": "anthropic", "reasoning_effort": "low"})
    await backend.run(node, "task", _make_context())
    orch_config = coordinator.last_spawn_kwargs.get("orchestrator_config", {})
    assert orch_config.get("reasoning_effort") == "low"


@pytest.mark.asyncio
async def test_backend_forwards_model():
    """llm_model from node attrs is forwarded to spawn call."""
    coordinator = MockCoordinator(spawn_result={"output": "ok", "session_id": "c-1"})
    backend = AmplifierBackend(
        coordinator=coordinator,
        profiles={"anthropic": "attractor-anthropic"},
    )
    node = _make_node(attrs={"llm_provider": "anthropic", "llm_model": "claude-sonnet-4-5"})
    await backend.run(node, "task", _make_context())
    prefs = coordinator.last_spawn_kwargs.get("provider_preferences")
    assert prefs is not None
    assert any(p.get("model") == "claude-sonnet-4-5" for p in prefs)
