"""Tests for the AmplifierBackend (CodergenBackend adapter).

This adapter spawns coding agent sub-sessions via the Amplifier
session.spawn capability. Tests mock the spawn function since it's
an app-layer capability.

Also tests the Path B fallback: a direct provider mini tool loop
when session.spawn is not available.

Spec coverage: Section 4.5 (CodergenBackend Interface), Section 1.4.
"""

import json
import sys
import types
from dataclasses import dataclass, field
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Provide a minimal amplifier_core stub so the backend's lazy imports work
# in the test environment where amplifier_core may not be installed.
# ---------------------------------------------------------------------------
if "amplifier_core" not in sys.modules:

    @dataclass
    class _StubMessage:
        role: str = "user"
        content: Any = ""
        tool_call_id: str | None = None
        name: str | None = None
        metadata: dict | None = None

    @dataclass
    class _StubToolCallBlock:
        id: str = ""
        name: str = ""
        input: dict = field(default_factory=dict)
        type: str = "tool_call"

    @dataclass
    class _StubChatRequest:
        messages: list = field(default_factory=list)
        tools: list | None = None
        tool_choice: str | None = None
        reasoning_effort: str | None = None

    _stub_core = types.ModuleType("amplifier_core")
    _stub_core.Message = _StubMessage  # type: ignore[attr-defined]
    _stub_core.ChatRequest = _StubChatRequest  # type: ignore[attr-defined]
    sys.modules["amplifier_core"] = _stub_core

    _stub_msg = types.ModuleType("amplifier_core.message_models")
    _stub_msg.ToolCallBlock = _StubToolCallBlock  # type: ignore[attr-defined]
    sys.modules["amplifier_core.message_models"] = _stub_msg

from amplifier_module_loop_pipeline.backend import AmplifierBackend
from amplifier_module_loop_pipeline.context import PipelineContext
from amplifier_module_loop_pipeline.graph import Node
from amplifier_module_loop_pipeline.outcome import Outcome, StageStatus


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class _MockSession:
    """Minimal stand-in for AmplifierSession."""

    config: dict[str, Any] = {}


class MockCoordinator:
    """Mock coordinator that tracks spawn calls."""

    def __init__(
        self,
        spawn_result: dict | None = None,
        agents: dict[str, Any] | None = None,
    ):
        self._spawn_result = spawn_result or {"output": "done", "session_id": "child-1"}
        self.spawn_called = False
        self.spawn_call_count = 0
        self.last_spawn_kwargs: dict = {}
        self._capabilities: dict = {}
        # Provide session and config like a real coordinator
        self.session = _MockSession()
        self.config: dict[str, Any] = {"agents": agents or {}}

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

    session = _MockSession()
    config: dict[str, Any] = {"agents": {}}

    def get_capability(self, name: str):
        if name == "session.spawn":
            return self._spawn_fn
        return None

    async def _spawn_fn(self, **kwargs):
        raise RuntimeError("Spawn failed: connection refused")


class NoSpawnCoordinator:
    """Coordinator that does not have session.spawn capability."""

    session = _MockSession()
    config: dict[str, Any] = {"agents": {}}

    def get_capability(self, name: str):
        return None


@dataclass
class _MockToolResult:
    """Minimal ToolResult replacement."""

    output: str = "tool output"
    success: bool = True


@dataclass
class _MockTextBlock:
    text: str
    type: str = "text"


@dataclass
class _MockToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class _MockChatResponse:
    content: list[Any] = field(default_factory=list)
    tool_calls: list[Any] | None = None


class _MockTool:
    def __init__(self, name: str, result: str = "tool done"):
        self._name = name
        self._result = result
        self.call_count = 0
        self.last_input: dict[str, Any] = {}
        self.parameters: dict[str, Any] = {"type": "object", "properties": {}}
        self.description = f"Mock tool {name}"

    @property
    def name(self) -> str:
        return self._name

    async def execute(self, input: dict[str, Any]) -> _MockToolResult:
        self.call_count += 1
        self.last_input = input
        return _MockToolResult(output=self._result)


class _MockProvider:
    """Mock provider that returns canned responses."""

    name = "mock"

    def __init__(self, responses: list[_MockChatResponse] | None = None):
        self._responses = (
            list(responses)
            if responses
            else [_MockChatResponse(content=[_MockTextBlock(text="done")])]
        )
        self._call_idx = 0

    async def complete(self, request: Any) -> _MockChatResponse:
        if self._call_idx < len(self._responses):
            resp = self._responses[self._call_idx]
            self._call_idx += 1
            return resp
        return _MockChatResponse(content=[_MockTextBlock(text="done")])

    def parse_tool_calls(self, response: Any) -> list[Any]:
        return list(response.tool_calls) if response.tool_calls else []


def _make_node(**kwargs) -> Node:
    defaults = {"id": "implement", "prompt": "Build it"}
    defaults.update(kwargs)
    return Node(**defaults)


def _make_context() -> PipelineContext:
    return PipelineContext()


# ---------------------------------------------------------------------------
# Core spawn tests (Path A)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_backend_spawns_session():
    """Backend uses coordinator session.spawn to create child session."""
    coordinator = MockCoordinator(
        spawn_result={"output": "done", "session_id": "child-1"}
    )
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


# --- Spawn signature tests (parent_session / agent_configs / sub_session_id) ---


@pytest.mark.asyncio
async def test_backend_passes_parent_session():
    """Spawn kwargs include parent_session from coordinator.session."""
    coordinator = MockCoordinator(spawn_result={"output": "ok", "session_id": "c-1"})
    backend = AmplifierBackend(
        coordinator=coordinator,
        profiles={"anthropic": "attractor-anthropic"},
    )
    await backend.run(_make_node(attrs={}), "task", _make_context())
    assert "parent_session" in coordinator.last_spawn_kwargs
    assert coordinator.last_spawn_kwargs["parent_session"] is coordinator.session


@pytest.mark.asyncio
async def test_backend_passes_agent_configs():
    """Spawn kwargs include agent_configs from coordinator.config."""
    agents = {"my-agent": {"description": "Test agent"}}
    coordinator = MockCoordinator(
        spawn_result={"output": "ok", "session_id": "c-1"},
        agents=agents,
    )
    backend = AmplifierBackend(
        coordinator=coordinator,
        profiles={"anthropic": "attractor-anthropic"},
    )
    await backend.run(_make_node(attrs={}), "task", _make_context())
    assert coordinator.last_spawn_kwargs.get("agent_configs") == agents


@pytest.mark.asyncio
async def test_backend_uses_sub_session_id_not_session_id():
    """Session reuse passes 'sub_session_id', not 'session_id'."""
    coordinator = MockCoordinator(
        spawn_result={"output": "ok", "session_id": "sess-abc"},
    )
    backend = AmplifierBackend(
        coordinator=coordinator,
        profiles={"anthropic": "attractor-anthropic"},
    )
    from amplifier_module_loop_pipeline.graph import Edge, Graph

    node1 = _make_node(
        id="step1",
        attrs={"llm_provider": "anthropic", "fidelity": "full", "thread_id": "t"},
    )
    node2 = _make_node(
        id="step2",
        attrs={"llm_provider": "anthropic", "fidelity": "full", "thread_id": "t"},
    )
    graph = Graph(
        name="test",
        nodes={
            "start": Node(id="start", shape="Mdiamond"),
            "step1": node1,
            "step2": node2,
            "exit": Node(id="exit", shape="Msquare"),
        },
        edges=[
            Edge(from_node="start", to_node="step1"),
            Edge(from_node="step1", to_node="step2"),
            Edge(from_node="step2", to_node="exit"),
        ],
    )
    edge = Edge(from_node="start", to_node="step1")

    await backend.run(node1, "First", _make_context(), incoming_edge=edge, graph=graph)
    await backend.run(node2, "Second", _make_context(), incoming_edge=edge, graph=graph)

    # Must use sub_session_id (not session_id) for the CLI spawn capability
    assert "sub_session_id" in coordinator.last_spawn_kwargs
    assert "session_id" not in coordinator.last_spawn_kwargs
    assert coordinator.last_spawn_kwargs["sub_session_id"] == "sess-abc"


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
async def test_backend_handles_no_spawn_no_provider():
    """No session.spawn and no provider returns FAIL."""
    coordinator = NoSpawnCoordinator()
    backend = AmplifierBackend(
        coordinator=coordinator,
        profiles={"anthropic": "attractor-anthropic"},
    )
    node = _make_node(attrs={"llm_provider": "anthropic"})
    result = await backend.run(node, "task", _make_context())
    assert isinstance(result, Outcome)
    assert result.status == StageStatus.FAIL
    assert "available" in (result.failure_reason or "").lower()


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
    node = _make_node(
        attrs={"llm_provider": "anthropic", "llm_model": "claude-sonnet-4-5"}
    )
    await backend.run(node, "task", _make_context())
    prefs = coordinator.last_spawn_kwargs.get("provider_preferences")
    assert prefs is not None
    assert any(getattr(p, "model", None) == "claude-sonnet-4-5" for p in prefs)


# ---------------------------------------------------------------------------
# Path B: Direct provider mini tool loop fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_backend_falls_back_to_tool_loop():
    """When spawn is unavailable but provider is given, uses direct tool loop."""
    coordinator = NoSpawnCoordinator()
    provider = _MockProvider()
    backend = AmplifierBackend(
        coordinator=coordinator,
        profiles={"anthropic": "attractor-anthropic"},
        provider=provider,
    )
    node = _make_node(attrs={"llm_provider": "anthropic"})
    result = await backend.run(node, "task", _make_context())
    assert isinstance(result, Outcome)
    assert result.status == StageStatus.SUCCESS


@pytest.mark.asyncio
async def test_tool_loop_executes_tools_then_returns():
    """Tool loop calls tools and feeds results back until model stops."""
    tool = _MockTool("write_file", result="file written")
    provider = _MockProvider(
        responses=[
            # Round 1: model requests a tool call
            _MockChatResponse(
                content=[_MockTextBlock(text="Let me write that file")],
                tool_calls=[
                    _MockToolCall(
                        id="tc-1", name="write_file", arguments={"path": "a.py"}
                    )
                ],
            ),
            # Round 2: model returns text only (done)
            _MockChatResponse(
                content=[_MockTextBlock(text="All done")],
                tool_calls=None,
            ),
        ]
    )
    coordinator = NoSpawnCoordinator()
    backend = AmplifierBackend(
        coordinator=coordinator,
        profiles={},
        provider=provider,
        tools={"write_file": tool},
    )
    result = await backend.run(_make_node(), "Write a file", _make_context())

    assert tool.call_count == 1
    assert tool.last_input == {"path": "a.py"}
    assert result.status == StageStatus.SUCCESS


@pytest.mark.asyncio
async def test_tool_loop_handles_unknown_tool():
    """Tool loop gracefully handles calls to unknown tools."""
    provider = _MockProvider(
        responses=[
            _MockChatResponse(
                content=[_MockTextBlock(text="")],
                tool_calls=[_MockToolCall(id="tc-1", name="nonexistent", arguments={})],
            ),
            _MockChatResponse(
                content=[_MockTextBlock(text="ok, no tool")],
                tool_calls=None,
            ),
        ]
    )
    coordinator = NoSpawnCoordinator()
    backend = AmplifierBackend(
        coordinator=coordinator,
        profiles={},
        provider=provider,
        tools={},
    )
    result = await backend.run(_make_node(), "task", _make_context())
    assert result.status == StageStatus.SUCCESS


@pytest.mark.asyncio
async def test_tool_loop_handles_provider_failure():
    """Tool loop returns FAIL when provider raises."""

    class FailingProvider:
        name = "failing"

        async def complete(self, request):
            raise ConnectionError("API unreachable")

        def parse_tool_calls(self, response):
            return []

    coordinator = NoSpawnCoordinator()
    backend = AmplifierBackend(
        coordinator=coordinator,
        profiles={},
        provider=FailingProvider(),
    )
    result = await backend.run(_make_node(), "task", _make_context())
    assert result.status == StageStatus.FAIL
    assert "unreachable" in (result.failure_reason or "").lower()
