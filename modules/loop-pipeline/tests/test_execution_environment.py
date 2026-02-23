"""Tests for execution environment lifecycle in PipelineOrchestrator.

Verifies that the orchestrator optionally creates/destroys an execution
environment around the pipeline engine run when configured.
"""

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from amplifier_module_loop_pipeline import PipelineOrchestrator
from amplifier_module_loop_pipeline.backend import AmplifierBackend
from amplifier_module_loop_pipeline.context import PipelineContext
from amplifier_module_loop_pipeline.graph import Node
from amplifier_module_loop_pipeline.outcome import Outcome, StageStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_env_create(container_id="abc123"):
    """Return a mock env_create tool whose execute returns a container_id."""
    tool = AsyncMock()
    tool.execute = AsyncMock(
        return_value=MagicMock(
            success=True,
            output=json.dumps(
                {"container_id": container_id, "name": "pipeline-workspace"}
            ),
        )
    )
    return tool


def _make_mock_env_destroy():
    """Return a mock env_destroy tool."""
    tool = AsyncMock()
    tool.execute = AsyncMock(
        return_value=MagicMock(
            success=True,
            output=json.dumps({"status": "destroyed"}),
        )
    )
    return tool


MINIMAL_DOT = """
digraph {
    start [shape=Mdiamond]
    work [prompt="Do work"]
    exit [shape=Msquare]
    start -> work -> exit
}
"""


def _make_orchestrator(execution_environment=None):
    """Create a PipelineOrchestrator with optional execution_environment config."""
    config = {"dot_source": MINIMAL_DOT}
    if execution_environment is not None:
        config["execution_environment"] = execution_environment
    return PipelineOrchestrator(config)


# ---------------------------------------------------------------------------
# Task 1: Environment lifecycle tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_env_lifecycle_creates_and_destroys():
    """env_create called before engine, env_destroy after, container_id in context."""
    env_create = _make_mock_env_create(container_id="abc123")
    env_destroy = _make_mock_env_destroy()
    captured_context = {}

    orchestrator = _make_orchestrator(
        execution_environment={
            "type": "docker",
            "name": "pipeline-workspace",
            "image": "python:3.12",
            "mount_cwd": True,
        }
    )

    async def capturing_engine_run(self_engine, *args, **kwargs):
        """Capture the pipeline context during engine.run()."""
        captured_context["container_id"] = self_engine.context.get(
            "internal.env_container_id"
        )
        captured_context["env_type"] = self_engine.context.get("internal.env_type")
        return Outcome(status=StageStatus.SUCCESS, notes="done")

    with patch(
        "amplifier_module_loop_pipeline.PipelineEngine.run",
        capturing_engine_run,
    ):
        await orchestrator.execute(
            prompt="Build it",
            context=None,
            providers={},
            tools={"env_create": env_create, "env_destroy": env_destroy},
            hooks=None,
            backend=MagicMock(),
        )

    # env_create was called with correct args (including pass-through config)
    env_create.execute.assert_called_once()
    create_args = env_create.execute.call_args[0][0]
    assert create_args["type"] == "docker"
    assert create_args["name"] == "pipeline-workspace"
    assert create_args["image"] == "python:3.12"
    assert create_args["mount_cwd"] is True

    # container_id was stored in context BEFORE engine ran
    assert captured_context["container_id"] == "abc123"
    assert captured_context["env_type"] == "docker"

    # env_destroy was called with correct instance name
    env_destroy.execute.assert_called_once()
    destroy_args = env_destroy.execute.call_args[0][0]
    assert destroy_args["instance"] == "pipeline-workspace"


@pytest.mark.asyncio
async def test_env_lifecycle_no_config_no_lifecycle():
    """No env lifecycle when execution_environment config is absent."""
    env_create = _make_mock_env_create()
    env_destroy = _make_mock_env_destroy()

    orchestrator = _make_orchestrator()  # No execution_environment

    with patch(
        "amplifier_module_loop_pipeline.PipelineEngine.run",
        new_callable=AsyncMock,
        return_value=Outcome(status=StageStatus.SUCCESS, notes="done"),
    ):
        result = await orchestrator.execute(
            prompt="Build it",
            context=None,
            providers={},
            tools={"env_create": env_create, "env_destroy": env_destroy},
            hooks=None,
            backend=MagicMock(),
        )

    env_create.execute.assert_not_called()
    env_destroy.execute.assert_not_called()
    parsed = json.loads(result)
    assert parsed["status"] == "success"


@pytest.mark.asyncio
async def test_env_lifecycle_config_but_no_env_tools(caplog):
    """Warning logged when config present but env_create tool missing."""
    orchestrator = _make_orchestrator(
        execution_environment={"type": "docker", "name": "pipeline-workspace"}
    )

    with caplog.at_level(logging.WARNING, logger="amplifier_module_loop_pipeline"):
        with patch(
            "amplifier_module_loop_pipeline.PipelineEngine.run",
            new_callable=AsyncMock,
            return_value=Outcome(status=StageStatus.SUCCESS, notes="done"),
        ):
            result = await orchestrator.execute(
                prompt="Build it",
                context=None,
                providers={},
                tools={},  # No env tools!
                hooks=None,
                backend=MagicMock(),
            )

    parsed = json.loads(result)
    assert parsed["status"] == "success"

    # Warning should mention env_create not available
    assert any(
        "env_create" in record.message
        for record in caplog.records
        if record.levelno >= logging.WARNING
    )


@pytest.mark.asyncio
async def test_env_lifecycle_destroy_called_on_failure():
    """env_destroy called even when engine.run() raises; exception propagates."""
    env_create = _make_mock_env_create(container_id="abc123")
    env_destroy = _make_mock_env_destroy()

    orchestrator = _make_orchestrator(
        execution_environment={"type": "docker", "name": "pipeline-workspace"}
    )

    with patch(
        "amplifier_module_loop_pipeline.PipelineEngine.run",
        new_callable=AsyncMock,
        side_effect=RuntimeError("Engine exploded"),
    ):
        with pytest.raises(RuntimeError, match="Engine exploded"):
            await orchestrator.execute(
                prompt="Build it",
                context=None,
                providers={},
                tools={"env_create": env_create, "env_destroy": env_destroy},
                hooks=None,
                backend=MagicMock(),
            )

    # env_create was called
    env_create.execute.assert_called_once()

    # env_destroy was STILL called despite the exception
    env_destroy.execute.assert_called_once()


@pytest.mark.asyncio
async def test_env_lifecycle_unparseable_create_response(caplog):
    """Falls back to local when env_create returns non-JSON output."""
    env_create = AsyncMock()
    env_create.execute = AsyncMock(
        return_value=MagicMock(
            success=True,
            output="not json",
        )
    )
    env_destroy = _make_mock_env_destroy()

    orchestrator = _make_orchestrator(
        execution_environment={"type": "docker", "name": "pipeline-workspace"}
    )

    with caplog.at_level(logging.WARNING, logger="amplifier_module_loop_pipeline"):
        with patch(
            "amplifier_module_loop_pipeline.PipelineEngine.run",
            new_callable=AsyncMock,
            return_value=Outcome(status=StageStatus.SUCCESS, notes="done"),
        ):
            result = await orchestrator.execute(
                prompt="Build it",
                context=None,
                providers={},
                tools={"env_create": env_create, "env_destroy": env_destroy},
                hooks=None,
                backend=MagicMock(),
            )

    # No exception raised, pipeline ran successfully
    parsed_result = json.loads(result)
    assert parsed_result["status"] == "success"

    # Warning logged about unparseable output
    assert any(
        "unparseable" in record.message.lower() or "env_create" in record.message
        for record in caplog.records
        if record.levelno >= logging.WARNING
    )

    # env_destroy is NOT called since no container was created
    env_destroy.execute.assert_not_called()


# ---------------------------------------------------------------------------
# Task 2: Backend attach-to injection tests
# ---------------------------------------------------------------------------


def _make_backend_with_mock_spawn():
    """Create an AmplifierBackend with a mock coordinator that has session.spawn.

    Returns (backend, mock_spawn_fn) so tests can inspect spawn call args.
    """
    mock_spawn = AsyncMock(
        return_value={
            "output": '{"status": "success", "notes": "done"}',
            "session_id": "child-1",
        }
    )

    coordinator = MagicMock()
    coordinator.get_capability = MagicMock(return_value=mock_spawn)
    coordinator.session = MagicMock()
    coordinator.config = {"agents": {}}

    backend = AmplifierBackend(
        coordinator=coordinator,
        profiles={"anthropic": "test-profile"},
    )
    return backend, mock_spawn


def _make_simple_node(node_id="work"):
    """Create a simple Node for backend tests."""
    return Node(id=node_id, prompt="Do some work")


@pytest.mark.asyncio
async def test_backend_passes_attach_to_when_container_in_context():
    """When internal.env_container_id is set, spawn kwargs include tools-env-all config."""
    backend, mock_spawn = _make_backend_with_mock_spawn()
    node = _make_simple_node()

    context = PipelineContext()
    context.set("internal.env_container_id", "container-xyz")
    context.set("internal.env_type", "docker")

    await backend.run(node, "Do work", context)

    mock_spawn.assert_called_once()
    call_kwargs = mock_spawn.call_args[1]  # keyword args

    # Should have tools list with env-all auto_attach config
    assert "tools" in call_kwargs, "Expected 'tools' in spawn kwargs"
    tools_list = call_kwargs["tools"]
    assert isinstance(tools_list, list)

    # Find the tools-env-all entry
    env_tool_entries = [t for t in tools_list if t.get("module") == "tools-env-all"]
    assert len(env_tool_entries) == 1, (
        f"Expected one tools-env-all entry, got {env_tool_entries}"
    )

    env_config = env_tool_entries[0]["config"]
    assert env_config["auto_attach"]["type"] == "docker"
    assert env_config["auto_attach"]["name"] == "pipeline-workspace"
    assert env_config["auto_attach"]["attach_to"] == "container-xyz"


@pytest.mark.asyncio
async def test_backend_no_attach_when_no_container_in_context():
    """When internal.env_container_id is NOT set, spawn kwargs have no tools-env-all config."""
    backend, mock_spawn = _make_backend_with_mock_spawn()
    node = _make_simple_node()

    context = PipelineContext()
    # No internal.env_container_id set

    await backend.run(node, "Do work", context)

    mock_spawn.assert_called_once()
    call_kwargs = mock_spawn.call_args[1]

    # Either no 'tools' key, or no tools-env-all entry in it
    tools_list = call_kwargs.get("tools", [])
    env_tool_entries = [
        t
        for t in tools_list
        if isinstance(t, dict) and t.get("module") == "tools-env-all"
    ]
    assert len(env_tool_entries) == 0, (
        f"Expected no tools-env-all entry, got {env_tool_entries}"
    )
