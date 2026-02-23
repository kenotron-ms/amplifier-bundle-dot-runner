"""Tests for execution environment lifecycle in PipelineOrchestrator.

Verifies that the orchestrator optionally creates/destroys an execution
environment around the pipeline engine run when configured.
"""

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from amplifier_module_loop_pipeline import PipelineOrchestrator
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
