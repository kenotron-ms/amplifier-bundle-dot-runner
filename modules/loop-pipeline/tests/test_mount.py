"""Tests for module mount function."""

import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_mount_registers_orchestrator():
    """mount() should register an orchestrator with the coordinator."""
    coordinator = MagicMock()
    coordinator.mount = AsyncMock()

    from amplifier_module_loop_pipeline import mount

    await mount(coordinator, config={})
    coordinator.mount.assert_called_once()
    args = coordinator.mount.call_args
    assert args[0][0] == "orchestrator"


@pytest.mark.asyncio
async def test_mount_with_dot_source():
    """mount() should accept inline DOT source via config."""
    coordinator = MagicMock()
    coordinator.mount = AsyncMock()

    from amplifier_module_loop_pipeline import mount

    await mount(coordinator, config={"dot_source": "digraph { A -> B }"})
    coordinator.mount.assert_called_once()
    orchestrator = coordinator.mount.call_args[0][1]
    assert orchestrator.config.get("dot_source") == "digraph { A -> B }"


@pytest.mark.asyncio
async def test_mount_with_dot_file():
    """mount() should accept a DOT file path via config."""
    coordinator = MagicMock()
    coordinator.mount = AsyncMock()

    from amplifier_module_loop_pipeline import mount

    await mount(coordinator, config={"dot_file": "pipeline.dot"})
    orchestrator = coordinator.mount.call_args[0][1]
    assert orchestrator.config.get("dot_file") == "pipeline.dot"


@pytest.mark.asyncio
async def test_orchestrator_has_execute_method():
    """The mounted orchestrator must have an execute() method."""
    coordinator = MagicMock()
    coordinator.mount = AsyncMock()

    from amplifier_module_loop_pipeline import mount

    await mount(coordinator, config={})
    orchestrator = coordinator.mount.call_args[0][1]
    assert hasattr(orchestrator, "execute")
    assert callable(orchestrator.execute)


@pytest.mark.asyncio
async def test_mount_with_none_config():
    """mount() should handle None config gracefully."""
    coordinator = MagicMock()
    coordinator.mount = AsyncMock()

    from amplifier_module_loop_pipeline import mount

    await mount(coordinator, config=None)
    coordinator.mount.assert_called_once()
