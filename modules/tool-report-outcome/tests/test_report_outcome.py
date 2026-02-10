"""Tests for report_outcome tool."""

import pytest

from amplifier_module_tool_report_outcome import ReportOutcomeTool


VALID_STATUSES = ["success", "fail", "partial_success", "retry"]


@pytest.mark.asyncio(loop_scope="session")
async def test_report_success():
    """Report a successful outcome with preferred label."""
    tool = ReportOutcomeTool(config={})
    result = await tool.execute(
        {
            "status": "success",
            "preferred_label": "tests_pass",
            "notes": "All 42 tests passing",
        }
    )
    assert result.success
    assert result.output["status"] == "success"
    assert result.output["preferred_label"] == "tests_pass"
    assert result.output["notes"] == "All 42 tests passing"


@pytest.mark.asyncio(loop_scope="session")
async def test_report_fail():
    """Report a failed outcome with failure reason."""
    tool = ReportOutcomeTool(config={})
    result = await tool.execute(
        {
            "status": "fail",
            "failure_reason": "3 tests still failing",
        }
    )
    assert result.success  # Tool itself succeeds
    assert result.output["status"] == "fail"
    assert result.output["failure_reason"] == "3 tests still failing"


@pytest.mark.asyncio(loop_scope="session")
async def test_report_partial_success():
    """Report partial success with context updates."""
    tool = ReportOutcomeTool(config={})
    result = await tool.execute(
        {
            "status": "partial_success",
            "context_updates": {"tests_passing": 39, "tests_failing": 3},
            "notes": "Most tests pass but 3 remaining failures",
        }
    )
    assert result.success
    assert result.output["status"] == "partial_success"
    assert result.output["context_updates"]["tests_passing"] == 39


@pytest.mark.asyncio(loop_scope="session")
async def test_report_retry():
    """Report retry status with suggested next IDs."""
    tool = ReportOutcomeTool(config={})
    result = await tool.execute(
        {
            "status": "retry",
            "suggested_next_ids": ["fix_tests", "review_changes"],
            "failure_reason": "Flaky test detected, retry needed",
        }
    )
    assert result.success
    assert result.output["status"] == "retry"
    assert result.output["suggested_next_ids"] == ["fix_tests", "review_changes"]


@pytest.mark.asyncio(loop_scope="session")
async def test_invalid_status_rejected():
    """Invalid status value returns error."""
    tool = ReportOutcomeTool(config={})
    result = await tool.execute({"status": "invalid_value"})
    assert not result.success
    assert "message" in result.error


@pytest.mark.asyncio(loop_scope="session")
async def test_missing_status_rejected():
    """Missing status parameter returns error."""
    tool = ReportOutcomeTool(config={})
    result = await tool.execute({"notes": "no status provided"})
    assert not result.success
    assert "message" in result.error


@pytest.mark.asyncio(loop_scope="session")
async def test_empty_input_rejected():
    """Empty input returns error."""
    tool = ReportOutcomeTool(config={})
    result = await tool.execute({})
    assert not result.success
    assert "message" in result.error


@pytest.mark.asyncio(loop_scope="session")
async def test_minimal_report():
    """Minimal report with only status is valid."""
    tool = ReportOutcomeTool(config={})
    result = await tool.execute({"status": "success"})
    assert result.success
    assert result.output["status"] == "success"


@pytest.mark.asyncio(loop_scope="session")
async def test_all_valid_statuses():
    """All defined status values are accepted."""
    tool = ReportOutcomeTool(config={})
    for status in VALID_STATUSES:
        result = await tool.execute({"status": status})
        assert result.success, f"Status {status!r} should be valid"
        assert result.output["status"] == status


@pytest.mark.asyncio(loop_scope="session")
async def test_outcome_stored_on_tool():
    """Tool stores the last reported outcome for retrieval."""
    tool = ReportOutcomeTool(config={})
    await tool.execute({"status": "success", "notes": "first"})
    await tool.execute({"status": "fail", "failure_reason": "second"})
    assert tool.last_outcome is not None
    assert tool.last_outcome["status"] == "fail"
    assert tool.last_outcome["failure_reason"] == "second"


@pytest.mark.asyncio(loop_scope="session")
async def test_outcome_not_stored_on_error():
    """Invalid reports don't overwrite the stored outcome."""
    tool = ReportOutcomeTool(config={})
    await tool.execute({"status": "success", "notes": "valid"})
    await tool.execute({"status": "bogus"})
    assert tool.last_outcome["status"] == "success"


@pytest.mark.asyncio(loop_scope="session")
async def test_tool_name_and_description():
    """Tool has correct name and description."""
    tool = ReportOutcomeTool(config={})
    assert tool.name == "report_outcome"
    assert "outcome" in tool.description.lower()


@pytest.mark.asyncio(loop_scope="session")
async def test_tool_input_schema():
    """Tool exposes correct input schema."""
    tool = ReportOutcomeTool(config={})
    schema = tool.input_schema
    assert schema["type"] == "object"
    assert "status" in schema["properties"]
    assert "status" in schema["required"]
    # Optional fields present
    assert "preferred_label" in schema["properties"]
    assert "suggested_next_ids" in schema["properties"]
    assert "context_updates" in schema["properties"]
    assert "notes" in schema["properties"]
    assert "failure_reason" in schema["properties"]


@pytest.mark.asyncio(loop_scope="session")
async def test_output_includes_confirmation():
    """Successful output includes a human-readable confirmation message."""
    tool = ReportOutcomeTool(config={})
    result = await tool.execute({"status": "success"})
    assert result.success
    assert "message" in result.output
    assert isinstance(result.output["message"], str)


@pytest.mark.asyncio(loop_scope="session")
async def test_event_emitted_via_coordinator():
    """When a coordinator with hooks is present, an outcome:reported event is emitted."""
    from unittest.mock import AsyncMock, MagicMock

    hooks = MagicMock()
    hooks.emit = AsyncMock()
    coordinator = MagicMock()
    coordinator.hooks = hooks

    tool = ReportOutcomeTool(config={}, coordinator=coordinator)
    result = await tool.execute(
        {"status": "success", "preferred_label": "done", "notes": "all good"}
    )
    assert result.success
    hooks.emit.assert_called_once()
    event_name = hooks.emit.call_args[0][0]
    event_data = hooks.emit.call_args[0][1]
    assert event_name == "outcome:reported"
    assert event_data["status"] == "success"
    assert event_data["preferred_label"] == "done"


@pytest.mark.asyncio(loop_scope="session")
async def test_no_event_without_coordinator():
    """Tool works without a coordinator (no event emission, no crash)."""
    tool = ReportOutcomeTool(config={})
    result = await tool.execute({"status": "success"})
    assert result.success


@pytest.mark.asyncio(loop_scope="session")
async def test_context_updates_type_validation():
    """context_updates must be a dict if provided."""
    tool = ReportOutcomeTool(config={})
    result = await tool.execute({"status": "success", "context_updates": "not_a_dict"})
    assert not result.success
    assert "message" in result.error


@pytest.mark.asyncio(loop_scope="session")
async def test_suggested_next_ids_type_validation():
    """suggested_next_ids must be a list if provided."""
    tool = ReportOutcomeTool(config={})
    result = await tool.execute(
        {"status": "success", "suggested_next_ids": "not_a_list"}
    )
    assert not result.success
    assert "message" in result.error
