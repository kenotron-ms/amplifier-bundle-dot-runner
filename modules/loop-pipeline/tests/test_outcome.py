"""Tests for the Outcome model.

Covers spec Section 5.2 (Outcome) and StageStatus enum.
"""

from amplifier_module_loop_pipeline.outcome import Outcome, StageStatus


# --- StageStatus enum ---


def test_stage_status_values():
    """StageStatus should have all five values from spec Section 5.2."""
    assert StageStatus.SUCCESS.value == "success"
    assert StageStatus.PARTIAL_SUCCESS.value == "partial_success"
    assert StageStatus.RETRY.value == "retry"
    assert StageStatus.FAIL.value == "fail"
    assert StageStatus.SKIPPED.value == "skipped"


def test_stage_status_count():
    """Exactly five status values per spec."""
    assert len(StageStatus) == 5


# --- Outcome construction ---


def test_success_outcome():
    """Basic success outcome."""
    o = Outcome(status=StageStatus.SUCCESS, preferred_label="tests_pass")
    assert o.status == StageStatus.SUCCESS
    assert o.preferred_label == "tests_pass"
    assert o.failure_reason is None
    assert o.notes is None
    assert o.context_updates is None
    assert o.suggested_next_ids is None


def test_fail_with_reason():
    """Failure outcome with reason."""
    o = Outcome(status=StageStatus.FAIL, failure_reason="3 tests failing")
    assert o.status == StageStatus.FAIL
    assert o.failure_reason == "3 tests failing"


def test_outcome_with_context_updates():
    """Outcome can carry context updates for the engine to apply."""
    o = Outcome(
        status=StageStatus.SUCCESS,
        context_updates={"last_stage": "implement", "last_response": "done"},
    )
    assert o.context_updates == {"last_stage": "implement", "last_response": "done"}


def test_outcome_with_suggested_next_ids():
    """Outcome can suggest next node IDs for edge selection."""
    o = Outcome(
        status=StageStatus.SUCCESS,
        suggested_next_ids=["validate", "exit"],
    )
    assert o.suggested_next_ids == ["validate", "exit"]


def test_outcome_with_notes():
    """Outcome can carry human-readable execution notes."""
    o = Outcome(
        status=StageStatus.PARTIAL_SUCCESS,
        notes="Completed 3 of 5 subtasks",
    )
    assert o.notes == "Completed 3 of 5 subtasks"


def test_retry_outcome():
    """Retry outcome with reason."""
    o = Outcome(
        status=StageStatus.RETRY,
        failure_reason="Rate limited, retrying",
    )
    assert o.status == StageStatus.RETRY
    assert o.failure_reason == "Rate limited, retrying"


def test_skipped_outcome():
    """Skipped outcome."""
    o = Outcome(status=StageStatus.SKIPPED, notes="Condition not met")
    assert o.status == StageStatus.SKIPPED


def test_outcome_is_success_property():
    """Convenience: is_success covers SUCCESS and PARTIAL_SUCCESS."""
    assert Outcome(status=StageStatus.SUCCESS).is_success is True
    assert Outcome(status=StageStatus.PARTIAL_SUCCESS).is_success is True
    assert Outcome(status=StageStatus.FAIL).is_success is False
    assert Outcome(status=StageStatus.RETRY).is_success is False
    assert Outcome(status=StageStatus.SKIPPED).is_success is False
