"""Pipeline event constants and helpers.

Defines the event names emitted by the pipeline engine at key execution
points.  The engine calls ``await hooks.emit(event_name, data)`` when a
hooks object is provided.

Spec coverage: EVT-001–008, Section 9.6.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Pipeline lifecycle
# ---------------------------------------------------------------------------
PIPELINE_START: str = "pipeline:start"
PIPELINE_COMPLETE: str = "pipeline:complete"

# ---------------------------------------------------------------------------
# Node lifecycle
# ---------------------------------------------------------------------------
PIPELINE_NODE_START: str = "pipeline:node_start"
PIPELINE_NODE_COMPLETE: str = "pipeline:node_complete"

# ---------------------------------------------------------------------------
# Edge selection
# ---------------------------------------------------------------------------
PIPELINE_EDGE_SELECTED: str = "pipeline:edge_selected"

# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------
PIPELINE_CHECKPOINT: str = "pipeline:checkpoint"

# ---------------------------------------------------------------------------
# Goal gates
# ---------------------------------------------------------------------------
PIPELINE_GOAL_GATE_CHECK: str = "pipeline:goal_gate_check"

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
PIPELINE_ERROR: str = "pipeline:error"

# ---------------------------------------------------------------------------
# Parallel execution (spec Section 9.6)
# ---------------------------------------------------------------------------
PIPELINE_PARALLEL_STARTED: str = "pipeline:parallel_started"
PIPELINE_PARALLEL_BRANCH_STARTED: str = "pipeline:parallel_branch_started"
PIPELINE_PARALLEL_BRANCH_COMPLETED: str = "pipeline:parallel_branch_completed"
PIPELINE_PARALLEL_COMPLETED: str = "pipeline:parallel_completed"

# ---------------------------------------------------------------------------
# Human interaction (spec Section 9.6)
# ---------------------------------------------------------------------------
PIPELINE_INTERVIEW_STARTED: str = "pipeline:interview_started"
PIPELINE_INTERVIEW_COMPLETED: str = "pipeline:interview_completed"
PIPELINE_INTERVIEW_TIMEOUT: str = "pipeline:interview_timeout"

# ---------------------------------------------------------------------------
# Retry lifecycle (spec Section 9.6)
# ---------------------------------------------------------------------------
PIPELINE_STAGE_RETRYING: str = "pipeline:stage_retrying"
PIPELINE_STAGE_FAILED: str = "pipeline:stage_failed"
