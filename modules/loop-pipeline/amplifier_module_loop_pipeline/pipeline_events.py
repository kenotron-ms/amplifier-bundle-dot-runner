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
