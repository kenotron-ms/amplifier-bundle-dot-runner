"""Checkpointing and resume for pipeline execution.

After every node execution, the engine saves a JSON checkpoint so the
pipeline can resume after crashes. The checkpoint captures the current
node, completed nodes with outcomes, context snapshot, and retry counters.

Spec coverage: CHKP-001–006, Section 5.3.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Checkpoint:
    """Serializable snapshot of pipeline execution state.

    Saved after each node completes. Enables crash recovery and resume.

    Spec Section 5.3: Checkpoint model.
    """

    current_node: str
    completed_nodes: dict[str, str]  # node_id -> outcome status
    context_snapshot: dict[str, Any]
    node_outcomes: dict[str, dict[str, Any]]  # node_id -> serialized Outcome
    timestamp: str
    node_retries: dict[str, int] = field(default_factory=dict)


def save_checkpoint(checkpoint: Checkpoint, path: str) -> None:
    """Write checkpoint to a JSON file.

    The JSON is indented for human readability during debugging.

    Spec Section 5.3: Checkpoint.save(path).
    """
    data = {
        "current_node": checkpoint.current_node,
        "completed_nodes": checkpoint.completed_nodes,
        "context": checkpoint.context_snapshot,
        "node_outcomes": checkpoint.node_outcomes,
        "timestamp": checkpoint.timestamp,
        "node_retries": checkpoint.node_retries,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_checkpoint(path: str) -> Checkpoint:
    """Read checkpoint from a JSON file.

    Raises FileNotFoundError if the file does not exist.

    Spec Section 5.3: Checkpoint.load(path).
    """
    with open(path) as f:
        data = json.load(f)
    return Checkpoint(
        current_node=data["current_node"],
        completed_nodes=data.get("completed_nodes", {}),
        context_snapshot=data.get("context", {}),
        node_outcomes=data.get("node_outcomes", {}),
        timestamp=data.get("timestamp", ""),
        node_retries=data.get("node_retries", {}),
    )
