"""Report Outcome Tool Module for Amplifier.

Allows a coding agent to report structured outcome data that the pipeline
can use for routing decisions. This is how structured data flows from the
coding agent back to the pipeline through the execute() -> str boundary.
"""

# Amplifier module metadata
__amplifier_module_type__ = "tool"

import logging
from typing import Any

__all__ = ["ReportOutcomeTool", "mount"]

logger = logging.getLogger(__name__)

VALID_STATUSES = frozenset({"success", "fail", "partial_success", "retry"})


class ReportOutcomeTool:
    """Report structured outcome data for pipeline routing.

    The coding agent calls this tool to signal completion status,
    preferred routing, context updates, and notes back to the
    pipeline orchestrator.
    """

    name = "report_outcome"
    description = (
        "Report the outcome of your work. Use this to signal success, failure, "
        "partial success, or a need to retry. The pipeline uses this to decide "
        "which step to run next."
    )

    def __init__(self, config: dict[str, Any], coordinator: Any = None):
        """Initialize ReportOutcomeTool.

        Args:
            config: Module configuration.
            coordinator: Optional module coordinator (used for event emission).
        """
        self.config = config
        self.coordinator = coordinator
        self.last_outcome: dict[str, Any] | None = None

    @property
    def input_schema(self) -> dict:
        """Return JSON schema for tool parameters."""
        return {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": list(VALID_STATUSES),
                    "description": (
                        "Outcome status: 'success', 'fail', "
                        "'partial_success', or 'retry'"
                    ),
                },
                "preferred_label": {
                    "type": "string",
                    "description": ("Which edge label to follow in the pipeline graph"),
                },
                "suggested_next_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Explicit next node IDs to consider",
                },
                "context_updates": {
                    "type": "object",
                    "description": ("Key-value pairs to merge into pipeline context"),
                },
                "notes": {
                    "type": "string",
                    "description": "Human-readable execution summary",
                },
                "failure_reason": {
                    "type": "string",
                    "description": "Reason for failure (when status is fail)",
                },
            },
            "required": ["status"],
        }

    async def execute(self, input: dict[str, Any]) -> Any:
        """Execute the report_outcome tool.

        Args:
            input: Outcome data with required 'status' field and optional
                preferred_label, suggested_next_ids, context_updates,
                notes, and failure_reason.

        Returns:
            ToolResult confirming the outcome was recorded.
        """
        from amplifier_core import ToolResult

        # Validate status is present
        status = input.get("status")
        if not status:
            return ToolResult(
                success=False,
                error={"message": "status parameter is required"},
            )

        # Validate status value
        if status not in VALID_STATUSES:
            return ToolResult(
                success=False,
                error={
                    "message": (
                        f"Invalid status: {status!r}. "
                        f"Must be one of: {', '.join(sorted(VALID_STATUSES))}"
                    )
                },
            )

        # Validate context_updates type if provided
        context_updates = input.get("context_updates")
        if context_updates is not None and not isinstance(context_updates, dict):
            return ToolResult(
                success=False,
                error={
                    "message": "context_updates must be an object (dict), "
                    f"got {type(context_updates).__name__}"
                },
            )

        # Validate suggested_next_ids type if provided
        suggested_next_ids = input.get("suggested_next_ids")
        if suggested_next_ids is not None and not isinstance(suggested_next_ids, list):
            return ToolResult(
                success=False,
                error={
                    "message": "suggested_next_ids must be an array (list), "
                    f"got {type(suggested_next_ids).__name__}"
                },
            )

        # Build the outcome record
        outcome: dict[str, Any] = {"status": status}

        # Include optional fields only if provided
        for key in (
            "preferred_label",
            "suggested_next_ids",
            "context_updates",
            "notes",
            "failure_reason",
        ):
            value = input.get(key)
            if value is not None:
                outcome[key] = value

        # Store the outcome for retrieval by the pipeline
        self.last_outcome = outcome

        # Emit event if coordinator with hooks is available
        if self.coordinator and hasattr(self.coordinator, "hooks"):
            try:
                await self.coordinator.hooks.emit("outcome:reported", outcome)
            except Exception:
                logger.debug("Failed to emit outcome:reported event", exc_info=True)

        # Build confirmation message
        message = f"Outcome reported: {status}"
        if input.get("preferred_label"):
            message += f" (preferred_label={input['preferred_label']})"
        if input.get("notes"):
            message += f" - {input['notes']}"

        return ToolResult(
            success=True,
            output={"message": message, **outcome},
        )


async def mount(coordinator: Any, config: dict[str, Any] | None = None) -> None:
    """Mount the report_outcome tool.

    Args:
        coordinator: Module coordinator for registering tools.
        config: Module configuration.
    """
    config = config or {}
    tool = ReportOutcomeTool(config, coordinator)
    await coordinator.mount("tools", tool, name=tool.name)
    logger.info("Mounted report_outcome tool")
