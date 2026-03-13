"""Tool node handler — executes shell commands.

Reads the tool_command from node attributes, executes it via subprocess,
captures stdout, and returns SUCCESS or FAIL based on exit code.

Supported attributes:
    tool_command   — Shell command to execute (required)
    parse_json     — If "true", json.loads() the stdout; if result is a dict,
                     inject each key-value into context and context_updates;
                     on JSONDecodeError logs WARNING and continues (SUCCESS)
    tool_env       — Comma-separated list of context variable names to expose
                     as uppercase environment variables to the subprocess

Spec coverage: TOOL-001–004, Section 4.10.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from ..context import PipelineContext
from ..graph import Graph, Node
from ..outcome import Outcome, StageStatus

logger = logging.getLogger(__name__)

_LOG_TRUNCATE_CHARS = 200


class ToolHandler:
    """Handler for tool nodes (shape=parallelogram).

    Executes an external command configured via the node's
    ``tool_command`` attribute.
    """

    async def execute(
        self,
        node: Node,
        context: PipelineContext,
        graph: Graph,
        logs_root: str,
    ) -> Outcome:
        """Execute a tool command and return the outcome.

        Writes command.txt and output.txt to the stage log directory.
        Stores stdout in context as ``tool.output``.
        """
        command = node.attrs.get("tool_command", "")
        if not command:
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason="No tool_command specified on node",
            )

        # Write command to logs
        stage_dir = os.path.join(logs_root, node.id)
        os.makedirs(stage_dir, exist_ok=True)
        _write_file(os.path.join(stage_dir, "command.txt"), command)

        # M-16: Read timeout from node attribute (seconds)
        timeout_s: float | None = None
        if node.timeout is not None:
            timeout_s = float(node.timeout)

        # Build environment for subprocess: handle tool_env attribute
        env: dict[str, str] | None = None
        tool_env_attr = node.attrs.get("tool_env", "")
        if tool_env_attr:
            env = dict(os.environ)
            var_names = [v.strip() for v in tool_env_attr.split(",") if v.strip()]
            for var_name in var_names:
                value = context.get(var_name)
                if value is not None:
                    env[var_name.upper()] = str(value)

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout_s
                )
            except asyncio.TimeoutError:
                # Kill the process on timeout
                proc.kill()
                await proc.wait()
                return Outcome(
                    status=StageStatus.FAIL,
                    failure_reason=f"Timeout after {timeout_s}s: {command}",
                )
            stdout_text = stdout_bytes.decode(errors="replace") if stdout_bytes else ""
            stderr_text = stderr_bytes.decode(errors="replace") if stderr_bytes else ""

            # Write output to logs
            _write_file(
                os.path.join(stage_dir, "output.txt"),
                stdout_text + stderr_text,
            )

            if proc.returncode != 0:
                return Outcome(
                    status=StageStatus.FAIL,
                    failure_reason=(
                        f"Command exited with code {proc.returncode}: "
                        f"{stderr_text.strip() or stdout_text.strip()}"
                    ),
                )

            # Store stdout in context
            context.set("tool.output", stdout_text)
            context_updates: dict[str, Any] = {"tool.output": stdout_text}

            # Handle parse_json attribute
            if node.attrs.get("parse_json", "") == "true":
                try:
                    parsed = json.loads(stdout_text)
                    if isinstance(parsed, dict):
                        for key, value in parsed.items():
                            context.set(key, value)
                            context_updates[key] = value
                except json.JSONDecodeError:
                    logger.warning(
                        "parse_json: failed to parse JSON output from node %r: %r",
                        node.id,
                        stdout_text[:_LOG_TRUNCATE_CHARS],
                    )

            return Outcome(
                status=StageStatus.SUCCESS,
                context_updates=context_updates,
                notes=f"Tool completed: {command}",
            )
        except Exception as e:
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason=str(e),
            )


def _write_file(path: str, content: str) -> None:
    """Write content to a file."""
    with open(path, "w") as f:
        f.write(content)
