"""Codergen handler — the default handler for LLM task nodes.

Reads the node's prompt, expands template variables, calls the LLM
backend, writes prompt/response/status to the logs directory, and
returns the outcome.

Spec coverage: CODER-001-011, Section 4.5
"""

from __future__ import annotations

import json
import os
from typing import Any, Protocol, runtime_checkable

from ..context import PipelineContext
from ..graph import Graph, Node
from ..outcome import Outcome, StageStatus
from ..transforms import expand_goal_variable, expand_params


@runtime_checkable
class CodergenBackend(Protocol):
    """Interface for LLM execution backends.

    Spec Section 4.5: CodergenBackend Interface.
    """

    async def run(
        self, node: Node, prompt: str, context: PipelineContext
    ) -> str | Outcome: ...


class CodergenHandler:
    """Handler for codergen (LLM task) nodes.

    Spec Section 4.5: Codergen Handler.
    """

    def __init__(self, backend: Any | None = None) -> None:
        self._backend = backend

    async def execute(
        self,
        node: Node,
        context: PipelineContext,
        graph: Graph,
        logs_root: str,
    ) -> Outcome:
        """Execute a codergen node.

        1. Build prompt (expand $goal)
        2. Write prompt to logs
        3. Call backend (or simulate)
        4. Write response and status to logs
        5. Return outcome
        """
        # 1. Build prompt
        prompt = node.prompt or node.attrs.get("llm_prompt") or node.label
        prompt = _expand_variables(prompt, graph, context)

        # 2. Write prompt to logs
        stage_dir = os.path.join(logs_root, node.id)
        os.makedirs(stage_dir, exist_ok=True)
        _write_file(os.path.join(stage_dir, "prompt.md"), prompt)

        # 3. Call LLM backend
        if self._backend is None:
            raise ValueError(
                "CodergenHandler requires a backend but none was provided. "
                "Pass backend=MockBackend() explicitly if you want simulated responses for testing."
            )
        try:
            result = await self._backend.run(node, prompt, context)
            if isinstance(result, Outcome):
                _write_status(stage_dir, result)
                return result
            response_text = str(result)
        except Exception as e:
            outcome = Outcome(status=StageStatus.FAIL, failure_reason=str(e))
            _write_status(stage_dir, outcome)
            return outcome

        # 4. Write response to logs
        _write_file(os.path.join(stage_dir, "response.md"), response_text)

        # 5. Build and write outcome
        outcome = Outcome(
            status=StageStatus.SUCCESS,
            notes=f"Stage completed: {node.id}",
            context_updates={
                "last_stage": node.id,
                "last_response": response_text[:200],
            },
        )
        _write_status(stage_dir, outcome)
        return outcome


def _expand_variables(prompt: str, graph: Graph, context: PipelineContext) -> str:
    """Expand template variables in a prompt string.

    L-17: Delegates to the shared expand_goal_variable utility for $goal.
    Runtime variable: $context resolves to the previous node's response
    (stored as ``last_response`` in the pipeline context).
    P7: Plain context keys (no "." in name) are expanded as $param tokens,
    enabling context.* attrs injected by parent folder/house nodes.

    Spec Section 4.5: Variable expansion.
    """
    # $goal — static for the whole pipeline (also expanded at parse time in transforms)
    context_goal = context.get("graph.goal") or ""
    result = expand_goal_variable(prompt, graph.goal, context_goal)

    # $context — runtime, changes after each node completes
    if "$context" in result:
        last_response = context.get("last_response", "") or ""
        result = result.replace("$context", str(last_response))

    # P7: Expand plain context keys injected via context.* parent node attrs.
    # Only expands keys without "." (namespaced keys like graph.goal are excluded).
    if "$" in result:
        plain_params = {
            k: str(v) for k, v in context.snapshot().items() if "." not in k
        }
        if plain_params:
            result = expand_params(result, plain_params)

    return result


def _write_file(path: str, content: str) -> None:
    """Write content to a file."""
    with open(path, "w") as f:
        f.write(content)


def _write_status(stage_dir: str, outcome: Outcome) -> None:
    """Write status.json for a stage outcome.

    M-19: Uses 'outcome' as the primary field name per spec Appendix C.
    Keeps 'status' as backward-compat alias.
    """
    data = {
        "outcome": outcome.status.value,
        "status": outcome.status.value,  # backward compat
        "preferred_next_label": outcome.preferred_label,
        "suggested_next_ids": outcome.suggested_next_ids,
        "context_updates": outcome.context_updates,
        "notes": outcome.notes,
        "failure_reason": outcome.failure_reason,
    }
    _write_file(os.path.join(stage_dir, "status.json"), json.dumps(data, indent=2))
