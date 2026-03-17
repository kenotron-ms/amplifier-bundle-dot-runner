"""Wait-for-human gate handler.

Maps to the spec's ``wait.human`` node (shape=hexagon). The handler
derives choices from outgoing edge labels, presents the question to
a human via the Interviewer interface, and routes based on the
human's selection.

Spec coverage: HUMAN-001–008, Section 4.10.
"""

from __future__ import annotations

from ..context import PipelineContext
from ..graph import Graph, Node
from ..interviewer import (
    Answer,
    AnswerValue,
    Interviewer,
    Option,
    Question,
    QuestionType,
)
from ..outcome import Outcome, StageStatus

# L-11: Patterns for accelerator key extraction from edge labels.
# Matches: "[Y] Yes", "[OK] Okay", "1) Option One", "2. Option Two"
import re

_BRACKET_KEY_RE = re.compile(r"^\[([^\]]+)\]\s+")
_NUMBER_KEY_RE = re.compile(r"^(\d+)[).]\s+")


def _parse_accelerator_key(label: str) -> str:
    """Extract accelerator key from a label, or return the full label.

    Supports patterns:
        [Y] Yes       -> "Y"
        [OK] Okay     -> "OK"
        1) Option One -> "1"
        2. Option Two -> "2"
        Approve       -> "Approve"  (no accelerator)
    """
    if not label:
        return label
    m = _BRACKET_KEY_RE.match(label)
    if m:
        return m.group(1)
    m = _NUMBER_KEY_RE.match(label)
    if m:
        return m.group(1)
    return label


class HumanGateHandler:
    """Handler for human gate nodes (shape=hexagon).

    Derives choices from outgoing edge labels, presents a
    multiple-choice question to the interviewer, and returns
    an Outcome with suggested_next_ids for unambiguous routing.

    Requires an explicit Interviewer instance. Pass
    AutoApproveInterviewer() for CI/testing environments.
    """

    def __init__(
        self,
        interviewer: Interviewer | None = None,
        hooks: object | None = None,
    ) -> None:
        self._interviewer = interviewer
        self._hooks = hooks

    async def _emit(self, event_name: str, data: dict) -> None:  # type: ignore[type-arg]
        """Emit an event via hooks, if provided."""
        if self._hooks is not None:
            await self._hooks.emit(event_name, data)  # type: ignore[union-attr]

    async def execute(
        self,
        node: Node,
        context: PipelineContext,
        graph: Graph,
        logs_root: str,
    ) -> Outcome:
        """Present choices to a human and route based on selection.

        1. Derive choices from outgoing edge labels.
        2. Build a Question with those choices.
        3. Ask the interviewer.
        4. Map the answer to suggested_next_ids for edge selection.
        """
        if self._interviewer is None:
            raise ValueError(
                "HumanGateHandler requires an Interviewer but none was provided. "
                "Pass interviewer=AutoApproveInterviewer() explicitly for auto-approve, "
                "or interviewer=ConsoleInterviewer() for interactive use."
            )
        # 1. Derive choices from outgoing edges and build label-to-node mapping
        edges = graph.outgoing_edges(node.id)
        choices: list[str] = []
        # Map each choice label to the list of target node IDs (M-12)
        label_to_targets: dict[str, list[str]] = {}
        for edge in edges:
            label = edge.label or edge.to_node
            if label not in choices:
                choices.append(label)
            label_to_targets.setdefault(label, []).append(edge.to_node)

        # 2. Build the question with accelerator keys (L-11)
        prompt = node.attrs.get("prompt") or node.label or f"Human gate: {node.id}"
        # Extract accelerator keys from labels for option keys
        key_to_label: dict[str, str] = {}
        options: list[Option] = []
        for c in choices:
            key = _parse_accelerator_key(c)
            key_to_label[key] = c
            options.append(Option(key=key, label=c))

        if choices:
            question = Question(
                text=prompt,
                type=QuestionType.MULTIPLE_CHOICE,
                options=options,
                stage=node.id,
            )
        else:
            # No labeled edges — use a simple confirmation
            question = Question(
                text=prompt,
                type=QuestionType.CONFIRMATION,
                stage=node.id,
            )

        # 3. Emit interview started event
        from ..pipeline_events import (
            PIPELINE_INTERVIEW_COMPLETED,
            PIPELINE_INTERVIEW_STARTED,
            PIPELINE_INTERVIEW_TIMEOUT,
        )

        await self._emit(
            PIPELINE_INTERVIEW_STARTED,
            {"node_id": node.id, "question": question.text},
        )

        # 4. Ask the interviewer
        answer = self._interviewer.ask(question)

        await self._emit(
            PIPELINE_INTERVIEW_COMPLETED,
            {
                "node_id": node.id,
                "answer": str(answer.value)
                if hasattr(answer, "value")
                else str(answer),
            },
        )

        # 5. Emit timeout event if the interviewer timed out
        if (
            isinstance(answer.value, AnswerValue)
            and answer.value == AnswerValue.TIMEOUT
        ):
            await self._emit(
                PIPELINE_INTERVIEW_TIMEOUT,
                {
                    "node_id": node.id,
                    "prompt": prompt,
                    "timeout": True,
                },
            )

        # 6. M-13: SKIPPED answer returns FAIL per spec
        if (
            isinstance(answer.value, AnswerValue)
            and answer.value == AnswerValue.SKIPPED
        ):
            return Outcome(
                status=StageStatus.FAIL,
                context_updates={
                    "human.gate.selected": None,
                    "human.gate.label": node.label,
                },
                notes=f"Human gate '{node.id}': interaction was skipped",
            )

        # 7. Determine the selected label and map to target node IDs (M-12)
        selected = self._resolve_selection(answer, choices, key_to_label)
        target_ids = label_to_targets.get(selected or "", []) if selected else []

        return Outcome(
            status=StageStatus.SUCCESS,
            suggested_next_ids=target_ids if target_ids else None,
            context_updates={
                "human.gate.selected": selected,
                "human.gate.label": node.label,
            },
            notes=f"Human gate '{node.id}': selected '{selected}'",
        )

    def _resolve_selection(
        self,
        answer: Answer,
        choices: list[str],
        key_to_label: dict[str, str] | None = None,
    ) -> str | None:
        """Map an Answer to a choice label.

        Falls back to the first choice on SKIPPED, TIMEOUT, or
        unrecognized answers. Supports accelerator key lookup via
        key_to_label mapping (L-11).
        """
        default = choices[0] if choices else None
        key_map = key_to_label or {}

        # Handle enum AnswerValues that mean "no real selection"
        if isinstance(answer.value, AnswerValue):
            if answer.value in (
                AnswerValue.SKIPPED,
                AnswerValue.TIMEOUT,
                AnswerValue.YES,
            ):
                return default
            if answer.value == AnswerValue.NO:
                # "No" maps to the second choice if available, else default
                return choices[1] if len(choices) > 1 else default

        # String value — try to match against choices or accelerator keys
        value_str = str(answer.value)
        # Check selected_option key -> map back to full label
        if answer.selected_option:
            opt_key = answer.selected_option.key
            if opt_key in key_map:
                return key_map[opt_key]
            if opt_key in choices:
                return opt_key
        # Check if value matches an accelerator key (L-11)
        if value_str in key_map:
            return key_map[value_str]
        if value_str in choices:
            return value_str

        return default
