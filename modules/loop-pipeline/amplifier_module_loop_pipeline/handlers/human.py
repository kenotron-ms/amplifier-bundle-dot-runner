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
    AutoApproveInterviewer,
    Interviewer,
    Option,
    Question,
    QuestionType,
)
from ..outcome import Outcome, StageStatus


class HumanGateHandler:
    """Handler for human gate nodes (shape=hexagon).

    Derives choices from outgoing edge labels, presents a
    multiple-choice question to the interviewer, and returns
    an Outcome with the selected label as preferred_label.

    Falls back to AutoApproveInterviewer when no interviewer
    is provided (e.g. in automated/CI environments).
    """

    def __init__(
        self,
        interviewer: Interviewer | None = None,
        hooks: object | None = None,
    ) -> None:
        self._interviewer = interviewer or AutoApproveInterviewer()
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
        4. Map the answer to a preferred_label for edge selection.
        """
        # 1. Derive choices from outgoing edges
        edges = graph.outgoing_edges(node.id)
        choices: list[str] = []
        for edge in edges:
            label = edge.label or edge.to_node
            if label not in choices:
                choices.append(label)

        # 2. Build the question
        prompt = node.attrs.get("prompt") or node.label or f"Human gate: {node.id}"
        options = [Option(key=c, label=c) for c in choices]

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

        # 5. Determine the selected label
        selected = self._resolve_selection(answer, choices)

        return Outcome(
            status=StageStatus.SUCCESS,
            preferred_label=selected,
            context_updates={
                "human.gate.selection": selected,
                "human.gate.node_id": node.id,
            },
            notes=f"Human gate '{node.id}': selected '{selected}'",
        )

    def _resolve_selection(self, answer: Answer, choices: list[str]) -> str | None:
        """Map an Answer to a choice label.

        Falls back to the first choice on SKIPPED, TIMEOUT, or
        unrecognized answers.
        """
        default = choices[0] if choices else None

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

        # String value — try to match against choices
        value_str = str(answer.value)
        if answer.selected_option and answer.selected_option.key in choices:
            return answer.selected_option.key
        if value_str in choices:
            return value_str

        return default
