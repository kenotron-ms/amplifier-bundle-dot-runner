"""Interviewer interface and implementations for human-in-the-loop.

All human interaction in Attractor goes through an Interviewer interface.
This abstraction allows the pipeline to present questions to a human and
receive answers through any frontend: CLI, web UI, Slack, or a
programmatic queue for testing.

Spec coverage: INTV-001-010, Section 6.1-6.4
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol, runtime_checkable


class QuestionType(Enum):
    """Types of questions the pipeline can ask a human.

    Spec Section 6.2: QuestionType.
    """

    YES_NO = "yes_no"
    MULTIPLE_CHOICE = "multiple_choice"
    FREEFORM = "freeform"
    CONFIRMATION = "confirmation"


class AnswerValue(Enum):
    """Predefined answer values.

    Spec Section 6.3: AnswerValue.
    """

    YES = "yes"
    NO = "no"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


@dataclass
class Option:
    """A selectable option for multiple-choice questions.

    Spec Section 6.2: Option.
    """

    key: str
    label: str


@dataclass
class Question:
    """A question to present to a human operator.

    Spec Section 6.2: Question model.
    """

    text: str
    type: QuestionType
    options: list[Option] = field(default_factory=list)
    default: Answer | None = None
    timeout_seconds: float | None = None
    stage: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Answer:
    """A human's response to a question.

    Spec Section 6.3: Answer model.
    """

    value: str | AnswerValue = ""
    selected_option: Option | None = None
    text: str = ""


@runtime_checkable
class Interviewer(Protocol):
    """Interface for human interaction.

    Spec Section 6.1: Interviewer interface.
    """

    def ask(self, question: Question) -> Answer: ...


class AutoApproveInterviewer:
    """Always approves — used for automated testing and CI/CD.

    Spec Section 6.4: AutoApproveInterviewer.
    """

    def ask(self, question: Question) -> Answer:
        if question.type in (QuestionType.YES_NO, QuestionType.CONFIRMATION):
            return Answer(value=AnswerValue.YES)
        if question.type == QuestionType.MULTIPLE_CHOICE and question.options:
            first = question.options[0]
            return Answer(value=first.key, selected_option=first)
        return Answer(value="auto-approved", text="auto-approved")


class QueueInterviewer:
    """Reads answers from a pre-filled queue — for deterministic testing.

    Spec Section 6.4: QueueInterviewer.
    """

    def __init__(self, answers: list[Answer]) -> None:
        self._answers: deque[Answer] = deque(answers)

    def ask(self, question: Question) -> Answer:
        if self._answers:
            return self._answers.popleft()
        return Answer(value=AnswerValue.SKIPPED)


class CallbackInterviewer:
    """Delegates to a provided callback function.

    Spec Section 6.4: CallbackInterviewer.
    """

    def __init__(self, callback: Callable[[Question], Answer]) -> None:
        self._callback = callback

    def ask(self, question: Question) -> Answer:
        return self._callback(question)


class ConsoleInterviewer:
    """Interactive stdin/stdout interviewer for CLI use.

    Presents questions on stdout and reads answers from stdin.

    Spec Section 6.4: ConsoleInterviewer (M-24).
    """

    def ask(self, question: Question) -> Answer:
        """Present a question on stdout and read an answer from stdin."""
        import sys

        # Display the question
        print(f"\n[Pipeline Question] {question.text}")

        if question.type == QuestionType.YES_NO:
            print("  (yes/no)", end="")
            if question.default:
                print(f" [default: {question.default.value}]", end="")
            print(": ", end="", flush=True)
            raw = sys.stdin.readline().strip().lower()
            if not raw and question.default:
                return question.default
            if raw in ("y", "yes"):
                return Answer(value=AnswerValue.YES)
            if raw in ("n", "no"):
                return Answer(value=AnswerValue.NO)
            return Answer(value=AnswerValue.SKIPPED)

        if question.type == QuestionType.CONFIRMATION:
            print("  (confirm/cancel)", end="")
            print(": ", end="", flush=True)
            raw = sys.stdin.readline().strip().lower()
            if raw in ("c", "confirm", "y", "yes"):
                return Answer(value=AnswerValue.YES)
            return Answer(value=AnswerValue.NO)

        if question.type == QuestionType.MULTIPLE_CHOICE:
            for opt in question.options:
                print(f"  [{opt.key}] {opt.label}")
            print("Choose: ", end="", flush=True)
            raw = sys.stdin.readline().strip()
            for opt in question.options:
                if raw == opt.key:
                    return Answer(value=opt.key, selected_option=opt)
            return Answer(value=AnswerValue.SKIPPED)

        # FREEFORM
        print("  Answer: ", end="", flush=True)
        raw = sys.stdin.readline().strip()
        return Answer(value=raw, text=raw)


class RecordingInterviewer:
    """Records all interactions for replay.

    Optionally seeded with preset answers. When presets are exhausted,
    returns SKIPPED. All (question, answer) pairs are recorded and can
    be retrieved via ``get_recordings()``.

    Spec Section 6.4: RecordingInterviewer (M-24).
    """

    def __init__(self, answers: list[Answer] | None = None) -> None:
        self._answers: deque[Answer] = deque(answers or [])
        self._recordings: list[tuple[Question, Answer]] = []

    def ask(self, question: Question) -> Answer:
        """Return next preset answer (or SKIPPED) and record the interaction."""
        if self._answers:
            answer = self._answers.popleft()
        else:
            answer = Answer(value=AnswerValue.SKIPPED)
        self._recordings.append((question, answer))
        return answer

    def get_recordings(self) -> list[tuple[Question, Answer]]:
        """Return all recorded (question, answer) pairs."""
        return list(self._recordings)
