"""Tests for the wait-for-human gate handler and interviewer.

Spec coverage: HUMAN-001–008, INTV-001–010, Section 6.
"""

import pytest

from amplifier_module_loop_pipeline.context import PipelineContext
from amplifier_module_loop_pipeline.graph import Edge, Graph, Node
from amplifier_module_loop_pipeline.handlers.human import HumanGateHandler
from amplifier_module_loop_pipeline.interviewer import (
    Answer,
    AnswerValue,
    AutoApproveInterviewer,
    CallbackInterviewer,
    Interviewer,
    Option,
    Question,
    QuestionType,
    QueueInterviewer,
)
from amplifier_module_loop_pipeline.outcome import Outcome, StageStatus


def _make_graph_with_human_gate() -> Graph:
    """Graph with a human gate node and two outgoing edges."""
    return Graph(
        name="test",
        nodes={
            "start": Node(id="start", shape="Mdiamond"),
            "review": Node(
                id="review",
                shape="hexagon",
                label="Approve changes?",
                attrs={"prompt": "Do you approve this code?"},
            ),
            "deploy": Node(id="deploy", prompt="Deploy to prod"),
            "fix": Node(id="fix", prompt="Fix issues"),
            "exit": Node(id="exit", shape="Msquare"),
        },
        edges=[
            Edge(from_node="start", to_node="review"),
            Edge(from_node="review", to_node="deploy", label="Approve"),
            Edge(from_node="review", to_node="fix", label="Reject"),
            Edge(from_node="deploy", to_node="exit"),
            Edge(from_node="fix", to_node="exit"),
        ],
    )


def _make_context() -> PipelineContext:
    return PipelineContext()


# --- Interviewer models ---


class TestQuestionModel:
    """INTV-001: Question model has required fields."""

    def test_question_has_text_and_type(self):
        q = Question(text="Approve?", type=QuestionType.MULTIPLE_CHOICE)
        assert q.text == "Approve?"
        assert q.type == QuestionType.MULTIPLE_CHOICE

    def test_question_with_options(self):
        opts = [
            Option(key="Y", label="Yes, approve"),
            Option(key="N", label="No, reject"),
        ]
        q = Question(
            text="Approve?",
            type=QuestionType.MULTIPLE_CHOICE,
            options=opts,
            stage="review",
        )
        assert len(q.options) == 2
        assert q.options[0].key == "Y"
        assert q.stage == "review"

    def test_question_with_timeout(self):
        q = Question(
            text="Approve?",
            type=QuestionType.YES_NO,
            timeout_seconds=30.0,
        )
        assert q.timeout_seconds == 30.0


class TestAnswerModel:
    """INTV-002: Answer model."""

    def test_answer_with_value(self):
        a = Answer(value=AnswerValue.YES)
        assert a.value == AnswerValue.YES

    def test_answer_with_selected_option(self):
        opt = Option(key="Y", label="Yes")
        a = Answer(value="Y", selected_option=opt)
        assert a.selected_option is not None
        assert a.selected_option.key == "Y"

    def test_answer_with_text(self):
        a = Answer(value="custom", text="My feedback")
        assert a.text == "My feedback"


# --- AutoApproveInterviewer ---


class TestAutoApproveInterviewer:
    """INTV-004: AutoApproveInterviewer for automated testing."""

    def test_yes_no_returns_yes(self):
        interviewer = AutoApproveInterviewer()
        q = Question(text="Continue?", type=QuestionType.YES_NO)
        answer = interviewer.ask(q)
        assert answer.value == AnswerValue.YES

    def test_confirmation_returns_yes(self):
        interviewer = AutoApproveInterviewer()
        q = Question(text="Confirm?", type=QuestionType.CONFIRMATION)
        answer = interviewer.ask(q)
        assert answer.value == AnswerValue.YES

    def test_multiple_choice_returns_first_option(self):
        interviewer = AutoApproveInterviewer()
        opts = [
            Option(key="A", label="Approve"),
            Option(key="R", label="Reject"),
        ]
        q = Question(text="Choose", type=QuestionType.MULTIPLE_CHOICE, options=opts)
        answer = interviewer.ask(q)
        assert answer.value == "A"
        assert answer.selected_option is not None
        assert answer.selected_option.key == "A"

    def test_freeform_returns_auto_approved(self):
        interviewer = AutoApproveInterviewer()
        q = Question(text="Comments?", type=QuestionType.FREEFORM)
        answer = interviewer.ask(q)
        assert answer.text == "auto-approved"


# --- QueueInterviewer ---


class TestQueueInterviewer:
    """INTV-007: QueueInterviewer for deterministic testing."""

    def test_returns_queued_answers(self):
        answers = [
            Answer(value=AnswerValue.YES),
            Answer(value=AnswerValue.NO),
        ]
        interviewer = QueueInterviewer(answers)
        a1 = interviewer.ask(Question(text="Q1", type=QuestionType.YES_NO))
        a2 = interviewer.ask(Question(text="Q2", type=QuestionType.YES_NO))
        assert a1.value == AnswerValue.YES
        assert a2.value == AnswerValue.NO

    def test_empty_queue_returns_skipped(self):
        interviewer = QueueInterviewer([])
        answer = interviewer.ask(Question(text="Q", type=QuestionType.YES_NO))
        assert answer.value == AnswerValue.SKIPPED


# --- CallbackInterviewer ---


class TestCallbackInterviewer:
    """INTV-006: CallbackInterviewer delegates to callback."""

    def test_delegates_to_callback(self):
        def my_callback(q: Question) -> Answer:
            return Answer(value=AnswerValue.NO)

        interviewer = CallbackInterviewer(my_callback)
        answer = interviewer.ask(Question(text="Ok?", type=QuestionType.YES_NO))
        assert answer.value == AnswerValue.NO


# --- HumanGateHandler ---


class TestHumanGateHandler:
    """HUMAN-001–008: Wait-for-human handler."""

    @pytest.mark.asyncio
    async def test_derives_choices_from_outgoing_edges(self):
        """HUMAN-002: Choices come from outgoing edge labels."""
        graph = _make_graph_with_human_gate()
        node = graph.nodes["review"]
        handler = HumanGateHandler(interviewer=AutoApproveInterviewer())
        outcome = await handler.execute(node, _make_context(), graph, "/tmp")
        # AutoApprove picks the first choice; either way it should succeed
        assert outcome.status == StageStatus.SUCCESS
        # preferred_label should match one of the edge labels
        assert outcome.preferred_label in ("Approve", "Reject")

    @pytest.mark.asyncio
    async def test_auto_approve_picks_first_choice(self):
        """HUMAN-003: AutoApprove picks first edge label."""
        graph = _make_graph_with_human_gate()
        node = graph.nodes["review"]
        handler = HumanGateHandler(interviewer=AutoApproveInterviewer())
        outcome = await handler.execute(node, _make_context(), graph, "/tmp")
        assert outcome.preferred_label == "Approve"

    @pytest.mark.asyncio
    async def test_queue_interviewer_selects_specific_choice(self):
        """HUMAN-004: QueueInterviewer routes based on queued answer."""
        graph = _make_graph_with_human_gate()
        node = graph.nodes["review"]
        # Queue an answer that matches the second option ("Reject")
        interviewer = QueueInterviewer([
            Answer(value="Reject", selected_option=Option(key="Reject", label="Reject")),
        ])
        handler = HumanGateHandler(interviewer=interviewer)
        outcome = await handler.execute(node, _make_context(), graph, "/tmp")
        assert outcome.status == StageStatus.SUCCESS
        assert outcome.preferred_label == "Reject"

    @pytest.mark.asyncio
    async def test_skipped_answer_uses_default_choice(self):
        """HUMAN-005: SKIPPED answer falls back to first choice."""
        graph = _make_graph_with_human_gate()
        node = graph.nodes["review"]
        interviewer = QueueInterviewer([])  # Will return SKIPPED
        handler = HumanGateHandler(interviewer=interviewer)
        outcome = await handler.execute(node, _make_context(), graph, "/tmp")
        assert outcome.status == StageStatus.SUCCESS
        assert outcome.preferred_label == "Approve"  # first edge label

    @pytest.mark.asyncio
    async def test_timeout_answer_uses_default_choice(self):
        """HUMAN-006: TIMEOUT answer falls back to default choice."""
        graph = _make_graph_with_human_gate()
        node = graph.nodes["review"]
        interviewer = QueueInterviewer([Answer(value=AnswerValue.TIMEOUT)])
        handler = HumanGateHandler(interviewer=interviewer)
        outcome = await handler.execute(node, _make_context(), graph, "/tmp")
        assert outcome.status == StageStatus.SUCCESS
        assert outcome.preferred_label == "Approve"  # first = default

    @pytest.mark.asyncio
    async def test_uses_prompt_attr_for_question_text(self):
        """HUMAN-007: Question text comes from node prompt attr or label."""
        graph = _make_graph_with_human_gate()
        node = graph.nodes["review"]
        captured_questions: list[Question] = []

        def capture_callback(q: Question) -> Answer:
            captured_questions.append(q)
            return Answer(value="Approve", selected_option=Option(key="Approve", label="Approve"))

        handler = HumanGateHandler(interviewer=CallbackInterviewer(capture_callback))
        await handler.execute(node, _make_context(), graph, "/tmp")
        assert len(captured_questions) == 1
        assert "approve" in captured_questions[0].text.lower() or "code" in captured_questions[0].text.lower()

    @pytest.mark.asyncio
    async def test_no_outgoing_edges_returns_success(self):
        """HUMAN-008: No outgoing edges still completes (edge case)."""
        graph = Graph(
            name="test",
            nodes={
                "start": Node(id="start", shape="Mdiamond"),
                "gate": Node(id="gate", shape="hexagon", label="Wait"),
                "exit": Node(id="exit", shape="Msquare"),
            },
            edges=[
                Edge(from_node="start", to_node="gate"),
                Edge(from_node="gate", to_node="exit"),  # no label
            ],
        )
        handler = HumanGateHandler(interviewer=AutoApproveInterviewer())
        outcome = await handler.execute(
            graph.nodes["gate"], _make_context(), graph, "/tmp"
        )
        assert outcome.status == StageStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_default_interviewer_is_auto_approve(self):
        """When no interviewer is provided, handler uses AutoApprove."""
        graph = _make_graph_with_human_gate()
        node = graph.nodes["review"]
        handler = HumanGateHandler()  # No interviewer arg
        outcome = await handler.execute(node, _make_context(), graph, "/tmp")
        assert outcome.status == StageStatus.SUCCESS
        assert outcome.preferred_label == "Approve"

    @pytest.mark.asyncio
    async def test_sets_context_updates(self):
        """Handler sets human.gate.* context keys."""
        graph = _make_graph_with_human_gate()
        node = graph.nodes["review"]
        handler = HumanGateHandler(interviewer=AutoApproveInterviewer())
        outcome = await handler.execute(node, _make_context(), graph, "/tmp")
        assert outcome.context_updates is not None
        assert "human.gate.selection" in outcome.context_updates


# --- Handler registration ---


class TestHumanHandlerRegistration:
    """Handler registry resolves hexagon shape to HumanGateHandler."""

    def test_registry_resolves_human_handler(self):
        from amplifier_module_loop_pipeline.handlers import HandlerRegistry

        registry = HandlerRegistry()
        node = Node(id="gate", shape="hexagon")
        handler = registry.get(node)
        assert isinstance(handler, HumanGateHandler)


@pytest.mark.asyncio
async def test_human_handler_emits_interview_events():
    """HumanGateHandler must emit interview lifecycle events."""
    from amplifier_module_loop_pipeline.pipeline_events import (
        PIPELINE_INTERVIEW_COMPLETED,
        PIPELINE_INTERVIEW_STARTED,
    )

    emitted: list[tuple[str, dict]] = []

    class MockHooks:
        async def emit(self, event_name, data):
            emitted.append((event_name, data))

    handler = HumanGateHandler(
        interviewer=AutoApproveInterviewer(), hooks=MockHooks()
    )

    node = Node(id="gate", shape="hexagon", label="Approve?")
    graph = Graph(
        name="test",
        nodes={"gate": node, "yes": Node(id="yes", shape="box")},
        edges=[Edge(from_node="gate", to_node="yes", label="Yes")],
    )

    ctx = PipelineContext()
    await handler.execute(node, ctx, graph, "/tmp/test")

    event_names = [e[0] for e in emitted]
    assert PIPELINE_INTERVIEW_STARTED in event_names
    assert PIPELINE_INTERVIEW_COMPLETED in event_names
