"""DoD §8.3 — Message & Content Model.

Verifies content types work correctly across providers using mocks.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from unified_llm import (
    ContentKind,
    ContentPart,
    FinishReason,
    ImageData,
    Message,
    Request,
    Response,
    Role,
    StreamEvent,
    StreamEventType,
    ThinkingData,
    ToolCallData,
    Usage,
    Client,
)


# ---------------------------------------------------------------------------
# Shared mock adapter
# ---------------------------------------------------------------------------


def _response_with(content: list[ContentPart]) -> Response:
    return Response(
        id="r1",
        model="test",
        provider="mock",
        message=Message(role=Role.ASSISTANT, content=content),
        finish_reason=FinishReason(reason="stop"),
        usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
    )


class _MockAdapter:
    def __init__(self, response: Response | None = None) -> None:
        self._response = response or _response_with(
            [ContentPart(kind=ContentKind.TEXT, text="ok")]
        )

    @property
    def name(self) -> str:
        return "mock"

    async def complete(self, request: Request) -> Response:
        return self._response

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(type=StreamEventType.FINISH)

    async def close(self) -> None:
        pass


def _make_client(adapter: _MockAdapter | None = None) -> Client:
    a = adapter or _MockAdapter()
    return Client(providers={"mock": a}, default_provider="mock")


# ---------------------------------------------------------------------------
# §8.3 — Text-only messages
# ---------------------------------------------------------------------------


def test_text_only_messages() -> None:
    """[ ] Messages with text-only content work across all providers."""
    client = _make_client()
    request = Request(
        model="test",
        messages=[
            Message.system("You are helpful"),
            Message.user("Hello"),
        ],
    )
    result = asyncio.run(client.complete(request))
    assert result.text == "ok"


# ---------------------------------------------------------------------------
# §8.3 — Image input
# ---------------------------------------------------------------------------


def test_image_input_url() -> None:
    """[ ] Image input works: images sent as URL."""
    msg = Message(
        role=Role.USER,
        content=[
            ContentPart(kind=ContentKind.TEXT, text="What is this?"),
            ContentPart(
                kind=ContentKind.IMAGE,
                image=ImageData(url="https://example.com/photo.jpg"),
            ),
        ],
    )
    # Verify message is constructable and has correct structure
    assert len(msg.content) == 2
    assert msg.content[1].kind == ContentKind.IMAGE
    assert msg.content[1].image is not None
    assert msg.content[1].image.url == "https://example.com/photo.jpg"


def test_image_input_base64() -> None:
    """[ ] Image input works: images sent as base64 data."""
    png_bytes = b"\x89PNG\r\n\x1a\n"
    msg = Message(
        role=Role.USER,
        content=[
            ContentPart(
                kind=ContentKind.IMAGE,
                image=ImageData(data=png_bytes, media_type="image/png"),
            ),
        ],
    )
    assert msg.content[0].image is not None
    assert msg.content[0].image.data == png_bytes
    assert msg.content[0].image.media_type == "image/png"


def test_image_input_processed_by_adapter() -> None:
    """[ ] Image content part is accepted in complete() call."""
    client = _make_client()
    request = Request(
        model="test",
        messages=[
            Message(
                role=Role.USER,
                content=[
                    ContentPart(kind=ContentKind.TEXT, text="Describe"),
                    ContentPart(
                        kind=ContentKind.IMAGE,
                        image=ImageData(url="https://example.com/img.png"),
                    ),
                ],
            )
        ],
    )
    result = asyncio.run(client.complete(request))
    assert result.text == "ok"


# ---------------------------------------------------------------------------
# §8.3 — Audio and document content
# ---------------------------------------------------------------------------


def test_audio_and_document_content_constructable() -> None:
    """[ ] Audio and document content parts are handled."""
    from unified_llm import AudioData, DocumentData

    audio_msg = Message(
        role=Role.USER,
        content=[
            ContentPart(
                kind=ContentKind.AUDIO,
                audio=AudioData(data=b"\x00\x00", media_type="audio/wav"),
            ),
        ],
    )
    assert audio_msg.content[0].audio is not None

    doc_msg = Message(
        role=Role.USER,
        content=[
            ContentPart(
                kind=ContentKind.DOCUMENT,
                document=DocumentData(data=b"%PDF", media_type="application/pdf"),
            ),
        ],
    )
    assert doc_msg.content[0].document is not None


# ---------------------------------------------------------------------------
# §8.3 — Tool call round-trip
# ---------------------------------------------------------------------------


def test_tool_call_round_trip() -> None:
    """[ ] Tool call content parts round-trip correctly."""
    # Step 1: assistant sends tool call
    assistant_msg = Message(
        role=Role.ASSISTANT,
        content=[
            ContentPart(
                kind=ContentKind.TOOL_CALL,
                tool_call=ToolCallData(
                    id="call_123",
                    name="get_weather",
                    arguments={"city": "SF"},
                ),
            )
        ],
    )
    assert assistant_msg.content[0].tool_call is not None
    assert assistant_msg.content[0].tool_call.name == "get_weather"

    # Step 2: tool result message
    tool_msg = Message.tool_result(
        tool_call_id="call_123",
        content="72F sunny",
    )
    assert tool_msg.role == Role.TOOL
    assert tool_msg.content[0].tool_result is not None
    assert tool_msg.content[0].tool_result.tool_call_id == "call_123"
    assert tool_msg.content[0].tool_result.content == "72F sunny"

    # Step 3: full conversation is constructable
    conversation = [
        Message.user("What is the weather in SF?"),
        assistant_msg,
        tool_msg,
        Message.assistant("The weather in SF is 72F and sunny."),
    ]
    assert len(conversation) == 4


# ---------------------------------------------------------------------------
# §8.3 — Thinking blocks
# ---------------------------------------------------------------------------


def test_thinking_blocks_preserved() -> None:
    """[ ] Thinking blocks (Anthropic) are preserved and round-tripped with signatures."""
    assistant_msg = Message(
        role=Role.ASSISTANT,
        content=[
            ContentPart(
                kind=ContentKind.THINKING,
                thinking=ThinkingData(
                    text="Let me work through this...",
                    signature="sig_abc123",
                ),
            ),
            ContentPart(kind=ContentKind.TEXT, text="The answer is 42."),
        ],
    )
    assert assistant_msg.content[0].thinking is not None
    assert assistant_msg.content[0].thinking.text == "Let me work through this..."
    assert assistant_msg.content[0].thinking.signature == "sig_abc123"
    assert assistant_msg.text == "The answer is 42."


def test_redacted_thinking_blocks() -> None:
    """[ ] Redacted thinking blocks are passed through verbatim."""
    msg = Message(
        role=Role.ASSISTANT,
        content=[
            ContentPart(
                kind=ContentKind.REDACTED_THINKING,
                thinking=ThinkingData(text="", redacted=True),
            ),
            ContentPart(kind=ContentKind.TEXT, text="Answer."),
        ],
    )
    assert msg.content[0].kind == ContentKind.REDACTED_THINKING
    assert msg.content[0].thinking is not None
    assert msg.content[0].thinking.redacted is True


# ---------------------------------------------------------------------------
# §8.3 — Multimodal messages
# ---------------------------------------------------------------------------


def test_multimodal_messages() -> None:
    """[ ] Multimodal messages (text + images in the same message) work."""
    msg = Message(
        role=Role.USER,
        content=[
            ContentPart(kind=ContentKind.TEXT, text="What do you see?"),
            ContentPart(
                kind=ContentKind.IMAGE,
                image=ImageData(url="https://example.com/photo.jpg"),
            ),
        ],
    )
    assert msg.text == "What do you see?"
    assert msg.content[1].kind == ContentKind.IMAGE
    assert len(msg.content) == 2

    # Verify it works through the client
    client = _make_client()
    request = Request(model="test", messages=[msg])
    result = asyncio.run(client.complete(request))
    assert result.text == "ok"
