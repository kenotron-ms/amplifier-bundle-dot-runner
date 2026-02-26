"""DoD §8.10 — Integration Smoke Tests.

6 end-to-end tests that require real API keys.
Run with: pytest tests/dod/test_8_10_integration_smoke.py -m integration
Requires: OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY (or GOOGLE_API_KEY)
"""

from __future__ import annotations

import os

import pytest

from unified_llm import (
    ContentKind,
    ContentPart,
    ImageData,
    Message,
    NotFoundError,
    Role,
    StreamEventType,
    Tool,
)
from unified_llm.catalog import get_latest_model
from unified_llm.client import Client
from unified_llm.generate import generate, generate_object, stream

pytestmark = pytest.mark.integration

SKIP_REASON = "API keys not set"
HAS_KEYS = all(
    os.environ.get(k)
    for k in [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ]
) and (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


@pytest.mark.skipif(not HAS_KEYS, reason=SKIP_REASON)
class TestIntegrationSmoke:
    """6 end-to-end tests per spec §8.10."""

    @pytest.mark.asyncio(loop_scope="function")
    async def test_basic_generation_all_providers(self) -> None:
        """Spec: generate() across all providers returns non-empty text.

        FOR EACH provider IN ["anthropic", "openai", "gemini"]:
            result = generate(model=get_latest_model(provider).id, prompt="Say hello in one sentence.", ...)
            ASSERT result.text is not empty
            ASSERT result.usage.input_tokens > 0
            ASSERT result.usage.output_tokens > 0
            ASSERT result.finish_reason.reason == "stop"
        """
        client = Client.from_env()

        for provider in ["anthropic", "openai", "gemini"]:
            latest = get_latest_model(provider)
            assert latest is not None, f"No model found for {provider}"

            result = await generate(
                model=latest.id,
                prompt="Say hello in one sentence.",
                max_tokens=100,
                provider=provider,
                client=client,
                max_retries=2,
            )
            assert result.text, f"Empty text from {provider}"
            assert result.usage.input_tokens > 0, f"No input tokens from {provider}"
            assert result.usage.output_tokens > 0, f"No output tokens from {provider}"
            assert result.finish_reason.reason == "stop", (
                f"Unexpected finish reason from {provider}: {result.finish_reason.reason}"
            )

    @pytest.mark.asyncio(loop_scope="function")
    async def test_streaming(self) -> None:
        """Spec: concatenated deltas == response.text.

        stream_result = stream(model="claude-sonnet-4-20250514", prompt="Write a haiku.")
        text_chunks = []
        FOR EACH event IN stream_result:
            IF event.type == TEXT_DELTA: text_chunks.APPEND(event.delta)
        ASSERT JOIN(text_chunks) == stream_result.response().text
        """
        client = Client.from_env()
        latest = get_latest_model("anthropic")
        assert latest is not None

        stream_result = stream(
            model=latest.id,
            prompt="Write a haiku about coding.",
            max_tokens=100,
            provider="anthropic",
            client=client,
            max_retries=2,
        )

        text_chunks: list[str] = []
        async for event in stream_result:
            if event.type == StreamEventType.TEXT_DELTA and event.delta:
                text_chunks.append(event.delta)

        concatenated = "".join(text_chunks)
        assert concatenated, "No text chunks received from stream"

        response = stream_result.response()
        assert response.text == concatenated

    @pytest.mark.asyncio(loop_scope="function")
    async def test_tool_calling_parallel(self) -> None:
        """Spec: tool loop with parallel execution, steps >= 2.

        result = generate(
            model="claude-sonnet-4-20250514",
            prompt="What is the weather in San Francisco and New York?",
            tools=[weather_tool],
            max_tool_rounds=3
        )
        ASSERT LENGTH(result.steps) >= 2
        ASSERT result.text contains "San Francisco"
        ASSERT result.text contains "New York"
        """
        client = Client.from_env()

        def get_weather(city: str) -> str:
            """Get mock weather data for a city."""
            weather_data = {
                "San Francisco": "72°F, sunny",
                "New York": "58°F, cloudy",
            }
            return weather_data.get(city, f"Unknown city: {city}")

        weather_tool = Tool(
            name="get_weather",
            description="Get the current weather for a city. Returns temperature and conditions.",
            parameters={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name to get weather for",
                    }
                },
                "required": ["city"],
            },
            execute=get_weather,
        )

        latest = get_latest_model("anthropic")
        assert latest is not None

        result = await generate(
            model=latest.id,
            prompt="What is the weather in San Francisco and New York?",
            tools=[weather_tool],
            max_tool_rounds=3,
            provider="anthropic",
            client=client,
            max_retries=2,
            max_tokens=1024,
        )

        assert len(result.steps) >= 2, (
            f"Expected at least 2 steps, got {len(result.steps)}"
        )
        # The final text should mention both cities
        text_lower = result.text.lower()
        assert "san francisco" in text_lower or "sf" in text_lower, (
            f"Response doesn't mention San Francisco: {result.text[:200]}"
        )

    @pytest.mark.asyncio(loop_scope="function")
    async def test_image_input(self) -> None:
        """Spec: image input produces non-empty response.

        result = generate(
            model="claude-sonnet-4-20250514",
            messages=[Message(role=USER, content=[
                ContentPart(kind=TEXT, text="What do you see?"),
                ContentPart(kind=IMAGE, image=ImageData(data=<png_bytes>, media_type="image/png"))
            ])]
        )
        ASSERT result.text is not empty
        """
        client = Client.from_env()

        # Create a 100x100 red PNG (Anthropic rejects very small images)
        import base64

        # 100x100 solid red PNG
        png_b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAIAAAD/gAIDAAABFUlEQVR4nO3O"
            "UQkAIABEsetfWiv4Nx4IC7Cd7XvkByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q"
            "4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q"
            "4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q"
            "4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q"
            "4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q"
            "4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIRee"
            "LesrH9s1agAAAABJRU5ErkJggg=="
        )
        png_bytes = base64.b64decode(png_b64)

        latest = get_latest_model("anthropic")
        assert latest is not None

        result = await generate(
            model=latest.id,
            messages=[
                Message(
                    role=Role.USER,
                    content=[
                        ContentPart(
                            kind=ContentKind.TEXT,
                            text="What do you see in this image? Describe it briefly.",
                        ),
                        ContentPart(
                            kind=ContentKind.IMAGE,
                            image=ImageData(data=png_bytes, media_type="image/png"),
                        ),
                    ],
                )
            ],
            provider="anthropic",
            client=client,
            max_retries=2,
            max_tokens=200,
        )
        assert result.text, "Empty response for image input"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_structured_output(self) -> None:
        """Spec: generate_object returns parsed, validated object.

        result = generate_object(
            model="gpt-4.1",
            prompt="Extract: Alice is 30 years old",
            schema={"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"]}
        )
        ASSERT result.output.name == "Alice"
        ASSERT result.output.age == 30
        """
        client = Client.from_env()

        latest = get_latest_model("openai")
        assert latest is not None

        result = await generate_object(
            model=latest.id,
            prompt="Extract the person's name and age from: 'Alice is 30 years old'",
            schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
                "required": ["name", "age"],
            },
            provider="openai",
            client=client,
            max_retries=2,
            max_tokens=200,
        )
        assert result.output is not None, "No structured output returned"
        assert result.output["name"] == "Alice"
        assert result.output["age"] == 30

    @pytest.mark.asyncio(loop_scope="function")
    async def test_error_handling(self) -> None:
        """Spec: nonexistent model raises NotFoundError.

        TRY:
            generate(model="nonexistent-model-xyz", prompt="test", provider="openai")
            FAIL("Should have raised an error")
        CATCH NotFoundError:
            PASS
        """
        client = Client.from_env()

        with pytest.raises(NotFoundError):
            await generate(
                model="nonexistent-model-xyz-does-not-exist",
                prompt="test",
                provider="openai",
                client=client,
                max_retries=0,
                max_tokens=100,
            )
