"""DoD §8.1 — Core Infrastructure.

Each checklist item from the spec becomes a test function.
Uses mocked adapters — no real API keys needed.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from unittest.mock import patch

from unified_llm import (
    Client,
    ConfigurationError,
    FinishReason,
    Message,
    Request,
    Response,
    StreamEvent,
    StreamEventType,
    Usage,
    get_default_client,
    get_latest_model,
    get_model_info,
    list_models,
    set_default_client,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_API_KEY_NAMES = frozenset(
    {"ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"}
)


def _env_without_keys() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in _API_KEY_NAMES}


def _make_response(provider: str = "mock") -> Response:
    return Response(
        id="r1",
        model="test-model",
        provider=provider,
        message=Message.assistant("hello"),
        finish_reason=FinishReason(reason="stop"),
        usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
    )


class _MockAdapter:
    """Minimal ProviderAdapter for DoD testing."""

    def __init__(self, name: str = "mock") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def complete(self, request: Request) -> Response:
        return _make_response(self._name)

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(type=StreamEventType.TEXT_DELTA, delta="hi")
        yield StreamEvent(
            type=StreamEventType.FINISH,
            finish_reason=FinishReason(reason="stop"),
            usage=Usage(input_tokens=5, output_tokens=2, total_tokens=7),
        )

    async def close(self) -> None:
        pass

    async def initialize(self) -> None:
        pass

    def supports_tool_choice(self, mode: str) -> bool:
        return True


# ---------------------------------------------------------------------------
# §8.1 Checklist Tests
# ---------------------------------------------------------------------------


def test_client_from_env() -> None:
    """[ ] Client can be constructed from environment variables."""
    env = _env_without_keys()
    env["OPENAI_API_KEY"] = "test-key"
    with patch.dict(os.environ, env, clear=True):
        client = Client.from_env()
        assert "openai" in client.providers
        assert client.default_provider == "openai"


def test_client_programmatic() -> None:
    """[ ] Client can be constructed programmatically."""
    adapter = _MockAdapter("test_provider")
    client = Client(
        providers={"test_provider": adapter},
        default_provider="test_provider",
    )
    assert "test_provider" in client.providers
    result = asyncio.run(client.complete(Request(model="m", messages=[Message.user("hi")])))
    assert result.text == "hello"


def test_provider_routing() -> None:
    """[ ] Provider routing dispatches correctly."""
    alpha = _MockAdapter("alpha")
    beta = _MockAdapter("beta")
    client = Client(providers={"alpha": alpha, "beta": beta})

    result = asyncio.run(
        client.complete(Request(model="m", messages=[Message.user("hi")], provider="beta"))
    )
    assert result.provider == "beta"


def test_default_provider() -> None:
    """[ ] Default provider used when omitted."""
    adapter = _MockAdapter("default_prov")
    client = Client(providers={"default_prov": adapter}, default_provider="default_prov")
    result = asyncio.run(
        client.complete(Request(model="m", messages=[Message.user("hi")]))
    )
    assert result.provider == "default_prov"


def test_no_provider_raises() -> None:
    """[ ] ConfigurationError when no provider configured."""
    client = Client(providers={})
    try:
        asyncio.run(
            client.complete(Request(model="m", messages=[Message.user("hi")]))
        )
        assert False, "Should raise ConfigurationError"
    except ConfigurationError:
        pass


def test_middleware_order() -> None:
    """[ ] Middleware chain order correct."""
    order: list[str] = []

    async def mw_a(request, next_fn):  # noqa: ANN001
        order.append("A_req")
        response = await next_fn(request)
        order.append("A_resp")
        return response

    async def mw_b(request, next_fn):  # noqa: ANN001
        order.append("B_req")
        response = await next_fn(request)
        order.append("B_resp")
        return response

    adapter = _MockAdapter()
    client = Client(
        providers={"mock": adapter},
        default_provider="mock",
        middleware=[mw_a, mw_b],
    )
    asyncio.run(client.complete(Request(model="m", messages=[Message.user("hi")])))
    # Request order: registration order. Response: reverse.
    assert order == ["A_req", "B_req", "B_resp", "A_resp"]


def test_default_client() -> None:
    """[ ] Module-level default client works."""
    import unified_llm.client as client_mod

    original = client_mod._default_client
    try:
        client_mod._default_client = None
        adapter = _MockAdapter()
        client = Client(providers={"mock": adapter}, default_provider="mock")
        set_default_client(client)
        assert get_default_client() is client
    finally:
        client_mod._default_client = original


def test_model_catalog() -> None:
    """[ ] Model catalog populated and working."""
    # list_models returns models
    models = list_models()
    assert len(models) > 0

    # get_model_info works
    info = get_model_info(models[0].id)
    assert info is not None
    assert info.id == models[0].id

    # get_latest_model works
    providers_seen = {m.provider for m in models}
    for provider in providers_seen:
        latest = get_latest_model(provider)
        assert latest is not None
        assert latest.provider == provider
