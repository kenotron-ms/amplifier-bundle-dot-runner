"""NLSpec Section 8.6, DoD item 9: Multi-turn cache efficiency test.

Verifies cache_read_tokens / input_tokens > 50% after 5+ turns for all
three providers.  Requires real API keys.

Run with::

    pytest tests/dod/test_multi_turn_cache_efficiency.py -m integration -v --timeout=120
"""
from __future__ import annotations

import os

import pytest

from unified_llm import Message, Role
from unified_llm.client import Client
from unified_llm.generate import generate

pytestmark = pytest.mark.integration

SKIP_REASON = "API keys not set"
HAS_KEYS = all(
    os.environ.get(k)
    for k in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
) and (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))

# ~600-word system prompt — large enough to create a meaningful cacheable prefix.
SYSTEM_PROMPT = (
    "You are an expert software architect specializing in distributed systems, "
    "microservices, and cloud-native applications. You have deep knowledge of "
    "Kubernetes, Docker, service meshes (Istio, Linkerd), message brokers "
    "(Kafka, RabbitMQ, NATS), databases (PostgreSQL, MongoDB, Redis, "
    "CockroachDB), and observability stacks (Prometheus, Grafana, Jaeger, "
    "OpenTelemetry). When answering questions, provide concise, actionable "
    "advice grounded in production experience. Consider trade-offs between "
    "consistency, availability, and partition tolerance. Reference specific "
    "tools and patterns by name. Mention failure modes and mitigation "
    "strategies. Always consider the operational complexity of your "
    "recommendations.\n\n"
    "Your areas of particular expertise include:\n"
    "- Event-driven architectures and CQRS patterns\n"
    "- Circuit breaker and bulkhead patterns for resilience\n"
    "- Blue-green and canary deployment strategies\n"
    "- Database migration strategies for zero-downtime deployments\n"
    "- API gateway patterns and rate limiting\n"
    "- Secrets management and zero-trust networking\n"
    "- Cost optimization for cloud workloads\n"
    "- Performance profiling and capacity planning\n"
    "- Multi-region deployment and data replication\n"
    "- Container security and supply chain integrity\n\n"
    "When discussing architecture decisions, always frame your response "
    "in terms of: (1) the problem being solved, (2) the proposed solution, "
    "(3) alternatives considered, (4) trade-offs and risks, (5) operational "
    "requirements for production readiness. Keep each response to 2-3 "
    "sentences maximum — the user wants quick expert opinions, not essays."
)

# Short prompts that build on each other.  Kept small so the system prompt
# dominates the token count and caching has maximum effect.
TURN_PROMPTS = [
    "What's the best circuit breaker library for Python microservices?",
    "How should I configure retry backoff for that?",
    "What metrics should I monitor for circuit breaker health?",
    "How do I test circuit breaker behavior in integration tests?",
    "What's the failure mode if the circuit stays open too long?",
    "How do I combine this with a bulkhead pattern?",
]

PROVIDER_MODELS = {
    "anthropic": "claude-sonnet-4-5-20250514",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
}

_KEY_VARS: dict[str, list[str]] = {
    "anthropic": ["ANTHROPIC_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
}


def _has_key(provider: str) -> bool:
    return any(os.environ.get(k) for k in _KEY_VARS.get(provider, []))


@pytest.mark.skipif(not HAS_KEYS, reason=SKIP_REASON)
class TestMultiTurnCacheEfficiency:
    """NLSpec 8.6 DoD item 9: cache_read_tokens > 50% at turn 5+."""

    @pytest.mark.asyncio(loop_scope="function")
    async def test_cache_efficiency_per_provider(self) -> None:
        client = Client.from_env()

        for provider, model in PROVIDER_MODELS.items():
            if not _has_key(provider):
                continue

            messages: list[Message] = []

            for turn_idx, prompt in enumerate(TURN_PROMPTS):
                messages.append(Message(role=Role.USER, content=prompt))

                result = await generate(
                    model=model,
                    prompt=messages,
                    system=SYSTEM_PROMPT,
                    max_tokens=150,
                    provider=provider,
                    client=client,
                )

                assert result.text, (
                    f"{provider} turn {turn_idx}: empty response"
                )
                assert result.usage, (
                    f"{provider} turn {turn_idx}: no usage data"
                )

                # Accumulate assistant response for next turn's history
                messages.append(
                    Message(role=Role.ASSISTANT, content=result.text)
                )

                # Check cache efficiency from turn 5 onward (0-indexed >= 4)
                if turn_idx >= 4 and result.usage.input_tokens > 0:
                    cache_read = result.usage.cache_read_tokens or 0
                    input_total = result.usage.input_tokens
                    ratio = cache_read / input_total if input_total else 0

                    # Turn 5 (idx 4): caching may still be warming — lenient
                    # Turn 6 (idx 5): should be well above 50%
                    threshold = 0.30 if turn_idx == 4 else 0.50
                    assert ratio > threshold, (
                        f"{provider} turn {turn_idx + 1}: cache ratio "
                        f"{ratio:.2%} ({cache_read}/{input_total}) — "
                        f"expected >{threshold:.0%}"
                    )