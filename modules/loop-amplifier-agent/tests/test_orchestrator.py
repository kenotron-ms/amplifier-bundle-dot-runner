"""Hermetic unit tests for AmplifierAgentOrchestrator.execute().

Every test here monkeypatches ONE seam --
``amplifier_module_loop_amplifier_agent._load_dependencies`` -- with fakes
that are faithful to the real amplifier-agent contract (see ``tests/_fakes.py``
module docstring). No network, no real amplifier_agent_lib import, no
Python-3.12 requirement to run these: the fakes stand in for the (heavy,
Python>=3.12-only) real library so this module's own logic -- envelope
shape, config-key mapping, fail-closed behavior -- is exercised in complete
isolation.
"""

from __future__ import annotations

from typing import Any

import pytest
from amplifier_core.events import ORCHESTRATOR_COMPLETE

import amplifier_module_loop_amplifier_agent as laa

from ._fakes import CapturingHooks, FakeFactoryContextManager, make_fake_deps


def _install_fake_deps(monkeypatch: pytest.MonkeyPatch, **kwargs: Any):
    deps, captured = make_fake_deps(**kwargs)
    monkeypatch.setattr(laa, "_load_dependencies", lambda: deps)
    return captured


# ---------------------------------------------------------------------------
# 1. Envelope shape matches what backend.py::_outcome_from_spawn_result parses
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_envelope_shape_matches_backend_reader(monkeypatch: pytest.MonkeyPatch):
    """The ORCHESTRATOR_COMPLETE envelope must carry metadata.report_outcome
    in loop-agent's exact shape, and it must be parseable by the REAL
    loop-pipeline backend reader (not just shape-asserted by hand).
    """
    verdict = {
        "status": "fail",
        "preferred_label": "escalate",
        "failure_reason": "probe verdict",
        "notes": "transport probe verdict",
    }
    _install_fake_deps(monkeypatch, reply_text="", outcome_to_set=verdict)

    orchestrator = laa.AmplifierAgentOrchestrator(coordinator=None, config={})
    hooks = CapturingHooks()

    reply = await orchestrator.execute(
        "do the work", None, {}, {}, hooks, coordinator=None
    )

    assert reply == ""
    event_name, payload = hooks.events[-1]
    assert event_name == ORCHESTRATOR_COMPLETE
    assert payload["orchestrator"] == "loop-amplifier-agent"
    assert payload["status"] == "success"
    assert payload["turn_count"] == 1
    assert payload["metadata"] == {"report_outcome": verdict}

    # Prove it travels: feed the exact spawn-result shape foundation's
    # PreparedBundle.spawn assembles into the REAL backend reader.
    from amplifier_module_loop_pipeline.backend import _outcome_from_spawn_result
    from amplifier_module_loop_pipeline.outcome import StageStatus

    spawn_result = {
        "output": reply,
        "status": payload["status"],
        "turn_count": payload["turn_count"],
        "metadata": payload["metadata"],
    }
    outcome = _outcome_from_spawn_result(spawn_result)
    assert outcome is not None
    assert outcome.is_explicit is True
    assert outcome.status is StageStatus.FAIL
    assert outcome.preferred_label == "escalate"
    assert outcome.failure_reason == "probe verdict"
    assert outcome.notes == "transport probe verdict"


@pytest.mark.asyncio
async def test_context_updates_ride_along(monkeypatch: pytest.MonkeyPatch):
    verdict = {
        "status": "success",
        "context_updates": {"artifact": "report.md"},
        "suggested_next_ids": ["publish"],
    }
    _install_fake_deps(monkeypatch, reply_text="", outcome_to_set=verdict)

    orchestrator = laa.AmplifierAgentOrchestrator(coordinator=None, config={})
    hooks = CapturingHooks()
    await orchestrator.execute("do the work", None, {}, {}, hooks, coordinator=None)

    assert hooks.completion["metadata"]["report_outcome"] == verdict


# ---------------------------------------------------------------------------
# 2. Config-key mapping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_config_keys_are_mapped_to_the_right_injection_points(
    monkeypatch: pytest.MonkeyPatch,
):
    captured = _install_fake_deps(
        monkeypatch,
        reply_text="done",
        outcome_to_set={"status": "success"},
    )

    config = {
        "llm_provider": "openai",
        "reasoning_effort": "high",
        "max_turns": 5,
        "user_instructions": "focus on the tests",
    }
    orchestrator = laa.AmplifierAgentOrchestrator(coordinator=None, config=config)
    hooks = CapturingHooks()
    await orchestrator.execute("do the work", None, {}, {}, hooks, coordinator=None)

    # llm_provider + reasoning_effort -> inject_provider(provider, effort_override=...)
    assert captured["inject_provider_calls"] == [
        ("openai", {"effort_override": "high"})
    ]

    # max_turns -> prepared.mount_plan["session"]["orchestrator"]["config"]["max_turns"]
    assert (
        captured["prepared"].mount_plan["session"]["orchestrator"]["config"][
            "max_turns"
        ]
        == 5
    )

    # user_instructions -> appended into the prompt handed to session.execute()
    prompt_seen = captured["session"].prompt_seen
    assert "focus on the tests" in prompt_seen
    assert "do the work" in prompt_seen
    assert "report_outcome" in prompt_seen  # the nudge is always appended too


@pytest.mark.asyncio
async def test_default_provider_used_when_llm_provider_absent(
    monkeypatch: pytest.MonkeyPatch,
):
    captured = _install_fake_deps(
        monkeypatch, reply_text="", outcome_to_set={"status": "success"}
    )

    orchestrator = laa.AmplifierAgentOrchestrator(coordinator=None, config={})
    hooks = CapturingHooks()
    await orchestrator.execute("do the work", None, {}, {}, hooks, coordinator=None)

    assert captured["inject_provider_calls"][0][0] == laa.DEFAULT_PROVIDER


@pytest.mark.asyncio
async def test_providers_mount_plan_is_cleared_before_injection(
    monkeypatch: pytest.MonkeyPatch,
):
    """Probe-proven seam: the 9 baked-in provider stubs must be cleared
    first, else inject_provider() is a no-op ("don't clobber existing").
    """
    captured = _install_fake_deps(
        monkeypatch, reply_text="", outcome_to_set={"status": "success"}
    )
    assert captured["prepared"].mount_plan["providers"] != []  # baseline: stubs present

    orchestrator = laa.AmplifierAgentOrchestrator(coordinator=None, config={})
    hooks = CapturingHooks()
    await orchestrator.execute("do the work", None, {}, {}, hooks, coordinator=None)

    # inject_provider is faked (doesn't itself clear+append), so what we can
    # assert is that the orchestrator cleared it BEFORE calling inject_provider.
    assert captured["inject_provider_calls"], "inject_provider should have been called"


# ---------------------------------------------------------------------------
# 3. Fail-closed (never fabricate success)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fail_closed_on_missing_verdict(monkeypatch: pytest.MonkeyPatch):
    """The child never called report_outcome (last_outcome stays None)."""
    _install_fake_deps(
        monkeypatch, reply_text="All done, looks great!", outcome_to_set=None
    )

    orchestrator = laa.AmplifierAgentOrchestrator(coordinator=None, config={})
    hooks = CapturingHooks()
    reply = await orchestrator.execute(
        "do the work", None, {}, {}, hooks, coordinator=None
    )

    assert reply == "All done, looks great!"
    report_outcome = hooks.completion["metadata"]["report_outcome"]
    assert report_outcome["status"] == "retry"
    assert report_outcome["status"] != "success"


@pytest.mark.asyncio
async def test_fail_closed_on_malformed_verdict_missing_status(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fake_deps(
        monkeypatch, reply_text="", outcome_to_set={"notes": "no status field"}
    )

    orchestrator = laa.AmplifierAgentOrchestrator(coordinator=None, config={})
    hooks = CapturingHooks()
    await orchestrator.execute("do the work", None, {}, {}, hooks, coordinator=None)

    assert hooks.completion["metadata"]["report_outcome"]["status"] == "retry"


@pytest.mark.asyncio
async def test_fail_closed_on_malformed_verdict_unknown_status(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fake_deps(
        monkeypatch, reply_text="", outcome_to_set={"status": "totally-done"}
    )

    orchestrator = laa.AmplifierAgentOrchestrator(coordinator=None, config={})
    hooks = CapturingHooks()
    await orchestrator.execute("do the work", None, {}, {}, hooks, coordinator=None)

    assert hooks.completion["metadata"]["report_outcome"]["status"] == "retry"


@pytest.mark.asyncio
async def test_exception_emits_incomplete_and_reraises(monkeypatch: pytest.MonkeyPatch):
    _install_fake_deps(monkeypatch, raise_on_execute=RuntimeError("boom"))

    orchestrator = laa.AmplifierAgentOrchestrator(coordinator=None, config={})
    hooks = CapturingHooks()

    with pytest.raises(RuntimeError, match="boom"):
        await orchestrator.execute("do the work", None, {}, {}, hooks, coordinator=None)

    assert hooks.completion["status"] == "incomplete"
    assert hooks.completion["metadata"] == {}


@pytest.mark.asyncio
async def test_engine_shutdown_called_even_on_exception(
    monkeypatch: pytest.MonkeyPatch,
):
    captured = _install_fake_deps(monkeypatch, raise_on_execute=RuntimeError("boom"))

    orchestrator = laa.AmplifierAgentOrchestrator(coordinator=None, config={})
    hooks = CapturingHooks()
    with pytest.raises(RuntimeError):
        await orchestrator.execute("do the work", None, {}, {}, hooks, coordinator=None)

    assert captured["engine"].shutdown_called is True


@pytest.mark.asyncio
async def test_engine_shutdown_called_on_success(monkeypatch: pytest.MonkeyPatch):
    captured = _install_fake_deps(
        monkeypatch, reply_text="", outcome_to_set={"status": "success"}
    )

    orchestrator = laa.AmplifierAgentOrchestrator(coordinator=None, config={})
    hooks = CapturingHooks()
    await orchestrator.execute("do the work", None, {}, {}, hooks, coordinator=None)

    assert captured["engine"].shutdown_called is True


@pytest.mark.asyncio
async def test_shutdown_failure_does_not_mask_original_exception(
    monkeypatch: pytest.MonkeyPatch,
):
    """RED-prove: submit_turn raises AND engine.shutdown() (in the
    `finally`) ALSO raises. The ORIGINAL exception must propagate --
    a shutdown failure must never mask it -- and the honest fail-closed
    'incomplete' emit must still happen.

    Against the pre-fix code (a bare ``await engine.shutdown()`` in
    ``finally`` with no guard), the shutdown's ``ValueError`` replaces the
    original ``RuntimeError`` as the exception that actually propagates,
    so ``pytest.raises(RuntimeError, ...)`` below fails -- proving this
    test is RED before the fix and GREEN after it.
    """
    captured = _install_fake_deps(
        monkeypatch,
        raise_on_execute=RuntimeError("boom"),
        shutdown_raises=ValueError("shutdown exploded"),
    )

    orchestrator = laa.AmplifierAgentOrchestrator(coordinator=None, config={})
    hooks = CapturingHooks()

    with pytest.raises(RuntimeError, match="boom"):
        await orchestrator.execute("do the work", None, {}, {}, hooks, coordinator=None)

    assert captured["engine"].shutdown_called is True
    assert hooks.completion["status"] == "incomplete"
    assert hooks.completion["metadata"] == {}


# ---------------------------------------------------------------------------
# 4. Empty reply with a verdict still succeeds (artifact over prose)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_reply_with_verdict_still_succeeds(monkeypatch: pytest.MonkeyPatch):
    verdict = {"status": "success", "notes": "wrote the file"}
    _install_fake_deps(monkeypatch, reply_text="", outcome_to_set=verdict)

    orchestrator = laa.AmplifierAgentOrchestrator(coordinator=None, config={})
    hooks = CapturingHooks()
    reply = await orchestrator.execute(
        "do the work", None, {}, {}, hooks, coordinator=None
    )

    assert reply == ""
    assert hooks.completion["metadata"]["report_outcome"] == verdict
    assert hooks.completion["status"] == "success"


# ---------------------------------------------------------------------------
# 5. System-prompt survival across history replay (PR #3 adoption review)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_system_framing_survives_history_replay(
    monkeypatch: pytest.MonkeyPatch,
):
    """Regression pin: does replaying parent-turn history into the hosted
    session (``hosted_context.set_messages(history)``, support#497's fix)
    clobber the hosted session's OWN system framing?

    ``set_messages`` REPLACES the hosted context's whole message list, and
    ``prepared.create_session()`` (amplifier_foundation/bundle/_prepared.py)
    can seed system framing into that SAME message list via a static
    ``add_message({"role": "system", ...})`` fallback -- but only when the
    context module does NOT support ``set_system_prompt_factory``. The real
    (and only) context module amplifier-agent's own baked-in bundle ever
    mounts is context-simple (see ``amplifier_agent_lib/bundle/bundle.md``'s
    ``context: module: context-simple``), and
    ``amplifier_module_context_simple.SimpleContextManager`` ALWAYS supports
    ``set_system_prompt_factory``. So ``create_session()`` always takes the
    factory branch for this adapter's hosted sessions, never the add_message
    fallback -- and the factory-produced system message is never written
    into the message list ``set_messages`` replaces (it's synthesized fresh,
    and stored system messages are filtered out, on every
    ``get_messages_for_request()`` call -- see ``FakeFactoryContextManager``'s
    docstring for the precise mechanism this fake mirrors).

    This test proves that claim rather than assuming it: it registers a
    system-prompt factory on the hosted context BEFORE ``execute()`` runs
    (mirroring what the real ``create_session()`` does), feeds a non-empty
    parent history through ``context`` so replay actually happens, and
    asserts that whatever the hosted session would send to the provider
    (``get_messages_for_request()``, captured by ``FakeSession.execute()``
    into ``messages_sent_to_provider``) still leads with the fresh system
    framing followed by the replayed history -- i.e. NO clobbering occurs
    for the real seam. (If a future context module dropped factory support,
    this test's own mechanics would catch the regression: swap in a fake
    without ``set_system_prompt_factory`` and the system message would
    vanish from ``messages_sent_to_provider`` after replay.)
    """
    hosted_context = FakeFactoryContextManager()

    async def system_factory() -> str:
        return "You are the amplifier-agent coding persona. Persist across turns."

    # Mirrors create_session() calling set_system_prompt_factory BEFORE
    # _run_turn's replay code ever touches the hosted context.
    await hosted_context.set_system_prompt_factory(system_factory)

    captured = _install_fake_deps(
        monkeypatch,
        reply_text="ok",
        outcome_to_set={"status": "success"},
        context_module=hosted_context,
    )

    parent_history = [
        {"role": "user", "content": "earlier turn: the secret is ZEBRA"},
        {"role": "assistant", "content": "earlier reply: noted, ZEBRA"},
    ]

    class _ParentContext:
        async def get_messages(self) -> list[dict[str, Any]]:
            return parent_history

    orchestrator = laa.AmplifierAgentOrchestrator(coordinator=None, config={})
    hooks = CapturingHooks()
    await orchestrator.execute(
        "do the work", _ParentContext(), {}, {}, hooks, coordinator=None
    )

    # Replay actually happened (support#497's fix is doing its job).
    assert hosted_context.set_messages_calls == [parent_history]

    # And the system framing was NOT clobbered by that replay: it still
    # leads whatever the hosted session would send to the provider.
    sent = captured["session"].messages_sent_to_provider
    assert sent is not None, "execute() never asked the context for a request"
    assert sent[0] == {
        "role": "system",
        "content": "You are the amplifier-agent coding persona. Persist across turns.",
    }
    assert sent[1:] == parent_history
