"""AmplifierBackend — CodergenBackend adapter using session spawning.

This is the "sessions all the way down" integration point. When the
pipeline engine hits a codergen node, the CodergenHandler calls this
backend, which spawns a coding agent sub-session via the Amplifier
``session.spawn`` capability.

When session.spawn is not available, falls back to a direct provider
mini tool loop that calls LLM → execute tool calls → repeat until the
model returns a text-only response.

Spec coverage: Section 4.5 (CodergenBackend Interface), Section 1.4,
               FID-001–010, Section 5.4.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .context import PipelineContext
from .fidelity import build_preamble, resolve_fidelity, resolve_thread_key
from .graph import Edge, Graph, Node
from .outcome import Outcome, StageStatus

logger = logging.getLogger(__name__)

# Map StageStatus value strings to enum members for parsing
_STATUS_MAP: dict[str, StageStatus] = {s.value: s for s in StageStatus}

# Maximum rounds for the direct tool loop fallback
_MAX_TOOL_LOOP_ROUNDS = 20


class AmplifierBackend:
    """CodergenBackend implementation using Amplifier session spawning.

    Resolves the provider profile from node attributes, spawns a child
    coding agent session, and parses the outcome from the response.

    Supports two execution paths:
    - **Path A (spawn)**: If ``session.spawn`` is available, delegates to
      a full child session with the complete tool loop.
    - **Path B (direct tool loop)**: If spawn is unavailable but a provider
      and tools are available, runs a mini agentic loop directly
      (LLM call → tool execution → repeat).

    Supports fidelity-based context control:
    - ``full``: Reuses sessions via a thread-keyed session pool.
    - ``compact``/``truncate``/``summary:*``: Fresh session with preamble.
    """

    def __init__(
        self,
        coordinator: Any,
        profiles: dict[str, str],
        provider: Any | None = None,
        tools: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the backend.

        Args:
            coordinator: Amplifier coordinator with session.spawn capability.
            profiles: Map of provider name to profile/bundle name.
                      e.g. {"anthropic": "attractor-anthropic", ...}
            provider: Optional LLM provider for direct tool loop fallback.
            tools: Optional tool dict for direct tool loop fallback.
        """
        self._coordinator = coordinator
        self._profiles = profiles
        self._provider = provider
        self._tools = tools or {}
        self._spawn_fn: Any | None = None
        self._spawn_checked = False
        self._session_pool: dict[str, str] = {}
        self._completed_nodes: dict[str, Outcome] = {}
        self._last_node_id: str | None = None

    async def run(
        self,
        node: Node,
        prompt: str,
        context: PipelineContext,
        incoming_edge: Edge | None = None,
        graph: Graph | None = None,
    ) -> Outcome:
        """Execute a coding task by spawning a child session.

        Falls back to a direct provider tool loop when session.spawn is
        not available.

        Args:
            node: The pipeline node being executed.
            prompt: The expanded prompt string.
            context: The current pipeline context.
            incoming_edge: The edge leading to this node (for fidelity resolution).
            graph: The pipeline graph (for fidelity resolution).

        Returns:
            Outcome parsed from the child session's response.
        """
        # 1. Get spawn capability (lazy resolution, checked once)
        if not self._spawn_checked:
            cap = self._coordinator.get_capability("session.spawn")
            if cap is not None:
                self._spawn_fn = cap
            self._spawn_checked = True

        # 2. Resolve provider and profile from node attributes
        provider = node.attrs.get("llm_provider", "anthropic")
        model = node.attrs.get("llm_model")
        reasoning_effort = node.attrs.get("reasoning_effort")
        profile_name = self._profiles.get(
            provider, next(iter(self._profiles.values()), "")
        )

        # 3. Resolve fidelity mode (spec FID-001–010)
        if graph is not None:
            fidelity = resolve_fidelity(node, incoming_edge, graph)
        else:
            # Fallback when graph not provided (backward compat)
            fidelity = node.attrs.get("fidelity", "compact")

        # 4. Build the instruction with preamble for non-full modes
        if fidelity == "full":
            instruction = prompt
        else:
            preamble = build_preamble(fidelity, context, self._completed_nodes)
            instruction = f"{preamble}\n\n---\n\n{prompt}" if preamble else prompt

        # 5. Route to Path A (spawn) or Path B (direct tool loop)
        if self._spawn_fn is not None:
            outcome = await self._run_with_spawn(
                node,
                instruction,
                provider,
                model,
                reasoning_effort,
                profile_name,
                fidelity,
                incoming_edge,
                graph,
            )
            # Fall back to direct tool loop if spawn failed and provider available
            if outcome.status == StageStatus.FAIL and self._provider is not None:
                logger.info(
                    "Spawn failed for node %s, falling back to direct tool loop",
                    node.id,
                )
                outcome = await self._run_with_tool_loop(
                    node,
                    instruction,
                    reasoning_effort,
                )
        elif self._provider is not None:
            outcome = await self._run_with_tool_loop(
                node,
                instruction,
                reasoning_effort,
            )
        else:
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason=(
                    "Neither session.spawn nor a direct provider is "
                    "available — cannot execute node"
                ),
            )

        # Record completed node outcome for future preambles
        self._completed_nodes[node.id] = outcome
        self._last_node_id = node.id

        return outcome

    # ------------------------------------------------------------------
    # Path A: Full child session via session.spawn
    # ------------------------------------------------------------------

    async def _run_with_spawn(
        self,
        node: Node,
        instruction: str,
        provider: str,
        model: str | None,
        reasoning_effort: str | None,
        profile_name: str,
        fidelity: str,
        incoming_edge: Edge | None,
        graph: Graph | None,
    ) -> Outcome:
        """Spawn a full child session via the CLI's session.spawn capability."""
        assert self._spawn_fn is not None  # guaranteed by caller

        # Obtain parent_session from coordinator
        parent_session = getattr(self._coordinator, "session", None)

        # Obtain agent_configs from coordinator config
        coordinator_config = getattr(self._coordinator, "config", None) or {}
        agent_configs: dict[str, Any] = coordinator_config.get("agents", {})

        # Build spawn kwargs matching the CLI spawn_capability signature
        spawn_kwargs: dict[str, Any] = {
            "agent_name": profile_name,
            "instruction": instruction,
            "parent_session": parent_session,
            "agent_configs": agent_configs,
            "orchestrator_config": {
                "reasoning_effort": reasoning_effort,
            },
        }
        if model:
            spawn_kwargs["provider_preferences"] = [
                {"provider": provider, "model": model}
            ]

        # Session pool for full fidelity (spec FID-001: thread reuse)
        if fidelity == "full" and graph is not None:
            thread_key = resolve_thread_key(
                node, incoming_edge, graph, self._last_node_id
            )
            existing_session = self._session_pool.get(thread_key)
            if existing_session is not None:
                spawn_kwargs["sub_session_id"] = existing_session

        # Spawn the child session
        try:
            result = await self._spawn_fn(**spawn_kwargs)
        except Exception as e:
            logger.warning("Spawn failed for node %s: %s", node.id, e)
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason=str(e),
            )

        # Parse outcome from result
        output = result.get("output", "") if isinstance(result, dict) else str(result)
        outcome = _parse_outcome(output)

        # Record session_id in pool for full fidelity reuse
        if fidelity == "full" and graph is not None:
            session_id = result.get("session_id") if isinstance(result, dict) else None
            if session_id:
                thread_key = resolve_thread_key(
                    node, incoming_edge, graph, self._last_node_id
                )
                self._session_pool[thread_key] = session_id

        return outcome

    # ------------------------------------------------------------------
    # Path B: Direct provider mini tool loop (fallback)
    # ------------------------------------------------------------------

    async def _run_with_tool_loop(
        self,
        node: Node,
        instruction: str,
        reasoning_effort: str | None,
    ) -> Outcome:
        """Execute a mini agentic loop directly (no child session).

        Calls the provider in a loop: LLM response → if tool calls,
        execute them and feed results back → repeat until the model
        returns a text-only response or max rounds is reached.
        """
        from amplifier_core import ChatRequest, Message

        messages: list[Message] = [Message(role="user", content=instruction)]

        # Build tool specs from available tools
        tool_specs = _build_tool_specs(self._tools)

        for _round in range(_MAX_TOOL_LOOP_ROUNDS):
            request = ChatRequest(
                messages=messages,
                tools=tool_specs or None,
                tool_choice="auto" if tool_specs else None,
                reasoning_effort=reasoning_effort,
            )

            try:
                response = await self._provider.complete(request)
            except Exception as exc:
                logger.warning(
                    "Provider call failed for node %s (round %d): %s",
                    node.id,
                    _round,
                    exc,
                )
                return Outcome(
                    status=StageStatus.FAIL,
                    failure_reason=str(exc),
                )

            # Extract text and tool calls from response
            text = _extract_text(response)
            tool_calls = _extract_tool_calls(response, self._provider)

            if not tool_calls:
                # No tool calls → model is done
                return (
                    _parse_outcome(text)
                    if text
                    else Outcome(
                        status=StageStatus.SUCCESS,
                        notes=f"Stage completed: {node.id}",
                    )
                )

            # Append assistant message with tool call blocks
            messages.append(_build_assistant_message(response))

            # Execute each tool call and append results
            for tc in tool_calls:
                tool = self._tools.get(tc.name)
                if tool is not None:
                    try:
                        result = await tool.execute(tc.arguments)
                        output = (
                            result.output if hasattr(result, "output") else str(result)
                        )
                    except Exception as exc:
                        output = f"Tool error: {exc}"
                else:
                    output = f"Unknown tool: {tc.name}"

                messages.append(
                    Message(
                        role="tool",
                        tool_call_id=tc.id,
                        content=str(output) if not isinstance(output, str) else output,
                    )
                )

        # Exhausted rounds
        return Outcome(
            status=StageStatus.PARTIAL_SUCCESS,
            notes=f"Max tool loop rounds ({_MAX_TOOL_LOOP_ROUNDS}) reached",
        )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _parse_outcome(output: str) -> Outcome:
    """Parse an outcome from child session output.

    Tries JSON first (from tool-report-outcome), falls back to
    wrapping plain text as SUCCESS.
    """
    # Try to parse JSON outcome
    stripped = output.strip()
    if stripped.startswith("{"):
        try:
            data = json.loads(stripped)
            if "status" in data:
                status = _STATUS_MAP.get(data["status"])
                if status is not None:
                    return Outcome(
                        status=status,
                        failure_reason=data.get("failure_reason"),
                        notes=data.get("notes"),
                        preferred_label=data.get("preferred_label"),
                        suggested_next_ids=data.get("suggested_next_ids"),
                        context_updates=data.get("context_updates"),
                    )
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    # Fall back to plain text → SUCCESS
    return Outcome(
        status=StageStatus.SUCCESS,
        notes=output[:200] if output else "No output",
    )


def _build_tool_specs(tools: dict[str, Any]) -> list[dict[str, Any]]:
    """Build ChatRequest-compatible tool specs from mounted tools."""
    specs: list[dict[str, Any]] = []
    for tool in tools.values():
        schema = getattr(tool, "parameters", None) or getattr(tool, "schema", None)
        if schema is None:
            schema = {"type": "object", "properties": {}}
        specs.append(
            {
                "name": getattr(tool, "name", str(tool)),
                "description": getattr(tool, "description", ""),
                "parameters": schema if isinstance(schema, dict) else {},
            }
        )
    return specs


def _extract_text(response: Any) -> str:
    """Extract concatenated text from response content blocks."""
    text = ""
    content = getattr(response, "content", None)
    if content:
        for block in content:
            if hasattr(block, "text"):
                text += block.text
    return text


def _extract_tool_calls(response: Any, provider: Any) -> list[Any]:
    """Extract tool calls from a provider response.

    Prefers ``response.tool_calls``; falls back to ``provider.parse_tool_calls``.
    """
    calls = getattr(response, "tool_calls", None)
    if calls:
        return list(calls)
    if hasattr(provider, "parse_tool_calls"):
        return provider.parse_tool_calls(response)
    return []


def _build_assistant_message(response: Any) -> Any:
    """Build an assistant Message from a ChatResponse for the tool loop.

    Separates text content into ``Message.content`` (string) and tool calls
    into ``Message.tool_calls`` (list of dicts).  This avoids a type mismatch
    when the provider later serializes the message — putting ToolCallBlock
    objects directly in ``content`` caused serialization failures.
    """
    from amplifier_core import Message

    text_parts: list[str] = []
    tool_calls_list: list[dict[str, Any]] = []

    # Collect text and tool-call blocks from response.content
    content = getattr(response, "content", None)
    if content:
        for block in content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
            elif hasattr(block, "tool_call_id") or hasattr(block, "id"):
                # Tool-call block — extract into a plain dict
                tc_id = getattr(block, "tool_call_id", None) or getattr(block, "id", "")
                tc_name = getattr(block, "tool_name", None) or getattr(
                    block, "name", ""
                )
                tc_args = getattr(block, "arguments", None) or getattr(
                    block, "input", {}
                )
                tool_calls_list.append(
                    {
                        "id": tc_id,
                        "name": tc_name,
                        "arguments": tc_args if isinstance(tc_args, dict) else {},
                    }
                )

    # Also pick up tool calls from response.tool_calls (some providers
    # surface them there instead of inline in content)
    resp_tool_calls = getattr(response, "tool_calls", None)
    if resp_tool_calls:
        existing_ids = {tc["id"] for tc in tool_calls_list}
        for tc in resp_tool_calls:
            tc_id = getattr(tc, "id", "")
            if tc_id not in existing_ids:
                tool_calls_list.append(
                    {
                        "id": tc_id,
                        "name": getattr(tc, "name", ""),
                        "arguments": getattr(tc, "arguments", {}),
                    }
                )

    text = "\n".join(text_parts) if text_parts else ""
    msg = Message(role="assistant", content=text)

    if tool_calls_list:
        # Message uses extra="allow", so this dynamic attribute is accepted.
        # The provider's _convert_messages() picks up msg.tool_calls.
        msg.tool_calls = tool_calls_list  # type: ignore[attr-defined]

    return msg
