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

try:
    from amplifier_foundation import ProviderPreference as _ProviderPreference
except ImportError:
    raise ImportError(
        "amplifier_foundation is required for ProviderPreference. "
        "Install it with: pip install amplifier-foundation"
    ) from None

from .context import PipelineContext
from .fidelity import build_preamble, resolve_fidelity, resolve_thread_key
from .graph import Edge, Graph, Node
from .outcome import Outcome, StageStatus
from .hook_bridge import _current_node_context, set_node_context
from .pipeline_events import PROVIDER_ERROR, PROVIDER_REQUEST, PROVIDER_RESPONSE

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
        unified_client: Any | None = None,
        hooks: Any | None = None,
    ) -> None:
        """Initialize the backend.

        Args:
            coordinator: Amplifier coordinator with session.spawn capability.
            profiles: Map of provider name to profile/bundle name.
                      e.g. {"anthropic": "attractor-anthropic", ...}
            provider: Optional LLM provider for direct tool loop fallback.
                      Used as a truthiness flag to enable Path B.
            tools: Optional tool dict for direct tool loop fallback.
            unified_client: Optional ``unified_llm.Client`` for LLM calls.
                            Created lazily via ``Client.from_env()`` if not provided.
            hooks: Optional HookRegistry for emitting provider-level events.
        """
        self._coordinator = coordinator
        self._profiles = profiles
        self._provider = provider
        self._tools = tools or {}
        self._unified_client = unified_client
        self._hooks = hooks
        self._spawn_fn: Any | None = None
        self._spawn_checked = False
        self._session_pool: dict[str, str] = {}
        self._completed_nodes: dict[str, Outcome] = {}
        self._last_node_id: str | None = None

    def clone(self) -> AmplifierBackend:
        """Create a clone with shared immutable refs but fresh mutable state.

        Used for parallel branch isolation so concurrent branches don't
        corrupt each other's session pools or completion tracking.
        """
        new = AmplifierBackend.__new__(AmplifierBackend)
        # Shared immutable refs
        new._coordinator = self._coordinator
        new._profiles = self._profiles
        new._provider = self._provider
        new._unified_client = self._unified_client
        new._hooks = self._hooks
        # Shallow-copy tools (tool objects shared, dict independent)
        new._tools = dict(self._tools)
        # Fresh mutable state
        new._spawn_fn = None
        new._spawn_checked = False
        new._session_pool = {}
        new._completed_nodes = {}
        new._last_node_id = None
        return new

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
        max_agent_turns_raw = node.attrs.get("max_agent_turns")
        max_agent_turns = (
            int(max_agent_turns_raw) if max_agent_turns_raw is not None else None
        )
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
                max_agent_turns,
                profile_name,
                fidelity,
                incoming_edge,
                graph,
                context,
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
                    max_agent_turns,
                )
        elif self._provider is not None:
            outcome = await self._run_with_tool_loop(
                node,
                instruction,
                reasoning_effort,
                max_agent_turns,
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
        max_agent_turns: int | None,
        profile_name: str,
        fidelity: str,
        incoming_edge: Edge | None,
        graph: Graph | None,
        context: PipelineContext | None = None,
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
                "max_turns": max_agent_turns,
            },
        }
        if model:
            spawn_kwargs["provider_preferences"] = [
                _ProviderPreference(provider=provider, model=model)
            ]

        # Inject shared execution environment attachment for child session
        if context is not None:
            container_id = context.get("internal.env_container_id")
            env_type = context.get("internal.env_type")
            if container_id:
                spawn_kwargs["tools"] = spawn_kwargs.get("tools", []) + [
                    {
                        "module": "tools-env-all",
                        "config": {
                            "auto_attach": {
                                "type": env_type,
                                "name": "pipeline-workspace",
                                "attach_to": container_id,
                            }
                        },
                    }
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
        max_agent_turns: int | None = None,
    ) -> Outcome:
        """Execute via unified_llm.generate() (no child session).

        Delegates the full agentic tool loop to the unified-llm-client
        library, which handles LLM calls, tool execution, retry, and
        error mapping internally.
        """
        import unified_llm

        client = self._get_or_create_unified_client()
        model = _resolve_model(node)
        provider_name = node.llm_provider or node.attrs.get("llm_provider", "anthropic")
        tools = _build_unified_tools(self._tools)

        # Set node context for the hook bridge middleware
        token = set_node_context({"node_id": node.id})

        try:
            # Emit provider:request before the LLM call
            pre_result = await self._emit(
                PROVIDER_REQUEST,
                {
                    "provider": provider_name,
                    "model": model,
                    "node_id": node.id,
                    "tool_names": [t.name for t in tools] if tools else [],
                    "message_count": 1,  # prompt-only = 1 message
                },
            )

            # Check for deny action from hooks (e.g., approval gates)
            if (
                pre_result is not None
                and getattr(pre_result, "action", "continue") == "deny"
            ):
                reason = getattr(pre_result, "reason", None) or "Denied by hook"
                return Outcome(
                    status=StageStatus.FAIL,
                    failure_reason=f"Denied by hook: {reason}",
                )

            result = await unified_llm.generate(
                model=model,
                prompt=instruction,
                tools=tools or None,
                max_tool_rounds=max_agent_turns
                if max_agent_turns is not None
                else _MAX_TOOL_LOOP_ROUNDS,
                reasoning_effort=reasoning_effort,
                provider=provider_name,
                client=client,
            )
        except unified_llm.SDKError as exc:
            logger.warning("unified_llm.generate failed for node %s: %s", node.id, exc)
            await self._emit(
                PROVIDER_ERROR,
                {
                    "provider": provider_name,
                    "model": model,
                    "node_id": node.id,
                    "error_type": type(exc).__name__,
                    "error_class": type(exc).__mro__[1].__name__,
                    "retryable": getattr(exc, "retryable", False),
                    "message": str(exc),
                },
            )
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason=str(exc),
            )
        except Exception as exc:
            logger.warning("Unexpected error in generate for node %s: %s", node.id, exc)
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason=str(exc),
            )
        finally:
            _current_node_context.reset(token)

        # Emit provider:response after successful LLM call
        await self._emit(
            PROVIDER_RESPONSE,
            {
                "provider": provider_name,
                "model": model,
                "node_id": node.id,
                "usage": {
                    "input_tokens": result.total_usage.input_tokens,
                    "output_tokens": result.total_usage.output_tokens,
                    "total_tokens": result.total_usage.total_tokens,
                    "reasoning_tokens": result.total_usage.reasoning_tokens,
                    "cache_read_tokens": result.total_usage.cache_read_tokens,
                    "cache_write_tokens": result.total_usage.cache_write_tokens,
                },
                "finish_reason": result.finish_reason.reason,
                "text_length": len(result.text) if result.text else 0,
                "step_count": len(result.steps),
            },
        )

        # Map GenerateResult → Outcome
        if result.text:
            return _parse_outcome(result.text)
        return Outcome(
            status=StageStatus.SUCCESS,
            notes=f"Stage completed: {node.id}",
        )

    def _get_or_create_unified_client(self) -> Any:
        """Return the injected client or lazily create one from environment."""
        if self._unified_client is not None:
            return self._unified_client
        import unified_llm

        self._unified_client = unified_llm.Client.from_env()
        return self._unified_client

    async def _emit(self, event_name: str, data: dict[str, Any]) -> Any:
        """Emit an event via hooks, if provided.

        Returns the HookResult from hooks.emit(), or None if hooks is not set.
        Unlike the engine's fire-and-forget _emit, this returns the result
        so callers can inspect the action (deny, modify, etc.).
        """
        if self._hooks is not None:
            return await self._hooks.emit(event_name, data)
        return None


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

# Default models per provider (used when node.llm_model is not set)
_DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o",
    "gemini": "gemini-2.0-flash",
    "test": "test-model",
}


def _resolve_model(node: Node) -> str:
    """Resolve the LLM model identifier from a pipeline node.

    Uses the node's ``llm_model`` if set, otherwise falls back to a
    sensible default based on the provider.
    """
    if node.llm_model:
        return node.llm_model
    provider = node.llm_provider or node.attrs.get("llm_provider", "anthropic")
    return _DEFAULT_MODELS.get(provider, "claude-sonnet-4-20250514")


def _make_tool_handler(pipeline_tool: Any) -> Any:
    """Create a unified_llm-compatible execute handler from a pipeline tool.

    Pipeline tools expect ``execute(input: dict)``.
    unified_llm calls ``tool.execute(**kwargs)``.
    This wrapper bridges the two conventions.
    """

    async def handler(**kwargs: Any) -> str:
        result = await pipeline_tool.execute(kwargs)
        if hasattr(result, "output"):
            return result.output
        return str(result)

    return handler


def _build_unified_tools(pipeline_tools: dict[str, Any]) -> list[Any]:
    """Convert pipeline tools to unified_llm.Tool objects."""
    import unified_llm

    tools: list[Any] = []
    for tool in pipeline_tools.values():
        schema = getattr(tool, "parameters", None) or getattr(tool, "schema", None)
        if schema is None:
            schema = {"type": "object", "properties": {}}

        execute_fn = None
        if hasattr(tool, "execute"):
            execute_fn = _make_tool_handler(tool)

        tools.append(
            unified_llm.Tool(
                name=getattr(tool, "name", str(tool)),
                description=getattr(tool, "description", ""),
                parameters=schema if isinstance(schema, dict) else {},
                execute=execute_fn,
            )
        )
    return tools


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

    return Outcome(
        status=StageStatus.FAIL,
        notes=f"Non-structured response (expected JSON with 'status' key): {output[:200] if output else 'No output'}",
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
                tc_name = (
                    getattr(block, "tool_name", None)
                    or getattr(block, "name", None)
                    or ""
                )
                if not tc_name:
                    continue  # Skip tool calls with no name — can't serialize
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
            if tc_id in existing_ids:
                continue
            tc_name = getattr(tc, "name", None) or getattr(tc, "tool_name", None) or ""
            if not tc_name:
                continue  # Skip tool calls with no name — can't serialize
            tool_calls_list.append(
                {
                    "id": tc_id,
                    "name": tc_name,
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
