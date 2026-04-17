"""Custom LLM handler that routes requests through the SessionPool.

Tools are handled natively via MCP (see bridge.py, session.py): tool
definitions, calls and results all flow through the Claude CLI's native
tool protocol, not through prompt injection.

A request with a session UUID uses a long-lived Claude subprocess. A
request without one runs in a degraded stateless mode: a short-lived
session per request, with no tool support.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any

from litellm import CustomLLM
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    Function,
    GenericStreamingChunk,
    Message,
    ModelResponse,
    Usage,
)

from claude_proxy.log import logger
from claude_proxy.middleware import session_var, tools_var
from claude_proxy.models import parse_model_string
from claude_proxy.pool import get_pool
from claude_proxy.session import Session, ToolCall, TurnEvent

# ---------------------------------------------------------------------------
# Message extraction helpers
# ---------------------------------------------------------------------------


def _content_to_text(content: str | list[dict[str, Any]] | None) -> str:
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if b.get("type") == "text")
    return str(content) if content else ""


def _extract_system_prompt(messages: list[dict[str, Any]]) -> str:
    for msg in messages:
        if msg.get("role") == "system":
            text = _content_to_text(msg.get("content", ""))
            if text:
                return text
    return "You are a helpful assistant."


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return _content_to_text(msg.get("content", ""))
    err = "No user message found in the request"
    raise ValueError(err)


def _format_history_as_prompt(messages: list[dict[str, Any]]) -> str:
    """Serialize the full conversation as a labeled prompt (stateless mode).

    Skips system messages (carried via --system-prompt). Used when no session
    header is present: the ephemeral Session has no prior memory, so the full
    history must be re-sent with each request.
    """
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            continue
        text = _content_to_text(msg.get("content", ""))
        if role == "user":
            parts.append(f"[user]\n{text}")
        elif role == "assistant" and text:
            parts.append(f"[assistant]\n{text}")
    return "\n\n".join(parts) or _last_user_text(messages)


def _trailing_tool_results(messages: list[dict[str, Any]]) -> dict[str, str] | None:
    """If the tail of messages is one-or-more `tool` messages, return them.

    Returns a dict mapping tool_call_id → content text. Returns None if the
    tail does not end in a tool-result sequence.
    """
    results: dict[str, str] = {}
    for msg in reversed(messages):
        role = msg.get("role")
        if role == "tool":
            call_id = msg.get("tool_call_id", "")
            content = _content_to_text(msg.get("content", ""))
            if call_id:
                results[call_id] = content
        else:
            break
    return results or None


# ---------------------------------------------------------------------------
# Response builders (TurnEvent → LiteLLM types)
# ---------------------------------------------------------------------------


def _usage_block(usage: dict[str, int] | None) -> dict[str, int] | None:
    if not usage:
        return None
    inp = usage["input_tokens"]
    out = usage["output_tokens"]
    return {"prompt_tokens": inp, "completion_tokens": out, "total_tokens": inp + out}


def _completion_from_events(
    events: list[TurnEvent],
    model: str,
) -> ModelResponse:
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}

    for evt in events:
        if evt.kind == "text":
            text_parts.append(evt.delta)
        elif evt.kind == "tool_calls":
            tool_calls = evt.calls
        elif evt.kind == "end":
            usage = evt.usage or usage
        elif evt.kind == "error":
            err = f"CLI error: {evt.message}"
            raise RuntimeError(err)

    if tool_calls:
        tc_objects = [
            ChatCompletionMessageToolCall(
                id=call.call_id, type="function",
                function=Function(name=call.name, arguments=json.dumps(call.arguments)),
            )
            for call in tool_calls
        ]
        message = Message(role="assistant", content=None, tool_calls=tc_objects)
        finish_reason = "tool_calls"
    else:
        message = Message(role="assistant", content="".join(text_parts))
        finish_reason = "stop"

    return ModelResponse(
        model=model,
        choices=[Choices(message=message, finish_reason=finish_reason, index=0)],
        usage=Usage(
            prompt_tokens=usage["input_tokens"],
            completion_tokens=usage["output_tokens"],
            total_tokens=usage["input_tokens"] + usage["output_tokens"],
        ),
    )


def _chunk(**overrides: Any) -> GenericStreamingChunk:
    base: dict[str, Any] = {
        "text": "", "is_finished": False, "finish_reason": None, "index": 0,
        "tool_use": None, "usage": None, "provider_specific_fields": None,
    }
    base.update(overrides)
    return base  # type: ignore[return-value]


def _text_chunk(text: str, thinking: str | None = None) -> GenericStreamingChunk:
    psf = {"reasoning_content": thinking} if thinking else None
    return _chunk(text=text, provider_specific_fields=psf)


def _tool_calls_chunk(calls: list[ToolCall]) -> GenericStreamingChunk:
    # LiteLLM's chunk carries only one tool_use; it assembles the rest.
    tc = calls[0]
    return _chunk(
        is_finished=True,
        finish_reason="tool_calls",
        tool_use={
            "id": tc.call_id, "type": "function",
            "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
            "index": 0,
        },
    )


def _end_chunk(usage: dict[str, int] | None) -> GenericStreamingChunk:
    return _chunk(is_finished=True, finish_reason="stop", usage=_usage_block(usage))


# ---------------------------------------------------------------------------
# Session resolution
# ---------------------------------------------------------------------------


def _get_tools(kwargs: dict[str, Any]) -> list[dict[str, Any]] | None:
    # LiteLLM strips tools from custom handlers; fall back to the middleware cv.
    optional = kwargs.get("optional_params") or {}
    return optional.get("tools") or tools_var.get()


def _log_request(
    kind: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    session: Session,
) -> None:
    sysprompt = _extract_system_prompt(messages)
    sys_hash = hashlib.sha256(sysprompt.encode()).hexdigest()[:8]
    logger.debug(
        "{kind} | model={model} msgs={n} user_sid={u} internal={i} sys={h} tools={t}",
        kind=kind, model=model, n=len(messages),
        u=session_var.get(), i=session.sid, h=sys_hash, t=len(tools or []),
    )


async def _resolve_session(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    model_str: str,
) -> tuple[Session, str]:
    """Return (session, ephemeral_sid) where ephemeral_sid is set if stateless."""
    model_name, effort = parse_model_string(model_str)
    system_prompt = _extract_system_prompt(messages)
    sid = session_var.get()
    ephemeral_sid = ""
    if sid is None:
        if tools:
            err = "Tool use requires a session header (x-session-id or equivalent)"
            raise ValueError(err)
        ephemeral_sid = str(uuid.uuid4())
        sid = ephemeral_sid

    session = await get_pool().get_or_create(
        sid,
        model=model_name, effort=effort,
        system_prompt=system_prompt, tools=tools,
    )
    return session, ephemeral_sid


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


class ClaudeProxyHandler(CustomLLM):
    """LiteLLM custom backend: one long-lived CLI subprocess per session."""

    def completion(self, model: str, messages: list, **kwargs: Any) -> ModelResponse:  # type: ignore[override]
        return asyncio.run(self.acompletion(model, messages, **kwargs))

    async def acompletion(  # type: ignore[override]
        self, model: str, messages: list[dict[str, Any]], **kwargs: Any,
    ) -> ModelResponse:
        tools = _get_tools(kwargs)
        session, ephemeral = await _resolve_session(messages, tools, model)
        _log_request("Request", model, messages, tools, session)

        try:
            events = [evt async for evt in self._stream_events(session, messages)]
            return _completion_from_events(events, model)
        finally:
            if ephemeral:
                await get_pool().drop(session.sid)

    def streaming(  # type: ignore[override]
        self, model: str, messages: list, **kwargs: Any,
    ) -> Iterator[GenericStreamingChunk]:
        gen = self.astreaming(model, messages, **kwargs)
        loop = asyncio.new_event_loop()
        try:
            while True:
                try:
                    chunk = loop.run_until_complete(gen.__anext__())
                except StopAsyncIteration:
                    return
                yield chunk
        finally:
            loop.close()

    async def astreaming(  # type: ignore[override]
        self, model: str, messages: list[dict[str, Any]], **kwargs: Any,
    ) -> AsyncIterator[GenericStreamingChunk]:
        tools = _get_tools(kwargs)
        session, ephemeral = await _resolve_session(messages, tools, model)
        _log_request("Stream", model, messages, tools, session)

        try:
            async for chunk in self._stream_chunks(session, messages):
                yield chunk
        finally:
            if ephemeral:
                await get_pool().drop(session.sid)

    # ------------------------------------------------------------------

    async def _stream_events(
        self, session: Session, messages: list[dict[str, Any]],
    ) -> AsyncIterator[TurnEvent]:
        tool_results = _trailing_tool_results(messages)
        if tool_results:
            gen = session.run_tool_result_turn(tool_results)
        else:
            # Stateless sessions have no prior memory: resend the history.
            stateless = session_var.get() is None
            text = _format_history_as_prompt(messages) if stateless else _last_user_text(messages)
            gen = session.run_user_turn(text)
        async for evt in gen:
            yield evt

    async def _stream_chunks(
        self, session: Session, messages: list[dict[str, Any]],
    ) -> AsyncIterator[GenericStreamingChunk]:
        async for evt in self._stream_events(session, messages):
            if evt.kind == "text":
                yield _text_chunk(evt.delta)
            elif evt.kind == "thinking":
                yield _text_chunk("", thinking=evt.delta)
            elif evt.kind == "tool_calls":
                yield _tool_calls_chunk(evt.calls)
                return
            elif evt.kind == "end":
                yield _end_chunk(evt.usage)
                return
            elif evt.kind == "error":
                err = f"CLI error: {evt.message}"
                raise RuntimeError(err)


handler = ClaudeProxyHandler()
