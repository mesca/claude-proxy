"""ASGI middleware for SSE streaming and tool_calls transformation."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from typing import Any


class ReasoningContentMiddleware:
    """Moves reasoning_content from provider_specific_fields to delta top level.

    LiteLLM puts it at delta.provider_specific_fields.reasoning_content,
    but clients (Vercel AI SDK / OpenCode) expect delta.reasoning_content.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def transform_send(message: dict) -> None:
            if message["type"] == "http.response.body":
                body = message.get("body", b"")
                if body:
                    message = {**message, "body": _transform_sse_body(body)}
            await send(message)

        await self.app(scope, receive, transform_send)


class ToolCallsMiddleware:
    """Detects raw tool_calls JSON in streamed content and rewrites to proper format.

    Buffers ALL response chunks, accumulates delta.content across SSE events.
    If the accumulated content is a tool_calls JSON, rewrites the entire response
    to emit a single tool_calls SSE event instead of the raw text.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if not path.endswith("/chat/completions"):
            await self.app(scope, receive, send)
            return

        # Buffer all response messages
        buffered: list[dict] = []
        headers_message: dict | None = None

        async def buffering_send(message: dict) -> None:
            nonlocal headers_message
            if message["type"] == "http.response.start":
                headers_message = message
            elif message["type"] == "http.response.body":
                buffered.append(message)

        await self.app(scope, receive, buffering_send)

        # Accumulate all content from the buffered body chunks
        all_body = b"".join(m.get("body", b"") for m in buffered)
        text = all_body.decode("utf-8", errors="replace")

        transformed = _try_rewrite_tool_calls(text)

        if transformed is not None:
            # Tool calls detected — send rewritten response
            new_body = transformed.encode("utf-8")
            if headers_message:
                await send(headers_message)
            await send({
                "type": "http.response.body",
                "body": new_body,
                "more_body": False,
            })
        else:
            # No tool calls — send original response unchanged
            if headers_message:
                await send(headers_message)
            for i, msg in enumerate(buffered):
                msg = {**msg, "more_body": i < len(buffered) - 1}
                await send(msg)


# ---------------------------------------------------------------------------
# Reasoning content flattening
# ---------------------------------------------------------------------------


def _transform_sse_body(body: bytes) -> bytes:
    """Transform SSE data lines to flatten reasoning_content into delta."""
    text = body.decode("utf-8", errors="replace")
    lines = text.split("\n")
    out: list[str] = []

    for line in lines:
        if line.startswith("data: ") and line != "data: [DONE]":
            try:
                chunk = json.loads(line[6:])
                _flatten_reasoning(chunk)
                out.append("data: " + json.dumps(chunk, separators=(",", ":")))
            except (json.JSONDecodeError, KeyError):
                out.append(line)
        else:
            out.append(line)

    return "\n".join(out).encode("utf-8")


def _flatten_reasoning(chunk: dict) -> None:
    """Move reasoning_content from provider_specific_fields to delta top level."""
    for choice in chunk.get("choices", []):
        delta = choice.get("delta")
        if not delta:
            continue
        psf = delta.pop("provider_specific_fields", None)
        if psf and "reasoning_content" in psf:
            delta["reasoning_content"] = psf["reasoning_content"]


# ---------------------------------------------------------------------------
# Tool calls detection and rewriting
# ---------------------------------------------------------------------------


def _parse_tool_calls_json(text: str) -> list[dict[str, Any]] | None:
    """Try to find and parse tool_calls JSON anywhere in text.

    Uses raw_decode to handle cases where Claude adds text before or after
    the JSON (e.g. "Let me check.\n{...}\n<tool_result>...").
    """
    idx = text.find('{"tool_calls"')
    if idx == -1:
        return None
    try:
        data, _ = json.JSONDecoder().raw_decode(text, idx)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    raw = data.get("tool_calls")
    if not raw:
        return None
    calls: list[dict[str, Any]] = [raw] if isinstance(raw, dict) else raw
    if not isinstance(calls, list) or not calls:
        return None
    result: list[dict[str, Any]] = []
    for i, tc in enumerate(calls):
        raw_id = tc.get("id") or uuid.uuid4().hex[:8]
        call_id = raw_id if str(raw_id).startswith("call_") else f"call_{raw_id}"
        result.append({
            "id": str(call_id),
            "type": "function",
            "index": i,
            "function": {
                "name": tc.get("name", ""),
                "arguments": json.dumps(tc.get("arguments", {})),
            },
        })
    return result


def _try_rewrite_tool_calls(text: str) -> str | None:
    """Check if response contains tool_calls content and rewrite if so.

    Handles both SSE streaming and non-streaming JSON responses.
    Returns rewritten text, or None if no tool_calls found.
    """
    if "data: " in text:
        return _try_rewrite_sse(text)
    return _try_rewrite_json(text)


def _try_rewrite_json(text: str) -> str | None:
    """Rewrite a non-streaming JSON response."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    changed = False
    for choice in data.get("choices", []):
        message = choice.get("message", {})
        content = message.get("content")
        if not content or not isinstance(content, str):
            continue
        tool_calls = _parse_tool_calls_json(content)
        if tool_calls:
            message["tool_calls"] = tool_calls
            message["content"] = None
            choice["finish_reason"] = "tool_calls"
            changed = True

    return json.dumps(data, separators=(",", ":")) if changed else None


def _try_rewrite_sse(text: str) -> str | None:
    """Rewrite SSE streaming response if content is a tool_calls JSON.

    Accumulates delta.content across all SSE data events,
    then checks the full content for tool_calls.
    """
    lines = text.split("\n")
    accumulated_content = ""
    last_chunk: dict | None = None
    last_chunk_line_idx = -1

    for i, line in enumerate(lines):
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        try:
            chunk = json.loads(line[6:])
        except json.JSONDecodeError:
            continue
        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})
            content = delta.get("content")
            if content and isinstance(content, str):
                accumulated_content += content
        last_chunk = chunk
        last_chunk_line_idx = i

    tool_calls = _parse_tool_calls_json(accumulated_content)
    if not tool_calls or last_chunk is None:
        return None

    # Build a single SSE event with the tool_calls
    # Use the last chunk as template (has model, id, etc.)
    for choice in last_chunk.get("choices", []):
        choice["delta"] = {"role": "assistant", "content": None, "tool_calls": tool_calls}
        choice["finish_reason"] = "tool_calls"

    out: list[str] = []
    for i, line in enumerate(lines):
        if not line.startswith("data: ") or line == "data: [DONE]":
            out.append(line)
        elif i == last_chunk_line_idx:
            # Replace with the tool_calls chunk
            out.append("data: " + json.dumps(last_chunk, separators=(",", ":")))
        # else: skip intermediate content chunks (they had partial JSON)

    return "\n".join(out)
