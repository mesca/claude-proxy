"""ASGI middleware for session header detection and tools extraction.

Sets two context vars consumed by the handler:
  - session_var: UUID5 derived from a well-known session header
  - tools_var:   the `tools` field from the JSON request body

Also flattens LiteLLM's reasoning_content onto delta for streaming clients
(Vercel AI SDK / OpenCode).
"""

from __future__ import annotations

import contextvars
import json
import os
import uuid
from collections.abc import Callable
from typing import Any

from claude_proxy.log import logger

session_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "session_id", default=None,
)
tools_var: contextvars.ContextVar[list[dict[str, Any]] | None] = contextvars.ContextVar(
    "tools", default=None,
)

_SESSION_NAMESPACE = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")
_WELL_KNOWN_HEADERS = [
    b"x-session-affinity",  # OpenCode
    b"x-session-id",
    b"x-conversation-id",
]
_warned_no_header = False


class ReasoningContentMiddleware:
    """Move reasoning_content from provider_specific_fields to delta top level."""

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


class RequestContextMiddleware:
    """Extract tools+session from each /chat/completions request."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if not scope.get("path", "").endswith("/chat/completions"):
            await self.app(scope, receive, send)
            return

        receive = await _extract_tools_from_request(receive)
        _detect_session(scope)
        await self.app(scope, receive, send)


# ---------------------------------------------------------------------------
# Request-side: extract tools from body, detect session from headers
# ---------------------------------------------------------------------------


async def _extract_tools_from_request(receive: Callable) -> Callable:
    """Read and buffer the body, set tools_var, return a replay receive."""
    import asyncio

    body_chunks: list[bytes] = []
    messages: list[dict] = []

    while True:
        message = await receive()
        messages.append(message)
        body_chunks.append(message.get("body", b""))
        if not message.get("more_body", False):
            break

    try:
        data = json.loads(b"".join(body_chunks))
        tools = data.get("tools")
        if tools and isinstance(tools, list):
            tools_var.set(tools)
        else:
            tools_var.set(None)
    except (json.JSONDecodeError, UnicodeDecodeError):
        tools_var.set(None)

    replay_iter = iter(messages)
    done = asyncio.Event()

    async def replay_receive() -> dict:
        msg = next(replay_iter, None)
        if msg is not None:
            return msg
        await done.wait()
        return {"type": "http.disconnect"}

    return replay_receive


def _detect_session(scope: dict) -> None:
    global _warned_no_header  # noqa: PLW0603
    if os.environ.get("CLAUDE_PROXY_STATELESS") == "1":
        session_var.set(None)
        return

    headers = dict(scope.get("headers") or [])
    custom = os.environ.get("CLAUDE_PROXY_SESSION_HEADER")
    if custom:
        value = headers.get(custom.lower().encode(), b"").decode()
        if value:
            session_var.set(str(uuid.uuid5(_SESSION_NAMESPACE, value)))
            return

    for header in _WELL_KNOWN_HEADERS:
        value = headers.get(header, b"").decode()
        if value:
            session_var.set(str(uuid.uuid5(_SESSION_NAMESPACE, value)))
            return

    session_var.set(None)
    if not _warned_no_header:
        _warned_no_header = True
        logger.warning(
            "No session header found. Running stateless (no tool support). "
            "Set --session-header or use --stateless to suppress this warning.",
        )


# ---------------------------------------------------------------------------
# Response-side: flatten reasoning_content for compatible clients
# ---------------------------------------------------------------------------


def _transform_sse_body(body: bytes) -> bytes:
    if b"provider_specific_fields" not in body:
        return body  # no reasoning_content to flatten — skip decode/parse
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
    for choice in chunk.get("choices", []):
        delta = choice.get("delta")
        if not delta:
            continue
        psf = delta.pop("provider_specific_fields", None)
        if psf and "reasoning_content" in psf:
            delta["reasoning_content"] = psf["reasoning_content"]
