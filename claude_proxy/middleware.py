"""ASGI middleware to fix SSE streaming format for reasoning_content."""

from __future__ import annotations

import json
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
