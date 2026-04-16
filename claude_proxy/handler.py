"""Custom LLM handler that routes requests to the claude CLI.

System prompt passthrough and session management. Tool_calls detection
and rewriting is handled by ToolCallsMiddleware in middleware.py.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator
from typing import Any

from litellm import CustomLLM
from litellm.types.utils import (
    Choices,
    GenericStreamingChunk,
    Message,
    ModelResponse,
    Usage,
)

from claude_proxy import cli
from claude_proxy.cli import ClaudeCliError
from claude_proxy.log import logger
from claude_proxy.models import parse_model_string

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_cwd() -> str | None:
    return os.environ.get("CLAUDE_PROXY_CWD")


def _append_system_prompt() -> bool:
    return os.environ.get("CLAUDE_PROXY_APPEND_SYSTEM_PROMPT") == "1"


def _content_to_text(content: str | list[dict[str, Any]] | None) -> str:
    """Extract text from OpenAI content (string or content-block list)."""
    if isinstance(content, list):
        parts: list[str] = [
            b.get("text", "") for b in content if b.get("type") == "text"
        ]
        return "".join(parts)
    return str(content) if content else ""


def _extract_prompt(messages: list[dict[str, Any]]) -> str:
    """Extract the last user message content from the message array."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return _content_to_text(msg.get("content", ""))
    err = "No user message found in the request"
    raise ClaudeCliError(400, err)


def _extract_system_prompt(messages: list[dict[str, Any]]) -> str | None:
    """Extract the system message content from the message array."""
    for msg in messages:
        if msg.get("role") == "system":
            text = _content_to_text(msg.get("content", ""))
            return text or None
    return None


def _is_new_conversation(messages: list[dict[str, Any]]) -> bool:
    """Detect new conversation: no assistant messages in the history."""
    return not any(m.get("role") == "assistant" for m in messages)


def _get_session_id(
    kwargs: dict[str, Any],
    messages: list[dict[str, Any]],
    stored_session_id: str | None,
) -> str | None:
    optional_params = kwargs.get("optional_params", {})
    explicit = optional_params.get("session_id")
    if explicit:
        return str(explicit)
    if _is_new_conversation(messages):
        return None
    return stored_session_id


def _get_cwd(kwargs: dict[str, Any]) -> str | None:
    optional_params = kwargs.get("optional_params", {})
    cwd = optional_params.get("cwd")
    if cwd:
        return str(cwd)
    return _default_cwd()


def _get_model_and_effort(model: str) -> tuple[str | None, str | None]:
    return parse_model_string(model)


# ---------------------------------------------------------------------------
# Response builders
# ---------------------------------------------------------------------------


def _build_model_response(
    result: dict[str, Any],
    model: str,
) -> ModelResponse:
    """Build a LiteLLM ModelResponse from a CLI result dict."""
    raw_usage = result.get("usage", {})
    input_tokens = raw_usage.get("input_tokens", 0)
    output_tokens = raw_usage.get("output_tokens", 0)

    return ModelResponse(
        model=model,
        system_fingerprint=result.get("session_id"),
        choices=[
            Choices(
                message=Message(
                    role="assistant", content=result.get("result", ""),
                ),
                finish_reason="stop",
                index=0,
            )
        ],
        usage=Usage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
    )


def _stream_event_to_chunk(event: dict[str, Any]) -> GenericStreamingChunk:
    """Convert a CLI stream event to a GenericStreamingChunk."""
    usage = event.get("usage")
    usage_block = None
    if usage:
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        usage_block = {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

    provider_fields = None
    thinking = event.get("thinking")
    if thinking:
        provider_fields = {"reasoning_content": thinking}

    return {  # type: ignore[return-value]
        "text": event.get("text", ""),
        "is_finished": event.get("is_finished", False),
        "finish_reason": "stop" if event.get("is_finished") else None,
        "index": 0,
        "tool_use": None,
        "usage": usage_block,
        "provider_specific_fields": provider_fields,
    }


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


class ClaudeProxyHandler(CustomLLM):
    """Custom LLM backend that routes requests to the claude CLI."""

    _session_id: str | None = None

    def _log_request(self, model: str, messages: list, kwargs: dict) -> None:  # noqa: ANN401
        logger.info(
            "Request | model={model} messages={n}",
            model=model,
            n=len(messages),
        )

    def _update_session(self, result: dict[str, Any]) -> None:
        session_id = result.get("session_id")
        if session_id:
            ClaudeProxyHandler._session_id = session_id

    def completion(self, model: str, messages: list, **kwargs: Any) -> ModelResponse:  # type: ignore[override]
        self._log_request(model, messages, kwargs)
        prompt = _extract_prompt(messages)
        system_prompt = _extract_system_prompt(messages)
        session_id = _get_session_id(kwargs, messages, self._session_id)
        model_name, effort = _get_model_and_effort(model)
        cwd = _get_cwd(kwargs)

        if _is_new_conversation(messages):
            ClaudeProxyHandler._session_id = None

        result = cli.run_sync(
            prompt, session_id=session_id, model=model_name, effort=effort,
            cwd=cwd, system_prompt=system_prompt, append_system_prompt=_append_system_prompt(),
        )
        self._update_session(result)
        return _build_model_response(result, model)

    async def acompletion(self, model: str, messages: list, **kwargs: Any) -> ModelResponse:  # type: ignore[override]
        self._log_request(model, messages, kwargs)
        prompt = _extract_prompt(messages)
        system_prompt = _extract_system_prompt(messages)
        session_id = _get_session_id(kwargs, messages, self._session_id)
        model_name, effort = _get_model_and_effort(model)
        cwd = _get_cwd(kwargs)

        if _is_new_conversation(messages):
            ClaudeProxyHandler._session_id = None

        result = await cli.run_async(
            prompt, session_id=session_id, model=model_name, effort=effort,
            cwd=cwd, system_prompt=system_prompt, append_system_prompt=_append_system_prompt(),
        )
        self._update_session(result)
        return _build_model_response(result, model)

    def streaming(self, model: str, messages: list, **kwargs: Any) -> Iterator[GenericStreamingChunk]:  # type: ignore[override]
        self._log_request(model, messages, kwargs)
        prompt = _extract_prompt(messages)
        system_prompt = _extract_system_prompt(messages)
        session_id = _get_session_id(kwargs, messages, self._session_id)
        model_name, effort = _get_model_and_effort(model)
        cwd = _get_cwd(kwargs)

        if _is_new_conversation(messages):
            ClaudeProxyHandler._session_id = None

        for event in cli.stream_sync(
            prompt, session_id=session_id, model=model_name, effort=effort,
            cwd=cwd, system_prompt=system_prompt, append_system_prompt=_append_system_prompt(),
        ):
            if event.get("session_id"):
                ClaudeProxyHandler._session_id = event["session_id"]
            yield _stream_event_to_chunk(event)

    async def astreaming(self, model: str, messages: list, **kwargs: Any) -> AsyncIterator[GenericStreamingChunk]:  # type: ignore[override]
        self._log_request(model, messages, kwargs)
        prompt = _extract_prompt(messages)
        system_prompt = _extract_system_prompt(messages)
        session_id = _get_session_id(kwargs, messages, self._session_id)
        model_name, effort = _get_model_and_effort(model)
        cwd = _get_cwd(kwargs)

        if _is_new_conversation(messages):
            ClaudeProxyHandler._session_id = None

        async for event in cli.stream_async(
            prompt, session_id=session_id, model=model_name, effort=effort,
            cwd=cwd, system_prompt=system_prompt, append_system_prompt=_append_system_prompt(),
        ):
            if event.get("session_id"):
                ClaudeProxyHandler._session_id = event["session_id"]
            yield _stream_event_to_chunk(event)


handler = ClaudeProxyHandler()
