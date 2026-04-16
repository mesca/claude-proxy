"""Custom LLM handler that routes requests to the claude CLI.

Tool-free requests go through the CLI with --resume for session continuity.
Tool-using requests add tool definitions to the system prompt and parse
Claude's response for structured tool_calls JSON.
"""

from __future__ import annotations

import json
import os
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

from claude_proxy import cli
from claude_proxy.cli import ClaudeCliError
from claude_proxy.log import logger
from claude_proxy.middleware import session_var, tools_var
from claude_proxy.models import parse_model_string

# ---------------------------------------------------------------------------
# Helpers shared by both paths
# ---------------------------------------------------------------------------


def _content_to_text(content: str | list[dict[str, Any]] | None) -> str:
    """Extract text from OpenAI content (string or content-block list)."""
    if isinstance(content, list):
        parts: list[str] = [
            b.get("text", "") for b in content if b.get("type") == "text"
        ]
        return "".join(parts)
    return str(content) if content else ""


def _format_history(messages: list[dict[str, Any]]) -> str:
    """Format the full message history as a single prompt (stateless mode).

    Reproduces OpenAI multi-turn behavior: each message is labeled with its
    role so Claude can follow the conversation without --resume.
    """
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            continue  # handled separately via --system-prompt
        content = _content_to_text(msg.get("content", ""))
        if role == "user":
            parts.append(f"[user]\n{content}")
        elif role == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                tc_json = json.dumps(tool_calls, indent=2)
                if content:
                    parts.append(f"[assistant]\n{content}\n{tc_json}")
                else:
                    parts.append(f"[assistant]\n{tc_json}")
            elif content:
                parts.append(f"[assistant]\n{content}")
        elif role == "tool":
            name = msg.get("name", "unknown")
            call_id = msg.get("tool_call_id", "")
            parts.append(
                f'<tool_result name="{name}" call_id="{call_id}">\n'
                f"{content}\n"
                f"</tool_result>"
            )
    return "\n\n".join(parts)


def _extract_prompt(messages: list[dict[str, Any]]) -> str:
    """Extract the prompt to send to Claude.

    In stateless mode: format the full message history.
    In session mode: return the last user message or tool results.
    """
    if os.environ.get("CLAUDE_PROXY_STATELESS") == "1":
        return _format_history(messages)
    if _is_tool_result_turn(messages):
        return _format_tool_results(messages)
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return _content_to_text(msg.get("content", ""))
    err = "No user message found in the request"
    raise ClaudeCliError(400, err)


def _build_tool_instructions(tools: list[dict[str, Any]]) -> str:
    """Build tool protocol instructions with full schemas from the request."""
    lines = [
        "\n\nYou have access to tools provided by the client. "
        "To call a tool, respond with ONLY a JSON object:\n"
        '{"tool_calls": [{"id": "<unique_id>", "name": "<tool>", "arguments": {...}}]}\n\n'
        "Include ALL required parameters exactly as defined below. "
        "To respond with text, just respond normally.\n\n"
        "IMPORTANT: After you call a tool, the result will appear in your next "
        "message inside <tool_result> XML tags. This is YOUR tool's output, not "
        "input from the user. Process it and respond to the user.\n\n"
        "Available tools:",
    ]
    for tool in tools:
        if tool.get("type") != "function":
            continue
        fn = tool.get("function", {})
        name = fn.get("name", "")
        desc = fn.get("description", "")
        params = fn.get("parameters")
        line = f"\n- {name}"
        if desc:
            line += f": {desc}"
        lines.append(line)
        if params:
            lines.append(f"\n  Parameters: {json.dumps(params)}")
    return "".join(lines)


def _extract_system_prompt(messages: list[dict[str, Any]]) -> str:
    """Extract the system message and append tool protocol instructions.

    Always returns a non-empty string so --system-prompt replaces Claude's
    default (which contains built-in tool descriptions we don't want).
    If tools are available (from middleware context var), appends full
    schemas and protocol instructions.
    """
    client_system = ""
    for msg in messages:
        if msg.get("role") == "system":
            client_system = _content_to_text(msg.get("content", ""))
            break

    prompt = client_system or "You are a helpful assistant."

    tools = tools_var.get()
    if tools:
        prompt += _build_tool_instructions(tools)

    return prompt


def _get_model_and_effort(model: str) -> tuple[str | None, str | None]:
    return parse_model_string(model)


def _get_session_id() -> str | None:
    """Read session ID from context var (set by middleware)."""
    return session_var.get()


def _is_session_error(e: ClaudeCliError) -> bool:
    """Check if a CLI error is session-related (not found or already in use)."""
    msg = e.message.lower()
    return "session" in msg or "conversation" in msg


def _run_with_session(
    prompt: str,
    *,
    model: str | None = None,
    effort: str | None = None,
    system_prompt: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Run CLI with session retry: --resume → --session-id → no session."""
    sid = _get_session_id()
    kw = {"model": model, "effort": effort, "system_prompt": system_prompt, "timeout": timeout}

    if not sid:
        return cli.run_sync(prompt, **kw)

    # Try resume existing session
    try:
        return cli.run_sync(prompt, session_id=sid, **kw)
    except ClaudeCliError as e:
        if not _is_session_error(e):
            raise
        logger.info("Session --resume failed, trying --session-id: {}", sid)

    # Try create new session with our UUID
    try:
        return cli.run_sync(prompt, session_id=sid, create_session=True, **kw)
    except ClaudeCliError as e:
        if not _is_session_error(e):
            raise
        logger.warning("Session --session-id also failed, running without session: {}", sid)

    # Fall back to no session
    return cli.run_sync(prompt, **kw)


async def _run_with_session_async(
    prompt: str,
    *,
    model: str | None = None,
    effort: str | None = None,
    system_prompt: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Async version of _run_with_session."""
    sid = _get_session_id()
    kw = {"model": model, "effort": effort, "system_prompt": system_prompt, "timeout": timeout}

    if not sid:
        return await cli.run_async(prompt, **kw)

    try:
        return await cli.run_async(prompt, session_id=sid, **kw)
    except ClaudeCliError as e:
        if not _is_session_error(e):
            raise
        logger.info("Session --resume failed, trying --session-id: {}", sid)

    try:
        return await cli.run_async(prompt, session_id=sid, create_session=True, **kw)
    except ClaudeCliError as e:
        if not _is_session_error(e):
            raise
        logger.warning("Session --session-id also failed, running without session: {}", sid)

    return await cli.run_async(prompt, **kw)


def _stream_with_session(
    prompt: str,
    *,
    model: str | None = None,
    effort: str | None = None,
    system_prompt: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Stream CLI with session retry: --resume → --session-id → no session."""
    sid = _get_session_id()
    kw = {"model": model, "effort": effort, "system_prompt": system_prompt}

    if not sid:
        yield from cli.stream_sync(prompt, **kw)
        return

    try:
        yield from cli.stream_sync(prompt, session_id=sid, **kw)
        return
    except ClaudeCliError as e:
        if not _is_session_error(e):
            raise
        logger.info("Session --resume failed, trying --session-id: {}", sid)

    try:
        yield from cli.stream_sync(prompt, session_id=sid, create_session=True, **kw)
        return
    except ClaudeCliError as e:
        if not _is_session_error(e):
            raise
        logger.warning("Session --session-id also failed, running without session: {}", sid)

    yield from cli.stream_sync(prompt, **kw)


async def _stream_with_session_async(
    prompt: str,
    *,
    model: str | None = None,
    effort: str | None = None,
    system_prompt: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Async stream CLI with session retry: --resume → --session-id → no session."""
    sid = _get_session_id()
    kw = {"model": model, "effort": effort, "system_prompt": system_prompt}

    if not sid:
        async for event in cli.stream_async(prompt, **kw):
            yield event
        return

    try:
        async for event in cli.stream_async(prompt, session_id=sid, **kw):
            yield event
        return
    except ClaudeCliError as e:
        if not _is_session_error(e):
            raise
        logger.info("Session --resume failed, trying --session-id: {}", sid)

    try:
        async for event in cli.stream_async(prompt, session_id=sid, create_session=True, **kw):
            yield event
        return
    except ClaudeCliError as e:
        if not _is_session_error(e):
            raise
        logger.warning("Session --session-id also failed, running without session: {}", sid)

    async for event in cli.stream_async(prompt, **kw):
        yield event


def _get_tools(kwargs: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Extract tool definitions from optional_params. Returns None if absent."""
    optional_params = kwargs.get("optional_params", {})
    tools = optional_params.get("tools")
    return tools if tools else None


# ---------------------------------------------------------------------------
# Tool protocol: system prompt, prompt formatting, response parsing
# ---------------------------------------------------------------------------

_TOOL_INSTRUCTIONS = """\
You have access to the following tools. To use a tool, respond with ONLY a \
JSON object (no other text before or after):
{"tool_calls": [{"id": "<unique_id>", "name": "<tool_name>", "arguments": {...}}]}

To respond with text, just respond normally without JSON.

Tool results will be provided in XML tags:
<tool_result name="tool_name" call_id="id">result content</tool_result>

Available tools:
"""


def _build_tool_system_prompt(
    client_system: str | None,
    tools: list[dict[str, Any]],
) -> str:
    """Build a system prompt that includes tool definitions."""
    parts: list[str] = []
    if client_system:
        parts.append(client_system)
        parts.append("")  # blank line separator

    parts.append(_TOOL_INSTRUCTIONS)
    for tool in tools:
        if tool.get("type") != "function":
            continue
        fn = tool["function"]
        name = fn.get("name", "unknown")
        desc = fn.get("description", "")
        params = fn.get("parameters", {})
        line = f"- {name}"
        if desc:
            line += f": {desc}"
        parts.append(line)
        if params:
            parts.append(f"  Parameters: {json.dumps(params)}")

    return "\n".join(parts)


def _is_tool_result_turn(messages: list[dict[str, Any]]) -> bool:
    """Detect a tool-result turn: at least one tool message after the last assistant."""
    # Walk backwards; if we hit a tool message before a user message, it's a tool result turn
    for msg in reversed(messages):
        role = msg.get("role")
        if role == "tool":
            return True
        if role == "user":
            return False
        if role == "assistant":
            return False
    return False


def _format_tool_results(messages: list[dict[str, Any]]) -> str:
    """Format tool result messages and optional trailing user message as the prompt.

    Expects messages ending with: ..., assistant(tool_calls), tool, [tool, ...], [user]
    """
    parts: list[str] = []
    trailing_user: str | None = None

    # Collect tool results and optional trailing user message
    for msg in reversed(messages):
        role = msg.get("role")
        if role == "user" and not parts:
            trailing_user = _content_to_text(msg.get("content", ""))
        elif role == "tool":
            name = msg.get("name", "unknown")
            call_id = msg.get("tool_call_id", "")
            content = str(msg.get("content", ""))
            parts.append(
                f'<tool_result name="{name}" call_id="{call_id}">\n'
                f"{content}\n"
                f"</tool_result>"
            )
        else:
            break  # stop at assistant or system

    parts.reverse()
    prompt = "\n".join(parts)
    if trailing_user:
        prompt += f"\n\n{trailing_user}"
    return prompt


def _parse_tool_response(result_text: str) -> list[dict[str, Any]] | None:
    """Try to parse a tool_calls JSON from Claude's response.

    Returns the tool_calls list if the response is a tool call, None otherwise.
    """
    idx = result_text.find('{"tool_calls"')
    if idx == -1:
        return None
    try:
        data, _ = json.JSONDecoder().raw_decode(result_text, idx)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    raw = data.get("tool_calls")
    if not raw:
        return None
    # Handle both single object and array
    tool_calls: list[dict[str, Any]] = [raw] if isinstance(raw, dict) else raw
    if not isinstance(tool_calls, list) or not tool_calls:
        return None
    # Normalize IDs to OpenAI call_* format
    for tc in tool_calls:
        raw_id = tc.get("id") or uuid.uuid4().hex[:8]
        tc["id"] = raw_id if str(raw_id).startswith("call_") else f"call_{raw_id}"
    return tool_calls


# ---------------------------------------------------------------------------
# Response builders
# ---------------------------------------------------------------------------


def _build_model_response(
    result: dict[str, Any],
    model: str,
) -> ModelResponse:
    """Build a LiteLLM ModelResponse from a CLI result dict.

    Checks the result text for tool_calls JSON first — if found, returns
    a proper tool_calls response instead of raw text.
    """
    result_text = result.get("result", "")

    # Always check if Claude responded with a tool call
    tool_calls = _parse_tool_response(result_text)
    if tool_calls:
        return _build_tool_call_response(tool_calls, result, model)

    raw_usage = result.get("usage", {})
    input_tokens = raw_usage.get("input_tokens", 0)
    output_tokens = raw_usage.get("output_tokens", 0)

    return ModelResponse(
        model=model,
        choices=[
            Choices(
                message=Message(role="assistant", content=result_text),
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


def _build_tool_call_response(
    tool_calls: list[dict[str, Any]],
    result: dict[str, Any],
    model: str,
) -> ModelResponse:
    """Build a ModelResponse with tool_calls (finish_reason: tool_calls)."""
    raw_usage = result.get("usage", {})
    input_tokens = raw_usage.get("input_tokens", 0)
    output_tokens = raw_usage.get("output_tokens", 0)

    tc_objects = [
        ChatCompletionMessageToolCall(
            id=tc["id"],
            type="function",
            function=Function(
                name=tc["name"],
                arguments=json.dumps(tc.get("arguments", {})),
            ),
        )
        for tc in tool_calls
    ]

    return ModelResponse(
        model=model,
        choices=[
            Choices(
                message=Message(role="assistant", content=None, tool_calls=tc_objects),
                finish_reason="tool_calls",
                index=0,
            )
        ],
        usage=Usage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
    )


def _tool_calls_to_chunk(tool_calls: list[dict[str, Any]]) -> GenericStreamingChunk:
    """Build a single streaming chunk carrying tool_calls (finish_reason: tool_calls)."""
    # Emit the first tool call; litellm handles the rest via the non-streaming path
    tc = tool_calls[0]
    return {  # type: ignore[return-value]
        "text": "",
        "is_finished": True,
        "finish_reason": "tool_calls",
        "index": 0,
        "tool_use": {
            "id": tc["id"],
            "type": "function",
            "function": {
                "name": tc["name"],
                "arguments": json.dumps(tc.get("arguments", {})),
            },
            "index": 0,
        },
        "usage": None,
        "provider_specific_fields": None,
    }


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

    def _log_request(self, model: str, messages: list, kwargs: dict) -> None:  # noqa: ANN401
        logger.info(
            "Request | model={model} messages={n}",
            model=model,
            n=len(messages),
        )

    # -- Dispatch: route tool vs non-tool requests --

    def completion(self, model: str, messages: list, **kwargs: Any) -> ModelResponse:  # type: ignore[override]
        self._log_request(model, messages, kwargs)
        tools = _get_tools(kwargs)
        if tools:
            return self._tool_completion(model, messages, tools, **kwargs)
        return self._cli_completion(model, messages, **kwargs)

    async def acompletion(self, model: str, messages: list, **kwargs: Any) -> ModelResponse:  # type: ignore[override]
        self._log_request(model, messages, kwargs)
        tools = _get_tools(kwargs)
        if tools:
            return await self._tool_acompletion(model, messages, tools, **kwargs)
        return await self._cli_acompletion(model, messages, **kwargs)

    def streaming(self, model: str, messages: list, **kwargs: Any) -> Iterator[GenericStreamingChunk]:  # type: ignore[override]
        self._log_request(model, messages, kwargs)
        tools = _get_tools(kwargs)
        if tools:
            yield from self._tool_streaming(model, messages, tools, **kwargs)
            return
        yield from self._cli_streaming(model, messages, **kwargs)

    async def astreaming(self, model: str, messages: list, **kwargs: Any) -> AsyncIterator[GenericStreamingChunk]:  # type: ignore[override]
        self._log_request(model, messages, kwargs)
        tools = _get_tools(kwargs)
        if tools:
            async for chunk in self._tool_astreaming(model, messages, tools, **kwargs):
                yield chunk
            return
        async for chunk in self._cli_astreaming(model, messages, **kwargs):
            yield chunk

    # -- Tool path (tool-using requests) --

    def _tool_completion(
        self, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]], **kwargs: Any,
    ) -> ModelResponse:
        client_system = _extract_system_prompt(messages)
        system_prompt = _build_tool_system_prompt(client_system, tools)
        model_name, effort = _get_model_and_effort(model)

        if _is_tool_result_turn(messages):
            prompt = _format_tool_results(messages)
        else:
            prompt = _extract_prompt(messages)

        result = _run_with_session(
            prompt, model=model_name, effort=effort,
            system_prompt=system_prompt,
        )


        result_text = result.get("result", "")
        tool_calls = _parse_tool_response(result_text)
        if tool_calls:
            return _build_tool_call_response(tool_calls, result, model)
        return _build_model_response(result, model)

    async def _tool_acompletion(
        self, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]], **kwargs: Any,
    ) -> ModelResponse:
        client_system = _extract_system_prompt(messages)
        system_prompt = _build_tool_system_prompt(client_system, tools)
        model_name, effort = _get_model_and_effort(model)

        if _is_tool_result_turn(messages):
            prompt = _format_tool_results(messages)
        else:
            prompt = _extract_prompt(messages)

        result = await _run_with_session_async(
            prompt, model=model_name, effort=effort,
            system_prompt=system_prompt,
        )


        result_text = result.get("result", "")
        tool_calls = _parse_tool_response(result_text)
        if tool_calls:
            return _build_tool_call_response(tool_calls, result, model)
        return _build_model_response(result, model)

    # -- Tool streaming (buffer text, detect tool_calls, preserve thinking) --

    def _tool_streaming(
        self, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]], **kwargs: Any,
    ) -> Iterator[GenericStreamingChunk]:
        client_system = _extract_system_prompt(messages)
        system_prompt = _build_tool_system_prompt(client_system, tools)
        model_name, effort = _get_model_and_effort(model)

        if _is_tool_result_turn(messages):
            prompt = _format_tool_results(messages)
        else:
            prompt = _extract_prompt(messages)

        # Stream events, buffering text to detect tool_calls at the end
        accumulated_text = ""
        buffered_chunks: list[GenericStreamingChunk] = []

        for event in _stream_with_session(
            prompt, model=model_name, effort=effort,
            system_prompt=system_prompt,
        ):

            chunk = _stream_event_to_chunk(event)
            text = event.get("text", "")
            accumulated_text += text

            if not accumulated_text.lstrip().startswith("{"):
                # Definitely not a tool call — flush buffer and stream normally
                for buffered in buffered_chunks:
                    yield buffered
                buffered_chunks.clear()
                yield chunk
            else:
                # Might be a tool call — buffer until we know
                buffered_chunks.append(chunk)

        tool_calls = _parse_tool_response(accumulated_text)
        if tool_calls:
            # Don't emit buffered text chunks — emit tool_calls instead
            yield _tool_calls_to_chunk(tool_calls)
        elif buffered_chunks:
            # Was buffering but turned out to be text — flush
            for buffered in buffered_chunks:
                yield buffered

    async def _tool_astreaming(
        self, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]], **kwargs: Any,
    ) -> AsyncIterator[GenericStreamingChunk]:
        client_system = _extract_system_prompt(messages)
        system_prompt = _build_tool_system_prompt(client_system, tools)
        model_name, effort = _get_model_and_effort(model)

        if _is_tool_result_turn(messages):
            prompt = _format_tool_results(messages)
        else:
            prompt = _extract_prompt(messages)

        accumulated_text = ""
        buffered_chunks: list[GenericStreamingChunk] = []

        async for event in _stream_with_session_async(
            prompt, model=model_name, effort=effort,
            system_prompt=system_prompt,
        ):

            chunk = _stream_event_to_chunk(event)
            text = event.get("text", "")
            accumulated_text += text

            if not accumulated_text.lstrip().startswith("{"):
                for buffered in buffered_chunks:
                    yield buffered
                buffered_chunks.clear()
                yield chunk
            else:
                buffered_chunks.append(chunk)

        tool_calls = _parse_tool_response(accumulated_text)
        if tool_calls:
            yield _tool_calls_to_chunk(tool_calls)
        elif buffered_chunks:
            for buffered in buffered_chunks:
                yield buffered

    # -- CLI path (tool-free requests) --

    def _cli_completion(self, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> ModelResponse:
        prompt = _extract_prompt(messages)
        system_prompt = _extract_system_prompt(messages)
        model_name, effort = _get_model_and_effort(model)

        result = _run_with_session(
            prompt, model=model_name, effort=effort,
            system_prompt=system_prompt,
        )

        return _build_model_response(result, model)

    async def _cli_acompletion(self, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> ModelResponse:
        prompt = _extract_prompt(messages)
        system_prompt = _extract_system_prompt(messages)
        model_name, effort = _get_model_and_effort(model)

        result = await _run_with_session_async(
            prompt, model=model_name, effort=effort,
            system_prompt=system_prompt,
        )

        return _build_model_response(result, model)

    def _cli_streaming(self, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> Iterator[GenericStreamingChunk]:
        prompt = _extract_prompt(messages)
        system_prompt = _extract_system_prompt(messages)
        model_name, effort = _get_model_and_effort(model)

        accumulated_text = ""
        buffered_chunks: list[GenericStreamingChunk] = []

        for event in _stream_with_session(
            prompt, model=model_name, effort=effort,
            system_prompt=system_prompt,
        ):

            chunk = _stream_event_to_chunk(event)
            text = event.get("text", "")
            accumulated_text += text

            if not accumulated_text.lstrip().startswith("{"):
                for buffered in buffered_chunks:
                    yield buffered
                buffered_chunks.clear()
                yield chunk
            else:
                buffered_chunks.append(chunk)

        tool_calls = _parse_tool_response(accumulated_text)
        if tool_calls:
            yield _tool_calls_to_chunk(tool_calls)
        elif buffered_chunks:
            for buffered in buffered_chunks:
                yield buffered

    async def _cli_astreaming(self, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> AsyncIterator[GenericStreamingChunk]:
        prompt = _extract_prompt(messages)
        system_prompt = _extract_system_prompt(messages)
        model_name, effort = _get_model_and_effort(model)

        accumulated_text = ""
        buffered_chunks: list[GenericStreamingChunk] = []

        async for event in _stream_with_session_async(
            prompt, model=model_name, effort=effort,
            system_prompt=system_prompt,
        ):

            chunk = _stream_event_to_chunk(event)
            text = event.get("text", "")
            accumulated_text += text

            if not accumulated_text.lstrip().startswith("{"):
                for buffered in buffered_chunks:
                    yield buffered
                buffered_chunks.clear()
                yield chunk
            else:
                buffered_chunks.append(chunk)

        tool_calls = _parse_tool_response(accumulated_text)
        if tool_calls:
            yield _tool_calls_to_chunk(tool_calls)
        elif buffered_chunks:
            for buffered in buffered_chunks:
                yield buffered


handler = ClaudeProxyHandler()
