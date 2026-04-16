"""Low-level subprocess wrapper for the claude CLI."""

from __future__ import annotations

import asyncio
import json
import subprocess
from collections.abc import AsyncIterator, Iterator
from typing import Any

from claude_proxy.log import logger

CLI_BINARY = "claude"


def build_command(
    prompt: str,
    *,
    session_id: str | None = None,
    model: str | None = None,
    effort: str | None = None,
    streaming: bool = False,
    system_prompt: str | None = None,
    append_system_prompt: bool = False,
) -> list[str]:
    """Build the claude CLI command arguments."""
    # Prepend a space if prompt starts with "-" to prevent the CLI
    # argument parser from interpreting it as a flag.
    safe_prompt = f" {prompt}" if prompt.startswith("-") else prompt

    cmd = [
        CLI_BINARY,
        "-p",
        safe_prompt,
        "--tools",
        "",
        "--allowedTools",
        "",
        "--disable-slash-commands",
        "--strict-mcp-config",
    ]

    if system_prompt:
        flag = "--append-system-prompt" if append_system_prompt else "--system-prompt"
        cmd.extend([flag, system_prompt])

    if streaming:
        cmd.extend(["--output-format", "stream-json", "--verbose"])
    else:
        cmd.extend(["--output-format", "json"])

    if session_id:
        cmd.extend(["--resume", session_id])

    if model:
        cmd.extend(["--model", model])

    if effort:
        cmd.extend(["--effort", effort])

    return cmd


def _parse_result(raw: str) -> dict[str, Any]:
    """Parse the JSON result from a non-streaming CLI call."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        msg = f"Failed to parse CLI output: {e}"
        raise ClaudeCliError(502, msg) from e

    if data.get("is_error"):
        msg = data.get("result", "Unknown CLI error")
        raise ClaudeCliError(500, msg)

    return data


def _parse_stream_line(line: str) -> dict[str, Any] | None:
    """Parse a single JSON line from stream-json output. Returns None for unparseable lines."""
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


class ClaudeCliError(Exception):
    """Error from the claude CLI."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        self.message = message
        super().__init__(message)


def _extract_content_from_blocks(
    content_blocks: list[dict[str, Any]],
) -> tuple[str, str]:
    """Extract cumulative thinking and text from content blocks.

    Returns (thinking, text) tuple.
    """
    thinking = ""
    text = ""
    for block in content_blocks:
        block_type = block.get("type")
        if block_type == "thinking":
            thinking += block.get("thinking", "")
        elif block_type == "text":
            text += block.get("text", "")
    return thinking, text


class _StreamState:
    """Tracks state across streaming events for delta computation."""

    __slots__ = ("previous_thinking", "previous_text")

    def __init__(self) -> None:
        self.previous_thinking = ""
        self.previous_text = ""


def _extract_stream_event(
    event: dict[str, Any],
    state: _StreamState,
) -> dict[str, Any] | None:
    """Extract a normalized stream event from a raw CLI JSON event.

    Returns None if the event should be skipped.
    Thinking content is returned separately from text content.
    """
    event_type = event.get("type")

    if event_type == "assistant":
        message = event.get("message", {})
        content_blocks = message.get("content", [])
        thinking, text = _extract_content_from_blocks(content_blocks)

        thinking_delta = thinking[len(state.previous_thinking) :]
        text_delta = text[len(state.previous_text) :]

        if not thinking_delta and not text_delta:
            return None

        state.previous_thinking = thinking
        state.previous_text = text

        return {
            "type": "delta",
            "text": text_delta,
            "thinking": thinking_delta or None,
            "session_id": None,
            "usage": None,
            "is_finished": False,
        }

    if event_type == "result":
        usage = event.get("usage", {})
        return {
            "type": "result",
            "text": "",
            "thinking": None,
            "session_id": event.get("session_id"),
            "usage": {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
            },
            "is_finished": True,
        }

    return None


def run_sync(
    prompt: str,
    *,
    session_id: str | None = None,
    model: str | None = None,
    effort: str | None = None,
    cwd: str | None = None,
    timeout: float | None = None,
    system_prompt: str | None = None,
    append_system_prompt: bool = False,
) -> dict[str, Any]:
    """Run a non-streaming claude CLI call synchronously."""
    cmd = build_command(prompt, session_id=session_id, model=model, effort=effort, streaming=False, system_prompt=system_prompt, append_system_prompt=append_system_prompt)
    logger.debug("Running command: {} cwd={}", cmd, cwd)

    try:
        result = subprocess.run(  # noqa: S603
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
    except FileNotFoundError as e:
        msg = f"claude CLI not found: {e}"
        raise ClaudeCliError(503, msg) from e
    except subprocess.TimeoutExpired as e:
        msg = f"claude CLI timed out after {timeout}s"
        raise ClaudeCliError(504, msg) from e

    if result.returncode != 0:
        msg = result.stderr.strip() or f"claude CLI exited with code {result.returncode}"
        raise ClaudeCliError(500, msg)

    return _parse_result(result.stdout)


async def run_async(
    prompt: str,
    *,
    session_id: str | None = None,
    model: str | None = None,
    effort: str | None = None,
    cwd: str | None = None,
    timeout: float | None = None,
    system_prompt: str | None = None,
    append_system_prompt: bool = False,
) -> dict[str, Any]:
    """Run a non-streaming claude CLI call asynchronously."""
    cmd = build_command(prompt, session_id=session_id, model=model, effort=effort, streaming=False, system_prompt=system_prompt, append_system_prompt=append_system_prompt)
    logger.debug("Running command: {} cwd={}", cmd, cwd)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
    except FileNotFoundError as e:
        msg = f"claude CLI not found: {e}"
        raise ClaudeCliError(503, msg) from e

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout,
        )
    except TimeoutError as e:
        proc.kill()
        await proc.wait()
        msg = f"claude CLI timed out after {timeout}s"
        raise ClaudeCliError(504, msg) from e

    stdout = stdout_bytes.decode() if stdout_bytes else ""
    stderr = stderr_bytes.decode() if stderr_bytes else ""

    if proc.returncode != 0:
        msg = stderr.strip() or f"claude CLI exited with code {proc.returncode}"
        raise ClaudeCliError(500, msg)

    return _parse_result(stdout)


def stream_sync(
    prompt: str,
    *,
    session_id: str | None = None,
    model: str | None = None,
    effort: str | None = None,
    cwd: str | None = None,
    timeout: float | None = None,
    system_prompt: str | None = None,
    append_system_prompt: bool = False,
) -> Iterator[dict[str, Any]]:
    """Stream claude CLI output synchronously."""
    cmd = build_command(prompt, session_id=session_id, model=model, effort=effort, streaming=True, system_prompt=system_prompt, append_system_prompt=append_system_prompt)
    logger.debug("Streaming command: {} cwd={}", cmd, cwd)

    try:
        proc = subprocess.Popen(  # noqa: S603
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
        )
    except FileNotFoundError as e:
        msg = f"claude CLI not found: {e}"
        raise ClaudeCliError(503, msg) from e

    state = _StreamState()
    try:
        for line in proc.stdout:  # type: ignore[union-attr]
            raw_event = _parse_stream_line(line)
            if raw_event is None:
                continue
            event = _extract_stream_event(raw_event, state)
            if event is None:
                continue
            yield event
    finally:
        proc.wait(timeout=timeout)
        if proc.returncode and proc.returncode != 0:
            stderr = proc.stderr.read() if proc.stderr else ""  # type: ignore[union-attr]
            logger.warning("claude CLI exited with code {}: {}", proc.returncode, stderr.strip())


async def stream_async(
    prompt: str,
    *,
    session_id: str | None = None,
    model: str | None = None,
    effort: str | None = None,
    cwd: str | None = None,
    timeout: float | None = None,
    system_prompt: str | None = None,
    append_system_prompt: bool = False,
) -> AsyncIterator[dict[str, Any]]:
    """Stream claude CLI output asynchronously."""
    cmd = build_command(prompt, session_id=session_id, model=model, effort=effort, streaming=True, system_prompt=system_prompt, append_system_prompt=append_system_prompt)
    logger.debug("Streaming command: {} cwd={}", cmd, cwd)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
    except FileNotFoundError as e:
        msg = f"claude CLI not found: {e}"
        raise ClaudeCliError(503, msg) from e

    state = _StreamState()
    try:
        while True:
            line_bytes = await proc.stdout.readline()  # type: ignore[union-attr]
            if not line_bytes:
                break
            line = line_bytes.decode()
            raw_event = _parse_stream_line(line)
            if raw_event is None:
                continue
            event = _extract_stream_event(raw_event, state)
            if event is None:
                continue
            yield event
    finally:
        await proc.wait()
        if proc.returncode and proc.returncode != 0:
            stderr_bytes = await proc.stderr.read() if proc.stderr else b""  # type: ignore[union-attr]
            logger.warning("claude CLI exited with code {}: {}", proc.returncode, stderr_bytes.decode().strip())
