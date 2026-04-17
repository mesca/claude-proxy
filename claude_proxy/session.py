"""Long-lived Claude CLI subprocess with structured I/O.

One Session wraps one `claude -p --input-format stream-json ...` process,
keyed by a session UUID. Turns are fed through stdin as JSON messages;
responses stream back through stdout. The MCP bridge (bridge.py) invokes
Session methods to coordinate tool calls.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from claude_proxy.log import logger

CLI_BINARY = "claude"
_MCP_PREFIX = "mcp__proxy__"

# Maximum time a parked MCP tools/call will wait for the client to post its
# result before giving up. Prevents a vanished client from pinning the CLI
# subprocess (and a FastAPI worker) forever.
TOOL_CALL_TIMEOUT_SECONDS = 5 * 60


class _SpawnQuickFailError(Exception):
    """Raised when the CLI subprocess exits fast after spawn (likely wrong session flag)."""


@dataclass
class ToolCall:
    """A tool invocation emitted by Claude, awaiting a client-side result."""

    call_id: str           # OpenAI-facing id (call_xxx), also used in pending_calls
    name: str              # unprefixed tool name (e.g. "read")
    arguments: dict[str, Any]
    future: asyncio.Future[str]  # resolved with the client's tool result text


@dataclass
class TurnEvent:
    """A normalized event emitted during a turn.

    kind ∈ {"text", "thinking", "tool_calls", "end", "error"}
    - text / thinking: delta carries the incremental string
    - tool_calls: calls carries the full list of ToolCall (all at once)
    - end: usage may carry token counts
    - error: message carries the error text
    """

    kind: str
    delta: str = ""
    calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] | None = None
    message: str = ""


class Session:
    """Per-sid long-lived CLI subprocess wrapper.

    Lifecycle:
      start() spawns the subprocess. Each turn is either an initial user
      message or a continuation carrying tool results; iterate the
      async generator returned by run_user_turn / run_tool_result_turn.
      close() shuts down the process.
    """

    def __init__(
        self,
        sid: str,
        *,
        model: str | None,
        effort: str | None,
        system_prompt: str | None,
        tools: list[dict[str, Any]] | None,
        bridge_url: str,
        resume: bool = False,
    ) -> None:
        self.sid = sid
        self.model = model
        self.effort = effort
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.bridge_url = bridge_url
        self.resume = resume

        self._proc: asyncio.subprocess.Process | None = None
        self._stdout_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stdout_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._mcp_calls: asyncio.Queue[ToolCall] = asyncio.Queue()

        # call_id → ToolCall (present from emission until client result resolves)
        self.pending_calls: dict[str, ToolCall] = {}

        self.lock = asyncio.Lock()
        self.last_activity = time.monotonic()
        self._closed = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def is_alive(self) -> bool:
        """Return True iff the subprocess is running and stdin is writable."""
        if self._closed or self._proc is None or self._proc.returncode is not None:
            return False
        stdin = self._proc.stdin
        return stdin is not None and not stdin.is_closing()

    def _build_cmd(self, session_flag: str) -> list[str]:
        mcp_cfg = json.dumps({
            "mcpServers": {
                "proxy": {
                    "type": "http",
                    "url": self.bridge_url,
                    "headers": {"x-sid": self.sid},
                },
            },
        })
        cmd = [
            CLI_BINARY, "-p",
            "--input-format", "stream-json",
            "--output-format", "stream-json", "--verbose",
            session_flag, self.sid,
            "--tools", "",
            "--allowedTools", "mcp__proxy",
            "--disable-slash-commands",
            "--strict-mcp-config",
            "--mcp-config", mcp_cfg,
        ]
        if self.system_prompt:
            cmd += ["--system-prompt", self.system_prompt]
        if self.model:
            cmd += ["--model", self.model]
        if self.effort:
            cmd += ["--effort", self.effort]
        return cmd

    async def start(self) -> None:
        """Spawn the CLI subprocess, falling back between --session-id and --resume.

        A session file exists iff `resume=True` OR the proxy was restarted with
        a sid it had spawned before. We try the expected flag first; if the CLI
        disagrees (session-already-in-use or session-not-found), we retry with
        the other flag.
        """
        flags = ["--resume", "--session-id"] if self.resume else ["--session-id", "--resume"]
        last_err: Exception | None = None
        for attempt, flag in enumerate(flags):
            try:
                await self._spawn_once(flag)
                return
            except _SpawnQuickFailError as e:
                last_err = e
                logger.info(
                    "Session {} spawn with {} failed: {} (attempt {}/{})",
                    self.sid, flag, e, attempt + 1, len(flags),
                )
        err = f"failed to spawn session {self.sid}: {last_err}"
        raise RuntimeError(err)

    async def _spawn_once(self, session_flag: str) -> None:
        cmd = self._build_cmd(session_flag)
        logger.info(
            "Spawning session {} with {} model={} effort={}",
            self.sid, session_flag, self.model, self.effort,
        )
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        # Detect fast-failing spawn (wrong session flag → CLI exits within ~100ms
        # with "already in use" / "not found"). Short window keeps latency low
        # while still catching the common misuse.
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=0.3)
        except TimeoutError:
            # Still running → good; wire up pumps.
            self._stdout_task = asyncio.create_task(
                self._pump_stdout(), name=f"session-stdout-{self.sid}",
            )
            self._stderr_task = asyncio.create_task(
                self._pump_stderr(), name=f"session-stderr-{self.sid}",
            )
            return
        # Exited fast — read stderr to determine retryable.
        stderr_bytes = b""
        if self._proc.stderr is not None:
            with contextlib.suppress(Exception):
                stderr_bytes = await self._proc.stderr.read()
        msg = stderr_bytes.decode(errors="replace").strip() or f"exit code {self._proc.returncode}"
        self._proc = None
        raise _SpawnQuickFailError(msg)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        logger.info("Closing session {}", self.sid)
        proc = self._proc
        if proc is None:
            return
        # Cancel any pending MCP futures so the bridge unwinds cleanly.
        for call in list(self.pending_calls.values()):
            if not call.future.done():
                call.future.set_exception(RuntimeError("session closed"))
        self.pending_calls.clear()
        with contextlib.suppress(Exception):
            if proc.stdin and not proc.stdin.is_closing():
                proc.stdin.close()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except TimeoutError:
            proc.kill()
            await proc.wait()
        if self._stdout_task:
            self._stdout_task.cancel()
        if self._stderr_task:
            self._stderr_task.cancel()

    # ------------------------------------------------------------------
    # Background stdout/stderr pumps
    # ------------------------------------------------------------------

    async def _pump_stdout(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None  # noqa: S101
        try:
            while True:
                line = await self._proc.stdout.readline()
                if not line:
                    break
                try:
                    event = json.loads(line.decode())
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                await self._stdout_queue.put(event)
        finally:
            await self._stdout_queue.put(None)  # EOF sentinel

    async def _pump_stderr(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None  # noqa: S101
        with contextlib.suppress(Exception):
            while True:
                line = await self._proc.stderr.readline()
                if not line:
                    break
                text = line.decode(errors="replace").rstrip()
                if text:
                    logger.debug("[cli stderr sid={}] {}", self.sid, text)

    # ------------------------------------------------------------------
    # MCP bridge integration (called from bridge handlers)
    # ------------------------------------------------------------------

    async def on_mcp_tool_call(self, name: str, arguments: dict[str, Any]) -> str:
        """Called by the MCP bridge when a tools/call arrives.

        Blocks until the HTTP client posts a tool result for this call_id.
        Returns the text to put in the MCP response content.
        """
        unprefixed = name.removeprefix(_MCP_PREFIX)
        loop = asyncio.get_running_loop()
        call = ToolCall(
            call_id=f"call_{uuid.uuid4().hex[:12]}",
            name=unprefixed,
            arguments=arguments,
            future=loop.create_future(),
        )
        self.pending_calls[call.call_id] = call
        await self._mcp_calls.put(call)
        logger.debug("mcp tools/call → {} name={} call_id={}", self.sid, unprefixed, call.call_id)

        try:
            return await asyncio.wait_for(call.future, timeout=TOOL_CALL_TIMEOUT_SECONDS)
        except TimeoutError:
            logger.warning(
                "Tool call {} timed out after {}s; client never posted result",
                call.call_id, TOOL_CALL_TIMEOUT_SECONDS,
            )
            err = f"tool call timed out after {TOOL_CALL_TIMEOUT_SECONDS}s"
            raise RuntimeError(err) from None
        finally:
            self.pending_calls.pop(call.call_id, None)

    def list_mcp_tools(self) -> list[dict[str, Any]]:
        """Translate stored OpenAI tools to MCP tool definitions."""
        out: list[dict[str, Any]] = []
        for tool in self.tools:
            if tool.get("type") != "function":
                continue
            fn = tool.get("function") or {}
            name = fn.get("name")
            if not name:
                continue
            out.append({
                "name": name,
                "description": fn.get("description", ""),
                "inputSchema": fn.get("parameters") or {"type": "object", "properties": {}},
            })
        return out

    # ------------------------------------------------------------------
    # Turn execution
    # ------------------------------------------------------------------

    async def run_user_turn(self, text: str) -> AsyncIterator[TurnEvent]:
        """Feed a user text message and iterate until tool_calls or end."""
        async for evt in self._run_turn(user_text=text, tool_results=None):
            yield evt

    async def run_tool_result_turn(
        self, results: dict[str, str],
    ) -> AsyncIterator[TurnEvent]:
        """Resolve pending tool futures (CLI unblocks) and iterate next iteration."""
        async for evt in self._run_turn(user_text=None, tool_results=results):
            yield evt

    async def _run_turn(
        self,
        *,
        user_text: str | None,
        tool_results: dict[str, str] | None,
    ) -> AsyncIterator[TurnEvent]:
        async with self.lock:
            self.last_activity = time.monotonic()

            if tool_results is not None:
                matched = 0
                for call_id, content in tool_results.items():
                    call = self.pending_calls.get(call_id)
                    if call and not call.future.done():
                        call.future.set_result(content)
                        matched += 1
                    else:
                        logger.warning("No pending call for id={} sid={}", call_id, self.sid)
                if matched == 0:
                    err = (
                        "tool_result(s) reference call_ids unknown to this session; "
                        "the underlying subprocess was respawned mid-cycle. "
                        "Restart the conversation with a fresh request."
                    )
                    raise RuntimeError(err)

            if user_text is not None:
                await self._send_user(user_text)

            try:
                async for evt in self._iterate_events():
                    self.last_activity = time.monotonic()
                    yield evt
            except (asyncio.CancelledError, GeneratorExit):
                # Client disconnected mid-turn. The subprocess is in an
                # undefined state (possibly mid-stream, possibly blocked on
                # an orphaned tool call). Mark the session dead so the next
                # request spawns a fresh subprocess via --resume.
                logger.warning("Session {} turn interrupted; marking for respawn", self.sid)
                self._closed = True
                raise

    async def _send_user(self, text: str) -> None:
        assert self._proc is not None and self._proc.stdin is not None  # noqa: S101
        msg = json.dumps({
            "type": "user",
            "message": {"role": "user", "content": [{"type": "text", "text": text}]},
        }) + "\n"
        self._proc.stdin.write(msg.encode())
        await self._proc.stdin.drain()

    async def _iterate_events(self) -> AsyncIterator[TurnEvent]:
        """Read one Claude iteration from stdout.

        Ends either with an `end` event (CLI emitted its turn-level result)
        or with a `tool_calls` event (assistant emitted tool_use blocks and
        we've collected all the matching /_mcp requests).
        """
        prev_thinking = ""
        prev_text = ""

        while True:
            event = await self._stdout_queue.get()
            if event is None:
                yield TurnEvent(kind="error", message="CLI subprocess exited unexpectedly")
                return

            etype = event.get("type")
            if etype == "assistant":
                msg = event.get("message", {}) or {}
                blocks = msg.get("content") or []
                thinking, text, tool_uses = _split_blocks(blocks)

                tdelta = thinking[len(prev_thinking):]
                xdelta = text[len(prev_text):]
                if tdelta:
                    yield TurnEvent(kind="thinking", delta=tdelta)
                if xdelta:
                    yield TurnEvent(kind="text", delta=xdelta)
                prev_thinking = thinking
                prev_text = text

                if tool_uses:
                    # Collect N matching /_mcp arrivals (CLI invokes them shortly).
                    calls: list[ToolCall] = []
                    for _ in tool_uses:
                        call = await self._mcp_calls.get()
                        calls.append(call)
                    yield TurnEvent(kind="tool_calls", calls=calls)
                    return

            elif etype == "result":
                usage_raw = event.get("usage") or {}
                usage = {
                    "input_tokens": int(usage_raw.get("input_tokens", 0) or 0),
                    "output_tokens": int(usage_raw.get("output_tokens", 0) or 0),
                }
                if event.get("is_error"):
                    yield TurnEvent(kind="error", message=str(event.get("result", "")))
                    return
                yield TurnEvent(kind="end", usage=usage)
                return

            # Ignore system/init/user/rate_limit_event events.


def _split_blocks(
    blocks: list[dict[str, Any]],
) -> tuple[str, str, list[dict[str, Any]]]:
    thinking_parts: list[str] = []
    text_parts: list[str] = []
    tool_uses: list[dict[str, Any]] = []
    for b in blocks:
        t = b.get("type")
        if t == "thinking":
            thinking_parts.append(b.get("thinking", ""))
        elif t == "text":
            text_parts.append(b.get("text", ""))
        elif t == "tool_use":
            tool_uses.append(b)
    return "".join(thinking_parts), "".join(text_parts), tool_uses
