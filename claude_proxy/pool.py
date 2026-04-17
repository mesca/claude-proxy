"""Session pool: one long-lived Claude CLI subprocess per (sid, sysprompt).

The pool is keyed by the user's session UUID combined with a hash of the
system prompt. Rationale:

- The system prompt defines the "kind" of conversation. OpenCode's title-gen
  request and the main chat request share a session header but have
  different system prompts; they must not contaminate each other's
  conversation history.
- Everything else — model, effort, tools — is state that may change during
  a conversation (user switches model mid-chat, server adds/removes a tool).
  Changes to these respawn the subprocess with --resume so the on-disk
  history carries over.

Same (sid, sysprompt) across requests → same CLI subprocess → prompt cache
hits and conversation continuity. Different sysprompt → different session,
naturally isolated.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
import time
import uuid
from typing import Any

from claude_proxy.log import logger
from claude_proxy.session import Session

IDLE_TIMEOUT_SECONDS = 15 * 60
REAP_INTERVAL_SECONDS = 60

_INTERNAL_NS = uuid.UUID("b2c3d4e5-f6a7-8901-bcde-f23456789012")


def _internal_sid(user_sid: str, system_prompt: str | None) -> str:
    """sid + sha256(system_prompt) → UUIDv5. Defines CLI session identity."""
    digest = hashlib.sha256((system_prompt or "").encode()).hexdigest()
    return str(uuid.uuid5(_INTERNAL_NS, f"{user_sid}|{digest}"))


class SessionPool:
    """Registry of live sessions, with periodic idle reaping.

    A session is keyed by sid. Its tools list, system prompt, model and
    effort are fixed at spawn time — if any of these change between turns,
    the old session is closed and a fresh one replaces it.
    """

    def __init__(self, bridge_url: str) -> None:
        self.bridge_url = bridge_url
        self._sessions: dict[str, Session] = {}
        self._create_locks: dict[str, asyncio.Lock] = {}
        self._reaper_task: asyncio.Task[None] | None = None

    def _ensure_reaper(self) -> None:
        """Start the reaper task lazily from an async context."""
        if self._reaper_task is not None:
            return
        with contextlib.suppress(RuntimeError):
            self._reaper_task = asyncio.create_task(
                self._reaper_loop(), name="session-reaper",
            )

    def get(self, sid: str) -> Session | None:
        return self._sessions.get(sid)

    async def get_or_create(
        self,
        user_sid: str,
        *,
        model: str | None,
        effort: str | None,
        system_prompt: str | None,
        tools: list[dict[str, Any]] | None,
    ) -> Session:
        """Return the CLI session for (user_sid, system_prompt).

        Model / effort / tools changes respawn the existing session with
        --resume so conversation history survives. A different system
        prompt maps to a different internal sid (different on-disk
        conversation file).
        """
        self._ensure_reaper()
        sid = _internal_sid(user_sid, system_prompt)
        lock = self._create_locks.setdefault(sid, asyncio.Lock())
        async with lock:
            existing = self._sessions.get(sid)
            resume = False
            if existing:
                if not existing.is_alive():
                    logger.warning("Session {} subprocess is dead, respawning", sid)
                    await existing.close()
                    self._sessions.pop(sid, None)
                    resume = True
                elif existing.pending_calls:
                    # Tool cycle in flight; preserve session regardless of
                    # model/tools change — new values apply to next turn.
                    return existing
                elif (
                    existing.model == model
                    and existing.effort == effort
                    and _tool_names(existing.tools) == _tool_names(tools or [])
                ):
                    return existing
                else:
                    logger.info(
                        "Session {} model/tools changed, respawning with --resume",
                        sid,
                    )
                    await existing.close()
                    self._sessions.pop(sid, None)
                    resume = True

            session = Session(
                sid,
                model=model,
                effort=effort,
                system_prompt=system_prompt,
                tools=tools,
                bridge_url=self.bridge_url,
                resume=resume,
            )
            # Pre-register so the MCP bridge can resolve this sid during the
            # CLI's initialize/tools/list handshake (which happens concurrently
            # with spawn). Pop on failure so the entry doesn't leak.
            self._sessions[sid] = session
            try:
                await session.start()
            except Exception:
                self._sessions.pop(sid, None)
                raise
            return session

    async def drop(self, sid: str) -> None:
        session = self._sessions.pop(sid, None)
        self._create_locks.pop(sid, None)
        if session:
            await session.close()

    async def _reaper_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(REAP_INTERVAL_SECONDS)
            except asyncio.CancelledError:
                return
            now = time.monotonic()
            # Only reap sessions that are idle AND not mid-turn. A locked
            # session has an in-flight request; closing it would corrupt the
            # generator that's reading stdout.
            stale = [
                sid for sid, s in self._sessions.items()
                if now - s.last_activity > IDLE_TIMEOUT_SECONDS
                and not s.lock.locked()
            ]
            for sid in stale:
                logger.info("Reaping idle session {}", sid)
                await self.drop(sid)


def _tool_names(tools: list[dict[str, Any]]) -> list[str]:
    return sorted(
        t.get("function", {}).get("name", "")
        for t in tools
        if t.get("type") == "function"
    )


# ---------------------------------------------------------------------------
# Module-level singleton for sharing between handler and bridge
# ---------------------------------------------------------------------------

_pool: SessionPool | None = None


def get_pool() -> SessionPool:
    global _pool  # noqa: PLW0603
    if _pool is None:
        bridge_url = os.environ.get("CLAUDE_PROXY_BRIDGE_URL")
        if not bridge_url:
            err = "CLAUDE_PROXY_BRIDGE_URL not set — proxy must run via __main__"
            raise RuntimeError(err)
        _pool = SessionPool(bridge_url)
    return _pool
