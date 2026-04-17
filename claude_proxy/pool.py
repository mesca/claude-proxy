"""Session pool: one long-lived Claude CLI subprocess per session UUID."""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from typing import Any

from claude_proxy.log import logger
from claude_proxy.session import Session

IDLE_TIMEOUT_SECONDS = 15 * 60
REAP_INTERVAL_SECONDS = 60


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
        sid: str,
        *,
        model: str | None,
        effort: str | None,
        system_prompt: str | None,
        tools: list[dict[str, Any]] | None,
    ) -> Session:
        """Return the session for sid, respawning if dead or configuration changed."""
        self._ensure_reaper()
        lock = self._create_locks.setdefault(sid, asyncio.Lock())
        async with lock:
            existing = self._sessions.get(sid)
            resume = False
            if existing:
                if not existing.is_alive():
                    logger.warning("Session {} subprocess is dead, respawning", sid)
                    await existing.close()
                    self._sessions.pop(sid, None)
                    resume = True  # session file exists on disk
                elif existing.pending_calls:
                    # A tool cycle is in flight; respawning would orphan the
                    # pending MCP RPC. Keep the live session regardless of
                    # config drift — the new model/tools/system prompt will
                    # take effect on the next turn, not mid-cycle.
                    return existing
                elif not _config_matches(existing, model, effort, system_prompt, tools):
                    logger.info("Session {} config changed, respawning", sid)
                    await existing.close()
                    self._sessions.pop(sid, None)
                    resume = True
                else:
                    return existing

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


def _config_matches(
    session: Session,
    model: str | None,
    effort: str | None,
    system_prompt: str | None,
    tools: list[dict[str, Any]] | None,
) -> bool:
    return (
        session.model == model
        and session.effort == effort
        and session.system_prompt == system_prompt
        and _tools_match(session.tools, tools or [])
    )


def _tools_match(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b, strict=True):
        if (x.get("function", {}).get("name") != y.get("function", {}).get("name")
            or x.get("function", {}).get("parameters") != y.get("function", {}).get("parameters")):
            return False
    return True


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
