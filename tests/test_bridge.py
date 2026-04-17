"""Unit tests for the MCP bridge JSON-RPC endpoint."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def app_with_bridge(monkeypatch):
    monkeypatch.setenv("CLAUDE_PROXY_BRIDGE_URL", "http://127.0.0.1:4999/_mcp")
    # Reset singleton pool between tests
    import claude_proxy.pool as pool_mod
    pool_mod._pool = None  # type: ignore[attr-defined]

    from claude_proxy.bridge import router

    app = FastAPI()
    app.include_router(router)
    return app


def test_initialize_returns_protocol(app_with_bridge):
    client = TestClient(app_with_bridge)
    r = client.post(
        "/_mcp",
        headers={"x-sid": "s1"},
        json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == 1
    assert body["result"]["protocolVersion"] == "2024-11-05"
    assert body["result"]["capabilities"]["tools"]["listChanged"] is False


def test_notification_initialized_returns_202(app_with_bridge):
    client = TestClient(app_with_bridge)
    r = client.post(
        "/_mcp",
        headers={"x-sid": "s1"},
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
    )
    assert r.status_code == 202


def test_tools_list_rejects_unknown_session(app_with_bridge):
    client = TestClient(app_with_bridge)
    r = client.post(
        "/_mcp",
        headers={"x-sid": "ghost"},
        json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    )
    body = r.json()
    assert "error" in body
    assert body["error"]["code"] == -32000


def test_method_not_found(app_with_bridge):
    client = TestClient(app_with_bridge)
    r = client.post(
        "/_mcp",
        headers={"x-sid": "s1"},
        json={"jsonrpc": "2.0", "id": 3, "method": "something/unknown", "params": {}},
    )
    body = r.json()
    assert body["error"]["code"] == -32601


def test_tools_list_for_registered_session(app_with_bridge):
    from claude_proxy.pool import get_pool
    from claude_proxy.session import Session

    pool = get_pool()
    sid = "s-registered"
    sess = Session.__new__(Session)
    sess.sid = sid
    sess.tools = [{
        "type": "function",
        "function": {
            "name": "echo",
            "description": "Echo",
            "parameters": {"type": "object", "properties": {"msg": {"type": "string"}}},
        },
    }]
    pool._sessions[sid] = sess  # type: ignore[attr-defined]

    client = TestClient(app_with_bridge)
    r = client.post(
        "/_mcp",
        headers={"x-sid": sid},
        json={"jsonrpc": "2.0", "id": 4, "method": "tools/list", "params": {}},
    )
    body = r.json()
    assert body["result"]["tools"][0]["name"] == "echo"


def test_tools_call_parks_future(app_with_bridge):
    """The tools/call handler awaits Session.on_mcp_tool_call."""
    from claude_proxy.pool import get_pool
    from claude_proxy.session import Session

    pool = get_pool()
    sid = "s-parked"
    sess = Session.__new__(Session)
    sess.sid = sid
    sess.tools = []
    sess.pending_calls = {}
    sess._mcp_calls = asyncio.Queue()

    async def fake_call(name, arguments):
        return f"OK:{name}:{arguments}"

    sess.on_mcp_tool_call = fake_call  # type: ignore[method-assign]
    pool._sessions[sid] = sess  # type: ignore[attr-defined]

    client = TestClient(app_with_bridge)
    r = client.post(
        "/_mcp",
        headers={"x-sid": sid},
        json={
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {"name": "echo", "arguments": {"msg": "hi"}},
        },
    )
    body = r.json()
    assert body["result"]["content"][0]["text"].startswith("OK:echo")
