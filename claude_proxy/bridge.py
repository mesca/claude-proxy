"""In-process MCP bridge mounted at /_mcp.

The Claude CLI is launched with a --mcp-config pointing here. For every
turn the CLI sends:
  initialize → notifications/initialized → tools/list → [tools/call]*

The bridge routes tools/call to the Session, which parks the call in a
Future until the OpenAI-protocol HTTP client posts a matching tool
result in a subsequent /v1/chat/completions request.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from claude_proxy.log import logger
from claude_proxy.pool import get_pool

router = APIRouter()

_PROTOCOL_VERSION = "2024-11-05"
_SERVER_INFO = {"name": "claude-proxy-bridge", "version": "1"}


@router.post("/_mcp")
async def mcp_endpoint(request: Request) -> Response:
    sid = request.headers.get("x-sid", "")
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        # Notifications may have empty body on some transports
        return Response(status_code=202)

    if isinstance(body, list):
        return _err(None, -32600, "batch not supported")

    method = body.get("method")
    req_id = body.get("id")
    params = body.get("params") or {}

    if method == "initialize":
        return _ok(req_id, {
            "protocolVersion": _PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": _SERVER_INFO,
        })

    if method in {"notifications/initialized", "notifications/cancelled"}:
        return Response(status_code=202)

    if method == "ping":
        return _ok(req_id, {})

    if method not in {"tools/list", "tools/call"}:
        return _err(req_id, -32601, f"method not found: {method}")

    session = get_pool().get(sid)
    if session is None:
        return _err(req_id, -32000, f"unknown session sid={sid}")

    if method == "tools/list":
        return _ok(req_id, {"tools": session.list_mcp_tools()})

    # tools/call
    name = params.get("name", "")
    arguments = params.get("arguments") or {}
    try:
        text = await session.on_mcp_tool_call(name, arguments)
        return _ok(req_id, {"content": [{"type": "text", "text": text}]})
    except Exception as e:  # noqa: BLE001
        logger.warning("tools/call failed sid={} name={}: {}", sid, name, e)
        return _ok(req_id, {
            "content": [{"type": "text", "text": f"tool error: {e}"}],
            "isError": True,
        })


def _ok(req_id: Any, result: dict[str, Any]) -> JSONResponse:  # noqa: ANN401
    return JSONResponse({"jsonrpc": "2.0", "id": req_id, "result": result})


def _err(req_id: Any, code: int, message: str) -> JSONResponse:  # noqa: ANN401
    return JSONResponse({"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}})
