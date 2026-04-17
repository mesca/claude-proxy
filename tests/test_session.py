"""Unit tests for Session helpers (no subprocess spawn)."""

from __future__ import annotations

from claude_proxy.session import Session, _split_blocks


def test_split_blocks_text_only():
    thinking, text, tools = _split_blocks([
        {"type": "text", "text": "hi "},
        {"type": "text", "text": "there"},
    ])
    assert thinking == ""
    assert text == "hi there"
    assert tools == []


def test_split_blocks_mixed():
    blocks = [
        {"type": "thinking", "thinking": "hmm "},
        {"type": "text", "text": "ok "},
        {"type": "tool_use", "id": "t1", "name": "read", "input": {"path": "/"}},
        {"type": "thinking", "thinking": "and"},
        {"type": "text", "text": "done"},
    ]
    thinking, text, tools = _split_blocks(blocks)
    assert thinking == "hmm and"
    assert text == "ok done"
    assert len(tools) == 1
    assert tools[0]["name"] == "read"


def test_list_mcp_tools_translates_schema():
    session = Session.__new__(Session)
    session.tools = [{
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        },
    }]
    mcp = session.list_mcp_tools()
    assert len(mcp) == 1
    assert mcp[0]["name"] == "read"
    assert mcp[0]["description"] == "Read a file"
    assert mcp[0]["inputSchema"] == {"type": "object", "properties": {"path": {"type": "string"}}}


def test_list_mcp_tools_defaults_empty_schema():
    session = Session.__new__(Session)
    session.tools = [{"type": "function", "function": {"name": "ping"}}]
    mcp = session.list_mcp_tools()
    assert mcp[0]["inputSchema"] == {"type": "object", "properties": {}}


def test_list_mcp_tools_skips_non_function():
    session = Session.__new__(Session)
    session.tools = [
        {"type": "function", "function": {"name": "ok"}},
        {"type": "other"},
        {"type": "function"},  # missing function.name
    ]
    assert [t["name"] for t in session.list_mcp_tools()] == ["ok"]
