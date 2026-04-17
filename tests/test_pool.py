"""Unit tests for SessionPool config/tool matching."""

from __future__ import annotations

from claude_proxy.pool import _config_matches, _tools_match
from claude_proxy.session import Session


def _make_session(
    *, model=None, effort=None, system_prompt=None, tools=None,
) -> Session:
    s = Session.__new__(Session)
    s.model = model
    s.effort = effort
    s.system_prompt = system_prompt
    s.tools = tools or []
    return s


def test_config_matches_identity():
    s = _make_session(model="sonnet", system_prompt="hi", tools=[])
    assert _config_matches(s, "sonnet", None, "hi", [])


def test_config_matches_model_differs():
    s = _make_session(model="sonnet")
    assert not _config_matches(s, "haiku", None, None, [])


def test_config_matches_system_prompt_differs():
    s = _make_session(system_prompt="A")
    assert not _config_matches(s, None, None, "B", [])


def test_tools_match_empty():
    assert _tools_match([], [])


def test_tools_match_same_name():
    a = [{"function": {"name": "read", "parameters": {}}}]
    b = [{"function": {"name": "read", "parameters": {}}}]
    assert _tools_match(a, b)


def test_tools_match_different_names():
    a = [{"function": {"name": "read"}}]
    b = [{"function": {"name": "write"}}]
    assert not _tools_match(a, b)


def test_tools_match_different_schemas():
    a = [{"function": {"name": "read", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}]
    b = [{"function": {"name": "read", "parameters": {"type": "object", "properties": {}}}}]
    assert not _tools_match(a, b)


def test_tools_match_different_length():
    assert not _tools_match(
        [{"function": {"name": "a"}}],
        [{"function": {"name": "a"}}, {"function": {"name": "b"}}],
    )
