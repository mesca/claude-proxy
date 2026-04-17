"""Unit tests for SessionPool internal-sid derivation."""

from __future__ import annotations

from claude_proxy.pool import _internal_sid, _tool_names


def test_internal_sid_stable():
    a = _internal_sid("u1", "hello")
    b = _internal_sid("u1", "hello")
    assert a == b


def test_internal_sid_different_user():
    a = _internal_sid("u1", "S")
    b = _internal_sid("u2", "S")
    assert a != b


def test_internal_sid_different_sysprompt():
    a = _internal_sid("u1", "A")
    b = _internal_sid("u1", "B")
    assert a != b


def test_internal_sid_ignores_model():
    # Model is not part of the identity — it's spawn-time state only.
    # Same sysprompt with different models must map to the same session.
    a = _internal_sid("u1", "S")
    b = _internal_sid("u1", "S")
    assert a == b


def test_internal_sid_none_sysprompt():
    a = _internal_sid("u1", None)
    b = _internal_sid("u1", "")
    assert a == b  # None and empty treated the same


def test_tool_names_sorted():
    tools = [
        {"type": "function", "function": {"name": "write"}},
        {"type": "function", "function": {"name": "read"}},
    ]
    assert _tool_names(tools) == ["read", "write"]


def test_tool_names_skips_non_function():
    tools = [
        {"type": "function", "function": {"name": "a"}},
        {"type": "other"},
    ]
    assert _tool_names(tools) == ["a"]
