"""Unit tests for handler helpers (pure functions, no subprocess)."""

from __future__ import annotations

from claude_proxy.handler import (
    _content_to_text,
    _extract_system_prompt,
    _format_history_as_prompt,
    _last_user_text,
    _trailing_tool_results,
)


def test_content_to_text_string():
    assert _content_to_text("hello") == "hello"


def test_content_to_text_blocks():
    assert _content_to_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "ab"


def test_content_to_text_ignores_non_text_blocks():
    assert _content_to_text([{"type": "image", "url": "x"}, {"type": "text", "text": "y"}]) == "y"


def test_content_to_text_none():
    assert _content_to_text(None) == ""


def test_extract_system_prompt_found():
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    assert _extract_system_prompt(msgs) == "S"


def test_extract_system_prompt_default():
    assert _extract_system_prompt([{"role": "user", "content": "U"}]) == "You are a helpful assistant."


def test_last_user_text():
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "second"},
    ]
    assert _last_user_text(msgs) == "second"


def test_last_user_text_blocks():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert _last_user_text(msgs) == "hi"


def test_trailing_tool_results_none_when_ending_with_user():
    msgs = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "tool_calls": []},
        {"role": "user", "content": "y"},
    ]
    assert _trailing_tool_results(msgs) is None


def test_trailing_tool_results_single():
    msgs = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "tool_calls": [{"id": "call_1"}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "result-a"},
    ]
    assert _trailing_tool_results(msgs) == {"call_1": "result-a"}


def test_trailing_tool_results_multiple_parallel():
    msgs = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "tool_calls": [{"id": "call_1"}, {"id": "call_2"}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "a"},
        {"role": "tool", "tool_call_id": "call_2", "content": "b"},
    ]
    assert _trailing_tool_results(msgs) == {"call_1": "a", "call_2": "b"}


def test_trailing_tool_results_ignores_missing_id():
    msgs = [
        {"role": "user", "content": "x"},
        {"role": "tool", "tool_call_id": "", "content": "bad"},
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
    ]
    assert _trailing_tool_results(msgs) == {"call_1": "ok"}


def test_format_history_skips_system_and_labels_roles():
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "again"},
    ]
    out = _format_history_as_prompt(msgs)
    assert "[user]\nhi" in out
    assert "[assistant]\nhello" in out
    assert "[user]\nagain" in out
    assert "S" not in out


def test_format_history_skips_empty_assistant():
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": ""}]
    out = _format_history_as_prompt(msgs)
    assert out == "[user]\nhi"
