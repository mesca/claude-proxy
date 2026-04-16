"""Tests for tool protocol support (middleware) and system prompt passthrough."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import litellm

from claude_proxy.cli import build_command
from claude_proxy.handler import (
    ClaudeProxyHandler,
    _extract_system_prompt,
)
from claude_proxy.middleware import _parse_tool_calls_json

MODEL = "claude-proxy/default"
FAKE_SESSION_ID = "tool-session-1234"


def _mock_subprocess_run(returncode=0, stdout="", stderr=""):
    mock = MagicMock()
    mock.returncode = returncode
    mock.stdout = stdout
    mock.stderr = stderr
    return mock


def _fake_cli_result(result_text: str, session_id: str = FAKE_SESSION_ID) -> str:
    return json.dumps({
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": result_text,
        "session_id": session_id,
        "usage": {"input_tokens": 10, "output_tokens": 20},
    })


# ---------------------------------------------------------------------------
# Middleware: _parse_tool_calls_json
# ---------------------------------------------------------------------------


class TestParseToolCallsJson:
    def test_valid_array(self):
        text = json.dumps({"tool_calls": [{"id": "1", "name": "bash", "arguments": {"command": "date"}}]})
        result = _parse_tool_calls_json(text)
        assert result is not None
        assert len(result) == 1
        assert result[0]["function"]["name"] == "bash"
        assert result[0]["id"] == "1"

    def test_single_object(self):
        text = json.dumps({"tool_calls": {"id": "1", "name": "bash", "arguments": {"command": "date"}}})
        result = _parse_tool_calls_json(text)
        assert result is not None
        assert len(result) == 1
        assert result[0]["function"]["name"] == "bash"

    def test_trailing_text(self):
        text = '{"tool_calls": [{"name": "bash", "arguments": {}}]}\n\n<tool_result>output</tool_result>\nDone.'
        result = _parse_tool_calls_json(text)
        assert result is not None
        assert result[0]["function"]["name"] == "bash"

    def test_text_before_json(self):
        text = 'Let me check that.\n{"tool_calls": [{"name": "bash", "arguments": {}}]}'
        result = _parse_tool_calls_json(text)
        assert result is not None

    def test_plain_text(self):
        assert _parse_tool_calls_json("Hello world") is None

    def test_empty_string(self):
        assert _parse_tool_calls_json("") is None

    def test_json_without_tool_calls(self):
        assert _parse_tool_calls_json('{"type": "text"}') is None

    def test_empty_array(self):
        assert _parse_tool_calls_json('{"tool_calls": []}') is None

    def test_invalid_json(self):
        assert _parse_tool_calls_json("{broken") is None

    def test_auto_generates_ids(self):
        text = json.dumps({"tool_calls": [{"name": "bash", "arguments": {}}]})
        result = _parse_tool_calls_json(text)
        assert result is not None
        assert result[0]["id"].startswith("call_")

    def test_preserves_existing_ids(self):
        text = json.dumps({"tool_calls": [{"id": "my_id", "name": "test", "arguments": {}}]})
        result = _parse_tool_calls_json(text)
        assert result is not None
        assert result[0]["id"] == "my_id"

    def test_normalizes_to_openai_format(self):
        text = json.dumps({"tool_calls": [{"id": "1", "name": "read_file", "arguments": {"path": "foo.py"}}]})
        result = _parse_tool_calls_json(text)
        assert result is not None
        tc = result[0]
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "read_file"
        assert json.loads(tc["function"]["arguments"]) == {"path": "foo.py"}


# ---------------------------------------------------------------------------
# build_command with system_prompt
# ---------------------------------------------------------------------------


class TestBuildCommandSystemPrompt:
    def test_replaces_by_default(self):
        cmd = build_command("hello", system_prompt="Be helpful")
        assert "--system-prompt" in cmd
        idx = cmd.index("--system-prompt")
        assert cmd[idx + 1] == "Be helpful"
        assert "--append-system-prompt" not in cmd

    def test_append_mode(self):
        cmd = build_command("hello", system_prompt="Be helpful", append_system_prompt=True)
        assert "--append-system-prompt" in cmd
        idx = cmd.index("--append-system-prompt")
        assert cmd[idx + 1] == "Be helpful"
        assert "--system-prompt" not in cmd

    def test_without_system_prompt(self):
        cmd = build_command("hello")
        assert "--system-prompt" not in cmd
        assert "--append-system-prompt" not in cmd


# ---------------------------------------------------------------------------
# System prompt extraction
# ---------------------------------------------------------------------------


class TestExtractSystemPrompt:
    def test_no_system_returns_default_with_instructions(self):
        result = _extract_system_prompt([{"role": "user", "content": "Hi"}])
        assert "helpful assistant" in result
        assert "tool_calls" in result

    def test_string_content_with_instructions(self):
        msgs = [{"role": "system", "content": "Be helpful"}, {"role": "user", "content": "Hi"}]
        result = _extract_system_prompt(msgs)
        assert result.startswith("Be helpful")
        assert "tool_calls" in result

    def test_empty_content_returns_default(self):
        result = _extract_system_prompt([{"role": "system", "content": ""}])
        assert "helpful assistant" in result
        assert "tool_calls" in result


# ---------------------------------------------------------------------------
# Integration: system prompt passthrough
# ---------------------------------------------------------------------------


class TestSystemPromptPassthrough:
    @staticmethod
    def _reset():
        ClaudeProxyHandler._session_id = None

    @patch("claude_proxy.cli.subprocess.run")
    def test_system_prompt_passed_to_cli(self, mock_run):
        self._reset()
        mock_run.return_value = _mock_subprocess_run(stdout=_fake_cli_result("Hello!"))

        litellm.completion(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Hi"},
            ],
        )

        cmd = mock_run.call_args[0][0]
        assert "--system-prompt" in cmd
        idx = cmd.index("--system-prompt")
        sys_prompt = cmd[idx + 1]
        assert sys_prompt.startswith("Be concise.")
        assert "tool_calls" in sys_prompt

    @patch("claude_proxy.cli.subprocess.run")
    def test_no_system_message_uses_default(self, mock_run):
        self._reset()
        mock_run.return_value = _mock_subprocess_run(stdout=_fake_cli_result("Hello!"))

        litellm.completion(
            model=MODEL,
            messages=[{"role": "user", "content": "Hi"}],
        )

        cmd = mock_run.call_args[0][0]
        # Always replaces default system prompt (even without client system message)
        assert "--system-prompt" in cmd
