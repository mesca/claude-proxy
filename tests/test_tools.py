"""Tests for the OpenAI tool protocol support."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import litellm

from claude_proxy.cli import build_command
from claude_proxy.handler import (
    ClaudeProxyHandler,
    _build_tool_system_prompt,
    _format_history,
    _format_tool_results,
    _get_tools,
    _is_tool_result_turn,
    _parse_tool_response,
)
from claude_proxy.middleware import _parse_tool_calls_json
from claude_proxy.models import resolve_model_name

MODEL = "claude-proxy/default"
FAKE_SESSION_ID = "tool-session-1234"

SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from disk",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
]


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
# _build_tool_system_prompt
# ---------------------------------------------------------------------------


class TestBuildToolSystemPrompt:
    def test_includes_tool_definitions(self):
        prompt = _build_tool_system_prompt(None, SAMPLE_TOOLS)
        assert "read_file" in prompt
        assert "write_file" in prompt
        assert "Read a file from disk" in prompt

    def test_includes_client_system(self):
        prompt = _build_tool_system_prompt("You are a coding assistant.", SAMPLE_TOOLS)
        assert prompt.startswith("You are a coding assistant.")
        assert "read_file" in prompt

    def test_no_client_system(self):
        prompt = _build_tool_system_prompt(None, SAMPLE_TOOLS)
        assert prompt.startswith("You have access")

    def test_includes_parameters(self):
        prompt = _build_tool_system_prompt(None, SAMPLE_TOOLS)
        assert '"path"' in prompt
        assert '"type": "string"' in prompt

    def test_includes_format_instructions(self):
        prompt = _build_tool_system_prompt(None, SAMPLE_TOOLS)
        assert "tool_calls" in prompt
        assert "<tool_result" in prompt

    def test_skips_non_function_tools(self):
        tools = [{"type": "code_interpreter"}, SAMPLE_TOOLS[0]]
        prompt = _build_tool_system_prompt(None, tools)
        assert "read_file" in prompt
        assert "code_interpreter" not in prompt


# ---------------------------------------------------------------------------
# _is_tool_result_turn
# ---------------------------------------------------------------------------


class TestIsToolResultTurn:
    def test_user_only(self):
        msgs = [{"role": "user", "content": "Hi"}]
        assert _is_tool_result_turn(msgs) is False

    def test_tool_result(self):
        msgs = [
            {"role": "user", "content": "Read foo.py"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "tc_1"}]},
            {"role": "tool", "tool_call_id": "tc_1", "content": "file contents"},
        ]
        assert _is_tool_result_turn(msgs) is True

    def test_tool_result_with_trailing_user(self):
        msgs = [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "tc_1"}]},
            {"role": "tool", "tool_call_id": "tc_1", "content": "file contents"},
            {"role": "user", "content": "Now explain it"},
        ]
        # Ends with user message, not a tool result turn
        assert _is_tool_result_turn(msgs) is False

    def test_no_tool_messages(self):
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "More"},
        ]
        assert _is_tool_result_turn(msgs) is False


# ---------------------------------------------------------------------------
# _format_tool_results
# ---------------------------------------------------------------------------


class TestFormatToolResults:
    def test_starts_with_framing(self):
        msgs = [
            {"role": "assistant", "content": None},
            {"role": "tool", "tool_call_id": "tc_1", "name": "read_file", "content": "x"},
        ]
        result = _format_tool_results(msgs)
        assert result.startswith("[SYSTEM NOTE:")
        assert "tool calls YOU made" in result

    def test_single_result(self):
        msgs = [
            {"role": "assistant", "content": None},
            {"role": "tool", "tool_call_id": "tc_1", "name": "read_file", "content": "print('hello')"},
        ]
        result = _format_tool_results(msgs)
        assert '<tool_result name="read_file" call_id="tc_1">' in result
        assert "print('hello')" in result
        assert "</tool_result>" in result

    def test_multiple_results(self):
        msgs = [
            {"role": "assistant", "content": None},
            {"role": "tool", "tool_call_id": "tc_1", "name": "read_file", "content": "code"},
            {"role": "tool", "tool_call_id": "tc_2", "name": "list_dir", "content": "a.py b.py"},
        ]
        result = _format_tool_results(msgs)
        assert "read_file" in result
        assert "list_dir" in result
        assert "tc_1" in result
        assert "tc_2" in result

    def test_trailing_user_message(self):
        msgs = [
            {"role": "assistant", "content": None},
            {"role": "tool", "tool_call_id": "tc_1", "name": "read_file", "content": "code"},
            {"role": "user", "content": "Explain this"},
        ]
        result = _format_tool_results(msgs)
        assert '<tool_result name="read_file"' in result
        assert "Explain this" in result

    def test_results_in_correct_order(self):
        msgs = [
            {"role": "assistant", "content": None},
            {"role": "tool", "tool_call_id": "tc_1", "name": "first", "content": "1"},
            {"role": "tool", "tool_call_id": "tc_2", "name": "second", "content": "2"},
        ]
        result = _format_tool_results(msgs)
        assert result.index("first") < result.index("second")


# ---------------------------------------------------------------------------
# _parse_tool_response
# ---------------------------------------------------------------------------


class TestParseToolResponse:
    def test_valid_tool_calls(self):
        response = json.dumps({
            "tool_calls": [
                {"id": "call_1", "name": "read_file", "arguments": {"path": "foo.py"}}
            ]
        })
        result = _parse_tool_response(response)
        assert result is not None
        assert len(result) == 1
        assert result[0]["name"] == "read_file"
        assert result[0]["arguments"]["path"] == "foo.py"

    def test_multiple_tool_calls(self):
        response = json.dumps({
            "tool_calls": [
                {"id": "c1", "name": "read_file", "arguments": {"path": "a.py"}},
                {"id": "c2", "name": "read_file", "arguments": {"path": "b.py"}},
            ]
        })
        result = _parse_tool_response(response)
        assert result is not None
        assert len(result) == 2

    def test_plain_text(self):
        assert _parse_tool_response("Here's the file content...") is None

    def test_empty_string(self):
        assert _parse_tool_response("") is None

    def test_json_without_tool_calls(self):
        assert _parse_tool_response('{"type": "text", "text": "hello"}') is None

    def test_auto_generates_ids(self):
        response = json.dumps({
            "tool_calls": [{"name": "read_file", "arguments": {"path": "foo.py"}}]
        })
        result = _parse_tool_response(response)
        assert result is not None
        assert result[0]["id"].startswith("call_")

    def test_normalizes_ids_to_call_prefix(self):
        response = json.dumps({
            "tool_calls": [{"id": "my_id", "name": "test", "arguments": {}}]
        })
        result = _parse_tool_response(response)
        assert result is not None
        assert result[0]["id"] == "call_my_id"

    def test_preserves_call_prefix_ids(self):
        response = json.dumps({
            "tool_calls": [{"id": "call_abc123", "name": "test", "arguments": {}}]
        })
        result = _parse_tool_response(response)
        assert result is not None
        assert result[0]["id"] == "call_abc123"

    def test_whitespace_around_json(self):
        response = '  \n{"tool_calls": [{"name": "test", "arguments": {}}]}\n  '
        result = _parse_tool_response(response)
        assert result is not None

    def test_invalid_json(self):
        assert _parse_tool_response("{broken json") is None

    def test_single_object_instead_of_array(self):
        response = json.dumps({
            "tool_calls": {"id": "1", "name": "bash", "arguments": {"command": "date"}}
        })
        result = _parse_tool_response(response)
        assert result is not None
        assert len(result) == 1
        assert result[0]["name"] == "bash"

    def test_empty_tool_calls_array(self):
        assert _parse_tool_response('{"tool_calls": []}') is None


# ---------------------------------------------------------------------------
# Middleware: _parse_tool_calls_json (multiple tool calls)
# ---------------------------------------------------------------------------


class TestParseToolCallsJson:
    def test_multiple_separate_objects(self):
        text = (
            '{"tool_calls": {"name": "read_file", "arguments": {"path": "a.py"}}}\n'
            '{"tool_calls": {"name": "read_file", "arguments": {"path": "b.py"}}}'
        )
        result = _parse_tool_calls_json(text)
        assert result is not None
        assert len(result) == 2
        assert result[0]["function"]["name"] == "read_file"
        assert result[1]["function"]["name"] == "read_file"
        assert result[0]["index"] == 0
        assert result[1]["index"] == 1

    def test_single_array(self):
        text = json.dumps({"tool_calls": [
            {"name": "read_file", "arguments": {"path": "a.py"}},
            {"name": "write_file", "arguments": {"path": "b.py", "content": "x"}},
        ]})
        result = _parse_tool_calls_json(text)
        assert result is not None
        assert len(result) == 2

    def test_text_between_tool_calls(self):
        text = (
            'Let me read both files.\n'
            '{"tool_calls": {"name": "read_file", "arguments": {"path": "a.py"}}}\n'
            'And also:\n'
            '{"tool_calls": {"name": "read_file", "arguments": {"path": "b.py"}}}'
        )
        result = _parse_tool_calls_json(text)
        assert result is not None
        assert len(result) == 2

    def test_no_tool_calls(self):
        assert _parse_tool_calls_json("Just a normal response.") is None


# ---------------------------------------------------------------------------
# _get_tools
# ---------------------------------------------------------------------------


class TestGetTools:
    def test_tools_present(self):
        kwargs = {"optional_params": {"tools": SAMPLE_TOOLS}}
        assert _get_tools(kwargs) is not None

    def test_no_tools(self):
        assert _get_tools({"optional_params": {}}) is None
        assert _get_tools({}) is None

    def test_empty_tools_list(self):
        assert _get_tools({"optional_params": {"tools": []}}) is None


# ---------------------------------------------------------------------------
# resolve_model_name
# ---------------------------------------------------------------------------


class TestResolveModelName:
    def test_sonnet(self):
        assert resolve_model_name("sonnet") == "claude-sonnet-4-6"

    def test_opus(self):
        assert resolve_model_name("opus") == "claude-opus-4-6"

    def test_haiku(self):
        assert resolve_model_name("haiku") == "claude-haiku-4-5"

    def test_none_defaults(self):
        assert resolve_model_name(None) == "claude-sonnet-4-6"

    def test_passthrough(self):
        assert resolve_model_name("custom-model") == "custom-model"


# ---------------------------------------------------------------------------
# build_command with system_prompt
# ---------------------------------------------------------------------------


class TestBuildCommandSystemPrompt:
    def test_with_system_prompt(self):
        cmd = build_command("hello", system_prompt="Be helpful")
        assert "--system-prompt" in cmd
        idx = cmd.index("--system-prompt")
        assert cmd[idx + 1] == "Be helpful"

    def test_without_system_prompt(self):
        cmd = build_command("hello")
        assert "--system-prompt" not in cmd


# ---------------------------------------------------------------------------
# System prompt extraction
# ---------------------------------------------------------------------------


class TestExtractSystemPrompt:
    def test_no_system_returns_fallback(self):
        from claude_proxy.handler import _extract_system_prompt

        result = _extract_system_prompt([{"role": "user", "content": "Hi"}])
        assert "helpful assistant" in result

    def test_string_content(self):
        from claude_proxy.handler import _extract_system_prompt

        msgs = [{"role": "system", "content": "Be helpful"}, {"role": "user", "content": "Hi"}]
        result = _extract_system_prompt(msgs)
        assert result.startswith("Be helpful")

    def test_empty_content_returns_fallback(self):
        from claude_proxy.handler import _extract_system_prompt

        result = _extract_system_prompt([{"role": "system", "content": ""}])
        assert "helpful assistant" in result


class TestBuildSystemPrompt:
    def test_client_system_only_without_tools(self):
        from claude_proxy.handler import _build_system_prompt

        result = _build_system_prompt([{"role": "system", "content": "Be concise."}])
        assert result == "Be concise."

    def test_fallback_without_tools(self):
        from claude_proxy.handler import _build_system_prompt

        result = _build_system_prompt([{"role": "user", "content": "Hi"}])
        assert "helpful assistant" in result
        assert "tool_calls" not in result  # no tools → no instructions

    def test_with_tools_appends_instructions_once(self):
        from claude_proxy.handler import _build_system_prompt
        from claude_proxy.middleware import tools_var

        tools_var.set([{
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run a command",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
            },
        }])
        try:
            result = _build_system_prompt([
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hi"},
            ])
            # Client system preserved
            assert result.startswith("Be helpful.")
            # Tool schemas present
            assert "bash" in result
            assert "command" in result
            assert "required" in result
            # Tool instructions present exactly once
            assert result.count("Available tools:") == 1
        finally:
            tools_var.set(None)


# ---------------------------------------------------------------------------
# Integration: tool routing through handler
# ---------------------------------------------------------------------------


class TestToolRouting:
    @staticmethod
    def _reset():
        ClaudeProxyHandler._session_id = None

    @patch("claude_proxy.cli.subprocess.run")
    def test_tool_call_response(self, mock_run):
        self._reset()
        tool_json = json.dumps({
            "tool_calls": [{"id": "call_1", "name": "read_file", "arguments": {"path": "foo.py"}}]
        })
        mock_run.return_value = _mock_subprocess_run(stdout=_fake_cli_result(tool_json))

        resp = litellm.completion(
            model=MODEL,
            messages=[{"role": "user", "content": "Read foo.py"}],
            tools=SAMPLE_TOOLS,
        )

        assert resp.choices[0].finish_reason == "tool_calls"
        tc = resp.choices[0].message.tool_calls
        assert tc is not None
        assert tc[0].function.name == "read_file"
        assert json.loads(tc[0].function.arguments) == {"path": "foo.py"}

    @patch("claude_proxy.cli.subprocess.run")
    def test_text_response_with_tools(self, mock_run):
        self._reset()
        mock_run.return_value = _mock_subprocess_run(
            stdout=_fake_cli_result("The file contains a hello world program.")
        )

        resp = litellm.completion(
            model=MODEL,
            messages=[{"role": "user", "content": "What does foo.py do?"}],
            tools=SAMPLE_TOOLS,
        )

        assert resp.choices[0].finish_reason == "stop"
        assert "hello world" in resp.choices[0].message.content

    @patch("claude_proxy.cli.subprocess.run")
    def test_tool_result_turn_formats_xml(self, mock_run):
        self._reset()
        # Simulate: first call stored session, now resuming with tool results
        ClaudeProxyHandler._session_id = FAKE_SESSION_ID
        mock_run.return_value = _mock_subprocess_run(
            stdout=_fake_cli_result("The file prints hello world.")
        )

        litellm.completion(
            model=MODEL,
            messages=[
                {"role": "user", "content": "Read foo.py"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "read_file", "arguments": '{"path":"foo.py"}'}}
                ]},
                {"role": "tool", "tool_call_id": "call_1", "name": "read_file",
                 "content": "print('hello world')"},
            ],
            tools=SAMPLE_TOOLS,
        )

        # Verify the prompt sent to CLI contains XML-formatted tool results
        cmd = mock_run.call_args[0][0]
        # Prompt is the last element, after "--"
        prompt = cmd[-1]
        assert '<tool_result name="read_file" call_id="call_1">' in prompt
        assert "print('hello world')" in prompt
        assert "</tool_result>" in prompt

    @patch("claude_proxy.cli.subprocess.run")
    def test_system_prompt_with_tools_includes_definitions(self, mock_run):
        self._reset()
        mock_run.return_value = _mock_subprocess_run(
            stdout=_fake_cli_result("Hello!")
        )

        litellm.completion(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a coding assistant."},
                {"role": "user", "content": "Hi"},
            ],
            tools=SAMPLE_TOOLS,
        )

        cmd = mock_run.call_args[0][0]
        sys_idx = cmd.index("--system-prompt") + 1
        sys_prompt = cmd[sys_idx]
        assert "You are a coding assistant." in sys_prompt
        assert "read_file" in sys_prompt
        assert "write_file" in sys_prompt

    @patch("claude_proxy.cli.subprocess.run")
    def test_system_prompt_without_tools(self, mock_run):
        self._reset()
        mock_run.return_value = _mock_subprocess_run(
            stdout=_fake_cli_result("Hello!")
        )

        litellm.completion(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Hi"},
            ],
        )

        cmd = mock_run.call_args[0][0]
        sys_idx = cmd.index("--system-prompt") + 1
        assert cmd[sys_idx].startswith("Be concise.")

    @patch("claude_proxy.cli.subprocess.run")
    def test_no_tools_routes_to_cli_path(self, mock_run):
        self._reset()
        mock_run.return_value = _mock_subprocess_run(
            stdout=_fake_cli_result("Hello from Claude!")
        )

        resp = litellm.completion(
            model=MODEL,
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert resp.choices[0].message.content == "Hello from Claude!"
        assert resp.choices[0].finish_reason == "stop"


# ---------------------------------------------------------------------------
# Stateless mode: full history formatting
# ---------------------------------------------------------------------------


class TestFormatHistory:
    def test_user_only(self):
        msgs = [{"role": "user", "content": "Hello"}]
        result = _format_history(msgs)
        assert result == "[user]\nHello"

    def test_multi_turn(self):
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ]
        result = _format_history(msgs)
        assert "[user]\nHi" in result
        assert "[assistant]\nHello!" in result
        assert "[user]\nHow are you?" in result
        # Correct order
        assert result.index("[user]\nHi") < result.index("[assistant]") < result.index("How are you?")

    def test_system_excluded(self):
        msgs = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
        ]
        result = _format_history(msgs)
        # [user] and [assistant] labels only (system filtered)
        assert "[user]\nHi" in result
        assert "[system]" not in result

    def test_tool_results_have_framing(self):
        msgs = [
            {"role": "user", "content": "Read foo.py"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": "print('x')"},
        ]
        result = _format_history(msgs)
        # Framing precedes the first tool_result in the group
        assert "[SYSTEM NOTE:" in result
        assert result.index("[SYSTEM NOTE:") < result.index("<tool_result")

    def test_framing_once_per_consecutive_group(self):
        msgs = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "x", "arguments": "{}"}},
                {"id": "c2", "type": "function", "function": {"name": "y", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": "a"},
            {"role": "tool", "tool_call_id": "c2", "content": "b"},
        ]
        result = _format_history(msgs)
        # Only one framing note for the consecutive group of two tool results
        assert result.count("[SYSTEM NOTE:") == 1

    def test_tool_calls(self):
        msgs = [
            {"role": "user", "content": "Read foo.py"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"foo.py"}'}}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "name": "read_file", "content": "print('hello')"},
            {"role": "user", "content": "Explain it"},
        ]
        result = _format_history(msgs)
        assert "[user]\nRead foo.py" in result
        assert "[assistant]" in result
        assert "read_file" in result
        assert '<tool_result name="read_file" call_id="call_1">' in result
        assert "print('hello')" in result
        assert "[user]\nExplain it" in result
