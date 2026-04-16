"""Tests for the Claude CLI backend."""

from __future__ import annotations

import io
import json
import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import pytest
from litellm.exceptions import APIConnectionError
from loguru import logger

from claude_proxy.cli import (
    ClaudeCliError,
    _extract_stream_event,
    _StreamState,
    build_command,
)
from claude_proxy.handler import (
    ClaudeProxyHandler,
    _extract_prompt,
    _get_cwd,
    _get_model_and_effort,
    _get_session_id,
    _is_new_conversation,
    handler,
)
from claude_proxy.models import generate_config, parse_model_string

MODEL = "claude-proxy/default"
FAKE_SESSION_ID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

FAKE_CLI_RESULT = json.dumps({
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "result": "Hello from Claude!",
    "session_id": FAKE_SESSION_ID,
    "usage": {"input_tokens": 10, "output_tokens": 20},
})

FAKE_STREAM_LINES = [
    json.dumps({"type": "system", "subtype": "init", "session_id": FAKE_SESSION_ID}),
    json.dumps({
        "type": "assistant",
        "message": {"content": [{"type": "text", "text": "Hello"}]},
    }),
    json.dumps({
        "type": "assistant",
        "message": {"content": [{"type": "text", "text": "Hello from Claude!"}]},
    }),
    json.dumps({
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": "Hello from Claude!",
        "session_id": FAKE_SESSION_ID,
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }),
]

FAKE_STREAM_WITH_THINKING = [
    json.dumps({"type": "system", "subtype": "init", "session_id": FAKE_SESSION_ID}),
    json.dumps({
        "type": "assistant",
        "message": {"content": [{"type": "thinking", "thinking": "Let me think"}]},
    }),
    json.dumps({
        "type": "assistant",
        "message": {"content": [
            {"type": "thinking", "thinking": "Let me think about this"},
            {"type": "text", "text": "The answer"},
        ]},
    }),
    json.dumps({
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": "The answer is 42",
        "session_id": FAKE_SESSION_ID,
        "usage": {"input_tokens": 10, "output_tokens": 30},
    }),
]


# --- Pure function tests ---


class TestBuildCommand:
    def test_basic(self):
        cmd = build_command("hello")
        assert cmd == [
            "claude", "-p", "hello",
            "--tools", "", "--allowedTools", "",
            "--disable-slash-commands", "--strict-mcp-config",
            "--dangerously-skip-permissions",
            "--output-format", "json",
        ]

    def test_streaming(self):
        cmd = build_command("hello", streaming=True)
        assert cmd[cmd.index("--output-format") + 1] == "stream-json"
        assert "--verbose" in cmd

    def test_with_session_id(self):
        cmd = build_command("hello", session_id="abc-123")
        assert cmd[cmd.index("--resume") + 1] == "abc-123"

    def test_with_model(self):
        cmd = build_command("hello", model="opus")
        assert cmd[cmd.index("--model") + 1] == "opus"

    def test_no_session_id(self):
        assert "--resume" not in build_command("hello")

    def test_no_model(self):
        assert "--model" not in build_command("hello")

    def test_sandbox_flags(self):
        cmd = build_command("hello")
        assert "--tools" in cmd
        assert cmd[cmd.index("--tools") + 1] == ""
        assert "--allowedTools" in cmd
        assert cmd[cmd.index("--allowedTools") + 1] == ""
        assert "--disable-slash-commands" in cmd
        assert "--strict-mcp-config" in cmd
        assert "--dangerously-skip-permissions" in cmd


class TestExtractPrompt:
    def test_single_user_message(self):
        assert _extract_prompt([{"role": "user", "content": "Hello"}]) == "Hello"

    def test_last_user_message(self):
        messages = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Response"},
            {"role": "user", "content": "Second"},
        ]
        assert _extract_prompt(messages) == "Second"

    def test_with_system_message(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        assert _extract_prompt(messages) == "Hello"

    def test_content_blocks(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello world"}]}]
        assert _extract_prompt(messages) == "Hello world"

    def test_no_user_message_raises(self):
        with pytest.raises(ClaudeCliError, match="No user message"):
            _extract_prompt([{"role": "system", "content": "System only"}])

    def test_empty_messages_raises(self):
        with pytest.raises(ClaudeCliError, match="No user message"):
            _extract_prompt([])


class TestIsNewConversation:
    def test_single_user_message(self):
        assert _is_new_conversation([{"role": "user", "content": "Hi"}]) is True

    def test_with_system_and_user(self):
        messages = [{"role": "system", "content": "..."}, {"role": "user", "content": "Hi"}]
        assert _is_new_conversation(messages) is True

    def test_with_assistant_message(self):
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "More"},
        ]
        assert _is_new_conversation(messages) is False


class TestGetSessionId:
    def test_explicit_session_id(self):
        kwargs = {"optional_params": {"session_id": "explicit-id"}}
        assert _get_session_id(kwargs, [], "stored-id") == "explicit-id"

    def test_new_conversation_returns_none(self):
        kwargs = {"optional_params": {}}
        messages = [{"role": "user", "content": "Hi"}]
        assert _get_session_id(kwargs, messages, "stored-id") is None

    def test_continued_conversation_returns_stored(self):
        kwargs = {"optional_params": {}}
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "More"},
        ]
        assert _get_session_id(kwargs, messages, "stored-id") == "stored-id"

    def test_continued_conversation_no_stored(self):
        kwargs = {"optional_params": {}}
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "More"},
        ]
        assert _get_session_id(kwargs, messages, None) is None


class TestGetCwd:
    def test_from_optional_params(self):
        kwargs = {"optional_params": {"cwd": "/tmp/project"}}
        assert _get_cwd(kwargs) == "/tmp/project"

    def test_no_cwd(self):
        result = _get_cwd({"optional_params": {}})
        assert result is None or isinstance(result, str)


class TestParseModelString:
    def test_default(self):
        assert parse_model_string("default") == (None, None)

    def test_empty(self):
        assert parse_model_string("") == (None, None)

    def test_simple_model(self):
        assert parse_model_string("sonnet") == ("sonnet", None)

    def test_model_with_effort(self):
        assert parse_model_string("sonnet:max") == ("sonnet", "max")

    def test_opus_thinking(self):
        assert parse_model_string("opus:max") == ("opus", "max")


class TestGetModelAndEffort:
    def test_default(self):
        assert _get_model_and_effort("default") == (None, None)

    def test_model_only(self):
        assert _get_model_and_effort("opus") == ("opus", None)

    def test_model_with_effort(self):
        assert _get_model_and_effort("sonnet:max") == ("sonnet", "max")


class TestBuildCommandEffort:
    def test_no_effort(self):
        cmd = build_command("hello", model="sonnet")
        assert "--effort" not in cmd

    def test_with_effort(self):
        cmd = build_command("hello", model="sonnet", effort="max")
        assert cmd[cmd.index("--effort") + 1] == "max"


class TestGenerateConfig:
    def test_generates_yaml(self):
        config = generate_config()
        assert "model_list:" in config
        assert "claude-proxy/sonnet" in config
        assert "claude-proxy/sonnet:max" in config
        assert "custom_handler: handler.handler" in config

    def test_has_effort_variants(self):
        config = generate_config()
        assert 'claude-sonnet-4-6-high' in config
        assert 'claude-sonnet-4-6-max' in config
        assert 'claude-opus-4-6-max' in config
        assert 'claude-haiku-4-5-max' in config


# --- Streaming tests ---


class TestStreamThinking:
    def test_thinking_in_separate_field(self):
        state = _StreamState()
        event = {
            "type": "assistant",
            "message": {"content": [{"type": "thinking", "thinking": "reasoning"}]},
        }
        result = _extract_stream_event(event, state)
        assert result is not None
        assert result["thinking"] == "reasoning"
        assert result["text"] == ""

    def test_text_in_text_field(self):
        state = _StreamState()
        event = {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "answer"}]},
        }
        result = _extract_stream_event(event, state)
        assert result is not None
        assert result["text"] == "answer"
        assert result["thinking"] is None

    def test_thinking_then_text(self):
        state = _StreamState()
        ev1 = {
            "type": "assistant",
            "message": {"content": [{"type": "thinking", "thinking": "step 1"}]},
        }
        r1 = _extract_stream_event(ev1, state)
        assert r1["thinking"] == "step 1"
        assert r1["text"] == ""

        ev2 = {
            "type": "assistant",
            "message": {"content": [
                {"type": "thinking", "thinking": "step 1 step 2"},
                {"type": "text", "text": "answer"},
            ]},
        }
        r2 = _extract_stream_event(ev2, state)
        assert r2["thinking"] == " step 2"  # delta
        assert r2["text"] == "answer"


# --- Mocked subprocess tests ---


def _mock_subprocess_run(returncode: int = 0, stdout: str = "", stderr: str = ""):
    mock = MagicMock(spec=subprocess.CompletedProcess)
    mock.returncode = returncode
    mock.stdout = stdout
    mock.stderr = stderr
    return mock


@pytest.fixture(autouse=True)
def _reset_session():
    """Reset stored session between tests."""
    ClaudeProxyHandler._session_id = None
    yield
    ClaudeProxyHandler._session_id = None


class TestCompletion:
    @patch("claude_proxy.cli.subprocess.run")
    def test_returns_response(self, mock_run):
        mock_run.return_value = _mock_subprocess_run(stdout=FAKE_CLI_RESULT)
        resp = litellm.completion(model=MODEL, messages=[{"role": "user", "content": "Hello"}])
        assert resp.choices[0].message.content == "Hello from Claude!"
        assert resp.system_fingerprint == FAKE_SESSION_ID

    @patch("claude_proxy.cli.subprocess.run")
    def test_usage(self, mock_run):
        mock_run.return_value = _mock_subprocess_run(stdout=FAKE_CLI_RESULT)
        resp = litellm.completion(model=MODEL, messages=[{"role": "user", "content": "Hello"}])
        assert resp.usage.prompt_tokens == 10  # type: ignore[union-attr]
        assert resp.usage.completion_tokens == 20  # type: ignore[union-attr]

    @patch("claude_proxy.cli.subprocess.run")
    def test_cwd_passed_to_subprocess(self, mock_run):
        mock_run.return_value = _mock_subprocess_run(stdout=FAKE_CLI_RESULT)
        litellm.completion(model=MODEL, messages=[{"role": "user", "content": "Hello"}], cwd="/tmp/myproject")
        assert mock_run.call_args[1]["cwd"] == "/tmp/myproject"


class TestAsyncCompletion:
    @patch("claude_proxy.cli.asyncio.create_subprocess_exec")
    async def test_returns_response(self, mock_exec):
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (FAKE_CLI_RESULT.encode(), b"")
        mock_proc.returncode = 0
        mock_exec.return_value = mock_proc
        resp = await litellm.acompletion(model=MODEL, messages=[{"role": "user", "content": "Hello"}])
        assert resp.choices[0].message.content == "Hello from Claude!"


class TestSessionAutoManagement:
    @patch("claude_proxy.cli.subprocess.run")
    def test_new_conversation_no_resume(self, mock_run):
        mock_run.return_value = _mock_subprocess_run(stdout=FAKE_CLI_RESULT)
        litellm.completion(model=MODEL, messages=[{"role": "user", "content": "Hi"}])
        cmd = mock_run.call_args[0][0]
        assert "--resume" not in cmd

    @patch("claude_proxy.cli.subprocess.run")
    def test_stores_session_id(self, mock_run):
        mock_run.return_value = _mock_subprocess_run(stdout=FAKE_CLI_RESULT)
        litellm.completion(model=MODEL, messages=[{"role": "user", "content": "Hi"}])
        assert ClaudeProxyHandler._session_id == FAKE_SESSION_ID

    @patch("claude_proxy.cli.subprocess.run")
    def test_continued_conversation_resumes(self, mock_run):
        mock_run.return_value = _mock_subprocess_run(stdout=FAKE_CLI_RESULT)
        # First request — new session
        litellm.completion(model=MODEL, messages=[{"role": "user", "content": "Hi"}])
        # Second request — has assistant message → continuation
        litellm.completion(
            model=MODEL,
            messages=[
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "More"},
            ],
        )
        cmd = mock_run.call_args[0][0]
        assert "--resume" in cmd
        assert cmd[cmd.index("--resume") + 1] == FAKE_SESSION_ID

    @patch("claude_proxy.cli.subprocess.run")
    def test_new_conversation_resets_session(self, mock_run):
        mock_run.return_value = _mock_subprocess_run(stdout=FAKE_CLI_RESULT)
        # First conversation
        litellm.completion(model=MODEL, messages=[{"role": "user", "content": "Hi"}])
        assert ClaudeProxyHandler._session_id == FAKE_SESSION_ID
        # New conversation (only user message)
        litellm.completion(model=MODEL, messages=[{"role": "user", "content": "New topic"}])
        cmd = mock_run.call_args[0][0]
        assert "--resume" not in cmd

    @patch("claude_proxy.cli.subprocess.run")
    def test_explicit_session_id_overrides(self, mock_run):
        mock_run.return_value = _mock_subprocess_run(stdout=FAKE_CLI_RESULT)
        litellm.completion(
            model=MODEL,
            messages=[{"role": "user", "content": "Hi"}],
            session_id="custom-id",
        )
        cmd = mock_run.call_args[0][0]
        assert "--resume" in cmd
        assert cmd[cmd.index("--resume") + 1] == "custom-id"


class TestStreaming:
    @patch("claude_proxy.cli.subprocess.Popen")
    def test_yields_chunks(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.stdout = iter(line + "\n" for line in FAKE_STREAM_LINES)
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read.return_value = ""
        mock_proc.wait.return_value = 0
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc

        resp = litellm.completion(model=MODEL, messages=[{"role": "user", "content": "Hello"}], stream=True)
        chunks = list(resp)
        texts = [c.choices[0].delta.content for c in chunks if c.choices[0].delta.content]
        assert "Hello" in "".join(texts)

    @patch("claude_proxy.cli.subprocess.Popen")
    def test_thinking_in_reasoning_content(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.stdout = iter(line + "\n" for line in FAKE_STREAM_WITH_THINKING)
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read.return_value = ""
        mock_proc.wait.return_value = 0
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc

        resp = litellm.completion(model=MODEL, messages=[{"role": "user", "content": "Think"}], stream=True)
        chunks = list(resp)
        # Check that reasoning_content appears at chunk top level
        assert any(hasattr(c, "reasoning_content") and c.reasoning_content for c in chunks)
        # And text content is separate
        texts = [c.choices[0].delta.content for c in chunks if c.choices[0].delta.content]
        full_text = "".join(texts)
        assert "answer" in full_text.lower()

    @patch("claude_proxy.cli.subprocess.Popen")
    def test_stores_session_from_stream(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.stdout = iter(line + "\n" for line in FAKE_STREAM_LINES)
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read.return_value = ""
        mock_proc.wait.return_value = 0
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc

        resp = litellm.completion(model=MODEL, messages=[{"role": "user", "content": "Hello"}], stream=True)
        list(resp)  # consume all chunks
        assert ClaudeProxyHandler._session_id == FAKE_SESSION_ID


class TestAsyncStreaming:
    @patch("claude_proxy.cli.asyncio.create_subprocess_exec")
    async def test_yields_chunks(self, mock_exec):
        lines = [line.encode() + b"\n" for line in FAKE_STREAM_LINES] + [b""]
        mock_proc = AsyncMock()
        mock_proc.stdout.readline = AsyncMock(side_effect=lines)
        mock_proc.returncode = 0
        mock_proc.wait = AsyncMock(return_value=0)
        mock_exec.return_value = mock_proc

        resp = await litellm.acompletion(model=MODEL, messages=[{"role": "user", "content": "Hello"}], stream=True)
        chunks = [c async for c in resp]
        texts = [c.choices[0].delta.content for c in chunks if c.choices[0].delta.content]
        assert "Hello" in "".join(texts)


class TestErrorHandling:
    @patch("claude_proxy.cli.subprocess.run")
    def test_cli_error_raises(self, mock_run):
        mock_run.return_value = _mock_subprocess_run(returncode=1, stderr="Something went wrong")
        with pytest.raises(APIConnectionError, match="Something went wrong"):
            litellm.completion(model=MODEL, messages=[{"role": "user", "content": "Hello"}])

    @patch("claude_proxy.cli.subprocess.run")
    def test_bad_json_raises(self, mock_run):
        mock_run.return_value = _mock_subprocess_run(stdout="not json")
        with pytest.raises(APIConnectionError, match="Failed to parse CLI output"):
            litellm.completion(model=MODEL, messages=[{"role": "user", "content": "Hello"}])

    @patch("claude_proxy.cli.subprocess.run", side_effect=FileNotFoundError("not found"))
    def test_cli_not_found_raises(self, mock_run):
        with pytest.raises(APIConnectionError, match="claude CLI not found"):
            litellm.completion(model=MODEL, messages=[{"role": "user", "content": "Hello"}])


class TestLogging:
    @patch("claude_proxy.cli.subprocess.run")
    def test_completion_logs_request(self, mock_run):
        mock_run.return_value = _mock_subprocess_run(stdout=FAKE_CLI_RESULT)
        sink = io.StringIO()
        handler_id = logger.add(sink, format="{message}", level="INFO")
        try:
            litellm.completion(model=MODEL, messages=[{"role": "user", "content": "test-log-message"}])
            log_output = sink.getvalue()
            assert "Request" in log_output
            assert "model=" in log_output
        finally:
            logger.remove(handler_id)


class TestHandlerInstance:
    def test_handler_is_custom_llm(self):
        assert isinstance(handler, ClaudeProxyHandler)
