# claude-proxy

OpenAI-compatible LLM proxy that routes requests to the Claude CLI.

Built with [LiteLLM](https://github.com/BerriAI/litellm) and a custom backend that calls `claude -p` as a subprocess.

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) installed and authenticated (`claude` command available in your PATH)

## Install

```bash
git clone https://github.com/mesca/claude-proxy.git
uv tool install ./claude-proxy
```

To update after pulling new changes:

```bash
cd claude-proxy
git pull
uv cache clean claude-proxy && uv tool install --force --no-cache .
```

Verify the installation:

```bash
claude-proxy --help
```

## Quick start

Start the proxy from your project directory:

```bash
cd /path/to/your/project
claude-proxy
```

The proxy starts on `http://localhost:4000`. All LiteLLM flags work (e.g. `claude-proxy --port 8080`).

To stop the proxy, press `Ctrl+C`. If a stale process remains, find and kill it:

```bash
lsof -i :4000
kill <PID>
```

## Available models

| Model name                   | Claude CLI flags                     |
|------------------------------|--------------------------------------|
| `claude-opus-4-6`            | `--model opus`                       |
| `claude-opus-4-6-high`       | `--model opus --effort high`         |
| `claude-opus-4-6-max`        | `--model opus --effort max`          |
| `claude-sonnet-4-6`          | `--model sonnet`                     |
| `claude-sonnet-4-6-high`     | `--model sonnet --effort high`       |
| `claude-sonnet-4-6-max`      | `--model sonnet --effort max`        |
| `claude-haiku-4-5`           | `--model haiku`                      |
| `claude-haiku-4-5-high`      | `--model haiku --effort high`        |
| `claude-haiku-4-5-max`       | `--model haiku --effort max`         |

The `-high` and `-max` variants set the `--effort` flag. Higher effort enables extended thinking — thinking content is sent as `reasoning_content` in SSE chunks.

### Updating models

When new Claude models become available, edit `claude_proxy/models.py` and regenerate the config:

```bash
claude-proxy update-models
```

This regenerates the bundled `config.yaml` from the model definitions. Reinstall after updating:

```bash
uv cache clean claude-proxy && uv tool install --force --no-cache .
```

## Usage

### Non-streaming

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-6",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Streaming

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-6:thinking",
    "messages": [{"role": "user", "content": "What is 99*101?"}],
    "stream": true
  }'
```

### Sessions

Sessions are managed automatically. The proxy detects new vs continued conversations from the message history:

- **New conversation** (no assistant messages in the array): starts a fresh Claude session
- **Continued conversation** (has assistant messages): resumes the previous session via `--resume`

For explicit control, pass `session_id` in the request body:

```json
{"model": "claude-sonnet-4-6", "messages": [...], "session_id": "<session_id>"}
```

The session ID is returned in the `system_fingerprint` field of every response.

### Working directory

The Claude CLI runs in the directory where you started `claude-proxy`. This determines the project context Claude sees.

```bash
cd /path/to/your/project
claude-proxy
```

Alternatively, set it via environment variable or per-request:

```bash
CLAUDE_PROXY_CWD=/path/to/project claude-proxy
```

```json
{"model": "claude-sonnet-4-6", "messages": [...], "cwd": "/path/to/project"}
```

## OpenCode configuration

Add this to your `opencode.json` (project root or `~/.config/opencode/opencode.json`):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "claude-proxy/claude-sonnet-4-6",
  "provider": {
    "claude-proxy": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Claude Proxy",
      "options": {
        "baseURL": "http://localhost:4000/v1",
        "apiKey": "not-needed"
      },
      "models": {
        "claude-opus-4-6": { "name": "Claude Opus 4.6" },
        "claude-opus-4-6-high": { "name": "Claude Opus 4.6 (high effort)" },
        "claude-opus-4-6-max": { "name": "Claude Opus 4.6 (max effort)" },
        "claude-sonnet-4-6": { "name": "Claude Sonnet 4.6" },
        "claude-sonnet-4-6-high": { "name": "Claude Sonnet 4.6 (high effort)" },
        "claude-sonnet-4-6-max": { "name": "Claude Sonnet 4.6 (max effort)" },
        "claude-haiku-4-5": { "name": "Claude Haiku 4.5" },
        "claude-haiku-4-5-high": { "name": "Claude Haiku 4.5 (high effort)" },
        "claude-haiku-4-5-max": { "name": "Claude Haiku 4.5 (max effort)" }
      }
    }
  }
}
```

### With Serena MCP

[Serena](https://github.com/oraios/serena) provides code intelligence tools (find symbols, read definitions, navigate references). To use Serena with OpenCode through the proxy, add a `mcp` section to your `opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "claude-proxy/claude-sonnet-4-6",
  "provider": {
    "claude-proxy": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Claude Proxy",
      "options": {
        "baseURL": "http://localhost:4000/v1",
        "apiKey": "not-needed"
      },
      "models": {
        "claude-opus-4-6": { "name": "Claude Opus 4.6" },
        "claude-opus-4-6-high": { "name": "Claude Opus 4.6 (high effort)" },
        "claude-opus-4-6-max": { "name": "Claude Opus 4.6 (max effort)" },
        "claude-sonnet-4-6": { "name": "Claude Sonnet 4.6" },
        "claude-sonnet-4-6-high": { "name": "Claude Sonnet 4.6 (high effort)" },
        "claude-sonnet-4-6-max": { "name": "Claude Sonnet 4.6 (max effort)" },
        "claude-haiku-4-5": { "name": "Claude Haiku 4.5" },
        "claude-haiku-4-5-high": { "name": "Claude Haiku 4.5 (high effort)" },
        "claude-haiku-4-5-max": { "name": "Claude Haiku 4.5 (max effort)" }
      }
    }
  },
  "mcp": {
    "serena": {
      "type": "local",
      "command": ["uvx", "--from", "serena-agent", "serena", "start-mcp-server", "--project-from-cwd"]
    }
  }
}
```

The `--project-from-cwd` flag auto-detects the project from the working directory. Start the proxy from your project directory and use OpenCode as usual. OpenCode sends Serena's tools to the proxy, Claude calls them, and OpenCode executes them locally via Serena.

**Test prompt**: Try asking OpenCode: `Find the definition of the main function and list all symbols in the entry point file.`

## Tool support

The proxy supports the OpenAI tool calling protocol. Clients send tool definitions in the `tools` field, and Claude responds with `tool_calls` when it wants to use a tool. The client executes the tool and sends results back.

### How it works

1. Client sends `tools` in the request — the proxy injects tool definitions into Claude's system prompt
2. Claude responds with a JSON `tool_calls` object — the proxy parses it and returns an OpenAI-format `tool_calls` response with `finish_reason: "tool_calls"`
3. Client executes the tools and sends results as `tool` role messages — the proxy formats them in XML tags and resumes the session
4. Claude continues with a text response (or more tool calls)

### Example: tool call request

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-6",
    "messages": [{"role": "user", "content": "Read the file hello.py"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "read_file",
        "description": "Read a file from disk",
        "parameters": {
          "type": "object",
          "properties": {"path": {"type": "string"}},
          "required": ["path"]
        }
      }
    }]
  }'
```

### Example: sending tool results

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-6",
    "messages": [
      {"role": "user", "content": "Read hello.py"},
      {"role": "assistant", "content": null, "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\":\"hello.py\"}"}}
      ]},
      {"role": "tool", "tool_call_id": "call_1", "name": "read_file", "content": "print(\"hello world\")"}
    ],
    "tools": [{
      "type": "function",
      "function": {
        "name": "read_file",
        "description": "Read a file from disk",
        "parameters": {
          "type": "object",
          "properties": {"path": {"type": "string"}},
          "required": ["path"]
        }
      }
    }]
  }'
```

### System prompt passthrough

The client's `system` message is passed to Claude via `--system-prompt`, replacing the default system prompt. This gives Claude a clean slate: no built-in tool descriptions, no CLAUDE.md, no project context — only what the client sends.

To keep Claude's default system prompt and append instead:

```bash
claude-proxy --append-system-prompt
```

## CLI flags

Every request to the Claude CLI includes these flags:

| Flag                              | Purpose                                         |
|-----------------------------------|--------------------------------------------------|
| `--tools ""`                      | Remove built-in tool descriptions from prompt    |
| `--allowedTools ""`               | Block execution of any remaining tools           |
| `--disable-slash-commands`        | Disable all skills                               |
| `--strict-mcp-config`            | Disable all MCP servers                           |
| `--dangerously-skip-permissions` | Skip permission prompts                           |

Claude acts as a pure LLM — all tool use (file reading, editing, shell commands) is handled by the client (e.g. OpenCode).

## Development

```bash
git clone https://github.com/mesca/claude-proxy.git
cd claude-proxy
uv sync --all-groups
uv run pytest              # run tests
uv run ruff check .        # lint
```
