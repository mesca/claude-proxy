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

| Model name          | Claude CLI `--model` |
|---------------------|----------------------|
| `claude-opus-4-6`   | `opus`               |
| `claude-sonnet-4-6` | `sonnet`             |
| `claude-haiku-4-5`  | `haiku`              |

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
    "model": "claude-sonnet-4-6",
    "messages": [{"role": "user", "content": "Hello!"}],
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

### Thinking / reasoning

When Claude uses extended thinking during streaming, the thought process is sent as `reasoning_content` in the SSE chunks — the standard format used by reasoning models in the OpenAI API.

## OpenCode configuration

Add this to your `opencode.json` (project root or `~/.config/opencode/opencode.json`):

```json
{
  "provider": "claude-proxy",
  "model": "claude-sonnet-4-6",
  "providers": {
    "claude-proxy": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Claude Proxy",
      "options": {
        "baseURL": "http://localhost:4000/v1",
        "apiKey": "not-needed"
      },
      "models": {
        "claude-opus-4-6": {
          "name": "Claude Opus 4.6"
        },
        "claude-sonnet-4-6": {
          "name": "Claude Sonnet 4.6"
        },
        "claude-haiku-4-5": {
          "name": "Claude Haiku 4.5"
        }
      }
    }
  }
}
```

## CLI flags

Every request to the Claude CLI includes these flags:

| Flag                              | Purpose                               |
|-----------------------------------|---------------------------------------|
| `--allowedTools ""`               | Disable all tools (passthrough mode)  |
| `--disable-slash-commands`        | Disable all skills                    |
| `--strict-mcp-config`            | Disable all MCP servers               |
| `--dangerously-skip-permissions` | Skip permission prompts               |

Claude acts as a pure LLM — all tool use (file reading, editing, shell commands) is handled by the client (e.g. OpenCode).

## Development

```bash
git clone https://github.com/mesca/claude-proxy.git
cd claude-proxy
uv sync --all-groups
uv run pytest              # run tests
uv run ruff check .        # lint
```
