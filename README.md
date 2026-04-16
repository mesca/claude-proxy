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

```bash
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
    "model": "claude-sonnet-4-6-max",
    "messages": [{"role": "user", "content": "What is 99*101?"}],
    "stream": true
  }'
```

## Tool support

The proxy supports the OpenAI tool calling protocol. All tool execution happens on the **client side** (e.g. OpenCode) — the proxy itself does not execute tools.

When Claude responds with a `{"tool_calls": ...}` JSON object, the proxy's ASGI middleware rewrites it into a proper OpenAI `tool_calls` response with `finish_reason: "tool_calls"`. The client executes the tools and sends results back as `tool` role messages. No special configuration required.

## System prompt

The client's `system` message replaces Claude's default system prompt via `--system-prompt`. This gives Claude a clean slate: no built-in tool descriptions, no CLAUDE.md, no project context — only what the client sends. When no `system` message is present, a generic fallback is used.

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

[Serena](https://github.com/oraios/serena) provides code intelligence tools (find symbols, read definitions, navigate references). Add a `mcp` section to your `opencode.json`:

```json
  "mcp": {
    "serena": {
      "type": "local",
      "command": ["uvx", "--from", "serena-agent", "serena", "start-mcp-server", "--project-from-cwd"]
    }
  }
```

OpenCode sends Serena's tools to the proxy, Claude calls them via `tool_calls`, and OpenCode executes them locally via Serena.

**Test prompt**: `Show me the body of the main function and list all symbols in the entry point file.`

## CLI flags

Every request to the Claude CLI includes these flags:

| Flag                       | Purpose                                       |
|----------------------------|-----------------------------------------------|
| `--tools ""`               | Remove built-in tool descriptions from prompt |
| `--allowedTools ""`        | Block execution of any remaining tools        |
| `--disable-slash-commands` | Disable all skills                            |
| `--strict-mcp-config`     | Disable all MCP servers                        |

Claude acts as a pure LLM — all tool use is handled by the client.

## Development

```bash
git clone https://github.com/mesca/claude-proxy.git
cd claude-proxy
uv sync --all-groups
uv run pytest              # run tests
uv run ruff check .        # lint
```
