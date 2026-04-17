# claude-proxy

OpenAI-compatible proxy that routes requests to the Claude CLI.

> **Disclaimer** — This project invokes the official Claude CLI as a subprocess using your own authenticated session. No credentials are intercepted or stored, no authentication mechanisms are bypassed, and no proprietary protocols are reverse-engineered. You are solely responsible for ensuring your use complies with Anthropic's [Terms of Service](https://www.anthropic.com/legal/consumer-terms) and [Acceptable Use Policy](https://www.anthropic.com/legal/aup), which may change at any time. This software is provided as-is, without warranty of any kind.

<details>
<summary><strong>Table of contents</strong></summary>

- [Prerequisites](#prerequisites)
- [Install](#install)
- [Quick start](#quick-start)
- [CLI reference](#cli-reference)
- [Models](#models)
- [Sessions](#sessions)
- [Tools](#tools)
- [System prompt](#system-prompt)
- [API usage](#api-usage)
- [OpenCode](#opencode)
- [Architecture](#architecture)
- [Limitations](#limitations)
- [Development](#development)

</details>

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) installed and authenticated

## Install

```bash
git clone https://github.com/mesca/claude-proxy.git
uv tool install ./claude-proxy
```

To update after pulling new changes:

```bash
cd claude-proxy && git pull
uv cache clean claude-proxy && uv tool install --force --no-cache .
```

## Quick start

```bash
claude-proxy
```

The proxy starts on `http://127.0.0.1:4000` and accepts standard OpenAI API requests.

Run in the background with tmux:

```bash
tmux new -d -s proxy 'claude-proxy'   # start
tmux attach -t proxy                   # logs (Ctrl+B D to detach)
tmux kill-session -t proxy             # stop
```

## CLI reference

```
claude-proxy [options] [command]
```

| Flag | Default | Description |
|---|---|---|
| `--host HOST` | `127.0.0.1` | Listen address |
| `--port PORT` | `4000` | Listen port |
| `--session-header HEADER` | auto-discover | HTTP header for session affinity |
| `--stateless` | off | Disable sessions — send full conversation history each turn |
| `-v, --version` | | Show version and exit |
| `-h, --help` | | Show help and exit |

| Command | Description |
|---|---|
| `list-models` | Show available models with their Claude CLI flags |

```bash
claude-proxy                                # default: 127.0.0.1:4000
claude-proxy --host 0.0.0.0 --port 8080    # custom host and port
claude-proxy --stateless                    # no session affinity, full history each turn
claude-proxy --session-header x-session-id  # custom session header
claude-proxy list-models                    # show models
```

## Models

| Model name | Effort |
|---|---|
| `claude-opus-4-6` | default |
| `claude-opus-4-6-high` | high |
| `claude-opus-4-6-max` | max (extended thinking) |
| `claude-sonnet-4-6` | default |
| `claude-sonnet-4-6-high` | high |
| `claude-sonnet-4-6-max` | max (extended thinking) |
| `claude-haiku-4-5` | default |
| `claude-haiku-4-5-high` | high |
| `claude-haiku-4-5-max` | max (extended thinking) |

Extended thinking content is sent as `reasoning_content` in SSE chunks.

## Sessions

The proxy maintains one long-lived Claude CLI subprocess per session for conversation continuity, low latency, and prompt caching (up to 90% cost savings on repeated context).

A session header is auto-discovered from each request. A deterministic UUID is derived from the combination of the header value and the request's system prompt (`sha256`) and used as the subprocess's `--session-id`. User messages are streamed into the subprocess's stdin; assistant output streams back on stdout. Idle subprocesses are reaped after 15 minutes of inactivity.

Concurrent requests sharing the same session header are serialized: the second request waits for the first to finish before running. This prevents two turns from racing on the same CLI subprocess.

Well-known headers (checked in order): `x-session-affinity` (OpenCode), `x-session-id`, `x-conversation-id`. Use `--session-header` to override. If no header is found, the proxy falls back to stateless mode with a warning.

In stateless mode (`--stateless` or when no session header is present), an ephemeral subprocess is spawned per request with the full conversation history formatted as the prompt. Context is preserved within a single HTTP request, but there is no prompt caching across turns, and **tool use is not supported** in stateless mode (tool cycles require session state across requests).

## Tools

The proxy supports the OpenAI tool calling protocol. Neither the proxy nor Claude execute tools — all tool execution happens on the **client side** (e.g. OpenCode).

Tools flow through the Claude CLI's native tool protocol via an in-process MCP bridge mounted at `/_mcp`:

- The client's `tools` list is published to the bridge
- Claude invokes tools through standard `tool_use` content blocks
- The bridge parks each call until the client returns the matching `tool` message in a subsequent request
- The result returns to Claude as a native `tool_result` block

No JSON is injected into prompts. No response parsing. Sessions are required for tool use — tool cycles span multiple HTTP requests.

## System prompt

The client's `system` message replaces Claude's default system prompt. This gives Claude a clean slate: no built-in tool descriptions, no CLAUDE.md, no project context — only what the client sends. When no `system` message is present, a generic fallback is used.

## API usage

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

## OpenCode

Add this to your `opencode.json` (project root or `~/.config/opencode/opencode.json`):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "claude-proxy/claude-sonnet-4-7",
  "small_model": "no/model",
  "share": "disabled",
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

OpenCode sends Serena's tools to the proxy, Claude calls them via `tool_calls`, and OpenCode executes them locally.

**Test prompt**: `Show me the body of the main function and list all symbols in the entry point file.`

---

## Architecture

```mermaid
flowchart TB
    Client(["Client (OpenCode, curl, ...)"])

    subgraph proxy ["claude-proxy"]
        direction TB
        MW["ASGI Middleware\n• request context extraction\n• reasoning_content flattening"]
        LiteLLM["LiteLLM Router"]
        Handler["Handler\n• session dispatch\n• event → OpenAI translation"]
        Pool["SessionPool\n• sid → Session\n• idle reaping"]
        Bridge["MCP Bridge (/_mcp)\n• tools/list, tools/call\n• parks calls in Session futures"]
        CLI["claude -p --input-format stream-json\n--session-id ‹uuid› --mcp-config ‹bridge›\n(one long-lived subprocess per session)"]

        MW --> LiteLLM --> Handler --> Pool --> CLI
        CLI <-.-> Bridge
        Bridge --> Handler
    end

    Claude(["Claude"])

    Client -- "OpenAI API" --> MW
    CLI -- "Anthropic API\n(CLI auth)" --> Claude

    style proxy fill:#f8f9fa,stroke:#333
    style Client fill:#e3f2fd,stroke:#1565c0
    style Claude fill:#fce4ec,stroke:#c62828
```

Claude is sandboxed as a pure LLM — all built-in tools, skills and user MCP servers are disabled; the only MCP server exposed is the proxy's own bridge:

| Flag | Purpose |
|---|---|
| `--tools ""` | Remove built-in tool descriptions from the API tools list |
| `--allowedTools "mcp__proxy"` | Only the proxy bridge's tools may execute |
| `--disable-slash-commands` | Disable all skills |
| `--strict-mcp-config` | Ignore all user MCP configs; only `--mcp-config` applies |
| `--mcp-config` | Register the in-process bridge at `/_mcp` |
| `--input-format stream-json` | Feed structured user/tool messages over stdin |
| `--session-id` | Persistent session UUID across turns in the same process |
| `--system-prompt` | Replace default system prompt with client's |

## Limitations

### Session identity includes the system prompt

The internal CLI session UUID is derived from two inputs: the session header value and the exact bytes of the `system` message. Two requests sharing a header but differing on the system prompt map to different CLI sessions with different on-disk conversation histories.

**Consequences:**

- **Model / effort / tools change, same system prompt** — same session, respawn with `--resume`. History preserved, prompt cache still warm after the respawn.
- **System prompt change** — new session, empty history, new on-disk file, cold prompt cache. Useful when it's semantically a different conversation (e.g. OpenCode's title-gen has a different system prompt from the main chat and must not contaminate it). Undesirable when the client regenerates the system prompt on every turn (e.g. embeds a timestamp or a rotating tool list) — you'll see a fresh `Spawning session …` log line every turn and lose all caching benefits.
- **Adding / removing an MCP server in the client** often forces the client to regenerate its system prompt (the new tools usually need to be described). In that case the old conversation does not carry over to the new tool set. If this matters, edit the client's system prompt to be stable across tool-set changes, or expect a fresh session on every toolchange.

### Concurrent requests on the same session header are serialized

A per-session lock in the middleware ensures only one HTTP request is in flight per session header at a time. The second request on the same header waits for the first to finish before running.

**Consequences:**

- Safe: no `--resume` races, no mid-stream subprocess kills, no tool-cycle contamination.
- Trade-off: if a client fires parallel requests on the same header expecting them to run concurrently, they will run sequentially instead. OpenCode in particular fires a background "title-gen" request alongside the main chat with the same `x-session-affinity` — the two run one after the other, not in parallel. The latency of the main chat is unaffected because the title-gen is usually fast, but throughput on heavy parallel workloads is capped at 1×.
- Workaround for clients that need true parallelism: use distinct session header values (e.g. `x-session-affinity: conv-42-primary` and `conv-42-titlegen`) so each lands in a different pool slot.

## Development

```bash
git clone https://github.com/mesca/claude-proxy.git
cd claude-proxy
uv sync --all-groups
uv run pytest              # run tests
uv run ruff check .        # lint
```

### Adding models

When new Claude models become available, edit `MODELS` in `claude_proxy/models.py`:

```python
MODELS = [
    {"alias": "opus", "name": "claude-opus-4-6"},
    {"alias": "sonnet", "name": "claude-sonnet-4-6"},
    {"alias": "haiku", "name": "claude-haiku-4-5"},
]
```

The `alias` is the CLI `--model` value. The `name` is the model ID exposed to clients. Effort variants (`-high`, `-max`) are generated automatically. Reinstall after editing.

Reference: [Anthropic model IDs](https://docs.anthropic.com/en/docs/about-claude/models)
