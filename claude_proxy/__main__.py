"""Entry point: starts the LiteLLM proxy with the bundled config."""

import os
import sys
from pathlib import Path


def _config_path() -> Path:
    return Path(__file__).resolve().parent / "config.yaml"


def update_models() -> None:
    """Regenerate config.yaml from model definitions."""
    from claude_proxy.models import generate_config

    path = _config_path()
    path.write_text(generate_config())
    print(f"Updated {path}")  # noqa: T201


def main() -> None:
    # Handle subcommands before LiteLLM takes over sys.argv
    if len(sys.argv) > 1 and sys.argv[1] == "update-models":
        update_models()
        return

    # Capture the user's working directory before LiteLLM changes anything
    os.environ.setdefault("CLAUDE_PROXY_CWD", os.getcwd())

    # --append-system-prompt: append to Claude's default system prompt instead of replacing it
    if "--append-system-prompt" in sys.argv:
        os.environ["CLAUDE_PROXY_APPEND_SYSTEM_PROMPT"] = "1"
        sys.argv.remove("--append-system-prompt")

    # Skip the remote model cost map fetch (adds seconds to startup)
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "true")

    config_path = _config_path()
    if not config_path.exists():
        print("Config not found. Run: claude-proxy update-models", file=sys.stderr)  # noqa: T201
        sys.exit(1)

    # Build CLI args, injecting --config and --host defaults if not provided
    user_args = sys.argv[1:]
    injected: list[str] = []

    if "--config" not in user_args and "-c" not in user_args:
        injected.extend(["--config", str(config_path)])

    if "--host" not in user_args:
        injected.extend(["--host", "127.0.0.1"])

    sys.argv = ["litellm", *injected, *user_args]

    # Add middleware to fix reasoning_content in SSE streams
    from litellm.proxy.proxy_server import app  # type: ignore[import-untyped]

    from claude_proxy.middleware import ReasoningContentMiddleware, ToolCallsMiddleware

    app.add_middleware(ReasoningContentMiddleware)
    app.add_middleware(ToolCallsMiddleware)

    from litellm.proxy.proxy_cli import run_server  # type: ignore[import-untyped]

    run_server()


if __name__ == "__main__":
    main()
