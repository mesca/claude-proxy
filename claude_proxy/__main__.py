"""Entry point: starts the LiteLLM proxy with the bundled config."""

import os
import sys
from pathlib import Path


def main() -> None:
    # Capture the user's working directory before LiteLLM changes anything
    os.environ.setdefault("CLAUDE_PROXY_CWD", os.getcwd())

    # Skip the remote model cost map fetch (adds seconds to startup)
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "true")

    config_path = Path(__file__).resolve().parent / "config.yaml"
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)  # noqa: T201
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

    from claude_proxy.middleware import ReasoningContentMiddleware

    app.add_middleware(ReasoningContentMiddleware)

    from litellm.proxy.proxy_cli import run_server  # type: ignore[import-untyped]

    run_server()


if __name__ == "__main__":
    main()
