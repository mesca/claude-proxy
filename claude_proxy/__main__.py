"""Entry point: starts the LiteLLM proxy with the bundled config."""

import argparse
import os
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    from claude_proxy import __version__

    parser = argparse.ArgumentParser(
        prog="claude-proxy",
        description="OpenAI-compatible proxy for the Claude CLI.",
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Listen address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=4000, help="Listen port (default: 4000)",
    )
    parser.add_argument(
        "--session-header", metavar="HEADER",
        help="HTTP header for session affinity (default: auto-discover)",
    )
    parser.add_argument(
        "--stateless", action="store_true",
        help="Disable sessions — full history sent each turn",
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("list-models", help="Show available models and effort variants")
    return parser


def _list_models() -> None:
    """Print available models and effort variants."""
    from claude_proxy.models import EFFORTS, MODELS

    for model in MODELS:
        for effort in EFFORTS:
            suffix = effort["suffix"]
            name = f"{model['name']}-{suffix}" if suffix else model["name"]
            flags = f"--model {model['alias']}"
            if effort["effort"]:
                flags += f" --effort {effort['effort']}"
            print(f"  {name:<30} {flags}")  # noqa: T201


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "list-models":
        _list_models()
        return

    # Store our flags as env vars (read by middleware/handler)
    if args.stateless:
        os.environ["CLAUDE_PROXY_STATELESS"] = "1"
    if args.session_header:
        os.environ["CLAUDE_PROXY_SESSION_HEADER"] = args.session_header

    # Skip the remote model cost map fetch (adds seconds to startup)
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "true")

    # Generate config at startup from model definitions (always in sync).
    # Written next to handler.py so LiteLLM can resolve the custom_handler import.
    from claude_proxy.models import generate_config

    config_path = Path(__file__).resolve().parent / "config.yaml"
    config_path.write_text(generate_config())

    # Build LiteLLM CLI args
    sys.argv = [
        "litellm",
        "--config", str(config_path),
        "--host", args.host,
        "--port", str(args.port),
    ]

    # Add middleware + mount MCP bridge
    from litellm.proxy.proxy_server import app  # type: ignore[import-untyped]

    from claude_proxy.bridge import router as bridge_router
    from claude_proxy.middleware import (
        ReasoningContentMiddleware,
        RequestContextMiddleware,
    )

    app.include_router(bridge_router)
    app.add_middleware(ReasoningContentMiddleware)
    app.add_middleware(RequestContextMiddleware)

    # Published to the pool via env var (lazy init: first request creates the pool)
    os.environ["CLAUDE_PROXY_BRIDGE_URL"] = f"http://{args.host}:{args.port}/_mcp"

    from litellm.proxy.proxy_cli import run_server  # type: ignore[import-untyped]

    # Suppress LiteLLM's verbose startup banner (hardcoded print statements)
    _real_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")  # noqa: SIM115, PTH123
    try:
        run_server()
    finally:
        sys.stdout.close()
        sys.stdout = _real_stdout


if __name__ == "__main__":
    main()
