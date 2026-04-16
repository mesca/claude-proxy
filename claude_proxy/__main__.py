"""Entry point: starts the LiteLLM proxy with the bundled config."""

import argparse
import os
import sys
from pathlib import Path


def _config_path() -> Path:
    return Path(__file__).resolve().parent / "config.yaml"


def _update_models() -> None:
    """Regenerate config.yaml from model definitions."""
    from claude_proxy.models import generate_config

    path = _config_path()
    path.write_text(generate_config())
    print(f"Updated {path}")  # noqa: T201


def _build_parser() -> argparse.ArgumentParser:
    from claude_proxy import __version__

    parser = argparse.ArgumentParser(
        prog="claude-proxy",
        description="OpenAI-compatible proxy for the Claude CLI.",
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s {__version__}",
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("update-models", help="Regenerate config.yaml from model definitions")

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
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "update-models":
        _update_models()
        return

    # Store our flags as env vars (read by middleware/handler)
    if args.stateless:
        os.environ["CLAUDE_PROXY_STATELESS"] = "1"
    if args.session_header:
        os.environ["CLAUDE_PROXY_SESSION_HEADER"] = args.session_header

    # Skip the remote model cost map fetch (adds seconds to startup)
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "true")

    config_path = _config_path()
    if not config_path.exists():
        print("Config not found. Run: claude-proxy update-models", file=sys.stderr)  # noqa: T201
        sys.exit(1)

    # Build LiteLLM CLI args
    sys.argv = [
        "litellm",
        "--config", str(config_path),
        "--host", args.host,
        "--port", str(args.port),
    ]

    # Add middleware
    from litellm.proxy.proxy_server import app  # type: ignore[import-untyped]

    from claude_proxy.middleware import ReasoningContentMiddleware, ToolCallsMiddleware

    app.add_middleware(ReasoningContentMiddleware)
    app.add_middleware(ToolCallsMiddleware)

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
