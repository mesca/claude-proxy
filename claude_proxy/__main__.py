"""Entry point: starts the LiteLLM proxy with the bundled config."""

import argparse
import os
import sys
import tempfile


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
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Store our flags as env vars (read by middleware/handler)
    if args.stateless:
        os.environ["CLAUDE_PROXY_STATELESS"] = "1"
    if args.session_header:
        os.environ["CLAUDE_PROXY_SESSION_HEADER"] = args.session_header

    # Skip the remote model cost map fetch (adds seconds to startup)
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "true")

    # Generate config at startup from model definitions (always in sync)
    from claude_proxy.models import generate_config

    config_file = tempfile.NamedTemporaryFile(  # noqa: SIM115
        mode="w", suffix=".yaml", prefix="claude-proxy-", delete=False,
    )
    config_file.write(generate_config())
    config_file.close()

    # Build LiteLLM CLI args
    sys.argv = [
        "litellm",
        "--config", config_file.name,
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
        os.unlink(config_file.name)


if __name__ == "__main__":
    main()
