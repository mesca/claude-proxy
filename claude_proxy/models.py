"""Model definitions and config generation.

This is the single source of truth for available models and effort levels.
Edit MODELS to add new Claude models, then run `claude-proxy update-models`.
"""

from __future__ import annotations

PROVIDER = "claude-proxy"

# Claude CLI model aliases → display names
MODELS = [
    {"alias": "opus", "name": "claude-opus-4-6"},
    {"alias": "sonnet", "name": "claude-sonnet-4-6"},
    {"alias": "haiku", "name": "claude-haiku-4-5"},
]

# Effort level variants (suffix=None → default, no --effort flag)
EFFORTS = [
    {"suffix": None, "effort": None},
    {"suffix": "high", "effort": "high"},
    {"suffix": "max", "effort": "max"},
]


def generate_config() -> str:
    """Generate config.yaml content from model definitions."""
    lines = ["model_list:"]

    for model in MODELS:
        for effort in EFFORTS:
            suffix = effort["suffix"]
            effort_flag = effort["effort"]

            if suffix:
                model_name = f"{model['name']}-{suffix}"
                internal = f"{PROVIDER}/{model['alias']}:{effort_flag}"
            else:
                model_name = model["name"]
                internal = f"{PROVIDER}/{model['alias']}"

            lines.append(f'  - model_name: "{model_name}"')
            lines.append("    litellm_params:")
            lines.append(f'      model: "{internal}"')
            lines.append("")

    lines.append("litellm_settings:")
    lines.append("  custom_provider_map:")
    lines.append(f'    - provider: "{PROVIDER}"')
    lines.append("      custom_handler: handler.handler")
    lines.append("")

    return "\n".join(lines)


def parse_model_string(model: str) -> tuple[str | None, str | None]:
    """Parse a model string into (cli_model, effort).

    Examples:
        "sonnet"      → ("sonnet", None)
        "sonnet:max"  → ("sonnet", "max")
        "default"     → (None, None)
    """
    if not model or model == "default":
        return None, None

    if ":" in model:
        cli_model, effort = model.split(":", 1)
        return cli_model, effort

    return model, None
