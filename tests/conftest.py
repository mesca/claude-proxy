"""Shared test fixtures."""

import litellm
import pytest

from claude_proxy.handler import handler


@pytest.fixture(autouse=True)
def _register_handler():
    """Register the custom handler for all tests."""
    litellm.custom_provider_map = [
        {"provider": "claude-proxy", "custom_handler": handler}
    ]
    yield
    litellm.custom_provider_map = []
