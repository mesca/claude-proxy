"""LLM proxy using LiteLLM with a custom backend."""

try:
    from claude_proxy._version import __version__
except ModuleNotFoundError:
    __version__ = "0.0.0+unknown"
