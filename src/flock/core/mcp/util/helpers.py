"""Helper functions for Flock MCP Functionality."""

from mcp.client.stdio import get_default_environment


def get_default_env() -> dict[str, str]:
    """Returns a default environment object.

    Including only environment-variables
    deemed safe to inherit.
    """
    return get_default_environment()
