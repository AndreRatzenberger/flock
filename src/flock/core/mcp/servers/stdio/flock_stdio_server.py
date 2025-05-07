from abc import ABC
from pathlib import Path
from typing import Literal
from opentelemetry import trace
from mcp import StdioServerParameters
from mcp.client.stdio import get_default_environment
from pydantic import Field

from flock.core.flock_server import FlockMCPServerBase, FlockMCPServerConfig
from flock.core.logging.logging import get_logger
from flock.core.mcp.flock_mcp_connection_manager_base import FlockMCPConnectionManagerBase
from flock.core.serialization.serializable import Serializable


logger = get_logger("stdio_mcp_server")
tracer = trace.get_tracer(__name__)


def get_default_env() -> dict[str, str]:
    """
    Returns a default environment object 
    including only environment-variables
    deemed safe to inherit.
    """
    return get_default_environment()


class FlockMCPStdioServerConfig(FlockMCPServerConfig):
    """
    Configuration Class
    for Flock MCP Servers using the stdio transport 
    protocol.
    """

    command: str = Field(
        ...,
        description="The executable to run to start the server."
    )

    args: list[str] = Field(
        ...,
        default_factory=list,
        description="Command line arguments to pass to the executable."
    )

    env: dict[str, str] | None = Field(
        default=None,
        default_factory=get_default_env,
        description="The environment to use when spawning the process. If not specified, the result of get_default_environment() will be used."
    )

    cwd: str | Path | None = Field(
        default=None,
        description="The working directory to use when spawning the process."
    )

    encoding: Literal["ascii", "utf-8", "utf-16", "utf-32"] = Field(
        default="utf-8",
        description="The text encoding used when sending/receiving a message between client and server."
    )

    encoding_error_handler: Literal["strict", "ignore", "replace"] = Field(
        default="strict",
        description="The text encoding error handler.",
    )

    max_restart_attempts: int = Field(
        default=3,
        description="How many times to attempt to restart the server before giving up."
    )


class FlockMCPStdioServer(FlockMCPServerBase, Serializable):
    """
    Class which represents a MCP Server using the Stdio 
    Transport type.

    This means (most likely) that the server is a locally
    executed script.
    """

    config: FlockMCPStdioServerConfig = Field(
        ...,
        description="Config for the server."
    )

    async def initialize(self):
        """
        Called when initializing the server
        """
        async with self.condition:
            # Check if we already initialized
            if not self.initialized:
                # Initialize the underlying Connection Pool
                if not self.connection_manager:
                    self.connection_manager = FlockMCPConnectionManagerBase(
                        transport_type="stdio",
                        server_name=self.config.server_name,
                        min_connections=1,
                        max_connections=1,
                        max_reconnect_attemtps=self.config.max_restart_attempts,
                        original_roots=self.config.mount_points,
                    )
