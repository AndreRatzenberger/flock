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
from flock.core.mcp.servers.stdio.flock_stdio_connection_manager import FlockStdioMCPConnectionManager
from flock.core.serialization.serializable import Serializable

from dspy.primitives import Tool as DSPyTool


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
        description="Command line arguments to pass to the executable."
    )

    env: dict[str, str] | None = Field(
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

    server_config: FlockMCPStdioServerConfig = Field(
        ...,
        description="Config for the server."
    )

    def add_module(self, module):
        return super().add_module(module)

    def remove_module(self, module_name):
        return super().remove_module(module_name)

    def get_module(self, module_name):
        return super().get_module(module_name)

    def get_enabled_modules(self):
        return super().get_enabled_modules()

    async def initialize(self):
        """
        Called when initializing the server
        """
        async with self.condition:
            # Check if we already initialized
            if not self.initialized:
                # Initialize the underlying Connection Pool
                if not self.connection_manager:
                    self.connection_manager = FlockStdioMCPConnectionManager(
                        transport_type="stdio",
                        server_name=self.server_config.server_name,
                        connection_parameters=StdioServerParameters(
                            command=self.server_config.command,
                            args=self.server_config.args,
                            env=self.server_config.env,
                            cwd=self.server_config.cwd,
                            encoding=self.server_config.encoding,
                            encoding_error_handler=self.server_config.encoding_error_handler
                        ),
                        min_connections=1,
                        max_connections=1,
                        max_reconnect_attemtps=self.server_config.max_restart_attempts,
                        original_roots=self.server_config.mount_points,
                        sampling_callback=self.server_config.sampling_callback,
                        logging_callback=self.server_config.logging_callback,
                        message_handler=self.server_config.message_handler,
                        list_roots_callback=self.server_config.list_roots_callback,
                    )
            self.condition.notify()

    async def get_tools(self) -> list[DSPyTool]:
        """
        Retrieve a list of tools from the server.
        """
        if not self.initialized:
            # Make sure a connection is up and running.
            await self.initialize()

        async with self.condition:
            try:
                result: list[DSPyTool] = await self.connection_manager.get_tools()
                return result
            except Exception as e:
                logger.error(
                    f"Unexpected Exception ocurred while trying go get tools from server '{self.server_config.server_name}'")
                return []
            finally:
                self.condition.notify()
