from abc import ABC
from pathlib import Path
from typing import Literal
from opentelemetry import trace
from mcp.client.stdio import get_default_environment
from pydantic import Field

from flock.core.mcp.flock_mcp_client_base import StdioServerParameters
from flock.core.mcp.flock_mcp_server import FlockMCPServerBase, FlockMCPServerConfig
from flock.core.logging.logging import get_logger
from flock.core.serialization.serializable import Serializable

from dspy import Tool as DSPyTool

from flock.mcp.servers.stdio.flock_mcp_stdio_client_manager import FlockMCPStdioClientManagerConfig, FlockStdioMCPClientManager


logger = get_logger("mcp.stdio.server")
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

    transport_type: Literal["stdio"] = Field(
        default="stdio",
        description="Stdio Transport Type."
    )

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

    async def initialize(self) -> FlockStdioMCPClientManager:
        """
        Called when initializing the server
        """
        client_manager = FlockStdioMCPClientManager(
            config=FlockMCPStdioClientManagerConfig(
                server_name=self.server_config.server_name,
                server_logging_level=self.server_config.server_logging_level,
                transport_type=self.server_config.transport_type,
                read_timeout_seconds=self.server_config.read_timeout_seconds,
                max_retries=self.server_config.max_restart_attempts,
                logging_callback=self.server_config.logging_callback,
                message_handler=self.server_config.message_handler,
                sampling_callback=self.server_config.sampling_callback,
                list_roots_callback=self.server_config.list_roots_callback,
                mount_points=self.server_config.mount_points,
                tool_cache_max_size=self.server_config.tool_cache_max_size,
                resource_contents_cache_max_size=self.server_config.resource_contents_cache_max_size,
                resource_list_cache_max_size=self.server_config.resource_list_cache_max_size,
                tool_cache_max_ttl=self.server_config.tool_cache_max_ttl,
                resource_contents_cache_max_ttl=self.server_config.resource_contents_cache_max_ttl,
                resource_list_cache_max_ttl=self.server_config.resource_list_cache_max_ttl,
                roots_enabled=self.server_config.roots_enabled,
                tools_enabled=self.server_config.tools_enabled,
                prompts_enabled=self.server_config.prompts_enabled,
                sampling_enabled=self.server_config.sampling_enabled,
                connection_parameters=StdioServerParameters(
                    command=self.server_config.command,
                    args=self.server_config.args,
                    cwd=self.server_config.cwd,
                    encoding=self.server_config.encoding,
                    encoding_error_handler=self.server_config.encoding_error_handler
                ),
                tool_result_cache_max_size=self.server_config.tool_result_cache_max_size,
                tool_result_cache_max_ttl=self.server_config.tool_result_cache_max_ttl,
            )
        )
        return client_manager
