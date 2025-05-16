"""Manages a pool of connections for a Stdio Server."""

from typing import Literal

from pydantic import Field

from flock.core.logging.logging import get_logger
from flock.core.mcp.flock_mcp_client_base import StdioServerParameters
from flock.core.mcp.flock_mcp_client_manager import (
    FlockMCPClientManager,
    FlockMCPClientManagerConfigBase,
)
from flock.mcp.servers.stdio.flock_stdio_client import (
    FlockStdioClient,
    FlockStdioMCPClientConfig,
)

logger = get_logger("mcp.stdio.connection_manager")


class FlockMCPStdioClientManagerConfig(FlockMCPClientManagerConfigBase):
    """Configuration for Stdio Client Managers."""

    transport_type: Literal["stdio"] = Field(
        default="stdio", description="What kind of transport to use."
    )

    connection_parameters: StdioServerParameters = Field(
        ..., description="Connection Parameters"
    )


class FlockStdioMCPClientManager(FlockMCPClientManager[FlockStdioClient]):
    """Handles Clients that connect to a Stdio-Transport Type Server."""

    config: FlockMCPStdioClientManagerConfig = Field(..., description="Config.")

    async def make_client(self):
        """Instantiate a client for an stdio-server."""
        new_client = FlockStdioClient(
            config=FlockStdioMCPClientConfig(
                server_name=self.config.server_name,
                transport_type=self.config.transport_type,
                server_logging_level=self.config.server_logging_level,
                connection_paramters=self.config.connection_parameters,
                max_retries=self.config.max_retries,
                mount_points=self.config.mount_points,
                read_timeout_seconds=self.config.read_timeout_seconds,
                sampling_callback=self.config.sampling_callback,
                list_roots_callback=self.config.list_roots_callback,
                logging_callback=self.config.logging_callback,
                message_handler=self.config.message_handler,
                tool_cache_max_size=self.config.tool_cache_max_size,
                tool_cache_max_ttl=self.config.tool_cache_max_ttl,
                resource_contents_cache_max_size=self.config.resource_contents_cache_max_size,
                resource_contents_cache_max_ttl=self.config.resource_contents_cache_max_ttl,
                resource_list_cache_max_size=self.config.resource_list_cache_max_size,
                resource_list_cache_max_ttl=self.config.resource_list_cache_max_ttl,
                tool_result_cache_max_size=self.config.tool_result_cache_max_size,
                tool_result_cache_max_ttl=self.config.tool_result_cache_max_ttl,
                roots_enabled=self.config.roots_enabled,
                tools_enabled=self.config.tools_enabled,
                prompts_enabled=self.config.prompts_enabled,
                sampling_enabled=self.config.sampling_enabled,
            )
        )

        return new_client
