from typing import Literal

from pydantic import Field
from flock.core.mcp.flock_mcp_client_base import SseServerParameters
from flock.core.mcp.flock_mcp_client_manager import FlockMCPClientManager, FlockMCPClientManagerConfigBase
from flock.mcp.servers.sse.flock_sse_client import FlockSseClient, FlockSseMCPClientConfig


class FlockMCPSseClientManagerConfig(FlockMCPClientManagerConfigBase):
    """
    Configuration for Sse Client Managers.
    """

    transport_type: Literal["sse"] = Field(
        default="sse",
        description="What kind of transport to use"
    )

    connection_parameters: SseServerParameters = Field(
        ...,
        description="Connection paramters."
    )


class FlockSseMCPClientManager(FlockMCPClientManager[FlockSseClient]):

    config: FlockMCPSseClientManagerConfig = Field(
        ...,
        description="Config."
    )

    async def make_client(self):
        """
        Instantiate a client for an sse-server.
        """

        new_client = FlockSseClient(
            config=FlockSseMCPClientConfig(
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
                tool_result_cache_max_size=self.config.tool_result_cache_max_size,
                tool_result_cache_max_ttl=self.config.tool_result_cache_max_ttl,
                roots_enabled=self.config.roots_enabled,
                tools_enabled=self.config.tools_enabled,
                prompts_enabled=self.config.prompts_enabled,
                sampling_enabled=self.config.sampling_enabled,
            )
        )
