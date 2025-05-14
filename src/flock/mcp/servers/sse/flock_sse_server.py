
from typing import Any, Literal
from opentelemetry import trace
from pydantic import AnyUrl, Field
from flock.core.logging.logging import get_logger
from flock.core.mcp.flock_mcp_client_base import SseServerParameters
from flock.core.mcp.flock_mcp_server import FlockMCPServerBase, FlockMCPServerConfig
from flock.mcp.servers.sse.flock_mcp_sse_client_manager import FlockMCPSseClientManagerConfig, FlockSseMCPClientManager


logger = get_logger("mcp.sse.server")
tracer = trace.get_tracer(__name__)


class FlockMCPSseServerConfig(FlockMCPServerConfig):
    """
    Configuration Class
    for Flock MCP Servers using the Sse transport method.
    """

    transport_type: Literal["sse"] = Field(
        default="sse",
        description="SSE Transport Type."
    )

    url: str | AnyUrl = Field(
        ...,
        description="The URL the server listens on."
    )

    headers: dict[str, Any] | None = Field(
        default=None,
        description="Headers for connection to server."
    )

    sse_read_timeout: int = Field(
        default=60*5,
        description="Timeout before connection closes."
    )


class FlockMCPSseServer(FlockMCPServerBase):
    """
    Class which represents a MCP Server using the
    SSE Transport type.

    This means (most likely) a remotely running
    FastMCP Server.
    """

    server_config: FlockMCPSseServerConfig = Field(
        ...,
        description="Config for the server."
    )

    async def initialize(self) -> FlockSseMCPClientManager:
        """
        Called when initializing the server.
        """
        client_manager = FlockSseMCPClientManager(
            config=FlockMCPSseClientManagerConfig(
                server_name=self.server_config.server_name,
                server_logging_level=self.server_config.server_logging_level,
                transport_type=self.server_config.transport_type,
                read_timeout_seconds=self.server_config.read_timeout_seconds,
                max_retries=self.server_config.max_restart_attempts,
                logging_callback=self.server_config.logging_callback,
                message_handler=self.server_config.message_handler,
                list_roots_callback=self.server_config.list_roots_callback,
                mount_points=self.server_config.mount_points,
                tool_cache_max_size=self.server_config.tool_cache_max_size,
                tool_cache_max_ttl=self.server_config.tool_cache_max_ttl,
                resource_contents_cache_max_size=self.server_config.resource_contents_cache_max_size,
                resource_contents_cache_max_ttl=self.server_config.resource_contents_cache_max_ttl,
                tool_result_cache_max_size=self.server_config.tool_result_cache_max_size,
                tool_result_cache_max_ttl=self.server_config.tool_result_cache_max_ttl,
                resource_list_cache_max_size=self.server_config.resource_list_cache_max_size,
                resource_list_cache_max_ttl=self.server_config.resource_list_cache_max_ttl,
                connection_parameters=SseServerParameters(
                    url=self.server_config.url,
                    headers=self.server_config.headers,
                    sse_read_timeout=self.server_config.sse_read_timeout,
                )

            )
        )

        return client_manager
