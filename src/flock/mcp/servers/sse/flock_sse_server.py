
from typing import Literal
from opentelemetry import trace
from pydantic import Field
from flock.core.logging.logging import get_logger
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
            )
        )
