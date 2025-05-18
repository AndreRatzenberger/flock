"""This module provides the Flock SSE Server functionality."""

from contextlib import AbstractAsyncContextManager
from typing import Literal

from anyio.streams.memory import (
    MemoryObjectReceiveStream,
    MemoryObjectSendStream,
)
from mcp.client.sse import sse_client
from mcp.types import JSONRPCMessage
from opentelemetry import trace
from pydantic import Field

from flock.core.logging.logging import get_logger
from flock.core.mcp.flock_mcp_server import FlockMCPServerBase
from flock.core.mcp.mcp_client import FlockMCPClientBase
from flock.core.mcp.mcp_client_manager import FlockMCPClientManager
from flock.core.mcp.mcp_config import (
    FlockMCPConfigurationBase,
    FlockMCPConnectionConfiguration,
)
from flock.core.mcp.types.types import SseServerParameters

logger = get_logger("mcp.sse.server")
tracer = trace.get_tracer(__name__)


class FlockSSEConnectionConfig(FlockMCPConnectionConfiguration):
    """Concrete ConnectionConfig for an SSEClient."""

    # Only thing we need to override here is the concrete transport_type
    # and connection_parameters fields.
    transport_type: Literal["sse"] = Field(
        default="sse", description="Use the sse transport type."
    )

    connection_parameters: SseServerParameters = Field(
        ..., description="SSE Server Connection Parameters."
    )


class FlockSSEConfig(FlockMCPConfigurationBase):
    """Configuration for SSE Clients."""

    # The only thing we need to override here is the concrete
    # connection config. The rest is generic enough to handle
    # everything else.
    connection_config: FlockSSEConnectionConfig = Field(
        ..., description="Concrete SSE Connection Configuration."
    )


class FlockSSEClient(FlockMCPClientBase):
    """Client for SSE Servers."""

    config: FlockSSEConfig = Field(..., description="Client configuration.")

    async def create_transport(
        self, params
    ) -> AbstractAsyncContextManager[
        tuple[
            MemoryObjectReceiveStream[JSONRPCMessage | Exception],
            MemoryObjectSendStream[JSONRPCMessage],
        ]
    ]:
        """Return an async context manager whose __aenter__ method yields (read_stream, send_stream)."""
        return sse_client(
            url=self.config.connection_config.connection_parameters.url,
            headers=self.config.connection_config.connection_parameters.headers,
            timeout=self.config.connection_config.connection_parameters.timeout,
            sse_read_timeout=self.config.connection_config.connection_parameters.sse_read_timeout,
        )


class FlockSSEClientManager(FlockMCPClientManager):
    """Manager for handling SSE Clients."""

    client_config: FlockSSEConfig = Field(
        ..., description="Configuration for clients."
    )

    async def make_client(self) -> FlockSSEClient:
        """Create a new client instance."""
        new_client = FlockSSEClient(config=self.client_config)
        return new_client


class FlockSSEServer(FlockMCPServerBase):
    """Class which represents a MCP Server using the SSE Transport type."""

    config: FlockSSEConfig = Field(..., description="Config for the server.")

    async def initialize(self) -> FlockSSEClientManager:
        """Called when initializing the server."""
        client_manager = FlockSSEClientManager(
            client_config=self.config,
        )

        return client_manager
