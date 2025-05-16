from typing import Literal

from mcp import stdio_client
from opentelemetry import trace
from pydantic import Field

from flock.core.logging.logging import get_logger
from flock.core.mcp.flock_mcp_client_base import (
    FlockMCPClientBase,
    FlockMCPClientBaseConfig,
    StdioServerParameters,
)

logger = get_logger("mcp.stdio_client")
tracer = trace.get_tracer(__name__)


class FlockStdioMCPClientConfig(FlockMCPClientBaseConfig):
    transport_type: Literal["stdio"] = Field(
        default="stdio", description="Stdio Transport."
    )

    connection_paramters: StdioServerParameters = Field(
        ..., description="Connection Parameters for Stdio server."
    )


class FlockStdioClient(FlockMCPClientBase):
    """Client for connecting to Stdio Servers."""

    config: FlockStdioMCPClientConfig = Field(
        ..., description="Configuration for stdio clients."
    )

    def get_init_function(self):
        """Define which function should be used to establish a read and write stream between the client and the server."""
        return stdio_client
