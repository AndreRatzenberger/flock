from typing import Literal

from pydantic import Field
from flock.core.mcp.flock_mcp_client_base import FlockMCPClientBase, FlockMCPClientBaseConfig, SseServerParameters
from mcp.client.sse import sse_client


class FlockSseMCPClientConfig(FlockMCPClientBaseConfig):

    transport_type: Literal["sse"] = Field(
        default="sse",
        description="SSe Transport"
    )

    connection_paramters: SseServerParameters = Field(
        ...,
        description="Connection Parameters for SSE Server."
    )


class FlockSseClient(FlockMCPClientBase):
    """
    Client for connecting to SSE Servers.
    """

    config: FlockSseMCPClientConfig = Field(
        ...,
        description="Configuration for Sse Clients."
    )

    def get_init_function(self):
        return sse_client
