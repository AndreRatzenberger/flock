from contextlib import AsyncExitStack, asynccontextmanager
from typing import Literal
from mcp import ClientSession, InitializeResult, StdioServerParameters, stdio_client
from pydantic import Field
from flock.core.mcp.flock_mcp_client_base import FlockMCPClientBase


from flock.core.logging.logging import get_logger
from opentelemetry import trace

logger = get_logger("mcp.stdio_client")
tracer = trace.get_tracer(__name__)


class FlockStdioClient(FlockMCPClientBase):
    """
    Client for connecting to Stdio Servers.
    """

    def get_init_function(self):
        """
        Define which function should be used to establish
        a read and write stream between the client and the server.
        """
        return stdio_client
