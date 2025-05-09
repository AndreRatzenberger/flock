"""Manages a pool of connections for a Stdio Server."""

from datetime import timedelta
from typing import Literal
from mcp import StdioServerParameters
from pydantic import Field
from flock.core.mcp.flock_mcp_connection_manager_base import FlockMCPConnectionManagerBase
from flock.core.logging.logging import get_logger
from flock.mcp.servers.stdio.flock_stdio_client import FlockStdioClient

logger = get_logger("mcp.stdio.connection_manager")


class FlockStdioMCPConnectionManager(FlockMCPConnectionManagerBase[FlockStdioClient]):
    """Handles Clients that connect to a Stdio-Transport Type Server."""

    transport_type: Literal['stdio'] = Field(
        default='stdio',
        description="Transport-type to use for Connections."
    )

    connection_parameters: StdioServerParameters = Field(
        ...,
        description="Connection parameters for the server script."
    )

    async def initialize(self):
        # Stdio-Servers should only hold one connection at most.
        self.max_connections = 1
        self.min_connections = 1
        return await super().initialize()

    async def _make_client(self) -> FlockStdioClient:
        logger.debug(f"Creating Stdio Client for server '{self.server_name}'")
        try:
            return FlockStdioClient(
                server_name=self.server_name,
                transport_type=self.transport_type,
                connection_parameters=self.connection_parameters,
                sampling_callback=self.sampling_callback,
                list_roots_callback=self.list_roots_callback,
                logging_callback=self.logging_callback,
                message_handler=self.message_handler,
                max_retries=self.max_reconnect_attemtps,
                read_timeout_seconds=timedelta(seconds=5),
            )
        except Exception as e:
            logger.error(
                f"Exception ocurred while attempting to create stdio client for server '{self.server_name}': {e}")
            raise e
