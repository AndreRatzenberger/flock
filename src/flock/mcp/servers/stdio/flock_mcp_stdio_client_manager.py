"""Manages a pool of connections for a Stdio Server."""

from datetime import timedelta
from typing import Literal
from pydantic import Field
from flock.core.mcp.flock_mcp_client_base import StdioServerParameters
from flock.core.mcp.flock_mcp_client_manager import FlockMCPClientManager
from flock.core.logging.logging import get_logger
from flock.mcp.servers.stdio.flock_stdio_client import FlockStdioClient

logger = get_logger("mcp.stdio.connection_manager")


class FlockStdioMCPClientManager(FlockMCPClientManager[FlockStdioClient]):
    """Handles Clients that connect to a Stdio-Transport Type Server."""

    transport_type: Literal["stdio"] = Field(
        default="stdio",
        description="What kind of transport to use."
    )

    connection_parameters: StdioServerParameters = Field(
        ...,
        description="Connection Parameters"
    )

    async def get_client(self, agent_id, run_id):
        """
        Speciality for Stdio Clients:
            Since the clients interact with a local script,
            we override this method such that each agent gets the 
            same client, irrespective of agent_id or run_id.
        """
        client = None
        client_dict = self.clients.get("default_stdio", None)
        if not client_dict:
            client = await self.make_client(agent_id=agent_id, run_id=run_id)
            self.clients["default_stdio"]["default_stdio"] = client
        else:
            client = client_dict.get("default_stdio", None)
            if not client:
                client = await self.make_client(agent_id=agent_id, run_id=run_id)
                self.clients["default_stdio"]["default_stdio"] = client

        # So now, we finally have a client
        return client

    async def make_client(self, agent_id, run_id):

        new_client = FlockStdioClient(
            server_name=self.server_name,
            agent_id=agent_id,
            run_id=run_id,
            transport_type="stdio",
            connection_parameters=self.connection_parameters,
            max_retries=self.max_retries,
            current_roots=self.initial_roots,
            is_busy=False,
            is_alive=True,
            has_error=False,
            error_message=None,
            tools_enabled=True,

        )

        return new_client
