"""Manages a pool of connections for a Stdio Server."""

from datetime import timedelta
from typing import Literal
from mcp import StdioServerParameters
from pydantic import Field
from flock.core.mcp.flock_mcp_client_manager import FlockMCPClientManager
from flock.core.logging.logging import get_logger
from flock.mcp.servers.stdio.flock_stdio_client import FlockStdioClient

logger = get_logger("mcp.stdio.connection_manager")


class FlockStdioMCPClientManager(FlockMCPClientManager[FlockStdioClient]):
    """Handles Clients that connect to a Stdio-Transport Type Server."""

    async def make_client(self, agent_id, run_id):
        return await super().make_client(agent_id, run_id)
