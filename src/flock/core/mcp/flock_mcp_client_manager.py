"""Manages a pool of connections for a particular server."""


from abc import ABC, abstractmethod
from asyncio import Condition, Lock
import asyncio
import random
from typing import Annotated, Any, Callable, Generic, Literal, TypeVar
from pydantic import AnyUrl, BaseModel, ConfigDict, Field, UrlConstraints
from opentelemetry import trace

from mcp import InitializeResult, StdioServerParameters

from dspy import Tool as DSPyTool

from flock.core.mcp.flock_mcp_client_base import FlockMCPClientBase, ServerParameters, SseServerParameters, WebSocketServerParameters
from flock.core.logging.logging import get_logger
from flock.core.mcp.types.mcp_callbacks import FlockLoggingMCPCallback

logger = get_logger("core.mcp.connection_manager_base")
tracer = trace.get_tracer(__name__)

TClient = TypeVar("TClient", bound="FlockMCPClientBase")

# TODO: Finish rework.


class FlockMCPClientManager(BaseModel, ABC, Generic[TClient]):
    """Handles a Pool of MCPClients of type TClient."""

    transport_type: Literal["stdio", "websockets", "sse", "custom"] = Field(
        ..., description="Transport-Type to use for Connections")

    connection_parameters: ServerParameters = Field(
        ...,
        description="Connection parameters for the server"
    )

    server_name: str = Field(...,
                             description="Name of the server to connect to.")

    lock: Lock = Field(
        default_factory=Lock,
        description="Lock for mutex access.",
        exclude=True,
    )

    clients: dict[str, dict[str, FlockMCPClientBase]] = Field(
        default_factory=dict,
        exclude=True,
        description="Internal Store for the clients."
    )

    logging_callback: FlockLoggingMCPCallback | None = Field(
        default=None,
        description="Logging Callback to pass on to clients"
    )

    # --- Pydantic v2 Configuratioin ---
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @abstractmethod
    async def make_client(self, agent_id: str, run_id: str) -> TClient:
        """
        Instantiate-but don't connect yet-a fresh client of the concrete subtype.
        """
        pass

    async def get_client(self, agent_id: str, run_id: str) -> TClient:
        """
        Provides a client
        from the pool.
        """

        # Attempt to get a client from the client store.
        # clients are stored like this: agent_id -> run_id -> client
        with self.lock:
            try:
                client = None
                run_clients = self.clients.get(agent_id, None)
                if run_clients is None:
                    # This means, that across all runs, no agent has ever needed a client.
                    # This also means that we need to create a client.
                    client = await self.make_client(agent_id=agent_id, run_id=run_id)
                    # Insert the freshly created client
                    self.clients[agent_id] = {}
                    self.clients[agent_id][run_id] = client

                else:
                    # This means there is at least one entry for the agent_id available
                    # Now, all we need to do is check if the run_id matches the entrie's run_id
                    client = await run_clients.get(run_id, None)
                    if client is None:
                        # Means no client here with the respective run_id
                        client = await self.make_client(agent_id=agent_id, run_id=run_id)
                        # Insert the freshly created client.
                        self.clients[agent_id][run_id] = client

                return client
            except Exception as e:
                # Log the exception and raise it so it becomes visible downstream
                logger.error(
                    f"Unexpected Exception ocurred while trying to get client for server '{self.server_name}' with agent_id: {agent_id} and run_id: {run_id}: {e}")
                raise e

    async def get_tools(self, agent_id: str, run_id: str) -> list[DSPyTool]:
        """
        Retrieves a list of tools for the agents to act on.
        """
        async with self.lock:
            try:
                client = await self.get_client(agent_id=agent_id, run_id=run_id)
                return await client.get_tools(agent_id=agent_id, run_id=run_id)
            except Exception as e:
                logger.error(
                    f"Exception occurred while trying to retrieve Tools for server '{self.server_name}' with agent_id: {agent_id} and run_id: {run_id}: {e}"
                )
                return []

    async def close_all(self) -> None:
        """
        Closes all connections in the pool and cancels background tasks.
        """
        async with self.lock:
            for agent_id, run_dict in self.clients.items():
                logger.debug(
                    f"Shutting down all clients for agent_id: {agent_id}")
                for run_id, client in run_dict.items():
                    logger.debug(
                        f"Shutting down client for agent_id {agent_id} and run_id {run_id}")
                    await client.disconnect()
            logger.info(
                f"All clients disconnected for server '{self.server_name}'")
