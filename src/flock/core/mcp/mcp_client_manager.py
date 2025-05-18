"""Manages a pool of connections for a particular server."""

from abc import ABC, abstractmethod
from asyncio import Lock
from typing import Generic, TypeVar

from opentelemetry import trace
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from flock.core.logging.logging import get_logger
from flock.core.mcp.flock_mcp_tool_base import FlockMCPToolBase
from flock.core.mcp.mcp_client import (
    FlockMCPClientBase,
)
from flock.core.mcp.mcp_config import FlockMCPConfigurationBase

logger = get_logger("core.mcp.connection_manager_base")
tracer = trace.get_tracer(__name__)

TClient = TypeVar("TClient", bound="FlockMCPClientBase")


class FlockMCPClientManager(BaseModel, ABC, Generic[TClient]):
    """Handles a Pool of MCPClients of type TClient."""

    client_config: FlockMCPConfigurationBase = Field(
        ..., description="Configuration for clients."
    )

    lock: Lock = Field(
        default_factory=Lock,
        description="Lock for mutex access.",
        exclude=True,
    )

    clients: dict[str, dict[str, FlockMCPClientBase]] = Field(
        default_factory=dict,
        exclude=True,
        description="Internal Store for the clients.",
    )

    # --- Pydantic v2 Configuratioin ---
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @abstractmethod
    async def make_client(self) -> type[TClient]:
        """Instantiate-but don't connect yet-a fresh client of the concrete subtype."""
        pass

    async def get_client(self, agent_id: str, run_id: str) -> type[TClient]:
        """Provides a client from the pool."""
        # Attempt to get a client from the client store.
        # clients are stored like this: agent_id -> run_id -> client
        with tracer.start_as_current_span("client_manager.get_client") as span:
            span.set_attribute("agent_id", agent_id)
            span.set_attribute("run_id", run_id)
            async with self.lock:
                try:
                    logger.debug(
                        f"Attempting to get client for server '{self.client_config.server_name}'"
                    )
                    client = None
                    run_clients = self.clients.get(agent_id, None)
                    if run_clients is None:
                        # This means, that across all runs, no agent has ever needed a client.
                        # This also means that we need to create a client.
                        client = await self.make_client()
                        # Insert the freshly created client
                        self.clients[agent_id] = {}
                        self.clients[agent_id][run_id] = client

                    else:
                        # This means there is at least one entry for the agent_id available
                        # Now, all we need to do is check if the run_id matches the entrie's run_id
                        client = run_clients.get(run_id, None)
                        if client is None:
                            # Means no client here with the respective run_id
                            client = await self.make_client()
                            # Insert the freshly created client.
                            self.clients[agent_id][run_id] = client

                    return client
                except Exception as e:
                    # Log the exception and raise it so it becomes visible downstream
                    logger.error(
                        f"Unexpected Exception ocurred while trying to get client for server '{self.client_config.server_name}' with agent_id: {agent_id} and run_id: {run_id}: {e}"
                    )
                    span.record_exception(e)
                    raise e

    async def get_tools(
        self, agent_id: str, run_id: str
    ) -> list[FlockMCPToolBase]:
        """Retrieves a list of tools for the agents to act on."""
        with tracer.start_as_current_span("client_manager.get_tools") as span:
            span.set_attribute("agent_id", agent_id)
            span.set_attribute("run_id", run_id)
            try:
                client = await self.get_client(agent_id=agent_id, run_id=run_id)
                tools: list[FlockMCPToolBase] = await client.get_tools(
                    agent_id=agent_id, run_id=run_id
                )
                return tools
            except Exception as e:
                logger.error(
                    f"Exception occurred while trying to retrieve Tools for server '{self.client_config.server_name}' with agent_id: {agent_id} and run_id: {run_id}: {e}"
                )
                span.record_exception(e)
                return []

    async def close_all(self) -> None:
        """Closes all connections in the pool and cancels background tasks."""
        with tracer.start_as_current_span("client_manager.close_all") as span:
            async with self.lock:
                for agent_id, run_dict in self.clients.items():
                    logger.debug(
                        f"Shutting down all clients for agent_id: {agent_id}"
                    )
                    for run_id, client in run_dict.items():
                        logger.debug(
                            f"Shutting down client for agent_id {agent_id} and run_id {run_id}"
                        )
                        try:
                            await client.disconnect()
                        except Exception as e:
                            logger.error(
                                f"Error when trying to disconnect client for server '{self.client_config.server_name}': {e}"
                            )
                            span.record_exception(e)
                self.clients = {}  # Let the GC take care of the rest.
                logger.info(
                    f"All clients disconnected for server '{self.client_config.server_name}'"
                )
