"""Manages a pool of connections for a particular server."""

from abc import ABC, abstractmethod
from asyncio import Lock
from datetime import timedelta
from typing import Generic, Literal, TypeVar

from dspy import Tool as DSPyTool
from opentelemetry import trace
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    create_model,
)

from flock.core.logging.logging import get_logger
from flock.core.mcp.flock_mcp_client_base import (
    FlockMCPClientBase,
    ServerParameters,
)
from flock.core.mcp.flock_mcp_tool_base import FlockMCPToolBase
from flock.core.mcp.types.mcp_callbacks import (
    FlockListRootsMCPCallback,
    FlockLoggingMCPCallback,
    FlockMessageHandlerMCPCallback,
    FlockSamplingMCPCallback,
)
from flock.core.mcp.types.mcp_types import Root

logger = get_logger("core.mcp.connection_manager_base")
tracer = trace.get_tracer(__name__)

TClient = TypeVar("TClient", bound="FlockMCPClientBase")

C = TypeVar("C", bound="FlockMCPClientManagerConfigBase")

LoggingLevel = Literal[
    "debug",
    "info",
    "notice",
    "warning",
    "error",
    "critical",
    "alert",
    "emergency",
]


class FlockMCPClientManagerConfigBase(BaseModel):
    """Base Configuration class for Client Manager Configurations.

    Each client manager subclass must implement their own
    subclass of this configuration class.
    """

    server_name: str = Field(
        ..., description="Name of the server to connect to."
    )

    server_logging_level: LoggingLevel = Field(
        default="error",
        description="Requested Logging Level for the remote server.",
    )

    transport_type: Literal["stdio", "websockets", "sse", "custom"] = Field(
        ..., description="Transport type to use for connection."
    )

    connection_parameters: ServerParameters = Field(
        ..., description="Connection parameters for the server."
    )

    read_timeout_seconds: timedelta = Field(
        default_factory=lambda: timedelta(seconds=10),
        description="How long until a connection times out.",
    )

    max_retries: int = Field(
        default=3,
        description="How many times to try to reconnect if an Exception occurs.",
    )

    logging_callback: FlockLoggingMCPCallback | None = Field(
        default=None,
        description="Logging Callback for Handling Logging Notifications sent by the server.",
    )

    message_handler: FlockMessageHandlerMCPCallback | None = Field(
        default=None,
        description="Message Handler Callback for Handling Message events from the server.",
    )

    sampling_callback: FlockSamplingMCPCallback | None = Field(
        default=None,
        description="Sampling Handler Callback for Handling Sampling requests from the server.",
    )

    list_roots_callback: FlockListRootsMCPCallback | None = Field(
        default=None, description="List Roots callback to pass on to clients."
    )

    mount_points: list[Root] | None = Field(
        default=None, description="Initial Mounting Points on the server."
    )

    tool_cache_max_size: float = Field(
        default=100, description="Max number of items in the tools cache."
    )

    tool_cache_max_ttl: float = Field(
        default=60 * 5,
        description="Max TTL for Items in the Tool Cache in Seconds.",
    )

    tool_result_cache_max_size: float = Field(
        default=100, description="Max number of items in the tool result cache."
    )

    tool_result_cache_max_ttl: float = Field(
        default=60 * 5,
        description="Max TTL for items in the tool result cache in seconds.",
    )

    resource_contents_cache_max_size: float = Field(
        default=100,
        description="Max number of entries in the Resource Contents Cache.",
    )

    resource_contents_cache_max_ttl: float = Field(
        default=60 * 2,
        description="Max TTL for Items on the Resource Contents Cache in seconds.",
    )

    resource_list_cache_max_size: float = Field(
        default=1000,
        description="Max number of Items in the Resource List Cache.",
    )

    resource_list_cache_max_ttl: float = Field(
        default=10,
        description="Max TTL for Items in the Resource List Cache in Seconds.",
    )

    roots_enabled: bool = Field(
        default=False,
        description="Whether or not the Roots feature is enabled.",
    )

    prompts_enabled: bool = Field(
        default=False,
        description="Whether or not the Prompts feature is enabled.",
    )

    sampling_enabled: bool = Field(
        default=False,
        description="Whether or not the Sampling feature is enabled.",
    )

    tools_enabled: bool = Field(
        default=False,
        description="Whether or not the Tools feature is enabled.",
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @classmethod
    def with_fields(cls: type[C], **field_definitions) -> type[C]:
        """Create a new config class with additional fields."""
        return create_model(
            f"Dynamic{cls.__name__}", __base__=cls, **field_definitions
        )


class FlockMCPClientManager(BaseModel, ABC, Generic[TClient]):
    """Handles a Pool of MCPClients of type TClient."""

    config: FlockMCPClientManagerConfigBase = Field(
        ..., description="Client Manager Configuration."
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
        """Provides a client
        from the pool.
        """
        # Attempt to get a client from the client store.
        # clients are stored like this: agent_id -> run_id -> client
        async with self.lock:
            try:
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
                    f"Unexpected Exception ocurred while trying to get client for server '{self.config.server_name}' with agent_id: {agent_id} and run_id: {run_id}: {e}"
                )
                raise e

    async def get_tools(self, agent_id: str, run_id: str) -> list[DSPyTool]:
        """Retrieves a list of tools for the agents to act on."""
        try:
            client = await self.get_client(agent_id=agent_id, run_id=run_id)
            tools: list[FlockMCPToolBase] = await client.get_tools(
                agent_id=agent_id, run_id=run_id
            )
            dspy_tools = [t.as_dspy_tool(mgr=self) for t in tools]
            return dspy_tools
        except Exception as e:
            logger.error(
                f"Exception occurred while trying to retrieve Tools for server '{self.config.server_name}' with agent_id: {agent_id} and run_id: {run_id}: {e}"
            )
            return []

    async def close_all(self) -> None:
        """Closes all connections in the pool and cancels background tasks."""
        async with self.lock:
            for agent_id, run_dict in self.clients.items():
                logger.debug(
                    f"Shutting down all clients for agent_id: {agent_id}"
                )
                for run_id, client in run_dict.items():
                    logger.debug(
                        f"Shutting down client for agent_id {agent_id} and run_id {run_id}"
                    )
                    await client.disconnect()
            self.clients = {}  # Let the GC take care of the rest.
            logger.info(
                f"All clients disconnected for server '{self.config.server_name}'"
            )
