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

logger = get_logger("core.mcp.connection_manager_base")
tracer = trace.get_tracer(__name__)

TClient = TypeVar("TClient", bound="FlockMCPClientBase")

# TODO: Do not use pooling, but rather implement a dict in which clients are stored thusly: agent_id -> run_id -> client
# TODO: Remove replenish task. Make creation of new clients synchronous within this task.


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

    min_connections: int = Field(
        default=1,
        description="Minimum amount of connections to keep alive.",
    )

    max_connections: int = Field(
        default=24,
        description="Upper bound for the maximum amount of connections to open."
    )

    max_reconnect_attemtps: int = Field(
        default=10,
        description="Maximum amounts to retry to create a client. After that the ConnectionManager throws an Exception.",
    )

    lock: Lock = Field(
        default_factory=Lock,
        description="Lock for mutex access.",
        exclude=True,
    )

    sampling_callback: Callable[..., Any] | None = Field(
        default=None,
        description="Callback for handling sampling requests."
    )

    list_roots_callback: Callable[..., Any] | None = Field(
        default=None,
        description="Callback for handling list_roots request."
    )

    logging_callback: Callable[..., Any] | None = Field(
        default=None,
        description="Callback for logging."
    )

    message_handler: Callable[..., Any] | None = Field(
        default=None,
        description="Message Handler Callback."
    )

    # --- Pydantic v2 Configuratioin ---
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @abstractmethod
    async def _make_client(self, agent_id: str, run_id: str) -> TClient:
        """
        Instantiate-but don't connect yet-a fresh client of the concrete subtype
        """
        pass

    async def get_client(self, agent_id: str, run_id: str) -> TClient:
        """
        Provides a client
        from the pool.
        """
        pass

    async def release_client(self, client: TClient) -> None:
        """
        Returns a client to the pool.
        """
        pass

    async def _get_available_client(self) -> TClient:
        """
        Retrieves a client for an agent based on agent_id and run_id.
        If the client is busy, it simply waits until it becomes available again.
        """
        pass

    async def _release_client(self, client: TClient) -> None:
        """
        Releases a client back into the available pool and notifies waiting tasks.
        Moves the client from busy to available list. Discards unhealthy clients.
        """

    # --- Public functions ---
    async def get_tools(self, agent_id: str, run_id: str) -> list[DSPyTool]:
        """
        Retrieves a list of tools for the agents to act on.
        """
        pass

    async def close_all(self) -> None:
        """
        Closes all connections in the pool and cancels background tasks.
        """
        pass
