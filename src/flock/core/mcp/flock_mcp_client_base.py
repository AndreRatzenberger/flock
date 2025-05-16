"""Wrapper Class for a mcp ClientSession Object."""

import asyncio
import random
from abc import ABC, abstractmethod
from asyncio import Lock
from collections.abc import Callable
from contextlib import (
    AbstractAsyncContextManager,
    AsyncExitStack,
)
from datetime import timedelta
from typing import (
    Any,
    Literal,
    TypeVar,
)

import httpx
from anyio import ClosedResourceError
from anyio.streams.memory import (
    MemoryObjectReceiveStream,
    MemoryObjectSendStream,
)
from cachetools import TTLCache, cached
from mcp import (
    ClientSession,
    InitializeResult,
    ListToolsResult,
    McpError,
    ServerCapabilities,
    StdioServerParameters as _MCPStdioServerParameters,
)
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.websocket import websocket_client
from mcp.types import CallToolResult, JSONRPCMessage, TextContent
from opentelemetry import trace
from pydantic import (
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    create_model,
)

from flock.core.logging.logging import get_logger
from flock.core.mcp.flock_mcp_tool_base import FlockMCPToolBase
from flock.core.mcp.types.mcp_callbacks import (
    FlockListRootsMCPCallback,
    FlockLoggingMCPCallback,
    FlockMessageHandlerMCPCallback,
    FlockSamplingMCPCallback,
)
from flock.core.mcp.types.mcp_types import Root

logger = get_logger("core.mcp.client_base")
tracer = trace.get_tracer(__name__)

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

_DEFAULT_EXCEPTION_TOOL_RESULT = CallToolResult(
    isError=True, content=[TextContent(type="text", text="Tool call failed.")]
)

MCPClientInitFunction = Callable[
    ...,
    AbstractAsyncContextManager[
        tuple[
            MemoryObjectReceiveStream[JSONRPCMessage | Exception],
            MemoryObjectSendStream[JSONRPCMessage],
        ]
    ],
]


class ServerParameters(BaseModel):
    """Base Type for server parameters."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class StdioServerParameters(_MCPStdioServerParameters, ServerParameters):
    """Base Type for Stdio Server params."""


class WebSocketServerParameters(ServerParameters):
    """Base Type for Websocket Server params."""

    url: str | AnyUrl = Field(..., description="Url the server listens at.")


class SseServerParameters(ServerParameters):
    """Base Type for SSE Server params."""

    url: str | AnyUrl = Field(..., description="The url the server listens at.")

    headers: dict[str, Any] | None = Field(
        default=None, description="Additional Headers to pass to the client."
    )

    timeout: float = Field(default=5, description="Http Timeout.")

    sse_read_timeout: float = Field(
        default=60 * 5,
        description="How long the client will wait before disconnecting from the server.",
    )


C = TypeVar("C", bound="FlockMCPClientBaseConfig")


class FlockMCPClientBaseConfig(BaseModel):
    """Base Configuration Class for MCP Clients.

    Each Client should implement their own
    config model by inheriting from this class.
    """

    server_name: str = Field(
        ..., description="Name of the server the client connects to."
    )

    transport_type: Literal["stdio", "websockets", "sse", "custom"] = Field(
        ..., description="Type of transport to use."
    )

    connection_paramters: ServerParameters = Field(
        ..., description="Connection parameters for the server."
    )

    max_retries: int = Field(
        default=3,
        description="How many times to attempt to establish the connection before giving up.",
    )

    mount_points: list[Root] | None = Field(
        default=None, description="Initial Mountpoints to operate under."
    )

    read_timeout_seconds: timedelta = Field(
        default_factory=lambda: timedelta(seconds=10),
        description="How many seconds to wait until a request times out.",
    )

    sampling_callback: FlockSamplingMCPCallback | None = Field(
        default=None,
        description="Callback for handling sampling requests from an external server.",
    )

    list_roots_callback: FlockListRootsMCPCallback | None = Field(
        default=None,
        description="Callback for list/roots requests from an external server.",
    )

    logging_callback: FlockLoggingMCPCallback | None = Field(
        default=None,
        description="Callback for handling logging events sent by a server.",
    )

    message_handler: FlockMessageHandlerMCPCallback | None = Field(
        default=None,
        description="Callback for handling messages not covered by other callbacks.",
    )

    tool_cache_max_size: float = Field(
        default=100, description="Maximum number of items in the Tool Cache."
    )

    tool_cache_max_ttl: float = Field(
        default=60,
        description="Max TTL for items in the tool cache in seconds.",
    )

    resource_contents_cache_max_size: float = Field(
        default=10,
        description="Maximum number of entries in the Resource Contents cache.",
    )

    resource_contents_cache_max_ttl: float = Field(
        default=60 * 5,
        description="Maximum number of items in the Resource Contents cache.",
    )

    resource_list_cache_max_size: float = Field(
        default=10,
        description="Maximum number of entries in the Resource List Cache.",
    )

    resource_list_cache_max_ttl: float = Field(
        default=100,
        description="Maximum TTL for entries in the Resource List Cache.",
    )

    tool_result_cache_max_size: float = Field(
        default=1000,
        description="Maximum number of entries in the Tool Result Cache.",
    )

    tool_result_cache_max_ttl: float = Field(
        default=20,
        description="Maximum TTL in seconds for entries in the Tool Result Cache.",
    )

    server_logging_level: LoggingLevel = Field(
        default="error",
        description="The logging level for the logging events of the remote server.",
    )

    roots_enabled: bool = Field(
        default=False,
        description="Whether or not Roots feature is enabled for this client.",
    )

    sampling_enabled: bool = Field(
        default=False,
        description="Whether or not Sampling Feature is enabled for this client.",
    )

    tools_enabled: bool = Field(
        default=False,
        description="Whether or not the Tools feature is enabled for this client.",
    )

    prompts_enabled: bool = Field(
        default=False,
        description="Whether or not the Prompts feature is enabled for this client.",
    )

    @classmethod
    def with_fields(cls: type[C], **field_definitions) -> type[C]:
        """Create a new config class with additional fields."""
        return create_model(
            f"Dynamic{cls.__name__}", __base__=cls, **field_definitions
        )


class FlockMCPClientBase(BaseModel, ABC):
    """Wrapper for mcp ClientSession.

    Class will attempt to re-establish connection if possible.
    If connection establishment fails after max_retries, then
    `has_error` will be set to true and `error_message` will
    contain the details of the exception.
    """

    # --- Properties ---
    config: FlockMCPClientBaseConfig = Field(
        ..., description="The config for this client instance."
    )

    tool_cache: TTLCache | None = Field(
        default=None,
        exclude=True,
        description="Cache for tools. Excluded from Serialization.",
    )

    tool_result_cache: TTLCache | None = Field(
        default=None,
        exclude=True,
        description="Cache for the result of tool call. Excluded from Serialization.",
    )

    resource_contents_cache: TTLCache | None = Field(
        default=None,
        exclude=True,
        description="Cache for resource contents. Excluded from Serialization.",
    )

    resource_list_cache: TTLCache | None = Field(
        default=None,
        exclude=True,
        description="Cache for Resource Lists. Excluded from Serialization.",
    )

    client_session: ClientSession | None = Field(
        default=None, exclude=True, description="ClientSession Reference."
    )

    connected_server_capabilities: ServerCapabilities | None = Field(
        default=None,
        exclude=True,
        description="Capabilities of the connected server.",
    )

    lock: Lock = Field(
        default_factory=Lock,
        exclude=True,
        description="Global lock for the client.",
    )

    session_stack: AsyncExitStack = Field(
        default_factory=AsyncExitStack,
        exclude=True,
        description="Internal AsyncExitStack for session.",
    )

    # Auto-reconnect proxy
    class _SessionProxy:
        def __init__(self, client: "FlockMCPClientBase"):
            self._client = client

        def __getattr__(self, name: str):
            # return an async function that auto-reconnects, then calls through.
            async def _method(*args, **kwargs):
                client = self._client
                cfg = client.config
                max_tries = cfg.max_retries or 1
                base_delay = 0.1

                for attempt in range(1, max_tries + 2):
                    await client._ensure_connected()
                    try:
                        # delegate the real session
                        return await getattr(client.client_session, name)(
                            *args, **kwargs
                        )
                    except McpError as e:
                        # only retry on a transport timeout
                        if e.error.code == httpx.codes.REQUEST_TIMEOUT:
                            kind = "timeout"
                        else:
                            # application-level MCP error -> give up immediately
                            logger.error(
                                f"MCP error in session.{name}: {e.error}"
                            )
                            return None
                    except (BrokenPipeError, ClosedResourceError) as e:
                        kind = type(e).__name__
                    except Exception as e:
                        # anything else is treated as transport failure
                        kind = type(e).__name__

                    # no more retries
                    if attempt > max_tries:
                        logger.error(
                            f"Session.{name} failed after {max_tries} retries ({kind}); giving up."
                        )
                        try:
                            await client.disconnect()
                        except Exception as e:
                            logger.warning(
                                f"Error tearing down stale session: {e}"
                            )
                        return None

                    # otherwise log + tear down + back off
                    logger.warning(
                        f"Session.{name} attempt {attempt}/{max_tries} failed. ({kind}). Reconnecting."
                    )
                    try:
                        await client.disconnect()
                        await client._connect()
                    except Exception as e:
                        logger.error(f"Reconnect failed: {e}")

                    # Exponential backoff + 10% jitter
                    delay = base_delay ** (2 ** (attempt - 1))
                    delay += random.uniform(0, delay * 0.1)
                    await asyncio.sleep(delay)

            return _method

    def __init__(
        self,
        config: FlockMCPClientBaseConfig,
        lock: Lock | None = None,
        tool_cache: TTLCache | None = None,
        tool_result_cache: TTLCache | None = None,
        resource_contents_cache: TTLCache | None = None,
        resource_list_cache: TTLCache | None = None,
        client_session: ClientSession | None = None,
        connected_server_capabilities: ServerCapabilities | None = None,
        session_stack: AsyncExitStack = AsyncExitStack(),
        **kwargs,
    ):
        """Init function."""
        lock = lock or Lock()
        super().__init__(
            config=config,
            lock=lock,
            tool_cache=tool_cache,
            tool_result_cache=tool_result_cache,
            resource_contents_cache=resource_contents_cache,
            resource_list_cache=resource_list_cache,
            client_session=client_session,
            connected_server_capabilities=connected_server_capabilities,
            session_stack=session_stack,
            **kwargs,
        )

        if not self.tool_cache:
            self.tool_cache = TTLCache(
                maxsize=self.config.tool_cache_max_size,
                ttl=self.config.tool_cache_max_ttl,
            )

        # set up the caches
        if not self.tool_result_cache:
            self.tool_cache = TTLCache(
                maxsize=self.config.tool_result_cache_max_size,
                ttl=self.config.tool_result_cache_max_ttl,
            )

        if not self.resource_contents_cache:
            self.resource_contents_cache = TTLCache(
                maxsize=self.config.resource_contents_cache_max_size,
                ttl=self.config.resource_contents_cache_max_ttl,
            )
        if not self.resource_list_cache:
            self.resource_list_cache = TTLCache(
                maxsize=self.config.resource_list_cache_max_size,
                ttl=self.config.resource_list_cache_max_ttl,
            )

    @property
    def session(self) -> _SessionProxy:
        """Always-connected proxy for client_session methods.

        Usage: await self.client_session.call_tool(...), await self.client_session.list_tools(...)
        """
        return self._SessionProxy(self)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
    )

    # --- Abstract methods / class methods ---
    @abstractmethod
    def get_init_function(self) -> MCPClientInitFunction:
        """Define which intialization function should be used to establish a read, and write stream between the client and the server.

        The returned function MUST Return an AbstractAsyncContextManager which
        returns a Tuple of MemoryObjectReceiveStream[JSONRPCMessage | Exception] and MemoryObjectWriteStream[JSONRPCMessage]

        in the case of stdio clients this is the function `stdio_client` (in mcp.client.stdio)
        in the case of sse clients this is the function `sse_client` (in mcp.client.sse)
        in the case of websocket clients this is the function `websocket_client` in (mcp.client.ws)
        """
        pass

    # --- Public methods ---
    async def get_tools(
        self,
        agent_id: str,
        run_id: str,
    ) -> list[FlockMCPToolBase]:
        """Gets a list of available tools from the server."""

        @cached(cache=self.tool_cache)
        async def _get_tools_cached(
            agent_id: str,
            run_id: str,
        ) -> list[FlockMCPToolBase]:
            if not self.config.tools_enabled:
                return []

            async def _get_tools_internal() -> list[FlockMCPToolBase]:
                response: ListToolsResult = await self.session.list_tools()
                flock_tools = []

                for tool in response.tools:
                    converted_tool = FlockMCPToolBase.from_mcp_tool(
                        tool,
                        agent_id=agent_id,
                        run_id=run_id,
                    )
                    if converted_tool:
                        flock_tools.append(converted_tool)
                return flock_tools

            return await _get_tools_internal()

        return await _get_tools_cached(agent_id=agent_id, run_id=run_id)

    async def call_tool(
        self, agent_id: str, run_id: str, name: str, arguments: dict[str, Any]
    ) -> CallToolResult:
        """Call a tool via the MCP Protocol on the client's server."""

        @cached(cache=self.tool_result_cache)
        async def _call_tool_cached(
            agent_id: str, run_id: str, name: str, arguments: dict[str, Any]
        ) -> CallToolResult:
            async def _call_tool_internal(
                name: str, arguments: dict[str, Any]
            ) -> CallToolResult:
                logger.debug(
                    f"Calling tool '{name}' with arguments {arguments}"
                )
                return await self.session.call_tool(
                    name=name,
                    arguments=arguments,
                )

            return await _call_tool_internal(name=name, arguments=arguments)

        return await _call_tool_cached(
            agent_id=agent_id, run_id=run_id, name=name, arguments=arguments
        )

    async def get_server_name(self) -> str:
        """Return the server_name.

        Uses a lock under the hood.
        """
        async with self.lock:
            return self.config.server_name

    async def get_roots(self) -> list[Root] | None:
        """Get the currently set roots of the client.

        Locks under the hood.
        """
        async with self.lock:
            return self.config.mount_points

    async def set_roots(self, new_roots: list[Root]) -> None:
        """Set the current roots of the client.

        Locks under the hood.
        """
        async with self.lock:
            self.config.mount_points = new_roots
            if self.session:
                await self.client_session.send_roots_list_changed()

    async def invalidate_tool_cache(self) -> None:
        """Invalidate the entries in the tool cache."""
        logger.debug(
            f"Invalidating tool_cache for server '{self.config.server_name}'"
        )
        async with self.lock:
            if self.tool_cache:
                self.tool_cache.clear()
                logger.debug(
                    f"Invalidated tool_cache for server '{self.config.server_name}'"
                )

    async def invalidate_resource_list_cache(self) -> None:
        """Invalidate the entries in the resource list cache."""
        logger.debug(
            f"Invalidating resource_list_cache for server '{self.config.server_name}'"
        )
        async with self.lock:
            if self.resource_list_cache:
                self.resource_list_cache.clear()
                logger.debug(
                    f"Invalidated resource_list_cache for server '{self.config.server_name}'"
                )

    async def invalidate_resource_contents_cache(self) -> None:
        """Invalidate the entries in the resource contents cache."""
        logger.debug(
            f"Invalidating resource_contents_cache for server '{self.config.server_name}'."
        )
        async with self.lock:
            if self.resource_contents_cache:
                self.resource_contents_cache.clear()
                logger.debug(
                    f"Invalidated resource_contents_cache for server '{self.config.server_name}'"
                )

    async def invalidate_resource_contents_cache_entry(self, key: str) -> None:
        """Invalidate a single entry in the resource contents cache."""
        logger.debug(
            f"Attempting to clear entry with key: {key} from resource_contents_cache for server '{self.config.server_name}'"
        )
        async with self.lock:
            if self.resource_contents_cache:
                try:
                    self.resource_contents_cache.pop(key, None)
                    logger.debug(
                        f"Cleared entry with key {key} from resource_contents_cache for server '{self.config.server_name}'"
                    )
                except Exception as e:
                    logger.debug(
                        f"No entry for key {key} found in resource_contents_cache for server '{self.config.server_name}'. Ignoring. (Exception was: {e})"
                    )
                    return  # do nothing

    async def disconnect(self) -> None:
        """If previously connected via `self._connect()`, tear it down."""
        async with self.lock:
            if self.session_stack:
                # manually __aexit__
                await self.session_stack.aclose()
                self.session_stack = None
                self.client_session = None

    # --- Private Methods ---
    async def _create_session(self) -> None:
        """Create and hol onto a single ClientSession + ExitStack."""
        stack = AsyncExitStack()
        await stack.__aenter__()

        server_params = self.config.connection_paramters
        gen_fn = (
            self.get_init_function()
            or {
                "stdio": stdio_client,
                "sse": sse_client,
                "websockets": websocket_client,
            }[self.config.transport_type]
        )

        read, write = await stack.enter_async_context(gen_fn(server_params))

        session = await stack.enter_async_context(
            ClientSession(
                read_stream=read,
                write_stream=write,
                read_timeout_seconds=self.config.read_timeout_seconds,
                list_roots_callback=self.config.list_roots_callback,
                message_handler=self.config.message_handler,
                sampling_callback=self.config.sampling_callback,
            )
        )

        # store for reuse
        self.session_stack = stack
        self.client_session = session

    async def _connect(self, retries: int | None = None) -> ClientSession:
        """Connect to an MCP Server and set self.client_session to ClientSession.

        Establish the transport and keep it open.
        """
        async with self.lock:
            # if already connected, return it
            if self.client_session:
                return self.client_session

            else:
                await self._create_session()

        await self._perform_initial_handshake()
        return self.client_session

    async def _perform_initial_handshake(self) -> None:
        """Tell the server who we are, what capabilities we have, and what roots we're interested in."""
        async with self.lock:
            # 1) do the LSP-style initialize handshake
            logger.debug(
                f"Performing intialize handshake with server '{self.config.server_name}'"
            )
            init: InitializeResult = await self.client_session.initialize()

            self.connected_server_capabilities = init

            init_report = f"""
            Server Init Handshake completed Server '{self.config.server_name}'
            Lists the following Capabilities:

            - Protocol Version: {init.protocolVersion}
            - Instructions: {init.instructions or "No specific Instructions"}
            - MCP Implementation:
                - Name: {init.serverInfo.name}
                - Version: {init.serverInfo.version}
            - Capabilities:
                {init.capabilities}
            """

            logger.debug(init_report)

            # 2) if we already know our current roots, notify the server
            #    so that it will follow up with a ListRootsRequest
            if self.config.mount_points:
                await self.client_session.send_roots_list_changed()

            # 3) Tell the server, what logging level we would like to use
            try:
                await self.client_session.set_logging_level(
                    level=self.config.server_logging_level
                )
            except McpError as e:
                logger.warning(
                    f"Trying to set logging level for server '{self.config.server_name}' resulted in Exception: {e}"
                )

    async def _ensure_connected(self) -> None:
        # if we've never connected, then connect.
        if not self.client_session:
            await self._connect()
            return

        # otherwise, ping and reconnect on error
        try:
            await self.client_session.send_ping()
        except Exception as e:
            logger.warning(
                f"Session to '{self.config.server_name}' died, reconnecting. Exception was: {e}"
            )
            await self.disconnect()
            await self._connect()

    async def _get_client_session(self) -> ClientSession:
        """Lazily start one session and reuse it forever (until closed)."""
        async with self.lock:
            if self.client_session is None:
                await self._create_session()

        return self.client_session
