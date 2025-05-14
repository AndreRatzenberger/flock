"""Wrapper Class for a mcp ClientSession Object"""

from abc import ABC, abstractmethod
from asyncio import Lock
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from datetime import timedelta
from typing import Annotated, Any, Callable, Generator, List, Literal, Tuple, Type, TypeVar, Union
from cachetools import TTLCache, cached
from mcp import ClientSession, InitializeResult, ListToolsResult, McpError, ServerCapabilities
from mcp import StdioServerParameters as _MCPStdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from mcp.client.websocket import websocket_client
from mcp.types import CallToolResult, TextContent, JSONRPCMessage
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, UrlConstraints, create_model
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

from flock.core.logging.logging import get_logger
from opentelemetry import trace


from flock.core.mcp.flock_mcp_tool_base import FlockMCPToolBase
from flock.core.mcp.types.mcp_callbacks import FlockListRootsMCPCallback, FlockLoggingMCPCallback, FlockMessageHandlerMCPCallback, FlockSamplingMCPCallback, default_flock_mcp_list_roots_callback_factory, default_flock_mcp_message_handler_callback_factory, default_flock_mcp_logging_callback_factory, default_flock_mcp_sampling_callback_factory
from flock.core.mcp.types.mcp_types import Root
from flock.core.mcp.util.decorators import mcp_error_handler


logger = get_logger("core.mcp.client_base")
tracer = trace.get_tracer(__name__)

LoggingLevel = Literal[
    "debug", "info", "notice", "warning", "error", "critical", "alert", "emergency"
]

_DEFAULT_EXCEPTION_TOOL_RESULT = CallToolResult(
    isError=True,
    content=[
        TextContent(
            type="text",
            text="Tool call failed."
        )
    ]
)

MCPClientInitFunction = Callable[
    ...,
    AbstractAsyncContextManager[
        Tuple[
            MemoryObjectReceiveStream[JSONRPCMessage | Exception],
            MemoryObjectSendStream[JSONRPCMessage]
        ]
    ]
]


class ServerParameters(BaseModel):
    """
    Base Type for server parameters.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class StdioServerParameters(_MCPStdioServerParameters, ServerParameters):
    """
    Base Type for Stdio Server params.
    """


class WebSocketServerParameters(ServerParameters):
    """
    Base Type for Websocket Server params.
    """

    url: str | AnyUrl = Field(
        ...,
        description="Url the server listens at."
    )


class SseServerParameters(ServerParameters):
    """
    Base Type for SSE Server params
    """

    url: str | AnyUrl = Field(
        ...,
        description="The url the server listens at."
    )

    headers: dict[str, Any] | None = Field(
        default=None,
        description="Additional Headers to pass to the client."
    )

    timeout: float = Field(
        default=5,
        description="Http Timeout."
    )

    sse_read_timeout: float = Field(
        default=60*5,
        description="How long the client will wait before disconnecting from the server."
    )


C = TypeVar("C", bound="FlockMCPClientBaseConfig")


class FlockMCPClientBaseConfig(BaseModel):
    """"
    Base Configuration Class for MCP Clients.

    Each Client should implement their own
    config model by inheriting from this class.

    """

    server_name: str = Field(
        ...,
        description="Name of the server the client connects to."
    )

    transport_type: Literal["stdio", "websockets", "sse", "custom"] = Field(
        ...,
        description="Type of transport to use."
    )

    connection_paramters: ServerParameters = Field(
        ...,
        description="Connection parameters for the server."
    )

    max_retries: int = Field(
        default=3,
        description="How many times to attempt to establish the connection before giving up."
    )

    mount_points: list[Root] | None = Field(
        default=None,
        description="Initial Mountpoints to operate under."
    )

    read_timeout_seconds: timedelta = Field(
        default_factory=lambda: timedelta(seconds=10),
        description="How many seconds to wait until a request times out."
    )

    sampling_callback: FlockSamplingMCPCallback | None = Field(
        default=None,
        description="Callback for handling sampling requests from an external server."
    )

    list_roots_callback: FlockListRootsMCPCallback | None = Field(
        default=None,
        description="Callback for list/roots requests from an external server."
    )

    logging_callback: FlockLoggingMCPCallback | None = Field(
        default=None,
        description="Callback for handling logging events sent by a server."
    )

    message_handler: FlockMessageHandlerMCPCallback | None = Field(
        default=None,
        description="Callback for handling messages not covered by other callbacks."
    )

    tool_cache_max_size: float = Field(
        default=100,
        description="Maximum number of items in the Tool Cache."
    )

    tool_cache_max_ttl: float = Field(
        default=60,
        description="Max TTL for items in the tool cache in seconds."
    )

    resource_contents_cache_max_size: float = Field(
        default=10,
        description="Maximum number of entries in the Resource Contents cache."
    )

    resource_contents_cache_max_ttl: float = Field(
        default=60*5,
        description="Maximum number of items in the Resource Contents cache."
    )

    resource_list_cache_max_size: float = Field(
        default=10,
        description="Maximum number of entries in the Resource List Cache."
    )

    resource_list_cache_max_ttl: float = Field(
        default=100,
        description="Maximum TTL for entries in the Resource List Cache."
    )

    tool_result_cache_max_size: float = Field(
        default=1000,
        description="Maximum number of entries in the Tool Result Cache."
    )

    tool_result_cache_max_ttl: float = Field(
        default=20,
        description="Maximum TTL in seconds for entries in the Tool Result Cache."
    )

    server_logging_level: LoggingLevel = Field(
        default="error",
        description="The logging level for the logging events of the remote server."
    )

    roots_enabled: bool = Field(
        default=False,
        description="Whether or not Roots feature is enabled for this client."
    )

    sampling_enabled: bool = Field(
        default=False,
        description="Whether or not Sampling Feature is enabled for this client."
    )

    tools_enabled: bool = Field(
        default=False,
        description="Whether or not the Tools feature is enabled for this client."
    )

    prompts_enabled: bool = Field(
        default=False,
        description="Whether or not the Prompts feature is enabled for this client."
    )

    @classmethod
    def with_fields(cls: type[C], **field_definitions) -> type[C]:
        """Create a new config class with additional fields."""
        return create_model(
            f"Dynamic{cls.__name__}", __base__=cls, **field_definitions
        )


class FlockMCPClientBase(BaseModel, ABC):
    """
    Wrapper for mcp ClientSession.
    Class will attempt to re-establish connection if possible.
    If connection establishment fails after max_retries, then
    `has_error` will be set to true and `error_message` will
    contain the details of the exception.
    """

    config: FlockMCPClientBaseConfig = Field(
        ...,
        description="The config for this client instance."
    )

    tool_cache: TTLCache | None = Field(
        default=None,
        exclude=True,
        description="Cache for tools. Excluded from Serialization."
    )

    tool_result_cache: TTLCache | None = Field(
        default=None,
        exclude=True,
        description="Cache for the result of tool call. Excluded from Serialization."
    )

    resource_contents_cache: TTLCache | None = Field(
        default=None,
        exclude=True,
        description="Cache for resource contents. Excluded from Serialization."
    )

    resource_list_cache: TTLCache | None = Field(
        default=None,
        exclude=True,
        description="Cache for Resource Lists. Excluded from Serialization."
    )

    client_session: ClientSession | None = Field(
        default=None,
        exclude=True,
        description="ClientSession Reference."
    )

    connected_server_capabilities: ServerCapabilities | None = Field(
        default=None,
        exclude=True,
        description="Capabilities of the connected server."
    )

    lock: Lock = Field(
        default_factory=Lock,
        exclude=True,
        description="Global lock for the client."
    )

    session_stack: AsyncExitStack = Field(
        default_factory=AsyncExitStack,
        exclude=True,
        description="Internal AsyncExitStack for session."
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
    )

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
        ** kwargs,
    ):
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

    async def _create_session(self) -> None:
        """Create and hold onto a single ClientSession + ExitStack"""
        stack = AsyncExitStack()
        await stack.__aenter__()

        server_params = self.config.connection_paramters
        gen_fn = self.get_init_function() or {
            "stdio": stdio_client,
            "sse": sse_client,
            "websockets": websocket_client,
        }[self.config.transport_type]

        read, write = await stack.enter_async_context(gen_fn(server_params))

        session = await stack.enter_async_context(
            ClientSession(
                read_stream=read,
                write_stream=write,
                read_timeout_seconds=self.config.read_timeout_seconds,
                list_roots_callback=self.config.list_roots_callback,
                logging_callback=self.config.logging_callback,
                message_handler=self.config.message_handler,
                sampling_callback=self.config.sampling_callback,
            )
        )
        # store for reuse
        self.session_stack = stack
        self.client_session = session

    async def get_server_name(self) -> str:
        async with self.lock:
            return self.config.server_name

    async def get_roots(self) -> list[Root] | None:
        async with self.lock:
            return self.config.mount_points

    async def set_roots(self, new_roots: list[Root]) -> None:
        async with self.lock:
            self.config.mount_points = new_roots
            if self.client_session:
                await self.client_session.send_roots_list_changed()

    async def invalidate_tool_cache(self) -> None:
        logger.debug(
            f"Invalidating tool_cache for server '{self.config.server_name}'")
        async with self.lock:
            if self.tool_cache:
                self.tool_cache.clear()
                logger.debug(
                    f"Invalidated tool_cache for server '{self.config.server_name}'")

    async def invalidate_resource_list_cache(self) -> None:
        logger.debug(
            f"Invalidating resource_list_cache for server '{self.config.server_name}'")
        async with self.lock:
            if self.resource_list_cache:
                self.resource_list_cache.clear()
                logger.debug(
                    f"Invalidated resource_list_cache for server '{self.config.server_name}'")

    async def invalidate_resource_contents_cache(self) -> None:
        logger.debug(
            f"Invalidating resource_contents_cache for server '{self.config.server_name}'.")
        async with self.lock:
            if self.resource_contents_cache:
                self.resource_contents_cache.clear()
                logger.debug(
                    f"Invalidated resource_contents_cache for server '{self.config.server_name}'")

    async def invalidate_resource_contents_cache_entry(self, key: str) -> None:
        logger.debug(
            f"Attempting to clear entry with key: {key} from resource_contents_cache for server '{self.config.server_name}'")
        async with self.lock:
            if self.resource_contents_cache:
                try:
                    self.resource_contents_cache.pop(key, None)
                    logger.debug(
                        f"Cleared entry with key {key} from resource_contents_cache for server '{self.config.server_name}'")
                except:
                    logger.debug(
                        f"No entry for key {key} found in resource_contents_cache for server '{self.config.server_name}'. Ignoring.")
                    return  # do nothing

    @abstractmethod
    def get_init_function(self) -> MCPClientInitFunction:
        """
        Define which intialization function should be used to establish
        a read, and write stream between the client and the server.

        The returned function MUST Return an AbstractAsyncContextManager which
        returns a Tuple of MemoryObjectReceiveStream[JSONRPCMessage | Exception] and MemoryObjectWriteStream[JSONRPCMessage]

        in the case of stdio clients this is the function `stdio_client` (in mcp.client.stdio)
        in the case of sse clients this is the function `sse_client` (in mcp.client.sse)
        in the case of websocket clients this is the function `websocket_client` in (mcp.client.ws)
        """
        pass

    async def connect(self, retries: int | None = None) -> ClientSession:
        """
        Connect to an MCP Server and set self.client_session to ClientSession

        Establish the transport and keep it open.
        """

        async with self.lock:
            # if already connected, return it
            if self.client_session:
                return self.client_session

            else:
                await self._create_session()

        await self.perform_initial_handshake()
        return self.client_session

    async def perform_initial_handshake(self) -> None:
        """
        tell the server who we are, what capabilities we have,
        and what roots we're interested in.
        """

        async with self.lock:

            # 1) do the LSP-style initialize handshake
            logger.debug(
                f"Performing intialize handshake with server '{self.config.server_name}'")
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
                await self.client_session.set_logging_level(level=self.config.server_logging_level)
            except McpError as e:
                logger.warning(
                    f"Trying to set logging level for server '{self.config.server_name}' resulted in Exception: {e}")

    async def ensure_connected(self) -> None:
        # if we've never connected, then connect.
        if not self.client_session:
            await self.connect()
            return

        # otherwise, ping and reconnect on error
        try:
            await self.client_session.send_ping()
        except Exception as e:
            logger.warning(
                f"Session to '{self.config.server_name}' died, reconnecting. Exception was: {e}")
            await self.disconnect()
            await self.connect()

    async def disconnect(self) -> None:
        """
        If previously connected via `self.connect()`, tear it down.
        """
        async with self.lock:
            if self.session_stack:
                # manually __aexit__
                await self.session_stack.aclose()
                self.session_stack = None
                self.client_session = None

    async def get_client_session(self) -> ClientSession:
        """Lazily start one session and reuse it forever (until closed)"""
        async with self.lock:
            if self.client_session is None:
                await self._create_session()

        return self.client_session

    async def get_tools(self, agent_id: str, run_id: str) -> List[FlockMCPToolBase]:
        """
        Gets a list of available tools from the server.
        """

        @cached(cache=self.tool_cache)
        async def _get_tools_cached(agent_id: str, run_id: str) -> List[FlockMCPToolBase]:

            async with self.lock:
                if not self.config.tools_enabled:
                    return []

            initialized = False
            async with self.lock:
                if self.client_session:
                    initialized = True

            if not initialized:
                # Underlying client session has not yet been initialized
                # This should not happen in practice, but it is good to be cautious
                logger.warning(
                    f"Underlying session for connection to server has not been initialized"
                )
                result = await self.connect(retries=self.config.max_retries)
                if isinstance(result, Exception):
                    # This means that the connection failed.
                    raise Exception(
                        f"Connection for client for server '{self.config.server_name}' under agent {agent_id} (run_id: {run_id}) failed with exception: {result}"
                    )
                logger.debug(
                    f"Underlying session for server '{self.config.server_name}' for agent {agent_id} in run {run_id} has been established.")
                initialized = True

            async def _get_tools_internal() -> List[FlockMCPToolBase]:

                if not self.config.tools_enabled:
                    return []

                async with self.lock:
                    response: ListToolsResult = await self.client_session.list_tools()
                    flock_tools = []

                    for tool in response.tools:
                        converted_tool = FlockMCPToolBase.from_mcp_tool(
                            tool, agent_id=agent_id, run_id=run_id)
                        if converted_tool:
                            flock_tools.append(converted_tool)

                    return flock_tools

            return await _get_tools_internal()

        return await _get_tools_cached(agent_id=agent_id, run_id=run_id)

    async def call_tool(self, agent_id: str, run_id: str, name: str, arguments: dict[str, Any]) -> CallToolResult:
        """
        Calls a tool on the respective mcp server.
        """

        @cached(cache=self.tool_result_cache)
        async def _call_tool_cached(agent_id: str, run_id: str, name: str, arguments: dict[str, Any]) -> CallToolResult:
            initialized = False
            async with self.lock:
                if self.client_session:
                    initialized = True

            if not initialized:
                # Underlying client session has not yet been initialized
                # This should not happen in practice, but its good to be catious
                logger.warning(
                    f"Underlying session for connection to server '{self.config.server_name}' has not been initialized."
                )
                await self.connect(retries=self.config.max_retries)
                logger.debug(
                    f"Underlying session for client for server '{self.config.server_name}' has been initialized."
                )
                initialized = True

            async def _call_tool_internal(name: str, arguments: dict[str, Any]) -> CallToolResult:
                async with self.lock:
                    logger.debug(
                        f"Calling tool '{name}' with arguments {arguments}"
                    )
                    return await self.client_session.call_tool(name=name, arguments=arguments)

            return await _call_tool_internal(name=name, arguments=arguments)

        return await _call_tool_cached(agent_id=agent_id, run_id=run_id, name=name, arguments=arguments)
