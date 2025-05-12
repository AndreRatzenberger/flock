"""Wrapper Class for a mcp ClientSession Object"""

from abc import ABC, abstractmethod
from asyncio import Lock
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from datetime import timedelta
from typing import Annotated, Any, Callable, Generator, List, Literal, Tuple, Type, Union
from cachetools import TTLCache, cached
from mcp import ClientSession, InitializeResult, ListToolsResult
from mcp import StdioServerParameters as _MCPStdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from mcp.client.websocket import websocket_client
from mcp.types import CallToolResult, TextContent, JSONRPCMessage
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, UrlConstraints
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

from flock.core.logging.logging import get_logger
from opentelemetry import trace


from flock.core.mcp.flock_mcp_tool_base import FlockMCPToolBase
from flock.core.mcp.types.mcp_callbacks import FlockLoggingMCPCallback, default_flock_mcp_list_roots_callback_factory, default_flock_mcp_message_handler_callback_factory, default_flock_mcp_logging_callback_factory, default_flock_mcp_sampling_callback_factory
from flock.core.mcp.util.decorators import mcp_error_handler


logger = get_logger("core.mcp.client_base")
tracer = trace.get_tracer(__name__)

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


class ServerParameters():
    """
    Base Type for server parameters.
    """


class StdioServerParameters(_MCPStdioServerParameters, ServerParameters):
    """
    Base Type for Stdio Server params.
    """


class WebSocketServerParameters(BaseModel, ServerParameters):
    """
    Base Type for Websocket Server params.
    """

    url: str | AnyUrl = Field(
        ...,
        description="Url the server listens at."
    )


class SseServerParameters(BaseModel, ServerParameters):
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


class FlockMCPClientBase(BaseModel, ABC):
    """
    Wrapper for mcp ClientSession.
    Class will attempt to re-establish connection if possible.
    If connection establishment fails after max_retries, then
    `has_error` will be set to true and `error_message` will
    contain the details of the exception.
    """

    server_name: str = Field(
        ...,
        description="Name of the server the client connects to."
    )

    transport_type: Literal["stdio", "websockets", "sse", "custom"] = Field(
        ...,
        description="Type of transport to use."
    )

    connection_parameters: ServerParameters = Field(
        ...,
        description="Connection parameters for the server."
    )

    is_busy: bool = Field(
        default=False,
        description="Whether or not this client is currently handling a request",
        exclude=True,
    )

    is_alive: bool = Field(
        default=True,
        description="Whether or not this ClientSession is still alive",
        exclude=True
    )

    has_error: bool = Field(
        default=False,
        description="Whether or not the client has errored.",
        exclude=True,
    )

    error_message: str | None = Field(
        default=None,
        description="The messsage of the exception that occurred within the client."
    )

    max_retries: int = Field(
        default=3,
        description="How many times to attempt to establish the connection before giving up."
    )

    lock: Lock = Field(
        default_factory=Lock,
        description="Lock for the client. Enabling it to act as a mutex."
    )

    current_roots: list[Annotated[AnyUrl, UrlConstraints(host_required=False)]] | list[str] | None = Field(
        default=None,
        description="Roots to operate under."
    )

    read_timeout_seconds: timedelta = Field(
        ...,
        description="How many seconds to wait until the request times out."
    )

    sampling_callback: Callable[..., Any] | None = Field(
        default=None,
        description="Callback for sampling requests from external server. Take a look at mcp/client/session.py for how it is used."
    )

    list_roots_callback: Callable[..., Any] | None = Field(
        default=None,
        description="Callback for list_roots requests from external server."
    )

    logging_callback: FlockLoggingMCPCallback | None = Field(
        default=None,
        description="Callback for logging purposes."
    )

    message_handler: Callable[..., Any] | None = Field(
        default=None,
        description="Message Handler Callback."
    )

    tool_cache_max_size: float = Field(
        default=100,
        description="Maximum number of items in the Tool Cache"
    )

    tool_cache_max_ttl: float = Field(
        default=60,
        description="Max TTL for the tools cache in Seconds."
    )

    resource_contents_cache_max_size: float = Field(
        default=10,
        description="Maximum number of items in the Resource Contents cache."
    )

    resource_contents_cache_max_ttl: float = Field(
        default=60*5,
        description="Max TTL in seconds for the resource contents cache."
    )

    resource_list_cache_max_size: float = Field(
        default=1000,
        description="Maximum number of items in the Resource List Cache."
    )

    resource_list_cache_max_ttl: float = Field(
        default=10,
        description="Max TTL in seconds for the Resource List Cache."
    )

    tool_result_cache_max_size: float = Field(
        default=1000,
        description="Maximum number of items in the Tool Result Cache"
    )

    tool_result_cache_max_ttl: float = Field(
        default=10,
        description="Maximum TTL in seconds for the Tool Result Cache"
    )

    _tool_cache: TTLCache

    _tool_result_cache: TTLCache

    _resource_contents_cache: TTLCache

    _resource_list_cache: TTLCache

    _client_context_manager: Any | None

    _client_session: ClientSession

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def __init__(
        self,
        server_name: str,
        transport_type: Literal["stdio", "websockets", "sse"],
        tool_cache_max_size: float,
        tool_cache_max_ttl: float,
        tool_result_cache_max_ttl: float,
        tool_result_cache_max_size: float,
        resource_list_cache_max_size: float,
        resource_list_cache_max_ttl: float,
        resource_contents_cache_max_size: float,
        resource_contents_cache_max_ttl: float,
        read_timeout_seconds: timedelta,
        connection_parameters: ServerParameters,
        is_busy: bool = False,
        is_alive: bool = True,
        has_error: bool = False,
        error_message: str | None = None,
        sampling_callback: Callable[..., Any] | None = None,
        list_roots_callback: Callable[..., Any] | None = None,
        logging_callback: Callable[..., Any] | None = None,
        message_handler: Callable[..., Any] | None = None,
        max_retries: int = 3,
        lock: Lock = Lock(),
        current_roots: list[Annotated[AnyUrl, UrlConstraints(
            host_required=False)]] | list[str] | None = None,
        **kwargs,
    ):
        super().__init__(
            server_name=server_name,
            transport_type=transport_type,
            tool_cache_max_size=tool_cache_max_size,
            tool_cache_max_ttl=tool_cache_max_ttl,
            tool_result_cache_max_size=tool_result_cache_max_size,
            tool_result_cache_max_ttl=tool_result_cache_max_ttl,
            resource_list_cache_max_size=resource_list_cache_max_size,
            resource_list_cache_max_ttl=resource_list_cache_max_ttl,
            resource_contents_cache_max_ttl=resource_contents_cache_max_ttl,
            resource_contents_cache_max_size=resource_contents_cache_max_size,
            resource_contents_cache_max_ttl=resource_contents_cache_max_ttl,
            read_timeout_seconds=read_timeout_seconds,
            connection_parameters=connection_parameters,
            is_busy=is_busy,
            is_alive=is_alive,
            has_error=has_error,
            error_message=error_message,
            max_retries=max_retries,
            lock=lock,
            current_roots=current_roots,
            sampling_callback=sampling_callback,
            list_roots_callback=list_roots_callback,
            logging_callback=logging_callback,
            message_handler=message_handler,
            **kwargs,
        )

        if not self._client_context_manager:
            # Will be set during the connect cycle.
            self._client_context_manager = None

        # After initialization, assign the default_callbacks if they have not been specified.
        if not self.logging_callback:
            self.logging_callback = default_flock_mcp_logging_callback_factory(
                self.server_name, logger)

        if not self.message_handler:
            self.message_handler = default_flock_mcp_message_handler_callback_factory()

        if not self.sampling_callback:
            self.sampling_callback = default_flock_mcp_sampling_callback_factory()

        if not self.list_roots_callback:
            self.list_roots_callback = default_flock_mcp_list_roots_callback_factory()

        if not self._tool_cache:
            self._tool_cache = TTLCache(
                maxsize=self.tool_cache_max_size,
                ttl=self.tool_cache_max_ttl,
            )

        # set up the caches
        if not self._tool_result_cache:
            self._tool_cache = TTLCache(
                maxsize=self.tool_result_cache_max_size,
                ttl=self.tool_result_cache_max_ttl,
            )

        if not self._resource_contents_cache:
            self._resource_contents_cache = TTLCache(
                maxsize=self.resource_contents_cache_max_size,
                ttl=self.resource_contents_cache_max_ttl,
            )
        if not self._resource_list_cache:
            self._resource_list_cache = TTLCache(
                maxsize=self.resource_list_cache_max_size,
                ttl=self.resource_list_cache_max_ttl,
            )

    async def get_server_name(self) -> str:
        async with self.lock:
            return self.server_name

    async def get_is_busy(self) -> bool:
        async with self.lock:
            return self.is_busy

    async def set_is_busy(self, status: bool) -> None:
        async with self.lock:
            self.is_busy = status

    async def get_is_alive(self) -> bool:
        async with self.lock:
            return self.is_alive

    async def set_is_alive(self, status: bool) -> None:
        async with self.lock:
            self.is_alive = status

    async def get_has_error(self) -> bool:
        async with self.lock:
            return self.has_error

    async def set_has_error(self, status: bool) -> None:
        async with self.lock:
            self.has_error = status

    async def get_error_message(self) -> str | None:
        async with self.lock:
            return self.error_message

    async def set_error_message(self, message: str | None) -> None:
        async with self.lock:
            self.error_message = message

    async def invalidate_tool_cache(self) -> None:
        logger.debug(
            f"Invalidating tool_cache for server '{self.server_name}'")
        async with self.lock:
            if self._tool_cache:
                self._tool_cache.clear()
                logger.debug(
                    f"Invalidated tool_cache for server '{self.server_name}'")

    async def invalidate_resource_list_cache(self) -> None:
        logger.debug(
            f"Invalidating resource_list_cache for server '{self.server_name}'")
        async with self.lock:
            if self._resource_list_cache:
                self._resource_list_cache.clear()
                logger.debug(
                    f"Invalidated resource_list_cache for server '{self.server_name}'")

    async def invalidate_resource_contents_cache(self) -> None:
        logger.debug(
            f"Invalidating resource_contents_cache for server '{self.server_name}'.")
        async with self.lock:
            if self._resource_contents_cache:
                self._resource_contents_cache.clear()
                logger.debug(
                    f"Invalidated resource_contents_cache for server '{self.server_name}'")

    async def invalidate_resource_contents_cache_entry(self, key: str) -> None:
        logger.debug(
            f"Attempting to clear entry with key: {key} from resource_contents_cache for server '{self.server_name}'")
        async with self.lock:
            if self._resource_contents_cache:
                try:
                    self._resource_contents_cache.pop(key, None)
                    logger.debug(
                        f"Cleared entry with key {key} from resource_contents_cache for server '{self.server_name}'")
                except:
                    logger.debug(
                        f"No entry for key {key} found in resource_contents_cache for server '{self.server_name}'. Ignoring.")
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
            if self._client_session:
                return self._client_session

            # establish the generator_function
            generator_manager = self.get_init_function()

            # grab the context manager
            self._client_context_manager = self.get_client_session(
                generator_manager=generator_manager)

            # manually __aenter__ the context manager
            self._client_session = await self._client_context_manager.__aenter__()
            return self._client_session

    @mcp_error_handler(default_return=None, logger=logger)
    async def communicate_client_state(self) -> None:
        """
        tell the server who we are, what capabilities we have,
        and what roots we're interested in.
        """

        # 1) do the LSP-style initialize handshake
        logger.debug(
            f"Performing intialize handshake with server '{self.server_name}'")
        init: InitializeResult = await self._client_session.initialize()

        init_report = f"""
        Server Init Handshake completed Server '{self.server_name}' 
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
        if self.current_roots:
            await self._client_session.send_roots_list_changed()

    async def _ensure_connected(self) -> None:
        # if we've never connected, then connect.
        if not self._client_session:
            await self.connect()
            return

        # otherwise, ping and reconnect on error
        try:
            await self._client_session.send_ping()
        except Exception:
            logger.warning(
                f"Session to '{self.server_name}' died, reconnecting.")
            await self.disconnect()
            await self.connect()

    async def disconnect(self) -> None:
        """
        If previously connected via `self.connect()`, tear it down.
        """
        async with self.lock:
            if not self._client_context_manager:
                return

            # manually __aexit__
            await self._client_context_manager.__aexit__(None, None, None)
            self._client_context_manager = None
            self._client_session = None

    @asynccontextmanager
    async def get_client_session(
        self,
        generator_manager: Union[
            Type[stdio_client],
            Type[sse_client],
            Type[websocket_client],
            MCPClientInitFunction
        ] | None,
    ):
        """generator_manager must be one of mcp.client... stdio_client, sse_client, or websocket_client"""

        server_params = self.connection_parameters

        if self.transport_type == "stdio" and not isinstance(server_params, StdioServerParameters):
            raise TypeError(
                f"Server Parameters for a stdio-client must be of type {type(StdioServerParameters)} got {type(server_params)}"
            )
        if self.transport_type == "sse" and not isinstance(server_params, SseServerParameters):
            raise TypeError(
                f"Server Parameters for a sse-client must be of type {type(SseServerParameters)} got {type(server_params)}"
            )
        if self.transport_type == "websockets" and not isinstance(server_params, WebSocketServerParameters):
            raise TypeError(
                f"Server Parameters for a websocket-client must be of type {type(WebSocketServerParameters)} got {type(server_params)}"
            )

        read: MemoryObjectReceiveStream[JSONRPCMessage |
                                        Exception] | None = None
        write: MemoryObjectSendStream[JSONRPCMessage] | None = None
        match self.transport_type:
            case "stdio":
                generator_manager = stdio_client if generator_manager is None else generator_manager
                async with self.lock:
                    stack = AsyncExitStack()
                    await stack.__aenter__()
                    read, write = await stack.enter_async_context(generator_manager(server_params))
            case "sse":
                generator_manager = sse_client if generator_manager is None else generator_manager
                async with self.lock:
                    stack = AsyncExitStack()
                    await stack.__aenter__()
                    read, write = await stack.enter_async_context(generator_manager(
                        url=server_params.url,
                        headers=server_params.headers,
                        timeout=server_params.timeout,
                        sse_read_timeout=server_params.sse_read_timeout,
                    ))
            case "websockets":
                generator_manager = websocket_client if generator_manager is None else generator_manager
                async with self.lock:
                    stack = AsyncExitStack()
                    await stack.__aenter__()
                    read, write = await stack.enter_async_context(generator_manager(
                        url=server_params.url
                    ))
            case "custom":
                async with self.lock:
                    stack = AsyncExitStack()
                    await stack.__aenter__()
                    read, write = await stack.enter_async_context(generator_manager(server_params))

        if not read or not write:
            raise Exception(
                f"Could not establish underlying read and write stream for client session. Transport type matched neither 'stdio', 'sse', 'websockets', or 'custom'. Or passed generator_manager was unable to establish streams.")

        client_session = await stack.enter_async_context(
            ClientSession(
                read_stream=read,
                write_stream=write,
                read_timeout_seconds=self.read_timeout_seconds,
                list_roots_callback=self.logging_callback,
                logging_callback=self.logging_callback,
                messsage_handler=self.message_handler,
            )
        )

        try:
            yield client_session
        finally:
            await stack.aclose()

    async def get_tools(self, agent_id: str, run_id: str) -> List[FlockMCPToolBase]:
        """
        Gets a list of available tools from the server.
        """

        @cached(cache=self._tool_cache)
        async def _get_tools_cached(agent_id: str, run_id: str) -> List[FlockMCPToolBase]:
            initialized = False
            async with self.lock:
                if self._client_session:
                    initialized = True

            if not initialized:
                # Underlying client session has not yet been initialized
                # This should not happen in practice, but it is good to be cautious
                logger.warning(
                    f"Underlying session for connection to server has not been initialized"
                )
                result = await self.connect(retries=self.max_retries)
                if isinstance(result, Exception):
                    # This means that the connection failed.
                    raise Exception(
                        f"Connection for client for server '{self.server_name}' under agent {agent_id} (run_id: {run_id}) failed with exception: {result}"
                    )
                logger.debug(
                    f"Underlying session for server '{self.server_name}' for agent {agent_id} in run {run_id} has been established.")
                initialized = True

            @mcp_error_handler(default_return=[], logger=logger)
            async def _get_tools_internal() -> List[FlockMCPToolBase]:

                async with self.lock:
                    response: ListToolsResult = await self._client_session.list_tools()
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

        @cached(cache=self._tool_result_cache)
        async def _call_tool_cached(agent_id: str, run_id: str, name: str, arguments: dict[str, Any]) -> CallToolResult:
            initialized = False
            async with self.lock:
                if self._client_session:
                    initialized = True

            if not initialized:
                # Underlying client session has not yet been initialized
                # This should not happen in practice, but its good to be catious
                logger.warning(
                    f"Underlying session for connection to server '{self.server_name}' has not been initialized."
                )
                await self.connect(retries=self.max_retries)
                logger.debug(
                    f"Underlying session for client for server '{self.server_name}' has been initialized."
                )
                initialized = True

            @mcp_error_handler(default_return=_DEFAULT_EXCEPTION_TOOL_RESULT, logger=logger)
            async def _call_tool_internal(name: str, arguments: dict[str, Any]) -> CallToolResult:
                async with self.lock:
                    logger.debug(
                        f"Calling tool '{name}' with arguments {arguments}"
                    )
                    return await self._client_session.call_tool(name=name, arguments=arguments)

            return await _call_tool_internal(name=name, arguments=arguments)

        return await _call_tool_cached(agent_id=agent_id, run_id=run_id, name=name, arguments=arguments)
