"""Wrapper Class for a mcp ClientSession Object"""

from abc import ABC, abstractmethod
from asyncio import Lock
import asyncio
from contextlib import AsyncExitStack
from datetime import timedelta
from typing import Annotated, Any, Callable, List, Literal
from mcp import ClientSession, InitializeResult, ListToolsResult, StdioServerParameters
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, UrlConstraints


from flock.core.logging.logging import get_logger
from opentelemetry import trace

from flock.core.mcp.flock_mcp_tool_base import FlockMCPToolBase
from flock.core.mcp.util.decorators import mcp_error_handler


logger = get_logger("mcp_client")
tracer = trace.get_tracer(__name__)


class WebSocketServerParameters(BaseModel):
    """
    Base Type for Websocket Server params.
    """

    url: str | AnyUrl = Field(
        ...,
        description="Url the server listens at."
    )


class SseServerParameters(BaseModel):
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

    client_session: ClientSession | None = Field(
        default=None,
        exclude=True,
        description="Internally managed client session."
    )

    transport_type: Literal["stdio", "websockets", "sse"] = Field(
        ...,
        description="Type of transport to use."
    )

    connection_parameters: StdioServerParameters | SseServerParameters | WebSocketServerParameters = Field(
        ...,
        description="Connection parameters for the server."
    )

    master_stack: AsyncExitStack | None = Field(
        default=None,
        exclude=True,
        description="Master async exit stack."
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

    # Used internally in the ClientSession Object like so:
    #     async def _received_request(
    #     self, responder: RequestResponder[types.ServerRequest, types.ClientResult]
    # ) -> None:
    #     ctx = RequestContext[ClientSession, Any](
    #         request_id=responder.request_id,
    #         meta=responder.request_meta,
    #         session=self,
    #         lifespan_context=None,
    #     )

    #     match responder.request.root:
    #         case types.CreateMessageRequest(params=params):
    #             with responder:
    #                 response = await self._sampling_callback(ctx, params)
    #                 client_response = ClientResponse.validate_python(response)
    #                 await responder.respond(client_response)

    #         case types.ListRootsRequest():
    #             with responder:
    #                 response = await self._list_roots_callback(ctx)
    #                 client_response = ClientResponse.validate_python(response)
    #                 await responder.respond(client_response)

    #         case types.PingRequest():
    #             with responder:
    #                 return await responder.respond(
    #                     types.ClientResult(root=types.EmptyResult())
    #                 )

    sampling_callback: Callable[..., Any] | None = Field(
        default=None,
        description="Callback for sampling requests from external server. Take a look at mcp/client/session.py for how it is used."
    )

    list_roots_callback: Callable[..., Any] | None = Field(
        default=None,
        description="Callback for list_roots requests from external server."
    )

    logging_callback: Callable[..., Any] | None = Field(
        default=None,
        description="Callback for logging purposes."
    )

    message_handler: Callable[..., Any] | None = Field(
        default=None,
        description="Message Handler Callback."
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @abstractmethod
    @mcp_error_handler(default_return=Exception("Client Intialization failed"), logger=logger)
    async def connect(self, retries: int | None = None) -> InitializeResult | None:
        """
        Connects to the client.

        Args:
            retries (int | None): Optional parameter for how often to retry the connection.
        """

        pass

    async def close(self) -> None:
        """
        Closes the connection and cleans up.
        """
        async with self.lock:
            if self.master_stack:
                await self.master_stack.aclose()
            self.is_alive = False
            self.is_busy = False
            self.has_error = False
            self.error_message = None
            self.client_session = None

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

    # TODO: caching
    @mcp_error_handler(default_return=[], logger=logger)
    async def get_tools(self) -> List[FlockMCPToolBase]:
        """
        Gets a list of available tools from the server.
        """
        # Check of the underlying client session has been initialized
        initialized = False
        async with self.lock:
            if self.client_session:
                initialized = True

        if not initialized:
            # Underlying client session has not yet been initialized
            # This should not happen in practice, but it is good to be cautious
            logger.warning(
                f"Underlying session for connection has not been initialized"
            )
            result = await self.connect(retries=self.max_retries)
            if isinstance(result, Exception):
                # This means that the connection failed.
                raise Exception(
                    f"Connection for client for server '{self.server_name}' failed with exception: {result}")
            logger.debug(f"Underlying session has been established.")
            initialized = True

        async with self.lock:
            response: ListToolsResult = await self.client_session.list_tools()

            flock_cmp_tools = []

            for tool in response.tools:
                converted_tool = FlockMCPToolBase.from_mcp_tool(tool)
                if converted_tool:
                    flock_cmp_tools.append(converted_tool)

            return flock_cmp_tools

    @mcp_error_handler(default_return=None, logger=logger)
    async def set_roots(self, roots: list[Annotated[AnyUrl, UrlConstraints(host_required=False)]] | list[str] | None) -> None:
        """
        Sets the roots for this client.
        """
        # TODO: Callback notification handling
        # TODO: Caching

        initialized = False
        async with self.lock:
            if self.client_session:
                initialized = True

        if not initialized:
            # Underlying client session has not yet been initialized
            # This should not happen in practice, but it is good to cautious
            logger.warning(
                f"Underlying session for connection has not been initialized")
            await self.connect(retries=self.max_retries)
            logger.debug(
                f"Underlying session for client for server '{self.server_name}' has been initialized")
            initialized = True

        async with self.lock:
            logger.debug(
                f"Setting roots for client for server '{self.server_name}' to: {roots}")
            self.current_roots = roots
            self.client_session.send_roots_list_changed()
            # TODO: handle roots/listChanged callback
            return
