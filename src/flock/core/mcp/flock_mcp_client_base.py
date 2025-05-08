"""Wrapper Class for a mcp ClientSession Object"""

from abc import ABC
from asyncio import Lock
import asyncio
from contextlib import AsyncExitStack
from datetime import timedelta
from typing import Annotated, Any, Callable, Literal, Type
import anyio
import httpx
from mcp import ClientNotification, ClientSession, InitializeResult, ListToolsResult, McpError, StdioServerParameters
from mcp.types import CallToolResult
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from mcp.client.websocket import websocket_client
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, UrlConstraints

from mcp.client.session import SamplingFnT, ListRootsFnT, LoggingFnT, MessageHandlerFnT

from dspy.primitives import Tool as DSPyTool

from flock.core.logging.logging import get_logger
from opentelemetry import trace

from flock.core.mcp.flock_mcp_tool_base import FlockMCPToolBase

# TODO: fine-grained error handling
# Standard JSON-RPC error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

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

    async def connect(self, retries: int | None) -> InitializeResult:
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
        pass

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
    async def get_tools(self) -> list[DSPyTool]:
        """
        Gets the list of available tools from the server.
        """
        # Check if underlying client session has been initialized.
        initialized = False
        async with self.lock:
            if self.client_session:
                initialized = True

        if not initialized:
            # Underlying client session has not yet been initialized.
            # This should never happen in practice, but it is good to be extra cautious
            logger.warning(
                f"Underlying session for connection has not been initialized.")
            try:
                await self.connect(retries=self.max_retries)
                logger.debug(f"Underlying session has been established.")
                initialized = True
            except Exception as e:
                logger.error(
                    f"Exception ocurred while initializing connection: {e}")
                self.is_alive = False
                self.has_error = True
                self.error_message = str(e)
                # Return an empty list as to not impede application flow.
                return []

        # Get the tools
        # Lock the client beforehand
        async with self.lock:
            try:
                response: ListToolsResult = await self.client_session.list_tools()

                flock_mcp_tools = []

                for tool in response.tools:
                    converted_tool = FlockMCPToolBase.try_from_mcp_tool(tool)
                    if converted_tool:
                        flock_mcp_tools.append(
                            converted_tool.convert_to_callable(connection_manager=self))

                return flock_mcp_tools
            except anyio.ClosedResourceError as closed_from_our_side:
                logger.error(
                    f"Exception ocurred during list/tools request to server '{self.server_name}' (Stream closed from our side): {closed_from_our_side}"
                )
                self.is_alive = False
                self.has_error = True
                self.error_message = str(closed_from_our_side)
                return []  # FIXME: More fine-grained return values.
            except anyio.BrokenResourceError as closed_from_remote_side:
                logger.error(
                    f"Exception ocurred during list/tools request to server '{self.server_name}' (Connection closed by remote): {closed_from_remote_side}"
                )
                self.is_alive = False
                self.has_error = True
                self.error_message = str(closed_from_remote_side)
                return []  # FIXME: More fine-grained return values.
            except McpError as mcp_error:
                if mcp_error.error.code == httpx.codes.TOO_MANY_REQUESTS or mcp_error.error.code == httpx.codes.TOO_EARLY:
                    # This means the server is simply receiving too many requests.
                    # But the client itself didn't crap its pants.
                    # This means, we can try again in the future.
                    # FIXME: Backoff-Logic. For now, we simply return an empty list of tools.
                    logger.warning(
                        f"Server '{self.server_name}' returned Code: {mcp_error.error.code}")
                    self.is_alive = True
                    self.has_error = False
                    self.error_message = None
                elif mcp_error.error.code == httpx.codes.REQUEST_TIMEOUT:
                    logger.error(
                        f"Call to list/tools for Server '{self.server_name}' timed out: {mcp_error}"
                    )
                    self.is_alive = False
                    self.has_error = True
                    self.error_message = str(mcp_error)
                return []  # FIXME: More fine-grained return values.
            except Exception as e:
                logger.error(
                    f"Unexpected Exception ocurred for list/tools request to server '{self.server_name}': {e}")
                self.is_alive = False
                self.has_error = True
                self.error_message = str(e)
                return []  # FIXME: More fine-grained return values.
            finally:
                # Release the lock no matter what.
                self.lock.release()

    async def set_roots(self, roots: list[Annotated[AnyUrl, UrlConstraints(host_required=False)]] | list[str] | None) -> None:
        """Sets the roots for this Client."""
        # TODO: Callback notifcation handling tomorrow (or rather on tuesday).
        # TODO: Caching
        async with self.lock:
            try:
                logger.debug(f"Setting roots for client {self} to: {roots}")
                if self.client_session:
                    # Set the roots
                    self.current_roots = roots
                    self.client_session.send_notification(
                        ClientNotification()
                    )
                    # The actual update will be handled by the list_roots_callback
                    await self.client_session.send_roots_list_changed()
                # else:
                    # TODO: Connection reinitialization. (IF needed.)
                    # await self.init_connection()
            except anyio.ClosedResourceError as closed_on_our_end:
                # This error indicates that the connection to the remote has been closed from our side.
                logger.error(
                    f"Exception Occurred during setting of roots for client {self} (Send stream closed): {closed_on_our_end}")
                self.is_alive = False
                self.has_error = True
                self.error_message = str(e)
            except anyio.BrokenResourceError as broken_on_remote_end:
                # This error indicates that the connection was terminated from the other side.
                logger.error(
                    f"Exception Occurred during setting of roots for client {self} (Remote closed send stream): {broken_on_remote_end}")
                self.is_alive = False
                self.has_error = True
                self.error_message = str(e)
            except McpError as mcp_error:
                if mcp_error.error.code == httpx.codes.REQUEST_TIMEOUT:
                    logger.error(
                        f"MCP Excpetion ocurred during list_tools call for client {self} (Request timed out.): {mcp_error}")
                    self.is_alive = True  # On Timeout we dare try again in the future
                    self.has_error = False
                    self.error_message = None
                elif mcp_error.error.code == httpx.codes.TOO_MANY_REQUESTS or mcp_error.error.code == httpx.codes.TOO_EARLY:
                    logger.error(
                        f"MCP Exception ocurred during list_tools call for client {self} (Too many requests or too early to call again): {mcp_error}"
                    )
                    self.is_alive = True
                    self.has_error = False
                    self.error_message = None
                else:
                    logger.error(
                        f"MCP Exception ocurred during list_tools call for client {self}: {mcp_error}")
                    self.is_alive = False
                    self.has_error = True
                    self.error_message = str(mcp_error)
            except Exception as e:
                logger.error(
                    f"Unexpected Excpetion occurred during setting of roots for client {self}: {e}")
                self.is_alive = False
                self.has_error = True
                self.error_message = str(e)
                # TODO: Close connection immediately
