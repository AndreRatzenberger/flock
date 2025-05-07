"""Wrapper Class for a mcp ClientSession Object"""

from abc import ABC
from asyncio import Lock
import asyncio
from contextlib import AsyncExitStack
from datetime import timedelta
from typing import Annotated, Any, Literal, Type
import anyio
import httpx
from mcp import ClientNotification, ClientSession, ListToolsResult, McpError, StdioServerParameters
from mcp.types import CallToolResult
from pydantic import BaseModel, Field, AnyUrl, UrlConstraints

from mcp.client.session import SamplingFnT, ListRootsFnT, LoggingFnT, MessageHandlerFnT

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


class FlockMCPClientBase(BaseModel, ABC):
    """
    Wrapper for mcp ClientSession.
    Class will attempt to re-establish connection if possible.
    If connection establishment fails after max_retries, then
    `has_error` will be set to true and `error_message` will
    contain the details of the exception.
    """

    client_session: ClientSession | None = Field(
        default=None,
        exclude=True,
        description="Internally managed client session."
    )

    transport_type: Literal["stdio", "websockets", "http"] = Field(
        ...,
        description="Type of transport to use."
    )

    connection_parameters: StdioServerParameters,

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

    sampling_callback: SamplingFnT | None = Field(
        default=None,
        description="Callback for sampling requests from external server. Take a look at mcp/client/session.py for how it is used."
    )

    list_roots_callback: ListRootsFnT | None = Field(
        default=None,
        description="Callback for list_roots requests from external server."
    )

    logging_callback: LoggingFnT | None = Field(
        default=None,
        description="Callback for logging purposes."
    )

    message_handler: MessageHandlerFnT | None = Field(
        default=None,
        description="Message Handler Callback."
    )

    model_config = {
        "arbitrary_types_allowed": True,
    }

    async def connect(self) -> None:
        """
        Connects to the client.
        """
        pass

    async def close(self) -> None:
        """Closes the connection and cleans up. Placeholder for now."""
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

    # --- MCP Functionality ---
    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None, read_timeout_seconds: timedelta | None = None):
        """
        Wrapper which allows agents to call MCPTools.

        This method should NEVER be called directly.
        Rather, the results from `get_tools` are converted
        into callables with type Callable[..., Any] which
        wrap around this method.
        The agent only sees callables which look like native
        code.

        Conversion of types is handled by these wrapper methods
        rather than the call_tool method.
        """
        async with self.lock:
            try:
                logger.debug(f"Calling tool {name} with args: {arguments}")
                timeout = read_timeout_seconds
                if timeout is None:
                    timeout = self.read_timeout_seconds
                result: CallToolResult = await self.client_session.call_tool(name=name, arguments=arguments, read_timeout_seconds=timeout)

                return result
            except Exception as e:
                logger.error(
                    f"Unexpected exception ocurred during call_tool call for {self}: {e}")
                self.is_alive = False
                self.has_error = True
                self.error_message = str(e)

    async def get_tools(self) -> list[FlockMCPToolBase] | None:
        # TODO: Caching
        """Get available tools from the server."""
        # TODO: Tools list changed callback tomorrow (or rather on tuesday).
        async with self.lock:
            tools: list[FlockMCPToolBase] = []
            try:
                logger.debug(f"Retrieving tools through client {self}")
                if self.client_session:
                    results: ListToolsResult = await self.client_session.list_tools()

                    # Convert the tools into a list of FlockMCPTools
                    if results.tools:
                        for t in results.tools:
                            converted_tool = FlockMCPToolBase.try_from_mcp_tool(
                                t)
                            if converted_tool:
                                tools.append(converted_tool)

                return tools
            except anyio.ClosedResourceError as closed_on_our_side:
                logger.error(
                    f"Exception ocurred during list_tools call for client {self} (Stream closed on our side): {closed_on_our_side}")
                self.is_alive = False
                self.has_error = True
                self.error_message = str(e)
                # TODO: close session. (or reopen it?)
            except anyio.BrokenResourceError as closed_on_remote_side:
                logger.error(
                    f"Exception ocurred during list_tools call for client {self} (Stream closed by remote): {closed_on_remote_side}")
                self.is_alive = False
                self.has_error = True
                self.error_message = str(e)
                # TODO: close session (or reopen it?)
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
                    f"Unexpected Exception occurred while attempting to retrieve tools with client {self}: {e}")
                self.is_alive = False
                self.has_error = True
                self.error_message = str(e)
                # TODO: close session. (or reopen it?)

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
