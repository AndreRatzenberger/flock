"""Wrapper Class for a mcp ClientSession Object"""

from asyncio import Lock
from typing import Annotated
import anyio
from mcp import ClientNotification, ClientSession, ListToolsResult
from pydantic import BaseModel, Field, AnyUrl, UrlConstraints


from flock.core.logging.logging import get_logger
from opentelemetry import trace

from flock.core.mcp.flock_mcp_tool import FlockMCPTool

logger = get_logger("mcp_client")
tracer = trace.get_tracer(__name__)


class FlockMCPCLient(BaseModel):
    """
    Wrapper for mcp ClientSession. 
    Class will attempt to re-establish connection if possible.
    If connection establishment fails after max_retries, then
    `has_error` will be set to true and `error_message` will
    contain the details of the exception.
    """

    client_session: ClientSession | None = Field(
        default=None,
        description="Internally managed client session."
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

    model_config = {
        "arbitrary_types_allowed": True,
    }

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
    async def get_tools(self) -> list[FlockMCPTool] | None:
        """Get available tools from the server."""
        # TODO: Tools list changed callback tomorrow (or rather on tuesday).
        async with self.lock:
            tools: list[FlockMCPTool] = []
            try:
                logger.debug(f"Retrieving tools through client {self}")
                if self.client_session:
                    results: ListToolsResult = await self.client_session.list_tools()

                    # Convert the tools into a list of FlockMCPTools
                    if results.tools:
                        for t in results.tools:
                            converted_tool = FlockMCPTool.try_from_mcp_tool(t)
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
            except Exception as e:
                logger.error(
                    f"Unexpected Excpetion occurred during setting of roots for client {self}: {e}")
                self.is_alive = False
                self.has_error = True
                self.error_message = str(e)
                # TODO: Close connection immediately
