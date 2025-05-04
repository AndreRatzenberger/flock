"""Wrapper Class for a mcp ClientSession Object"""

from asyncio import Lock
from typing import Annotated
from mcp import ClientNotification, ClientSession
from pydantic import BaseModel, Field, AnyUrl, UrlConstraints


from flock.core.logging.logging import get_logger
from opentelemetry import trace

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

    async def set_roots(self, roots: list[Annotated[AnyUrl, UrlConstraints(host_required=False)]] | list[str] | None) -> None:
        """Sets the roots for this Client."""
        # TODO: Callback notifcation handling tomorrow.
        async with self.lock:
            try:
                logger.debug(f"Setting roots for client {self} to: {roots}")
                if self.client_session:
                    # Set the roots
                    self.current_roots = roots
                    self.client_session.send_notification(
                        ClientNotification()
                    )
                    result = await self.client_session.send_roots_list_changed()
