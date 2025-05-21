"""Types for Flock's MCP functionality."""

from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from typing import Any

from anyio.streams.memory import (
    MemoryObjectReceiveStream,
    MemoryObjectSendStream,
)
from mcp import (
    ClientSession,
    CreateMessageResult,
    StdioServerParameters as _MCPStdioServerParameters,
)
from mcp.shared.context import RequestContext
from mcp.shared.session import RequestResponder
from mcp.types import (
    CancelledNotification as _MCPCancelledNotification,
    ClientResult,
    CreateMessageRequestParams,
    ErrorData,
    JSONRPCMessage,
    ListRootsResult,
    LoggingMessageNotification as _MCPLoggingMessageNotification,
    LoggingMessageNotificationParams as _MCPLoggingMessageNotificationParams,
    ProgressNotification as _MCPProgressNotification,
    PromptListChangedNotification as _MCPPromptListChangedNotification,
    ResourceListChangedNotification as _MCPResourceListChangedNotification,
    ResourceUpdatedNotification as _MCPResourceUpdateNotification,
    Root as _MCPRoot,
    ServerNotification as _MCPServerNotification,
    ServerRequest,
    ToolListChangedNotification as _MCPToolListChangedNotification,
)
from pydantic import AnyUrl, BaseModel, ConfigDict, Field

from flock.core.mcp.util.helpers import get_default_env


class ServerNotification(_MCPServerNotification):
    """A notification message sent by the server side."""


class CancelledNotification(_MCPCancelledNotification):
    """Notification, which can be sent bei either side to indicate that it is cancelling a previously issued request."""


class ProgressNotification(_MCPProgressNotification):
    """An out-of band notification used to inform the receiver of a progress update for a long-running request."""


class LoggingMessageNotification(_MCPLoggingMessageNotification):
    """A notification message sent by the server side containing a logging message."""


class ResourceUpdatedNotification(_MCPResourceUpdateNotification):
    """A notification message sent by the server side informing a client about a change in a resource."""


class ResourceListChangedNotification(_MCPResourceListChangedNotification):
    """A notification message sent by the server side informing a client about a change in the list of resources."""


class ToolListChangedNotification(_MCPToolListChangedNotification):
    """A notification message sent by the server side informing a client about a change in the offered tools."""


class PromptListChangedNotification(_MCPPromptListChangedNotification):
    """A notification message sent by the server side informing a client about a change in the list of offered Prompts."""


class FlockLoggingMessageNotificationParams(
    _MCPLoggingMessageNotificationParams
):
    """Parameters contained within a Logging Message Notification."""


class MCPRoot(_MCPRoot):
    """Wrapper for mcp.types.Root."""


class ServerParameters(BaseModel):
    """Base Type for server parameters."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class StdioServerParameters(_MCPStdioServerParameters, ServerParameters):
    """Base Type for Stdio Server parameters."""

    env: dict[str, str] | None = Field(
        default_factory=get_default_env,
        description="Environment for the MCP Server.",
    )


class WebsocketServerParameters(ServerParameters):
    """Base Type for Websocket Server params."""

    url: str | AnyUrl = Field(..., description="Url the server listens at.")


class SseServerParameters(ServerParameters):
    """Base Type for SSE Server params."""

    url: str | AnyUrl = Field(..., description="The url the server listens at.")

    headers: dict[str, Any] | None = Field(
        default=None, description="Additional Headers to pass to the client."
    )

    timeout: float | int = Field(default=5, description="Http Timeout")

    sse_read_timeout: float | int = Field(
        default=60 * 5,
        description="How long the client will wait before disconnecting from the server.",
    )


MCPCLientInitFunction = Callable[
    ...,
    AbstractAsyncContextManager[
        tuple[
            MemoryObjectReceiveStream[JSONRPCMessage | Exception],
            MemoryObjectSendStream[JSONRPCMessage],
        ]
    ],
]


FlockSamplingMCPCallback = Callable[
    [RequestContext, CreateMessageRequestParams],
    Awaitable[CreateMessageResult | ErrorData],
]


FlockListRootsMCPCallback = Callable[
    [RequestContext[ClientSession, Any]],
    Awaitable[ListRootsResult | ErrorData],
]

FlockLoggingMCPCallback = Callable[
    [FlockLoggingMessageNotificationParams],
    Awaitable[None],
]

FlockMessageHandlerMCPCallback = Callable[
    [
        RequestResponder[ServerRequest, ClientResult]
        | ServerNotification
        | Exception
    ],
    Awaitable[None],
]
