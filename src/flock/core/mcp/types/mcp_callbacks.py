"""Default Callback functions and factories for MCP Clients"""
from typing import Any, Literal, Protocol
import anyio
import anyio.lowlevel
from mcp import ClientSession, LoggingLevel
from mcp.types import LoggingMessageNotificationParams as _MCPParams
from mcp.types import ServerNotification as _MCPServerNotification
from mcp.types import CancelledNotification as _MCPCancelledNotification
from mcp.types import ProgressNotification as _MCPProgressNotification
from mcp.types import LoggingMessageNotification as _MCPLoggingMessageNotification
from mcp.types import ResourceUpdatedNotification as _MCPResourceUpdateNotification
from mcp.types import ResourceListChangedNotification as _MCPResourceListChangedNotification
from mcp.types import ToolListChangedNotification as _MCPToolListChangedNotification
from mcp.types import PromptListChangedNotification as _MCPPromptListChangedNotification
from mcp.types import ListRootsResult, LoggingMessageNotificationParams, LoggingMessageNotification
from mcp.shared.context import RequestContext
from pydantic import BaseModel, ConfigDict, Field


from flock.core.logging.logging import FlockLogger, get_logger
from flock.core.mcp.flock_mcp_client_base import FlockMCPClientBase

default_logging_callback_logger = get_logger("core.mcp.callback.logging")
default_sampling_callback_logger = get_logger("core.mcp.callback.sampling")
default_list_roots_callback_logger = get_logger("core.mcp.callback.sampling")
default_message_handler_logger = get_logger("core.mcp.callback.message")

# --- Types ---
# TODO: Continue tomorrow (12.05.2025)


class ServerNotification(_MCPServerNotification):
    """"""


class CancelledNotification(_MCPCancelledNotification):
    """
    This type of notification can be sent by either side to 
    indicate that it is cancelling a previously issued request.
    """


class ProgressNotification(_MCPProgressNotification):
    """
    An out-of-band notification used to inform the
    receiver of a progress update for a long-running
    request.

    See: https://modelcontextprotocol.io/specification/2025-03-26/basic/utilities/progress
    """


class LoggingMessageNotification(_MCPLoggingMessageNotification):
    """"""


class ResourceUpdatedNotification(_MCPResourceUpdateNotification):
    """"""


class ResourceListChangedNotification(_MCPResourceListChangedNotification):
    """"""


class ToolListChangedNotification(_MCPToolListChangedNotification):
    """"""


class PromptListChangedNotificiation(_MCPPromptListChangedNotification):
    """"""


class FlockLoggingMessageNotificationParams(LoggingMessageNotificationParams):
    """
    Parameters for logging Message Notifications.
    """

    level: LoggingLevel = Field(
        ...,
        description="The severity of this log message."
    )

    logger: str | None = Field(
        default=None,
        description="An Optional name of the logger issuing this message."
    )

    data: Any = Field(
        ...,
        description="The data to be logged, such as a string message or an object. Any JSON serializable type is allowed here."
    )

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
    )


class FlockLoggingMCPCallback(Protocol):
    """
    Defines the type signature for a MCP logging callback.
    """

    async def __call__(
        self,
        params: FlockLoggingMessageNotificationParams,
    ) -> None: ...

# --- Default Callback Factories ---


def default_flock_mcp_message_handler_callback_factory(associated_client: FlockMCPClientBase, logger: FlockLogger) -> None:
    """
    Creates a fallback for handling incoming messages.

    NOTE:
      Incoming Messages differ from incoming requests.
      Requests can do things like list roots, create_messages etc.

      While Incoming Messages mainly consist of miscellanious information 
      sent by the server.
    """
    logger_to_use = logger if logger else default_message_handler_logger
    server_name = associated_client.server_name

    async def handle_incoming_server_notification(n: ServerNotification) -> None:
        """
        Process an incoming server notification.
        """

        logging_callback = default_flock_mcp_logging_callback_factory(
            server_name=server_name, logger=logger_to_use)

        # React to the different types of notifications.
        match n.root:
            case ResourceListChangedNotification():
                await handle_resource_list_changed_notification(n=n.root, logger_to_use=logger_to_use, associated_client=associated_client)

            case ResourceUpdatedNotification():
                await handle_resource_update_notification(n=n.root, logger_to_use=logger_to_use, associated_client=associated_client)

            case LoggingMessageNotification():
                # this is simply passed on to the logging callback
                await logging_callback(params=n.root)

            case ProgressNotification():
                await handle_progress_notification(n=n.root, logger_to_use=logger_to_use, server_name=server_name)

            case CancelledNotification():
                await handle_cancellation_notification(n=n.root, logger_to_use=logger_to_use, server_name=server_name)

    async def handle_incoming_message(req: RequestResponder[ServerRequest, ClientResult]) -> None:
        """
        Process an incoming server message.
        """

    async def default_message_handler(req: RequestResponder[ServerRequest, ClientResult] | ServerNotification | Exception) -> None:

        if isinstance(req, Exception):
            await handle_incoming_exception(req)
        elif isinstance(req, ServerNotification):
            await handle_incoming_server_notification(req)
        elif isinstance(req, RequestResponder[ServerRequest, ClientResult]):
            await handle_incoming_message(req)

            # Check for cancellation.
        await anyio.lowlevel.checkpoint()
        pass

    return default_message_handler


def default_flock_mcp_sampling_callback_factory(server_name: str, logger: FlockLogger, sampling_agent: Any, enabled: bool = False) -> None:
    pass


def default_flock_mcp_list_roots_callback_factory() -> None:
    pass


def default_flock_mcp_logging_callback_factory(server_name: str, logger: FlockLogger | None = None) -> FlockLoggingMCPCallback:
    """
    Creates a fallback for a logging callback for a client.
    """

    logger_to_use = logger if logger else default_logging_callback_logger

    async def default_logging_callback(params: FlockLoggingMessageNotificationParams) -> None:
        """
        The default logging callback for all flock mcp clients.
        """
        level = params.level
        method = logger_to_use.debug
        logger_name = params.logger if params.logger else "unknown_remote_logger"
        metadata = params.meta or {}

        str_level = "DEBUG: "
        prefix = f"Message from Remote MCP Logger '{logger_name}' for server '{server_name}': "

        match level:
            case "info":
                method = logger_to_use.info
                str_level = "INFO: "
            case "notice":
                method = logger_to_use.info
                str_level = "NOTICE: "
            case "alert":
                method = logger.warning
                str_level = "WARNING: "
            case "critical":
                method = logger.error
                str_level = "CRITICAL: "
            case "error":
                method = logger.error
                str_level = "ERROR: "
            case "emergency":
                method = logger.error
                str_level = "EMERGENCY! "
            case _:
                pass

        full_msg = f"{prefix}{str_level}{params.data} (Meta Data: {metadata})"
        method(full_msg)

    return default_logging_callback


# --- Helper functions ---
async def handle_incoming_exception(e: Exception, logger_to_use: FlockLogger, associated_client: FlockMCPClientBase) -> None:
    """
    Process an incoming exception Message.
    """

    server_name = await associated_client.server_name

    # For now, simply log it.
    logger_to_use.error(
        f"Encountered Exception while communicating with server: '{server_name}': {e}")


async def handle_progress_notification(n: ProgressNotification, logger_to_use: FlockLogger, server_name: str) -> None:

    params = n.params
    progress = params.progress
    total = params.total or "Unknown"
    progress_token = params.progressToken
    metadata = params.meta or {}

    message = f"PROGRESS_NOTIFICATION: Server '{server_name}' reports Progress: {progress}/{total}. (Token: {progress_token}) (Meta Data: {metadata})"

    logger_to_use.info(message)


async def handle_cancellation_notification(n: CancelledNotification, logger_to_use: FlockLogger, server_name: str) -> None:

    params = n.params
    request_id_to_cancel = params.requestId
    reason = params.reason or "no reason given"
    metadata = params.meta or {}

    message = f"CANCELLATION_REQUEST: Server '{server_name}' requests to cancel request with id: {request_id_to_cancel}. Reason: {reason}. (Metadata: {metadata})"

    logger_to_use.warning(message)


async def handle_resource_update_notification(n: ResourceUpdatedNotification, logger_to_use: FlockLogger, associated_client: FlockMCPClientBase) -> None:
    # This also means that the associated client needs to invalidate its resource_contents-cache at the entriy with the associated uri.

    params = n.params
    metadata = params.meta or {}
    uri = params.uri

    message = f"RESOURCE_UPDATE: Server '{associated_client.server_name}' reports change on resoure at: {uri}. (Meta Data: {metadata})"

    logger_to_use.info(message)

    await associated_client.invalidate_resource_contents_cache_entry(key=uri)


async def handle_resource_list_changed_notification(n: ResourceListChangedNotification, logger_to_use: FlockLogger, associated_client: FlockMCPClientBase) -> None:
    # This also means that the associated client needs to invalidate its resource cache.

    params = n.params or {}
    metadata = params.meta or {}

    message = f"RESOURCE_LIST_CHANGED: Server '{associated_client.server_name}' reports a change in their resource list: {metadata}"

    logger_to_use.info(message)
    await associated_client.invalidate_resource_list_cache()


async def handle_tool_list_changed_notification(n: ToolListChangedNotification, logger_to_use: FlockLogger, associated_client: FlockMCPClientBase) -> None:
    # This also means that the associated client needs to invalidate it's tool cache.

    params = n.params or {}
    metadata = params.meta or {}

    message = f"TOOLS_LIST_CHANGED: Server '{associated_client.server_name}' reports a change in their tools list: {metadata}. Resetting Tools Cache for associated clients."

    logger_to_use.info(message)
    await associated_client.invalidate_tool_cache()
