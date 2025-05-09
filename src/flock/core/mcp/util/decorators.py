import functools
import asyncio
from typing import Any, Awaitable, Callable, ParamSpec, TypeVar
import anyio
import httpx

from mcp import McpError
from mcp.types import CallToolResult, TextContent

from flock.core.logging.logging import FlockLogger, get_logger
from flock.core.mcp.types.mcp_protocols import MCPClientProto

R = TypeVar("R")
P = ParamSpec("P")


# a default logger if no logger is being passed
_DEFAULT_LOGGER = get_logger("mcp.error_handler_decorator")


def mcp_error_handler(default_return: R, logger: FlockLogger = _DEFAULT_LOGGER) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """
    Decorator to catch MCP, transport and other exceptions,
    update client state via the protocol and return a safe default.
    """

    def deco(fn: Callable[P, Awaitable[R]]):
        @functools.wraps(fn)
        async def wrapper(self: MCPClientProto, *args: P.args, **kwargs: P.kwargs) -> R:
            try:
                return await fn(self, *args, **kwargs)
            except anyio.ClosedResourceError as e:
                msg = f"Stream closed: {e}"
                logger.error(msg)
                await self.set_is_alive(False)
                await self.set_has_error(True)
                await self.set_error_message(msg)
                return default_return
            except anyio.BrokenResourceError as e:
                msg = f"Connection broken: {e}"
                logger.error(msg)
                await self.set_is_alive(False)
                await self.set_has_error(True)
                await self.set_error_message(msg)
                return default_return
            except httpx.HTTPError as e:
                msg = f"HTTP error: {e}"
                logger.error(msg)
                await self.set_is_alive(False)
                await self.set_has_error(True)
                await self.set_error_message(msg)
                return default_return
            except McpError as me:
                code = me.error.code
                # treat 429/425 as transient
                if code in (httpx.codes.TOO_MANY_REQUESTS, httpx.codes.TOO_EARLY):
                    logger.warning(f"Transient MCP {{code}}: {{me}}")
                    await self.set_is_alive(True)
                    await self.set_has_error(False)
                    await self.set_error_message(None)
                else:
                    msg = f"MCP error {code}: {me}"
                    logger.error(msg)
                    await self.set_is_alive(False)
                    await self.set_has_error(True)
                    await self.set_error_message(msg)
                return default_return
            except Exception as e:
                # General exception handling
                msg = f"Exception ocurred: {e}"
                logger.error(msg)
                await self.set_is_alive(False)
                await self.set_has_error(True)
                await self.set_error_message(msg)
                return default_return
        return wrapper
    return deco
