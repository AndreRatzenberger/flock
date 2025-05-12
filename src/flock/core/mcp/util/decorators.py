import functools
import asyncio
import random
from typing import Any, Awaitable, Callable, ParamSpec, TypeVar
import anyio
import httpx
from anyio import BrokenResourceError, ClosedResourceError

from mcp import McpError
from mcp.types import CallToolResult, TextContent

from flock.core.logging.logging import FlockLogger, get_logger


R = TypeVar("R")
P = ParamSpec("P")


# a default logger if no logger is being passed
_DEFAULT_LOGGER = get_logger("core.mcp.decorator")


def mcp_error_handler(default_return: R, logger: FlockLogger = _DEFAULT_LOGGER) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """
    Wrap an async fn so that on any *transport*-level failure we:
    - disconnect + reconnect
    - retry with exponential backoff + jitter
    - give up after self.max_retries
    - finally return `default_return`
    """
    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            self = args[0]
            # how many times to retry on transport error
            max_tries = getattr(self, "max_retries", 1)
            base_delay = 0.1

            for attempt in range(1, max_tries + 2):
                # ensure that we've at least connected once
                await self._ensure_connected()
                try:
                    return await fn(*args, **kwargs)
                except McpError as e:
                    # only reconnect on a *timeout* error
                    if e.error.code == httpx.codes.REQUEST_TIMEOUT:
                        kind = "timeout"
                    else:
                        # application-level MCP error: don't retry
                        logger.error(f"MCP error in {fn.__name__}: {e.error}")
                        return default_return
                except (BrokenPipeError, ClosedResourceError) as e:
                    kind = type(e).__name__
                except Exception as e:
                    # any other unexpected exception: treat as transport
                    kind = type(e).__name__

                # if we get here, it was a transport-level failure we want to retry
                if attempt > max_tries:
                    logger.error(
                        f"{fn.__name__} failed after {attempt-1} retries ({kind}); giving up."
                    )
                    logger.debug(f"Killing off session")

                    # tear down the stale session so the next call reconnects
                    try:
                        await self.disconnect()
                    except Exception as e:
                        logger.warning(
                            f"Error while disconnecting stale session: {e}")
                    return default_return
                # otherwise, disconnect / reconnect + backoff
                logger.warning(
                    f"{fn.__name__} attempt {attempt}/{max_tries} failed ({kind}), reconnecting"
                )
                try:
                    await self.disconnect()
                    await self.connect()
                except Exception as conn_err:
                    logger.error(f"Reconnect failed: {conn_err}")

                # exponential backoff + jitter
                delay = base_delay * (2 ** (attempt - 1))
                delay = delay + random.uniform(0, delay * 0.1)
                await asyncio.sleep(delay)

            return default_return

        return wrapper
    return decorator
