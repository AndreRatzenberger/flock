"""Manages a pool of connections for a particular server."""


from asyncio import Condition, Lock
import asyncio
from contextlib import asynccontextmanager
import random
from typing import Annotated, AsyncIterator, Literal
from pydantic import AnyUrl, BaseModel, Field, UrlConstraints
from opentelemetry import trace

from mcp import ClientSession

from flock.core.mcp.flock_mcp_client_base import FlockMCPClientBase
from flock.core.logging.logging import get_logger

logger = get_logger("mcp_server")
tracer = trace.get_tracer(__name__)


class FlockMCPConnectionManagerBase(BaseModel):
    """Handles a Pool of MCPClients."""

    transport_type: Literal["stdio", "websockets", "http"] = Field(
        ..., description="Transport-Type to use for Connections")

    server_name: str = Field(...,
                             description="Name of the server to connect to.")

    min_connections: int = Field(
        default=1,
        description="Minimum amount of connections to keep alive.",
    )

    max_connections: int = Field(
        default=24,
        description="Upper bound for the maximum amount of connections to open."
    )

    max_reconnect_attemtps: int = Field(
        default=10,
        description="Maximum amounts to retry to create a client. After that the ConnectionManager throws an Exception.",
    )

    # --- Internal State ---
    available_connections: list[FlockMCPClientBase] = Field(
        default=[],
        description="Connections which are capable of handling a request at the moment.",
        exclude=True,
    )

    busy_connections: list[FlockMCPClientBase] = Field(
        default=[],
        description="Connections which are currently handling a request.",
        exclude=True,
    )

    lock: Lock = Field(
        default_factory=Lock,
        description="Lock for mutex access.",
        exclude=True,
    )

    condition: Condition = Field(
        default_factory=Condition,
        description="Condition variable to wait for available connections.",
        exclude=True,
    )

    replenish_task: asyncio.Task | None = Field(
        default=None,
        description="Task for background replenishment.",
        exclude=True,
        init_var=False,
    )

    original_roots: list[Annotated[AnyUrl, UrlConstraints(host_required=False)]] | list[str] | None = Field(
        default=None,
        description="The original roots of the managed clients",
    )

    # --- Pydantic v2 Configuratioin ---
    model_config = {
        "arbitrary_types_allowed": True,
    }

    async def initialize(self) -> None:
        """
        Initializes the connection Manager and populates the pool.
        """
        logger.debug(f"ConnectionManager for {self.server_name} initializing.")
        try:
            await self._initialize_pool(self)
            logger.debug(
                f"ConnectionManager for {self.server_name} initialized.")
        except Exception as e:
            logger.error(f"Exception occurred during pool initialization: {e}")
            # Re-Throw any exceptions, so the caller knows that something is going on.
            raise

    async def _initialize_pool(self) -> None:
        """
        Populates the pool up to min_connections.
        Should be called after manager creation.
        """
        logger.info(
            f"Initializing connection pool for {self.server_name} (min: {self.min_connections}, max: {self.max_connections})")

        # Trigger an initial replenishment_check and wait for population of connection pool
        self._trigger_replenishment_check()
        if self.replenish_task:
            await self.replenish_task

    async def _create_new_connection_with_retry(self) -> FlockMCPClientBase | None:
        """
        Attempts to create and connect a single new client with retries.
        Handles retries based on max_reconnect attempts.
        """
        last_exception = None
        for attempt in range(self.max_reconnect_attemtps + 1):
            logger.info(
                f"Attempt {attempt + 1}/{self.max_reconnect_attemtps + 1} to create connection to {self.server_name}")
            try:
                client = FlockMCPClientBase(max_retries=3)

                # Let manager handle retries initially
                await client.connect(retries=0)

                if await client.get_is_alive() and not await client.get_has_error():
                    logger.info(
                        f"Successfully created and connected: {client}")
                    return client
                else:
                    message = await client.get_error_message()
                    logger.warning(
                        f"Connection attempt {attempt + 1} failed for client {client}. Error: {message}")

                    if message := await client.get_error_message():
                        last_exception = Exception(message)

                    await client.set_is_alive(False)
                    await client.set_has_error(False)

                    # Close potentially partially opened resources if client has a close method
                    if hasattr(client, "close") and asyncio.iscoroutinefunction(client.close):
                        try:
                            await client.close()
                        except Exception as close_err:
                            logger.error(
                                f"Error closing failed client {client} during retry: {close_err}")
            except Exception as e:
                logger.error(
                    f"Exception during connection attempt {attempt + 1} to {self.server_name}: {e}", exc_info=True)
                last_exception = e
        # If not the last attempt, wait before retrying
        if attempt < self.max_reconnect_attemtps:
            # Exponential backoff with jitter
            delay = (2 ** attempt) * 0.5 + random.uniform(0, 0.5)
            logger.info(f"Waiting {delay:.2f}s before next connection attempt")
            await asyncio.sleep(delay)

        logger.error(
            f"Failed to create connection to {self.server_name} after {self.max_reconnect_attemtps + 1}")

        raise ConnectionError(
            f"Failed to create connection after retries: {last_exception}") from last_exception

    # --- Core Pool Logic ---
    @asynccontextmanager
    async def get_client(self) -> AsyncIterator[FlockMCPClientBase]:
        """
        Provides a client
        from the pool via an async context manager, ensuring
        release.

        Usage:
            async with manager.get_client() as client:
                # Use the client here
                tools = await client.get_tools()
            # Client is automatically released back into the pool here.
        """
        client: FlockMCPClientBase | None = None
        try:
            # Acquire the client using the internal logic
            client = await self._get_available_client()
            # Yield the client to the `async with block`
            yield client
        except Exception as e:
            logger.error(
                f"Exception within get_client context for {client}: {e}", exc_info=True)
            # Re-raise the exception so the caller knwos something went wrong
            raise
        finally:
            # No matter what happens, clients need to be returned
            # to the pool
            if client:
                logger.debug(
                    f"Releaseing client via context manager: {client}")
                await self._release_client(client)

    async def _get_available_client(self) -> FlockMCPClientBase:
        """
        Retrieves a non-busy, live client from the connection-pool.
        If no client is available, it waits until one is returned or created.
        Moves the client from the available to the busy list.
        """

        async with self.condition:
            while True:
                while not self.available_connections:
                    # Check if we can create more connections **before** waiting
                    total_connections = len(
                        self.available_connections) + len(self.busy_connections)
                    if total_connections < self.max_connections:
                        logger.debug(
                            f"Pool below capacity, attempting to create new connection (current: {total_connections})")
                        # Release the condition briefly to allow creation.
                        self.condition.release()
                        new_client = None
                        try:
                            # Attempt to create one connection immediately if below max
                            new_client = await self._create_new_connection_with_retry()
                        except Exception as e:
                            logger.error(
                                f"Unexpected error during immediate connection creation: {e}", exc_info=True)
                        finally:
                            # We MUST re-acquire the lock, regardless of outcome
                            await self.condition.acquire()

                        if new_client:
                            # Successfully created one
                            await client.set_is_busy(False)
                            self.available_connections.append(new_client)
                            logger.info(
                                f"Immediately added new client {new_client} to pool.")
                            self.condition.notify()  # Notify self/others
                        else:
                            # Creation failed or returned None, wait normally.
                            logger.warning(
                                "Immediate connection creation failed or returned None. Waiting.")
                            # Trigger check just in case, but still wait.
                            self._trigger_replenishment_check()
                            await self.condition.wait()
                    else:
                        # Max connections reached, must wait for one to be returned
                        logger.debug(
                            f"Connection pool for server '{self.server_name}' is full ({total_connections}/{self.max_connections})"
                        )
                        await self.condition.wait()
                # We have availabel connections now (or woke up and need to re-check)
                if not self.available_connections:
                    continue  # Woke up, but someone else grabbed the connection, re-wait

                client = self.available_connections.pop(0)

                # Check if the client we just retrieved is alive.
                if await client.get_is_alive() and not await client.get_has_error():
                    # It's alive. Mark it as busy and return it.
                    await client.set_is_busy(True)
                    self.busy_connections.append(client)
                    logger.debug(f"Handing out client: {client}")

                    return client
                else:
                    message = await client.get_error_message()
                    # Client is dead/has errors, discard it and trigger replenishment check
                    logger.warning(
                        f"Found dead/error client {client} in available pool. Discarding: Error: {message}"
                    )
                    # Don't put it back, just let it be garbage collected
                    # Ensure replenishment is triggered
                    self._trigger_replenishment_check()
                    # Continue the outer loop to find another client.

    async def _release_client(self, client: FlockMCPClientBase) -> None:
        """
        Releases a client back into the available pool and notifies waiting tasks.
        Moves the client from busy to available list. Discards unhealthy clients.
        """
        async with self.condition:
            try:
                if client in self.busy_connections:
                    self.busy_connections.remove(client)
                else:
                    # If not in busy, maybe it was already released or never assigned?
                    logger.warning(
                        f"Attempted to release client not in busy list: {client}")
                    # If it's healthy, and somehow not available add it back
                    if await client.get_is_alive() and not await client.get_has_error() and client not in self.available_connections:
                        logger.warning(
                            f"Adding released client {client} back to available pool as it was healthy and not busy/available")
                        await client.set_is_busy(False)
                        # IMPORTANT: Reset roots
                        await client.set_roots(self.original_roots)
                        self.available_connections.append(client)
                        self.condition.notify(1)
                    elif not await client.get_is_alive() or await client.get_has_error():
                        logger.warning(
                            f"Discarding unhealthy client {client} released but not found in busy list.")
                        self._trigger_replenishment_check()
                    return  # Exit early if not found in busy.

                # If we removed it from busy, proceed to check health
                await client.set_is_busy(False)

                if await client.get_is_alive() and not await client.get_has_error():
                    # Return healthy clients to the pool
                    self.available_connections.append(client)
                    logger.debug(f"Client returned to pool: {client}")
                    self.condition.notify(1)

                else:
                    message = await client.get_error_message()
                    logger.warning(
                        f"Released client {client} is dead/has error. Discarding. Error: {message}")
                    self._trigger_replenishment_check()
            except Exception as e:
                logger.error(
                    f"Error during _release_client for {client}: {e}", exc_info=True)
                self._trigger_replenishment_check()

    async def close_all(self) -> None:
        """
        Closes all connections in the pool and cancels background tasks.
        """
        logger.info(
            f"Closing all connections in the pool for {self.server_name}")
        async with self.lock:
            # Cancel replenishment task first, if running
            if self.replenish_task and not self.replenish_task.done():
                logger.info("Cancelling background replenishment task.")
                self.replenish_task.cancel()
                try:
                    await self.replenish_task
                except asyncio.CancelledError:
                    logger.info("Replenishment task successfully cancelled.")
                except Exception as e:
                    logger.error(
                        f"Error encountered while awaiting cancelled replenishment task: {e}")
                self.replenish_task = None

            all_connections = self.available_connections + self.busy_connections
            self.available_connections.clear()
            self.busy_connections.clear()
            logger.debug(
                f"Cleared internal connection lists. Found {len(all_connections)} total connections to close.")

            close_tasks = []
            for client in all_connections:
                if hasattr(client, "close") and asyncio.iscoroutinefunction(client.close):
                    logger.debug(f"Scheduling close for client: {client}")
                    close_tasks.append(asyncio.create_task(
                        client.close(), name=f"close-{client}"))
                else:
                    logger.warning(
                        f"Client {client} does not have an async close method.")
            if close_tasks:
                logger.info(
                    f"Waiting for {len(close_tasks)} client close tasks to complete.")
                results = await asyncio.gather(*close_tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"Error closing client: {result}")
            logger.info(f"All connections closed for {self.server_name}")

    # --- Replenishment Logic for the Pool ---
    def _trigger_replenishment_check(self) -> None:
        """Creates a task to check and replenish connections if not already running."""
        # Check if a replenishment task is already scheduled or running
        if self.replenish_task is None or self.replenish_task.done():
            logger.info(
                "Triggering background connection replenishment check.")
            self.replenish_task = asyncio.create_task(
                self._check_and_replenish(), name=f"replenish-{self.server_name}")
            self.replenish_task.add_done_callback(self._clear_replenish_task)

    def _clear_replenish_task(self, task: asyncio.Task) -> None:
        """Callback to clear the _replenish_task reference and log errors."""
        # Check fi the task that finished is indeed the one we stored
        # (Handles potential race conditions if trigger is called rapidly)
        if self.replenish_task is task:
            self.replenish_task = None
            try:
                # Check if the task failed
                exception = task.exception()
                if exception:
                    logger.error(
                        f"Replenishment task for {self.server_name} failed: {exception}", exc_info=True)
                else:
                    logger.debug(
                        f"Replenishment task for {self.server_name} completed.")
            except asyncio.CancelledError:
                logger.info(
                    f"Replenishment taks for {self.server_name} was cancelled.")
            except Exception as e:
                logger.error(
                    f"Unexpected error in _clear_replenish_task for {self.server_name}: {e}", exc_info=True)

    async def _check_and_replenish(self):
        """Checks connection count and creates new ones if below minimum."""
        # Short delay to prevend rapid-fire checks if multiple clients die at once
        await asyncio.sleep(0.1)

        async with self.lock:  # Use the primary lock, not condition
            await self._ensure_minimum_connections_locked()

    async def _ensure_minimum_connections_locked(self):
        """
        Internal method to ensure minimum amount of connections.
        MUST be called with self._lock already held.
        """
        try:
            # Prune dead connections from available list only
            live_available = [
                c for c in self.available_connections if c.is_alive and not c.has_error]
            dead_available_count = len(
                self.available_connections) - len(live_available)
            if dead_available_count > 0:
                logger.warning(
                    f"Pruning {dead_available_count} dead/error connections from available pool")
                self.available_connections = live_available

            # Count current live connections (available + busy)
            # Assume busy connections are live until proven otherwise by get/release
            current_total_count = len(
                self.available_connections) + len(self.busy_connections)

            # Approximation
            current_live_count = len(live_available) + \
                len(self.busy_connections)

            logger.debug(
                f"Checking connections for {self.server_name}"
                f"{len(self.available_connections)} avail (all live), {len(self.busy_connections)} busy."
                f"Total: {current_total_count}. Approx Live: {current_live_count}. Min required: {self.min_connections}"
            )

            needed = self.min_connections - current_live_count

            if needed <= 0:
                logger.debug(
                    f"Connection count ({current_live_count}) meets minimum ({self.min_connections})")
                return  # We alerady have enough or more

            # Respect max connections limit
            can_create = self.max_connections - current_total_count
            create_count = min(needed, can_create)

            if create_count <= 0:
                if needed > 0:
                    logger.warning(
                        f"Need {needed} connections for {self.server_name}, but max connections ({self.max_connections}) has already been reached"
                        f"Current total: {current_total_count}"
                    )
                return

            logger.info(
                f"Need {needed} connections for {self.server_name}, attempting to create {create_count}"
                f"(current total: {current_total_count}, max: {self.max_connections})"
            )

            # --- Parallel Creation ---
            tasks = [
                asyncio.create_task(self._create_new_connection_with_retry(
                ), name=f"create-{self.server_name}-{i}")
                for i in range(create_count)
            ]

            # Release the main lock while waiting for connections to be created
            self.lock.release()
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                # We MUST re-acquire the main lock
                await self.lock.acquire()

            new_clients = []
            failed_count = 0
            for i, result in enumerate(results):
                if isinstance(result, FlockMCPClientBase) and result.is_alive and not result.has_error:
                    new_clients.append(result)
                elif isinstance(result, Exception):
                    logger.error(
                        f"Failed to create connection for {self.server_name} during replenishment. (Task {i}): {result}", exc_info=True)
                else:
                    logger.warning(
                        f"Connection creation task {i} for {self.server_name} returned None or an unhealthy client.")
                    failed_count += 1

            if new_clients:
                logger.info(
                    f"Successfully created {len(new_clients)} new connections for {self.server_name}")

                # Need to acquire condition lock briefly to modify list and notify
                async with self.condition:
                    self.available_connections.extend(new_clients)
                    # Notify potentiall waiting getters for each new connection added
                    self.condition.notify(len(new_clients))
            else:
                logger.warning(
                    f"Failed to create any new connections for {self.server_name} during replenishment")

            # If some failed, trigger another check later in case the issue was temporary
            if failed_count > 0:
                # Schedule another check slightly later, don't trigger immediately
                # to avoid tight loops if the server is down.
                await asyncio.sleep(0.01)
                pass  # For now, rely on subsequent calls to get/release to trigger checks
        except Exception as e:
            logger.error(
                f"Error during _ensure_minimum_connections_locked for {self.server_name}: {e}", exc_info=True)
            # Release lock if held due to excpetion before acquire/release pair
            if self.lock.locked():
                self.lock.release()
