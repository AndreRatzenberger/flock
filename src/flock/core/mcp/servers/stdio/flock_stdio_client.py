from mcp import ClientSession, InitializeResult, StdioServerParameters, stdio_client
from flock.core.mcp.flock_mcp_client_base import FlockMCPClientBase


from flock.core.logging.logging import get_logger
from opentelemetry import trace

logger = get_logger("mcp_stdio_client")
tracer = trace.get_tracer(__name__)


class FlockStdioClient(FlockMCPClientBase):
    """
    Client for connecting to Stdio Servers.
    """

    async def _init_connection_stdio(self) -> InitializeResult:
        server_params = self.connection_parameters

        if isinstance(server_params, StdioServerParameters):
            async with self.lock:
                try:
                    stdio_transport = await self.master_stack.enter_async_context(stdio_client(server_params))
                    read, write = stdio_transport
                    self.client_session = await self.master_stack.enter_async_context(ClientSession(
                        read_stream=read,
                        write_stream=write,
                        read_timeout_seconds=self.read_timeout_seconds,
                        sampling_callback=self.sampling_callback,
                        list_roots_callback=self.list_roots_callback,
                        logging_callback=self.logging_callback,
                        message_handler=self.message_handler,
                    ))

                    initialize_result: InitializeResult = await self.client_session.initialize()
                    self.is_alive = True
                    self.has_error = False
                    self.error_message = None
                    return initialize_result

                finally:
                    # Release the lock no matter what.
                    self.lock.release()
        else:
            raise TypeError(
                f"Connection Parameters for Stdio Transport type must be of type {type(StdioServerParameters)}")

    async def connect(self, retries: int | None) -> InitializeResult:
        """
        Connects to a local MCP Server.

        Will attempt to connect with a max number of retries.
        """
        last_exception = None
        current_attempt = 0
        retry_attempts = 1
        if not retries and self.max_retries:
            retry_attempts = self.max_retries
        elif retries:
            if retries <= 0:
                logger.warning(f"Number of Retries cannot be smaller than 1")
                retry_attempts = 1
            else:
                retry_attempts = retries

        while current_attempt < retry_attempts:
            try:
                logger.debug(
                    f"Attempting to connect to stdio server with params: {self.connection_parameters}")
                result = await self._init_connection_stdio()
                logger.info(
                    f"Connected to stdio server. Server-Info: {result}")
                return result
            except Exception as e:
                logger.error(
                    f"Attempted to connect to stdio server: Attempt: {current_attempt+1}/{retries}. Exception Occurred: {e}. Retrying.")
                last_exception = e
                current_attempt += 1

        if last_exception:
            logger.error(
                f"Exception ocurred when attempting to connect to stdio server after {retries} attempts. Exception: {e}")
            raise last_exception

    async def close(self):
        """
        Closes the connection and cleans up.

        This means stopping the server script.
        """
        async with self.lock:
            if self.client_session:
                try:
                    # Exit the spawned subprocess.
                    await self.master_stack.aclose()
                except Exception as e:
                    # TODO: propper logging
                    logger.warning(
                        f"Error shutting down transports for server '{self.server_name}': {e}")

            # mark the client as dead.
            self.is_alive = False
