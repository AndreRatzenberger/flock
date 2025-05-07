from mcp import ClientSession, StdioServerParameters, stdio_client
from flock.core.mcp.flock_mcp_client_base import FlockMCPClientBase


from flock.core.logging.logging import get_logger
from opentelemetry import trace

logger = get_logger("mcp_stdio_client")
tracer = trace.get_tracer(__name__)


class FlockStdioClient(FlockMCPClientBase):
    """
    Client for connecting to Stdio Servers.
    """

    async def _init_connection_stdio(self) -> None:
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

                    await self.client_session.initialize()

                finally:
                    # Release the lock no matter what.
                    self.lock.release()
        else:
            raise TypeError(
                f"Connection Parameters for Stdio Transport type must be of type {type(StdioServerParameters)}")

    async def connect(self, retries: int | None) -> None | Exception:
        # TODO: continue tomorrow
