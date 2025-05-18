"""FlockMCPServer is the core, declarative base class for all types of MCP-Servers in the Flock framework."""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Literal, TypeVar

from dspy import Tool as DSPyTool
from opentelemetry import trace
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from flock.core.flock_module import FlockModule
from flock.core.logging.logging import get_logger
from flock.core.mcp.flock_mcp_tool_base import FlockMCPToolBase
from flock.core.mcp.mcp_client_manager import FlockMCPClientManager
from flock.core.mcp.mcp_config import FlockMCPConfigurationBase

logger = get_logger("core.mcp.server_base")
tracer = trace.get_tracer(__name__)
T = TypeVar("T", bound="FlockMCPServerBase")

LoggingLevel = Literal[
    "debug",
    "info",
    "notice",
    "warning",
    "error",
    "critical",
    "alert",
    "emergency",
]


class FlockMCPServerBase(BaseModel, ABC):
    """Base class for all Flock MCP Server Types.

    Servers serve as an abstraction-layer between the underlying MCPClientSession
    which is the actual connection between Flock and a (remote) MCP-Server.

    Servers hook into the lifecycle of their assigned agents and take care
    of establishing sessions, getting and converting tools and other functions
    without agents having to worry about the details.

    Tools (if provided) will be injected into the list of tools of any attached
    agent automatically.

    Servers provide lifecycle-hooks (`initialize`, `get_tools`, `get_prompts`, `list_resources`, `get_resource_contents`, `set_roots`, etc)
    which allow modules to hook into them. This can be used to modify data or
    pass headers from authentication-flows to a server.

    Each Server should define its configuration requirements either by:
    1. Creating a subclass of FlockMCPServerConfig
    2. Using FlockMCPServerConfig.with_fields() to create a config class.
    """

    config: FlockMCPConfigurationBase = Field(
        ..., description="Config for clients connecting to the server."
    )

    initialized: bool = Field(
        default=False,
        exclude=True,
        description="Whether or not this Server has already initialized.",
    )

    modules: dict[str, FlockModule] = Field(
        default={},
        description="Dictionary of FlockModules attached to this Server.",
    )

    # --- Underlying ConnectionManager ---
    # (Manages a pool of ClientConnections and does the actual talking to the MCP Server)
    # (Excluded from Serialization)
    client_manager: FlockMCPClientManager | None = Field(
        default=None,
        exclude=True,
        description="Underlying Connection Manager. Handles the actual underlying connections to the server.",
    )

    condition: asyncio.Condition = Field(
        default_factory=asyncio.Condition,
        exclude=True,
        description="Condition for asynchronous operations.",
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def add_module(self, module: FlockModule) -> None:
        """Add a module to this server."""
        if not module.name:
            logger.error("Module must have a name to be added.")
            return
        if self.modules and module.name in self.modules:
            logger.warning(f"Overwriting existing module: {module.name}")

        self.modules[module.name] = module
        logger.debug(
            f"Added module '{module.name}' to server {self.config.server_name}"
        )
        return

    def remove_module(self, module_name: str) -> None:
        """Remove a module from this server."""
        if module_name in self.modules:
            del self.modules[module_name]
            logger.debug(
                f"Removed module '{module_name}' from server '{self.config.server_name}'"
            )
        else:
            logger.warning(
                f"Module '{module_name}' not found on server '{self.config.server_name}'"
            )
        return

    def get_module(self, module_name: str) -> FlockModule | None:
        """Get a module by name."""
        return self.modules.get(module_name)

    def get_enabled_modules(self) -> list[FlockModule]:
        """Get a list of currently enabled modules attached to this server."""
        return [m for m in self.modules.values() if m.config.enabled]

    # --- Lifecycle Hooks ---
    @abstractmethod
    async def initialize(self) -> FlockMCPClientManager:
        """Called when initializing the server."""
        pass

    async def call_tool(
        self, agent_id: str, run_id: str, name: str, arguments: dict[str, Any]
    ) -> Any:
        """Call a tool via the MCP Protocol on the client's server."""
        with tracer.start_as_current_span("server.call_tool") as span:
            span.set_attribute("agent_id", agent_id)
            span.set_attribute("run_id", run_id)
            span.set_attribute("tool.name", name)
            span.set_attribute("arguments", str(arguments))
            if not self.initialized or not self.client_manager:
                async with self.condition:
                    await self.pre_init()
                    self.client_manager = await self.initialize()
                    self.initialized = True
                    await self.post_init()
            async with self.condition:
                try:
                    await self.pre_mcp_call()
                    additional_params: dict[str, Any] = {
                        "refresh_client": False,
                        "override_headers": False,
                    }  # initialize the additional params as an empty dict.

                    await self.before_connect(
                        additional_params=additional_params
                    )
                    result = await self.client_manager.call_tool(
                        agent_id=agent_id,
                        run_id=run_id,
                        name=name,
                        arguments=arguments,
                        additional_params=additional_params,
                    )
                    # re-set addtional-params, just to be sure.
                    await self.post_mcp_call(result=result)
                    return result
                except Exception as mcp_error:
                    logger.error(
                        "Error during server.call_tool",
                        server=self.config.server_name,
                        error=str(mcp_error),
                    )
                    span.record_exception(mcp_error)
                    return None

    async def get_tools(self, agent_id: str, run_id: str) -> list[DSPyTool]:
        """Retrieves a list of available tools from this server."""
        with tracer.start_as_current_span("server.get_tools") as span:
            span.set_attribute("server.name", self.config.server_name)
            span.set_attribute("agent_id", agent_id)
            span.set_attribute("run_id", run_id)
            if not self.initialized or not self.client_manager:
                async with self.condition:
                    await self.pre_init()
                    self.client_manager = await self.initialize()
                    self.initialized = True
                    await self.post_init()

            async with self.condition:
                try:
                    await self.pre_mcp_call()
                    # TODO: inject additional params here.
                    result: list[
                        FlockMCPToolBase
                    ] = await self.client_manager.get_tools(
                        agent_id=agent_id, run_id=run_id
                    )
                    converted_tools = [
                        t.as_dspy_tool(server=self) for t in result
                    ]
                    await self.post_mcp_call(result=converted_tools)
                    return converted_tools
                except Exception as e:
                    logger.error(
                        f"Unexpected Exception ocurred while trying to get tools from server '{self.config.server_name}': {e}"
                    )
                    await self.on_error(error=e)
                    span.record_exception(e)
                    return []
                finally:
                    self.condition.notify()

    async def before_connect(self, additional_params: dict[str, Any]) -> None:
        """Run before_connect hooks on modules."""
        logger.debug(
            f"Running before_connect hooks for modules in server '{self.config.server_name}'."
        )
        with tracer.start_as_current_span("server.before_connect") as span:
            span.set_attribute("server.name", self.config.server_name)
            try:
                for module in self.get_enabled_modules():
                    await module.before_connect(
                        server=self, additional_params=additional_params
                    )
            except Exception as module_error:
                logger.error(
                    "Error during before_connect",
                    server=self.config.server_name,
                    error=str(module_error),
                )
                span.record_exception(module_error)

    async def pre_init(self) -> None:
        """Run pre-init hooks on modules."""
        logger.debug(
            f"Running pre-init hooks for modules in server '{self.config.server_name}'"
        )
        with tracer.start_as_current_span("server.pre_init") as span:
            span.set_attribute("server.name", self.config.server_name)
            try:
                for module in self.get_enabled_modules():
                    await module.pre_server_init(self)
            except Exception as module_error:
                logger.error(
                    "Error during pre_init",
                    server=self.config.server_name,
                    error=str(module_error),
                )
                span.record_exception(module_error)

    async def post_init(self) -> None:
        """Run post-init hooks on modules."""
        logger.debug(
            f"Running post_init hooks for modules in server '{self.config.server_name}'"
        )
        with tracer.start_as_current_span("server.post_init") as span:
            span.set_attribute("server.name", self.config.server_name)
            try:
                for module in self.get_enabled_modules():
                    await module.post_server_init(self)
            except Exception as module_error:
                logger.error(
                    "Error during post_init",
                    server=self.config.server_name,
                    error=str(module_error),
                )
                span.record_exception(module_error)

    async def pre_terminate(self) -> None:
        """Run pre-terminate hooks on modules."""
        logger.debug(
            f"Running post_init hooks for modules in server: '{self.config.server_name}'"
        )
        with tracer.start_as_current_span("server.pre_terminate") as span:
            span.set_attribute("server.name", self.config.server_name)
            try:
                for module in self.get_enabled_modules():
                    await module.pre_server_terminate(self)
            except Exception as module_error:
                logger.error(
                    "Error during pre_terminate",
                    server=self.config.server_name,
                    error=str(module_error),
                )
                span.record_exception(module_error)

    async def post_terminate(self) -> None:
        """Run post-terminate hooks on modules."""
        logger.debug(
            f"Running post_terminat hooks for modules in server: '{self.config.server_name}'"
        )
        with tracer.start_as_current_span("server.post_terminate") as span:
            span.set_attribute("server.name", self.config.server_name)
            try:
                for module in self.get_enabled_modules():
                    await module.post_server_teminate(server=self)
            except Exception as module_error:
                logger.error(
                    "Error during post_terminate",
                    server=self.config.server_name,
                    error=str(module_error),
                )
                span.record_exception(module_error)

    async def on_error(self, error: Exception) -> None:
        """Run on_error hooks on modules."""
        logger.debug(
            f"Running on_error hooks for modules in server '{self.config.server_name}'"
        )
        with tracer.start_as_current_span("server.on_error") as span:
            span.set_attribute("server.name", self.config.server_name)
            try:
                for module in self.get_enabled_modules():
                    await module.on_server_error(server=self, error=error)
            except Exception as module_error:
                logger.error(
                    "Error during on_error",
                    server=self.config.server_name,
                    error=str(module_error),
                )
                span.record_exception(module_error)

    async def pre_mcp_call(self) -> None:
        """Run pre_mcp_call-hooks on modules."""
        logger.debug(
            f"Running pre_mcp_call hooks for modules in server '{self.config.server_name}'"
        )
        with tracer.start_as_current_span("server.pre_mcp_call") as span:
            span.set_attribute("server.name", self.config.server_name)
            try:
                for module in self.get_enabled_modules():
                    await module.pre_mcp_call(server=self)
            except Exception as module_error:
                logger.error(
                    "Error during pre_mcp_call",
                    server=self.config.server_name,
                    error=str(module_error),
                )
                span.record_exception(module_error)

    async def post_mcp_call(self, result: Any) -> None:
        """Run Post MCP_call hooks on modules."""
        logger.debug(
            f"Running post_mcp_call hooks for modules in server '{self.config.server_name}'"
        )
        with tracer.start_as_current_span("server.post_mcp_call") as span:
            span.set_attribute("server.name", self.config.server_name)
            try:
                for module in self.get_enabled_modules():
                    await module.post_mcp_call(server=self, result=result)
            except Exception as module_error:
                logger.error(
                    "Error during post_mcp_call",
                    server=self.config.server_name,
                    error=str(module_error),
                )
                span.record_exception(module_error)

    # --- Async Methods ---
    async def __aenter__(self) -> "FlockMCPServerBase":
        """Enter the asynchronous context for the server."""
        # Spin up the client-manager
        with tracer.start_as_current_span("server.__aenter__") as span:
            span.set_attribute("server.name", self.config.server_name)
            logger.info(f"server.__aenter__", server=self.config.server_name)
            try:
                await self.pre_init()
                self.client_manager = await self.initialize()
                await self.post_init()
                self.initialized = True
            except Exception as server_error:
                logger.error(
                    f"Error during __aenter__ for server '{self.config.server_name}'",
                    server=self.config.server_name,
                    error=server_error,
                )
                span.record_exception(server_error)

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Exit the asynchronous context for the server."""
        # tell the underlying client-manager to terminate connections
        # and unwind the clients.
        with tracer.start_as_current_span("server.__aexit__") as span:
            span.set_attribute("server.name", self.config.server_name)
            try:
                await self.pre_terminate()
                if self.initialized and self.client_manager:
                    # means we ran through the initialize()-method
                    # and the client manager is present
                    await self.client_manager.close_all()
                    self.client_manager = None
                    self.initialized = False
                await self.post_terminate()
                return
            except Exception as server_error:
                logger.error(
                    f"Error during __aexit__ for server '{self.config.server_name}'",
                    server=self.config.server_name,
                    error=server_error,
                )
                await self.on_error(error=server_error)
                span.record_exception(server_error)
