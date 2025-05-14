"""FlockMCPServer is the core, declarative base class for all types of MCP-Servers in the Flock framework"""


import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Annotated, Any, Coroutine, Literal, TypeVar

from flock.core.mcp.flock_mcp_client_manager import FlockMCPClientManager

from opentelemetry import trace
from pydantic import AnyUrl, BaseModel, ConfigDict, Field, UrlConstraints, create_model, field_validator, validator

from dspy import Tool as DSPyTool
from mcp.types import LoggingMessageNotificationParams

from flock.core.context.context import FlockContext
from flock.core.flock_module import FlockModule, FlockModuleConfig
from flock.core.logging.logging import get_logger
from flock.core.mcp.types.mcp_callbacks import FlockListRootsMCPCallback, FlockLoggingMCPCallback, FlockMessageHandlerMCPCallback, FlockSamplingMCPCallback
from flock.core.mcp.types.mcp_types import Root
from flock.core.serialization.serializable import Serializable
from flock.core.serialization.serialization_utils import deserialize_component, serialize_item

logger = get_logger("core.mcp.server_base")
tracer = trace.get_tracer(__name__)
T = TypeVar("T", bound="FlockMCPServerBase")
M = TypeVar("M", bound="FlockMCPServerConfig")

LoggingLevel = Literal[
    "debug", "info", "notice", "warning", "error", "critical", "alert", "emergency"
]


class FlockMCPServerConfig(BaseModel):
    """"
    Base configuration class for Flock MCP Servers

    This class serves as the base for all server-specific configurations.
    Each Type of Server (Stdio, Websocket, HTTP, GRPC) should
    define its own config class inheriting from this one.
    """

    server_name: str = Field(
        ...,
        description="Unique server name"
    )

    description: str | Callable[..., str] | None = Field(
        "",
        description="A human-readable description or a callable returning one."
    )

    server_logging_level: LoggingLevel = Field(
        default="error",
        description="The logging level to request from the mcp-server's logger."
    )

    transport_type: Literal["stdio", "sse", "websockets", "custom"] = Field(
        ...,
        description="What kind of transport this server uses."
    )

    read_timeout_seconds: timedelta = Field(
        default_factory=lambda: timedelta(seconds=10),
        description="How many seconds until timeout."
    )

    mount_points: list[Root] | None = Field(
        default=None,
        description="The initial mounting points of the server."
    )

    resources_enabled: bool = Field(
        default=False,
        description="Whether or not this Server should make resources available to the agents"
    )

    tools_enabled: bool = Field(
        default=False,
        description="Whether or not this Server should provide agents with tools."
    )

    prompts_enabled: bool = Field(
        default=False,
        description="Whether or not this Server should provide agents with prompts."
    )

    roots_enabled: bool = Field(
        default=False,
        description="Wheter or not this Server should allow agents to dynamically change their mountpoints."
    )

    sampling_enabled: bool = Field(
        default=False,
        description="Whether or not this Server is capable of using FLock's LLMs to accept Sampling requests from remote servers."
    )

    mount_points: list[Root] = Field(
        default_factory=list,
        description="The original set of mount points",
    )

    sampling_callback: FlockSamplingMCPCallback | None = Field(
        default=None,
        description="Callback for handling sampling requests."
    )

    list_roots_callback: FlockListRootsMCPCallback | None = Field(
        default=None,
        description="Callback for handling list_roots request."
    )

    logging_callback: FlockLoggingMCPCallback | None = Field(
        default=None,
        description="Callback for logging. MCP Servers send the output they are generating directly to the client."
    )

    message_handler: FlockMessageHandlerMCPCallback | None = Field(
        default=None,
        description="Message Handler Callback."
    )

    tool_cache_max_size: float = Field(
        default=100,
        description="How many items to hold in the tool cache."
    )

    tool_cache_max_ttl: float = Field(
        default=60*5,
        description="Max TTL for each item in the tool cache in seconds."
    )

    resource_contents_cache_max_size: float = Field(
        default=100,
        description="How many items to store in the resource contents cache."
    )

    resource_contents_cache_max_ttl: float = Field(
        default=60*5,
        description="Max TTL for each item in the resource contents cache in seconds."
    )

    resource_list_cache_max_size: float = Field(
        default=100,
        description="How many items to to store in the resource list cache."
    )

    resource_list_cache_max_ttl: float = Field(
        default=60*5,
        description="Max TTL for each item in the resource list cache in seconds."
    )

    tool_result_cache_max_size: float = Field(
        default=100,
        description="Max number of items in the tool result cache."
    )

    tool_result_cache_max_ttl: float = Field(
        default=60*5,
        description="Max TTL for items in the tool result cache in seconds."
    )

    max_restart_attempts: int = Field(
        default=3,
        description="How many times to attempt to restart the server before giving up."
    )

    @classmethod
    def with_fields(cls: type[M], **field_definitions) -> type[M]:
        """Create a new config class with additional fields"""
        return create_model(
            f"Dynamic{cls.__name__}", __base__=cls, **field_definitions
        )


class FlockMCPServerBase(BaseModel, Serializable, ABC):
    """
    Base class for all Flock MCP Server Types.

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

    server_config: FlockMCPServerConfig = Field(
        ...,
        description="Config for the server."
    )

    initialized: bool = Field(
        default=False,
        exclude=True,
        description="Whether or not this Server has already initialized."
    )

    modules: dict[str, FlockModule] = Field(
        default={},
        description="Dictionary of FlockModules attached to this Server."
    )

    # --- Underlying ConnectionManager ---
    # (Manages a pool of ClientConnections and does the actual talking to the MCP Server)
    # (Excluded from Serialization)
    client_manager: FlockMCPClientManager | None = Field(
        default=None,
        exclude=True,
        description="Underlying Connection Manager. Handles the actual underlying connections to the server."
    )

    condition: asyncio.Condition = Field(
        default_factory=asyncio.Condition,
        exclude=True,
        description="Condition for asynchronous operations."
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @abstractmethod
    def add_module(self, module: FlockModule) -> None:
        """Add a module to this server."""
        if not module.name:
            logger.error("Module must have a name to be added.")
            return
        if self.modules and module.name in self.modules:
            logger.warning(f"Overwriting existing module: {module.name}")

        self.modules[module.name] = module
        logger.debug(
            f"Added module '{module.name}' to server {self.server_config.server_name}")
        return

    @abstractmethod
    def remove_module(self, module_name: str) -> None:
        """Remove a module from this server."""
        if module_name in self.modules:
            del self.modules[module_name]
            logger.debug(
                f"Removed module '{module_name}' from server '{self.server_config.server_name}'")
        else:
            logger.warning(
                f"Module '{module_name}' not found on server '{self.server_config.server_name}'")
        return

    @abstractmethod
    def get_module(self, module_name: str) -> FlockModule | None:
        """Get a module by name"""
        return self.modules.get(module_name)

    @abstractmethod
    def get_enabled_modules(self) -> list[FlockModule]:
        """Get a list of currently enabled modules attached to this server"""
        return [m for m in self.modules.values() if m.config.enabled]

    # --- Lifecycle Hooks ---
    @abstractmethod
    async def initialize(self) -> FlockMCPClientManager:
        """
        Called when initializing the server.
        """
        pass

    async def get_tools(self, agent_id: str, run_id: str) -> list[DSPyTool]:
        """
        Retrieves a list of available tools from this server.
        """
        if not self.initialized or not self.client_manager:
            async with self.condition:
                self.client_manager = await self.initialize()
                self.initialized = True

        async with self.condition:
            try:
                result: list[DSPyTool] = await self.client_manager.get_tools(agent_id=agent_id, run_id=run_id)
                return result
            except Exception as e:
                logger.error(
                    f"Unexpected Exception ocurred while trying to get tools from server '{self.server_config.server_name}': {e}"
                )
                return []
            finally:
                self.condition.notify()

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        """Deserialize the server from a dictionary."""
        from flock.core.flock_registry import (
            get_registry
        )

        registry = get_registry()
        logger.debug(
            f"Deserializing server from dict. Keys: {list(data.keys())}")

        # --- Separate Data ---
        component_configs = {}
        server_data = {}

        component_keys = [
            "modules",
        ]

        for key, value in data.items():
            if key in component_keys and value is not None:
                component_configs[key] = value
            elif key not in component_keys:
                server_data[key] = value

        # --- Deserialize Base Server ---
        # Ensure required fields like 'name' are present if needed by __init__
        if "name" not in server_data:
            raise ValueError(
                "Server data must include a 'name' field for deserialization.")

        server_name_log = server_data["name"]  # For logging
        logger.info(f"Deserializing base server data for '{server_name_log}'")

        # Pydantic should handle base fields based on type hints in __init__
        server = cls(**server_data)
        logger.debug(
            f"Base server '{server.server_config.server_name}' instantiated")

        # --- Deserialize components ---
        logger.debug(
            f"Deserializing components for '{server.server_config.server_name}'")

        # Modules
        if "modules" in component_configs:
            server.modules = {}  # Intialize modules dict
            for module_name, module_data in component_configs["modules"].items():
                try:
                    module_instance = deserialize_component(
                        module_data, FlockModule
                    )
                    if module_instance:
                        # Use add_module for potential logic within it
                        server.add_module(module_instance)
                        logger.debug(
                            f"Deserialized and added module '{module_name}' for '{server.server_config.server_name}'")
                except Exception as e:
                    logger.error(
                        f"Failed to deserialize module '{module_name}' for '{server.server_config.server_name}'",
                        exc_info=True
                    )
        logger.info(
            f"Successfully deserialized server '{server.server_config.server_name}'.")
        return server

    def to_dict(self) -> dict[str, Any]:
        """Convert instance to dictionary representation suitable for serialization."""
        from flock.core.flock_registry import get_registry

        FlockRegistry = get_registry()

        exclude = ["context", "modules"]

        is_description_callable = False
        is_input_callable = False
        is_output_callable = False

        # if self.config.description is a callable, exclude it
        if callable(self.server_config.description):
            is_description_callable = True
            exclude.append("description")

        # if self.input is a callable, exclude it
        if callable(self.input):
            is_input_callable = True
            exclude.append("input")

        # if self.output is a callable, exclude it
        if callable(self.output):
            is_output_callable = True
            exclude.append("output")

        logger.debug(
            f"Serializing server '{self.server_config.server_name}' to dict.")

        # Use Pydantic's dump, exclude manually handled fields and runtime context.
        data = self.model_dump(
            exclude=exclude,
            # Use json mode for better handling of standard types by Pydantic.
            mode="json",
            # Exclude None values for cleaner output
            exclude_none=True,
        )
        logger.debug(
            f"Base server data for '{self.server_config.server_name}': {list(data.keys())}")
        serialized_modules = {}

        def add_serialized_component(component: Any, field_name: str):
            if component:
                comp_type = type(component)
                type_name = FlockRegistry.get_component_type_name(
                    comp_type
                )
                if type_name:
                    try:
                        serialized_component_data = serialize_item(component)

                        if not isinstance(serialized_component_data, dict):
                            logger.error(
                                f"Serialization of component {type_name} for field '{field_name} did not result in a dictionary. Got: {type(serialized_component_data)}"
                            )
                            serialized_modules[field_name] = {
                                "type": type_name,
                                "name": getattr(component, "name", "unknown"),
                                "error": "serialization_failed_non_dict",
                            }
                        else:
                            serialized_component_data["type"] = type_name
                            serialized_modules[field_name] = (
                                serialized_component_data
                            )
                            logger.debug(
                                f"Successfully serialized component for field '{field_name}' (type: {type_name})"
                            )
                    except Exception as e:
                        logger.error(
                            f"Failed to serialize component {type_name} for field '{field_name}'",
                            exc_info=True
                        )
                        serialized_modules[field_name] = {
                            "type": type_name,
                            "name": getattr(component, "name", "unknown"),
                            "error": "serialization_failed",
                        }
                else:
                    logger.warning(
                        f"Cannot serialize unregistered component {comp_type.__name__} for field '{field_name}'"
                    )
        serialized_modules = {}
        for module in self.modules.values():
            add_serialized_component(module, module.name)

        if serialized_modules:
            data["modules"] = serialized_modules
            logger.debug(
                f"Added {len(serialized_modules)} modules to server '{self.server_config.server_name}'"
            )

        if is_description_callable:
            path_str = FlockRegistry.get_callable_path_string(
                self.server_config.description)
            if path_str:
                func_name = path_str.split(".")[-1]
                data["description_callable"] = func_name
                logger.debug(
                    f"Added description '{func_name}' (from path '{path_str}') to server."
                )
            else:
                logger.warning(
                    f"Could not get path string for description {self.server_config.description} in server '{self.server_config.server_name}'. Skipping..."
                )

        if is_input_callable:
            path_str = FlockRegistry.get_callable_path_string(self.input)
            if path_str:
                func_name = path_str.split(".")[-1]
                data["input_callable"] = func_name
                logger.debug(
                    f"Added input '{func_name}' (from path '{path_str}') to server '{self.server_config.server_name}'"
                )
            else:
                logger.warning(
                    f"Could not get path string for input {self.input} in server '{self.server_config.server_name}'. Skipping..."
                )

        if is_output_callable:
            path_str = FlockRegistry.get_callable_path_string(self.output)
            if path_str:
                func_name = path_str.split(".")[-1]
                data["output_callable"] = func_name
                logger.debug(
                    f"Added output '{func_name}' (from path '{path_str}') to server '{self.server_config.server_name}'"
                )
            else:
                logger.warning(
                    f"Could not get path string for output {self.output} in server '{self.server_config.server_name}'. Skipping...")

        # No need to call _filter_none_values here as model_dump(exclude_none=True) handles it
        logger.info(
            f"Serialization of server '{self.server_config.server_name}' complete with {len(data)} fields"
        )
        return data

    async def __aenter__(self):
        if not self.client_manager or not self.initialized:
            # Connection has not yet been established.
            await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        # Check if the connection_manager is there:
        if self.client_manager:
            await self.client_manager.close_all()
