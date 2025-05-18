"""Base Config for MCP Clients."""

from datetime import timedelta
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, create_model

from flock.core.mcp.types.types import (
    FlockListRootsMCPCallback,
    FlockLoggingMCPCallback,
    FlockMessageHandlerMCPCallback,
    FlockSamplingMCPCallback,
    Root,
    ServerParameters,
)

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


A = TypeVar("A", bound="FlockMCPCallbackConfiguration")
B = TypeVar("B", bound="FlockMCPConnectionConfiguration")
C = TypeVar("C", bound="FlockMCPConfigurationBase")
D = TypeVar("D", bound="FlockMCPCachingConfiguration")
E = TypeVar("E", bound="FlockMCPFeatureConfiguration")


class FlockMCPCachingConfiguration(BaseModel):
    """Configuration for Caching in Clients."""

    tool_cache_max_size: float = Field(
        default=100, description="Maximum number of items in the Tool Cache."
    )

    tool_cache_max_ttl: float = Field(
        default=60,
        description="Max TTL for items in the tool cache in seconds.",
    )

    resource_contents_cache_max_size: float = Field(
        default=10,
        description="Maximum number of entries in the Resource Contents cache.",
    )

    resource_contents_cache_max_ttl: float = Field(
        default=60 * 5,
        description="Maximum number of items in the Resource Contents cache.",
    )

    resource_list_cache_max_size: float = Field(
        default=10,
        description="Maximum number of entries in the Resource List Cache.",
    )

    resource_list_cache_max_ttl: float = Field(
        default=100,
        description="Maximum TTL for entries in the Resource List Cache.",
    )

    tool_result_cache_max_size: float = Field(
        default=1000,
        description="Maximum number of entries in the Tool Result Cache.",
    )

    tool_result_cache_max_ttl: float = Field(
        default=20,
        description="Maximum TTL in seconds for entries in the Tool Result Cache.",
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
    )

    @classmethod
    def with_fields(cls: type[D], **field_definitions) -> type[D]:
        """Create a new config class with additional fields."""
        return create_model(
            f"Dynamic{cls.__name__}", __base__=cls, **field_definitions
        )


class FlockMCPCallbackConfiguration(BaseModel):
    """Base Configuration Class for Callbacks for Clients."""

    sampling_callback: FlockSamplingMCPCallback | None = Field(
        default=None,
        description="Callback for handling sampling requests from an external server.",
    )

    list_roots_callback: FlockListRootsMCPCallback | None = Field(
        default=None, description="Callback for handling list roots requests."
    )

    logging_callback: FlockLoggingMCPCallback | None = Field(
        default=None,
        description="Callback for handling logging messages from an external server.",
    )

    message_handler: FlockMessageHandlerMCPCallback | None = Field(
        default=None,
        description="Callback for handling messages not covered by other callbacks.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    @classmethod
    def with_fields(cls: type[A], **field_definitions) -> type[A]:
        """Create a new config class with additional fields."""
        return create_model(
            f"Dynamic{cls.__name__}", __base__=cls, **field_definitions
        )


class FlockMCPConnectionConfiguration(BaseModel):
    """Base Configuration Class for Connection Parameters for a client."""

    max_retries: int = Field(
        default=3,
        description="How many times to attempt to establish the connection before giving up.",
    )

    connection_parameters: ServerParameters = Field(
        ..., description="Connection parameters for the server."
    )

    transport_type: Literal["stdio", "websockets", "sse", "custom"] = Field(
        ..., description="Type of transport to use."
    )

    mount_points: list[Root] | None = Field(
        default=None, description="Initial Mountpoints to operate under."
    )

    read_timeout_seconds: float | timedelta = Field(
        default=60 * 5, description="Read Timeout."
    )

    server_logging_level: LoggingLevel = Field(
        default="error",
        description="The logging level for logging events from the remote server.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    @classmethod
    def with_fields(cls: type[B], **field_definitions) -> type[B]:
        """Create a new config class with additional fields."""
        return create_model(
            f"Dynamic{cls.__name__}", __base__=cls, **field_definitions
        )


class FlockMCPFeatureConfiguration(BaseModel):
    """Base Configuration Class for switching MCP Features on and off."""

    roots_enabled: bool = Field(
        default=False,
        description="Whether or not the Roots feature is enabled for this client.",
    )

    sampling_enabled: bool = Field(
        default=False,
        description="Whether or not the Sampling feature is enabled for this client.",
    )

    tools_enabled: bool = Field(
        default=False,
        description="Whether or not the Tools feature is enabled for this client.",
    )

    prompts_enabled: bool = Field(
        default=False,
        description="Whether or not the Prompts feature is enabled for this client.",
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
    )

    @classmethod
    def with_fields(cls: type[E], **field_definitions) -> type[E]:
        """Create a new config class with additional fields."""
        return create_model(
            f"Dynamic{cls.__name__}", __base__=cls, **field_definitions
        )


class FlockMCPConfigurationBase(BaseModel):
    """Base Configuration Class for MCP Clients.

    Each Client should implement their own config
    model by inheriting from this class.
    """

    server_name: str = Field(
        ..., description="Name of the server the client connects to."
    )

    connection_config: FlockMCPConnectionConfiguration = Field(
        ..., description="MCP Connection Configuration for a client."
    )

    caching_config: FlockMCPCachingConfiguration = Field(
        default_factory=FlockMCPCachingConfiguration,
        description="Configuration for the internal caches of the client.",
    )

    callback_config: FlockMCPCallbackConfiguration = Field(
        default_factory=FlockMCPCallbackConfiguration,
        description="Callback configuration for the client.",
    )

    feature_config: FlockMCPFeatureConfiguration = Field(
        default_factory=FlockMCPFeatureConfiguration,
        description="Feature configuration for the client.",
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
    )

    @classmethod
    def with_fields(cls: type[C], **field_definitions) -> type[C]:
        """Create a new config class with additional fields."""
        return create_model(
            f"Dynamic{cls.__name__}", __base__=cls, **field_definitions
        )
