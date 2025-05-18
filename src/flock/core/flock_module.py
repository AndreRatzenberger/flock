"""Base classes and implementations for the Flock module system."""

from abc import ABC
from typing import Any, TypeVar

from pydantic import BaseModel, Field, create_model

from flock.core.context.context import FlockContext

T = TypeVar("T", bound="FlockModuleConfig")


class FlockModuleConfig(BaseModel):
    """Base configuration class for Flock modules.

    This class serves as the base for all module-specific configurations.
    Each module should define its own config class inheriting from this one.

    Example:
        class MemoryModuleConfig(FlockModuleConfig):
            file_path: str = Field(default="memory.json")
            save_after_update: bool = Field(default=True)
    """

    enabled: bool = Field(
        default=True, description="Whether the module is currently enabled"
    )

    @classmethod
    def with_fields(cls: type[T], **field_definitions) -> type[T]:
        """Create a new config class with additional fields."""
        return create_model(
            f"Dynamic{cls.__name__}", __base__=cls, **field_definitions
        )


class FlockModule(BaseModel, ABC):
    """Base class for all Flock modules.

    Modules can hook into agent lifecycle events and modify or enhance agent behavior.
    They are initialized when added to an agent and can maintain their own state.

    Each module should define its configuration requirements either by:
    1. Creating a subclass of FlockModuleConfig
    2. Using FlockModuleConfig.with_fields() to create a config class
    """

    name: str = Field(
        default="", description="Unique identifier for the module"
    )
    config: FlockModuleConfig = Field(
        default_factory=FlockModuleConfig, description="Module configuration"
    )

    async def on_initialize(
        self,
        agent: Any,
        inputs: dict[str, Any],
        context: FlockContext | None = None,
    ) -> None:
        """Called when the agent starts running."""
        pass

    async def on_pre_evaluate(
        self,
        agent: Any,
        inputs: dict[str, Any],
        context: FlockContext | None = None,
    ) -> dict[str, Any]:
        """Called before agent evaluation, can modify inputs."""
        return inputs

    async def on_post_evaluate(
        self,
        agent: Any,
        inputs: dict[str, Any],
        context: FlockContext | None = None,
        result: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Called after agent evaluation, can modify results."""
        return result

    async def on_terminate(
        self,
        agent: Any,
        inputs: dict[str, Any],
        context: FlockContext | None = None,
        result: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Called when the agent finishes running."""
        return result

    async def on_error(
        self,
        agent: Any,
        inputs: dict[str, Any],
        context: FlockContext | None = None,
        error: Exception | None = None,
    ) -> None:
        """Called when an error occurs during agent execution."""
        pass

    async def pre_server_init(self, server: Any) -> None:
        """Called before a server initializes."""
        pass

    async def post_server_init(self, server: Any) -> None:
        """Called after a server initialized."""
        pass

    async def pre_server_terminate(self, server: Any) -> None:
        """Called before a server terminates."""
        pass

    async def post_server_teminate(self, server: Any) -> None:
        """Called after a server terminates."""
        pass

    async def on_server_error(self, server: Any, error: Exception) -> None:
        """Called when a server errors."""
        pass

    async def before_connect(
        self,
        server: Any,
        additional_params: dict[str, Any],
    ) -> None:
        """Called before a connection is being established to a mcp server.

        Mutate additional_params in order to pass additional parameters to
        mcp clients. (Such as additional headers, changing timeouts etc.)
        """
        pass

    async def pre_mcp_call(
        self,
        server: Any,
    ) -> None:
        """Called before any MCP Calls."""
        pass

    async def post_mcp_call(
        self,
        server: Any,
        result: Any,
    ) -> None:
        """Called after any MCP Calls."""
        pass
