"""Callback module for handling agent lifecycle hooks."""

from typing import Any

from pydantic import Field

from flock.core import FlockModule, FlockModuleConfig
from flock.core.context.context import FlockContext
from flock.core.flock_registry import flock_component


class OpikModuleConfig(FlockModuleConfig):
    """Configuration for opik module."""
    project_name: str | None= Field(
        default=None,
        description="The name of the opik project to log data. Default to the Flock run_id.",
    )



@flock_component(config_class=OpikModuleConfig)
class OpikModule(FlockModule):
    """Module that provides opik functionality for agent lifecycle events."""

    name: str = "opik"
    config: OpikModuleConfig = Field(
        default_factory=OpikModuleConfig,
        description="Opik module configuration",
    )

    async def pre_initialize(
        self,
        agent: Any,
        inputs: dict[str, Any],
        context: FlockContext | None = None,
    ) -> None:
        """Run initialize callback if configured."""
        if self.config.initialize_callback:
            await self.config.initialize_callback(agent, inputs)

    async def on_pre_evaluate(
        self,
        agent: Any,
        inputs: dict[str, Any],
        context: FlockContext | None = None,
    ) -> dict[str, Any]:
        """Run evaluate callback if configured."""
        if self.config.evaluate_callback:
            return await self.config.evaluate_callback(agent, inputs)
        return inputs

    async def pre_terminate(
        self,
        agent: Any,
        inputs: dict[str, Any],
        context: FlockContext | None = None,
        result: dict[str, Any] | None = None,
    ) -> None:
        """Run terminate callback if configured."""
        if self.config.terminate_callback:
            await self.config.terminate_callback(agent, inputs, result)

    async def on_error(
        self,
        agent: Any,
        error: Exception,
        inputs: dict[str, Any],
        context: FlockContext | None = None,
    ) -> None:
        """Run error callback if configured."""
        if self.config.on_error_callback:
            await self.config.on_error_callback(agent, error, inputs)
