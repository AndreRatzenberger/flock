# src/flock/components/routing/default_routing_component.py
"""Default routing component implementation for the unified component architecture."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import Field

from flock.core.component.agent_component_base import AgentComponentConfig
from flock.core.component.routing_component_base import RoutingModuleBase
from flock.core.context.context import FlockContext
from flock.core.flock_registry import flock_component
from flock.core.flock_router import HandOffRequest
from flock.core.logging.logging import get_logger

if TYPE_CHECKING:
    from flock.core.flock_agent import FlockAgent

logger = get_logger("components.routing.default")


class DefaultRoutingConfig(AgentComponentConfig):
    """Configuration for the default routing component."""

    hand_off: str | HandOffRequest | Callable[..., HandOffRequest] = Field(
        default="", description="Next agent to hand off to"
    )


@flock_component(config_class=DefaultRoutingConfig)
class DefaultRoutingComponent(RoutingModuleBase):
    """Default routing component implementation.

    This router simply uses the configured hand_off property to determine the next agent.
    It does not perform any dynamic routing based on agent results.

    Configuration can be:
    - A string: Simple agent name to route to
    - A HandOffRequest: Full routing configuration
    - A callable: Function that takes (context, result) and returns HandOffRequest
    """

    config: DefaultRoutingConfig = Field(
        default_factory=DefaultRoutingConfig,
        description="Default routing configuration",
    )

    def __init__(
        self,
        name: str = "default_router",
        config: DefaultRoutingConfig | None = None,
        **data,
    ):
        """Initialize the DefaultRoutingComponent.

        Args:
            name: The name of the routing component
            config: The routing configuration
        """
        if config is None:
            config = DefaultRoutingConfig()
        super().__init__(name=name, config=config, **data)

    async def determine_next_step(
        self,
        agent: "FlockAgent",
        result: dict[str, Any],
        context: FlockContext | None = None,
    ) -> HandOffRequest | None:
        """Determine the next agent to hand off to based on configuration.

        Args:
            agent: The agent that just completed execution
            result: The output from the current agent
            context: The global execution context

        Returns:
            A HandOffRequest containing the next agent and input data, or None to end workflow
        """
        handoff = self.config.hand_off

        # If empty string, end the workflow
        if handoff == "":
            logger.debug("No handoff configured, ending workflow")
            return None

        # If callable, invoke it with context and result
        if callable(handoff):
            logger.debug("Invoking handoff callable")
            try:
                handoff = handoff(context, result)
            except Exception as e:
                logger.error("Error invoking handoff callable: %s", e)
                return None

        # If string, convert to HandOffRequest
        if isinstance(handoff, str):
            logger.debug(
                "Converting string handoff to HandOffRequest: %s", handoff
            )
            handoff = HandOffRequest(
                next_agent=handoff, output_to_input_merge_strategy="match"
            )

        # Validate it's a HandOffRequest
        if not isinstance(handoff, HandOffRequest):
            logger.error(
                "Invalid handoff type: %s. Expected HandOffRequest, str, or callable",
                type(handoff),
            )
            return None

        logger.debug("Routing to agent: %s", handoff.next_agent)
        return handoff
