# src/flock/core/agent/flock_agent_lifecycle.py
"""Lifecycle management functionality for FlockAgent."""

from typing import TYPE_CHECKING, Any

from opentelemetry import trace
from flock.core.logging.logging import get_logger

if TYPE_CHECKING:
    from flock.core.flock_agent import FlockAgent

logger = get_logger("agent.lifecycle")
tracer = trace.get_tracer(__name__)


class FlockAgentLifecycle:
    """Handles lifecycle management for FlockAgent including initialization, evaluation, and termination."""

    def __init__(self, agent: "FlockAgent"):
        self.agent = agent

    async def initialize(self, inputs: dict[str, Any]) -> None:
        """Initialize agent and run module initializers."""
        logger.debug(f"Initializing agent '{self.agent.name}'")
        with tracer.start_as_current_span("agent.initialize") as span:
            span.set_attribute("agent.name", self.agent.name)
            span.set_attribute("inputs", str(inputs))
            logger.info(
                f"agent.initialize",
                agent=self.agent.name,
            )
            try:
                for module in self.agent._components.get_enabled_modules():
                    await module.on_initialize(self.agent, inputs, self.agent.context)
            except Exception as module_error:
                logger.error(
                    "Error during initialize",
                    agent=self.agent.name,
                    error=str(module_error),
                )
                span.record_exception(module_error)

    async def terminate(
        self, inputs: dict[str, Any], result: dict[str, Any]
    ) -> None:
        """Terminate agent and run module terminators."""
        logger.debug(f"Terminating agent '{self.agent.name}'")
        with tracer.start_as_current_span("agent.terminate") as span:
            span.set_attribute("agent.name", self.agent.name)
            span.set_attribute("inputs", str(inputs))
            span.set_attribute("result", str(result))
            logger.info(
                f"agent.terminate",
                agent=self.agent.name,
            )
            try:
                current_result = result
                for module in self.agent._components.get_enabled_modules():
                    tmp_result = await module.on_terminate(
                        self.agent, inputs, self.agent.context, current_result
                    )
                    # If the module returns a result, use it
                    if tmp_result:
                        current_result = tmp_result

                if self.agent.write_to_file:
                    self.agent._save_output(self.agent.name, current_result)

                if self.agent.wait_for_input:
                    # simple input prompt
                    input("Press Enter to continue...")

            except Exception as module_error:
                logger.error(
                    "Error during terminate",
                    agent=self.agent.name,
                    error=str(module_error),
                )
                span.record_exception(module_error)

    async def on_error(self, error: Exception, inputs: dict[str, Any]) -> None:
        """Handle errors and run module error handlers."""
        logger.error(f"Error occurred in agent '{self.agent.name}': {error}")
        with tracer.start_as_current_span("agent.on_error") as span:
            span.set_attribute("agent.name", self.agent.name)
            span.set_attribute("inputs", str(inputs))
            try:
                for module in self.agent._components.get_enabled_modules():
                    await module.on_error(self.agent, inputs, self.agent.context, error)
            except Exception as module_error:
                logger.error(
                    "Error during on_error",
                    agent=self.agent.name,
                    error=str(module_error),
                )
                span.record_exception(module_error)

    async def evaluate(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Core evaluation logic, calling the assigned evaluator and modules."""
        if not self.agent.evaluator:
            raise RuntimeError(
                f"Agent '{self.agent.name}' has no evaluator assigned."
            )
        with tracer.start_as_current_span("agent.evaluate") as span:
            span.set_attribute("agent.name", self.agent.name)
            span.set_attribute("inputs", str(inputs))
            logger.info(
                f"agent.evaluate",
                agent=self.agent.name,
            )

            logger.debug(f"Evaluating agent '{self.agent.name}'")
            current_inputs = inputs

            # Pre-evaluate hooks
            for module in self.agent._components.get_enabled_modules():
                current_inputs = await module.on_pre_evaluate(
                    self.agent, current_inputs, self.agent.context
                )

            # Actual evaluation
            try:
                # Get tools and MCP tools through integration handler
                registered_tools = []
                if self.agent.tools:
                    registered_tools = self.agent.tools

                # Retrieve available mcp_tools if the evaluator needs them
                mcp_tools = []
                if self.agent.servers:
                    mcp_tools = await self.agent._integration.get_mcp_tools()

                # --------------------------------------------------
                # Optional DI middleware pipeline
                # --------------------------------------------------
                result = await self.agent._integration.execute_with_middleware(
                    current_inputs, registered_tools, mcp_tools
                )

            except Exception as eval_error:
                logger.error(
                    "Error during evaluate",
                    agent=self.agent.name,
                    error=str(eval_error),
                )
                span.record_exception(eval_error)
                await self.on_error(
                    eval_error, current_inputs
                )  # Call error hook
                raise  # Re-raise the exception

            # Post-evaluate hooks
            current_result = result
            for module in self.agent._components.get_enabled_modules():
                tmp_result = await module.on_post_evaluate(
                    self.agent,
                    current_inputs,
                    self.agent.context,
                    current_result,
                )
                # If the module returns a result, use it
                if tmp_result:
                    current_result = tmp_result

            logger.debug(f"Evaluation completed for agent '{self.agent.name}'")
            return current_result
