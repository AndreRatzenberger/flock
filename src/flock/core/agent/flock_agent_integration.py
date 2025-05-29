# src/flock/core/agent/flock_agent_integration.py
"""Tool and server integration functionality for FlockAgent."""

from typing import TYPE_CHECKING, Any

from flock.core.context.context import FlockContext
from flock.core.logging.logging import get_logger
from flock.core.mcp.flock_mcp_server import FlockMCPServerBase

if TYPE_CHECKING:
    from flock.core.flock_agent import FlockAgent

logger = get_logger("agent.integration")


class FlockAgentIntegration:
    """Handles tool and server integration for FlockAgent including MCP servers and callable tools."""

    def __init__(self, agent: "FlockAgent"):
        self.agent = agent

    def resolve_callables(self, context: FlockContext | None = None) -> None:
        """Resolves callable fields (description, input, output) using context."""
        if callable(self.agent.description):
            self.agent.description = self.agent.description(
                context
            )  # Pass context if needed by callable
        if callable(self.agent.input):
            self.agent.input = self.agent.input(context)
        if callable(self.agent.output):
            self.agent.output = self.agent.output(context)

    def resolve_description(self, context: FlockContext | None = None) -> str:
        if callable(self.agent.description):
            try:
                # Attempt to call without context first.
                return self.agent.description(context)
            except Exception as e:
                logger.error(
                    f"Error resolving callable description for agent '{self.agent.name}': {e}"
                )
                return None
        elif isinstance(self.agent.description, str):
            return self.agent.description
        return None

    async def get_mcp_tools(self) -> list[Any]:
        """Get tools from registered MCP servers."""
        mcp_tools = []
        if self.agent.servers:
            from flock.core.flock_registry import get_registry

            FlockRegistry = get_registry()  # Get the registry
            for server in self.agent.servers:
                registered_server: FlockMCPServerBase | None = None
                server_tools = []
                if isinstance(server, FlockMCPServerBase):
                    # check if registered
                    server_name = server.config.name
                    registered_server = FlockRegistry.get_server(
                        server_name
                    )
                else:
                    # servers must be registered.
                    registered_server = FlockRegistry.get_server(
                        name=server
                    )
                if registered_server:
                    server_tools = await registered_server.get_tools(
                        agent_id=self.agent.agent_id,
                        run_id=self.agent.context.run_id,
                    )
                else:
                    logger.warning(
                        f"No Server with name '{server.config.name if isinstance(server, FlockMCPServerBase) else server}' registered! Skipping."
                    )
                mcp_tools = mcp_tools + server_tools
        return mcp_tools

    async def execute_with_middleware(
        self,
        current_inputs: dict[str, Any],
        registered_tools: list[Any],
        mcp_tools: list[Any]
    ) -> dict[str, Any]:
        """Execute evaluator with optional DI middleware pipeline."""
        container = None
        if self.agent.context is not None:
            container = self.agent.context.get_variable("di.container")

        # If a MiddlewarePipeline is registered in DI, wrap the evaluator
        result: dict[str, Any] | None = None

        if container is not None:
            try:
                from wd.di.middleware import (
                    MiddlewarePipeline,
                )

                pipeline: MiddlewarePipeline | None = None
                try:
                    pipeline = container.get_service(MiddlewarePipeline)
                except Exception:
                    pipeline = None

                if pipeline is not None:
                    # Build execution chain where the evaluator is the terminal handler

                    async def _final_handler():
                        return await self.agent.evaluator.evaluate_core(
                            self.agent, current_inputs, self.agent.context, registered_tools, mcp_tools
                        )

                    idx = 0

                    async def _invoke_next():
                        nonlocal idx

                        if idx < len(pipeline._middleware):
                            mw = pipeline._middleware[idx]
                            idx += 1
                            return await mw(self.agent.context, _invoke_next)  # type: ignore[arg-type]
                        return await _final_handler()

                    # Execute pipeline
                    result = await _invoke_next()
                else:
                    # No pipeline registered, direct evaluation
                    result = await self.agent.evaluator.evaluate_core(
                        self.agent, current_inputs, self.agent.context, registered_tools, mcp_tools
                    )
            except ImportError:
                # wd.di not installed – fall back
                result = await self.agent.evaluator.evaluate_core(
                    self.agent, current_inputs, self.agent.context, registered_tools, mcp_tools
                )
        else:
            # No DI container – standard execution
            result = await self.agent.evaluator.evaluate_core(
                self.agent,
                current_inputs,
                self.agent.context,
                registered_tools,
                mcp_tools,
            )

        return result
