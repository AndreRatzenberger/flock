"""Represents a MCP Tool in a format which is compatible with Flock's ecosystem."""
from typing import Any, Callable, TypeVar

from mcp import Tool
from mcp.types import ToolAnnotations, CallToolResult, TextContent
from pydantic import BaseModel, Field

from dspy.primitives.tool import Tool as DSPyTool

from flock.core.logging.logging import get_logger
from flock.core.mcp.flock_mcp_client_base import FlockMCPClientBase
from flock.core.mcp.flock_mcp_connection_manager_base import FlockMCPConnectionManagerBase

logger = get_logger("mcp_tool")


T = TypeVar("T", bound="FlockMCPToolBase")


class FlockMCPToolBase(BaseModel):
    name: str = Field(
        ...,
        description="Name of the tool"
    )

    description: str | None = Field(
        ...,
        description="A human-readable description of the tool"
    )

    input_schema: dict[str, Any] = Field(
        ...,
        description="A JSON Schema object defining the expected parameters for the tool."
    )

    annotations: ToolAnnotations | None = Field(
        ...,
        description="Optional additional tool information."
    )w

    @classmethod
    def try_from_mcp_tool(cls: type[T], mcp_tool: Tool) -> T | None:
        """
        Convert a mcp Tool into a FlockMCPTool
        """
        return cls(
            name=mcp_tool.name,
            description=mcp_tool.description,
            input_schema=mcp_tool.inputSchema,
            annotations=mcp_tool.annotations,
        )

    @classmethod
    def try_to_mcp_tool(cls: type[T], instance: T) -> Tool | None:
        """
        Convert a flock mcp tool into a mcp tool.
        """
        return Tool(
            name=instance.name,
            description=instance.description,
            inputSchema=instance.input_schema,
            annotations=instance.annotations,
        )

    def convert_to_callable(self, connection_manager: FlockMCPConnectionManagerBase) -> Callable[..., Any] | None:
        """
        Returns a dspy.Tool wrapper around this MCP tool, so that
        dspy.ReAct / dspy.Predict can consume its name/desc/args.
        """
        async def _call(**arguments: Any) -> CallToolResult:
            # delay import to avoid circularity
            from flock.core.mcp.flock_mcp_connection_manager_base import FlockMCPConnectionManagerBase

            logger.debug(
                f"MCP Tool '{self.name}' called with arguments: {arguments}")
            try:
                client: FlockMCPClientBase = await connection_manager.get_client()
                tool_call_result: CallToolResult = client.call_tool(
                    name=self.name,
                    arguments=arguments,
                )

                if tool_call_result.isError:
                    # log the error for tracking purposes.
                    logger.warning(
                        f"LLM Tried to call mcp function '{self.name}' with arguments {arguments} and produced exception.")

                return tool_call_result

            except Exception as e:
                # Log it, but do not return the Exception to the LLM
                # The reason being, that any exception here likely
                # originates from the surrounding code-framework
                # and not from how the llm called the Function
                # Also, passing on exceptions from the inside
                # of the framework might expose sensitive information.
                logger.error(
                    f"Unexpected Exception ocurred when calling mcp_function '{self.name}': {e}")

                # However, we need to return *something* to the llm, so
                # we tell the llm, that something went wrong in order to inform it.
                return CallToolResult(
                    isError=True,
                    content=[
                        TextContent(
                            type='text',
                            text=f"Tool call for tool '{self.name}' failed. An Exception ocurred."
                        )
                    ]
                )

        # wrap the inner coroutinge in a Dspy Tool
        return DSPyTool(
            func=_call,
            name=self.name,
            desc=self.description or "",
            args=self.input_schema or {}
        )
