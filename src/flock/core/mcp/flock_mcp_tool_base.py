"""Represents a MCP Tool in a format which is compatible with Flock's ecosystem."""
from typing import Any, Callable, TypeVar

from mcp import Tool
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from dspy.primitives.tool import Tool as DSPyTool


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
    )

    @classmethod
    def try_from_mcp_tool(cls: type[T], mcp_tool: Tool) -> T | None:
        """
        Convert a mcp Tool into a FlockMCPTool
        """
        return T(
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

    def convert_to_callable(self, connection_manager: "FlockConnectionManagerBase") -> Callable[..., Any] | None:
        """
        Returns a dspy.Tool wrapper around this MCP tool, so that
        dspy.ReAct / dspy.Predict can consume its name/desc/args.
        """
        async def _call(**arguments: Any) -> Any:
            # delay import to avoid circularity
            from flock.core.mcp.flock_mcp_connection_manager_base import FlockMCPConnectionManagerBase
            async with connection_manager.get_client() as client:
                try:
                    return await client.call_tool(
                        name=self.name,
                        arguments=arguments,
                    )
                except Exception as e:
                    # TODO: propper logging
                    return str(e)

        # wrap the inner coroutinge in a Dspy Tool
        return DSPyTool(
            func=_call,
            name=self.name,
            desc=self.description or "",
            args=self.input_schema or {}
        )
