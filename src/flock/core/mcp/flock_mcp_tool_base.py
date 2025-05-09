"""Represents a MCP Tool in a format which is compatible with Flock's ecosystem."""
from abc import ABC, abstractmethod
from typing import Any, Callable, TypeVar

from mcp import Tool
from mcp.types import ToolAnnotations, CallToolResult, TextContent
from pydantic import BaseModel, Field

from dspy.primitives.tool import Tool as DSPyTool

from flock.core.logging.logging import get_logger
from flock.core.mcp.types.mcp_protocols import MCPClientProto, MCPConnectionMgrProto

logger = get_logger("mcp_tool")


T = TypeVar("T", bound="FlockMCPToolBase")


class FlockMCPToolBase(BaseModel, ABC):
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
    def from_mcp_tool(cls: type[T], tool: Tool) -> T:
        return cls(
            name=tool.name,
            description=tool.description,
            input_schema=tool.inputSchema,
            annotations=tool.annotations,
        )

    def to_mcp_tool(self) -> Tool:
        return Tool(
            name=self.name,
            description=self.description,
            inputSchema=self.input_schema,
            annotations=self.annotations
        )

    @classmethod
    def to_mcp_tool(cls: type[T], instance: T) -> Tool | None:
        """
        Convert a flock mcp tool into a mcp tool.
        """
        return Tool(
            name=instance.name,
            description=instance.description,
            inputSchema=instance.input_schema,
            annotations=instance.annotations,
        )

    @abstractmethod
    def get_connection_manager(self) -> MCPConnectionMgrProto:
        """
        Must return an instance of your connection manager
        """
        ...

    def as_dspy_tool(self) -> DSPyTool:
        """
        Wrap this tool as a DSPyTool for downstream.
        """
        async def _invoke(**kwargs: Any) -> CallToolResult:
            client: MCPClientProto | None = None
            mgr = self.get_connection_manager()
            try:
                client = await mgr.get_client()
                res: CallToolResult = await client.call_tool(self.name, kwargs)
                if res.isError:
                    # optional hook
                    self.on_error(res, kwargs)
                    return res
            except Exception as e:
                logger.error(f"Error in '{self.name}': {e}")
                if client:
                    await client.set_is_alive(False)
                    await client.set_has_error(True)
                    await client.set_error_message(str(e))
                return CallToolResult(
                    isError=True,
                    content=[
                        TextContent(
                            type="text",
                            text=f"Tool '{self.name}' failed."
                        )
                    ]
                )
            finally:
                await mgr.release_client(client)

        schema = self.input_schema or {}
        arg_schemas = schema.get("properties", {})

        return DSPyTool(
            func=_invoke,
            name=self.name,
            desc=self.description or "",
            args=arg_schemas,
        )
