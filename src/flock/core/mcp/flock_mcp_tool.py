"""Represents a MCP Tool in a format which is compatible with Flock's ecosystem."""


from typing import Any, Callable, TypeVar

from mcp import Tool
from pydantic import BaseModel, Field


T = TypeVar("T", bound="FlockMCPTool")


class FlockMCPTool(BaseModel):
    name: str = Field(
        ...,
        description="Name of the tool"
    )

    @classmethod
    def try_from_mcp_tool(cls: type[T], mcp_tool: Tool) -> type[T] | None:
        """
        Convert a mcp Tool into a FlockMCPTool
        """
        pass

    @classmethod
    def try_to_mcp_tool(cls: type[T], instance: T) -> Tool | None:
        """
        Convert a flock mcp tool into a mcp tool.
        """
        pass

    def convert_to_callable(self) -> Callable[..., Any] | None:
        return FlockMCPTool._dummy_print_function
