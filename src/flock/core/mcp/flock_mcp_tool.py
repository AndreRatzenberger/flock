"""Represents a MCP Tool in a format which is compatible with Flock's ecosystem."""


from typing import Any, Callable

from pydantic import BaseModel, Field


class FlockMCPTool(BaseModel):
    name: str = Field(
        ...,
        description="Name of the tool"
    )

    @classmethod
    def _dummy_print_function() -> str:
        return "Hello"

    def convert_to_callable(self) -> Callable[..., Any] | None:
        return FlockMCPTool._dummy_print_function
