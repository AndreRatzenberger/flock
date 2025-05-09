

from typing import Protocol, Any, Dict, TypeVar
from mcp.types import CallToolResult, TextContent

C = TypeVar("C", bound="MCPClientProto")


class MCPClientProto(Protocol):
    async def call_tool(self, tool_name: str,
                        arguments: Dict[str, Any]) -> CallToolResult: ...

    async def set_is_alive(self, alive: bool) -> None: ...
    async def set_has_error(self, has_error: bool) -> None: ...
    async def set_error_message(self, msg: str) -> None: ...


class MCPConnectionMgrProto(Protocol[C]):
    async def get_client(self) -> C: ...
    async def release_client(self, client: C) -> None: ...
