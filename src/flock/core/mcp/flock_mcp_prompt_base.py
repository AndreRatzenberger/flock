"""Represents a prompt that can be retrieved from a MCP-Server."""

from pydantic import BaseModel


class FlockMCPPromptBase(BaseModel):
    """Represents a prompt retrieved from a remote MCP-Server.

    Documentation:
        https://modelcontextprotocol.io/docs/concepts/prompts

    Summary:
        Provides reusable prompt templates and workflows

        Prompts enable servers to define reusable prompt templates
        and workflows that clients can easily pass on to Users and LLMs.

        Prompts are predefined templates that can:
        - Accept dynamic arguments
        - Include context from resources
        - Chain multiple interactions
        - Guid specific workflows
        - Surface as UI elements (like slash commands)
    """

    pass
