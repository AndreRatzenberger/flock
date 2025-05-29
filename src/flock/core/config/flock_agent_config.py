"""Configuration settings for FlockAgent."""

from pydantic import BaseModel, Field


class FlockAgentConfig(BaseModel):
    """FlockAgentConfig is a class that holds the configuration for a Flock agent.
    
    It is used to store various settings and parameters that can be accessed throughout the agent's lifecycle.
    """

    write_to_file: bool = Field(
        default=False,
        description="Write the agent's output to a file.",
    )
    wait_for_input: bool = Field(
        default=False,
        description="Wait for user input after the agent's output is displayed.",
    )
