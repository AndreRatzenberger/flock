from datetime import datetime

from pydantic import BaseModel, Field


class SharedLinkConfig(BaseModel):
    """Configuration for a shared Flock agent execution link."""

    share_id: str = Field(..., description="Unique identifier for the shared link.")
    agent_name: str = Field(..., description="The name of the agent being shared.")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Timestamp of when the link was created."
    )
    # Placeholder for future enhancement: pre-filled input values
    # input_values: Optional[Dict[str, Any]] = Field(
    #     None, description="Optional pre-filled input values for the agent."
    # )

    class Config:
        # For Pydantic V2, use model_config instead of Config class if appropriate
        # For Pydantic V1 style
        # orm_mode = True # If you were to load this from an ORM
        # For Pydantic V2 style (if you update Pydantic later)
        from_attributes = True
