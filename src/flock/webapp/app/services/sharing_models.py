from datetime import datetime

from pydantic import BaseModel, Field


class SharedLinkConfig(BaseModel):
    """Configuration for a shared Flock agent execution link."""

    share_id: str = Field(..., description="Unique identifier for the shared link.")
    agent_name: str = Field(..., description="The name of the agent being shared.")
    flock_definition: str = Field(..., description="The YAML/JSON string definition of the Flock the agent belongs to.")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Timestamp of when the link was created."
    )
    # Placeholder for future enhancement: pre-filled input values
    # input_values: Optional[Dict[str, Any]] = Field(
    #     None, description="Optional pre-filled input values for the agent."
    # )

    model_config = {
        "from_attributes": True,
        "json_schema_extra": {
            "examples": [
                {
                    "share_id": "abcdef123456",
                    "agent_name": "MyChatAgent",
                    "flock_definition": "name: MySharedFlock\nagents:\n  MyChatAgent:\n    input: 'message: str'\n    output: 'response: str'\n    # ... rest of flock YAML ...",
                    "created_at": "2023-10-26T10:00:00Z",
                }
            ]
        }
    }
