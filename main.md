\
```python
# Add these imports to the appropriate sections of src/flock/webapp/app/main.py

# Standard library imports (if not already present)
import uuid

# Pydantic and FastAPI imports (ensure these are grouped with similar imports)
from pydantic import BaseModel
from fastapi import Depends, HTTPException 

# Imports from your project for the sharing feature
from flock.webapp.app.services.sharing_models import SharedLinkConfig
from flock.webapp.app.services.sharing_store import SharedLinkStoreInterface
from flock.webapp.app.dependencies import get_shared_link_store
```

```python
# Add this code block in src/flock/webapp/app/main.py
# A good place is after the app.include_router(...) lines 
# and before other utility function definitions or page route definitions.

# --- Share Link API Models and Endpoint ---
class CreateShareLinkRequest(BaseModel):
    agent_name: str

class CreateShareLinkResponse(BaseModel):
    share_url: str

@app.post("/api/v1/share/link", response_model=CreateShareLinkResponse, tags=["UI Sharing"])
async def create_share_link(
    request_data: CreateShareLinkRequest,
    store: SharedLinkStoreInterface = Depends(get_shared_link_store)
):
    \"\"\"Creates a new shareable link for an agent.\"\"\"
    share_id = uuid.uuid4().hex
    agent_name = request_data.agent_name

    if not agent_name: # Basic validation
        raise HTTPException(status_code=400, detail="Agent name cannot be empty.")

    config = SharedLinkConfig(share_id=share_id, agent_name=agent_name)
    try:
        await store.save_config(config)
        share_url = f"/ui/shared-run/{share_id}\" # Relative URL
        # Ensure 'logger' is available in this scope, e.g., defined globally in main.py
        logger.info(f"Created share link for agent \'{agent_name}\' with ID \'{share_id}\'. URL: {share_url}")
        return CreateShareLinkResponse(share_url=share_url)
    except Exception as e:
        logger.error(f"Failed to create share link for agent \'{agent_name}\': {e}", exc_info=True)
        # The f-string formatting for the exception detail was corrected here.
        raise HTTPException(status_code=500, detail=f"Failed to create share link: {str(e)}")

# --- End Share Link API --- 